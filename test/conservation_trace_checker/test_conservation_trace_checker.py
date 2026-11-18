"""conservation_trace_checker: gizmo.conservation_trace (stella_dev python_src/gizmo/conservation_trace.py with the fix/trace-imf-2026-11-17
changes) on synthetic trace files; no GIZMO build. Mass rows in the port's mass_balance_steps.txt layout
(OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS: call tag time M_bary dM_bary M_Z_gas dM_Z_gas M_Z_stars M_Z_netyield M_Z_ledger dM_Z_ledger
M_Z_inflight), so the metal ledger is read from columns 10 and 11, not from the last two."""

import pytest

from gizmo.conservation_trace import check_conservation_traces, assert_conservation_traces

MASS_HDR = "# call tag time M_bary dM_bary M_Z_gas dM_Z_gas M_Z_stars M_Z_netyield_cumulative M_Z_ledger dM_Z_ledger M_Z_inflight\n"
INFLIGHT = 3.0e-9  # non-zero in-flight metals: a reader of the last two columns would see a jump of 3e-9 / dL


def mass_line(n, tag, M, dM, L, dL):
    return f"{n} {tag} 1.0e-03 {M:.16e} {dM:+.16e} 1.0e-08 +0.0e+00 0.0 +0.0 {L:.16e} {dL:+.16e} {INFLIGHT:+.16e}\n"


ENERGY_HDR = "# Energy step trace\n# pot_valid=1 every=1\n"


def sync_line(step, t, int_res, tot_res, scale=1.0, sn=0.0, ia=0.0):
    return (f"SYNC step={step} t={t:.6e} E_kin=0.5 E_th=0.5 E_mag=0 E_cr=0 E_rad=0 E_pot=-1 E_tot=0 dE_tot=0 "
            f"int_res={int_res:+.6e} tot_res={tot_res:+.6e} E_scale={scale:.6e} s_fb_sn_ia=+0 b_fb_sn={sn:+.6e} b_fb_ia={ia:+.6e}\n")


def stage_line(step, stage, res, bar=1e-13, eabs=1.0):
    return f"STAGE step={step} t=1e-3 stage={stage} res={res:+.6e} bar={bar:.6e} booked=+0 d_kin=+0 d_th={res:+.6e} E_abs={eabs:.6e}\n"


def write(tmp_path, mass_lines, energy_lines, energy_hdr=ENERGY_HDR):
    (tmp_path / "mass_balance_steps.txt").write_text(MASS_HDR + "".join(mass_lines))
    if energy_lines is not None:
        (tmp_path / "energy_step_trace.txt").write_text(energy_hdr + "".join(energy_lines))
    return tmp_path


CLEAN_MASS = [mass_line(0, "pre_fb_mom_p0", 1.0, 0.0, 1e-8, 0.0), mass_line(1, "post_diff_unpack", 1.0, 1e-16, 1e-8, 1e-24)]
CLEAN_ENERGY = [sync_line(1, 0.0, 0.0, 0.0), sync_line(10, 1e-3, 1e-4, 1e-4)]


def test_clean_traces_pass(tmp_path):
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, CLEAN_ENERGY))
    assert rep.ok, rep.summary()
    assert_conservation_traces(tmp_path)


def test_stage_violation_is_a_finding(tmp_path):
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, CLEAN_ENERGY + [stage_line(5, "fb_sn_ia", 2e-2), stage_line(9, "fb_sn_ia", 1e-2)]))
    assert not rep.ok
    f = rep.findings[0]
    assert f["kind"] == "energy_stage" and f["stage"] == "fb_sn_ia" and f["count"] == 2
    assert f["max_rel"] == pytest.approx(2e-2) and f["sum_res"] == pytest.approx(3e-2)
    with pytest.raises(AssertionError, match="fb_sn_ia"):
        assert_conservation_traces(tmp_path)


def test_integration_drift_beyond_bar_fails(tmp_path):
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, [sync_line(1, 0.0, 0.0, 0.0), sync_line(9, 1e-3, 0.05, 0.05)]))
    assert not rep.ok and rep.findings[0]["kind"] == "energy_integration"
    assert rep.findings[0]["max_rel"] == pytest.approx(0.05)


def test_integration_not_checked_without_potential(tmp_path):
    hdr = "# Energy step trace\n# pot_valid=0 every=1\n"
    lines = [l.replace("int_res=+0.000000e+00", "int_res=nan") for l in CLEAN_ENERGY]
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, lines, hdr))
    assert rep.ok and "integration not checked" in rep.summary()


def test_mass_jump_fails(tmp_path):
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS + [mass_line(2, "post_imf_accrete", 1.0, 1e-9, 1e-8, 0.0)], CLEAN_ENERGY))
    assert not rep.ok
    f = rep.findings[0]
    assert f["kind"] == "mass" and f["stage"] == "post_imf_accrete" and f["max_rel"] == pytest.approx(1e-9)


def test_metal_ledger_jump_fails(tmp_path):
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS + [mass_line(2, "post_fb_thermal", 1.0, 0.0, 1e-8, 1e-15)], CLEAN_ENERGY))
    assert not rep.ok and rep.findings[0]["kind"] == "metal_ledger"


def test_missing_energy_trace_fails_when_required(tmp_path):
    write(tmp_path, CLEAN_MASS, None)
    assert not check_conservation_traces(tmp_path).ok
    assert check_conservation_traces(tmp_path, require_energy=False).ok


def test_restart_resets_integration_baseline(tmp_path):
    lines = [sync_line(1, 0.0, 0.0, 0.0), sync_line(5, 1e-3, 1e-3, 1e-3), "RESTART t=1e-3\n", sync_line(6, 1e-3, 0.0, 0.0), sync_line(9, 2e-3, 2e-3, 2e-3)]
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, lines))
    assert rep.ok, rep.summary()


def test_integration_scale_fixed_at_first_sync(tmp_path):
    """A scale that grows with the bookings must not loosen the bar: the first sync's E_scale is used throughout."""
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, [sync_line(1, 0.0, 0.0, 0.0, scale=1.0), sync_line(9, 1e-3, 0.05, 0.05, scale=100.0)]))
    assert not rep.ok and rep.findings[0]["kind"] == "energy_integration"
    assert rep.findings[0]["max_rel"] == pytest.approx(0.05)


def test_scale_grows_with_booked_sn_energy_only_from_the_sn_on(tmp_path):
    """Before the SN the bar is on the first sync's scale; from the SN on, on the SN + Ia energy booked so far."""
    pre = [sync_line(1, 0.0, 0.0, 0.0, scale=1e-4), sync_line(5, 1e-3, 2e-5, 2e-5, scale=1e-4)]  # 2e-5 / 1e-4 before the SN
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, pre + [sync_line(9, 2e-3, 2e-5, 2e-5, scale=1e-4, sn=5e-3)]))
    assert not rep.ok and rep.findings[0]["count"] == 1 and rep.findings[0]["max_rel"] == pytest.approx(0.2)
    ok = [sync_line(1, 0.0, 0.0, 0.0, scale=1e-4), sync_line(9, 2e-3, 2e-5, 2e-5, scale=1e-4, sn=4e-3, ia=1e-3)]  # 2e-5 / 5e-3
    assert check_conservation_traces(write(tmp_path, CLEAN_MASS, ok)).ok


def test_integration_finding_reports_latest_residual(tmp_path):
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, [sync_line(1, 0.0, 0.0, 0.0), sync_line(5, 1e-3, 0.03, 0.03), sync_line(9, 2e-3, 0.05, 0.05)]))
    f = rep.findings[0]
    assert f["count"] == 2 and f["sum_res"] == pytest.approx(0.05)


def test_cooling_solve_checked_against_its_reported_rate(tmp_path):
    """Non-stiff cells: the chemistry solve's energy change must match its own reported net cooling rate x dt (5e-2)."""
    def line(solve, rate):
        return sync_line(9, 1e-3, 0.0, 0.0).replace("\n", f" b_info_cool_solve_ns={solve:+.6e} b_info_cool_rate_ns={rate:+.6e}\n")
    assert check_conservation_traces(write(tmp_path, CLEAN_MASS, [sync_line(1, 0.0, 0.0, 0.0), line(-1e-3, -1.02e-3)])).ok
    rep = check_conservation_traces(write(tmp_path, CLEAN_MASS, [sync_line(1, 0.0, 0.0, 0.0), line(-1e-3, -2e-3)]))
    assert not rep.ok and rep.findings[0]["kind"] == "cooling_rate"


def test_inflight_column_is_not_the_ledger(tmp_path):
    """The last column is the in-flight exchange, not dM_Z_ledger: a ledger change of 1e-13 with metals in flight passes
    (read from the last two columns it would be a jump of 3e-9 on a ledger of 1e-21)."""
    rep = check_conservation_traces(write(tmp_path, [mass_line(0, "a", 1.0, 0.0, 1e-8, 0.0), mass_line(1, "b", 1.0, 0.0, 1e-8, 1e-21)], CLEAN_ENERGY))  # 1e-13 of the ledger
    assert rep.ok, rep.summary()
