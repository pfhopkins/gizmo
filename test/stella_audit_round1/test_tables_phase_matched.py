"""STL-391/392/411: phase-matched stellar-table lookup (GALSF_RESOLVEDISM_TABLES_PHASE_MATCHED_LOOKUP): each grid star of the
(M, Z) cell is read at the same fraction of its own lifetime; photon tables stay log-averaged (Uli's choice); v_wind nodes
of 0 (a track after its death) are dead. Star: M = sqrt(12 x 14) = 12.96 Msun (half-way in log M between two grid stars), Z = 0.014.

Reference of each quantity, fixed before the run:
- wind by death: the bilinear average of the two grid stars' lifetime wind totals (each at its own death), 0.418 Msun.
  At death every grid star sits at its own end, so the delivered wind equals it by construction.
  Table driver: |w / ref - 1| <= 1e-6. Run (natural death, wind = M_birth - M_pre): <= 1e-3.
  Expected without the flag: 0.655 (checker 06, section 3).
- ionising photons: the lifetime integral of the phase-matched, log-averaged rate computed here from the HDF5 tables
  with the driver's samples. Bar |N / ref - 1| <= 1e-6. Expected without the flag: about 0.955 (checker: 0.828 equal-age,
  0.867 phase-matched log, both against the corner average of the two lifetime totals; that corner average cannot be
  matched by an interpolation that keeps t_life between the corners, so it is reported, not used as the bar).
- v_wind at death: the phase-matched value with dead nodes dropped, |v / ref - 1| <= 1e-6. Expected without the flag:
  pulled toward 0 by the dead node after the lighter grid star's death."""

import os
import subprocess
from os.path import join

import numpy as np
import pytest

from audit_round1_common import HOOKS, ROUND1_FLAGS, TEST_DIR, Tables, UNIT_T_YR, binary, driver, kv, make_box, run, write_params

M_STAR, Z_STAR = float(np.sqrt(12.0 * 14.0)), 0.014
ARMS = {"off": HOOKS, "on": HOOKS + ROUND1_FLAGS}
RUN_BAR, DRIVER_BAR = 1e-3, 1e-6


def references():
    T = Tables()
    iz, fz, im, fm = T.cell(M_STAR, Z_STAR)
    w = 0.0
    for dz, wz in ((0, 1 - fz), (1, fz)):
        for dm, wm in ((0, 1 - fm), (1, fm)):
            if wz * wm > 0:
                w += wz * wm * T.wind_total_at_own_death(iz + dz, im + dm).sum()
    lt = T.lifetime(M_STAR, Z_STAR)[1]
    N = 20000
    ages = (np.arange(N) + 0.5) / N * lt
    q = sum(10 ** T.phase_matched(T.logQ, M_STAR, Z_STAR, a, -90.0, -99.0) for a in ages) * lt / N
    corner = 0.0  # the corner average of the two lifetime totals (reported only)
    for dm, wm in ((0, 1 - fm), (1, fm)):
        ltc = T.lt[iz, im + dm]; ac = (np.arange(N) + 0.5) / N * ltc
        tot = sum(10 ** (lambda r: r[0] / r[1] if r[1] > 0 else -99.0)(T.interp_age(T.logQ, iz, im + dm, np.log10(max(a, 100.0)), -90.0)) for a in ac) * ltc / N
        corner += wm * tot
    v = T.phase_matched(T.vw, M_STAR, Z_STAR, lt, 0.0, 0.0)
    return w, q, corner, v


def driver_values(arm):
    binary("round1_" + arm, add=ARMS[arm], drivers=("phase_driver",))
    d = join(TEST_DIR, "run_phase_driver_" + arm); os.makedirs(d, exist_ok=True)
    if not os.path.islink(join(d, "gizmo")):
        os.symlink(os.path.abspath(join(TEST_DIR, "..", "..")), join(d, "gizmo"))
    make_box(join(d, "box_ics.hdf5"))
    write_params(d, {"InitCondFile": "box_ics"})
    r = subprocess.run([driver("round1_" + arm, "phase_driver"), "run.params", repr(M_STAR), repr(Z_STAR)], capture_output=True, text=True, cwd=d, timeout=3600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    q = float(kv(r.stdout, "PHASEQ")[0]["Qint"]); v = float(kv(r.stdout, "PHASEVW")[0]["v"]); w = float(kv(r.stdout, "PHASEWC")[0]["w"])
    return w, q, v


def check_driver(arm):
    w_ref, q_ref, corner, v_ref = references()
    w, q, v = driver_values(arm)
    print(f"phase driver ({arm}): wind at death {w:.10e} / reference {w_ref:.10e} = {w / w_ref:.6f}; photons {q:.10e} / phase-matched "
          f"reference {q_ref:.10e} = {q / q_ref:.6f} (against the corner average {corner:.10e}: {q / corner:.6f}, reference {q_ref / corner:.6f}); "
          f"v_wind at death {v:.6e} / reference {v_ref:.6e} = {v / v_ref:.6f}")
    assert abs(w / w_ref - 1) <= DRIVER_BAR
    assert abs(q / q_ref - 1) <= DRIVER_BAR
    assert abs(v / v_ref - 1) <= DRIVER_BAR


def check_run(arm):
    w_ref = references()[0]
    exe = binary("round1_" + arm, add=ARMS[arm])
    T = Tables()
    t_death = T.lifetime(M_STAR, Z_STAR)[0]
    name = "phase_run_" + arm
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"))
    t_end = 6.0e4
    text = run(name, exe, {"InitCondFile": "box_ics", "TestStarMass": repr(M_STAR), "TestNumStars": 1, "TestStarInitAgeYr": t_death - 2.0e4,
                           "TestSNTime0": 0, "TimeMax": t_end / UNIT_T_YR, "TimeBetSnapshot": t_end / UNIT_T_YR})[1]
    death = kv(text, "STARDEATH")
    assert len(death) == 1, f"expected one natural death, got {len(death)}"
    w = float(death[0]["M_birth"]) - float(death[0]["M_pre"])
    print(f"phase run ({arm}): wind by death {w:.10e} Msun, reference {w_ref:.10e}: ratio {w / w_ref:.6f}")
    assert abs(w / w_ref - 1) <= RUN_BAR


def test_phase_matched_driver_flag_on():
    check_driver("on")


@pytest.mark.xfail(strict=True, reason="STL-391/392/411 without the flag: equal-age lookup (expected wind 0.655, photons ~0.955)")
def test_phase_matched_driver_flag_off():
    check_driver("off")


def test_phase_matched_run_flag_on():
    check_run("on")


@pytest.mark.xfail(strict=True, reason="STL-391 without the flag: the run delivers 0.655 of the wind by death")
def test_phase_matched_run_flag_off():
    check_run("off")
