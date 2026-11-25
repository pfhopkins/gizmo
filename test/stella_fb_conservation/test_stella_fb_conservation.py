"""Stella feedback conservation on the production build (Config_stella_default.sh plus the isolated-feedback test hooks and
the energy step trace): every channel deposits what it books, mass and every element close, and the death ejecta add up.

Built with HYDRO_EXACT_PAIR_FLUX_EXCHANGE and HYDRO_TIMESTEP_FROM_PAIR_FORCE (Uli 2026-11-21): the cross-timebin pair fluxes are
exchanged exactly, so the energy traces test the feedback itself. Without them the default build leaves a 1-3 % time-integration
residual after hot deposits (STL-55, STL-127); test_default_build_integration_residual records that as a strict expected
failure. Every run also checks the mass, element and energy step traces (gizmo.conservation_trace, bars fixed there). Units: kpc, 1e10 Msun, km/s. Boxes: n_H = 10, 1e4 K, 1 Msun cells, 16 pc, periodic.

test_death_ejecta_table (no run): the death ejecta the deposit uses (stellar_death_ejecta), on 1500 masses (0.9-300 Msun)
  x the table metallicities and their midpoints, with M_death = M_init - lifetime winds. For every sample
  0 <= rem <= M_death and no m_k < 0, and at grid stars rem / M_death = rem_tab / (rem_tab + sum ej_tab) (1e-12). With the
  fractions, rem <= M_death, rem >= 0 and m_k >= 0 hold by construction (clipped fractions, sum <= 1, rem = M_death - sum m_k):
  they fail only on the old absolute-remnant arithmetic. The grid remnant fraction is the check the new code can fail.
test_failed_sn_composition: a failed SN (42.3 Msun, log Z = -1.70, table ejecta 0 at the nearest grid star) and a direct
  collapse (a table grid star), each in its own box of gas with 1 km/s random velocities; element ledger, mass and energy
  traces close. Every failed SN and direct collapse in the table ejects nothing, so with exact death ejecta they deposit
  nothing; the old arithmetic deposited 5.2 Msun of hydrogen with zero energy for the failed SN, whose ejecta merge heat
  1/2 mu |v_star - v_j|^2 was then partly diluted away while booked in full (fb_sn_ia stage).
test_sibling_sne: two stars with the same ID (one parent cell), different child numbers, explode in the same step on one
  rank: each is injected once (2 x 1e51 erg booked = deposited), mass and elements close.
test_agb_energy / test_wind_metals / test_ia_iron / test_ecsn_energy: one run with a 3 Msun star (forced AGB death, then
  a forced Type Ia), a 60 Msun star (forced death: its remaining wind leaves first) and an ECSN (1e51 erg by default).
  AGB: the deposit delivers 1/2 M_ej (30 km/s)^2 (kick + inelastic heat); booked as 1/2 p^2 / M_ej with the delivered p,
  so the fb_wind_agb stage closes. Wind metals: FeedbackBudget books the metals the walk delivered (WINDZLEDGER, 1e-12).
  WINDZLEDGER compares two accumulations of the walk's own deposit, so it guards against a return to the old surface-abundance
  booking; that the gas gains the booked metals is checked by the element ledger of the trace.
  Ia: the event delivers the table's M_Ch of Ia matter, 0.744 Msun Fe (IAFE, 1e-12): the WD particle plus
  M_Ch - M_WD of receiver gas converted in place. With the rate normalised to IA_DTD_NORM per formed mass, Ia iron per
  formed mass is then rate x table yield.
test_sn_energies_default / _variable: a PISN, an ECSN and a CCSN in one box. Default: all 1e51 erg. With
  RESOLVED_ISM_VARIABLE_SN_ENERGY (Uli's choice 2026-11-19): PISN 1e52 erg, ECSN 5e50 erg, CCSN 1e51 (for reference: the He-core
  fit of Heger & Woosley 2002 via Wise+12a gives 4.87e52 erg at 200 Msun; Kitaura, Janka & Hillebrandt 2006 ~1e50 erg for an
  ECSN). Both: booked = deposited (no exact-stage line in the trace).
"""

import os
import re
import shutil
import signal
import subprocess
import time
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_fb_conservation"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXTRA_FLAGS = ("BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "OUTPUT_ENERGY_STEP_TRACE",
               "TEST_PRINT_DIFFUSION_H_CLOSURE")
# cross-timebin pair fluxes exchanged exactly, so the traces test the feedback itself (Uli 2026-11-21, B1 option a)
PAIR_FLUX_FLAGS = ("HYDRO_EXACT_PAIR_FLUX_EXCHANGE", "HYDRO_TIMESTEP_FROM_PAIR_FORCE")
BAR = 1e-12
RUN_TIMEOUT = 5400  # s
ZSUN = 1 - 0.7375 - 0.2485  # metal mass fraction of resolvedism_solar_massfrac (fresh starts take the composition from InitMetallicity)
PC, M_P, K_B, MSUN = 3.085678e18, 1.6726e-24, 1.3807e-16, 1.989e33
SOLAR = {2: 2.53e-3, 3: 7.41e-4, 4: 6.13e-3, 5: 1.34e-3, 6: 7.57e-4, 7: 7.12e-4, 8: 3.31e-4, 9: 6.87e-5, 10: 1.38e-3}

# scenario stars (Msun, log Z)
FSN0 = (42.3, -1.70)  # failed SN off the grid: the nearest grid star has no SN ejecta, the old arithmetic deposited 5.2 Msun of pure hydrogen


def table_star(rtype, ejecta, logZ_near, logM_near):
    """A grid star (M, log Z) of remnant type rtype (1 ECSN, 3 failed SN, 6 direct collapse) from the stellar table,
    with table SN ejecta > 0.5 Msun (ejecta True) or 0 (False), nearest to (logZ_near, logM_near)."""
    with h5py.File(join(REPO, "galaxy_sf", "stellar_tables", "stellar_tables_unified.hdf5"), "r") as f:
        lM, lZ = f["log_M_init"][:], f["log_Z"][:]
        rt = f["remnant_type"][:].reshape(len(lZ), len(lM))
        ej = f["elem_ej_SN_mass"][:].reshape(len(lZ), len(lM), -1).sum(-1)
    ok = (rt == rtype) & ((ej > 0.5) if ejecta else (ej <= 0))
    iz, im = np.nonzero(ok)
    assert len(iz), f"no grid star of type {rtype}"
    k = np.argmin((lZ[iz] - logZ_near) ** 2 + (lM[im] - logM_near) ** 2)
    return float(10 ** lM[im[k]]), float(lZ[iz[k]])


def _write_config():
    cfg = open(join(REPO, "Config_stella_default.sh")).read()
    with open(join(TEST_DIR, "Config.sh"), "w") as f:
        f.write(cfg + "\n# test hooks\n" + "\n".join(EXTRA_FLAGS) + "\n")


@pytest.fixture(scope="module")
def build():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    _write_config()
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=PAIR_FLUX_FLAGS,
                             post_build=lambda: link_test_driver(join(TEST_DIR, "death_ejecta_driver.cc"), join(TEST_DIR, "death_ejecta_driver")))
        shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "GIZMO_default"))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO_default")


@pytest.fixture(scope="module")
def build_variable_sn_energy():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    _write_config()
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=PAIR_FLUX_FLAGS + ("RESOLVED_ISM_VARIABLE_SN_ENERGY",))
        shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "GIZMO_variable_sn_energy"))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO_variable_sn_energy")


def make_box(fname, Z_abs, n_H=10.0, box_pc=16.0, m_cell=1.0, T=1.0e4, sigma_v=0.0, seed=1):
    """Uniform lattice, gas only, element mass fractions scaled-solar to total metallicity Z_abs; the cells nearest the
    centre and at +-L/4 carry IDs 1, 2, 3 (the test sites). sigma_v: random velocities [km/s]."""
    X_Z = Z_abs; X_He = 0.25 + 1.6 * X_Z; X_H = 1 - X_He - X_Z
    m_gas = n_H * (box_pc * PC) ** 3 * M_P / X_H / MSUN
    ng = int(round((m_gas / m_cell) ** (1 / 3))); n = ng ** 3
    box = box_pc * 1e-3
    g = (np.arange(ng) + 0.5) * box / ng
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ids = np.arange(n, dtype=np.uint32) + 10
    for k, c in enumerate(([0.5, 0.5, 0.5], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5])):
        ids[np.argmin(np.sum((pos - np.array(c) * box) ** 2, axis=1))] = k + 1
    ea = np.zeros((n, 27), dtype=np.float64)
    zsum = sum(SOLAR.values())
    for k, x in SOLAR.items():
        ea[:, k] = x / zsum * X_Z
    ea[:, 1] = X_He; ea[:, 0] = 1 - ea[:, 1:].sum(axis=1)
    vel = np.random.default_rng(seed).normal(0, sigma_v, (n, 3)) if sigma_v > 0 else np.zeros((n, 3))
    u = 1.5 * K_B * T / (1.22 * M_P) / 1e10
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs.update(BoxSize=box, NumPart_ThisFile=np.array([n, 0, 0, 0, 0, 0], dtype=np.int32), NumPart_Total=np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32),
                       NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32), MassTable=np.zeros(6), Time=0.0, Redshift=0.0, NumFilesPerSnapshot=1,
                       Flag_Sfr=1, Flag_Cooling=1, Flag_StellarAge=1, Flag_Metals=27, Flag_Feedback=1, Flag_DoublePrecision=0, HubbleParam=1.0)
        p = f.create_group("PartType0")
        f32 = np.float32
        p["Coordinates"] = pos.astype(f32); p["Velocities"] = vel.astype(f32)
        p["Masses"] = np.full(n, m_gas / n * MSUN / 1.989e43, dtype=f32); p["ParticleIDs"] = ids
        p["InternalEnergy"] = np.full(n, u, dtype=f32); p["ElementAbundance"] = ea.astype(f32)
        b = np.zeros((n, 3), dtype=f32); b[:, 2] = 1e-8; p["MagneticField"] = b
    return n


def run_killable(cmd, timeout, **kw):
    """subprocess.run in its own process group. On timeout: SIGTERM to the group (srun forwards it to the step's tasks and
    cancels the step; mpirun forwards it too), then SIGKILL after 30 s."""
    p = subprocess.Popen(cmd, start_new_session=True, **kw)
    try:
        return subprocess.CompletedProcess(cmd, p.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait()
        raise


def test_timeout_kills_the_launched_step():
    """run_killable on a step that would hang (sleep under the MPI launcher): after the timeout no process of it is left."""
    tag = f"fbc_hang_{os.getpid()}"
    with pytest.raises(subprocess.TimeoutExpired):
        run_killable(mpi_launch_command(2) + ["bash", "-c", f"exec -a {tag} sleep 600"], 5)
    time.sleep(10)
    left = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
    assert not left, f"processes of the timed-out step still running: {left}"


def zpar(logZ):
    return {"InitMetallicity": 10 ** logZ / ZSUN, "InitialMetallicity": 10 ** logZ / ZSUN}


def run_case(name, binary, params, nranks=2, restart_flag=0):
    """Run GIZMO in TEST_DIR/run_<name> with the base parameter file plus overrides; returns (log text, output dir)."""
    d = join(TEST_DIR, "run_" + name)
    out = join(d, "output")
    if not os.path.isdir(d):
        os.makedirs(d)
    if not os.path.islink(join(d, "gizmo")):
        os.symlink(REPO, join(d, "gizmo"))
    base = open(join(TEST_DIR, f"{TEST_NAME}.params")).read().splitlines()
    keys = set(params)
    lines = [l for l in base if not (l.split() and l.split()[0] in keys)] + [f"{k} {v}" for k, v in params.items()]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    if restart_flag == 0:
        shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    with open(join(d, "run.log"), "w") as log:
        r = run_killable(mpi_launch_command(nranks) + [binary, "run.params", str(restart_flag)], RUN_TIMEOUT, stdout=log, stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0 and "Simulation ends." in text, f"{name}: GIZMO did not reach TimeMax ({r.returncode}): " + text[-1500:]
    return text, out


def kv_lines(text, tag):
    return [{k: float(v) for k, v in (t.split("=") for t in l.split()[1:])} for l in text.splitlines() if l.startswith(tag + " ")]


def traces_ok(out):
    rep = keep_and_check_traces(out, out)
    assert rep.ok, rep.summary()


# ---------------------------------------------------------------------------------------------------------------------

def test_death_ejecta_table(build):
    d = join(TEST_DIR, "run_driver"); os.makedirs(d, exist_ok=True)
    if not os.path.islink(join(d, "gizmo")):
        os.symlink(REPO, join(d, "gizmo"))
    make_box(join(d, "box_ics.hdf5"), ZSUN)
    base = open(join(TEST_DIR, f"{TEST_NAME}.params")).read().splitlines()
    open(join(d, "run.params"), "w").write("\n".join([l for l in base if not l.startswith("InitCondFile")] + ["InitCondFile box_ics"]) + "\n")
    r = subprocess.run(mpi_launch_command(1) + [join(TEST_DIR, "death_ejecta_driver"), "run.params"], capture_output=True, text=True, cwd=d, timeout=RUN_TIMEOUT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ej = np.array([l.split()[1:] for l in r.stdout.splitlines() if l.startswith("EJ ")], dtype=float)
    grid = np.array([l.split()[1:] for l in r.stdout.splitlines() if l.startswith("GRID ")], dtype=float)
    assert len(ej) >= 29 * 1501 and len(grid) > 100  # 1501 masses x 15 grid metallicities and 14 midpoints
    M, rem, s, mn = ej[:, 3], ej[:, 4], ej[:, 5], ej[:, 6]
    over = (rem - M) / M
    n_over = int((over > BAR).sum()); n_neg = int((mn < 0).sum()); n_rem_neg = int((rem < -BAR * M).sum())
    grid_err = np.abs(grid[:, 3] - grid[:, 4]).max()
    worst = ej[np.argmax(over)]
    print(f"{len(ej)} samples: remnant above the star {n_over} (max {over.max():+.3e} at logM={worst[0]:.3f} logZ={worst[1]:.3f} "
          f"type={int(worst[2])} M_death={worst[3]:.3f} rem={worst[4]:.3f}), negative m_k {n_neg}, negative rem {n_rem_neg}; "
          f"grid remnant fraction max |err| {grid_err:.3e}")
    assert n_over == 0 and n_neg == 0 and n_rem_neg == 0
    assert grid_err < 1e-12


@pytest.mark.parametrize("case", ["failed_sn", "direct_collapse"])
def test_failed_sn_composition(build, case):
    M, logZ = FSN0 if case == "failed_sn" else table_star(6, False, -2.0, np.log10(60.0))
    d = join(TEST_DIR, "run_" + case); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), 10 ** logZ, sigma_v=1.0)
    text, out = run_case(case, build, {"InitCondFile": "box_ics", "TestNumStars": 1, "TestStarMass": M, "TestSNTime0": 1.0e-6, "TimeMax": 4.0e-6, **zpar(logZ)})
    sn = re.findall(r"RESOLVEDISM SN: .* rem_type=(\d+)", text)
    print(f"{case}: M = {M:.2f} Msun, log Z = {logZ:.2f}, rem_types {sn}")
    assert [int(x) for x in sn] == [3 if case == "failed_sn" else 6], f"expected one {case}, got rem_types {sn}"
    traces_ok(out)


def test_sibling_sne(build):
    d = join(TEST_DIR, "run_sibling"); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), 0.0142 * 0.3)
    # phase 1: write the start as a snapshot (no star forms), then give cell 2 the ID of cell 1 with another child number
    run_case("sibling", build, {"InitCondFile": "box_ics", "TestStarMass": 0, "TimeMax": 1.0e-9, "TimeBetSnapshot": 1.0e-9, **zpar(np.log10(0.3 * ZSUN))}, nranks=1)
    snap = join(d, "output", "snapshot_000.hdf5")
    with h5py.File(snap, "r+") as f:
        ids = f["PartType0/ParticleIDs"][:]; ch = f["PartType0/ParticleChildIDsNumber"][:]
        j2 = np.where(ids == 2)[0][0]; ids[j2] = 1; ch[j2] = 2
        f["PartType0/ParticleIDs"][...] = ids; f["PartType0/ParticleChildIDsNumber"][...] = ch
    shutil.copy(snap, join(d, "sibling_000.hdf5"))
    text, out = run_case("sibling", build, {"InitCondFile": "sibling_000", "TestNumStars": 1, "TestStarMass": 20.0,
                                            "TestSNTime0": 1.0e-6, "TimeMax": 2.5e-6, **zpar(np.log10(0.3 * ZSUN))}, nranks=1, restart_flag=2)
    sn = re.findall(r"RESOLVEDISM SN: .* E=([0-9.e+]+)\[erg\]", text)
    assert len(sn) == 2, f"expected 2 SNe, got {len(sn)}"
    traces_ok(out)


@pytest.fixture(scope="module")
def late_run(build):
    M_ecsn, logZ = table_star(1, True, np.log10(ZSUN), np.log10(9.0))
    print(f"ECSN: M = {M_ecsn:.2f} Msun, log Z = {logZ:.2f}")
    d = join(TEST_DIR, "run_late"); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), 10 ** logZ, n_H=10.0)
    return run_case("late", build, {"InitCondFile": "box_ics", "TestNumStars": 3, "TestStarMass": 3.0, "TestStarMass2": 60.0,
                                    "TestStarMass3": M_ecsn, "TestSNTime0": 1.0e-6, "TestSNSpacing": 1.0e-6, "TestIaTime": 3.5e-6,
                                    "TimeMax": 5.0e-6, **zpar(logZ)}, nranks=2)


def test_agb_energy(late_run):
    text, out = late_run
    assert "RESOLVEDISM AGB:" in text, "no AGB death"
    traces_ok(out)


def test_wind_metals(late_run):
    text, out = late_run
    rows = [r for r in kv_lines(text, "WINDZLEDGER") if r["Z_dep"] > 0 or r["Z_book"] > 0]
    assert rows, "no wind metals delivered"
    err = max(abs(r["Z_book"] / r["Z_dep"] - 1) if r["Z_dep"] > 0 else 1.0 for r in rows)
    print(f"{len(rows)} wind steps, max |Z_book / Z_delivered - 1| {err:.3e}")
    assert err < BAR


def test_ia_iron(late_run):
    text, out = late_run
    rows = kv_lines(text, "IAFE")
    assert len(rows) == 1, f"expected one Ia, got {len(rows)}"
    err = abs(rows[0]["Fe"] / rows[0]["Fe_table"] - 1)
    print(f"Ia iron delivered {rows[0]['Fe']:.6f} Msun, table {rows[0]['Fe_table']:.6f} Msun, rel. err {err:.3e}")
    assert err < BAR
    traces_ok(out)


def cubic_spline_w(u):
    """GIZMO's default kernel (KERNEL_FUNCTION 3) without its normalisation."""
    return np.where(u < 0.5, 1 + 6 * (u - 1) * u * u, 2 * (1 - u) ** 3)


def ia_conversion_check(text):
    """The gas turned into Ia composition is c_j = w_j dM_conv, with the ejecta's weight w_j = m_j W(u_j) / sum m W recomputed
    here from each receiver's printed u and m (not taken from the code), and c_j = m_j where w_j dM_conv > m_j (the clamp). The
    clamp's shortfall is booked (RESOLVEDISM TYPE_IA CLAMP line) and sum c_j + shortfall = dM_conv. Ia iron delivered =
    Fe_table (M_WD + sum c_j) / M_Ch. All bars BAR = 1e-12 of dM_conv or of the quantity."""
    src = kv_lines(text, "IACONVSRC")
    assert src and len({r["id"] for r in src}) == 1, f"expected one Ia, got {src}"
    s = src[0]; dM = s["dM_conv"]
    cells = kv_lines(text, "IACONVCELL")
    assert cells, "no Ia receiver printed"
    u, m, c = (np.array([r[k] for r in cells]) for k in ("u", "m", "c"))
    w = m * cubic_spline_w(u); w = w / w.sum()
    want = w * dM
    clamped = want > m
    dev = np.abs(np.where(clamped, c - m, c - want)).max() / dM
    short = (want - m)[clamped].sum()
    tot = abs((c.sum() + short) / dM - 1)
    book = re.findall(r"RESOLVEDISM TYPE_IA CLAMP: t=\S+ cells=(\d+) short=(\S+)\[code\]", text)
    n_book = sum(int(a) for a, _ in book); s_book = sum(float(b) for _, b in book)
    fe = kv_lines(text, "IAFE")
    assert len(fe) == 1, f"expected one IAFE line, got {len(fe)}"
    fe_err = abs(fe[0]["Fe"] / fe[0]["Fe_table"] - (s["Mej"] + c.sum()) / s["Mch"])
    print(f"Ia conversion: {len(cells)} receivers, dM_conv {dM:.6e}, max |c_j - expected| / dM_conv {dev:.3e}, "
          f"|sum c_j + shortfall - dM_conv| / dM_conv {tot:.3e}, clamped {clamped.sum()} (booked {n_book}), shortfall / dM_conv "
          f"{short / dM:.3e} (booked {s_book / dM:.3e}), Fe err {fe_err:.3e}")
    assert dev < BAR
    assert tot < BAR
    assert n_book == clamped.sum()
    assert abs(s_book - short) < BAR * dM
    assert fe_err < BAR
    return clamped.sum()


def test_ia_conversion_follows_ejecta_weights(late_run):
    """STL-180: the converted gas is spread with the ejecta's kernel weight (cubic spline, no HEALPix in this build), so a receiver
    at the kernel edge gets none of it (Enzyme audit check C8c: jump of 0.147 of sum |d(m X)| for the old top-hat)."""
    text, out = late_run
    cfg = open(join(TEST_DIR, "Config.sh")).read()
    assert not re.search(r"^\s*(KERNEL_FUNCTION|GALSF_RESOLVEDISM_FB_HEALPIX)\b", cfg, re.M), "the check assumes the cubic spline weight"
    ia_conversion_check(text)
    traces_ok(out)


def clamp_box_run(binary, name):
    """A lone 3 Msun star (AGB death, then a forced Ia) in a 4 pc box of 0.02 Msun cells, where w_j dM_conv exceeds the inner
    cells' mass, so the Ia clamp converts some cells whole."""
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), ZSUN, n_H=10.0, box_pc=4.0, m_cell=0.02)
    return run_case(name, binary, {"InitCondFile": "box_ics", "TestNumStars": 1, "TestStarMass": 3.0, "TestStarMass2": 0,
                                   "TestStarMass3": 0, "TestSNTime0": 1.0e-6, "TestIaTime": 3.5e-6, "TimeMax": 5.0e-6,
                                   **zpar(np.log10(ZSUN))})


@pytest.fixture(scope="module")
def clamp_run(build):
    return clamp_box_run(build, "ia_clamp")


def diffusion_check(text):
    """STL-277: after every diffusion unpack (TEST_PRINT_DIFFUSION_H_CLOSURE), no slot floor fired in any cell (each slot's mass
    = m X + exchange to 1e-12 m), and every cell keeps sum X(1..27) = 1 and X0 = sum of its metal slots to BAR."""
    rows = kv_lines(text, "DIFFSLOT")
    assert rows, "no DIFFSLOT rows: build without TEST_PRINT_DIFFUSION_H_CLOSURE?"
    assert all("pairs_over_count" in r for r in rows), "DIFFSLOT rows without pairs_over_count: not the pairwise exchange (STL-277)"
    nover = sum(r["pairs_over_count"] for r in rows)
    assert sum(r["evals"] for r in rows) > 0, "no pair exchange evaluated"
    nf = sum(r["floor_cells"] for r in rows); fm = sum(r["floor_abs"] for r in rows)
    sd = max(r["max_slot_sum_dev"] for r in rows); zd = max(r["max_z0_dev"] for r in rows)
    nneg = sum(r.get("neg_in_cells", 0) for r in rows)  # slots another module left below zero (round-off); their credit is booked unfloored
    print(f"diffusion: {len(rows)} unpacks, floor fired in {nf:.0f} cell-steps (|mass| {fm:.3e} code), worst per-cell |sum X - 1| {sd:.3e}, "
          f"|X0 - sum metals| {zd:.3e}, cells with more exchange evaluations than counted pairs {nover:.0f}, "
          f"cell-steps with a slot below zero before the exchange {nneg:.0f}")
    assert nover == 0, "a cell had more exchange evaluations than its counted pairs: the debit bound is not strict"
    assert nf == 0, "a slot floor fired in the diffusion unpack"
    assert sd <= BAR and zd <= BAR


def test_ia_conversion_clamp(clamp_run):
    """STL-180 clamp: the clamp must fire (else this run does not test it), every clamped cell is converted whole, the shortfall
    is booked, and the traces close."""
    text, out = clamp_run
    assert ia_conversion_check(text) > 0, "the clamp did not fire"
    traces_ok(out)


def test_diffusion_slots_clamp_box(clamp_run):
    """STL-277: cells converted whole to Ia composition (H ~ 0) next to H-rich gas: no floor fires and every cell's slots close."""
    diffusion_check(clamp_run[0])


@pytest.fixture(scope="module")
def build_equal_timesteps():
    """The production build with FORCE_EQUAL_TIMESTEPS: every pair exchanges on both sides at once."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    _write_config()
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=PAIR_FLUX_FLAGS + ("FORCE_EQUAL_TIMESTEPS",))
        shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "GIZMO_equal_timesteps"))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO_equal_timesteps")


def test_clamp_box_equal_timesteps(build_equal_timesteps):
    """STL-277 at equal timesteps: the clamp box with every exchange applied to both cells of a pair at once. Every element and
    the metal ledger close at the trace bars (unchanged), no floor fires, every cell's slots close."""
    text, out = clamp_box_run(build_equal_timesteps, "ia_clamp_equal_dt")
    assert ia_conversion_check(text) > 0, "the clamp did not fire"
    diffusion_check(text)
    traces_ok(out)


def test_ecsn_energy(late_run):
    text, out = late_run
    e = [float(x) for x in re.findall(r"RESOLVEDISM SN: .* rem_type=1 E=([0-9.e+]+)\[erg\]", text)]
    assert e == [1.0e51], f"ECSN energies {e} (default: fixed 1e51)"
    traces_ok(out)


def sn_energy_run(binary, name):
    """A PISN, an ECSN and a CCSN (table grid stars at one metallicity) explode in one box (forced, 1 kyr apart)."""
    M_pisn, logZ = table_star(5, True, -4.0, np.log10(200.0))
    M_ecsn, z2 = table_star(1, True, logZ, np.log10(9.0))
    M_ccsn, z3 = table_star(2, True, logZ, np.log10(20.0))
    assert z2 == logZ and z3 == logZ
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), 10 ** logZ)
    text, out = run_case(name, binary, {"InitCondFile": "box_ics", "TestNumStars": 3, "TestStarMass": M_pisn, "TestStarMass2": M_ecsn,
                                        "TestStarMass3": M_ccsn, "TestSNTime0": 1.0e-6, "TestSNSpacing": 1.0e-6, "TimeMax": 4.5e-6,
                                        "TestFUVZero": 1, **zpar(logZ)})
    e = {int(t): float(x) for t, x in re.findall(r"RESOLVEDISM SN: .* rem_type=(\d+) E=([0-9.e+]+)\[erg\]", text)}
    print(f"{name}: PISN {M_pisn:.0f} Msun, ECSN {M_ecsn:.0f}, CCSN {M_ccsn:.0f} at log Z {logZ:.2f}: energies by remnant type {e}")
    return e, out, M_pisn


def stages_closed(out):
    """booked = deposited: no exact-stage line in the energy trace, and the mass and element ledgers close."""
    rep = keep_and_check_traces(out, out)
    bad = [f for f in rep.findings if f["kind"] in ("energy_stage", "mass", "element_ledger", "metal_ledger")]
    assert not bad, rep.summary()
    return rep


def test_metal_ledger_catches_a_real_change(build):
    """The PISN box (production build): one cell gains 1.001e-12 of the box's metal mass, unbooked, at 2 kyr (test hook). The mass
    trace must report exactly that change and nothing else: one metal_ledger finding at pre_fb_mom_p0 with |dL|/L = 1.001e-12
    (bar 1e-12, unchanged), no other mass, metal or element finding. Before the ledger sums were exact enough, the double summation
    of the ~1e-8 metal terms left jumps of up to 5e-12 of L at later checkpoints."""
    M_pisn, logZ = table_star(5, True, -4.0, np.log10(200.0))
    M_ecsn, _ = table_star(1, True, logZ, np.log10(9.0))
    M_ccsn, _ = table_star(2, True, logZ, np.log10(20.0))
    d = join(TEST_DIR, "run_metal_kick"); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), 10 ** logZ)
    with h5py.File(join(d, "box_ics.hdf5"), "r") as f:
        L = float(f["PartType0/Masses"][:].astype(np.float64).sum()) * 10 ** logZ  # the box's metal mass (InitMetallicity sets every cell)
    kick = 1.001e-12 * L
    text, out = run_case("metal_kick", build, {"InitCondFile": "box_ics", "TestNumStars": 3, "TestStarMass": M_pisn, "TestStarMass2": M_ecsn,
                                               "TestStarMass3": M_ccsn, "TestSNTime0": 1.0e-6, "TestSNSpacing": 1.0e-6, "TimeMax": 4.5e-6,
                                               "TestFUVZero": 1, "TestMetalKickTime": 2.0e-6, "TestMetalKickMass": f"{kick:.17e}", **zpar(logZ)})
    assert text.count("TESTMETALKICK") == 1
    rep = keep_and_check_traces(out, out)
    found = [f for f in rep.findings if f["kind"] in ("mass", "metal_ledger", "element_ledger")]
    print(rep.summary())
    assert len(found) == 1 and found[0]["kind"] == "metal_ledger" and found[0]["stage"] == "pre_fb_mom_p0" and found[0]["count"] == 1, rep.summary()
    assert abs(found[0]["max_rel"] / 1.001e-12 - 1) < 0.02, found[0]
    # residual floor (fixed 2026-11-20, before the run): every other checkpoint |dL|/L < 1e-13. Excepted: post_sf_spawn and
    # post_imf_accrete, where the new star's 1e-10 Msun placeholder moves 2.2e-13 of L (gone with the zero-mass spawn of
    # fix/trace-imf-2026-11-17, a13d49557)
    rows = [l.split() for l in open(join(out, "mass_balance_steps.txt")) if l.strip() and not l.startswith("#")][1:]
    floor = max((abs(float(r[-1])) / float(r[-2]) for r in rows if not (r[1] == "pre_fb_mom_p0" and abs(float(r[2]) - found[0]["first_t"]) < 1e-12)
                 and r[1] not in ("post_sf_spawn", "post_imf_accrete")), default=0.0)
    print(f"metal ledger residual floor (other checkpoints): {floor:.3e} of L")
    assert floor < 1e-13, floor


def test_sn_energies_default(build):
    """Default: every explosive SN type deposits and books 1e51 erg."""
    e, out, M_pisn = sn_energy_run(build, "sn_energy_default")
    assert e == {5: 1.0e51, 1: 1.0e51, 2: 1.0e51}, e
    rep = stages_closed(out)
    assert rep.ok, rep.summary()


def test_sn_energies_variable(build_variable_sn_energy):
    """RESOLVED_ISM_VARIABLE_SN_ENERGY (Uli 2026-11-19): PISN 1e52, ECSN 5e50, CCSN 1e51."""
    e, out, M_pisn = sn_energy_run(build_variable_sn_energy, "sn_energy_variable")
    assert set(e) == {5, 1, 2}, e
    assert e == {5: 1.0e52, 1: 5.0e50, 2: 1.0e51}, e
    rep = stages_closed(out)
    assert rep.ok, rep.summary()


@pytest.fixture(scope="module")
def build_default_hydro():
    """The production build without the pair-flux flags."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    _write_config()
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME)
        shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "GIZMO_default_hydro"))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO_default_hydro")


@pytest.mark.xfail(strict=True, reason="known property of the default build (STL-55, STL-127): with individual timesteps the one-sided "
                   "cross-timebin pair fluxes leave a 1-3 % time-integration residual after hot deposits; measured at fix/fb-audit "
                   "4748ca111 (fbfix_all10): sibling SNe 1.0e-2, AGB/Ia/ECSN box 1.7e-2, PISN box with RESOLVED_ISM_VARIABLE_SN_ENERGY "
                   "2.3e-2, stella_fb_wakeup 2.3e-2, boundary-wind flag-on arm 2.9e-2 of E_scale against the 1e-2 bar; all close with "
                   "HYDRO_EXACT_PAIR_FLUX_EXCHANGE + HYDRO_TIMESTEP_FROM_PAIR_FORCE. A pass means the default build changed: re-examine.")
def test_default_build_integration_residual(build_default_hydro):
    """The PISN, ECSN and CCSN box (fixed 1e51 each) on the default hydro: the time-integration bar of the trace."""
    e, out, _ = sn_energy_run(build_default_hydro, "sn_energy_default_hydro")
    rep = keep_and_check_traces(out, out)
    assert rep.ok, rep.summary()
