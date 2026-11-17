"""Stella feedback conservation on the production build (Config_stella_default.sh plus the isolated-feedback test hooks and
the energy step trace): every channel deposits what it books, mass and every element close, and the death ejecta add up.

One build serves all tests. Every run also checks the mass, element and energy step traces (gizmo.conservation_trace,
bars fixed there). Units: kpc, 1e10 Msun, km/s.

test_death_ejecta_table (no run): the death ejecta the deposit uses (stellar_death_ejecta), on 1500 masses (0.9-300 Msun)
  x the table metallicities and their midpoints, with M_death = M_init - lifetime winds. For every sample
  rem + sum_k m_k = M_death (1e-12), no m_k < 0, rem >= 0 (the star never ejects more than it holds) and no ejecta
  without composition; at grid stars rem / M_death = rem_tab / (rem_tab + sum ej_tab).
test_failed_sn_composition: a failed SN (42.3 Msun, log Z = -1.70, table ejecta 0 at the nearest grid star) and a direct
  collapse (a table grid star), each in its own box; element ledger, mass and energy traces close.
test_failed_sn_merge_heat: failed SN with ejecta, at a grid star, in gas with random velocities: the ejecta merge turns
  1/2 mu |v_star - v_j|^2 into heat (booked as info); deposit = booking, so the fb_sn_ia stage closes.
test_sibling_sne: two stars with the same ID (one parent cell), different child numbers, explode in the same step on one
  rank: each is injected once (2 x 1e51 erg booked = deposited), mass and elements close.
test_agb_energy / test_wind_metals / test_ia_iron / test_ecsn_energy: one run with a 3 Msun star (forced AGB death, then
  a forced Type Ia), a 60 Msun star (forced death: its remaining wind leaves first) and an ECSN.
  AGB: the deposit delivers 1/2 M_ej (30 km/s)^2 (kick + inelastic heat); booked as 1/2 p^2 / M_ej with the delivered p,
  so the fb_wind_agb stage closes. Wind metals: FeedbackBudget books the metals the walk delivered (WINDZLEDGER, 1e-12).
  Ia: the event delivers the table's M_Ch of Ia matter, 0.744 Msun Fe (IAFE, 1e-12): the WD particle plus
  M_Ch - M_WD of receiver gas converted in place. With the rate normalised to IA_DTD_NORM per formed mass, Ia iron per
  formed mass is then rate x table yield. ECSN: 1e50 erg (Kitaura, Janka & Hillebrandt 2006, A&A 450, 345: 1.2e50 erg
  for the 8.8 Msun O-Ne-Mg core collapse), booked = deposited.
"""

import os
import re
import shutil
import subprocess
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_fb_conservation"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXTRA_FLAGS = ("BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "OUTPUT_ENERGY_STEP_TRACE")
BAR = 1e-12
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
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, "death_ejecta_driver.cc"), join(TEST_DIR, "death_ejecta_driver")))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO")


def make_box(fname, Z_abs, n_H=10.0, box_pc=16.0, m_cell=1.0, T=100.0, sigma_v=0.0, seed=1, child_ids=None):
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
        r = subprocess.run(mpi_launch_command(nranks) + [binary, "run.params", str(restart_flag)], stdout=log, stderr=subprocess.STDOUT, cwd=d)
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
    r = subprocess.run(mpi_launch_command(1) + [join(TEST_DIR, "death_ejecta_driver"), "run.params"], capture_output=True, text=True, cwd=d)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ej = np.array([l.split()[1:] for l in r.stdout.splitlines() if l.startswith("EJ ")], dtype=float)
    grid = np.array([l.split()[1:] for l in r.stdout.splitlines() if l.startswith("GRID ")], dtype=float)
    assert len(ej) > 50000 and len(grid) > 100
    M, rem, s, mn = ej[:, 3], ej[:, 4], ej[:, 5], ej[:, 6]
    closure = np.abs(rem + s - M) / M
    n_bad = int((closure > BAR).sum()); n_neg = int((mn < 0).sum()); n_rem_neg = int((rem < -BAR * M).sum())
    grid_err = np.abs(grid[:, 3] - grid[:, 4]).max()
    worst = ej[np.argmax(closure)]
    print(f"{len(ej)} samples: closure fails {n_bad} (max {closure.max():.3e} at logM={worst[0]:.3f} logZ={worst[1]:.3f} type={int(worst[2])} "
          f"M_death={worst[3]:.3f} rem={worst[4]:.3f} sum_m={worst[5]:.3f}), negative m_k {n_neg}, negative rem {n_rem_neg}; "
          f"grid remnant fraction max |err| {grid_err:.3e}")
    assert n_bad == 0 and n_neg == 0 and n_rem_neg == 0
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


def test_failed_sn_merge_heat(build):
    M, logZ = table_star(3, True, -2.0, np.log10(40.0))
    d = join(TEST_DIR, "run_failed_sn_heat"); os.makedirs(d, exist_ok=True)
    make_box(join(d, "box_ics.hdf5"), 10 ** logZ, sigma_v=3.0)
    print(f"failed SN with ejecta: M = {M:.2f} Msun, log Z = {logZ:.2f}")
    text, out = run_case("failed_sn_heat", build, {"InitCondFile": "box_ics", "TestNumStars": 1, "TestStarMass": M,
                                                   "TestSNTime0": 1.0e-6, "TimeMax": 3.0e-6, **zpar(logZ)})
    assert re.search(r"RESOLVEDISM SN: .* rem_type=3 ", text), "no failed SN"
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


def test_ecsn_energy(late_run):
    text, out = late_run
    e = [float(x) for x in re.findall(r"RESOLVEDISM SN: .* rem_type=1 E=([0-9.e+]+)\[erg\]", text)]
    assert e == [1.0e50], f"ECSN energies {e}"
    traces_ok(out)
