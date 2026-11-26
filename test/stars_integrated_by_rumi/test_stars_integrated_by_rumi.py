"""Type-4 stars integrated by the RUMI++ library (STARS_INTEGRATED_BY_RUMI): the eccentric binary and hierarchical triple of
test/stars_hermite_direct_gravity against the same Newtonian reference orbits. Runs only with RUMI_PATH set (the library install)."""

import glob
import importlib.util
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, build_gizmo_for_test, mpi_launch_command, variant_output_dir, variant_suffix

TEST_NAME = "stars_integrated_by_rumi"
TEST_DIR = f"test/{TEST_NAME}"
REF_DIR = "test/stars_hermite_direct_gravity"
_spec = importlib.util.spec_from_file_location("fewbody_reference", f"{REF_DIR}/fewbody_reference.py")
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

pytestmark = pytest.mark.skipif(not os.environ.get("RUMI_PATH"), reason="RUMI_PATH (RUMI++ library install) not set")

T_BINARY = 2.29e-4  # ten orbits of the a = 1000 AU binary
T_TRIPLE = 3.74e-5  # two outer orbits of the triple (52 inner orbits)


def _run(ics, tmax, flags=(), params=None):
    subprocess.run([sys.executable, os.path.abspath(f"{REF_DIR}/fewbody_reference.py")], cwd=TEST_DIR, check=True)
    p = {"InitCondFile": ics, "TimeMax": tmax, "TimeBetSnapshot": tmax, "TimeBetStatistics": tmax}
    p.update(params or {})
    build_and_run_test(TEST_NAME, 1, 1, extra_config_flags=flags, param_overrides=p)
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5"))
    with h5py.File(f"{TEST_DIR}/{ics}.hdf5", "r") as f:  # the ICs: the first snapshot already carries the opening half kick
        g = f["PartType4"]; ic = (g["Coordinates"][:], g["Velocities"][:], g["Masses"][:])
    return ic, ref.read(snaps[-1]), open(f"{TEST_DIR}/test_{TEST_NAME}{variant_suffix(flags)}.out").read()


def _errors(ics, tmax, flags=(), params=None):
    (x0, v0, m), (x1, v1, _, t1), log = _run(ics, tmax, flags, params)
    assert abs(t1 / tmax - 1) < 1e-6
    xr, vr = ref.reference(x0, v0, m, t1)
    de = abs(ref.energy(x1, v1, m) / ref.energy(x0, v0, m) - 1)
    a0 = ref.elements(x0[:2], v0[:2], m[:2])[0]
    dx = np.max(np.linalg.norm((x1 - x1[0]) - (xr - xr[0]), axis=1)) / a0
    return de, dx, log


def test_startup():
    """the library starts: its build id in the log, one RUMI++ step count at the end"""
    _, _, log = _errors("binary_ics", T_BINARY / 10)
    assert "RUMI++: RUMI_LIB_ID api=3" in log and "RUMI++ totals:" in log and "RUMI++ WARNING" not in log


# RUMI++ integrates the stars; bars 3x measured (binary dE/E 5.9e-13, offset 2.7e-10 a; triple 1.4e-9, 5.3e-7 a_in)
def test_binary():
    de, dx, log = _errors("binary_ics", T_BINARY)
    print(f"binary: dE/E {de:.3e} position {dx:.3e}")
    assert de < 2e-12
    assert "RUMI++ WARNING" not in log and dx < 1e-9


def test_triple():
    de, dx, log = _errors("triple_ics", T_TRIPLE)
    print(f"triple: dE/E {de:.3e} position {dx:.3e}")
    assert de < 4.5e-9
    assert "RUMI++ WARNING" not in log and dx < 1.6e-6


def test_snapshots_inside_steps():
    """outputs at times that fall inside the stars' common steps show RUMI++'s state there (the step is split at the output):
    energy and positions of every snapshot against the reference"""
    p = {"TimeBetSnapshot": T_BINARY / 7.3, "InitCondFile": "binary_ics", "TimeMax": T_BINARY, "TimeBetStatistics": T_BINARY}
    subprocess.run([sys.executable, os.path.abspath(f"{REF_DIR}/fewbody_reference.py")], cwd=TEST_DIR, check=True)
    build_and_run_test(TEST_NAME, 1, 1, param_overrides=p)
    with h5py.File(f"{TEST_DIR}/binary_ics.hdf5", "r") as f:
        g = f["PartType4"]; x0, v0, m = g["Coordinates"][:], g["Velocities"][:], g["Masses"][:]
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, ()) + "/snapshot_*.hdf5"))[1:]
    worst_de = worst_dx = 0
    for s in snaps:
        x, v, _, t = ref.read(s); xr, vr = ref.reference(x0, v0, m, t)
        worst_de = max(worst_de, abs(ref.energy(x, v, m) / ref.energy(x0, v0, m) - 1))
        worst_dx = max(worst_dx, np.linalg.norm((x[0] - x[1]) - (xr[0] - xr[1])) / (1000 * ref.AU))
    print(f"{len(snaps)} snapshots: worst dE/E {worst_de:.3e} position {worst_dx:.3e}")
    assert len(snaps) >= 7 and worst_de < 4e-10 and worst_dx < 8e-10  # 3x measured (1.2e-10, 2.6e-10 a)


STOP_STEP = 40


def test_restart():
    """stopped at a step and restarted from the restart files (RUMI++'s state among them), the run ends where the
    uninterrupted run ends"""
    flags = ("TEST_RESTART_AT_STEP",)
    subprocess.run([sys.executable, os.path.abspath(f"{REF_DIR}/fewbody_reference.py")], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME, 1, flags)
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        prm = "restart_test.params"
        with open(f"{TEST_NAME}.params") as f, open(prm, "w") as g:
            for line in f:
                k = line.split()[0] if line.split() else ""
                g.write({"InitCondFile": "InitCondFile triple_ics\n", "TimeMax": f"TimeMax {T_TRIPLE}\n", "TimeBetSnapshot": f"TimeBetSnapshot {T_TRIPLE}\n"}.get(k, line))
        out = {}
        for case, env in (("plain", {}), ("restarted", {"GIZMO_TEST_RESTART_AT_STEP": str(STOP_STEP)})):
            shutil.rmtree("output", ignore_errors=True)
            subprocess.run(mpi_launch_command(1, 1) + ["./GIZMO", prm, "0"], stdout=open(f"{case}.out", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, **env), check=False)
            if env:
                res = subprocess.run(mpi_launch_command(1, 1) + ["./GIZMO", prm, "1"], stdout=open("restart.out", "w"), stderr=subprocess.STDOUT)
                assert res.returncode == 0 and "RUMI++: restarted from" in open("restart.out").read()
            x, v, _, t = ref.read(sorted(glob.glob("output/snapshot_*.hdf5"))[-1]); out[case] = (x, v, t)
    finally:
        os.chdir(cwd)
    (xa, va, ta), (xb, vb, tb) = out["plain"], out["restarted"]
    print(f"restart: max|dx| {np.abs(xa - xb).max():.3e} max|dv| {np.abs(va - vb).max():.3e}")
    assert ta == tb and np.array_equal(xa, xb) and np.array_equal(va, vb)


def test_merger():
    """a grazing pass (pericentre 0.8 (R1 + R2), the ICs of test/ketju_star_collision) merges inside RUMI++: the heavier star
    survives with the pair's mass and CoM velocity, mass and momentum are conserved"""
    coll = os.path.abspath("test/ketju_star_collision")
    subprocess.run([sys.executable, f"{coll}/make_ketju_star_collision_ics.py"], cwd=TEST_DIR, check=True)
    p = {"InitCondFile": "ketju_star_collision_graze_ics", "TimeMax": 1e-7, "TimeBetSnapshot": 2e-8, "TimeBetStatistics": 2e-8, "MaxSizeTimestep": 1e-8}
    build_and_run_test(TEST_NAME, 1, 1, param_overrides=p)
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, ()) + "/snapshot_*.hdf5"))
    log = open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()

    def load(s):
        with h5py.File(s, "r") as f:
            g = f["PartType4"]; o = np.argsort(g["ParticleIDs"][:])
            return g["ParticleIDs"][:][o], g["Masses"][:][o], g["Velocities"][:][o]
    with h5py.File(f"{TEST_DIR}/ketju_star_collision_graze_ics.hdf5", "r") as f:
        g = f["PartType4"]; i0, m0, v0 = g["ParticleIDs"][:], g["Masses"][:], g["Velocities"][:]
    i1, m1, v1 = load(snaps[-1])
    print("RUMI++ MERGER" in log, len(i0), len(i1))
    assert "RUMI++ MERGER: ID 2" in log and len(i1) == len(i0) - 1 and 1 in i1 and 2 not in i1
    assert abs(m1.sum() / m0.sum() - 1) < 1e-12
    p0 = (m0[:, None] * v0).sum(0); p1 = (m1[:, None] * v1).sum(0)
    assert np.linalg.norm(p1 - p0) / (m0 * np.linalg.norm(v0, axis=1)).sum() < 1e-9
    k = list(i1).index(1); M = m0[0] + m0[1]; vcom = (m0[0] * v0[0] + m0[1] * v0[1]) / M
    assert abs(m1[k] / M - 1) < 1e-12 and np.linalg.norm(v1[k] - vcom) < 0.05 * np.linalg.norm(v0[0] - v0[1])


def test_energy_books():
    """RumiPrintInterval 1: rumi.txt has one row per common step; with no gas the books' residual stays at the integration
    error and no kick work enters; every row comes with a RUMI ENERGY line"""
    _, _, log = _errors("binary_ics", T_BINARY / 10, params={"RumiPrintInterval": 1})
    rows = [l.split() for l in open(variant_output_dir(TEST_NAME, ()) + "/rumi.txt") if not l.startswith("#")]
    steps = int(log.split("RUMI++ totals: ")[1].split()[0])
    resid = max(abs(float(r[5])) for r in rows); kick = max(abs(float(r[7])) for r in rows)
    print(f"rumi.txt: {len(rows)} rows, {steps} steps, max |residual| {resid:.3e}, max |W_kick| {kick:.3e}")
    assert len(rows) == steps and log.count("RUMI ENERGY: t=") == steps
    assert resid < 1e-12 and kick == 0


def test_cpu_row_and_header():
    """cpu.txt has the rumi row; snapshots name the RUMI++ library build in their header"""
    _, _, log = _errors("binary_ics", T_BINARY / 10)
    out = variant_output_dir(TEST_NAME, ())
    rows = [l for l in open(f"{out}/cpu.txt") if l.startswith("rumi ")]
    with h5py.File(sorted(glob.glob(f"{out}/snapshot_*.hdf5"))[-1], "r") as f:
        rid = f["Header"].attrs["RUMI_lib_id"]
    rid = rid.decode() if isinstance(rid, bytes) else str(rid)
    print(f"{len(rows)} rumi rows, header {rid}")
    assert rows and rid.startswith("RUMI_LIB_ID api=3 commit=") and rid in log
