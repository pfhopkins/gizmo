"""stella_merge_kinetic_veto test: in Stella builds no gas merge may turn most of the energy involved into heat
(MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION on for every gas merge, opt-out MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION), and a merge
returns the pair's centre-of-mass kinetic energy as heat. Stella cloud production flags, 16^3 lattice of 1 Msun cells at
rest with two 0.2 Msun cells (merge threshold 0.49 Msun): one at rest, one at 10 km/s. See README.md."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "stella_merge_kinetic_veto"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_invariant_checks"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402
from test_single_stars_invariant_checks import solar27  # noqa: E402

N = 16
SLOW, FAST = 1 + 4 * 256 + 4 * 16 + 4, 1 + 11 * 256 + 11 * 16 + 11  # IDs = lattice index + 1, far apart
M_LIGHT, V_FAST = 0.2, 10.0  # Msun, km/s
U51 = 1.0e51 / (1.989e33 * 1.0e10)  # 1e51 erg in code units (Msun (km/s)^2)


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def run(tag, tables, flags=(), v_fast=V_FAST, build=True):
    if build:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    T = tables["StellarTablesFile"]
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_single_stars_metals_ics(str(ic), solar27(T), n=N)
    with h5py.File(ic, "r+") as F:
        ids = F["PartType0/ParticleIDs"][()]
        m = F["PartType0/Masses"][()]; v = F["PartType0/Velocities"][()]
        m[ids == SLOW] = M_LIGHT; m[ids == FAST] = M_LIGHT; v[ids == FAST] = [v_fast, 0, 0]
        F["PartType0/Masses"][...] = m; F["PartType0/Velocities"][...] = v
        F["Header"].attrs["Minimum_Mass_For_Cell_Merge"] = 0.49
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = re.sub(r"^BoxSize .*$", f"BoxSize                     {L}", p, flags=re.M)
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as f:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=f,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]
    with h5py.File(snap, "r") as F:
        ids = set(F["PartType0/ParticleIDs"][()].tolist())
    names = open(d / "output" / "energy_ledger.txt").readline().split()[1:]
    led = np.loadtxt(d / "output" / "energy_ledger.txt", ndmin=2)
    first = led[0]  # first full-sync report, after the first merge pass
    return ids, first[0], (first[names.index("E_kin")] + first[names.index("E_th")]) * U51


def ke_com():
    mu = M_LIGHT * 1.0 / (M_LIGHT + 1.0)
    return 0.5 * mu * V_FAST ** 2


def test_veto_refuses_kinetic_merges(tables):
    """bar (written before the run): the cell at rest merges; the 10 km/s cell (KE_com / (KE_com + E_th,light) ~ 1, above 0.5)
    is still there after 1e-4 time units"""
    ids, _, _ = run("veto", tables)
    print(f"slow cell present: {SLOW in ids}, fast cell present: {FAST in ids}")
    assert SLOW not in ids
    assert FAST in ids


def test_opt_out_merges_and_returns_kinetic_energy_as_heat(tables):
    """bar (written before the run): with MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION both light cells merge; against the same run
    with the second light cell at rest, the gas energy (E_kin + E_th) at the first report is higher by that cell's initial
    kinetic energy (1/2) 0.2 Msun (10 km/s)^2 = 10, to 5% of KE_com = 8.33 (a merge dropping KE_com would leave 1.67)"""
    ids, t, e = run("allow", tables, ("MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION",))
    ids0, t0, e0 = run("allow_at_rest", tables, ("MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION",), v_fast=0.0, build=False)
    ke_fast = 0.5 * M_LIGHT * V_FAST ** 2
    print(f"slow cell present: {SLOW in ids}, fast cell present: {FAST in ids}; E(t={t:g}) {e:.6f} vs at rest (t={t0:g}) {e0:.6f}: "
          f"difference {e - e0:.4f}, expected {ke_fast:.4f}, KE_com {ke_com():.4f}")
    assert SLOW not in ids and FAST not in ids and FAST not in ids0
    assert t == t0
    assert abs((e - e0) - ke_fast) < 0.05 * ke_com()
