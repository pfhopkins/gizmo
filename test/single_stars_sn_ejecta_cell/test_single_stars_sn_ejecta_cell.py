"""single_stars_sn_ejecta_cell: a supernova whose kernel holds less gas than f x its ejecta mass (GALSF_SINGLE_STARS_SN_EJECTA_CELL=f, here 1)
is not deposited: its ejecta, explosion energy and yields become one new gas cell at the star, made at the merge/split point, and the
split routine refines it. Production model (SGChem net 5, tables, 2 ranks). A 16^3 lattice of 1 Msun cells at n_H = 10, four stars past
their lifetime at Z = 0.014: 20 Msun (ejecta below the ~32 Msun of its kernel: deposited), 40 Msun (failed, no energy: deposited),
150 and 320 Msun (ejecta above the kernel gas: a new cell). All of it in 100 yr, so cooling and the chemistry change the books by little.

Bars, written before the first run:
  test_new_cells_hold_the_ejecta: new cells are the cells with a child ID (the 1 Msun lattice neither splits nor merges by itself). Their mass
    equals the ejecta of the 150 and 320 Msun stars to 1e-10, their metal mass the ejecta metals to 1e-8, and they exist at all.
  test_books: gas gains all ejecta (1e-12), every star keeps exactly its table remnant, the 320 Msun star leaves none, one event row per
    star; total energy gains the three explosions to 5 %, total momentum changes by less than 5 % of what the ejecta carry.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_sn_ejecta_cell"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_supernovae"))
from make_single_stars_supernovae_ics import make_single_stars_supernovae_ics  # noqa: E402

E_UNIT = 1.989e33 * 1e10  # erg per code energy unit
N = 16


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "run"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    L, _ = make_single_stars_supernovae_ics(str(ic), t["StellarTablesFile"], n=N, n_metals=11)
    with h5py.File(ic, "a") as F:
        F["PartType0/SGChemTracAbund"] = np.tile([0.0, 1.0e-4, 0.0], (N ** 3, 1))
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    T = t["StellarTablesFile"]
    p = p.replace("StellarTablesFile                  SET_BY_TEST", f"StellarTablesFile                  {T}")
    p = p.replace("SGChemTablesPath                   SET_BY_TEST", f"SGChemTablesPath                   {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    return ic, snap, rows


def gas(fname):
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        child = g["ParticleChildIDsNumber"][()] if "ParticleChildIDsNumber" in g else np.zeros(len(g["Masses"]), dtype=int)
        s = F["PartType4"]
        return dict(m=g["Masses"][()], z=g["Metallicity"][()][:, 0], u=g["InternalEnergy"][()], v=g["Velocities"][()], child=child,
                    sm=s["Masses"][()], sv=s["Velocities"][()], sid=s["ParticleIDs"][()])


def energy(a):
    return (a["m"] * a["u"]).sum() + 0.5 * (a["m"] * (a["v"] ** 2).sum(1)).sum() + 0.5 * (a["sm"] * (a["sv"] ** 2).sum(1)).sum()


def momentum(a):
    return (a["m"][:, None] * a["v"]).sum(0) + (a["sm"][:, None] * a["sv"]).sum(0)


def test_new_cells_hold_the_ejecta(run):
    _, snap, rows = run
    a = gas(snap)
    new = a["child"] != 0
    spawned = rows[rows[:, 4] > 100]  # the 150 and 320 Msun stars
    assert new.sum() >= 2
    assert abs(a["m"][new].sum() / spawned[:, 6].sum() - 1) < 1e-10, (a["m"][new].sum(), spawned[:, 6])
    assert abs((a["m"] * a["z"])[new].sum() / spawned[:, 9].sum() - 1) < 1e-8


def test_books(run):
    ic, snap, rows = run
    a, b = gas(ic), gas(snap)
    assert len(rows) == 4 and len(set(rows[:, 1])) == 4
    assert abs(b["m"].sum() - a["m"].sum() - rows[:, 6].sum()) <= 1e-12 * b["m"].sum()
    left = dict(zip(b["sid"].tolist(), b["sm"].tolist()))
    for r in rows:
        if r[7] > 0:
            assert left[int(r[1])] == r[7]
        else:
            assert int(r[1]) not in left
    dE = (energy(b) - energy(a)) * E_UNIT
    assert abs(dE / rows[:, 8].sum() - 1) < 0.05, dE / rows[:, 8].sum()
    p_ej = (rows[:, 6] * np.linalg.norm(a["sv"][0])).sum()
    assert np.abs(momentum(b) - momentum(a)).max() < 0.05 * p_ej
