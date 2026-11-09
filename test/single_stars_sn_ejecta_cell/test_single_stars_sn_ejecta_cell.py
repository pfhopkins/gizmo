"""single_stars_sn_ejecta_cell: a supernova whose kernel holds less gas than f x its ejecta mass (GALSF_SINGLE_STARS_SN_EJECTA_CELL=f, here 5)
is not deposited: its ejecta, explosion energy and yields become one new gas cell at the star, made at the merge/split point, and the
split routine refines it. Production model (SGChem net 5, tables, 2 ranks). A 16^3 lattice of 1 Msun cells at n_H = 10, three stars past
their lifetime at Z = 0.014, 100 yr. The kernels hold 72 to 84 Msun (printed at the decision, h = 3.5 to 3.8 pc): 12 Msun (10.7 Msun ejecta,
threshold 53 Msun: deposited), 20 Msun (18.5 Msun ejecta, threshold 92 Msun: a new cell), 320 Msun (a new cell, no remnant).

Bars:
  test_new_cells_hold_the_ejecta: new cells are the cells with a child ID whose ID is one of the IC cells within 2.5 cells of the 20 and 320 Msun
    stars (a new cell takes the ID of the nearest local cell). Their mass equals the ejecta of those stars to 1e-10, their metal mass the ejecta
    metals to 1e-8, and they exist at all.
  test_books: gas gains all ejecta (1e-12), every star keeps exactly its table remnant, the 320 Msun star leaves none, one event row per
    star; total momentum changes by less than 5 % of what the ejecta carry.
  test_energy: the gas around the 12 and 20 Msun stars gains their two explosions to 5 %.
Moved after the first runs, with these reasons:
  - new cells: the first version took every cell with a child ID, assuming the lattice does not split, but the deposit splits lattice cells (the
    unchanged code gave 440.7 Msun against 432.1). With the IDs around the stars the unchanged code fails at 412.8 against 432.1: the 320 Msun
    deposit splits cells there too, which a new cell does not do.
  - energy: the first version had 20, 40, 150, 320 Msun at f = 1 and asked the total for 3e51 erg to 5 %, on the claim that 100 yr of cooling changes
    the books by little. That claim is false for the heavy ejecta cells: 320 Msun at Z = 0.96 keeps 2e48 of its 1e51 erg, 150 Msun keeps 6.2e50, the
    20 Msun deposit 9.9e50. The energy bar now sits on the ejecta mass the option is meant for, and the 320 Msun star only on mass and metals.
  - f: 2 took the kernel gas as 32 Msun (DesNumNgb), but the kernels hold 72 to 84 Msun, so the 20 Msun star deposited; test_new_cells_hold_the_ejecta
    caught it (330.4 against 338.5 Msun), and test_energy had passed on two deposits. 5 clears the printed kernel gas on both sides.
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
STARS = [(20.0, (0.25, 0.25, 0.25)), (12.0, (0.75, 0.25, 0.75)), (320.0, (0.75, 0.75, 0.25))]


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
    L, _ = make_single_stars_supernovae_ics(str(ic), t["StellarTablesFile"], n=N, n_metals=11, stars=STARS)
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
        return dict(m=g["Masses"][()], z=g["Metallicity"][()].reshape(len(g["Masses"]), -1)[:, 0], u=g["InternalEnergy"][()], v=g["Velocities"][()], child=child, id=g["ParticleIDs"][()],
                    x=g["Coordinates"][()], sm=s["Masses"][()], sv=s["Velocities"][()], sid=s["ParticleIDs"][()], sx=s["Coordinates"][()], L=F["Header"].attrs["BoxSize"])


def energy(a):
    return (a["m"] * a["u"]).sum() + 0.5 * (a["m"] * (a["v"] ** 2).sum(1)).sum() + 0.5 * (a["sm"] * (a["sv"] ** 2).sum(1)).sum()


def momentum(a):
    return (a["m"][:, None] * a["v"]).sum(0) + (a["sm"][:, None] * a["sv"]).sum(0)


def test_new_cells_hold_the_ejecta(run):
    ic, snap, rows = run
    a, g0 = gas(snap), gas(ic)
    spawned = rows[rows[:, 4] != 12]  # the 20 and 320 Msun stars
    near = []
    for sid in spawned[:, 1].astype(int):
        d = g0["x"] - g0["sx"][g0["sid"] == sid]
        d -= g0["L"] * np.round(d / g0["L"])
        near.append(g0["id"][(d ** 2).sum(1) < (2.5 * g0["L"] / N) ** 2])
    new = (a["child"] != 0) & np.isin(a["id"], np.concatenate(near))
    assert new.sum() >= 2
    assert abs(a["m"][new].sum() / spawned[:, 6].sum() - 1) < 1e-10, (a["m"][new].sum(), spawned[:, 6])
    assert abs((a["m"] * a["z"])[new].sum() / spawned[:, 9].sum() - 1) < 1e-8


def test_books(run):
    ic, snap, rows = run
    a, b = gas(ic), gas(snap)
    assert len(rows) == 3 and len(set(rows[:, 1])) == 3
    assert abs(b["m"].sum() - a["m"].sum() - rows[:, 6].sum()) <= 1e-12 * b["m"].sum()
    left = dict(zip(b["sid"].tolist(), b["sm"].tolist()))
    for r in rows:
        if r[7] > 0:
            assert left[int(r[1])] == r[7]
        else:
            assert int(r[1]) not in left
    p_ej = (rows[:, 6] * np.linalg.norm(a["sv"][0])).sum()
    assert np.abs(momentum(b) - momentum(a)).max() < 0.05 * p_ej


def test_energy(run):
    ic, snap, _ = run
    a, b = gas(ic), gas(snap)
    d = b["x"][None, :, :] - a["sx"][:, None, :]
    d -= a["L"] * np.round(d / a["L"])
    own = (d ** 2).sum(2).argmin(0)
    near = (own == 0) | (own == 1)  # the 20 and 12 Msun stars
    dE = ((b["m"] * (b["u"] - a["u"][0]))[near].sum() + 0.5 * (b["m"] * (b["v"] ** 2).sum(1))[near].sum()) * E_UNIT
    assert abs(dE / 2.0e51 - 1) < 0.05, dE / 2.0e51
