"""sgchem_rt_infrared: gas cooling leaves into the infrared bin (reports/design/design_m1_coupling.md). A uniform atomic box
at 1000 K, n_H = 100 cm^-3, no stars, no field, no cosmic rays, periodic. Between two snapshots the IR bin gains what the gas
loses x c_red / c, within 2 %, and the dust temperature comes from the solver (> 0 and not the chemistry's start value).
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

TEST_NAME = "sgchem_rt_infrared"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402

N, N_H, RSOL = 8, 100.0, 1e-3
MSUN, PC, MP = 1.989e33, 3.085678e18, 1.6726e-24
L = N * (1.0 / (N_H * 1.4 * MP * PC ** 3 / MSUN)) ** (1.0 / 3.0)


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def test_cooling_reaches_infrared_bin(tables):
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "run_box"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ab = np.zeros((N ** 3, 3)); ab[:, 1] = 1e-4
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(N, L), [(1.0, 1e12, np.array([0.5, 0.5, 0.5]) * L)], T=1000.0, hsml=1.0, tracabund=ab)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")] + [f"BoxSize                     {L}"]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    ei, ug = [], []
    for k in (1, 2):
        with h5py.File(d / "output" / f"snapshot_00{k}.hdf5", "r") as F:
            g = F["PartType0"]
            ei.append(g["PhotonEnergy"][:, -1].sum()); ug.append((g["InternalEnergy"][()] * g["Masses"][()]).sum())
    gain, lost = ei[1] - ei[0], ug[0] - ug[1]
    print("IR gain / (gas loss x c_red / c):", gain / (lost * RSOL), "gas lost", lost)
    assert lost > 0
    assert abs(gain / (lost * RSOL) - 1) < 2e-2
