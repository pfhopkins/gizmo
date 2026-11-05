"""sgchem_rt_photons: what the moment solver absorbs in the ionising bins is what the chemistry consumes
(reports/design/design_m1_coupling.md). A 40 Msun star in a uniform box at n_H = 100 cm^-3, 0.25 Msun cells, four ionising
bins, RSOL 0.1 so light outruns the front, over 0.3 recombination times. Rate of new ionisations between the second and
last snapshots, plus recombinations (alpha_B n_e n_p), against the table's photon rate.
  test_atomic: atomic gas, (d N_H+ + N_rec) / (Q dt) within 0.8-1.2.
  test_molecular: all H in H2: H+ forms at more than half the photon rate and the front stays inside the box. The star's
    Lyman-Werner light also dissociates H2 here; test sgchem_rt_h2_photons has the ledger on H2 with ionising light alone.
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

TEST_NAME = "sgchem_rt_photons"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

N, N_H, M_CELL, M_STAR = 24, 100.0, 0.25, 40.0
MSUN, PC, MP = 1.989e33, 3.085678e18, 1.6726e-24
T_UNIT = PC / 1e5
RHO = N_H * 1.4 * MP * PC ** 3 / MSUN
L = N * (M_CELL / RHO) ** (1.0 / 3.0)
ALPHA_B = 2.6e-13
BUILD = []


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    if not BUILD:
        build_gizmo_for_test(TEST_NAME)
        BUILD.append(1)
    return t


def run(tag, tables, x_h2):
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M_STAR), np.log10(0.0142)
    tms = 0.3 * tab._i2(tab.life, lM, lZ)
    age = tab._i2(tab.t_pms, lM, lZ) + tms
    with h5py.File(tables["StellarTablesFile"], "r") as F:
        Q = 10 ** tab._i3(F["log_Q_ion"][()].astype(np.float32).astype(np.float64), lM, lZ, np.log10(tms))
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ab = np.zeros((N ** 3, 3)); ab[:, 0] = x_h2; ab[:, 1] = 1e-4
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(N, L), [(M_STAR, age, np.array([0.51, 0.51, 0.51]) * L)],
                            gas_mass=M_CELL, hsml=1.0, tracabund=ab)
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
        subprocess.run(mpi_launch_command(4) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    n_cell = M_CELL * MSUN / (1.4 * MP)
    rows = []
    for k in range(1, 5):
        with h5py.File(d / "output" / f"snapshot_00{k}.hdf5", "r") as F:
            ab = F["PartType0/SGChemTracAbund"][()]
            t = F["Header"].attrs["Time"] * T_UNIT
            r = np.linalg.norm(F["PartType0/Coordinates"][()] - F["PartType4/Coordinates"][0], axis=1)
        rows.append((t, ab[:, 1].sum() * n_cell, ab[:, 0].sum() * n_cell, ALPHA_B * N_H * (ab[:, 1] ** 2).sum() * n_cell, r[ab[:, 1] > 0.5].max(initial=0) / L))
    return np.array(rows), Q


def ledger(rows, Q):
    t, nion, nh2, rec, _ = rows.T
    n_rec = np.trapezoid(rec[1:], t[1:])  # snapshots 2-4
    return (nion[-1] - nion[1], nh2[1] - nh2[-1], n_rec, Q * (t[-1] - t[1]))


def test_atomic(tables):
    rows, Q = run("atomic", tables, 0.0)
    a, _, r, q = ledger(rows, Q)
    print("atomic new H+, recombined, photons:", a, r, q, (a + r) / q, rows[:, 4])
    assert 0.8 < (a + r) / q < 1.2
    assert rows[-1, 4] < 0.45


def test_molecular(tables):
    rows, Q = run("molecular", tables, 0.5)
    a, h2, r, q = ledger(rows, Q)
    print("molecular new H+, H2 lost, recombined, photons:", a, h2, r, q, (a + r) / q, rows[:, 4])
    assert (a + r) / q > 0.5
    assert rows[-1, 4] < 0.45
