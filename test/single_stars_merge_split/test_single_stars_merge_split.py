"""single_stars_merge_split: cell merges and the split threshold under single stars, chemistry on. A uniform 12^3 lattice
of 1 Msun cells at n_H ~ 100 cm^-3, a sparse sublattice of 0.3 Msun cells carrying their own H2, H+, CO abundances and
dust temperature. The minimum merge mass is fixed at 0.5 Msun (MERGESPLIT_HARDCODE_MIN_MASS), the run takes a few steps of
1e-10 code time, so the network changes the species totals by less than 1e-6.
  test_chemistry_state_conserved: merges remove light cells; total m x of H2, H+, CO matches the IC to 1e-6 (the survivor
    takes the mass-weighted abundances, stella_dev merge_split.cc).
  test_split_threshold: an IC without a saved threshold gets MaxMassForParticleSplit = 2.01 x the largest IC cell (stella_dev).
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

TEST_NAME = "single_stars_merge_split"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402

N_H, N, MSUN, PC, MP = 100.0, 12, 1.989e33, 3.085678e18, 1.6726e-24
M_LIGHT, X_LIGHT, X_HEAVY = 0.3, (0.4, 0.02, 1.0e-5), (0.0, 1.0e-4, 0.0)
EVERY = 37  # one light cell in EVERY


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    rho = N_H * 1.4 * MP * PC ** 3 / MSUN
    dx = (1.0 / rho) ** (1.0 / 3.0); L = N * dx
    d = TEST_DIR / "run"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    pos = lattice(N, L)
    light = (np.arange(len(pos)) % EVERY) == 5
    x = np.array([X_LIGHT if f else X_HEAVY for f in light])
    make_treecol_pixels_ics(str(ic), L, pos, [(5.0, 1e6, (0.01 * dx, 0.01 * dx, 0.01 * dx))], hsml=2 * dx, tracabund=x)
    with h5py.File(ic, "a") as F:
        F["PartType0/Masses"][...] = np.where(light, M_LIGHT, 1.0)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    T = t["StellarTablesFile"]
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    p = "\n".join(l for l in p.splitlines() if not (l.split() and l.split()[0] == "BoxSize")) + f"\nBoxSize {L}\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return d


def snapshot(d, k):
    fn = sorted((d / "output").glob("snapshot_*.hdf5"))[k]
    with h5py.File(fn) as F:
        g = F["PartType0"]
        return dict(F["Header"].attrs), g["Masses"][:], g["SGChemTracAbund"][:]


def test_chemistry_state_conserved(run):
    _, m0, x0 = snapshot(run, 0)
    _, m1, x1 = snapshot(run, -1)
    print(f"cells {len(m0)} -> {len(m1)}, mass {m0.sum()} -> {m1.sum()}")
    assert len(m1) < len(m0) - 20 and abs(m1.sum() / m0.sum() - 1) < 1e-10, "light cells merged, mass conserved"
    for k, name in enumerate(("H2", "H+", "CO")):
        a, b = (m0 * x0[:, k]).sum(), (m1 * x1[:, k]).sum()
        print(f"{name}: total m x {a:.8g} -> {b:.8g}")
        assert abs(b / a - 1) < 1e-6, name


def test_split_threshold(run):
    h, _, _ = snapshot(run, -1)
    print(f"split threshold {h['Maximum_Mass_For_Cell_Split']}")
    assert abs(h["Maximum_Mass_For_Cell_Split"] / 2.01 - 1) < 1e-6
