"""sgchem_rt_start: with the moment solver the first force pass reads the chemistry's start, not a fully neutral cell.

The uniform box of test/sgchem_box (8^3 cells) with IC abundances x(H+) = 0.02, RT_M1 and RT_CHEM_PHOTOION, no stars.
Snapshot 0 is written before any chemistry solve. Its electrons must equal x(H+) plus the C and Si pools, which the
first solve reports: Ne - HII is the same pool in snapshots 0 and 1. Cosmic rays divide by this fraction (ion Alfven speed).
"""

import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables, get_stellar_evolution_tables, problem_output_dir

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_rt_start"
X_HP = 0.02
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))


@pytest.fixture(scope="module")
def out():
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    t = get_stellar_evolution_tables()
    ic = path.join(HERE, "sgchem_rt_start_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        n = len(f["PartType0/ParticleIDs"])
        f["PartType0/SGChemTracAbund"] = np.column_stack([np.zeros(n), np.full(n, X_HP), np.zeros(n)])
    build_and_run_test(NAME, 2, 1, param_overrides={"SGChemTablesPath": chemistry_tables_dir(), "StellarTablesFile": t["StellarTablesFile"]})
    return problem_output_dir(NAME, None, ())


def read(out, snap):
    with h5py.File(path.join(out, f"snapshot_{snap:03d}.hdf5")) as f:
        return f["PartType0/ElectronAbundance"][:], f["PartType0/HII"][:], f["PartType0/SGChemTracAbund"][:, 1]


def test_start_is_the_chemistry_state(out):
    ne0, hii0, xhp0 = read(out, 0)
    assert np.all(xhp0 == X_HP), "snapshot 0 holds the IC abundance"
    assert np.all(hii0 == X_HP), "HII at the start is the chemistry's x(H+)"
    assert np.all(ne0 > X_HP), "electrons at the start: x(H+) and the metal pools"


def test_start_electrons_match_the_first_solve(out):
    ne0, hii0, _ = read(out, 0)
    ne1, hii1, _ = read(out, 1)
    pool0, pool1 = ne0 - hii0, ne1 - hii1
    print(f"C + Si electrons per H: start {pool0.mean():.6e}, after the first solves {pool1.mean():.6e}")
    assert np.all(np.abs(pool0 / pool1 - 1) < 1e-9), "the same pool before and after the first solve"
