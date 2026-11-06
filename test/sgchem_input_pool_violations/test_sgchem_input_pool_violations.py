"""sgchem_input_pool_violations test: TEST_SGCHEM_PRINT_INPUT_POOL_VIOLATIONS prints, for every cell whose species hold more of an
element than its pool when they are handed to the solver, the cell and the violation (before the clamp). The uniform box of
test/sgchem_box with H2 and CO set by ID % 4 (SGCHEM_KEEP_IC_ABUNDANCES). See README.md."""
import re
import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_input_pool_violations"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))
# x(H2), x(CO) by ID % 4; x(H+) = 0.02. Violation bits: 1 = CO above min(C, O) gas pool (solar ~1.4e-4), 2 = 2 x(H2) + x(H+) above 1
H2 = np.array([0.05, 0.15, 0.7, 0.9])
CO = np.array([1.0e-5, 5.0e-3, 1.0e-5, 1.0e-5])
EXPECTED = {0: 0, 1: 1, 2: 2, 3: 2}


@pytest.fixture(scope="module")
def first_violation():
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    ic = path.join(HERE, f"{NAME}_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        k = f["PartType0/ParticleIDs"][:] % 4
        f["PartType0/SGChemTracAbund"] = np.column_stack([H2[k], np.full(len(k), 0.02), CO[k]])
    build_and_run_test(NAME, 2, 1, param_overrides={"SGChemTablesPath": chemistry_tables_dir()})
    with open(path.join(HERE, f"test_{NAME}.out")) as f:
        text = f.read()
    first = {}
    for pid, viol in re.findall(r"^SGCHEM_INPUT_VIOLATION: ID=(\d+) viol=(\d+)", text, re.M):
        first.setdefault(int(pid), int(viol))
    return first


def test_violating_cells_are_reported(first_violation):
    """bar (written before the run): exactly the cells whose IC violates a pool are reported, each with the bit of its violation
    (CO above its pool: 1; hydrogen above 1: 2); none of the consistent cells"""
    assert len(first_violation) > 100, "the run reports the violating cells"
    assert all(v == EXPECTED[pid % 4] for pid, v in first_violation.items())
    ids = np.array(sorted(first_violation))
    assert set(ids % 4) == {1, 2, 3}
