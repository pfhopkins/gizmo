"""sgchem_threaded_cooling_crosscheck test: with TEST_COOLING_COMPARE_THREADED_PASS_WITH_SERIAL_PASS every chemistry step first
runs the cells serially on copies, then threaded, and compares the cells byte for byte (the run stops on a difference). The uniform
box of test/sgchem_box with molecular ICs, 2 ranks x 2 threads. See README.md."""
import re
import sys
from os import environ, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, get_cooling_tables

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_threaded_cooling_crosscheck"
sys.path.insert(0, path.join(ROOT, "test", "sgchem_box"))


def test_threaded_pass_equals_serial_pass():
    """bar (written before the run): the run reaches TimeMax and reports, in at least 4 steps, that the threaded pass equals the
    serial pass for more than 100 cells in total"""
    from test_sgchem_box import chemistry_tables_dir
    from make_sgchem_box_ics import make_ics
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    ic = path.join(HERE, f"{NAME}_ics.hdf5")
    make_ics(ic)
    with h5py.File(ic, "r+") as f:
        k = f["PartType0/ParticleIDs"][:] % 4
        f["PartType0/SGChemTracAbund"] = np.column_stack([0.05 + 0.1 * k, np.full(len(k), 0.02), np.full(len(k), 1.0e-5)])
    build_and_run_test(NAME, 2, 2, param_overrides={"SGChemTablesPath": chemistry_tables_dir()})
    with open(path.join(HERE, f"test_{NAME}.out")) as f:
        rows = re.findall(r"^COOLING_CROSSCHECK: (\d+) cells", f.read(), re.M)
    assert len(rows) >= 4 and sum(int(n) for n in rows) > 100
