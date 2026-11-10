"""treecol_nongas test: the TreeCol column density bins (RT_USE_TREECOL_FOR_NH) of a gas cell do not change when a gasless star of
1e-10 of a cell's mass sits among the gas. an earlier walk kept the previous leaf's gas mass for non-gas leaves and added it to
the bins. Read from the TREECOL_DIAG print of the first ten gas cells at step 0 (GIZMO_TREECOL_DIAG); 1 rank. See README.md."""
import re
import shutil
import sys
from os import chdir, getcwd, makedirs, path

import numpy as np

from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "treecol_nongas"
TEST_DIR = f"test/{TEST_NAME}"
HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)
from make_treecol_nongas_ics import make_ics  # noqa: E402


def diag_bins(with_star):
    cwd = getcwd()
    try:
        chdir(TEST_DIR)
        shutil.rmtree("output", ignore_errors=True)
        makedirs("output")
        make_ics(f"{TEST_NAME}_ics.hdf5", with_star)
        run_test(TEST_NAME, 1)
        text = open(f"test_{TEST_NAME}.out").read()
    finally:
        chdir(cwd)
    rows = re.findall(r"TREECOL_DIAG step=0 i=\d+ ID=(\d+) .* bins=\[([^\]]*)\]", text)
    return {int(i): np.array([float(v) for v in b.split(",")]) for i, b in rows}


def test_treecol_nongas():
    """bar (written before the run): ten gas cells reported in both runs, a column of at least 1e-3 in code units in the first, and
    no bin changing by more than 1e-5 of the largest bin of its cell"""
    build_gizmo_for_test(TEST_NAME)
    without, with_star = diag_bins(False), diag_bins(True)
    assert len(without) == 10 and without.keys() == with_star.keys()
    for i in without:
        assert without[i].max() > 1e-3, i
        assert np.abs(with_star[i] - without[i]).max() < 1e-5 * without[i].max(), (i, without[i], with_star[i])
