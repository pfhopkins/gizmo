"""TreeCol columns must not change when a gasless star sits among the gas: force_treeevaluate kept the previous
leaf's gas mass for non-gas leaves and added it to the column bins."""

import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname, join

import numpy as np
import pytest

sys.path.insert(0, dirname(abspath(__file__)))
from gizmo.test import build_gizmo_for_test, link_test_driver
from make_treecol_nongas_ics import make_ics

TEST_NAME = "treecol_nongas"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))


def columns(with_star):
    make_ics(join(TEST_DIR, f"{TEST_NAME}_ics.hdf5"), with_star)
    shutil.rmtree(join(TEST_DIR, "output", "run"), ignore_errors=True)
    os.makedirs(join(TEST_DIR, "output", "run"))
    r = subprocess.run([join(TEST_DIR, "output", "treecol_nongas_driver"), f"{TEST_NAME}.params"], capture_output=True, text=True, cwd=TEST_DIR)
    cols = {(int(l.split()[1]), int(l.split()[2])): float(l.split()[3]) for l in r.stdout.splitlines() if l.startswith("COL ")}
    assert cols, "driver printed no columns (exit %d):\n%s" % (r.returncode, (r.stdout + r.stderr)[-3000:])
    return cols


def test_treecol_nongas():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, "treecol_nongas_driver.cc"), join(TEST_DIR, "output", "treecol_nongas_driver")))
    finally:
        os.chdir(cwd)
    without, with_star = columns(False), columns(True)
    assert without.keys() == with_star.keys()
    a, b = np.array([without[k] for k in sorted(without)]), np.array([with_star[k] for k in sorted(without)])
    assert a.max() > 0, "no column density at all"
    rel = np.abs(b - a) / a.max()
    assert rel.max() < 1e-6, f"a gasless star changed a gas cell's column by {rel.max():.2e} of the largest column"
