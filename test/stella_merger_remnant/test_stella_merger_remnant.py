"""KETJU/RUMI++ star merger with a heavier compact remnant as survivor (audit K4), on Config_stella_default.sh + KETJU star mergers.

MSTAR keeps the heavier body (ketju-integrator mergers.cpp); with an 8 Msun remnant (sampled, MstarSampleIMF[0] = 0, the state
resolvedism_fb.cc leaves after a star's death) and a 3 Msun live star the remnant survives. It must stay a remnant: mass the sum
(11 Msun, mass conserved exactly), metals mass-weighted, MstarSampleIMF[0] = 0 so no feedback treats it as a star; the victim is
emptied. Before the fix the survivor took the victim's ZAMS mass (MstarSampleIMF[0] = 3): a live 3 Msun star of the remnant's
age holding 11 Msun, whose later winds/SN would return the remnant's mass to the gas. Bars: exact (mass 1e-12, metals 1e-6
for float storage)."""

import os
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver

TEST_NAME = "stella_merger_remnant"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", f"{TEST_NAME}_driver")


def test_stella_merger_remnant():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, f"{TEST_NAME}_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))
    subprocess.run([sys.executable, join(REPO, "test", "sgchem_cooling_path", "make_sgchem_cooling_path_ics.py"), join(run, f"{TEST_NAME}_ics.hdf5")], check=True)
    r = subprocess.run([DRIVER, join(TEST_DIR, f"{TEST_NAME}.params")], capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", "checks.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    lines = r.stdout.splitlines()
    assert [l for l in lines if l.startswith("DONE")], "driver did not finish:\n" + (r.stdout + r.stderr)[-3000:]
    failed = [l for l in lines if l.startswith("FAIL")]
    print("\n".join(l for l in lines if l.startswith(("PASS", "FAIL"))))
    assert not failed, "\n".join(failed)
