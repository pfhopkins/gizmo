"""FUV floor of the G0_VARIABLE paths with the ISRF parameter switched off (WLM showcase: G0 = 0, CosmicRayIonRate = 3e-18),
on the production configuration (Config_stella_default.sh), one gas cell without stars or stellar flux.

The floor is the z = 0 UV background value 0.324e-2 Habing. It was written 0.324e-2 x All.G0 at every site, so G0 = 0 removed it and
the cell saw G0 = 0. Now G0_FLOOR_BACKGROUND = max(0.324e-2 x All.G0, 0.324e-2): unchanged for All.G0 >= 1 (production 1.7),
never below the z = 0 value. The default configuration has no CR_SCALE_WITH_G0, so zeta is the configured rate.
Bars (1e-12 relative): G0 = 0.324e-2 and zeta = CosmicRayIonRate = 3e-18 s^-1.
Driver: test/tallbox_gravity_cr/zeta_gravity_driver.cc."""

import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver

TEST_NAME = "sgchem_g0_floor"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", "zeta_gravity_driver")


def test_sgchem_g0_floor():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(REPO, "test", "tallbox_gravity_cr", "zeta_gravity_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_"); os.makedirs(join(run, "output")); os.symlink(REPO, join(run, "gizmo"))
    subprocess.run([sys.executable, join(REPO, "test", "sgchem_cooling_path", "make_sgchem_cooling_path_ics.py"), join(run, f"{TEST_NAME}_ics.hdf5")], check=True)
    r = subprocess.run([DRIVER, join(TEST_DIR, f"{TEST_NAME}.params")], capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", "driver.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    shutil.rmtree(run, ignore_errors=True)
    z = [l.split() for l in r.stdout.splitlines() if l.startswith("ZETA")]
    assert z, "driver did not finish:\n" + (r.stdout + r.stderr)[-3000:]
    zeta, g0, rate = float(z[0][1]), float(z[0][3]), float(z[0][4])
    zeta_ref = rate
    print(f"G0 = {g0:.6e} (floor 3.24e-3), zeta = {zeta:.6e} (expected {zeta_ref:.6e}), CosmicRayIonRate {rate:.1e}")
    assert rate == 3e-18
    assert abs(g0 / 0.324e-2 - 1) < 1e-12
    assert abs(zeta / zeta_ref - 1) < 1e-12
