"""Driven isothermal turbulence held at Mach 10 by the TURB_DRIVING_TARGET_SIGMA control loop.

32^3 periodic box, 10 pc, c_s = 0.2 km/s, solenoidal driving with a parabolic spectrum on
1 <= kL/2pi <= 3, hydro and MHD (uniform B_z, initial plasma beta 1), each with the driving force
recomputed every step or only on driving updates (TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE, "lazy").
Starting from rest, the loop sets the gain on the driving force from e = ln(v_target / v_rms). The test
measures the Mach number from the snapshots -- independently of the loop's own measurement -- and
checks that it settles on the target and stays there. See turb_driving_common.py.
"""

import subprocess
import sys
from os import path

import pytest

import turb_driving_common as common

TEST_NAME = "turbulent_box"
TEST_DIR = f"test/{TEST_NAME}"


def _ensure_ic():
    if path.exists(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"):
        return
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)


@pytest.mark.parametrize("lazy", (False, True), ids=("full", "lazy"))
@pytest.mark.parametrize("variant", common.VARIANTS)
def test_turbulent_box(variant, lazy):
    _ensure_ic()
    common.run_and_check(TEST_NAME, variant, lazy=lazy)
