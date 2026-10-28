"""Turbsphere: a uniform cloud in a 10 R periodic box, held by the STARFORGE_GMC_TURBINIT potential and
stirred to Mach 10 inside 2R by the TURB_DRIVING_TARGET_SIGMA control loop in its aperture mode, which
measures only the gas within 2R of the box centre. Hydro and MHD (uniform B_z, plasma beta 1 in the
cloud), with the lazy driving force only (TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE in Config.sh), since
that is how turbsphere stirring runs are made. Same checks as turbulent_box, with the Mach number
measured over the same aperture.
"""

import subprocess
import sys
from os import path
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "turbulent_box"))
import turb_driving_common as common  # noqa: E402

TEST_NAME = "turbsphere"
TEST_DIR = f"test/{TEST_NAME}"
RADIUS = 5.0  # nominal cloud radius of the ICs
APERTURE = 2 * RADIUS  # TURB_DRIVING_TARGET_SIGMA=2 in Config.sh, in nominal cloud radii


def _ensure_ic():
    if path.exists(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"):
        return
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)


@pytest.mark.parametrize("variant", common.VARIANTS)
def test_turbsphere(variant):
    _ensure_ic()
    common.run_and_check(TEST_NAME, variant, aperture=APERTURE)
