"""Cloud cosmic-ray ionisation floor on the production cloud configuration (Config_stella_cloud.sh, stella_cloud.params; audit row 18).

CR_BACKGROUND_IONRATE_ATTENUATED scales CosmicRayIonRate by STARFORGE's column attenuation: 1 below Sigma_0 = 1e21 m_H cm^-2
(2.23e-3 g cm^-2), else exp(-Sigma/100 g cm^-2) Sigma_0/Sigma, pixel mean. Upstream's Get_CosmicRayIonizationRate_cgs adds the
radioactive ionisation that no column shields: 1e-21 Z/Zsun (K-40 and other long-lived species) plus 1e-19 Fe/Fe_sun (short-lived
isotopes, Adams+2014). At Sigma = 1e3 g cm^-2 in every pixel the attenuated rate is 3e-17 x e^-10 x 2.2e-6 ~ 3e-27 s^-1.
Bar, set before running: the chemistry's zeta for that cell >= 1e-21 Z/Zsun. Driver: test/tallbox_gravity_cr/zeta_gravity_driver.cc
(startup, one gravity pass, then chem_fill_inputs for gas cell 0 with its TreeCol gas columns set)."""

import os
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver

TEST_NAME = "sgchem_cr_floor"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", "zeta_gravity_driver")


def test_sgchem_cr_floor():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(REPO, "test", "tallbox_gravity_cr", "zeta_gravity_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_"); os.makedirs(join(run, "output")); os.symlink(REPO, join(run, "gizmo"))
    subprocess.run([sys.executable, join(REPO, "test", "sgchem_cooling_path", "make_sgchem_cooling_path_ics.py"), join(run, f"{TEST_NAME}_ics.hdf5")], check=True)
    r = subprocess.run([DRIVER, join(TEST_DIR, f"{TEST_NAME}.params"), "1e3"], capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", "driver.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    shutil.rmtree(run, ignore_errors=True)
    z = [l.split() for l in r.stdout.splitlines() if l.startswith("ZETA")]
    assert z, "driver did not finish:\n" + (r.stdout + r.stderr)[-3000:]
    zeta, zrel = float(z[0][1]), float(z[0][2])
    print(f"zeta at Sigma = 1e3 g/cm^2: {zeta:.3e} (floor 1e-21 Z/Zsun = {1e-21 * zrel:.3e})")
    assert zeta >= 1e-21 * zrel
