"""Two Type 5 sinks in KETJU (the STARFORGE path): e = 0.9 binary, a few orbits, against the Kepler orbit."""

import glob
import subprocess
import sys

import h5py
import numpy as np

from gizmo.test import build_and_run_test

TEST_NAME = "ketju_sinks"
TEST_DIR = f"test/{TEST_NAME}"
G = 4.300710573e-3


def test_sink_binary():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    build_and_run_test(TEST_NAME, 1, 1)
    assert "Found 1 region(s) from 2 center(s)" in open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()  # the sinks are in a chain
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType5"]; m, x, v = g["Masses"][:], g["HermiteSyncCoordinates"][:], g["HermiteSyncVelocities"][:]
    dr, dv = x[0] - x[1], v[0] - v[1]; a = 1 / (2 / np.linalg.norm(dr) - dv @ dv / (G * m.sum()))
    assert abs(a * 206264.806 / 1000 - 1) < 1e-3


def test_pn_binary():
    """two 1e6 Msun PN sinks, circular, a = 1e-5 pc: the 2.5PN term alone: the orbit shrinks at the Peters rate"""
    import make_pn_ics  # noqa: F401
    a0, m = 1e-5, 1e6; c = 299792.458
    build_and_run_test(TEST_NAME, 1, 1, extra_config_flags=("KETJU_PN_COMPACT_OBJECTS",), param_overrides={"InitCondFile": "ketju_pn_ics", "TimeMax": 3e-7, "TimeBetSnapshot": 3e-7, "TimeBetStatistics": 3e-7, "MaxSizeTimestep": 1e-9, "KetjuPNTerms": "5", "KetjuRegionRadius": 1e-4, "Softening_Type5": 1e-8, "Softening_Type5_MaxPhysLimit": 1e-8, "MinSizeTimestep": 1e-18})
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output_KETJU_PN_COMPACT_OBJECTS/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType5"]; x, v = g["HermiteSyncCoordinates"][:], g["HermiteSyncVelocities"][:]
    dr, dv = x[0] - x[1], v[0] - v[1]; a = 1 / (2 / np.linalg.norm(dr) - dv @ dv / (G * 2 * m))
    peters = (a0 ** 4 - 256 / 5 * G ** 3 * m * m * 2 * m / c ** 5 * 3e-7) ** 0.25
    assert abs((a0 - a) / (a0 - peters) - 1) < 0.2
