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


