"""Snapshot states of KETJU region members part-way through their step.

An isolated binary in a region whose steps (1e-5) span ~40 orbits, written every 3.3e-6, i.e. mostly inside steps. Each
snapshot's relative separation must match the exact two-body orbit from the IC state (double, read exactly). Before
the fix mid-step members were written on the drift chord, O(1) off; now 5.6e-5 (MSTAR) and 2.2e-5 (Kepler).
Variant: KetjuKeplerPairs = 1 advances the pair by the exact Kepler drift instead of MSTAR, to the same accuracy."""

import glob
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gizmo.test import build_and_run_test, clean_test_outputs

TEST_NAME = "ketju_snapshot_midstep"
TEST_DIR = f"test/{TEST_NAME}"
G = 4.300710573e-6


@pytest.mark.parametrize("kepler", [0, 1])
def test_ketju_snapshot_midstep(kepler):
    subprocess.run([sys.executable, "make_ketju_snapshot_midstep_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1, param_overrides={"KetjuKeplerPairs": kepler})
    with h5py.File(f"{TEST_DIR}/ketju_snapshot_midstep_ics.hdf5", "r") as f:
        g = f["PartType4"]; x = np.array(g["Coordinates"], dtype=float); v = np.array(g["Velocities"], dtype=float)
        m = np.array(g["Masses"], dtype=float) * 1e10
    y0 = np.concatenate([x[1] - x[0], v[1] - v[0]]); mu = G * (m[0] + m[1])
    rhs = lambda t, y: np.concatenate([y[3:], -mu * y[:3] / np.linalg.norm(y[:3]) ** 3])
    worst = 0
    for s in sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[1:]:
        with h5py.File(s, "r") as f:
            t = f["Header"].attrs["Time"]; g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
            p = np.array(g["Coordinates"])[o]
        y = solve_ivp(rhs, (0, t), y0, method="DOP853", rtol=1e-13, atol=1e-22).y[:3, -1]
        worst = max(worst, np.linalg.norm(p[1] - p[0] - y) / np.linalg.norm(y))
    print(f"KetjuKeplerPairs={kepler}: worst snapshot |dr|/r = {worst:.2e}")
    assert worst < 2e-3
