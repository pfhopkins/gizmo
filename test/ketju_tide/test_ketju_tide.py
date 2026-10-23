"""KETJU external tide on a region: a 200 AU pair (P ~ 2000 yr) tidally perturbed by a 20 Msun star at 5000 AU for
0.2 Myr, against a DOP853 three-body integration from the IC. The region's steps span a good
fraction of the pair's orbit, so the external field reaches the pair as kicks at random phases. Measured |da/a|:
default 7.4e-4 (907 steps); MaxSizeTimestep 1e-8: 1.9e-6.
- KetjuTidalStepRes 0.1 (step floor 0.1 P (1-e)^1.5 above the energy budget): 1.7e-5, 4166 steps.
- KetjuSmoothTide 1 (linear tide inside MSTAR, host kicks the CoM): 1.0e-4, 851 steps."""

import glob
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gizmo.test import build_and_run_test, clean_test_outputs

TEST_NAME = "ketju_tide"
TEST_DIR = f"test/{TEST_NAME}"
G = 4.300710573e-6


def _elements(m, x, v):
    d = x[1] - x[0]; u = v[1] - v[0]; mu = G * (m[0] + m[1]); E = 0.5 * u @ u - mu / np.linalg.norm(d)
    h = np.cross(d, u); return -mu / (2 * E), np.sqrt(max(0.0, 1 + 2 * E * (h @ h) / mu ** 2))


@pytest.mark.parametrize("overrides,da_max", [
    ({"KetjuTidalStepRes": 0.1, "KetjuTidalErrorBudget": 1e-6}, 1e-4),
    ({"KetjuSmoothTide": 1}, 3e-4),
])
def test_ketju_tide(overrides, da_max):
    subprocess.run([sys.executable, "make_ketju_tide_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1, param_overrides=overrides)
    with h5py.File(f"{TEST_DIR}/ketju_tide_ics.hdf5", "r") as f:
        g = f["PartType4"]; m = np.array(g["Masses"], dtype=float) * 1e10
        x0 = np.array(g["Coordinates"], dtype=float); v0 = np.array(g["Velocities"], dtype=float)

    def rhs(t, y):
        p = y[:9].reshape(3, 3); a = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i != j:
                    d = p[j] - p[i]; a[i] += G * m[j] * d / np.linalg.norm(d) ** 3
        return np.concatenate([y[9:], a.ravel()])
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        t = f["Header"].attrs["Time"]; g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        x = np.array(g["Coordinates"])[o]; v = np.array(g["Velocities"])[o]
    y = solve_ivp(rhs, (0, t), np.concatenate([x0.ravel(), v0.ravel()]), method="DOP853", rtol=1e-13, atol=1e-22).y[:, -1]
    a_ref, _ = _elements(m, y[:9].reshape(3, 3), y[9:].reshape(3, 3)); a, _ = _elements(m, x, v)
    print(f"{overrides}: |da/a| = {abs(a / a_ref - 1):.2e}")
    assert abs(a / a_ref - 1) < da_max
