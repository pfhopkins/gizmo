"""KETJU region step caps and the fly-by step, against a DOP853 (rtol 1e-13) three-body integration of the same ICs.

Pair (1 + 1 Msun, a = 200 AU, e = 0.5) in a KETJU region; a 1 Msun star passes at 30 km/s with impact parameter 1500 AU
(outside the region: integrated by the host) or 600 AU (captured into the region).
- motion caps (default): with a field star around the pair the region step is capped (Mannerkoski+ 2023 eqs. 4, 6).
- fly-by step (KetjuFlybyStepEta 0.2): the intruder's own steps resolve the passage. Without it the intruder takes one
  3000-yr step through the encounter (151 AU off at the end) and the pair is 4e-3 off in a."""

import subprocess
import sys

import h5py
import glob
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gizmo.test import build_and_run_test, clean_test_outputs

TEST_NAME = "ketju_flyby"
TEST_DIR = f"test/{TEST_NAME}"
sys.path.insert(0, TEST_DIR)
T_END = 6.4e-6
AU = 1.0 / 206264.806 / 1e3
G = 4.300710573e-6


def _reference(b):
    from make_ketju_flyby_ics import state
    m, x, v = state(b)

    def rhs(t, y):
        p = y[:9].reshape(3, 3); a = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i != j:
                    d = p[j] - p[i]; a[i] += G * m[j] * d / np.linalg.norm(d) ** 3
        return np.concatenate([y[9:], a.ravel()])
    y = solve_ivp(rhs, (0, T_END), np.concatenate([x.ravel(), v.ravel()]), method="DOP853", rtol=1e-13, atol=1e-20).y[:, -1]
    return y[:9].reshape(3, 3), y[9:].reshape(3, 3)


def _sma(x, v):
    d = x[1] - x[0]; u = v[1] - v[0]; mu = G * 2
    return -mu / (2 * (0.5 * u @ u - mu / np.linalg.norm(d)))


def _errors(b):
    xr, vr = _reference(b)
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        x = np.array(g["Coordinates"])[o]; v = np.array(g["Velocities"])[o]
    return abs(_sma(x, v) / _sma(xr, vr) - 1), np.linalg.norm(x[2] - xr[2]) / AU


@pytest.mark.parametrize("b,ic,eta,da_max,dx_max", [
    (1500, "ketju_flyby_ics", 0.2, 3e-4, 1.0),
    (600, "ketju_flyby_b600_ics", 0.2, 3e-4, 1.0),
])
def test_ketju_flyby_step(b, ic, eta, da_max, dx_max):
    subprocess.run([sys.executable, "make_ketju_flyby_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1, param_overrides={"InitCondFile": ic, "KetjuFlybyStepEta": eta})
    da, dx = _errors(b)
    print(f"b={b} AU: pair |da/a| {da:.2e}, intruder off by {dx:.3g} AU")
    assert da < da_max and dx < dx_max


def test_ketju_flyby_motion_caps():
    subprocess.run([sys.executable, "make_ketju_flyby_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    caps = [l for l in open(f"{TEST_DIR}/test_{TEST_NAME}.out") if l.startswith("KETJU CAPS")]
    assert caps, "no KETJU CAPS line"
    n_motion = int(caps[-1].split("motion cap=")[1].split()[0])
    da, _ = _errors(1500)
    print(f"motion caps set {n_motion} region steps; pair |da/a| {da:.2e}")
    assert n_motion > 0 and da < 1e-2
