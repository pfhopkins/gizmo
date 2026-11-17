"""KETJU tight-binary expansion (audit K3): a hard 1.2 + 0.8 Msun binary (a = 2 AU, e = 0.6, P ~ 2 yr) with a 1 Msun star at
400 AU in the same region and KetjuExpandBinariesFactor = 100, so an expansion fires while the binary's period is
below 100 region steps.

Derivation (Mannerkoski+ 2023 intent: double a, keep e). About the pair CoM, r -> 2r and v -> v/sqrt(2): the pair's specific
energy eps = v^2/2 - G M/r -> eps/2, so a = -G M/(2 eps) -> 2a; the specific angular momentum h = |r x v| -> sqrt(2) h, and from
h^2 = G M a (1 - e^2) e stays the same; CoM position and velocity are unchanged. Each expansion doubles a: after k events
(counted from the "expanded tight binary" log lines) a = 2^k a0 and e = e0. The third star at 400 AU perturbs a by
~(a/r)^3 per orbit-averaged step: bars, set before running, |a/(2^k a0) - 1| < 1e-3 and |e - e0| < 1e-3 at t = 1e-5, k >= 1.
Each event's energy (dE = -E_pair/2 > 0 plus the tiny change of the pair-third star term) is booked in the coupling ledger."""

import glob
import re
import subprocess
import sys

import h5py
import numpy as np

from gizmo.test import build_and_run_test, clean_test_outputs

TEST_NAME = "ketju_expand_binary"
TEST_DIR = f"test/{TEST_NAME}"
G = 4.300710573e-6


def _ae(m, x, v):
    d = x[1] - x[0]; u = v[1] - v[0]; mu = G * (m[0] + m[1]); eps = 0.5 * u @ u - mu / np.linalg.norm(d)
    h = np.cross(d, u); a = -mu / (2 * eps); return a, np.sqrt(max(0.0, 1 + 2 * eps * (h @ h) / mu ** 2))


def test_ketju_expand_binary():
    subprocess.run([sys.executable, "make_ketju_expand_binary_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    with h5py.File(f"{TEST_DIR}/ketju_expand_binary_ics.hdf5", "r") as f:
        g = f["PartType4"]; m = np.array(g["Masses"], dtype=float)[:2] * 1e10
        a0, e0 = _ae(m, np.array(g["Coordinates"], dtype=float)[:2], np.array(g["Velocities"], dtype=float)[:2])
    k = len(re.findall(r"KETJU: expanded tight binary", open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()))
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        a, e = _ae(m, np.array(g["Coordinates"])[o][:2], np.array(g["Velocities"])[o][:2])
    print(f"{k} expansions: a/a0 = {a / a0:.6f} (expected {2 ** k}), e = {e:.6f} (e0 = {e0:.6f})")
    assert k >= 1
    assert abs(a / (2 ** k * a0) - 1) < 1e-3
    assert abs(e - e0) < 1e-3
