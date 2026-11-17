"""KETJU region members under PMGRID: the ketju_tide pair (1 + 1 Msun, a = 200 AU, in a KETJU region) and a 20 Msun star 5000 AU
away on a circular outer orbit, in a periodic box L = 50000 AU with PMGRID = 64, for a quarter of the outer orbit.

Derivation. GIZMO splits gravity (Springel 2005, eq. 21-22): the tree applies F_N(r) S(r) with
S = erfc(r/2r_s) + (r/(r_s sqrt(pi))) exp(-r^2/4r_s^2) (shortrange_table), the mesh the rest F_N (1 - S) (GravPM),
r_s = ASMTH L/PMGRID = 1.25 x 781 AU = 977 AU and the tree cuts at RCUT r_s = 4400 AU. A member's kick must be
tree + PM + KETJU internal: the tree half-kicks are applied to members (kicks.cc), KETJU's negative half-kicks subtract the full
softened Newtonian pair force F_N between members (tree_pair_gravity_factors, no S), and MSTAR integrates F_N inside the chain.
Chain-internal sum: S F_N (tree) + (1 - S) F_N (mesh) - F_N (negative kicks) + F_N (MSTAR) = F_N, counted once. Without the PM
kick on members the chain gets F_N - (1 - S) F_N and, here, the pair feels no 20 Msun star at all (5000 AU > RCUT r_s: its force
is entirely in the mesh), so the pair moves on a straight line. In a quarter outer orbit, t = (pi/2) r/v, a missing central force
displaces the pair by 1/2 (v^2/r) t^2 = 1.23 r. Bars, set before running: the pair CoM relative to the 20 Msun star within
0.05 r_out of the open-box DOP853 three-body solution (PM force errors are ~1 %, periodic images change the field by
(r_out/L)^3 ~ 1e-3), the pair's semi-major axis within 1e-2 of the reference; mass and energy step traces close."""

import glob
import os
import subprocess
import sys

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from gizmo.test import build_and_run_test, clean_test_outputs
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "ketju_pm"
TEST_DIR = f"test/{TEST_NAME}"
G = 4.300710573e-6
AU = 1.0 / 206264.806 / 1e3


def _a(m, x, v):
    d = x[1] - x[0]; u = v[1] - v[0]; mu = G * (m[0] + m[1]); return -mu / (2 * (0.5 * u @ u - mu / np.linalg.norm(d)))


def test_ketju_pm():
    subprocess.run([sys.executable, "make_ketju_pm_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    with h5py.File(f"{TEST_DIR}/ketju_pm_ics.hdf5", "r") as f:
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
    assert t > 0.99 * 1.8e-5
    y = solve_ivp(rhs, (0, t), np.concatenate([x0.ravel(), v0.ravel()]), method="DOP853", rtol=1e-13, atol=1e-22).y[:, -1]
    xr, vr = y[:9].reshape(3, 3), y[9:].reshape(3, 3)
    rel = lambda p: (m[0] * p[0] + m[1] * p[1]) / (m[0] + m[1]) - p[2]
    r_out = np.linalg.norm(rel(x0))
    d_rel = np.linalg.norm(rel(x) - rel(xr)) / r_out
    da = abs(_a(m, x, v) / _a(m, xr, vr) - 1)
    print(f"t = {t:.3e}: pair CoM vs reference {d_rel:.3e} r_out, |da/a| {da:.3e}")
    assert d_rel < 0.05
    assert da < 1e-2
    rep = keep_and_check_traces(os.path.join(TEST_DIR, "output"), os.path.join(TEST_DIR, "output"))
    assert rep.ok, rep.summary()
