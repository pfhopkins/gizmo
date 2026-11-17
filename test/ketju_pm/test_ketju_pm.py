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


def test_ketju_pm_comoving():
    """Production combination of item K1: comoving (a = 0.5, Omega0 = 1), PMGRID, 2 ranks, 2 KETJU regions. Two binaries at
    +x and +y, 5000 AU physical from a 20 Msun star (force on them entirely in the mesh), single 2 Msun stars at the mirror
    positions. The box and the mesh are mirror-symmetric about the 20 Msun star, so each binary's CoM must stay the mirror image
    of its single star: |CoM_pair + x_single - 2 x_centre| / r < 1e-3 after a quarter outer orbit (the binaries' own tide,
    (a/r)^2 = 1e-4 of the force, is far below). Without the PM kick the binaries do not orbit at all (1.2 r off)."""
    subprocess.run([sys.executable, "make_ketju_pm_comoving_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    t_end = 0.5 * (1 + 0.5 * 0.2828427 * 1.8e-5)  # a after a quarter outer orbit (H(a=0.5) = 0.283 / code time)
    build_and_run_test(TEST_NAME, 2, 1, param_overrides={"InitCondFile": "ketju_pm_comoving_ics", "ComovingIntegrationOn": 1,
                       "Omega0": 1, "TimeBegin": 0.5, "TimeMax": t_end, "TimeOfFirstSnapshot": t_end, "TimeBetSnapshot": 1.1,
                       "TimeBetStatistics": 1.1, "BoxSize": 1e5 * AU})
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        a = f["Header"].attrs["Time"]; g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        x = np.array(g["Coordinates"])[o] * a; m = np.array(g["Masses"])[o]
    assert a >= 0.999999 * t_end
    c, r = x[0], 5000 * AU
    for k in (0, 1):
        i = 1 + 3 * k
        com = (m[i] * x[i] + m[i + 1] * x[i + 1]) / (m[i] + m[i + 1])
        d = np.linalg.norm(com + x[i + 2] - 2 * c) / r
        print(f"region {k}: |CoM + mirror single| = {d:.3e} r, single moved {np.linalg.norm(x[i + 2] - c - (x[i + 2] - c) / np.linalg.norm(x[i + 2] - c) * r) / r:.3e} r")
        assert d < 1e-3
    rep = keep_and_check_traces(os.path.join(TEST_DIR, "output"), os.path.join(TEST_DIR, "output"))
    assert rep.ok, rep.summary()
