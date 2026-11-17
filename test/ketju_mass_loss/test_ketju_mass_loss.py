"""KETJU start subtraction after mass loss (audit K2). An isolated 1.2 + 0.8 Msun binary (a = 50 AU, e = 0.6, P ~ 250 yr) in a
KETJU region whose steps (1e-5) span ~40 orbits (the ketju_snapshot_midstep IC). A test hook halves star 1's mass right after the
gravity call that ends the first region step, where Stella feedback takes mass from a star.

Derivation. The next first host half-kick gives partner i the tree pair force of that gravity call, G m_j(old) r^/r^2 dt/2; the
negative half-kick must remove exactly that, i.e. with the masses (and softening/zeta) of that call, while MSTAR integrates the
new masses. With the current mass the residual impulse is dv_i = G dm r^/r^2 dt/2, dv/v ~ (dm/M) pi dt/P ~ 0.3 x 40 pi ~ 40
here: the binary is torn apart. Reference: instantaneous mass loss of a Kepler binary, a' = -G M' / (2 E') from the exact
two-body state (r, v) at the mass-loss time with M' = M - dm; a' stays constant afterwards. Bar, set before running: the
snapshot's semi-major axis at t = 2e-5 within 1e-3 of a'.
No step traces: the hook removes mass outside the physics on purpose, which the mass trace (correctly) reports; the production
path of the same subtraction (Stella SN in a member binary) is the end-to-end follow-up."""

import glob
import subprocess
import sys

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from gizmo.test import build_and_run_test, clean_test_outputs

TEST_NAME = "ketju_mass_loss"
TEST_DIR = f"test/{TEST_NAME}"
G = 4.300710573e-6


def _a(m, x, v):
    d = x[1] - x[0]; u = v[1] - v[0]; mu = G * (m[0] + m[1]); return -mu / (2 * (0.5 * u @ u - mu / np.linalg.norm(d)))


def test_ketju_mass_loss():
    subprocess.run([sys.executable, "make_ketju_mass_loss_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    with h5py.File(f"{TEST_DIR}/ketju_mass_loss_ics.hdf5", "r") as f:
        g = f["PartType4"]; m0 = np.array(g["Masses"], dtype=float)[:2] * 1e10
        x0 = np.array(g["Coordinates"], dtype=float)[:2]; v0 = np.array(g["Velocities"], dtype=float)[:2]
    t_ml = float(open(f"{TEST_DIR}/output/test_mass_loss.txt").read())

    def rhs(t, y):
        d = y[3:6] - y[0:3]; a = G * d / np.linalg.norm(d) ** 3
        return np.concatenate([y[6:9], y[9:12], m0[1] * a, -m0[0] * a])
    y = solve_ivp(rhs, (0, t_ml), np.concatenate([x0.ravel(), v0.ravel()]), method="DOP853", rtol=1e-13, atol=1e-22).y[:, -1]
    a_ref = _a(np.array([0.5 * m0[0], m0[1]]), y[:6].reshape(2, 3), y[6:].reshape(2, 3))
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        t = f["Header"].attrs["Time"]; g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        m = np.array(g["Masses"], dtype=float)[o][:2] * 1e10; x = np.array(g["Coordinates"])[o][:2]; v = np.array(g["Velocities"])[o][:2]
    assert t > t_ml and abs(m[0] / (0.5 * m0[0]) - 1) < 1e-12
    a = _a(m, x, v)
    print(f"mass loss at t = {t_ml:.6e}: a' = {a_ref:.6e}, snapshot a = {a:.6e} at t = {t:.3e}, |da/a| = {abs(a / a_ref - 1):.3e}")
    assert a > 0 and abs(a / a_ref - 1) < 1e-3
