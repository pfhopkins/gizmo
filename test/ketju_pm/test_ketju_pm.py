"""KETJU region members under PMGRID: a 1 + 1 Msun pair (a = 200 AU, e = 0.5, in a KETJU region of 800 AU) and a 20 Msun star
R_out = 20000 AU away on a circular outer orbit, in a periodic box L = 2e5 AU with PMGRID = 128, for a quarter of the outer orbit
(1.54e-4 code time).

Derivation. GIZMO splits gravity (Springel 2005, eq. 21-22): the tree applies F_N(r) S(r) with
S = erfc(r/2r_s) + (r/(r_s sqrt(pi))) exp(-r^2/4r_s^2) (shortrange_table), the mesh the rest F_N (1 - S) (GravPM),
r_s = ASMTH L/PMGRID = 1.25 x 1563 AU = 1953 AU, and the tree cuts at RCUT r_s = 8790 AU < R_out: the 20 Msun star's pull on the
pair is entirely in the mesh, 12.8 cells away. A member's kick must be tree + PM + KETJU internal: the tree half-kicks are applied to
members (kicks.cc), KETJU's negative half-kicks subtract the full softened Newtonian pair force F_N between members
(tree_pair_gravity_factors, no S), MSTAR integrates F_N inside the chain. Chain-internal sum: S F_N (tree) + (1 - S) F_N (mesh)
- F_N + F_N = F_N, up to the mesh's error on its (1 - S) F_N share, which for r <= 300 AU << r_s is (r/r_s)^3 ~ 4e-3 of F_N times
the mesh's relative error (the run-time guard keeps KetjuRegionRadius <= r_s/2, where 1 - S <= 1.1e-2). Without the PM kick on
members the pair feels no 20 Msun star at all and moves on a straight line, 1.23 R_out off after a quarter orbit.
Bars, set before running: the pair CoM relative to the 20 Msun star within 0.05 R_out of the open-box DOP853 three-body solution
(periodic images change the field by (R_out/L)^3 = 1e-3), the pair's semi-major axis within 1e-2 of the reference, the mass trace and
the energy step trace closes with its own bars (integration residual 1e-2 of E_scale), as set before the first run.
The trace's E_pot leaves out each particle's own mesh self-potential, the exact CIC value (PM_EXACT_SELF_POTENTIAL_DIAG; the
continuum -G m/(sqrt(pi) r_s) differs from it by up to ~20% with sub-cell position). 2026-11-20: residual 2.0e-2 of E_scale at one
sync (pair at pericentre, step 9), all others <= 6.1e-3; with the continuum self term 4.9e-2 at 39 syncs. Still fails the 1e-2 bar."""

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
T_QUARTER = 1.54e-4


def _sync_res(path):
    out = {}
    for l in open(path):
        if l.startswith("SYNC"):
            kv = dict(tok.split("=", 1) for tok in l.split()[1:] if "=" in tok)
            out[round(float(kv["t"]), 12)] = (float(kv["int_res"]), float(kv["E_scale"]))
    return out


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
    assert t > 0.99 * T_QUARTER
    y = solve_ivp(rhs, (0, t), np.concatenate([x0.ravel(), v0.ravel()]), method="DOP853", rtol=1e-13, atol=1e-22).y[:, -1]
    xr, vr = y[:9].reshape(3, 3), y[9:].reshape(3, 3)
    rel = lambda p: (m[0] * p[0] + m[1] * p[1]) / (m[0] + m[1]) - p[2]
    r_out = np.linalg.norm(rel(x0))
    d_rel = np.linalg.norm(rel(x) - rel(xr)) / r_out
    da = abs(_a(m, x, v) / _a(m, xr, vr) - 1)
    print(f"t = {t:.3e}: pair CoM vs reference {d_rel:.3e} r_out, |da/a| {da:.3e}")
    assert d_rel < 0.05
    assert da < 1e-2
    res = _sync_res(f"{TEST_DIR}/output/energy_step_trace.txt")
    print(f"integration residual: max {max(abs(r) / s for r, s in res.values()):.3e} of E_scale over {len(res)} syncs, E_scale {max(s for r, s in res.values()):.3e}")
    rep = keep_and_check_traces(os.path.join(TEST_DIR, "output"), os.path.join(TEST_DIR, "output"))
    assert rep.ok, rep.summary()


def test_ketju_pm_comoving():
    """Production combination of item K1: comoving (a = 0.5), PMGRID, 2 ranks, 2 KETJU regions, the physical geometry of
    test_ketju_pm (make_ketju_pm_comoving_ics.py). Two binaries at +x and +y, R_out = 20000 AU physical from a 20 Msun star (force
    on them entirely in the mesh), single 2 Msun stars at the mirror positions. Box and mesh are mirror-symmetric about the 20 Msun
    star, so each binary's CoM must stay the mirror image of its single star: |CoM_pair + x_single - 2 x_centre| / R_out < 1e-3
    after a quarter outer orbit (the binaries' own tide, (a/R_out)^2 = 6e-6 of the force, is far below). Without the PM kick the
    binaries do not orbit at all.
    GIZMO requires Omega0 to match the box's mean density (check_omega, absolute tolerance 1e-2), so Omega0 is computed here exactly
    as GIZMO does (G and H0 from constants.h with the params' units, h = 1). a_end = a0 (1 + H(a0) dt), H(a0) = H0 sqrt(Omega0/a0^3),
    dt the quarter orbit; MaxSizeTimestep (in d ln a) = H(a0) x 1.6e-6, the non-comoving cap. The mirror symmetry holds for any
    background. Mass and energy step traces must close."""
    subprocess.run([sys.executable, "make_ketju_pm_ics.py"], cwd=TEST_DIR, check=True)  # the harness expects <name>_ics.hdf5
    subprocess.run([sys.executable, "make_ketju_pm_comoving_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    with h5py.File(f"{TEST_DIR}/ketju_pm_comoving_ics.hdf5", "r") as f:
        m_tot = float(np.sum(f["PartType4"]["Masses"]))
    L = 4e5 * AU
    G_code = 6.672e-8 * 1.989e43 / (3.085678e21 * (1e5 * 1e5))
    H0 = 3.2407789e-18 * (3.085678e21 / 1e5)
    omega = m_tot / (L * L * L) / (3 * H0 * H0 / (8 * np.pi * G_code))
    H_a0 = H0 * np.sqrt(omega / 0.5 ** 3)
    t_end = 0.5 * (1 + H_a0 * T_QUARTER)
    build_and_run_test(TEST_NAME, 2, 1, param_overrides={"InitCondFile": "ketju_pm_comoving_ics", "ComovingIntegrationOn": 1,
                       "Omega0": omega, "TimeBegin": 0.5, "TimeMax": t_end, "TimeOfFirstSnapshot": t_end, "TimeBetSnapshot": 1.1,
                       "TimeBetStatistics": 1.1, "BoxSize": L, "MaxSizeTimestep": H_a0 * 1.6e-6,
                       "SofteningGasMaxPhys": 1.0e-4, "SofteningHaloMaxPhys": 0.05, "SofteningDiskMaxPhys": 0.02,
                       "SofteningBulgeMaxPhys": 0.02, "SofteningStarsMaxPhys": 1e-6, "SofteningBndryMaxPhys": 0.005})
    with h5py.File(sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))[-1], "r") as f:
        a = f["Header"].attrs["Time"]; g = f["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        x = np.array(g["Coordinates"])[o] * a; m = np.array(g["Masses"])[o]
    assert a >= 0.999999 * t_end
    c, r = x[0], 20000 * AU
    for k in (0, 1):
        i = 1 + 3 * k
        com = (m[i] * x[i] + m[i + 1] * x[i + 1]) / (m[i] + m[i + 1])
        d = np.linalg.norm(com + x[i + 2] - 2 * c) / r
        print(f"region {k}: |CoM + mirror single - 2 centre| = {d:.3e} R_out")
        assert d < 1e-3
    rep = keep_and_check_traces(os.path.join(TEST_DIR, "output"), os.path.join(TEST_DIR, "output"))
    assert rep.ok, rep.summary()


def test_ketju_pm_guard():
    """Run-time guard: KETJU members' mutual force must stay tree-dominated, i.e. member separations <= r_s/2 (mesh share
    1 - S <= 1.1e-2). In a box of 5e4 AU, r_s = 1.25 x 5e4/128 AU = 488 AU, so the 200 AU pair (apocentre 300 AU) exceeds r_s/2 = 244 AU
    and GIZMO must abort with the guard's message (endrun 8862). KetjuRegionRadius 240 AU (< r_s/2, a = 200 AU pair still captured): a
    check on the radius alone would pass; the members' separation does not."""
    import pytest
    subprocess.run([sys.executable, "make_ketju_pm_ics.py"], cwd=TEST_DIR, check=True)  # the harness expects <name>_ics.hdf5
    subprocess.run([sys.executable, "make_ketju_pm_ics.py", "ketju_pm_guard_ics.hdf5", "5e4"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    with pytest.raises(RuntimeError):
        build_and_run_test(TEST_NAME, 1, 1, param_overrides={"InitCondFile": "ketju_pm_guard_ics", "BoxSize": 5e4 * AU, "KetjuRegionRadius": 240 * AU})
    out = open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()
    assert "exceed half the PM split scale" in out
