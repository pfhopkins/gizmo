"""KETJU tight-binary expansion (audit K3): a hard 1.2 + 0.8 Msun binary (a = 2 AU, e = 0.6, P ~ 2 yr) with a 1 Msun star at
900 AU in the same region and KetjuExpandBinariesFactor = 0.1: an expansion fires while the binary's period is below 0.1
region steps; the region step grows as the binary widens, so it fires a few times up to t = 1e-6 (a <= 32 AU, r3/a >= 28).

Derivation (Mannerkoski+ 2023 intent: double a, keep e). About the pair CoM, r -> 2r and v -> v/sqrt(2): the pair's specific
energy eps = v^2/2 - G M/r -> eps/2, so a = -G M/(2 eps) -> 2a; the specific angular momentum h = |r x v| -> sqrt(2) h, and from
h^2 = G M a (1 - e^2) e stays the same; CoM position and velocity are unchanged. Each expansion doubles a: after k events
(counted from the "expanded tight binary" log lines) a = 2^k a0 and e = e0. The third star at 900 AU perturbs a by
~(a/r)^3 per orbit-averaged step: bars, set before running, |a/(2^k a0) - 1| < 1e-3 and |e - e0| < 1e-3 at t = 1e-6, k >= 1.
Energy: each event injects dE = -E_pair/2 > 0 (plus the pair-third star term, ~(a/r3)^2 ~ 5e-6 of it). The first event's logged
dE must equal G m1 m2 / (4 a0) to 1e-4 (the third-star term is ~5e-6 of it);
the step trace's booking b_ketju_expand must equal the sum of the logged
events after the trace's baseline (its first full sync, where E0 is taken and bookings start) to 1e-8, and the mass and energy
traces must close (gizmo.conservation_trace bars). Factor 0.1 makes an expansion fire after that baseline."""

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
    ev = [(float(t_), float(x)) for t_, x in re.findall(r"KETJU: expanded tight binary \(t=([-+0-9.eE]+) .*?dE=([-+0-9.eE]+)\)", open(f"{TEST_DIR}/test_{TEST_NAME}.out").read())]
    dE_ref = G * m[0] * m[1] / (4 * a0) / 1e10  # code energy (1e10 Msun (km/s)^2)
    sync = [l for l in open(f"{TEST_DIR}/output/energy_step_trace.txt") if l.startswith("SYNC")]
    t0 = float([tok for tok in sync[0].split() if tok.startswith("t=")][0][2:])  # the trace's baseline (first full sync)
    book = [float(tok.split("=")[1]) for tok in sync[-1].split() if tok.startswith("b_ketju_expand=")]
    after = sum(dE for t_, dE in ev if t_ >= t0)
    print(f"events {ev}; first dE expected {dE_ref:.10e}; trace baseline t = {t0:.3e}, booked {book}, logged after baseline {after:.10e}")
    assert len(ev) == k and abs(ev[0][1] / dE_ref - 1) < 1e-4
    assert after > 0, "no expansion after the trace baseline: the booking is not exercised"
    assert len(book) == 1 and abs(book[0] / after - 1) < 1e-8
    from gizmo.conservation_trace import keep_and_check_traces
    rep = keep_and_check_traces(f"{TEST_DIR}/output", f"{TEST_DIR}/output")
    assert rep.ok, rep.summary()
