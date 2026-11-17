"""Clamped oscillating plate: the first bending mode of a cantilevered elastic plate against thin-plate theory (Gray,
Monaghan & Swift 2001; Zhang et al. 2023, sec. 6.1).

A 2D plate (plane strain, see make_ics.py) clamped at one end is given the velocity profile of its first bending mode
and oscillates. Thin-plate (Euler-Bernoulli) theory with the plane-strain modulus gives the angular frequency
w^2 = E H^2 k^4 / (12 rho (1 - nu^2)), k L = 1.875: period 0.2541 for these parameters, and a tip amplitude v_tip / w.
Thin-plate theory neglects shear deformation and rotary inertia, so the true period of this plate (H/L = 0.1) is a few
per cent longer; published SPH results at H/dp = 30 are 0.262-0.273. The run covers just over one period (about 8 minutes at the default rank and thread counts, so it is marked slow).

Validates:
  1. The run reaches its end time with mass conserved and the clamped cells unmoved.
  2. The period of the tip motion is within 10% of thin-plate theory (measured at H/dp = 10: +7.5%; +1.7% at H/dp = 20).
  3. The first tip excursion is within 10% of the thin-plate amplitude v_tip / w (measured: +2%).
  4. The oscillation loses less than 20% of its amplitude per period (measured: 10% at H/dp = 10, 6.5% at H/dp = 20)
     and does not grow (ratio below 1.05, allowing for the fit's noise).
"""

import glob

import h5py
import numpy as np
import pytest
from scipy.optimize import curve_fit

from gizmo.test import (
    build_and_run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
)

TEST_NAME = "oscillating_plate"

# must match make_ics.py
L, H, RHO_0, K, MU, V_FRAC, N_Y, KL = 0.2, 0.02, 1000.0, 3.25e6, 7.15e5, 0.01, 10, 1.875


def thin_plate():
    """angular frequency and tip amplitude of the first bending mode, plane strain"""
    E = 9 * K * MU / (3 * K + MU)
    nu = (3 * K - 2 * MU) / (2 * (3 * K + MU))
    w = np.sqrt(E * H**2 * (KL / L)**4 / (12 * RHO_0 * (1 - nu**2)))
    return w, V_FRAC * np.sqrt(K / RHO_0) / w


def _make_ics():
    import subprocess, sys
    subprocess.run([sys.executable, "make_ics.py"], check=True, cwd=f"test/{TEST_NAME}")


@pytest.mark.slow
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_oscillating_plate(num_mpi_ranks, num_omp_threads):
    _make_ics()
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    snaps = sorted(glob.glob(f"test/{TEST_NAME}/output/snapshot_*.hdf5"))
    assert snaps, "no snapshots: GIZMO did not run"
    assert_final_time(snaps[-1], TEST_NAME)

    def load(snap):
        with h5py.File(snap, "r") as f:
            g = f["PartType0"]
            ids, x, m = g["ParticleIDs"][:], g["Coordinates"][:], g["Masses"][:]
            return f["Header"].attrs["Time"], ids, x, m

    t0, ids0, x0, m0 = load(snaps[0])
    clamp0 = x0[ids0 == 0]
    free = ids0 > 0
    order0 = np.argsort(ids0[free])
    xf0 = x0[free][order0]
    tip = xf0[:, 0] > xf0[:, 0].max() - 0.5 * H / N_Y
    t, y = [], []
    for snap in snaps:
        ts, ids, x, m = load(snap)
        xf = x[ids > 0][np.argsort(ids[ids > 0])]
        t.append(ts); y.append(np.mean(xf[tip, 1] - xf0[tip, 1]))
    t, y = np.array(t), np.array(y)
    _, ids1, x1, m1 = load(snaps[-1])

    assert abs(m1.sum() / m0.sum() - 1) < 1e-12, "mass not conserved"
    clamp1 = x1[ids1 == 0]
    rows = lambda c: c[np.lexsort(c.T[::-1])]          # sort whole coordinate rows, not each column separately
    assert len(clamp1) == len(clamp0) and np.abs(rows(clamp1) - rows(clamp0)).max() == 0, "clamped cells moved"

    w_theory, amp_theory = thin_plate()
    model = lambda tt, c0, c1, a, g, w, ph: c0 + c1 * tt + a * np.exp(-g * tt) * np.cos(w * tt + ph)
    s = t > 0.02
    (c0, c1, a, gamma, w, ph), _ = curve_fit(model, t[s], y[s], p0=[0, 0, amp_theory, 2.0, w_theory, -np.pi / 2], maxfev=20000)
    period, period_theory = 2 * np.pi / abs(w), 2 * np.pi / w_theory
    ratio = np.exp(-gamma * period)
    first = y[t < 0.5 * period].max()
    print(f"period {period:.4f} vs thin-plate {period_theory:.4f} ({period / period_theory - 1:+.3f}); first excursion {first:.5f} vs v_tip/w {amp_theory:.5f}; "
          f"amplitude ratio per period {ratio:.3f}")

    assert abs(period / period_theory - 1) < 0.10, f"period {period:.4f} vs thin-plate theory {period_theory:.4f}"
    assert abs(first / amp_theory - 1) < 0.10, f"first tip excursion {first:.5f} vs thin-plate {amp_theory:.5f}"
    assert 0.8 < ratio < 1.05, f"amplitude ratio {ratio:.3f} per period (above 1 means the oscillation grows: an instability)"
