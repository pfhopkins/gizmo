"""Taylor anvil test: a copper rod hits a rigid wall at 227 m/s, flows plastically and stops, mushroomed (the Abaqus
"impact of a copper rod" benchmark; Taylor 1948, Wilkins & Guinan 1973).

Two mirror-image rods (see make_ics.py) collide on the symmetry plane, which stands in for a rigid frictionless wall;
the measurements use the rod at x > 10. The masses are first calibrated so that every cell starts at the reference
density (a few runs to TimeMax ~0, see make_ics.py), then the impact runs to 80 us after contact, by which time the
rod has stopped. Reference (Abaqus, ten element formulations, converged): the free end moves 13.10-13.18 mm after
contact (shortening) and the impact-face corner moves out 5.52-6.06 mm (widening).

The test runs at R/dp = 4, which is far from converged: shortening is 9.7 mm here and rises with resolution (11.0 and
11.6 mm at R/dp = 6 and 8, the gap travel removed the same way), while the widening rises past the benchmark range
(5.1, 5.9, 7.1 mm), an open discrepancy. The bands below are therefore those of this resolution, broad enough not to
overfit it, and are meant to catch failures of the method, above all the tensile runaway: with the tensile-instability
correction active in the plastically flowing material (as it was before it was made to fade there), the rod blows
apart, gaining over twenty times its initial kinetic energy.

Validates (at the end; energy over both rods, everything else for the rod at x > 10):
  1. The run reaches its end time with mass conserved.
  2. Energy: kinetic plus internal energy of both rods changes by less than 10% of their initial kinetic energy (measured: +4.5%; the rod
     at x > 10 alone shows +6-7%, varying from run to run, because energy crosses the contact between the rods).
  3. The rod has stopped (kinetic energy below 2% of the initial value; measured 0.06%) and is one body: the largest
     connected group of cells (linked within 2 lattice spacings) holds at least 99% of them; no cell has crossed the
     wall by more than half a lattice spacing.
  4. Shortening (free-end displacement minus the travel before contact) between 8 and 13.5 mm, widening (radial
     displacement of the impact-face corner, median over its cells) between 4 and 7 mm.
"""

import glob
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from gizmo.test import (
    build_and_run_test,
    run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
)

TEST_NAME = "taylor_bar"
TEST_DIR = f"test/{TEST_NAME}"
CENTER, DP, GAP, V_IMPACT = 10.0, 0.08, 0.24, 2.27e4      # must match make_ics.py
CALIBRATION_TOLERANCE, CALIBRATION_PASSES = 3e-3, 6


def _calibrate(num_mpi_ranks, num_omp_threads):
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):        # validating an external run: keep its calibrated IC
        return float("nan")
    subprocess.run([sys.executable, "make_ics.py"], check=True, cwd=TEST_DIR)
    zero_time = {"TimeMax": 1.0e-12}
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, param_overrides=zero_time)
    for _ in range(CALIBRATION_PASSES):
        err = float(subprocess.run([sys.executable, "make_ics.py", "--calibrate"], check=True, cwd=TEST_DIR, capture_output=True, text=True).stdout)
        if err < CALIBRATION_TOLERANCE:
            for leftover in glob.glob(f"{TEST_DIR}/{TEST_NAME}_override.params*"):
                os.remove(leftover)
            return err
        os.chdir(TEST_DIR)
        run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, param_overrides=zero_time)
        os.chdir("../../")
    raise AssertionError(f"mass calibration did not converge: density error {err:.2e} after {CALIBRATION_PASSES} passes")


def _load(snap):
    with h5py.File(snap, "r") as f:
        g = f["PartType0"]; order = np.argsort(g["ParticleIDs"][:])
        return {k: g[k][:][order] for k in ("Coordinates", "Velocities", "Masses", "InternalEnergy")}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_taylor_bar(num_mpi_ranks, num_omp_threads):
    err = _calibrate(num_mpi_ranks, num_omp_threads)
    print(f"masses calibrated: density error {err:.2e}")
    clean_test_outputs(TEST_NAME)
    os.chdir(TEST_DIR)
    run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    os.chdir("../../")
    snaps = sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))
    assert snaps, "no snapshots: GIZMO did not run"
    assert_final_time(snaps[-1], TEST_NAME)

    s0, s1 = _load(snaps[0]), _load(snaps[-1])
    x0, x1 = s0["Coordinates"], s1["Coordinates"]
    rod = x0[:, 0] > CENTER
    kinetic = lambda s, sel: 0.5 * np.sum(s["Masses"][sel] * np.sum(s["Velocities"][sel]**2, 1))
    energy = lambda s, sel: kinetic(s, sel) + np.sum(s["Masses"][sel] * s["InternalEnergy"][sel])
    KE0 = kinetic(s0, rod)
    both = np.ones(len(rod), bool)   # energy is exchanged between the rods across the contact (the jitter is not mirror-symmetric)

    assert abs(s1["Masses"].sum() / s0["Masses"].sum() - 1) < 1e-12, "mass not conserved"
    dE = (energy(s1, both) - energy(s0, both)) / kinetic(s0, both)
    dE_rod = (energy(s1, rod) - energy(s0, rod)) / KE0
    KE_left = kinetic(s1, rod) / KE0
    n_comp, labels = connected_components(cKDTree(x1[rod]).sparse_distance_matrix(cKDTree(x1[rod]), 2 * DP), directed=False)
    largest = np.bincount(labels).max() / rod.sum()

    r0, r1 = np.hypot(x0[:, 1] - CENTER, x0[:, 2] - CENTER), np.hypot(x1[:, 1] - CENTER, x1[:, 2] - CENTER)
    face0, end0 = x0[rod, 0].min(), x0[rod, 0].max()
    first, last = rod & (x0[:, 0] < face0 + 0.5 * DP), rod & (x0[:, 0] > end0 - 0.5 * DP)
    corner = first & (r0 > r0[first].max() - 0.5 * DP)
    shortening = 10 * (-np.median(x1[last, 0] - x0[last, 0]) - GAP / 2)      # mm; GAP / 2 = travel to contact
    widening = 10 * np.median(r1[corner] - r0[corner])
    print(f"(E-E0)/KE0 both rods {dE:+.3e} (this rod alone {dE_rod:+.3e}); KE left {KE_left:.2e}; largest connected group {largest:.4f}; wall crossing {CENTER - x1[rod, 0].min():+.3f} cm; "
          f"shortening {shortening:.2f} mm, widening {widening:.2f} mm (Abaqus 13.10-13.18 / 5.52-6.06)")

    assert abs(dE) < 0.1, f"energy changed by {dE:+.3f} of the initial kinetic energy"
    assert KE_left < 0.02, f"the rod has not stopped: kinetic energy {KE_left:.3f} of the initial"
    assert largest > 0.99, f"the rod broke up: largest connected group holds {largest:.3f} of its cells"
    assert x1[rod, 0].min() > CENTER - 0.5 * DP, "cells crossed the wall"
    assert 8.0 < shortening < 13.5, f"shortening {shortening:.2f} mm outside 8-13.5"
    assert 4.0 < widening < 7.0, f"widening {widening:.2f} mm outside 4-7"
