"""1D thermonuclear detonation test.

A tube of He4 fuel at WD conditions (rho=1e7 g/cc) with a hot spot (T=2 GK)
at the left end triggers a triple-alpha detonation wave. The test verifies:

1. The detonation propagates (burning front moves rightward)
2. He4 is consumed behind the front and C12 is produced
3. Energy is released (internal energy increases behind front)
4. The simulation runs stably to completion

Reference: Timmes & Niemeyer 2000 (ApJ 537, 993) for C/O detonations;
this test uses He4 fuel which is simpler but demonstrates the same physics.
"""

import pytest
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os

from gizmo.test import build_and_run_test, assert_final_time, default_mpi_ranks, default_omp_threads, get_final_snapshot


@pytest.fixture(scope="module", autouse=True)
def generate_ics():
    """Generate IC file before the test runs."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ic_file = os.path.join(test_dir, "nuclear_detonation_ics.hdf5")
    if not os.path.exists(ic_file):
        from make_detonation_ics import make_detonation_ics
        make_detonation_ics(ic_file)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_nuclear_detonation(num_mpi_ranks, num_omp_threads):
    test_name = "nuclear_detonation"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as F:
        x = F["PartType0/Coordinates"][:, 0]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        T = F["PartType0/Temperature"][:]
        met = F["PartType0/Metallicity"][:]  # shape (N, num_metal + num_nuclear)

    # nuclear species start after NUM_METAL_SPECIES=1 (total Z)
    X_He4 = met[:, 1]  # He4 mass fraction
    X_C12 = met[:, 2]  # C12 mass fraction

    # sort by position
    order = x.argsort()
    x, rho, u, T, X_He4, X_C12 = [arr[order] for arr in [x, rho, u, T, X_He4, X_C12]]

    # plot profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, rho, ".", ms=1); axes[0, 0].set_ylabel("Density [g/cc]")
    axes[0, 1].plot(x, T, ".", ms=1); axes[0, 1].set_ylabel("Temperature [K]"); axes[0, 1].set_yscale("log")
    axes[1, 0].plot(x, X_He4, ".", ms=1, label="He4"); axes[1, 0].plot(x, X_C12, ".", ms=1, label="C12")
    axes[1, 0].set_ylabel("Mass fraction"); axes[1, 0].legend()
    axes[1, 1].plot(x, u, ".", ms=1); axes[1, 1].set_ylabel("Internal energy [erg/g]")
    for ax in axes.flat:
        ax.set_xlabel("x [cm]")
    plt.suptitle("1D He4 Detonation")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/detonation_profiles.png", dpi=150)
    plt.close()

    # validation checks
    N = len(x)
    assert np.all(np.isfinite(rho)), "Non-finite density"
    assert np.all(rho > 0), "Negative density"

    # the hot spot should have burned: He4 should be depleted in the left region
    left_quarter = slice(0, N // 4)
    right_quarter = slice(3 * N // 4, N)

    X_He4_left = X_He4[left_quarter].mean()
    X_He4_right = X_He4[right_quarter].mean()
    X_C12_left = X_C12[left_quarter].mean()

    print(f"Left quarter:  <X(He4)> = {X_He4_left:.4f}, <X(C12)> = {X_C12_left:.4f}")
    print(f"Right quarter: <X(He4)> = {X_He4_right:.4f}")

    # behind the detonation front, He4 should be consumed and C12 produced
    assert X_He4_left < X_He4_right, \
        f"He4 should be consumed in the burned region: left={X_He4_left:.4f} vs right={X_He4_right:.4f}"
    assert X_C12_left > 0.01, \
        f"C12 should be produced behind the front: <X(C12)>_left = {X_C12_left:.4f}"

    # internal energy should be higher in the burned region
    u_left = u[left_quarter].mean()
    u_right = u[right_quarter].mean()
    assert u_left > u_right, \
        f"Energy should be higher behind front: u_left={u_left:.4e} vs u_right={u_right:.4e}"
