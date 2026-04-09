"""1D thermonuclear detonation test.

A tube of He4 fuel at WD conditions (rho=1e7 g/cc) with a hot spot (T=2 GK)
at the left end triggers a triple-alpha detonation wave. The test verifies:

1. The detonation propagates (burning front moves rightward)
2. He4 is consumed behind the front and C12 is produced
3. Energy is released (internal energy increases behind front)
4. The simulation runs stably to completion

This test generates its own ICs locally (no remote download needed).
"""

import pytest
import subprocess
import sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os

from gizmo.test import (build_gizmo_for_test, run_test, get_final_snapshot,
                         assert_final_time, default_mpi_ranks, default_omp_threads,
                         clean_test_outputs)


def _generate_ics_and_table(test_dir):
    """Generate IC file and ensure helm_table.dat exists."""
    sys.path.insert(0, test_dir)
    from make_detonation_ics import make_detonation_ics
    ic_path = os.path.join(test_dir, "nuclear_detonation_ics.hdf5")
    if not os.path.exists(ic_path):
        make_detonation_ics(ic_path)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_nuclear_detonation(num_mpi_ranks, num_omp_threads):
    test_name = "nuclear_detonation"
    test_dir = os.path.join("test", test_name)

    # step 1: build GIZMO
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)

    # step 2: generate ICs and ensure helm_table.dat (locally, no remote download)
    _generate_ics_and_table(os.path.abspath(test_dir))

    # step 3: run
    cwd = os.getcwd()
    os.chdir(test_dir)
    run_test(test_name, num_mpi_ranks, num_omp_threads)
    os.chdir(cwd)

    # step 4: analyze
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as F:
        x = F["PartType0/Coordinates"][:, 0]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        met = F["PartType0/Metallicity"][:]

    # nuclear species start after NUM_METAL_SPECIES=1 (total Z)
    X_He4 = met[:, 1]
    X_C12 = met[:, 2]

    order = x.argsort()
    x, rho, u, X_He4, X_C12 = [arr[order] for arr in [x, rho, u, X_He4, X_C12]]

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, rho, ".", ms=1); axes[0, 0].set_ylabel("Density [g/cc]")
    axes[0, 1].plot(x, u, ".", ms=1); axes[0, 1].set_ylabel("Internal energy [erg/g]")
    axes[1, 0].plot(x, X_He4, ".", ms=1, label="He4"); axes[1, 0].plot(x, X_C12, ".", ms=1, label="C12")
    axes[1, 0].set_ylabel("Mass fraction"); axes[1, 0].legend()
    for ax in axes.flat:
        ax.set_xlabel("x [cm]")
    plt.suptitle("1D He4 Detonation")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/detonation_profiles.png", dpi=150)
    plt.close()

    # validation
    N = len(x)
    assert np.all(np.isfinite(rho)), "Non-finite density"
    assert np.all(rho > 0), "Negative density"

    left_quarter = slice(0, N // 4)
    right_quarter = slice(3 * N // 4, N)

    X_He4_left = X_He4[left_quarter].mean()
    X_He4_right = X_He4[right_quarter].mean()
    X_C12_left = X_C12[left_quarter].mean()

    print(f"Left quarter:  <X(He4)> = {X_He4_left:.4f}, <X(C12)> = {X_C12_left:.4f}")
    print(f"Right quarter: <X(He4)> = {X_He4_right:.4f}")

    assert X_He4_left < X_He4_right, \
        f"He4 should be consumed in burned region: left={X_He4_left:.4f} vs right={X_He4_right:.4f}"
    assert X_C12_left > 0.01, \
        f"C12 should be produced: <X(C12)>_left = {X_C12_left:.4f}"

    u_left = u[left_quarter].mean()
    u_right = u[right_quarter].mean()
    assert u_left > u_right, \
        f"Energy should be higher behind front: u_left={u_left:.4e} vs u_right={u_right:.4e}"
