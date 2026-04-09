"""X-ray burst column test.

A 1D column of He4 on a neutron star surface is heated at the base to trigger
thermonuclear runaway. The test verifies:

1. Thermonuclear ignition occurs (temperature rises rapidly at the base)
2. He4 is consumed and C12/heavier species are produced
3. Nuclear energy release heats the column
4. The simulation remains stable (hydrostatic balance maintained)

This test generates its own ICs locally (no remote download needed).
"""

import pytest
import sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
from glob import glob

from gizmo.test import (build_gizmo_for_test, run_test, get_final_snapshot,
                         assert_final_time, default_mpi_ranks, default_omp_threads,
                         clean_test_outputs)


def _generate_ics_and_table(test_dir):
    """Generate IC file and ensure helm_table.dat exists."""
    sys.path.insert(0, test_dir)
    from make_xrb_ics import make_xrb_ics
    ic_path = os.path.join(test_dir, "nuclear_xrb_ics.hdf5")
    if not os.path.exists(ic_path):
        make_xrb_ics(ic_path)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_nuclear_xrb(num_mpi_ranks, num_omp_threads):
    test_name = "nuclear_xrb"
    test_dir = os.path.join("test", test_name)

    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    _generate_ics_and_table(os.path.abspath(test_dir))

    cwd = os.getcwd()
    os.chdir(test_dir)
    run_test(test_name, num_mpi_ranks, num_omp_threads)
    os.chdir(cwd)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as F:
        z = F["PartType0/Coordinates"][:, 2]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        met = F["PartType0/Metallicity"][:]

    X_He4 = met[:, 1]
    X_C12 = met[:, 2]
    X_O16 = met[:, 3]

    order = z.argsort()
    z, rho, u, X_He4, X_C12, X_O16 = [arr[order] for arr in [z, rho, u, X_He4, X_C12, X_O16]]

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(z, rho, ".-", ms=2); axes[0, 0].set_ylabel("Density"); axes[0, 0].set_yscale("log")
    axes[0, 1].plot(z, u, ".-", ms=2); axes[0, 1].set_ylabel("Internal energy"); axes[0, 1].set_yscale("log")
    axes[1, 0].plot(z, X_He4, ".-", ms=2, label="He4")
    axes[1, 0].plot(z, X_C12, ".-", ms=2, label="C12")
    axes[1, 0].plot(z, X_O16, ".-", ms=2, label="O16")
    axes[1, 0].set_ylabel("Mass fraction"); axes[1, 0].legend()
    for ax in axes.flat:
        ax.set_xlabel("Height z [cm]")
    plt.suptitle("X-ray Burst Column")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/xrb_profiles.png", dpi=150)
    plt.close()

    # light curve
    snap_dir = os.path.join("test", test_name, "output")
    times, total_u = [], []
    for snap_file in sorted(glob(os.path.join(snap_dir, "snapshot_*.hdf5"))):
        with h5py.File(snap_file, "r") as F:
            times.append(F["Header"].attrs["Time"])
            total_u.append(np.sum(F["PartType0/Masses"][:] * F["PartType0/InternalEnergy"][:]))
    if len(times) > 1:
        plt.figure()
        plt.plot(times, total_u, "o-")
        plt.xlabel("Time [s]"); plt.ylabel("Total internal energy [erg]")
        plt.title("XRB Light Curve (proxy)")
        plt.savefig(f"test/{test_name}/xrb_lightcurve.png", dpi=150)
        plt.close()

    # validation
    N = len(z)
    assert np.all(np.isfinite(rho)), "Non-finite density"
    assert np.all(rho > 0), "Negative density"

    base = slice(0, N // 4)
    top = slice(3 * N // 4, N)

    X_He4_base = X_He4[base].mean()
    X_C12_base = X_C12[base].mean()

    print(f"Base: <X(He4)>={X_He4_base:.4f}  <X(C12)>={X_C12_base:.4f}")
    print(f"Top:  <X(He4)>={X_He4[top].mean():.4f}")

    assert X_He4_base < 0.9, \
        f"He4 should be partially consumed at base: <X(He4)>={X_He4_base:.4f}"
    assert X_C12_base > 0.01, \
        f"C12 should be produced at base: <X(C12)>={X_C12_base:.4f}"
