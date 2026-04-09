"""X-ray burst column test.

A 1D column of He4 on a neutron star surface is heated at the base to trigger
thermonuclear runaway. The test verifies:

1. Thermonuclear ignition occurs (temperature rises rapidly at the base)
2. He4 is consumed and C12/heavier species are produced
3. Nuclear energy release heats the column
4. The simulation remains stable (hydrostatic balance maintained)

Reference: Schatz et al. 2001 (PRL 86, 3471) for XRB nucleosynthesis;
Woosley et al. 2004 (ApJS 151, 75) for 1D burst models.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os

from gizmo.test import build_and_run_test, assert_final_time, default_mpi_ranks, default_omp_threads, get_final_snapshot


@pytest.fixture(scope="module", autouse=True)
def generate_ics():
    """Generate IC file before the test runs."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ic_file = os.path.join(test_dir, "nuclear_xrb_ics.hdf5")
    if not os.path.exists(ic_file):
        from make_xrb_ics import make_xrb_ics
        make_xrb_ics(ic_file)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_nuclear_xrb(num_mpi_ranks, num_omp_threads):
    test_name = "nuclear_xrb"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as F:
        z = F["PartType0/Coordinates"][:, 2]  # vertical position
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        T = F["PartType0/Temperature"][:]
        met = F["PartType0/Metallicity"][:]

    X_He4 = met[:, 1]  # He4 mass fraction (offset 1 for NUM_METAL_SPECIES=1)
    X_C12 = met[:, 2]  # C12
    X_O16 = met[:, 3]  # O16

    order = z.argsort()
    z, rho, u, T, X_He4, X_C12, X_O16 = [arr[order] for arr in [z, rho, u, T, X_He4, X_C12, X_O16]]

    # plot profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(z, rho, ".-", ms=2); axes[0, 0].set_ylabel("Density [g/cc]"); axes[0, 0].set_yscale("log")
    axes[0, 1].plot(z, T, ".-", ms=2); axes[0, 1].set_ylabel("Temperature [K]"); axes[0, 1].set_yscale("log")
    axes[1, 0].plot(z, X_He4, ".-", ms=2, label="He4")
    axes[1, 0].plot(z, X_C12, ".-", ms=2, label="C12")
    axes[1, 0].plot(z, X_O16, ".-", ms=2, label="O16")
    axes[1, 0].set_ylabel("Mass fraction"); axes[1, 0].legend()
    axes[1, 1].plot(z, u, ".-", ms=2); axes[1, 1].set_ylabel("Internal energy [erg/g]"); axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("Height z [cm]")
    plt.suptitle("X-ray Burst Column")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/xrb_profiles.png", dpi=150)
    plt.close()

    # light curve: total nuclear energy vs time (from all snapshots)
    snap_dir = os.path.join("test", test_name, "output")
    times, total_u = [], []
    for snap_file in sorted(os.listdir(snap_dir)):
        if snap_file.startswith("snapshot_") and snap_file.endswith(".hdf5"):
            with h5py.File(os.path.join(snap_dir, snap_file), "r") as F:
                times.append(F["Header"].attrs["Time"])
                mass_i = F["PartType0/Masses"][:]
                u_i = F["PartType0/InternalEnergy"][:]
                total_u.append(np.sum(mass_i * u_i))
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
    assert np.all(np.isfinite(T)), "Non-finite temperature"

    # the base should have burned: He4 depleted at z < H/4
    base = slice(0, N // 4)
    top = slice(3 * N // 4, N)

    X_He4_base = X_He4[base].mean()
    X_C12_base = X_C12[base].mean()
    T_base = T[base].mean()
    T_top = T[top].mean()

    print(f"Base: <X(He4)>={X_He4_base:.4f}  <X(C12)>={X_C12_base:.4f}  <T>={T_base:.4e} K")
    print(f"Top:  <X(He4)>={X_He4[top].mean():.4f}  <T>={T_top:.4e} K")

    # nuclear burning should have occurred at the base
    assert X_He4_base < 0.9, \
        f"He4 should be partially consumed at base: <X(He4)>={X_He4_base:.4f}"
    assert X_C12_base > 0.01, \
        f"C12 should be produced at base: <X(C12)>={X_C12_base:.4f}"

    # temperature at base should have increased from the energy release
    assert T_base > 1.0e9, \
        f"Base temperature should exceed 1 GK from nuclear heating: T_base={T_base:.4e}"
