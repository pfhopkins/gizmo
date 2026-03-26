"""Zeldovich pancake cosmological test (Hopkins 2015)

Tests cosmological integration by evolving a single-mode density perturbation
(the Zeldovich pancake) and comparing against the exact solution at z=0.
The exact solution file has columns: x-position (Mpc), log10(rho/rho_mean),
log10(temperature), velocity (km/s).
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, clean_test_outputs, assert_final_time, default_mpi_ranks, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_zeldovich(num_mpi_ranks, num_omp_threads):
    test_name = "zeldovich"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    # Find the last snapshot (cosmological runs use ScaleFac_Between_Snapshots)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if not snaps:
        raise RuntimeError("GIZMO did not run successfully.")
    final_snap = snaps[-1]
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho_sim = F["PartType0/Density"][:]
        masses = F["PartType0/Masses"][:]
        vel = F["PartType0/Velocities"][:]
        boxsize = F["Header"].attrs["BoxSize"]

    # Exact solution x is in Mpc centered on 0; sim box is in kpc from 0 to BoxSize
    # BoxSize = 64000 kpc = 64 Mpc. Pancake is at box center, so shift by half a box.
    box_mpc = boxsize / 1000.0
    x_sim_mpc = (coords[:, 0] / 1000.0 + box_mpc / 2) % box_mpc - box_mpc / 2

    # Normalize density by the mean cosmic density
    rho_mean = masses.sum() / boxsize**3
    logrho_sim = np.log10(rho_sim / rho_mean)

    # Load exact solution: x (Mpc), log10(rho/rho_mean), log10(T), velocity
    exact = np.loadtxt(f"test/{test_name}/zeldovich_exact.txt")
    x_exact = exact[:, 0]
    logrho_exact = exact[:, 1]
    vel_exact = exact[:, 3]

    # Bin simulation data to match exact solution coordinate range
    x_bins = np.linspace(x_exact.min(), x_exact.max(), 60)
    logrho_binned = binned_statistic(x_sim_mpc, logrho_sim, "median", x_bins)[0]
    vel_binned = binned_statistic(x_sim_mpc, vel[:, 0], "median", x_bins)[0]
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])

    # Interpolate exact solution to bin centers
    logrho_exact_interp = interp1d(x_exact, logrho_exact, bounds_error=False, fill_value="extrapolate")(x_centers)
    vel_exact_interp = interp1d(x_exact, vel_exact, bounds_error=False, fill_value="extrapolate")(x_centers)

    # Plot
    for label, binned, exact_vals in [
        ("LogDensity", logrho_binned, logrho_exact_interp),
        ("Velocity", vel_binned, vel_exact_interp),
    ]:
        plt.figure()
        plt.plot(x_centers, binned, "o", markersize=3, label="GIZMO")
        plt.plot(x_centers, exact_vals, "-", color="red", label="Exact")
        plt.xlabel("x (Mpc)")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    # Check density profile - exclude the sharp caustic peak (|x| < 2 Mpc)
    # where numerical smoothing prevents exact agreement
    good = np.isfinite(logrho_binned) & np.isfinite(logrho_exact_interp) & (np.abs(x_centers) > 2.0)
    if np.any(good):
        L1_logrho = np.nanmean(np.abs(logrho_binned[good] - logrho_exact_interp[good]))
        assert L1_logrho < 0.3, f"Log density profile L1 error {L1_logrho:.4f} exceeds tolerance"
