"""Sedov-Taylor blast wave test (Hopkins 2015)

Compares radial profiles of the blast wave against the analytic Sedov solution.
The exact solution file has columns: radius, temperature, density, radial_velocity.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import h5py

from meshoid import Meshoid
from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    flush_colorbar,
    assert_final_time,
    get_final_snapshot,
    default_omp_threads,
)


def plot_sedov_density_slice(coords, rho, output_dir="."):
    """Plot a density slice through the Sedov blast center."""
    box_center = 3.0
    M = Meshoid(coords)
    center = np.array([box_center, box_center, box_center])
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=center, size=4.0, order=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-2, 2, -2, 2])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sedov Blast - Density Slice")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize(
    "num_mpi_ranks,num_omp_threads",
    [(default_mpi_ranks(), default_omp_threads())],
)
def test_sedov(num_mpi_ranks, num_omp_threads):
    test_name = "sedov"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho_sim = F["PartType0/Density"][:]
        vel = F["PartType0/Velocities"][:]

    # Compute radius from box center
    box_center = 3.0  # BoxSize/2
    r_sim = np.sqrt(np.sum((coords - box_center) ** 2, axis=1))
    vr_sim = np.sum(vel * (coords - box_center), axis=1) / (r_sim + 1e-30)

    # Load exact solution: radius, temperature, density, radial_velocity
    exact = np.loadtxt(f"test/{test_name}/sedov_exact.txt")
    r_exact = exact[:, 0]
    rho_exact = exact[:, 2]
    vr_exact = exact[:, 3]

    # Bin simulation data by radius and compute median
    r_bins = np.linspace(0, r_exact.max() * 0.9, 30)
    rho_binned = binned_statistic(r_sim, rho_sim, "median", r_bins)[0]
    vr_binned = binned_statistic(r_sim, vr_sim, "median", r_bins)[0]
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Interpolate exact solution to bin centers
    rho_exact_interp = interp1d(r_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(r_centers)
    vr_exact_interp = interp1d(r_exact, vr_exact, bounds_error=False, fill_value="extrapolate")(r_centers)

    plot_sedov_density_slice(coords, rho_sim, output_dir=f"test/{test_name}")

    # Plot comparison
    for label, binned, exact_vals in [
        ("Density", rho_binned, rho_exact_interp),
        ("RadialVelocity", vr_binned, vr_exact_interp),
    ]:
        plt.figure()
        plt.plot(r_centers, binned, "o", markersize=3, label="GIZMO")
        plt.plot(r_centers, exact_vals, "-", color="red", label="Exact")
        plt.xlabel("r")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    # Check that the binned density profile roughly matches the exact solution
    # Use bins where the exact solution has significant density (post-shock)
    good = rho_exact_interp > 0.01
    if np.any(good):
        L1_rho = np.nanmean(np.abs(rho_binned[good] - rho_exact_interp[good])) / np.nanmean(
            np.abs(rho_exact_interp[good])
        )
        assert L1_rho < 0.3, f"Density profile L1 error {L1_rho:.4f} exceeds tolerance"
