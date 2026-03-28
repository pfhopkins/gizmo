"""Noh implosion test (Hopkins 2015)

The Noh problem has an analytic solution: an inward-moving flow creates a strong
accretion shock. In 3D with gamma=5/3:
- Pre-shock: rho = (1 + t/r)^2, v_r = -1
- Post-shock (r < r_shock): rho = 64, v_r = 0
- Shock position: r_shock = t/3
"""

import pytest
import numpy as np
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import h5py

from meshoid import Meshoid
from gizmo.test import build_and_run_test, flush_colorbar, assert_final_time, default_mpi_ranks, default_omp_threads, get_final_snapshot


def plot_noh_density_slice(coords, rho, output_dir="."):
    """Plot a density slice through the Noh implosion center."""
    box_center = 3.0
    M = Meshoid(coords)
    center = np.array([box_center, box_center, box_center])
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=center, size=4., order=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-2, 2, -2, 2])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Noh Implosion - Density Slice")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_noh(num_mpi_ranks, num_omp_threads):
    test_name = "noh"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho_sim = F["PartType0/Density"][:]
        vel = F["PartType0/Velocities"][:]
        time = F["Header"].attrs["Time"]

    # Compute radius from box center
    box_center = 3.0  # BoxSize/2
    r_sim = np.sqrt(np.sum((coords - box_center) ** 2, axis=1))

    # Analytic solution
    r_shock = time / 3.0
    rho_analytic = np.where(r_sim < r_shock, 64.0, (1.0 + time / r_sim) ** 2)

    # Bin by radius
    r_bins = np.linspace(0.01, 2.5, 40)
    rho_binned = binned_statistic(r_sim, rho_sim, "median", r_bins)[0]
    rho_analytic_binned = binned_statistic(r_sim, rho_analytic, "median", r_bins)[0]
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    plot_noh_density_slice(coords, rho_sim, output_dir=f"test/{test_name}")

    # Plot
    plt.figure()
    plt.plot(r_centers, rho_binned, "o", markersize=3, label="GIZMO")
    plt.plot(r_centers, rho_analytic_binned, "-", color="red", label="Analytic")
    plt.xlabel("r")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"test/{test_name}/Density.png")
    plt.close()

    # Check post-shock density: median density inside shock should be close to 64
    post_shock = r_sim < r_shock * 0.8  # well inside the shock
    if np.sum(post_shock) > 10:
        median_post_shock_rho = np.median(rho_sim[post_shock])
        assert abs(median_post_shock_rho - 64.0) / 64.0 < 0.15, (
            f"Post-shock density {median_post_shock_rho:.1f} deviates >15% from analytic value 64"
        )
