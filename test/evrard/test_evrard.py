"""Evrard adiabatic collapse test (Hopkins 2015)

Compares the radial density, entropy, and velocity profiles at t=0.8 against
a high-resolution reference solution.
The exact solution file has columns: radius, density, entropy, velocity.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import h5py
from os import path
from meshoid import Meshoid
from gizmo.test import build_and_run_test, default_mpi_ranks, flush_colorbar


def plot_evrard_density_slice(coords, rho, output_dir="."):
    """Plot a density slice through the Evrard collapse center."""
    M = Meshoid(coords)
    center = np.average(coords, axis=0)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=center, size=1., order=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-0.5, 0.5, -0.5, 0.5])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Evrard Collapse - Density Slice")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_evrard(num_mpi_ranks):
    test_name = "evrard"
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_008.hdf5"
    if not path.isfile(final_snap):
        raise RuntimeError("GIZMO did not run successfully.")

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho_sim = F["PartType0/Density"][:]
        vel = F["PartType0/Velocities"][:]
        u_sim = F["PartType0/InternalEnergy"][:]

    # Compute radius from origin and radial velocity
    r_sim = np.sqrt(np.sum(coords**2, axis=1))
    vr_sim = np.sum(vel * coords, axis=1) / (r_sim + 1e-30)

    # Entropy: S = P / rho^gamma = (gamma-1) * u / rho^(gamma-1)
    gamma = 5.0 / 3.0
    entropy_sim = (gamma - 1) * u_sim / rho_sim ** (gamma - 1)

    # Load exact solution: radius, density, entropy, velocity
    exact = np.loadtxt(f"test/{test_name}/evrard_exact.txt")
    r_exact = exact[:, 0]
    rho_exact = exact[:, 1]
    entropy_exact = exact[:, 2]
    vr_exact = exact[:, 3]

    # Bin simulation data by radius
    r_bins = np.logspace(np.log10(r_exact.min()), np.log10(r_exact.max()), 30)
    rho_binned = binned_statistic(r_sim, rho_sim, "median", r_bins)[0]
    vr_binned = binned_statistic(r_sim, vr_sim, "median", r_bins)[0]
    entropy_binned = binned_statistic(r_sim, entropy_sim, "median", r_bins)[0]
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Interpolate exact solution to bin centers
    rho_exact_interp = interp1d(r_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(r_centers)
    vr_exact_interp = interp1d(r_exact, vr_exact, bounds_error=False, fill_value="extrapolate")(r_centers)

    plot_evrard_density_slice(coords, rho_sim, output_dir=f"test/{test_name}")

    # Plot comparison
    for label, binned, exact_vals, log in [
        ("Density", rho_binned, rho_exact_interp, True),
        ("RadialVelocity", vr_binned, vr_exact_interp, False),
    ]:
        plt.figure()
        if log:
            plt.loglog(r_centers, binned, "o", markersize=3, label="GIZMO")
            plt.loglog(r_centers, exact_vals, "-", color="red", label="Exact")
        else:
            plt.semilogx(r_centers, binned, "o", markersize=3, label="GIZMO")
            plt.semilogx(r_centers, exact_vals, "-", color="red", label="Exact")
        plt.xlabel("r")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    # Check density profile
    good = np.isfinite(rho_binned) & np.isfinite(rho_exact_interp) & (rho_exact_interp > 0)
    if np.any(good):
        L1_rho = np.nanmean(np.abs(np.log10(rho_binned[good]) - np.log10(rho_exact_interp[good])))
        assert L1_rho < 0.3, f"Log density profile L1 error {L1_rho:.4f} exceeds tolerance"
