"""Interacting blast waves test (Woodward & Colella 1984; Hopkins 2015)

Compares the final snapshot against a high-resolution reference solution.
The exact solution file has columns: i-zone, x, density, V1, V2, V3, pressure.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py
from os import path
from gizmo.test import build_and_run_test, default_mpi_ranks


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
def test_interactblast(num_mpi_ranks):
    test_name = "interactblast"
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_010.hdf5"
    if not path.isfile(final_snap):
        raise RuntimeError("GIZMO did not run successfully.")

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        x_sim = F["PartType0/Coordinates"][:, 0]
        rho_sim = F["PartType0/Density"][:]
        vel_sim = F["PartType0/Velocities"][:, 0]

    # Load exact solution: i-zone, x, density, V1, V2, V3, pressure
    exact = np.loadtxt(f"test/{test_name}/interactblast_exact.txt")
    x_exact = exact[:, 1]
    rho_exact = exact[:, 2]
    vel_exact = exact[:, 3]

    # Interpolate exact solution to particle positions
    rho_interp = interp1d(x_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(x_sim)
    vel_interp = interp1d(x_exact, vel_exact, bounds_error=False, fill_value="extrapolate")(x_sim)

    # Compute L1 errors
    L1_rho = np.mean(np.abs(rho_sim - rho_interp)) / np.mean(np.abs(rho_interp))
    L1_vel = np.mean(np.abs(vel_sim - vel_interp)) / (np.mean(np.abs(vel_interp)) + 1e-10)

    # Plot comparison
    order = x_sim.argsort()
    for label, sim, exact_vals in [
        ("Density", rho_sim, rho_interp),
        ("Velocity", vel_sim, vel_interp),
    ]:
        plt.figure()
        plt.plot(x_sim[order], sim[order], ".", markersize=1, label="GIZMO")
        plt.plot(x_sim[order], exact_vals[order], "-", color="red", linewidth=0.5, label="Exact")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    assert L1_rho < 0.1, f"Density L1 error {L1_rho:.4f} exceeds tolerance"
    assert L1_vel < 0.15, f"Velocity L1 error {L1_vel:.4f} exceeds tolerance"
