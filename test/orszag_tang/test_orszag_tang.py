"""Orszag-Tang MHD vortex test (Hopkins & Raives 2016)

Classic 2D MHD turbulence problem. Tests the development of MHD shocks from
smooth initial conditions. Since there is no exact solution, this test verifies
that the simulation runs and checks energy conservation.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
from os import path
from gizmo.test import build_and_run_test, default_mpi_ranks


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_orszag_tang(num_mpi_ranks):
    test_name = "orszag_tang"
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    final_snap = outputdir + "/snapshot_005.hdf5"
    init_snap = outputdir + "/snapshot_000.hdf5"
    if not path.isfile(final_snap):
        raise RuntimeError("GIZMO did not run successfully.")

    def compute_total_energy(snapfile):
        with h5py.File(snapfile, "r") as F:
            mass = F["PartType0/Masses"][:]
            vel = F["PartType0/Velocities"][:]
            u = F["PartType0/InternalEnergy"][:]
            B = F["PartType0/MagneticField"][:]
            rho = F["PartType0/Density"][:]
        KE = 0.5 * np.sum(mass[:, None] * vel**2)
        TE = np.sum(mass * u)
        # Magnetic energy: B^2 / (8*pi) * volume, where volume = mass/rho
        ME = np.sum(np.sum(B**2, axis=1) / (8 * np.pi) * mass / rho)
        return KE + TE + ME

    E_init = compute_total_energy(init_snap)
    E_final = compute_total_energy(final_snap)

    # Plot density at final time
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=np.log10(rho), s=0.1, cmap="viridis")
    plt.colorbar(label="log10(Density)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"test/{test_name}/Density_2D.png", dpi=150)
    plt.close()

    # Energy should be conserved to within ~10% (shock dissipation is expected)
    dE = abs(E_final - E_init) / abs(E_init)
    assert dE < 0.1, f"Total energy changed by {dE:.4f} (>10%)"
