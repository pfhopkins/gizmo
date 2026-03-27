"""Orszag-Tang MHD vortex test (Hopkins & Raives 2016)

Classic 2D MHD turbulence problem. Tests the development of MHD shocks from
smooth initial conditions. Since there is no exact solution, this test verifies
that the simulation runs and checks energy conservation.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py

from meshoid import Meshoid
from gizmo.test import build_and_run_test, flush_colorbar, assert_final_time, default_mpi_ranks, default_omp_threads, get_final_snapshot


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_orszag_tang(num_mpi_ranks, num_omp_threads):
    test_name = "orszag_tang"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    final_snap = get_final_snapshot(test_name)
    init_snap = outputdir + "/snapshot_000.hdf5"
    assert_final_time(final_snap, test_name)

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

    # Plot density at final time using Meshoid slice interpolation
    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
    M = Meshoid(coords, boxsize=1.)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z",center=np.array([0.5,0.5,0.5]),size=1.,order=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="viridis", extent=[coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max()])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(f"test/{test_name}/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Energy should be conserved to within ~10% (shock dissipation is expected)
    dE = abs(E_final - E_init) / abs(E_init)
    assert dE < 0.1, f"Total energy changed by {dE:.4f} (>10%)"
