"""Square advection test (Hopkins 2015)

Tests the advection of a density square through a periodic box. The square
should be preserved with minimal diffusion after one full advection period.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from meshoid import Meshoid
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, flush_colorbar, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_square(num_mpi_ranks, num_omp_threads):
    test_name = "square"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        rho0 = F["PartType0/Density"][:]
        ids0 = F["PartType0/ParticleIDs"][:]
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        ids_f = F["PartType0/ParticleIDs"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Plot final density using Meshoid slice interpolation
    M = Meshoid(pos_f, boxsize=1.)
    rho_slice = M.Slice(rho_f, res=1024, plane="z", center=np.array([0.5, 0.5, 0.5]), size=1., order=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="viridis", extent=[0, 1, 0, 1])
    flush_colorbar(im, ax=ax, label="Density")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Square Advection - Final Density")
    fig.savefig(f"test/{test_name}/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Sort both by particle ID for comparison
    order0 = ids0.argsort()
    order_f = ids_f.argsort()
    rho0_sorted = rho0[order0]
    rho_f_sorted = rho_f[order_f]

    # The density contrast should be preserved
    contrast0 = rho0_sorted.max() / rho0_sorted.min()
    contrast_f = rho_f_sorted.max() / rho_f_sorted.min()

    # MFM should preserve the density contrast very well
    assert contrast_f > 0.5 * contrast0, (
        f"Density contrast degraded too much: {contrast0:.2f} -> {contrast_f:.2f}"
    )

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"
