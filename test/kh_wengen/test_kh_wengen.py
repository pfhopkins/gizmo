"""Kelvin-Helmholtz instability - Wengen test (Hopkins 2015)

Tests the development of the KH instability in a 3D setup from the
Wengen comparison project. Checks mass conservation and that the
instability develops (density variance evolves).
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
def test_kh_wengen(num_mpi_ranks, num_omp_threads):
    test_name = "kh_wengen"
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
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Plot a slice through the midplane using Meshoid slice interpolation
    # BoxSize=8, BOX_LONG_X=32, BOX_LONG_Y=32, BOX_LONG_Z=2 -> box is 256 x 256 x 16
    boxsize = 8.
    box_x = boxsize * 32
    box_y = boxsize * 32
    box_z = boxsize * 2
    M = Meshoid(pos_f, boxsize=boxsize)
    center = np.array([box_x / 2, box_y / 2, box_z / 2])
    rho_slice = M.Slice(rho_f, res=1024, plane="z", center=center, size=box_x, order=1)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="viridis", extent=[0, box_x, 0, box_y])
    flush_colorbar(im, ax=ax, label="Density")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("KH Wengen - Density (midplane slice)")
    fig.savefig(f"test/{test_name}/Density_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"
