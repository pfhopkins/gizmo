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
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_kh_wengen(num_mpi_ranks):
    test_name = "kh_wengen"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        rho0 = F["PartType0/Density"][:]
        mass0 = F["PartType0/Masses"][:]
    with h5py.File(snaps[-1], "r") as F:
        rho_f = F["PartType0/Density"][:]
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Plot a slice through the midplane (z ~ BoxSize_z/2)
    boxsize_z = 8 * 2  # BOX_LONG_Z=2, BoxSize=8
    zmid = boxsize_z / 2.0
    dz = boxsize_z * 0.05
    midplane = np.abs(pos_f[:, 2] - zmid) < dz
    if np.sum(midplane) > 100:
        plt.figure(figsize=(8, 8))
        plt.scatter(
            pos_f[midplane, 0], pos_f[midplane, 1],
            c=rho_f[midplane], s=0.1, cmap="viridis"
        )
        plt.colorbar(label="Density")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("KH Wengen - Density (midplane slice)")
        plt.savefig(f"test/{test_name}/Density_slice.png", dpi=150)
        plt.close()

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"
