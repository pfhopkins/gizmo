"""Isolated disk galaxy test (Hopkins 2015)

Tests the evolution of an isolated disk galaxy with cooling. The disk
should remain stable and develop spiral structure. Checks that the
disk doesn't blow apart or lose too much mass from the disk plane.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from os import path, chdir
from gizmo.test import build_gizmo_for_test, download_test_files, run_test, default_mpi_ranks, clean_test_outputs, get_cooling_tables


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_isodisk(num_mpi_ranks):
    test_name = "isodisk"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name)
    chdir(f"test/{test_name}/")

    download_test_files(test_name)
    get_cooling_tables()

    run_test(test_name, num_mpi_ranks)
    chdir("../../")

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        pos0 = F["PartType0/Coordinates"][:]
        mass0 = F["PartType0/Masses"][:]
        boxsize = F["Header"].attrs["BoxSize"]
    with h5py.File(snaps[-1], "r") as F:
        pos_f = F["PartType0/Coordinates"][:]
        mass_f = F["PartType0/Masses"][:]
        rho_f = F["PartType0/Density"][:]

    center = boxsize / 2.0

    # Plot face-on view of the disk
    plt.figure(figsize=(6, 6))
    r_xy = np.sqrt((pos_f[:, 0] - center) ** 2 + (pos_f[:, 1] - center) ** 2)
    disk = (r_xy < 30) & (np.abs(pos_f[:, 2] - center) < 5)
    if np.sum(disk) > 100:
        plt.scatter(
            pos_f[disk, 0] - center, pos_f[disk, 1] - center,
            c=np.log10(rho_f[disk]), s=0.1, cmap="inferno"
        )
        plt.colorbar(label="log10(Density)")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.title("Isolated Disk - Face-on")
    plt.savefig(f"test/{test_name}/Density_faceon.png", dpi=150)
    plt.close()

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # Disk should still exist: most gas mass should be within the disk region
    r0 = np.sqrt(np.sum((pos0 - center) ** 2, axis=1))
    rf = np.sqrt(np.sum((pos_f - center) ** 2, axis=1))
    mass_in_disk0 = mass0[r0 < 50].sum()
    mass_in_disk_f = mass_f[rf < 50].sum()
    assert mass_in_disk_f > 0.8 * mass_in_disk0, (
        f"Disk lost too much mass: {mass_in_disk_f/mass_in_disk0:.2%} remaining"
    )
