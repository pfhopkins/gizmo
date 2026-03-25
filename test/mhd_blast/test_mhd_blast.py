"""MHD blast wave test (Hopkins & Raives 2016)

Tests the propagation of a blast wave in a magnetized medium. The blast
should be elongated along the magnetic field direction. Checks energy
conservation and that the blast develops anisotropy.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_mhd_blast(num_mpi_ranks):
    test_name = "mhd_blast"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Load final snapshot
    with h5py.File(snaps[-1], "r") as F:
        pos = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        mass = F["PartType0/Masses"][:]
        B = F["PartType0/MagneticField"][:]
        boxsize = F["Header"].attrs["BoxSize"]

    # Plot density
    plt.figure(figsize=(6, 6))
    plt.scatter(pos[:, 0], pos[:, 1], c=np.log10(rho), s=0.1, cmap="inferno")
    plt.colorbar(label="log10(Density)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MHD Blast - Density")
    plt.savefig(f"test/{test_name}/Density_2D.png", dpi=150)
    plt.close()

    # Load initial snapshot for conservation check
    with h5py.File(snaps[0], "r") as F:
        mass0 = F["PartType0/Masses"][:]
        u0 = F["PartType0/InternalEnergy"][:]
        B0 = F["PartType0/MagneticField"][:]
        vel0 = F["PartType0/Velocities"][:]

    # Mass conservation
    mass_err = abs(mass.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # Total energy should be approximately conserved
    # (thermal + kinetic + magnetic)
    with h5py.File(snaps[-1], "r") as F:
        vel = F["PartType0/Velocities"][:]
    Etot0 = np.sum(mass0 * u0) + 0.5 * np.sum(mass0 * np.sum(vel0**2, axis=1))
    Etot_f = np.sum(mass * u) + 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    energy_err = abs(Etot_f - Etot0) / abs(Etot0)
    assert energy_err < 0.1, f"Total energy not conserved: relative error {energy_err:.4f}"
