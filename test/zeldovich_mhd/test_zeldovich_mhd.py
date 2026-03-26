"""MHD Zeldovich pancake test (Hopkins & Raives 2016)

Cosmological pancake collapse with an initial magnetic field. The field
should be compressed and amplified in the pancake. Checks density
structure and magnetic field amplification.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_zeldovich_mhd(num_mpi_ranks, num_omp_threads):
    test_name = "zeldovich_mhd"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[1], "r") as F:
        B0 = F["PartType0/MagneticField"][:]
        rho0 = F["PartType0/Density"][:]
    with h5py.File(snaps[-1], "r") as F:
        Bf = F["PartType0/MagneticField"][:]
        rho_f = F["PartType0/Density"][:]
        pos_f = F["PartType0/Coordinates"][:]
        mass_f = F["PartType0/Masses"][:]
        boxsize = F["Header"].attrs["BoxSize"]

    # Plot density and |B| vs x
    x = pos_f[:, 0]
    Bmag = np.sqrt(np.sum(Bf**2, axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = x.argsort()
    axes[0].semilogy(x[order], rho_f[order], ".", markersize=1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Density")
    axes[1].semilogy(x[order], Bmag[order], ".", markersize=1)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|B|")
    axes[1].set_title("Magnetic Field")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/profiles.png", dpi=150)
    plt.close()

    # Pancake should have formed: density contrast should be significant
    rho_contrast = rho_f.max() / rho_f.min()
    assert rho_contrast > 10, (
        f"Pancake did not form: density contrast = {rho_contrast:.1f}"
    )

    # Magnetic field should be amplified in the dense pancake
    Bmag0_max = np.max(np.sqrt(np.sum(B0**2, axis=1)))
    Bmagf_max = np.max(Bmag)
    assert Bmagf_max > Bmag0_max, "Magnetic field should be amplified in the pancake"
