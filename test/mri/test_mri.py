"""Magneto-rotational instability test (Hopkins & Raives 2016)

Tests the growth of the MRI in a shearing box. The magnetic energy
should grow exponentially from the initial seed field as the MRI develops.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_mri(num_mpi_ranks, num_omp_threads):
    test_name = "mri"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Track magnetic energy over time
    times = []
    Emag_list = []
    Emagx_list = []
    Emagy_list = []
    Emagz_list = []
    for snap in snaps:
        with h5py.File(snap, "r") as F:
            t = F["Header"].attrs["Time"]
            B = F["PartType0/MagneticField"][:]
            mass = F["PartType0/Masses"][:]
            rho = F["PartType0/Density"][:]
            vol = mass / rho
            Emag = np.sum(np.sum(B**2, axis=1) * vol)/(8.*np.pi)
            Emag_x = np.sum(B[:,0]**2 * vol)/(8.*np.pi)
            Emag_y = np.sum(B[:,2]**2 * vol)/(8.*np.pi) # note the unusual convention of the axes for the shearing box
            Emag_z = np.sum(B[:,1]**2 * vol)/(8.*np.pi)
            times.append(t)
            Emag_list.append(Emag)
            Emagx_list.append(Emag_x)
            Emagy_list.append(Emag_y)
            Emagz_list.append(Emag_z)

    times = np.array(times)
    Emag = np.array(Emag_list)
    Emag_x = np.array(Emagx_list)
    Emag_y = np.array(Emagy_list)
    Emag_z = np.array(Emagz_list)

    # Plot magnetic energy evolution
    plt.figure()
    plt.semilogy(times, Emag / Emag[0], "o-")
    plt.semilogy(times, Emag_x / Emag[0], "o-", label="B_x")
    plt.semilogy(times, Emag_y / Emag[0], "o-", label="B_y")
    plt.semilogy(times, Emag_z / Emag[0], "o-", label="B_z")
    plt.xlim(times[0], times[-1])
    plt.ylim(1.e-6, 1.1)
    plt.xlabel("Time")
    plt.ylabel("E_mag / E_mag(0)")
    plt.title("MRI - Magnetic Energy Growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/Emag_evolution.png")
    plt.close()

    # MRI should amplify the magnetic field
    assert Emag_y[-1] > 2 * Emag_y[1], (
        f"MRI did not amplify B-field enough: E_mag ratio = {Emag[-1]/Emag[0]:.2f}"
    )
