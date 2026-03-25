"""Magneto-rotational instability test (Hopkins & Raives 2016)

Tests the growth of the MRI in a shearing box. The magnetic energy
should grow exponentially from the initial seed field as the MRI develops.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_mri(num_mpi_ranks):
    test_name = "mri"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")

    # Track magnetic energy over time
    times = []
    Emag_list = []
    for snap in snaps:
        with h5py.File(snap, "r") as F:
            t = F["Header"].attrs["Time"]
            B = F["PartType0/MagneticField"][:]
            mass = F["PartType0/Masses"][:]
            Emag = np.sum(np.sum(B**2, axis=1) * mass)
            times.append(t)
            Emag_list.append(Emag)

    times = np.array(times)
    Emag = np.array(Emag_list)

    # Plot magnetic energy evolution
    plt.figure()
    plt.semilogy(times, Emag / Emag[0], "o-")
    plt.xlabel("Time")
    plt.ylabel("E_mag / E_mag(0)")
    plt.title("MRI - Magnetic Energy Growth")
    plt.savefig(f"test/{test_name}/Emag_evolution.png")
    plt.close()

    # MRI should amplify the magnetic field
    assert Emag[-1] > 2 * Emag[0], (
        f"MRI did not amplify B-field enough: E_mag ratio = {Emag[-1]/Emag[0]:.2f}"
    )
