"""Blob test - wind-cloud interaction (Hopkins 2015)

Tests the survival of a dense cloud in a wind. Checks that the cloud
retains a significant fraction of its mass after a few cloud-crushing times.
"""

import pytest
import numpy as np
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
def test_blob(num_mpi_ranks):
    test_name = "blob"
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

    # The blob is the overdense region (density > 5x mean)
    rho_mean0 = np.mean(rho0)
    blob_mass_initial = mass0[rho0 > 5 * rho_mean0].sum()
    total_mass_initial = mass0.sum()
    total_mass_final = mass_f.sum()

    # Mass conservation
    mass_err = abs(total_mass_final - total_mass_initial) / total_mass_initial
    assert mass_err < 1e-3, f"Total mass not conserved: relative error {mass_err:.6f}"

    # Blob should still have significant mass (not fully disrupted at t=2)
    rho_mean_f = np.mean(rho_f)
    blob_mass_final = mass_f[rho_f > 5 * rho_mean_f].sum()
    assert blob_mass_final > 0.1 * blob_mass_initial, (
        f"Blob lost too much mass: {blob_mass_final/blob_mass_initial:.2%} remaining"
    )
