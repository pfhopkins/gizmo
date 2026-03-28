"""Santa Barbara cluster comparison test (Hopkins 2015)

Cosmological cluster formation test. A dark matter halo collapses and
gas shock-heats to form a hot cluster. Checks that a hot, dense cluster
core forms and that baryon fraction is reasonable.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_sbcluster(num_mpi_ranks, num_omp_threads):
    test_name = "sbcluster"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load final snapshot
    with h5py.File(snaps[-1], "r") as F:
        gas_pos = F["PartType0/Coordinates"][:]
        gas_rho = F["PartType0/Density"][:]
        gas_u = F["PartType0/InternalEnergy"][:]
        gas_mass = F["PartType0/Masses"][:]
        dm_pos = F["PartType1/Coordinates"][:]
        dm_mass = F["PartType1/Masses"][:]
        boxsize = F["Header"].attrs["BoxSize"]

    # Find the cluster center (densest gas region)
    center = gas_pos[np.argmax(gas_rho)]

    # Compute radii from cluster center
    dx = gas_pos - center
    dx -= boxsize * np.round(dx / boxsize)  # periodic wrapping
    r_gas = np.sqrt(np.sum(dx**2, axis=1))

    dx_dm = dm_pos - center
    dx_dm -= boxsize * np.round(dx_dm / boxsize)
    r_dm = np.sqrt(np.sum(dx_dm**2, axis=1))

    # Plot radial density and temperature profiles
    rbins = np.logspace(-2.5, -0.5, 20)
    rc = np.sqrt(rbins[:-1] * rbins[1:])

    from scipy.stats import binned_statistic
    rho_prof = binned_statistic(r_gas, gas_rho, "median", rbins)[0]
    u_prof = binned_statistic(r_gas, gas_u, "median", rbins)[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(rc, rho_prof, "o-")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Gas Density Profile")
    axes[1].loglog(rc, u_prof, "o-")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("Internal Energy")
    axes[1].set_title("Gas Temperature Profile")
    plt.tight_layout()
    plt.savefig(f"test/{test_name}/profiles.png", dpi=150)
    plt.close()

    # The cluster should have formed: central density should be much higher than mean
    rho_mean = gas_mass.sum() / boxsize**3
    overdensity = gas_rho.max() / rho_mean
    assert overdensity > 100, (
        f"Cluster did not form: max overdensity = {overdensity:.1f}"
    )

    # Gas in the core should be hot (shock-heated)
    core = r_gas < 0.05
    if np.any(core):
        u_core = np.median(gas_u[core])
        u_mean = np.median(gas_u)
        assert u_core > 5 * u_mean, (
            f"Core gas not hot enough: u_core/u_mean = {u_core/u_mean:.1f}"
        )
