"""Keplerian disk test (Hopkins 2015)

Tests a Keplerian disk with analytic gravity. The disk should maintain
its structure and orbital profile over several orbits.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_keplerian(num_mpi_ranks, num_omp_threads):
    test_name = "keplerian"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        pos0 = F["PartType0/Coordinates"][:]
        vel0 = F["PartType0/Velocities"][:]
        rho0 = F["PartType0/Density"][:]
        mass0 = F["PartType0/Masses"][:]
        boxsize = F["Header"].attrs["BoxSize"]
    with h5py.File(snaps[-1], "r") as F:
        pos_f = F["PartType0/Coordinates"][:]
        vel_f = F["PartType0/Velocities"][:]
        rho_f = F["PartType0/Density"][:]
        mass_f = F["PartType0/Masses"][:]

    center = boxsize / 2.0
    r0 = np.sqrt((pos0[:, 0] - center) ** 2 + (pos0[:, 1] - center) ** 2)
    rf = np.sqrt((pos_f[:, 0] - center) ** 2 + (pos_f[:, 1] - center) ** 2)

    # Compute azimuthal velocity v_phi = (x*vy - y*vx) / r
    vphi0 = ((pos0[:, 0] - center) * vel0[:, 1] - (pos0[:, 1] - center) * vel0[:, 0]) / (r0 + 1e-10)
    vphi_f = ((pos_f[:, 0] - center) * vel_f[:, 1] - (pos_f[:, 1] - center) * vel_f[:, 0]) / (rf + 1e-10)

    # Plot radial profile of azimuthal velocity
    plt.figure()
    rbins = np.linspace(0.5, 3.5, 30)
    from scipy.stats import binned_statistic

    vphi0_binned = binned_statistic(r0, vphi0, "median", rbins)[0]
    vphi_f_binned = binned_statistic(rf, vphi_f, "median", rbins)[0]
    rc = 0.5 * (rbins[:-1] + rbins[1:])
    plt.plot(rc, vphi0_binned, "b-", label="Initial")
    plt.plot(rc, vphi_f_binned, "r--", label="Final")
    plt.plot(rc, 1.0 / np.sqrt(rc), "k:", label=r"$v_\phi \propto r^{-1/2}$")
    plt.xlabel("r")
    plt.ylabel(r"$v_\phi$")
    plt.legend()
    plt.savefig(f"test/{test_name}/vphi_profile.png")
    plt.close()

    # The azimuthal velocity profile should remain close to Keplerian (v ~ r^-1/2)
    good = np.isfinite(vphi0_binned) & np.isfinite(vphi_f_binned) & (rc > 0.8) & (rc < 3.0)
    L1 = np.mean(np.abs(vphi_f_binned[good] - vphi0_binned[good]) / np.abs(vphi0_binned[good]))
    assert L1 < 0.15, f"Azimuthal velocity profile degraded: L1 relative error {L1:.4f}"

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"
