"""GMC cooling and chemistry test"""

from gizmo.test import build_and_run_test, get_cooling_tables
from os import path
from matplotlib import pyplot as plt
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np
import pytest


def compute_test_statistic(f, save_reference_solution=False, plot=False):
    """Returns the test statistic to be compared with the reference solution. Optionally, saves the input snapshot data as the reference solution."""

    # get data required to plot n_H vs. T
    with h5py.File(f, "r") as F:
        Z = F["PartType0/Metallicity"][:]
        XH = 1 - F["PartType0/Metallicity"][:, 0] - F["PartType0/Metallicity"][:, 1]
        rho_to_nH = XH * (u.Msun / u.pc**3).to(c.m_p / u.cm**3)
        rho = F["PartType0/Density"][:]
        nH = F["PartType0/Density"][:] * rho_to_nH
        T = F["PartType0/Temperature"][:]

    if save_reference_solution:
        # this option will save the current test solution as the reference solution
        with h5py.File("test/gmc_cooling/gmc_cooling_exact.hdf5", "w") as F:
            F.create_dataset("PartType0/Metallicity", data=Z)
            F.create_dataset("PartType0/Density", data=rho)
            F.create_dataset("PartType0/Temperature", data=T)

    if plot:  # generate a plot of n_H vs T for the test and benchmark solutions
        with h5py.File("test/gmc_cooling/gmc_cooling_exact.hdf5", "r") as F:
            nH_ref = F["PartType0/Density"][:] * rho_to_nH
            T_ref = F["PartType0/Temperature"][:]
        plt.loglog(nH_ref, T_ref, ".", markersize=0.3, color="red", label="Benchmark")
        plt.loglog(nH, T, ".", markersize=0.3, color="black", label="Test")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling/nH_vs_T.png", bbox_inches="tight")
        plt.close()

    # set logarithmic bins for n_H in which to measure the median temperature
    nH_bins = np.logspace(1, 3, 10)

    # return nH-binned median temperature
    return binned_statistic(nH, T, "median", nH_bins)[0]


@pytest.mark.parametrize("num_mpi_ranks", (1, 16))
def test_gmc_cooling(num_mpi_ranks):
    # specify the test name
    test_name = "gmc_cooling"
    test_directory = "test/gmc_cooling"

    # download necessary cooling tables (needed for tests with COOLING flag)
    get_cooling_tables(test_directory)

    # build GIZMO and run the test
    build_and_run_test(test_name, num_mpi_ranks)

    # Check that the specific required output file exists:
    if not path.isfile("test/gmc_cooling/output/snapshot_010.hdf5"):
        raise (RuntimeError("GIZMO did not run successfully."))

    # Compute a test statistic from the output
    test_stats = compute_test_statistic("test/gmc_cooling/output/snapshot_010.hdf5", plot=True)

    # compute that same test statistic from a benchmark snapshot
    benchmark_stats = compute_test_statistic("test/gmc_cooling/gmc_cooling_exact.hdf5")

    # check that the test and benchmark agree within 10%
    assert np.all(np.isclose(test_stats, benchmark_stats, rtol=0.1))
