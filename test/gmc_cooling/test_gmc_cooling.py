"""GMC cooling and chemistry test"""

from gizmo.test import build_and_run_test, get_cooling_tables
from os import path
from matplotlib import pyplot as plt
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np


def compute_test_statistic(f, save_reference_solution=False, plot=False):
    """Returns the test statistic to be compared with the reference solution. Optionally, saves the input snapshot data as the reference solution."""
    with h5py.File(f, "r") as F:
        Z = F["PartType0/Metallicity"][:]
        XH = 1 - F["PartType0/Metallicity"][:, 0] - F["PartType0/Metallicity"][:, 1]
        rho_to_nH = XH * (u.Msun / u.pc**3).to(c.m_p / u.cm**3)
        rho = F["PartType0/Density"][:]
        nH = F["PartType0/Density"][:] * rho_to_nH
        T = F["PartType0/Temperature"][:]

    if save_reference_solution:
        with h5py.File("gmc_cooling_exact.hdf5", "w") as F:
            F.create_dataset("PartType0/Metallicity", data=Z)
            F.create_dataset("PartType0/Density", data=rho)
            F.create_dataset("PartType0/Temperature", data=T)

    if plot:
        plt.loglog(nH, T, ".", markersize=1, color="black", label="Test")
        with h5py.File("gmc_cooling_exact.hdf5", "r") as F:
            nH = F["PartType0/Density"][:] * rho_to_nH
            T = F["PartType0/Temperature"][:]
        plt.loglog(nH, T, ".", markersize=1, color="red", label="Benchmark")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling/nH_vs_T.png", bbox_inches="tight")

    nH_bins = np.logspace(1, 3, 10)
    return binned_statistic(nH, T, "median", nH_bins)[0]


def test_gmc_cooling():
    test_name = "gmc_cooling"
    get_cooling_tables()
    build_and_run_test(test_name)
    if not path.isfile("output/snapshot_010.hdf5"):
        raise (RuntimeError("GIZMO did not run successfully."))

    test_stats = compute_test_statistic("output/snapshot_010.hdf5", plot=True)
    benchmark_stats = compute_test_statistic("gmc_cooling_exact.hdf5")
    assert np.all(np.isclose(test_stats, benchmark_stats, rtol=0.1))
