"""GMC cooling and chemistry test"""

import pytest
from gizmo.test import build_and_run_test, get_cooling_tables
from os import path
from matplotlib import pyplot as plt
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np


def plot_quantiles_vs_nH(nH, quantity, nH_bins=np.logspace(-1, 4, 21), plotargs={}, label=None):
    quantiles = [binned_statistic(nH, quantity, lambda x: np.percentile(x, q), nH_bins)[0] for q in (16, 50, 84)]

    plt.loglog(np.sqrt(nH_bins[1:] * nH_bins[:-1]), quantiles[1], label=label, **plotargs)
    plt.fill_between(np.sqrt(nH_bins[1:] * nH_bins[:-1]), quantiles[0], quantiles[2], **plotargs, alpha=0.5)


def compute_test_statistic(f, save_reference_solution=False, plot=False):
    """Returns the test statistic to be compared with the reference solution. Optionally, saves the input snapshot data as the reference solution."""

    code_to_evcm3 = (u.km**2 / u.s**2 * u.Msun / u.pc**3).to(u.eV / u.cm**3)
    with h5py.File(f, "r") as F:
        Z = F["PartType0/Metallicity"][:]
        XH = 1 - F["PartType0/Metallicity"][:, 0] - F["PartType0/Metallicity"][:, 1]
        rho_to_nH = XH * (u.Msun / u.pc**3).to(c.m_p / u.cm**3)
        xe = F["PartType0/ElectronAbundance"][:]
        rho = F["PartType0/Density"][:]
        nH = F["PartType0/Density"][:] * rho_to_nH
        T = F["PartType0/Temperature"][:]
        Trad = F["PartType0/IRBand_Radiation_Temperature"][:]
        Tdust = F["PartType0/Dust_Temperature"][:]
        urad_eV_cm3 = F["PartType0/PhotonEnergy"][:] * (rho / F["PartType0/Masses"][:])[:, None] * code_to_evcm3

    # if save_reference_solution:
    #     with h5py.File("gmc_cooling_rt_exact.hdf5", "w") as F:
    #         F.create_dataset("PartType0/Metallicity", data=Z)
    #         F.create_dataset("PartType0/Density", data=rho)
    #         F.create_dataset("PartType0/Temperature", data=T)

    if plot:
        with h5py.File("gmc_cooling_rt_exact.hdf5", "r") as F:
            XH = 1 - F["PartType0/Metallicity"][:, 0] - F["PartType0/Metallicity"][:, 1]
            xe_ref = F["PartType0/ElectronAbundance"][:]
            rho_ref = F["PartType0/Density"][:]
            nH_ref = rho_ref * rho_to_nH
            T_ref = F["PartType0/Temperature"][:]
            Trad_ref = F["PartType0/IRBand_Radiation_Temperature"][:]
            Tdust_ref = F["PartType0/Dust_Temperature"][:]
            urad_eV_cm3_ref = (
                F["PartType0/PhotonEnergy"][:] * (rho_ref / F["PartType0/Masses"][:])[:, None] * code_to_evcm3
            )

        plot_quantiles_vs_nH(nH_ref, T_ref, label="Benchmark")
        plot_quantiles_vs_nH(nH, T, label="Test")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling_rt/nH_vs_T.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_nH(nH_ref, Tdust_ref, label="Benchmark")
        plot_quantiles_vs_nH(nH, Tdust, label="Test")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T_{\rm dust} (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling_rt/nH_vs_Tdust.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_nH(nH_ref, Trad_ref, label="Benchmark")
        plot_quantiles_vs_nH(nH, Trad, label="Test")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T_{\rm rad} (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling_rt/nH_vs_Trad.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_nH(nH_ref, xe_ref, label="Benchmark")
        plot_quantiles_vs_nH(nH, xe, label="Test")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$x_e$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling_rt/nH_vs_xe.png", bbox_inches="tight")
        plt.close()

        for i, band in enumerate(("EUV", "FUV", "NUV", "ONIR", "FIR")):
            plot_quantiles_vs_nH(nH, urad_eV_cm3[:, i], label=band)
            # plt.loglog(nH, urad_eV_cm3[:, i], ".", markersize=0.3, label=band)
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$u_{\rm rad} (\rm eV\,cm^{-3})$")
        plt.ylim(1e-6, 10)
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling_rt/nH_vs_urad_test.png", bbox_inches="tight")
        plt.close()

        for i, band in enumerate(("EUV", "FUV", "NUV", "ONIR", "FIR")):
            plot_quantiles_vs_nH(nH_ref, urad_eV_cm3_ref[:, i], label=band)
            # plt.loglog(nH_ref, urad_eV_cm3_ref[:, i], ".", markersize=0.3, label=band)
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$u_{\rm rad} (\rm eV\,cm^{-3})$")
        plt.ylim(1e-6, 10)
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling_rt/nH_vs_urad_benchmark.png", bbox_inches="tight")
        plt.close()

    nH_bins = np.logspace(1, 3, 10)

    stats_to_check = T, Tdust, Trad, urad_eV_cm3[:, 1], urad_eV_cm3[:, 4], xe
    return np.array([binned_statistic(nH, s, "median", nH_bins)[0] for s in stats_to_check])


@pytest.mark.parametrize("num_mpi_ranks", (1, 2, 4))
def test_gmc_cooling_rt(num_mpi_ranks):
    test_name = "gmc_cooling_rt"
    get_cooling_tables()
    build_and_run_test(test_name, num_mpi_ranks)
    if not path.isfile("test/gmc_cooling_rt/output/snapshot_010.hdf5"):
        raise (RuntimeError("GIZMO did not run successfully."))

    test_stats = compute_test_statistic("test/gmc_cooling_rt/output/snapshot_010.hdf5", plot=True)
    benchmark_stats = compute_test_statistic("gmc_cooling_rt_exact.hdf5")
    assert np.all(np.isclose(test_stats, benchmark_stats, rtol=0.1))
