"""Test of a 120msun protostellar core with a r^-2 density profile; should have approximately constant, high accretion rate and a smooth profile in gas, radiation, and dust temperature"""

import pytest
from gizmo.test import (
    build_and_run_test,
    get_cooling_tables,
    assert_final_time,
    default_omp_threads,
    default_mpi_ranks,
    get_final_snapshot,
)

from matplotlib import pyplot as plt
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np


def plot_quantiles_vs_radius(radius, quantity, radius_bins=np.logspace(-4, -1, 21), plotargs={}, label=None):
    quantiles = [
        binned_statistic(radius, quantity, lambda x: np.percentile(x, q), radius_bins)[0] for q in (16, 50, 84)
    ]

    plt.loglog(np.sqrt(radius_bins[1:] * radius_bins[:-1]), quantiles[1], label=label, **plotargs)
    plt.fill_between(np.sqrt(radius_bins[1:] * radius_bins[:-1]), quantiles[0], quantiles[2], **plotargs, alpha=0.5)


def compute_test_statistic(f, save_reference_solution=False, plot=False, extra_config_flags=None):
    """Returns the test statistic to be compared with the reference solution. Optionally, saves the input snapshot data as the reference solution."""

    code_to_evcm3 = (u.m**2 / u.s**2 * u.Msun / u.pc**3).to(u.eV / u.cm**3)
    with h5py.File(f, "r") as F:
        xe = F["PartType0/ElectronAbundance"][:]
        rho = F["PartType0/Density"][:]
        T = F["PartType0/Temperature"][:]
        Trad = F["PartType0/IRBand_Radiation_Temperature"][:]
        Tdust = F["PartType0/Dust_Temperature"][:]
        urad_eV_cm3 = F["PartType0/PhotonEnergy"][:] * (rho / F["PartType0/Masses"][:])[:, None] * code_to_evcm3
        r = F["PartType0/Coordinates"][:] - F["PartType5/Coordinates"][0]
        r = np.sum(r * r, axis=1) ** 0.5

    if plot:
        with h5py.File("test/shu_M120/shu_M120_exact.hdf5", "r") as F:
            xe_ref = F["PartType0/ElectronAbundance"][:]
            rho_ref = F["PartType0/Density"][:]
            T_ref = F["PartType0/Temperature"][:]
            Trad_ref = F["PartType0/IRBand_Radiation_Temperature"][:]
            Tdust_ref = F["PartType0/Dust_Temperature"][:]
            urad_eV_cm3_ref = (
                F["PartType0/PhotonEnergy"][:] * (rho_ref / F["PartType0/Masses"][:])[:, None] * code_to_evcm3
            )
            r_ref = F["PartType0/Coordinates"][:] - F["PartType5/Coordinates"][0]
            r_ref = np.sum(r_ref * r_ref, axis=1) ** 0.5

        if extra_config_flags:
            plotname_suffix = "_" + "_".join(extra_config_flags)
        else:
            plotname_suffix = ""

        plot_quantiles_vs_radius(r_ref, T_ref, label="Benchmark")
        plot_quantiles_vs_radius(r, T, label="Test")
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig(f"test/shu_M120/r_vs_T{plotname_suffix}.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_radius(r_ref, Tdust_ref, label="Benchmark")
        plot_quantiles_vs_radius(r, Tdust, label="Test")
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(r"$T_{\rm dust} (\rm K)$")
        plt.legend(loc=3)
        plt.savefig(f"test/shu_M120/r_vs_Tdust{plotname_suffix}.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_radius(r_ref, Trad_ref, label="Benchmark")
        plot_quantiles_vs_radius(r, Trad, label="Test")
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(r"$T_{\rm rad} (\rm K)$")
        plt.legend(loc=3)
        plt.savefig(f"test/shu_M120/r_vs_Trad{plotname_suffix}.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_radius(r_ref, xe_ref, label="Benchmark")
        plot_quantiles_vs_radius(r, xe, label="Test")
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(r"$x_e$")
        plt.legend(loc=3)
        plt.savefig(f"test/shu_M120/r_vs_xe{plotname_suffix}.png", bbox_inches="tight")
        plt.close()

        plot_quantiles_vs_radius(r, urad_eV_cm3[:, 4], label="Test")
        plot_quantiles_vs_radius(r_ref, urad_eV_cm3_ref[:, 4], label="Benchmark")
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(r"$u_{\rm rad} (\rm eV\,cm^{-3})$")
        plt.legend(loc=3)
        plt.savefig(f"test/shu_M120/r_vs_urad{plotname_suffix}.png", bbox_inches="tight")
        plt.close()

    r_bins = np.logspace(-4, 0, 21)

    stat_names = ["T", "Tdust", "Trad"]
    stats_to_check = T, Tdust, Trad
    return {name: binned_statistic(r, s, "median", r_bins)[0] for name, s in zip(stat_names, stats_to_check)}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    [
        (),
        ("TRANSPORT_SUBCYCLE=10",),
        ("TRANSPORT_SUBCYCLE=10", "TRANSPORT_SUBCYCLE_COOLING"),
    ],
    ids=["baseline", "subcycle_rt", "subcycle_rt_cooling"],
)
def test_shu_M120(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "shu_M120"
    test_dir = "test/shu_M120"
    get_cooling_tables(test_dir)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)

    from gizmo.test import variant_output_dir
    test_snap = variant_output_dir(test_name, extra_config_flags) + "/snapshot_005.hdf5"
    test_stats = compute_test_statistic(test_snap, plot=True, extra_config_flags=extra_config_flags)
    benchmark_stats = compute_test_statistic(test_dir + "/shu_M120_exact.hdf5")
    for name in test_stats:
        assert test_stats[name] == pytest.approx(benchmark_stats[name], rel=0.1), \
            f"{name}: max rel diff = {np.max(np.abs(test_stats[name] - benchmark_stats[name]) / np.abs(benchmark_stats[name] + 1e-300)):.3f}"

    with h5py.File(test_snap, "r") as F:
        m_test = F["PartType5/Masses"][:]
    with h5py.File(test_dir + "/shu_M120_exact.hdf5", "r") as F:
        m_bench = F["PartType5/Masses"][:]

    assert len(m_test) == 1
    assert m_test[0] == pytest.approx(m_bench[0], rel=0.1)
