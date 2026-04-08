"""GMC cooling and chemistry test"""

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


def plot_quantiles_vs_nH(nH, quantity, nH_bins=np.logspace(-1, 4, 21), plotargs={}, label=None):
    quantiles = [binned_statistic(nH, quantity, lambda x: np.percentile(x, q), nH_bins)[0] for q in (16, 50, 84)]

    plt.loglog(np.sqrt(nH_bins[1:] * nH_bins[:-1]), quantiles[1], label=label, **plotargs)
    plt.fill_between(np.sqrt(nH_bins[1:] * nH_bins[:-1]), quantiles[0], quantiles[2], **plotargs, alpha=0.5)


_variant_data = {}


def _variant_label(extra_config_flags):
    return "+".join(extra_config_flags) if extra_config_flags else "baseline"


def _load_gmc_data(f):
    code_to_evcm3 = (u.km**2 / u.s**2 * u.Msun / u.pc**3).to(u.eV / u.cm**3)
    with h5py.File(f, "r") as F:
        XH = 1 - F["PartType0/Metallicity"][:, 0] - F["PartType0/Metallicity"][:, 1]
        rho_to_nH = XH * (u.Msun / u.pc**3).to(c.m_p / u.cm**3)
        rho = F["PartType0/Density"][:]
        urad = F["PartType0/PhotonEnergy"][:] * (rho / F["PartType0/Masses"][:])[:, None] * code_to_evcm3
        return {
            "nH": rho * rho_to_nH,
            "T": F["PartType0/Temperature"][:],
            "Trad": F["PartType0/IRBand_Radiation_Temperature"][:],
            "Tdust": F["PartType0/Dust_Temperature"][:],
            "xe": F["PartType0/ElectronAbundance"][:],
            "urad": urad,
        }


def _render_combined_gmc_plots(test_dir):
    """Re-render nH-vs-X plots with the reference solution + all accumulated variants overlaid."""
    ref = _load_gmc_data(f"{test_dir}/gmc_cooling_rt_exact.hdf5")
    plot_specs = [
        ("T", r"$T (\rm K)$", "nH_vs_T.png"),
        ("Tdust", r"$T_{\rm dust} (\rm K)$", "nH_vs_Tdust.png"),
        ("Trad", r"$T_{\rm rad} (\rm K)$", "nH_vs_Trad.png"),
        ("xe", r"$x_e$", "nH_vs_xe.png"),
    ]
    for field, ylabel, fname in plot_specs:
        plot_quantiles_vs_nH(ref["nH"], ref[field], label="Benchmark")
        for vlabel, d in _variant_data.items():
            plot_quantiles_vs_nH(d["nH"], d[field], label=vlabel)
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(ylabel)
        plt.legend(loc=3)
        plt.savefig(f"{test_dir}/{fname}", bbox_inches="tight")
        plt.close()
    # Per-band radiation: one figure per variant + reference (5 bands each)
    for label, d in [("benchmark", ref)] + list(_variant_data.items()):
        for i, band in enumerate(("EUV", "FUV", "NUV", "ONIR", "FIR")):
            plot_quantiles_vs_nH(d["nH"], d["urad"][:, i], label=band)
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$u_{\rm rad} (\rm eV\,cm^{-3})$")
        plt.ylim(1e-6, 10)
        plt.legend(loc=3)
        safe = label.replace("/", "_").replace("=", "_")
        plt.savefig(f"{test_dir}/nH_vs_urad_{safe}.png", bbox_inches="tight")
        plt.close()


def compute_test_statistic(f):
    """Returns the test statistic to be compared with the reference solution."""
    d = _load_gmc_data(f)
    nH_bins = np.logspace(1, 3, 10)
    stat_names = ["T", "Tdust", "Trad", "urad_FUV", "urad_FIR", "xe"]
    stats_to_check = (d["T"], d["Tdust"], d["Trad"], d["urad"][:, 1], d["urad"][:, 4], d["xe"])
    return {name: binned_statistic(d["nH"], s, "median", nH_bins)[0] for name, s in zip(stat_names, stats_to_check)}


_baseline_stats_cache = {}


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", [
    (),
    ("TRANSPORT_SUBCYCLE=10",),
    ("TRANSPORT_SUBCYCLE=10", "TRANSPORT_SUBCYCLE_COOLING"),
], ids=["baseline", "subcycle_rt", "subcycle_rt_cooling"])
def test_gmc_cooling_rt(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "gmc_cooling_rt"
    test_dir = "test/gmc_cooling_rt"
    get_cooling_tables(test_dir)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)

    from gizmo.test import variant_output_dir
    test_snap = variant_output_dir(test_name, extra_config_flags) + "/snapshot_010.hdf5"
    test_stats = compute_test_statistic(test_snap)

    # Accumulate this variant and re-render combined comparison plots
    _variant_data[_variant_label(extra_config_flags)] = _load_gmc_data(test_snap)
    _render_combined_gmc_plots(test_dir)
    if not extra_config_flags:
        # baseline: cache stats for subcycled variants, then compare against reference
        _baseline_stats_cache["stats"] = test_stats
        benchmark_stats = compute_test_statistic(test_dir + "/gmc_cooling_rt_exact.hdf5")
    else:
        # subcycled: compare against the baseline run
        benchmark_stats = _baseline_stats_cache.get("stats")
        if benchmark_stats is None:
            pytest.skip("baseline must run first")
    for name in test_stats:
        assert test_stats[name] == pytest.approx(benchmark_stats[name], rel=0.1), \
            f"{name}: max rel diff = {np.max(np.abs(test_stats[name] - benchmark_stats[name]) / np.abs(benchmark_stats[name] + 1e-300)):.3f}"
