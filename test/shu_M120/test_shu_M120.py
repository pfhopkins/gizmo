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

    centers = np.sqrt(radius_bins[1:] * radius_bins[:-1])
    plt.loglog(centers, quantiles[1], label=label, **plotargs)
    fill_kwargs = {k: v for k, v in plotargs.items()
                   if k not in ("marker", "markersize", "markerfacecolor", "linestyle", "alpha")}
    plt.fill_between(centers, quantiles[0], quantiles[2], **fill_kwargs, alpha=0.2)


_VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_VARIANT_COLORS = ["C0", "C2", "C1", "C3", "C4", "C5", "C6", "C7"]


def _short(label, maxlen=30):
    return label if len(label) <= maxlen else label[: maxlen - 3] + "..."


_variant_data = {}


def _variant_label(extra_config_flags):
    return "+".join(extra_config_flags) if extra_config_flags else "baseline"


def _load_shu_data(f):
    code_to_evcm3 = (u.m**2 / u.s**2 * u.Msun / u.pc**3).to(u.eV / u.cm**3)
    with h5py.File(f, "r") as F:
        rho = F["PartType0/Density"][:]
        d = {
            "xe": F["PartType0/ElectronAbundance"][:],
            "T": F["PartType0/Temperature"][:],
            "Trad": F["PartType0/IRBand_Radiation_Temperature"][:],
            "Tdust": F["PartType0/Dust_Temperature"][:],
            "urad_FIR": F["PartType0/PhotonEnergy"][:][:, 4] * (rho / F["PartType0/Masses"][:]) * code_to_evcm3,
        }
        rvec = F["PartType0/Coordinates"][:] - F["PartType5/Coordinates"][0]
        d["r"] = np.sum(rvec * rvec, axis=1) ** 0.5
    return d


def _render_combined_shu_plots(test_dir):
    """Re-render r-vs-X plots with the reference solution + all accumulated variants overlaid."""
    ref = _load_shu_data(f"{test_dir}/shu_M120_exact.hdf5")
    plot_specs = [
        ("T", r"$T (\rm K)$", "r_vs_T.png"),
        ("Tdust", r"$T_{\rm dust} (\rm K)$", "r_vs_Tdust.png"),
        ("Trad", r"$T_{\rm rad} (\rm K)$", "r_vs_Trad.png"),
        ("xe", r"$x_e$", "r_vs_xe.png"),
        ("urad_FIR", r"$u_{\rm rad} (\rm eV\,cm^{-3})$", "r_vs_urad.png"),
    ]
    for field, ylabel, fname in plot_specs:
        plot_quantiles_vs_radius(ref["r"], ref[field], label="Benchmark", plotargs={"color": "black"})
        for i, (vlabel, d) in enumerate(_variant_data.items()):
            plot_quantiles_vs_radius(
                d["r"], d[field], label=_short(vlabel),
                plotargs={"marker": _VARIANT_MARKERS[i % len(_VARIANT_MARKERS)],
                          "markersize": max(10 - 2 * i, 4), "markerfacecolor": "none",
                          "linestyle": "none", "alpha": 0.85,
                          "color": _VARIANT_COLORS[i % len(_VARIANT_COLORS)]},
            )
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(ylabel)
        plt.legend(loc="best", fontsize="x-small")
        plt.savefig(f"{test_dir}/{fname}", bbox_inches="tight")
        plt.close()


def compute_test_statistic(f):
    """Returns the test statistic to be compared with the reference solution."""
    d = _load_shu_data(f)
    r_bins = np.logspace(-4, 0, 21)
    stat_names = ["T", "Tdust", "Trad"]
    return {name: binned_statistic(d["r"], d[name], "median", r_bins)[0] for name in stat_names}


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
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags, timeout=600)
    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)

    from gizmo.test import variant_output_dir
    test_snap = variant_output_dir(test_name, extra_config_flags) + "/snapshot_005.hdf5"
    test_stats = compute_test_statistic(test_snap)
    benchmark_stats = compute_test_statistic(test_dir + "/shu_M120_exact.hdf5")

    # Accumulate this variant and re-render combined comparison plots
    _variant_data[_variant_label(extra_config_flags)] = _load_shu_data(test_snap)
    _render_combined_shu_plots(test_dir)
    for name in test_stats:
        assert test_stats[name] == pytest.approx(benchmark_stats[name], rel=0.1), \
            f"{name}: max rel diff = {np.max(np.abs(test_stats[name] - benchmark_stats[name]) / np.abs(benchmark_stats[name] + 1e-300)):.3f}"

    with h5py.File(test_snap, "r") as F:
        m_test = F["PartType5/Masses"][:]
    with h5py.File(test_dir + "/shu_M120_exact.hdf5", "r") as F:
        m_bench = F["PartType5/Masses"][:]

    assert len(m_test) == 1
    assert m_test[0] == pytest.approx(m_bench[0], rel=0.1)
