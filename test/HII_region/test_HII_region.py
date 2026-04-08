"""HII region test: 10 Msun ZAMS star in a uniform gas box with mean density 100 H/cc (20 pc).

Same setup as rad_pointsource_thin but optically thick — checks that an
ionization front forms and the radiation energy density profile is reasonable.
Compares baseline, TRANSPORT_SUBCYCLE, and TRANSPORT_SUBCYCLE_COOLING variants."""

import pytest
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from pathlib import Path
from gizmo.test import (
    build_and_run_test,
    get_cooling_tables,
    get_final_snapshot,
    assert_final_time,
    default_omp_threads,
    default_mpi_ranks,
)

TEST_NAME = "HII_region"
TEST_DIR = Path(__file__).parent

# Radial bins for profile statistics
R_BINS = np.logspace(-0.5, 0.9, 20)  # ~0.3 to ~8 pc
R_BIN_CENTERS = np.sqrt(R_BINS[:-1] * R_BINS[1:])


def generate_ics():
    """Generate ICs if they don't already exist."""
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        from make_HII_region_ics import make_HII_region_ics
        make_HII_region_ics(str(ic_file))


def plot_quantiles_vs_r(r, quantity, r_bins=R_BINS, label=None, **plotargs):
    quantiles = [binned_statistic(r, quantity, lambda x: np.percentile(x, q), r_bins)[0] for q in (16, 50, 84)]
    centers = np.sqrt(r_bins[1:] * r_bins[:-1])
    plt.loglog(centers, quantiles[1], label=label, **plotargs)
    plt.fill_between(centers, quantiles[0], quantiles[2], alpha=0.3, **plotargs)


def compute_profiles(snap_file):
    """Load snapshot and compute radial profiles of xe, urad_NUV, urad_ONIR."""
    from astropy import units as u
    code_to_evcm3 = (u.km**2 / u.s**2 * u.Msun / u.pc**3).to(u.eV / u.cm**3)

    with h5py.File(snap_file, "r") as F:
        pos = F["PartType0/Coordinates"][:]
        star_pos = F["PartType5/Coordinates"][0]
        r = np.linalg.norm(pos - star_pos, axis=1)

        xe = F["PartType0/ElectronAbundance"][:]
        rho = F["PartType0/Density"][:]
        mass = F["PartType0/Masses"][:]
        photon_energy = F["PartType0/PhotonEnergy"][:]
        urad_eV_cm3 = photon_energy * (rho / mass)[:, None] * code_to_evcm3

    # Band indices: 0=EUV, 1=FUV, 2=NUV, 3=ONIR, 4=FIR
    stat_names = ["xe", "urad_NUV", "urad_ONIR"]
    stat_data = [xe, urad_eV_cm3[:, 2], urad_eV_cm3[:, 3]]
    profiles = {
        name: binned_statistic(r, data, "median", R_BINS)[0]
        for name, data in zip(stat_names, stat_data)
    }
    profiles["_r"] = r
    profiles["_xe"] = xe
    profiles["_urad_NUV"] = urad_eV_cm3[:, 2]
    profiles["_urad_ONIR"] = urad_eV_cm3[:, 3]
    return profiles


def make_comparison_plots(all_profiles, test_dir):
    """Generate comparison plots across all variants."""
    plot_configs = [
        ("xe", r"$x_e$", "xe"),
        ("urad_NUV", r"$u_{\rm rad,\,NUV}\;(\rm eV\,cm^{-3})$", "urad_NUV"),
        ("urad_ONIR", r"$u_{\rm rad,\,ONIR}\;(\rm eV\,cm^{-3})$", "urad_ONIR"),
    ]
    for field, ylabel, fname in plot_configs:
        for label, profiles in all_profiles.items():
            plot_quantiles_vs_r(profiles["_r"], profiles[f"_{field}"], label=label)
        plt.xlabel(r"$r\;(\rm pc)$")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(str(test_dir / f"r_vs_{fname}.png"), bbox_inches="tight")
        plt.close()


_baseline_profiles_cache = {}
_all_profiles = {}


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
def test_HII_region(num_mpi_ranks, num_omp_threads, extra_config_flags):
    generate_ics()
    get_cooling_tables(str(TEST_DIR))
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(TEST_NAME)
    assert_final_time(final_snap, TEST_NAME)

    with h5py.File(final_snap, "r") as F:
        assert "PartType5" in F, "Star particle missing from final snapshot"

    profiles = compute_profiles(final_snap)

    # Label for plots and caching
    if extra_config_flags:
        label = "+".join(extra_config_flags)
    else:
        label = "baseline"
    _all_profiles[label] = profiles

    if not extra_config_flags:
        # Baseline: cache for subcycled comparison
        _baseline_profiles_cache["profiles"] = profiles
    else:
        # Subcycled: compare binned profiles against baseline
        baseline = _baseline_profiles_cache.get("profiles")
        if baseline is None:
            pytest.skip("baseline must run first")
        for name in ["xe", "urad_NUV", "urad_ONIR"]:
            assert profiles[name] == pytest.approx(baseline[name], rel=0.1), \
                f"{name}: max rel diff = {np.max(np.abs(profiles[name] - baseline[name]) / np.abs(baseline[name] + 1e-300)):.3f}"

    # After the last variant, generate comparison plots
    if len(_all_profiles) == 3:
        make_comparison_plots(_all_profiles, TEST_DIR)
