"""HII region test: 10 Msun ZAMS star in a uniform gas box with mean density 100 H/cc (20 pc).

Same setup as rad_pointsource_thin but optically thick — checks that an
ionization front forms and the radiation energy density profile is reasonable.
Compares baseline, TRANSPORT_SUBCYCLE, and TRANSPORT_SUBCYCLE_COOLING variants."""

import pytest
import numpy as np
import h5py
from glob import glob
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
    variant_output_dir,
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


_VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _short(label, maxlen=30):
    return label if len(label) <= maxlen else label[: maxlen - 3] + "..."


def plot_quantiles_vs_r(r, quantity, r_bins=R_BINS, label=None, marker=None, markersize=6, **plotargs):
    quantiles = [binned_statistic(r, quantity, lambda x: np.percentile(x, q), r_bins)[0] for q in (16, 50, 84)]
    centers = np.sqrt(r_bins[1:] * r_bins[:-1])
    if marker is not None:
        plt.loglog(centers, quantiles[1], marker=marker, markersize=markersize, markerfacecolor="none",
                   linestyle="none", label=label, alpha=0.8, **plotargs)
    else:
        plt.loglog(centers, quantiles[1], label=label, **plotargs)
    plt.fill_between(centers, quantiles[0], quantiles[2], alpha=0.15, **plotargs)


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
        for i, (label, profiles) in enumerate(all_profiles.items()):
            # Decreasing marker size + open markers so overlapping points remain visible
            ms = 10 - 2 * i
            plot_quantiles_vs_r(
                profiles["_r"], profiles[f"_{field}"], label=_short(label),
                marker=_VARIANT_MARKERS[i % len(_VARIANT_MARKERS)], markersize=max(ms, 4),
            )
        plt.xlabel(r"$r\;(\rm pc)$")
        plt.ylabel(ylabel)
        plt.legend(loc="best", fontsize="x-small")
        plt.savefig(str(test_dir / f"r_vs_{fname}.png"), bbox_inches="tight")
        plt.close()


_baseline_profiles_cache = {}
_all_profiles = {}
_all_ifront_evolution = {}


def compute_ifront_radius(snap_file):
    """Return (time, r_ifront) for a snapshot. r_ifront is where binned-mean
    PartType0/HII crosses 0.5 (linear interp in log r); NaN if no crossing."""
    with h5py.File(snap_file, "r") as F:
        time = float(F["Header"].attrs["Time"])
        if "PartType5" not in F:
            return time, np.nan
        pos = F["PartType0/Coordinates"][:]
        star_pos = F["PartType5/Coordinates"][0]
        r = np.linalg.norm(pos - star_pos, axis=1)
        if "PartType0/HII" not in F:
            return time, np.nan
        hii = F["PartType0/HII"][:]
    mean_hii, _, _ = binned_statistic(r, hii, "mean", R_BINS)
    centers = R_BIN_CENTERS
    valid = np.isfinite(mean_hii)
    if valid.sum() < 2:
        return time, np.nan
    mh = mean_hii[valid]
    cc = centers[valid]
    # Find rightmost bin where mean_hii >= 0.5 followed by < 0.5
    above = mh >= 0.5
    if not above.any() or above.all():
        return time, np.nan
    # last index where above is True and next is False
    idx = np.where(above[:-1] & ~above[1:])[0]
    if len(idx) == 0:
        return time, np.nan
    i = idx[-1]
    # linear interp in log(r)
    x0, x1 = np.log(cc[i]), np.log(cc[i + 1])
    y0, y1 = mh[i], mh[i + 1]
    r_if = np.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return time, r_if


def compute_ifront_evolution(test_name, extra_config_flags=()):
    """Compute (times, r_ifront) over all snapshots in the variant output directory."""
    snaps = sorted(glob(f"{variant_output_dir(test_name, extra_config_flags)}/snapshot_*.hdf5"))
    times, radii = [], []
    for s in snaps:
        t, r = compute_ifront_radius(s)
        times.append(t)
        radii.append(r)
    return np.array(times), np.array(radii)


def make_ifront_evolution_plot(all_evo, test_dir):
    """Plot ionization-front radius vs time for each flag combination."""
    plt.figure()
    for label, (times, radii) in all_evo.items():
        plt.plot(times, radii, marker="o", label=label)
    plt.xlabel(r"$t$ (code units)")
    plt.ylabel(r"$r_{\rm IF}$ (pc)")
    plt.legend()
    plt.savefig(str(test_dir / "ifront_evolution.png"), bbox_inches="tight")
    plt.close()


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
    final_snap = get_final_snapshot(TEST_NAME, extra_config_flags)
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
    # Capture ionization-front time evolution before the next variant overwrites output/
    _all_ifront_evolution[label] = compute_ifront_evolution(TEST_NAME, extra_config_flags)

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
        make_ifront_evolution_plot(_all_ifront_evolution, TEST_DIR)
