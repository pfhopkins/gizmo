"""Shared procedure for the HII_region test family.

This module is the single source of truth for the HII-region test *procedure*:
loading snapshots, computing radial profiles, the physical temperature check, the
plots, and the build+run+check driver. The concrete tests

  - HII_region            (baseline: full RT band set)
  - HII_region_subcycle   (TRANSPORT_SUBCYCLE variants vs baseline)
  - HII_region_simple     (ionizing band only, radiation pressure off)

each supply their own Config.sh / params and call into here, so the analysis and
assertions stay identical across variants. It is a plain (non-test_*) module so it
can be imported from sibling test directories without pytest collection clashes.
"""

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
    variant_output_dir,
)

# Directory of the canonical HII_region test (this file lives there); it owns the ICs
# and cooling tables that the derived tests share.
HII_REGION_DIR = Path(__file__).parent
HII_REGION_NAME = "HII_region"

# Physical HII-region temperature window (photoionization equilibrium at ~solar Z).
T_HII_MIN = 7000.0
T_HII_MAX = 11000.0

# Radial bins for profile statistics
R_BINS = np.logspace(-0.5, 0.9, 20)  # ~0.3 to ~8 pc
R_BIN_CENTERS = np.sqrt(R_BINS[:-1] * R_BINS[1:])

_VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_VARIANT_COLORS = ["C0", "C2", "C1", "C3", "C4", "C5", "C6", "C7"]


# --------------------------------------------------------------------------------------
# Input setup (ICs + cooling tables)
# --------------------------------------------------------------------------------------
def generate_ics():
    """Generate the canonical HII_region ICs in HII_REGION_DIR if absent."""
    import sys
    ic_file = HII_REGION_DIR / f"{HII_REGION_NAME}_ics.hdf5"
    if not ic_file.exists():
        if str(HII_REGION_DIR) not in sys.path:
            sys.path.insert(0, str(HII_REGION_DIR))
        from make_HII_region_ics import make_HII_region_ics
        make_HII_region_ics(str(ic_file))


def prepare_inputs(test_dir, test_name):
    """Ensure the ICs and cooling tables needed to run <test_name> exist in test_dir.

    For the canonical HII_region this generates them in place. For a derived test
    (different directory) it symlinks the ICs and cooling tables back to HII_region,
    so every variant runs on byte-identical inputs without duplicating large files."""
    test_dir = Path(test_dir)
    generate_ics()                        # make sure the shared source IC exists
    get_cooling_tables(str(HII_REGION_DIR))
    if test_dir.resolve() == HII_REGION_DIR.resolve():
        get_cooling_tables(str(test_dir))
        return
    # derived test: symlink shared inputs from HII_region
    links = {
        f"{test_name}_ics.hdf5": HII_REGION_DIR / f"{HII_REGION_NAME}_ics.hdf5",
        "spcool_tables": HII_REGION_DIR / "spcool_tables",
        "TREECOOL": HII_REGION_DIR / "TREECOOL",
    }
    for name, target in links.items():
        link = test_dir / name
        if not target.exists():
            continue
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(Path("..") / HII_REGION_NAME / target.name)


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------
def compute_profiles(snap_file):
    """Load a snapshot and compute radial (binned-median) profiles of xe, urad_NUV,
    urad_ONIR, and T. Robust to the reduced band set of the 'simple' config: if the
    non-ionizing bands are absent (PhotonEnergy is 1-D / single band), the urad_NUV and
    urad_ONIR profiles come back as NaN rather than raising."""
    from astropy import units as u
    code_to_evcm3 = (u.km**2 / u.s**2 * u.Msun / u.pc**3).to(u.eV / u.cm**3)

    with h5py.File(snap_file, "r") as F:
        pos = F["PartType0/Coordinates"][:]
        star_pos = F["PartType5/Coordinates"][0]
        r = np.linalg.norm(pos - star_pos, axis=1)
        xe = F["PartType0/ElectronAbundance"][:]
        T_gas = F["PartType0/Temperature"][:]
        rho = F["PartType0/Density"][:]
        mass = F["PartType0/Masses"][:]
        photon_energy = F["PartType0/PhotonEnergy"][:]

    # Band indices (full config): 0=EUV, 1=FUV, 2=NUV, 3=ONIR, 4=FIR. The 'simple'
    # config keeps only the ionizing band, so PhotonEnergy may be 1-D.
    n_bands = photon_energy.shape[1] if photon_energy.ndim > 1 else 1
    if photon_energy.ndim > 1:
        urad_eV_cm3 = photon_energy * (rho / mass)[:, None] * code_to_evcm3
    else:
        urad_eV_cm3 = None

    def _band(idx):
        if urad_eV_cm3 is None or idx >= n_bands:
            return np.full(r.shape, np.nan)
        return urad_eV_cm3[:, idx]

    urad_NUV, urad_ONIR = _band(2), _band(3)
    stat_names = ["xe", "urad_NUV", "urad_ONIR", "T"]
    stat_data = [xe, urad_NUV, urad_ONIR, T_gas]
    profiles = {
        name: binned_statistic(r, data, "median", R_BINS)[0]
        for name, data in zip(stat_names, stat_data)
    }
    profiles["_r"] = r
    profiles["_xe"] = xe
    profiles["_urad_NUV"] = urad_NUV
    profiles["_urad_ONIR"] = urad_ONIR
    profiles["_T"] = T_gas
    return profiles


def assert_hii_temperature(profiles, lo=T_HII_MIN, hi=T_HII_MAX):
    """Photoionized gas should sit at a physical HII-region temperature (~1e4 K). Guards
    against the too-hot regime (missing nebular cooling and/or an over-hard ionizing band
    -> tens of kK) as well as spurious over-cooling. Uses the peak of the binned-median
    T(r) profile, robust to single-cell spikes near the source/front."""
    T_hii_peak = np.nanmax(profiles["T"])
    assert lo < T_hii_peak < hi, \
        f"HII-region peak temperature {T_hii_peak:.0f} K outside expected [{lo:.0f}, {hi:.0f}] K"
    return T_hii_peak


def assert_subcycle_matches_baseline(base, var, base_rif, var_rif, rel=0.1, rif_rel=0.1, min_bins=3):
    """A subcycled variant must reproduce the non-subcycled baseline.

    Compares xe only on radial bins that are finite AND ionized (xe > 0.5) in BOTH runs,
    and checks the ionization-front position separately via its radius. This is robust to
    two features of the (30 Msun) setup that a naive per-bin comparison chokes on:
      - the evacuated interior leaves the innermost bins empty -> binned_statistic returns
        NaN there, which is neither a match nor a meaningful mismatch;
      - the single ionization-front bin has xe swinging 1->0, so a sub-bin difference in
        front position blows up the per-bin relative diff even when the front *position*
        (r_IF) agrees to a few percent.
    The interior xe and r_IF together pin down "does subcycling change the answer?" without
    the binning artifacts. (Cross-variant urad/T are still shown in the comparison plots.)"""
    interior = (np.isfinite(base["xe"]) & (base["xe"] > 0.5)
                & np.isfinite(var["xe"]) & (var["xe"] > 0.5))
    n = int(interior.sum())
    assert n >= min_bins, \
        f"only {n} ionized-interior bins common to baseline & variant (need >= {min_bins})"
    b, v = base["xe"][interior], var["xe"][interior]
    assert np.allclose(v, b, rtol=rel), \
        f"interior xe max rel diff = {np.max(np.abs(v - b) / np.abs(b)):.3f} > {rel}"
    assert np.isfinite(base_rif) and np.isfinite(var_rif), \
        f"ionization-front radius not found (baseline={base_rif}, variant={var_rif})"
    assert abs(var_rif - base_rif) <= rif_rel * base_rif, \
        f"ionization-front radius {var_rif:.2f} pc vs baseline {base_rif:.2f} pc (> {rif_rel:.0%} apart)"


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
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


def make_temperature_snapshot_plot(label, output_dir, test_dir):
    """Plot T vs r for every snapshot in output_dir, colored by simulation time."""
    snaps = sorted(glob(f"{output_dir}/snapshot_*.hdf5"))
    if not snaps:
        return
    times = []
    for s in snaps:
        with h5py.File(s, "r") as F:
            times.append(float(F["Header"].attrs["Time"]))
    tmin, tmax = min(times), max(times)
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis
    for s, t in zip(snaps, times):
        with h5py.File(s, "r") as F:
            if "PartType5" not in F:
                continue
            pos = F["PartType0/Coordinates"][:]
            star_pos = F["PartType5/Coordinates"][0]
            r = np.linalg.norm(pos - star_pos, axis=1)
            T_gas = F["PartType0/Temperature"][:]
        color = cmap((t - tmin) / max(tmax - tmin, 1e-30))
        T_med = binned_statistic(r, T_gas, "median", R_BINS)[0]
        valid = np.isfinite(T_med) & (T_med > 0)
        if valid.any():
            ax.loglog(R_BIN_CENTERS[valid], T_med[valid], color=color, alpha=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(tmin, tmax))
    plt.colorbar(sm, ax=ax, label=r"$t$ (code units)")
    ax.set_xlabel(r"$r\;(\rm pc)$")
    ax.set_ylabel(r"$T\;(\rm K)$")
    ax.set_title(label)
    fig.savefig(str(Path(test_dir) / f"T_vs_r_snapshots_{label.replace('+','_')}.png"), bbox_inches="tight")
    plt.close(fig)


def make_comparison_plots(all_profiles, test_dir):
    """Generate comparison plots across all variants (xe, urad, T vs r)."""
    plot_configs = [
        ("xe", r"$x_e$", "xe"),
        ("urad_NUV", r"$u_{\rm rad,\,NUV}\;(\rm eV\,cm^{-3})$", "urad_NUV"),
        ("urad_ONIR", r"$u_{\rm rad,\,ONIR}\;(\rm eV\,cm^{-3})$", "urad_ONIR"),
        ("T", r"$T\;(\rm K)$", "T"),
    ]
    for field, ylabel, fname in plot_configs:
        for i, (label, profiles) in enumerate(all_profiles.items()):
            ms = 10 - 2 * i
            plot_quantiles_vs_r(
                profiles["_r"], profiles[f"_{field}"], label=_short(label),
                marker=_VARIANT_MARKERS[i % len(_VARIANT_MARKERS)], markersize=max(ms, 4),
                color=_VARIANT_COLORS[i % len(_VARIANT_COLORS)],
            )
        plt.xlabel(r"$r\;(\rm pc)$")
        plt.ylabel(ylabel)
        plt.legend(loc="best", fontsize="x-small")
        plt.savefig(str(Path(test_dir) / f"r_vs_{fname}.png"), bbox_inches="tight")
        plt.close()


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
    above = mh >= 0.5
    if not above.any() or above.all():
        return time, np.nan
    idx = np.where(above[:-1] & ~above[1:])[0]
    if len(idx) == 0:
        return time, np.nan
    i = idx[-1]
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
    plt.savefig(str(Path(test_dir) / "ifront_evolution.png"), bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------------------
# Build + run + check driver
# --------------------------------------------------------------------------------------
def run_variant(test_name, test_dir, num_mpi_ranks, num_omp_threads, extra_config_flags=()):
    """Build+run one variant, validate the final snapshot, compute profiles, assert the
    HII-region temperature, and emit the per-snapshot T(r) plot. Returns (profiles, label)."""
    prepare_inputs(test_dir, test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)
    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)
    with h5py.File(final_snap, "r") as F:
        assert "PartType5" in F, "Star particle missing from final snapshot"

    profiles = compute_profiles(final_snap)
    assert_hii_temperature(profiles)

    label = "+".join(extra_config_flags) if extra_config_flags else "baseline"
    make_temperature_snapshot_plot(label, variant_output_dir(test_name, extra_config_flags), test_dir)
    return profiles, label
