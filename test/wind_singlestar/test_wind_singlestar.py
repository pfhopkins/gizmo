"""wind_singlestar test: wind-blown bubble from a 100 Msun ZAMS star.

Same box setup as SN_singlestar (50000 Msun, n_H = 100 cm^-3) but the star is
100 Msun at zero age with SINGLE_STAR_FB_WINDS=2.

Parametrized over:
  - Wind parameters (Mdot in Msun/yr, v_w in km/s)
  - Resolution (glass cube side length: 64, 128, 256)
  - Cooling variant (adiabatic vs COOLING+COOLING_OPERATOR_SPLIT)

Adiabatic runs check energy conservation; cooling runs check Weaver shell radius.
"""

import pytest
import numpy as np
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from glob import glob
import h5py
import sys

from meshoid import Meshoid
from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    flush_colorbar,
    assert_final_time,
    get_final_snapshot,
    variant_output_dir,
)
from pathlib import Path

TEST_NAME = "wind_singlestar"
TEST_DIR = Path(__file__).parent

# Code units: pc, Msun, km/s
M_PROTON_G = 1.6726e-24
PC_IN_CM = 3.0857e18
MSUN_IN_G = 1.989e33
X_H = 0.70
RHO_AMBIENT_CODE = (100.0 * M_PROTON_G / X_H) * (PC_IN_CM**3) / MSUN_IN_G


def wind_luminosity_code(Mdot_msun_yr, v_wind_kms):
    """Wind mechanical luminosity L_w = 0.5 * Mdot * v_w^2 in code units."""
    Mdot_cgs = Mdot_msun_yr * MSUN_IN_G / (365.25 * 86400)
    L_w_cgs = 0.5 * Mdot_cgs * (v_wind_kms * 1e5) ** 2
    code_power_cgs = 1.989e43 / (PC_IN_CM / 1e5)
    return L_w_cgs / code_power_cgs


def wind_luminosity_cgs(Mdot_msun_yr, v_wind_kms):
    Mdot_cgs = Mdot_msun_yr * MSUN_IN_G / (365.25 * 86400)
    return 0.5 * Mdot_cgs * (v_wind_kms * 1e5) ** 2


def weaver_bubble_radius(L_w, rho_0, t):
    """Weaver+ 1977 similarity solution for the swept-up shell radius."""
    return 0.76 * (L_w / rho_0) ** 0.2 * t**0.6


def generate_ics(res=128):
    """Generate ICs, regenerating if the resolution changed."""
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if ic_file.exists():
        with h5py.File(str(ic_file), "r") as F:
            existing_ngas = int(F["Header"].attrs["NumPart_Total"][0])
        if existing_ngas != res**3:
            ic_file.unlink()
    if not ic_file.exists():
        sys.path.insert(0, str(TEST_DIR))
        from make_wind_singlestar_ics import make_wind_singlestar_ics

        make_wind_singlestar_ics(str(ic_file), res=res)


def make_wind_config_flags(Mdot, v_w, wind_mode=None):
    """Return extra_config_flags tuple encoding wind parameters."""
    L_w_cgs = wind_luminosity_cgs(Mdot, v_w)
    flags = (f"WIND_MDOT={Mdot:.6g}", f"WIND_LUMINOSITY={L_w_cgs:.6g}")
    if wind_mode is not None:
        flags += (f"SINGLE_STAR_WIND_MODE={wind_mode}",)
    return flags


def load_snapshot_radial(snap_file):
    """Load a snapshot and return (time, box_size, coords, r, rho, vr)."""
    with h5py.File(snap_file, "r") as F:
        time = float(F["Header"].attrs["Time"])
        box_size = float(F["Header"].attrs["BoxSize"])
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        vel = F["PartType0/Velocities"][:]
        if "PartType5" in F and "Coordinates" in F["PartType5"]:
            center = F["PartType5/Coordinates"][0]
        else:
            center = np.array([box_size / 2.0] * 3)
    dr = coords - center
    r = np.sqrt(np.sum(dr * dr, axis=1))
    vr = np.sum(vel * dr, axis=1) / (r + 1e-30)
    return time, box_size, coords, r, rho, vr


def plot_slices(snap_file, output_dir=".", suffix=""):
    """Plot density and temperature slices from a snapshot."""
    with h5py.File(snap_file, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        temp = F["PartType0/Temperature"][:]
        box_center = float(F["Header"].attrs["BoxSize"]) / 2.0

    M = Meshoid(coords)
    center = np.array([box_center, box_center, box_center])
    ext = [-box_center, box_center, -box_center, box_center]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    rho_slice = M.Slice(np.log10(rho), res=512, plane="z", center=center, size=2 * box_center, order=0)
    im0 = axes[0].imshow(rho_slice.T, origin="lower", cmap="inferno", extent=ext)
    flush_colorbar(im0, ax=axes[0], label=r"$\log_{10}\rho$ (Msun/pc$^3$)")
    axes[0].set_xlabel("x (pc)")
    axes[0].set_ylabel("y (pc)")
    axes[0].set_title("Density")
    temp_slice = M.Slice(np.log10(temp), res=512, plane="z", center=center, size=2 * box_center, order=0)
    im1 = axes[1].imshow(temp_slice.T, origin="lower", cmap="inferno", extent=ext)
    flush_colorbar(im1, ax=axes[1], label=r"$\log_{10} T$ (K)")
    axes[1].set_xlabel("x (pc)")
    axes[1].set_ylabel("y (pc)")
    axes[1].set_title("Temperature")
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / f"Density_Temperature_2D{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def measure_shell_radius(snap_file, n_bins=120, density_contrast=1.5, min_particles_per_bin=10):
    """Return (time, r_shell) where r_shell is the outermost radius at which the
    spherically-averaged density exceeds density_contrast * rho_ambient."""
    time, box_size, coords, r_sim, rho_sim, vr_sim = load_snapshot_radial(snap_file)
    N = len(r_sim)
    # Set innermost bin edge so the first bin contains >= min_particles_per_bin
    r_min = box_size * (min_particles_per_bin * 3 / (4 * np.pi * N)) ** (1.0 / 3.0)
    r_bins = np.linspace(r_min, box_size / 2.0, n_bins)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    with h5py.File(snap_file, "r") as F:
        masses = F["PartType0/Masses"][:]
    mass_in_bin = binned_statistic(r_sim, masses, "sum", r_bins)[0]
    rho_sph = mass_in_bin / shell_vol
    elevated = np.isfinite(rho_sph) & (rho_sph > density_contrast * RHO_AMBIENT_CODE)
    if not elevated.any():
        return time, np.nan
    return time, r_centers[np.where(elevated)[0][-1]]


def plot_energy_vs_time(snaps, L_w, output_dir=".", suffix=""):
    """Plot kinetic and thermal energy vs time alongside expected and code-tracked injection."""
    times, E_kin, E_therm = [], [], []
    for s in snaps:
        with h5py.File(s, "r") as F:
            t = float(F["Header"].attrs["Time"])
            masses = F["PartType0/Masses"][:]
            vel = F["PartType0/Velocities"][:]
            u = F["PartType0/InternalEnergy"][:]
        times.append(t)
        E_kin.append(0.5 * np.sum(masses * np.sum(vel**2, axis=1)))
        E_therm.append(np.sum(masses * u))

    times = np.array(times)
    E_kin = np.array(E_kin)
    E_therm = np.array(E_therm)
    E_injected = L_w * times
    E_kin_excess = E_kin - E_kin[0]
    E_therm_excess = E_therm - E_therm[0]

    outdir = Path(snaps[0]).parent
    logfile = outdir / "MechFB_EnergyInjected.txt"
    has_log = logfile.exists() and logfile.stat().st_size > 0
    if has_log:
        data = np.loadtxt(str(logfile))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t_log = data[:, 0]
        cum_inj = data[:, 1] + data[:, 2]  # KE + TE

    plt.figure()
    mask = times > 0
    plt.plot(times[mask], E_kin_excess[mask], "bo-", markersize=4, label="KE (gas)")
    plt.plot(times[mask], E_therm_excess[mask], "rs-", markersize=4, label="Thermal (gas)")
    plt.plot(times[mask], E_kin_excess[mask] + E_therm_excess[mask], "k^-", markersize=5, label="KE + Thermal")
    plt.plot(times[mask], E_injected[mask], "g--", linewidth=2, label="Expected (Lw * t)")
    if has_log:
        plt.plot(t_log, cum_inj, "m-", linewidth=1.5, alpha=0.7, label="Code-tracked injection")
    plt.xlabel("t (code units)")
    plt.ylabel("Energy (code units)")
    plt.legend(fontsize=9)
    plt.title("Energy budget")
    plt.savefig(str(Path(output_dir) / f"Energy_vs_t{suffix}.png"), bbox_inches="tight")
    plt.close()

    return times, E_kin_excess, E_therm_excess, E_injected


def get_snapshots(test_name, extra_config_flags=()):
    outdir = variant_output_dir(test_name, extra_config_flags)
    return sorted(glob(f"{outdir}/snapshot_*.hdf5"))


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(default_mpi_ranks(), default_omp_threads())])
@pytest.mark.parametrize(
    "res",
    [32, 64, 128],
    ids=lambda x: f"N{x}",
)
@pytest.mark.parametrize(
    "Mdot_vw", [(1e-6, 3000.0), (1e-5, 3000.0), (1e-4, 3000.0)], ids=lambda x: f"Mdot{x[0]:.0e}_vw{x[1]:.0f}"
)
@pytest.mark.parametrize("wind_mode", [None, 1, 2], ids=lambda x: f"wm{x}" if x else "wm_auto")
@pytest.mark.parametrize(
    "cooling_flags",
    [(), ("COOLING",)],
    ids=["adiabatic", "cooling"],
)
def test_wind_singlestar(num_mpi_ranks, num_omp_threads, Mdot_vw, res, wind_mode, cooling_flags):
    Mdot, v_w = Mdot_vw
    L_w = wind_luminosity_code(Mdot, v_w)

    # Build extra_config_flags from wind params + wind mode + cooling.
    wind_flags = make_wind_config_flags(Mdot, v_w, wind_mode=wind_mode)
    # WIND_TEST_NRES is unused by the code but ensures a unique output directory per resolution
    extra_config_flags = wind_flags + (f"WIND_TEST_NRES={res}",) + cooling_flags

    generate_ics(res=res)

    # Set Sink_outflow_particlemass = 1e-2 * gas mass resolution
    m_gas = 50000.0 / res**3
    params_file = TEST_DIR / f"{TEST_NAME}.params"
    params_text = params_file.read_text()
    import re

    params_text = re.sub(
        r"Sink_outflow_particlemass\s+\S+",
        f"Sink_outflow_particlemass {1e-2 * m_gas:.6g}",
        params_text,
    )
    params_file.write_text(params_text)

    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)

    snaps = get_snapshots(TEST_NAME, extra_config_flags)
    assert len(snaps) > 1, f"No snapshots produced for {extra_config_flags}"
    final_snap = snaps[-1]

    # Measure shell radius
    times, r_shells = [], []
    for s in snaps:
        t, r_sh = measure_shell_radius(s)
        if t > 0:
            times.append(t)
            r_shells.append(r_sh)
    times = np.array(times)
    r_shells = np.array(r_shells)

    wm_str = f"wm{wind_mode}" if wind_mode is not None else "wm_auto"
    label = f"Mdot={Mdot:.0e} vw={v_w:.0f} N={res} {wm_str}"
    suffix = f"_Mdot{Mdot:.0e}_vw{v_w:.0f}_N{res}_{wm_str}"
    if cooling_flags:
        cooling_label = "+".join(cooling_flags)
        label += f" +{cooling_label}"
        suffix += "_" + "_".join(cooling_flags)

    # Save shell radius data to file
    np.savetxt(
        str(TEST_DIR / f"Rshell_data{suffix}.txt"),
        np.column_stack([times, r_shells]),
        header=f"t(code) R_shell(pc) | {label}",
        fmt="%.6e",
    )

    # Individual shell radius plot
    good = np.isfinite(r_shells) & (r_shells > 0) & (times > 0)
    if good.any():
        t_plot = np.logspace(np.log10(times[good].min()), np.log10(times[good].max() * 1.1), 200)
        R_weaver_plot = weaver_bubble_radius(L_w, RHO_AMBIENT_CODE, t_plot)
        from numpy.polynomial.polynomial import polyfit

        fit_mask = good & (times >= 0.01)
        if not fit_mask.any():
            fit_mask = good
        c = polyfit(np.log10(times[fit_mask]), np.log10(r_shells[fit_mask]), 1)
        alpha, A = c[1], 10 ** c[0]

        plt.figure()
        plt.loglog(times[good], r_shells[good], "ko", markersize=4, label="GIZMO")
        plt.loglog(t_plot, R_weaver_plot, "r-", linewidth=1.5, label="Weaver 1977")
        #        plt.loglog(t_plot, A * t_plot**alpha, "b--", label=f"Best fit ($t^{{{alpha:.2f}}}$)")
        plt.xlabel("t (code units)")
        plt.ylabel("R_shell (pc)")
        plt.legend()
        plt.title(label)
        plt.savefig(str(TEST_DIR / f"Rshell_vs_t{suffix}.png"), bbox_inches="tight")
        plt.close()

    # Plots
    snap_times, E_kin_x, E_therm_x, E_inj = plot_energy_vs_time(snaps, L_w, output_dir=str(TEST_DIR), suffix=suffix)
    plot_slices(final_snap, output_dir=str(TEST_DIR), suffix=suffix)

    # Final-snapshot radial profiles
    time, box_size, coords, r_sim, rho_sim, vr_sim = load_snapshot_radial(final_snap)
    R_weaver = weaver_bubble_radius(L_w, RHO_AMBIENT_CODE, time)

    r_max = box_size / 2.0
    r_bins = np.linspace(0.0, r_max, 40)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    with h5py.File(final_snap, "r") as F:
        masses_final = F["PartType0/Masses"][:]
    mass_in_bin = binned_statistic(r_sim, masses_final, "sum", r_bins)[0]
    rho_binned = mass_in_bin / shell_vol
    vr_binned = binned_statistic(r_sim, vr_sim, "median", r_bins)[0]

    for field, binned in [("Density", rho_binned), ("RadialVelocity", vr_binned)]:
        plt.figure()
        plt.plot(r_centers, binned, "o", markersize=3, label="GIZMO")
        plt.axvline(R_weaver, color="red", linestyle="--", label="Weaver $R_b$")
        if field == "Density":
            plt.axhline(RHO_AMBIENT_CODE, color="grey", linestyle=":", label=r"$\rho_{\rm amb}$")
            plt.yscale("log")
        plt.xlabel("r (pc)")
        plt.ylabel(field)
        plt.legend()
        plt.title(f"t = {time:.4f}")
        plt.savefig(str(TEST_DIR / f"{field}{suffix}.png"))
        plt.close()

    # --- Assertions ---

    if not cooling_flags:
        # Adiabatic: verify energy conservation
        mask = snap_times > 0
        if mask.any():
            dE = E_kin_x[mask] + E_therm_x[mask]
            ratio = dE / E_inj[mask]
            assert np.all(
                np.abs(ratio - 1.0) < 0.05
            ), f"Energy not conserved: ratio dE/E_injected = {ratio[-1]:.3f} at t={snap_times[mask][-1]:.4f}"

    # Shell should exist and be expanding
    valid = np.isfinite(rho_binned)
    assert valid.any(), "No valid density bins"
    i_peak = np.nanargmax(rho_binned)
    r_peak = r_centers[i_peak]
    assert r_peak > r_centers[1], f"Density peak at r={r_peak:.3f} pc is too close to the star"
    assert vr_binned[i_peak] > 0, f"No outflow at shell: vr = {vr_binned[i_peak]}"

    if cooling_flags:
        rel_err = abs(r_peak - R_weaver) / R_weaver
        assert rel_err < 0.3, (
            f"Shell-radius mismatch: peak at {r_peak:.3f} pc vs Weaver {R_weaver:.3f} pc "
            f"(relative error {rel_err:.3f})"
        )
