"""wind_singlestar test: wind-blown bubble from a 100 Msun ZAMS star.

Same box setup as SN_singlestar (50000 Msun, n_H = 100 cm^-3) but the star is
100 Msun at zero age with SINGLE_STAR_FB_WINDS=2. The test checks that the wind
drives an expanding bubble whose shell radius agrees with the Weaver+ 1977
similarity solution R_b = 0.76 * (L_w / rho_0)^(1/5) * t^(3/5).
"""

import pytest
import numpy as np
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from glob import glob
import h5py

from meshoid import Meshoid
from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    flush_colorbar,
    assert_final_time,
    get_final_snapshot,
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


def generate_ics():
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        from make_wind_singlestar_ics import make_wind_singlestar_ics

        make_wind_singlestar_ics(str(ic_file))


def wind_luminosity_code(Mdot_msun_yr=1e-5, v_wind_kms=3000.0):
    """Wind mechanical luminosity L_w = 0.5 * Mdot * v_w^2 in code units."""
    Mdot_cgs = Mdot_msun_yr * MSUN_IN_G / (365.25 * 86400)
    L_w_cgs = 0.5 * Mdot_cgs * (v_wind_kms * 1e5) ** 2
    code_power_cgs = 1.989e43 / (PC_IN_CM / 1e5)
    return L_w_cgs / code_power_cgs


def weaver_bubble_radius(L_w, rho_0, t):
    """Weaver+ 1977 similarity solution for the swept-up shell radius."""
    return 0.76 * (L_w / rho_0) ** 0.2 * t**0.6


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


def plot_slices(snap_file, output_dir="."):
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
    fig.savefig(str(Path(output_dir) / "Density_Temperature_2D.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def measure_shell_radius(snap_file, r_bins=np.linspace(0.0, 12.0, 40)):
    """Return (time, r_shell) where r_shell is the radius of peak spherically-averaged density."""
    time, box_size, coords, r_sim, rho_sim, vr_sim = load_snapshot_radial(snap_file)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    mass_in_bin = binned_statistic(r_sim, rho_sim * 0, "count", r_bins)[0]  # just need counts
    # Actually sum the masses directly
    with h5py.File(snap_file, "r") as F:
        masses = F["PartType0/Masses"][:]
    mass_in_bin = binned_statistic(r_sim, masses, "sum", r_bins)[0]
    rho_sph = mass_in_bin / shell_vol
    valid = np.isfinite(rho_sph) & (shell_vol > 0)
    if not valid.any():
        return time, np.nan
    return time, r_centers[np.nanargmax(rho_sph)]


def plot_shell_radius_vs_time(times, r_shells, L_w, rho_0, output_dir="."):
    """Plot measured shell radius vs time alongside the Weaver solution and a best-fit power law."""
    good = np.isfinite(r_shells) & (r_shells > 0) & (times > 0)
    t_weaver = np.logspace(np.log10(times[good].min()), np.log10(max(times) * 1.1), 200)
    R_weaver = weaver_bubble_radius(L_w, rho_0, t_weaver)

    # Best-fit power law: log R = alpha * log t + log A
    from numpy.polynomial.polynomial import polyfit

    log_t = np.log10(times[good])
    log_R = np.log10(r_shells[good])
    c = polyfit(log_t, log_R, 1)  # c[0] = intercept, c[1] = slope
    alpha = c[1]
    A = 10 ** c[0]
    R_fit = A * t_weaver**alpha

    plt.figure()
    plt.loglog(times[good], r_shells[good], "ko", markersize=4, label="GIZMO")
    plt.loglog(t_weaver, R_weaver, "r-", label="Weaver 1977 ($t^{3/5}$)")
    plt.loglog(t_weaver, R_fit, "b--", label=f"Best fit ($t^{{{alpha:.2f}}}$)")
    plt.xlabel("t (code units)")
    plt.ylabel("R_shell (pc)")
    plt.legend()
    plt.title("Shell radius vs time")
    plt.savefig(str(Path(output_dir) / "Rshell_vs_t.png"), bbox_inches="tight")
    plt.close()


def plot_energy_vs_time(snaps, L_w, output_dir="."):
    """Plot kinetic and thermal energy vs time alongside the expected injected energy."""
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

    plt.figure()
    mask = times > 0
    plt.plot(times[mask], E_kin_excess[mask], "bo-", label="KE (gas)")
    plt.plot(times[mask], E_therm_excess[mask], "rs-", label="Thermal (gas)")
    plt.plot(times[mask], E_kin_excess[mask] + E_therm_excess[mask], "k^-", label="KE + Thermal")
    plt.plot(times[mask], E_injected[mask], "g--", label="Injected (Lw * t)")
    plt.xlabel("t (code units)")
    plt.ylabel("Energy (code units)")
    plt.legend()
    plt.title("Energy budget")
    plt.savefig(str(Path(output_dir) / "Energy_vs_t.png"), bbox_inches="tight")
    plt.close()


@pytest.mark.parametrize(
    "num_mpi_ranks,num_omp_threads",
    [(default_mpi_ranks(), default_omp_threads())],
)
def test_wind_singlestar(num_mpi_ranks, num_omp_threads):
    generate_ics()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(TEST_NAME)
    assert_final_time(final_snap, TEST_NAME)

    # --- Measure shell radius across all snapshots ---
    snaps = sorted(glob(f"test/{TEST_NAME}/output/snapshot_*.hdf5"))
    times, r_shells = [], []
    for s in snaps:
        t, r_sh = measure_shell_radius(s)
        if t > 0:
            times.append(t)
            r_shells.append(r_sh)
    times = np.array(times)
    r_shells = np.array(r_shells)

    # Estimate wind luminosity from the final snapshot's star properties
    L_w = wind_luminosity_code()

    # Plot R_shell(t) vs Weaver
    plot_shell_radius_vs_time(times, r_shells, L_w, RHO_AMBIENT_CODE, output_dir=str(TEST_DIR))

    # Plot energy budget
    plot_energy_vs_time(snaps, L_w, output_dir=str(TEST_DIR))

    # --- Final-snapshot radial profile plots ---
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

    plot_slices(final_snap, output_dir=str(TEST_DIR))

    for label, binned in [("Density", rho_binned), ("RadialVelocity", vr_binned)]:
        plt.figure()
        plt.plot(r_centers, binned, "o", markersize=3, label="GIZMO")
        plt.axvline(R_weaver, color="red", linestyle="--", label="Weaver $R_b$")
        if label == "Density":
            plt.axhline(RHO_AMBIENT_CODE, color="grey", linestyle=":", label=r"$\rho_{\rm amb}$")
            plt.yscale("log")
        plt.xlabel("r (pc)")
        plt.ylabel(label)
        plt.legend()
        plt.title(f"t = {time:.4f}")
        plt.savefig(str(TEST_DIR / f"{label}.png"))
        plt.close()

    # --- Weaver bubble checks at final time ---
    valid = np.isfinite(rho_binned)
    assert valid.any(), "No valid density bins"
    i_peak = np.nanargmax(rho_binned)
    r_peak = r_centers[i_peak]

    rel_err = abs(r_peak - R_weaver) / R_weaver
    assert rel_err < 0.3, (
        f"Shell-radius mismatch: peak at {r_peak:.3f} pc vs Weaver {R_weaver:.3f} pc " f"(relative error {rel_err:.3f})"
    )

    assert r_peak > r_centers[1], f"Density peak at r={r_peak:.3f} pc is too close to the star"
    assert vr_binned[i_peak] > 0, f"No outflow at shell: vr = {vr_binned[i_peak]}"
