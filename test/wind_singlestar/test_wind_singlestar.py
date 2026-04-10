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


def estimate_wind_luminosity(snap_file):
    """Estimate the wind mechanical luminosity L_w = 0.5 * Mdot * v_w^2 from
    the star's properties in a snapshot, mirroring GIZMO's SINGLE_STAR_FB_WINDS=2
    formulae in stellar_evolution.cc."""
    with h5py.File(snap_file, "r") as F:
        M_star = float(F["PartType5/Masses"][0])  # code = Msun
        L_solar = float(F["PartType5/StarLuminosity_Solar"][0])
        R_solar = float(F["PartType5/ProtoStellarRadius_inSolar"][0])

    m_solar = M_star
    ZZ = 1.0  # solar metallicity

    # Effective temperature
    T_eff = 5814.33 * (L_solar / R_solar**2) ** 0.25

    # Escape speed and wind velocity (Lamers 1995 branches)
    v_esc = 617.7 * np.sqrt(m_solar / R_solar)  # km/s
    if T_eff < 1.25e4:
        v_wind = 0.7 * v_esc
    elif T_eff < 2.1e4:
        v_wind = 1.3 * v_esc
    else:
        v_wind = 2.6 * v_esc
    vinf_over_vesc = v_wind / v_esc

    # Mass loss rate (de Jager/3 default, weak-wind limiter, Sahahit high-Ledd)
    logmdot = -6.0 + 1.5 * np.log10(L_solar / 1e6) + 0.69 * np.log10(max(ZZ, 1e-10))
    logmdot = min(logmdot, -7.65 + 2.9 * np.log10(L_solar / 1e5))  # weak-wind
    logmdot_high = (
        -8.445
        + 4.77 * np.log10(L_solar / 1e5)
        - 3.99 * np.log10(m_solar / 30)
        - 1.226 * np.log10(vinf_over_vesc / 2)
        + 0.761 * np.log10(max(ZZ, 1e-10))
    )
    logmdot = max(logmdot, logmdot_high)

    Mdot_cgs = 10**logmdot * MSUN_IN_G / (365.25 * 86400)  # g/s
    v_wind_cgs = v_wind * 1e5  # cm/s
    L_w_cgs = 0.5 * Mdot_cgs * v_wind_cgs**2

    # Convert to code power units: [Msun * (km/s)^2 / (pc/(km/s))]
    code_power_cgs = 1.989e43 / (PC_IN_CM / 1e5)  # erg/s per code power unit
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


def plot_density_slice(coords, rho, box_center, output_dir="."):
    M = Meshoid(coords)
    center = np.array([box_center, box_center, box_center])
    rho_slice = M.Slice(np.log10(rho), res=512, plane="z", center=center, size=2 * box_center, order=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        rho_slice.T, origin="lower", cmap="inferno", extent=[-box_center, box_center, -box_center, box_center]
    )
    flush_colorbar(im, ax=ax, label=r"$\log_{10}\rho$ (Msun/pc$^3$)")
    ax.set_xlabel("x (pc)")
    ax.set_ylabel("y (pc)")
    ax.set_title("wind_singlestar - Density Slice")
    fig.savefig(str(Path(output_dir) / "Density_2D.png"), dpi=150, bbox_inches="tight")
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
    """Plot measured shell radius vs time alongside the Weaver solution."""
    t_weaver = np.linspace(1e-3, max(times) * 1.1, 200)
    R_weaver = weaver_bubble_radius(L_w, rho_0, t_weaver)

    plt.figure()
    plt.loglog(times, r_shells, "ko", markersize=4, label="GIZMO")
    plt.loglog(t_weaver, R_weaver, "r-", label="Weaver 1977")
    plt.xlabel("t (code units)")
    plt.ylabel("R_shell (pc)")
    plt.legend()
    plt.title("Shell radius vs time")
    plt.savefig(str(Path(output_dir) / "Rshell_vs_t.png"), bbox_inches="tight")
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
    L_w = estimate_wind_luminosity(final_snap)

    # Plot R_shell(t) vs Weaver
    plot_shell_radius_vs_time(times, r_shells, L_w, RHO_AMBIENT_CODE, output_dir=str(TEST_DIR))

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

    plot_density_slice(coords, rho_sim, box_size / 2.0, output_dir=str(TEST_DIR))

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
