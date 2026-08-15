"""SN_singlestar test: Sedov-Taylor blast driven by an actual single-star supernova.

A 10 Msun, 50 Myr-old star at the center of a 50000 Msun, n_H = 100 cm^-3 box goes
supernova on the first timestep, depositing 1e51 erg into the surrounding gas.

Three variants are tested:
  1. Adiabatic (no cooling): verifies the shock radius and density jump match the
     analytic Sedov solution at early time (t ~ 0.015 code units).
  2. With COOLING: verifies the blast still produces a recognizable shell at the
     Sedov radius (before the radiative phase sets in).
  3. With COOLING + SINGLE_STAR_FB_RAD: same shell check, with the star's own radiative
     feedback also active. RAD needs COOLING to mean anything, so it is not tested alone.
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
    get_cooling_tables,
    get_final_snapshot,
    variant_output_dir,
)
from pathlib import Path

TEST_NAME = "SN_singlestar"
TEST_DIR = Path(__file__).parent

# Code units: pc, Msun, km/s. Energy unit = Msun*(km/s)^2 = 1.989e43 erg.
E_SN_CGS = 1.0e51
ENERGY_UNIT_CGS = 1.989e43
E_SN_CODE = E_SN_CGS / ENERGY_UNIT_CGS

X_H = 0.70
M_PROTON_G = 1.6726e-24
PC_IN_CM = 3.0857e18
MSUN_IN_G = 1.989e33
RHO_AMBIENT_CODE = (100.0 * M_PROTON_G / X_H) * (PC_IN_CM**3) / MSUN_IN_G

T_SEDOV_CHECK = 0.015  # code time at which to compare against Sedov


def generate_ics():
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        from make_SN_singlestar_ics import make_SN_singlestar_ics
        make_SN_singlestar_ics(str(ic_file))


def sedov_shock_radius(E, rho, t):
    """Self-similar Sedov-Taylor shock radius for a 3D point explosion (gamma=5/3)."""
    return 1.1517 * (E * t * t / rho) ** 0.2


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


def get_snapshot_nearest_time(test_name, target_time, extra_config_flags=()):
    """Return the snapshot file whose header time is closest to target_time."""
    outdir = variant_output_dir(test_name, extra_config_flags)
    snaps = sorted(glob(f"{outdir}/snapshot_*.hdf5"))
    best, best_dt = None, np.inf
    for s in snaps:
        with h5py.File(s, "r") as F:
            t = float(F["Header"].attrs["Time"])
        if abs(t - target_time) < best_dt:
            best_dt = abs(t - target_time)
            best = s
    return best


def plot_density_slice(coords, rho, box_center, output_dir=".", suffix=""):
    M = Meshoid(coords)
    center = np.array([box_center, box_center, box_center])
    rho_slice = M.Slice(np.log10(rho), res=512, plane="z", center=center,
                        size=2 * box_center, order=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno",
                   extent=[-box_center, box_center, -box_center, box_center])
    flush_colorbar(im, ax=ax, label=r"$\log_{10}\rho$ (Msun/pc$^3$)")
    ax.set_xlabel("x (pc)")
    ax.set_ylabel("y (pc)")
    ax.set_title("SN_singlestar - Density Slice")
    fig.savefig(str(Path(output_dir) / f"Density_2D{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.parametrize(
    "num_mpi_ranks,num_omp_threads",
    [(default_mpi_ranks(), default_omp_threads())],
)
@pytest.mark.parametrize(
    "extra_config_flags",
    [
        (),
        ("COOLING",),
        ("COOLING", "SINGLE_STAR_FB_RAD"),
    ],
    ids=["adiabatic", "cooling", "rad"],
)
def test_SN_singlestar(num_mpi_ranks, num_omp_threads, extra_config_flags):
    generate_ics()
    if "COOLING" in extra_config_flags:
        get_cooling_tables(str(TEST_DIR))
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)

    final_snap = get_final_snapshot(TEST_NAME, extra_config_flags)
    assert_final_time(final_snap, TEST_NAME)

    suffix = ""
    if extra_config_flags:
        suffix = "_" + "_".join(extra_config_flags)

    # Plot final snapshot density slice
    time_final, box_size, coords_final, _, rho_final, _ = load_snapshot_radial(final_snap)
    plot_density_slice(coords_final, rho_final, box_size / 2.0, output_dir=str(TEST_DIR), suffix=suffix)

    # Sedov check at early time (before radiative phase)
    sedov_snap = get_snapshot_nearest_time(TEST_NAME, T_SEDOV_CHECK, extra_config_flags)
    assert sedov_snap is not None, "No snapshots found"
    time, box_size, coords, r_sim, rho_sim, vr_sim = load_snapshot_radial(sedov_snap)

    R_shock = sedov_shock_radius(E_SN_CODE, RHO_AMBIENT_CODE, time)

    # Bin radial profiles
    r_max = min(2.0 * R_shock, box_size / 2.0)
    r_bins = np.linspace(0.0, r_max, 30)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    rho_binned = binned_statistic(r_sim, rho_sim, "median", r_bins)[0]
    vr_binned = binned_statistic(r_sim, vr_sim, "median", r_bins)[0]

    for label, binned in [("Density", rho_binned), ("RadialVelocity", vr_binned)]:
        plt.figure()
        plt.plot(r_centers, binned, "o", markersize=3, label="GIZMO")
        plt.axvline(R_shock, color="red", linestyle="--", label="Sedov R_shock")
        if label == "Density":
            plt.axhline(RHO_AMBIENT_CODE, color="grey", linestyle=":", label=r"$\rho_{\rm amb}$")
            plt.axhline(4.0 * RHO_AMBIENT_CODE, color="green", linestyle=":",
                        label=r"$4\rho_{\rm amb}$ (strong-shock jump)")
        plt.xlabel("r (pc)")
        plt.ylabel(label)
        plt.legend()
        plt.title(f"t = {time:.4f} (Sedov check)")
        plt.savefig(str(TEST_DIR / f"{label}{suffix}.png"))
        plt.close()

    # --- Assertions ---

    # Shock radius should be near the analytic Sedov radius
    valid = np.isfinite(rho_binned)
    assert valid.any(), "No valid density bins"
    i_peak = np.nanargmax(rho_binned)
    r_peak = r_centers[i_peak]
    rel_err = abs(r_peak - R_shock) / R_shock
    assert rel_err < 0.3, (
        f"Shock-radius mismatch: peak at {r_peak:.3f} pc vs Sedov {R_shock:.3f} pc "
        f"(relative error {rel_err:.3f})"
    )

    # Only the 'adiabatic' variant has no COOLING, so these hold only for it -- and checking it rather
    # than just the bare run is what catches a regression on whichever wakeup path is not the default.
    if "COOLING" not in extra_config_flags:
        # Adiabatic: strong-shock density jump should be ~ 4x ambient
        assert rho_binned[i_peak] > 2.0 * RHO_AMBIENT_CODE, (
            f"Post-shock density {rho_binned[i_peak]:.3f} too low; "
            f"expected > {2.0 * RHO_AMBIENT_CODE:.3f}"
        )

        # Adiabatic: verify energy conservation (total gas KE+TE should equal E_SN)
        with h5py.File(sedov_snap, "r") as F:
            masses = F["PartType0/Masses"][:]
            vel = F["PartType0/Velocities"][:]
            u = F["PartType0/InternalEnergy"][:]
        with h5py.File(get_snapshot_nearest_time(TEST_NAME, 0.0, extra_config_flags), "r") as F0:
            masses0 = F0["PartType0/Masses"][:]
            vel0 = F0["PartType0/Velocities"][:]
            u0 = F0["PartType0/InternalEnergy"][:]
        E_kin = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
        E_therm = np.sum(masses * u)
        E_kin_0 = 0.5 * np.sum(masses0 * np.sum(vel0**2, axis=1))
        E_therm_0 = np.sum(masses0 * u0)
        dE = (E_kin + E_therm) - (E_kin_0 + E_therm_0)
        energy_ratio = dE / E_SN_CODE
        assert abs(energy_ratio - 1.0) < 0.1, (
            f"Energy not conserved: dE/E_SN = {energy_ratio:.3f}"
        )

    # Outflow at the shock
    assert vr_binned[i_peak] > 0, f"No outflow at shock: vr = {vr_binned[i_peak]}"
