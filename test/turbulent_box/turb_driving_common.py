"""Shared procedure for the driven-turbulence tests (turbulent_box, turbsphere): IC writing, run,
Mach-number measurement from snapshots, plot, and the pass/fail checks on the
TURB_DRIVING_TARGET_SIGMA control loop.

Units are pc - km/s - Msun throughout. Gas is isothermal via EOS_ENFORCE_ADIABAT = CS^2, i.e.
P = CS^2 rho^GAMMA. Magnetic fields are written in gauss (UnitMagneticField_in_gauss = 1); GIZMO's
internal field unit is sqrt(4 pi P_unit), in which the magnetic pressure is B^2/2.
"""

import glob
import re
from os import path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import ticker

from gizmo.test import build_and_run_test, default_mpi_ranks, variant_output_dir, variant_suffix
from gizmo.units import UNIT_LENGTH_IN_CM, UNIT_MASS_IN_G, UNIT_VELOCITY_IN_CM_PER_S

GAMMA = 1.001
CS = 0.2  # isothermal sound speed, km/s
MACH_TARGET = 10.0
T_EDDY = 5.0  # crossing time: box side (turbulent_box) or cloud diameter (turbsphere) over the target v_rms
T_SETTLED = 2 * T_EDDY  # start of the window the steady-state checks use

MAX_MEAN_ERROR = 0.03  # |<Mach>/target - 1| over the settled window
MAX_SETTLED_DEVIATION = 0.1  # max |Mach/target - 1| at any settled snapshot
MAX_OVERSHOOT = 0.1  # max Mach/target - 1 over the whole run, including spin-up from rest

# MHD with constrained gradients at the level STARFORGE runs use (SINGLE_STAR_STARFORGE_DEFAULTS turns it on)
VARIANTS = {"hydro": (), "mhd": ("MAGNETIC", "MHD_CONSTRAINED_GRADIENT=1")}
# Lazy force: TurbAccel is recomputed only on driving updates, the way STARFORGE stirring runs use it,
# with the update cadence they use, t_cross/10 (t_cross = TurbDrive_MaxWavelength / v_rms = T_EDDY)
LAZY_FLAG = "TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE"
LAZY_PARAMS = {"TurbDrive_TimeBetweenTurbUpdates": 0.1 * T_EDDY}

UNIT_PRESSURE_IN_CGS = UNIT_MASS_IN_G * UNIT_VELOCITY_IN_CM_PER_S**2 / UNIT_LENGTH_IN_CM**3
UNIT_B_IN_GAUSS = np.sqrt(4 * np.pi * UNIT_PRESSURE_IN_CGS)  # GIZMO's internal field unit


def pressure(rho):
    return CS**2 * rho**GAMMA


def b_field_gauss(rho, beta):
    """Field strength in gauss giving plasma beta = P / (B^2/2) [code units] at density rho."""
    return np.sqrt(2 * pressure(rho) / beta) * UNIT_B_IN_GAUSS


def write_ics(out, pos, mass, box_size, bz_gauss):
    """Gas at rest, isothermal, with a uniform B_z. Builds without MAGNETIC ignore the field."""
    ngas = len(pos)
    with h5py.File(out, "w") as f:
        npart = np.array([ngas, 0, 0, 0, 0, 0], dtype=np.int32)
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = box_size
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        for flag in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback", "Flag_DoublePrecision"):
            h.attrs[flag] = 0
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((ngas, 3), dtype=np.float32))
        g.create_dataset("Masses", data=np.full(ngas, mass, dtype=np.float32))
        g.create_dataset("ParticleIDs", data=np.arange(1, ngas + 1, dtype=np.uint32))
        # P/rho = CS^2 up to rho^(GAMMA-1); EOS_ENFORCE_ADIABAT resets it anyway
        g.create_dataset("InternalEnergy", data=np.full(ngas, CS**2 / (GAMMA - 1), dtype=np.float32))
        g.create_dataset("MagneticField", data=np.tile([0, 0, bz_gauss], (ngas, 1)).astype(np.float32))


def snapshot_stats(snapshot, aperture=None):
    """Mach number (mass-weighted rms velocity about the mean, over the mass-weighted isothermal sound
    speed) and plasma beta (volume-weighted P over B^2/2), of gas within `aperture` of the box centre
    if given, else of all gas. Also the half-mass radius of all gas about its median position."""
    with h5py.File(snapshot, "r") as f:
        hdr = f["Header"].attrs
        t, box = hdr["Time"], hdr["BoxSize"]
        g = f["PartType0"]
        x, m, v, u, rho = (g[k][:] for k in ("Coordinates", "Masses", "Velocities", "InternalEnergy", "Density"))
        b = g["MagneticField"][:] * hdr["UnitMagneticField_In_Gauss"] / UNIT_B_IN_GAUSS if "MagneticField" in g else None
    xc = np.median(x, axis=0)
    r_half = np.median(np.linalg.norm(x - xc, axis=1))  # equal-mass particles
    s = np.linalg.norm(x - box / 2, axis=1) < aperture if aperture else np.ones(len(m), bool)
    m, v, u, rho = m[s], v[s], u[s], rho[s]
    vmean = (m[:, None] * v).sum(0) / m.sum()
    v_rms = np.sqrt((m * ((v - vmean) ** 2).sum(1)).sum() / m.sum())
    mach = v_rms / np.sqrt((GAMMA - 1) * (m * u).sum() / m.sum())
    beta = np.nan
    if b is not None:
        vol = m / rho
        beta = ((GAMMA - 1) * rho * u * vol).sum() / (0.5 * (b[s] ** 2).sum(1) * vol).sum()
    return t, mach, beta, r_half


def history(output_dir, aperture=None):
    snaps = sorted(glob.glob(f"{output_dir}/snapshot_*.hdf5"))
    return np.array([snapshot_stats(s, aperture) for s in snaps]).T


def controller_log(stdout_file):
    """(t, v_rms, gain) at every driving update, from the loop's stdout line."""
    pat = re.compile(r"TURB_DRIVING_TARGET_SIGMA: t=(\S+) v_rms=(\S+) v_target=\S+ gain=(\S+)")
    rows = [m.groups() for m in map(pat.match, open(stdout_file)) if m] if path.isfile(stdout_file) else []
    return np.array(rows, dtype=float).T if rows else np.zeros((3, 0))


def plot(t, mach, beta, log, fname, r_half=None):
    mhd = np.isfinite(beta).any()
    n = 2 + mhd + (r_half is not None)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(6, 2 * n + 1))
    if log.size:
        axes[0].plot(log[0] / T_EDDY, log[1] / CS, color="0.6", lw=0.8, label="loop measurement")
        axes[1].plot(log[0] / T_EDDY, log[2], color="k", lw=0.8)
    axes[0].plot(t / T_EDDY, mach, "o", color="k", ms=3, label="snapshots")
    axes[0].axhline(MACH_TARGET, color="C3", lw=0.8)
    axes[0].set_ylabel("Mach")
    axes[0].legend(frameon=False)
    axes[1].set_yscale("log")
    if log.size and log[2].max() < 10 * log[2].min():  # under a decade, log ticks leave most of the axis unlabeled
        axes[1].yaxis.set_major_locator(ticker.MaxNLocator(5))
        axes[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
        axes[1].yaxis.set_minor_locator(ticker.NullLocator())
    axes[1].set_ylabel("driving gain")
    if mhd:
        axes[2].semilogy(t / T_EDDY, beta, "o-", color="k", ms=3, lw=0.8)
        axes[2].set_ylabel(r"plasma $\beta$")
    if r_half is not None:
        axes[-1].plot(t / T_EDDY, r_half, "o-", color="k", ms=3, lw=0.8)
        axes[-1].set_ylim(0, None)
        axes[-1].set_ylabel(r"gas $r_{\rm half}$ [pc]")
    axes[-1].set_xlabel(r"$t / t_{\rm eddy}$")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_and_check(test_name, variant, aperture=None, lazy=False, max_ranks=8):
    """lazy adds the lazy-force flag and update cadence; a test whose Config.sh already has them leaves it False."""
    flags = VARIANTS[variant] + ((LAZY_FLAG,) if lazy else ())
    build_and_run_test(test_name, num_mpi_ranks=default_mpi_ranks(max_ranks=max_ranks), extra_config_flags=flags,
                       param_overrides=LAZY_PARAMS if lazy else None)

    outdir = variant_output_dir(test_name, flags)
    log = f"{outdir}/test_{test_name}.out" if flags else f"test/{test_name}/test_{test_name}.out"
    t, mach, beta, r_half = history(outdir, aperture)
    # r_half only means something for an isolated cloud, not for gas filling a periodic box
    plot(t, mach, beta, controller_log(log), f"test/{test_name}/mach_vs_time{variant_suffix(flags)}.png",
         r_half=r_half if aperture else None)

    settled = t >= T_SETTLED
    assert settled.sum() >= 5, f"only {settled.sum()} snapshots after t = {T_SETTLED}"
    err = mach[settled] / MACH_TARGET - 1
    print(f"{test_name} {variant}{' lazy' if lazy else ''}: settled Mach mean {mach[settled].mean():.3f}, max |error| {np.abs(err).max():.3f}; "
          f"peak Mach {mach.max():.3f}; beta {beta[0]:.3g} -> {beta[-1]:.3g}; r_half {r_half[0]:.3g} -> {r_half[-1]:.3g}")
    assert abs(mach[settled].mean() / MACH_TARGET - 1) < MAX_MEAN_ERROR
    assert np.abs(err).max() < MAX_SETTLED_DEVIATION
    assert mach.max() / MACH_TARGET - 1 < MAX_OVERSHOOT
