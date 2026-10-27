"""Driven isothermal turbulence held at Mach 10 by the TURB_DRIVING_TARGET_SIGMA control loop.

32^3 periodic box, 10 pc, c_s = 0.2 km/s, solenoidal driving with a parabolic spectrum on
1 <= kL/2pi <= 3. Starting from rest, the loop sets the gain on the driving force from
e = ln(v_target / v_rms). The test measures the Mach number from the snapshots -- independently of
the loop's own measurement -- and checks that it settles on the target and stays there.
"""

import glob
import re
import subprocess
import sys
from os import path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from gizmo.test import build_and_run_test, default_mpi_ranks, variant_output_dir

TEST_NAME = "turbulent_box"
TEST_DIR = f"test/{TEST_NAME}"
GAMMA = 1.001
MACH_TARGET = 10.0
T_EDDY = 5.0  # TurbDrive_MaxWavelength / TurbDrive_ApproxRMSVturb, the loop's unit of time
T_SETTLED = 2 * T_EDDY  # start of the window the steady-state checks use

MAX_MEAN_ERROR = 0.03  # |<Mach>/target - 1| over the settled window
MAX_SETTLED_DEVIATION = 0.1  # max |Mach/target - 1| at any settled snapshot
MAX_OVERSHOOT = 0.1  # max Mach/target - 1 over the whole run, including spin-up from rest


def _ensure_ic():
    if path.exists(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"):
        return
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)


def mach_number(snapshot):
    """Mass-weighted rms velocity about the mean, over the mass-weighted isothermal sound speed."""
    with h5py.File(snapshot, "r") as f:
        t = f["Header"].attrs["Time"]
        m = f["PartType0/Masses"][:]
        v = f["PartType0/Velocities"][:]
        u = f["PartType0/InternalEnergy"][:]
    vmean = (m[:, None] * v).sum(0) / m.sum()
    v_rms = np.sqrt((m * ((v - vmean) ** 2).sum(1)).sum() / m.sum())
    c_iso = np.sqrt((GAMMA - 1) * (m * u).sum() / m.sum())
    return t, v_rms / c_iso


def mach_history(output_dir):
    snaps = sorted(glob.glob(f"{output_dir}/snapshot_*.hdf5"))
    return np.array([mach_number(s) for s in snaps]).T


def controller_log(stdout_file):
    """(t, v_rms, gain) at every driving update, from the loop's stdout line."""
    pat = re.compile(r"TURB_DRIVING_TARGET_SIGMA: t=(\S+) v_rms=(\S+) v_target=\S+ gain=(\S+)")
    rows = [m.groups() for m in map(pat.match, open(stdout_file)) if m]
    return np.array(rows, dtype=float).T if rows else np.zeros((3, 0))


def plot(t, mach, log, fname):
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    if log.size:
        ax0.plot(log[0] / T_EDDY, log[1] / 0.2, color="0.6", lw=0.8, label="loop measurement")
        ax1.plot(log[0] / T_EDDY, log[2], color="k", lw=0.8)
    ax0.plot(t / T_EDDY, mach, "o", color="k", ms=3, label="snapshots")
    ax0.axhline(MACH_TARGET, color="C3", lw=0.8)
    ax0.set_ylabel("Mach")
    ax0.legend(frameon=False)
    ax1.set_yscale("log")
    ax1.set_ylabel("driving gain")
    ax1.set_xlabel(r"$t / t_{\rm eddy}$")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)


def test_turbulent_box():
    _ensure_ic()
    build_and_run_test(TEST_NAME, num_mpi_ranks=default_mpi_ranks(max_ranks=8))

    t, mach = mach_history(variant_output_dir(TEST_NAME))
    plot(t, mach, controller_log(f"{TEST_DIR}/test_{TEST_NAME}.out"), f"{TEST_DIR}/mach_vs_time.png")

    settled = t >= T_SETTLED
    assert settled.sum() >= 5, f"only {settled.sum()} snapshots after t = {T_SETTLED}"
    err = mach[settled] / MACH_TARGET - 1
    print(f"settled Mach: mean {mach[settled].mean():.3f}, max |error| {np.abs(err).max():.3f}; "
          f"peak Mach {mach.max():.3f}")
    assert abs(mach[settled].mean() / MACH_TARGET - 1) < MAX_MEAN_ERROR
    assert np.abs(err).max() < MAX_SETTLED_DEVIATION
    assert mach.max() / MACH_TARGET - 1 < MAX_OVERSHOOT
