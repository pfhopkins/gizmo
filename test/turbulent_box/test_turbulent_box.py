"""Driven isothermal turbulence steered toward Mach 10 by the TURB_DRIVING_TARGET_SIGMA control loop.

32^3 periodic box, 10 pc, c_s = 0.2 km/s, solenoidal driving with a parabolic spectrum on 1 <= kL/2pi <= 3,
starting from rest. With a fixed driving amplitude this box settles near Mach 15 for TurbDrive_ApproxRMSVturb
set to Mach 10: the driving parameters only estimate the velocity they produce. The loop corrects that
order-unity mismatch; it is not meant to hold the velocity exactly. Two variants: hydro with the driving force
recomputed every step, and MHD (uniform B_z, initial plasma beta 1) with the force recomputed only on driving
updates (TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE), the way stirring runs use it.

The Mach number is read from the snapshots: mass-weighted rms velocity about the mean over c_s. Snapshot
velocities, like the loop's own measurement, are each cell's velocity at its last kick, so this is a check that
the loop works, not an independent measurement; on this box the difference from velocities synchronized to the
output time is ~0.1-0.3% once the gas is moving. The limits are broad on purpose: the check is that the loop
removes the order-unity error without running away, not how closely it tracks the target.

Config.sh is the fixed-amplitude box (also a timing vehicle), so the loop is added here.
"""

import glob
import subprocess
import sys
from os import path

import h5py
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gizmo.test import build_and_run_test, default_mpi_ranks, default_omp_threads, variant_output_dir, variant_suffix

TEST_NAME = "turbulent_box"
TEST_DIR = f"test/{TEST_NAME}"
CS = 0.2  # isothermal sound speed, km/s (EOS_ENFORCE_ADIABAT = CS^2)
MACH_TARGET = 10.0  # TurbDrive_ApproxRMSVturb / CS
T_EDDY = 5.0  # TurbDrive_MaxWavelength / TurbDrive_ApproxRMSVturb
T_SETTLED = 2 * T_EDDY

MAX_MEAN_ERROR = 0.10  # |<Mach>/target - 1| after T_SETTLED (measured 0.5%)
MAX_SETTLED_DEVIATION = 0.20  # any snapshot after T_SETTLED (measured 3.5%)
MAX_PEAK = 1.30  # whole run, including spin-up from rest (measured 1.03)

VARIANTS = {
    "hydro": ((), None),
    # driving updates every t_eddy/20 when the force is only recomputed then
    "mhd_lazy": (("MAGNETIC", "MHD_CONSTRAINED_GRADIENT=1", "TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE"),
                 {"TurbDrive_TimeBetweenTurbUpdates": 0.05 * T_EDDY}),
}
SLOW = {"mhd_lazy"}  # ~30 min at 2 ranks


def _ensure_ic():
    if not path.exists(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"):
        subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)


def _mach_history(outdir):
    rows = []
    for s in sorted(glob.glob(f"{outdir}/snapshot_*.hdf5")):
        with h5py.File(s, "r") as f:
            m = f["PartType0/Masses"][:].astype(np.float64)
            v = f["PartType0/Velocities"][:].astype(np.float64)
            vmean = (m[:, None] * v).sum(0) / m.sum()
            rows.append((f["Header"].attrs["Time"], np.sqrt((m * ((v - vmean) ** 2).sum(1)).sum() / m.sum()) / CS))
    return np.array(rows).T


@pytest.mark.parametrize("variant", [pytest.param(v, marks=pytest.mark.slow) if v in SLOW else v for v in VARIANTS])
def test_turbulent_box(variant):
    _ensure_ic()
    flags = ("TURB_DRIVING_TARGET_SIGMA",) + VARIANTS[variant][0]
    build_and_run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads(), flags, param_overrides=VARIANTS[variant][1])

    t, mach = _mach_history(variant_output_dir(TEST_NAME, flags))
    plt.plot(t / T_EDDY, mach, "o-", color="k", ms=3, lw=0.8)
    plt.axhline(MACH_TARGET, color="C3", lw=0.8)
    plt.xlabel(r"$t / t_{\rm eddy}$"); plt.ylabel("Mach")
    plt.savefig(f"{TEST_DIR}/mach_vs_time{variant_suffix(flags)}.png", bbox_inches="tight", dpi=150)
    plt.close()

    settled = t >= T_SETTLED
    assert settled.sum() >= 5, f"only {settled.sum()} snapshots after t = {T_SETTLED}"
    print(f"{variant}: settled Mach mean {mach[settled].mean():.3f}, max |error| "
          f"{np.abs(mach[settled] / MACH_TARGET - 1).max():.3f}, peak {mach.max():.3f}")
    assert abs(mach[settled].mean() / MACH_TARGET - 1) < MAX_MEAN_ERROR
    assert np.abs(mach[settled] / MACH_TARGET - 1).max() < MAX_SETTLED_DEVIATION
    assert mach.max() / MACH_TARGET < MAX_PEAK
