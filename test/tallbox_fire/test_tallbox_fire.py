"""tallbox_fire: low-resolution FIRE-3 stratified shearing box of the solar neighbourhood (scripts/tallbox), 300 Myr.

1 kpc x 1 kpc x 8 kpc, Sigma_gas = 10 Msun/pc^2, R8 stellar disk + DM potential, q = 1, Omega = 28 km/s/kpc, 1e3 Msun
gas cells (~1e4 cells; ~15 min on 4 ranks). Exercises BOX_SHEARING, GRAVITY_PERIODIC_XY, GRAVITY_ANALYTIC_STRATIFIED_DISK
and FIRE-3 star formation and feedback together.

Check: the mean star formation rate surface density over 100-300 Myr, from the stellar mass formed in that window, is
within a factor of 2 of the benchmark. The benchmark is our own 1e3 Msun run (R8_m1e3_v3, same setup), not TIGRESS:
it is a regression test. At this resolution the SFR is ~4x below TIGRESS R8 (~2.5e-3), and the hot phase is absent.
The 200 Myr average spans ~4 cycles of the SFR; 50 Myr windows of the benchmark run vary by 1.7x.
"""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from gizmo.test import build_and_run_test, default_mpi_ranks, get_cooling_tables, get_final_snapshot, assert_final_time

TEST_NAME = "tallbox_fire"
TEST_DIR = Path(__file__).parent
REPO = TEST_DIR.parent.parent
MYR = 0.9778  # code time unit (pc / (km/s)) in Myr
L = 1000.0    # pc
SFR_BENCHMARK = 5.2e-4  # Msun/yr/kpc^2, 100-300 Myr, R8_m1e3_v3
SFR_FACTOR = 2.0


def test_tallbox_fire():
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        subprocess.run([sys.executable, str(REPO / "scripts/tallbox/make_tallbox_ics.py"), "--mgas", "1000",
                        "-o", str(ic)], check=True)
    get_cooling_tables(str(TEST_DIR))
    build_and_run_test(TEST_NAME, min(default_mpi_ranks(), 4), 1)
    assert_final_time(get_final_snapshot(TEST_NAME), TEST_NAME)

    t, mstar = [], []
    for s in sorted((TEST_DIR / "output").glob("snapshot_*.hdf5")):
        with h5py.File(s, "r") as f:
            t.append(f["Header"].attrs["Time"] * MYR)
            mstar.append(f["PartType4/Masses"][:].sum() if "PartType4" in f else 0.0)
    t, mstar = np.array(t), np.array(mstar)
    i, j = np.argmin(np.abs(t - 100)), np.argmin(np.abs(t - 300))
    sfr = (mstar[j] - mstar[i]) / ((t[j] - t[i]) * 1e6) / (L / 1e3) ** 2
    print(f"tallbox_fire: Sigma_SFR({t[i]:.0f}-{t[j]:.0f} Myr) = {sfr:.3e} Msun/yr/kpc^2 "
          f"(benchmark {SFR_BENCHMARK:.2e}, ratio {sfr / SFR_BENCHMARK:.2f})")
    assert SFR_BENCHMARK / SFR_FACTOR < sfr < SFR_BENCHMARK * SFR_FACTOR, \
        f"Sigma_SFR = {sfr:.3e} is not within {SFR_FACTOR}x of the benchmark {SFR_BENCHMARK:.2e}"
