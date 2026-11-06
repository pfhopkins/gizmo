"""tallbox_fire: low-resolution FIRE-3 stratified shearing box of the solar neighbourhood (scripts/tallbox), 300 Myr.

1 kpc x 1 kpc x 8 kpc, Sigma_gas = 10 Msun/pc^2, R8 stellar disk + DM potential, q = 1, Omega = 28 km/s/kpc, 1e3 Msun
gas cells (~1e4 cells; ~15 min on 4 ranks). Exercises BOX_SHEARING, GRAVITY_PERIODIC_XY, GRAVITY_ANALYTIC_STRATIFIED_DISK
and FIRE-3 star formation and feedback together.

Check: the mean star formation rate surface density over 100-300 Myr, from the stellar mass formed in that window, is
within a factor of 2 of the benchmark. The benchmark is the mean of our own 1e3 Msun runs of this setup with the
SN-coupling fix (ejecta-mass weighting) and the LEBRON flux fix, 8.6e-4 to 1.03e-3 across four runs; it is a regression
test, not TIGRESS. The earlier benchmark (5.2e-4) came from runs that over-injected SN energy. At this resolution the
SFR is ~2.5x below TIGRESS R8 (~2.5e-3) and the hot phase is absent. The 200 Myr average spans ~4 cycles of the SFR.

Check: the long-range (LEBRON) FUV flux reaches the cool mid-plane gas. Each tree walk rebuilds a cell's raw flux and
selfshield_local_incident_uv_flux converts it to Habing units in place; with adaptive tree-force updates (the FIRE-3
default) a cell that skipped its walk had last step's value converted again (x8.6e-5) and lost its photoelectric
heating, down to the UV background. Here that left on average ~10% of the cool mid-plane gas below 1e-28 erg/s of
photoelectric heating per H atom (3-19% per snapshot), against ~1% with the fix.
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
SFR_BENCHMARK = 9.6e-4  # Msun/yr/kpc^2, 100-300 Myr
SFR_FACTOR = 2.0
NH = 1.989e33 / 3.0857e18**3 * 0.7 / 1.6726e-24  # code density -> n_H [cm^-3]
DARK_HEATING = 1e-28  # erg/s per H atom: G0 ~ 1e-3, the UV background alone
DARK_MAX = 0.03


def dark_fraction(snap):
    """mass fraction of the cool (T < 1e4 K) gas within 500 pc of the mid-plane with photoelectric heating per H atom
    below DARK_HEATING"""
    with h5py.File(snap[1], "r") as f:
        g = f["PartType0"]
        m, T, z = g["Masses"][:], g["Temperature"][:], g["Coordinates"][:, 2]
        heat = g["PElecHeatingRate"][:].astype(np.float64) * g["Density"][:] * NH  # rate is per n_H^2
    s = (T < 1e4) & (np.abs(z - np.average(z, weights=m)) < 500)
    return m[s][heat[s] < DARK_HEATING].sum() / m[s].sum()


def test_tallbox_fire():
    ic = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic.exists():
        subprocess.run([sys.executable, str(REPO / "scripts/tallbox/make_tallbox_ics.py"), "--mgas", "1000",
                        "-o", str(ic)], check=True)
    get_cooling_tables(str(TEST_DIR))
    build_and_run_test(TEST_NAME, min(default_mpi_ranks(), 4), 1)
    assert_final_time(get_final_snapshot(TEST_NAME), TEST_NAME)

    t, mstar, snaps = [], [], []
    for s in sorted((TEST_DIR / "output").glob("snapshot_*.hdf5")):
        with h5py.File(s, "r") as f:
            t.append(f["Header"].attrs["Time"] * MYR)
            mstar.append(f["PartType4/Masses"][:].sum() if "PartType4" in f else 0.0)
        snaps.append((t[-1], s))
    t, mstar = np.array(t), np.array(mstar)
    i, j = np.argmin(np.abs(t - 100)), np.argmin(np.abs(t - 300))
    sfr = (mstar[j] - mstar[i]) / ((t[j] - t[i]) * 1e6) / (L / 1e3) ** 2
    print(f"tallbox_fire: Sigma_SFR({t[i]:.0f}-{t[j]:.0f} Myr) = {sfr:.3e} Msun/yr/kpc^2 "
          f"(benchmark {SFR_BENCHMARK:.2e}, ratio {sfr / SFR_BENCHMARK:.2f})")
    dark = [dark_fraction(s) for s in snaps if s[0] >= 100]
    print(f"cool mid-plane gas mass fraction with photoelectric heating < {DARK_HEATING:.0e} erg/s/H, t >= 100 Myr: "
          + " ".join(f"{d:.2f}" for d in dark))
    assert SFR_BENCHMARK / SFR_FACTOR < sfr < SFR_BENCHMARK * SFR_FACTOR, \
        f"Sigma_SFR = {sfr:.3e} is not within {SFR_FACTOR}x of the benchmark {SFR_BENCHMARK:.2e}"
    assert np.mean(dark) < DARK_MAX, f"on average {np.mean(dark):.3f} of the cool mid-plane gas sees no long-range FUV"
