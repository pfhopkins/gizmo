"""Energy a feedback event gives the gas must not depend on the star's mass or the cell's.

One star in a uniform periodic lattice, adiabatic, no gravity. Total gas energy (thermal + kinetic) from the snapshots.
Supernovae (example model, GALSF_FB_MECHANICAL alone): 14.8 Msun at 1e51 erg each, whatever the star's mass, so energy per event = 1e51 erg
at star:cell mass 1:1, 100:1, 1000:1. Stellar mass return (FIRE-3 winds, young star, no supernovae yet): gas energy gained = sum of
mass returned x v_wind^2 / 2. n_H = 0.1 cm^-3 keeps nearly all the energy on the direct heating branch of the apply step.
A star moving through the gas delivers the same energy in its own frame, where the ejecta start at rest.
Bars fixed before the first run: 3 % per supernova (hydro drift of a blast ~1.5 %), 2 % for the winds; the same bars for a moving star.
"""
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, variant_output_dir, default_omp_threads, default_mpi_ranks

TEST_NAME = "mechfb_energy"
TEST_DIR = Path(__file__).parent
E_UNIT = 1.99e43  # erg per Msun (km/s)^2
M_STAR = 3.0e4
GAS_RETURN_RATE, GAS_RETURN_ENERGY = 300.0, 0.01  # FIRE-3 returns mass in 1 % packets: several events in the run, at about 300 km/s


def ics(m_star, m_cell, age_myr, v_star=0.0):
    sys.path.insert(0, str(TEST_DIR))
    from make_mechfb_energy_ics import make_ics
    return make_ics(str(TEST_DIR / f"{TEST_NAME}_ics.hdf5"), m_star, m_cell, age_myr, v_star)


def budget(fname):
    """Time, gas energy in the star's frame, gas mass, star mass."""
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][()]
        v = g["Velocities"][()] - F["PartType4/Velocities"][()][0]
        e = np.sum(m * g["InternalEnergy"][()]) + 0.5 * np.sum(m * np.sum(v ** 2, axis=1))
        return float(F["Header"].attrs["Time"]), e, m.sum(), float(F["PartType4/Masses"][()].sum())


def v_wind_fire3(age_gyr, z_solar):
    return np.sqrt(GAS_RETURN_ENERGY) * z_solar**0.12 * (3000. / (1. + (age_gyr / 0.003) ** 2.5) + 600. / (1. + (np.sqrt(z_solar) * age_gyr / 0.05) ** 6 + (z_solar / 0.2) ** 1.5) + 30.)


@pytest.mark.parametrize("ratio,v_star", [(1, 0), (100, 0), (1000, 0), (1, 300), (1, 1000)])
def test_energy_per_supernova(ratio, v_star):
    L = ics(M_STAR, M_STAR / ratio, 1.0, v_star)
    build_and_run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads(), (),
                       param_overrides={"BoxSize": L, "TimeMax": 1.0, "TimeBetSnapshot": 0.25, "TimeBetStatistics": 0.25, "MaxSizeTimestep": 1.0e-3})
    s = [budget(f) for f in sorted(glob(f"{variant_output_dir(TEST_NAME, ())}/snapshot_*.hdf5"))]
    n_sn = (s[-1][2] - s[0][2]) / 14.8
    e_per_sn = (s[-1][1] - s[0][1]) * E_UNIT / n_sn / 1e51
    print(f"star:cell {ratio}:1, {v_star} km/s: {n_sn:.2f} SNe, {e_per_sn:.4f} x 1e51 erg each")
    assert n_sn >= 1
    assert abs(e_per_sn - 1) < 0.03, f"star:cell {ratio}:1, {v_star} km/s: {e_per_sn:.3f} x 1e51 erg per SN"


@pytest.mark.parametrize("v_star", [0, 300])
def test_energy_of_winds(v_star):
    L = ics(1.0e4, 100.0, 0.0, v_star)
    flags = ("GALSF_FB_FIRE_STELLAREVOLUTION=3",)
    build_and_run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads(), flags,
                       param_overrides={"BoxSize": L, "TimeMax": 0.8, "TimeBetSnapshot": 0.1, "TimeBetStatistics": 0.1, "MaxSizeTimestep": 1.0e-3,
                                       "GasReturnFraction": GAS_RETURN_RATE, "GasReturnEnergy": GAS_RETURN_ENERGY})
    s = [budget(f) for f in sorted(glob(f"{variant_output_dir(TEST_NAME, flags)}/snapshot_*.hdf5"))]
    offered, gained = 0.0, s[-1][1] - s[0][1]
    for a, b in zip(s[:-1], s[1:]):
        age = 0.5 * (a[0] + b[0]) * 0.9778e-3  # the harness starts stars at age 0
        offered += 0.5 * (a[3] - b[3]) * v_wind_fire3(age, 0.014 / 0.0142) ** 2
    print(f"winds, {v_star} km/s: gained / offered = {gained / offered:.4f}, offered {offered * E_UNIT:.3e} erg")
    assert offered > 0
    assert abs(gained / offered - 1) < 0.02, f"winds, {v_star} km/s: gained / offered = {gained / offered:.3f}"
