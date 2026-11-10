"""single_stars_cr_injection: supernovae and fast winds of IMF-sampled single stars inject cosmic rays (COSMIC_RAY_FLUID):
every explosion gives the fraction f = CosmicRay_SNeFraction (default 0.1) of its energy to the CR fluid and the rest to heat
(resolvedism_fb_thermal.cc:406), winds faster than 1000 km/s give f of their kinetic energy (resolvedism_fb_momentum.cc:338).
The box runs the production path: MAGNETIC, operator-split CR-gas exchange, and a diffusion coefficient of 1e5 pc km/s (3e28 cm^2/s).

An earlier version ran without a magnetic field at coefficient 1 and lost nearly all the CR energy within steps; without a field the
streaming loss takes the sound speed of the gas as Alfven speed, and at that coefficient it turns the CRs into heat in the hot cells.
It also cut explosions at 1000 km/s, which the model does not. Hence the new setup.

Bars (set before the first run of this version):
- supernovae (single_stars_supernovae box: core-collapse, failed, PPISN, PISN): ledger rows E_cr_offered and E_cr_injected_sn = f x the
  energy of all explosions to 1e-3; gas thermal + kinetic gain = (1 - f) x that to 1e-2.
- wind (a 35 Msun star at 1000+ km/s): E_cr_injected_wind = f x the kinetic energy booked for its ejecta to 1e-2, no SN row.
"""
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_cr_injection"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_supernovae"))
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_winds_agb_ia"))
from make_single_stars_supernovae_ics import make_single_stars_supernovae_ics  # noqa: E402
from make_single_stars_winds_agb_ia_ics import make_single_stars_winds_agb_ia_ics  # noqa: E402

E_UNIT = 1.989e33 * 1e10  # erg per code energy unit (Msun (km/s)^2)
F_CR = 0.1


@pytest.fixture(scope="module")
def tables():
    t = get_stellar_evolution_tables()
    build_gizmo_for_test(TEST_NAME)
    return t


def run(tag, tables, make_ic, overrides=()):
    d = TEST_DIR / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    L = make_ic(str(ic), tables["StellarTablesFile"])
    with h5py.File(ic, "a") as F:
        b = np.zeros((len(F["PartType0/Masses"]), 3))
        b[:, 2] = 2e-6
        F["PartType0/MagneticField"] = b
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
    for a, b in overrides:
        params = params.replace(a, b)
    (d / f"{TEST_NAME}.params").write_text(params)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, check=True)
    head = (d / "output" / "ledger.txt").read_text().splitlines()[0].split()[1:]
    last = np.atleast_2d(np.loadtxt(d / "output" / "ledger.txt"))[-1]
    return d, ic, dict(zip(head, last))


def gas_energy(fname):
    with h5py.File(fname, "r") as F:
        g = F["PartType0"]
        m, v, u = g["Masses"][()], g["Velocities"][()], g["InternalEnergy"][()]
        e = (m * u).sum() + 0.5 * (m * (v ** 2).sum(1)).sum()
        if "PartType4" in F:
            s = F["PartType4"]
            e += 0.5 * (s["Masses"][()] * (s["Velocities"][()] ** 2).sum(1)).sum()
    return e * E_UNIT


def test_supernovae(tables):
    d, ic, row = run("sn", tables, lambda f, t: make_single_stars_supernovae_ics(f, t)[0])
    events = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    e_sn = events[:, 8].sum()
    assert e_sn > 0
    snap = sorted((d / "output").glob("snapshot_*.hdf5"))[-1]
    gained = gas_energy(snap) - gas_energy(ic)
    print(f"offered {row['E_cr_offered_erg']:.4e}, deposited {row['E_cr_injected_sn_erg']:.4e}, f E {F_CR * e_sn:.4e} erg; gas gained {gained:.4e} of {(1 - F_CR) * e_sn:.4e}")
    assert abs(row["E_cr_offered_erg"] / (F_CR * e_sn) - 1) < 1e-3
    assert abs(row["E_cr_injected_sn_erg"] / (F_CR * e_sn) - 1) < 1e-3
    assert abs(gained / ((1 - F_CR) * e_sn) - 1) < 1e-2


def test_wind(tables):
    tmax = (("TimeMax                            5e-5", "TimeMax                            0.01"), ("TimeBetSnapshot                    5e-5", "TimeBetSnapshot                    0.01"),
            ("MaxSizeTimestep                    1e-5", "MaxSizeTimestep                    2.5e-4"))
    d, ic, row = run("wind", tables, lambda f, t: make_single_stars_winds_agb_ia_ics(f, t, which=("wind",)), tmax)
    print(f"wind ejecta {row['E_single_star_ejecta_erg']:.4e} erg, CR offered {row['E_cr_offered_erg']:.4e}, deposited {row['E_cr_injected_wind_erg']:.4e}")
    assert row["E_single_star_ejecta_erg"] > 0
    assert abs(row["E_cr_injected_wind_erg"] / (F_CR * row["E_single_star_ejecta_erg"]) - 1) < 1e-2
    assert row["E_cr_injected_sn_erg"] == 0
