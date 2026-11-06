"""single_stars_radiation_pinned: test hooks of the single-star radiation. TEST_SINGLE_STARS_LUMINOSITY_PINNED sets the bolometric
luminosity of every star above the mass cut to TestFixedLbol [erg/s] from birth, as stella_dev: FUV 0.5 L (all in the
8-13.6 eV band), NUV 0.2 L, optical 0.3 L, Lyman-Werner 0.15 L, no ionising photons;
TEST_SINGLE_STARS_RADIATION_OFF silences every band. Observed through the momentum the local radiation-pressure loop offers
(MomWinds.txt), same setup as single_stars_radiation_pressure (one 40 Msun star, n_H = 100, 1 Msun cells).
Bars, written before the runs: pinned: offered momentum per time = L_abs (1 + tau_IR) / c with L_abs = min(f_UV 0.5 L + f_NUV 0.2 L
+ f_OPT 0.3 L, L), estimator of single_stars_radiation_pressure, to 2 % at every step, and constant in time to 2 %;
off: no momentum offered in any step."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_radiation_pinned"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice, YR_PER_TIME_UNIT  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

BUILDS = {}
N_H, NGB, M_STAR, F_LIFE = 100.0, 32, 40.0, 0.3
MSUN, PC, MP, C = 1.989e33, 3.085678e18, 1.6726e-24, 2.9979e10


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(tag, extra=()):
    if tag not in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(extra))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / f"GIZMO_{tag}")
        BUILDS[tag] = TEST_DIR / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables, n=16, stars=((M_STAR, (0.5, 0.5, 0.5), True),)):
    """stars: (M, position in box units, with a record)"""
    rho = N_H * 1.4 * MP * PC ** 3 / MSUN  # Msun / pc^3 (H nuclei per mass as the chemistry)
    dx = (1.0 / rho) ** (1.0 / 3.0); L = n * dx
    tab = Tables(tables["StellarTablesFile"])
    lZ = np.log10(0.0142)
    ages = [tab._i2(tab.t_pms, np.log10(M), lZ) + F_LIFE * tab._i2(tab.life, np.log10(M), lZ) for M, _, _ in stars]
    age = ages[0]
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ic = d / "output" / "snapshot_000.hdf5"
    make_treecol_pixels_ics(str(ic), L, lattice(n, L), [(M, a, np.asarray(x) * L + 0.31 * dx) for (M, x, _), a in zip(stars, ages)], hsml=2 * dx)
    with h5py.File(ic, "a") as F:
        F["PartType4/SingleStarDrawnMass"][...] = [M if rec else 0.0 for M, _, rec in stars]  # 0: no record
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    over = {"BoxSize": L, "TimeMax": 2e-3, "MaxSizeTimestep": 1e-4, "DesNumNgb": NGB}
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    rows = np.atleast_2d(np.loadtxt(d / "output" / "MomWinds.txt"))
    return rows, tab, age, rho


def expected_pinned(rho, L=1.0e39):
    """stella_dev estimator for the pinned split, [Msun km/s per code time]"""
    h = (3 * max(NGB, 64) / (4 * np.pi * rho)) ** (1.0 / 3.0) * PC
    rho_cgs = rho * MSUN / PC ** 3
    NH, Sigma, dgr = rho_cgs / MP * h, rho_cgs * h, 0.0142 / 0.014
    f = [1 - np.exp(-s * 1e-21 * dgr * NH) for s in (2.0, 1.3, 0.5)]
    L_abs = min(f[0] * 0.5 * L + f[1] * 0.2 * L + f[2] * 0.3 * L, L)
    return L_abs * (1 + 5.0 * dgr * Sigma) / C * (PC / 1e5) / (MSUN * 1e5)


def supply_per_time(rows):
    """MomWinds.txt: time, n_checked, n_firing, n_touched, <R>, momentum offered, momentum coupled, ..."""
    dt = np.diff(rows[:, 0]); ok = dt > 0
    return rows[:-1][ok, 5] / dt[ok], rows[:-1][ok, 0]


def test_pinned_luminosity(tables):
    rows, tab, age, rho = run(build("pinned", ("TEST_SINGLE_STARS_LUMINOSITY_PINNED",)), "pinned", tables)
    offered, t = supply_per_time(rows)
    offered = offered[offered > 0]
    assert len(offered) >= 5
    np.testing.assert_allclose(offered, expected_pinned(rho), rtol=2e-2)
    assert offered.max() / offered.min() - 1 < 2e-2


def test_radiation_off(tables):
    rows, tab, age, rho = run(build("off", ("TEST_SINGLE_STARS_RADIATION_OFF",)), "off", tables)
    assert len(rows) >= 5 and rows[:, 5].sum() == 0, rows[:, 5]
