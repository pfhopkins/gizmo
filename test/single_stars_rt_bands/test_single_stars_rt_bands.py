"""single_stars_rt_bands: IMF-sampled single stars source the moment solver's bins from the tables
(GALSF_SINGLE_STARS_RT_TABLE_BANDS; reports/design/design_m1_coupling.md). Two 40 Msun main-sequence stars at opposite
corners of a periodic box of very thin gas (2 ranks, so each rank holds a star), and one dead star. Periodic and thin, so
every photon stays in the bins.
  test_band_energy: energy per bin over the box = sum over stars of L_table(band) x t x c_red / c (2 %). Bins PE 8-13.6,
    LW 11.2-13.6 (inside PE, as stella_dev), 6-8, H0, He0, He1, He2. Their sum stays below L_bol.
  test_single_ionising_bin: one ionising bin holds the table's photon rate at the bin's mean photon energy (2 %).
  test_excluded_channels: the star-side radiation pressure and photoionisation channels do not build with the moment forces.
The rates are taken between two snapshots: the first injection arrives two steps after t = 0.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "single_stars_rt_bands"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice, YR_PER_TIME_UNIT  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

M_STAR, F_LIFE, L_BOX, N, T_END = 40.0, 0.3, 2.0, 8, 1e-2
C_RED_OVER_C = 1e-3
UNIT_LUM = 1.989e33 * 1e10 / (3.085678e18 / 1e5)  # erg/s per code unit
EV = 1.602176634e-12
BUILDS = {}
LZ = np.log10(0.0142)


@pytest.fixture(scope="module")
def tables():
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(tag, extra=()):
    if tag not in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=tuple(extra))
        shutil.move(TEST_DIR / "GIZMO", TEST_DIR / f"GIZMO_{tag}")
        BUILDS[tag] = TEST_DIR / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables):
    tab = Tables(tables["StellarTablesFile"])
    lM = np.log10(M_STAR)
    age = tab._i2(tab.t_pms, lM, LZ) + F_LIFE * tab._i2(tab.life, lM, LZ)
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    x = L_BOX * np.array([[0.13, 0.13, 0.13], [0.63, 0.63, 0.63], [0.37, 0.63, 0.13]])
    stars = [(M_STAR, age, x[0]), (M_STAR, age, x[1]), (M_STAR, 2 * tab._i2(tab.life, lM, LZ) + age, x[2])]
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L_BOX, lattice(N, L_BOX), stars, gas_mass=1e-7, hsml=0.5)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(2) + [str(exe), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    E = []
    for k in (1, 2):
        with h5py.File(d / "output" / f"snapshot_00{k}.hdf5", "r") as F:
            E.append(F["PartType0/PhotonEnergy"][()].sum(axis=0))
            t = F["Header"].attrs["Time"]
    return E[1] - E[0], t, age, tab  # the rate between two snapshots: injection starts two steps after t = 0


def band(tab, key, age):
    if key not in tab.t3:
        with h5py.File(tab.fname, "r") as F:
            tab.t3[key] = F[key][()].astype(np.float32).astype(np.float64)
    return 10 ** tab._i3(tab.t3[key], np.log10(M_STAR), LZ, np.log10(age - tab._i2(tab.t_pms, np.log10(M_STAR), LZ)))


def expected(tab, age, keys):
    return {k: 2 * band(tab, k, age) / UNIT_LUM * (T_END / 2) * C_RED_OVER_C for k in keys}


def nu_eff(lo, hi, T=4e4):
    """rt_get_sigma: number-weighted mean photon energy of his blackbody in a band [eV]"""
    e = np.linspace(lo, hi, 10000)
    I = e ** 3 / np.expm1(e / (8.617333e-5 * T))
    return np.trapezoid(I, e) / np.trapezoid(I / e, e)


def test_band_energy(tables):
    E, t, age, tab = run(build("multi", ("RT_CHEM_PHOTOION=2", "RT_DISABLE_RAD_PRESSURE")), "multi", tables)
    assert abs(t - T_END) < 1e-9
    assert len(E) == 7  # H0 He0 He1 He2 PE LW 6-8
    xp = expected(tab, age, ("log_L_ion_H0", "log_L_ion_He0", "log_L_ion_He1", "log_L_ion_He2", "log_L_FUV_total", "log_L_FUV_M1", "log_L_LW", "log_L_bol"))
    l68 = xp["log_L_FUV_total"] - xp["log_L_FUV_M1"]
    want = [xp["log_L_ion_H0"], xp["log_L_ion_He0"], xp["log_L_ion_He1"], xp["log_L_ion_He2"], xp["log_L_FUV_M1"], xp["log_L_LW"], l68]
    print("bins / table:", E / np.array(want))
    np.testing.assert_allclose(E, want, rtol=2e-2)
    assert E.sum() < xp["log_L_bol"]
    Q = expected(tab, age, ("log_Q_ion",))["log_Q_ion"]  # photons in the bins at his photon energies against the table
    e_ion = np.array([nu_eff(13.6, 24.6), nu_eff(24.6, 54.4), nu_eff(54.4, 70.0), nu_eff(70.0, 500.0)]) * EV
    ratio = (E[:4] / e_ion).sum() / Q
    print("photons in the bins / table:", ratio)
    assert 0.7 < ratio < 1.3


def test_single_ionising_bin(tables):
    E, t, age, tab = run(build("single", ("RT_CHEM_PHOTOION=1", "RT_DISABLE_RAD_PRESSURE")), "single", tables)
    assert len(E) == 4
    Q = expected(tab, age, ("log_Q_ion",))["log_Q_ion"]
    assert abs(E[0] / Q / EV / nu_eff(13.6, 500.0) - 1) < 2e-2, E[0] / Q / EV


@pytest.mark.parametrize("flag", ["GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS", "GALSF_FB_FIRE_RT_HIIHEATING", "GALSF_FB_PHOTOIONIZATION_PER_DIRECTION"])
def test_excluded_channels(flag):
    src = TEST_DIR.parents[1] / "src"
    flags = [l.strip() for l in (TEST_DIR / "Config.sh").read_text().splitlines() if l.strip()] + [flag]
    defs = ["-D" + f for f in flags]
    out = subprocess.run(["mpicxx", "-std=c++20", "-E", *defs, "-include", "declarations/precompiler_logic.h", "-x", "c++", "/dev/null", "-o", "/dev/null"],
                         cwd=src, capture_output=True, text=True)
    assert out.returncode != 0 and "counted once" in out.stderr, out.stderr[-400:]
