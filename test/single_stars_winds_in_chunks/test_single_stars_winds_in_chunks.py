"""single_stars_winds_in_chunks: GALSF_SINGLE_STARS_WINDS_IN_CHUNKS holds the wind of a massive star until the mass owed exceeds
a fraction of the drawn mass (1 % at 8 Msun, 20 % at 350 Msun), then returns it at once at the current wind speed; at the end of
life the rest goes out whatever its size, before the supernova. 12^3 cells of 1 Msun, SGChem, 2 ranks, 100 yr, one 35 Msun star.

Bars, written before the runs: (1) mass owed reaches the threshold 50 yr in: exactly one delivery, after 25 yr, mass = threshold to
1e-3, speed = table speed to 1 %, star mass down by that delivery to 1e-9; every step delivers without the flag. (2) death 80 yr in
owing half the threshold: no delivery before the death, one delivery of half the threshold to 1e-3 at or after it, one supernova.
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

TEST_NAME = "single_stars_winds_in_chunks"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_winds_agb_ia"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402
from test_single_stars_winds_agb_ia import wind_cumulative, wind_speed  # noqa: E402

L, N_SIDE, M, Z = 12.0, 12, 35.0, 0.014
YR = 3.085678e13 / 3.155e7  # yr per code time unit
MSUN_KMS = 1.989e33 * 1e5
THRESH = M * (0.01 + 0.19 * (M - 8.0) / 342.0)


@pytest.fixture(scope="module")
def built():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M), np.log10(Z)
    tables["t_pms"], tables["life"] = tab._i2(tab.t_pms, lM, lZ), tab._i2(tab.life, lM, lZ)
    return tables


def last_age_owing(tf, la_to, owed):
    """table log age from which owed Msun are due at la_to"""
    c = wind_cumulative(tf, M, Z, la_to) - owed
    lo, hi = 2.0, la_to
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if wind_cumulative(tf, M, Z, mid) < c else (lo, mid)
    return 0.5 * (lo + hi)


def run(tag, tables, age, last_age):
    d = TEST_DIR / f"output_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    ic = d / "snapshot_000.hdf5"
    make_treecol_pixels_ics(str(ic), L, lattice(N_SIDE, L), [(M, age, np.full(3, 0.5 * L + 0.17))], z=Z, hsml=2.0)
    with h5py.File(ic, "a") as f:
        f["PartType4/SingleStarLastWindLogAge"] = np.array([last_age])
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    for k, v in (("StellarTablesFile", tables["StellarTablesFile"]), ("SGChemTablesPath", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))):
        params = params.replace(f"{k:35s}SET_BY_TEST", f"{k:35s}{v}")
    params = params.replace("OutputDir                          output", f"OutputDir                          {d.name}")
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out, stderr=subprocess.STDOUT, check=True)
    def rows(n):
        return np.loadtxt(d / n, ndmin=2) if (d / n).exists() else np.zeros((0, 20))
    with h5py.File(sorted(d.glob("snapshot_*.hdf5"))[-1]) as f:
        m_star = f["PartType4/Masses"][()][0]
    return rows("WINDinfo.txt"), rows("SNinfo.txt"), m_star


def test_delivery_when_threshold_reached(built):
    tf = built["StellarTablesFile"]
    age = built["t_pms"] + 0.5 * built["life"]
    la0, la1 = np.log10(age - built["t_pms"]), np.log10(age + 50.0 - built["t_pms"])
    grown = wind_cumulative(tf, M, Z, la1) - wind_cumulative(tf, M, Z, la0)
    wind, sn, m_star = run("reach", built, age, last_age_owing(tf, la0, THRESH - grown))
    print(f"threshold {THRESH:.5f} Msun, {grown:.3e} Msun grows in 50 yr; {len(wind)} deliveries at t[yr] {wind[:, 0][:5] * YR}, masses {wind[:, 6][:5]}")
    assert len(wind) == 1
    t, dM, p = wind[0, 0] * YR, wind[0, 6], wind[0, 7]
    v = wind_speed(tf, M, Z, np.log10(age + t - built["t_pms"]))
    print(f"t {t:.1f} yr, dM/threshold {dM / THRESH:.6f}, v {p / dM / MSUN_KMS:.1f} km/s (table {v:.1f}), star {M - m_star:.6f} Msun lost")
    assert t > 25.0
    assert abs(dM / THRESH - 1) < 1e-3
    assert abs(p / dM / MSUN_KMS / v - 1) < 0.01
    assert abs((M - m_star) / dM - 1) < 1e-9


def test_rest_goes_out_at_the_end_of_life(built):
    tf = built["StellarTablesFile"]
    la_end = np.log10(built["life"])
    t_death = 80.0
    wind, sn, _ = run("death", built, built["t_pms"] + built["life"] - t_death, last_age_owing(tf, la_end, 0.5 * THRESH))
    print(f"{len(wind)} deliveries at t[yr] {wind[:, 0][:5] * YR} of {wind[:, 6][:5]} Msun, {len(sn)} SN at {sn[:, 0] * YR}, threshold {THRESH:.5f}")
    assert len(wind) == 1
    assert wind[0, 0] * YR >= t_death * (1 - 1e-3)
    assert abs(wind[0, 6] / (0.5 * THRESH) - 1) < 1e-3
    assert len(sn) == 1 and sn[0, 0] >= wind[0, 0]
