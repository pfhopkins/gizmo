"""single_stars_type_ia_rate test: the Type Ia hazard of white dwarfs, counted over many stars against the delay-time distribution
and the table's dead-progenitor fraction. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_type_ia_rate"
TEST_DIR = Path(__file__).parent
PC_KMS_IN_YR = 3.085678e13 / 3.155e7
N_WD, M_DRAWN, AGE_YR, MULTIPLIER = 3000, 3.0, 1.0e9, 1.0e6  # MULTIPLIER as in Config.sh
IA_DTD_NORM, IA_F_WD, IA_T_MIN_GYR = 2.93e-4, 0.76, 0.04  # galaxy_sf/resolvedism_fb_shared.h


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tables, ranks=2):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_type_ia_rate_ics import make_single_stars_type_ia_rate_ics
    d = TEST_DIR / "run_r2"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
        (d / "output").mkdir(parents=True)
        L = make_single_stars_type_ia_rate_ics(str(d / "output" / "snapshot_000.hdf5"), tables["StellarTablesFile"], n_wd=N_WD, m_drawn=M_DRAWN, age_yr=AGE_YR)
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(params)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"],
                           cwd=d, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_events.txt"))
    return sorted((d / "output").glob("snapshot_*.hdf5")), rows


def expected_fraction(t_end_code):
    """fraction of the white dwarfs exploded by t_end (code time): survival exp(-H) of the per-star hazard
    h = MULTIPLIER * IA_DTD_NORM / t * M_drawn * IA_F_WD / f_dead(t) [per Gyr], with f_dead from the reference implementation"""
    ref = np.loadtxt(TEST_DIR.parent / "stellar_evolution_tables" / "unit" / "lookup_reference_ia.txt").reshape(8, 40)[5]  # log Z = -1.85
    t_ref = 10.0 ** (6.3 + np.arange(40) * (10.5 - 6.3) / 39.0)  # yr
    t = AGE_YR + np.linspace(0.0, t_end_code * PC_KMS_IN_YR, 2001)  # yr
    assert t[0] > IA_T_MIN_GYR * 1e9
    fdead = np.interp(np.log10(t), np.log10(t_ref), ref)
    h = MULTIPLIER * IA_DTD_NORM / (t * 1e-9) * M_DRAWN * IA_F_WD / fdead  # per Gyr
    H = np.trapezoid(h, t * 1e-9)
    return 1.0 - np.exp(-H)


def test_rate(built):
    snaps, rows = run(built)
    import h5py
    with h5py.File(snaps[-1], "r") as F:
        t_end = float(F["Header"].attrs["Time"])
    ia = rows[rows[:, 2] == 4]
    assert len(ia) > 100 and len(np.unique(ia[:, 1])) == len(ia)  # every white dwarf explodes at most once
    for t_cut in (0.5 * t_end, t_end):
        n = int(np.sum(ia[:, 0] <= t_cut))
        p = expected_fraction(t_cut)
        sigma = np.sqrt(N_WD * p * (1 - p))
        assert abs(n - N_WD * p) <= 4 * sigma + 0.01 * N_WD * p, (t_cut, n, N_WD * p, sigma)
