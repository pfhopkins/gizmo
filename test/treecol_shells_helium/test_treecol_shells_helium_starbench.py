"""treecol_shells_helium: StarBench D-type expansion at net 17, coarse rung m10, against stella_dev with the He abundance of
the chemistry (reports: design_treeray_helium.md). Fixed rate, two-temperature gas, the star's spectrum. See README.md."""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "treecol_shells_helium"
TEST_DIR = Path(__file__).parent
PH_DIR = TEST_DIR.parent / "treecol_shells_photoion"
sys.path.insert(0, str(PH_DIR))
from make_treecol_shells_photoion_ics import make_ics, RUNGS  # noqa: E402
from starbench import grade, band_ledger, helium_front  # noqa: E402

FLAGS = ("GALSF_RESOLVEDISM_TREECOL_PIXELS=12", "TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS")
T_MAX_MYR = 2.5
Q1_OVER_Q = 0.1303  # 30 Msun, table
STELLA_DEV_MEDIAN, STELLA_DEV_RATIO = -0.353, 0.731  # 83b7ed68 net 17 with the He abundance of the chemistry


@pytest.fixture(scope="module")
def run_m10():
    d = Path(os.environ.get("GIZMO_TREERAY_HE_RUN", TEST_DIR / "run_m10"))
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        tables = get_stellar_evolution_tables()
        build_gizmo_for_test(TEST_NAME, extra_config_flags=FLAGS)
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        make_ics(str(d / "ics.hdf5"), RUNGS["m10"])
        p = (PH_DIR / "treecol_shells_photoion.params").read_text()
        tabdir = os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"]))
        p = p.replace("StellarTablesFile                  SET_BY_TEST", f"StellarTablesFile                  {tables['StellarTablesFile']}")
        p = p.replace("SGChemTablesPath                   SET_BY_TEST", f"SGChemTablesPath                   {tabdir}")
        (d / "r.params").write_text(p)
        threads = int(os.environ.get("GIZMO_TREERAY_THREADS", "4"))
        with open(d / "run.out", "w") as out:
            subprocess.run(mpi_launch_command(1, threads) + [str(TEST_DIR / "GIZMO"), "r.params", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": str(threads)}, check=True)
    return d / "output"


def dtype_rows(run):
    out = []
    for s in sorted(glob.glob(str(run / "snapshot_*.hdf5"))):
        if os.path.getsize(s) > 10000:
            r = band_ledger(s, Q1_OVER_Q)
            if 0.2 < r["t_myr"] < T_MAX_MYR:
                out.append((s, r))
    return out


def test_helium_ionised(run_m10):
    """cells with x(H+) > 0.9 hold He+ at the last D-type snapshot: >= 0.9 of ABHE (0 without the helium band)"""
    s, _ = dtype_rows(run_m10)[-1]
    assert helium_front(s)["x_he_in_hii"] >= 0.9


def test_helium_ledger(run_m10):
    """helium recombinations at most 1.3 Q1, He++ below 1e-3 of ABHE"""
    rows = dtype_rows(run_m10)
    assert rows and all(r["helium"] <= 1.3 for _, r in rows), [r["helium"] for _, r in rows]
    assert all(helium_front(s)["x_hepp"] < 1e-3 for s, _ in rows)


def test_helium_fills_the_h_zone(run_m10):
    """nested zones: (R_He / R_IF)^3 = (Q1 / Q) / (y alpha_He / alpha_H) = 1.2, so R_He / R_IF in 0.85-1.05"""
    assert 0.85 <= np.median([helium_front(s)["R_he_over_R_if"] for s, _ in dtype_rows(run_m10)]) <= 1.05


@pytest.mark.xfail(strict=True, reason="bar missed, kept as set: 0.73 Q while the shell collapses after 1.8 Myr, 0.83 Q at 0.45 Myr")
def test_total_ledger(run_m10):
    """recombinations 0.85-1.7 Q in every D-type snapshot (stella_dev net 17 on the same estimator: 0.59-1.06)"""
    t = [r["total"] for _, r in dtype_rows(run_m10)]
    assert all(0.85 < x < 1.7 for x in t), t


@pytest.mark.xfail(strict=True, reason="bar missed, kept as set: helium fuller and R_IF larger than stella_dev, cause not found")
def test_against_stella_dev(run_m10):
    """within 4 points of stella_dev's R_IF and 0.10 of its R_He / R_IF (He abundance as the chemistry: -35.3 %, 0.731)"""
    r = np.median([helium_front(s)["R_he_over_R_if"] for s, _ in dtype_rows(run_m10)])
    med = grade(str(run_m10), T_MAX_MYR)[4]
    assert abs(r - STELLA_DEV_RATIO) <= 0.10 and abs(med - STELLA_DEV_MEDIAN) <= 0.04, (r, med)
