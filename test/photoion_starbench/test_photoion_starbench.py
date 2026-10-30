"""photoion_starbench test: StarBench D-type expansion of an HII region (Bisbas et al. 2015) with the per-direction
photoionisation, at three resolutions; must reproduce stella_dev (-28.8 / -16.9 / -6.7 %) within 2 points. See README.md."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "photoion_starbench"
TEST_DIR = Path(__file__).parent
STELLA_DEV = {"m10": -0.288, "m1": -0.169, "m0.1": -0.067}  # D-type median R_IF / R_StarBench - 1


def chemistry_tables_dir():
    import sys
    sys.path.insert(0, str(TEST_DIR.parent / "sgchem_box"))
    from test_sgchem_box import chemistry_tables_dir as d
    return d()


def setup_run(d, rung, tables):
    """the run directory of one rung: ICs and parameters"""
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_photoion_starbench_ics import make_photoion_starbench_ics, RUNGS
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    make_photoion_starbench_ics(str(d / "ics.hdf5"), RUNGS[rung])
    p = Path(TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile                  SET_BY_TEST", f"StellarTablesFile                  {tables['StellarTablesFile']}")
    p = p.replace("SGChemTablesPath                   SET_BY_TEST", f"SGChemTablesPath                   {chemistry_tables_dir()}")
    (d / f"{TEST_NAME}.params").write_text(p)


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def check(outdir, rung):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from starbench import grade
    _, _, _, ref, med = grade(str(outdir))
    assert abs(med - STELLA_DEV[rung]) <= 0.02, (rung, med, ref)


def test_m10(built):
    """the coarsest rung (40^3 cells), run here: a few minutes on 4 cores"""
    d = TEST_DIR / "run_m10"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        setup_run(d, "m10", built)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(1, 4) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "4"}, check=True)
    check(d / "output", "m10")


@pytest.mark.parametrize("rung", ["m1", "m0.1"])
def test_finer_rungs(rung):
    """the finer rungs (86^3, 185^3 cells) run as batch jobs (run_ladder.sh); graded when their output is present"""
    d = Path(os.environ.get("GIZMO_STARBENCH_DIR", TEST_DIR)) / f"run_{rung}" / "output"
    import h5py
    snaps = sorted(d.glob("snapshot_*.hdf5")) if d.exists() else []
    if not snaps or h5py.File(snaps[-1], "r")["Header"].attrs["Time"] < 4.5e-3:
        pytest.skip(f"no finished run in {d}: run run_ladder.sh first")
    check(d, rung)
