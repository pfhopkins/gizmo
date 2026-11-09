"""single_stars_birth_age test: TEST_SINGLE_STARS_BORN_WITH_GIVEN_AGE back-dates the formation time of a star born in the run,
so that its evolution starts at the given age (parameter TestStarInitAgeYr). Lattice of the single_stars_collect_gas test, one
star of 35 Msun drawn mass born at the first step (time 0). Bars (written before the run, from the definition): the star's
StellarFormationTime in the snapshot is earlier by the age in code time (2e6 yr / 0.97779e6 yr per code time unit) than in a twin
with TestStarInitAgeYr 0, to 1e-9; the twin's formation time lies within the run."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "single_stars_birth_age"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_collect_gas"))
from make_single_stars_collect_gas_ics import make_single_stars_collect_gas_ics, cell_id  # noqa: E402

YR_PER_TIME_UNIT = 3.085678e13 / 3.155e7  # [s -> yr]: 3.085678e13 km / (1 km/s) / seconds per year


@pytest.fixture(scope="module")
def built():
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return True


def run(tag, age_yr):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_collect_gas_ics(str(d / "output" / "snapshot_000.hdf5"))
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", f"{cell_id(12, 12, 12)}:35.0")
        p = p.replace("BoxSize                            15.84", f"BoxSize                            {L}")
        p = p.replace("TestStarInitAgeYr                  2.0e6", f"TestStarInitAgeYr                  {age_yr}")
        (d / f"{TEST_NAME}.params").write_text(p)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1], "r") as F:
        assert len(F["PartType4/Masses"]) == 1
        return float(F["PartType4/StellarFormationTime"][0])


def test_born_with_given_age(built):
    t0, t_age = run("zero", 0.0), run("age", 2.0e6)
    assert 0 <= t0 <= 1e-5, t0  # born at a step of the run
    assert abs((t0 - t_age) - 2.0e6 / YR_PER_TIME_UNIT) < 1e-9 * 2.0e6 / YR_PER_TIME_UNIT, (t0, t_age)
