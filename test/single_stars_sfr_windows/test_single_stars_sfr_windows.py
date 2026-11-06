"""single_stars_sfr_windows test: sfr_rism.txt (GALSF_SINGLE_STARS_EVENT_LOGS), as stella_dev: in the steps that build the tree,
the star-formation rate [Msun/yr] over the last 50, 40, 30, 20, 5 and 1 Myr from the stars' drawn masses and the number of stars.
Two 20 Msun stars 3 Myr old (the lattice of single_stars_forced_events), 8 kyr, 2 ranks. Bars, written before the run: at every row
N_stars = 2; SFR_w = 2 x 20 Msun / (w Myr) to 1e-6 (the file prints 7 digits; the first version of this bar, 1e-12, was logically wrong) for w = 50, 40, 30, 20, 5 and 0 for 1 Myr (3 Myr old: inside the windows
down to 5 Myr, outside 1 Myr); row times strictly increasing, at least one row (the start) and not more than the number of steps."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_sfr_windows"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_forced_events"))
from make_single_stars_forced_events_ics import make_single_stars_forced_events_ics  # noqa: E402


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def test_sfr_windows(built):
    d = TEST_DIR / "run_windows"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_forced_events_ics(str(d / "output" / "snapshot_000.hdf5"), age_yr=3.0e6)
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", built["StellarTablesFile"])
        (d / f"{TEST_NAME}.params").write_text(p.replace("BoxSize                            45.0", f"BoxSize                            {L}"))
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    rows = np.atleast_2d(np.loadtxt(d / "output" / "sfr_rism.txt"))
    steps = int(re.findall(r"Step (\d+), Time", (d / "output" / "cpu.txt").read_text())[-1])
    assert 1 <= len(rows) <= steps + 1, (len(rows), steps)
    assert (rows[:, 1] == 2).all() and np.all(np.diff(rows[:, 0]) > 0)
    want = np.array([40.0 / (w * 1.0e6) for w in (50, 40, 30, 20, 5)] + [0.0])
    assert np.allclose(rows[:, 2:], want, rtol=1e-6, atol=0), rows[0]
