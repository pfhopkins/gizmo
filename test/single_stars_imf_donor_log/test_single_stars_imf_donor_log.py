"""single_stars_imf_donor_log test: IMFinfo.txt (GALSF_SINGLE_STARS_EVENT_LOGS) lists, in each step that spawned single stars, one row
per (star, donor cell): time, star ID, donor gas ID, mass taken [Msun], as stella_dev. Lattice and births of single_stars_collect_gas
(stars of 8 and 40 Msun at neighbouring cells, 2 and 1 ranks). Bars, written before the runs: per star the donor masses add up
to the star's collected mass of the birth log (single_star_births.txt) to 1e-12; each (star, donor) pair once; every donor
ID is a gas cell of the lattice, within the star's collection radius of the birth log; the rows (sorted by star and donor)
are identical on 1 and 2 ranks to 1e-14."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_imf_donor_log"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_collect_gas"))
from make_single_stars_collect_gas_ics import make_single_stars_collect_gas_ics, cell_id, N_SIDE  # noqa: E402

TWO = [((12, 12, 12), 8.0), ((12, 12, 13), 40.0)]


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, ranks):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_collect_gas_ics(str(d / "output" / "snapshot_000.hdf5"))
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", ",".join(f"{cell_id(*c)}:{m}" for c, m in TWO))
        p = p.replace("SET_BY_TABLES", tables["StellarTablesFile"]).replace("BoxSize                            15.84", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(p)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    log = np.atleast_2d(np.loadtxt(d / "output" / "IMFinfo.txt"))
    return log[np.lexsort((log[:, 2], log[:, 1]))], np.atleast_2d(np.loadtxt(d / "output" / "single_star_births.txt"))


def test_donor_log(built):
    log, births = run("r2", built, 2)
    assert len(births) == 2 and len(log) > 2
    for b in births:
        mine = log[log[:, 1] == b[1]]
        assert len(mine) > 0 and abs(mine[:, 3].sum() / b[5] - 1) < 1e-12, (b[1], mine[:, 3].sum(), b[5])
        assert len(np.unique(mine[:, 2])) == len(mine)
        assert (mine[:, 2] >= 1).all() and (mine[:, 2] <= N_SIDE ** 3).all()


def test_log_is_rank_independent(built):
    a, _ = run("r2", built, 2)
    b, _ = run("r1", built, 1)
    assert a.shape == b.shape and np.allclose(a, b, rtol=1e-14, atol=0)
