"""single_stars_instant_sn test: GALSF_SINGLE_STARS_INSTANT_SN_BEFORE_TIME makes every single star of drawn mass >= 8 Msun
explode at the first feedback step while the code time is below TimeInstantSN, whatever its age (lifetime 0 for
such stars). Two 20 Msun stars 1 Myr old (table death ~10 Myr) in the lattice of single_stars_forced_events, run 4 kyr.
Bars, written before the runs: with TimeInstantSN 0.002 each star has exactly one supernova row, at a time < 3e-4 (the first
steps; the step is at most 1e-4), and mass (gas + stars) is conserved to 1e-12; with TimeInstantSN 0 (no effect) no
supernova occurs in the run."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_instant_sn"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_forced_events"))
from make_single_stars_forced_events_ics import make_single_stars_forced_events_ics  # noqa: E402


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, t_instant):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_forced_events_ics(str(ic))
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
        p = p.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        p = p.replace("TimeInstantSN                      0.002", f"TimeInstantSN                      {t_instant}")
        (d / f"{TEST_NAME}.params").write_text(p)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    events = d / "output" / "single_star_events.txt"
    rows = np.atleast_2d(np.loadtxt(events)) if events.exists() else np.zeros((0, 12))
    return ic, sorted((d / "output").glob("snapshot_*.hdf5"))[-1], rows[rows[:, 2] == 1] if len(rows) else rows


def total_mass(fname):
    with h5py.File(fname, "r") as F:
        return sum(F[g]["Masses"][()].sum() for g in ("PartType0", "PartType4"))


def test_instant_sn(built):
    ic, snap, sn = run("instant", built, 0.002)
    assert sorted(sn[:, 1].astype(int).tolist()) == [1, 2], sn
    assert (sn[:, 0] < 3e-4).all(), sn[:, 0]
    assert abs(total_mass(snap) / total_mass(ic) - 1) < 1e-12


def test_no_effect_at_zero(built):
    ic, snap, sn = run("zero", built, 0)
    assert len(sn) == 0, sn
