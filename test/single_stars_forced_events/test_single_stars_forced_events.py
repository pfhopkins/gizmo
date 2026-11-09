"""single_stars_forced_events test: TEST_SINGLE_STARS_SN_AT_FORCED_TIMES (star ID k of the first TestNumStars explodes at
TestSNTime0 + (k-1) TestSNSpacing, whatever the table lifetime) and TEST_SINGLE_STARS_NO_GAS_IN_REACH_UNTIL_TIME (until
TestFBNoGasUntilTime the feedback of every event finds no gas, so a supernova waits and fires later: the retry of an event
without neighbour weight). Two 20 Msun stars (death at ~10 Myr, run 8 kyr), 1 Myr old, in a 24^3 lattice. See README.md.
Bars, written before the runs: each star has exactly one supernova row (kind 1); its time t satisfies t_forced <= t <
t_forced + 1.5e-4 (code time; the step is at most 1e-4); with no gas until 0.004 the first star (forced 0.002) fires at
0.004 <= t < 0.004 + 1.5e-4 after at least 10 steps without deposit, the second (forced 0.005, later than 0.004) is not delayed;
mass (gas + stars) is conserved to 1e-12 and the star ends with the table remnant (mass residual 0)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "single_stars_forced_events"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_single_stars_forced_events_ics import make_single_stars_forced_events_ics  # noqa: E402


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, **over):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_forced_events_ics(str(ic))
        lines = []
        for l in (TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines():
            k = l.split()[0] if l.split() else ""
            if k in over:
                l = f"{k:34s} {over[k]}"
            lines.append(l.replace("SET_BY_TEST", tables["StellarTablesFile"]))
        text = "\n".join(lines) + "\n"
        text = text.replace("BoxSize                            45.0", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(text)
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(mpi_launch_command(2) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                           stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    events = d / "output" / "single_star_events.txt"
    rows = np.atleast_2d(np.loadtxt(events)) if events.exists() else np.zeros((0, 12))
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    return ic, snaps[-1], rows


def total_mass(fname):
    with h5py.File(fname, "r") as F:
        return sum(F[g]["Masses"][()].sum() for g in ("PartType0", "PartType4"))


def sn_rows(rows):
    return {int(r[1]): r for r in rows if int(r[2]) == 1}


def test_forced_times(built):
    ic, snap, rows = run("times", built)
    sn = sn_rows(rows)
    assert sorted(sn) == [1, 2] and len([r for r in rows if int(r[2]) == 1]) == 2, rows
    for k, t_forced in ((1, 0.002), (2, 0.005)):
        assert t_forced <= sn[k][0] < t_forced + 1.5e-4, (k, sn[k][0])
    assert abs(total_mass(snap) / total_mass(ic) - 1) < 1e-12


def test_no_gas_retry(built):
    ic, snap, rows = run("nogas", built, TestFBNoGasUntilTime=0.004)
    sn = sn_rows(rows)
    assert sorted(sn) == [1, 2], rows
    assert 0.004 <= sn[1][0] < 0.004 + 1.5e-4, sn[1][0]
    assert sn[1][11] >= 10, sn[1]  # steps without deposit before the event coupled
    assert 0.005 <= sn[2][0] < 0.005 + 1.5e-4 and sn[2][11] == 0, sn[2]
    assert abs(sn[1][10]) < 1e-9 and abs(sn[2][10]) < 1e-9, (sn[1][10], sn[2][10])  # mass residual [Msun]
    assert abs(total_mass(snap) / total_mass(ic) - 1) < 1e-12
