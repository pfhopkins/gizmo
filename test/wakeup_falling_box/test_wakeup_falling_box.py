"""wakeup_falling_box: kick bookkeeping of particles woken by the WAKEUP limiter.

A Sedov blast in a periodic box under a uniform acceleration g. Pressure forces cancel pairwise, so the
mass-weighted mean velocity must follow v_cm = g*t. A cell woken mid-step has been kicked only to its old
step's midpoint and is not in the first half-kick of its new step; unless process_wake_ups supplies both
kicks, each woken cell leaves an error ~ g*(dt_old + dt_new)/2 in v_cm. Only the final snapshot is checked: every particle is
synchronised there, while at intermediate outputs inactive particles hold half-kicked velocities.
"""

import re
import pytest
import numpy as np
import h5py
from pathlib import Path

from gizmo.test import (
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    assert_final_time,
    get_final_snapshot,
)

TEST_NAME = "wakeup_falling_box"
TEST_DIR = Path(__file__).parent
G_Y = -0.5  # GravAccel_RayleighTaylorTest
MAX_WAKEUP_DEEPENING = 2  # bins; 0 with the wake-up floor, 4 without

# one row of an "Occupied timebins" report: bin, non-cell and cell counts, dt, cumulative, flags, avg-time, cpu-frac
_TIMEBIN_ROW = re.compile(r"^ [X ]  bin=\s*(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+\d+ . .\s+(\S+)\s+(\S+)%$")


def timebin_reports(stdout_file):
    """Every "Occupied timebins" report in a GIZMO stdout log, as (sync point, woken, rows): woken says whether a
    wake-up had been processed before the report, rows are (bin, particles, avg-time, cpu-frac %). Fails if there is
    nothing to parse, so the checks built on it cannot pass vacuously."""
    reports, rows, sync, woken = [], None, None, False
    with open(stdout_file, errors="replace") as f:
        for line in f:
            if rows is not None:
                m = _TIMEBIN_ROW.match(line.rstrip("\n"))
                if m:
                    rows.append((int(m.group(1)), int(m.group(2)) + int(m.group(3)), float(m.group(4)),
                                 float(m.group(5))))
                    continue
                reports.append((sync, woken, rows))
                rows = None
            if line.startswith("Sync-Point "):
                sync = int(line.split()[1].rstrip(","))
            elif line.startswith("Occupied timebins"):
                rows = []
            elif "activated (in wakeup check)" in line:
                woken = True
    n_rows = sum(len(r) for _, _, r in reports)
    assert len(reports) > 100 and n_rows > 0, (
        f"parsed {len(reports)} 'Occupied timebins' reports with {n_rows} rows from {stdout_file}: the report "
        f"format changed or the run did not print it, so the timebin checks cannot run")
    return reports


def wakeup_deepening(reports):
    """Bins by which the deepest occupied bin drops below its depth before the first wake-up. The reference is the
    sync points from the first one (all cells start in bin 0 at sync point 0) up to the first wake-up: there the bins
    are what the timestep criteria ask for, independent of the ICs, parameters and timebase."""
    def deepest(rows):
        return min(b for b, n, _, _ in rows if n > 0)

    ref = [deepest(r) for s, w, r in reports if s and not w and r]
    after = [deepest(r) for s, w, r in reports if w and r]
    assert ref and after, (f"{len(ref)} reports before the first wake-up and {len(after)} after it: the wake-up "
                           f"floor check needs both")
    return min(ref) - min(after)


def generate_ics():
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        from make_wakeup_falling_box_ics import make_wakeup_falling_box_ics
        make_wakeup_falling_box_ics(str(ic_file))


@pytest.mark.parametrize(
    "num_mpi_ranks,num_omp_threads",
    [(default_mpi_ranks(), default_omp_threads())],
)
def test_wakeup_falling_box(num_mpi_ranks, num_omp_threads):
    generate_ics()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(TEST_NAME)
    assert_final_time(final_snap, TEST_NAME)

    # the check is vacuous without wakeups
    with open(TEST_DIR / f"test_{TEST_NAME}.out") as f:
        n_woken = sum(int(l.split()[0]) for l in f if "activated (in wakeup check)" in l)
    assert n_woken > 1000, f"only {n_woken} wakeups: the test no longer exercises the wakeup path"

    # Wake-up floor: a hydro wake-up may not place a cell below the lowest occupied active bin. Without the floor,
    # cells woken by the blast shell land up to 3 bins below it and cascade, 4 bins below the pre-wake-up depth.
    reports = timebin_reports(TEST_DIR / f"test_{TEST_NAME}.out")
    drop = wakeup_deepening(reports)
    assert drop <= MAX_WAKEUP_DEEPENING, (
        f"the deepest occupied timebin fell {drop} bins below its depth before the first wake-up "
        f"(> {MAX_WAKEUP_DEEPENING}): hydro wake-ups are placing cells below the wake-up floor")

    with h5py.File(final_snap, "r") as F:
        t = F["Header"].attrs["Time"]
        m = F["PartType0/Masses"][:].astype(np.float64)
        vy = F["PartType0/Velocities"][:, 1].astype(np.float64)
    rel_err = (m @ vy / m.sum() - G_Y * t) / (G_Y * t)
    # 4e-6 with the woken-particle kicks, 2.5e-3 with the step truncated but not kicked, 1.3e-2 with the
    # legacy kick reversal
    assert abs(rel_err) < 1e-4, f"v_cm,y deviates from g*t by {rel_err:.3e} (relative) over {n_woken} wakeups"
