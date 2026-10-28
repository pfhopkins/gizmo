"""wakeup_falling_box: kick bookkeeping of particles woken by the WAKEUP limiter.

A Sedov blast in a periodic box under a uniform acceleration g. Pressure forces cancel pairwise, so the
mass-weighted mean velocity must follow v_cm = g*t. A cell woken mid-step has been kicked only to its old
step's midpoint and is not in the first half-kick of its new step; unless process_wake_ups supplies both
kicks, each woken cell leaves an error ~ g*(dt_old + dt_new)/2 in v_cm. Only the final snapshot is checked: every particle is
synchronised there, while at intermediate outputs inactive particles hold half-kicked velocities.
"""

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

    with h5py.File(final_snap, "r") as F:
        t = F["Header"].attrs["Time"]
        m = F["PartType0/Masses"][:].astype(np.float64)
        vy = F["PartType0/Velocities"][:, 1].astype(np.float64)
    rel_err = (m @ vy / m.sum() - G_Y * t) / (G_Y * t)
    # 4e-6 with the woken-particle kicks, 2.5e-3 with the step truncated but not kicked, 1.3e-2 with the
    # legacy kick reversal
    assert abs(rel_err) < 1e-4, f"v_cm,y deviates from g*t by {rel_err:.3e} (relative) over {n_woken} wakeups"
