"""Falling Sedov: WAKEUP kick test.

A point blast in cold uniform gas wakes cells sitting on long timesteps. The whole box falls in the
uniform field of GRAVITY_TESTPROBLEM_RT (g = -0.5 in y). The box is periodic and hydro forces act in
pairs, so the centre-of-mass velocity must stay exactly v_cm,y = -0.5 t whatever the blast does. A woken
cell that loses part of its field kick shows up directly as a centre-of-mass velocity error. Energy cannot
see this: the blast's own integration error is larger than the missing kick energy.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from gizmo.test import build_and_run_test, default_mpi_ranks

TEST_NAME = "wakeup_falling_sedov"
TEST_DIR = Path(__file__).resolve().parent
G_FIELD = 0.5
TOL = 1e-3  # hydro across timebins leaves ~1e-4 even with g = 0; lost wakeup kicks give ~3e-2


def generate_ics():
    ic_file = TEST_DIR / f"{TEST_NAME}_ics.hdf5"
    if not ic_file.exists():
        sys.path.insert(0, str(TEST_DIR))
        from make_wakeup_falling_sedov_ics import make_wakeup_falling_sedov_ics
        make_wakeup_falling_sedov_ics(str(ic_file))


def synced_momentum(logfile):
    """(t, M, P_y) from the ENERGY_BUDGET_DIAGNOSTIC full-synchronisation lines."""
    pat = re.compile(r"Energy \(synced\) t=(\S+) .* M_gas=(\S+) P_gas=(\S+) (\S+) (\S+)")
    rows = [m.groups() for m in map(pat.search, open(logfile)) if m]
    t, M, _, Py, _ = np.array(rows, dtype=float).T
    return t, M, Py


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(default_mpi_ranks(max_ranks=4), 1)])
def test_wakeup_falling_sedov(num_mpi_ranks, num_omp_threads):
    generate_ics()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)

    t, M, Py = synced_momentum(TEST_DIR / f"test_{TEST_NAME}.out")
    ok = t > 0
    assert ok.sum() >= 10, f"only {ok.sum()} synchronised samples"
    err = np.abs(Py[ok] / M[ok] + G_FIELD * t[ok]) / (G_FIELD * t[ok])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(t[ok], err, "o-", ms=3)
    ax.axhline(TOL, color="k", ls="--", lw=1)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|v_{\rm cm,y} + g t| / (g t)$")
    fig.tight_layout()
    fig.savefig(TEST_DIR / "com_velocity_error.png", dpi=130)
    plt.close(fig)

    assert err.max() < TOL, f"centre-of-mass velocity error {err.max():.3e} (woken cells lose field kicks)"
