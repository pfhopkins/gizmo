"""star_direct_siblings: the direct star-star sum between Type-4 stars that share one ParticleID.

Stars spawned from one cell keep the cell's ParticleID and differ in child number and generation. The direct sum
(STARS_HERMITE_DIRECT_GRAVITY) must count such siblings as different stars: each feels the others' force.

Four stars at rest, three with one ID (see make_star_direct_siblings_ics.py), G = 1, softening 1e-4 of the separations.
Two builds, each on 1 and 2 ranks: tree only (Config.sh as it is) and the direct sum. Checked against the Newtonian
pair sum of the IC positions:

  * the acceleration of every star, relative error at most ACC_RTOL, in both builds. A snapshot restart writes no
    snapshot at its start time, so the accelerations are read from a run of TimeMax = 1e-9, in which no star moves by
    a bit of its coordinates;
  * the two builds agree with each other to the same bar;
  * the displacement of every star at TimeMax = 1e-3 (about 1e-3 dynamical times) is acceleration x time^2 / 2, to
    DISP_RTOL.
"""
import glob
import os
import sys
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

import subprocess

from gizmo.test import build_gizmo_for_test, mpi_launch_command, write_params_with_overrides

TEST_NAME = "star_direct_siblings"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
VARIANTS = {"tree": (), "direct": ("STARS_HERMITE_DIRECT_GRAVITY",)}
RANKS = (1, 2)
ACC_RTOL = 1e-10  # double-precision output of an exact pair sum
T_FORCE = 1e-9    # TimeMax of the run the accelerations are read from
DISP_RTOL = 1e-4  # dx = a t^2 / 2 holds to (t / t_dyn)^2 ~ 1e-5 at TimeMax

sys.path.insert(0, TEST_DIR)
from make_star_direct_siblings_ics import (CHILD, IDS, POS, analytic_accelerations,  # noqa: E402
                                           make_star_direct_siblings_ics)


def _read(snap):
    """Stars in the order of the IC, matched by (ID, child number)."""
    with h5py.File(snap, "r") as f:
        g = f["PartType4"]
        key = list(zip(g["ParticleIDs"][:].tolist(), g["ParticleChildIDsNumber"][:].tolist()))
        order = [key.index(k) for k in zip(IDS.tolist(), CHILD.tolist())]
        return float(f["Header"].attrs["Time"]), g["Acceleration"][:][order], g["Coordinates"][:][order]


def _run_from_snapshot(ranks, overrides):
    """One run with restart flag 2 (the IC is read as a snapshot, which keeps its child numbers and generations), in TEST_DIR."""
    params = write_params_with_overrides(f"{TEST_NAME}.params", overrides)
    with open(f"test_{TEST_NAME}_{overrides['OutputDir']}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks) + ["./GIZMO", params, "2"], stdout=out, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)


@pytest.fixture(scope="module")
def runs():
    """Build each variant once and run it on each rank count; returns {(variant, ranks): (acc at t=0, t_end, pos at t_end)}."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    out = {}
    cwd = os.getcwd()
    try:
        for variant, flags in VARIANTS.items():
            os.chdir(REPO)
            build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
            os.chdir(TEST_DIR)
            make_star_direct_siblings_ics(f"{TEST_NAME}_ics.hdf5")
            for ranks in RANKS:
                outdir = f"output_{variant}_{ranks}"
                _run_from_snapshot(ranks, {"OutputDir": outdir + "_force", "TimeMax": T_FORCE, "TimeBetSnapshot": T_FORCE, "TimeBetStatistics": T_FORCE, "MaxSizeTimestep": T_FORCE})
                _, acc0, pos0 = _read(sorted(glob.glob(join(outdir + "_force", "snapshot_*.hdf5")))[-1])
                assert np.array_equal(pos0, POS), "stars moved in the force run"
                _run_from_snapshot(ranks, {"OutputDir": outdir})
                t_end, _, pos = _read(sorted(glob.glob(join(outdir, "snapshot_*.hdf5")))[-1])
                out[(variant, ranks)] = (acc0, t_end, pos)
    finally:
        os.chdir(cwd)
    return out


def _rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_star_feels_every_other(runs, variant, ranks):
    """Acceleration at the IC positions and displacement at TimeMax against the Newtonian pair sum over all four stars."""
    acc0, t_end, pos = runs[(variant, ranks)]
    exact = analytic_accelerations()
    err_a, err_x = _rel(acc0, exact), _rel(pos - POS, 0.5 * exact * t_end**2)
    print(f"{variant} {ranks} rank(s): max rel. error acceleration {err_a.max():.3e}, displacement {err_x.max():.3e}; "
          f"per star acceleration {np.array2string(err_a, precision=2)}")
    assert err_a.max() <= ACC_RTOL, f"{variant}: acceleration off the pair sum by {err_a} (stars with one ID skipped as self?)"
    assert t_end > 0 and err_x.max() <= DISP_RTOL, f"{variant}: displacement at t = {t_end:g} off acceleration x time^2 / 2 by {err_x}"


@pytest.mark.parametrize("ranks", RANKS)
def test_direct_sum_matches_tree(runs, ranks):
    """The direct sum and the tree-only build give the same accelerations."""
    err = _rel(runs[("direct", ranks)][0], runs[("tree", ranks)][0])
    print(f"{ranks} rank(s): direct sum against tree, max rel. difference {err.max():.3e}")
    assert err.max() <= ACC_RTOL, f"direct sum differs from the tree-only build by {err}"
