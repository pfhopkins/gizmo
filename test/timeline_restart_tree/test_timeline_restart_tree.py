"""A restart that doubles TimeMax also halves the time stamps of the tree in the restart files. Gas, 2 ranks, stopped part-way through
long steps. Without that the first drift of a node runs over a wrong interval: the restart stops with endrun 90000103 or drifts wrong.
Bars: the restart reaches the end; it ends where a plain restart at the same step ends, velocities within 1e-10 km/s."""

import glob
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "timeline_restart_tree"
TEST_DIR = f"test/{TEST_NAME}"
T2 = 0.002
DV_MAX = 1e-10
STOP_STEP = 8


def _final(out):
    with h5py.File(sorted(glob.glob(f"{out}/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType0"]; o = np.argsort(g["ParticleIDs"][:])
        return f["Header"].attrs["Time"], g["Coordinates"][:][o], g["Velocities"][:][o]


def _stop_and_restart(first_params):
    shutil.rmtree("output", ignore_errors=True)
    subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", first_params, "0"], stdout=open("first.out", "w"), stderr=subprocess.STDOUT,
                   env=dict(os.environ, GIZMO_TEST_RESTART_AT_STEP=str(STOP_STEP)))
    res = subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", f"{TEST_NAME}.params", "1"], stdout=open("restart.out", "w"), stderr=subprocess.STDOUT)
    log = open("restart.out").read()
    assert res.returncode == 0, "restart run failed: " + log[-400:]
    return log


def test_timeline_restart_tree():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME, 1)
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        log = _stop_and_restart(f"{TEST_NAME}.params")
        assert "adjust integer timeline" not in log
        shutil.rmtree("output_plain", ignore_errors=True); shutil.move("output", "output_plain")
        log = _stop_and_restart("halfway.params")
        assert "adjust integer timeline" in log, "the restart did not halve the timeline"
        ta, xa, va = _final("output_plain"); tb, xb, vb = _final("output")
    finally:
        os.chdir(cwd)
    assert abs(ta - T2) < 1e-12 and abs(tb - T2) < 1e-12
    dx, dv = np.abs(xa - xb).max(), np.abs(va - vb).max()
    print(f"halved vs plain restart: max|dx| {dx:.3e} max|dv| {dv:.3e}")
    assert dv < DV_MAX
