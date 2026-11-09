"""A restart that doubles TimeMax halves the integer timeline. With the run stopped part-way through long steps, it must
end where a plain restart at the same step ends, whose timeline is the same."""

import glob
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

from gizmo.test import build_gizmo_for_test, mpi_launch_command, run_test

TEST_NAME = "timeline_restart"
TEST_DIR = f"test/{TEST_NAME}"
T2 = 0.0004
DV_MAX = 1e-12
STOP_STEP = 507


def _final(out):
    with h5py.File(sorted(glob.glob(f"{out}/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType4"]; o = np.argsort(g["ParticleIDs"][:])
        return f["Header"].attrs["Time"], g["Coordinates"][:][o], g["Velocities"][:][o]


def _stop_and_restart(first_params):
    """Stop at step STOP_STEP with first_params, restart with the full TimeMax; returns the restart log."""
    shutil.rmtree("output", ignore_errors=True)
    subprocess.run(mpi_launch_command(1, 1) + ["./GIZMO", first_params, "0"], stdout=open("first.out", "w"), stderr=subprocess.STDOUT,
                   env=dict(os.environ, GIZMO_TEST_RESTART_AT_STEP=str(STOP_STEP)))
    res = subprocess.run(mpi_launch_command(1, 1) + ["./GIZMO", f"{TEST_NAME}.params", "1"], stdout=open("restart.out", "w"), stderr=subprocess.STDOUT)
    assert res.returncode == 0, "restart run failed"
    return open("restart.out").read()


def test_timeline_restart():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME, 1)
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        log = _stop_and_restart(f"{TEST_NAME}.params")  # plain restart: same TimeMax, same timeline
        assert "adjust integer timeline" not in log
        shutil.rmtree("output_plain", ignore_errors=True); shutil.move("output", "output_plain")
        log = _stop_and_restart("halfway.params")  # TimeMax doubled at the restart
        assert "adjust integer timeline" in log, "the restart did not halve the timeline"
        ta, xa, va = _final("output_plain"); tb, xb, vb = _final("output")
    finally:
        os.chdir(cwd)
    assert abs(ta - T2) < 1e-12 and abs(tb - T2) < 1e-12
    dx, dv = np.abs(xa - xb).max(), np.abs(va - vb).max()
    print(f"halved vs plain restart: max|dx| {dx:.3e} max|dv| {dv:.3e}")
    assert dv < DV_MAX
