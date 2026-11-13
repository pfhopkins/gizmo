"""A particle whose position is set rather than drifted stays inside every node box above it: sink re-centred on its
accreted mass and jet cells spawned into the standing tree, audit build (TEST_TREE_NODE_BOX_AUDIT)."""

import glob
import re
import subprocess
import sys

import h5py
import numpy as np

from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir

TEST_NAME = "tree_node_boxes"
TEST_DIR = f"test/{TEST_NAME}"
M_SPAWN = 7.8125e-06


def test_set_positions_inside_node_boxes():
    """bar (set before the run): the sink accretes and spawns jet cells, and no particle is ever outside a box above it"""
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    lines = [l for l in open(f"{TEST_DIR}/test_{TEST_NAME}.out") if l.startswith("TREE_NODE_BOX_AUDIT")]
    assert lines, "no audit line"
    a = {k: float(v) for k, v in re.findall(r"(\w+) (\S+)", lines[-1].split(None, 1)[1])}
    with h5py.File(sorted(glob.glob(f"{variant_output_dir(TEST_NAME, ())}/snapshot_*.hdf5"))[-1]) as f:
        spawned = int(np.sum(np.isclose(f["PartType0/Masses"][:], M_SPAWN, rtol=0.01)))
        sink = f["PartType5/Masses"][0]
    print(lines[-1].strip(), f"spawned {spawned} sink {sink:.4f}")
    assert a["calls"] > 0 and spawned > 0 and sink > 0.5
    assert a["outside"] == 0
