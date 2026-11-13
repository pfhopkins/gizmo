"""A sink moved onto the potential minimum of its neighbours stays inside every node box above it (tree geometry audit build)."""

import glob
import re
import subprocess
import sys

import h5py

from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir

TEST_NAME = "tree_node_boxes"
TEST_DIR = f"test/{TEST_NAME}"


def test_repositioned_sink_inside_node_boxes():
    """bar: the sink is re-centred, and no audit finds a particle outside a box above it"""
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    out = variant_output_dir(TEST_NAME, ())
    line = [l for l in open(f"{out}/cpu.txt") if l.startswith("TreeGeoAudit:")][-1]
    audits, bad = (int(v) for v in re.findall(r"audits=(\d+) box_excess_audits=(\d+)", line)[0])
    with h5py.File(sorted(glob.glob(f"{out}/snapshot_*.hdf5"))[-1]) as f:
        moved = 0.5 + 1.2e-2 - f["PartType5/Coordinates"][0][0]
    print(line.strip(), f"sink moved {moved:.4f}")
    assert audits > 0 and moved > 1e-3
    assert bad == 0
