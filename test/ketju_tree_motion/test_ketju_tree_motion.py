"""The gravity tree follows KETJU: audit build (TEST_KETJU_TREE_MOTION_AUDIT) on a cluster of hard binaries.

- 1 rank: chain members stay under their nodes' motion bounds and boxes; the velocity trick and the tree build leave the
  nodes' bulk velocity equal to the members' to roundoff.
- 1 rank, colliding pairs: the same across KETJU's mergers.
- 2 ranks: the same bounds on every rank's copy of the top-level nodes; imported leaves of chain members stay with their
  chains (without this they fly off at orbital speed until the next build)."""

import re
import subprocess
import sys

import glob

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, clean_test_outputs, variant_output_dir

TEST_NAME = "ketju_tree_motion"
TEST_DIR = f"test/{TEST_NAME}"
LEAF_MEAN_MAX = 3e-6


def _audit():
    lines = [l for l in open(f"{TEST_DIR}/test_{TEST_NAME}.out") if l.startswith("KETJU_TREE_AUDIT")]
    assert lines, "no KETJU_TREE_AUDIT line"
    print(lines[-1])
    return {k: float(v) for k, v in re.findall(r"(\w+) (\S+)", lines[-1].split(None, 1)[1])}


def _run(ranks, ic="ketju_tree_motion_ics", flags=(), params=None):
    subprocess.run([sys.executable, "make_ketju_tree_motion_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, ranks, 1, extra_config_flags=flags, param_overrides=dict({"InitCondFile": ic}, **(params or {})))
    return _audit()








def test_regions_across_ranks():
    """chains with members on several ranks, softening far above the orbits so the tree plays no part: the pairs keep their orbits"""
    subprocess.run([sys.executable, "make_ketju_tree_motion_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    fl = ("TEST_KETJU_REGION_LOG",)
    build_and_run_test(TEST_NAME, 4, 1, extra_config_flags=fl, param_overrides={"Softening_Type4": 1e-3, "Softening_Type4_MaxPhysLimit": 1e-3})
    assert "KETJU_REGION_ACROSS_RANKS" in open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()
    with h5py.File(sorted(glob.glob(variant_output_dir(TEST_NAME, fl) + "/snapshot_*.hdf5"))[-1], "r") as f:
        g = f["PartType4"]; o = np.argsort(g["ParticleIDs"][:]); m, x, v = g["Masses"][:][o], g["Coordinates"][:][o], g["Velocities"][:][o]
    dr, dv = x[::2] - x[1::2], v[::2] - v[1::2]; mu = 43007.1 * (m[::2] + m[1::2])
    a = 1 / (2 / np.linalg.norm(dr, axis=1) - (dv * dv).sum(1) / mu)
    assert np.median(np.abs(a / (1000 * 4.848137e-9) - 1)) < 0.01


