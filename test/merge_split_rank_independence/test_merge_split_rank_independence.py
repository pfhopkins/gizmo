"""MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: the daughters of a split are placed at the same positions on 1, 2 and 4 ranks.

Plain MFM in a periodic box without gravity; a lattice at rest in which a tenth of the cells is above the split mass, once
jittered and once exact. On the exact lattice the direction a cell is split along (the eigenvector of the largest eigenvalue of
its neighbour-distribution matrix) is not unique, so it must not be left to round-off. The first snapshot after the split is
compared between rank counts, cells matched by (ID, child number, generation).
- with the flag, both lattices: the same cells exist, positions agree to 1e-12 of the kernel length, thermal + kinetic energy
  to 1e-10;
- control, without the flag, jittered lattice: the 4-rank run differs from the 1-rank run by more than that (the nearest
  neighbour of some split cell is on another rank); if it does not, the box tests nothing and the test fails.
"""

import glob
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

sys.path.insert(0, dirname(abspath(__file__)))
from gizmo.test import build_gizmo_for_test, mpi_launch_command
from make_merge_split_rank_independence_ics import make_ics

TEST_NAME = "merge_split_rank_independence"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
FLAG = "MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS"
JITTER = {"jittered": 0.15, "lattice": 0.0}
TOL_POS, TOL_E = 1e-12, 1e-10


def _run(binary, box, nranks, tag):
    d = join(TEST_DIR, f"run_{tag}_{box}_r{nranks}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(join(d, "output"))
    n_ic, n_heavy = make_ics(join(d, f"{TEST_NAME}_ics.hdf5"), JITTER[box])
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), d)
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(nranks, 1) + [binary, f"{TEST_NAME}.params"], stdout=log, stderr=subprocess.STDOUT, cwd=d)
    assert r.returncode == 0, f"{tag}, {box}, {nranks} ranks: " + open(join(d, "run.log")).read()[-800:]
    for snap in sorted(glob.glob(join(d, "output", "snapshot_*.hdf5")))[1:]:  # the first output after the split
        with h5py.File(snap, "r") as f:
            p = f["PartType0"]
            if len(p["ParticleIDs"]) == n_ic + n_heavy:
                key = [tuple(k) for k in np.stack([p["ParticleIDs"][:], p["ParticleChildIDsNumber"][:], p["ParticleIDGenerationNumber"][:]], 1).astype(np.int64)]
                order = sorted(range(len(key)), key=lambda n: key[n])
                e = p["Masses"][:] * (p["InternalEnergy"][:] + 0.5 * (p["Velocities"][:] ** 2).sum(1))
                return dict(key=[key[n] for n in order], pos=p["Coordinates"][:][order], h=p["KernelMaxRadius"][:][order], E=e.sum(), box=f["Header"].attrs["BoxSize"])
    raise AssertionError(f"{tag}, {box}, {nranks} ranks: no snapshot in which every heavy cell has split exactly once")


@pytest.fixture(scope="module")
def runs():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    binaries = {}
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        for tag, extra in (("on", (FLAG,)), ("off", ())):
            build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
            binaries[tag] = join(TEST_DIR, f"GIZMO_{tag}")
            shutil.move(join(TEST_DIR, "GIZMO"), binaries[tag])
    finally:
        os.chdir(cwd)
    return {(tag, box, n): _run(binaries[tag], box, n, tag) for tag, box in (("on", "jittered"), ("on", "lattice"), ("off", "jittered")) for n in (1, 2, 4)}


def _difference(a, b):
    """largest position difference in kernel lengths (None if other cells exist) and the relative energy difference"""
    if a["key"] != b["key"]:
        return None, None
    dp = a["pos"] - b["pos"]
    dp -= a["box"] * np.round(dp / a["box"])
    return (np.sqrt((dp**2).sum(1)) / a["h"]).max(), abs(a["E"] - b["E"]) / a["E"]


@pytest.mark.parametrize("box", ["jittered", "lattice"])
def test_split_daughters_independent_of_ranks(runs, box):
    for n in (2, 4):
        d, de = _difference(runs[("on", box, 1)], runs[("on", box, n)])
        print(f"with the flag, {box}, {n} ranks vs 1: largest position difference {d} kernel lengths, energy {de}")
        assert d is not None, f"{n} ranks: other cells than on 1 rank"
        assert d <= TOL_POS and de <= TOL_E


def test_control_without_the_flag_depends_on_ranks(runs):
    d, de = _difference(runs[("off", "jittered", 1)], runs[("off", "jittered", 4)])
    print(f"without the flag, 4 ranks vs 1: largest position difference {d} kernel lengths, energy {de}")
    assert d is None or d > TOL_POS, "no split cell has its nearest neighbour on another rank: the box tests nothing"
