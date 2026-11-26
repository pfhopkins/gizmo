"""MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: a merge/split pass gives the same cells on 1, 2 and 4 ranks.

Plain MFM in a periodic box without gravity; a jittered lattice at rest in which a tenth of the cells is above the split mass
(split box; the same on an exact lattice, where the direction to split along is not unique: lattice box) or below the
merge mass (merge box). The first snapshot after the pass is compared between rank counts, cells
matched by (ID, child number, generation).
- with the flag: the same cells exist with the same masses, positions agree to 1e-12 of the kernel length, thermal + kinetic
  energy to 1e-10;
- merge box: the first pass merges within 20 % of the number the rank-local choice merges (1 rank);
- control, without the flag: the 4-rank run differs from the 1-rank run (the nearest neighbour or the merge partner of some cell
  is on another rank); if it does not, the box tests nothing and the test fails.
"""

import glob
import os
import re
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
TOL_POS, TOL_M, TOL_E = 1e-12, 1e-12, 1e-10


def _run(binary, kind, nranks, tag):
    d = join(TEST_DIR, f"run_{tag}_{kind}_r{nranks}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(join(d, "output"))
    n_ic, n_special = make_ics(join(d, f"{TEST_NAME}_ics.hdf5"), "split" if kind == "lattice" else kind, 0.0 if kind == "lattice" else 0.15)
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), d)
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(nranks, 1) + [binary, f"{TEST_NAME}.params"], stdout=log, stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0, f"{tag}, {kind}, {nranks} ranks: " + text[-800:]
    merged = [int(x) for x in re.findall(r"(\d+) particles merged", text) if int(x) > 0]
    snap = sorted(glob.glob(join(d, "output", "snapshot_*.hdf5")))[1]  # the first output after the pass
    with h5py.File(snap, "r") as f:
        p = f["PartType0"]
        key = [tuple(k) for k in np.stack([p["ParticleIDs"][:], p["ParticleChildIDsNumber"][:], p["ParticleIDGenerationNumber"][:]], 1).astype(np.int64)]
        order = sorted(range(len(key)), key=lambda n: key[n])
        e = p["Masses"][:] * (p["InternalEnergy"][:] + 0.5 * (p["Velocities"][:] ** 2).sum(1))
        res = dict(key=[key[n] for n in order], pos=p["Coordinates"][:][order], m=p["Masses"][:][order], h=p["KernelMaxRadius"][:][order], E=e.sum(),
                   box=f["Header"].attrs["BoxSize"], n_ic=n_ic, n_special=n_special, first_pass_merges=merged[0] if merged else 0)
    return res


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
    return {(tag, kind, n): _run(binaries[tag], kind, n, tag) for tag, kind in (("on", "split"), ("on", "merge"), ("on", "lattice"), ("off", "split"), ("off", "merge")) for n in (1, 2, 4)}


def _difference(a, b):
    """largest position difference in kernel lengths, largest relative mass difference (None, None if other cells exist), relative energy difference"""
    if a["key"] != b["key"]:
        return None, None, abs(a["E"] - b["E"]) / a["E"]
    dp = a["pos"] - b["pos"]
    dp -= a["box"] * np.round(dp / a["box"])
    return (np.sqrt((dp**2).sum(1)) / a["h"]).max(), np.abs(a["m"] / b["m"] - 1).max(), abs(a["E"] - b["E"]) / a["E"]


@pytest.mark.parametrize("kind", ["split", "merge", "lattice"])
def test_same_cells_on_any_rank_count(runs, kind):
    ref = runs[("on", kind, 1)]
    if kind != "merge":
        assert len(ref["key"]) == ref["n_ic"] + ref["n_special"], "not every heavy cell split exactly once"
    else:
        assert ref["first_pass_merges"] > 0 and len(ref["key"]) < ref["n_ic"], "no merge happened"
    for n in (2, 4):
        d, dm, de = _difference(ref, runs[("on", kind, n)])
        print(f"with the flag, {kind} box, {n} ranks vs 1: largest position difference {d} kernel lengths, mass {dm}, energy {de}")
        assert d is not None, f"{n} ranks: other cells than on 1 rank"
        assert d <= TOL_POS and dm <= TOL_M and de <= TOL_E


def test_merges_per_pass_close_to_the_rank_local_choice(runs):
    n_on, n_off = runs[("on", "merge", 1)]["first_pass_merges"], runs[("off", "merge", 1)]["first_pass_merges"]
    print(f"first pass, 1 rank: {n_on} merges with the flag, {n_off} without")
    assert all(runs[("on", "merge", n)]["first_pass_merges"] == n_on for n in (2, 4))
    assert abs(n_on / n_off - 1) <= 0.2


@pytest.mark.parametrize("kind", ["split", "merge"])
def test_control_without_the_flag_depends_on_ranks(runs, kind):
    d, dm, de = _difference(runs[("off", kind, 1)], runs[("off", kind, 4)])
    print(f"without the flag, {kind} box, 4 ranks vs 1: largest position difference {d} kernel lengths, mass {dm}, energy {de}")
    assert d is None or d > TOL_POS or dm > TOL_M, "the box does not put a neighbour or partner on another rank: it tests nothing"
