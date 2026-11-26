"""MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: a merge/split pass gives the same result on 1, 2 and 4 ranks (port of stella_dev's test).

The Sedov box of test/pairflux_exact_exchange (its build config, parameter file and initial conditions) with the flag and
MERGE_SPLIT_CONSERVE_ENERGY, merge/split at every sync:
- split box: a shell of heavy cells above the IC-header split threshold, which the rank boundaries of the 2- and 4-rank runs cut;
- centre-merge box: every cell is below the IC-header merge threshold.
The initial state is identical for every rank count, so the first pass sees the same cells.

Bars, fixed before the first run (2026-11-25):
- split box, first step with splits: the same parents split (IDs) and each parent's two daughters sit at the same offsets to
  1e-12 of the parent's kernel length;
- centre-merge box, first step with merges: the same pairs (absorbed cell -> survivor, by ID, child number and generation);
- both: gas mass equal to 1e-12 and thermal + kinetic energy equal to 1e-10 (relative) at the first sync row after that pass;
- in the 4-rank run at least one split parent has its nearest neighbour on another rank and at least one merge is between cells
  on different ranks (counted from the dump; otherwise the search and the merge across ranks are not tested);
- control arm: the same runs built WITHOUT the flag differ from 1 rank (if they do not, the boxes test nothing).
Not covered: star particles (the boxes are gas only), so the copy slot is only used with no particle behind the gas block.
"""

import os
import re
import shutil
import subprocess
import sys
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "merge_split_rank_independence"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
SRC = join(REPO, "test", "pairflux_exact_exchange")
FLAG = "MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS"
COMMON = ("MERGE_SPLIT_CONSERVE_ENERGY", "TEST_WAKEUP_DUMP")
BOXES = {"split": "pairflux_exact_exchange_split_ics", "merge": "pairflux_exact_exchange_merge_centre_ics"}
TOL_POS, TOL_M, TOL_E = 1e-12, 1e-12, 1e-10


def _run(binary, box, nranks, tag):
    d = join(TEST_DIR, f"run_{tag}_{box}_r{nranks}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(join(d, "output"))
    shutil.copy(join(SRC, BOXES[box] + ".hdf5"), d)
    lines = [l for l in open(join(SRC, "pairflux_exact_exchange.params")).read().splitlines() if l.split()[:1] not in (["InitCondFile"], ["TreeDomainUpdateFrequency"], ["TimeMax"])]
    lines += [f"InitCondFile {BOXES[box]}", "TreeDomainUpdateFrequency 0", "TimeMax 0.002"]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(nranks, 1) + [binary, "run.params", "0"], stdout=log, stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0, f"{tag} {box} {nranks} ranks: " + text[-800:]
    splits = {}; merges = {}
    for m in re.finditer(r"^TEST_SPLIT_DUMP step (\d+) task \d+ ID (\d+) h (\S+) r_near (\S+) d_r (\S+) parent (\S+) (\S+) (\S+) offset (\S+) (\S+) (\S+)$", text, re.M):
        splits.setdefault(int(m.group(1)), {})[int(m.group(2))] = (float(m.group(3)), float(m.group(4)), np.array([float(m.group(k)) for k in (9, 10, 11)]))
    crossing = {"merge": {}, "split": {}}  # per step: merges whose two cells, splits whose nearest neighbour, are on different ranks
    for m in re.finditer(r"^TEST_MERGE_PAIR step (\d+) absorbed (\d+) (\d+) (-?\d+) into (\d+) (\d+) (-?\d+) ranks (\d+) (\d+)$", text, re.M):
        merges.setdefault(int(m.group(1)), set()).add(tuple(int(m.group(k)) for k in range(2, 8)))
        crossing["merge"][int(m.group(1))] = crossing["merge"].get(int(m.group(1)), 0) + (m.group(8) != m.group(9))
    for m in re.finditer(r"^TEST_SPLIT_NEIGHBOR step (\d+) ID \d+ task (\d+) neighbour_task (\d+)$", text, re.M):
        crossing["split"][int(m.group(1))] = crossing["split"].get(int(m.group(1)), 0) + (m.group(2) != m.group(3))
    tot = np.loadtxt(join(d, "output", "conserved_totals.txt"))
    return dict(splits=splits, merges=merges, crossing=crossing, steps=tot[:, 0], M=tot[:, 2], E=tot[:, 6] + tot[:, 7], nmerged=sum(int(x) for x in re.findall(r"(\d+) particles merged", text)),
                nsplit=sum(int(x) for x in re.findall(r"(\d+) particles split", text)))


@pytest.fixture(scope="module")
def runs():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    subprocess.run([sys.executable, "make_pairflux_exact_exchange_ics.py"], cwd=SRC, check=True)
    shutil.copy(join(SRC, "Config.sh"), join(TEST_DIR, "Config.sh"))
    binaries = {}
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        for tag, extra in (("on", COMMON + (FLAG,)), ("off", COMMON)):
            build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
            binaries[tag] = join(TEST_DIR, f"GIZMO_{tag}"); shutil.move(join(TEST_DIR, "GIZMO"), binaries[tag])
    finally:
        os.chdir(cwd)
    res = {}
    for tag in ("on", "off"):
        for box in BOXES:
            for n in (1, 2, 4):
                res[(tag, box, n)] = _run(binaries[tag], box, n, tag)
                r = res[(tag, box, n)]
                print(f"flag {tag}, {box} box, {n} ranks: {r['nsplit']} splits, {r['nmerged']} merges, first split step {min(r['splits']) if r['splits'] else None}, "
                      f"first merge-pair step {min(r['merges']) if r['merges'] else None}")
    return res


def _row_after(r, step):
    k = int(np.searchsorted(r["steps"], step)); return min(k, len(r["steps"]) - 1)


def _split_difference(a, b):
    """largest daughter-offset difference / kernel length at the first split step; None if the parents differ"""
    s = min(a["splits"]); pa, pb = a["splits"][s], b["splits"].get(s, {})
    if set(pa) != set(pb):
        return None, s
    return max(np.linalg.norm(pa[i][2] - pb[i][2]) / pa[i][0] for i in pa), s


def test_split_daughters_independent_of_ranks(runs):
    ref = runs[("on", "split", 1)]; assert ref["splits"], "no split in the 1-rank run"
    for n in (2, 4):
        d, s = _split_difference(ref, runs[("on", "split", n)])
        k = _row_after(ref, s); r = runs[("on", "split", n)]
        dM = abs(r["M"][k] - ref["M"][k]) / ref["M"][k]; dE = abs(r["E"][k] - ref["E"][k]) / ref["E"][k]
        print(f"{n} ranks: step {s}, {len(ref['splits'][s])} parents, largest daughter-offset difference / h {d}, |dM|/M {dM:.2e}, |dE|/E {dE:.2e}")
        assert d is not None, f"{n} ranks: other parents split"
        assert d <= TOL_POS and dM <= TOL_M and dE <= TOL_E
    nx = runs[("on", "split", 4)]["crossing"]["split"].get(s, 0)
    print(f"4 ranks, step {s}: {nx} split parents have their nearest neighbour on another rank")
    assert nx > 0, "no split parent with its nearest neighbour on another rank: the search across ranks is not tested"


def test_merge_pairs_independent_of_ranks(runs):
    ref = runs[("on", "merge", 1)]; assert ref["merges"], "no merge in the 1-rank run"
    s = min(ref["merges"])
    for n in (2, 4):
        r = runs[("on", "merge", n)]; pairs = r["merges"].get(s, set())
        k = _row_after(ref, s)
        dM = abs(r["M"][k] - ref["M"][k]) / ref["M"][k]; dE = abs(r["E"][k] - ref["E"][k]) / ref["E"][k]
        print(f"{n} ranks: step {s}, pairs {len(pairs)} (1 rank: {len(ref['merges'][s])}), in one run only {len(pairs ^ ref['merges'][s])}, |dM|/M {dM:.2e}, |dE|/E {dE:.2e}")
        assert pairs == ref["merges"][s]
        assert dM <= TOL_M and dE <= TOL_E
    nx = runs[("on", "merge", 4)]["crossing"]["merge"].get(s, 0)
    print(f"4 ranks, step {s}: {nx} merges between cells on different ranks")
    assert nx > 0, "no merge between cells on different ranks: the merge across ranks is not tested"


def test_control_without_the_flag_depends_on_ranks(runs):
    """the boxes do put neighbours and partners across rank boundaries: without the flag the rank count changes the result"""
    ref = runs[("off", "split", 1)]; d4, s = _split_difference(ref, runs[("off", "split", 4)])
    print(f"without the flag, split box, 4 ranks vs 1: largest daughter-offset difference / h {d4}")
    assert d4 is None or d4 > TOL_POS, "split box: the nearest neighbour never lies on another rank"
    refm = runs[("off", "merge", 1)]; r4 = runs[("off", "merge", 4)]
    k = _row_after(refm, min(runs[("on", "merge", 1)]["merges"]))  # the first sync row after the first merge pass
    n1, n4 = refm["nmerged"], r4["nmerged"]; dE = abs(r4["E"][k] - refm["E"][k]) / refm["E"][k]
    print(f"without the flag, centre-merge box: merges {n1} on 1 rank, {n4} on 4; |dE|/E at the first row {dE:.2e}")
    assert n1 != n4 or dE > TOL_E, "centre-merge box: the result does not depend on the rank count even without the flag"
