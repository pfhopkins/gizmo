"""MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS (STL-324): a merge/split pass gives the same result on 1, 2 and 4 ranks.

The Sedov box of test/pairflux_exact_exchange (its build config, parameter file and initial conditions) with the flag and
MERGE_SPLIT_CONSERVE_ENERGY, merge/split at every sync:
- split box: a shell of heavy cells above the IC-header split threshold, which the rank boundaries of the 2- and 4-rank runs cut;
- centre-merge box: every cell is below the IC-header merge threshold;
- lattice box: an exact 16^3 lattice at rest (positions exact in binary), a tenth of the cells above the split threshold. There
  the neighbour-distribution matrix of a split cell has equal eigenvalues, so the split axis is not unique; without a tie-break it
  follows round-off in the neighbour sums, whose order depends on the rank count (bar added 2026-11-26 before its first run).
The initial state is identical for every rank count, so the first pass sees the same cells.

Bars, fixed before the first run (2026-11-25):
- split and lattice boxes, first step with splits: the same parents split (IDs) and each parent's two daughters sit at the same offsets to
  1e-12 of the parent's kernel length;
- centre-merge box, first step with merges: the same pairs (absorbed cell -> survivor, by ID, child number and generation);
- both: gas mass equal to 1e-12 and thermal + kinetic energy equal to 1e-10 (relative) at the first sync row after that pass;
- in the 4-rank run at least one split parent has its nearest neighbour on another rank and at least one merge is between cells
  on different ranks (counted from the dump; otherwise the search and the merge across ranks are not tested);
- control arm: the same runs built WITHOUT the flag differ from 1 rank (if they do not, the boxes test nothing);
- the first merge pass of the centre-merge box merges within 20 % of the number the rank-local choice merges (several matching rounds
  per pass; bar fixed 2026-11-25 before the first run with rounds).
Not covered: star particles (the boxes are gas only), so the copy slot is only used with no particle behind the gas block.
"""

import os
import re
import shutil
import subprocess
import sys
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "merge_split_rank_independence"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
SRC = join(REPO, "test", "pairflux_exact_exchange")
FLAG = "MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS"
COMMON = ("MERGE_SPLIT_CONSERVE_ENERGY", "TEST_WAKEUP_DUMP")
BOXES = {"split": "pairflux_exact_exchange_split_ics", "merge": "pairflux_exact_exchange_merge_centre_ics", "lattice": "merge_split_lattice_ics"}
TOL_POS, TOL_M, TOL_E = 1e-12, 1e-12, 1e-10


def _write_lattice(path, n=16):
    """exact lattice in a unit box (rho = 1, u = 1e-2, at rest); a fixed tenth of the cells 2.2 x heavier at equal pressure, split threshold 2.01 x"""
    x = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    m0 = 1.0 / n**3; m = np.full(n**3, m0); u = np.full(n**3, 1e-2)
    heavy = np.random.default_rng(5).permutation(n**3)[: n**3 // 10]; m[heavy] *= 2.2; u[heavy] /= 2.2
    with h5py.File(path, "w") as f:
        h = f.create_group("Header"); N = n**3
        h.attrs["Maximum_Mass_For_Cell_Split"] = 2.01 * m0
        for k, val in dict(NumPart_ThisFile=np.array([N, 0, 0, 0, 0, 0], np.int32), NumPart_Total=np.array([N, 0, 0, 0, 0, 0], np.uint32), NumPart_Total_HighWord=np.zeros(6, np.uint32),
                           MassTable=np.zeros(6), Time=0.0, Redshift=0.0, BoxSize=1.0, NumFilesPerSnapshot=1, Omega0=0., OmegaLambda=0., HubbleParam=1., Flag_Sfr=0, Flag_Cooling=0,
                           Flag_StellarAge=0, Flag_Metals=0, Flag_Feedback=0, Flag_DoublePrecision=0).items():
            h.attrs[k] = val
        g = f.create_group("PartType0")
        for k, a in dict(Coordinates=x, Velocities=np.zeros((N, 3)), Masses=m, InternalEnergy=u).items():
            g[k] = a.astype(np.float32)
        g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)


def _run(binary, box, nranks, tag):
    d = join(TEST_DIR, f"run_{tag}_{box}_r{nranks}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(join(d, "output"))
    if box == "lattice":
        _write_lattice(join(d, BOXES[box] + ".hdf5"))
    else:
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
    passes = [int(x) for x in re.findall(r"(\d+) particles merged", text) if int(x) > 0]  # merges per pass, passes with merges only
    cost = [(int(a), int(b), int(c), float(d)) for a, b, c, d in re.findall(r"^TEST_MERGE_SPLIT_PASS step \d+ rounds (\d+) queries (\d+) merges (\d+) seconds (\S+)$", text, re.M)]
    return dict(splits=splits, merges=merges, crossing=crossing, passes=passes, cost=cost, steps=tot[:, 0], M=tot[:, 2], E=tot[:, 6] + tot[:, 7], nmerged=sum(int(x) for x in re.findall(r"(\d+) particles merged", text)),
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


@pytest.mark.parametrize("box", ["split", "lattice"])
def test_split_daughters_independent_of_ranks(runs, box):
    ref = runs[("on", box, 1)]; assert ref["splits"], "no split in the 1-rank run"
    for n in (2, 4):
        d, s = _split_difference(ref, runs[("on", box, n)])
        k = _row_after(ref, s); r = runs[("on", box, n)]
        dM = abs(r["M"][k] - ref["M"][k]) / ref["M"][k]; dE = abs(r["E"][k] - ref["E"][k]) / ref["E"][k]
        print(f"{box} box, {n} ranks: step {s}, {len(ref['splits'][s])} parents, largest daughter-offset difference / h {d}, |dM|/M {dM:.2e}, |dE|/E {dE:.2e}")
        assert d is not None, f"{n} ranks: other parents split"
        assert d <= TOL_POS and dM <= TOL_M and dE <= TOL_E
    nx = runs[("on", box, 4)]["crossing"]["split"].get(s, 0)
    print(f"{box} box, 4 ranks, step {s}: {nx} split parents have their nearest neighbour on another rank")
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


def test_merges_per_pass_close_to_the_rank_local_choice(runs):
    """the matching is repeated within a pass until nothing more merges: the first pass merges within 20 % of what the rank-local greedy
    choice merges (1 rank, same initial state); the rounds and the time of the passes are printed"""
    n_on, n_off = runs[("on", "merge", 1)]["passes"][0], runs[("off", "merge", 1)]["passes"][0]
    c = runs[("on", "merge", 4)]["cost"]
    print(f"first pass, 1 rank: {n_on} merges with the flag, {n_off} without; 4 ranks with the flag: {len(c)} passes, rounds per pass up to {max(x[0] for x in c)}, "
          f"queries per pass up to {max(x[1] for x in c)}, seconds per pass mean {sum(x[3] for x in c) / len(c):.2e} max {max(x[3] for x in c):.2e}")
    assert all(runs[("on", "merge", n)]["passes"][0] == n_on for n in (2, 4)), "the first pass merges a different number on 2 or 4 ranks"
    assert abs(n_on / n_off - 1) <= 0.2


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
