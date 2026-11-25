"""Feedback receivers, the wake and the kick's u limiter (STL-183), and the rank count (STL-324, GKS-98).

A feedback deposit wakes its receivers. Without RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED and WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP
an inactive receiver is put on the highest active bin with its pre-injection signal speed; at its next closing kick the forces of
the heated state act over that long step, the u limiter catches the cell and the energy it adds is created (booked as kick_floor).
With the flags the deposit raises the receiver's signal speed and the wake puts it on its own timestep.

Config_stella_default.sh (feedback before the hydro forces) plus the isolated-feedback test hooks, adiabatic, the
total-energy-exact kick (it books what the limiter adds), TEST_KICK_FLOOR_DUMP and TEST_WAKEUP_DUMP. Uniform 100 K gas of 1 Msun
cells in a periodic box, one 60 Msun star blowing its wind from birth, stellar radiation off, 250 kyr, steps up to 50 kyr (as
the cloud runs), so the gas sits on longer bins than the star and receivers are inactive at a deposit. Two boxes:
- dense: n_H = 1e3, 3.45 pc, gas softening 0.3 pc. The limiter fires without the flags (found in job 242652 after four boxes
  that did not show it: 16 pc wind with a 10 or 50 kyr cap, forced SN, n_H = 1e4 and 1e5). Its self-gravity makes runs on
  different rank counts differ already at the deposit step (2e-8 in energy, tree forces), so it is not used to compare ranks.
- thin: n_H = 10, 16 pc, gas softening 0.1 pc: no limiter firing, gravity negligible; used to compare rank counts.

Metric and bars, fixed before the first run (2026-11-25), dense box, 2 ranks:
  r = max over the full syncs with feedback of (kick_floor booked so far) / (wind energy booked so far), from energy_step_trace.txt.
  - with the flags: r <= 1e-2 (Uli's bar for the r26 cloud, STL-183); job 242652: 1.96e-4;
  - without them: r > 1e-2 (the problem must be present, otherwise the test shows nothing) and kick lines in the dump; job 242652:
    2.38e-2 (on 1 rank 3.08e-2, on 4 ranks 5.4e-3: the size depends on the rank count through the merges, see below).
Thin box: both flag states print receiver lines; the first deposit wakes the same receivers, with the flags on no longer bin than
without; with the flags no receiver is woken by a hydro request in place of its own (4 without, job 242652).
Rank count (thin box; 1, 2 and 4 ranks, one thread; stella_dev as it is, no wind flag): the first deposit's receivers (IDs) and,
at that step and the three steps after it, the woken cells (IDs with old and new bin) are identical, and kinetic + thermal
energy after each of those steps agrees with the 1-rank run to 1e-10 relative:
  - built with PREVENT_PARTICLE_MERGE_SPLIT: met (the wake does not depend on the rank count);
  - built with MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: met with merges and splits happening;
  - as stella_dev is: NOT met, strict expected failure. The merge partner, and the nearest neighbour that sets a split's daughter
    spacing, are searched among the cells of the calling rank only: 77 / 75 / 71 merges on 1 / 2 / 4 ranks in the step after the
    deposit, energy then different by 1.4e-2 and 3.2e-3; split counts can agree while the daughters sit elsewhere (pair-flux
    split box, job 242629: nearest neighbour at 0.44 h, seen at 0.48 h by the rank, daughters displaced by 0.018 h). Uli
    2026-11-25: accepted without the flag; the flag is the fix.
"""

import os
import re
import shutil
import subprocess
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "stella_fb_receiver_wake"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXTRA_FLAGS = ("BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "OUTPUT_ENERGY_STEP_TRACE",
               "RESOLVEDISM_ADIABATIC_NOCOOL", "HYDRO_KICK_TOTAL_ENERGY_EXACT", "TEST_KICK_FLOOR_DUMP=1e-14", "TEST_WAKEUP_DUMP")
FLAGS = ("RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED", "WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP")
BUILDS = (("on", FLAGS), ("off", ()), ("own", ("WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP",)), ("nomerge", ("PREVENT_PARTICLE_MERGE_SPLIT",)), ("across", ("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS",)))
BOXES = {"dense": (1.0e3, 3.45, 3.0e-4), "thin": (10.0, 16.0, 1.0e-4)}  # n_H [cm^-3], box [pc], gas softening [kpc]
RUNS = (("on", 2, "dense"), ("off", 2, "dense"), ("on", 2, "thin"), ("off", 2, "thin"), ("own", 2, "thin"), ("off", 1, "thin"), ("off", 4, "thin"),
        ("nomerge", 1, "thin"), ("nomerge", 2, "thin"), ("nomerge", 4, "thin"), ("across", 1, "thin"), ("across", 2, "thin"), ("across", 4, "thin"))
BAR = 1e-2
BAR_RANKS = 1e-10
STEPS_AFTER = 3
RUN_TIMEOUT = 5400  # s


def _ratio(out):
    r = 0.0; n = 0
    for l in open(join(out, "energy_step_trace.txt")):
        if l.startswith("SYNC"):
            v = dict(t.split("=", 1) for t in l.split()[1:] if "=" in t)
            fb = float(v["b_fb_wind"]) + float(v["b_fb_sn"])
            if fb > 0:
                r = max(r, abs(float(v["b_kick_floor"])) / fb); n += 1
    return r, n


def _wake_record(text):
    """first deposit step, its receivers, woken cells per step, energy per step, receivers woken by a hydro request"""
    recv = {}; woken = {}; energy = {}; flag = {}
    for m in re.finditer(r"^TEST_KICK_FLOOR_DUMP recv step (\d+) t \S+ ID (\d+) site (\d) ", text, re.M):
        recv.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    for m in re.finditer(r"^TEST_WAKEUP_DUMP woken step (\d+) ID (\d+) from (\d+) to (\d+) wakeup (-?\d+)", text, re.M):
        woken.setdefault(int(m.group(1)), set()).add((int(m.group(2)), int(m.group(3)), int(m.group(4))))
        flag.setdefault(int(m.group(1)), {})[int(m.group(2))] = int(m.group(5))
    for m in re.finditer(r"^TEST_WAKEUP_DUMP energy step (\d+) t \S+ E (\S+)$", text, re.M):
        energy[int(m.group(1))] = float(m.group(2))
    s0 = min(recv) if recv else None
    replaced = sum(1 for k in recv for i in recv[k] if flag.get(k, {}).get(i, -1) > 0)
    return s0, (recv[s0] if recv else set()), woken, energy, replaced


def _run(binary, fbc, name, nranks, box):
    d = join(TEST_DIR, "run_" + name); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    os.symlink(REPO, join(d, "gizmo"))
    n_H, box_pc, soft = BOXES[box]
    fbc.make_box(join(d, "box_ics.hdf5"), fbc.ZSUN, n_H=n_H, box_pc=box_pc, T=100.0)
    base = open(join(REPO, "test", "stella_fb_conservation", "stella_fb_conservation.params")).read().splitlines()
    over = {"TestNumStars": 1, "TestStarMass": 60.0, "TimeMax": 2.5e-4, "MaxSizeTimestep": 5.0e-5, "TimeBetStatistics": 5.0e-5,
            "TimeBetSnapshot": 2.5e-4, "TestFUVZero": 1, "BoxSize": box_pc * 1e-3, "SofteningGas": soft}
    lines = [l for l in base if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    with open(join(d, "run.log"), "w") as log:
        r = fbc.run_killable(mpi_launch_command(nranks) + [binary, "run.params"], RUN_TIMEOUT, stdout=log, stderr=subprocess.STDOUT, cwd=d)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0 and "Simulation ends." in text, f"{name}: " + text[-1500:]
    assert "RESOLVEDISM WIND:" in text and "RESOLVEDISM SN:" not in text, f"{name}: wind expected, no SN"
    return text, out


@pytest.fixture(scope="module")
def runs():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fbc", join(REPO, "test", "stella_fb_conservation", "test_stella_fb_conservation.py"))
    fbc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fbc)
    cfg = open(join(REPO, "Config_stella_default.sh")).read()
    open(join(TEST_DIR, "Config.sh"), "w").write(cfg + "\n# test hooks\n" + "\n".join(EXTRA_FLAGS) + "\n")
    binaries = {}
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        for arm, extra in BUILDS:
            build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
            binaries[arm] = join(TEST_DIR, f"GIZMO_{arm}")
            shutil.move(join(TEST_DIR, "GIZMO"), binaries[arm])
    finally:
        os.chdir(cwd)
    res = {}
    for arm, nranks, box in RUNS:
        text, out = _run(binaries[arm], fbc, f"{arm}_r{nranks}_{box}", nranks, box)
        r, n = _ratio(out)
        res[(arm, nranks, box)] = dict(r=r, n=n, nrecv=text.count("TEST_KICK_FLOOR_DUMP recv"), nkick=text.count("TEST_KICK_FLOOR_DUMP kick"), wake=_wake_record(text),
                                       merged=sum(int(x) for x in re.findall(r"(\d+) particles merged", text)))
        e = res[(arm, nranks, box)]
        print(f"{arm}, {nranks} ranks, {box} box: max kick_floor / wind {r:.3e} over {n} syncs with feedback; dump lines: {e['nrecv']} receivers, "
              f"{e['nkick']} kicks; first deposit at step {e['wake'][0]}; cells merged {e['merged']}")
    return res


def _compare_ranks(runs, arm, first=0, last=STEPS_AFTER):
    """differences to the 1-rank run at the steps s0 + first .. s0 + last after the first deposit (step s0)"""
    s0, recv0, woken0, energy0, _ = runs[(arm, 1, "thin")]["wake"]
    bad = []
    if s0 is None or len(recv0) == 0:
        return ["no receiver in the 1-rank run"]
    if first == 0 and sum(len(woken0.get(k, ())) for k in range(s0, s0 + STEPS_AFTER + 1)) == 0:
        return ["no cell woken in those steps: nothing to compare"]
    for nranks in (2, 4):
        s, recv, woken, energy, _ = runs[(arm, nranks, "thin")]["wake"]
        print(f"{arm}, {nranks} ranks: first deposit step {s} (1 rank: {s0}), receivers {len(recv)} (1 rank: {len(recv0)})")
        if s != s0 or recv != recv0:
            bad.append(f"{nranks} ranks: first deposit step or receivers differ"); continue
        for k in range(s0 + first, s0 + last + 1):
            wa, wb = woken0.get(k, set()), woken.get(k, set())
            de = abs(energy[k] - energy0[k]) / abs(energy0[k]) if k in energy and k in energy0 else float("nan")
            print(f"  step {k}: woken {len(wb)} (1 rank: {len(wa)}), only in one run {len(wa ^ wb)}, |dE|/E {de:.3e}")
            if wa != wb:
                bad.append(f"{nranks} ranks, step {k}: {len(wa ^ wb)} woken cells differ")
            if not de <= BAR_RANKS:
                bad.append(f"{nranks} ranks, step {k}: |dE|/E {de:.3e}")
    return bad


def test_problem_present_without_the_flags(runs):
    e = runs[("off", 2, "dense")]
    assert e["nkick"] > 0, "without the flags no kick fired the u limiter above the dump threshold"
    assert e["r"] > BAR, f"without the flags the limiter adds only {e['r']:.3e} of the wind energy in the dense box"


def test_receivers_on_their_own_timestep(runs):
    e = runs[("on", 2, "dense")]
    assert e["n"] > 0 and e["r"] <= BAR, f"with the flags the limiter adds {e['r']:.3e} of the wind energy"


def test_builds_run_and_dump(runs):
    for key in (("on", 2, "thin"), ("off", 2, "thin")):
        assert runs[key]["n"] > 0, f"{key}: no full sync with feedback"
        assert runs[key]["nrecv"] > 0, f"{key}: no receiver line"
        s0, recv, woken, _, _ = runs[key]["wake"]
        assert len(recv) > 0 and len(woken.get(s0, ())) > 0, f"{key}: no receiver woken at the first deposit"
    s0, _, w_on, _, _ = runs[("on", 2, "thin")]["wake"]; _, _, w_off, _, _ = runs[("off", 2, "thin")]["wake"]
    to_on = {i: b for i, _, b in w_on[s0]}; to_off = {i: b for i, _, b in w_off[s0]}
    assert set(to_on) == set(to_off), "the first deposit wakes different cells with and without the flags"
    assert all(to_on[i] <= to_off[i] for i in to_on), "with the flags a receiver was put on a longer bin than without"
    print(f"first deposit: {len(to_on)} receivers woken; bins with the flags {sorted(set(to_on.values()))}, without {sorted(set(to_off.values()))}")


def test_hydro_wake_does_not_replace_the_receiver_wake(runs):
    """with WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP alone (no raised signal speed, which would itself keep hydro requests away) a
    receiver woken in the step of its deposit keeps its own request (-1): no hydro request (> 0) in its place, and it is still woken"""
    n_own, n_on, n_off = (runs[(a, 2, "thin")]["wake"][4] for a in ("own", "on", "off"))
    print(f"receivers woken with a hydro request in place of their own: {n_off} without the flags, {n_own} with the own-timestep wake alone, {n_on} with both flags")
    assert n_off > 0, "the case does not occur in this box without the flags: nothing tested"
    assert n_own == 0 and n_on == 0
    s0, recv_off, w_off, _, _ = runs[("off", 2, "thin")]["wake"]; _, recv_own, w_own, _, _ = runs[("own", 2, "thin")]["wake"]
    assert recv_own == recv_off and {i for i, _, _ in w_own.get(s0, ())} == {i for i, _, _ in w_off.get(s0, ())}, "other cells woken at the first deposit"


def test_runs_agree_through_the_deposit_step(runs):
    """every build: same first deposit step, receivers, woken cells and energy on 1, 2 and 4 ranks at the deposit step"""
    for arm in ("off", "nomerge", "across"):
        bad = _compare_ranks(runs, arm, first=0, last=0)
        assert not bad, arm + ": " + "; ".join(bad)


def test_independent_of_ranks_with_merge_split_search_across_ranks(runs):
    """MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: with merges and splits happening, the same bars as without merge/split"""
    assert all(runs[("across", n, "thin")]["merged"] > 0 for n in (1, 2, 4)), "no merge in this box"
    bad = _compare_ranks(runs, "across")
    assert not bad, "; ".join(bad)


def test_wake_independent_of_ranks_without_merge_split(runs):
    assert all(runs[("nomerge", n, "thin")]["merged"] == 0 for n in (1, 2, 4))
    bad = _compare_ranks(runs, "nomerge")
    assert not bad, "; ".join(bad)


@pytest.mark.xfail(strict=True, reason="STL-324, accepted without MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS (Uli 2026-11-25): merge partner and split "
                   "neighbour are searched on the calling rank only; 77 / 75 / 71 merges on 1 / 2 / 4 ranks in the step after the deposit, "
                   "energy then different by 1.4e-2 (2 ranks) and 3.2e-3 (4 ranks) against the bar of 1e-10; daughters of a split near a rank "
                   "boundary are placed differently (0.018 h in the pair-flux split box)")
def test_wake_independent_of_ranks(runs):
    """only the steps AFTER the deposit step are expected to differ; the deposit step itself is asserted in test_runs_agree_through_the_deposit_step"""
    bad = _compare_ranks(runs, "off", first=1)
    assert not bad, "; ".join(bad)
