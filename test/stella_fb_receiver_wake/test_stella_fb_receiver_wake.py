"""Feedback receivers, the wake and the kick's u limiter (STL-183, GKS-98).

A feedback deposit wakes its receivers. Without RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED and WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP
an inactive receiver is put on the highest active bin with its pre-injection signal speed; at its next closing kick the forces of
the heated state act over that long step, the u limiter catches the cell and the energy it adds is created (booked as kick_floor).
With the flags the deposit raises the receiver's signal speed and the wake puts it on its own timestep.

Config_stella_default.sh (feedback before the hydro forces) plus the isolated-feedback test hooks, adiabatic, the
total-energy-exact kick (it books what the limiter adds), TEST_KICK_FLOOR_DUMP and TEST_WAKEUP_DUMP. Uniform n_H = 10, 1 Msun
cells, 16 pc periodic box at 100 K; one 60 Msun star blows its wind from birth; stellar radiation off; 250 kyr. The step cap is
50 kyr as in the cloud runs, so that the gas sits on longer bins than the star and receivers are inactive at a deposit.
(Versions with a 10 kyr cap, wind only (job 242377) and with a forced SN (job 242423), had the star and all gas on one bin: every
receiver active, no cell woken, no limiter firing with or without the flags, so they tested nothing.)

Metric and bars, fixed before the first run of this version (2026-11-25):
  r = max over the full syncs with feedback of (kick_floor booked so far) / (wind energy booked so far), from energy_step_trace.txt.
  - with the flags: r <= 1e-2 (Uli's bar for the r26 cloud, STL-183);
  - without them: r > 1e-2 (the problem must be present in this box, otherwise the test shows nothing);
  - both runs print receiver lines, the run without the flags also kick lines.
Rank independence without the flags (stella_dev as it is; Viper's question from GKS-98), 1, 2 and 4 ranks, one thread:
  the first deposit's receivers (IDs) and, at that step and the three steps after it, the woken cells (IDs with old and new bin) are
  identical, and kinetic + thermal energy after each of those steps agrees with the 1-rank run to 1e-10 relative.
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
    """first deposit step, its receivers, woken cells per step and energy per step"""
    recv = {}; woken = {}; energy = {}
    for m in re.finditer(r"^TEST_KICK_FLOOR_DUMP recv step (\d+) t \S+ ID (\d+) site (\d) ", text, re.M):
        recv.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    for m in re.finditer(r"^TEST_WAKEUP_DUMP woken step (\d+) ID (\d+) from (\d+) to (\d+) ", text, re.M):
        woken.setdefault(int(m.group(1)), set()).add((int(m.group(2)), int(m.group(3)), int(m.group(4))))
    for m in re.finditer(r"^TEST_WAKEUP_DUMP energy step (\d+) t \S+ E (\S+)$", text, re.M):
        energy[int(m.group(1))] = float(m.group(2))
    s0 = min(recv) if recv else None
    return s0, (recv[s0] if recv else set()), woken, energy


def _run(binary, fbc, name, nranks):
    d = join(TEST_DIR, "run_" + name); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    os.symlink(REPO, join(d, "gizmo"))
    fbc.make_box(join(d, "box_ics.hdf5"), fbc.ZSUN, T=100.0)
    base = open(join(REPO, "test", "stella_fb_conservation", "stella_fb_conservation.params")).read().splitlines()
    over = {"TestNumStars": 1, "TestStarMass": 60.0, "TimeMax": 2.5e-4, "MaxSizeTimestep": 5.0e-5,
            "TimeBetStatistics": 5.0e-5, "TimeBetSnapshot": 2.5e-4, "TestFUVZero": 1}
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
        for arm, extra in (("on", FLAGS), ("off", ())):
            build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
            binaries[arm] = join(TEST_DIR, f"GIZMO_{arm}")
            shutil.move(join(TEST_DIR, "GIZMO"), binaries[arm])
    finally:
        os.chdir(cwd)
    res = {}
    for arm, nranks in (("on", 2), ("off", 2), ("off", 1), ("off", 4)):
        text, out = _run(binaries[arm], fbc, f"{arm}_r{nranks}", nranks)
        r, n = _ratio(out)
        res[(arm, nranks)] = dict(r=r, n=n, nrecv=text.count("TEST_KICK_FLOOR_DUMP recv"), nkick=text.count("TEST_KICK_FLOOR_DUMP kick"), wake=_wake_record(text))
        print(f"flags {arm}, {nranks} ranks: max kick_floor / wind {r:.3e} over {n} syncs with feedback; dump lines: "
              f"{res[(arm, nranks)]['nrecv']} receivers, {res[(arm, nranks)]['nkick']} kicks; first deposit at step {res[(arm, nranks)]['wake'][0]}")
    return res


def test_dump_lines(runs):
    for key in (("on", 2), ("off", 2)):
        assert runs[key]["n"] > 0, f"{key}: no full sync with feedback"
        assert runs[key]["nrecv"] > 0, f"{key}: no receiver line"
    assert runs[("off", 2)]["nkick"] > 0, "without the flags no kick fired the u limiter above the dump threshold"


def test_problem_present_without_the_flags(runs):
    assert runs[("off", 2)]["r"] > BAR, f"without the flags the limiter adds only {runs[('off', 2)]['r']:.3e} of the feedback energy in this box"


def test_receivers_on_their_own_timestep(runs):
    assert runs[("on", 2)]["r"] <= BAR, f"with the flags the limiter adds {runs[('on', 2)]['r']:.3e} of the feedback energy"


def test_wake_independent_of_ranks(runs):
    s0, recv0, woken0, energy0 = runs[("off", 1)]["wake"]
    assert s0 is not None and len(recv0) > 0, "no receiver in the 1-rank run"
    assert sum(len(woken0.get(k, ())) for k in range(s0, s0 + STEPS_AFTER + 1)) > 0, "no cell woken in those steps: nothing to compare"
    bad = []
    for nranks in (2, 4):
        s, recv, woken, energy = runs[("off", nranks)]["wake"]
        print(f"{nranks} ranks: first deposit step {s} (1 rank: {s0}), receivers {len(recv)} (1 rank: {len(recv0)})")
        if s != s0 or recv != recv0:
            bad.append(f"{nranks} ranks: first deposit step or receivers differ"); continue
        for k in range(s0, s0 + STEPS_AFTER + 1):
            wa, wb = woken0.get(k, set()), woken.get(k, set())
            de = abs(energy[k] - energy0[k]) / abs(energy0[k]) if k in energy and k in energy0 else float("nan")
            print(f"  step {k}: woken {len(wb)} (1 rank: {len(wa)}), only in one run {len(wa ^ wb)}, |dE|/E {de:.3e}")
            if wa != wb:
                bad.append(f"{nranks} ranks, step {k}: {len(wa ^ wb)} woken cells differ")
            if not de <= BAR_RANKS:
                bad.append(f"{nranks} ranks, step {k}: |dE|/E {de:.3e}")
    assert not bad, "; ".join(bad)
