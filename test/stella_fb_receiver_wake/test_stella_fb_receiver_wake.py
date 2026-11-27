"""Feedback receivers, the wake and the kick's u limiter, and the rank count (port of stella_dev's test/stella_fb_receiver_wake).

A feedback deposit wakes its receivers. Without RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED and WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP
an inactive receiver is put on the highest active bin with its pre-injection signal speed; at its next closing kick the forces of
the heated state act over that long step, the u limiter catches the cell and the energy it adds is created. With the flags the
deposit raises the receiver's signal speed and the wake puts it on its own timestep.

Set-up here (stella_dev's uses hooks this branch does not have: a star spawned by parameter, an adiabatic switch, the energy step
trace's kick_floor booking): the build of test/single_stars_supernovae (no gravity, no cooling) with the deposit before the hydro
forces, the total-energy-exact kick, TEST_KICK_FLOOR_DUMP and TEST_WAKEUP_DUMP. Uniform 100 K gas of 1 Msun cells (11^3) in a
periodic box, one 60 Msun star 1 Myr before the end of its life with its wind account settled to now (it owes no wind at the
start and does not die in the run), 0.25 time units (245 kyr), gas steps up to 0.05 (49 kyr), stellar steps up to 10 kyr, so
receivers are inactive at a deposit. Two boxes as on stella_dev: dense (n_H = 1e3) and thin (n_H = 10).

Metric and bars (stella_dev's bars; fixed before the first run on this branch, 2026-11-25), dense box, 2 ranks:
  r = max over the full syncs with feedback of |energy the kick's u limiter and floor added so far| / (energy the deposits added
  to their receivers so far), both from the dump lines (stella_dev reads the two from its energy step trace).
  - with the flags: r <= 1e-2;
  - without them: r > 1e-2 and kick lines in the dump (the problem must be present, otherwise the test shows nothing).
Thin box: both flag states print receiver lines; the first deposit wakes the same receivers, with the flags on no longer bin than
without; with WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP no receiver is woken by a hydro request in place of its own.
Rank count (thin box; 1, 2 and 4 ranks, one thread): the first deposit's receivers (IDs) and, at that step and the three steps
after it, the woken cells (IDs with old and new bin) are identical, and kinetic + thermal energy after each of those steps agrees
with the 1-rank run to 1e-10 relative:
  - built with PREVENT_PARTICLE_MERGE_SPLIT;
  - with the default merging and splitting: the deposit step itself is asserted; the steps after it are not compared, since
    merge partners and split neighbours are searched on the calling rank, by design, so different rank counts are different,
    equally valid realisations.
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

from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "stella_fb_receiver_wake"
TEST_DIR = dirname(abspath(__file__))
SN_DIR = abspath(join(TEST_DIR, "..", "single_stars_supernovae"))
FLAGS = ("RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED", "WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP")
BUILDS = (("on", FLAGS), ("off", ()), ("own", ("WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP",)), ("nomerge", ("PREVENT_PARTICLE_MERGE_SPLIT",)))
BOXES = {"dense": 1.0e3, "thin": 10.0}  # n_H [cm^-3]
RUNS = (("on", 2, "dense"), ("off", 2, "dense"), ("on", 2, "thin"), ("off", 2, "thin"), ("own", 2, "thin"), ("off", 1, "thin"), ("off", 4, "thin"),
        ("nomerge", 1, "thin"), ("nomerge", 2, "thin"), ("nomerge", 4, "thin"))
STAR_MASS, BEFORE_DEATH_YR, Z = 60.0, 1.0e6, 0.014
TIME_MAX, MAX_STEP = 0.25, 0.05
BAR = 1e-2
BAR_RANKS = 1e-10
STEPS_AFTER = 3


def _ratio(text):
    """max over the full syncs with feedback of |limiter energy so far| / deposited energy so far; number of such syncs"""
    t_of = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^TEST_WAKEUP_DUMP energy step (\d+) t (\S+) E \S+$", text, re.M)}
    dep = {}; lim = {}
    for m in re.finditer(r"^TEST_KICK_FLOOR_DUMP recv step (\d+) .* dE (\S+)$", text, re.M):
        dep[int(m.group(1))] = dep.get(int(m.group(1)), 0.0) + float(m.group(2))
    for m in re.finditer(r"^TEST_KICK_FLOOR_DUMP kick step (\d+) .* dE (\S+)$", text, re.M):
        lim[int(m.group(1))] = lim.get(int(m.group(1)), 0.0) + float(m.group(2))
    r = 0.0; n = 0; cd = 0.0; cl = 0.0
    for s in sorted(t_of):
        cd += dep.get(s, 0.0); cl += lim.get(s, 0.0)
        if cd > 0 and abs(t_of[s] / MAX_STEP - round(t_of[s] / MAX_STEP)) < 1e-9:
            r = max(r, abs(cl) / cd); n += 1
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


def _run(binary, tables, name, nranks, box):
    sys.path.insert(0, SN_DIR)
    from make_single_stars_supernovae_ics import make_single_stars_supernovae_ics
    d = join(TEST_DIR, "run_" + name); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    T = tables["StellarTablesFile"]
    with h5py.File(T, "r") as F:  # the wind account is settled to the star's age now: last wind age = main-sequence life - time to death
        Zt, Mt = F["Z"][()], F["M_init"][()]
        ms_life = np.interp(np.log10(STAR_MASS), np.log10(Mt), F["lifetime_yr"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(np.log10(Zt) - np.log10(Z))))])
    assert ms_life > 2 * BEFORE_DEATH_YR
    L, _ = make_single_stars_supernovae_ics(join(out, "snapshot_000.hdf5"), T, n=11, n_H=BOXES[box], stars=[(STAR_MASS, (0.5, 0.5, 0.5))], z=Z,
                                            before_death_yr=BEFORE_DEATH_YR, winds_owed_dex=-np.log10(1.0 - BEFORE_DEATH_YR / ms_life))
    over = {"TimeMax": TIME_MAX, "MaxSizeTimestep": MAX_STEP, "TimeBetStatistics": MAX_STEP, "TimeBetSnapshot": TIME_MAX, "BoxSize": L, "StellarTablesFile": T}
    lines = open(join(SN_DIR, "single_stars_supernovae.params")).read().splitlines()
    lines = [l for l in lines if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(nranks, 1) + [binary, "run.params", "2", "0"], stdout=log, stderr=subprocess.STDOUT, cwd=d, env={**os.environ, "OMP_NUM_THREADS": "1"})
    text = open(join(d, "run.log"), errors="replace").read()
    assert r.returncode == 0, f"{name}: " + text[-1500:]
    ev = join(out, "single_star_events.txt")
    assert not (os.path.exists(ev) and os.path.getsize(ev) > 0 and len(np.atleast_2d(np.loadtxt(ev))) > 0), f"{name}: a star died; wind only expected"
    return text


@pytest.fixture(scope="module")
def runs():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    tables = get_stellar_evolution_tables()
    binaries = {}
    for arm, extra in BUILDS:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
        binaries[arm] = join(TEST_DIR, f"GIZMO_{arm}")
        shutil.move(join(TEST_DIR, "GIZMO"), binaries[arm])
    res = {}
    for arm, nranks, box in RUNS:
        text = _run(binaries[arm], tables, f"{arm}_r{nranks}_{box}", nranks, box)
        r, n = _ratio(text)
        res[(arm, nranks, box)] = dict(r=r, n=n, nrecv=text.count("TEST_KICK_FLOOR_DUMP recv"), nkick=text.count("TEST_KICK_FLOOR_DUMP kick"), wake=_wake_record(text),
                                       merged=sum(int(x) for x in re.findall(r"(\d+) particles merged", text)), split=sum(int(x) for x in re.findall(r"(\d+) particles split", text)))
        e = res[(arm, nranks, box)]
        print(f"{arm}, {nranks} ranks, {box} box: max limiter / deposited energy {r:.3e} over {n} full syncs with feedback; dump lines: {e['nrecv']} receivers, "
              f"{e['nkick']} kicks; first deposit at step {e['wake'][0]}; cells merged {e['merged']}, split {e['split']}; receivers woken by a hydro request {e['wake'][4]}")
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
    assert e["r"] > BAR, f"without the flags the limiter adds only {e['r']:.3e} of the deposited energy in the dense box"


def test_receivers_on_their_own_timestep(runs):
    e = runs[("on", 2, "dense")]
    assert e["n"] > 0 and e["r"] <= BAR, f"with the flags the limiter adds {e['r']:.3e} of the deposited energy"


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
    for arm in ("off", "nomerge"):
        bad = _compare_ranks(runs, arm, first=0, last=0)
        assert not bad, arm + ": " + "; ".join(bad)



def test_wake_independent_of_ranks_without_merge_split(runs):
    assert all(runs[("nomerge", n, "thin")]["merged"] + runs[("nomerge", n, "thin")]["split"] == 0 for n in (1, 2, 4))
    bad = _compare_ranks(runs, "nomerge")
    assert not bad, "; ".join(bad)

