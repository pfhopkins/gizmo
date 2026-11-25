"""HYDRO_EXACT_PAIR_FLUX_EXCHANGE: a Sedov blast with a deep timestep hierarchy and wake-ups, 4 ranks, no gravity.
Gas mass, momentum, thermal + kinetic (+ magnetic) energy and sum B*V from the conserved variables after every sync
(conserved_totals.txt). Bars (fixed before running), all at TOL = 1e-12 relative:
- plain: |dM|/M, |dP|/sum m|v|, |dE|/E at every sync; the u floor never fires; no share to a non-gas particle; a second
  identical run gives the same conserved_totals.txt byte for byte.
- stop part-way and restart (the restart files hold the records waiting for the opening kick): the same bars over both runs;
  the stopping run returns 0 at the stop step (its last row is that step or the one before) and the restart continues with the next step.
- hook arms (test hook TEST_PAIRFLUX_ADD_MASS_AND_WAKE, changes booked and subtracted), the same bars:
  mass (inactive cells gain mass, then woken), merge (no hook: a shell of light cells far from the blast is below the IC-header merge threshold and merges into
  heavier neighbours at domain steps; a neighbour in a larger bin is woken),
  split (no hook: a shell of heavy cells far from the blast is above the IC-header split threshold; the split wakes the daughters).
- test_merge_centre / test_merge_centre_energy_exact: every cell is below the IC-header merge threshold, so the hot centre cells merge into
  inactive long-step cells; built with MERGE_INACTIVE_PARENT_COMPLETES_STEP_WITH_OWN_RATES and the own-timestep merge wake. Mass and momentum
  at TOL, merges in both halves of the parent's step, |dE|/E <= 1e-4; the bar of at most 5 limiter firings and the TOL energy bar are strict
  expected failures (measured 6 firings and 9.5e-5).
- MHD (Dedner cleaning; Powell terms off to isolate the exchange): mass, momentum and sum B*V, with the Dedner correction
  limiter and the face-area closure never firing; energy is reported (the kick's magnetic part is second order).
  The same with the IMF-like hook (mass, B, BPred, DtB scaled together on active and inactive woken cells).
- MHD with the Powell terms on (full solver): mass only; momentum and energy are reported next to the Powell work."""

import filecmp
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "pairflux_exact_exchange"
TEST_DIR = f"test/{TEST_NAME}"
TOL = 1e-12
STOP_STEP = 60
NRANKS = 4
MERGESPLIT = "TreeDomainUpdateFrequency 0\n"  # merge/split run at domain steps: at every sync, not only full ones (thresholds 0.49/2.01 x cell mass)
MERGE = MERGESPLIT + "InitCondFile pairflux_exact_exchange_merge_ics\n"  # a light shell far from the blast, below the IC-header merge threshold
SPLIT = MERGESPLIT + "InitCondFile pairflux_exact_exchange_split_ics\n"  # a heavy shell far from the blast, above the IC-header split threshold
MERGE_CENTRE = MERGESPLIT + "InitCondFile pairflux_exact_exchange_merge_centre_ics\n"  # every cell below the threshold: hot centre cells merge into inactive long-step cells (STL-292)
MHD = "BiniX 3e-7\nBiniY 1e-7\nBiniZ 0\nDivBcleaningParabolicSigma 0.2\nDivBcleaningHyperbolicSigma 1.0\n"


def _totals(path):
    a = np.loadtxt(path)
    w = a.shape[1]
    inj = a[:, 35:40] if w >= 40 else np.zeros((len(a), 5))
    injb = a[:, 40:43] if w >= 43 else np.zeros((len(a), 3))
    return dict(M=a[:, 2] - inj[:, 0], P=a[:, 3:6] - inj[:, 1:4], E=a[:, 6] + a[:, 7] + a[:, 8] - inj[:, 4], BV=a[:, 10:13] - injb,
                EB=a[:, 8], pscale=a[0, 9], nfloor=a[-1, 14], nstar=a[-1, 16], floor_paths=a[-1, 18:28], steps=a[:, 0], rows=len(a),
                limiters=a[-1, 43:45] if w >= 45 else np.zeros(2), powell=a[-1, 45:49] if w >= 49 else np.zeros(4),
                mem=(a[:, 28].max(), a[-1, 29], a[:, 52].max() if w >= 53 else 0.0))


def _check(ref, t, energy=True, bv=False, momentum=True):
    dM = np.abs(t["M"] - ref["M"][0]).max() / ref["M"][0]
    dP = np.abs(t["P"] - ref["P"][0]).max() / ref["pscale"]
    dE = np.abs(t["E"] - ref["E"][0]).max() / ref["E"][0]
    bscale = np.abs(ref["BV"][0]).sum() + 1e-300
    dB = np.abs(t["BV"] - ref["BV"][0]).max() / bscale
    print(f"rows {t['rows']}  max|dM|/M {dM:.2e}  max|dP|/sum m|v| {dP:.2e}  max|dE|/E {dE:.2e}  max|dBV|/sum|BV| {dB:.2e}  "
          f"floor firings {int(t['nfloor'])} by path (closing, wake, opening, mass change, kick) {t['floor_paths'][0::2].astype(int).tolist()}  "
          f"limiters (Dedner correction, closure) {t['limiters'].astype(int).tolist()}  Powell work (p, E) {t['powell'].tolist()}  shares to non-gas {int(t['nstar'])}  records peak MB {t['mem'][0]:.3f} spills {int(t['mem'][1])} helpers peak MB {t['mem'][2]:.3f}")
    assert t["nstar"] == 0, "a share went to a particle that is no longer gas"
    assert dM <= TOL
    if momentum:
        assert dP <= TOL
    if energy:
        assert t["nfloor"] == 0, "the u floor fired"
        assert dE <= TOL
    if bv:
        assert t["limiters"][0] == 0 and t["limiters"][1] == 0, "a per-cell B limiter fired"
        assert dB <= TOL


def _run(args, out, env=None, params=f"{TEST_NAME}.params"):
    return subprocess.run(mpi_launch_command(NRANKS, 1) + ["./GIZMO", params] + args, stdout=open(out, "w"), stderr=subprocess.STDOUT, env=env)


def _build(flags, name):
    build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    shutil.move(f"{TEST_DIR}/GIZMO", f"{TEST_DIR}/GIZMO_{name}")


def _arm(name, extra_params=""):
    shutil.copy(f"GIZMO_{name}", "GIZMO"); shutil.rmtree("output", ignore_errors=True)
    params = f"{name}.params"; lines = open(f"{TEST_NAME}.params").read().splitlines()
    for e in extra_params.splitlines():  # a key already in the params is replaced, others appended
        k = e.split()[0]; hit = [n for n, l in enumerate(lines) if l.split()[:1] == [k]]
        if hit: lines[hit[0]] = e
        else: lines.append(e)
    open(params, "w").write("\n".join(lines) + "\n")
    res = _run(["0"], f"{name}.out", params=params); assert res.returncode == 0, open(f"{name}.out").read()[-400:]
    return _totals("output/conserved_totals.txt"), open(f"{name}.out").read()


def test_pairflux_exact_exchange():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    hook = lambda mode: ("TEST_PAIRFLUX_ADD_MASS_AND_WAKE=40", f"TEST_PAIRFLUX_HOOK_MODE={mode}")
    mhd = ("MAGNETIC", "MHD_B_SET_IN_PARAMS")
    _build((), "hydro")
    _build(hook(1), "mass")
    _build(("MERGE_SPLIT_CONSERVE_ENERGY",), "merge")
    _build(("MERGE_SPLIT_CONSERVE_ENERGY",), "split")
    _build(mhd + ("TEST_MHD_NO_POWELL_SOURCE_TERMS",), "mhd")
    _build(mhd + ("TEST_MHD_NO_POWELL_SOURCE_TERMS",) + hook(4), "mhd_imf")
    _build(mhd, "mhd_powell")
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        shutil.copy("GIZMO_hydro", "GIZMO")
        for k in (1, 2):
            shutil.rmtree("output", ignore_errors=True)
            res = _run(["0"], f"plain{k}.out"); assert res.returncode == 0, open(f"plain{k}.out").read()[-400:]
            shutil.copy("output/conserved_totals.txt", f"plain{k}_totals.txt")
        plain = _totals("plain1_totals.txt")
        _check(plain, plain)
        assert filecmp.cmp("plain1_totals.txt", "plain2_totals.txt", shallow=False), "two identical runs differ"
        shutil.rmtree("output", ignore_errors=True)
        res = _run(["0"], "first.out", env=dict(os.environ, GIZMO_TEST_RESTART_AT_STEP=str(STOP_STEP)))
        assert res.returncode == 0, open("first.out").read()[-400:]
        first = _totals("output/conserved_totals.txt"); shutil.copy("output/conserved_totals.txt", "first_totals.txt")
        assert int(first["steps"][-1]) in (STOP_STEP - 1, STOP_STEP), f"the stopping run ended at step {first['steps'][-1]}, not at {STOP_STEP}"
        res = _run(["1"], "restart.out"); assert res.returncode == 0, open("restart.out").read()[-400:]
        second = _totals("output/conserved_totals.txt")
        assert int(second["steps"][0]) == int(first["steps"][-1]) + 1, f"the restart began at step {second['steps'][0]}, the stop was at {first['steps'][-1]}"
        _check(first, second)
        t, log = _arm("mass"); assert "mode 1 changed" in log; _check(t, t)
        t, log = _arm("merge", MERGE); assert re.search(r"[1-9][0-9]* particles merged", log), "no merge happened"; _check(t, t)
        t, log = _arm("split", SPLIT); assert re.search(r"[1-9][0-9]* particles split", log), "no split happened"; _check(t, t)
        t, _ = _arm("mhd", MHD); _check(t, t, energy=False, bv=True)
        t, log = _arm("mhd_imf", MHD); assert "mode 4 changed" in log; _check(t, t, energy=False, bv=True)
        t, _ = _arm("mhd_powell", MHD); _check(t, t, energy=False, momentum=False)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def merge_centre():
    """the centre-merge run: totals, log and the merge's parent-kick lines (half 1: the parent's midpoint was ahead of the merge, 2: behind)"""
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    _build(("MERGE_SPLIT_CONSERVE_ENERGY", "WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP", "WAKEUP_TRUNCATE_STEP_ON_DEMOTION", "MERGE_INACTIVE_PARENT_COMPLETES_STEP_WITH_OWN_RATES",
            "TEST_PAIRFLUX_MERGE_DUMP"), "merge_centre")
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        t, log = _arm("merge_centre", MERGE_CENTRE)  # asserts the run returns 0
    finally:
        os.chdir(cwd)
    halves = [int(h) for h in re.findall(r"TEST_PAIRFLUX_MERGE_PARENT_KICK .* half (\d) ", log)]
    return t, log, halves


def test_merge_centre(merge_centre):
    """bars that hold (fixed 2026-11-25 before the run of this version): merges into inactive parents in both halves of the parent's step,
    mass and momentum at TOL, |dE|/E <= 1e-4 (3.87 without the flag); the firing count has its own test below"""
    t, log, halves = merge_centre
    assert re.search(r"[1-9][0-9]* particles merged", log), "no merge happened"
    print(f"parent kicks: {halves.count(1)} with the midpoint ahead, {halves.count(2)} with the midpoint behind")
    assert halves.count(1) > 0, "no merge into an inactive parent in the first half of its step"
    assert halves.count(2) > 0, "no merge into an inactive parent in the second half of its step"
    dM = np.abs(t["M"] - t["M"][0]).max() / t["M"][0]; dP = np.abs(t["P"] - t["P"][0]).max() / t["pscale"]; dE = np.abs(t["E"] - t["E"][0]).max() / t["E"][0]
    print(f"max|dM|/M {dM:.2e}  max|dP|/sum m|v| {dP:.2e}  max|dE|/E {dE:.2e}  floor firings {int(t['nfloor'])}")
    assert t["nstar"] == 0 and dM <= TOL and dP <= TOL
    assert dE <= 1e-4


@pytest.mark.xfail(strict=True, reason="bar fixed 2026-11-25 from job 242310 (5 firings) before the merge covered the second half of the parent's step; "
                   "job 242377 with that version: 6 firings, |dE|/E 9.54e-5. Not met; kept unchanged, not relaxed")
def test_merge_centre_limiter_firings(merge_centre):
    assert merge_centre[0]["nfloor"] <= 5, "more firings of the u limiter than measured"


@pytest.mark.xfail(strict=True, reason="STL-292 remainder: the 50 % u limiter fires on closing impulses of merged cells (5 firings, |dE|/E 8.1e-5 in job 242310); "
                   "only this energy bar is expected to fail")
def test_merge_centre_energy_exact(merge_centre):
    t = merge_centre[0]
    assert t["nfloor"] == 0 and np.abs(t["E"] - t["E"][0]).max() / t["E"][0] <= TOL
