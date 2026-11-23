"""HYDRO_EXACT_PAIR_FLUX_EXCHANGE (port of stella_dev test/pairflux_exact_exchange): a Sedov blast with a deep timestep
hierarchy and wake-ups, 4 ranks, no gravity, MFM without MHD. Gas mass, momentum, thermal + kinetic energy from the conserved
variables after every sync (conserved_totals.txt). Bars (stella_dev's, fixed before running), all at TOL = 1e-12 relative:
- plain: |dM|/M, |dP|/sum m|v|, |dE|/E at every sync; the u floor never fires; no share to a non-gas particle; a second
  identical run gives the same conserved_totals.txt byte for byte.
- stop part-way and restart (the restart files hold the records waiting for the opening kick): the same bars over both runs;
  the stopping run ends at the stop step (or the one before) and the restart continues with the next step.
- hook arms (test hook TEST_PAIRFLUX_ADD_MASS_AND_WAKE, changes booked and subtracted), the same bars:
  mass (inactive cells gain mass, then woken), merge (cells shrink and merge into neighbours), split (cells grow and split).
The MHD arms of stella_dev are not here: the B*V and Dedner records are not ported to gku (the flag is a build error with MAGNETIC)."""

import filecmp
import os
import re
import shutil
import subprocess
import sys

import numpy as np

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "pairflux_exact_exchange"
TEST_DIR = f"test/{TEST_NAME}"
TOL = 1e-12
STOP_STEP = 60
NRANKS = 4
M0 = 1.0 / 16**3  # IC cell mass


def _totals(path):
    a = np.loadtxt(path)
    w = a.shape[1]
    inj = a[:, 35:40] if w >= 40 else np.zeros((len(a), 5))
    return dict(M=a[:, 2] - inj[:, 0], P=a[:, 3:6] - inj[:, 1:4], E=a[:, 6] + a[:, 7] + a[:, 8] - inj[:, 4],
                pscale=a[0, 9], nfloor=a[-1, 14], nstar=a[-1, 16], floor_paths=a[-1, 18:28], steps=a[:, 0], rows=len(a),
                mem=(a[:, 28].max(), a[:, 52].max() if w >= 53 else 0.0))


def _check(ref, t):
    dM = np.abs(t["M"] - ref["M"][0]).max() / ref["M"][0]
    dP = np.abs(t["P"] - ref["P"][0]).max() / ref["pscale"]
    dE = np.abs(t["E"] - ref["E"][0]).max() / ref["E"][0]
    print(f"rows {t['rows']}  max|dM|/M {dM:.2e}  max|dP|/sum m|v| {dP:.2e}  max|dE|/E {dE:.2e}  "
          f"floor firings {int(t['nfloor'])} by path (closing, wake, opening, mass change, kick) {t['floor_paths'][0::2].astype(int).tolist()}  "
          f"shares to non-gas {int(t['nstar'])}  records peak MB {t['mem'][0]:.3f} helpers peak MB {t['mem'][1]:.3f}")
    assert t["nstar"] == 0, "a share went to a particle that is no longer gas"
    assert dM <= TOL
    assert dP <= TOL
    assert t["nfloor"] == 0, "the u floor fired"
    assert dE <= TOL


def _run(args, out, env=None, params=f"{TEST_NAME}.params"):
    return subprocess.run(mpi_launch_command(NRANKS, 1) + ["./GIZMO", params] + args, stdout=open(out, "w"), stderr=subprocess.STDOUT, env=env)


def _build(flags, name):
    build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    shutil.move(f"{TEST_DIR}/GIZMO", f"{TEST_DIR}/GIZMO_{name}")


def _arm(name):
    shutil.copy(f"GIZMO_{name}", "GIZMO"); shutil.rmtree("output", ignore_errors=True)
    res = _run(["0"], f"{name}.out"); assert res.returncode == 0, open(f"{name}.out").read()[-400:]
    return _totals("output/conserved_totals.txt"), open(f"{name}.out").read()


def test_pairflux_exact_exchange():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    hook = lambda mode: ("TEST_PAIRFLUX_ADD_MASS_AND_WAKE=40", f"TEST_PAIRFLUX_HOOK_MODE={mode}")
    _build((), "hydro")
    _build(hook(1), "mass")
    _build(hook(2), "merge")
    _build(hook(3) + (f"MERGE_SPLIT_HARDCODE_MAX_MASS={2.01*M0:.6e}",), "split")
    build_gizmo_for_test(TEST_NAME, config_file="Config_noexchange.sh")
    shutil.move(f"{TEST_DIR}/GIZMO", f"{TEST_DIR}/GIZMO_control")
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    failures = []
    try:
        shutil.copy("GIZMO_hydro", "GIZMO")
        for k in (1, 2):
            shutil.rmtree("output", ignore_errors=True)
            res = _run(["0"], f"plain{k}.out"); assert res.returncode == 0, open(f"plain{k}.out").read()[-400:]
            shutil.copy("output/conserved_totals.txt", f"plain{k}_totals.txt")
        plain = _totals("plain1_totals.txt")

        def arm(label, f):
            try:
                f()
                print(f"ARM {label}: PASS")
            except AssertionError as e:
                print(f"ARM {label}: FAIL {e}")
                failures.append(label)

        arm("plain", lambda: _check(plain, plain))
        def rerun_arm():
            assert filecmp.cmp("plain1_totals.txt", "plain2_totals.txt", shallow=False), "two identical runs differ"
        arm("identical rerun", rerun_arm)

        def restart_arm():
            shutil.rmtree("output", ignore_errors=True)
            res = _run(["0"], "first.out", env=dict(os.environ, GIZMO_TEST_RESTART_AT_STEP=str(STOP_STEP)))
            assert res.returncode == 0, open("first.out").read()[-400:]
            first = _totals("output/conserved_totals.txt"); shutil.copy("output/conserved_totals.txt", "first_totals.txt")
            assert int(first["steps"][-1]) in (STOP_STEP - 1, STOP_STEP), f"the stopping run ended at step {first['steps'][-1]}, not at {STOP_STEP}"
            res = _run(["1"], "restart.out"); assert res.returncode == 0, open("restart.out").read()[-400:]
            second = _totals("output/conserved_totals.txt")
            assert int(second["steps"][0]) == int(first["steps"][-1]) + 1, f"the restart began at step {second['steps'][0]}, the stop was at {first['steps'][-1]}"
            _check(first, second)
        arm("restart", restart_arm)

        def mass_arm():
            t, log = _arm("mass"); assert "mode 1 changed" in log; _check(t, t)
        arm("mass", mass_arm)

        def merge_arm():
            t, log = _arm("merge"); assert re.search(r"[1-9][0-9]* particles merged", log), "no merge happened"; _check(t, t)
        arm("merge", merge_arm)

        def split_arm():
            t, log = _arm("split"); assert re.search(r"[1-9][0-9]* particles split", log), "no split happened"; _check(t, t)
        arm("split", split_arm)

        def control_arm():  # the same run without the exchange must miss the bar, or the test shows nothing
            t, _ = _arm("control")
            dE = np.abs(t["E"] - t["E"][0]).max() / t["E"][0]
            dP = np.abs(t["P"] - t["P"][0]).max() / t["pscale"]
            print(f"control without the exchange: max|dE|/E {dE:.2e}  max|dP|/sum m|v| {dP:.2e}")
            assert max(dE, dP) > TOL, "the run without the exchange already meets the bar"
        arm("control", control_arm)
    finally:
        os.chdir(cwd)
    assert not failures, f"failed arms: {failures}"
