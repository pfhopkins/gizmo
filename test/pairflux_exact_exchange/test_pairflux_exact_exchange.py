"""HYDRO_EXACT_PAIR_FLUX_EXCHANGE: a Sedov blast with a deep timestep hierarchy and wake-ups, 4 ranks, no gravity.
Gas mass, momentum and thermal + kinetic (+ magnetic) energy from the conserved variables after every sync (conserved_totals.txt).
Bars (fixed before running):
- hydro, plain: |dM|/M, |dP|/sum m|v|, |dE|/E <= 1e-12 at every sync; the u floor never fires; no share to a non-gas particle;
  a second identical run gives the same conserved_totals.txt byte for byte.
- stop part-way and restart (restart files hold the records waiting for the opening kick): same bars over both runs; the
  stopping run must return 0 having reached the stop step.
- mass change: at step 40 inactive cells near the blast gain half their mass (test hook, booked) and are woken; the same
  bars on the totals minus what the hook added.
- MHD (Dedner cleaning, Powell terms off to isolate the exchange): |dM|/M and |dP|/sum m|v| <= 1e-12; energy is reported,
  not barred (the kick's magnetic energy and the Dedner correction are second order)."""

import filecmp
import os
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


def _totals(path):
    a = np.loadtxt(path)
    inj = a[:, 35:40] if a.shape[1] >= 40 else np.zeros((len(a), 5))
    M = a[:, 2] - inj[:, 0]; P = a[:, 3:6] - inj[:, 1:4]; E = a[:, 6] + a[:, 7] + a[:, 8] - inj[:, 4]
    return dict(M=M, P=P, E=E, pscale=a[0, 9], nfloor=a[-1, 14], nstar=a[-1, 16], steps=a[:, 0], rows=len(a))


def _check(ref, t, energy=True):
    dM = np.abs(t["M"] - ref["M"][0]).max() / ref["M"][0]
    dP = np.abs(t["P"] - ref["P"][0]).max() / ref["pscale"]
    dE = np.abs(t["E"] - ref["E"][0]).max() / ref["E"][0]
    print(f"rows {t['rows']}  max|dM|/M {dM:.2e}  max|dP|/sum m|v| {dP:.2e}  max|dE|/E {dE:.2e}  floor firings {int(t['nfloor'])}  shares to non-gas {int(t['nstar'])}")
    assert t["nstar"] == 0, "a share went to a particle that is no longer gas"
    assert dM <= TOL and dP <= TOL
    if energy:
        assert t["nfloor"] == 0, "the u floor fired"
        assert dE <= TOL


def _run(args, out, env=None, params=f"{TEST_NAME}.params"):
    return subprocess.run(mpi_launch_command(NRANKS, 1) + ["./GIZMO", params] + args, stdout=open(out, "w"), stderr=subprocess.STDOUT, env=env)


def _build(flags, name):
    build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    shutil.move(f"{TEST_DIR}/GIZMO", f"{TEST_DIR}/GIZMO_{name}")


def _use(name):
    shutil.copy(f"GIZMO_{name}", "GIZMO")


def test_pairflux_exact_exchange():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    _build((), "hydro")
    _build(("TEST_PAIRFLUX_ADD_MASS_AND_WAKE=40",), "mass")
    _build(("MAGNETIC", "MHD_B_SET_IN_PARAMS", "TEST_MHD_NO_POWELL_SOURCE_TERMS"), "mhd")
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        _use("hydro")
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
        assert first["steps"][-1] >= STOP_STEP - 1, f"the stopping run ended at step {first['steps'][-1]}, not at {STOP_STEP}"
        res = _run(["1"], "restart.out"); assert res.returncode == 0, open("restart.out").read()[-400:]
        _check(first, _totals("output/conserved_totals.txt"))
        _use("mass"); shutil.rmtree("output", ignore_errors=True)
        res = _run(["0"], "mass.out"); assert res.returncode == 0, open("mass.out").read()[-400:]
        assert "inactive cells gained mass and were woken" in open("mass.out").read()
        mass = _totals("output/conserved_totals.txt"); _check(mass, mass)
        _use("mhd"); shutil.rmtree("output", ignore_errors=True)
        open("mhd.params", "w").write(open(f"{TEST_NAME}.params").read() + "BiniX 3e-7\nBiniY 1e-7\nBiniZ 0\nDivBcleaningParabolicSigma 0.2\nDivBcleaningHyperbolicSigma 1.0\n")
        res = _run(["0"], "mhd.out", params="mhd.params"); assert res.returncode == 0, open("mhd.out").read()[-400:]
        mhd = _totals("output/conserved_totals.txt"); _check(mhd, mhd, energy=False)
    finally:
        os.chdir(cwd)
