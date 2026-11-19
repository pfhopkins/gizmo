"""HYDRO_EXACT_PAIR_FLUX_EXCHANGE: a Sedov blast with a deep timestep hierarchy and wake-ups, 2 ranks, no gravity.
Gas mass, momentum and thermal + kinetic energy from the conserved variables after every sync (conserved_totals.txt).
Bars: |dM|/M, |dP|/sum m|v|, |dE|/E <= 1e-12 at every sync; the u floor never fires; no share is sent to a non-gas particle.
A second run stops part-way (restart files hold the records waiting for the opening kick) and continues: same bars."""

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


def _totals(path):
    a = np.loadtxt(path)
    return a[:, 2], a[:, 3:6], a[:, 6] + a[:, 7] + a[:, 8], a[0, 9], a[-1, 14], a[-1, 16]


def _check(a0, rows):
    M, P, E, pscale, nfloor, nstar = rows
    M0, P0, E0 = a0
    dM = np.abs(M - M0).max() / M0; dP = np.abs(P - P0).max() / pscale; dE = np.abs(E - E0).max() / E0
    print(f"rows {len(M)}  max|dM|/M {dM:.2e}  max|dP|/sum m|v| {dP:.2e}  max|dE|/E {dE:.2e}  floor firings {int(nfloor)}  shares to non-gas {int(nstar)}")
    assert nstar == 0, "a share went to a particle that is no longer gas"
    assert nfloor == 0, "the u floor fired"
    assert dM <= TOL and dP <= TOL and dE <= TOL


def test_pairflux_exact_exchange():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME)
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        shutil.rmtree("output", ignore_errors=True)
        res = subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", f"{TEST_NAME}.params", "0"], stdout=open("plain.out", "w"), stderr=subprocess.STDOUT)
        assert res.returncode == 0, open("plain.out").read()[-400:]
        plain = _totals("output/conserved_totals.txt")
        _check((plain[0][0], plain[1][0], plain[2][0]), plain)
        shutil.rmtree("output_plain", ignore_errors=True); shutil.move("output", "output_plain")
        subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", f"{TEST_NAME}.params", "0"], stdout=open("first.out", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, GIZMO_TEST_RESTART_AT_STEP=str(STOP_STEP)))
        first = _totals("output/conserved_totals.txt"); shutil.copy("output/conserved_totals.txt", "first_totals.txt")
        res = subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", f"{TEST_NAME}.params", "1"], stdout=open("restart.out", "w"), stderr=subprocess.STDOUT)
        assert res.returncode == 0, open("restart.out").read()[-400:]
        second = _totals("output/conserved_totals.txt")
    finally:
        os.chdir(cwd)
    _check((first[0][0], first[1][0], first[2][0]), second)
