"""Regression test for the equilibrium dust-temperature solvers, rt_eqm_dust_temp (A, RT kick side) and
rt_ir_lambdadust (B, inside the implicit cooling solve).

Builds GIZMO with the STARFORGE full-physics configuration, links dust_solver_driver.cc against its objects,
and runs both solvers on the cell states in cases.txt. The driver also brute-force scans each solver's own
residual, so every case is judged against the root of the residual in the build under test: a physics
change that moves the root does not break the test, a solver that misses the root does.

Failures these solvers have had, which this test catches (cases.txt groups):
  1xx  B pinned at MAX_DUST_TEMP (no gas-dust coupling) although its residual has a root below.
  2xx  A stopped well short of its root (step-size convergence test, untightened bracket ends).
  3xx  B resolved Tdust but not T-Tdust: LambdaDust = alpha*(T-Tdust) is what the gas sees, and in tightly
       coupled gas T-Tdust is 1e-10..1e-3 K, so a 1e-6 relative tolerance on Tdust alone gets it wrong by O(1).
  4xx  B's answer depended on the previous dust temperature it was entered with.
  5xx  controls.
No simulation is run; the build is the only slow part.
"""

import os
import subprocess
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_cooling_tables, link_test_driver

TEST_NAME = "dust_temp_solvers"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
BUILD_DIR = join(TEST_DIR, "output")
DRIVER = join(BUILD_DIR, "dust_solver_driver")
MAX_DUST_TEMP = 1e4  # rt_ir_lambdadust's ceiling

TOL_TD_B = 1e-4      # B: relative distance from the residual's lowest root
TOL_TD_A = 1e-3      # A: relative distance from the residual's root
TOL_COUPLED = 1e-3   # B: |Tdust - root| relative to |T - root|
TOL_ENTRY = 1e-4     # B: spread of the answers for one cell over the entry dust temperatures


def _link_driver():
    link_test_driver(join(TEST_DIR, "dust_solver_driver.cc"), DRIVER)


def _run_driver():
    get_cooling_tables(TEST_DIR)
    with open(join(TEST_DIR, "cases.txt")) as fin:
        # relative parameter-file path: GIZMO truncates long ones
        r = subprocess.run([DRIVER, "dust_temp_solvers.params"], stdin=fin, capture_output=True, text=True, cwd=TEST_DIR)
    assert r.returncode == 0, "driver failed:\n%s" % r.stderr[-4000:]
    cols = "id T TdA nrootA rootAlo rootAhi TdB nrootB rootBlo rootBhi".split()
    rows = [l.split() for l in r.stdout.splitlines() if l[:1].isdigit()]
    a = np.array(rows, dtype=float)
    return {c: a[:, j] for j, c in enumerate(cols)}


def _rel(x, y):
    return abs(x - y) / abs(y)


def test_dust_temp_solvers():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=_link_driver)
    finally:
        os.chdir(cwd)
    r = _run_driver()
    with open(join(TEST_DIR, "cases.txt")) as f:
        ncases = sum(1 for l in f if l[:1].isdigit())
    assert len(r["id"]) == ncases, "driver returned %d of %d cases" % (len(r["id"]), ncases)

    failures = []
    for k in range(len(r["id"])):
        i, group = int(r["id"][k]), int(r["id"][k]) // 100
        T, TdA, TdB, Alo, Blo = r["T"][k], r["TdA"][k], r["TdB"][k], r["rootAlo"][k], r["rootBlo"][k]
        why = []
        if group in (1, 3, 4, 5):
            if r["nrootB"][k] < 1:
                why.append("B residual has no root to compare against")
            elif group == 3:
                if abs(TdB - Blo) > TOL_COUPLED * abs(T - Blo) + 1e-12 * T:
                    why.append("B T-Tdust off by %.2g of itself (T-Td=%.3g K)" % (abs(TdB - Blo) / abs(T - Blo), T - Blo))
            elif TdB >= 0.999 * MAX_DUST_TEMP and Blo < 0.9 * MAX_DUST_TEMP:
                why.append("B pinned at the ceiling, lowest root %.6g K" % Blo)
            elif _rel(TdB, Blo) > TOL_TD_B:
                why.append("B Td=%.8g, lowest root %.8g (rel %.2g)" % (TdB, Blo, _rel(TdB, Blo)))
        if group in (2, 5):
            if r["nrootA"][k] < 1:
                why.append("A residual has no root to compare against")
            elif _rel(TdA, Alo) > TOL_TD_A:
                why.append("A Td=%.8g, root %.8g (rel %.2g)" % (TdA, Alo, _rel(TdA, Alo)))
        if why:
            failures.append("  %3d T=%-10.5g %s" % (i, T, "; ".join(why)))

    for first in (400, 410):
        m = (r["id"] >= first) & (r["id"] < first + 10)
        td = r["TdB"][m]
        if td.max() > (1 + TOL_ENTRY) * td.min():
            failures.append("  %d-%d B depends on the entry dust temperature: %s" % (first, first + 9, " ".join("%.6g" % x for x in td)))

    print("\n%d cases, %d failing" % (ncases, len(failures)))
    assert not failures, "dust-temperature solver regressions:\n" + "\n".join(failures)
