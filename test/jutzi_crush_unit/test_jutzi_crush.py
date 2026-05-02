"""Phase 17e bit-2 P-alpha analytic crush-curve unit test.

Builds a tiny standalone C++ harness around solids/jutzi_crush_curve.h, sweeps a
range of pressures and material params, and asserts the output bit-matches the
Python reference implementation of Jutzi+ 2008 Eq. 8.

No GIZMO build or MPI required.
"""

import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
HARNESS_SRC = THIS_DIR / "jutzi_crush_unit_harness.cc"
HARNESS_BIN = THIS_DIR / "jutzi_crush_unit_harness"


def _ref_eq8(P, alpha_0, P_e, P_s):
    """Python reference: Jutzi 2008 Eq. 8 (irreversible compaction is applied
    by the caller, not in the equilibrium curve itself)."""
    if not (alpha_0 > 1.0):
        return 1.0
    if not (P_s > P_e):
        return alpha_0
    if P <= P_e:
        return alpha_0
    if P >= P_s:
        return 1.0
    r = (P_s - P) / (P_s - P_e)
    return 1.0 + (alpha_0 - 1.0) * r * r


@pytest.fixture(scope="module")
def harness():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    subprocess.run(
        [cxx, "-std=c++17", "-O2", str(HARNESS_SRC), "-o", str(HARNESS_BIN)],
        check=True,
    )
    yield HARNESS_BIN
    try:
        os.remove(HARNESS_BIN)
    except OSError:
        pass


# Sample (alpha_0, P_e, P_s) triplets covering Jutzi+ 2008 published basalt
# values + an ice-like and a non-porous (alpha_0=1) edge case.
MATERIALS = [
    (1.275, 1.0e8, 7.0e10),  # basalt-like (low porosity)
    (2.0,   1.0e7, 1.5e9),   # icy regolith
    (1.0,   1.0e8, 1.0e10),  # solid (alpha_0=1, should always return 1)
    (1.5,   0.0,   1.0e9),   # P_e = 0 edge case
]


def test_eq8_bitmatch_cxx_vs_python(harness):
    """Sweep pressures from below P_e to well above P_s; the C++ harness must
    return the same alpha as the Python reference to bit precision."""
    pressures = [
        -1.0e9, 0.0, 1.0e6, 5.0e7, 1.0e8, 3.0e8, 1.0e9, 5.0e9,
        1.0e10, 7.0e10, 1.0e12,
    ]
    cases = []
    for (a0, Pe, Ps) in MATERIALS:
        for P in pressures:
            cases.append((P, a0, Pe, Ps))
    stdin_blob = "\n".join(f"{P:.17g} {a0:.17g} {Pe:.17g} {Ps:.17g}"
                           for (P, a0, Pe, Ps) in cases)
    proc = subprocess.run([str(harness)], input=stdin_blob, capture_output=True,
                          text=True, check=True)
    out = proc.stdout.strip().splitlines()
    assert len(out) == len(cases), f"harness returned {len(out)} lines, expected {len(cases)}"
    for (P, a0, Pe, Ps), got_str in zip(cases, out):
        got = float(got_str)
        ref = _ref_eq8(P, a0, Pe, Ps)
        assert got == ref, (
            f"mismatch at P={P}, a0={a0}, P_e={Pe}, P_s={Ps}: "
            f"C++={got!r}, py={ref!r}"
        )


def test_eq8_known_jutzi_2008_basalt_values(harness):
    """Spot-check the C++ harness against the analytic basalt curve at points
    drawn from Jutzi+ 2008 Fig. 1 / Table 1 (alpha_0=1.275, P_e=1e8, P_s=7e10).

    These are exact analytic values from Eq. 8, not measured paper data:
    they confirm the formula matches the paper's mathematical statement."""
    a0, Pe, Ps = 1.275, 1.0e8, 7.0e10
    def _r(P):
        return (Ps - P) / (Ps - Pe)
    expected = {
        Pe:        a0,                                # at the elastic limit
        Ps:        1.0,                               # at full compaction
        0.5 * Ps:  1.0 + (a0 - 1.0) * _r(0.5 * Ps)**2,
        0.1 * Ps:  1.0 + (a0 - 1.0) * _r(0.1 * Ps)**2,
    }
    stdin_blob = "\n".join(f"{P:.17g} {a0} {Pe} {Ps}" for P in expected)
    proc = subprocess.run([str(harness)], input=stdin_blob, capture_output=True,
                          text=True, check=True)
    got = [float(x) for x in proc.stdout.strip().splitlines()]
    for (P, exp), g in zip(expected.items(), got):
        assert math.isclose(g, exp, rel_tol=1e-12), f"P={P}: got {g}, expected {exp}"


def test_eq8_irreversibility_invariants(harness):
    """At fixed (alpha_0, P_e, P_s), alpha(P) must be monotonically non-
    increasing in P, exactly equal to alpha_0 for P<=P_e, and exactly 1
    for P>=P_s. (Irreversibility is applied by the caller; this checks the
    equilibrium curve itself.)"""
    a0, Pe, Ps = 1.5, 2.0e8, 5.0e10
    Ps_grid = [Pe * 0.5, Pe, Pe * 1.5, 1.0e10, Ps * 0.5, Ps, Ps * 1.5]
    stdin_blob = "\n".join(f"{P:.17g} {a0} {Pe} {Ps}" for P in Ps_grid)
    proc = subprocess.run([str(harness)], input=stdin_blob, capture_output=True,
                          text=True, check=True)
    got = [float(x) for x in proc.stdout.strip().splitlines()]
    # below P_e -> alpha_0
    assert got[0] == a0
    assert got[1] == a0
    # above P_s -> 1
    assert got[5] == 1.0
    assert got[6] == 1.0
    # monotonic non-increasing across the full grid
    for prev, nxt in zip(got, got[1:]):
        assert nxt <= prev + 1.0e-15
