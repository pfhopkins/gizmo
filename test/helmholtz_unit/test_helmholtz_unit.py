"""Helmholtz EOS standalone unit test.

Compiles and runs a C++ test program that reads the Helmholtz free energy
table (helm_table.dat), evaluates the EOS at various (rho, T, abar, Ye)
points, and verifies:
  - Analytic ion/radiation contributions match exactly
  - Temperature inversion round-trips to machine precision
  - Thermodynamic consistency (Maxwell relations)
  - Monotonicity in temperature
  - Physical limits (P_rad, sound speed < c)

No GIZMO build or simulation is required — this tests the Helmholtz
module in isolation.
"""

import subprocess
import sys
from os.path import isfile, dirname, abspath, join


def test_helmholtz_unit():
    test_dir = dirname(abspath(__file__))
    src = join(test_dir, "test_helmholtz.cc")
    exe = join(test_dir, "test_helmholtz")
    helm_cc = join(test_dir, "..", "..", "src", "eos", "helmholtz", "helmholtz.cc")
    table = join(test_dir, "..", "..", "src", "eos", "helmholtz", "helm_table.dat")

    if not isfile(table):
        from urllib.request import urlretrieve
        urlretrieve("http://www.tapir.caltech.edu/~phopkins/public/helm_table.dat",
                    table)

    # Compile
    result = subprocess.run(
        [
            "mpicxx", "-std=c++17", "-O2",
            "-DEOS_HELMHOLTZ",
            "-I", join(test_dir, "..", ".."),
            src, helm_cc,
            "-o", exe, "-lm",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"

    # Run
    result = subprocess.run(
        [exe, table],
        capture_output=True, text=True,
        cwd=test_dir,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"Unit test failed:\n{result.stdout}\n{result.stderr}"
    assert "FAILED:" not in result.stdout or "FAILED: 0" in result.stdout, \
        f"Some checks failed:\n{result.stdout}"
