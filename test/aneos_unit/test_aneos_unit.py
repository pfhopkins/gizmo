"""ANEOS standalone unit test.

Compiles and runs a C++ test program that generates an ideal-gas SESAME
table, loads it with aneos_read_table(), and verifies direct (rho,T)
lookups, temperature inversion, derived quantities (Cv, Gruneisen),
and boundary clamping against analytic values.

No GIZMO build or simulation is required — this tests the ANEOS module
in isolation.
"""

import os
import subprocess
import sys
from os.path import isfile, dirname, abspath, join


def test_aneos_unit():
    test_dir = dirname(abspath(__file__))
    src = join(test_dir, "test_aneos_unit.cc")
    exe = join(test_dir, "test_aneos_unit")
    aneos_cc = join(test_dir, "..", "..", "src", "eos", "aneos.cc")

    # Compile (allvars.h includes hdf5.h: its include directory from the environment the GIZMO Makefile uses)
    hdf5_dirs = [os.path.join(os.environ[v], "include") for v in ("HDF5_HOME", "HDF5_ROOT") if os.environ.get(v)]
    hdf5_dirs += [os.environ["TACC_HDF5_INC"]] if os.environ.get("TACC_HDF5_INC") else []
    hdf5_include = [f"-I{d}" for d in hdf5_dirs if os.path.isdir(d)]
    result = subprocess.run(
        [
            "mpicxx", "-std=c++17", "-O2",
            "-DEOS_ANEOS", "-DGIZMO_config_H",
            "-I", join(test_dir, "..", ".."),
            *hdf5_include,
            src, aneos_cc,
            "-o", exe, "-lm",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"

    # Run
    result = subprocess.run(
        [exe],
        capture_output=True, text=True,
        cwd=test_dir,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"Unit test failed:\n{result.stdout}\n{result.stderr}"
    assert "FAILED" not in result.stdout, f"Some checks failed:\n{result.stdout}"
