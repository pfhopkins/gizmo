"""Nuclear network standalone unit test.

Compiles and runs a C++ test program that exercises the aprox13 alpha-chain
solver, NSE composition lookup, Ye/Abar computation, and energy conservation.

No GIZMO build or simulation is required — this tests the nuclear network
module in isolation by linking against nuclear_physics.cc and nuclear.cc
with the full GIZMO header infrastructure.
"""

import os
import subprocess
import sys
from os.path import dirname, abspath, exists, join


def test_nuclear_unit():
    test_dir = dirname(abspath(__file__))
    root = join(test_dir, "..", "..")
    src_dir = join(root, "src")
    src = join(test_dir, "test_nuclear.cc")
    exe = join(test_dir, "test_nuclear")

    # The nuclear network math is header-only (nuclear_physics_functions.h); the
    # GIZMO headers pull GIZMO_config.h from src/.  Write a minimal config with
    # the required flags, preserving any real (build-generated) one to restore.
    config_path = join(src_dir, "GIZMO_config.h")
    config_backup = config_path + ".nuclear_unit_bak"
    had_config = exists(config_path)
    if had_config:
        os.replace(config_path, config_backup)
    with open(config_path, "w") as f:
        f.write("#define NUCLEAR_NETWORK\n")
        f.write("#define EOS_HELMHOLTZ\n")
        f.write("#define NUCLEAR_NETWORK_NSE_TABLE\n")

    try:
        # Compile
        result = subprocess.run(
            [
                "mpicxx", "-std=c++17", "-O2",
                "-DNUCLEAR_NETWORK", "-DEOS_HELMHOLTZ",
                "-DNUCLEAR_NETWORK_NSE_TABLE",
                "-I", root,
                src,
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
        assert "FAILED:" not in result.stdout or "FAILED: 0" in result.stdout, \
            f"Some checks failed:\n{result.stdout}"
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
        if had_config:
            os.replace(config_backup, config_path)
