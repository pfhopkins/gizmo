"""Unit test for src/solids/ism_dust_hu_kernels.h (Hu+19/23/26 dust sputtering, drag and growth).

Compiles test_ism_dust_hu_kernels.cc against the header in this checkout with plain -O2 and with the production
optimisation flags, and runs it. No GIZMO build or simulation is needed.
"""

import subprocess
import sys
from os.path import abspath, dirname, join

import pytest

FLAGSETS = {"O2": ["-O2"], "production": ["-O3", "-ffast-math", "-funroll-loops"]}


@pytest.mark.parametrize("flagset", sorted(FLAGSETS))
def test_ism_dust_hu_kernels(flagset):
    test_dir = dirname(abspath(__file__))
    repo = abspath(join(test_dir, "..", "..", "src"))
    src = join(test_dir, "test_ism_dust_hu_kernels.cc")
    exe = join(test_dir, "test_ism_dust_hu_kernels_" + flagset)
    cmd = ["mpicxx", "-std=c++17"] + FLAGSETS[flagset] + ["-I", repo, src, "-o", exe, "-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, "Compilation failed:\n%s" % result.stderr

    result = subprocess.run([exe], capture_output=True, text=True, cwd=test_dir)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, "ISM dust Hu kernel test failed (%s):\n%s" % (flagset, result.stdout)
