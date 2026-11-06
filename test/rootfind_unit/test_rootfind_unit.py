"""Unit test for brent_rootfind() in src/system/bracketed_rootfind_functions.h (used by DoCooling, the H2 solve and
rt_ir_lambdadust).

Compiles test_rootfind_unit.cc against the header in this checkout, once with plain -O2 and once with the
production optimisation flags (-O3 -ffast-math), and runs it. No GIZMO build or simulation is needed.
Set ROOTFIND_HEADER to test a different copy of the header.
"""

import os
import subprocess
import sys
from os.path import abspath, dirname, join

import pytest

FLAGSETS = {"O2": ["-O2"], "production": ["-O3", "-ffast-math", "-funroll-loops"]}


@pytest.mark.xfail(strict=True, reason=(
    "fixed by pr/rt-cooling: brent_rootfind returns unevaluated or off-root points (3 defects)"))
@pytest.mark.parametrize("flagset", sorted(FLAGSETS))
def test_rootfind_unit(flagset):
    test_dir = dirname(abspath(__file__))
    repo = abspath(join(test_dir, "..", ".."))
    src = join(test_dir, "test_rootfind_unit.cc")
    exe = join(test_dir, "test_rootfind_unit_" + flagset)
    cmd = ["mpicxx", "-std=c++17"] + FLAGSETS[flagset] + ["-I", repo, src, "-o", exe, "-lm"]
    header = os.environ.get("ROOTFIND_HEADER")
    if header:
        cmd.insert(2, "-DROOTFIND_HEADER=\"%s\"" % header)
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, "Compilation failed:\n%s" % result.stderr

    result = subprocess.run([exe], capture_output=True, text=True, cwd=test_dir)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, "root-finder unit test failed (%s):\n%s" % (flagset, result.stdout)
