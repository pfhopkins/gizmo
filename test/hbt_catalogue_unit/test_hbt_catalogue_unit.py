"""Unit test (T-C1 of the SUBFIND-HBT plan, step 2) for the group-catalogue offsets in structure/hbt/.

Compiles test_hbt_catalogue_unit.cc with hbt_catalogue_mpi.cc from this checkout and runs it on 1 and 4 MPI ranks.
No GIZMO build or simulation is needed.
"""

import subprocess
from os.path import abspath, dirname, join

import pytest


@pytest.mark.parametrize("nranks", [1, 4])
def test_hbt_catalogue_unit(nranks):
    test_dir = dirname(abspath(__file__))
    repo = abspath(join(test_dir, "..", ".."))
    exe = join(test_dir, "test_hbt_catalogue_unit")
    cmd = ["mpicxx", "-std=c++17", "-O2", "-Wall", "-I", repo, join(test_dir, "test_hbt_catalogue_unit.cc"),
           join(repo, "structure", "hbt", "hbt_catalogue_mpi.cc"), "-o", exe]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0, "Compilation failed:\n%s" % result.stderr
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", str(nranks), exe], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, cwd=test_dir)
    print(result.stdout)
    assert result.returncode == 0, "hbt catalogue unit test failed on %d ranks:\n%s\n%s" % (nranks, result.stdout, result.stderr)
