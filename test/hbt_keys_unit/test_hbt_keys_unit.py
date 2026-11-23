"""Unit test (T1 of the SUBFIND-HBT plan) for the stable particle key in structure/hbt/.

Compiles test_hbt_keys_unit.cc with hbt_key_mpi.cc from this checkout and runs it on 1 and 4 MPI ranks.
No GIZMO build or simulation is needed.
"""

import subprocess
from os.path import abspath, dirname, join

import pytest


def test_star_spawn_generation_rule():
    """The C++ test emulates sfr_eff.cc: every star spawn must give the star the parent's incremented generation (STL-148)."""
    repo = abspath(join(dirname(abspath(__file__)), "..", ".."))
    lines = open(join(repo, "galaxy_sf", "sfr_eff.cc")).read().splitlines()
    spawn_child = [n for n, s in enumerate(lines) if "ID_child_number = P[i].ID_child_number + (MyIDType)(1 << ((int)P[i].ID_generation))" in s]
    assert len(spawn_child) == 2, "expected the Stella and the non-Stella star spawn in sfr_eff.cc, found %d" % len(spawn_child)
    for n in spawn_child:
        star = lines[n].split(".ID_child_number")[0].strip()          # P[ns] or P[i_star]
        window = " ".join(lines[n:n + 5])
        assert ("%s.ID_generation = P[i].ID_generation;" % star) in window, "sfr_eff.cc:%d: star generation not taken from the parent:\n%s" % (n + 1, "\n".join(lines[n:n + 5]))


@pytest.mark.parametrize("nranks", [1, 4])
def test_hbt_keys_unit(nranks):
    test_dir = dirname(abspath(__file__))
    repo = abspath(join(test_dir, "..", ".."))
    exe = join(test_dir, "test_hbt_keys_unit")
    cmd = ["mpicxx", "-std=c++17", "-O2", "-Wall", "-I", repo, join(test_dir, "test_hbt_keys_unit.cc"),
           join(repo, "structure", "hbt", "hbt_key_mpi.cc"), "-o", exe]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0, "Compilation failed:\n%s" % result.stderr
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", str(nranks), exe], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, cwd=test_dir)
    print(result.stdout)
    assert result.returncode == 0, "hbt key unit test failed on %d ranks:\n%s\n%s" % (nranks, result.stdout, result.stderr)
