"""Unit test (T1 of the SUBFIND-HBT plan) for the stable particle key in structure/hbt/.

Compiles test_hbt_keys_unit.cc with hbt_key_mpi.cc from this checkout and runs it on 1 and 4 MPI ranks.
No GIZMO build or simulation is needed.
"""

import subprocess
from os.path import abspath, dirname, join

import pytest


def test_star_spawn_generation_rule():
    """gku: every spawn (star spawn in sfr_eff.cc, split and merge-split in mesh/merge_split.cc) goes through
    single_star_identity_from_parent, which gives the child the parent's incremented generation (GKS-145; STL-148 rule)."""
    repo = abspath(join(dirname(abspath(__file__)), "..", "..", "src"))
    sfr = open(join(repo, "galaxy_sf", "sfr_eff.cc")).read()
    assert "single_star_identity_from_parent(P[i], P[i_star])" in sfr, "sfr_eff.cc star spawn no longer uses single_star_identity_from_parent"
    ms = open(join(repo, "mesh", "merge_split.cc")).read()
    assert ms.count("single_star_identity_from_parent(") >= 2, "merge_split.cc splits no longer use single_star_identity_from_parent"
    hdr = open(join(repo, "galaxy_sf", "single_star_identity.h")).read()
    assert "ID_child_number = p.ID_child_number + (decltype(p.ID_child_number))(1 << ((int)p.ID_generation))" in hdr
    assert "p.ID_generation = p.ID_generation + 1;" in hdr and "c.ID_generation = p.ID_generation;" in hdr


@pytest.mark.parametrize("nranks", [1, 4])
def test_hbt_keys_unit(nranks):
    test_dir = dirname(abspath(__file__))
    repo = abspath(join(test_dir, "..", "..", "src"))
    exe = join(test_dir, "test_hbt_keys_unit")
    cmd = ["mpicxx", "-std=c++17", "-O2", "-Wall", "-I", repo, join(test_dir, "test_hbt_keys_unit.cc"),
           join(repo, "structure", "hbt", "hbt_key_mpi.cc"), "-o", exe]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0, "Compilation failed:\n%s" % result.stderr
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", str(nranks), exe], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, cwd=test_dir)
    print(result.stdout)
    assert result.returncode == 0, "hbt key unit test failed on %d ranks:\n%s\n%s" % (nranks, result.stdout, result.stderr)
