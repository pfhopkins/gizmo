"""KETJU integrator (MSTAR) googletest suite against GIZMO's vendored ketju-integrator.

Copies ketju-integrator/ to a temporary directory (its CMake writes bin/ and lib/ into the
source tree), builds unit-tests/ with the vendored googletest, and runs it on 1, 2 and 4
MPI tasks. No GIZMO build or simulation is involved.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KETJU_SRC = REPO_ROOT / "ketju-integrator"
EXE = "ketju-integrator-unit-tests"
NUM_MAKE_JOBS = 4

# the 26 cases of the original suite and 8 for GIZMO's additions; a missing one means the suite did not run completely
EXPECTED_TESTS = {
    "Integration_versus_old_integrator.binary_system_full_pn_no_spin",
    "Integration_versus_old_integrator.binary_system_no_pn",
    "Integration_threebody_terms.binary_system_with_and_without_threebody_terms",
    "Integration_units.binary_system_full_pn_spin_generic_units",
    "Integration_with_mergers.head_on_merger",
    "Integration_with_mergers.merger_from_orbit",
    "Instant_mergers.binary_merger_with_extra_data",
    "Instant_mergers.multiple_merger_with_extra_data1",
    "Instant_mergers.multiple_merger_with_extra_data2",
    "Instant_mergers.multiple_merger_with_extra_data3",
    # GIZMO's additions (unit-tests/gizmo_additions.cpp)
    "Gizmo_collision_radius.null_after_create",
    "Gizmo_star_collisions.head_on_contact",
    "Gizmo_star_collisions.graze_and_miss",
    "Gizmo_collision_energy.includes_other_members_potential",
    "Gizmo_collision_energy.tide_and_softening",
    "Gizmo_pericentre_guard.osculating_pericentre_time",
    "Gizmo_pericentre_guard.k3_triple_no_false_collision",
    "Gizmo_external_tide.harmonic_com_and_energy",
} | {f"MST/ConstructionTest.random_stars/{alg}_{n}_particles" for alg in ("Prim", "DnC") for n in (2, 4, 10, 20, 50, 100, 500, 1000)}


def _run(cmd, cwd, timeout, env=None):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    assert res.returncode == 0, f"{' '.join(map(str, cmd))} failed ({res.returncode}):\n{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    return res


@pytest.fixture(scope="module")
def unit_test_exe(tmp_path_factory):
    for tool in ("cmake", "make", "mpicc", "mpicxx", "mpiexec"):
        assert shutil.which(tool), f"{tool} not on PATH (load the gcc, openmpi and cmake modules)"
    assert (KETJU_SRC / "thirdparty" / "googletest" / "CMakeLists.txt").is_file(), "vendored googletest missing"
    work = tmp_path_factory.mktemp("ketju_unit")
    src = work / "ketju-integrator"
    shutil.copytree(KETJU_SRC, src, ignore=shutil.ignore_patterns("bin", "lib", "build*", "*.so", "*.a"))
    build = work / "build"
    build.mkdir()
    env = dict(os.environ, CC="mpicc", CXX="mpicxx")
    _run(["cmake", str(src), "-DUNIT_TESTS=ON", "-DBUILD_GMOCK=OFF", "-DINSTALL_GTEST=OFF", "-DCMAKE_BUILD_TYPE=Release"], build, 600, env)
    _run(["make", f"-j{NUM_MAKE_JOBS}", EXE], build, 1800)
    exe = src / "bin" / EXE
    assert exe.is_file(), f"{exe} was not built"
    return exe


@pytest.mark.parametrize("ntasks", [1, 2, 4])
def test_ketju_unit(unit_test_exe, ntasks):
    # rank 0 prints every test and gathers failures from all ranks; the exit code is nonzero on any failure
    res = subprocess.run(
        ["mpiexec", "-np", str(ntasks), "--oversubscribe", str(unit_test_exe)],
        cwd=unit_test_exe.parent, capture_output=True, text=True, timeout=600,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"unit tests exited {res.returncode} on {ntasks} task(s):\n{out[-6000:]}"
    ran = set(re.findall(r"^\*\*\* Test (\S+) ending\.$", res.stdout, re.M))
    assert ran == EXPECTED_TESTS, f"missing: {sorted(EXPECTED_TESTS - ran)}; unexpected: {sorted(ran - EXPECTED_TESTS)}"
