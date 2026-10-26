"""KETJU integrator (MSTAR) unit tests against GIZMO's ketju-integrator submodule.

Default: the C++/Kokkos library's tests (kokkos/tests, ctest): the GIZMO-addition tests on mstar::mstar, C vs Kokkos
(mstar_compare_c), extra forces, embedded tide, driver, and KETJU's unit tests on the C library (serial MPI stand-in).
With KETJU_LEGACY_C_LIBRARY in GIZMO_EXTRA_CONFIG_FLAGS: also the original googletest suite on the C library, on 1, 2
and 4 MPI tasks. Sources are copied to a temporary directory (the top-level CMake writes bin/ and lib/ into the source
tree). No GIZMO build or simulation is involved.
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


LEGACY_C = "KETJU_LEGACY_C_LIBRARY" in os.environ.get("GIZMO_EXTRA_CONFIG_FLAGS", "").split()
KOKKOS_CTESTS = {"ketju_unit_c", "mstar_gizmo_additions", "mstar_compare_c", "mstar_extra_force", "mstar_embedded_tide", "mstar_driver"}


def _kokkos_root():
    # KOKKOS_ROOT from the environment, else the Makefile's default (as the GIZMO build)
    if os.environ.get("KOKKOS_ROOT"):
        return os.environ["KOKKOS_ROOT"]
    m = re.search(r"^KOKKOS_ROOT \?= (\S+)", (REPO_ROOT / "Makefile").read_text(), re.M)
    assert m, "KOKKOS_ROOT not set and no default in the Makefile"
    return m.group(1)


def test_ketju_kokkos_ctests(tmp_path):
    for tool in ("cmake", "make", "gcc", "g++"):
        assert shutil.which(tool), f"{tool} not on PATH (load the gcc and cmake modules)"
    src = tmp_path / "ketju-integrator"
    shutil.copytree(KETJU_SRC, src, ignore=shutil.ignore_patterns("bin", "lib", "build*", "*.so", "*.a", ".git"))
    build = tmp_path / "build"
    _run(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++", "-DKETJU_KOKKOS=ON",
          "-DKETJU_SERIAL_MPI=ON", f"-DKokkos_ROOT={_kokkos_root()}", "-DCMAKE_BUILD_TYPE=Release"], tmp_path, 600)
    _run(["cmake", "--build", str(build), "-j", str(NUM_MAKE_JOBS)], tmp_path, 1800)
    res = subprocess.run(["ctest", "--output-on-failure"], cwd=build, capture_output=True, text=True, timeout=1200)
    assert res.returncode == 0, f"ctest failed:\n{res.stdout[-6000:]}\n{res.stderr[-2000:]}"
    ran = set(re.findall(r"Test +#\d+: (\S+) \.+ +Passed", res.stdout))
    assert ran == KOKKOS_CTESTS, f"missing: {sorted(KOKKOS_CTESTS - ran)}; unexpected: {sorted(ran - KOKKOS_CTESTS)}"


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


@pytest.mark.skipif(not LEGACY_C, reason="C library googletests: KETJU_LEGACY_C_LIBRARY in GIZMO_EXTRA_CONFIG_FLAGS")
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
