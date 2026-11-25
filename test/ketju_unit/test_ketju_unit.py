"""KETJU integrator (MSTAR) unit tests against a source checkout of ketju-integrator (C++/Kokkos, branch main; KETJU_SOURCE_DIR):
KETJU is a separate install, not part of GIZMO.

The C++/Kokkos library's tests (tests/, ctest): KETJU's unit tests and the GIZMO additions on mstar::mstar, the comparison
against stored results of the former C library (mstar_compare_c; frozen numbers in tests/reference_data, the C library is
not built), extra forces, embedded tide, driver.
No GIZMO build or simulation is involved.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KETJU_SRC = Path(os.environ.get("KETJU_SOURCE_DIR", "/nonexistent"))  # checkout of ketju-integrator at the Makefile's KETJU_REF
if not KETJU_SRC.is_dir():
    pytest.skip("set KETJU_SOURCE_DIR to the ketju-integrator source checkout", allow_module_level=True)
NUM_MAKE_JOBS = 4


def _run(cmd, cwd, timeout, env=None):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    assert res.returncode == 0, f"{' '.join(map(str, cmd))} failed ({res.returncode}):\n{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    return res


KOKKOS_CTESTS = {"mstar_unit_tests", "mstar_compare_c", "mstar_extra_force", "mstar_embedded_tide", "mstar_driver"}


def _kokkos_root():
    # KOKKOS_ROOT from the environment, else the Makefile's default (as the GIZMO build)
    if os.environ.get("KOKKOS_ROOT"):
        return os.environ["KOKKOS_ROOT"]
    m = re.search(r"^KOKKOS_ROOT \?= (\S+)", (REPO_ROOT / "Makefile").read_text(), re.M)
    assert m, "KOKKOS_ROOT not set and no default in the Makefile"
    return m.group(1)


def test_ketju_kokkos_ctests(tmp_path):
    for tool in ("cmake", "make", "g++"):
        assert shutil.which(tool), f"{tool} not on PATH (load the gcc and cmake modules)"
    assert (KETJU_SRC / "include" / "mstar" / "mstar.hpp").is_file(), "KETJU_SOURCE_DIR has no include/mstar/mstar.hpp"
    build = tmp_path / "build"
    _run(["cmake", "-S", str(KETJU_SRC), "-B", str(build), "-DCMAKE_CXX_COMPILER=g++",
          f"-DKokkos_ROOT={_kokkos_root()}", "-DCMAKE_BUILD_TYPE=Release"], tmp_path, 600)
    _run(["cmake", "--build", str(build), "-j", str(NUM_MAKE_JOBS)], tmp_path, 1800)
    res = subprocess.run(["ctest", "--output-on-failure"], cwd=build, capture_output=True, text=True, timeout=1200)
    assert res.returncode == 0, f"ctest failed:\n{res.stdout[-6000:]}\n{res.stderr[-2000:]}"
    ran = set(re.findall(r"Test +#\d+: (\S+) \.+ +Passed", res.stdout))
    assert ran == KOKKOS_CTESTS, f"missing: {sorted(KOKKOS_CTESTS - ran)}; unexpected: {sorted(ran - KOKKOS_CTESTS)}"
