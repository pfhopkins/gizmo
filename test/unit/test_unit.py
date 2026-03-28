"""C++ unit tests: discovers, compiles, and runs all test_*.cc files in this directory."""

import subprocess
import pytest
import tempfile
from pathlib import Path

UNIT_DIR = Path(__file__).parent
GIZMO_ROOT = UNIT_DIR.parent.parent
CXX = "c++"
CXXFLAGS = ["-std=c++17", "-O2", "-Wall", "-Werror", f"-I{UNIT_DIR}", f"-I{GIZMO_ROOT}"]


def discover_tests():
    """Find all test_*.cc files in the unit test directory."""
    return sorted(UNIT_DIR.glob("test_*.cc"))


@pytest.fixture(params=discover_tests(), ids=lambda p: p.stem)
def compiled_test(request, tmp_path):
    """Compile a C++ test file and return the path to the binary."""
    src = request.param
    binary = tmp_path / src.stem
    result = subprocess.run(
        [CXX] + CXXFLAGS + [str(src), "-o", str(binary)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Compilation failed for {src.name}:\n{result.stderr}")
    return binary, src.name


def test_unit(compiled_test):
    """Run a compiled C++ unit test binary."""
    binary, name = compiled_test
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        pytest.fail(f"{name} failed:\n{output}")
