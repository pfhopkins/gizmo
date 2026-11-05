"""A NUCLEAR_NETWORK_SOLVER=1 (SkyNet) build links.

Needs SkyNet: set SKYNET_DIR to its install prefix and make its headers and libraries (and the HDF5 C++, Boost, GSL and LAPACK ones it
uses) visible to the compiler and linker (CPATH, LIBRARY_PATH, LD_LIBRARY_PATH, link flags). Skipped without SKYNET_DIR.
"""
import os
from pathlib import Path

import pytest

from gizmo.test import build_gizmo_for_test, default_omp_threads

TEST_NAME = "nuclear_skynet_link"


@pytest.mark.skipif("SKYNET_DIR" not in os.environ, reason="SKYNET_DIR (SkyNet install prefix) not set")
def test_nuclear_skynet_link():
    build_gizmo_for_test(TEST_NAME, default_omp_threads())  # a failed link raises here
    assert (Path(__file__).resolve().parent / "GIZMO").is_file()  # the harness moves the binary into the test directory
