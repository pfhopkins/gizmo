"""GMC cooling and chemistry test"""

from gizmo.test import build_and_run_test, get_cooling_tables
from os import system


def test_gmc_cooling():
    test_name = "gmc_cooling"
    get_cooling_tables()
    build_and_run_test(test_name)
    assert True
