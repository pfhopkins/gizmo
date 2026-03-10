"""GMC cooling and chemistry test"""

from gizmo.test import build_and_run_test, get_cooling_tables
from os import system, path


def test_gmc_cooling():
    test_name = "gmc_cooling"
    get_cooling_tables()
    build_and_run_test(test_name)
    if not path.isfile("output/snapshot_010.hdf5"):
        assert False
    assert True
