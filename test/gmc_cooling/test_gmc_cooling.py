"""A trivial test for pytest"""

from os import system, environ


def test_trivial():
    environ["TEST_NAME"] = "gmc_cooling"
    system("bash test/build_gizmo_for_test.sh")
    assert True
