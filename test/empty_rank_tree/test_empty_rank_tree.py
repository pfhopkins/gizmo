"""A rank left with no particles after holding some: the run must go on."""

import subprocess
import sys

from gizmo.test import build_and_run_test

TEST_NAME = "empty_rank_tree"


def test_empty_rank_tree():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=f"test/{TEST_NAME}", check=True)
    build_and_run_test(TEST_NAME, 2, 1)
