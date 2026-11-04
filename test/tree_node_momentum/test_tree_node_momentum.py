"""Node bulk velocity against the particles after a tree build and after a moment refresh, pure N-body."""

import re
import subprocess
import sys

import pytest

from gizmo.test import build_and_run_test

TEST_NAME = "tree_node_momentum"
TEST_DIR = f"test/{TEST_NAME}"
VEL_ERR_MAX = 1e-10


def _audit(flags, ranks):
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    build_and_run_test(TEST_NAME, ranks, 1, extra_config_flags=flags)
    lines = [l for l in open(f"{TEST_DIR}/test_{TEST_NAME}.out") if l.startswith("TREE_NODE_MOMENTUM_AUDIT")]
    assert lines, "no audit line"
    print(lines[-1])
    return {k: float(v) for k, v in re.findall(r"(\w+) (\S+)", lines[-1].split(None, 1)[1])}


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("mode", [1, 2], ids=["build", "refresh"])
def test_node_momentum(mode, ranks):
    a = _audit((f"TEST_TREE_NODE_MOMENTUM_AUDIT={mode}",), ranks)
    assert a["calls"] > 0 and a["vel_err"] < VEL_ERR_MAX
