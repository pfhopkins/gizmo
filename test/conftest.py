"""Pytest configuration: ensure tests import `gizmo` (the test harness) from the
local `test/harness/` source tree, regardless of whether the package is pip-installed."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "test" / "harness"

if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))
