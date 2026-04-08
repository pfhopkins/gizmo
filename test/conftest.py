"""Pytest configuration: ensure tests import `gizmo` from the local
`python_src/` source tree, regardless of whether the package is pip-installed."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python_src"

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))
