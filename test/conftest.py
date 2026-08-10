"""Pytest configuration: ensure tests import `gizmo` from the local
`python_src/` source tree, regardless of whether the package is pip-installed, and
guarantee each test leaves the working directory where it found it.

NOT enforced, but worth knowing: do not run two sessions against one working tree. Every
test builds ./GIZMO and writes Config.sh in the repo root, and writes test/<name>/output/
before moving it to a per-variant name, so concurrent sessions silently interleave
snapshots and swap each other's binaries. The symptoms are a variant directory whose
snapshot mtimes are not monotonic, a run reporting no snapshots at all because another
session moved output/ out from under it, or a variant whose startup config listing does
not match the flags it was supposed to build with. Note the sbatch wrapper does NOT create
a worktree despite its name; it cds to $SLURM_SUBMIT_DIR."""

import os
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python_src"

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore the working directory after every test, however the test exits.

    Many tests chdir into test/<name>/ and chdir back with a trailing `chdir("../../")`
    that is NOT in a finally block, so anything raising in between leaks the cwd to every
    subsequent test -- and the expected path does raise, since run_test() implements the
    GIZMO timeout as pytest.skip(). Observed: one c_shock timeout left the cwd there, so
    every later `make clean` ran from the wrong directory, giving 54 spurious build
    failures. A safety net, not a licence to skip try/finally: it warns so the offending
    test still gets fixed.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        current = os.getcwd()
        if current != original:
            os.chdir(original)
            warnings.warn(
                f"test left cwd at {current!r} instead of {original!r}; restored. "
                "Wrap the chdir in try/finally (or use build_and_run_test, which does).",
                stacklevel=1,
            )
