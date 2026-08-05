"""Pytest configuration: ensure tests import `gizmo` from the local
`python_src/` source tree, regardless of whether the package is pip-installed,
guarantee each test leaves the working directory where it found it, and refuse to run
two test sessions concurrently in the same working tree."""

import os
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python_src"

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


@pytest.fixture(scope="session", autouse=True)
def exclusive_tree_lock():
    """Refuse to run if another test session is already using this working tree.

    Every test builds ./GIZMO in the repo root and writes to test/<name>/output/ before that
    directory is moved to its per-variant name, so two concurrent sessions silently interleave
    each other's snapshots and binaries. The failure is invisible: each output directory ends up
    holding a mix of two runs, and the resulting energy budgets are spliced across physically
    different setups. Two overlapping Slurm jobs produced exactly that -- a 27% apparent
    conservation error that was purely an artifact of the splice, plus a spurious "No snapshots
    produced" when one job moved output/ out from under the other. Fail loudly instead.

    Use a separate checkout or `git worktree` for concurrent runs; that is what the sbatch
    wrappers are for. Set GIZMO_TEST_ALLOW_CONCURRENT=1 to override (e.g. if a stale lock
    survived a hard kill).
    """
    if os.environ.get("GIZMO_TEST_ALLOW_CONCURRENT"):
        yield
        return
    import fcntl

    lock_path = REPO_ROOT / ".pytest_tree.lock"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        pytest.exit(
            f"another pytest session already holds {lock_path}. Concurrent runs in one working "
            "tree corrupt each other's output (interleaved snapshots, shared ./GIZMO). Use a "
            "separate checkout or git worktree, or set GIZMO_TEST_ALLOW_CONCURRENT=1 to override.",
            returncode=2,
        )
    fh.write(f"pid={os.getpid()} host={os.uname().nodename} job={os.environ.get('SLURM_JOB_ID','-')}\n")
    fh.flush()
    try:
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


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
