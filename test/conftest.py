"""Pytest configuration: ensure tests import `gizmo` (the test harness) from the
local `test/harness/` source tree, regardless of whether the package is pip-installed."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "test" / "harness"

if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))


@pytest.fixture(autouse=True)
def _restore_cwd():
    """Put the working directory back after every test.

    Most tests address data as repo-root-relative paths ("test/<name>/<name>_ics.hdf5"),
    and ~25 of them chdir into their own directory with the pattern

        chdir(f"test/{name}/")   ...body...   chdir("../../")

    whose restore is skipped when the body raises. One failure then leaves the process
    inside that directory and every later test resolving a relative path dies with a
    confusing FileNotFoundError instead of its real assertion. In the first whole-suite
    run this turned a single early failure (c_shock, 2nd of 279) into ~100 downstream
    ones. Guarding it centrally fixes all of them without touching each test, and keeps
    working if someone adds another chdir.
    """
    saved = os.getcwd()
    try:
        yield
    finally:
        if os.getcwd() != saved:
            os.chdir(saved)


# ---------------------------------------------------------------------------
# Global figure-format switch.
#
# Test plots are saved as PNG by default (universally safe -- some clusters
# lack the libraries for vector backends). For line/point figures a vector
# format is far more useful, so allow opting in WITHOUT editing every test:
#
#     pytest test/*  --figure-format pdf          # or svg
#     GIZMO_TEST_FIGURE_FORMAT=pdf  pytest test/*  # env var, same effect
#
# The mechanism rewrites a trailing ".png" passed to matplotlib's savefig to
# the requested extension (matplotlib's pdf/svg backends are built in, so no
# external lib is needed; the change is a no-op in the default "png" mode).
# Calls that already target another extension, that pass an explicit
# format=..., or that write to non-path targets (e.g. os.devnull) are left
# untouched.
# ---------------------------------------------------------------------------

_FIGURE_FORMAT = "png"


def pytest_addoption(parser):
    parser.addoption(
        "--figure-format",
        action="store",
        default=os.environ.get("GIZMO_TEST_FIGURE_FORMAT", "png").lower(),
        choices=("png", "pdf", "svg"),
        help="Output format for test figures (default: png). Use pdf/svg for "
             "vector output of line/point plots. Overrides $GIZMO_TEST_FIGURE_FORMAT.",
    )


def pytest_configure(config):
    global _FIGURE_FORMAT
    _FIGURE_FORMAT = config.getoption("--figure-format").lower()
    if _FIGURE_FORMAT == "png":
        return  # default behaviour, nothing to patch

    import matplotlib.figure

    if getattr(matplotlib.figure.Figure.savefig, "_gizmo_format_patched", False):
        return  # idempotent (e.g. plugin re-entry)

    _orig_savefig = matplotlib.figure.Figure.savefig

    def _retarget(fname):
        """Swap a trailing .png for the requested format; pass anything else through."""
        try:
            s = os.fspath(fname)  # str/PathLike -> str; raises TypeError for file objects etc.
        except TypeError:
            return fname
        if isinstance(s, bytes):
            return fname
        if s.lower().endswith(".png"):
            return s[:-4] + "." + _FIGURE_FORMAT
        return fname

    def savefig(self, *args, **kwargs):
        # respect an explicit format=... ; only rewrite the filename's .png suffix
        if "format" not in kwargs:
            if args:
                args = (_retarget(args[0]),) + args[1:]
            elif "fname" in kwargs:
                kwargs["fname"] = _retarget(kwargs["fname"])
        return _orig_savefig(self, *args, **kwargs)

    savefig._gizmo_format_patched = True
    matplotlib.figure.Figure.savefig = savefig  # covers both fig.savefig and plt.savefig (-> gcf().savefig)
    print(f"\n[gizmo-tests] figure output format: {_FIGURE_FORMAT} (.png targets retargeted)")
