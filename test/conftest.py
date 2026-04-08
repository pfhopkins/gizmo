"""Pytest configuration: verify the installed `gizmo` Python package matches the
local `python_src/gizmo` source tree before any tests run. If they differ,
fail collection with instructions to reinstall."""

import filecmp
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PKG = REPO_ROOT / "python_src" / "gizmo"


def _diff_summary(local_dir: Path, installed_dir: Path):
    """Return a list of human-readable differences between two package directories.
    Compares all .py files; ignores __pycache__ and .egg-info."""
    diffs = []
    local_files = {p.relative_to(local_dir) for p in local_dir.rglob("*.py") if "__pycache__" not in p.parts}
    installed_files = {
        p.relative_to(installed_dir) for p in installed_dir.rglob("*.py") if "__pycache__" not in p.parts
    }
    for missing in sorted(local_files - installed_files):
        diffs.append(f"  missing from install: {missing}")
    for extra in sorted(installed_files - local_files):
        diffs.append(f"  stale in install (not in source): {extra}")
    for rel in sorted(local_files & installed_files):
        if not filecmp.cmp(local_dir / rel, installed_dir / rel, shallow=False):
            diffs.append(f"  differs: {rel}")
    return diffs


def pytest_configure(config):
    try:
        import gizmo  # type: ignore
    except ImportError as e:
        raise pytest.UsageError(
            f"Cannot import the installed `gizmo` package: {e}\n"
            f"Install it from the local source: pip install -e {REPO_ROOT / 'python_src'}"
        )

    installed_pkg = Path(gizmo.__file__).resolve().parent
    if installed_pkg == LOCAL_PKG:
        return  # editable install pointing at the source tree — always in sync

    diffs = _diff_summary(LOCAL_PKG, installed_pkg)
    if diffs:
        msg = (
            f"Installed `gizmo` package is out of sync with local source.\n"
            f"  installed: {installed_pkg}\n"
            f"  local:     {LOCAL_PKG}\n"
            f"Differences:\n" + "\n".join(diffs) + "\n"
            f"Reinstall with: pip install -e {REPO_ROOT / 'python_src'}"
        )
        raise pytest.UsageError(msg)
