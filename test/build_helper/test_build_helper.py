"""build_helper: the test build helper and the tracked test configurations; no GIZMO build.

.gitignore ignores Config.sh (the build file in src/), so a test's Config.sh is tracked only when added with git add -f. Without
it the helper copied nothing and built whatever src/Config.sh held."""

import os
import re
import subprocess
from pathlib import Path

import pytest

from gizmo import test as gt

ROOT = Path(__file__).resolve().parent.parent.parent
BUILDS = re.compile(r"(?:build_gizmo_for_test|build_and_run_test)\(\s*([^,)]+)")


def test_missing_test_config_stops_the_build(tmp_path, monkeypatch):
    """bar: no make when test/<name>/Config.sh is missing, and the error names the file"""
    calls = []
    monkeypatch.setattr(gt, "system", lambda cmd: calls.append(cmd) or 0)
    monkeypatch.delenv("GIZMO_TEST_SKIP_BUILD_RUN", raising=False)
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "test" / "no_config_here")
    with pytest.raises(FileNotFoundError, match="Config.sh"):
        gt.build_gizmo_for_test("no_config_here")
    assert not any("make" in c for c in calls), f"make ran without the test's Config.sh: {calls}"


def _flag_value(text, name):
    m = re.search(rf"^\s*{name}\s*=\s*([0-9()+|\s]+?)\s*(?:#.*)?$", text, re.M)
    return None if m is None else eval(m.group(1), {"__builtins__": {}})  # digits, +, |, brackets only


def test_no_adaptive_star_softening_in_single_star_tests():
    """bar: no test or Stella script config with single stars (STELLA_DEFAULTS or GALSF_SINGLE_STARS_*) gives stars (Type 4,
    bit 16) adaptive softening; Stella runs use fixed star softening"""
    files = sorted((ROOT / "test").glob("*/Config.sh")) + sorted((ROOT / "scripts" / "stella").glob("Config_stella_*.sh"))
    bad = []
    for f in files:
        text = f.read_text(errors="ignore")
        if not re.search(r"^\s*(STELLA_DEFAULTS|GALSF_SINGLE_STARS_\w+)\b", text, re.M):
            continue
        v = _flag_value(text, "ADAPTIVE_GRAVSOFT_FORALL")
        if v is not None and v & 16:
            bad.append(str(f.relative_to(ROOT)))
    assert not bad, f"single-star configs with adaptive star softening: {bad}"


def test_every_building_test_has_a_tracked_config():
    """bar: every test directory whose tests build through the helpers with their own name has its Config.sh in git (a build
    named after another test's module, mod.TEST_NAME, uses that test's file)"""
    tracked = set(subprocess.run(["git", "ls-files", "test/*/Config.sh"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split())
    missing = []
    for d in sorted(p for p in (ROOT / "test").iterdir() if p.is_dir() and p != Path(__file__).resolve().parent):
        args = [a.strip() for f in d.glob("test_*.py") for a in BUILDS.findall(f.read_text(errors="ignore"))]
        own = [a for a in args if not re.fullmatch(r"\w+\.TEST_NAME", a) and "/" not in a]
        if own and f"test/{d.name}/Config.sh" not in tracked:
            missing.append(d.name)
    assert not missing, f"tests that build without a tracked Config.sh (add it with git add -f): {missing}"
