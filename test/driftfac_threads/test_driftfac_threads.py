"""Thread safety of the comoving drift and gravity-kick factors (core/driftfac.cc).

Links driftfac_threads_driver.cc against the build and calls get_drift_factor/get_gravkick_factor from up to 8
threads at once, cycling through a few intervals, and requires every result to equal the serial value for its
interval. Function-scope memo statics shared across threads return one interval's factor for another's.
"""

import os
import subprocess
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver

TEST_NAME = "driftfac_threads"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXE = join(TEST_DIR, "output", "driftfac_threads_driver")
MAX_THREADS = 8


def test_driftfac_threads():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    nthreads = min(MAX_THREADS, os.cpu_count() or 1)
    if nthreads < 2:
        pytest.skip("needs at least 2 cores")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, num_openmp_threads=nthreads,
                             post_build=lambda: link_test_driver(join(TEST_DIR, "driftfac_threads_driver.cc"), EXE))
    finally:
        os.chdir(cwd)
    env = dict(os.environ, OMP_NUM_THREADS=str(nthreads))
    r = subprocess.run([EXE], capture_output=True, text=True, cwd=TEST_DIR, env=env, timeout=600)
    lines = r.stdout.splitlines()
    assert any(l.startswith("DONE") for l in lines), f"driver did not finish (exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "\n".join(failed)
