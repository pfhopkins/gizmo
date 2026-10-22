"""Merge/split conservation and memory safety (mesh/merge_split.cc).

Links merge_split_driver.cc against the build and runs split_particle_i, merge_particles_ij and the SN ejecta spawn on
hand-built cells: every extensive quantity conserved to 1e-13, no CellP write past MaxPartGas.
merge_split_conservation_starforge runs the same checks under STARFORGE with SINK_THERMALFEEDBACK.
"""

import os
import shlex
import subprocess
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, get_cooling_tables

DRIVER_SRC = join(dirname(abspath(__file__)), "merge_split_driver.cc")
REPO = abspath(join(dirname(abspath(__file__)), "..", ".."))


def _link_driver(test_dir):
    """Link the driver against the objects of the build that just finished."""
    build_dir = join(test_dir, "output")
    os.makedirs(build_dir, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(build_dir, "merge_split_driver.o"), join(build_dir, "merge_split_driver")
    cc = shlex.split(compile_main[-1])
    cc[cc.index("core/main.cc")] = DRIVER_SRC
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + REPO)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])


def run_merge_split_checks(test_name, cooling_tables=False):
    """Build test_name, run the driver, fail on any failed check. cooling_tables: non-CHEMCOOL cooling needs its tables."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    test_dir = join(REPO, "test", test_name)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(test_name, post_build=lambda: _link_driver(test_dir))
    finally:
        os.chdir(cwd)
    if cooling_tables:
        get_cooling_tables(test_dir)
    r = subprocess.run([join(test_dir, "output", "merge_split_driver"), f"{test_name}.params"],
                       capture_output=True, text=True, cwd=test_dir)
    lines = r.stdout.splitlines()
    with open(join(test_dir, "output", "checks.txt"), "w") as f:
        f.write(r.stdout)
    done = [l for l in lines if l.startswith("DONE")]
    assert done, "driver did not finish (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "%d of %d merge/split checks failed:\n%s" % (len(failed), sum(l[:4] in ("PASS", "FAIL") for l in lines), "\n".join(failed))


def test_merge_split_conservation():
    run_merge_split_checks("merge_split_conservation")
