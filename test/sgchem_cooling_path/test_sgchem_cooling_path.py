"""SGCHEM through the standard cooling path (cooling/cooling.cc, Stella Config without COOLING_OPERATOR_SPLIT, with
SINK_THERMALFEEDBACK). The driver builds a real cell with begrun() on a small Stella box and checks that the cooling path
deposits sink thermal energy, applies unsplit hydro heating, and returns the chemistry's electron abundance.
With SGCHEM_LEGACY_COOLING all three fail: the legacy path bypasses them.
sgchem_cooling_path_net15 runs the same checks with CHEMISTRYNETWORK=15.
"""

import os
import shlex
import subprocess
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test

DRIVER_DIR = dirname(abspath(__file__))  # driver and IC generator, shared by the variants
REPO = abspath(join(DRIVER_DIR, "..", ".."))
DRIVER = "sgchem_cooling_path"


def _link_driver(test_dir):
    """Link the driver against the objects of the build that just finished, in place of core/main.o."""
    out = join(test_dir, "output")
    os.makedirs(out, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(out, f"{DRIVER}_driver.o"), join(out, f"{DRIVER}_driver")
    cc = shlex.split(compile_main[-1])
    cc[cc.index("core/main.cc")] = join(DRIVER_DIR, f"{DRIVER}_driver.cc")
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + REPO)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])


def run_cooling_path_checks(test_name):
    """Build test_name (its Config.sh and params), run the shared driver on a small Stella box, fail on any failed check."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    test_dir = join(REPO, "test", test_name)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(test_name, post_build=lambda: _link_driver(test_dir))
    finally:
        os.chdir(cwd)
    # outside the test tree: the gizmo/ link to the repo root would otherwise make pytest collect the whole suite
    run = tempfile.mkdtemp(prefix=f"{test_name}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))  # SGCHEM and the stellar tables open their data under gizmo/
    subprocess.run(["python3", join(DRIVER_DIR, f"make_{DRIVER}_ics.py"), join(run, f"{DRIVER}_ics.hdf5")], check=True)
    r = subprocess.run(["mpirun", "-np", "1", join(test_dir, "output", f"{DRIVER}_driver"), join(test_dir, f"{test_name}.params")],
                       capture_output=True, text=True, cwd=run)
    with open(join(test_dir, "output", "checks.txt"), "w") as f:
        f.write(r.stdout)
    lines = r.stdout.splitlines()
    assert [l for l in lines if l.startswith("DONE")], "driver did not finish (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "%d cooling-path checks failed:\n%s" % (len(failed), "\n".join(failed))


def test_sgchem_cooling_path():
    run_cooling_path_checks("sgchem_cooling_path")
