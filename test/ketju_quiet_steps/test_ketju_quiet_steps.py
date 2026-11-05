"""No regions, no extra collectives: every MPI collective of a run with KETJU compiled in, counted from outside (LD_PRELOAD),
against the same run without KETJU.
- no star heavy enough to be a chain candidate: not one collective more;
- candidates but none close: at most two more per step, the gather of the candidates."""

import os
import re
import subprocess
import sys

from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "ketju_quiet_steps"
TEST_DIR = f"test/{TEST_NAME}"
RANKS = 2


def _run(extra_params=""):
    open("run.params", "w").write(open(f"{TEST_NAME}.params").read() + extra_params)
    env = dict(os.environ, LD_PRELOAD=os.path.abspath("mpi_collective_count.so"))
    out = subprocess.run(mpi_launch_command(RANKS, 1) + ["-x", "LD_PRELOAD", "./GIZMO", "run.params", "0"], capture_output=True, text=True, env=env).stdout
    m = re.search(r"MPI_COLLECTIVES total (\d+)", out)
    assert m, out[-500:]
    print(re.search(r"MPI_COLLECTIVES.*", out).group(0))
    return int(m.group(1)), len(re.findall(r"^Sync-Point", out, re.M))


def test_collectives_without_regions():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py"], cwd=TEST_DIR, check=True)
    subprocess.run(["mpicc", "-shared", "-fPIC", "-O1", "mpi_collective_count.c", "-o", "mpi_collective_count.so"], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME, 1)
    cwd = os.getcwd(); os.chdir(TEST_DIR)
    try:
        c0, s0 = _run()
    finally:
        os.chdir(cwd)
    build_gizmo_for_test(TEST_NAME, 1, ("KETJU_REGULARIZATION",))
    os.chdir(TEST_DIR)
    try:
        c1, s1 = _run("\nKetjuRegionRadius 5e-6\nKetjuMinStarMass 5e-9\nKetjuMinBHMass 0\nKetjuUseStarStarSoftening 0\nKetjuMaxStepCount 10000000\n")
        c2, s2 = _run("\nKetjuRegionRadius 5e-6\nKetjuMinStarMass 5e-11\nKetjuMinBHMass 0\nKetjuUseStarStarSoftening 0\nKetjuMaxStepCount 10000000\n")
    finally:
        os.chdir(cwd)
    print(f"collectives: no KETJU {c0} in {s0} steps; no candidates {c1} in {s1}; candidates {c2} in {s2}")
    assert s0 == s1 == s2
    assert c1 == c0
    assert c2 <= c0 + 2 * s2
