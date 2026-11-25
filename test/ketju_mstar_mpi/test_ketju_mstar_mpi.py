"""ketju_mstar_mpi: KETJU_MSTAR_MPI (a region integrated by all its compute tasks together) against the unchanged code
(the region integrated by one task) on 4 MPI ranks.

256 circular binaries in a Plummer sphere (plummer_binaries setup); KetjuRegionRadius 0.1 pc merges neighbouring binaries
into regions of 4+ members and KetjuMinParticlesPerTask 2 gives them several compute tasks. The variant lowers the
sharing threshold to 4 bodies (KETJU_MSTAR_MPI_MIN_N=4) so these regions are shared. See README.md.
  - control: the unchanged code twice, bitwise equal (else nothing below can be judged);
  - KETJU_MSTAR_MPI: every snapshot bitwise equal to the unchanged code, and the log shows shared integrations.
Needs a KETJU install built with MSTAR_ENABLE_MPI at KETJU_PATH (its ketju.stamp says so); skipped otherwise.
"""
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from os import path

import pytest

from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "ketju_mstar_mpi"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
NUM_MPI_RANKS = 4
N_BINARIES = 256
TIMING = ("TEST_KETJU_MSTAR_TIMING=4",)  # one log line per MSTAR call of a region with 4+ bodies
MPI = ("KETJU_MSTAR_MPI", "KETJU_MSTAR_MPI_MIN_N=4") + TIMING
CONTROL_DIR = f"{TEST_DIR}/output_control"
TIMING_LINE = re.compile(r"KETJU MSTAR TIMING: n (\d+) comm (\d+) sharing (\d+) ")


def _mpi_install():
    stamp = path.join(os.environ.get("KETJU_PATH", "/nonexistent"), "ketju.stamp")
    return path.isfile(stamp) and "MSTAR_ENABLE_MPI" in open(stamp).read()


def _ensure_ic():
    if path.isfile(IC_FILE):
        return
    spec = importlib.util.spec_from_file_location("make_plummer_binaries_ics", "test/plummer_binaries/make_plummer_binaries_ics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_plummer_binaries_ics(N_binaries=N_BINARIES, m_star=1.0, a_cluster=1.0, binary_separation_au=1000.0,
                                  boxsize=300.0, seed=42, outfile=IC_FILE)


def _snapshots(d):
    snaps = sorted(glob.glob(d + "/snapshot_*.hdf5"))
    assert len(snaps) >= 2, f"no snapshots in {d}"
    return snaps


def _compare(da, db, log, label):
    a, b = _snapshots(da), _snapshots(db)
    assert [path.basename(s) for s in a] == [path.basename(s) for s in b], f"{label}: different snapshot sets"
    res = subprocess.run([sys.executable, f"{TEST_DIR}/compare_snaps.py"] + [x for ab in zip(a, b) for x in ab], capture_output=True, text=True)
    log.write(f"== {label}\n{res.stdout}{res.stderr}")
    return res.returncode == 0, res.stdout


def _calls(tag):
    """(n, comm size, sharing tasks) of every logged MSTAR call of the run just made; its log is kept as <tag>.out."""
    out = f"{TEST_DIR}/test_{TEST_NAME}.out"
    shutil.copy(out, f"{TEST_DIR}/{TEST_NAME}_{tag}.out")
    return [tuple(map(int, m)) for m in TIMING_LINE.findall(open(out).read())]


@pytest.mark.skipif(not _mpi_install(), reason="KETJU_PATH is not an MSTAR install built with MSTAR_ENABLE_MPI")
def test_ketju_mstar_mpi():
    _ensure_ic()
    build_and_run_test(TEST_NAME, NUM_MPI_RANKS, 1, TIMING)
    shutil.rmtree(CONTROL_DIR, ignore_errors=True)
    shutil.move(variant_output_dir(TEST_NAME, TIMING), CONTROL_DIR)
    build_and_run_test(TEST_NAME, NUM_MPI_RANKS, 1, TIMING)
    off = _calls("off")
    assert any(c > 1 for n, c, s in off), "no region got several compute tasks: nothing to share"
    assert all(s == 1 for n, c, s in off), "the unchanged code shared an integration"
    build_and_run_test(TEST_NAME, NUM_MPI_RANKS, 1, MPI)
    on = _calls("on")
    shared = [(n, c, s) for n, c, s in on if s > 1]
    assert shared and all(s == c for n, c, s in shared), f"KETJU_MSTAR_MPI: {len(shared)} shared calls of {len(on)}"
    assert all(s > 1 for n, c, s in on if c > 1), "a region with several compute tasks was not shared"
    base = variant_output_dir(TEST_NAME, TIMING)
    with open(f"{TEST_DIR}/{TEST_NAME}_compare.txt", "w") as log:
        log.write(f"MSTAR calls logged (rank 0 of the region's tasks, 4+ bodies): unchanged {len(off)}, with several compute tasks "
                  f"{sum(c > 1 for n, c, s in off)}; KETJU_MSTAR_MPI {len(on)}, shared {len(shared)}, largest shared region "
                  f"{max(n for n, c, s in shared)} bodies on {max(s for n, c, s in shared)} tasks\n")
        ok, txt = _compare(CONTROL_DIR, base, log, "control: unchanged twice")
        assert ok, f"the unchanged code is not reproducible, nothing can be judged:\n{txt[-2000:]}"
        ok, txt = _compare(base, variant_output_dir(TEST_NAME, MPI), log, "KETJU_MSTAR_MPI vs unchanged")
        assert ok, f"KETJU_MSTAR_MPI changed the results:\n{txt[-3000:]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
