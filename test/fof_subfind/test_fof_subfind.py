import glob
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, default_mpi_ranks

from make_fof_subfind_ics import make_fof_subfind_ics


TEST_NAME = "fof_subfind"


def _clean_outputs(test_dir):
    for rel in ("output", "fof_subfind_ics.hdf5", "GIZMO"):
        path = test_dir / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for suffix in ("out", "err"):
        path = test_dir / f"test_fof_subfind.{suffix}"
        if path.exists():
            path.unlink()


def _run_fof_subfind(test_dir, num_mpi_ranks, num_omp_threads):
    make_fof_subfind_ics(test_dir)
    build_gizmo_for_test(TEST_NAME, num_omp_threads)

    env = os.environ.copy()
    if num_omp_threads > 0:
        env["OMP_NUM_THREADS"] = str(num_omp_threads)

    if env.get("GIZMO_TEST_USE_IBRUN"):
        cmd = ["ibrun", "./GIZMO", "fof_subfind.params", "3", "0"]
    else:
        cmd = [
            "mpirun",
            "-np",
            str(num_mpi_ranks),
            "--use-hwthread-cpus",
            "./GIZMO",
            "fof_subfind.params",
            "3",
            "0",
        ]
    try:
        with open(test_dir / "test_fof_subfind.out", "w") as stdout, open(test_dir / "test_fof_subfind.err", "w") as stderr:
            result = subprocess.run(cmd, cwd=test_dir, env=env, stdout=stdout, stderr=stderr, check=False, timeout=180)
    except subprocess.TimeoutExpired:
        stdout_text = (test_dir / "test_fof_subfind.out").read_text(errors="replace")
        stderr_text = (test_dir / "test_fof_subfind.err").read_text(errors="replace")
        pytest.fail("FOF/SUBFIND run timed out\n" + stdout_text + "\nSTDERR:\n" + stderr_text)

    stdout_text = (test_dir / "test_fof_subfind.out").read_text(errors="replace")
    stderr_text = (test_dir / "test_fof_subfind.err").read_text(errors="replace")
    assert result.returncode == 0, stdout_text + "\nSTDERR:\n" + stderr_text
    return stdout_text, stderr_text


def _read_catalogue(test_dir):
    files = sorted(glob.glob(str(test_dir / "output" / "groups_000" / "sub_000*.hdf5")))
    assert files, "SUBFIND did not write a sub_000 catalogue"

    group_lens = []
    group_nsubs = []
    sub_lens = []
    for fname in files:
        with h5py.File(fname, "r") as f:
            if "Group/GroupLen" in f:
                group_lens.append(f["Group/GroupLen"][:])
            if "Group/GroupNsubs" in f:
                group_nsubs.append(f["Group/GroupNsubs"][:])
            if "Subhalo/SubhaloLen" in f:
                sub_lens.append(f["Subhalo/SubhaloLen"][:])

    group_lens = np.concatenate(group_lens) if group_lens else np.array([], dtype=np.int64)
    group_nsubs = np.concatenate(group_nsubs) if group_nsubs else np.array([], dtype=np.int64)
    sub_lens = np.concatenate(sub_lens) if sub_lens else np.array([], dtype=np.int64)
    return group_lens, group_nsubs, sub_lens


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (0,))
def test_fof_subfind_modern_restart_path(num_mpi_ranks, num_omp_threads):
    test_dir = Path("test") / TEST_NAME
    _clean_outputs(test_dir)

    stdout_text, stderr_text = _run_fof_subfind(test_dir, num_mpi_ranks, num_omp_threads)
    combined = stdout_text + stderr_text

    assert "Modern FOF neighbor-list construction" in combined
    assert "Start modern find_linkngb" in combined
    assert "Start modern finding nearest two" in combined
    assert "Finished with SUBFIND" in combined

    group_lens, group_nsubs, sub_lens = _read_catalogue(test_dir)
    assert len(group_lens) >= 2
    assert group_lens.max() >= 200
    assert len(sub_lens) >= 2
    assert sub_lens.max() >= 40
    assert group_nsubs.max() >= 1
