"""ketju_task_schedule: the KETJU scheduling flags give bitwise the unchanged code's results on several MPI ranks.

64 circular binaries in a Plummer sphere (the plummer_binaries setup), each a 2-member chain region, on 8 ranks. The
unchanged code and each flag variant run to t = 0.648; every snapshot must agree in every dataset, bit for bit. See README.md.
"""
import glob
import importlib.util
import subprocess
import sys
from os import path

import pytest

from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "ketju_task_schedule"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
NUM_MPI_RANKS = 8
N_BINARIES = 64
VARIANTS = [(), ("KETJU_SCATTER_AFTER_ALL_REGIONS",), ("KETJU_ONE_TASK_PER_REGION",),
            ("KETJU_SCATTER_AFTER_ALL_REGIONS", "KETJU_ONE_TASK_PER_REGION")]


def _ensure_ic():
    if path.isfile(IC_FILE):
        return
    spec = importlib.util.spec_from_file_location("make_plummer_binaries_ics", "test/plummer_binaries/make_plummer_binaries_ics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_plummer_binaries_ics(N_binaries=N_BINARIES, m_star=1.0, a_cluster=1.0, binary_separation_au=1000.0,
                                  boxsize=300.0, seed=42, outfile=IC_FILE)


def _snapshots(flags):
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, flags) + "/snapshot_*.hdf5"))
    assert len(snaps) >= 2, f"no snapshots for variant {flags}"
    return snaps


def test_ketju_task_schedule():
    _ensure_ic()
    for flags in VARIANTS:
        build_and_run_test(TEST_NAME, NUM_MPI_RANKS, 1, flags)
    ref = _snapshots(())
    with open(f"{TEST_DIR}/{TEST_NAME}_compare.txt", "w") as log:
        for flags in VARIANTS[1:]:
            snaps = _snapshots(flags)
            assert [path.basename(s) for s in snaps] == [path.basename(s) for s in ref], f"{flags}: different snapshot set"
            pairs = [x for ab in zip(ref, snaps) for x in ab]
            res = subprocess.run([sys.executable, f"{TEST_DIR}/compare_snaps.py"] + pairs, capture_output=True, text=True)
            log.write(f"== {' '.join(flags)}\n{res.stdout}{res.stderr}")
            assert res.returncode == 0, f"{flags} differs from the unchanged code:\n{res.stdout[-3000:]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
