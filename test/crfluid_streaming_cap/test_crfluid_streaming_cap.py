"""CRFLUID_STREAMING_LOSS_CAPPED_AT_CR_SPEED: the Alfven speed in the CR streaming loss is at most the CR speed (default with single
stars, stella_dev cosmic_ray_utilities.cc:747). In a box with Alfven speed 3000 km/s and CR speed 300 km/s the CRs lose far less.

Bars (set before the first run): CR energy lost with the cap <= 0.5 x without, and within 5 % of stella_dev on the same ICs (stella_dev_reference.txt).
"""
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "crfluid_streaming_cap"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR))
from make_crfluid_streaming_cap_ics import make_crfluid_streaming_cap_ics  # noqa: E402


def lost(flags, label):
    build_gizmo_for_test(TEST_NAME, 0, flags)
    binary = TEST_DIR / f"GIZMO_{label}"
    shutil.move(str(TEST_DIR / "GIZMO"), binary)
    d = TEST_DIR / f"run_{label}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    make_crfluid_streaming_cap_ics(str(d / "output" / "snapshot_000.hdf5"))
    (d / "run.params").write_text((TEST_DIR / f"{TEST_NAME}.params").read_text())
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(binary), "run.params", "2", "0"], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
    files = sorted((d / "output").glob("snapshot_*.hdf5"))
    e = [h5py.File(f, "r")["PartType0/CosmicRayEnergy"][()].sum() for f in (files[0], files[-1])]
    return 1 - e[1] / e[0]


def test_crfluid_streaming_cap():
    ref = float((TEST_DIR / "stella_dev_reference.txt").read_text().split("\n")[1].split()[-1])
    capped = lost(("CRFLUID_STREAMING_LOSS_CAPPED_AT_CR_SPEED",), "cap")
    free = lost((), "nocap")
    print(f"fraction of the CR energy lost: capped {capped:.5f}, not capped {free:.5f}, stella_dev {ref:.5f}")
    assert capped <= 0.5 * free
    assert abs(capped / ref - 1) <= 0.05
