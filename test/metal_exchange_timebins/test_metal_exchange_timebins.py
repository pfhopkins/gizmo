"""METALS_EXACT_PAIRWISE_EXCHANGE: metal mass closure with individual timesteps (temporary check, see README.md).

A cold | hot box (1e3 K | 1e7 K, equal density) puts neighbours in five or more timebins. Five species, one a step into
exactly zero and one a random field with exact zeros. Each species' metal mass, summed over the cells of a
double-precision snapshot, must stay at its initial value to 1e-12; with MFM at every snapshot (cell masses are fixed),
with MFV at the end of the run, the one snapshot written after the last synchronised kick. The default exchange, same
IC, must drift by more than 1e-9, so the check can tell the two apart.
"""

import math
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gizmo.test import build_gizmo_for_test, mpi_launch_command
from make_ics import make_ics

TEST_NAME = "metal_exchange_timebins"
TEST_DIR = f"test/{TEST_NAME}"
EXACT = "METALS_EXACT_PAIRWISE_EXCHANGE"
MFM = "HYDRO_MESHLESS_FINITE_MASS"
MFV = "HYDRO_MESHLESS_FINITE_VOLUME"  # Config.sh names no solver: each case does
_built = None


def run_case(tag, ranks, flags, final_only=False):
    """Builds (if the flags changed), writes the IC, runs from it as a snapshot; returns the snapshot files."""
    global _built
    if _built != flags or not os.path.isfile(f"{TEST_DIR}/GIZMO"):
        build_gizmo_for_test(TEST_NAME, 0, flags)
        _built = flags
    d = f"{TEST_DIR}/run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(f"{d}/output")
    make_ics(f"{d}/ics_000.hdf5")
    lines = open(f"{TEST_DIR}/{TEST_NAME}.params").read().splitlines()
    if final_only:
        lines = [("TimeBetSnapshot 1.0" if l.startswith("TimeBetSnapshot") else l) for l in lines]
    open(f"{d}/run.params", "w").write("\n".join(lines) + "\n")
    with open(f"{d}/gizmo.out", "w") as f:
        rc = subprocess.run(mpi_launch_command(ranks) + [os.path.abspath(f"{TEST_DIR}/GIZMO"), "run.params", "2"], cwd=d,
                            stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}).returncode
    assert rc == 0, f"GIZMO exited with {rc}, see {d}/gizmo.out"
    snaps = sorted(f"{d}/output/{s}" for s in os.listdir(f"{d}/output") if s.startswith("snapshot_"))
    return [f"{d}/ics_000.hdf5"] + snaps


def metal_masses(snapshot):
    """Per-species metal mass (exact sum of the double-precision products) and the smallest metallicity."""
    with h5py.File(snapshot, "r") as f:
        m, Z = f["PartType0/Masses"][:], f["PartType0/Metallicity"][:]
    return np.array([math.fsum(m * Z[:, k]) for k in range(Z.shape[1])]), Z.min()


def timebins(logfile):
    """Most timebins holding cells at once, from the 'Occupied timebins' tables."""
    n, best = None, 0
    for line in open(logfile):
        if line.startswith("Occupied timebins"):
            n = 0
        elif n is not None and " bin=" in line:
            n += int(line.split("bin=")[1].split()[2]) > 0
            best = max(best, n)
        else:
            n = None
    return best


def worst_drift(snaps):
    MZ0 = metal_masses(snaps[0])[0]
    drift = max(np.abs(metal_masses(s)[0] - MZ0).max() / MZ0.min() for s in snaps[1:])
    zmin = min(metal_masses(s)[1] for s in snaps)
    return drift, zmin


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_mfm_closes_at_every_snapshot(ranks):
    snaps = run_case(f"mfm_{ranks}", ranks, (MFM, EXACT))
    drift, zmin = worst_drift(snaps)
    assert len(snaps) >= 5
    assert timebins(os.path.dirname(snaps[1]) + "/../gizmo.out") >= 5
    assert drift < 1e-12, f"metal mass drift {drift:.3e}"
    assert zmin >= 0


@pytest.mark.parametrize("loworder", [False, True])
def test_mfv_closes_at_the_synchronised_end(loworder):
    flags = (MFV, EXACT) + (("TURB_DIFF_METALS_LOWORDER",) if loworder else ())
    snaps = run_case(f"mfv_lo{int(loworder)}", 4, flags, final_only=True)
    drift, zmin = worst_drift([snaps[0], snaps[-1]])
    assert drift < 1e-12, f"metal mass drift {drift:.3e}"
    assert zmin >= 0


def test_default_exchange_drifts():
    snaps = run_case("mfm_default", 4, (MFM,))
    drift, _ = worst_drift(snaps)
    assert drift > 1e-9, f"default exchange drift {drift:.3e}: the check no longer separates the two paths"
