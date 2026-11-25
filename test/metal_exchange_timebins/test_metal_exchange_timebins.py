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
COUNT = "TEST_METALS_PAIRWISE_COUNT_EVALUATIONS"  # TD_PAIRCOUNT line per hydro pass
MFM = "HYDRO_MESHLESS_FINITE_MASS"
MFV = "HYDRO_MESHLESS_FINITE_VOLUME"  # Config.sh names no solver: each case does
_built = None


def run_case(tag, ranks, flags, final_only=False, params=None):
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
    for key, val in (params or {}).items():
        assert any(l.split()[:1] == [key] for l in lines), key
        lines = [(f"{key} {val}" if l.split()[:1] == [key] else l) for l in lines]
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


def pair_counts(logfile):
    """Sums of the TD_PAIRCOUNT fields over the run: exchange evaluations, cells with more evaluations booked than pairs counted
    in the gradient pass (the debit bound divides by that count), cells floored at zero, pairs the bound scaled."""
    tot = {"evals": 0.0, "cells_over_count": 0.0, "floor_cells": 0.0, "scaled": 0.0}
    n = 0
    for line in open(logfile):
        if line.startswith("TD_PAIRCOUNT"):
            w = line.split(); n += 1
            for key in tot:
                tot[key] += float(w[w.index(key + "=") + 1])
    assert n > 0, "no TD_PAIRCOUNT lines"
    return tot


def worst_drift(snaps):
    MZ0 = metal_masses(snaps[0])[0]
    drift = max(np.abs(metal_masses(s)[0] - MZ0).max() / MZ0.min() for s in snaps[1:])
    zmin = min(metal_masses(s)[1] for s in snaps)
    return drift, zmin


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_mfm_closes_at_every_snapshot(ranks):
    snaps = run_case(f"mfm_{ranks}", ranks, (MFM, EXACT, COUNT))
    drift, zmin = worst_drift(snaps)
    log = os.path.dirname(snaps[1]) + "/../gizmo.out"
    c = pair_counts(log)
    print(f"mfm {ranks} ranks: drift {drift:.3e}, min Z {zmin:.3e}, {c}")
    assert len(snaps) >= 5
    assert timebins(log) >= 5
    assert drift < 1e-12, f"metal mass drift {drift:.3e}"
    assert zmin >= 0
    assert c["evals"] > 0
    assert c["cells_over_count"] == 0, "a cell had more exchange evaluations than counted pairs: the debit bound is not strict"
    assert c["floor_cells"] == 0, "the floor at zero fired"


@pytest.mark.parametrize("ranks", [1, 4])
def test_mfm_debit_bound_binds(ranks):
    """Diffusion 1e4 times stronger (two species step into exactly zero): the debit bound must scale pairs, the floor must not
    fire, and the metal mass still closes."""
    snaps = run_case(f"mfm_strong_{ranks}", ranks, (MFM, EXACT, COUNT), final_only=True,
                     params={"TurbDiffusionCoefficient": "1.0e4", "TimeMax": "0.003125"})
    drift, zmin = worst_drift([snaps[0], snaps[-1]])
    c = pair_counts(os.path.dirname(snaps[1]) + "/../gizmo.out")
    print(f"mfm strong {ranks} ranks: drift {drift:.3e}, min Z {zmin:.3e}, {c}")
    assert c["scaled"] > 0, "the debit bound never bound: the case does not test it"
    assert c["cells_over_count"] == 0
    assert c["floor_cells"] == 0, "the floor at zero fired"
    assert drift < 1e-12, f"metal mass drift {drift:.3e}"
    assert zmin >= 0


@pytest.mark.parametrize("loworder", [False, True])
def test_mfv_closes_at_the_synchronised_end(loworder):
    flags = (MFV, EXACT) + (("TURB_DIFF_METALS_LOWORDER",) if loworder else ())
    flags = flags + (COUNT,)
    snaps = run_case(f"mfv_lo{int(loworder)}", 4, flags, final_only=True)
    drift, zmin = worst_drift([snaps[0], snaps[-1]])
    c = pair_counts(os.path.dirname(snaps[1]) + "/../gizmo.out")  # the bound covers the diffusive debits, not the advected metal: floor only reported
    print(f"mfv loworder {loworder}: drift {drift:.3e}, min Z {zmin:.3e}, {c}")
    assert c["cells_over_count"] == 0
    assert drift < 1e-12, f"metal mass drift {drift:.3e}"
    assert zmin >= 0


def test_default_exchange_drifts():
    snaps = run_case("mfm_default", 4, (MFM,))
    drift, _ = worst_drift(snaps)
    assert drift > 1e-9, f"default exchange drift {drift:.3e}: the check no longer separates the two paths"
