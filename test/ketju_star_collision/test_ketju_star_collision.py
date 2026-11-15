"""KETJU stellar collisions: Type-4 stars inside a region merge at R1 + R2, found inside MSTAR (a contact, or a pericentre
passed since the last accepted step whose two-body pericentre is below R1 + R2).

Three cases (make_ketju_star_collision_ics.py): head-on and grazing (pericentre 0.8 (R1 + R2)) must merge, a near miss
(1.25 (R1 + R2)) must not. Asserted: merge iff expected, total mass and momentum conserved, the product has the pair's
mass and its CoM velocity (the grazing pair's CoM is pulled by its bound third star, hence the tolerance)."""

import glob
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, build_gizmo_for_test, clean_test_outputs, mpi_launch_command

TEST_NAME = "ketju_star_collision"
TEST_DIR = f"test/{TEST_NAME}"


def _load(f):
    with h5py.File(f, "r") as h:
        g = h["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        return (np.array(g["ParticleIDs"])[o], np.array(g["Masses"])[o], np.array(g["Velocities"])[o])


def _check_merger_table(path, m0):
    """ketju_output.hdf5 "mergers": one row per collision (survivor ID 1, merged ID 2, masses), none for a miss"""
    with h5py.File(path, "r") as h:
        t = h["mergers"][()]
    if m0 is None:
        assert len(t) == 0
        return
    assert len(t) == 1 and t["ID1"][0] == 1 and t["ID2"][0] == 2 and t["is_collision"][0] == 1
    assert abs(t["M1"][0] / m0[0] - 1) < 1e-12 and abs(t["M2"][0] / m0[1] - 1) < 1e-12
    assert abs(t["M_remnant"][0] / (m0[0] + m0[1]) - 1) < 1e-12


@pytest.mark.parametrize("case,expect_merge", [("headon", True), ("graze", True), ("miss", False)])
def test_ketju_star_collision(case, expect_merge):
    subprocess.run([sys.executable, "make_ketju_star_collision_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    ic = "ketju_star_collision_ics" if case == "headon" else f"ketju_star_collision_{case}_ics"
    build_and_run_test(TEST_NAME, 1, 1, param_overrides={"InitCondFile": ic})

    snaps = sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))
    assert len(snaps) >= 2, "the run wrote no snapshots"
    i0, m0, v0 = _load(snaps[0]); i1, m1, v1 = _load(snaps[-1])
    log = open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()

    assert ("KETJU STAR COLLISION" in log) == expect_merge
    assert abs(m1.sum() / m0.sum() - 1) < 1e-12, "mass not conserved"
    p0 = (m0[:, None] * v0).sum(0); p1 = (m1[:, None] * v1).sum(0); scale = (m0 * np.linalg.norm(v0, axis=1)).sum()
    assert np.linalg.norm(p1 - p0) / scale < 1e-9, "momentum not conserved"
    if expect_merge:
        assert len(i1) == len(i0) - 1 and 1 in i1 and 2 not in i1, "the heavier star (ID 1) must survive, ID 2 go"
        k = list(i1).index(1); M = m0[0] + m0[1]
        assert abs(m1[k] / M - 1) < 1e-12
        vcom = (m0[0] * v0[0] + m0[1] * v0[1]) / M
        assert np.linalg.norm(v1[k] - vcom) < 0.05 * np.linalg.norm(v0[0] - v0[1])
    else:
        assert len(i1) == len(i0)
    _check_merger_table(f"{TEST_DIR}/output/ketju_output.hdf5", m0 if expect_merge else None)


@pytest.mark.parametrize("nranks", [1, 2])
def test_ketju_star_collision_snapshot_start(nranks):
    """bar (written before the run): a snapshot start (RestartFlag 2) into an empty output directory creates
    ketju_output.hdf5, fills its timesteps, and the head-on collision's row reaches rank 0 whichever rank integrates it"""
    subprocess.run([sys.executable, "make_ketju_star_collision_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME, 1)
    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    with open(f"{TEST_DIR}/test_{TEST_NAME}_snapshot_start.out", "w") as out:
        subprocess.run(mpi_launch_command(nranks) + ["./GIZMO", f"{TEST_NAME}.params", "2"], cwd=TEST_DIR, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    with h5py.File(f"{TEST_DIR}/output/ketju_output.hdf5", "r") as h:
        assert len(h["timesteps"]) > 0
    _check_merger_table(f"{TEST_DIR}/output/ketju_output.hdf5", _load(f"{TEST_DIR}/ketju_star_collision_ics.hdf5")[1])


