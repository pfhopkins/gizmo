"""KETJU stellar collisions: Type-4 stars inside a region merge at R1 + R2, found inside MSTAR (a contact, or a pericentre
passed since the last accepted step whose two-body pericentre is below R1 + R2).

Three cases (make_ketju_star_collision_ics.py): head-on and grazing (pericentre 0.8 (R1 + R2)) must merge, a near miss
(1.25 (R1 + R2)) must not. Asserted: merge iff expected, total mass and momentum conserved, the product has the pair's
mass and its CoM velocity (the grazing pair's CoM is pulled by its bound third star, hence the tolerance)."""

import glob
import subprocess
import sys

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, clean_test_outputs

TEST_NAME = "ketju_star_collision"
TEST_DIR = f"test/{TEST_NAME}"


def _load(f):
    with h5py.File(f, "r") as h:
        g = h["PartType4"]; o = np.argsort(np.array(g["ParticleIDs"]))
        return (np.array(g["ParticleIDs"])[o], np.array(g["Masses"])[o], np.array(g["Velocities"])[o])


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


def test_ketju_star_collision_energy():
    """KETJU ENERGY ledger: the head-on collision removes the pair's internal energy, so the direct-sum total energy
    changes by exactly the booked collision energy; MSTAR's own error stays at round-off."""
    subprocess.run([sys.executable, "make_ketju_star_collision_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1)
    lines = [l for l in open(f"{TEST_DIR}/test_{TEST_NAME}.out") if l.startswith("KETJU ENERGY")]
    assert lines, "no KETJU ENERGY line"
    f = dict(kv.split("=") for kv in lines[-1].split() if "=" in kv and not kv.startswith("t="))
    de_coll, de_tot, de_mstar = float(f["collisions-dE"]), float(f["dE"]), float(f["MSTAR-dE"])
    print(f"collisions {de_coll:.6e} ETOT {de_tot:.6e} MSTAR {de_mstar:.2e} (units |E0|)")
    assert de_coll < 0 and abs(de_tot - de_coll) < 1e-3 * abs(de_coll) and abs(de_mstar) < 1e-8


def test_ketju_star_collision_triple():
    """No false collision in a hierarchical triple: a perturbed wide pair whose r.v changes sign with a nearly radial relative
    velocity has an osculating pericentre below R1 + R2 that lies far in the past; it must not count as a passage."""
    subprocess.run([sys.executable, "make_ketju_star_collision_ics.py"], cwd=TEST_DIR, check=True)
    clean_test_outputs(TEST_NAME)
    build_and_run_test(TEST_NAME, 1, 1, param_overrides={
        "InitCondFile": "ketju_star_collision_triple_ics", "TimeMax": 2e-3, "TimeBetSnapshot": 2e-4, "TimeBetStatistics": 2e-4,
        "MaxSizeTimestep": 1.25e-7, "SofteningStars": 3.117e-8, "KetjuRegionRadius": 1e-5, "KetjuRegionMotionCapFactor": 0,
        "KetjuKeplerPairs": 1, "KetjuFlybyStepEta": 0.2})
    log = open(f"{TEST_DIR}/test_{TEST_NAME}.out").read()
    assert "KETJU STAR COLLISION" not in log
    lines = [l for l in log.splitlines() if l.startswith("KETJU ENERGY")]
    de_tot = float(dict(kv.split("=") for kv in lines[-1].split() if "=" in kv and not kv.startswith("t="))["dE"])
    assert abs(de_tot) < 1e-6
