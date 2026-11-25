"""The face matrix of the adaptive-softening kernel loop, P[].NV_T, must be the inverse of the neighbour moment
matrix sum_j W_ij dx dx^T: the collisionless-fluid (CBE_INTEGRATOR) and fuzzy-DM modules build their effective
faces and gradients from it."""

import os
import shutil
import subprocess
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver

TEST_NAME = "ags_face_matrix"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
DRIVER = join(TEST_DIR, "output", f"{TEST_NAME}_driver")
N_SIDE = 8


def make_ics(filename):
    """Jittered lattice of type-1 particles in a periodic unit box. The jitter matters: on a regular lattice
    the off-diagonal moments vanish and the test could not see an error in them."""
    g = (np.arange(N_SIDE) + 0.5) / N_SIDE
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    x += np.random.default_rng(7).uniform(-0.3, 0.3, x.shape) / N_SIDE
    n = len(x)
    with h5py.File(filename, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total"] = np.array([0, n, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["BoxSize"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1
        p = f.create_group("PartType1")
        p["Coordinates"] = x
        p["Velocities"] = np.zeros_like(x)
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint32)
        p["Masses"] = np.full(n, 1.0 / n)
    return n


def test_ags_face_matrix():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_test_driver(join(TEST_DIR, f"{TEST_NAME}_driver.cc"), DRIVER))
    finally:
        os.chdir(cwd)
    n = make_ics(join(TEST_DIR, f"{TEST_NAME}_ics.hdf5"))
    shutil.rmtree(join(TEST_DIR, "output", "run"), ignore_errors=True)
    os.makedirs(join(TEST_DIR, "output", "run"))
    r = subprocess.run([DRIVER, f"{TEST_NAME}.params"], capture_output=True, text=True, cwd=TEST_DIR)
    rows = np.array([l.split()[2:] for l in r.stdout.splitlines() if l.startswith("NVT ")], dtype=float)
    assert len(rows) == n, "driver printed %d of %d particles (exit %d):\n%s" % (len(rows), n, r.returncode, (r.stdout + r.stderr)[-3000:])

    stored = rows[:, :9].reshape(-1, 3, 3)
    xx, xy, xz, yy, yz, zz = rows[:, 9:].T
    moment = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]).transpose(2, 0, 1)
    offdiag = np.abs(np.array([xy, xz, yz])).max(axis=0) / (xx + yy + zz)
    assert np.median(offdiag) > 0.01, "particle distribution too regular to test the off-diagonal moments"

    resid = np.abs(stored @ moment - np.eye(3)).max(axis=(1, 2))
    print(f"[{TEST_NAME}] max |NV_T.M - 1| = {resid.max():.3g}, median = {np.median(resid):.3g}, "
          f"median off-diagonal/trace of M = {np.median(offdiag):.3g}")
    assert resid.max() < 1e-10, f"P[].NV_T is not the inverse of the neighbour moment matrix: max |NV_T.M - 1| = {resid.max():.3g}"
