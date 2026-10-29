"""Curvilinear fixed mesh motion (MFV, HYDRO_FIX_MESH_MOTION=2 and 3).

A cold, static, uniform ball of gas (radius 0.4, centred on the origin, no box) whose mesh-generating points carry fixed
velocities from the ICs (ParticleVelocities). The fluid is at rest and nearly pressureless, so the check is pure kinematics
of the mesh points, which the code moves by a rigid turn of each point's direction from the centre in the plane of its
radius and tangential velocity, through the angle that velocity sweeps:

  cylindrical (2): v_mesh = omega z x r + v_z z: the cylindrical radius is kept, the azimuth advances by omega t and z by
                   v_z t, exactly, whatever the steps;
  spherical (3):   v_mesh = omega z x r: each point moves along the great circle set by its radius and velocity, with
                   constant speed, so it turns about the axis r x v by the angle |v| t / |r|.

Before mesh motion 2/3 moved the points, both came back unchanged."""

import glob
import numpy as np
import h5py
import pytest
from os import path
from gizmo.test import build_and_run_test, variant_output_dir

TEST_NAME = "mesh_motion_curvilinear"
TEST_DIR = path.join("test", TEST_NAME)
OMEGA, V_Z, T_END, R_BALL = 1.0, 0.1, 0.5, 0.4
TOL = 1e-9  # double-precision kinematics over ~100 steps


def write_ics(mode, n=20):
    g = (np.arange(n) + 0.5) / n * 2 * R_BALL - R_BALL
    x = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    x = x[np.sum(x**2, axis=1) < R_BALL**2]
    w = OMEGA * np.column_stack([-x[:, 1], x[:, 0], np.zeros(len(x))])
    if mode == 2:
        w[:, 2] = V_Z
    m = np.full(len(x), (2 * R_BALL / n) ** 3)  # unit density
    with h5py.File(path.join(TEST_DIR, f"{TEST_NAME}_ics.hdf5"), "w") as f:
        h = f.create_group("Header")
        npart = np.array([len(x), 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["BoxSize"] = 4.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        p = f.create_group("PartType0")
        p.create_dataset("Coordinates", data=x)
        p.create_dataset("Velocities", data=np.zeros_like(x))
        p.create_dataset("ParticleVelocities", data=w)
        p.create_dataset("Masses", data=m)
        p.create_dataset("ParticleIDs", data=np.arange(1, len(x) + 1, dtype=np.uint32))
        p.create_dataset("InternalEnergy", data=np.full(len(x), 1e-8))


def expected(mode, x0, w0, t):
    if mode == 2:
        c, s = np.cos(OMEGA * t), np.sin(OMEGA * t)
        return np.column_stack([c * x0[:, 0] - s * x0[:, 1], s * x0[:, 0] + c * x0[:, 1], x0[:, 2] + V_Z * t])
    r, v = np.linalg.norm(x0, axis=1), np.linalg.norm(w0, axis=1)
    n = np.cross(x0, w0)
    nn = np.linalg.norm(n, axis=1)
    k = n / np.where(nn > 0, nn, 1)[:, None]
    a = np.where(r > 0, v * t / np.where(r > 0, r, 1), 0)
    c, s = np.cos(a)[:, None], np.sin(a)[:, None]
    return x0 * c + np.cross(k, x0) * s + k * np.sum(k * x0, axis=1)[:, None] * (1 - c)  # Rodrigues


@pytest.mark.parametrize("mode", (2, 3), ids=("cylindrical", "spherical"))
def test_mesh_motion_curvilinear(mode):
    flags = (f"HYDRO_FIX_MESH_MOTION={mode}",)
    write_ics(mode)
    build_and_run_test(TEST_NAME, 4, 0, flags)
    snaps = sorted(glob.glob(f"{variant_output_dir(TEST_NAME, flags)}/snapshot_*.hdf5"))
    with h5py.File(path.join(TEST_DIR, f"{TEST_NAME}_ics.hdf5"), "r") as F:
        ids0, x0, w0 = (F["PartType0"][k][:] for k in ("ParticleIDs", "Coordinates", "ParticleVelocities"))
    with h5py.File(snaps[-1], "r") as F:
        t = F["Header"].attrs["Time"]
        ids, x = F["PartType0/ParticleIDs"][:], F["PartType0/Coordinates"][:]
    x0, w0 = x0[np.argsort(ids0)], w0[np.argsort(ids0)]
    x = x[np.argsort(ids)]
    err = np.linalg.norm(x - expected(mode, x0, w0, t), axis=1) / R_BALL
    moved = np.linalg.norm(x - x0, axis=1) / R_BALL
    print(f"\nmode {mode}, t = {t:g}: mesh points moved by up to {moved.max():.4g} R; max error vs the rigid turn {err.max():.3e} R")
    assert abs(t - T_END) < 1e-6
    assert err.max() < TOL, f"mesh points off the rigid turn by up to {err.max():.3e} R (moved {moved.max():.3e} R)"
