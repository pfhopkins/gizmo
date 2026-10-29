"""Explicitly integrated cell volumes relax toward the kernel density in a Tillotson build without strength.

A static periodic 1D lattice of zero-pressure Tillotson material (every pressure coefficient zero; the sound speed is
the EOS floor, 1/sqrt(rho)) restarts from a snapshot whose density carries a 10% sinusoid that the particle positions
do not: the kernel density is uniform. Nothing moves, so the integrated density changes only by its relaxation toward
the kernel density, on ~10 sound-crossing times of the gradient length. Without EOS_ELASTIC there is no shear modulus,
and capping the restoring speed by sqrt(mu/rho) with mu = 0 switched the relaxation off.
"""

import glob
from os import chdir, makedirs

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, run_test, clean_test_outputs, assert_final_time

TEST_NAME = "tillotson_volume_relaxation"


def make_ics(fname, n=64, amplitude=0.1):
    x = (np.arange(n) + 0.5) / n
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = 1.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        g = F.create_group("PartType0")
        g.create_dataset("Coordinates", data=np.column_stack([x, np.zeros(n), np.zeros(n)]))
        g.create_dataset("Velocities", data=np.zeros((n, 3)))
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
        g.create_dataset("Masses", data=np.full(n, 1.0 / n))
        g.create_dataset("InternalEnergy", data=np.ones(n))
        g.create_dataset("Density", data=1 + amplitude * np.sin(2 * np.pi * x))
        g.create_dataset("KernelMaxRadius", data=np.full(n, 2.0 / n))  # a snapshot restart starts from the file's kernel radii


def density_spread(snap):
    with h5py.File(snap, "r") as F:
        rho = F["PartType0/Density"][:]
        moved = np.abs(F["PartType0/Coordinates"][:, 0] - (F["PartType0/ParticleIDs"][:] - 0.5) / len(rho)).max()
    return np.std(rho) / np.mean(rho), moved


def test_tillotson_volume_relaxation():
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME)
    chdir(f"test/{TEST_NAME}")
    try:
        make_ics("volume_relaxation_ics.hdf5")
        makedirs("output", exist_ok=True)  # a snapshot restart does not create OutputDir
        run_test(TEST_NAME, 1, 0, restart_flag=2)  # flag 2: the integrated density continues the file's
    finally:
        chdir("../..")
    snaps = sorted(glob.glob(f"test/{TEST_NAME}/output/snapshot_*.hdf5"))
    assert_final_time(snaps[-1], TEST_NAME)
    spread0, _ = density_spread(f"test/{TEST_NAME}/volume_relaxation_ics.hdf5")
    spread, moved = density_spread(snaps[-1])
    print(f"density spread {spread0:.4e} -> {spread:.4e} (ratio {spread / spread0:.4f}); max displacement {moved:.2e}")
    assert moved < 1e-6, "the zero-pressure lattice moved"
    assert spread / spread0 < 0.9, "the integrated density did not relax toward the kernel density"
