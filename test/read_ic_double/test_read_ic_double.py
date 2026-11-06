"""HDF5 ICs keep the precision they are stored in.

A reader that stages Coordinates/Velocities/Masses in a float buffer (unless INPUT_IN_DOUBLEPRECISION or INPUT_POSITIONS_IN_DOUBLE)
loses ~1e-7 of |x| from a double IC: 1e-5..1e-3 in the semi-major axis of a hard binary inside a cluster. HDF5 converts a float
dataset to double exactly, so reading the three blocks through a double buffer keeps float ICs bit-identical.

Both IC precisions, one build, no forces: the t=0 snapshot (OUTPUT_IN_DOUBLEPRECISION) must hold the IC values exactly -- the double
values for a double IC, the float values for a float IC.
"""

import glob
from os import chdir, getcwd
from shutil import rmtree

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, clean_test_outputs, run_test

from make_read_ic_double_ics import make_ic

TEST_NAME = "read_ic_double"
TEST_DIR = f"test/{TEST_NAME}"
FIELDS = ("Coordinates", "Velocities", "Masses")


def _load(fname):
    with h5py.File(fname, "r") as f:
        g = f["PartType4"]
        o = np.argsort(g["ParticleIDs"][:])
        return {k: np.asarray(g[k][:], dtype=np.float64)[o] for k in FIELDS}


@pytest.mark.parametrize("num_mpi_ranks", (2,))
def test_read_ic_double(num_mpi_ranks):
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME)
    for dtype in (np.float64, np.float32):
        cwd = getcwd()
        try:
            chdir(TEST_DIR)
            rmtree("output", ignore_errors=True)
            make_ic(f"{TEST_NAME}_ics.hdf5", dtype)
            run_test(TEST_NAME, num_mpi_ranks)
            ic, snap = _load(f"{TEST_NAME}_ics.hdf5"), _load(sorted(glob.glob("output/snapshot_*.hdf5"))[0])
        finally:
            chdir(cwd)
        for k in FIELDS:
            assert np.array_equal(ic[k], snap[k]), f"{np.dtype(dtype).name} IC: {k} not read back exactly, max |diff| {np.abs(ic[k] - snap[k]).max():.3e}"
