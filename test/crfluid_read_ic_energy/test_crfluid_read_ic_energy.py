"""INPUT_READ_COSMICRAYENERGY: a fresh start (start flag 0) keeps the CosmicRayEnergy of the ICs, as stella_dev (init.cc:900).

Bar (set before the first run): every cell of the first snapshot holds the CR energy of the ICs to 1e-5.
"""
import importlib.util
from os import path

import h5py
import numpy as np

from gizmo.test import build_and_run_test, default_mpi_ranks, default_omp_threads

TEST_NAME = "crfluid_read_ic_energy"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"


def test_crfluid_read_ic_energy():
    src = path.join(path.dirname(path.abspath(__file__)), "..", "crfluid_exchange_pdv_only", "make_lattice_blast_ics.py")
    spec = importlib.util.spec_from_file_location("make_lattice_blast_ics", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n = mod.make_lattice_blast_ics(IC_FILE, e_blast=8 / 4096)
    e0 = 1.0 + np.arange(n) / n  # a different value in each cell, by ID
    with h5py.File(IC_FILE, "a") as F:
        F["PartType0/CosmicRayEnergy"] = e0
    build_and_run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads())
    with h5py.File(f"{TEST_DIR}/output/snapshot_000.hdf5", "r") as F:
        g = F["PartType0"]
        ids, e = g["ParticleIDs"][:], g["CosmicRayEnergy"][:]
    print(f"CR energy: {e.sum():.6f} of {e0.sum():.6f}")
    assert np.all(np.abs(e / e0[ids - 1] - 1) < 1e-5)
