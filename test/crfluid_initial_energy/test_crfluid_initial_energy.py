"""CosmicRayInitialAmbient_eVcm3 / CosmicRayInitialCloud_eVcm3 / CosmicRayInitialCloudThreshold_nH: initial CR energy density of a
fresh start (start flag 0, which otherwise starts the CRs at zero), split by hydrogen density.

A 16^3 periodic lattice at rest with the left half ten times denser (n_H about 2850 against 285 cm^-3 in these units), threshold 1000 cm^-3, ambient
1 eV cm^-3 and cloud 0.1 eV cm^-3. The CR energy density in the first snapshot (cell energy / cell volume) must equal the value of its
side of the threshold to 1% in every cell, and the two sides must differ by the ratio 10.
"""
import importlib.util
from os import chdir, path

import h5py
import numpy as np

from gizmo.test import build_and_run_test, default_mpi_ranks, default_omp_threads

TEST_NAME = "crfluid_initial_energy"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
UNIT_PRESSURE_CGS = 1.989e43 * 1e10 / 3.085678e21 ** 3  # code energy density -> erg cm^-3 (kpc, 1e10 Msun, km/s)
EV = 1.602e-12


def _ensure_ic():
    src = path.join(path.dirname(path.abspath(__file__)), "..", "crfluid_exchange_pdv_only", "make_lattice_blast_ics.py")
    spec = importlib.util.spec_from_file_location("make_lattice_blast_ics", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_lattice_blast_ics(IC_FILE, e_blast=8 / 4096)  # the hot cells as cold as the rest
    with h5py.File(IC_FILE, "a") as F:  # left half ten times denser: ten times the mass in the same cell volume
        g = F["PartType0"]
        left = g["Coordinates"][:, 0] < 0.5
        m = g["Masses"][:]
        m[left] *= 10.0
        g["Masses"][...] = m


def test_crfluid_initial_energy():
    _ensure_ic()
    build_and_run_test(TEST_NAME, default_mpi_ranks(), default_omp_threads())
    with h5py.File(f"{TEST_DIR}/output/snapshot_000.hdf5", "r") as F:
        g = F["PartType0"]
        x, rho, m, e = g["Coordinates"][:, 0], g["Density"][:], g["Masses"][:], g["CosmicRayEnergy"][:]
    e_density_ev = e / (m / rho) * UNIT_PRESSURE_CGS / EV  # energy / volume
    left = x < 0.5
    print(f"CR energy density [eV cm^-3]: dense side {e_density_ev[left].mean():.5f}, ambient side {e_density_ev[~left].mean():.5f}")
    assert np.all(np.abs(e_density_ev[left] / 0.1 - 1) < 0.01), "dense cells must start at CosmicRayInitialCloud_eVcm3"
    assert np.all(np.abs(e_density_ev[~left] / 1.0 - 1) < 0.01), "thin cells must start at CosmicRayInitialAmbient_eVcm3"
