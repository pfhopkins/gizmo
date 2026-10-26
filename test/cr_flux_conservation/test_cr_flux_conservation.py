"""Energy conservation of two-moment cosmic-ray transport (COSMIC_RAY_FLUID, eos/cosmic_ray_fluid/cosmic_ray_diffusion.h).

A Gaussian CR bump diffuses through static, cold gas for ~33 equal steps. Each cell of a pair computes the face flux
from both cells' fluxes, so both sides must use the same time level (the predicted flux); otherwise the two sides
exchange different amounts and CR energy is created or destroyed. Requires |dE_total| / E_cr0 < 1e-5 at the end,
where E_total = E_cr + E_thermal + E_kinetic. Mixing time levels gave -1.9e-4; consistent ones give 3.8e-7.
"""

import glob
import importlib.util
import os

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, clean_test_outputs, run_test

TEST_NAME = "cr_flux_conservation"
TEST_DIR = f"test/{TEST_NAME}"
TOL = 1e-5


def _energies(snap):
    with h5py.File(snap, "r") as f:
        p = f["PartType0"]
        m, u, v = p["Masses"][:], p["InternalEnergy"][:], p["Velocities"][:]
        return p["CosmicRayEnergy"][:].sum(), (m * u).sum(), 0.5 * (m[:, None] * v**2).sum()


def test_cr_flux_conservation():
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME)
    spec = importlib.util.spec_from_file_location("make_ics", f"{TEST_DIR}/make_{TEST_NAME}_ics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_ics(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5")
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        run_test(TEST_NAME, 1, 0, restart_flag=2)  # flag 2 keeps the IC's CR energies
    finally:
        os.chdir(cwd)
    snaps = sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))
    assert len(snaps) >= 2, "expected an initial and a final snapshot"
    e0, e1 = np.array(_energies(snaps[0])), np.array(_energies(snaps[-1]))
    drift = (e1 - e0).sum() / e0[0]
    assert abs(drift) < TOL, f"total energy changed by {drift:.3e} of the initial CR energy"
