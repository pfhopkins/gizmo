"""Energy conservation of two-moment cosmic-ray transport (COSMIC_RAY_FLUID, eos/cosmic_ray_fluid/cosmic_ray_diffusion.h).

A Gaussian CR bump diffuses along a uniform magnetic field through static, cold gas for ~33 equal steps. Each cell
of a pair computes the face flux from both cells' fluxes, so both sides must use the same time level (the predicted
flux); otherwise the two sides exchange different amounts and CR energy is created or destroyed. Requires
|dE| / E_cr0 < 1e-5 between the IC and the final snapshot, with E = E_cr + E_thermal + E_kinetic. Mixing time levels
gave -7.3e-4; consistent ones give 2e-7.

The field is uniform and the gas stays static, so magnetic energy is constant and is left out: its snapshot value
differs from the IC's only through the kernel volume estimate (the volumes sum to ~0.99, not 1).

MAGNETIC is required for a meaningful test: without it the M1 flux relaxes toward +grad(P_cr) and the bump sharpens
instead of spreading, while energy is still conserved. So the test also requires the bump to spread along B.
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


def _state(fn):
    """(E_cr, E_thermal, E_kinetic) and the CR-weighted second moments of the bump along x (= B) and y."""
    with h5py.File(fn, "r") as f:
        p = f["PartType0"]
        m, u, v, x, e = p["Masses"][:], p["InternalEnergy"][:], p["Velocities"][:], p["Coordinates"][:], p["CosmicRayEnergy"][:]
    w = e - e.min()  # the bump above the uniform floor
    sx2, sy2 = ((w * (x[:, a] - 0.5) ** 2).sum() / w.sum() for a in (0, 1))
    return np.array([e.sum(), (m * u).sum(), 0.5 * (m[:, None] * v**2).sum()]), sx2, sy2


def test_cr_flux_conservation():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("compares against the IC it writes itself")
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME)
    ic = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
    spec = importlib.util.spec_from_file_location("make_ics", f"{TEST_DIR}/make_{TEST_NAME}_ics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_ics(ic)
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        run_test(TEST_NAME, 1, 0, restart_flag=2)  # flag 2 keeps the IC's CR energies
    finally:
        os.chdir(cwd)
    snaps = sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5"))
    assert snaps, "no snapshots written"
    (e0, sx0, sy0), (e1, sx1, sy1) = _state(ic), _state(snaps[-1])
    drift = (e1 - e0).sum() / e0[0]
    assert abs(drift) < TOL, f"CR + gas energy changed by {drift:.3e} of the initial CR energy"
    assert sx1 > 1.5 * sx0 and (sx1 - sx0) > 3 * (sy1 - sy0), \
        f"bump did not diffuse along B: <dx^2> {sx0:.4f} -> {sx1:.4f}, <dy^2> {sy0:.4f} -> {sy1:.4f}"
