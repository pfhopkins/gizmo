"""shearing_box_nbody: test particles on a cold epicycle in the 3D shearing box (BOX_SHEARING=3), self-gravity on.

Units: Omega = 1, L = 1, q = 3/2, G = 1 with negligible masses (1e-12), so the motion is the shearing-sheet epicycle
alone: v_r = vr0 cos(kappa t), eta = -(2-q) Omega vr0 sin(kappa t) / kappa, kappa^2 = 2 (2-q) Omega^2, where
eta = v_phi + q Omega X and X = x - L/2; the epicycle energy E = v_r^2 + 2 eta^2 / (2-q) is conserved exactly.
Run for 2 epicycles.

Two particle sets, 8 each, at random positions with the same epicycle:
  type 5  Hermite-integrated (HERMITE_INTEGRATION=32): the Coriolis and tidal terms must enter the Hermite force
          evaluations, which run without the forces added after gravity.
  type 1  kick-drift-kick with the Boris Coriolis step: the Hermite passes must not add the analytic terms again to
          particles they do not re-evaluate.

Two builds: as is (every particle walks the tree every step), and with ADAPTIVE_TREEFORCE_UPDATE, where the type-1
particles (negligible self-gravity, so a long tidal time) skip their walks for most of the run: the tidal term,
which changes on the epicycle timescale, must be re-evaluated each step rather than carried with the skipped walk's
extrapolated gravity, and the Coriolis term must not ride along on it.

Type 5 is read from the IO_HERMITE_SYNC datasets (a synchronised state); type-1 velocities in intermediate snapshots
are half-kicked, and the final snapshot is synchronised.
"""

from pathlib import Path
import glob
import os
import shutil
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "shearing_box_nbody"
TEST_DIR = Path(__file__).parent
Q, VR0, N_EACH = 1.5, 0.1, 8
KAPPA = np.sqrt(2 * (2 - Q))
VARIANTS = {"walk": (), "adaptive": ("ADAPTIVE_TREEFORCE_UPDATE=0.0625",)}

# dt = 4 pi / 512 (MaxSizeTimestep), 2 epicycles. Hermite (third order here: the force depends on the predicted velocity,
# with one corrector pass) is on the epicycle to 6e-5 vr0 and keeps E to 1.2e-4; kick-drift-kick + Boris keeps E to
# 5e-8 at the synchronised end.
TOL_ENERGY = 1e-3
TOL_TRAJ_HERMITE = 1e-3
# half-kicked intermediate type-1 velocities are off by up to |a| dt / 2 = Omega dt vr0 = 0.025 vr0
TOL_TRAJ_KDK = 0.05


def write_ics(path):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        npart = [0, N_EACH, 0, 0, 0, N_EACH]
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart
        h.attrs["NumPart_Total_HighWord"] = [0] * 6
        h.attrs["MassTable"] = [0.0] * 6
        h.attrs["BoxSize"] = 1.0
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        for k, ptype in enumerate((1, 5)):
            x = 0.15 + 0.7 * rng.random((N_EACH, 3))
            v = np.zeros((N_EACH, 3))
            v[:, 0] = VR0
            v[:, 1] = -Q * (x[:, 0] - 0.5)
            p = f.create_group(f"PartType{ptype}")
            p["Coordinates"] = x
            p["Velocities"] = v
            p["Masses"] = np.full(N_EACH, 1e-12)
            p["ParticleIDs"] = np.arange(1 + k * N_EACH, 1 + (k + 1) * N_EACH, dtype=np.uint32)


@pytest.fixture(scope="module", params=list(VARIANTS), ids=list(VARIANTS))
def run(request):
    variant = request.param
    build_gizmo_for_test(TEST_NAME, 0, VARIANTS[variant])
    cwd = os.getcwd()
    os.chdir(TEST_DIR)
    try:
        write_ics("shearing_box_nbody_ics.hdf5")
        outdir = f"output_{variant}"
        if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
            shutil.rmtree(outdir, ignore_errors=True)
        run_test(TEST_NAME, 1, 0, param_overrides={"OutputDir": outdir})
    finally:
        os.chdir(cwd)
    return variant, str(TEST_DIR / outdir)


def load(outdir, ptype):
    snaps = []
    for name in sorted(glob.glob(f"{outdir}/snapshot_*.hdf5")):
        with h5py.File(name, "r") as f:
            p = f[f"PartType{ptype}"]
            sync = "HermiteSyncVelocities" in p
            x = p["HermiteSyncCoordinates" if sync else "Coordinates"][:]
            v = p["HermiteSyncVelocities" if sync else "Velocities"][:]
            snaps.append((float(f["Header"].attrs["Time"]), x, v))
    assert snaps, f"no snapshots in {outdir}"
    return snaps


def check_epicycle(outdir, ptype, tol_traj):
    worst, worst_t = 0.0, 0.0
    snaps = load(outdir, ptype)
    for t, x, v in snaps:
        eta = v[:, 1] + Q * (x[:, 0] - 0.5)
        err = max(np.abs(v[:, 0] - VR0 * np.cos(KAPPA * t)).max(),
                  np.abs(eta + (2 - Q) * VR0 * np.sin(KAPPA * t) / KAPPA).max()) / VR0
        if err > worst:
            worst, worst_t = err, t
    t, x, v = snaps[-1]
    eta = v[:, 1] + Q * (x[:, 0] - 0.5)
    dE = np.abs((v[:, 0] ** 2 + 2 * eta ** 2 / (2 - Q)) / VR0 ** 2 - 1).max()
    print(f"type {ptype}: max deviation from the epicycle {worst:.2e} vr0 (t = {worst_t:.3g}); "
          f"|dE/E| = {dE:.2e} at t = {t:.3g} ({t * KAPPA / (2 * np.pi):.1f} epicycles)")
    assert worst < tol_traj, f"type {ptype} left the epicycle: {worst:.3e} vr0 at t = {worst_t:.4g}"
    assert dE < TOL_ENERGY, f"type {ptype} epicycle energy changed by {dE:.3e}"


def test_hermite_epicycle(run):
    variant, outdir = run
    check_epicycle(outdir, 5, TOL_TRAJ_HERMITE)


def test_kdk_epicycle(run):
    variant, outdir = run
    check_epicycle(outdir, 1, TOL_TRAJ_KDK)
