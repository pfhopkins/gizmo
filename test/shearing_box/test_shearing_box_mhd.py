"""shearing_box, MAGNETIC variant: ideal MHD in the 3D shear-periodic box (BOX_SHEARING=3), q = 3/2.
Built from this directory's Config.sh with MAGNETIC added; run with shearing_box_mhd.params.

Units: Omega = 1, L = 1, c_s = 1, density 1, code B with v_A = B / sqrt(rho). The magnetic field is shear-periodic,
B(x + L, y - q Omega L t, z) = B(x, y, z), so the boundary remaps positions and v_phi but leaves B unchanged.

  windup  uniform B = (B0, 0, B0) in the equilibrium shear flow on a lattice. Shear winds the radial field into an
          azimuthal one, B_phi = -q Omega t B_r, while the field stays uniform: no force acts and the flow keeps its
          equilibrium. Checks: B stays uniform to round-off (any inconsistency across the x-boundary breaks
          uniformity), B_phi / B_r follows the exact wind-up, and the flow stays at v_r = eta = v_z = 0. |B| itself
          oscillates by ~1e-3 with the kernel volume estimate of the sliding lattice (the density does the same in
          the hydro-only equilibrium), so it is reported, not asserted.
  mri     the fastest-growing axisymmetric MRI eigenmode of a vertical field B_z0, with k = 2 pi / L along z and
          k v_A = sqrt(15/16) Omega, growth rate s = q Omega / 2 = 0.75 Omega (Balbus & Hawley 1991). beta ~ 84, so
          the incompressible linear theory applies. The mode is seeded with amplitude 1e-4 c_s and grown ~3 e-folds;
          the growth rate fitted to rms B_r must be within 2% of theory at 32 particles per wavelength (and growing
          at 16).
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
from gizmo.test import build_gizmo_for_test, run_test, default_mpi_ranks
from make_shearing_box_ics import make_shearing_box_ics

TEST_NAME = "shearing_box"
PARAMS_NAME = "shearing_box_mhd"
TEST_DIR = Path(__file__).parent
NUM_MPI_RANKS = default_mpi_ranks(max_ranks=4)
Q = 1.5

WIND_N, WIND_B0, WIND_T = 16, 0.05, 4 * np.pi  # two orbits: B_phi grows to -0.94
MRI_N, MRI_AMP, MRI_T = (16, 32), 1e-4, 9.0
MRI_K = 2 * np.pi
MRI_BZ = np.sqrt(15 / 16) / MRI_K  # fastest-growing mode at k = 2 pi / L

TOL_SPREAD = 1e-6
# MRI growth rate at 32 particles per wavelength (-0.5% measured; -3.0% at 16)
TOL_MRI_RATE = 0.02


def mri_eigenmode(q=Q, k=MRI_K, bz=MRI_BZ):
    """fastest-growing eigenvalue and eigenvector of (dv_r, deta, dB_r, dB_phi) ~ exp(i k z + s t)"""
    w = 1j * k * bz
    M = np.array([[0, 2, w, 0],
                  [-(2 - q), 0, 0, w],
                  [w, 0, 0, 0],
                  [0, w, -q, 0]], dtype=complex)
    s, V = np.linalg.eig(M)
    i = np.argmax(s.real)
    return s[i], V[:, i]


def write_case(tag, N, B, dv=None):
    """lattice shear-flow ICs plus a magnetic field; B and dv are callables of positions returning [n,3]"""
    fn = f"{tag}_ics.hdf5"
    make_shearing_box_ics(fn, Q, N=N)
    with h5py.File(fn, "r+") as f:
        p = f["PartType0"]
        x = p["Coordinates"][:]
        if dv is not None:
            p["Velocities"][...] = p["Velocities"][:] + dv(x)
        p["MagneticField"] = B(x)
    return tag


def mri_fields(N):
    s, e = mri_eigenmode()
    e = e / np.abs(e[:2]).max() * MRI_AMP
    ph = lambda x: np.exp(1j * MRI_K * x[:, 2])[:, None]
    dv = lambda x: np.real(ph(x) * np.array([e[0], e[1], 0])[None, :])
    B = lambda x: np.real(ph(x) * np.array([e[2], e[3], 0])[None, :]) + np.array([0, 0, MRI_BZ])[None, :]
    return B, dv


def cases():
    c = [("windup", WIND_N, lambda x: np.tile([WIND_B0, 0, WIND_B0], (len(x), 1)), None, WIND_T, 0.5)]
    for N in MRI_N:
        B, dv = mri_fields(N)
        c.append((f"mri_N{N}", N, B, dv, MRI_T, 0.25))
    return c


@pytest.fixture(scope="module")
def runs():
    build_gizmo_for_test(TEST_NAME, 0, (f"BOX_SHEARING_Q=({Q!r})", "MAGNETIC"))
    out = {}
    cwd = os.getcwd()
    os.chdir(TEST_DIR)
    try:
        for tag, N, B, dv, t_max, dt_snap in cases():
            write_case(tag, N, B, dv)
            outdir = f"output_mhd_{tag}"
            if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
                shutil.rmtree(outdir, ignore_errors=True)
            run_test(TEST_NAME, NUM_MPI_RANKS, 0, params_name=PARAMS_NAME, param_overrides={
                "InitCondFile": f"{tag}_ics", "OutputDir": outdir, "TimeMax": f"{t_max:.12g}",
                "TimeBetSnapshot": f"{dt_snap:.12g}"})
            out[tag] = (str(TEST_DIR / outdir), t_max)
    finally:
        os.chdir(cwd)
    return out


def load(outdir, t_max):
    snaps = []
    for s in sorted(glob.glob(f"{outdir}/snapshot_*.hdf5")):
        with h5py.File(s, "r") as f:
            p = f["PartType0"]
            o = np.argsort(p["ParticleIDs"][:])
            snaps.append(dict(t=float(f["Header"].attrs["Time"]), x=p["Coordinates"][:][o], v=p["Velocities"][:][o],
                              B=p["MagneticField"][:][o], rho=p["Density"][:][o]))
    assert snaps and abs(snaps[-1]["t"] / t_max - 1) < 1e-6, f"{outdir}: run did not reach t = {t_max}"
    return snaps


def test_windup(runs):
    snaps = load(*runs["windup"])
    worst_spread, worst_mean, worst_flow, worst_mag = 0, 0, 0, 0
    for s in snaps:
        B, t = s["B"], s["t"]
        Bm = B.mean(axis=0)
        worst_spread = max(worst_spread, (B.std(axis=0) / WIND_B0).max())
        worst_mean = max(worst_mean, abs(Bm[1] / Bm[0] + Q * t) / max(1, Q * t), abs(Bm[2] / Bm[0] - 1))
        worst_mag = max(worst_mag, abs(Bm[0] / WIND_B0 - 1))
        dv = np.c_[s["v"][:, 0], s["v"][:, 1] + Q * (s["x"][:, 0] - 0.5), s["v"][:, 2]]
        worst_flow = max(worst_flow, np.abs(dv).max())
    print(f"windup: B spread {worst_spread:.2e} B0, wind-up error {worst_mean:.2e} (relative), flow {worst_flow:.2e}, "
          f"|B_r/B0 - 1| up to {worst_mag:.1e}")
    assert worst_spread < TOL_SPREAD, f"field not uniform: spread {worst_spread:.3e} B0"
    assert worst_mean < TOL_SPREAD, f"B_phi/B_r departs from the exact wind-up -q Omega t: {worst_mean:.3e}"
    assert worst_flow < TOL_SPREAD, f"flow departs from equilibrium: {worst_flow:.3e}"


def mri_rate(snaps):
    t = np.array([s["t"] for s in snaps])
    br = np.array([np.sqrt(np.mean(s["B"][:, 0] ** 2)) for s in snaps])
    m = (t > 2) & (br < 30 * br[0])  # past the start-up, still linear
    return np.polyfit(t[m], np.log(br[m]), 1)[0], br[-1] / br[0]


def test_mri_growth(runs):
    s_th = mri_eigenmode()[0].real
    rates = {}
    for N in MRI_N:
        rates[N], amp = mri_rate(load(*runs[f"mri_N{N}"]))
        print(f"MRI N={N}: growth rate {rates[N]:.4f} (theory {s_th:.4f}, {rates[N] / s_th - 1:+.3f}); B_r grew x{amp:.1f}")
    for N, r in rates.items():
        assert r > 0.5 * s_th, f"N={N}: MRI not growing at the linear rate: {r:.4f} vs {s_th:.4f}"
    assert abs(rates[MRI_N[-1]] / s_th - 1) < TOL_MRI_RATE, f"N={MRI_N[-1]}: MRI growth rate {rates[MRI_N[-1]]:.4f} vs {s_th:.4f}"
