"""cbe_harmonic_1d — 1D hot Gaussian in a fixed analytic harmonic potential.

Type=1 particles carry a 2-basis internal velocity distribution.  The external
potential a = -Omega^2 (x - x_c) (Omega=1, x_c = BoxSize/2) makes a Gaussian
f ~ exp[-(v^2 + Omega^2 (x-x_c)^2)/(2 sigma^2)] an exact stationary CBE
equilibrium.  The IC seeds a width mismatch (xwidth_factor=0.7), so the analytic
response is an UNDAMPED width oscillation at 2*Omega ("breathing").  Gates:
  - the run reaches TimeMax (no collapse to the timestep floor);
  - the mass-weighted sigma_x breathing envelope is preserved (reaches both the
    compressed and expanded extremes, and never runs away) -- numerical damping
    would shrink the envelope, a collapse would blow it up;
  - total mass and the absolute-frame second moment stay bounded;
  - per-basis |p/m| stays bounded (a collapse signature is |p/m| >> 1).
"""

import os
import sys
import subprocess
import glob
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
from gizmo.test import build_and_run_test, get_final_snapshot, assert_final_time


# ----------------------------------------------------------------------------
# Inlined CBE snapshot helpers (RELATIVE-FRAME storage convention).
# Storage: basis_p_stored = m_b * (v_b_abs - P.Vel).  Basis velocity decodes to
# ABSOLUTE frame: v_b_abs = p_stored/m + P.Vel.
# ----------------------------------------------------------------------------
TEST_NAME = "cbe_harmonic_1d"
DIM = 1
NBASIS = 2   # = Config CBE_INTEGRATOR (number of velocity bases per particle)
XC = 6.0     # equilibrium center = BoxSize/2

# Breathing-envelope band (mass-weighted sigma_x).  The xwidth_factor=0.7 IC
# oscillates between ~0.70 (compressed) and ~1.00 (expanded) at 2*Omega; an
# undamped run reaches both extremes.  Bands padded around the measured
# envelope [0.698, 0.998]; the outer [0.60, 1.05] catches damping/blowup.
SIGMA_COMPRESSED_MAX = 0.75   # envelope minimum must dip below this
SIGMA_EXPANDED_MIN   = 0.95   # envelope maximum must rise above this
SIGMA_LO, SIGMA_HI   = 0.60, 1.05
PM_BOUND = 3.0                # per-basis |p/m| bound (collapse signature ~100+)
MASS_TOL = 1e-10


def read_snapshot(path):
    with h5py.File(path, "r") as f:
        p = f["PartType1"]
        x   = p["Coordinates"][:]
        vel = p["Velocities"][:]
        m   = p["Masses"][:]
        ids = p["ParticleIDs"][:]
        vm  = p["VlasovMoments"][:]
        t   = float(f["Header"].attrs.get("Time", -1.0))
    order = np.argsort(ids)
    x, vel, m, vm = x[order], vel[order], m[order], vm[order]
    N = vm.shape[0]; nm = vm.shape[1] // NBASIS
    vm = vm.reshape(N, NBASIS, nm)
    return {"x": x, "vel": vel, "mass": m, "vm": vm, "t": t}


def sigma_x(snap):
    """Mass-weighted spatial dispersion about the equilibrium center."""
    x = snap["x"][:, 0]; m = snap["mass"]
    return float(np.sqrt((m * (x - XC) ** 2).sum() / m.sum()))


def max_abs_pm(snap):
    """Max over particles/bases of |p_rel| / m_b (relative-frame basis speed)."""
    bm = snap["vm"][:, :, 0]; bp = snap["vm"][:, :, 1]
    pm = np.divide(np.abs(bp), bm, out=np.zeros_like(bp), where=bm > 1e-12)
    return float(pm.max())


def tr_T_abs(snap):
    """Total absolute-frame raw second-moment trace (frame rule per
    feedback_validation: Tr[T_abs]=Tr[T_rel]+2 V.p_rel+m|V|^2)."""
    vm = snap["vm"]; V = snap["vel"]; m = vm[:, :, 0]
    total = 0.0
    for a in range(DIM):
        p_rel_a  = vm[:, :, 1 + a]
        T_rel_aa = vm[:, :, 1 + DIM + a]
        Va = V[:, a][:, None]
        total += float((T_rel_aa + 2.0 * Va * p_rel_a + m * Va * Va).sum())
    return total


def make_ic_for_test():
    """Regenerate the breathing 2-basis IC (make_ic.py N sigma_v xwidth nbasis)
    so a bare `pytest test/cbe_harmonic_1d` is self-contained."""
    subprocess.run([sys.executable, "make_ic.py", "256", "1.0", "0.7", "2"],
                   cwd=os.path.dirname(os.path.abspath(__file__)), check=True)


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (1, 2)])
def test_cbe_harmonic_1d(num_mpi_ranks, num_omp_threads):
    make_ic_for_test()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    outputdir = f"test/{TEST_NAME}/output"
    final = get_final_snapshot(TEST_NAME)

    assert_final_time(final, TEST_NAME)  # reached TimeMax (30); no collapse to the dt floor

    snaps = sorted(glob.glob(f"{outputdir}/snapshot_*.hdf5"))
    snap0 = read_snapshot(snaps[0])
    snapf = read_snapshot(final)

    # mass + second-moment bounded
    M0, Mf = float(snap0["mass"].sum()), float(snapf["mass"].sum())
    dM_rel = abs(Mf - M0) / M0
    assert dM_rel <= MASS_TOL, f"total mass not conserved: dM/M={dM_rel:.2e}"

    # breathing envelope over the whole trajectory
    sx = np.array([sigma_x(read_snapshot(s)) for s in snaps])
    pm = max(max_abs_pm(read_snapshot(s)) for s in snaps)
    print(f"  sigma_x envelope=[{sx.min():.3f},{sx.max():.3f}]  "
          f"max|p/m|={pm:.3f}  dM/M={dM_rel:.2e}")
    assert sx.min() <= SIGMA_COMPRESSED_MAX, \
        f"breathing under-compresses (damped?): min sigma_x={sx.min():.3f}"
    assert sx.max() >= SIGMA_EXPANDED_MIN, \
        f"breathing under-expands (damped?): max sigma_x={sx.max():.3f}"
    assert sx.min() >= SIGMA_LO and sx.max() <= SIGMA_HI, \
        f"sigma_x envelope [{sx.min():.3f},{sx.max():.3f}] outside [{SIGMA_LO},{SIGMA_HI}]"
    assert pm <= PM_BOUND, f"per-basis |p/m|={pm:.3f} exceeds bound {PM_BOUND} (collapse?)"
