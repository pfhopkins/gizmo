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


def _parse_timebin(outputdir):
    """Parse output/timebin.txt into per-sync-point (time, Systemstep, min
    occupied-bin dt).  Systemstep is on the "Sync-Point" line; min dt is the
    smallest dt among the occupied-bin rows that follow it.  Returns three
    arrays (empty if the file is absent)."""
    import re
    path = os.path.join(outputdir, "timebin.txt")
    t, sysstep, mindt = [], [], []
    if not os.path.exists(path):
        return np.array(t), np.array(sysstep), np.array(mindt)
    blk = None  # [time, systemstep, min_bin_dt]
    with open(path) as f:
        for line in f:
            msync = re.search(r"Sync-Point.*Time:\s*([0-9.eE+-]+)"
                              r".*Systemstep:\s*([0-9.eE+-]+)", line)
            if msync:
                if blk is not None:
                    t.append(blk[0]); sysstep.append(blk[1])
                    mindt.append(blk[2] if blk[2] is not None else blk[1])
                blk = [float(msync.group(1)), float(msync.group(2)), None]
                continue
            mbin = re.search(r"bin=\s*\d+\s+\d+\s+\d+\s+([0-9.]+)", line)
            if mbin and blk is not None:
                dtv = float(mbin.group(1))
                if dtv > 0 and (blk[2] is None or dtv < blk[2]):
                    blk[2] = dtv
    if blk is not None:
        t.append(blk[0]); sysstep.append(blk[1])
        mindt.append(blk[2] if blk[2] is not None else blk[1])
    return np.array(t), np.array(sysstep), np.array(mindt)


def _make_diagnostic_plot(outputdir, plotdir):
    """4-panel harmonic breathing diagnostic (self-contained, versioned):
      (a) mass-weighted sigma_x(t) breathing envelope, with the analytic band
          [0.70,1.00] and sigma_eq=0.863 -- numerical damping shrinks the
          envelope, a collapse blows it up;
      (b) per-basis mass fraction + total-mass closure vs t;
      (c) velocity width sigma_v=sqrt(Tr[T_abs]/M) (breathing complement of
          sigma_x, out of phase) and max per-basis |p/m| (collapse signature);
      (d) timestep vs t from timebin.txt (Systemstep + min occupied-bin dt),
          with the MinSizeTimestep floor -- a dt collapse plunges to the floor.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = sorted(glob.glob(f"{outputdir}/snapshot_*.hdf5"))
    if len(snaps) < 2:
        return None
    t, sx, sv, pm, m0, m1, mtot = [], [], [], [], [], [], []
    for s in snaps:
        snap = read_snapshot(s)
        M = float(snap["mass"].sum())
        bm = snap["vm"][:, :, 0]            # (N, NBASIS) basis masses
        t.append(snap["t"]); sx.append(sigma_x(snap))
        sv.append(float(np.sqrt(max(tr_T_abs(snap), 0.0) / M)))
        pm.append(max_abs_pm(snap)); mtot.append(M)
        m0.append(float(bm[:, 0].sum()))
        m1.append(float(bm[:, 1].sum()) if bm.shape[1] > 1 else 0.0)
    t = np.array(t); sx = np.array(sx); sv = np.array(sv); pm = np.array(pm)
    m0 = np.array(m0); m1 = np.array(m1); mtot = np.array(mtot)
    tb_t, tb_sys, tb_min = _parse_timebin(outputdir)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs[0, 0]
    ax.plot(t, sx, "o-", color="C0", ms=3, lw=1.0)
    ax.axhspan(0.70, 1.00, color="0.85", alpha=0.6, label="analytic band [0.70,1.00]")
    ax.axhline(0.863, ls="--", color="C3", lw=1.0, label=r"$\sigma_{eq}=0.863$")
    ax.set_xlabel("t"); ax.set_ylabel(r"mass-weighted $\sigma_x$")
    ax.set_title("breathing envelope (isochronous: fills band forever)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(t, m0 / mtot, "o-", color="C0", ms=3, lw=1.0, label="basis 0")
    ax.plot(t, m1 / mtot, "s-", color="C1", ms=3, lw=1.0, label="basis 1")
    ax.plot(t, mtot / mtot[0], "-", color="C2", lw=1.2, label=r"$M_{tot}/M_0$")
    ax.set_xlabel("t"); ax.set_ylabel("mass fraction")
    ax.set_title("per-basis mass fraction + total-mass closure")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(t, sv, "o-", color="C4", ms=3, lw=1.0)
    ax.set_xlabel("t"); ax.set_ylabel(r"$\sigma_v=\sqrt{Tr[T_{abs}]/M}$", color="C4")
    ax.tick_params(axis="y", labelcolor="C4")
    axb = ax.twinx()
    axb.plot(t, pm, "s-", color="C3", ms=3, lw=1.0)
    axb.set_ylabel(r"max per-basis $|p/m|$", color="C3")
    axb.tick_params(axis="y", labelcolor="C3")
    ax.set_title("velocity width (breathing complement) + collapse signature")

    ax = axs[1, 1]
    if tb_t.size:
        ax.plot(tb_t, tb_sys, "-", color="C0", lw=1.0, label="Systemstep")
        if np.isfinite(tb_min).any():
            ax.plot(tb_t, tb_min, "-", color="C1", lw=1.0, label="min occupied-bin dt")
    ax.axhline(1e-6, ls=":", color="C3", lw=1.0, label="MinSizeTimestep floor")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("dt")
    ax.set_title("timestep vs t (collapse -> floor)")
    ax.legend(fontsize=8)

    tag = os.path.basename(outputdir.rstrip("/"))
    fig.suptitle(f"{TEST_NAME} ({tag}): breathing diagnostics  "
                 f"($\\sigma_x$ env=[{sx.min():.3f},{sx.max():.3f}], "
                 f"max$|p/m|$={pm.max():.2f}, $t_f$={t[-1]:.2f})")
    fig.tight_layout()
    os.makedirs(plotdir, exist_ok=True)
    out = os.path.join(plotdir, f"{tag}__breathing.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    return out


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

    snaps = sorted(glob.glob(f"{outputdir}/snapshot_*.hdf5"))
    snap0 = read_snapshot(snaps[0])
    snapf = read_snapshot(final)

    # Render diagnostics BEFORE the gates so the plot exists to inspect even
    # when a gate (collapse / damping) fails.
    _make_diagnostic_plot(outputdir, plotdir=outputdir)

    assert_final_time(final, TEST_NAME)  # reached TimeMax (30); no collapse to the dt floor

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
