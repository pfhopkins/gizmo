"""cbe_free_slot_3d — 3D slab free-slot injection CBE test.

Same physics as cbe_free_slot_1d on a [1.0 x 0.0625 x 0.0625] slab (long
in x, thin in y,z), velocities along x only. Confirms the free-slot
routing + conservation survive in 3D (NMOMENTS=4) and under domain
decomposition (multi-rank).

NOTE (2026-06-02): meaningful only after the C7 IC-reader merge.
"""

import os
import sys
import subprocess
import glob
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
from gizmo.test import build_and_run_test, get_final_snapshot


# ----------------------------------------------------------------------------
# Inlined CBE snapshot/diagnostics/plot helpers (RELATIVE-FRAME storage).
# Storage: basis_p_stored = m_b * (v_b_abs − P.Vel).  read_snapshot decodes
# basis velocity in ABSOLUTE frame: v_b_abs = p_stored/m + P.Vel.
# ----------------------------------------------------------------------------
def _n_moments_for_dim(dim, secondmoment=False):
    return dim + 1 + (dim * (dim + 1) // 2 if secondmoment else 0)


def read_snapshot(path, dim):
    """Return dict with x, vel(bulk), mass, ids, t, plus per-basis (m, v_xyz)
    decoded from VlasovMoments.  Basis velocity is ABSOLUTE-frame:
    v_b_abs = p_b_stored/m_b + P.Vel."""
    with h5py.File(path, "r") as f:
        p = f["PartType1"]
        x   = p["Coordinates"][:]
        vel = p["Velocities"][:]
        m   = p["Masses"][:]
        ids = p["ParticleIDs"][:]
        vm  = p["VlasovMoments"][:] if "VlasovMoments" in p else None
        t   = float(f["Header"].attrs.get("Time", -1.0))
    order = np.argsort(ids)
    x, vel, m, ids = x[order], vel[order], m[order], ids[order]
    bases = None
    if vm is not None:
        vm = vm[order]
        # nbasis is the Config CBE_INTEGRATOR value; nm follows from the row
        # length so SECONDMOMENT (nm > dim+1) and cold ICs both parse correctly.
        N = vm.shape[0]; nbasis = NBASIS
        nm = vm.shape[1] // nbasis
        secondmoment = (nm > dim + 1)
        vm = vm.reshape(N, nbasis, nm)
        bm = vm[:, :, 0]
        bv = np.zeros((N, nbasis, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(dim):
                v_rel = np.where(bm > 0, vm[:, :, 1 + k] / bm, 0.0)
                bv[:, :, k] = v_rel + vel[:, k][:, None]
        bases = {"m": bm, "v": bv, "nbasis": nbasis, "vm": vm, "nm": nm,
                 "secondmoment": secondmoment}
    return {"x": x, "vel": vel, "mass": m, "ids": ids, "t": t, "bases": bases}


def tr_T_abs(snap):
    """Total ABSOLUTE-frame raw second-moment trace Sum_i Sum_b Tr[T_abs,b] --
    the conserved 2nd-moment invariant (cold IC -> 0).  Frame rule (see
    feedback_validation): Tr[T_abs] = Tr[T_rel] + 2 V.p_rel + m|V|^2, evaluated
    on the stored relative slots (diagonal stress at slot 1+dim+a)."""
    b = snap["bases"]
    if b is None or not b.get("secondmoment"):
        return 0.0
    vm = b["vm"]; V = snap["vel"]; m = vm[:, :, 0]
    total = 0.0
    for a in range(DIM):
        p_rel_a  = vm[:, :, 1 + a]
        T_rel_aa = vm[:, :, 1 + DIM + a]
        Va = V[:, a][:, None]
        total += float((T_rel_aa + 2.0 * Va * p_rel_a + m * Va * Va).sum())
    return total


def conservation(snap):
    """Total mass + absolute-frame momentum (mass*Velocities; equals basis-sum
    by IC closure)."""
    M = float(snap["mass"].sum())
    P = (snap["mass"][:, None] * snap["vel"]).sum(axis=0)
    return {"M": M, "P": P}


def read_cbe_diagnostics(outputdir):
    """Parse output/cbe_diagnostics.txt.  Columns:
      1 time, 2 face_residual_max, 3 face_residual_sum, 4 bracket_fail,
      5 rho_clamp, 6 S_clamp, 7 repair_dP, 8 repair_dT, 9 free_slot_count."""
    path = os.path.join(outputdir, "cbe_diagnostics.txt")
    cols = ["time", "face_res_max", "face_res_sum", "bracket_fail",
            "rho_clamp", "S_clamp", "repair_dP", "repair_dT", "free_slot"]
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            vals = line.split()
            if len(vals) < 9: continue
            rows.append({c: float(vals[i]) for i, c in enumerate(cols)})
    return rows


def total_band_mass(snap, v_target, tol=0.3):
    """Total mass (summed over all particles + bases) in absolute-frame velocity
    band |v_b_abs - v_target| < tol.  Used for per-stream-band comparisons."""
    b = snap["bases"]
    if b is None: return 0.0
    vx = b["v"][:, :, 0]
    return float((b["m"] * (np.abs(vx - v_target) < tol)).sum())


def _make_diagnostic_plot(outputdir, plotdir):
    """5-band per-stream density vs x (the thin y,z slab collapsed onto x), IC
    vs final, symlog y, with the analytic collisionless-advection reference
    (IC translated by v_b*t, periodic on the x-extent) overlaid on the final
    panel.  Production PDF colocated with the run output.  Self-contained.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py

    snaps = sorted(glob.glob(f"{outputdir}/snapshot_*.hdf5"))
    if len(snaps) < 2:
        return None
    s0 = read_snapshot(snaps[0], DIM)
    sf = read_snapshot(snaps[-1], DIM)
    with h5py.File(snaps[0], "r") as f:
        Lx = float(f["Header"].attrs["BoxSize"]) * BOX_LONG_X   # periodic x-extent

    def band_density_vs_x(snap, v_target, tol=0.3, nbins=64):
        # Collapse the thin y,z slab onto FIXED x-bins over the periodic extent.
        # (The Type-1 mesh points drift, so grouping by raw particle-x would give
        #  ~all-distinct positions and a meaningless near-zero bin width.)
        x = snap["x"][:, 0] % Lx
        vx = snap["bases"]["v"][:, :, 0]
        m_in = (snap["bases"]["m"] * (np.abs(vx - v_target) < tol)).sum(axis=1)
        edges = np.linspace(0.0, Lx, nbins + 1)
        band, _ = np.histogram(x, bins=edges, weights=m_in)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, band / (Lx / nbins)

    t_final = sf["t"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, snap, label, show_analytic in [
            (axs[0], s0, f"IC (t={s0['t']:.3f})", False),
            (axs[1], sf, f"final (t={t_final:.3f})", True)]:
        for v_t, lab, color in V_BANDS:
            xs, rho = band_density_vs_x(snap, v_t)
            ax.plot(xs, rho, "o-", color=color, ms=2.5, lw=0.9, label=f"v={lab}")
        if show_analytic:
            for v_t, lab, color in V_BANDS:
                xs0, rho0 = band_density_vs_x(s0, v_t)
                # Forward advection rho(x,t)=rho(x-v*t,0): IC value at xs0 -> xs0+v_t*t.
                x_shift = (xs0 + v_t * t_final) % Lx
                o = np.argsort(x_shift)
                ax.plot(x_shift[o], rho0[o], "--", color=color, lw=0.8, alpha=0.6)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\rho_\alpha$")
        ax.set_title(label); ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=8, loc="lower left", ncol=5)
        ax.grid(True, alpha=0.3, which="both")
    tag = os.path.basename(outputdir.rstrip("/"))
    fig.suptitle(f"{TEST_NAME} ({tag}): per-stream density vs x (slab collapsed), "
                 f"symlog; right panel dashed = analytic (IC translated by v_b·t)")
    fig.tight_layout()
    os.makedirs(plotdir, exist_ok=True)
    out = os.path.join(plotdir, f"{tag}__vs_analytic_symlog.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    return out

TEST_NAME = "cbe_free_slot_3d"
DIM = 3
NBASIS = 4   # = Config CBE_INTEGRATOR (number of velocity bases per particle)
BOX_LONG_X = 16   # = Config BOX_LONG_X; periodic x-extent = Header BoxSize * BOX_LONG_X
V_BANDS = [(+2.0, "+2", "C4"), (+1.0, "+1", "C0"), (0.0, "0", "C7"),
           (-1.0, "-1", "C2"), (-2.0, "-2", "C3")]
# Under adaptive timesteps momentum and the 2nd moment are integration-advanced:
# they conserve only to integration order (mass is FP via the explicit per-pair
# net-dm=0 closure; P and Tr[T_abs] have no such closure and vanish to ~1e-15
# only under lockstep).  Asymmetric free-slot streams (+1/0/-1/-2) -> net P != 0
# and a nonzero drift.  Tolerances are integration-error level, padded above the
# measured adaptive drift.
MOM_TOL = 3e-3   # abs momentum-drift tol (asymmetric -> measured ~6.8e-4 adaptive; ~4x pad)
TR_TOL  = 3e-3   # rel Tr[T_abs]-drift tol (measured ~7.9e-4 adaptive; ~4x pad)


def make_ic_for_test():
    """(Re)generate the canonical cold IC via this test's make_ic.py so a bare
    `pytest test/cbe_free_slot_3d` is self-contained."""
    subprocess.run([sys.executable, "make_ic.py"],
                   cwd=os.path.dirname(os.path.abspath(__file__)), check=True)


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (4, 0)])
def test_cbe_free_slot_3d(num_mpi_ranks, num_omp_threads):
    make_ic_for_test()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    outputdir = f"test/{TEST_NAME}/output"
    final = get_final_snapshot(TEST_NAME)

    snap0 = read_snapshot(f"{outputdir}/snapshot_000.hdf5", DIM)
    snapf = read_snapshot(final, DIM)

    c0, cf = conservation(snap0), conservation(snapf)
    dM_rel = abs(cf["M"] - c0["M"]) / c0["M"]
    dP = np.abs(cf["P"] - c0["P"])
    T0, Tf = tr_T_abs(snap0), tr_T_abs(snapf)
    dTr_rel = abs(Tf - T0) / abs(T0) if T0 else 0.0
    print(f"  conservation: dM/M={dM_rel:.2e}  dP={dP}  dTr[Tabs]/T0={dTr_rel:.2e}  "
          f"(mom tol {MOM_TOL:.1e}, 2nd-moment tol {TR_TOL:.1e})")
    assert dM_rel <= 1e-10, "total mass not conserved"
    assert np.all(dP <= MOM_TOL), \
        f"momentum drift {dP} exceeds integration-error tol {MOM_TOL:.1e}"
    assert dTr_rel <= TR_TOL, \
        f"2nd-moment Tr[T_abs] drift {dTr_rel:.2e} exceeds integration-error tol {TR_TOL:.1e}"

    diag = read_cbe_diagnostics(outputdir)
    assert diag, "no cbe_diagnostics.txt rows"
    assert diag[0]["face_res_max"] <= 1e-11, "face mass-flux residual too large"
    assert diag[0]["bracket_fail"] == 0, "root-find bracket failures"

    for v, lab, _ in V_BANDS:
        print(f"  band v={lab:>3s}: M0={total_band_mass(snap0, v):.4e}  "
              f"Mf={total_band_mass(snapf, v):.4e}")
    _make_diagnostic_plot(outputdir, plotdir=outputdir)
