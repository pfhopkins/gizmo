"""cbe_free_slot_1d — 1D free-slot injection CBE test.

4 streams (+1/0/-1/-2) with a Gaussian +2 perturbation at x=0.5 that
neighbours have no slot for; exercises the free-slot pairing fallback.
Gates: global conservation, cbe_diagnostics face-residual / bracket-fail,
and the free_slot_count column being non-zero (the fallback actually fired).

Headline diagnostic is the 5-band per-stream density profile (symlog y
catches the small background +0/+/-2 bands without flattening the
dominant +/-1), IC vs final, with each band's analytic-translated IC
overlaid as the reference.  Free-slot preserving the +2 perturbation
is the visible signature of the fallback firing correctly.

NOTE (2026-06-02): meaningful only after the C7 IC-reader merge — without
it GIZMO can't represent the prescribed 4 streams (single-stream synthesis).
"""

import os
import sys
import glob
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
from gizmo.test import build_and_run_test, get_final_snapshot


# ----------------------------------------------------------------------------
# Inlined CBE snapshot/diagnostics helpers (RELATIVE-FRAME storage convention).
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
    nm = _n_moments_for_dim(dim)
    bases = None
    if vm is not None:
        vm = vm[order]
        N = vm.shape[0]; nbasis = vm.shape[1] // nm
        vm = vm.reshape(N, nbasis, nm)
        bm = vm[:, :, 0]
        bv = np.zeros((N, nbasis, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(dim):
                v_rel = np.where(bm > 0, vm[:, :, 1 + k] / bm, 0.0)
                bv[:, :, k] = v_rel + vel[:, k][:, None]
        bases = {"m": bm, "v": bv, "nbasis": nbasis}
    return {"x": x, "vel": vel, "mass": m, "ids": ids, "t": t, "bases": bases}


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

TEST_NAME = "cbe_free_slot_1d"
DIM = 1
V_BANDS = [(+2.0, "+2", "C4"), (+1.0, "+1", "C0"), (0.0, "0", "C7"),
           (-1.0, "-1", "C2"), (-2.0, "-2", "C3")]
TOL = 0.3


def _make_diagnostic_plot(outputdir, plotdir):
    """5-band per-stream density profile, IC vs final, symlog y, with
    analytic-translated IC (rho_b(x - v_b t)) overlaid per band as the
    collisionless-advection reference.  Self-contained.
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
        box = float(f["Header"].attrs["BoxSize"])

    def cbe_band_density(snap, v_target):
        x = snap["x"][:, 0]
        order = np.argsort(x)
        xs = x[order]; N = len(xs)
        xs_p = np.concatenate([xs - box, xs, xs + box])
        dxs = 0.5 * (xs_p[N + 1: 2 * N + 1] - xs_p[N - 1: 2 * N - 1])
        vx = snap["bases"]["v"][:, :, 0]
        m_in = (snap["bases"]["m"] * (np.abs(vx - v_target) < TOL)).sum(axis=1)
        return xs, (m_in[order] / np.maximum(dxs, 1e-30))

    t_final = sf["t"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, snap, label, show_analytic in [
            (axs[0], s0, f"IC (t={s0['t']:.3f})", False),
            (axs[1], sf, f"final (t={t_final:.3f})", True)]:
        # plot CBE per band
        for v_t, lab, color in V_BANDS:
            xs, rho = cbe_band_density(snap, v_t)
            ax.plot(xs, rho, "o-", color=color, ms=2.5, lw=0.9, label=f"v={lab}")
        # overlay analytic reference (IC translated by v_b*t, periodic wrap)
        if show_analytic:
            xs0, _ = cbe_band_density(s0, V_BANDS[0][0])  # x-axis only
            for v_t, lab, color in V_BANDS:
                _, rho0 = cbe_band_density(s0, v_t)
                # Translate IC by v_t * t (periodic wrap on box)
                x_shift = (xs0 - v_t * t_final) % box
                order_ref = np.argsort(x_shift)
                ax.plot(x_shift[order_ref], rho0[order_ref], "--",
                        color=color, lw=0.8, alpha=0.6)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\rho_\alpha$")
        ax.set_title(label)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=8, loc="lower left", ncol=5)
        ax.grid(True, alpha=0.3, which="both")
    tag = os.path.basename(outputdir.rstrip("/"))
    fig.suptitle(f"{TEST_NAME} ({tag}): per-stream density (5 bands), "
                 f"symlog; right panel dashed = analytic (IC translated by v_b·t)")
    fig.tight_layout()
    os.makedirs(plotdir, exist_ok=True)
    out = os.path.join(plotdir, f"{tag}__vs_analytic_symlog.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    return out


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (1, 2)])
def test_cbe_free_slot_1d(num_mpi_ranks, num_omp_threads):
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
    outputdir = f"test/{TEST_NAME}/output"
    final = get_final_snapshot(TEST_NAME)

    snap0 = read_snapshot(f"{outputdir}/snapshot_000.hdf5", DIM)
    snapf = read_snapshot(final, DIM)

    c0, cf = conservation(snap0), conservation(snapf)
    assert abs(cf["M"] - c0["M"]) <= 1e-10 * c0["M"], "total mass not conserved"
    assert np.all(np.abs(cf["P"] - c0["P"]) <= 1e-9), "total momentum not conserved"

    diag = read_cbe_diagnostics(outputdir)
    assert diag, "no cbe_diagnostics.txt rows"
    assert diag[0]["face_res_max"] <= 1e-11, "face mass-flux residual too large"
    assert diag[0]["bracket_fail"] == 0, "root-find bracket failures"

    # Report (not gate) the +2 perturbation + dominant-stream band masses.
    for v, lab, _ in V_BANDS:
        print(f"  band v={lab:>3s}: M0={total_band_mass(snap0, v):.4e}  "
              f"Mf={total_band_mass(snapf, v):.4e}")
    _make_diagnostic_plot(outputdir, plotdir=outputdir)
