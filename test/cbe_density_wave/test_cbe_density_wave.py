"""cbe_density_wave — 1D density-modulated counter-streaming CBE test.

Density-modulated particle spacing (rho=1+eps cos(2 pi x/L)) with the same
(+v/-v) 2-basis internal distribution.  Gates: global conservation, the
cbe_diagnostics face-residual / bracket-fail receipt, clean controlled stop.
Per-stream density vs analytic advection serves as the headline diagnostic.

NOTE (2026-06-02): per-stream gates meaningful only after the C7 IC-reader
merge (see cbe_two_stream).
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

TEST_NAME = "cbe_density_wave"
DIM = 1
EPS_IC = 0.2   # density-wave amplitude in make_ic.py default
V0_IC  = 1.0   # counter-stream speed
TOL    = 0.4   # velocity-band tolerance


def _make_diagnostic_plot(outputdir, plotdir):
    """Per-stream density vs ANALYTIC advection (exact for collisionless pure
    advection: rho_b(x, t) = rho_b(x - v_b * t, 0)).  3-panel: +stream,
    -stream, total density.  CBE = dots, analytic = line.  PDF colocated
    with the run's output dir.  Self-contained — no external harness dep.
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

    # Identify the two stream velocities from the IC (median over particles where m_b>0)
    streams = []
    for b in range(s0["bases"]["nbasis"]):
        mask = s0["bases"]["m"][:, b] > 1e-12
        if mask.any():
            streams.append(float(np.median(s0["bases"]["v"][mask, b, 0])))
    streams = sorted(set(np.round(streams, 3)))
    v_plus  = max(streams)
    v_minus = min(streams)
    M_tot   = float(s0["bases"]["m"].sum())
    rho_b0  = M_tot / (len(streams) * box)   # per-stream mean density

    def cbe_band_density(snap, v_target):
        """Per-particle band density: sum of basis mass with |v_b - v_target|<TOL,
        divided by local cell spacing (neighbor-mid-point in 1D, periodic)."""
        x = snap["x"][:, 0]
        order = np.argsort(x)
        xs = x[order]
        N = len(xs)
        # neighbor-avg dx, periodic
        xs_p = np.concatenate([xs - box, xs, xs + box])
        dxs = 0.5 * (xs_p[N + 1: 2 * N + 1] - xs_p[N - 1: 2 * N - 1])
        vx = snap["bases"]["v"][:, :, 0]
        m_in = (snap["bases"]["m"] * (np.abs(vx - v_target) < TOL)).sum(axis=1)
        return xs, (m_in[order] / np.maximum(dxs, 1e-30))

    def analytic_band(x_arr, v_target, t_now):
        """rho_b(x,t) = rho_b0 (1 + eps cos(k (x - v_b t)))  for the cosine IC."""
        k = 2.0 * np.pi / box
        return rho_b0 * (1.0 + EPS_IC * np.cos(k * (x_arr - v_target * t_now)))

    xs_f, rho_plus_cbe  = cbe_band_density(sf, v_plus)
    _,    rho_minus_cbe = cbe_band_density(sf, v_minus)
    rho_total_cbe = rho_plus_cbe + rho_minus_cbe
    x_an = np.linspace(0, box, 512, endpoint=False)
    rho_plus_an  = analytic_band(x_an, v_plus,  sf["t"])
    rho_minus_an = analytic_band(x_an, v_minus, sf["t"])
    rho_total_an = rho_plus_an + rho_minus_an

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    panels = [(axs[0], v_plus,  rho_plus_cbe,  rho_plus_an,  "+stream", "C1"),
              (axs[1], v_minus, rho_minus_cbe, rho_minus_an, "-stream", "C2"),
              (axs[2], None,    rho_total_cbe, rho_total_an, "total",   "C3")]
    for ax, v, cbe, ana, lab, color in panels:
        ax.plot(x_an, ana, "-", color="C0", lw=1.5, label="analytic")
        ax.plot(xs_f, cbe, "o", color=color, ms=3.5, label="MFM-CBE")
        ax.set_xlabel("x")
        ttl = lab if v is None else f"{lab}  v={v:+.2f}"
        ylab = r"$\rho_{total}$" if v is None else rf"$\rho_{{v={v:+.2f}}}$"
        ax.set_ylabel(ylab); ax.set_title(ttl); ax.legend(fontsize=9)
    tag = os.path.basename(outputdir.rstrip("/"))
    fig.suptitle(f"{TEST_NAME} ({tag}): per-stream density vs analytic (t={sf['t']:.3f})")
    fig.tight_layout()
    os.makedirs(plotdir, exist_ok=True)
    out = os.path.join(plotdir, f"{tag}__vs_analytic.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    return out


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (1, 2)])
def test_cbe_density_wave(num_mpi_ranks, num_omp_threads):
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

    _make_diagnostic_plot(outputdir, plotdir=outputdir)
