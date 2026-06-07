"""cbe_free_slot_3d — 3D slab free-slot injection CBE test.

Same physics as cbe_free_slot_1d on a [1.0 x 0.0625 x 0.0625] slab (long
in x, thin in y,z), velocities along x only. Confirms the free-slot
routing + conservation survive in 3D (NMOMENTS=4) and under domain
decomposition (multi-rank).

NOTE (2026-06-02): meaningful only after the C7 IC-reader merge.
"""

import os
import sys
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


def _per_stream_mass_density(snap, v_target, tol=0.3):
    """Per-particle band-mass with |v_b_abs - v_target| < tol.  Returns
    (x_sorted, band_mass)."""
    b = snap["bases"]
    if b is None:
        return snap["x"][:, 0], np.zeros(snap["x"].shape[0])
    vx = b["v"][:, :, 0]
    band_mass = (b["m"] * (np.abs(vx - v_target) < tol)).sum(axis=1)
    return snap["x"][:, 0], band_mass


def plot_cbe_streams(test_name, snap_initial, snap_final, dim, v_bands,
                     output_dir=None):
    """Per-stream band-mass profiles vs x: IC overlaid with final snapshot.
    One panel per velocity band.  Writes <test_name>_streams.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if output_dir is None:
        output_dir = f"test/{test_name}"
    nb = len(v_bands)
    ncols = min(3, nb)
    nrows = (nb + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                            squeeze=False)
    for i, (v, lab) in enumerate(v_bands):
        ax = axs[i // ncols, i % ncols]
        xi, bi = _per_stream_mass_density(snap_initial, v)
        xf, bf = _per_stream_mass_density(snap_final, v)
        oi = np.argsort(xi); of = np.argsort(xf)
        ax.plot(xi[oi], bi[oi], "-",  color="C0", lw=0.8, label="IC")
        ax.plot(xf[of], bf[of], "o", color="C3", ms=3,
                label=f"t={snap_final['t']:.4g}")
        ax.set_xlabel("x"); ax.set_ylabel("band mass / particle")
        ax.set_title(f"stream v={lab}"); ax.legend(fontsize=8)
    for j in range(nb, nrows * ncols):
        axs[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"{test_name}: per-stream band mass (IC vs final)")
    fig.tight_layout()
    out = os.path.join(output_dir, f"{test_name}_streams.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out

TEST_NAME = "cbe_free_slot_3d"
DIM = 3
V_BANDS = [(+2.0, "+2"), (+1.0, "+1"), (0.0, "0"), (-1.0, "-1"), (-2.0, "-2")]


@pytest.mark.parametrize("num_mpi_ranks,num_omp_threads", [(2, 0), (4, 0)])
def test_cbe_free_slot_3d(num_mpi_ranks, num_omp_threads):
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

    for v, lab in V_BANDS:
        print(f"  band v={lab:>3s}: M0={total_band_mass(snap0, v):.4e}  "
              f"Mf={total_band_mass(snapf, v):.4e}")
    plot_cbe_streams(TEST_NAME, snap0, snapf, DIM, V_BANDS,
                     output_dir=f"test/{TEST_NAME}")
