"""Compare a GIZMO snapshot against the initial conditions for the
hernquist_sidm-style test problem (small isolated Hernquist DM halo,
~10^3 Type=1 particles).

Use this for any short evolution where the halo should be statistically
indistinguishable from the IC (orbital phase advances; the distribution
function does not). Catches gross drifts in mass profile, kinetic energy,
velocity dispersion, or isotropy that would flag a broken force/scatter
implementation.

Outputs:
  * Console: COM drift, per-particle drift quantiles, total KE / momentum,
    per-component velocity moments, per-radial-shell counts and 3D
    velocity dispersions sigma_v(r), max relative residual of cumulative
    M(<|r|) and M(<|v|).
  * Figure: 2x3 panel comparing IC vs final snapshot, saved alongside
    the snapshots as `validate_snapshot_vs_ic.png`.

Defaults assume the script is run from `test/hernquist_sidm/` against the
hernquist_sidm IC and the final snapshot of the run; pass --ic and
--snapshot to override.
"""

import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_IC       = "hernquist_sidm_ics.hdf5"
DEFAULT_SNAPSHOT = "output/snapshot_002.hdf5"
DEFAULT_PLOT_OUT = "validate_snapshot_vs_ic.png"
MIN_PARTICLES_FOR_BIN = 10   # log-binning starts where ~10 particles are inside


def load_snapshot(path):
    with h5py.File(path, "r") as f:
        p = f["PartType1"]
        return {
            "x":   p["Coordinates"][:],
            "v":   p["Velocities"][:],
            "m":   p["Masses"][:],
            "ids": p["ParticleIDs"][:],
            "t":   float(f["Header"].attrs.get("Time", -1.0)),
        }


def sort_by_ids(snap):
    order = np.argsort(snap["ids"])
    return {k: (v[order] if k != "t" else v) for k, v in snap.items()}


def com(x, m):
    return (m[:, None] * x).sum(axis=0) / m.sum()


def cumulative_profile(values, weights):
    """Return (sorted_values, cumulative_weight)."""
    order = np.argsort(values)
    return values[order], np.cumsum(weights[order])


def log_bins_from_min_count(values, min_count, n_bins):
    """Log-spaced bin edges from the value where >= min_count points lie below."""
    s = np.sort(values)
    vmin = s[min_count - 1]
    vmax = s[-1]
    if vmin <= 0:
        vmin = s[s > 0][0]
    return np.geomspace(vmin, vmax, n_bins + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ic", default=DEFAULT_IC, help=f"IC HDF5 file (default {DEFAULT_IC})")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help=f"Final-state HDF5 file (default {DEFAULT_SNAPSHOT})")
    parser.add_argument("--plot", default=DEFAULT_PLOT_OUT, help=f"Output PNG (default {DEFAULT_PLOT_OUT})")
    args = parser.parse_args()

    ic = sort_by_ids(load_snapshot(args.ic))
    fn = sort_by_ids(load_snapshot(args.snapshot))
    plot_out = args.plot
    assert np.array_equal(ic["ids"], fn["ids"]), "ParticleID set differs"

    com_ic = com(ic["x"], ic["m"])
    com_fn = com(fn["x"], fn["m"])

    print(f"IC      t={ic['t']:.5g}  N={len(ic['ids'])}  Mtot={ic['m'].sum():.6g}")
    print(f"Final   t={fn['t']:.5g}  N={len(fn['ids'])}  Mtot={fn['m'].sum():.6g}")
    print(f"COM IC  = {com_ic}")
    print(f"COM Fin = {com_fn}")
    print(f"COM drift |d|         = {np.linalg.norm(com_fn - com_ic):.3e}")

    # per-particle drifts (matched by sorted ID)
    dx = np.linalg.norm(fn["x"] - ic["x"], axis=1)
    dv = np.linalg.norm(fn["v"] - ic["v"], axis=1)
    print(f"per-particle |dx|     med={np.median(dx):.3e}  p95={np.percentile(dx,95):.3e}  max={dx.max():.3e}")
    print(f"per-particle |dv|     med={np.median(dv):.3e}  p95={np.percentile(dv,95):.3e}  max={dv.max():.3e}")

    # halo-frame |r|, |v|
    r_ic = np.linalg.norm(ic["x"] - com_ic, axis=1)
    r_fn = np.linalg.norm(fn["x"] - com_ic, axis=1)
    v_ic = np.linalg.norm(ic["v"], axis=1)
    v_fn = np.linalg.norm(fn["v"], axis=1)

    # cumulative M(<r), M(<v)
    rs_ic, Mr_ic = cumulative_profile(r_ic, ic["m"])
    rs_fn, Mr_fn = cumulative_profile(r_fn, fn["m"])
    vs_ic, Mv_ic = cumulative_profile(v_ic, ic["m"])
    vs_fn, Mv_fn = cumulative_profile(v_fn, fn["m"])

    # Resample M(<x) onto a common log grid for residuals
    n_bins = 32
    r_edges = log_bins_from_min_count(np.concatenate([r_ic, r_fn]), MIN_PARTICLES_FOR_BIN, n_bins)
    v_edges = log_bins_from_min_count(np.concatenate([v_ic, v_fn]), MIN_PARTICLES_FOR_BIN, n_bins)

    def Mle(values, weights, edges):
        # M(<edge) at each bin edge
        order = np.argsort(values)
        vs = values[order]; ws = weights[order]; cum = np.cumsum(ws)
        idx = np.searchsorted(vs, edges, side="right") - 1
        idx = np.clip(idx, -1, len(cum) - 1)
        return np.where(idx >= 0, cum[idx], 0.0)

    Mr_ic_at = Mle(r_ic, ic["m"], r_edges)
    Mr_fn_at = Mle(r_fn, fn["m"], r_edges)
    Mv_ic_at = Mle(v_ic, ic["m"], v_edges)
    Mv_fn_at = Mle(v_fn, fn["m"], v_edges)

    # fractional residual where IC profile is nonzero
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_r = np.where(Mr_ic_at > 0, Mr_fn_at / Mr_ic_at - 1.0, 0.0)
        frac_v = np.where(Mv_ic_at > 0, Mv_fn_at / Mv_ic_at - 1.0, 0.0)

    print(f"max |dM(<r)/M_IC|     = {np.nanmax(np.abs(frac_r)):.3e}")
    print(f"max |dM(<v)/M_IC|     = {np.nanmax(np.abs(frac_v)):.3e}")

    # Energy / momentum
    def KE(v, m): return 0.5 * (m * np.sum(v * v, axis=1)).sum()
    def Pmag(v, m): return np.linalg.norm((m[:, None] * v).sum(axis=0))
    print(f"IC      KE={KE(ic['v'], ic['m']):.6g}   |P|={Pmag(ic['v'], ic['m']):.3e}")
    print(f"Final   KE={KE(fn['v'], fn['m']):.6g}   |P|={Pmag(fn['v'], fn['m']):.3e}")
    print(f"  dKE/KE = {(KE(fn['v'], fn['m']) - KE(ic['v'], ic['m'])) / KE(ic['v'], ic['m']):+.3e}")
    print(f"  per-comp mean v_IC  = ({ic['v'][:,0].mean():+.4g}, {ic['v'][:,1].mean():+.4g}, {ic['v'][:,2].mean():+.4g})")
    print(f"  per-comp mean v_Fin = ({fn['v'][:,0].mean():+.4g}, {fn['v'][:,1].mean():+.4g}, {fn['v'][:,2].mean():+.4g})")
    print(f"  per-comp sig  v_IC  = ({ic['v'][:,0].std():.4g}, {ic['v'][:,1].std():.4g}, {ic['v'][:,2].std():.4g})")
    print(f"  per-comp sig  v_Fin = ({fn['v'][:,0].std():.4g}, {fn['v'][:,1].std():.4g}, {fn['v'][:,2].std():.4g})")

    # Per-shell velocity dispersion (3D); same shell edges for IC and final
    shell_edges = log_bins_from_min_count(np.concatenate([r_ic, r_fn]), MIN_PARTICLES_FOR_BIN, 8)
    print("\nradial shells (3D sigma_v = sqrt(<|v|^2>)):")
    print(f"  {'r_lo':>8} {'r_hi':>8}   {'N_ic':>5} {'sig_ic':>9}    {'N_fn':>5} {'sig_fn':>9}    dsig/sig")
    shell_ctr = []; sig_ic_arr = []; sig_fn_arr = []; n_ic_arr = []; n_fn_arr = []
    for lo, hi in zip(shell_edges[:-1], shell_edges[1:]):
        sel_ic = (r_ic >= lo) & (r_ic < hi)
        sel_fn = (r_fn >= lo) & (r_fn < hi)
        if sel_ic.sum() < 3 or sel_fn.sum() < 3:
            continue
        s_ic = np.sqrt(((ic['v'][sel_ic]) ** 2).sum(axis=1).mean())
        s_fn = np.sqrt(((fn['v'][sel_fn]) ** 2).sum(axis=1).mean())
        shell_ctr.append(np.sqrt(lo * hi))
        sig_ic_arr.append(s_ic); sig_fn_arr.append(s_fn)
        n_ic_arr.append(sel_ic.sum()); n_fn_arr.append(sel_fn.sum())
        print(f"  {lo:8.3f} {hi:8.3f}   {sel_ic.sum():5d} {s_ic:9.4g}    {sel_fn.sum():5d} {s_fn:9.4g}    {(s_fn-s_ic)/s_ic:+.3e}")
    shell_ctr = np.array(shell_ctr)
    sig_ic_arr = np.array(sig_ic_arr); sig_fn_arr = np.array(sig_fn_arr)
    n_ic_arr = np.array(n_ic_arr); n_fn_arr = np.array(n_fn_arr)

    # ---------- plot ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    ax.loglog(rs_ic, Mr_ic, "k-",  lw=1.6, label="IC")
    ax.loglog(rs_fn, Mr_fn, "C3--", lw=1.4, label=f"final t={fn['t']:.4g}")
    ax.set_xlabel(r"$|r-r_{\rm COM}|$  [code]")
    ax.set_ylabel(r"$M(<r)$")
    ax.set_title("cumulative mass vs radius")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[0, 1]
    ax.loglog(vs_ic, Mv_ic, "k-",  lw=1.6, label="IC")
    ax.loglog(vs_fn, Mv_fn, "C3--", lw=1.4, label=f"final t={fn['t']:.4g}")
    ax.set_xlabel(r"$|v|$  [code]")
    ax.set_ylabel(r"$M(<|v|)$")
    ax.set_title("cumulative mass vs |v|")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[0, 2]
    if len(shell_ctr) > 0:
        ax.loglog(shell_ctr, sig_ic_arr, "k-o",  ms=4, lw=1.4, label="IC")
        ax.loglog(shell_ctr, sig_fn_arr, "C3--s", ms=4, lw=1.2, label=f"final t={fn['t']:.4g}")
    ax.set_xlabel(r"$|r-r_{\rm COM}|$  [code]")
    ax.set_ylabel(r"$\sigma_v(r) = \sqrt{\langle |v|^2 \rangle}$")
    ax.set_title("radial-shell velocity dispersion")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[1, 0]
    ax.semilogx(r_edges, frac_r, "C0o-", ms=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$|r-r_{\rm COM}|$  [code]")
    ax.set_ylabel(r"$M_{\rm fin}(<r) / M_{\rm IC}(<r) - 1$")
    ax.set_title("fractional residual M(<r)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[1, 1]
    ax.semilogx(v_edges, frac_v, "C0o-", ms=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$|v|$  [code]")
    ax.set_ylabel(r"$M_{\rm fin}(<|v|) / M_{\rm IC}(<|v|) - 1$")
    ax.set_title("fractional residual M(<|v|)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    ax = axes[1, 2]
    if len(shell_ctr) > 0:
        ax.semilogx(shell_ctr, (sig_fn_arr - sig_ic_arr) / sig_ic_arr, "C2o-", ms=4, label=r"$\Delta\sigma_v/\sigma_v$")
        # Poisson floor: 1/sqrt(2N) for sigma estimator
        poisson = 1.0 / np.sqrt(2.0 * np.minimum(n_ic_arr, n_fn_arr))
        ax.fill_between(shell_ctr, -poisson, poisson, color="gray", alpha=0.2, label="Poisson $1\\sigma$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$|r-r_{\rm COM}|$  [code]")
    ax.set_ylabel(r"$\Delta \sigma_v / \sigma_v$")
    ax.set_title("fractional change of $\\sigma_v$ per shell")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("hernquist_sidm PAA: IC vs t={:.4g}  (N={}, Mtot={:.3g})".format(
        fn["t"], len(ic["ids"]), ic["m"].sum()))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {plot_out}")


if __name__ == "__main__":
    main()
