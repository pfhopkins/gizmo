"""Reference solutions and comparison helpers for the 1D MHD shock-tube tests (briowu, toth).

The references are converged Athena++ (Stone et al. 2020, ApJS 249, 4) solutions: HLLD fluxes,
PLM reconstruction, VL2 integrator, 8192 cells on the same periodic [0, 4] domain as the GIZMO
runs, so they also contain the second, inverted discontinuity at x = 0 = L. They are stored as
plain text in test/<name>/<name>_exact.txt (columns x rho vx vy vz By Bz P; Bx and provenance in
the header). test/mhd_reference_athena/README.md has the athinput files and the exact commands to
regenerate them.

Athena++ and GIZMO's internal units agree (MU_0 = 1): magnetic pressure B^2/2, Alfven speed
B/sqrt(rho). With all unit parameters left at their defaults (as in the briowu and toth params),
GIZMO's IC and snapshot MagneticField are in these units, so B compares directly.

`clean_window` finds the region around the main jump that the waves from the periodic image have
not reached, which is where the tests compare.

    ref = load_reference("briowu")                   # test/briowu/briowu_exact.txt
    rho = np.interp(x, ref["x"], ref["rho"])

The rest of the module bins a 2D GIZMO snapshot onto x and compares it with a reference
(`compare_snapshot`, `plot_comparison`); the tests keep their own tolerances.
"""

from pathlib import Path

import numpy as np

PRIM_NAMES = ("rho", "vx", "vy", "vz", "By", "Bz", "P")

# Left and right primitive states (rho, vx, vy, vz, By, Bz, P), Bx, gamma, box length, jump
# position and end time, matching test/<name>/<name>_ics.hdf5 and <name>.params.
PROBLEMS = {
    # Brio & Wu 1988, on a 4x longer box run 2x longer than the classic [0,1], t=0.1 setup.
    "briowu": dict(
        left=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        right=(0.125, 0.0, 0.0, 0.0, -1.0, 0.0, 0.1),
        Bx=0.75, gamma=2.0, L=4.0, x0=2.0, t_end=0.2,
    ),
    # Toth 2000 (Ryu & Jones 1995 test 1a), as in Hopkins & Raives 2016: colliding flows, two
    # fast and two slow shocks around a contact.
    "toth": dict(
        left=(1.0, 10.0, 0.0, 0.0, 5.0 / np.sqrt(4 * np.pi), 0.0, 20.0),
        right=(1.0, -10.0, 0.0, 0.0, 5.0 / np.sqrt(4 * np.pi), 0.0, 1.0),
        Bx=5.0 / np.sqrt(4 * np.pi), gamma=5.0 / 3.0, L=4.0, x0=2.0, t_end=0.08,
    ),
}


def load_reference(name):
    """The reference solution for a problem in PROBLEMS, read from test/<name>/<name>_exact.txt:
    x (cell centres), the primitives in PRIM_NAMES, plus Bx, gamma, L, x0, nx and the initial
    left/right states from PROBLEMS."""
    p = PROBLEMS[name]
    path = Path(__file__).resolve().parent / name / f"{name}_exact.txt"
    Bx_file = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# Bx = "):
                Bx_file = float(line.split()[3])
    assert Bx_file is not None and abs(Bx_file / p["Bx"] - 1) < 1e-9, f"{path}: Bx {Bx_file} != {p['Bx']}"
    d = np.loadtxt(path)
    ref = dict(zip(("x",) + PRIM_NAMES, d.T))
    ref.update(Bx=p["Bx"], gamma=p["gamma"], L=p["L"], x0=p["x0"], nx=len(d),
               left=np.array(p["left"], dtype=float), right=np.array(p["right"], dtype=float))
    return ref


def clean_window(ref, pad, rtol=1e-6):
    """The x-range around the main jump to compare in: the disturbed region of the main Riemann
    problem, widened by `pad` on each side, after checking that waves from the periodic image of
    the jump have not reached it.

    A cell is undisturbed if every primitive equals the initial state on its side of x0 to rtol of
    that field's scale. The run of disturbed cells around x0 must be flanked on both sides by
    undisturbed gas more than `pad` wide, so nothing in the window came from the image. Raises
    AssertionError otherwise. Returns (x_lo, x_hi).
    """
    x, x0 = ref["x"], float(ref["x0"])
    W = np.array([ref[n] for n in PRIM_NAMES])
    W0 = np.where(x[None, :] < x0, ref["left"][:, None], ref["right"][:, None])
    scale = np.maximum(np.abs(ref["left"]), np.abs(ref["right"]))
    scale = np.where(scale > 0, scale, 1.0)[:, None]
    disturbed = np.any(np.abs(W - W0) > rtol * scale, axis=0)
    i0 = int(np.searchsorted(x, x0))
    lo = i0 - 1
    while lo > 0 and disturbed[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(x) - 1 and disturbed[hi + 1]:
        hi += 1
    x_lo, x_hi = x[lo] - pad, x[hi] + pad
    # undisturbed flanks of twice the pad: the window plus a margin of at least pad to the image
    gap_l = (x >= x_lo - pad) & (x < x[lo])
    gap_r = (x > x[hi]) & (x <= x_hi + pad)
    assert gap_l.any() and not disturbed[gap_l].any(), (
        f"waves from the periodic image reach the comparison window from the left (x < {x[lo]:.3f})")
    assert gap_r.any() and not disturbed[gap_r].any(), (
        f"waves from the periodic image reach the comparison window from the right (x > {x[hi]:.3f})")
    return float(x_lo), float(x_hi)


def bin_on_x(x, values, edges):
    """Mean of each array in `values` over the particles in each x-bin (2D runs averaged over y),
    and the particle count per bin. Empty bins are NaN."""
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    n = np.bincount(idx[ok], minlength=len(edges) - 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = [np.bincount(idx[ok], weights=v[ok], minlength=len(edges) - 1) / n for v in values]
    return means, n


def regional_l1(x_bins, f_bins, ref, field, window):
    """Mean |f - f_ref| over the bins inside window, divided by the range (max - min) of the
    reference field there. The range rather than the initial jump, since several fields have
    no jump across x0 (v in briowu, rho and By in toth)."""
    sel = (x_bins >= window[0]) & (x_bins <= window[1]) & np.isfinite(f_bins)  # skip empty bins
    f_ref = np.interp(x_bins[sel], ref["x"], ref[field])
    in_win = (ref["x"] >= window[0]) & (ref["x"] <= window[1])
    scale = np.ptp(ref[field][in_win])
    return float(np.mean(np.abs(f_bins[sel] - f_ref)) / scale)


def plateau(ref, field, x_anchor, rtol=2e-3):
    """The interval around x_anchor over which the reference field stays within rtol (relative to
    its value at x_anchor) of that value, and the value. Used to pick out the constant states
    between waves, whose values do not depend on resolution."""
    x = ref["x"]
    f = ref[field]
    i = int(np.argmin(np.abs(x - x_anchor)))
    flat = np.abs(f - f[i]) <= rtol * abs(f[i])
    lo, hi = i, i
    while lo > 0 and flat[lo - 1]:
        lo -= 1
    while hi < len(x) - 1 and flat[hi + 1]:
        hi += 1
    return (float(x[lo]), float(x[hi])), float(np.mean(f[lo:hi + 1]))


# ---------------------------------------------------------------------------------------------
# comparison of a GIZMO snapshot with a reference solution
# ---------------------------------------------------------------------------------------------

COMPARED_FIELDS = ("rho", "vx", "vy", "By")


def read_snapshot(path, gamma):
    """Gas fields of a snapshot as 1D-problem primitives, plus what the divB diagnostic needs."""
    import h5py

    with h5py.File(path, "r") as F:
        g = F["PartType0"]
        header = dict(F["Header"].attrs)
        x = g["Coordinates"][:].astype(float)
        v = g["Velocities"][:].astype(float)
        B = g["MagneticField"][:].astype(float)
        rho = g["Density"][:].astype(float)
        snap = dict(
            x=x[:, 0], rho=rho, vx=v[:, 0], vy=v[:, 1], vz=v[:, 2], By=B[:, 1], Bz=B[:, 2], Bx=B[:, 0],
            P=(gamma - 1) * rho * g["InternalEnergy"][:].astype(float),
            mass=g["Masses"][:].astype(float), Bmag=np.linalg.norm(B, axis=1),
            divB=g["DivergenceOfMagneticField"][:].astype(float) if "DivergenceOfMagneticField" in g else None,
            box_y=float(header["BoxSize"]), time=float(header["Time"]),
        )
    # MagneticField is written in units of UnitMagneticField_In_Gauss; it equals the internal
    # (MU_0 = 1, magnetic pressure B^2/2) field only when that matches Internal_UnitB_In_Gauss.
    # Older snapshots lack the attributes; the default unit parameters make the two equal.
    if "UnitMagneticField_In_Gauss" in header and "Internal_UnitB_In_Gauss" in header:
        ratio = float(header["UnitMagneticField_In_Gauss"]) / float(header["Internal_UnitB_In_Gauss"])
        snap["B_unit_ratio"] = ratio
    return snap


def compare_snapshot(snap, ref, plateaus=(), pad_bins=5, shrink_bins=3):
    """Bin the particles onto x (mean over y), with bin width twice the mean interparticle spacing
    sqrt(Lx*Ly/N) (a rarefaction stretches the spacing by up to ~1.5x, so one-spacing bins go
    empty there), and compare with the reference in the causally clean window.

    plateaus: (label, field, x_anchor) triples; each plateau is the reference's flat interval
    around x_anchor (see plateau()), shrunk by shrink_bins bins at each end to keep shock
    smearing out, and the binned mean over it is compared with the reference value.

    Returns a dict with the bins, the window, L1[field] (see regional_l1), plateau results
    {label/field: (sim, ref, rel_err, nbins)}, and the divB statistics over the window.
    """
    Lx, Ly, N = float(ref["L"]), snap["box_y"], len(snap["x"])
    dx_bin = 2.0 * np.sqrt(Lx * Ly / N)
    nbins = int(round(Lx / dx_bin))
    edges = np.linspace(0.0, Lx, nbins + 1)
    xb = 0.5 * (edges[1:] + edges[:-1])
    dx_bin = Lx / nbins
    means, count = bin_on_x(snap["x"], [snap[f] for f in COMPARED_FIELDS], edges)
    binned = dict(zip(COMPARED_FIELDS, means))
    window = clean_window(ref, pad=pad_bins * dx_bin)
    in_win = (xb >= window[0]) & (xb <= window[1])
    out = dict(x_bins=xb, binned=binned, count=count, dx_bin=dx_bin, window=window,
               empty_bins_in_window=int(np.sum(count[in_win] == 0)),
               empty_bin_fraction=float(np.mean(count[in_win] == 0)))
    out["L1"] = {f: regional_l1(xb, binned[f], ref, f, window) for f in COMPARED_FIELDS}

    out["plateaus"] = {}
    for label, field, anchor in plateaus:
        (a, b), value = plateau(ref, field, anchor)
        sel = (xb > a + shrink_bins * dx_bin) & (xb < b - shrink_bins * dx_bin)
        sim = float(np.nanmean(binned[field][sel])) if sel.any() else np.nan
        out["plateaus"][f"{label}/{field}"] = (sim, value, abs(sim / value - 1), int(sel.sum()))

    if snap["divB"] is not None:
        # |divB| h/|B| with h = (m/rho)^(1/2), the 2D cell size: the scale over which the
        # face-integrated divergence is measured. The kernel radius is ~2.5x larger at
        # DesNumNgb=20 and would inflate the metric for the same field. |B| >= |Bx| > 0 here.
        sel = (snap["x"] >= window[0]) & (snap["x"] <= window[1])
        h = np.sqrt(snap["mass"] / snap["rho"])
        err = np.abs(snap["divB"]) * h / snap["Bmag"]
        out["divB_rms"] = float(np.sqrt(np.mean(err[sel] ** 2)))
        out["divB_p99"] = float(np.percentile(err[sel], 99))
        out["divB_rms_global"] = float(np.sqrt(np.mean(err**2)))
    return out


def plot_comparison(snap, ref, cmp, test_dir):
    """The existing per-field profile plots (particles vs x), with the binned means and the
    reference overlaid and the comparison window marked."""
    from matplotlib import pyplot as plt

    order = np.argsort(snap["x"])
    for label, field in (("Density", "rho"), ("Velocity", "vx"), ("VelocityY", "vy"), ("By", "By")):
        fig, ax = plt.subplots()
        ax.plot(snap["x"][order], snap[field][order], ".", markersize=1, color="0.6", label="GIZMO particles")
        ax.plot(cmp["x_bins"], cmp["binned"][field], "-", lw=0.8, color="C0", label="GIZMO, binned")
        ax.plot(ref["x"], ref[field], "-", lw=0.8, color="C3", label=f"Athena++ ({int(ref['nx'])} cells)")
        for xw in cmp["window"]:
            ax.axvline(xw, color="k", lw=0.5, ls=":")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(f"{test_dir}/{label}.png", dpi=150)
        plt.close(fig)
