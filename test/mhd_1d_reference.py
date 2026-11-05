"""Independent 1D ideal-MHD reference solutions for the MHD shock-tube tests (briowu, toth).

A compact finite-volume solver, deliberately sharing no code with GIZMO: HLLD fluxes (Miyoshi &
Kusano 2005, J. Comput. Phys. 208, 315), piecewise-linear reconstruction of the primitive
variables with a minmod limiter, and second-order SSP Runge-Kutta in time. Bx is constant in 1D;
the evolved variables are (rho, rho v, By, Bz, E). Minmod rather than MC: MC leaves percent-level
post-shock oscillations behind the slow shock in briowu at 1024 cells; minmod converges
monotonically, and at 4096 cells differs from the 2048-cell solution by <0.4% (L1 over the
comparison window, relative to each field's range), well below the test tolerances.

Units are Heaviside-Lorentz, as inside GIZMO (MU_0 = 1): magnetic pressure B^2/2, Alfven speed
B/sqrt(rho). With all unit parameters left at their defaults (as in the briowu and toth params),
GIZMO's IC and snapshot MagneticField are in these units, so B compares directly.

The box is periodic like the GIZMO runs, so the reference also contains the second, inverted
discontinuity at x = 0 = L. `clean_window` finds the region around the main jump that the waves
from that image have not reached, which is where the tests compare.

    ref = load_or_solve("briowu", "test/briowu")        # cached in test/briowu/*.npz
    rho = np.interp(x, ref["x"], ref["rho"])

The second half of the module bins a 2D GIZMO snapshot onto x and compares it with a reference
(`compare_snapshot`, `plot_comparison`); the tests keep their own tolerances.
"""

from pathlib import Path

import numpy as np

# Bump when the scheme changes, so stale caches are recomputed rather than reused.
SOLVER_VERSION = 1

LIMITER = "minmod"

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
    # Toth 2000, as in Hopkins & Raives 2016: colliding flows, two fast and two
    # slow shocks around a contact.
    "toth": dict(
        left=(1.0, 10.0, 0.0, 0.0, 5.0 / np.sqrt(4 * np.pi), 0.0, 20.0),
        right=(1.0, -10.0, 0.0, 0.0, 5.0 / np.sqrt(4 * np.pi), 0.0, 1.0),
        Bx=5.0 / np.sqrt(4 * np.pi), gamma=5.0 / 3.0, L=4.0, x0=2.0, t_end=0.08,
    ),
}


def _prim_to_cons(W, Bx, gamma):
    rho, vx, vy, vz, By, Bz, P = W
    E = P / (gamma - 1) + 0.5 * rho * (vx * vx + vy * vy + vz * vz) + 0.5 * (Bx * Bx + By * By + Bz * Bz)
    return np.array([rho, rho * vx, rho * vy, rho * vz, By, Bz, E])


def _cons_to_prim(U, Bx, gamma):
    rho, mx, my, mz, By, Bz, E = U
    vx, vy, vz = mx / rho, my / rho, mz / rho
    P = (gamma - 1) * (E - 0.5 * rho * (vx * vx + vy * vy + vz * vz) - 0.5 * (Bx * Bx + By * By + Bz * Bz))
    return np.array([rho, vx, vy, vz, By, Bz, P])


def _fast_speed(W, Bx, gamma):
    rho, P, By, Bz = W[0], W[6], W[4], W[5]
    gp = gamma * P
    B2 = Bx * Bx + By * By + Bz * Bz
    a = (gp + B2) / rho
    return np.sqrt(0.5 * (a + np.sqrt(np.maximum(a * a - 4 * gp * Bx * Bx / (rho * rho), 0.0))))


def _flux(W, U, Bx):
    rho, vx, vy, vz, By, Bz, P = W
    pT = P + 0.5 * (Bx * Bx + By * By + Bz * Bz)
    vB = vx * Bx + vy * By + vz * Bz
    return np.array([
        rho * vx,
        rho * vx * vx + pT - Bx * Bx,
        rho * vx * vy - Bx * By,
        rho * vx * vz - Bx * Bz,
        By * vx - Bx * vy,
        Bz * vx - Bx * vz,
        (U[6] + pT) * vx - Bx * vB,
    ])


def _hlld(WL, WR, Bx, gamma):
    """HLLD flux (Miyoshi & Kusano 2005), vectorised over faces; equation numbers refer to MK05."""
    UL, UR = _prim_to_cons(WL, Bx, gamma), _prim_to_cons(WR, Bx, gamma)
    FL, FR = _flux(WL, UL, Bx), _flux(WR, UR, Bx)
    rL, uL, vL, wL, ByL, BzL, pL = WL
    rR, uR, vR, wR, ByR, BzR, pR = WR
    pTL = pL + 0.5 * (Bx * Bx + ByL * ByL + BzL * BzL)
    pTR = pR + 0.5 * (Bx * Bx + ByR * ByR + BzR * BzR)
    cf = np.maximum(_fast_speed(WL, Bx, gamma), _fast_speed(WR, Bx, gamma))
    SL = np.minimum(uL, uR) - cf  # (67)
    SR = np.maximum(uL, uR) + cf
    dL, dR = rL * (SL - uL), rR * (SR - uR)
    SM = (dR * uR - dL * uL - pTR + pTL) / (dR - dL)  # (38)
    pTs = (dR * pTL - dL * pTR + dL * dR * (uR - uL)) / (dR - dL)  # (41)

    def star(u, v, w, By, Bz, E, pT, d, S):
        rs = d / (S - SM)  # (43)
        den = d * (S - SM) - Bx * Bx
        degenerate = np.abs(den) < 1e-12 * (d * (S - SM) + Bx * Bx)
        den = np.where(degenerate, 1.0, den)
        vs = np.where(degenerate, v, v - Bx * By * (SM - u) / den)  # (44)
        ws = np.where(degenerate, w, w - Bx * Bz * (SM - u) / den)  # (46)
        fac = (d * (S - u) - Bx * Bx) / den
        Bys = np.where(degenerate, By, By * fac)  # (45)
        Bzs = np.where(degenerate, Bz, Bz * fac)  # (47)
        vBs = SM * Bx + vs * Bys + ws * Bzs
        Es = ((S - u) * E - pT * u + pTs * SM
              + Bx * (u * Bx + v * By + w * Bz - vBs)) / (S - SM)  # (48)
        return rs, vs, ws, Bys, Bzs, Es, vBs

    rLs, vLs, wLs, ByLs, BzLs, ELs, vBLs = star(uL, vL, wL, ByL, BzL, UL[6], pTL, dL, SL)
    rRs, vRs, wRs, ByRs, BzRs, ERs, vBRs = star(uR, vR, wR, ByR, BzR, UR[6], pTR, dR, SR)
    sL, sR = np.sqrt(rLs), np.sqrt(rRs)
    sgn = np.sign(Bx)
    SLs, SRs = SM - np.abs(Bx) / sL, SM + np.abs(Bx) / sR  # (51)
    inv = 1.0 / (sL + sR)
    vss = (sL * vLs + sR * vRs + (ByRs - ByLs) * sgn) * inv  # (59)
    wss = (sL * wLs + sR * wRs + (BzRs - BzLs) * sgn) * inv  # (60)
    Byss = (sL * ByRs + sR * ByLs + sL * sR * (vRs - vLs) * sgn) * inv  # (61)
    Bzss = (sL * BzRs + sR * BzLs + sL * sR * (wRs - wLs) * sgn) * inv  # (62)
    vBss = SM * Bx + vss * Byss + wss * Bzss
    ELss = ELs - sL * (vBLs - vBss) * sgn  # (63)
    ERss = ERs + sR * (vBRs - vBss) * sgn

    ULs = np.array([rLs, rLs * SM, rLs * vLs, rLs * wLs, ByLs, BzLs, ELs])
    URs = np.array([rRs, rRs * SM, rRs * vRs, rRs * wRs, ByRs, BzRs, ERs])
    ULss = np.array([rLs, rLs * SM, rLs * vss, rLs * wss, Byss, Bzss, ELss])
    URss = np.array([rRs, rRs * SM, rRs * vss, rRs * wss, Byss, Bzss, ERss])

    FLs = FL + SL * (ULs - UL)  # (64)
    FRs = FR + SR * (URs - UR)
    FLss = FLs + SLs * (ULss - ULs)  # (65)
    FRss = FRs + SRs * (URss - URs)
    return np.select(
        [SL > 0, SLs >= 0, SM >= 0, SRs >= 0, SR >= 0],
        [FL, FLs, FLss, FRss, FRs],
        FR,
    )


def _limited_slope(W, limiter):
    """Limited slopes (per cell) of periodic primitive arrays: "minmod" or "mc" (monotonized
    central)."""
    dl = W - np.roll(W, 1, axis=1)
    dr = np.roll(W, -1, axis=1) - W
    if limiter == "minmod":
        s = np.minimum(np.abs(dl), np.abs(dr))
    else:
        s = np.minimum(np.minimum(2 * np.abs(dl), 2 * np.abs(dr)), 0.5 * np.abs(dl + dr))
    return np.where(dl * dr > 0, np.sign(dl) * s, 0.0)


def _rhs(U, Bx, gamma, dx, bc, limiter):
    W = _cons_to_prim(U, Bx, gamma)
    if bc == "outflow":  # two ghost cells copied from the edges, removed again below
        W = np.concatenate([W[:, :1], W[:, :1], W, W[:, -1:], W[:, -1:]], axis=1)
    dW = _limited_slope(W, limiter)
    Wm, Wp = W - 0.5 * dW, W + 0.5 * dW  # face values at the left/right edge of each cell
    # positivity: drop to first order in any cell whose reconstruction would go non-physical
    bad = (Wm[0] <= 0) | (Wp[0] <= 0) | (Wm[6] <= 0) | (Wp[6] <= 0)
    Wm = np.where(bad, W, Wm)
    Wp = np.where(bad, W, Wp)
    F = _hlld(Wp, np.roll(Wm, -1, axis=1), Bx, gamma)  # F[:, i] is the flux through face i+1/2
    dU = -(F - np.roll(F, 1, axis=1)) / dx
    if bc == "outflow":
        dU = dU[:, 2:-2]
    return dU


def solve(problem, t_end=None, nx=4096, gamma=None, bc="periodic", cfl=0.4, limiter=LIMITER):
    """Evolve a named problem (key of PROBLEMS) or a dict with the same fields to t_end.

    The domain is [0, L) with the jump at x0. bc="periodic" reproduces the GIZMO runs, including
    the inverted discontinuity at x = 0 = L; bc="outflow" has only the jump at x0.

    Returns a dict: x (cell centres), the primitives in PRIM_NAMES, Bx, gamma, t, nx and
    max_signal_speed (the largest |vx| + c_fast seen during the run, an upper bound on every wave
    speed in the solution).
    """
    p = dict(PROBLEMS[problem]) if isinstance(problem, str) else dict(problem)
    gamma = p["gamma"] if gamma is None else gamma
    t_end = p["t_end"] if t_end is None else t_end
    Bx, L, x0 = p["Bx"], p["L"], p["x0"]
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    W = np.where(x[None, :] < x0, np.array(p["left"])[:, None], np.array(p["right"])[:, None])
    U = _prim_to_cons(W, Bx, gamma)
    t, smax_run = 0.0, 0.0
    while t < t_end:
        W = _cons_to_prim(U, Bx, gamma)
        if not (np.all(W[0] > 0) and np.all(W[6] > 0) and np.all(np.isfinite(W))):
            raise FloatingPointError(f"non-physical state at t={t:.4g}")
        smax = float(np.max(np.abs(W[1]) + _fast_speed(W, Bx, gamma)))
        smax_run = max(smax_run, smax)
        dt = min(cfl * dx / smax, t_end - t)
        U1 = U + dt * _rhs(U, Bx, gamma, dx, bc, limiter)
        U = 0.5 * (U + U1 + dt * _rhs(U1, Bx, gamma, dx, bc, limiter))
        t += dt
    W = _cons_to_prim(U, Bx, gamma)
    out = {name: W[k] for k, name in enumerate(PRIM_NAMES)}
    out.update(x=x, Bx=Bx, gamma=gamma, t=t, nx=nx, L=L, x0=x0, max_signal_speed=smax_run,
               left=np.array(p["left"], dtype=float), right=np.array(p["right"], dtype=float))
    return out


def load_or_solve(problem, cache_dir, nx=4096):
    """solve(problem, nx=nx), cached in <cache_dir>/<problem>_reference_<nx>.npz. The cache is
    reused only if it was made by the same SOLVER_VERSION, limiter and problem definition."""
    p = PROBLEMS[problem]
    key = np.array([SOLVER_VERSION, {"minmod": 0, "mc": 1}[LIMITER], nx, p["t_end"], p["gamma"], p["Bx"],
                    p["L"], p["x0"], *p["left"], *p["right"]], dtype=float)
    cache = Path(cache_dir) / f"{problem}_reference_{nx}.npz"
    if cache.exists():
        try:
            with np.load(cache) as f:
                if f["key"].shape == key.shape and np.array_equal(f["key"], key):
                    return {k: (f[k][()] if f[k].ndim == 0 else f[k]) for k in f.files if k != "key"}
        except (OSError, ValueError, KeyError):
            pass  # unreadable or stale: recompute
    ref = solve(problem, nx=nx)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, key=key, **ref)
    tmp.replace(cache)
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
        ax.plot(ref["x"], ref[field], "-", lw=0.8, color="C3", label=f"1D reference ({int(ref['nx'])} cells)")
        for xw in cmp["window"]:
            ax.axvline(xw, color="k", lw=0.5, ls=":")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(f"{test_dir}/{label}.png", dpi=150)
        plt.close(fig)
