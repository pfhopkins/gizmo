"""Mixture equation-of-state tables (GIZMO SESAME format) from component tables, for composition bins of a graded interior.

APPROXIMATION: additive volume of IMMISCIBLE components at a common (P, T): v = sum_j X_j v_j(P, T), u = sum_j X_j u_j, s = sum_j X_j s_j
(immiscible: no mixing entropy). Each component keeps ITS OWN phase equilibrium (no melting-point depression). Where component k is on its
own coexistence line (a flat isotherm in its rho-T table), the mixture has a flat-P band at that pressure: only k is two-phase and its lever
fraction is the free variable (v, u, s linear in v across the band). Only ONE component may be two-phase at a state: where two components'
flat or near-flat stretches (|dlnP/dlnrho| <= 0.05) overlap in pressure on an isotherm, the mixture nodes in that pressure range are
FLAGGED (phase 9) and counted -- never given two independent phase fractions.

Components are evaluated exactly as GIZMO evaluates their tables (sesame_loader: bilinear in log rho / log T, log P and log u, the
per-density validity floor); a component state is usable where it lies on or above its valid floor and P > 0. Mixture axes: T = the first
component's T axis, rho = log-uniform. Written sound speed = Wood's formula over the components, each at its own tabulated sound speed (a
frozen-composition value, continuous at band edges); the first-law sound speed of the finished table, cs^2 = P_rho + P_T (P / rho^2 - u_rho)
/ u_T (in a flat band P_rho = 0: the equilibrium two-phase value), is computed as the check and reported beside it. Phase column: 0 single
phase, 1 + k component k two-phase, 9 overlap flag. The sha256 of every input and of the output are printed.

Not supported: mixtures whose components condense out of one another at low pressure (e.g. water in an H/He envelope near condensation),
which need a chemical-potential treatment rather than additive volumes.

    python3 -m initial_conditions.eos_tools.mix_tables <out.sesame> <table_1> <X_1> [<table_2> <X_2> ...] [--accept-u-order-pairs]
    (fractions sum to 1). A table with pressure falling with density, an undefined sound speed or a density gap is refused; one where u
    does not rise with T (some energies name several temperatures) is refused unless --accept-u-order-pairs."""
import hashlib, os, sys
import numpy as np
from .sesame_loader import SesameTable

FLAT_EXACT = 1e-10        # |d log10 P| between table nodes below which a stretch is an exact (lever) band
NEAR_FLAT = 0.05          # |dlnP/dlnrho| below which a stretch counts as two-phase (used for the overlap flag)
EPS = 1e-12


def isotherm(tab, T):
    """component tab on the isotherm T at its own density nodes, as GIZMO evaluates it; returns arrays ordered by increasing rho (P
    non-decreasing): log10 rho, log10 P, u, s, near-flat mask per interval; only usable nodes (on / above the floor, P > 0)."""
    rho = tab.rho_axis; Tv = np.full(len(rho), T)
    ok = ~tab.below_valid(rho, Tv)
    P = tab.pressure_v(rho, Tv); u = tab.internal_energy_v(rho, Tv); s = tab.entropy_v(rho, Tv)
    ok &= (P > 0) & np.isfinite(P) & (u > 0) & np.isfinite(u) & np.isfinite(s)
    idx = np.where(ok)[0]
    if len(idx) < 3: return None
    # keep the contiguous stretch with P non-decreasing in rho that spans the widest pressure range (the table's validity floor can open
    # a gap at densities far above any mixture state, e.g. Stewart forsterite at 1.1e4 K between 103 and 172 g/cc); the rest is dropped and counted
    lr, lP = np.log10(rho[idx]), np.log10(P[idx])
    cut = np.concatenate([[0], np.where((np.diff(lP) < -EPS) | (np.diff(idx) != 1))[0] + 1, [len(idx)]])
    spans = [(lP[b - 1] - lP[a], a, b) for a, b in zip(cut[:-1], cut[1:]) if b - a >= 2]
    if not spans: return None
    _, k0, k1 = max(spans)
    dropped = len(idx) - (k1 - k0)
    idx, lr, lP = idx[k0:k1], lr[k0:k1], lP[k0:k1]
    if len(idx) < 3: return None
    slope = np.diff(lP) / np.diff(lr)
    return dict(lr=lr, lP=lP, u=u[idx], s=s[idx], near_flat=np.abs(slope) <= NEAR_FLAT, dropped=int(dropped))


def at_pressure(c, lPq):
    """component state (log10 v, u, s) at log10 P values lPq on its isotherm c, as GIZMO evaluates the table between its density nodes:
    log P, log u (and log v) linear in log rho, s linear in log rho -- so at a pressure between two nodes log v, log u and s are linear in
    log P. Exact flats are collapsed: evaluation at a flat's own pressure returns its LOW-density end (the caller adds the high-density end)."""
    lP, lv = c['lP'], -c['lr']
    # strictly increasing pressure abscissa: inside an exact flat keep both ends, nudged apart by EPS
    x = lP.copy()
    for k in range(1, len(x)):
        if x[k] <= x[k - 1] + EPS: x[k] = x[k - 1] + EPS
    lvq = np.interp(lPq, x, lv); uq = 10.0 ** np.interp(lPq, x, np.log10(c['u'])); sq = np.interp(lPq, x, c['s'])
    return lvq, uq, sq


def flats(c):
    """exact flats of a component isotherm: list of (log10 P, i_lo, i_hi) with node indices of the band ends (increasing rho)"""
    out = []; k = 0; d = np.diff(c['lP'])
    while k < len(d):
        if abs(d[k]) <= FLAT_EXACT:
            k0 = k
            while k < len(d) and abs(d[k]) <= FLAT_EXACT: k += 1
            out.append((c['lP'][k0], k0, k))
        else: k += 1
    return out


def near_flat_ranges(c):
    """pressure ranges [lo, hi] (log10 P) of near-flat stretches, exact flats included"""
    out = []; m = c['near_flat']; k = 0
    while k < len(m):
        if m[k]:
            k0 = k
            while k < len(m) and m[k]: k += 1
            out.append((c['lP'][k0], c['lP'][k]))
        else: k += 1
    return out


def mixture_isotherm(comps, X, T):
    """the mixture path on isotherm T: arrays ordered by increasing density (log10 v decreasing): lv, lP, u, s, band (component index or -1
    per segment start), plus overlap pressure ranges"""
    cs = [isotherm(t, T) for t in comps]
    if any(c is None for c in cs): return None
    lo = max(c['lP'][0] for c in cs); hi = min(c['lP'][-1] for c in cs)
    if hi <= lo: return None
    # overlap of near-flat stretches of two different components
    overlaps = []
    nf = [near_flat_ranges(c) for c in cs]
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            for (p0, p1) in nf[a]:
                for (q0, q1) in nf[b]:
                    l, h = max(p0, q0), min(p1, q1)
                    if h >= l and h > lo and l < hi: overlaps.append((max(l, lo), min(h, hi), a, b))
    Pq = np.unique(np.concatenate([c['lP'][(c['lP'] >= lo) & (c['lP'] <= hi)] for c in cs] + [np.array([lo, hi])]))
    st = [at_pressure(c, Pq) for c in cs]
    pts = []          # (lP, lv_mix, u, s, band_component_or_-1)
    fl = [{round(f[0], 9): f for f in flats(c)} for c in cs]
    for n, p in enumerate(Pq):
        v = np.array([10 ** st[k][0][n] for k in range(len(cs))]); u = np.array([st[k][1][n] for k in range(len(cs))])
        s = np.array([st[k][2][n] for k in range(len(cs))])
        hit = [k for k in range(len(cs)) if round(p, 9) in fl[k]]
        if not hit:
            pts.append((p, np.log10(X @ v), X @ u, X @ s, -1, v)); continue
        k = hit[0]                     # a second simultaneous exact flat is inside an overlap range (flagged below)
        _, i_lo, i_hi = fl[k][round(p, 9)]
        c = cs[k]
        for i_end in (i_lo, i_hi):     # low-density end, then high-density end
            vv, uu, ss = v.copy(), u.copy(), s.copy()
            vv[k], uu[k], ss[k] = 10 ** (-c['lr'][i_end]), c['u'][i_end], c['s'][i_end]
            pts.append((p, np.log10(X @ vv), X @ uu, X @ ss, k if i_end == i_lo else -1, vv))
    a = np.array([q[:4] for q in pts]); band = np.array([q[4] for q in pts], int); vk = np.array([q[5] for q in pts])
    return dict(lP=a[:, 0], lv=a[:, 1], u=a[:, 2], s=a[:, 3], band=band, vk=vk, overlaps=overlaps,
                dropped=[c['dropped'] for c in cs], comps=cs)


def onto_grid(path, lr_grid, with_volumes=False):
    """mixture P, u, s and phase at the grid densities (NaN outside the path): inside a band segment P is constant and u, s are linear in v
    (lever); elsewhere log P, log u and s linear in log v (as GIZMO interpolates a table). with_volumes: also each component's specific
    volume at each node (log v_k linear in log v in a single-phase segment, v_k linear in v inside a band)."""
    lv = path['lv']; order = np.argsort(lv)                      # increasing lv for searchsorted
    lvs, lPs, us, ss, bs = lv[order], path['lP'][order], path['u'][order], path['s'][order], path['band'][order]
    q = -lr_grid
    P = np.full(len(q), np.nan); u = P.copy(); s = P.copy(); ph = np.zeros(len(q), int)
    inside = (q >= lvs[0]) & (q <= lvs[-1])
    j = np.clip(np.searchsorted(lvs, q[inside]) - 1, 0, len(lvs) - 2)
    a, b = lvs[j], lvs[j + 1]
    f = np.where(b > a, (q[inside] - a) / np.where(b > a, b - a, 1), 0.0)
    is_band = np.abs(lPs[j + 1] - lPs[j]) <= FLAT_EXACT
    # band: linear in v (not log v)
    fv = np.where(b > a, (10 ** q[inside] - 10 ** a) / np.where(b > a, 10 ** b - 10 ** a, 1), 0.0)
    w = np.where(is_band, fv, f)
    # single phase: log P and log u linear in log v (GIZMO's own interpolation); band: P constant, u linear in v (lever); s linear in the weight
    lus = np.log10(us)
    u_band = us[j] + w * (us[j + 1] - us[j]); u_single = 10 ** (lus[j] + w * (lus[j + 1] - lus[j]))
    P[inside] = 10 ** (lPs[j] + w * (lPs[j + 1] - lPs[j])); u[inside] = np.where(is_band, u_band, u_single); s[inside] = ss[j] + w * (ss[j + 1] - ss[j])
    # band component: the path's band marker sits on the LOW-density end point (larger lv) = index j+1 in increasing-lv order
    ph[inside] = np.where(is_band, 1 + np.maximum(bs[j + 1], bs[j]), 0)
    for (l, h, _, _) in path['overlaps']:
        m = inside & (np.log10(np.where(np.isfinite(P), P, 1)) >= l) & (np.log10(np.where(np.isfinite(P), P, 1)) <= h)
        ph[m] = 9
    if not with_volumes: return P, u, s, ph
    vks = path['vk'][order]; VK = np.full((len(q), vks.shape[1]), np.nan)
    # each component's specific volume: log v_k linear in the weight in a single-phase segment (log v_k linear in log P, as GIZMO interpolates
    # the component); inside a band, v_k linear in v (lever)
    lvk = np.log10(vks)
    VK[inside] = np.where(is_band[:, None], vks[j] + w[:, None] * (vks[j + 1] - vks[j]), 10 ** (lvk[j] + w[:, None] * (lvk[j + 1] - lvk[j])))
    return P, u, s, ph, VK


def build(comps, X, T_axis, lr_grid):
    nr, nT = len(lr_grid), len(T_axis)
    P = np.full((nr, nT), np.nan); U = P.copy(); S = P.copy(); PH = np.zeros((nr, nT), int); CW = P.copy()
    info = dict(overlap_rows=0, dropped=[])
    for jT, T in enumerate(T_axis):
        path = mixture_isotherm(comps, X, T)
        if path is None: continue
        P[:, jT], U[:, jT], S[:, jT], PH[:, jT], VK = onto_grid(path, lr_grid, with_volumes=True)
        CW[:, jT] = wood_cs(comps, X, T, lr_grid, VK)
        info['overlap_rows'] += bool(path['overlaps']); info['dropped'].append(path['dropped'])
    return P, U, S, PH, CW, info


def wood_cs(comps, X, T, lr_grid, VK):
    """Wood's formula over the components, 1/(rho c^2) = sum_k phi_k / (rho_k c_k^2) with volume fractions phi_k = X_k rho / rho_k and each
    component's OWN tabulated sound speed at its state (rho_k, T) -- for a component inside its band that is its own band value. A
    frozen-composition sound speed: a bounded regularisation, NOT the equilibrium two-phase value"""
    rho = 10 ** lr_grid; inv = np.zeros(len(rho)); ok = np.all(np.isfinite(VK), axis=1)
    for k, (c, x) in enumerate(zip(comps, X)):
        rk = np.where(ok, 1.0 / np.where(ok, VK[:, k], 1.0), 1.0)
        ck = 10.0 ** c._interp(c.logcs, *c._cell_v(rk, np.full(len(rk), T)))
        inv += x * (rho / rk) / (rk * ck ** 2)
    return np.where(ok, 1.0 / np.sqrt(rho * inv), np.nan)


def first_law_cs2(lr, lT, P, U):
    """cs^2 = P_rho + P_T (P / rho^2 - u_rho) / u_T, centred differences on the grid (one-sided at the edges)"""
    rho = 10 ** lr[:, None]; T = 10 ** lT[None, :]
    dP_dlr = np.gradient(P, lr, axis=0); dP_dlT = np.gradient(P, lT, axis=1)
    du_dlr = np.gradient(U, lr, axis=0); du_dlT = np.gradient(U, lT, axis=1)
    ln10 = np.log(10.0)
    P_rho = dP_dlr / (rho * ln10); P_T = dP_dlT / (T * ln10); u_rho = du_dlr / (rho * ln10); u_T = du_dlT / (T * ln10)
    return P_rho + P_T * (P / rho ** 2 - u_rho) / u_T, u_T


def local_cs2(comps, X, T_axis, lr, d=1e-5):
    """first-law cs^2 and cV at every grid node from the mixture construction itself on a small stencil (log T +- d, log rho +- d), so a
    node inside a band is differenced inside that band (the coarse-grid stencil leaves narrow bands and gives cs^2 <= 0 there);
    NaN where a stencil point is undefined"""
    nr, nT = len(lr), len(T_axis); cs2 = np.full((nr, nT), np.nan); cV = cs2.copy()
    ln10 = np.log(10.0); rho = 10 ** lr
    for j, T in enumerate(T_axis):
        p0 = mixture_isotherm(comps, X, T); pm = mixture_isotherm(comps, X, T * 10 ** -d); pp = mixture_isotherm(comps, X, T * 10 ** d)
        if p0 is None or pm is None or pp is None: continue
        P0, U0, _, _ = onto_grid(p0, lr)
        Pr_p, Ur_p, _, _ = onto_grid(p0, lr + d); Pr_m, Ur_m, _, _ = onto_grid(p0, lr - d)
        PT_p, UT_p, _, _ = onto_grid(pp, lr); PT_m, UT_m, _, _ = onto_grid(pm, lr)
        P_rho = (Pr_p - Pr_m) / (2 * d * ln10 * rho); u_rho = (Ur_p - Ur_m) / (2 * d * ln10 * rho)
        P_T = (PT_p - PT_m) / (2 * d * ln10 * T); u_T = (UT_p - UT_m) / (2 * d * ln10 * T)
        cs2[:, j] = P_rho + P_T * (P0 / rho ** 2 - u_rho) / u_T; cV[:, j] = u_T
    return cs2, cV


def lowest_valid(P, U, S):
    """per density: lowest T row from which every node up to the top is defined; -1 if none"""
    good = np.isfinite(P) & np.isfinite(U) & np.isfinite(S) & (P > 0) & (U > 0)
    out = np.full(P.shape[0], -1)
    for i in range(P.shape[0]):
        bad = np.where(~good[i])[0]
        k = bad[-1] + 1 if len(bad) else 0
        out[i] = k if k <= P.shape[1] - 2 else -1
    return out


def write(path, rho, T, P, U, S, C, PH, lvt, mat_id=400):
    nr, nT = P.shape
    rows = [f'{mat_id} {nr} {nT} 7\n', ''.join(f'{v:.10e}\n' for v in rho), ''.join(f'{v:.10e}\n' for v in T)]
    for i in range(nr):
        rows.append(''.join(f'{rho[i]:.10e} {T[j]:.10e} {P[i, j]:.10e} {U[i, j]:.10e} {S[i, j]:.10e} {C[i, j]:.10e} {PH[i, j]:d}\n'
                            for j in range(nT)))
    rows.append('lowest_valid_T_index\n' + ' '.join(str(int(k)) for k in lvt) + '\n')
    text = ''.join(rows)
    assert text.count('\n') == 1 + nr + nT + nr * nT + 2
    with open(path, 'w') as f: f.write(text)
    back = SesameTable(path); assert back.nrho == nr and back.nT == nT
    return back


def make(out, specs, nrho=1431, lrho=(-6.0, np.log10(20.0)), T_from=0, verbose=True, accept_u_order_pairs=False):
    """specs = [(table path, X)]; T axis = that of specs[T_from]; returns (SesameTable of the written file, report lines).
    Refuses (ValueError, nothing written) a table with pressure decreasing in density, a written sound speed undefined or <= 0, or
    densities missing inside the axis; u not rising in T (where one energy names several temperatures) is refused unless
    accept_u_order_pairs (e.g. pairs inherited from a component that has them; the run stops on a state with several roots there)."""
    comps = [SesameTable(p) for p, _ in specs]; X = np.array([x for _, x in specs], float)
    if abs(X.sum() - 1) > 1e-12 or np.any(X < 0):
        raise ValueError(f'mass fractions must be >= 0 and sum to 1 (got {X})')
    T_axis = comps[T_from].T_axis; lr = np.linspace(lrho[0], lrho[1], nrho)
    P, U, S, PH, CW, info = build(comps, X, T_axis, lr)
    lvt = lowest_valid(P, U, S)
    finite = np.isfinite(P) & (P > 0) & np.isfinite(U) & (U > 0)
    lost = int(np.sum(finite & (np.arange(len(T_axis))[None, :] < np.where(lvt < 0, len(T_axis), lvt)[:, None])))
    # densities with no valid row at all are dropped (the format needs a row) -- only at the ends of the axis: a gap inside it would be
    # interpolated across
    keep = lvt >= 0
    if keep.any():
        inner = np.arange(np.argmax(keep), len(keep) - np.argmax(keep[::-1]))
        if not keep[inner].all():
            raise ValueError(f'refusing {out}: {int(np.sum(~keep[inner]))} densities inside the axis have no valid row '
                             f'(log rho {lr[inner][~keep[inner]].min():.3f} .. {lr[inner][~keep[inner]].max():.3f})')
    rep = [f'mixture {", ".join(f"{os.path.basename(p)} {x:g}" for p, x in specs)}: grid {nrho} rho x {len(T_axis)} T, '
           f'rho {10 ** lr[0]:.3g}-{10 ** lr[-1]:.3g} g/cc, T {T_axis[0]:.3g}-{T_axis[-1]:.3g} K; densities with no valid row: {np.sum(~keep)}; '
           f'defined nodes below a density\'s floor (holes cut by the contiguity rule): {lost}']
    lr, P, U, S, PH, CW, lvt = lr[keep], P[keep], U[keep], S[keep], PH[keep], CW[keep], lvt[keep]
    # below each density's floor, copy the floor row (GIZMO returns the floor state there anyway; the values are never interpolated)
    for i in range(len(lr)):
        k = lvt[i]
        if k > 0: P[i, :k], U[i, :k], S[i, :k], PH[i, :k] = P[i, k], U[i, k], S[i, k], PH[i, k]
    cs2, cV = local_cs2(comps, X, T_axis, lr)
    cs2_grid, _ = first_law_cs2(lr, np.log10(T_axis), P, U)
    undefined = ~np.isfinite(cs2)                  # a stencil point off the mixture (its edges): the grid estimate there
    cs2 = np.where(undefined, cs2_grid, cs2); cV = np.where(undefined, np.gradient(U, np.log10(T_axis), axis=1) / (np.log(10) * T_axis[None, :]), cV)
    valid = np.arange(len(T_axis))[None, :] >= lvt[:, None]
    nbad_cs = int(np.sum(valid & ~(cs2 > 0)))
    # cV > 0 ON THE TABLE GIZMO READS: log u linear in log T between nodes -> u must rise node to node along T at every density (valid pairs)
    vp = valid[:, 1:] & valid[:, :-1]; nbad_cv = int(np.sum(vp & ~(np.diff(U, axis=1) > 0)))
    # monotone P(rho) per isotherm on valid nodes
    vv = valid[1:] & valid[:-1]; nonmono = int(np.sum(vv & (np.diff(P, axis=0) < -1e-12 * P[1:])))
    rep.append(f'  valid nodes {valid.sum()}; P(rho) decreasing pairs {nonmono}; u not rising in T (node pairs, = cV <= 0 of the table) {nbad_cv}; first-law (equilibrium) cs^2 <= 0 nodes {nbad_cs}; '
               f'band nodes {np.sum(valid & (PH > 0) & (PH < 9))}; overlap-flag nodes {np.sum(valid & (PH == 9))} on {info["overlap_rows"]} isotherms; '
               f'cs from the coarse grid (stencil off the mixture) at {np.sum(valid & undefined)} valid nodes')
    # WRITTEN sound speed = Wood over the components (regularisation; continuous at band edges); the first-law value is the check
    C = CW.copy()
    sp = valid & (PH == 0) & (cs2 > 0); bd = valid & (PH > 0) & (PH < 9) & (cs2 > 0)
    r_sp = np.sqrt(cs2[sp]) / C[sp]; r_bd = np.sqrt(cs2[bd]) / C[bd]
    q = (lambda a, x: np.percentile(a, x) if a.size else np.nan)
    rep.append(f'  written cs = Wood over components; first-law / Wood: single-phase p1 {q(r_sp, 1):.3f} median {q(r_sp, 50):.3f} '
               f'p99 {q(r_sp, 99):.3f}; band (equilibrium / frozen) p1 {np.percentile(r_bd, 1) if bd.any() else np.nan:.3f} median '
               f'{np.median(r_bd) if bd.any() else np.nan:.3f}; Wood undefined / <= 0 at {np.sum(valid & ~(C > 0))} valid nodes')
    for i in range(len(lr)):
        k = lvt[i]
        if k > 0: C[i, :k] = C[i, k]
    bad = ~np.isfinite(P) | ~np.isfinite(U) | ~np.isfinite(S) | ~np.isfinite(C)
    if bad.any():
        raise ValueError(f'refusing {out}: {bad.sum()} non-finite nodes (valid region: {np.sum(bad & valid)})')
    if nonmono:
        raise ValueError(f'refusing {out}: pressure decreases with density at {nonmono} node pairs')
    if np.sum(valid & ~(C > 0)):
        raise ValueError(f'refusing {out}: written sound speed undefined or <= 0 at {int(np.sum(valid & ~(C > 0)))} valid nodes')
    if nbad_cv and not accept_u_order_pairs:
        ii, jj = np.nonzero(vp & ~(np.diff(U, axis=1) > 0))
        raise ValueError(f'refusing {out}: u does not rise with T at {nbad_cv} node pairs (rho {10 ** lr[ii].min():.3g} .. {10 ** lr[ii].max():.3g} g/cc, '
                         f'T {T_axis[jj].min():.4g} .. {T_axis[jj + 1].max():.4g} K): there one energy names several temperatures; '
                         'pass accept_u_order_pairs=True (command line: --accept-u-order-pairs) if they are inherited from a component and accepted')
    back = write(out, 10 ** lr, T_axis, P, U, S, C, PH, lvt)
    rep.append(f'  wrote {out} sha256 {sha256_file(out)}; inputs: ' + ', '.join(f'{os.path.basename(p)} sha256 {sha256_file(p)}' for p, _ in specs))
    if verbose: print('\n'.join(rep))
    return back, rep, dict(lr=lr, T=T_axis, P=P, U=U, S=S, PH=PH, lvt=lvt, cs2=cs2, cV=cV, comps=comps, X=X)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''): h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    a = sys.argv[1:]
    accept = "--accept-u-order-pairs" in a
    a = [x for x in a if x != "--accept-u-order-pairs"]
    if len(a) < 3 or len(a) % 2 == 0:
        sys.exit(__doc__)
    out, specs = a[0], [(a[k], float(a[k + 1])) for k in range(1, len(a), 2)]
    make(out, specs, accept_u_order_pairs=accept)
