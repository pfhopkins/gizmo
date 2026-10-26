"""Kernel density estimator + internal-energy calibration step for the IC
builder.

Why this exists:  the HSE solver gives each particle (rho_HSE, P_HSE, u_HSE)
such that EOS(rho_HSE, u_HSE) = P_HSE.  GIZMO will then recompute the density
at each particle position from its kernel, which differs from rho_HSE
(especially near the center of a centrally-concentrated body, at material
boundaries and at the free surface), so EOS(rho_GIZMO, u_HSE) != P_HSE and the
body is no longer in HSE.

The fix is to compute GIZMO's own density at IC time and set u per particle so
that EOS(rho_GIZMO, u) = P_HSE(r).  The body is then in HSE with respect to the
density GIZMO will actually use.

The estimator is GIZMO's meshless (MFM/MFV) kernel density: rho_i = m_i *
sum_j W(r_ij, h_i) (self included), with the kernel radius h_i solving
(4 pi / 3) h_i^3 sum_j W(r_ij, h_i) = DesNumNgb, for the cubic-spline kernel.
GIZMO stops its own h iteration anywhere inside the band DesNumNgb +-
MaxNumNgbDeviation, so the two agree to that tolerance; for the exact values
GIZMO will use, run GIZMO on the placed IC for zero time and pass its
snapshot_000 densities to calibrate_u (the two-step route in the User Guide).
"""

import numpy as np
from scipy.spatial import cKDTree

# Cubic-spline kernel normalization (3D, support radius h)
_CUBIC_SPLINE_NORM_3D = 8.0 / np.pi


def _W_cubic(q):
    """Cubic spline kernel value at q = r/h, 3D normalization (no 1/h^3)."""
    out = np.zeros_like(q)
    m1 = q < 0.5
    m2 = (q >= 0.5) & (q < 1.0)
    out[m1] = 1.0 - 6.0 * q[m1] ** 2 + 6.0 * q[m1] ** 3
    out[m2] = 2.0 * (1.0 - q[m2]) ** 3
    return _CUBIC_SPLINE_NORM_3D * out


def estimate_sph_density(coords, masses, des_num_ngb=32, n_candidates=None, rtol=1e-8):
    """GIZMO's meshless kernel density at each particle (cubic spline). Returns rho [g/cm^3] and h.

    For each particle the n_candidates nearest neighbours (default 4 DesNumNgb, enough for a free-surface
    particle whose kernel is half empty) bound the search; h is bisected on the monotone weighted neighbour
    number until it matches DesNumNgb to rtol."""
    coords = np.asarray(coords, float); masses = np.asarray(masses, float); n = len(coords)
    k = min(n, int(n_candidates if n_candidates else 4 * des_num_ngb))
    d, _ = cKDTree(coords).query(coords, k=k)
    def n_ngb(h):
        return (4.0 * np.pi / 3.0) * (_W_cubic(d / h[:, None]).sum(1))
    lo = np.maximum(d[:, 1], 1e-300); hi = d[:, -1].copy()
    short = n_ngb(hi) < des_num_ngb
    if short.any():
        raise ValueError(f"{short.sum()} particles have fewer than DesNumNgb={des_num_ngb} weighted neighbours inside "
                         f"their {k} nearest; raise n_candidates")
    for _ in range(200):
        mid = 0.5 * (lo + hi); below = n_ngb(mid) < des_num_ngb
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        if np.all(hi - lo <= rtol * hi): break
    h = 0.5 * (lo + hi)
    rho = masses * _W_cubic(d / h[:, None]).sum(1) / h ** 3
    return rho, h


def _material_branch_profile(r, prof, zone_of_particle, key):
    """prof[key] at radius r on the particle's own zone (each zone's samples held flat beyond its range, so a
    particle just across an interface reads its own material's value, not the neighbour's)"""
    rr = prof["r"][::-1]; yy = prof[key][::-1]; zz = prof["zone_id"][::-1]
    out = np.empty(len(r))
    for z in np.unique(zone_of_particle):
        s = zone_of_particle == z; on = zz == z
        out[s] = np.interp(np.clip(r[s], rr[on].min(), rr[on].max()), rr[on], yy[on])
    return out


def calibrate_u(coords, masses, comp, rho, prof, zones, verbose=True):
    """u per particle so that EOS(rho_i, u_i) = P_HSE(r_i) at the given densities (GIZMO's).

    Tabulated (SESAME) materials: P(rho_i, T) - P_HSE is scanned on the table's own T axis, within its valid
    region; of the sign changes, the bracket nearest the profile temperature is bisected (the state continuously
    connected to the intended one, not the first root). Analytic (Tillotson) materials: bisection in log u.
    A particle whose target pressure no valid state reaches keeps the closest-pressure state and is counted and
    reported -- never silently clamped. Returns (u, T, noroot)."""
    coords = np.asarray(coords, float); rho = np.asarray(rho, float); comp = np.asarray(comp)
    r = np.linalg.norm(coords - (masses[:, None] * coords).sum(0) / masses.sum(), axis=1)
    P_hse = np.interp(r, prof["r"][::-1], prof["P"][::-1])
    zone_by_ct = {z.material.composition_type: iz for iz, z in enumerate(zones)}
    zone_of = np.array([zone_by_ct[c] for c in comp])
    T_prof = _material_branch_profile(r, prof, zone_of, "T")
    u = np.empty(len(r)); T = np.full(len(r), np.nan); noroot = np.zeros(len(r), bool)
    for iz, z in enumerate(zones):
        s = np.where(zone_of == iz)[0]
        if s.size == 0: continue
        mat = z.material
        if mat.kind == "sesame":
            tbl = mat._backend; lT = tbl.logT
            D = tbl.pressure_v(np.repeat(rho[s], lT.size), np.tile(10 ** lT, s.size)).reshape(s.size, lT.size) - P_hse[s, None]
            i_rho, _ = tbl._locate_v(tbl.logrho, np.log10(rho[s]))
            valid = lT[None, :] >= tbl.logT_floor(i_rho)[:, None]                      # rows inside the valid region
            sc = (np.signbit(D[:, :-1]) != np.signbit(D[:, 1:])) & valid[:, :-1]       # brackets [lT_j, lT_j+1]
            dist = np.abs(0.5 * (lT[:-1] + lT[1:])[None, :] - np.log10(T_prof[s])[:, None])
            dist = np.where(sc, dist, np.inf); jb = np.argmin(dist, 1); ok = np.isfinite(dist[np.arange(s.size), jb])
            lo, hi = lT[jb].copy(), lT[jb + 1].copy()
            if (~ok).any():  # no reachable target: the closest valid pressure on the axis
                jbest = np.argmin(np.where(valid[~ok], np.abs(D[~ok]), np.inf), 1); lo[~ok] = hi[~ok] = lT[jbest]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                same = np.signbit(tbl.pressure_v(rho[s], 10 ** mid) - P_hse[s]) == np.signbit(tbl.pressure_v(rho[s], 10 ** lo) - P_hse[s])
                lo = np.where(same & ok, mid, lo); hi = np.where(~same & ok, mid, hi)
            T[s] = 10 ** (0.5 * (lo + hi)); u[s] = tbl.internal_energy_v(rho[s], T[s]); noroot[s] = ~ok
        else:
            for i in s:
                llo, lhi = 0.0, 16.0   # log10 u [erg/g]
                Plo = mat.pressure(rho[i], 10 ** llo, "u"); Phi = mat.pressure(rho[i], 10 ** lhi, "u")
                if not (Plo <= P_hse[i] <= Phi):
                    noroot[i] = True; u[i] = 10 ** (llo if abs(Plo - P_hse[i]) < abs(Phi - P_hse[i]) else lhi); continue
                for _ in range(80):
                    lm = 0.5 * (llo + lhi)
                    if mat.pressure(rho[i], 10 ** lm, "u") < P_hse[i]: llo = lm
                    else: lhi = lm
                u[i] = 10 ** (0.5 * (llo + lhi))
    if verbose:
        msg = f"  u calibration: {len(r)} particles; no valid state reaching P_HSE: {noroot.sum()}"
        if noroot.any():
            for iz, z in enumerate(zones):
                nz = noroot & (zone_of == iz)
                if nz.any(): msg += f" | {z.material.name}: {nz.sum()} at r/R {np.min(r[nz])/prof['R']:.3f}..{np.max(r[nz])/prof['R']:.3f}"
        print(msg)
    return u, T, noroot


def correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, verbose=False):
    """Set placed['u_p'] so that EOS(rho_GIZMO, u) = P_HSE, with rho_GIZMO from estimate_sph_density.

    Stamps placed['rho_sph'] (and placed['T_p'] for tabulated materials, placed['noroot']) and returns the rms
    |rho_GIZMO/rho_HSE - 1| as a quality metric."""
    rho_sph, h = estimate_sph_density(placed["coords"], placed["masses"], des_num_ngb=des_num_ngb)
    u, T, noroot = calibrate_u(placed["coords"], placed["masses"], placed["comp_p"], rho_sph, prof, zones, verbose=verbose)
    placed["u_p"] = u; placed["rho_sph"] = rho_sph; placed["noroot"] = noroot
    if np.any(np.isfinite(T)): placed["T_p"] = np.where(np.isfinite(T), T, placed.get("T_p", T))
    rel_dev = rho_sph / placed["rho_p"] - 1.0
    if verbose:
        print(f"  kernel-vs-HSE rho: rms dev = {np.sqrt(np.mean(rel_dev**2)):.3e}  max = {np.max(np.abs(rel_dev)):.3e}")
    return float(np.sqrt(np.mean(rel_dev ** 2)))
