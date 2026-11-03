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
(4 pi / 3) h_i^3 sum_j W(r_ij, h_i) = DesNumNgb, for the kernel the run is
compiled with (KERNEL_FUNCTION). The kernel matters at the percent level:
on a uniform lattice the Wendland C2 kernel reads the density ~1-1.5% high
with 64 neighbours (~0.3% with 125), the cubic spline ~0.1% low with 32.
GIZMO stops its own h iteration anywhere inside the band DesNumNgb +-
MaxNumNgbDeviation, and with HYDRO_KERNEL_SURFACE_VOLCORR it also corrects
the density near free surfaces, which this estimator does not; for the exact
values GIZMO will use, run GIZMO on the placed IC for zero time and calibrate
on its snapshot_000 densities (the two-step route in the User Guide).
"""

import numpy as np
from scipy.spatial import cKDTree

# GIZMO's 3D kernels (src/mesh/kernel.h), by KERNEL_FUNCTION: (normalisation, shape at q = r/h < 1); W = norm * shape / h^3
_KERNELS_3D = {
    3: (8.0 / np.pi, lambda q: np.where(q < 0.5, 1.0 + 6.0 * (q - 1.0) * q * q, 2.0 * (1.0 - q) ** 3)),    # cubic spline
    4: (2187.0 / (40.0 * np.pi), lambda q: (1.0 - q) ** 5 - 6.0 * np.clip(2.0 / 3.0 - q, 0, None) ** 5
        + 15.0 * np.clip(1.0 / 3.0 - q, 0, None) ** 5),                                                    # quartic spline
    5: (15625.0 / (512.0 * np.pi), lambda q: (1.0 - q) ** 4 - 5.0 * np.clip(0.6 - q, 0, None) ** 4
        + 10.0 * np.clip(0.2 - q, 0, None) ** 4),                                                          # quintic spline
    6: (21.0 / (2.0 * np.pi), lambda q: (1.0 - q) ** 4 * (1.0 + 4.0 * q)),                                 # Wendland C2
    7: (495.0 / (32.0 * np.pi), lambda q: (1.0 - q) ** 6 * (1.0 + 6.0 * q + (35.0 / 3.0) * q * q)),       # Wendland C4
    9: (1365.0 / (64.0 * np.pi), lambda q: (1.0 - q) ** 8 * (1.0 + 8.0 * q + 25.0 * q * q + 32.0 * q ** 3)),  # Wendland C6
}


def kernel_W(q, kernel=3):
    """GIZMO's 3D kernel W at q = r/h (without the 1/h^3), for KERNEL_FUNCTION = kernel (3, 4, 5, 6, 7 or 9); 0 for q >= 1"""
    if kernel not in _KERNELS_3D:
        raise ValueError(f"kernel {kernel}: the builder knows GIZMO's KERNEL_FUNCTION {sorted(_KERNELS_3D)}")
    norm, shape = _KERNELS_3D[kernel]
    q = np.asarray(q, float)
    return np.where(q < 1.0, norm * shape(np.minimum(q, 1.0)), 0.0)


def estimate_sph_density(coords, masses, des_num_ngb=32, kernel=3, n_candidates=None, rtol=1e-8, block=200000):
    """GIZMO's meshless kernel density at each particle, for KERNEL_FUNCTION = kernel and DesNumNgb. Returns rho [g/cm^3] and h.

    For each particle the n_candidates nearest neighbours (default 4 DesNumNgb, enough for a free-surface
    particle whose kernel is half empty) bound the search; h is bisected on the monotone weighted neighbour
    number until it matches DesNumNgb to rtol. Particles are processed in blocks of `block`, so the memory
    is bounded (the neighbour distances of one block) whatever the particle number."""
    coords = np.asarray(coords, float); masses = np.asarray(masses, float); n = len(coords)
    k = min(n, int(n_candidates if n_candidates else 4 * des_num_ngb))
    tree = cKDTree(coords)
    rho, h = np.empty(n), np.empty(n)
    for i0 in range(0, n, block):
        s = slice(i0, min(n, i0 + block))
        rho[s], h[s] = _kernel_density_block(tree.query(coords[s], k=k)[0], masses[s], des_num_ngb, kernel, k, rtol)
    return rho, h


def _kernel_density_block(d, masses, des_num_ngb, kernel, k, rtol):
    """estimate_sph_density for one block of particles, given their sorted neighbour distances d"""
    def n_ngb(h):
        return (4.0 * np.pi / 3.0) * (kernel_W(d / h[:, None], kernel).sum(1))
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
    rho = masses * kernel_W(d / h[:, None], kernel).sum(1) / h ** 3
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


def _calibrate_u_table_block(tbl, s, rho, P_hse, T_prof, u, T, noroot):
    """calibrate_u for the particles s of one tabulated material: writes u[s], T[s] and noroot[s]"""
    lT = tbl.logT
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


def calibrate_u(coords, masses, comp, rho, prof, zones, verbose=True, block=50000):
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
        if mat.kind == "sesame":   # in blocks: the scan holds one (particles x table T axis) array per block
            for i0 in range(0, s.size, block):
                _calibrate_u_table_block(mat._backend, s[i0:i0 + block], rho, P_hse, T_prof, u, T, noroot)
        else:   # analytic: bisection in log10 u [erg/g], all particles of the material at once
            r_s, P_s = rho[s], P_hse[s]
            llo, lhi = np.zeros(s.size), np.full(s.size, 16.0)
            Plo = mat.pressure(r_s, 10 ** llo, "u"); Phi = mat.pressure(r_s, 10 ** lhi, "u")
            ok = (Plo <= P_s) & (P_s <= Phi)
            for _ in range(80):
                lm = 0.5 * (llo + lhi); below = mat.pressure(r_s, 10 ** lm, "u") < P_s
                llo = np.where(below, lm, llo); lhi = np.where(below, lhi, lm)
            u[s] = np.where(ok, 10 ** (0.5 * (llo + lhi)), 10 ** np.where(np.abs(Plo - P_s) < np.abs(Phi - P_s), 0.0, 16.0))
            noroot[s] = ~ok
    if verbose:
        msg = f"  u calibration: {len(r)} particles; no valid state reaching P_HSE: {noroot.sum()}"
        if noroot.any():
            for iz, z in enumerate(zones):
                nz = noroot & (zone_of == iz)
                if nz.any(): msg += f" | {z.material.name}: {nz.sum()} at r/R {np.min(r[nz])/prof['R']:.3f}..{np.max(r[nz])/prof['R']:.3f}"
        print(msg)
    return u, T, noroot


def correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, kernel=3, verbose=False):
    """Set placed['u_p'] so that EOS(rho_GIZMO, u) = P_HSE, with rho_GIZMO from estimate_sph_density (KERNEL_FUNCTION kernel).

    Stamps placed['rho_sph'] (and placed['T_p'] for tabulated materials, placed['noroot']) and returns the rms
    |rho_GIZMO/rho_HSE - 1| as a quality metric."""
    rho_sph, h = estimate_sph_density(placed["coords"], placed["masses"], des_num_ngb=des_num_ngb, kernel=kernel)
    u, T, noroot = calibrate_u(placed["coords"], placed["masses"], placed["comp_p"], rho_sph, prof, zones, verbose=verbose)
    placed["u_p"] = u; placed["rho_sph"] = rho_sph; placed["noroot"] = noroot
    if np.any(np.isfinite(T)): placed["T_p"] = np.where(np.isfinite(T), T, placed.get("T_p", T))
    rel_dev = rho_sph / placed["rho_p"] - 1.0
    if verbose:
        print(f"  kernel-vs-HSE rho: rms dev = {np.sqrt(np.mean(rel_dev**2)):.3e}  max = {np.max(np.abs(rel_dev)):.3e}")
    return float(np.sqrt(np.mean(rel_dev ** 2)))


def calibrate_selected_masses(masses, rho_run, target_rho, select, max_change=0.25):
    """Masses for which the run's density of the selected particles equals target_rho, for ICs whose density at a few cells is set by
    neighbours of another kind -- e.g. solid cells beside gas in a gas+solid build, which keep the density they start with (their
    kernel sum counts the gas). rho_run = the run's density at the current masses (GIZMO's zero-time snapshot, matched to these
    particles); the run's number density depends on positions only, so one pass sets the density. Only `select` (a boolean mask)
    changes; a change larger than max_change (relative) stops, since it means the selection is not a thin surface layer. Returns the
    new masses."""
    masses = np.asarray(masses, float); select = np.asarray(select, bool)
    ratio = np.broadcast_to(np.asarray(target_rho, float), masses.shape) / np.asarray(rho_run, float)
    worst = np.max(np.abs(ratio[select] - 1)) if select.any() else 0.0
    if worst > max_change:
        raise ValueError(f"calibrate_selected_masses: a selected particle's mass would change by {worst:.3f} (> max_change {max_change})")
    return np.where(select, masses * ratio, masses)
