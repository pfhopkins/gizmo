"""Python SPH density estimator + u-correction step for the Phase 17h IC
builder.

Why this exists:  the HSE solver gives each particle (rho_HSE, P_HSE, u_HSE)
such that EOS(rho_HSE, u_HSE) = P_HSE.  GIZMO will then recompute rho_SPH at
each particle position via the SPH kernel.  rho_SPH != rho_HSE in general
(especially near the center of a centrally-concentrated body and at
material boundaries), so EOS(rho_SPH, u_HSE) != P_HSE and the body is no
longer in HSE.

The fix is to compute rho_SPH at IC time in Python and adjust u per particle
so that EOS(rho_SPH, u_corr) = P_HSE.  The body is then in HSE with respect
to the SPH-evaluated density that GIZMO will actually use.

Uses the cubic-spline kernel (GIZMO default) and a single Newton-Raphson
adaptive-h iteration to converge h to satisfy the constant-mass-in-kernel
constraint (DesNumNgb).
"""

import numpy as np

# Cubic-spline kernel normalization (GIZMO default, 3D)
_CUBIC_SPLINE_NORM_3D = 8.0 / np.pi


def _W_cubic(q):
    """Cubic spline kernel value at q = r/h, 3D normalization (no 1/h^3)."""
    out = np.zeros_like(q)
    m1 = q < 0.5
    m2 = (q >= 0.5) & (q < 1.0)
    out[m1] = 1.0 - 6.0 * q[m1] ** 2 + 6.0 * q[m1] ** 3
    out[m2] = 2.0 * (1.0 - q[m2]) ** 3
    return _CUBIC_SPLINE_NORM_3D * out


def estimate_sph_density(coords, masses, des_num_ngb=32, h_init=None,
                         max_iter=12, tol=0.05):
    """Compute SPH density at each particle.  Returns rho [g/cm^3] and h.

    Iterates h until ~des_num_ngb neighbors lie within r<h (i.e., the SPH
    constant-mass-in-kernel constraint).  Naive O(N^2) implementation; fine
    for IC builds up to ~50k particles.
    """
    n = len(coords)
    # Initial h guess: uniform-density estimate from total volume.
    if h_init is None:
        # Span of point cloud
        span = coords.max(axis=0) - coords.min(axis=0)
        V = span[0] * span[1] * span[2]
        h_init = (3.0 * V * des_num_ngb / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    h = np.full(n, h_init)

    # Pairwise distance matrix (O(N^2)).
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    rho = np.zeros(n)
    for it in range(max_iter):
        h_inv = 1.0 / h
        q = d * h_inv[:, None]
        W = _W_cubic(q) * h_inv[:, None] ** 3
        rho = (W * masses[None, :]).sum(axis=1)
        # Effective neighbor count: N = (4/3) pi h^3 rho / m_avg
        m_avg = masses.mean()
        N_eff = (4.0 / 3.0) * np.pi * h ** 3 * rho / m_avg
        ratio = (des_num_ngb / np.maximum(N_eff, 1e-30)) ** (1.0 / 3.0)
        if np.all(np.abs(ratio - 1.0) < tol):
            break
        h *= ratio
    return rho, h


def correct_u_for_sph_density(placed, prof, zones, des_num_ngb=32, verbose=False):
    """Re-evaluate u per particle so that EOS(rho_SPH, u_corr) = P_HSE.

    Modifies placed['u_p'] in-place; also stamps placed['rho_sph'] for
    diagnostics.  Returns the rms |rho_SPH/rho_HSE - 1| as a quality metric.
    """
    coords = placed["coords"]; masses = placed["masses"]
    rho_sph, h = estimate_sph_density(coords, masses, des_num_ngb=des_num_ngb)

    r = np.linalg.norm(coords, axis=1)
    # Interpolate HSE pressure profile at each particle's radius.
    r_grid = prof["r"][::-1]
    P_grid = prof["P"][::-1]
    P_hse = np.interp(r, r_grid, P_grid)

    # For each particle, find u such that material EOS(rho_sph, u) = P_hse.
    # Tillotson is monotone in u for u below shock-vaporization range; bisect.
    n = len(coords)
    u_corr = placed["u_p"].copy()
    n_failed = 0
    for i in range(n):
        zid = placed["comp_p"][i]
        # Find which zone owns this composition_type.
        zone = next((z for z in zones if z.material.composition_type == zid), None)
        if zone is None:
            continue
        mat = zone.material
        # Bisect on u: invert P(rho_sph, u) = P_hse.
        u_lo, u_hi = 1e6, 1e13
        try:
            P_lo = mat.pressure(rho_sph[i], u_lo, "u")
            P_hi = mat.pressure(rho_sph[i], u_hi, "u")
            if P_hse[i] <= P_lo:
                u_corr[i] = u_lo
            elif P_hse[i] >= P_hi:
                u_corr[i] = u_hi
            else:
                for _ in range(60):
                    um = np.sqrt(u_lo * u_hi)
                    Pm = mat.pressure(rho_sph[i], um, "u")
                    if Pm < P_hse[i]:
                        u_lo = um
                    else:
                        u_hi = um
                    if (u_hi / u_lo) - 1.0 < 1e-6:
                        break
                u_corr[i] = np.sqrt(u_lo * u_hi)
        except Exception:
            n_failed += 1

    placed["u_p"] = u_corr
    placed["rho_sph"] = rho_sph
    rho_hse = placed["rho_p"]
    rel_dev = (rho_sph / rho_hse - 1.0)
    if verbose:
        print(f"  SPH-vs-HSE rho: rms dev = {np.sqrt(np.mean(rel_dev**2)):.3e}  "
              f"max = {np.max(np.abs(rel_dev)):.3e}  "
              f"u-correction failures: {n_failed}/{n}")
    return float(np.sqrt(np.mean(rel_dev ** 2)))
