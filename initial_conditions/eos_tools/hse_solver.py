"""1-D hydrostatic-equilibrium shooting solver for layered self-gravitating
bodies. Used by Phase 17h IC builder.

Equations (CGS):
    dP/dr   = -G m rho / r^2
    dm/dr   =  4 pi r^2 rho
Adiabatic closure within a zone (T_profile='adiabatic'):
    du/dr   =  P/rho^2 * drho/dr   (first law at ds=0)
Isothermal zone (T_profile='isothermal'):
    T(r) = T_zone fixed; u recovered from material EOS as needed.

Material boundaries: T is held continuous across the boundary. For an
adiabatic outer zone, the boundary T is computed from the local (rho,u) of
the outer-zone integrator and reused as the entry T of the inner zone (i.e.,
each zone runs on its own adiabat starting from the boundary T — standard
layered-planet practice).

Outer iteration: bisect on body radius R such that integrating inward from
r=R with m(R)=M_total reaches m=0 at r=0.
"""

import numpy as np

G_CGS = 6.674e-8


class Zone:
    """Specification of one material zone, ordered outer -> inner.

    inner_mass_frac: M_inside_this_zone's_inner_boundary / M_total.
        Outermost zone's inner_mass_frac = mass fraction of all deeper zones.
        Innermost zone's inner_mass_frac = 0.

    Example for Earth (mantle 68%, core 32%):
        Zone(olivine, inner_mass_frac=0.32)   # mantle, deeper interior is core
        Zone(iron,    inner_mass_frac=0.00)   # innermost
    """

    def __init__(self, material, inner_mass_frac, T_profile="adiabatic", T_iso=None):
        if T_profile not in ("adiabatic", "isothermal"):
            raise ValueError(f"T_profile={T_profile!r} not in {{adiabatic,isothermal}}")
        if T_profile == "isothermal" and T_iso is None:
            raise ValueError("isothermal zone requires T_iso")
        self.material = material
        self.inner_mass_frac = float(inner_mass_frac)
        self.T_profile = T_profile
        self.T_iso = T_iso


def _state_deriv(r, P, m, val, mode, mat):
    """Return (dP/dr, dm/dr, dval/dr, rho) where val is u (adiabatic) or T (isothermal)."""
    rho = mat.rho_from_P(P, val, mode)
    if r <= 0:
        return 0.0, 0.0, 0.0, rho
    dPdr = -G_CGS * m * rho / (r * r)
    dmdr = 4.0 * np.pi * r * r * rho
    if mode == "u":
        # ds=0 -> du = (P/rho²) drho. drho = dP/cs². Use analytic cs² from EOS.
        _, cs2 = mat.pressure_cs2(rho, val, mode)
        cs2 = max(cs2, 1e-30)
        drhodr = dPdr / cs2
        dvaldr = (P / (rho * rho)) * drhodr
    else:
        dvaldr = 0.0  # isothermal
    return dPdr, dmdr, dvaldr, rho


def _rk4_step(r, P, m, val, dr, mode, mat):
    k1P, k1m, k1v, _ = _state_deriv(r, P, m, val, mode, mat)
    k2P, k2m, k2v, _ = _state_deriv(r + 0.5 * dr, P + 0.5 * dr * k1P,
                                    m + 0.5 * dr * k1m, val + 0.5 * dr * k1v, mode, mat)
    k3P, k3m, k3v, _ = _state_deriv(r + 0.5 * dr, P + 0.5 * dr * k2P,
                                    m + 0.5 * dr * k2m, val + 0.5 * dr * k2v, mode, mat)
    k4P, k4m, k4v, _ = _state_deriv(r + dr, P + dr * k3P, m + dr * k3m,
                                    val + dr * k3v, mode, mat)
    Pn = P + (dr / 6.0) * (k1P + 2 * k2P + 2 * k3P + k4P)
    mn = m + (dr / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
    vn = val + (dr / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Pn, mn, vn


def _integrate_inward(R, M_total, T_surface, P_surface, zones, n_steps=4000):
    """Integrate (P, m, val) inward from r=R to r=0. Returns dict of arrays."""
    r = np.linspace(R, 0.0, n_steps + 1)
    dr = r[1] - r[0]  # negative
    P = np.zeros_like(r); m = np.zeros_like(r); rho = np.zeros_like(r)
    u = np.zeros_like(r); T = np.zeros_like(r); zone_id = np.zeros(len(r), dtype=int)

    # Outer surface state. Outer zone is zones[0]; mass_frac_outer is 1.0 by
    # convention there (entire body inward of r=R).
    z = zones[0]
    mat = z.material
    P[0] = P_surface
    m[0] = M_total
    if z.T_profile == "adiabatic":
        u[0] = mat.u_from_T(mat.rho_ref, T_surface); val = u[0]; mode = "u"
    else:
        T[0] = z.T_iso; val = z.T_iso; mode = "T"
    rho[0] = mat.rho_from_P(P[0], val, mode)
    if z.T_profile == "isothermal":
        u[0] = mat.u_from_T(rho[0], z.T_iso); T[0] = z.T_iso
    else:
        T[0] = mat.T_from_u(rho[0], u[0])
    zone_id[0] = 0

    iz = 0
    r_at_mzero = -1.0  # tracks first index where m hit 0; -1 = never

    for k in range(n_steps):
        if r[k] <= 0 or m[k] <= 0:
            if r_at_mzero < 0 and m[k] <= 0:
                r_at_mzero = r[k]
            P[k + 1:], m[k + 1:], u[k + 1:], T[k + 1:], rho[k + 1:] = P[k], 0.0, u[k], T[k], rho[k]
            zone_id[k + 1:] = iz
            break
        P_n, m_n, val_n = _rk4_step(r[k], P[k], m[k], val, dr, mode, mat)
        # Floor pressure to keep EOS inversion sane.
        if P_n < P_surface * 1e-6: P_n = P_surface * 1e-6
        if m_n < 0: m_n = 0.0
        P[k + 1] = P_n; m[k + 1] = m_n; val = val_n
        rho[k + 1] = mat.rho_from_P(P_n, val_n, mode)
        if mode == "u":
            u[k + 1] = val_n; T[k + 1] = mat.T_from_u(rho[k + 1], val_n)
        else:
            T[k + 1] = val_n; u[k + 1] = mat.u_from_T(rho[k + 1], val_n)
        zone_id[k + 1] = iz

        # Zone transition: leave the active zone when m drops below its
        # inner_mass_frac threshold. (For the innermost zone, threshold=0
        # so we never leave.)
        while iz < len(zones) - 1 and m[k + 1] <= zones[iz].inner_mass_frac * M_total:
            T_boundary = T[k + 1]; P_boundary = P[k + 1]
            iz += 1
            z = zones[iz]; mat = z.material
            if z.T_profile == "adiabatic":
                rho_b = mat.rho_from_P(P_boundary, T_boundary, "T")
                val = mat.u_from_T(rho_b, T_boundary); mode = "u"
            else:
                val = z.T_iso; mode = "T"
            zone_id[k + 1] = iz

    return {"r": r, "P": P, "m": m, "rho": rho, "u": u, "T": T,
            "zone_id": zone_id, "r_at_mzero": r_at_mzero}


def solve_hse(M_total, T_surface, P_surface, zones,
              R_lo=None, R_hi=None, tol=1e-3, maxit=40, n_steps=2000, verbose=False):
    """Bisect on body radius R such that m(r=0) = 0. Returns the converged
    integrated profile dict from `_integrate_inward`, plus 'R'."""
    # Auto-bracket if not supplied: use the outer-zone reference density to
    # estimate a uniform-density radius and search 0.2x .. 5x around it.
    if R_lo is None or R_hi is None:
        rho_ref = zones[0].material.rho_ref
        R_uniform = (3.0 * M_total / (4.0 * np.pi * rho_ref)) ** (1.0 / 3.0)
        R_lo = R_lo if R_lo is not None else 0.2 * R_uniform
        R_hi = R_hi if R_hi is not None else 5.0 * R_uniform

    def residual(R):
        prof = _integrate_inward(R, M_total, T_surface, P_surface, zones, n_steps)
        # Signed residual:
        #   R too small (over-compressed) -> m hits 0 at r_at_mzero > 0;
        #     residual = -r_at_mzero/R   (negative)
        #   R right                       -> m hits 0 at r ~ 0; residual ~ 0
        #   R too large (under-compressed) -> m > 0 at r=0;
        #     residual = +m_center/M_total (positive)
        if prof["r_at_mzero"] > 0:
            return -prof["r_at_mzero"] / R, prof
        return prof["m"][-1] / M_total, prof

    res_lo, prof_lo = residual(R_lo)
    res_hi, prof_hi = residual(R_hi)
    if verbose:
        print(f"  bracket: R_lo={R_lo:.3e} resid={res_lo:+.3e}  "
              f"R_hi={R_hi:.3e} resid={res_hi:+.3e}")
    if res_lo * res_hi > 0:
        # Expand the bracket geometrically if monotonic on one side.
        for _ in range(8):
            R_lo *= 0.5; R_hi *= 2.0
            res_lo, prof_lo = residual(R_lo)
            res_hi, prof_hi = residual(R_hi)
            if res_lo * res_hi <= 0: break
        else:
            raise RuntimeError(f"could not bracket R: residuals {res_lo:.3e} {res_hi:.3e}. "
                               "Check zone spec / T_surface / Cv.")

    for it in range(maxit):
        R_mid = 0.5 * (R_lo + R_hi)
        res_mid, prof_mid = residual(R_mid)
        if verbose:
            print(f"  it={it:2d} R={R_mid:.4e}  m_resid={res_mid:+.3e}")
        if res_mid * res_lo < 0:
            R_hi = R_mid; res_hi = res_mid; prof_hi = prof_mid
        else:
            R_lo = R_mid; res_lo = res_mid; prof_lo = prof_mid
        if abs(res_mid) < tol: break
    prof_mid["R"] = R_mid
    prof_mid["M_total"] = M_total
    return prof_mid
