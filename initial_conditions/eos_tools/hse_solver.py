"""1-D hydrostatic-equilibrium shooting solver for layered self-gravitating
bodies. Used by IC builder.

Equations (CGS):
    dP/dr   = -G m rho / r^2
    dm/dr   =  4 pi r^2 rho
Adiabatic closure within a zone (T_profile='adiabatic'):
    analytic (Tillotson) materials: du/dr = P/rho^2 * drho/dr   (first law at ds=0)
    tabulated (SESAME) materials: the zone follows the adiabat through its entry state
        (surface (P_surface, T_surface), or its top boundary), tabulated once per zone, so
        rho(P) is an interpolation rather than a nested inversion; the entry density is the
        condensed-side root at the entry (P, T). adiabat_route "first_law" (the default):
        du = (P/rho^2) drho with P(rho, u) from the table exactly as GIZMO inverts it, i.e.
        the adiabat a run itself follows; "entropy": the table's entropy column held fixed
        (Material.adiabat).
Isothermal zone (T_profile='isothermal'):
    T(r) = T_zone fixed; u recovered from material EOS as needed.

Material boundaries: P is continuous across the boundary. T is continuous too by
default (standard layered-planet practice), so each zone runs on its own adiabat starting
from the boundary T; a zone may instead enter with a temperature jump dT_top or an explicit
T_top (thermal boundary layers, hot / cold interior variants).

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

    Thermal entry of an adiabatic zone below the outermost one (the outermost enters at T_surface): T continuous across
    its top boundary (default), or T_top = the temperature at its top, or dT_top = the jump at its top (T_top = boundary
    T + dT_top).

    placement_group: consecutive zones with the same (not None) value are placed as ONE domain (shell_placement.seagen_shells):
    the composition bins of a graded interior, which differ in their equation of state but not in how particles are laid out.
    """

    def __init__(self, material, inner_mass_frac, T_profile="adiabatic", T_iso=None, T_top=None, dT_top=None, placement_group=None):
        if T_profile not in ("adiabatic", "isothermal"):
            raise ValueError(f"T_profile={T_profile!r} not in {{adiabatic,isothermal}}")
        if T_profile == "isothermal" and T_iso is None:
            raise ValueError("isothermal zone requires T_iso")
        if T_top is not None and dT_top is not None:
            raise ValueError("give T_top or dT_top, not both")
        if T_profile == "isothermal" and (T_top is not None or dT_top is not None):
            raise ValueError("an isothermal zone takes T_iso, not T_top / dT_top")
        self.material = material
        self.inner_mass_frac = float(inner_mass_frac)
        self.T_profile = T_profile
        self.T_iso = T_iso
        self.T_top = T_top
        self.dT_top = dT_top
        self.placement_group = placement_group

    def entry_T(self, T_boundary):
        """the temperature this zone starts at, given the temperature at the bottom of the zone above"""
        if self.T_top is not None: return float(self.T_top)
        return T_boundary + (self.dT_top or 0.0)


def _state_deriv(r, P, m, val, mode, mat, adb=None):
    """Return (dP/dr, dm/dr, dval/dr, rho) where val is u (adiabatic), T (isothermal), or unused (tabulated adiabat)."""
    rho = adb.rho(P) if mode == "adiabat" else mat.rho_from_P(P, val, mode)
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


def _rk4_step(r, P, m, val, dr, mode, mat, adb=None):
    k1P, k1m, k1v, _ = _state_deriv(r, P, m, val, mode, mat, adb)
    k2P, k2m, k2v, _ = _state_deriv(r + 0.5 * dr, P + 0.5 * dr * k1P,
                                    m + 0.5 * dr * k1m, val + 0.5 * dr * k1v, mode, mat, adb)
    k3P, k3m, k3v, _ = _state_deriv(r + 0.5 * dr, P + 0.5 * dr * k2P,
                                    m + 0.5 * dr * k2m, val + 0.5 * dr * k2v, mode, mat, adb)
    k4P, k4m, k4v, _ = _state_deriv(r + dr, P + dr * k3P, m + dr * k3m,
                                    val + dr * k3v, mode, mat, adb)
    Pn = P + (dr / 6.0) * (k1P + 2 * k2P + 2 * k3P + k4P)
    mn = m + (dr / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
    vn = val + (dr / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Pn, mn, vn


def _integrate_inward(R, M_total, T_surface, P_surface, zones, n_steps=4000, surface_adiabat=None, adiabat_route="first_law"):
    """Integrate (P, m, val) inward from r=R to r=0. Returns dict of arrays.
    surface_adiabat: the outer zone's adiabat, if it is a tabulated adiabatic zone (the same for every R)."""
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
    adb = None; adiabats = []
    if z.T_profile == "adiabatic" and mat.kind == "sesame":
        adb = surface_adiabat if surface_adiabat is not None else mat.adiabat(P_surface, T_surface, adiabat_route)
        val = 0.0; mode = "adiabat"; adb.n_beyond = 0; adiabats.append(adb)
        rho[0], T[0], u[0] = adb.state(P[0])
    else:
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
        # One step, split at a zone's inner mass threshold if the step crosses it: integrate to the crossing
        # (located by linear interpolation of m inside the step), switch to the inner zone there with the boundary
        # (P, T), and finish the step with the inner material. Snapping the switch to the step grid instead makes
        # the core mass, and with it the central structure, jump with R, so the radius bisection cannot converge.
        r_now, P_now, m_now, dr_left = r[k], P[k], m[k], dr
        while True:
            P_n, m_n, val_n = _rk4_step(r_now, P_now, m_now, val, dr_left, mode, mat, adb)
            if iz >= len(zones) - 1 or m_n > zones[iz].inner_mass_frac * M_total:
                break
            thr = zones[iz].inner_mass_frac * M_total
            f = min(1.0, max(0.0, (m_now - thr) / (m_now - m_n))) if m_now > m_n else 1.0
            P_b, m_b, val_b = _rk4_step(r_now, P_now, m_now, val, f * dr_left, mode, mat, adb)
            if mode == "adiabat":
                T_boundary = adb.state(P_b)[1]
            elif mode == "u":
                T_boundary = mat.T_from_u(mat.rho_from_P(P_b, val_b, mode), val_b)
            else:
                T_boundary = val_b
            P_boundary = P_b
            iz += 1
            z = zones[iz]; mat = z.material
            T_boundary = z.entry_T(T_boundary) if z.T_profile == "adiabatic" else T_boundary
            if z.T_profile == "adiabatic" and mat.kind == "sesame":
                try:
                    adb = mat.adiabat(P_boundary, T_boundary, adiabat_route); val = 0.0; mode = "adiabat"; adiabats.append(adb)
                except ValueError as refusal:
                    # no valid adiabat through this boundary state in the table (the trial body is compressed beyond it, or
                    # the state has no unique temperature): score this R as over-compressed, i.e. mass runs out before the
                    # centre, and say so (solve_hse prints how many trial radii this decided)
                    r_at_mzero = r_now + f * dr_left
                    m[k + 1:] = 0.0; P[k + 1:], u[k + 1:], T[k + 1:], rho[k + 1:] = P[k], u[k], T[k], rho[k]
                    zone_id[k + 1:] = iz
                    return {"r": r, "P": P, "m": m, "rho": rho, "u": u, "T": T,
                            "zone_id": zone_id, "r_at_mzero": r_at_mzero,
                            "n_beyond_adiabat": sum(s.n_beyond for s in adiabats), "n_dropped_adiabat": 0, "adiabat_ends": [],
                            "entry_refused": f"{mat.name}: {refusal}"}
            elif z.T_profile == "adiabatic":
                rho_b = mat.rho_from_P(P_boundary, T_boundary, "T")
                val = mat.u_from_T(rho_b, T_boundary); mode = "u"
            else:
                val = z.T_iso; mode = "T"
            r_now, P_now, m_now, dr_left = r_now + f * dr_left, P_b, m_b, (1.0 - f) * dr_left
        if r[k + 1] <= 0:
            # the step into the centre: a regular centre has dP/dr = -(4 pi/3) G rho^2 r, whatever mass the radius
            # bisection leaves (it would otherwise act as a central point mass and send P(0) off the table)
            P_n = P[k] + (2.0 * np.pi / 3.0) * G_CGS * rho[k] ** 2 * r[k] ** 2
            if mode == "u": val_n = val + (P[k] / rho[k] ** 2) * (P_n - P[k]) / max(mat.pressure_cs2(rho[k], val, mode)[1], 1e-30)
        # Floor pressure to keep EOS inversion sane.
        if P_n < P_surface * 1e-6: P_n = P_surface * 1e-6
        if m_n < 0: m_n = 0.0
        P[k + 1] = P_n; m[k + 1] = m_n; val = val_n
        if mode == "adiabat":
            rho[k + 1], T[k + 1], u[k + 1] = adb.state(P_n)
        else:
            rho[k + 1] = mat.rho_from_P(P_n, val_n, mode)
            if mode == "u":
                u[k + 1] = val_n; T[k + 1] = mat.T_from_u(rho[k + 1], val_n)
            else:
                T[k + 1] = val_n; u[k + 1] = mat.u_from_T(rho[k + 1], val_n)
        zone_id[k + 1] = iz   # the zone whose material computed this sample's state

    return {"r": r, "P": P, "m": m, "rho": rho, "u": u, "T": T,
            "zone_id": zone_id, "r_at_mzero": r_at_mzero,
            "n_beyond_adiabat": sum(s.n_beyond for s in adiabats),
            "n_dropped_adiabat": sum(s.n_dropped for s in adiabats),
            "adiabat_ends": [s.end for s in adiabats if s.n_beyond > 0]}


def solve_hse(M_total, T_surface, P_surface, zones,
              R_lo=None, R_hi=None, tol=1e-7, maxit=80, n_steps=2000, verbose=False, adiabat_route="first_law",
              allow_invalid=False):
    """Bisect on body radius R such that m(r=0) = 0. Returns the converged
    integrated profile dict from `_integrate_inward`, plus 'R'. The mass left at the
    centre acts as a point mass on the innermost steps (dP/dr = -G m rho / r^2), so tol
    (the fraction of M_total allowed to remain) must be small: 1e-3 left the central
    pressure of an Earth-mass body ~20x too high. adiabat_route: "first_law" (default) or "entropy", see the module docstring.
    A trial profile is VALID when it is finite, every inner zone's adiabat could start, and no point lay beyond the end of a
    tabulated adiabat (where its end state would stand in for states the table does not have). Invalid trials may steer the
    bisection, but a solution is accepted only when the residual is within tol and the final bracket is formed by valid
    profiles on both sides; otherwise (no convergence, or the EOS-valid domain ends at the root) it raises. allow_invalid=True
    (diagnostics only) returns such a profile instead, with prof["valid"] False and the reasons printed."""
    if zones[0].T_top is not None or zones[0].dT_top is not None:
        raise ValueError("the outermost zone enters at T_surface; T_top / dT_top apply to the zones below it")
    # Auto-bracket if not supplied: a uniform-density radius from the zones' mass-weighted (harmonic) mean density, each
    # material's condensed-side density at 1 GPa and 1000 K (a gas envelope's surface density would put the guess orders of
    # magnitude too wide), searched 0.2x .. 5x around it.
    z0 = zones[0]
    surface_adiabat = (z0.material.adiabat(P_surface, T_surface, adiabat_route)
                       if (z0.T_profile == "adiabatic" and z0.material.kind == "sesame") else None)
    if R_lo is None or R_hi is None:
        outer = np.r_[1.0, [z.inner_mass_frac for z in zones[:-1]]]; frac = outer - np.r_[[z.inner_mass_frac for z in zones]]
        rho_z = [float(z.material._backend.rho_from_PT_v(np.array([1e10]), 1000.0)[0]) if z.material.kind == "sesame"
                 else z.material.rho_ref for z in zones]
        rho_ref = 1.0 / np.sum(frac / np.array(rho_z))
        R_uniform = (3.0 * M_total / (4.0 * np.pi * rho_ref)) ** (1.0 / 3.0)
        R_lo = R_lo if R_lo is not None else 0.2 * R_uniform
        R_hi = R_hi if R_hi is not None else 5.0 * R_uniform

    def residual(R):
        prof = _integrate_inward(R, M_total, T_surface, P_surface, zones, n_steps, surface_adiabat, adiabat_route)
        why = []
        if "entry_refused" in prof: why.append(f"an inner zone's adiabat could not start ({prof['entry_refused']})")
        if prof["n_beyond_adiabat"] > 0:
            why.append(f"{prof['n_beyond_adiabat']} points beyond a tabulated adiabat's range (it ends at "
                       f"{', '.join(sorted(set(prof['adiabat_ends'])))}): the table does not reach this pressure")
        # Signed residual: the mass runs out before the centre -> -r_at_mzero/R (negative); mass is left at the centre ->
        # +m_center/M_total (positive); ~0 at the solution. Which side a too-small R falls on depends on the body: a
        # condensed body with too small a radius cannot hold its mass even at its table's densest state, which leaves mass at
        # the centre, and with too large a radius its near-incompressible outer layers hold all of it; the bisection only
        # needs the sign change.
        res = -prof["r_at_mzero"] / R if prof["r_at_mzero"] > 0 else prof["m"][-1] / M_total
        if not np.isfinite(res):
            raise RuntimeError(f"solve_hse: non-finite profile at trial radius {R:.4e} (check the zone spec and the tables' range)")
        prof["invalid_because"] = why
        return res, prof

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

    converged = False
    for it in range(maxit):
        R_mid = 0.5 * (R_lo + R_hi)
        res_mid, prof_mid = residual(R_mid)
        if verbose:
            print(f"  it={it:2d} R={R_mid:.4e}  m_resid={res_mid:+.3e}" + ("  (invalid)" if prof_mid["invalid_because"] else ""))
        if res_mid * res_lo < 0:
            R_hi = R_mid; res_hi = res_mid; prof_hi = prof_mid
        else:
            R_lo = R_mid; res_lo = res_mid; prof_lo = prof_mid
        if abs(res_mid) < tol: converged = True; break
    problems = []
    if not converged:
        problems.append(f"the radius bisection did not reach tol={tol:.1e} in {maxit} iterations (final residual {res_mid:+.3e}; "
                        "the leftover central mass would act as a point mass)")
    for side, pr in (("the solution", prof_mid), ("the bracket's lower end", prof_lo), ("the bracket's upper end", prof_hi)):
        if pr["invalid_because"]:
            problems.append(f"{side} is not a valid profile: " + "; ".join(pr["invalid_because"]))
    if prof_mid["n_dropped_adiabat"] > 0:
        print(f"  NOTE: {prof_mid['n_dropped_adiabat']} adiabat points dropped where the interpolated pressure did not rise "
              "(two-phase bands: the profile density jumps across them)")
    prof_mid["adiabat_route"] = adiabat_route
    prof_mid["R"] = R_mid
    prof_mid["M_total"] = M_total
    prof_mid["valid"] = not problems
    if problems:
        msg = "solve_hse: no valid hydrostatic solution: " + " | ".join(problems)
        if not allow_invalid:
            raise RuntimeError(msg)
        print("  WARNING (allow_invalid=True, diagnostics only): " + msg)
    return prof_mid


# ---- the particle-free outer atmosphere of an H/He envelope ------------------------------------------------------------
# A tabulated H/He EOS usually starts above a planet's photospheric temperature (CD21 at 100 K; Uranus / Neptune are ~76 K at
# 1 bar). The body's particles then stop at a truncation pressure P_t inside the table, and the gas above it is described by
# an ideal H2 + He adiabat: it sets the envelope's temperature at P_t from the published 1-bar temperature, and gives the
# reported 1-bar radius and the mass the truncation leaves out. It never carries particles.
K_B, M_U = 1.380649e-16, 1.66053907e-24
THETA_ROT_H2, THETA_VIB_H2 = 85.3, 6332.0     # K: rotational and vibrational temperatures of H2


def _cv_levels_over_k(T, E, g):
    """heat capacity / k_B of one molecule over the levels E (in K) with degeneracies g, Boltzmann-populated at T (canonical)"""
    x = np.exp(-(E - E.min()) / T) * g; Z = x.sum(); e1 = (x * E).sum() / Z; e2 = (x * E * E).sum() / Z
    return (e2 - e1 * e1) / (T * T)


def cp_molar_h2(T, ortho_para="equilibrium"):
    """c_P of H2 per molecule / k_B: 5/2 translation + rotation + vibration. ortho_para "equilibrium": the ortho / para ratio
    follows T (conversion included in the heat capacity); "normal": frozen at 3:1, each species with its own levels."""
    J = np.arange(0, 24); E = J * (J + 1) * THETA_ROT_H2; even = J % 2 == 0
    if ortho_para == "equilibrium":
        rot = _cv_levels_over_k(T, E, (2 * J + 1) * np.where(even, 1, 3))
    elif ortho_para == "normal":
        rot = 0.25 * _cv_levels_over_k(T, E[even], (2 * J + 1)[even]) + 0.75 * _cv_levels_over_k(T, E[~even], (2 * J + 1)[~even])
    else:
        raise ValueError(f"ortho_para {ortho_para!r} not in {{equilibrium, normal}}")
    xv = THETA_VIB_H2 / T
    vib = xv * xv * np.exp(xv) / (np.exp(xv) - 1.0) ** 2 if xv < 700 else 0.0
    return 2.5 + rot + vib


def ideal_gas_atmosphere(T_1bar, Y, P_t, M, R_t=None, ortho_para="equilibrium", P_ref=1e6, n=2000):
    """The ideal H2 + He adiabat (helium mass fraction Y) through (P_ref = 1 bar, T_1bar), evaluated at the truncation pressure
    P_t > P_ref: returns the temperature there (the envelope's entry temperature, which must lie inside the H/He table). With the
    body's truncation radius R_t (the HSE radius solved from P_t) it also integrates hydrostatic equilibrium outward from R_t to
    P_ref and returns the 1-bar radius and the gas mass between them (gravity of M, the body mass). Returns a dict (T_t, R_1bar,
    M_above, mu)."""
    if not P_t > P_ref:
        raise ValueError("the truncation pressure must lie below the 1-bar level (P_t > 1 bar)")
    x_he = (Y / 4.002602) / ((1.0 - Y) / 2.01588 + Y / 4.002602)            # number fraction of He among H2 + He molecules
    mu = (1.0 - x_he) * 2.01588 + x_he * 4.002602
    lnP = np.linspace(np.log(P_ref), np.log(P_t), n); lnT = np.empty(n); lnT[0] = np.log(T_1bar)
    nabla = lambda T: 1.0 / ((1.0 - x_he) * cp_molar_h2(T, ortho_para) + x_he * 2.5)   # dlnT/dlnP = k / c_P per molecule
    for k in range(n - 1):                                                  # RK2 in ln P
        h = lnP[k + 1] - lnP[k]; a = nabla(np.exp(lnT[k]))
        lnT[k + 1] = lnT[k] + h * nabla(np.exp(lnT[k] + 0.5 * h * a))
    out = dict(T_t=float(np.exp(lnT[-1])), mu=mu, R_1bar=np.nan, M_above=np.nan)
    if R_t is not None:                                                     # outward from P_t: dr = -dP / (rho g)
        P = np.exp(lnP[::-1]); T = np.exp(lnT[::-1]); rho = P * mu * M_U / (K_B * T)
        r = R_t; m_above = 0.0
        for k in range(n - 1):
            g = G_CGS * M / (r * r); dP = P[k] - P[k + 1]
            dr = dP / (0.5 * (rho[k] + rho[k + 1]) * g)
            m_above += 4.0 * np.pi * r * r * 0.5 * (rho[k] + rho[k + 1]) * dr; r += dr
        out["R_1bar"], out["M_above"] = r, m_above
    return out
