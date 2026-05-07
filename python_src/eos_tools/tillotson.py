"""Analytic Tillotson EOS evaluator for Phase 17h IC builder.

Direct port of the formula in declarations/cell_data.h::calculate_tillotson_eos
and the material parameter table in solids/elastic_physics.cc::tillotson_eos_init.
All quantities CGS.

Tillotson is parameterized in (rho, u). For the IC builder we also need:
    - rho_from_Pu(P, u)        : invert P(rho, u) at fixed u  (adiabatic step uses this)
    - rho_from_PT(P, T)        : with u = Cv * T              (isothermal mode)
    - sound_speed(rho, u)      : cs from c² = (∂P/∂ρ)|_S returned by the formula

Cv values per material are an approximation (Tillotson has no native T notion).
Defaults below match published planet-IC convention; user may override per-zone.
"""

import numpy as np

# Material parameter table — exact CGS values from
# solids/elastic_physics.cc::tillotson_eos_init (j_t=1..6).
# Columns: a, b, u_0, rho_0, A, B, u_s, u_s_prime, alpha, beta, mu_shear, Y_0
_TILLOTSON_PARAMS = {
    "granite": [0.5, 1.3, 1.60e11, 2.700, 1.80e11, 1.80e11, 3.50e10, 1.80e11, 5.0, 5.0, 2.17e11, 3.8e10],
    "basalt":  [0.5, 1.5, 4.87e12, 2.700, 2.67e11, 2.67e11, 4.72e10, 1.82e11, 5.0, 5.0, 2.27e11, 3.5e10],
    "iron":    [0.5, 1.5, 9.50e10, 7.860, 1.28e12, 1.05e12, 1.42e10, 8.45e10, 5.0, 5.0, 7.75e11, 8.5e10],
    "ice":     [0.3, 0.1, 1.00e11, 0.917, 9.47e10, 9.47e10, 7.730e9, 3.04e10, 10., 5.0, 2.80e10, 1.0e10],
    "olivine": [0.5, 1.4, 5.50e12, 3.500, 1.31e12, 4.90e11, 4.50e10, 1.50e11, 5.0, 5.0, 8.13e11, 9.0e10],
    "water":   [0.5, 0.9, 2.00e10, 1.000, 2.00e11, 1.00e11, 4.000e9, 2.00e10, 5.0, 5.0, 1.000e0, 1.00e0],
}

# Default specific heats Cv [erg/g/K] used to relate u <-> T when an isothermal
# zone profile is requested. Tillotson has no native temperature.
_DEFAULT_CV = {
    "granite": 7.9e6, "basalt": 8.4e6, "iron": 4.5e6,
    "ice": 2.1e7, "olivine": 8.3e6, "water": 4.2e7,
}


class TillotsonEOS:
    def __init__(self, material, Cv=None):
        if material not in _TILLOTSON_PARAMS:
            raise ValueError(f"unknown Tillotson material '{material}'; "
                             f"options: {list(_TILLOTSON_PARAMS)}")
        self.material = material
        p = _TILLOTSON_PARAMS[material]
        (self.a, self.b, self.u0, self.rho0, self.A, self.B,
         self.u_s, self.u_s_prime, self.alpha, self.beta,
         self.mu_shear, self.Y0) = p
        self.Cv = Cv if Cv is not None else _DEFAULT_CV[material]

    def pressure_cs2(self, rho, u):
        """Return (P, cs²) at (rho, u). Direct port of cell_data.h formula.
        cs² is dP/drho|_S in CGS; the GIZMO C++ stores cs/rho first and adds
        shear modulus for elastic, but for HSE shooting we want pure dP/dρ|_S
        of the fluid Tillotson, so we drop the elastic shear addition.
        """
        a, b = self.a, self.b
        u0, rho0, A0, B0 = self.u0, self.rho0, self.A, self.B
        u_s, u_sp = self.u_s, self.u_s_prime
        alpha, beta = self.alpha, self.beta

        eta = rho / rho0
        mu = eta - 1.0
        u_u0eta2 = 1.0 + u / (u0 * eta * eta)
        p0 = u * rho
        z = 1.0 / eta - 1.0
        press_min = 1.0e-10 * u0 * rho0

        Pc = (a + b / u_u0eta2) * p0 + A0 * mu + B0 * mu * mu
        c2c_rho = ((1.0 + a + b / u_u0eta2) * Pc + A0 + B0 * (eta * eta - 1.0)
                   + b * (u_u0eta2 - 1.0) * (2.0 * p0 - Pc) / (u_u0eta2 ** 2))

        exp_az2 = np.exp(-alpha * z * z)
        exp_bz = np.exp(-beta * z)
        Pe = a * p0 + (b / u_u0eta2 * p0 + A0 * mu * exp_bz) * exp_az2
        c2e_rho = ((1.0 + a + b / u_u0eta2 * exp_az2) * Pe
                   + A0 * eta * (1.0 + mu * (beta + 2.0 * alpha * z - eta) / (eta * eta)) * exp_bz * exp_az2
                   + b * p0 / (u_u0eta2 ** 2 * eta * eta)
                   * (2.0 * alpha * z * u_u0eta2 * eta + (Pe / (u0 * rho) - 2.0 * u / u0)) * exp_az2)

        if u <= u_s:
            press, c2_rho = Pc, c2c_rho
        elif u >= u_sp:
            press, c2_rho = Pe, c2e_rho
        else:
            press = (Pe * (u - u_s) + Pc * (u_sp - u)) / (u_sp - u_s)
            c2_rho = (c2e_rho * (u - u_s) + c2c_rho * (u_sp - u)) / (u_sp - u_s)

        if c2_rho <= 0:
            c2_rho = press_min
        cs2 = c2_rho / rho  # this is c² = (1/rho) * c²_rho per the C++ definition
        return press, cs2

    def pressure(self, rho, u):
        return self.pressure_cs2(rho, u)[0]

    def soundspeed(self, rho, u):
        return np.sqrt(max(self.pressure_cs2(rho, u)[1], 0.0))

    def rho_from_Pu(self, P, u, rho_lo=None, rho_hi=None, tol=1e-10, maxit=120):
        """Invert P(rho, u) at fixed u via bisection on the compressive branch.
        Default search bracket: [0.01*rho0, 20*rho0]."""
        rho_lo = rho_lo if rho_lo is not None else 0.01 * self.rho0
        rho_hi = rho_hi if rho_hi is not None else 20.0 * self.rho0
        Plo = self.pressure(rho_lo, u)
        Phi = self.pressure(rho_hi, u)
        if P <= Plo: return rho_lo
        if P >= Phi: return rho_hi
        for _ in range(maxit):
            rm = 0.5 * (rho_lo + rho_hi)
            Pm = self.pressure(rm, u)
            if Pm < P: rho_lo = rm
            else: rho_hi = rm
            if (rho_hi - rho_lo) / max(rm, 1e-300) < tol: break
        return 0.5 * (rho_lo + rho_hi)

    def rho_from_PT(self, P, T, **kwargs):
        """Invert with isothermal closure u = Cv * T."""
        return self.rho_from_Pu(P, self.Cv * T, **kwargs)

    def u_from_T(self, T):
        return self.Cv * T

    def T_from_u(self, u):
        return u / self.Cv
