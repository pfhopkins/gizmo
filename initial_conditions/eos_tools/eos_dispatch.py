"""Uniform EOS dispatch for the IC builder.

A Material binds a backend (Tillotson analytic, or a SESAME table for ANEOS /
CD21 H/He) to a CompositionType integer and a Cv. The HSE shooter only needs
the small set of methods on this class:

    pressure(rho, u_or_T, mode)   -> P
    rho_from_P(P, u_or_T, mode)   -> rho       (mode "u" or "T"; tabulated materials: "T")
    isentrope(P_entry, T_entry)   -> Isentrope (tabulated materials: rho, T, u along S = S(entry))
    u_from_T(rho, T)              -> u
    T_from_u(rho, u)              -> T

The CompositionType integer follows eos/composition_registry.h conventions:
    Tillotson presets:  1 granite, 2 basalt, 3 iron, 4 ice, 5 olivine, 6 water
    ANEOS sub-indices:  0,1,2,...  (offset by N_TILLOTSON_MATERIALS=7 in the
                                    dual-flag layout; user supplies absolute id)
"""

import numpy as np

from .tillotson import TillotsonEOS, _TILLOTSON_PARAMS, _DEFAULT_CV
from .sesame_loader import SesameTable

_TILLOTSON_NAME_TO_ID = {
    "granite": 1, "basalt": 2, "iron": 3, "ice": 4, "olivine": 5, "water": 6,
}


class Isentrope:
    """(rho, T, u) along one isentrope, interpolated in log10 P. A pressure above the tabulated range returns the
    end state and is counted once per state lookup in n_beyond (the solver reports it); n_dropped counts points
    dropped at construction because the interpolated pressure did not rise."""

    def __init__(self, P, rho, T, u, S, n_dropped=0):
        self.logP = np.log10(P); self.logrho = np.log10(rho); self.logT = np.log10(T); self.logu = np.log10(u)
        self.S = S; self.n_beyond = 0; self.n_dropped = n_dropped

    def rho(self, P):
        return 10.0 ** np.interp(np.log10(P), self.logP, self.logrho)

    def state(self, P):
        """(rho, T, u) at pressure P."""
        lP = np.log10(P)
        if lP > self.logP[-1]: self.n_beyond += 1
        return tuple(10.0 ** np.interp(lP, self.logP, a) for a in (self.logrho, self.logT, self.logu))


class Material:
    """Backend-agnostic material wrapper used by the HSE solver."""

    def __init__(self, kind, composition_type, name, backend, Cv):
        self.kind = kind                     # "tillotson" | "sesame"
        self.composition_type = composition_type
        self.name = name
        self._backend = backend              # TillotsonEOS or SesameTable
        self.Cv = Cv

    @classmethod
    def tillotson(cls, name, composition_type=None, Cv=None):
        if composition_type is None:
            composition_type = _TILLOTSON_NAME_TO_ID[name]
        eos = TillotsonEOS(name, Cv=Cv)
        return cls("tillotson", composition_type, name, eos, eos.Cv)

    @classmethod
    def sesame(cls, name, table_path, composition_type, Cv):
        tbl = SesameTable(table_path)
        return cls("sesame", composition_type, name, tbl, Cv)

    def isentrope(self, P_entry, T_entry, n=800):
        """The isentrope through the state (P_entry, T_entry) (SESAME materials only). The entry density is the
        condensed-side root of P(rho, T_entry) = P_entry and its entropy S0 defines the isentrope, which is
        tabulated on a density grid from there up to the table's highest density: T(rho) solves S(rho, T) = S0,
        then P(rho) follows. It keeps only points genuinely on the isentrope (S0 bracketed by the table's
        temperature range, S reproduced) and ends at the first that is not; pressures along it must rise,
        and points where the interpolated pressure does not (flat stretches) are dropped and counted."""
        if self.kind != "sesame":
            raise ValueError("isentrope() needs a tabulated (SESAME) material")
        if not (P_entry > 0 and T_entry > 0):
            raise ValueError(f"{self.name}: isentrope entry state needs P > 0 and T > 0 (got {P_entry}, {T_entry})")
        tbl = self._backend
        rho0 = float(tbl.rho_from_PT_v(np.array([P_entry]), T_entry)[0])
        S0 = tbl.entropy(rho0, T_entry)
        rho = np.logspace(np.log10(rho0), np.log10(tbl.rho_axis[-1]), n)
        T, bracketed = tbl.isentrope_T(rho, S0)
        ok = bracketed & (np.abs(tbl.entropy_v(rho, T) - S0) <= 1e-6 * max(abs(S0), 1.0))
        if not ok[0]:
            raise ValueError(f"{self.name}: no valid isentrope through P={P_entry:.3e}, T={T_entry:.1f} in this table")
        n_ok = n if ok.all() else int(np.argmin(ok))
        rho, T = rho[:n_ok], T[:n_ok]
        P = tbl.pressure_v(rho, T)
        rising = np.concatenate(([True], P[1:] > np.maximum.accumulate(P)[:-1]))
        return Isentrope(P[rising], rho[rising], T[rising], tbl.internal_energy_v(rho[rising], T[rising]), S0,
                         n_dropped=int((~rising).sum()))

    # --- forward to backend, papering over the (rho,u) vs (rho,T) split ---
    #
    # For Tillotson the natural state variable is u, so mode='T' converts via
    # the analytic Cv = const. For SESAME the natural state is T (tables are
    # (rho,T)-indexed); converting u<->T uses the table itself, NOT a scalar
    # Cv. Tabulated Cv depends on (rho,T), so the old Cv shortcut produced
    # wrong T for hot/dense interiors and pushed downstream HSE inversions
    # off-trajectory. T_from_u below bisects on the table at fixed rho.

    def pressure(self, rho, val, mode):
        """mode='u': val=internal energy [erg/g]; mode='T': val=temperature [K]."""
        if self.kind == "tillotson":
            u = val if mode == "u" else self.Cv * val
            return self._backend.pressure(rho, u)
        T = val if mode == "T" else self.T_from_u(rho, val)
        return self._backend.pressure(rho, T)

    def pressure_cs2(self, rho, val, mode):
        """Return (P, cs²). Analytic for Tillotson, table-evaluated for SESAME."""
        if self.kind == "tillotson":
            u = val if mode == "u" else self.Cv * val
            return self._backend.pressure_cs2(rho, u)
        T = val if mode == "T" else self.T_from_u(rho, val)
        cs = self._backend.soundspeed(rho, T)
        return self._backend.pressure(rho, T), cs * cs

    def rho_from_P(self, P, val, mode):
        """Return rho such that the EOS yields the requested P at the supplied
        thermodynamic state (val, mode): mode 'u' or 'T' for Tillotson, mode 'T'
        for a tabulated material (bisection on the table at fixed T)."""
        if self.kind == "tillotson":
            u = val if mode == "u" else self.Cv * val
            return self._backend.rho_from_Pu(P, u)
        if mode == "T":
            return self._backend.rho_from_PT(P, val)
        # an adiabatic zone of a tabulated material follows its isentrope (Material.isentrope), not a fixed u
        raise ValueError(f"{self.name}: rho_from_P at fixed u is not defined for a tabulated material; use isentrope()")

    def u_from_T(self, rho, T):
        if self.kind == "tillotson":
            return self.Cv * T
        return self._backend.internal_energy(rho, T)

    def T_from_u(self, rho, u):
        if self.kind == "tillotson":
            return u / self.Cv
        # Newton on T s.t. u_table(rho, T) == u; bisect over table T axis.
        tbl = self._backend
        T_lo, T_hi = tbl.T_axis[0], tbl.T_axis[-1]
        u_lo = tbl.internal_energy(rho, T_lo)
        u_hi = tbl.internal_energy(rho, T_hi)
        if u <= u_lo: return T_lo
        if u >= u_hi: return T_hi
        for _ in range(80):
            Tm = np.sqrt(T_lo * T_hi)  # geometric midpoint (log axis)
            um = tbl.internal_energy(rho, Tm)
            if um < u: T_lo = Tm
            else: T_hi = Tm
            if (T_hi / T_lo) - 1.0 < 1e-8: break
        return np.sqrt(T_lo * T_hi)

    @property
    def rho_ref(self):
        """Reference density for initial-bracket guesses."""
        if self.kind == "tillotson":
            return _TILLOTSON_PARAMS[self.name][3]
        return self._backend.rho_axis[len(self._backend.rho_axis) // 2]
