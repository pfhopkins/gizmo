"""Uniform EOS dispatch for the Phase 17h IC builder.

A Material binds a backend (Tillotson analytic, or a SESAME table for ANEOS /
CD21 H/He) to a CompositionType integer and a Cv. The HSE shooter only needs
the small set of methods on this class:

    pressure(rho, u_or_T, mode)   -> P
    rho_from_P(P, u_or_T, mode)   -> rho       (mode "u" or "T")
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
        thermodynamic state (val,mode). For SESAME mode='u' this is a nested
        inversion: T tracks u along the rho axis, and P(rho, T(rho,u)) is
        bisected on rho. Monotone in rho on the compressive branch, which is
        where the HSE shooter operates."""
        if self.kind == "tillotson":
            u = val if mode == "u" else self.Cv * val
            return self._backend.rho_from_Pu(P, u)
        if mode == "T":
            return self._backend.rho_from_PT(P, val)
        # mode == 'u': bisect rho s.t. pressure(rho, T_from_u(rho,u)) = P.
        u = val
        tbl = self._backend
        rho_lo = tbl.rho_axis[0]
        rho_hi = tbl.rho_axis[-1]
        def _Pfn(rho):
            T = self.T_from_u(rho, u)
            return tbl.pressure(rho, T)
        Plo = _Pfn(rho_lo)
        Phi = _Pfn(rho_hi)
        if P <= Plo: return rho_lo
        if P >= Phi: return rho_hi
        for _ in range(80):
            rm = 0.5 * (rho_lo + rho_hi)
            Pm = _Pfn(rm)
            if Pm < P: rho_lo = rm
            else: rho_hi = rm
            if (rho_hi - rho_lo) / max(rm, 1e-300) < 1e-8: break
        return 0.5 * (rho_lo + rho_hi)

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
