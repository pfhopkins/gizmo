"""Uniform EOS dispatch for the IC builder.

A Material binds a backend (Tillotson analytic, or a SESAME table for ANEOS /
CD21 H/He) to a CompositionType integer and a Cv. The HSE shooter only needs
the small set of methods on this class:

    pressure(rho, u_or_T, mode)   -> P
    rho_from_P(P, u_or_T, mode)   -> rho       (mode "u" or "T"; tabulated materials: "T")
    adiabat(P_entry, T_entry)     -> Adiabat (tabulated materials: rho, T, u along the adiabat through the entry state)
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


class Adiabat:
    """(rho, T, u) along one adiabat, interpolated in log10 P. A pressure above the tabulated range returns the
    end state and is counted once per state lookup in n_beyond (the solver reports it); n_dropped counts points
    dropped at construction because the interpolated pressure did not rise (a two-phase band, where P is flat in rho:
    the profile's density then jumps across the band); end says why the adiabat stops where it does."""

    def __init__(self, P, rho, T, u, n_dropped=0, end=""):
        self.logP = np.log10(P); self.logrho = np.log10(rho); self.logT = np.log10(T); self.logu = np.log10(u)
        self.n_beyond = 0; self.n_dropped = n_dropped; self.end = end

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

    def adiabat(self, P_entry, T_entry, route="first_law"):
        """The adiabat through the state (P_entry, T_entry) (SESAME materials only); its entry density is the
        condensed-side root of P(rho, T_entry) = P_entry. route "first_law" (the default): the adiabat GIZMO itself
        follows under slow compression, du = (P / rho^2) drho with P from (rho, u) through the table exactly as a run
        inverts it (adiabatic_first_law). route "entropy": the table's tabulated entropy held at its entry value
        (isentrope_from_entropy), for reproducing ICs built that way; where the table's P, u and S are not mutually
        consistent the two differ."""
        if self.kind != "sesame":
            raise ValueError("adiabat() needs a tabulated (SESAME) material")
        if not (P_entry > 0 and T_entry > 0):
            raise ValueError(f"{self.name}: adiabat entry state needs P > 0 and T > 0 (got {P_entry}, {T_entry})")
        if route == "first_law": return self.adiabat_first_law(P_entry, T_entry)
        if route == "entropy": return self.isentrope_from_entropy(P_entry, T_entry)
        raise ValueError(f"adiabat route {route!r} not in {{first_law, entropy}}")

    def adiabat_first_law(self, P_entry, T_entry, rtol=1e-8):
        """du/dln(rho) = P(rho, u) / rho integrated from the entry state up to the table's highest density, adaptive RK4
        with step doubling (relative error per step <= rtol in u), so a two-phase band (P flat in rho) and the kinks at
        table cells are stepped through. Every P(rho, u) is an exact inversion of the table's interpolant; the adiabat
        ends where an inversion leaves the table (below its lowest valid temperature, above its hottest) or is not
        unique (u not monotone in T at that density), and is refused if that happens at the entry state."""
        tbl = self._backend
        rho0 = float(tbl.rho_from_PT_v(np.array([P_entry]), T_entry)[0])
        u0 = float(tbl.internal_energy_v(np.array([rho0]), np.array([T_entry]))[0])
        def P_of(x, u):                      # (P, T, status) at ln(rho) = x
            T, st = tbl.T_from_u_v(np.array([np.exp(x)]), np.array([u]))
            return float(tbl.pressure_v(np.array([np.exp(x)]), T)[0]), float(T[0]), int(st[0])
        P0, T0, st = P_of(np.log(rho0), u0)
        if st != tbl.INVERT_OK:
            raise ValueError(f"{self.name}: no unique table state at the adiabat entry P={P_entry:.3e}, T={T_entry:.1f} "
                             f"(inversion status {st})")
        x, x_end, u = np.log(rho0), np.log(tbl.rho_axis[-1]), u0
        rec = [(rho0, T0, u0, P0)]; h = 1e-3; end = "the table's highest density"
        def rk4(x, u, h):
            P1, _, s1 = P_of(x, u); k1 = P1 / np.exp(x)
            P2, _, s2 = P_of(x + h / 2, u + h / 2 * k1); k2 = P2 / np.exp(x + h / 2)
            P3, _, s3 = P_of(x + h / 2, u + h / 2 * k2); k3 = P3 / np.exp(x + h / 2)
            P4, _, s4 = P_of(x + h, u + h * k3); k4 = P4 / np.exp(x + h)
            return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), max(s1, s2, s3, s4)
        while x < x_end:
            h = min(h, x_end - x)
            u_full, s_full = rk4(x, u, h)
            u_half, s_a = rk4(x, u, h / 2)
            u_two, s_b = rk4(x + h / 2, u_half, h / 2)
            st = max(s_full, s_a, s_b)
            if st == tbl.INVERT_OK and abs(u_two - u_full) <= rtol * abs(u_two):
                x += h; u = u_two
                P, T, st = P_of(x, u)
                if st != tbl.INVERT_OK: end = f"an inversion failure (status {st})"; break
                rec.append((np.exp(x), T, u, P))
                if abs(u_two - u_full) < 0.1 * rtol * abs(u_two): h = min(2 * h, 0.05)
            else:
                h /= 2
                if h < 1e-9:
                    end = ("an inversion failure (status %d)" % st) if st != tbl.INVERT_OK else "a step-size collapse"
                    break
        rho, T, u, P = (np.array(a) for a in zip(*rec))
        rising = np.concatenate(([True], P[1:] > np.maximum.accumulate(P)[:-1]))
        return Adiabat(P[rising], rho[rising], T[rising], u[rising], n_dropped=int((~rising).sum()), end=end)

    def isentrope_from_entropy(self, P_entry, T_entry, n=800):
        """The isentrope of the table's entropy column through (P_entry, T_entry): S0 = S(entry), tabulated on a density
        grid from the entry up to the table's highest density, T(rho) solving S(rho, T) = S0, then P(rho). It keeps
        only points genuinely on the isentrope (S0 bracketed by the table's temperature range, S reproduced) and ends at
        the first that is not; pressures along it must rise, and points where the interpolated pressure does not (flat
        stretches) are dropped and counted."""
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
        return Adiabat(P[rising], rho[rising], T[rising], tbl.internal_energy_v(rho[rising], T[rising]),
                       n_dropped=int((~rising).sum()), end="the table's highest density" if n_ok == n else "the entropy column's range")

    # --- forward to backend, papering over the (rho,u) vs (rho,T) split ---
    #
    # For Tillotson the natural state variable is u, so mode='T' converts via
    # the analytic Cv = const. For SESAME the natural state is T (tables are
    # (rho,T)-indexed); converting u<->T uses the table itself, NOT a scalar
    # Cv. Tabulated Cv depends on (rho,T), so the old Cv shortcut produced
    # wrong T for hot/dense interiors and pushed downstream HSE inversions
    # off-trajectory. T_from_u below inverts the table's interpolant exactly (SesameTable.T_from_u_v).

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
        # an adiabatic zone of a tabulated material follows its adiabat (Material.adiabat), not a fixed u
        raise ValueError(f"{self.name}: rho_from_P at fixed u is not defined for a tabulated material; use adiabat()")

    def u_from_T(self, rho, T):
        if self.kind == "tillotson":
            return self.Cv * T
        return self._backend.internal_energy(rho, T)

    def T_from_u(self, rho, u):
        if self.kind == "tillotson":
            return u / self.Cv
        # exact inversion of the table's interpolant; a state off the table (below its lowest valid temperature or
        # above its hottest) returns that edge, as a GIZMO lookup does; a non-unique root is refused
        T, st = self._backend.T_from_u_v(np.array([rho]), np.array([u]))
        if st[0] == self._backend.INVERT_NOT_UNIQUE:
            raise ValueError(f"{self.name}: u = {u:.4e} at rho = {rho:.4e} has more than one temperature in this table")
        return float(T[0])

    @property
    def rho_ref(self):
        """Reference density for initial-bracket guesses."""
        if self.kind == "tillotson":
            return _TILLOTSON_PARAMS[self.name][3]
        return self._backend.rho_axis[len(self._backend.rho_axis) // 2]
