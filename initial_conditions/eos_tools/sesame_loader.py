"""SESAME-format ANEOS-style table loader for IC builder.

Same table contract as eos/aneos.cc::aneos_read_table and the lookups in eos/aneos.h:
dense 2D grids in (log10 rho, log10 T) with arbitrarily spaced, strictly increasing
axes, columns rho, T, P, u, S [, cs [, phase]] (CGS). A query is bracketed against the
actual axis values; log10 P/u/cs are interpolated bilinearly in (log10 rho, log10 T)
and entropy (signed) linearly in value. A table that aneos.cc would refuse is refused
here too.

Table file layout (matches cms_to_sesame.py, convert_stewart_table.py and aneos.cc):
    line 1:  mat_id  nrho  nT  ncols
    next nrho values: rho [g/cm^3];  next nT values: T [K]
    next nrho*nT rows (outer rho, inner T): rho T P u S [cs] [phase]
    ncols = 5 (no cs), 6 (with cs) or 7 (with cs and an integer phase flag, read and ignored)
    optional: the token lowest_valid_T_index, then nrho integers -- per density, the lowest temperature
    row the EOS is valid at; below it a lookup returns the state at that row, as in aneos.h
"""

import numpy as np

AXIS_MATCH_RTOL = 1.0e-8   # ANEOS_AXIS_MATCH_RTOL in aneos.cc


class SesameTable:
    """Bilinear interpolator on an arbitrarily spaced (log10 rho, log10 T) grid."""

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            tok = f.read().split()
        def refuse(why):
            raise ValueError(f"refusing table {path}: {why}")
        if len(tok) < 4:
            refuse("header unreadable")
        self.mat_id, self.nrho, self.nT, self.ncols = (int(x) for x in tok[:4])
        nrho, nT, cols = self.nrho, self.nT, self.ncols
        if nrho < 2 or nT < 2:
            refuse("fewer than two points on an axis")
        if cols not in (5, 6, 7):
            refuse("ncols must be 5, 6 or 7")
        n = nrho * nT
        n_data = 4 + nrho + nT + n * cols
        self.lowest_valid_T = None
        if len(tok) == n_data + 1 + nrho and tok[n_data] == "lowest_valid_T_index":
            self.lowest_valid_T = np.array(tok[n_data + 1:], dtype=int)
            if np.any(self.lowest_valid_T < 0) or np.any(self.lowest_valid_T > nT - 2):
                refuse("lowest valid row outside [0, nT-2]")
        elif len(tok) != n_data:
            refuse(f"expected {n_data} values (or the lowest_valid_T_index block after them), found {len(tok)}")
        vals = np.array(tok[4:n_data], dtype=float)
        self.rho_axis = vals[:nrho]
        self.T_axis = vals[nrho:nrho + nT]
        flat = vals[nrho + nT:].reshape(n, cols)
        for name, axis in (("density", self.rho_axis), ("temperature", self.T_axis)):
            if not (np.all(np.isfinite(axis)) and np.all(axis > 0)):
                refuse(f"{name} axis value not positive and finite")
            if not np.all(np.diff(axis) > 0) or not np.all(np.diff(np.log10(axis)) > 0):
                refuse(f"{name} axis not strictly increasing (in value and in log10)")
        # each row repeats its (rho, T); they must match the axes
        if not (np.allclose(flat[:, 0], np.repeat(self.rho_axis, nT), rtol=AXIS_MATCH_RTOL, atol=0)
                and np.allclose(flat[:, 1], np.tile(self.T_axis, nrho), rtol=AXIS_MATCH_RTOL, atol=0)):
            refuse("data-row (rho, T) do not match the table axes")
        if not np.all(np.isfinite(flat[:, 2:min(cols, 6)])):
            refuse("non-finite value in data rows")

        self.logrho = np.log10(self.rho_axis)
        self.logT = np.log10(self.T_axis)
        # P, u, cs in log10 with zero/negative values floored at 1e-30, as in aneos.cc; entropy signed, as a value
        floor_log = lambda a: np.log10(np.where(a > 0, a, 1e-30))
        self.logP = floor_log(flat[:, 2]).reshape(nrho, nT)
        self.logu = floor_log(flat[:, 3]).reshape(nrho, nT)
        self.S = flat[:, 4].reshape(nrho, nT)
        self.has_cs = cols >= 6
        self.logcs = floor_log(flat[:, 5]).reshape(nrho, nT) if self.has_cs else None

    @staticmethod
    def _locate(axis, x):
        # Clamp to the axis, then bracket against the actual axis values; at a knot the interval above it
        # is selected (below it at the last knot), as in aneos.h.
        x = min(max(x, axis[0]), axis[-1])
        i = int(np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2))
        t = min(1.0, max(0.0, (x - axis[i]) / (axis[i + 1] - axis[i])))
        return i, t

    def logT_floor(self, i):
        """lowest valid log10(T) in density interval i (arrays allowed): the higher of its two columns' bounds"""
        if self.lowest_valid_T is None:
            return np.full(np.shape(i), self.logT[0]) if np.ndim(i) else self.logT[0]
        k = np.maximum(self.lowest_valid_T[i], self.lowest_valid_T[np.asarray(i) + 1])
        return self.logT[k]

    def below_valid(self, rho, T):
        """True where (rho, T) lies below the table's valid region (a lookup there returns the edge state)"""
        i, _ = self._locate_v(self.logrho, np.log10(np.asarray(rho, float)))
        return np.log10(np.asarray(T, float)) < self.logT_floor(i)

    def _cell(self, rho, T):
        i, t = self._locate(self.logrho, np.log10(rho))
        return (i, t) + self._locate(self.logT, max(np.log10(T), self.logT_floor(i)))

    @staticmethod
    def _interp(table, i, t, j, u):
        return ((1 - t) * (1 - u) * table[i, j] + t * (1 - u) * table[i + 1, j]
                + (1 - t) * u * table[i, j + 1] + t * u * table[i + 1, j + 1])

    def _dlog_dlogT(self, table, i, t, j, u):
        return ((1 - t) * (table[i, j + 1] - table[i, j]) + t * (table[i + 1, j + 1] - table[i + 1, j])) / (self.logT[j + 1] - self.logT[j])

    def _dlog_dlogrho(self, table, i, t, j, u):
        return ((1 - u) * (table[i + 1, j] - table[i, j]) + u * (table[i + 1, j + 1] - table[i, j + 1])) / (self.logrho[i + 1] - self.logrho[i])

    def _bilin_log(self, lr, lT, table):
        i, t = self._locate(self.logrho, lr)
        j, u = self._locate(self.logT, lT)
        return self._interp(table, i, t, j, u)

    def pressure(self, rho, T):
        """P(rho, T) [dyn/cm^2]."""
        return 10.0 ** self._interp(self.logP, *self._cell(rho, T))

    def internal_energy(self, rho, T):
        """u(rho, T) [erg/g]."""
        return 10.0 ** self._interp(self.logu, *self._cell(rho, T))

    def entropy(self, rho, T):
        """S(rho, T) [erg/g/K], signed."""
        return self._interp(self.S, *self._cell(rho, T))

    def soundspeed(self, rho, T):
        """cs(rho, T) [cm/s]: the tabulated value, or for a table without a cs column the same estimate
        aneos_compute_P makes, cs^2 = (dP/drho)_T + T (dP/dT)^2 / (rho^2 Cv), from the exact slopes of the cell."""
        c = self._cell(rho, T)
        if self.has_cs:
            return 10.0 ** self._interp(self.logcs, *c)
        rho_c = 10.0 ** np.clip(np.log10(rho), self.logrho[0], self.logrho[-1])
        T_c = 10.0 ** np.clip(np.log10(T), self.logT_floor(c[0]), self.logT[-1])
        P = 10.0 ** self._interp(self.logP, *c); u = 10.0 ** self._interp(self.logu, *c)
        cv = (u / T_c) * self._dlog_dlogT(self.logu, *c)
        cv = cv if cv > 0 else 1e-10
        dP_dT = (P / T_c) * self._dlog_dlogT(self.logP, *c)
        dP_drho = (P / rho_c) * self._dlog_dlogrho(self.logP, *c)
        cs2 = dP_drho + T_c * dP_dT * dP_dT / (rho_c * rho_c * cv)
        return np.sqrt(cs2) if cs2 > 0 else 0.0

    # ---- vectorised forms (arrays of points), same interpolation as the scalar ones above ----
    @staticmethod
    def _locate_v(axis, x):
        x = np.clip(x, axis[0], axis[-1])
        i = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, len(axis) - 2)
        t = np.clip((x - axis[i]) / (axis[i + 1] - axis[i]), 0.0, 1.0)
        return i, t

    def _cell_v(self, rho, T):
        i, t = self._locate_v(self.logrho, np.log10(rho))
        j, u = self._locate_v(self.logT, np.maximum(np.log10(T), self.logT_floor(i)))
        return i, t, j, u

    def pressure_v(self, rho, T):
        return 10.0 ** self._interp(self.logP, *self._cell_v(rho, T))

    def internal_energy_v(self, rho, T):
        return 10.0 ** self._interp(self.logu, *self._cell_v(rho, T))

    def entropy_v(self, rho, T):
        return self._interp(self.S, *self._cell_v(rho, T))

    def rho_from_PT_v(self, P, T, n_iter=64):
        """rho with P(rho, T) = P for arrays of (P, T): bisection in log10(rho) over the table axis. P is
        non-decreasing in rho at fixed T, so this converges to the upper edge of the set P(rho) < P -- the
        condensed-side root when P lies above a flat (two-phase or floored) stretch."""
        P = np.asarray(P, float); T = np.broadcast_to(np.asarray(T, float), P.shape)
        lo = np.full(P.shape, self.logrho[0]); hi = np.full(P.shape, self.logrho[-1])
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            below = self.pressure_v(10.0 ** mid, T) < P
            lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        return 10.0 ** (0.5 * (lo + hi))

    def isentrope_T(self, rho_grid, S0, n_iter=64):
        """T on the isentrope S = S0 at each density of rho_grid. At fixed density the entropy rises with T
        (Cv > 0), so a bisection in log10(T) over the whole table axis finds the unique solution -- including
        inside a two-phase region, which in a (rho, T) table is just an interior point. Also returns whether
        S0 lies within [S(rho, T_min), S(rho, T_max)] at each density (otherwise the result is pinned to an end)."""
        rho_grid = np.asarray(rho_grid, float)
        lo = np.full(rho_grid.shape, self.logT[0]); hi = np.full(rho_grid.shape, self.logT[-1])
        bracketed = (self.entropy_v(rho_grid, 10.0 ** lo) <= S0) & (S0 <= self.entropy_v(rho_grid, 10.0 ** hi))
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            below = self.entropy_v(rho_grid, 10.0 ** mid) < S0
            lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        return 10.0 ** (0.5 * (lo + hi)), bracketed

    def rho_from_PT(self, P, T, rho_lo=None, rho_hi=None, tol=1e-8, maxit=80):
        """Invert P(rho, T) at fixed T via bisection. Monotone in rho on the
        compressive branch; clamps to table range."""
        rho_lo = rho_lo if rho_lo is not None else self.rho_axis[0]
        rho_hi = rho_hi if rho_hi is not None else self.rho_axis[-1]
        Plo = self.pressure(rho_lo, T)
        Phi = self.pressure(rho_hi, T)
        if P <= Plo: return rho_lo
        if P >= Phi: return rho_hi
        for _ in range(maxit):
            rm = 0.5 * (rho_lo + rho_hi)
            Pm = self.pressure(rm, T)
            if Pm < P: rho_lo = rm
            else: rho_hi = rm
            if (rho_hi - rho_lo) / max(rm, 1e-300) < tol: break
        return 0.5 * (rho_lo + rho_hi)
