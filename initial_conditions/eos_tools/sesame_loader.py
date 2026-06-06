"""SESAME-format ANEOS-style table loader for Phase 17h IC builder.

Mirrors the loader in eos/aneos.cc::aneos_read_table — tables are dense 2D
grids in (log10 rho, log10 T) on a uniform log-log mesh, columns rho, T, P,
u, S, cs (CGS). Bilinear interpolation in log-log space for P/u/cs.

Table file layout (matches cms_to_sesame.py output and aneos.cc reader):
    line 1:  mat_id  nrho  nT  ncols
    next nrho lines: rho values [g/cm^3]
    next nT   lines: T values   [K]
    next nrho*nT lines (outer rho, inner T): rho T P u S [cs]
    ncols = 5 (no cs) or 6 (with cs).  No phase column for the smooth tables
    we use here; if a phase column is present it is read and ignored.
"""

import numpy as np


class SesameTable:
    """Bilinear interpolator on a uniform (log10 rho, log10 T) grid."""

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            tok = f.read().split()
        idx = 0
        self.mat_id = int(tok[idx]); idx += 1
        self.nrho = int(tok[idx]); idx += 1
        self.nT = int(tok[idx]); idx += 1
        self.ncols = int(tok[idx]); idx += 1
        self.rho_axis = np.array([float(tok[idx + k]) for k in range(self.nrho)])
        idx += self.nrho
        self.T_axis = np.array([float(tok[idx + k]) for k in range(self.nT)])
        idx += self.nT

        # Data block: outer loop rho, inner loop T. Columns: rho T P u S [cs].
        n = self.nrho * self.nT
        cols = self.ncols
        flat = np.array(tok[idx:idx + n * cols], dtype=float).reshape(n, cols)
        self.P = flat[:, 2].reshape(self.nrho, self.nT)
        self.u = flat[:, 3].reshape(self.nrho, self.nT)
        if cols >= 6:
            self.cs = flat[:, 5].reshape(self.nrho, self.nT)
        else:
            # Fallback: cs² ≈ P/ρ * γ (γ≈5/3 sentinel; tables always ship cs).
            self.cs = np.sqrt(5.0 / 3.0 * self.P / self.rho_axis[:, None])

        self.logrho = np.log10(self.rho_axis)
        self.logT = np.log10(self.T_axis)
        self.logP = np.log10(np.maximum(self.P, 1e-300))
        self.logu = np.log10(np.maximum(self.u, 1e-300))
        self.logcs = np.log10(np.maximum(self.cs, 1e-300))

        # Uniform-grid step (matches aneos.cc which assumes uniform log axes).
        self.d_logrho = self.logrho[1] - self.logrho[0]
        self.d_logT = self.logT[1] - self.logT[0]

    def _bilin_log(self, lr, lT, table):
        # Clamp + bilinear in (logrho, logT) on the uniform grid.
        lr = np.clip(lr, self.logrho[0], self.logrho[-1])
        lT = np.clip(lT, self.logT[0], self.logT[-1])
        fi = (lr - self.logrho[0]) / self.d_logrho
        fj = (lT - self.logT[0]) / self.d_logT
        i = int(np.clip(int(fi), 0, self.nrho - 2))
        j = int(np.clip(int(fj), 0, self.nT - 2))
        t = max(0.0, min(1.0, fi - i))
        u = max(0.0, min(1.0, fj - j))
        return ((1 - t) * (1 - u) * table[i, j]
                + t * (1 - u) * table[i + 1, j]
                + (1 - t) * u * table[i, j + 1]
                + t * u * table[i + 1, j + 1])

    def pressure(self, rho, T):
        """P(rho, T) [dyn/cm^2]."""
        return 10.0 ** self._bilin_log(np.log10(rho), np.log10(T), self.logP)

    def internal_energy(self, rho, T):
        """u(rho, T) [erg/g]."""
        return 10.0 ** self._bilin_log(np.log10(rho), np.log10(T), self.logu)

    def soundspeed(self, rho, T):
        """cs(rho, T) [cm/s]."""
        return 10.0 ** self._bilin_log(np.log10(rho), np.log10(T), self.logcs)

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
