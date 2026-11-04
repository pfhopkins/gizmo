"""Band luminosities of a single star from the stella-tables file, interpolated as GIZMO does
(galaxy_sf/stellar_evolution_tables.cc: bilinear t_PMS and lifetime, trilinear float32 log-luminosity tables
with dead (-99) corners left out)."""
import numpy as np
import h5py

YR_PER_TIME_UNIT = 3.085678e13 / 3.155e7  # pc / (km/s) in the code's years (constants.h)


class Tables:
    def __init__(self, fname):
        with h5py.File(fname, "r") as F:
            self.lZ, self.lM, self.la = F["log_Z"][()], F["log_M_init"][()], F["log_age_yr"][()]
            self.t_pms, self.life = F["t_PMS"][()], F["lifetime_yr"][()]
            self.t3 = {k: F[k][()].astype(np.float32).astype(np.float64)
                       for k in ("log_L_FUV_total", "log_L_FUV_M1", "log_L_LW", "log_L_NUV", "log_L_OPT_NIR", "log_L_ion_tot", "log_Q_ion")}

    @staticmethod
    def _idx(g, v):
        if v <= g[0]:
            return 0, 0.0
        if v >= g[-1]:
            return len(g) - 2, 1.0
        lo = int(np.searchsorted(g, v, side="right") - 1)
        return lo, (v - g[lo]) / (g[lo + 1] - g[lo])

    def _i2(self, a, lM, lZ):
        iz, fz = self._idx(self.lZ, lZ); im, fm = self._idx(self.lM, lM)
        return ((1 - fz) * (1 - fm) * a[iz, im] + (1 - fz) * fm * a[iz, im + 1]
                + fz * (1 - fm) * a[iz + 1, im] + fz * fm * a[iz + 1, im + 1])

    def _i3(self, a, lM, lZ, la):
        iz, fz = self._idx(self.lZ, lZ); im, fm = self._idx(self.lM, lM); ia, fa = self._idx(self.la, la)
        vs = ws = 0.0
        for dz, wz in ((0, 1 - fz), (1, fz)):
            for dm, wm in ((0, 1 - fm), (1, fm)):
                for da, wa in ((0, 1 - fa), (1, fa)):
                    v = a[iz + dz, im + dm, ia + da]
                    if v > -90.0:
                        vs += wz * wm * wa * v; ws += wz * wm * wa
        return vs / ws if ws > 0 else -99.0

    def bands(self, M, Z, age_yr):
        """(PE 8-13.6, LW 11.2-13.6, 6-8, NUV 3.444-6, OPT) [erg/s]; zeros before the main sequence and after death"""
        lM, lZ = np.log10(M), np.log10(max(Z, 1e-10))
        t_pms = self._i2(self.t_pms, lM, lZ)
        if age_yr >= t_pms + self._i2(self.life, lM, lZ) or age_yr - t_pms <= 0:
            return np.zeros(5)
        la = np.log10(max(age_yr - t_pms, 100.0))
        L = {k: 10.0 ** self._i3(a, lM, lZ, la) for k, a in self.t3.items()}
        uv = L["log_L_FUV_total"]; pe = L["log_L_FUV_M1"] if uv > 0 else 0.0
        return np.array([pe, L["log_L_LW"], uv - pe, max(0.0, L["log_L_NUV"] - (uv - pe)), L["log_L_OPT_NIR"]])

    def ionising(self, M, Z, age_yr):
        """ionising energy [erg/s] and photon rate [1/s] on the main sequence"""
        lM, lZ = np.log10(M), np.log10(max(Z, 1e-10))
        la = np.log10(max(age_yr - self._i2(self.t_pms, lM, lZ), 100.0))
        return 10.0 ** self._i3(self.t3["log_L_ion_tot"], lM, lZ, la), 10.0 ** self._i3(self.t3["log_Q_ion"], lM, lZ, la)
