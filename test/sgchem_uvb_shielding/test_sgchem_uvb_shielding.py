"""UV-background shielding factors (SGCHEM_UVB_TREECOL_SHIELDING, src/cooling/sgchem/sgchem_uvb_shield.h): the 10-band attenuation of
the HI, HeI photoionisation and heating rates behind HI, HeI columns against the exact integral over the FG20 z = 6.5 spectrum with
Verner+96 cross sections (within 10 %, 15 % below 0.05, where above 0.01); exactly 1 at zero column; heating per ionisation hardens
behind the column. The stored gas column itself: treecol_pixels::test_gas_column_for_uvb_shielding."""
import subprocess
from os.path import abspath, dirname, join

import numpy as np

HERE = dirname(abspath(__file__))
REPO = abspath(join(HERE, "..", ".."))
RY_EV = 13.6057
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def verner(E, Eth, E0, s0, ya, P, yw, y0, y1):
    x = E / E0 - y0
    y = np.sqrt(x * x + y1 * y1)
    F = ((x - 1) ** 2 + yw ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / ya)) ** (-P)
    return np.where(E >= Eth, s0 * F * 1e-18, 0.0)


THRESH = (13.6, 24.59, 54.42)
SIGMA = (lambda E: verner(E, 13.6, 0.4298, 5.475e4, 32.88, 2.963, 0, 0, 0),
         lambda E: verner(E, 24.59, 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136),
         lambda E: verner(E, 54.42, 1.720, 1.369e4, 32.88, 2.963, 0, 0, 0))


class FG20:
    """exact attenuation of the HI, HeI photoionisation and heating rates behind columns (N_HI, N_HeI, N_HeII), z = 6.5"""

    def __init__(self):
        d = np.loadtxt(join(HERE, "fg20_spectrum_z6_7.txt"))
        self.nu = np.logspace(0, np.log10(2000), 20000)
        J = np.exp(np.interp(np.log(self.nu), np.log(d[:, 0]), np.log(np.maximum(d[:, 2], 1e-300))))
        self.n = J / self.nu
        self.E = self.nu * RY_EV
        self.S = np.array([s(self.E) for s in SIGMA])

    def factors(self, N):
        att = np.exp(-np.minimum((self.S * np.asarray(N)[:, None]).sum(0), 700))
        out = []
        for c in (0, 1):
            w = self.n * self.S[c]
            we = w * (self.E - THRESH[c])
            out.append((TRAPZ(w * att, self.nu) / TRAPZ(w, self.nu), TRAPZ(we * att, self.nu) / TRAPZ(we, self.nu)))
        return out


def factor_rows(tmp_path):
    exe = tmp_path / "uvb_shield_factors"
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Werror", f"-I{REPO}", join(HERE, "uvb_shield_factors.cc"), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    return [{k: float(v) for k, v in (kv.split("=") for kv in l.split()[1:])} for l in out.splitlines() if l.startswith("FAC ")]


def test_uvb_shield_factors(tmp_path):
    fac, ref, fails = factor_rows(tmp_path), FG20(), []
    for d in fac:
        N = (d["NHI"], d["NHeI"], 0.0)
        if N[0] == 0:
            assert all(d[k] == 1.0 for k in ("fgHI", "fgHeI", "fgHeII", "fqHI", "fqHeI", "fqHeII")), d
            continue
        ex = ref.factors(N)
        for c, (kg, kq) in enumerate((("fgHI", "fqHI"), ("fgHeI", "fqHeI"))):
            for k, e in ((kg, ex[c][0]), (kq, ex[c][1])):
                tol = 0.10 if e > 0.05 else 0.15
                if e > 0.01 and abs(d[k] / e - 1) > tol:
                    fails.append("N_HI=%.3g N_HeI=%.3g %s=%.4g, FG20 spectrum %.4g" % (N[0], N[1], k, d[k], e))
    for he in (0, 1):  # heating per ionisation rises behind the column
        r = [d["fqHI"] / d["fgHI"] for d in sorted(fac, key=lambda d: d["NHI"]) if d["NHI"] > 0 and (d["NHeI"] > 0) == he and d["fgHI"] > 1e-6]
        if any(b < a * (1 - 1e-9) for a, b in zip(r, r[1:])) or r[-1] < 2.0:
            fails.append("HI heating per ionisation does not harden behind the column: %s" % r)
    assert not fails, "\n".join(fails)
