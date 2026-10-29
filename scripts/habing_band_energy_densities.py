#!/usr/bin/env python3
"""Habing (1968) and Draine (1978) ISRF energy densities in the Stella G0 bands.

Spectral forms (Draine 2011, Physics of the ISM, eqs. 12.6-12.7), lam3 = lambda/1000 A:
  Habing:  lam*u_lam = 1e-14 * (-25/6 lam3^3 + 25/2 lam3^2 - 13/3 lam3)   erg cm^-3
  Draine:  lam*u_lam = 6.84e-14 * lam3^-5 * (31.016 lam3^2 - 49.913 lam3 + 19.897)  erg cm^-3
Band energy density u = int u_lam dlam. Cross-check: Habing 6-13.6 eV = 5.29e-14 (Draine 2011 sec. 12.5).
Chemistry unit (SGChem photoinit_ism.F): Draine 1978 shape, scaled so that G0 = 1.7 is the Draine field, i.e. Draine/1.7.
The Draine/1.7 values, rounded to 4 digits, are HABING_EGY_DENSITY_* in cooling/chemcool/chem_interface.h.
"""
from scipy.integrate import quad
import numpy as np

HC_EV_A = 12398.42  # h*c [eV Angstrom]

def habing(lam):
    x = lam / 1000.
    return 1e-14 * (-25./6.*x**3 + 25./2.*x**2 - 13./3.*x) / lam

def draine(lam):
    x = lam / 1000.
    return 6.84e-14 * x**-5 * (31.016*x**2 - 49.913*x + 19.897) / lam

def mmp83(lam):
    """Mathis, Mezger & Panagia 1983 ISRF (D_G = 10 kpc), u_lam [erg cm^-3 A^-1]; 4 pi J_lam in erg cm^-2 s^-1 um^-1."""
    l = lam / 1e4
    if l < 0.0912: j = 0.
    elif l < 0.110: j = 38.57 * l**3.4172
    elif l < 0.134: j = 2.045e-2
    elif l < 0.246: j = 7.115e-4 * l**-1.6678
    else:
        lc = l * 1e-4
        j = sum(W * 4 * np.pi * 2 * 6.62607e-27 * 2.99792458e10**2 / lc**5 / np.expm1(6.62607e-27 * 2.99792458e10 / (lc * 1.380649e-16 * T)) * 1e-4
                for W, T in ((1e-14, 7500), (1e-13, 4000), (4e-13, 3000)))
    return j / 2.99792458e10 * 1e-4

def band(f, e_lo, e_hi):
    return quad(f, HC_EV_A / e_hi, HC_EV_A / e_lo, limit=200)[0]

if __name__ == "__main__":
    bands = [("FUV 6-13.6", 6.0, 13.6), ("PE 8-13.6", 8.0, 13.6), ("LW 11.2-13.6", 11.2, 13.6), ("6-8", 6.0, 8.0)]
    print(f"{'band [eV]':14s} {'Habing':>11s} {'Draine78':>11s} {'Draine/1.7':>11s} {'D/H':>6s}")
    for name, lo, hi in bands:
        h, d = band(habing, lo, hi), band(draine, lo, hi)
        print(f"{name:14s} {h:11.4e} {d:11.4e} {d/1.7:11.4e} {d/h:6.3f}")
    # grain heating: MMP83 absorbed-power shares with the band dust opacities of rt_utilities.cc (PE 720, 6-8 eV 600, NUV 480, OPT 180 cm^2/g)
    m = {k: quad(mmp83, HC_EV_A / hi, HC_EV_A / lo, limit=400, points=[1100, 1340, 2460])[0] for k, lo, hi in
         (("F", 6, 13.6), ("PE", 8, 13.6), ("68", 6, 8), ("N", 3.4, 6), ("O", 0.4, 3.4))}
    a = {"F": 720 * m["PE"] + 600 * m["68"], "N": 480 * m["N"], "O": 180 * m["O"]}
    uF = band(draine, 6.0, 13.6) / 1.7
    for k in ("F", "N", "O"):
        print(f"MMP83 {k}: u = {m[k]:.4e}, absorbed share = {a[k] / sum(a.values()):.4f}, per unit dust G0 = {m[k] * uF / m['F']:.4e}")
