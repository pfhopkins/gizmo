#!/usr/bin/env python3
"""Habing (1968) and Draine (1978) ISRF energy densities in the Stella G0 bands.

Spectral forms (Draine 2011, Physics of the ISM, eqs. 12.6-12.7), lam3 = lambda/1000 A:
  Habing:  lam*u_lam = 1e-14 * (-25/6 lam3^3 + 25/2 lam3^2 - 13/3 lam3)   erg cm^-3
  Draine:  lam*u_lam = 6.84e-14 * lam3^-5 * (31.016 lam3^2 - 49.913 lam3 + 19.897)  erg cm^-3
Band energy density u = int u_lam dlam. Cross-check: Habing 6-13.6 eV = 5.29e-14 (Draine 2011 sec. 12.5).
Values, rounded to 3 digits, are HABING_EGY_DENSITY_* in cooling/chemcool/chem_interface.h.
"""
from scipy.integrate import quad

HC_EV_A = 12398.42  # h*c [eV Angstrom]

def habing(lam):
    x = lam / 1000.
    return 1e-14 * (-25./6.*x**3 + 25./2.*x**2 - 13./3.*x) / lam

def draine(lam):
    x = lam / 1000.
    return 6.84e-14 * x**-5 * (31.016*x**2 - 49.913*x + 19.897) / lam

def band(f, e_lo, e_hi):
    return quad(f, HC_EV_A / e_hi, HC_EV_A / e_lo, limit=200)[0]

if __name__ == "__main__":
    bands = [("FUV 6-13.6", 6.0, 13.6), ("PE 8-13.6", 8.0, 13.6), ("LW 11.2-13.6", 11.2, 13.6), ("6-8", 6.0, 8.0)]
    print(f"{'band [eV]':14s} {'Habing':>11s} {'Draine78':>11s} {'D/H':>6s}")
    for name, lo, hi in bands:
        h, d = band(habing, lo, hi), band(draine, lo, hi)
        print(f"{name:14s} {h:11.4e} {d:11.4e} {d/h:6.3f}")
