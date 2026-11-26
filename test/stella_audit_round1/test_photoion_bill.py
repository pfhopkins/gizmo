"""STL-409: the subgrid photoionisation bills the He+ electrons that network 17 sets in a flagged cell
(GALSF_RESOLVEDISM_PHOTOION_BILL_HE_ELECTRONS, network 17 only; Uli 2026-11-26).

One-zone check of the bill, on the zoom configuration (network 17, SGCHEM_PRIMORDIAL_HELIUM): the driver evaluates the
walk's bill for nine (rho, M) pairs. Reference: alpha_B n_e n_H V with n_H = X rho / m_p, V = M / rho and
n_e = n_H (1 + n_He/n_H), n_He/n_H = 0.079 (He+ = abhe in sgchem_set_ionized; Y = 0.24 with X = 0.76).
Bar (fixed before the run): |bill / reference - 1| <= 1e-12 for every pair. Expected without the flag: 1/1.079, 7.3 % low
(the reference 7.9 % above the bill)."""

import subprocess

import pytest

from audit_round1_common import binary, driver, kv

ABHE_NET17 = 0.079
BAR = 1e-12


def check(arm):
    flags = ("GALSF_RESOLVEDISM_PHOTOION_BILL_HE_ELECTRONS",) if arm == "on" else ()
    binary("zoom_" + arm, base="Config_stella_zoom.sh", add=flags, drivers=("bill_driver",))
    r = subprocess.run([driver("zoom_" + arm, "bill_driver")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rows = kv(r.stdout, "BILL")
    assert len(rows) == 9
    worst = 0.0
    for b in rows:
        a, rho, M, X, mp, bill = (float(b[k]) for k in ("alpha", "rho", "M", "X", "mp", "bill"))
        n_H = X * rho / mp
        ref = a * n_H * (1 + ABHE_NET17) * n_H * (M / rho)
        if abs(bill / ref - 1) > abs(worst):
            worst = bill / ref - 1
    print(f"photoionisation bill ({arm}): worst bill / reference - 1 = {worst:+.6e} (X = {rows[0]['X']})")
    assert abs(worst) <= BAR


def test_bill_he_electrons_flag_on():
    check("on")


@pytest.mark.xfail(strict=True, reason="STL-409 without the flag: n_e = n_H, the bill is 1/1.079 of alpha_B n_e n_H V")
def test_bill_he_electrons_flag_off():
    check("off")
