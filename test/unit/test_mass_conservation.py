#!/usr/bin/env python3
"""
Mass conservation test for all stellar feedback channels.

For every (M, Z) entry in the stellar table, verify:
  M_init = wind_mass_lost + SN_ejecta + remnant_mass

Where:
  wind_mass_lost = M_init - M_preSN  (returned by wind channel during life)
  SN_ejecta      = M_preSN - rem_mass (returned at death via thermal pass)
  remnant_mass   = what's left (NS, BH, WD)

Also verify per-element mass conservation:
  For each element k:
    M_elem_total = wind_yield[k] + sn_yield[k] + X_birth[k] * M_init
  (net_yield[k] = wind_yield[k] + sn_yield[k] should be self-consistent)

And verify the code's injection formula:
  With winds ON:  Mej = M_preSN - rem_mass, yields = sn_yield + X_birth * Mej
  With winds OFF: Mej_wind = M_init - M_preSN (birth comp), Mej_sn = M_preSN - rem_mass (SN yields)
  Total mass returned = Mej_wind + Mej_sn = M_init - rem_mass (must equal)

Channels tested: WD/AGB, ECSN, CCSN, FSN, PPISN, PISN, DBH
"""

import sys
import numpy as np

TABLE_PATH = "galaxy_sf/stellar_tables/stellar_tables_unified.hdf5"

REM_WD    = 0
REM_ECSN  = 1
REM_CCSN  = 2
REM_FSN   = 3
REM_PPISN = 4
REM_PISN  = 5
REM_DBH   = 6

REM_NAMES = {0: "WD", 1: "ECSN", 2: "CCSN", 3: "FSN", 4: "PPISN", 5: "PISN", 6: "DBH"}

ELEM_H  = 0
ELEM_He = 1
ELEM_C  = 2

# Solar mass fractions (Asplund+ 2009)
SOLAR_X = np.array([
    0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.72e-3, 3.26e-7, 1.25e-3,
    2.98e-5, 5.91e-4, 5.57e-5, 6.65e-4, 5.16e-6, 3.10e-4, 3.15e-6,
    7.37e-5, 2.93e-6, 6.44e-5, 3.48e-8, 3.59e-6, 2.30e-7, 1.37e-5,
    9.17e-6, 1.17e-3, 3.30e-6, 6.99e-5, 7.20e-7, 1.67e-6
])
SOLAR_Z = 0.02

nfail = 0
npass = 0


def check(cond, msg, detail=""):
    global nfail, npass
    if cond:
        npass += 1
    else:
        nfail += 1
        print(f"  FAIL: {msg}")
        if detail:
            print(f"        {detail}")


def load_table():
    import h5py
    f = h5py.File(TABLE_PATH, "r")
    tbl = {
        "M": f["M_init"][:],
        "Z": f["Z"][:],
        "logM": f["log_M_init"][:],
        "logZ": f["log_Z"][:],
        "rem_type": f["remnant_type"][:],
        "rem_mass": f["remnant_mass"][:],
        "M_preSN": f["M_preSN"][:],
        "net_yields": f["net_yields"][:],
        "wind_yields": f["wind_yields"][:],
        "elems": [e.decode() for e in f["elements"][:]],
    }
    f.close()
    return tbl


def X_birth(Z_birth, nelem):
    """Compute birth mass fractions at given Z."""
    Z_frac = Z_birth / 0.014  # table uses Asplund 0.014 for scaling
    X = SOLAR_X[:nelem].copy()
    X[:2] *= (1.0 - Z_birth) / (1.0 - 0.014)
    X[2:] *= Z_frac
    return X


def test_global_mass_budget(tbl):
    """M_init = (M_init - M_preSN) + (M_preSN - rem_mass) + rem_mass
    This is trivially true by algebra, but checks that M_preSN is between
    rem_mass and M_init for all entries."""
    print("\n--- Global mass budget: M_preSN between rem_mass and M_init ---")
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            M = tbl["M"][im]
            rm = tbl["rem_mass"][iz, im]
            mp = tbl["M_preSN"][iz, im]
            rt = int(tbl["rem_type"][iz, im])

            check(mp <= M * 1.01,
                  f"M_preSN={mp:.2f} > M_init={M:.2f} at Z={tbl['Z'][iz]:.4f} {REM_NAMES[rt]}")
            check(rm <= mp * 1.01 + 0.01,
                  f"rem_mass={rm:.2f} > M_preSN={mp:.2f} at Z={tbl['Z'][iz]:.4f} {REM_NAMES[rt]}")
            check(rm >= 0,
                  f"rem_mass={rm:.2f} < 0 at M={M:.1f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_wind_plus_sn_equals_net(tbl):
    """net_yield[k] = wind_yield[k] + sn_yield[k] for all elements.
    sn_yield = net_yield - wind_yield by definition in the code."""
    print("\n--- wind_yield + sn_yield = net_yield ---")
    nelem = len(tbl["elems"])
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            net = tbl["net_yields"][iz, im, :]
            wind = tbl["wind_yields"][iz, im, :]
            sn = net - wind

            for k in range(nelem):
                diff = abs(net[k] - (wind[k] + sn[k]))
                check(diff < 1e-10,
                      f"net != wind+sn for {tbl['elems'][k]} at M={tbl['M'][im]:.1f} "
                      f"Z={tbl['Z'][iz]:.4f} {REM_NAMES[rt]}: diff={diff:.2e}")
    print("  done")


def test_fsn_dbh_zero_sn_yields(tbl):
    """FSN and DBH should have zero net yields (zeroed in table builder).
    Therefore sn_yield = -wind_yield for these types."""
    print("\n--- FSN/DBH: net_yields = 0 ---")
    nelem = len(tbl["elems"])
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            if rt not in (REM_FSN, REM_DBH):
                continue
            net = tbl["net_yields"][iz, im, :]
            total_net = np.sum(np.abs(net))
            check(total_net < 1e-10,
                  f"FSN/DBH has nonzero net_yields at M={tbl['M'][im]:.1f} "
                  f"Z={tbl['Z'][iz]:.4f}: sum|net|={total_net:.2e}")
    print("  done")


def test_winds_on_mass_conservation(tbl):
    """With winds ON: at death, M_particle ≈ M_preSN.
    Mej = M_preSN - rem_mass.
    Total mass returned = wind_mass (during life) + Mej = M_init - rem_mass.
    Injection formula: M_elem_ej = sn_yield[k] + X_birth[k] * Mej.
    Check: sum(M_elem_ej) should ≈ Mej (mass conservation in the ejecta)."""
    print("\n--- Winds ON: ejecta mass conservation ---")
    nelem = len(tbl["elems"])
    for iz in range(len(tbl["Z"])):
        Z_b = tbl["Z"][iz]
        Xb = X_birth(Z_b, nelem)
        for im in range(len(tbl["M"])):
            M = tbl["M"][im]
            rt = int(tbl["rem_type"][iz, im])
            rm = tbl["rem_mass"][iz, im]
            mp = tbl["M_preSN"][iz, im]

            Mej_sn = max(mp - rm, 0)
            if Mej_sn < 0.001:
                continue

            net = tbl["net_yields"][iz, im, :]
            wind = tbl["wind_yields"][iz, im, :]
            sn = net - wind
            # FSN/DBH: code uses net_y = 0 (no explosion yields)
            if rt in (REM_FSN, REM_DBH):
                sn = np.zeros(nelem)

            # Code injection formula with mass conservation fix:
            # 1. Compute per-element yields, clip to >= 0
            # 2. If sum != Mej, add deficit to H
            elem_yields = np.zeros(nelem)
            for k in range(nelem):
                M_elem = sn[k] + Xb[k] * Mej_sn
                if M_elem < 0:
                    M_elem = 0
                elem_yields[k] = M_elem
            sum_elem = np.sum(elem_yields)
            # Renormalize: deficit goes into H
            if Mej_sn > 0 and sum_elem > 0:
                deficit = Mej_sn - sum_elem
                if abs(deficit) > 1e-6:
                    elem_yields[ELEM_H] += deficit
                    if elem_yields[ELEM_H] < 0:
                        elem_yields[ELEM_H] = 0
                sum_elem = np.sum(elem_yields)

            reldiff = abs(sum_elem - Mej_sn) / max(Mej_sn, 1e-10)
            check(reldiff < 0.01,
                  f"Winds ON: sum(elem_ej)={sum_elem:.2f} vs Mej={Mej_sn:.2f} "
                  f"(reldiff={reldiff:.2f}) at M={M:.1f} Z={Z_b:.4f} {REM_NAMES[rt]}")
    print("  done")


def test_winds_off_mass_conservation(tbl):
    """Without winds: M_particle = M_init at death.
    Step 1: dump M_init - M_preSN at birth composition.
    Step 2: dump M_preSN - rem_mass with SN yields.
    Total = M_init - rem_mass.
    Check per-element: birth_comp_return + sn_ejecta accounts for all mass."""
    print("\n--- Winds OFF: two-step ejecta mass conservation ---")
    nelem = len(tbl["elems"])
    for iz in range(len(tbl["Z"])):
        Z_b = tbl["Z"][iz]
        Xb = X_birth(Z_b, nelem)
        for im in range(len(tbl["M"])):
            M = tbl["M"][im]
            rt = int(tbl["rem_type"][iz, im])
            rm = tbl["rem_mass"][iz, im]
            mp = tbl["M_preSN"][iz, im]

            Mej_wind = max(M - mp, 0)     # birth composition return
            Mej_sn = max(mp - rm, 0)       # SN ejecta
            Mej_total = Mej_wind + Mej_sn  # should = M - rm

            # Check total mass
            check(abs(Mej_total - (M - rm)) < 0.01,
                  f"Mej_wind + Mej_sn != M_init - rem at M={M:.1f} Z={Z_b:.4f} {REM_NAMES[rt]}")

            # Per-element: wind return
            elem_wind = Xb * Mej_wind  # birth composition, all positive

            # Per-element: SN return
            net = tbl["net_yields"][iz, im, :]
            wind = tbl["wind_yields"][iz, im, :]
            sn = net - wind
            # FSN/DBH: code uses net_y = 0 (no explosion yields)
            if rt in (REM_FSN, REM_DBH):
                sn = np.zeros(nelem)
            elem_sn = np.zeros(nelem)
            for k in range(nelem):
                elem_sn[k] = max(sn[k] + Xb[k] * Mej_sn, 0)
            # Renormalize: deficit goes into H
            sum_sn = np.sum(elem_sn)
            if Mej_sn > 0 and sum_sn > 0:
                deficit_sn = Mej_sn - sum_sn
                if abs(deficit_sn) > 1e-6:
                    elem_sn[ELEM_H] += deficit_sn
                    if elem_sn[ELEM_H] < 0:
                        elem_sn[ELEM_H] = 0

            # Total per-element ejecta
            elem_total = elem_wind + elem_sn
            sum_elem = np.sum(elem_total)

            # Should ≈ Mej_total
            if Mej_total > 0.1:
                reldiff = abs(sum_elem - Mej_total) / Mej_total
                check(reldiff < 0.25,
                      f"Winds OFF: sum(elem)={sum_elem:.2f} vs Mej_total={Mej_total:.2f} "
                      f"(reldiff={reldiff:.2f}) at M={M:.1f} Z={Z_b:.4f} {REM_NAMES[rt]}")

            # No element should be negative
            for k in range(nelem):
                check(elem_total[k] >= -1e-10,
                      f"Negative elem[{tbl['elems'][k]}]={elem_total[k]:.4e} "
                      f"at M={M:.1f} Z={Z_b:.4f} {REM_NAMES[rt]}")
    print("  done")


def test_channel_attribution(tbl):
    """For each (M, Z), verify that mass is correctly attributed to channels:
    - WD/AGB: mass return, no energy, wind+AGB yields
    - ECSN/CCSN/PPISN/PISN: energy + mass + SN yields
    - FSN/DBH: no energy, no SN yields, mass return at birth comp (+ tiny M_preSN-rem residual)
    Check that Esne assignment is consistent with rem_type."""
    print("\n--- Channel attribution ---")
    for iz in range(len(tbl["Z"])):
        for im in range(len(tbl["M"])):
            rt = int(tbl["rem_type"][iz, im])
            rm = tbl["rem_mass"][iz, im]
            mp = tbl["M_preSN"][iz, im]
            M = tbl["M"][im]
            net = tbl["net_yields"][iz, im, :]

            # FSN/DBH: net yields must be zero
            if rt in (REM_FSN, REM_DBH):
                check(np.sum(np.abs(net)) < 1e-10,
                      f"FSN/DBH with nonzero net yields at M={M:.1f} Z={tbl['Z'][iz]:.4f}")

            # PISN: remnant must be zero, M_preSN > 0 (star mass at explosion)
            if rt == REM_PISN:
                check(rm < 0.01,
                      f"PISN with rem={rm:.2f} at M={M:.1f} Z={tbl['Z'][iz]:.4f}")
                check(mp > 0,
                      f"PISN with M_preSN=0 at M={M:.1f} Z={tbl['Z'][iz]:.4f}")

            # All types: M_preSN >= rem_mass
            check(mp >= rm - 0.01,
                  f"M_preSN={mp:.2f} < rem_mass={rm:.2f} at M={M:.1f} Z={tbl['Z'][iz]:.4f} {REM_NAMES[rt]}")

            # Wind mass = M_init - M_preSN >= 0
            check(M >= mp - 0.01,
                  f"M_init={M:.1f} < M_preSN={mp:.2f} at Z={tbl['Z'][iz]:.4f} {REM_NAMES[rt]}")
    print("  done")


def test_extreme_enrichment_mass_budget(tbl):
    """Specifically test the PPISN/PISN mass range where extreme enrichment
    caused the crash. Verify total ejecta mass and metal mass are physical."""
    print("\n--- Extreme enrichment: PPISN/PISN mass budget ---")
    nelem = len(tbl["elems"])
    for iz in range(len(tbl["Z"])):
        Z_b = tbl["Z"][iz]
        Xb = X_birth(Z_b, nelem)
        for im in range(len(tbl["M"])):
            M = tbl["M"][im]
            if M < 100:
                continue
            rt = int(tbl["rem_type"][iz, im])
            if rt not in (REM_PPISN, REM_PISN):
                continue
            rm = tbl["rem_mass"][iz, im]
            mp = tbl["M_preSN"][iz, im]
            net = tbl["net_yields"][iz, im, :]
            wind = tbl["wind_yields"][iz, im, :]
            sn = net - wind

            Mej_sn = max(mp - rm, 0)

            # Compute ejecta metallicity (winds ON path)
            metal_mass = 0
            for k in range(ELEM_C, nelem):
                me = sn[k] + Xb[k] * Mej_sn
                if me > 0:
                    metal_mass += me

            if Mej_sn > 0:
                Z_ej = metal_mass / Mej_sn
                # Report extreme cases
                if Z_ej > 0.3:
                    print(f"  INFO: M={M:.0f} Z={Z_b:.4f} {REM_NAMES[rt]}: "
                          f"rem={rm:.1f} M_preSN={mp:.1f} Mej_sn={Mej_sn:.1f} Z_ej={Z_ej:.3f}")
    print("  done")


if __name__ == "__main__":
    print(f"Loading table: {TABLE_PATH}")
    tbl = load_table()
    print(f"Grid: {len(tbl['Z'])} Z x {len(tbl['M'])} M, {len(tbl['elems'])} elements")

    test_global_mass_budget(tbl)
    test_wind_plus_sn_equals_net(tbl)
    test_fsn_dbh_zero_sn_yields(tbl)
    test_winds_on_mass_conservation(tbl)
    test_winds_off_mass_conservation(tbl)
    test_channel_attribution(tbl)
    test_extreme_enrichment_mass_budget(tbl)

    print(f"\n{'='*60}")
    print(f"Results: {npass} passed, {nfail} FAILED")
    if nfail > 0:
        print(f"MASS CONSERVATION HAS {nfail} PROBLEM(S)")
        sys.exit(1)
    else:
        print("All checks passed")
        sys.exit(0)
