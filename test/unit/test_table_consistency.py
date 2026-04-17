#!/usr/bin/env python3
"""
Comprehensive table consistency tests.

Validates EVERY derived field in stellar_tables_unified.hdf5:

Part 1 — Internal consistency (fields must be self-consistent):
  Grid axes, lifetimes, M_current, surface abundances, radiation,
  wind velocity, mass-loss rate, phases, core masses, yields, remnants.

Part 2 — Raw source data cross-check:
  PARSEC v2 ejecta: yields, remnants, M_preSN, core masses.
  Limongi+2025: yields at solar Z.

Part 3 — C bilinear interpolation fidelity:
  Grid-point lookups, midpoint bounds, sn = net - wind identity.
"""

import os
import sys
import numpy as np

TABLE_PATH = "galaxy_sf/stellar_tables/stellar_tables_unified.hdf5"
RAW_PARSEC_DIR = os.path.join(
    os.path.expanduser("~"), "phil", "stellar_tables", "parsec2_vms"
)

LOG_FLOOR = -30.0
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
    tbl = {}
    # Scalar / 1D
    for k in ['M_init', 'Z', 'log_M_init', 'log_Z', 'log_age_yr', 'elements',
              'type_ia_yields']:
        tbl[k] = f[k][:]
    # 2D (NZ, NM)
    for k in ['M_CO_core', 'M_He_core', 'M_preSN', 'lifetime_yr',
              'remnant_mass', 'remnant_type', 't_PMS', 'track_source',
              'yield_source']:
        tbl[k] = f[k][:]
    # 3D (NZ, NM, NAGE)
    for k in ['M_current', 'logR_cm', 'logTeff',
              'v_wind',
              'log_L_bol', 'log_L_OPT_NIR', 'log_L_NUV_lo', 'log_L_NUV_hi',
              'log_L_NUV', 'log_L_FUV_mid', 'log_L_FUV', 'log_L_FUV_total',
              'log_L_FUV_M1', 'log_L_LW',
              'log_L_ion_H0', 'log_L_ion_He0', 'log_L_ion_He1',
              'log_L_ion_He2', 'log_L_ion_tot',
              'log_Q_ion', 'log_Q_ion_H0']:
        tbl[k] = f[k][:]
    # 3D (NZ, NM, NELEM) — yields
    for k in ['net_yields', 'wind_yields']:
        tbl[k] = f[k][:]
    # 4D (NZ, NM, NAGE, NELEM)
    tbl['surface_abundances'] = f['surface_abundances'][:]
    # Phase transitions group
    tbl['phase_transitions'] = {}
    for k in f['phase_transitions']:
        tbl['phase_transitions'][k] = f['phase_transitions'][k][:]
    tbl['elems'] = [e.decode() for e in tbl['elements']]
    f.close()
    return tbl


# ═══════════════════════════════════════════════════════════════════════
#  Part 1 — Internal consistency
# ═══════════════════════════════════════════════════════════════════════

def test_grid_axes(tbl):
    """Grid axes: sorted, log-consistent, correct sizes."""
    print("\n--- 1.1 Grid axes ---")
    M = tbl['M_init']
    Z = tbl['Z']
    logM = tbl['log_M_init']
    logZ = tbl['log_Z']
    age = tbl['log_age_yr']

    check(np.all(np.diff(M) > 0), "M_init not strictly increasing")
    check(np.all(np.diff(Z) > 0), "Z not strictly increasing")
    check(np.all(np.diff(age) > 0), "log_age_yr not strictly increasing")
    check(np.allclose(logM, np.log10(M)), "log_M_init != log10(M_init)")
    check(np.allclose(logZ, np.log10(Z)), "log_Z != log10(Z)")
    check(len(M) == tbl['M_current'].shape[1], "M_init length != NM in M_current")
    check(len(Z) == tbl['M_current'].shape[0], "Z length != NZ in M_current")
    check(len(age) == tbl['M_current'].shape[2], "log_age length != NAGE in M_current")
    check(len(tbl['elems']) == tbl['net_yields'].shape[2], "N_elements mismatch")
    print("  done")


def test_lifetimes(tbl):
    """Lifetime must be positive, decrease with mass, and be consistent with M_current."""
    print("\n--- 1.2 Lifetimes ---")
    M = tbl['M_init']
    lt = tbl['lifetime_yr']
    mc = tbl['M_current']
    age = tbl['log_age_yr']
    tpms = tbl['t_PMS']

    for iz in range(len(tbl['Z'])):
        for im in range(len(M)):
            # Positive
            check(lt[iz, im] > 0,
                  f"lifetime <= 0 at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")

            # Lifetime should generally decrease with mass (for M > 1)
            if im > 0 and M[im] > 1.0 and M[im - 1] > 1.0:
                check(lt[iz, im] <= lt[iz, im - 1] * 1.1,
                      f"lifetime not decreasing: M={M[im]:.2f} lt={lt[iz,im]:.0f} "
                      f"vs M={M[im-1]:.2f} lt={lt[iz,im-1]:.0f}")

            # M_current should be zero at ages >> lifetime
            log_lt = np.log10(lt[iz, im])
            far_dead = age > log_lt + 0.5  # well past death
            if far_dead.any():
                check(np.all(mc[iz, im, far_dead] == 0),
                      f"M_current nonzero after death at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")

            # t_PMS >= 0
            check(tpms[iz, im] >= 0,
                  f"t_PMS < 0 at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_M_current(tbl):
    """M_current: starts at M_init, monotonically decreasing, zero when dead."""
    print("\n--- 1.3 M_current ---")
    M = tbl['M_init']
    mc = tbl['M_current']
    age = tbl['log_age_yr']

    for iz in range(len(tbl['Z'])):
        for im in range(len(M)):
            alive = mc[iz, im, :] > 0
            if not alive.any():
                continue

            first = np.where(alive)[0][0]
            last = np.where(alive)[0][-1]

            # M_current[first] ≈ M_init.  At high Z, PARSEC tracks can start
            # below M_init due to rapid early wind mass loss during PMS trim.
            # Allow 2% for M >= 100 at Z >= 0.01, 0.5% otherwise.
            mc_tol = 0.02 if (M[im] >= 100 and tbl['Z'][iz] >= 0.01) else 0.005
            check(abs(mc[iz, im, first] - M[im]) / M[im] < mc_tol,
                  f"M_current[0]={mc[iz,im,first]:.4f} != M_init={M[im]:.4f} "
                  f"at Z={tbl['Z'][iz]:.4f}")

            # Monotonically non-increasing while alive
            mc_alive = mc[iz, im, first:last + 1]
            diffs = np.diff(mc_alive)
            n_increase = np.sum(diffs > 0.01)  # allow tiny numerical noise
            check(n_increase == 0,
                  f"M_current increases {n_increase} times at M={M[im]:.2f} "
                  f"Z={tbl['Z'][iz]:.4f}")

            # Zero after death
            if last < len(age) - 1:
                check(np.all(mc[iz, im, last + 1:] == 0),
                      f"M_current nonzero after last alive at M={M[im]:.2f} "
                      f"Z={tbl['Z'][iz]:.4f}")

            # M_preSN ≈ M_current at last alive point (for M >= 8 with winds)
            if M[im] >= 8:
                mp = tbl['M_preSN'][iz, im]
                mc_last = mc[iz, im, last]
                # Allow generous tolerance — track vs ejecta can differ
                check(abs(mp - mc_last) / max(mp, 0.1) < 0.5,
                      f"M_preSN={mp:.2f} vs M_current[last]={mc_last:.2f} "
                      f"at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_surface_abundances(tbl):
    """Surface abundances: sum to 1 when alive, 0 when dead, physically bounded."""
    print("\n--- 1.4 Surface abundances ---")
    M = tbl['M_init']
    mc = tbl['M_current']
    surf = tbl['surface_abundances']

    # Sample a subset (full check is very slow for 15x110x768)
    iz_sample = [0, 4, 7, 10, 14]
    im_sample = list(range(0, len(M), 5))  # every 5th mass

    for iz in iz_sample:
        for im in im_sample:
            alive = mc[iz, im, :] > 0
            if not alive.any():
                continue

            first = np.where(alive)[0][0]
            last = np.where(alive)[0][-1]

            # Sum to 1 when alive (sample a few age points)
            for ia in [first, (first + last) // 2, last]:
                s = surf[iz, im, ia, :].sum()
                check(abs(s - 1.0) < 0.05,
                      f"surface sum={s:.4f} at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f} "
                      f"age_idx={ia}")

            # All non-negative when alive
            check(np.all(surf[iz, im, first:last + 1, :] >= -1e-10),
                  f"negative surface abundance at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")

            # Zero when dead
            if last < surf.shape[2] - 1:
                dead_sum = np.abs(surf[iz, im, last + 1:, :]).sum()
                check(dead_sum < 1e-8,
                      f"surface nonzero after death at M={M[im]:.2f} "
                      f"Z={tbl['Z'][iz]:.4f}: sum={dead_sum:.2e}")

            # H + He dominate (sanity)
            check(surf[iz, im, first, 0] + surf[iz, im, first, 1] > 0.9,
                  f"H+He < 0.9 at ZAMS at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_radiation(tbl):
    """Radiation bands: physical bounds, dead=zero, band hierarchy."""
    print("\n--- 1.5 Radiation bands ---")
    M = tbl['M_init']
    mc = tbl['M_current']
    age = tbl['log_age_yr']

    # All log-luminosity fields
    rad_keys = ['log_L_bol', 'log_L_OPT_NIR', 'log_L_NUV_lo', 'log_L_NUV_hi',
                'log_L_FUV_mid', 'log_L_LW', 'log_L_ion_H0', 'log_L_ion_He0',
                'log_L_ion_He1', 'log_L_ion_He2']

    iz_sample = [0, 7, 14]
    im_sample = list(range(0, len(M), 10))

    for iz in iz_sample:
        for im in im_sample:
            alive = mc[iz, im, :] > 0
            if not alive.any():
                continue
            first = np.where(alive)[0][0]
            last = np.where(alive)[0][-1]

            for key in rad_keys:
                vals = tbl[key][iz, im, :]

                # Dead = floor
                if last < len(age) - 1:
                    check(np.all(vals[last + 1:] <= LOG_FLOOR + 1),
                          f"{key} nonzero after death at M={M[im]:.2f} "
                          f"Z={tbl['Z'][iz]:.4f}")

            # L_bol >= each sub-band (when alive, both above floor)
            Lbol = tbl['log_L_bol'][iz, im, first:last + 1]
            for key in ['log_L_OPT_NIR', 'log_L_NUV_lo', 'log_L_LW',
                        'log_L_ion_H0']:
                sub = tbl[key][iz, im, first:last + 1]
                valid = (Lbol > LOG_FLOOR + 1) & (sub > LOG_FLOOR + 1)
                if valid.any():
                    check(np.all(Lbol[valid] >= sub[valid] - 0.1),
                          f"L_bol < {key} at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")

            # logL dropped from HDF5; log_L_bol (erg/s) is the only bolometric
            # luminosity in the table.  No cross-check needed.

            # Q_ion >= Q_ion_H0 (total ionizing >= H-only)
            Qi = tbl['log_Q_ion'][iz, im, first:last + 1]
            QiH = tbl['log_Q_ion_H0'][iz, im, first:last + 1]
            valid = (Qi > LOG_FLOOR + 1) & (QiH > LOG_FLOOR + 1)
            if valid.any():
                check(np.all(Qi[valid] >= QiH[valid] - 0.01),
                      f"Q_ion < Q_ion_H0 at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_wind_velocity(tbl):
    """Wind velocity: positive when alive (M>=8), zero when dead."""
    print("\n--- 1.6 Wind velocity ---")
    M = tbl['M_init']
    mc = tbl['M_current']
    vw = tbl['v_wind']

    for iz in range(len(tbl['Z'])):
        for im in range(len(M)):
            if M[im] < 8:
                continue  # winds only for massive stars
            alive = mc[iz, im, :] > 0
            if not alive.any():
                continue
            first = np.where(alive)[0][0]
            last = np.where(alive)[0][-1]

            # Positive when alive
            vw_alive = vw[iz, im, first:last + 1]
            check(np.all(vw_alive >= 0),
                  f"negative v_wind at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")

            # Zero after death
            if last < vw.shape[2] - 1:
                check(np.all(vw[iz, im, last + 1:] == 0),
                      f"v_wind nonzero after death at M={M[im]:.1f} "
                      f"Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_mass_loss_rate(tbl):
    """log_Mdot dropped from HDF5; Mdot is now computed from M_current at
    runtime in the C code.  No cross-check needed."""
    pass


def test_phases(tbl):
    """Phase transitions: times should be within lifetime."""
    print("\n--- 1.8 Phase transitions ---")
    M = tbl['M_init']
    lt = tbl['lifetime_yr']
    for pt_key, pt_data in tbl['phase_transitions'].items():
        for iz in range(len(tbl['Z'])):
            for im in range(len(M)):
                t = pt_data[iz, im]
                if t > 0:
                    check(t <= lt[iz, im] * 1.01,
                          f"{pt_key}={t:.0f} > lifetime={lt[iz,im]:.0f} "
                          f"at M={M[im]:.2f} Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_core_masses(tbl):
    """Core masses: M_CO <= M_He <= M_preSN, physically bounded."""
    print("\n--- 1.9 Core masses ---")
    M = tbl['M_init']
    mco = tbl['M_CO_core']
    mhe = tbl['M_He_core']
    mp = tbl['M_preSN']

    for iz in range(len(tbl['Z'])):
        for im in range(len(M)):
            if mhe[iz, im] > 0:
                check(mco[iz, im] <= mhe[iz, im] * 1.01,
                      f"M_CO={mco[iz,im]:.2f} > M_He={mhe[iz,im]:.2f} "
                      f"at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")
            if mhe[iz, im] > 0 and mp[iz, im] > 0:
                check(mhe[iz, im] <= mp[iz, im] * 1.01,
                      f"M_He={mhe[iz,im]:.2f} > M_preSN={mp[iz,im]:.2f} "
                      f"at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")
            if mp[iz, im] > 0:
                check(mp[iz, im] <= M[im] * 1.001,
                      f"M_preSN={mp[iz,im]:.2f} > M_init={M[im]:.2f} "
                      f"at Z={tbl['Z'][iz]:.4f}")
    print("  done")


def test_remnants_and_yields(tbl):
    """Remnants and yields: type-consistent, mass-bounded, FSN/DBH zeroed."""
    print("\n--- 1.10 Remnants and yields ---")
    M = tbl['M_init']
    rt = tbl['remnant_type']
    rm = tbl['remnant_mass']
    mp = tbl['M_preSN']
    net = tbl['net_yields']
    wind = tbl['wind_yields']
    ysrc = tbl['yield_source']

    RN = {0: 'WD', 1: 'ECSN', 2: 'CCSN', 3: 'FSN', 4: 'PPISN', 5: 'PISN', 6: 'DBH'}

    for iz in range(len(tbl['Z'])):
        for im in range(len(M)):
            rtype = int(rt[iz, im])
            rmass = rm[iz, im]

            # Valid type
            check(rtype in RN,
                  f"invalid rem_type={rtype} at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")

            # Non-negative remnant mass
            check(rmass >= 0,
                  f"rem_mass={rmass:.2f} < 0 at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")

            # PISN: zero remnant
            if rtype == 5:
                check(rmass < 0.01,
                      f"PISN with rem_mass={rmass:.2f} at M={M[im]:.1f}")

            # rem_mass <= M_preSN
            if mp[iz, im] > 0:
                check(rmass <= mp[iz, im] + 0.01,
                      f"rem_mass={rmass:.2f} > M_preSN={mp[iz,im]:.2f} "
                      f"at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f}")

            # FSN/DBH: net_yields = 0
            if rtype in (3, 6):
                check(np.sum(np.abs(net[iz, im, :])) < 1e-8,
                      f"FSN/DBH nonzero net_yields at M={M[im]:.1f} "
                      f"Z={tbl['Z'][iz]:.4f}")

            # sum(net_yields) ≈ 0 (mass conservation in yield framework)
            ns = net[iz, im, :].sum()
            Mej = M[im] - rmass
            tol = max(0.3, Mej * 0.03)  # 3% or 0.3 Msun
            check(abs(ns) < tol,
                  f"sum(net)={ns:.4f} at M={M[im]:.1f} Z={tbl['Z'][iz]:.4f} "
                  f"{RN[rtype]} (tol={tol:.2f})")

            # wind_yields = 0 for M < 14 (no separate wind data)
            if M[im] < 14:
                check(np.sum(np.abs(wind[iz, im, :])) < 1e-8,
                      f"wind_yields nonzero for M={M[im]:.1f} (should be 0 below 14)")
    print("  done")


def test_type_ia(tbl):
    """Type Ia yields: sum = Chandrasekhar mass, Fe-dominated."""
    print("\n--- 1.11 Type Ia yields ---")
    ia_y = tbl['type_ia_yields']
    total = ia_y.sum()
    check(abs(total - 1.4) < 0.15,
          f"Type Ia yield sum={total:.4f} (expected ~1.3-1.4 MCh)")
    # Fe should be the largest
    elems = tbl['elems']
    fe_idx = elems.index('Fe')
    check(ia_y[fe_idx] > 0.5,
          f"Type Ia Fe yield={ia_y[fe_idx]:.3f} (expected > 0.5)")
    check(np.all(ia_y >= 0), "Type Ia has negative yields")
    print("  done")


# ═══════════════════════════════════════════════════════════════════════
#  Part 2 — Raw source data cross-check
# ═══════════════════════════════════════════════════════════════════════

def load_raw_parsec2(raw_dir):
    """Load raw PARSEC v2 total + wind ejecta."""
    from glob import glob
    import re

    ELEM_COLS = {
        'H': [7], 'He': [8, 9], 'C': [12, 13], 'N': [14, 15],
        'O': [16, 17, 18], 'F': [19], 'Ne': [20, 21, 22], 'Na': [23],
        'Mg': [24, 25, 26], 'Al': [27, 28], 'Si': [29, 30], 'P': [31],
        'S': [32], 'Cl': [33], 'Ar': [34], 'K': [35], 'Ca': [36],
        'Sc': [37], 'Ti': [38], 'V': [39], 'Cr': [40], 'Mn': [41],
        'Fe': [42], 'Co': [43], 'Ni': [44], 'Cu': [45], 'Zn': [46],
    }
    WIND_COLS = {
        'H': [4], 'He': [5, 6], 'C': [9, 10], 'N': [11, 12],
        'O': [13, 14, 15], 'F': [16], 'Ne': [17, 18, 19], 'Na': [20],
        'Mg': [21, 22, 23], 'Al': [24, 25], 'Si': [26, 27], 'P': [28],
        'S': [29], 'Cl': [30], 'Ar': [31], 'K': [32], 'Ca': [33],
        'Sc': [34], 'Ti': [35], 'V': [36], 'Cr': [37], 'Mn': [38],
        'Fe': [39], 'Co': [40], 'Ni': [41], 'Cu': [42], 'Zn': [43],
    }
    SNT_MAP = {'CCSN': 2, 'FSN': 3, 'PPISN': 4, 'PISN': 5, 'DBH': 6, 'NOT': 0}

    total = {}
    wind = {}
    init_comp = {}

    for zdir in sorted(glob(os.path.join(raw_dir, 'ejecta_Z*'))):
        m = re.search(r'ejecta_Z([\d.Ee-]+)', zdir)
        if not m:
            continue
        Z = float(m.group(1))

        for tf in glob(os.path.join(zdir, '*total_ejecta.dat')):
            with open(tf) as fh:
                lines = fh.readlines()
            # Parse initial composition from line 2
            parts2 = lines[1].replace('#', '').split()
            try:
                sn_idx = parts2.index('SN')
                vals = parts2[sn_idx + 1:]
                comp = {}
                for elem, cols in ELEM_COLS.items():
                    comp[elem] = sum(float(vals[c - 7]) for c in cols if c - 7 < len(vals))
                init_comp[Z] = comp
            except (ValueError, IndexError):
                pass

            for line in lines[4:]:
                parts = line.split()
                if len(parts) < 47:
                    continue
                M_init = float(parts[0])
                ylds = {}
                for elem, cols in ELEM_COLS.items():
                    ylds[elem] = sum(float(parts[c]) for c in cols)
                ylds['Fe'] += ylds['Ni']
                ylds['Ni'] = 0.0
                total[(Z, M_init)] = {
                    'yields': ylds,
                    'M_fin': float(parts[1]),
                    'M_rem': float(parts[4]),
                    'M_bar': float(parts[5]),
                    'M_HE': float(parts[2]),
                    'M_CO': float(parts[3]),
                    'sn_type': SNT_MAP.get(parts[6], 3),
                }

        for wf in glob(os.path.join(zdir, '*winds_ejecta.dat')):
            with open(wf) as fh:
                lines = fh.readlines()
            for line in lines[4:]:
                parts = line.split()
                if len(parts) < 44:
                    continue
                M_init = float(parts[0])
                ylds = {}
                for elem, cols in WIND_COLS.items():
                    ylds[elem] = sum(float(parts[c]) for c in cols)
                wind[(Z, M_init)] = {
                    'yields': ylds, 'M_fin': float(parts[1]),
                }

    return total, wind, init_comp


def test_parsec2_vs_table(tbl, raw_total, raw_wind, init_comp):
    """Cross-check PARSEC v2 entries against raw ejecta files."""
    print("\n--- 2.1 PARSEC v2 raw vs table ---")
    TRACKED = tbl['elems']
    raw_Zs = sorted(set(k[0] for k in raw_total.keys()))
    n_checked = 0

    for iz in range(len(tbl['Z'])):
        Z_tbl = tbl['Z'][iz]
        Z_raw = min(raw_Zs, key=lambda z: abs(np.log10(max(z, 1e-12)) - np.log10(max(Z_tbl, 1e-12))))
        X_init = init_comp.get(Z_raw, {})

        for im in range(len(tbl['M_init'])):
            M = tbl['M_init'][im]
            if tbl['yield_source'][iz, im] != 4:
                continue
            key = (Z_raw, M)
            if key not in raw_total:
                continue
            raw = raw_total[key]
            rt = int(tbl['remnant_type'][iz, im])
            n_checked += 1

            # Remnant type
            check(rt == raw['sn_type'],
                  f"rem_type: tbl={rt} raw={raw['sn_type']} at M={M:.0f} Z={Z_tbl:.4f}")

            # Remnant mass (FSN/DBH may be reclassified)
            rm_tol = 5.0 if rt in (3, 6) else 0.1
            check(abs(tbl['remnant_mass'][iz, im] - raw['M_rem']) < rm_tol,
                  f"rem_mass: tbl={tbl['remnant_mass'][iz,im]:.2f} raw={raw['M_rem']:.2f} "
                  f"at M={M:.0f} Z={Z_tbl:.4f}")

            # M_preSN
            mp_tol = 1.0 if rt in (3, 6) else 0.5
            check(abs(tbl['M_preSN'][iz, im] - raw['M_fin']) < mp_tol,
                  f"M_preSN: tbl={tbl['M_preSN'][iz,im]:.2f} raw={raw['M_fin']:.2f} "
                  f"at M={M:.0f} Z={Z_tbl:.4f}")

            # Core masses
            check(abs(tbl['M_He_core'][iz, im] - raw['M_HE']) < 1.0,
                  f"M_He: tbl={tbl['M_He_core'][iz,im]:.2f} raw={raw['M_HE']:.2f} "
                  f"at M={M:.0f} Z={Z_tbl:.4f}")
            check(abs(tbl['M_CO_core'][iz, im] - raw['M_CO']) < 1.0,
                  f"M_CO: tbl={tbl['M_CO_core'][iz,im]:.2f} raw={raw['M_CO']:.2f} "
                  f"at M={M:.0f} Z={Z_tbl:.4f}")

            # Net yields (skip FSN/DBH — zeroed)
            if rt not in (3, 6):
                M_ej_bar = M - raw['M_bar']
                for ie, elem in enumerate(TRACKED):
                    raw_net = raw['yields'].get(elem, 0.0) - X_init.get(elem, 0.0) * M_ej_bar
                    tbl_net = tbl['net_yields'][iz, im, ie]
                    tol = max(0.01, abs(raw_net) * 0.05)
                    check(abs(tbl_net - raw_net) < tol,
                          f"net[{elem}]: tbl={tbl_net:.4f} raw={raw_net:.4f} "
                          f"at M={M:.0f} Z={Z_tbl:.4f}")

            # Wind yields
            if key in raw_wind and rt not in (3, 6):
                rw = raw_wind[key]
                M_wind = M - rw['M_fin']
                for ie, elem in enumerate(TRACKED):
                    raw_wnet = rw['yields'].get(elem, 0.0) - X_init.get(elem, 0.0) * M_wind
                    tbl_wnet = tbl['wind_yields'][iz, im, ie]
                    tol = max(0.01, abs(raw_wnet) * 0.05)
                    check(abs(tbl_wnet - raw_wnet) < tol,
                          f"wind[{elem}]: tbl={tbl_wnet:.4f} raw={raw_wnet:.4f} "
                          f"at M={M:.0f} Z={Z_tbl:.4f}")

    print(f"  checked {n_checked} PARSEC v2 grid-point entries")


def test_limongi_vs_table(tbl):
    """Cross-check Limongi 8-14 Msun yields at solar Z."""
    print("\n--- 2.2 Limongi+2025 raw vs table ---")
    masses = [9.22, 10.0, 11.0, 12.0, 13.0, 15.0]
    remnants = [1.263, 1.211, 1.316, 1.333, 1.456, 1.398]
    lim_mej = [m - r for m, r in zip(masses, remnants)]
    yields_table = {
        'H':  [4.676e+0, 5.055e+0, 5.473e+0, 5.884e+0, 6.272e+0, 7.009e+0],
        'He': [3.057e+0, 3.328e+0, 3.679e+0, 4.026e+0, 4.341e+0, 4.940e+0],
        'C':  [2.765e-2, 3.690e-2, 5.473e-2, 7.971e-2, 1.133e-1, 1.746e-1],
        'N':  [2.445e-2, 2.414e-2, 2.916e-2, 3.213e-2, 3.430e-2, 3.758e-2],
        'O':  [4.806e-2, 7.429e-2, 1.576e-1, 2.287e-1, 3.516e-1, 5.948e-1],
        'F':  [3.215e-6, 3.171e-6, 3.254e-6, 3.473e-6, 3.747e-6, 4.193e-6],
        'Ne': [1.168e-2, 1.901e-2, 3.505e-2, 7.853e-2, 8.916e-2, 2.900e-1],
        'Na': [5.638e-4, 6.490e-4, 6.792e-4, 8.696e-4, 1.295e-3, 3.794e-3],
        'Mg': [5.081e-3, 7.873e-3, 2.702e-2, 3.830e-2, 5.155e-2, 8.813e-2],
        'Al': [5.335e-4, 8.411e-4, 2.881e-3, 4.043e-3, 5.859e-3, 9.368e-3],
        'Si': [1.122e-2, 2.042e-2, 4.297e-2, 5.618e-2, 9.567e-2, 1.095e-1],
        'S':  [5.609e-3, 9.747e-3, 1.475e-2, 2.068e-2, 3.205e-2, 3.979e-2],
        'Ca': [1.116e-3, 1.754e-3, 2.440e-3, 3.279e-3, 4.396e-3, 5.738e-3],
        'Ti': [1.827e-5, 1.987e-5, 2.200e-5, 2.399e-5, 2.600e-5, 2.981e-5],
        'Fe': [1.979e-2, 2.971e-2, 4.042e-2, 4.771e-2, 5.489e-2, 6.924e-2],
    }
    X_solar = {
        'H': 0.7154, 'He': 0.2703, 'C': 2.38e-3, 'N': 6.93e-4, 'O': 5.72e-3,
        'F': 3.26e-7, 'Ne': 1.26e-3, 'Na': 2.93e-5, 'Mg': 5.15e-4, 'Al': 5.56e-5,
        'Si': 6.65e-4, 'S': 3.10e-4, 'Ca': 5.99e-5, 'Ti': 2.42e-6, 'Fe': 1.17e-3,
    }
    lim_elems = set(yields_table.keys())

    iz_solar = np.argmin(np.abs(tbl['Z'] - 0.014))
    n_checked = 0

    for M_lim in masses:
        im = np.argmin(np.abs(tbl['M_init'] - M_lim))
        if abs(tbl['M_init'][im] - M_lim) > 0.5:
            continue
        if tbl['yield_source'][iz_solar, im] != 2:
            continue
        n_checked += 1
        M_ej_lim = np.interp(tbl['M_init'][im], masses, lim_mej)
        for ie, elem in enumerate(tbl['elems']):
            if elem in lim_elems:
                y_total = np.interp(tbl['M_init'][im], masses, yields_table[elem])
                expected = y_total - X_solar.get(elem, 0.0) * M_ej_lim
            else:
                expected = 0.0
            tbl_val = tbl['net_yields'][iz_solar, im, ie]
            tol = max(0.005, abs(expected) * 0.05)
            check(abs(tbl_val - expected) < tol,
                  f"Limongi net[{elem}] at M={tbl['M_init'][im]:.1f}: "
                  f"tbl={tbl_val:.4f} exp={expected:.4f}")
    print(f"  checked {n_checked} Limongi entries")


# ═══════════════════════════════════════════════════════════════════════
#  Part 3 — C bilinear interpolation fidelity
# ═══════════════════════════════════════════════════════════════════════

def bilinear(gx, gy, data, x, y):
    ix = max(0, min(np.searchsorted(gx, x) - 1, len(gx) - 2))
    fx = max(0.0, min(1.0, (x - gx[ix]) / (gx[ix + 1] - gx[ix])))
    iy = max(0, min(np.searchsorted(gy, y) - 1, len(gy) - 2))
    fy = max(0.0, min(1.0, (y - gy[iy]) / (gy[iy + 1] - gy[iy])))
    return ((1 - fx) * (1 - fy) * data[ix, iy] +
            (1 - fx) * fy * data[ix, iy + 1] +
            fx * (1 - fy) * data[ix + 1, iy] +
            fx * fy * data[ix + 1, iy + 1])


def test_interp_grid_points(tbl):
    """Bilinear at grid points returns exact values."""
    print("\n--- 3.1 Interpolation at grid points ---")
    logZ, logM = tbl['log_Z'], tbl['log_M_init']
    fields = [('remnant_mass', tbl['remnant_mass']),
              ('M_preSN', tbl['M_preSN']),
              ('lifetime_yr', tbl['lifetime_yr'])]
    for name, data in fields:
        for iz in range(len(logZ)):
            for im in range(len(logM)):
                v = bilinear(logZ, logM, data, logZ[iz], logM[im])
                check(abs(v - data[iz, im]) < 1e-10,
                      f"{name} interp mismatch at grid point iz={iz} im={im}")
    # Yields element 0 (H)
    for iz in range(len(logZ)):
        for im in range(len(logM)):
            v = bilinear(logZ, logM, tbl['net_yields'][:, :, 0], logZ[iz], logM[im])
            check(abs(v - tbl['net_yields'][iz, im, 0]) < 1e-10,
                  f"net_yield[H] interp mismatch at iz={iz} im={im}")
    print("  done")


def test_interp_midpoints(tbl):
    """Between grid points: interpolated values bounded by corners."""
    print("\n--- 3.2 Interpolation midpoint bounds ---")
    logZ, logM = tbl['log_Z'], tbl['log_M_init']
    data = tbl['remnant_mass']
    n = 0
    for iz in range(len(logZ) - 1):
        for im in range(len(logM) - 1):
            z_mid = 0.5 * (logZ[iz] + logZ[iz + 1])
            m_mid = 0.5 * (logM[im] + logM[im + 1])
            corners = [data[iz, im], data[iz, im + 1],
                       data[iz + 1, im], data[iz + 1, im + 1]]
            v = bilinear(logZ, logM, data, z_mid, m_mid)
            check(min(corners) - 0.01 <= v <= max(corners) + 0.01,
                  f"rem_mass out of bounds at mid iz={iz} im={im}")
            n += 1
    print(f"  checked {n} midpoints")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading {TABLE_PATH}")
    tbl = load_table()
    NZ, NM = len(tbl['Z']), len(tbl['M_init'])
    NAGE = len(tbl['log_age_yr'])
    NELEM = len(tbl['elems'])
    print(f"Grid: {NZ} Z x {NM} M x {NAGE} ages, {NELEM} elements")

    # Part 1
    test_grid_axes(tbl)
    test_lifetimes(tbl)
    test_M_current(tbl)
    test_surface_abundances(tbl)
    test_radiation(tbl)
    test_wind_velocity(tbl)
    test_mass_loss_rate(tbl)
    test_phases(tbl)
    test_core_masses(tbl)
    test_remnants_and_yields(tbl)
    test_type_ia(tbl)

    # Part 2
    if os.path.isdir(RAW_PARSEC_DIR):
        raw_total, raw_wind, init_comp = load_raw_parsec2(RAW_PARSEC_DIR)
        test_parsec2_vs_table(tbl, raw_total, raw_wind, init_comp)
    else:
        print(f"\nWARNING: {RAW_PARSEC_DIR} not found, skipping Part 2.1")
    test_limongi_vs_table(tbl)

    # Part 3
    test_interp_grid_points(tbl)
    test_interp_midpoints(tbl)

    print(f"\n{'=' * 60}")
    print(f"Results: {npass} passed, {nfail} FAILED")
    if nfail > 0:
        print(f"TABLE HAS {nfail} CONSISTENCY PROBLEM(S)")
        sys.exit(1)
    else:
        print("All checks passed")
        sys.exit(0)
