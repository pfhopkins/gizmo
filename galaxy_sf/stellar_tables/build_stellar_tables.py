#!/usr/bin/env python3
"""
Build unified stellar evolution + yield tables for GIZMO.

Track sources:
  - PARSEC v1.2s: all masses < 20 Msun, and >= 20 Msun where BoOST unavailable
  - BoOST v1.3:   masses >= 20 Msun where closest BoOST Z is within 0.3 dex

Yield sources:
  - Karakas & Lugaro (2016): AGB yields, 1-8 Msun, Z=0.007/0.014/0.03
  - Karakas (2010):          AGB yields, 1-6.5 Msun, Z=0.0001/0.004/0.008/0.02
  - Chieffi & Limongi (2018): CCSNe yields, 13-120 Msun
  - Heger & Woosley (2002):  PISN yields, > 120 Msun

Output: single HDF5 with time-dependent evolution + end-of-life yields.
"""

import numpy as np
import os
import re
import sys
from glob import glob
from scipy.integrate import quad

# ═══════════════════════════════════════════════════════════════
# SECTION 1: Constants and configuration
# ═══════════════════════════════════════════════════════════════

h_planck  = 6.62607015e-27   # erg s
k_boltz   = 1.380649e-16     # erg/K
c_light   = 2.99792458e10    # cm/s
sigma_sb  = 5.670374419e-5   # erg cm^-2 s^-1 K^-4
eV_to_erg = 1.602176634e-12
L_sun     = 3.828e33          # erg/s
R_sun     = 6.957e10          # cm
M_sun_g   = 1.989e33          # g
G_cgs     = 6.674e-8          # cm^3 g^-1 s^-2

E_FUV_lo, E_FUV_hi = 6.0, 11.2    # eV
E_LW_lo,  E_LW_hi  = 11.2, 13.6   # eV

TRACKED_ELEMENTS = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ca', 'Ti', 'Fe']
N_ELEM = len(TRACKED_ELEMENTS)

# Solar mass fractions (Asplund+ 2009) for scaling initial abundances
Z_SOLAR = 0.0134
SOLAR_MASS_FRAC = {
    'H': 0.7381, 'He': 0.2485, 'C': 2.36e-3, 'N': 6.91e-4, 'O': 5.72e-3,
    'F': 3.26e-7, 'Ne': 1.25e-3, 'Na': 2.98e-5, 'Mg': 5.91e-4,
    'Al': 5.57e-5, 'Si': 6.65e-4, 'S': 3.10e-4, 'Ca': 6.44e-5,
    'Ti': 3.59e-6, 'Fe': 1.17e-3,
}
# Metal fractions relative to Z (for scaling to arbitrary Z)
SOLAR_METAL_FRAC = {e: SOLAR_MASS_FRAC[e] / Z_SOLAR for e in TRACKED_ELEMENTS
                    if e not in ('H', 'He')}

# PARSEC master Z grid
PARSEC_Z_GRID = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008,
                 0.01, 0.014, 0.017, 0.02, 0.03, 0.04, 0.06]

# BoOST grid names and their metallicities (computed from initial surface abundances)
BOOST_GRIDS = {
    'MW':     0.00876,
    'LMC':    0.00467,
    'SMC':    0.00208,
    'dwarfA': 0.00105,
    'dwarfB': 0.000426,
    'IZw18':  0.000217,
    'dwarfD': 0.000101,
    'dwarfE': 0.0000431,
}

# BoOST surface isotope columns (0-indexed) → element mapping
BOOST_SURFACE_ELEM = {
    'H':  [26, 27],         # sH1 + sH2
    'He': [28, 29],         # sHe3 + sHe4
    'C':  [38, 39],         # sC12 + sC13
    'N':  [41, 42],         # sN14 + sN15
    'O':  [43, 44, 45],     # sO16 + sO17 + sO18
    'F':  [46],             # sF19
    'Ne': [47, 48, 49],     # sNe20 + sNe21 + sNe22
    'Na': [],               # not in BoOST → solar-scaled
    'Mg': [50, 51, 52, 53], # sMg22 + sMg24 + sMg25 + sMg26
    'Al': [54, 55],         # sAl26 + sAl27
    'Si': [56, 57, 58],     # sSi28 + sSi29 + sSi30
    'S':  [],               # not in BoOST → solar-scaled
    'Ca': [],               # not in BoOST → solar-scaled
    'Ti': [],               # not in BoOST → solar-scaled
    'Fe': [59],             # sFe56
}

# MIST surface column names → element mapping
MIST_SURFACE_ELEM = {
    'H':  ['surface_h1'],
    'He': ['surface_he3', 'surface_he4'],
    'C':  ['surface_c12', 'surface_c13'],
    'N':  ['surface_n14'],
    'O':  ['surface_o16'],
    'F':  ['surface_f19'],
    'Ne': ['surface_ne20'],
    'Na': ['surface_na23'],
    'Mg': ['surface_mg24'],
    'Al': [],               # not in MIST → solar-scaled
    'Si': ['surface_si28'],
    'S':  ['surface_s32'],
    'Ca': ['surface_ca40'],
    'Ti': ['surface_ti48'],
    'Fe': ['surface_fe56'],
}

# MIST [Fe/H] → approximate Z mapping (Asplund+ 2009)
# Z = 0.751 * ZX / (1 + 2.5*ZX) where ZX = (Z/X)_solar * 10^[Fe/H]
_ZX_SOLAR = Z_SOLAR / SOLAR_MASS_FRAC['H']
MIST_FEH_GRID = {
    'm4.00': -4.0, 'm3.50': -3.5, 'm3.00': -3.0, 'm2.50': -2.5,
    'm2.00': -2.0, 'm1.75': -1.75, 'm1.50': -1.5, 'm1.25': -1.25,
    'm1.00': -1.0, 'm0.75': -0.75, 'm0.50': -0.5, 'm0.25': -0.25,
    'p0.00': 0.0, 'p0.25': 0.25, 'p0.50': 0.5,
}
MIST_Z_GRID = {}
for feh_str, feh_val in MIST_FEH_GRID.items():
    ZX = _ZX_SOLAR * 10**feh_val
    MIST_Z_GRID[feh_str] = 0.751 * ZX / (1.0 + 2.5 * ZX)

# PARSEC v2 VMS surface columns → element mapping (from .TAB track files)
# Only 7 surface elements available; rest use initial composition (solar-scaled)
PARSEC2_SURFACE_COLS = {
    'H':  ['Xsup'],
    'He': ['Ysup'],            # total He (He3+He4)
    'C':  ['XCsup', 'XC13sup'],  # C12 + C13
    'N':  ['XNsup'],           # N14
    'O':  ['XOsup', 'XO18sup'],  # O16 + O18
    'F':  [],                  # not on surface → initial-scaled
    'Ne': ['XNEsup'],
    'Na': [],                  # not on surface → initial-scaled
    'Mg': ['XMGsup'],
    'Al': [],                  # not on surface → initial-scaled
    'Si': [],                  # not on surface → initial-scaled
    'S':  [],                  # not on surface → initial-scaled
    'Ca': [],                  # not on surface → initial-scaled
    'Ti': [],                  # not on surface → initial-scaled
    'Fe': [],                  # not on surface → initial-scaled
}

# PARSEC v2 VMS metallicity grid (Z values from directory names)
PARSEC2_Z_GRID = [1e-11, 1e-6, 0.0001, 0.001, 0.002, 0.004, 0.006, 0.008,
                  0.01, 0.014, 0.017, 0.02, 0.03]

# Stitch: PARSEC v2 primary (M >= 2), MIST fallback (M < 2 or gaps), BoOST/PARSEC v1 last resort
MASS_STITCH_P2_LO = 2.0       # PARSEC v2 primary above this (its min mass)
# Legacy BoOST stitch (fallback only)
MASS_STITCH_BOOST_LO = 40.0   # BoOST fallback lower limit
MASS_STITCH_BOOST_HI = 80.0   # BoOST fallback upper limit

Z_MATCH_THRESHOLD = 0.3  # dex: max log(Z) distance for BoOST match

LOG_FLOOR = -99.0

# Simplified evolutionary phase encoding (for C code)
# PARSEC v1.2s phases 1-4 → PMS, 5 → MS, 6 → HG, 7-8 → RGB, 9 → CHeB,
# 10-11 → AGB, 12+ → late (He-shell/C-burn for massive, TP-AGB for intermediate)
PARSEC_PHASE_MAP = {
    1: 0, 2: 0, 3: 0, 4: 0,   # PMS
    5: 1,                       # MS
    6: 2,                       # SGB/HG
    7: 3, 8: 3,                 # RGB
    9: 4,                       # CHeB/HB
    10: 5, 11: 5,               # AGB (E-AGB + TP-AGB)
    12: 6, 13: 6, 14: 6, 15: 6, # post-AGB / late burning
}
# Unified phase encoding:
# 0=PMS, 1=MS, 2=HG(SGB), 3=RGB, 4=CHeB, 5=AGB, 6=post-AGB/WR, 7=dead
PHASE_NAMES = '0=PMS, 1=MS, 2=HG/SGB, 3=RGB, 4=CHeB, 5=AGB, 6=post-AGB/WR, 7=dead'

# MIST phase codes → unified encoding
# MIST has no dedicated SGB/HG phase (it jumps from MS=0 to RGB=2)
MIST_PHASE_MAP = {
    -1: 0,  # PMS
    0: 1,   # MS
    2: 3,   # RGB (includes SGB/HG)
    3: 4,   # CHeB
    4: 5,   # EAGB → AGB
    5: 5,   # TPAGB → AGB
    6: 6,   # postAGB → post-AGB/WR
    9: 6,   # WR → post-AGB/WR
}

# Common age grid
N_AGE = 512
LOG_AGE_MIN, LOG_AGE_MAX = 2.0, 10.3  # 100 yr to 20 Gyr


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Planck integral LUT (for blackbody radiation)
# ═══════════════════════════════════════════════════════════════

def _planck_photon(x):
    return x**2 / (np.exp(x) - 1.0) if x < 500 else 0.0

def _planck_energy(x):
    return x**3 / (np.exp(x) - 1.0) if x < 500 else 0.0

def build_planck_lut(N_T=2000):
    """Pre-compute Planck integrals on a fine log(T) grid."""
    log_T = np.linspace(3.0, 5.7, N_T)  # 1000 K to 500,000 K
    T = 10**log_T
    Qion_per_L = np.zeros(N_T)
    FUV_frac   = np.zeros(N_T)
    LW_frac    = np.zeros(N_T)

    print("  Building Planck integral LUT...", flush=True)
    for i in range(N_T):
        kT = k_boltz * T[i]
        x_fuv_lo = E_FUV_lo * eV_to_erg / kT
        x_fuv_hi = E_FUV_hi * eV_to_erg / kT
        x_lw_lo  = E_LW_lo  * eV_to_erg / kT
        x_lw_hi  = E_LW_hi  * eV_to_erg / kT

        I_Q = I_F = I_L = 0.0
        if x_lw_hi < 200:
            I_Q, _ = quad(_planck_photon, x_lw_hi, np.inf, limit=200)
        if x_fuv_lo < 200:
            I_F, _ = quad(_planck_energy, x_fuv_lo, min(x_fuv_hi, 500), limit=200)
        if x_lw_lo < 200:
            I_L, _ = quad(_planck_energy, x_lw_lo, min(x_lw_hi, 500), limit=200)

        inv_sT4 = 1.0 / (sigma_sb * T[i]**4)
        Qion_per_L[i] = inv_sT4 * (2*np.pi/c_light**2) * (kT/h_planck)**3 * I_Q
        epref = inv_sT4 * (2*np.pi*h_planck/c_light**2) * (kT/h_planck)**4
        FUV_frac[i] = epref * I_F
        LW_frac[i]  = epref * I_L

    print("  Done.", flush=True)
    return log_T, Qion_per_L, FUV_frac, LW_frac

PLANCK_LUT = None  # filled lazily

def get_planck_lut():
    global PLANCK_LUT
    if PLANCK_LUT is None:
        PLANCK_LUT = build_planck_lut()
    return PLANCK_LUT

def compute_radiation_from_lut(logTeff_arr, logL_arr):
    """Vectorized radiation computation using Planck LUT."""
    log_T, Qp, Fp, Lp = get_planck_lut()
    L_erg = 10**logL_arr * L_sun
    q = np.interp(logTeff_arr, log_T, Qp, left=0, right=0)
    f = np.interp(logTeff_arr, log_T, Fp, left=0, right=0)
    l = np.interp(logTeff_arr, log_T, Lp, left=0, right=0)
    return L_erg * q, L_erg * f, L_erg * l


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Wind velocity
# ═══════════════════════════════════════════════════════════════

def compute_v_wind(Teff, M_Msun, R_Rsun):
    """
    Terminal wind velocity [km/s] from stellar parameters.

    Prescriptions:
      Teff > 21000 K: v_wind = 2.6 * v_esc  (Lamers+ 1995, hot side of bistability)
      10000 < Teff <= 21000 K: v_wind = 1.3 * v_esc  (cool side of bistability jump)
      Teff <= 10000 K: v_wind = max(0.7 * v_esc, 10 km/s)  (RSG/AGB regime)
    """
    R_cm = R_Rsun * R_sun
    M_g = M_Msun * M_sun_g
    v_esc_cms = np.sqrt(2 * G_cgs * M_g / R_cm)  # cm/s
    v_esc_kms = v_esc_cms / 1e5  # km/s

    v_wind = np.where(
        Teff > 21000, 2.6 * v_esc_kms,
        np.where(Teff > 10000, 1.3 * v_esc_kms,
                 np.maximum(0.7 * v_esc_kms, 10.0)))
    return v_wind


# ═══════════════════════════════════════════════════════════════
# SECTION 4: PARSEC track loading
# ═══════════════════════════════════════════════════════════════

def discover_parsec_tracks(base_dir):
    """Index all PARSEC tracks. Returns dict: (Z, M_init) → filepath."""
    index = {}
    for zdir in sorted(glob(os.path.join(base_dir, 'Z*Y*'))):
        if not os.path.isdir(zdir):
            continue
        m = re.match(r'Z([\d.]+)Y', os.path.basename(zdir))
        if not m:
            continue
        Z = float(m.group(1))
        for f in glob(os.path.join(zdir, '*.DAT')):
            mm = re.search(r'_M([\d.]+)\.DAT', os.path.basename(f))
            if mm:
                M = float(mm.group(1))
                index[(Z, M)] = f
    return index


def load_parsec_track_raw(filepath):
    """Parse a PARSEC .DAT file. Returns dict of arrays."""
    data = []
    with open(filepath) as fh:
        header = fh.readline().split()
        for line in fh:
            vals = line.split()
            if len(vals) == len(header):
                data.append([float(v) for v in vals])
    arr = np.array(data)
    return {name: arr[:, i] for i, name in enumerate(header)}


def extract_phase_transitions(age_yr, phase):
    """
    Extract phase transition times from a phase time-series.
    Returns dict of transition ages in years.
    Phases: 0=PMS, 1=MS, 2=HG, 3=RGB, 4=CHeB, 5=AGB, 6=post-AGB/WR, 7=dead
    """
    t = {}
    # t_MS_start = end of PMS = first time phase >= 1
    ms_mask = phase >= 1
    t['t_MS_start'] = age_yr[ms_mask][0] if np.any(ms_mask) else age_yr[-1]

    # t_MS_end = TAMS = first time phase >= 2 (after being on MS)
    post_ms = phase >= 2
    if np.any(post_ms):
        t['t_MS_end'] = age_yr[post_ms][0]
    else:
        t['t_MS_end'] = age_yr[-1]  # never leaves MS (low-mass)

    # t_RGB_start = first time phase >= 3
    rgb_mask = phase >= 3
    t['t_RGB_start'] = age_yr[rgb_mask][0] if np.any(rgb_mask) else 0.0

    # t_CHeB_start = first time phase >= 4
    cheb_mask = phase >= 4
    t['t_CHeB_start'] = age_yr[cheb_mask][0] if np.any(cheb_mask) else 0.0

    # t_AGB_start = first time phase == 5
    agb_mask = phase == 5
    t['t_AGB_start'] = age_yr[agb_mask][0] if np.any(agb_mask) else 0.0

    # t_AGB_end = last time phase == 5
    t['t_AGB_end'] = age_yr[agb_mask][-1] if np.any(agb_mask) else 0.0

    # Phase 6 means WR (massive, no AGB) or post-AGB (intermediate, had AGB)
    post_mask = phase >= 6
    had_agb = np.any(phase == 5)
    if np.any(post_mask):
        if had_agb:
            t['t_postAGB_start'] = age_yr[post_mask][0]
            t['t_WR_start'] = 0.0
        else:
            t['t_WR_start'] = age_yr[post_mask][0]
            t['t_postAGB_start'] = 0.0
    else:
        t['t_postAGB_start'] = 0.0
        t['t_WR_start'] = 0.0

    return t


def parsec_track_to_common(filepath):
    """Load PARSEC track and return standardized dict."""
    d = load_parsec_track_raw(filepath)

    Teff = 10**d['LOG_TE']
    logL = d['LOG_L']
    R_cm = 10**d['LOG_R']     # PARSEC LOG_R = log(R/cm)
    R_Rsun = R_cm / R_sun

    Q_ion, L_FUV, L_LW = compute_radiation_from_lut(d['LOG_TE'], logL)
    L_bol = 10**logL * L_sun
    v_wind = compute_v_wind(Teff, d['MASS'], R_Rsun)

    # Surface abundances: H, He, C, N, O from PARSEC; Ne, Mg, Si, S, Ca, Fe scaled from Z
    surface = np.zeros((len(Teff), N_ELEM))
    parsec_cols = {'H': 'H_SUP', 'He': 'HE_SUP', 'C': 'C_SUP', 'N': 'N_SUP', 'O': 'O_SUP'}
    Z_star = 1.0 - d['H_SUP'][0] - d['HE_SUP'][0]  # initial Z from first timestep
    for ie, elem in enumerate(TRACKED_ELEMENTS):
        col = parsec_cols.get(elem)
        if col and col in d:
            surface[:, ie] = d[col]
        else:
            # Ne, Mg, Si, S, Ca, Fe: constant at solar-scaled initial value
            # These elements are unmodified by H/He shell burning in < 20 Msun stars
            surface[:, ie] = Z_star * SOLAR_METAL_FRAC[elem]

    # CO core mass at end of life (PARSEC column M_CORE_C)
    M_CO_core = d['M_CORE_C'][-1] if 'M_CORE_C' in d else 0.0
    M_He_core = d['M_CORE_HE'][-1] if 'M_CORE_HE' in d else 0.0

    # Evolutionary phase (simplified from PARSEC's detailed scheme)
    if 'PHASE' in d:
        phase = np.array([PARSEC_PHASE_MAP.get(int(p), 6) for p in d['PHASE']])
    else:
        phase = np.ones(len(Teff), dtype=int)  # assume MS if no phase info

    # Phase transition times (from raw track, full resolution)
    phase_transitions = extract_phase_transitions(d['AGE'], phase)

    return {
        'age_yr': d['AGE'],
        'logL': logL,
        'logTeff': d['LOG_TE'],
        'logR_cm': d['LOG_R'],  # already log(R/cm)
        'M_current': d['MASS'],
        'log_Mdot': d['LOG_RAT'],
        'Q_ion': Q_ion, 'L_FUV': L_FUV, 'L_LW': L_LW, 'L_bol': L_bol,
        'v_wind': v_wind,
        'surface': surface,
        'M_CO_core': M_CO_core,
        'M_He_core': M_He_core,
        'phase': phase,
        'phase_transitions': phase_transitions,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 5: BoOST track loading
# ═══════════════════════════════════════════════════════════════

def discover_boost_tracks(boost_dir):
    """Index all BoOST tracks. Returns dict: (grid_name, M_init) → filepath."""
    index = {}
    for d in os.listdir(boost_dir):
        full = os.path.join(boost_dir, d)
        if not os.path.isdir(full) or not d.startswith('BoOST-stellarmodels-'):
            continue
        grid = d.replace('BoOST-stellarmodels-', '')
        if grid == 'IZw18CHE':  # skip CHE (fast rotation)
            continue
        for f in glob(os.path.join(full, 'f*.dat')):
            m = re.match(r'f(\d+)-', os.path.basename(f))
            if m:
                M = int(m.group(1))
                index[(grid, M)] = f
    return index


def boost_track_to_common(filepath):
    """Load BoOST track and return standardized dict."""
    data = np.loadtxt(filepath)

    age_yr   = data[:, 0]
    M_cur    = data[:, 1]
    Teff     = data[:, 2]      # K (linear)
    logL     = data[:, 3]      # log(L/Lsun)
    R_Rsun   = data[:, 4]      # Rsun (linear)
    log_Mdot = data[:, 5]      # log(Msun/yr)

    logTeff = np.log10(np.maximum(Teff, 1.0))
    Q_ion, L_FUV, L_LW = compute_radiation_from_lut(logTeff, logL)
    L_bol = 10**logL * L_sun
    v_wind = compute_v_wind(Teff, M_cur, R_Rsun)

    # Surface abundances from BoOST isotope columns
    surface = np.zeros((len(age_yr), N_ELEM))
    H_init = data[0, 26] + data[0, 27]  # sH1 + sH2
    He_init = data[0, 28] + data[0, 29]  # sHe3 + sHe4
    Z_star = 1.0 - H_init - He_init
    for ie, elem in enumerate(TRACKED_ELEMENTS):
        cols = BOOST_SURFACE_ELEM[elem]
        if cols:
            for c in cols:
                surface[:, ie] += data[:, c]
        else:
            # S, Ca: not in BoOST — use solar-scaled value (constant, unchanged by burning)
            surface[:, ie] = Z_star * SOLAR_METAL_FRAC[elem]

    # CO and He core masses at end of life (BoOST columns 25, 24, 0-indexed)
    M_CO_core = data[-1, 25]  # coreCO[Msun]
    M_He_core = data[-1, 24]  # coreHe[Msun]

    # Infer evolutionary phase from Teff and surface H (BoOST has no phase column)
    # BoOST tracks are all massive (>= 9 Msun), so: MS → HG → CHeB/RSG → WR
    # Note: BoOST core mass columns (24, 25) are final values, not time-dependent
    H_surf = surface[:, 0]  # H mass fraction
    T_init = Teff[0] if Teff[0] > 1 else 30000.0

    phase = np.ones(len(age_yr), dtype=int)  # default: MS
    # HG: Teff drops below 85% of initial (leaving MS band)
    phase[Teff < 0.85 * T_init] = 2  # HG
    # CHeB/RSG: Teff < 10000 K (red side of HR diagram), still H-rich
    phase[Teff < 10000] = 4  # CHeB (blue loop) or RSG
    # WR: H depleted from surface (envelope stripped by winds)
    phase[H_surf < 0.3] = 6  # WR

    phase_transitions = extract_phase_transitions(age_yr, phase)

    return {
        'age_yr': age_yr,
        'logL': logL,
        'logTeff': logTeff,
        'logR_cm': np.log10(np.maximum(R_Rsun, 0.01) * R_sun),
        'M_current': M_cur,
        'log_Mdot': log_Mdot,
        'Q_ion': Q_ion, 'L_FUV': L_FUV, 'L_LW': L_LW, 'L_bol': L_bol,
        'v_wind': v_wind,
        'surface': surface,
        'M_CO_core': M_CO_core,
        'M_He_core': M_He_core,
        'phase': phase,
        'phase_transitions': phase_transitions,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 5b: MIST track loading
# ═══════════════════════════════════════════════════════════════

def discover_mist_tracks(mist_dir):
    """Index all MIST EEP tracks. Returns dict: (feh_str, M_init) → filepath."""
    index = {}
    for feh_str in MIST_FEH_GRID:
        dirname = f'MIST_v1.2_feh_{feh_str}_afe_p0.0_vvcrit0.0_EEPS'
        full = os.path.join(mist_dir, dirname)
        if not os.path.isdir(full):
            continue
        for f in glob(os.path.join(full, '*M.track.eep')):
            m = re.match(r'(\d+)M\.track\.eep', os.path.basename(f))
            if m:
                M = int(m.group(1)) / 100.0
                index[(feh_str, M)] = f
    return index


def mist_track_to_common(filepath):
    """Load MIST EEP track and return standardized dict. Returns None if truncated."""
    col_names = None
    data_lines = []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith('#'):
                if 'star_age' in line:
                    col_names = line.strip().lstrip('#').split()
                continue
            vals = line.split()
            if col_names and len(vals) == len(col_names):
                data_lines.append([float(v) for v in vals])

    if not data_lines or len(data_lines) < 250:
        return None  # truncated track (e.g. [Fe/H]=-0.25 broken 15-50 Msun)

    arr = np.array(data_lines)
    d = {name: arr[:, i] for i, name in enumerate(col_names)}

    age_yr = d['star_age']
    M_cur = d['star_mass']
    logL = d['log_L']
    logTeff_arr = d['log_Teff']
    logR_Rsun = d['log_R']  # MIST log_R = log(R/Rsun)
    logR_cm = logR_Rsun + np.log10(R_sun)

    # Mass loss rate: MIST star_mdot is linear Msun/yr (negative for loss)
    Mdot_abs = np.abs(d['star_mdot'])
    log_Mdot = np.where(Mdot_abs > 0, np.log10(Mdot_abs), -30.0)

    Teff = 10**logTeff_arr
    R_Rsun_lin = 10**logR_Rsun

    Q_ion, L_FUV, L_LW = compute_radiation_from_lut(logTeff_arr, logL)
    L_bol = 10**logL * L_sun
    v_wind = compute_v_wind(Teff, M_cur, R_Rsun_lin)

    # Surface abundances
    surface = np.zeros((len(age_yr), N_ELEM))
    H_init = d['surface_h1'][0]
    He_init = d.get('surface_he3', np.zeros(1))[0] + d.get('surface_he4', np.zeros(1))[0]
    Z_star = 1.0 - H_init - He_init
    for ie, elem in enumerate(TRACKED_ELEMENTS):
        cols = MIST_SURFACE_ELEM.get(elem, [])
        if cols:
            for c in cols:
                if c in d:
                    surface[:, ie] += d[c]
        elif elem not in ('H', 'He'):
            surface[:, ie] = Z_star * SOLAR_METAL_FRAC[elem]

    # Core masses at end of life
    M_He_core = d['he_core_mass'][-1] if 'he_core_mass' in d else 0.0
    M_CO_core = d['c_core_mass'][-1] if 'c_core_mass' in d else 0.0

    # Phase mapping (MIST → unified)
    if 'phase' in d:
        phase = np.array([MIST_PHASE_MAP.get(int(round(p)), 1) for p in d['phase']])
    else:
        phase = np.ones(len(age_yr), dtype=int)

    phase_transitions = extract_phase_transitions(age_yr, phase)

    return {
        'age_yr': age_yr,
        'logL': logL,
        'logTeff': logTeff_arr,
        'logR_cm': logR_cm,
        'M_current': M_cur,
        'log_Mdot': log_Mdot,
        'Q_ion': Q_ion, 'L_FUV': L_FUV, 'L_LW': L_LW, 'L_bol': L_bol,
        'v_wind': v_wind,
        'surface': surface,
        'M_CO_core': M_CO_core,
        'M_He_core': M_He_core,
        'phase': phase,
        'phase_transitions': phase_transitions,
    }


def find_mist_feh_for_Z(Z):
    """Find the closest MIST [Fe/H] string for a given metallicity Z."""
    best_str = None
    best_dist = 1e10
    for feh_str, Z_mist in MIST_Z_GRID.items():
        dist = abs(np.log10(max(Z, 1e-8)) - np.log10(max(Z_mist, 1e-8)))
        if dist < best_dist:
            best_dist = dist
            best_str = feh_str
    return best_str


# ═══════════════════════════════════════════════════════════════
# SECTION 5c: PARSEC v2 VMS track loading
# ═══════════════════════════════════════════════════════════════

def discover_parsec2_vms_tracks(p2_dir):
    """Index all PARSEC v2 VMS tracks. Returns dict: (Z_str, M_init) → filepath."""
    index = {}
    for d in sorted(os.listdir(p2_dir)):
        if not d.startswith('tracks_Z') or not os.path.isdir(os.path.join(p2_dir, d)):
            continue
        Z_str = d.replace('tracks_Z', '')
        for f in glob(os.path.join(p2_dir, d, '*.TAB')):
            m = re.search(r'_M(\d+\.?\d*)\.TAB$', os.path.basename(f))
            if m:
                M = float(m.group(1))
                index[(Z_str, M)] = f
    return index


def find_parsec2_Z_str_for_Z(Z):
    """Find the closest PARSEC v2 Z string for a target metallicity."""
    best_str = None
    best_dist = 1e10
    for Z_p2 in PARSEC2_Z_GRID:
        dist = abs(np.log10(max(Z, 1e-12)) - np.log10(max(Z_p2, 1e-12)))
        if dist < best_dist:
            best_dist = dist
            if Z_p2 < 1e-10:
                best_str = '1E-11'
            elif Z_p2 < 1e-5:
                best_str = '1E-6'
            else:
                best_str = f'{Z_p2:g}'
    return best_str


def parsec2_vms_track_to_common(filepath):
    """Load PARSEC v2 VMS track (.TAB) and return standardized dict."""
    # Read header (3 lines: comment, metadata, column names)
    col_names = None
    with open(filepath) as fh:
        lines = fh.readlines()

    # Line 3 has column names (0-indexed line 2)
    col_names = lines[2].split()
    data_lines = []
    for line in lines[3:]:
        vals = line.split()
        if len(vals) == len(col_names):
            data_lines.append([float(v) for v in vals])

    if len(data_lines) < 50:
        return None

    arr = np.array(data_lines)
    d = {name: arr[:, i] for i, name in enumerate(col_names)}

    age_yr = d['AGE']
    M_cur = d['MASS']
    logL = d['LOG_L']
    logTeff_arr = d['LOG_TE']
    R_cm = d['RSTAR']  # already in cm
    logR_cm = np.log10(np.maximum(R_cm, 1.0))
    R_Rsun = R_cm / R_sun

    # Mass loss rate: RATE is in Msun/yr (positive = mass loss)
    Mdot_abs = np.abs(d['RATE'])
    log_Mdot = np.where(Mdot_abs > 0, np.log10(Mdot_abs), -30.0)

    Teff = 10**logTeff_arr

    Q_ion, L_FUV, L_LW = compute_radiation_from_lut(logTeff_arr, logL)
    L_bol = 10**logL * L_sun
    v_wind = compute_v_wind(Teff, M_cur, R_Rsun)

    # Surface abundances: 7 elements from .TAB, rest solar-scaled from initial Z
    surface = np.zeros((len(age_yr), N_ELEM))
    H_init = d['Xsup'][0]
    He_init = d['Ysup'][0]
    Z_star = 1.0 - H_init - He_init

    for ie, elem in enumerate(TRACKED_ELEMENTS):
        cols = PARSEC2_SURFACE_COLS.get(elem, [])
        if cols:
            for c in cols:
                if c in d:
                    surface[:, ie] += d[c]
        elif elem not in ('H', 'He'):
            # Scale from initial Z using solar metal fractions
            surface[:, ie] = Z_star * SOLAR_METAL_FRAC[elem]

    # Core masses: QHEL and QCAROX are mass fractions (core_mass / total_mass)
    M_He_core = d['QHEL'][-1] * M_cur[-1]
    M_CO_core = d['QCAROX'][-1] * M_cur[-1]

    # Phase classification from surface H and Teff
    # PARSEC v2 has no explicit phase column; infer from physical state
    H_surf = surface[:, 0]
    phase = np.ones(len(age_yr), dtype=int)  # default: MS

    # PMS: first few steps where star is still contracting (very cool, large)
    # Criterion: Teff increasing rapidly OR logTeff < 4.0 at the start
    # Find where H-burning ignites: central H starts decreasing
    Xc = d['XCEN']
    Xc_init = Xc[0]
    ms_start = np.argmax(Xc < Xc_init - 1e-4)  # first significant H depletion
    if ms_start == 0:
        ms_start = 1  # safety: at least skip first point
    phase[:ms_start] = 0  # PMS

    # MS end: central H exhaustion (XCEN < 1e-4)
    ms_end_mask = Xc < 1e-4
    if np.any(ms_end_mask):
        ms_end = np.argmax(ms_end_mask)
        # Post-MS phases for massive stars:
        # RGB-like / HG when Teff drops and H still on surface
        # CHeB when central He burns (YCEN decreasing)
        # WR when surface H < 0.4 and logTeff > 4.0
        Yc = d['YCEN']
        Yc_at_ms_end = Yc[ms_end] if ms_end < len(Yc) else 0

        for i in range(ms_end, len(age_yr)):
            if H_surf[i] < 0.4 and logTeff_arr[i] > 4.0:
                phase[i] = 6  # WR / post-AGB
            elif Yc[i] > 1e-3 and Yc[i] < Yc_at_ms_end - 1e-3:
                phase[i] = 4  # CHeB
            elif logTeff_arr[i] < 3.7 and H_surf[i] > 0.01:
                phase[i] = 3  # RGB/RSG
            else:
                phase[i] = 2  # HG/SGB

    phase_transitions = extract_phase_transitions(age_yr, phase)

    return {
        'age_yr': age_yr,
        'logL': logL,
        'logTeff': logTeff_arr,
        'logR_cm': logR_cm,
        'M_current': M_cur,
        'log_Mdot': log_Mdot,
        'Q_ion': Q_ion, 'L_FUV': L_FUV, 'L_LW': L_LW, 'L_bol': L_bol,
        'v_wind': v_wind,
        'surface': surface,
        'M_CO_core': M_CO_core,
        'M_He_core': M_He_core,
        'phase': phase,
        'phase_transitions': phase_transitions,
    }


def interpolate_parsec2_to_mass(p2_index, Z_str, M_target, log_age_grid):
    """Interpolate PARSEC v2 VMS tracks to a target mass."""
    return _interpolate_to_mass(
        parsec2_vms_track_to_common, p2_index, Z_str, M_target,
        MASS_STITCH_P2_LO, log_age_grid)


# ═══════════════════════════════════════════════════════════════
# SECTION 6: Resampling onto common grid
# ═══════════════════════════════════════════════════════════════

def resample_onto_age_grid(track, log_age_grid):
    """Resample a track dict onto the common log(age) grid. Dead → floor values."""
    age = track['age_yr']
    lifetime = age[-1]
    log_lifetime = np.log10(max(lifetime, 1.0))

    mask = age > 0
    if np.sum(mask) < 2:
        return None

    log_age_track = np.log10(age[mask])
    N = len(log_age_grid)

    # Quantities to resample
    result = {}

    # Log-scale quantities (floor when dead)
    for key in ['logL', 'logTeff', 'logR_cm', 'log_Mdot']:
        vals = track[key][mask]
        interp = np.interp(log_age_grid, log_age_track, vals, left=vals[0], right=LOG_FLOOR)
        interp[log_age_grid > log_lifetime] = LOG_FLOOR
        result[key] = interp

    # Radiation: store as log, floor when dead
    for key in ['Q_ion', 'L_FUV', 'L_LW', 'L_bol']:
        vals = track[key][mask]
        log_vals = np.where(vals > 0, np.log10(vals), LOG_FLOOR)
        interp = np.interp(log_age_grid, log_age_track, log_vals, left=log_vals[0], right=LOG_FLOOR)
        interp[log_age_grid > log_lifetime] = LOG_FLOOR
        result['log_' + key] = interp

    # Linear quantities (0 when dead)
    for key in ['M_current', 'v_wind']:
        vals = track[key][mask]
        interp = np.interp(log_age_grid, log_age_track, vals, left=vals[0], right=0.0)
        interp[log_age_grid > log_lifetime] = 0.0
        result[key] = interp

    # Surface abundances (0 when dead)
    surf = track['surface'][mask]
    surf_interp = np.zeros((N, N_ELEM))
    for ie in range(N_ELEM):
        s = np.interp(log_age_grid, log_age_track, surf[:, ie], left=surf[0, ie], right=0.0)
        s[log_age_grid > log_lifetime] = 0.0
        surf_interp[:, ie] = s
    result['surface'] = surf_interp

    # Evolutionary phase (true nearest-neighbor — discrete quantity, no linear interp)
    if 'phase' in track:
        ph = track['phase'][mask]
        # For each grid point, find nearest track point and copy its phase
        ph_interp = np.full(N, 7, dtype=int)
        for i in range(N):
            if log_age_grid[i] <= log_lifetime:
                idx = np.argmin(np.abs(log_age_track - log_age_grid[i]))
                ph_interp[i] = ph[idx]
            else:
                ph_interp[i] = 7  # dead
    else:
        ph_interp = np.where(log_age_grid <= log_lifetime, 1, 7).astype(int)
    result['phase'] = ph_interp

    # ── Enforce Mdot / M_current consistency ──
    # Resampling M_current and log_Mdot independently breaks their consistency.
    # Recompute Mdot from M_current so that dM/dt matches the actual mass evolution.
    M_cur = result['M_current']
    age_grid_yr = 10**log_age_grid
    alive_mask = log_age_grid <= log_lifetime
    mdot_consistent = np.full(N, -30.0)  # log10(Msun/yr), floor = -30
    for i in range(N):
        if not alive_mask[i]:
            continue
        if i < N - 1 and alive_mask[i + 1]:
            dM = M_cur[i] - M_cur[i + 1]
            dt = age_grid_yr[i + 1] - age_grid_yr[i]
        elif i > 0 and alive_mask[i - 1]:
            dM = M_cur[i - 1] - M_cur[i]
            dt = age_grid_yr[i] - age_grid_yr[i - 1]
        else:
            continue
        if dM > 0 and dt > 0:
            mdot_consistent[i] = np.log10(dM / dt)
    result['log_Mdot'] = mdot_consistent

    result['lifetime'] = lifetime
    # Pre-SN mass: last alive M_current from raw track
    alive_raw = track['M_current'][mask]
    result['M_preSN_scalar'] = alive_raw[-1] if len(alive_raw) > 0 else 0.0
    # Core masses (scalars, passed through from track)
    result['M_CO_core'] = track.get('M_CO_core', 0.0)
    result['M_He_core'] = track.get('M_He_core', 0.0)
    # Phase transition times (pass through from raw track)
    result['phase_transitions'] = track.get('phase_transitions', {})

    return result


def find_boost_grid_for_Z(Z):
    """Find the closest BoOST grid for a given Z. Returns grid_name or None."""
    best_grid = None
    best_dist = 1e10
    for grid, Z_boost in BOOST_GRIDS.items():
        dist = abs(np.log10(Z) - np.log10(Z_boost))
        if dist < best_dist:
            best_dist = dist
            best_grid = grid
    if best_dist <= Z_MATCH_THRESHOLD:
        return best_grid
    return None


def _resample_track_onto_tau_grid(track, tau_grid):
    """
    Resample a raw track dict onto a fractional-lifetime grid tau in [0, 1].
    tau=0 is ZAMS, tau=1 is death. All quantities are interpolated in tau-space.
    Returns a dict similar to resample_onto_age_grid output but indexed by tau.
    """
    age = track['age_yr']
    lifetime = age[-1]

    mask = age > 0
    if np.sum(mask) < 2:
        return None

    age_valid = age[mask]
    # Compute fractional lifetime for each track point
    tau_track = age_valid / lifetime  # [0, 1]

    N = len(tau_grid)
    result = {}

    # Log-scale quantities
    for key in ['logL', 'logTeff', 'logR_cm', 'log_Mdot']:
        vals = track[key][mask]
        result[key] = np.interp(tau_grid, tau_track, vals, left=vals[0], right=vals[-1])

    # Radiation: store as log
    for key in ['Q_ion', 'L_FUV', 'L_LW', 'L_bol']:
        vals = track[key][mask]
        log_vals = np.where(vals > 0, np.log10(vals), LOG_FLOOR)
        result['log_' + key] = np.interp(tau_grid, tau_track, log_vals,
                                         left=log_vals[0], right=log_vals[-1])

    # Linear quantities
    for key in ['M_current', 'v_wind']:
        vals = track[key][mask]
        result[key] = np.interp(tau_grid, tau_track, vals, left=vals[0], right=vals[-1])

    # Surface abundances
    surf = track['surface'][mask]
    surf_interp = np.zeros((N, N_ELEM))
    for ie in range(N_ELEM):
        surf_interp[:, ie] = np.interp(tau_grid, tau_track, surf[:, ie],
                                       left=surf[0, ie], right=surf[-1, ie])
    result['surface'] = surf_interp

    # Evolutionary phase (nearest-neighbor, discrete quantity)
    if 'phase' in track:
        ph = track['phase'][mask]
        ph_interp = np.zeros(N, dtype=int)
        for i in range(N):
            idx = np.argmin(np.abs(tau_track - tau_grid[i]))
            ph_interp[i] = ph[idx]
    else:
        ph_interp = np.ones(N, dtype=int)
    result['phase'] = ph_interp

    # Scalars
    result['lifetime'] = lifetime
    alive_raw = track['M_current'][mask]
    result['M_preSN_scalar'] = alive_raw[-1] if len(alive_raw) > 0 else 0.0
    result['M_CO_core'] = track.get('M_CO_core', 0.0)
    result['M_He_core'] = track.get('M_He_core', 0.0)

    # Phase transitions stored as fractional lifetime
    pt_raw = track.get('phase_transitions', {})
    pt_tau = {}
    for key, val in pt_raw.items():
        pt_tau[key] = val / lifetime if lifetime > 0 else 0.0
    result['phase_transitions'] = pt_tau

    return result


def _interpolate_two_tracks_tau(t_lo, t_hi, f, log_age_grid):
    """
    Interpolate between two raw tracks in fractional-lifetime (tau) space,
    then map back to physical age and resample onto log_age_grid.

    This avoids the stitching bug where the heavier star dies first in absolute
    age, causing an unphysical mass jump when falling back to the lighter star's
    M_current.

    f: interpolation weight (0 = t_lo, 1 = t_hi) in log-mass space.
    """
    N_TAU = 1024  # fine tau grid for interpolation
    tau_grid = np.linspace(0.0, 1.0, N_TAU)

    r_lo = _resample_track_onto_tau_grid(t_lo, tau_grid)
    r_hi = _resample_track_onto_tau_grid(t_hi, tau_grid)

    if r_lo is None and r_hi is None:
        return None
    if r_lo is None:
        return resample_onto_age_grid(t_hi, log_age_grid)
    if r_hi is None:
        return resample_onto_age_grid(t_lo, log_age_grid)

    # --- Interpolate all quantities at each tau point ---
    interp = {}

    # Log-scale quantities
    for key in ['logL', 'logTeff', 'logR_cm', 'log_Mdot',
                'log_Q_ion', 'log_L_FUV', 'log_L_LW', 'log_L_bol']:
        lo, hi = r_lo[key], r_hi[key]
        alive = (lo > LOG_FLOOR + 1) & (hi > LOG_FLOOR + 1)
        val = np.full_like(lo, LOG_FLOOR)
        val[alive] = lo[alive] + f * (hi[alive] - lo[alive])
        only_lo = (lo > LOG_FLOOR + 1) & (hi <= LOG_FLOOR + 1)
        only_hi = (lo <= LOG_FLOOR + 1) & (hi > LOG_FLOOR + 1)
        val[only_lo] = lo[only_lo]
        val[only_hi] = hi[only_hi]
        interp[key] = val

    # Linear quantities
    for key in ['M_current', 'v_wind']:
        interp[key] = r_lo[key] + f * (r_hi[key] - r_lo[key])

    # Surface abundances
    surf_lo, surf_hi = r_lo['surface'], r_hi['surface']
    surf = np.zeros_like(surf_lo)
    for ie in range(N_ELEM):
        surf[:, ie] = surf_lo[:, ie] + f * (surf_hi[:, ie] - surf_lo[:, ie])
    interp['surface'] = surf

    # Phase (nearest from lo track in tau-space; both are aligned so this is fine)
    interp['phase'] = r_lo['phase'].copy()

    # Interpolate lifetime in log-space
    lifetime = np.exp(np.log(r_lo['lifetime']) + f * (np.log(r_hi['lifetime']) - np.log(r_lo['lifetime'])))
    interp['lifetime'] = lifetime

    # Scalars
    interp['M_preSN_scalar'] = r_lo['M_preSN_scalar'] + f * (r_hi['M_preSN_scalar'] - r_lo['M_preSN_scalar'])
    interp['M_CO_core'] = r_lo['M_CO_core'] + f * (r_hi['M_CO_core'] - r_lo['M_CO_core'])
    interp['M_He_core'] = r_lo['M_He_core'] + f * (r_hi['M_He_core'] - r_lo['M_He_core'])

    # Phase transitions: interpolate in log-space (stored as fractional tau)
    pt_lo = r_lo.get('phase_transitions', {})
    pt_hi = r_hi.get('phase_transitions', {})
    pt = {}
    for key in pt_lo:
        v_lo = pt_lo[key]
        v_hi = pt_hi.get(key, v_lo)
        if v_lo > 0 and v_hi > 0:
            pt_tau = np.exp(np.log(v_lo) + f * (np.log(v_hi) - np.log(v_lo)))
        else:
            pt_tau = v_lo + f * (v_hi - v_lo)
        # Convert from fractional to physical age
        pt[key] = pt_tau * lifetime
    interp['phase_transitions'] = pt

    # --- Map from tau-space back to physical age ---
    # Convert tau grid to physical age: age = tau * lifetime
    age_phys = tau_grid * lifetime
    # Remove tau=0 (age=0) to avoid log(0)
    valid = age_phys > 0
    if np.sum(valid) < 2:
        return None
    log_age_phys = np.log10(age_phys[valid])
    log_lifetime = np.log10(max(lifetime, 1.0))

    N = len(log_age_grid)
    result = {}

    # Resample log-scale quantities onto log_age_grid
    for key in ['logL', 'logTeff', 'logR_cm', 'log_Mdot',
                'log_Q_ion', 'log_L_FUV', 'log_L_LW', 'log_L_bol']:
        vals = interp[key][valid]
        resampled = np.interp(log_age_grid, log_age_phys, vals, left=vals[0], right=LOG_FLOOR)
        resampled[log_age_grid > log_lifetime] = LOG_FLOOR
        result[key] = resampled

    # Resample linear quantities
    for key in ['M_current', 'v_wind']:
        vals = interp[key][valid]
        resampled = np.interp(log_age_grid, log_age_phys, vals, left=vals[0], right=0.0)
        resampled[log_age_grid > log_lifetime] = 0.0
        result[key] = resampled

    # Resample surface abundances
    surf_valid = interp['surface'][valid]
    surf_resampled = np.zeros((N, N_ELEM))
    for ie in range(N_ELEM):
        s = np.interp(log_age_grid, log_age_phys, surf_valid[:, ie],
                      left=surf_valid[0, ie], right=0.0)
        s[log_age_grid > log_lifetime] = 0.0
        surf_resampled[:, ie] = s
    result['surface'] = surf_resampled

    # Resample phase (nearest-neighbor)
    ph = interp['phase'][valid]
    ph_resampled = np.full(N, 7, dtype=int)
    for i in range(N):
        if log_age_grid[i] <= log_lifetime:
            idx = np.argmin(np.abs(log_age_phys - log_age_grid[i]))
            ph_resampled[i] = ph[idx]
    result['phase'] = ph_resampled

    # ── Enforce Mdot / M_current consistency ──
    M_cur = result['M_current']
    age_grid_yr = 10**log_age_grid
    alive_mask = log_age_grid <= log_lifetime
    mdot_consistent = np.full(N, -30.0)
    for i in range(N):
        if not alive_mask[i]:
            continue
        if i < N - 1 and alive_mask[i + 1]:
            dM = M_cur[i] - M_cur[i + 1]
            dt = age_grid_yr[i + 1] - age_grid_yr[i]
        elif i > 0 and alive_mask[i - 1]:
            dM = M_cur[i - 1] - M_cur[i]
            dt = age_grid_yr[i] - age_grid_yr[i - 1]
        else:
            continue
        if dM > 0 and dt > 0:
            mdot_consistent[i] = np.log10(dM / dt)
    result['log_Mdot'] = mdot_consistent

    result['lifetime'] = lifetime
    result['M_preSN_scalar'] = interp['M_preSN_scalar']
    result['M_CO_core'] = interp['M_CO_core']
    result['M_He_core'] = interp['M_He_core']
    result['phase_transitions'] = interp['phase_transitions']

    return result


def _interpolate_two_tracks(r_lo, r_hi, f):
    """Interpolate between two resampled tracks. f=0 → r_lo, f=1 → r_hi.
    NOTE: Legacy absolute-age interpolation, kept for reference.
    New code uses _interpolate_two_tracks_tau for mass interpolation."""
    result = {}
    for key in ['logL', 'logTeff', 'logR_cm', 'log_Mdot',
                'log_Q_ion', 'log_L_FUV', 'log_L_LW', 'log_L_bol']:
        lo, hi = r_lo[key], r_hi[key]
        alive = (lo > LOG_FLOOR + 1) & (hi > LOG_FLOOR + 1)
        interp = np.full_like(lo, LOG_FLOOR)
        interp[alive] = lo[alive] + f * (hi[alive] - lo[alive])
        only_lo = (lo > LOG_FLOOR + 1) & (hi <= LOG_FLOOR + 1)
        only_hi = (lo <= LOG_FLOOR + 1) & (hi > LOG_FLOOR + 1)
        interp[only_lo] = lo[only_lo]
        interp[only_hi] = hi[only_hi]
        result[key] = interp

    for key in ['M_current', 'v_wind']:
        lo, hi = r_lo[key], r_hi[key]
        alive_lo = lo > 0
        alive_hi = hi > 0
        interp = np.zeros_like(lo)
        both = alive_lo & alive_hi
        interp[both] = lo[both] + f * (hi[both] - lo[both])
        only_lo = alive_lo & ~alive_hi
        only_hi = ~alive_lo & alive_hi
        interp[only_lo] = lo[only_lo]
        interp[only_hi] = hi[only_hi]
        result[key] = interp

    surf_lo, surf_hi = r_lo['surface'], r_hi['surface']
    result['surface'] = np.zeros_like(surf_lo)
    for ie in range(N_ELEM):
        lo_s, hi_s = surf_lo[:, ie], surf_hi[:, ie]
        alive_lo = lo_s > 0
        alive_hi = hi_s > 0
        both = alive_lo & alive_hi
        s = np.zeros_like(lo_s)
        s[both] = lo_s[both] + f * (hi_s[both] - lo_s[both])
        s[alive_lo & ~alive_hi] = lo_s[alive_lo & ~alive_hi]
        s[~alive_lo & alive_hi] = hi_s[~alive_lo & alive_hi]
        result['surface'][:, ie] = s

    result['lifetime'] = np.exp(np.log(r_lo['lifetime']) + f * (np.log(r_hi['lifetime']) - np.log(r_lo['lifetime'])))
    result['M_preSN_scalar'] = r_lo['M_preSN_scalar'] + f * (r_hi['M_preSN_scalar'] - r_lo['M_preSN_scalar'])
    result['M_CO_core'] = r_lo['M_CO_core'] + f * (r_hi['M_CO_core'] - r_lo['M_CO_core'])
    result['M_He_core'] = r_lo['M_He_core'] + f * (r_hi['M_He_core'] - r_lo['M_He_core'])

    result['phase'] = r_lo['phase'].copy()

    pt_lo = r_lo.get('phase_transitions', {})
    pt_hi = r_hi.get('phase_transitions', {})
    pt = {}
    for key in pt_lo:
        v_lo = pt_lo[key]
        v_hi = pt_hi.get(key, v_lo)
        if v_lo > 0 and v_hi > 0:
            pt[key] = np.exp(np.log(v_lo) + f * (np.log(v_hi) - np.log(v_lo)))
        else:
            pt[key] = v_lo + f * (v_hi - v_lo)
    result['phase_transitions'] = pt

    return result


def _interpolate_to_mass(track_loader, index, key_prefix, M_target, M_min, log_age_grid):
    """
    Generic mass interpolation for any track source.
    track_loader: function(filepath) → common dict or None
    index: dict of (key_prefix, M) → filepath
    M_min: minimum mass to consider from this source
    Returns resampled dict on log_age_grid, or None.

    Uses fractional-lifetime (tau) interpolation when both bracket tracks
    exist, to avoid the stitching bug where the heavier star dies first.
    """
    avail = sorted([m for (k, m) in index if k == key_prefix and m >= M_min])
    if not avail:
        return None

    if M_target < avail[0]:
        return None

    # Above highest available: use highest
    if M_target >= avail[-1]:
        fp = index.get((key_prefix, avail[-1]))
        if fp is None:
            return None
        track = track_loader(fp)
        if track is None:
            return None
        return resample_onto_age_grid(track, log_age_grid)

    # Find bracketing masses
    for k in range(len(avail) - 1):
        if avail[k] <= M_target <= avail[k+1]:
            M_lo, M_hi = avail[k], avail[k+1]
            break
    else:
        return None

    # Exact match
    if M_target == M_lo:
        fp = index.get((key_prefix, M_lo))
        if fp:
            track = track_loader(fp)
            if track is not None:
                return resample_onto_age_grid(track, log_age_grid)
    if M_target == M_hi:
        fp = index.get((key_prefix, M_hi))
        if fp:
            track = track_loader(fp)
            if track is not None:
                return resample_onto_age_grid(track, log_age_grid)

    # Load both bracketing tracks (raw, not yet resampled)
    fp_lo = index.get((key_prefix, M_lo))
    fp_hi = index.get((key_prefix, M_hi))
    t_lo = track_loader(fp_lo) if fp_lo else None
    t_hi = track_loader(fp_hi) if fp_hi else None

    if t_lo is None and t_hi is None:
        return None

    # If only one track exists, fall back to simple resampling
    if t_lo is None:
        return resample_onto_age_grid(t_hi, log_age_grid)
    if t_hi is None:
        return resample_onto_age_grid(t_lo, log_age_grid)

    # Both tracks exist: interpolate in fractional-lifetime (tau) space
    f = (np.log10(M_target) - np.log10(M_lo)) / (np.log10(M_hi) - np.log10(M_lo))
    return _interpolate_two_tracks_tau(t_lo, t_hi, f, log_age_grid)


def interpolate_boost_to_mass(boost_index, grid_name, M_target, log_age_grid):
    """Interpolate BoOST tracks to a target mass (fallback only)."""
    return _interpolate_to_mass(
        boost_track_to_common, boost_index, grid_name, M_target,
        9.0, log_age_grid)  # BoOST starts at 9 Msun


def interpolate_mist_to_mass(mist_index, feh_str, M_target, log_age_grid):
    """Interpolate MIST tracks to a target mass."""
    return _interpolate_to_mass(
        mist_track_to_common, mist_index, feh_str, M_target,
        0.0, log_age_grid)


# ═══════════════════════════════════════════════════════════════
# SECTION 7: Yield tables (from process_yields.py logic)
# ═══════════════════════════════════════════════════════════════

ISOTOPE_TO_ELEMENT_CL = {
    'H': 'H', 'H2': 'H', 'H3': 'H',
    'He3': 'He', 'He4': 'He',
    'C12': 'C', 'C13': 'C', 'C14': 'C',
    'N13': 'N', 'N14': 'N', 'N15': 'N',
    'O16': 'O', 'O17': 'O', 'O18': 'O',
    'F19': 'F',
    'Ne20': 'Ne', 'Ne21': 'Ne', 'Ne22': 'Ne',
    'Na23': 'Na',
    'Mg24': 'Mg', 'Mg25': 'Mg', 'Mg26': 'Mg',
    'Al26': 'Al', 'Al27': 'Al',
    'Si28': 'Si', 'Si29': 'Si', 'Si30': 'Si',
    'S32': 'S', 'S33': 'S', 'S34': 'S', 'S36': 'S',
    'Ca40': 'Ca', 'Ca42': 'Ca', 'Ca43': 'Ca', 'Ca44': 'Ca', 'Ca46': 'Ca', 'Ca48': 'Ca',
    'Ti46': 'Ti', 'Ti47': 'Ti', 'Ti48': 'Ti', 'Ti49': 'Ti', 'Ti50': 'Ti',
    'Fe54': 'Fe', 'Fe56': 'Fe', 'Fe57': 'Fe', 'Fe58': 'Fe',
    'Ni56': 'Fe',
}

ISOTOPE_TO_ELEMENT_KAR = {
    'p': 'H', 'd': 'H',
    'he3': 'He', 'he4': 'He',
    'c12': 'C', 'c13': 'C',
    'n13': 'N', 'n14': 'N', 'n15': 'N',
    'o16': 'O', 'o17': 'O', 'o18': 'O',
    'f19': 'F',
    'ne20': 'Ne', 'ne21': 'Ne', 'ne22': 'Ne',
    'na23': 'Na',
    'mg24': 'Mg', 'mg25': 'Mg', 'mg26': 'Mg',
    'al26': 'Al', 'al27': 'Al',
    'si28': 'Si', 'si29': 'Si', 'si30': 'Si',
    's32': 'S', 's33': 'S', 's34': 'S',
    'ca40': 'Ca', 'ca42': 'Ca', 'ca43': 'Ca', 'ca44': 'Ca',
    'ti46': 'Ti', 'ti48': 'Ti',
    'fe54': 'Fe', 'fe56': 'Fe', 'fe57': 'Fe', 'fe58': 'Fe',
    'ni56': 'Fe',
}


def load_CL18_yields(cl_dir):
    """Load Chieffi & Limongi 2018 yields and pre-SN properties."""
    CL_MASSES = np.array([13, 15, 20, 25, 30, 40, 60, 80, 120], dtype=float)

    # table8: isotopic yields
    raw = {}
    with open(os.path.join(cl_dir, 'table8.dat')) as f:
        for line in f:
            vel = int(line[0:3])
            feh = int(line[4:6])
            iso = line[7:12].strip()
            vals = [float(line[13+i*11:13+(i+1)*11]) for i in range(9)]
            raw[(vel, feh, iso)] = np.array(vals)

    # Aggregate to elements (non-rotating)
    fehs = sorted(set(k[1] for k in raw.keys()))
    elem_yields = {}
    for feh in fehs:
        for elem in TRACKED_ELEMENTS:
            elem_yields[(feh, elem)] = np.zeros(9)
    for (v, feh, iso), vals in raw.items():
        if v != 0:
            continue
        elem = ISOTOPE_TO_ELEMENT_CL.get(iso)
        if elem and elem in TRACKED_ELEMENTS:
            elem_yields[(feh, elem)] += vals

    # table7: pre-SN properties
    presn = []
    with open(os.path.join(cl_dir, 'table7.dat')) as f:
        for line in f:
            presn.append({
                'vel': int(line[0:3]), 'feh': int(line[4:6]),
                'mass': int(line[7:10]), 'fe_core': float(line[23:27]),
                'xi25': float(line[34:38]),
            })

    return elem_yields, fehs, CL_MASSES, presn


def load_karakas_yields(kar_dir):
    """Load Karakas 2010 AGB yields."""
    elem_yields = {}
    wd_masses = {}
    for table_file in ['tablea2.dat', 'tablea3.dat', 'tablea4.dat', 'tablea5.dat']:
        path = os.path.join(kar_dir, table_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                M0 = float(line[0:4])
                Z0 = float(line[5:11])
                M_final = float(line[12:17])
                iso = line[21:25].strip().lower()
                net_yield = float(line[34:48])
                wd_masses[(Z0, M0)] = M_final
                elem = ISOTOPE_TO_ELEMENT_KAR.get(iso)
                if elem and elem in TRACKED_ELEMENTS:
                    key = (Z0, M0, elem)
                    elem_yields[key] = elem_yields.get(key, 0.0) + net_yield
    return elem_yields, wd_masses


# K&L 2016 element name → our tracked element mapping
KL16_ELEM_MAP = {
    'p': 'H', 'he': 'He', 'c': 'C', 'n': 'N', 'o': 'O', 'f': 'F',
    'ne': 'Ne', 'na': 'Na', 'mg': 'Mg', 'al': 'Al', 'si': 'Si',
    's': 'S', 'ca': 'Ca', 'ti': 'Ti', 'fe': 'Fe',
}


def load_kl2016_yields(kl_dir):
    """
    Load Karakas & Lugaro 2016 AGB yields (Z=0.007, 0.014, 0.03).

    Data format gives total expelled mass per element. We convert to net yields:
        net_yield(i) = Mass(i)_expelled - X_init(i) * M_expelled
    where X_init is the initial composition (solar-scaled metals).

    For each mass with multiple M_mix variants, selects the standard model
    per Table 2 of K&L 2016:
        M <= 3:        M_mix = 2e-3
        3 < M <= 4:    M_mix = 1e-3
        4 < M < 5:     M_mix = 1e-4
        M >= 5:        M_mix = 0
    """
    def standard_mmix(M):
        if M <= 3.0: return 2e-3
        elif M <= 4.0: return 1e-3
        elif M < 5.0: return 1e-4
        else: return 0.0

    elem_yields = {}
    wd_masses = {}

    for zfile in ['yield_z007.dat', 'yield_z014.dat', 'yield_z03.dat']:
        path = os.path.join(kl_dir, zfile)
        if not os.path.exists(path):
            continue

        # Parse all models from file
        models = []
        with open(path) as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('# Initial mass'):
                m = re.search(
                    r'Initial mass = ([\d.]+).*Z =\s+([\d.]+).*Y =\s+([\d.]+)'
                    r'.*M_mix = ([\d.Ee+-]+)', line)
                if not m:
                    i += 1
                    continue
                M_init = float(m.group(1))
                Z = float(m.group(2))
                Y = float(m.group(3))
                M_mix = float(m.group(4))
                i += 1
                m2 = re.search(
                    r'Final mass = ([\d.]+).*Mass expelled =\s+([\d.]+)',
                    lines[i])
                M_final = float(m2.group(1))
                M_expelled = float(m2.group(2))
                i += 2  # skip column header line
                elems = {}
                while i < len(lines) and not lines[i].startswith('#'):
                    parts = lines[i].split()
                    if len(parts) >= 7:
                        elems[parts[0].lower()] = float(parts[6])
                    i += 1
                models.append({
                    'M_init': M_init, 'Z': Z, 'Y': Y, 'M_mix': M_mix,
                    'M_final': M_final, 'M_expelled': M_expelled,
                    'elems': elems,
                })
            else:
                i += 1

        # Select standard M_mix model for each mass
        selected = {}
        for mod in models:
            M = mod['M_init']
            std = standard_mmix(M)
            key = (mod['Z'], M)
            if key not in selected or \
               abs(mod['M_mix'] - std) < abs(selected[key]['M_mix'] - std):
                selected[key] = mod

        # Convert to net yields
        for (Z, M), mod in selected.items():
            X_H = 1.0 - mod['Y'] - Z
            M_exp = mod['M_expelled']
            wd_masses[(Z, M)] = mod['M_final']

            for kl_el, our_el in KL16_ELEM_MAP.items():
                if our_el not in TRACKED_ELEMENTS:
                    continue
                mass_expelled = mod['elems'].get(kl_el, 0.0)
                # Initial mass fraction (solar-scaled)
                if our_el == 'H':
                    X_init = X_H
                elif our_el == 'He':
                    X_init = mod['Y']
                else:
                    X_init = SOLAR_METAL_FRAC.get(our_el, 0.0) * Z
                net = mass_expelled - X_init * M_exp
                key = (Z, M, our_el)
                elem_yields[key] = elem_yields.get(key, 0.0) + net

    return elem_yields, wd_masses


def _parse_parsec2_init_comp(ejecta_dir):
    """
    Parse initial elemental mass fractions from PARSEC v2 ejecta headers.
    Returns dict: Z → {elem: mass_fraction}.
    Line 2 of each total_ejecta file has the initial composition as mass fractions
    in the same column order as the data (H, HE3, HE4, ..., ZN).
    """
    # Column indices matching ELEM_COLS_P2 in load_parsec2_ejecta
    INIT_COLS = {
        'H': [7], 'He': [8, 9], 'C': [12, 13], 'N': [14, 15], 'O': [16, 17, 18],
        'F': [19], 'Ne': [20, 21, 22], 'Na': [23], 'Mg': [24, 25, 26],
        'Al': [27, 28], 'Si': [29, 30], 'S': [32], 'Ca': [36], 'Ti': [38], 'Fe': [42],
    }
    result = {}
    for zdir in sorted(glob(os.path.join(ejecta_dir, 'ejecta_Z*'))):
        m = re.search(r'ejecta_Z([\d.Ee-]+)', zdir)
        if not m:
            continue
        Z = float(m.group(1))
        total_files = glob(os.path.join(zdir, '*total_ejecta.dat'))
        if not total_files:
            continue
        with open(total_files[0]) as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue
        # Line 2 (index 1) has initial composition as "# val val val ... SN val val ..."
        parts = lines[1].replace('#', '').split()
        # Find "SN" marker and parse values after it
        try:
            sn_idx = parts.index('SN')
        except ValueError:
            continue
        vals = parts[sn_idx + 1:]  # mass fractions in isotope order
        comp = {}
        for elem, cols in INIT_COLS.items():
            total = 0.0
            for c in cols:
                idx = c - 7  # offset: data cols start at 7, init vals start at 0
                if idx < len(vals):
                    total += float(vals[idx])
            comp[elem] = total
        result[Z] = comp
    return result


def load_parsec2_ejecta(ejecta_dir):
    """
    Load PARSEC v2 VMS ejecta tables (14-600+ Msun).
    Returns dict: (Z, M_init) → {'yields': {elem: Msun}, 'M_rem', 'sn_type', ...}
    """
    # Column indices (0-based) after splitting data line
    ELEM_COLS_P2 = {
        'H': [7], 'He': [8, 9], 'C': [12, 13], 'N': [14, 15], 'O': [16, 17, 18],
        'F': [19], 'Ne': [20, 21, 22], 'Na': [23], 'Mg': [24, 25, 26],
        'Al': [27, 28], 'Si': [29, 30], 'S': [32], 'Ca': [36], 'Ti': [38], 'Fe': [42],
    }
    # Map PARSEC v2 SN type strings to our remnant type codes
    # CCSN→2(NS), FSN→3(BH), PPISN→4, PISN→5, DBH→6(BH), NOT→0(WD)
    SNT_MAP = {'CCSN': 2, 'FSN': 3, 'PPISN': 4, 'PISN': 5, 'DBH': 6, 'NOT': 0}

    data = {}
    for zdir in sorted(glob(os.path.join(ejecta_dir, 'ejecta_Z*'))):
        m = re.search(r'ejecta_Z([\d.Ee-]+)', zdir)
        if not m:
            continue
        Z = float(m.group(1))
        total_files = glob(os.path.join(zdir, '*total_ejecta.dat'))
        if not total_files:
            continue
        with open(total_files[0]) as f:
            lines = f.readlines()
        for line in lines[4:]:  # skip 4 header lines
            parts = line.split()
            if len(parts) < 47:
                continue
            M_init = float(parts[0])
            sn_type_str = parts[6]
            yields = {}
            for elem, cols in ELEM_COLS_P2.items():
                yields[elem] = sum(float(parts[c]) for c in cols)
            # Ni56 → Fe (radioactive decay)
            yields['Fe'] += float(parts[44])  # col 44 = NI
            data[(Z, M_init)] = {
                'yields': yields,
                'M_fin': float(parts[1]),
                'M_HE': float(parts[2]),
                'M_CO': float(parts[3]),
                'M_rem': float(parts[4]),
                'sn_type_str': sn_type_str,
                'sn_type': SNT_MAP.get(sn_type_str, 3),
            }
    return data


def load_nugrid_yields(nugrid_dir):
    """
    Load NuGrid Set1ext element yields (Ritter+ 2018, Fryer12 delay).
    Returns dict: (Z, M) → {'yields': {elem: Msun}, 'Mfinal', 'lifetime'}
    """
    data = {}
    filepath = os.path.join(nugrid_dir, 'element_yield_table_MESAonly_fryer12_delay_total.txt')
    if not os.path.exists(filepath):
        return data
    current = None
    with open(filepath) as f:
        for line in f:
            m = re.match(r'H Table: \(M=([\d.]+),Z=([\d.]+)\)', line)
            if m:
                M, Z = float(m.group(1)), float(m.group(2))
                current = (Z, M)
                data[current] = {'yields': {}, 'Mfinal': 0, 'lifetime': 0}
                continue
            if current is None:
                continue
            lm = re.match(r'H Lifetime: (.+)', line)
            if lm:
                data[current]['lifetime'] = float(lm.group(1))
            fm = re.match(r'H Mfinal: (.+)', line)
            if fm:
                data[current]['Mfinal'] = float(fm.group(1))
            if line.startswith('&'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    elem = parts[0].lstrip('&')
                    try:
                        yld = float(parts[1].lstrip('&'))
                        if elem in TRACKED_ELEMENTS:
                            data[current]['yields'][elem] = yld
                    except ValueError:
                        pass
    return data


def get_limongi25_yields():
    """
    Limongi+2025 CCSNe yields for 9-15 Msun at solar Z.
    From Table 2 of arXiv:2505.22030 (Limongi, Chieffi & Roberti 2025).
    Note: Fe includes Ni56→Fe56 decay.
    """
    # Masses available: 9.22, 10, 11, 12, 13, 15 Msun
    # These are TOTAL ejected masses per isotope (not net yields)
    # Remnant masses from Table 1 (iron core masses)
    masses = [9.22, 10.0, 11.0, 12.0, 13.0, 15.0]
    remnant_masses = [1.263, 1.211, 1.316, 1.333, 1.456, 1.398]
    # Total ejected element masses (sum of relevant isotopes, Msun)
    # Fe = Fe56 + Ni56 (Ni56 decays to Fe56)
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
    return masses, remnant_masses, yields_table


def get_pisn_yields():
    """PISN yields from Heger & Woosley 2002 (approximate)."""
    return {
        150: {'H': 30, 'He': 55, 'C': 0.6, 'N': 0.01, 'O': 15,
              'F': 0.0, 'Ne': 0.3, 'Na': 0.001, 'Mg': 1.5,
              'Al': 0.01, 'Si': 8, 'S': 4, 'Ca': 0.5, 'Ti': 0.01, 'Fe': 5},
        200: {'H': 20, 'He': 60, 'C': 0.3, 'N': 0.01, 'O': 25,
              'F': 0.0, 'Ne': 0.5, 'Na': 0.002, 'Mg': 3,
              'Al': 0.02, 'Si': 15, 'S': 8, 'Ca': 1, 'Ti': 0.02, 'Fe': 15},
        260: {'H': 10, 'He': 50, 'C': 0.1, 'N': 0.01, 'O': 40,
              'F': 0.0, 'Ne': 1, 'Na': 0.005, 'Mg': 5,
              'Al': 0.05, 'Si': 25, 'S': 15, 'Ca': 2, 'Ti': 0.05, 'Fe': 40},
    }


def classify_remnant_initial(M_init, Z, presn, wd_masses):
    """
    Initial classification (before He/CO core masses are known).
    Returns (type, mass).
      0=WD, 1=ECSN(NS), 2=CCSN(NS), 3=CCSN(BH), 4=PPISN(BH), 5=PISN
    """
    if M_init > 120:
        return 5, 0.0
    if M_init < 8:
        best_key, best_dist = None, 1e10
        for (z, m), mwd in wd_masses.items():
            dist = abs(np.log10(z / max(Z, 1e-6))) + abs(m - M_init)
            if dist < best_dist:
                best_dist, best_key = dist, (z, m)
        if best_key and best_dist < 3.0:
            mwd = min(wd_masses[best_key], M_init)  # WD can't exceed progenitor
            return 0, mwd
        return 0, min(0.08 * M_init + 0.489, 1.4, M_init)

    # ECSN: ~8-10 Msun (narrow window)
    if M_init <= 10:
        return 1, 1.25  # ECSN → low-mass NS

    # Core-collapse: NS vs BH from compactness
    feh = np.log10(max(Z, 1e-6) / 0.014)
    feh_grid = [0, -1, -2, -3]
    feh_near = min(feh_grid, key=lambda x: abs(x - feh))
    best = None
    for m in presn:
        if m['vel'] == 0 and m['feh'] == feh_near:
            if best is None or abs(m['mass'] - M_init) < abs(best['mass'] - M_init):
                best = m
    if best:
        if best['xi25'] > 0.35:
            return 3, best['fe_core']  # BH (will be updated with CO core later)
        return 2, 1.4  # NS
    return (3, max(1.5, 0.1 * M_init)) if M_init > 25 else (2, 1.4)


def reclassify_with_cores(rem_type, rem_mass, M_He_core, M_CO_core, M_preSN, M_grid,
                          yield_src=None):
    """
    Update remnant classification using He/CO core masses from tracks.

    Entries with M >= 14 Msun already have PARSEC v2 remnant types
    (yield_src 3 or 4) and are SKIPPED.

    Only applies to M < 14 Msun entries where remnant was set from
    initial mass classification and may benefit from core mass refinement.
    """
    N_Z, N_M = rem_type.shape
    n_pisn_upgrade = 0
    n_ppisn = 0
    n_skipped_p2 = 0

    for iz in range(N_Z):
        for im in range(N_M):
            # Skip entries that got remnant from PARSEC v2 (yield_src 3=CL18+P2rem, 4=P2PISN)
            if yield_src is not None and yield_src[iz, im] in (3, 4):
                n_skipped_p2 += 1
                continue

            mhe = M_He_core[iz, im]
            mco = M_CO_core[iz, im]

            # PISN from He core (overrides ZAMS-based classification)
            if mhe > 65 and rem_type[iz, im] != 0:
                rem_type[iz, im] = 5  # PISN
                rem_mass[iz, im] = 0.0
                n_pisn_upgrade += 1
                continue

            # PPISN from He core
            if mhe > 35 and rem_type[iz, im] in [2, 3]:  # was NS or BH
                rem_type[iz, im] = 4  # PPISN → BH
                # BH mass capped at ~45 Msun (Farmer+ 2019 mass gap)
                rem_mass[iz, im] = min(mco, 45.0)
                n_ppisn += 1
                continue

            # BH: use CO core mass
            if rem_type[iz, im] == 3 and mco > 0:
                rem_mass[iz, im] = mco
                # Sanity: BH can't exceed pre-SN mass
                if M_preSN[iz, im] > 0:
                    rem_mass[iz, im] = min(rem_mass[iz, im], M_preSN[iz, im])

    print(f"  Reclassified: {n_pisn_upgrade} → PISN (He core), {n_ppisn} → PPISN")
    if n_skipped_p2 > 0:
        print(f"  Skipped {n_skipped_p2} PARSEC v2 entries (native SN types)")


# ═══════════════════════════════════════════════════════════════
# Type Ia SN yields and DTD
# ═══════════════════════════════════════════════════════════════

def get_type_ia_yields():
    """
    Type Ia SN yields from W7 model (Iwamoto+ 1999).
    Complete disruption of Chandrasekhar-mass C/O WD.
    Returns dict: element → yield in Msun.
    """
    return {
        'H':  0.0,
        'He': 0.0,
        'C':  0.0486,
        'N':  0.0,
        'O':  0.143,
        'F':  0.0,
        'Ne': 0.00429,
        'Na': 5e-5,     # trace, W7 model
        'Mg': 0.00885,
        'Al': 9e-4,     # W7 model
        'Si': 0.156,
        'S':  0.0864,
        'Ca': 0.0126,
        'Ti': 3.6e-4,   # W7 model
        'Fe': 0.744,    # includes 56Ni → 56Fe decay
    }

TYPE_IA_DTD = {
    't_min_Myr': 40.0,           # minimum delay time
    'slope': -1.0,               # P(t) ∝ t^slope for t > t_min
    'rate_per_Msun': 1.3e-3,     # Ia per Msun of stars formed (Maoz & Mannucci 2012)
    'E_SN_erg': 1.0e51,          # explosion energy
    'ejecta_mass_Msun': 1.378,   # total ejecta (Chandrasekhar mass WD)
}

# ═══════════════════════════════════════════════════════════════
# Magnetar prescription (parametric, for C code)
# ═══════════════════════════════════════════════════════════════

MAGNETAR_PARAMS = {
    'description': 'Parametric magnetar channel: fraction of NS-forming CCSNe produce magnetars',
    'f_mag_base': 0.10,       # base magnetar fraction (~10% of NS births)
    'mass_boost_exp': 0.5,    # f_mag scales as (M/20)^exp for M > 20 Msun
    'Z_boost_exp': -0.3,      # f_mag scales as (Z/Zsun)^exp (more magnetars at low Z)
    'E_total_erg': 1e52,      # fixed total energy for magnetar SN (SN + spindown combined)
    'note': 'Magnetar SN: E_total = 1e52 erg (flat). No separate spindown — '
            'deposited at SN time. Low Z → more magnetars. '
            'Without rotating tracks this is a statistical prescription.',
}


def _interp_p2_yields(p2_ejecta, Z_p2, M, p2_Ms, init_comp):
    """
    Interpolate PARSEC v2 ejecta to target mass M.
    Returns (net_yields_dict, rem_mass, sn_type) or None if no data.
    Converts total ejected → net yields using initial composition.
    """
    if not p2_Ms:
        return None
    X_init = init_comp.get(Z_p2, {})

    # Clamp or interpolate
    if M <= p2_Ms[0]:
        d = p2_ejecta[(Z_p2, p2_Ms[0])]
        M_ej = p2_Ms[0] - d['M_rem']
        net = {}
        for elem in TRACKED_ELEMENTS:
            net[elem] = d['yields'].get(elem, 0.0) - X_init.get(elem, 0.0) * M_ej
        return net, d['M_rem'], d['sn_type']
    if M >= p2_Ms[-1]:
        d = p2_ejecta[(Z_p2, p2_Ms[-1])]
        M_ej = p2_Ms[-1] - d['M_rem']
        net = {}
        for elem in TRACKED_ELEMENTS:
            net[elem] = d['yields'].get(elem, 0.0) - X_init.get(elem, 0.0) * M_ej
        return net, d['M_rem'], d['sn_type']

    # Bracket and linearly interpolate
    idx = int(np.searchsorted(p2_Ms, M)) - 1
    M_lo, M_hi = p2_Ms[idx], p2_Ms[idx + 1]
    f = (M - M_lo) / (M_hi - M_lo)
    d_lo = p2_ejecta[(Z_p2, M_lo)]
    d_hi = p2_ejecta[(Z_p2, M_hi)]

    M_rem = d_lo['M_rem'] * (1 - f) + d_hi['M_rem'] * f
    # SN type from nearest mass
    sn_type = d_lo['sn_type'] if f < 0.5 else d_hi['sn_type']
    # Override: if either bracket is PISN (type 5), check He core boundary
    if d_lo['sn_type'] == 5 or d_hi['sn_type'] == 5:
        # PISN is complete disruption — use the actual bracket's type
        sn_type = d_lo['sn_type'] if f < 0.5 else d_hi['sn_type']
        if sn_type == 5:
            M_rem = 0.0

    M_ej_lo = M_lo - d_lo['M_rem']
    M_ej_hi = M_hi - d_hi['M_rem']
    net = {}
    for elem in TRACKED_ELEMENTS:
        y_lo = d_lo['yields'].get(elem, 0.0) - X_init.get(elem, 0.0) * M_ej_lo
        y_hi = d_hi['yields'].get(elem, 0.0) - X_init.get(elem, 0.0) * M_ej_hi
        net[elem] = y_lo * (1 - f) + y_hi * f
    return net, M_rem, sn_type


def build_yield_grid(Z_grid, M_grid, base_dir):
    """
    Build yield + remnant arrays.

    Yields (nucleosynthesis):
      M < 8:      Karakas AGB (K&L 2016 for Z>=0.007, K2010 for Z<0.007)
      8 ≤ M < 14: Limongi+2025 CCSNe + NuGrid Z-scaling
      M ≥ 14:     CL18 explosion yields (13-120 Msun), PARSEC v2 for PISN

    Remnants (what the star becomes):
      M < 8:      WD (from core mass)
      8 ≤ M < 14: ECSN(NS) or CCSN(NS)
      M ≥ 14:     PARSEC v2 verdict (CCSN/FSN/PPISN/PISN/DBH)
                   → FSN/DBH: yields zeroed (failed SN, no ejecta)
                   → CCSN: CL18 yields kept
                   → PPISN: CL18 yields scaled by ejecta fraction
                   → PISN: PARSEC v2 yields (complete disruption)
    """
    # Load all yield sources
    print("  Loading Karakas 2010 AGB yields...", flush=True)
    kar_yields, kar_wd = load_karakas_yields(os.path.join(base_dir, 'karakas2010'))

    # Load K&L 2016 and merge (K&L 2016 preferred for Z >= 0.007)
    kl16_dir = os.path.join(base_dir, 'karakas_lugaro2016', 'yields')
    if os.path.isdir(kl16_dir):
        print("  Loading Karakas & Lugaro 2016 AGB yields (Z=0.007,0.014,0.03)...",
              flush=True)
        kl16_yields, kl16_wd = load_kl2016_yields(kl16_dir)
        n_kl16 = len(set((k[0], k[1]) for k in kl16_yields))
        print(f"    {n_kl16} (Z, M) grid points from K&L 2016", flush=True)
        # Merge: K&L 2016 overwrites K2010 at overlapping Z
        kar_yields.update(kl16_yields)
        kar_wd.update(kl16_wd)
    else:
        print("  WARNING: K&L 2016 data not found, using K2010 only", flush=True)

    kar_Z = sorted(set(k[0] for k in kar_yields.keys()))
    kar_M = sorted(set(k[1] for k in kar_yields.keys()))

    print("  Loading PARSEC v2 VMS ejecta (for remnants + PISN yields)...", flush=True)
    p2_ejecta = load_parsec2_ejecta(os.path.join(base_dir, 'parsec2_vms'))
    p2_Zs = sorted(set(k[0] for k in p2_ejecta.keys()))
    p2_init_comp = _parse_parsec2_init_comp(os.path.join(base_dir, 'parsec2_vms'))

    print("  Loading NuGrid Set1ext yields...", flush=True)
    nugrid = load_nugrid_yields(os.path.join(base_dir, 'nugrid'))
    nugrid_Zs = sorted(set(k[0] for k in nugrid.keys()))
    NUGRID_REF_Z = 0.02  # NuGrid's approximate solar

    print("  Loading Limongi+2025 CCSNe yields...", flush=True)
    lim25_masses, lim25_remnants, lim25_yields = get_limongi25_yields()

    print("  Loading C&L 2018 yields + pre-SN data...", flush=True)
    cl_yields, cl_fehs, CL_MASSES, cl_presn = load_CL18_yields(
        os.path.join(base_dir, 'chieffi_limongi2018'))

    # Solar initial composition for Limongi25 total→net conversion
    X_init_solar = p2_init_comp.get(0.014, {})
    if not X_init_solar:
        print("  WARNING: no PARSEC v2 Z=0.014 initial composition, using Z=0.01")
        X_init_solar = p2_init_comp.get(0.01, {})

    # Pre-compute NuGrid Z-scaling factors (ratio vs solar for 12 Msun)
    nugrid_ref = nugrid.get((NUGRID_REF_Z, 12.0), {}).get('yields', {})
    nugrid_z_scale = {}
    for Z_ng in nugrid_Zs:
        y_z = nugrid.get((Z_ng, 12.0), {}).get('yields', {})
        ratios = {}
        for elem in TRACKED_ELEMENTS:
            ref_val = nugrid_ref.get(elem, 0)
            z_val = y_z.get(elem, 0)
            if ref_val > 0 and z_val > 0:
                ratios[elem] = z_val / ref_val
            else:
                ratios[elem] = 1.0
        nugrid_z_scale[Z_ng] = ratios

    N_Z, N_M = len(Z_grid), len(M_grid)
    yields = np.zeros((N_Z, N_M, N_ELEM))
    rem_type = np.zeros((N_Z, N_M), dtype=int)
    rem_mass = np.zeros((N_Z, N_M))
    yield_src = np.zeros((N_Z, N_M), dtype=int)
    # yield_src: 0=none, 1=Karakas_AGB, 2=Limongi25+NuGrid, 3=CL18, 4=PARSEC_v2_PISN

    n_src = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for iz, Z in enumerate(Z_grid):
        # Nearest Z in each source
        log_Z = np.log10(max(Z, 1e-12))
        Z_p2 = min(p2_Zs, key=lambda z: abs(np.log10(max(z, 1e-12)) - log_Z)) if p2_Zs else None
        Z_ng = min(nugrid_Zs, key=lambda z: abs(np.log10(max(z, 1e-12)) - log_Z)) if nugrid_Zs else None
        Z_kar = min(kar_Z, key=lambda z: abs(np.log10(z) - np.log10(max(Z, 1e-6)))) if kar_Z else None
        feh = np.log10(max(Z, 1e-6) / 0.014)
        feh_near = min(cl_fehs, key=lambda x: abs(x - feh))

        # PARSEC v2 masses at this Z (for remnant classification)
        p2_Ms = sorted(m for (z, m) in p2_ejecta if z == Z_p2) if Z_p2 else []

        for im, M in enumerate(M_grid):

            # ═══ AGB / WD: M < 8 Msun ═══
            if M < 8:
                rt, rm = classify_remnant_initial(M, Z, cl_presn, kar_wd)
                rem_type[iz, im] = rt
                rem_mass[iz, im] = rm
                if Z_kar is not None:
                    yield_src[iz, im] = 1
                    for ie, elem in enumerate(TRACKED_ELEMENTS):
                        pairs = [(km, kar_yields[(Z_kar, km, elem)])
                                 for km in kar_M if (Z_kar, km, elem) in kar_yields]
                        if pairs:
                            m_a = np.array([p[0] for p in pairs])
                            y_a = np.array([p[1] for p in pairs])
                            yields[iz, im, ie] = np.interp(
                                M, m_a, y_a, left=y_a[0], right=y_a[-1])
                n_src[yield_src[iz, im]] += 1
                continue

            # ═══ Low-mass CCSNe: 8 ≤ M < 14 Msun ═══
            if M < 14:
                # Remnant: ECSN for 8-10 Msun, CCSN NS for 10-13+
                if M <= 10:
                    rem_type[iz, im] = 1   # ECSN → NS
                    rem_mass[iz, im] = 1.25
                else:
                    rem_type[iz, im] = 2   # CCSN → NS
                    rem_mass[iz, im] = np.interp(M, lim25_masses, lim25_remnants)

                yield_src[iz, im] = 2
                M_ej = M - rem_mass[iz, im]

                # Base yields from Limongi+2025 (total ejected at solar Z)
                for ie, elem in enumerate(TRACKED_ELEMENTS):
                    y_total = np.interp(M, lim25_masses, lim25_yields[elem])
                    y_net = y_total - X_init_solar.get(elem, 0.0) * M_ej

                    # Z-scaling from NuGrid for non-solar metallicities
                    if Z_ng is not None and abs(log_Z - np.log10(0.014)) > 0.15:
                        scale = nugrid_z_scale.get(Z_ng, {}).get(elem, 1.0)
                        y_net *= scale

                    yields[iz, im, ie] = y_net

                n_src[2] += 1
                continue

            # ═══ Massive stars: M ≥ 14 Msun ═══
            # Step 1: Get REMNANT from PARSEC v2 (type + mass)
            p2_rem_type = None
            p2_rem_mass = 0.0
            if p2_Ms:
                result = _interp_p2_yields(p2_ejecta, Z_p2, M, p2_Ms, p2_init_comp)
                if result is not None:
                    _, p2_rem_mass, p2_rem_type = result

            # Step 2: Get YIELDS from Limongi25/CL18/NuGrid or PARSEC v2 (PISN)
            # Always set remnant from PARSEC v2
            if p2_rem_type is not None:
                rem_type[iz, im] = p2_rem_type
                rem_mass[iz, im] = p2_rem_mass
            else:
                # No PARSEC v2 coverage: fall back to initial mass classification
                rt, rm = classify_remnant_initial(M, Z, cl_presn, kar_wd)
                rem_type[iz, im] = rt
                rem_mass[iz, im] = rm

            # PISN: complete disruption — use PARSEC v2 yields
            if rem_type[iz, im] == 5:
                if result is not None:
                    net_y, _, _ = result
                    for ie, elem in enumerate(TRACKED_ELEMENTS):
                        yields[iz, im, ie] = net_y.get(elem, 0.0)
                yield_src[iz, im] = 4  # PARSEC_v2_PISN
                rem_mass[iz, im] = 0.0
                n_src[4] += 1

            # FSN or DBH: failed SN — no explosion yields
            elif rem_type[iz, im] in (3, 6):
                yield_src[iz, im] = 3
                # yields stay at 0
                n_src[3] += 1

            # CCSN / PPISN: get explosion yields from Limongi25 or CL18
            else:
                # Prefer Limongi+2025 for 14-15 Msun (newer, overlaps)
                if M <= lim25_masses[-1]:
                    yield_src[iz, im] = 2
                    M_ej = M - rem_mass[iz, im]
                    for ie, elem in enumerate(TRACKED_ELEMENTS):
                        y_total = np.interp(M, lim25_masses, lim25_yields[elem])
                        y_net = y_total - X_init_solar.get(elem, 0.0) * M_ej
                        if Z_ng is not None and abs(log_Z - np.log10(0.014)) > 0.15:
                            scale = nugrid_z_scale.get(Z_ng, {}).get(elem, 1.0)
                            y_net *= scale
                        yields[iz, im, ie] = y_net
                    n_src[2] += 1

                elif M <= CL_MASSES[-1]:
                    # CL18 for 15-120 Msun
                    yield_src[iz, im] = 3
                    M_ej = M - rem_mass[iz, im]
                    X_init_z = p2_init_comp.get(Z_p2, {}) if Z_p2 else {}
                    for ie, elem in enumerate(TRACKED_ELEMENTS):
                        cl_y = cl_yields.get((feh_near, elem))
                        if cl_y is not None:
                            y_gross = np.interp(M, CL_MASSES, cl_y)
                            yields[iz, im, ie] = y_gross - X_init_z.get(elem, 0.0) * M_ej
                    n_src[3] += 1

                else:
                    # M > 120 Msun, PPISN with CL18 unavailable: use P2 yields
                    if result is not None:
                        net_y, _, _ = result
                        yield_src[iz, im] = 4
                        for ie, elem in enumerate(TRACKED_ELEMENTS):
                            yields[iz, im, ie] = net_y.get(elem, 0.0)
                        n_src[4] += 1
                    else:
                        # Last resort: CL18 edge extrapolation
                        yield_src[iz, im] = 3
                        M_ej = M - rem_mass[iz, im]
                        X_init_z = p2_init_comp.get(Z_p2, {}) if Z_p2 else {}
                        for ie, elem in enumerate(TRACKED_ELEMENTS):
                            cl_y = cl_yields.get((feh_near, elem))
                            if cl_y is not None:
                                y_gross = cl_y[-1]
                                yields[iz, im, ie] = y_gross - X_init_z.get(elem, 0.0) * M_ej
                        n_src[3] += 1

                # PPISN: scale yields by ejecta fraction
                if rem_type[iz, im] == 4 and rem_mass[iz, im] > 0:
                    f_ej = max(0, (M - rem_mass[iz, im]) / M)
                    yields[iz, im, :] *= f_ej

    print(f"  Yield sources: Karakas={n_src[1]}, Limongi25={n_src[2]}, "
          f"CL18={n_src[3]}, PARSEC_v2_PISN={n_src[4]}, none={n_src[0]}")

    return yields, rem_type, rem_mass, yield_src


def scale_yields_for_fallback(yields, rem_type, rem_mass, M_preSN, M_grid, yield_src=None):
    """
    Safety pass: ensure yields are consistent with remnant type.

    FSN/DBH (types 3, 6): yields should be ~0 (already zeroed in build_yield_grid)
    PISN (type 5): no remnant, yields already from PARSEC v2
    PPISN (type 4): already scaled in build_yield_grid

    This function catches any remaining inconsistencies.
    """
    N_Z, N_M = rem_type.shape
    n_zeroed = 0

    for iz in range(N_Z):
        for im in range(N_M):
            rt = rem_type[iz, im]
            # FSN and DBH: ensure yields are zero (failed SN)
            if rt in (3, 6):
                if np.any(yields[iz, im, :] != 0):
                    yields[iz, im, :] = 0.0
                    n_zeroed += 1

    if n_zeroed > 0:
        print(f"  Zeroed yields for {n_zeroed} FSN/DBH entries (consistency check)")


# ═══════════════════════════════════════════════════════════════
# SECTION 8: Main builder
# ═══════════════════════════════════════════════════════════════

def build_all(base_dir, output_file):
    import h5py

    parsec_dir = os.path.join(base_dir, 'parsec1.2')
    boost_dir  = os.path.join(base_dir, 'boost')
    mist_dir   = os.path.join(base_dir, 'mist')
    p2_dir     = os.path.join(base_dir, 'parsec2_vms')

    # ── Discover all tracks ──
    print("Discovering tracks...", flush=True)
    parsec_index = discover_parsec_tracks(parsec_dir)
    boost_index  = discover_boost_tracks(boost_dir)
    mist_index   = discover_mist_tracks(mist_dir)
    p2_index     = discover_parsec2_vms_tracks(p2_dir)

    parsec_Z = sorted(set(k[0] for k in parsec_index.keys()))
    parsec_M = sorted(set(k[1] for k in parsec_index.keys()))
    mist_fehs = sorted(set(k[0] for k in mist_index.keys()))
    mist_M = sorted(set(k[1] for k in mist_index.keys()))
    p2_Z_strs = sorted(set(k[0] for k in p2_index.keys()))
    p2_M = sorted(set(k[1] for k in p2_index.keys()))

    print(f"  PARSEC v1: {len(parsec_Z)} Z x {len(parsec_M)} M = {len(parsec_index)} tracks")
    print(f"  BoOST:     {len(boost_index)} tracks across {len(set(k[0] for k in boost_index))} grids")
    print(f"  MIST:      {len(mist_index)} tracks across {len(mist_fehs)} [Fe/H] grids")
    print(f"  PARSEC v2: {len(p2_Z_strs)} Z x {len(p2_M)} M = {len(p2_index)} tracks")

    # ── Build master grids ──
    Z_grid = np.array(PARSEC_Z_GRID)
    # Mass grid: PARSEC masses + extend to 500 Msun
    M_grid = sorted(set(parsec_M) | {500.0})
    M_grid = np.array(M_grid)
    log_age_grid = np.linspace(LOG_AGE_MIN, LOG_AGE_MAX, N_AGE)

    N_Z, N_M = len(Z_grid), len(M_grid)
    print(f"\nMaster grid: {N_Z} Z x {N_M} M x {N_AGE} ages")

    # Determine source mapping for each Z
    boost_map = {}
    mist_map = {}
    p2_map = {}
    for iz, Z in enumerate(Z_grid):
        g = find_boost_grid_for_Z(Z)
        boost_map[iz] = g
        mf = find_mist_feh_for_Z(Z)
        mist_map[iz] = mf
        p2z = find_parsec2_Z_str_for_Z(Z)
        p2_map[iz] = p2z
        parts = []
        if mf:
            parts.append(f"MIST([Fe/H]={MIST_FEH_GRID[mf]:+.2f}, Z={MIST_Z_GRID[mf]:.5f})")
        if p2z:
            parts.append(f"PARSEC2(Z={p2z})")
        if g:
            parts.append(f"BoOST({g}, Z={BOOST_GRIDS[g]:.5f})")
        parts.append("PARSEC_v1(fallback)")
        print(f"  Z={Z:.5f} → {' + '.join(parts)}")

    # ── Initialize Planck LUT ──
    get_planck_lut()

    # ── Allocate output arrays ──
    shape = (N_Z, N_M, N_AGE)
    log_Q_ion = np.full(shape, LOG_FLOOR)
    log_L_FUV = np.full(shape, LOG_FLOOR)
    log_L_LW  = np.full(shape, LOG_FLOOR)
    log_L_bol = np.full(shape, LOG_FLOOR)
    logR_cm   = np.full(shape, LOG_FLOOR)
    logL      = np.full(shape, LOG_FLOOR)
    logTeff   = np.full(shape, LOG_FLOOR)
    log_Mdot  = np.full(shape, -30.0)
    M_current = np.zeros(shape)
    v_wind_arr= np.zeros(shape)
    surface   = np.zeros((N_Z, N_M, N_AGE, N_ELEM))
    lifetime  = np.zeros((N_Z, N_M))
    M_preSN   = np.zeros((N_Z, N_M))
    M_CO_core_arr = np.zeros((N_Z, N_M))
    M_He_core_arr = np.zeros((N_Z, N_M))
    # track_src: 0=PARSEC_v1, 1=BoOST, 2=MIST, 3=PARSEC_v2
    track_src = np.zeros((N_Z, N_M), dtype=int)
    phase_arr = np.full(shape, 7, dtype=np.int8)  # default: dead

    # Phase transition times (2D: Z × M), in years
    TRANS_KEYS = ['t_MS_start', 't_MS_end', 't_RGB_start', 't_CHeB_start',
                  't_AGB_start', 't_AGB_end', 't_postAGB_start', 't_WR_start']
    phase_trans = {k: np.zeros((N_Z, N_M)) for k in TRANS_KEYS}

    # ── Fill evolution grid ──
    # Priority: PARSEC v2 (M >= 2) → MIST → BoOST → PARSEC v1
    n_total = N_Z * N_M
    n_done = 0
    src_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # PARSECv1, BoOST, MIST, PARSECv2
    for iz, Z in enumerate(Z_grid):
        boost_grid = boost_map[iz]
        mist_feh = mist_map[iz]
        p2_zstr = p2_map[iz]
        pZ = min(parsec_Z, key=lambda z: abs(np.log10(z) - np.log10(Z)))

        for im, M in enumerate(M_grid):
            n_done += 1
            if n_done % 200 == 0 or n_done == 1:
                print(f"  [{n_done}/{n_total}] Z={Z}, M={M}", flush=True)

            result = None
            src = 0  # default: PARSEC v1

            # 1) PARSEC v2: primary for M >= 2 Msun
            if M >= MASS_STITCH_P2_LO and p2_zstr is not None:
                result = interpolate_parsec2_to_mass(p2_index, p2_zstr, M, log_age_grid)
                if result is not None:
                    src = 3

            # 2) MIST fallback: M < 2 or PARSEC v2 gap
            if result is None and mist_feh is not None:
                result = interpolate_mist_to_mass(mist_index, mist_feh, M, log_age_grid)
                if result is not None:
                    src = 2

            # 3) BoOST fallback for massive stars
            if result is None and boost_grid is not None and M >= 9.0:
                result = interpolate_boost_to_mass(boost_index, boost_grid, M, log_age_grid)
                if result is not None:
                    src = 1

            # 4) PARSEC v1 last resort
            if result is None:
                pM = min(parsec_M, key=lambda m: abs(m - M))
                fp = parsec_index.get((pZ, pM))
                if fp is not None:
                    track = parsec_track_to_common(fp)
                    result = resample_onto_age_grid(track, log_age_grid)
                    src = 0

            if result is None:
                continue

            src_counts[src] += 1
            track_src[iz, im] = src
            lifetime[iz, im] = result['lifetime']
            M_preSN[iz, im] = result.get('M_preSN_scalar', 0.0)
            M_CO_core_arr[iz, im] = result.get('M_CO_core', 0.0)
            M_He_core_arr[iz, im] = result.get('M_He_core', 0.0)

            log_Q_ion[iz, im] = result['log_Q_ion']
            log_L_FUV[iz, im] = result['log_L_FUV']
            log_L_LW[iz, im]  = result['log_L_LW']
            log_L_bol[iz, im] = result['log_L_bol']
            logR_cm[iz, im]   = result['logR_cm']
            logL[iz, im]      = result['logL']
            logTeff[iz, im]   = result['logTeff']
            log_Mdot[iz, im]  = result['log_Mdot']
            M_current[iz, im] = result['M_current']
            v_wind_arr[iz, im]= result['v_wind']
            surface[iz, im]   = result['surface']
            if 'phase' in result:
                phase_arr[iz, im] = result['phase']
            pt = result.get('phase_transitions', {})
            for k in TRANS_KEYS:
                phase_trans[k][iz, im] = pt.get(k, 0.0)

    print(f"\n  Track sources: PARSEC_v2={src_counts[3]}, MIST={src_counts[2]}, "
          f"BoOST={src_counts[1]}, PARSEC_v1={src_counts[0]}")

    # ── Fill gaps: copy nearest valid track for any missing (Z, M) entries ──
    # Gaps arise when no track source has data (e.g. 0.09 Msun at certain Z).
    # Use lifetime == 0 as gap indicator; search in log(Z)-log(M) space.
    log_Z_arr = np.log10(Z_grid)
    log_M_arr = np.log10(M_grid)
    n_filled = 0
    for iz in range(N_Z):
        for im in range(N_M):
            if lifetime[iz, im] > 0:
                continue
            # Find nearest valid entry in log(Z)-log(M) space
            best_dist = np.inf
            best_jz, best_jm = -1, -1
            for jz in range(N_Z):
                for jm in range(N_M):
                    if lifetime[jz, jm] <= 0:
                        continue
                    # Weight Z and M equally in log space
                    d = (log_Z_arr[iz] - log_Z_arr[jz])**2 + (log_M_arr[im] - log_M_arr[jm])**2
                    if d < best_dist:
                        best_dist = d
                        best_jz, best_jm = jz, jm
            if best_jz < 0:
                continue  # no valid tracks at all (shouldn't happen)
            # Copy all arrays from the donor entry
            log_Q_ion[iz, im] = log_Q_ion[best_jz, best_jm]
            log_L_FUV[iz, im] = log_L_FUV[best_jz, best_jm]
            log_L_LW[iz, im]  = log_L_LW[best_jz, best_jm]
            log_L_bol[iz, im] = log_L_bol[best_jz, best_jm]
            logR_cm[iz, im]   = logR_cm[best_jz, best_jm]
            logL[iz, im]      = logL[best_jz, best_jm]
            logTeff[iz, im]   = logTeff[best_jz, best_jm]
            log_Mdot[iz, im]  = log_Mdot[best_jz, best_jm]
            M_current[iz, im] = M_current[best_jz, best_jm]
            v_wind_arr[iz, im]= v_wind_arr[best_jz, best_jm]
            surface[iz, im]   = surface[best_jz, best_jm]
            phase_arr[iz, im] = phase_arr[best_jz, best_jm]
            lifetime[iz, im]  = lifetime[best_jz, best_jm]
            M_preSN[iz, im]   = M_preSN[best_jz, best_jm]
            M_CO_core_arr[iz, im] = M_CO_core_arr[best_jz, best_jm]
            M_He_core_arr[iz, im] = M_He_core_arr[best_jz, best_jm]
            track_src[iz, im] = track_src[best_jz, best_jm]
            for k in TRANS_KEYS:
                phase_trans[k][iz, im] = phase_trans[k][best_jz, best_jm]
            n_filled += 1
            print(f"  Gap filled: Z={Z_grid[iz]}, M={M_grid[im]:.2f} "
                  f"← Z={Z_grid[best_jz]}, M={M_grid[best_jm]:.2f} "
                  f"(d_log={np.sqrt(best_dist):.3f})")
    if n_filled > 0:
        print(f"  Filled {n_filled} gap(s) with nearest-neighbor tracks")

    # M_preSN is filled per-track in the loop above via result['M_preSN_scalar']

    # ── Compute total FUV (6-13.6 eV) = L_FUV + L_LW in log space ──
    log_L_FUV_total = np.full(shape, LOG_FLOOR)
    alive = (log_L_FUV > LOG_FLOOR + 1) | (log_L_LW > LOG_FLOOR + 1)
    L_fuv_lin = np.where(log_L_FUV > LOG_FLOOR + 1, 10**log_L_FUV, 0.0)
    L_lw_lin  = np.where(log_L_LW > LOG_FLOOR + 1, 10**log_L_LW, 0.0)
    total = L_fuv_lin + L_lw_lin
    log_L_FUV_total[alive & (total > 0)] = np.log10(total[alive & (total > 0)])

    # ── Build yield grid ──
    print("\nBuilding yield tables...", flush=True)
    yields, rem_type, rem_mass, yield_src = build_yield_grid(Z_grid, M_grid, base_dir)

    # ── Reclassify remnants using He/CO core masses from tracks ──
    print("  Reclassifying remnants with core masses...", flush=True)
    reclassify_with_cores(rem_type, rem_mass, M_He_core_arr, M_CO_core_arr, M_preSN, M_grid,
                          yield_src=yield_src)

    # ── Scale yields for BH/PPISN fallback ──
    scale_yields_for_fallback(yields, rem_type, rem_mass, M_preSN, M_grid, yield_src=yield_src)

    # ── Write HDF5 ──
    print(f"\nWriting {output_file}...", flush=True)
    with h5py.File(output_file, 'w') as f:
        # Grid axes
        f.create_dataset('Z', data=Z_grid)
        f.create_dataset('log_Z', data=np.log10(Z_grid))
        f.create_dataset('M_init', data=M_grid)
        f.create_dataset('log_M_init', data=np.log10(M_grid))
        f.create_dataset('log_age_yr', data=log_age_grid)
        f.create_dataset('elements', data=np.array(TRACKED_ELEMENTS, dtype='S4'))

        # Evolution tables (time-dependent)
        comp = dict(compression='gzip', compression_opts=4)
        for name, data, units in [
            ('log_Q_ion',   log_Q_ion,  'log10(photons/s), E > 13.6 eV'),
            ('log_L_FUV',   log_L_FUV,  'log10(erg/s), 6.0-11.2 eV, photoelectric'),
            ('log_L_LW',    log_L_LW,   'log10(erg/s), 11.2-13.6 eV, Lyman-Werner'),
            ('log_L_FUV_total', log_L_FUV_total, 'log10(erg/s), 6.0-13.6 eV, total FUV'),
            ('log_L_bol',   log_L_bol,  'log10(erg/s), bolometric, for radiation pressure'),
            ('logL',        logL,       'log10(L/L_sun)'),
            ('logTeff',     logTeff,    'log10(T_eff / K)'),
            ('logR_cm',     logR_cm,    'log10(R / cm)'),
            ('M_current',   M_current,  'Msun'),
            ('log_Mdot',    log_Mdot,   'log10(dM/dt in Msun/yr), derived from M_current for consistency'),
            ('v_wind',      v_wind_arr, 'km/s, terminal wind velocity'),
        ]:
            ds = f.create_dataset(name, data=data, **comp)
            ds.attrs['units'] = units
            ds.attrs['shape'] = '(N_Z, N_M, N_age)'

        ds = f.create_dataset('surface_abundances', data=surface, **comp)
        ds.attrs['units'] = 'mass fractions, (N_Z, N_M, N_age, N_elements)'
        ds.attrs['elements'] = TRACKED_ELEMENTS

        ds = f.create_dataset('phase', data=phase_arr, **comp)
        ds.attrs['encoding'] = PHASE_NAMES
        ds.attrs['shape'] = '(N_Z, N_M, N_age)'
        ds.attrs['usage'] = ('Use to distinguish MS winds (fast, hot) from AGB winds '
                             '(slow, dense, dust-rich). AGB phase=5 is where '
                             'envelope ejection and PN formation occur.')

        # Phase transition times (2D: Z × M)
        grp = f.create_group('phase_transitions')
        grp.attrs['description'] = ('Transition times in yr for each evolutionary phase. '
                                    '2D arrays (N_Z, N_M). Zero means phase not reached.')
        grp.attrs['usage'] = ('Compare star_age against these to determine current phase '
                               'without interpolating the 3D phase array.')
        for k in TRANS_KEYS:
            ds = grp.create_dataset(k, data=phase_trans[k])
            ds.attrs['units'] = 'yr'

        f.create_dataset('lifetime_yr', data=lifetime)
        f['lifetime_yr'].attrs['units'] = 'yr'

        f.create_dataset('track_source', data=track_src)
        f['track_source'].attrs['encoding'] = '0=PARSEC_v1, 1=BoOST, 2=MIST, 3=PARSEC_v2'

        f.create_dataset('M_preSN', data=M_preSN)
        f['M_preSN'].attrs['units'] = 'Msun, pre-SN stellar mass (last alive M_current)'
        f['M_preSN'].attrs['usage'] = 'ejecta_mass = M_preSN - remnant_mass'

        f.create_dataset('M_CO_core', data=M_CO_core_arr)
        f['M_CO_core'].attrs['units'] = 'Msun, CO core mass at end of life'
        f['M_CO_core'].attrs['source'] = 'BoOST coreCO / PARSEC M_CORE_C'
        f['M_CO_core'].attrs['usage'] = 'BH remnant mass via fallback prescription'

        f.create_dataset('M_He_core', data=M_He_core_arr)
        f['M_He_core'].attrs['units'] = 'Msun, He core mass at end of life'

        # Yield tables (end-of-life)
        ds = f.create_dataset('net_yields', data=yields, **comp)
        ds.attrs['units'] = 'Msun (net yield per element)'
        ds.attrs['shape'] = '(N_Z, N_M, N_elements)'

        f.create_dataset('remnant_type', data=rem_type)
        f['remnant_type'].attrs['encoding'] = ('0=WD, 1=ECSN(NS), 2=CCSN(NS), 3=FSN(BH), '
                                                '4=PPISN(BH), 5=PISN, 6=DBH(BH)')
        f.create_dataset('remnant_mass', data=rem_mass)
        f['remnant_mass'].attrs['units'] = 'Msun'
        f['remnant_mass'].attrs['note'] = ('For PARSEC v2: native remnant from full models. '
                                           'Others: BH from CO core, PPISN capped 45 Msun.')
        f.create_dataset('yield_source', data=yield_src)
        f['yield_source'].attrs['encoding'] = ('0=none, 1=Karakas_AGB(KL16+K10), '
                                                 '2=Limongi25+NuGrid, '
                                                 '3=CL18, 4=PARSEC_v2_PISN')
        f['yield_source'].attrs['note'] = ('Remnants always from PARSEC v2. '
                                           'Explosion yields from CL18/Limongi25. '
                                           'FSN/DBH: yields=0 (failed SN). PISN: PARSEC v2 yields.')

        # Type Ia yields (mass-independent, single yield vector)
        ia_yields = get_type_ia_yields()
        ia_arr = np.array([ia_yields[e] for e in TRACKED_ELEMENTS])
        ds = f.create_dataset('type_ia_yields', data=ia_arr)
        ds.attrs['units'] = 'Msun per event'
        ds.attrs['elements'] = TRACKED_ELEMENTS
        ds.attrs['source'] = 'Iwamoto+ 1999, W7 model'
        ds.attrs['note'] = 'Complete disruption of Chandrasekhar-mass C/O WD'

        # Type Ia DTD parameters
        grp = f.create_group('type_ia_dtd')
        for k, v in TYPE_IA_DTD.items():
            grp.attrs[k] = v
        grp.attrs['description'] = ('Delay time distribution: P(t) ~ t^slope for t > t_min. '
                                    'Rate normalized to rate_per_Msun Ia per Msun formed '
                                    '(Maoz & Mannucci 2012).')

        # Magnetar parameters (parametric prescription for C code)
        grp = f.create_group('magnetar')
        for k, v in MAGNETAR_PARAMS.items():
            grp.attrs[k] = v

        # Global attributes
        f.attrs['description'] = 'Unified stellar evolution + yield tables for GIZMO'
        f.attrs['evolution_sources'] = ('PARSEC v2 VMS (M >= 2, primary) + MIST v1.2 (M < 2 + fallback) '
                                          '+ BoOST v1.3 (fallback) + PARSEC v1.2s (last resort)')
        f.attrs['yield_sources'] = ('Karakas AGB [K&L(2016) Z>=0.007 + K(2010) Z<0.007] '
                                    '+ Limongi+(2025) 8-13Msun '
                                    '+ CL18 14-120Msun + NuGrid Z-scaling '
                                    '+ PARSEC v2 PISN yields + Iwamoto(1999) Type Ia')
        f.attrs['remnant_source'] = 'PARSEC v2 VMS (CCSN/FSN/PPISN/PISN/DBH) for M>=14'
        f.attrs['tracked_elements'] = TRACKED_ELEMENTS
        f.attrs['N_Z'] = N_Z
        f.attrs['N_M'] = N_M
        f.attrs['N_age'] = N_AGE
        f.attrs['N_elements'] = N_ELEM
        f.attrs['mass_stitch_p2_lo'] = MASS_STITCH_P2_LO
        f.attrs['mass_stitch_boost_lo'] = MASS_STITCH_BOOST_LO
        f.attrs['mass_stitch_boost_hi'] = MASS_STITCH_BOOST_HI
        f.attrs['band_edges_eV'] = [6.0, 11.2, 13.6]
        f.attrs['v_wind_prescription'] = 'Lamers+1995: 2.6*v_esc (T>21kK), 1.3*v_esc (10-21kK), 0.7*v_esc (cool)'
        f.attrs['E_ECSN_erg'] = 1e50       # weaker: degenerate O/Ne/Mg core
        f.attrs['E_CCSN_erg'] = 1e51       # canonical core-collapse
        f.attrs['E_FSN_erg'] = 0.0         # failed SN: no explosion, direct BH
        f.attrs['E_PPISN_erg'] = 1e51      # pulsational pair-instability
        f.attrs['E_PISN_erg'] = 1e52       # thermonuclear disruption of entire star
        f.attrs['E_DBH_erg'] = 0.0         # direct BH: no explosion
        f.attrs['E_magnetar_erg'] = 1e52   # CCSN + spindown combined
        f.attrs['E_typeIa_erg'] = 1e51     # Chandrasekhar-mass WD detonation
        f.attrs['energy_note'] = 'FSN and DBH have zero explosion energy. User may override in C code.'
        f.attrs['remnant_types'] = '0=WD, 1=ECSN(NS), 2=CCSN(NS), 3=FSN(BH), 4=PPISN(BH), 5=PISN, 6=DBH(BH)'

    sz = os.path.getsize(output_file) / 1e6
    print(f"Done. File: {sz:.1f} MB")
    return output_file


# ═══════════════════════════════════════════════════════════════
# SECTION 9: Verification
# ═══════════════════════════════════════════════════════════════

def verify(output_file):
    import h5py
    with h5py.File(output_file, 'r') as f:
        Z = f['Z'][:]
        M = f['M_init'][:]
        log_age = f['log_age_yr'][:]
        tsrc = f['track_source'][:]
        lt = f['lifetime_yr'][:]

        iz_sol = np.argmin(np.abs(Z - 0.014))
        i_zams = np.argmin(np.abs(log_age - 5.0))  # ~100 kyr

        print(f"\n{'='*80}")
        print(f"Verification at Z={Z[iz_sol]} (solar-ish)")
        print(f"{'='*80}")

        # Track source summary
        n_parsec = np.sum(tsrc == 0)
        n_boost = np.sum(tsrc == 1)
        n_mist = np.sum(tsrc == 2)
        n_p2 = np.sum(tsrc == 3)
        print(f"Track sources: {n_p2} PARSEC_v2, {n_mist} MIST, {n_boost} BoOST, {n_parsec} PARSEC_v1")

        # Sample stars
        print(f"\n{'M':>6s} {'src':>6s} {'life[Myr]':>10s} {'logQ_ion':>9s} "
              f"{'logLFUV':>8s} {'logLbol':>8s} {'v_wind':>7s} {'logMdot':>8s}")
        print('-' * 75)
        src_name = {0: 'PARv1', 1: 'BoOST', 2: 'MIST', 3: 'PARv2'}
        for Mcheck in [1, 5, 10, 20, 25, 40, 60, 80, 120, 250, 350, 500]:
            im = np.argmin(np.abs(M - Mcheck))
            s = src_name[tsrc[iz_sol, im]]
            life = lt[iz_sol, im] / 1e6
            lQ = f['log_Q_ion'][iz_sol, im, i_zams]
            lF = f['log_L_FUV'][iz_sol, im, i_zams]
            lB = f['log_L_bol'][iz_sol, im, i_zams]
            vw = f['v_wind'][iz_sol, im, i_zams]
            lM = f['log_Mdot'][iz_sol, im, i_zams]
            print(f"{M[im]:6.1f} {s:>6s} {life:10.3f} {lQ:9.2f} "
                  f"{lF:8.2f} {lB:8.2f} {vw:7.1f} {lM:8.2f}")

        # Yield spot checks
        yields = f['net_yields'][:]
        rtype = f['remnant_type'][:]
        rmass = f['remnant_mass'][:]
        elems = [e.decode() for e in f['elements'][:]]

        print(f"\n{'M':>6s} {'rem':>6s} {'Mrem':>6s} {'M_CO':>6s} {'M_He':>6s} "
              f"{'C':>8s} {'O':>8s} {'Fe':>8s}")
        print('-' * 70)
        type_names = {0: 'WD', 1: 'ECSN', 2: 'CCSN', 3: 'FSN', 4: 'PPISN', 5: 'PISN', 6: 'DBH'}
        mco = f['M_CO_core'][:]
        mhe = f['M_He_core'][:]
        for Mcheck in [1, 5, 10, 13, 25, 40, 80, 120, 150, 250]:
            im = np.argmin(np.abs(M - Mcheck))
            t = type_names.get(rtype[iz_sol, im], '?')
            r = rmass[iz_sol, im]
            co = mco[iz_sol, im]
            he = mhe[iz_sol, im]
            yC = yields[iz_sol, im, elems.index('C')]
            yO = yields[iz_sol, im, elems.index('O')]
            yFe = yields[iz_sol, im, elems.index('Fe')]
            print(f"{M[im]:6.1f} {t:>6s} {r:6.2f} {co:6.2f} {he:6.2f} "
                  f"{yC:8.3f} {yO:8.3f} {yFe:8.3f}")

        # Remnant type census
        print(f"\nRemnant census (all Z):")
        for code, name in type_names.items():
            n = np.sum(rtype == code)
            print(f"  {name:>6s}: {n}")

        # Type Ia check
        if 'type_ia_yields' in f:
            ia = f['type_ia_yields'][:]
            print(f"\nType Ia yields (Msun/event): "
                  f"Fe={ia[elems.index('Fe')]:.3f}, Si={ia[elems.index('Si')]:.3f}, "
                  f"O={ia[elems.index('O')]:.3f}")
            print(f"Type Ia DTD: rate={f['type_ia_dtd'].attrs['rate_per_Msun']:.1e}/Msun, "
                  f"t_min={f['type_ia_dtd'].attrs['t_min_Myr']:.0f} Myr")

        # Z-dependence of BH mass
        print(f"\nBH mass vs Z for M~120 Msun:")
        im120 = np.argmin(np.abs(M - 120))
        for iz in range(len(Z)):
            if rtype[iz, im120] in [3, 4]:
                print(f"  Z={Z[iz]:.5f}: M_BH={rmass[iz, im120]:.1f}, "
                      f"M_CO={mco[iz, im120]:.1f}, M_He={mhe[iz, im120]:.1f}")


if __name__ == '__main__':
    base_dir = '/raven/u/uli/phil/stellar_tables'
    output_file = os.path.join(base_dir, 'stellar_tables_unified.hdf5')
    build_all(base_dir, output_file)
    verify(output_file)
