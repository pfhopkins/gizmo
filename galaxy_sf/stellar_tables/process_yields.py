#!/usr/bin/env python3
"""
Process SN and AGB yield tables into a unified HDF5 lookup table for GIZMO.

Sources:
  1. Chieffi & Limongi (2018) — CCSNe yields, 13-120 Msun, [Fe/H]=0,-1,-2,-3
  2. Heger & Woosley (2002)  — PISN yields, >120 Msun (zero metallicity approx)
  3. Karakas (2010)          — AGB yields, 1-6 Msun, Z=0.02,0.008,0.004,0.0001

Output for each (M_init, Z):
  - remnant_type: 0=WD, 1=NS, 2=BH, 3=PISN (no remnant)
  - remnant_mass: Msun
  - Yield of each tracked element (net yield in Msun)

Currently tracked elements: H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe
(easily extensible by adding to TRACKED_ELEMENTS)
"""

import numpy as np
import os
import sys
from glob import glob

# ═══════════════════════════════════════════════════════════════════════
# Configuration — extend here to add new elements
# ═══════════════════════════════════════════════════════════════════════

TRACKED_ELEMENTS = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']

# Mapping from isotope names to element totals
# C&L uses isotope names like "H", "H2", "He4", "C12", "O16", etc.
# Karakas uses "p", "he3", "he4", "c12", "n14", "o16", etc.
ISOTOPE_TO_ELEMENT_CL = {
    'H': 'H', 'H2': 'H', 'H3': 'H',
    'He3': 'He', 'He4': 'He',
    'C12': 'C', 'C13': 'C', 'C14': 'C',
    'N13': 'N', 'N14': 'N', 'N15': 'N',
    'O16': 'O', 'O17': 'O', 'O18': 'O',
    'Ne20': 'Ne', 'Ne21': 'Ne', 'Ne22': 'Ne',
    'Mg24': 'Mg', 'Mg25': 'Mg', 'Mg26': 'Mg',
    'Si28': 'Si', 'Si29': 'Si', 'Si30': 'Si',
    'S32': 'S', 'S33': 'S', 'S34': 'S', 'S36': 'S',
    'Ca40': 'Ca', 'Ca42': 'Ca', 'Ca43': 'Ca', 'Ca44': 'Ca',
    'Ca46': 'Ca', 'Ca48': 'Ca',
    'Fe54': 'Fe', 'Fe56': 'Fe', 'Fe57': 'Fe', 'Fe58': 'Fe',
    # Radioactive decays that end up as Fe:
    'Ni56': 'Fe',  # 56Ni → 56Co → 56Fe
}

ISOTOPE_TO_ELEMENT_KAR = {
    'p': 'H', 'd': 'H',
    'he3': 'He', 'he4': 'He',
    'c12': 'C', 'c13': 'C',
    'n13': 'N', 'n14': 'N', 'n15': 'N',
    'o16': 'O', 'o17': 'O', 'o18': 'O',
    'ne20': 'Ne', 'ne21': 'Ne', 'ne22': 'Ne',
    'mg24': 'Mg', 'mg25': 'Mg', 'mg26': 'Mg',
    'si28': 'Si', 'si29': 'Si', 'si30': 'Si',
    's32': 'S', 's33': 'S', 's34': 'S',
    'ca40': 'Ca', 'ca42': 'Ca', 'ca43': 'Ca', 'ca44': 'Ca',
    'fe54': 'Fe', 'fe56': 'Fe', 'fe57': 'Fe', 'fe58': 'Fe',
    'ni56': 'Fe',
}


# ═══════════════════════════════════════════════════════════════════════
# Chieffi & Limongi 2018 — Core-collapse SN yields
# ═══════════════════════════════════════════════════════════════════════

def parse_CL18_yields(table8_path):
    """
    Parse C&L 2018 table8.dat (isotopic yields, recommended set).
    Returns dict keyed by (vel, feh, isotope) → array of yields for
    masses [13, 15, 20, 25, 30, 40, 60, 80, 120] Msun.
    """
    CL_MASSES = np.array([13, 15, 20, 25, 30, 40, 60, 80, 120], dtype=float)
    yields = {}
    with open(table8_path) as f:
        for line in f:
            vel = int(line[0:3])
            feh = int(line[4:6])
            iso = line[7:12].strip()
            vals = [float(line[13+i*11:13+(i+1)*11]) for i in range(9)]
            yields[(vel, feh, iso)] = np.array(vals)
    return yields, CL_MASSES


def parse_CL18_presn(table7_path):
    """
    Parse C&L 2018 table7.dat (presupernova properties).
    Returns list of dicts with vel, feh, mass, fe_core, ebind, xi25, sntype.
    """
    models = []
    with open(table7_path) as f:
        for line in f:
            models.append({
                'vel': int(line[0:3]),
                'feh': int(line[4:6]),
                'mass': int(line[7:10]),
                'fe_core': float(line[23:27]),
                'ebind': float(line[28:33]),
                'xi25': float(line[34:38]),
                'sntype': line[71:76].strip(),
            })
    return models


def aggregate_CL18_element_yields(yields, vel=0):
    """
    Sum isotopic yields into element yields for tracked elements.
    Returns dict: (feh, element) → array[9 masses] of yields in Msun.
    Uses non-rotating models (vel=0) by default.
    """
    fehs = sorted(set(k[1] for k in yields.keys()))
    elem_yields = {}
    for feh in fehs:
        for elem in TRACKED_ELEMENTS:
            elem_yields[(feh, elem)] = np.zeros(9)

    for (v, feh, iso), vals in yields.items():
        if v != vel:
            continue
        elem = ISOTOPE_TO_ELEMENT_CL.get(iso)
        if elem and elem in TRACKED_ELEMENTS:
            elem_yields[(feh, elem)] += vals

    return elem_yields, fehs


# ═══════════════════════════════════════════════════════════════════════
# Karakas 2010 — AGB yields
# ═══════════════════════════════════════════════════════════════════════

def parse_karakas_yields(filepath):
    """
    Parse a Karakas tablea[2-5].dat file.
    Returns list of dicts with M0, Z0, M_final, isotope, A, net_yield, mass_lost.
    """
    entries = []
    with open(filepath) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            entries.append({
                'M0': float(line[0:4]),
                'Z0': float(line[5:11]),
                'M_final': float(line[12:17]),
                'iso': line[21:25].strip().lower(),
                'A': int(line[30:32]),
                'net_yield': float(line[34:48]),
                'mass_lost': float(line[49:63]),
            })
    return entries


def aggregate_karakas_element_yields(base_dir):
    """
    Aggregate Karakas AGB yields by element for each (M0, Z0).
    Returns dict: (Z, M, element) → net yield in Msun,
    plus dict: (Z, M) → WD mass.
    """
    elem_yields = {}
    wd_masses = {}

    for table_file in ['tablea2.dat', 'tablea3.dat', 'tablea4.dat', 'tablea5.dat']:
        path = os.path.join(base_dir, table_file)
        if not os.path.exists(path):
            continue
        entries = parse_karakas_yields(path)
        for e in entries:
            Z = e['Z0']
            M = e['M0']
            wd_masses[(Z, M)] = e['M_final']
            elem = ISOTOPE_TO_ELEMENT_KAR.get(e['iso'])
            if elem and elem in TRACKED_ELEMENTS:
                key = (Z, M, elem)
                elem_yields[key] = elem_yields.get(key, 0.0) + e['net_yield']

    return elem_yields, wd_masses


# ═══════════════════════════════════════════════════════════════════════
# PISN yields — Heger & Woosley 2002 (simplified)
# ═══════════════════════════════════════════════════════════════════════

def get_pisn_yields():
    """
    Approximate PISN yields from Heger & Woosley 2002 Table 3.
    Complete disruption, no remnant. Yields in Msun for He core masses.
    We map to ZAMS masses assuming M_He ≈ 0.45 * M_ZAMS (rough).

    Returns dict: (M_ZAMS, element) → yield in Msun.
    For simplicity, we provide yields at a few representative masses
    and interpolate. Zero metallicity approximation.
    """
    # From Heger & Woosley 2002 Table 3 (selected He core masses)
    # He core: 65, 80, 100, 130 Msun → ZAMS ~ 150, 180, 220, 300
    # Yields in Msun (approximate from their tables)
    pisn_data = {
        # M_ZAMS: {element: yield_Msun}
        150: {'H': 30.0, 'He': 55.0, 'C': 0.6, 'N': 0.01, 'O': 15.0,
              'Ne': 0.3, 'Mg': 1.5, 'Si': 8.0, 'S': 4.0, 'Ca': 0.5, 'Fe': 5.0},
        200: {'H': 20.0, 'He': 60.0, 'C': 0.3, 'N': 0.01, 'O': 25.0,
              'Ne': 0.5, 'Mg': 3.0, 'Si': 15.0, 'S': 8.0, 'Ca': 1.0, 'Fe': 15.0},
        260: {'H': 10.0, 'He': 50.0, 'C': 0.1, 'N': 0.01, 'O': 40.0,
              'Ne': 1.0, 'Mg': 5.0, 'Si': 25.0, 'S': 15.0, 'Ca': 2.0, 'Fe': 40.0},
    }
    return pisn_data


# ═══════════════════════════════════════════════════════════════════════
# Remnant classification
# ═══════════════════════════════════════════════════════════════════════

def classify_remnant(M_init, Z, presn_models=None, wd_masses=None):
    """
    Determine remnant type and mass for a star of given initial mass and Z.

    Returns (remnant_type, remnant_mass):
      0 = WD,  1 = NS,  2 = BH,  3 = PISN (no remnant)

    Prescriptions:
      - M < 8 Msun: WD. Mass from Karakas if available, else IFMR.
      - 8-120 Msun: NS or BH based on compactness (xi2.5 from C&L).
        xi2.5 > 0.35 → BH (Ertl+ 2016 / O'Connor & Ott 2011 inspired).
        Remnant mass: NS → 1.4 Msun, BH → Fe core mass (from C&L table7).
      - >120 Msun: PISN (complete disruption, remnant_mass = 0).
    """
    # PISN
    if M_init > 120:
        return 3, 0.0

    # WD
    if M_init < 8:
        # Try Karakas WD mass
        if wd_masses:
            # Find closest (Z, M)
            best_key = None
            best_dist = 1e10
            for (z, m), mwd in wd_masses.items():
                dist = abs(np.log10(z/max(Z, 1e-6))) + abs(m - M_init)
                if dist < best_dist:
                    best_dist = dist
                    best_key = (z, m)
            if best_key and best_dist < 3.0:
                return 0, wd_masses[best_key]
        # Fallback IFMR (Cummings+ 2018 linear fit)
        m_wd = 0.08 * M_init + 0.489
        return 0, min(m_wd, 1.4)

    # Core-collapse: NS or BH
    feh = np.log10(max(Z, 1e-6) / 0.014)  # approximate [Fe/H]
    feh_grid = [0, -1, -2, -3]
    feh_nearest = min(feh_grid, key=lambda x: abs(x - feh))

    # Find matching presn model
    if presn_models:
        best = None
        for m in presn_models:
            if m['vel'] == 0 and m['feh'] == feh_nearest:
                if best is None or abs(m['mass'] - M_init) < abs(best['mass'] - M_init):
                    best = m
        if best:
            # Use compactness to decide NS vs BH
            if best['xi25'] > 0.35:
                # BH — remnant mass ~ Fe core (lower bound)
                return 2, best['fe_core']
            else:
                return 1, 1.4  # NS

    # Fallback: simple mass-based
    if M_init > 25:
        return 2, max(1.5, 0.1 * M_init)  # BH
    return 1, 1.4  # NS


# ═══════════════════════════════════════════════════════════════════════
# Build unified table
# ═══════════════════════════════════════════════════════════════════════

def build_yield_tables(base_dir, output_file):
    import h5py

    cl_dir = os.path.join(base_dir, 'chieffi_limongi2018')
    kar_dir = os.path.join(base_dir, 'karakas2010')

    # ── Parse all data sources ──
    print("Loading Chieffi & Limongi 2018...", flush=True)
    cl_yields_raw, CL_MASSES = parse_CL18_yields(os.path.join(cl_dir, 'table8.dat'))
    cl_presn = parse_CL18_presn(os.path.join(cl_dir, 'table7.dat'))
    cl_elem_yields, cl_fehs = aggregate_CL18_element_yields(cl_yields_raw, vel=0)
    print(f"  Masses: {CL_MASSES}")
    print(f"  [Fe/H]: {cl_fehs}")

    # Map C&L [Fe/H] to Z
    CL_FEH_TO_Z = {0: 0.014, -1: 0.0014, -2: 0.00014, -3: 0.000014}

    print("Loading Karakas 2010...", flush=True)
    kar_elem_yields, kar_wd_masses = aggregate_karakas_element_yields(kar_dir)
    kar_Z = sorted(set(k[0] for k in kar_elem_yields.keys()))
    kar_M = sorted(set(k[1] for k in kar_elem_yields.keys()))
    print(f"  Z: {kar_Z}")
    print(f"  Masses: {kar_M}")

    print("Loading PISN yields (Heger & Woosley 2002)...", flush=True)
    pisn_data = get_pisn_yields()
    pisn_masses = sorted(pisn_data.keys())
    print(f"  Masses: {pisn_masses}")

    # ── Build unified mass and Z grids ──
    # Combine all masses from all sources
    all_masses = sorted(set(
        list(kar_M) + list(CL_MASSES) + list(pisn_masses)
    ))
    # Use PARSEC Z grid for consistency
    all_Z = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008,
             0.01, 0.014, 0.017, 0.02, 0.03, 0.04, 0.06]

    N_Z = len(all_Z)
    N_M = len(all_masses)
    N_elem = len(TRACKED_ELEMENTS)

    print(f"\nUnified grid: {N_Z} metallicities x {N_M} masses x {N_elem} elements")

    # Output arrays
    yield_table = np.zeros((N_Z, N_M, N_elem))   # net yields in Msun
    remnant_type = np.zeros((N_Z, N_M), dtype=int)  # 0=WD, 1=NS, 2=BH, 3=PISN
    remnant_mass = np.zeros((N_Z, N_M))
    source_flag = np.zeros((N_Z, N_M), dtype=int)   # 0=none, 1=Karakas, 2=C&L, 3=PISN

    for iz, Z in enumerate(all_Z):
        feh = np.log10(max(Z, 1e-6) / 0.014)
        feh_nearest = min(cl_fehs, key=lambda x: abs(x - feh))

        for im, M in enumerate(all_masses):
            # Classify remnant
            rtype, rmass = classify_remnant(M, Z, cl_presn, kar_wd_masses)
            remnant_type[iz, im] = rtype
            remnant_mass[iz, im] = rmass

            if M > 120:
                # ── PISN yields ──
                source_flag[iz, im] = 3
                # Interpolate between PISN mass points
                for ie, elem in enumerate(TRACKED_ELEMENTS):
                    if M <= pisn_masses[0]:
                        yield_table[iz, im, ie] = pisn_data[pisn_masses[0]].get(elem, 0)
                    elif M >= pisn_masses[-1]:
                        yield_table[iz, im, ie] = pisn_data[pisn_masses[-1]].get(elem, 0)
                    else:
                        # Linear interpolation
                        for k in range(len(pisn_masses)-1):
                            if pisn_masses[k] <= M <= pisn_masses[k+1]:
                                f = (M - pisn_masses[k]) / (pisn_masses[k+1] - pisn_masses[k])
                                y0 = pisn_data[pisn_masses[k]].get(elem, 0)
                                y1 = pisn_data[pisn_masses[k+1]].get(elem, 0)
                                yield_table[iz, im, ie] = y0 + f * (y1 - y0)
                                break

            elif M >= 13:
                # ── C&L CCSNe yields ──
                source_flag[iz, im] = 2
                for ie, elem in enumerate(TRACKED_ELEMENTS):
                    cl_y = cl_elem_yields.get((feh_nearest, elem))
                    if cl_y is not None:
                        yield_table[iz, im, ie] = np.interp(M, CL_MASSES, cl_y)

            elif M >= 1:
                # ── Karakas AGB yields ──
                source_flag[iz, im] = 1
                # Find nearest Z in Karakas
                kar_Z_nearest = min(kar_Z, key=lambda z: abs(np.log10(z) - np.log10(max(Z, 1e-6))))
                for ie, elem in enumerate(TRACKED_ELEMENTS):
                    # Get yields at this Z for available masses
                    m_y_pairs = []
                    for km in kar_M:
                        key = (kar_Z_nearest, km, elem)
                        if key in kar_elem_yields:
                            m_y_pairs.append((km, kar_elem_yields[key]))
                    if m_y_pairs:
                        m_arr = np.array([p[0] for p in m_y_pairs])
                        y_arr = np.array([p[1] for p in m_y_pairs])
                        yield_table[iz, im, ie] = np.interp(M, m_arr, y_arr,
                                                             left=y_arr[0], right=y_arr[-1])

            # Stars 8-13 Msun: gap between Karakas (up to 6.5) and C&L (13+)
            # Interpolate between the two
            elif 6.5 < M < 13:
                source_flag[iz, im] = 2  # use C&L lower bound
                for ie, elem in enumerate(TRACKED_ELEMENTS):
                    cl_y = cl_elem_yields.get((feh_nearest, elem))
                    if cl_y is not None:
                        yield_table[iz, im, ie] = np.interp(M, CL_MASSES, cl_y,
                                                             left=cl_y[0])

    # ── Write HDF5 ──
    print(f"\nWriting {output_file}...", flush=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('Z', data=np.array(all_Z))
        f.create_dataset('log_Z', data=np.log10(np.array(all_Z)))
        f.create_dataset('M_init', data=np.array(all_masses))
        f.create_dataset('elements', data=np.array(TRACKED_ELEMENTS, dtype='S4'))

        f.create_dataset('net_yields', data=yield_table,
                         compression='gzip', compression_opts=4)
        f['net_yields'].attrs['units'] = 'Msun (net yield per element)'
        f['net_yields'].attrs['shape'] = '(N_Z, N_M, N_elements)'

        f.create_dataset('remnant_type', data=remnant_type)
        f['remnant_type'].attrs['encoding'] = '0=WD, 1=NS, 2=BH, 3=PISN'

        f.create_dataset('remnant_mass', data=remnant_mass)
        f['remnant_mass'].attrs['units'] = 'Msun'

        f.create_dataset('source_flag', data=source_flag)
        f['source_flag'].attrs['encoding'] = '0=none, 1=Karakas_AGB, 2=CL18_CCSN, 3=PISN_HW02'

        f.attrs['description'] = 'Unified stellar yield tables for GIZMO resolved ISM'
        f.attrs['sources'] = 'Karakas(2010) + Chieffi&Limongi(2018) + Heger&Woosley(2002)'
        f.attrs['tracked_elements'] = TRACKED_ELEMENTS
        f.attrs['N_Z'] = N_Z
        f.attrs['N_M'] = N_M
        f.attrs['N_elements'] = N_elem

    sz = os.path.getsize(output_file) / 1e6
    print(f"Done. File: {sz:.2f} MB", flush=True)


def verify_yields(output_file):
    import h5py
    with h5py.File(output_file, 'r') as f:
        Z = f['Z'][:]
        M = f['M_init'][:]
        elems = [e.decode() for e in f['elements'][:]]
        yields = f['net_yields'][:]
        rtype = f['remnant_type'][:]
        rmass = f['remnant_mass'][:]
        source = f['source_flag'][:]

        iz_solar = np.argmin(np.abs(Z - 0.014))
        print(f"\n── Yield table sanity checks (Z={Z[iz_solar]}) ──")
        print(f"Elements: {elems}")
        print(f"Mass grid: {M[0]:.1f} - {M[-1]:.0f} Msun ({len(M)} points)")

        print(f"\n{'M':>6s} {'type':>5s} {'M_rem':>6s} {'src':>4s}  ", end='')
        for e in ['C', 'N', 'O', 'Fe']:
            print(f'{e:>8s}', end='')
        print()
        print('-' * 60)

        type_names = {0: 'WD', 1: 'NS', 2: 'BH', 3: 'PISN'}
        src_names = {0: 'none', 1: 'Kar', 2: 'C&L', 3: 'PISN'}
        for M_check in [1, 2, 3, 5, 8, 13, 15, 20, 25, 40, 80, 120, 150, 200, 300]:
            im = np.argmin(np.abs(M - M_check))
            t = type_names[rtype[iz_solar, im]]
            r = rmass[iz_solar, im]
            s = src_names[source[iz_solar, im]]
            print(f'{M[im]:6.1f} {t:>5s} {r:6.2f} {s:>4s}  ', end='')
            for e in ['C', 'N', 'O', 'Fe']:
                ie = elems.index(e)
                y = yields[iz_solar, im, ie]
                print(f'{y:8.3f}', end='')
            print()


if __name__ == '__main__':
    base_dir = '/raven/u/uli/phil/stellar_tables'
    output_file = os.path.join(base_dir, 'stellar_yields.hdf5')
    build_yield_tables(base_dir, output_file)
    verify_yields(output_file)
