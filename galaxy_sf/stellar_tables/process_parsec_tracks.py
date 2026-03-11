#!/usr/bin/env python3
"""
Process PARSEC v1.2s stellar evolution tracks into lookup tables for GIZMO.

For each (Z, M_initial), produces time-dependent:
  - Q_ion:    ionizing photon rate (>13.6 eV) [photons/s]
  - L_FUV:    FUV/photoelectric heating luminosity (6-11.2 eV) [erg/s]
  - L_LW:     Lyman-Werner luminosity (11.2-13.6 eV) [erg/s]
  - R_star:   stellar radius [cm]
  - M_star:   current stellar mass [Msun]
  - Mdot:     wind mass loss rate [Msun/yr]
  - Surface abundances: H, He, C, N, O (mass fractions)

Blackbody integration via pre-tabulated Planck integrals for speed.
Output: HDF5 file with regular grids for trilinear interpolation in C code.
"""

import numpy as np
import os
import re
import sys
from scipy.integrate import quad
from glob import glob

# ── Physical constants (CGS) ──
h_planck = 6.62607015e-27   # erg s
k_boltz  = 1.380649e-16     # erg/K
c_light  = 2.99792458e10    # cm/s
sigma_sb = 5.670374419e-5   # erg cm^-2 s^-1 K^-4
eV_to_erg = 1.602176634e-12 # erg per eV
L_sun    = 3.828e33          # erg/s
R_sun    = 6.957e10          # cm

# ── Energy band edges ──
E_FUV_lo = 6.0    # eV (FUV/photoelectric heating lower edge)
E_FUV_hi = 11.2   # eV (FUV upper edge)
E_LW_lo  = 11.2   # eV (Lyman-Werner lower edge)
E_LW_hi  = 13.6   # eV (Lyman-Werner upper edge = ionization threshold)

# M1 RT ionizing sub-bands (RT_CHEM_PHOTOION=2 thresholds)
E_ion_H0_lo  = 13.6   # eV (H ionization threshold)
E_ion_H0_hi  = 24.6   # eV (He I ionization threshold)
E_ion_He0_lo = 24.6   # eV
E_ion_He0_hi = 54.4   # eV (He II ionization threshold)
E_ion_He1_lo = 54.4   # eV
E_ion_He1_hi = 70.0   # eV
E_ion_He2_lo = 70.0   # eV (hard UV, >70 eV to infinity)


# ═══════════════════════════════════════════════════════════════════════
# Pre-tabulated Planck integrals as functions of temperature
# ═══════════════════════════════════════════════════════════════════════
#
# Q_ion = (L/σT⁴) × (2π/c²) × (kT/h)³ × I_photon(x0)
#   where x0 = 13.6 eV / kT,  I_photon(x0) = ∫_{x0}^∞ x²/(e^x-1) dx
#
# L_band = (L/σT⁴) × (2πh/c²) × (kT/h)⁴ × I_energy(x1, x2)
#   where xi = Ei/kT,  I_energy(x1,x2) = ∫_{x1}^{x2} x³/(e^x-1) dx

def _planck_photon_integrand(x):
    if x > 500: return 0.0
    return x**2 / (np.exp(x) - 1.0)

def _planck_energy_integrand(x):
    if x > 500: return 0.0
    return x**3 / (np.exp(x) - 1.0)


def build_planck_lut(N_T=2000):
    """
    Pre-compute Planck integrals on a fine log(T) grid.
    Returns interpolation functions for Q_ion prefactor and band luminosities.
    """
    log_T_min, log_T_max = 3.0, 5.7  # 1000 K to 500,000 K
    log_T_grid = np.linspace(log_T_min, log_T_max, N_T)
    T_grid = 10**log_T_grid

    # For each T, compute the dimensionless integrals
    # These depend only on x_i = E_i / kT
    I_Qion    = np.zeros(N_T)  # ∫_{x0}^∞ x²/(e^x-1) dx, x0 = 13.6 eV/kT
    I_FUV     = np.zeros(N_T)  # ∫_{x1}^{x2} x³/(e^x-1) dx, 6.0-11.2 eV
    I_LW      = np.zeros(N_T)  # ∫_{x1}^{x2} x³/(e^x-1) dx, 11.2-13.6 eV
    I_ion_tot = np.zeros(N_T)  # ∫_{13.6}^∞  x³/(e^x-1) dx  (total ionizing energy)
    I_ion_H0  = np.zeros(N_T)  # ∫_{13.6}^{24.6} x³/(e^x-1) dx
    I_ion_He0 = np.zeros(N_T)  # ∫_{24.6}^{54.4} x³/(e^x-1) dx
    I_ion_He1 = np.zeros(N_T)  # ∫_{54.4}^{70.0} x³/(e^x-1) dx
    I_ion_He2 = np.zeros(N_T)  # ∫_{70.0}^∞      x³/(e^x-1) dx

    print(f"  Building Planck integral LUT ({N_T} temperatures)...", flush=True)
    for i, T in enumerate(T_grid):
        kT = k_boltz * T
        x_fuv_lo = E_FUV_lo * eV_to_erg / kT
        x_fuv_hi = E_FUV_hi * eV_to_erg / kT
        x_lw_lo  = E_LW_lo  * eV_to_erg / kT
        x_lw_hi  = E_LW_hi  * eV_to_erg / kT
        x_H0_lo  = E_ion_H0_lo  * eV_to_erg / kT
        x_H0_hi  = E_ion_H0_hi  * eV_to_erg / kT
        x_He0_lo = E_ion_He0_lo * eV_to_erg / kT
        x_He0_hi = E_ion_He0_hi * eV_to_erg / kT
        x_He1_lo = E_ion_He1_lo * eV_to_erg / kT
        x_He1_hi = E_ion_He1_hi * eV_to_erg / kT
        x_He2_lo = E_ion_He2_lo * eV_to_erg / kT

        if x_lw_hi < 200:
            I_Qion[i], _ = quad(_planck_photon_integrand, x_lw_hi, np.inf, limit=200)
        if x_fuv_lo < 200:
            I_FUV[i], _ = quad(_planck_energy_integrand, x_fuv_lo, min(x_fuv_hi, 500), limit=200)
        if x_lw_lo < 200:
            I_LW[i], _ = quad(_planck_energy_integrand, x_lw_lo, min(x_lw_hi, 500), limit=200)
        if x_H0_lo < 200:
            I_ion_tot[i], _ = quad(_planck_energy_integrand, x_H0_lo, np.inf, limit=200)
            I_ion_H0[i], _  = quad(_planck_energy_integrand, x_H0_lo, min(x_H0_hi, 500), limit=200)
        if x_He0_lo < 200:
            I_ion_He0[i], _ = quad(_planck_energy_integrand, x_He0_lo, min(x_He0_hi, 500), limit=200)
        if x_He1_lo < 200:
            I_ion_He1[i], _ = quad(_planck_energy_integrand, x_He1_lo, min(x_He1_hi, 500), limit=200)
        if x_He2_lo < 200:
            I_ion_He2[i], _ = quad(_planck_energy_integrand, x_He2_lo, np.inf, limit=200)

    # Store full prefactored quantities per unit luminosity:
    # Q_ion / L = (1/σT⁴) × (2π/c²) × (kT/h)³ × I_photon
    # L_band / L = (1/σT⁴) × (2πh/c²) × (kT/h)⁴ × I_energy
    Qion_per_L   = np.zeros(N_T)
    FUV_frac     = np.zeros(N_T)
    LW_frac      = np.zeros(N_T)
    ion_tot_frac = np.zeros(N_T)
    ion_H0_frac  = np.zeros(N_T)
    ion_He0_frac = np.zeros(N_T)
    ion_He1_frac = np.zeros(N_T)
    ion_He2_frac = np.zeros(N_T)

    for i, T in enumerate(T_grid):
        kT = k_boltz * T
        inv_sigT4 = 1.0 / (sigma_sb * T**4)
        Qion_per_L[i]  = inv_sigT4 * (2*np.pi / c_light**2) * (kT/h_planck)**3 * I_Qion[i]
        energy_pref = inv_sigT4 * (2*np.pi*h_planck / c_light**2) * (kT/h_planck)**4
        FUV_frac[i]     = energy_pref * I_FUV[i]
        LW_frac[i]      = energy_pref * I_LW[i]
        ion_tot_frac[i] = energy_pref * I_ion_tot[i]
        ion_H0_frac[i]  = energy_pref * I_ion_H0[i]
        ion_He0_frac[i] = energy_pref * I_ion_He0[i]
        ion_He1_frac[i] = energy_pref * I_ion_He1[i]
        ion_He2_frac[i] = energy_pref * I_ion_He2[i]

    print(f"  Done. T range: {T_grid[0]:.0f} - {T_grid[-1]:.0f} K", flush=True)
    return (log_T_grid, Qion_per_L, FUV_frac, LW_frac,
            ion_tot_frac, ion_H0_frac, ion_He0_frac, ion_He1_frac, ion_He2_frac)


def apply_planck_lut(logTeff_arr, logL_arr, log_T_grid, Qion_per_L, FUV_frac, LW_frac,
                     ion_tot_frac, ion_H0_frac, ion_He0_frac, ion_He1_frac, ion_He2_frac):
    """
    Vectorized lookup: given arrays of logTeff and logL, return band luminosities.
    Returns: Q_ion, L_FUV, L_LW, L_ion_tot, L_ion_H0, L_ion_He0, L_ion_He1, L_ion_He2
    """
    L_arr = 10**logL_arr * L_sun  # erg/s

    # Interpolate the per-unit-L quantities
    q_per_L  = np.interp(logTeff_arr, log_T_grid, Qion_per_L,   left=0.0, right=0.0)
    fuv_f    = np.interp(logTeff_arr, log_T_grid, FUV_frac,     left=0.0, right=0.0)
    lw_f     = np.interp(logTeff_arr, log_T_grid, LW_frac,      left=0.0, right=0.0)
    ion_tot_f = np.interp(logTeff_arr, log_T_grid, ion_tot_frac, left=0.0, right=0.0)
    ion_H0_f  = np.interp(logTeff_arr, log_T_grid, ion_H0_frac,  left=0.0, right=0.0)
    ion_He0_f = np.interp(logTeff_arr, log_T_grid, ion_He0_frac, left=0.0, right=0.0)
    ion_He1_f = np.interp(logTeff_arr, log_T_grid, ion_He1_frac, left=0.0, right=0.0)
    ion_He2_f = np.interp(logTeff_arr, log_T_grid, ion_He2_frac, left=0.0, right=0.0)

    return (L_arr * q_per_L, L_arr * fuv_f, L_arr * lw_f,
            L_arr * ion_tot_f, L_arr * ion_H0_f, L_arr * ion_He0_f,
            L_arr * ion_He1_f, L_arr * ion_He2_f)


# ═══════════════════════════════════════════════════════════════════════
# Track I/O
# ═══════════════════════════════════════════════════════════════════════

def parse_track_file(filepath):
    """Parse a PARSEC track .DAT file. Returns dict of arrays."""
    data = []
    with open(filepath, 'r') as f:
        header = f.readline().split()
        for line in f:
            vals = line.split()
            if len(vals) != len(header):
                continue
            data.append([float(v) for v in vals])
    if len(data) == 0:
        return None
    arr = np.array(data)
    return {name: arr[:, i] for i, name in enumerate(header)}


def extract_metallicity(dirname):
    m = re.match(r'Z([\d.]+)Y', dirname)
    return float(m.group(1)) if m else None


def extract_initial_mass(filename):
    m = re.search(r'_M([\d.]+)\.DAT', filename)
    return float(m.group(1)) if m else None


def process_all_tracks(base_dir):
    """Load all PARSEC tracks. Returns list of dicts."""
    tracks = []
    zdirs = sorted(glob(os.path.join(base_dir, 'Z*Y*')))
    zdirs = [d for d in zdirs if os.path.isdir(d)]
    print(f"Found {len(zdirs)} metallicity directories", flush=True)

    for zdir in zdirs:
        dirname = os.path.basename(zdir)
        Z = extract_metallicity(dirname)
        if Z is None:
            continue
        dat_files = sorted(glob(os.path.join(zdir, '*.DAT')))
        print(f"  Z={Z}: {len(dat_files)} track files", flush=True)

        for datfile in dat_files:
            M_init = extract_initial_mass(os.path.basename(datfile))
            if M_init is None:
                continue
            d = parse_track_file(datfile)
            if d is None:
                continue
            tracks.append({
                'Z': Z, 'M_init': M_init,
                'age': d['AGE'], 'logL': d['LOG_L'], 'logTeff': d['LOG_TE'],
                'logR': d['LOG_R'], 'mass': d['MASS'], 'log_mdot': d['LOG_RAT'],
                'H_sup': d['H_SUP'], 'He_sup': d['HE_SUP'],
                'C_sup': d['C_SUP'], 'N_sup': d['N_SUP'], 'O_sup': d['O_SUP'],
                'phase': d['PHASE'],
            })
    return tracks


def resample_track(track, Q_ion, L_FUV, L_LW, L_ion_tot, L_ion_H0, L_ion_He0,
                   L_ion_He1, L_ion_He2, log_age_grid):
    """Resample a track onto common log(age) grid. Beyond lifetime → 0."""
    age = track['age']
    lifetime = age[-1]
    log_lifetime = np.log10(max(lifetime, 1.0))

    mask = age > 0
    if np.sum(mask) < 2:
        return None

    log_age_track = np.log10(age[mask])

    # Convert radiation quantities to log10 for interpolation
    LOG_FLOOR = -99.0
    log_Q_ion    = np.where(Q_ion[mask] > 0,    np.log10(Q_ion[mask]),    LOG_FLOOR)
    log_L_FUV    = np.where(L_FUV[mask] > 0,    np.log10(L_FUV[mask]),    LOG_FLOOR)
    log_L_LW     = np.where(L_LW[mask] > 0,     np.log10(L_LW[mask]),     LOG_FLOOR)
    log_L_ion_tot = np.where(L_ion_tot[mask] > 0, np.log10(L_ion_tot[mask]), LOG_FLOOR)
    log_L_ion_H0  = np.where(L_ion_H0[mask] > 0,  np.log10(L_ion_H0[mask]),  LOG_FLOOR)
    log_L_ion_He0 = np.where(L_ion_He0[mask] > 0, np.log10(L_ion_He0[mask]), LOG_FLOOR)
    log_L_ion_He1 = np.where(L_ion_He1[mask] > 0, np.log10(L_ion_He1[mask]), LOG_FLOOR)
    log_L_ion_He2 = np.where(L_ion_He2[mask] > 0, np.log10(L_ion_He2[mask]), LOG_FLOOR)

    quantities = {
        'log_Q_ion': log_Q_ion, 'log_L_FUV': log_L_FUV, 'log_L_LW': log_L_LW,
        'log_L_ion_tot': log_L_ion_tot, 'log_L_ion_H0': log_L_ion_H0,
        'log_L_ion_He0': log_L_ion_He0, 'log_L_ion_He1': log_L_ion_He1,
        'log_L_ion_He2': log_L_ion_He2,
        'logL': track['logL'][mask], 'logTeff': track['logTeff'][mask],
        'logR': track['logR'][mask], 'mass': track['mass'][mask],
        'log_mdot': track['log_mdot'][mask],
        'H_sup': track['H_sup'][mask], 'He_sup': track['He_sup'][mask],
        'C_sup': track['C_sup'][mask], 'N_sup': track['N_sup'][mask],
        'O_sup': track['O_sup'][mask],
    }

    result = {}
    for key, vals in quantities.items():
        dead_val = LOG_FLOOR if key.startswith('log_') else 0.0
        interp = np.interp(log_age_grid, log_age_track, vals, left=vals[0], right=dead_val)
        interp[log_age_grid > log_lifetime] = dead_val
        result[key] = interp

    result['lifetime'] = lifetime
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════

def build_lookup_tables(base_dir, output_file):
    import h5py

    # Build Planck integral lookup table (one-time cost, ~30s)
    (log_T_grid, Qion_per_L, FUV_frac, LW_frac,
     ion_tot_frac, ion_H0_frac, ion_He0_frac, ion_He1_frac, ion_He2_frac) = build_planck_lut(N_T=2000)

    # Load all tracks
    tracks = process_all_tracks(base_dir)
    if not tracks:
        print("No tracks found!")
        return

    Z_vals = sorted(set(t['Z'] for t in tracks))
    M_vals = sorted(set(t['M_init'] for t in tracks))
    print(f"\nMetallicities ({len(Z_vals)}): {Z_vals}")
    print(f"Masses ({len(M_vals)}): {M_vals[0]:.3f} - {M_vals[-1]:.1f} Msun", flush=True)

    # Common log(age) grid
    N_age = 300
    log_age_min, log_age_max = 2.0, 10.3
    log_age_grid = np.linspace(log_age_min, log_age_max, N_age)

    N_Z, N_M = len(Z_vals), len(M_vals)
    Z_to_idx = {z: i for i, z in enumerate(Z_vals)}
    M_to_idx = {m: i for i, m in enumerate(M_vals)}

    shape = (N_Z, N_M, N_age)
    tables = {name: np.zeros(shape) for name in [
        'log_Q_ion', 'log_L_FUV', 'log_L_LW',
        'log_L_ion_tot', 'log_L_ion_H0', 'log_L_ion_He0', 'log_L_ion_He1', 'log_L_ion_He2',
        'logL', 'logTeff', 'logR', 'mass',
        'H_sup', 'He_sup', 'C_sup', 'N_sup', 'O_sup']}
    tables['log_mdot'] = np.full(shape, -30.0)
    lifetime_table = np.zeros((N_Z, N_M))

    n_total = len(tracks)
    for idx, track in enumerate(tracks):
        iz = Z_to_idx[track['Z']]
        im = M_to_idx[track['M_init']]

        if (idx + 1) % 200 == 0 or idx == 0:
            print(f"  Track {idx+1}/{n_total}: Z={track['Z']}, M={track['M_init']} Msun", flush=True)

        # Vectorized blackbody computation via LUT
        (Q_ion, L_FUV, L_LW, L_ion_tot, L_ion_H0, L_ion_He0,
         L_ion_He1, L_ion_He2) = apply_planck_lut(
            track['logTeff'], track['logL'], log_T_grid, Qion_per_L, FUV_frac, LW_frac,
            ion_tot_frac, ion_H0_frac, ion_He0_frac, ion_He1_frac, ion_He2_frac)

        result = resample_track(track, Q_ion, L_FUV, L_LW, L_ion_tot, L_ion_H0,
                                L_ion_He0, L_ion_He1, L_ion_He2, log_age_grid)
        if result is None:
            continue

        for key in tables:
            if key in result:
                tables[key][iz, im, :] = result[key]
        lifetime_table[iz, im] = result['lifetime']

    # Write HDF5
    print(f"\nWriting {output_file}...", flush=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('Z', data=np.array(Z_vals))
        f.create_dataset('log_Z', data=np.log10(np.array(Z_vals)))
        f.create_dataset('M_init', data=np.array(M_vals))
        f.create_dataset('log_M_init', data=np.log10(np.array(M_vals)))
        f.create_dataset('log_age_yr', data=log_age_grid)
        f.create_dataset('lifetime_yr', data=lifetime_table)

        ds_names = {
            'log_Q_ion': ('log_Q_ion', 'log10(photons/s), blackbody E > 13.6 eV'),
            'log_L_FUV': ('log_L_FUV', 'log10(erg/s), blackbody 6.0-11.2 eV, FUV/photoelectric'),
            'log_L_LW': ('log_L_LW', 'log10(erg/s), blackbody 11.2-13.6 eV, Lyman-Werner'),
            'log_L_ion_tot': ('log_L_ion_tot', 'log10(erg/s), blackbody > 13.6 eV, total ionizing'),
            'log_L_ion_H0': ('log_L_ion_H0', 'log10(erg/s), blackbody 13.6-24.6 eV, H ionizing'),
            'log_L_ion_He0': ('log_L_ion_He0', 'log10(erg/s), blackbody 24.6-54.4 eV, He I ionizing'),
            'log_L_ion_He1': ('log_L_ion_He1', 'log10(erg/s), blackbody 54.4-70.0 eV, He II ionizing'),
            'log_L_ion_He2': ('log_L_ion_He2', 'log10(erg/s), blackbody > 70.0 eV, hard UV'),
            'logL': ('logL', 'log10(L/L_sun)'),
            'logTeff': ('logTeff', 'log10(T_eff/K)'),
            'logR': ('logR_cm', 'log10(R/cm)'),
            'mass': ('M_current', 'Msun'),
            'log_mdot': ('log_Mdot', 'log10(dM/dt in Msun/yr)'),
            'H_sup': ('H_surface', 'mass fraction'),
            'He_sup': ('He_surface', 'mass fraction'),
            'C_sup': ('C_surface', 'mass fraction'),
            'N_sup': ('N_surface', 'mass fraction'),
            'O_sup': ('O_surface', 'mass fraction'),
        }
        for key, (dsname, units) in ds_names.items():
            ds = f.create_dataset(dsname, data=tables[key],
                                  compression='gzip', compression_opts=4)
            ds.attrs['units'] = units

        f.attrs['description'] = 'PARSEC v1.2s stellar evolution lookup tables for GIZMO'
        f.attrs['N_Z'] = N_Z
        f.attrs['N_M'] = N_M
        f.attrs['N_age'] = N_age
        f.attrs['band_edges_eV'] = [6.0, 11.2, 13.6, 24.6, 54.4, 70.0]

    sz = os.path.getsize(output_file) / 1e6
    print(f"Done. Shape: ({N_Z}, {N_M}, {N_age}), file: {sz:.1f} MB", flush=True)


def verify_sanity(output_file):
    import h5py
    with h5py.File(output_file, 'r') as f:
        Z = f['Z'][:]
        M = f['M_init'][:]
        log_age = f['log_age_yr'][:]
        logQ = f['log_Q_ion'][:]
        lifetime = f['lifetime_yr'][:]

        iz_solar = np.argmin(np.abs(Z - 0.014))
        im_20 = np.argmin(np.abs(M - 20.0))
        im_1  = np.argmin(np.abs(M - 1.0))
        im_100 = np.argmin(np.abs(M - 100.0))
        i_zams = np.argmin(np.abs(log_age - 5.0))  # ~100 kyr

        print(f"\n── Sanity checks (Z={Z[iz_solar]}) ──")
        print(f"Lifetime(1 Msun):   {lifetime[iz_solar, im_1]:.3e} yr")
        print(f"Lifetime(20 Msun):  {lifetime[iz_solar, im_20]:.3e} yr")
        print(f"Lifetime(100 Msun): {lifetime[iz_solar, im_100]:.3e} yr")
        print(f"log Q_ion(1 Msun, 100kyr):   {logQ[iz_solar, im_1, i_zams]:.2f}")
        print(f"log Q_ion(20 Msun, 100kyr):  {logQ[iz_solar, im_20, i_zams]:.2f}")
        print(f"log Q_ion(100 Msun, 100kyr): {logQ[iz_solar, im_100, i_zams]:.2f}")

        logFUV = f['log_L_FUV'][:]
        logLW = f['log_L_LW'][:]
        print(f"log L_FUV(20 Msun, 100kyr):  {logFUV[iz_solar, im_20, i_zams]:.2f}  (6-11.2 eV)")
        print(f"log L_LW(20 Msun, 100kyr):   {logLW[iz_solar, im_20, i_zams]:.2f}  (11.2-13.6 eV)")

        mass = f['M_current'][:]
        logR = f['logR_cm'][:]
        R_sun_cm = 6.957e10
        R_20 = 10**(logR[iz_solar, im_20, i_zams]) / R_sun_cm
        print(f"M_current(100 Msun, end): {mass[iz_solar, im_100, -1]:.1f} Msun (started 100)")
        print(f"R(20 Msun, 100kyr): {R_20:.1f} R_sun")


if __name__ == '__main__':
    base_dir = '/raven/u/uli/phil/stellar_tables/parsec1.2'
    output_file = os.path.join(base_dir, 'parsec_v12s_tables.hdf5')
    build_lookup_tables(base_dir, output_file)
    verify_sanity(output_file)
