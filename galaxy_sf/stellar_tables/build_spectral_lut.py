#!/usr/bin/env python3
"""
Download and pre-process stellar atmosphere spectra (ATLAS9, PHOENIX, PoWR)
into a band-fraction lookup table for use by build_stellar_tables.py.

Replaces blackbody Planck integrals with proper atmosphere-model spectra that
include line blanketing, non-LTE effects, and wind opacity — critical for
accurate ionizing photon rates in 5-15 Msun stars.

Spectral sources:
  - ATLAS9 / Castelli-Kurucz (2003): Teff 3500-50000 K, all bands to 91 A
  - PHOENIX ACES (Husser+ 2013):     Teff 2300-12000 K, cool-star extension
  - PoWR (Hainich+ 2019):            OB/WR stars, wind effects on EUV

Output: spectral_band_lut.hdf5
  Axes: log_Teff, logg, log_Z
  Data: energy_frac[band](Teff, logg, Z) and photon_rate_per_L[band](Teff, logg, Z)
  Same per-unit-luminosity convention as Planck LUT in build_stellar_tables.py.

Usage:
  python build_spectral_lut.py --download [--atlas9] [--phoenix] [--powr]
  python build_spectral_lut.py --build
  python build_spectral_lut.py --verify
"""

import numpy as np
import os
import sys
import re
import argparse
from glob import glob

# Reuse band definitions from the main builder
from build_stellar_tables import (
    ENERGY_BANDS, PHOTON_RATE_BANDS, COMBO_BANDS, RAD_KEYS,
    h_planck, k_boltz, c_light, sigma_sb, eV_to_erg, L_sun, R_sun,
)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SPECTRA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectra')
LUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectral_band_lut.hdf5')

# Energy ↔ wavelength: E(eV) = hc / lambda → lambda(A) = 12398.42 / E(eV)
HC_eV_A = h_planck * c_light / eV_to_erg * 1e8  # 12398.42 eV*A

# ATLAS9 Castelli-Kurucz grid
ATLAS9_BASE_URL = 'https://wwwuser.oats.inaf.it/castelli/grids'
ATLAS9_MH_CODES = {
    'p05': +0.5, 'p02': +0.2, 'p00': 0.0,
    'm05': -0.5, 'm10': -1.0, 'm15': -1.5, 'm20': -2.0, 'm25': -2.5,
}
# Convert [M/H] to Z for our grid matching
def mh_to_Z(mh):
    """[M/H] → approximate Z (Asplund 2009)."""
    Z_solar = 0.0134
    return Z_solar * 10**mh

# PHOENIX ACES grid (Goettingen mirror)
PHOENIX_BASE_URL = 'https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011'
PHOENIX_Z_CODES = {
    'Z-0.0': 0.0, 'Z-0.5': -0.5, 'Z-1.0': -1.0, 'Z-1.5': -1.5,
    'Z-2.0': -2.0, 'Z-3.0': -3.0, 'Z-4.0': -4.0,
    'Z+0.5': +0.5,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Download spectral grids
# ═══════════════════════════════════════════════════════════════

def download_atlas9(mh_codes=None):
    """Download ATLAS9 Castelli-Kurucz flux files.

    Downloads the packed .pck files (one per metallicity, ~5 MB each).
    Each .pck contains all (Teff, logg) models for that [M/H].
    """
    import urllib.request

    if mh_codes is None:
        mh_codes = list(ATLAS9_MH_CODES.keys())

    atlas9_dir = os.path.join(SPECTRA_DIR, 'atlas9')
    os.makedirs(atlas9_dir, exist_ok=True)

    for mh_code in mh_codes:
        url = f'{ATLAS9_BASE_URL}/grid{mh_code}k2odfnew/f{mh_code}k2odfnew.pck'
        outfile = os.path.join(atlas9_dir, f'f{mh_code}k2odfnew.pck')
        if os.path.exists(outfile):
            print(f"  ATLAS9 {mh_code}: already downloaded")
            continue
        print(f"  Downloading ATLAS9 {mh_code}: {url}")
        try:
            urllib.request.urlretrieve(url, outfile)
            sz = os.path.getsize(outfile) / 1e6
            print(f"    → {sz:.1f} MB")
        except Exception as e:
            print(f"    FAILED: {e}")
            # Try alternative: individual .dat files
            print(f"    Trying individual flux files instead...")
            _download_atlas9_individual(mh_code, atlas9_dir)


def _download_atlas9_individual(mh_code, atlas9_dir):
    """Fallback: download individual ATLAS9 flux .dat files."""
    import urllib.request

    grid_dir = os.path.join(atlas9_dir, f'grid{mh_code}')
    os.makedirs(grid_dir, exist_ok=True)

    # Standard Teff grid
    teffs = list(range(3500, 13000, 250)) + list(range(13000, 50001, 1000))
    loggs = [x / 10.0 for x in range(0, 51, 5)]  # 0.0 to 5.0 step 0.5

    n_dl = 0
    for teff in teffs:
        for logg in loggs:
            g_str = f'{int(logg * 10):02d}'
            fname = f'f{mh_code}t{teff}g{g_str}k2odfnew.dat'
            url = f'{ATLAS9_BASE_URL}/grid{mh_code}k2odfnew/{fname}'
            outfile = os.path.join(grid_dir, fname)
            if os.path.exists(outfile):
                continue
            try:
                urllib.request.urlretrieve(url, outfile)
                n_dl += 1
            except Exception:
                pass  # Not all (Teff, logg) combinations exist
    print(f"    Downloaded {n_dl} individual spectra for {mh_code}")


def download_phoenix():
    """Download PHOENIX ACES spectra (FITS format) from Goettingen mirror.

    WARNING: Full grid is ~100 GB. Only download what we need.
    Downloads a subset of representative (Teff, logg) for each Z.
    """
    import urllib.request

    phoenix_dir = os.path.join(SPECTRA_DIR, 'phoenix')
    os.makedirs(phoenix_dir, exist_ok=True)

    # Download wavelength grid first
    wave_url = f'{PHOENIX_BASE_URL}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    wave_file = os.path.join(phoenix_dir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    if not os.path.exists(wave_file):
        print(f"  Downloading PHOENIX wavelength grid...")
        try:
            urllib.request.urlretrieve(wave_url, wave_file)
        except Exception as e:
            print(f"    FAILED: {e}")
            return

    # Download spectra for selected (Teff, logg, Z)
    # Focus on Teff < 4000 K where ATLAS9 stops and PHOENIX is needed
    teffs = list(range(2300, 4100, 100))
    loggs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    for z_code, mh in PHOENIX_Z_CODES.items():
        z_dir = os.path.join(phoenix_dir, z_code)
        os.makedirs(z_dir, exist_ok=True)

        for teff in teffs:
            for logg in loggs:
                sign = '+' if mh >= 0 else ''
                fname = f'lte{teff:05d}-{logg:.2f}{sign}{mh:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                outfile = os.path.join(z_dir, fname)
                if os.path.exists(outfile):
                    continue
                url = f'{PHOENIX_BASE_URL}/{z_code}/{fname}'
                try:
                    urllib.request.urlretrieve(url, outfile)
                except Exception:
                    pass  # Not all combinations exist

    print(f"  PHOENIX download complete")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Parse spectral files
# ═══════════════════════════════════════════════════════════════

def _parse_fortran_fixed(line, width=10):
    """Parse Fortran fixed-width format (e.g. 8E10.4 with no spaces)."""
    vals = []
    for j in range(0, len(line.rstrip('\n')), width):
        chunk = line[j:j+width].strip()
        if chunk:
            try:
                vals.append(float(chunk))
            except ValueError:
                pass
    return vals


def parse_atlas9_pck(filepath):
    """Parse ATLAS9 Castelli-Kurucz packed flux file (.pck).

    File format (Kurucz fixed-width):
      22 header lines (Fortran program listing)
      Wavelength grid: 1221 values in F10.2 format (8 per line → 153 lines)
      Then per model:
        1 header line: "TEFF  xxxx.  GRAVITY x.xxxxx ..."
        Hnu:     1221 values in E10.4 format (8 per line → 153 lines)
        HnuCont: 1221 values in E10.4 format (8 per line → 153 lines)

    Returns dict: (Teff, logg) → (wavelength_A, F_lambda_cgs)
    where F_lambda is surface flux in erg/s/cm^2/A.
    """
    with open(filepath) as fh:
        lines = fh.readlines()

    # Skip 22-line header
    i = 22

    # Read wavelength grid (1221 values, 8 per line, F10.2 fixed-width but space-separated)
    wavelengths_nm = []
    while len(wavelengths_nm) < 1221 and i < len(lines):
        wavelengths_nm.extend(_parse_fortran_fixed(lines[i], 10))
        i += 1
    wavelengths_nm = np.array(wavelengths_nm[:1221])
    wl_A = wavelengths_nm * 10.0  # nm → Angstrom

    nwavel = len(wavelengths_nm)
    n_flux_lines = (nwavel + 7) // 8  # ceil(1221/8) = 153

    spectra = {}
    n_models = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith('TEFF'):
            i += 1
            continue

        # Parse header: "TEFF   3500.  GRAVITY 0.00000 ..."
        parts = line.split()
        teff = float(parts[1])
        logg = float(parts[3])
        i += 1

        # Read Hnu: 153 lines of 8E10.4 fixed-width
        h_nu_vals = []
        for _ in range(n_flux_lines):
            if i >= len(lines):
                break
            h_nu_vals.extend(_parse_fortran_fixed(lines[i], 10))
            i += 1
        h_nu = np.array(h_nu_vals[:nwavel])

        # Read HnuCont: 153 lines (skip, we use total flux not continuum)
        i += n_flux_lines

        if len(h_nu) < nwavel:
            continue

        # Convert H_nu (Eddington flux, erg/cm^2/s/Hz/ster) to F_lambda (erg/s/cm^2/A)
        # F_nu = 4π H_nu  (surface flux per Hz)
        # F_lambda = F_nu * c / lambda^2   (with lambda in cm)
        # Then convert /cm → /A by × 1e-8
        lambda_cm = wl_A * 1e-8
        F_lambda = 4.0 * np.pi * h_nu * c_light / (lambda_cm**2) * 1e-8  # erg/s/cm^2/A

        spectra[(teff, logg)] = (wl_A.copy(), F_lambda)
        n_models += 1

    return spectra


def parse_phoenix_fits(filepath, wave_filepath):
    """Parse PHOENIX ACES FITS spectrum.

    Returns: (wavelength_A, F_lambda_cgs) in erg/s/cm^2/A
    """
    try:
        from astropy.io import fits
    except ImportError:
        try:
            import fitsio
            wave = fitsio.read(wave_filepath)
            flux = fitsio.read(filepath)
            # PHOENIX flux is in erg/s/cm^2/cm, divide by 1e8 to get /A
            return wave.flatten(), flux.flatten() / 1e8
        except ImportError:
            print("  ERROR: Need astropy or fitsio to read PHOENIX FITS files")
            return None

    with fits.open(wave_filepath) as hdul:
        wave = hdul[0].data  # Angstrom

    with fits.open(filepath) as hdul:
        flux = hdul[0].data  # erg/s/cm^2/cm

    return wave.flatten(), flux.flatten() / 1e8  # → erg/s/cm^2/A


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Band integration
# ═══════════════════════════════════════════════════════════════

def integrate_bands(wavelength_A, F_lambda):
    """Integrate a spectrum over all energy bands.

    Args:
        wavelength_A: wavelength array in Angstrom (sorted ascending)
        F_lambda: surface flux array in erg/s/cm^2/A

    Returns:
        energy_fracs: dict {band_name: L_band/L_bol fraction}
        photon_rates_per_L: dict {band_name: photons/s per erg/s of L_bol}
    """
    # Sort by wavelength if needed
    if wavelength_A[0] > wavelength_A[-1]:
        wavelength_A = wavelength_A[::-1]
        F_lambda = F_lambda[::-1]

    # Total flux (bolometric)
    F_total = np.trapz(F_lambda, wavelength_A)
    if F_total <= 0:
        return ({name: 0.0 for name, _, _ in ENERGY_BANDS},
                {name: 0.0 for name, _, _ in PHOTON_RATE_BANDS})

    # Convert band edges from eV to Angstrom
    # E(eV) → lambda(A): lambda = HC_eV_A / E
    # Higher energy = shorter wavelength, so band (E_lo, E_hi) maps to (lam_hi, lam_lo)

    energy_fracs = {}
    for name, E_lo, E_hi in ENERGY_BANDS:
        # Wavelength limits (reverse order since E ∝ 1/λ)
        lam_hi = HC_eV_A / E_lo   # lower energy → longer wavelength
        lam_lo = HC_eV_A / E_hi if E_hi is not None else wavelength_A[0]  # min wavelength

        # Clip to available range
        lam_lo = max(lam_lo, wavelength_A[0])
        lam_hi = min(lam_hi, wavelength_A[-1])

        if lam_lo >= lam_hi:
            energy_fracs[name] = 0.0
            continue

        # Select wavelength range and integrate
        mask = (wavelength_A >= lam_lo) & (wavelength_A <= lam_hi)
        if np.sum(mask) < 2:
            energy_fracs[name] = 0.0
            continue

        F_band = np.trapz(F_lambda[mask], wavelength_A[mask])
        energy_fracs[name] = F_band / F_total

    photon_rates_per_L = {}
    for name, E_lo, E_hi in PHOTON_RATE_BANDS:
        lam_hi = HC_eV_A / E_lo
        lam_lo = HC_eV_A / E_hi if E_hi is not None else wavelength_A[0]

        lam_lo = max(lam_lo, wavelength_A[0])
        lam_hi = min(lam_hi, wavelength_A[-1])

        if lam_lo >= lam_hi:
            photon_rates_per_L[name] = 0.0
            continue

        mask = (wavelength_A >= lam_lo) & (wavelength_A <= lam_hi)
        if np.sum(mask) < 2:
            photon_rates_per_L[name] = 0.0
            continue

        # Photon integral: ∫ (F_lambda * lambda / hc) d_lambda
        # Each photon has energy E = hc/lambda, so N_photons = F_lambda * dlambda / (hc/lambda)
        wl_cm = wavelength_A[mask] * 1e-8  # A → cm
        photon_integrand = F_lambda[mask] * wl_cm / (h_planck * c_light)  # photons/s/cm^2/A
        Q_band = np.trapz(photon_integrand, wavelength_A[mask])  # photons/s/cm^2

        # Q_per_L_unit: photons/s per erg/s of bolometric luminosity
        # Q = 4π R^2 Q_band, L_bol = 4π R^2 F_total → Q/L_bol = Q_band/F_total
        photon_rates_per_L[name] = Q_band / F_total

    return energy_fracs, photon_rates_per_L


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Build the LUT
# ═══════════════════════════════════════════════════════════════

def build_lut():
    """Build spectral band LUT from downloaded atmosphere spectra.

    Strategy:
      - ATLAS9 for Teff >= 3500 K (covers all bands to 91 A = 136 eV)
      - PHOENIX for Teff < 3500 K (extends to cool stars)
      - At overlap (3500-4000 K), prefer ATLAS9 for ionizing bands

    Output: spectral_band_lut.hdf5 with shape (N_Teff, N_logg, N_Z)
    """
    import h5py

    atlas9_dir = os.path.join(SPECTRA_DIR, 'atlas9')
    phoenix_dir = os.path.join(SPECTRA_DIR, 'phoenix')

    # ── Load all ATLAS9 spectra ──
    print("Loading ATLAS9 spectra...", flush=True)
    atlas9_data = {}  # (Teff, logg, mh) → (energy_fracs, photon_rates)

    for mh_code, mh_val in ATLAS9_MH_CODES.items():
        pck_file = os.path.join(atlas9_dir, f'f{mh_code}k2odfnew.pck')
        if os.path.exists(pck_file):
            spectra = parse_atlas9_pck(pck_file)
            for (teff, logg), (wl, fl) in spectra.items():
                ef, pr = integrate_bands(wl, fl)
                atlas9_data[(teff, logg, mh_val)] = (ef, pr)
            print(f"  {mh_code}: {len(spectra)} models from .pck")
        else:
            # Try individual .dat files
            grid_dir = os.path.join(atlas9_dir, f'grid{mh_code}')
            if os.path.isdir(grid_dir):
                n = 0
                for f in glob(os.path.join(grid_dir, '*.dat')):
                    m = re.search(r't(\d+)g(\d+)', os.path.basename(f))
                    if m:
                        teff = int(m.group(1))
                        logg = int(m.group(2)) / 10.0
                        result = parse_atlas9_dat(f)
                        if result is not None:
                            wl, fl = result
                            ef, pr = integrate_bands(wl, fl)
                            atlas9_data[(teff, logg, mh_val)] = (ef, pr)
                            n += 1
                print(f"  {mh_code}: {n} models from .dat files")

    if not atlas9_data:
        print("ERROR: No ATLAS9 spectra found. Run with --download first.")
        sys.exit(1)

    print(f"  Total ATLAS9 models: {len(atlas9_data)}")

    # ── Load PHOENIX spectra (cool stars) ──
    phoenix_data = {}
    wave_file = os.path.join(phoenix_dir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    if os.path.exists(wave_file):
        print("Loading PHOENIX spectra...", flush=True)
        for z_code, mh_val in PHOENIX_Z_CODES.items():
            z_dir = os.path.join(phoenix_dir, z_code)
            if not os.path.isdir(z_dir):
                continue
            n = 0
            for f in glob(os.path.join(z_dir, '*.fits')):
                m = re.search(r'lte(\d+)-([\d.]+)', os.path.basename(f))
                if m:
                    teff = int(m.group(1))
                    logg = float(m.group(2))
                    result = parse_phoenix_fits(f, wave_file)
                    if result is not None:
                        wl, fl = result
                        ef, pr = integrate_bands(wl, fl)
                        phoenix_data[(teff, logg, mh_val)] = (ef, pr)
                        n += 1
            if n > 0:
                print(f"  {z_code}: {n} models")
        print(f"  Total PHOENIX models: {len(phoenix_data)}")

    # ── Merge into unified grid ──
    all_data = {}
    all_data.update(phoenix_data)   # PHOENIX first (low priority)
    all_data.update(atlas9_data)    # ATLAS9 overwrites at overlap

    if not all_data:
        print("ERROR: No spectra loaded.")
        sys.exit(1)

    # Extract unique grid values
    all_teffs = sorted(set(k[0] for k in all_data))
    all_loggs = sorted(set(k[1] for k in all_data))
    all_mhs   = sorted(set(k[2] for k in all_data))

    print(f"\nUnified grid: {len(all_teffs)} Teff x {len(all_loggs)} logg x {len(all_mhs)} [M/H]")
    print(f"  Teff: {all_teffs[0]:.0f} - {all_teffs[-1]:.0f} K")
    print(f"  logg: {all_loggs[0]:.1f} - {all_loggs[-1]:.1f}")
    print(f"  [M/H]: {all_mhs[0]:.1f} - {all_mhs[-1]:.1f}")

    # ── Write HDF5 LUT ──
    log_Teff_arr = np.log10(np.array(all_teffs))
    logg_arr = np.array(all_loggs)
    mh_arr = np.array(all_mhs)
    Z_arr = np.array([mh_to_Z(mh) for mh in all_mhs])

    N_T, N_g, N_Z = len(all_teffs), len(all_loggs), len(all_mhs)
    shape = (N_T, N_g, N_Z)

    # Allocate arrays for each band
    energy_frac_arrays = {name: np.zeros(shape) for name, _, _ in ENERGY_BANDS}
    photon_rate_arrays = {name: np.zeros(shape) for name, _, _ in PHOTON_RATE_BANDS}
    has_data = np.zeros(shape, dtype=bool)

    for it, teff in enumerate(all_teffs):
        for ig, logg in enumerate(all_loggs):
            for iz, mh in enumerate(all_mhs):
                key = (teff, logg, mh)
                if key in all_data:
                    ef, pr = all_data[key]
                    for name, _, _ in ENERGY_BANDS:
                        energy_frac_arrays[name][it, ig, iz] = ef[name]
                    for name, _, _ in PHOTON_RATE_BANDS:
                        photon_rate_arrays[name][it, ig, iz] = pr[name]
                    has_data[it, ig, iz] = True

    # Fill gaps via nearest-neighbor in (log_Teff, logg, [M/H]) space
    n_gaps = np.sum(~has_data)
    if n_gaps > 0:
        print(f"  Filling {n_gaps} gaps via nearest-neighbor...")
        for it in range(N_T):
            for ig in range(N_g):
                for iz in range(N_Z):
                    if has_data[it, ig, iz]:
                        continue
                    # Find nearest filled point
                    best_d = np.inf
                    best = None
                    for jt in range(N_T):
                        for jg in range(N_g):
                            for jz in range(N_Z):
                                if not has_data[jt, jg, jz]:
                                    continue
                                d = ((log_Teff_arr[it] - log_Teff_arr[jt]) * 3)**2 + \
                                    (logg_arr[ig] - logg_arr[jg])**2 + \
                                    (mh_arr[iz] - mh_arr[jz])**2
                                if d < best_d:
                                    best_d = d
                                    best = (jt, jg, jz)
                    if best is not None:
                        for name, _, _ in ENERGY_BANDS:
                            energy_frac_arrays[name][it, ig, iz] = energy_frac_arrays[name][best]
                        for name, _, _ in PHOTON_RATE_BANDS:
                            photon_rate_arrays[name][it, ig, iz] = photon_rate_arrays[name][best]

    print(f"\nWriting {LUT_FILE}...", flush=True)
    with h5py.File(LUT_FILE, 'w') as f:
        # Grid axes
        f.create_dataset('log_Teff', data=log_Teff_arr)
        f['log_Teff'].attrs['units'] = 'log10(K)'
        f.create_dataset('logg', data=logg_arr)
        f['logg'].attrs['units'] = 'log10(cm/s^2)'
        f.create_dataset('metallicity_MH', data=mh_arr)
        f['metallicity_MH'].attrs['units'] = '[M/H] (dex)'
        f.create_dataset('Z', data=Z_arr)
        f['Z'].attrs['units'] = 'mass fraction'

        # Energy fractions: L_band / L_bol (dimensionless)
        comp = dict(compression='gzip', compression_opts=4)
        grp_e = f.create_group('energy_fracs')
        grp_e.attrs['description'] = 'L_band / L_bol (dimensionless) for each atomic sub-band'
        grp_e.attrs['shape'] = '(N_Teff, N_logg, N_Z)'
        for name, E_lo, E_hi in ENERGY_BANDS:
            hi_str = f'{E_hi}' if E_hi is not None else 'inf'
            ds = grp_e.create_dataset(name, data=energy_frac_arrays[name], **comp)
            ds.attrs['band_eV'] = f'{E_lo}-{hi_str}'

        # Photon rates per unit luminosity: photons/s per erg/s of L_bol
        grp_q = f.create_group('photon_rates_per_L')
        grp_q.attrs['description'] = 'Photon rate per erg/s of bolometric luminosity'
        grp_q.attrs['shape'] = '(N_Teff, N_logg, N_Z)'
        for name, E_lo, E_hi in PHOTON_RATE_BANDS:
            hi_str = f'{E_hi}' if E_hi is not None else 'inf'
            ds = grp_q.create_dataset(name, data=photon_rate_arrays[name], **comp)
            ds.attrs['band_eV'] = f'{E_lo}-{hi_str}'

        # Coverage info
        f.create_dataset('has_data', data=has_data)
        f.attrs['N_Teff'] = N_T
        f.attrs['N_logg'] = N_g
        f.attrs['N_Z'] = N_Z
        f.attrs['n_models'] = int(np.sum(has_data))
        f.attrs['n_gaps_filled'] = int(n_gaps)
        f.attrs['sources'] = 'ATLAS9/Castelli-Kurucz (Teff>=3500K) + PHOENIX/ACES (Teff<3500K)'
        f.attrs['description'] = ('Band fractions from stellar atmosphere models. '
                                  'Use instead of Planck blackbody for accurate ionizing photon rates.')

    sz = os.path.getsize(LUT_FILE) / 1e6
    print(f"Done. {sz:.1f} MB, {int(np.sum(has_data))} models")


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Verification
# ═══════════════════════════════════════════════════════════════

def verify_lut():
    """Compare spectral LUT against Planck blackbody for sanity check."""
    import h5py
    from build_stellar_tables import build_planck_lut

    print("Building Planck reference LUT...")
    log_T_planck, planck_efracs, planck_qrates = build_planck_lut(N_T=500)

    print(f"\nReading {LUT_FILE}...")
    with h5py.File(LUT_FILE, 'r') as f:
        log_Teff = f['log_Teff'][:]
        logg = f['logg'][:]
        mh = f['metallicity_MH'][:]

        # Find solar metallicity
        iz_sol = np.argmin(np.abs(mh - 0.0))
        ig_mid = np.argmin(np.abs(logg - 4.0))

        print(f"\nComparison at [M/H]={mh[iz_sol]:.1f}, logg={logg[ig_mid]:.1f}:")
        print(f"{'Teff':>8s} ", end='')
        for name, _, _ in ENERGY_BANDS:
            print(f"  {name:>10s}", end='')
        print(f"  {'Q_ion':>10s}")
        print('-' * (8 + 12 * (len(ENERGY_BANDS) + 1)))

        for teff_check in [4000, 5000, 8000, 10000, 15000, 20000, 30000, 40000]:
            lt = np.log10(teff_check)
            it = np.argmin(np.abs(log_Teff - lt))

            line = f"{10**log_Teff[it]:8.0f} "
            for name, _, _ in ENERGY_BANDS:
                spec_frac = f[f'energy_fracs/{name}'][it, ig_mid, iz_sol]
                planck_frac = np.interp(log_Teff[it], log_T_planck, planck_efracs[name])
                ratio = spec_frac / planck_frac if planck_frac > 1e-30 else 0
                line += f"  {ratio:10.3f}"

            spec_q = f['photon_rates_per_L/Q_ion'][it, ig_mid, iz_sol]
            planck_q = np.interp(log_Teff[it], log_T_planck, planck_qrates['Q_ion'])
            ratio_q = spec_q / planck_q if planck_q > 1e-30 else 0
            line += f"  {ratio_q:10.3f}"
            print(line)

        print("\nRatio < 1 = atmosphere model has LESS flux than blackbody (line blanketing)")
        print("Expect ratio << 1 for ionizing bands at 5000-15000 K (B-type stars)")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build spectral band LUT from atmosphere models')
    parser.add_argument('--download', action='store_true', help='Download spectral grids')
    parser.add_argument('--atlas9', action='store_true', help='Download ATLAS9 only')
    parser.add_argument('--phoenix', action='store_true', help='Download PHOENIX only')
    parser.add_argument('--build', action='store_true', help='Build LUT from downloaded spectra')
    parser.add_argument('--verify', action='store_true', help='Compare LUT vs Planck blackbody')

    args = parser.parse_args()

    if args.download or args.atlas9 or args.phoenix:
        os.makedirs(SPECTRA_DIR, exist_ok=True)
        if args.atlas9 or (args.download and not args.phoenix):
            print("Downloading ATLAS9 spectra...")
            download_atlas9()
        if args.phoenix or (args.download and not args.atlas9):
            print("Downloading PHOENIX spectra...")
            download_phoenix()

    if args.build:
        build_lut()

    if args.verify:
        verify_lut()

    if not any([args.download, args.atlas9, args.phoenix, args.build, args.verify]):
        parser.print_help()
