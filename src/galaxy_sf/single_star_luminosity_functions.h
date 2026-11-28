/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Band luminosities of IMF-sampled single stars from the stellar-evolution tables at the star's current age: one
   KOKKOS_INLINE_FUNCTION per quantity, on a table view (stellar_evolution_tables.h) and the star-record columns it needs.
   The host wrappers in single_star_luminosity.cc call them with the host tables and SingleStars; a device kernel with
   All.StellarTablesDev and All.SingleStarRecDev (read through the translation unit's All mirror). Include after allvars.h
   and proto.h; in a Kokkos translation unit after gpu_all_mirror.h. */
#ifndef SINGLE_STAR_LUMINOSITY_FUNCTIONS_H
#define SINGLE_STAR_LUMINOSITY_FUNCTIONS_H

#include "stellar_evolution_tables.h"
#include "../gravity/cosmology_functions.h"   /* evaluate_time_since_t_initial_in_Gyr_core */

#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_SINGLE_STARS_FROM_IMF)

/* the host records as a view (SingleStars lives in shared space: the same addresses as All.SingleStarRecDev) */
#define SINGLE_STAR_RECORD_HOST_VIEW() (SingleStarRecordView{SingleStars.M_drawn, SingleStars.Z_birth, SingleStars.rho_birth})

KOKKOS_INLINE_FUNCTION double single_star_band_lum_cgs(double log_l) {double l = pow(10., log_l); return (isfinite(l) && l > 0) ? l : 0;}

/* true time since birth [yr] (single_star_age_yr): the cosmology core on the device (a device unit opens it with
   GRAVTREE_SOURCE_DEVICE_TU, gravity/cosmology_functions.h), the host function, which calls the same core, on the host.
   Without GRAVTREE_SOURCE_LAZY_SUPPORTED (GR_TABULATED_COSMOLOGY*) the age reads host-only tables and no device code calls
   these lookups: the host function then serves both passes */
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(GRAVTREE_SOURCE_DEVICE_TU) && defined(GRAVTREE_SOURCE_LAZY_SUPPORTED)
#define SINGLE_STAR_TIME_SINCE_GYR(t) evaluate_time_since_t_initial_in_Gyr_core(t)
#else
#define SINGLE_STAR_TIME_SINCE_GYR(t) evaluate_time_since_t_initial_in_Gyr(t)
#endif
KOKKOS_INLINE_FUNCTION double single_star_age_yr_core(const struct particle_data *pp, int i) {return DMAX(SINGLE_STAR_TIME_SINCE_GYR(pp[i].StellarAge), 0.) * 1.e9;}

/* age [yr] at which a star of 8 Msun or more starts to ionise: the end of its pre-main sequence, at least the shorter of
   1.5e7 M^-2.1 yr and the free-fall time of its birth cloud, at most 1 Myr; -1 below 8 Msun or without a record */
KOKKOS_INLINE_FUNCTION double single_star_ionizing_onset_yr_core(const StellarTablesView &T, const SingleStarRecordView &R, const struct particle_data *pp, int i)
{
    if(pp[i].StarIdx <= 0) {return -1;}
    const int s = pp[i].StarIdx - 1;
    double logM = log10(R.M_drawn[s]), logZ = log10(DMAX(R.Z_birth[s], 1e-10));
    if(logM < log10(8.0)) {return -1;}
    double t_on = stbl_interp2d(T, T.t_PMS, logM, logZ) * (1. + 1.e-12);
    if(R.rho_birth[s] > 0) {
        double t_ff = sqrt(3. * M_PI / (32. * GRAVITY_G_CGS * R.rho_birth[s] * UNIT_DENSITY_IN_CGS)) / SECONDS_PER_YEAR;
        t_on = DMAX(t_on, DMIN(DMIN(1.5e7 * pow(10., -2.1 * logM), t_ff), 1.e6));
    }
    return t_on;
}

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS) || defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS)

/* L[SINGLE_STAR_BAND_*] [erg/s] and the bolometric luminosity; returns 0 (all zero) before the main sequence, after death,
   without a record or at or below IMFSampleStellarMassCut. 6-8 eV = L(6-13.6) - L(8-13.6), unclipped */
KOKKOS_INLINE_FUNCTION int single_star_band_luminosities_core(const StellarTablesView &T, const SingleStarRecordView &R, const struct particle_data *pp, int i, double *L, double *L_bol)
{
    for(int b = 0; b < SINGLE_STAR_NBANDS; b++) {L[b] = 0;}
    *L_bol = 0;
    if(pp[i].StarIdx <= 0) {return 0;}
    const int s = pp[i].StarIdx - 1; /* single_star_slot */
    if(R.M_drawn[s] <= All.IMFSampleStellarMassCut) {return 0;}
#ifdef TEST_SINGLE_STARS_LUMINOSITY_PINNED /* test only: constant L from birth, FUV 0.5 (in the 8-13.6 eV band), LW 0.15, NUV 0.2, optical 0.3, no ionising photons */
    L[SINGLE_STAR_BAND_PE] = 0.5 * All.TestFixedLbol; L[SINGLE_STAR_BAND_LW] = 0.15 * All.TestFixedLbol; L[SINGLE_STAR_BAND_NUV] = 0.2 * All.TestFixedLbol;
    L[SINGLE_STAR_BAND_OPT] = 0.3 * All.TestFixedLbol; *L_bol = All.TestFixedLbol;
    return 1;
#endif
#ifdef TEST_SINGLE_STARS_RADIATION_OFF
    return 0; /* test only: no radiation */
#endif
    double logM = log10(R.M_drawn[s]), logZ = log10(DMAX(R.Z_birth[s], 1e-10));
    double age_yr = single_star_age_yr_core(pp, i), t_pms = stbl_interp2d(T, T.t_PMS, logM, logZ);
    if(age_yr >= t_pms + stbl_interp2d(T, T.lifetime_yr, logM, logZ) || age_yr - t_pms <= 0) {return 0;}
    double la = log10(DMAX(age_yr - t_pms, 100.));
    double L_uv = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_FUV_total, logM, logZ, la)); /* 6-13.6 eV */
    L[SINGLE_STAR_BAND_PE] = (L_uv > 0) ? single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_FUV_M1, logM, logZ, la)) : 0;
    L[SINGLE_STAR_BAND_68] = L_uv - L[SINGLE_STAR_BAND_PE];
    L[SINGLE_STAR_BAND_LW] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_LW, logM, logZ, la));
    L[SINGLE_STAR_BAND_NUV] = DMAX(0., single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_NUV, logM, logZ, la)) - L[SINGLE_STAR_BAND_68]); /* 3.444-6 eV */
    L[SINGLE_STAR_BAND_OPT] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_OPT_NIR, logM, logZ, la));
    *L_bol = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_bol, logM, logZ, la));
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
    L[SINGLE_STAR_BAND_ION_H0] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_H0, logM, logZ, la));
    L[SINGLE_STAR_BAND_ION_HE0] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_He0, logM, logZ, la));
    L[SINGLE_STAR_BAND_ION_HE1] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_He1, logM, logZ, la));
    L[SINGLE_STAR_BAND_ION_HE2] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_He2, logM, logZ, la));
    L[SINGLE_STAR_BAND_Q] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_Q_ion, logM, logZ, la));
#endif
    return 1;
}

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
/* ionising energy L_b [erg/s] and photon rate Q_b [1/s] per band of a shining star from the tables at its current age. With
   helium the bands start at 13.6, 24.6, 54.4 and 70 eV: energies from the table's bands, photons above 54.4 eV at the mean
   energies of a 4e4 K blackbody, photons in 24.6-54.4 eV as the difference of two log tables, kept inside the band where that
   difference is rounding; energies and photons add up to the star's */
KOKKOS_INLINE_FUNCTION void single_star_ionizing_rates_core(const StellarTablesView &T, const SingleStarRecordView &R, const struct particle_data *pp, int i, double *L_b, double *Q_b)
{
    const int s = pp[i].StarIdx - 1;
    double logM = log10(R.M_drawn[s]), logZ = log10(DMAX(R.Z_birth[s], 1e-10));
    double la = log10(DMAX(single_star_age_yr_core(pp, i) - stbl_interp2d(T, T.t_PMS, logM, logZ), 100.));
#if defined(TEST_SINGLE_STARS_LUMINOSITY_PINNED) || defined(TEST_SINGLE_STARS_RADIATION_OFF)
    for(int b = 0; b < TREECOL_SHELLS_ION_NBANDS; b++) {L_b[b] = Q_b[b] = 0;} (void) la; return; /* test only */
#endif
    double L = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_tot, logM, logZ, la)), Q = single_star_band_lum_cgs(stbl_interp3d(T, T.log_Q_ion, logM, logZ, la));
#ifdef TREECOL_SHELLS_ION_HELIUM
    L_b[1] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_He0, logM, logZ, la));
    L_b[2] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_He1, logM, logZ, la)); Q_b[2] = L_b[2] / (58.006 * ELECTRONVOLT_IN_ERGS);
    L_b[3] = single_star_band_lum_cgs(stbl_interp3d(T, T.log_L_ion_He2, logM, logZ, la)); Q_b[3] = L_b[3] / (73.785 * ELECTRONVOLT_IN_ERGS);
    Q_b[1] = DMIN(DMAX(Q - single_star_band_lum_cgs(stbl_interp3d(T, T.log_Q_ion_H0, logM, logZ, la)) - Q_b[2] - Q_b[3], L_b[1] / (54.4 * ELECTRONVOLT_IN_ERGS)), L_b[1] / (24.6 * ELECTRONVOLT_IN_ERGS));
    L_b[0] = DMAX(L - L_b[1] - L_b[2] - L_b[3], 0); Q_b[0] = DMAX(Q - Q_b[1] - Q_b[2] - Q_b[3], 0);
#else
    L_b[0] = L; Q_b[0] = Q;
#endif
#ifdef TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestTreeRayFixedRate > 0)
    {
#ifdef TREECOL_SHELLS_ION_HELIUM
        for(int b = 0; b < TREECOL_SHELLS_ION_NBANDS; b++) {L_b[b] *= All.TestTreeRayFixedRate / Q; Q_b[b] *= All.TestTreeRayFixedRate / Q;} /* the star's spectrum */
#else
        Q_b[0] = All.TestTreeRayFixedRate; L_b[0] = Q_b[0] * 27.2 * ELECTRONVOLT_IN_ERGS;
#endif
    }
#endif
}
#endif

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS)
/* as rt_get_source_luminosity: fills lum[] [code units], returns 1 for a shining star */
KOKKOS_INLINE_FUNCTION int single_star_rt_band_luminosities_core(const StellarTablesView &T, const SingleStarRecordView &R, int i, int mode, double *lum, const struct particle_data *pp)
{
    double L[SINGLE_STAR_NBANDS], L_bol;
#ifdef RT_STARBENCH_TEST
    if(mode >= 0) {lum[RT_FREQ_BIN_H0] = 1e49 * All.rt_nu_eff_eV[RT_FREQ_BIN_H0] * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS * (C_LIGHT_CODE_REDUCED / C_LIGHT_CODE);}
    return 1;
#endif
    if(!single_star_band_luminosities_core(T, R, pp, i, L, &L_bol)) {return 0;}
    if(mode < 0) {return 1;}
    lum[RT_FREQ_BIN_PHOTOELECTRIC] = L[SINGLE_STAR_BAND_PE] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_LYMAN_WERNER] = L[SINGLE_STAR_BAND_LW] / UNIT_LUM_IN_CGS;
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
    lum[RT_FREQ_BIN_NUV] = DMAX(0., L[SINGLE_STAR_BAND_68]) / UNIT_LUM_IN_CGS; /* 6-8 eV */
#ifdef RT_OPTICAL_NIR
    lum[RT_FREQ_BIN_OPTICAL_NIR] = L[SINGLE_STAR_BAND_OPT] / UNIT_LUM_IN_CGS;
#endif
#else
    lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = L[SINGLE_STAR_BAND_68] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_NUV] = L[SINGLE_STAR_BAND_NUV] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_OPTICAL_NIR] = L[SINGLE_STAR_BAND_OPT] / UNIT_LUM_IN_CGS;
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
    double L_ion[TREECOL_SHELLS_ION_NBANDS], Q_ion[TREECOL_SHELLS_ION_NBANDS]; single_star_ionizing_rates_core(T, R, pp, i, L_ion, Q_ion);
    for(int b = 0; b < TREECOL_SHELLS_ION_NBANDS; b++) {
        lum[RT_FREQ_BIN_ION_ENERGY + 2 * b] = L_ion[b] / UNIT_LUM_IN_CGS;
        lum[RT_FREQ_BIN_ION_PHOTONS + 2 * b] = Q_ion[b] * 13.6 * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS;
    }
#endif
#endif
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
#ifdef RT_CHEM_PHOTOION
#ifdef RT_PHOTOION_MULTIFREQUENCY
    lum[RT_FREQ_BIN_H0] = L[SINGLE_STAR_BAND_ION_H0] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_He0] = L[SINGLE_STAR_BAND_ION_HE0] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_He1] = L[SINGLE_STAR_BAND_ION_HE1] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_He2] = L[SINGLE_STAR_BAND_ION_HE2] / UNIT_LUM_IN_CGS;
#else
    lum[RT_FREQ_BIN_H0] = L[SINGLE_STAR_BAND_Q] * All.rt_nu_eff_eV[RT_FREQ_BIN_H0] * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS; /* the table's photons at the bin's photon energy */
#endif
#endif
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {lum[k] *= (C_LIGHT_CODE_REDUCED / C_LIGHT_CODE);} /* as rt_get_source_luminosity */
#endif
    return 1;
}
#endif

#endif /* band fluxes */
#endif /* tables */
#endif /* SINGLE_STAR_LUMINOSITY_FUNCTIONS_H */
