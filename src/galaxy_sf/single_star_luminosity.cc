/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Band luminosities of IMF-sampled single stars from the stellar-evolution tables at the star's current age, in the
   disjoint bands PE 8-13.6, LW 11.2-13.6 (inside PE), 6-8, NUV 3.444-6, OPT 0.4133-3.444 eV: as gravity-tree radiation
   sources (GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) and for the radiation-pressure budget at the star
   (GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS). Host only. */
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "single_star_record.h"
#include "stellar_evolution_tables.h"
#ifdef SGCHEM
#include "../cooling/sgchem/sgchem_dust_to_gas.h"
#endif

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS)

static inline double single_star_band_lum_cgs(double log_l) {double l = pow(10., log_l); return (isfinite(l) && l > 0) ? l : 0;}

/* L[SINGLE_STAR_BAND_*] [erg/s] and the bolometric luminosity; returns 0 (all zero) before the main sequence, after death
   or without a record. 6-8 eV = L(6-13.6) - L(8-13.6), unclipped as stella_dev */
int single_star_band_luminosities_cgs(int i, double *L, double *L_bol)
{
    for(int b = 0; b < SINGLE_STAR_NBANDS; b++) {L[b] = 0;}
    *L_bol = 0;
    if(P[i].StarIdx <= 0) {return 0;}
    int s = single_star_slot(P[i].StarIdx);
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double age_yr = evaluate_stellar_age_Gyr(i) * 1.e9, t_pms = stellar_t_PMS(logM, logZ);
    if(age_yr >= t_pms + stellar_lifetime(logM, logZ) || age_yr - t_pms <= 0) {return 0;}
    double la = log10(DMAX(age_yr - t_pms, 100.));
    double L_uv = single_star_band_lum_cgs(stellar_log_L_FUV_total(logM, logZ, la)); /* 6-13.6 eV */
    L[SINGLE_STAR_BAND_PE] = (L_uv > 0) ? single_star_band_lum_cgs(stellar_log_L_FUV_M1(logM, logZ, la)) : 0;
    L[SINGLE_STAR_BAND_68] = L_uv - L[SINGLE_STAR_BAND_PE];
    L[SINGLE_STAR_BAND_LW] = single_star_band_lum_cgs(stellar_log_L_LW(logM, logZ, la));
    L[SINGLE_STAR_BAND_NUV] = DMAX(0., single_star_band_lum_cgs(stellar_log_L_NUV(logM, logZ, la)) - L[SINGLE_STAR_BAND_68]); /* 3.444-6 eV */
    L[SINGLE_STAR_BAND_OPT] = single_star_band_lum_cgs(stellar_log_L_OPT_NIR(logM, logZ, la));
    *L_bol = single_star_band_lum_cgs(stellar_log_L_bol(logM, logZ, la));
    return 1;
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
/* as rt_get_source_luminosity: fills lum[] [code units], returns 1 for a shining star (pp is P here) */
int single_star_rt_band_luminosities(int i, int mode, double *lum, struct particle_data *pp)
{
    double L[SINGLE_STAR_NBANDS], L_bol; (void) pp;
    if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return 0;}
    if(mode < 0) {return 1;}
    lum[RT_FREQ_BIN_PHOTOELECTRIC] = L[SINGLE_STAR_BAND_PE] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_LYMAN_WERNER] = L[SINGLE_STAR_BAND_LW] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = L[SINGLE_STAR_BAND_68] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_NUV] = L[SINGLE_STAR_BAND_NUV] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_OPTICAL_NIR] = L[SINGLE_STAR_BAND_OPT] / UNIT_LUM_IN_CGS;
    return 1;
}
#endif

#ifdef GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS
/* photon momentum a single star hands to the gas this step [g cm/s] (stella_dev estimator): band luminosities absorbed
   over the kernel column at the star, f_b = 1 - exp(-sigma_b DGR N_H) with sigma = 2.0 (PE + 6-8), 1.3 (NUV), 0.5 (OPT)
   x 1e-21 cm^2, LW not counted (inside PE), capped at L_bol; plus the IR boost tau_IR = 5 DGR Sigma [cm^2/g] on the
   absorbed light. GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE adds 6-8 eV to the NUV term as stella_dev production */
double single_star_radpressure_momentum_cgs(int i, double dt_cgs, double *tau_IR)
{
    double L[SINGLE_STAR_NBANDS], L_bol;
    *tau_IR = 0;
    if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return 0;}
#ifdef SGCHEM
    double dgr = DMAX(sgchem_dust_to_gas_ratio(P[i].Metallicity[0]), 0.);
#else
    double dgr = DMAX(DMIN(P[i].Metallicity[0] / All.SolarAbundances[0], 10.), 0.);
#endif
    double rho = P[i].DensityAroundParticle * All.cf_a3inv, h = P[i].KernelRadius * All.cf_atime;
    double N_H = rho * UNIT_DENSITY_IN_NHCGS * h * UNIT_LENGTH_IN_CGS, Sigma = rho * UNIT_DENSITY_IN_CGS * h * UNIT_LENGTH_IN_CGS;
    double f_uv = 1.0 - exp(-2.0e-21 * dgr * N_H), f_nuv = 1.0 - exp(-1.3e-21 * dgr * N_H), f_opt = 1.0 - exp(-0.5e-21 * dgr * N_H);
    double L_nuv = L[SINGLE_STAR_BAND_NUV];
#ifdef GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
    L_nuv += L[SINGLE_STAR_BAND_68];
#endif
    double L_abs = DMIN(f_uv * (L[SINGLE_STAR_BAND_PE] + L[SINGLE_STAR_BAND_68]) + f_nuv * L_nuv + f_opt * L[SINGLE_STAR_BAND_OPT], L_bol);
    *tau_IR = 5.0 * dgr * Sigma;
    return L_abs * (1.0 + *tau_IR) * dt_cgs / C_LIGHT_CGS;
}
#endif
