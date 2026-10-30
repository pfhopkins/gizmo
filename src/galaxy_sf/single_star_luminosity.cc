/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Band luminosities of IMF-sampled single stars for the gravity-tree source arrays (GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES):
   the stellar-evolution tables at the star's current age, in the disjoint bands PE 8-13.6, LW 11.2-13.6 (inside PE), 6-8, NUV 3.444-6, OPT 0.4133-3.444 eV. */
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "single_star_record.h"
#include "stellar_evolution_tables.h"

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES

static inline double single_star_band_lum_cgs(double log_l) {double l = pow(10., log_l); return (isfinite(l) && l > 0) ? l : 0;}

/* as rt_get_source_luminosity: fills lum[] [code units], returns 1 for a shining star; pre-main-sequence and dead stars are dark */
int single_star_rt_band_luminosities(int i, int mode, double *lum, struct particle_data *pp)
{
    if(pp[i].StarIdx <= 0) {return 0;}
    int s = single_star_slot(pp[i].StarIdx);
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double age_yr = evaluate_stellar_age_Gyr(i) * 1.e9 /* host particle arrays (pp is P here) */, t_pms = stellar_t_PMS(logM, logZ);
    if(age_yr >= t_pms + stellar_lifetime(logM, logZ) || age_yr - t_pms <= 0) {return 0;}
    if(mode < 0) {return 1;}
    double la = log10(DMAX(age_yr - t_pms, 100.));
    double L_uv = single_star_band_lum_cgs(stellar_log_L_FUV_total(logM, logZ, la)); /* 6-13.6 eV */
    double L_pe = (L_uv > 0) ? single_star_band_lum_cgs(stellar_log_L_FUV_M1(logM, logZ, la)) : 0; /* 8-13.6 eV */
    double L_68 = L_uv - L_pe; /* 6-8 eV, unclipped as stella_dev */
    double L_nuv = single_star_band_lum_cgs(stellar_log_L_NUV(logM, logZ, la)); /* 3.444-8 eV */
    lum[RT_FREQ_BIN_PHOTOELECTRIC] = L_pe / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_LYMAN_WERNER] = single_star_band_lum_cgs(stellar_log_L_LW(logM, logZ, la)) / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = L_68 / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_NUV] = DMAX(0., L_nuv - L_68) / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_OPTICAL_NIR] = single_star_band_lum_cgs(stellar_log_L_OPT_NIR(logM, logZ, la)) / UNIT_LUM_IN_CGS;
    return 1;
}

#endif
