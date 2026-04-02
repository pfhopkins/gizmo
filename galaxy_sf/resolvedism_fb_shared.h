#ifndef RESOLVEDISM_FB_SHARED_H
#define RESOLVEDISM_FB_SHARED_H

/* Shared helpers, constants, and extern declarations for the split
 * feedback routines: resolvedism_fb_thermal.cc and resolvedism_fb_momentum.cc.
 * The orchestrator (resolvedism_fb.cc) owns the global accumulators. */

#ifdef GALSF_RESOLVEDISM_FB

/* =========================================================================== */
/*  Type Ia DTD constants (Maoz & Mannucci 2012, t^{-1} power-law DTD)        */
/* =========================================================================== */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
#define IA_T_MIN_GYR     0.04    /* 40 Myr minimum delay time [Gyr] */
#define IA_N_PER_MSUN    1.3e-3  /* total Ia per Msun formed */
#define IA_F_WD          0.76    /* mass fraction in WD progenitors (Kroupa IMF) */
#define IA_DTD_NORM      2.93e-4 /* [Gyr^{-1} Msun^{-1}] rate constant */
#define IA_EJECTA_MASS   1.378   /* total ejecta mass [Msun] (Chandrasekhar mass) */
#define IA_ENERGY_ERG    1.0e51  /* canonical Type Ia energy [erg] */
#endif

/* =========================================================================== */
/*  Global accumulators (defined in resolvedism_fb.cc)                         */
/* =========================================================================== */
extern double CumulFeedbackEnergy;
extern double CumulFeedbackMass;
extern double CumulFeedbackMetals;
extern int CumulSNe, CumulAGB, CumulIa;
extern double CumulStarMassFormed;
extern double RadPressure_dp_thisStep;

/* =========================================================================== */
/*  Shared helper functions                                                    */
/* =========================================================================== */

/* Get stellar mass and metallicity for the single star in particle i */
static inline int get_star_info(int i, double *Mstar_out, double *logM_out, double *logZ_out)
{
    double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
    Mstar = P[i].Mstar;
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(P[i].sampled) Mstar = P[i].MstarSampleIMF[0];
#endif
    if(Mstar <= 0) return 0;
    *Mstar_out = Mstar;
    *logM_out = log10(Mstar);
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    *logZ_out = log10(DMAX(P[i].BirthMetallicity, 1e-10));
#else
    *logZ_out = log10(0.014); /* solar default */
#endif
    return 1;
}

/* Get total lifetime for a star (birth to death, includes PMS) */
static inline double get_star_lifetime(double Mstar, double logM, double logZ)
{
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    return stellar_t_PMS(logM, logZ) + stellar_lifetime(logM, logZ);
#else
    return get_lifetime(Mstar, pow(10.0, logZ));
#endif
}

/* Get table-effective age: star_age minus PMS duration.
 * Table age axis starts at ZAMS. Returns <= 0 during PMS (no feedback). */
static inline double get_star_table_age(double star_age_yr, double logM, double logZ)
{
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    return star_age_yr - stellar_t_PMS(logM, logZ);
#else
    return star_age_yr;
#endif
}

/* =========================================================================== */
/*  Function prototypes for the two tree walk routines                         */
/* =========================================================================== */
void resolvedism_fb_thermal_calc(void);
void resolvedism_fb_momentum_calc(int fb_loop_iteration);

#endif /* GALSF_RESOLVEDISM_FB */
#endif /* RESOLVEDISM_FB_SHARED_H */
