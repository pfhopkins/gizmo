/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
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
#define IA_DTD_NORM      2.93e-4 /* [Msun^{-1}] DTD constant: rate = NORM/t */
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

/* per-event serialization token: one star, keyed by its unique (ID, child number, generation) */
static inline int resolvedism_fb_token_admits(int i) {return FB_SerialEventID == 0 || (P[i].ID == FB_SerialEventID && P[i].ID_child_number == FB_SerialEventChild && (long long)P[i].ID_generation == FB_SerialEventGen);}

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

#ifdef GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION
#if !defined(GALSF_RESOLVEDISM_STELLAR_TABLES) || !defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL)
#error "GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION needs GALSF_RESOLVEDISM_STELLAR_TABLES and GALSF_RESOLVEDISM_METALS_INDIVIDUAL"
#endif
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
#error "GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION does not reach the SN_SPAWN payload (SpawnEjZ): not combinable yet"
#endif
extern double CumulEjectaBirthClip; /* ejecta mass clipped from slots below zero by the birth-composition swap [Msun] */
/* net-yield form (STL-403): the table's unprocessed share carries the star's birth composition instead of the table's X_init,
 * m_k += M_ej (X_star,k - X_init,k(Z)); slots below zero are set to zero and the sum restored by one common factor.
 * Returns the clipped mass (same units as m_k). */
static inline double resolvedism_ejecta_birth_composition(int i, double logZ, double *m_k)
{
    double Mej = 0, sum = 0, clip = 0; int k;
    for(k = 0; k < STBL_NELEM; k++) {Mej += m_k[k];}
    if(!(Mej > 0)) {return 0;}
    for(k = 0; k < STBL_NELEM; k++) {
        m_k[k] += Mej * (P[i].Metallicity[MET_OF(k)] - stellar_X_init(logZ, k));
        if(m_k[k] < 0) {clip -= m_k[k]; m_k[k] = 0;}
        sum += m_k[k];
    }
    if(sum > 0) {double f = Mej / sum; for(k = 0; k < STBL_NELEM; k++) {m_k[k] *= f;}}
    return clip;
}
#endif

#ifdef COSMIC_RAY_FLUID
#ifdef GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE
#if (N_CR_PARTICLE_BINS > 1)
#error "GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE divides out the injection's reduced-speed factor of a single CR bin"
#endif
#endif
/* CR energy dEcr into cell j at a Stella deposit; with GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE the cell receives all of it (STL-356):
 * inject_cosmic_rays scales an injection by the reduced-speed factor, which is divided out here (transport and losses keep it) */
static inline void resolvedism_inject_cr(double dEcr, double v_inj, int j, double *dir)
{
#ifdef GALSF_RESOLVEDISM_CR_INJECT_FULL_SHARE
    double r = evaluate_cr_transport_reductionfactor(j, 0, 0, CellP);
    if(r > 0 && r < 1) {dEcr /= r;}
#endif
    inject_cosmic_rays(dEcr, v_inj, 0, j, dir, CellP, -1);
}
#endif

/* =========================================================================== */
/*  Coupled mode: Stella yields collapsed onto the FIRE metal slots            */
/* =========================================================================== */
#if defined(GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL) && defined(METALS)
#include <float.h>
#include "resolvedism_selfcheck.h"
#if (NUM_LIVE_SPECIES_FOR_COOLTABLES != 10)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL packs Stella yields into the FIRE slots Z,He,C,N,O,Ne,Mg,Si,S,Ca,Fe: requires COOL_METAL_LINES_BY_SPECIES"
#endif
#define STELLA_FIRE_YIELD_SLOTS 11 /* FIRE slots 0=Z 1=He 2=C 3=N 4=O 5=Ne 6=Mg 7=Si 8=S 9=Ca 10=Fe (init.cc SolarAbundances) */
static const int stella_fire_slot_elem[STELLA_FIRE_YIELD_SLOTS] = {-1, ELEM_He, ELEM_C, ELEM_N, ELEM_O, ELEM_Ne, ELEM_Mg, ELEM_Si, ELEM_S, ELEM_Ca, ELEM_Fe}; /* table element per slot; slot 0 = sum of all table metals */

/* pack code-mass table yields y[STBL_NELEM] into fy[]; returns the total metal yield (= fy[0]) */
static inline double stella_pack_fire_yields(const double *y, MyDouble *fy, MyIDType id)
{
    int k, s; double z = 0, zsub = 0;
    for(k = 0; k < STBL_NELEM; k++) {if(!(y[k] >= 0)) {printf("STELLA FIRE yields: star=%llu element %d yield %.17g < 0\n", (unsigned long long)id, k, y[k]); endrun(8461);}}
    for(k = ELEM_C; k < STBL_NELEM; k++) {z += y[k];}
    fy[0] = z;
    for(s = 1; s < STELLA_FIRE_YIELD_SLOTS; s++) {fy[s] = y[stella_fire_slot_elem[s]];}
    for(s = 2; s < STELLA_FIRE_YIELD_SLOTS; s++) {zsub += fy[s];} /* same order as z, all terms >= 0: zsub <= z exactly */
    if(!(zsub <= z)) {printf("STELLA FIRE yields: star=%llu sum(C..Fe)=%.17g > Z=%.17g\n", (unsigned long long)id, zsub, z); endrun(8462);}
    return z;
}

/* mass-fraction blend of slots 1..NUM_METAL_SPECIES-1 for ejecta dM; tracer slots get no yield, only dilution.
 * c: mass of the cell's own gas turned into ejecta composition in place (Ia companion share), of total ejecta Mej;
 * the converted gas keeps no tracer either (age or other tracer slots lose c X_old) */
static inline void stella_blend_fire_slots(int j, double M_old, double dM, double wk, const MyDouble *fy, double c, double Mej)
{
    int s;
    for(s = 1; s < NUM_METAL_SPECIES; s++) {
        double X_old;
        #pragma omp atomic read
        X_old = P[j].Metallicity[s];
        double dMX = ((s < STELLA_FIRE_YIELD_SLOTS) ? (wk + ((c > 0) ? c / Mej : 0)) * fy[s] : 0) - c * X_old;
        double dX = (dMX - X_old * dM) / (M_old + dM);
        #pragma omp atomic
        P[j].Metallicity[s] += dX;
    }
}

#ifdef STELLA_SELFCHECK
static inline void stella_fire_yields_log(const char *ch, MyIDType id, const double *y, const MyDouble *fy)
{
    int k; printf("STELLA_FIRE_YIELDS ch=%s ID=%llu t=%.17g table", ch, (unsigned long long)id, All.Time);
    for(k = 0; k < STBL_NELEM; k++) {printf(" %.17g", y[k]);}
    printf(" slots"); for(k = 0; k < STELLA_FIRE_YIELD_SLOTS; k++) {printf(" %.17g", fy[k]);}
    printf("\n"); fflush(stdout);
}
/* receiver check: slot s gains exactly (wk + c/Mej) fy[s] - c X_old[s] in mass; Z - sum(C..Fe) >= 0 is preserved */
static inline void stella_fire_receiver_check(int j, double M_old, double dM, double wk, const MyDouble *fy, const double *X_old, MyIDType donor, double c, double Mej)
{
    int s; double M_new = M_old + dM, sub_old = 0, sub_new = 0;
    for(s = 0; s < STELLA_FIRE_YIELD_SLOTS; s++) {
        double add = (wk + ((c > 0) ? c / Mej : 0)) * fy[s];
        double X_new = P[j].Metallicity[s], dm_s = M_new * X_new - M_old * X_old[s], want = add - c * X_old[s];
        double tol = 8 * DBL_EPSILON * (M_old * X_old[s] + add + c * X_old[s] + M_new * X_new); /* bound of the blend + this difference */
        STELLA_CHECK("fire_slot_deposit", fabs(dm_s - want) <= tol, "donor=%llu cell=%llu slot=%d dm=%.17g want=%.17g tol=%.3g", (unsigned long long)donor, (unsigned long long)P[j].ID, s, dm_s, want, tol);
        if(s >= 2) {sub_old += X_old[s]; sub_new += X_new;}
    }
    STELLA_CHECK("fire_slot_invariant_cell", !(sub_old <= X_old[0]) || (sub_new <= P[j].Metallicity[0]), "donor=%llu cell=%llu Z=%.17g sum(C..Fe)=%.17g (before %.17g %.17g)", (unsigned long long)donor, (unsigned long long)P[j].ID, P[j].Metallicity[0], sub_new, X_old[0], sub_old);
}
#endif
#endif

#if defined(RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED) || defined(TEST_KICK_FLOOR_DUMP)
/* a deposit into cell j: with RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED its signal speed is raised to the post-injection value, so the
   timestep criterion and an own-timestep wake see the heated state (dv_phys: the deposit's velocity change, physical) */
static inline void resolvedism_fb_receiver_deposit_done(int j, double dv_phys, int site)
{
    (void) dv_phys; (void) site;
#ifdef RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED
    {double u_j;
    #pragma omp atomic read
    u_j = CellP[j].InternalEnergy;
    double vsig_new = 2. * sqrt(CellP[j].soundspeed2_from_u(u_j)) + dv_phys;
    #pragma omp critical(resolvedism_fb_receiver_vsig)
    {if(vsig_new > CellP[j].MaxSignalVel) {CellP[j].MaxSignalVel = vsig_new;}}}
#endif
#ifdef TEST_KICK_FLOOR_DUMP
    printf("TEST_KICK_FLOOR_DUMP recv step %lld t %.10e ID %llu site %d bin %d active %d u %.6e vsig %.6e h %.6e m %.6e\n", (long long) All.NumCurrentTiStep, All.Time,
           (unsigned long long) P[j].ID, site, (int) P[j].TimeBin, (int) TimeBinActive[P[j].TimeBin], (double) CellP[j].InternalEnergy, (double) CellP[j].MaxSignalVel, (double) P[j].KernelRadius, (double) P[j].Mass);
#endif
}
#endif

/* =========================================================================== */
/*  Function prototypes for the two tree walk routines                         */
/* =========================================================================== */
void resolvedism_fb_thermal_calc(void);
void resolvedism_fb_momentum_calc(int fb_loop_iteration);

#endif /* GALSF_RESOLVEDISM_FB */
#endif /* RESOLVEDISM_FB_SHARED_H */
