/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "resolvedism_sn_energy.h"
#include "../core/proto.h"
#include "resolvedism_star_age.h"
#include "../mesh/kernel.h"
#include "resolvedism_selfcheck.h"
#include "resolvedism_fb_shared.h"
#include "resolvedism_rng.h"
#include <vector>

#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
/* forced-SN test star k: key (k, 1) from the site hook (sfr_eff.cc forms only from a generation-0 site, child 2^0) or (k, 0) in the ICs */
static inline int resolvedism_is_forced_sn_test_star(int i)
{
    return P[i].ID >= 1 && P[i].ID <= (MyIDType)((All.TestNumStars > 0) ? All.TestNumStars : 1) && P[i].ID_child_number <= 1;
}
#endif

/* Death-event log rows (SNinfo/AGBinfo), written when the outcome is known (booking), so failed
 * and no-gas events appear once each with their real (possibly zero) energy and mass return. */
struct fb_death_row {double t, x, y, z, u, rho, mstar, mej, zej, yC, yO, ySi, yFe, age, life, E, nH, T, h_pc, nngb, nH_birth, nH_stale; long long id; int remtype, tb, status, agb;};
static std::vector<fb_death_row> FbDeathRows;
#define FB_LOG_EXPLODED 0  /* exploded/ejected and deposited */
#define FB_LOG_FAILED   1  /* failed SN (FSN/DBH): no energy, no ejecta */
#define FB_LOG_NOGAS    2  /* attempt reached no gas: nothing booked, retried later */

static void fb_death_row_push(int i, int agb, int remtype, int status, double E, const double y[6], double age, double life)
{
    fb_death_row r; r.t = All.Time; r.x = P[i].Pos[0]; r.y = P[i].Pos[1]; r.z = P[i].Pos[2];
    r.u = P[i].InternalEnergyAroundParticle; r.rho = P[i].DensityAroundParticle; r.mstar = P[i].MstarSampleIMF[0];
    r.mej = y[0]; r.zej = y[1]; r.yC = y[2]; r.yO = y[3]; r.ySi = y[4]; r.yFe = y[5]; r.age = age; r.life = life; r.E = E;
    { /* ambient state now, birth density, birth-frozen density */
        const double nH_per_rho = HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
        r.nH = P[i].DensityAroundParticle * All.cf_a3inv * nH_per_rho;
        r.T = P[i].InternalEnergyAroundParticle * UNIT_SPECEGY_IN_CGS * (GAMMA_DEFAULT - 1.) * 1.22 * PROTONMASS_CGS / BOLTZMANN_CGS; /* neutral mu=1.22 */
        r.h_pc = P[i].KernelRadius * All.cf_atime * UNIT_LENGTH_IN_PC;
        r.nngb = P[i].NumNgb;
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF)
        r.nH_birth = P[i].FormationDensity * nH_per_rho;
#else
        r.nH_birth = 0;
#endif
        double af = (All.ComovingIntegrationOn && P[i].StellarAge > 0) ? P[i].StellarAge / All.Time : 1.;
        r.nH_stale = r.nH_birth * af * af * af;
    }
    r.id = (long long)P[i].ID; r.remtype = remtype; r.tb = P[i].TimeBin; r.status = status; r.agb = agb;
    FbDeathRows.push_back(r);
}

static void fb_death_rows_flush(void)
{
    int n_loc = (int)FbDeathRows.size(), n_tot = 0;
    MPI_Allreduce(&n_loc, &n_tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(n_tot == 0) {FbDeathRows.clear(); return;}
    int nb = n_loc * (int)sizeof(fb_death_row), *cnt = NULL, *dsp = NULL; std::vector<fb_death_row> all;
    if(ThisTask == 0) {cnt = (int *)malloc(NTask * sizeof(int)); dsp = (int *)malloc(NTask * sizeof(int)); all.resize(n_tot);}
    MPI_Gather(&nb, 1, MPI_INT, cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {dsp[0] = 0; for(int t = 1; t < NTask; t++) dsp[t] = dsp[t-1] + cnt[t-1];}
    MPI_Gatherv(FbDeathRows.data(), nb, MPI_BYTE, ThisTask == 0 ? (void *)all.data() : NULL, cnt, dsp, MPI_BYTE, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        for(int k = 0; k < n_tot; k++) {
            const fb_death_row &r = all[k];
            if(r.agb) fprintf(FdAGBinfo, "%12.6f  %10.6f %10.6f %10.6f  %10.3f %12.6e  %10lld  %8.3f  %8.3f  %8.3f  %8.4f %8.4f %8.4f %8.4f  %.6e  %.6e  %3d  %d  %.4e %.4e %.4e %7.2f %.4e %.4e\n",
                              r.t, r.x, r.y, r.z, r.u, r.rho, r.id, r.mstar, r.mej, r.zej, r.yC, r.yO, r.ySi, r.yFe, r.age, r.life, r.tb, r.status, r.nH, r.T, r.h_pc, r.nngb, r.nH_birth, r.nH_stale);
            else fprintf(FdSNinfo, "%12.6f  %10.6f %10.6f %10.6f  %10.3f %12.6e  %10lld  %8.3f  %d  %8.3f  %8.3f  %8.4f %8.4f %8.4f %8.4f  %.6e  %.6e  %3d  %.4e  %d  %.4e %.4e %.4e %7.2f %.4e %.4e\n",
                              r.t, r.x, r.y, r.z, r.u, r.rho, r.id, r.mstar, r.remtype, r.mej, r.zej, r.yC, r.yO, r.ySi, r.yFe, r.age, r.life, r.tb, r.E, r.status, r.nH, r.T, r.h_pc, r.nngb, r.nH_birth, r.nH_stale);
        }
        fflush(FdSNinfo); fflush(FdAGBinfo); free(dsp); free(cnt);
    }
    FbDeathRows.clear();
}

/* Resolved-ISM supernova feedback for individually sampled single stars.
 * Each star particle represents ONE star drawn from the Kroupa IMF.
 * When a massive star (>=8 Msun) reaches the end of its lifetime, it
 * explodes as a SN (ECSN/CCSN/PISN/PPISN) or collapses directly (FSN/DBH).
 *
 * Explosion types and energies:
 *   ECSN:  10^51 erg thermal + yields + mass
 *   CCSN:  10^51 erg thermal + yields + mass
 *   PISN:  10^52 erg thermal + complete disruption + all yields
 *   PPISN: 10^51 erg thermal + yields + BH remnant
 *   FSN:   No explosion, direct collapse to BH
 *   DBH:   No explosion, direct collapse to BH
 *
 * Uses the code_block_xchange framework for MPI-parallel tree walks.
 * KernelRadius and DensityAroundParticle come from the density loop. */

#ifdef GALSF_RESOLVEDISM_FB

double CumulFeedbackEnergy = 0;  /* cumulative feedback energy injected, all channels [erg] */
double CumulFeedbackCoupled = 0; /* cumulative measured coupled energy, channels 0-5 [erg] */
#ifdef COSMIC_RAY_FLUID
double CumulCRInjectedErg = 0;   /* CR energy measured at the feedback deposit [erg] */
#endif
double CumulSFEnergyRemoved = 0; /* gas thermal+kinetic energy removed by star formation [erg] */
double CumulFeedbackMass = 0;    /* cumulative mass returned to gas by feedback [Msun] */
double CumulFeedbackMetals = 0;  /* cumulative metals returned to gas by feedback [Msun] */
long double CumulMetalYieldToGas = 0, CumulMetalEjectedFromStars = 0; /* metal ledger on this rank: yield metals deposited, star metal mass removed [code mass] */
#if defined(METALS) && defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL)
double CumulElemToGas[NUM_METAL_SPECIES], CumulElemFromStars[NUM_METAL_SPECIES]; /* element ledger on this rank, per Metallicity slot [code mass] */
#endif
#ifdef GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION
double CumulEjectaBirthClip = 0; /* ejecta mass clipped by the birth-composition swap on this rank [Msun] */
static void resolvedism_book_birth_clip(int i, int channel, double clip)
{
    if(!(clip > 0)) {return;}
    CumulEjectaBirthClip += clip;
    printf("RESOLVEDISM EJECTA BIRTH CLIP: Task=%d ID=%llu channel=%d clipped=%.10e Msun cumul=%.10e Msun\n", ThisTask, (unsigned long long)P[i].ID, channel, clip, CumulEjectaBirthClip);
}
#ifdef GALSF_RESOLVEDISM_WINDS
/* the wind deposit's element masses as the momentum walk computes them (table interval since last_wind_log_age, rescaled to dM),
 * then the birth-composition swap; returns its clip [Msun] */
static double resolvedism_wind_birth_clip(int i, double logM, double logZ, double dM)
{
    double log_now = log10(DMAX(get_star_table_age(resolvedism_star_age_yr(i), logM, logZ), 100.0)), d[STBL_NELEM], s = 0;
    for(int k = 0; k < STBL_NELEM; k++) {d[k] = stellar_elem_ej_wind_cumulative(logM, logZ, log_now, k) - stellar_elem_ej_wind_cumulative(logM, logZ, P[i].last_wind_log_age, k); s += d[k];}
    if(s <= 0 && dM > 0) {s = 0; for(int k = 0; k < STBL_NELEM; k++) {d[k] = stellar_surface_abundance(logM, logZ, log_now, k); s += d[k];}}
    double f = (s > 0 && dM > 0) ? dM / s : 0;
    for(int k = 0; k < STBL_NELEM; k++) {d[k] *= f;}
    return resolvedism_ejecta_birth_composition(i, logZ, d);
}
#endif
#endif
int CumulSNe = 0, CumulAGB = 0, CumulIa = 0; /* cumulative event counts */
double CumulStarMassFormed = 0;  /* cumulative stellar mass formed [Msun] — set in IMF sampling */
double RadPressure_dp_thisStep = 0; /* total radpressure momentum this step [g*cm/s] */
extern double resolvedism_radp_starside_dp_code(int i); /* fb_momentum.cc: single source for star-side radp dp */

#ifndef DO_DENSITY_AROUND_NONGAS_PARTICLES
#error "GALSF_RESOLVEDISM_FB requires DO_DENSITY_AROUND_NONGAS_PARTICLES for kernel weights (wt_sum)"
#endif

/* Type Ia constants, get_star_info(), get_star_lifetime() are in resolvedism_fb_shared.h */


/* =========================================================================== */
/*  Check the single star in each active star particle for a death event.      */
/*  Each particle has exactly one sampled star in MstarSampleIMF[0].           */
/*  ALL stars that exceed their lifetime die:                                  */
/*    M >= 8: ECSN/CCSN/PISN/PPISN (explosive) or FSN/DBH (silent collapse)  */
/*    M <  8: WD (AGB end-of-life mass dump, no energy)                       */
/*  STELLA_FB_EVENT:  1=explosive SN, 2=AGB death, 3=wind, 4=Ia, -1=done    */
/* =========================================================================== */
/* wind closure reference: birth mass after IMF accretion [Msun]; the IMF draw for stars without the slot (older ICs/restarts) */
static inline double resolvedism_birth_mass_solar(int i)
{
#if (N_STELLAR_MASS >= 5)
    if(P[i].MstarSampleIMF[4] > 0) {return P[i].MstarSampleIMF[4] * UNIT_MASS_IN_SOLAR;}
#endif
    return P[i].MstarSampleIMF[0];
}

void resolvedism_determine_SNe(void)
{
    if(All.Time <= 0) return;
    int i;
    int n_sne_local = 0, n_sne_total = 0;
    int n_agb_local = 0, n_agb_total = 0;
    int n_collapse_local = 0;
    int n_wind_local = 0, n_wind_total = 0;

    /* Reset SNe flags for active particles (allows re-entry after wind dump or previous -1 marking) */
    for(int i : ActiveParticleList) {
        if(STELLA_OWNS_STAR(P[i])) STELLA_FB_EVENT(P[i]) = 0;
    }

    /* ---- Wind accumulation for living massive stars ----
     * Principled approach: single source of truth = stellar_elem_ej_wind_cumulative table.
     *   cum_table_now  = Σ_k elem_ej_wind_cumulative(t, k)         (table prediction)
     *   cum_injected   = M_init - P[i].Mass                         (what's actually been removed)
     *   dM_pending     = max(0, cum_table_now - cum_injected)       (what still needs to be injected)
     * P[i].WindMassAccum is overwritten to dM_pending each step (no longer a step-by-step accumulator).
     * P[i].WindMomentumAccum is set to dM_pending * v_wind at the current age. */
#ifdef GALSF_RESOLVEDISM_WINDS
    for(int i : ActiveParticleList)
    {
        if(!STELLA_OWNS_STAR(P[i]) || P[i].Mass <= 0 || STELLA_FB_EVENT(P[i]) != 0) continue;
        double Mstar, logM, logZ;
        if(!get_star_info(i, &Mstar, &logM, &logZ)) continue;
        if(Mstar < 8.0) continue; /* only massive stars have significant winds */

        double star_age_yr = resolvedism_star_age_yr(i);
        if(star_age_yr <= 0) continue;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
        if(star_age_yr >= lifetime_yr) continue; /* dead — handled in death loop below */

        double table_age = get_star_table_age(star_age_yr, logM, logZ);
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        if(All.TestFixedWindMdot <= 0)
#endif
        if(table_age <= 0) continue; /* PMS: no winds yet */
        double log_age = log10(DMAX(table_age, 100.0));

        double cum_now_total = 0, v_w = 0;
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        if(All.TestFixedWindMdot > 0) {
            /* Weaver-bubble test (2026-07-15): constant Mdot [Msun/yr] and v_wind [km/s]
             * pinned (bypasses tables + PMS delay) -> L_w = 0.5*Mdot*v^2 known exactly
             * for the analytic R(t) = 0.76 (L_w t^3 / rho)^(1/5) comparison. Same
             * cumulative-minus-injected closure as the table path (bit-exact mass). */
            cum_now_total = All.TestFixedWindMdot * star_age_yr;      /* Msun */
            v_w = DMAX(All.TestFixedWindVel, 10.0);                    /* km/s */
        } else
#endif
        {
            for(int kk = 0; kk < STBL_NELEM; kk++)
                cum_now_total += stellar_elem_ej_wind_cumulative(logM, logZ, log_age, kk);
            v_w = stellar_v_wind(logM, logZ, log_age);
            if(v_w < 10.0) v_w = 10.0;
        }
        double M_init_solar = resolvedism_birth_mass_solar(i);
        double cum_injected = M_init_solar - P[i].Mass * UNIT_MASS_IN_SOLAR;
        double dM_pending = cum_now_total - cum_injected;
        if(dM_pending < 0) dM_pending = 0; /* interp noise — wait for table to catch up */

        P[i].WindMassAccum = dM_pending;                /* pending mass to inject */
        P[i].WindMomentumAccum = dM_pending * v_w;      /* matching momentum (v at current age) */

#ifdef GALSF_RESOLVEDISM_WINDS_CONTINUOUS
        /* TRUE CONTINUOUS wind injection (2026-07-05): fire every active step with
         * whatever dM = int Mdot dt accumulated since the last step. Momentum uses
         * v_w at the CURRENT age each step -> p = int Mdot(t) v_w(t) dt exact (the
         * chunked mode below applies fire-time v_w to the whole accumulated chunk —
         * wrong when v_w evolves, e.g. MS 3000 km/s -> LBV few 100 km/s). Injection
         * goes through ONE batched walk pair per step for all wind stars together
         * (see inject_fb_energy), not the per-event serialized pass — wind parcels
         * are tiny, stale-weight error nil, and cluster runs stay O(1) walks/step. */
        if(dM_pending > 1.0e-10) {
            STELLA_FB_EVENT(P[i]) = 3;
            n_wind_local++;
        }
#else
        /* Trigger: mass-dependent fractional threshold. Low-mass (8 Msun): 1%; high-mass (300 Msun): ~15%. */
        double wind_frac = 0.01 + 0.19 * DMAX(0, (Mstar - 8.0)) / (350.0 - 8.0);
        if(wind_frac > 0.20) wind_frac = 0.20;
        if(dM_pending > wind_frac * Mstar) {
            STELLA_FB_EVENT(P[i]) = 3;
            n_wind_local++;
        }
#endif
    }
#endif

    /* first pass: count local death events so we can allocate buffers */
    int n_death_count = 0;
    for(int i : ActiveParticleList)
    {
        if(!STELLA_OWNS_STAR(P[i]) || P[i].Mass <= 0 || STELLA_FB_EVENT(P[i]) != 0) continue;
        double star_age_yr = resolvedism_star_age_yr(i);
        if(star_age_yr <= 0) continue;
        double Mstar, logM, logZ;
        if(!get_star_info(i, &Mstar, &logM, &logZ)) continue;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
        if(All.Time < All.TimeInstantSN && Mstar >= 8.0) {lifetime_yr = 0;}
#endif
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        if(All.TestSNTime0 > 0 && resolvedism_is_forced_sn_test_star(i)) { /* forced-SN sites: count exactly as the flagging pass decides */
            if(All.Time >= All.TestSNTime0 + ((double)P[i].ID - 1.0) * All.TestSNSpacing) {n_death_count++;}
            continue;
        }
#endif
        if(star_age_yr > lifetime_yr) {n_death_count++;}
    }

    /* allocate local death event info buffers */
    double *sn_x = (double *)mymalloc("sn_x", DMAX(n_death_count,1) * sizeof(double));
    double *sn_y = (double *)mymalloc("sn_y", DMAX(n_death_count,1) * sizeof(double));
    double *sn_z = (double *)mymalloc("sn_z", DMAX(n_death_count,1) * sizeof(double));
    double *sn_u = (double *)mymalloc("sn_u", DMAX(n_death_count,1) * sizeof(double));
    double *sn_rho = (double *)mymalloc("sn_rho", DMAX(n_death_count,1) * sizeof(double));
    double *sn_mstar = (double *)mymalloc("sn_mstar", DMAX(n_death_count,1) * sizeof(double));
    long long *sn_id = (long long *)mymalloc("sn_id", DMAX(n_death_count,1) * sizeof(long long));
    int *sn_remtype = (int *)mymalloc("sn_remtype", DMAX(n_death_count,1) * sizeof(int));
    double *sn_age = (double *)mymalloc("sn_age", DMAX(n_death_count,1) * sizeof(double));
    double *sn_lifetime = (double *)mymalloc("sn_lifetime", DMAX(n_death_count,1) * sizeof(double));
    double *sn_mej = (double *)mymalloc("sn_mej", DMAX(n_death_count,1) * sizeof(double));
    double *sn_zej = (double *)mymalloc("sn_zej", DMAX(n_death_count,1) * sizeof(double));
    double *sn_yC = (double *)mymalloc("sn_yC", DMAX(n_death_count,1) * sizeof(double));
    double *sn_yO = (double *)mymalloc("sn_yO", DMAX(n_death_count,1) * sizeof(double));
    double *sn_ySi = (double *)mymalloc("sn_ySi", DMAX(n_death_count,1) * sizeof(double));
    double *sn_yFe = (double *)mymalloc("sn_yFe", DMAX(n_death_count,1) * sizeof(double));
    int *sn_timebin = (int *)mymalloc("sn_timebin", DMAX(n_death_count,1) * sizeof(int));
    int n_logged = 0;

    /* second pass: flag death events and handle direct collapses */
    for(int i : ActiveParticleList)
    {
        if(!STELLA_OWNS_STAR(P[i]) || P[i].Mass <= 0 || STELLA_FB_EVENT(P[i]) != 0) continue;
        double star_age_yr = resolvedism_star_age_yr(i);
        if(star_age_yr <= 0) continue;
        double Mstar, logM, logZ;
        if(!get_star_info(i, &Mstar, &logM, &logZ)) continue;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
        if(All.Time < All.TimeInstantSN && Mstar >= 8.0) {lifetime_yr = 0;}
#endif
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        /* Clustered-SN test (2026-07-05): force star ID=k to explode at
         * TestSNTime0+(k-1)*TestSNSpacing, bypassing table lifetimes. Only the
         * designated test stars (resolvedism_is_forced_sn_test_star) are affected. */
        if(All.TestSNTime0 > 0 && resolvedism_is_forced_sn_test_star(i)) {
            double t_forced = All.TestSNTime0 + ((double)P[i].ID - 1.0) * All.TestSNSpacing;
            if(All.Time < t_forced) continue;   /* not yet */
            /* fall through: dies now */
        } else
#endif
        if(star_age_yr <= lifetime_yr) continue;

        /* Star has died — determine fate */
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
        int rem_type = stellar_remnant_type(logM, logZ);
        double rem_mass = stellar_remnant_mass(logM, logZ);
#else
        int rem_type = REM_CCSN; /* default: core-collapse */
        double rem_mass = 1.4;   /* default: 1.4 Msun NS */
#endif

        /* Force-dump any remaining wind mass before the SN/AGB event. Refresh
           dM_pending from the table at age=lifetime: the accumulator loop above
           skips dead stars, so WindMassAccum may carry a stale (often zero) value
           even when cum_table_at_end - cum_injected > 0. Without this refresh, the
           SN step would have to absorb the leftover wind mass as zero-composition
           bulk ejecta (FSN case), breaking Σ EA closure on recipient cells. */
#ifdef GALSF_RESOLVEDISM_WINDS
        {
            double table_age_end = get_star_table_age(lifetime_yr, logM, logZ);
            if(table_age_end > 0) {
                double log_age_end = log10(DMAX(table_age_end, 100.0));
                double cum_now_total = 0;
                for(int kk = 0; kk < STBL_NELEM; kk++)
                    cum_now_total += stellar_elem_ej_wind_cumulative(logM, logZ, log_age_end, kk);
                double M_init_solar = resolvedism_birth_mass_solar(i);
                double cum_injected = M_init_solar - P[i].Mass * UNIT_MASS_IN_SOLAR;
                double dM_pending = cum_now_total - cum_injected;
                if(dM_pending < 0) dM_pending = 0;
                /* Threshold above float precision (~1e-15 Msun): otherwise the
                 * death-loop fires a wind event every step that injects zero
                 * mass but consumes the SN-flag path via the `continue` below. */
                if(dM_pending > 1.0e-6 && P[i].FB_NoDepositCount < 2) { /* after 2 failed flushes the SN carries the residual */
                    double v_w = stellar_v_wind(logM, logZ, log_age_end);
                    if(v_w < 10.0) v_w = 10.0;
                    P[i].WindMassAccum = dM_pending;
                    P[i].WindMomentumAccum = dM_pending * v_w;
                    STELLA_FB_EVENT(P[i]) = 3;
                    n_wind_local++;
                    continue; /* wind dump this step; SN flagged next step */
                }
            }
        }
#endif
        /* Without winds: M_particle = M_init at death. The full Mej = M_init - rem_mass
           is injected through the thermal pass. The sn_yield accounts for nucleosynthesis
           on the M_preSN - rem_mass portion; the extra M_init - M_preSN portion gets
           birth composition automatically via X_birth * Mej in the yield formula. */
        /* Determine flag: 1 = explosive SN (ECSN/CCSN/PISN/PPISN), 2 = AGB/WD death */
        if(rem_type == REM_WD) {
            STELLA_FB_EVENT(P[i]) = 2; /* AGB: mass+metals, no energy */
            n_agb_local++;
        } else {
            STELLA_FB_EVENT(P[i]) = 1; /* SN: energy+mass+metals */
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
            printf("FBTL_DET task=%d ID=%llu flag=1 set (t=%g)\n", ThisTask, (unsigned long long)P[i].ID, All.Time); fflush(stdout);
#endif
            n_sne_local++;
        }
        /* Snapshot pre-walk mass for bookkeeping in resolvedism_inject_fb_energy().
         * The FB walks below will drain P[i].Mass down to ~rem_mass, so a later
         * read of P[i].Mass would give Mej~0.  This snapshot preserves the
         * correct value for the FeedbackBudget log. */
        P[i].M_at_SN_trigger = (MyFloat)(P[i].Mass * UNIT_MASS_IN_SOLAR);
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        printf("STARDEATH ID=%llu M_birth=%.17g M_pre=%.17g t=%.17g\n", (unsigned long long)P[i].ID, resolvedism_birth_mass_solar(i), P[i].Mass * UNIT_MASS_IN_SOLAR, All.Time); fflush(stdout);
#endif
        /* NOTE: do NOT zero MstarSampleIMF here — particle2in needs it */
        STELLA_CHECK("sn_log_buffer", n_logged < DMAX(n_death_count,1), "ID=%llu n_logged=%d allocated=%d", (unsigned long long)P[i].ID, n_logged, (int)DMAX(n_death_count,1));
        STELLA_CHECK_RAN("sn_log_buffer", 1);
        sn_x[n_logged] = P[i].Pos[0];
        sn_y[n_logged] = P[i].Pos[1];
        sn_z[n_logged] = P[i].Pos[2];
        sn_u[n_logged] = P[i].InternalEnergyAroundParticle;
        sn_rho[n_logged] = P[i].DensityAroundParticle;
        sn_mstar[n_logged] = Mstar;
        sn_id[n_logged] = (long long)P[i].ID;
        sn_remtype[n_logged] = rem_type;
        sn_age[n_logged] = star_age_yr;
        sn_lifetime[n_logged] = lifetime_yr;
        /* Pre-compute ejecta mass and metal yield for logging — read from absolute
         * elem_ej tables (matches what fb_thermal.cc actually injects). */
        {
            double mej_log = 0;
            double zej_log = 0, yC_log = 0, yO_log = 0, ySi_log = 0, yFe_log = 0;
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
            { /* what fb_thermal injects (stellar_death_ejecta of the pre-walk mass) */
                double m_log[STBL_NELEM];
                double rem_log = stellar_death_ejecta(logM, logZ, P[i].M_at_SN_trigger, m_log);
                for(int kk = 0; kk < STBL_NELEM; kk++) {mej_log += m_log[kk];} /* = M - rem_log */
                for(int kk = ELEM_C; kk < STBL_NELEM; kk++) {zej_log += m_log[kk];}
                yC_log = m_log[ELEM_C]; yO_log = m_log[ELEM_O]; ySi_log = m_log[ELEM_Si]; yFe_log = m_log[ELEM_Fe];
            }
#endif
            sn_mej[n_logged] = mej_log;
            sn_zej[n_logged] = zej_log;
            sn_yC[n_logged] = yC_log;
            sn_yO[n_logged] = yO_log;
            sn_ySi[n_logged] = ySi_log;
            sn_yFe[n_logged] = yFe_log;
        }
        sn_timebin[n_logged] = P[i].TimeBin;
        n_logged++;
    }

    /* MPI reduce counts for logging */
    int n_collapse_total = 0, n_logged_total = 0;
    { /* one reduction for all counts */
        int cnt[5] = {n_sne_local, n_agb_local, n_collapse_local, 0, n_logged};
#ifdef GALSF_RESOLVEDISM_WINDS
        cnt[3] = n_wind_local;
#endif
        MPI_Allreduce(MPI_IN_PLACE, cnt, 5, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        n_sne_total = cnt[0]; n_agb_total = cnt[1]; n_collapse_total = cnt[2]; n_logged_total = cnt[4];
#ifdef GALSF_RESOLVEDISM_WINDS
        n_wind_total = cnt[3];
#endif
    }
    if(n_sne_total > 0 || n_agb_total > 0 || n_collapse_total > 0 || n_wind_total > 0)
    {
        if(ThisTask == 0) {
            printf("RESOLVEDISM FB: %d SNe + %d AGB + %d collapses + %d wind-inj at t=%g\n",
                   n_sne_total, n_agb_total, n_collapse_total, n_wind_total, All.Time);
            fflush(stdout);
        }
    }

    if(n_logged_total > 0)
    {
        int *recvcounts = NULL, *displs = NULL;
        double *all_x = NULL, *all_y = NULL, *all_z = NULL, *all_u = NULL, *all_rho = NULL, *all_mstar = NULL;
        double *all_age = NULL, *all_lifetime = NULL, *all_mej = NULL, *all_zej = NULL;
        double *all_yC = NULL, *all_yO = NULL, *all_ySi = NULL, *all_yFe = NULL;
        long long *all_id = NULL;
        int *all_remtype = NULL, *all_timebin = NULL;

        if(ThisTask == 0)
        {
            recvcounts = (int *)mymalloc("recvcounts", NTask * sizeof(int));
            displs = (int *)mymalloc("displs", NTask * sizeof(int));
        }
        MPI_Gather(&n_logged, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
        {
            displs[0] = 0;
            for(i = 1; i < NTask; i++) {displs[i] = displs[i-1] + recvcounts[i-1];}
            all_x = (double *)mymalloc("all_x", n_logged_total * sizeof(double));
            all_y = (double *)mymalloc("all_y", n_logged_total * sizeof(double));
            all_z = (double *)mymalloc("all_z", n_logged_total * sizeof(double));
            all_u = (double *)mymalloc("all_u", n_logged_total * sizeof(double));
            all_rho = (double *)mymalloc("all_rho", n_logged_total * sizeof(double));
            all_mstar = (double *)mymalloc("all_mstar", n_logged_total * sizeof(double));
            all_id = (long long *)mymalloc("all_id", n_logged_total * sizeof(long long));
            all_remtype = (int *)mymalloc("all_remtype", n_logged_total * sizeof(int));
            all_age = (double *)mymalloc("all_age", n_logged_total * sizeof(double));
            all_lifetime = (double *)mymalloc("all_lifetime", n_logged_total * sizeof(double));
            all_mej = (double *)mymalloc("all_mej", n_logged_total * sizeof(double));
            all_zej = (double *)mymalloc("all_zej", n_logged_total * sizeof(double));
            all_yC  = (double *)mymalloc("all_yC",  n_logged_total * sizeof(double));
            all_yO  = (double *)mymalloc("all_yO",  n_logged_total * sizeof(double));
            all_ySi = (double *)mymalloc("all_ySi", n_logged_total * sizeof(double));
            all_yFe = (double *)mymalloc("all_yFe", n_logged_total * sizeof(double));
            all_timebin = (int *)mymalloc("all_timebin", n_logged_total * sizeof(int));
        }
        MPI_Gatherv(sn_x, n_logged, MPI_DOUBLE, all_x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_y, n_logged, MPI_DOUBLE, all_y, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_z, n_logged, MPI_DOUBLE, all_z, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_u, n_logged, MPI_DOUBLE, all_u, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_rho, n_logged, MPI_DOUBLE, all_rho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_mstar, n_logged, MPI_DOUBLE, all_mstar, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_id, n_logged, MPI_LONG_LONG, all_id, recvcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_remtype, n_logged, MPI_INT, all_remtype, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_age, n_logged, MPI_DOUBLE, all_age, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_lifetime, n_logged, MPI_DOUBLE, all_lifetime, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_mej, n_logged, MPI_DOUBLE, all_mej, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_zej, n_logged, MPI_DOUBLE, all_zej, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_yC, n_logged, MPI_DOUBLE, all_yC, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_yO, n_logged, MPI_DOUBLE, all_yO, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_ySi, n_logged, MPI_DOUBLE, all_ySi, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_yFe, n_logged, MPI_DOUBLE, all_yFe, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sn_timebin, n_logged, MPI_INT, all_timebin, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        if(ThisTask == 0)
        {
            for(i = 0; i < n_logged_total; i++) {
                (void)all_remtype; /* rows are written at booking (fb_death_rows_flush) */
            }
            fflush(FdSNinfo); fflush(FdAGBinfo);
            myfree(all_timebin);
            myfree(all_yFe); myfree(all_ySi); myfree(all_yO); myfree(all_yC);
            myfree(all_zej);
            myfree(all_mej);
            myfree(all_lifetime);
            myfree(all_age);
            myfree(all_remtype);
            myfree(all_id);
            myfree(all_mstar);
            myfree(all_rho);
            myfree(all_u);
            myfree(all_z);
            myfree(all_y);
            myfree(all_x);
            myfree(displs);
            myfree(recvcounts);
        }
    }
    myfree(sn_timebin);
    myfree(sn_yFe); myfree(sn_ySi); myfree(sn_yO); myfree(sn_yC);
    myfree(sn_zej);
    myfree(sn_mej);
    myfree(sn_lifetime);
    myfree(sn_age);
    myfree(sn_remtype);
    myfree(sn_id);
    myfree(sn_mstar);
    myfree(sn_rho);
    myfree(sn_u);
    myfree(sn_z);
    myfree(sn_y);
    myfree(sn_x);

    /* ---- Type Ia: stochastic DTD check on WD remnants ---- */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    {
        int n_ia_local = 0, n_ia_total = 0;
        for(int i : ActiveParticleList)
        {
            if(!STELLA_OWNS_STAR(P[i]) || P[i].Mass <= 0) continue;
            if(STELLA_FB_EVENT(P[i]) != 0) continue; /* already flagged for something else */
            if(P[i].M_drawn_Ia <= 0) continue; /* not a WD remnant */
            if(P[i].M_at_SN_trigger > 0) { /* Ia drawn earlier but not deposited: re-fire */
                STELLA_FB_EVENT(P[i]) = 4; n_ia_local++;
                continue;
            }
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
            if(All.TestIaTime > 0 && All.Time >= All.TestIaTime && resolvedism_is_forced_sn_test_star(i)) { /* same test-star key as the forced SNe (374ed5036) */
                STELLA_FB_EVENT(P[i]) = 4; P[i].M_at_SN_trigger = (MyFloat)(P[i].Mass * UNIT_MASS_IN_SOLAR); n_ia_local++; /* test: forced Ia of a site WD */
                continue;
            }
#endif

            double star_age_Gyr = resolvedism_star_age_yr(i) * 1.0e-9;
            if(star_age_Gyr < IA_T_MIN_GYR) continue; /* below minimum delay time */

            /* DTD probability: P = (IA_DTD_NORM / t) * M_drawn * dt */
            double dt_Gyr = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i) * UNIT_TIME_IN_GYR;
            if(dt_Gyr <= 0) continue;
            double P_Ia = (IA_DTD_NORM / star_age_Gyr) * P[i].M_drawn_Ia * dt_Gyr;
            /* only dead WD progenitors carry the hazard: rescale so the population rate matches the DTD */
            double f_dead = stellar_ia_fdead(star_age_Gyr * 1.0e9, log10(DMAX(P[i].BirthMetallicity, 1e-10)));
            P_Ia *= IA_F_WD / DMAX(f_dead, 1.0e-6);
#ifdef GALSF_RESOLVEDISM_IA_DTD_BOOST
            /* TEST-ONLY rate multiplier (ia_dtd case): pure scaling of the hazard so a
             * ~100-WD box yields O(20) events in 0.5 Gyr; the checker divides it out. */
            P_Ia *= (double)GALSF_RESOLVEDISM_IA_DTD_BOOST;
#endif

            /* Stochastic check */
            unsigned long long ia_rng = resolvedism_rng_seed(i, RESOLVEDISM_RNG_DRAW_IA); /* rank-independent */
            double rn = resolvedism_rng_uniform(&ia_rng);
            if(rn < P_Ia) {
                STELLA_FB_EVENT(P[i]) = 4; /* Type Ia */
                P[i].M_at_SN_trigger = (MyFloat)(P[i].Mass * UNIT_MASS_IN_SOLAR);
                n_ia_local++;
            }
        }
        MPI_Allreduce(&n_ia_local, &n_ia_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(n_ia_total > 0 && ThisTask == 0) {
            printf("RESOLVEDISM TYPE_IA: %d Type Ia SNe this timestep at t=%g\n", n_ia_total, All.Time);
            fflush(stdout);
        }

        /* Log Type Ia events to SNinfo.txt (same format, rem_type=7) */
        if(n_ia_total > 0)
        {
            double *ia_x, *ia_y, *ia_z, *ia_u, *ia_rho, *ia_mstar, *ia_age;
            long long *ia_id;
            ia_x = (double *)mymalloc("ia_x", n_ia_local * sizeof(double));
            ia_y = (double *)mymalloc("ia_y", n_ia_local * sizeof(double));
            ia_z = (double *)mymalloc("ia_z", n_ia_local * sizeof(double));
            ia_u = (double *)mymalloc("ia_u", n_ia_local * sizeof(double));
            ia_rho = (double *)mymalloc("ia_rho", n_ia_local * sizeof(double));
            ia_mstar = (double *)mymalloc("ia_mstar", n_ia_local * sizeof(double));
            ia_id = (long long *)mymalloc("ia_id", n_ia_local * sizeof(long long));
            ia_age = (double *)mymalloc("ia_age", n_ia_local * sizeof(double));
            int n_ia_logged = 0;
            for(int i : ActiveParticleList) {
                if(STELLA_OWNS_STAR(P[i]) && STELLA_FB_EVENT(P[i]) == 4) {
                    ia_x[n_ia_logged] = P[i].Pos[0];
                    ia_y[n_ia_logged] = P[i].Pos[1];
                    ia_z[n_ia_logged] = P[i].Pos[2];
                    ia_u[n_ia_logged] = P[i].InternalEnergyAroundParticle;
                    ia_rho[n_ia_logged] = P[i].DensityAroundParticle;
                    ia_mstar[n_ia_logged] = P[i].MstarSampleIMF[0];
                    ia_id[n_ia_logged] = (long long)P[i].ID;
                    ia_age[n_ia_logged] = resolvedism_star_age_yr(i);
                    n_ia_logged++;
                }
            }
            int *recvcounts_ia = NULL, *displs_ia = NULL;
            double *all_ia_x=NULL, *all_ia_y=NULL, *all_ia_z=NULL, *all_ia_u=NULL, *all_ia_rho=NULL, *all_ia_mstar=NULL, *all_ia_age=NULL;
            long long *all_ia_id = NULL;
            if(ThisTask == 0) {
                recvcounts_ia = (int *)mymalloc("recvcounts_ia", NTask * sizeof(int));
                displs_ia = (int *)mymalloc("displs_ia", NTask * sizeof(int));
            }
            MPI_Gather(&n_ia_logged, 1, MPI_INT, recvcounts_ia, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                displs_ia[0] = 0;
                for(i = 1; i < NTask; i++) {displs_ia[i] = displs_ia[i-1] + recvcounts_ia[i-1];}
                all_ia_x = (double *)mymalloc("all_ia_x", n_ia_total * sizeof(double));
                all_ia_y = (double *)mymalloc("all_ia_y", n_ia_total * sizeof(double));
                all_ia_z = (double *)mymalloc("all_ia_z", n_ia_total * sizeof(double));
                all_ia_u = (double *)mymalloc("all_ia_u", n_ia_total * sizeof(double));
                all_ia_rho = (double *)mymalloc("all_ia_rho", n_ia_total * sizeof(double));
                all_ia_mstar = (double *)mymalloc("all_ia_mstar", n_ia_total * sizeof(double));
                all_ia_id = (long long *)mymalloc("all_ia_id", n_ia_total * sizeof(long long));
                all_ia_age = (double *)mymalloc("all_ia_age", n_ia_total * sizeof(double));
            }
            MPI_Gatherv(ia_x, n_ia_logged, MPI_DOUBLE, all_ia_x, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_y, n_ia_logged, MPI_DOUBLE, all_ia_y, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_z, n_ia_logged, MPI_DOUBLE, all_ia_z, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_u, n_ia_logged, MPI_DOUBLE, all_ia_u, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_rho, n_ia_logged, MPI_DOUBLE, all_ia_rho, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_mstar, n_ia_logged, MPI_DOUBLE, all_ia_mstar, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_id, n_ia_logged, MPI_LONG_LONG, all_ia_id, recvcounts_ia, displs_ia, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_age, n_ia_logged, MPI_DOUBLE, all_ia_age, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                /* Ia rows are written at booking (fb_death_rows_flush) */
                myfree(all_ia_age); myfree(all_ia_id); myfree(all_ia_mstar); myfree(all_ia_rho);
                myfree(all_ia_u); myfree(all_ia_z); myfree(all_ia_y); myfree(all_ia_x);
                myfree(displs_ia); myfree(recvcounts_ia);
            }
            myfree(ia_age); myfree(ia_id); myfree(ia_mstar); myfree(ia_rho);
            myfree(ia_u); myfree(ia_z); myfree(ia_y); myfree(ia_x);
        }
    }
#endif
}


/* =========================================================================== */
/*  Injection is split into two separate tree walks:                           */
/*    resolvedism_fb_momentum.cc — wind/AGB/radpressure (pairs_threads)       */
/*    resolvedism_fb_thermal.cc  — SN/Ia thermal+mass+metals (variable_threads)*/
/* =========================================================================== */

/* (old xchange INPUT/OUTPUT/evaluate/particle2in/out2particle/active_check/fb_calc
 *  removed — now in resolvedism_fb_thermal.cc and resolvedism_fb_momentum.cc) */


/* Per-event serialization token: when nonzero, the momentum (wind/AGB) and thermal
 * (SN/Ia) active_checks admit ONLY the star with this ID, so each event runs its own
 * weighting pre-pass against the CURRENT neighbor masses immediately before its own
 * injection. 0 = no restriction (radpressure and any legacy batched passes). */
MyIDType FB_SerialEventID = 0;
MyIDType FB_SerialEventChild = 0; long long FB_SerialEventGen = 0;

/* lexicographic minimum of (ID, child number, generation) keys, the unique particle identity */
static void fb_event_key_min(void *in, void *inout, int *len, MPI_Datatype *dt)
{
    unsigned long long *a = (unsigned long long *)in, *b = (unsigned long long *)inout;
    for(int n = 0; n < *len; n++, a += 3, b += 3) {
        if(a[0] < b[0] || (a[0] == b[0] && (a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])))) {b[0] = a[0]; b[1] = a[1]; b[2] = a[2];}
    }
}

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
/* a dying star's SN deposit carries mass or energy (failed SNe and direct collapse without ejecta carry neither) */
static int resolvedism_death_has_payload(int i)
{
    double Ms, lM, lZ, m[STBL_NELEM], s = 0;
    if(!get_star_info(i, &Ms, &lM, &lZ)) {return 1;}
    int rt = stellar_remnant_type(lM, lZ);
    if(rt != REM_FSN && rt != REM_DBH) {return 1;}
    stellar_death_ejecta(lM, lZ, (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger : P[i].Mass * UNIT_MASS_IN_SOLAR, m);
    for(int k = 0; k < STBL_NELEM; k++) {s += m[k];}
    return s > 0;
}
#endif
#ifdef GALSF_RESOLVEDISM_WINDS_CONTINUOUS
int FB_WindBatchOnly = 0;   /* 1 = momentum walk admits ONLY flag-3 (wind) stars, batch mode */
#endif
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
extern double FB_TestWindPabs, FB_TestWindPvec[3], FB_TestWindMcoupled, FB_TestWindZcoupled;
extern double FB_TestRadpPabs, FB_TestRadpPvec[3];
extern double FB_TestIaFe;
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
extern long FB_IaClampCells; extern double FB_IaClampShort;
#endif
#ifdef TEST_DUMP_STATE_AROUND_WIND_DEPOSIT
/* test: raw double state of all local particles (ID,type,m,v,u) appended to <OutputDir>/wind_dep_dump.<task>.bin */
static void fb_test_state_dump(int phase)
{
    static long long ncall = 0;
    char fn[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(fn, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/wind_dep_dump.%d.bin", All.OutputDir, ThisTask);
    FILE *fd = fopen(fn, "ab"); if(!fd) {return;}
    long long n = NumPart, rec = ncall / 2; double t = All.Time;
    fwrite(&phase, sizeof(int), 1, fd); fwrite(&rec, sizeof(long long), 1, fd); fwrite(&t, sizeof(double), 1, fd); fwrite(&n, sizeof(long long), 1, fd);
    for(int i = 0; i < NumPart; i++) {
        unsigned long long id = P[i].ID; int ty = P[i].Type;
        double row[5] = {(double)P[i].Mass, (double)P[i].Vel[0], (double)P[i].Vel[1], (double)P[i].Vel[2], (P[i].Type == 0) ? (double)CellP[i].InternalEnergy : 0.0};
        fwrite(&id, sizeof(unsigned long long), 1, fd); fwrite(&ty, sizeof(int), 1, fd); fwrite(row, sizeof(double), 5, fd);
    }
    fclose(fd); ncall++;
}
#endif

/* Serialized weight+inject driver for one FB pass (2026-07-03).
 * WHY: the batched pattern (weight ALL donors, then inject ALL donors) leaves
 * Σwk != 1 whenever two same-step events share neighbors: the later event's weights
 * were measured before the earlier event's mass landed. Measured over 190 Myr in
 * test_SN_PI_G0 this mis-deposited ~16% of SN metals (remnants absorb the closure
 * error; PISN remnant kept 48 Msun with rem_table=0). Serializing per event makes
 * Σwk = 1 exact at each event's own injection time. Cost: ~2 collective walks +
 * 1 allreduce per event, at O(1-3) events/step.
 * BUBBLE RETRY: if the weight walk measures wt_sum <= 0 (no gas in kernel — star
 * inside an evacuated superbubble), grow the kernel x1.4 and re-measure (<=3 tries).
 * MPI SAFETY: every rank executes the identical sequence of collective calc() calls;
 * the per-iteration winner and retry decisions are agreed by MPI_Allreduce. */
/* Star recoil from a momentum walk, applied once with the final star mass: v -= Σdp / M_final
 * (ejecta carry v_star into the gas, so this conserves momentum; isotropic sources give Σdp = 0). */
/* local gas thermal + all-particle kinetic energy [erg]; brackets each deposit pass */
double resolvedism_mech_energy_local(void)
{
    double e = 0, a2inv = 1. / (All.cf_atime * All.cf_atime);
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) continue;
        e += 0.5 * P[i].Mass * (P[i].Vel[0]*P[i].Vel[0] + P[i].Vel[1]*P[i].Vel[1] + P[i].Vel[2]*P[i].Vel[2]) * a2inv;
        if(P[i].Type == 0) {e += P[i].Mass * CellP[i].InternalEnergy;}
    }
#ifdef COSMIC_RAY_FLUID
    return e * UNIT_ENERGY_IN_CGS + resolvedism_cr_energy_local();
#endif
    return e * UNIT_ENERGY_IN_CGS;
}

#ifdef COSMIC_RAY_FLUID
double resolvedism_cr_energy_local(void)
{
    double e = 0;
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {e += CellP[i].CosmicRayEnergy[k];}}}
    return e * UNIT_ENERGY_IN_CGS;
}
#endif

/* fully disrupted star: zero the deposit residue so the particle is removed, not left near-massless in the AGS kernels (1/m force term) */
static void resolvedism_zero_disrupted_star(int i, double M_pre_solar, const char *what)
{
    double resid = P[i].Mass * UNIT_MASS_IN_SOLAR;
    if(fabs(resid) > 1.0e-10 * M_pre_solar) {printf("RESOLVEDISM FB WARNING: %s ID=%llu kept %g of %g Msun after full disruption\n", what, (unsigned long long)P[i].ID, resid, M_pre_solar); fflush(stdout);}
    mbary_book_removed(i); /* the residue leaves the box: booked in the mass and element traces (STL-407) */
    P[i].Mass = 0; TreeMomentsStaleFlag = 1;
}

static void resolvedism_fb_apply_recoil(void)
{
    int ii, k;
    for(int ii : ActiveParticleList) {
        if(!STELLA_OWNS_STAR(P[ii])) continue;
        if(P[ii].Mass > 0) {for(k = 0; k < 3; k++) {P[ii].Vel[k] -= P[ii].FB_RecoilAccum[k] * All.cf_atime / P[ii].Mass;} force_raise_motion_bound(ii);}
        for(k = 0; k < 3; k++) {P[ii].FB_RecoilAccum[k] = 0;}
    }
    force_raise_noted_motion_bounds(); /* gas kicked by the injection that preceded this call */
}

static void resolvedism_fb_serialized_pass(void (*calc_fn)(int), int flagA, int flagB)
{
    int ii;
    /* collect local candidates (flagged donors eligible for this pass) */
    int ncand = 0, ccap = 256;
    int *cand = (int *)mymalloc("fbser_cand", ccap * sizeof(int));
    for(int ii : ActiveParticleList) {
        if(!STELLA_OWNS_STAR(P[ii])) continue;
        if(STELLA_FB_EVENT(P[ii]) != flagA && STELLA_FB_EVENT(P[ii]) != flagB) continue;
        if(P[ii].KernelRadius <= 0 || P[ii].NumNgb <= 0) continue;
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
        if(STELLA_FB_EVENT(P[ii]) == 1 && !resolvedism_death_has_payload(ii)) continue; /* nothing to deposit: no weighting walks, no kernel growth */
#endif
        if(ncand >= ccap) {ccap *= 2; cand = (int *)myrealloc_movable(cand, ccap * sizeof(int));}
        cand[ncand++] = ii;
    }
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    { int nflag=0, ntot=0; for(int ii : ActiveParticleList) {
        if(!STELLA_OWNS_STAR(P[ii])) continue; ntot++;
        if(STELLA_FB_EVENT(P[ii]) == flagA || STELLA_FB_EVENT(P[ii]) == flagB) { nflag++;
            printf("FBSER_CAND task=%d ID=%llu flag=%g KernelR=%g NumNgb=%g -> cand=%d\n", ThisTask,
                (unsigned long long)P[ii].ID, (double)STELLA_FB_EVENT(P[ii]),
                (double)P[ii].KernelRadius, (double)P[ii].NumNgb,
                (P[ii].KernelRadius>0 && P[ii].NumNgb>0)?1:0); fflush(stdout); } }
      if(nflag>0) {printf("FBSER_ENTRY task=%d flags=%d/%d ncand=%d (pass %d/%d)\n", ThisTask, nflag, ntot, ncand, flagA, flagB); fflush(stdout);} }
#endif
    char *done = (char *)mymalloc("fbser_done", (ncand > 0 ? ncand : 1) * sizeof(char));
    memset(done, 0, (ncand > 0 ? ncand : 1) * sizeof(char));

    while(1)
    {
        /* pick the globally smallest unprocessed event key (ID, child number, generation); sentinel = max */
        static MPI_Datatype key_type = MPI_DATATYPE_NULL; static MPI_Op key_min_op = MPI_OP_NULL;
        if(key_type == MPI_DATATYPE_NULL) {MPI_Type_contiguous(3, MPI_UNSIGNED_LONG_LONG, &key_type); MPI_Type_commit(&key_type); MPI_Op_create(fb_event_key_min, 1, &key_min_op);}
        unsigned long long my_key[3] = {~0ULL, ~0ULL, ~0ULL}, glob_key[3]; int my_idx = -1;
        for(ii = 0; ii < ncand; ii++) {
            if(done[ii]) continue;
            unsigned long long k_[3] = {(unsigned long long)P[cand[ii]].ID, (unsigned long long)P[cand[ii]].ID_child_number, (unsigned long long)P[cand[ii]].ID_generation};
            int s_ = 1; fb_event_key_min(k_, my_key, &s_, &key_type);
            if(my_key[0] == k_[0] && my_key[1] == k_[1] && my_key[2] == k_[2]) {my_idx = ii;}
        }
        MPI_Allreduce(my_key, glob_key, 1, key_type, key_min_op, MPI_COMM_WORLD);
        if(glob_key[0] == ~0ULL && glob_key[1] == ~0ULL && glob_key[2] == ~0ULL) break; /* no events left anywhere */
        unsigned long long glob_min = glob_key[0];
        FB_SerialEventID = (MyIDType)glob_key[0]; FB_SerialEventChild = (MyIDType)glob_key[1]; FB_SerialEventGen = (long long)glob_key[2];
        int i_owner = (my_idx >= 0 && my_key[0] == glob_key[0] && my_key[1] == glob_key[1] && my_key[2] == glob_key[2]) ? cand[my_idx] : -1;

        /* fresh weighting for THIS event against current neighbor masses,
         * with bubble retry (kernel growth) if no gas weight was found */
        int try_num, ok_local, ok_glob;
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
        int spawn_this = 0, sp_glob = 0;
#endif
        for(try_num = 0; try_num < 4; try_num++) {
            for(int ii : ActiveParticleList) {
                if(STELLA_OWNS_STAR(P[ii])) {P[ii].FB_Area_weighted_sum = 0;
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
                    P[ii].FB_KernelGasMass = 0;
#endif
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
                    {int hp_; for(hp_=0;hp_<12;hp_++) {P[ii].FB_HpxCount[hp_] = 0;}}
#endif
                }
            }
            calc_fn(-1); /* weighting pre-pass (collective) */
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
            if(i_owner >= 0 && All.Time < All.TestFBNoGasUntilTime) {P[i_owner].FB_Area_weighted_sum = 0;} /* test: no gas in reach */
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
            if(i_owner >= 0 && All.Time < All.TestFBNoGasUntilTime) {P[i_owner].FB_KernelGasMass = 0;}
#endif
#endif
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
            /* ---- SN ejecta-spawn decision (prototype, 2026-07-05) ----
             * If the weight walk found less gas in the kernel than f*Mej (f = flag value),
             * kernel deposition would couple across a cavity to a distant shell. Instead:
             * store the payload (Mej + THERMAL E + yields) on the star; a new gas cell is
             * created at the next merge/split phase and the split machinery + hydro do the
             * rest. Thermal SN pass only (flagA==1, explosive remnants). Collective skip. */
            if(try_num == 0 && flagA == 1 && i_owner >= 0 && STELLA_FB_EVENT(P[i_owner]) == 1) {
                double Mstar_sp = P[i_owner].MstarSampleIMF[0];
                double Zbirth_sp = (P[i_owner].BirthMetallicity > 0) ? P[i_owner].BirthMetallicity : All.InitMetallicityinSolar*All.SolarAbundances[0];
                double logM_sp = log10(DMAX(Mstar_sp, 0.1)), logZ_sp = log10(DMAX(Zbirth_sp, 1e-8));
                int rt_sp = stellar_remnant_type(logM_sp, logZ_sp);
                double Mpre_sp = (P[i_owner].M_at_SN_trigger > 0) ? (double)P[i_owner].M_at_SN_trigger : P[i_owner].Mass*UNIT_MASS_IN_SOLAR;
                double m_k_sp[STBL_NELEM];
                double rem_sp = stellar_death_ejecta(logM_sp, logZ_sp, Mpre_sp, m_k_sp);
                if(rt_sp != REM_FSN && rt_sp != REM_DBH) {   /* explosive only (Esne>0) */
                    double Mej_sp = 0; for(int ksp = 0; ksp < STBL_NELEM; ksp++) {Mej_sp += m_k_sp[ksp];}
                    double f_thresh = (double)(GALSF_RESOLVEDISM_SN_SPAWN);
                    if(P[i_owner].FB_KernelGasMass*UNIT_MASS_IN_SOLAR < f_thresh*Mej_sp && Mej_sp > 0) {
                        /* payload: the same death ejecta as fb_thermal */
                        int ksp;
                        for(ksp = 0; ksp < STBL_NELEM; ksp++) P[i_owner].SpawnEjZ[ksp] = (MyFloat)(m_k_sp[ksp]/UNIT_MASS_IN_SOLAR);
                        P[i_owner].SpawnEjMass = (MyFloat)(Mej_sp/UNIT_MASS_IN_SOLAR);
                        P[i_owner].SpawnEjEnergy = (MyFloat)(resolvedism_sn_energy_erg(rt_sp == REM_PISN, rt_sp == REM_ECSN, 1)/UNIT_ENERGY_IN_CGS);
                        ESTEP_BOOK(EBK_FB_SN, -P[i_owner].SpawnEjEnergy); /* booked in this step's E_injected, delivered by the ejecta cell at the next merge/split */
                        P[i_owner].SN_SpawnPending = 1;
                        /* keep STELLA_FB_EVENT==1 and M_at_SN_trigger: the post-walk
                         * conversion must still run for this star (FeedbackBudget entries +
                         * MstarSampleIMF/luminosity zeroing + RemnantType), otherwise the
                         * star re-flags as a live SN every step and re-spawns (round-2 bug:
                         * 22 DECIDEs / repeated EXECs from 9 stars). Kernel re-injection
                         * cannot happen: this event is marked done[] below and the
                         * serialization token never re-admits it. The mass drain happens
                         * at EXEC time in merge_split.cc, star -> spawned cell atomically.
                         * Validated round-3 (job 173065): 99 SNe, DECIDE=EXEC=conv=99, 0 dups. */
                        printf("SPAWNSN_DECIDE: star=%llu Mkern=%.3e Msun < %.2f*Mej=%.3e -> payload stored (rem=%.3f)\n",
                            (unsigned long long)P[i_owner].ID, P[i_owner].FB_KernelGasMass*UNIT_MASS_IN_SOLAR,
                            f_thresh, Mej_sp, rem_sp); fflush(stdout);
                        spawn_this = 1;
                    }
                }
            }
            if(try_num == 0) {MPI_Allreduce(&spawn_this, &sp_glob, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); if(sp_glob) break;} /* ejecta cell first; kernel growth only if none */
#endif
            ok_local = 1;
            if(i_owner >= 0 && P[i_owner].FB_Area_weighted_sum <= 0) {ok_local = 0;}
            MPI_Allreduce(&ok_local, &ok_glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if(ok_glob) break;
            if(i_owner >= 0) {
                P[i_owner].KernelRadius *= 1.4; /* star in an evacuated bubble: widen and retry */
                printf("RESOLVEDISM FB: event ID=%llu found no gas in kernel; growing KernelRadius to %g (try %d)\n",
                       (unsigned long long)glob_min, P[i_owner].KernelRadius, try_num + 1);
            }
        }
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
        if(sp_glob) {if(i_owner >= 0) {done[my_idx] = 1; if(spawn_this) {P[i_owner].FB_Deposited = 1;}} continue;}  /* all ranks skip injection for this event */
#endif
        calc_fn(0); /* injection for this single event (collective) */
        resolvedism_fb_apply_recoil();
        if(i_owner >= 0 && ok_glob) {P[i_owner].FB_Deposited = 1;}

#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        if(i_owner >= 0) { /* FBSER trace (2026-07-04, twin-SN clip hunt): AWS + post-injection mass */
            printf("FBSER event=%llu task=%d AWS=%.8e KernelR=%.6e NumNgb=%g M_after=%.8e\n",
                   (unsigned long long)glob_min, ThisTask, (double)P[i_owner].FB_Area_weighted_sum,
                   (double)P[i_owner].KernelRadius, (double)P[i_owner].NumNgb, (double)P[i_owner].Mass);
            fflush(stdout);
        }
#endif
        if(i_owner >= 0) {done[my_idx] = 1;} /* owner retires the event */
    }
    FB_SerialEventID = 0; FB_SerialEventChild = 0; FB_SerialEventGen = 0; /* release: subsequent passes see all particles */
    myfree(done);
    myfree(cand);
}

void resolvedism_inject_fb_energy(void)
{
    int ii;
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    { int q; for(q=0;q<NumPart;q++) { if(STELLA_OWNS_STAR(P[q]) && STELLA_FB_EVENT(P[q]) != 0 && STELLA_FB_EVENT(P[q]) != -1) {
        int act=0; for(int aa : ActiveParticleList) {if(aa==q){act=1;break;}}
        printf("FBTL_INJ task=%d ID=%llu flag=%g active=%d t=%g\n", ThisTask,
               (unsigned long long)P[q].ID, (double)STELLA_FB_EVENT(P[q]), act, All.Time); fflush(stdout); } } }
#endif
    /* STARFORGE two-pass pattern (adapted from mechanical_fb_calc_toplevel):
     *   1. Weighting pre-pass walks neighbors and measures Σ_j (Mass_j × kernel.wk)
     *      onto P[i].FB_Area_weighted_sum
     *   2. Injection pass uses that measured sum as denominator → Σwk = 1 exactly
     *   → all per-event sums (Mej, Esne, p_ejecta, yields[k]) conserve bit-exact.
     * SERIALIZED PER EVENT (2026-07-03): each event re-measures its weights against
     * the current neighbor masses immediately before its own injection, so same-step
     * events sharing neighbors no longer break Σwk = 1 (see
     * resolvedism_fb_serialized_pass above). */
    #ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    #define FB_ZERO_AWS() do { for(int ii : ActiveParticleList) { if(STELLA_OWNS_STAR(P[ii])) {P[ii].FB_Area_weighted_sum = 0; int hp_; for(hp_=0;hp_<12;hp_++) {P[ii].FB_HpxCount[hp_] = 0;}} } } while(0)
#else
    #define FB_ZERO_AWS() do { for(int ii : ActiveParticleList) { if(STELLA_OWNS_STAR(P[ii])) P[ii].FB_Area_weighted_sum = 0; } } while(0)
#endif

    for(int ii : ActiveParticleList) {if(STELLA_OWNS_STAR(P[ii])) {P[ii].FB_Deposited = 0; P[ii].FB_DepositedP = P[ii].FB_DepositedZ = 0;}}
    double E_mech0 = resolvedism_mech_energy_local(), E_mech1 = E_mech0, E_mech2, E_mech3;
#ifdef COSMIC_RAY_FLUID
    double E_cr0 = resolvedism_cr_energy_local(), E_cr3 = E_cr0; /* CR injection by winds and SNe */
#endif

    /* ---- Pass 0: wind + AGB (momentum + mass + metals).  Fires first so final
     *      winds happen before the star explodes as SN. Serialized per event. ---- */
#if defined(GALSF_RESOLVEDISM_ISOLATED_FB_TEST) && defined(METALS)
    { /* test: one unbooked metal change, which the mass trace must catch */
        static int kicked = 0;
        if(!kicked && All.TestMetalKickTime > 0 && All.Time >= All.TestMetalKickTime) {
            kicked = 1;
            for(int q = 0; q < NumPart; q++) {if(P[q].Type == 0 && P[q].ID == 10 && P[q].Mass > 0) {P[q].Metallicity[0] += All.TestMetalKickMass / P[q].Mass; printf("TESTMETALKICK t=%.10e dMZ=%.17e\n", All.Time, All.TestMetalKickMass);}}
        }
    }
#endif
    MBARY_STEP("pre_fb_mom_p0");
    ESTEP_BEGIN(ESTEP_FB_WIND_AGB);
#ifdef GALSF_RESOLVEDISM_WINDS_CONTINUOUS
    /* AGB deaths (big single dumps) stay serialized; WINDS batch: one collective
     * walk pair per step for ALL wind stars together (tiny parcels, stale-weight
     * error nil; O(1) walks/step regardless of the number of blowing stars). */
    resolvedism_fb_serialized_pass(resolvedism_fb_momentum_calc, 2, 2);
    E_mech1 = resolvedism_mech_energy_local();
#ifdef TEST_DUMP_STATE_AROUND_WIND_DEPOSIT
    int fb_test_nwind = 0, fb_test_nwind_glob = 0;
    for(int ii : ActiveParticleList) {if(STELLA_OWNS_STAR(P[ii]) && STELLA_FB_EVENT(P[ii]) == 3) {fb_test_nwind++;}}
    MPI_Allreduce(&fb_test_nwind, &fb_test_nwind_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(fb_test_nwind_glob > 0) {fb_test_state_dump(0);}
#endif
    FB_WindBatchOnly = 1;
    FB_ZERO_AWS();
    resolvedism_fb_momentum_calc(-1);
    for(int ii : ActiveParticleList) {
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
        if(STELLA_OWNS_STAR(P[ii]) && STELLA_FB_EVENT(P[ii]) == 3 && All.Time < All.TestFBNoGasUntilTime) {P[ii].FB_Area_weighted_sum = 0;} /* test: no gas in reach */
#endif
        if(STELLA_OWNS_STAR(P[ii]) && STELLA_FB_EVENT(P[ii]) == 3 && P[ii].FB_Area_weighted_sum > 0) {P[ii].FB_Deposited = 1;}
    }
    resolvedism_fb_momentum_calc(0);
    resolvedism_fb_apply_recoil();
    FB_WindBatchOnly = 0;
#else
    resolvedism_fb_serialized_pass(resolvedism_fb_momentum_calc, 2, 3);
#endif
    ESTEP_END(ESTEP_FB_WIND_AGB);
    MBARY_STEP("post_fb_mom_p0");
    E_mech2 = resolvedism_mech_energy_local();
#if defined(TEST_DUMP_STATE_AROUND_WIND_DEPOSIT) && defined(GALSF_RESOLVEDISM_WINDS_CONTINUOUS)
    if(fb_test_nwind_glob > 0) {fb_test_state_dump(1);}
#endif
#if defined(GALSF_RESOLVEDISM_ISOLATED_FB_TEST) && defined(GALSF_RESOLVEDISM_WINDS_CONTINUOUS)
    { /* test: booked vs delivered for this step's wind batch, full precision */
        double loc[11] = {0}, glob[11];
        const double pconv = UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s / (SOLAR_MASS_CGS * 1.0e5); /* code p -> Msun km/s */
        for(int ii : ActiveParticleList) {
            if(!STELLA_OWNS_STAR(P[ii]) || STELLA_FB_EVENT(P[ii]) != 3 || !P[ii].FB_Deposited) continue;
            double dMs = P[ii].WindMassAccum, ps = P[ii].WindMomentumAccum;
            loc[0] += 1; loc[1] += dMs; loc[2] += ps;
            if(dMs > 0) {loc[3] += 0.5 * ps * ps / dMs * SOLAR_MASS_CGS * 1.0e10;}
            double aws = P[ii].FB_Area_weighted_sum, m2 = 0; int k_;
            if(aws > 0) {for(k_ = 0; k_ < 3; k_++) {m2 += (P[ii].FB_VecSum[k_] / aws) * (P[ii].FB_VecSum[k_] / aws);}}
            loc[4] += m2;
        }
        loc[5] = E_mech2 - E_mech1;
        loc[6] = FB_TestWindPabs * pconv;
        for(int k_ = 0; k_ < 3; k_++) {loc[7 + k_] = FB_TestWindPvec[k_] * pconv;}
        loc[10] = FB_TestWindMcoupled * UNIT_MASS_IN_SOLAR;
        MPI_Allreduce(loc, glob, 11, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0 && glob[0] > 0) {
            double pv = sqrt(glob[7]*glob[7] + glob[8]*glob[8] + glob[9]*glob[9]);
            printf("WINDLEDGER t=%.17e n=%g M_book=%.17e M_coupled=%.17e p_book=%.17e p_abs=%.17e p_vec=%.17e E_book=%.17e E_coupled=%.17e m2=%.17e dE_rel=%.6e dP_rel=%.6e\n",
                   All.Time, glob[0], glob[1], glob[10], glob[2], glob[6], pv, glob[3], glob[5], glob[4],
                   (glob[3] > 0) ? glob[5] / glob[3] - 1 : 0.0, (glob[2] > 0) ? glob[6] / glob[2] - 1 : 0.0);
            fflush(stdout);
        }
        FB_TestWindPabs = FB_TestWindMcoupled = 0; FB_TestWindPvec[0] = FB_TestWindPvec[1] = FB_TestWindPvec[2] = 0;
    }
#endif

    /* ---- Pass 1: SN + Type Ia (mass + thermal energy + metals). Serialized per event. ---- */
    MBARY_STEP("pre_fb_thermal");
    ESTEP_BEGIN(ESTEP_FB_SN_IA);
    resolvedism_fb_serialized_pass(resolvedism_fb_thermal_calc, 1, 4);
    ESTEP_END(ESTEP_FB_SN_IA);
    MBARY_STEP("post_fb_thermal");
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    { /* Ia conversion clamp: cells whose share of M_Ch - M_WD exceeded their mass were converted whole; the rest is not converted */
        static long n_tot = 0; static double s_tot = 0;
        double loc[2] = {(double)FB_IaClampCells, FB_IaClampShort}, glob[2];
        MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(glob[0] > 0) {
            n_tot += (long)glob[0]; s_tot += glob[1];
            if(ThisTask == 0) {printf("RESOLVEDISM TYPE_IA CLAMP: t=%.17e cells=%ld short=%.17e[code] short_Msun=%.6e cells_total=%ld short_total_Msun=%.6e\n",
                                      All.Time, (long)glob[0], glob[1], glob[1] * UNIT_MASS_IN_SOLAR, n_tot, s_tot * UNIT_MASS_IN_SOLAR); fflush(stdout);}
        }
        FB_IaClampCells = 0; FB_IaClampShort = 0;
    }
#endif
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    { /* test: iron of Ia composition delivered this step [Msun] */
        double fe_loc = FB_TestIaFe * UNIT_MASS_IN_SOLAR, fe_glob = 0;
        MPI_Allreduce(&fe_loc, &fe_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0 && fe_glob > 0) {printf("IAFE t=%.17e Fe=%.17e Fe_table=%.17e\n", All.Time, fe_glob, stellar_type_ia_yield(ELEM_Fe)); fflush(stdout);}
        FB_TestIaFe = 0;
        extern double FB_TestIaNegCells, FB_TestIaMinX;
        double ng_loc[2] = {FB_TestIaNegCells, -FB_TestIaMinX}, ng_glob[2] = {0, 0}; /* count summed; lowest fraction as the maximum of its negative */
        MPI_Allreduce(&ng_loc[0], &ng_glob[0], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); MPI_Allreduce(&ng_loc[1], &ng_glob[1], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(ThisTask == 0 && fe_glob > 0) {printf("IANEGX t=%.17e neg_cells=%.0f min_x=%.17e\n", All.Time, ng_glob[0], -ng_glob[1]); fflush(stdout);}
        FB_TestIaNegCells = 0; FB_TestIaMinX = 0;
    }
#endif
    ESTEP_BEGIN(ESTEP_FB_POST);
#if defined(STELLA_FIRE_YIELD_SLOTS) && defined(STELLA_SELFCHECK)
    { /* every step: each gas cell keeps sum(C..Fe) <= Z and non-negative FIRE slots */
        int ibad = -1, nbad = 0, ngas = 0;
        for(int ig = 0; ig < N_gas; ig++) {
            if(P[ig].Type != 0 || P[ig].Mass <= 0) continue;
            double sub = 0; int s, neg = 0; ngas++;
            for(s = 0; s < NUM_METAL_SPECIES; s++) {if(P[ig].Metallicity[s] < 0) {neg = 1;}}
            for(s = 2; s < STELLA_FIRE_YIELD_SLOTS; s++) {sub += P[ig].Metallicity[s];}
            if(neg || !(sub <= P[ig].Metallicity[0])) {nbad++; if(ibad < 0) {ibad = ig;}}
        }
        STELLA_CHECK("fire_slot_invariant", nbad == 0, "t=%.10g %d of %d cells, first cell=%llu Z=%.17g C=%.17g Fe=%.17g", All.Time, nbad, ngas,
                     (unsigned long long)(ibad >= 0 ? P[ibad].ID : 0), ibad >= 0 ? P[ibad].Metallicity[0] : 0., ibad >= 0 ? P[ibad].Metallicity[2] : 0., ibad >= 0 ? P[ibad].Metallicity[10] : 0.);
        STELLA_CHECK_RAN("fire_slot_invariant", ngas);
    }
#endif

    /* Budget tracking: [0]=SN, [1]=AGB, [2]=wind, [3]=radpressure, [4]=Ia, [5]=failed SN / no-gas attempt (E=M=0), [6]=subgrid PI heating */
#define NFBCH 7
    double n_events[NFBCH] = {0};
    double M_injected[NFBCH] = {0};   /* ejecta mass [Msun] */
    double M_removed[NFBCH] = {0};    /* mass removed from star [Msun] */
    double E_injected[NFBCH] = {0};   /* energy [erg] */
    double dp_injected[NFBCH] = {0};  /* |momentum| [g*cm/s] */
    double Z_injected[NFBCH] = {0};   /* metals injected [Msun] */
    double E_coupled[NFBCH] = {0};    /* measured gas+star energy change of the deposit pass [erg] */

    /* Wind event logging buffers */
    int n_wind_logged = 0, n_wind_alloc = 1; /* sized to this step's wind events (was a fixed 64 that dropped rows) */
    for(int iw : ActiveParticleList) {if(STELLA_OWNS_STAR(P[iw]) && STELLA_FB_EVENT(P[iw]) == 3) n_wind_alloc++;}
    double *wind_log_x = (double *)mymalloc("wind_log_x", n_wind_alloc * sizeof(double));
    double *wind_log_y = (double *)mymalloc("wind_log_y", n_wind_alloc * sizeof(double));
    double *wind_log_z = (double *)mymalloc("wind_log_z", n_wind_alloc * sizeof(double));
    double *wind_log_mstar = (double *)mymalloc("wind_log_mstar", n_wind_alloc * sizeof(double));
    double *wind_log_dm = (double *)mymalloc("wind_log_dm", n_wind_alloc * sizeof(double));
    double *wind_log_dp = (double *)mymalloc("wind_log_dp", n_wind_alloc * sizeof(double));
    double *wind_log_zw = (double *)mymalloc("wind_log_zw", n_wind_alloc * sizeof(double));
    long long *wind_log_id = (long long *)mymalloc("wind_log_id", n_wind_alloc * sizeof(long long));
    int *wind_log_tb = (int *)mymalloc("wind_log_tb", n_wind_alloc * sizeof(int));

    /* Post-injection: handle latching, cleanup, and budget tracking for pass 0 */
    int i;
#ifdef STELLA_SELFCHECK
    int n_selfcheck_book = 0;
#endif
    for(int i : ActiveParticleList)
    {
        if(STELLA_OWNS_STAR(P[i]) && STELLA_FB_EVENT(P[i]) >= 1 && STELLA_FB_EVENT(P[i]) <= 4) {
            int has_payload = 1; /* silent collapse with no ejecta has nothing to deposit */
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
            if(STELLA_FB_EVENT(P[i]) == 1) {has_payload = resolvedism_death_has_payload(i);}
#endif
            if(has_payload && !P[i].FB_Deposited) {
                /* no gas reached: book nothing, keep the star's mass, retry next step */
                P[i].FB_NoDepositCount++;
                if(STELLA_FB_EVENT(P[i]) != 3) { /* SN/AGB/Ia attempt that reached no gas: record with zero energy and mass */
                    const double y0[6] = {0,0,0,0,0,0}; double Ms0_, lM0_, lZ0_; int rt0_ = (STELLA_FB_EVENT(P[i]) == 4) ? 7 : 0;
                    if(STELLA_FB_EVENT(P[i]) != 4 && get_star_info(i, &Ms0_, &lM0_, &lZ0_)) {rt0_ = stellar_remnant_type(lM0_, lZ0_);}
                    n_events[5] += 1;
                    fb_death_row_push(i, STELLA_FB_EVENT(P[i]) == 2, rt0_, FB_LOG_NOGAS, 0.0, y0, resolvedism_star_age_yr(i), 0.0);
                }
                printf("RESOLVEDISM FB: event flag=%d ID=%llu deposited nothing (no gas); not booked, retry %d\n",
                       (int)STELLA_FB_EVENT(P[i]), (unsigned long long)P[i].ID, (int)P[i].FB_NoDepositCount);
                if(STELLA_FB_EVENT(P[i]) != 4) {P[i].M_at_SN_trigger = -1;} /* Ia keeps its trigger to re-fire */
#ifdef GALSF_RESOLVEDISM_WINDS
                if(STELLA_FB_EVENT(P[i]) == 3) {P[i].WindMassAccum = 0; P[i].WindMomentumAccum = 0;}
#endif
                STELLA_FB_EVENT(P[i]) = -1;
                continue;
            }
            P[i].FB_NoDepositCount = 0;
#if defined(STELLA_SELFCHECK) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
            if(STELLA_FB_EVENT(P[i]) == 3 && P[i].WindMassAccum > 0) { /* a booked wind must have found gas */
                STELLA_CHECK("fb_booked_deposited", P[i].FB_Deposited, "flag=3 ID=%llu dM=%g booked without gas", (unsigned long long)P[i].ID, (double)P[i].WindMassAccum);
                n_selfcheck_book++;
            }
            if(STELLA_FB_EVENT(P[i]) != 3 && P[i].M_at_SN_trigger > 0) { /* a booked event with ejecta must have left the star */
                double Ms_, lM_, lZ_, ej_ = P[i].M_at_SN_trigger, m_[STBL_NELEM];
                if(STELLA_FB_EVENT(P[i]) != 4 && get_star_info(i, &Ms_, &lM_, &lZ_)) {
                    ej_ = P[i].M_at_SN_trigger - stellar_death_ejecta(lM_, lZ_, P[i].M_at_SN_trigger, m_);}
                double Mtrig_ = P[i].M_at_SN_trigger, Mnow_ = P[i].Mass * UNIT_MASS_IN_SOLAR;
                if(ej_ > 1.0e-6) {
                    STELLA_CHECK("fb_booked_deposited", Mnow_ < Mtrig_ * (1 - 1.0e-6), "flag=%d ID=%llu M_trigger=%g M_now=%g",
                                 (int)STELLA_FB_EVENT(P[i]), (unsigned long long)P[i].ID, Mtrig_, Mnow_);
                    n_selfcheck_book++;
                }
            }
#endif
        }
#ifdef GALSF_RESOLVEDISM_WINDS
        /* Wind injection complete: tally budget only.  Star mass has already
         * been reduced by the measured-coupling out2particle in the fb_momentum
         * walk above (out->M_coupled), so do NOT reduce P[i].Mass here — that
         * would double-deduct. */
        if(STELLA_FB_EVENT(P[i]) == 3) {
            double dM_wind = P[i].WindMassAccum; /* Msun, for budget tracking */
            double dp_wind = P[i].WindMomentumAccum * SOLAR_MASS_CGS * 1.0e5; /* g*cm/s */

            n_events[2] += 1;
            M_injected[2] += dM_wind;
            M_removed[2] += dM_wind;
            dp_injected[2] += dp_wind;
            /* source-side wind kinetic energy E = 1/2 dM v_w^2 = 1/2 dp^2/dM [erg];
             * same semantics as the SN channel's 1e51 source booking (what the walk
             * thermalizes of it is mode-dependent: full in THERMAL_ONLY, inelastic
             * residual in WINDS_THERMAL, none in momentum-only). Was never booked —
             * FeedbackBudget E column read 0 for winds in all modes (2026-07-06). */
            if(dM_wind > 0) {E_injected[2] += 0.5 * dp_wind * dp_wind / (dM_wind * SOLAR_MASS_CGS);}
#ifdef COSMIC_RAY_FLUID
            if(dM_wind > 0 && P[i].WindMomentumAccum > 1000. * dM_wind) {dp_injected[2] -= dp_wind * (1. - sqrt(1. - All.CosmicRay_SNeFraction));} /* fast wind: f_CR of the KE went to CRs */
#endif
            Z_injected[2] += P[i].FB_DepositedZ * UNIT_MASS_IN_SOLAR; /* wind metals the walk delivered */
#ifdef GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION
            {double Ms_, lM_, lZ_; if(dM_wind > 0 && get_star_info(i, &Ms_, &lM_, &lZ_)) {resolvedism_book_birth_clip(i, 2, resolvedism_wind_birth_clip(i, lM_, lZ_, dM_wind));}}
#endif

            /* Log wind event */
            if(n_wind_logged < n_wind_alloc) {
                double Z_w_log = P[i].FB_DepositedZ * UNIT_MASS_IN_SOLAR;
                wind_log_x[n_wind_logged] = P[i].Pos[0];
                wind_log_y[n_wind_logged] = P[i].Pos[1];
                wind_log_z[n_wind_logged] = P[i].Pos[2];
                wind_log_id[n_wind_logged] = (long long)P[i].ID;
                wind_log_mstar[n_wind_logged] = P[i].MstarSampleIMF[0];
                wind_log_dm[n_wind_logged] = dM_wind;
                wind_log_dp[n_wind_logged] = dp_wind;
                wind_log_zw[n_wind_logged] = Z_w_log;
                wind_log_tb[n_wind_logged] = P[i].TimeBin;
                n_wind_logged++;
            }
            printf("RESOLVEDISM WIND: Task=%d ID=%llu M_init=%.2f M_cur=%.4f dM=%.4f dp=%.3e frac=%.3f bin=%d t=%.6f\n",
                ThisTask, (unsigned long long)P[i].ID, P[i].MstarSampleIMF[0],
                P[i].Mass * UNIT_MASS_IN_SOLAR, dM_wind, dp_wind,
                dM_wind / P[i].MstarSampleIMF[0], P[i].TimeBin, All.Time);
            P[i].WindMassAccum = 0;
            P[i].WindMomentumAccum = 0;
            /* Advance the cumulative-table pointer so the next wind FB call
             * injects only what the table integrated since this moment. */
            {
                double age_yr_now = resolvedism_star_age_yr(i);
                double Mstar_log = P[i].MstarSampleIMF[0];
                double logM_log  = log10(DMAX(Mstar_log, 0.08));
                double logZ_log  = log10(DMAX(P[i].BirthMetallicity, 1e-10));
                double table_age_now = get_star_table_age(age_yr_now, logM_log, logZ_log);
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                printf("WINDAGES ID=%llu t=%.17g log_age_prev=%.17g log_age_now=%.17g dM=%.17g\n", (unsigned long long)P[i].ID, All.Time, (double)P[i].last_wind_log_age, log10(DMAX(table_age_now, 100.0)), dM_wind); fflush(stdout);
#endif
                P[i].last_wind_log_age = log10(DMAX(table_age_now, 100.0));
            }
            STELLA_FB_EVENT(P[i]) = -1;
            continue;
        }
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
        /* Type Ia: WD fully disrupted.  Actual injected mass is the WD progenitor's
         * current particle mass (P[i].Mass), not the canonical Chandrasekhar mass,
         * because single-star sampling can leave the WD below 1.378 Msun after prior
         * wind/AGB phases.  Yields are rescaled by (actual / Chandrasekhar) ratio. */
        if(STELLA_FB_EVENT(P[i]) == 4) {
            /* Use pre-walk snapshot — P[i].Mass has already been drained by thermal walk */
            double M_WD = (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger
                                                     : P[i].Mass * UNIT_MASS_IN_SOLAR;
            n_events[4] += 1;
            M_injected[4] += M_WD;     /* actual injected mass (= WD mass) */
            M_removed[4] += M_WD;      /* particle mass removed */
            E_injected[4] += IA_ENERGY_ERG;
            double Z_ia = 0; /* the table's Ia metals: the deposit delivers the full M_Ch of Ia matter */
            for(int kk = ELEM_C; kk < STBL_NELEM; kk++) Z_ia += stellar_type_ia_yield(kk);
            Z_injected[4] += P[i].FB_DepositedZ * UNIT_MASS_IN_SOLAR; /* metals the walk delivered */
            {double yia[6] = {M_WD, Z_ia, stellar_type_ia_yield(ELEM_C), stellar_type_ia_yield(ELEM_O),
                              stellar_type_ia_yield(ELEM_Si), stellar_type_ia_yield(ELEM_Fe)};
             fb_death_row_push(i, 0, 7, FB_LOG_EXPLODED, IA_ENERGY_ERG, yia, resolvedism_star_age_yr(i), 0.0);}
            printf("RESOLVEDISM TYPE_IA: Task=%d ID=%llu M_WD=%.3f E=%.2e[erg]\n",
                ThisTask, (unsigned long long)P[i].ID, M_WD, IA_ENERGY_ERG);
            resolvedism_zero_disrupted_star(i, M_WD, "Ia");
            P[i].M_drawn_Ia = 0; /* no longer eligible */
            P[i].M_at_SN_trigger = -1; /* reset bookkeeping snapshot */
            STELLA_FB_EVENT(P[i]) = -1;
            continue;
        }
#endif
        if(STELLA_FB_EVENT(P[i]) != 1 && STELLA_FB_EVENT(P[i]) != 2) continue;

        double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        Mstar = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar = P[i].Mstar;
#endif

        int channel = (STELLA_FB_EVENT(P[i]) == 2) ? 1 : 0; /* 0=SN, 1=AGB */
        /* Use the pre-walk snapshot captured in resolvedism_determine_SNe().
         * P[i].Mass has already been drained by the FB walks to ~rem_mass,
         * so reading it here would log Mej~0.  Snapshot preserves the true
         * pre-walk mass for accurate FeedbackBudget bookkeeping. Fall back to
         * current P[i].Mass if snapshot is unavailable (legacy/IC case). */
        double M_star_old = (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger
                                                       : P[i].Mass * UNIT_MASS_IN_SOLAR;

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
        if(Mstar > 0) {
            double logM = log10(Mstar);
            double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
            int rem_type = stellar_remnant_type(logM, logZ);
            double m_led[STBL_NELEM]; /* the deposit's element ejecta (stellar_death_ejecta of the pre-walk mass) */
            double rem_mass = stellar_death_ejecta(logM, logZ, M_star_old, m_led);
#ifdef GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION
            resolvedism_book_birth_clip(i, channel, resolvedism_ejecta_birth_composition(i, logZ, m_led)); /* as the deposit */
#endif
            double Mej_solar = 0, Z_ej = 0;
            for(int kk = 0; kk < STBL_NELEM; kk++) {Mej_solar += m_led[kk];}
            for(int kk = ELEM_C; kk < STBL_NELEM; kk++) {Z_ej += m_led[kk];}
            if(channel == 0 && (rem_type == REM_FSN || rem_type == REM_DBH)) {channel = 5;} /* failed SN: recorded, zero energy */
            double Esne_erg = resolvedism_sn_energy_erg(rem_type == REM_PISN, rem_type == REM_ECSN, rem_type != REM_FSN && rem_type != REM_DBH);
            {
                double yl[6] = {Mej_solar, Z_ej, 0, 0, 0, 0}; const int ke[4] = {ELEM_C, ELEM_O, ELEM_Si, ELEM_Fe};
                for(int q = 0; q < 4; q++) {yl[2 + q] = m_led[ke[q]];}
                double age_l = resolvedism_star_age_yr(i);
                fb_death_row_push(i, channel == 1, (channel == 1) ? REM_WD : rem_type, (channel == 5) ? FB_LOG_FAILED : FB_LOG_EXPLODED,
                                  (channel == 0) ? Esne_erg : 0.0, yl, age_l, get_star_lifetime(Mstar, logM, logZ));
            }

            n_events[channel] += 1;
            M_injected[channel] += Mej_solar;
            M_removed[channel] += Mej_solar;
            Z_injected[channel] += P[i].FB_DepositedZ * UNIT_MASS_IN_SOLAR; /* metals the walk delivered */
            if(channel == 0) { /* SN: resolvedism_sn_energy_erg (1e51; PISN 1e52 and ECSN 5e50 with RESOLVED_ISM_VARIABLE_SN_ENERGY), 0 for FSN/DBH */
                E_injected[channel] += Esne_erg;
            }
            if(channel == 1 && Mej_solar > 0) { /* AGB: delivered momentum and its kinetic energy 1/2 p^2 / M_ej (kick + inelastic heat), as winds */
                double p_agb = P[i].FB_DepositedP * UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s; /* g cm/s */
                dp_injected[1] += p_agb;
                E_injected[1] += 0.5 * p_agb * p_agb / (Mej_solar * SOLAR_MASS_CGS);
            }

            /* P[i].Mass has already been reduced by the measured-coupling pattern in
             * the fb_thermal / fb_momentum walks' out2particle (FIRE-style): star
             * lost exactly Σ_j dM that was deposited on gas neighbors, so total
             * mass is bit-exact conserved per event.  The final mass differs from
             * the table remnant mass by ~1% scatter (whatever Σ wk_j actually was);
             * trading exact-remnant-mass for exact-budget-closure on the injection. */
            if(channel != 1) {
                printf("RESOLVEDISM SN: Task=%d ID=%llu M_init=%.2f M_ej=%.2f M_rem=%.2f rem_type=%d E=%.2e[erg]\n",
                    ThisTask, (unsigned long long)P[i].ID, Mstar, Mej_solar, P[i].Mass * UNIT_MASS_IN_SOLAR, rem_type,
                    Esne_erg);
            } else {
                printf("RESOLVEDISM AGB: Task=%d ID=%llu M_init=%.2f M_ej=%.2f M_rem=%.2f\n",
                    ThisTask, (unsigned long long)P[i].ID, Mstar, Mej_solar, P[i].Mass * UNIT_MASS_IN_SOLAR);
            }
            if(rem_type == REM_PISN) {resolvedism_zero_disrupted_star(i, M_star_old, "PISN");}

#ifdef GALSF_RESOLVEDISM_TYPE_IA
            /* Mark WD remnants as eligible for future Type Ia */
            if(rem_type == REM_WD) {
                P[i].M_drawn_Ia = Mstar; /* store original mass for DTD probability */
            }
#endif
#ifdef KETJU_PN_REMNANT_TAG
            /* Persistent tag of stellar death outcome so KETJU can flag compact remnants
             * (NS, BH variants) as PN-eligible without needing Type 5 / sink promotion. */
            P[i].RemnantType = (signed char) rem_type;
#endif
#ifdef GALSF_RESOLVEDISM_BH_PROMOTION
            /* BH-producing channels: promote to sink after ejecta injection */
            if(rem_type == REM_PPISN || rem_type == REM_FSN || rem_type == REM_DBH) {
                P[i].Type = 5;
                P[i].SinkSubType = 1;
                P[i].Sink_Mass = P[i].Mass;
                P[i].Sink_Formation_Mass = P[i].Mass;
                P[i].Sink_Mdot = 0;
                P[i].Sink_TimeBinGasNeighbor = 0;
                P[i].SwallowID = 0;
                P[i].IndexMapToTempStruc = -1;
                P[i].KernelRadius = All.ForceSoftening[5];
                TreeReconstructFlag = 1;
            }
#endif
        }
#else
        /* Without tables: set to NS mass (1.4 Msun) for SN, 0.6 Msun for AGB */
        n_events[channel] += 1;
        if(STELLA_FB_EVENT(P[i]) == 2) {
            M_removed[channel] += M_star_old - 0.6;
            P[i].Mass = 0.6 / UNIT_MASS_IN_SOLAR;
        } else {
            E_injected[channel] += 1.0e51;
            M_removed[channel] += M_star_old - 1.4;
            P[i].Mass = 1.4 / UNIT_MASS_IN_SOLAR;
        }
#endif

        /* Zero out stellar mass and luminosity to prevent re-trigger and stale inf values */
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        P[i].MstarSampleIMF[0] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL
        P[i].StarModel = STAR_MODEL_STELLA_REMNANT;
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        P[i].UV_luminosity = 0;
        P[i].PE_luminosity = 0;
        P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
        P[i].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
        P[i].OPT_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
        P[i].Lyman_photons_per_sec = 0;
#endif
#endif
#ifdef GALSF_RESOLVEDISM_WINDS
        P[i].WindMassAccum = 0;
        P[i].WindMomentumAccum = 0;
        P[i].M_current_old = 0;
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        P[i].Mstar = 0;
#endif
        P[i].M_at_SN_trigger = -1; /* reset bookkeeping snapshot */
        STELLA_FB_EVENT(P[i]) = -1;
    }
    if(n_events[0]+n_events[1]+n_events[2]+n_events[3]+n_events[4]+n_events[5] > 0) {TreeMomentsStaleFlag = 1;} /* star/gas masses changed */
#ifdef STELLA_SELFCHECK
    STELLA_CHECK_RAN("fb_booked_deposited", n_selfcheck_book);
    { int n_ms = 0, n_bad = 0; /* CellP.Mass authoritative for gas: every Stella gas-mass write must mirror it */
      for(int ig = 0; ig < N_gas; ig++) { if(P[ig].Type != 0) continue; n_ms++;
          int ok_ = fabs((double)P[ig].Mass - (double)CellP[ig].Mass) <= 1.0e-10 * fabs((double)P[ig].Mass);
          if(!ok_ && ++n_bad <= 20) STELLA_CHECK("mass_sync", 0, "ID=%llu P.Mass=%.10e CellP.Mass=%.10e", (unsigned long long)P[ig].ID, (double)P[ig].Mass, (double)CellP[ig].Mass); }
      STELLA_CHECK_RAN("mass_sync", n_ms); }
#endif

    /* ---- Write wind events to WINDinfo.txt ---- */
    {
        int n_wind_total_log = 0;
        MPI_Allreduce(&n_wind_logged, &n_wind_total_log, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(n_wind_total_log > 0) {
            int *wrc = NULL, *wdp = NULL;
            double *aw_x=NULL, *aw_y=NULL, *aw_z=NULL, *aw_ms=NULL, *aw_dm=NULL, *aw_dp=NULL, *aw_zw=NULL;
            long long *aw_id = NULL;
            int *aw_tb = NULL;
            if(ThisTask == 0) {
                wrc = (int *)mymalloc("wrc", NTask * sizeof(int));
                wdp = (int *)mymalloc("wdp", NTask * sizeof(int));
            }
            MPI_Gather(&n_wind_logged, 1, MPI_INT, wrc, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                wdp[0] = 0; for(i=1;i<NTask;i++) wdp[i] = wdp[i-1]+wrc[i-1];
                aw_x  = (double *)mymalloc("aw_x",  n_wind_total_log * sizeof(double));
                aw_y  = (double *)mymalloc("aw_y",  n_wind_total_log * sizeof(double));
                aw_z  = (double *)mymalloc("aw_z",  n_wind_total_log * sizeof(double));
                aw_ms = (double *)mymalloc("aw_ms", n_wind_total_log * sizeof(double));
                aw_dm = (double *)mymalloc("aw_dm", n_wind_total_log * sizeof(double));
                aw_dp = (double *)mymalloc("aw_dp", n_wind_total_log * sizeof(double));
                aw_zw = (double *)mymalloc("aw_zw", n_wind_total_log * sizeof(double));
                aw_id = (long long *)mymalloc("aw_id", n_wind_total_log * sizeof(long long));
                aw_tb = (int *)mymalloc("aw_tb", n_wind_total_log * sizeof(int));
            }
            MPI_Gatherv(wind_log_x,     n_wind_logged, MPI_DOUBLE,    aw_x,  wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_y,     n_wind_logged, MPI_DOUBLE,    aw_y,  wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_z,     n_wind_logged, MPI_DOUBLE,    aw_z,  wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_mstar, n_wind_logged, MPI_DOUBLE,    aw_ms, wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_dm,    n_wind_logged, MPI_DOUBLE,    aw_dm, wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_dp,    n_wind_logged, MPI_DOUBLE,    aw_dp, wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_zw,    n_wind_logged, MPI_DOUBLE,    aw_zw, wrc, wdp, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_id,    n_wind_logged, MPI_LONG_LONG, aw_id, wrc, wdp, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Gatherv(wind_log_tb,    n_wind_logged, MPI_INT,       aw_tb, wrc, wdp, MPI_INT,       0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                for(i = 0; i < n_wind_total_log; i++)
                    fprintf(FdWINDinfo, "%12.6f  %10.6f %10.6f %10.6f  %10lld  %8.3f  %10.4f  %12.4e  %12.4e  %3d\n",
                        All.Time, aw_x[i], aw_y[i], aw_z[i], aw_id[i], aw_ms[i], aw_dm[i], aw_dp[i], aw_zw[i], aw_tb[i]);
                fflush(FdWINDinfo);
                myfree(aw_tb); myfree(aw_id); myfree(aw_zw); myfree(aw_dp); myfree(aw_dm);
                myfree(aw_ms); myfree(aw_z); myfree(aw_y); myfree(aw_x);
                myfree(wdp); myfree(wrc);
            }
        }
    }
    /* Free wind logging buffers (LIFO) */
    myfree(wind_log_tb); myfree(wind_log_id); myfree(wind_log_zw); myfree(wind_log_dp); myfree(wind_log_dm);
    myfree(wind_log_mstar); myfree(wind_log_z); myfree(wind_log_y); myfree(wind_log_x);

    ESTEP_END(ESTEP_FB_POST);
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    { /* test: wind metals booked (FeedbackBudget) vs delivered by the walk this step [Msun] */
        double loc[2] = {Z_injected[2], FB_TestWindZcoupled * UNIT_MASS_IN_SOLAR}, glob[2];
        MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0 && (glob[0] > 0 || glob[1] > 0)) {printf("WINDZLEDGER t=%.17e Z_book=%.17e Z_dep=%.17e\n", All.Time, glob[0], glob[1]); fflush(stdout);}
        FB_TestWindZcoupled = 0;
    }
#endif
    /* energy step trace: what each pass books it deposits (E_injected is erg, on the star's rank) */
    ESTEP_BOOK(EBK_FB_WIND, E_injected[2] / UNIT_ENERGY_IN_CGS); ESTEP_BOOK(EBK_FB_AGB, E_injected[1] / UNIT_ENERGY_IN_CGS);
    ESTEP_BOOK(EBK_FB_SN, E_injected[0] / UNIT_ENERGY_IN_CGS); ESTEP_BOOK(EBK_FB_IA, E_injected[4] / UNIT_ENERGY_IN_CGS);
    ESTEP_BOOK(EBK_FB_FAILED, E_injected[5] / UNIT_ENERGY_IN_CGS);

    /* ---- Pass 2: radiation pressure (momentum) ---- */
    ESTEP_BEGIN(ESTEP_FB_RADP);
#ifdef COSMIC_RAY_FLUID
    E_cr3 = resolvedism_cr_energy_local();
#endif
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    E_mech3 = resolvedism_mech_energy_local();
    RadPressure_dp_thisStep = 0;
    /* Radpressure is mass-neutral but go through the same brackets for safety;
     * any FB_M_coupled_pending should stay 0 (no P[j].Mass += dM happens). */
    /* Radpressure is mass-neutral, but follow same two-pass pattern for the kernel-weighted
     * momentum kicks → Σ wk = 1 exactly → Σ Δp_j = p_radpressure exactly. */
    FB_ZERO_AWS();
    for(int ii : ActiveParticleList) {if(STELLA_OWNS_STAR(P[ii])) {P[ii].FB_DepositedP = 0;}}
    resolvedism_fb_momentum_calc(-2);   /* weighting pre-pass for radpressure */
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    for(int ii : ActiveParticleList) {if(STELLA_OWNS_STAR(P[ii]) && All.Time < All.TestFBNoGasUntilTime) {P[ii].FB_Area_weighted_sum = 0;}} /* test: no gas in reach */
#endif
    resolvedism_fb_momentum_calc(1);    /* radpressure injection (IR-trapping boost; DIRECT term
                                         * offloaded to the per-cell tree-flux pass below) */
    resolvedism_fb_apply_recoil();
    /* channel-3 ledger: the momentum each star's walk delivered (sum |dp_j|, all returns) */
    for(int i : ActiveParticleList) {
        if(!STELLA_OWNS_STAR(P[i]) || P[i].FB_DepositedP <= 0) continue;
        RadPressure_dp_thisStep += P[i].FB_DepositedP * (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);
        n_events[3] += 1;
    }
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    { /* test: star-side radp booked (channel 3, every star) vs delivered this step, full precision (code units) */
        double loc[7] = {0}, glob[7];
        loc[0] = RadPressure_dp_thisStep / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);
        for(int i : ActiveParticleList) {
            if(!STELLA_OWNS_STAR(P[i]) || P[i].Mass <= 0) continue;
            if(P[i].FB_DepositedP > 0) {loc[6] += resolvedism_radp_starside_dp_code(i);} /* the budget of each star that deposited */
            double aws = P[i].FB_Area_weighted_sum, m2 = 0;
            if(aws <= 0) continue;
            for(int k_ = 0; k_ < 3; k_++) {m2 += (P[i].FB_VecSum[k_] / aws) * (P[i].FB_VecSum[k_] / aws);}
            loc[5] += m2;
        }
        loc[1] = FB_TestRadpPabs; for(int k_ = 0; k_ < 3; k_++) {loc[2 + k_] = FB_TestRadpPvec[k_];}
        MPI_Allreduce(loc, glob, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0 && (glob[0] > 0 || glob[1] > 0)) {
            double pv = sqrt(glob[2]*glob[2] + glob[3]*glob[3] + glob[4]*glob[4]);
            printf("RADPLEDGER t=%.17e p_book=%.17e p_abs=%.17e p_vec=%.17e m2=%.17e p_budget=%.17e\n", All.Time, glob[0], glob[1], pv, glob[5], glob[6]);
            fflush(stdout);
        }
        FB_TestRadpPabs = 0; FB_TestRadpPvec[0] = FB_TestRadpPvec[1] = FB_TestRadpPvec[2] = 0;
    }
#endif
#ifdef GALSF_RESOLVEDISM_RADPRESSURE_TREERAD
    /* LEBRON-style DIRECT radp: per gas cell, from the tree attenuated per-pixel
     * fluxes finalized in the gravity walk (deposited locally in the absorbing cell,
     * with correct direction + shadowing). Also accumulates into RadPressure_dp_thisStep
     * so channel-3 (radpres) of the FeedbackBudget ledger stays complete. */
    resolvedism_radpressure_treerad();
#endif

    dp_injected[3] = RadPressure_dp_thisStep; /* delivered: star-side walk plus the per-cell term */
    E_coupled[3] = resolvedism_mech_energy_local() - E_mech3;
#else
    E_mech3 = resolvedism_mech_energy_local();
#endif
    ESTEP_END(ESTEP_FB_RADP);
    /* split a shared pass's measured coupling by booked energy (else by events) */
    double gE_inj[NFBCH], gN_ev[NFBCH]; /* split weights must be global: receiver-only ranks book no events */
    MPI_Allreduce(E_injected, gE_inj, NFBCH, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(n_events, gN_ev, NFBCH, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto fb_split = [&](double dE, int a, int b) {
        double wa = gE_inj[a], wb = gE_inj[b];
        if(wa + wb <= 0) {wa = gN_ev[a]; wb = gN_ev[b];}
        if(wa + wb <= 0) {wa = 1; wb = 0;}
        E_coupled[a] += dE * wa / (wa + wb); E_coupled[b] += dE * wb / (wa + wb);
    };
#ifdef GALSF_RESOLVEDISM_WINDS_CONTINUOUS
    E_coupled[1] += E_mech1 - E_mech0; E_coupled[2] += E_mech2 - E_mech1;
#else
    fb_split(E_mech2 - E_mech0, 2, 1);
#endif
    if(gE_inj[0] + gE_inj[4] + gN_ev[0] + gN_ev[4] <= 0 && gN_ev[5] > 0) {E_coupled[5] += E_mech3 - E_mech2;} /* failed SNe deposit ejecta: their own channel */
    else {fb_split(E_mech3 - E_mech2, 0, 4);}
#ifdef SGCHEM
    n_events[6] = PIHeatPendingCells; E_injected[6] = E_coupled[6] = PIHeatPendingErg; /* heated by the chemistry since the last row */
    PIHeatPendingErg = PIHeatPendingCells = 0;
#endif

    fb_death_rows_flush();

    /* ---- MPI reduce and write feedback budget ---- */
    double glob_n[NFBCH], glob_Mi[NFBCH], glob_Mr[NFBCH], glob_E[NFBCH], glob_dp[NFBCH], glob_Z[NFBCH], glob_Ec[NFBCH];
    MPI_Reduce(n_events, glob_n, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(M_injected, glob_Mi, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(M_removed, glob_Mr, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(E_injected, glob_E, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(dp_injected, glob_dp, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Z_injected, glob_Z, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(E_coupled, glob_Ec, NFBCH, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef COSMIC_RAY_FLUID
    {double dcr = E_cr3 - E_cr0, dcr_glob = 0; MPI_Reduce(&dcr, &dcr_glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); if(ThisTask == 0) {CumulCRInjectedErg += dcr_glob;}}
#endif

    if(ThisTask == 0) {
        for(int ch = 0; ch < NFBCH; ch++) {
            if(glob_n[ch] > 0 || glob_Ec[ch] != 0) {
                fprintf(FdFeedbackBudget, "%12.6f  %d  %6.0f  %12.4f  %12.4f  %12.4e  %12.4e  %12.4f  %12.4e\n",
                    All.Time, ch, glob_n[ch], glob_Mi[ch], glob_Mr[ch],
                    glob_E[ch], glob_dp[ch], glob_Z[ch], glob_Ec[ch]);
                if(ch == 6) continue; /* PI cumulated per rank by the chemistry (CumulPIHeatErg) */
                CumulFeedbackEnergy += glob_E[ch];
                CumulFeedbackCoupled += glob_Ec[ch];
                CumulFeedbackMass += glob_Mi[ch];
                CumulFeedbackMetals += glob_Z[ch];
                if(ch == 0) CumulSNe += (int)glob_n[ch];
                if(ch == 1) CumulAGB += (int)glob_n[ch];
                if(ch == 4) CumulIa += (int)glob_n[ch];
            }
        }
        fflush(FdFeedbackBudget);
        /* Print budget summary to stdout (domain decomp steps only) */
        if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {
            const char *chname[NFBCH] = {"SN", "AGB", "wind", "radpres", "Ia", "failed", "PI"};
            for(int ch = 0; ch < NFBCH; ch++) {
                if(glob_n[ch] > 0) {
                    printf("RESOLVEDISM BUDGET: ch=%s n=%.0f M_inj=%.4f M_rem=%.4f E=%.2e dp=%.2e Z_inj=%.4f [Msun/erg/cgs] at t=%g\n",
                        chname[ch], glob_n[ch], glob_Mi[ch], glob_Mr[ch], glob_E[ch], glob_dp[ch], glob_Z[ch], All.Time);
                }
            }
            fflush(stdout);
        }
        /* Radpressure summary (only on domain decomp steps) */
        if(glob_n[3] > 0 && All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {
            printf("RESOLVEDISM RADPRESSURE: %.0f active stars, dp_tot=%.3e [cgs] at t=%g\n",
                   glob_n[3], glob_dp[3], All.Time);
            fflush(stdout);
        }
    }

    /* ---- Stellar census + cumulative summary (domain decomp steps only) ---- */
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
    {
        int local_census[5] = {0,0,0,0,0}; /* bins: M<1, 1-8, 8-20, 20-40, >40 */
        for(int i : ActiveParticleList) {
            if(!STELLA_OWNS_STAR(P[i]) || P[i].Mass <= 0) continue;
            double ms = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            if(P[i].sampled) ms = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
            ms = P[i].Mstar;
#endif
            if(ms <= 0) continue;
            if(ms < 1.0) local_census[0]++;
            else if(ms < 8.0) local_census[1]++;
            else if(ms < 20.0) local_census[2]++;
            else if(ms < 40.0) local_census[3]++;
            else local_census[4]++;
        }
        int glob_census[5];
        MPI_Reduce(local_census, glob_census, 5, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(ThisTask == 0) {
            int ntot = glob_census[0]+glob_census[1]+glob_census[2]+glob_census[3]+glob_census[4];
            printf("RESOLVEDISM CENSUS: %d stars alive (<1:%d  1-8:%d  8-20:%d  20-40:%d  >40:%d) at t=%g\n",
                ntot, glob_census[0], glob_census[1], glob_census[2], glob_census[3], glob_census[4], All.Time);
            printf("RESOLVEDISM CUMUL: %d SNe + %d AGB + %d Ia | M_returned=%.1f Msun | Z_returned=%.1f Msun | E_total=%.2e erg\n",
                CumulSNe, CumulAGB, CumulIa, CumulFeedbackMass, CumulFeedbackMetals, CumulFeedbackEnergy);
            fflush(stdout);
        }

        /* Gas phase diagnostics */
        double local_gas[6] = {0,0,0,0,0,0}; /* M_gas, M_metals, rho_max, T_max, nH_max, u_max */
        for(i = 0; i < N_gas; i++) {
            if(P[i].Type != 0 || P[i].Mass <= 0) continue;
            double mi = P[i].Mass * UNIT_MASS_IN_SOLAR;
            local_gas[0] += mi;
            double rho_cgs = CellP[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
            double nH = rho_cgs * HYDROGEN_MASSFRAC / PROTONMASS_CGS;
            if(nH > local_gas[4]) local_gas[4] = nH;
            if(CellP[i].Density > local_gas[2]) local_gas[2] = CellP[i].Density;
#ifdef SGCHEM
            if(CellP[i].Temp > local_gas[3]) local_gas[3] = CellP[i].Temp;
#endif
        }
        double glob_gas_sum[2], glob_gas_max[4];
        MPI_Reduce(&local_gas[0], &glob_gas_sum[0], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); /* M_gas */
        MPI_Reduce(&local_gas[1], &glob_gas_sum[1], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); /* M_metals */
        MPI_Reduce(&local_gas[2], &glob_gas_max[0], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /* rho_max */
        MPI_Reduce(&local_gas[3], &glob_gas_max[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /* T_max */
        MPI_Reduce(&local_gas[4], &glob_gas_max[2], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /* nH_max */
        if(ThisTask == 0) {
            printf("RESOLVEDISM GAS: M_gas=%.2e Msun | nH_max=%.2e cm-3 | T_max=%.1f K\n",
                glob_gas_sum[0], glob_gas_max[2], glob_gas_max[1]);
            fflush(stdout);
        }
    }
}
/* code_block_xchange finalize removed — now in resolvedism_fb_thermal.cc and resolvedism_fb_momentum.cc */


#endif /* GALSF_RESOLVEDISM_FB */
