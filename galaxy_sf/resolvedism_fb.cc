#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "resolvedism_fb_shared.h"

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
double CumulFeedbackMass = 0;    /* cumulative mass returned to gas by feedback [Msun] */
double CumulFeedbackMetals = 0;  /* cumulative metals returned to gas by feedback [Msun] */
int CumulSNe = 0, CumulAGB = 0, CumulIa = 0; /* cumulative event counts */
double CumulStarMassFormed = 0;  /* cumulative stellar mass formed [Msun] — set in IMF sampling */
double RadPressure_dp_thisStep = 0; /* total radpressure momentum this step [g*cm/s] */

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
/*  SNe_ThisTimeStep: 1=explosive SN, 2=AGB death, 3=wind, 4=Ia, -1=done    */
/* =========================================================================== */
void resolvedism_determine_SNe(void)
{
    if(All.Time <= 0) return;
    int i;
    int n_sne_local = 0, n_sne_total = 0;
    int n_agb_local = 0, n_agb_total = 0;
    int n_collapse_local = 0;
    int n_wind_local = 0, n_wind_total = 0;

    /* Reset SNe flags for active particles (allows re-entry after wind dump or previous -1 marking) */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
        if(P[i].Type == 4) P[i].SNe_ThisTimeStep = 0;
    }

    /* ---- Wind accumulation for living massive stars ----
     * Principled approach: single source of truth = stellar_elem_ej_wind_cumulative table.
     *   cum_table_now  = Σ_k elem_ej_wind_cumulative(t, k)         (table prediction)
     *   cum_injected   = M_init - P[i].Mass                         (what's actually been removed)
     *   dM_pending     = max(0, cum_table_now - cum_injected)       (what still needs to be injected)
     * P[i].WindMassAccum is overwritten to dM_pending each step (no longer a step-by-step accumulator).
     * P[i].WindMomentumAccum is set to dM_pending * v_wind at the current age. */
#ifdef GALSF_RESOLVEDISM_WINDS
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4 || P[i].Mass <= 0 || P[i].SNe_ThisTimeStep != 0) continue;
        double Mstar, logM, logZ;
        if(!get_star_info(i, &Mstar, &logM, &logZ)) continue;
        if(Mstar < 8.0) continue; /* only massive stars have significant winds */

        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) continue;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
        if(star_age_yr >= lifetime_yr) continue; /* dead — handled in death loop below */

        double table_age = get_star_table_age(star_age_yr, logM, logZ);
        if(table_age <= 0) continue; /* PMS: no winds yet */
        double log_age = log10(DMAX(table_age, 100.0));

        double cum_now_total = 0;
        for(int kk = 0; kk < STBL_NELEM; kk++)
            cum_now_total += stellar_elem_ej_wind_cumulative(logM, logZ, log_age, kk);
        double M_init_solar = P[i].MstarSampleIMF[0];
        double cum_injected = M_init_solar - P[i].Mass * UNIT_MASS_IN_SOLAR;
        double dM_pending = cum_now_total - cum_injected;
        if(dM_pending < 0) dM_pending = 0; /* interp noise — wait for table to catch up */

        double v_w = stellar_v_wind(logM, logZ, log_age);
        if(v_w < 10.0) v_w = 10.0;

        P[i].WindMassAccum = dM_pending;                /* pending mass to inject */
        P[i].WindMomentumAccum = dM_pending * v_w;      /* matching momentum (v at current age) */

        /* Trigger: mass-dependent fractional threshold. Low-mass (8 Msun): 1%; high-mass (300 Msun): ~15%. */
        double wind_frac = 0.01 + 0.19 * DMAX(0, (Mstar - 8.0)) / (350.0 - 8.0);
        if(wind_frac > 0.20) wind_frac = 0.20;
        if(dM_pending > wind_frac * Mstar) {
            P[i].SNe_ThisTimeStep = 3;
            n_wind_local++;
        }
    }
#endif

    /* first pass: count local death events so we can allocate buffers */
    int n_death_count = 0;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4 || P[i].Mass <= 0 || P[i].SNe_ThisTimeStep != 0) continue;
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) continue;
        double Mstar, logM, logZ;
        if(!get_star_info(i, &Mstar, &logM, &logZ)) continue;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
        if(All.Time < All.TimeInstantSN && Mstar >= 8.0) {lifetime_yr = 0;}
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
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4 || P[i].Mass <= 0 || P[i].SNe_ThisTimeStep != 0) continue;
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) continue;
        double Mstar, logM, logZ;
        if(!get_star_info(i, &Mstar, &logM, &logZ)) continue;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
        if(All.Time < All.TimeInstantSN && Mstar >= 8.0) {lifetime_yr = 0;}
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
                double M_init_solar = P[i].MstarSampleIMF[0];
                double cum_injected = M_init_solar - P[i].Mass * UNIT_MASS_IN_SOLAR;
                double dM_pending = cum_now_total - cum_injected;
                if(dM_pending < 0) dM_pending = 0;
                /* Threshold above float precision (~1e-15 Msun): otherwise the
                 * death-loop fires a wind event every step that injects zero
                 * mass but consumes the SN-flag path via the `continue` below. */
                if(dM_pending > 1.0e-6) {
                    double v_w = stellar_v_wind(logM, logZ, log_age_end);
                    if(v_w < 10.0) v_w = 10.0;
                    P[i].WindMassAccum = dM_pending;
                    P[i].WindMomentumAccum = dM_pending * v_w;
                    P[i].SNe_ThisTimeStep = 3;
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
            P[i].SNe_ThisTimeStep = 2; /* AGB: mass+metals, no energy */
            n_agb_local++;
        } else {
            P[i].SNe_ThisTimeStep = 1; /* SN: energy+mass+metals */
            n_sne_local++;
        }
        /* Snapshot pre-walk mass for bookkeeping in resolvedism_inject_fb_energy().
         * The FB walks below will drain P[i].Mass down to ~rem_mass, so a later
         * read of P[i].Mass would give Mej~0.  This snapshot preserves the
         * correct value for the FeedbackBudget log. */
        P[i].M_at_SN_trigger = (MyFloat)(P[i].Mass * UNIT_MASS_IN_SOLAR);
        /* NOTE: do NOT zero MstarSampleIMF here — particle2in needs it */
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
#ifndef GALSF_RESOLVEDISM_WINDS
            double t_end_yr_log = stellar_lifetime(logM, logZ);
            double log_age_end_log = log10(DMAX(t_end_yr_log, 100.0));
#endif
            for(int kk = 0; kk < STBL_NELEM; kk++) {
                double me = stellar_elem_ej_SN(logM, logZ, kk);
#ifndef GALSF_RESOLVEDISM_WINDS
                me += stellar_elem_ej_wind_cumulative(logM, logZ, log_age_end_log, kk);
#endif
                mej_log += me;
                if(kk >= ELEM_C) zej_log += me;
                if(kk == ELEM_C)  yC_log  = me;
                if(kk == ELEM_O)  yO_log  = me;
                if(kk == ELEM_Si) ySi_log = me;
                if(kk == ELEM_Fe) yFe_log = me;
            }
            /* LEDGER FIX (2026-07-03): log what fb_thermal ACTUALLY injects, not the raw
             * table values. The injector rescales all yields by
             *   scale = (M_pre - rem_mass) / Mej_table
             * (heavy-fallback remnants at low Z swallow most of the nominal ejecta:
             * rem ~ M_star for 25-60 Msun at Z~1e-3 -> scale ~ 0 for FSN). Logging the
             * unscaled table values made the metal ledger over-count by ~20% vs the
             * double-precision M_Z tracker. Mirror fb_thermal.cc exactly. */
            {
                double M_pre_log = P[i].M_at_SN_trigger; /* Msun, pre-walk */
                double rem_log = rem_mass;
                double Mej_actual_log = DMAX(M_pre_log - rem_log, 0);
                double scale_log = (mej_log > 0) ? Mej_actual_log / mej_log : 0;
                mej_log *= scale_log; zej_log *= scale_log;
                yC_log *= scale_log; yO_log *= scale_log; ySi_log *= scale_log; yFe_log *= scale_log;
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
    MPI_Allreduce(&n_sne_local, &n_sne_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&n_agb_local, &n_agb_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int n_collapse_total = 0;
    MPI_Allreduce(&n_collapse_local, &n_collapse_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#ifdef GALSF_RESOLVEDISM_WINDS
    MPI_Allreduce(&n_wind_local, &n_wind_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(n_sne_total > 0 || n_agb_total > 0 || n_collapse_total > 0 || n_wind_total > 0)
    {
        if(ThisTask == 0) {
            printf("RESOLVEDISM FB: %d SNe + %d AGB + %d collapses + %d wind-inj at t=%g\n",
                   n_sne_total, n_agb_total, n_collapse_total, n_wind_total, All.Time);
            fflush(stdout);
        }
    }

    int n_logged_total = 0;
    MPI_Allreduce(&n_logged, &n_logged_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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
                if(all_remtype[i] == REM_WD) {
                    fprintf(FdAGBinfo, "%12.6f  %10.6f %10.6f %10.6f  %10.3f %12.6e  %10lld  %8.3f  %8.3f  %8.3f  %8.4f %8.4f %8.4f %8.4f  %.6e  %.6e  %3d\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_u[i], all_rho[i],
                        all_id[i], all_mstar[i], all_mej[i], all_zej[i],
                        all_yC[i], all_yO[i], all_ySi[i], all_yFe[i],
                        all_age[i], all_lifetime[i], all_timebin[i]);
                } else {
                    fprintf(FdSNinfo, "%12.6f  %10.6f %10.6f %10.6f  %10.3f %12.6e  %10lld  %8.3f  %d  %8.3f  %8.3f  %8.4f %8.4f %8.4f %8.4f  %.6e  %.6e  %3d\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_u[i], all_rho[i],
                        all_id[i], all_mstar[i], all_remtype[i], all_mej[i], all_zej[i],
                        all_yC[i], all_yO[i], all_ySi[i], all_yFe[i],
                        all_age[i], all_lifetime[i], all_timebin[i]);
                }
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
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            if(P[i].Type != 4 || P[i].Mass <= 0) continue;
            if(P[i].SNe_ThisTimeStep != 0) continue; /* already flagged for something else */
            if(P[i].M_drawn_Ia <= 0) continue; /* not a WD remnant */

            double star_age_Gyr = evaluate_stellar_age_Gyr(i);
            if(star_age_Gyr < IA_T_MIN_GYR) continue; /* below minimum delay time */

            /* DTD probability: P = (IA_DTD_NORM / t) * M_drawn * dt */
            double dt_Gyr = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i) * UNIT_TIME_IN_GYR;
            if(dt_Gyr <= 0) continue;
            double P_Ia = (IA_DTD_NORM / star_age_Gyr) * P[i].M_drawn_Ia * dt_Gyr;

            /* Stochastic check */
            double rn = get_random_number(P[i].ID + 7 * ThisTask + 13 * All.NumCurrentTiStep);
            if(rn < P_Ia) {
                P[i].SNe_ThisTimeStep = 4; /* Type Ia */
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
            for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
                if(P[i].Type == 4 && P[i].SNe_ThisTimeStep == 4) {
                    ia_x[n_ia_logged] = P[i].Pos[0];
                    ia_y[n_ia_logged] = P[i].Pos[1];
                    ia_z[n_ia_logged] = P[i].Pos[2];
                    ia_u[n_ia_logged] = P[i].InternalEnergyAroundParticle;
                    ia_rho[n_ia_logged] = P[i].DensityAroundParticle;
                    ia_mstar[n_ia_logged] = P[i].MstarSampleIMF[0];
                    ia_id[n_ia_logged] = (long long)P[i].ID;
                    ia_age[n_ia_logged] = evaluate_stellar_age_Gyr(i) * 1.0e9;
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
                for(i = 0; i < n_ia_total; i++) {
                    fprintf(FdSNinfo, "%12.6f  %10.6f %10.6f %10.6f  %10.3f %12.6e  %10lld  %8.3f  %d  %8.3f  %8.3f  %.6e  %.6e\n",
                        All.Time, all_ia_x[i], all_ia_y[i], all_ia_z[i], all_ia_u[i], all_ia_rho[i],
                        all_ia_id[i], all_ia_mstar[i], 7, IA_EJECTA_MASS, 0.0, all_ia_age[i], 0.0); /* 7=TypeIa, M_ej=Mch, Z_ej=0(Fe-peak in budget), no lifetime */
                }
                fflush(FdSNinfo);
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
static void resolvedism_fb_serialized_pass(void (*calc_fn)(int), int flagA, int flagB)
{
    int ii;
    /* collect local candidates (flagged donors eligible for this pass) */
    int ncand = 0, ccap = 256;
    int *cand = (int *)mymalloc("fbser_cand", ccap * sizeof(int));
    for(ii = FirstActiveParticle; ii >= 0; ii = NextActiveParticle[ii]) {
        if(P[ii].Type != 4) continue;
        if(P[ii].SNe_ThisTimeStep != flagA && P[ii].SNe_ThisTimeStep != flagB) continue;
        if(P[ii].KernelRadius <= 0 || P[ii].NumNgb <= 0) continue;
        if(ncand >= ccap) {ccap *= 2; cand = (int *)myrealloc_movable(cand, ccap * sizeof(int));}
        cand[ncand++] = ii;
    }
    char *done = (char *)mymalloc("fbser_done", (ncand > 0 ? ncand : 1) * sizeof(char));
    memset(done, 0, (ncand > 0 ? ncand : 1) * sizeof(char));

    while(1)
    {
        /* pick globally-smallest unprocessed event ID (sentinel = max) */
        unsigned long long my_min = ~0ULL, glob_min = ~0ULL; int my_idx = -1;
        for(ii = 0; ii < ncand; ii++) {
            if(done[ii]) continue;
            if((unsigned long long)P[cand[ii]].ID < my_min) {my_min = (unsigned long long)P[cand[ii]].ID; my_idx = ii;}
        }
        MPI_Allreduce(&my_min, &glob_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        if(glob_min == ~0ULL) break; /* no events left anywhere */
        FB_SerialEventID = (MyIDType)glob_min;
        int i_owner = (my_idx >= 0 && (unsigned long long)P[cand[my_idx]].ID == glob_min) ? cand[my_idx] : -1;

        /* fresh weighting for THIS event against current neighbor masses,
         * with bubble retry (kernel growth) if no gas weight was found */
        int try_num, ok_local, ok_glob;
        for(try_num = 0; try_num < 4; try_num++) {
            for(ii = FirstActiveParticle; ii >= 0; ii = NextActiveParticle[ii]) {
                if(P[ii].Type == 4) P[ii].FB_Area_weighted_sum = 0;
            }
            calc_fn(-1); /* weighting pre-pass (collective) */
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
        calc_fn(0); /* injection for this single event (collective) */

        if(i_owner >= 0) {done[my_idx] = 1;} /* owner retires the event */
    }
    FB_SerialEventID = 0; /* release: subsequent passes see all particles */
    myfree(done);
    myfree(cand);
}

void resolvedism_inject_fb_energy(void)
{
    int ii;
    /* STARFORGE two-pass pattern (adapted from mechanical_fb_calc_toplevel):
     *   1. Weighting pre-pass walks neighbors and measures Σ_j (Mass_j × kernel.wk)
     *      onto P[i].FB_Area_weighted_sum
     *   2. Injection pass uses that measured sum as denominator → Σwk = 1 exactly
     *   → all per-event sums (Mej, Esne, p_ejecta, yields[k]) conserve bit-exact.
     * SERIALIZED PER EVENT (2026-07-03): each event re-measures its weights against
     * the current neighbor masses immediately before its own injection, so same-step
     * events sharing neighbors no longer break Σwk = 1 (see
     * resolvedism_fb_serialized_pass above). */
    #define FB_ZERO_AWS() do { for(ii = FirstActiveParticle; ii >= 0; ii = NextActiveParticle[ii]) { if(P[ii].Type == 4) P[ii].FB_Area_weighted_sum = 0; } } while(0)

    /* ---- Pass 0: wind + AGB (momentum + mass + metals).  Fires first so final
     *      winds happen before the star explodes as SN. Serialized per event. ---- */
    MBARY_STEP("pre_fb_mom_p0");
    resolvedism_fb_serialized_pass(resolvedism_fb_momentum_calc, 2, 3);
    MBARY_STEP("post_fb_mom_p0");

    /* ---- Pass 1: SN + Type Ia (mass + thermal energy + metals). Serialized per event. ---- */
    MBARY_STEP("pre_fb_thermal");
    resolvedism_fb_serialized_pass(resolvedism_fb_thermal_calc, 1, 4);
    MBARY_STEP("post_fb_thermal");

    /* Budget tracking: local accumulators for [0]=SN, [1]=AGB, [2]=wind, [3]=radpressure, [4]=Ia */
    double n_events[5] = {0,0,0,0,0};
    double M_injected[5] = {0,0,0,0,0};   /* ejecta mass [Msun] */
    double M_removed[5] = {0,0,0,0,0};    /* mass removed from star [Msun] */
    double E_injected[5] = {0,0,0,0,0};   /* energy [erg] */
    double dp_injected[5] = {0,0,0,0,0};  /* |momentum| [g*cm/s] */
    double Z_injected[5] = {0,0,0,0,0};   /* metals injected [Msun] */

    /* Wind event logging buffers */
    int n_wind_logged = 0, n_wind_alloc = 64;
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
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef GALSF_RESOLVEDISM_WINDS
        /* Wind injection complete: tally budget only.  Star mass has already
         * been reduced by the measured-coupling out2particle in the fb_momentum
         * walk above (out->M_coupled), so do NOT reduce P[i].Mass here — that
         * would double-deduct. */
        if(P[i].SNe_ThisTimeStep == 3) {
            double dM_wind = P[i].WindMassAccum; /* Msun, for budget tracking */
            double dp_wind = P[i].WindMomentumAccum * SOLAR_MASS_CGS * 1.0e5; /* g*cm/s */

            n_events[2] += 1;
            M_injected[2] += dM_wind;
            M_removed[2] += dM_wind;
            dp_injected[2] += dp_wind;
            /* Wind metals: estimate from surface abundances */
            double Mstar_w, logM_w, logZ_w;
            if(get_star_info(i, &Mstar_w, &logM_w, &logZ_w)) {
                double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
                double table_age_w = get_star_table_age(star_age_yr, logM_w, logZ_w);
                double log_age_w = log10(DMAX(table_age_w, 100.0));
                double Z_wind = 0;
                for(int kk = ELEM_C; kk < STBL_NELEM; kk++) {
                    Z_wind += stellar_surface_abundance(logM_w, logZ_w, log_age_w, kk) * dM_wind;
                }
                Z_injected[2] += Z_wind;
            }

            /* Log wind event */
            if(n_wind_logged < n_wind_alloc) {
                double Z_w_log = 0;
                double Mstar_wl, logM_wl, logZ_wl;
                if(get_star_info(i, &Mstar_wl, &logM_wl, &logZ_wl)) {
                    double age_wl = evaluate_stellar_age_Gyr(i) * 1.0e9;
                    double table_age_wl = get_star_table_age(age_wl, logM_wl, logZ_wl);
                    double lage_wl = log10(DMAX(table_age_wl, 100.0));
                    for(int kk = ELEM_C; kk < STBL_NELEM; kk++)
                        Z_w_log += stellar_surface_abundance(logM_wl, logZ_wl, lage_wl, kk) * dM_wind;
                }
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
                double age_yr_now = evaluate_stellar_age_Gyr(i) * 1.0e9;
                double Mstar_log = P[i].MstarSampleIMF[0];
                double logM_log  = log10(DMAX(Mstar_log, 0.08));
                double logZ_log  = log10(DMAX(P[i].BirthMetallicity, 1e-10));
                double table_age_now = get_star_table_age(age_yr_now, logM_log, logZ_log);
                P[i].last_wind_log_age = log10(DMAX(table_age_now, 100.0));
            }
            P[i].SNe_ThisTimeStep = -1;
            continue;
        }
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
        /* Type Ia: WD fully disrupted.  Actual injected mass is the WD progenitor's
         * current particle mass (P[i].Mass), not the canonical Chandrasekhar mass,
         * because single-star sampling can leave the WD below 1.378 Msun after prior
         * wind/AGB phases.  Yields are rescaled by (actual / Chandrasekhar) ratio. */
        if(P[i].SNe_ThisTimeStep == 4) {
            /* Use pre-walk snapshot — P[i].Mass has already been drained by thermal walk */
            double M_WD = (P[i].M_at_SN_trigger > 0) ? (double)P[i].M_at_SN_trigger
                                                     : P[i].Mass * UNIT_MASS_IN_SOLAR;
            double yield_rescale = M_WD / IA_EJECTA_MASS;
            n_events[4] += 1;
            M_injected[4] += M_WD;     /* actual injected mass (= WD mass) */
            M_removed[4] += M_WD;      /* particle mass removed */
            E_injected[4] += IA_ENERGY_ERG;
            double Z_ia = 0;
            for(int kk = ELEM_C; kk < STBL_NELEM; kk++) Z_ia += stellar_type_ia_yield(kk) * yield_rescale;
            Z_injected[4] += Z_ia;
            printf("RESOLVEDISM TYPE_IA: Task=%d ID=%llu M_WD=%.3f E=%.2e[erg]\n",
                ThisTask, (unsigned long long)P[i].ID, M_WD, IA_ENERGY_ERG);
            /* Do NOT set P[i].Mass = 0 here — the fb_thermal walk's out2particle
             * has already reduced star mass via measured-coupling (M_coupled).
             * Forcing 0 would silently absorb any residual from Σ wk_j ≠ 1, hiding
             * a leak.  WD is effectively disrupted: star mass is whatever the
             * walk actually deposited to the gas neighbors (~0). */
            P[i].M_drawn_Ia = 0; /* no longer eligible */
            P[i].M_at_SN_trigger = -1; /* reset bookkeeping snapshot */
            P[i].SNe_ThisTimeStep = -1;
            continue;
        }
#endif
        if(P[i].SNe_ThisTimeStep != 1 && P[i].SNe_ThisTimeStep != 2) continue;

        double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        Mstar = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar = P[i].Mstar;
#endif

        int channel = (P[i].SNe_ThisTimeStep == 2) ? 1 : 0; /* 0=SN, 1=AGB */
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
            double rem_mass = stellar_remnant_mass(logM, logZ);
            if(rem_type == REM_PISN) rem_mass = 0; /* complete disruption */

            /* Mej_actual = (current particle mass) - rem_mass. Matches the injection and ensures
             * exact mass conservation. Z_ej is table-shape rescaled so total = Mej_actual. */
#ifndef GALSF_RESOLVEDISM_WINDS
            double t_end_yr_acc = stellar_lifetime(logM, logZ);
            double log_age_end_acc = log10(DMAX(t_end_yr_acc, 100.0));
#endif
            double Mej_table = 0, Z_ej_table = 0;
            for(int kk = 0; kk < STBL_NELEM; kk++) {
                double m_k;
                if(channel == 0) {
                    m_k = stellar_elem_ej_SN(logM, logZ, kk);
#ifndef GALSF_RESOLVEDISM_WINDS
                    m_k += stellar_elem_ej_wind_cumulative(logM, logZ, log_age_end_acc, kk);
#endif
                } else {
                    m_k = stellar_elem_ej_AGB(logM, logZ, kk);
                }
                Mej_table += m_k;
                if(kk >= ELEM_C) Z_ej_table += m_k;
            }
            double Mej_actual = M_star_old - rem_mass;
            if(Mej_actual < 0) Mej_actual = 0;
            double scale_diag = (Mej_table > 0) ? Mej_actual / Mej_table : 0;
            double Mej_solar = Mej_actual;
            double Z_ej = Z_ej_table * scale_diag;

            n_events[channel] += 1;
            M_injected[channel] += Mej_solar;
            M_removed[channel] += Mej_solar;
            Z_injected[channel] += Z_ej;
            if(channel == 0) { /* SN: has energy. Uniform 1e51 for explosive remnants; 0 for FSN/DBH (no explosion). */
                double Esne_erg = (rem_type == REM_FSN || rem_type == REM_DBH) ? 0.0 : 1.0e51;
                E_injected[channel] += Esne_erg;
            }

            /* P[i].Mass has already been reduced by the measured-coupling pattern in
             * the fb_thermal / fb_momentum walks' out2particle (FIRE-style): star
             * lost exactly Σ_j dM that was deposited on gas neighbors, so total
             * mass is bit-exact conserved per event.  The final mass differs from
             * the table remnant mass by ~1% scatter (whatever Σ wk_j actually was);
             * trading exact-remnant-mass for exact-budget-closure on the injection. */
            if(channel == 0) {
                printf("RESOLVEDISM SN: Task=%d ID=%llu M_init=%.2f M_ej=%.2f M_rem=%.2f rem_type=%d E=%.2e[erg]\n",
                    ThisTask, (unsigned long long)P[i].ID, Mstar, Mej_solar, rem_mass, rem_type,
                    (rem_type==REM_FSN||rem_type==REM_DBH)?0.0:1.0e51);
            } else {
                printf("RESOLVEDISM AGB: Task=%d ID=%llu M_init=%.2f M_ej=%.2f M_rem=%.2f\n",
                    ThisTask, (unsigned long long)P[i].ID, Mstar, Mej_solar, rem_mass);
            }

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
        if(P[i].SNe_ThisTimeStep == 2) {
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
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        P[i].UV_luminosity = 0;
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
        P[i].SNe_ThisTimeStep = -1;
    }

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

    /* ---- Pass 2: radiation pressure (momentum) ---- */
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    RadPressure_dp_thisStep = 0;
    /* Radpressure is mass-neutral but go through the same brackets for safety;
     * any FB_M_coupled_pending should stay 0 (no P[j].Mass += dM happens). */
    /* Radpressure is mass-neutral, but follow same two-pass pattern for the kernel-weighted
     * momentum kicks → Σ wk = 1 exactly → Σ Δp_j = p_radpressure exactly. */
    FB_ZERO_AWS();
    resolvedism_fb_momentum_calc(-2);   /* weighting pre-pass for radpressure */
    resolvedism_fb_momentum_calc(1);    /* radpressure injection */

    /* Count radiation pressure events */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
        if(P[i].Type != 4 || P[i].Mass <= 0) continue;
        double Mstar_rp = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[i].sampled) Mstar_rp = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar_rp = P[i].Mstar;
#endif
        if(Mstar_rp >= 2.0) n_events[3] += 1;
    }
    dp_injected[3] = RadPressure_dp_thisStep; /* exact dp from particle2in */
#endif

    /* ---- MPI reduce and write feedback budget ---- */
    double glob_n[5], glob_Mi[5], glob_Mr[5], glob_E[5], glob_dp[5], glob_Z[5];
    MPI_Reduce(n_events,    glob_n,  5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(M_injected,  glob_Mi, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(M_removed,   glob_Mr, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(E_injected,  glob_E,  5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(dp_injected, glob_dp, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(Z_injected,  glob_Z,  5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(ThisTask == 0) {
        for(int ch = 0; ch < 5; ch++) {
            if(glob_n[ch] > 0) {
                /* Radpressure (ch=3): only log on full (domain decomp) steps to avoid flooding */
                if(ch == 3 && All.HighestActiveTimeBin != All.HighestOccupiedTimeBin) continue;
                fprintf(FdFeedbackBudget, "%12.6f  %d  %6.0f  %12.4f  %12.4f  %12.4e  %12.4e  %12.4f\n",
                    All.Time, ch, glob_n[ch], glob_Mi[ch], glob_Mr[ch],
                    glob_E[ch], glob_dp[ch], glob_Z[ch]);
                CumulFeedbackEnergy += glob_E[ch];
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
            const char *chname[5] = {"SN", "AGB", "wind", "radpres", "Ia"};
            for(int ch = 0; ch < 5; ch++) {
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
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4 || P[i].Mass <= 0) continue;
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
#ifdef CHEMCOOL
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
