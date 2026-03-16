#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

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
double CumulStarMassFormed = 0;  /* cumulative stellar mass formed [Msun] — set in IMF sampling */

#ifndef DO_DENSITY_AROUND_NONGAS_PARTICLES
#error "GALSF_RESOLVEDISM_FB requires DO_DENSITY_AROUND_NONGAS_PARTICLES for kernel weights (wt_sum)"
#endif

/* =========================================================================== */
/*  Type Ia DTD constants (Maoz & Mannucci 2012, t^{-1} power-law DTD)        */
/*  R_Ia(t) = IA_DTD_NORM / t[Gyr]  for t > IA_T_MIN_GYR                     */
/*  Normalization: N_Ia/M_formed = 1.3e-3 /Msun integrated over Hubble time.  */
/*  Corrected by 1/f_WD (f_WD~0.76 for Kroupa) since we only apply to WDs.   */
/*  Total ejecta = Chandrasekhar mass (1.378 Msun); yields from W7 model.     */
/* =========================================================================== */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
#define IA_T_MIN_GYR     0.04    /* 40 Myr minimum delay time [Gyr] */
#define IA_N_PER_MSUN    1.3e-3  /* total Ia per Msun formed */
#define IA_F_WD          0.76    /* mass fraction in WD progenitors (Kroupa IMF) */
/* K = (N/M) / ln(t_max/t_min) / f_WD, with t_max=13.8 Gyr, t_min=0.04 Gyr */
/* = 1.3e-3 / ln(345) / 0.76 = 2.93e-4 */
#define IA_DTD_NORM      2.93e-4 /* [Gyr^{-1} Msun^{-1}] rate constant */
#define IA_EJECTA_MASS   1.378   /* total ejecta mass [Msun] (Chandrasekhar mass) */
#define IA_ENERGY_ERG    1.0e51  /* canonical Type Ia energy [erg] */
#endif

/* =========================================================================== */
/*  Helper: get stellar mass and metallicity for the single star in particle i */
/* =========================================================================== */
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

/* =========================================================================== */
/*  Helper: get lifetime for a star, using tables if available                 */
/* =========================================================================== */
static inline double get_star_lifetime(double Mstar, double logM, double logZ)
{
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    return stellar_lifetime(logM, logZ);
#else
    return get_lifetime(Mstar, pow(10.0, logZ));
#endif
}


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

    /* ---- Wind accumulation for living massive stars ---- */
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

        double log_age = log10(DMAX(star_age_yr, 100.0));
        double M_new = stellar_M_current(logM, logZ, log_age);

        if(P[i].M_current_old > 0 && M_new < P[i].M_current_old) {
            double dM = P[i].M_current_old - M_new; /* wind mass lost this step [Msun] */
            double v_w = stellar_v_wind(logM, logZ, log_age); /* km/s */
            if(v_w < 10.0) v_w = 10.0; /* floor: minimum 10 km/s */
            P[i].WindMassAccum += dM;
            P[i].WindMomentumAccum += dM * v_w;
        }
        P[i].M_current_old = M_new;

        /* Check wind injection threshold: 1% of initial stellar mass */
        if(P[i].WindMassAccum > 0.01 * Mstar) {
            P[i].SNe_ThisTimeStep = 3; /* flag for wind injection */
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

        /* FSN/DBH: direct collapse to BH, no explosion */
        if(rem_type == REM_FSN || rem_type == REM_DBH) {
            P[i].Mass = rem_mass / UNIT_MASS_IN_SOLAR;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            P[i].MstarSampleIMF[0] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
            P[i].Mstar = 0;
#endif
            P[i].SNe_ThisTimeStep = -1; /* mark as done, no explosion */
#ifdef GALSF_RESOLVEDISM_BH_PROMOTION
            /* Promote to Type 5 sink (stellar-mass BH) */
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
#endif
            n_collapse_local++;
            continue;
        }

        /* Determine flag: 1 = explosive SN (ECSN/CCSN/PISN/PPISN), 2 = AGB/WD death */
        if(rem_type == REM_WD) {
            P[i].SNe_ThisTimeStep = 2; /* AGB: mass+metals, no energy */
            n_agb_local++;
        } else {
            P[i].SNe_ThisTimeStep = 1; /* SN: energy+mass+metals */
            n_sne_local++;
        }
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
            printf("ResolvedISM: %d SNe + %d AGB + %d collapses + %d wind-inj this timestep at t=%g\n",
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
        double *all_age = NULL, *all_lifetime = NULL;
        long long *all_id = NULL;
        int *all_remtype = NULL;

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

        if(ThisTask == 0)
        {
            for(i = 0; i < n_logged_total; i++) {
                if(all_remtype[i] == REM_WD) {
                    fprintf(FdAGBinfo, "%.16g %g %g %g %g %g %lld %g %.6e %.6e\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_u[i], all_rho[i],
                        all_id[i], all_mstar[i], all_age[i], all_lifetime[i]);
                } else {
                    fprintf(FdSNinfo, "%.16g %g %g %g %g %g %lld %g %d %.6e %.6e\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_u[i], all_rho[i],
                        all_id[i], all_mstar[i], all_remtype[i], all_age[i], all_lifetime[i]);
                }
            }
            fflush(FdSNinfo); fflush(FdAGBinfo);
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
                n_ia_local++;
            }
        }
        MPI_Allreduce(&n_ia_local, &n_ia_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(n_ia_total > 0 && ThisTask == 0) {
            printf("ResolvedISM: %d Type Ia SNe this timestep at t=%g\n", n_ia_total, All.Time);
            fflush(stdout);
        }

        /* Log Type Ia events to SNinfo.txt (same format, rem_type=7) */
        if(n_ia_total > 0)
        {
            double *ia_x, *ia_y, *ia_z, *ia_u, *ia_rho, *ia_mstar;
            long long *ia_id;
            ia_x = (double *)mymalloc("ia_x", n_ia_local * sizeof(double));
            ia_y = (double *)mymalloc("ia_y", n_ia_local * sizeof(double));
            ia_z = (double *)mymalloc("ia_z", n_ia_local * sizeof(double));
            ia_u = (double *)mymalloc("ia_u", n_ia_local * sizeof(double));
            ia_rho = (double *)mymalloc("ia_rho", n_ia_local * sizeof(double));
            ia_mstar = (double *)mymalloc("ia_mstar", n_ia_local * sizeof(double));
            ia_id = (long long *)mymalloc("ia_id", n_ia_local * sizeof(long long));
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
                    n_ia_logged++;
                }
            }
            int *recvcounts_ia = NULL, *displs_ia = NULL;
            double *all_ia_x=NULL, *all_ia_y=NULL, *all_ia_z=NULL, *all_ia_u=NULL, *all_ia_rho=NULL, *all_ia_mstar=NULL;
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
            }
            MPI_Gatherv(ia_x, n_ia_logged, MPI_DOUBLE, all_ia_x, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_y, n_ia_logged, MPI_DOUBLE, all_ia_y, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_z, n_ia_logged, MPI_DOUBLE, all_ia_z, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_u, n_ia_logged, MPI_DOUBLE, all_ia_u, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_rho, n_ia_logged, MPI_DOUBLE, all_ia_rho, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_mstar, n_ia_logged, MPI_DOUBLE, all_ia_mstar, recvcounts_ia, displs_ia, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(ia_id, n_ia_logged, MPI_LONG_LONG, all_ia_id, recvcounts_ia, displs_ia, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                for(i = 0; i < n_ia_total; i++) {
                    fprintf(FdSNinfo, "%.16g %g %g %g %g %g %lld %g %d\n",
                        All.Time, all_ia_x[i], all_ia_y[i], all_ia_z[i], all_ia_u[i], all_ia_rho[i],
                        all_ia_id[i], all_ia_mstar[i], 7); /* 7 = Type Ia */
                }
                fflush(FdSNinfo);
                myfree(all_ia_id); myfree(all_ia_mstar); myfree(all_ia_rho);
                myfree(all_ia_u); myfree(all_ia_z); myfree(all_ia_y); myfree(all_ia_x);
                myfree(displs_ia); myfree(recvcounts_ia);
            }
            myfree(ia_id); myfree(ia_mstar); myfree(ia_rho);
            myfree(ia_u); myfree(ia_z); myfree(ia_y); myfree(ia_x);
        }
    }
#endif
}


/* =========================================================================== */
/*  Single-star SN energy + mass + metal injection via code_block_xchange.     */
/*  Injects thermal energy, ejecta mass, and metal yields, kernel-weighted.    */
/* =========================================================================== */

struct kernel_resolvedismFB {double dp[3], r, wk, dwk, hinv, hinv3, hinv4;};

#define CORE_FUNCTION_NAME resolvedismFB_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismFB
#define OUTPUTFUNCTION_NAME out2particle_resolvedismFB
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismFB_active_check(i,loop_iteration))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius, Esne, Mej, wt_sum;
    MyDouble WindMomentum; /* total wind momentum to inject [code units: mass*vel] */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    MyDouble ElemYields[NUM_RESOLVEDISM_ELEMENTS]; /* element masses in ejecta [code units] */
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    MyDouble DustYields[NUM_RESOLVEDISM_DUST]; /* dust mass in ejecta [code units] */
#endif
    MyDouble MetalMass; /* total metal mass in ejecta [code units] (Z > He) */
    int fb_channel; /* feedback channel: 0=SN, 1=AGB, 2=wind, 3=Ia */
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismFB(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    in->wt_sum = 0; in->Esne = 0; in->Mej = 0; in->MetalMass = 0; in->WindMomentum = 0;
    /* Map SNe_ThisTimeStep (1=SN,2=AGB,3=wind,4=Ia) to fb_channel (0=SN,1=AGB,2=wind,3=Ia) */
    in->fb_channel = DMAX(P[i].SNe_ThisTimeStep - 1, 0);
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    for(k=0; k<NUM_RESOLVEDISM_ELEMENTS; k++) in->ElemYields[k] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    for(k=0; k<NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = 0;
#endif
    if(P[i].Mass <= 0) return;
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    in->wt_sum = P[i].DensityAroundParticle;
    if(P[i].DensityAroundParticle <= 0) return;
#endif

    /* ---- Type Ia: WD remnants have MstarSampleIMF=0, handle before get_star_info ---- */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    if(P[i].SNe_ThisTimeStep == 4) {
        in->Esne = IA_ENERGY_ERG / UNIT_ENERGY_IN_CGS;
        in->Mej = IA_EJECTA_MASS / UNIT_MASS_IN_SOLAR;
        double metal_mass_solar = 0;
        for(int k = 0; k < STBL_NELEM; k++) {
            double y_k = stellar_type_ia_yield(k);
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
            in->ElemYields[k] = y_k / UNIT_MASS_IN_SOLAR;
#endif
            if(k >= ELEM_C) metal_mass_solar += y_k;
        }
        in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;
#ifdef GALSF_RESOLVEDISM_DUST
        {
            double metal_yields_solar[STBL_NELEM], dust_yields_solar[NUM_RESOLVEDISM_DUST];
            for(int kk = 0; kk < STBL_NELEM; kk++) metal_yields_solar[kk] = in->ElemYields[kk] * UNIT_MASS_IN_SOLAR;
            resolvedism_dust_condensation(4, metal_yields_solar, dust_yields_solar);
            for(int kk = 0; kk < NUM_RESOLVEDISM_DUST; kk++) in->DustYields[kk] = dust_yields_solar[kk] / UNIT_MASS_IN_SOLAR;
        }
#endif
        return;
    }
#endif

    /* Get stellar mass and metallicity */
    double Mstar, logM, logZ;
    if(!get_star_info(i, &Mstar, &logM, &logZ)) return;

#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    /* ---- Radiation pressure (loop_iteration == 1) ---- */
    if(loop_iteration == 1) {
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr <= 0) return;
        double lifetime_yr = get_star_lifetime(Mstar, logM, logZ);
        if(star_age_yr >= lifetime_yr) return; /* dead star, no luminosity */

        double log_age = log10(DMAX(star_age_yr, 100.0));
        double log_Lbol = stellar_log_L_bol(logM, logZ, log_age);
        double Lbol_cgs = pow(10.0, log_Lbol); /* erg/s — table already stores log10(L [erg/s]) */
        if(Lbol_cgs <= 0) return;

        double dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i); /* code time */
        if(dt <= 0) return;
        double dt_cgs = dt * UNIT_TIME_IN_CGS;

        /* Dust-to-gas ratio */
#ifdef GALSF_RESOLVEDISM_DUST
        /* Use kernel-weighted local DGR from density loop, normalized to solar (DGR_solar ~ 0.01) */
        double DGR = P[i].DGR_around / 0.01;
#else
        double DGR = All.DGRnormalized;
#ifdef DGR_SCALE_WITH_Z
        DGR = All.DGRnormalized * All.InitialMetallicity;
#endif
#endif

        /* Local column density estimate: Sigma ~ rho * h */
        double rho_phys = P[i].DensityAroundParticle * All.cf_a3inv; /* physical density [code] */
        double h_phys = P[i].KernelRadius * All.cf_atime; /* physical smoothing length [code] */
        double NH = rho_phys * UNIT_DENSITY_IN_NHCGS * h_phys * UNIT_LENGTH_IN_CGS; /* H column [cm^-2] */

        /* UV dust optical depth: sigma_dust = 2e-21 * DGR cm^2/H (same as CHEMCOOL calc_photo.F) */
        double sigma_dust = 2.0e-21 * DGR; /* cm^2 per H atom */
        double tau_UV = sigma_dust * NH;

        /* Absorbed fraction of photon momentum */
        double f_abs = 1.0 - exp(-tau_UV);
        if(f_abs < 1.0e-6) return; /* negligible dust: no radiation pressure */

        /* Direct radiation pressure: dp = f_abs * L*dt/c */
        double dp_cgs = f_abs * Lbol_cgs * dt_cgs / C_LIGHT_CGS; /* g*cm/s */

        /* IR trapping boost: absorbed UV is re-emitted as IR, can be re-absorbed */
        /* tau_IR from dust IR opacity: kappa_IR ~ 5 * DGR cm^2/g (Semenov et al. 2003) */
        double Sigma_cgs = rho_phys * UNIT_DENSITY_IN_CGS * h_phys * UNIT_LENGTH_IN_CGS; /* g/cm^2 */
        double kappa_IR = 5.0 * DGR; /* cm^2/g, scales linearly with dust content */
        double tau_IR = kappa_IR * Sigma_cgs;
        if(tau_IR > 0) dp_cgs += tau_IR * Lbol_cgs * dt_cgs / C_LIGHT_CGS;

        in->WindMomentum = dp_cgs / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s); /* code momentum */
        if(Mstar >= 8.0) /* only log massive stars to avoid flooding stdout */
            printf("RADPRESSURE: Task=%d ID=%llu M=%.2f logL=%.2f tau_UV=%.2f tau_IR=%.2f f_abs=%.3e dp=%.3e[g*cm/s] WindMom=%.3e\n",
                ThisTask, (unsigned long long)P[i].ID, Mstar, log_Lbol, tau_UV, tau_IR, f_abs, dp_cgs, in->WindMomentum);
        return;
    }
#endif

    /* ---- Pass 0: SN/AGB/wind/Ia injection ---- */
    if(P[i].SNe_ThisTimeStep < 1 || P[i].SNe_ThisTimeStep > 4) return;

    /* ---- Wind injection (SNe_ThisTimeStep == 3) ---- */
#ifdef GALSF_RESOLVEDISM_WINDS
    if(P[i].SNe_ThisTimeStep == 3) {
        /* Convert accumulated wind momentum: Msun*km/s -> code units */
        double dp_cgs = P[i].WindMomentumAccum * SOLAR_MASS_CGS * 1.0e5; /* g*cm/s */
        in->WindMomentum = dp_cgs / (UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s);
        in->Mej = P[i].WindMassAccum / UNIT_MASS_IN_SOLAR;

        /* Wind metals from surface abundances at current age */
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        double log_age = log10(DMAX(star_age_yr, 100.0));
        double metal_mass_solar = 0;
        for(int k = 0; k < STBL_NELEM; k++) {
            double X_surf = stellar_surface_abundance(logM, logZ, log_age, k);
            double M_elem = X_surf * P[i].WindMassAccum; /* Msun */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
            in->ElemYields[k] = M_elem / UNIT_MASS_IN_SOLAR;
#endif
            if(k >= ELEM_C) metal_mass_solar += M_elem;
        }
        in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;
        return;
    }
#endif

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    int rem_type = stellar_remnant_type(logM, logZ);
    double rem_mass = stellar_remnant_mass(logM, logZ);

    /* Energy depends on remnant type */
    switch(rem_type) {
        case REM_WD:
            /* AGB end-of-life: mass+metals only, no energy */
            in->Esne = 0;
            break;
        case REM_PISN:
            /* Pair-instability: ~10^52 erg (scales with progenitor mass) */
            in->Esne = 1.0e52 / UNIT_ENERGY_IN_CGS;
            rem_mass = 0; /* complete disruption */
            break;
        case REM_PPISN:
            /* Pulsational pair-instability: standard energy, BH remnant */
            in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS;
            break;
        case REM_ECSN:
        case REM_CCSN:
        default:
            /* Standard core-collapse: 10^51 erg */
            in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS;
            break;
    }

    double Mej_solar = Mstar - rem_mass;
    if(Mej_solar < 0) Mej_solar = 0;
    in->Mej = Mej_solar / UNIT_MASS_IN_SOLAR;

    /* Compute yields: total element mass in ejecta = net_yield + X_birth * Mej */
    double metal_mass_solar = 0;
    for(k = 0; k < STBL_NELEM; k++) {
        double net_y = stellar_net_yield(logM, logZ, k); /* net yield in Msun */
        double X_birth = 0;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        X_birth = P[i].ElementAbundance[k]; /* birth mass fraction */
#else
        /* Without individual elements, approximate birth composition */
        if(k == ELEM_H) X_birth = 0.74;
        else if(k == ELEM_He) X_birth = 0.24;
        else {
            /* Scale metals by total Z, assuming solar ratios */
            double Z_birth = DMAX(P[i].BirthMetallicity, 1e-10);
            /* Rough solar mass fractions for metals (Z_sun = 0.014) */
            double solar_frac[STBL_NELEM] = {0.7381, 0.2485, 2.36e-3, 6.91e-4, 5.73e-3,
                3.70e-7, 1.25e-3, 3.34e-5, 7.08e-4, 5.67e-5,
                6.65e-4, 3.10e-4, 6.44e-5, 3.63e-6, 1.29e-3};
            X_birth = solar_frac[k] * (Z_birth / 0.014);
        }
#endif
        double M_elem_ej = net_y + X_birth * Mej_solar; /* total element mass ejected [Msun] */
        if(M_elem_ej < 0) M_elem_ej = 0;

#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        in->ElemYields[k] = M_elem_ej / UNIT_MASS_IN_SOLAR;
#endif
        /* Sum metals (everything heavier than He) */
        if(k >= ELEM_C) metal_mass_solar += M_elem_ej;
    }
    in->MetalMass = metal_mass_solar / UNIT_MASS_IN_SOLAR;

#ifdef GALSF_RESOLVEDISM_DUST
    {
        /* Compute dust yields from metal yields: sne_flag = 1 for CCSN, 2 for AGB */
        int dust_flag = (P[i].SNe_ThisTimeStep == 2) ? 2 : 1;
        double metal_yields_solar[STBL_NELEM], dust_yields_solar[NUM_RESOLVEDISM_DUST];
        for(k = 0; k < STBL_NELEM; k++) metal_yields_solar[k] = in->ElemYields[k] * UNIT_MASS_IN_SOLAR;
        resolvedism_dust_condensation(dust_flag, metal_yields_solar, dust_yields_solar);
        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) in->DustYields[k] = dust_yields_solar[k] / UNIT_MASS_IN_SOLAR;
    }
#endif

#else
    /* Fallback: no tables, just thermal energy, no mass/metal injection */
    in->Esne = 1.0e51 / UNIT_ENERGY_IN_CGS;
#endif
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat MomentumInjected[3]; /* total momentum deposited to gas [code units] */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    /* Star recoil for momentum conservation (winds and radiation pressure) */
    int do_recoil = 0;
#ifdef GALSF_RESOLVEDISM_WINDS
    if(P[i].SNe_ThisTimeStep == 3) do_recoil = 1;
#endif
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    if(loop_iteration == 1) do_recoil = 1;
#endif
    if(do_recoil && P[i].Mass > 0) {
        int k;
        for(k = 0; k < 3; k++) {
            P[i].Vel[k] -= out->MomentumInjected[k] / P[i].Mass;
        }
    }
}

int resolvedismFB_active_check(int i, int loop_iteration);
int resolvedismFB_active_check(int i, int loop_iteration)
{
    if(P[i].Type != 4) {return 0;}
    if(P[i].KernelRadius <= 0) {return 0;}
    if(P[i].NumNgb <= 0) {return 0;}

    if(loop_iteration == 0) {
        /* Pass 0: SN + AGB + wind + Type Ia injection */
        if(P[i].SNe_ThisTimeStep >= 1 && P[i].SNe_ThisTimeStep <= 4) {return 1;}
    }
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    if(loop_iteration == 1) {
        /* Pass 1: radiation pressure for all living luminous stars */
        if(P[i].Mass <= 0) {return 0;}
        double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[i].sampled) Mstar = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar = P[i].Mstar;
#endif
        if(Mstar > 0) {return 1;} /* living star with mass has luminosity */
    }
#endif
    return 0;
}


/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below */
int resolvedismFB_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double u, r2, h2;
    struct kernel_resolvedismFB kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismFB(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.Esne <= 0 && local.Mej <= 0 && local.WindMomentum <= 0) return 0;
    if(local.KernelRadius <= 0) return 0;
    if(local.wt_sum <= 0) return 0;
    h2 = local.KernelRadius * local.KernelRadius;
    kernel_hinv(local.KernelRadius, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) {return -2;}
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) {continue;}
                double Mass_j;
                #pragma omp atomic read
                Mass_j = P[j].Mass;
                if(Mass_j <= 0) {continue;}

                for(k=0;k<3;k++) {kernel.dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1);
                r2=0; for(k=0;k<3;k++) {r2 += kernel.dp[k]*kernel.dp[k];}
                if(r2 <= 0) {continue;}
                if(r2 >= h2) {continue;}
                kernel.r = sqrt(r2);
                if(kernel.r <= 0) {continue;}
                u = kernel.r * kernel.hinv;
                if(u<1) {kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 0);} else {kernel.wk=kernel.dwk=0;}
                if((kernel.wk <= 0)||(isnan(kernel.wk))) {continue;}

                double wk = Mass_j * kernel.wk / local.wt_sum; /* normalized weight function */

                /* ---- Thermal energy injection ---- */
                if(local.Esne > 0) {
#ifdef COSMIC_RAY_FLUID
                    double cr_frac = All.CosmicRay_SNeFraction; /* fraction of SN energy into CRs (typically 0.1) */
                    double dE = wk * local.Esne * (1.0 - cr_frac) / Mass_j;
#else
                    double dE = wk * local.Esne / Mass_j;
#endif
                    #pragma omp atomic
                    CellP[j].InternalEnergy += dE;
                    #pragma omp atomic
                    CellP[j].InternalEnergyPred += dE;
#ifdef COSMIC_RAY_FLUID
                    double dEcr = wk * local.Esne * cr_frac;
                    double v_ej = (local.Mej > 0) ? sqrt(2.0 * local.Esne / local.Mej) : 3000.0/UNIT_VEL_IN_KMS;
                    double crdir[3]; for(k=0;k<3;k++) {crdir[k] = -kernel.dp[k] / kernel.r;}
                    inject_cosmic_rays(dEcr, v_ej, 0, j, crdir);
#endif
                }

                /* ---- Mass injection ---- */
                if(local.Mej > 0) {
                    double dM = wk * local.Mej;

#ifdef METALS
                    /* Update total metallicity: Z_new = (M_old*Z_old + dMZ) / (M_old + dM) */
                    {
                        double Z_old, M_old = Mass_j;
                        #pragma omp atomic read
                        Z_old = P[j].Metallicity[0];
                        double dMZ = wk * local.MetalMass;
                        double dZ = (dMZ - Z_old * dM) / (M_old + dM);
                        #pragma omp atomic
                        P[j].Metallicity[0] += dZ;
                    }
                    /* Per-channel metal origin tracking: same dilution formula, only the active channel gets dMZ */
                    {
                        int ch = local.fb_channel;
                        double Mnew_j = Mass_j + dM;
                        for(int c = 0; c < 4; c++) {
                            double F_old;
                            #pragma omp atomic read
                            F_old = CellP[j].MetalMassFrom[c];
                            double dMZ_c = (c == ch) ? wk * local.MetalMass : 0;
                            double dF = (dMZ_c - F_old * dM) / Mnew_j;
                            #pragma omp atomic
                            CellP[j].MetalMassFrom[c] += dF;
                        }
                    }
#endif

#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                    /* Update individual element abundances */
                    for(k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++) {
                        double X_old;
                        #pragma omp atomic read
                        X_old = P[j].ElementAbundance[k];
                        double dMX = wk * local.ElemYields[k];
                        double dX = (dMX - X_old * dM) / (Mass_j + dM);
                        #pragma omp atomic
                        P[j].ElementAbundance[k] += dX;
                    }
#endif

                    /* Finally update gas mass */
                    #pragma omp atomic
                    P[j].Mass += dM;
                }

#ifdef GALSF_RESOLVEDISM_DUST
                /* ---- SN shock destruction of pre-existing dust ---- */
                if(local.Esne > 0) {
                    double E_into_cell = wk * local.Esne;
                    double Rho_j;
                    #pragma omp atomic read
                    Rho_j = CellP[j].Density;
                    double frac_dest = resolvedism_dust_sn_destruction_frac(Rho_j, E_into_cell, Mass_j);
                    if(frac_dest > 0) {
                        for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {
                            #pragma omp atomic
                            CellP[j].Dust[k] *= (1.0 - frac_dest);
                        }
                    }
                }
                /* ---- Inject new dust from ejecta ---- */
                if(local.Mej > 0) {
                    double dM = wk * local.Mej;
                    double Mnew_j = Mass_j + dM; /* mass already updated above */
                    for(k = 0; k < NUM_RESOLVEDISM_DUST; k++) {
                        double D_old;
                        #pragma omp atomic read
                        D_old = CellP[j].Dust[k];
                        double dMD = wk * local.DustYields[k];
                        double dD = (dMD - D_old * dM) / Mnew_j;
                        #pragma omp atomic
                        CellP[j].Dust[k] += dD;
                    }
                }
#endif

#ifdef GALSF_RESOLVEDISM_WINDS
                /* ---- Wind momentum kick ---- */
                if(local.WindMomentum > 0) {
                    /* Radial direction: star -> neighbor (opposite of kernel.dp which is star - neighbor) */
                    double dp_share = wk * local.WindMomentum; /* momentum share for this neighbor [code units] */
                    for(k = 0; k < 3; k++) {
                        double dp_k = dp_share * (-kernel.dp[k] / kernel.r); /* momentum component */
                        double dv_k = dp_k / Mass_j;
                        #pragma omp atomic
                        P[j].Vel[k] += dv_k;
                        #pragma omp atomic
                        CellP[j].VelPred[k] += dv_k;
                        #pragma omp atomic
                        P[j].dp[k] += dp_k / All.cf_atime; /* kick integrator: physical momentum units */
                        out.MomentumInjected[k] += dp_k;
                    }
                }
#endif

                /* NaN/Inf sanity check on neighbor after injection */
                {
                    double u_j;
                    #pragma omp atomic read
                    u_j = CellP[j].InternalEnergy;
                    if(!isfinite(u_j) || u_j < 0) {
                        printf("NAN_CHECK_FB: Task=%d neighbor ID=%llu u=%.6e after injection from star target=%d\n",
                            ThisTask, (unsigned long long)P[j].ID, u_j, target);
                    }
                    for(k=0;k<3;k++) {
                        double vk;
                        #pragma omp atomic read
                        vk = P[j].Vel[k];
                        if(!isfinite(vk)) {
                            printf("NAN_CHECK_FB: Task=%d neighbor ID=%llu Vel[%d]=%.6e after injection from star target=%d\n",
                                ThisTask, (unsigned long long)P[j].ID, k, vk, target);
                        }
                    }
                }

            } /* for(n = 0; n < numngb; n++) */
        } /* while(startnode >= 0) inner */
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}
            }
        }
    } /* while(startnode >= 0) outer */

    if(mode == 0) {out2particle_resolvedismFB(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


/* =========================================================================== */
/*  Generic feedback pass: called with fb_loop_iteration to select mode.      */
/*  0 = SN + AGB + wind injection                                            */
/*  1 = radiation pressure (all living luminous stars)                        */
/* =========================================================================== */
static void resolvedism_fb_calc(int fb_loop_iteration)
{
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    loop_iteration = fb_loop_iteration;
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"
}

void resolvedism_inject_sn_energy(void)
{
    /* ---- Pass 0: SN + AGB + wind injection ---- */
    PRINT_STATUS(" ..injecting single-star SN/AGB energy + mass + metals");
    resolvedism_fb_calc(0);

    /* Budget tracking: local accumulators for [0]=SN, [1]=AGB, [2]=wind, [3]=radpressure, [4]=Ia */
    double n_events[5] = {0,0,0,0,0};
    double M_injected[5] = {0,0,0,0,0};   /* ejecta mass [Msun] */
    double M_removed[5] = {0,0,0,0,0};    /* mass removed from star [Msun] */
    double E_injected[5] = {0,0,0,0,0};   /* energy [erg] */
    double dp_injected[5] = {0,0,0,0,0};  /* |momentum| [g*cm/s] */
    double Z_injected[5] = {0,0,0,0,0};   /* metals injected [Msun] */

    /* Post-injection: handle latching, cleanup, and budget tracking for pass 0 */
    int i;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef GALSF_RESOLVEDISM_WINDS
        /* Wind injection complete: reset accumulators, reduce star mass */
        if(P[i].SNe_ThisTimeStep == 3) {
            double dM_wind = P[i].WindMassAccum; /* Msun */
            double dp_wind = P[i].WindMomentumAccum * SOLAR_MASS_CGS * 1.0e5; /* g*cm/s */
            P[i].Mass -= dM_wind / UNIT_MASS_IN_SOLAR;
            if(P[i].Mass < 0) P[i].Mass = 0;

            n_events[2] += 1;
            M_injected[2] += dM_wind;
            M_removed[2] += dM_wind;
            dp_injected[2] += dp_wind;
            /* Wind metals: estimate from surface abundances */
            double Mstar_w, logM_w, logZ_w;
            if(get_star_info(i, &Mstar_w, &logM_w, &logZ_w)) {
                double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
                double log_age_w = log10(DMAX(star_age_yr, 100.0));
                double Z_wind = 0;
                for(int kk = ELEM_C; kk < STBL_NELEM; kk++) {
                    Z_wind += stellar_surface_abundance(logM_w, logZ_w, log_age_w, kk) * dM_wind;
                }
                Z_injected[2] += Z_wind;
            }

            printf("WIND: Task=%d ID=%llu M_init=%.2f dM=%.4f dp=%.3e\n",
                ThisTask, (unsigned long long)P[i].ID, P[i].MstarSampleIMF[0], dM_wind, dp_wind);
            P[i].WindMassAccum = 0;
            P[i].WindMomentumAccum = 0;
            P[i].SNe_ThisTimeStep = -1;
            continue;
        }
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
        /* Type Ia: WD fully disrupted, inject Chandrasekhar mass + energy */
        if(P[i].SNe_ThisTimeStep == 4) {
            double M_WD = P[i].Mass * UNIT_MASS_IN_SOLAR; /* WD mass before disruption [Msun] */
            n_events[4] += 1;
            M_injected[4] += IA_EJECTA_MASS;
            M_removed[4] += M_WD; /* particle mass removed */
            E_injected[4] += IA_ENERGY_ERG;
            double Z_ia = 0;
            for(int kk = ELEM_C; kk < STBL_NELEM; kk++) Z_ia += stellar_type_ia_yield(kk);
            Z_injected[4] += Z_ia;
            printf("TYPE_IA: Task=%d ID=%llu M_WD=%.3f E=%.2e[erg]\n",
                ThisTask, (unsigned long long)P[i].ID, M_WD, IA_ENERGY_ERG);
            P[i].Mass = 0; /* WD fully disrupted */
            P[i].M_drawn_Ia = 0; /* no longer eligible */
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
        double M_star_old = P[i].Mass * UNIT_MASS_IN_SOLAR; /* pre-remnant mass [Msun] */

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
        if(Mstar > 0) {
            double logM = log10(Mstar);
            double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
            int rem_type = stellar_remnant_type(logM, logZ);
            double rem_mass = stellar_remnant_mass(logM, logZ);
            if(rem_type == REM_PISN) rem_mass = 0; /* complete disruption */

            double Mej_solar = Mstar - rem_mass;
            if(Mej_solar < 0) Mej_solar = 0;

            n_events[channel] += 1;
            M_injected[channel] += Mej_solar;
            M_removed[channel] += M_star_old - rem_mass;
            if(channel == 0) { /* SN: has energy */
                double Esne_erg = 1.0e51;
                if(rem_type == REM_PISN) Esne_erg = 1.0e52;
                E_injected[channel] += Esne_erg;
            }

            /* Metals: net yields + birth composition of ejecta */
            double Z_ej = 0;
            for(int kk = ELEM_C; kk < STBL_NELEM; kk++) {
                double net_y = stellar_net_yield(logM, logZ, kk);
                double X_birth = 0;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                X_birth = P[i].ElementAbundance[kk];
#endif
                double M_elem = net_y + X_birth * Mej_solar;
                if(M_elem < 0) M_elem = 0;
                Z_ej += M_elem;
            }
            Z_injected[channel] += Z_ej;

            /* Set particle mass to remnant mass */
            P[i].Mass = rem_mass / UNIT_MASS_IN_SOLAR;
            if(channel == 0) {
                printf("SN: Task=%d ID=%llu M_init=%.2f M_ej=%.2f M_rem=%.2f rem_type=%d E=%.2e[erg]\n",
                    ThisTask, (unsigned long long)P[i].ID, Mstar, Mej_solar, rem_mass, rem_type, (rem_type==REM_PISN)?1.0e52:1.0e51);
            } else {
                printf("AGB: Task=%d ID=%llu M_init=%.2f M_ej=%.2f M_rem=%.2f\n",
                    ThisTask, (unsigned long long)P[i].ID, Mstar, Mej_solar, rem_mass);
            }

#ifdef GALSF_RESOLVEDISM_TYPE_IA
            /* Mark WD remnants as eligible for future Type Ia */
            if(rem_type == REM_WD) {
                P[i].M_drawn_Ia = Mstar; /* store original mass for DTD probability */
            }
#endif
#ifdef GALSF_RESOLVEDISM_BH_PROMOTION
            /* PPISN: explosive SN that still leaves a BH remnant — promote after ejecta injection */
            if(rem_type == REM_PPISN) {
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

        /* Zero out stellar mass to prevent re-trigger */
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        P[i].MstarSampleIMF[0] = 0;
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        P[i].Mstar = 0;
#endif
        P[i].SNe_ThisTimeStep = -1;
    }

    /* ---- Pass 1: radiation pressure ---- */
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    PRINT_STATUS(" ..injecting single-star radiation pressure");
    resolvedism_fb_calc(1);

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
        if(Mstar_rp > 0) n_events[3] += 1;
    }
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
                fprintf(FdFeedbackBudget, "%.16g %d %.0f %g %g %g %g %g\n",
                    All.Time, ch, glob_n[ch], glob_Mi[ch], glob_Mr[ch],
                    glob_E[ch], glob_dp[ch], glob_Z[ch]);
                CumulFeedbackEnergy += glob_E[ch];
                CumulFeedbackMass += glob_Mi[ch];
            }
        }
        fflush(FdFeedbackBudget);
    }
}
#include "../system/code_block_xchange_finalize.h"


#endif /* GALSF_RESOLVEDISM_FB */
