#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gpu_gravtree.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#include "./analytic_gravity.h"

/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This file contains the code for the gravitational force computation by
 *  means of the tree algorithm. To this end, a tree force is computed for all
 *  active local elements, and elements are exported to other processors if
 *  needed, where they can receive additional force contributions. If the
 *  TreePM algorithm is enabled, the force computed will only be the
 *  short-range part.
 */

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * substantially (condensed, new feedback routines added, many different
 * types of walk and calculations added, structures in memory changed,
 * switched options for nodes, optimizations, new physics modules and
 * calcutions, and new variable/memory conventions added)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 * Mike Grudic has also made major revisions to code the Hermitian calculations and binary timestepping.
 */

double Ewaldcount, Costtotal;
long long N_nodesinlist;
int Ewald_iter;			/* global in file scope, for simplicity */
void sum_top_level_node_costfactors(void);


/*! This function computes the gravitational forces for all active elements. If needed, a new tree is constructed, otherwise the dynamically updated
 *  tree is used.  Elements are only exported to other processors when needed. */
void gravity_tree(void)
{
    /* initialize variables */
    long long n_exported = 0; int i, j, maxnumnodes, iter; i = 0; j = 0; iter = 0; maxnumnodes=0;
    double t0, t1, timeall = 0, timetree1 = 0, timetree2 = 0, timetree, timewait, timecomm;
    double timecommsumm1 = 0, timecommsumm2 = 0, timewait1 = 0, timewait2 = 0, sum_costtotal, ewaldtot;
    double maxt, sumt, maxt1, sumt1, maxt2, sumt2, sumcommall, sumwaitall, plb, plb_max;
    CPU_Step[CPU_MISC] += measure_time();

    /* set new softening lengths */
    if(All.ComovingIntegrationOn) {set_softenings();}

    /* Refresh the per-particle ForceSoftening cache for active particles.
     * Single source of truth shared by CPU walk, GPU walk, and tree-build
     * split-scale computation in force_treebuild().  Inputs (KernelRadius,
     * AGS_KernelRadius, tidal_tensor_mag_prev, StarParticleEffectiveSize)
     * are guaranteed valid here: hydro/AGS density already ran earlier in
     * the timestep, set_softenings() above just updated All.ForceSoftening[].
     * Inactive particles retain their cached value from when they were last
     * active -- inputs only mutate during active processing, so the cached
     * value is still correct. */
    compute_all_force_softening(0);

    /* construct tree if needed */
#ifdef HERMITE_INTEGRATION
    if(!HermiteOnlyFlag)
#endif
    if(TreeReconstructFlag)
    {
        PRINT_STATUS("Tree construction initiated (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));
        CPU_Step[CPU_MISC] += measure_time();
        move_particles(All.Ti_Current);
        rearrange_particle_sequence();
        MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_DRIFT] += measure_time(); /* sync before we do the treebuild */
        force_treebuild(NumPart, NULL);
        MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_TREEBUILD] += measure_time(); /* and sync after treebuild as well */
        TreeReconstructFlag = 0;
        TreeMomentsStaleFlag = 0;
        PRINT_STATUS(" ..Tree construction done.");
    }

    /* refresh tree moments if stale (e.g. after star formation or sink mass change).
       This must run before ANY gravity evaluation including Hermite calls, since
       stale moments produce wrong forces. Much cheaper than a full treebuild. */
    {
        int TreeMomentsStaleFlag_global;
        MPI_Allreduce(&TreeMomentsStaleFlag, &TreeMomentsStaleFlag_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(TreeMomentsStaleFlag_global)
        {
            CPU_Step[CPU_MISC] += measure_time();
            force_refresh_node_moments();
            CPU_Step[CPU_TREEBUILD] += measure_time();
            TreeMomentsStaleFlag = 0;
        }
    }

    CPU_Step[CPU_TREEMISC] += measure_time(); t0 = my_second();
#ifndef SELFGRAVITY_OFF
    /* allocate buffers to arrange communication */
    PRINT_STATUS(" ..Begin tree force. (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));
    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (long) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                             sizeof(struct gravdata_in) + sizeof(struct gravdata_out) +
                                             sizemax(sizeof(struct gravdata_in),sizeof(struct gravdata_out))));
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {if(ThisTask == 0) printf(" ..All.BunchSize=%ld\n", All.BunchSize);}
    int k, ewald_max, diff, save_NextParticle, ndone, ndone_flag, place, recvTask; double tstart, tend, ax, ay, az; MPI_Status status;
    Ewaldcount = 0; Costtotal = 0; N_nodesinlist = 0; ewald_max=0;
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC) && !defined(PMGRID)
    ewald_max = 1; /* the tree-code will need to iterate to perform the periodic boundary condition corrections */
#endif

    if(GlobNumForceUpdate > All.TreeDomainUpdateFrequency * All.TotNumPart)
    { /* we have a fresh tree and would like to measure gravity cost */
        /* find the closest level */
        for(i = 1, TakeLevel = 0, diff = abs(All.LevelToTimeBin[0] - All.HighestActiveTimeBin); i < GRAVCOSTLEVELS; i++)
        {
            if(diff > abs(All.LevelToTimeBin[i] - All.HighestActiveTimeBin))
                {TakeLevel = i; diff = abs(All.LevelToTimeBin[i] - All.HighestActiveTimeBin);}
        }
        if(diff != 0) /* we have not found a matching slot */
        {
            if(All.HighestOccupiedTimeBin - All.HighestActiveTimeBin < GRAVCOSTLEVELS)	/* we should have space */
            {
                /* clear levels that are out of range */
                for(i = 0; i < GRAVCOSTLEVELS; i++)
                {
                    if(All.LevelToTimeBin[i] > All.HighestOccupiedTimeBin) {All.LevelToTimeBin[i] = 0;}
                    if(All.LevelToTimeBin[i] < All.HighestOccupiedTimeBin - (GRAVCOSTLEVELS - 1)) {All.LevelToTimeBin[i] = 0;}
                }
            }
            for(i = 0, TakeLevel = -1; i < GRAVCOSTLEVELS; i++)
            {
                if(All.LevelToTimeBin[i] == 0)
                {
                    All.LevelToTimeBin[i] = All.HighestActiveTimeBin;
                    TakeLevel = i;
                    break;
                }
            }
            if(TakeLevel < 0 && All.HighestOccupiedTimeBin - All.HighestActiveTimeBin < GRAVCOSTLEVELS)	/* we should have space */
                {terminate("TakeLevel < 0, even though we should have a slot");}
        }
    }
    else
    { /* in this case we do not measure gravity cost. Check whether this time-level
         has previously mean measured. If yes, then delete it so to make sure that it is not out of time */
        for(i = 0; i < GRAVCOSTLEVELS; i++) {if(All.LevelToTimeBin[i] == All.HighestActiveTimeBin) {All.LevelToTimeBin[i] = 0;}}
        TakeLevel = -1;
    }
    if(TakeLevel >= 0) {for(i = 0; i < NumPart; i++) {P[i].GravCost[TakeLevel] = 0;}} /* re-zero the cost [will be re-summed] */

    /* cache which particles need a new tree force BEFORE the tree walk runs: the tree walk can modify
       quantities like Min_Sink_FeedbackTime that needs_new_treeforce() depends on, so re-calling it
       in the post-processing loop could give a different answer, causing a particle that was computed
       by the tree walk (raw GravAccel, raw GravJerk) to incorrectly take the jerk-skip path (which
       expects G-multiplied GravAccel/GravJerk from a previous step). */
#ifdef ADAPTIVE_TREEFORCE_UPDATE
    std::vector<int> treeforce_skip_flag(ActiveParticleList.size(), 0);
    for(int ii = 0; ii < (int)ActiveParticleList.size(); ii++) {
        int i = ActiveParticleList[ii];
        if(!needs_new_treeforce(i)) {treeforce_skip_flag[ii] = 1;}
    }
#endif

    /* begin main communication and tree-walk loop. note the ewald-iter terms here allow for multiple iterations for periodic-tree corrections if needed */
    for(Ewald_iter = 0; Ewald_iter <= ewald_max; Ewald_iter++)
    {
        NextParticle = 0;	/* begin with this index */
        memset(ProcessedFlag, 0, All.MaxPart * sizeof(unsigned char));
        BufferCollisionFlag = 0; /* set to zero before operations begin */

        /* Step 13 Phase 4: speculative GPU pre-pass. Walks the local tree
         * on GPU for each active particle; on success, writes GravAccel
         * and marks ProcessedFlag so the CPU primary loop below skips it.
         * On pseudo-particle hit, leaves the particle untouched for the
         * CPU loop + MPI export machinery to handle unchanged. Ewald_iter
         * splits primary (==0) vs Ewald-correction (==1) walks; both are
         * active on all Kokkos builds. */
        if(Ewald_iter == 0) {gpu_gravtree_walk_primary();}
        else                {gpu_ewald_walk_primary();}

        do /* primary point-element loop */
        {
            iter++;
            BufferFullFlag = 0; Nexport = 0; save_NextParticle = NextParticle; tstart = my_second();

#ifdef _OPENMP
#pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else
                int mainthreadid = 0;
#endif
                gravity_primary_loop(&mainthreadid);	/* do local particles and prepare export list */
            }
            tend = my_second(); timetree1 += timediff(tstart, tend);

            /* ============================================================
             * Phase 9.4 RETIREMENT: CPU gravity export round-trip
             * ------------------------------------------------------------
             * On the GPU path the GPU pre-pass + Locally Essential
             * Tree (Phase 9.0-9.3) supply all foreign-rank gravity locally,
             * so the legacy MPI export round-trip is dead.  The block below
             * (BufferFullFlag compaction, MPI_Alltoall/Sendrecv exchange,
             * gravity_secondary_loop, scatter-back to P[]) is gated out in
             * retired in Step 5 C4. Final deletion is scheduled in
             * the Step 7 dead-code cleanup; the export infrastructure
             * (BunchSize, DataIndexTable, DataNodeList, gravdata_in/out)
             * is kept alive for surviving consumers (mg_gradient_correction,
             * potential.cc, subfind_potential).  See memory:
             * project_post_let_porting.md for the full audit.
             *
             * If Nexport > 0 here it means a particle's gravity is not
             * covered by LET -- raise LETAllocFactor in params.
             * ============================================================ */
            if(Nexport > 0) {
                printf("Phase 9.4 LET export retirement: rank %d has Nexport=%d particles needing foreign gravity not covered by LET -- raise LETAllocFactor\n", ThisTask, Nexport);
                fflush(stdout);
                endrun(914040);
            }

            /* Phase 9.4: export-back loop is retired under GPU offload, so the arena
             * is not invalidated by host-side P[] writes here.
             * Phase 8a Round 3b (2026-05-03): the prior "no-op safety net"
             * gpu_particles_arena_invalidate() that was here is REMOVED. The
             * comment authored above said it was already a no-op when arena state
             * is coherent. With the arena-coherence work in Round 1-2-3, the
             * arena IS coherent at this point (gpu_gravtree_walk_primary
             * invalidates internally at line 1979 if its host scatter happens;
             * mirror-update conversions in subsequent Round 3 commits will
             * replace that with mark_clean too). The redundant double-invalidate
             * here was costing nothing in old code but blocks fast-path
             * acquires after Round 2D's refresh. Removing it.
             *
             * If we ever surface a pre-acquire host mutation that makes arena
             * stale at this point, the GIZMO_GPU_ARENA_DEBUG=1 byte-compare
             * guard will abort with site name. Ship the trip wire instead of
             * cargo-cult invalidate. */
            if(NextParticle >= (int)ActiveParticleList.size()) {ndone_flag = 1;} else {ndone_flag = 0;} /* figure out if we are done with the particular active set here */
            tstart = my_second();
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); /* call an allreduce to figure out if all tasks are also done here, otherwise we need to iterate */
            tend = my_second(); timewait2 += timediff(tstart, tend);
        }
        while(ndone < NTask);
    } /* Ewald_iter */
    myfree(DataNodeList); myfree(DataIndexTable);

    /* assign node cost to particles */
    if(TakeLevel >= 0) {
        /* Modern GPU/LET gravity executes work on the target-owning rank, so
         * gpu_gravtree_walk_primary() records target-side interaction counts
         * directly in P[target].GravCost[TakeLevel]. */
    }


    /* now perform final operations on results [communication loop is done] */
#ifndef GRAVITY_HYBRID_OPENING_CRIT  // in collisional systems we don't want to rely on the relative opening criterion alone, because aold can be dominated by a binary companion but we still want accurate contributions from distant nodes. Thus we combine BH and relative criteria. - MYG
    if(header.flag_ic_info == FLAG_SECOND_ORDER_ICS) {if(!(All.Ti_Current == 0 && RestartFlag == 0)) {if(All.TypeOfOpeningCriterion == 1) {All.ErrTolTheta = 0;}}} else {if(All.TypeOfOpeningCriterion == 1) {All.ErrTolTheta = 0;}} /* This will switch to the relative opening criterion for the following force computations */
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int ii = 0; ii < (int)ActiveParticleList.size(); ii++)
    {
        int i = ActiveParticleList[ii];
#ifdef HERMITE_INTEGRATION
        if(HermiteOnlyFlag) {if(!eligible_for_hermite(i)) continue;} /* if we are completing an extra loop required for the Hermite integration, all of the below would be double-calculated, so skip it */
#endif      
#ifdef ADAPTIVE_TREEFORCE_UPDATE
        double dt = get_particle_timestep_in_physical(i);
        if(treeforce_skip_flag[ii]) { // use cached decision from BEFORE tree walk to avoid mismatch if tree walk modified quantities that needs_new_treeforce depends on
            P[i].GravAccel += P[i].GravJerk * (dt * All.cf_a2inv); // a^-1 from converting velocity term in the jerk to physical; a^-3 from the 1/r^3; a^2 from converting the physical dt * j increment to GravAccel back to the units for GravAccel; result is a^-2; note that Ewald and PMGRID terms are neglected from the jerk at present
            P[i].time_since_last_treeforce += dt;
            continue;
        } else {
            P[i].time_since_last_treeforce = dt;
        }
#endif
        /* before anything: multiply by G for correct units [be sure operations above/below are aware of this!] */
        P[i].GravAccel *= All.G;
#if (SINGLE_STAR_TIMESTEPPING > 0)
        P[i].COM_GravAccel *= All.G;
#endif

#ifdef EVALPOTENTIAL
        P[i].Potential *= All.G;
#ifdef BOX_PERIODIC
        if(All.ComovingIntegrationOn) {P[i].Potential -= All.G * 2.8372975 * pow(P[i].Mass, 2.0 / 3) * pow(All.OmegaMatter * 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G), 1.0 / 3);} else {if(All.OmegaLambda>0) {P[i].Potential -= 0.5*All.OmegaLambda*All.Hubble_H0_CodeUnits*All.Hubble_H0_CodeUnits * (P[i].Pos.norm_sq());}}
#endif
#ifdef PMGRID
        P[i].Potential += P[i].PM_Potential; /* add in long-range potential */
#endif
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
        P[i].TreeMass += P[i].Mass;
        if(P[i].Type == 5) printf("Particle %d sees mass %g in the gravity tree\n", P[i].ID, P[i].TreeMass);
#endif

#ifdef SPECIAL_POINT_WEIGHTED_MOTION
        if(P[i].Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES)
        {
            P[i].vel_of_nearest_special /= P[i].weight_sum_for_special_point_smoothing;
            P[i].acc_of_nearest_special /= P[i].weight_sum_for_special_point_smoothing;
            /* now reset the local values for this to actually match these, recalling the special particle in this module is just a tracer element */
            double dtime_phys = (All.Time - P[i].Time_Of_Last_SmoothedVelUpdate) / All.cf_hubble_a; /* want to convert to physical units */
            if(dtime_phys > 0) {
                P[i].Acc_Total_PrevStep = (P[i].vel_of_nearest_special - P[i].Vel) / (All.cf_atime * dtime_phys * All.cf_a2inv); /* converting to cosmological units here */
                P[i].Vel = P[i].vel_of_nearest_special;
            }
        }
#endif

        /* calculate 'old acceleration' for use in the relative tree-opening criterion */
        if(!(header.flag_ic_info == FLAG_SECOND_ORDER_ICS && All.Ti_Current == 0 && RestartFlag == 0)) /* to prevent that we overwrite OldAcc in the first evaluation for 2lpt ICs */
            {
                auto accel = P[i].GravAccel;
#ifdef PMGRID
                accel += P[i].GravPM;
#endif
                P[i].OldAcc = accel.norm() / All.G; /* convert back to the non-G units for convenience to match units in loops assumed */
            }

#if (SINGLE_STAR_TIMESTEPPING > 0) /* Subtract component of force from companion if in binary, because we will operator-split this */
        if((P[i].Type == 5) && (P[i].is_in_a_binary == 1)) {subtract_companion_gravity(i);}
#endif

#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE /* final operations to compute the tidal tensor and related quantities */
        P[i].tidal_tensorps *= All.G; /* give this the proper units */
#ifdef COMPUTE_JERK_IN_GRAVTREE
        P[i].GravJerk *= All.G; /* units */
#endif
#if defined(PMGRID) && !defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)
        P[i].tidal_tensorps += P[i].tidal_tensorpsPM; /* add the long-range (pm-grid) contribution; but make sure to do this after the unit multiplication by G above, since the PM term already has G built into it */
#endif
#endif /* COMPUTE_TIDAL_TENSOR_IN_GRAVTREE */

#if defined(RT_OTVET) /* normalize the Eddington tensors we just calculated by walking the tree (normalize to trace=1) */
        if(P[i].Type == 0) {
            int k_freq; for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
            {double trace = CellP[i].ET[k_freq].trace();
                if(!isnan(trace) && (trace>0)) {CellP[i].ET[k_freq] /= trace;} else {CellP[i].ET[k_freq].set_isotropic(1./3.);}}}
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY) /* normalize to energy density with C, and multiply by volume to use standard 'finite volume-like' quantity as elsewhere in-code */
        if(P[i].Type==0) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {CellP[i].Rad_E_gamma[kf] *= P[i].Mass/(CellP[i].Density*All.cf_a3inv * C_LIGHT_CODE_REDUCED);}}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        if(P[i].Type==0) {CellP[i].SubGrid_CosmicRayEnergyDensity *= cr_get_source_shieldfac(i, P, CellP);}
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX) /* multiply by volume to use standard 'finite volume-like' quantity as elsewhere in-code */
        if(P[i].Type==0) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {CellP[i].Rad_Flux[kf] *= P[i].Mass/(CellP[i].Density*All.cf_a3inv);}} // convert to standard finite-volume-like units //
#if !defined(RT_DISABLE_RAD_PRESSURE) // if we save the fluxes, we didnt apply forces on-the-spot, which means we appky them here //
        if((P[i].Type==0) && (P[i].Mass>0))
        {
            int kfreq; double vol_inv=CellP[i].Density*All.cf_a3inv/P[i].Mass, h_i=P[i].Get_Particle_Size()*All.cf_atime, sigma_eff_i=P[i].Mass/(h_i*h_i);
            Vec3<double> radacc={};
            for(kfreq=0; kfreq<N_RT_FREQ_BINS; kfreq++)
            {
                double f_slab=1, erad_i=0, kappa_rad=rt_kappa(i,kfreq, P, CellP), tau_eff=kappa_rad*sigma_eff_i; if(tau_eff > 1.e-4) {f_slab = (1.-exp(-tau_eff)) / tau_eff;} // account for optically thick local 'slabs' self-shielding themselves
                double acc_norm = kappa_rad * f_slab / C_LIGHT_CODE_REDUCED; // pre-factor for radiation pressure acceleration
#if defined(RT_LEBRON)
                acc_norm *= All.PhotonMomentum_Coupled_Fraction; // allow user to arbitrarily increase/decrease strength of RP forces for testing
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
                erad_i = CellP[i].Rad_E_gamma_Pred[kfreq]*vol_inv; // if can, include the O[v/c] terms
#endif
                Vec3<double> flux_i = CellP[i].Rad_Flux_Pred[kfreq] * vol_inv;
                Vec3<double> vel_i = CellP[i].VelPred * (1.0/All.cf_atime);
                double flux_mag2 = flux_i.norm_sq() + MIN_REAL_NUMBER, vdotflux = dot(vel_i, flux_i); // initialize a bunch of variables we will need
                Vec3<double> vdot_h = (vel_i + flux_i * (vdotflux/flux_mag2)) * erad_i; // calculate volume integral of scattering coefficient t_inv * (gas_vel . [e_rad*I + P_rad_tensor]), which gives an additional time-derivative term. this is the P term //
                radacc += (flux_i - vdot_h) * acc_norm; // note these 'vdoth' terms shouldn't be included in FLD, since its really assuming the entire right-hand-side of the flux equation reaches equilibrium with the pressure tensor, which gives the expression in rt_utilities
            }
#if defined(RT_RAD_PRESSURE_OUTPUT)
            CellP[i].Rad_Accel = radacc; // here units are the same as hydroaccel, so no extra comoving units
#else
            P[i].GravAccel += radacc * (1.0/All.cf_a2inv); // convert into our code units for GravAccel, which are comoving gm/r^2 units //
#endif
        }
#endif
#endif

#ifdef RT_USE_TREECOL_FOR_NH  /* compute the effective column density that gives equivalent attenuation of a uniform background: -log(avg(exp(-tau)))/kappa */
        double attenuation=0, minimum_column=MAX_REAL_NUMBER; int kbin;
        double kappa_photoelectric = 500. * DMAX(1e-4, (P[i].Metallicity[0]/All.SolarAbundances[0])*return_dust_to_metals_ratio_vs_solar(i,0, P, CellP)); // dust opacity in cgs
        for(kbin=0; kbin<RT_USE_TREECOL_FOR_NH; kbin++) {
	      attenuation += exp(DMAX(-P[i].ColumnDensityBins[kbin] * UNIT_SURFDEN_IN_CGS * kappa_photoelectric,-100));
	      minimum_column = DMIN(minimum_column,P[i].ColumnDensityBins[kbin]);
	    } // we put a floor here to avoid underflow errors where exp(-large) = 0 - will just return a very high surface density that will be in the highly optically thick regime where both the ISRF and cooling radiation escape will be negligible
        P[i].SigmaEff = -log(attenuation/RT_USE_TREECOL_FOR_NH) / (kappa_photoelectric * UNIT_SURFDEN_IN_CGS);
	    if(P[i].SigmaEff < minimum_column) {P[i].SigmaEff = minimum_column;} // if in the overflowing regime just take the minimum column density to extrapolate better to the IR-thick regime
#ifdef GIZMO_TREECOL_DIAG
        if(ThisTask==0 && P[i].Type==0 && i<10 && All.NumCurrentTiStep<4) {
            double csum=0; for(int kb=0;kb<RT_USE_TREECOL_FOR_NH;kb++) csum+=P[i].ColumnDensityBins[kb];
            printf("TREECOL_DIAG step=%d i=%d ID=%llu SigmaEff=%g binsum=%g bins=[%g,%g,%g,%g,%g,%g] LET=%d\n",
                   All.NumCurrentTiStep,(int)i,(unsigned long long)P[i].ID,P[i].SigmaEff,csum,
                   P[i].ColumnDensityBins[0],P[i].ColumnDensityBins[1],P[i].ColumnDensityBins[2],
                   P[i].ColumnDensityBins[3],P[i].ColumnDensityBins[4],P[i].ColumnDensityBins[5],
                   (MaxForeignNodes>0)?1:0); fflush(stdout);
        }
#endif
#endif

#if !defined(BOX_PERIODIC) && !defined(PMGRID) /* some factors here in case we are trying to do comoving simulations in a non-periodic box (special use cases) */
        if(All.ComovingIntegrationOn) {P[i].GravAccel += P[i].Pos * (0.5*All.OmegaMatter*All.Hubble_H0_CodeUnits*All.Hubble_H0_CodeUnits);}
        if(All.ComovingIntegrationOn==0) {P[i].GravAccel += P[i].Pos * (All.OmegaLambda*All.Hubble_H0_CodeUnits*All.Hubble_H0_CodeUnits);}
#ifdef EVALPOTENTIAL
        if(All.ComovingIntegrationOn) {P[i].Potential -= 0.5*All.OmegaMatter*All.Hubble_H0_CodeUnits*All.Hubble_H0_CodeUnits * P[i].Pos.norm_sq();}
#endif
#endif

    } /* end of loop over active particles*/

    /* Phase 8a Round 3d: mirror the post-processed host P[i] / CellP[i] for
     * active particles into the arena, then mark_clean instead of invalidating.
     * The post-loop above wrote GravAccel(×G), OldAcc, Potential, tidal_*,
     * Rad_*, SigmaEff, etc. for active i — many fields under various #ifdef
     * branches. Per-touched-i full struct copy avoids the maintenance burden
     * of enumerating each conditional field. Cost is O(N_active) which is
     * trivial compared to the 1.1s slow-path memcpy that the next acquire
     * would otherwise pay. */
    {
        struct particle_data *P_arena_post     = gpu_particles_arena_P();
        struct gas_cell_data *CellP_arena_post = gpu_particles_arena_CellP();
        if(P_arena_post) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for(int ii = 0; ii < (int)ActiveParticleList.size(); ii++) {
                int i = ActiveParticleList[ii];
                P_arena_post[i] = P[i];
                if(CellP_arena_post && P[i].Type == 0) {
                    CellP_arena_post[i] = CellP[i];
                }
            }
            gpu_particles_arena_mark_clean_after_scatter("gravity_tree_post_loop");
        } else {
            /* Arena unavailable (e.g. capacity issue or already invalidated by
             * something we missed) — defensive invalidate keeps correctness. */
            gpu_particles_arena_invalidate();
        }
    }

#endif /* end SELFGRAVITY operations (check if SELFGRAVITY_OFF not enabled) */


    add_analytic_gravitational_forces(); /* add analytic terms, which -CAN- be enabled even if self-gravity is not */


    /* Now the force computation is finished: gather timing and diagnostic information */
    t1 = WallclockTime = my_second(); timeall = timediff(t0, t1);
    timetree = timetree1 + timetree2; timewait = timewait1 + timewait2; timecomm = timecommsumm1 + timecommsumm2;
    MPI_Reduce(&timetree, &sumt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree1, &sumt1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree1, &maxt1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree2, &sumt2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timetree2, &maxt2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timewait, &sumwaitall, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timecomm, &sumcommall, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Costtotal, &sum_costtotal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Ewaldcount, &ewaldtot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    sumup_longs(1, &n_exported, &n_exported);
    sumup_longs(1, &N_nodesinlist, &N_nodesinlist);
    All.TotNumOfForces += GlobNumForceUpdate;
    plb = (NumPart / ((double) All.TotNumPart)) * NTask;
    MPI_Reduce(&plb, &plb_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Numnodestree, &maxnumnodes, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    CPU_Step[CPU_TREEMISC] += timeall - (timetree + timewait + timecomm);
    CPU_Step[CPU_TREEWALK1] += timetree1; CPU_Step[CPU_TREEWALK2] += timetree2;
    CPU_Step[CPU_TREESEND] += timecommsumm1; CPU_Step[CPU_TREERECV] += timecommsumm2;
    CPU_Step[CPU_TREEWAIT1] += timewait1; CPU_Step[CPU_TREEWAIT2] += timewait2;
#ifdef OUTPUT_ADDITIONAL_RUNINFO
    if(ThisTask == 0)
    {
        fprintf(FdTimings, "Step= %lld  t= %.16g  dt= %.16g \n",(long long) All.NumCurrentTiStep, All.Time, All.TimeStep);
        fprintf(FdTimings, "Nf= %d%09d  total-Nf= %d%09d  ex-frac= %g (%g) iter= %d\n", (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), (int) (All.TotNumOfForces / 1000000000), (int) (All.TotNumOfForces % 1000000000), n_exported / ((double) GlobNumForceUpdate), N_nodesinlist / ((double) n_exported + 1.0e-10), iter); /* note: on Linux, the 8-byte integer could be printed with the format identifier "%qd", but doesn't work on AIX */
        fprintf(FdTimings, "work-load balance: %g (%g %g) rel1to2=%g   max=%g avg=%g\n", maxt / (1.0e-6 + sumt / NTask), maxt1 / (1.0e-6 + sumt1 / NTask), maxt2 / (1.0e-6 + sumt2 / NTask), sumt1 / (1.0e-6 + sumt1 + sumt2), maxt, sumt / NTask);
        fprintf(FdTimings, "particle-load balance: %g\n", plb_max);
        fprintf(FdTimings, "max. nodes: %d, filled: %g\n", maxnumnodes, maxnumnodes / (All.TreeAllocFactor * All.MaxPart + NTopnodes));
        fprintf(FdTimings, "part/sec=%g | %g  ia/part=%g (%g)\n", GlobNumForceUpdate / (sumt + 1.0e-20), GlobNumForceUpdate / (1.0e-6 + maxt * NTask), ((double) (sum_costtotal)) / (1.0e-20 + GlobNumForceUpdate), ((double) ewaldtot) / (1.0e-20 + GlobNumForceUpdate)); fprintf(FdTimings, "\n");
        fflush(FdTimings);
    }
    double costtotal_new = 0, sum_costtotal_new;
    if(TakeLevel >= 0)
    {
        for(i = 0; i < NumPart; i++) {costtotal_new += P[i].GravCost[TakeLevel];}
        MPI_Reduce(&costtotal_new, &sum_costtotal_new, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(sum_costtotal>0) {PRINT_STATUS(" ..relative error in the total number of tree-gravity interactions = %g", (sum_costtotal - sum_costtotal_new) / sum_costtotal);} /* can be non-zero if THREAD_SAFE_COSTS is not used (and due to round-off errors). */
    }
#endif
    CPU_Step[CPU_TREEMISC] += measure_time();
}




void *gravity_primary_loop(void *p)
{
    int i, j, ret, thread_id = *(int *) p, *exportflag, *exportnodecount, *exportindex;
    exportflag = Exportflag + thread_id * NTask; exportnodecount = Exportnodecount + thread_id * NTask; exportindex = Exportindex + thread_id * NTask;
    for(j = 0; j < NTask; j++) {exportflag[j] = -1;} /* Note: exportflag is local to each thread */
#ifdef _OPENMP
    if(BufferCollisionFlag && thread_id) {return NULL;} /* force to serial for this subloop if threads simultaneously cross the Nexport bunchsize threshold */
#endif
#ifndef GRAVITY_PRIMARY_LOOP_BATCH_SIZE
#define GRAVITY_PRIMARY_LOOP_BATCH_SIZE 8
#endif
    while(1)
    {
        int batch[GRAVITY_PRIMARY_LOOP_BATCH_SIZE], batch_count = 0;
#ifdef _OPENMP
#pragma omp critical(_nextlistgravprim_)
#endif
        {
            while(batch_count < GRAVITY_PRIMARY_LOOP_BATCH_SIZE && BufferFullFlag == 0 && NextParticle < (int)ActiveParticleList.size())
            {
                int idx = ActiveParticleList[NextParticle]; NextParticle++;
                if(!ProcessedFlag[idx]) {batch[batch_count++] = idx;}
            }
        }
        if(batch_count == 0) {break;}
        int buffer_full = 0;
        for(int b = 0; b < batch_count; b++)
        {
            i = batch[b];
#ifdef HERMITE_INTEGRATION /* if we are in the Hermite extra loops and a particle is not flagged for this, simply mark it done and move on */
            if(HermiteOnlyFlag && !eligible_for_hermite(i)) {ProcessedFlag[i]=1; continue;}
#endif
#ifdef ADAPTIVE_TREEFORCE_UPDATE
            if(!needs_new_treeforce(i)) {ProcessedFlag[i]=1; continue;}
#endif

#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC) && !defined(PMGRID)
            if(Ewald_iter)
            {
                ret = force_treeevaluate_ewald_correction(i, 0, exportflag, exportnodecount, exportindex);
                if(ret >= 0) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                    Ewaldcount += ret;
                } else {buffer_full = 1; break;}
            }
            else
#endif
            {
                ret = force_treeevaluate(i, 0, exportflag, exportnodecount, exportindex);
                if(ret < 0) {buffer_full = 1; break;}
#ifdef _OPENMP
#pragma omp atomic
#endif
                Costtotal += ret;
            }
            ProcessedFlag[i] = 1;
        }
        if(buffer_full) {break;}
    } // while loop
    return NULL;
}


void *gravity_secondary_loop(void *p)
{
    int j, nodesinlist, dummy, ret;
#ifndef GRAVITY_SECONDARY_LOOP_BATCH_SIZE
#define GRAVITY_SECONDARY_LOOP_BATCH_SIZE 8
#endif
    while(1)
    {
        int batch[GRAVITY_SECONDARY_LOOP_BATCH_SIZE], batch_count = 0;
#ifdef _OPENMP
#pragma omp critical(_nextlistgravsec_)
#endif
        {
            while(batch_count < GRAVITY_SECONDARY_LOOP_BATCH_SIZE && NextJ < Nimport)
            {
                batch[batch_count++] = NextJ; NextJ++;
            }
        }
        if(batch_count == 0) {break;}
        for(int b = 0; b < batch_count; b++)
        {
            j = batch[b];
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC) && !defined(PMGRID)
            if(Ewald_iter)
            {
                int cost = force_treeevaluate_ewald_correction(j, 1, &dummy, &dummy, &dummy);
#ifdef _OPENMP
#pragma omp atomic
#endif
                Ewaldcount += cost;
            }
            else
#endif
            {
                ret = force_treeevaluate(j, 1, &nodesinlist, &dummy, &dummy);
#ifdef _OPENMP
#pragma omp atomic
#endif
                N_nodesinlist += nodesinlist;
#ifdef _OPENMP
#pragma omp atomic
#endif
                Costtotal += ret;
            }
        }
    }
    return NULL;
}


void sum_top_level_node_costfactors(void)
{
    double *costlist = (double*)mymalloc("costlist", NTopnodes * sizeof(double));
    double *costlist_all = (double*)mymalloc("costlist_all", NTopnodes * sizeof(double));
    int i; for(i = 0; i < NTopnodes; i++) {costlist[i] = Nodes[All.MaxPart + i].GravCost;}
    MPI_Allreduce(costlist, costlist_all, NTopnodes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for(i = 0; i < NTopnodes; i++) {Nodes[All.MaxPart + i].GravCost = costlist_all[i];}
    myfree(costlist_all); myfree(costlist);
}


/*! This function sets the (comoving) softening length of all particle types in the table All.ForceSoftening[...].
 We check that the physical softening length is bounded by the Softening-MaxPhys values */
void set_softenings(void)
{
    int i; double soft[6];
    soft[0] = All.SofteningGas;
    soft[1] = All.SofteningHalo;
    soft[2] = All.SofteningDisk;
    soft[3] = All.SofteningBulge;
    soft[4] = All.SofteningStars;
    soft[5] = All.SofteningBndry;
    if(All.ComovingIntegrationOn)
    {
        double soft_temp[6], cf_atime = 1./All.Time;
        soft_temp[0] = All.SofteningGasMaxPhys * cf_atime;
        soft_temp[1] = All.SofteningHaloMaxPhys * cf_atime;
        soft_temp[2] = All.SofteningDiskMaxPhys * cf_atime;
        soft_temp[3] = All.SofteningBulgeMaxPhys * cf_atime;
        soft_temp[4] = All.SofteningStarsMaxPhys * cf_atime;
        soft_temp[5] = All.SofteningBndryMaxPhys * cf_atime;
        for(i=0; i<6; i++) {if(soft_temp[i]<soft[i]) {soft[i]=soft_temp[i];}}
    }
    for(i=0; i<6; i++) {All.ForceSoftening[i] = soft[i] / KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER;}
    All.MinKernelRadius = All.MinGasKernelRadiusFractional * All.ForceSoftening[0]; /* set the minimum gas kernel length to be used this timestep */
#ifndef SELFGRAVITY_OFF
    if(All.MinKernelRadius <= 5.0*EPSILON_FOR_TREERND_SUBNODE_SPLITTING * All.ForceSoftening[0]) {All.MinKernelRadius = 5.0*EPSILON_FOR_TREERND_SUBNODE_SPLITTING * All.ForceSoftening[0];}
#endif
}


/*! This function is used as a comparison kernel in a sort routine. It is used to group particles in the communication buffer that are going to be sent to the same CPU */
int data_index_compare(const void *a, const void *b)
{
    if(((struct data_index *) a)->Task < (((struct data_index *) b)->Task)) {return -1;}
    if(((struct data_index *) a)->Task > (((struct data_index *) b)->Task)) {return +1;}
    if(((struct data_index *) a)->Index < (((struct data_index *) b)->Index)) {return -1;}
    if(((struct data_index *) a)->Index > (((struct data_index *) b)->Index)) {return +1;}
    if(((struct data_index *) a)->IndexGet < (((struct data_index *) b)->IndexGet)) {return -1;}
    if(((struct data_index *) a)->IndexGet > (((struct data_index *) b)->IndexGet)) {return +1;}
    return 0;
}


static void msort_dataindex_with_tmp(struct data_index *b, size_t n, struct data_index *t)
{
    if(n <= 1) {return;}
    struct data_index *tmp;
    struct data_index *b1, *b2;
    size_t n1, n2;
    n1 = n / 2;
    n2 = n - n1;
    b1 = b;
    b2 = b + n1;
    msort_dataindex_with_tmp(b1, n1, t);
    msort_dataindex_with_tmp(b2, n2, t);
    tmp = t;
    while(n1 > 0 && n2 > 0)
    {
        if(b1->Task < b2->Task || (b1->Task == b2->Task && b1->Index <= b2->Index))
        {
            --n1;
            *tmp++ = *b1++;
        }
        else
        {
            --n2;
            *tmp++ = *b2++;
        }
    }
    if(n1 > 0) {memcpy(tmp, b1, n1 * sizeof(struct data_index));}
    memcpy(b, t, (n - n2) * sizeof(struct data_index));
}


void mysort_dataindex(void *b, size_t n, size_t s, int (*cmp) (const void *, const void *))
{
    const size_t size = n * s;
    struct data_index *tmp = (struct data_index *) mymalloc("struct data_index *tmp", size);
    msort_dataindex_with_tmp((struct data_index *) b, n, tmp);
    myfree(tmp);
}


#if (SINGLE_STAR_TIMESTEPPING > 0)
void subtract_companion_gravity(int i)
{
    /* Remove contribution to gravitational field and tidal tensor from the stars in the binary to the center of mass */
    double u, dr, fac, fac2, h, h_inv, h3_inv, u2; SymmetricTensor2<MyFloat> tidal_tensorps; int i1, i2;
    dr = P[i].comp_dx.norm();
    h = SinkParticle_GravityKernelRadius;  h_inv = 1.0 / h; h3_inv = h_inv*h_inv*h_inv; u = dr*h_inv; u2=u*u;
    fac = P[i].comp_Mass / (dr*dr*dr); fac2 = 3.0 * P[i].comp_Mass / (dr*dr*dr*dr*dr); /* no softening nonsense */
    if(dr < h) /* second derivatives needed -> calculate them from softened potential */
    {
	    fac = P[i].comp_Mass * kernel_gravity(u, h_inv, h3_inv, 1);
        fac2 = P[i].comp_Mass * kernel_gravity(u, h_inv, h3_inv, 2);
    }
    P[i].COM_GravAccel = P[i].GravAccel - P[i].comp_dx * (fac * All.G); /* this assumes the 'G' has been put into the units for the grav accel */

    /* Adjusting tidal tensor according to terms above */
    tidal_tensorps = P[i].tidal_tensorps - fac2 * outer_product(P[i].comp_dx);
    tidal_tensorps[0][0] += fac; tidal_tensorps[1][1] += fac; tidal_tensorps[2][2] += fac;

#ifdef SINK_OUTPUT_MOREINFO
    printf("Corrected center of mass acceleration %g %g %g tidal tensor diagonal elements %g %g %g \n", P[i].COM_GravAccel[0], P[i].COM_GravAccel[1], P[i].COM_GravAccel[2], tidal_tensorps[0][0],tidal_tensorps[1][1],tidal_tensorps[2][2]);
#endif
    P[i].COM_dt_tidal = sqrt(1.0 / (All.G * tidal_tensorps.frobenius_norm()));
}
#endif

#ifdef ADAPTIVE_TREEFORCE_UPDATE
int needs_new_treeforce(int n){
    if(P[n].Type > 0){ // in this implementation we only do the lazy updating for gas cells whose timesteps are otherwise constrained by multiphysics (e.g. radiation, feedback)
        return 1;
    } else {
        if(P[n].time_since_last_treeforce >= P[n].tdyn_step_for_treeforce * ADAPTIVE_TREEFORCE_UPDATE) {return 1;}
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
        else if(P[n].time_since_last_treeforce >= P[n].Min_Sink_FeedbackTime) {return 1;} // we want ejecta to re-calculate their feedback time so they don't get stuck on a short timestep
#endif        
        else {return 0;}
    }
}
#endif
