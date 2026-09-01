#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"
#include "../core/proto.h"
#include "gpu_gravtree.h"
#include "gpu_gravity_tree.h"   /* gpu_gravity_tree_mark_born_current */
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#include "./analytic_gravity.h"

/*! Host-vs-device routing for the gravity walk and the dynamic tree update, keyed on the
 *  RANK-LOCAL count of active gravity candidates. The device path must drift every node
 *  in the tree before its parallel walk can be race-free, so its floor is set by the tree
 *  size rather than by the active set; the host walk drifts each node only when it opens
 *  it. Below the threshold the sweep costs more than the walk it enables.
 *
 *  The threshold is conservative against a crossover measured near 6e4 rank-local
 *  candidates on 16-rank FIRE, where routing the whole tree walk to the host cut the
 *  cost of steps with fewer than 1e4 global active elements by a third. Above it the
 *  device path wins and the host walk's serial node drift becomes the bottleneck. */
int gravity_walk_route_to_host(long long n_local_active)
{
    /* Once any node has been drifted lazily at this time, the host owns the rest of the
     * time step: the device sweep skips nodes already at its target time, so it can no
     * longer bring their mirror up to date, and a second gravity evaluation at the same
     * time (a Hermite correction pass, a repeated walk for the opening criterion) would
     * otherwise read that stale geometry.  A tree built after that drift is
     * exempt: the build rewrote every node and every mirror, so the record of an
     * earlier lazy drift no longer describes anything. */
    if(!gpu_gravity_tree_nodes_current_at(All.Ti_Current)
            && force_host_lazy_drift_ti() == All.Ti_Current) {return 1;}

    return (All.GravityHostWalkBelowActive > 0 && n_local_active < (long long)All.GravityHostWalkBelowActive) ? 1 : 0;
}

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


/*! This function computes the gravitational forces for all active elements. If needed, a new tree is constructed, otherwise the dynamically updated
 *  tree is used.  Elements are only exported to other processors when needed. */
/*! Promote the staged tree-opening acceleration scale into the value the criterion reads.
 *
 *  Runs ONCE per tree build, immediately before the tree (and with it the imported ghost tree) is
 *  constructed, and nowhere else.  The import is pruned against this value: a sender ships a node
 *  as a childless multipole precisely when no target on the receiving rank would open it, and
 *  OldAcc enters that test.  Were it to change while the tree still stands, a later walk could
 *  apply a stricter criterion than the sender did, ask to descend a node whose children were never
 *  sent, and either stop on the completeness guard or -- under a topleaf that has a sibling --
 *  silently skip that node's mass.  Promoting here ties the value to the lifetime of the structure
 *  pruned against it, so every walk on a given tree opens exactly what its import covers.
 *
 *  All local particles, not just the active ones: a particle may become active while this tree
 *  still stands, and its opening decisions must be covered by the same import.  For a particle
 *  whose walk measured nothing new this rewrites the identical value.
 */
void refresh_old_acceleration_for_tree_opening(void)
{
    /* OldAcc is overloaded on a 2lpt start: the IC reader leaves the particle masses in it, and
     * init.cc deliberately does not zero it for that reason.  Nothing has been staged yet either,
     * so promoting here would replace those masses with zeros before the first evaluation is done
     * with them.  The same condition suppresses the staging site after the walk; the walk that
     * follows this build does not read OldAcc anyway, because the criterion stays Barnes-Hut for
     * exactly this case. */
    if(header.flag_ic_info == FLAG_SECOND_ORDER_ICS && All.Ti_Current == 0 && RestartFlag == 0) {return;}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < NumPart; i++) {P[i].OldAcc = P[i].OldAcc_LatestWalk;}
}

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
        refresh_old_acceleration_for_tree_opening();
        gizmo_exit_bad_stop_if_requested("gravtree:before_treebuild"); CPU_Step[CPU_DRIFT] += measure_time(); /* sync before we do the treebuild */
        force_treebuild(NumPart, NULL);
        /* The tree just built is current by construction: the build set every
         * node's Ti_current to All.Ti_Current and refilled the whole SoA mirror,
         * local and foreign, from those nodes.  Record that here, at the call
         * site that KNOWS this is the main step tree, rather than inside
         * force_treebuild -- which is also used to build group-local and subset
         * trees whose geometry must never be certified for the step's device
         * consumers.  The full-drift test is the remaining precondition: it is
         * what makes the node geometry describe this time rather than merely
         * being freshly written, and move_particles above drifts only the active
         * set.  Absent that proof nothing is recorded and consumers fall back to
         * the host, which is correct but slower -- never wrong.
         * Recorded AFTER the bad-stop drain below: force_treebuild can request a
         * controlled stop during GPU finalize / LET / pseudo handling and still
         * return, and a tree whose build asked to stop must never be recorded as
         * current -- not even for the few statements before the poll exits. */
        gizmo_exit_bad_stop_if_requested("gravtree:after_treebuild"); CPU_Step[CPU_TREEBUILD] += measure_time(); /* and sync after treebuild as well */
        if(gizmo_full_drift_ti() == All.Ti_Current) {gpu_gravity_tree_mark_born_current(All.Ti_Current);}
        report_memory_ledger_on_growth("post-treebuild");  /* after force_treebuild (LET exchange ran); rebuild-only all-rank boundary */
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
            gizmo_exit_bad_stop_if_requested("gravtree:after_refresh_moments"); /* drain refresh bad-stop before any gravity walk */
            CPU_Step[CPU_TREEBUILD] += measure_time();
            TreeMomentsStaleFlag = 0;
        }
    }

    CPU_Step[CPU_TREEMISC] += measure_time(); t0 = my_second(); double child0_span = CPU_ChildCharged;
#ifndef SELFGRAVITY_OFF
    /* allocate buffers to arrange communication */
    PRINT_STATUS(" ..Begin tree force. (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));
    /* These two tables only RECORD targets whose gravity the locally-built tree cannot supply:
     * the MPI export round-trip is retired (gravity runs on the target-owning rank via the GPU
     * pre-pass and the locally-essential tree), so nothing here is ever sent, and a single entry
     * is enough to stop the run below. They are therefore held to a fixed small capacity rather
     * than to the communication chunk size, which used to reserve ~100 MB per rank on every
     * gravity call for something a healthy run never writes to at all. */
    All.BunchSize = GRAVITY_LET_DETECTOR_ENTRIES;
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    int k, ewald_max, diff, ndone, ndone_flag, place, recvTask; double tstart, tend, ax, ay, az; MPI_Status status;
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
                {
                    if(ThisTask == 0) {printf("TakeLevel < 0, even though we should have a slot\n"); fflush(stdout);}
                    endrun(90001008);
                    gizmo_exit_bad_stop_if_requested("gravtree:takelevel_no_slot");  /* symmetric (global LevelToTimeBin + bins): all ranks poll together */
                }
        }
    }
    else
    { /* in this case we do not measure gravity cost. Check whether this time-level
         has previously mean measured. If yes, then delete it so to make sure that it is not out of time */
        for(i = 0; i < GRAVCOSTLEVELS; i++) {if(All.LevelToTimeBin[i] == All.HighestActiveTimeBin) {All.LevelToTimeBin[i] = 0;}}
        TakeLevel = -1;
    }
    if(TakeLevel >= 0) {
        /* Under UVM-canonical particles, arena_P aliases host P[] — an arena
         * mirror write to P_arena_zero[i] is a self-assignment, so the single
         * host write is sufficient for both views. */
        for(i = 0; i < NumPart; i++) { P[i].GravCost[TakeLevel] = 0; }
    } /* re-zero the cost [will be re-summed] */

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

        /* Speculative GPU pre-pass: walks the local tree on GPU for each
         * active particle; on success, writes GravAccel and marks
         * ProcessedFlag so the CPU primary loop below skips it.
         * On pseudo-particle hit, leaves the particle untouched for the
         * CPU loop + MPI export machinery to handle unchanged. Ewald_iter
         * splits primary (==0) vs Ewald-correction (==1) walks; both are
         * active on all Kokkos builds. */
        if(Ewald_iter == 0) {gpu_gravtree_walk_primary();}
        else                {gpu_ewald_walk_primary();}

        do /* primary point-element loop */
        {
            iter++;
            BufferFullFlag = 0; Nexport = 0; tstart = my_second();

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
             * CPU gravity export round-trip is RETIRED.
             * ------------------------------------------------------------
             * The GPU pre-pass + Locally Essential Tree supply all
             * foreign-rank gravity on the target-owning rank, so the legacy
             * MPI export round-trip (compaction, MPI_Alltoall/Sendrecv,
             * imported-particle walk, scatter-back) and its gravdata_in/out
             * buffers are fully removed.  What survives is a DETECTOR:
             * DataIndexTable/DataNodeList are no longer shipped -- the tree
             * walk only records LET-incompleteness into Nexport.
             *
             * If Nexport > 0 a particle's gravity is not covered by LET, and
             * we request a graceful controlled-stop here (drained at the
             * all-rank poll below, no retry, no new collective).  It is not a
             * capacity shortfall: an import too large for the foreign-node
             * index range is reported by the exchange, which raises the range
             * and rebuilds before this walk runs.
             * ============================================================ */
            /* Import completeness.  The walk records rather than reports: it runs threaded over
             * targets, one incompleteness can involve many nodes, and a stop request only takes
             * effect at the poll below.  Speak once for the pass here, then ask for that stop --
             * the forces on those targets were computed from an import that did not cover them. */
            if(gravity_report_incomplete_import() > 0) {
                gizmo_request_controlled_stop(90000087, "gravtree: the imported tree did not carry the structure the walk resolved",
                                              __FILE__, __LINE__, __FUNCTION__);
            }

            if(Nexport > 0) {
                printf("The locally essential tree did not cover the gravity of %ld particles on rank %d. Stopping.\n", Nexport, ThisTask);
                fflush(stdout);
                /* Graceful soft-stop: the export round-trip is retired, so we cannot service
                 * these particles -- but the same loop iteration reaches the all-rank
                 * MPI_Allreduce + gizmo_exit_bad_stop_if_requested poll below, which drains
                 * this flag cleanly (no retry, no new collective). */
                gizmo_request_controlled_stop(914040, "gravtree: the locally essential tree did not cover some targets' gravity", __FILE__, __LINE__, __FUNCTION__);
            }

            /* Export-back loop is retired under GPU offload, so the arena
             * is not invalidated by host-side P[] writes here.
             * There is no gpu_particles_arena_invalidate() call at this point:
             * the arena is coherent here (gpu_gravtree_walk_primary invalidates
             * internally if its host scatter happens). A redundant
             * double-invalidate would cost nothing but blocks fast-path
             * acquires after the arena refresh, so it is intentionally absent.
             *
             * If a pre-acquire host mutation ever makes the arena stale at this
             * point, the GIZMO_GPU_ARENA_DEBUG=1 byte-compare guard will abort
             * with the site name -- rely on that trip wire instead of a
             * cargo-cult invalidate. */
            if(NextParticle >= (int)ActiveParticleList.size()) {ndone_flag = 1;} else {ndone_flag = 0;} /* figure out if we are done with the particular active set here */
            tstart = my_second();
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); /* call an allreduce to figure out if all tasks are also done here, otherwise we need to iterate */
            tend = my_second(); timewait2 += timediff(tstart, tend);
            gizmo_exit_bad_stop_if_requested("gravtree:tree_export_loop"); /* drain a buffer-too-small bad-stop here instead of retrying the export with zero progress */
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
    /* Switch to the relative opening criterion for the following force computations.
     * (Second-order ICs keep Barnes-Hut on the very first step.) */
    double errtol_before = All.ErrTolTheta;
    int enable_relative_opening =
        (All.TypeOfOpeningCriterion == 1) &&
        !(header.flag_ic_info == FLAG_SECOND_ORDER_ICS && All.Ti_Current == 0 && RestartFlag == 0);
    if(enable_relative_opening) { All.ErrTolTheta = 0; }
    /* The opening criterion just changed; the installed LET was built/exported under the previous
     * criterion and is not valid for the next walk. Rebuild the tree+LET before it is reused. */
    if(errtol_before != 0 && All.ErrTolTheta == 0) { TreeReconstructFlag = 1; }
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
        double dt = get_particle_timestep_in_physical(i, P);
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

        /* Measure the acceleration scale the relative tree-opening criterion will use, HERE: after
         * the G multiplication above, and before the companion subtraction, radiation pressure and
         * analytic-gravity terms below.  That keeps it a property of the force the gravity tree
         * computes -- the error the opening criterion exists to control -- rather than of every
         * force acting on the particle, which would let rapidly varying radiation or an external
         * field decide how finely the tree is opened.  It is only staged here; the tree build
         * promotes it into OldAcc, so the value cannot shift under a tree whose import was pruned
         * against it.  (Particles that skipped the walk above never reach this and keep the scale
         * from their last real walk, as before.) */
        if(!(header.flag_ic_info == FLAG_SECOND_ORDER_ICS && All.Ti_Current == 0 && RestartFlag == 0)) /* 2lpt ICs keep masses in OldAcc until the first evaluation */
        {
            auto accel_for_opening = P[i].GravAccel;
#ifdef PMGRID
            accel_for_opening += P[i].GravPM;
#endif
            P[i].OldAcc_LatestWalk = accel_for_opening.norm() / All.G;   /* back to non-G units, matching the predicate */
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
        if((P[i].Type==0) && (P[i].Mass>0)
#ifdef HYDRO_MULTIFLUID_DM
           && (P[i].FluidType != FLUID_DM)   /* dark fluid does not feel baryonic RT radiation pressure */
#endif
          )
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

    /* Arena mirror-update is a no-op under UVM-canonical (arena_P
     * aliases host P[]); the post-loop above already wrote canonical state. */

#endif /* end SELFGRAVITY operations (check if SELFGRAVITY_OFF not enabled) */


    add_analytic_gravitational_forces(); /* add analytic terms, which -CAN- be enabled even if self-gravity is not */


    /* Now the force computation is finished: gather timing and diagnostic information */
    t1 = my_second(); cpu_chain_sync(t1); timeall = cpu_minus_children(timediff(t0, t1), child0_span);
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
    /* The span from the end of the build to here is the force walk.  Only the
     * wait is separately measured inside it, so the walk row is the rest of the
     * span; charging the remainder to `misc` instead reported the whole walk as
     * unattributed.  The send/recv and second-walk timers this routine used to
     * split out have no writer any more and are not charged. */
    CPU_Step[CPU_TREEWALK1] += timeall - timewait2;
    CPU_Step[CPU_TREEWAIT2] += timewait2;
#ifdef OUTPUT_ADDITIONAL_RUNINFO
    if(ThisTask == 0)
    {
        fprintf(FdTimings, "Step= %lld  t= %.16g  dt= %.16g \n",(long long) All.NumCurrentTiStep, All.Time, All.TimeStep);
        fprintf(FdTimings, "Nf= %d%09d  total-Nf= %d%09d  ex-frac= %g (%g) iter= %d\n", (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), (int) (All.TotNumOfForces / 1000000000), (int) (All.TotNumOfForces % 1000000000), n_exported / ((double) GlobNumForceUpdate), N_nodesinlist / ((double) n_exported + 1.0e-10), iter); /* note: on Linux, the 8-byte integer could be printed with the format identifier "%qd", but doesn't work on AIX */
        fprintf(FdTimings, "work-load balance: %g (%g %g) rel1to2=%g   max=%g avg=%g\n", maxt / (1.0e-6 + sumt / NTask), maxt1 / (1.0e-6 + sumt1 / NTask), maxt2 / (1.0e-6 + sumt2 / NTask), sumt1 / (1.0e-6 + sumt1 + sumt2), maxt, sumt / NTask);
        fprintf(FdTimings, "particle-load balance: %g\n", plb_max);
        fprintf(FdTimings, "max. nodes: %d, filled: %g\n", maxnumnodes, maxnumnodes / ((double) MaxNodes));
        fprintf(FdTimings, "part/sec=%g | %g  ia/part=%g (%g)\n", GlobNumForceUpdate / (sumt + 1.0e-20), GlobNumForceUpdate / (1.0e-6 + maxt * NTask), ((double) (sum_costtotal)) / (1.0e-20 + GlobNumForceUpdate), ((double) ewaldtot) / (1.0e-20 + GlobNumForceUpdate)); fprintf(FdTimings, "\n");
        fflush(FdTimings);
    }
    double costtotal_new = 0, sum_costtotal_new;
    if(TakeLevel >= 0)
    {
        for(i = 0; i < NumPart; i++) {costtotal_new += P[i].GravCost[TakeLevel];}
        MPI_Reduce(&costtotal_new, &sum_costtotal_new, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        /* Both walks accumulate the same per-target interaction count into GravCost and
         * into Costtotal, so the two totals describe the same quantity and this should be
         * at round-off whichever path each target took. A non-negligible value means the
         * two are no longer measuring the same thing. */
        if(sum_costtotal>0) {PRINT_STATUS(" ..relative error in the total number of tree-gravity interactions = %g", (sum_costtotal - sum_costtotal_new) / sum_costtotal);}
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
            /* SSOT pre-walk candidacy (Mass>0 + Hermite eligibility + needs_new_treeforce);
             * non-candidates are marked done so the finalization loop skips them. */
            if(!gravity_treewalk_candidate_prewalk(i)) {ProcessedFlag[i]=1; continue;}

#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC) && !defined(PMGRID)
            if(Ewald_iter)
            {
                ret = force_treeevaluate_ewald_correction(i, exportflag, exportnodecount, exportindex);
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
                ret = force_treeevaluate(i, exportflag, exportnodecount, exportindex);
                if(ret < 0) {buffer_full = 1; break;}
                /* Work weight for the next domain decomposition: the count of
                 * interactions this target performed. The device walk records the
                 * same quantity (gpu_gravtree.cc), so a step whose walks are split
                 * between the two paths feeds one consistent measure to
                 * domain_particle_costfactor(). Each thread writes only its own
                 * target, so no synchronization is needed. */
                if(TakeLevel >= 0) {P[i].GravCost[TakeLevel] = ret;}
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


/* The DataIndexTable sorter (data_index_compare / mysort_dataindex) is retired with
 * the gravity export round-trip: the detector never ships or sorts its records. */


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

/* SSOT pre-walk gravity tree-walk candidacy: true iff active particle i will
 * receive a real tree-force walk this step (and thus consume the installed LET).
 * Shared by the CPU primary, GPU primary, and GPU Ewald walk filters (and, later,
 * the LET-freshness check) so the freshness basis matches the actual LET consumer.
 * Mass>0 enforces the scheduler contract (Mass<=0 = scheduled-for-deletion, never a
 * valid gravity target) uniformly -- a defensive parity guard (the Ewald walk
 * already filtered it; the primary walks relied on the active-list builder and the
 * device early-return). ProcessedFlag is NOT part of candidacy -- it is per-walk
 * done-bookkeeping each caller keeps separately. */
int gravity_treewalk_candidate_prewalk(int i)
{
    if(P[i].Mass <= 0) {return 0;}
#ifdef HERMITE_INTEGRATION
    if(HermiteOnlyFlag && !eligible_for_hermite(i)) {return 0;}
#endif
#ifdef ADAPTIVE_TREEFORCE_UPDATE
    if(!needs_new_treeforce(i)) {return 0;}
#endif
    return 1;
}
