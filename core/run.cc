#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#ifdef GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
#include "../mesh/neighbor_list.h"
#endif


/*! \file run.c
 *  \brief  iterates over timesteps, main loop
 */
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * heavily (adding/removing calls, re-ordering some routines, and
 * adding hooks to new elements such as particle splitting, as necessary)
 * for GIZMO by Phil Hopkins (phopkins@caltech.edu) and Mike Grudic (also
 * adding options needed for higher-order Runge-Kutta and Hermite integration)
 */


/* RT_STEP_DIAG: checksum function for bisecting RT divergence */
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
static int rt_step_diag_count = 0;
static void rt_step_checksum(const char *label) {
    double sum_RadE = 0, sum_Trad = 0, sum_u = 0, sum_Tdust = 0, sum_ne = 0;
    int ngas = 0;
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type == 0 && P[i].Mass > 0) {
            sum_RadE += CellP[i].Rad_E_gamma[RT_FREQ_BIN_INFRARED];
            sum_Trad += CellP[i].Radiation_Temperature;
            sum_Tdust += CellP[i].Dust_Temperature;
            sum_u += CellP[i].InternalEnergy;
            sum_ne += CellP[i].Ne;
            ngas++;
            /* RT_PART: per-particle tracking for specific IDs */
            if(P[i].ID == 1 || P[i].ID == 100 || P[i].ID == 1000) {
                printf("[RT_PART] %-20s ID=%llu T=%.6e Tdust=%.6e Trad=%.6e u=%.10e Ne=%.6e RadE_IR=%.6e P=%.6e\n",
                    label, (unsigned long long)P[i].ID, CellP[i].Temperature, CellP[i].Dust_Temperature,
                    CellP[i].Radiation_Temperature, CellP[i].InternalEnergy, CellP[i].Ne,
                    CellP[i].Rad_E_gamma[RT_FREQ_BIN_INFRARED], CellP[i].Pressure);
            }
        }
    }
    if(ThisTask == 0) {
        printf("[RT_STEP] %-20s  ngas=%d  sum_RadE_IR=%.10e  sum_Trad=%.6e  sum_Tdust=%.6e  sum_u=%.10e  sum_Ne=%.6e\n",
            label, ngas, sum_RadE, sum_Trad, sum_Tdust, sum_u, sum_ne);
        fflush(stdout);
    }
}
#endif

/*! This routine contains the main simulation loop that iterates over
 * single timesteps. The loop terminates when the cpu-time limit is
 * reached, when a `stop' file is found in the output directory, or
 * when the simulation ends because we arrived at TimeMax.
 */
void run(void)
{
    CPU_Step[CPU_MISC] += measure_time();

    if(RestartFlag != 1)		/* need to compute forces at initial synchronization time, unless we restarted from restart files */
    {
        output_log_messages();

        domain_Decomposition(0, 0, 0);

        set_non_standard_physics_for_current_time();

        compute_grav_accelerations();	/* compute gravitational accelerations for synchronous particles */

        compute_hydro_densities_and_forces();	/* densities, gradients, & hydro-accels for synchronous particles */

        calculate_non_standard_physics();	/* source terms are here treated in a strang-split fashion */
    }

    while(1)			/* main timestep iteration loop */
    {
        compute_statistics();	/* regular statistics outputs (like total energy) */

        write_cpu_log();		/* output some CPU usage log-info (accounts for everything needed up to the current sync-point) */

        if((All.Ti_Current >= TIMEBASE) || (All.Time > All.TimeMax)) /* check whether we reached the final time */
        {
            if(ThisTask == 0) {printf("\nFinal time=%g reached. Simulation ends.\n", All.TimeMax);}
            restart(0); /* write a restart file to allow continuation of the run for a larger value of TimeMax */
            if(All.Ti_lastoutput != All.Ti_Current) {savepositions(All.SnapshotFileCount++);} /* make a snapshot at the final time in case none has produced at this time; this will be overwritten if All.TimeMax is increased and the run is continued */
            break;
        }

        find_timesteps();		/* find-timesteps */

        /* RT_STEP_DIAG: print RT field checksums after each major phase to locate divergence. */
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
        if(rt_step_diag_count < 50) { rt_step_diag_count++; rt_step_checksum("after_find_timesteps"); }
#endif
        int TreeReconstructFlag_local = TreeReconstructFlag;
        /* Phase 9.6: auto-rebuild guardrail.  force_add_element_to_tree
         * insertions stale the LET / pseudo-particle moments shipped on
         * the last full build.  When the global insertion count exceeds
         * 1% of TotNumPart, force a full rebuild on the next step.
         * Mass+CoM remain conserved so the bound is conservative; tighten
         * if a workload exposes drift in moment-sensitive diagnostics. */
        long long add_elem_calls_global = 0;
        MPI_Allreduce(&ForceAddElementToTree_CallsSinceBuild, &add_elem_calls_global,
                      1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(add_elem_calls_global > (long long)(0.01 * All.TotNumPart))
        {
            if(ThisTask == 0)
                printf("Phase 9.6 LET guardrail: %lld force_add_element_to_tree insertions since last build "
                       "(> 1%% of TotNumPart=%lld) -- forcing tree rebuild this step.\n",
                       add_elem_calls_global, (long long) All.TotNumPart);
            TreeReconstructFlag_local = 1;
        }
#ifdef HERMITE_INTEGRATION
        HermiteOnlyFlag = 1;
        gravity_tree();	/* re-compute gravitational accelerations for synchronous particles */
        HermiteOnlyFlag = 0;
#endif
        do_first_halfstep_kick();	/* half-step kick at beginning of timestep for synchronous particles */
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
        if(rt_step_diag_count <= 50) rt_step_checksum("after_kick1");
#endif

        find_next_sync_point_and_drift();	/* find next synchronization point and drift particles to this time.
                                             * If needed, this function will also write an output file
                                             * at the desired time.
                                             */

        output_log_messages();	/* write some info to log-files */

        set_non_standard_physics_for_current_time();	/* update auxiliary physics for current time */

        int reconstructed_tree = 0;
        int NeedFullDomainDecomp = TreeReconstructFlag; /* save whether a full rebuild was requested before the SINGLE_STAR counter check */
#if defined(SINGLE_STAR_SINK_DYNAMICS)
        if(All.NumForcesSinceLastDomainDecomp > All.TreeDomainUpdateFrequency * All.TotNumPart) {TreeReconstructFlag_local = 1;}
#endif
        MPI_Allreduce(&TreeReconstructFlag_local, &TreeReconstructFlag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // if one process reconstructs the tree then everbody has to
        MPI_Allreduce(MPI_IN_PLACE, &NeedFullDomainDecomp, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(GlobNumForceUpdate > All.TreeDomainUpdateFrequency * All.TotNumPart)	/* check whether we have a big step */
        {
#ifdef DOMAIN_LIGHTWEIGHT_REPARTITION
            if(!NeedFullDomainDecomp) {domain_Decomposition_light(0);}  /* lightweight repartition: reuse top tree, just rebalance */
            else
#endif
            {domain_Decomposition(0, 0, 1);}  /* full decomposition needed */
            reconstructed_tree = 1;
        }
        else if(TreeReconstructFlag) {domain_Decomposition(0, 0, 1); reconstructed_tree = 1;}
        else
        {
            force_update_tree();	/* update tree dynamically with kicks of last step so that it can be reused */
            make_list_of_active_particles();	/* now we can set the new chain list of active particles */
        }

        compute_grav_accelerations();	/* compute gravitational accelerations for synchronous particles */

#ifdef GALSF_SUBGRID_WINDS
#if (GALSF_SUBGRID_WIND_SCALING==2)
/*
#ifdef PMGRID
        //if(All.Ti_Current == All.PM_Ti_endstep && get_random_number(1+All.Ti_Current) < 0.05) // compute the DM velocity dispersion around gas particles every 20 PM steps, should be sufficient ? not ideal for many applications, in fact, now only acts on active //
#else
        //if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) // only acts on top-level timebin -- only enable this if you are trying to radically reduce the number of operations of this mode //
#endif
*/
        {
            disp_density();
        }
#endif
#endif

        /* flag particles which will be feedback centers, so kernel lengths can be computed for them */
#ifdef GALSF_FB_MECHANICAL
        determine_where_SNe_occur(); // for mechanical FB models
#endif
#ifdef GALSF_FB_THERMAL
        determine_where_addthermalFB_events_occur(); // (same, but for simple thermal feedback models)
#endif

        compute_hydro_densities_and_forces();	/* densities, gradients, & hydro-accels for synchronous particles */
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
        if(rt_step_diag_count <= 50) rt_step_checksum("after_hydro");
#endif

#ifdef PARTICLE_MERGE_SPLIT_EVERY_TIMESTEP // do merge/split routines every single timestep - need to do it here if we didn't do it during domain decomp on a coarse timestep
        if(!reconstructed_tree)
        {
            merge_and_split_particles();
            rearrange_particle_sequence();
        }
#endif
        
        do_second_halfstep_kick();	/* this does the half-step kick at the end of the timestep */
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
        if(rt_step_diag_count <= 50) rt_step_checksum("after_kick2");
#endif

        calculate_non_standard_physics();	/* source terms are here treated in a strang-split fashion */

#ifdef HERMITE_INTEGRATION // we do a prediction step using the saved "old" pos, accel and jerk from the beginning of the timestep. Then we recompute accel and jerk and do the correction
        do_hermite_prediction();
        HermiteOnlyFlag = 2;
        gravity_tree();	/* re-compute gravitational accelerations for synchronous particles */
        HermiteOnlyFlag = 0;
        do_hermite_correction();
#endif
        /* Check whether we need to interrupt the run */
        int stopflag = 0;
        if(ThisTask == 0)
        {
            FILE *fd;
            char stopfname[DEFAULT_PATH_BUFFERSIZE_TOUSE];
            snprintf(stopfname, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%sstop", All.OutputDir);
            if((fd = fopen(stopfname, "r")))	/* Is the stop-file present? If yes, interrupt the run. */
            {
                fclose(fd);
                stopflag = 1;
                unlink(stopfname);
            }

            if(CPUThisRun > 0.85 * All.TimeLimitCPU)	/* are we running out of CPU-time ? If yes, interrupt run. */
            {
                printf("reaching time-limit. stopping.\n");
                stopflag = 2;
            }
        }

        MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stopflag)
        {
            restart(0);		/* write restart file */
            MPI_Barrier(MPI_COMM_WORLD);

            if(stopflag == 2 && ThisTask == 0)
            {
                FILE *fd;
                char contfname[DEFAULT_PATH_BUFFERSIZE_TOUSE];
                snprintf(contfname, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%scont", All.OutputDir);
                if((fd = fopen(contfname, "w")))
                    fclose(fd);

                if(All.ResubmitOn)
                    execute_resubmit_command();
            }
            return;
        }

        if(ThisTask == 0)
        {
            /* is it time to write one of the regularly space restart-files? */
            if((CPUThisRun - All.TimeLastRestartFile) >= All.CpuTimeBetRestartFile)
            {
                All.TimeLastRestartFile = CPUThisRun;
                stopflag = 3;
            }
            else
                stopflag = 0;
        }

        MPI_Bcast(&stopflag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stopflag == 3)
        {
            restart(0);		/* write an occasional restart file */
            stopflag = 0;
            All.TimeLastRestartFile += report_time();
        }

        report_memory_usage(&HighMark_run, "RUN");
    }

}



void set_non_standard_physics_for_current_time(void)
{
#if defined(COOLING) && !defined(CHIMES)
    /* set UV background for the current time */
    IonizeParams();
#endif

#if defined(COOL_METAL_LINES_BY_SPECIES) && !defined(CHIMES)
    /* load the metal-line cooling tables appropriate for the UV background */
    if(All.ComovingIntegrationOn) {LoadMultiSpeciesTables();}
#endif

#if defined(GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF)
    update_stellarnumber_and_timedistribofstarformation();
#endif
}



void calculate_non_standard_physics(void)
{
#ifdef PARTICLE_EXCISION
    apply_excision();
#endif


#if defined(TURB_DRIVING) && defined(TURB_DRIVING_SPECTRUMGRID)
    if(All.Time >= All.TimeNextTurbSpectrum) {powerspec_turb(All.FileNumberTurbSpectrum++); All.TimeNextTurbSpectrum += All.TimeBetTurbSpectrum;}
#endif


#ifdef SINK_PARTICLES /***** sink accretion and feedback *****/
    CPU_Step[CPU_MISC] += measure_time();
#ifdef GALSF_LIMIT_FBTIMESTEPS_FROM_BELOW
    if(All.Dt_Since_LastFBCalc_Gyr >= All.Dt_Min_Between_FBCalc_Gyr)
#endif
    {
        sink_accretion();
#ifdef SINK_WIND_SPAWN
        double Max_Unspawned_MassUnits_fromSink_global;
        MPI_Allreduce(&Max_Unspawned_MassUnits_fromSink, &Max_Unspawned_MassUnits_fromSink_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(Max_Unspawned_MassUnits_fromSink_global > 1)
        {
            spawn_sink_wind_feedback();
            rearrange_particle_sequence();
            Max_Unspawned_MassUnits_fromSink=Max_Unspawned_MassUnits_fromSink_global=0.;
        }
#if defined(SNE_NONSINK_SPAWN)
        {int i; for(i=0;i<NumPart;i++) {if(P[i].Type != 4) {continue;}
            double n_unspawned = P[i].unspawned_wind_mass / ((SINK_WIND_SPAWN)*target_mass_for_wind_spawning(i)); // number of spawned gas cells that can be made from the mass in the reservoir
            if(n_unspawned> Max_Unspawned_MassUnits_fromSink) {Max_Unspawned_MassUnits_fromSink = n_unspawned;} // track the maximum integer number of elements this sink could spawn
        }}
#endif
#endif
        MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_SINKS] += measure_time();
    }
#endif


#if (defined(SINK_PARTICLES) || defined(GALSF_SUBGRID_WINDS)) && defined(FOF)
    if(All.Time >= All.TimeNextOnTheFlyFoF) {fof_fof(-1); /* this will find new sink seed halos and/or assign host halo masses for the variable wind model */
        if(All.ComovingIntegrationOn) {All.TimeNextOnTheFlyFoF *= All.TimeBetOnTheFlyFoF;} else {All.TimeNextOnTheFlyFoF += All.TimeBetOnTheFlyFoF;}}
#endif

#ifdef TRANSPORT_SUBCYCLE
    /* --- compute the global number of transport subcycles --- */
    {
        double min_transport_dt_local = MAX_REAL_NUMBER, max_hydro_dt_local = 0;
        for(int idx : ActiveParticleList) {
            if(P[idx].Type != 0 || P[idx].Mass <= 0) continue;
            double hydro_dt = get_particle_timestep_in_physical(idx);
            min_transport_dt_local = DMIN(min_transport_dt_local, CellP[idx].Transport_Dt_Subcycle);
            max_hydro_dt_local = DMAX(max_hydro_dt_local, hydro_dt);
        }
        double min_transport_dt_global, max_hydro_dt_global;
        MPI_Allreduce(&min_transport_dt_local, &min_transport_dt_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&max_hydro_dt_local, &max_hydro_dt_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        All.Transport_Subcycle_N = 1;
        if(min_transport_dt_global > 0 && min_transport_dt_global < MAX_REAL_NUMBER && max_hydro_dt_global > min_transport_dt_global)
            All.Transport_Subcycle_N = IMIN((int)ceil(max_hydro_dt_global / min_transport_dt_global), TRANSPORT_SUBCYCLE);
        All.Transport_Subcycle_dt_fraction = 1.0 / (double)All.Transport_Subcycle_N;
        if(ThisTask == 0 && All.Transport_Subcycle_N > 1)
            printf("Transport subcycling: %d sub-steps (hydro_dt/transport_dt = %.1f)\n",
                   All.Transport_Subcycle_N, max_hydro_dt_global / min_transport_dt_global);
    }
    /* Save the hydro-pass DtInternalEnergy before the subcycle loop. rt_update_driftkick adds
       IR gas heating to DtInternalEnergy each sub-step; without resetting, it accumulates N-fold.
       We reset to the hydro value before each kick so only one sub-step's IR contribution is present. */
#if defined(RT_INFRARED) && defined(COOLING)
    for(int idx : ActiveParticleList) {
        if(P[idx].Type == 0 && P[idx].Mass > 0)
            CellP[idx].DtIE_IR_Subcycle = CellP[idx].DtInternalEnergy;
    }
#endif
    for(int transport_sub = 0; transport_sub < All.Transport_Subcycle_N; transport_sub++) {
#endif // TRANSPORT_SUBCYCLE

#ifdef RADTRANSFER
    CPU_Step[CPU_MISC] += measure_time();
#if defined(RT_SOURCE_INJECTION)
#ifdef TRANSPORT_SUBCYCLE
    if(transport_sub == 0) /* source injection only on first sub-step */
#endif
    {
        int flag; flag=1;
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
        flag = Flag_FullStep; /* for continous injection, requires all sources and gas be active synchronously or else 2x-counts */
#endif
#if !defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
        if(flag) {rt_source_injection();} /* source injection into neighbor gas particles (only on full timesteps, if using non-discrete scheme) */
#endif
    }
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
    if(rt_step_diag_count <= 50) rt_step_checksum("after_rt_source");
#endif
#endif
#if defined(RT_DIFFUSION_CG) /* use the CG method to solve the RT diffusion equation implicitly for all particles; do only on full timesteps, requires synchronous timestepping right now */
    if(Flag_FullStep) {All.Radiation_Ti_endstep = All.Ti_Current; rt_diffusion_cg_solve(); All.Radiation_Ti_begstep = All.Radiation_Ti_endstep;}
#endif
#if defined(RT_CHEM_PHOTOION) && !defined(COOLING)
#ifdef TRANSPORT_SUBCYCLE
    if(transport_sub == 0) /* chemistry update only on first sub-step (cooling handles it on subsequent sub-steps if TRANSPORT_SUBCYCLE_COOLING) */
#endif
    rt_update_chemistry(); /* chemistry updated at sub-stepping as well */
#ifdef OUTPUT_ADDITIONAL_RUNINFO
    if(Flag_FullStep) {rt_write_chemistry_stats();}
#endif
#endif
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_RTNONFLUXOPS] += measure_time();
#endif // RADTRANSFER block

#ifdef TRANSPORT_SUBCYCLE
    /* --- recompute transport fluxes every sub-step and apply kick --- */
    transport_subcycle_exchange_fluxes();
    /* Reset DtInternalEnergy to hydro-pass value before each kick, so rt_update_driftkick's
       IR gas heating contribution doesn't accumulate across sub-steps. */
#if defined(RT_INFRARED) && defined(COOLING)
    for(int idx : ActiveParticleList) {
        if(P[idx].Type == 0 && P[idx].Mass > 0)
            CellP[idx].DtInternalEnergy = CellP[idx].DtIE_IR_Subcycle;
    }
#endif
    transport_subcycle_kick();
#endif

#if defined(TRANSPORT_SUBCYCLE_COOLING) && defined(COOLING)
    /* save DtInternalEnergy before cooling (it gets overwritten with CGS-converted value inside cooling).
       Restore the original code-units value before each sub-step so the CGS conversion isn't applied twice. */
    {
#ifndef COOLING_OPERATOR_SPLIT
        for(int idx : ActiveParticleList) {
            if(P[idx].Type == 0 && P[idx].Mass > 0)
                CellP[idx].Dt_Transport_Subcycle_Saved = CellP[idx].DtInternalEnergy;
        }
#endif
        cooling_parent_routine();
#ifndef COOLING_OPERATOR_SPLIT
        for(int idx : ActiveParticleList) {
            if(P[idx].Type == 0 && P[idx].Mass > 0)
                CellP[idx].DtInternalEnergy = CellP[idx].Dt_Transport_Subcycle_Saved;
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_COOLINGSFR] += measure_time();
#endif

#ifdef TRANSPORT_SUBCYCLE
    } /* end transport subcycle loop */
#ifdef GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
    /* TRANSPORT_SUBCYCLE path: symlist + ghosts were kept alive past hydro_force
       (gizmo_hydro_cleanup_symlist_and_ghosts() skipped itself under this flag).
       Free them now that the RT subcycle loop is done. */
    gizmo_sym_neighbor_list_free();
    ghost_exchange_cleanup();
#endif
    /* After the loop DtInternalEnergy = DtIE_IR_Subcycle + IR_rate_last_substep, which is correct:
       the pre-kick reset already prevents N-fold accumulation, so the cooling solver and second
       KDK half-kick see exactly one sub-step's IR contribution at the final Rad_E_gamma state. */
#if defined(TRANSPORT_SUBCYCLE_COOLING) && !defined(COOLING_OPERATOR_SPLIT)
    /* zero DtInternalEnergy after the subcycle loop — the hydro work has been fully applied across all sub-steps */
    for(int idx : ActiveParticleList) {
        if(P[idx].Type == 0 && P[idx].Mass > 0 && CellP[idx].CoolingIsOperatorSplitThisTimestep==0)
            CellP[idx].DtInternalEnergy = 0;
    }
#endif
#endif

#ifdef NUCLEAR_NETWORK
    nuclear_parent_routine(); // nuclear burning (operator-split, before cooling; fixup is done inside on compact arrays) //
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_COOLINGSFR] += measure_time();
#endif

#if defined(COOLING) && !defined(TRANSPORT_SUBCYCLE_COOLING)
    cooling_parent_routine(); // top-level cooling and chemistry subroutine //
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_COOLINGSFR] += measure_time(); // finish time calc for SFR+cooling
#endif
#if defined(RT_INFRARED) && defined(COOLING) && defined(GIZMO_DEBUG_RT_COOLING)
        if(rt_step_diag_count <= 50) rt_step_checksum("after_cooling");
#endif


#ifdef GALSF /* star/sink particle formation */
    star_formation_parent_routine(); // top-level star formation routine (because this involves common particle conversions, want to keep this at end of this subroutine) //
    MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_COOLINGSFR] += measure_time(); // finish time calc for SFR+cooling
#endif

#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    int i; for (int i : ActiveParticleList){if(P[i].Type == 5 && P[i].do_gas_search_this_timestep){P[i].dt_since_last_gas_search = 0;}}
#endif

}



void compute_statistics(void)
{
    if((All.Time - All.TimeLastStatistics) >= All.TimeBetStatistics)
    {
#if !defined(EVALPOTENTIAL)          // compute_potential is not defined if EVALPOTENTIAL is on //
#ifdef COMPUTE_POTENTIAL_ENERGY
        compute_potential();
#endif
#endif
        energy_statistics();	/* compute and output energy statistics */
        All.TimeLastStatistics += All.TimeBetStatistics;
    }
}



void execute_resubmit_command(void)
{
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s", All.ResubmitCommand);
    system(buf);
}



/*! This function finds the next synchronization point of the system
 * (i.e. the earliest point of time any of the particles needs a force
 * computation), and drifts the system to this point of time.  If the
 * system drifts over the desired time of a snapshot file, the
 * function will drift to this moment, generate an output, and then
 * resume the drift.
 */
void find_next_sync_point_and_drift(void)
{
  int n, i, prev;
  integertime dt_bin, ti_next_for_bin, ti_next_kick, ti_next_kick_global;
  int highest_active_bin, highest_occupied_bin;
  double timeold;

  timeold = All.Time;

  All.NumCurrentTiStep++;	/* we are now moving to the next sync point */

  /* find the next kick time */
  for(n = 0, ti_next_kick = TIMEBASE, highest_occupied_bin = 0; n < TIMEBINS; n++)
    {
      if(TimeBinCount[n])
	{
	  if(n > 0)
	    {
	      highest_occupied_bin = n;
	      dt_bin = GET_INTEGERTIME_FROM_TIMEBIN(n);
	      ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin;	/* next kick time for this timebin */
	    }
	  else
	    {
	      dt_bin = 0;
	      ti_next_for_bin = All.Ti_Current;
	    }

	  if(ti_next_for_bin < ti_next_kick)
	    ti_next_kick = ti_next_for_bin;
	}
    }

  MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_TYPE_TIME, MPI_MIN, MPI_COMM_WORLD);

  while(ti_next_kick_global >= All.Ti_nextoutput && All.Ti_nextoutput >= 0)
    {
        All.Ti_Current = All.Ti_nextoutput;

        if(All.ComovingIntegrationOn) {All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);}
            else {All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;}

        set_cosmo_factors_for_current_time();

        move_particles(All.Ti_nextoutput);
        MPI_Barrier(MPI_COMM_WORLD); CPU_Step[CPU_DRIFT] += measure_time();

#ifdef OUTPUT_POTENTIAL
#if !defined(EVALPOTENTIAL) || (defined(EVALPOTENTIAL) && defined(OUTPUT_RECOMPUTE_POTENTIAL))
        domain_Decomposition(0, 0, 0);
        compute_potential();
#endif
#endif

        savepositions(All.SnapshotFileCount++);	/* write snapshot file */
        All.Ti_nextoutput = find_next_outputtime(All.Ti_nextoutput + 1);
    }


  All.Previous_Ti_Current = All.Ti_Current;
  All.Ti_Current = ti_next_kick_global;

  if(All.ComovingIntegrationOn) {All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);}
    else {All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;}

  set_cosmo_factors_for_current_time();
#ifdef OPENMP_GPU_OFFLOAD
  gizmo_gpu_sync_all();
#endif
#ifdef BOX_SHEARING
    calc_shearing_box_pos_offset();
#endif

#ifdef GR_TABULATED_COSMOLOGY_G
  All.G = All.Gini * dGfak(All.Time);
#endif

  All.TimeStep = All.Time - timeold;

  /* mark the bins that will be active */
  for(n = 1, TimeBinActive[0] = 1, NumForceUpdate = TimeBinCount[0], highest_active_bin = 0; n < TIMEBINS; n++)
    {
      dt_bin = GET_INTEGERTIME_FROM_TIMEBIN(n);
      if((ti_next_kick_global % dt_bin) == 0)
	{
	  TimeBinActive[n] = 1;
	  NumForceUpdate += TimeBinCount[n];
	  if(TimeBinCount[n])
	    highest_active_bin = n;
	}
      else
	TimeBinActive[n] = 0;
    }

  sumup_large_ints(1, &NumForceUpdate, &GlobNumForceUpdate);
  All.NumForcesSinceLastDomainDecomp += GlobNumForceUpdate;
  MPI_Allreduce(&highest_active_bin, &All.HighestActiveTimeBin, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&highest_occupied_bin, &All.HighestOccupiedTimeBin, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if(GlobNumForceUpdate == All.TotNumPart)
    {
      Flag_FullStep = 1;
      if(All.HighestActiveTimeBin != All.HighestOccupiedTimeBin)
	terminate("Something is wrong with the time bins.\n");
    }
  else
    Flag_FullStep = 0;




  /* move the new set of active/synchronized particles. Note: We do not yet call make_list_of_active_particles(), since we
   * may still need to old list in the dynamic tree update */
  for(n = 0, prev = -1; n < TIMEBINS; n++)
    {if(TimeBinActive[n]) {for(i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i]) {drift_particle(i, All.Ti_Current);}}}

#ifdef OPENMP_GPU_OFFLOAD
  refresh_timestep_dilation_factors_for_gpu();
#endif

}


void make_list_of_active_particles(void)
{
    ActiveParticleList.clear();
    for(int n = 0; n < TIMEBINS; n++)
    {
        if(TimeBinActive[n])
        {
            for(int i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i])
            {
                if(P[i].Mass > 0) {ActiveParticleList.push_back(i);}
            }
        }
    }
}





/*! this function returns the next output time that is equal or larger to
 *  ti_curr
 */
integertime find_next_outputtime(integertime ti_curr)
{
  long long i, iter = 0;
  integertime ti, ti_next;
  double next, time;

  DumpFlag = 1;
  ti_next = -1;


  if(All.OutputListOn)
    {
      for(i = 0; i < All.OutputListLength; i++)
	{
	  time = All.OutputListTimes[i];

	  if(time >= All.TimeBegin && time <= All.TimeMax)
	    {
	      if(All.ComovingIntegrationOn) {ti = (integertime) (log(time / All.TimeBegin) / All.Timebase_interval);}
          else {ti = (integertime) ((time - All.TimeBegin) / All.Timebase_interval);}

	      if(ti >= ti_curr)
		{
		  if(ti_next == -1)
		    {
		      ti_next = ti;
		      DumpFlag = All.OutputListFlag[i];
		      if(i > All.SnapshotFileCount) {All.SnapshotFileCount = i;}
		    }

		  if(ti_next > ti)
		    {
		      ti_next = ti;
		      DumpFlag = All.OutputListFlag[i];
		      if(i > All.SnapshotFileCount) {All.SnapshotFileCount = i;}
		    }
		}
	    }
	}
    }
  else
    {
      if(All.ComovingIntegrationOn)
	{
	  if(All.TimeBetSnapshot <= 1.0)
	    {
	      printf("TimeBetSnapshot > 1.0 required for your simulation.\n");
	      endrun(13123);
	    }
	}
      else
	{
	  if(All.TimeBetSnapshot <= 0.0)
	    {
	      printf("TimeBetSnapshot > 0.0 required for your simulation.\n");
	      endrun(13123);
	    }
	}
      time = All.TimeOfFirstSnapshot;

      iter = 0;

      while(time < All.TimeBegin)
	{
	  if(All.ComovingIntegrationOn)
	    time *= All.TimeBetSnapshot;
	  else
	    time += All.TimeBetSnapshot;

	  iter++;

	  if(iter > 10000000000)
	    {
          printf("Can't determine next output time. iter=%lld time=%g All.TimeBegin=%g All.TimeBetSnapshot=%g All.TimeOfFirstSnapshot=%g \n",iter,time,All.TimeBegin,All.TimeBetSnapshot,All.TimeOfFirstSnapshot);
	      endrun(110);
	    }
	}
      while(time <= All.TimeMax)
	{
	  if(All.ComovingIntegrationOn) {ti = (integertime) (log(time / All.TimeBegin) / All.Timebase_interval);}
        else {ti = (integertime) ((time - All.TimeBegin) / All.Timebase_interval);}

	  if(ti >= ti_curr)
	    {
	      ti_next = ti;
	      break;
	    }

	  if(All.ComovingIntegrationOn)
	    time *= All.TimeBetSnapshot;
	  else
	    time += All.TimeBetSnapshot;

	  iter++;

	  if(iter > 10000000000)
	    {
          printf("Can't determine next output time. iter=%lld time=%g All.TimeBegin=%g All.TimeMax=%g All.TimeBetSnapshot=%g All.TimeOfFirstSnapshot=%g All.Timebase_interval=%g \n",iter,time,All.TimeBegin,All.TimeMax,All.TimeBetSnapshot,All.TimeOfFirstSnapshot,All.Timebase_interval);
	      endrun(111);
	    }
	}
    }


  if(ti_next == -1)
    {
      ti_next = 2 * TIMEBASE;	/* this will prevent any further output */
      if(ThisTask == 0) {printf("\nThere is no valid time for a further snapshot file.\n");}
    }
  else
    {
      if(All.ComovingIntegrationOn) {next = All.TimeBegin * exp(ti_next * All.Timebase_interval);}
      else {next = All.TimeBegin + ti_next * All.Timebase_interval;}

      if(ThisTask == 0) {printf("\nSetting next time for snapshot file to Time_next= %.16g  (DumpFlag=%d)\n", next, DumpFlag);}

    }

  return ti_next;
}




/*! This routine writes for every synchronisation point in the timeline information to two log-files:
 * In FdInfo, we just list the timesteps that have been done, while in
 * FdTimebins we inform about the distribution of particles over the timebins, and which timebins are active on this step.
 * code is stored.
 */
void output_log_messages(void)
{
  double z;
  int i, j;
  long long tot, tot_gas;
  long long tot_count[TIMEBINS];
  long long tot_count_gas[TIMEBINS];
  long long tot_cumulative[TIMEBINS];
  int weight, corr_weight;
  double sum, avg_CPU_TimeBin[TIMEBINS], frac_CPU_TimeBin[TIMEBINS];

  sumup_large_ints(TIMEBINS, TimeBinCount, tot_count);
  sumup_large_ints(TIMEBINS, TimeBinCountGas, tot_count_gas);

#if defined(IO_SUPPRESS_TIMEBIN_STDOUT)
    if((ThisTask == 0) && (All.HighestActiveTimeBin>=(TIMEBINS-IO_SUPPRESS_TIMEBIN_STDOUT)))
#else
    if(ThisTask == 0)
#endif
    {
        if(All.ComovingIntegrationOn)
        {
            z = 1.0 / (All.Time) - 1;
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdInfo, "Sync-Point %lld, Time: %.16g, Redshift: %g, Nf = %d%09d, Systemstep: %g, Dloga: %g\n",
                    (long long) All.NumCurrentTiStep, All.Time, z, (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
            fflush(FdInfo);
            fprintf(FdTimebin, "Sync-Point %lld, Time: %.16g, Redshift: %g, Systemstep: %g, Dloga: %g\n", (long long) All.NumCurrentTiStep, All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
#endif
            printf("\nSync-Point %lld, Time: %.16g, Redshift: %g, Systemstep: %g, Dloga: %g\n", (long long) All.NumCurrentTiStep, All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
        }
        else
        {
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdInfo, "Sync-Point %lld, Time: %.16g, Nf = %d%09d, Systemstep: %g\n", (long long) All.NumCurrentTiStep,
                    All.Time, (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), All.TimeStep);
            fflush(FdInfo);
            fprintf(FdTimebin, "Sync-Point %lld, Time: %.16g, Systemstep: %g\n", (long long) All.NumCurrentTiStep, All.Time, All.TimeStep);
#endif
            printf("\nSync-Point %lld, Time: %.16g, Systemstep: %g\n", (long long) All.NumCurrentTiStep, All.Time, All.TimeStep);
        }

        for(i = 1, tot_cumulative[0] = tot_count[0]; i < TIMEBINS; i++) {tot_cumulative[i] = tot_count[i] + tot_cumulative[i - 1];}


      for(i = 0; i < TIMEBINS; i++)
	{
	  for(j = 0, sum = 0; j < All.CPU_TimeBinCountMeasurements[i]; j++) {sum += All.CPU_TimeBinMeasurements[i][j];}
	  if(All.CPU_TimeBinCountMeasurements[i]) {avg_CPU_TimeBin[i] = sum / All.CPU_TimeBinCountMeasurements[i];} else {avg_CPU_TimeBin[i] = 0;}
	}

      for(i = All.HighestOccupiedTimeBin, weight = 1, sum = 0; i >= 0 && tot_count[i] > 0; i--, weight *= 2)
	{
	  if(weight > 1) {corr_weight = weight / 2;} else {corr_weight = weight;}
	  frac_CPU_TimeBin[i] = corr_weight * avg_CPU_TimeBin[i];
	  sum += frac_CPU_TimeBin[i];
	}

      for(i = All.HighestOccupiedTimeBin; i >= 0 && tot_count[i] > 0; i--) {if(sum) {frac_CPU_TimeBin[i] /= sum;}}


        printf("Occupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac\n");
#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdTimebin,"Occupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac\n");
#endif
        for(i = TIMEBINS - 1, tot = tot_gas = 0; i >= 0; i--)
            if(tot_count_gas[i] > 0 || tot_count[i] > 0)
            {
                printf(" %c  bin=%2d      %10llu  %10llu   %16.12f       %10llu %c %c  %10.2f    %5.1f%%\n", TimeBinActive[i] ? 'X' : ' ', i, tot_count[i] - tot_count_gas[i], tot_count_gas[i],
                       GET_INTEGERTIME_FROM_TIMEBIN(i) * All.Timebase_interval, tot_cumulative[i], (i == All.HighestActiveTimeBin) ? '<' : ' ',
                       (tot_cumulative[i] > All.TreeDomainUpdateFrequency * All.TotNumPart) ? '*' : ' ', avg_CPU_TimeBin[i], 100.0 * frac_CPU_TimeBin[i]);
#ifdef OUTPUT_ADDITIONAL_RUNINFO
                fprintf(FdTimebin," %c  bin=%2d      %10llu  %10llu   %16.12f       %10llu %c %c  %10.2f    %5.1f%%\n", TimeBinActive[i] ? 'X' : ' ', i, tot_count[i] - tot_count_gas[i], tot_count_gas[i],
                        GET_INTEGERTIME_FROM_TIMEBIN(i) * All.Timebase_interval, tot_cumulative[i], (i == All.HighestActiveTimeBin) ? '<' : ' ',
                        (tot_cumulative[i] > All.TreeDomainUpdateFrequency * All.TotNumPart) ? '*' : ' ', avg_CPU_TimeBin[i], 100.0 * frac_CPU_TimeBin[i]);
#endif
                if(TimeBinActive[i])
                {
                    tot += tot_count[i];
                    tot_gas += tot_count_gas[i];
                }
            }
        printf("               ------------------------\n");
#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdTimebin, "               ------------------------\n");
#endif
#ifdef PMGRID
        if(All.PM_Ti_endstep == All.Ti_Current)
        {
            printf("PM-Step. Total: %10llu  %10llu    Sum: %10llu\n\n", tot - tot_gas, tot_gas, tot);
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdTimebin, "PM-Step. Total: %10llu  %10llu    Sum: %10llu\n", tot - tot_gas, tot_gas, tot);
#endif
        }
        else
#endif
        {
            printf("Total active:   %10llu  %10llu    Sum: %10llu\n\n", tot - tot_gas, tot_gas, tot);
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            fprintf(FdTimebin, "Total active:   %10llu  %10llu    Sum: %10llu\n", tot - tot_gas, tot_gas, tot);
#endif
        }
#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdTimebin, "\n");
        fflush(FdTimebin);
#endif
    }

  output_extra_log_messages();
}




void write_cpu_log(void)
{
  double max_CPU_Step[CPU_PARTS], avg_CPU_Step[CPU_PARTS], t0, t1, tsum; int i; t0=0; t1=0; tsum=0;
  CPU_Step[CPU_MISC] += measure_time();

  for(i = 1, CPU_Step[0] = 0; i < CPU_PARTS; i++) {CPU_Step[0] += CPU_Step[i];}

  MPI_Reduce(CPU_Step, max_CPU_Step, CPU_PARTS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(CPU_Step, avg_CPU_Step, CPU_PARTS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
    {
      for(i = 0; i < CPU_PARTS; i++) {avg_CPU_Step[i] /= NTask;}

#ifdef OUTPUT_ADDITIONAL_RUNINFO
      put_symbol(0.0, 1.0, '#');
      for(i = 1, tsum = 0.0; i < CPU_PARTS; i++)
      {
            if(max_CPU_Step[i] > 0)
            {
              t0 = tsum; t1 = t0 + avg_CPU_Step[i] * (avg_CPU_Step[i] / max_CPU_Step[i]);
              put_symbol(t0 / avg_CPU_Step[0], t1 / avg_CPU_Step[0], CPU_Symbol[i]);
              tsum += t1 - t0;

              t0 = tsum; t1 = t0 + avg_CPU_Step[i] * ((max_CPU_Step[i] - avg_CPU_Step[i]) / max_CPU_Step[i]);
              put_symbol(t0 / avg_CPU_Step[0], t1 / avg_CPU_Step[0], CPU_SymbolImbalance[i]);
              tsum += t1 - t0;
            }
      }
      put_symbol(tsum / max_CPU_Step[0], 1.0, '-');
      fprintf(FdBalance, "Step=%7lld  sec=%10.3f  Nf=%2d%09d  %s\n", (long long) All.NumCurrentTiStep, max_CPU_Step[0], (int) (GlobNumForceUpdate / 1000000000), (int) (GlobNumForceUpdate % 1000000000), CPU_String); fflush(FdBalance);
#endif

      if(All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin] == NUMBER_OF_MEASUREMENTS_TO_RECORD)
	{
	  All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin]--;
	  memmove(&All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][0], &All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][1], (NUMBER_OF_MEASUREMENTS_TO_RECORD - 1) * sizeof(double));
	}

      All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin]++] = max_CPU_Step[0];
    }

    CPUThisRun += CPU_Step[0];

    for(i = 0; i < CPU_PARTS; i++) {CPU_Step[i] = 0;}
    if(ThisTask == 0)
    {
        for(i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] += avg_CPU_Step[i];}
    }

#ifndef OUTPUT_ADDITIONAL_RUNINFO
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) // only do the actual -print- operation on global timesteps
#endif
  if(ThisTask == 0)
    {
      fprintf(FdCPU, "Step %lld, Time: %.16g, CPUs: %d\n",(long long) All.NumCurrentTiStep, All.Time, NTask);
      fprintf(FdCPU, "Nactive=%lld, Imbal(Max/Mean)=%g \n", (long long) GlobNumForceUpdate, (max_CPU_Step[0]/(MIN_REAL_NUMBER + avg_CPU_Step[0])-1.)*NTask+1.);
      fprintf(FdCPU,
	      "total         %10.2f  %5.1f%%\n"
	      "tree+gravity  %10.2f  %5.1f%%\n"
	      "   treebuild  %10.2f  %5.1f%%\n"
	      "   treewalk   %10.2f  %5.1f%%\n"
	      "   treecomm   %10.2f  %5.1f%%\n"
	      "   treeimbal  %10.2f  %5.1f%%\n"
#ifdef PMGRID
          "pm-gravity    %10.2f  %5.1f%%\n"
#endif
#if !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUT_POTENTIAL))
          "potentialeval %10.2f  %5.1f%%\n"
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
	      "ags-nongas    %10.2f  %5.1f%%\n"
	      "   agsdensity %10.2f  %5.1f%%\n"
	      "   agscomm    %10.2f  %5.1f%%\n"
	      "   agsimbal   %10.2f  %5.1f%%\n"
          "   agsmisc    %10.2f  %5.1f%%\n"
#endif
#ifdef TURB_DIFF_DYNAMIC
          "dyndiff       %10.2f  %5.1f%%\n"
          "   compute    %10.2f  %5.1f%%\n"
          "   comm       %10.2f  %5.1f%%\n"
          "   wait       %10.2f  %5.1f%%\n"
          "   misc       %10.2f  %5.1f%%\n"
          "velsmooth     %10.2f  %5.1f%%\n"
          "   compute    %10.2f  %5.1f%%\n"
          "   comm       %10.2f  %5.1f%%\n"
          "   wait       %10.2f  %5.1f%%\n"
          "   misc       %10.2f  %5.1f%%\n"
#endif
#ifdef GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
	      "hydro/fluids  %10.2f  %5.1f%%\n"
	      "   density    %10.2f  %5.1f%%\n"
	      "   ghost_xchg %10.2f  %5.1f%%\n"
	      "   gradients  %10.2f  %5.1f%%\n"
	      "   hydro_frc  %10.2f  %5.1f%%\n"
	      "   symlist    %10.2f  %5.1f%%\n"
	      "   (unused)   %10.2f  %5.1f%%\n"
	      "   hmaxupdate %10.2f  %5.1f%%\n"
          "   misc_hydro %10.2f  %5.1f%%\n"
#else
	      "hydro/fluids  %10.2f  %5.1f%%\n"
	      "   dens+grad  %10.2f  %5.1f%%\n"
	      "   denscomm   %10.2f  %5.1f%%\n"
	      "   densimbal  %10.2f  %5.1f%%\n"
	      "   hydrofrc   %10.2f  %5.1f%%\n"
	      "   hydcomm    %10.2f  %5.1f%%\n"
	      "   hydimbal   %10.2f  %5.1f%%\n"
	      "   hmaxupdate %10.2f  %5.1f%%\n"
          "   hydmisc    %10.2f  %5.1f%%\n"
#endif
	      "domain        %10.2f  %5.1f%%\n"
          "peano         %10.2f  %5.1f%%\n"
#ifdef FOF
          "fof/subfind   %10.2f  %5.1f%%\n"
#endif
          "drift/splitmg %10.2f  %5.1f%%\n"
	      "kicks         %10.2f  %5.1f%%\n"
	      "io/snapshots  %10.2f  %5.1f%%\n"
#ifdef COOLING
	      "cooling+chem  %10.2f  %5.1f%%\n"
#endif
#ifdef CHIMES
	      " coolchmimbal %10.2f  %5.1f%%\n"
#endif
#ifdef SINK_PARTICLES
	      "sinks         %10.2f  %5.1f%%\n"
#endif
#ifdef GRAIN_FLUID
          "grains        %10.2f  %5.1f%%\n"
#endif
#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)
          "mech_fb_loop  %10.2f  %5.1f%%\n"
#endif
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
          "hII_fb_loop   %10.2f  %5.1f%%\n"
#endif
#if defined(GALSF_FB_FIRE_RT_LOCALRP)
          "localwindkik  %10.2f  %5.1f%%\n"
#endif
#if defined(RADTRANSFER)
          "rt_nonfluxops %10.2f  %5.1f%%\n"
#endif
          "misc          %10.2f  %5.1f%%\n",

    All.CPU_Sum[CPU_ALL], 100.0,
    All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2] + All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]
              + All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2] + All.CPU_Sum[CPU_TREEBUILD] + All.CPU_Sum[CPU_TREEMISC],
    (All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2] + All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]
              + All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2] + All.CPU_Sum[CPU_TREEBUILD] + All.CPU_Sum[CPU_TREEMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEBUILD], (All.CPU_Sum[CPU_TREEBUILD]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2], (All.CPU_Sum[CPU_TREEWALK1] + All.CPU_Sum[CPU_TREEWALK2]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV], (All.CPU_Sum[CPU_TREESEND] + All.CPU_Sum[CPU_TREERECV]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2], (All.CPU_Sum[CPU_TREEWAIT1] + All.CPU_Sum[CPU_TREEWAIT2]) / All.CPU_Sum[CPU_ALL] * 100,
#ifdef PMGRID
    All.CPU_Sum[CPU_MESH], (All.CPU_Sum[CPU_MESH]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUT_POTENTIAL))
    All.CPU_Sum[CPU_POTENTIAL], (All.CPU_Sum[CPU_POTENTIAL]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    All.CPU_Sum[CPU_AGSDENSCOMPUTE] + All.CPU_Sum[CPU_AGSDENSWAIT] + All.CPU_Sum[CPU_AGSDENSCOMM] + All.CPU_Sum[CPU_AGSDENSMISC],
              (All.CPU_Sum[CPU_AGSDENSCOMPUTE] + All.CPU_Sum[CPU_AGSDENSWAIT] + All.CPU_Sum[CPU_AGSDENSCOMM] + All.CPU_Sum[CPU_AGSDENSMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSCOMPUTE], (All.CPU_Sum[CPU_AGSDENSCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSCOMM], (All.CPU_Sum[CPU_AGSDENSCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSWAIT], (All.CPU_Sum[CPU_AGSDENSWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_AGSDENSMISC], (All.CPU_Sum[CPU_AGSDENSMISC]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef TURB_DIFF_DYNAMIC
    (All.CPU_Sum[CPU_DYNDIFFCOMPUTE] + All.CPU_Sum[CPU_DYNDIFFWAIT] + All.CPU_Sum[CPU_DYNDIFFCOMM] + All.CPU_Sum[CPU_DYNDIFFMISC]), (All.CPU_Sum[CPU_DYNDIFFCOMPUTE] + All.CPU_Sum[CPU_DYNDIFFWAIT] + All.CPU_Sum[CPU_DYNDIFFCOMM] + All.CPU_Sum[CPU_DYNDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFCOMPUTE], (All.CPU_Sum[CPU_DYNDIFFCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFWAIT], (All.CPU_Sum[CPU_DYNDIFFWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFCOMM], (All.CPU_Sum[CPU_DYNDIFFCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DYNDIFFMISC], (All.CPU_Sum[CPU_DYNDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    (All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE] + All.CPU_Sum[CPU_IMPROVDIFFWAIT] + All.CPU_Sum[CPU_IMPROVDIFFCOMM] + All.CPU_Sum[CPU_IMPROVDIFFMISC]), (All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE] + All.CPU_Sum[CPU_IMPROVDIFFWAIT] + All.CPU_Sum[CPU_IMPROVDIFFCOMM] + All.CPU_Sum[CPU_IMPROVDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE], (All.CPU_Sum[CPU_IMPROVDIFFCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFWAIT], (All.CPU_Sum[CPU_IMPROVDIFFWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFCOMM], (All.CPU_Sum[CPU_IMPROVDIFFCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_IMPROVDIFFMISC], (All.CPU_Sum[CPU_IMPROVDIFFMISC]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
    All.CPU_Sum[CPU_DENSCOMPUTE] + All.CPU_Sum[CPU_DENSCOMM] + All.CPU_Sum[CPU_DENSWAIT] + All.CPU_Sum[CPU_DENSMISC]
              + All.CPU_Sum[CPU_HYDCOMPUTE] + All.CPU_Sum[CPU_HYDCOMM] + All.CPU_Sum[CPU_HYDMISC]
              + All.CPU_Sum[CPU_HYDWAIT] + All.CPU_Sum[CPU_TREEHMAXUPDATE],
    (All.CPU_Sum[CPU_DENSCOMPUTE] + All.CPU_Sum[CPU_DENSCOMM] + All.CPU_Sum[CPU_DENSWAIT] + All.CPU_Sum[CPU_DENSMISC]
              + All.CPU_Sum[CPU_HYDCOMPUTE] + All.CPU_Sum[CPU_HYDCOMM] + All.CPU_Sum[CPU_HYDMISC]
              + All.CPU_Sum[CPU_HYDWAIT] + All.CPU_Sum[CPU_TREEHMAXUPDATE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DENSCOMPUTE], (All.CPU_Sum[CPU_DENSCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DENSCOMM], (All.CPU_Sum[CPU_DENSCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DENSWAIT], (All.CPU_Sum[CPU_DENSWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDCOMPUTE], (All.CPU_Sum[CPU_HYDCOMPUTE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDCOMM], (All.CPU_Sum[CPU_HYDCOMM]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDWAIT], (All.CPU_Sum[CPU_HYDWAIT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TREEHMAXUPDATE], (All.CPU_Sum[CPU_TREEHMAXUPDATE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_HYDMISC] + All.CPU_Sum[CPU_DENSMISC], (All.CPU_Sum[CPU_HYDMISC] + All.CPU_Sum[CPU_DENSMISC]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_DOMAIN], (All.CPU_Sum[CPU_DOMAIN]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_PEANO], (All.CPU_Sum[CPU_PEANO]) / All.CPU_Sum[CPU_ALL] * 100,
#ifdef FOF
    All.CPU_Sum[CPU_FOF], (All.CPU_Sum[CPU_FOF]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
    All.CPU_Sum[CPU_DRIFT], (All.CPU_Sum[CPU_DRIFT]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_TIMELINE], (All.CPU_Sum[CPU_TIMELINE]) / All.CPU_Sum[CPU_ALL] * 100,
    All.CPU_Sum[CPU_SNAPSHOT], (All.CPU_Sum[CPU_SNAPSHOT]) / All.CPU_Sum[CPU_ALL] * 100,
#ifdef COOLING
    All.CPU_Sum[CPU_COOLINGSFR], (All.CPU_Sum[CPU_COOLINGSFR]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef CHIMES
    All.CPU_Sum[CPU_COOLSFRIMBAL], (All.CPU_Sum[CPU_COOLSFRIMBAL]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef SINK_PARTICLES
    All.CPU_Sum[CPU_SINKS], (All.CPU_Sum[CPU_SINKS]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#ifdef GRAIN_FLUID
    All.CPU_Sum[CPU_DRAGFORCE], (All.CPU_Sum[CPU_DRAGFORCE]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)
    All.CPU_Sum[CPU_SNIIHEATING], (All.CPU_Sum[CPU_SNIIHEATING]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
    All.CPU_Sum[CPU_HIIHEATING], (All.CPU_Sum[CPU_HIIHEATING]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(GALSF_FB_FIRE_RT_LOCALRP)
    All.CPU_Sum[CPU_LOCALWIND], (All.CPU_Sum[CPU_LOCALWIND]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
#if defined(RADTRANSFER)
    All.CPU_Sum[CPU_RTNONFLUXOPS], (All.CPU_Sum[CPU_RTNONFLUXOPS]) / All.CPU_Sum[CPU_ALL] * 100,
#endif
    All.CPU_Sum[CPU_MISC], (All.CPU_Sum[CPU_MISC]) / All.CPU_Sum[CPU_ALL] * 100);

    fprintf(FdCPU, "\n");
    fflush(FdCPU);
    }
}


void put_symbol(double t0, double t1, char c)
{
    int i, j;
    i = (int) (t0 * CPU_STRING_LEN + 0.5);
    j = (int) (t1 * CPU_STRING_LEN);
    if(i < 0) {i = 0;}
    if(j < 0) {j = 0;}
    if(i >= CPU_STRING_LEN) {i = CPU_STRING_LEN;}
    if(j >= CPU_STRING_LEN) {j = CPU_STRING_LEN;}
    while(i <= j) {CPU_String[i++] = c;}
    CPU_String[CPU_STRING_LEN] = 0;
}




/*! This routine first calls a computation of various global
 * quantities of the particle distribution, and then writes some
 * statistics about the energies in the various particle components to
 * the file FdEnergy.
 */
void energy_statistics(void)
{
#ifdef OUTPUT_ADDITIONAL_RUNINFO
  compute_global_quantities_of_system();

  if(ThisTask == 0)
    {
      fprintf(FdEnergy,
	      "%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g",
	      All.Time, SysState.EnergyInt, SysState.EnergyPot, SysState.EnergyKin, SysState.EnergyIntComp[0],
	      SysState.EnergyPotComp[0], SysState.EnergyKinComp[0], SysState.EnergyIntComp[1],
	      SysState.EnergyPotComp[1], SysState.EnergyKinComp[1], SysState.EnergyIntComp[2],
	      SysState.EnergyPotComp[2], SysState.EnergyKinComp[2], SysState.EnergyIntComp[3],
	      SysState.EnergyPotComp[3], SysState.EnergyKinComp[3], SysState.EnergyIntComp[4],
	      SysState.EnergyPotComp[4], SysState.EnergyKinComp[4], SysState.EnergyIntComp[5],
	      SysState.EnergyPotComp[5], SysState.EnergyKinComp[5], SysState.MassComp[0],
	      SysState.MassComp[1], SysState.MassComp[2], SysState.MassComp[3], SysState.MassComp[4],
	      SysState.MassComp[5]);

      fprintf(FdEnergy," \n");
      fflush(FdEnergy);
    }
#endif
}



void output_extra_log_messages(void)
{
#if defined(TURB_DRIVING) && defined(OUTPUT_ADDITIONAL_RUNINFO)
    log_turb_temp();
#endif

#if defined(GR_TABULATED_COSMOLOGY) && defined(OUTPUT_ADDITIONAL_RUNINFO)
    if((ThisTask == 0) && (All.ComovingIntegrationOn == 1)
    {
        double hubble_a;

        hubble_a = hubble_function(All.Time);
        fprintf(FdDE, "%lld %.16g %e ", (long long) All.NumCurrentTiStep, All.Time, hubble_a);
#ifndef GR_TABULATED_COSMOLOGY_W
        fprintf(FdDE, "%e ", All.DarkEnergyConstantW);
#else
        fprintf(FdDE, "%e %e ", get_wa(All.Time), DarkEnergy_a(All.Time));
#endif
#ifdef GR_TABULATED_COSMOLOGY_G
        fprintf(FdDE, "%e %e", dHfak(All.Time), dGfak(All.Time));
#endif
        fprintf(FdDE, "\n");
        fflush(FdDE);
    }
#endif
}
