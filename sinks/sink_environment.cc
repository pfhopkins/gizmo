/*! \file sink_environment.c
*  \brief routines for evaluating sink particle environment
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "sinks_gpu_decls.h"
#include "sink_environment_mode_b.h"
/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/


#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //


/* for new quantities calculated in environment loop, divide out weights and convert to physical units */
static void sink_normalize_temp_info_struct_after_environment_loop(int i)
{
    if(SinkTempInfo[i].Mgas_in_Kernel > 0)
    {
        SinkTempInfo[i].Sink_SurroudingGasInternalEnergy /= SinkTempInfo[i].Mgas_in_Kernel;
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        SinkTempInfo[i].Sink_SurroundingGasVel /= SinkTempInfo[i].Mgas_in_Kernel * All.cf_atime;
#endif
    }
    else {SinkTempInfo[i].Sink_SurroudingGasInternalEnergy = 0;}
    SinkTempInfo[i].Malt_in_Kernel += (SinkTempInfo[i].Mgas_in_Kernel + SinkTempInfo[i].Mstar_in_Kernel);
    SinkTempInfo[i].Jalt_in_Kernel += (SinkTempInfo[i].Jgas_in_Kernel + SinkTempInfo[i].Jstar_in_Kernel);
#ifdef SINK_REPOSITION_ON_POTMIN
    double Mass_in_Kernel = SinkTempInfo[i].DF_rms_vel;
    if(Mass_in_Kernel > 0) {SinkTempInfo[i].DF_mean_vel /= Mass_in_Kernel * All.cf_atime;}
#endif
}


void sink_environment_loop(void)
{
    /* GPU neighbor-list path: build active-sink index + radius arrays, dispatch
       GPU kernel, scatter outputs into SinkTempInfo. */
    CPU_Step[CPU_SINK_ENV] += measure_time();

    /* Count active sinks FIRST (cheap host loop over ActiveParticleList).
     * If no rank has any active sinks, skip ghost_prep + arena_acquire +
     * everything else. ghost_prep does an all-particles drift loop (12.4M
     * cache-miss reads on fire_m11i) so this guard saves ~1s/step on
     * sink-inactive timebins. Multi-rank correctness requires the
     * MPI_Allreduce so all ranks agree to skip or proceed together. */
    int num_active = 0;
    for(int i : ActiveParticleList) { if(sink_isactive(i)) num_active++; }
    int global_num_active = num_active;
    MPI_Allreduce(&num_active, &global_num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(global_num_active == 0) {
        CPU_Step[CPU_SINK_ENV] += measure_time();
        return;
    }

    int *nl_active = (int *) mymalloc("sinkenv_nl_active", (num_active > 0 ? num_active : 1) * sizeof(int));
    double *nl_radii = (double *) mymalloc("sinkenv_nl_radii", (num_active > 0 ? num_active : 1) * sizeof(double));
    struct sink_env_gpu_out *nl_outs = (struct sink_env_gpu_out *)
        mymalloc("sinkenv_nl_outs", (num_active > 0 ? num_active : 1) * sizeof(struct sink_env_gpu_out));

    int aa = 0;
    for(int i : ActiveParticleList) {
        if(sink_isactive(i)) { nl_active[aa] = i; nl_radii[aa] = P[i].KernelRadius; aa++; }
    }

    /* Mode B SPIKE dispatch (env-gated, 2026-05-07). When the env flag is OFF,
     * this branch is bypassed entirely — no new collective, no behavior change.
     * When ON, the dispatch is collective (Allreduce sum/max of num_active)
     * and ALL ranks take the same Mode B vs legacy path.
     *
     * SPIKE GUARDS — both DELETE in runner extraction. The real Mode A/B
     * dispatch threshold lives in the runner spec, not as a hard-coded constant. */
    bool sinkenv_imported_ghosts = false;
    bool mode_b_taken             = false;
    if(sink_env1_mode_b_env_enabled()) {
        const int MODE_B_SINK_ENV1_SPIKE_GUARD_SUM = 64;
        const int MODE_B_SINK_ENV1_SPIKE_GUARD_MAX = 64;
        int local_act = num_active;
        int sum_act = 0, max_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_act, &max_act, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        mode_b_taken = (sum_act > 0)
                    && (sum_act <= MODE_B_SINK_ENV1_SPIKE_GUARD_SUM)
                    && (max_act <= MODE_B_SINK_ENV1_SPIKE_GUARD_MAX);
        if(mode_b_taken) {
            /* SPIKE: Mode B path. NO ghost prep, NO GPU NGL, NO SIDX. */
            sink_env1_mode_b_evaluate(P, CellP, nl_active, num_active, nl_radii, nl_outs);
        }
    }
    if(!mode_b_taken) {
        /* Existing path — byte-identical to pre-spike when env is off. */
        sinkenv_imported_ghosts = gizmo_explicit_query_prep_ghosts_fresh(
            "sink_env1", NGB_SEARCH_SYMMETRIC, (unsigned int)SINK_NEIGHBOR_BITFLAG,
            nl_active, num_active, nl_radii, gizmo_ghost_safety_factor());

        ghost_write_detector_begin("sink_environment");
#ifdef SINGLE_STAR_SINK_DYNAMICS
        ghost_writeback_zero_swallowtime();
#endif
        sink_environment_evaluate_gpu(P, CellP, NumPart, nl_active, num_active,
                                      nl_radii, SINK_NEIGHBOR_BITFLAG, nl_outs);
#ifdef SINGLE_STAR_SINK_DYNAMICS
        ghost_writeback_swallowtime();
#endif
        ghost_write_detector_end();

        /* MODEB_XVAL pre-scatter dump for cross-validation against env-on
         * (Mode B) run. Same line shape as sink_environment_mode_b.cc emits.
         * Gated by GIZMO_MODE_B_XVAL_DUMP=1. Instrumentation-only; no behavior
         * change when off. */
        {
            static const char *xv_env = getenv("GIZMO_MODE_B_XVAL_DUMP");
            static const int xv_on = (xv_env && xv_env[0] == '1') ? 1 : 0;
            if(xv_on) {
                for(int a = 0; a < num_active; a++) {
                    printf("MODEB_XVAL rank=%d caller=sink_env1 active_local=%d "
                           "Mgas=%g Mstar=%g Malt=%g IE=%g "
                           "Jgas=%g,%g,%g Jstar=%g,%g,%g Jalt=%g,%g,%g\n",
                           ThisTask, a,
                           (double)nl_outs[a].Mgas_in_Kernel,
                           (double)nl_outs[a].Mstar_in_Kernel,
                           (double)nl_outs[a].Malt_in_Kernel,
                           (double)nl_outs[a].Sink_SurroudingGasInternalEnergy,
                           (double)nl_outs[a].Jgas_in_Kernel[0], (double)nl_outs[a].Jgas_in_Kernel[1], (double)nl_outs[a].Jgas_in_Kernel[2],
                           (double)nl_outs[a].Jstar_in_Kernel[0], (double)nl_outs[a].Jstar_in_Kernel[1], (double)nl_outs[a].Jstar_in_Kernel[2],
                           (double)nl_outs[a].Jalt_in_Kernel[0], (double)nl_outs[a].Jalt_in_Kernel[1], (double)nl_outs[a].Jalt_in_Kernel[2]);
                }
                fflush(stdout);
            }
        }
    }

    /* Scatter per-active-sink outputs into SinkTempInfo */
    for(int a = 0; a < num_active; a++) {
        int i = nl_active[a];
        int t = P[i].IndexMapToTempStruc;
        SinkTempInfo[t].Sink_SurroudingGasInternalEnergy += nl_outs[a].Sink_SurroudingGasInternalEnergy;
        SinkTempInfo[t].Mgas_in_Kernel  += nl_outs[a].Mgas_in_Kernel;
        SinkTempInfo[t].Mstar_in_Kernel += nl_outs[a].Mstar_in_Kernel;
        SinkTempInfo[t].Malt_in_Kernel  += nl_outs[a].Malt_in_Kernel;
        for(int k = 0; k < 3; k++) {
            SinkTempInfo[t].Jgas_in_Kernel[k]  += nl_outs[a].Jgas_in_Kernel[k];
            SinkTempInfo[t].Jstar_in_Kernel[k] += nl_outs[a].Jstar_in_Kernel[k];
            SinkTempInfo[t].Jalt_in_Kernel[k]  += nl_outs[a].Jalt_in_Kernel[k];
        }
#ifdef SINK_REPOSITION_ON_POTMIN
        SinkTempInfo[t].DF_rms_vel += nl_outs[a].DF_rms_vel;
        for(int k = 0; k < 3; k++) SinkTempInfo[t].DF_mean_vel[k] += nl_outs[a].DF_mean_vel[k];
        if(nl_outs[a].DF_mmax_particles > SinkTempInfo[t].DF_mmax_particles)
            SinkTempInfo[t].DF_mmax_particles = nl_outs[a].DF_mmax_particles;
#endif
#if defined(SINK_OUTPUT_MOREINFO)
        SinkTempInfo[t].Sfr_in_Kernel += nl_outs[a].Sfr_in_Kernel;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        for(int k = 0; k < 3; k++) SinkTempInfo[t].Sink_SurroundingGasVel[k] += nl_outs[a].Sink_SurroundingGasVel[k];
#endif
#if defined(JET_DIRECTION_FROM_KERNEL_AND_SINK)
        for(int k = 0; k < 3; k++) SinkTempInfo[t].Sink_SurroundingGasCOM[k] += nl_outs[a].Sink_SurroundingGasCOM[k];
#endif
#if (SINK_GRAVACCRETION == 8)
        SinkTempInfo[t].hubber_mdot_bondi_limiter   += nl_outs[a].hubber_mdot_bondi_limiter;
        SinkTempInfo[t].hubber_mdot_vr_estimator    += nl_outs[a].hubber_mdot_vr_estimator;
        SinkTempInfo[t].hubber_mdot_disk_estimator  += nl_outs[a].hubber_mdot_disk_estimator;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
        SinkTempInfo[t].mass_to_swallow_edd += nl_outs[a].mass_to_swallow_edd;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
        for(int k = 0; k < 3; k++) SinkTempInfo[t].angmom_prepass_sum_for_passback[k] += nl_outs[a].angmom_prepass_sum_for_passback[k];
#endif
#if defined(SINK_RETURN_BFLUX)
        SinkTempInfo[t].kernel_norm_topass_in_swallowloop += nl_outs[a].kernel_norm_topass_in_swallowloop;
#endif
    }

    myfree(nl_outs); myfree(nl_radii); myfree(nl_active);
    if(sinkenv_imported_ghosts && NTask > 1) { ghost_exchange_cleanup(); }
    /* final operations on results */
    {int i; for(i=0; i<N_active_loc_Sink; i++) {sink_normalize_temp_info_struct_after_environment_loop(i);}}
    CPU_Step[CPU_SINK_ENV] += measure_time(); /* collect timings and reset clock for next timing */
}








/* -----------------------------------------------------------------------------------------------------
 * DAA: modified versions of sink_environment_loop and sink_environment_evaluate for a second
 * environment loop. Here we do a Bulge-Disk kinematic decomposition for gravitational torque accretion
 * ----------------------------------------------------------------------------------------------------- */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)


void sink_environment_second_loop(void)
{
    /* Stage E2: GPU path — pure aggregator, no j-writes, same active set + radii +
     * j_type_bitmask as the first environment pass. */
    CPU_Step[CPU_SINK_ENV] += measure_time();

    /* Count-first guard (see sink_environment_loop for rationale). */
    int num_active = 0;
    for(int i : ActiveParticleList) { if(sink_isactive(i)) num_active++; }
    int global_num_active = num_active;
    MPI_Allreduce(&num_active, &global_num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(global_num_active == 0) {
        CPU_Step[CPU_SINK_ENV] += measure_time();
        return;
    }

    int alloc_n = (num_active > 0 ? num_active : 1);
    int    *nl_active   = (int *)    mymalloc("sinkenv2_nl_active", alloc_n * sizeof(int));
    double *nl_radii    = (double *) mymalloc("sinkenv2_nl_radii",  alloc_n * sizeof(double));
    MyFloat (*nl_Jgas)[3]  = (MyFloat (*)[3]) mymalloc("sinkenv2_Jgas",  alloc_n * sizeof(MyFloat[3]));
    MyFloat (*nl_Jstar)[3] = (MyFloat (*)[3]) mymalloc("sinkenv2_Jstar", alloc_n * sizeof(MyFloat[3]));
    struct sink_env_second_gpu_out *nl_outs = (struct sink_env_second_gpu_out *)
        mymalloc("sinkenv2_outs", alloc_n * sizeof(struct sink_env_second_gpu_out));

    {int aa = 0; for(int i : ActiveParticleList) {
        if(!sink_isactive(i)) continue;
        int t = P[i].IndexMapToTempStruc;
        nl_active[aa] = i;
        nl_radii[aa]  = P[i].KernelRadius;
        for(int k = 0; k < 3; k++) {
            nl_Jgas[aa][k]  = SinkTempInfo[t].Jgas_in_Kernel[k];
            nl_Jstar[aa][k] = SinkTempInfo[t].Jstar_in_Kernel[k];
        }
        aa++;
    }}

    bool sinkenv2_imported_ghosts = gizmo_explicit_query_prep_ghosts_fresh(
        "sink_env2", NGB_SEARCH_SYMMETRIC, (unsigned int)SINK_NEIGHBOR_BITFLAG,
        nl_active, num_active, nl_radii, gizmo_ghost_safety_factor());

    ghost_write_detector_begin("sink_environment_second");
    sink_environment_second_evaluate_gpu(P, CellP, NumPart,
                                          nl_active, num_active, nl_radii,
                                          nl_Jgas, nl_Jstar,
                                          SINK_NEIGHBOR_BITFLAG, nl_outs);
    ghost_write_detector_end();

    for(int a = 0; a < num_active; a++) {
        int i = nl_active[a];
        int t = P[i].IndexMapToTempStruc;
        SinkTempInfo[t].MgasBulge_in_Kernel  += nl_outs[a].MgasBulge_in_Kernel;
        SinkTempInfo[t].MstarBulge_in_Kernel += nl_outs[a].MstarBulge_in_Kernel;
    }

    myfree(nl_outs); myfree(nl_Jstar); myfree(nl_Jgas); myfree(nl_radii); myfree(nl_active);
    if(sinkenv2_imported_ghosts && NTask > 1) { ghost_exchange_cleanup(); }
    CPU_Step[CPU_SINK_ENV] += measure_time();
}

#endif   //SINK_GRAVACCRETION == 0




#endif // top-level flag
