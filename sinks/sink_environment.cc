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
#include "../mesh/neighbor_loop_runner.h"
#include "sink_env1_loop.h"
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


/* Scatter the per-active accumulator buffer back into SinkTempInfo. Pure
 * code motion from the inline loop body that used to live in
 * sink_environment_loop; per-loop physics, no engine. */
static void sink_env1_scatter_to_temp_info(const int *active_list,
                                            int num_active,
                                            const struct sink_env_gpu_out *per_active_accum)
{
    for(int a = 0; a < num_active; a++) {
        int i = active_list[a];
        int t = P[i].IndexMapToTempStruc;
        SinkTempInfo[t].Sink_SurroudingGasInternalEnergy += per_active_accum[a].Sink_SurroudingGasInternalEnergy;
        SinkTempInfo[t].Mgas_in_Kernel  += per_active_accum[a].Mgas_in_Kernel;
        SinkTempInfo[t].Mstar_in_Kernel += per_active_accum[a].Mstar_in_Kernel;
        SinkTempInfo[t].Malt_in_Kernel  += per_active_accum[a].Malt_in_Kernel;
        for(int k = 0; k < 3; k++) {
            SinkTempInfo[t].Jgas_in_Kernel[k]  += per_active_accum[a].Jgas_in_Kernel[k];
            SinkTempInfo[t].Jstar_in_Kernel[k] += per_active_accum[a].Jstar_in_Kernel[k];
            SinkTempInfo[t].Jalt_in_Kernel[k]  += per_active_accum[a].Jalt_in_Kernel[k];
        }
#ifdef SINK_REPOSITION_ON_POTMIN
        SinkTempInfo[t].DF_rms_vel += per_active_accum[a].DF_rms_vel;
        for(int k = 0; k < 3; k++) SinkTempInfo[t].DF_mean_vel[k] += per_active_accum[a].DF_mean_vel[k];
        if(per_active_accum[a].DF_mmax_particles > SinkTempInfo[t].DF_mmax_particles)
            SinkTempInfo[t].DF_mmax_particles = per_active_accum[a].DF_mmax_particles;
#endif
#if defined(SINK_OUTPUT_MOREINFO)
        SinkTempInfo[t].Sfr_in_Kernel += per_active_accum[a].Sfr_in_Kernel;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        for(int k = 0; k < 3; k++) SinkTempInfo[t].Sink_SurroundingGasVel[k] += per_active_accum[a].Sink_SurroundingGasVel[k];
#endif
#if defined(JET_DIRECTION_FROM_KERNEL_AND_SINK)
        for(int k = 0; k < 3; k++) SinkTempInfo[t].Sink_SurroundingGasCOM[k] += per_active_accum[a].Sink_SurroundingGasCOM[k];
#endif
#if (SINK_GRAVACCRETION == 8)
        SinkTempInfo[t].hubber_mdot_bondi_limiter   += per_active_accum[a].hubber_mdot_bondi_limiter;
        SinkTempInfo[t].hubber_mdot_vr_estimator    += per_active_accum[a].hubber_mdot_vr_estimator;
        SinkTempInfo[t].hubber_mdot_disk_estimator  += per_active_accum[a].hubber_mdot_disk_estimator;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
        SinkTempInfo[t].mass_to_swallow_edd += per_active_accum[a].mass_to_swallow_edd;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
        for(int k = 0; k < 3; k++) SinkTempInfo[t].angmom_prepass_sum_for_passback[k] += per_active_accum[a].angmom_prepass_sum_for_passback[k];
#endif
#if defined(SINK_RETURN_BFLUX)
        SinkTempInfo[t].kernel_norm_topass_in_swallowloop += per_active_accum[a].kernel_norm_topass_in_swallowloop;
#endif
    }
}


void sink_environment_loop(void)
{
    CPU_Step[CPU_SINK_ENV] += measure_time();

    /* ---------- engine: build active list (count + Allreduce + skip-if-empty + fill) ---------- */
    int *active_list = nullptr;
    int  num_active = 0, num_global_active = 0;
    if(!nlr_build_active_list(SinkEnv1Spec::is_active,
                               &active_list, &num_active, &num_global_active,
                               "sinkenv_active_list")) {
        CPU_Step[CPU_SINK_ENV] += measure_time();
        return;
    }

    /* ---------- physics: per-active accumulator buffer + Spec::Aux pointer ---------- */
    int alloc_n = (num_active > 0) ? num_active : 1;
    struct sink_env_gpu_out *per_active_accum = (struct sink_env_gpu_out *)
        mymalloc("sinkenv_per_active_accum", alloc_n * sizeof(struct sink_env_gpu_out));
    SinkEnv1Spec::Aux aux;
    aux.per_active_accum = per_active_accum;

    /* ---------- engine: hand off to runner ---------- */
    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    args.aux         = &aux;
    run_neighbor_loop<SinkEnv1Spec>(args);

    /* ---------- physics: scatter into SinkTempInfo ---------- */
    sink_env1_scatter_to_temp_info(active_list, num_active, per_active_accum);

    /* ---------- engine: free + post-pass normalize ---------- */
    myfree(per_active_accum);
    nlr_free_active_list(active_list);

    /* ---------- physics: per-active normalization ---------- */
    for(int i = 0; i < N_active_loc_Sink; i++) {
        sink_normalize_temp_info_struct_after_environment_loop(i);
    }

    CPU_Step[CPU_SINK_ENV] += measure_time();
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

    bool sinkenv2_imported_ghosts = gizmo_request_filtered_ghost_import_fresh(
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
