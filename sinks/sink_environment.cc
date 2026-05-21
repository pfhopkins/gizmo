/*! \file sink_environment.c
*  \brief routines for evaluating sink particle environment
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* Kokkos_Core.hpp MUST precede allvars.h: macros.h #defines `terminate(...)`
 * which would mangle std::terminate inside Kokkos's transitive <exception>
 * include. Required because sink_env1_loop.h emits Kokkos::atomic_min
 * inline under SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS. */
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "sinks_gpu_decls.h"
#include "../mesh/neighbor_loop_runner.h"
#include "sink_env1_loop.h"
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
#include "sink_env2_loop.h"
#endif
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


/* Scatter the per-active accumulator buffer back into SinkTempInfo.
 * Per-loop physics; the runner doesn't see this. Operations match the
 * pair_kernel per-field write ops (the oracle protects merge_accum
 * specifically, but this scatter manifest must agree on op semantics
 * field-for-field — drift is silent because it lives downstream of the
 * runner's oracle gate).
 *
 * Adding a new scattered field for this loop = ONE LINE under the
 * appropriate physics flag's #ifdef. */
static void sink_env1_scatter_to_temp_info(const int *active_list,
                                            int num_active,
                                            const struct sink_env_gpu_out *per_active_accum)
{
    for(int a = 0; a < num_active; a++) {
        int i = active_list[a];
        int t = P[i].IndexMapToTempStruc;

        /* Local op macros — scoped to this loop body via #undef below.
         * Semantically identical expansion to the prior hand-written
         * per-field assignment statements. */
#define SCATTER_ADD(field)       SinkTempInfo[t].field += per_active_accum[a].field;
#define SCATTER_ADD_VEC3(field)  for(int k = 0; k < 3; k++) SinkTempInfo[t].field[k] += per_active_accum[a].field[k];
#define SCATTER_MAX(field)       if(per_active_accum[a].field > SinkTempInfo[t].field) SinkTempInfo[t].field = per_active_accum[a].field;

        SCATTER_ADD(Sink_SurroudingGasInternalEnergy)
        SCATTER_ADD(Mgas_in_Kernel)
        SCATTER_ADD(Mstar_in_Kernel)
        SCATTER_ADD(Malt_in_Kernel)
        SCATTER_ADD_VEC3(Jgas_in_Kernel)
        SCATTER_ADD_VEC3(Jstar_in_Kernel)
        SCATTER_ADD_VEC3(Jalt_in_Kernel)
#ifdef SINK_REPOSITION_ON_POTMIN
        SCATTER_ADD(DF_rms_vel)
        SCATTER_ADD_VEC3(DF_mean_vel)
        SCATTER_MAX(DF_mmax_particles)
#endif
#if defined(SINK_OUTPUT_MOREINFO)
        SCATTER_ADD(Sfr_in_Kernel)
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        SCATTER_ADD_VEC3(Sink_SurroundingGasVel)
#endif
#if defined(JET_DIRECTION_FROM_KERNEL_AND_SINK)
        SCATTER_ADD_VEC3(Sink_SurroundingGasCOM)
#endif
#if (SINK_GRAVACCRETION == 8)
        SCATTER_ADD(hubber_mdot_bondi_limiter)
        SCATTER_ADD(hubber_mdot_vr_estimator)
        SCATTER_ADD(hubber_mdot_disk_estimator)
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
        SCATTER_ADD(mass_to_swallow_edd)
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
        SCATTER_ADD_VEC3(angmom_prepass_sum_for_passback)
#endif
#if defined(SINK_RETURN_BFLUX)
        SCATTER_ADD(kernel_norm_topass_in_swallowloop)
#endif

#undef SCATTER_ADD
#undef SCATTER_ADD_VEC3
#undef SCATTER_MAX
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


/* Caller-side scatter for sink_env2. Two additive fields into SinkTempInfo;
 * adding a new scattered field for this loop = ONE LINE under its
 * physics flag's #ifdef. */
static void sink_env2_scatter_to_temp_info(const int *active_list,
                                            int num_active,
                                            const struct sink_env_second_gpu_out *per_active_accum)
{
    for(int a = 0; a < num_active; a++) {
        int i = active_list[a];
        int t = P[i].IndexMapToTempStruc;

#define SCATTER_ADD(dst, src_field)  dst += per_active_accum[a].src_field;

        SCATTER_ADD(SinkTempInfo[t].MgasBulge_in_Kernel,  MgasBulge_in_Kernel)
        SCATTER_ADD(SinkTempInfo[t].MstarBulge_in_Kernel, MstarBulge_in_Kernel)

#undef SCATTER_ADD
        (void)t;
    }
}


void sink_environment_second_loop(void)
{
    /* Stage E2: pure aggregator (Bulge-Disk kinematic decomposition).
     * Phase 4 3d.2 — runner-driven via SinkEnv2Spec. Same active set +
     * radii + j_type_bitmask as Stage E1 (sink_environment_loop). */
    CPU_Step[CPU_SINK_ENV] += measure_time();

    /* ---------- engine: build active list ---------- */
    int *active_list = nullptr;
    int  num_active = 0, num_global_active = 0;
    if(!nlr_build_active_list(SinkEnv2Spec::is_active,
                               &active_list, &num_active, &num_global_active,
                               "sinkenv2_active_list")) {
        CPU_Step[CPU_SINK_ENV] += measure_time();
        return;
    }

    /* ---------- physics: per-active accumulator + per-active host inputs ---------- */
    int alloc_n = (num_active > 0) ? num_active : 1;
    struct sink_env_second_gpu_out *per_active_accum = (struct sink_env_second_gpu_out *)
        mymalloc("sinkenv2_per_active_accum", alloc_n * sizeof(struct sink_env_second_gpu_out));
    SinkEnv2PerActiveIn *host_inputs = (SinkEnv2PerActiveIn *)
        mymalloc("sinkenv2_host_inputs", alloc_n * sizeof(SinkEnv2PerActiveIn));

    /* Fill per-active Jgas / Jstar from SinkTempInfo (populated by
     * Stage E1's first pass). populate_device_context will copy this
     * into the UVM array attached to the runner's DeviceContext. */
    for(int a = 0; a < num_active; a++) {
        int i = active_list[a];
        int t = P[i].IndexMapToTempStruc;
        for(int k = 0; k < 3; k++) {
            host_inputs[a].Jgas[k]  = SinkTempInfo[t].Jgas_in_Kernel[k];
            host_inputs[a].Jstar[k] = SinkTempInfo[t].Jstar_in_Kernel[k];
        }
    }

    SinkEnv2Spec::Aux aux;
    aux.per_active_accum = per_active_accum;
    aux.host_inputs      = host_inputs;

    /* ---------- engine: hand off to runner ---------- */
    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    args.aux         = &aux;
    run_neighbor_loop<SinkEnv2Spec>(args);

    /* ---------- physics: caller scatter into SinkTempInfo ---------- */
    myfree(host_inputs);
    sink_env2_scatter_to_temp_info(active_list, num_active, per_active_accum);

    /* ---------- engine: free + return ---------- */
    myfree(per_active_accum);
    nlr_free_active_list(active_list);

    CPU_Step[CPU_SINK_ENV] += measure_time();
}

#endif   //SINK_GRAVACCRETION == 0




#endif // top-level flag
