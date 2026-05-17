/* sinks_gpu_decls.h — consolidated GPU dispatch declarations for sinks
 * kernels: sink_feed, sink_swallow_and_kick, sink_environment.  Step 5
 * Phase E1b (2026-04-30) — merges sink_feed_gpu.h / sink_swallow_and_kick_gpu.h
 * / sink_environment_gpu.h.  All three share #ifdef SINK_PARTICLES, applied
 * once around the whole file.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef SINKS_GPU_DECLS_H
#define SINKS_GPU_DECLS_H

#ifdef SINK_PARTICLES

#include "../declarations/allvars.h"


/* ---- sink_feed (3d.1) ----
 * Ported to runner template; declarations live in sinks/sink_feed_loop.h.
 * Legacy `sink_feed_evaluate_gpu` + `gizmo_gpu_sync_all_sinkfeed` retired
 * (sinks/sink_feed_gpu.cc + sinks/sink_feed_functions.h removed from
 * build in this commit; source files deleted in follow-up cleanup). */


/* ---- sink_swallow_and_kick ----
 * Ported to runner template in 3d.3; declarations live in sinks/sink_swk_loop.h
 * (SinkSwkSpec + SinkSwallowLocalIn + SinkSwallowOut). Legacy
 * sink_swallow_and_kick_evaluate_gpu + gizmo_gpu_sync_all_sinkswallow retired
 * (sinks/sink_swallow_and_kick_gpu.cc + sinks/sink_swallow_and_kick_functions.h
 * deleted in this commit). */


/* ---- sink_environment ----
 * Stage E1 (the "first pass" sink-environment loop) runs through
 * mesh/neighbor_loop_runner.cc::run_neighbor_loop<SinkEnv1Spec>; the per-active
 * accumulator type is sink_env_gpu_out below, which is also SinkEnv1Spec::AccumData.
 * Results are scattered into SinkTempInfo on the host by the runner caller
 * (sink_environment.cc). Stage E2 (Bulge-Disk aggregator under
 * SINK_GRAVACCRETION==0) was ported in 3d.2 and now runs through
 * run_neighbor_loop<SinkEnv2Spec> (sinks/sink_env2_loop.h); only the
 * AccumData struct (sink_env_second_gpu_out) remains here for the
 * caller-side scatter manifest to share with the Spec.
 */
struct sink_env_gpu_out {
    MyFloat Sink_SurroudingGasInternalEnergy;
    MyFloat Mgas_in_Kernel;
    MyFloat Mstar_in_Kernel;
    MyFloat Malt_in_Kernel;
    MyFloat Jgas_in_Kernel[3];
    MyFloat Jstar_in_Kernel[3];
    MyFloat Jalt_in_Kernel[3];
#ifdef SINK_REPOSITION_ON_POTMIN
    MyFloat DF_rms_vel;
    MyFloat DF_mean_vel[3];
    MyFloat DF_mmax_particles;
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    MyFloat Sfr_in_Kernel;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    MyFloat Sink_SurroundingGasVel[3];
#endif
#if defined(JET_DIRECTION_FROM_KERNEL_AND_SINK)
    MyFloat Sink_SurroundingGasCOM[3];
#endif
#if (SINK_GRAVACCRETION == 8)
    MyFloat hubber_mdot_vr_estimator;
    MyFloat hubber_mdot_disk_estimator;
    MyFloat hubber_mdot_bondi_limiter;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    MyFloat mass_to_swallow_edd;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    MyFloat angmom_prepass_sum_for_passback[3];
#endif
#if defined(SINK_RETURN_BFLUX)
    MyFloat kernel_norm_topass_in_swallowloop;
#endif
};

/* Stage E2 — second environment pass: Bulge-Disk kinematic decomposition
 * (SINK_GRAVACCRETION==0). Pure aggregator, no j-writes. Ported to
 * runner template in 3d.2 (sinks/sink_env2_loop.h). Output struct
 * remains here for the caller-side scatter manifest in
 * sinks/sink_environment.cc to share with SinkEnv2Spec::AccumData. */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
struct sink_env_second_gpu_out {
    MyFloat MgasBulge_in_Kernel;
    MyFloat MstarBulge_in_Kernel;
};
#endif


#endif /* SINK_PARTICLES */
#endif /* SINKS_GPU_DECLS_H */
