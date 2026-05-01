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


/* ---- sink_feed (C3) ---- */
void sink_feed_evaluate_gpu(struct particle_data *p,
                              struct gas_cell_data *cp,
                              int num_total,
                              int *i_active_host, int num_active,
                              const double *src_radii_host);
void gizmo_gpu_sync_all_sinkfeed(struct global_data_all_processes *p);


/* ---- sink_swallow_and_kick (D1) ----
 * The host driver in sink_swallow_and_kick.cc builds a LOCAL active-sink list
 * (matching sink_isactive(i) && P[i].SwallowID==0), packs per-source
 * SinkSwallowLocalIn + per-source SinkSwallowOut buffers, and invokes
 * sink_swallow_and_kick_evaluate_gpu().  Results are scattered into
 * SinkTempInfo + P[i] on the host after kernel completion.  MPI_Reduce on the
 * per-source swallow counters reproduces today's logging behavior.
 */
void sink_swallow_and_kick_evaluate_gpu(struct particle_data *P_host,
                                         struct gas_cell_data *CellP_host,
                                         int num_total,
                                         int *i_active_host, int num_active,
                                         const double *i_radii_host,
                                         int j_type_bitmask);
void gizmo_gpu_sync_all_sinkswallow(struct global_data_all_processes *);


/* ---- sink_environment ----
 * The host driver in sink_environment.cc calls sink_environment_evaluate_gpu()
 * once per active-sink set; results are scattered into SinkTempInfo on the host.
 * Under SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS the kernel also does
 * atomic_min on P[j].SwallowTime; ghost_writeback_{zero_,}swallowtime() must
 * bracket the call to propagate those writes back to home ranks.
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

void sink_environment_evaluate_gpu(struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   int num_total,
                                   int *i_active_host, int num_active,
                                   const double *i_radii_host,
                                   int j_type_bitmask,
                                   struct sink_env_gpu_out *out_host);

/* Stage E2 — second environment pass: Bulge-Disk kinematic decomposition
 * (SINK_GRAVACCRETION==0). Pure aggregator, no j-writes. Reads per-sink
 * Jgas/Jstar from SinkTempInfo (populated by the first pass) and returns
 * MgasBulge / MstarBulge to be scattered back into SinkTempInfo on host. */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
struct sink_env_second_gpu_out {
    MyFloat MgasBulge_in_Kernel;
    MyFloat MstarBulge_in_Kernel;
};

void sink_environment_second_evaluate_gpu(struct particle_data *P_host,
                                           struct gas_cell_data *CellP_host,
                                           int num_total,
                                           int *i_active_host, int num_active,
                                           const double *i_radii_host,
                                           const MyFloat (*i_Jgas_host)[3],
                                           const MyFloat (*i_Jstar_host)[3],
                                           int j_type_bitmask,
                                           struct sink_env_second_gpu_out *out_host);
#endif

void gizmo_gpu_sync_all_sinkenv(struct global_data_all_processes *);

#endif /* SINK_PARTICLES */
#endif /* SINKS_GPU_DECLS_H */
