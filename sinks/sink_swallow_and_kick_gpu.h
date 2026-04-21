/* sink_swallow_and_kick_gpu.h — GPU dispatch declaration for D1.
 *
 * The host driver in sink_swallow_and_kick.cc builds a LOCAL active-sink
 * list (matching sink_isactive(i) && P[i].SwallowID==0), packs per-source
 * SinkSwallowLocalIn + per-source SinkSwallowOut buffers, and invokes
 * sink_swallow_and_kick_evaluate_gpu().  Results are scattered into
 * SinkTempInfo + P[i] on the host after kernel completion.  MPI_Reduce on
 * the per-source swallow counters reproduces today's logging behavior.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifdef SINK_PARTICLES

#include "../declarations/allvars.h"

void sink_swallow_and_kick_evaluate_gpu(struct particle_data *P_host,
                                         struct gas_cell_data *CellP_host,
                                         int num_total,
                                         int *i_active_host, int num_active,
                                         const double *i_radii_host,
                                         int j_type_bitmask);

void gizmo_gpu_sync_all_sinkswallow(struct global_data_all_processes *);

#endif /* SINK_PARTICLES */
