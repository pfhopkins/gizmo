/* sink_feed_gpu.h — GPU dispatch declarations for sink_feed (C3). */
#pragma once

void sink_feed_evaluate_gpu(struct particle_data *p,
                              struct gas_cell_data *cp,
                              int num_total);
void gizmo_gpu_sync_all_sinkfeed(struct global_data_all_processes *p);
