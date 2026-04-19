/* thermal_fb_gpu.h — GPU dispatch declarations for thermal_fb (B6).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

void thermal_fb_evaluate_gpu(struct particle_data *p,
                              struct gas_cell_data *cp,
                              int num_total);
void gizmo_gpu_sync_all_thermalfb(struct global_data_all_processes *p);
