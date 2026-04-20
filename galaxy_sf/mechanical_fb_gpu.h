/* mechanical_fb_gpu.h — GPU dispatch declarations for mechanical_fb (B8).
 *
 * Runs all 6 modes (-2, -1, 0, 1, 2, 3) of the default-scheme
 * addFB_evaluate on the GPU. Host caller owns LocalGasMechFBInfoTemp
 * and calls verify_and_assign_local_mechfb_integrals afterward.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#include "mechanical_fb_types.h"  /* full struct MechFBGasDelta */

/* Dispatches all 6 modes on the GPU. Caller provides:
 *   P_host, CellP_host, num_total : host P[]/CellP[] and their size (num_total = N_local + N_ghost)
 *   gas_delta_host                : per-gas delta buffer, sized N_gas, zeroed by caller
 *   i_active_host, num_active     : local active-star superset list (any star active in any of the 6 modes)
 *   src_radii_host                : per-source kernel radii parallel to i_active_host
 * Returns via:
 *   gas_delta_host                : populated with coupling deltas (home cells only after ghost writeback)
 *   P_host[i].Mass / Sink_Mass    : decremented for each source by total M_coupled across modes
 *   P_host[i].Area_weighted_sum[] : updated from modes -2 and -1 (mirrors CPU out2particle_addFB path)
 *   *n_couplings_out              : set to number of gas cells with N_injected > 0 (== N_Gas_Couplings_ThisTask)
 */
void mechanical_fb_evaluate_gpu(struct particle_data *P_host,
                                 struct gas_cell_data *CellP_host,
                                 int num_total,
                                 struct MechFBGasDelta *gas_delta_host,
                                 int n_gas,
                                 int *i_active_host, int num_active,
                                 const double *src_radii_host,
                                 int *n_couplings_out);

void gizmo_gpu_sync_all_mechfb(struct global_data_all_processes *p);
