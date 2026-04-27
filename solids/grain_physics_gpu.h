/* grain_physics_gpu.h — GPU dispatch declarations for B7.
 *
 * Two independent loops:
 *   grain_backrx_evaluate_gpu                      (GRAIN_BACKREACTION)
 *   interpolate_fluxes_opacities_gasgrains_evaluate_gpu
 *                                                  (RT_OPACITY_FROM_EXPLICIT_GRAINS)
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

void grain_backrx_evaluate_gpu(struct particle_data *P_host,
                                struct gas_cell_data *CellP_host,
                                int num_total,
                                int *i_active_host, int num_active,
                                const double *src_radii_host);

/* Runs BOTH directions internally (gas→grains and grains→gas). */
void interpolate_fluxes_opacities_gasgrains_evaluate_gpu(struct particle_data *P_host,
                                                          struct gas_cell_data *CellP_host,
                                                          int num_total,
                                                          int *i_active_gas_host, int num_active_gas,
                                                          const double *src_radii_gas_host,
                                                          int *i_active_grain_host, int num_active_grain,
                                                          const double *src_radii_grain_host);

void gizmo_gpu_sync_all_grainphysics(struct global_data_all_processes *p);
