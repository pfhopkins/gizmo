/* galsf_gpu_decls.h — consolidated GPU dispatch declarations for galaxy_sf
 * kernels: thermal_fb, mechanical_fb, radfb_local (radiation_pressure_winds),
 * dm_dispersion.  Step 5 Phase E1a (2026-04-30) — merges the four single-line
 * radfb_local_gpu.h / thermal_fb_gpu.h / mechanical_fb_gpu.h / dm_dispersion_gpu.h
 * headers to reduce file count.
 *
 * Each section preserves its original feature gate.  Includers should pull
 * this single header instead of the per-kernel headers.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GALSF_GPU_DECLS_H
#define GALSF_GPU_DECLS_H

#include "mechanical_fb_types.h"  /* full struct MechFBGasDelta */

/* ---- thermal_fb (B6) ---- */
void thermal_fb_evaluate_gpu(struct particle_data *p,
                              struct gas_cell_data *cp,
                              int num_total,
                              int *i_active_host, int num_active,
                              const double *src_radii_host);
void gizmo_gpu_sync_all_thermalfb(struct global_data_all_processes *p);


/* ---- mechanical_fb (B8) ----
 * Runs all 6 modes (-2, -1, 0, 1, 2, 3) of the default-scheme addFB_evaluate
 * on the GPU.  Host caller owns LocalGasMechFBInfoTemp and calls
 * verify_and_assign_local_mechfb_integrals afterward.
 *
 * Caller provides:
 *   P_host, CellP_host, num_total : host P[]/CellP[] and their size (num_total = N_local + N_ghost)
 *   gas_delta_host                : per-gas delta buffer, sized N_gas, zeroed by caller
 *   i_active_host, num_active     : local active-star superset list (any star active in any of the 6 modes)
 *   src_radii_host                : per-source kernel radii parallel to i_active_host
 * Returns via:
 *   gas_delta_host                : populated with coupling deltas (home cells only after ghost writeback)
 *   P_host[i].Mass / Sink_Mass    : decremented for each source by total M_coupled across modes
 *   P_host[i].Area_weighted_sum[] : updated from modes -2 and -1 (mirrors CPU out2particle_addFB path)
 *   *n_couplings_out              : set to number of gas cells with N_injected > 0
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


/* ---- radfb_local: radiation_pressure_winds ---- */
#ifdef GALSF_FB_FIRE_RT_LOCALRP
void radiation_pressure_winds_gpu(struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   int num_total,
                                   const int *i_active_host, int num_active);
#endif


/* ---- dm_dispersion (GALSF_SUBGRID_WINDS, scaling==2) ---- */
#ifdef GALSF_SUBGRID_WINDS
#if (GALSF_SUBGRID_WIND_SCALING==2)

/* Per-active-gas output from the DM-dispersion neighbor accumulation. */
struct dispdens_gpu_out {
    double Ngb;
    double DM_Vx;
    double DM_Vy;
    double DM_Vz;
    double DM_Vel_Disp;
};

/* GPU dispatch: build cross-type CSR (gas i → DM j, radius = radii_host[aa]),
   run accumulation kernel, return per-active outputs in out_host[aa]. */
void disp_density_evaluate_gpu(struct particle_data *P_host, int num_total,
                               int *i_active_host, int num_active,
                               const double *i_radii_host,
                               void *out_host_void);

#endif /* GALSF_SUBGRID_WIND_SCALING==2 */
#endif /* GALSF_SUBGRID_WINDS */

#endif /* GALSF_GPU_DECLS_H */
