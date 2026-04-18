/* dm_dispersion_gpu.h — shared types + prototype for GPU DM dispersion port.
 *
 * Exposes the per-active output struct and the dispatch entry point shared
 * between dm_dispersion_rkern.cc (host driver) and dm_dispersion_gpu.cc
 * (GPU kernel TU).
 */
#ifndef DM_DISPERSION_GPU_H
#define DM_DISPERSION_GPU_H

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

#endif /* DM_DISPERSION_GPU_H */
