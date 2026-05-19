/* sidm_gpu_decls.h — GPU dispatch declarations for SIDM/CBE helpers.
 *
 * CBE drift/kick remains a direct adaptive CPU/GPU dispatch routine. DM_FUZZY
 * DMGrad moved to the runner-template path in dm_fuzzy_loop.{h,cc}; no DM_FUZZY
 * declarations live here.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef SIDM_GPU_DECLS_H
#define SIDM_GPU_DECLS_H


/* ---- cbe_integrator (C1) ---- */
void cbe_drift_kick_evaluate_gpu(struct particle_data *P_host,
                                 const int *active_indices, int num_active,
                                 const double *dt_host);
void cbe_postgravity_evaluate_gpu(struct particle_data *P_host,
                                  const int *active_indices, int num_active);

#endif /* SIDM_GPU_DECLS_H */
