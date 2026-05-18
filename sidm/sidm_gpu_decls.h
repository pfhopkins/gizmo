/* sidm_gpu_decls.h — consolidated GPU dispatch declarations for sidm
 * kernels: dm_fuzzy (DMGrad) and cbe_integrator (CBE drift+kick).  Step 5
 * Phase E1c (2026-04-30) — merges dm_fuzzy_gpu.h + cbe_integrator_gpu.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef SIDM_GPU_DECLS_H
#define SIDM_GPU_DECLS_H


/* ---- cbe_integrator (C1) ---- */
void cbe_drift_kick_evaluate_gpu(struct particle_data *P_host, int num_total,
                                  const int *active_host, int num_active,
                                  const double *dt_host);

#endif /* SIDM_GPU_DECLS_H */
