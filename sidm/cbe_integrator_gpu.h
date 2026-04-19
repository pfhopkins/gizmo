/* cbe_integrator_gpu.h — entry-point declaration for the GPU EP port of
 * do_cbe_drift_kick (C1).  The host driver in kicks.cc calls
 * cbe_drift_kick_evaluate_gpu() after collecting active CBE particles.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef CBE_INTEGRATOR_GPU_H
#define CBE_INTEGRATOR_GPU_H

void cbe_drift_kick_evaluate_gpu(struct particle_data *P_host, int num_total,
                                  const int *active_host, int num_active,
                                  const double *dt_host);
void gizmo_gpu_sync_all_cbeintegrator(struct global_data_all_processes *p);

#endif /* CBE_INTEGRATOR_GPU_H */
