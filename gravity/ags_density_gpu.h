/* ags_density_gpu.h — shared types + prototype for GPU AGS-density port.
 *
 * Exposes the per-active output struct and the dispatch entry point shared
 * between ags_rkern.cc (host driver) and ags_density_gpu.cc (GPU kernel TU).
 * The host driver partitions active AGS particles by their shared
 * neighbor-type bitmask (returned by ags_gravity_kernel_shared_BITFLAG) and
 * calls the GPU dispatch once per group — see project_tier_b_infra_scope.md
 * for the design rationale.
 */
#ifndef AGS_DENSITY_GPU_H
#define AGS_DENSITY_GPU_H

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

struct ags_density_gpu_out {
    double Ngb;
    double DrkernNgb;
    double AGS_zeta;
    double AGS_vsig;          /* MAX reduction (not sum) */
    double Particle_DivVel;
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
    double NV_T[3][3];
#endif
};

/* Dispatch entry: builds a cross-type CSR for one bitmask group, runs the
   AGS-density accumulation kernel, returns per-active outputs.
   - i_active_host: indices of searcher particles (all share j_type_bitmask)
   - i_radii_host:  per-searcher AGS_KernelRadius values
   - j_type_bitmask: shared neighbor-type bitmask for this group
   Atomically writes P[j].wakeup = -1 for any neighbor that fires the wakeup
   condition; writes NeedToWakeupParticles_local = 1 if any wakeup occurred. */
void ags_density_evaluate_gpu(struct particle_data *P_host,
                              struct gas_cell_data *CellP_host,
                              int num_total,
                              int *i_active_host, int num_active,
                              const double *i_radii_host,
                              int j_type_bitmask,
                              void *out_host_void);

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */
#endif /* AGS_DENSITY_GPU_H */
