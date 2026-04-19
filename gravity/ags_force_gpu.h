/* ags_force_gpu.h — shared types + prototype for the B2 AGSForce GPU port.
 *
 * Dispatch entry called once per bitmask group from AGSForce_calc in
 * ags_rkern.cc. The driver partitions active AGSForce particles by their
 * ags_gravity_kernel_shared_BITFLAG(Type) and calls this once per group.
 *
 * Writes i-side accumulators into out_host[aa] for the caller to scatter
 * back into P[i]. j-side modifications (P[j].Vel, P[j].dp, P[j].NInteractions,
 * P[j].wakeup) are applied atomically inside the kernel; the caller wraps the
 * dispatch with ghost_writeback_{zero_,}agsforce so ghost-side deltas are
 * reverse-communicated to their home ranks afterwards.
 */
#ifndef AGS_FORCE_GPU_H
#define AGS_FORCE_GPU_H

#include "../declarations/allvars.h"

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

struct ags_force_gpu_out {
#if defined(DM_SIDM)
    double sidm_kick[3];
    double dtime_sidm;
    int si_count;
#endif
#ifdef DM_FUZZY
    double acc[3];
    double AGS_Dt_Numerical_QuantumPotential;
#if (DM_FUZZY > 0)
    double AGS_Dt_Psi_Re, AGS_Dt_Psi_Im, AGS_Dt_Psi_Mass;
#endif
#endif
#if defined(CBE_INTEGRATOR)
    double AGS_vsig; /* MAX reduction (not sum) */
    double CBE_basis_moments_dt[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
#endif
};


/* Dispatch entry: one group, one GPU kernel.
 *  - P_host:         full particle array including ghosts (see ghost_exchange)
 *  - num_total:      NumPart including ghosts
 *  - i_active_host:  indices of searcher particles (all share j_type_bitmask)
 *  - i_radii_host:   per-i AGS_KernelRadius (NOT inflated; GPU applies the
 *                    SIDM 3x search-radius factor internally)
 *  - j_type_bitmask: allowed neighbor types (from ags_gravity_kernel_shared_BITFLAG)
 *  - out_host:       num_active-long output buffer (zeroed on return)
 */
void ags_force_evaluate_gpu(struct particle_data *P_host,
                            int num_total,
                            int *i_active_host, int num_active,
                            const double *i_radii_host,
                            int j_type_bitmask,
                            struct ags_force_gpu_out *out_host);

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */
#endif /* AGS_FORCE_GPU_H */
