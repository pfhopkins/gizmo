/* ags_gpu_decls.h — GPU dispatch declarations for the AGS
 * adaptive-gravitational-softening kernels.
 *
 * As of 3d.4 (2026-05-11), ags_density is runner-driven via AgsDensitySpec
 * in gravity/ags_density_loop.{h,cc}; only ags_force_evaluate_gpu remains
 * here. The legacy ags_density_evaluate_gpu / its bm-group host driver in
 * ags_rkern.cc were retired by the 3d.4 cleanup commit.
 *
 * Entry point is gated by AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef AGS_GPU_DECLS_H
#define AGS_GPU_DECLS_H

#include "../declarations/allvars.h"

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

/* ---- ags_force (B2) ---- */
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
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & 7)
    /* Phase 17b pairwise outcomes (bits 0|1|2 = COAG|FRAG|SHAT). Accumulators
     * for the local (absorber/eroded) side of each pair; scattered to P[ii]
     * by the host-side post-pass in ags_rkern.cc.
     *   Grain_DeltaCoagMass:                additive mass gained from COAG
     *   Grain_DeltaCoag_CompositionMass[s]: per-species mass gained (for
     *                                       composition mixing on absorber)
     *   Grain_DeltaErosionFrac:             multiplicative size shrink factor
     *                                       from FRAG/SHAT events (1 - sum of
     *                                       per-event fractional losses;
     *                                       applied as a^new = a^old * factor) */
    double Grain_DeltaCoagMass;
    double Grain_DeltaCoag_CompositionMass[GRAIN_NUM_SPECIES];
    double Grain_DeltaErosionFrac;
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
#endif /* AGS_GPU_DECLS_H */
