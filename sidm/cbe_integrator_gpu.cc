/* cbe_integrator_gpu.cc — GPU/OMP per-particle EP dispatch for the CBE
 * integrator: drift-kick (called from core/kicks.cc each half-step) and
 * post-gravity finalization (called from gravity/ags_force_loop.cc after the
 * AGSForce iterative neighbor loop closes).
 *
 * Both entries follow the same shape as solids/grain_drag_gpu.cc:
 *   - tiny-N (num_active < GPU_MIN_PARTICLES_FOR_OFFLOAD): direct host call
 *     in an OMP parallel-for loop. No arena, no Kokkos allocation, no
 *     compact gather. Each active index is unique → thread-safe.
 *   - large-N: compact gather of particle_data[num_active] in
 *     Kokkos shared-space, single kernel launch, narrow scatter of only
 *     the fields the kernel writes, then kokkos_free.
 *
 * The narrow scatter is load-bearing: it prevents the kernel from
 * inadvertently overwriting unrelated P[i] fields touched by other code
 * between the gather and the scatter.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"   /* MUST precede allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../declarations/macros.h"
#include "../declarations/constants.h"
#include "../system/gpu_particles_arena.h"
#include "sidm_gpu_decls.h"

#if defined(CBE_INTEGRATOR)

#include "cbe_integrator_functions.h"

/* ----------------------------------------------------------------------------
 * cbe_drift_kick_evaluate_gpu — first/second half-step CBE drift-kick.
 *
 * Kernel: do_cbe_drift_kick_kernel(pi, dt).
 * Writes: pi.CBE_basis_moments[NBASIS][NMOMENTS] only.
 * Reads: All.Time, All.TimeBegin, All.Ti_Current (via the kernel's basis-
 *        resplit branch); All-mirror handles the device read.
 * --------------------------------------------------------------------------*/
void cbe_drift_kick_evaluate_gpu(struct particle_data *P_host,
                                 const int *active_indices, int num_active,
                                 const double *dt_host)
{
    if(num_active <= 0) return;

    if(num_active < GPU_MIN_PARTICLES_FOR_OFFLOAD)
    {
        /* Tiny-N OMP path: pure host calls, no arena, no Kokkos. Each
         * active index writes its own particle (unique by construction). */
        PRINT_STATUS("  CBE drift-kick (OMP): %d active", num_active);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int a = 0; a < num_active; a++)
            do_cbe_drift_kick_kernel(P_host[active_indices[a]], dt_host[a]);
        return;
    }

    /* Large-N GPU path: compact gather → kernel → narrow scatter. */
    GIZMO_GPU_ENSURE_ALL_FRESH();

    struct particle_data *compact_P = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(struct particle_data));
    int    *d_active = (int *)    Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    double *d_dt     = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(double));
    for(int a = 0; a < num_active; a++) compact_P[a] = P_host[active_indices[a]];
    memcpy(d_active, active_indices, num_active * sizeof(int));
    memcpy(d_dt,     dt_host,        num_active * sizeof(double));

    PRINT_STATUS("  CBE drift-kick (GPU): %d active", num_active);
    {
        struct particle_data *kp = compact_P;
        double *kdt = d_dt;
        gizmo_gpu_kernel_launch("cbe_drift_kick", num_active, KOKKOS_LAMBDA(int a) {
            do_cbe_drift_kick_kernel(kp[a], kdt[a]);
        });
    }

    /* Narrow scatter: only CBE_basis_moments is touched by the kernel. */
    for(int a = 0; a < num_active; a++) {
        int ii = active_indices[a];
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                P_host[ii].CBE_basis_moments[m][k] = compact_P[a].CBE_basis_moments[m][k];
    }

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_dt);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);
}


/* ----------------------------------------------------------------------------
 * cbe_postgravity_evaluate_gpu — per-active post-AGSForce finalization.
 *
 * Kernel: do_cbe_postgravity_kernel(pi).
 * Writes: pi.GravAccel, pi.CBE_basis_moments_dt[NBASIS][NMOMENTS].
 * Reads: All.cf_a2inv (via All-mirror on device).
 *
 * Called with active_indices = ActiveParticleList.data(),
 * num_active = ActiveParticleList.size() (no extra gating — matches the
 * legacy CPU loop in gravity/ags_force_loop.cc).
 * --------------------------------------------------------------------------*/
void cbe_postgravity_evaluate_gpu(struct particle_data *P_host,
                                  const int *active_indices, int num_active)
{
    if(num_active <= 0) return;

    if(num_active < GPU_MIN_PARTICLES_FOR_OFFLOAD)
    {
        PRINT_STATUS("  CBE postgravity (OMP): %d active", num_active);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int a = 0; a < num_active; a++)
            do_cbe_postgravity_kernel(P_host[active_indices[a]]);
        return;
    }

    GIZMO_GPU_ENSURE_ALL_FRESH();

    struct particle_data *compact_P = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(struct particle_data));
    for(int a = 0; a < num_active; a++) compact_P[a] = P_host[active_indices[a]];

    PRINT_STATUS("  CBE postgravity (GPU): %d active", num_active);
    {
        struct particle_data *kp = compact_P;
        gizmo_gpu_kernel_launch("cbe_postgravity", num_active, KOKKOS_LAMBDA(int a) {
            do_cbe_postgravity_kernel(kp[a]);
        });
    }

    /* Narrow scatter: GravAccel and CBE_basis_moments_dt only. */
    for(int a = 0; a < num_active; a++) {
        int ii = active_indices[a];
        P_host[ii].GravAccel = compact_P[a].GravAccel;
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                P_host[ii].CBE_basis_moments_dt[m][k] = compact_P[a].CBE_basis_moments_dt[m][k];
    }

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);
}


#else /* stubs for builds without CBE_INTEGRATOR */

void cbe_drift_kick_evaluate_gpu(struct particle_data *, const int *, int, const double *) {}
void cbe_postgravity_evaluate_gpu(struct particle_data *, const int *, int) {}

#endif /* CBE_INTEGRATOR */
