/* cbe_integrator_gpu.cc — GPU-accelerated CBE drift-kick EP kernel (C1).
 *
 * Ports do_cbe_drift_kick (sidm/cbe_integrator.cc) to a Kokkos parallel_for
 * kernel.  This is a pure per-particle EP operation — no neighbor list needed.
 * Active particles are collected by the host driver in kicks.cc and passed as
 * an index array with per-particle dt values.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../declarations/macros.h"

#if defined(CBE_INTEGRATOR) && defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)

#include "cbe_integrator_functions.h"

void cbe_drift_kick_evaluate_gpu(struct particle_data *P_host, int num_total,
                                  const int *active_host, int num_active,
                                  const double *dt_host)
{
    if(num_active <= 0) return;

    /* Step 13 Phase 1 arena. P-only kernel; pass NULL for CellP. */
    gpu_particles_arena_acquire(num_total, P_host, NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();

    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        num_active * sizeof(int));
    double *d_dt = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        num_active * sizeof(double));
    memcpy(d_active, active_host, num_active * sizeof(int));
    memcpy(d_dt,     dt_host,     num_active * sizeof(double));

    Kokkos::parallel_for("cbe_drift_kick", num_active, KOKKOS_LAMBDA(const int a) {
        int i = d_active[a];
        double dt = d_dt[a];
        do_cbe_drift_kick_kernel(P_gpu[i], dt);
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("cbe_drift_kick", num_active);

    memcpy(P_host, P_gpu, num_total * sizeof(struct particle_data));

    /* Full P scatter syncs arena→host (P-side); CellP unmodified by this kernel. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_dt);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);

    printf("  GPU cbe_drift_kick: %d active\n", num_active);
}

GPU_ALL_SYNC_FUNC(cbeintegrator)

#else

void cbe_drift_kick_evaluate_gpu(struct particle_data *p, int num_total,
                                  const int *active, int num_active,
                                  const double *dt_arr)
{
    (void)p; (void)num_total; (void)active; (void)num_active; (void)dt_arr;
}
void gizmo_gpu_sync_all_cbeintegrator(struct global_data_all_processes *p) { (void)p; }

#endif /* CBE_INTEGRATOR && OPENMP_GPU_OFFLOAD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */
