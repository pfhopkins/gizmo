/* cbe_integrator_gpu.cc — GPU/OMP per-particle EP dispatch for the CBE
 * integrator: drift-kick (called from core/kicks.cc each half-step) and
 * post-gravity finalization (called from gravity/ags_force_loop.cc after the
 * AGSForce iterative neighbor loop closes).
 *
 * Both entries follow the same shape as solids/grain_drag.cc:
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
 * Kernel: do_cbe_drift_kick_kernel(pi, dt, dT_out).
 * Writes: pi.CBE_basis_moments[NBASIS][NMOMENTS] AND pi.Vel[0..2]. The
 *        absolute-update round-trip (codex 2026-06-04) derives V_new from
 *        the post-update mass-weighted-mean basis velocity and writes it
 *        back to pi.Vel directly. The GPU narrow scatter therefore must
 *        copy BOTH fields back from compact_P.
 * Reads: All.Time, All.TimeBegin, All.Ti_Current (via the kernel's basis-
 *        resplit branch); All-mirror handles the device read.
 *
 * Wave-CBE Commit 5 (2026-05-26): the kernel now returns a per-particle
 * SPD-repair trace increment (sum over basis of trace_after - trace_before)
 * via *dT_out. Compact per-active scratch dT_scratch[a] is host-summed and
 * forwarded to cbe_step_diagnostics_observe_repair (col-7/8). The whole
 * accumulator path is guarded on OUTPUT_ADDITIONAL_RUNINFO ||
 * CBE_INTEGRATOR_OUTPUT_MOREINFO so production builds pay zero overhead
 * (kernel gets nullptr; repair still happens, only the diagnostic
 * bookkeeping disappears). dP is identically 0 (SPD touches only stress).
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
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
        double *dT_scratch = (double *) malloc(num_active * sizeof(double));
        if(!dT_scratch) {
            fprintf(stderr, "[task %d] cbe_drift_kick_evaluate_gpu: malloc(%zu) failed for dT_scratch (num_active=%d)\n",
                    ThisTask, (size_t)(num_active * sizeof(double)), num_active);
            endrun(91501);
        }
#else
        double *dT_scratch = nullptr;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int a = 0; a < num_active; a++) {
            double dT_local = 0.0;
            do_cbe_drift_kick_kernel(P_host[active_indices[a]], dt_host[a],
                                     dT_scratch ? &dT_local : (double*)nullptr);
            if(dT_scratch) dT_scratch[a] = dT_local;
        }
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
        {
            double dT_sum = 0.0;
            for(int a = 0; a < num_active; a++) dT_sum += dT_scratch[a];
            cbe_step_diagnostics_observe_repair(/* dP */ 0.0, dT_sum);
            free(dT_scratch);
        }
#endif
        return;
    }

    /* Large-N GPU path: compact gather → kernel → narrow scatter. */
    GIZMO_GPU_ENSURE_ALL_FRESH();

    struct particle_data *compact_P = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(struct particle_data));
    int    *d_active = (int *)    Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    double *d_dt     = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(double));
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
    double *dT_scratch = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(double));
#else
    double *dT_scratch = nullptr;
#endif
    for(int a = 0; a < num_active; a++) compact_P[a] = P_host[active_indices[a]];
    memcpy(d_active, active_indices, num_active * sizeof(int));
    memcpy(d_dt,     dt_host,        num_active * sizeof(double));

    PRINT_STATUS("  CBE drift-kick (GPU): %d active", num_active);
    {
        struct particle_data *kp = compact_P;
        double *kdt = d_dt;
        double *kdT = dT_scratch;
        gizmo_gpu_kernel_launch("cbe_drift_kick", num_active, KOKKOS_LAMBDA(int a) {
            double dT_local = 0.0;
            do_cbe_drift_kick_kernel(kp[a], kdt[a],
                                     kdT ? &dT_local : (double*)nullptr);
            if(kdT) kdT[a] = dT_local;
        });
    }
    /* gizmo_gpu_kernel_launch fences before returning (the narrow scatter
     * below already relies on this); host reads of dT_scratch[a] follow. */

    /* Narrow scatter: kernel writes CBE_basis_moments AND pi.Vel (the
     * latter from the absolute-update round-trip's V_new = MMV derivation,
     * codex 2026-06-04). Both must be scattered back to P_host. */
    for(int a = 0; a < num_active; a++) {
        int ii = active_indices[a];
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                P_host[ii].CBE_basis_moments[m][k] = compact_P[a].CBE_basis_moments[m][k];
        for(int k = 0; k < 3; k++) P_host[ii].Vel[k] = compact_P[a].Vel[k];
    }

#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
    {
        double dT_sum = 0.0;
        for(int a = 0; a < num_active; a++) dT_sum += dT_scratch[a];
        cbe_step_diagnostics_observe_repair(/* dP */ 0.0, dT_sum);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(dT_scratch);
    }
#endif
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_dt);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);
}


/* ----------------------------------------------------------------------------
 * cbe_postgravity_evaluate_gpu — per-active post-AGSForce finalization.
 *
 * Kernel: do_cbe_postgravity_kernel(pi).
 * Writes: pi.CBE_basis_moments_dt[basis][0] only (mass-closure safety net;
 *         codex 2026-06-04). The CBE bulk-velocity injection into
 *         pi.GravAccel is GONE in the post-2026-06-04 design — bulk V is
 *         now derived from the post-update mass-weighted-mean basis
 *         velocity inside the drift-kick absolute round-trip.
 * Reads: pi.Mass, pi.CBE_basis_moments_dt[basis][0], pi.CBE_basis_moments[basis][0].
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

    /* Narrow scatter: kernel writes only the mass slot of the dt-accumulator
     * (closure-enforcing subtraction). Scatter that slot back; pi.GravAccel
     * is NOT touched anymore in the post-2026-06-04 design. */
    for(int a = 0; a < num_active; a++) {
        int ii = active_indices[a];
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            P_host[ii].CBE_basis_moments_dt[m][0] = compact_P[a].CBE_basis_moments_dt[m][0];
    }

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);
}


#else /* stubs for builds without CBE_INTEGRATOR */

void cbe_drift_kick_evaluate_gpu(struct particle_data *, const int *, int, const double *) {}
void cbe_postgravity_evaluate_gpu(struct particle_data *, const int *, int) {}

#endif /* CBE_INTEGRATOR */
