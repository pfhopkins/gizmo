/* density_gpu.cc — GPU-accelerated density evaluation via Kokkos.
 *
 * GPU translation unit (Kokkos/nvcc_wrapper). It provides:
 *   1. GPU neighbor list construction (BVH traversal via Kokkos::parallel_for)
 *   2. GPU density kernel (CSR neighbor iteration via Kokkos::parallel_for)
 *
 * Entry point: density_evaluate_gpu() — called from density.cc during
 * the h-iteration loop, replacing the CPU CSR path for each iteration.
 *
 * Architecture follows the cooling.cc pattern:
 *   - __managed__ All_dev with #define All All_dev for GPU global access
 *   - Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE> for device-accessible data
 *   - CPU gather → GPU kernel → CPU scatter
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

/* Standard and Kokkos headers BEFORE global_data_all_struct.h
 * (macros.h #define terminate(x) conflicts with std::terminate in <exception>) */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "../core/timestep_functions.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/sfc_tiles_functions.h"
#include "density_functions.h"
#include "gradient_functions.h"
#include "hydro_structs.h"
#include "../system/eigen_symmetric.h"
#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#ifndef HYDRO_SPH
#include "reimann.h"
#endif
/* Define xchange macro names so hydro_functions.h can reference the structs */
#define INPUT_STRUCT_NAME hydro_data_in
#define OUTPUT_STRUCT_NAME hydro_data_out
#include "hydro_functions.h"
#undef INPUT_STRUCT_NAME
#undef OUTPUT_STRUCT_NAME


/* TILE_PERIODIC_X/Y/Z defined in sfc_tiles.h (included via gpu_neighbor_list.h) */

/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */
GPU_ALL_SYNC_FUNC(density)


/* ================================================================
   GPU neighbor list construction (B1)
   ================================================================
   Strategy:
   - Build tiles + BVH on CPU (recursive, serial — fine for ~100s of tiles)
   - Copy tiles, BVH, pool, active_indices to SharedSpace
   - Pass 1: Kokkos::parallel_for counting neighbors per particle
   - Exclusive prefix scan → CSR offsets
   - Pass 2: Kokkos::parallel_for filling neighbor indices
   ================================================================ */

/* Structs and build/free functions are in mesh/gpu_neighbor_list.h/.cc,
   shared with the symmetric neighbor list build in accel.cc. */
#include "../mesh/gpu_neighbor_list.h"

/* Density's spatial index is now the module-shared g_step_sidx (mesh/gpu_neighbor_list.cc),
 * persisted across density rounds + symlist within a step, invalidated after drift in run.cc. */


/* ================================================================
   GPU density kernel (B2) — persistent session API
   ================================================================
   Session pattern avoids re-copying full P/CellP arrays on every
   h-iteration. Call sequence:
     density_gpu_session_begin()  — one-time full copy to SharedSpace
     density_evaluate_gpu() ×N   — reuses persistent arrays per iteration
     density_gpu_session_end()    — free SharedSpace arrays
   ================================================================ */

/* P/CellP storage migrated to decomp-scoped arena (system/gpu_particles_arena.h, Step 13 Phase 1).
 * density_session_num_total tracks the size acquired for THIS density h-iteration loop;
 * it gates the cached-CSR reuse logic below. */
#include "../system/gpu_particles_arena.h"
static int density_session_num_total = 0;

/* Cached oversized CSR neighbor list for h-iteration reuse.
   Built on the first iteration with inflated KernelRadius (DENSITY_H_BUFFER_FACTOR).
   Subsequent iterations reuse the same CSR and filter by current KernelRadius in the kernel.
   Uses a particle-index → CSR-offset lookup table so subsets of active particles
   can efficiently index into the original full CSR. */
#define DENSITY_H_BUFFER_FACTOR 1.3 /* oversize initial search radius by this factor */
static gpu_neighbor_list_t density_cached_gnl = {};
static int *density_csr_offset_lookup = NULL; /* maps particle index ii → position aa in original CSR */
static double *density_csr_max_h = NULL; /* buffered search radius used per particle when CSR was built */
static int density_cached_gnl_valid = 0;

void density_gpu_session_begin(struct particle_data *P_host, struct gas_cell_data *CellP_host, int num_total)
{
    density_session_num_total = num_total;
    gpu_particles_arena_set_site("density_gpu_session_begin");
    gpu_particles_arena_acquire(num_total, P_host, CellP_host);
    density_cached_gnl_valid = 0;
}

void density_gpu_session_end(void)
{
    /* Free cached CSR neighbor list */
    if(density_cached_gnl_valid) {
        gpu_ngb_list_free(&density_cached_gnl, gpu_step_sidx_ptr());
        density_cached_gnl_valid = 0;
    }
    if(density_csr_max_h) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(density_csr_max_h); density_csr_max_h = NULL;}
    if(density_csr_offset_lookup) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(density_csr_offset_lookup); density_csr_offset_lookup = NULL;}
    /* g_step_sidx is owned by gpu_neighbor_list module — invalidated by run.cc after drift, NOT here. */
    /* Note: do NOT release the arena here — it is decomp-scoped and may be consumed
     * by gradient/hydro/etc. after density returns. But DO invalidate: the host
     * postloop in density.cc (NV_T inversion, NumNgb normalization, etc.) mutates
     * P/CellP after the kernel scatter, so arena state is stale by session_end. */
    gpu_particles_arena_invalidate();
    density_session_num_total = 0;
}

/* Entry point called from density.cc during each h-iteration.
 * First iteration: builds oversized CSR neighbor list (KernelRadius * DENSITY_H_BUFFER_FACTOR)
 * and caches it. Subsequent iterations: reuses cached CSR, only re-runs density kernel
 * with current KernelRadius (pairs outside current h are filtered by the kernel).
 * If any particle's h grew beyond the buffered radius, the CSR is rebuilt. */
void density_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                          int num_total, int *active_indices_host, int num_active)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(density);
    /* Wrapper fast-path: with no active particles, the kernel does no work.
     * Skip the arena/scatter/diag overhead entirely so a no-active-gas h-iter
     * costs ms not seconds. */
    if(num_active == 0) { return; }
    double t_dens_gpu_start = my_second(), t_dens_gpu_phase;
    double t_diag_setup_end = 0, t_diag_ngb_end = 0, t_diag_scatter_end = 0; /* DIAG */
    double t_sidx_prebuild = 0; /* DIAG sub: gas SIDX prebuild only */
    struct particle_data *P_gpu;
    struct gas_cell_data *CellP_gpu;
    int session_active = (gpu_particles_arena_valid() && density_session_num_total == num_total
                          && gpu_particles_arena_capacity() >= num_total);

    if(session_active) {
        P_gpu = gpu_particles_arena_P();
        CellP_gpu = gpu_particles_arena_CellP();
        /* Sync KernelRadius from host for active particles (h changed during iteration).
         * Mark only the active indices dirty so the next cached ngb_build refreshes
         * compact_xyzh[ii*4+3] for these and skips the other ~12.4M-num_active. */
        if(num_active > 0) {
            for(int aa = 0; aa < num_active; aa++) {
                int ii = active_indices_host[aa];
                P_gpu[ii].KernelRadius = P_host[ii].KernelRadius;
            }
            /* Density iteration just mutated KernelRadius for active gas.
             * Mark dirty in BOTH the GPU SIDX tracker (gas-only and alltypes
             * caches both pick it up via range-route) AND the host glt cache
             * tracker (so the next ghost_exchange narrow-refits just those
             * tiles, ~ms vs ~0.2-0.5s full rescan). One helper call. */
            gizmo_mark_kernel_radius_dirty_indices(active_indices_host, num_active);
        }
    } else {
        P_gpu = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
        CellP_gpu = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct gas_cell_data));
        memcpy(P_gpu, P_host, num_total * sizeof(struct particle_data));
        memcpy(CellP_gpu, CellP_host, num_total * sizeof(struct gas_cell_data));
    }
    t_diag_setup_end = my_second(); /* DIAG: end of arena/P_gpu setup */

    /* Build the gas-only persistent SIDX (g_step_sidx) BEFORE density inflates
     * KernelRadius. The BVH's per-node hmax is set at build time and not refreshed
     * later — building it from inflated h would inflate node->hmax, causing the
     * SYMMETRIC consumer (symlist for gradients) to over-search by ~2.2× volume.
     * Building from un-inflated h gives correct hmax; compact_xyzh.h is refreshed
     * per call (so density still gets inflated h_i for its own oversize search).
     *
     * num_active==0 short-circuit: when no gas particles are active, density does
     * not inflate any h (the inflate loop below is gated on aa<num_active), so the
     * "build-before-inflate" correctness constraint is moot.  Skip the prebuild;
     * any downstream caller (mech_fb, sinks, etc.) that needs the SIDX will lazy-
     * build inside gpu_ngb_list_build with un-inflated h, which is correct.  Saves
     * ~1.3s on small-bin steps where no gas is active (e.g. DM/sink-only timebins). */
    if(session_active && num_active > 0 && !gpu_step_sidx_ptr()->valid) {
        double t_sidx_start = my_second();
        gpu_spatial_index_build(P_gpu, num_total, 1 /* gas only */, gpu_step_sidx_ptr(), "density-prebuild");
        t_sidx_prebuild = timediff(t_sidx_start, my_second());
    }

    /* Check if we can reuse the cached CSR list */
    int reuse_csr = 0;
    if(session_active && density_cached_gnl_valid && density_csr_offset_lookup && density_csr_max_h) {
        reuse_csr = 1;
        for(int aa = 0; aa < num_active; aa++) {
            int ii = active_indices_host[aa];
            if(density_csr_offset_lookup[ii] < 0) {
                reuse_csr = 0; break; /* particle wasn't in original active list */
            }
            if(P_host[ii].KernelRadius > density_csr_max_h[ii]) {
                reuse_csr = 0; break; /* h grew beyond the buffered search radius */
            }
        }
    }

    gpu_neighbor_list_t gnl;
    int *csr_lookup = NULL; /* maps particle index → CSR row for this call */

    if(reuse_csr) {
        /* Reuse cached CSR. Upload only the new active indices and use the lookup table
           to map each active particle to its CSR row in the cached list. */
        gnl = density_cached_gnl; /* share the CSR arrays (don't free them!) */
        csr_lookup = density_csr_offset_lookup;
        /* Re-upload active indices for this iteration */
        gnl.d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
        memcpy(gnl.d_active, active_indices_host, num_active * sizeof(int));
        gnl.num_active = num_active;
    } else {
        /* Build fresh CSR (first iteration, or buffer exceeded) */
        if(density_cached_gnl_valid) {
            gpu_ngb_list_free(&density_cached_gnl, gpu_step_sidx_ptr());
            density_cached_gnl_valid = 0;
        }

        /* Pass inflated search radii via the NGL API instead of mutating
         * P_gpu.KernelRadius and round-tripping a mark_h_dirty + restore +
         * mark_h_dirty again. NGB_SEARCH_ONEWAY only reads h_i (the source
         * radius), so an explicit per-active radii buffer is exactly equivalent
         * to the inflate; the kernel's per-pair predicate uses search_radii[aa]
         * for h_i and never touches compact_xyzh[j*4+3] for h_j. This removes
         * 2 of the ~6-8 compact_h_refresh fires per step (~1.25s each at
         * full N when h-dirty), the largest remaining tiny-N floor on gas
         * paths post-866aad55 BVH fix. */
        std::vector<double> inflated_radii;
        const double *radii_arg = NULL;
        if(num_active > 0) {
            inflated_radii.resize(num_active);
            for(int aa = 0; aa < num_active; aa++) {
                int ii = active_indices_host[aa];
                inflated_radii[aa] = P_host[ii].KernelRadius * DENSITY_H_BUFFER_FACTOR;
            }
            radii_arg = inflated_radii.data();
        }

        gpu_ngb_list_build(P_gpu, num_total, active_indices_host, num_active,
                           NGB_SEARCH_ONEWAY, 1 /* gas only */, &gnl,
                           gpu_step_sidx_ptr(),
                           1.0, radii_arg, NULL, "density");

        /* Cache the CSR and build lookup table */
        if(session_active) {
            density_cached_gnl = gnl;
            density_cached_gnl_valid = 1;
            /* Build particle-index → CSR-row lookup table */
            if(!density_csr_offset_lookup) {
                density_csr_offset_lookup = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(int));
            }
            memset(density_csr_offset_lookup, -1, num_total * sizeof(int));
            /* Store the buffered search radius used for each particle */
            if(!density_csr_max_h) {
                density_csr_max_h = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(double));
            }
            memset(density_csr_max_h, 0, num_total * sizeof(double));
            for(int aa = 0; aa < num_active; aa++) {
                int ii = active_indices_host[aa];
                density_csr_offset_lookup[ii] = aa;
                density_csr_max_h[ii] = P_host[ii].KernelRadius * DENSITY_H_BUFFER_FACTOR;
            }
            csr_lookup = density_csr_offset_lookup;
        }
    }

    t_diag_ngb_end = my_second(); /* DIAG: end of NGP/CSR build */
    t_dens_gpu_phase = t_diag_ngb_end;

    /* GPU density accumulation kernel */
    {
        int *offsets = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active = gnl.d_active;
        int *csr_map = csr_lookup; /* maps particle index → CSR row (NULL if direct aa indexing) */
        int use_csr_map = reuse_csr ? 1 : 0;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;

        gizmo_gpu_kernel_launch("density_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* When reusing cached CSR, map particle index to its row in the original CSR */
            int csr_row = aa;
            if(use_csr_map && csr_map) { csr_row = csr_map[ii]; }

            /* particle2in: load searching particle data */
            struct density_evaluate_data_in_ local;
            local.Type = kp[ii].Type;
            local.KernelRadius = kp[ii].KernelRadius;
            local.Pos = kp[ii].Pos;
            if(kp[ii].Type == 0) { local.Vel = kc[ii].VelPred; } else { local.Vel = kp[ii].Vel; }
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
            if(kp[ii].Type == 0) { local.Accel = All.cf_a2inv * kp[ii].GravAccel + kc[ii].HydroAccel; }
#endif
#ifdef GALSF_SUBGRID_WINDS
            if(kp[ii].Type == 0) { local.DelayTime = kc[ii].DelayTime; } else { local.DelayTime = 0; }
#endif

            /* Initialize output */
            struct density_evaluate_data_out_ out;
            memset(&out, 0, sizeof(out));
#if defined(SINK_PARTICLES)
            out.Sink_TimeBinGasNeighbor = TIMEBINS;
#endif

            /* Kernel setup */
            struct kernel_density kernel;
            double h2 = local.KernelRadius * local.KernelRadius;
            kernel_hinv(local.KernelRadius, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

            /* Accumulate over neighbors — the CSR list may contain extra neighbors from
               the oversized initial search; density_accumulate_neighbor filters by r2 >= h2 */
            for(int idx = offsets[csr_row]; idx < offsets[csr_row + 1]; idx++)
            {
                int j = neighbors[idx];
                density_accumulate_neighbor(&local, &out, &kernel, j, h2, kp, kc);
            }

            /* out2particle: scatter results into SharedSpace arrays */
            kp[ii].NumNgb = out.Ngb;
            kp[ii].DrkernNgbFactor = out.DrkernNgb;
            kp[ii].Particle_DivVel = out.Particle_DivVel;

            if(kp[ii].Type == 0)
            {
                kc[ii].Density = out.Rho;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
                kc[ii].ParticleVel = out.ParticleVel;
#endif
                for(int k = 0; k < 6; k++) { kc[ii].NV_T.data[k] = out.NV_T.data[k]; }
                kc[ii].NV_T_face_weights = out.NV_T_face_weights;
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
                kc[ii].GradH_numer = out.GradH_numer;
                kc[ii].GradH_denom = out.GradH_denom;
#endif
#ifdef HYDRO_SPH
                kc[ii].DrkernHydroSumFactor = out.DrkernHydroSumFactor;
#endif
#ifdef HYDRO_PRESSURE_SPH
                kc[ii].EgyWtDensity = out.EgyRho;
#endif
#if defined(TURB_DRIVING)
                kc[ii].SmoothedVel = out.GasVel;
#endif
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
                for(int k1 = 0; k1 < 3; k1++)
                    for(int k2 = 0; k2 < 3; k2++) {
                        kc[ii].NV_D[k1][k2] = out.NV_D[k1][k2];
                        kc[ii].NV_A[k1][k2] = out.NV_A[k1][k2];
                    }
#endif
            }

#if defined(GRAIN_FLUID)
            if((1 << kp[ii].Type) & (GRAIN_PTYPES)) {
                kp[ii].Gas_Density = out.Rho; /* must write to P.Gas_Density, not CellP.Density */
                kp[ii].Gas_InternalEnergy = out.Gas_InternalEnergy;
                kp[ii].Gas_Velocity = out.GasVel;
#if defined(GRAIN_LORENTZFORCE)
                kp[ii].Gas_B = out.Gas_B;
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
                for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { kp[ii].Gas_VolatileSpecies[kv] = out.Gas_VolatileSpecies[kv]; }
#endif
            }
#endif
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
            kp[ii].DensityAroundParticle = out.Rho; /* must scatter both DensityAroundParticle and GradRho */
            kp[ii].GradRho = out.GradRho;
#endif
#if defined(RT_SOURCE_INJECTION)
#if defined(RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION)
            if(All.TimeStep == 0)
#endif
            if((1 << kp[ii].Type) & (RT_SOURCES)) {kp[ii].KernelSum_Around_RT_Source = out.KernelSum_Around_RT_Source;}
#endif
#if defined(SINK_PARTICLES)
            if(kp[ii].Type == 5) {
                kp[ii].Sink_TimeBinGasNeighbor = out.Sink_TimeBinGasNeighbor;
#if defined(BH_ACCRETE_NEARESTFIRST) || defined(SINGLE_STAR_TIMESTEPPING)
                kp[ii].Sink_dr_to_NearestGasNeighbor = out.Sink_dr_to_NearestGasNeighbor;
#endif
            }
#endif
        }); /* end gizmo_gpu_kernel_launch */
    }

    if(reuse_csr) {
        /* Only free the d_active we uploaded this iteration; CSR arrays belong to cache */
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl.d_active);
    } else if(!session_active || !density_cached_gnl_valid) {
        /* No session or cache not valid: free everything */
        gpu_ngb_list_free(&gnl, gpu_step_sidx_ptr());
    }
    /* If session_active and we just built + cached the CSR, don't free it (cache owns it) */

    double t_dens_kernel = timediff(t_dens_gpu_phase, my_second());
    t_dens_gpu_phase = my_second();

    /* Scatter results back to host arrays for active particles */
    {double t_diag_scatter_start = my_second();
    for(int aa = 0; aa < num_active; aa++) {
        int ii = active_indices_host[aa];
        P_host[ii] = P_gpu[ii];
        CellP_host[ii] = CellP_gpu[ii];
    }
    t_diag_scatter_end = my_second() - t_diag_scatter_start;} /* DIAG: reuse as elapsed */

    /* Only free if no persistent session (session_end handles cleanup) */
    if(!session_active) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(CellP_gpu);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_gpu);
    }

    {
        double t_arena   = timediff(t_dens_gpu_start, t_diag_setup_end);
        double t_ngb     = timediff(t_diag_setup_end, t_diag_ngb_end);
        double t_kernel  = t_dens_kernel;
        double t_scatter = t_diag_scatter_end; /* stored as elapsed */
        /* Phase 7 sub-buckets — env-gated via GIZMO_VERBOSE_DIAG; no-op when off.
         * Note t_ngb includes the SIDX prebuild when present; record SIDX
         * separately so density_ngb_build reflects only the ngb_list_build cost. */
        gizmo_step_phase_record("density_arena", t_arena);
        gizmo_step_phase_record("density_sidx_prebuild", t_sidx_prebuild);
        gizmo_step_phase_record("density_ngb_build", t_ngb - t_sidx_prebuild);
        gizmo_step_phase_record("density_kernel", t_kernel);
        gizmo_step_phase_record("density_scatter", t_scatter);
        if(ThisTask == 0 && gizmo_verbose_diag()) {
            double t_total = timediff(t_dens_gpu_start, my_second());
            printf("[DIAG_DENS step=%d N=%d] total=%.3f arena=%.3f ngb_build=%.3f sidx_prebuild=%.3f gpu_kernel=%.3f scatter=%.3f\n",
                   (int)All.NumCurrentTiStep, num_active, t_total, t_arena, t_ngb, t_sidx_prebuild, t_kernel, t_scatter);
            fflush(stdout);
        }
    }
}

/* ================================================================
   GPU gradient kernel (B3)
   ================================================================
   For each active particle: load data, iterate symmetric CSR neighbors,
   accumulate gradients via gradient_functions.h, return output structs.
   The caller (gradients.cc) does the scatter into CellP + GasGradDataPasser.
   ================================================================ */

/* Entry point for GPU gradient evaluation.
 * Takes a pre-built symmetric CSR neighbor list (offsets/neighbors).
 * Fills out_host[0..num_active-1] with accumulated gradient outputs.
 * Caller must allocate out_host (num_active * sizeof(GasGraddata_out_)).
 * gradient_iteration: 0 = standard gradients, >0 = MHD constrained gradient iteration */
void gradient_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                           int num_total, int *active_indices_host, int num_active,
                           int *csr_offsets_host, int *csr_neighbors_host, int csr_total_pairs,
                           void *out_host_void, int gradient_iteration)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(density);
    double t_grad_start = my_second(); /* DIAG */
    struct GasGraddata_out_ *out_host = (struct GasGraddata_out_ *)out_host_void;

    /* Persistent decomp-scoped arena (Step 13 Phase 1). Replaces per-call
     * SharedSpace alloc+memcpy of P/CellP. Acquire is a no-op fast path when
     * the arena is already valid from a prior kernel in this step. */
    gpu_particles_arena_set_site("density_gpu_gradients");
    gpu_particles_arena_acquire(num_total, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();
    double t_grad_arena = my_second(); /* DIAG: end of arena acquire */

    /* Copy CSR neighbor list to SharedSpace */
    int *d_offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));
    int *d_neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((csr_total_pairs > 0) ? csr_total_pairs : 1) * sizeof(int));
    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(d_offsets, csr_offsets_host, (num_active + 1) * sizeof(int));
    memcpy(d_neighbors, csr_neighbors_host, csr_total_pairs * sizeof(int));
    memcpy(d_active, active_indices_host, num_active * sizeof(int));
    double t_grad_csr_copy = my_second(); /* DIAG: end of CSR copy to SharedSpace */

    /* Allocate output array in SharedSpace */
    struct GasGraddata_out_ *d_out = (struct GasGraddata_out_ *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(struct GasGraddata_out_));

    PRINT_STATUS("  GPU gradient: %d active, %d pairs", num_active, csr_total_pairs);

    /* GPU gradient accumulation kernel */
    {
        int *offsets = d_offsets;
        int *neighbors = d_neighbors;
        int *active = d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct GasGraddata_out_ *kout = d_out;
        int grad_iter = gradient_iteration;

        gizmo_gpu_kernel_launch("gradient_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* particle2in equivalent: load searching particle data */
            struct GasGraddata_in_ local;
            memset(&local, 0, sizeof(local));
            local.Pos = kp[ii].Pos;
            local.KernelRadius = kp[ii].KernelRadius;
            local.Mass = kp[ii].Mass;
            if(local.Mass < 0) {local.Mass = 0;}
            int sph_gradients_flag_i = SHOULD_I_USE_SPH_GRADIENTS(kc[ii].ConditionNumber);
            if(sph_gradients_flag_i) {local.Mass *= -1;}
#ifdef MHD_CONSTRAINED_GRADIENT
            if(grad_iter > 0) {if(kc[ii].FlagForConstrainedGradients <= 0) {local.Mass = 0;}}
#endif

            /* Load gradient quantities */
            local.GQuant.Density = kc[ii].Density;
            local.GQuant.Pressure = kc[ii].Pressure;
            local.GQuant.Velocity = kc[ii].VelPred;
#ifdef MAGNETIC
            local.GQuant.B = kc[ii].BPred * (kc[ii].Density / kp[ii].Mass);
#ifdef DIVBCLEANING_DEDNER
            local.GQuant.Phi = kc[ii].PhiPred / kp[ii].Mass;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
            local.GQuant.InternalEnergy = kc[ii].InternalEnergyPred;
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
            local.GQuant.ElectronNumberDensity = kc[ii].n_e();
            local.GQuant.ElectronTemperature   = kc[ii].T_e();
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
            local.GQuant.E_battery_T2 = kc[ii].E_battery_T2_cell;
#endif
#ifdef DOGRAD_SOUNDSPEED
            local.GQuant.SoundSpeed = kc[ii].effective_soundspeed();
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {local.GQuant.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(ii, k, kc);}
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.GQuant.Metallicity[k] = kp[ii].Metallicity[k];}
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
            {for(int k=0;k<N_RT_FREQ_BINS;k++) {
                 local.GQuant.Rad_E_gamma[k] = kc[ii].Rad_E_gamma_Pred[k]; /* store RAW, kernel applies V_i_inv */
                 local.GQuant.Rad_E_gamma_ET[k] = kc[ii].ET[k]; /* store RAW ET tensor, kernel multiplies with Rad_E_gamma*V_i_inv */
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
                 for(int k2=0;k2<3;k2++) {local.GQuant.Rad_Flux[k][k2] = kc[ii].Rad_Flux_Pred[k][k2];} /* store RAW, kernel applies V_i_inv */
#endif
             }}
#endif
#ifdef TURB_DIFF_DYNAMIC
            local.GQuant.Velocity_bar = kc[ii].Velocity_bar;
            local.Norm_hat = kc[ii].Norm_hat;
#ifdef GALSF_SUBGRID_WINDS
            local.DelayTime = kc[ii].DelayTime;
#endif
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
            local.ConditionNumber = kc[ii].ConditionNumber;
            local.NV_T = kc[ii].NV_T;
            for(int k=0;k<3;k++) for(int k2=0;k2<3;k2++) {local.BGrad[k][k2] = kc[ii].Gradients.B[k][k2];}
#ifdef MHD_MODIFIED_GRADIENT
            local.MG_cgcoeff = kc[ii].MG_cgcoeff;
#endif
#ifdef MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV
            local.PhiGrad = kc[ii].Gradients.Phi;
#endif
#endif

            if(sph_gradients_flag_i) {local.Mass *= -1;} /* negate Mass as flag for SPH gradients */

            /* Initialize output */
            struct GasGraddata_out_ out;
            memset(&out, 0, sizeof(out));

            /* Pre-compute kernel quantities */
            double h_i = local.KernelRadius;
            double hinv, hinv3, hinv4;
            kernel_hinv(h_i, &hinv, &hinv3, &hinv4);
            if(local.Mass < 0) {local.Mass *= -1;} /* restore for V_i computation */
            double V_i = local.Mass / local.GQuant.Density;
            if(sph_gradients_flag_i) {local.Mass *= -1;} /* re-negate for kernel */

            int kernel_mode_i = -1;
            if(sph_gradients_flag_i) kernel_mode_i = 0;
#if defined(HYDRO_SPH) || defined(KERNEL_CRK_FACES)
            kernel_mode_i = 0;
#endif

            struct kernel_GasGrad kernel;
            kernel.h_i = h_i;

            /* Accumulate over symmetric neighbors */
            for(int idx = offsets[aa]; idx < offsets[aa + 1]; idx++)
            {
                int j = neighbors[idx];
                gradient_accumulate_neighbor(&local, &out, &kernel, j,
                                             sph_gradients_flag_i, V_i,
                                             hinv, hinv3, hinv4, kernel_mode_i,
                                             kp, kc);
            }

            /* Store output for this particle */
            kout[aa] = out;
        }); /* end gizmo_gpu_kernel_launch */
    }
    double t_grad_kernel = my_second(); /* DIAG: kernel already fence'd by gizmo_gpu_kernel_launch */

    /* Copy output back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct GasGraddata_out_));
    double t_grad_copyout = my_second(); /* DIAG: end of output copy */

    /* Cleanup SharedSpace (P/CellP owned by arena — do NOT free here).
     * Invalidate: gradients.cc post-scatters d_out values into host CellP.Gradients
     * after we return, so host will diverge from arena before the next kernel. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_offsets);
    gpu_particles_arena_invalidate();

    {
        double t_arena    = timediff(t_grad_start,    t_grad_arena);
        double t_csr_copy = timediff(t_grad_arena,    t_grad_csr_copy);
        double t_kernel   = timediff(t_grad_csr_copy, t_grad_kernel);
        double t_out_copy = timediff(t_grad_kernel,   t_grad_copyout);
        /* Phase 7 sub-buckets — env-gated; no-op when GIZMO_VERBOSE_DIAG off */
        gizmo_step_phase_record("gradient_arena",    t_arena);
        gizmo_step_phase_record("gradient_csr_copy", t_csr_copy);
        gizmo_step_phase_record("gradient_kernel",   t_kernel);
        gizmo_step_phase_record("gradient_out_copy", t_out_copy);
        if(ThisTask == 0 && gizmo_verbose_diag()) {
            long long pairs_mb = (long long)csr_total_pairs * 2 * sizeof(int) / (1024*1024);
            printf("[DIAG_GRAD step=%d N=%d pairs=%d(~%lldMB)] arena=%.3f csr_copy=%.3f gpu_kernel=%.3f out_copy=%.3f total=%.3f\n",
                   (int)All.NumCurrentTiStep, num_active, csr_total_pairs, pairs_mb,
                   t_arena, t_csr_copy, t_kernel, t_out_copy,
                   timediff(t_grad_start, t_grad_copyout));
            fflush(stdout);
        }
    }
}


/* ================================================================
   GPU hydro force kernel (B3b)
   ----------------------------------------------------------------
   Kernel writes (host-visible, by index):
     i-side (active sources, indexed by aa->ii):
       kout[aa]                            — per-active output (flux deltas, MaxSignalVel, etc.)
     j-side (neighbors, indexed by j = neighbors[idx]):
       P_gpu[j].wakeup                     — HYDRO_ATOMIC_MAX
     #ifdef HYDRO_MESHLESS_FINITE_VOLUME
       CellP_gpu[j].dMass                  — HYDRO_ATOMIC_ADD
     #endif
     scalar:
       *NeedToWakeup                       — HYDRO_ATOMIC_STORE on wakeup trigger

   Sparse-scatter target: walk neighbors[0..csr_total_pairs] and scatter
   only the (deduplicated by idempotent assignment) j fields above. NEVER
   memcpy the full P/CellP arrays — those are populated host->arena once
   per arena cycle and not mutated by the hydro kernel.
   ================================================================ */

void hydro_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                        int num_total, int *active_indices_host, int num_active,
                        int *csr_offsets_host, int *csr_neighbors_host, int csr_total_pairs,
                        void *out_host_void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(density);
    struct hydro_data_out *out_host = (struct hydro_data_out *)out_host_void;
    double t_hyd_start = my_second(); /* Phase 7 sub-bucket timing */

    /* Persistent decomp-scoped arena (Step 13 Phase 1). Replaces per-call
     * SharedSpace alloc+memcpy. Fast path skips memcpy when arena is valid. */
    gpu_particles_arena_set_site("density_gpu_hydro");
    gpu_particles_arena_acquire(num_total, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();
    double t_hyd_arena = my_second();

    /* Copy CSR neighbor list to SharedSpace */
    int *d_offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));
    int *d_neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((csr_total_pairs > 0) ? csr_total_pairs : 1) * sizeof(int));
    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(d_offsets, csr_offsets_host, (num_active + 1) * sizeof(int));
    memcpy(d_neighbors, csr_neighbors_host, csr_total_pairs * sizeof(int));
    memcpy(d_active, active_indices_host, num_active * sizeof(int));
    double t_hyd_csr_copy = my_second();

    /* Copy TimeBinActive to SharedSpace */
    int *d_TimeBinActive = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(TIMEBINS * sizeof(int));
    memcpy(d_TimeBinActive, TimeBinActive, TIMEBINS * sizeof(int));

    /* Wakeup flag in SharedSpace */
    int *d_NeedToWakeup = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
    *d_NeedToWakeup = 0;

    /* Output array in SharedSpace */
    struct hydro_data_out *d_out = (struct hydro_data_out *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(struct hydro_data_out));

    PRINT_STATUS("  GPU hydro: %d active, %d pairs", num_active, csr_total_pairs);

    /* Verification harness (β): GIZMO_VERIFY_KERNEL_WRITES=1 snapshots the
     * arena fields the kernel is permitted to write (P[j].wakeup, CellP[j].dMass
     * MFV-only) so the post-kernel sparse scatter can prove the kernel touched
     * ONLY j's that appear in csr_neighbors_host[]. O(N) cost, off by default.
     * Used once after each new wrapper sparse-scatter conversion to confirm
     * the kernel-writes header comment is complete. */
    static int gizmo_verify_init = 0, gizmo_verify_on = 0;
    if(!gizmo_verify_init) {
        gizmo_verify_init = 1;
        gizmo_verify_on = (getenv("GIZMO_VERIFY_KERNEL_WRITES") != nullptr) ? 1 : 0;
    }
    short int *verify_snap_wakeup = nullptr;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble *verify_snap_dMass = nullptr;
#endif
    if(gizmo_verify_on && num_total > 0) {
        verify_snap_wakeup = (short int *) malloc((size_t)num_total * sizeof(short int));
        for(int j = 0; j < num_total; j++) verify_snap_wakeup[j] = P_gpu[j].wakeup;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        verify_snap_dMass = (MyDouble *) malloc((size_t)num_total * sizeof(MyDouble));
        for(int j = 0; j < num_total; j++) verify_snap_dMass[j] = CellP_gpu[j].dMass;
#endif
    }

#if defined(GALSF_ISMDUSTCHEM_MODEL)
    /* Pre-compute ISMDustChem passive scalar diffusion values on host.
     * return_ismdustchem_species_of_interest_for_diffusion_and_yields() reads
     * All.* grain tables not mirrored to All_dev, so it is not device-callable. */
    double *d_ismdc = nullptr;
    const int n_ismdc = NUM_ISMDUSTCHEM_PASSIVE_SCALARS;
    if(n_ismdc > 0 && num_active > 0) {
        double *ismdc_host = new double[(size_t)num_active * n_ismdc];
        for(int aa = 0; aa < num_active; aa++) {
            int ii = active_indices_host[aa];
            for(int ki = 0; ki < n_ismdc; ki++) {
                ismdc_host[aa * n_ismdc + ki] =
                    return_ismdustchem_species_of_interest_for_diffusion_and_yields(
                        ii, ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY + ki, 0, CellP_host);
            }
        }
        d_ismdc = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((size_t)num_active * n_ismdc * sizeof(double));
        memcpy(d_ismdc, ismdc_host, (size_t)num_active * n_ismdc * sizeof(double));
        delete[] ismdc_host;
    }
#endif

    /* GPU hydro force kernel */
    {
        int *offsets = d_offsets;
        int *neighbors = d_neighbors;
        int *active = d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct hydro_data_out *kout = d_out;
        int *kTimeBinActive = d_TimeBinActive;
        int *kNeedWakeup = d_NeedToWakeup;
#if defined(GALSF_ISMDUSTCHEM_MODEL)
        double *ismdc = d_ismdc;
        const int ismdc_n = n_ismdc;
#endif

        Kokkos::parallel_for("hydro_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* particle2in equivalent: load searching particle data */
            struct hydro_data_in local;
            memset(&local, 0, sizeof(local));
            local.Pos = kp[ii].Pos;
            local.Vel = kc[ii].VelPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            local.ParticleVel = kc[ii].VelPred;
#endif
            local.KernelRadius = kp[ii].KernelRadius;
            local.Mass = kp[ii].Mass;
            local.Density = kc[ii].Density;
            local.Pressure = kc[ii].Pressure;
            local.ConditionNumber = kc[ii].ConditionNumber;
#ifdef MHD_CONSTRAINED_GRADIENT
            if(kc[ii].FlagForConstrainedGradients == 0) {local.ConditionNumber *= -1;} /* sign encodes gradient flag status, matching particle2in_hydra */
#endif
            local.FaceClosureError = kc[ii].FaceClosureError;
            local.InternalEnergyPred = kc[ii].InternalEnergyPred;
            local.SoundSpeed = kc[ii].effective_soundspeed();
            local.dt_hydrostep_i = get_particle_timestep_in_physical(ii, kp);
            local.DrkernNgbFactor = kp[ii].DrkernNgbFactor;
            local.Gradients.Density = kc[ii].Gradients.Density;
            local.Gradients.Pressure = kc[ii].Gradients.Pressure;
            local.Gradients.Velocity = kc[ii].Gradients.Velocity;
            local.NV_T = kc[ii].NV_T;
            local.TimeBin = kp[ii].TimeBin;
#ifdef MAGNETIC
            local.BPred = kc[ii].Bfield(); /* Bfield() = BPred * Density/Mass: convert from mass-weighted storage to physical B */
            local.Gradients.B = kc[ii].Gradients.B;
#ifdef MHD_BATTERY_MECHANISMS
#if (MHD_BATTERY_MECHANISMS & 1)
            local.Gradients.ElectronNumberDensity = kc[ii].Gradients.ElectronNumberDensity;
            local.Gradients.ElectronTemperature = kc[ii].Gradients.ElectronTemperature;
#endif
#endif
#ifdef DIVBCLEANING_DEDNER
            local.PhiPred = kc[ii].PhiPred / kp[ii].Mass;
            local.Gradients.Phi = kc[ii].Gradients.Phi;
#endif
#ifdef MHD_MODIFIED_GRADIENT
            local.MG_cgcoeff = kc[ii].MG_cgcoeff;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
            local.Gradients.InternalEnergy = kc[ii].Gradients.InternalEnergy;
#endif
#ifdef DOGRAD_SOUNDSPEED
            local.Gradients.SoundSpeed = kc[ii].Gradients.SoundSpeed;
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.Metallicity[k] = kp[ii].Metallicity[k];}
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {local.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(ii, k, kc);}
#endif
#ifdef CONDUCTION
            local.Kappa_Conduction = kc[ii].Kappa_Conduction;
#endif
#ifdef VISCOSITY
            local.Eta_ShearViscosity = kc[ii].Eta_ShearViscosity;
            local.Zeta_BulkViscosity = kc[ii].Zeta_BulkViscosity;
#endif
#ifdef TURB_DIFFUSION
            local.TD_DiffCoeff = kc[ii].TD_DiffCoeff;
#endif
#if defined(KERNEL_CRK_FACES)
            for(int k=0;k<16;k++) {local.Tensor_CRK_Face_Corrections[k] = kc[ii].Tensor_CRK_Face_Corrections[k];}
#endif
#ifdef HYDRO_SPH
            local.DrkernHydroSumFactor = kc[ii].DrkernHydroSumFactor;
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
            local.alpha = kc[ii].alpha_limiter * kc[ii].alpha;
#else
            local.alpha = kc[ii].alpha_limiter;
#endif
#endif
#ifdef HYDRO_PRESSURE_SPH
            local.EgyWtRho = kc[ii].EgyWtDensity;
#endif
#ifdef MHD_NON_IDEAL
            local.Eta_MHD_OhmicResistivity_Coeff = kc[ii].Eta_MHD_OhmicResistivity_Coeff;
            local.Eta_MHD_HallEffect_Coeff = kc[ii].Eta_MHD_HallEffect_Coeff;
            local.Eta_MHD_AmbiPolarDiffusion_Coeff = kc[ii].Eta_MHD_AmbiPolarDiffusion_Coeff;
#endif
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
            local.Balpha = kc[ii].Balpha;
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.Gradients.Metallicity[k] = kc[ii].Gradients.Metallicity[k];}
#endif
#if defined(GALSF_ISMDUSTCHEM_MODEL)
            for(int ki=0;ki<ismdc_n;ki++) {local.Metallicity[ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY+ki] = ismdc[aa*ismdc_n+ki];}
#endif
#ifdef CHIMES_TURB_DIFF_IONS
            for(int k=0;k<ChimesGlobalVars.totalNumberOfSpecies;k++) {local.ChimesNIons[k] = kc[ii].ChimesNIons[k];}
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
            for(int k=0;k<N_RT_FREQ_BINS;k++) {local.Gradients.Rad_E_gamma_ET[k] = kc[ii].Gradients.Rad_E_gamma_ET[k];}
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int k=0;k<N_RT_FREQ_BINS;k++) {
                for(int k2=0;k2<3;k2++) {local.Gradients.Rad_E_gamma_Grad[k][k2] = kc[ii].Gradients.Rad_E_gamma_Grad[k][k2];}
                for(int k2=0;k2<3;k2++) {for(int k3=0;k3<3;k3++) {local.Gradients.Rad_Flux_Grad[k][k2][k3] = kc[ii].Gradients.Rad_Flux_Grad[k][k2][k3];}}
            }
#endif
#ifdef RT_SOLVER_EXPLICIT
            for(int k=0;k<N_RT_FREQ_BINS;k++) {
                local.Rad_E_gamma[k] = kc[ii].Rad_E_gamma_Pred[k];
                local.Rad_Kappa[k] = kc[ii].Rad_Kappa[k];
                local.RT_DiffusionCoeff[k] = rt_diffusion_coefficient(ii, k, kc);
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
                local.ET[k] = kc[ii].ET[k];
#endif
#ifdef RT_EVOLVE_FLUX
                local.Rad_Flux[k] = kc[ii].Rad_Flux_Pred[k];
#endif
#if defined(RT_EVOLVE_INTENSITIES)
                for(int k2=0;k2<N_RT_INTENSITY_BINS;k2++) {local.Rad_Intensity_Pred[k][k2] = kc[ii].Rad_Intensity_Pred[k][k2];}
#endif
            }
#ifdef RT_INFRARED
            local.Radiation_Temperature = kc[ii].Radiation_Temperature;
#endif
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {
                local.CosmicRayDiffusionCoeff[k] = kc[ii].CosmicRayDiffusionCoeff[k];
                local.CosmicRayFlux[k] = kc[ii].CosmicRayFluxPred[k];
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
                for(int k2=0;k2<2;k2++) {local.CosmicRayAlfvenEnergy[k][k2] = kc[ii].CosmicRayAlfvenEnergyPred[k][k2];}
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
                local.CR_number_to_energy_ratio[k] = kc[ii].CosmicRay_Number_in_Bin[k] / (kc[ii].CosmicRayEnergy[k] + MIN_REAL_NUMBER);
                local.CR_number_to_energy_ratio[k] *= kc[ii].Flux_Number_to_Energy_Correction_Factor[k];
#endif
            }
#endif
#if defined(EOS_ELASTIC) || defined(EOS_ANEOS)
            local.CompositionType = kc[ii].CompositionType;
#endif
#ifdef EOS_ELASTIC
            for(int k=0;k<3;k++) {for(int k2=0;k2<3;k2++) {local.Elastic_Stress_Tensor[k][k2] = kc[ii].Elastic_Stress_Tensor_Pred[k][k2];}}
#endif
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
            local.Damage = kc[ii].Damage;
#endif
#ifdef GALSF_SUBGRID_WINDS
            local.DelayTime = kc[ii].DelayTime;
#endif

            /* Initialize output and workspace */
            struct hydro_data_out out;
            memset(&out, 0, sizeof(out));
            struct kernel_hydra kernel;
            memset(&kernel, 0, sizeof(kernel));
            struct Conserved_var_Riemann Fluxes;

            kernel.h_i = local.KernelRadius;
            kernel.sound_i = local.SoundSpeed;
            kernel.spec_egy_u_i = local.InternalEnergyPred;
            out.MaxSignalVel = kernel.sound_i; /* must match hydro_evaluate.h line 82 */
#ifdef MAGNETIC
            {
                double fac_magnetic_pressure_loc = 1.0 / All.cf_atime; /* B*B*fac = pressure units */
                kernel.b2_i = local.BPred.norm_sq(); /* raw B^2, without fac_magnetic_pressure */
                kernel.alfven2_i = kernel.b2_i * fac_magnetic_pressure_loc / local.Density;
                kernel.alfven2_i = DMIN(kernel.alfven2_i, 1000. * kernel.sound_i * kernel.sound_i);
            }
#endif

            /* Accumulate over symmetric neighbors */
            for(int idx = offsets[aa]; idx < offsets[aa + 1]; idx++)
            {
                int j = neighbors[idx];
                memset(&Fluxes, 0, sizeof(Fluxes));
                hydro_accumulate_neighbor(local, out, kernel, Fluxes, j,
                                          local.dt_hydrostep_i, kp, kc,
                                          kTimeBinActive, kNeedWakeup);
            }

            /* Store output for this particle */
            kout[aa] = out;
        }); /* end KOKKOS_LAMBDA */
        Kokkos::fence();

        gizmo_gpu_check_last_error("hydro kernel", num_active);
    }
    double t_hyd_kernel = my_second();

    /* Copy output back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct hydro_data_out));

    /* Copy wakeup flag back */
    if(*d_NeedToWakeup) NeedToWakeupParticles_local = 1;
    double t_hyd_out_copy = my_second();

    /* SPARSE scatter (replaces former full-num_total loop). The kernel writes
     * only P_gpu[j].wakeup (atomic_max) and, MFV-only, CellP_gpu[j].dMass
     * (atomic_add) — and only for j's that appear in the CSR neighbors[].
     * Walk the host-side CSR directly; idempotent assignment makes duplicate
     * j's across active-i ranges correct without a dedupe pass.
     * Ghost writeback below handles MPI communication of these deltas. */
    for(int idx = 0; idx < csr_total_pairs; idx++) {
        int j = csr_neighbors_host[idx];
        P_host[j].wakeup = P_gpu[j].wakeup;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP_host[j].dMass = CellP_gpu[j].dMass;
#endif
    }
    double t_hyd_scatter = my_second();

    /* Verification harness (β): for any untouched j, P_gpu[j] must match the
     * pre-kernel snapshot. Any violation means the kernel-writes header is
     * incomplete (a write the audit missed). Off by default. */
    if(verify_snap_wakeup) {
        char *touched = (char *) calloc((size_t)num_total, sizeof(char));
        for(int idx = 0; idx < csr_total_pairs; idx++) touched[csr_neighbors_host[idx]] = 1;
        long wakeup_violations = 0;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        long dMass_violations = 0;
#endif
        for(int j = 0; j < num_total; j++) {
            if(touched[j]) continue;
            if(P_gpu[j].wakeup != verify_snap_wakeup[j]) wakeup_violations++;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            if(CellP_gpu[j].dMass != verify_snap_dMass[j]) dMass_violations++;
#endif
        }
        if(wakeup_violations) {
            printf("[VERIFY_KERNEL_WRITES] hydro: %ld untouched j's had P_gpu.wakeup mutated (rank=%d, num_total=%d, csr_pairs=%d)\n",
                   wakeup_violations, ThisTask, num_total, csr_total_pairs);
            fflush(stdout);
        }
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        if(dMass_violations) {
            printf("[VERIFY_KERNEL_WRITES] hydro: %ld untouched j's had CellP_gpu.dMass mutated (rank=%d)\n",
                   dMass_violations, ThisTask);
            fflush(stdout);
        }
        free(verify_snap_dMass);
#endif
        free(touched);
        free(verify_snap_wakeup);
    }

    /* Cleanup SharedSpace (P/CellP owned by arena — do NOT free here).
     * The scatter above syncs wakeup+dMass; any other arena-side writes by
     * the kernel may be unscattered, so invalidate to force the next kernel
     * to re-acquire from host. (ghost_writeback_hydro already invalidates,
     * but be defensive in case hydro is invoked again before that runs.) */
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    if(d_ismdc) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_ismdc); }
#endif
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_NeedToWakeup);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_TimeBinActive);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_offsets);
    gpu_particles_arena_invalidate();

    /* Phase 7 sub-buckets — env-gated; no-op when GIZMO_VERBOSE_DIAG off */
    {
        double t_postloop_end = my_second();
        gizmo_step_phase_record("hydro_arena",       timediff(t_hyd_start,    t_hyd_arena));
        gizmo_step_phase_record("hydro_csr_copy",    timediff(t_hyd_arena,    t_hyd_csr_copy));
        gizmo_step_phase_record("hydro_kernel",      timediff(t_hyd_csr_copy, t_hyd_kernel));
        gizmo_step_phase_record("hydro_out_copy",    timediff(t_hyd_kernel,   t_hyd_out_copy));
        gizmo_step_phase_record("hydro_scatter",     timediff(t_hyd_out_copy, t_hyd_scatter));
        gizmo_step_phase_record("hydro_postloop",    timediff(t_hyd_scatter,  t_postloop_end));
    }
}

