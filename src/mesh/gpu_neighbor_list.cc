/* gpu_neighbor_list.cc — GPU-accelerated neighbor list construction.
 *
 * Extracted from hydro/density_gpu.cc for reuse by any code that needs
 * a GPU-built CSR neighbor list (density, symmetric list, future loops).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iterator>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../system/gpu_particles_arena.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"

#include "sfc_tiles.h"
#include "sfc_tiles_functions.h"
#include "gpu_neighbor_list.h"
#include "gpu_dirty_tracker.h"
#include "neighbor_list.h"
#include "ghost_writeback.h"  /* ghost_write_detector_resnapshot_after_lazy_drift */

/* TILE_PERIODIC_X/Y/Z defined in sfc_tiles.h (included via gpu_neighbor_list.h) */

/* Persistent gas-only SIDX shared across density+symlist within a step.
 * Lifetime: built lazily on first gas (type_bitmask=1) ngb_list_build, reused
 * for all subsequent gas builds, freed by gpu_step_sidx_invalidate() after drift. */
static gpu_spatial_index_t g_step_sidx{};
static gpu_spatial_index_t g_step_sidx_alltypes{};

/* Lazy-drift h-slack: under lazy drift, neighbor j's KernelRadius in P[] may
 * be at j's old Ti_current, not at time1. drift_particle's gas-extras block
 * grows h via exp(divv_fac/NDIMS) per drift (capped at exp(divv_fac_max=±0.3
 * /NDIMS) ≈ ±10% per drift), accumulating over multiple deferred drifts.
 * compact_xyzh[j*4+3] inherits the same staleness when populated from P[].
 *
 * The BVH walk reads compact_xyzh[j*4+3] for tile-overlap decisions in
 * symmetric mode. Stale-small h_j → BVH overlap test underestimates →
 * legitimate neighbors missed.
 *
 * Mitigation: at compact_xyzh population time, multiply h by (1 + slack) so
 * the BVH over-includes tiles, absorbing accumulated h-growth from
 * undrifted particles. Per-pair r² acceptance inside kernels reads the
 * REAL P[j].KernelRadius (UVM) — over-inclusion is wasted work, never a
 * silent miss. 0.5 (50% inflation) covers the per-decomp-interval h-growth
 * with margin. */
static constexpr double SIDX_H_SLACK = 0.5;

gpu_spatial_index_t *gpu_step_sidx_ptr(void) { return &g_step_sidx; }
gpu_spatial_index_t *gpu_step_sidx_alltypes_ptr(void) { return &g_step_sidx_alltypes; }


/* Dirty-index tracking for compact_xyzh.h field.
 *
 * Pre-tracker: a single global g_dirty_list/g_dirty_all pair was shared by
 * both g_step_sidx (gas-only) and g_step_sidx_alltypes. Both caches persist
 * across ghost-import-only changes, so one cache consuming and clearing the
 * shared global state would silently leave the other stale -- a physics-
 * correctness hole. Now: per-cache state via gpu_dirty_tracker.
 *
 * Caches register their dense particle-index range [base, base+count) on
 * build, unregister on free. Marks route to ALL caches whose range covers
 * the j (each cache has its own bitset). Refresh consumes only its own
 * cache's bitset.
 *
 * mark_h_dirty_all preserves global semantics: it sets all_dirty on every
 * registered cache (matching the old "unknown-scope mutation"). Per-cache
 * promote-to-all still fires when one cache's popcount exceeds threshold
 * inside the tracker. */

void gpu_compact_xyzh_mark_h_dirty_all(void)
{
    gpu_dirty_tracker_mark_all_global();
}

void gpu_compact_xyzh_mark_h_dirty_idx(int i)
{
    if(i < 0) return;
    int idx_arr[1] = { i };
    gpu_dirty_tracker_mark_indices(idx_arr, 1);
}

void gpu_compact_xyzh_mark_h_dirty_range(int start, int end)
{
    gpu_dirty_tracker_mark_range(start, end);
}

void gpu_compact_xyzh_mark_h_dirty_indices(const int *indices, int n)
{
    gpu_dirty_tracker_mark_indices(indices, n);
}

/* Backwards-compat: any caller that doesn't know which indices it dirtied
 * conservatively forces a full-pool refresh on every cache. */
void gpu_compact_xyzh_mark_h_dirty(void) { gpu_compact_xyzh_mark_h_dirty_all(); }

/* SSOT mark helpers — see header for design.  Route to BOTH the GPU SIDX
 * dirty tracker AND the host glt cache dirty tracker. Adding a further cache
 * later requires only register/unregister inside that cache's lifetime —
 * these helpers automatically include it. */
void gizmo_mark_kernel_radius_dirty_indices(const int *indices, int n)
{
    if(!indices || n <= 0) return;
    gpu_dirty_tracker_mark_indices(indices, n);
    ghost_exchange_local_tree_mark_h_dirty_indices(indices, n);
}
void gizmo_mark_kernel_radius_dirty_range(int start, int end)
{
    if(end <= start) return;
    gpu_dirty_tracker_mark_range(start, end);
    ghost_exchange_local_tree_mark_h_dirty_range(start, end);
}

/* SIDX lifecycle epoch counters, bumped by the notify hooks below on every
 * ghost import, ghost cleanup and pool membership change. gpu_ngb_list_build
 * stamps them into each index at build and requires them to still match before
 * reusing it, so a cached index cannot survive a change of ghost or pool
 * identity that happens to preserve the particle count. */
static uint64_t g_sidx_ghost_epoch = 0;
static uint64_t g_sidx_pool_epoch  = 0;

void gpu_sidx_notify_ghost_imported(int start, int count)
{
    /* Contract requires unconditional call on every rank, including count==0. */
    (void)start; (void)count;
    g_sidx_ghost_epoch++;
}

void gpu_sidx_notify_ghost_cleanup(void)
{
    /* Called BEFORE NumPart shrinks. */
    g_sidx_ghost_epoch++;
}

void gpu_sidx_notify_pool_changed(void)
{
    /* Type/Mass/membership change in the home pool — invalidate any pool-
     * dependent cache on next access. */
    g_sidx_pool_epoch++;
}


/* Drift-time SIDX refresh (incremental rebuild).
 *
 * The unconditional full rebuild post-drift in the prior code cost
 * ~1.3s/step on fire_m11i tiny-N (the dominant tiny-N bucket post-
 * UVM-canonical). Of that, ~1s is build_sfc_tiles streaming 12.4M
 * particles to re-derive pool membership and re-tile it — work that's
 * only needed when particle layout actually changes (i.e.,
 * domain_decomp). There is no sort to skip: P[] arrives Peano-Hilbert
 * ordered from the domain decomposition and tiles are consecutive runs
 * of it, so the cost is the passes over P[], not any ordering step.
 *
 * Between domain decomps, particles drift but their pool/tile
 * assignments are still meaningful: each tile still references the
 * same particles. Their positions just changed slightly. So the
 * refresh path is:
 *
 *   1. For each tile, recompute lo/hi/hmax from the current particle
 *      positions in its pool slice (host OMP — random P[].Pos reads
 *      go to host memory directly under UVM-canonical, no device
 *      page-fault storm).
 *   2. Re-fit the BVH from the updated tile bboxes via build_tile_bvh
 *      (CPU, O(ntiles) ≈ ms for ~70k tiles; structurally identical
 *      to the original BVH because ntiles and tile order are unchanged).
 *   3. Stage updated tiles + BVH to device (deep_copy, ms-scale).
 *   4. Refresh compact_xyzh[i*4+0..2] device-side (existing parallel_for,
 *      ms-scale). NOTE: drift_particle DOES change KernelRadius (predict.cc
 *      lines ~160 and ~229: P[i].KernelRadius *= exp(divv_fac/NUMDIMS)).
 *      The drift-time refresh here updates positions only; the h component
 *      is refreshed separately via mark_h_dirty machinery, which the lazy-
 *      drift path inside gpu_ngb_list_build populates with the j's it
 *      drifted (so compact_xyzh[j*4+3] gets re-read from P_shared on the
 *      next build).
 *
 * Correctness invariant: each tile's bbox covers all current positions
 * of the particles in its pool. Tile assignments are frozen, so as
 * particles "wander" spatially they accumulate into multiple tiles'
 * bbox regions — BVH queries may visit a couple extra tiles per query
 * (inefficiency, never a missed neighbor; each tile still iterates its
 * pool, particles' actual positions are checked).
 *
 * Reset boundary: gpu_step_sidx_invalidate_full() at every domain_decomp
 * frees the SIDX, forcing a fresh rebuild of pool and tiles. Domain
 * decomp is already a heavy step, so the marginal cost is small. This
 * naturally bounds bbox dispersion within a decomp interval and resets
 * tile assignments to current spatial layout.
 *
 */

/* Recomputes tile bboxes from each pool member's "virtual at-time1 position"
 * AND fills the host-side position-staging buffer (idx->h_pos_buf) with the
 * same values.
 *
 * "Virtual at-time1 position" = P[j].Pos + P[j].Vel * get_drift_factor(
 *     P[j].Ti_current, All.Ti_Current, j, 0). This matches what
 * drift_particle's Pos update would produce IF / WHEN the particle is
 * lazily drifted by a downstream consumer. Under the current full-drift
 * regime (move_particles iterates every NumPart particle), Ti_current ==
 * All.Ti_Current for all j, dt = 0, virt_pos == P[j].Pos — i.e. this code
 * is a no-op in absolute value, just exercising the threadsafe drift-factor
 * code path so it's already wired for a future active-only iteration mode
 * in move_particles.
 *
 * Correctness invariant: the bbox covers each particle's actual location
 * at time1, regardless of whether the particle has been drifted yet. BVH
 * queries against the bbox find every potentially-relevant pool member;
 * the consumer then drifts the particle on first read. */
static void sidx_refresh_tile_bboxes_host(gpu_spatial_index_t *idx,
                                             struct particle_data *P_shared)
{
    sfc_tile_t *h_tiles = idx->h_tiles;
    int *h_pool = idx->h_pool;
    int ntiles = idx->ntiles;
    double *pos_buf = idx->h_pos_buf;
    /* SSOT per-particle reach under the cached policy (LEGACY for non-runner
     * callers → byte-equivalent to legacy P[j].KernelRadius aggregation). */
    const mode_b_radius_policy_t policy_capture = idx->cache_radius_policy;
    /* Out-of-line host accessor for the host-side drift-factor input. */
    integertime time1 = gizmo_host_ti_current();

    #pragma omp parallel for schedule(static)
    for(int t = 0; t < ntiles; t++) {
        sfc_tile_t *tile = &h_tiles[t];
        if(tile->count <= 0) continue;
        int j0 = h_pool[tile->first];
        double dt0 = get_drift_factor_omp_safe(P_shared[j0].Ti_current, time1, j0, 0);
        double x0 = P_shared[j0].Pos[0] + P_shared[j0].Vel[0] * dt0;
        double y0 = P_shared[j0].Pos[1] + P_shared[j0].Vel[1] * dt0;
        double z0 = P_shared[j0].Pos[2] + P_shared[j0].Vel[2] * dt0;
        double lo0 = x0, hi0 = x0;
        double lo1 = y0, hi1 = y0;
        double lo2 = z0, hi2 = z0;
        double hmax = nlr_particle_symmetric_radius(P_shared[j0], policy_capture);
        /* Per-type bands recomputed alongside scalar hmax: under the new
         * invariant the bands are policy-aware and would otherwise stay
         * frozen at build-time values, contradicting the conservative-
         * upper-bound rule.  Restart from 0 and aggregate from current
         * particles. */
        double hbt[TILE_NUM_PTYPES] = {0};
        {
            int t0 = (int)P_shared[j0].Type;
            if(t0 >= 0 && t0 < TILE_NUM_PTYPES && hmax > hbt[t0]) hbt[t0] = hmax;
        }
        if(pos_buf) { pos_buf[j0*3+0] = x0; pos_buf[j0*3+1] = y0; pos_buf[j0*3+2] = z0; }
        for(int s = 1; s < tile->count; s++) {
            int j = h_pool[tile->first + s];
            double dt = get_drift_factor_omp_safe(P_shared[j].Ti_current, time1, j, 0);
            double x = P_shared[j].Pos[0] + P_shared[j].Vel[0] * dt;
            double y = P_shared[j].Pos[1] + P_shared[j].Vel[1] * dt;
            double z = P_shared[j].Pos[2] + P_shared[j].Vel[2] * dt;
            if(x < lo0) lo0 = x; else if(x > hi0) hi0 = x;
            if(y < lo1) lo1 = y; else if(y > hi1) hi1 = y;
            if(z < lo2) lo2 = z; else if(z > hi2) hi2 = z;
            double h = nlr_particle_symmetric_radius(P_shared[j], policy_capture);
            if(h > hmax) hmax = h;
            int tj = (int)P_shared[j].Type;
            if(tj >= 0 && tj < TILE_NUM_PTYPES && h > hbt[tj]) hbt[tj] = h;
            if(pos_buf) { pos_buf[j*3+0] = x; pos_buf[j*3+1] = y; pos_buf[j*3+2] = z; }
        }
        tile->lo[0] = lo0; tile->hi[0] = hi0;
        tile->lo[1] = lo1; tile->hi[1] = hi1;
        tile->lo[2] = lo2; tile->hi[2] = hi2;
        tile->hmax = hmax;
        for(int tt = 0; tt < TILE_NUM_PTYPES; tt++) tile->hmax_by_type[tt] = hbt[tt];

    }
}

/* Refresh compact_xyzh positions on device via host-staged buffer.
 *
 * Avoids the UVM-fault-storm cost of a parallel_for that reads P_shared.Pos
 * directly on device (which costs ~1.25s/step on fire_m11i 12.4M after host
 * drift just wrote those pages — every page faults migration to GPU on first
 * access). Instead: positions were already filled into idx->h_pos_buf by the
 * host bbox-recompute loop; bulk deep_copy to d_pos_buf (~200MB at NVLink
 * ~600GB/s = sub-ms), then a small device kernel scatters into the
 * interleaved d_compact_xyzh array. h field intentionally untouched here —
 * NOTE: drift_particle DOES change KernelRadius (predict.cc:160,229), but
 * THIS function is the position-only fast path used at drift-time; the h
 * component is refreshed via the mark_h_dirty machinery on the next
 * gpu_ngb_list_build, which consumes the dirty list populated by every
 * h-writer (lazy drift, density iter, etc.). Splitting pos and h refresh
 * lets us amortize the position update across the whole step while only
 * the touched h slots get refreshed per-build. */
static void sidx_refresh_compact_positions_device(gpu_spatial_index_t *idx)
{
    int num_total = idx->num_total;
    if(!idx->h_pos_buf || !idx->d_pos_buf) return;
    /* Bulk host->device copy of the staged position buffer. Single linear
     * cudaMemcpy under the hood; pages are migrated as one transfer rather
     * than fault-by-fault on first device read. */
    using UV = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<double*, Kokkos::HostSpace, UV>            h_v(idx->h_pos_buf, 3 * num_total);
    Kokkos::View<double*, GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_v(idx->d_pos_buf, 3 * num_total);
    Kokkos::deep_copy(d_v, h_v);

    double *compact = idx->d_compact_xyzh;
    double *pos_buf = idx->d_pos_buf;
    Kokkos::parallel_for("compact_xyzh_pos_scatter", num_total, KOKKOS_LAMBDA(int i) {
        compact[i*4+0] = pos_buf[i*3+0];
        compact[i*4+1] = pos_buf[i*3+1];
        compact[i*4+2] = pos_buf[i*3+2];
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("compact_xyzh_pos_scatter", num_total);
}

/* Re-fit BVH from updated tile bboxes (structural rebuild — left/right links
 * are recomputed identically since tile order is unchanged, but bboxes/hmax
 * propagate from the refreshed tiles). Calls build_tile_bvh (which mymalloc's
 * a fresh BVH), copies into the persistent HostSpace h_bvh buffer, then frees
 * the mymalloc'd transient. The HostSpace buffer was allocated to size
 * 2*ntiles-1 at build time; ntiles is unchanged across drifts, so the buffer
 * always has room. */
static void sidx_rebuild_bvh_inplace(gpu_spatial_index_t *idx)
{
    tile_bvh_node_t *h_bvh_tmp = NULL;
    int new_nnodes = build_tile_bvh(idx->h_tiles, idx->ntiles, &h_bvh_tmp);
    if(new_nnodes > 0 && h_bvh_tmp && idx->h_bvh) {
        memcpy(idx->h_bvh, h_bvh_tmp, new_nnodes * sizeof(tile_bvh_node_t));
    }
    if(h_bvh_tmp) myfree(h_bvh_tmp);
    idx->h_bvh_nnodes = new_nnodes;
    idx->bvh_root = new_nnodes - 1;
}

/* Stage updated host h_tiles + h_bvh to device d_tiles + d_bvh.
 * h_pool / d_pool unchanged across drifts (tile assignments frozen). */
static void sidx_stage_to_device(gpu_spatial_index_t *idx)
{
    using UV = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<sfc_tile_t*,      Kokkos::HostSpace, UV>            h_tiles_v(idx->h_tiles, idx->ntiles);
    Kokkos::View<sfc_tile_t*,      GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_tiles_v(idx->d_tiles, idx->ntiles);
    Kokkos::View<tile_bvh_node_t*, Kokkos::HostSpace, UV>            h_bvh_v(idx->h_bvh, idx->h_bvh_nnodes);
    Kokkos::View<tile_bvh_node_t*, GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_bvh_v(idx->d_bvh, idx->h_bvh_nnodes);
    Kokkos::deep_copy(d_tiles_v, h_tiles_v);
    Kokkos::deep_copy(d_bvh_v,   h_bvh_v);
}

/* Forward decl */
void gpu_step_sidx_invalidate_full(void);

/* Drift-time refresh: reuse pool/tile membership, recompute bboxes/BVH, refresh compact_xyzh. */
static void sidx_refresh_after_drift(gpu_spatial_index_t *idx,
                                      struct particle_data *P_shared)
{
    sidx_refresh_tile_bboxes_host(idx, P_shared);
    sidx_rebuild_bvh_inplace(idx);
    sidx_stage_to_device(idx);
    sidx_refresh_compact_positions_device(idx);
}

void gpu_step_sidx_invalidate(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    struct particle_data *P_shared = gpu_particles_arena_P();
    if(!P_shared) {
        /* No arena -> no canonical particle storage; fall back to full free. */
        gpu_step_sidx_invalidate_full();
        return;
    }

    /* Current scope: refresh the gas SIDX (the hot-path 1.3s/step bucket
     * from density_sidx_prebuild); alltypes goes through full-free since it
     * only builds on sink-active steps and isn't on the dominant tiny-N
     * path. The refresh path could be extended to alltypes once gas is
     * validated. */
    if(g_step_sidx_alltypes.valid) gpu_spatial_index_free(&g_step_sidx_alltypes);

    gpu_spatial_index_t *idx = &g_step_sidx;
    if(idx->valid) {
        if(!idx->h_tiles || !idx->h_pool || !idx->d_compact_xyzh || idx->ntiles <= 0) {
            /* Defensive: incomplete state -> free, fall back to full rebuild. */
            gpu_spatial_index_free(idx);
            gpu_compact_xyzh_mark_h_dirty_all();
        } else {
            sidx_refresh_after_drift(idx, P_shared);
            /* h-dirty state intentionally left intact. drift_particle DOES
             * change KernelRadius (predict.cc:160,229) — those h updates are
             * marked into the per-cache dirty tracker (gpu_dirty_tracker) by
             * the lazy-drift loop in the previous step's gpu_ngb_list_build,
             * by move_particles/gizmo_full_drift_to (predict.cc:307,351), and by other
             * h-writers like density iter. The next gpu_ngb_list_build
             * consume()s this cache's bits and refreshes compact_xyzh[*4+3]
             * from current P_shared.KernelRadius before walking. */
        }
    }

}

void gpu_step_sidx_invalidate_full(void)
{
    if(g_step_sidx_alltypes.valid) gpu_spatial_index_free(&g_step_sidx_alltypes);
    if(g_step_sidx.valid) gpu_spatial_index_free(&g_step_sidx);
    /* Force full-mode dirty so next build's compact_xyzh seeds correctly. */
    gpu_compact_xyzh_mark_h_dirty_all();
}


void gpu_spatial_index_build(struct particle_data *P_shared, int num_total,
                             int type_bitmask, gpu_spatial_index_t *idx,
                             const char *caller_label,
                             mode_b_radius_policy_t radius_policy)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Capture periodicity parameters */
    idx->periodic_flags[0] = TILE_PERIODIC_X;
    idx->periodic_flags[1] = TILE_PERIODIC_Y;
    idx->periodic_flags[2] = TILE_PERIODIC_Z;
    idx->box_sizes[0] = boxSize_X; idx->box_sizes[1] = boxSize_Y; idx->box_sizes[2] = boxSize_Z;
    idx->box_halves[0] = boxHalf_X; idx->box_halves[1] = boxHalf_Y; idx->box_halves[2] = boxHalf_Z;
    /* Hard guard: bbox_overlaps_sphere_gpu's periodic-wrap math goes pathological
     * (every node "overlaps" every query, BVH degenerates to exhaustive scan)
     * if box_sizes[k] is 0 while periodic_flags[k] is on. Fail loud at the
     * call site so any regression in the per-TU AllDeviceMirror sync can never
     * silently corrupt performance again. */
    for(int k = 0; k < 3; k++) {
        if(idx->periodic_flags[k] && !(idx->box_sizes[k] > 0.0)) {
            printf("gpu_spatial_index_build: periodic_flags[%d]=1 but box_sizes[%d]=%g (caller='%s'). "
                   "Likely cause: this TU's AllDeviceMirror not synced from host All. "
                   "Confirm gizmo_gpu_sync_all() has run for this timestep.\n",
                   k, k, idx->box_sizes[k], caller_label ? caller_label : "?");
            fflush(stdout);
            endrun(913004);
        }
    }

    /* Build SFC tiles + BVH on CPU */
    sfc_tile_t *h_tiles;
    int *h_pool;
    int num_pool;
    int ntiles = build_sfc_tiles(P_shared, num_total, type_bitmask, TILE_TARGET_SIZE,
                                 &h_tiles, &h_pool, &num_pool, radius_policy);
    idx->ntiles = ntiles;

    tile_bvh_node_t *h_bvh;
    int bvh_nnodes = build_tile_bvh(h_tiles, ntiles, &h_bvh);
    idx->bvh_root = bvh_nnodes - 1;

    /* Allocate kernel-read-path arrays in DEVICE_SPACE (CudaSpace HBM on GPU
     * builds, falls back to SharedSpace elsewhere).  This eliminates HMM/TLB-
     * miss overhead on the small-N kernel hot path where one thread does
     * ~1000s of scattered reads through bvh/tiles/pool/compact_xyzh — that
     * scattered-UVM-access pattern is the suspected source of the residual
     * 1.4s "fused_fnc" floor on 1-active-particle calls.  CPU host arrays
     * h_tiles/h_bvh/h_pool are transferred via Kokkos::deep_copy through
     * unmanaged-View wrappers (cudaMemcpy under the hood on CUDA builds);
     * compact_xyzh is built directly on-device by a parallel_for that reads
     * from UVM-backed P_shared and writes to DEVICE_SPACE compact_xyzh. */
    int bvh_size = (2 * ntiles - 1);
    if(bvh_size < 1) bvh_size = 1;
    int pool_size = (num_pool > 0) ? num_pool : 1;
    idx->d_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_dev_tiles", ntiles * sizeof(sfc_tile_t));
    idx->d_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_dev_bvh", bvh_size * sizeof(tile_bvh_node_t));
    idx->d_pool = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_dev_pool", pool_size * sizeof(int));

    /* Stage host buffers into device memory.  On non-CUDA builds DEVICE_SPACE
     * == SharedSpace and Kokkos::deep_copy reduces to a memcpy. */
    {
        using UV = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        Kokkos::View<sfc_tile_t*,        Kokkos::HostSpace, UV>            h_tiles_v(h_tiles, ntiles);
        Kokkos::View<sfc_tile_t*,        GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_tiles_v(idx->d_tiles, ntiles);
        Kokkos::View<tile_bvh_node_t*,   Kokkos::HostSpace, UV>            h_bvh_v(h_bvh, bvh_nnodes);
        Kokkos::View<tile_bvh_node_t*,   GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_bvh_v(idx->d_bvh, bvh_nnodes);
        Kokkos::View<int*,               Kokkos::HostSpace, UV>            h_pool_v(h_pool, num_pool);
        Kokkos::View<int*,               GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_pool_v(idx->d_pool, num_pool);
        Kokkos::deep_copy(d_tiles_v, h_tiles_v);
        Kokkos::deep_copy(d_bvh_v,   h_bvh_v);
        Kokkos::deep_copy(d_pool_v,  h_pool_v);
    }

    /* Build compact double4 position+h array for cache-efficient GPU BVH traversal.
       DOUBLE positions: float ABSOLUTE positions are invalid for GIZMO's ~1e11
       dynamic range (see §37/§38) — they must NOT decide neighbour inclusion.
       ~64MB for 2M particles vs 800MB for full P_shared. h (slot 3) is a relative
       reach; kept double so the leaf accept stays consistent with the double opener.
       In DEVICE_SPACE so the BVH-walk kernels read from HBM directly. */
    idx->d_compact_xyzh = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_dev_compact_xyzh", num_total * 4 * sizeof(double));
    {
        double *compact = idx->d_compact_xyzh;
        double h_inflate = 1.0 + SIDX_H_SLACK; /* lazy-drift slack: see SIDX_H_SLACK comment */
        const mode_b_radius_policy_t policy_capture = radius_policy;
        Kokkos::parallel_for("compact_xyzh_build", num_total, KOKKOS_LAMBDA(int i) {
            compact[i*4+0] = P_shared[i].Pos[0];
            compact[i*4+1] = P_shared[i].Pos[1];
            compact[i*4+2] = P_shared[i].Pos[2];
            /* SSOT per-particle reach under policy_capture — see nlr_radius_policy.h.
             * LEGACY default policy recovers raw P[j].KernelRadius for every type. */
            double h_j = nlr_particle_symmetric_radius(P_shared[i], policy_capture);
            compact[i*4+3] = h_j * h_inflate;
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("compact_xyzh_build", num_total);
    }

    /* Keep host-side persistent copies of tiles/pool/BVH alive across drifts
     * so the incremental refresh path (gpu_step_sidx_invalidate ->
     * sidx_refresh_after_drift) can recompute tile bboxes from current
     * particle positions on host without re-deriving pool membership or
     * re-tiling it.
     *
     * MUST use Kokkos::HostSpace allocator (heap-based) NOT mymalloc — the
     * latter is LIFO-stack-disciplined and persistent SIDX buffers would
     * sit on top of any subsequent transient allocation (e.g. density's
     * per-step mymalloc'd Left/Right arrays), preventing those transients
     * from being freed in LIFO order. The transient mymalloc'd h_tiles /
     * h_pool / h_bvh from build_sfc_tiles + build_tile_bvh are copied
     * out then myfree'd in proper LIFO order below. */
    idx->h_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<Kokkos::HostSpace>("ngl_host_tiles", ntiles * sizeof(sfc_tile_t));
    memcpy(idx->h_tiles, h_tiles, ntiles * sizeof(sfc_tile_t));
    idx->h_pool = (int *) Kokkos::kokkos_malloc<Kokkos::HostSpace>("ngl_host_pool", pool_size * sizeof(int));
    memcpy(idx->h_pool, h_pool, num_pool * sizeof(int));
    idx->h_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<Kokkos::HostSpace>("ngl_host_bvh", bvh_size * sizeof(tile_bvh_node_t));
    memcpy(idx->h_bvh, h_bvh, bvh_nnodes * sizeof(tile_bvh_node_t));
    idx->h_bvh_nnodes = bvh_nnodes;
    idx->num_pool = num_pool;

    /* Drift-refresh staging buffers: h_pos_buf is filled per-pool-member by the
     * host bbox-recompute loop; bulk deep_copy to d_pos_buf; device scatter
     * into d_compact_xyzh. Sized 3*num_total doubles so each pool index can
     * write directly to h_pos_buf[j*3+0..2] without remapping. Non-pool
     * entries stay uninitialized in h_pos_buf and their stale d_compact_xyzh
     * positions are never read by BVH queries (BVH only visits tiles, tiles
     * only contain pool members). */
    int pos_buf_count = (num_total > 0 ? num_total : 1);
    idx->h_pos_buf = (double *) Kokkos::kokkos_malloc<Kokkos::HostSpace>("ngl_host_pos_buf", 3 * pos_buf_count * sizeof(double));
    idx->d_pos_buf = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_dev_pos_buf", 3 * pos_buf_count * sizeof(double));


    /* Free the transient mymalloc'd build buffers in proper LIFO order
     * (build_tile_bvh allocated h_bvh last; build_sfc_tiles allocated
     * h_pool then h_tiles). */
    myfree(h_bvh);
    myfree(h_tiles);
    myfree(h_pool);

    idx->num_total = num_total;
    idx->cache_tbm = type_bitmask;
    idx->cache_radius_policy = radius_policy;
    idx->ghost_epoch_when_built = g_sidx_ghost_epoch;
    idx->pool_epoch_when_built  = g_sidx_pool_epoch;
    idx->valid = 1;
    /* Register this cache with the dirty tracker over [0, num_total). The
     * compact_xyzh build above wrote every row's h from the live P[] under this
     * cache's radius policy, and nothing between there and here can mutate it,
     * so the range starts clean: the first refresh would recompute values it
     * already holds, over the whole pool. */
    if(idx->dirty_handle >= 0) gpu_dirty_tracker_unregister(idx->dirty_handle);
    idx->dirty_handle = gpu_dirty_tracker_register(0, num_total, 1);

}

void gpu_spatial_index_free(gpu_spatial_index_t *idx)
{
    if(idx->d_compact_xyzh) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_compact_xyzh); idx->d_compact_xyzh = NULL;}
    if(idx->d_pool) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_pool); idx->d_pool = NULL;}
    if(idx->d_bvh) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_bvh); idx->d_bvh = NULL;}
    if(idx->d_tiles) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_tiles); idx->d_tiles = NULL;}
    /* Free host-side persistent buffers kept alive across drifts.
     * mymalloc uses LIFO stack discipline; free in reverse-allocation order:
     * h_pool -> h_tiles -> h_bvh
     * (h_bvh allocated first by build_tile_bvh, but build_tile_bvh may have
     * been re-called via sidx_rebuild_bvh_inplace which freed-then-allocated,
     * so h_bvh is on top of the stack at this point in normal flow). */
    /* Persistent host-side buffers live in Kokkos::HostSpace (heap-allocated,
     * not mymalloc) so they don't pin the LIFO stack across other transient
     * mymalloc'd state. Free order doesn't matter. */
    if(idx->h_bvh)   { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_bvh);   idx->h_bvh = NULL; }
    if(idx->h_tiles) { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_tiles); idx->h_tiles = NULL; }
    if(idx->h_pool)  { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_pool);  idx->h_pool = NULL; }
    if(idx->h_pos_buf) { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_pos_buf); idx->h_pos_buf = NULL; }
    if(idx->d_pos_buf) { Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_pos_buf); idx->d_pos_buf = NULL; }
    idx->h_bvh_nnodes = 0;
    idx->num_pool = 0;
    idx->num_total = 0;
    idx->cache_tbm = -1;
    idx->cache_radius_policy = MODE_B_RADIUS_DEFAULT;
    idx->valid = 0;
    if(idx->dirty_handle >= 0) {
        gpu_dirty_tracker_unregister(idx->dirty_handle);
        idx->dirty_handle = -1;
    }
}



/* ---- L4 Step-1a device neighbour-inclusion precision oracle (DIAGNOSTIC) ----
 * Gate GIZMO_NGB_PRECISION_ORACLE (SPIKE/test only; teardown ledger). Quantifies
 * device neighbour under/over-inclusion caused by the single-precision ABSOLUTE
 * compact positions (invalid for GIZMO's ~1e11 dynamic range) against a
 * double-precision physical-truth reference. NOT production; the real fix is the
 * Step-1b compact-DOUBLE device arrays. Report tag [NGL_PRECISION_ORACLE ...]. */

/* SSOT predicted-double side buffers — built in ONE place, consumed by BOTH the
 * host double-truth loop AND the device double-leaf kernel (no second prediction
 * path). For each particle j:
 *   position = P[j].Pos + P[j].Vel * drift_factor(Ti_current -> time1)  (predicted-at-time1,
 *              matching the SIDX bbox/compact refresh convention above)
 *   reach    = nlr_particle_symmetric_radius(P[j], policy)              (PHYSICAL, no SIDX slack)
 * Buffers are SHARED_SPACE (managed) so host + device read the same memory. */
static void ngl_precision_build_ref_buffers(struct particle_data *P_shared, int num_total,
                                            mode_b_radius_policy_t radius_policy,
                                            double **out_pos_dbl, double **out_h_dbl,
                                            long *out_n_dt_nonzero)
{
    integertime time1 = gizmo_host_ti_current();
    size_t nt = (size_t)(num_total > 0 ? num_total : 1);
    double *pos_dbl = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_tilebuild_pos", nt * 3 * sizeof(double));
    double *h_dbl   = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_tilebuild_h", nt * sizeof(double));
    long n_dt_nonzero = 0;
    #pragma omp parallel for reduction(+:n_dt_nonzero) schedule(static)
    for(int j = 0; j < num_total; j++) {
        double dt = get_drift_factor_omp_safe(P_shared[j].Ti_current, time1, j, 0);
        if(dt != 0.0) n_dt_nonzero++;
        pos_dbl[j*3+0] = P_shared[j].Pos[0] + P_shared[j].Vel[0] * dt;
        pos_dbl[j*3+1] = P_shared[j].Pos[1] + P_shared[j].Vel[1] * dt;
        pos_dbl[j*3+2] = P_shared[j].Pos[2] + P_shared[j].Vel[2] * dt;
        h_dbl[j]       = nlr_particle_symmetric_radius(P_shared[j], radius_policy);
    }
    *out_pos_dbl = pos_dbl;
    *out_h_dbl   = h_dbl;
    *out_n_dt_nonzero = n_dt_nonzero;
}


void gpu_ngb_list_build(struct particle_data *P_shared, int num_total,
                        int *active_indices_host, int num_active,
                        int search_mode, int type_bitmask,
                        gpu_neighbor_list_t *gnl,
                        gpu_spatial_index_t *cached_idx,
                        double search_radius_factor,
                        const double *search_radii_host,
                        const double *source_positions_host,
                        const char *caller_label,
                        double j_kernel_radius_scale,
                        mode_b_radius_policy_t radius_policy)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    gnl->num_active = num_active;
    double t_entry = my_second(); /* DIAG: entry */
    const double cpu_rows_child0 = CPU_ChildCharged;
    /* Defensive guard: a cached SIDX's compact_xyzh / pool only contains the
     * originally-built types. If the caller's type_bitmask differs from the
     * cache's, the walker would return neighbors of types outside the
     * caller's mask (e.g., DM neighbors leaking into a gas-only density
     * walk → lazy-drift attempts on Type=1 → drift_particle abort
     * 'no prediction into past allowed'). HARD-ABORT with a clear message
     * so any future Spec-author who routes a tbm-mismatched cache fails at
     * the right layer, not via downstream nonsense. The companion warning
     * in gpu_neighbor_list.h on gpu_step_sidx_alltypes_ptr documents this
     * invariant. */
    if (cached_idx && cached_idx->valid && cached_idx->cache_tbm >= 0 &&
        cached_idx->cache_tbm != type_bitmask) {
        fprintf(stderr,
            "gpu_ngb_list_build FATAL: caller='%s' type_bitmask=0x%x but cached "
            "SIDX was built with tbm=0x%x. The cached compact_xyzh / pool only "
            "contains the originally-built types — walking it under a different "
            "mask returns wrong-type neighbors and triggers downstream aborts "
            "(e.g. drift_particle 'no prediction into past allowed' on Type=1 "
            "during a gas-only density walk). Spec author: declare "
            "sidx_cache_kind matching your neighbor_type_mask, or rebuild "
            "the cache. See gpu_neighbor_list.h docstring on "
            "gpu_step_sidx_alltypes_ptr.\n",
            caller_label ? caller_label : "?", type_bitmask, cached_idx->cache_tbm);
        fflush(stderr);
        endrun(913005);
    }
    /* Companion HARD-ABORT for radius_policy mismatch.  The
     * cached compact_xyzh[j*4+3] reflects the build-time policy's per-j reach;
     * walking it under a different Spec policy would silently use the wrong
     * leaf-side h_j and miss valid pairs.  Same shape as the cache_tbm gate. */
    if (cached_idx && cached_idx->valid &&
        cached_idx->cache_radius_policy != radius_policy) {
        fprintf(stderr,
            "gpu_ngb_list_build FATAL: caller='%s' radius_policy=0x%x but cached "
            "SIDX was built with radius_policy=0x%x. The cached compact_xyzh[j*4+3] "
            "encodes the build-time policy's per-particle pair-search reach; "
            "walking it under a different policy returns wrong leaf-h_j values "
            "and silently misses valid pairs.  Spec author: ensure Spec::radius_policy "
            "matches the cache's owner (or declare SidxCacheKind::None to rebuild "
            "per call).\n",
            caller_label ? caller_label : "?",
            (unsigned)radius_policy, (unsigned)cached_idx->cache_radius_policy);
        fflush(stderr);
        endrun(913006);
    }
    /* Active-source-in-pool DIAGNOSTIC GUARD (env-gated GIZMO_NLR_DIAG>=1).
     * Complements the compile-time runner static_assert for DIRECT (non-runner)
     * callers: when a CACHED SIDX is used with source_positions_host==NULL OR
     * search_radii_host==NULL, the active source position/radius is read from
     * compact_xyzh[active_index], which is the gas/pool-only array and is stale
     * for a non-pool active on a reused cache. Warn (once per caller) if any
     * active type is outside type_bitmask. See the docstring. */
    {
        static const char *g_srcpool_env = getenv("GIZMO_NLR_DIAG");
        static const int srcpool_diag_on =
            (g_srcpool_env && g_srcpool_env[0] >= '1' && g_srcpool_env[0] <= '9') ? 1 : 0;
        if (srcpool_diag_on && cached_idx && cached_idx->valid &&
            (source_positions_host == NULL || search_radii_host == NULL)) {
            int n_bad = 0, first_bad = -1;
            for (int aa = 0; aa < num_active; aa++) {
                const int i = active_indices_host[aa];
                const int t = (int)P_shared[i].Type;
                /* range-check before the shift so malformed/direct-caller usage
                 * cannot trigger UB in this diagnostic path. */
                if (t < 0 || t >= 6 || !((1u << (unsigned)t) & (unsigned)type_bitmask)) {
                    n_bad++; if (first_bad < 0) first_bad = i;
                }
            }
            if (n_bad > 0) {
                /* First-warning cap: at most one warning per distinct caller label
                 * (string literals have stable addresses, so pointer identity works). */
                static const char *seen[32]; static int n_seen = 0; int already = 0;
                for (int s = 0; s < n_seen; s++) { if (seen[s] == caller_label) { already = 1; break; } }
                if (!already) {
                    if (n_seen < 32) seen[n_seen++] = caller_label;
                    const char *which_arg =
                        (source_positions_host == NULL && search_radii_host == NULL)
                            ? "source_positions_host and search_radii_host"
                            : (source_positions_host == NULL ? "source_positions_host" : "search_radii_host");
                    const char *which_fix =
                        (source_positions_host == NULL && search_radii_host == NULL)
                            ? "source positions and radii"
                            : (source_positions_host == NULL ? "source positions" : "search radii");
                    fprintf(stderr,
                        "[NLR DIAG WARN] gpu_ngb_list_build caller='%s': %d/%d active sources are NOT pool "
                        "members (type_bitmask=0x%x) but %s==NULL -> they read STALE compact_xyzh on a reused "
                        "cache. First: idx=%d Type=%d. Pass explicit %s (see gpu_neighbor_list.h).\n",
                        caller_label ? caller_label : "?", n_bad, num_active, type_bitmask, which_arg,
                        first_bad, first_bad >= 0 ? (int)P_shared[first_bad].Type : -1, which_fix);
                    fflush(stderr);
                }
            }
        }
    }

    /* Early-out: with no active particles there is nothing to search.
     * Skip the SIDX build/refresh AND all kernel launches.  Allocate 1-element
     * stubs so the caller's gpu_ngb_list_free path is well-defined (it always
     * frees neighbors/offsets/d_active).  Use cached SIDX pointers if available
     * so the free path's "is this from cache?" comparison still works. */
    if(num_active == 0) {
        gpu_spatial_index_t *idx_for_stubs = NULL;
        if(cached_idx && cached_idx->valid && cached_idx->num_total == num_total) {
            idx_for_stubs = cached_idx;
        }
        if(idx_for_stubs) {
            gnl->d_tiles  = idx_for_stubs->d_tiles;
            gnl->d_bvh    = idx_for_stubs->d_bvh;
            gnl->d_pool   = idx_for_stubs->d_pool;
            gnl->d_compact_xyzh = idx_for_stubs->d_compact_xyzh;
            gnl->ntiles   = idx_for_stubs->ntiles;
            gnl->bvh_root = idx_for_stubs->bvh_root;
            memcpy(gnl->periodic_flags, idx_for_stubs->periodic_flags, 3 * sizeof(int));
            memcpy(gnl->box_sizes,      idx_for_stubs->box_sizes,      3 * sizeof(double));
            memcpy(gnl->box_halves,     idx_for_stubs->box_halves,     3 * sizeof(double));
        } else {
            gnl->d_tiles = NULL; gnl->d_bvh = NULL; gnl->d_pool = NULL; gnl->d_compact_xyzh = NULL;
            gnl->ntiles = 0; gnl->bvh_root = 0;
            memset(gnl->periodic_flags, 0, 3*sizeof(int));
            memset(gnl->box_sizes,      0, 3*sizeof(double));
            memset(gnl->box_halves,     0, 3*sizeof(double));
        }
        gnl->d_active  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_active_stub", sizeof(int));
        gnl->offsets   = (int64_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_offsets_stub", sizeof(int64_t));
        gnl->offsets[0] = 0;
        gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_neighbors_stub", sizeof(int));
        gnl->total_pairs = 0;
        cpu_charge_child(CPU_NGB_BUILD, cpu_minus_children(timediff(t_entry, my_second()), cpu_rows_child0));
        return;
    }

    /* Use cached spatial index if available, otherwise build fresh.
     * If caller provided a cached_idx but it's not yet built, populate it (this
     * enables persistent caching across calls — caller controls invalidation). */
    gpu_spatial_index_t local_idx = {NULL, NULL, NULL, 0, 0, {0}, {0}, {0}, NULL, 0, 0};
    gpu_spatial_index_t *idx;
    /* Invalidate the cached SIDX unless it still describes the same particles.
     * num_total: the compact_xyzh and pool arrays were sized for the old count,
     * so accessing beyond them is UB (ghost exchange redo, particle creation).
     * Epochs: a cleanup-and-reimport can land the SAME ghost count with
     * different ghost contents, which no count test can see. The index would
     * then hold stale positions, tile bounds and BVH, because a ghost import
     * marks h-dirty only and the refresh kernels rewrite compact_xyzh[i*4+3]
     * alone. Drift does not bump either epoch -- membership is unchanged there,
     * and the drift refresh path handles moved positions -- so this costs no
     * rebuild on the common path. */
    if(cached_idx && cached_idx->valid &&
       (cached_idx->num_total          != num_total          ||
        cached_idx->ghost_epoch_when_built != g_sidx_ghost_epoch ||
        cached_idx->pool_epoch_when_built  != g_sidx_pool_epoch)) {
        gpu_spatial_index_free(cached_idx);
    }
    if(cached_idx && cached_idx->valid) {
        idx = cached_idx;
    } else if(cached_idx) {
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, cached_idx, caller_label, radius_policy);
        idx = cached_idx;
    } else {
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, &local_idx, caller_label, radius_policy);
        idx = &local_idx;
    }

    /* Copy spatial index pointers to gnl for use by free */
    gnl->d_tiles = idx->d_tiles;
    gnl->d_bvh = idx->d_bvh;
    gnl->d_pool = idx->d_pool;
    gnl->d_compact_xyzh = idx->d_compact_xyzh;
    gnl->ntiles = idx->ntiles;
    gnl->bvh_root = idx->bvh_root;
    memcpy(gnl->periodic_flags, idx->periodic_flags, 3 * sizeof(int));
    memcpy(gnl->box_sizes, idx->box_sizes, 3 * sizeof(double));
    memcpy(gnl->box_halves, idx->box_halves, 3 * sizeof(double));

    /* Refresh the h component of the compact array. Two modes (driven by the
     * per-cache gpu_dirty_tracker):
     *  - all-dirty: full-pool parallel_for(num_total, ...) — pays ~1.1-1.2s
     *    per call on fire_m11i 12.4M pool (UVM fault latency on P_shared
     *    KernelRadius reads).
     *  - bitset drain: parallel_for(n_dirty, ...) reading indices staged from
     *    this cache's bitset to a device buffer. ~ms per call when n_dirty is
     *    the active set. Per-cache state means consuming-and-clearing this
     *    cache's bits leaves the other registered caches' bitsets untouched.
     * Skipped entirely when this cache has neither all_dirty nor any set bits. */
    /* Per-cache dirty tracker query: this cache's bitset is independent of
     * other caches' state. consume() iterates set bits, populates d_dirty,
     * then clears bitset+all_dirty for THIS cache only. */
    int do_refresh = 0, refresh_all = 0;
    int handle = cached_idx ? cached_idx->dirty_handle : -1;
    if(cached_idx && cached_idx->valid && handle >= 0) {
        if(gpu_dirty_tracker_is_all_dirty(handle)) { do_refresh = 1; refresh_all = 1; }
        else if(gpu_dirty_tracker_popcount(handle) > 0) { do_refresh = 1; refresh_all = 0; }
    }
    if(do_refresh) {
        double *compact = idx->d_compact_xyzh;
        double h_inflate = 1.0 + SIDX_H_SLACK; /* see SIDX_H_SLACK — lazy-drift over-search slack */
        /* SSOT per-j reach under the cached policy.  HARD-ABORT above guarantees
         * cached_idx->cache_radius_policy == caller's radius_policy. */
        const mode_b_radius_policy_t policy_capture = idx->cache_radius_policy;
        if(refresh_all) {
            Kokkos::parallel_for("compact_h_refresh_all", num_total, KOKKOS_LAMBDA(int i) {
                double h_j = nlr_particle_symmetric_radius(P_shared[i], policy_capture);
                compact[i*4+3] = h_j * h_inflate;
            });
            /* Drain the bitset (no kernel use; just clear it). */
            struct {} dummy;
            gpu_dirty_tracker_consume(handle,
                [](int j, void *ud){ (void)j; (void)ud; },
                &dummy);
        } else {
            /* Drain bitset → host vector → device buffer → kernel. */
            std::vector<int> dirty_host;
            dirty_host.reserve(gpu_dirty_tracker_popcount(handle));
            gpu_dirty_tracker_consume(handle,
                [](int j, void *ud){ ((std::vector<int> *)ud)->push_back(j); },
                &dirty_host);
            int n_dirty = (int)dirty_host.size();
            int *d_dirty = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_dirty", n_dirty * sizeof(int));
            {
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    hv(dirty_host.data(), n_dirty);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    dv(d_dirty, n_dirty);
                Kokkos::deep_copy(dv, hv);
            }
            Kokkos::parallel_for("compact_h_refresh_idx", n_dirty, KOKKOS_LAMBDA(int k) {
                int i = d_dirty[k];
                double h_j = nlr_particle_symmetric_radius(P_shared[i], policy_capture);
                compact[i*4+3] = h_j * h_inflate;
            });
            Kokkos::fence();
            gizmo_gpu_check_last_error("compact_h_refresh_idx", n_dirty);
            Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_dirty);
        }
        Kokkos::fence();
        gizmo_gpu_check_last_error("compact_h_refresh", num_total);
    }

    /* Active indices: always re-uploaded (changes per call) */
    gnl->d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_active", ((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(gnl->d_active, active_indices_host, num_active * sizeof(int));

    /* Optional explicit per-active search radii (for loops with a different kernel
       than P[i].KernelRadius, e.g. KernelRadiusDM or AGS_Hsml). NULL → use P[i].KernelRadius. */
    double *d_radii = NULL;
    if(search_radii_host) {
        d_radii = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_radii", ((num_active > 0) ? num_active : 1) * sizeof(double));
        memcpy(d_radii, search_radii_host, num_active * sizeof(double));
    }

    /* Optional explicit per-active source positions (for sources not backed by
       P[] entries, e.g. arbitrary grid cells). NULL → read pos from P[active[aa]].
       Layout in caller's array: source_positions_host[aa*3 + k] for axis k. */
    double *d_source_pos = NULL;
    if(source_positions_host) {
        d_source_pos = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_source_pos", ((num_active > 0) ? num_active : 1) * 3 * sizeof(double));
        memcpy(d_source_pos, source_positions_host, num_active * 3 * sizeof(double));
    }

    /* Allocate CSR offsets (64-bit row pointers) */
    gnl->offsets = (int64_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("ngl_offsets", (size_t)(num_active + 1) * sizeof(int64_t));

    /* Per-particle scratchpad for fused single-pass build. Each active particle
     * gets a fixed stride (NGL_SCRATCH_STRIDE) of int slots in d_scratch; the BVH
     * walk emits j-indices directly there, with the count tracked in d_counts.
     * After scan + compact we transcribe into the dense CSR neighbors[] array.
     * Memory: stride * num_active * 4 bytes (e.g. 256 * 2M * 4 = 2GB). */
    constexpr int NGL_SCRATCH_STRIDE = 512;
    /* size_t cast required: int * int overflows for num_active > ~4.19M (e.g. fire_m11i
     * gas-per-rank), wrapping to negative int → ~UINT64_MAX after promotion to size_t. */
    size_t na_safe = (size_t)((num_active > 0) ? num_active : 1);
    int *d_scratch = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_scratch", na_safe * (size_t)NGL_SCRATCH_STRIDE * sizeof(int));
    int *d_counts  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_counts", na_safe * sizeof(int));

    /* Drain any prior async GPU work before the passes below. */
    Kokkos::fence();
    /* Fused single pass: BVH walk + write neighbors into per-particle scratchpad */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *scratch = d_scratch;
        int *counts = d_counts;
        int ntiles = gnl->ntiles;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];

        double sr_fac = search_radius_factor;
        double j_rad_scale = j_kernel_radius_scale;
        const double *radii = d_radii;
        const double *src_pos = d_source_pos;
        const double *compact_xyzh = gnl->d_compact_xyzh;
        Kokkos::parallel_for("ngb_fused", num_active, KOKKOS_LAMBDA(int aa) {
            int pf[3] = {pf0, pf1, pf2};
            double bs[3] = {bs0, bs1, bs2};
            double bh[3] = {bh0, bh1, bh2};
            int i = active[aa];
            double h_i = (radii ? radii[aa] : (double)compact_xyzh[i*4+3]) * sr_fac;
            double pos_i[3];
            if(src_pos) { pos_i[0] = src_pos[aa*3+0]; pos_i[1] = src_pos[aa*3+1]; pos_i[2] = src_pos[aa*3+2]; }
            else        { pos_i[0] = (double)compact_xyzh[i*4+0]; pos_i[1] = (double)compact_xyzh[i*4+1]; pos_i[2] = (double)compact_xyzh[i*4+2]; }
            int cnt = search_neighbors_sfc_gpu(compact_xyzh, pos_i, h_i, j_rad_scale,
                                               tiles, ntiles, pool, smode,
                                               bvh, bvh_root,
                                               &scratch[(size_t)aa * NGL_SCRATCH_STRIDE],
                                               NGL_SCRATCH_STRIDE,
                                               pf, bs, bh);
            counts[aa] = cnt;
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("ngb_fused", num_active);

        /* L4 Step-1b validation oracle (DIAGNOSTIC; gate GIZMO_NGB_PRECISION_ORACLE;
         * SYMMETRIC + num_active>0). Validates the double-position substrate: the
         * PRODUCTION device CSR must contain every host DOUBLE physical-truth
         * neighbour — MISSING = under-inclusion RED-ALERT (must be 0). EXTRA = the
         * SIDX_H_SLACK candidate margin, acceptable ONLY for consumers that re-gate
         * the exact predicate (verified density_loop.h / gradient_functions.h; NOT
         * assumed universal). Per-rank, no MPI collective. Sample cap
         * GIZMO_NGB_PRECISION_ORACLE_NMAX (default 128; brute is O(sampled x pool)).
         * Report tag [NGL_PRECISION_ORACLE]. Teardown ledger §39. */
        if(const char *prec_env = getenv("GIZMO_NGB_PRECISION_ORACLE");
           prec_env && atoi(prec_env) != 0 && smode == NGB_SEARCH_SYMMETRIC && num_active > 0) {
            const char *lbl = caller_label ? caller_label : "?";
            if(idx->num_pool <= 0 || !idx->h_pool) {
                if(ThisTask == 0)
                    printf("[NGL_PRECISION_ORACLE] caller=%s mode=SYMM UNAVAILABLE (empty supply pool)\n", lbl);
            } else {
                int num_pool = idx->num_pool;
                const int *h_pool = idx->h_pool;

                /* SSOT predicted-double physical truth: pos = Pos+Vel*drift(->time1),
                 * reach = nlr_particle_symmetric_radius (physical, NO slack). Used for
                 * BOTH the query (when src_pos absent) and the supply. */
                double *j_pos_dbl = NULL, *j_h_dbl = NULL; long n_dt_nonzero = 0;
                ngl_precision_build_ref_buffers(P_shared, num_total, radius_policy,
                                                &j_pos_dbl, &j_h_dbl, &n_dt_nonzero);
                if(!j_pos_dbl || !j_h_dbl) endrun(919231);

                /* Pull production CSR + compact pool to host. */
                using UV = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
                std::vector<int> h_prod_sc((size_t)num_active * NGL_SCRATCH_STRIDE), h_prod_ct(num_active);
                std::vector<double> h_compact(4 * (size_t)num_total);
                Kokkos::deep_copy(Kokkos::View<int*, Kokkos::HostSpace, UV>(h_prod_sc.data(), (size_t)num_active*NGL_SCRATCH_STRIDE),
                                  Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, UV>(scratch, (size_t)num_active*NGL_SCRATCH_STRIDE));
                Kokkos::deep_copy(Kokkos::View<int*, Kokkos::HostSpace, UV>(h_prod_ct.data(), num_active),
                                  Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, UV>(counts, num_active));
                Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace, UV>(h_compact.data(), 4*(size_t)num_total),
                                  Kokkos::View<const double*, GIZMO_KOKKOS_DEVICE_SPACE, UV>(compact_xyzh, 4*(size_t)num_total));

                const char *nmax_env = getenv("GIZMO_NGB_PRECISION_ORACLE_NMAX");
                int n_max = nmax_env ? atoi(nmax_env) : 128;   /* brute truth is O(n_max x pool); raise for tiny synthetics */
                if(n_max < 1) n_max = 1;
                int n_cap = (num_active < n_max) ? num_active : n_max;

                /* SET-compare per sampled active: production CSR vs host double truth.
                 * r via the SAME NGB_PERIODIC_BOX_LONG convention as the device leaf. */
                long tot_truth=0, tot_prod=0, miss=0, extra=0, ovf_rows=0;
                #pragma omp parallel for schedule(dynamic) reduction(+:tot_truth,tot_prod,miss,extra,ovf_rows)
                for(int aa = 0; aa < n_cap; aa++) {
                    /* Overflow guard: search_neighbors_sfc_gpu counts ALL matches but only
                     * writes the first NGL_SCRATCH_STRIDE. A truncated device row would
                     * report FALSE missing (oracle runs BEFORE production's overflow
                     * re-walk into the final CSR). Skip + count LOUD. */
                    if(h_prod_ct[aa] > NGL_SCRATCH_STRIDE) { ovf_rows++; continue; }
                    MyDouble xtmp = 0;
                    int i = active[aa];
                    /* query: production's own double query where present, else the physical
                     * double for particle i (the double version of the SAME P[i] query the
                     * production kernel reads from the double compact substrate). */
                    double pos_i[3];
                    if(src_pos) { pos_i[0]=src_pos[aa*3+0]; pos_i[1]=src_pos[aa*3+1]; pos_i[2]=src_pos[aa*3+2]; }
                    else        { pos_i[0]=j_pos_dbl[i*3+0]; pos_i[1]=j_pos_dbl[i*3+1]; pos_i[2]=j_pos_dbl[i*3+2]; }
                    double h_i = (radii ? radii[aa] : j_h_dbl[i]) * sr_fac;
                    std::vector<int> truth;
                    for(int p = 0; p < num_pool; p++) {
                        int j = h_pool[p];
                        double dxr = pos_i[0]-j_pos_dbl[j*3+0], dyr = pos_i[1]-j_pos_dbl[j*3+1], dzr = pos_i[2]-j_pos_dbl[j*3+2];
                        double adx = NGB_PERIODIC_BOX_LONG_X(dxr,dyr,dzr,1), ady = NGB_PERIODIC_BOX_LONG_Y(dxr,dyr,dzr,1), adz = NGB_PERIODIC_BOX_LONG_Z(dxr,dyr,dzr,1);
                        double r2 = adx*adx + ady*ady + adz*adz;
                        double h_j = j_h_dbl[j] * j_rad_scale;
                        double cut = (h_i > h_j) ? h_i : h_j;
                        if(r2 < cut*cut) truth.push_back(j);
                    }
                    std::sort(truth.begin(), truth.end());
                    int pc = h_prod_ct[aa];
                    std::vector<int> prod(h_prod_sc.begin()+(size_t)aa*NGL_SCRATCH_STRIDE, h_prod_sc.begin()+(size_t)aa*NGL_SCRATCH_STRIDE+pc);
                    std::sort(prod.begin(), prod.end());
                    tot_truth += (long)truth.size(); tot_prod += (long)prod.size();
                    std::vector<int> t;
                    std::set_difference(truth.begin(),truth.end(), prod.begin(),prod.end(), std::back_inserter(t)); miss  += (long)t.size();
                    t.clear(); std::set_difference(prod.begin(),prod.end(), truth.begin(),truth.end(), std::back_inserter(t)); extra += (long)t.size();
                }

                /* First-few production-missing examples (single-threaded rescan; detail). */
                const int KMISS = 8; int nmiss = 0;
                for(int aa = 0; aa < n_cap && nmiss < KMISS; aa++) {
                    if(h_prod_ct[aa] > NGL_SCRATCH_STRIDE) continue;
                    MyDouble xtmp = 0;
                    int i = active_indices_host[aa];   /* host source list — human-readable identity */
                    double pos_i[3];
                    if(src_pos) { pos_i[0]=src_pos[aa*3+0]; pos_i[1]=src_pos[aa*3+1]; pos_i[2]=src_pos[aa*3+2]; }
                    else        { pos_i[0]=j_pos_dbl[i*3+0]; pos_i[1]=j_pos_dbl[i*3+1]; pos_i[2]=j_pos_dbl[i*3+2]; }
                    double h_i = (radii ? radii[aa] : j_h_dbl[i]) * sr_fac;
                    int pc = h_prod_ct[aa];
                    std::vector<int> prod(h_prod_sc.begin()+(size_t)aa*NGL_SCRATCH_STRIDE, h_prod_sc.begin()+(size_t)aa*NGL_SCRATCH_STRIDE+pc);
                    std::sort(prod.begin(), prod.end());
                    for(int p = 0; p < num_pool && nmiss < KMISS; p++) {
                        int j = h_pool[p];
                        double dxr = pos_i[0]-j_pos_dbl[j*3+0], dyr = pos_i[1]-j_pos_dbl[j*3+1], dzr = pos_i[2]-j_pos_dbl[j*3+2];
                        double adx = NGB_PERIODIC_BOX_LONG_X(dxr,dyr,dzr,1), ady = NGB_PERIODIC_BOX_LONG_Y(dxr,dyr,dzr,1), adz = NGB_PERIODIC_BOX_LONG_Z(dxr,dyr,dzr,1);
                        double r = sqrt(adx*adx + ady*ady + adz*adz);
                        double h_j = j_h_dbl[j] * j_rad_scale;
                        double cut = (h_i > h_j) ? h_i : h_j;
                        if(r < cut && !std::binary_search(prod.begin(), prod.end(), j)) {   /* physical neighbour, production missed */
                            double dtj = get_drift_factor_omp_safe(P_shared[j].Ti_current, gizmo_host_ti_current(), j, 0);
                            printf("  [NGL_PRECISION_ORACLE rank=%d] miss caller=%s i.ID=%lld j.ID=%lld r=%.10g cutoff=%.10g margin=%.3g dt_j=%.3g "
                                   "compact_xyzh=(%.10g,%.10g,%.10g;%.6g) truth_xyzh=(%.10g,%.10g,%.10g;%.6g)\n",
                                   ThisTask, lbl, (long long)P_shared[i].ID, (long long)P_shared[j].ID, r, cut, r-cut, dtj,
                                   h_compact[j*4+0],h_compact[j*4+1],h_compact[j*4+2],h_compact[j*4+3],
                                   j_pos_dbl[j*3+0],j_pos_dbl[j*3+1],j_pos_dbl[j*3+2],j_h_dbl[j]);
                            nmiss++;
                        }
                    }
                }

                /* Per-rank LOCAL report (NO MPI collective — inside the per-rank
                 * num_active>0 gate; a collective would deadlock 0-active ranks). */
                printf("[NGL_PRECISION_ORACLE rank=%d] caller=%s mode=SYMM sampled=%d/%d%s pool=%d overflow_rows_skipped=%ld\n"
                       "  truth=%ld prod=%ld  MISSING(truth\\prod)=%ld EXTRA(prod\\truth)=%ld\n"
                       "  [MISSING=under-inclusion RED-ALERT (must be 0); EXTRA=SIDX_H_SLACK candidate margin, consumers re-gate exact predicate]\n"
                       "  dt_j!=0: %ld/%d (%.4f)  [stale-vs-predicted regime; slack must cover raw-vs-predicted when >0]\n",
                       ThisTask, lbl, n_cap, num_active, (num_active > n_cap) ? " CAPPED" : "", num_pool, ovf_rows,
                       tot_truth, tot_prod, miss, extra,
                       n_dt_nonzero, num_total, num_total > 0 ? (double)n_dt_nonzero/(double)num_total : 0.0);
                fflush(stdout);
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(j_pos_dbl);
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(j_h_dbl);
            }
        }
    }

    /* Count overflow particles (count > stride: they need a re-walk in compact phase) */
    int overflow_count = 0;
    {
        int *counts = d_counts;
        Kokkos::parallel_reduce("ngb_overflow_check", num_active,
            KOKKOS_LAMBDA(int aa, int &local) {
                if(counts[aa] > NGL_SCRATCH_STRIDE) local++;
            }, overflow_count);
        Kokkos::fence();
    }

    /* GPU exclusive prefix scan: counts → offsets, returning total.
       Counts are correct even for overflow particles (search_neighbors_sfc_gpu
       returns the true count regardless of bounded write). */
    int64_t total_ll = 0;
    {
        int *counts = d_counts;
        int64_t *offsets = gnl->offsets;
        Kokkos::parallel_scan("ngb_offsets_scan", num_active,
            KOKKOS_LAMBDA(int aa, int64_t &update, const bool final) {
                int64_t v = (int64_t)counts[aa];
                if(final) offsets[aa] = update;
                update += v;
            }, total_ll);
        Kokkos::fence();
    }
    /* Sanity guard: the CSR index (gnl->total_pairs / gnl->offsets) is 64-bit,
     * so INT_MAX is no longer a ceiling. Still abort cleanly on a nonsensical
     * total_pairs (negative => prefix-scan corruption; or a count so large the
     * neighbors allocation byte size would overflow size_t) rather than letting
     * a bad value reach kokkos_malloc / the device compact pass. */
    if(total_ll < 0 || (uint64_t)total_ll > (uint64_t)(SIZE_MAX / sizeof(int))) {
        fprintf(stderr,
            "[NGL FATAL rank=%d] CSR total_pairs implausible: caller=%s num_active=%d "
            "total_pairs=%lld  search_radius_factor=%g j_kernel_radius_scale=%g\n",
            ThisTask, caller_label ? caller_label : "?", num_active, (long long)total_ll,
            search_radius_factor, j_kernel_radius_scale);
        fflush(stderr);
        endrun(915100);
    }
    int64_t total = total_ll;
    gnl->offsets[num_active] = total;
    gnl->total_pairs = total;

    /* Allocate CSR neighbors array (length is 64-bit; element type stays int) */
    gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("ngl_neighbors", (size_t)((total > 0) ? total : 1) * sizeof(int));

    /* Compact: copy from per-particle scratchpad into dense CSR neighbors[]. */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *scratch = d_scratch;
        int *counts = d_counts;
        int64_t *offsets = gnl->offsets;
        int *neighbors = gnl->neighbors;
        int ntiles = gnl->ntiles;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];
        double sr_fac = search_radius_factor;
        double j_rad_scale = j_kernel_radius_scale;
        const double *radii = d_radii;
        const double *src_pos = d_source_pos;
        const double *compact_xyzh = gnl->d_compact_xyzh;
        Kokkos::parallel_for("ngb_compact", num_active, KOKKOS_LAMBDA(int aa) {
            int n = counts[aa];
            int64_t dst = offsets[aa];
            if(n <= NGL_SCRATCH_STRIDE) {
                size_t src = (size_t)aa * NGL_SCRATCH_STRIDE;
                for(int k = 0; k < n; k++) neighbors[dst + k] = scratch[src + k];
            } else {
                /* Overflow path: re-walk BVH writing directly into neighbors[] */
                int pf[3] = {pf0, pf1, pf2};
                double bs[3] = {bs0, bs1, bs2};
                double bh[3] = {bh0, bh1, bh2};
                int i = active[aa];
                double h_i = (radii ? radii[aa] : (double)compact_xyzh[i*4+3]) * sr_fac;
                double pos_i[3];
                if(src_pos) { pos_i[0] = src_pos[aa*3+0]; pos_i[1] = src_pos[aa*3+1]; pos_i[2] = src_pos[aa*3+2]; }
                else        { pos_i[0] = (double)compact_xyzh[i*4+0]; pos_i[1] = (double)compact_xyzh[i*4+1]; pos_i[2] = (double)compact_xyzh[i*4+2]; }
                search_neighbors_sfc_gpu(compact_xyzh, pos_i, h_i, j_rad_scale,
                                         tiles, ntiles, pool, smode,
                                         bvh, bvh_root,
                                         &neighbors[dst], 0x7fffffff,
                                         pf, bs, bh);
            }
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("ngb_compact", num_active);
    }

    /* Free temporaries */
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_scratch);
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_counts);
    if(d_radii) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    if(d_source_pos) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_source_pos);


    /* Lazy-drift hook (Attack C): drift each neighbor in the freshly-built
     * CSR list to time1 on host. Active particle i is already drifted
     * (move_particles iterated ActiveParticleList). Each pool member j touched
     * by this kernel needs its predicted state (CellP[j].VelPred / Density /
     * InternalEnergyPred / KernelRadius) at time1 before the kernel reads it;
     * drift_particle(j, time1) handles all of that with a single call.
     *
     * drift_particle's "if(time1 == time0) return" early-exit dedups: a j
     * already drifted (e.g. it was in another active i's neighbor list
     * earlier this step, or it IS an active particle) is a fast no-op.
     *
     * Marks h_dirty for the touched j's so that the NEXT gpu_ngb_list_build
     * call's compact_h_refresh updates compact_xyzh[j*4+3] from the freshly-
     * drifted KernelRadius. The CURRENT call's compact_xyzh h field is
     * stale by up to one drift step; the SIDX_H_SLACK inflation in
     * compact_xyzh write paths absorbs that staleness in the BVH tile-overlap
     * test. Per-pair r² acceptance reads the actual P[j].KernelRadius (now
     * freshly drifted), so correctness is preserved. */
    if(gnl->total_pairs > 0 && gnl->neighbors) {
        std::vector<int> ngb_host((size_t)gnl->total_pairs);
        gpu_ngb_copy_neighbors_to_host(gnl, ngb_host.data());
        /* Out-of-line host accessor. Lazy-drift target for CSR
         * neighbors — host-side drift_particle calls. */
        integertime time1 = gizmo_host_ti_current();
        for(int64_t idx_n = 0; idx_n < gnl->total_pairs; idx_n++) {
            int j = ngb_host[idx_n];
            if(j >= 0 && j < num_total) drift_particle(j, time1);
        }
        /* Lazy drift just called drift_particle on each j in ngb_host.
         * drift_particle mutates Ti_current, Pos, AND KernelRadius
         * (predict.cc:160,229 — *= exp(divv_fac/N)). Mark h-dirty for both
         * GPU SIDX tracker and host glt cache via the SSOT helper.
         * Promote-to-all kicks in per cache if any cache's bitset popcount
         * exceeds threshold. */
        if(gnl->total_pairs <= (int64_t)INT_MAX) {
            gizmo_mark_kernel_radius_dirty_indices(ngb_host.data(), (int)gnl->total_pairs);
        } else {
            /* >2^31 neighbor pairs: an index list this large is hugely
             * redundant (num_total < 2^31), so truncating n would mark a
             * wrong subset. Escalate to a full-pool mark across both caches. */
            gizmo_mark_kernel_radius_dirty_range(0, num_total);
        }
        /* Move detector baseline past the lazy drift's Ti_current/Pos updates
         * — those are predicted-state setup, not kernel writes that need
         * writeback. Subsequent kernel-side writes to ghost particles will
         * still be flagged by ghost_write_detector_end(). No-op when
         * GIZMO_GPU_ARENA_DEBUG is undefined or detector is inactive. */
        ghost_write_detector_resnapshot_after_lazy_drift();
    }

    /* Charge list-build wall, less any kernel time already charged inside it,
     * so the two rows never overlap. */
    cpu_charge_child(CPU_NGB_BUILD, cpu_minus_children(timediff(t_entry, my_second()), cpu_rows_child0));
}


void gpu_ngb_list_free(gpu_neighbor_list_t *gnl, gpu_spatial_index_t *cached_idx)
{
    if(gnl->neighbors) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(gnl->neighbors);
    if(gnl->offsets)   Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->offsets);
    if(gnl->d_active)  Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_active);
    /* Only free tiles/BVH/pool/compact_xyzh if they were NOT from the cached index.
     * Pointers may also be NULL (early-out path with no cache); guard each.
     * d_compact_xyzh in particular was previously leaked here for every
     * non-cached call (~199 MB for an all-types pool, ~73 MB for gas-only),
     * accumulating with each mech_fb/radfb_g/sink call that passes cached_idx=NULL. */
    if(!cached_idx || !cached_idx->valid ||
       gnl->d_tiles != cached_idx->d_tiles) {
        if(gnl->d_compact_xyzh) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(gnl->d_compact_xyzh);
        if(gnl->d_pool)  Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(gnl->d_pool);
        if(gnl->d_bvh)   Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(gnl->d_bvh);
        if(gnl->d_tiles) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(gnl->d_tiles);
    }
}

/* Copy gnl->neighbors (DEVICE_SPACE) into a host buffer. Caller owns host_dest.
   For host-side per-source loops (radfb_local, merge_split, density.cc:868
   HYDRO_VOLUME_CORRECTIONS path, turb_powerspectra, twopoint) that index
   neighbors[] from CPU code. */
void gpu_ngb_copy_neighbors_to_host(const gpu_neighbor_list_t *gnl, int *host_dest)
{
    if(!host_dest || !gnl || gnl->total_pairs <= 0 || !gnl->neighbors) {return;}
    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h(host_dest, (size_t)gnl->total_pairs);
    Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        d(gnl->neighbors, (size_t)gnl->total_pairs);
    Kokkos::deep_copy(h, d);
}


/* High-level wrapper: build a symmetric neighbor list on GPU and return it
   in the mymalloc-based neighbor_list_t format expected by gradient/hydro.
   Called from accel.cc (which is NOT compiled by nvcc).
   search_radius_factor: multiplier on KernelRadius (default 1.0; >1 for TURB_DIFF_DYNAMIC). */
void gpu_build_symmetric_neighbor_list(struct particle_data *P_host, int num_total,
                                       int *active_indices, int num_active,
                                       neighbor_list_t *out,
                                       double search_radius_factor)
{
    /* Use the per-step particle arena instead of a dedicated full-NumPart memcpy.
     * The arena's fast path is a no-op when valid (e.g. when gradient/hydro
     * already populated it earlier in the step), avoiding ~2.3s of redundant
     * P-copy on small-N symlist invocations.  Pass the global CellP so the
     * arena's "valid" state remains consistent across mixed P/CellP consumers. */
    gpu_particles_arena_set_site("gpu_build_symmetric_neighbor_list");
    gpu_particles_arena_acquire(num_total, P_host, CellP);
    struct particle_data *P_shared = gpu_particles_arena_P();

    /* Build GPU CSR — share gas-only SIDX with density via the step-persistent cache.
     *
     * search_radius_factor is applied to BOTH the i-side query radius and the
     * j-side kernel radius — a genuinely symmetric scaled search. This repairs
     * the j-side under-search in the shared symlist (hydro-gradient Velocity_hat
     * wide filter under TURB_DIFF_DYNAMIC).
     *
     * RADIUS SEMANTICS: pass EXPLICIT raw per-active radii (P[i].KernelRadius)
     * so search_radius_factor multiplies the RAW kernel radius. With NULL
     * radii the builder would derive h_i from compact_xyzh[i*4+3], which is
     * already slack-inflated (P.KernelRadius * (1+SIDX_H_SLACK)) — compounding
     * the slack into the physics widening factor (e.g. fac=2 -> effective ~3h
     * search, ~27x neighbor volume, CSR-overflow / kernel stall). The runner
     * Spec path already passes explicit fac*raw radii; this matches it. */
    std::vector<double> symlist_raw_radii((num_active > 0) ? (size_t)num_active : 1);
    for(int aa = 0; aa < num_active; aa++) {
        symlist_raw_radii[aa] = (double) P_shared[active_indices[aa]].KernelRadius;
    }
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, active_indices, num_active,
                       NGB_SEARCH_SYMMETRIC, 1 /* gas only */, &gpu_nl, gpu_step_sidx_ptr(),
                       search_radius_factor, symlist_raw_radii.data(), NULL, "symlist",
                       search_radius_factor /* j_kernel_radius_scale */);

    /* Copy CSR into mymalloc neighbor_list_t */
    out->num_active = num_active;
    out->total_pairs = gpu_nl.total_pairs;
    out->offsets = (int64_t *) mymalloc("ngb_offsets", (size_t)(num_active + 1) * sizeof(int64_t));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (size_t)(gpu_nl.total_pairs > 0 ? gpu_nl.total_pairs : 1) * sizeof(int));
    /* gpu_nl.offsets is SharedSpace (UVM) → host memcpy is fine.
     * gpu_nl.neighbors is DEVICE_SPACE (CudaSpace) → must use deep_copy, not host memcpy. */
    memcpy(out->offsets, gpu_nl.offsets, (size_t)(num_active + 1) * sizeof(int64_t));
    if(gpu_nl.total_pairs > 0) {
        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_neighbors(out->neighbors, (size_t)gpu_nl.total_pairs);
        Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_neighbors(gpu_nl.neighbors, (size_t)gpu_nl.total_pairs);
        Kokkos::deep_copy(h_neighbors, d_neighbors);
    }

    /* Free GPU temporaries (keep tiles/BVH alive — owned by g_step_sidx).
     * Arena is intentionally not released — subsequent gradient/hydro callers
     * benefit from the fast-path skip. */
    gpu_ngb_list_free(&gpu_nl, gpu_step_sidx_ptr());

}


/* Cross-type variant: i-side is the caller's active_indices (any type), j-side
   is filtered by j_type_bitmask. Caller supplies per-active search radii (so
   e.g. KernelRadiusDM or AGS_Hsml can be used instead of P[i].KernelRadius).
   Returns a neighbor_list_t in the mymalloc format, same as the symmetric
   variant — consumers look identical. */
void gpu_build_cross_type_neighbor_list(struct particle_data *P_host, int num_total,
                                        int *i_active_indices, int num_active,
                                        const double *i_search_radii_host,
                                        int j_type_bitmask, int search_mode,
                                        neighbor_list_t *out)
{
    /* Use the per-step particle arena to avoid a redundant full-NumPart memcpy
     * (see gpu_build_symmetric_neighbor_list for rationale). */
    gpu_particles_arena_set_site("gpu_build_cross_type_neighbor_list");
    gpu_particles_arena_acquire(num_total, P_host, CellP);
    struct particle_data *P_shared = gpu_particles_arena_P();

    /* Build GPU CSR with explicit per-i radii and j-side type filter */
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, i_active_indices, num_active,
                       search_mode, j_type_bitmask, &gpu_nl, NULL,
                       1.0 /* search_radius_factor */, i_search_radii_host, NULL, "xtype");

    /* Copy CSR into mymalloc neighbor_list_t */
    out->num_active = num_active;
    out->total_pairs = gpu_nl.total_pairs;
    out->offsets = (int64_t *) mymalloc("ngb_offsets", (size_t)(num_active + 1) * sizeof(int64_t));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (size_t)(gpu_nl.total_pairs > 0 ? gpu_nl.total_pairs : 1) * sizeof(int));
    /* See gpu_build_symmetric_neighbor_list for why neighbors needs deep_copy. */
    memcpy(out->offsets, gpu_nl.offsets, (size_t)(num_active + 1) * sizeof(int64_t));
    if(gpu_nl.total_pairs > 0) {
        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_neighbors(out->neighbors, (size_t)gpu_nl.total_pairs);
        Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_neighbors(gpu_nl.neighbors, (size_t)gpu_nl.total_pairs);
        Kokkos::deep_copy(h_neighbors, d_neighbors);
    }

    /* Free GPU temporaries.  Arena is intentionally retained for subsequent callers. */
    gpu_ngb_list_free(&gpu_nl, NULL);
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */
