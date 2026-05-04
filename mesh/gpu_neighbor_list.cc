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
#include <cmath>
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/step_phases.h"
#include "../system/gpu_particles_arena.h"
#include "../core/proto.h"

#include "sfc_tiles.h"
#include "sfc_tiles_functions.h"
#include "gpu_neighbor_list.h"
#include "neighbor_list.h"

/* TILE_PERIODIC_X/Y/Z defined in sfc_tiles.h (included via gpu_neighbor_list.h) */

/* Persistent gas-only SIDX shared across density+symlist within a step.
 * Lifetime: built lazily on first gas (type_bitmask=1) ngb_list_build, reused
 * for all subsequent gas builds, freed by gpu_step_sidx_invalidate() after drift. */
static gpu_spatial_index_t g_step_sidx = {NULL,NULL,NULL,0,0,{0},{0},{0},NULL,0};
static gpu_spatial_index_t g_step_sidx_alltypes = {NULL,NULL,NULL,0,0,{0},{0},{0},NULL,0};

gpu_spatial_index_t *gpu_step_sidx_ptr(void) { return &g_step_sidx; }
gpu_spatial_index_t *gpu_step_sidx_alltypes_ptr(void) { return &g_step_sidx_alltypes; }

/* GIZMO_NGB_QUIET=1 suppresses DIAG_NGL/DIAG_SIDX/DIAG_DENS/DIAG_SYMNL/DIAG_GRAD
 * prints. Used to estimate the wallclock contribution of the diagnostic prints
 * themselves (typically a few hundred per step on rank 0; printf+fflush to a
 * scratch-mounted slurm log can plausibly add ms-scale overhead each).
 * Cached on first read: getenv is reasonably cheap but called per-DIAG site
 * adds up across ~hundreds of fires per step. */
static int g_ngb_diag_quiet_cached = -1;
extern "C" int gizmo_ngb_diag_quiet(void)
{
    if(g_ngb_diag_quiet_cached < 0) {
        const char *q = getenv("GIZMO_NGB_QUIET");
        g_ngb_diag_quiet_cached = (q && q[0] && q[0] != '0') ? 1 : 0;
    }
    return g_ngb_diag_quiet_cached;
}

/* Dirty-index tracking for compact_xyzh.h field. Two modes:
 *  - g_dirty_all = true: full-pool refresh required (fresh arena alloc,
 *    >threshold accumulated dirty indices, or unknown-scope mutator via the
 *    backwards-compat _mark_h_dirty() entry point).
 *  - g_dirty_all = false: refresh only the indices in g_dirty_list.
 *
 * The list auto-promotes to "all" when its size crosses
 * G_DIRTY_PROMOTE_THRESHOLD (a refresh of N indices via list costs ~the same
 * as a full-pool scan once N is a substantial fraction of num_total, plus
 * we'd be paying device-buffer staging cost for the list itself). */
static bool g_dirty_all = true;
static std::vector<int> g_dirty_list;
static const int G_DIRTY_PROMOTE_THRESHOLD = 1 << 20; /* 1M indices */

static inline void g_dirty_clear_(void)
{
    g_dirty_all = false;
    g_dirty_list.clear();
}

void gpu_compact_xyzh_mark_h_dirty_all(void)
{
    g_dirty_all = true;
    g_dirty_list.clear(); /* superseded; reclaim memory at next op if huge */
}

void gpu_compact_xyzh_mark_h_dirty_idx(int i)
{
    if(g_dirty_all) {return;} /* already covered */
    if(i < 0) {return;}
    g_dirty_list.push_back(i);
    if((int)g_dirty_list.size() > G_DIRTY_PROMOTE_THRESHOLD) {
        gpu_compact_xyzh_mark_h_dirty_all();
    }
}

void gpu_compact_xyzh_mark_h_dirty_range(int start, int end)
{
    if(g_dirty_all) {return;}
    if(end <= start) {return;}
    int n = end - start;
    if((int)(g_dirty_list.size() + n) > G_DIRTY_PROMOTE_THRESHOLD) {
        gpu_compact_xyzh_mark_h_dirty_all();
        return;
    }
    g_dirty_list.reserve(g_dirty_list.size() + n);
    for(int i = start; i < end; i++) {g_dirty_list.push_back(i);}
}

void gpu_compact_xyzh_mark_h_dirty_indices(const int *indices, int n)
{
    if(g_dirty_all) {return;}
    if(!indices || n <= 0) {return;}
    if((int)(g_dirty_list.size() + n) > G_DIRTY_PROMOTE_THRESHOLD) {
        gpu_compact_xyzh_mark_h_dirty_all();
        return;
    }
    g_dirty_list.reserve(g_dirty_list.size() + n);
    for(int k = 0; k < n; k++) {g_dirty_list.push_back(indices[k]);}
}

/* Backwards-compat: any caller that doesn't know which indices it dirtied
 * conservatively forces a full-pool refresh. */
void gpu_compact_xyzh_mark_h_dirty(void) { gpu_compact_xyzh_mark_h_dirty_all(); }

/* Drift-time SIDX refresh (Attack B v2: incremental rebuild).
 *
 * The unconditional full rebuild post-drift in the prior code cost
 * ~1.3s/step on fire_m11i tiny-N (the dominant tiny-N bucket post-
 * UVM-canonical). Of that, ~1s is build_sfc_tiles' SFC sort over
 * 12.4M particles — work that's only needed when particle layout
 * actually changes (i.e., domain_decomp).
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
 *      ms-scale). h is unchanged by drift; mark_h_dirty machinery
 *      handles h-changes from density iter etc. independently.
 *
 * Correctness invariant: each tile's bbox covers all current positions
 * of the particles in its pool. Tile assignments are frozen, so as
 * particles "wander" spatially they accumulate into multiple tiles'
 * bbox regions — BVH queries may visit a couple extra tiles per query
 * (inefficiency, never a missed neighbor; each tile still iterates its
 * pool, particles' actual positions are checked).
 *
 * Reset boundary: gpu_step_sidx_invalidate_full() at every domain_decomp
 * frees the SIDX, forcing a fresh rebuild including SFC sort. Domain
 * decomp is already a heavy step, so the marginal cost is small. This
 * naturally bounds bbox dispersion within a decomp interval and resets
 * tile assignments to current spatial layout.
 *
 * Diagnostics: sidx_drift_max_extent_ratio tracks max(new_extent /
 * orig_extent_at_last_full_rebuild). Watch this to confirm the design
 * assumption that bbox-spread is bounded between decomps. If it grows
 * pathologically over many drifts, v3 (SFC reassignment) would be
 * justified.
 */

/* Recomputes tile bboxes from current particle positions AND fills the
 * host-side position-staging buffer (idx->h_pos_buf) for every pool
 * member. Returns max (new_extent / orig_extent) across all tiles. */
static double sidx_refresh_tile_bboxes_host(gpu_spatial_index_t *idx,
                                             struct particle_data *P_shared)
{
    sfc_tile_t *h_tiles = idx->h_tiles;
    int *h_pool = idx->h_pool;
    int ntiles = idx->ntiles;
    const double *orig_extent = idx->h_tile_orig_max_extent;
    float *pos_buf = idx->h_pos_buf;
    double max_ratio = 0.0;

    #pragma omp parallel for reduction(max:max_ratio) schedule(static)
    for(int t = 0; t < ntiles; t++) {
        sfc_tile_t *tile = &h_tiles[t];
        if(tile->count <= 0) continue;
        int j0 = h_pool[tile->first];
        double x0 = P_shared[j0].Pos[0], y0 = P_shared[j0].Pos[1], z0 = P_shared[j0].Pos[2];
        double lo0 = x0, hi0 = x0;
        double lo1 = y0, hi1 = y0;
        double lo2 = z0, hi2 = z0;
        double hmax = P_shared[j0].KernelRadius;
        if(pos_buf) { pos_buf[j0*3+0] = (float)x0; pos_buf[j0*3+1] = (float)y0; pos_buf[j0*3+2] = (float)z0; }
        for(int s = 1; s < tile->count; s++) {
            int j = h_pool[tile->first + s];
            double x = P_shared[j].Pos[0];
            double y = P_shared[j].Pos[1];
            double z = P_shared[j].Pos[2];
            if(x < lo0) lo0 = x; else if(x > hi0) hi0 = x;
            if(y < lo1) lo1 = y; else if(y > hi1) hi1 = y;
            if(z < lo2) lo2 = z; else if(z > hi2) hi2 = z;
            double h = P_shared[j].KernelRadius;
            if(h > hmax) hmax = h;
            if(pos_buf) { pos_buf[j*3+0] = (float)x; pos_buf[j*3+1] = (float)y; pos_buf[j*3+2] = (float)z; }
        }
        tile->lo[0] = lo0; tile->hi[0] = hi0;
        tile->lo[1] = lo1; tile->hi[1] = hi1;
        tile->lo[2] = lo2; tile->hi[2] = hi2;
        tile->hmax = hmax;

        if(orig_extent && orig_extent[t] > 0) {
            double ext = hi0 - lo0;
            double e1 = hi1 - lo1; if(e1 > ext) ext = e1;
            double e2 = hi2 - lo2; if(e2 > ext) ext = e2;
            double r = ext / orig_extent[t];
            if(r > max_ratio) max_ratio = r;
        }
    }
    return max_ratio;
}

/* Refresh compact_xyzh positions on device via host-staged buffer.
 *
 * Avoids the UVM-fault-storm cost of a parallel_for that reads P_shared.Pos
 * directly on device (which costs ~1.25s/step on fire_m11i 12.4M after host
 * drift just wrote those pages — every page faults migration to GPU on first
 * access). Instead: positions were already filled into idx->h_pos_buf by the
 * host bbox-recompute loop; bulk deep_copy to d_pos_buf (~200MB at NVLink
 * ~600GB/s = sub-ms), then a small device kernel scatters into the
 * interleaved d_compact_xyzh array. h field intentionally untouched —
 * drift doesn't change h; mark_h_dirty machinery handles h refresh
 * separately. */
static void sidx_refresh_compact_positions_device(gpu_spatial_index_t *idx)
{
    int num_total = idx->num_total;
    if(!idx->h_pos_buf || !idx->d_pos_buf) return;
    /* Bulk host->device copy of the staged position buffer. Single linear
     * cudaMemcpy under the hood; pages are migrated as one transfer rather
     * than fault-by-fault on first device read. */
    using UV = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<float*, Kokkos::HostSpace, UV>            h_v(idx->h_pos_buf, 3 * num_total);
    Kokkos::View<float*, GIZMO_KOKKOS_DEVICE_SPACE, UV>    d_v(idx->d_pos_buf, 3 * num_total);
    Kokkos::deep_copy(d_v, h_v);

    float *compact = idx->d_compact_xyzh;
    float *pos_buf = idx->d_pos_buf;
    Kokkos::parallel_for("compact_xyzh_pos_scatter", num_total, KOKKOS_LAMBDA(int i) {
        compact[i*4+0] = pos_buf[i*3+0];
        compact[i*4+1] = pos_buf[i*3+1];
        compact[i*4+2] = pos_buf[i*3+2];
    });
    Kokkos::fence();
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

/* Drift-time refresh: skip the SFC sort, recompute bboxes/BVH, refresh compact_xyzh. */
static void sidx_refresh_after_drift(gpu_spatial_index_t *idx,
                                      struct particle_data *P_shared,
                                      double *max_extent_ratio_out,
                                      double *t_bbox_out, double *t_bvh_out,
                                      double *t_stage_out, double *t_compact_out)
{
    double t0 = my_second();
    double max_ratio = sidx_refresh_tile_bboxes_host(idx, P_shared);
    double t1 = my_second();
    sidx_rebuild_bvh_inplace(idx);
    double t2 = my_second();
    sidx_stage_to_device(idx);
    double t3 = my_second();
    sidx_refresh_compact_positions_device(idx);
    double t4 = my_second();
    *max_extent_ratio_out = max_ratio;
    *t_bbox_out    = timediff(t0, t1);
    *t_bvh_out     = timediff(t1, t2);
    *t_stage_out   = timediff(t2, t3);
    *t_compact_out = timediff(t3, t4);
}

void gpu_step_sidx_invalidate(void)
{
    double t_total_start = my_second();
    int refreshed = 0;
    double max_extent_ratio = 0.0;
    double t_bbox = 0, t_bvh = 0, t_stage = 0, t_compact = 0;

    struct particle_data *P_shared = gpu_particles_arena_P();
    if(!P_shared) {
        /* No arena -> no canonical particle storage; fall back to full free. */
        gpu_step_sidx_invalidate_full();
        return;
    }

    /* v2 scope: refresh the gas SIDX (the hot-path 1.3s/step bucket from
     * density_sidx_prebuild); alltypes goes through full-free since it
     * only builds on sink-active steps and isn't on the dominant tiny-N
     * path. Future v2.1 can extend the refresh path to alltypes once
     * gas is validated. */
    if(g_step_sidx_alltypes.valid) gpu_spatial_index_free(&g_step_sidx_alltypes);

    gpu_spatial_index_t *idx = &g_step_sidx;
    if(idx->valid) {
        if(!idx->h_tiles || !idx->h_pool || !idx->d_compact_xyzh || idx->ntiles <= 0) {
            /* Defensive: incomplete state -> free, fall back to full rebuild. */
            gpu_spatial_index_free(idx);
            gpu_compact_xyzh_mark_h_dirty_all();
        } else {
            double r = 0, tb = 0, tv = 0, ts = 0, tc = 0;
            sidx_refresh_after_drift(idx, P_shared, &r, &tb, &tv, &ts, &tc);
            max_extent_ratio = r;
            t_bbox = tb; t_bvh = tv; t_stage = ts; t_compact = tc;
            refreshed = 1;
            /* h-dirty state intentionally left intact: drift doesn't change h, and
             * any pending dirties from density iter etc. will be consumed by the
             * next gpu_ngb_list_build refresh as usual. */
        }
    }

    double t_total = timediff(t_total_start, my_second());
    gizmo_step_phase_record("sidx_drift_refresh_calls",     (double)refreshed);
    gizmo_step_phase_record("sidx_drift_refresh_total",     t_total);
    gizmo_step_phase_record("sidx_drift_bbox_recompute",    t_bbox);
    gizmo_step_phase_record("sidx_drift_bvh_refit",         t_bvh);
    gizmo_step_phase_record("sidx_drift_stage_to_device",   t_stage);
    gizmo_step_phase_record("sidx_drift_compact_refresh",   t_compact);
    gizmo_step_phase_record("sidx_drift_max_extent_ratio",  max_extent_ratio);
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
                             const char *caller_label)
{
    /* Capture periodicity parameters */
    idx->periodic_flags[0] = TILE_PERIODIC_X;
    idx->periodic_flags[1] = TILE_PERIODIC_Y;
    idx->periodic_flags[2] = TILE_PERIODIC_Z;
    idx->box_sizes[0] = boxSize_X; idx->box_sizes[1] = boxSize_Y; idx->box_sizes[2] = boxSize_Z;
    idx->box_halves[0] = boxHalf_X; idx->box_halves[1] = boxHalf_Y; idx->box_halves[2] = boxHalf_Z;
    /* Hard guard: bbox_overlaps_sphere_gpu's periodic-wrap math goes pathological
     * (every node "overlaps" every query, BVH degenerates to exhaustive scan)
     * if box_sizes[k] is 0 while periodic_flags[k] is on. This was the latent
     * bug pre-866aad55 caused by missing gizmo_gpu_sync_all_ngb registration.
     * Fail loud at build time so any future regression in the All_dev sync
     * path can never silently corrupt performance again. */
    for(int k = 0; k < 3; k++) {
        if(idx->periodic_flags[k] && !(idx->box_sizes[k] > 0.0)) {
            printf("gpu_spatial_index_build: periodic_flags[%d]=1 but box_sizes[%d]=%g (caller='%s'). "
                   "Likely cause: gpu_neighbor_list TU's All_dev not synced from host All. "
                   "Check that gizmo_gpu_sync_all_ngb is registered in gizmo_gpu_sync_all() "
                   "(cooling/cooling.cc).\n",
                   k, k, idx->box_sizes[k], caller_label ? caller_label : "?");
            fflush(stdout);
            endrun(913004);
        }
    }

    double t_si0 = my_second(); /* DIAG */
    /* Build SFC tiles + BVH on CPU */
    sfc_tile_t *h_tiles;
    int *h_pool;
    int num_pool;
    int ntiles = build_sfc_tiles(P_shared, num_total, type_bitmask, TILE_TARGET_SIZE,
                                 &h_tiles, &h_pool, &num_pool);
    idx->ntiles = ntiles;
    double t_si1 = my_second(); /* DIAG: after build_sfc_tiles */

    tile_bvh_node_t *h_bvh;
    int bvh_nnodes = build_tile_bvh(h_tiles, ntiles, &h_bvh);
    idx->bvh_root = bvh_nnodes - 1;
    double t_si2 = my_second(); /* DIAG: after build_tile_bvh */

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
    idx->d_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(ntiles * sizeof(sfc_tile_t));
    idx->d_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(bvh_size * sizeof(tile_bvh_node_t));
    idx->d_pool = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(pool_size * sizeof(int));

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
    double t_si3 = my_second(); /* DIAG: after staging tiles/BVH/pool to DEVICE_SPACE */

    /* Build compact float4 position+h array for cache-efficient GPU BVH traversal.
       32MB for 2M particles vs 800MB for full P_shared — fits in H100 L2 (50MB),
       eliminating the random-access cache misses that dominate the GPU count/fill passes.
       In DEVICE_SPACE so the BVH-walk kernels read from HBM directly. */
    idx->d_compact_xyzh = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(num_total * 4 * sizeof(float));
    {
        float *compact = idx->d_compact_xyzh;
        Kokkos::parallel_for("compact_xyzh_build", num_total, KOKKOS_LAMBDA(int i) {
            compact[i*4+0] = (float)P_shared[i].Pos[0];
            compact[i*4+1] = (float)P_shared[i].Pos[1];
            compact[i*4+2] = (float)P_shared[i].Pos[2];
            compact[i*4+3] = (float)P_shared[i].KernelRadius;
        });
        Kokkos::fence();
    }
    double t_si4 = my_second(); /* DIAG: after compact array build */

    /* Keep host-side persistent copies of tiles/pool/BVH alive across drifts
     * so the incremental refresh path (gpu_step_sidx_invalidate ->
     * sidx_refresh_after_drift) can recompute tile bboxes from current
     * particle positions on host without re-running the SFC sort.
     *
     * MUST use Kokkos::HostSpace allocator (heap-based) NOT mymalloc — the
     * latter is LIFO-stack-disciplined and persistent SIDX buffers would
     * sit on top of any subsequent transient allocation (e.g. density's
     * per-step mymalloc'd Left/Right arrays), preventing those transients
     * from being freed in LIFO order. The transient mymalloc'd h_tiles /
     * h_pool / h_bvh from build_sfc_tiles + build_tile_bvh are copied
     * out then myfree'd in proper LIFO order below. */
    idx->h_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<Kokkos::HostSpace>(ntiles * sizeof(sfc_tile_t));
    memcpy(idx->h_tiles, h_tiles, ntiles * sizeof(sfc_tile_t));
    idx->h_pool = (int *) Kokkos::kokkos_malloc<Kokkos::HostSpace>(pool_size * sizeof(int));
    memcpy(idx->h_pool, h_pool, num_pool * sizeof(int));
    idx->h_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<Kokkos::HostSpace>(bvh_size * sizeof(tile_bvh_node_t));
    memcpy(idx->h_bvh, h_bvh, bvh_nnodes * sizeof(tile_bvh_node_t));
    idx->h_bvh_nnodes = bvh_nnodes;
    idx->num_pool = num_pool;

    /* Drift-refresh staging buffers: h_pos_buf is filled per-pool-member by the
     * host bbox-recompute loop; bulk deep_copy to d_pos_buf; device scatter
     * into d_compact_xyzh. Sized 3*num_total floats so each pool index can
     * write directly to h_pos_buf[j*3+0..2] without remapping. Non-pool
     * entries stay uninitialized in h_pos_buf and their stale d_compact_xyzh
     * positions are never read by BVH queries (BVH only visits tiles, tiles
     * only contain pool members). */
    int pos_buf_count = (num_total > 0 ? num_total : 1);
    idx->h_pos_buf = (float *) Kokkos::kokkos_malloc<Kokkos::HostSpace>(3 * pos_buf_count * sizeof(float));
    idx->d_pos_buf = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(3 * pos_buf_count * sizeof(float));

    /* Snapshot per-tile original max-axis extent for the drift refresh's
     * extent-ratio diagnostic (sidx_drift_max_extent_ratio). */
    idx->h_tile_orig_max_extent = (double *) Kokkos::kokkos_malloc<Kokkos::HostSpace>(
                                              (ntiles > 0 ? ntiles : 1) * sizeof(double));
    for(int t = 0; t < ntiles; t++) {
        double ext = idx->h_tiles[t].hi[0] - idx->h_tiles[t].lo[0];
        double e1 = idx->h_tiles[t].hi[1] - idx->h_tiles[t].lo[1]; if(e1 > ext) ext = e1;
        double e2 = idx->h_tiles[t].hi[2] - idx->h_tiles[t].lo[2]; if(e2 > ext) ext = e2;
        idx->h_tile_orig_max_extent[t] = ext;
    }

    /* Free the transient mymalloc'd build buffers in proper LIFO order
     * (build_tile_bvh allocated h_bvh last; build_sfc_tiles allocated
     * h_pool then h_tiles). */
    myfree(h_bvh);
    myfree(h_tiles);
    myfree(h_pool);

    idx->num_total = num_total;
    idx->valid = 1;
    g_dirty_clear_(); /* fresh build seeded compact_xyzh from current h */

    if(ThisTask == 0 && !gizmo_ngb_diag_quiet()) { /* DIAG: spatial index build breakdown — env-gated by GIZMO_NGB_QUIET */
        printf("[DIAG_SIDX caller=%s tbm=0x%x ntiles=%d pool=%d] sfc_tiles=%.3f bvh_build=%.3f memcpy=%.3f compact=%.3f total=%.3f\n",
               caller_label ? caller_label : "?", type_bitmask, ntiles, num_pool,
               timediff(t_si0, t_si1), timediff(t_si1, t_si2), timediff(t_si2, t_si3),
               timediff(t_si3, t_si4), timediff(t_si0, t_si4));
        fflush(stdout);
    }
}

void gpu_spatial_index_free(gpu_spatial_index_t *idx)
{
    if(idx->d_compact_xyzh) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_compact_xyzh); idx->d_compact_xyzh = NULL;}
    if(idx->d_pool) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_pool); idx->d_pool = NULL;}
    if(idx->d_bvh) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_bvh); idx->d_bvh = NULL;}
    if(idx->d_tiles) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_tiles); idx->d_tiles = NULL;}
    /* Free host-side persistent buffers kept alive across drifts.
     * mymalloc uses LIFO stack discipline; free in reverse-allocation order:
     * orig_extent (allocated last in build) -> h_pool -> h_tiles -> h_bvh
     * (h_bvh allocated first by build_tile_bvh, but build_tile_bvh may have
     * been re-called via sidx_rebuild_bvh_inplace which freed-then-allocated,
     * so h_bvh is on top of the stack at this point in normal flow). */
    /* Persistent host-side buffers live in Kokkos::HostSpace (heap-allocated,
     * not mymalloc) so they don't pin the LIFO stack across other transient
     * mymalloc'd state. Free order doesn't matter. */
    if(idx->h_bvh)   { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_bvh);   idx->h_bvh = NULL; }
    if(idx->h_tile_orig_max_extent) { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_tile_orig_max_extent); idx->h_tile_orig_max_extent = NULL; }
    if(idx->h_tiles) { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_tiles); idx->h_tiles = NULL; }
    if(idx->h_pool)  { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_pool);  idx->h_pool = NULL; }
    if(idx->h_pos_buf) { Kokkos::kokkos_free<Kokkos::HostSpace>(idx->h_pos_buf); idx->h_pos_buf = NULL; }
    if(idx->d_pos_buf) { Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(idx->d_pos_buf); idx->d_pos_buf = NULL; }
    idx->h_bvh_nnodes = 0;
    idx->num_pool = 0;
    idx->valid = 0;
}


void gpu_ngb_list_build(struct particle_data *P_shared, int num_total,
                        int *active_indices_host, int num_active,
                        int search_mode, int type_bitmask,
                        gpu_neighbor_list_t *gnl,
                        gpu_spatial_index_t *cached_idx,
                        double search_radius_factor,
                        const double *search_radii_host,
                        const double *source_positions_host,
                        const char *caller_label)
{
    gnl->num_active = num_active;
    double t_entry = my_second(); /* DIAG: entry */

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
        gnl->d_active  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
        gnl->offsets   = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
        gnl->offsets[0] = 0;
        gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(sizeof(int));
        gnl->total_pairs = 0;
        if(ThisTask == 0 && !gizmo_ngb_diag_quiet()) {
            printf("[DIAG_NGL caller=%s tbm=0x%x N=0 Ntot=%d pairs=0 ovflw=0 sidx_cached=%d earlyout=1] "
                   "total=%.3f\n",
                   caller_label ? caller_label : "?", type_bitmask, num_total,
                   idx_for_stubs ? 1 : 0, timediff(t_entry, my_second()));
            fflush(stdout);
        }
        return;
    }

    /* Use cached spatial index if available, otherwise build fresh.
     * If caller provided a cached_idx but it's not yet built, populate it (this
     * enables persistent caching across calls — caller controls invalidation). */
    gpu_spatial_index_t local_idx = {NULL, NULL, NULL, 0, 0, {0}, {0}, {0}, NULL, 0, 0};
    gpu_spatial_index_t *idx;
    /* Invalidate cached SIDX if num_total changed (ghost exchange redo, particle creation, etc.).
     * The compact_xyzh and pool arrays were sized for the old count; accessing beyond them is UB. */
    if(cached_idx && cached_idx->valid && cached_idx->num_total != num_total)
        gpu_spatial_index_free(cached_idx);
    if(cached_idx && cached_idx->valid) {
        idx = cached_idx;
    } else if(cached_idx) {
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, cached_idx, caller_label);
        idx = cached_idx;
    } else {
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, &local_idx, caller_label);
        idx = &local_idx;
    }
    double t_after_sidx = my_second(); /* DIAG: after SIDX (re)use decision */

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

    /* Refresh the h component of the compact array. Two modes:
     *  - g_dirty_all: full-pool parallel_for(num_total, ...) — same as legacy
     *    behavior. Pays ~1.1-1.2s per call on fire_m11i 12.4M pool (UVM fault
     *    latency on P_shared.KernelRadius reads).
     *  - dirty list: parallel_for(n_dirty, ...) reading indices via a staged
     *    device buffer. ~ms per call when n_dirty is the active set. This is
     *    the gas-side win that step (3) of the codex_dialogue plan delivers.
     * Skipped entirely when neither mode has anything to refresh. */
    double t_refresh_launch_in = 0, t_refresh_launch_out = 0, t_refresh_fence_out = 0; /* DIAG */
    int did_refresh = 0;
    if(cached_idx && cached_idx->valid && (g_dirty_all || !g_dirty_list.empty())) {
        did_refresh = 1;
        float *compact = idx->d_compact_xyzh;
        t_refresh_launch_in = my_second();
        if(g_dirty_all) {
            Kokkos::parallel_for("compact_h_refresh_all", num_total, KOKKOS_LAMBDA(int i) {
                compact[i*4+3] = (float)P_shared[i].KernelRadius;
            });
        } else {
            int n_dirty = (int)g_dirty_list.size();
            int *d_dirty = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(n_dirty * sizeof(int));
            {
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    hv(g_dirty_list.data(), n_dirty);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    dv(d_dirty, n_dirty);
                Kokkos::deep_copy(dv, hv);
            }
            Kokkos::parallel_for("compact_h_refresh_idx", n_dirty, KOKKOS_LAMBDA(int k) {
                int i = d_dirty[k];
                compact[i*4+3] = (float)P_shared[i].KernelRadius;
            });
            Kokkos::fence();
            Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_dirty);
        }
        t_refresh_launch_out = my_second();
        Kokkos::fence();
        t_refresh_fence_out = my_second();
        g_dirty_clear_(); /* compact_xyzh now in sync with arena KernelRadius */
    }
    double t_after_refresh = my_second(); /* DIAG */

    /* Active indices: always re-uploaded (changes per call) */
    gnl->d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(gnl->d_active, active_indices_host, num_active * sizeof(int));

    /* Optional explicit per-active search radii (for loops with a different kernel
       than P[i].KernelRadius, e.g. KernelRadiusDM or AGS_Hsml). NULL → use P[i].KernelRadius. */
    double *d_radii = NULL;
    if(search_radii_host) {
        d_radii = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(double));
        memcpy(d_radii, search_radii_host, num_active * sizeof(double));
    }

    /* Optional explicit per-active source positions (for sources not backed by
       P[] entries, e.g. arbitrary grid cells). NULL → read pos from P[active[aa]].
       Layout in caller's array: source_positions_host[aa*3 + k] for axis k. */
    double *d_source_pos = NULL;
    if(source_positions_host) {
        d_source_pos = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * 3 * sizeof(double));
        memcpy(d_source_pos, source_positions_host, num_active * 3 * sizeof(double));
    }

    /* Allocate CSR offsets */
    gnl->offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));

    /* Per-particle scratchpad for fused single-pass build. Each active particle
     * gets a fixed stride (NGL_SCRATCH_STRIDE) of int slots in d_scratch; the BVH
     * walk emits j-indices directly there, with the count tracked in d_counts.
     * After scan + compact we transcribe into the dense CSR neighbors[] array.
     * Memory: stride * num_active * 4 bytes (e.g. 256 * 2M * 4 = 2GB). */
    constexpr int NGL_SCRATCH_STRIDE = 512;
    double t_alloc0 = my_second(); /* DIAG */
    /* size_t cast required: int * int overflows for num_active > ~4.19M (e.g. fire_m11i
     * gas-per-rank), wrapping to negative int → ~UINT64_MAX after promotion to size_t. */
    size_t na_safe = (size_t)((num_active > 0) ? num_active : 1);
    int *d_scratch = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(na_safe * (size_t)NGL_SCRATCH_STRIDE * sizeof(int));
    int *d_counts  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(na_safe * sizeof(int));
    double t_alloc1 = my_second(); /* DIAG: end of scratch alloc */

    /* DIAG: drain any prior async GPU work so subsequent fence times only this kernel */
    Kokkos::fence();
    double t_drain_done = my_second();
    double t_nl0 = t_drain_done; /* DIAG: start of GPU passes */
    double t_fused_launch_in = 0, t_fused_launch_out = 0; /* DIAG */
    double t_noop_launch_in = 0, t_noop_launch_out = 0, t_noop_fence_out = 0; /* DIAG: empty-kernel probe */

    /* Empty-kernel probe: distinguishes Kokkos/CUDA fence floor (platform overhead)
     * from actual GPU work.  If noop_fnc ≈ fused_fnc the 1.4s is the fence floor;
     * if noop_fnc ≈ µs the 1.4s is real kernel work (e.g. UVM page migration). */
    t_noop_launch_in = my_second();
    Kokkos::parallel_for("noop_probe", 1, KOKKOS_LAMBDA(int) {});
    t_noop_launch_out = my_second();
    Kokkos::fence();
    t_noop_fence_out = my_second();

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
        const double *radii = d_radii;
        const double *src_pos = d_source_pos;
        const float *compact_xyzh = gnl->d_compact_xyzh;
        t_fused_launch_in = my_second();
        Kokkos::parallel_for("ngb_fused", num_active, KOKKOS_LAMBDA(int aa) {
            int pf[3] = {pf0, pf1, pf2};
            double bs[3] = {bs0, bs1, bs2};
            double bh[3] = {bh0, bh1, bh2};
            int i = active[aa];
            double h_i = (radii ? radii[aa] : (double)compact_xyzh[i*4+3]) * sr_fac;
            double pos_i[3];
            if(src_pos) { pos_i[0] = src_pos[aa*3+0]; pos_i[1] = src_pos[aa*3+1]; pos_i[2] = src_pos[aa*3+2]; }
            else        { pos_i[0] = (double)compact_xyzh[i*4+0]; pos_i[1] = (double)compact_xyzh[i*4+1]; pos_i[2] = (double)compact_xyzh[i*4+2]; }
            int cnt = search_neighbors_sfc_gpu(compact_xyzh, pos_i, h_i,
                                               tiles, ntiles, pool, smode,
                                               bvh, bvh_root,
                                               &scratch[(size_t)aa * NGL_SCRATCH_STRIDE],
                                               NGL_SCRATCH_STRIDE,
                                               pf, bs, bh);
            counts[aa] = cnt;
        });
        t_fused_launch_out = my_second();
        Kokkos::fence();

        /* MICROBENCHMARK PROBE — fires once per process when GIZMO_NGB_MICROBENCH=1
         * env var is set and we hit a small-N cached call. Re-launches the SAME
         * ngb_fused kernel 20 times back-to-back with per-iteration fence timing,
         * plus 20 empty-kernel "noop" launches for comparison. Settles whether
         * the ~1.45s fused floor is per-call (every iteration ~1.45s) or
         * first-touch/JIT (only iteration 0 slow). */
        static bool g_microbench_done = false;
        if(!g_microbench_done && getenv("GIZMO_NGB_MICROBENCH")
           && cached_idx && cached_idx->valid && num_active <= 16) {
            g_microbench_done = true;
            printf("[NGB_MICROBENCH] caller=%s N=%d sidx_cached=1 search_mode=%d — running 20 repeated fused + 20 noop launches\n",
                   caller_label, num_active, search_mode);
            printf("[NGB_MICROBENCH] box_sizes=(%g,%g,%g) ntiles=%d periodic=(%d,%d,%d)\n",
                   bs0, bs1, bs2, ntiles, pf0, pf1, pf2);
            fflush(stdout);

            /* Tile-bbox stats: pull tile bboxes back to host and report extent
             * distribution per axis. If extents are ~box/cube_root(ntiles) the
             * SFC tiling is healthy; if extents ~ box_size the SFC sort failed
             * and every bbox overlap test will pass regardless of query. */
            {
                std::vector<sfc_tile_t> h_tiles(ntiles);
                Kokkos::View<sfc_tile_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    hv(h_tiles.data(), ntiles);
                Kokkos::View<sfc_tile_t*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    dv(tiles, ntiles);
                Kokkos::deep_copy(hv, dv);
                double bs_axes[3] = {bs0, bs1, bs2};
                for(int k = 0; k < 3; k++) {
                    std::vector<double> ext(ntiles);
                    for(int t = 0; t < ntiles; t++) ext[t] = h_tiles[t].hi[k] - h_tiles[t].lo[k];
                    std::sort(ext.begin(), ext.end());
                    double mn = ext.front(), md = ext[ntiles/2], mx = ext.back();
                    double sum = 0; for(double v : ext) sum += v;
                    double mean = sum / ntiles;
                    printf("[NGB_MICROBENCH] tile_bbox axis=%d  min=%.6g  median=%.6g  mean=%.6g  max=%.6g  box=%.6g  max/box=%.4f\n",
                           k, mn, md, mean, mx, bs_axes[k], (bs_axes[k]>0 ? mx/bs_axes[k] : 0.0));
                }
                /* Per-active query stats: h_i, h_i/box, position. */
                {
                    std::vector<int> h_active(num_active);
                    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ha(h_active.data(), num_active);
                    Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> da(active, num_active);
                    Kokkos::deep_copy(ha, da);
                    std::vector<float> h_compact(4 * (size_t)num_total);
                    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hc(h_compact.data(), 4 * num_total);
                    Kokkos::View<const float*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dc(compact_xyzh, 4 * num_total);
                    Kokkos::deep_copy(hc, dc);
                    for(int aa = 0; aa < num_active; aa++) {
                        int i = h_active[aa];
                        float x = h_compact[i*4+0], y = h_compact[i*4+1], z = h_compact[i*4+2], h = h_compact[i*4+3];
                        printf("[NGB_MICROBENCH]   active aa=%2d i=%d  pos=(%.4g,%.4g,%.4g)  h=%.4g  h/box=(%.4f,%.4f,%.4f)\n",
                               aa, i, x, y, z, h, h/bs0, h/bs1, h/bs2);
                    }
                }
                fflush(stdout);
            }
            for(int rep = 0; rep < 20; rep++) {
                double tA = my_second();
                Kokkos::parallel_for("ngb_fused_probe", num_active, KOKKOS_LAMBDA(int aa) {
                    int pf[3] = {pf0, pf1, pf2};
                    double bs[3] = {bs0, bs1, bs2};
                    double bh[3] = {bh0, bh1, bh2};
                    int i = active[aa];
                    double h_i = (radii ? radii[aa] : (double)compact_xyzh[i*4+3]) * sr_fac;
                    double pos_i[3];
                    if(src_pos) { pos_i[0] = src_pos[aa*3+0]; pos_i[1] = src_pos[aa*3+1]; pos_i[2] = src_pos[aa*3+2]; }
                    else        { pos_i[0] = (double)compact_xyzh[i*4+0]; pos_i[1] = (double)compact_xyzh[i*4+1]; pos_i[2] = (double)compact_xyzh[i*4+2]; }
                    int cnt = search_neighbors_sfc_gpu(compact_xyzh, pos_i, h_i,
                                                       tiles, ntiles, pool, smode,
                                                       bvh, bvh_root,
                                                       &scratch[(size_t)aa * NGL_SCRATCH_STRIDE],
                                                       NGL_SCRATCH_STRIDE,
                                                       pf, bs, bh);
                    counts[aa] = cnt;
                });
                Kokkos::fence();
                double tB = my_second();
                double tC = my_second();
                Kokkos::parallel_for("noop_probe2", 1, KOKKOS_LAMBDA(int) {});
                Kokkos::fence();
                double tD = my_second();
                printf("[NGB_MICROBENCH] rep=%2d fused=%.4fs noop=%.6fs\n",
                       rep, tB - tA, tD - tC);
                fflush(stdout);
            }
            /* One additional launch with BVH visit counters wired up. Tells us
             * whether the 1.48s/call is BVH traversal pathology, candidate-test
             * floods, or something else (e.g. underutilization-induced latency
             * with reasonable visit counts). */
            int *d_nodes  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)num_active * sizeof(int));
            int *d_tilesV = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)num_active * sizeof(int));
            int *d_tested = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)num_active * sizeof(int));
            int *d_accept = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)num_active * sizeof(int));
            {
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_n(d_nodes,  num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_t(d_tilesV, num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_x(d_tested, num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_a(d_accept, num_active);
                Kokkos::deep_copy(v_n, 0); Kokkos::deep_copy(v_t, 0); Kokkos::deep_copy(v_x, 0); Kokkos::deep_copy(v_a, 0);
            }
            double tA = my_second();
            Kokkos::parallel_for("ngb_fused_count", num_active, KOKKOS_LAMBDA(int aa) {
                int pf[3] = {pf0, pf1, pf2};
                double bs[3] = {bs0, bs1, bs2};
                double bh[3] = {bh0, bh1, bh2};
                int i = active[aa];
                double h_i = (radii ? radii[aa] : (double)compact_xyzh[i*4+3]) * sr_fac;
                double pos_i[3];
                if(src_pos) { pos_i[0] = src_pos[aa*3+0]; pos_i[1] = src_pos[aa*3+1]; pos_i[2] = src_pos[aa*3+2]; }
                else        { pos_i[0] = (double)compact_xyzh[i*4+0]; pos_i[1] = (double)compact_xyzh[i*4+1]; pos_i[2] = (double)compact_xyzh[i*4+2]; }
                int n_nodes = 0, n_tiles = 0, n_test = 0, n_acc = 0;
                (void) search_neighbors_sfc_gpu(compact_xyzh, pos_i, h_i,
                                                tiles, ntiles, pool, smode,
                                                bvh, bvh_root,
                                                &scratch[(size_t)aa * NGL_SCRATCH_STRIDE],
                                                NGL_SCRATCH_STRIDE,
                                                pf, bs, bh,
                                                &n_nodes, &n_tiles, &n_test, &n_acc);
                d_nodes[aa]  = n_nodes;
                d_tilesV[aa] = n_tiles;
                d_tested[aa] = n_test;
                d_accept[aa] = n_acc;
            });
            Kokkos::fence();
            double tB = my_second();
            /* Pull counters back to host */
            std::vector<int> h_nodes(num_active), h_tiles(num_active), h_test(num_active), h_acc(num_active);
            {
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv_n(h_nodes.data(), num_active);
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv_t(h_tiles.data(), num_active);
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv_x(h_test.data(),  num_active);
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv_a(h_acc.data(),   num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv_n(d_nodes,  num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv_t(d_tilesV, num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv_x(d_tested, num_active);
                Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv_a(d_accept, num_active);
                Kokkos::deep_copy(hv_n, dv_n); Kokkos::deep_copy(hv_t, dv_t);
                Kokkos::deep_copy(hv_x, dv_x); Kokkos::deep_copy(hv_a, dv_a);
            }
            auto stats = [&](const std::vector<int> &v, const char *name) {
                long sum = 0; int mn = v[0], mx = v[0];
                for(int i = 0; i < num_active; i++) { sum += v[i]; if(v[i]<mn) mn=v[i]; if(v[i]>mx) mx=v[i]; }
                printf("[NGB_MICROBENCH]   %-22s sum=%ld  min=%d  max=%d  mean=%.1f\n",
                       name, sum, mn, mx, (double)sum/num_active);
            };
            printf("[NGB_MICROBENCH] counter-pass fused=%.4fs (ntiles=%d, pool_total ~ scaled to active set)\n",
                   tB - tA, ntiles);
            stats(h_nodes, "bvh_nodes_visited");
            stats(h_tiles, "bvh_tiles_visited");
            stats(h_test,  "candidates_tested");
            stats(h_acc,   "candidates_accepted");
            Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_accept);
            Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_tested);
            Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_tilesV);
            Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_nodes);
            printf("[NGB_MICROBENCH] done\n"); fflush(stdout);
        }
    }
    double t_nl1 = my_second(); /* DIAG: after fused BVH pass */

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
    int total = 0;
    {
        int *counts = d_counts;
        int *offsets = gnl->offsets;
        Kokkos::parallel_scan("ngb_offsets_scan", num_active,
            KOKKOS_LAMBDA(int aa, int &update, const bool final) {
                int v = counts[aa];
                if(final) offsets[aa] = update;
                update += v;
            }, total);
        Kokkos::fence();
    }
    gnl->offsets[num_active] = total;
    gnl->total_pairs = total;
    double t_nl2 = my_second(); /* DIAG: after GPU prefix scan */

    /* Allocate CSR neighbors array */
    gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(((total > 0) ? total : 1) * sizeof(int));

    /* Compact: copy from per-particle scratchpad into dense CSR neighbors[]. */
    double t_compact_launch_in = 0, t_compact_launch_out = 0; /* DIAG */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *scratch = d_scratch;
        int *counts = d_counts;
        int *offsets = gnl->offsets;
        int *neighbors = gnl->neighbors;
        int ntiles = gnl->ntiles;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];
        double sr_fac = search_radius_factor;
        const double *radii = d_radii;
        const double *src_pos = d_source_pos;
        const float *compact_xyzh = gnl->d_compact_xyzh;
        t_compact_launch_in = my_second();
        Kokkos::parallel_for("ngb_compact", num_active, KOKKOS_LAMBDA(int aa) {
            int n = counts[aa];
            int dst = offsets[aa];
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
                search_neighbors_sfc_gpu(compact_xyzh, pos_i, h_i,
                                         tiles, ntiles, pool, smode,
                                         bvh, bvh_root,
                                         &neighbors[dst], 0x7fffffff,
                                         pf, bs, bh);
            }
        });
        t_compact_launch_out = my_second();
        Kokkos::fence();
    }
    double t_nl3 = my_second(); /* DIAG: after compact pass */

    /* Free temporaries */
    double t_free0 = my_second(); /* DIAG */
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_scratch);
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(d_counts);
    if(d_radii) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    if(d_source_pos) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_source_pos);
    double t_free1 = my_second(); /* DIAG: end of scratch free */

    int sidx_cached_now = (cached_idx && cached_idx->valid) ? 1 : 0;
    if(ThisTask == 0 && !gizmo_ngb_diag_quiet()) { /* DIAG: NGP build fine-grained breakdown */
        /* Sub-times for compact_h_refresh (when run): launch return vs trailing fence */
        double refresh_launch = did_refresh ? timediff(t_refresh_launch_in, t_refresh_launch_out) : 0;
        double refresh_fence  = did_refresh ? timediff(t_refresh_launch_out, t_refresh_fence_out) : 0;
        /* Drain fence cost (prior async GPU work) */
        double drain_fence    = timediff(t_alloc1, t_drain_done);
        /* Fused launch return vs trailing fence */
        double fused_launch   = (num_active > 0) ? timediff(t_fused_launch_in, t_fused_launch_out) : 0;
        double fused_fence    = (num_active > 0) ? timediff(t_fused_launch_out, t_nl1) : 0;
        /* Compact launch return vs trailing fence */
        double compact_launch = (num_active > 0) ? timediff(t_compact_launch_in, t_compact_launch_out) : 0;
        double compact_fence  = (num_active > 0) ? timediff(t_compact_launch_out, t_nl3) : 0;
        double noop_launch = timediff(t_noop_launch_in, t_noop_launch_out);
        double noop_fence  = timediff(t_noop_launch_out, t_noop_fence_out);
        printf("[DIAG_NGL caller=%s tbm=0x%x N=%d Ntot=%d pairs=%d ovflw=%d sidx_cached=%d] "
               "sidx_dec=%.3f refresh_lnch=%.3f refresh_fnc=%.3f drain=%.3f "
               "noop_lnch=%.4f noop_fnc=%.4f "
               "fused_lnch=%.3f fused_fnc=%.3f scan=%.3f "
               "compact_lnch=%.3f compact_fnc=%.3f free=%.3f total=%.3f\n",
               caller_label ? caller_label : "?", type_bitmask, num_active, num_total,
               total, overflow_count, sidx_cached_now,
               timediff(t_entry, t_after_sidx),
               refresh_launch, refresh_fence,
               drain_fence,
               noop_launch, noop_fence,
               fused_launch, fused_fence,
               timediff(t_nl1, t_nl2),
               compact_launch, compact_fence,
               timediff(t_free0, t_free1),
               timediff(t_entry, t_free1));
        fflush(stdout);
    }
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
        h(host_dest, gnl->total_pairs);
    Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        d(gnl->neighbors, gnl->total_pairs);
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
    /* Phase 7 Round A1: sub-bucket the 1.6s gradient_prep_symlist cost.
     * env-gated via GIZMO_STEP_PHASES; no-op when off. */
    double t_sym_start = my_second();

    /* Use the per-step particle arena instead of a dedicated full-NumPart memcpy.
     * The arena's fast path is a no-op when valid (e.g. when gradient/hydro
     * already populated it earlier in the step), avoiding ~2.3s of redundant
     * P-copy on small-N symlist invocations.  Pass the global CellP so the
     * arena's "valid" state remains consistent across mixed P/CellP consumers. */
    gpu_particles_arena_set_site("gpu_build_symmetric_neighbor_list");
    gpu_particles_arena_acquire(num_total, P_host, CellP);
    struct particle_data *P_shared = gpu_particles_arena_P();
    double t_sym_arena = my_second();

    /* Build GPU CSR — share gas-only SIDX with density via the step-persistent cache */
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, active_indices, num_active,
                       NGB_SEARCH_SYMMETRIC, 1 /* gas only */, &gpu_nl, gpu_step_sidx_ptr(),
                       search_radius_factor, NULL, NULL, "symlist");
    double t_sym_ngb = my_second();

    /* Copy CSR into mymalloc neighbor_list_t */
    out->num_active = num_active;
    out->total_pairs = gpu_nl.total_pairs;
    out->offsets = (int *) mymalloc("ngb_offsets", (num_active + 1) * sizeof(int));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (gpu_nl.total_pairs > 0 ? gpu_nl.total_pairs : 1) * sizeof(int));
    /* gpu_nl.offsets is SharedSpace (UVM) → host memcpy is fine.
     * gpu_nl.neighbors is DEVICE_SPACE (CudaSpace) → must use deep_copy, not host memcpy. */
    memcpy(out->offsets, gpu_nl.offsets, (num_active + 1) * sizeof(int));
    if(gpu_nl.total_pairs > 0) {
        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_neighbors(out->neighbors, gpu_nl.total_pairs);
        Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_neighbors(gpu_nl.neighbors, gpu_nl.total_pairs);
        Kokkos::deep_copy(h_neighbors, d_neighbors);
    }
    double t_sym_csr = my_second();

    /* Free GPU temporaries (keep tiles/BVH alive — owned by g_step_sidx).
     * Arena is intentionally not released — subsequent gradient/hydro callers
     * benefit from the fast-path skip. */
    gpu_ngb_list_free(&gpu_nl, gpu_step_sidx_ptr());
    double t_sym_free = my_second();

    gizmo_step_phase_record("symlist_arena_acquire", timediff(t_sym_start, t_sym_arena));
    gizmo_step_phase_record("symlist_ngb_build",     timediff(t_sym_arena, t_sym_ngb));
    gizmo_step_phase_record("symlist_csr_copy",      timediff(t_sym_ngb,   t_sym_csr));
    gizmo_step_phase_record("symlist_free",          timediff(t_sym_csr,   t_sym_free));
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
    out->offsets = (int *) mymalloc("ngb_offsets", (num_active + 1) * sizeof(int));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (gpu_nl.total_pairs > 0 ? gpu_nl.total_pairs : 1) * sizeof(int));
    /* See gpu_build_symmetric_neighbor_list for why neighbors needs deep_copy. */
    memcpy(out->offsets, gpu_nl.offsets, (num_active + 1) * sizeof(int));
    if(gpu_nl.total_pairs > 0) {
        Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_neighbors(out->neighbors, gpu_nl.total_pairs);
        Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_neighbors(gpu_nl.neighbors, gpu_nl.total_pairs);
        Kokkos::deep_copy(h_neighbors, d_neighbors);
    }

    /* Free GPU temporaries.  Arena is intentionally retained for subsequent callers. */
    gpu_ngb_list_free(&gpu_nl, NULL);
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */
GPU_ALL_SYNC_FUNC(ngb)

