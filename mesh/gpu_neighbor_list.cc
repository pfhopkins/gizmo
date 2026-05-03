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
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
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

gpu_spatial_index_t *gpu_step_sidx_ptr(void) { return &g_step_sidx; }

/* Tracks whether g_step_sidx's compact_xyzh.h field is in sync with
 * P[].KernelRadius.  Set to 1 (dirty) when KernelRadius is mutated;
 * cleared to 0 when compact_h_refresh runs (or when the SIDX is freshly
 * built — the build phase fills compact_xyzh from current h).
 *
 * The compact_h_refresh kernel iterates over num_total particles writing
 * h values to compact_xyzh; for fire_m11i (12.4M particles) this is
 * ~1.1-1.2s of UVM page-state churn per call.  Within a single step, only
 * density mutates KernelRadius (inflate / restore / iter sync); subsequent
 * gas-only callers (mech_fb, hii_fb, symlist) just read it, so they can
 * legitimately skip the refresh.
 *
 * Default 1 (force first refresh) so missed mark-dirty sites are
 * fail-correct (slightly slower) rather than fail-incorrect (stale h
 * → missed neighbors). */
static int g_compact_xyzh_h_dirty = 1;

void gpu_compact_xyzh_mark_h_dirty(void) { g_compact_xyzh_h_dirty = 1; }

void gpu_step_sidx_invalidate(void)
{
    if(g_step_sidx.valid) gpu_spatial_index_free(&g_step_sidx);
    g_compact_xyzh_h_dirty = 1; /* compact_xyzh is gone; next build/refresh syncs */
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

    myfree(h_bvh);
    myfree(h_tiles);
    myfree(h_pool);
    idx->num_total = num_total;
    idx->valid = 1;
    g_compact_xyzh_h_dirty = 0; /* fresh build seeded compact_xyzh from current h */

    if(ThisTask == 0) { /* DIAG: spatial index build breakdown — remove after profiling */
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
        if(ThisTask == 0) {
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

    /* Refresh the h component of the compact array from current P_shared[i].KernelRadius.
     * Skipped when g_compact_xyzh_h_dirty==0 — i.e. compact_xyzh.h is already in
     * sync with P_shared[i].KernelRadius from the previous refresh or build.
     * Saves ~1.1-1.2s per call on fire_m11i (12.4M-particle parallel_for over UVM). */
    double t_refresh_launch_in = 0, t_refresh_launch_out = 0, t_refresh_fence_out = 0; /* DIAG */
    int did_refresh = 0;
    if(cached_idx && cached_idx->valid && g_compact_xyzh_h_dirty) {
        did_refresh = 1;
        float *compact = idx->d_compact_xyzh;
        t_refresh_launch_in = my_second();
        Kokkos::parallel_for("compact_h_refresh", num_total, KOKKOS_LAMBDA(int i) {
            compact[i*4+3] = (float)P_shared[i].KernelRadius;
        });
        t_refresh_launch_out = my_second();
        Kokkos::fence();
        t_refresh_fence_out = my_second();
        g_compact_xyzh_h_dirty = 0; /* compact_xyzh now in sync with P_shared.KernelRadius */
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
            printf("[NGB_MICROBENCH] caller=%s N=%d sidx_cached=1 — running 20 repeated fused + 20 noop launches\n",
                   caller_label, num_active);
            fflush(stdout);
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
    if(ThisTask == 0) { /* DIAG: NGP build fine-grained breakdown */
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
    /* Use the per-step particle arena instead of a dedicated full-NumPart memcpy.
     * The arena's fast path is a no-op when valid (e.g. when gradient/hydro
     * already populated it earlier in the step), avoiding ~2.3s of redundant
     * P-copy on small-N symlist invocations.  Pass the global CellP so the
     * arena's "valid" state remains consistent across mixed P/CellP consumers. */
    gpu_particles_arena_acquire(num_total, P_host, CellP);
    struct particle_data *P_shared = gpu_particles_arena_P();

    /* Build GPU CSR — share gas-only SIDX with density via the step-persistent cache */
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, active_indices, num_active,
                       NGB_SEARCH_SYMMETRIC, 1 /* gas only */, &gpu_nl, gpu_step_sidx_ptr(),
                       search_radius_factor, NULL, NULL, "symlist");

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

