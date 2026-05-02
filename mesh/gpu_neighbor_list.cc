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
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
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

void gpu_step_sidx_invalidate(void)
{
    if(g_step_sidx.valid) gpu_spatial_index_free(&g_step_sidx);
}


void gpu_spatial_index_build(struct particle_data *P_shared, int num_total,
                             int type_bitmask, gpu_spatial_index_t *idx)
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

    /* Copy to SharedSpace */
    int bvh_size = (2 * ntiles - 1);
    if(bvh_size < 1) bvh_size = 1;
    idx->d_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(ntiles * sizeof(sfc_tile_t));
    idx->d_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bvh_size * sizeof(tile_bvh_node_t));
    idx->d_pool = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_pool > 0) ? num_pool : 1) * sizeof(int));

    memcpy(idx->d_tiles, h_tiles, ntiles * sizeof(sfc_tile_t));
    memcpy(idx->d_bvh, h_bvh, bvh_nnodes * sizeof(tile_bvh_node_t));
    memcpy(idx->d_pool, h_pool, num_pool * sizeof(int));
    double t_si3 = my_second(); /* DIAG: after memcpy to SharedSpace */

    /* Build compact float4 position+h array for cache-efficient GPU BVH traversal.
       32MB for 2M particles vs 800MB for full P_shared — fits in H100 L2 (50MB),
       eliminating the random-access cache misses that dominate the GPU count/fill passes. */
    idx->d_compact_xyzh = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * 4 * sizeof(float));
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

    if(ThisTask == 0) { /* DIAG: spatial index build breakdown — remove after profiling */
        printf("[DIAG_SIDX ntiles=%d pool=%d] sfc_tiles=%.3f bvh_build=%.3f memcpy=%.3f compact=%.3f total=%.3f\n",
               ntiles, num_pool,
               timediff(t_si0, t_si1), timediff(t_si1, t_si2), timediff(t_si2, t_si3),
               timediff(t_si3, t_si4), timediff(t_si0, t_si4));
        fflush(stdout);
    }
}

void gpu_spatial_index_free(gpu_spatial_index_t *idx)
{
    if(idx->d_compact_xyzh) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(idx->d_compact_xyzh); idx->d_compact_xyzh = NULL;}
    if(idx->d_pool) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(idx->d_pool); idx->d_pool = NULL;}
    if(idx->d_bvh) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(idx->d_bvh); idx->d_bvh = NULL;}
    if(idx->d_tiles) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(idx->d_tiles); idx->d_tiles = NULL;}
    idx->valid = 0;
}


void gpu_ngb_list_build(struct particle_data *P_shared, int num_total,
                        int *active_indices_host, int num_active,
                        int search_mode, int type_bitmask,
                        gpu_neighbor_list_t *gnl,
                        gpu_spatial_index_t *cached_idx,
                        double search_radius_factor,
                        const double *search_radii_host,
                        const double *source_positions_host)
{
    gnl->num_active = num_active;

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
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, cached_idx);
        idx = cached_idx;
    } else {
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, &local_idx);
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

    /* Refresh the h component of the compact array from current P_shared[i].KernelRadius.
       Positions are stable while the spatial index is cached, but h changes between
       calls (density h-iteration mutates KernelRadius). Without this refresh, source
       h_i (when search_radii_host==NULL) and j-side h_j (SYMMETRIC mode) read stale
       values, breaking convergence. Cost: ~1ms for 2M particles. */
    if(cached_idx && cached_idx->valid) {
        float *compact = idx->d_compact_xyzh;
        Kokkos::parallel_for("compact_h_refresh", num_total, KOKKOS_LAMBDA(int i) {
            compact[i*4+3] = (float)P_shared[i].KernelRadius;
        });
        Kokkos::fence();
    }

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
    int *d_scratch = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(na_safe * (size_t)NGL_SCRATCH_STRIDE * sizeof(int));
    int *d_counts  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(na_safe * sizeof(int));
    double t_alloc1 = my_second(); /* DIAG: end of scratch alloc */

    double t_nl0 = my_second(); /* DIAG: start of GPU passes */
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
        Kokkos::fence();
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
    gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((total > 0) ? total : 1) * sizeof(int));

    /* Compact: copy from per-particle scratchpad into dense CSR neighbors[].
       Overflow particles (count > stride) re-walk the BVH with unbounded store
       directly into neighbors[offsets[aa]]; this keeps the fast path fast while
       remaining correct for arbitrarily clumpy problems. */
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
        Kokkos::fence();
    }
    double t_nl3 = my_second(); /* DIAG: after compact pass */

    /* Free temporaries */
    double t_free0 = my_second(); /* DIAG */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_scratch);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_counts);
    if(d_radii) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    if(d_source_pos) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_source_pos);
    double t_free1 = my_second(); /* DIAG: end of scratch free */

    int sidx_cached = (cached_idx && cached_idx->valid) ? 1 : 0;
    if(ThisTask == 0) { /* DIAG: NGP build phase breakdown — remove after profiling */
        printf("[DIAG_NGL N=%d pairs=%d ovflw=%d sidx_cached=%d] alloc=%.3f gpu_fused=%.3f gpu_scan=%.3f gpu_compact=%.3f free=%.3f total=%.3f\n",
               num_active, total, overflow_count, sidx_cached,
               timediff(t_alloc0, t_alloc1),
               timediff(t_nl0, t_nl1), timediff(t_nl1, t_nl2), timediff(t_nl2, t_nl3),
               timediff(t_free0, t_free1),
               timediff(t_alloc0, t_free1));
        fflush(stdout);
    }
}


void gpu_ngb_list_free(gpu_neighbor_list_t *gnl, gpu_spatial_index_t *cached_idx)
{
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->offsets);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_active);
    /* Only free tiles/BVH/pool if they were NOT from the cached index */
    if(!cached_idx || !cached_idx->valid ||
       gnl->d_tiles != cached_idx->d_tiles) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_pool);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_bvh);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_tiles);
    }
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
    /* Copy P to SharedSpace for GPU kernel access */
    struct particle_data *P_shared = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
    memcpy(P_shared, P_host, num_total * sizeof(struct particle_data));

    /* Build GPU CSR — share gas-only SIDX with density via the step-persistent cache */
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, active_indices, num_active,
                       NGB_SEARCH_SYMMETRIC, 1 /* gas only */, &gpu_nl, gpu_step_sidx_ptr(),
                       search_radius_factor);

    /* Copy CSR into mymalloc neighbor_list_t */
    out->num_active = num_active;
    out->total_pairs = gpu_nl.total_pairs;
    out->offsets = (int *) mymalloc("ngb_offsets", (num_active + 1) * sizeof(int));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (gpu_nl.total_pairs > 0 ? gpu_nl.total_pairs : 1) * sizeof(int));
    memcpy(out->offsets, gpu_nl.offsets, (num_active + 1) * sizeof(int));
    memcpy(out->neighbors, gpu_nl.neighbors, gpu_nl.total_pairs * sizeof(int));

    /* Free GPU temporaries (keep tiles/BVH alive — owned by g_step_sidx) */
    gpu_ngb_list_free(&gpu_nl, gpu_step_sidx_ptr());
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_shared);
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
    /* Copy P to SharedSpace for GPU kernel access */
    struct particle_data *P_shared = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
    memcpy(P_shared, P_host, num_total * sizeof(struct particle_data));

    /* Build GPU CSR with explicit per-i radii and j-side type filter */
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, i_active_indices, num_active,
                       search_mode, j_type_bitmask, &gpu_nl, NULL,
                       1.0 /* search_radius_factor */, i_search_radii_host);

    /* Copy CSR into mymalloc neighbor_list_t */
    out->num_active = num_active;
    out->total_pairs = gpu_nl.total_pairs;
    out->offsets = (int *) mymalloc("ngb_offsets", (num_active + 1) * sizeof(int));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (gpu_nl.total_pairs > 0 ? gpu_nl.total_pairs : 1) * sizeof(int));
    memcpy(out->offsets, gpu_nl.offsets, (num_active + 1) * sizeof(int));
    memcpy(out->neighbors, gpu_nl.neighbors, gpu_nl.total_pairs * sizeof(int));

    /* Free GPU temporaries */
    gpu_ngb_list_free(&gpu_nl, NULL);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_shared);
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */
GPU_ALL_SYNC_FUNC(ngb)

