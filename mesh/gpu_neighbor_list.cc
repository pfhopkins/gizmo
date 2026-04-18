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
#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"

#include "sfc_tiles.h"
#include "sfc_tiles_functions.h"
#include "gpu_neighbor_list.h"
#include "neighbor_list.h"

/* TILE_PERIODIC_X/Y/Z defined in sfc_tiles.h (included via gpu_neighbor_list.h) */

#if defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)

void gpu_spatial_index_build(struct particle_data *P_shared, int num_total,
                             int type_bitmask, gpu_spatial_index_t *idx)
{
    /* Capture periodicity parameters */
    idx->periodic_flags[0] = TILE_PERIODIC_X;
    idx->periodic_flags[1] = TILE_PERIODIC_Y;
    idx->periodic_flags[2] = TILE_PERIODIC_Z;
    idx->box_sizes[0] = boxSize_X; idx->box_sizes[1] = boxSize_Y; idx->box_sizes[2] = boxSize_Z;
    idx->box_halves[0] = boxHalf_X; idx->box_halves[1] = boxHalf_Y; idx->box_halves[2] = boxHalf_Z;

    /* Build SFC tiles + BVH on CPU */
    sfc_tile_t *h_tiles;
    int *h_pool;
    int num_pool;
    int ntiles = build_sfc_tiles(P_shared, num_total, type_bitmask, TILE_TARGET_SIZE,
                                 &h_tiles, &h_pool, &num_pool);
    idx->ntiles = ntiles;

    tile_bvh_node_t *h_bvh;
    int bvh_nnodes = build_tile_bvh(h_tiles, ntiles, &h_bvh);
    idx->bvh_root = bvh_nnodes - 1;

    /* Copy to SharedSpace */
    int bvh_size = (2 * ntiles - 1);
    if(bvh_size < 1) bvh_size = 1;
    idx->d_tiles = (sfc_tile_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(ntiles * sizeof(sfc_tile_t));
    idx->d_bvh = (tile_bvh_node_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bvh_size * sizeof(tile_bvh_node_t));
    idx->d_pool = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_pool > 0) ? num_pool : 1) * sizeof(int));

    memcpy(idx->d_tiles, h_tiles, ntiles * sizeof(sfc_tile_t));
    memcpy(idx->d_bvh, h_bvh, bvh_nnodes * sizeof(tile_bvh_node_t));
    memcpy(idx->d_pool, h_pool, num_pool * sizeof(int));

    myfree(h_bvh);
    myfree(h_tiles);
    myfree(h_pool);
    idx->valid = 1;
}

void gpu_spatial_index_free(gpu_spatial_index_t *idx)
{
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
                        const double *search_radii_host)
{
    gnl->num_active = num_active;

    /* Use cached spatial index if available, otherwise build fresh */
    gpu_spatial_index_t local_idx = {NULL, NULL, NULL, 0, 0, {0}, {0}, {0}, 0};
    gpu_spatial_index_t *idx;
    if(cached_idx && cached_idx->valid) {
        idx = cached_idx;
    } else {
        gpu_spatial_index_build(P_shared, num_total, type_bitmask, &local_idx);
        idx = &local_idx;
    }

    /* Copy spatial index pointers to gnl for use by free */
    gnl->d_tiles = idx->d_tiles;
    gnl->d_bvh = idx->d_bvh;
    gnl->d_pool = idx->d_pool;
    gnl->ntiles = idx->ntiles;
    gnl->bvh_root = idx->bvh_root;
    memcpy(gnl->periodic_flags, idx->periodic_flags, 3 * sizeof(int));
    memcpy(gnl->box_sizes, idx->box_sizes, 3 * sizeof(double));
    memcpy(gnl->box_halves, idx->box_halves, 3 * sizeof(double));

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

    /* Allocate CSR offsets */
    gnl->offsets = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((num_active + 1) * sizeof(int));

    /* Pass 1: count neighbors per active particle */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *offsets = gnl->offsets;
        int ntiles = gnl->ntiles;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];

        double sr_fac = search_radius_factor;
        const double *radii = d_radii; /* NULL → fall back to P[i].KernelRadius */
        Kokkos::parallel_for("ngb_count", num_active, KOKKOS_LAMBDA(int aa) {
            int pf[3] = {pf0, pf1, pf2};
            double bs[3] = {bs0, bs1, bs2};
            double bh[3] = {bh0, bh1, bh2};
            int i = active[aa];
            double h_i = (radii ? radii[aa] : P_shared[i].KernelRadius) * sr_fac;
            int cnt = search_neighbors_sfc_gpu(P_shared, i, h_i,
                                               tiles, ntiles, pool, smode,
                                               bvh, bvh_root, NULL, pf, bs, bh);
            offsets[aa] = cnt;
        });
        Kokkos::fence();
    }

    /* Exclusive prefix scan on host */
    int total = 0;
    for(int aa = 0; aa < num_active; aa++) {
        int cnt = gnl->offsets[aa];
        gnl->offsets[aa] = total;
        total += cnt;
    }
    gnl->offsets[num_active] = total;
    gnl->total_pairs = total;

    /* Allocate CSR neighbors array */
    gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((total > 0) ? total : 1) * sizeof(int));

    /* Pass 2: fill neighbor indices */
    {
        sfc_tile_t *tiles = gnl->d_tiles;
        tile_bvh_node_t *bvh = gnl->d_bvh;
        int *pool = gnl->d_pool;
        int *active = gnl->d_active;
        int *offsets = gnl->offsets;
        int *neighbors = gnl->neighbors;
        int ntiles = gnl->ntiles;
        int bvh_root = gnl->bvh_root;
        int smode = search_mode;
        int pf0 = gnl->periodic_flags[0], pf1 = gnl->periodic_flags[1], pf2 = gnl->periodic_flags[2];
        double bs0 = gnl->box_sizes[0], bs1 = gnl->box_sizes[1], bs2 = gnl->box_sizes[2];
        double bh0 = gnl->box_halves[0], bh1 = gnl->box_halves[1], bh2 = gnl->box_halves[2];

        double sr_fac = search_radius_factor;
        const double *radii = d_radii; /* NULL → fall back to P[i].KernelRadius */
        Kokkos::parallel_for("ngb_fill", num_active, KOKKOS_LAMBDA(int aa) {
            int pf[3] = {pf0, pf1, pf2};
            double bs[3] = {bs0, bs1, bs2};
            double bh[3] = {bh0, bh1, bh2};
            int i = active[aa];
            double h_i = (radii ? radii[aa] : P_shared[i].KernelRadius) * sr_fac;
            search_neighbors_sfc_gpu(P_shared, i, h_i,
                                     tiles, ntiles, pool, smode,
                                     bvh, bvh_root, &neighbors[offsets[aa]],
                                     pf, bs, bh);
        });
        Kokkos::fence();
    }

    /* Free the temporary radii mirror (if we allocated one) */
    if(d_radii) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
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

    /* Build GPU CSR */
    gpu_neighbor_list_t gpu_nl;
    gpu_ngb_list_build(P_shared, num_total, active_indices, num_active,
                       NGB_SEARCH_SYMMETRIC, 1 /* gas only */, &gpu_nl, NULL,
                       search_radius_factor);

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

#else /* !OPENMP_GPU_OFFLOAD || !GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */

/* Stubs when GPU neighbor list is not available */
void gpu_spatial_index_build(struct particle_data *, int, int, gpu_spatial_index_t *) {}
void gpu_spatial_index_free(gpu_spatial_index_t *) {}
void gpu_ngb_list_build(struct particle_data *, int, int *, int, int, int,
                        gpu_neighbor_list_t *, gpu_spatial_index_t *, double, const double *) {}
void gpu_ngb_list_free(gpu_neighbor_list_t *, gpu_spatial_index_t *) {}
void gpu_build_symmetric_neighbor_list(struct particle_data *, int, int *, int, neighbor_list_t *, double) {}
void gpu_build_cross_type_neighbor_list(struct particle_data *, int, int *, int, const double *, int, int, neighbor_list_t *) {}
void gizmo_gpu_sync_all_ngb(struct global_data_all_processes *p) { (void)p; }

#endif /* OPENMP_GPU_OFFLOAD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */
