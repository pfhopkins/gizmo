/* gpu_neighbor_list.h — GPU-accelerated neighbor list construction.
 *
 * Provides gpu_ngb_list_build(): builds a CSR neighbor list using
 * SFC tiles + BVH spatial index with GPU-parallel neighbor search.
 * The tile/BVH construction runs on CPU; the per-particle search
 * runs on GPU via Kokkos parallel_for.
 *
 * Reusable for any neighbor list type: density (one-way, gas-only),
 * symmetric (max(h_i,h_j)), or cross-type (different type_bitmask).
 * The spatial index can optionally be cached and reused across calls
 * with the same particle positions.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GPU_NEIGHBOR_LIST_H
#define GPU_NEIGHBOR_LIST_H

#include "sfc_tiles.h"

/* GPU-resident neighbor list: CSR arrays in SharedSpace */
struct gpu_neighbor_list_t {
    int *offsets;       /* [num_active+1] in SharedSpace */
    int *neighbors;     /* [total_pairs] in SharedSpace */
    int num_active;
    int total_pairs;

    /* Device-resident copies of spatial index data */
    sfc_tile_t *d_tiles;
    tile_bvh_node_t *d_bvh;
    int *d_pool;
    int *d_active;
    int ntiles;
    int bvh_root;

    /* Periodicity parameters (copied to avoid global access in kernels) */
    int periodic_flags[3];
    double box_sizes[3];
    double box_halves[3];
};


/* Cached spatial index: tiles + BVH in SharedSpace.
   Depends only on particle positions (not KernelRadius), so can be
   reused across multiple neighbor list builds with different search radii. */
struct gpu_spatial_index_t {
    sfc_tile_t *d_tiles;
    tile_bvh_node_t *d_bvh;
    int *d_pool;
    int ntiles;
    int bvh_root;
    int periodic_flags[3];
    double box_sizes[3];
    double box_halves[3];
    int valid;  /* 1 if built and usable */
};


/* Build spatial index (tiles + BVH) on CPU, copy to SharedSpace.
   P_shared must be in SharedSpace (managed memory). */
void gpu_spatial_index_build(struct particle_data *P_shared, int num_total,
                             int type_bitmask, gpu_spatial_index_t *idx);

/* Free spatial index SharedSpace memory. */
void gpu_spatial_index_free(gpu_spatial_index_t *idx);

/* Build GPU-accelerated CSR neighbor list.
   If cached_idx is non-NULL and valid, reuses its tiles+BVH.
   Otherwise builds a fresh spatial index internally.
   P_shared must be accessible from GPU (SharedSpace or managed memory).
   active_indices_host: host-side array of particle indices to search FROM.
   search_mode: NGB_SEARCH_ONEWAY or NGB_SEARCH_SYMMETRIC.
   type_bitmask: which particle types to include in the search pool. */
void gpu_ngb_list_build(struct particle_data *P_shared, int num_total,
                        int *active_indices_host, int num_active,
                        int search_mode, int type_bitmask,
                        gpu_neighbor_list_t *gnl,
                        gpu_spatial_index_t *cached_idx);

/* Free CSR arrays + active indices. Does NOT free tiles/BVH/pool if they
   belong to the cached spatial index (use gpu_spatial_index_free for those). */
void gpu_ngb_list_free(gpu_neighbor_list_t *gnl, gpu_spatial_index_t *cached_idx);

#endif /* GPU_NEIGHBOR_LIST_H */
