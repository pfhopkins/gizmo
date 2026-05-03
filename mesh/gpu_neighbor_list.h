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

/* GPU-resident neighbor list: CSR arrays.
   offsets: SharedSpace (host writes offsets[num_active]=total after scan).
   neighbors: DeviceSpace (GPU HBM on CUDA; never host-accessed directly). */
struct gpu_neighbor_list_t {
    int *offsets;       /* [num_active+1] in SharedSpace */
    int *neighbors;     /* [total_pairs] in DeviceSpace (GPU HBM — no UVM fault) */
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

    /* Compact position+h array for BVH traversal cache efficiency (points into
       spatial index memory; do NOT free from gnl — owned by gpu_spatial_index_t). */
    float *d_compact_xyzh;
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
    /* Compact position+h array: d_compact_xyzh[i*4+0..3] = x,y,z,h for particle i.
       Fits ~32MB for 2M particles vs ~800MB for full P[], dramatically improving
       GPU BVH traversal cache efficiency. Built in gpu_spatial_index_build. */
    float *d_compact_xyzh;
    int num_total;  /* particle count when built; mismatch → invalidate */
    int valid;  /* 1 if built and usable */
};


/* Build spatial index (tiles + BVH) on CPU, copy to SharedSpace.
   P_shared must be in SharedSpace (managed memory).
   caller_label: short tag printed in DIAG_SIDX so we can attribute rebuilds. */
void gpu_spatial_index_build(struct particle_data *P_shared, int num_total,
                             int type_bitmask, gpu_spatial_index_t *idx,
                             const char *caller_label = "?");

/* Free spatial index SharedSpace memory. */
void gpu_spatial_index_free(gpu_spatial_index_t *idx);

/* Module-level persistent SIDX for gas-only (type_bitmask=1) neighbor builds.
 * Shared across density rounds + symlist within a single step so the BVH +
 * compact_xyzh build only happens once per step instead of 4× per step.
 * Must be invalidated after drift (positions change) — caller (run.cc) calls
 * gpu_step_sidx_invalidate() after find_next_sync_point_and_drift(). */
gpu_spatial_index_t *gpu_step_sidx_ptr(void);
void gpu_step_sidx_invalidate(void);

/* Mark the compact_xyzh h-field as out of sync with P[].KernelRadius.
 * Call from any code that mutates KernelRadius (host or arena P) BEFORE the
 * next gpu_ngb_list_build that wants the change reflected.  The next build
 * with sidx_cached=1 will then re-run compact_h_refresh; otherwise refresh is
 * skipped (saving ~1.1-1.2s on fire_m11i 12.4M-particle pool).
 * Safe to over-call (false positive = slightly slower); fail-correct default
 * is dirty=1, so missing a real mutation is the only fail-incorrect path. */
void gpu_compact_xyzh_mark_h_dirty(void);

/* Build GPU-accelerated CSR neighbor list.
   If cached_idx is non-NULL and valid, reuses its tiles+BVH.
   Otherwise builds a fresh spatial index internally.
   P_shared must be accessible from GPU (SharedSpace or managed memory).
   active_indices_host: host-side array of source identifiers, size num_active.
     Default mode (source_positions_host == NULL): these are P[] indices and
     source positions are read as P[active[aa]].Pos.
     Override mode (source_positions_host != NULL): these are caller-defined
     opaque IDs; pass any sentinel (e.g. 0..num_active-1) since the kernel
     reads positions from source_positions_host instead.
   search_mode: NGB_SEARCH_ONEWAY or NGB_SEARCH_SYMMETRIC.
   type_bitmask: which particle types to include in the search pool (j-side).
   search_radius_factor: multiplier on per-source radius (default 1.0).
   search_radii_host: optional per-active-source explicit search radii
     (size num_active). NULL → use P[active[aa]].KernelRadius * search_radius_factor.
     REQUIRED when source_positions_host is non-NULL (override sources have no
     P[] entry to fall back to).
   source_positions_host: optional per-active-source position array (size
     num_active * 3, doubles, layout pos[aa*3+k] for axis k). NULL → use
     P[active[aa]].Pos (current behavior). Non-NULL → arbitrary source
     positions decoupled from any P[] index (e.g. TURB_DRIVING_SPECTRUMGRID
     grid cell centers). */
void gpu_ngb_list_build(struct particle_data *P_shared, int num_total,
                        int *active_indices_host, int num_active,
                        int search_mode, int type_bitmask,
                        gpu_neighbor_list_t *gnl,
                        gpu_spatial_index_t *cached_idx,
                        double search_radius_factor = 1.0,
                        const double *search_radii_host = NULL,
                        const double *source_positions_host = NULL,
                        const char *caller_label = "?");

/* Free CSR arrays + active indices. Does NOT free tiles/BVH/pool if they
   belong to the cached spatial index (use gpu_spatial_index_free for those). */
void gpu_ngb_list_free(gpu_neighbor_list_t *gnl, gpu_spatial_index_t *cached_idx);

/* Copy gnl->neighbors (DEVICE_SPACE / CudaSpace) into a caller-allocated host
   buffer. Use when host code needs to index gnl.neighbors[] directly (e.g.
   per-source CPU loops in radfb_local, merge_split). host_dest must hold at
   least gnl->total_pairs ints; no-op when total_pairs <= 0. */
void gpu_ngb_copy_neighbors_to_host(const gpu_neighbor_list_t *gnl, int *host_dest);

/* Cross-type high-level wrapper: i-list is caller-supplied active indices of
   any type(s); j-side is filtered by j_type_bitmask. Caller supplies explicit
   per-active search radii (for loops whose kernel isn't P[i].KernelRadius —
   e.g. KernelRadiusDM, AGS_Hsml). Returns a neighbor_list_t in the mymalloc
   format used by the existing symlist API. */
struct neighbor_list_t; /* forward decl from mesh/neighbor_list.h */
void gpu_build_cross_type_neighbor_list(struct particle_data *P_host, int num_total,
                                        int *i_active_indices, int num_active,
                                        const double *i_search_radii_host,
                                        int j_type_bitmask, int search_mode,
                                        neighbor_list_t *out);

#endif /* GPU_NEIGHBOR_LIST_H */
