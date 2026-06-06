/* sfc_tiles.h — SFC-ordered tile spatial index for neighbor finding and ghost exchange.
 *
 * Tiles are contiguous groups of ~TILE_TARGET_SIZE particles in Peano-Hilbert
 * order. Each tile stores a bounding box and max kernel radius (hmax).
 * This provides finer spatial granularity than the top-level tree leaves
 * (~100-1000 leaves) while being cheaper than per-particle checks.
 *
 * Used for:
 *   1. Neighbor finding: tile overlap check replaces uniform grid cell-list
 *   2. Ghost exchange: tile-level overlap criterion replaces leaf-level
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef SFC_TILES_H
#define SFC_TILES_H

#include "neighbor_list.h"

#define TILE_TARGET_SIZE 64  /* particles per tile (tunable) */
#define TILE_BVH_STACK_SIZE 64  /* traversal stack depth (log2(ntiles) + margin) */

/* Per-type hmax: stored as fixed-size vector indexed by P[].Type ∈ [0,5].
 * Dimension matches the GHOST_TYPE_* bitmask (bit k ↔ Type k). hmax_by_type[t]
 * is the max KernelRadius across particles of Type t in the subtree; 0 if
 * none. Opener computes hmax_eff = max over t in supply_mask of hmax_by_type[t]
 * — this avoids the scalar-hmax contamination where a single DM particle
 * with KernelRadius=582 poisons every search whose supply mask doesn't even
 * include DM. (See ghost_exchange_roadmap_2026-05-05.md "per-type hmax".) */
#define TILE_NUM_PTYPES 6

/* Axis periodicity flags for tile-based neighbor search and ghost exchange */
#if defined(BOX_PERIODIC) && !defined(BOX_REFLECT_X) && !defined(BOX_OUTFLOW_X)
#define TILE_PERIODIC_X 1
#else
#define TILE_PERIODIC_X 0
#endif
#if defined(BOX_PERIODIC) && !defined(BOX_REFLECT_Y) && !defined(BOX_OUTFLOW_Y)
#define TILE_PERIODIC_Y 1
#else
#define TILE_PERIODIC_Y 0
#endif
#if defined(BOX_PERIODIC) && !defined(BOX_REFLECT_Z) && !defined(BOX_OUTFLOW_Z)
#define TILE_PERIODIC_Z 1
#else
#define TILE_PERIODIC_Z 0
#endif

struct sfc_tile_t {
    int first;                        /* first particle index (into pool index array) */
    int count;                        /* number of particles in this tile */
    double lo[3];                     /* bounding box lower corner */
    double hi[3];                     /* bounding box upper corner */
    double hmax;                      /* max kernel radius in tile (any type) — kept for back-compat callers */
    double hmax_by_type[TILE_NUM_PTYPES]; /* max kernel radius PER TYPE (Bucket roadmap) */
};

/* BVH node over SFC tiles. Built bottom-up from SFC-sorted tiles via
 * recursive midpoint subdivision. Enables O(log ntiles) spatial pruning
 * for neighbor search, critical for zoom-in sims with h/box ~ 10^-6. */
struct tile_bvh_node_t {
    double lo[3], hi[3];                  /* bounding box of subtree */
    double hmax;                          /* max kernel radius in subtree (any type) — back-compat */
    double hmax_by_type[TILE_NUM_PTYPES]; /* max kernel radius PER TYPE in subtree */
    int left, right;                      /* children: >= 0 = internal node index, < 0 = -(tile_index+1) for leaf */
};

/* Build BVH over tiles. Returns number of internal nodes.
 * bvh_out: allocated array of internal nodes (caller frees via myfree).
 * Root is at index (num_internal_nodes - 1). */
int build_tile_bvh(sfc_tile_t *tiles, int ntiles, tile_bvh_node_t **bvh_out);

/* Build SFC tiles from particles in P[0..num_total-1].
 * Only includes particles matching type_bitmask with Mass > 0.
 * Particles must already be Peano-Hilbert sorted (from peano_hilbert_order()).
 *
 * Returns number of tiles. Caller must call free_sfc_tiles() when done.
 * pool_indices_out: allocated array of particle indices in SFC order (pool only)
 * tiles_out: allocated array of tiles
 * num_pool_out: number of particles in the pool
 */
int build_sfc_tiles(struct particle_data *P, int num_total,
                    int type_bitmask, int target_tile_size,
                    sfc_tile_t **tiles_out, int **pool_indices_out,
                    int *num_pool_out);

void free_sfc_tiles(sfc_tile_t *tiles, int *pool_indices);

/* Build a CSR neighbor list using SFC tiles as the spatial index.
 * Same interface and output format as build_neighbor_list(). */
void build_neighbor_list_sfc(struct particle_data *P, struct gas_cell_data *CellP,
                             int num_total, int *active_indices, int num_active,
                             int search_mode, int type_bitmask,
                             neighbor_list_t *out);

#endif /* SFC_TILES_H */
