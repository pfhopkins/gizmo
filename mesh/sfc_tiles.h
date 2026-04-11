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

struct sfc_tile_t {
    int first;          /* first particle index (into pool index array) */
    int count;          /* number of particles in this tile */
    double lo[3];       /* bounding box lower corner */
    double hi[3];       /* bounding box upper corner */
    double hmax;        /* max kernel radius in tile */
};

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
