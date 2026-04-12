/* sfc_tiles_functions.h — GPU-callable SFC-tile neighbor search functions.
 *
 * Contains: bbox_overlaps_sphere(), check_tile_particles_gpu(),
 *   search_neighbors_sfc_gpu().
 *
 * These are the per-particle search functions extracted from sfc_tiles.cc.
 * On CPU: sfc_tiles.cc includes this with KOKKOS_INLINE_FUNCTION redefined
 *   to empty (or just uses its own static copies directly).
 * On GPU: the Kokkos kernel includes this with KOKKOS_INLINE_FUNCTION as
 *   __device__ __host__ inline, making these callable from parallel_for.
 *
 * The functions take explicit pointers to tiles/BVH/pool arrays (no globals)
 * and explicit periodicity parameters, making them pure functions suitable
 * for GPU execution.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef SFC_TILES_FUNCTIONS_H
#define SFC_TILES_FUNCTIONS_H

#include "sfc_tiles.h"

/* Check if a search sphere overlaps an axis-aligned bounding box (with periodic wrapping).
 * Returns 1 if overlap, 0 otherwise.
 * Uses center+halfwidth approach for correct periodic distance to AABB.
 * periodic_flags[k]: 1 if axis k is periodic, 0 otherwise.
 * box_sizes[k]: box size along axis k (only used if periodic). */
KOKKOS_INLINE_FUNCTION
int bbox_overlaps_sphere_gpu(double box_lo[3], double box_hi[3],
                             double pos[3], double search_r, double search_r2,
                             const int periodic_flags[3], const double box_sizes[3])
{
    double dist2 = 0;
    for(int k = 0; k < 3; k++)
    {
        double bsize = box_sizes[k];

        /* Distance from pos to center of bbox, with periodic wrapping */
        double center = 0.5 * (box_lo[k] + box_hi[k]);
        double halfwidth = 0.5 * (box_hi[k] - box_lo[k]);
        double dx = fabs(pos[k] - center);
        if(periodic_flags[k] && dx > 0.5 * bsize) dx = bsize - dx;

        /* Gap = distance from pos to nearest edge of bbox */
        double gap = dx - halfwidth;
        if(gap <= 0) continue; /* pos is inside bbox on this axis */

        if(gap > search_r) return 0; /* early rejection */
        dist2 += gap * gap;
    }
    return (dist2 < search_r2);
}


/* Process a single tile's particles against particle i. Returns neighbor count.
 * Explicit periodic wrapping via box_sizes/box_halves arrays. */
KOKKOS_INLINE_FUNCTION
int check_tile_particles_gpu(struct particle_data *P, int i, double h_i, double h2_i,
                             sfc_tile_t *tile, int *pool, int search_mode,
                             int *store_neighbors, int count,
                             const double box_sizes[3], const double box_halves[3],
                             const int periodic_flags[3])
{
    for(int s = 0; s < tile->count; s++)
    {
        int j = pool[tile->first + s];
        double dx = P[i].Pos[0] - P[j].Pos[0];
        double dy = P[i].Pos[1] - P[j].Pos[1];
        double dz = P[i].Pos[2] - P[j].Pos[2];

        /* periodic wrapping */
        if(periodic_flags[0]) { if(dx > box_halves[0]) dx -= box_sizes[0]; else if(dx < -box_halves[0]) dx += box_sizes[0]; }
        if(periodic_flags[1]) { if(dy > box_halves[1]) dy -= box_sizes[1]; else if(dy < -box_halves[1]) dy += box_sizes[1]; }
        if(periodic_flags[2]) { if(dz > box_halves[2]) dz -= box_sizes[2]; else if(dz < -box_halves[2]) dz += box_sizes[2]; }

        double adx = fabs(dx);

        double pair_search_r2;
        if(search_mode == NGB_SEARCH_ONEWAY) {
            pair_search_r2 = h2_i;
        } else {
            double h_max = (h_i > P[j].KernelRadius) ? h_i : P[j].KernelRadius;
            pair_search_r2 = h_max * h_max;
        }

        if(adx > h_i && (search_mode == NGB_SEARCH_ONEWAY || adx * adx > pair_search_r2)) continue;
        double r2 = dx * dx + dy * dy + dz * dz;
        if(r2 < pair_search_r2) {
            if(store_neighbors) store_neighbors[count] = j;
            count++;
        }
    }
    return count;
}


/* Search for neighbors of particle i using BVH traversal over tiles.
 * Iterative stack-based traversal (GPU-friendly, no recursion).
 * Returns count; if store_neighbors != NULL, writes indices there. */
KOKKOS_INLINE_FUNCTION
int search_neighbors_sfc_gpu(struct particle_data *P, int i, double h_i,
                             sfc_tile_t *tiles, int ntiles,
                             int *pool, int search_mode,
                             tile_bvh_node_t *bvh, int bvh_root,
                             int *store_neighbors,
                             const int periodic_flags[3],
                             const double box_sizes[3],
                             const double box_halves[3])
{
    double h2_i = h_i * h_i;
    double pos_i[3] = {P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]};
    int count = 0;

    /* Stack-based iterative BVH traversal */
    int stack[TILE_BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = bvh_root;

    while(sp > 0)
    {
        int node_idx = stack[--sp];
        tile_bvh_node_t *node = &bvh[node_idx];

        /* Compute search radius for this node: for SYMMETRIC, use node's subtree hmax */
        double search_r = (search_mode == NGB_SEARCH_ONEWAY) ? h_i : ((h_i > node->hmax) ? h_i : node->hmax);
        double search_r2 = search_r * search_r;

        /* Check if node's bbox (expanded by search_r) overlaps particle i */
        if(!bbox_overlaps_sphere_gpu(node->lo, node->hi, pos_i, search_r, search_r2,
                                     periodic_flags, box_sizes)) continue;

        /* Check if leaf */
        if(node->left < 0)
        {
            /* Leaf node: process the tile's particles */
            int tile_idx = -(node->left + 1);
            count = check_tile_particles_gpu(P, i, h_i, h2_i, &tiles[tile_idx], pool, search_mode,
                                             store_neighbors, count,
                                             box_sizes, box_halves, periodic_flags);
        }
        else
        {
            /* Internal node: push children onto stack */
            if(sp + 2 > TILE_BVH_STACK_SIZE) {
                /* Stack overflow — should not happen with STACK_SIZE=64 */
                break;
            }
            stack[sp++] = node->left;
            stack[sp++] = node->right;
        }
    }
    return count;
}

#endif /* SFC_TILES_FUNCTIONS_H */
