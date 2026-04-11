/* sfc_tiles.cc — SFC-ordered tile construction and tile-based neighbor search.
 *
 * Particles are already Peano-Hilbert sorted from domain decomposition.
 * We group them into tiles of ~TILE_TARGET_SIZE particles and compute
 * per-tile bounding boxes and hmax values. Neighbor search then checks
 * tile-level overlap before doing pairwise distance checks within tiles.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "sfc_tiles.h"


/* Axis periodicity flags (matching neighbor_list.cc) */
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


int build_sfc_tiles(struct particle_data *P, int num_total,
                    int type_bitmask, int target_tile_size,
                    sfc_tile_t **tiles_out, int **pool_indices_out,
                    int *num_pool_out)
{
    /* Step 1: Build pool index array (particles matching type/mass filter, in P[] order).
       Since P[] is already SFC-sorted, pool_indices preserves SFC order. */
    int num_pool = 0;
    int i;
    for(i = 0; i < num_total; i++) {
        if(!((1 << P[i].Type) & type_bitmask)) continue;
        if(P[i].Mass <= 0) continue;
        num_pool++;
    }

    int *pool = (int *) mymalloc("sfc_pool", (num_pool > 0 ? num_pool : 1) * sizeof(int));
    int p = 0;
    for(i = 0; i < num_total; i++) {
        if(!((1 << P[i].Type) & type_bitmask)) continue;
        if(P[i].Mass <= 0) continue;
        pool[p++] = i;
    }

    /* Step 2: Compute number of tiles */
    int ntiles = (num_pool + target_tile_size - 1) / target_tile_size;
    if(ntiles < 1) ntiles = 1;

    sfc_tile_t *tiles = (sfc_tile_t *) mymalloc("sfc_tiles", ntiles * sizeof(sfc_tile_t));

    /* Step 3: Build tiles — single pass over pool */
    int t;
    for(t = 0; t < ntiles; t++)
    {
        int start = t * target_tile_size;
        int count = target_tile_size;
        if(start + count > num_pool) count = num_pool - start;

        tiles[t].first = start;
        tiles[t].count = count;
        tiles[t].hmax = 0;

        /* Initialize bbox from first particle */
        int j0 = pool[start];
        tiles[t].lo[0] = tiles[t].hi[0] = P[j0].Pos[0];
        tiles[t].lo[1] = tiles[t].hi[1] = P[j0].Pos[1];
        tiles[t].lo[2] = tiles[t].hi[2] = P[j0].Pos[2];
        if(P[j0].KernelRadius > tiles[t].hmax) tiles[t].hmax = P[j0].KernelRadius;

        /* Expand bbox with remaining particles */
        int s;
        for(s = 1; s < count; s++)
        {
            int j = pool[start + s];
            int k;
            for(k = 0; k < 3; k++) {
                if(P[j].Pos[k] < tiles[t].lo[k]) tiles[t].lo[k] = P[j].Pos[k];
                if(P[j].Pos[k] > tiles[t].hi[k]) tiles[t].hi[k] = P[j].Pos[k];
            }
            if(P[j].KernelRadius > tiles[t].hmax) tiles[t].hmax = P[j].KernelRadius;
        }
    }

    *tiles_out = tiles;
    *pool_indices_out = pool;
    *num_pool_out = num_pool;
    return ntiles;
}


void free_sfc_tiles(sfc_tile_t *tiles, int *pool_indices)
{
    /* Free in reverse mymalloc order: tiles allocated after pool */
    if(tiles) myfree(tiles);
    if(pool_indices) myfree(pool_indices);
}


/* Check if a search sphere overlaps a tile's bounding box (with periodic wrapping).
 * Returns 1 if overlap, 0 otherwise.
 * Uses center+halfwidth approach for correct periodic distance to AABB. */
static int tile_overlaps_sphere(sfc_tile_t *tile, double pos[3], double search_r,
                                double search_r2)
{
    double dist2 = 0;
    int k;
    for(k = 0; k < 3; k++)
    {
        int is_periodic = (k == 0) ? TILE_PERIODIC_X : ((k == 1) ? TILE_PERIODIC_Y : TILE_PERIODIC_Z);
        double bsize = (k == 0) ? boxSize_X : ((k == 1) ? boxSize_Y : boxSize_Z);

        /* Distance from pos to center of tile bbox, with periodic wrapping */
        double center = 0.5 * (tile->lo[k] + tile->hi[k]);
        double halfwidth = 0.5 * (tile->hi[k] - tile->lo[k]);
        double dx = fabs(pos[k] - center);
        if(is_periodic && dx > 0.5 * bsize) dx = bsize - dx;

        /* Gap = distance from pos to nearest edge of bbox */
        double gap = dx - halfwidth;
        if(gap <= 0) continue; /* pos is inside bbox on this axis */

        if(gap > search_r) return 0; /* early rejection */
        dist2 += gap * gap;
    }
    return (dist2 < search_r2);
}


/* Search for neighbors of particle i within all overlapping tiles.
 * Returns count; if store_neighbors != NULL, writes indices there. */
static int search_neighbors_sfc(struct particle_data *P, int i, double h_i,
                                sfc_tile_t *tiles, int ntiles,
                                int *pool, int search_mode,
                                int *store_neighbors)
{
    double h2_i = h_i * h_i;
    MyDouble xtmp = 0; /* required by NGB_PERIODIC_BOX_LONG_* macros */
    int count = 0;

    int t;
    for(t = 0; t < ntiles; t++)
    {
        /* Determine search radius for this tile */
        double search_r;
        if(search_mode == NGB_SEARCH_ONEWAY) {
            search_r = h_i;
        } else {
            search_r = DMAX(h_i, tiles[t].hmax);
        }
        double search_r2 = search_r * search_r;

        /* Check tile-level overlap before iterating particles */
        double pos_i[3] = {P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]};
        if(!tile_overlaps_sphere(&tiles[t], pos_i, search_r, search_r2)) continue;

        /* Iterate particles in this tile */
        int s;
        for(s = 0; s < tiles[t].count; s++)
        {
            int j = pool[tiles[t].first + s];

            /* Distance check using same macros as tree walk */
            double dx_raw = P[i].Pos[0] - P[j].Pos[0];
            double dy_raw = P[i].Pos[1] - P[j].Pos[1];
            double dz_raw = P[i].Pos[2] - P[j].Pos[2];
            double adx = NGB_PERIODIC_BOX_LONG_X(dx_raw, dy_raw, dz_raw, 1);
            double ady = NGB_PERIODIC_BOX_LONG_Y(dx_raw, dy_raw, dz_raw, 1);
            double adz = NGB_PERIODIC_BOX_LONG_Z(dx_raw, dy_raw, dz_raw, 1);

            double pair_search_r2;
            if(search_mode == NGB_SEARCH_ONEWAY) {
                pair_search_r2 = h2_i;
            } else {
                double h_max = DMAX(h_i, P[j].KernelRadius);
                pair_search_r2 = h_max * h_max;
            }

            if(adx > h_i && (search_mode == NGB_SEARCH_ONEWAY || adx * adx > pair_search_r2)) continue;
            double r2 = adx * adx + ady * ady + adz * adz;
            if(r2 < pair_search_r2) {
                if(store_neighbors) store_neighbors[count] = j;
                count++;
            }
        }
    }
    return count;
}


void build_neighbor_list_sfc(struct particle_data *P, struct gas_cell_data *CellP,
                             int num_total, int *active_indices, int num_active,
                             int search_mode, int type_bitmask,
                             neighbor_list_t *out)
{
    out->num_active = num_active;

    /* Build SFC tiles */
    sfc_tile_t *tiles;
    int *pool;
    int num_pool;
    int ntiles = build_sfc_tiles(P, num_total, type_bitmask, TILE_TARGET_SIZE,
                                 &tiles, &pool, &num_pool);

    if(ThisTask == 0) {
        double max_h = 0;
        int t;
        for(t = 0; t < ntiles; t++) if(tiles[t].hmax > max_h) max_h = tiles[t].hmax;
        PRINT_STATUS("SFC-tile neighbor finder: %d tiles (%d particles/tile avg), max_h=%.4g, pool=%d particles",
                     ntiles, num_pool / (ntiles > 0 ? ntiles : 1), max_h, num_pool);
    }

    if(num_pool == 0 || ntiles == 0) {
        out->total_pairs = 0;
        myfree(tiles); myfree(pool);
        out->offsets = (int *) mymalloc("ngb_offsets", (num_active + 1) * sizeof(int));
        memset(out->offsets, 0, (num_active + 1) * sizeof(int));
        out->neighbors = NULL;
        return;
    }

    /* Pass 1: count neighbors per active particle */
    int *counts = (int *) mymalloc("sfc_counts", num_active * sizeof(int));
    int a;
    for(a = 0; a < num_active; a++) {
        int i = active_indices[a];
        counts[a] = search_neighbors_sfc(P, i, P[i].KernelRadius,
                        tiles, ntiles, pool, search_mode, NULL);
    }

    int total_pairs = 0;
    for(a = 0; a < num_active; a++) total_pairs += counts[a];

    /* Free pass-1 temporaries in reverse mymalloc order before allocating output.
       Stack is: pool, tiles, counts (top). Output arrays must be at stack base. */
    myfree(counts);
    myfree(tiles);
    myfree(pool);

    /* Allocate output CSR (these persist after return) */
    out->total_pairs = total_pairs;
    out->offsets = (int *) mymalloc("ngb_offsets", (num_active + 1) * sizeof(int));
    out->neighbors = (int *) mymalloc("ngb_neighbors", (total_pairs > 0 ? total_pairs : 1) * sizeof(int));

    /* Pass 2: rebuild tiles and fill neighbor indices */
    ntiles = build_sfc_tiles(P, num_total, type_bitmask, TILE_TARGET_SIZE,
                             &tiles, &pool, &num_pool);

    out->offsets[0] = 0;
    for(a = 0; a < num_active; a++) {
        int i = active_indices[a];
        int n = search_neighbors_sfc(P, i, P[i].KernelRadius,
                        tiles, ntiles, pool, search_mode,
                        &out->neighbors[out->offsets[a]]);
        out->offsets[a + 1] = out->offsets[a] + n;
    }

    /* Cleanup pass-2 temporaries (reverse order) */
    myfree(tiles);
    myfree(pool);
}
