/* sfc_tiles.cc — SFC-ordered tile construction (the spatial index itself).
 *
 * Particles are already Peano-Hilbert sorted from domain decomposition.
 * We group them into tiles of ~TILE_TARGET_SIZE particles and compute
 * per-tile bounding boxes and hmax values, then a BVH over those tiles.
 * The traversal that consumes them — tile-level overlap test followed by
 * pairwise distance checks within a tile — lives in sfc_tiles_functions.h.
 *
 * Nothing here reads All: membership is type mask plus positive mass, and
 * the per-tile fold takes its reach from nlr_particle_symmetric_radius.
 * Keep it that way. Periodicity and box size enter only at traversal time,
 * which is why the traversal is not in this file.
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


/* TILE_PERIODIC_X/Y/Z defined in sfc_tiles.h */


/* Supply-pool membership. The single definition — every path that selects
 * pool members tests through this. */
static inline int sfc_pool_member(const struct particle_data *p, int type_bitmask)
{
    if(!((1 << p->Type) & type_bitmask)) return 0;
    if(p->Mass <= 0) return 0;
    return 1;
}

/* Open a tile covering pool slots from `first`, with nothing folded in yet.
 *
 * The box starts INVERTED rather than zeroed. The BVH unions child boxes with
 * min/max, so a zeroed box would drag every ancestor out to the coordinate
 * origin and stop the opener pruning that whole subtree; inverted bounds are
 * neutral under the union and always fail the sphere-overlap gap test, which
 * is what a tile with nothing live in it needs. Folding the first member then
 * sets lo=hi=its position, so no separate "seeded" case is required. */
static inline void sfc_tile_begin(sfc_tile_t *tile, int first)
{
    tile->first = first;
    tile->count = 0;
    tile->hmax = 0;
    for(int t = 0; t < TILE_NUM_PTYPES; t++) tile->hmax_by_type[t] = 0;
    for(int k = 0; k < 3; k++) { tile->lo[k] = MAX_REAL_NUMBER; tile->hi[k] = -MAX_REAL_NUMBER; }
}

/* Fold one live member into a tile's box and reach. The tiling rule lives here
 * and nowhere else, so the two ways of enumerating members (deriving the pool
 * in the same pass, or walking an existing one) cannot drift apart.
 *
 * hmax aggregates the SSOT per-particle reach under radius_policy, scaled by
 * scale_factor (default 1.0 -> bare per-policy reach for Mode A; ghost_exchange
 * passes j_radius_scale * safety_factor to keep BVH bands and leaf compact h on
 * the same supply-side reach). */
static inline void sfc_tile_fold(sfc_tile_t *tile, const struct particle_data *p,
                                 mode_b_radius_policy_t radius_policy, double scale_factor)
{
    for(int k = 0; k < 3; k++) {
        if(p->Pos[k] < tile->lo[k]) tile->lo[k] = p->Pos[k];
        if(p->Pos[k] > tile->hi[k]) tile->hi[k] = p->Pos[k];
    }
    double hj = nlr_particle_symmetric_radius(*p, radius_policy) * scale_factor;
    if(hj > tile->hmax) tile->hmax = hj;
    int tj = (int)p->Type;
    if(tj >= 0 && tj < TILE_NUM_PTYPES && hj > tile->hmax_by_type[tj])
        tile->hmax_by_type[tj] = hj;
}

int build_sfc_supply_pool(struct particle_data *P, int num_total,
                          int type_bitmask, int **pool_indices_out)
{
    int num_pool = 0;
    for(int i = 0; i < num_total; i++) {
        if(!sfc_pool_member(&P[i], type_bitmask)) continue;
        num_pool++;
    }
    if(!pool_indices_out) return num_pool;

    int *pool = (int *) mymalloc("sfc_pool", (num_pool > 0 ? num_pool : 1) * sizeof(int));
    int p = 0;
    for(int i = 0; i < num_total; i++) {
        if(!sfc_pool_member(&P[i], type_bitmask)) continue;
        pool[p++] = i;
    }
    *pool_indices_out = pool;
    return num_pool;
}

int build_sfc_tiles(struct particle_data *P, int num_total,
                    int type_bitmask, int target_tile_size,
                    sfc_tile_t **tiles_out, int **pool_indices_out,
                    int *num_pool_out,
                    mode_b_radius_policy_t radius_policy,
                    double scale_factor)
{
    /* Deriving the pool and tiling it are the same walk over P[], so do them in
     * ONE pass and avoid several full streams over a large particle struct: this
     * is a bandwidth-bound loop over every particle, and each extra stream is
     * another cache line per particle. Membership and the tiling rule are the
     * shared helpers above, so this states neither of them a second time.
     *
     * Both arrays are sized to their upper bounds because the member count is
     * not known until the pass ends. That costs a transient int[num_total] plus
     * the tiles for a full pool; the persistent copies callers keep are cut to
     * the exact counts returned here. Tiles are mymalloc'd above the pool, so
     * the caller must free tiles first (free_sfc_tiles already does). */
    int pool_capacity = (num_total > 0) ? num_total : 1;
    int tile_capacity = (num_total + target_tile_size - 1) / target_tile_size;
    if(tile_capacity < 1) tile_capacity = 1;

    int *pool = (int *) mymalloc("sfc_pool", pool_capacity * sizeof(int));
    sfc_tile_t *tiles = (sfc_tile_t *) mymalloc("sfc_tiles", tile_capacity * sizeof(sfc_tile_t));

    int num_pool = 0, ntiles = 0;
    for(int i = 0; i < num_total; i++)
    {
        if(!sfc_pool_member(&P[i], type_bitmask)) continue;
        /* A member starting a fresh tile opens it; tiles cover consecutive runs
         * of target_tile_size pool slots, so `first` is the slot it opens at. */
        if((num_pool % target_tile_size) == 0) sfc_tile_begin(&tiles[ntiles++], num_pool);
        pool[num_pool++] = i;
        tiles[ntiles - 1].count++;
        sfc_tile_fold(&tiles[ntiles - 1], &P[i], radius_policy, scale_factor);
    }
    /* An empty pool still publishes one (empty, inverted-box) tile so the BVH
     * always has a root to build over. */
    if(ntiles == 0) { sfc_tile_begin(&tiles[0], 0); ntiles = 1; }

    *tiles_out = tiles;
    *pool_indices_out = pool;
    *num_pool_out = num_pool;
    return ntiles;
}

int build_sfc_tiles_from_pool(struct particle_data *P, const int *pool, int num_pool,
                              int target_tile_size, sfc_tile_t **tiles_out,
                              mode_b_radius_policy_t radius_policy,
                              double scale_factor)
{
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

        sfc_tile_begin(&tiles[t], start);
        /* count is the SLOT count, not the live count: a retained pool can hold
         * entries marked dead after it was built, and the slots stay addressable.
         * A tile whose slots are all dead keeps the inverted box sfc_tile_begin
         * left, which is what an empty tile needs. */
        tiles[t].count = count;

        /* Eliminated elements (Mass <= 0) contribute nothing — a dead slot's
         * stale reach would widen the band this rank advertises as supply. */
        for(int s = 0; s < count; s++)
        {
            int j = pool[start + s];
            if(P[j].Mass <= 0) continue;
            sfc_tile_fold(&tiles[t], &P[j], radius_policy, scale_factor);
        }
    }

    *tiles_out = tiles;
    return ntiles;
}


void free_sfc_tiles(sfc_tile_t *tiles, int *pool_indices)
{
    /* Free in reverse mymalloc order: tiles allocated after pool */
    if(tiles) myfree(tiles);
    if(pool_indices) myfree(pool_indices);
}


/* ================================================================
   BVH over SFC tiles — recursive midpoint subdivision.
   Since tiles are SFC-sorted, consecutive tiles are spatially local,
   so midpoint subdivision produces a spatially balanced tree.
   ================================================================ */

/* Recursive builder. Returns the index of the created node in bvh[].
   *next_node is the next free slot. */
static int build_bvh_recursive(sfc_tile_t *tiles, int tile_start, int tile_end,
                               tile_bvh_node_t *bvh, int *next_node)
{
    if(tile_end - tile_start == 1)
    {
        /* Leaf: create a node pointing to a single tile */
        int idx = (*next_node)++;
        int t = tile_start;
        for(int k = 0; k < 3; k++) { bvh[idx].lo[k] = tiles[t].lo[k]; bvh[idx].hi[k] = tiles[t].hi[k]; }
        bvh[idx].hmax = tiles[t].hmax;
        for(int tt = 0; tt < TILE_NUM_PTYPES; tt++) bvh[idx].hmax_by_type[tt] = tiles[t].hmax_by_type[tt];
        bvh[idx].left = -(t + 1);   /* negative encoding: leaf = -(tile_index + 1) */
        bvh[idx].right = -(t + 1);  /* same tile for both (signals leaf) */
        return idx;
    }

    /* Internal node: split at midpoint */
    int mid = (tile_start + tile_end) / 2;

    /* Reserve this node's slot first (ensures parent index > children for root-last ordering) */
    /* Actually, build children first, then this node, so we can compute union bbox */
    int left_idx = build_bvh_recursive(tiles, tile_start, mid, bvh, next_node);
    int right_idx = build_bvh_recursive(tiles, mid, tile_end, bvh, next_node);

    int idx = (*next_node)++;
    for(int k = 0; k < 3; k++) {
        bvh[idx].lo[k] = DMIN(bvh[left_idx].lo[k], bvh[right_idx].lo[k]);
        bvh[idx].hi[k] = DMAX(bvh[left_idx].hi[k], bvh[right_idx].hi[k]);
    }
    bvh[idx].hmax = DMAX(bvh[left_idx].hmax, bvh[right_idx].hmax);
    for(int tt = 0; tt < TILE_NUM_PTYPES; tt++)
        bvh[idx].hmax_by_type[tt] = DMAX(bvh[left_idx].hmax_by_type[tt], bvh[right_idx].hmax_by_type[tt]);
    bvh[idx].left = left_idx;
    bvh[idx].right = right_idx;
    return idx;
}

int build_tile_bvh(sfc_tile_t *tiles, int ntiles, tile_bvh_node_t **bvh_out)
{
    if(ntiles <= 0) { *bvh_out = NULL; return 0; }

    /* A binary tree with ntiles leaves has (2*ntiles - 1) total nodes */
    int max_nodes = 2 * ntiles - 1;
    tile_bvh_node_t *bvh = (tile_bvh_node_t *) mymalloc("tile_bvh", max_nodes * sizeof(tile_bvh_node_t));

    int next_node = 0;
    build_bvh_recursive(tiles, 0, ntiles, bvh, &next_node);

    *bvh_out = bvh;
    return next_node; /* root is at index (next_node - 1) */
}
