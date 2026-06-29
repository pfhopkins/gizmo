/* topleaf_router_functions.h -- pure helpers for top-leaf-targeted request-driven
 * ghost routing (FIRE perf-regression fix; helper-only stage).
 *
 * GPU header: pulls <Kokkos_Core.hpp> via gpu_peano_walk_functions.h, so include
 * it ONLY from TUs compiled by GPU_CC (nvcc_wrapper).  Do NOT include it from a
 * plain-mpicxx host TU (e.g. ghost_exchange.cc) -- that trips the
 * "KOKKOS_ENABLE_CUDA but compiler is not nvcc" catastrophic error on Vista.
 *
 * Two DISTINCT primitives -- never conflate them:
 *   (1) tlr_point_to_topleaf[_owner]: the single top-leaf CONTAINING a POINT.
 *       Use it only to bucket owned-local SUPPLY particles into the per-top-leaf
 *       reach band, and for diagnostics.  It is NOT a routing primitive.
 *   (2) tlr_route_query_over_topleaves: a query sphere overlaps MANY leaves; a
 *       correct router exports to EVERY overlapping leaf owner.  Routing on the
 *       containing leaf alone would UNDER-ROUTE (silent missed neighbours).
 *
 * Opener geometry reproduces the legacy top-tree opener margins
 * (system/ngb_codeblock_after_condition_unthreaded.h:78-86 +
 *  system/ngb_codeblock_checknode.h on gizmo-cpp):
 *     dist0 = reach + 0.5*len          (per-axis box gate)
 *     dist1 = dist0 + CUBE_EDGEFACTOR_1*len   (enclosing-sphere gate)
 * where reach = (SYMMETRIC ? max(h_query, cell supply band) : h_query).  The
 * periodic convention is minimum-image, matching the receiver walk
 * gx_walk_local_bvh / gx_bbox_overlaps_sphere (ghost_exchange.cc) -- the opener
 * is factored into ONE helper (tlr_cell_overlaps_query); never reimplement it.
 *
 * All helpers are pure functions of caller-supplied mirrors (TopNodes, the
 * per-top-leaf {center,len} geometry, DomainTask, the per-leaf band) -- no
 * globals; host- and device-callable.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GIZMO_TOPLEAF_ROUTER_FUNCTIONS_H
#define GIZMO_TOPLEAF_ROUTER_FUNCTIONS_H

#include <stdint.h>

#include "../declarations/constants.h"            /* CUBE_EDGEFACTOR_1 */
#include "../gravity/gpu_peano_walk_functions.h"  /* gpu_peano_and_morton_key, gpu_topleaf_for_key,
                                                     gpu_morton_double_to_int42, Morton128, peanokey,
                                                     topnode_data, BITS_PER_DIMENSION */

/* Per-type band dimension.  MUST equal sfc_tiles.h TILE_NUM_PTYPES; the .cc that
 * uses both static_asserts the equality so this never silently drifts. */
#define TLR_NUM_PTYPES 6

/* ---- (1) point -> containing top-leaf -------------------------------------
 * Mirrors gpu_topology_build.cc's topo_keys_and_assign normalization EXACTLY so
 * a supply particle is bucketed into the SAME leaf the domain decomposition
 * assigns it (its owner == this rank by construction).  Float positions are NOT
 * acceptable here -- pass the double P[j].Pos so boundary cases match. */
KOKKOS_INLINE_FUNCTION int
tlr_point_to_topleaf(const struct topnode_data *tn,
                     const double pos[3], const double corner[3],
                     double inv_dlen, int bits)
{
    double fx = (pos[0] - corner[0]) * inv_dlen;
    double fy = (pos[1] - corner[1]) * inv_dlen;
    double fz = (pos[2] - corner[2]) * inv_dlen;
    if(fx < 0.0) {fx = 0.0;} if(fx >= 1.0) {fx = 0.99999999999999988897;}
    if(fy < 0.0) {fy = 0.0;} if(fy >= 1.0) {fy = 0.99999999999999988897;}
    if(fz < 0.0) {fz = 0.0;} if(fz >= 1.0) {fz = 0.99999999999999988897;}
    uint64_t ix = gpu_morton_double_to_int42(fx + 1.0);
    uint64_t iy = gpu_morton_double_to_int42(fy + 1.0);
    uint64_t iz = gpu_morton_double_to_int42(fz + 1.0);
    Morton128 m;
    peanokey pkey = gpu_peano_and_morton_key(ix, iy, iz, bits, &m);
    return gpu_topleaf_for_key(tn, pkey);
}

KOKKOS_INLINE_FUNCTION int
tlr_point_to_topleaf_owner(const struct topnode_data *tn, const int *domain_task,
                           const double pos[3], const double corner[3],
                           double inv_dlen, int bits)
{
    return domain_task[ tlr_point_to_topleaf(tn, pos, corner, inv_dlen, bits) ];
}

/* ---- (2) single overlap helper -------------------------------------------
 * Returns 1 iff the top-leaf cube (center,len) can hold a neighbour of the query
 * under the legacy opener + minimum-image convention.  `reach` is the already
 * combined symmetric reach: rkern for ONEWAY, max(rkern, cell band) for
 * SYMMETRIC (the caller combines it; this helper is purely geometric).
 * Conservative by construction -- it is the legacy export opener. */
KOKKOS_INLINE_FUNCTION int
tlr_cell_overlaps_query(const double center[3], double len,
                        const double pos_q[3], double reach,
                        const int periodic_flags[3], const double box_sizes[3])
{
    const double dist0 = reach + 0.5 * len;
    const double dist1 = dist0 + CUBE_EDGEFACTOR_1 * len;
    double d[3];
    for(int k = 0; k < 3; k++) {
        double dd = pos_q[k] - center[k];
        if(periodic_flags[k]) {
            double b = box_sizes[k], h = 0.5 * b;
            if(dd > h) dd -= b; else if(dd < -h) dd += b;
        }
        double ad = (dd > 0) ? dd : -dd;
        if(ad > dist0) return 0;        /* per-axis box gate (legacy dx>dist) */
        d[k] = ad;
    }
    /* enclosing-sphere gate (legacy dist += CUBE_EDGEFACTOR_1*len) */
    if(d[0]*d[0] + d[1]*d[1] + d[2]*d[2] > dist1 * dist1) return 0;
    return 1;
}

/* ---- (2b) AABB-vs-top-leaf-cube overlap (SYMM band attribution) -----------
 * Returns 1 iff the top-leaf cube (center,len) overlaps the axis-aligned box
 * [aabb_lo,aabb_hi] inflated by `reach`, under the SAME legacy opener margins +
 * minimum-image convention as tlr_cell_overlaps_query.  It is exactly that point
 * opener generalized from a point to a box: the per-axis cube-centre-to-box
 * separation is the point separation reduced by the box half-width, so a
 * degenerate box (lo==hi) reproduces tlr_cell_overlaps_query bit-for-bit.
 * Conservative by construction.  Used to attach a supply tile's reach to the
 * R-OWNED routing leaves it can reach (the walker-rank band invariant). */
KOKKOS_INLINE_FUNCTION int
tlr_cell_overlaps_aabb(const double center[3], double len,
                       const double aabb_lo[3], const double aabb_hi[3], double reach,
                       const int periodic_flags[3], const double box_sizes[3])
{
    const double dist0 = reach + 0.5 * len;
    const double dist1 = dist0 + CUBE_EDGEFACTOR_1 * len;
    double d[3];
    for(int k = 0; k < 3; k++) {
        double tc  = 0.5 * (aabb_lo[k] + aabb_hi[k]);   /* box centre */
        double thw = 0.5 * (aabb_hi[k] - aabb_lo[k]);   /* box half-width (>=0) */
        double dd = center[k] - tc;
        if(periodic_flags[k]) {
            double b = box_sizes[k], h = 0.5 * b;
            if(dd > h) dd -= b; else if(dd < -h) dd += b;
        }
        double ad = (dd > 0) ? dd : -dd;
        ad -= thw;                      /* nearest-surface separation cube-centre <-> box */
        if(ad < 0) ad = 0;              /* cube centre within the box's axis extent */
        if(ad > dist0) return 0;        /* per-axis box gate */
        d[k] = ad;
    }
    if(d[0]*d[0] + d[1]*d[1] + d[2]*d[2] > dist1 * dist1) return 0;   /* enclosing-sphere gate */
    return 1;
}

/* ---- (3) query -> overlapping top-leaf OWNERS (FLAT reference) ------------
 * Exports to every overlapping leaf owner, deduped -- NOT just the containing
 * leaf.  FLAT O(ntl)-per-query scan: correct but a diagnostic/reference; the
 * production router will descend the TopNodes tree hierarchically (design §4).
 *
 *   leaf_center  [ntl*3], leaf_len [ntl], domain_task [ntl] -- geometry cache.
 *   ntask        -- number of MPI ranks; bounds the owner id and the dedup
 *                   scratch.  An owner outside [0,ntask) is FAILED CLOSED here
 *                   (skipped) so a stale/corrupt domain_task can never OOB-write
 *                   seen[]; the host wrapper/oracle should additionally promote
 *                   any such case to a broadcast fallback / loud error.
 *   band_by_type [ntl*TLR_NUM_PTYPES] or NULL.  NULL  => ONEWAY (reach=rkern).
 *                                               non-NULL => SYMMETRIC (band).
 *   supply_mask  -- which types contribute to the SYMMETRIC band.
 *   owners_out   [>= ntask] -- distinct owners written.
 *   seen         [ntask], ZERO-initialised by caller -- dedup scratch.
 * Returns the number of distinct owners written. */
KOKKOS_INLINE_FUNCTION int
tlr_route_query_over_topleaves(const double *leaf_center, const double *leaf_len,
                               const int *domain_task, int ntl, int ntask,
                               const double pos_q[3], double rkern,
                               const double *band_by_type, unsigned int supply_mask,
                               const int periodic_flags[3], const double box_sizes[3],
                               int *owners_out, char *seen)
{
    int n = 0;
    for(int leaf = 0; leaf < ntl; leaf++) {
        double reach = rkern;
        if(band_by_type) {
            const double *b = &band_by_type[(size_t)leaf * TLR_NUM_PTYPES];
            double be = 0;
            for(int t = 0; t < TLR_NUM_PTYPES; t++) {
                if((supply_mask & (1u << (unsigned)t)) == 0u) continue;
                if(b[t] > be) be = b[t];
            }
            if(be > reach) reach = be;
        }
        if(reach <= 0) continue;
        if(!tlr_cell_overlaps_query(&leaf_center[(size_t)leaf * 3], leaf_len[leaf],
                                    pos_q, reach, periodic_flags, box_sizes)) continue;
        int owner = domain_task[leaf];
        if(owner < 0 || owner >= ntask) continue;   /* fail closed: never OOB seen[] */
        if(!seen[owner]) { seen[owner] = 1; owners_out[n++] = owner; }
    }
    return n;
}

/* ---- (3b) HIERARCHICAL tile-AABB -> overlapping OWNED top-leaves (H4b-perf) -
 * Descends the TopNodes octree collecting the leaves owned by self_rank whose cube
 * overlaps the box [aabb_lo,aabb_hi] inflated by `reach` (tlr_cell_overlaps_aabb).
 * The SYMM band attribution's cheap form: O(depth x fanout) per tile instead of the
 * flat O(NTopleaves) owned-leaf scan.  PROVABLY non-pruning: a parent topnode cube
 * contains its child cubes, so if the inflated box reaches any descendant leaf it
 * also overlaps the parent -> the parent is never pruned; the leaf-level test is the
 * SAME tlr_cell_overlaps_aabb the flat scan applies -> identical owned-leaf set.
 *   leaves_out [>= leaves_cap], leaves_cap = max leaves to collect (>= n_owned).
 * Returns #owned leaves written, or -1 on stack/buffer overflow (caller falls back). */
#define TLR_HIER_STACK 512   /* >> 8 x realistic TopNodes depth; overflow -> -1 */
KOKKOS_INLINE_FUNCTION int
tlr_collect_owned_leaves_for_aabb(const struct topnode_data *tn, int ntn,
                                  const double *tn_center, const double *tn_len,
                                  const int *domain_task, int self_rank,
                                  const double aabb_lo[3], const double aabb_hi[3], double reach,
                                  const int periodic_flags[3], const double box_sizes[3],
                                  int *leaves_out, int leaves_cap)
{
    int n = 0;
    int stack[TLR_HIER_STACK];
    int sp = 0;
    if(ntn <= 0) return 0;
    stack[sp++] = 0;                       /* root topnode */
    while(sp > 0) {
        int no = stack[--sp];
        if(no < 0 || no >= ntn) continue;  /* defensive */
        if(!tlr_cell_overlaps_aabb(&tn_center[(size_t)no * 3], tn_len[no],
                                   aabb_lo, aabb_hi, reach, periodic_flags, box_sizes)) continue;  /* prune subtree */
        if(tn[no].Daughter < 0) {          /* domain leaf */
            int leaf = tn[no].Leaf;
            if(domain_task[leaf] != self_rank) continue;   /* OWNED leaves only */
            if(n >= leaves_cap) return -1;                  /* buffer overflow -> caller fallback */
            leaves_out[n++] = leaf;
        } else {
            if(sp + 8 > TLR_HIER_STACK) return -1;          /* stack overflow -> caller fallback */
            for(int c = 0; c < 8; c++) stack[sp++] = tn[no].Daughter + c;
        }
    }
    return n;
}

/* ---- (4) HIERARCHICAL query -> overlapping top-leaf OWNERS (H1/H4b) ---------
 * Descends the TopNodes octree (`tn[no].Daughter`/`.Leaf`) with the SAME opener
 * as the flat router, pruning whole subtrees -> O(depth x fanout) instead of the
 * flat O(NTopleaves).
 *
 *   tn            -- topnode_data tree (Daughter<0 marks a domain leaf).
 *   tn_center [ntn*3], tn_len [ntn] -- per-topnode cube geometry (from
 *                    Nodes[TopNodeNodeIndex[no]]; NOT derived from the PH child).
 *   domain_task   -- [NTopleaves]; owner = domain_task[tn[no].Leaf].
 *   ntask         -- bounds owner id + the dedup scratch (fail-closed OOB).
 *   node_band [ntn*TLR_NUM_PTYPES] or NULL.  NULL  => ONEWAY (reach = rkern at
 *                    every level; conservative because a leaf cube within reach
 *                    implies all ancestor cubes are within reach).  non-NULL =>
 *                    SYMMETRIC: per-node reach = max(rkern, max over supply_mask
 *                    types of node_band[no]).  Because node_band is the MAX over a
 *                    node's subtree leaf bands, the descent reach at any node is
 *                    >= the reach at every leaf below it, so no leaf the flat SYMM
 *                    router would export is ever pruned -> hier owner set == flat.
 *   supply_mask   -- which types contribute to the SYMMETRIC band (ignored if
 *                    node_band == NULL).
 *   self_rank     -- excluded (ghost import is remote-only).
 *   owners_out    [>= ntask], seen [ntask] ZERO-initialised by caller.
 * Returns distinct remote owners written, or -1 on stack overflow (caller treats
 * as error -> oracle reports unavailable, never a false mismatch). */
KOKKOS_INLINE_FUNCTION int
tlr_route_query_hierarchical(const struct topnode_data *tn, int ntn,
                             const double *tn_center, const double *tn_len,
                             const int *domain_task, int ntask,
                             const double pos_q[3], double rkern,
                             const double *node_band, unsigned int supply_mask,
                             const int periodic_flags[3], const double box_sizes[3],
                             int self_rank, int *owners_out, char *seen)
{
    int n = 0;
    int stack[TLR_HIER_STACK];
    int sp = 0;
    if(ntn <= 0) return 0;
    stack[sp++] = 0;                       /* root topnode */
    while(sp > 0) {
        int no = stack[--sp];
        if(no < 0 || no >= ntn) continue;  /* defensive */
        double reach = rkern;
        if(node_band) {                    /* SYMMETRIC: widen reach by the node band */
            const double *b = &node_band[(size_t)no * TLR_NUM_PTYPES];
            double be = 0;
            for(int t = 0; t < TLR_NUM_PTYPES; t++) {
                if((supply_mask & (1u << (unsigned)t)) == 0u) continue;
                if(b[t] > be) be = b[t];
            }
            if(be > reach) reach = be;
        }
        if(reach <= 0) continue;
        if(!tlr_cell_overlaps_query(&tn_center[(size_t)no * 3], tn_len[no],
                                    pos_q, reach, periodic_flags, box_sizes)) continue;  /* prune subtree */
        if(tn[no].Daughter < 0) {          /* domain leaf */
            int owner = domain_task[tn[no].Leaf];
            if(owner < 0 || owner >= ntask) continue;   /* fail closed */
            if(owner == self_rank) continue;            /* remote-only */
            if(!seen[owner]) { seen[owner] = 1; owners_out[n++] = owner; }
        } else {
            if(sp + 8 > TLR_HIER_STACK) return -1;       /* overflow -> caller error */
            for(int c = 0; c < 8; c++) stack[sp++] = tn[no].Daughter + c;
        }
    }
    return n;
}

#endif /* GIZMO_TOPLEAF_ROUTER_FUNCTIONS_H */
