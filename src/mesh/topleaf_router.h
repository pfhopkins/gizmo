/* topleaf_router.h -- host lifecycle for the top-leaf-targeted request-driven
 * ghost router (FIRE perf-regression fix; helper-only stage).
 *
 * Plain-C declarations only (no Kokkos) so a host TU may include it.  The pure
 * device/host routing helpers live in topleaf_router_functions.h (GPU header).
 *
 * Router-owned state, two caches:
 *   - GEOMETRY cache: per-top-leaf cube {center[3], len} read from
 *     Nodes[DomainNodeIndex[leaf]] (the domain top-node SSOT), mirrored to
 *     SharedSpace.  Built by topleaf_router_geometry_acquire(); valid only after
 *     domain decomposition + force_treebuild populate DomainNodeIndex.  Kept on
 *     the router's own lifecycle so hot routing code never reaches into the
 *     gravity tree internals.
 *   - BAND cache: per-top-leaf supply reach hmax_by_type[6] for SYMMETRIC
 *     callers, aggregated from the owned-local supply pool (g_glt_cache) via
 *     gx_policy_scaled_h.  Epoch-keyed to that supply cache.
 *
 * Validity is explicit: if any precondition is missing the cache is marked
 * INVALID and the caller MUST fall back to broadcast discovery -- never route
 * with stale/approximate bounds.  This stage adds NO call sites on any runtime
 * path; the entry points exist for the oracle/transport stages to wire.
 *
 * LIFECYCLE CONTRACT (binding for the wiring stages):
 *   - The geometry cache snapshots Nodes[DomainNodeIndex[leaf]] at acquire time.
 *     It goes STALE after any domain decomposition or force_treebuild/
 *     force_treefree boundary (DomainNodeIndex and the node slots are rebuilt).
 *   - topleaf_router_geometry_acquire() MUST be called after every such rebuild,
 *     before any routing use.  topleaf_router_geometry_valid() must NOT be
 *     trusted across a domain/tree rebuild boundary unless the router has been
 *     re-acquired (or invalidated) in between.
 *   - topleaf_router_geometry_invalidate() marks the cache (and band) INVALID
 *     without freeing storage; C2 MUST call it from the same sites that fire the
 *     ghost local-tree invalidation (drift/domain/tree rebuild), so a missing
 *     re-acquire fails closed to broadcast instead of routing on stale geometry.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GIZMO_TOPLEAF_ROUTER_H
#define GIZMO_TOPLEAF_ROUTER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Read-only view of the owned-local supply pool + its epoch key, exported by
 * ghost_exchange.cc for the band builder.  EXCLUDES ghosts by construction (it
 * is exactly the pool gx_walk_local_bvh walks).  compact_xyzh[p*4+3] is the
 * baked gx_policy_scaled_h supply reach for pool[p].  pool[p] is the particle
 * index j (use P[j].Pos for the leaf mapping -- float compact pos must NOT be
 * used for bucketing).  Returns num_pool, or -1 if the supply cache is invalid. */
struct gx_supply_pool_view {
    const int   *pool;          /* [num_pool] owned-local particle indices j */
    const int   *pool_types;    /* [num_pool] P[j].Type                      */
    const float *compact_xyzh;  /* [num_pool*4]; [3] = gx_policy_scaled_h(j) */
    int          num_pool;
    /* epoch key (any change => band must rebuild) */
    int          numpart_when_built;
    long long    ti_when_built;
    double       safety_when_built;
    unsigned int eligible_mask_when_built;
    int          radius_policy_when_built;
    double       j_scale_when_built;
};
int ghost_exchange_supply_pool_view(struct gx_supply_pool_view *out);

/* --- geometry cache --- */
int  topleaf_router_geometry_acquire(void);     /* 0 = built & valid; nonzero = invalid (fall back to broadcast) */
void topleaf_router_geometry_release(void);     /* free storage */
void topleaf_router_geometry_invalidate(void);  /* mark INVALID (cache+band), keep storage; for rebuild boundaries */
int  topleaf_router_geometry_valid(void);       /* 1 if usable, else 0 */
const double *topleaf_router_leaf_center(void);  /* [NTopleaves*3] or NULL */
const double *topleaf_router_leaf_len(void);     /* [NTopleaves]   or NULL */
int           topleaf_router_ntopleaves(void);   /* leaf count the caches were built for */

/* Route a batch of queries to their overlapping top-leaf owners, EXCLUDING
 * self_rank (ghost import is remote-only).  CSR output: query i routes to
 * owners[off[i] .. off[i+1]).  oneway != 0 => ONEWAY (band ignored, reach=h);
 * oneway == 0 => SYMMETRIC (requires a valid band — not used in C2).
 * periodic_flags/box_sizes MUST be the receiver walk's convention (passed by the
 * caller so the opener and the receiver share one SSOT).  Returns the total
 * number of owner entries written (>=0), or -1 if geometry/band is invalid or
 * the owners buffer overflowed -- in which case the CALLER MUST collectively
 * fall back to broadcast (never route partially).  Lives in the router GPU TU so
 * the device routing helpers stay out of host-only TUs. */
int topleaf_router_route_queries(const double *q_pos, const double *q_h, int nq,
                                 unsigned int supply_mask, int oneway,
                                 const int periodic_flags[3], const double box_sizes[3],
                                 int *off, int *owners, long owners_cap, int self_rank);

/* H1 flat-vs-hierarchical owner-set oracle (ONEWAY).  Returns #queries whose
 * remote-owner sets differ between the flat leaf-scan and the hierarchical
 * tree-descent (0 = all equal), or a NEGATIVE code if UNAVAILABLE:
 *   -1 = geometry invalid,  -2 = scratch alloc fail,  -3 = hier stack overflow.
 * *first_bad = first mismatching query index.  Caller separates mismatch from
 * unavailable and does the collective Allreduce + controlled-stop (both fatal for
 * a validation gate). */
int topleaf_router_hier_vs_flat_check(const double *q_pos, const double *q_h, int nq,
                                      const int periodic_flags[3], const double box_sizes[3],
                                      int self_rank, int *first_bad);

/* --- per-top-leaf supply band (SYMMETRIC callers) --- */
int  topleaf_router_band_build(void);           /* 0 = built & valid; nonzero = unavailable */
int  topleaf_router_band_valid(void);
const double *topleaf_router_band(int *ntl_out); /* [NTopleaves*6] or NULL */
void topleaf_router_band_invalidate(void);

#ifdef __cplusplus
}
#endif

#endif /* GIZMO_TOPLEAF_ROUTER_H */
