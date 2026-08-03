/* topleaf_router.cc -- host lifecycle for the top-leaf-targeted request-driven
 * ghost router (FIRE perf-regression fix; helper-only stage).
 *
 * Builds two router-owned caches and exposes the pure routing helpers in
 * topleaf_router_functions.h.  This stage is production-neutral: nothing here is
 * called on a normal step yet (the oracle/transport stages wire the call sites).
 *
 * GPU TU (compiled by GPU_CC): uses gpu_peano_walk_functions.h key helpers for
 * the leaf mapping and Kokkos SharedSpace for the device-ready mirrors.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#include <mpi.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"   /* MUST precede allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "sfc_tiles.h"                         /* TILE_NUM_PTYPES (static_assert) */
#include "topleaf_router.h"
#include "topleaf_router_functions.h"          /* tlr_point_to_topleaf, TLR_NUM_PTYPES */

/* The router band index width must match the supply tile per-type width. */
static_assert(TLR_NUM_PTYPES == TILE_NUM_PTYPES,
              "TLR_NUM_PTYPES must equal sfc_tiles.h TILE_NUM_PTYPES");

namespace {

/* --- geometry cache (per-top-leaf cube {center[3], len}) --- */
static double *g_leaf_center = NULL;   /* SharedSpace [cap*3] */
static double *g_leaf_len    = NULL;   /* SharedSpace [cap]   */
static int     g_geom_cap    = 0;
static int     g_geom_ntl    = 0;
static int     g_geom_valid  = 0;

/* --- per-TOPNODE cube {center[3], len} for the hierarchical router (H1) --- */
/* From Nodes[TopNodeNodeIndex[no]] for every topnode (internal + leaf), so the
 * hierarchical descent never reads gravity-tree internals in hot code (same
 * lifecycle as the leaf cache; rebuilt together in geometry_acquire). */
static double *g_tn_center = NULL;      /* SharedSpace [cap*3] */
static double *g_tn_len    = NULL;      /* SharedSpace [cap]   */
static int     g_tn_cap    = 0;
static int     g_tn_count  = 0;        /* NTopnodes the cache was built for */

/* --- per-top-leaf supply band hmax_by_type[TLR_NUM_PTYPES] (SYMMETRIC) --- */
static double *g_band        = NULL;   /* SharedSpace [cap*TLR_NUM_PTYPES] */
static int     g_band_cap    = 0;
static int     g_band_ntl    = 0;
static int     g_band_valid  = 0;
/* band epoch key (mirrors the supply-pool-view epoch) */
static int          g_band_numpart = -1;
static long long    g_band_ti      = -1;
static double       g_band_safety  = 0.0;
static unsigned int g_band_mask    = 0;
static int          g_band_policy  = -1;
static double       g_band_jscale  = 0.0;

/* --- per-TOPNODE supply band for the hierarchical SYMM router (H4a) ---
 * node_band[no][type] = max over the band of every top-leaf in no's subtree,
 * propagated up the replicated TopNodes octree.  Built by the COLLECTIVE band
 * builder (topleaf_router_band_build_collective): g_band is first made GLOBAL via
 * a cross-rank Allreduce(MAX) (each rank fills only the leaves it owns), then
 * propagated here.  Keyed by its own epoch (g_cband_*) so the collective builder
 * is independent of the legacy local builder's g_band_* key. */
static double *g_tn_band      = NULL;   /* SharedSpace [cap*TLR_NUM_PTYPES] */
static int     g_tn_band_cap  = 0;
static int     g_cband_valid  = 0;      /* GLOBAL band + node band valid on this rank */
static int     g_cband_ntl    = 0;
static int     g_cband_ntn    = 0;
/* collective-band epoch key (supply-pool-view epoch + topology counts) */
static int          g_cband_numpart = -1;
static long long    g_cband_ti      = -1;
static double       g_cband_safety  = 0.0;
static unsigned int g_cband_mask    = 0;
static int          g_cband_policy  = -1;
static double       g_cband_jscale  = 0.0;

static double *shared_alloc_double(double *old, int old_cap, int new_cap, const char *what)
{
    if(old) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(old);
    (void)old_cap;
    long bytes = (long)new_cap * (long)sizeof(double);
    double *p = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_runner", bytes);
    if(!p) printf("topleaf_router: %s alloc failed (%d doubles, %ld B)\n", what, new_cap, bytes);
    return p;
}

}  /* anonymous namespace */


extern "C" int topleaf_router_geometry_acquire(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    g_geom_valid = 0;

    const int ntl = NTopleaves;
    /* Loud validity guard: usable only once domain decomp + force_treebuild have
     * populated DomainNodeIndex, DomainTask, and the node tree.  Otherwise leave
     * INVALID so callers fall back to broadcast. */
    if(ntl <= 0 || DomainNodeIndex == NULL || DomainTask == NULL || Nodes == NULL) {
        return 1;
    }

    if(g_geom_cap < ntl) {
        g_leaf_center = shared_alloc_double(g_leaf_center, g_geom_cap, ntl * 3, "leaf_center");
        g_leaf_len    = shared_alloc_double(g_leaf_len,    g_geom_cap, ntl,     "leaf_len");
        if(!g_leaf_center || !g_leaf_len) { g_geom_cap = 0; return 1; }
        g_geom_cap = ntl;
    }

    const int node_lo = All.MaxPart;
    const int node_hi = All.MaxPart + MaxNodes;   /* valid internal-node slot range */
    for(int leaf = 0; leaf < ntl; leaf++) {
        int node = DomainNodeIndex[leaf];
        if(node < node_lo || node >= node_hi) {
            /* A leaf maps to no valid node -> the whole cache is untrustworthy. */
            printf("topleaf_router: invalid DomainNodeIndex[%d]=%d (range [%d,%d)); geometry cache marked INVALID\n",
                   leaf, node, node_lo, node_hi);
            return 1;
        }
        /* Owner must be a real rank: validate here so the router's pre-routing
         * gate (and the route helper's fail-closed owner check) can never face an
         * out-of-range DomainTask -> no OOB into the dedup scratch, no asymmetric
         * mid-routing bail. */
        int owner = DomainTask[leaf];
        if(owner < 0 || owner >= NTask) {
            printf("topleaf_router: invalid DomainTask[%d]=%d (range [0,%d)); geometry cache marked INVALID\n",
                   leaf, owner, NTask);
            return 1;
        }
        g_leaf_center[leaf * 3 + 0] = (double)Nodes[node].center[0];
        g_leaf_center[leaf * 3 + 1] = (double)Nodes[node].center[1];
        g_leaf_center[leaf * 3 + 2] = (double)Nodes[node].center[2];
        g_leaf_len[leaf]            = (double)Nodes[node].len;
    }

    /* Per-topnode cube cache (H1) from Nodes[TopNodeNodeIndex[no]] for the
     * hierarchical descent.  TopNodeNodeIndex is the geometry SSOT (avoids the
     * Peano-vs-Morton child-index trap); validated complete+in-range at build. */
    const int ntn = NTopnodes;
    if(ntn <= 0 || TopNodeNodeIndex == NULL) {
        printf("topleaf_router: TopNodeNodeIndex unavailable (NTopnodes=%d); geometry cache marked INVALID\n", ntn);
        return 1;
    }
    if(g_tn_cap < ntn) {
        g_tn_center = shared_alloc_double(g_tn_center, g_tn_cap, ntn * 3, "tn_center");
        g_tn_len    = shared_alloc_double(g_tn_len,    g_tn_cap, ntn,     "tn_len");
        if(!g_tn_center || !g_tn_len) { g_tn_cap = 0; return 1; }
        g_tn_cap = ntn;
    }
    for(int no = 0; no < ntn; no++) {
        int node = TopNodeNodeIndex[no];
        if(node < node_lo || node >= node_hi) {
            printf("topleaf_router: invalid TopNodeNodeIndex[%d]=%d (range [%d,%d)); geometry cache marked INVALID\n",
                   no, node, node_lo, node_hi);
            return 1;
        }
        g_tn_center[no * 3 + 0] = (double)Nodes[node].center[0];
        g_tn_center[no * 3 + 1] = (double)Nodes[node].center[1];
        g_tn_center[no * 3 + 2] = (double)Nodes[node].center[2];
        g_tn_len[no]            = (double)Nodes[node].len;
    }
    g_tn_count = ntn;

    g_geom_ntl   = ntl;
    g_geom_valid = 1;
    /* Geometry changed -> any prior band is keyed to a different topology. */
    g_band_valid  = 0;
    g_cband_valid = 0;
    return 0;
}

extern "C" void topleaf_router_geometry_release(void)
{
    if(g_leaf_center) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_leaf_center); g_leaf_center = NULL; }
    if(g_leaf_len)    { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_leaf_len);    g_leaf_len    = NULL; }
    if(g_tn_center)   { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_tn_center);   g_tn_center   = NULL; }
    if(g_tn_len)      { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_tn_len);      g_tn_len      = NULL; }
    g_tn_cap = 0; g_tn_count = 0;
    g_geom_cap = 0; g_geom_ntl = 0; g_geom_valid = 0;
    if(g_band) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_band); g_band = NULL; }
    g_band_cap = 0; g_band_ntl = 0; g_band_valid = 0;
    if(g_tn_band) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_tn_band); g_tn_band = NULL; }
    g_tn_band_cap = 0; g_cband_valid = 0; g_cband_ntl = 0; g_cband_ntn = 0;
}

/* Mark the geometry cache (and the topology-keyed band) INVALID without freeing
 * storage.  Call from domain/tree rebuild boundaries so a missing re-acquire
 * fails closed to broadcast instead of routing on stale geometry. */
extern "C" void topleaf_router_geometry_invalidate(void) { g_geom_valid = 0; g_band_valid = 0; g_cband_valid = 0; }

extern "C" int           topleaf_router_geometry_valid(void)  { return g_geom_valid; }
extern "C" const double *topleaf_router_leaf_center(void)      { return g_geom_valid ? g_leaf_center : NULL; }
extern "C" const double *topleaf_router_leaf_len(void)         { return g_geom_valid ? g_leaf_len    : NULL; }
extern "C" int           topleaf_router_ntopleaves(void)       { return g_geom_ntl; }

extern "C" void topleaf_router_band_invalidate(void) { g_band_valid = 0; g_cband_valid = 0; }
extern "C" int  topleaf_router_band_valid(void)      { return g_band_valid; }
extern "C" const double *topleaf_router_band(int *ntl_out)
{
    if(ntl_out) *ntl_out = g_band_ntl;
    return g_band_valid ? g_band : NULL;
}

/* GLOBAL band + per-topnode node band validity (collective builder). */
extern "C" int topleaf_router_global_band_valid(void) { return g_cband_valid; }
extern "C" const double *topleaf_router_node_band(int *ntn_out)
{
    if(ntn_out) *ntn_out = g_cband_valid ? g_cband_ntn : 0;
    return g_cband_valid ? g_tn_band : NULL;
}

/* Route a batch of queries to their overlapping top-leaf owners (excl self).
 * See topleaf_router.h for the contract.  Runs the device routing helper on the
 * host pass (KOKKOS_INLINE_FUNCTION is host-callable) so the only owner of the
 * geometry helpers stays this GPU TU. */
extern "C" int topleaf_router_route_queries(const double *q_pos, const double *q_h, int nq,
                                            unsigned int supply_mask, int oneway,
                                            const int periodic_flags[3], const double box_sizes[3],
                                            int *off, int *owners, long owners_cap, int self_rank)
{
    if(!g_geom_valid) return -1;
    const int ntl = g_geom_ntl;

    const double *band = NULL;
    if(!oneway) {
        int bn = 0;
        band = topleaf_router_band(&bn);
        if(!band || bn != ntl) return -1;   /* SYMM needs a topology-matched band */
    }

    if(nq <= 0) { if(off) off[0] = 0; return 0; }

    char *seen      = (char *) malloc((size_t)(NTask > 0 ? NTask : 1) * sizeof(char));
    int  *owners_tmp= (int *)  malloc((size_t)(NTask > 0 ? NTask : 1) * sizeof(int));
    if(!seen || !owners_tmp) { free(seen); free(owners_tmp); return -1; }
    for(int t = 0; t < NTask; t++) seen[t] = 0;

    long cursor = 0;
    off[0] = 0;
    int rc = 0;
    for(int i = 0; i < nq; i++) {
        int n = tlr_route_query_over_topleaves(g_leaf_center, g_leaf_len, DomainTask,
                                               ntl, NTask, &q_pos[(size_t)i * 3], q_h[i],
                                               band, supply_mask, periodic_flags, box_sizes,
                                               owners_tmp, seen);
        for(int k = 0; k < n; k++) {
            int owner = owners_tmp[k];
            seen[owner] = 0;                 /* reset dedup scratch for next query */
            if(owner == self_rank) continue; /* ghost import is remote-only */
            if(cursor >= owners_cap) { rc = -1; }   /* overflow -> caller fallback */
            else owners[cursor++] = owner;
        }
        off[i + 1] = (int)cursor;
        if(rc) break;
    }

    free(seen); free(owners_tmp);
    return rc ? -1 : (int)cursor;
}

/* Receiver local-start derivation (Candidate L bounded fine-tree walk).  For ONE
 * received query, return the fine-tree START NODES = DomainNodeIndex[leaf] for
 * every top-leaf THIS rank owns that the query opens, using the SAME opener +
 * band the router routed it with (SSOT).  Local-only: nothing is shipped; the
 * receiver re-derives its own opened top-leaves.  The bounded walk continues from
 * each start node into the local fine subtree (stopping at the next top-level
 * boundary), replacing the whole-root receiver walk.
 *   oneway != 0  -> reach = h_q (band-free); oneway == 0 -> reach = max(h_q, per-
 *                   leaf SYMM band over supply_mask) (needs a topology-matched band).
 * Returns 0 on success (*n_starts set); -1 geometry/band unavailable (caller
 * collective-falls-back to broadcast); -2 start buffer overflow (fallback, NEVER
 * truncate).  An out-of-range DomainNodeIndex despite valid geometry fails closed
 * (-1) rather than silently dropping a region. */
extern "C" int topleaf_router_local_starts_for_query(const double pos_q[3], double h_q,
                                                     unsigned int supply_mask, int oneway,
                                                     const int periodic_flags[3], const double box_sizes[3],
                                                     int self_rank, int *starts_out, int starts_cap,
                                                     int *n_starts)
{
    if(n_starts) *n_starts = 0;
    if(!g_geom_valid) return -1;
    if(DomainNodeIndex == NULL || DomainTask == NULL || Nodes == NULL) return -1;
    const int ntl = g_geom_ntl;
    const double *band = NULL;
    if(!oneway) { int bn = 0; band = topleaf_router_band(&bn); if(!band || bn != ntl) return -1; }
    const int maxpart = All.MaxPart;
    const int nnodes  = Numnodestree;
    if(nnodes <= 0) return -1;

    int n = 0;
    for(int leaf = 0; leaf < ntl; leaf++) {
        if(DomainTask[leaf] != self_rank) continue;          /* local opened leaves only */
        double reach = h_q;
        if(band) {
            const double *b = &band[(size_t)leaf * TLR_NUM_PTYPES];
            double be = 0;
            for(int t = 0; t < TLR_NUM_PTYPES; t++) {
                if((supply_mask & (1u << (unsigned)t)) == 0u) continue;
                if(b[t] > be) be = b[t];
            }
            if(be > reach) reach = be;
        }
        if(reach <= 0) continue;
        if(!tlr_cell_overlaps_query(&g_leaf_center[(size_t)leaf * 3], g_leaf_len[leaf],
                                    pos_q, reach, periodic_flags, box_sizes)) continue;
        int node = DomainNodeIndex[leaf];
        if(node < maxpart || node >= maxpart + nnodes) return -1;   /* fail closed, not a silent skip */
        if(n >= starts_cap) return -2;                              /* overflow -> caller fallback */
        starts_out[n++] = node;
    }
    if(n_starts) *n_starts = n;
    return 0;
}

/* HIERARCHICAL routed-query CSR (H2/H4c): same output contract as
 * topleaf_router_route_queries(), but descends the TopNodes octree
 * (O(depth x fanout)) instead of the flat O(NTopleaves) leaf scan.  ONEWAY uses
 * reach=h_q (band-free); SYMMETRIC (oneway==0) uses the per-topnode band g_tn_band
 * (reach=max(h_q, node_band)), which requires the collective GLOBAL band
 * (g_cband_valid) -- else returns -1 so the caller collective-falls-back to
 * broadcast.  The hierarchical helper excludes self + dedups; returns -1 on
 * stack/cap overflow or invalid geometry/band -> caller collective-fallback. */
extern "C" int topleaf_router_route_queries_hier(const double *q_pos, const double *q_h, int nq,
                                                 unsigned int supply_mask, int oneway,
                                                 const int periodic_flags[3], const double box_sizes[3],
                                                 int *off, int *owners, long owners_cap, int self_rank)
{
    if(!g_geom_valid) return -1;
    const double *node_band = NULL;
    if(!oneway) {
        if(!g_cband_valid) return -1;  /* SYMM hierarchical needs the propagated node band */
        node_band = g_tn_band;
    }
    const int ntn = g_tn_count;
    if(ntn <= 0) { if(off) off[0] = 0; return 0; }
    if(nq <= 0)  { if(off) off[0] = 0; return 0; }

    char *seen       = (char *) malloc((size_t)(NTask > 0 ? NTask : 1) * sizeof(char));
    int  *owners_tmp = (int *)  malloc((size_t)(NTask > 0 ? NTask : 1) * sizeof(int));
    if(!seen || !owners_tmp) { free(seen); free(owners_tmp); return -1; }
    for(int t = 0; t < NTask; t++) seen[t] = 0;

    long cursor = 0;
    off[0] = 0;
    int rc = 0;
    for(int i = 0; i < nq; i++) {
        int n = tlr_route_query_hierarchical(TopNodes, ntn, g_tn_center, g_tn_len, DomainTask, NTask,
                                             &q_pos[(size_t)i * 3], q_h[i], node_band, supply_mask,
                                             periodic_flags, box_sizes, self_rank, owners_tmp, seen);
        if(n < 0) { rc = -1; break; }    /* hierarchical stack overflow -> fallback */
        for(int k = 0; k < n; k++) {
            int owner = owners_tmp[k];
            seen[owner] = 0;             /* reset dedup scratch (self already excluded by helper) */
            if(cursor >= owners_cap) { rc = -1; }
            else owners[cursor++] = owner;
        }
        off[i + 1] = (int)cursor;
        if(rc) break;
    }

    free(seen); free(owners_tmp);
    return rc ? -1 : (int)cursor;
}

/* LEGACY local-only band builder.  Builds the per-top-leaf supply reach band
 * hmax_by_type[] from the OWNED-LOCAL supply pool only -- correct just for leaves
 * this rank owns.  SUPERSEDED for routing by topleaf_router_band_build_collective
 * (which exchanges the band cross-rank so a query can route to remote leaves);
 * uncalled.  Retire in cleanup once the collective builder is wired + validated. */
extern "C" int topleaf_router_band_build(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    if(!g_geom_valid) return 1;   /* need leaf geometry/topology first */

    struct gx_supply_pool_view v;
    int num_pool = ghost_exchange_supply_pool_view(&v, GX_POOL_IDENTITY | GX_POOL_GEOMETRY);
    if(num_pool < 0) return 1;     /* supply cache invalid -> band unavailable */

    const int ntl = g_geom_ntl;

    /* Fast path: same topology + same supply epoch -> reuse. */
    if(g_band_valid && g_band_ntl == ntl &&
       g_band_numpart == v.numpart_when_built &&
       g_band_ti      == v.ti_when_built &&
       g_band_safety  == v.safety_when_built &&
       g_band_mask    == v.eligible_mask_when_built &&
       g_band_policy  == v.radius_policy_when_built &&
       g_band_jscale  == v.j_scale_when_built) {
        return 0;
    }

    if(g_band_cap < ntl) {
        g_band = shared_alloc_double(g_band, g_band_cap, ntl * TLR_NUM_PTYPES, "band");
        if(!g_band) { g_band_cap = 0; g_band_valid = 0; return 1; }
        g_band_cap = ntl;
    }
    for(int k = 0; k < ntl * TLR_NUM_PTYPES; k++) g_band[k] = 0.0;

    const double corner[3] = { DomainCorner[0], DomainCorner[1], DomainCorner[2] };
    const double inv_dlen  = (DomainLen > 0.0) ? (1.0 / DomainLen) : 0.0;
    const int    bits      = BITS_PER_DIMENSION;
    long n_owner_mismatch  = 0;

    for(int p = 0; p < num_pool; p++) {
        int j = v.pool[p];
        if(j < 0 || j >= NumPart) continue;
        int type = v.pool_types[p];
        if(type < 0 || type >= TLR_NUM_PTYPES) continue;
        double pos[3] = { P[j].Pos[0], P[j].Pos[1], P[j].Pos[2] };
        int leaf = tlr_point_to_topleaf(TopNodes, pos, corner, inv_dlen, bits);
        if(leaf < 0 || leaf >= ntl) continue;
        /* Owned-local supply: its leaf owner must be this rank.  A mismatch
         * signals a mapping/ownership inconsistency -- count and report (bounded). */
        if(DomainTask && DomainTask[leaf] != ThisTask) n_owner_mismatch++;
        double h = (double)v.compact_xyzh[(size_t)p * 4 + 3];
        if(h > g_band[(size_t)leaf * TLR_NUM_PTYPES + type])
            g_band[(size_t)leaf * TLR_NUM_PTYPES + type] = h;
    }

    if(n_owner_mismatch > 0) {
        printf("topleaf_router: WARNING rank %d band has %ld supply particles whose leaf owner != self "
               "(mapping inconsistency)\n", ThisTask, n_owner_mismatch);
    }

    g_band_ntl     = ntl;
    g_band_numpart = v.numpart_when_built;
    g_band_ti      = v.ti_when_built;
    g_band_safety  = v.safety_when_built;
    g_band_mask    = v.eligible_mask_when_built;
    g_band_policy  = v.radius_policy_when_built;
    g_band_jscale  = v.j_scale_when_built;
    g_band_valid   = 1;
    return 0;
}

/* COLLECTIVE per-top-leaf GLOBAL band + per-topnode propagation (H4a), the SSOT
 * band producer for SYMMETRIC routing.
 *
 * Collective-safe by construction: every rank runs the SAME sequence of MPI
 * collectives regardless of its local data; the only branches are on already-
 * reduced all-or-none flags, so no rank returns (or skips a collective) before a
 * collective that other ranks will enter.  Caller MUST invoke it collectively
 * (all ranks, same step) behind a rank-uniform gate (env + search_mode SYMMETRIC).
 *
 *   *available        = 1  -> GLOBAL g_band + node band g_tn_band valid on ALL ranks
 *   *available        = 0  -> unavailable on >=1 rank; ALL ranks fall back to broadcast
 *   *consistency_fail = 1  -> the unavailability was a DOMAIN/TOPOLOGY CONSISTENCY BUG
 *                             (owner-map mismatch, non-uniform top-tree counts,
 *                             INT-count overflow, or topnode-layout violation), NOT a
 *                             benign not-ready (geometry/supply absent, alloc/OOM).
 * The caller picks policy: production may fall back on either; a VALIDATION oracle
 * MUST controlled-stop on consistency_fail (a real bug), and must not print OK on
 * any *available==0 (it could not validate).
 *
 * PRECONDITION (binding; asserted at the H4b call site, not here): the owned-local
 * supply cache (g_glt_cache, surfaced by ghost_exchange_supply_pool_view) MUST have
 * been (re)built for THE CALLER'S spec this step -- its eligible mask / radius policy
 * / safety / j-scale must correspond to this caller.  The band stores all 6 types per
 * leaf (the consumer masks by supply_mask at routing time), so it is spec-independent
 * EXCEPT through which particles populate the supply pool; a stale/superset/subset
 * supply cache would silently UNDER-ROUTE.  The epoch key below captures those supply
 * fields so a spec change forces a rebuild ONLY IF the supply cache was refreshed first.
 *
 * Band source is SSOT: per-leaf max of compact_xyzh[p*4+3] = the same baked
 * gx_policy_scaled_h the receiver walk uses.  The per-leaf band is made GLOBAL via
 * one Allreduce(MAX) (each rank fills only leaves it owns; others contribute 0),
 * then propagated up the replicated TopNodes octree (node_band = max over subtree
 * leaves).  Epoch-keyed: a collective near-no-op when topology + supply epoch are
 * unchanged. */
extern "C" void topleaf_router_band_build_collective(const int periodic_flags[3], const double box_sizes[3],
                                                     int collect_diag, int *available, int *consistency_fail)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(available)        *available = 0;
    if(consistency_fail) *consistency_fail = 0;

    /* --- Gate 1: geometry + supply present on ALL ranks (benign not-ready) --- */
    struct gx_supply_pool_view v;
    int num_pool   = ghost_exchange_supply_pool_view(&v, GX_POOL_IDENTITY | GX_POOL_GEOMETRY);
    int avail_local = (g_geom_valid && num_pool >= 0) ? 1 : 0;
    int avail_all   = 0;
    MPI_Allreduce(&avail_local, &avail_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!avail_all) {
        if(!avail_local)
            printf("topleaf_router: SYMM band not ready rank %d (geom_valid=%d num_pool=%d) -> all fall back to broadcast\n",
                   ThisTask, g_geom_valid, num_pool);
        return;   /* avail_all uniform -> all ranks return together (consistency_fail=0) */
    }

    const int ntl = g_geom_ntl;
    const int ntn = g_tn_count;

    /* --- Gate 2: epoch freshness (all-or-none collective skip) --- */
    int fresh_local = (g_cband_valid && g_cband_ntl == ntl && g_cband_ntn == ntn &&
                       g_cband_numpart == v.numpart_when_built &&
                       g_cband_ti      == v.ti_when_built &&
                       g_cband_safety  == v.safety_when_built &&
                       g_cband_mask    == v.eligible_mask_when_built &&
                       g_cband_policy  == v.radius_policy_when_built &&
                       g_cband_jscale  == v.j_scale_when_built) ? 1 : 0;
    int fresh_all = 0;
    MPI_Allreduce(&fresh_local, &fresh_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(fresh_all) { if(available) *available = 1; return; }   /* band + node band already valid */

    /* --- Gate 3: top-tree counts MUST be globally uniform + INT-count-safe.
     * The band Allreduce below uses ntl*6 as its MPI count; a non-uniform ntl/ntn
     * across ranks makes that collective malformed (UB/hang) and itself signals a
     * domain-topology consistency bug.  Overflow means the band exceeds an int MPI
     * count.  Both are HARD failures, surfaced loud -- never route on mismatched
     * counts.  All ranks see the SAME reduced dims -> all branch identically. */
    {
        int dims_local[2] = { ntl, ntn };
        int dims_min[2] = {0, 0}, dims_max[2] = {0, 0};
        MPI_Allreduce(dims_local, dims_min, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(dims_local, dims_max, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int nonuniform = (dims_min[0] != dims_max[0]) || (dims_min[1] != dims_max[1]);
        int overflow   = ((long)ntl * TLR_NUM_PTYPES > (long)INT_MAX) ||
                         ((long)ntn * TLR_NUM_PTYPES > (long)INT_MAX);
        if(nonuniform || overflow) {
            if(ThisTask == 0)
                printf("topleaf_router: SYMM band HARD FAIL (ntl[%d,%d] ntn[%d,%d] nonuniform=%d overflow=%d) -> all fall back\n",
                       dims_min[0], dims_max[0], dims_min[1], dims_max[1], nonuniform, overflow);
            if(consistency_fail) *consistency_fail = 1;
            return;
        }
    }

    /* --- rebuild: (re)alloc caches; alloc failure folded into the fail reduce --- */
    int alloc_fail_local = 0;
    if(g_band_cap < ntl) {
        g_band = shared_alloc_double(g_band, g_band_cap, ntl * TLR_NUM_PTYPES, "band");
        if(!g_band) { g_band_cap = 0; alloc_fail_local = 1; } else g_band_cap = ntl;
    }
    if(!alloc_fail_local && g_tn_band_cap < ntn) {
        g_tn_band = shared_alloc_double(g_tn_band, g_tn_band_cap, ntn * TLR_NUM_PTYPES, "tn_band");
        if(!g_tn_band) { g_tn_band_cap = 0; alloc_fail_local = 1; } else g_tn_band_cap = ntn;
    }

    /* --- TILE-BASED owned-leaf band attribution (walker-rank invariant).
     * Each owned-local supply TILE attaches its per-type reach to every leaf THIS RANK
     * OWNS whose cube overlaps the tile bbox inflated by the tile reach.  Routing
     * destination DomainTask[leaf]==ThisTask = the rank that actually walks the tile's
     * particles, so a particle that drifted across a top-leaf boundary still inflates
     * THIS rank's nearby owned leaves (no under-route) -- unlike a per-particle
     * current-leaf band, which loses walker-rank identity after drift.
     *
     * The fill is HIERARCHICAL: descend the TopNodes octree per tile to the
     * overlapping owned leaves (O(depth x fanout)), replacing a flat O(ntiles x
     * owned-leaves) scan that cost ~100M overlap tests/call at FIRE scale.
     * Conservative: it may over-cover, never under-cover. */
    long band_writes = 0, sum_leaves_per_tile = 0, max_leaves_per_tile = 0;
    long n_owned_leaves = 0;
    long zero_leaf_tiles = 0, zero_leaf_tiles_pos_h = 0;
    long band_overflow_local = 0;
    int *owned_leaves = NULL, *hier_buf = NULL;
    if(!alloc_fail_local) {
        owned_leaves = (int *) malloc((size_t)(ntl > 0 ? ntl : 1) * sizeof(int));
        hier_buf     = (int *) malloc((size_t)(ntl > 0 ? ntl : 1) * sizeof(int));
        if(!owned_leaves || !hier_buf) {
            alloc_fail_local = 1;
        } else {
            for(int leaf = 0; leaf < ntl; leaf++)
                if(DomainTask && DomainTask[leaf] == ThisTask) owned_leaves[n_owned_leaves++] = leaf;

            /* PRODUCTION: hierarchical fill into g_band. */
            for(int k = 0; k < ntl * TLR_NUM_PTYPES; k++) g_band[k] = 0.0;
            for(int t = 0; t < v.ntiles && !band_overflow_local; t++) {
                const struct sfc_tile_t *tile = &v.tiles[t];
                if(tile->count <= 0) continue;
                double reach = tile->hmax;
                int n = tlr_collect_owned_leaves_for_aabb(TopNodes, g_tn_count, g_tn_center, g_tn_len,
                                                          DomainTask, ThisTask, tile->lo, tile->hi, reach,
                                                          periodic_flags, box_sizes, hier_buf, ntl);
                if(n < 0) { band_overflow_local = 1; break; }   /* descent overflow -> benign fallback */
                for(int k = 0; k < n; k++) {
                    double *slot = &g_band[(size_t)hier_buf[k] * TLR_NUM_PTYPES];
                    for(int ty = 0; ty < TLR_NUM_PTYPES; ty++)
                        if(tile->hmax_by_type[ty] > slot[ty]) slot[ty] = tile->hmax_by_type[ty];
                    band_writes++;
                }
                sum_leaves_per_tile += n;
                if((long)n > max_leaves_per_tile) max_leaves_per_tile = n;
                if(n == 0) { zero_leaf_tiles++; if(reach > 0.0) zero_leaf_tiles_pos_h++; }
            }
        }
    }
    free(owned_leaves); free(hier_buf);

    /* --- drift DIAGNOSTIC (collect_diag only; load-bearing perf/correctness
     * instrumentation): count owned-local supply whose current-position top-leaf is
     * owned by ANOTHER rank (= drifted across a boundary since the last decomp).  This
     * is NO LONGER a failure -- the tile band handles it by construction -- but the
     * magnitude (vs top-leaf size) tells us whether the band reach is sufficient. */
    long n_drift_local = 0;
    if(collect_diag && !alloc_fail_local) {
        const double corner[3] = { DomainCorner[0], DomainCorner[1], DomainCorner[2] };
        const double inv_dlen  = (DomainLen > 0.0) ? (1.0 / DomainLen) : 0.0;
        const int    bits      = BITS_PER_DIMENSION;
        for(int p = 0; p < num_pool; p++) {
            int j = v.pool[p];
            if(j < 0 || j >= NumPart) continue;
            double pos[3] = { P[j].Pos[0], P[j].Pos[1], P[j].Pos[2] };
            int leaf = tlr_point_to_topleaf(TopNodes, pos, corner, inv_dlen, bits);
            if(leaf < 0 || leaf >= ntl) continue;
            if(!DomainTask || DomainTask[leaf] != ThisTask) n_drift_local++;
        }
    }

    /* --- Gate 4: benign resource fail (alloc OOM or hierarchical-descent overflow) on
     * ANY rank -> all fall back to broadcast (consistency_fail=0; not a correctness bug). --- */
    int benign_fail_local = (alloc_fail_local || band_overflow_local) ? 1 : 0;
    int benign_fail_any = 0;
    MPI_Allreduce(&benign_fail_local, &benign_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(benign_fail_any) {
        if(benign_fail_local)
            printf("topleaf_router: SYMM band rank %d FALLBACK (alloc_fail=%d hier_overflow=%ld) -> broadcast\n",
                   ThisTask, alloc_fail_local, band_overflow_local);
        g_cband_valid = 0;
        return;   /* benign_fail_any uniform -> all return together */
    }

    /* --- Gate 4b: UNDER-ROUTE guard.  A positive-reach owned-local supply tile that
     * attributed to ZERO walker-owned routing leaves means the tile band is
     * insufficient for those particles -> CONSISTENCY fail (a real correctness signal,
     * not benign).  Cheap and catches exactly the failure mode the tile band is meant
     * to exclude.  (alloc passed uniformly -> all ranks reach this reduce.) --- */
    long zlt_pos_any = 0;
    MPI_Allreduce(&zero_leaf_tiles_pos_h, &zlt_pos_any, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if(zlt_pos_any > 0) {
        if(zero_leaf_tiles_pos_h > 0)
            printf("topleaf_router: SYMM band rank %d UNDER-ROUTE risk: %ld positive-reach supply tiles "
                   "attributed to 0 walker-owned leaves -> consistency fail\n", ThisTask, zero_leaf_tiles_pos_h);
        if(consistency_fail) *consistency_fail = 1;
        g_cband_valid = 0;
        return;
    }

    /* --- cross-rank exchange: GLOBAL per-leaf band = MAX over owners --- */
    MPI_Allreduce(MPI_IN_PLACE, g_band, ntl * TLR_NUM_PTYPES, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    /* --- propagate up the replicated TopNodes octree: node_band = max(children).
     * Reverse-index sweep relies on Daughter+c > no (children appended after the
     * parent in force_create_empty_nodes); the child<=no guard makes a layout
     * change FAIL LOUD instead of reading an uncomputed slot. */
    int prop_fail_local = 0;
    for(int k = 0; k < ntn * TLR_NUM_PTYPES; k++) g_tn_band[k] = 0.0;
    for(int no = ntn - 1; no >= 0 && !prop_fail_local; no--) {
        double *nb = &g_tn_band[(size_t)no * TLR_NUM_PTYPES];
        if(TopNodes[no].Daughter < 0) {                 /* domain leaf */
            int leaf = TopNodes[no].Leaf;
            if(leaf < 0 || leaf >= ntl) { prop_fail_local = 1; break; }
            const double *lb = &g_band[(size_t)leaf * TLR_NUM_PTYPES];
            for(int t = 0; t < TLR_NUM_PTYPES; t++) nb[t] = lb[t];
        } else {                                        /* internal node */
            for(int c = 0; c < 8; c++) {
                int child = TopNodes[no].Daughter + c;
                if(child <= no || child >= ntn) { prop_fail_local = 1; break; }
                const double *cb = &g_tn_band[(size_t)child * TLR_NUM_PTYPES];
                for(int t = 0; t < TLR_NUM_PTYPES; t++) if(cb[t] > nb[t]) nb[t] = cb[t];
            }
        }
    }
    int prop_fail_any = 0;
    MPI_Allreduce(&prop_fail_local, &prop_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(prop_fail_any) {
        if(prop_fail_local)
            printf("topleaf_router: SYMM node-band propagation FAIL rank %d (topnode layout) -> all fall back\n", ThisTask);
        if(consistency_fail) *consistency_fail = 1;
        g_cband_valid = 0;
        return;
    }

    g_cband_ntl     = ntl;
    g_cband_ntn     = ntn;
    g_cband_numpart = v.numpart_when_built;
    g_cband_ti      = v.ti_when_built;
    g_cband_safety  = v.safety_when_built;
    g_cband_mask    = v.eligible_mask_when_built;
    g_cband_policy  = v.radius_policy_when_built;
    g_cband_jscale  = v.j_scale_when_built;
    g_cband_valid   = 1;
    /* Keep the flat-band accessor consistent: g_band now holds the GLOBAL band. */
    g_band_ntl   = ntl;
    g_band_valid = 1;
    if(available) *available = 1;

    /* --- DIAGNOSTICS (collect_diag, success path only): drift signal + over-route
     * guardrails.  collect_diag is rank-uniform and we only reach here on the uniform
     * success path, so these collectives stay symmetric.  band_writes / leaves-per-tile
     * bound the over-attribution (the SYMM perf risk); drifted_supply confirms the
     * boundary-drift magnitude the tile band absorbs. */
    if(collect_diag) {
        long diag_local[7] = { band_writes, sum_leaves_per_tile, max_leaves_per_tile,
                               n_drift_local, (long)v.ntiles, n_owned_leaves,
                               zero_leaf_tiles };
        long diag_sum[7] = {0,0,0,0,0,0,0}, diag_max[7] = {0,0,0,0,0,0,0};
        MPI_Allreduce(diag_local, diag_sum, 7, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(diag_local, diag_max, 7, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
        if(ThisTask == 0) {
            double avg_lpt = (diag_sum[4] > 0) ? (double)diag_sum[1] / (double)diag_sum[4] : 0.0;
            /* band_writes is the hierarchical production fill. */
            printf("[SYMM_BAND_DIAG ntl=%d ntn=%d tiles(sum)=%ld owned_leaves(sum)=%ld "
                   "band_writes(sum)=%ld leaves/tile avg=%.2f max=%ld zero_leaf_tiles(sum)=%ld "
                   "drifted_supply(sum)=%ld]\n",
                   ntl, ntn, diag_sum[4], diag_sum[5], diag_sum[0], avg_lpt, diag_max[2],
                   diag_sum[6], diag_sum[3]);
            fflush(stdout);
        }
    }
}

static int tlr_cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* Per-leaf GLOBAL band distribution for a SYMM caller's supply_mask -- answers
 * "is the over-route a few giant leaves (tail) or many moderately-inflated leaves
 * (broad)?".  The global band is identical on every rank (post Allreduce MAX), so
 * rank 0 reports; NO collective.  band_eff[leaf] = max over supply_mask types;
 * the load-bearing signal is band_eff / leaf_len (how far the band reaches beyond
 * its own cube) -- percentiles + threshold counts + which type dominates. */
extern "C" void topleaf_router_band_distribution_report(unsigned int supply_mask, const char *caller, int this_call)
{
    if(ThisTask != 0) return;
    if(!g_cband_valid) return;
    const int ntl = g_cband_ntl;
    if(ntl <= 0) return;
    double *band_eff = (double *) malloc((size_t)ntl * sizeof(double));
    double *norm     = (double *) malloc((size_t)ntl * sizeof(double));
    if(!band_eff || !norm) { free(band_eff); free(norm); return; }
    long type_dom[TLR_NUM_PTYPES] = {0,0,0,0,0,0};
    long c1 = 0, c4 = 0, c16 = 0, c64 = 0;
    int nv = 0;
    for(int leaf = 0; leaf < ntl; leaf++) {
        const double *b = &g_band[(size_t)leaf * TLR_NUM_PTYPES];
        double be = 0; int dom = -1;
        for(int t = 0; t < TLR_NUM_PTYPES; t++) {
            if((supply_mask & (1u << (unsigned)t)) == 0u) continue;
            if(b[t] > be) { be = b[t]; dom = t; }
        }
        if(be <= 0) continue;                       /* leaf carries no band for this mask */
        double len = g_leaf_len[leaf];
        double nrm = (len > 0) ? be / len : 0.0;
        band_eff[nv] = be;
        norm[nv]     = nrm;
        nv++;
        if(dom >= 0) type_dom[dom]++;
        if(nrm > 1.0)  c1++;
        if(nrm > 4.0)  c4++;
        if(nrm > 16.0) c16++;
        if(nrm > 64.0) c64++;
    }
    if(nv <= 0) { free(band_eff); free(norm); return; }
    qsort(band_eff, (size_t)nv, sizeof(double), tlr_cmp_double);
    qsort(norm,     (size_t)nv, sizeof(double), tlr_cmp_double);
    #define TLR_PCT(arr, p) ((arr)[(int)((p) * (double)(nv - 1))])
    printf("[SYMM_BAND_DIST call=%d caller=%s mask=0x%x leaves_with_band=%d/%d | "
           "band p50=%.3g p90=%.3g p99=%.3g p99.9=%.3g max=%.3g | "
           "band/len p50=%.2f p90=%.2f p99=%.2f p99.9=%.2f max=%.2f | "
           ">1x=%ld >4x=%ld >16x=%ld >64x=%ld | "
           "type_dom=[%ld,%ld,%ld,%ld,%ld,%ld]]\n",
           this_call, (caller ? caller : "?"), supply_mask, nv, ntl,
           TLR_PCT(band_eff, 0.50), TLR_PCT(band_eff, 0.90), TLR_PCT(band_eff, 0.99), TLR_PCT(band_eff, 0.999), band_eff[nv-1],
           TLR_PCT(norm, 0.50), TLR_PCT(norm, 0.90), TLR_PCT(norm, 0.99), TLR_PCT(norm, 0.999), norm[nv-1],
           c1, c4, c16, c64,
           type_dom[0], type_dom[1], type_dom[2], type_dom[3], type_dom[4], type_dom[5]);
    fflush(stdout);
    #undef TLR_PCT
    free(band_eff); free(norm);
}
