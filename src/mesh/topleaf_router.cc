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

static double *shared_alloc_double(double *old, int old_cap, int new_cap, const char *what)
{
    if(old) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(old);
    (void)old_cap;
    long bytes = (long)new_cap * (long)sizeof(double);
    double *p = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bytes);
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
    g_band_valid = 0;
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
}

/* Mark the geometry cache (and the topology-keyed band) INVALID without freeing
 * storage.  Call from domain/tree rebuild boundaries so a missing re-acquire
 * fails closed to broadcast instead of routing on stale geometry. */
extern "C" void topleaf_router_geometry_invalidate(void) { g_geom_valid = 0; g_band_valid = 0; }

extern "C" int           topleaf_router_geometry_valid(void)  { return g_geom_valid; }
extern "C" const double *topleaf_router_leaf_center(void)      { return g_geom_valid ? g_leaf_center : NULL; }
extern "C" const double *topleaf_router_leaf_len(void)         { return g_geom_valid ? g_leaf_len    : NULL; }
extern "C" int           topleaf_router_ntopleaves(void)       { return g_geom_ntl; }

extern "C" void topleaf_router_band_invalidate(void) { g_band_valid = 0; }
extern "C" int  topleaf_router_band_valid(void)      { return g_band_valid; }
extern "C" const double *topleaf_router_band(int *ntl_out)
{
    if(ntl_out) *ntl_out = g_band_ntl;
    return g_band_valid ? g_band : NULL;
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

/* H1 flat-vs-hierarchical owner-set oracle (ONEWAY): for each query build the
 * flat (leaf-scan) and hierarchical (tree-descent) REMOTE owner sets on the SAME
 * geometry and compare as SETS.  Returns the count of queries whose sets differ
 * (0 = all equal), or a NEGATIVE code if UNAVAILABLE (distinct so the caller can
 * report a reason and FAIL LOUD — a validation oracle must never look green when
 * it could not validate):  -1 = geometry invalid, -2 = scratch alloc fail,
 * -3 = hierarchical stack overflow.  *first_bad = first mismatching query index
 * (or -1).  Pure host check; caller does the collective Allreduce + controlled-stop. Validates
 * the hierarchical traversal + per-topnode geometry against the trusted flat
 * router before the hierarchical constructor is ever made authoritative. */
extern "C" int topleaf_router_hier_vs_flat_check(
    const double *q_pos, const double *q_h, int nq,
    const int periodic_flags[3], const double box_sizes[3], int self_rank,
    int *first_bad)
{
    if(first_bad) *first_bad = -1;
    if(!g_geom_valid) return -1;
    const int ntask = NTask;
    int  *flat_owners = (int *)  malloc((size_t)(ntask > 0 ? ntask : 1) * sizeof(int));
    int  *hier_owners = (int *)  malloc((size_t)(ntask > 0 ? ntask : 1) * sizeof(int));
    char *seen_f = (char *) calloc((size_t)(ntask > 0 ? ntask : 1), sizeof(char));
    char *seen_h = (char *) calloc((size_t)(ntask > 0 ? ntask : 1), sizeof(char));
    char *member = (char *) calloc((size_t)(ntask > 0 ? ntask : 1), sizeof(char));
    if(!flat_owners || !hier_owners || !seen_f || !seen_h || !member) {
        free(flat_owners); free(hier_owners); free(seen_f); free(seen_h); free(member);
        return -2;   /* scratch alloc failure */
    }
    int mismatch = 0, unavailable = 0;
    for(int i = 0; i < nq; i++) {
        const double *pos = &q_pos[(size_t)i * 3];
        double h = q_h[i];
        /* flat (ONEWAY band=NULL); includes self -> filtered below */
        int nf = tlr_route_query_over_topleaves(g_leaf_center, g_leaf_len, DomainTask, g_geom_ntl, ntask,
                                                pos, h, NULL, 0u, periodic_flags, box_sizes, flat_owners, seen_f);
        int nh = tlr_route_query_hierarchical(TopNodes, g_tn_count, g_tn_center, g_tn_len, DomainTask, ntask,
                                              pos, h, periodic_flags, box_sizes, self_rank, hier_owners, seen_h);
        for(int k = 0; k < nf; k++) seen_f[flat_owners[k]] = 0;   /* reset dedup scratch */
        if(nh >= 0) for(int k = 0; k < nh; k++) seen_h[hier_owners[k]] = 0;
        if(nh < 0) { unavailable = 1; break; }                    /* hier overflow -> bail, not a mismatch */
        /* flat REMOTE set in member[]; compare to hier set */
        int nf_remote = 0;
        for(int k = 0; k < nf; k++) { int o = flat_owners[k]; if(o == self_rank) continue; member[o] = 1; nf_remote++; }
        int eq = (nf_remote == nh);
        if(eq) for(int k = 0; k < nh; k++) { if(!member[hier_owners[k]]) { eq = 0; break; } }
        for(int k = 0; k < nf; k++) { if(flat_owners[k] != self_rank) member[flat_owners[k]] = 0; }  /* reset */
        if(!eq) { mismatch++; if(first_bad && *first_bad < 0) *first_bad = i; }
    }
    free(flat_owners); free(hier_owners); free(seen_f); free(seen_h); free(member);
    return unavailable ? -3 : mismatch;   /* -3 = hierarchical stack overflow */
}

/* Build the per-top-leaf supply reach band hmax_by_type[] for SYMMETRIC callers,
 * from the owned-local supply pool via gx_policy_scaled_h (already baked into
 * compact_xyzh[p*4+3]).  Epoch-keyed to the supply cache: a no-op fast return
 * when the epoch is unchanged.  Returns 0 if valid, nonzero if unavailable. */
extern "C" int topleaf_router_band_build(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    if(!g_geom_valid) return 1;   /* need leaf geometry/topology first */

    struct gx_supply_pool_view v;
    int num_pool = ghost_exchange_supply_pool_view(&v);
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
