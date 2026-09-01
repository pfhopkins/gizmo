/*! \file ghost_exchange.cc
 *  \brief Ghost particle exchange for GPU-ready neighbor finding.
 *
 *  Replaces the pseudo-particle export mechanism with an upfront "import-the-neighbors"
 *  pattern: before any neighbor loop, exchange boundary particles between MPI ranks so
 *  that all neighbors are local. Subsequent kernels iterate over local + ghost particles
 *  without secondary MPI phases.
 *
 *  Ghost particles are appended to P[] and CellP[] arrays at indices >= NumPart_before_ghost.
 *  After all neighbor operations complete, ghost_exchange_cleanup() resets NumPart/N_gas.
 *
 *  This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 *
 *  KNOWN LIMITATIONS (to be optimized):
 *  - Sends full P[i]/CellP[i] structs per ghost (~2-5 KB/particle). Should use compact
 *    struct with only fields needed by the active kernel set (~200 bytes).
 *  - O(NTopleaves^2) overlap check between local and remote leaves. Should use spatial
 *    sorting or tree-based pruning for simulations with many top-level leaves.
 *  - Global MPI_Allreduce on need_leaf (NTopleaves ints). Could use point-to-point for
 *    sparse communication patterns.
 *  - Per-task routing is approximate: uses MPI_Allreduce(MPI_MAX) on need_leaf, so a leaf
 *    requested by ANY task is sent to ALL requesting tasks. Per-task leaf request lists
 *    would reduce traffic.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../declarations/lifecycle_counters.h"
#include "../core/proto.h"
#include "../system/mpi_alltoallv_typed.h"
#include "gpu_neighbor_list.h" /* gpu_compact_xyzh_mark_h_dirty_range */
#include "sfc_tiles.h"           /* build_sfc_tiles, build_tile_bvh, sfc_tile_t, tile_bvh_node_t */
#include "neighbor_list.h"       /* NGB_SEARCH_ONEWAY, NGB_SEARCH_SYMMETRIC */
#include "ghost_exchange_functions.h" /* gx_pair_accept_wrap_and_test: shared accept, wraps via the canonical macros */
#include "ghost_writeback.h"     /* ghost_get_num_local (bounded fine-tree walk) */
#include "ghost_exchange_spec.h"
#include "mode_b_local_walker.h"
#ifdef _OPENMP
#include <omp.h>                 /* threaded sender export + receiver walk below */
#endif

/* Defined in neighbor_loop_runner.cc. Declared here rather than including the
 * runner header, which is a heavy template TU. Lets this file's diagnostic output
 * share ONE gate with PHASE0_NLR, so Mode-A and Mode-B runs are collectable in the
 * same configuration and remain comparable. */
bool gizmo_nlr_phase0_diag_enabled(void);

/*
 * ============================================================================
 * COMPACT GHOST STRUCT FIELD REQUIREMENTS (for future optimization)
 *
 * Currently sends full P[i] + CellP[i] structs. When optimizing, the minimum
 * fields needed per kernel are:
 *
 * ALL kernels need from P: Pos[3], Mass, Type, KernelRadius, NumNgb, TimeBin
 *
 * density_evaluate additionally needs:
 *   P: Vel[3]
 *   CellP: VelPred[3], InternalEnergyPred, Density
 *   + #ifdef MAGNETIC: CellP.BPred[3]
 *   + #ifdef COSMIC_RAY_FLUID: CellP.CosmicRayEnergyPred[N_CR_PARTICLE_BINS]
 *   + #ifdef RADTRANSFER: CellP.Rad_E_gamma[N_RT_FREQ_BINS], Rad_E_gamma_Pred
 *
 * hydro_gradient_calc additionally needs:
 *   All of density fields, plus:
 *   CellP: Pressure, MaxSignalVel
 *
 * hydro_force_evaluate additionally needs:
 *   All of gradient fields, plus:
 *   CellP: DtInternalEnergy, HydroAccel[3], SoundSpeed, Gradients
 *   P: GravAccel[3]
 *   + #ifdef DIVBCLEANING_DEDNER: CellP.PhiPred
 *   + all MHD/RT/CR evolved quantities
 * ============================================================================
 */

/* saved state for cleanup */
static int NumPart_before_ghost = -1;
static int N_gas_before_ghost = -1;
static int NumGhostParticles = 0;
/* The largest ghost import this rank has completed, over the epoch running now and the one before
 * it.  The capacity a rank needs is set by its worst import, not its most recent one, and ghost
 * demand is strongly uneven between ranks, so this is kept per rank and is the measured term the
 * epoch sizing asks for.  Two epochs rather than one because a single quiet epoch would otherwise
 * be enough to justify releasing storage that the next one immediately asks for again, and paying
 * two migrations to save memory for one epoch is a poor trade.  Neither value is carried across a
 * restart: a restored capacity is already whatever the run had grown to, and the sizing refuses to
 * lower it until it has observed an import. */
static int GhostEpochHighWater = 0;
static int GhostPreviousEpochHighWater = 0;

/* Ghost provenance map: for each ghost particle, the home MPI rank and index.
   Used by ghost_writeback to reverse-communicate j-particle modifications.
   Allocated with malloc (not mymalloc) to avoid stack ordering issues. */
static int *ghost_home_rank_map = NULL;     /* [NumGhostParticles] home MPI rank */
static int *ghost_home_index_map = NULL;    /* [NumGhostParticles] home P[]/CellP[] index */
static int *ghost_wb_recv_count = NULL;     /* [NTask] ghosts received from each rank */
static int *ghost_wb_recv_disp = NULL;      /* [NTask] displacement by source rank */
static int *ghost_wb_send_count = NULL;     /* [NTask] ghosts we sent to each rank */
static int *ghost_wb_send_disp = NULL;      /* [NTask] displacement for what each rank got from us */

/* Send-side provenance for ghost_refresh_values(): the ordered list of LOCAL
   indices this rank exported at the last import (grouped by ghost_wb_send_*),
   so a value-only refresh can re-pack current owner P/CellP without re-running
   discovery. Preserved unconditionally at import (ownership taken from the
   per-import send_home_idx buffer), freed in ghost_exchange_cleanup(). The actual
   refresh guard is (non-NULL ghost_send_home_idx) + (send/recv totals match the
   live pool): a non-NULL pointer implies "an import happened and no cleanup since"
   (cleanup NULLs it). The monotonic epoch below counts completed imports; it has
   two consumers: (a) the refresh diagnostic harness asserts a value-refresh does
   NO reimport (epoch unchanged) while a full cleanup+reimport bumps it; (b) the
   hydro corridor records the epoch its published CSR was built from and permits
   the value-refresh fast path ONLY on an epoch match — a live pool from some
   OTHER import could pass the count checks by coincidence while the CSR still
   indexes the old slot layout. */
static int *ghost_send_home_idx = NULL;     /* [ghost_send_home_count] exported local indices, send order */
static int  ghost_send_home_count = 0;      /* == total_send at last import */
static unsigned long long g_ghost_provenance_epoch = 0; /* import counter (see above) */

/* Persistent local-tree cache (SIDX overlay) for the
 * request-driven ghost exchange path. Within a step, the local pool of
 * particles [0..NumPart_local) is stable across multiple ghost_exchange
 * calls (3-5 calls/step typical). Building tiles+BVH+compact_xyzh once per
 * call costs ~0.19s (gas) / ~0.65s (all-types) on the fire_m11i 6.2M/9.5M
 * pool. Caching them across calls saves N-1 of those builds per step.
 *
 * Invalidation is wired to the same hooks as gpu_step_sidx_invalidate_*:
 *   - run.cc post-drift            -> ghost_exchange_local_tree_invalidate_drift()
 *   - the decomposition itself     -> ghost_exchange_local_tree_invalidate_full()
 * Drift/h updates mark the cache for exact refit from P[] on the next hit;
 * domain decomposition and pool/ordering changes fully free it.
 *
 * Cache key (NumPart, safety_factor, eligible pool mask) is also checked
 * at use time as a defensive cross-check; pool/ordering changes force a
 * rebuild. Drift/h changes mark the cache for an exact refit from P[].
 *
 * Memory footprint: ~165MB (Type-0/cell pool, 6.2M pool) or ~250MB (all-types,
 * 9.5M pool) per rank. Tolerable on Vista host. Allocated via plain
 * malloc/free to avoid mymalloc-stack LIFO violation when the cache
 * outlives the function frame. */
/* Membership/order epoch for the supply pool.  Rank-local and compared only for
 * equality: it answers "is the pool I cached still the same set, in the same
 * order?", nothing more.  Bumped ONLY where particles are created, eliminated,
 * or moved between slots (rearrange_particle_sequence); NEVER by drift, h, or a
 * radius policy, none of which can change membership.  64-bit so wraparound is
 * not a case anyone has to reason about. */
static long long g_supply_identity_epoch = 0;

extern "C" void ghost_exchange_supply_identity_changed(const char *reason)
{
    (void)reason;   /* named at the call site so the reason is greppable there */
    g_supply_identity_epoch++;
}

/* Which parts of the supply cache a consumer requires.  IDENTITY (pool,
 * pool_types, num_pool) is membership only: type-mask + positive mass, so it
 * stays valid as particles move.  GEOMETRY (compact_xyzh, tiles) carries
 * positions and the baked per-member reach, and is built only for the walkers
 * that need it.  A consumer that reads geometry without requesting it would
 * silently read NULL, so the request is mandatory and checked. */
#define GX_POOL_IDENTITY  0x1u
#define GX_POOL_GEOMETRY  0x2u

struct ghost_local_tree_cache_t {
    int valid;
    int NumPart_when_built;
    /* Identity generation this entry's pool/j_to_pool were built against. */
    long long identity_epoch_when_built;
    integertime Ti_when_built;
    double safety_factor_when_built;
    unsigned int eligible_type_mask_when_built;
    int needs_refit;
    /* What this cache entry actually holds (GX_POOL_IDENTITY / GX_POOL_GEOMETRY).
     * Geometry is skipped for calls whose producer never walks it, so `valid`
     * alone does not imply tiles/bvh/compact_xyzh are present. */
    unsigned int caps;
    int ntiles;
    int num_pool;
    int bvh_nnodes;
    int bvh_root;
    sfc_tile_t *tiles;            /* [ntiles] malloc */
    int *pool;                     /* [num_pool] malloc */
    tile_bvh_node_t *bvh;          /* [bvh_nnodes] malloc */
    float *compact_xyzh;           /* [num_pool*4] malloc; h field = supply-side policy * j_scale * safety baked in */
    int *pool_types;               /* [num_pool] malloc */
    int *j_to_pool;                /* [NumPart_when_built] malloc, j -> pool_pos or -1 */
    /* SSOT supply-side reach contract. These four fields, together with the
     * (NumPart, safety, eligible_pool_mask) triple above, form the cache-key
     * invariant: any mismatch on any of them forces a full rebuild (NOT a
     * refit).  Ti and the needs_refit dirty bit continue to trigger
     * glt_cache_refit_from_particles() instead of a rebuild — Ti is NOT a
     * rebuild key: rebuilding every step would be exactly the work
     * explosion this design is supposed to prevent. */
    mode_b_radius_policy_t radius_policy_when_built;
    double j_radius_scale_when_built;
};
/* Designated initialisers: the positional form silently mis-assigns whenever a
 * field is added to the struct above. */
static struct ghost_local_tree_cache_t g_glt_cache = {
    .valid = 0,
    .NumPart_when_built = -1,
    .identity_epoch_when_built = -1,
    .Ti_when_built = -1,
    .safety_factor_when_built = 0.0,
    .eligible_type_mask_when_built = GHOST_TYPE_ALL,
    .needs_refit = 0,
    .caps = 0u,
    .ntiles = 0,
    .num_pool = 0,
    .bvh_nnodes = 0,
    .bvh_root = 0,
    .tiles = NULL,
    .pool = NULL,
    .bvh = NULL,
    .compact_xyzh = NULL,
    .pool_types = NULL,
    .j_to_pool = NULL,
    .radius_policy_when_built = MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES,
    .j_radius_scale_when_built = 1.0,
};

/* gx_policy_scaled_h — SSOT supply-side reach inside ghost_exchange.cc.
 *
 * Single helper used at every site that writes a per-particle supply h into
 * the ghost local-tree cache: fresh build (compact + tile bands via
 * build_sfc_tiles' scale_factor pathway), refit (glt_recompute_tile_), and
 * any future site that needs the supply-side reach.  The output is identical
 * to what build_sfc_tiles aggregates internally when called with
 *   (radius_policy, scale_factor = j_radius_scale * safety_factor)
 * so leaf compact h and BVH band hmax see the exact same supply-side reach
 * for every particle — closing the leaf-vs-band scale gap that would
 * otherwise let the BVH prune pairs the leaf would have accepted.
 *
 * Compile-flag gating on AGS_KernelRadius lives in nlr_radius_policy.h's
 * wrapper; never replicate the `#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE`
 * inside this TU. */
static inline double gx_policy_scaled_h(int j,
                                        mode_b_radius_policy_t radius_policy,
                                        double j_radius_scale,
                                        double safety_factor)
{
    return nlr_particle_symmetric_radius(P[j], radius_policy)
           * j_radius_scale * safety_factor;
}

/* Bucket 3 narrow-refit machinery. Mirrors gpu_neighbor_list.cc's g_dirty_list
 * pattern for the GPU compact_xyzh refresh — same call sites populate both, but
 * different consumers (host ghost_exchange cache vs device GPU NL builder), so
 * the two lists have independent lifecycles.
 *
 *  - g_glt_dirty_all = true  : full refit needed (drift, fresh build seed, or
 *    list overflow promotion). Default = true so first refit is full.
 *  - g_glt_dirty_all = false : refresh only the indices in g_glt_dirty_list.
 *    Indices outside the cache pool (j_to_pool[j] == -1) are skipped cleanly.
 *
 * Promote-to-all threshold matches GPU side (1M indices). When the list grows
 * past that, narrow refit costs ~the same as full, so flip to dirty_all. */
static const int G_GLT_DIRTY_PROMOTE_THRESHOLD = 1 << 20; /* 1M indices */
static bool g_glt_dirty_all = true;
static std::vector<int> g_glt_dirty_list;
static inline void g_glt_dirty_clear_(void)
{
    g_glt_dirty_all = false;
    g_glt_dirty_list.clear();
}
static inline void g_glt_dirty_mark_all_(void)
{
    g_glt_dirty_all = true;
    g_glt_dirty_list.clear();
}

/* Diagnostic counters. */
static long g_glt_cache_hits = 0;
static long g_glt_cache_misses = 0;
static long g_glt_cache_refits = 0;
static long g_glt_cache_narrow_refits = 0;

static void glt_cache_free(void)
{
    if(g_glt_cache.tiles)        { free(g_glt_cache.tiles);        g_glt_cache.tiles = NULL; }
    if(g_glt_cache.pool)         { free(g_glt_cache.pool);         g_glt_cache.pool = NULL; }
    if(g_glt_cache.bvh)          { free(g_glt_cache.bvh);          g_glt_cache.bvh = NULL; }
    if(g_glt_cache.compact_xyzh) { free(g_glt_cache.compact_xyzh); g_glt_cache.compact_xyzh = NULL; }
    if(g_glt_cache.pool_types)   { free(g_glt_cache.pool_types);   g_glt_cache.pool_types = NULL; }
    if(g_glt_cache.j_to_pool)    { free(g_glt_cache.j_to_pool);    g_glt_cache.j_to_pool = NULL; }
    g_glt_cache.valid = 0;
    g_glt_cache.NumPart_when_built = -1;
    g_glt_cache.identity_epoch_when_built = -1;
    g_glt_cache.Ti_when_built = -1;
    g_glt_cache.safety_factor_when_built = 0.0;
    g_glt_cache.eligible_type_mask_when_built = GHOST_TYPE_ALL;
    g_glt_cache.radius_policy_when_built = MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES;
    g_glt_cache.j_radius_scale_when_built = 1.0;
    g_glt_cache.needs_refit = 0;
    g_glt_cache.caps = 0u;
    g_glt_cache.ntiles = 0;
    g_glt_cache.num_pool = 0;
    g_glt_cache.bvh_nnodes = 0;
    g_glt_cache.bvh_root = 0;
    /* Cache gone -> no narrow-refit basis remains; force full on next build. */
    g_glt_dirty_mark_all_();
}

/* Drop ONLY the position/radius-dependent half, keeping pool/j_to_pool/num_pool.
 * Used when a caller needs geometry under a different radius policy or scale:
 * membership does not depend on those, so re-deriving it would be pure waste. */
static void glt_cache_free_geometry_(void)
{
    if(g_glt_cache.tiles)        { free(g_glt_cache.tiles);        g_glt_cache.tiles = NULL; }
    if(g_glt_cache.bvh)          { free(g_glt_cache.bvh);          g_glt_cache.bvh = NULL; }
    if(g_glt_cache.compact_xyzh) { free(g_glt_cache.compact_xyzh); g_glt_cache.compact_xyzh = NULL; }
    if(g_glt_cache.pool_types)   { free(g_glt_cache.pool_types);   g_glt_cache.pool_types = NULL; }
    g_glt_cache.ntiles = 0;
    g_glt_cache.bvh_nnodes = 0;
    g_glt_cache.bvh_root = 0;
    g_glt_cache.caps &= ~GX_POOL_GEOMETRY;
}

extern "C" void ghost_exchange_local_tree_invalidate_drift(void)
{
    if(g_glt_cache.valid) g_glt_cache.needs_refit = 1;
    /* Drift is a pool-wide event (every particle's Pos may have changed):
     * the narrow-refit fast path can't represent that, so promote to full. */
    g_glt_dirty_mark_all_();
}
extern "C" void ghost_exchange_local_tree_invalidate_full(void)  { glt_cache_free(); }


extern "C" void ghost_exchange_local_tree_mark_h_dirty_indices(const int *indices, int n)
{
    if(n <= 0 || !indices) return;
    if(g_glt_dirty_all) return; /* already covered by full-refit promotion */
    if((int)(g_glt_dirty_list.size() + (size_t)n) > G_GLT_DIRTY_PROMOTE_THRESHOLD) {
        g_glt_dirty_mark_all_();
        return;
    }
    g_glt_dirty_list.reserve(g_glt_dirty_list.size() + (size_t)n);
    for(int k = 0; k < n; k++) {
        int j = indices[k];
        if(j >= 0) g_glt_dirty_list.push_back(j);
    }
    /* Mark cache for refit-on-next-hit even though it isn't fully invalidated.
     * This wakes up the refit branch in the request-driven build path. */
    if(g_glt_cache.valid) g_glt_cache.needs_refit = 1;
}

extern "C" void ghost_exchange_local_tree_mark_h_dirty_range(int start, int end)
{
    if(end <= start) return;
    if(g_glt_dirty_all) return;
    int n = end - start;
    if((int)(g_glt_dirty_list.size() + (size_t)n) > G_GLT_DIRTY_PROMOTE_THRESHOLD) {
        g_glt_dirty_mark_all_();
        return;
    }
    g_glt_dirty_list.reserve(g_glt_dirty_list.size() + (size_t)n);
    for(int j = start; j < end; j++) g_glt_dirty_list.push_back(j);
    if(g_glt_cache.valid) g_glt_cache.needs_refit = 1;
}

/* Stage-3 producer mode (Step 5).  Selects HOW the matched ghost set is produced;
 * Stages 1-2 (route CSR -> Alltoall -> Alltoallv -> received queries) are shared
 * SSOT regardless of mode.  HOST_ONLY = host walk only; HOST_AND_DEVICE_VALIDATE =
 * host walk + device Stage-3 compared (the Step-4 oracle); DEVICE_ONLY_AUTHORITY =
 * production device Stage-3 with NO host walk / NO broadcast / NO compare (reserved
 * for the device-routed install arm; no caller uses it yet). */
enum gx_producer_mode {
    GX_PRODUCER_HOST_ONLY = 0,
    GX_PRODUCER_HOST_AND_DEVICE_VALIDATE,
    GX_PRODUCER_DEVICE_ONLY_AUTHORITY
};

/* Recompute one tile's lo/hi/hmax/hmax_by_type fully from current P[] over its
 * pool members.  Also rewrites compact_xyzh + pool_types for those members.
 * Used by both full and narrow refit paths. */
static inline void glt_recompute_tile_(int t)
{
    sfc_tile_t *tile = &g_glt_cache.tiles[t];
    tile->hmax = 0;
    for(int tt = 0; tt < TILE_NUM_PTYPES; tt++) tile->hmax_by_type[tt] = 0;
    /* Empty tiles carry an INVERTED box so they stay neutral under the BVH's
     * min/max union and always fail the sphere-overlap test; a zeroed box would
     * stretch every ancestor to the coordinate origin and defeat pruning there. */
    if(tile->count <= 0) {
        for(int k = 0; k < 3; k++) { tile->lo[k] = MAX_REAL_NUMBER; tile->hi[k] = -MAX_REAL_NUMBER; }
        return;
    }
    /* SSOT supply-side reach pulled from the cache's stored policy/scale.
     * For runner-driven imports this carries the
     * Spec's radius_policy + nlr_spec_symmetric_j_radius_scale<Spec>();
     * for legacy ghost_exchange wrappers it carries
     * MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES + 1.0 → byte-equivalent to the
     * pre-policy code that read P[j].KernelRadius * safety_factor.
     * Leaf compact_xyzh[p*4+3] and tile band hmax_by_type[] use the
     * IDENTICAL gx_policy_scaled_h output → no leaf-vs-band scale gap. */
    const mode_b_radius_policy_t policy = g_glt_cache.radius_policy_when_built;
    const double j_scale = g_glt_cache.j_radius_scale_when_built;
    const double safety  = g_glt_cache.safety_factor_when_built;
    /* Eliminated elements (Mass <= 0) are excluded from the band and the bbox:
     * the pool is mass-filtered when built, but a cached entry outlives a
     * Mass->0 marking until rearrange_particle_sequence() compacts it away, and
     * a dead slot's stale reach would otherwise widen what this rank advertises
     * as supply.  Dropping it can only remove pairs WITH the dead element, which
     * must not be discovered anyway; live pairs carry their own reach.  The leaf
     * walk applies the same test, so leaf h and band hmax stay on one reach.
     * The bbox is seeded from the first live member rather than the tile's first
     * member, which may itself be dead. */
    int seeded = 0;
    for(int s = 0; s < tile->count; s++) {
        int p = tile->first + s;
        int j = g_glt_cache.pool[p];
        int pt = (int)P[j].Type;
        double h = gx_policy_scaled_h(j, policy, j_scale, safety);
        g_glt_cache.compact_xyzh[p*4+0] = (float)P[j].Pos[0];
        g_glt_cache.compact_xyzh[p*4+1] = (float)P[j].Pos[1];
        g_glt_cache.compact_xyzh[p*4+2] = (float)P[j].Pos[2];
        g_glt_cache.compact_xyzh[p*4+3] = (float)h;
        g_glt_cache.pool_types[p] = pt;
        if(P[j].Mass <= 0) continue;
        if(!seeded) {
            for(int k = 0; k < 3; k++) tile->lo[k] = tile->hi[k] = P[j].Pos[k];
            seeded = 1;
        } else {
            for(int k = 0; k < 3; k++) {
                if(P[j].Pos[k] < tile->lo[k]) tile->lo[k] = P[j].Pos[k];
                if(P[j].Pos[k] > tile->hi[k]) tile->hi[k] = P[j].Pos[k];
            }
        }
        if(h > tile->hmax) tile->hmax = h;
        if(pt >= 0 && pt < TILE_NUM_PTYPES && h > tile->hmax_by_type[pt])
            tile->hmax_by_type[pt] = h;
    }
    /* Every member dead: same inverted empty-tile box as count <= 0 above. */
    if(!seeded) { for(int k = 0; k < 3; k++) { tile->lo[k] = MAX_REAL_NUMBER; tile->hi[k] = -MAX_REAL_NUMBER; } }
}

/* Pull one BVH node's lo/hi/hmax/hmax_by_type from its children/leaf-tile.
 * BVH is in children-first order so calling this for indices 0..bvh_nnodes-1
 * in order updates internal nodes after their children. */
static inline void glt_recompute_bvh_node_(int n)
{
    tile_bvh_node_t *node = &g_glt_cache.bvh[n];
    if(node->left < 0) {
        int t = -(node->left + 1);
        if(t < 0 || t >= g_glt_cache.ntiles) return;
        sfc_tile_t *tile = &g_glt_cache.tiles[t];
        for(int k = 0; k < 3; k++) { node->lo[k] = tile->lo[k]; node->hi[k] = tile->hi[k]; }
        node->hmax = tile->hmax;
        for(int tt = 0; tt < TILE_NUM_PTYPES; tt++) node->hmax_by_type[tt] = tile->hmax_by_type[tt];
    } else {
        tile_bvh_node_t *left = &g_glt_cache.bvh[node->left];
        tile_bvh_node_t *right = &g_glt_cache.bvh[node->right];
        for(int k = 0; k < 3; k++) {
            node->lo[k] = DMIN(left->lo[k], right->lo[k]);
            node->hi[k] = DMAX(left->hi[k], right->hi[k]);
        }
        node->hmax = DMAX(left->hmax, right->hmax);
        for(int tt = 0; tt < TILE_NUM_PTYPES; tt++)
            node->hmax_by_type[tt] = DMAX(left->hmax_by_type[tt], right->hmax_by_type[tt]);
    }
}

/* Find the tile containing pool position pp via binary search on tile->first.
 * Tiles are stored in ascending pool-position order; tile.first values are
 * monotonically non-decreasing.  Returns -1 on out-of-range. */
static inline int glt_tile_of_pool_pos_(int pp)
{
    int lo = 0, hi = g_glt_cache.ntiles - 1;
    if(pp < 0 || g_glt_cache.ntiles <= 0) return -1;
    if(pp < g_glt_cache.tiles[0].first) return -1;
    while(lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if(g_glt_cache.tiles[mid].first <= pp) lo = mid; else hi = mid - 1;
    }
    /* Sanity: pp must lie within [first, first+count). */
    sfc_tile_t *tile = &g_glt_cache.tiles[lo];
    if(pp >= tile->first && pp < tile->first + tile->count) return lo;
    return -1;
}

static void glt_cache_refit_from_particles(void)
{
    if(!g_glt_cache.valid || !g_glt_cache.tiles || !g_glt_cache.pool ||
       !g_glt_cache.bvh || !g_glt_cache.compact_xyzh || !g_glt_cache.pool_types) return;

    /* Narrow path: refresh only tiles touched by g_glt_dirty_list, then walk
     * BVH bottom-up updating only ancestor nodes of those tiles. */
    if(!g_glt_dirty_all && !g_glt_dirty_list.empty() && g_glt_cache.j_to_pool) {
        int n_dirty = (int)g_glt_dirty_list.size();
        int ntiles = g_glt_cache.ntiles;
        int bvh_nnodes = g_glt_cache.bvh_nnodes;
        int NumPart_b = g_glt_cache.NumPart_when_built;
        unsigned char *tile_dirty = (unsigned char *) calloc((size_t)(ntiles > 0 ? ntiles : 1), 1);
        unsigned char *node_dirty = (unsigned char *) calloc((size_t)(bvh_nnodes > 0 ? bvh_nnodes : 1), 1);

        /* Phase 1: refresh compact_xyzh + pool_types for each dirty j; mark its tile dirty. */
        for(int k = 0; k < n_dirty; k++) {
            int j = g_glt_dirty_list[k];
            if(j < 0 || j >= NumPart_b) continue;
            int pp = g_glt_cache.j_to_pool[j];
            if(pp < 0 || pp >= g_glt_cache.num_pool) continue;
            int t = glt_tile_of_pool_pos_(pp);
            if(t < 0) continue;
            tile_dirty[t] = 1;
        }

        /* Phase 2: re-scan each touched tile fully (cheaper than tracking which
         * member exactly; tile_size ~64, dirty_count typically ~few-hundred). */
        for(int t = 0; t < ntiles; t++) {
            if(tile_dirty[t]) glt_recompute_tile_(t);
        }

        /* Phase 3: BVH bottom-up single pass.  At each leaf, propagate
         * tile_dirty[tile] -> node_dirty[n].  At each internal node, OR its
         * children's flags; if dirty, recompute lo/hi/hmax/hmax_by_type from
         * children.  Children-first ordering is guaranteed by build_bvh_recursive. */
        for(int n = 0; n < bvh_nnodes; n++) {
            tile_bvh_node_t *node = &g_glt_cache.bvh[n];
            if(node->left < 0) {
                int t = -(node->left + 1);
                if(t >= 0 && t < ntiles && tile_dirty[t]) {
                    node_dirty[n] = 1;
                    glt_recompute_bvh_node_(n);
                }
            } else {
                int L = node->left, R = node->right;
                int dL = (L >= 0 && L < bvh_nnodes) ? node_dirty[L] : 0;
                int dR = (R >= 0 && R < bvh_nnodes) ? node_dirty[R] : 0;
                if(dL || dR) {
                    node_dirty[n] = 1;
                    glt_recompute_bvh_node_(n);
                }
            }
        }

        free(tile_dirty);
        free(node_dirty);
        g_glt_dirty_clear_();
        g_glt_cache.Ti_when_built = All.Ti_Current;
        g_glt_cache.needs_refit = 0;
        g_glt_cache_narrow_refits++;
        return;
    }

    /* Full path: scan every tile + every BVH node. */
    for(int t = 0; t < g_glt_cache.ntiles; t++) glt_recompute_tile_(t);
    for(int n = 0; n < g_glt_cache.bvh_nnodes; n++) glt_recompute_bvh_node_(n);
    g_glt_dirty_clear_();
    g_glt_cache.Ti_when_built = All.Ti_Current;
    g_glt_cache.needs_refit = 0;
    g_glt_cache_refits++;
}


/* ---- Utility: walk TopNodes to find which leaf a particle belongs to ---- */
static inline int ghost_toptree_leaf(peanokey key)
{
    int no = 0;
    peanokey mask = ((peanokey)7) << (3 * (BITS_PER_DIMENSION - 1));
    int shift = 3 * (BITS_PER_DIMENSION - 1);
    while(TopNodes[no].Daughter >= 0)
    {
        no = TopNodes[no].Daughter + (int)((key & mask) >> shift);
        mask >>= 3;
        shift -= 3;
    }
    return TopNodes[no].Leaf;
}


/*!
 * \brief Main ghost exchange routine. Call before neighbor operations.
 *
 * For each remote top-level leaf whose bounding region overlaps any local
 * particle's search sphere, imports all particles from that leaf.
 * Ghost particles are appended to P[]/CellP[] starting at NumPart.
 *
 * After all neighbor operations, call ghost_exchange_cleanup() to remove ghosts.
 *
 * safety_factor: multiplier on search_radius for the overlap criterion.
 *   1.0 = normal (previous-step hmax is accurate).
 *   >1.0 = inflate search radius to account for h-growth during density iteration
 *          (e.g. 2.0 on first timestep when densities are just guesses).
 */
static inline int ghost_type_passes(int ptype, unsigned int mask) { return (mask & (1u << (unsigned)ptype)) != 0u; }

/* Forward decls. */
/* Result of a single-backend ghost-exchange discovery attempt. The dispatcher
 * (ghost_exchange_impl) owns admission policy: a tile attempt that cannot fit
 * particle slots, or whose counts overflow the int MPI transport representation,
 * returns WITHOUT materialising ghosts (clean rollback), and the dispatcher
 * falls back to exact request-driven discovery. */
enum ghost_exchange_result {
    GHOST_EXCHANGE_COMPLETED = 0,
    GHOST_EXCHANGE_PARTICLE_CAPACITY_EXCEEDED,
    GHOST_EXCHANGE_COUNT_RANGE_EXCEEDED
};

static ghost_exchange_result ghost_exchange_request_driven_impl(const struct ghost_exchange_spec_t *spec);
static ghost_exchange_result ghost_exchange_tile_overlap_impl(const struct ghost_exchange_spec_t *spec);

/* Is this spec eligible for the walk-export routed producer (sender fine-tree
 * export + bounded receiver walk)?  Keyed on the SEARCH MODE and structural spec
 * fields only — never on a caller name, so every loop of a given class routes.
 *
 * ONEWAY: always eligible.  Its reach is the query's own radius, which the sender
 * knows exactly, so routing needs nothing from the supply side.  The traversal AND
 * export opener are both R_open = h_q (mode_b_local_walker.cc, the R_open branch
 * and the topleaf export re-test), and the accept ignores h_j entirely
 * (ghost_exchange_functions.h gx_pair_accept_wrap_and_test, ONEWAY branch).  Query h already
 * carries the spec safety factor, so a widened query cannot outgrow the opener.
 * KEEP THOSE THREE IN STEP: if ONEWAY accept ever gains an h_j term, or the opener
 * stops using h_q, this eligibility no longer holds and must be re-derived.
 *
 * SYMMETRIC: eligible only when the supply-side reach is bounded by the per-type
 * node band the sender opener walks against — otherwise a reach beyond the band
 * silently under-imports.  ONE structural condition:
 *   supply_band_dominated  the spec's reach is proven bounded by that band
 * A safety factor above 1 (TURB_DIFF_DYNAMIC) is NOT a disqualifier: both walks
 * scale the j-side reach they search with by the spec's safety factor, so their
 * reach equals the accept's at any safety and a widened query cannot outgrow the
 * opener.  Fails closed: anything unproven keeps the broadcast path it uses today.
 * Rank-uniform — search_mode and both fields are spec constants, identical on
 * every rank, so this never splits ranks across a collective. */
static inline int gx_walk_export_eligible(const struct ghost_exchange_spec_t *spec)
{
    if(!spec) return 0;
    if(spec->search_mode == NGB_SEARCH_ONEWAY) return 1;
    /* safety_factor is NOT a disqualifier: the walk-export sender opener and
     * receiver walk fold it into the j-side reach they search with, so their
     * reach equals the accept's for any safety. Only an unproven supply band
     * still forces broadcast. */
    return spec->search_mode == NGB_SEARCH_SYMMETRIC
           && spec->supply_band_dominated;
}

/* Announce, ONCE per caller per run, that a SYMMETRIC caller is on broadcast.
 * Once-only so a timing arm is never perturbed by per-call stdout; loud enough
 * that an unpromoted caller cannot hide in a log. */
static void gx_report_symm_broadcast(const struct ghost_exchange_spec_t *spec)
{
    enum { GX_SYMM_REPORT_MAX = 64 };   /* > the number of SYMMETRIC specs in the tree */
    static const char *seen[GX_SYMM_REPORT_MAX];
    static int n_seen = 0;
    static int table_full_reported = 0;
    if(ThisTask != 0 || !spec) return;
    const char *name = spec->caller_name ? spec->caller_name : "?";
    /* compare by CONTENT: callers pass distinct string objects (spec literals,
     * Spec::loop_name), so pointer identity would let one caller report twice. */
    for(int k = 0; k < n_seen; k++) { if(strcmp(seen[k], name) == 0) return; }
    if(n_seen >= GX_SYMM_REPORT_MAX) {
        /* Never degrade to per-call printing: that would flood a log and perturb
         * the very timing this report exists to keep honest. Say so once, then stop. */
        if(!table_full_reported) {
            table_full_reported = 1;
            printf("GHOST_SYMM_BCAST caller=<table full at %d> reason=further_callers_unreported\n",
                   GX_SYMM_REPORT_MAX);
            fflush(stdout);
        }
        return;
    }
    seen[n_seen++] = name;
    printf("GHOST_SYMM_BCAST caller=%s reason=%s\n", name,
           "band_unproven");   /* the only remaining disqualifier */
    fflush(stdout);
}

static void ghost_exchange_impl(const struct ghost_exchange_spec_t *spec)
{
    /* Tiny-N corridor counter: increments on API entry, before any
     * dispatch. Mode B paths in run_neighbor_loop must NOT enter this
     * function. Counts even single-rank early-out cases by design. See
     * declarations/lifecycle_counters.h. */
    g_ghost_import_counter++;

    /* Dispatch policy: explicit-query callers (runner-issued specs,
     * n_queries >= 0) use the request-driven path — tile-overlap cannot consume
     * an explicit query list (it scans ActiveParticleList filtered by
     * request_type_mask, so a spec with request_type_mask=0u + an explicit list
     * would import zero ghosts).  Every ONEWAY request also uses request-driven,
     * regardless of caller: routed ONEWAY discovery is a property of the search
     * mode, not of any specific loop.  SYMMETRIC requests use request-driven when
     * the spec proves supply-band domination (gx_walk_export_eligible); all other
     * SYMMETRIC callers stay on tile-overlap/broadcast.  Which producer then
     * supplies the matched set inside the request-driven path is a separate,
     * likewise structural decision (same predicate). */
    const int explicit_queries = (spec && spec->n_queries >= 0);
    const int want_request_driven = explicit_queries
                                    || (spec && spec->search_mode == NGB_SEARCH_ONEWAY)
                                    || gx_walk_export_eligible(spec);
    if(spec && spec->search_mode == NGB_SEARCH_SYMMETRIC && !gx_walk_export_eligible(spec))
        gx_report_symm_broadcast(spec);
    if(want_request_driven) {
        ghost_exchange_request_driven_impl(spec);
    } else {
        ghost_exchange_result result = ghost_exchange_tile_overlap_impl(spec);
        if(result != GHOST_EXCHANGE_COMPLETED) {
            /* Tile admission failed: it could not fit particle slots (or its counts exceeded
             * the int transport range) COLLECTIVELY — every rank returns the same
             * result (Allreduce in the tile impl), so all ranks take this branch
             * together. Drain any unrelated pending controlled-stop, then fall
             * back to exact request-driven discovery (changes discovery only, not
             * the downstream kernel). */
            const char *reason = (result == GHOST_EXCHANGE_COUNT_RANGE_EXCEEDED)
                                 ? "count_range" : "particle_capacity";
            if(ThisTask == 0) {
                printf("GHOST_ADMIT caller=%s attempted=tile selected=request_driven reason=%s\n",
                       spec->caller_name ? spec->caller_name : "?", reason);
                fflush(stdout);
            }
            gizmo_exit_bad_stop_if_requested("ghost_exchange:tile_fallback");
            ghost_exchange_request_driven_impl(spec);
        }
    }
}

/* Public entry for new-style callers that build their own spec literal at
 * the call site (mech_fb_v1 onward). The literal IS the single source of
 * truth for that loop's physics — to flip mode / supply_mask / query list,
 * edit the literal at the caller. Dispatch keys only on spec fields
 * (explicit query list, or search_mode == NGB_SEARCH_ONEWAY). */
extern "C" void ghost_exchange_run(const struct ghost_exchange_spec_t *spec)
{
    ghost_exchange_impl(spec);
}

/* Tile-build SSOT for the tile ghost-exchange pool: particles are chunked
 * GHOST_TILE_TARGET at a time in pool order, giving the pool + tiling + per-tile
 * supply hmax. Any second consumer of this tiling must use the same value or the
 * two disagree about which particles share a tile. */
static constexpr int GHOST_TILE_TARGET = 64;

/* Does particle i join the tile supply pool? Mass-positive + type in supply_mask. */
static inline int ghost_tile_pool_includes(int i, unsigned int supply_mask)
{
    if(P[i].Mass <= 0) return 0;
    if(!ghost_type_passes((int)P[i].Type, supply_mask)) return 0;
    return 1;
}

/* Effective ghost search radius this rank must SUPPLY for particle j under the
 * tile-build rule. Typed callers use the bare KernelRadius (matches the legacy
 * ngb search; avoids conflating AGS / DM / wind kernels into the
 * hydro/sink/feedback radius). The all-types pool fans in the widest enabled
 * per-type kernel so ghost bboxes cover any active kernel. */
static inline double ghost_tile_effective_radius(int j, unsigned int supply_mask)
{
    if(supply_mask != GHOST_TYPE_ALL) {
        if(!ghost_type_passes((int)P[j].Type, supply_mask)) return 0.0;
        return (double)P[j].KernelRadius;
    }
    double h = P[j].KernelRadius;
#ifdef DM_DISPERSION_LOOP_ACTIVE
    if(P[j].Type == 0 && j < N_gas) {
        if((double)CellP[j].KernelRadiusDM > h) h = CellP[j].KernelRadiusDM;
    }
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    if((double)P[j].AGS_KernelRadius > h) h = P[j].AGS_KernelRadius;
#endif
    return h;
}

/* Pure particle-slot fit predicate: do `required` total (local+ghost) particles
 * fit P[] and CellP[]? Policy (what to do on a miss) lives in the caller, NOT
 * here — keep this free of multi-space/budget logic.
 * Both capacities are tested because an imported ghost occupies P[j] and, when
 * the run has gas, CellP[j] at the same index: the received range is written to
 * CellP[NumPart..] whatever the ghost types are. The two capacities are equal by
 * construction (gizmo_set_gas_capacity_from_maxpart), so the gas test is
 * redundant today and states the requirement rather than assuming it. */
static inline int ghost_particle_slots_fit(long long required)
{
    if(required > (long long)All.MaxPart) {return 0;}
    if(All.TotN_gas > 0 && required > (long long)All.MaxPartGas) {return 0;}
    return 1;
}

/* SSOT for "what a send slot contains": one exported particle's P (+ gas CellP,
   zeroed for non-gas). Used by BOTH import pack loops and ghost_refresh_values()
   so the refresh cannot drift from import. src_P/src_CellP are the value source
   (production import + refresh pass P/CellP; the refresh harness passes a copy). */
static inline void gx_pack_send_slot(const struct particle_data *src_P,
                                     const struct gas_cell_data *src_CellP,
                                     int j,
                                     struct particle_data *dst_P,
                                     struct gas_cell_data *dst_CellP)
{
    *dst_P = src_P[j];
    if(src_P[j].Type == 0 && j < N_gas) *dst_CellP = src_CellP[j];
    else                                memset(dst_CellP, 0, sizeof(struct gas_cell_data));
}

/* SSOT for the forward particle+cell transport: the two typed Alltoallv calls
   (P always; CellP only when gas exists globally) with element-unit counts.
   Used by BOTH import impls (materialising ghosts at &P[NumPart]) and
   ghost_refresh_values() (overwriting existing ghost slots at
   &P[NumPart_before_ghost]). Verbose per-impl diagnostics stay at the call
   sites; only the transport is factored here. */
static void gx_forward_particle_exchange(const struct particle_data *send_P,
                                         const struct gas_cell_data *send_CellP,
                                         const int *send_count, const int *send_disp,
                                         struct particle_data *dst_P,
                                         struct gas_cell_data *dst_CellP,
                                         const int *recv_count, const int *recv_disp)
{
    gizmo_mpi_alltoallv_typed((void *)send_P, (int *)send_count, (int *)send_disp,
                              dst_P, (int *)recv_count, (int *)recv_disp,
                              sizeof(struct particle_data), MPI_COMM_WORLD);
    if(All.TotN_gas > 0) {
        gizmo_mpi_alltoallv_typed((void *)send_CellP, (int *)send_count, (int *)send_disp,
                                  dst_CellP, (int *)recv_count, (int *)recv_disp,
                                  sizeof(struct gas_cell_data), MPI_COMM_WORLD);
    }
}

static ghost_exchange_result ghost_exchange_tile_overlap_impl(const struct ghost_exchange_spec_t *spec)
{
    const double safety_factor = spec->safety_factor;
    const unsigned int request_mask = spec->request_type_mask;
    const unsigned int supply_mask  = spec->supply_type_mask;
    const int  search_mode = spec->search_mode;
    if(NTask <= 1) return GHOST_EXCHANGE_COMPLETED;
    double t_ghost_start = my_second();

    /* save current state for cleanup */
    NumPart_before_ghost = NumPart;
    N_gas_before_ghost = N_gas;
    NumGhostParticles = 0;

    int i, k, task;
    int tile_target = GHOST_TILE_TARGET;

    /* ================================================================
       Step 1: Build SFC tiles from local particles.
       Particles are already Peano-Hilbert sorted. Group into tiles of
       ~tile_target particles, computing bbox and hmax per tile.
       Uses malloc (not mymalloc) for tile metadata to avoid stack issues.
       ================================================================ */
    int local_ntiles = 0, num_pool = 0;

    /* Count pool particles. supply_mask gates which Types may join the pool;
     * removes tile-hmax pollution from non-supply types (e.g. DM init-time
     * KernelRadius poisoning hydro tile bboxes). */
    for(i = 0; i < NumPart; i++) {
        if(!ghost_tile_pool_includes(i, supply_mask)) continue;
        num_pool++;
    }

    /* Build pool index array */
    int *pool = (int *) malloc((num_pool > 0 ? num_pool : 1) * sizeof(int));
    int p = 0;
    for(i = 0; i < NumPart; i++) {
        if(!ghost_tile_pool_includes(i, supply_mask)) continue;
        pool[p++] = i;
    }

    local_ntiles = (num_pool + tile_target - 1) / tile_target;
    if(local_ntiles < 1) local_ntiles = 1;

    /* Compact tile metadata for exchange. Two parallel bbox/hmax sets:
     *   (lo, hi, hmax, count)         — over ALL particles in tile
     *   (active_lo, active_hi, active_hmax, active_count) — over ACTIVE only
     *
     * The all-particle set governs what this rank can SUPPLY as ghosts (j may
     * be inactive on its home rank but still a neighbor of an active i on a
     * peer rank — WAKEUP-style semantics demand this).
     *
     * The active-only set governs what this rank actually NEEDS: only tiles
     * containing at least one active particle drive remote-tile imports.
     *
     * Old code (pre-tile, tree-based ngb_treefind_variable_threads) iterated
     * FirstActiveParticle and built per-rank exports off active i's search
     * radius — the new tile path silently dropped that gating and was
     * shipping ~all of the global pool on tiny-N steps. */
    struct tile_meta_t {
        double lo[3], hi[3], hmax;
        int count;
        double active_lo[3], active_hi[3], active_hmax;
        int active_count;
    };

    tile_meta_t *local_meta = (tile_meta_t *) malloc(local_ntiles * sizeof(tile_meta_t));
    int *tile_first = (int *) malloc(local_ntiles * sizeof(int)); /* index into pool[] */

    /* Per-particle active flag — ActiveParticleList lookup at O(1). The active
     * stats below restrict the tile's "what do I need from peers?" criterion
     * to particles that will actually walk neighbors this step. Allocate as
     * char rather than bool to keep the address-stable contract. */
    char *is_active = (char *) calloc(NumPart > 0 ? NumPart : 1, sizeof(char));
    for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
        int i_act = ActiveParticleList[kk];
        if(i_act < 0 || i_act >= NumPart) continue;
        /* request_mask gates which types' actives drive imports. */
        if(!ghost_type_passes((int)P[i_act].Type, request_mask)) continue;
        is_active[i_act] = 1;
    }

    for(int t = 0; t < local_ntiles; t++)
    {
        int start = t * tile_target;
        int count = tile_target;
        if(start + count > num_pool) count = num_pool - start;
        if(count < 0) count = 0;
        tile_first[t] = start;
        local_meta[t].count = count;
        local_meta[t].hmax = 0;
        local_meta[t].active_count = 0;
        local_meta[t].active_hmax = 0;
        for(k = 0; k < 3; k++) {
            local_meta[t].active_lo[k] = 0;  /* will be overwritten on first active */
            local_meta[t].active_hi[k] = 0;
        }
        if(count <= 0) {
            /* Rank owns no supply-pool particles for this mask -> empty sentinel
             * tile. Zero bbox/hmax, NO pool[] access (pool may be a 1-elt stub).
             * active_count=0 + hmax=0 means it neither needs nor supplies ghosts
             * downstream. */
            for(k = 0; k < 3; k++) { local_meta[t].lo[k] = 0; local_meta[t].hi[k] = 0; }
            continue;
        }

        int j0 = pool[start];
        for(k = 0; k < 3; k++) local_meta[t].lo[k] = local_meta[t].hi[k] = P[j0].Pos[k];
        {double h0 = ghost_tile_effective_radius(j0, supply_mask); if(h0 > local_meta[t].hmax) local_meta[t].hmax = h0;}
        if(is_active[j0]) {
            local_meta[t].active_count = 1;
            for(k = 0; k < 3; k++) local_meta[t].active_lo[k] = local_meta[t].active_hi[k] = P[j0].Pos[k];
            double h0 = ghost_tile_effective_radius(j0, supply_mask);
            local_meta[t].active_hmax = h0;
        }

        for(int s = 1; s < count; s++) {
            int j = pool[start + s];
            for(k = 0; k < 3; k++) {
                if(P[j].Pos[k] < local_meta[t].lo[k]) local_meta[t].lo[k] = P[j].Pos[k];
                if(P[j].Pos[k] > local_meta[t].hi[k]) local_meta[t].hi[k] = P[j].Pos[k];
            }
            double hj = ghost_tile_effective_radius(j, supply_mask);
            if(hj > local_meta[t].hmax) local_meta[t].hmax = hj;
            if(is_active[j]) {
                if(local_meta[t].active_count == 0) {
                    for(k = 0; k < 3; k++) local_meta[t].active_lo[k] = local_meta[t].active_hi[k] = P[j].Pos[k];
                    local_meta[t].active_hmax = hj;
                } else {
                    for(k = 0; k < 3; k++) {
                        if(P[j].Pos[k] < local_meta[t].active_lo[k]) local_meta[t].active_lo[k] = P[j].Pos[k];
                        if(P[j].Pos[k] > local_meta[t].active_hi[k]) local_meta[t].active_hi[k] = P[j].Pos[k];
                    }
                    if(hj > local_meta[t].active_hmax) local_meta[t].active_hmax = hj;
                }
                local_meta[t].active_count++;
            }
        }
    }
    free(is_active);


    /* ================================================================
       Step 2: Gather tile metadata from all ranks.
       Each rank sends its tile count and metadata to all ranks.
       ================================================================ */
    int *all_ntiles = (int *) malloc(NTask * sizeof(int));
    MPI_Allgather(&local_ntiles, 1, MPI_INT, all_ntiles, 1, MPI_INT, MPI_COMM_WORLD);

    int *tile_disp = (int *) malloc(NTask * sizeof(int));
    int total_tiles = 0;
    for(task = 0; task < NTask; task++) {
        tile_disp[task] = total_tiles;
        total_tiles += all_ntiles[task];
    }

    /* Exchange tile metadata using MPI_Allgatherv with MPI_BYTE */
    tile_meta_t *all_meta = (tile_meta_t *) malloc(total_tiles * sizeof(tile_meta_t));
    int *meta_counts = (int *) malloc(NTask * sizeof(int));
    int *meta_disps = (int *) malloc(NTask * sizeof(int));
    for(task = 0; task < NTask; task++) {
        meta_counts[task] = all_ntiles[task] * sizeof(tile_meta_t);
        meta_disps[task] = tile_disp[task] * sizeof(tile_meta_t);
    }
    MPI_Allgatherv(local_meta, local_ntiles * sizeof(tile_meta_t), MPI_BYTE,
                   all_meta, meta_counts, meta_disps, MPI_BYTE, MPI_COMM_WORLD);
    free(meta_counts); free(meta_disps);

    if(ThisTask == 0) {
        double hmax_min = 1e30, hmax_max = 0, hmax_sum = 0;
        for(int t = 0; t < local_ntiles; t++) {
            if(local_meta[t].hmax < hmax_min) hmax_min = local_meta[t].hmax;
            if(local_meta[t].hmax > hmax_max) hmax_max = local_meta[t].hmax;
            hmax_sum += local_meta[t].hmax;
        }
        PRINT_STATUS("Ghost exchange (tile-based): %d local tiles, %d total across %d ranks, tile hmax=[%.4g, %.4g] avg=%.4g",
                     local_ntiles, total_tiles, NTask, hmax_min, hmax_max, hmax_sum / local_ntiles);
    }


    /* Diagnostic: number this ghost_exchange call to track progress in multi-call steps */
    static int ghost_call_seq = 0;
    ghost_call_seq++;
    int this_call = ghost_call_seq;

    /* ================================================================
       Step 3: Per-task tile overlap check.
       For each remote task, check which of its tiles overlap with any
       of our tiles AND which of our tiles overlap with any of its tiles.
       No global Allreduce — each rank independently computes per-task
       recv and send lists using the symmetric overlap criterion.
       ================================================================ */

    /* Per-tile need flags: need_from[total_tiles] = do WE need this remote tile?
       send_to[local_ntiles * NTask] = does task t need our tile lt?
     *
     * Asymmetric criterion:
     *   need_from[rt]            = our active tiles' bbox vs rt's all-bbox
     *   send_to[lt][task]        = task's active tiles' bbox vs our lt's all-bbox
     * Active-aware on each rank's REQUEST side, all-particle on the SUPPLY
     * side (a remote inactive j may still be a neighbor of an active i — see
     * WAKEUP). Tile pairs where the request side has zero actives early-exit
     * — that's where the tiny-N wins come from. */
    int *need_from = (int *) calloc(total_tiles, sizeof(int));
    int *send_to = (int *) calloc(local_ntiles * NTask, sizeof(int));
    int my_tile_start = tile_disp[ThisTask];

    /* Tile-pair overlap is a point against the Minkowski sum of the two boxes.
     * The box wrap belongs to the canonical macro family (the former per-axis
     * form here could not represent a shearing box's x-y coupling), so it lives
     * in gx_boxpair_overlap_wrap_and_test; the exact clamped-gap acceptance
     * this pass relies on is unchanged. */

    /* Pass 1: need_from[rt] — driven by OUR active tiles. Outer loop over
     * local tiles with active_count > 0 only (tiny-N: just a handful). */
    for(int lt_idx = 0; lt_idx < local_ntiles; lt_idx++)
    {
        int lt = my_tile_start + lt_idx;
        tile_meta_t *lm = &all_meta[lt];
        if(lm->active_count == 0) continue;
        double c_lo[3], c_hw[3];
        for(k = 0; k < 3; k++) {
            c_lo[k] = 0.5 * (lm->active_lo[k] + lm->active_hi[k]);
            c_hw[k] = 0.5 * (lm->active_hi[k] - lm->active_lo[k]);
        }
        for(task = 0; task < NTask; task++)
        {
            if(task == ThisTask) continue;
            int t_start = tile_disp[task];
            int t_count = all_ntiles[task];
            for(int rt_idx = 0; rt_idx < t_count; rt_idx++)
            {
                int rt = t_start + rt_idx;
                if(need_from[rt]) continue;            /* already flagged by another lt */
                tile_meta_t *rm = &all_meta[rt];
                /* Search radius depends on caller mode:
                 *   ONEWAY (density): r_ij < h_i — only the LOCAL active's h
                 *     matters. Importing a remote tile because of its own
                 *     particles' large h_j is incorrect for density.
                 *   SYMMETRIC (gradients/sinks/etc.): r_ij < max(h_i, h_j) —
                 *     remote h_j matters because j's kernel may reach back
                 *     to i. */
                double search_r = (search_mode == NGB_SEARCH_ONEWAY)
                                    ? lm->active_hmax * safety_factor
                                    : DMAX(lm->active_hmax, rm->hmax) * safety_factor;
                if(search_r <= 0) continue;
                double search_r2 = search_r * search_r;
                double dc[3], hw_sum[3];
                for(k = 0; k < 3; k++) {
                    const double c_r  = 0.5 * (rm->lo[k] + rm->hi[k]);
                    const double hw_r = 0.5 * (rm->hi[k] - rm->lo[k]);
                    dc[k]     = c_lo[k] - c_r;
                    hw_sum[k] = c_hw[k] + hw_r;
                }
                if(gx_boxpair_overlap_wrap_and_test(dc[0], dc[1], dc[2],
                                                    hw_sum[0], hw_sum[1], hw_sum[2],
                                                    search_r, search_r2)) need_from[rt] = 1;
            }
        }
    }

    /* Pass 2: derive send_to[lt][task] from peers' need_from directly.
     *
     * Earlier versions recomputed an "active-on-the-other-side" overlap test
     * here. That's algebraically the same predicate as peer's pass 1 — but
     * the two ranks running floating-point min-distance arithmetic on the
     * same all_meta blob can disagree on a tile pair that lies right at the
     * search-radius threshold. When they do, A says send_count[B]=N and B
     * says recv_count[A]=N±1, and the downstream Alltoallv truncates with
     * MPI_ERR_TRUNCATE. Killing the recompute entirely makes the two sides
     * bit-identical by construction.
     *
     * Cost of the Allgather: total_tiles*NTask ints (~1.5 MB at 2 ranks ×
     * 200k tiles, scales as O(NTask^2) — switch to Alltoall of per-rank
     * slices if NTask grows past ~100). The pass-1 cost is unchanged: outer
     * loop active-gated, ~handful of tiles on tiny-N. */
    int *all_need_from = (int *) malloc((size_t)NTask * total_tiles * sizeof(int));
    MPI_Allgather(need_from, total_tiles, MPI_INT,
                  all_need_from, total_tiles, MPI_INT, MPI_COMM_WORLD);
    for(task = 0; task < NTask; task++)
    {
        if(task == ThisTask) continue;
        const int *peer_need = all_need_from + (size_t)task * total_tiles;
        for(int lt_idx = 0; lt_idx < local_ntiles; lt_idx++) {
            send_to[lt_idx * NTask + task] = peer_need[my_tile_start + lt_idx];
        }
    }
    free(all_need_from);

    /* ================================================================
       Step 4: Compute per-task send/recv counts from overlap results.
       ================================================================ */
    int *recv_count = (int *) mymalloc("ghost_rc", NTask * sizeof(int));
    int *send_count = (int *) mymalloc("ghost_sc", NTask * sizeof(int));
    memset(recv_count, 0, NTask * sizeof(int));
    memset(send_count, 0, NTask * sizeof(int));

    for(task = 0; task < NTask; task++) {
        if(task == ThisTask) continue;
        /* Recv: remote tiles we need from this task */
        for(int rt = tile_disp[task]; rt < tile_disp[task] + all_ntiles[task]; rt++) {
            if(need_from[rt]) recv_count[task] += all_meta[rt].count;
        }
        /* Send: our tiles this task needs */
        for(int lt = 0; lt < local_ntiles; lt++) {
            if(send_to[lt * NTask + task]) send_count[task] += local_meta[lt].count;
        }
    }

    /* Compute totals + displacements with CHECKED int64 accumulation. The MPI
     * pack consumes int counts/displacements; a value exceeding INT_MAX cannot
     * be represented and must NOT silently wrap into a false "fits" decision.
     * Such a representation overflow is reported distinctly from a particle-slot
     * overflow (the collective slot admission below). Per-peer counts are bounded by one
     * rank's pool (<= INT_MAX); the prefix sums + totals are the overflow risk. */
    int *recv_disp = (int *) mymalloc("ghost_rd", NTask * sizeof(int));
    int *send_disp = (int *) mymalloc("ghost_sd", NTask * sizeof(int));
    long long total_recv_ll = 0, total_send_ll = 0;
    int count_range_ok = 1;
    {
        long long rdisp = 0, sdisp = 0;
        for(task = 0; task < NTask; task++) {
            if(rdisp <= INT_MAX) recv_disp[task] = (int)rdisp; else { recv_disp[task] = 0; count_range_ok = 0; }
            if(sdisp <= INT_MAX) send_disp[task] = (int)sdisp; else { send_disp[task] = 0; count_range_ok = 0; }
            rdisp += recv_count[task];
            sdisp += send_count[task];
        }
        total_recv_ll = rdisp; total_send_ll = sdisp;
        if(total_recv_ll > INT_MAX || total_send_ll > INT_MAX) count_range_ok = 0;
    }
    int total_recv = count_range_ok ? (int)total_recv_ll : 0;
    int total_send = count_range_ok ? (int)total_send_ll : 0;

    int tiles_needed = 0, tiles_sent = 0;
    for(int rt = 0; rt < total_tiles; rt++) tiles_needed += need_from[rt];
    for(int lt = 0; lt < local_ntiles; lt++) {
        for(task = 0; task < NTask; task++) { if(send_to[lt * NTask + task]) { tiles_sent++; break; } }
    }


    /* Frees the pre-materialisation tile scratch ONLY (mymalloc LIFO, then
     * malloc). ONE free list, shared by the Stage-0A fallback bail and the
     * normal-completion cleanup; captures only scratch allocated up to here (the
     * later packing allocs are freed explicitly by the normal path). It does NOT
     * touch NumGhostParticles: on normal completion that field already holds the
     * materialised ghost count and MUST survive (ghost-writeback / cleanup /
     * cleanup reads it); the fallback bail resets it explicitly. */
    auto tile_preflight_cleanup = [&]() {
        myfree(send_disp); myfree(recv_disp);
        myfree(send_count); myfree(recv_count);
        free(send_to); free(need_from);
        free(all_meta); free(tile_disp); free(all_ntiles);
        free(tile_first); free(local_meta); free(pool);
    };

    /* === Admission: collective particle-slot fit ===
     * Decide tile-vs-fallback BEFORE materialising any ghosts (NumPart unchanged
     * to here). Two DISTINCT failure modes — a representation overflow must never
     * masquerade as a capacity decision:
     *   2 = COUNT_RANGE — int MPI count/displacement representation overflowed
     *   1 = CAPACITY    — ghosts would not fit P[]/CellP[] (All.MaxPart)
     * Either, on ANY rank, makes ALL ranks abandon tile (clean rollback) and
     * fall back to exact request-driven discovery in the dispatcher. The Allreduce
     * is a continuing-branch decision, distinct from the terminal controlled-stop
     * poll (which still drains unrelated stops on the fit path below). */
    int local_fail = (!count_range_ok) ? 2
                   : (!ghost_particle_slots_fit((long long)NumPart + total_recv_ll)) ? 1 : 0;
    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(global_fail != 0) {
        tile_preflight_cleanup();
        NumGhostParticles = 0;   /* rollback: no ghosts materialised this call */
        return (global_fail == 2) ? GHOST_EXCHANGE_COUNT_RANGE_EXCEEDED
                                  : GHOST_EXCHANGE_PARTICLE_CAPACITY_EXCEEDED;
    }
    /* Tile fits on all ranks. Drain any UNRELATED pending controlled-stop
     * (preserves the original all-rank poll) before materialising ghosts. */
    gizmo_exit_bad_stop_if_requested("ghost_exchange:capacity");

    /* ================================================================
       Step 5: Pack particles from tiles needed by each task.
       ================================================================ */
    struct particle_data *send_P = (struct particle_data *) mymalloc("ghost_sP",
        (total_send > 0 ? total_send : 1) * sizeof(struct particle_data));
    struct gas_cell_data *send_CellP = (struct gas_cell_data *) mymalloc("ghost_sC",
        (total_send > 0 ? total_send : 1) * sizeof(struct gas_cell_data));
    /* Record home index of each sent particle for ghost writeback */
    int *send_home_idx = (int *) malloc((total_send > 0 ? total_send : 1) * sizeof(int));

    int *task_offset = (int *) mymalloc("ghost_toff", NTask * sizeof(int));
    memcpy(task_offset, send_disp, NTask * sizeof(int));

    for(task = 0; task < NTask; task++)
    {
        if(send_count[task] <= 0) continue;
        for(int t = 0; t < local_ntiles; t++)
        {
            if(!send_to[t * NTask + task]) continue;
            for(int s = 0; s < local_meta[t].count; s++)
            {
                if(task_offset[task] >= send_disp[task] + send_count[task]) break;
                int j = pool[tile_first[t] + s];
                int off = task_offset[task]++;
                gx_pack_send_slot(P, CellP, j, &send_P[off], &send_CellP[off]);
                send_home_idx[off] = j; /* record home index for writeback + refresh provenance */
            }
        }
    }

    /* ================================================================
       Step 6: Exchange via MPI_Alltoallv.
       ================================================================ */
    /* Per-particle exchange: send_count/recv_count/send_disp/recv_disp are
     * already in element units. gizmo_mpi_alltoallv_typed builds a contiguous
     * MPI_Datatype per call so element-count int*'s drive the wire — dodging
     * the 2.1 GB per-peer int-overflow that bites fire_m11i at >~6M parts/rank. */
    gizmo_mpi_alltoallv_typed(send_P, send_count, send_disp,
                              &P[NumPart], recv_count, recv_disp,
                              sizeof(struct particle_data), MPI_COMM_WORLD);

    /* CellP exchange: only meaningful when the simulation has any gas
       particles globally. With TotN_gas==0 (N-body / DM-only runs), CellP
       is allocated to size 0, so writing to &CellP[NumPart] would dereference
       an out-of-bounds pointer. Skip the CellP alltoallv in that case —
       no gas ghosts can exist if no gas exists anywhere. */
    if(All.TotN_gas > 0) {
        gizmo_mpi_alltoallv_typed(send_CellP, send_count, send_disp,
                                  &CellP[NumPart], recv_count, recv_disp,
                                  sizeof(struct gas_cell_data), MPI_COMM_WORLD);
    }

    /* Update counts */
    NumGhostParticles = total_recv;
    NumPart += total_recv;

    /* Multi-rank correctness: ghost slots [NumPart_before_ghost, NumPart) just
     * received fresh particle_data from remote ranks via MPI_Alltoallv. Their
     * KernelRadius values overwrote whatever was in those local P[] slots
     * (uninitialized or stale from prior step's ghost import). Any subsequent
     * cached gpu_ngb_list_build reading compact_xyzh[j*4+3] for h_j on those
     * slots would see stale values without this dirty-mark, silently missing
     * symmetric neighbor pairs where r<h_ghost.  Single-rank runs hit this
     * branch only when NumGhostParticles>0 (rare for true 1-rank), so this
     * fix is functionally a multi-rank correctness guarantee. */
    if(NumGhostParticles > 0) {
        gpu_compact_xyzh_mark_h_dirty_range(NumPart_before_ghost, NumPart);
    }
    /* SIDX lifecycle notify: fires on every rank for every exchange completion,
     * INCLUDING the count==0 / no-receive path. count==0 invalidates any
     * cached ghost segment from a prior import (no stale-ghost survival). */
    gpu_sidx_notify_ghost_imported(NumPart_before_ghost, NumGhostParticles);

    /* ================================================================
       Step 7: Build ghost provenance map for writeback.
       Exchange home indices so each ghost knows its home rank + index.
       ================================================================ */
    {
        /* Exchange home indices: send_home_idx[total_send] → recv_home_idx[total_recv] */
        int *recv_home_idx = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
        gizmo_mpi_alltoallv_typed(send_home_idx, send_count, send_disp,
                                  recv_home_idx, recv_count, recv_disp,
                                  sizeof(int), MPI_COMM_WORLD);

        /* Build per-ghost provenance: home_rank and home_index */
        ghost_home_rank_map = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
        ghost_home_index_map = recv_home_idx; /* take ownership — freed in cleanup */
        for(task = 0; task < NTask; task++) {
            for(int g = 0; g < recv_count[task]; g++) {
                ghost_home_rank_map[recv_disp[task] + g] = task;
            }
        }

        /* Preserve comm maps for reverse Alltoallv (malloc copies of mymalloc'd arrays) */
        ghost_wb_recv_count = (int *) malloc(NTask * sizeof(int));
        ghost_wb_recv_disp  = (int *) malloc(NTask * sizeof(int));
        ghost_wb_send_count = (int *) malloc(NTask * sizeof(int));
        ghost_wb_send_disp  = (int *) malloc(NTask * sizeof(int));
        memcpy(ghost_wb_recv_count, recv_count, NTask * sizeof(int));
        memcpy(ghost_wb_recv_disp,  recv_disp,  NTask * sizeof(int));
        memcpy(ghost_wb_send_count, send_count, NTask * sizeof(int));
        memcpy(ghost_wb_send_disp,  send_disp,  NTask * sizeof(int));

        /* Preserve send-side provenance for ghost_refresh_values() (take
           ownership of send_home_idx; the free() below then no-ops on NULL). */
        if(ghost_send_home_idx) free(ghost_send_home_idx);
        ghost_send_home_idx   = send_home_idx;
        ghost_send_home_count = total_send;
        send_home_idx         = NULL;
        g_ghost_provenance_epoch++;
    }

    double t_ghost_end = my_second();
    double t_ghost_total = timediff(t_ghost_start, t_ghost_end);

    if(ThisTask == 0) {
        PRINT_STATUS("Ghost exchange: %d local + %d ghost = %d total (recv %d tiles, sent %d/%d) [%.4f s]",
                     NumPart_before_ghost, NumGhostParticles, NumPart,
                     tiles_needed, tiles_sent, local_ntiles, t_ghost_total);
    }
    /* Report a rank that is approaching the largest capacity this run may ever reach.  The test is
     * against that ceiling and not against the current capacity: the current one is raised whenever
     * an import does not fit, so running close to it is the intended state and saying so every time
     * would be noise on every rank of every exchange.  Approaching the ceiling is the condition that
     * actually ends a run, and it is reported once, because it is a property of the run rather than
     * of the exchange that happened to notice it. */
    if(All.MaxPartExpandable > 0 && NumPart > 0.8 * All.MaxPartExpandable) {
        static int reported_near_ceiling = 0;
        if(!reported_near_ceiling) {
            reported_near_ceiling = 1;
            PRINT_WARNING("Ghost exchange: task %d holds %d particles, %.0f%% of the %d it can ever hold. "
                          "Local particles plus imported ghosts are approaching the ceiling fixed at startup; "
                          "more ranks/nodes, or a smaller ghost-import demand, would relieve it.",
                          ThisTask, NumPart, 100.0 * (double)NumPart / (double)All.MaxPartExpandable,
                          All.MaxPartExpandable);
        }
    }

    /* Cleanup: later packing allocs (mymalloc LIFO + malloc), then the shared
     * preflight cleanup frees the discovery scratch (same free list the Stage-0A
     * fallback bail uses). */
    myfree(task_offset);
    myfree(send_CellP); myfree(send_P);
    free(send_home_idx);
    tile_preflight_cleanup();
    return GHOST_EXCHANGE_COMPLETED;
}


/* ============================================================================
 * Request-driven ghost exchange.
 *
 * Replaces the tile-overlap candidate generator + whole-tile MPI payload of
 * ghost_exchange_tile_overlap_impl with:
 *   1. Each rank builds a list of compact query records {pos[3], h, _pad}
 *      for each LOCAL active particle matching spec->request_type_mask.
 *   2. Per-rank query counts via Allgather; queries themselves via Allgatherv.
 *   3. For each remote rank's queries, walk the LOCAL pool and apply the
 *      EXACT predicate per particle (r_ij < h_i for ONEWAY,
 *      r_ij < max(h_i, h_j) for SYMMETRIC). Mark matched local pool indices
 *      in a bitmask per peer rank to dedupe (same j matched by N queries
 *      ships once).
 *   4. Per-peer match list is the basis for Alltoallv pack.
 *   5. Alltoallv particle_data + gas_cell_data + home_idx (existing typed
 *      Alltoallv used for the legacy path).
 *   6. Install as ghosts (NumPart += total_recv, mark dirty for compact_xyzh,
 *      build ghost_home_*_map, save ghost_wb_* arrays).
 *
 * Killshot diagnostic on Vista (job 694703) showed the legacy path imports
 * with 99.97-100% waste — millions of particles imported per call, of which
 * <1000 are actual neighbors under the criterion that imported them. This
 * path's match step performs the EXACT predicate before pack, so per-call
 * waste is zero by construction (modulo a small over-supply factor from
 * ghosts that pass for one query but turn out unused after kernel).
 *
 * For now the local-pool walk is a flat O(N_local_tiles * N_remote_queries)
 * scan with bbox-vs-sphere prune at the tile level; per-particle accept at
 * the leaf. A BVH speedup is straightforward later (the all-particle BVH
 * already used by the GPU neighbor list can be reused) but not a
 * blocker here — even the flat scan is dominated by per-particle work for the
 * tiny-N case that motivates this restructure.
 * ============================================================================ */
struct gx_query_t {
    double pos[3];
    double h;
    int    type;       /* for caller diagnostics; predicate uses h_i directly */
    int    _pad;
};

/* gx_export_envelope_t (the sender->supply wire record) is declared in
 * mesh/neighbor_list.h: the device receiver traversal compiles in another
 * translation unit and consumes the same records. */

/* Host-only BVH bbox-vs-sphere overlap test. Mirrors bbox_overlaps_sphere_gpu
 * (mesh/sfc_tiles_functions.h). For each axis, take the periodic-shortened
 * gap between sphere center and bbox; if any gap exceeds search_r, prune.
 * Otherwise sum-of-squares vs search_r2 for the final accept. */
static inline int gx_bbox_overlaps_sphere(const double bbox_lo[3], const double bbox_hi[3],
                                          const double pos[3], double search_r, double search_r2)
{
    (void)search_r2;   /* the shared predicate squares the radius itself */
    /* The box wrap is the canonical macro family's job, and it needs a centre
     * separation rather than a lo/hi interval, so convert here.  Rounding the
     * half-width UP (the larger of the two sides) keeps the test conservative
     * under FP, which is what the direct point-to-interval form used to buy;
     * over-opening costs a leaf test, under-opening drops a real neighbour. */
    double c[3], hw[3];
    for(int k = 0; k < 3; k++) {
        c[k]  = 0.5 * (bbox_lo[k] + bbox_hi[k]);
        double up = bbox_hi[k] - c[k], dn = c[k] - bbox_lo[k];
        hw[k] = (up > dn) ? up : dn;
    }
    return gx_extended_overlap_wrap_and_test(c[0] - pos[0], c[1] - pos[1], c[2] - pos[2],
                                             hw[0], hw[1], hw[2], search_r);
}

/* Host-only BVH walk for the request-driven path. Mirrors
 * search_neighbors_sfc_gpu's iterative stack-based walk. For each query,
 * traverses the local BVH, accepts at leaves with per-particle r_ij vs the
 * caller-specified predicate (ONEWAY or SYMMETRIC), marks matched pool
 * indices in match_bitmask. */
/* Compute hmax_eff over the supply types only. Avoids the scalar-hmax
 * contamination where a particle of a type the caller didn't ask for poisons
 * the opener. Inlined; on a hot path. */
static inline double gx_node_hmax_supply(const tile_bvh_node_t *node, unsigned int supply_mask)
{
    double m = 0;
    for(int t = 0; t < TILE_NUM_PTYPES; t++) {
        if((supply_mask & (1u << t)) == 0u) continue;
        if(node->hmax_by_type[t] > m) m = node->hmax_by_type[t];
    }
    return m;
}

static void gx_walk_local_bvh(const float *compact_xyzh,
                              const sfc_tile_t *tiles, int ntiles,
                              const int *pool, int num_pool,
                              const int *pool_types, /* [num_pool] — P[].Type per pool slot, for leaf supply-mask filter */
                              unsigned int supply_mask,
                              const tile_bvh_node_t *bvh, int bvh_root,
                              const double pos_q[3], double h_q, int search_mode,
                              char *match_bitmask /* size num_pool */,
                              long *n_exact_hits /* optional (NULL in production): count EXACT matches
                                                  * this call would produce, computing r2 even for
                                                  * already-set slots so the per-query count is
                                                  * unbiased by the dedup skip (over-route diagnostic) */)
{
    (void)ntiles;
    int stack[TILE_BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = bvh_root;
    while(sp > 0) {
        int node_idx = stack[--sp];
        const tile_bvh_node_t *node = &bvh[node_idx];
        /* OPENER criterion. ONEWAY: r_ij<h_q so search_r is just h_q (subtree
         * h's irrelevant). SYMMETRIC: r_ij<max(h_q,h_j); search_r = max(h_q,
         * subtree-max-h-of-supply-types). Per-type hmax filter avoids node
         * hmax being dominated by a type the caller didn't ask for. */
        double node_hmax_eff = gx_node_hmax_supply(node, supply_mask);
        double search_r = (search_mode == NGB_SEARCH_ONEWAY)
                            ? h_q
                            : ((h_q > node_hmax_eff) ? h_q : node_hmax_eff);
        if(search_r <= 0) continue;
        double search_r2 = search_r * search_r;
        if(!gx_bbox_overlaps_sphere(node->lo, node->hi, pos_q, search_r, search_r2)) continue;
        if(node->left < 0) {
            /* Leaf: per-particle accept against EXACT predicate. */
            int tile_idx = -(node->left + 1);
            const sfc_tile_t *tile = &tiles[tile_idx];
            for(int s = 0; s < tile->count; s++) {
                int pool_pos = tile->first + s;
                if(pool_pos < 0 || pool_pos >= num_pool) continue;
                int already = match_bitmask[pool_pos];
                if(already && !n_exact_hits) continue;   /* already matched: skip unless the caller wants unbiased hit counts */
                /* Supply-mask filter at leaf: skip particles of a type the
                 * caller didn't ask for (no-op when tree was built with the
                 * same mask, but required when tree is shared across callers). */
                if(pool_types) {
                    int pt = pool_types[pool_pos];
                    if(pt < 0 || pt >= TILE_NUM_PTYPES) continue;
                    if((supply_mask & (1u << (unsigned)pt)) == 0u) continue;
                }
                /* Geometric accept is the shared SSOT predicate; supply-mask
                 * filter + dedup bookkeeping stay caller-side (above/here).
                 * BOTH position AND reach are DOUBLE: P[j].Pos (float absolute
                 * coordinates are invalid for GIZMO's dynamic range) and the double
                 * gx_policy_scaled_h reach (the node/tile hmax the opener prunes with
                 * is built from this same double reach, so the opener conservatively
                 * dominates the leaf — a float leaf h would let the opener under-prune
                 * boundary pairs).  Recompute here for the host path; the production
                 * GPU path needs a double h cache to avoid the per-candidate recompute.
                 * The pool walked here is always g_glt_cache's, so its build-time
                 * policy/scale/safety reproduce the cached double reach exactly. */
                int j_neighbor = pool[pool_pos];
                /* The pool is mass-filtered when it is built, but this pool is
                 * cached across calls and an entry outlives a Mass->0 marking
                 * until rearrange_particle_sequence() compacts it away.  Zero mass
                 * is the code's marker for an eliminated element, which must never
                 * be offered as a ghost source, so re-check it here instead of
                 * trusting build time.  Ahead of the reach recompute so a dead
                 * slot costs one compare. */
                if(P[j_neighbor].Mass <= 0) continue;
                double hj_dbl = gx_policy_scaled_h(j_neighbor, g_glt_cache.radius_policy_when_built,
                                                   g_glt_cache.j_radius_scale_when_built,
                                                   g_glt_cache.safety_factor_when_built);
                if(gx_pair_accept_wrap_and_test(pos_q[0] - (double)P[j_neighbor].Pos[0],
                                                pos_q[1] - (double)P[j_neighbor].Pos[1],
                                                pos_q[2] - (double)P[j_neighbor].Pos[2],
                                                h_q, hj_dbl, search_mode)) {
                    if(n_exact_hits) (*n_exact_hits)++;
                    if(!already) match_bitmask[pool_pos] = 1;
                }
            }
        } else {
            if(sp + 2 > TILE_BVH_STACK_SIZE) break;  /* defensive — shouldn't happen */
            stack[sp++] = node->left;
            stack[sp++] = node->right;
        }
    }
}



/* Broadcast discovery walk. Walks every remote rank's
 * queries (from the Allgatherv'd all_queries) against the pre-built local supply
 * snapshot and returns the per-peer match bitmask matched[t*num_pool+p] that the
 * shared pack/install steps consume.  Pure local walk over already-exchanged
 * queries -- no routing, no collectives.  CALLER OWNS the returned buffer (free
 * it).  The walk-export producer returns the identical layout, so the install
 * path stays shared between the two. */
static char *compute_matched_broadcast(
    const struct gx_query_t *all_queries, const int *q_disps, const int *all_q_counts,
    const float *h_compact_xyzh, const sfc_tile_t *h_tiles, int ntiles,
    const int *h_pool, int num_pool, const int *h_pool_types, unsigned int supply_mask,
    const tile_bvh_node_t *h_bvh, int bvh_root, int search_mode)
{
    /* Per-peer match bitmask over pool indices (dedup multiple queries → one ghost). */
    char *matched = (char *) calloc((size_t)NTask * (size_t)(num_pool > 0 ? num_pool : 1), sizeof(char));
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask) continue;
        int q_start = q_disps[t];
        int q_count = all_q_counts[t];
        char *match_for_t = matched + (size_t)t * (size_t)num_pool;
        for(int qi = 0; qi < q_count; qi++) {
            const struct gx_query_t *q = &all_queries[q_start + qi];
            /* q->h already includes safety_factor (set at query-build time).
             * compact_xyzh[*4+3] also includes safety_factor. So leaf r² check
             * uses inflated radii on both sides of max(h_q, h_j). */
            gx_walk_local_bvh(h_compact_xyzh, h_tiles, ntiles, h_pool, num_pool,
                              h_pool_types, supply_mask,
                              h_bvh, bvh_root,
                              q->pos, q->h, search_mode,
                              match_for_t, NULL);
        }
    }
    return matched;
}

/* Lazy, idempotent collective broadcast of the local query lists to all ranks
 * (the legacy Step-2 Allgather/Allgatherv).  Safe to call multiple times: no-op
 * once *available.  MUST be called collectively (all ranks together).  Fills the
 * three malloc'd arrays (caller frees) + total_queries.  Enables the late
 * fallback: if routed discovery fails AFTER Step 2 was skipped, all ranks return
 * to the same point and run this collectively before the broadcast walk. */
static void ensure_broadcast_queries(int *available,
                                     const struct gx_query_t *local_queries, int n_local_queries,
                                     int **all_q_counts_io, int **q_disps_io,
                                     struct gx_query_t **all_queries_io, int *total_queries_io)
{
    if(*available) return;
    int *all_q_counts = (int *) malloc(NTask * sizeof(int));
    MPI_Allgather(&n_local_queries, 1, MPI_INT, all_q_counts, 1, MPI_INT, MPI_COMM_WORLD);
    int *q_disps = (int *) malloc(NTask * sizeof(int));
    int total_queries = 0;
    for(int t = 0; t < NTask; t++) { q_disps[t] = total_queries; total_queries += all_q_counts[t]; }
    struct gx_query_t *all_queries = (struct gx_query_t *)
        malloc((size_t)(total_queries > 0 ? total_queries : 1) * sizeof(struct gx_query_t));
    int *q_byte_counts = (int *) malloc(NTask * sizeof(int));
    int *q_byte_disps  = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        q_byte_counts[t] = all_q_counts[t] * (int)sizeof(struct gx_query_t);
        q_byte_disps[t]  = q_disps[t]      * (int)sizeof(struct gx_query_t);
    }
    MPI_Allgatherv(local_queries, n_local_queries * (int)sizeof(struct gx_query_t), MPI_BYTE,
                   all_queries, q_byte_counts, q_byte_disps, MPI_BYTE, MPI_COMM_WORLD);
    free(q_byte_counts); free(q_byte_disps);
    *all_q_counts_io = all_q_counts; *q_disps_io = q_disps;
    *all_queries_io = all_queries;   *total_queries_io = total_queries;
    *available = 1;
}

/* Walk-export routed producer — the discovery path for every spec that passes
 * gx_walk_export_eligible().  Sender: per local query mode_b_walk_and_export -> per-peer
 * NodeList -> fixed-size envelopes -> Alltoallv.  Receiver: mode_b_walk_from_start_nodes
 * (resume from the exported NodeList) -> gx_pair_accept_wrap_and_test (the ghost-exchange SSOT predicate)
 * -> matched[t*num_pool+p] bitmap, the same layout the shared Steps 4-6 install consume.
 * MODE-GENERIC (search_mode is passed through): this is the install target for both search
 * modes, so ONEWAY and SYMMETRIC discover on ONE substrate rather than two.
 *
 * COLLECTIVE-SAFE (C MPI buffers): every C allocation preceding a collective — the index
 * arrays before the Alltoall, the envelope buffers before the Alltoallv, the matched bitmap
 * — is Allreduce-checked, so a NULL on ANY rank makes ALL ranks return the same status and
 * ranks never diverge across a collective.  NOT covered: the C++ containers
 * (ModeBExportSink, the per-peer envelope vectors) throw std::bad_alloc rank-locally rather
 * than returning NULL, so an allocation failure there aborts that rank instead of returning
 * a uniform status.  Returns a caller-owned matched bitmap on GX_WALK_EXPORT_OK, else NULL.
 *
 * THREADING (host): the sender-query loop and the received-envelope walk are both
 * `omp parallel for` — per-thread export sink and per-thread send buffers on the sender,
 * pre-sized per-envelope candidate slots on the receiver, so each thread writes only its own
 * index.  The walker's lazy node drift is the one shared mutation and is serialized under
 * critical(_modebdrift_).  Merge, accept and bitmap set run serially afterwards, which keeps
 * the resulting SET order-independent and therefore deterministic across thread counts.
 *
 * MEMORY SHAPE: the threaded receiver materializes one candidate list per received envelope
 * before the serial accept pass.  That is bounded by the exported envelope volume, which is
 * measured sparse; a pathologically clustered geometry with very large tot_r would grow it,
 * so it is a known scaling watch-point rather than a fixed bound. */
enum { GX_WALK_EXPORT_OK = 0, GX_WALK_EXPORT_UNAVAILABLE = 1, GX_WALK_EXPORT_ALLOC_FAIL = 2 };
struct gx_walk_export_result {
    int status;   /* GX_WALK_EXPORT_OK / _UNAVAILABLE / _ALLOC_FAIL, uniform across ranks */
};

static char *compute_matched_walk_export(
    const struct ghost_exchange_spec_t *spec,
    const struct gx_query_t *local_queries, int n_local_queries,
    int num_pool, unsigned int supply_mask, int search_mode,
    struct gx_walk_export_result *res)
{
    if(res) memset(res, 0, sizeof(*res));
    /* The walk searches with the SAME j-side reach the accept admits with.
     * The accept uses gx_policy_scaled_h = radius(j,policy) * j_radius_scale *
     * safety_factor, so both factors are folded here; the query side already
     * carries safety (queries are built as h*safety). Were the walk to search
     * with a smaller reach than the accept admits, it would silently discover
     * fewer pairs than the caller asked for -- which is why a safety factor
     * above 1 previously had to fall back to broadcast. */
    const double walker_j_reach_scale = spec->j_radius_scale * spec->safety_factor;
    /* (a) tree availability — collective all-or-none (a rank-local skip would deadlock the
     * envelope Alltoallv below / the caller's compare Allreduce). */
    int ok_local = (All.TreeNodeIndexBase > 0 && Nodes != NULL && Nextnode != NULL) ? 1 : 0;
    int ok_all = 0;
    MPI_Allreduce(&ok_local, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!ok_all) { if(res) res->status = GX_WALK_EXPORT_UNAVAILABLE; return NULL; }

    /* (b) SENDER: build per-peer envelope lists (export is a byproduct of the walk).
     * THREADED, following the same shape the runner uses: the topleaf map is built once
     * and READ-ONLY during the walk (shared); each thread uses its own ModeBExportSink + its own
     * per-peer send buffers; the walker's node lazy-drift is race-safe (omp
     * critical(_modebdrift_) + release/acquire).  Merge is serial.  The routed SET is
     * unchanged (order-independent bitmap); only per-peer envelope ORDER differs (D6: FP-reorder only). */
    ModeBTopleafMap map; map.build();
    std::vector<std::vector<struct gx_export_envelope_t>> send(NTask);
    {
#ifdef _OPENMP
        int nthr = omp_get_max_threads();
#else
        int nthr = 1;
#endif
        if(nthr < 1) nthr = 1;
        std::vector<ModeBExportSink> tsink(nthr);
        for(int th = 0; th < nthr; th++) tsink[th].ensure_size(NTask);
        std::vector<std::vector<std::vector<struct gx_export_envelope_t>>> tsend(nthr);
        for(int th = 0; th < nthr; th++) tsend[th].assign(NTask, std::vector<struct gx_export_envelope_t>{});
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for(int qi = 0; qi < n_local_queries; qi++) {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            ModeBExportSink &sink = tsink[tid];
            sink.clear_all();
            mode_b_walk_and_export(local_queries[qi].pos, local_queries[qi].h,
                                   supply_mask, search_mode, spec->radius_policy,
                                   /*cand_out=*/NULL, map, sink, walker_j_reach_scale, /*drift_ctr=*/NULL);
            for(int t = 0; t < NTask; t++) {
                if(t == ThisTask) continue;
                long nn = (long)sink.nodes_per_peer[t].size();
                if(nn <= 0) continue;
                const std::vector<int> &nl = sink.nodes_per_peer[t];
                for(long base = 0; base < nn; base += NODELISTLENGTH) {
                    struct gx_export_envelope_t e;
                    e.pos[0] = local_queries[qi].pos[0];
                    e.pos[1] = local_queries[qi].pos[1];
                    e.pos[2] = local_queries[qi].pos[2];
                    e.h = local_queries[qi].h;
                    e.n_nodes = (int)((nn - base < NODELISTLENGTH) ? (nn - base) : NODELISTLENGTH);
                    for(int k = 0; k < e.n_nodes; k++) e.nodes[k] = nl[base + k];
                    e._pad = 0;
                    tsend[tid][t].push_back(e);
                }
            }
        }
        /* serial merge: concatenate per-thread envelopes per peer. */
        for(int th = 0; th < nthr; th++) {
            for(int t = 0; t < NTask; t++)
                send[t].insert(send[t].end(), tsend[th][t].begin(), tsend[th][t].end());
        }
    }

    /* (c) EXCHANGE: counts Alltoall + typed Alltoallv of fixed-size envelopes. */
    int *sc = (int *) calloc(NTask, sizeof(int));
    int *sd = (int *) calloc(NTask, sizeof(int));
    int *rc = (int *) calloc(NTask, sizeof(int));
    int *rd = (int *) calloc(NTask, sizeof(int));
    if(!sc || !sd || !rc || !rd) {
        /* index-array alloc failure is per-rank; make it collective before the Alltoall. */
        int a_ok_local = 0, a_ok_all = 0;
        MPI_Allreduce(&a_ok_local, &a_ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        free(sc); free(sd); free(rc); free(rd);
        if(res) res->status = GX_WALK_EXPORT_ALLOC_FAIL; return NULL;
    } else {
        int a_ok_local = 1, a_ok_all = 0;
        MPI_Allreduce(&a_ok_local, &a_ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(!a_ok_all) { free(sc); free(sd); free(rc); free(rd); if(res) res->status = GX_WALK_EXPORT_ALLOC_FAIL; return NULL; }
    }
    /* Per-peer counts are bounded by this rank's query set, but the prefix sums and
     * totals are not: the typed Alltoallv below consumes int displacements, so a value
     * past INT_MAX cannot be represented and would wrap into a NEGATIVE displacement,
     * making the memcpy that fills sendbuf an out-of-bounds write before MPI is ever
     * reached.  Same failure mode, and the same collective treatment, as the Step-4
     * transport guard: detect before mutating anything, agree across ranks, and report
     * a uniform status rather than truncating. */
    long tot_s = 0, tot_r = 0;
    int env_range_ok = 1;
    for(int t = 0; t < NTask; t++) {
        long n = (long)send[t].size();
        if(n > INT_MAX) { env_range_ok = 0; n = 0; }
        sc[t] = (int)n;
    }
    MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD);
    for(int t = 0; t < NTask; t++) {
        if(tot_s <= INT_MAX) sd[t] = (int)tot_s; else { sd[t] = 0; env_range_ok = 0; }
        if(tot_r <= INT_MAX) rd[t] = (int)tot_r; else { rd[t] = 0; env_range_ok = 0; }
        tot_s += sc[t];
        tot_r += rc[t];
    }
    if(tot_s > INT_MAX || tot_r > INT_MAX) env_range_ok = 0;
    {
        int range_ok_all = 0;
        MPI_Allreduce(&env_range_ok, &range_ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(!range_ok_all) {
            if(ThisTask == 0) {
                printf("ERROR: walk-export envelope counts exceed int MPI transport range (caller=%s)\n",
                       spec->caller_name ? spec->caller_name : "?");
                fflush(stdout);
            }
            free(sc); free(sd); free(rc); free(rd);
            if(res) res->status = GX_WALK_EXPORT_UNAVAILABLE;
            return NULL;
        }
    }
    struct gx_export_envelope_t *sendbuf = (struct gx_export_envelope_t *) malloc((size_t)(tot_s > 0 ? tot_s : 1) * sizeof(struct gx_export_envelope_t));
    struct gx_export_envelope_t *recv    = (struct gx_export_envelope_t *) malloc((size_t)(tot_r > 0 ? tot_r : 1) * sizeof(struct gx_export_envelope_t));
    /* buffer alloc — collective before the Alltoallv (a NULL recv would fault the collective). */
    int buf_ok_local = (sendbuf && recv) ? 1 : 0, buf_ok_all = 0;
    MPI_Allreduce(&buf_ok_local, &buf_ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!buf_ok_all) { free(sendbuf); free(recv); free(sc); free(sd); free(rc); free(rd);
                      if(res) res->status = GX_WALK_EXPORT_ALLOC_FAIL; return NULL; }
    for(int t = 0; t < NTask; t++)
        if(sc[t] > 0) memcpy(sendbuf + sd[t], send[t].data(), (size_t)sc[t] * sizeof(struct gx_export_envelope_t));
    gizmo_mpi_alltoallv_typed(sendbuf, sc, sd, recv, rc, rd,
                              sizeof(struct gx_export_envelope_t), MPI_COMM_WORLD);
    free(sendbuf); free(sc); free(sd);

    /* (d) matched bitmap — collective before the RECEIVER walk (the caller reduces over the
     * compare, so a NULL here on one rank would diverge the caller's Allreduce). */
    char *matched_walk_export = (char *) calloc((size_t)NTask * (size_t)(num_pool > 0 ? num_pool : 1), 1);
    int m_ok_local = (matched_walk_export != NULL) ? 1 : 0, m_ok_all = 0;
    MPI_Allreduce(&m_ok_local, &m_ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!m_ok_all) { free(matched_walk_export); free(recv); free(rc); free(rd);
                    if(res) res->status = GX_WALK_EXPORT_ALLOC_FAIL; return NULL; }

    /* (e) RECEIVER: bounded resume-walk from the exported NodeList -> SSOT accept -> bitmap.
     * Two interchangeable backends produce the SAME bitmap; the set is
     * order-independent (a bit is set or it is not), so which one ran is not
     * observable downstream.
     *
     * The device traversal is tried first and answers whenever the tree mirror
     * is current.  It declines rank-locally otherwise, and this window holds no
     * collectives, so a rank that declines simply does the work itself and no
     * other rank needs to agree.  Declining is for an unusable device tree state
     * or an allocation failure -- never for a disagreement, which would be a bug
     * to fix rather than to route around. */
    int receiver_done_on_device = 0;
    if(tot_r > 0) {
        std::vector<int> envelope_peer((size_t)tot_r, -1);
        for(int t = 0; t < NTask; t++) {
            for(int r = 0; r < rc[t]; r++) {envelope_peer[(size_t)rd[t] + r] = t;}
        }
        receiver_done_on_device =
            (gx_device_receiver_walk(recv, tot_r, envelope_peer.data(),
                                     supply_mask, search_mode,
                                     spec->radius_policy, walker_j_reach_scale,
                                     g_glt_cache.j_to_pool, g_glt_cache.NumPart_when_built,
                                     num_pool, matched_walk_export) == 0);
    }
    /* Host backend.  THREADED: the WALK (dominant cost) runs per received envelope into a pre-sized
     * per-envelope cand slot — each thread writes ONLY its own index (no shared write), walker
     * race-safe.  The ACCEPT + bitmap set run SERIALLY afterward (cheap), which keeps the
     * matched_walk_export bitmap race-free. */
    if(!receiver_done_on_device) {
        std::vector<std::vector<int>> per_recv_cands((size_t)(tot_r > 0 ? tot_r : 0));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for(long k = 0; k < tot_r; k++) {
            const struct gx_export_envelope_t *e = &recv[k];
            std::vector<int> &cvk = per_recv_cands[k];
            cvk.clear();
            mode_b_walk_from_start_nodes(e->pos, e->h, supply_mask, search_mode,
                                         spec->radius_policy, e->nodes, e->n_nodes,
                                         cvk, walker_j_reach_scale, NULL);
        }
        for(int t = 0; t < NTask; t++) {
            if(t == ThisTask) continue;
            char *mf = matched_walk_export + (size_t)t * num_pool;
            for(int r = 0; r < rc[t]; r++) {
                const long k = (long)rd[t] + r;
                const struct gx_export_envelope_t *e = &recv[k];
                const std::vector<int> &cvk = per_recv_cands[k];
                for(size_t c = 0; c < cvk.size(); c++) {
                    int j = cvk[c];
                    if(j < 0 || j >= g_glt_cache.NumPart_when_built) continue;
                    int pp = g_glt_cache.j_to_pool ? g_glt_cache.j_to_pool[j] : -1;
                    if(pp < 0 || pp >= num_pool) continue;
                    /* Reach comes from THIS caller's spec, not from whatever policy
                     * the cached pool happened to be built under.  The sender walk
                     * above already uses the spec, so taking it from the cache here
                     * would let a caller inherit another caller's j-side reach now
                     * that pool membership is reused across differing radius
                     * policies.  Only the tile/BVH/compact geometry may use the
                     * build-time policy, because its leaf h was baked with it. */
                    double hj_dbl = gx_policy_scaled_h(j, spec->radius_policy,
                                                       spec->j_radius_scale,
                                                       spec->safety_factor);
                    if(gx_pair_accept_wrap_and_test(e->pos[0] - (double)P[j].Pos[0],
                                                    e->pos[1] - (double)P[j].Pos[1],
                                                    e->pos[2] - (double)P[j].Pos[2],
                                                    e->h, hj_dbl, search_mode)) {
                        mf[pp] = 1;   /* idempotent: set semantics, duplicates are a no-op */
                    }
                }
            }
        }
    }
    free(recv); free(rc); free(rd);
    if(res) res->status = GX_WALK_EXPORT_OK;
    return matched_walk_export;
}

/* Non-finite test that survives fast-math (isnan/isfinite may fold to false under
 * -ffinite-math-only): exponent all-ones => Inf or NaN. A non-finite query position
 * or radius makes every distance test "match" and would import the whole domain, so
 * the request-driven build below fails closed on it. */
static inline int gx_query_nonfinite(double v) {
    unsigned long long b; memcpy(&b, &v, sizeof(b));
    return (((b >> 52) & 0x7ffULL) == 0x7ffULL);
}

static ghost_exchange_result ghost_exchange_request_driven_impl(const struct ghost_exchange_spec_t *spec)
{
    if(NTask <= 1) return GHOST_EXCHANGE_COMPLETED;
    const double safety_factor = spec->safety_factor;
    const unsigned int request_mask = spec->request_type_mask;
    const unsigned int supply_mask  = spec->supply_type_mask;
    const int  search_mode = spec->search_mode;
    double t_ghost_start = my_second();
    NumPart_before_ghost = NumPart;
    N_gas_before_ghost = N_gas;
    NumGhostParticles = 0;

    static int gx_call_seq_rd = 0;
    gx_call_seq_rd++;
    int this_call = gx_call_seq_rd;

    /* === Step 1: build local query list === */
    /* Two paths into the wire-format queries:
     *   (a) Caller-explicit (mech_fb migration): spec->n_queries >= 0 — caller
     *       supplied the source list (positions+h). We just copy into
     *       gx_query_t with safety_factor applied. No request_type_mask
     *       filtering — caller did that already in their isactive scan.
     *   (b) Legacy back-compat: spec->n_queries < 0 — scan ActiveParticleList,
     *       filter by spec->request_type_mask, build queries from
     *       P[i].Pos/KernelRadius. Used by ghost_exchange / ghost_exchange_hydro
     *       / ghost_exchange_hydro_oneway wrappers. */
    int n_local_queries = 0;
    struct gx_query_t *local_queries = NULL;
    int q_bad = -1;   /* first query index with a non-finite pos/h; -1 = all finite */
    if(spec->n_queries >= 0) {
        n_local_queries = spec->n_queries;
        local_queries = (struct gx_query_t *)
            malloc((size_t)(n_local_queries > 0 ? n_local_queries : 1) * sizeof(struct gx_query_t));
        for(int q = 0; q < n_local_queries; q++) {
            local_queries[q].pos[0] = spec->query_pos[q][0];
            local_queries[q].pos[1] = spec->query_pos[q][1];
            local_queries[q].pos[2] = spec->query_pos[q][2];
            local_queries[q].h    = spec->query_h[q] * safety_factor;
            local_queries[q].type = -1;   /* unspecified for caller-explicit */
            local_queries[q]._pad = 0;
            if(q_bad < 0 &&
               (gx_query_nonfinite(local_queries[q].pos[0]) || gx_query_nonfinite(local_queries[q].pos[1]) ||
                gx_query_nonfinite(local_queries[q].pos[2]) || gx_query_nonfinite(local_queries[q].h))) q_bad = q;
        }
    } else {
        for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
            int i = ActiveParticleList[kk];
            if(i < 0 || i >= NumPart) continue;
            if(P[i].Mass <= 0) continue;
            if(!ghost_type_passes((int)P[i].Type, request_mask)) continue;
            n_local_queries++;
        }
        local_queries = (struct gx_query_t *)
            malloc((size_t)(n_local_queries > 0 ? n_local_queries : 1) * sizeof(struct gx_query_t));
        int q = 0;
        for(size_t kk = 0; kk < ActiveParticleList.size(); kk++) {
            int i = ActiveParticleList[kk];
            if(i < 0 || i >= NumPart) continue;
            if(P[i].Mass <= 0) continue;
            if(!ghost_type_passes((int)P[i].Type, request_mask)) continue;
            local_queries[q].pos[0] = P[i].Pos[0];
            local_queries[q].pos[1] = P[i].Pos[1];
            local_queries[q].pos[2] = P[i].Pos[2];
            double h = (double)P[i].KernelRadius;
            local_queries[q].h    = h * safety_factor;
            local_queries[q].type = (int)P[i].Type;
            local_queries[q]._pad = 0;
            if(q_bad < 0 &&
               (gx_query_nonfinite(local_queries[q].pos[0]) || gx_query_nonfinite(local_queries[q].pos[1]) ||
                gx_query_nonfinite(local_queries[q].pos[2]) || gx_query_nonfinite(local_queries[q].h))) q_bad = q;
            q++;
        }
    }

    /* Fail-closed: a non-finite query position or radius makes every distance test
     * "match" and would import the entire domain (a corrupt query must never turn
     * into "ship everything"). Stop loudly instead; drains collectively at the poll. */
    if(q_bad >= 0) {
        printf("ERROR: non-finite request-driven query on task %d (caller=%s q=%d pos=(%g,%g,%g) h=%g)\n",
               ThisTask, (spec->caller_name ? spec->caller_name : "?"), q_bad,
               local_queries[q_bad].pos[0], local_queries[q_bad].pos[1], local_queries[q_bad].pos[2], local_queries[q_bad].h);
        gizmo_request_controlled_stop(7708, "ghost_exchange (request-driven): non-finite query position or radius", __FILE__, __LINE__, __FUNCTION__);
    }
    gizmo_exit_bad_stop_if_requested("ghost_exchange:query_finite");

    /* === Step 2: LAZY query distribution === The broadcast Allgather/
     * Allgatherv now runs on demand via ensure_broadcast_queries(): SKIPPED in
     * routed-production mode, run up-front for the oracle / when routed is
     * unavailable, and as a collective LATE fallback if routed fails after the
     * skip.  Declarations only here; the gather (if any) happens after the
     * supply-snapshot build below, alongside the routed-vs-broadcast selection. */
    int   *all_q_counts = NULL;
    int   *q_disps      = NULL;
    struct gx_query_t *all_queries = NULL;
    int    total_queries = 0;
    int    bcast_queries_available = 0;

    /* === Step 3: per-rank, walk local BVH against each remote rank's queries ===
     *
     * Build a host-side SFC tile + BVH index over the supply-mask-filtered pool.
     * For each remote query, walk the BVH (O(log N + matches) per query, vs the
     * O(N) brute-force scan that had been a tiny-N proof-of-concept). Per-particle
     * acceptance at leaves uses the EXACT predicate (ONEWAY: r²<h_q²; SYMMETRIC:
     * r²<max(h_q,h_j)²) — same predicate the kernel applies later. */

    /* SHARED-TREE build: build over GHOST_TYPE_ALL (all types with mass>0), not
     * just the caller's supply_mask. The walker's per-type hmax filter +
     * per-particle leaf Type-vs-supply_mask check delivers the same imports
     * as the old per-supply-mask build.
     *
     * Bucket 3 (SIDX overlay): the local tree (tiles, pool, bvh,
     * compact_xyzh, pool_types) is cached across calls within a step.
     * Cache invalidated via ghost_exchange_local_tree_invalidate_*()
     * hooks at drift / domain_decomp boundaries (run.cc). Within a step
     * the pool is stable, so 2nd..Nth calls skip this whole stanza. */

    sfc_tile_t *h_tiles = NULL;
    int *h_pool = NULL;
    int num_pool = 0;
    int ntiles = 0;
    tile_bvh_node_t *h_bvh = NULL;
    int bvh_nnodes = 0;
    int bvh_root = 0;
    float *h_compact_xyzh = NULL;
    int *h_pool_types = NULL;
    int from_cache = 0;
    /* All particle types are eligible as ghost sources. */
    unsigned int desired_pool_mask = GHOST_TYPE_ALL;
    /* Tile/BVH/compact geometry is built ONLY for a call that actually walks it.
     * The walk-export producer discovers through the tree and reads nothing
     * position-dependent from this cache (it needs pool membership to index the
     * matched set, and takes its supply band from Extnodes), so when it is the
     * authority the geometry would be built and never read.  Structural, keyed on
     * the same predicate as producer selection — no caller names. */
    /* ONE predicate drives BOTH what the cache holds and which producers may run
     * below: geometry is skipped exactly when no consumer of it can execute.  The
     * diag path is part of that condition — it re-enables the broadcast walk, which
     * reads this geometry — so a second, weaker copy of this test here would let a
     * diag run walk NULL tiles.  Keep this as the single definition. */
    const int walk_export_only    = gx_walk_export_eligible(spec)
                           && !gizmo_nlr_phase0_diag_enabled();
    const unsigned int wanted_caps = GX_POOL_IDENTITY | (walk_export_only ? 0u : GX_POOL_GEOMETRY);
    /* TWO validities, because the cache holds two payloads with different
     * dependencies.  IDENTITY (pool, j_to_pool, num_pool) is membership and order:
     * it depends on {NumPart, type mask, epoch} and on nothing positional, so it
     * survives drift and survives a caller arriving with a different radius policy.
     * GEOMETRY (tiles, bvh, compact_xyzh, pool_types) additionally depends on
     * {safety, radius_policy, j_radius_scale} and on live positions/h, so it is
     * rebuilt on a policy change and REFIT (not rebuilt) on drift.
     *
     * Keeping these separate is the point: rebuilding the identity map costs a
     * full pass over the local particles, and callers within a step differ in
     * radius policy far more often than the particle set changes.
     *
     * Mask is compared for EXACT equality, not coverage. Reuse across a narrowed
     * mask is unproven here — a narrower request would also make in-place Type
     * changes membership-relevant, which the epoch does not track — so anything
     * other than the all-types pool falls through to a full rebuild. */
    const int mask_reusable = (desired_pool_mask == GHOST_TYPE_ALL);
    const int identity_valid = (g_glt_cache.valid
                       && (g_glt_cache.caps & GX_POOL_IDENTITY)
                       && g_glt_cache.pool && g_glt_cache.j_to_pool
                       && mask_reusable
                       && g_glt_cache.eligible_type_mask_when_built == desired_pool_mask
                       && g_glt_cache.NumPart_when_built == NumPart
                       && g_glt_cache.identity_epoch_when_built == g_supply_identity_epoch);
    const int geometry_valid = (identity_valid
                       && (g_glt_cache.caps & GX_POOL_GEOMETRY)
                       && g_glt_cache.safety_factor_when_built == safety_factor
                       && g_glt_cache.radius_policy_when_built == spec->radius_policy
                       && g_glt_cache.j_radius_scale_when_built == spec->j_radius_scale);
    const int want_geometry = (wanted_caps & GX_POOL_GEOMETRY) ? 1 : 0;
    int cache_match = identity_valid && (!want_geometry || geometry_valid);
    int geometry_rebuilt = 0;

    /* Identity still good, geometry stale or absent: rebuild geometry ONLY, over
     * the pool we already hold.  This is the case a single fused key could not
     * express, and it is the common one — successive callers in a step share the
     * particle set and differ only in radius policy. */
    if(identity_valid && want_geometry && !geometry_valid) {
        glt_cache_free_geometry_();
        sfc_tile_t *g_tiles = NULL;
        int g_ntiles = build_sfc_tiles_from_pool(P, g_glt_cache.pool, g_glt_cache.num_pool,
                                                 TILE_TARGET_SIZE, &g_tiles,
                                                 spec->radius_policy,
                                                 spec->j_radius_scale * safety_factor);
        tile_bvh_node_t *g_bvh = NULL;
        int g_bvh_nnodes = build_tile_bvh(g_tiles, g_ntiles, &g_bvh);
        size_t gz_tiles   = (size_t)(g_ntiles > 0 ? g_ntiles : 1) * sizeof(sfc_tile_t);
        size_t gz_bvh     = (size_t)(g_bvh_nnodes > 0 ? g_bvh_nnodes : 1) * sizeof(tile_bvh_node_t);
        size_t gz_compact = (size_t)(g_glt_cache.num_pool > 0 ? g_glt_cache.num_pool : 1) * 4 * sizeof(float);
        size_t gz_types   = (size_t)(g_glt_cache.num_pool > 0 ? g_glt_cache.num_pool : 1) * sizeof(int);
        sfc_tile_t      *gc_tiles   = (sfc_tile_t *)      malloc(gz_tiles);
        tile_bvh_node_t *gc_bvh     = (tile_bvh_node_t *) malloc(gz_bvh);
        float           *gc_compact = (float *)           malloc(gz_compact);
        int             *gc_types   = (int *)             malloc(gz_types);
        if(!gc_tiles || !gc_bvh || !gc_compact || !gc_types) {
            /* Same fail-closed shape as the full rebuild below: this producer is the
             * only supplier, so drop the whole entry and let the full path re-decide
             * rather than publishing a half-built one. */
            free(gc_tiles); free(gc_bvh); free(gc_compact); free(gc_types);
            if(g_bvh)   myfree(g_bvh);
            if(g_tiles) myfree(g_tiles);
            glt_cache_free();
        } else {
            if(g_ntiles > 0)      memcpy(gc_tiles, g_tiles, (size_t)g_ntiles * sizeof(sfc_tile_t));
            if(g_bvh_nnodes > 0)  memcpy(gc_bvh,   g_bvh,   (size_t)g_bvh_nnodes * sizeof(tile_bvh_node_t));
            for(int p = 0; p < g_glt_cache.num_pool; p++) {
                int j = g_glt_cache.pool[p];
                gc_compact[p*4+0] = (float)P[j].Pos[0];
                gc_compact[p*4+1] = (float)P[j].Pos[1];
                gc_compact[p*4+2] = (float)P[j].Pos[2];
                gc_compact[p*4+3] = (float)gx_policy_scaled_h(j, spec->radius_policy,
                                                              spec->j_radius_scale, safety_factor);
                gc_types[p] = (int)P[j].Type;
            }
            if(g_bvh)   myfree(g_bvh);
            if(g_tiles) myfree(g_tiles);
            g_glt_cache.tiles        = gc_tiles;
            g_glt_cache.bvh          = gc_bvh;
            g_glt_cache.compact_xyzh = gc_compact;
            g_glt_cache.pool_types   = gc_types;
            g_glt_cache.ntiles       = g_ntiles;
            g_glt_cache.bvh_nnodes   = g_bvh_nnodes;
            g_glt_cache.bvh_root     = g_bvh_nnodes - 1;
            g_glt_cache.safety_factor_when_built  = safety_factor;
            g_glt_cache.radius_policy_when_built  = spec->radius_policy;
            g_glt_cache.j_radius_scale_when_built = spec->j_radius_scale;
            g_glt_cache.Ti_when_built = All.Ti_Current;
            g_glt_cache.needs_refit = 0;
            g_glt_cache.caps |= GX_POOL_GEOMETRY;
            /* Geometry was just seeded from current P[]; marks predating it are
             * obsolete, exactly as after a full build. */
            g_glt_dirty_clear_();
            cache_match = 1;
            geometry_rebuilt = 1;
        }
    }
    if(cache_match) {
        h_tiles        = g_glt_cache.tiles;
        h_pool         = g_glt_cache.pool;
        num_pool       = g_glt_cache.num_pool;
        ntiles         = g_glt_cache.ntiles;
        h_bvh          = g_glt_cache.bvh;
        bvh_nnodes     = g_glt_cache.bvh_nnodes;
        bvh_root       = g_glt_cache.bvh_root;
        h_compact_xyzh = g_glt_cache.compact_xyzh;
        h_pool_types   = g_glt_cache.pool_types;
        from_cache = 1;
        /* A geometry-only rebuild reused the pool but did real build work, so it is
         * not a hit. Counting it as one would overstate the cache and hide the very
         * cost this split exists to measure. */
        if(geometry_rebuilt) g_glt_cache_misses++; else g_glt_cache_hits++;
        /* Refit refreshes position-dependent geometry only; an identity-only
         * entry has none and its membership does not move with the particles. */
        if((g_glt_cache.caps & GX_POOL_GEOMETRY)
           && (g_glt_cache.needs_refit || g_glt_cache.Ti_when_built != All.Ti_Current)) {
            glt_cache_refit_from_particles();
        }
    } else {
        /* RANK-LOCAL BRANCH — NO MPI CALLS IN HERE.  cache_match keys on rank-local
         * NumPart, so ranks enter this independently; any collective placed here
         * deadlocks as soon as one rank rebuilds and another does not. */
        g_glt_cache_misses++;
        /* Call-local on purpose: a file-static would latch across calls, and every
         * later rebuild would then skip fill+install while still publishing the
         * pool pointers below -- a NULL walk. */
        int cache_alloc_failed = 0;
        /* Free any stale entry before rebuild (cache key changed). */
        if(g_glt_cache.valid) glt_cache_free();

        /* Fresh build via existing mymalloc path; we copy the result into
         * malloc-backed cache buffers so it can outlive this function frame
         * without violating mymalloc LIFO ordering. */
        sfc_tile_t *tmp_tiles = NULL;
        int *tmp_pool = NULL;
        int tmp_num_pool = 0;
        /* Tile bands and leaf compact h are computed from the IDENTICAL
         * supply-side reach: nlr_particle_symmetric_radius(P[j], spec->radius_policy)
         * * spec->j_radius_scale * safety_factor.  build_sfc_tiles bakes the
         * scale into tile->hmax / tile->hmax_by_type[]; the loop below bakes it
         * into c_compact[p*4+3] via gx_policy_scaled_h.  No leaf-vs-band scale
         * gap, no BVH-prune-misses-pairs failure mode. */
        int tmp_ntiles = 0;
        tile_bvh_node_t *tmp_bvh = NULL;
        int tmp_bvh_nnodes = 0;
        if(wanted_caps & GX_POOL_GEOMETRY) {
            tmp_ntiles = build_sfc_tiles(P, NumPart, (int)desired_pool_mask, TILE_TARGET_SIZE,
                                         &tmp_tiles, &tmp_pool, &tmp_num_pool,
                                         spec->radius_policy,
                                         spec->j_radius_scale * safety_factor);
            tmp_bvh_nnodes = build_tile_bvh(tmp_tiles, tmp_ntiles, &tmp_bvh);
        } else {
            /* Membership only — same selection, none of the position-dependent work. */
            tmp_num_pool = build_sfc_supply_pool(P, NumPart, (int)desired_pool_mask, &tmp_pool);
        }
        int tmp_bvh_root = tmp_bvh_nnodes - 1;

        /* Allocate persistent cache buffers + copy. */
        size_t sz_tiles   = (size_t)(tmp_ntiles > 0 ? tmp_ntiles : 1) * sizeof(sfc_tile_t);
        size_t sz_pool    = (size_t)(tmp_num_pool > 0 ? tmp_num_pool : 1) * sizeof(int);
        size_t sz_bvh     = (size_t)(tmp_bvh_nnodes > 0 ? tmp_bvh_nnodes : 1) * sizeof(tile_bvh_node_t);
        size_t sz_compact = (size_t)(tmp_num_pool > 0 ? tmp_num_pool : 1) * 4 * sizeof(float);
        size_t sz_types   = (size_t)(tmp_num_pool > 0 ? tmp_num_pool : 1) * sizeof(int);
        const int with_geometry = (wanted_caps & GX_POOL_GEOMETRY) ? 1 : 0;
        sfc_tile_t      *c_tiles   = with_geometry ? (sfc_tile_t *)      malloc(sz_tiles)   : NULL;
        int             *c_pool    =                 (int *)             malloc(sz_pool);
        tile_bvh_node_t *c_bvh     = with_geometry ? (tile_bvh_node_t *) malloc(sz_bvh)     : NULL;
        float           *c_compact = with_geometry ? (float *)           malloc(sz_compact) : NULL;
        int             *c_types   =                 (int *)             malloc(sz_types);
        /* Reverse map j -> pool_pos (-1 if j is not in this build's pool). Sized
         * to NumPart_when_built; bounds-checked at narrow-refit lookup time. */
        size_t sz_jtop = (size_t)(NumPart > 0 ? NumPart : 1) * sizeof(int);
        int             *c_jtop    = (int *)             malloc(sz_jtop);
        /* An allocation failure here would otherwise be a segfault: the buffers are
         * written unconditionally just below, and this producer is now the only
         * supplier, so there is nothing to fall back to.  The request is RANK-LOCAL
         * on purpose -- this branch is entered per-rank (cache_match keys on
         * rank-local NumPart), so a collective here would deadlock whenever ranks
         * disagree about rebuilding.  The matching drain runs just past the branch,
         * where every rank converges and before anything reads the pool. */
        if(!c_pool || !c_types || !c_jtop
           || (with_geometry && (!c_tiles || !c_bvh || !c_compact))) {
            printf("ERROR: supply-cache allocation failed on task %d (num_pool=%d NumPart=%d geometry=%d)\n",
                   ThisTask, tmp_num_pool, NumPart, with_geometry);
            fflush(stdout);
            free(c_tiles); free(c_pool); free(c_bvh); free(c_compact); free(c_types); free(c_jtop);
            c_tiles = NULL; c_pool = NULL; c_bvh = NULL; c_compact = NULL; c_types = NULL; c_jtop = NULL;
            if(tmp_bvh)   myfree(tmp_bvh);
            if(tmp_tiles) myfree(tmp_tiles);
            if(tmp_pool)  myfree(tmp_pool);
            gizmo_request_controlled_stop(7724, "ghost_exchange: supply-cache allocation failed",
                                          __FILE__, __LINE__, __FUNCTION__);
            cache_alloc_failed = 1;
        }
        if(!cache_alloc_failed) {
        for(int j = 0; j < NumPart; j++) c_jtop[j] = -1;
        if(tmp_ntiles > 0)     memcpy(c_tiles, tmp_tiles, (size_t)tmp_ntiles * sizeof(sfc_tile_t));
        if(tmp_num_pool > 0)   memcpy(c_pool,  tmp_pool,  (size_t)tmp_num_pool * sizeof(int));
        if(tmp_bvh_nnodes > 0) memcpy(c_bvh,   tmp_bvh,   (size_t)tmp_bvh_nnodes * sizeof(tile_bvh_node_t));
        for(int p = 0; p < tmp_num_pool; p++) {
            int j = tmp_pool[p];
            if(with_geometry) {
                c_compact[p*4+0] = (float)P[j].Pos[0];
                c_compact[p*4+1] = (float)P[j].Pos[1];
                c_compact[p*4+2] = (float)P[j].Pos[2];
                /* SSOT: leaf compact h uses the SAME formula as build_sfc_tiles'
                 * per-particle aggregation above (= gx_policy_scaled_h).  Result:
                 * leaf h_j == tile band band's contribution from this particle. */
                c_compact[p*4+3] = (float)gx_policy_scaled_h(j, spec->radius_policy,
                                                            spec->j_radius_scale,
                                                            safety_factor);
            }
            c_types[p] = (int)P[j].Type;
            if(j >= 0 && j < NumPart) c_jtop[j] = p;
        }

        /* Free mymalloc temps in LIFO order. */
        if(tmp_bvh)   myfree(tmp_bvh);
        if(tmp_tiles) myfree(tmp_tiles);
        if(tmp_pool)  myfree(tmp_pool);

        /* Install in cache. */
        g_glt_cache.tiles = c_tiles;
        g_glt_cache.pool  = c_pool;
        g_glt_cache.bvh   = c_bvh;
        g_glt_cache.compact_xyzh = c_compact;
        g_glt_cache.pool_types   = c_types;
        g_glt_cache.j_to_pool    = c_jtop;
        g_glt_cache.ntiles    = tmp_ntiles;
        g_glt_cache.num_pool  = tmp_num_pool;
        g_glt_cache.bvh_nnodes = tmp_bvh_nnodes;
        g_glt_cache.bvh_root  = tmp_bvh_root;
        g_glt_cache.NumPart_when_built = NumPart;
        g_glt_cache.identity_epoch_when_built = g_supply_identity_epoch;
        g_glt_cache.Ti_when_built = All.Ti_Current;
        g_glt_cache.safety_factor_when_built = safety_factor;
        g_glt_cache.radius_policy_when_built = spec->radius_policy;
        g_glt_cache.j_radius_scale_when_built = spec->j_radius_scale;
        /* Fresh build seeded compact_xyzh from current P[]; any pre-existing
         * dirty marks are obsolete.  Next refresh decides full vs narrow from
         * marks accumulated AFTER this point. */
        g_glt_dirty_clear_();
        g_glt_cache.eligible_type_mask_when_built = desired_pool_mask;
        g_glt_cache.needs_refit = 0;
        g_glt_cache.caps = wanted_caps;
        g_glt_cache.valid = 1;
        }   /* end fill+install (skipped when the cache allocation failed) */

        h_tiles = c_tiles; h_pool = c_pool; num_pool = tmp_num_pool;
        ntiles = tmp_ntiles; h_bvh = c_bvh; bvh_nnodes = tmp_bvh_nnodes;
        bvh_root = tmp_bvh_root;
        h_compact_xyzh = c_compact; h_pool_types = c_types;
    }
    /* Both branches converge here, so this poll is reached by every rank: it drains a
     * rank-local supply-cache allocation failure into an all-rank controlled stop
     * BEFORE anything below dereferences the pool. */
    gizmo_exit_bad_stop_if_requested("ghost_exchange:supply_cache_alloc");
    (void)bvh_nnodes;

    /* Periodic flags / box sizes for the BVH walker. */


    /* Matched producer selection: for ONEWAY callers the routed top-leaf
     * discovery installs when top-leaf geometry is collectively available;
     * broadcast is the fail-closed path for any spec that is not routing-eligible.
     * Broadcast query gather is LAZY (ensure_broadcast_queries) — skipped when
     * routed installs. */
    /* Routed set from the walk-export producer, built further below so it reads the same
     * supply-cache snapshot as the rest of this call. */
    char *matched_walk_export = NULL;
    struct gx_walk_export_result walk_export_res;
    memset(&walk_export_res, 0, sizeof(walk_export_res));

    char *matched = NULL;
    int   used_routed = 0;


    /* walk_export_only (defined with the cache capabilities above) means: no consumer of the
     * tile/BVH geometry runs on this call, so the broadcast walk is skipped and the
     * geometry was never built.  On producer failure there is no geometry-based
     * fallback left, which is why that case is a controlled stop rather than a
     * silent switch to a walk whose inputs are absent.  Rank-uniform (spec constants
     * + diag), so ranks never split across the collectives below. */
    if(!walk_export_only) {
        /* Collective broadcast gather + walk.  (walk_export_only SKIPS this
         * — that path installs at the producer below and its failure is caught by the controlled
         * stop before Step 4, so matched is never NULL entering the count/pack loops.) */
        ensure_broadcast_queries(&bcast_queries_available, local_queries, n_local_queries,
                                 &all_q_counts, &q_disps, &all_queries, &total_queries);
        matched = compute_matched_broadcast(all_queries, q_disps, all_q_counts,
                                            h_compact_xyzh, h_tiles, ntiles,
                                            h_pool, num_pool, h_pool_types, supply_mask,
                                            h_bvh, bvh_root, search_mode);
        /* Broadcast is the safety path — its alloc failing is terminal.  Drain
         * COLLECTIVELY here, BEFORE Step 4 dereferences matched: a per-rank NULL
         * must become an all-rank controlled stop, never a NULL walk/segfault. */
        int bcast_fail_local = (matched == NULL) ? 1 : 0;
        int bcast_fail_any   = 0;
        MPI_Allreduce(&bcast_fail_local, &bcast_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(bcast_fail_any) {
            if(bcast_fail_local) {
                printf("ERROR: request-driven broadcast matched alloc failed on task %d.\n", ThisTask);
                gizmo_request_controlled_stop(7705, "ghost_exchange (request-driven): broadcast matched alloc failed",
                                              __FILE__, __LINE__, __FUNCTION__);
            }
            gizmo_exit_bad_stop_if_requested("ghost_exchange:broadcast_matched_alloc");
        }
    }

    /* An eligible SYMM spec installs the routed set BELOW this print, so used_routed is
     * not yet set for it; report the path that will actually be used or the parity grep
     * reads "bcast" on a routed call. A producer failure after this point prints
     * GX_R1_FALLBACK, which already invalidates the run as a routed timing arm. */
    const char *qdist = (used_routed || gx_walk_export_eligible(spec)) ? "routed" : "bcast";

    /* Walk-export discovery: produce the routed set (sender export + bounded receiver
     * walk, collective-safe) and INSTALL it for an eligible spec via the shared
     * ownership-transfer.  Placed here so it reads the SAME g_glt_cache snapshot as the
     * rest of this call.  Membership comes from the SSOT accept (gx_pair_accept_wrap_and_test), so the
     * only way this set can differ from a full walk is routing COVERAGE, which is what
     * the per-spec supply-band domination proof establishes. */
    const int walk_export_install = gx_walk_export_eligible(spec);
    if(walk_export_install && NTask > 1) {
        matched_walk_export = compute_matched_walk_export(spec, local_queries, n_local_queries,
                                                  num_pool, supply_mask, search_mode,
                                                  &walk_export_res);
        /* Install via the shared ownership-transfer, so Steps 4-6 are reached by exactly
         * one path whichever producer supplied the set. */
        if(matched_walk_export && walk_export_res.status == GX_WALK_EXPORT_OK) {
            if(matched) free(matched);
            matched = matched_walk_export; matched_walk_export = NULL;
            used_routed = 1;
        } else if(ThisTask == 0) {
            /* Report what ACTUALLY happens next, which depends on whether a reference
             * set exists.  In production (walk_export_only) none does, so the stop below
             * fires.  Under diag the broadcast walk ran, so the call continues on that
             * set -- correct physics, but NOT the routed substrate, so the run is not a
             * valid routed arm either way.
             * The GX_R1_FALLBACK token is kept only because the current run-validity
             * checks grep for it; it does not describe the mechanism.  Rename it to match
             * the surrounding names, or fold it into the general import-failure
             * reporting, once nothing greps for the old spelling. */
            printf("[GX_R1_FALLBACK call=%d caller=%s reason=producer_status_%d -> %s; INVALID as a routed arm]\n",
                   this_call, (spec->caller_name ? spec->caller_name : "?"), walk_export_res.status,
                   walk_export_only ? "controlled stop (no correctness-proven fallback)"
                                    : "continuing on the broadcast reference set");
            fflush(stdout);
        }
        free(matched_walk_export); matched_walk_export = NULL;
    }

    /* An eligible caller skipped the broadcast walk, so if the walk-export producer
     * did not install (UNAVAILABLE/ALLOC_FAIL) there is no set to install and no
     * substrate left that is known to be correct.  The tile/BVH and broadcast walks
     * both read the same cached geometry, which has been measured producing wrong
     * densities on a decomposition where the cached geometry went stale, so falling
     * back to either would trade a visible failure for a silent one.  Stop instead.
     * walk_export_only and the producer status are rank-uniform, so all ranks stop together.
     * The dominant failure mode is envelope allocation under memory pressure; the
     * recovery that fits it is a retry at reduced import padding inside this producer,
     * which does not exist yet — until it does, the honest outcome is this stop. */
    if(walk_export_only && matched == NULL) {
        gizmo_request_controlled_stop(7723,
            "ghost_exchange: walk-export producer unavailable and no correctness-proven fallback exists",
            __FILE__, __LINE__, __FUNCTION__);
        gizmo_exit_bad_stop_if_requested("ghost_exchange:walk_export_unavailable");
    }

    /* === Step 4: per-peer counts + index list === */
    int *send_count = (int *) mymalloc("gx_rd_sc", NTask * sizeof(int));
    int *recv_count = (int *) mymalloc("gx_rd_rc", NTask * sizeof(int));
    int *send_disp  = (int *) mymalloc("gx_rd_sd", NTask * sizeof(int));
    int *recv_disp  = (int *) mymalloc("gx_rd_rd", NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) { send_count[t] = 0; recv_count[t] = 0; }
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask) continue;
        char *match_for_t = matched + (size_t)t * (size_t)num_pool;
        int s = 0;
        for(int p = 0; p < num_pool; p++) if(match_for_t[p]) s++;
        send_count[t] = s;
    }
    MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    /* CHECKED int64 totals + prefix displacements (same rationale as the tile
     * impl). Request-driven is the last-resort Mode-A discovery — there is NO
     * further fallback — so both a representation overflow and a particle-slot
     * overflow fail HONESTLY via the collective controlled-stop poll, never a
     * silent int wrap or OOB append. */
    long long total_send_ll = 0, total_recv_ll = 0;
    int count_range_ok = 1;
    {
        long long sdisp = 0, rdisp = 0;
        for(int t = 0; t < NTask; t++) {
            if(sdisp <= INT_MAX) send_disp[t] = (int)sdisp; else { send_disp[t] = 0; count_range_ok = 0; }
            if(rdisp <= INT_MAX) recv_disp[t] = (int)rdisp; else { recv_disp[t] = 0; count_range_ok = 0; }
            sdisp += send_count[t];
            rdisp += recv_count[t];
        }
        total_send_ll = sdisp; total_recv_ll = rdisp;
        if(total_send_ll > INT_MAX || total_recv_ll > INT_MAX) count_range_ok = 0;
    }
    int total_send = count_range_ok ? (int)total_send_ll : 0;
    int total_recv = count_range_ok ? (int)total_recv_ll : 0;

    /* Ghosts cannot be refused: this is the last-resort Mode-A discovery and there is nothing
     * further to fall back to, so an import that does not fit the current capacity raises it.
     * The capacity is the size of one memory block, and growing it is legal here with the gravity
     * tree standing and no ghost yet written -- that is what makes it movable mid-step. The need is
     * EXACT, since request-driven discovery already counted it, so nothing is added on top: a
     * capacity only ever rises, and a margin would raise the run's footprint permanently on the
     * strength of one crowded step. Whether the memory exists is the allocator's answer rather than
     * a prediction of it; on failure the resize requests a controlled stop and leaves every array at
     * a capacity the advertised one is backed by. Purely local -- no communication, and nothing on
     * the common path but one comparison. Runs BEFORE the pack so the capacity is settled before
     * anything is written at &P[NumPart]; the guard below still decides whether the append happens. */
    {
        const long long required = (long long) NumPart + total_recv_ll;
        if(count_range_ok && required > (long long) All.MaxPart && required <= (long long) INT_MAX) {
            (void) resize_particle_storage((int) required);
        }
    }

    /* Check space (mirrors legacy guard).  Request-driven is the last-resort
     * Mode-A discovery — there is NO further fallback — so a count/displacement
     * overflow of the int MPI transport range, or ghosts that would not fit
     * P[]/CellP[], fail HONESTLY via the collective controlled-stop poll below.
     * This stays the ONE predicate that decides whether the append may happen: if
     * the growth above was refused or failed, it is this guard that stops the run. */
    if(!count_range_ok) {
        printf("ERROR: request-driven ghost exchange counts exceed int MPI transport range on task %d.\n", ThisTask);
        gizmo_request_controlled_stop(7703, "ghost_exchange (request-driven): ghost count/displacement exceeds int MPI transport range", __FILE__, __LINE__, __FUNCTION__);
    } else if(!ghost_particle_slots_fit((long long)NumPart + total_recv_ll)) {
        /* NOTE: growing P[]/CellP[] HERE does not work, and the attempt is instructive.
         * A capacity raise (resize_particle_storage, allocate.cc) reallocates correctly, but this call
         * site is inside an in-flight iterative neighbour loop: the runner refreshes its own
         * effective_args after a ghost import (neighbor_loop_runner.cc, "may have realloc'd
         * P/CellP") while the Spec hooks still read the original args, so DensitySpec::after_iter
         * dereferences the freed buffer and segfaults. That refresh was defensive and had never
         * been exercised, because MaxPart was fixed and nothing ever actually realloc'd.
         * Making growth safe here means auditing every holder of P/CellP across the loop
         * machinery -- and a missed one reads stale-but-valid memory, i.e. silent corruption
         * rather than a crash. Sizing is handled up front instead (see the ghost-headroom floor
         * in read_ic.cc/restart.cc), which is why this path should now be rare. */
        printf("ERROR: request-driven ghost exchange needs %d ghosts on task %d, only %d free.\n",
               total_recv, ThisTask, All.MaxPart - NumPart);
        gizmo_request_controlled_stop(7702, "ghost_exchange (request-driven): ghost append would exceed the particle capacity and the capacity could not be raised to hold it (add ranks/nodes, or reduce ghost-import demand)", __FILE__, __LINE__, __FUNCTION__);
    }
    /* Per-rank capacity check above is asymmetric; drain it at this all-rank poll
     * BEFORE Step 5, so no rank appends ghosts past MaxPart (OOB) or desyncs the
     * collective pack/exchange. Every rank reaches this unconditionally. */
    gizmo_exit_bad_stop_if_requested("ghost_exchange:capacity_rd");

    /* === Step 5: pack particle data + cell data + home_idx === */
    struct particle_data *send_P = (struct particle_data *) mymalloc("gx_rd_sP",
        (total_send > 0 ? total_send : 1) * sizeof(struct particle_data));
    struct gas_cell_data *send_CellP = (struct gas_cell_data *) mymalloc("gx_rd_sC",
        (total_send > 0 ? total_send : 1) * sizeof(struct gas_cell_data));
    int *send_home_idx = (int *) malloc((total_send > 0 ? total_send : 1) * sizeof(int));
    /* Two integer-only streams over the match bitmap instead of one that also
     * carries the payload copy.  Packing while streaming meant every particle_data
     * + gas_cell_data copy was interleaved with a walk over an NTask x num_pool
     * bitmap that is almost entirely zero, so the pack paid the sparse walk's
     * memory behaviour; the destination offsets also had to be tracked per rank
     * as the walk went.  Here the walk only records which pool slots match, and
     * the pack then runs over that dense list.
     * Send order is unchanged -- destination rank ascending, then pool index
     * ascending, each rank's run based at send_disp[t] -- so the packed buffers
     * are identical to what the interleaved walk produced. */
    int *send_pool_slot = (int *) malloc((total_send > 0 ? total_send : 1) * sizeof(int));
    if(!send_pool_slot) {
        printf("ERROR: request-driven ghost exchange send-slot list allocation failed on task %d.\n", ThisTask);
        gizmo_request_controlled_stop(7726, "ghost_exchange (request-driven): send-slot list allocation failed",
                                      __FILE__, __LINE__, __FUNCTION__);
    }
    gizmo_exit_bad_stop_if_requested("ghost_exchange:send_slot_alloc_rd");
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask) continue;
        char *match_for_t = matched + (size_t)t * (size_t)num_pool;
        int *dst = send_pool_slot + send_disp[t];
        int  k = 0;
        for(int p = 0; p < num_pool; p++) if(match_for_t[p]) dst[k++] = p;
    }
    for(int off = 0; off < total_send; off++) {
        int j = h_pool[send_pool_slot[off]];
        gx_pack_send_slot(P, CellP, j, &send_P[off], &send_CellP[off]);
        send_home_idx[off] = j;
    }
    free(send_pool_slot);

    /* === Step 6: Alltoallv particles + cells + home_idx === */
    gx_forward_particle_exchange(send_P, send_CellP, send_count, send_disp,
                                 &P[NumPart], &CellP[NumPart], recv_count, recv_disp);

    /* Update counts now so home_idx receive can land at &P[NumPart_before_ghost+...] */
    NumGhostParticles = total_recv;
    NumPart += total_recv;

    /* Mark dirty for compact_xyzh refresh (same as legacy). */
    if(NumGhostParticles > 0) {
        gpu_compact_xyzh_mark_h_dirty_range(NumPart_before_ghost, NumPart);
    }
    /* SIDX lifecycle notify: see comment in tile-overlap impl. Unconditional. */
    gpu_sidx_notify_ghost_imported(NumPart_before_ghost, NumGhostParticles);

    /* Home-index exchange + provenance maps. */
    int *recv_home_idx = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
    gizmo_mpi_alltoallv_typed(send_home_idx, send_count, send_disp,
                              recv_home_idx, recv_count, recv_disp,
                              sizeof(int), MPI_COMM_WORLD);
    ghost_home_rank_map = (int *) malloc((total_recv > 0 ? total_recv : 1) * sizeof(int));
    ghost_home_index_map = recv_home_idx;
    for(int t = 0; t < NTask; t++) {
        for(int g = 0; g < recv_count[t]; g++) {
            ghost_home_rank_map[recv_disp[t] + g] = t;
        }
    }
    /* Preserve comm maps for reverse Alltoallv (ghost writeback). */
    ghost_wb_recv_count = (int *) malloc(NTask * sizeof(int));
    ghost_wb_recv_disp  = (int *) malloc(NTask * sizeof(int));
    ghost_wb_send_count = (int *) malloc(NTask * sizeof(int));
    ghost_wb_send_disp  = (int *) malloc(NTask * sizeof(int));
    memcpy(ghost_wb_recv_count, recv_count, NTask * sizeof(int));
    memcpy(ghost_wb_recv_disp,  recv_disp,  NTask * sizeof(int));
    memcpy(ghost_wb_send_count, send_count, NTask * sizeof(int));
    memcpy(ghost_wb_send_disp,  send_disp,  NTask * sizeof(int));

    /* Preserve send-side provenance for ghost_refresh_values() (take ownership
       of send_home_idx; the free() below then no-ops on NULL). */
    if(ghost_send_home_idx) free(ghost_send_home_idx);
    ghost_send_home_idx   = send_home_idx;
    ghost_send_home_count = total_send;
    send_home_idx         = NULL;
    g_ghost_provenance_epoch++;

    double t_ghost_total = timediff(t_ghost_start, my_second());


    if(ThisTask == 0) {
        PRINT_STATUS("Ghost exchange (request-driven, %s, %s, qdist=%s): %d local + %d ghost  queries=%d total_queries=%d num_pool=%d  [%.4f s]",
                     (spec->caller_name ? spec->caller_name : "?"),
                     (search_mode == NGB_SEARCH_ONEWAY ? "ONEWAY" : "SYMMETRIC"),
                     (used_routed ? "routed" : "bcast"),
                     NumPart_before_ghost, NumGhostParticles,
                     n_local_queries, (used_routed ? -1 : total_queries), num_pool, t_ghost_total);
    }

    /* Diagnostic: ghost composition + import-waste ratio (should be ~0% for
     * the request-driven path by construction since per-particle accept ran
     * before pack — provides direct A/B vs the legacy tile-overlap waste). */

    /* Cleanup local. mymalloc requires LIFO free order. Tile/BVH/pool/
     * compact_xyzh/pool_types are now owned by g_glt_cache (malloc-backed)
     * and outlive this frame; do NOT free them here. They're freed at
     * cache invalidation (drift / domain_decomp hooks) via glt_cache_free. */
    myfree(send_CellP);
    myfree(send_P);
    myfree(recv_disp);
    myfree(send_disp);
    myfree(recv_count);
    myfree(send_count);
    free(send_home_idx);
    free(matched);
    free(all_queries); free(q_disps); free(all_q_counts); free(local_queries);
    (void)from_cache;
    return GHOST_EXCHANGE_COMPLETED;
}

/* Public wrappers — each fills a spec, calls the single _impl. New callers
 * add a wrapper line; do not duplicate logic.
 *
 * radius_policy + j_radius_scale on the spec are part of the SSOT supply-side
 * contract.  Legacy non-runner wrappers explicitly pass
 * MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES + 1.0 to preserve their pre-policy
 * behavior byte-for-byte (raw P[j].KernelRadius * safety_factor as the
 * supply-side reach).  Runner Mode A passes Spec::radius_policy +
 * nlr_spec_symmetric_j_radius_scale<Spec>() via gizmo_request_filtered_ghost_import_fresh
 * — see ghost_symlist_lifecycle.h. */
void ghost_exchange(double safety_factor)
{
    /* supply_band_dominated=1: this spec's reach is P[j].KernelRadius (the legacy
     * all-types policy) for every type, times safety_factor (checked <= 1 at
     * dispatch). The per-type opener band is seeded per particle from the
     * conservative source union capped at All.MaxKernelRadius (ForceSoftening
     * uncapped; force_hmax_per_type_particle_radius) and exchanged cross-rank on
     * the nodes the export walk descends — so under the standing invariant that
     * no P[].KernelRadius of any type exceeds All.MaxKernelRadius (drift clamps
     * in predict.cc; the density-convergence maxsoft clamp; the sink accretion-
     * radius cap in ags_rkern.cc), the band dominates this reach per type.
     * Any future physics path that writes a non-gas KernelRadius must preserve
     * that invariant or routed discovery under-imports. */
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_ALL, GHOST_TYPE_ALL, NGB_SEARCH_SYMMETRIC, safety_factor, "all_types", -1, NULL, NULL,
                                       MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES, 1.0, 1};
    ghost_exchange_impl(&sp);
}
void ghost_exchange_hydro(double safety_factor)
{
    /* supply_band_dominated=1: gas-only supply at the legacy all-types kernel
     * radius. Every KernelRadius is held at or below MaxKernelRadius (density
     * sink setup clamps it; ags_return_maxsoft clamps the drift path), which is
     * exactly the quantity the per-type node band is built from — so the band is
     * a valid upper bound on this spec's reach and routed discovery is complete.
     * safety_factor is checked separately at dispatch (a >1 factor widens the
     * query beyond the band until the opener scales with it). */
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_0, GHOST_TYPE_0, NGB_SEARCH_SYMMETRIC, safety_factor, "hydro_symmetric", -1, NULL, NULL,
                                       MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES, 1.0, 1};
    ghost_exchange_impl(&sp);
}
void ghost_exchange_hydro_oneway(double safety_factor)
{
    /* ONEWAY routes on search mode alone; the flag is unread here. */
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_0, GHOST_TYPE_0, NGB_SEARCH_ONEWAY, safety_factor, "hydro_oneway", -1, NULL, NULL,
                                       MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES, 1.0, 0};
    ghost_exchange_impl(&sp);
}


/*!
 * \brief Remove ghost particles after neighbor operations complete.
 *
 * Resets NumPart and N_gas to pre-exchange values. Must be called after
 * all neighbor loops (density, gradients, hydro force) that use ghosts.
 */
void ghost_exchange_cleanup(void)
{
    if(NumPart_before_ghost < 0) return;
    /* Ghost slots are about to leave scope (NumPart shrinks back to local).
     * No dirty-state scrubbing is done here. Marks are per-cache: one landing
     * outside a cache's registered index range is dropped at mark time, and a
     * cache that WAS registered over these ghost slots is freed -- handle
     * unregistered with it -- by the particle-count change on its next build.
     * Either way no stale ghost-slot bit reaches compact_h_refresh. New ghost
     * slots are marked dirty at import time (mark_h_dirty_range above), so
     * symmetric h-reads on ghosts stay fresh. */
    /* SIDX lifecycle notify BEFORE NumPart shrinks. Called whether or not
     * NumGhostParticles>0 — a cleanup from the no-ghost-imported state is
     * a valid signal that bumps the epoch. */
    gpu_sidx_notify_ghost_cleanup();
    if(NumGhostParticles > GhostEpochHighWater) {GhostEpochHighWater = NumGhostParticles;}
    NumPart = NumPart_before_ghost;
    N_gas = N_gas_before_ghost;
    NumGhostParticles = 0;
    NumPart_before_ghost = -1;
    /* Free ghost provenance map */
    if(ghost_home_rank_map)  { free(ghost_home_rank_map);  ghost_home_rank_map = NULL; }
    if(ghost_home_index_map) { free(ghost_home_index_map); ghost_home_index_map = NULL; }
    if(ghost_wb_recv_count)  { free(ghost_wb_recv_count);  ghost_wb_recv_count = NULL; }
    if(ghost_wb_recv_disp)   { free(ghost_wb_recv_disp);   ghost_wb_recv_disp = NULL; }
    if(ghost_wb_send_count)  { free(ghost_wb_send_count);  ghost_wb_send_count = NULL; }
    if(ghost_wb_send_disp)   { free(ghost_wb_send_disp);   ghost_wb_send_disp = NULL; }
    if(ghost_send_home_idx)  { free(ghost_send_home_idx);  ghost_send_home_idx = NULL; }
    ghost_send_home_count = 0;
    /* g_ghost_provenance_epoch is a monotonic stamp — NOT reset here. */
}

/* Make refreshed host ghost values visible to the device. With unified-memory
   particles (P/CellP in Kokkos SharedSpace) this is a no-op: a host write to a
   ghost slot is coherent to the next kernel, and compact_xyzh caches Pos+h only
   (unchanged by a value refresh). This helper is ALWAYS in the refresh call path
   (never elided) so a backend with explicit host/device particle buffers (no
   unified-memory coherence) has ONE mandatory place to add an explicit
   host->device ghost copy. Parity with import: import marks compact_xyzh h-dirty
   + notifies SIDX; a value refresh changes neither Pos/h nor the slot set, so it
   does neither — but any explicit device copy import gains MUST be mirrored here. */
static inline void ghost_refresh_make_device_visible(int ghost_base, int ghost_count)
{
    (void)ghost_base; (void)ghost_count;
}

int ghost_refresh_values(void)
{
    /* Fail-closed guards (production callers fall back to full cleanup+reimport).
       A non-NULL ghost_send_home_idx already implies an import happened with no
       cleanup since (cleanup NULLs it). */
    if(NTask <= 1)               return GHOST_REFRESH_SKIP_SERIAL;
    if(NumPart_before_ghost < 0) return GHOST_REFRESH_FAIL_NO_POOL;
    if(!ghost_send_home_idx || !ghost_wb_send_count || !ghost_wb_send_disp ||
       !ghost_wb_recv_count || !ghost_wb_recv_disp)
                                 return GHOST_REFRESH_FAIL_NO_PROVENANCE;
    /* Live-pool consistency: current pool counts must match the preserved
       provenance (topology unchanged; no intervening reimport with different
       totals). */
    long long send_tot = 0, recv_tot = 0;
    for(int t = 0; t < NTask; t++) { send_tot += ghost_wb_send_count[t]; recv_tot += ghost_wb_recv_count[t]; }
    if(NumPart != NumPart_before_ghost + NumGhostParticles) return GHOST_REFRESH_FAIL_POOL_MUTATED;
    if(recv_tot != (long long)NumGhostParticles)            return GHOST_REFRESH_FAIL_POOL_MUTATED;
    if(send_tot != (long long)ghost_send_home_count)        return GHOST_REFRESH_FAIL_POOL_MUTATED;

    int ns = ghost_send_home_count;
    /* new[], not malloc: particle_data is over-aligned (32 bytes), and the C
     * allocator has no type information so it cannot honour that. new[] selects
     * the aligned form automatically for an over-aligned type. */
    struct particle_data *send_P = new struct particle_data[(ns > 0 ? ns : 1)];
    struct gas_cell_data *send_CellP = new struct gas_cell_data[(ns > 0 ? ns : 1)];
    /* Re-pack current owner values via the SAME slot helper import uses, in the
       SAME send order (ghost_send_home_idx) that produced this pool. */
    for(int k = 0; k < ns; k++) {
        gx_pack_send_slot(P, CellP, ghost_send_home_idx[k], &send_P[k], &send_CellP[k]);
    }
    /* Replay ONLY the forward transport, overwriting the EXISTING ghost slots at
       [NumPart_before_ghost, NumPart). Slots land at identical offsets by
       construction, so ghost_home_rank/index maps + any built CSR stay valid. */
    gx_forward_particle_exchange(send_P, send_CellP, ghost_wb_send_count, ghost_wb_send_disp,
                                 &P[NumPart_before_ghost], &CellP[NumPart_before_ghost],
                                 ghost_wb_recv_count, ghost_wb_recv_disp);
    delete[] send_P; delete[] send_CellP;
    ghost_refresh_make_device_visible(NumPart_before_ghost, NumGhostParticles);
    return GHOST_REFRESH_OK;
}

/* Import-epoch accessor (see g_ghost_provenance_epoch): read by the hydro
   corridor, which fast-paths a value-refresh only when the live pool's epoch
   matches the one its published CSR was built from. */
unsigned long long ghost_provenance_epoch(void) { return g_ghost_provenance_epoch; }

/* Accessors for ghost provenance data — used by ghost_writeback.cc */
/* True iff a ghost import is live (pool materialized, between import and cleanup).
   Distinguishes "live pool with zero ghosts" from "no pool" — callers must not
   infer liveness from ghost_get_num_ghosts(), which returns 0 in both states.
   Used by the neighbor-loop runner to enforce the caller-owned-pool contract
   for external-CSR consumers (see neighbor_loop_runner.h). */
int ghost_pool_is_live(void) { return (NumPart_before_ghost >= 0) ? 1 : 0; }
int ghost_get_num_ghosts(void) { return NumGhostParticles; }
int ghost_get_epoch_high_water(void)
{
    return (GhostPreviousEpochHighWater > GhostEpochHighWater) ? GhostPreviousEpochHighWater
                                                               : GhostEpochHighWater;
}
void ghost_reset_epoch_high_water(void)
{
    GhostPreviousEpochHighWater = GhostEpochHighWater;
    GhostEpochHighWater = 0;
}
int ghost_get_num_local(void)  { return (NumPart_before_ghost >= 0) ? NumPart_before_ghost : NumPart; }
int *ghost_get_home_rank(void)  { return ghost_home_rank_map; }
int *ghost_get_home_index(void) { return ghost_home_index_map; }
int *ghost_get_wb_recv_count(void) { return ghost_wb_recv_count; }
int *ghost_get_wb_recv_disp(void)  { return ghost_wb_recv_disp; }
int *ghost_get_wb_send_count(void) { return ghost_wb_send_count; }
int *ghost_get_wb_send_disp(void)  { return ghost_wb_send_disp; }
