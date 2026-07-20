/*! \file let_pack.cc
 *  \brief Step 13 Phase 9.1c-e -- Locally Essential Tree (LET) pack, exchange,
 *         and unpack.  Replaces the iterative gravity export loop with a
 *         one-shot subtree exchange.
 *
 *  Flow per gravity_tree() invocation (after force_treebuild and
 *  force_exchange_pseudodata):
 *
 *    let_run_exchange()
 *      1. let_compute_local_payload  -- our bbox + worst-case opening bounds
 *      2. let_exchange_payloads       -- MPI_Allgather payloads to all ranks
 *      3. let_pack_for_rank(R) for each remote R
 *           recurse our local tree from each topnode; ship every node
 *           let_node_essential_for_rank() flags as "could be opened by some
 *           particle in R".  At leaves, synthesize single-particle NODEs
 *           covering all #ifdef payloads (mirrors force_update_node_recursive
 *           single-particle accumulation, forcetree.cc:752-861).  After a topleaf
 *           subtree is fully emitted, let_relabel_subtree() resolves every sibling
 *           continuation to a WIRE index or the single LET_WIRE_EXIT sentinel, so
 *           the shipped graph is self-contained (no sender indices in topology).
 *      4. let_exchange_nodes          -- MPI_Alltoall counts + MPI_Alltoallv payloads
 *      5. let_unpack_and_install      -- copy NODE+extNODE bytes into the foreign slot
 *           range [MaxPart+MaxNodes, MaxPart+MaxNodes+Numforeignnodes); rebase each
 *           wire index to slot_base+wire; map LET_WIRE_EXIT to the local topleaf's
 *           continuation; redirect each affected local topleaf's u.d.nextnode at the
 *           foreign subtree root.  No sender-index reconstruction on the receiver.
 *
 *  Buffer-overflow policy (Phase 9.0): if Numforeignnodes would exceed
 *  MaxForeignNodes, endrun() with the LETAllocFactor restart message.
 *  Future option (b) -- graceful shrink + temporary fallback to legacy export
 *  -- documented but not implemented unless practical memory limits require.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"            /* gizmo_verbose_diag */
#include "../system/gpu_particles_arena.h"  /* gpu_particles_arena_invalidate */
#include "../system/mpi_alltoallv_typed.h"   /* int-overflow-safe MPI_Alltoallv wrapper */
#include "../core/step_phases.h"              /* gizmo_verbose_diag() */
#include "let_data.h"
#include "gravtree_opening.h"   /* shared Stage-2 opening predicate (cell/AABB variant) */
#include "gravtree_moment_kernel.h"  /* shared node-moment CONSTRUCTION SSOT (add_particle/finalize) */
#include "gravtree_moment_sources.h" /* shared per-particle RT/sink/CR source-input gates */
#include "gpu_pseudo_update.h"  /* gpu_scatter_foreign_to_soa */


/* Temporary EMIT-time encoding for a "last child of a level" sibling whose continuation index
 * is not yet known (it ships later in walk order).  The last child stores
 * LET_EDGE_SENTINEL_ENCODE(sender_continuation); once the topleaf subtree is fully emitted,
 * let_relabel_subtree() resolves every encoded value to a WIRE index (via the per-subtree
 * sender->wire map, which includes synthesized particle leaves) or LET_WIRE_EXIT.  These encoded
 * values never reach the receiver -- the installed wire graph carries only wire indices and
 * LET_WIRE_EXIT.  Encoding is valid for non-negative sender indices only (negative GADGET
 * terminators map straight to LET_WIRE_EXIT at the encode sites).
 */
#define LET_EDGE_SENTINEL_BASE         (-1000000)
#define LET_EDGE_SENTINEL_ENCODE(orig) (LET_EDGE_SENTINEL_BASE - (orig))
/* `<=` (not `<`): ENCODE(0) == BASE must be detected as encoded.  The encode sites only ever
 * pass a NON-NEGATIVE orig (negative GADGET terminators are mapped straight to LET_WIRE_EXIT),
 * so every real encoded value is <= BASE; LET_WIRE_EXIT (-2) is far above BASE, no collision. */
#define LET_EDGE_SENTINEL_IS_ENCODED(v)  ((v) <= LET_EDGE_SENTINEL_BASE)
#define LET_EDGE_SENTINEL_DECODE(v)      (LET_EDGE_SENTINEL_BASE - (v))

/* Resolved subtree-exit marker on the wire (post pack-side RELABEL): a sibling/
 * nextnode equal to LET_WIRE_EXIT means "leave this foreign subtree", which the
 * receiver maps to the local topleaf's continuation (topleaf_sibling).  Distinct
 * from any valid wire index (>= 0) and from the encoded terminators (< BASE). */
#define LET_WIRE_EXIT  (-2)


/* ----------------------------------------------------------------------
 * Phase 9.5: active-only LET helpers
 *
 * Each rank computes a per-topleaf bitmap (one bit per topleaf, NTopleaves
 * total) where bit tl == 1 iff topleaf tl is owned by ThisTask AND at least
 * one particle in ActiveParticleList lives in tl.  Bitmaps are MPI_Allgathered
 * so every rank sees every other rank's active-topleaf set.  In the pack
 * loop, sender S short-circuits packing for receiver R when R's bitmap is
 * all zero (R has no active particles -> R does not need any LET data).
 *
 * Particle -> topleaf lookup uses Father[i] -> ancestor chain until the
 * first BITFLAG_TOPLEVEL node, which is necessarily a topleaf (only
 * topleaves contain particles; internal topnodes have only topnode children).
 * ---------------------------------------------------------------------- */
static inline int let_bitmap_word_count(int n_topleaves)
{
    return (n_topleaves + 63) / 64;
}

static inline int let_bitmap_test(const uint64_t *b, int tl)
{
    return (int) ((b[tl >> 6] >> (tl & 63)) & 1ULL);
}

static inline int let_bitmap_any_set(const uint64_t *b, int n_words)
{
    if(!b) return 1;  /* NULL bitmap -> conservative: assume some bits set */
    for(int w = 0; w < n_words; w++) if(b[w]) return 1;
    return 0;
}

extern "C" void let_compute_local_active_bitmap(uint64_t *bitmap, int n_words)
{
    memset(bitmap, 0, (size_t)n_words * sizeof(uint64_t));
    if(NTopleaves <= 0) return;

    /* Build inverse lookup: Nodes[no] -> topleaf index (or -1) for MY topleaves only. */
    int *my_tl_lookup = (int *) mymalloc("LET_my_tl_lookup", (size_t)MaxNodes * sizeof(int));
    for(int j = 0; j < MaxNodes; j++) my_tl_lookup[j] = -1;
    for(int t = 0; t < NTopleaves; t++)
    {
        if(DomainTask[t] != ThisTask) continue;
        int no = DomainNodeIndex[t];
        if(no >= All.MaxPart && no < All.MaxPart + MaxNodes)
            my_tl_lookup[no - All.MaxPart] = t;
    }

    /* Fallback: if ActiveParticleList is empty (e.g. tree rebuilt before
     * make_list_of_active_particles ran for the current step), set all MY
     * topleaves' bits.  Preserves Phase 9.4 conservative behavior. */
    if(ActiveParticleList.empty())
    {
        for(int t = 0; t < NTopleaves; t++)
        {
            if(DomainTask[t] != ThisTask) continue;
            bitmap[t >> 6] |= (1ULL << (t & 63));
        }
        myfree(my_tl_lookup);
        return;
    }

    for(size_t k = 0; k < ActiveParticleList.size(); k++)
    {
        int i = ActiveParticleList[k];
        if(i < 0 || i >= NumPart) continue;
        if(P[i].Mass <= 0) continue;
        int no = Father[i];
        int guard = 0;
        while(no >= All.MaxPart && no < All.MaxPart + MaxNodes && guard++ < 1024)
        {
            int tl = my_tl_lookup[no - All.MaxPart];
            if(tl >= 0)
            {
                bitmap[tl >> 6] |= (1ULL << (tl & 63));
                break;
            }
            no = Nodes[no].u.d.father;
        }
    }
    myfree(my_tl_lookup);
}

/* ----------------------------------------------------------------------
 * Per-topleaf scalar table: global [NTopleaves], each rank fills its OWNED slice
 * (let_compute_local_payload) and it is Allgatherv'd (mirroring the DomainMoment
 * exchange) so every sender holds every receiver-topleaf's scalars. Grown-or-reused
 * across builds (process-lifetime scratch, like the cover-tree g_cover* arrays).
 * UNBUCKETABLE guard: a local particle whose Father chain never reaches an owned topleaf
 * loses its scalars from every leaf summary (a real under-import hazard) -> LOUD count +
 * collective controlled stop in let_run_exchange (never a silent fold-into-every-leaf downgrade).
 * ---------------------------------------------------------------------- */
static struct LETTopleafScalars *g_topleaf_scalars = NULL;
static int g_topleaf_scalars_cap = 0;
static long long g_let_unbucketable = 0;          /* local count this build */
static long long g_let_unbucketable_first_id = -1;/* first offending P[i].ID (diagnostic) */

static void let_topleaf_scalars_ensure(int n)
{
    if(g_topleaf_scalars_cap >= n) return;
    struct LETTopleafScalars *nt = (struct LETTopleafScalars *)
        realloc(g_topleaf_scalars, (size_t) n * sizeof(struct LETTopleafScalars));
    if(!nt) { printf("let_topleaf_scalars_ensure: realloc failed (n=%d, rank=%d). Stopping.\n",
                     n, ThisTask); fflush(stdout); endrun(90000093); }
    g_topleaf_scalars = nt; g_topleaf_scalars_cap = n;
}

/* ----------------------------------------------------------------------
 * Drift-orphan receiver-cover records (see struct LETOrphanRecord).
 *   g_my_orphans      -- THIS rank's records for the current build (producer side).
 *   g_orphan_all      -- every rank's records, concatenated by rank (post-Allgatherv).
 *   g_orphan_off      -- prefix-sum offsets: rank R's records = g_orphan_all[off[R], off[R+1]).
 * Process-lifetime scratch (like g_topleaf_scalars / the g_cover* arrays); grow-only. */
static struct LETOrphanRecord *g_my_orphans = NULL;
static int g_my_orphans_n = 0, g_my_orphans_cap = 0;
static struct LETOrphanRecord *g_orphan_all = NULL;
static int g_orphan_all_cap = 0;
static int *g_orphan_off = NULL;
static int g_orphan_off_cap = 0;

/* Merge one local orphan target's opening scalars into an orphan record (same worst-case ops as the
 * per-topleaf table: min OldAcc, max soft-by-type, min soft, OR has_sink). */
static inline void let_orphan_merge(struct LETOrphanRecord *r, double oa, double soft, int ptype)
{
    if(oa > 0 && oa < r->s.min_OldAcc) r->s.min_OldAcc = oa;
    if(soft > r->s.max_soft_by_type[ptype]) r->s.max_soft_by_type[ptype] = soft;
    if(soft < r->s.min_soft) r->s.min_soft = soft;
    if(ptype == 5) r->s.has_sink = 1;
}

/* Find (or append) THIS rank's orphan record for reached foreign topleaf t, initialised empty.
 * Orphans are rare, so a compact list with linear-search dedup keeps the blast radius minimal (no
 * NTopleaves-sized scratch); if orphan counts ever explode that is itself a perf signal to report. */
static struct LETOrphanRecord *let_my_orphan_for_topleaf(int t)
{
    for(int k = 0; k < g_my_orphans_n; k++) if(g_my_orphans[k].topleaf == t) return &g_my_orphans[k];
    if(g_my_orphans_n >= g_my_orphans_cap) {
        int nc = g_my_orphans_cap ? 2 * g_my_orphans_cap : 8;
        struct LETOrphanRecord *nb = (struct LETOrphanRecord *)
            realloc(g_my_orphans, (size_t) nc * sizeof(struct LETOrphanRecord));
        if(!nb) { printf("let_my_orphan_for_topleaf: realloc failed (nc=%d, rank=%d). Stopping.\n",
                         nc, ThisTask); fflush(stdout); endrun(90000095); }
        g_my_orphans = nb; g_my_orphans_cap = nc;
    }
    struct LETOrphanRecord *r = &g_my_orphans[g_my_orphans_n++];
    r->topleaf = t; r->_pad = 0;
    r->s.min_OldAcc = DBL_MAX; for(int k = 0; k < 6; k++) r->s.max_soft_by_type[k] = 0.0;
    r->s.min_soft = DBL_MAX; r->s.has_sink = 0; r->s._pad = 0;
    return r;
}

/* ----------------------------------------------------------------------
 * Step 1: per-rank-payload computation
 * ---------------------------------------------------------------------- */
extern "C" void let_compute_local_payload(struct LETPerRankPayload *out,
                                          const uint64_t *active_bitmap,
                                          int bitmap_n_words)
{
    /* bbox: union of OUR topleaf bboxes (each topleaf's [center-len/2, center+len/2]).
     * NOTE: as of the per-topleaf cover fix the bbox COORDS no longer feed LET
     * essentiality (the pack walks the receiver's per-topleaf cover TREE, built
     * sender-side from replicated topleaf geometry, not this single union box);
     * only has_cover (>=1 owned topleaf) is still consumed. The coords are kept
     * for wire-format compatibility + debug. min_OldAcc/soft/sink below ARE still
     * used (whole-rank worst-case scalars fed to the same predicate). If
     * active_bitmap is non-NULL, restrict to ACTIVE topleaves (Phase 9.5). */
    out->bbox_min[0] = out->bbox_min[1] = out->bbox_min[2] = DBL_MAX;
    out->bbox_max[0] = out->bbox_max[1] = out->bbox_max[2] = -DBL_MAX;
    int found_any = 0;
    for(int i = 0; i < NTopleaves; i++)
    {
        if(DomainTask[i] != ThisTask) continue;
        if(active_bitmap && !let_bitmap_test(active_bitmap, i)) continue;
        int no = DomainNodeIndex[i];
        if(no < All.MaxPart || no >= All.MaxPart + MaxNodes) continue;
        double cx = (double) Nodes[no].center[0];
        double cy = (double) Nodes[no].center[1];
        double cz = (double) Nodes[no].center[2];
        double half = 0.5 * (double) Nodes[no].len;
        if(cx - half < out->bbox_min[0]) out->bbox_min[0] = cx - half;
        if(cy - half < out->bbox_min[1]) out->bbox_min[1] = cy - half;
        if(cz - half < out->bbox_min[2]) out->bbox_min[2] = cz - half;
        if(cx + half > out->bbox_max[0]) out->bbox_max[0] = cx + half;
        if(cy + half > out->bbox_max[1]) out->bbox_max[1] = cy + half;
        if(cz + half > out->bbox_max[2]) out->bbox_max[2] = cz + half;
        found_any = 1;
    }
    if(!found_any)
    {
        /* No local topleaves -> no real cover.  Set a degenerate origin bbox; the
         * has_cover flag below marks this rank so senders ship it nothing (the
         * no-real-receiver guard in let_pack_for_rank).  Without that flag the
         * degenerate cover + min_OldAcc==0 would make the relative criterion open
         * every node and over-ship the whole tree to an empty rank. */
        for(int k = 0; k < 3; k++) {out->bbox_min[k] = out->bbox_max[k] = 0.0;}
    }
    out->has_cover = found_any;   /* >=1 owned topleaf -> a real cover exists */

    /* Per-particle bounds -> PER-TOPLEAF scalar table (owned slice) + WHOLE-RANK reduce.
     * Each local particle is bucketed to its owning topleaf via the Father chain; the topleaf's
     * scalars are the worst case over ITS particles (target-local, tighter), and the whole-rank
     * payload scalars below are the reduce over all owned topleaves -- a conservative fallback that
     * can never drift from the table (SSOT). active_bitmap non-NULL -> ActiveParticleList (Phase 9.5
     * tight); else all NumPart (Phase 9.4 conservative -- safer for RT/TREECOL non-active). */
    out->min_OldAcc = DBL_MAX;
    for(int t = 0; t < 6; t++) out->max_soft_by_type[t] = 0.0;
    out->min_soft = DBL_MAX;
    out->has_sink = 0;

    let_topleaf_scalars_ensure(NTopleaves > 0 ? NTopleaves : 1);
    g_let_unbucketable = 0; g_let_unbucketable_first_id = -1;

    /* Node[no] -> topleaf index, over ALL topleaves (owned AND foreign). Ownership is a property
     * checked AFTER lookup (DomainTask[t]==ThisTask), not baked into the map -- one SSOT lookup.
     * The owner also inits its OWN topleaf-scalar slice to conservative-empty here (EMPTY sentinel =
     * min_soft==DBL_MAX; ForceSoftening_KernelRadius>0 for any real particle so a non-empty leaf sets it). */
    int *tl_lookup = (int *) mymalloc("LET_tl_lookup_scalars", (size_t) MaxNodes * sizeof(int));
    for(int j = 0; j < MaxNodes; j++) tl_lookup[j] = -1;
    for(int t = 0; t < NTopleaves; t++)
    {
        int no = DomainNodeIndex[t];
        if(no >= All.MaxPart && no < All.MaxPart + MaxNodes) tl_lookup[no - All.MaxPart] = t;
        if(DomainTask[t] != ThisTask) continue;
        struct LETTopleafScalars *s = &g_topleaf_scalars[t];
        s->min_OldAcc = DBL_MAX; for(int k = 0; k < 6; k++) s->max_soft_by_type[k] = 0.0;
        s->min_soft = DBL_MAX; s->has_sink = 0; s->_pad = 0;
    }
    g_my_orphans_n = 0;   /* drift-orphan records rebuilt each payload compute */

    int n_iter = (active_bitmap && !ActiveParticleList.empty()) ? (int) ActiveParticleList.size() : NumPart;
    for(int kk = 0; kk < n_iter; kk++)
    {
        int i = (active_bitmap && !ActiveParticleList.empty()) ? ActiveParticleList[kk] : kk;
        if(i < 0 || i >= NumPart) continue;
        if(P[i].Mass <= 0) continue;
        int t = P[i].Type;
        if(t < 0 || t > 5) continue;

        /* Map particle -> the topleaf whose box contains it, via the Father chain. Top-tree leaves do
         * not nest, so the chain passes through EXACTLY ONE topleaf (the containing one) before climbing
         * internal top-tree nodes to the root -- the first topleaf reached IS that topleaf. */
        int tl = -1, no = Father[i], guard = 0;
        while(no >= All.MaxPart && no < All.MaxPart + MaxNodes && guard++ < 1024)
        {
            int cand = tl_lookup[no - All.MaxPart];
            if(cand >= 0) { tl = cand; break; }
            no = Nodes[no].u.d.father;
        }
        if(tl < 0)
        {
            /* No topleaf on the Father chain at all -> genuine tree-topology corruption (NOT mere
             * drift): the target has no geometry in ANY cover. Count for the collective controlled
             * stop in let_run_exchange. NEVER a silent fold-into-every-leaf downgrade. */
            if(g_let_unbucketable == 0) g_let_unbucketable_first_id = (long long) P[i].ID;
            g_let_unbucketable++;
            continue;
        }

        double oa = (double) P[i].OldAcc;
        double soft = (double) ForceSoftening_KernelRadius(i);

        if(DomainTask[tl] != ThisTask)
        {
            /* DRIFT-ORPHAN: this LOCAL/resident target drifted into a topleaf owned by another rank
             * (current tree geometry only -- NOT a change of which rank owns the target). Its scalars
             * ride an orphan record so senders open nodes essential to it against topleaf tl's box;
             * otherwise it is dropped from this rank's owned-topleaf cover (the pre-fix under-import
             * hazard, exposed under ADAPTIVE_TREEFORCE_UPDATE tree reuse). */
            let_orphan_merge(let_my_orphan_for_topleaf(tl), oa, soft, t);
        }
        else
        {
            struct LETTopleafScalars *s = &g_topleaf_scalars[tl];
            /* per-topleaf (target-local worst case) */
            if(oa > 0 && oa < s->min_OldAcc) s->min_OldAcc = oa;
            if(soft > s->max_soft_by_type[t]) s->max_soft_by_type[t] = soft;
            if(soft < s->min_soft) s->min_soft = soft;
            if(t == 5) s->has_sink = 1;
        }
        /* whole-rank reduce (derived fallback) -- over ALL local targets incl. orphans, so the
         * empty-owned-topleaf fallback baked in below stays a true worst case. */
        if(oa > 0 && oa < out->min_OldAcc) out->min_OldAcc = oa;
        if(soft > out->max_soft_by_type[t]) out->max_soft_by_type[t] = soft;
        if(soft < out->min_soft) out->min_soft = soft;
        if(t == 5) out->has_sink = 1;
    }
    myfree(tl_lookup);

    /* Resolve orphan-record sentinels to conservative bounds (same convention as the owned slice
     * below): min_OldAcc==DBL_MAX -> 0 (maximally-open relaccel), min_soft==DBL_MAX -> 0. A record
     * exists only if >=1 orphan target set min_soft, so min_soft!=DBL_MAX in practice; guard anyway. */
    for(int k = 0; k < g_my_orphans_n; k++)
    {
        struct LETOrphanRecord *r = &g_my_orphans[k];
        if(r->s.min_OldAcc == DBL_MAX) r->s.min_OldAcc = 0.0;
        if(r->s.min_soft   == DBL_MAX) r->s.min_soft   = 0.0;
    }
    out->n_orphans = g_my_orphans_n;

    /* Resolve whole-rank sentinels. NOTE: min_OldAcc==0 does NOT disable the relative criterion --
     * it makes t_aold=0, i.e. the relaccel test is MAXIMALLY OPEN (conservative). First-step/BH logic
     * is what actually avoids the branch; here we only preserve the conservative-zero semantics. */
    if(out->min_OldAcc == DBL_MAX) out->min_OldAcc = 0.0;  /* no positive OldAcc; conservative-zero (maximally open) */
    if(out->min_soft   == DBL_MAX) out->min_soft   = 0.0;  /* empty cover; conservative (opens softening) */

    /* Finalize MY owned topleaf slice: EMPTY leaves (min_soft still DBL_MAX) -> whole-rank fallback
     * (conservative, no worse than today, preserves drift forgiveness for a particle that drifts in);
     * non-empty leaves -> resolve the per-leaf min_OldAcc==DBL_MAX sentinel to conservative-zero. */
    for(int t = 0; t < NTopleaves; t++)
    {
        if(DomainTask[t] != ThisTask) continue;
        struct LETTopleafScalars *s = &g_topleaf_scalars[t];
        if(s->min_soft == DBL_MAX)
        {
            s->min_OldAcc = out->min_OldAcc;
            for(int k = 0; k < 6; k++) s->max_soft_by_type[k] = out->max_soft_by_type[k];
            s->min_soft = out->min_soft;
            s->has_sink = out->has_sink;
        }
        else if(s->min_OldAcc == DBL_MAX) { s->min_OldAcc = 0.0; }  /* has particles, none with positive OldAcc */
    }
    /* The relative-criterion activation is not shipped: the cell predicate recomputes it sender-side
     * from All.ErrTolTheta + the first-step test (both global, identical on every rank). */
}

/* ----------------------------------------------------------------------
 * Step 2: payload exchange (Allgather)
 * ---------------------------------------------------------------------- */
extern "C" int let_exchange_payloads(const struct LETPerRankPayload *local,
                                      struct LETPerRankPayload *all_ranks)
{
    return MPI_Allgather(local, sizeof(struct LETPerRankPayload), MPI_BYTE,
                         all_ranks, sizeof(struct LETPerRankPayload), MPI_BYTE,
                         MPI_COMM_WORLD);
}

/* ============================================================================
 * Per-receiver COVER TREE. The receiver R's owned-topleaf geometric boxes
 * ([center - len/2, center + len/2]) arranged as a balanced binary AABB tree, so
 * LET essentiality tests each source node against a HIERARCHY of compact per-
 * topleaf covers instead of one whole-rank union box. The single union box spans
 * ~the whole domain for a spatially-spread rank, driving min_dist(node,cover)->0,
 * which defeats the PM cutoff + theta cull and imports ~the whole global tree;
 * per-topleaf covers restore that selectivity. Built sender-side each exchange
 * from REPLICATED top-tree geometry (DomainTask/DomainNodeIndex/Nodes topnodes)
 * + R's existing payload scalars -- no wire-format change. The opening predicate
 * (gravtree_open_decision_cell) is the untouched SSOT; this only refines WHICH
 * cover boxes are fed to it. Scratch is grown once and reused across receivers/
 * exchanges (no allocation in the pack recursion). The tree for "the receiver
 * currently being packed" is file-scope state (g_cover*), set by let_pack_for_rank
 * before its pack loop -- same idiom as g_let_pack_oom.
 * ========================================================================== */
/* Cover-tree node carries BOTH geometry (box) AND the conservative per-cover scalar bounds,
 * combined up the tree from the per-topleaf table (min OldAcc / max soft-by-type / min soft /
 * OR has_sink -- the easiest-to-open value over the subtree, so the aggregate-prune stays
 * conservative per field, identical monotonicity to the geometry proof). let_cover_opens feeds
 * these node-local bounds to the untouched predicate instead of the rank-wide payload scalars. */
struct LETCoverNode {
    double bmin[3], bmax[3];
    int c0, c1;                     /* c0<0 => leaf topleaf */
    double min_OldAcc;              /* relaccel: min over subtree */
    double max_soft_by_type[6];     /* relsoft:  max per type over subtree */
    double min_soft;                /* node-softening open: min over subtree */
    int    has_sink;                /* sink-direct: OR over subtree */
};
static struct LETCoverNode *g_cover      = NULL;  /* AABB-tree nodes, root at index 0 */
static int                   g_cover_cap  = 0;
static int                   g_cover_n    = 0;     /* nodes used by the current build */
static int                  *g_cover_leaf = NULL;  /* R's owned-topleaf Nodes[] indices */
static int                  *g_cover_leaf_tl = NULL;/* parallel: topleaf index (geometry provenance/debug) */
static struct LETTopleafScalars *g_cover_leaf_scal = NULL;/* parallel: this leaf's opening scalars BY VALUE --
                                              owned leaf <- g_topleaf_scalars[t]; orphan leaf <- its record.
                                              Self-contained so a cover leaf's scalars are decoupled from the
                                              per-topleaf table (an orphan leaf's geometry is a foreign topleaf). */
static int                   g_cover_leaf_cap = 0;

/* Union box of the leaf range [lo,hi) into out[bmin/bmax]. */
static void cover_union_box(int lo, int hi, double bmin[3], double bmax[3])
{
    bmin[0]=bmin[1]=bmin[2]= DBL_MAX;
    bmax[0]=bmax[1]=bmax[2]=-DBL_MAX;
    for(int k = lo; k < hi; k++) {
        int no = g_cover_leaf[k];
        double h = 0.5 * (double) Nodes[no].len;
        for(int d = 0; d < 3; d++) {
            double c = (double) Nodes[no].center[d];
            if(c - h < bmin[d]) bmin[d] = c - h;
            if(c + h > bmax[d]) bmax[d] = c + h;
        }
    }
}

/* BVH build over the leaf range [lo,hi): split at the SPATIAL MEDIAN of the
 * range's longest axis (partition g_cover_leaf in place by leaf-center along that
 * axis), so aggregate boxes are tight and the walk prunes early. Falls back to the
 * index median for a degenerate partition (coincident centers). g_cover is pre-
 * sized to 2*NTopleaves so no realloc moves g_cover[idx] during the recursion. */
static int cover_build(int lo, int hi)
{
    int idx = g_cover_n++;
    double *bmin = g_cover[idx].bmin, *bmax = g_cover[idx].bmax;
    cover_union_box(lo, hi, bmin, bmax);
    if(hi - lo <= 1) {
        g_cover[idx].c0 = -1; g_cover[idx].c1 = -1;
        /* leaf: this cover leaf's opening scalars (owned topleaf's table entry, or an orphan record) */
        const struct LETTopleafScalars *s = &g_cover_leaf_scal[lo];
        g_cover[idx].min_OldAcc = s->min_OldAcc;
        for(int t = 0; t < 6; t++) g_cover[idx].max_soft_by_type[t] = s->max_soft_by_type[t];
        g_cover[idx].min_soft = s->min_soft;
        g_cover[idx].has_sink = s->has_sink;
        return idx;
    }

    int ax = 0; double ext = bmax[0] - bmin[0];
    if(bmax[1] - bmin[1] > ext) { ax = 1; ext = bmax[1] - bmin[1]; }
    if(bmax[2] - bmin[2] > ext) { ax = 2; ext = bmax[2] - bmin[2]; }
    double split = 0.5 * (bmin[ax] + bmax[ax]);
    int i = lo, j = hi - 1;
    while(i <= j) {
        while(i <= j && (double) Nodes[g_cover_leaf[i]].center[ax] <  split) i++;
        while(i <= j && (double) Nodes[g_cover_leaf[j]].center[ax] >= split) j--;
        if(i < j) { int t = g_cover_leaf[i]; g_cover_leaf[i] = g_cover_leaf[j]; g_cover_leaf[j] = t;
                    int u = g_cover_leaf_tl[i]; g_cover_leaf_tl[i] = g_cover_leaf_tl[j]; g_cover_leaf_tl[j] = u;
                    struct LETTopleafScalars sw = g_cover_leaf_scal[i]; g_cover_leaf_scal[i] = g_cover_leaf_scal[j]; g_cover_leaf_scal[j] = sw;
                    i++; j--; }
    }
    int mid = i;
    if(mid <= lo || mid >= hi) mid = (lo + hi) / 2;   /* degenerate split -> index median */

    int l = cover_build(lo, mid);
    int r = cover_build(mid, hi);
    g_cover[idx].c0 = l; g_cover[idx].c1 = r;
    /* internal node: combine children conservatively (easiest-to-open per field) */
    g_cover[idx].min_OldAcc = (g_cover[l].min_OldAcc < g_cover[r].min_OldAcc) ? g_cover[l].min_OldAcc : g_cover[r].min_OldAcc;
    for(int t = 0; t < 6; t++) {
        double a = g_cover[l].max_soft_by_type[t], b = g_cover[r].max_soft_by_type[t];
        g_cover[idx].max_soft_by_type[t] = (a > b) ? a : b;
    }
    g_cover[idx].min_soft = (g_cover[l].min_soft < g_cover[r].min_soft) ? g_cover[l].min_soft : g_cover[r].min_soft;
    g_cover[idx].has_sink = g_cover[l].has_sink | g_cover[r].has_sink;
    return idx;
}

/* Build receiver R's cover tree into the scratch (g_cover* / root = index 0).
 * Leaves g_cover_n = 0 if R owns no topleaves (caller's has_cover guard already
 * excludes that case). realloc failure -> loud controlled stop, never a segfault. */
static void let_build_cover_tree(int R, const uint64_t *receiver_active_bitmap)
{
    /* receiver_active_bitmap is NULL on the default all-local path (all of R's topleaves);
     * under LET_ACTIVE_RECEIVER_COVER_EXPERIMENTAL it restricts the cover to R's ACTIVE
     * topleaves, matching the active-restricted per-topleaf scalar table computed under the
     * same guard (so cover geometry and cover scalars stay consistent -- no partial feature). */
    g_cover_n = 0;
    /* Cover leaves = R's owned topleaves + R's drift-orphan records (extra leaves at foreign topleaf
     * boxes). Size for both; orphans are rare so cap grows to NTopleaves + R's orphan count. */
    int n_orph_R = (g_orphan_off ? g_orphan_off[R + 1] - g_orphan_off[R] : 0);
    int cap_need = NTopleaves + n_orph_R;
    if(g_cover_leaf_cap < cap_need) {
        int *nl = (int *) realloc(g_cover_leaf, (size_t) cap_need * sizeof(int));
        int *nt = (int *) realloc(g_cover_leaf_tl, (size_t) cap_need * sizeof(int));
        struct LETTopleafScalars *ns = (struct LETTopleafScalars *) realloc(g_cover_leaf_scal, (size_t) cap_need * sizeof(struct LETTopleafScalars));
        if(!nl || !nt || !ns) { printf("let_build_cover_tree: cover-leaf realloc failed (cap_need=%d, rank=%d). Stopping.\n",
                         cap_need, ThisTask); fflush(stdout); endrun(90000091); }
        g_cover_leaf = nl; g_cover_leaf_tl = nt; g_cover_leaf_scal = ns; g_cover_leaf_cap = cap_need;
    }
    int nleaf = 0;
    for(int i = 0; i < NTopleaves; i++) {
        if(DomainTask[i] != R) continue;
        if(receiver_active_bitmap && !let_bitmap_test(receiver_active_bitmap, i)) continue;  /* active-cover guard */
        int no = DomainNodeIndex[i];
        if(no < All.MaxPart || no >= All.MaxPart + MaxNodes) continue;
        g_cover_leaf[nleaf] = no;
        g_cover_leaf_tl[nleaf] = i;                 /* geometry provenance (debug) */
        g_cover_leaf_scal[nleaf] = g_topleaf_scalars[i];   /* owned leaf: table entry, by value */
        nleaf++;
    }
    /* Drift-orphan extra leaves for R: geometry = the reached foreign topleaf's box (replicated top-tree
     * geometry, valid on every rank), scalars = the merged orphan record. Added unconditionally (a
     * conservative completeness extension -- never drops coverage, so safe under the active-cover mode). */
    for(int k = g_orphan_off ? g_orphan_off[R] : 0; k < (g_orphan_off ? g_orphan_off[R + 1] : 0); k++) {
        int t  = g_orphan_all[k].topleaf;
        int no = (t >= 0 && t < NTopleaves) ? DomainNodeIndex[t] : -1;
        if(no < All.MaxPart || no >= All.MaxPart + MaxNodes) continue;
        g_cover_leaf[nleaf] = no;
        g_cover_leaf_tl[nleaf] = t;
        g_cover_leaf_scal[nleaf] = g_orphan_all[k].s;
        nleaf++;
    }
    if(nleaf == 0) return;
    int need = 2 * nleaf;                  /* balanced tree over nleaf leaves has <= 2*nleaf-1 nodes */
    if(g_cover_cap < need) {
        struct LETCoverNode *nc = (struct LETCoverNode *) realloc(g_cover, (size_t) need * sizeof(struct LETCoverNode));
        if(!nc) { printf("let_build_cover_tree: cover-node realloc failed (need=%d, rank=%d). Stopping.\n",
                         need, ThisTask); fflush(stdout); endrun(90000092); }
        g_cover = nc; g_cover_cap = need;
    }
    cover_build(0, nleaf);                 /* root = node 0 */
}

/* Does ANY of R's per-topleaf covers OPEN this source node? Walk the cover tree
 * with the SAME predicate; prune a subtree whose AGGREGATE box does not open
 * (conservativeness: child box subset of aggregate => min_dist(node,child) >=
 * min_dist(node,aggregate); every distance-decreasing open test that fires for a
 * child fires for the aggregate, and the PM/theta culls only get looser at the
 * larger aggregate distance => aggregate-not-OPEN => no child OPENs). Short-
 * circuits at the first LEAF topleaf that opens. Iterative (balanced tree depth
 * <= ~log2(NTopleaves) < 40; the 64-slot stack cannot overflow for a balanced
 * build, and the guarded conservative return keeps it correct if it ever could). */
static int let_cover_opens(double cx, double cy, double cz,
                           double sx, double sy, double sz,
                           double len, double mass, double maxsoft, int node_nsink,
                           double rcut, double rcut2, int is_first_step)
{
    if(g_cover_n <= 0) return 0;
    int stack[64]; int sp = 0; stack[sp++] = 0;   /* root */
    while(sp > 0) {
        int ci = stack[--sp];
        const struct LETCoverNode *cn = &g_cover[ci];
        /* per-cover (target-local) scalar bounds -> the untouched predicate; the source node's own
         * maxsoft/n_sink stay the caller's node fields (msoft/n_sink), the cover carries the TARGET side. */
        double t_soft_max = 0.0;
        for(int t = 0; t < 6; t++) if(cn->max_soft_by_type[t] > t_soft_max) t_soft_max = cn->max_soft_by_type[t];
        double t_aold_min = cn->min_OldAcc * All.ErrTolForceAcc;
        gravtree_open_t d = gravtree_open_decision_cell(cx, cy, cz, sx, sy, sz, len, mass, maxsoft,
            node_nsink, cn->bmin, cn->bmax,
            t_soft_max, cn->min_soft, t_aold_min, cn->has_sink, rcut, rcut2, is_first_step);
        if(d != GRAV_OPEN_NODE) continue;         /* aggregate doesn't open -> prune this subtree */
        if(cn->c0 < 0) return 1;                  /* a real topleaf opens -> essential */
        if(sp + 2 <= 64) { stack[sp++] = cn->c0; stack[sp++] = cn->c1; }
        else return 1;                            /* depth guard (unreachable for balanced tree): conservative */
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Essential-node check (worst-case opening criterion for "any particle in R")
 *
 * Returns 1 if SOME particle in R might open this node (must ship + recurse).
 * Returns 0 if NO particle in R will open this node (ship as multipole-only OR
 * skip).
 *
 * Routes the decision through the single shared opening predicate
 * (gravtree_open_decision_cell) -- the SAME acceptance geometry the GPU/CPU
 * walk uses, evaluated over R's per-topleaf cover TREE.  essential == (the walk
 * would OPEN).  Each cover node carries BOTH geometry (box) and the TARGET-LOCAL
 * worst-case scalars (min OldAcc / max soft-by-type / min soft / has_sink,
 * combined up from the exchanged per-topleaf table) -- so the cell opens whenever
 * ANY target in that cover would, but no longer inflated to the rank-wide worst
 * case.  The source node's own maxsoft + n_sink stay caller-supplied node fields.
 * ---------------------------------------------------------------------- */
static int let_node_essential_for_rank(double cx, double cy, double cz,
                                       double sx, double sy, double sz,
                                       double len, double mass, double maxsoft,
                                       int n_sink)
{
    double rcut = 0.0, rcut2 = 0.0;
#ifdef PMGRID
    rcut = (double) All.Rcut[0];
#ifdef PM_PLACEHIGHRESREGION
    if((double) All.Rcut[1] > rcut) rcut = (double) All.Rcut[1];
#endif
    rcut2 = rcut * rcut;
#endif

    /* Same first-step value the walk uses (hybrid opening: relative suppressed on step 0). */
    int is_first_step = (All.Ti_Current == 0 && RestartFlag != 1) ? 1 : 0;

    /* Essential iff ANY of R's per-topleaf covers opens this node.  Cover geometry AND
     * per-cover scalar bounds refine the whole-rank worst case (which spanned ~the whole
     * domain / rank-wide scalar extrema and so never let the PM/theta/relative cull prune);
     * the opening predicate itself is untouched. */
    return let_cover_opens(cx, cy, cz, sx, sy, sz, len, mass, maxsoft, n_sink,
                           rcut, rcut2, is_first_step);
}

/* ----------------------------------------------------------------------
 * Single-particle leaf NODE/extNODE synthesis.
 *
 * Builds a LET wire node for ONE particle by routing it through the shared
 * node-moment construction kernel (gravtree_moment_kernel.h) -- the SAME
 * physics body the live GPU node-moment venues use -- instead of a hand-
 * inlined finalized copy.  A moment_node_ref points at the wire's payload
 * storage; the zeroed wire (memset below) is the fresh accumulator, one
 * particle is added, and moment_finalize normalizes in place.  RT/sink/CR
 * source-input gates come from the shared host helper
 * gravtree_fill_particle_source_inputs (gravtree_moment_sources.h).
 *
 * Sets BITFLAG_MULTIPLEPARTICLES (forced after finalize) so the receiver's
 * walk uses the synthesized multipole directly (exact for a single particle)
 * instead of skipping to a non-existent source-side particle.
 *
 * Two payloads differ from the previous hand-inlined form, both for a
 * single-particle leaf and both matching the live GPU venues + the CPU
 * anchor (so this is a correctness alignment, not dead-field noise):
 *  - rt_source_lum_s: a sink leaf with sink_lum>0 but no stellar luminosity
 *    now gets the geometric-center (== particle position) fallback from
 *    moment_finalize, not 0.  This feeds grav_sink_fb_angleweight under
 *    RT_SEPARATELY_TRACK_LUMPOS + SINK_PHOTONMOMENTUM at np>=2 -- a real
 *    single-particle correctness fix.
 *  - sink_lum_grad: a gated-in sink whose sink_lum_bol returns exactly 0 now
 *    gets finalize's {0,0,1} fallback rather than the raw angle vector.  The
 *    walk filters on sink_lum>0, so this field is never read in that case.
 *
 * Edge pointers (sibling/nextnode) start as the terminator sentinel; the
 * surrounding pack_recurse caller updates them based on the position of
 * this leaf in the iteration chain.
 * ---------------------------------------------------------------------- */
static void let_synthesize_particle_leaf(int p_idx, int sib_terminator_sentinel,
                                          struct LETNodeWire *w)
{
    memset(w, 0, sizeof(struct LETNodeWire));

    w->remote_id = -1 - p_idx;  /* negative encoding distinguishes synthesized particle leaves
                                  * from real internal nodes (whose remote_id >= MaxPart).  Unpack
                                  * uses remote_id < 0 to recognize synthesized leaves -- they
                                  * still get a foreign slot and remap entry, but their pointer
                                  * fields are simpler (no inbound references except from parent). */

    struct particle_data *pa = &P[p_idx];
    Vec3<MyFloat> pos = {(MyFloat) pa->Pos[0], (MyFloat) pa->Pos[1], (MyFloat) pa->Pos[2]};

    /* Build the per-particle source POD for the moment kernel.  Geometry/bounds
     * are venue-owned; RT/sink/CR source inputs come from the shared host gate
     * helper (the same gates the GPU precompute venues apply). */
    moment_particle_src<MyFloat> src = {};
    src.mass              = (double) pa->Mass;
    src.pos[0] = (double) pa->Pos[0]; src.pos[1] = (double) pa->Pos[1]; src.pos[2] = (double) pa->Pos[2];
    src.vel[0] = (double) pa->Vel[0]; src.vel[1] = (double) pa->Vel[1]; src.vel[2] = (double) pa->Vel[2];
    src.type              = pa->Type;
    src.kernel_radius     = (double) pa->KernelRadius;
    src.max_kernel_radius = (double) All.MaxKernelRadius;
    src.force_softening   = ForceSoftening_KernelRadius(p_idx);
    src.particle_divvel   = (double) pa->Particle_DivVel;
#if defined(SINK_ALPHADISK_ACCRETION) && defined(RT_USE_TREECOL_FOR_NH)
    src.sink_mass_reservoir = (double) pa->Sink_Mass_Reservoir;
#endif
#if defined(SPECIAL_POINT_MOTION)
    src.acc_prevstep[0] = (double) pa->Acc_Total_PrevStep[0];
    src.acc_prevstep[1] = (double) pa->Acc_Total_PrevStep[1];
    src.acc_prevstep[2] = (double) pa->Acc_Total_PrevStep[2];
#endif
#if defined(SINK_CALC_DISTANCES) && defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    src.max_feedback_vel = (double) pa->MaxFeedbackVel;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    for(int k = 0; k < 6; k++) { src.tidal_prevstep[k] = (double) pa->tidal_tensorps_prevstep.data[k]; }
#endif
    {
        struct gravtree_source_inputs_t in;
        gravtree_fill_particle_source_inputs(p_idx, P, CellP, &in);
#ifdef RT_USE_GRAVTREE
        if(in.rt_active) {
            for(int k = 0; k < N_RT_FREQ_BINS; k++) { src.src_lum[k] = (double) in.src_lum[k]; }
#ifdef CHIMES_STELLAR_FLUXES
            for(int k = 0; k < CHIMES_LOCAL_UV_NBINS; k++) {
                src.src_lum_G0[k]  = in.src_lum_G0[k];
                src.src_lum_ion[k] = in.src_lum_ion[k];
            }
#endif
        }
#endif
#ifdef SINK_PHOTONMOMENTUM
        if(in.bh_active) {
            src.bh_lum      = (double) in.bh_lum;
            src.bh_angle[0] = (double) in.bh_angle[0];
            src.bh_angle[1] = (double) in.bh_angle[1];
            src.bh_angle[2] = (double) in.bh_angle[2];
        }
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        src.cr_inject = (double) in.cr_inject;
#endif
    }

    /* Point a moment_node_ref at the wire's payload storage and run the shared
     * add_particle + finalize (zero == the memset above for a fresh leaf). */
    moment_node_ref<MyFloat> ref = {};
    ref.mass     = &w->node.u.d.mass;
    ref.s        = &w->node.u.d.s;
    ref.vs       = &w->extnode.vs;
    ref.Npart    = &w->node.N_part;
    ref.hmax     = &w->extnode.hmax;
    ref.vmax     = &w->extnode.vmax;
    ref.divVmax  = &w->extnode.divVmax;
    ref.maxsoft  = &w->node.maxsoft;
    ref.bitflags = NULL;   /* multiple-particles bit forced below, after finalize */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    ref.gasmass  = &w->node.gasmass;
#endif
#ifdef RT_USE_GRAVTREE
    ref.stellar_lum = &w->node.stellar_lum[0];
#ifdef CHIMES_STELLAR_FLUXES
    ref.chimes_G0  = &w->node.chimes_stellar_lum_G0[0];
    ref.chimes_ion = &w->node.chimes_stellar_lum_ion[0];
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    ref.rt_s  = &w->node.rt_source_lum_s;
    ref.rt_vs = &w->extnode.rt_source_lum_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
    ref.sink_lum      = &w->node.sink_lum;
    ref.sink_lum_grad = &w->node.sink_lum_grad;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    ref.cr_inject = &w->node.cr_injection;
#endif
#ifdef SINK_CALC_DISTANCES
    ref.sink_mass = &w->node.sink_mass;
    ref.sink_pos  = &w->node.sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    ref.N_SINK   = &w->node.N_SINK;
    ref.sink_vel = &w->node.sink_vel;
#endif
#if defined(SPECIAL_POINT_MOTION)
    ref.sink_acc = &w->node.sink_acc;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    ref.max_fbvel = &w->node.MaxFeedbackVel;
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    ref.tidal = &w->node.tidal_tensorps_prevstep.data[0];
#endif
#ifdef DM_SCALARFIELD_SCREENING
    ref.mass_dm = &w->node.mass_dm;
    ref.s_dm    = &w->node.s_dm;
    ref.vs_dm   = &w->extnode.vs_dm;
#endif

    moment_accum_add_particle<moment_plain_ops, MyFloat>(ref, src);
    moment_finalize<MyFloat>(ref, pos);

    /* wire topology + the forced multiple-particles bit (finalize left bitflags
     * untouched since ref.bitflags was null). */
    w->node.center       = pos;
    w->node.len          = 0;   /* zero size -- never opens, always treated as multipole */
    w->node.u.d.bitflags = (1u << BITFLAG_MULTIPLEPARTICLES);
    w->node.u.d.sibling  = sib_terminator_sentinel;
    w->node.u.d.nextnode = sib_terminator_sentinel;
    w->node.u.d.father   = -1;  /* foreign nodes have no father in OUR tree */
    w->node.GravCost     = 0;
    w->node.Ti_current   = All.Ti_Current;
    w->node.N_part       = 1;
    w->extnode.Ti_lastkicked = All.Ti_Current;
    w->extnode.Flag = 0;

    /* C1 foreign-leaf identity sidecar: this singleton ships as a synthesized terminal multipole,
     * but the receiver must consume it with particle-leaf secondary-source semantics.  The node
     * moment already carries every other source field (and for a singleton equals the particle
     * value); the two it cannot carry are Type and AGS_zeta.  Mirror the local-leaf AGS guards
     * EXACTLY: Type always, zeta only when the adaptive-softening field exists, else 0. */
    w->leaf_tag      = 1;
    w->leaf_type     = pa->Type;
    w->leaf_force_softening = (MyFloat) src.force_softening;  /* pure ForceSoftening, NOT the node's
                                                              * conflated maxsoft=max(soft,kernel) */
    w->leaf_ags_zeta = 0;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
    if(pa->Type == 0) { w->leaf_ags_zeta = (MyFloat) pa->AGS_zeta; }
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
    w->leaf_ags_zeta = (MyFloat) pa->AGS_zeta;
#endif
}

/* ----------------------------------------------------------------------
 * Step 3: pack -- recursive walk producing LETNodeWire array for one rank
 *
 * Buffer growth: amortized doubling.  Caller passes (buf, count, capacity);
 * pack functions grow buf in place via realloc.
 * ---------------------------------------------------------------------- */
/* Set when a LET pack-buffer realloc fails. Reset at the top of
 * let_run_exchange. On failure the packer bails (zeroing the failed rank's
 * payload, no partial subtrees) and let_run_exchange returns nonzero so the
 * caller soft-stops; the Alltoallv still runs (zero counts for failed ranks)
 * before the drain, keeping the MPI choreography intact. */
static int g_let_pack_oom = 0;

/* Foreign-leaf identity sidecar arrays (declared in let_data.h).  Allocated/freed with the
 * foreign-node arena in force_treeallocate/force_treefree; memset-0 reset in let_run_exchange. */
int     *ForeignLeafTag  = NULL;
int     *ForeignLeafType = NULL;
MyFloat *ForeignLeafZeta = NULL;
MyFloat *ForeignLeafSoft = NULL;

static void grow_wire_buf(struct LETNodeWire **buf, int needed, int *capacity)
{
    if(needed <= *capacity) return;
    int new_cap = (*capacity == 0) ? 1024 : *capacity;
    while(new_cap < needed) new_cap *= 2;
    struct LETNodeWire *nb = (struct LETNodeWire *) realloc(*buf, (size_t)new_cap * sizeof(struct LETNodeWire));
    if(!nb)
    {
        printf("LET pack: realloc failed (cap=%d, sizeof=%zu, total=%g MB)\n",
               new_cap, sizeof(struct LETNodeWire),
               (double)new_cap * sizeof(struct LETNodeWire) / (1024.0*1024.0));
        g_let_pack_oom = 1;   /* leave *buf/*capacity unchanged; caller bails before any OOB write */
        return;
    }
    *buf = nb;
    *capacity = new_cap;
}

static void grow_hdr_buf(struct LETSubtreeHeader **buf, int needed, int *capacity)
{
    if(needed <= *capacity) return;
    int new_cap = (*capacity == 0) ? 16 : *capacity;
    while(new_cap < needed) new_cap *= 2;
    struct LETSubtreeHeader *nb = (struct LETSubtreeHeader *) realloc(*buf, (size_t)new_cap * sizeof(struct LETSubtreeHeader));
    if(!nb)
    {
        printf("LET pack: hdr realloc failed (cap=%d)\n", new_cap);
        g_let_pack_oom = 1;   /* leave *buf/*capacity unchanged; caller bails before any OOB write */
        return;
    }
    *buf = nb;
    *capacity = new_cap;
}

static void pack_recurse(int no, int sib_terminator,
                          int subtree_root_topleaf_no,  /* unused; kept for future per-subtree edge encoding */
                          struct LETNodeWire **buf, int *count, int *capacity)
{
    /* Bounds: must be a local internal node */
    if(no < All.MaxPart || no >= All.MaxPart + MaxNodes) return;

    /* Check essential-for-R BEFORE shipping.  If not essential, the walk will
     * close at the parent; we don't need to ship this node either (parent's
     * multipole already covers it). */
    double cx = (double) Nodes[no].center[0];
    double cy = (double) Nodes[no].center[1];
    double cz = (double) Nodes[no].center[2];
    double sx = (double) Nodes[no].u.d.s[0];
    double sy = (double) Nodes[no].u.d.s[1];
    double sz = (double) Nodes[no].u.d.s[2];
    double len = (double) Nodes[no].len;
    double mass = (double) Nodes[no].u.d.mass;
    double maxsoft = (double) Nodes[no].maxsoft;
    int node_nsink = 0;
#if (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES)) && defined(SINGLE_STAR_DIRECT_GRAVITY_RADIUS)
    node_nsink = Nodes[no].N_SINK;
#endif
    int is_essential = let_node_essential_for_rank(cx, cy, cz, sx, sy, sz, len, mass, maxsoft, node_nsink);

    /* Always ship the node (parent expects it).  If not essential, ship as
     * multipole-only (no recursion).  If essential, ship + recurse to children. */
    grow_wire_buf(buf, *count + 1, capacity);
    if(g_let_pack_oom) return;   /* realloc failed: bail before the OOB write */
    int my_idx = (*count)++;
    struct LETNodeWire *w = &(*buf)[my_idx];
    w->remote_id = no;
    w->leaf_tag = 0; w->leaf_type = 0; w->leaf_ags_zeta = 0; w->leaf_force_softening = 0; w->_pad1 = 0;  /* default: node */
    w->node = Nodes[no];     /* full struct copy including all #ifdef payloads */
    w->extnode = Extnodes[no];
    /* Edge pointers default to the subtree-exit marker; EMIT overwrites sibling with a
     * wire index (or an encoded terminator resolved by RELABEL) and nextnode with the
     * first shipped child's wire index when this node is opened-into. */
    w->node.u.d.sibling = LET_WIRE_EXIT;
    w->node.u.d.nextnode = LET_WIRE_EXIT;
    w->node.u.d.father = -1;  /* foreign nodes have no local father */

    if(!is_essential)
    {
        /* Multipole-only: receiver's walk will close on this node (their criterion
         * matches our worst-case bound), so they never descend.  No recursion. */
        return;
    }

    /* Single-particle leaf in source tree: bitflag=0 means walk would skip
     * to nextnode (a particle).  We can't ship the particle, so override
     * bitflag to MULTIPLEPARTICLES so receiver uses our multipole.  Exact
     * for single particle. */
    if(!(Nodes[no].u.d.bitflags & (1u << BITFLAG_MULTIPLEPARTICLES)))
    {
        /* C1: this single-particle source-tree node is shipped as a multipole, so the receiver
         * must consume it as a real foreign leaf.  Recover the underlying particle from the
         * ORIGINAL Nodes[no] (its nextnode is that particle; w->node is a copy whose nextnode was
         * already overwritten to LET_WIRE_EXIT) and tag the wire with the same leaf identity as the
         * synthesized-leaf path.  Hard-check the child is a real particle (p < MaxPart); if not, the
         * tree is malformed -- leave the record untagged so the receiver's predicate-keyed guard
         * surfaces it rather than mis-routing a non-particle as a leaf. */
        int p = Nodes[no].u.d.nextnode;
        if(p >= 0 && p < All.MaxPart)
        {
            struct particle_data *pa = &P[p];
            w->leaf_tag      = 1;
            w->leaf_type     = pa->Type;
            w->leaf_force_softening = (MyFloat) ForceSoftening_KernelRadius(p);
            w->leaf_ags_zeta = 0;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
            if(pa->Type == 0) { w->leaf_ags_zeta = (MyFloat) pa->AGS_zeta; }
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
            w->leaf_ags_zeta = (MyFloat) pa->AGS_zeta;
#endif
        }
        else
        {
            printf("LET pack: single-particle node %d has non-particle nextnode %d (rank %d); "
                   "shipping untagged (receiver guard will surface it).\n", no, p, ThisTask);
            fflush(stdout);
        }
        w->node.u.d.bitflags |= (1u << BITFLAG_MULTIPLEPARTICLES);
        return;  /* no children to recurse into */
    }

    /* Multi-particle internal node: enumerate children via nextnode/sibling chain.
     * For each child:
     *   - particle (< MaxPart): synthesize a leaf wire
     *   - internal node: recurse
     *   - pseudo (>= MaxPart+MaxNodes+MaxForeignNodes): skip (R has its own access via S->R LET pack)
     *   - foreign (in [MaxPart+MaxNodes, +MaxForeignNodes)): shouldn't appear during pack (we run before unpack)
     *
     * After processing all children, link them into a sibling chain in the WIRE buffer
     * by setting our nextnode to first-child wire-idx, and each child's sibling to
     * next-child wire-idx (last child's sibling = sentinel = "exit subtree"). */
    int first_child_wire_idx = -1;
    int last_child_wire_idx = -1;
    int child = Nodes[no].u.d.nextnode;
    while(child != sib_terminator && child >= 0)
    {
        int next_child;
        int child_wire_idx = -1;

        if(child < All.MaxPart)
        {
            /* Particle leaf -- synthesize */
            grow_wire_buf(buf, *count + 1, capacity);
            if(g_let_pack_oom) return;   /* realloc failed: bail before the OOB write */
            child_wire_idx = (*count)++;
            let_synthesize_particle_leaf(child, LET_WIRE_EXIT, &(*buf)[child_wire_idx]);
            next_child = Nextnode[child];  /* particle's next walk target */
        }
        else if(child < All.MaxPart + MaxNodes)
        {
            /* Local internal node -- recurse */
            int child_sib = Nodes[child].u.d.sibling;
            child_wire_idx = *count;
            pack_recurse(child, child_sib, subtree_root_topleaf_no, buf, count, capacity);
            if(g_let_pack_oom) return;   /* recursion hit a realloc OOM: bail */
            /* If pack_recurse added zero entries (skipped), child_wire_idx == old count;
             * we need to detect that and not link. */
            if(*count == child_wire_idx) child_wire_idx = -1;  /* nothing added */
            next_child = child_sib;
        }
        else if(child < All.MaxPart + MaxNodes + MaxForeignNodes)
        {
            /* Foreign node -- shouldn't happen during pack */
            next_child = Nodes[child].u.d.sibling;
            child_wire_idx = -1;
        }
        else
        {
            /* Pseudo-particle -- skip */
            next_child = Nextnode[child - MaxNodes - MaxForeignNodes];
            child_wire_idx = -1;
        }

        /* Link this child into the sibling chain */
        if(child_wire_idx >= 0)
        {
            if(first_child_wire_idx < 0) first_child_wire_idx = child_wire_idx;
            if(last_child_wire_idx >= 0)
            {
                /* Update prior-last child's sibling to point to this child's wire idx
                 * (we'll convert wire idx -> remote_id in a moment) */
                (*buf)[last_child_wire_idx].node.u.d.sibling = child_wire_idx;
            }
            last_child_wire_idx = child_wire_idx;
        }

        child = next_child;
    }

    /* Set our nextnode to first child (or sentinel if no children shipped) */
    /* (re-fetch &(*buf)[my_idx] -- realloc inside grow_wire_buf may have moved) */
    if(first_child_wire_idx >= 0)
    {
        (*buf)[my_idx].node.u.d.nextnode = first_child_wire_idx;
        if(last_child_wire_idx >= 0)
        {
            /* Last child's sibling is the level's continuation, resolved to a wire index by
             * RELABEL.  A negative sender terminator (GADGET "end of walk") is a true subtree
             * exit -> LET_WIRE_EXIT directly (the encoding is only valid for non-negative
             * indices: ENCODE(-1) would alias above BASE and escape IS_ENCODED). */
            (*buf)[last_child_wire_idx].node.u.d.sibling =
                (sib_terminator < 0) ? LET_WIRE_EXIT : LET_EDGE_SENTINEL_ENCODE(sib_terminator);
        }
    }
    /* else: no children shipped (e.g., all were particles outside essential range);
     *       nextnode stays LET_WIRE_EXIT; receiver will treat as multipole-leaf. */
}

/* ----------------------------------------------------------------------
 * Wire-space terminator resolution (pack side).
 *
 * EMIT links consecutive shipped children by their wire index directly; the
 * only deferred links are the "last child of a level" sibling terminators,
 * emitted as LET_EDGE_SENTINEL_ENCODE(sender_continuation) because the
 * continuation's wire index is not known yet.  Once a topleaf subtree is fully
 * emitted, RELABEL resolves every encoded terminator to a WIRE index (via a
 * per-subtree sender->wire map that INCLUDES synthesized particle leaves, keyed
 * by particle index) or the single LET_WIRE_EXIT sentinel.  The receiver then
 * never reconstructs sender topology: install rebases wire -> slot_base+wire and
 * maps LET_WIRE_EXIT -> the local topleaf's continuation.
 * ---------------------------------------------------------------------- */
struct let_kw { int key; int wire; };
static int let_kw_cmp(const void *a, const void *b)
{
    int ka = ((const struct let_kw *)a)->key, kb = ((const struct let_kw *)b)->key;
    return (ka > kb) - (ka < kb);
}

/* Resolve a sender continuation index `x` to a subtree-local wire index or
 * LET_WIRE_EXIT, mirroring the pack walk's skip of unshipped pseudo/foreign
 * nodes.  A missing in-subtree internal/particle continuation is a hard error
 * (loud + graceful stop): silently skipping it is exactly the class of graph
 * truncation this rewrite removes.  map is sorted by key over [lo_w,hi_w). */
static int let_resolve_continuation(int x, int topleaf_term,
                                    const struct let_kw *map, int map_n, int lo_w, int hi_w)
{
    int bound = (hi_w - lo_w) + NTopleaves + 16, guard = 0;
    while(++guard < bound)
    {
        if(x < 0 || x == topleaf_term) return LET_WIRE_EXIT;          /* the true subtree exit */
        struct let_kw probe; probe.key = x;
        const struct let_kw *hit = (const struct let_kw *)
            bsearch(&probe, map, map_n, sizeof(struct let_kw), let_kw_cmp);
        if(hit) return hit->wire;                                     /* shipped -> wire (incl synth leaf) */
        /* x is a non-shipped continuation; classify by EXPLICIT range (this is topology-repair
         * code -- an unclassifiable positive index must abort, never index Nodes[] blindly). */
        if(x >= 0 && x < All.MaxPart)
        {
            printf("LET pack FATAL: particle %d referenced as a sibling continuation but not shipped "
                   "and not the subtree terminator (rank %d).\n", x, ThisTask); fflush(stdout); endrun(90000072);
            return LET_WIRE_EXIT;
        }
        else if(x >= All.MaxPart && x < All.MaxPart + MaxNodes)
        {
            printf("LET pack FATAL: in-subtree internal node %d not shipped but referenced as a "
                   "sibling continuation (rank %d).\n", x, ThisTask); fflush(stdout); endrun(90000071);
            return LET_WIRE_EXIT;
        }
        else if(x >= All.MaxPart + MaxNodes && x < All.MaxPart + MaxNodes + MaxForeignNodes)
            x = Nodes[x].u.d.sibling;                                 /* foreign: skip as the walk does */
        else if(x >= All.MaxPart + MaxNodes + MaxForeignNodes)
            x = Nextnode[x - MaxNodes - MaxForeignNodes];             /* pseudo: skip as the walk does */
        else
        {
            printf("LET pack FATAL: unclassifiable continuation index %d (rank %d).\n",
                   x, ThisTask); fflush(stdout); endrun(90000076);
            return LET_WIRE_EXIT;
        }
    }
    printf("LET pack FATAL: resolve_continuation guard exceeded (rank %d).\n", ThisTask);
    fflush(stdout); endrun(90000073);
    return LET_WIRE_EXIT;
}

/* RELABEL a just-emitted topleaf subtree [lo_w,hi_w): build the sender->wire map
 * and resolve every encoded sibling terminator to a wire index / LET_WIRE_EXIT.
 * map_scratch is caller-owned, sized >= (hi_w-lo_w). */
static void let_relabel_subtree(struct LETNodeWire *buf, int lo_w, int hi_w,
                                int topleaf_term, struct let_kw *map_scratch)
{
    int n = hi_w - lo_w;
    for(int i = 0; i < n; i++)
    {
        int rid = buf[lo_w + i].remote_id;
        map_scratch[i].key  = (rid < 0) ? (-1 - rid) : rid;   /* synth leaf -> particle index; node -> node index */
        map_scratch[i].wire = lo_w + i;
    }
    qsort(map_scratch, n, sizeof(struct let_kw), let_kw_cmp);
    for(int w = lo_w; w < hi_w; w++)
    {
        int sib = buf[w].node.u.d.sibling;
        if(LET_EDGE_SENTINEL_IS_ENCODED(sib))
            buf[w].node.u.d.sibling =
                let_resolve_continuation(LET_EDGE_SENTINEL_DECODE(sib), topleaf_term, map_scratch, n, lo_w, hi_w);
        /* nextnode is never encoded: it is either a plain first-child wire index or LET_WIRE_EXIT. */
    }
}

extern "C" int let_pack_for_rank(int R,
                                  const struct LETPerRankPayload *all_ranks,
                                  struct LETNodeWire **out_buf,
                                  int *out_capacity,
                                  struct LETSubtreeHeader **out_hdr_buf,
                                  int *out_hdr_capacity,
                                  int *out_hdr_count,
                                  const uint64_t *receiver_active_bitmap,
                                  int bitmap_n_words)
{
    int count = 0;
    int hdr_count = 0;
    /* No-real-receiver guard: ship nothing to a receiver with no local cover at all (no owned
     * topleaves -> degenerate origin bbox).  Inactive particles still count; only a rank with
     * zero local cover is skipped.  Without this the all-local cover would over-open the whole
     * tree for an empty rank (its min_OldAcc==0 makes the relative criterion open everything). */
    if(!all_ranks[R].has_cover)
    {
        *out_hdr_count = 0;
        return 0;
    }
    /* Phase 9.5 (experimental active-receiver-cover mode only): short-circuit when receiver R
     * has zero active particles.  receiver_active_bitmap is NULL on the default all-local path. */
    if(receiver_active_bitmap && !let_bitmap_any_set(receiver_active_bitmap, bitmap_n_words))
    {
        *out_hdr_count = 0;
        return 0;
    }
    /* Build R's per-topleaf cover tree (file-scope g_cover*, consumed by
     * let_node_essential_for_rank during this receiver's pack). Cheap: one pass
     * over the replicated topleaf list + a balanced build; scratch reused. */
    let_build_cover_tree(R, receiver_active_bitmap);
    /* Walk the LOCAL tree from each topnode that's NOT in R's domain.  Topnodes
     * in R's domain are R's own data; they won't help R (and would create a
     * self-reference if shipped). */
    /* Iterate via DomainNodeIndex over OUR topleaves, then use the topleaf's
     * subtree (root via nextnode / sibling chain) as the pack starting point.
     * Actually we need to ship from ROOT downward filtering by essential, since
     * R needs to traverse from the root to find what to multipole-vs-open. */

    /* Strategy: walk our local topnode tree from root (Nodes[All.MaxPart]).  For
     * each topleaf owned by US (so R doesn't already have it as pseudo), we
     * enter pack_recurse from that topleaf with sib_terminator = topleaf.sibling. */

    /* NOTE: simpler -- pack each of our local topleaves' subtrees independently.
     * R will see each shipped subtree as the foreign-content for that topleaf
     * (the unpack step rewrites Nodes[topleaf_in_R].u.d.nextnode = subtree_root). */
    for(int i = 0; i < NTopleaves; i++)
    {
        if(DomainTask[i] != ThisTask) continue;
        int topleaf_no = DomainNodeIndex[i];
        if(topleaf_no < All.MaxPart || topleaf_no >= All.MaxPart + MaxNodes) continue;

        int subtree_root = Nodes[topleaf_no].u.d.nextnode;
        if(subtree_root < 0) continue;  /* empty topleaf */
        if(subtree_root == Nodes[topleaf_no].u.d.sibling) continue;  /* topleaf has no descendants */

        int sib_term = Nodes[topleaf_no].u.d.sibling;

        /* Record wire offset BEFORE this topleaf's pack, so we can emit a
         * subtree header covering [wire_offset, *count) on the way out. */
        int wire_offset_before = count;

        /* Walk the topleaf's children via the sibling chain starting from subtree_root.
         * Track first/last wire indices so we can link consecutive children's sibling
         * pointers — without this, the receiver's walk would follow child1.sibling =
         * LET_EDGE_SENTINEL_BASE and exit after child1, missing child2..childN. */
        int first_child_wire_idx = -1;
        int last_child_wire_idx  = -1;
        int child = subtree_root;
        while(child != sib_term && child >= 0)
        {
            int next_child;
            int child_wire_idx = -1;
            if(child < All.MaxPart)
            {
                /* Particle directly under topleaf -- synthesize leaf */
                grow_wire_buf(out_buf, count + 1, out_capacity);
                if(g_let_pack_oom) goto pack_oom_bail;   /* realloc failed: ship nothing for R */
                child_wire_idx = count;
                let_synthesize_particle_leaf(child, LET_WIRE_EXIT, &(*out_buf)[count]);
                count++;
                next_child = Nextnode[child];
            }
            else if(child < All.MaxPart + MaxNodes)
            {
                int child_sib = Nodes[child].u.d.sibling;
                child_wire_idx = count;
                pack_recurse(child, child_sib, topleaf_no, out_buf, &count, out_capacity);
                if(g_let_pack_oom) goto pack_oom_bail;   /* recursion hit a realloc OOM: ship nothing for R */
                if(count == child_wire_idx) child_wire_idx = -1;  /* pack_recurse added nothing */
                next_child = child_sib;
            }
            else
            {
                next_child = Nextnode[child - MaxNodes - MaxForeignNodes];
            }
            /* Link this child into the top-level sibling chain */
            if(child_wire_idx >= 0)
            {
                if(first_child_wire_idx < 0) first_child_wire_idx = child_wire_idx;
                if(last_child_wire_idx >= 0)
                    (*out_buf)[last_child_wire_idx].node.u.d.sibling = child_wire_idx;
                last_child_wire_idx = child_wire_idx;
            }
            child = next_child;
        }
        /* Last top-level child: continuation = sib_term (= topleaf.sibling), resolved to a wire
         * index / LET_WIRE_EXIT by RELABEL.  Negative terminator -> LET_WIRE_EXIT directly. */
        if(last_child_wire_idx >= 0)
            (*out_buf)[last_child_wire_idx].node.u.d.sibling =
                (sib_term < 0) ? LET_WIRE_EXIT : LET_EDGE_SENTINEL_ENCODE(sib_term);

        /* Emit subtree header if this topleaf shipped any nodes */
        int subtree_count = count - wire_offset_before;
        if(subtree_count > 0)
        {
            /* RELABEL: the subtree is fully emitted, so resolve every encoded sibling
             * terminator to a wire index (incl. synth-leaf particles) or LET_WIRE_EXIT.
             * After this, the wire graph is self-contained: install only rebases indices. */
            struct let_kw *map_scratch = (struct let_kw *) malloc((size_t)subtree_count * sizeof(struct let_kw));
            if(!map_scratch) { g_let_pack_oom = 1; goto pack_oom_bail; }
            let_relabel_subtree(*out_buf, wire_offset_before, count, sib_term, map_scratch);
            free(map_scratch);

            grow_hdr_buf(out_hdr_buf, hdr_count + 1, out_hdr_capacity);
            if(g_let_pack_oom) goto pack_oom_bail;   /* realloc failed: ship nothing for R */
            (*out_hdr_buf)[hdr_count].topleaf_idx = i;
            (*out_hdr_buf)[hdr_count].wire_offset = wire_offset_before;
            (*out_hdr_buf)[hdr_count].count       = subtree_count;
            (*out_hdr_buf)[hdr_count]._pad0       = 0;
            hdr_count++;
        }
    }
    *out_hdr_count = hdr_count;
    return count;

pack_oom_bail:
    /* realloc failed mid-pack: ship a zero payload for this rank (not a partial
     * subtree). let_run_exchange returns nonzero and the run drains after the
     * pseudodata exchange completes. */
    *out_hdr_count = 0;
    return 0;
}

/* ----------------------------------------------------------------------
 * Step 4 + 5: MPI exchange + install in one scope.
 *
 * GIZMO mymalloc is a strict LIFO stack.  An earlier draft returned
 * flat_recv / flat_hdr_recv to the caller while freeing intermediates
 * in this function -- that left the recv buffers mid-stack and
 * triggered "Wrong call of myfree(): not the last allocated block!"
 * the moment any caller tried to free anything below them.  Solution:
 * keep the unpack call inside this scope, so all temporaries can be
 * released in strict reverse-alloc order before returning.
 * ---------------------------------------------------------------------- */
extern "C" let_exchange_status_t let_exchange_nodes(struct LETNodeWire **send_buf_per_rank,
                                   const int *send_count_per_rank,
                                   struct LETSubtreeHeader **send_hdr_per_rank,
                                   const int *send_hdr_count_per_rank,
                                   long long *foreign_needed_out)
{
    double t_let_start = my_second();
    /* Phase 1: exchange node-counts AND header-counts */
    int *send_counts_int = (int *) mymalloc("LET_send_counts",     NTask * sizeof(int));
    int *recv_counts_int = (int *) mymalloc("LET_recv_counts",     NTask * sizeof(int));
    int *send_hdr_counts = (int *) mymalloc("LET_send_hdr_counts", NTask * sizeof(int));
    int *recv_hdr_counts = (int *) mymalloc("LET_recv_hdr_counts", NTask * sizeof(int));
    for(int r = 0; r < NTask; r++) {
        send_counts_int[r] = send_count_per_rank[r];
        send_hdr_counts[r] = send_hdr_count_per_rank[r];
    }
    MPI_Alltoall(send_counts_int, 1, MPI_INT, recv_counts_int, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(send_hdr_counts, 1, MPI_INT, recv_hdr_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_send = 0, total_recv = 0;
    int total_hdr_send = 0, total_hdr_recv = 0;
    for(int r = 0; r < NTask; r++) {
        total_send += send_counts_int[r]; total_recv += recv_counts_int[r];
        total_hdr_send += send_hdr_counts[r]; total_hdr_recv += recv_hdr_counts[r];
    }

    /* Allocate offsets for both exchanges (in units of struct elements, NOT
     * bytes), then flat_send / flat_recv for both data streams. All
     * temporaries are freed in strict reverse order at the end of this
     * function.
     *
     * Element-count units (instead of MPI_BYTE) are required for scaling:
     * MPI_Alltoallv's count/displ arguments are int*, so byte-based counts
     * overflow once any per-peer payload exceeds 2.1 GB. fire_m11i at 12.4M
     * particles on 2 ranks already trips this (~2.4 GB per peer at
     * sizeof(LETNodeWire)~400 B). Using a contiguous MPI_Datatype for the
     * struct moves the int limit from 2.1 GB to 2.1 G *elements*, i.e.
     * sizeof(struct)*2.1G bytes — effectively unbounded for any realistic
     * problem. */
    int *send_offsets     = (int *) mymalloc("LET_send_offsets",     NTask * sizeof(int));
    int *recv_offsets     = (int *) mymalloc("LET_recv_offsets",     NTask * sizeof(int));
    int *send_hdr_offsets = (int *) mymalloc("LET_send_hdr_offsets", NTask * sizeof(int));
    int *recv_hdr_offsets = (int *) mymalloc("LET_recv_hdr_offsets", NTask * sizeof(int));

    int s_off = 0, r_off = 0, hs_off = 0, hr_off = 0;
    for(int r = 0; r < NTask; r++)
    {
        send_offsets[r]     = s_off;
        recv_offsets[r]     = r_off;
        send_hdr_offsets[r] = hs_off;
        recv_hdr_offsets[r] = hr_off;
        s_off  += send_counts_int[r];
        r_off  += recv_counts_int[r];
        hs_off += send_hdr_counts[r];
        hr_off += recv_hdr_counts[r];
    }

    struct LETNodeWire *flat_send = (struct LETNodeWire *) mymalloc("LET_flat_send",
        (size_t)total_send * sizeof(struct LETNodeWire) + 1);
    struct LETNodeWire *flat_recv = (struct LETNodeWire *) mymalloc("LET_flat_recv",
        (size_t)total_recv * sizeof(struct LETNodeWire) + 1);
    struct LETSubtreeHeader *flat_hdr_send = (struct LETSubtreeHeader *) mymalloc("LET_flat_hdr_send",
        (size_t)total_hdr_send * sizeof(struct LETSubtreeHeader) + 1);
    struct LETSubtreeHeader *flat_hdr_recv = (struct LETSubtreeHeader *) mymalloc("LET_flat_hdr_recv",
        (size_t)total_hdr_recv * sizeof(struct LETSubtreeHeader) + 1);

    /* Concatenate per-rank send buffers */
    int s_pos = 0, hs_pos = 0;
    for(int r = 0; r < NTask; r++)
    {
        if(send_counts_int[r] > 0 && send_buf_per_rank[r])
        {
            memcpy(flat_send + s_pos, send_buf_per_rank[r],
                   (size_t)send_counts_int[r] * sizeof(struct LETNodeWire));
        }
        if(send_hdr_counts[r] > 0 && send_hdr_per_rank[r])
        {
            memcpy(flat_hdr_send + hs_pos, send_hdr_per_rank[r],
                   (size_t)send_hdr_counts[r] * sizeof(struct LETSubtreeHeader));
        }
        s_pos  += send_counts_int[r];
        hs_pos += send_hdr_counts[r];
    }

    /* MPI exchanges (parallel for nodes + headers). Counts and displacements
     * are in element units, not bytes — see system/mpi_alltoallv_typed.h. */
    double t_let_mpi = my_second();
    gizmo_mpi_alltoallv_typed(flat_send,     send_counts_int, send_offsets,
                              flat_recv,     recv_counts_int, recv_offsets,
                              sizeof(struct LETNodeWire), MPI_COMM_WORLD);
    gizmo_mpi_alltoallv_typed(flat_hdr_send, send_hdr_counts, send_hdr_offsets,
                              flat_hdr_recv, recv_hdr_counts, recv_hdr_offsets,
                              sizeof(struct LETSubtreeHeader), MPI_COMM_WORLD);
    double t_let_alltoallv = timediff(t_let_mpi, my_second());

    /* Install foreign tree contents while flat_recv / flat_hdr_recv are
     * still alive on the mymalloc stack. Capture the status so an unpack
     * overflow propagates up to let_run_exchange. */
    let_exchange_status_t unpack_status = let_unpack_and_install(flat_recv, recv_counts_int, total_recv,
                            flat_hdr_recv, recv_hdr_counts, total_hdr_recv, foreign_needed_out);

    /* Free everything in strict reverse-alloc order */
    myfree(flat_hdr_recv);
    myfree(flat_hdr_send);
    myfree(flat_recv);
    myfree(flat_send);
    myfree(recv_hdr_offsets);
    myfree(send_hdr_offsets);
    myfree(recv_offsets);
    myfree(send_offsets);
    myfree(recv_hdr_counts);
    myfree(send_hdr_counts);
    myfree(recv_counts_int);
    myfree(send_counts_int);
    double t_let_total = timediff(t_let_start, my_second());
    if(ThisTask == 0 && gizmo_verbose_diag()) {
        printf("  let_exchange_nodes: total=%.4f alltoallv=%.4f total_send=%d total_recv=%d hdr_send=%d hdr_recv=%d\n",
               t_let_total, t_let_alltoallv, total_send, total_recv, total_hdr_send, total_hdr_recv);
        fflush(stdout);
    }
    return unpack_status;
}

/* ----------------------------------------------------------------------
 * Step 5: install received nodes into the Nodes_base[] foreign slot range and
 *         splice them into the local walk.  The wire graph is self-contained
 *         (pack's let_relabel_subtree() resolved all topology to wire indices /
 *         LET_WIRE_EXIT), so the receiver does NOT reconstruct sender topology.
 *
 * Per sender r (payload length recv_count_per_rank[r]):
 *   - byte-copy each LETNodeWire into a consecutive foreign slot;
 *   - rebase each u.d.{sibling,nextnode} value V: V in [0,rcount) -> slot_base+V,
 *     V == LET_WIRE_EXIT -> mapped (Pass 2) to the owning local topleaf's
 *     continuation; anything else is a malformed graph and aborts loudly;
 *   - each subtree header h (topleaf_idx in the SHARED DomainNodeIndex[]/
 *     DomainTask[] partition) redirects Nodes[DomainNodeIndex[h.topleaf_idx]]
 *     .u.d.nextnode -> slot_base + h.wire_offset (the subtree root).
 *
 * Buffer-overflow policy (Phase 9.0): if Numforeignnodes would exceed
 * MaxForeignNodes, return LET_OVERFLOW_RETRYABLE without installing; the
 * force_treebuild loop ratchets the arena and retries.
 * ---------------------------------------------------------------------- */
extern "C" let_exchange_status_t let_unpack_and_install(const struct LETNodeWire *recv_buf,
                                       const int *recv_count_per_rank,
                                       int recv_count_total,
                                       const struct LETSubtreeHeader *recv_hdr_buf,
                                       const int *recv_hdr_count_per_rank,
                                       int recv_hdr_count_total,
                                       long long *foreign_needed_out)
{
    if(foreign_needed_out) *foreign_needed_out = 0;
    if(recv_count_total == 0) return LET_OK;

    /* Capacity check (long long: capacity-bound, avoid int overflow on the sum).
     * Report the full needed capacity (Numforeignnodes + recv_count_total -- the
     * "+Numforeignnodes" is for a future per-sender/incremental install; today it
     * is 0 at entry) and, on overflow, return a RETRYABLE status WITHOUT installing
     * or aborting. force_treebuild ratchets RuntimeMinLETForeignNodes, rebuilds with
     * a larger arena, and retries; all overflow messaging is owned by that loop. */
    long long foreign_needed = (long long)Numforeignnodes + (long long)recv_count_total;
    if(foreign_needed_out) *foreign_needed_out = foreign_needed;
    if(foreign_needed > (long long)MaxForeignNodes)
    {
        return LET_OVERFLOW_RETRYABLE;
    }

    int node_off = 0;     /* running offset into recv_buf (per-sender) */
    int hdr_off  = 0;     /* running offset into recv_hdr_buf (per-sender) */

    for(int r = 0; r < NTask; r++)
    {
        int rcount = recv_count_per_rank[r];
        int hcount = recv_hdr_count_per_rank[r];
        if(rcount == 0)
        {
            /* skip — no nodes from this sender; should also have no headers */
            hdr_off += hcount;
            continue;
        }

        int slot_base = All.MaxPart + MaxNodes + Numforeignnodes;

        /* Pass 1: byte-copy nodes and rebase wire-local topology to absolute slots.
         * The wire graph is self-contained -- pack RELABEL resolved every terminator to a
         * wire index or LET_WIRE_EXIT -- so each sibling/nextnode is either LET_WIRE_EXIT
         * (resolved to the local topleaf continuation in Pass 2) or an intra-sender wire
         * index in [0,rcount).  No sender-index reconstruction; remote_id is identity-only. */
        for(int j = 0; j < rcount; j++)
        {
            int abs_idx = slot_base + j;
            Nodes[abs_idx]    = recv_buf[node_off + j].node;
            Extnodes[abs_idx] = recv_buf[node_off + j].extnode;

            int sib  = Nodes[abs_idx].u.d.sibling;
            int next = Nodes[abs_idx].u.d.nextnode;
            if(sib != LET_WIRE_EXIT)
            {
                if(sib >= 0 && sib < rcount) Nodes[abs_idx].u.d.sibling = slot_base + sib;
                else { printf("LET install FATAL: node slot %d sibling wire %d out of [0,%d) and != EXIT "
                              "(rank %d).\n", abs_idx, sib, rcount, ThisTask); fflush(stdout); endrun(90000074); }
            }
            if(next != LET_WIRE_EXIT)
            {
                if(next >= 0 && next < rcount) Nodes[abs_idx].u.d.nextnode = slot_base + next;
                else { printf("LET install FATAL: node slot %d nextnode wire %d out of [0,%d) and != EXIT "
                              "(rank %d).\n", abs_idx, next, rcount, ThisTask); fflush(stdout); endrun(90000075); }
            }
            Nodes[abs_idx].u.d.father = -1;  /* foreign nodes have no local father */

            /* C1: install the foreign-leaf identity sidecar at this slot's foreign_slot
             * (= no - (MaxPart+MaxNodes), NOT the SoA index no-MaxPart). Non-leaf records carry
             * leaf_tag=0 from the packer, so this write is correct (and explicit) for both. */
            int foreign_slot = abs_idx - (All.MaxPart + MaxNodes);
            if(ForeignLeafTag && foreign_slot >= 0 && foreign_slot < MaxForeignNodes)
            {
                ForeignLeafTag[foreign_slot]  = recv_buf[node_off + j].leaf_tag;
                ForeignLeafType[foreign_slot] = recv_buf[node_off + j].leaf_type;
                ForeignLeafZeta[foreign_slot] = recv_buf[node_off + j].leaf_ags_zeta;
                ForeignLeafSoft[foreign_slot] = recv_buf[node_off + j].leaf_force_softening;
            }
        }

        /* Pass 2: resolve remaining plain sentinels (LET_EDGE_SENTINEL_BASE) to topleaf_sibling,
         * and redirect each affected local topleaf's u.d.nextnode at the foreign subtree root. */
        for(int hh = 0; hh < hcount; hh++)
        {
            const struct LETSubtreeHeader *h = &recv_hdr_buf[hdr_off + hh];
            int topleaf_idx = h->topleaf_idx;
            int wire_off    = h->wire_offset;
            int wire_cnt    = h->count;

            /* Defensive bounds */
            if(topleaf_idx < 0 || topleaf_idx >= NTopleaves) continue;
            if(wire_off < 0 || wire_cnt <= 0 || wire_off + wire_cnt > rcount) continue;

            int local_topleaf_no = DomainNodeIndex[topleaf_idx];
            if(local_topleaf_no < All.MaxPart || local_topleaf_no >= All.MaxPart + MaxNodes) continue;

            int topleaf_sibling = Nodes[local_topleaf_no].u.d.sibling;
            int subtree_root    = slot_base + wire_off;

            /* Map this subtree's LET_WIRE_EXIT markers to the local topleaf's continuation, and
             * assert HEADER-RANGE CONFINEMENT: every intra-subtree topology link must stay within
             * this header's foreign range [subtree_root, subtree_root+wire_cnt); the only edge that
             * leaves it is the exit -> topleaf_sibling.  Pack builds wire indices per-subtree so this
             * holds by construction; the check makes a cross-header link loud rather than silent. */
            int sub_lo = subtree_root, sub_hi = subtree_root + wire_cnt;
            for(int j = wire_off; j < wire_off + wire_cnt; j++)
            {
                int abs_idx = slot_base + j;
                if(Nodes[abs_idx].u.d.sibling  == LET_WIRE_EXIT) Nodes[abs_idx].u.d.sibling  = topleaf_sibling;
                if(Nodes[abs_idx].u.d.nextnode == LET_WIRE_EXIT) Nodes[abs_idx].u.d.nextnode = topleaf_sibling;
                int s = Nodes[abs_idx].u.d.sibling, n = Nodes[abs_idx].u.d.nextnode;
                if(s != topleaf_sibling && !(s >= sub_lo && s < sub_hi))
                { printf("LET install FATAL: node slot %d sibling %d escapes header range [%d,%d) and "
                         "!= topleaf_sibling %d (rank %d).\n", abs_idx, s, sub_lo, sub_hi, topleaf_sibling, ThisTask);
                  fflush(stdout); endrun(90000077); }
                if(n != topleaf_sibling && !(n >= sub_lo && n < sub_hi))
                { printf("LET install FATAL: node slot %d nextnode %d escapes header range [%d,%d) and "
                         "!= topleaf_sibling %d (rank %d).\n", abs_idx, n, sub_lo, sub_hi, topleaf_sibling, ThisTask);
                  fflush(stdout); endrun(90000078); }
            }

            /* Redirect local topleaf at the foreign subtree root (AoS + SoA). */
            Nodes[local_topleaf_no].u.d.nextnode = subtree_root;
            gpu_set_soa_nextnode(local_topleaf_no, subtree_root);

        }
        /* Pass 3: AoS -> SoA scatter for the foreign-node range we just
         * installed.  GPU walk reads node fields via SoA only; without this
         * the foreign nodes would have garbage SoA entries. */
        gpu_scatter_foreign_to_soa(slot_base, rcount);

        Numforeignnodes += rcount;
        node_off += rcount;
        hdr_off  += hcount;
    }

    return LET_OK;
}

/* ----------------------------------------------------------------------
 * Top-level orchestrator
 * ---------------------------------------------------------------------- */
extern "C" let_exchange_status_t let_run_exchange(long long *foreign_needed_out)
{
    if(foreign_needed_out) *foreign_needed_out = 0;
    /* Defensive no-op if no foreign-node headroom was allocated.  GPU builds
     * reject LETAllocFactor<=0 during parameter validation because the legacy
     * gravity export fallback is retired there. */
    if(MaxForeignNodes <= 0) return LET_OK;

    g_let_pack_oom = 0;   /* fresh status for this exchange */

    /* let_synthesize_particle_leaf and let_compute_local_payload read
     * P/CellP and may transitively invoke RT/sink/CR helpers that mutate
     * cached fields in CellP.  Invalidate the GPU particles arena up front
     * so the next gpu_particles_arena_acquire re-seeds from host. */
    gpu_particles_arena_invalidate();

    /* Reset foreign count -- fresh LET each tree-build cycle */
    Numforeignnodes = 0;

    /* C1: reset the foreign-leaf identity sidecar for the fresh exchange.  These host arrays are
     * read only by host code (the CPU walk and gpu_scatter_foreign_to_soa); the GPU mirror lives in
     * the tree SoA and is fully rewritten by the scatter after install.  let_run_exchange runs at
     * tree-build time, after the previous step's gravity walk has fenced and returned, so no GPU
     * walk is in flight against the old sidecar -- a device fence is not required for this host memset. */
    if(ForeignLeafTag)  memset(ForeignLeafTag,  0, (size_t)MaxForeignNodes * sizeof(int));
    if(ForeignLeafType) memset(ForeignLeafType, 0, (size_t)MaxForeignNodes * sizeof(int));
    if(ForeignLeafZeta) memset(ForeignLeafZeta, 0, (size_t)MaxForeignNodes * sizeof(MyFloat));
    if(ForeignLeafSoft) memset(ForeignLeafSoft, 0, (size_t)MaxForeignNodes * sizeof(MyFloat));

    /* All-local receiver cover (WHICH particles): the payload worst-case scalars
     * (min_OldAcc/soft/sink) still cover ALL of R's particles, not just active ones,
     * so inactive / TREECOL / RT consumers stay complete (an active-only cover would
     * silently drop them; still behind the audit-gated, default-OFF compile guard).
     * Note this is ORTHOGONAL to the per-topleaf cover GEOMETRY (how the boxes are
     * grouped) that the pack now uses: per-topleaf refines min_dist selectivity while
     * this all-local scalar cover keeps completeness. The shared cell predicate
     * (conservative + walk-consistent) + the self-sizing foreign arena keep it safe. */
    struct LETPerRankPayload my_payload;
#ifdef LET_ACTIVE_RECEIVER_COVER_EXPERIMENTAL
    int bitmap_n_words = let_bitmap_word_count(NTopleaves);
    if(bitmap_n_words < 1) bitmap_n_words = 1;
    uint64_t *my_active_bitmap = (uint64_t *) mymalloc("LET_my_active_bitmap",
        (size_t) bitmap_n_words * sizeof(uint64_t));
    let_compute_local_active_bitmap(my_active_bitmap, bitmap_n_words);

    uint64_t *all_active_bitmaps = (uint64_t *) mymalloc("LET_all_active_bitmaps",
        (size_t) NTask * (size_t) bitmap_n_words * sizeof(uint64_t));
    MPI_Allgather(my_active_bitmap, bitmap_n_words, MPI_UINT64_T,
                  all_active_bitmaps, bitmap_n_words, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    /* Tighten the cover to MY active topleaves only (may regress TREECOL self-shielding for
     * non-active particles -- that is exactly what the audit gate guards). */
    let_compute_local_payload(&my_payload, my_active_bitmap, bitmap_n_words);
#else
    let_compute_local_payload(&my_payload, NULL, 0);
#endif

    struct LETPerRankPayload *all_payloads =
        (struct LETPerRankPayload *) mymalloc("LET_payloads", NTask * sizeof(struct LETPerRankPayload));
    let_exchange_payloads(&my_payload, all_payloads);

    /* Exchange the per-topleaf scalar table: every rank filled its OWNED slice in
     * let_compute_local_payload; Allgatherv (mirroring the DomainMoment layout, one slice per
     * MULTIPLEDOMAINS) so every sender holds every receiver-topleaf's scalars for the cover-tree
     * essentiality test. Blocking (the table must be ready before the pack loop; ~72 B/topleaf). */
    if(NTopleaves > 0)
    {
        for(int m = 0; m < MULTIPLEDOMAINS; m++)
        {
            int *rc = (int *) mymalloc("LET_tls_rc", NTask * sizeof(int));
            int *ro = (int *) mymalloc("LET_tls_ro", NTask * sizeof(int));
            for(int recvTask = 0; recvTask < NTask; recvTask++)
            {
                rc[recvTask] = (DomainEndList[recvTask * MULTIPLEDOMAINS + m] -
                                DomainStartList[recvTask * MULTIPLEDOMAINS + m] + 1)
                               * (int) sizeof(struct LETTopleafScalars);
                ro[recvTask] = DomainStartList[recvTask * MULTIPLEDOMAINS + m]
                               * (int) sizeof(struct LETTopleafScalars);
            }
            MPI_Allgatherv(MPI_IN_PLACE, rc[ThisTask], MPI_BYTE,
                           g_topleaf_scalars, rc, ro, MPI_BYTE, MPI_COMM_WORLD);
            myfree(ro); myfree(rc);
        }
    }

    /* Exchange drift-orphan receiver-cover records. The counts already rode the payload Allgather
     * (all_payloads[r].n_orphans), so every rank agrees on the global total and enters/skips the
     * orphan Allgatherv COLLECTIVELY (no rank-local branch around a collective). g_orphan_off[R..R+1)
     * then selects R's records when building R's cover. */
    {
        if(g_orphan_off_cap < NTask + 1) {
            int *nof = (int *) realloc(g_orphan_off, (size_t)(NTask + 1) * sizeof(int));
            if(!nof) { printf("let orphan exchange: g_orphan_off realloc failed (NTask=%d, rank=%d). Stopping.\n",
                             NTask, ThisTask); fflush(stdout); endrun(90000096); }
            g_orphan_off = nof; g_orphan_off_cap = NTask + 1;
        }
        g_orphan_off[0] = 0;
        for(int r = 0; r < NTask; r++) {
            int c = all_payloads[r].n_orphans; if(c < 0) c = 0;
            g_orphan_off[r + 1] = g_orphan_off[r] + c;
        }
        int total_orphans = g_orphan_off[NTask];
        if(total_orphans > 0) {
            if(g_orphan_all_cap < total_orphans) {
                struct LETOrphanRecord *na = (struct LETOrphanRecord *) realloc(g_orphan_all,
                    (size_t) total_orphans * sizeof(struct LETOrphanRecord));
                if(!na) { printf("let orphan exchange: g_orphan_all realloc failed (total=%d, rank=%d). Stopping.\n",
                                 total_orphans, ThisTask); fflush(stdout); endrun(90000097); }
                g_orphan_all = na; g_orphan_all_cap = total_orphans;
            }
            int *rc = (int *) mymalloc("LET_orph_rc", NTask * sizeof(int));
            int *ro = (int *) mymalloc("LET_orph_ro", NTask * sizeof(int));
            for(int r = 0; r < NTask; r++) {
                rc[r] = (g_orphan_off[r + 1] - g_orphan_off[r]) * (int) sizeof(struct LETOrphanRecord);
                ro[r] =  g_orphan_off[r]                        * (int) sizeof(struct LETOrphanRecord);
            }
            /* Seed MY slice, then in-place Allgatherv (mirrors the per-topleaf scalar table exchange). */
            if(g_my_orphans_n > 0)
                memcpy((char *) g_orphan_all + ro[ThisTask], g_my_orphans,
                       (size_t) g_my_orphans_n * sizeof(struct LETOrphanRecord));
            MPI_Allgatherv(MPI_IN_PLACE, rc[ThisTask], MPI_BYTE,
                           g_orphan_all, rc, ro, MPI_BYTE, MPI_COMM_WORLD);
            myfree(ro); myfree(rc);
        }
    }

    /* Collective controlled stop if ANY rank saw a local particle whose Father chain reaches NO topleaf
     * at all -> genuine tree-topology corruption (its target has geometry in no cover). A particle that
     * merely drifted under a FOREIGN topleaf is NOT unbucketable -- it is handled above as a drift-orphan
     * cover extension. This guard is the real-corruption backstop, NOT a fold-into-every-leaf downgrade. */
    {
        long long unbuck_max = 0;
        MPI_Allreduce(&g_let_unbucketable, &unbuck_max, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if(unbuck_max > 0)
        {
            if(g_let_unbucketable > 0)
                printf("LET cover: rank=%d had %lld local particle(s) (first ID=%lld) whose Father chain "
                       "reached NO topleaf -- tree-topology corruption. Stopping.\n",
                       ThisTask, g_let_unbucketable, g_let_unbucketable_first_id);
            fflush(stdout);
            endrun(90000094);
        }
    }

    /* Pack per remote rank.  Per-rank send buffers grown via realloc.  */
    struct LETNodeWire **send_per_rank = (struct LETNodeWire **) mymalloc("LET_send_perrank",
        NTask * sizeof(struct LETNodeWire *));
    struct LETSubtreeHeader **send_hdr_per_rank = (struct LETSubtreeHeader **) mymalloc("LET_send_hdr_perrank",
        NTask * sizeof(struct LETSubtreeHeader *));
    int *send_count     = (int *) mymalloc("LET_send_count",     NTask * sizeof(int));
    int *send_hdr_count = (int *) mymalloc("LET_send_hdr_count", NTask * sizeof(int));
    for(int r = 0; r < NTask; r++) {
        send_per_rank[r] = NULL; send_count[r] = 0;
        send_hdr_per_rank[r] = NULL; send_hdr_count[r] = 0;
    }

    for(int r = 0; r < NTask; r++)
    {
        if(r == ThisTask) {send_count[r] = 0; send_hdr_count[r] = 0; continue;}
        int cap = 0, hcap = 0, hcnt = 0;
#ifdef LET_ACTIVE_RECEIVER_COVER_EXPERIMENTAL
        const uint64_t *r_bitmap = all_active_bitmaps + (size_t) r * (size_t) bitmap_n_words;
        send_count[r] = let_pack_for_rank(r, all_payloads,
                                           &send_per_rank[r], &cap,
                                           &send_hdr_per_rank[r], &hcap, &hcnt,
                                           r_bitmap, bitmap_n_words);
#else
        send_count[r] = let_pack_for_rank(r, all_payloads,
                                           &send_per_rank[r], &cap,
                                           &send_hdr_per_rank[r], &hcap, &hcnt,
                                           NULL, 0);
#endif
        send_hdr_count[r] = hcnt;
    }

    /* Exchange + install (let_exchange_nodes inlines let_unpack_and_install
     * to keep mymalloc LIFO discipline correct). Runs on ALL ranks even after a
     * local pack OOM (failed ranks carry zero send counts) so the Alltoallv stays
     * matched; the nonzero status drains via the caller after this returns. */
    let_exchange_status_t exch_status = let_exchange_nodes(send_per_rank, send_count,
                       send_hdr_per_rank, send_hdr_count, foreign_needed_out);

    /* Free per-rank send buffers (allocated via realloc, not mymalloc) */
    for(int r = 0; r < NTask; r++) {
        if(send_per_rank[r]) free(send_per_rank[r]);
        if(send_hdr_per_rank[r]) free(send_hdr_per_rank[r]);
    }

    /* Cleanup */
    myfree(send_hdr_count);
    myfree(send_count);
    myfree(send_hdr_per_rank);
    myfree(send_per_rank);
    myfree(all_payloads);
#ifdef LET_ACTIVE_RECEIVER_COVER_EXPERIMENTAL
    myfree(all_active_bitmaps);
    myfree(my_active_bitmap);
#endif
    /* Worst-status wins: a send-buffer malloc failure (g_let_pack_oom) is not
     * fixable by a larger foreign arena, so it outranks a retryable overflow. */
    if(g_let_pack_oom) return LET_PACK_OOM;
    return exch_status;
}

/* ----------------------------------------------------------------------
 * let_finalize_unredirected_foreign_topleaves
 *
 * LET completeness invariant. After let_run_exchange() (foreign subtrees
 * installed + topleaf redirects done) AND force_exchange_pseudodata_complete()
 * (foreign topleaf moments + N_part populated in the AoS NODE), every foreign
 * topleaf reachable by the GPU gravity walk must be in exactly one of:
 *
 *   (1) redirected -- Nodes[DomainNodeIndex[t]].u.d.nextnode points into the
 *       foreign-node range: an installed LET subtree.  Left untouched.
 *   (2) provably empty -- still pointing into the pseudo range, AND the
 *       foreign topleaf has BOTH zero mass moment AND zero N_part.  Such a
 *       topleaf carries no gravitational moment and no structural payload
 *       (RT/CR/sink/etc. ride on real particles, of which there are none),
 *       so opening it contributes exactly nothing: rewrite nextnode := sibling
 *       (AoS + SoA) so the walk skips it instead of hitting the pseudo.
 *
 * Anything else -- a topleaf still pointing into the pseudo range that is not
 * provably empty by BOTH measures -- is a LET correctness failure: Phase 9.4
 * retired the CPU gravity export path, so the LET MUST supply every non-empty
 * foreign subtree.  Abort loudly with full context.
 *
 * mass is the physics criterion; N_part is the structural-invariant guard --
 * a topleaf with N_part>0 must never be silently skipped even if its gravity
 * mass moment is zero, and mass>0 with N_part==0 is an inconsistent moment.
 * Both are transmitted to foreign nodes by force_exchange_pseudodata
 * (forcetree.cc: pack line ~1071, unpack line ~1185).
 *
 * Must be called from force_treebuild() AFTER gpu_scatter_pseudo_to_soa(),
 * before any GPU gravity walk reads the SoA.
 * ---------------------------------------------------------------------- */
extern "C" void let_finalize_unredirected_foreign_topleaves(void)
{
    if(MaxForeignNodes <= 0) return;   /* non-GPU build: LET inactive */

    const long long pseudo_lo = (long long)All.MaxPart + MaxNodes + MaxForeignNodes;
    const long long pseudo_hi = pseudo_lo + NTopleaves;
    int n_patched = 0;

    for(int t = 0; t < NTopleaves; t++)
    {
        if(DomainTask[t] == ThisTask) continue;          /* local topleaf */
        int no = DomainNodeIndex[t];
        if(no < All.MaxPart || no >= All.MaxPart + MaxNodes)
        {
            printf("LET finalize FATAL: foreign topleaf t=%d owner=%d has out-of-range "
                   "DomainNodeIndex=%d (local node range [%d,%d)).\n",
                   t, DomainTask[t], no, All.MaxPart, All.MaxPart + MaxNodes);
            fflush(stdout); endrun(90000063); continue; /* soft bad-stop + skip this topleaf before the Nodes[no] deref; drains at gravtree:after_treebuild before the walk */
        }
        const long long nn = Nodes[no].u.d.nextnode;
        if(nn < pseudo_lo || nn >= pseudo_hi) continue;  /* already redirected */

        /* Still pointing at a pseudo. Skip ONLY if provably empty by BOTH
         * the gravity moment and the structural particle count. */
        const double mass  = (double) Nodes[no].u.d.mass;
        const long   npart = Nodes[no].N_part;
        if(mass <= 0.0 && npart == 0)
        {
            Nodes[no].u.d.nextnode = Nodes[no].u.d.sibling;
            gpu_set_soa_nextnode(no, Nodes[no].u.d.sibling);
            n_patched++;
        }
        else
        {
            printf("LET finalize FATAL: foreign topleaf t=%d owner_rank=%d node=%d still "
                   "unredirected (nextnode=%d in pseudo range) and NOT provably empty: "
                   "mass=%g N_part=%ld len=%g sibling=%d. The Locally Essential Tree failed "
                   "to ship this subtree; Phase 9.4 retired the CPU gravity export path, so "
                   "the LET must be complete. (rank=%d)\n",
                   t, DomainTask[t], no, (int)nn, mass, npart, (double)Nodes[no].len,
                   Nodes[no].u.d.sibling, ThisTask);
            fflush(stdout); endrun(90000064); continue; /* soft bad-stop: incomplete LET state drains at gravtree:after_treebuild before the GPU walk reads it */
        }
    }
    if(n_patched > 0 && gizmo_verbose_diag()) {
        printf("LET finalize: rank=%d skip-to-sibling applied to %d provably-empty foreign topleaves.\n",
               ThisTask, n_patched);
        fflush(stdout);
    }
}

