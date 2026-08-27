/*! \file let_data.h
 *  \brief Locally Essential Tree (LET) wire-format and
 *         per-rank-payload data structures.
 *
 *  Replaces the iterative do-while(ndone<NTask) gravity export with a
 *  one-shot LET exchange: each rank ships every other rank the local-tree
 *  subtrees that are "essential" for that rank's particles, then walks its
 *  own particles through the resulting foreign tree without mid-walk MPI.
 *
 *  This header defines:
 *    - struct LETPerRankPayload  -- ~96 bytes / remote rank, broadcast via
 *      MPI_Allgather to inform the LET pack on every rank what each remote
 *      rank's particles look like (worst-case bounds for opening criteria).
 *    - struct LETNodeWire        -- fixed-size record for one shipped foreign
 *      node, packed contiguously: { remote_id, NODE, extNODE }.  Remote_id
 *      is the source rank's Nodes_base[] index, used as the lookup key in
 *      the unpack pointer-remap step.
 */
#ifndef GIZMO_LET_DATA_H
#define GIZMO_LET_DATA_H

#include <stdint.h>
#include "../declarations/allvars.h"

/* ----------------------------------------------------------------------
 * Per-rank payload (LET pack input)
 *
 * Each rank computes its own LETPerRankPayload at gravity_tree() entry
 * (or tree-build entry) and MPI_Allgathers to all NTask ranks.  The
 * receiving rank uses the payload to evaluate "could ANY particle in R's
 * domain open my node n" — the over-include criterion that decides what
 * to ship.
 *
 * Worst-case bounds:
 *   bbox_min/max         : union AABB of R's owned topleaf boxes. RETAINED FOR
 *                          WIRE COMPAT + has_cover ONLY -- the LET pack no longer
 *                          uses these coords for essentiality: it walks R's
 *                          per-topleaf cover TREE (each topleaf box tested
 *                          separately, built sender-side from replicated topleaf
 *                          geometry), which restores the min_dist selectivity the
 *                          single union box destroyed for spatially-spread ranks.
 *   min_OldAcc           : min(P[i].OldAcc) over R's particles.  The
 *                          relative criterion `M*len² > r⁴ * OldAcc * C`
 *                          opens more aggressively when OldAcc is small;
 *                          shipping the min over-includes safely.
 *   max_soft_by_type[6]  : max softening kernel radius per particle Type.
 *                          Adaptive softening (ADAPTIVE_GRAVSOFT_*) makes
 *                          this a real per-particle scan; for fixed soft
 *                          this is just All.SofteningTable[t].
 *   min_soft             : min softening kernel radius over the cover.  The
 *                          min counterpart of max_soft_by_type, feeding the
 *                          non-NEIGHBORS node-softening open (t_h < msoft).
 *   has_sink             : (any P[i].Type == 5).  Gates the
 *                          SINGLE_STAR_DIRECT_GRAVITY_RADIUS criterion.
 *   has_cover            : this rank owns at least one local topleaf (a real
 *                          cover).  A receiver with has_cover==0 -- an empty
 *                          rank with no particles -- is shipped nothing; its
 *                          degenerate origin-cover would otherwise over-open the
 *                          whole tree (min_OldAcc==0 -> relative crit opens all).
 *
 * NOT shipped per-rank (the sender recomputes it identically from global
 * state): the relative-criterion activation (All.ErrTolTheta + the first-step
 * test).  Also already known to all: All.ErrTolForceAcc, All.Rcut[0..1],
 * All.BoxSize, SINGLE_STAR_DIRECT_GRAVITY_RADIUS.
 * ---------------------------------------------------------------------- */
struct LETPerRankPayload {
    double bbox_min[3];
    double bbox_max[3];
    double min_OldAcc;
    double max_soft_by_type[6];
    double min_soft;        /* min target softening over the cover; feeds the non-NEIGHBORS
                               node-softening open (t_h < msoft), the min counterpart of
                               max_soft_by_type used by the relative softening open. */
    int    has_sink;
    int    has_cover;       /* rank owns >=1 local topleaf; receivers with has_cover==0 (empty
                               ranks) are shipped nothing -- the no-real-receiver guard. */
    int    n_orphans;       /* count of this rank's drift-orphan receiver-cover records (see
                               LETOrphanRecord). Rides this Allgather so every rank learns every
                               rank's orphan count -> the orphan-record Allgatherv is entered
                               collectively (all ranks agree total>0) or skipped (total==0). */
    int    n_clusters;      /* count of this rank's rule-1 cluster cover leaves (see LETCoverLeaf).
                               Rides this Allgather exactly like n_orphans so the cluster-cover
                               Allgatherv is entered collectively (all ranks agree total>0). */
    /* NOTE (per-cover scalar refinement): min_OldAcc/max_soft_by_type/min_soft/has_sink here are
       the WHOLE-RANK worst case, DERIVED (reduced) from the per-topleaf LETTopleafScalars table so
       they can never drift from it (SSOT). LET essentiality now reads the per-TOPLEAF scalars from
       that table (target-local, tighter); these whole-rank values remain for has_cover + wire-compat
       + a conservative fallback (e.g. the empty-owned-topleaf case, baked into the table by the owner). */
};
/* LETPerRankPayload sizeof = 14 doubles + 3 ints (+4 tail pad) = 128 B (8-B aligned).
 * Per-rank Allgather bandwidth: NTask * 128 B (e.g. 128 KB at NTask=1024) -- negligible. */

/* Per-topleaf worst-case opening-criterion scalars for the LET cover tree. One record per topleaf,
 * aggregated over the OWNER rank's particles falling in that topleaf, Allgatherv'd (mirroring the
 * DomainMoment exchange) so every sender holds every receiver-topleaf's scalars. The LET cover tree
 * feeds each source node the DECIDING topleaf's OWN scalars instead of the rank-wide worst case:
 * a strictly TARGET-LOCAL refinement of the same opening predicate (fewer spurious opens; strict
 * subset of the whole-rank essential set; the permanent under-import detector is the hard gate). */
struct LETTopleafScalars {
    double min_OldAcc;           /* relative-accel criterion (min over the topleaf's particles) */
    double max_soft_by_type[6];  /* relative-softening open (max per type over the topleaf) */
    double min_soft;             /* non-NEIGHBORS node-softening open (min over the topleaf) */
    int    has_sink;             /* sink-direct gate (any Type-5 in the topleaf) */
    int    populated;            /* 1 iff this owned topleaf actually holds >=1 local target particle; 0 = empty
                                  * (no real targets). Empty leaves formerly got a whole-rank worst-case fallback
                                  * (port-safety over-coverage) which made them FALSE targets and defeated the
                                  * per-topleaf PM-cull -> whole-tree LET import. Cover excludes !populated leaves. */
};

/* Drift-orphan receiver-cover record.  Under ADAPTIVE_TREEFORCE_UPDATE the tree is REUSED while
 * particles drift; a particle still RESIDENT/local on rank R can drift so its Father chain reaches a
 * topleaf t whose DomainTask[t] != R (t is only the current tree geometry containing it -- NOT a
 * change of which rank owns the target).  Such a target is outside every one of R's OWNED-topleaf
 * cover boxes, so R's per-topleaf cover would drop it -> under-import.  Instead R publishes one orphan
 * record per reached foreign topleaf t (merged worst-case over R's local orphans reaching t); every
 * sender appends it to R's cover as an extra leaf with topleaf t's box geometry + these scalars.
 * The receiver rank R is implied by the Allgatherv slice this record arrives in (see g_orphan_off). */
struct LETOrphanRecord {
    int    topleaf;              /* reached foreign topleaf t: cover-leaf geometry = DomainNodeIndex[t] box */
    int    _pad;                 /* 8-B align */
    struct LETTopleafScalars s;  /* merged worst-case opening scalars over R's local targets reaching t */
};

/* One RULE-1 cluster cover leaf: an arbitrary AABB over a group of R's target particles + the conservative
 * opening scalars over that group.  A rank's targets are grouped WITHIN each populated owned topleaf by
 * softening OCTAVE (floor(log2(ForceSoftening_KernelRadius))) -- one cluster per distinct octave present --
 * so an over-softened subset gets its OWN tight box carrying its large softening instead of smearing that
 * softening over the whole topleaf box (the measured LET over-import).  Clusters are Allgatherv'd (like
 * orphan records) so every sender holds every receiver's cluster leaves for the cover-tree essentiality
 * test.  This struct doubles as the in-memory cover-leaf scratch (clusters + orphans both become these). */
struct LETCoverLeaf {
    double bmin[3], bmax[3];     /* tight AABB over the group's member positions (arbitrary, NOT Nodes[]-backed) */
    struct LETTopleafScalars s;  /* min_OldAcc / max_soft_by_type / min_soft / has_sink over the group (populated=1) */
};

/* ----------------------------------------------------------------------
 * Wire format for one shipped foreign node (LET payload exchange)
 *
 * Each shipped node is a packed record { remote_id, NODE, extNODE }.  NODE+extNODE
 * bytes are copied verbatim from the sender's UVM Nodes_base/Extnodes_base — both
 * sides have identical compile flags so byte layouts match.
 *
 * By the time a node is shipped, the SENDER has relabelled NODE.u.d.{sibling,nextnode}
 * into WIRE-LOCAL topology: each value is either a wire index into this sender's
 * payload (the receiver rebases it to slot_base+wire) or LET_WIRE_EXIT (the receiver
 * maps it to the owning topleaf's continuation).  The receiver never reconstructs
 * sender topology; remote_id is retained for identity/debug only, NOT consulted for
 * graph wiring.  (.father is set to -1 on install: foreign nodes have no local father.)
 * ---------------------------------------------------------------------- */
struct LETNodeWire {
    int    remote_id;      /* sender's Nodes_base index (identity/debug only; NOT topology) */
    int    leaf_tag;       /* 1 = real foreign single-particle leaf source; 0 = node/multipole.
                            * A foreign singleton (one particle, shipped as a synthesized terminal
                            * multipole) must be consumed with PARTICLE-LEAF secondary-source
                            * semantics, not node semantics -- see the leaf-identity sidecar below
                            * and the receiver walk's foreign-leaf routing. */
    struct NODE    node;   /* full NODE struct, sender's u.d form (post-build) */
    struct extNODE extnode;/* full extNODE struct */
    /* Leaf-identity sidecar -- meaningful iff leaf_tag==1.  The synthesized node moment already
     * carries every other source field (mass/pos/vel/soft/RT/sink/CR/tidal), and for a singleton
     * those aggregates equal the particle value.  The two fields a node moment CANNOT carry are the
     * particle Type and AGS_zeta, which the leaf secondary-source path needs (ptype_sec / zeta_sec).
     * Carried at MyFloat (double) so a foreign leaf delivers the SAME zeta the source particle's home
     * rank delivers through its local-leaf path -- preserving force reciprocity exactly. */
    MyFloat leaf_ags_zeta; /* source particle AGS_zeta -> zeta_sec; 0 when adaptive softening is off */
    MyFloat leaf_force_softening; /* source particle ForceSoftening -> h_p.  The node moment carries
                            * maxsoft = max(ForceSoftening, KernelRadius) (conflated for the opening
                            * criterion), which OVER-softens an accepted leaf relative to the local-leaf
                            * force (which uses the pure ForceSoftening) -- the rank-asymmetric softening
                            * that injects momentum under adaptive softening.  Carry the pure value. */
    int     leaf_type;     /* source particle Type -> ptype_sec */
    int     _pad1;         /* explicit tail pad to keep the record 8B-aligned */
#ifdef HERMITE_INTEGRATION
    MyIDType leaf_id;      /* source particle ID -> key into the per-Hermite-pass ghost source
                            * table (HermiteGhostTab). A foreign leaf carries no Old* state, so
                            * the walk cannot re-predict it the way it does a local inactive
                            * source; the ID lets it look up the owner-side predicted state
                            * instead. 0 for non-leaf records. */
#endif
};
#ifdef __cplusplus
#include <cstddef>
/* The wire record is the unit MPI_Alltoallv exchanges (sizeof(LETNodeWire) bytes); all ranks compile
 * the same struct, so the ABI is self-consistent per build.  Adding the leaf sidecar is an INTENTIONAL
 * wire-layout change -- assert structural/alignment invariants (NOT a brittle compile-flag-independent
 * numeric size) so an accidental reorder/misalignment fails loudly at compile time. */
static_assert(offsetof(struct LETNodeWire, node) % alignof(struct NODE) == 0,
              "LETNodeWire.node misaligned vs struct NODE alignment");
static_assert(sizeof(struct LETNodeWire) >= sizeof(struct NODE) + sizeof(struct extNODE),
              "LETNodeWire smaller than its NODE+extNODE payload");
static_assert(sizeof(struct LETNodeWire) % 8 == 0,
              "LETNodeWire not 8-byte aligned");
#endif
/* Wire size: 8 B header + sizeof(NODE) + sizeof(extNODE).  NODE is ALIGN(32);
 * with all #ifdef payloads off, sizeof(NODE) ~ 144 B, sizeof(extNODE) ~ 80 B,
 * total ~ 232 B per shipped node.  With heavy payloads (RT_USE_GRAVTREE,
 * SINK_*, ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION) it inflates to ~400-500 B.
 * The post-build pad in NODE.u.d (after the struct fields, before the union
 * close) is included in sizeof(NODE) so the wire layout is unambiguous. */

/* ----------------------------------------------------------------------
 * Foreign-leaf sidecar arrays (receiver side).
 *
 * The receiver walk reads node fields from Nodes_base[] (CPU) / the GPU SoA, neither of which can
 * carry per-foreign-leaf particle identity (NODE is shared with the local tree and must not be
 * fattened).  These three parallel arrays hold the installed leaf identity for the foreign-node
 * range, sized MaxForeignNodes and indexed by foreign_slot = no - (MaxPart + MaxNodes) -- NOT the
 * ordinary node SoA index no - MaxPart.  Allocated/freed with the foreign-node arena in
 * force_treeallocate/force_treefree; memset-0 reset on each LET exchange (let_run_exchange).  The GPU
 * mirror lives in the tree SoA (foreign_leaf_*), scattered from these in gpu_scatter_foreign_to_soa.
 * ---------------------------------------------------------------------- */
extern int     *ForeignLeafTag;   /* 1 = real foreign single-particle leaf; 0 = node/multipole */
extern int     *ForeignLeafType;  /* source particle Type for a foreign leaf  (-> ptype_sec) */
extern MyFloat *ForeignLeafZeta;  /* source particle AGS_zeta for a foreign leaf (-> zeta_sec) */
extern MyFloat *ForeignLeafSoft;  /* source particle ForceSoftening for a foreign leaf (-> h_p) */
#ifdef HERMITE_INTEGRATION
extern MyIDType *ForeignLeafID;   /* source particle ID for a foreign leaf (-> HermiteGhostTab key) */
#endif

/* ----------------------------------------------------------------------
 * Per-step refresh of imported foreign moments.
 *
 * The LET ships a build-time snapshot; between rebuilds the OWNER's copies of the same
 * physical nodes are continuously maintained (drifted, kick-updated vs) while the imports
 * only drift with pack-time vs and can never be opened deeper than shipped. Measured on
 * hernquist (32k, one crossing): np=2 drifts 45x worse than np=1 and secularly, at ANY
 * rebuild cadence at np=1 -- the staleness is specifically the imported state. The refresh
 * re-sends (s, vs) [+ sink moments] for every already-shipped record each step, keyed by
 * nothing: both sides replay the exchange's flat wire ORDER (sender retains remote_id per
 * record; receiver retains per-sender slot ranges). Granularity stays frozen -- that is the
 * remaining "owner-continuation" channel, not addressed here.
 * ---------------------------------------------------------------------- */
struct LETRefreshRec {
    MyFloat s[3];        /* node CoM (or particle Pos for a synthesized leaf), drifted to now */
    MyFloat vs[3];       /* node mass-weighted velocity (or particle Vel) */
#ifdef SINK_NODE_MOTION_TRACKED
    MyFloat sink_pos[3];
    MyFloat sink_vel[3];
#endif
};

#ifdef __cplusplus
extern "C" {
#endif
void let_refresh_moments(void);     /* collective; no-op until a valid exchange exists */
void let_refresh_invalidate(void);  /* call when the tree/slots die (force_treefree) */
#ifdef __cplusplus
}
#endif

/* Subtree-exit on the wire: a relabelled sibling/nextnode that leaves the shipped
 * subtree carries LET_WIRE_EXIT (defined in let_pack.cc), which the receiver maps to
 * the owning topleaf's continuation per the subtree header.  It is the only edge that
 * leaves a subtree; all other topology is wire-local. */

/* ----------------------------------------------------------------------
 * Per-subtree header
 *
 * Sender emits one LETSubtreeHeader per OUR topleaf shipped to a given
 * receiver R.  The header tells the receiver:
 *   - which LOCAL topleaf (DomainNodeIndex[topleaf_idx]) this foreign
 *     subtree is the contents of, so the receiver can redirect that
 *     topleaf's u.d.nextnode at the foreign subtree root, AND know which
 *     local topleaf's continuation each LET_WIRE_EXIT maps to.
 *   - the range [wire_offset, wire_offset+count) inside the sender's flat
 *     LETNodeWire payload that contains this subtree.  The first wire
 *     entry (at wire_offset) is the subtree root.
 *
 * All ranks share the same DomainTask[]/DomainNodeIndex[] partition on a
 * given build, so topleaf_idx is rank-agnostic and well-defined on the
 * receiver.
 *
 * Headers are exchanged in their own parallel MPI_Alltoall+Alltoallv
 * step, mirroring the node-payload exchange.
 * ---------------------------------------------------------------------- */
struct LETSubtreeHeader {
    int topleaf_idx;    /* index into DomainNodeIndex[]; rank-agnostic */
    int wire_offset;    /* offset into the sender's flat LETNodeWire array */
    int count;          /* number of LETNodeWire entries in this subtree */
    int _pad0;          /* align to 16 B */
};

/* ----------------------------------------------------------------------
 * Function prototypes (defined in let_pack.cc)
 * ---------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! Compute this rank's LETPerRankPayload from local particle state.  Called
 *  once per gravity-tree build, just before the LET exchange.
 *
 *  If active_bitmap is non-NULL: bbox spans only ACTIVE topleaves on this
 *  rank, and per-particle bounds (OldAcc, softening, has_sink) span only
 *  ActiveParticleList.  This is the "tight" mode -- safe iff the
 *  receiver's walk only consults LET data for its active particles.
 *
 *  If active_bitmap is NULL: bbox spans all MY topleaves and per-particle
 *  bounds span all NumPart.  Conservative mode -- always safe,
 *  including for RT/TREECOL walks that may consult foreign-tree column
 *  density for non-active particles. */
void let_compute_local_payload(struct LETPerRankPayload *out,
                               const uint64_t *active_bitmap,
                               int bitmap_n_words);

/*! Compute this rank's active-topleaf bitmap.  bitmap is sized
 *  n_words = (NTopleaves+63)/64 uint64_t.  Bit tl is set iff topleaf tl is
 *  owned by this rank AND at least one particle in ActiveParticleList lives
 *  in it.  If ActiveParticleList is empty, all of MY topleaves' bits are set
 *  (conservative fallback).  Bits for topleaves owned by other ranks are 0. */
void let_compute_local_active_bitmap(uint64_t *bitmap, int n_words);

/*! MPI_Allgather all ranks' LETPerRankPayload into a buffer indexed by rank
 *  (so payload[r] gives rank r's payload).  Returns 0 on success. */
int  let_exchange_payloads(const struct LETPerRankPayload *local,
                           struct LETPerRankPayload *all_ranks /* sized NTask */);

/*! Pack the local subtree of nodes essential for rank R into a LETNodeWire
 *  array.  Walks Nodes_base[] from the root, including each node n the shared
 *  opening predicate would OPEN for R's cover (let_node_essential_for_rank).
 *  Ships nothing to a receiver with no local cover (all_ranks[R].has_cover==0).
 *  receiver_active_bitmap is consulted ONLY in the experimental active-receiver-
 *  cover mode (LET_ACTIVE_RECEIVER_COVER_EXPERIMENTAL); it is NULL on the default
 *  all-local path.  Returns the number of nodes packed; *out grown as needed. */
int  let_pack_for_rank(int R,
                       const struct LETPerRankPayload *all_ranks,
                       struct LETNodeWire **out_buf,
                       int *out_capacity,
                       struct LETSubtreeHeader **out_hdr_buf,
                       int *out_hdr_capacity,
                       int *out_hdr_count,
                       const uint64_t *receiver_active_bitmap,
                       int bitmap_n_words);

/*! LET exchange outcome.  Distinguishes the retryable foreign-arena overflow
 *  (force_treebuild grows the arena and rebuilds) from non-retryable failures
 *  (a bigger arena cannot fix a send-buffer malloc failure or a malformed
 *  exchange).  Worst status wins when reduced across ranks:
 *  PACK_OOM/UNPACK_INTERNAL > OVERFLOW_RETRYABLE > OK. */
typedef enum {
    LET_OK = 0,
    LET_OVERFLOW_RETRYABLE,   /* receiver foreign arena too small; rebuild larger */
    LET_PACK_OOM,             /* send-buffer realloc failed; not retryable */
    LET_UNPACK_INTERNAL       /* malformed exchange; not retryable */
} let_exchange_status_t;

/*! Adaptive lower bound (in nodes) on the LET foreign-node arena MaxForeignNodes.
 *  0 at startup; force_treebuild ratchets it up on a retryable overflow so the
 *  arena self-sizes to the run's evolving clustering.  force_treeallocate is the
 *  sole consumer.  NOT a parameter: All.LETAllocFactor is the input floor, this is
 *  the runtime adaptive floor.  Not restart-persisted (recomputed each run). */
extern long long RuntimeMinLETForeignNodes;

/*! Since-start high-water of Numforeignnodes (the peak foreign nodes actually
 *  installed by any LET exchange). Diagnostic only: the memory ledger reports
 *  foreign "used" from this, because Numforeignnodes itself is reset to 0 by
 *  force_treeallocate before a controlled-stop ledger fires -- current-at-print
 *  would read a misleading 0. Updated at the let_pack install site. */
extern long long Numforeignnodes_highwater;

/*! Two-phase MPI exchange + local install in one scope.  Phase 1 Alltoalls
 *  per-rank node-counts and header-counts; Phase 2 Alltoallvs the
 *  LETNodeWire and LETSubtreeHeader bytes; then directly calls
 *  let_unpack_and_install on the received buffers before freeing all
 *  scratch in strict LIFO order.  Inlining the install keeps mymalloc
 *  stack discipline correct -- earlier draft returned flat_recv /
 *  flat_hdr_recv to the caller, leaving them mid-stack and triggering
 *  "not the last allocated block" aborts when intermediates were freed. */
let_exchange_status_t let_exchange_nodes(struct LETNodeWire **send_buf_per_rank,
                        const int *send_count_per_rank,
                        struct LETSubtreeHeader **send_hdr_per_rank,
                        const int *send_hdr_count_per_rank,
                        long long *foreign_needed_out);

/*! Install received foreign nodes into Nodes_base[] / Extnodes_base[] /
 *  SoA at slots [MaxPart+MaxNodes, MaxPart+MaxNodes+Numforeignnodes): rebase each
 *  wire-local sibling/nextnode to slot_base+wire, map LET_WIRE_EXIT to the owning
 *  topleaf's continuation, redirect each affected local topleaf's u.d.nextnode to
 *  the corresponding foreign subtree root.  Returns LET_OK on success.
 *
 *  CRITICAL: if Numforeignnodes would exceed MaxForeignNodes after install,
 *  endrun() with the LETAllocFactor restart message.  Future option (b) --
 *  graceful shrink + revert to legacy export -- not implemented unless
 *  practical memory limits demand it. */
let_exchange_status_t let_unpack_and_install(const struct LETNodeWire *recv_buf,
                            const int *recv_count_per_rank,
                            int recv_count_total,
                            const struct LETSubtreeHeader *recv_hdr_buf,
                            const int *recv_hdr_count_per_rank,
                            int recv_hdr_count_total,
                            long long *foreign_needed_out);

/*! Top-level LET exchange.  Called from gravity_tree() after the local tree
 *  is built and force_exchange_pseudodata() has run.  Composes the four
 *  steps above; resets Numforeignnodes to 0 first.  Returns a typed status and
 *  (on overflow) the total foreign-node capacity the receiver needed, so the
 *  force_treebuild retry loop can ratchet RuntimeMinLETForeignNodes. */
let_exchange_status_t let_run_exchange(long long *foreign_needed_out);

/*! LET completeness finalizer.  Call from force_treebuild() after
 *  gpu_scatter_pseudo_to_soa(): redirects provably-empty unredirected foreign
 *  topleaves to skip-to-sibling, aborts on any non-empty unredirected one. */
void let_finalize_unredirected_foreign_topleaves(void);

#ifdef __cplusplus
}
#endif


#endif /* GIZMO_LET_DATA_H */
