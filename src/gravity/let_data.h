/*! \file let_data.h
 *  \brief Step 13 Phase 9.1b -- Locally Essential Tree (LET) wire-format and
 *         per-rank-payload data structures.
 *
 *  Phase 9 of the Step-13 GPU domain/gravity port replaces the iterative
 *  do-while(ndone<NTask) gravity export with a one-shot LET exchange: each
 *  rank ships every other rank the local-tree subtrees that are "essential"
 *  for that rank's particles, then walks its own particles through the
 *  resulting foreign tree without mid-walk MPI.
 *
 *  This header defines:
 *    - struct LETPerRankPayload  -- ~96 bytes / remote rank, broadcast via
 *      MPI_Allgather to inform the LET pack on every rank what each remote
 *      rank's particles look like (worst-case bounds for opening criteria).
 *    - struct LETNodeWire        -- fixed-size record for one shipped foreign
 *      node, packed contiguously: { remote_id, NODE, extNODE }.  Remote_id
 *      is the source rank's Nodes_base[] index, used as the lookup key in
 *      the unpack pointer-remap step (9.1e).
 *
 *  See ~/.claude/projects/.../memory/handoff_step13_phase9_walk_audit.md for
 *  the audit of all 20 walk-kernel opening branches that drove the LET
 *  payload design.
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
 *   bbox_min/max         : tightest AABB containing all of R's particles
 *                          (or R's topleaf bboxes — implementation choice).
 *                          min_dist(node, bbox) gives r-bound for all r-in-test.
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
};
/* Sizeof = 14 doubles + 2 ints = 120 B (8-B aligned, no tail pad).
 * Total bandwidth for the per-rank Allgather: NTask * 120 B (e.g. 120 KB at
 * NTask=1024) -- negligible. */

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
    int    _pad0;          /* alignment to 8B */
    struct NODE    node;   /* full NODE struct, sender's u.d form (post-build) */
    struct extNODE extnode;/* full extNODE struct */
};
/* Wire size: 8 B header + sizeof(NODE) + sizeof(extNODE).  NODE is ALIGN(32);
 * with all #ifdef payloads off, sizeof(NODE) ~ 144 B, sizeof(extNODE) ~ 80 B,
 * total ~ 232 B per shipped node.  With heavy payloads (RT_USE_GRAVTREE,
 * SINK_*, ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION) it inflates to ~400-500 B.
 * The post-build pad in NODE.u.d (after the struct fields, before the union
 * close) is included in sizeof(NODE) so the wire layout is unambiguous. */

/* Subtree-exit on the wire: a relabelled sibling/nextnode that leaves the shipped
 * subtree carries LET_WIRE_EXIT (defined in let_pack.cc), which the receiver maps to
 * the owning topleaf's continuation per the subtree header.  It is the only edge that
 * leaves a subtree; all other topology is wire-local. */

/* ----------------------------------------------------------------------
 * Per-subtree header (Phase 9.1e_v2)
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
 * Function prototypes (defined in let_pack.cc, 9.1c-e)
 * ---------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! Compute this rank's LETPerRankPayload from local particle state.  Called
 *  once per gravity-tree build, just before the LET exchange.
 *
 *  If active_bitmap is non-NULL: bbox spans only ACTIVE topleaves on this
 *  rank, and per-particle bounds (OldAcc, softening, has_sink) span only
 *  ActiveParticleList.  This is the Phase 9.5 "tight" mode -- safe iff the
 *  receiver's walk only consults LET data for its active particles.
 *
 *  If active_bitmap is NULL: bbox spans all MY topleaves and per-particle
 *  bounds span all NumPart.  Phase 9.4 conservative mode -- always safe,
 *  including for RT/TREECOL walks that may consult foreign-tree column
 *  density for non-active particles. */
void let_compute_local_payload(struct LETPerRankPayload *out,
                               const uint64_t *active_bitmap,
                               int bitmap_n_words);

/*! Phase 9.5: compute this rank's active-topleaf bitmap.  bitmap is sized
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
 *  endrun() with the LETAllocFactor restart message (matches Phase 9.0
 *  buffer-overflow policy).  Future option (b) -- graceful shrink + revert
 *  to legacy export -- documented in handoff_step13_phase9_locked.md but
 *  not implemented unless practical memory limits demand it. */
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
