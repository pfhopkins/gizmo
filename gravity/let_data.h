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
 *    - Inline helpers for min_dist(point, AABB) and the essential-node check.
 *
 *  See ~/.claude/projects/.../memory/handoff_step13_phase9_walk_audit.md for
 *  the audit of all 20 walk-kernel opening branches that drove the LET
 *  payload design.
 */
#ifndef GIZMO_LET_DATA_H
#define GIZMO_LET_DATA_H

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
 *   has_sink             : (any P[i].Type == 5).  Gates the
 *                          SINGLE_STAR_DIRECT_GRAVITY_RADIUS criterion.
 *   use_rel_crit         : whether the relative criterion is active for
 *                          R right now.  Under GRAVITY_HYBRID_OPENING_CRIT
 *                          the relative test is suppressed at startup;
 *                          otherwise always 1.
 *
 * NOT shipped per-rank (already known to all): All.ErrTolTheta,
 * All.ErrTolForceAcc, All.Rcut[0..1], All.BoxSize, SINGLE_STAR_DIRECT_GRAVITY_RADIUS.
 * ---------------------------------------------------------------------- */
struct LETPerRankPayload {
    double bbox_min[3];
    double bbox_max[3];
    double min_OldAcc;
    double max_soft_by_type[6];
    int    has_sink;
    int    use_rel_crit;
};
/* Sizeof = 11 doubles + 2 ints = 96 B (plus possible 4-B tail pad).
 * Total bandwidth for the per-rank Allgather: NTask * 96 B (e.g. 96 KB at
 * NTask=1024) -- negligible. */

/* ----------------------------------------------------------------------
 * Wire format for one shipped foreign node (LET payload exchange)
 *
 * Each shipped node is a packed record { remote_id, NODE, extNODE } where
 * remote_id is the sender's Nodes_base[] index for this node (used as the
 * lookup key during unpack pointer-remap in 9.1e).  NODE+extNODE bytes are
 * copied verbatim from the sender's UVM Nodes_base/Extnodes_base — both
 * sides have identical compile flags so byte layouts match.
 *
 * The NODE.u.d fields .sibling/.nextnode/.father carry the SENDER'S node
 * indices (in their Nodes_base index space).  The unpack step (9.1e) walks
 * the just-installed foreign subtree, builds a remote_id -> local_foreign_idx
 * translation table, and remaps these pointers in-place.  Pointers that
 * reference nodes OUTSIDE the shipped set (the source-side topleaf's
 * sibling/nextnode pointing back into the source's tree) are remapped to
 * point back into OUR local tree at the corresponding topleaf's continuation
 * — preserving the walk's "exit foreign subtree, resume local walk" semantics.
 * ---------------------------------------------------------------------- */
struct LETNodeWire {
    int    remote_id;      /* sender's Nodes_base index for this node */
    int    _pad0;          /* alignment to 8B; reserved for future use (e.g. flags) */
    struct NODE    node;   /* full NODE struct, sender's u.d form (post-build) */
    struct extNODE extnode;/* full extNODE struct */
};
/* Wire size: 8 B header + sizeof(NODE) + sizeof(extNODE).  NODE is ALIGN(32);
 * with all #ifdef payloads off, sizeof(NODE) ~ 144 B, sizeof(extNODE) ~ 80 B,
 * total ~ 232 B per shipped node.  With heavy payloads (RT_USE_GRAVTREE,
 * SINK_*, ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION) it inflates to ~400-500 B.
 * The post-build pad in NODE.u.d (after the struct fields, before the union
 * close) is included in sizeof(NODE) so the wire layout is unambiguous. */

/* ----------------------------------------------------------------------
 * Sentinels for pointer remap (9.1e unpack)
 *
 * The foreign subtree's "outermost" nextnode/sibling pointers reference
 * source-side nodes that aren't in the shipped set.  These are remapped to
 * a sentinel that means "exit foreign subtree, resume local walk at the
 * topleaf's nextnode".  The walk kernel (9.2) recognizes this sentinel and
 * jumps back to the local-tree continuation.
 *
 * Choice: the existing convention `nextnode = -1` already means "end of
 * walk" in the legacy code, so a separate sentinel is unsafe.  Instead, we
 * encode the local topleaf index directly: when a foreign subtree's edge
 * pointer would otherwise reference a remote-only node, we set it to the
 * local topleaf's u.d.nextnode value (already correctly pointing into our
 * tree).  The unpack step has access to `topleaf_no` per shipped subtree,
 * so this works without a sentinel. */
#define LET_FOREIGN_REMAP_TO_TOPLEAF_NEXT  (-2)  /* tentative; walk kernel
   will recognize this if we choose sentinel encoding instead.  Currently
   we plan to do direct topleaf-nextnode substitution (no sentinel). */

/* ----------------------------------------------------------------------
 * Per-subtree header (Phase 9.1e_v2)
 *
 * Sender emits one LETSubtreeHeader per OUR topleaf shipped to a given
 * receiver R.  The header tells the receiver:
 *   - which LOCAL topleaf (DomainNodeIndex[topleaf_idx]) this foreign
 *     subtree is the contents of, so the receiver can redirect that
 *     topleaf's u.d.nextnode at the foreign subtree root, AND know which
 *     local topleaf's sibling each LET_EDGE_SENTINEL_BASE maps to.
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
 * Inline helpers (header-only, GPU-callable as needed)
 * ---------------------------------------------------------------------- */

/*! Squared distance from point (cx,cy,cz) to axis-aligned bounding box
 *  defined by [bbox_min[i], bbox_max[i]].  Zero if point is inside.  Used
 *  by the LET essential-node check to bound the minimum r-distance from
 *  any of R's particles to a node we might ship to R. */
static inline double let_point_to_bbox_dist_sq(const double *bbox_min,
                                                const double *bbox_max,
                                                double cx, double cy, double cz)
{
    double dx, dy, dz;
    if      (cx < bbox_min[0]) dx = bbox_min[0] - cx;
    else if (cx > bbox_max[0]) dx = cx - bbox_max[0];
    else                       dx = 0.0;
    if      (cy < bbox_min[1]) dy = bbox_min[1] - cy;
    else if (cy > bbox_max[1]) dy = cy - bbox_max[1];
    else                       dy = 0.0;
    if      (cz < bbox_min[2]) dz = bbox_min[2] - cz;
    else if (cz > bbox_max[2]) dz = cz - bbox_max[2];
    else                       dz = 0.0;
    return dx*dx + dy*dy + dz*dz;
}

/* ----------------------------------------------------------------------
 * Function prototypes (defined in let_pack.cc, 9.1c-e)
 * ---------------------------------------------------------------------- */
#ifdef OPENMP_GPU_OFFLOAD

#ifdef __cplusplus
extern "C" {
#endif

/*! Compute this rank's LETPerRankPayload from local particle state.  Called
 *  once per gravity-tree build, just before the LET exchange.  Scans the
 *  local active-particle list for OldAcc/softening/sink bounds and the
 *  topleaf bbox.  Caller-supplied buffer must be sized sizeof(LETPerRankPayload). */
void let_compute_local_payload(struct LETPerRankPayload *out);

/*! MPI_Allgather all ranks' LETPerRankPayload into a buffer indexed by rank
 *  (so payload[r] gives rank r's payload).  Returns 0 on success. */
int  let_exchange_payloads(const struct LETPerRankPayload *local,
                           struct LETPerRankPayload *all_ranks /* sized NTask */);

/*! Pack the local subtree of nodes essential for rank R into a LETNodeWire
 *  array.  Walks Nodes_base[] from the root, including each node n where
 *     let_node_essential_for_rank(n, all_ranks[R]) == 1.
 *  Returns the number of nodes packed; *out is realloc'd or grown as needed. */
int  let_pack_for_rank(int R,
                       const struct LETPerRankPayload *all_ranks,
                       struct LETNodeWire **out_buf,
                       int *out_capacity,
                       struct LETSubtreeHeader **out_hdr_buf,
                       int *out_hdr_capacity,
                       int *out_hdr_count);

/*! Two-phase MPI exchange:
 *    Phase 1: MPI_Alltoall the per-rank node-counts AND header-counts so
 *             receivers can size their receive buffers.
 *    Phase 2: MPI_Alltoallv the actual LETNodeWire bytes AND
 *             LETSubtreeHeader bytes (parallel exchanges).
 *  Returns total foreign nodes received in *recv_count_total and
 *  total subtree headers in *recv_hdr_count_total. */
int  let_exchange_nodes(struct LETNodeWire **send_buf_per_rank,
                        const int *send_count_per_rank,
                        struct LETSubtreeHeader **send_hdr_per_rank,
                        const int *send_hdr_count_per_rank,
                        struct LETNodeWire **recv_buf,
                        int *recv_count_per_rank,
                        int *recv_count_total,
                        struct LETSubtreeHeader **recv_hdr_buf,
                        int *recv_hdr_count_per_rank,
                        int *recv_hdr_count_total);

/*! Install received foreign nodes into Nodes_base[] / Extnodes_base[] /
 *  SoA at slots [MaxPart+MaxNodes, MaxPart+MaxNodes+Numforeignnodes), build
 *  the remote_id->local_foreign_idx translation table, remap intra-subtree
 *  pointers, redirect each affected local topleaf's u.suns[0] to the
 *  corresponding foreign subtree root.  Returns 0 on success.
 *
 *  CRITICAL: if Numforeignnodes would exceed MaxForeignNodes after install,
 *  endrun() with the LETAllocFactor restart message (matches Phase 9.0
 *  buffer-overflow policy).  Future option (b) -- graceful shrink + revert
 *  to legacy export -- documented in handoff_step13_phase9_locked.md but
 *  not implemented unless practical memory limits demand it. */
int  let_unpack_and_install(const struct LETNodeWire *recv_buf,
                            const int *recv_count_per_rank,
                            int recv_count_total,
                            const struct LETSubtreeHeader *recv_hdr_buf,
                            const int *recv_hdr_count_per_rank,
                            int recv_hdr_count_total);

/*! Top-level LET exchange.  Called from gravity_tree() after the local tree
 *  is built and force_exchange_pseudodata() has run.  Composes the four
 *  steps above; resets Numforeignnodes to 0 first. */
int  let_run_exchange(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_LET_DATA_H */
