/* Mode B local neighbor walker — host-side range-walk over the existing
 * gravity tree (Nodes[]/Nextnode[]/sibling). Designed to answer remote
 * Mode B queries from a peer rank's local particle set without touching
 * GPU SIDX state.
 *
 * Two implementations behind one API:
 *   - mode_b_local_neighbor_walk()  — TREE-WALK path. Fast.
 *   - mode_b_local_brute_walk()     — BRUTE-FORCE path. The oracle.
 *
 * Both return the same set of LOCAL real P[] indices intersecting the
 * spherical query (pos, h_q) with type/mode filter. Sorted ascending.
 *
 * Oracle: cross-rank Mode B oracle is owned by the runner via
 * GIZMO_NLR_ORACLE (mesh/neighbor_loop_runner.cc). The runner calls
 * both walks for each query and diffs results.
 *
 * Design constraints:
 *   - Walk uses Nodes[]/Nextnode[] for pruning. Node bounds come from
 *     force_drift_node(); they are conservative for current Ti_Current.
 *   - Returned candidates are LOCAL real P[] indices in [0, num_local)
 *     where num_local = ghost_get_num_local(). Never LET pseudo nodes,
 *     never ghost imports.
 *   - Particle drift is the CALLER's responsibility. The walker does
 *     NOT drift P[j] before evaluating the predicate.
 *   - Not thread-safe with concurrent particle drift / tree mutation.
 *   - Symmetric mode without per-node max-h tracking degenerates to
 *     "always open"; ONEWAY is the fast path.
 */

#ifndef MODE_B_LOCAL_WALKER_H
#define MODE_B_LOCAL_WALKER_H

#include <vector>

/* search_mode constants — match NGB_SEARCH_ONEWAY/SYMMETRIC in
 * mesh/ghost_exchange_spec.h to keep the existing NGL contract. */
#ifndef MODE_B_SEARCH_ONEWAY
#define MODE_B_SEARCH_ONEWAY    0
#define MODE_B_SEARCH_SYMMETRIC 1
#endif

/* Type-aware "symmetric radius" policy bitmask + SSOT helper live in
 * mesh/nlr_radius_policy.h; included here so existing Specs that include
 * mode_b_local_walker.h transitively pick up the policy types.  See the
 * docstring there for the full audit table + per-Spec policy assignments. */
#include "nlr_radius_policy.h"

/* Returns the per-j symmetric radius to use under the given policy.
 * Returns 0 when SYMMETRIC should degenerate to ONEWAY for this j's type. */
double mode_b_neighbor_symmetric_radius(int j, mode_b_radius_policy_t policy);

/* Tree-walk path. Fast for spatially-localized queries. APPENDS local-
 * real-particle candidates (P[] indices) onto `out` via push_back; the
 * caller passes a (typically `clear()`'d) vector and the walker grows it
 * geometrically as needed. Does NOT clear `out` itself (so callers may
 * accumulate across queries if desired). Does NOT sort.
 *
 * radius_policy controls how SYMMETRIC search uses h_j for non-gas types.
 * Most callers want MODE_B_RADIUS_DEFAULT (gas KR only).
 *
 * Contract: previous int*+capacity signature replaced by
 * std::vector<int>& to remove the per-query num_local-sized allocation
 * the runner used to make ahead of each call (~24 MB × N_active on
 * fire_m11i — a dominant tiny-N cost). Geometric growth
 * via push_back handles correctness for any-size match set without
 * imposing full-pool memory traffic on tiny-N. */
/* Lazy per-node drift accounting for a threaded walk. A walk drifts a stale
 * tree node to All.Ti_Current under an omp critical; when several threads walk
 * concurrently one thread performs the drift and the rest observe it already
 * fresh. Passing a non-null counter (one instance PER THREAD, reduced after the
 * region) records that behaviour with no hot-path atomics:
 *   stale_node_hits      = fast-path acquire-load found the node stale
 *   lazy_drift_performed = entered the critical, node STILL stale -> this
 *                          thread drifted it
 *   lazy_drift_raced     = entered the critical, node now fresh -> another
 *                          thread drifted it first
 * A serial walk passes nullptr (no accounting, no cost). */
struct ModeBDriftCounters {
    long long stale_node_hits = 0;
    long long lazy_drift_performed = 0;
    long long lazy_drift_raced = 0;
    void add(const ModeBDriftCounters& o) {
        stale_node_hits      += o.stale_node_hits;
        lazy_drift_performed += o.lazy_drift_performed;
        lazy_drift_raced     += o.lazy_drift_raced;
    }
};

/* j_radius_scale: SYMMETRIC-mode multiplier on the j-side kernel radius
 * (1.0 = legacy). TURB_DIFF_DYNAMIC wide-filter loops pass
 * All.TurbDynamicDiffFac so the Mode B reach matches the Mode A scaled-
 * symmetric NGL.
 *
 * drift_ctr (optional, default nullptr): per-thread lazy-drift accounting for a
 * threaded caller; nullptr for a serial walk. */
void mode_b_local_neighbor_walk(const double pos[3],
                                double h_q,
                                unsigned int type_mask,
                                int search_mode,
                                mode_b_radius_policy_t radius_policy,
                                std::vector<int>& out,
                                double j_radius_scale = 1.0,
                                ModeBDriftCounters* drift_ctr = nullptr);

/* Brute-force path. Iterates 0..num_local. Slow but obviously correct.
 * Used as the runner-owned oracle. Same append-oriented contract as the
 * tree walker: appends matches to `out` via push_back; does NOT clear
 * or sort. */
void mode_b_local_brute_walk(const double pos[3],
                             double h_q,
                             unsigned int type_mask,
                             int search_mode,
                             mode_b_radius_policy_t radius_policy,
                             std::vector<int>& out,
                             double j_radius_scale = 1.0);

/* Cross-rank targeted-export support (restores the legacy source-tree export
 * the port dropped). The three
 * public walks below (mode_b_local_neighbor_walk above + the two here) are thin
 * wrappers over ONE shared traversal body, mirroring legacy's 8 ngb_treefind_*
 * wrappers over one codeblock. Legacy templates: system/ngb_codeblock_
 * after_condition_unthreaded.h + hydro/density.cc mode==0/mode==1. */

/* Topleaf reverse map — the READ-ONLY, walk-invariant half of the export
 * machinery. Identifies remote-owned TOP-LEAVES during the walk: post-LET, a
 * shipped remote topleaf's pseudo child is replaced by the imported foreign
 * subtree (let_pack.cc install), so the export event is the OPEN DECISION ON
 * THE TOPLEAF itself — exactly legacy's pseudo-hit set (a pseudo child is
 * reached iff its parent topleaf is opened). SSOT = the DomainNodeIndex[]/
 * DomainTask[] arrays (the same arrays legacy ngb.cc and the modern Ewald
 * export detector use); the map is derived from them per call, never assumed
 * from slot layout. build() once per call; read-only during the walk, so ONE
 * instance is safely shared across all walking threads. */
struct ModeBTopleafMap {
    std::vector<int> leaf_of_topnode;               /* [no - All.MaxPart] -> topleaf id, -1 = not a topleaf */
    int topnode_map_size = 0;                       /* valid offsets: [0, topnode_map_size) */
    void build(void);                               /* fill from DomainNodeIndex[0..NTopleaves) */
    /* Returns the topleaf id for internal node `no`, or -1 if not a topleaf. */
    inline int topleaf_of(int no, int max_part) const {
        const int off = no - max_part;
        if(off < 0 || off >= topnode_map_size) return -1;
        return leaf_of_topnode[off];
    }
};

/* Per-query export sink — the WRITE half. add() records, for the CURRENT
 * query, the DomainNodeIndex start-nodes to export to each owning peer task
 * (legacy pseudo-node export: after_condition_unthreaded.h:19-67). Reused
 * across queries: ensure_size() once per call, clear_all() per query (retains
 * capacity). Write-only during the walk, so each thread owns its OWN instance
 * (no shared export counter). */
struct ModeBExportSink {
    std::vector<std::vector<int>> nodes_per_peer;   /* [owner_task] -> DomainNodeIndex list */
    void ensure_size(int ntask) {
        if((int)nodes_per_peer.size() != ntask) nodes_per_peer.assign(ntask, std::vector<int>{});
    }
    void clear_all() { for(auto &v : nodes_per_peer) v.clear(); }
    void add(int owner_task, int domain_node_index) {
        nodes_per_peer[owner_task].push_back(domain_node_index);
    }
};

/* SENDER walk (legacy mode==0): walk the local tree from the root; at every
 * remote pseudo-node the query reaches, record a targeted export into
 * `exporter` (the owner peer + that node's DomainNodeIndex). Optionally also
 * append local real-particle candidates to `cand_out` in the SAME traversal
 * (pass nullptr to skip). Passing cand_out != nullptr is the FUSED legacy
 * mode==0 walk: one traversal, two sinks, each gated by its own predicate
 * (candidates by the per-type leaf test, exports by the scalar reach) — see
 * the ModeBWalkReach table in mode_b_local_walker.cc. */
void mode_b_walk_and_export(const double pos[3],
                            double h_q,
                            unsigned int type_mask,
                            int search_mode,
                            mode_b_radius_policy_t radius_policy,
                            std::vector<int>* cand_out,
                            const ModeBTopleafMap& topleaf_map,
                            ModeBExportSink& sink,
                            double j_radius_scale = 1.0,
                            ModeBDriftCounters* drift_ctr = nullptr);

/* RECEIVER walk (legacy mode==1): for each exported start-node in
 * node_list[0..n_nodes) (a DomainNodeIndex, -1 terminates early), resume the
 * walk from that node's children and stop when the walk re-enters the
 * top-level tree (BITFLAG_TOPLEVEL) — the bounded subtree resume from
 * hydro/density.cc:272,351 + codeblock:74-81. Appends local candidates to
 * `out`. No export. */
void mode_b_walk_from_start_nodes(const double pos[3],
                                  double h_q,
                                  unsigned int type_mask,
                                  int search_mode,
                                  mode_b_radius_policy_t radius_policy,
                                  const int *node_list,
                                  int n_nodes,
                                  std::vector<int>& out,
                                  double j_radius_scale = 1.0,
                                  ModeBDriftCounters* drift_ctr = nullptr);

/* Targeted-export eligibility gate RETIRED. Every Mode-B loop now
 * routes via targeted export: the sender's SYMMETRIC node prune uses the cross-rank
 * PER-TYPE band (mode_b_node_symmetric_radius), which dominates every radius_policy
 * (allvars.h invariant; seed capped at MaxKernelRadius, kernel/AGS radii <=
 * MaxKernelRadius by construction, FS seeded uncapped) and is cross-rank-fresh (S2
 * DomainNODE exchange + force_update_hmax post-density exchange). The historical
 * gas-kernel-only restriction (gas-biased scalar hmax under-covered non-gas / AGS /
 * ForceSoftening loops: sink_env1/2/feed/swk, ags_force, grain) is gone. The
 * broadcast source path in neighbor_loop_runner.cc is now compile-time-dead cleanup
 * debt (targeted_export_ok hard-coded true), pending physical deletion. */

/* Lazy-drift contract for Mode B (mirrors gpu_ngb_list_build:1542-1580).
 *
 * GPU NGL contract: candidate walk runs on whatever P[j].Pos state exists,
 * THEN drift_particle(j, All.Ti_Current) is called for each j in the CSR,
 * THEN the pair kernel reads drifted P[j]. The walk may see slightly stale
 * positions (h-slack absorbs that); the kernel always reads fresh.
 *
 * Mode B must honor the same contract. Walker collects candidates first;
 * call this helper to drift them; then run the pair kernel.
 *
 * Drifts each j in indices[] to current Ti via drift_particle(). Marks
 * kernel-radius dirty for the GPU SIDX tracker so the next gpu_ngb_list_build
 * call sees fresh compact_h. drift_particle's early-return on time1==time0
 * dedupes naturally.
 *
 * NOT thread-safe with concurrent drift / tree mutation. Caller serializes.
 */
void mode_b_lazy_drift_candidates(const int *indices, int n);

#endif /* MODE_B_LOCAL_WALKER_H */
