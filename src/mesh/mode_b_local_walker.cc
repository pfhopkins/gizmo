/* Mode B local neighbor walker — host-side range-walk + brute-force oracle.
 *
 * See header for design constraints. Both paths return LOCAL real P[]
 * indices in [0, ghost_get_num_local()) intersecting (pos, h_q).
 *
 * SYMMETRIC tree walk prunes internal nodes by the per-type hmax bands
 * (Extnodes[no].hmax_per_type via mode_b_node_symmetric_radius); ONEWAY
 * prunes by h_q alone. The brute walk is the oracle for both.
 *
 * The three public tree walks (mode_b_local_neighbor_walk / _walk_and_export /
 * _walk_from_start_nodes) are thin wrappers over the one shared traversal body
 * mode_b_walk_impl below — the same SSOT-body/many-wrappers shape legacy uses
 * (ngb.cc's 8 ngb_treefind_* over one codeblock).
 *
 * Periodic boundaries: handled via NEAREST_XYZ on the displacement.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"
#include "ghost_writeback.h"      /* ghost_get_num_local */
#include "gpu_neighbor_list.h"    /* gizmo_mark_kernel_radius_dirty_indices */
#include "mode_b_local_walker.h"

/* Public helper: per-j symmetric radius under policy. Delegates to the SSOT
 * per-particle wrapper in nlr_radius_policy.h, which owns the
 * AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE compile-flag gate. */
double mode_b_neighbor_symmetric_radius(int j, mode_b_radius_policy_t policy)
{
    return nlr_particle_symmetric_radius(P[j], policy);
}

/* Predicate: does P[j] satisfy the query (pos, h_q) under search_mode? */
/* j_radius_scale: SYMMETRIC-mode multiplier on the j-side kernel radius
 * (1.0 = legacy). TURB_DIFF_DYNAMIC wide-filter loops pass
 * All.TurbDynamicDiffFac so the Mode B reach matches the Mode A scaled-
 * symmetric NGL. See OPEN_3d_difffilter_design.md §3. */
static inline int particle_passes(int j,
                                  const double pos[3],
                                  double h_q,
                                  unsigned int type_mask,
                                  int search_mode,
                                  mode_b_radius_policy_t radius_policy,
                                  double j_radius_scale)
{
    if(!(type_mask & (1u << P[j].Type))) return 0;
    if(P[j].Mass <= 0) return 0;
    double dx = (double)P[j].Pos[0] - pos[0];
    double dy = (double)P[j].Pos[1] - pos[1];
    double dz = (double)P[j].Pos[2] - pos[2];
    NEAREST_XYZ(dx, dy, dz, 1);
    double r2 = dx*dx + dy*dy + dz*dz;
    double cutoff = h_q;
    if(search_mode == MODE_B_SEARCH_SYMMETRIC) {
        /* Type-aware: only use h_j when it's physically meaningful for
         * j's type under the active radius_policy. For non-gas types
         * without AGS opt-in, h_j contribution is 0 and the test
         * collapses to ONEWAY r < h_q for that j. (Phil + codex 2026-05-07.)
         */
        double hj = mode_b_neighbor_symmetric_radius(j, radius_policy) * j_radius_scale;
        if(hj > cutoff) cutoff = hj;
    }
    return r2 < cutoff * cutoff;
}

void mode_b_local_brute_walk(const double pos[3],
                             double h_q,
                             unsigned int type_mask,
                             int search_mode,
                             mode_b_radius_policy_t radius_policy,
                             std::vector<int>& out,
                             double j_radius_scale)
{
    const int num_local = ghost_get_num_local();
    for(int j = 0; j < num_local; j++) {
        if(!particle_passes(j, pos, h_q, type_mask, search_mode, radius_policy, j_radius_scale)) continue;
        out.push_back(j);
    }
}

/* Sphere-vs-AABB pruning test. Returns 1 if the sphere of radius R
 * centered at pos overlaps the AABB defined by node center + half-len.
 * Uses NEAREST_XYZ for periodic minimum image. */
static inline int sphere_aabb_overlap(const double pos[3],
                                      const struct NODE *nop,
                                      double R)
{
    double dx = (double)nop->center[0] - pos[0];
    double dy = (double)nop->center[1] - pos[1];
    double dz = (double)nop->center[2] - pos[2];
    NEAREST_XYZ(dx, dy, dz, 1);
    /* Legacy per-axis AABB reject at R + 0.5*len (ngb_codeblock_checknode.h normal
     * branch: if(NGB_PERIODIC_BOX_LONG_* > R+0.5*len) continue, per axis). The node box
     * spans center +- 0.5*len per axis; a single-axis separation > R+0.5*len means the
     * query sphere (radius R) cannot reach the box, so no overlap. Conservative — a
     * neighbor inside the node forces box/sphere intersection, so this never drops a real
     * neighbor; it removes the corner-case nodes the enclosing-sphere bound alone opens. */
    const double half = R + 0.5 * (double)nop->len;
    if(fabs(dx) > half || fabs(dy) > half || fabs(dz) > half) return 0;
    /* Legacy enclosing-sphere test at R + CUBE_EDGEFACTOR_1*len = R + 0.866*len
     * (0.5 + 0.366025 = sqrt(3)/2 = SQRT3_OVER_2) — byte-identical to legacy's radial. */
    const double SQRT3_OVER_2 = 0.86602540378443864676;
    double r_max = R + (double)nop->len * SQRT3_OVER_2;
    return (dx*dx + dy*dy + dz*dz) < r_max * r_max;
}

/* Mode B SYMMETRIC effective radius for an internal node. Returns
 *   max over types-in-mask of Extnodes[no].hmax_per_type[t]
 * inflated by a slack factor. Per the invariant in allvars.h, each band is
 * already a conservative upper bound across every leaf-policy-selectable
 * radius source for that type — so node-prune does not need to know the
 * caller's radius_policy. Leaf-level predicate (mode_b_neighbor_symmetric_radius)
 * applies the exact policy. Over-opening here is safe (extra candidates
 * filter at the leaf); under-opening would be a correctness bug.
 *
 * Returns 0 if no requested type has a populated band (degenerates to ONEWAY
 * pruning, which is still correct given the leaf-level filter).
 *
 * Node-open slack: between force_update_hmax() refreshes, per-particle
 * KernelRadius can grow under drift — up to ~exp(divv_fac_max/NUMDIMS) ≈ 1.105
 * per refresh for gas, more for AGS-active — while node hmax decays under a
 * different (looser) clamp. Legacy carried NO extra node-open slack: with
 * force_update_hmax refreshed every step, the within-step gas growth is covered
 * by the enclosing-sphere 0.866*len node term, so no inflation is needed. Set
 * to 0 to match legacy. Over-search is safe (extra candidates filter at the
 * leaf); under-search is a correctness bug — full-oracle membership on the
 * downsampled m11i confirmed 0 lost neighbors under real FIRE (gas) drift at
 * slack 0. AGS-active builds (larger per-step growth) rely on the same
 * per-step refresh cadence and are not independently oracle-checked here;
 * re-verify with the oracle if a SYMMETRIC Mode-B loop runs in an AGS config. */
static constexpr double MODE_B_NODE_H_SLACK = 0.0;  /* no node-open drift slack; legacy has none (relies on force_update_hmax cadence + 0.866*len node term) */

static inline double mode_b_node_symmetric_radius(int no,
                                                  unsigned int type_mask,
                                                  double j_radius_scale)
{
    double rmax = 0.0;
    for(int t = 0; t < 6; t++) {
        if(!(type_mask & (1u << t))) continue;
        double v = (double)Extnodes[no].hmax_per_type[t];
        if(v > rmax) rmax = v;
    }
    return rmax * (1.0 + MODE_B_NODE_H_SLACK) * j_radius_scale;
}

/* Which node band the SYMMETRIC open test uses:
 *  - LegacyScalar: the cross-rank scalar Extnodes.hmax (DomainMoment-exchanged;
 *    forcetree.cc:766/886). The SENDER export walk MUST use this so its
 *    targeting covers REMOTE ranks' h_j — the per-type bands are rank-local
 *    (allvars.h) so they under-cover remote peers.
 *  - PerTypeLocal: the rank-local per-type bands (mode_b_node_symmetric_radius),
 *    fresh + tight for THIS rank's own pool. Used by local + receiver walks.
 * Scalar hmax is gas-biased: the runner only
 * takes the export path for policies it dominates (gate in neighbor_loop_runner.cc). */
enum class ModeBNodeBand { PerTypeLocal, LegacyScalar };

/* Build the topleaf reverse map from the DomainNodeIndex SSOT (never from a
 * slot-layout assumption). Sized to the max observed offset; entries outside
 * any topleaf stay -1. O(NTopleaves), rebuilt per export call (topnode indices
 * are stable between tree builds; per-call rebuild avoids any staleness). */
void ModeBExportCollector::build_topleaf_map(void)
{
    const int max_part = All.MaxPart;
    int max_off = -1;
    for(int i = 0; i < NTopleaves; i++) {
        const int off = DomainNodeIndex[i] - max_part;
        if(off < 0) { topnode_map_size = 0; return; }   /* malformed map: disable topleaf detection (walk still exports via pseudo branch) */
        if(off > max_off) max_off = off;
    }
    topnode_map_size = max_off + 1;
    leaf_of_topnode.assign(topnode_map_size, -1);
    for(int i = 0; i < NTopleaves; i++) {
        leaf_of_topnode[DomainNodeIndex[i] - max_part] = i;
    }
}

/* Shared traversal body (SSOT for all three public tree walks).
 *
 * Walks from `start_no`. Local real-particle matches are appended to `cand_out`
 * (nullptr = do not collect). At every remote pseudo-node reached, if
 * `export_out != nullptr` the owner peer + node's DomainNodeIndex is recorded
 * (legacy mode==0 targeted export, ngb_codeblock_after_condition_unthreaded.h:
 * 19-67; modern symbol form matches gravity/forcetree.cc:2173-2210). If
 * `stop_at_toplevel` (legacy mode==1 receiver), the walk returns when it
 * re-enters the top-level tree — the exported subtree is exhausted
 * (after_condition_unthreaded.h:74-81).
 *
 * SYMMETRIC internal-node pruning uses the per-type hmax bands
 * (mode_b_node_symmetric_radius); ONEWAY prunes by h_q alone. Bands are
 * rank-local and re-seeded every build/refresh; a query against another rank's
 * pool is shipped there and answered with that rank's own fresh bands, so no
 * cross-rank band exchange is needed. */
static void mode_b_walk_impl(const double pos[3],
                             double h_q,
                             unsigned int type_mask,
                             int search_mode,
                             mode_b_radius_policy_t radius_policy,
                             double j_radius_scale,
                             int start_no,
                             bool stop_at_toplevel,
                             ModeBNodeBand band,
                             std::vector<int>* cand_out,
                             ModeBExportCollector* export_out)
{
    if(All.MaxPart <= 0 || Nodes == NULL || Nextnode == NULL) return;
    const int num_local = ghost_get_num_local();
    const int max_part  = All.MaxPart;
    const int pseudo_start = max_part + MaxNodes + MaxForeignNodes;
    const int oneway = (search_mode == MODE_B_SEARCH_ONEWAY);

    int no = start_no;

    while(no >= 0) {
        if(no < max_part) {
            /* Particle leaf. Only return domain-owned local particles (not a
             * ghost import). cand_out==nullptr on a pure export-discovery walk. */
            if(cand_out && no < num_local &&
               particle_passes(no, pos, h_q, type_mask, search_mode, radius_policy, j_radius_scale)) {
                cand_out->push_back(no);
            }
            no = Nextnode[no];
        } else if(no < pseudo_start) {
            /* Internal node. */
            struct NODE *nop = &Nodes[no];
            /* Receiver (legacy mode==1): re-entering the top-level tree means
             * the exported branch is done. */
            if(stop_at_toplevel && (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))) return;
            /* Drift if stale, then prune. */
            if(nop->Ti_current != All.Ti_Current) {
#ifdef _OPENMP
#pragma omp critical(_modebdrift_)
#endif
                {
                    if(nop->Ti_current != All.Ti_Current) {
                        force_drift_node(no, All.Ti_Current);
                    }
                }
            }
            double R_eff;
            if(oneway) {
                R_eff = h_q;
            } else {
                /* SYMMETRIC: R_eff = max(h_q, node band). LegacyScalar uses the
                 * cross-rank scalar hmax (sender export — must cover remote h_j);
                 * PerTypeLocal uses the rank-local per-type bands (local/receiver
                 * walks). Over-opening is safe; under-opening is a correctness bug. */
                double node_h;
                if(band == ModeBNodeBand::LegacyScalar) {
                    node_h = (double)Extnodes[no].hmax * (1.0 + MODE_B_NODE_H_SLACK) * j_radius_scale;
                } else {
                    node_h = mode_b_node_symmetric_radius(no, type_mask, j_radius_scale);
                }
                R_eff = (node_h > h_q) ? node_h : h_q;
            }
            int do_open = sphere_aabb_overlap(pos, nop, R_eff);
            /* SENDER export event (legacy pseudo-hit equivalence): the walk
             * decided to OPEN a remote-owned TOPLEAF. Legacy would descend and
             * hit the pseudo child (export + skip); post-LET the child may be
             * the imported foreign subtree instead — which holds NO local
             * candidates and must NOT be descended in source-export semantics.
             * Export (owner task, DomainNodeIndex) and skip to sibling. The
             * open test just applied uses the SAME cross-rank-exchanged scalar
             * hmax + node geometry legacy applied to this topleaf, so the
             * exported set matches legacy's exactly. */
            if(do_open && export_out) {
                const int leaf = export_out->topleaf_of(no, max_part);
                if(leaf >= 0 && DomainTask[leaf] != ThisTask) {
                    export_out->add(DomainTask[leaf], DomainNodeIndex[leaf]);
                    no = nop->u.d.sibling;
                    continue;
                }
            }
            if(do_open) {
                const int child = nop->u.d.nextnode;
                /* Legacy foreign-subtree skip: in a local-candidate walk from root
                 * (self-collect / broadcast-fallback receiver), a remote topleaf's child
                 * is the imported foreign subtree (post-LET rewire), whose nodes are in
                 * [MaxPart+MaxNodes, pseudo_start) and hold NO owned-local P[] candidates.
                 * Descending it is wasted opens; legacy skipped remote subtrees in the
                 * local walk too. Take the sibling instead. The targeted receiver walk
                 * (owner's own subtree) never has foreign children so this never fires
                 * there; the export-discovery walk (cand_out==nullptr) skips remote
                 * topleaves earlier. */
                if(cand_out && !export_out &&
                   child >= max_part + MaxNodes && child < pseudo_start) {
                    no = nop->u.d.sibling;
                } else {
                    no = child;
                }
            } else {
                no = nop->u.d.sibling;
            }
        } else {
            /* Pseudo-particle node (cross-rank subtree root; reached only for
             * remote topleaves the LET did not ship/redirect — with the
             * topleaf-boundary export above this branch is normally never hit,
             * but it keeps non-LET / unshipped-leaf configs correct). Record
             * the same targeted export, then skip forward exactly as the
             * legacy force walkers: the pseudo index is shifted by the
             * local-node + foreign-node reservation (Phase-9 layout). */
            if(export_out) {
                const int leaf = no - pseudo_start;
                if(leaf >= 0 && leaf < NTopleaves && DomainTask[leaf] != ThisTask) {
                    export_out->add(DomainTask[leaf], DomainNodeIndex[leaf]);
                }
            }
            no = Nextnode[no - MaxNodes - MaxForeignNodes];
        }
    }
}

/* LOCAL-candidates walk (legacy-mode==0 without export): from root, collect
 * local matches, skip pseudo-nodes. Behavior byte-identical to the prior broadcast-only
 * walker (export_out=nullptr, stop_at_toplevel=false). */
void mode_b_local_neighbor_walk(const double pos[3],
                                double h_q,
                                unsigned int type_mask,
                                int search_mode,
                                mode_b_radius_policy_t radius_policy,
                                std::vector<int>& out,
                                double j_radius_scale)
{
    mode_b_walk_impl(pos, h_q, type_mask, search_mode, radius_policy, j_radius_scale,
                     /*start_no=*/All.MaxPart, /*stop_at_toplevel=*/false,
                     ModeBNodeBand::PerTypeLocal, &out, /*export_out=*/nullptr);
}

/* SENDER walk: from root, record targeted exports at reached pseudo-nodes;
 * optionally also collect local candidates (cand_out may be nullptr). Uses the
 * cross-rank scalar hmax band (LegacyScalar) so targeting covers remote h_j. */
void mode_b_walk_and_export(const double pos[3],
                            double h_q,
                            unsigned int type_mask,
                            int search_mode,
                            mode_b_radius_policy_t radius_policy,
                            std::vector<int>* cand_out,
                            ModeBExportCollector& exporter,
                            double j_radius_scale)
{
    mode_b_walk_impl(pos, h_q, type_mask, search_mode, radius_policy, j_radius_scale,
                     /*start_no=*/All.MaxPart, /*stop_at_toplevel=*/false,
                     ModeBNodeBand::LegacyScalar, cand_out, &exporter);
}

/* RECEIVER walk: resume from each exported start-node (open its children like
 * legacy density.cc:272), stop at the top-level boundary. Multiple entries
 * (legacy NodeList) cover disjoint exported subtrees. */
void mode_b_walk_from_start_nodes(const double pos[3],
                                  double h_q,
                                  unsigned int type_mask,
                                  int search_mode,
                                  mode_b_radius_policy_t radius_policy,
                                  const int *node_list,
                                  int n_nodes,
                                  std::vector<int>& out,
                                  double j_radius_scale)
{
    if(All.MaxPart <= 0 || Nodes == NULL || Nextnode == NULL) return;
    const int max_part      = All.MaxPart;
    const int pseudo_start  = max_part + MaxNodes + MaxForeignNodes;
    for(int k = 0; k < n_nodes; k++) {
        const int nl = node_list[k];
        if(nl < 0) break;   /* -1 terminator (legacy NodeList convention) */
        /* Defensive: a start-node arrives over MPI; it must be an internal
         * node index. Skip a corrupt entry rather than dereference OOB. */
        if(nl < max_part || nl >= pseudo_start) continue;
        const int start = Nodes[nl].u.d.nextnode;   /* open the exported node */
        mode_b_walk_impl(pos, h_q, type_mask, search_mode, radius_policy, j_radius_scale,
                         start, /*stop_at_toplevel=*/true,
                         ModeBNodeBand::PerTypeLocal, &out, /*export_out=*/nullptr);
    }
}

/* Lazy-drift Mode B candidates to current Ti before the pair kernel reads
 * P[j] / CellP[j]. Mirrors gpu_ngb_list_build:1542-1580 contract. */
void mode_b_lazy_drift_candidates(const int *indices, int n)
{
    if(!indices || n <= 0) return;
    const int num_local = ghost_get_num_local();
    integertime time1 = All.Ti_Current;
    for(int k = 0; k < n; k++) {
        int j = indices[k];
        if(j >= 0 && j < num_local) {
            /* drift_particle's early-return on time1==time0 makes repeat
             * calls a fast no-op (e.g. j touched by an earlier query in
             * this same evaluator call). */
            drift_particle(j, time1);
        }
    }
    /* drift_particle mutates Pos and KernelRadius (the *= exp(divv_fac/N)
     * factor in predict.cc:160,229). The next gpu_ngb_list_build call (for
     * non-Mode-B callers) needs to refresh compact_h from these freshly
     * drifted KernelRadius values. */
    gizmo_mark_kernel_radius_dirty_indices(indices, n);
}

/* Per-call same-rank tree-vs-brute oracle is owned by the runner via
 * GIZMO_NLR_ORACLE in mesh/neighbor_loop_runner.cc::run_mode_b_*_with_oracle.
 * The legacy public wrapper mode_b_local_walk_with_oracle() and its
 * GIZMO_MODE_B_ORACLE env helper were retired in this commit; both had
 * zero callers post-3c.3. */
