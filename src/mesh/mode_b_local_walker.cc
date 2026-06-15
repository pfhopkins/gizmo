/* Mode B local neighbor walker — host-side range-walk + brute-force oracle.
 *
 * See header for design constraints. Both paths return LOCAL real P[]
 * indices in [0, ghost_get_num_local()) intersecting (pos, h_q).
 *
 * Status: WORKING for ONEWAY mode (the density-iter target). SYMMETRIC
 * tree walk currently always opens (no per-node max-h tracking yet);
 * brute walk handles SYMMETRIC correctly. Oracle catches any divergence.
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
    /* Conservative bound: cube half-diagonal sqrt(3)/2 * len */
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
 * Slack rationale (matches mesh/gpu_neighbor_list.cc:SIDX_H_SLACK = 0.5):
 * between force_update_hmax() refreshes, per-particle KernelRadius can
 * grow under drift up to factor exp(divv_fac_max/NUMDIMS) ≈ 1.105 (gas)
 * or exp(4/3) ≈ 3.79 (AGS-active). Node hmax decays under a different
 * (looser) clamp and can fall BELOW the live max-particle radius. The
 * 50% inflation absorbs this asymmetry conservatively — over-search is
 * safe (extra candidates → physics kernel filters), under-search is a
 * correctness bug. Same convention as GPU NGL's BVH tile-overlap test. */
static inline double mode_b_node_symmetric_radius(int no,
                                                  unsigned int type_mask,
                                                  double j_radius_scale)
{
    static constexpr double H_SLACK = 0.5;  /* matches SIDX_H_SLACK */
    double rmax = 0.0;
    for(int t = 0; t < 6; t++) {
        if(!(type_mask & (1u << t))) continue;
        double v = (double)Extnodes[no].hmax_per_type[t];
        if(v > rmax) rmax = v;
    }
    return rmax * (1.0 + H_SLACK) * j_radius_scale;
}

void mode_b_local_neighbor_walk(const double pos[3],
                                double h_q,
                                unsigned int type_mask,
                                int search_mode,
                                mode_b_radius_policy_t radius_policy,
                                std::vector<int>& out,
                                double j_radius_scale)
{
    if(All.MaxPart <= 0 || Nodes == NULL || Nextnode == NULL) return;
    const int num_local = ghost_get_num_local();
    const int max_part  = All.MaxPart;
    const int pseudo_start = max_part + MaxNodes + MaxForeignNodes;

    /* SYMMETRIC pruning uses the per-type hmax bands maintained in extNODE
     * (see allvars.h). For each internal node:
     *   R_eff = max(h_query, max_{t in type_mask} Extnodes[no].hmax_per_type[t])
     * then sphere-vs-AABB overlap with R_eff. Per-type generalization of the
     * legacy scalar-hmax node prune. ONEWAY ignores hmax and uses h_query alone.
     *
     * The bands are rank-local by design: this walker returns only rank-local
     * candidates and skips pseudo/foreign nodes (handled at the bottom of the
     * loop), and the bands are re-seeded on every rank at every build/refresh.
     * A query against another rank's pool is shipped to that rank and answered
     * with its own locally-fresh bands, so no cross-rank band exchange is
     * required for correctness. */
    const int oneway = (search_mode == MODE_B_SEARCH_ONEWAY);

    int no = max_part; /* root node */

    while(no >= 0) {
        if(no < max_part) {
            /* Particle leaf. Only return if it's a domain-owned local
             * particle (not a ghost import). */
            if(no < num_local && particle_passes(no, pos, h_q, type_mask, search_mode, radius_policy, j_radius_scale)) {
                out.push_back(no);
            }
            no = Nextnode[no];
        } else if(no < pseudo_start) {
            /* Internal node — drift if stale, then prune. */
            struct NODE *nop = &Nodes[no];
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
                /* SYMMETRIC: R_eff = max(h_q, per-type hmax under mask).
                 * Bands are conservative across every radius_policy source
                 * for their type — node-prune is policy-independent. */
                double node_h = mode_b_node_symmetric_radius(no, type_mask, j_radius_scale);
                R_eff = (node_h > h_q) ? node_h : h_q;
            }
            int do_open = sphere_aabb_overlap(pos, nop, R_eff);
            no = do_open ? nop->u.d.nextnode : nop->u.d.sibling;
        } else {
            /* Pseudo-particle node (LET / cross-rank). These are not
             * addressable NODE slots in the Phase-9 layout. Match the
             * legacy force walkers: skip through the corresponding
             * Nextnode[] entry, whose pseudo index is shifted by the
             * local-node and foreign-node reservation. */
            no = Nextnode[no - MaxNodes - MaxForeignNodes];
        }
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
