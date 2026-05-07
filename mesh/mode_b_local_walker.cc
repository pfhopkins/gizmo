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

int mode_b_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *env = getenv("GIZMO_MODE_B_DENSITY");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached;
}

int mode_b_oracle_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *env = getenv("GIZMO_MODE_B_ORACLE");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached;
}

/* Public helper: per-j symmetric radius under policy. See header. */
double mode_b_neighbor_symmetric_radius(int j, mode_b_radius_policy_t policy)
{
    /* Gas: KernelRadius represents finite-volume cell extent. Always
     * physically meaningful for gas. */
    if(P[j].Type == 0) {
        if(policy & MODE_B_RADIUS_GAS_KERNEL) return (double)P[j].KernelRadius;
        return 0.0;
    }
    /* Non-gas: P[j].KernelRadius is NOT physical extent (it's stale-IC or
     * the density-search radius for THIS particle's gas neighbors, not
     * its own physical kernel). For SYMMETRIC pair searches the correct
     * source is AGS_KernelRadius, used only by AGS/SIDM/grain physics. */
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if(policy & MODE_B_RADIUS_AGS_FOR_NONGAS) {
        return (double)P[j].AGS_KernelRadius;
    }
#endif
    /* Default for non-gas: 0 → SYMMETRIC degenerates to ONEWAY for this j. */
    return 0.0;
}

/* Predicate: does P[j] satisfy the query (pos, h_q) under search_mode? */
static inline int particle_passes(int j,
                                  const double pos[3],
                                  double h_q,
                                  unsigned int type_mask,
                                  int search_mode,
                                  mode_b_radius_policy_t radius_policy)
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
        double hj = mode_b_neighbor_symmetric_radius(j, radius_policy);
        if(hj > cutoff) cutoff = hj;
    }
    return r2 < cutoff * cutoff;
}

int mode_b_local_brute_walk(const double pos[3],
                            double h_q,
                            unsigned int type_mask,
                            int search_mode,
                            mode_b_radius_policy_t radius_policy,
                            int *out_candidates,
                            int out_capacity)
{
    const int num_local = ghost_get_num_local();
    int count = 0;
    for(int j = 0; j < num_local; j++) {
        if(!particle_passes(j, pos, h_q, type_mask, search_mode, radius_policy)) continue;
        if(count >= out_capacity) return -1;
        out_candidates[count++] = j;
    }
    return count;
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

int mode_b_local_neighbor_walk(const double pos[3],
                               double h_q,
                               unsigned int type_mask,
                               int search_mode,
                               mode_b_radius_policy_t radius_policy,
                               int *out_candidates,
                               int out_capacity)
{
    if(All.MaxPart <= 0 || Nodes == NULL || Nextnode == NULL) return 0;
    const int num_local = ghost_get_num_local();
    const int max_part  = All.MaxPart;
    const int last_node = max_part + Numnodestree;

    /* SYMMETRIC mode without per-node max-h: must always open internal
     * nodes (cannot prune by h_q alone). Falls through to behaviour ~=
     * brute walk over all leaves; oracle will catch any mismatch. */
    const int prune_internal = (search_mode == MODE_B_SEARCH_ONEWAY);

    int count = 0;
    int no = max_part; /* root node */

    while(no >= 0) {
        if(no < max_part) {
            /* Particle leaf. Only return if it's a domain-owned local
             * particle (not a ghost import). */
            if(no < num_local && particle_passes(no, pos, h_q, type_mask, search_mode, radius_policy)) {
                if(count >= out_capacity) return -1;
                out_candidates[count++] = no;
            }
            no = Nextnode[no];
        } else if(no < last_node) {
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
            int do_open;
            if(prune_internal) {
                do_open = sphere_aabb_overlap(pos, nop, h_q);
            } else {
                do_open = 1;  /* SYMMETRIC: always open until per-node max-h tracked */
            }
            no = do_open ? nop->u.d.nextnode : nop->u.d.sibling;
        } else {
            /* Foreign pseudo-particle node (LET / cross-rank). Codex
             * constraint: never include in candidate list. Skip via
             * the pseudo-particle's sibling (held in the underlying
             * NODE struct still indexable via Nodes[no]). */
            no = Nodes[no].u.d.sibling;
        }
    }
    return count;
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

/* Sorted bit-exact diff of two index lists. Returns 0 if equal, prints
 * details to stderr and returns 1 if different. */
static int diff_index_lists(const int *a, int na, const int *b, int nb,
                            const char *label)
{
    int *as = (int*)malloc(na > 0 ? na * sizeof(int) : 1);
    int *bs = (int*)malloc(nb > 0 ? nb * sizeof(int) : 1);
    if(na > 0) memcpy(as, a, na * sizeof(int));
    if(nb > 0) memcpy(bs, b, nb * sizeof(int));
    std::sort(as, as + na);
    std::sort(bs, bs + nb);
    int eq = (na == nb);
    if(eq) for(int i = 0; i < na; i++) if(as[i] != bs[i]) { eq = 0; break; }
    if(!eq) {
        fprintf(stderr, "[mode_b ORACLE MISMATCH %s rank=%d] tree=%d brute=%d ",
                label, ThisTask, na, nb);
        if(na <= 16 && nb <= 16) {
            fprintf(stderr, "tree_idx={");
            for(int i = 0; i < na; i++) fprintf(stderr, "%s%d", i?",":"", as[i]);
            fprintf(stderr, "} brute_idx={");
            for(int i = 0; i < nb; i++) fprintf(stderr, "%s%d", i?",":"", bs[i]);
            fprintf(stderr, "}");
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    free(as); free(bs);
    return eq ? 0 : 1;
}

/* Public oracle-wrapped entry: if GIZMO_MODE_B_ORACLE=1, runs both paths
 * and diffs. Returns the tree-walk count (treats tree as authoritative
 * once oracle has passed for a session). */
extern "C" int mode_b_local_walk_with_oracle(const double pos[3],
                                             double h_q,
                                             unsigned int type_mask,
                                             int search_mode,
                                             int *out_candidates,
                                             int out_capacity)
{
    /* Default policy for the generic wrapper: gas-only KernelRadius for
     * SYMMETRIC. Callers needing AGS opt in via the per-walker entry points. */
    const mode_b_radius_policy_t policy = MODE_B_RADIUS_DEFAULT;
    int n_tree = mode_b_local_neighbor_walk(pos, h_q, type_mask, search_mode,
                                            policy, out_candidates, out_capacity);
    if(mode_b_oracle_enabled()) {
        const int CAP = 4096;
        int *brute = (int*)malloc(CAP * sizeof(int));
        int n_brute = mode_b_local_brute_walk(pos, h_q, type_mask, search_mode,
                                              policy, brute, CAP);
        if(n_brute < 0) {
            fprintf(stderr, "[mode_b ORACLE] brute capacity exceeded (>%d); "
                            "skipping diff for this call\n", CAP);
            fflush(stderr);
        } else if(n_tree < 0) {
            fprintf(stderr, "[mode_b ORACLE] tree capacity exceeded (>%d); "
                            "skipping diff for this call\n", out_capacity);
            fflush(stderr);
        } else {
            diff_index_lists(out_candidates, n_tree, brute, n_brute, "walk");
        }
        free(brute);
    }
    return n_tree;
}
