/* Mode B local neighbor walker — host-side range-walk over the existing
 * gravity tree (Nodes[]/Nextnode[]/sibling). Designed to answer remote
 * Mode B queries from a peer rank's local particle set without touching
 * GPU SIDX state.
 *
 * NOT yet wired in. Skeleton lands Day 1 of Phase 0; real implementation
 * lives behind GIZMO_MODE_B_DENSITY=1.
 *
 * Design constraints (codex 2026-05-07):
 *   - Walk uses Nodes[]/Nextnode[] for pruning. Node bounds come from
 *     force_drift_node(); they are conservative for current Ti_Current.
 *   - Returned candidates are LOCAL real particles (P[j] indices), never
 *     LET/foreign tree state. Local rank is the authority for its own
 *     particles.
 *   - Particle drift is the CALLER's responsibility. The walker does NOT
 *     drift P[j] before returning indices. Caller must match the lazy-
 *     drift policy of the existing NGL path before evaluating predicates.
 */

#ifndef MODE_B_LOCAL_WALKER_H
#define MODE_B_LOCAL_WALKER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Find local particles whose KernelRadius/position bounds intersect a
 * spherical query of radius h_q centered at pos. Pruning uses gravity-tree
 * node bounds; candidates are real local P[] indices.
 *
 * type_mask: bitmask of allowed P[].Type values (1<<type for each allowed).
 * search_mode: NGB_SEARCH_ONEWAY (r < h_q) or NGB_SEARCH_SYMMETRIC
 *   (r < max(h_q, h_j)). For SYMMETRIC the walker must consult P[j].KernelRadius.
 *
 * Writes up to out_capacity candidate indices into out_candidates and
 * returns the actual count. Returns -1 on overflow (caller must grow buffer
 * and retry). Returns 0 if no candidates.
 *
 * NOT thread-safe with concurrent particle drift / tree mutation. Caller
 * must serialize against move_particles, force_treebuild, domain decomp.
 */
int mode_b_local_neighbor_walk(const double pos[3],
                               double h_q,
                               unsigned int type_mask,
                               int search_mode,
                               int *out_candidates,
                               int out_capacity);

/* Returns 1 if Mode B is enabled this run (GIZMO_MODE_B_DENSITY=1). Static
 * — read once at first call and cached. */
int mode_b_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* MODE_B_LOCAL_WALKER_H */
