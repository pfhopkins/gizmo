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
 * GIZMO_MODE_B_ORACLE=1: every call to the tree walk also runs the
 * brute walk and asserts equality (prints diff to stderr on mismatch).
 *
 * Design constraints (codex 2026-05-07):
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

#ifdef __cplusplus
extern "C" {
#endif

/* search_mode constants — match NGB_SEARCH_ONEWAY/SYMMETRIC in
 * mesh/ghost_exchange_spec.h to keep the existing NGL contract. */
#ifndef MODE_B_SEARCH_ONEWAY
#define MODE_B_SEARCH_ONEWAY    0
#define MODE_B_SEARCH_SYMMETRIC 1
#endif

int mode_b_enabled(void);

/* Tree-walk path. Fast for spatially-localized queries. Returns count of
 * local-real-particle candidates (P[] indices) appended to out_candidates;
 * -1 if out_capacity is exceeded. Does NOT sort. */
int mode_b_local_neighbor_walk(const double pos[3],
                               double h_q,
                               unsigned int type_mask,
                               int search_mode,
                               int *out_candidates,
                               int out_capacity);

/* Brute-force path. Iterates 0..num_local. Slow but obviously correct.
 * Used as the same-run oracle. Same return contract. Does NOT sort. */
int mode_b_local_brute_walk(const double pos[3],
                            double h_q,
                            unsigned int type_mask,
                            int search_mode,
                            int *out_candidates,
                            int out_capacity);

/* Oracle helper: returns 1 if env GIZMO_MODE_B_ORACLE=1 is set. The
 * walker entry point checks this and runs both paths + diff on mismatch. */
int mode_b_oracle_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* MODE_B_LOCAL_WALKER_H */
