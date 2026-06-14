/* gpu_topology_build.h -- Step 13 Phase 6.5c2+
 *
 * GPU tree-build orchestration: assigns local particles to topleaves via
 * device-side Peano walk, computes 128-bit Morton keys, sorts particles
 * within each topleaf range, and (in 6.5c3+) emits internal-node topology
 * directly into the SoA `Nodes_dev` mirror.
 *
 * 6.5c2 implements the data path through the per-topleaf Morton sort.
 * Topology emission, collocation handling, and overflow retry follow in
 * 6.5c3 / 6.5c4.  Wiring into force_treebuild lands in 6.5d.
 *
 * Internal scratch lives in static SharedSpace buffers, reused across
 * tree builds.  Lifecycle: data path allocates / grows on first call;
 * gpu_topology_build_release() frees at shutdown.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_TOPOLOGY_BUILD_H
#define GIZMO_GPU_TOPOLOGY_BUILD_H


#ifdef __cplusplus
extern "C" {
#endif

/* 6.5c2 data path: for each particle in [0..npart) (= NumPart for the
 * caller), compute its (Peano, Morton) keys, walk TopNodes to find its
 * topleaf, bucket the particle into that topleaf's range in sorted_idx,
 * and Morton-sort within each range.  After this returns, the scratch
 * accessors below provide:
 *
 *   sorted_idx[topleaf_start[t] .. topleaf_start[t] + topleaf_count[t])
 *       -- particle indices in topleaf t, sorted by Morton key
 *   topleaf_start[NTopleaves] = npart   (end-of-buckets sentinel)
 *
 * Pre-conditions:
 *   - gpu_particles_arena_acquire() has been called (P_dev populated).
 *   - TopNodes / DomainNodeIndex are populated on host (i.e.
 *     force_create_empty_nodes has run).
 *
 * mp (optional): when non-NULL, build the tree over the subset P[mp[i].index]
 * for i in 0..npart-1 (e.g. SUBFIND per-species / collective unbinding). When
 * NULL, build over P[0..npart-1] (full/identity). The pipeline is slot-indexed;
 * the slot->particle map is owned by this TU's scratch and consumed through
 * gpu_topology_emit_bfs (freed by gpu_topology_build_release).
 *
 * Returns 0 on success. */
struct unbind_data;
int gpu_topology_build_data_path(int npart, const struct unbind_data *mp);

/* Read-only accessors.  Return NULL / 0 before data_path has run.
 * Buffers live in SharedSpace -- safe for both host and device reads. */
const int *gpu_topology_build_sorted_idx(void);
const int *gpu_topology_build_topleaf_start(void);   /* [NTopleaves + 1] */
const int *gpu_topology_build_topleaf_count(void);   /* [NTopleaves]     */
const int *gpu_topology_build_particle_topleaf(void);/* [npart] -- inverse */

/* Phase 6.5c3 BFS topology emission: take the Morton-sorted-per-topleaf
 * data laid out by gpu_topology_build_data_path and emit internal-node
 * topology (center, len, father, suns_backup) directly into the
 * gpu_gravity_tree SoA mirror.
 *
 * Pre-conditions:
 *   - gpu_topology_build_data_path has populated sorted_idx / topleaf_*.
 *   - gpu_gravity_tree_acquire has seeded SoA from CPU AoS for the topnode
 *     range (topleaves' geometry already there from force_create_empty_nodes
 *     + seed_from_aos).
 *   - Numnodestree (host) holds the post-topnode-skeleton end of the SoA;
 *     new GPU-built nodes start at this offset.
 *
 * On success: SoA holds the full inside-topleaf topology.  *new_node_count
 * is set to the new value of Numnodestree (= old value + #nodes emitted).
 *
 * Return codes:
 *    0   success
 *    1   MaxNodes overflow during emission (6.5c4 will add retry path)
 *    2   collocation detected: a sub-range of >1 particles share full LCP
 *        (= 126 bits).  6.5c4 will add the RNG-fallback branch.
 *    >=3 other failure (allocation, missing dependencies, etc.).
 *
 * `start_node_index` -- the SoA index at which the BFS may begin allocating
 * new internal nodes (= Numnodestree at call time). */
int gpu_topology_emit_bfs(int start_node_index, int *new_node_count_out);

/* Phase 6.5d helper: copy GPU-built topology (suns_backup, center, len)
 * back into the AoS Nodes_base[] array for the SoA index range
 * [first_soa_idx, last_soa_idx).  Required while force_update_node_recursive
 * still walks AoS u.suns to set sibling/father for the whole tree (Phase
 * 6.7+ retires force_update_node_recursive on the GPU compile path,
 * eliminating this writeback).
 *
 * Runs as an OMP host loop -- Nodes_base is host malloc, not SharedSpace.
 * The SoA fields live in SharedSpace UVM so reads incur first-touch page
 * faults but no explicit copy.  Cost is bounded by the number of GPU-built
 * inside-topleaf internal nodes and is one-time per tree-build.
 *
 * Returns 0 on success. */
int gpu_topology_writeback_to_aos(int first_soa_idx, int last_soa_idx);

/* Free internal SharedSpace scratch.  Idempotent. */
void gpu_topology_build_release(void);

#ifdef __cplusplus
}
#endif


#endif /* GIZMO_GPU_TOPOLOGY_BUILD_H */
