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

#ifdef OPENMP_GPU_OFFLOAD

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
 * Returns 0 on success. */
int gpu_topology_build_data_path(int npart);

/* Read-only accessors.  Return NULL / 0 before data_path has run.
 * Buffers live in SharedSpace -- safe for both host and device reads. */
const int *gpu_topology_build_sorted_idx(void);
const int *gpu_topology_build_topleaf_start(void);   /* [NTopleaves + 1] */
const int *gpu_topology_build_topleaf_count(void);   /* [NTopleaves]     */
const int *gpu_topology_build_particle_topleaf(void);/* [npart] -- inverse */

/* Free internal SharedSpace scratch.  Idempotent. */
void gpu_topology_build_release(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_TOPOLOGY_BUILD_H */
