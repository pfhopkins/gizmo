/* gpu_peano_walk.h -- Step 13 Phase 6.5c1
 *
 * Host-side mirror lifecycle for the device Peano walk.  GPU kernels that
 * need to map particles to topleaves (Phase 6.5c2 onward) call into the
 * device-side primitives in gpu_peano_walk_functions.h, which read from
 * SharedSpace mirrors of TopNodes[] and DomainNodeIndex[].  This header
 * declares the host-side lifecycle:
 *
 *   - gpu_peano_walk_acquire(): copy host TopNodes[] / DomainNodeIndex[]
 *     into the SharedSpace mirrors.  Idempotent.  Cheap (NTopnodes is
 *     typically O(100) entries).  Should be called after force_create_empty_nodes
 *     (which populates DomainNodeIndex) and before any GPU build kernel.
 *
 *   - gpu_peano_walk_release(): free SharedSpace mirrors.  Called at
 *     shutdown / from gpu_gravity_tree_release().
 *
 *   - gpu_peano_walk_topnodes() / gpu_peano_walk_domain_node_index():
 *     accessors for the SharedSpace pointers.  Return NULL until acquire()
 *     has been called.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_PEANO_WALK_H
#define GIZMO_GPU_PEANO_WALK_H

#ifdef OPENMP_GPU_OFFLOAD

struct topnode_data;  /* forward decl; defined in declarations/allvars.h */

#ifdef __cplusplus
extern "C" {
#endif

/* Copy host TopNodes[0..NTopnodes) and DomainNodeIndex[0..NTopleaves) into
 * SharedSpace mirrors.  Allocates / grows mirrors on first call or capacity
 * growth.  Returns 0 on success. */
int gpu_peano_walk_acquire(void);

/* Free SharedSpace mirrors.  Idempotent. */
void gpu_peano_walk_release(void);

/* Accessors -- returns NULL if not yet acquired. */
const struct topnode_data *gpu_peano_walk_topnodes(void);
const int                 *gpu_peano_walk_domain_node_index(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_PEANO_WALK_H */
