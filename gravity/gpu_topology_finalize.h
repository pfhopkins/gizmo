/* gpu_topology_finalize.h — Step 13 Phase 6.6
 *
 * Final stage of GPU tree-build: replaces the sibling / father / Father[]
 * outputs of force_update_node_recursive with three GPU passes.
 *
 *   gpu_topology_finalize_father()    Writes soa->father[k] for every
 *                                     internal node and Father[i] for every
 *                                     particle child in one kernel that
 *                                     scans soa->suns_backup.
 *
 *   gpu_topology_finalize_sibling()   Writes soa->sibling[k] for every
 *                                     internal node by walking up the father
 *                                     chain to find the next occupied
 *                                     sibling-or-aunt slot.  Pre-condition:
 *                                     finalize_father has run.
 *
 *   gpu_topology_writeback_d_to_aos() Pushes soa->sibling and soa->father
 *                                     back into Nodes_base[k].u.d.{sibling,
 *                                     father} for legacy CPU walks.  Clobbers
 *                                     u.suns via the union — safe because
 *                                     soa->suns_backup is the SoA truth.
 *
 * All three are no-ops on the non-GPU build path; force_update_node_recursive
 * still runs there.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_TOPOLOGY_FINALIZE_H
#define GIZMO_GPU_TOPOLOGY_FINALIZE_H

#ifdef OPENMP_GPU_OFFLOAD

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 6.6a: father pass.  Writes:
 *   - soa->father[k] for k in [0, n) (n = Numnodestree).  Index k=0 (root)
 *     gets -1.
 *   - Father[i] for i in [0, MaxPart) (UVM-mirrored host array).
 * Pre-condition: soa->suns_backup populated for [0, n).  Returns 0 on
 * success. */
int gpu_topology_finalize_father(int n);

/* Phase 6.6b: sibling pass.  Writes soa->sibling[k] for k in [0, n).
 * Pre-condition: finalize_father has run (reads soa->father).  Returns 0
 * on success. */
int gpu_topology_finalize_sibling(int n);

/* Phase 6.6c: AoS writeback for d.sibling/d.father.  Host loop over
 * [0, n).  Idempotent.  Returns 0 on success. */
int gpu_topology_writeback_d_to_aos(int n);

/* Phase 6.8f: GPU kernel that resets per-node ephemeral fields (GravCost,
 * Ti_current, dp, Ti_lastkicked, Flag, optional payloads) after topology
 * finalize and before moment_refresh accumulates fresh moments.  Operates
 * on the absolute Nodes[] index range [MaxPart .. MaxPart+n).  On the CPU
 * path FUNR does this work inline; on the GPU path FUNR is retired
 * (Phase 6.6) so this kernel takes its place.  Returns 0 on success. */
int gpu_node_reset_ephemeral(int n);

/* Phase 6.6: Father[] is UVM (SharedSpace) on GPU build so the GPU father
 * kernel can write into it directly and host readers page-fault on touch.
 * These wrappers keep the kokkos_malloc/free out of forcetree.cc (non-GPU TU
 * has no Kokkos include). */
int  *gpu_father_alloc(int maxpart);   /* returns NULL on failure */
void  gpu_father_free(int *p);

/* Phase 6.8: extend the same UVM-allocator pattern to Nodes_base, Extnodes_base
 * and Nextnode.  After 6.8d/e all four tree-storage arrays live in SharedSpace,
 * so GPU kernels can read/write them directly and the Phase-6 host writeback
 * loops over node-count are replaced with single-pass GPU kernels (Phase 6.8f).
 *
 * NODE / extNODE struct definitions live in forcetree.h (which this TU
 * includes); the helpers take byte counts so callers stay struct-agnostic. */
void *gpu_tree_alloc_bytes(size_t bytes);  /* generic SharedSpace alloc */
void  gpu_tree_free_bytes(void *p);        /* matching free */

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_TOPOLOGY_FINALIZE_H */
