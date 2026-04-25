/* gpu_pseudo_update.h — Step 13 Phase 6.7
 *
 * GPU/SoA-native replacements for the three CPU end-game stages that run
 * after gpu_topology_writeback_d_to_aos in force_treebuild (and in
 * force_refresh_node_moments):
 *
 *   6.7a  gpu_force_flag_localnodes()    replaces force_flag_localnodes()
 *   6.7b  gpu_scatter_pseudo_to_soa()    post-exchange AoS→SoA scatter
 *   6.7c  gpu_topnode_moment_resum()     replaces force_treeupdate_pseudos()
 *
 * After all three are wired up the gpu_gravity_tree_mark_all_dirty() call
 * at the end of force_treebuild (forcetree.cc:206) is removed: the SoA
 * stays authoritative throughout without a full reseed.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_PSEUDO_UPDATE_H
#define GIZMO_GPU_PSEUDO_UPDATE_H

#ifdef OPENMP_GPU_OFFLOAD

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 6.7a: Bitflag pass.  Three GPU parallel_for waves up father_soa:
 *   1. BITFLAG_TOPLEVEL              from each DomainNodeIndex[i]
 *   2. BITFLAG_INTERNAL_TOPLEVEL     from parent of each DomainNodeIndex[i]
 *   3. BITFLAG_DEPENDS_ON_LOCAL_ELEMENT  from each ThisTask-owned topleaf
 * Then mirrors bitflags_soa[0..NTopnodes) → Nodes[].u.d.bitflags (AoS) so
 * that force_exchange_pseudodata (still CPU in 6.7a/b) can read correct
 * bitflags from AoS.  Replaces force_flag_localnodes().  Returns 0 on
 * success. */
int gpu_force_flag_localnodes(void);

/* Phase 6.7b: After force_exchange_pseudodata() has scattered foreign-rank
 * pseudo moments into AoS Nodes[], copy them into the corresponding SoA
 * slots so the SoA is authoritative for all topleaf nodes.  O(NTopleaves)
 * CPU loop.  Returns 0 on success. */
int gpu_scatter_pseudo_to_soa(void);

/* Phase 6.7c: Re-sum ancestor topnode moments directly in SoA after the
 * foreign pseudo data is in place (i.e. after gpu_scatter_pseudo_to_soa).
 * Recursive CPU walk of the topnode tree (all NTopnodes nodes; typically
 * O(100-1000) so CPU is fine).  Reads/writes SoA bitflags, s, mass,
 * node_vs, hmax, vmax, divVmax, N_part, maxsoft + all conditional payloads.
 * Replaces force_treeupdate_pseudos().  Returns 0 on success. */
int gpu_topnode_moment_resum(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_PSEUDO_UPDATE_H */
