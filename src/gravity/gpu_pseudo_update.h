/* gpu_pseudo_update.h
 *
 * GPU/SoA-native replacements for the three CPU end-game stages that run
 * after gpu_topology_writeback_d_to_aos in force_treebuild (and in
 * force_refresh_node_moments):
 *
 *   6.7a  gpu_force_flag_localnodes()    replaces force_flag_localnodes()
 *   6.7b  gpu_scatter_pseudo_to_soa()    post-exchange AoS→SoA scatter
 *   6.7c  gpu_topnode_moment_resum()     replaces force_treeupdate_pseudos()
 *
 * The trailing gpu_gravity_tree_mark_all_dirty() call and the entire
 * mark_dirty/dirty_/seed_dirty_ machinery have been removed.
 * The SoA stays authoritative throughout: build kernels populate it, the
 * GPU drift kernel updates it in-place per stale-Ti_current node.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_PSEUDO_UPDATE_H
#define GIZMO_GPU_PSEUDO_UPDATE_H


#ifdef __cplusplus
extern "C" {
#endif

/* Bitflag pass.  Three GPU parallel_for waves up father_soa:
 *   1. BITFLAG_TOPLEVEL              from each DomainNodeIndex[i]
 *   2. BITFLAG_INTERNAL_TOPLEVEL     from parent of each DomainNodeIndex[i]
 *   3. BITFLAG_DEPENDS_ON_LOCAL_ELEMENT  from each ThisTask-owned topleaf
 * Then mirrors bitflags_soa[0..NTopnodes) → Nodes[].u.d.bitflags (AoS) so
 * that force_exchange_pseudodata (still CPU) can read correct
 * bitflags from AoS.  Replaces force_flag_localnodes().  Returns 0 on
 * success. */
int gpu_force_flag_localnodes(void);

/* After force_exchange_pseudodata() has scattered foreign-rank
 * pseudo moments into AoS Nodes[], copy them into the corresponding SoA
 * slots so the SoA is authoritative for all topleaf nodes.  O(NTopleaves)
 * CPU loop.  Returns 0 on success. */
int gpu_scatter_pseudo_to_soa(void);

/* After let_unpack_and_install copies a foreign sender's
 * NODE+extNODE bytes into the AoS foreign-node range, mirror the same
 * data into the SoA so the GPU walk can read foreign nodes via SoA.
 * `slot_base_abs` is the absolute Node index of the first foreign node
 * for this sender (= TreeNodeIndexBase + MaxNodes + Numforeignnodes_at_start_of_sender),
 * `count` is the number of consecutive foreign nodes to scatter.
 * Returns 0 on success. */
int gpu_scatter_foreign_to_soa(int slot_base_abs, int count);

/* Mirror a single AoS field write into the corresponding SoA
 * slot.  Used by let_unpack_and_install to redirect a local topleaf's
 * nextnode at the foreign subtree root inside SoA, since the topleaf's SoA
 * entry was last populated by the GPU build pipeline (which knew nothing
 * about LET).  abs_idx is the full Node index. */
int gpu_set_soa_nextnode(int abs_idx, int new_nextnode);

/* Sibling twin of gpu_set_soa_nextnode (same role for u.d.sibling).  Used by the
 * MAINTAIN_TREE_IN_REARRANGE pointer fixups to keep the SoA node thread coherent
 * with their AoS edits between full tree builds. */
int gpu_set_soa_sibling(int abs_idx, int new_sibling);

/* Re-sum ancestor topnode moments directly in SoA after the
 * foreign pseudo data is in place (i.e. after gpu_scatter_pseudo_to_soa).
 * Recursive CPU walk of the topnode tree (all NTopnodes nodes; typically
 * O(100-1000) so CPU is fine).  Reads/writes SoA bitflags, s, mass,
 * node_vs, hmax, vmax, divVmax, N_part, maxsoft + all conditional payloads.
 * Replaces force_treeupdate_pseudos().  Returns 0 on success. */
int gpu_topnode_moment_resum(void);

#ifdef __cplusplus
}
#endif


#endif /* GIZMO_GPU_PSEUDO_UPDATE_H */
