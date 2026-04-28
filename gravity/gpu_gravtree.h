/* gpu_gravtree.h — Step 13 Phase 4 Tier 1a
 *
 * Core gravity tree walk on GPU (no optional payloads).
 *
 * Runs BEFORE the OpenMP-parallelized CPU primary loop (gravity_primary_loop)
 * as an opportunistic accelerator. For each active particle:
 *   - GPU thread walks the local tree using the Phase-3 SoA mirror.
 *   - If it encounters a pseudo-particle (remote node owned by another rank),
 *     it sets failed[i]=1 and exits. The host then leaves ProcessedFlag[i]
 *     unset so the CPU primary loop handles it (with the existing MPI export
 *     machinery, unchanged).
 *   - If it completes successfully, it writes P[i].GravAccel and sets
 *     ProcessedFlag[i]=1 so the CPU loop skips it.
 *
 * Gated at compile time by OPENMP_GPU_OFFLOAD. PMGRID, ADAPTIVE_GRAVSOFT_*,
 * EVALPOTENTIAL, RT/SINK/SINGLE_STAR/CR/TIDAL/JERK payloads, periodic Ewald,
 * HERMITE/ATFU, and FIRE_BHS MencInRcrit are all supported via the Phase 9-10
 * LET work.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_GRAVTREE_H
#define GIZMO_GPU_GRAVTREE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Run the GPU pre-pass over all active particles.
 *
 * Called from gravity_tree() before the primary_loop OpenMP fan-out. Updates
 * P[i].GravAccel and ProcessedFlag[i] for successful particles. Particles
 * that hit a pseudo-particle (remote node) are left untouched, so the
 * existing CPU primary loop handles them as usual.
 *
 * Returns the number of particles that completed successfully on GPU. When
 * the build lacks OPENMP_GPU_OFFLOAD, this is an unconditional no-op
 * (returns 0) so callers in gravity_tree() stay simple. */
/* Phase 7.b: GPU gravity-cost assignment.  Replaces the host Father-chain walk
 * at gravtree.cc:487-495 that accumulates Nodes[].GravCost into
 * P[i].GravCost[takelevel] for each particle.  Zeros the slot inline before
 * accumulating (so the CPU zeroing at gravtree.cc:145 is skipped on GPU builds).
 * All inputs UVM (Father Phase 6.6, Nodes Phase 6.8d, P Phase 1).
 * No-op stub when OPENMP_GPU_OFFLOAD is not defined. */
void gpu_assign_gravcost(int takelevel);

int gpu_gravtree_walk_primary(void);

/* GPU Ewald-correction walk. Called from gravity_tree() when Ewald_iter==1
 * (pure-tree periodic, BOX_PERIODIC && !GRAVITY_NOT_PERIODIC && !PMGRID).
 * Mirrors force_treeevaluate_ewald_correction mode=0: walks the local tree a
 * second time, accumulates the periodic-image correction via trilinear
 * interpolation of the fcorrx/y/z look-up tables, and adds the result to
 * P[i].GravAccel.
 *
 * Only the local tree walk runs on GPU. Pseudo-particle hits leave
 * ProcessedFlag unset, so the CPU Ewald secondary loop finishes those via
 * MPI export.  No-op when the build lacks OPENMP_GPU_OFFLOAD. Returns
 * number of successfully walked targets. */
int gpu_ewald_walk_primary(void);

#ifdef __cplusplus
}
#endif

#endif /* GIZMO_GPU_GRAVTREE_H */
