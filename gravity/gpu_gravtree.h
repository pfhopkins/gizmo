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
 * Gated at compile time by GIZMO_GPU_GRAVTREE + OPENMP_GPU_OFFLOAD. The kernel
 * #errors out if any unsupported payload #ifdef is set (PMGRID, ADAPTIVE_GRAVSOFT_*,
 * EVALPOTENTIAL, RT_USE_GRAVTREE, SINK_*, COMPUTE_TIDAL_*, etc.). Those are
 * added in subsequent tier commits within Phase 4.
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
 * GIZMO_GPU_GRAVTREE is not defined or the build lacks OPENMP_GPU_OFFLOAD,
 * this is an unconditional no-op (returns 0) so callers in gravity_tree()
 * stay simple. */
int gpu_gravtree_walk_primary(void);

#ifdef __cplusplus
}
#endif

#endif /* GIZMO_GPU_GRAVTREE_H */
