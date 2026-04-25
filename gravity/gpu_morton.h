/* gpu_morton.h — Step 13 Phase 6.5a
 *
 * Morton encoding + parallel sort infrastructure for the GPU tree-build
 * insertion kernel (Phase 6.5c).  The Phase 6.5 build flow is:
 *
 *   1. Compute a 63-bit Morton key per local particle from its global
 *      domain-bbox-relative position (this header / TU).
 *   2. For each topleaf, sort the local particles assigned to it by their
 *      Morton key (this header / TU).
 *   3. LCP-analyse the sorted keys to find octree split points (Phase 6.5b).
 *   4. Emit tree topology into the SoA `Nodes_dev` mirror (Phase 6.5c).
 *
 * The Morton encoding mirrors what GIZMO's CPU build already does internally
 * via peano_and_morton_key() / BITS_PER_DIMENSION, just packed into a
 * 64-bit word for sort efficiency.  Uses 21 bits per axis (63 bits total)
 * computed from (P[i].Pos - DomainCorner) / DomainLen.  Particles whose
 * positions land within a Morton cell of size below GIZMO's collocation
 * threshold (EPSILON_FOR_TREERND_SUBNODE_SPLITTING * split_scale) will share
 * keys and fall through to the RNG branch in 6.5c, matching CPU semantics.
 *
 * Scratch storage lives in static file-scope state, allocated lazily and
 * grown on demand.  Released via gpu_morton_release() at shutdown (or via
 * gpu_gravity_tree_release() if that's the cleanup hook).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_MORTON_H
#define GIZMO_GPU_MORTON_H

#include <stdint.h>

#ifdef OPENMP_GPU_OFFLOAD

#ifdef __cplusplus
extern "C" {
#endif

/* Compute 63-bit Morton keys for particles [0..npart) from P_dev[i].Pos using
 * the global domain bbox (DomainCorner, DomainLen).  Allocates / grows the
 * internal SharedSpace key buffer as needed.  Caller must have an active
 * gpu_particles_arena (P_dev populated).  Returns 0 on success.
 *
 * Edge handling: positions outside [DomainCorner, DomainCorner+DomainLen]
 * clamp to the unit interval, mapping to the nearest 21-bit cell.  This
 * matches GIZMO's expectation that particles outside the domain box are
 * pre-wrapped by domain decomp before reaching the tree build. */
int gpu_morton_compute_global_keys(int npart);

/* Sort `indices_inout[0..count)` in-place by Morton key, where the key for
 * a given index `i = indices_inout[j]` is the previously computed
 * gpu_morton_keys()[i].  `indices_inout` must point to SharedSpace memory
 * (caller-owned, e.g. via Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>).
 *
 * Uses Kokkos::Experimental::sort_by_key on a device-local scratch copy of
 * (keys, indices), then writes the sorted indices back to indices_inout.
 * The underlying sort is deterministic + stable (Thrust radix on CUDA,
 * std::sort fallback on host).
 *
 * Returns 0 on success, nonzero on failure. */
int gpu_morton_sort_indices(int count, int *indices_inout);

/* Read-only accessor for the internal Morton key buffer.  Returns NULL if
 * gpu_morton_compute_global_keys() has not yet been called.  The buffer
 * is sized to hold at least `npart` entries from the most recent compute
 * call; callers indexing past that are out of bounds. */
const uint64_t *gpu_morton_keys(void);

/* Free the internal SharedSpace key buffer.  Idempotent. */
void gpu_morton_release(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_MORTON_H */
