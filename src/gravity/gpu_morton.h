/* gpu_morton.h -- Step 13 Phase 6.5a/6.5c0
 *
 * Morton encoding + parallel sort infrastructure for the GPU tree-build
 * insertion kernel (Phase 6.5c2+).  The Phase 6.5 build flow is:
 *
 *   1. Compute a 126-bit Morton key per local particle from its global
 *      domain-bbox-relative position (this header / TU).
 *   2. For each topleaf, sort the local particles assigned to it by their
 *      Morton key (this header / TU).
 *   3. LCP-analyse the sorted keys to find octree split points
 *      (Phase 6.5b -- gpu_morton_functions.h).
 *   4. Emit tree topology into the SoA `Nodes_dev` mirror (Phase 6.5c3).
 *
 * The Morton encoding mirrors GIZMO's CPU build precision: 42 bits per axis
 * via the IEEE-754 mantissa-bit-trick (matches host domain_double_to_int
 * with BITS_PER_DIMENSION = 42).  Keys are stored as the Morton128 struct
 * defined in gpu_morton_functions.h: a (hi, lo) pair of uint64_t,
 * lexicographically equivalent to the full 126-bit Morton ordering.
 *
 * Particles whose positions land within a Morton cell of size below GIZMO's
 * collocation threshold (EPSILON_FOR_TREERND_SUBNODE_SPLITTING * split_scale)
 * will share keys and fall through to the RNG branch in 6.5c4, matching CPU
 * semantics.
 *
 * Scratch storage lives in static file-scope state, allocated lazily and
 * grown on demand.  Released via gpu_morton_release() at shutdown.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_MORTON_H
#define GIZMO_GPU_MORTON_H

#include <stdint.h>


#include "gpu_morton_functions.h"   /* for Morton128 type used by accessor */

#ifdef __cplusplus
extern "C" {
#endif

/* Compute 128-bit Morton keys for particles [0..npart) from P_dev[i].Pos
 * using the global domain bbox (DomainCorner, DomainLen).  Allocates /
 * grows the internal SharedSpace key buffer as needed.  Caller must have
 * an active gpu_particles_arena (P_dev populated).  Returns 0 on success.
 *
 * Edge handling: positions outside [DomainCorner, DomainCorner+DomainLen]
 * clamp to the unit interval, mapping to the nearest 42-bit cell.  This
 * matches GIZMO's expectation that particles outside the domain box are
 * pre-wrapped by domain decomp before reaching the tree build. */
int gpu_morton_compute_global_keys(int npart);

/* Acquire a writable handle to the internal Morton key buffer, ensuring
 * capacity for at least `npart` entries.  Allocates / grows on demand.
 * Returns a SharedSpace pointer suitable for kernel writes (e.g. from
 * gpu_topology_build's combined Peano+Morton compute).  Returns NULL on
 * allocation failure. */
struct Morton128 *gpu_morton_keys_acquire(int npart);

/* Sort `indices_inout[0..count)` in-place by Morton key, where the key for
 * a given index `i = indices_inout[j]` is the previously computed
 * gpu_morton_keys()[i].  `indices_inout` must point to SharedSpace memory.
 *
 * Uses Kokkos::Experimental::sort_by_key with the Morton128Less comparator
 * overload.  Stable + deterministic (lexicographic comparator falls back
 * to merge-sort on the CUDA backend rather than radix; same correctness
 * guarantees, slightly slower constant).
 *
 * Returns 0 on success, nonzero on failure. */
int gpu_morton_sort_indices(int count, int *indices_inout);

/* Read-only accessor for the internal Morton key buffer.  Returns NULL if
 * gpu_morton_compute_global_keys() has not yet been called.  The buffer
 * is sized to hold at least `npart` entries from the most recent compute
 * call; callers indexing past that are out of bounds. */
const struct Morton128 *gpu_morton_keys(void);

/* Free the internal SharedSpace key buffer.  Idempotent. */
void gpu_morton_release(void);

#ifdef __cplusplus
}
#endif


#endif /* GIZMO_GPU_MORTON_H */
