/* gpu_particles_arena.h — Step 13 Phase 1
 *
 * Decomp-scoped persistent SharedSpace arrays for P[] and CellP[].
 * Replaces per-kernel transient kokkos_malloc/memcpy/free pairs in
 * full-neighbor GPU kernels (density, gradient, hydro, AGS, sinks, feedback,
 * turb, sidm, solids, RT-source, ngb-list).
 *
 * Lifetime: allocated on first acquire after domain decomp; reused across
 * all GPU kernels in the step; invalidated by domain decomp, merge/split,
 * and ghost-exchange writeback. Storage lives in GIZMO_KOKKOS_SHARED_SPACE
 * (= Kokkos::SharedSpace = CudaUVMSpace / HIPManagedSpace), so host writes
 * are visible to device kernels without explicit deep_copy.
 *
 * Batch-active kernels (cooling, nuclear, rt_chem) use compact 32 KB
 * allocators and intentionally bypass this arena.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_PARTICLES_ARENA_H
#define GIZMO_GPU_PARTICLES_ARENA_H

struct particle_data;
struct gas_cell_data;


#ifdef __cplusplus
extern "C" {
#endif

/* Acquire arena pointers, sized for at least min_capacity entries.
 * If a valid arena of sufficient capacity already exists, host data is
 * re-memcpy'd into it (cheap on UVM if pages are still device-resident).
 * Otherwise frees any existing arena, allocates fresh, and copies.
 *
 * min_capacity should be NumPart + N_ghosts (or a safe upper bound including
 * Step-8 adaptive ghost headroom). Caller passes the host P/CellP pointers
 * to seed the arena. */
void gpu_particles_arena_acquire(int min_capacity,
                                 struct particle_data *P_host,
                                 struct gas_cell_data *CellP_host);

/* Mark the arena stale without freeing. The next acquire() will re-memcpy
 * host data. Call from: domain decomp completion, merge_split after NumPart
 * change, ghost-exchange host-side writeback. */
void gpu_particles_arena_invalidate(void);

/* Phase 8a Round 1 — contract API for mirror-update call sites.
 *
 * A wrapper whose host postloop mutates P/CellP fields after the GPU kernel
 * returns has two options to keep the arena coherent:
 *   (a) gpu_particles_arena_invalidate()  — defensive; forces next acquire
 *       to slow-path memcpy the full N entries.
 *   (b) Mirror the same writes into the arena (write to BOTH P_host[i].field
 *       and arena_P[i].field), then call this function.  Asserts that arena
 *       now holds host-equivalent data for all touched fields/indices; the
 *       next acquire fast-paths.
 *
 * Optional GIZMO_GPU_ARENA_DEBUG=1 env var enables a byte-compare guard on
 * the next acquire that aborts if the (b) contract was violated, naming the
 * most recent site that asserted clean. Use during development of every new
 * mirror-update site, then disable for production runs. */
void gpu_particles_arena_mark_clean_after_scatter(const char *site);

/* Phase 8a Round 2 (option D): post-full-host-drift coherence point.
 *
 * Used by move_particles() (and any future full-N host mutator) to refresh
 * arena from current host state and mark clean, replacing the prior
 * invalidate-then-let-the-next-acquire-slow-path pattern. Costs one full
 * memcpy here, but unlocks fast-path acquires in all subsequent wrappers
 * for the rest of the step (when combined with Round 3 wrappers that
 * also mark_clean instead of invalidating defensively).
 *
 * Behavior:
 *   - If arena exists with capacity >= min_capacity: memcpy host->arena,
 *     set arena_valid_=1, record memcpy site for diagnostics.
 *   - Otherwise: fall back to invalidate (next acquire will allocate).
 *
 * site: caller tag for diagnostic output (e.g. "move_particles_post_drift").
 */
void gpu_particles_arena_refresh_from_host(int min_capacity,
                                           struct particle_data *P_host,
                                           struct gas_cell_data *CellP_host,
                                           const char *site);

/* DIAGNOSTIC: tag the upcoming acquire with a short descriptive string so
 * that GIZMO_GPU_ARENA_DEBUG mismatch messages identify the call site. */
void gpu_particles_arena_set_site(const char *site);

/* Free all SharedSpace storage. Called at shutdown. */
void gpu_particles_arena_release(void);

/* Accessors. Return NULL / 0 when arena is not currently held. */
struct particle_data *gpu_particles_arena_P(void);
struct gas_cell_data *gpu_particles_arena_CellP(void);
int gpu_particles_arena_capacity(void);
int gpu_particles_arena_valid(void);

#ifdef __cplusplus
}
#endif


#endif /* GIZMO_GPU_PARTICLES_ARENA_H */
