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

#ifdef OPENMP_GPU_OFFLOAD

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

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_PARTICLES_ARENA_H */
