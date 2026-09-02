/* gpu_particles_arena.h
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
struct DriftKickTableView;

#include <stddef.h>  /* size_t */

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

/* Contract API for mirror-update call sites.
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

/* Post-full-host-drift coherence point.
 *
 * Used by move_particles() (and any future full-N host mutator) to refresh
 * arena from current host state and mark clean, replacing the prior
 * invalidate-then-let-the-next-acquire-slow-path pattern. Costs one full
 * memcpy here, but unlocks fast-path acquires in all subsequent wrappers
 * for the rest of the step (when combined with wrappers that also
 * mark_clean instead of invalidating defensively).
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

/* Compact staging for batched device work over a subset of particles.
 *
 * Used by any loop that runs whole-struct per-particle physics on the device for a
 * subset of the particles: gather the subset into compact buffers, run one kernel
 * over them, scatter the results back. Buffers are acquired once per call, outside
 * the batch loop, and released at the end of it.
 *
 * Why plain host memory and explicit device memory, rather than the SharedSpace the
 * arena above uses. Two measured reasons:
 *   - The host is the cheap reader of P and CellP: their pages sit host-resident
 *     precisely because the host touches them constantly, so a host-side gather reads
 *     them from the side they live on, while a device kernel reading them directly
 *     crosses the fabric.
 *   - On ROCm, any bulk copy whose SOURCE is managed memory takes a pathological path
 *     in the runtime, whereas a copy out of ordinary host memory does not.
 * The choice is therefore about placement, not about the memory space as such: a
 * private SharedSpace buffer performs well, an aliased long-lived one does not.
 *
 * Footprint while a call holds them: capacity * (sizeof(particle_data) +
 * sizeof(gas_cell_data)) for the host copy and again for the device copy -- both
 * copies, not one. At the 32768 batch cap that is about 202 MB per rank, which is why
 * they are not kept resident between calls: the measured allocate-and-free cost is a
 * small fraction of a second across a whole run, so holding the memory buys almost
 * nothing and costs it on every rank at once.
 *
 * CellP holds only All.MaxPartGas entries while an index list may run over every
 * particle type, so a cell is staged only for the gas subset. `cell` may be null when
 * the run has no gas at all -- a legal configuration that allocates no gas cell
 * storage -- and staging then proceeds with gas_count == 0; gas slots with a null
 * `cell` are refused instead. The gather partitions
 * the slots so gas comes first and reports how many there are: slot j always has a
 * particle, and has a cell exactly while j < gas_count -- the same condition every
 * cell access in the drift and cooling bodies is already guarded by. */
struct ParticleStagingBatch
{
    struct particle_data *host_P;
    struct gas_cell_data *host_Cell;
    struct particle_data *dev_P;
    struct gas_cell_data *dev_Cell;
    int *index;      /* particle index of each staged slot, gas slots first */
    int  capacity;
    int  count;      /* slots filled by the last gather */
    int  gas_count;  /* slots [0, gas_count) have a valid cell */
};

/* Obtain buffers for at least `capacity` slots. Zero-initialise the batch before the
 * first call -- `struct ParticleStagingBatch batch = {};` -- after which acquire and
 * release may be called in any order and any number of times: acquire releases
 * whatever the batch already held, so re-acquiring cannot strand the old buffers.
 *
 * Returns 0 if the memory could not be had. What the caller does then is its own
 * decision and the two existing callers differ: a drift MUST fall back to its host
 * loop, because skipping it leaves particles at the wrong position, whereas cooling
 * treats it as "no cooling on this rank" and carries on. The helper does not impose
 * either policy. */
int particle_staging_acquire(struct ParticleStagingBatch *batch, int capacity);

/* Release them. A zero-initialised or already-released batch is left alone. */
void particle_staging_release(struct ParticleStagingBatch *batch);

/* Gather pp[idx[0..n)] (and the cells of the gas among them) into the compact buffers
 * and copy them to the device. The particle arrays are passed in rather than taken
 * from the globals: this helper exists to move work into a different index space, and
 * a helper that silently reached for P[]/CellP[] would be the very confusion it is
 * meant to prevent.
 *
 * Threaded: the indices are distinct, so the fill has no write collisions.
 *
 * Returns 1 when the batch is staged and 0 when it refused, which happens only if the
 * caller asks to stage more elements than it acquired slots for -- a programming
 * error, so the refusal also requests a controlled stop. On a refusal NOTHING is
 * staged and `count` is zero: a partial gather would leave the caller running its
 * kernel over slots that were never written, and scattering back through indices that
 * were never written, which is an out-of-bounds write and not merely a wrong answer.
 * Callers must check, and must drive their own loops off `batch->count`. */
int particle_staging_gather(struct ParticleStagingBatch *batch, const int *idx, int n,
                            struct particle_data *pp, struct gas_cell_data *cell);

/* Copy the compact buffers back from the device and scatter them into the same arrays
 * the gather read. Threaded for the same reason. */
void particle_staging_scatter(struct ParticleStagingBatch *batch,
                              struct particle_data *pp, struct gas_cell_data *cell);

/* Device-visible mirror of the drift and gravitational-kick tables.
 *
 * The interpolator in core/timestep_functions.h reads the tables through a view of
 * plain pointers, so a device kernel needs the table bytes somewhere it can read.
 * Without relocatable device code a device symbol cannot be shared between
 * translation units, so the storage is per-TU: each caller keeps its own pointer and
 * passes it here, and this owns the allocate-and-fill policy so the two device drift
 * paths cannot drift apart in how they build the view.
 *
 * The refresh is unconditional by design: init() runs before init_drift_table() in
 * begrun, so a device drift issued during startup would otherwise cache the
 * still-zeroed tables and never recover. Two tables of DRIFT_TABLE_LENGTH doubles is
 * 16 KB, which is in the noise next to the work it serves.
 *
 * On a non-cosmological run the tables are never built and never read: nothing is
 * allocated, and the view selects the elapsed-time branch of the interpolator.
 *
 * `storage` is the caller's own pointer, zero-initialised before first use and left
 * owned by the caller. Returns nonzero if the mirror could not be provided, in which
 * case the caller must not launch; a controlled stop has already been requested. */
int drift_kick_table_mirror_refresh(double **storage, struct DriftKickTableView *view);

/* Free all SharedSpace storage. Called at shutdown. */
void gpu_particles_arena_release(void);

/* UVM-canonical particles: backing-storage allocator for P[] and CellP[].
 * Allocates `nbytes` of Kokkos::SharedSpace memory (CudaUVMSpace on GH200,
 * HIPManagedSpace on AMD).  Zeros the buffer so callers see the same
 * implicit-zero behavior they got from mymalloc.  Returns NULL on failure.
 * Called once at startup from system/allocate.cc; the returned pointer
 * persists until process exit.  This wrapper exists so allocate.cc — which
 * is compiled as a host (non-CUDA) TU — does not have to include
 * <Kokkos_Core.hpp> directly. */
void *gpu_particles_uvm_alloc(size_t nbytes, const char *label);

/* Release a buffer from gpu_particles_uvm_alloc. Paired with it so a capacity change can hold
 * the old and the new buffer at once and roll back cleanly. NULL is a no-op. */
void gpu_particles_uvm_free(void *ptr);

/* Accessors. Return NULL / 0 when arena is not currently held. */
struct particle_data *gpu_particles_arena_P(void);
struct gas_cell_data *gpu_particles_arena_CellP(void);
int gpu_particles_arena_capacity(void);
int gpu_particles_arena_valid(void);

#ifdef __cplusplus
}
#endif


#endif /* GIZMO_GPU_PARTICLES_ARENA_H */
