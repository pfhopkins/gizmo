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

/* Bulk migration of the live particle storage (P[0,NumPart), CellP[0,N_gas)) between host and
 * device, for a PCIe-attached GPU whose managed memory is not host-coherent. There every phase that
 * sweeps the particles on the other side otherwise moves the pages one fault at a time; a prefetch
 * moves them in one pass at copy-engine bandwidth. Called where a sweep of the whole particle set
 * begins: _to_device before the bulk device phases (neighbour loops, tree build), _to_host before
 * the bulk host phases (writebacks, finalizes, kicks, domain). A pure placement hint: it never
 * changes what any code reads or writes, so a missing or extra call only costs time. No-op on
 * coherent systems (Grace-Hopper), on non-CUDA builds, and when GIZMO_UVM_PREFETCH=0 is set in the
 * environment. */
void gizmo_particles_prefetch_to_device(const char *site);
void gizmo_particles_prefetch_to_host(const char *site);
/* As _to_device for the rank's own particles [0, n_owned) and their cells only, leaving imported ghosts in place. */
void gizmo_particles_prefetch_owned_to_device(int n_owned, const char *site);
/* As _to_device for P only, for a device phase that reads no cell data (the tree build). */
void gizmo_particles_prefetch_P_to_device(const char *site);
/* As _to_host for P only, after a device phase that touched no cell data. */
void gizmo_particles_prefetch_P_to_host(const char *site);
/* Where the last bulk prefetch sent the particle storage: 1 the device, 2 the host, 0 none yet or
 * prefetching off. A hint for code that can run either side, to run where the particles are. */
int gizmo_particles_prefetch_side(void);
/* Whether a host phase about to touch n particles right after a device pass should move the arrays back in bulk
 * first: for a sweep of most of them, and after any smaller device pass since the last such move, which then
 * carries only the pages that pass pulled over (the host would otherwise fault them back one at a time).
 * GIZMO_UVM_TOHOST_ALWAYS=0 in the environment keeps only the first case, for a same-binary comparison. */
int gizmo_particles_back_to_host_wanted(long long n);
/* The same bulk move for one other managed buffer about to be swept on the host (e.g. a kernel's
 * per-active results read back by a host writeback). Same conditions and no-op cases as above. */
void gizmo_managed_prefetch_to_host(const void *ptr, size_t bytes, const char *site);
/* The same for one managed buffer about to be swept on the device. */
void gizmo_managed_prefetch_to_device(const void *ptr, size_t bytes, const char *site);
/* Leaving the rank's own particles on the device across a short host interval that touches no
 * particle storage of its own, so the next device pass finds them resident instead of costing a
 * to-host and a to-device bulk move.  _keep_owned_on_device is asked where a to-host prefetch would
 * otherwise be issued: it returns 1 (the caller skips the prefetch) only when bulk prefetching is on,
 * the last bulk prefetch went to the device, and GIZMO_UVM_KEEP_DEVICE is not 0 in the environment.
 * The state lasts until the next bulk particle prefetch in either direction; while it lasts,
 * _owned_kept_on_device returns 1 and host code in the interval must read particles through the
 * device helpers below (or take the to-host prefetch itself).  Placement only: a host touch in the
 * interval stays correct and only costs page faults. */
int gizmo_particles_keep_owned_on_device(const char *site);
int gizmo_particles_owned_kept_on_device(void);
/* Device-side helpers for that interval.  _count_not_at_time: how many of idx[0,n) (host array; entries
 * outside [0,NumPart) ignored) have P[j].Ti_current != t_now.  _gather_send_records: the ghost-export
 * pack (whole P[j] record, CellP[j] for gas below N_gas, else zeros; idx[k] < 0 gives a zero slot),
 * gathered on the device and copied into the host buffers dst_P[0,n) / dst_CellP[0,n). */
long long gizmo_particles_count_not_at_time(const int *idx, int n, long long t_now);   /* t_now: an integertime */
void gizmo_particles_gather_send_records(const int *idx, int n, struct particle_data *dst_P, struct gas_cell_data *dst_CellP);
/* While the rank's own particles are kept on the device: move the imported ghosts' records (P and cells at
 * [n_owned, NumPart)) to the host for a host phase that reads or writes only them, leaving the rest in place. */
void gizmo_particles_prefetch_ghosts_to_host(int n_owned, const char *site);
/* A device-only copy of the live particle storage for a device pass that writes none of it except P[].wakeup
 * and (MFV) CellP[].dMass, which _mirror_copy_back_writes returns to the storage afterwards: the host keeps its
 * pages, so no to-host move follows the pass.  _mirror_fill copies P[0,n_part) and the cells over the range a
 * bulk prefetch would cover and returns the device pointers, or returns 0 (use the storage itself) when the
 * mirror is off (GIZMO_UVM_MIRROR=0), prefetching is off, or the buffer cannot be had. */
int gizmo_particles_mirror_fill(int n_part, struct particle_data **P_dev, struct gas_cell_data **CellP_dev, const char *site);
void gizmo_particles_mirror_copy_back_writes(void);
/* Seconds and bytes spent in the prefetches so far (for the run's timing report). */
void gizmo_particles_prefetch_stats(double *seconds, double *bytes_to_device, double *bytes_to_host, long long *calls);

/* Device-resident tree-to-density span (GIZMO_GPU_TREEHOST).  On a step that rebuilds the tree over most of the
 * particles and then runs a density pass over most of them, the tree build moves P to the device and the host work
 * between the build and the density pass (the OldAcc promotion, the per-type hmax seed, zeroing GravAccel, the
 * active-gas counts, the density active list, its radius clamp and the runner's radii) used to pull P back to the
 * host only for the density pass to push it to the device again.  With the span that work runs as device kernels and
 * P stays on the device until the density pass, whose own bulk move then carries only the cells.
 *  _begin(n_active): asked by the tree build before it starts; arms the span (returns 1) when GIZMO_GPU_TREEHOST is
 *     not 0 in the environment, the configuration is one whose host work in that interval has a device port
 *     (GIZMO_GPU_TREEHOST_SUPPORTED), bulk prefetching is on and n_active is at least 1/8 of the particles.
 *  _armed(): between _begin and _disarm; the OldAcc promotion and the hmax seed take their device form.
 *  _open(): the hmax seed ran on the device and P was left there; host code until the density pass reads and
 *     writes P only through the device helpers below while _span_active() returns 1.
 *  _cells_to_device(site): the density pass's bulk move of the cells, taken early by a device kernel that reads them.
 *  _consume(site): at a density pass's bulk move to the device; returns 1 when the span leaves both arrays resident
 *     (moving the cells first if they are not yet), so the caller skips its own move.  Ends the span.
 *  _end(site): ends a span nothing consumed, moving P back to the host.
 * Any bulk move of P to the host also ends it.  Placement only: a host touch missed while the span lasts costs page
 * faults, never a wrong value. */
int  gizmo_treehost_begin(long long n_active);
int  gizmo_treehost_armed(void);
void gizmo_treehost_disarm(void);
void gizmo_treehost_open(void);
int  gizmo_treehost_span_active(void);
void gizmo_treehost_cells_to_device(const char *site);
int  gizmo_treehost_consume(const char *site);
void gizmo_treehost_end(const char *site);
/* Device forms of the host work in the span.  Each is the host loop it replaces with the same reads and writes
 * (copies, zeros, compares and integer counts only).  idx is a host array of particle indices.
 *  _copy_oldacc: P[i].OldAcc = P[i].OldAcc_LatestWalk for i in [0, n).
 *  _zero_gravaccel: P[idx[k]].GravAccel = 0.
 *  _count_active_gas: how many of idx[0,n) are gas, and gas with positive mass. */
void gizmo_particles_copy_oldacc_on_device(int n);
void gizmo_particles_zero_gravaccel_on_device(const int *idx, int n);
void gizmo_particles_count_active_gas_on_device(const int *idx, int n, long long *n_gas_with_mass, long long *n_gas);
/* Grow-only device scratch for the span's kernels in other translation units: `slot` 0..3 are independent buffers. */
void *gizmo_treehost_device_scratch(int slot, size_t bytes);

/* Imported ghosts kept on the device.  The hydro corridor's ghost pool can be delivered straight from the MPI
 * receive (page-locked) to the device, and kept there until the pool is cleaned up, instead of being written on
 * the host and moved across with every bulk prefetch of the corridor's device passes.
 *   _enabled: whether this run does so (rank-uniform: build + environment).  Compiled where the hydro pass's ghost
 *     writeback is the wakeup maximum alone; on by default only for the corridor configurations audited for host
 *     reads of the ghost range, GIZMO_GPU_GHOSTDEV=0/1 in the environment overrides either way.
 *   _set_live(base, n): the ghost range [base, base+n) now lives on the device; n <= 0 clears it.  While it is set,
 *     bulk to-host prefetches leave that range where it is.
 *   _copy_to_device: n records from the host into P/CellP[base, base+n), on the device (destination prefetched
 *     there first); src_CellP may be NULL.  0 on success; 1 on a failed copy, after which the caller copies on
 *     the host.
 *   _pinned_buffer: grow-only page-locked scratch of the ghost transport (slots 0/1 receive P/CellP, 2/3 send
 *     P/CellP, 4/5 send/receive trimmed records, 6 discovery candidates); never zeroed; NULL when the memory cannot be had.
 *   _zero_wakeup / _gather_wakeup: P[j].wakeup over the ghost range, set to 0 / copied into dst[0,n), on the device. */
int gizmo_ghostdev_enabled(void);
void gizmo_ghostdev_set_live(int base, int n);
int gizmo_ghostdev_live(void);
int gizmo_ghost_range_copy_to_device(int base, int n, const struct particle_data *src_P, const struct gas_cell_data *src_CellP);
void *gizmo_ghost_pinned_buffer(int slot, size_t bytes);
void gizmo_ghost_range_zero_wakeup_on_device(int base, int n);
void gizmo_ghost_range_gather_wakeup(int base, int n, int *dst);
/* The trimmed ghost-export pack on the device, for owners kept there: record k (rec_bytes, zeroed first) holds
 * segment s (words[s] 4-byte words) of P[idx[k]] (cell[s] == 0) or, for gas below N_gas, of CellP[idx[k]], from byte
 * obj_off[s] to rec_off[s]; idx[k] < 0 leaves it zero.  Copied into dst (host memory).  At most 8 segments.  0 on the
 * device; 1 when it packed on the host instead (correct, slower). */
int gizmo_particles_gather_send_segments(const int *idx, int n, void *dst, int rec_bytes, int nseg,
                                         const int *cell, const int *obj_off, const int *rec_off, const int *words);
/* The reverse for a few fields: segment s of P (cell[s] == 0) or CellP[base + g], bytes[s] bytes from byte obj_off[s],
 * into record g of records (rec_bytes each, host memory) at rec_off[s], read on the device.  At most 8 segments. */
void gizmo_ghost_range_gather_records(int base, int n, void *records, int rec_bytes, int nseg,
                                      const int *cell, const int *obj_off, const int *rec_off, const int *bytes);
/* n packed records of rec_bytes each (host memory) into P/CellP[base, base+n), on the device: segment s of a record,
 * words[s] 4-byte words at rec_off[s], lands at byte obj_off[s] of the slot's P (cell[s] == 0) or CellP record.  At
 * most 8 segments.  0 on the device; 1 when it fell back to the host (correct, slower). */
int gizmo_ghost_range_scatter_records(int base, int n, const void *records, int rec_bytes, int nseg,
                                      const int *cell, const int *obj_off, const int *rec_off, const int *words);
/* Seconds and bytes of the ghost copies to the device so far. */
void gizmo_ghostdev_stats(double *seconds, double *bytes, long long *calls);

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
 * <Kokkos_Core.hpp> directly.
 *
 * `particle_arena_bytes` is the total per-rank size of the bulk particle record arrays
 * (P plus CellP) that this buffer belongs to, or 0 for a buffer that is not one of them.
 * It selects the placement policy for the managed pages; see the policy block in
 * gpu_particles_arena.cc for what is chosen and why. Passing 0 leaves the allocation with
 * whatever placement the runtime would give it. */
void *gpu_particles_uvm_alloc(size_t nbytes, const char *label, size_t particle_arena_bytes);

/* Release a buffer from gpu_particles_uvm_alloc. Paired with it so a capacity change can hold
 * the old and the new buffer at once and roll back cleanly. NULL is a no-op. */
void gpu_particles_uvm_free(void *ptr);

/* Grow-only device buffers for per-call transients, one per site, kept until gizmo_gpu_pool_release (at
 * finalize). A device allocation made and freed on every call of a hot path costs driver calls that can wait
 * on the device (0.1-1 ms each measured on PCIe H100 under MPS). _take returns at least `bytes` of device
 * memory for the site (contents unspecified, not zeroed) or NULL; _give returns it, and callers give a buffer
 * back only after fencing the kernels that use it. A site already taken (a nested call), an unknown site, or
 * GIZMO_GPU_POOL_VIEWS=0 in the environment gets an ordinary allocation that _give frees, as before. */
enum gizmo_gpu_pool_site {
    GIZMO_POOL_NGL_SCRATCH = 0,   /* gpu_ngb_list_build: per-active neighbour scratchpad */
    GIZMO_POOL_NGL_COUNTS,        /* gpu_ngb_list_build: per-active neighbour counts */
    GIZMO_POOL_MODEA_ACTIVES,     /* run_mode_a: staged ActiveData */
    GIZMO_POOL_SIDX_RAISE,        /* sidx_raise_members: staged raise records */
    GIZMO_POOL_SIDX_SCRATCH,      /* the index build's working blocks, one site per block: 16 sites from here */
    GIZMO_POOL_NSITES = GIZMO_POOL_SIDX_SCRATCH + 16
};
void *gizmo_gpu_pool_take(int site, size_t bytes, const char *label);
void  gizmo_gpu_pool_give(int site, void *p);
void  gizmo_gpu_pool_release(void);

/* Accessors. Return NULL / 0 when arena is not currently held. */
struct particle_data *gpu_particles_arena_P(void);
struct gas_cell_data *gpu_particles_arena_CellP(void);
int gpu_particles_arena_capacity(void);
int gpu_particles_arena_valid(void);

#ifdef __cplusplus
}
#endif

/* The configurations the device-resident tree-to-density span is written for: hydro MFM with no self-gravity and
 * none of the physics that adds host work on the particles between the tree build and the density pass (analytic
 * gravity, sinks, star formation and feedback, grains, RT sources, AGS, magnetic fields, transport subcycling).  The
 * density active-list predicate's device form is exact only here.  Evaluated only after allvars.h (which brings the
 * Config macros); elsewhere the span is compiled out and the host path stays. */
#if defined(ALLVARS_H) && defined(GPU_PCIE_OPTIMIZATIONS) && defined(SELFGRAVITY_OFF) && defined(HYDRO_MESHLESS_FINITE_MASS) \
    && !defined(SUBFIND) && !defined(PMGRID) && !defined(HERMITE_INTEGRATION) \
    && !defined(GRAVITY_ANALYTIC) && !defined(GRAVITY_ANALYTIC_HARMONIC) && !defined(STARFORGE_GMC_TURBINIT) \
    && !defined(STARFORGE_FILAMENT_TURBINIT) && !defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES) \
    && !defined(BOX_SHEARING) && !defined(GRAIN_RDI_TESTPROBLEM) && !defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE) \
    && !defined(SINK_PARTICLES) && !defined(SINK_INTERACT_ON_GAS_TIMESTEP) && !defined(GALSF) \
    && !defined(GALSF_FB_MECHANICAL) && !defined(GALSF_FB_THERMAL) && !defined(GRAIN_FLUID) \
    && !defined(RT_SOURCE_INJECTION) && !defined(DO_DENSITY_AROUND_NONGAS_PARTICLES) && !defined(DM_DISPERSION_LOOP_ACTIVE) \
    && !defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE) && !defined(MAGNETIC) && !defined(TRANSPORT_SUBCYCLE)
#define GIZMO_GPU_TREEHOST_SUPPORTED 1
#endif


#endif /* GIZMO_GPU_PARTICLES_ARENA_H */
