/* gpu_particles_arena.cc
 *
 * See gpu_particles_arena.h for design notes.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>
#include <exception>
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>   /* hipMemAdvise, for the particle-storage placement policy below */
#elif defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>      /* cudaMemAdvise, for the particle-storage placement policy below */
#endif

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/lifecycle_counters.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"
#include "gpu_particles_arena.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/ghost_writeback.h"   /* ghost_get_num_ghosts */


/* Under UVM-canonical particles, P[] and CellP[] live in
 * Kokkos::SharedSpace and the arena is a pure pointer alias. Acquire copies
 * pointers; invalidate / mark_clean / refresh / set_site are pure no-ops kept
 * as stable API for callers. The prior debug-guard infrastructure
 * (per-acquire serial counters, per-call-site tracking strings) was removed
 * since the byte-compare guard it served is unreachable under the alias
 * scheme. */
static struct particle_data *arena_P     = NULL;
static struct gas_cell_data *arena_CellP = NULL;
static int arena_capacity_ = 0;
static int arena_valid_    = 0;

extern "C" void gpu_particles_arena_set_site(const char *site) { (void)site; }

extern "C" void gpu_particles_arena_acquire(int min_capacity,
                                            struct particle_data *P_host,
                                            struct gas_cell_data *CellP_host)
{
    /* Tiny-N corridor counter: increments on API entry. Mode B paths in
     * run_neighbor_loop must NOT enter this function. See
     * declarations/lifecycle_counters.h. */
    g_gpu_arena_acquire_counter++;

    if(min_capacity <= 0) {min_capacity = 1;}
    arena_P         = P_host;
    arena_CellP     = CellP_host;
    arena_capacity_ = min_capacity;
    arena_valid_    = 1;
}

extern "C" void gpu_particles_arena_invalidate(void) {}

/* ---- bulk prefetch of the live particle storage (contract in gpu_particles_arena.h) ---- */
static int    prefetch_enabled_ = -1;   /* -1: not decided yet */
static int    prefetch_side_    = 0;    /* where the last bulk prefetch went: 1 device, 2 host */
static double prefetch_seconds_ = 0, prefetch_bytes_dev_ = 0, prefetch_bytes_host_ = 0;
static long long prefetch_calls_ = 0;
#if defined(KOKKOS_ENABLE_CUDA)
static int cuda_managed_memory_is_host_coherent(void);
#endif

static int particles_prefetch_enabled(void)
{
    if(prefetch_enabled_ < 0) {
        prefetch_enabled_ = 0;
#if defined(KOKKOS_ENABLE_CUDA) && defined(GPU_PCIE_OPTIMIZATIONS)
        if(std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value && !cuda_managed_memory_is_host_coherent()) {prefetch_enabled_ = 1;}
        const char *env = getenv("GIZMO_UVM_PREFETCH");
        if(env && env[0]) {prefetch_enabled_ = (atoi(env) != 0) && std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value;}
#endif
#if defined(GPU_PCIE_OPTIMIZATIONS)
        if(ThisTask == 0) {printf("Particle storage: bulk prefetch between host and device phases is %s.\n", prefetch_enabled_ ? "on" : "off"); fflush(stdout);}
#endif
    }
    return prefetch_enabled_;
}

static void particles_prefetch(int to_device, const char *site, int with_cells = 1)
{
    (void) site;
    if(!particles_prefetch_enabled()) {return;}
#if defined(KOKKOS_ENABLE_CUDA)
    const size_t bytes_p = (P && NumPart > 0) ? (size_t) NumPart * sizeof(struct particle_data) : 0;
    /* Imported ghosts sit at the end of [0, NumPart) in both arrays, past N_gas: cover them while the
       corridor holds any (unused cell slots between N_gas and the ghosts only cost the copy). */
    const int n_cells = (ghost_get_num_ghosts() > 0) ? ((NumPart < All.MaxPartGas) ? NumPart : All.MaxPartGas) : N_gas;
    const size_t bytes_c = (with_cells && CellP && n_cells > 0) ? (size_t) n_cells * sizeof(struct gas_cell_data) : 0;
    if(bytes_p + bytes_c == 0) {return;}
    const double t0 = my_second();
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess) {return;}
    const int dst = to_device ? dev : cudaCpuDeviceId;
    /* Ordered after all device work already queued (the device must be done with the pages
       before they go to the host), and complete before return (the host phase that follows
       touches them at once; a kernel launched after this finds them resident). */
    Kokkos::fence("gizmo_particles_prefetch");
    if(bytes_p) {cudaMemPrefetchAsync(P, bytes_p, dst, 0);}
    if(bytes_c) {cudaMemPrefetchAsync(CellP, bytes_c, dst, 0);}
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {(void) cudaGetLastError(); prefetch_enabled_ = 0; if(ThisTask == 0) {printf("Particle storage: bulk prefetch failed at %s (%s); switched off.\n", site, cudaGetErrorString(rc)); fflush(stdout);}}
    prefetch_side_ = prefetch_enabled_ ? (to_device ? 1 : 2) : 0;
    prefetch_seconds_ += timediff(t0, my_second());
    if(to_device) {prefetch_bytes_dev_ += (double) (bytes_p + bytes_c);} else {prefetch_bytes_host_ += (double) (bytes_p + bytes_c);}
    prefetch_calls_++;
#else
    (void) to_device;
#endif
}

extern "C" void gizmo_managed_prefetch_to_host(const void *ptr, size_t bytes, const char *site)
{
    if(!ptr || bytes == 0 || !particles_prefetch_enabled()) {return;}
#if defined(KOKKOS_ENABLE_CUDA)
    cudaPointerAttributes attr;
    if(cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {(void) cudaGetLastError(); return;}
    if(attr.type != cudaMemoryTypeManaged) {return;}   /* ordinary host or device memory: nothing to move */
    const double t0 = my_second();
    Kokkos::fence("gizmo_managed_prefetch_to_host");
    cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, 0);
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {(void) cudaGetLastError(); if(ThisTask == 0) {printf("Particle storage: buffer prefetch failed at %s (%s).\n", site, cudaGetErrorString(rc)); fflush(stdout);}}
    prefetch_seconds_ += timediff(t0, my_second());
    prefetch_bytes_host_ += (double) bytes;
    prefetch_calls_++;
#else
    (void) site;
#endif
}

extern "C" int gizmo_particles_prefetch_side(void) {return prefetch_side_;}
extern "C" void gizmo_particles_prefetch_to_device(const char *site) {particles_prefetch(1, site);}
extern "C" void gizmo_particles_prefetch_P_to_device(const char *site) {particles_prefetch(1, site, 0);}
extern "C" void gizmo_particles_prefetch_to_host(const char *site) {particles_prefetch(0, site);}
extern "C" void gizmo_particles_prefetch_P_to_host(const char *site) {particles_prefetch(0, site, 0);}
extern "C" void gizmo_particles_prefetch_stats(double *seconds, double *bytes_to_device, double *bytes_to_host, long long *calls)
{
    if(seconds) {*seconds = prefetch_seconds_;}
    if(bytes_to_device) {*bytes_to_device = prefetch_bytes_dev_;}
    if(bytes_to_host) {*bytes_to_host = prefetch_bytes_host_;}
    if(calls) {*calls = prefetch_calls_;}
}

extern "C" void gpu_particles_arena_mark_clean_after_scatter(const char *site)
{
    (void)site;
}

extern "C" void gpu_particles_arena_refresh_from_host(int min_capacity,
                                                     struct particle_data *P_host,
                                                     struct gas_cell_data *CellP_host,
                                                     const char *site)
{
    (void)min_capacity; (void)P_host; (void)CellP_host; (void)site;
}

/* ---- compact staging buffers (see the contract in gpu_particles_arena.h) ---- */

extern "C" int particle_staging_acquire(struct ParticleStagingBatch *batch, int capacity)
{
    if(!batch) {return 0;}
    /* Free anything the batch is already holding, so re-acquiring cannot strand the
       previous buffers. Requires the caller to have zero-initialised it once. */
    particle_staging_release(batch);
    if(capacity <= 0) {capacity = 1;}
    /* new[] rather than malloc for the host side: particle_data is over-aligned (32
       bytes, measured, against a max_align_t of 8) and only the C++ allocator honours
       that; malloc'd storage faults on the aligned vector moves the compiler emits. */
    try {
        batch->host_P    = new struct particle_data[(size_t)capacity];
        batch->host_Cell = new struct gas_cell_data[(size_t)capacity];
        batch->index     = new int[(size_t)capacity];
    }
    catch(const std::exception &) { particle_staging_release(batch); return 0; }
    batch->dev_P    = (struct particle_data *) gizmo_gpu_alloc_device(
                          (size_t)capacity * sizeof(struct particle_data), "particle_staging_P");
    batch->dev_Cell = (struct gas_cell_data *) gizmo_gpu_alloc_device(
                          (size_t)capacity * sizeof(struct gas_cell_data), "particle_staging_Cell");
    if(!batch->host_P || !batch->host_Cell || !batch->index || !batch->dev_P || !batch->dev_Cell)
        { particle_staging_release(batch); return 0; }
    batch->capacity = capacity;
    return 1;
}

extern "C" void particle_staging_release(struct ParticleStagingBatch *batch)
{
    if(!batch) {return;}
    delete[] batch->host_P;    batch->host_P    = NULL;
    delete[] batch->host_Cell; batch->host_Cell = NULL;
    delete[] batch->index;     batch->index     = NULL;
    if(batch->dev_P)    {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(batch->dev_P);    batch->dev_P    = NULL;}
    if(batch->dev_Cell) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(batch->dev_Cell); batch->dev_Cell = NULL;}
    batch->capacity = batch->count = batch->gas_count = 0;
}

using UmHostP = Kokkos::View<struct particle_data*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UmHostC = Kokkos::View<struct gas_cell_data*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UmDevP  = Kokkos::View<struct particle_data*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UmDevC  = Kokkos::View<struct gas_cell_data*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

extern "C" int particle_staging_gather(struct ParticleStagingBatch *batch, const int *idx, int n,
                                       struct particle_data *pp, struct gas_cell_data *cell)
{
    if(!batch) {return 0;}
    batch->count = batch->gas_count = 0;
    if(!idx || !pp || n <= 0) {return 0;}
    if(n > batch->capacity) {
        /* Stage nothing. Staging a prefix and letting the caller keep driving its own
           loops would feed the kernel slots that were never written, and scatter back
           through index entries that were never written -- an out-of-bounds write into
           the particle arrays, which can land before the controlled stop drains. */
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "particle staging: asked to stage %d elements into %d slots",
                 n, batch->capacity);
        gizmo_request_controlled_stop(7718, msg, __FILE__, __LINE__, __FUNCTION__);
        return 0;
    }

    /* Order the slots so the gas comes first. The physics touches a cell only for gas,
       and CellP is not allocated beyond the gas particles, so this is what makes the
       cell staging both correct and a contiguous copy. Reordering is free of
       consequence here: the drift and cooling bodies act on one particle each, with no
       pair coupling and no random draws. */
    int n_gas = 0, tail = n, k;
    for(k = 0; k < n; k++)
    {
        if(pp[idx[k]].Type == 0) {batch->index[n_gas++] = idx[k];}
        else                     {batch->index[--tail]  = idx[k];}
    }
    /* CellP is null when the run has no gas at all, which is a legal configuration
       (a pure N-body run allocates no gas cell storage). Nothing here needs it in that
       case: the partition found no gas slots, and the drift body reaches a cell only
       under its Type==0 guard. Gas with no CellP to read is a different thing -- an
       inconsistency the caller cannot recover from -- so that one is refused. */
    if(n_gas > 0 && !cell) {
        gizmo_request_controlled_stop(7719, "particle staging: gas particles staged but CellP is null",
                                      __FILE__, __LINE__, __FUNCTION__);
        return 0;
    }

    batch->count     = n;
    batch->gas_count = n_gas;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int j = 0; j < n; j++)
    {
        const int i = batch->index[j];
        batch->host_P[j] = pp[i];
        if(j < n_gas) {batch->host_Cell[j] = cell[i];}
    }

    Kokkos::deep_copy(UmDevP(batch->dev_P, (size_t)n), UmHostP(batch->host_P, (size_t)n));
    if(n_gas > 0) {Kokkos::deep_copy(UmDevC(batch->dev_Cell, (size_t)n_gas), UmHostC(batch->host_Cell, (size_t)n_gas));}
    return 1;
}

extern "C" void particle_staging_scatter(struct ParticleStagingBatch *batch,
                                        struct particle_data *pp, struct gas_cell_data *cell)
{
    if(!batch || !pp || batch->count <= 0) {return;}
    const int n = batch->count, n_gas = batch->gas_count;
    if(n_gas > 0 && !cell) {return;}   /* gather refuses this state; never reached */

    Kokkos::deep_copy(UmHostP(batch->host_P, (size_t)n), UmDevP(batch->dev_P, (size_t)n));
    if(n_gas > 0) {Kokkos::deep_copy(UmHostC(batch->host_Cell, (size_t)n_gas), UmDevC(batch->dev_Cell, (size_t)n_gas));}

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int j = 0; j < n; j++)
    {
        const int i = batch->index[j];
        pp[i] = batch->host_P[j];
        if(j < n_gas) {cell[i] = batch->host_Cell[j];}
    }
}

/* See the header for why the storage is per-caller and why the refresh is
   unconditional. */
extern "C" int drift_kick_table_mirror_refresh(double **storage, struct DriftKickTableView *view)
{
    if(!storage || !view) {return 1;}

    *view = drift_kick_table_view(NULL, NULL, 0.0, 0.0, All.Timebase_interval,
                                  All.ComovingIntegrationOn ? 1 : 0);
    if(!view->comoving) {return 0;}   /* tables never built; the elapsed-time branch needs none of them */

    if(!*storage) {
        *storage = (double *) gizmo_gpu_alloc_shared(2 * DRIFT_TABLE_LENGTH * sizeof(double), NULL);
        if(!*storage) {
            gizmo_request_controlled_stop(929701, "drift/gravkick table mirror allocation failed",
                                          __FILE__, __LINE__, __FUNCTION__);
            return 1;
        }
    }
    for(int i = 0; i < DRIFT_TABLE_LENGTH; i++) {
        (*storage)[i]                      = DriftTable[i];
        (*storage)[DRIFT_TABLE_LENGTH + i] = GravKickTable[i];
    }
    *view = drift_kick_table_view(*storage, *storage + DRIFT_TABLE_LENGTH,
                                  DriftTable_logTimeBegin, DriftTable_logTimeMax,
                                  All.Timebase_interval, 1);
    return 0;
}

extern "C" void gpu_particles_arena_release(void)
{
    /* P/CellP storage is owned by allocate.cc and persists to process exit;
     * the arena does not own it under UVM-canonical, so nothing to free. */
    arena_P         = NULL;
    arena_CellP     = NULL;
    arena_capacity_ = 0;
    arena_valid_    = 0;
}

/* ---------------------------------------------------------------------------------------
 * Placement policy for the bulk managed particle arrays (P and CellP).
 *
 * On AMD GPUs these arrays live in HIP managed memory and are demand-paged: a page moves
 * to whichever side touched it last. Every timestep both sides sweep them -- the host in
 * domain exchange, Peano-Hilbert ordering, ghost writeback and merge/split, the device in
 * the tree data path, the neighbour walks and the physics kernels -- so the traffic
 * pattern decides whether the pages settle anywhere at all.
 *
 * Which way that falls depends on how much reuse each page gets, and measurement on
 * Frontier shows the two regimes cleanly. On a SMALL footprint the device's repeated walks
 * give each page enough reuse that it settles device-resident, and biasing the pages toward
 * the host merely turns those reads into permanent Infinity-Fabric traffic: a 1e7-particle
 * run loses 2-5%. On a LARGE footprint no page gets that reuse, every cycle re-migrates the
 * array wholesale, and the migration never converges: a 4e7-particle run gains ~10% from a
 * stable host placement, and a 291.8-million-particle restart on 16 nodes gains a factor of
 * 4.2 on the first density pass -- without it that job spent its entire 30-minute wall on
 * startup and reached one sync-point, against 78 with it. Adding nodes does not help,
 * because the page count is set by the problem size and not by the rank count.
 *
 * So the policy is chosen from the per-rank footprint rather than applied unconditionally,
 * and it is applied only to P and CellP. It must NOT be extended to the tree's device
 * mirror, which is read tens of thousands of times per build and is unambiguously
 * device-owned; the same advice applied there costs ~43 s in the gravity walk alone.
 *
 * What the advice does is set a PREFERRED location and map the other side in as an
 * accessor. It is a placement bias, not a prohibition on migrating, and it does not change
 * what any code may read or write. Correctness never depends on it: if the runtime declines
 * the hint the run is slower, and nothing else changes.
 *
 * NVIDIA GPUs split by memory model. A PCIe-attached GPU (A100, H100 PCIe) demand-pages managed
 * memory between host and device as above, but measured there the device side should own it: on
 * a 12M-particle cosmological zoom on Lonestar6 A100 nodes (6 ranks, ~7 GB of particle storage
 * per rank) device-preferred placement ran 5% faster than no hint, while host-preferred gained in
 * the tree build and the gradient and force loops and gave all of it back in the density walk,
 * the kicks and the sinks, finishing level with no hint. So a PCIe-attached GPU prefers the
 * device at any footprint. A Grace-Hopper node reads managed memory coherently from both sides;
 * there the default is no hint, since a host preference would hold the arrays in the Grace
 * memory, which the GPU reaches at a fraction of its own memory's bandwidth. An explicit
 * override applies on both.
 * -------------------------------------------------------------------------------------- */

/*! Footprint above which host-preferred placement is chosen, as total per-rank bytes of P
 *  plus CellP. Empirical, from the three Frontier workloads above, at 64 to 128 ranks: the
 *  4e7-particle case measures ~2.3 GB per rank and wins, the 3e8-particle case ~9 GB per rank
 *  and wins outright, and the 1e7-particle case -- several times smaller again, and with a
 *  much smaller gas fraction -- loses. That leaves a wide window rather than a boundary, so
 *  the exact value is not delicate; it is a first cut to be revisited as production-size
 *  measurements accumulate, and a run can be forced either way to measure it.
 *
 *  Note that this is ALLOCATED CAPACITY, not the count of live particles, so it carries
 *  PartAllocFactor with it -- as do the three figures above, which were read from the
 *  allocations themselves, so the cutoff and the calibration are in the same units. Live
 *  particles are the better predictor of how many pages actually move, and the two part
 *  company on a run with an unusually generous allocation factor; a run near the cutoff
 *  should be measured both ways rather than reasoned about. */
#define PARTICLE_STORAGE_HOST_PREFERRED_MIN_BYTES ((size_t) 1024 * 1024 * 1024)

/*! ⚠ THE HOST-PREFERRED PLACEMENT IS A STOPGAP, NOT A DESIGN. It is here because the largest
 *  runs would otherwise not complete, and it is accepted on those terms alone. Keeping the bulk
 *  particle storage on the host is the opposite of what this port is for: every routine moved
 *  onto the device has to reach across the fabric for it, so the hint buys a large win today by
 *  making a larger one harder to reach. It is a debt to be repaid by putting enough of the
 *  consumers on the device that the storage can follow them, not a setting to tune around.
 *
 *  ⛔ THEREFORE: a single device flip measured against this placement is NOT a verdict on that
 *  flip. Priced one at a time against host-resident storage, every one of them loses, because
 *  each pays the fabric crossing alone while the saving only appears once enough of them move
 *  together. Any such measurement must be paired with an arm that moves the placement too --
 *  which is what the DEVICE setting below exists for.
 *
 *  Where this rank's particle storage should prefer to live. `particle_arena_bytes` is the total
 *  for P plus CellP, so the two arrays always decide together and a gas-free run is judged on P
 *  alone. GPU_PARTICLE_STORAGE_PLACEMENT overrides the decision for validation and tuning;
 *  leaving it unset is the production path. */
#define PARTICLE_STORAGE_PLACEMENT_NONE   0   /* no hint: pages settle wherever they are used */
#define PARTICLE_STORAGE_PLACEMENT_HOST   1   /* prefer host, map the device in as an accessor */
#define PARTICLE_STORAGE_PLACEMENT_DEVICE 2   /* prefer device, map the host in as an accessor */

#if defined(KOKKOS_ENABLE_CUDA)
/*! Whether the host reads and writes CUDA managed memory in place (Grace-Hopper) rather than through page
 *  migration (a PCIe-attached GPU). Storage outside CUDA managed memory is not paged at all and counts as
 *  coherent; an attribute that cannot be read counts as not coherent. */
static int cuda_managed_memory_is_host_coherent(void)
{
    static int coherent = -1;
    if(coherent < 0) {
        int attr = 0, dev = 0;
        if(!std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value) {coherent = 1;}
        else if(cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&attr, cudaDevAttrDirectManagedMemAccessFromHost, dev) == cudaSuccess) {coherent = attr ? 1 : 0;}
        else {coherent = 0;}
    }
    return coherent;
}
#endif

static int particle_storage_placement(size_t particle_arena_bytes)
{
#if defined(GPU_PARTICLE_STORAGE_PLACEMENT)
    (void) particle_arena_bytes;
    /* Anything outside the three named values keeps the original meaning of a nonzero setting,
     * so a Config carrying an older value still selects what it always selected. */
    return (GPU_PARTICLE_STORAGE_PLACEMENT == PARTICLE_STORAGE_PLACEMENT_NONE)
             ? PARTICLE_STORAGE_PLACEMENT_NONE
         : (GPU_PARTICLE_STORAGE_PLACEMENT == PARTICLE_STORAGE_PLACEMENT_DEVICE)
             ? PARTICLE_STORAGE_PLACEMENT_DEVICE
             : PARTICLE_STORAGE_PLACEMENT_HOST;
#else
#if defined(KOKKOS_ENABLE_CUDA)
    (void) particle_arena_bytes;
    return cuda_managed_memory_is_host_coherent() ? PARTICLE_STORAGE_PLACEMENT_NONE : PARTICLE_STORAGE_PLACEMENT_DEVICE;
#else
    return (particle_arena_bytes >= PARTICLE_STORAGE_HOST_PREFERRED_MIN_BYTES)
             ? PARTICLE_STORAGE_PLACEMENT_HOST : PARTICLE_STORAGE_PLACEMENT_NONE;
#endif
#endif
}

/*! Ask the driver for one placement on one managed buffer: a preferred location on one side and
 *  in-place access from the other, so whichever side is preferred, the other reaches the pages
 *  without moving them. The two requests are independent and either can be refused alone, so each
 *  outcome is returned as text for the caller's report. Returns 1 if both were accepted, 0 if
 *  either was refused, and -1 where nothing was asked of the driver (no GPU, no current device,
 *  or storage that is not managed memory), so the caller reports neither a hint nor a refusal. */
static int managed_memory_advise(void *p, size_t nbytes, int placement, const char **pref_result, const char **access_result)
{
    *pref_result = *access_result = "not applicable";
    const int on_device = (placement == PARTICLE_STORAGE_PLACEMENT_DEVICE);
#if defined(KOKKOS_ENABLE_HIP)
    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess) {return -1;}   /* nothing was asked of the driver */
    hipError_t rc_pref = hipMemAdvise(p, nbytes, hipMemAdviseSetPreferredLocation, on_device ? dev : hipCpuDeviceId);
    hipError_t rc_acc  = hipMemAdvise(p, nbytes, hipMemAdviseSetAccessedBy, on_device ? hipCpuDeviceId : dev);
    *pref_result = hipGetErrorString(rc_pref); *access_result = hipGetErrorString(rc_acc);
    return (rc_pref == hipSuccess && rc_acc == hipSuccess);
#elif defined(KOKKOS_ENABLE_CUDA)
    if(!std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value) {return -1;}
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess) {return -1;}   /* nothing was asked of the driver */
    cudaError_t rc_pref = cudaMemAdvise(p, nbytes, cudaMemAdviseSetPreferredLocation, on_device ? dev : cudaCpuDeviceId);
    cudaError_t rc_acc  = cudaMemAdvise(p, nbytes, cudaMemAdviseSetAccessedBy, on_device ? cudaCpuDeviceId : dev);
    *pref_result = cudaGetErrorString(rc_pref); *access_result = cudaGetErrorString(rc_acc);
    return (rc_pref == cudaSuccess && rc_acc == cudaSuccess);
#else
    (void) p; (void) nbytes; (void) on_device;
    return -1;
#endif
}


/*! Apply the placement policy to one freshly allocated buffer, before anything touches it, so
 *  that the first write already lands where the policy wants it. `particle_arena_bytes` is
 *  zero for buffers that are not one of the bulk particle record arrays, which is how
 *  everything else served by this allocator is left alone. */
static void particle_storage_apply_placement(void *p, size_t nbytes, size_t particle_arena_bytes)
{
    if(!p || nbytes == 0 || particle_arena_bytes == 0) {return;}
    const int placement = particle_storage_placement(particle_arena_bytes);
    if(placement == PARTICLE_STORAGE_PLACEMENT_NONE) {return;}
    const char *where = (placement == PARTICLE_STORAGE_PLACEMENT_DEVICE) ? "device" : "host";
    const char *pref_result, *access_result;
    const int accepted = managed_memory_advise(p, nbytes, placement, &pref_result, &access_result);
    if(accepted < 0) {return;}
    if(!accepted)
    {
        /* The two calls are independent, so one of them can be refused on its own and leave
         * the other standing. Name which, rather than claim the default placement is back:
         * a later measurement made against a half-applied hint is otherwise read as a
         * measurement of the default. Said once; the run continues either way. */
        static int reported = 0;
        if(!reported && ThisTask == 0)
        {
            reported = 1;
            printf("Particle storage: the %s-preferred placement hint was not fully applied "
                   "(preferred location: %s; other-side access: %s). Whatever part of it was accepted "
                   "stands, and the run continues at whatever speed that gives.\n",
                   where, pref_result, access_result);
            fflush(stdout);
        }
        return;
    }
    /* Deliberately not latched: the capacity can change while the run is going, and each
     * change reallocates and re-applies. One line per application is what shows that it did,
     * and capacity changes are rare enough to be worth a line of their own anyway. */
    if(ThisTask == 0)
    {
        printf("Particle storage: %s-preferred placement applied to %g MByte "
               "(particle arrays total %g MByte on this rank).\n",
               where,
               (double) nbytes / (1024.0 * 1024.0),
               (double) particle_arena_bytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

extern "C" void *gpu_particles_uvm_alloc(size_t nbytes, const char *label, size_t particle_arena_bytes)
{
    if(nbytes == 0) {return NULL;}
    /* kokkos_malloc THROWS on host-OOM; catch -> NULL so the caller's NULL-check
       (allocate.cc alloc_fail_local) fires instead of a hard terminate. The label
       names the buffer in the Kokkos allocation stream and any future OOM message. */
    void *p = NULL;
    try { p = Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label ? label : "particle_soa_unlabeled", nbytes); }
    catch(const std::exception &) { return NULL; }
    if(p) {particle_storage_apply_placement(p, nbytes, particle_arena_bytes); memset(p, 0, nbytes);}
    return p;
}

/* Whether MPI must receive into the particle storage (P, CellP) through host staging rather than directly.
 * Where CUDA managed memory is not coherent with the host (a PCIe-attached GPU) its pages migrate between host
 * and device, and an MPI receive written straight into them can be lost: measured on Lonestar6 A100 nodes,
 * imported and exchanged particles arrived still holding what their slots held before.  Where the host reads
 * managed memory in place (GH200) nothing migrates, and the receive is direct.  HIP builds receive directly:
 * their large particle arrays are placed host-preferred (particle_storage_apply_placement) and no loss has been
 * seen there; host builds hold the storage in ordinary host memory.  Decided here, where the Kokkos memory space
 * and the device are visible, for the host-compiled exchange code that cannot see them; an attribute that cannot
 * be read counts as not coherent. */
extern "C" int gizmo_particle_storage_needs_staged_mpi_receive(void)
{
#if defined(KOKKOS_ENABLE_CUDA)
    return cuda_managed_memory_is_host_coherent() ? 0 : 1;
#else
    return 0;
#endif
}

/* Non-throwing allocation for the GPU transients, one entry point per memory space. Kokkos throws
   when it cannot serve a request, and an exception leaving a dispatcher takes the rank down where it
   stands, before the phase boundary that drains a controlled stop -- so one rank dies and the others
   wait on it. Returning NULL instead lets the caller name what it could not get, ask for the stop and
   return, and the run finishes the way every other failure does. Nothing is caught on success, so a
   run that never exhausts memory pays nothing. The label is forwarded exactly as given, including
   absent: the memory ledger buckets allocations by label prefix, so inventing one here would move a
   call site into a different bucket. */
extern "C" void *gizmo_gpu_alloc_shared(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

/*! Place a shared-space buffer that one side reads far more than the other.
 *
 *  The particle records get their placement inside their own allocator, keyed on how big the
 *  arena is; this is for the buffers that are not particle records and were therefore, until now,
 *  left with no hint at all -- which is not a neutral state. An unhinted managed buffer settles
 *  where it is FIRST TOUCHED and is then reached from the other side page by page, so a structure
 *  the host builds and the device reads ends up host-resident for the whole of every device read
 *  of it, at whatever the fabric costs.
 *
 *  `placement` is one of the PARTICLE_STORAGE_PLACEMENT_* values, so a caller can carry its own
 *  tri-state knob and this stays the one place that knows how to speak to the driver. */
extern "C" void gizmo_gpu_place_shared(void *p, size_t nbytes, int placement, const char *what)
{
    if(!p || nbytes == 0 || placement == PARTICLE_STORAGE_PLACEMENT_NONE) {return;}
    const int on_device = (placement == PARTICLE_STORAGE_PLACEMENT_DEVICE);
    const char *pref_result, *access_result;
    const int accepted = managed_memory_advise(p, nbytes, placement, &pref_result, &access_result);
    if(accepted < 0) {return;}
    /* Reported once per buffer kind rather than per allocation: the tree reallocates whenever it
     * grows, and a line each time would bury the one fact worth having, which is whether the hint
     * was accepted at all. A refusal is named because a later measurement taken against a
     * half-applied hint otherwise reads as a measurement of the default. */
    static int reported_ok = 0, reported_bad = 0;
    if(!accepted) {
        if(!reported_bad && ThisTask == 0) {
            reported_bad = 1;
            printf("%s placement: the %s-preferred hint was not fully applied (preferred location: %s; "
                   "other-side access: %s). Whatever part of it was accepted stands.\n",
                   what ? what : "shared buffer", on_device ? "device" : "host",
                   pref_result, access_result);
            fflush(stdout);
        }
        return;
    }
    if(!reported_ok && ThisTask == 0) {
        reported_ok = 1;
        printf("%s placement: %s-preferred applied (first buffer %g MByte).\n",
               what ? what : "shared buffer", on_device ? "device" : "host",
               (double) nbytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

extern "C" void *gizmo_gpu_alloc_device(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void gizmo_gpu_free_device(void *ptr)
{
    if(ptr) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(ptr);}
}

extern "C" void gizmo_gpu_copy_to_device(void *dst_device, const void *src_host, size_t nbytes)
{
    if(nbytes == 0) {return;}
    Kokkos::deep_copy(Kokkos::View<char*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((char *) dst_device, nbytes),
                      Kokkos::View<const char*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((const char *) src_host, nbytes));
}

extern "C" void gizmo_gpu_copy_to_host(void *dst_host, const void *src_device, size_t nbytes)
{
    if(nbytes == 0) {return;}
    Kokkos::fence("gizmo_gpu_copy_to_host");   /* device-wide: the source may have been written by a library on its own stream */
    Kokkos::deep_copy(Kokkos::View<char*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((char *) dst_host, nbytes),
                      Kokkos::View<const char*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((const char *) src_device, nbytes));
}

extern "C" void *gizmo_gpu_alloc_host(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<Kokkos::HostSpace>(label, nbytes)
                       : Kokkos::kokkos_malloc<Kokkos::HostSpace>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

/* Page-locked host memory is a node resource: it cannot be paged out, so what one rank pins is
   gone for every other process on the node until it is released. Callers hold it for a bounded
   span and check the result like every allocation here. */
extern "C" void *gizmo_gpu_alloc_pinned_host(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_PINNED_HOST_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_PINNED_HOST_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void gizmo_gpu_free_pinned_host(void *ptr)
{
    if(ptr) {Kokkos::kokkos_free<GIZMO_KOKKOS_PINNED_HOST_SPACE>(ptr);}
}

extern "C" void gpu_particles_uvm_free(void *ptr)
{
    /* Paired with gpu_particles_uvm_alloc so a capacity change can allocate the replacement
       buffer before releasing the old one. Lives here, not in allocate.cc, for the same reason
       the alloc does: that TU must not include <Kokkos_Core.hpp>. */
    if(ptr) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ptr);}
}

extern "C" struct particle_data *gpu_particles_arena_P(void)     {return arena_valid_ ? arena_P     : NULL;}
extern "C" struct gas_cell_data *gpu_particles_arena_CellP(void) {return arena_valid_ ? arena_CellP : NULL;}
extern "C" int gpu_particles_arena_capacity(void)                {return arena_capacity_;}
extern "C" int gpu_particles_arena_valid(void)                   {return arena_valid_;}
