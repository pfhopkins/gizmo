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

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/lifecycle_counters.h"
#include "../core/proto.h"
#include "gpu_particles_arena.h"
#include "../mesh/gpu_neighbor_list.h"


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
    if(!idx || !pp || !cell || n <= 0) {return 0;}
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
    if(!batch || !pp || !cell || batch->count <= 0) {return;}
    const int n = batch->count, n_gas = batch->gas_count;

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

extern "C" void gpu_particles_arena_release(void)
{
    /* P/CellP storage is owned by allocate.cc and persists to process exit;
     * the arena does not own it under UVM-canonical, so nothing to free. */
    arena_P         = NULL;
    arena_CellP     = NULL;
    arena_capacity_ = 0;
    arena_valid_    = 0;
}

extern "C" void *gpu_particles_uvm_alloc(size_t nbytes, const char *label)
{
    if(nbytes == 0) {return NULL;}
    /* kokkos_malloc THROWS on host-OOM; catch -> NULL so the caller's NULL-check
       (allocate.cc alloc_fail_local) fires instead of a hard terminate. The label
       names the buffer in the Kokkos allocation stream and any future OOM message. */
    void *p = NULL;
    try { p = Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label ? label : "particle_soa_unlabeled", nbytes); }
    catch(const std::exception &) { return NULL; }
    if(p) {memset(p, 0, nbytes);}
    return p;
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

extern "C" void *gizmo_gpu_alloc_device(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void *gizmo_gpu_alloc_host(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<Kokkos::HostSpace>(label, nbytes)
                       : Kokkos::kokkos_malloc<Kokkos::HostSpace>(nbytes); }
    catch(const std::exception &) { return NULL; }
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
