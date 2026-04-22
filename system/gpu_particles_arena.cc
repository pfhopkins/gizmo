/* gpu_particles_arena.cc — Step 13 Phase 1
 *
 * See gpu_particles_arena.h for design notes.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "gpu_particles_arena.h"

#ifdef OPENMP_GPU_OFFLOAD

static struct particle_data *arena_P     = NULL;
static struct gas_cell_data *arena_CellP = NULL;
static int arena_capacity_ = 0;
static int arena_valid_    = 0;

extern "C" void gpu_particles_arena_acquire(int min_capacity,
                                            struct particle_data *P_host,
                                            struct gas_cell_data *CellP_host)
{
    if(min_capacity <= 0) {min_capacity = 1;}

    if(arena_P && arena_CellP && arena_capacity_ >= min_capacity) {
        if(arena_valid_) {
#ifdef GIZMO_GPU_ARENA_DEBUG
            /* Debug guard: arena claims to be in sync with host; verify by byte-compare.
             * If a host mutation site forgot to call gpu_particles_arena_invalidate(),
             * this aborts at the offending kernel call rather than yielding silent
             * stale-data corruption downstream. */
            if(memcmp(arena_P, P_host, min_capacity * sizeof(struct particle_data)) != 0 ||
               (CellP_host && memcmp(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data)) != 0)) {
                printf("gpu_particles_arena_acquire: arena_valid_==1 but host data differs from arena.\n"
                       "  Some host mutation site missed calling gpu_particles_arena_invalidate().\n"
                       "  Capacity = %d. Aborting.\n", min_capacity);
                endrun(913002);
            }
#endif
            /* Fast path: arena holds the latest host state already (no invalidate
             * fired since the previous acquire). Skip memcpy entirely — the win
             * compounds across kernels in a single timestep. */
            return;
        }
        /* Slow path: arena is stale (some host mutation site invalidated us).
         * Re-seed from host. UVM keeps pages device-resident if they weren't
         * dirtied on the device side since last access. CellP_host may be NULL
         * for N-body callers (no gas); arena_CellP storage exists but is unused. */
        memcpy(arena_P,     P_host,     min_capacity * sizeof(struct particle_data));
        if(CellP_host) {memcpy(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data));}
        arena_valid_ = 1;
        return;
    }

    /* Need fresh allocation: capacity grew or first-ever acquire. */
    if(arena_CellP) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_CellP); arena_CellP = NULL;}
    if(arena_P)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_P);     arena_P     = NULL;}

    arena_P     = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(min_capacity * sizeof(struct particle_data));
    arena_CellP = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(min_capacity * sizeof(struct gas_cell_data));
    if(!arena_P || !arena_CellP) {
        printf("gpu_particles_arena_acquire: kokkos_malloc failed for capacity=%d\n", min_capacity);
        endrun(913001);
    }
    memcpy(arena_P,     P_host,     min_capacity * sizeof(struct particle_data));
    if(CellP_host) {memcpy(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data));}
    arena_capacity_ = min_capacity;
    arena_valid_    = 1;
}

extern "C" void gpu_particles_arena_invalidate(void)
{
    arena_valid_ = 0;
}

extern "C" void gpu_particles_arena_release(void)
{
    if(arena_CellP) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_CellP); arena_CellP = NULL;}
    if(arena_P)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_P);     arena_P     = NULL;}
    arena_capacity_ = 0;
    arena_valid_    = 0;
}

extern "C" struct particle_data *gpu_particles_arena_P(void)     {return arena_valid_ ? arena_P     : NULL;}
extern "C" struct gas_cell_data *gpu_particles_arena_CellP(void) {return arena_valid_ ? arena_CellP : NULL;}
extern "C" int gpu_particles_arena_capacity(void)                {return arena_capacity_;}
extern "C" int gpu_particles_arena_valid(void)                   {return arena_valid_;}

#endif /* OPENMP_GPU_OFFLOAD */
