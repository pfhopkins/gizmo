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

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/lifecycle_counters.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
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
    gizmo_step_phase_record("arena_acquire_calls", 1.0);
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

extern "C" void gpu_particles_arena_release(void)
{
    /* P/CellP storage is owned by allocate.cc and persists to process exit;
     * the arena does not own it under UVM-canonical, so nothing to free. */
    arena_P         = NULL;
    arena_CellP     = NULL;
    arena_capacity_ = 0;
    arena_valid_    = 0;
}

extern "C" void *gpu_particles_uvm_alloc(size_t nbytes)
{
    if(nbytes == 0) {return NULL;}
    void *p = Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(nbytes);
    if(p) {memset(p, 0, nbytes);}
    return p;
}

extern "C" struct particle_data *gpu_particles_arena_P(void)     {return arena_valid_ ? arena_P     : NULL;}
extern "C" struct gas_cell_data *gpu_particles_arena_CellP(void) {return arena_valid_ ? arena_CellP : NULL;}
extern "C" int gpu_particles_arena_capacity(void)                {return arena_capacity_;}
extern "C" int gpu_particles_arena_valid(void)                   {return arena_valid_;}
