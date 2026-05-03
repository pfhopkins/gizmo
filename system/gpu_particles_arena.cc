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

#include <Kokkos_Core.hpp>

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "gpu_particles_arena.h"
#include "../mesh/gpu_neighbor_list.h"


static struct particle_data *arena_P     = NULL;
static struct gas_cell_data *arena_CellP = NULL;
static int arena_capacity_ = 0;
static int arena_valid_    = 0;

/* DIAGNOSTIC: per-acquire sequence counter and call-site tag.
 * Call gpu_particles_arena_set_site("tag") just before acquire to label it. */
static int  g_acquire_serial  = 0;
static int  g_valid_memcpy_serial = 0;  /* serial# of last acquire that did memcpy */
static const char *g_arena_site = "(unknown)";  /* set by caller before acquire */

/* Phase 8a Round 1: track last call site that asserted "I just mirror-updated
 * the arena" — for diagnostic output when the debug byte-compare guard fails. */
static const char *g_arena_last_clean_site = "(none)";

/* Phase 8a Round 1.5: also track who did the most-recent slow-path memcpy
 * (that made arena valid) and who most recently invalidated. The acquire
 * trio (memcpy, invalidate, current) names the suspects when the guard fires. */
static const char *g_arena_last_memcpy_site    = "(none)";
static const char *g_arena_last_invalidate_site = "(none)";

extern "C" void gpu_particles_arena_set_site(const char *site) { g_arena_site = site; }

extern "C" void gpu_particles_arena_acquire(int min_capacity,
                                            struct particle_data *P_host,
                                            struct gas_cell_data *CellP_host)
{
    if(min_capacity <= 0) {min_capacity = 1;}
    g_acquire_serial++;
    int my_serial = g_acquire_serial;

    /* Phase 7 Round A4: arena state diagnostics. env-gated; no-op when off. */
    gizmo_step_phase_record("arena_acquire_calls", 1.0);

    /* UVM-canonical particles: P[] and CellP[] live in Kokkos::SharedSpace
     * (see system/allocate.cc).  The arena is a pure alias — no separate
     * allocation, no memcpy, no coherence dance.  Both fast-path acquire
     * and any prior slow-path collapse to the same trivial pointer copy. */
    arena_P         = P_host;
    arena_CellP     = CellP_host;
    arena_capacity_ = min_capacity;
    arena_valid_    = 1;
    g_valid_memcpy_serial    = my_serial;
    g_arena_last_memcpy_site = g_arena_site;
    gizmo_step_phase_record("arena_acquire_fastpath", 1.0);
}

/* Under UVM-canonical particles, invalidate is a counter-only no-op.  The
 * arena IS host; there is nothing to mark stale.  Existing call sites are
 * kept as cost-free counters for diagnostic correlation with STEP_PHASES. */
extern "C" void gpu_particles_arena_invalidate(void)
{
    gizmo_step_phase_record("arena_invalidate_calls", 1.0);
    g_arena_last_invalidate_site = g_arena_site;
}

extern "C" void gpu_particles_arena_mark_clean_after_scatter(const char *site)
{
    /* No-op under UVM-canonical: arena is always coherent because it IS host. */
    g_arena_last_clean_site = (site ? site : "(unnamed)");
    gizmo_step_phase_record("arena_mark_clean_calls", 1.0);
}

extern "C" void gpu_particles_arena_refresh_from_host(int min_capacity,
                                                     struct particle_data *P_host,
                                                     struct gas_cell_data *CellP_host,
                                                     const char *site)
{
    /* No-op under UVM-canonical: P_host/CellP_host already are the arena. */
    gizmo_step_phase_record("arena_refresh_calls", 1.0);
    g_arena_last_memcpy_site = (site ? site : "(unnamed_refresh)");
    (void)min_capacity; (void)P_host; (void)CellP_host;
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

