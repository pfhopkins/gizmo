#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#ifdef RT_SOURCE_INJECTION
#include "rt_source_injection_local_in.h"
#include "rt_source_injection_loop_api.h"
#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION) && defined(RT_EVOLVE_FLUX)
#include "../system/gpu_particles_arena.h"
#include "../mesh/ghost_writeback.h"   /* ghost_get_num_local for owner-only slab predicate */
#endif
#endif

/*! \file rt_source_injection.c
 *  \brief inject luminosity from point sources to neighboring gas particles
 *
 *  This file contains a loop modeled on the gas density computation which will
 *    share luminosity from non-gas particles to the surrounding gas particles,
 *    so that it can be treated within e.g. the flux-limited diffusion or other
 *    radiation-hydrodynamic approximations. Basically the same concept as
 *    injecting the radiation 'in a cell' surrounding the particle
 *
 *  The neighbor loop is carried by RtSrcInjectionSpec via run_neighbor_loop<>.
 *  The legacy `rt_source_injection_evaluate_gpu` in rt_source_injection_gpu.cc
 *  is superseded and slated for removal. This TU keeps the host orchestration:
 *  the gas self-source preloop, the rt_sourceinjection_active_check filter,
 *  the toplevel active-list build, and (under
 *  GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION) the post-runner testproblem
 *  boundary-condition fixup. The runner-template wrapper lives in
 *  rt_source_injection_loop.cc; the host TU sees only its forward decl in
 *  rt_source_injection_loop_api.h (no Kokkos drag).
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef RT_SOURCE_INJECTION


int rt_sourceinjection_active_check(int i);
int rt_sourceinjection_active_check(int i)
{
    if(P[i].NumNgb <= 0) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].Mass <= 0) return 0;
    if(P[i].KernelSum_Around_RT_Source <= 0) return 0;
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if(P[i].Type == 5 && !P[i].do_gas_search_this_timestep) return 0;
#endif
    double lum[N_RT_FREQ_BINS];
    return rt_get_source_luminosity(i,-1,lum, P, CellP);
}


/*! operations that need to be performed before entering the main loop */
void rt_source_injection_initial_operations_preloop(void);
void rt_source_injection_initial_operations_preloop(void)
{
    /* first, we do a loop over the gas particles themselves. these are trivial -- they don't need to share any information,
     they just determine their own source functions. so we don't need to do any loops. and we can zero everything before the loop below.
     Active-gas loop: each cell's own source function updates on its own timestep;
     ActiveParticleList also never contains ghost slots, so this stays correct
     while a ghost pool is materialized inside NumPart. */
    if(!(RT_SOURCES & 1)) return; // we skip this if gas cells don't have explicit source terms

    for(int j : ActiveParticleList) {
        if(P[j].Type==0) {
            double lum[N_RT_FREQ_BINS]; int k;
            for(k=0;k<N_RT_FREQ_BINS;k++) {CellP[j].Rad_Je[k]=0;} // need to zero -before- calling injection //
            int active_check = rt_get_source_luminosity(j,0,lum, P, CellP);
            /* here is where we would need to code some source luminosity for the gas */
            for(k=0;k<N_RT_FREQ_BINS;k++) if(active_check) {CellP[j].Rad_Je[k]=lum[k];}
        }
    }
}


#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION) && defined(RT_EVOLVE_FLUX)
/* Testproblem-specific boundary-condition fixup.
 *
 * Replaces the legacy in-pair atomic_exchange block (formerly in the
 * pre-port rt_source_injection_functions.h) which mixed additive injection
 * with assignment semantics; that pattern is incompatible with the additive
 * ghost-writeback bundle the runner-port uses.
 *
 * This is an OWNER-LOCAL postpass: every rank applies the same slab-geometry
 * predicate to its own owned gas cells. No MPI semantics. After mutating
 * CellP we invalidate the GPU particles arena so the next GPU acquire
 * re-copies fresh device state.
 *
 * Predicate is intentionally simple and editable per future variants of this
 * test problem (Phil-flagged 2026-05-17: "would probably need to be edited
 * for future test problems anyways"). Codebase convention `boxSize_Z` is used
 * for the z-slab (see e.g. rt_utilities.cc:108-110 for the same convention).
 *
 * Runs unconditionally on every rank — does NOT short-circuit on
 * global_num_active==0. This is INTENTIONAL: the helper imposes a problem-
 * specific boundary CONDITION on flux fields in the testproblem's z-slab
 * boundary layer regardless of which sources fired this step. It is boundary
 * maintenance, NOT "touched by source" cleanup. (The legacy in-pair
 * atomic_exchange branch only fired for cells touched by a source, but that
 * was the wrong layer; the boundary condition logically belongs to the slab
 * geometry, not to the source-list.)
 *
 * Gated additionally on RT_EVOLVE_FLUX because Rad_Flux / Rad_Flux_Pred are
 * the fields it overwrites; outside RT_EVOLVE_FLUX the fields are not
 * meaningfully evolved and the testproblem semantics don't apply.
 */
static void rt_source_injection_apply_grain_rdi_boundary_fixup(void)
{
    int modified_any = 0;
    const int num_local = ghost_get_num_local();
    for (int j = 0; j < num_local; j++) {
        if (P[j].Type != 0)  continue;
        if (P[j].Mass <= 0)  continue;
        /* Edit this predicate for future RDI testproblem variants. */
        if (P[j].Pos[2] >= 0.1 * boxSize_Z) continue;

        for (int k = 0; k < N_RT_FREQ_BINS; k++) {
            double qtau = (0.75 * All.Grain_Q_at_MaxGrainSize) /
                          ((All.Grain_Internal_Density / UNIT_DENSITY_IN_CGS) *
                           (All.Grain_Size_Max          / UNIT_LENGTH_IN_CGS));
            double e0       = (P[j].Mass / CellP[j].Density) * All.Vertical_Grain_Accel / qtau;
            double tau_tot  = All.Dust_to_Gas_Mass_Ratio * qtau;
            double flux_egy_0 = C_LIGHT_CODE_REDUCED *
                                DMAX(CellP[j].Rad_E_gamma[k], CellP[j].Rad_E_gamma_Pred[k]) /
                                (1.0 + 3.0 * tau_tot);
            double f0 = DMAX(flux_egy_0,
                             DMAX(C_LIGHT_CODE_REDUCED * e0,
                                  DMAX(CellP[j].Rad_Flux[k][2], CellP[j].Rad_Flux_Pred[k][2])));
            CellP[j].Rad_Flux[k][0] = 0;
            CellP[j].Rad_Flux[k][1] = 0;
            CellP[j].Rad_Flux[k][2] = f0;
            CellP[j].Rad_Flux_Pred[k][0] = 0;
            CellP[j].Rad_Flux_Pred[k][1] = 0;
            CellP[j].Rad_Flux_Pred[k][2] = f0;
            modified_any = 1;
        }
    }
    if (modified_any) gpu_particles_arena_invalidate();
}
#endif /* GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION && RT_EVOLVE_FLUX */


/*! routine to do the top-level loop over particles, for the source injection (photons put into surrounding gas) */
void rt_source_injection(void)
{
    PRINT_STATUS(" ..injecting radiation onto grid for RHD steps");
    rt_source_injection_initial_operations_preloop(); /* operations before the main loop */

    /* Build LOCAL active source list + per-source LocalIn pack on host.
     * rt_source_injection_fill_local clears P[i].Sink_accreted_photon_energy
     * under RT_REINJECT_ACCRETED_PHOTONS+Type==5 — that's a deterministic
     * host-serial state mutation that MUST happen once per source, here, before
     * the runner kernel sees any of it. Iterating ActiveParticleList (NOT
     * NumPart) avoids double-deposit from ghost-imported sources. */
    std::vector<int>          active_src;
    std::vector<RtSrcLocalIn> host_locals;
    for (int i : ActiveParticleList) {
        if (!rt_sourceinjection_active_check(i)) continue;
        RtSrcLocalIn loc;
        rt_source_injection_fill_local(i, P, CellP, &loc);
        active_src.push_back(i);
        host_locals.push_back(loc);
    }
    const int num_active = (int)active_src.size();

    /* Collective-symmetry: every rank must enter the runner so MPI collectives
     * inside (ghost_exchange, Allreduce, ghost_writeback Alltoallv) stay in
     * lockstep on multi-rank runs. Short-circuit only when globally zero. */
    int global_num_active = 0;
    MPI_Allreduce(&num_active, &global_num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_num_active > 0) {
        rt_source_injection_run_loop(active_src.data(), host_locals.data(), num_active);
    }

#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION) && defined(RT_EVOLVE_FLUX)
    /* Owner-local testproblem boundary fixup. Runs on every rank regardless of
     * local active source count (boundary cells may be owned by ranks with
     * num_active==0). See helper comment block above. */
    rt_source_injection_apply_grain_rdi_boundary_fixup();
#endif

    CPU_Step[CPU_RTNONFLUXOPS] += measure_time(); /* collect timings and reset clock for next timing */
}


#endif
