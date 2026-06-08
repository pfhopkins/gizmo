#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* Kokkos_Core must precede allvars.h (its macros may conflict with stdlib
 * names). Unconditional so the include order is uniform across Config branches. */
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"   /* MUST precede thermal_fb_loop.h below — kernel.h
                                 has no include guards; thermal_fb_loop.h's inline
                                 pair body references kernel_hinv / kernel_main. */

/* Routines for pure thermal/scalar feedback/enrichment models: these are intended to represent
    extremely simplified models for stellar feedback manifest as a "pure thermal energy dump"
    (potentially with some cooling turnoff). This is -not- a model for mechanical feedback
    (which, critically, must include the actual momentum and solve for wind/SNe shock properties
    at the interface with the ISM). Those physics are included in the mechanical_fb.c file and
    algorithms therein. This also uses an extremely simple kernel-weighting, rather than a
    more self-consistent area weighting. */

/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#if defined(GALSF_FB_THERMAL)

#include "../mesh/neighbor_loop_runner.h"   /* nlr_build_active_list, run_neighbor_loop, nlr_default_args */
#include "thermal_fb_loop.h"                /* ThermalFBSpec, ThermalFBLocalIn, thermal_fb_local_fill */


/* routine that evaluates whether a FB event occurs in a given particle, in a given timestep */
void determine_where_addthermalFB_events_occur(void)
{
    double check = 0;
    for (int i : ActiveParticleList)
    {
        /* Cosmology-aware stellar filter — mirrors mechanical_fb.cc:75-82.
         * Pre-port code strict-gated on Type==4 only; that missed valid
         * stars (Types 2/3) in non-cosmological sims. Fix landed 3e.2
         * (Phil confirmed 2026-05-15; SSOT'd with ThermalFBSpec::is_active). */
        if(All.ComovingIntegrationOn) {
            if(P[i].Type != 4) {continue;}
        } else {
            if((P[i].Type < 2) || (P[i].Type > 4)) {continue;}
        }
        if(P[i].Mass <= 0) {continue;}
        check += mechanical_fb_calculate_eventrates(i,1); // this should do the calculation and add to number of SNe as needed //
    } // for (int i : ActiveParticleList) //
    (void)check;
}


/* Deposit thermal feedback to gas — runner-template caller (Phase 4 / Wave 3 / 3e.2).
 *
 * Replaces the legacy active-list construction + thermal_fb_evaluate_gpu call.
 * ThermalFBSpec::is_active is the SSOT active predicate (former
 * addthermalFB_evaluate_active_check is retired in this commit; all paths
 * route through Spec::is_active). The runner handles ghost-import collective,
 * Mode-A/B dispatch, oracle gating, j-side ghost-writeback (via the manifest
 * bundle in thermal_fb_loop.cc), and per-active apply_active_writeback
 * (source-side Mass / dp loss). */
void thermal_fb_calc(void)
{
    PRINT_STATUS(" ..depositing thermal feedback to gas");

    /* ---- Build active list via the engine helper (filters by Spec::is_active,
     *      MPI_Allreduce-gates on global count). */
    int *active_list = nullptr;
    int  num_active = 0, num_global_active = 0;
    if (!nlr_build_active_list(ThermalFBSpec::is_active,
                                &active_list, &num_active, &num_global_active,
                                "thermalfb_active_list")) {
        CPU_Step[CPU_SNIIHEATING] += measure_time();
        return;
    }

    /* ---- Per-active host-fill pack (Vel needed by apply_active_writeback;
     *      Pos/KernelRadius/wt_sum/Msne/Esne/kernel_zero/yields read by the
     *      inline pair kernel). Lives on the toplevel stack so it stays valid
     *      across the full run_neighbor_loop call (apply_active_writeback
     *      reads aux->host_locals[slot].Vel during the writeback phase). */
    const int alloc_n = (num_active > 0) ? num_active : 1;
    ThermalFBLocalIn *host_locals = (ThermalFBLocalIn *)
        mymalloc("thermalfb_host_locals", alloc_n * sizeof(ThermalFBLocalIn));
    for (int a = 0; a < num_active; a++) {
        thermal_fb_local_fill(active_list[a], P, CellP, &host_locals[a]);
    }

    ThermalFBSpec::Aux aux;
    aux.host_locals = host_locals;

    /* ---- Engine handoff. Inside run_neighbor_loop:
     *      Mode A: detector_begin → writeback_begin (snapshot ghost fields)
     *              → device kernel → writeback_end (snapshot-diff reverse-comm
     *              → home ranks via manifest bundle) → detector_end →
     *              per-active apply_active_writeback (source-side host writes
     *              to P[i].Mass / dp via aux->host_locals[slot].Vel).
     *      Mode B: per-active local + remote walkers; no ghost bundle (j-side
     *              writes are local). Oracle dual-walk routes through
     *              set_oracle_brute_pass(ctx, true) for the brute pass.
     *      Phase 4.A.0 populate_device_context stages host_locals → UVM
     *      ctx.per_active_local before kernel launch; cleanup frees UVM. */
    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    args.aux         = &aux;
    run_neighbor_loop<ThermalFBSpec>(args);

    /* Any source active globally means some receiver got a positive-wakeup
     * write via atomic_max in the pair kernel. Set the host-side
     * NeedToWakeupParticles_local flag so process_wake_ups() at the next
     * sync actually scans for and applies the wakeup writes. Setting this
     * unconditionally (when num_global_active > 0) is safe: process_wake_ups
     * only mutates particles with P[i].wakeup != 0, so a no-op scan is
     * harmless when no actual wakeup was written. Thermalfb has no UVM
     * need_wakeup signal of its own (unlike hydro_force / ags_force); this
     * is the simplest collective-safe equivalent. */
    extern int NeedToWakeupParticles_local;
    if (num_global_active > 0) NeedToWakeupParticles_local = 1;

    /* ---- Free + return. host_locals must outlive run_neighbor_loop because
     *      apply_active_writeback (called from inside) reads it; freed here
     *      after the runner returns. */
    myfree(host_locals);
    nlr_free_active_list(active_list);

    CPU_Step[CPU_SNIIHEATING] += measure_time();
}


#endif /* GALSF_FB_THERMAL */
