/* galaxy_sf/dm_dispersion.cc — DM velocity dispersion caller (runner-template path).
 *
 * Provides disp_density(), which drives the iterative runner-template port of
 * the DM-dispersion neighbor loop (DMDispersionSpec). Replaces the legacy
 * do-while + disp_density_evaluate_gpu() structure with a simplified caller
 * that mirrors the density() pattern in hydro/density_loop.cc.
 *
 * Ghost lifecycle: runner owns Mode A ghost import via args.ghost_safety_factor;
 * caller does NOT call gizmo_density_prep_ghosts or ghost_exchange_cleanup.
 *
 * This file was originally written by Qirong Zhu, for GIZMO, based on
 * Phil Hopkins's adaptive gravitational softening routine. Modified by Phil
 * and adapted to the runner-template pattern by Claude for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"                        /* MUST precede dm_dispersion_loop.h */
#include "../mesh/ghost_symlist_lifecycle.h"        /* gizmo_ghost_safety_factor */
#include "../system/gpu_particles_arena.h"          /* gpu_particles_arena_invalidate */
#include "dm_dispersion_loop.h"

#ifdef DM_DISPERSION_LOOP_ACTIVE

void disp_density(void)
{
    const struct global_data_all_processes *host_all = nlr_host_all_ptr();

    CPU_Step[CPU_MISC] += measure_time();
    const double t00_truestart = my_second();

    const double gsl_safety = gizmo_ghost_safety_factor();

    /* Build active list (gas particles satisfying DMDispersionSpec::is_active). */
    std::vector<int> active_list;
    active_list.reserve(N_gas);
    for (int i : ActiveParticleList) {
        if (DMDispersionSpec::is_active(i)) active_list.push_back(i);
    }
    const int num_active = (int)active_list.size();

    /* Pre-pass: invalidate GPU arena and clamp KernelRadiusDM to a safe
     * initial range, mirroring density()'s prepass (density_loop.cc:1431-1471).
     * Legacy disp_density() initialized Left[i]=Right[i]=0 (handled by runner
     * zeroing IterScratch) and NumNgbDM=0 (handled by finalize scatter).
     *
     * Use the same per-particle maxsoft as after_iter (DMIN(MaxKernelRadius,
     * 10*P[i].KernelRadius)) so the CSR built at iter-0 matches the
     * convergence cap. Clamping only against MaxKernelRadius can leave
     * KernelRadiusDM above the per-particle cap, causing after_iter to
     * force-converge at a too-large radius without a re-evaluation. */
    gpu_particles_arena_invalidate();
    {
        const double global_maxsoft = host_all->MaxKernelRadius;
        for (int i : active_list) {
            const double per_particle_maxsoft =
                DMIN(global_maxsoft, 10.0 * (double)P[i].KernelRadius);
            if ((CellP[i].KernelRadiusDM <= 0) || !isfinite(CellP[i].KernelRadiusDM)
                    || (CellP[i].KernelRadiusDM > 0.99 * per_particle_maxsoft)) {
                CellP[i].KernelRadiusDM = (MyFloat)(0.99 * per_particle_maxsoft);
            }
        }
    }

    /* Allocate Aux buffers. */
    DMDispersionSpec::Aux aux;
    aux.per_active_final_accum.resize(num_active);
    aux.per_active_final_h    .resize(num_active);

    /* Single DM-only subgroup (j_type_bitmask = 1u<<1 = DM).
     * Must match Spec::neighbor_type_mask. */
    NlrSubgroup sg{};
    sg.j_type_bitmask   = (1u << 1);
    sg.active_indices   = active_list.data();
    sg.num_active_local = num_active;
    std::vector<NlrSubgroup> subgroups{sg};

    /* Build args and drive runner. */
    neighbor_loop_args_iterative args{};
    static_cast<neighbor_loop_args&>(args) = nlr_default_args();
    args.P                   = P;
    args.CellP               = (host_all->TotN_gas > 0) ? CellP : nullptr;
    args.num_total           = NumPart;
    args.active_list         = active_list.data();
    args.num_active          = num_active;
    args.aux                 = &aux;
    args.num_subgroups       = (int)subgroups.size();
    args.subgroups           = subgroups.data();
    args.ghost_safety_factor = gsl_safety;
    run_neighbor_loop_iterative<DMDispersionSpec>(args);

    /* Finalize: normalize velocity stats and scatter to CellP[]. */
    dm_dispersion_finalize_post_runner(args);

    /* Timing (mirrors legacy dm_dispersion.cc:222-224 bins). */
    const double t1 = my_second();
    WallclockTime = t1;
    const double timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_AGSDENSMISC] += timeall;
    if (ThisTask == 0) {
        PRINT_STATUS("  dm_dispersion done (%.4f s) via runner", timeall);
    }
}

#endif /* DM_DISPERSION_LOOP_ACTIVE */
