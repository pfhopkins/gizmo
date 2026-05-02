/* grain_evolution.cc -- top-level dispatch for per-superparticle grain evolution.
 *
 * Phase-17b owning translation unit. Provides:
 *   grain_evolution_local_parent_routine() -- single call site from core/run.cc,
 *       fired alongside the cooling parent routine. Loops over local Type-3 grain
 *       super-particles and applies the active local-operator bits (3|4|5|6) of
 *       GRAIN_EVOLUTION via grain_evolution_local_step() in
 *       solids/grain_evolution_functions.h.
 *
 * Pairwise outcomes (bits 0|1|2) are NOT dispatched from here -- they piggyback
 * on the existing GRAIN_COLLISIONS pair kernel in sidm/sidm_core_flux_functions.h
 * (wired up in commits C7-C9).
 *
 * C1 is the no-op scaffolding pass: the parent routine exists and is called
 * each step, but grain_evolution_local_step() returns immediately under any
 * GRAIN_EVOLUTION value. Operators populate over commits C3..C6.
 *
 * This is the owning TU: any non-inline symbols (e.g. helper tables that may
 * later need single-definition storage) belong here, per
 * feedback_gpu_tu_symbol_ownership.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"
#include "grain_evolution_functions.h"

#ifdef GRAIN_EVOLUTION

void grain_evolution_local_parent_routine(void)
{
#if (GRAIN_EVOLUTION & (8|16|32|64))
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 3 || P[i].Mass <= 0) continue;
        if(!TimeBinActive[P[i].TimeBin]) continue;
        double dt = get_particle_timestep_in_physical(i, P);
        if(dt <= 0) continue;
        grain_evolution_local_step(i, P, dt);
    }
#endif
}

#endif /* GRAIN_EVOLUTION */
