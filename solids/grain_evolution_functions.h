/* grain_evolution_functions.h -- per-superparticle stochastic grain evolution
 * (Phase 17b). Header-only, KOKKOS_INLINE_FUNCTION-callable, no rdc.
 *
 * Provides (across the GRAIN_EVOLUTION bitfield):
 *   pairwise outcomes (bits 0|1|2 = COAG|FRAG|SHAT):
 *       grain_evolution_resolve_pairwise(...)
 *           - called from inside the GRAIN_COLLISIONS pair kernel after a
 *             scatter is resolved, picks coag/frag/shat branch by |dv|, mass
 *             ratio and species thresholds, mutates Mass+Grain_Size+Composition
 *             on both super-particles atomically.
 *
 *   local operators (bits 3|4|5|6 = THERM_SPUT|NTHERM_SPUT|COND|SUBL):
 *       grain_evolution_local_step(...)
 *           - single-particle local update fired from the parent loop in
 *             grain_evolution.cc (called from core/run.cc next to the cooling
 *             routine). Reads the cached local Gas_* fields and the host gas
 *             cell's VolatileSpecies array, writes Mass + Grain_Size +
 *             Composition[], and (bits 5|6) deposits or withdraws latent heat
 *             into CellP[host].DtInternalEnergy.
 *
 * No physics is implemented in C1 -- this is the no-op scaffolding pass.
 * Operators populate over commits C3..C9; activation gates them on individual
 * bits so each commit either no-ops or activates exactly one bit (bisectable).
 *
 * Yield curves, threshold velocities, sticking coefficients, latent heats and
 * per-species bulk densities live in solids/grain_collisional_outcomes.h
 * (extracted from solids/ism_dust_chemistry.cc in commit C2 so the fluid
 * ISMDustChem module and this per-superparticle module share one
 * source-of-truth for the physics constants).
 *
 * Pairwise + local operators always live in the same compile unit (called
 * back-to-back per timestep from the same dispatch); no benefit to splitting.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GRAIN_EVOLUTION_FUNCTIONS_H
#define GRAIN_EVOLUTION_FUNCTIONS_H

#ifdef GRAIN_EVOLUTION

#include "../declarations/allvars.h"

/* Pairwise outcome resolver. C1 stub: returns without touching state.
 * Activated under (GRAIN_EVOLUTION & 7) in C7-C9. */
template <typename LocalT>
KOKKOS_INLINE_FUNCTION
void grain_evolution_resolve_pairwise(const LocalT &local, int j, struct particle_data *P, double dv_mag)
{
#if (GRAIN_EVOLUTION & 7)
    (void)local; (void)j; (void)P; (void)dv_mag;
    /* C7: bit 0 COAG; C8: bit 1 FRAG; C9: bit 2 SHAT. */
#else
    (void)local; (void)j; (void)P; (void)dv_mag;
#endif
}

/* Per-superparticle local step. C1 stub. Activated bit-by-bit in C3-C6. */
KOKKOS_INLINE_FUNCTION
void grain_evolution_local_step(int i, struct particle_data *P, double dt)
{
#if (GRAIN_EVOLUTION & (8|16|32|64))
    (void)i; (void)P; (void)dt;
    /* C3: bit 3 THERM_SPUT; C4: bit 4 NTHERM_SPUT; C5: bit 5 COND; C6: bit 6 SUBL. */
#else
    (void)i; (void)P; (void)dt;
#endif
}

#endif /* GRAIN_EVOLUTION */
#endif /* GRAIN_EVOLUTION_FUNCTIONS_H */
