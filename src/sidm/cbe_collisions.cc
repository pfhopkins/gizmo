/* cbe_collisions.cc -- top-level CBE collision-operator driver.
 *
 * Operator-split, run-level routine analogous to cooling_parent_routine: called
 * once per timestep from calculate_non_standard_physics() (core/run.cc), it loops
 * over active CBE particles and applies the enabled local (intra-cell) collision
 * operator to each cell's velocity basis functions. The per-cell physics lives in
 * the device-callable worker cbe_collision_apply (sidm/cbe_collisions_functions.h).
 *
 * Fully gated: a no-op unless CBE_INTEGRATOR_COLLISIONS is compiled AND the
 * cross-section parameter CBECollisionCrossSection > 0.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#if defined(CBE_INTEGRATOR) && defined(CBE_INTEGRATOR_COLLISIONS)
#include "cbe_collisions_functions.h"

/* Top-level collision driver. Intra-cell + embarrassingly parallel (no
 * neighbors), so this is a plain active-particle loop over the local set;
 * a Kokkos parallel_for is a drop-in future swap (the worker is device-callable). */
void cbe_collision_parent_routine(void)
{
    const double sigma_over_mu = All.CBECollisionCrossSection;   /* code units (area/mass) */
    if(!(sigma_over_mu > 0)) return;                             /* no-op if unset */

    for(int i : ActiveParticleList) {
        if(!CBE_INTEGRATOR_DOES_TYPE(P[i].Type)) continue;
        if(!(P[i].Mass > 0)) continue;

        const double dt_phys = get_particle_timestep_in_physical(i, P);
        if(!(dt_phys > 0)) continue;
        const double V_a = get_particle_volume_ags(i);
        if(!(V_a > 0)) continue;

        const double rho_phys = (P[i].Mass / V_a) * All.cf_a3inv;   /* physical density */
        cbe_collision_apply(P[i].CBE_basis_moments, rho_phys, sigma_over_mu,
                            All.cf_atime, dt_phys, CBE_COLLISION_MODEL);
    }
}

#else

void cbe_collision_parent_routine(void) { }   /* gated-off stub */

#endif  /* CBE_INTEGRATOR && CBE_INTEGRATOR_COLLISIONS */
