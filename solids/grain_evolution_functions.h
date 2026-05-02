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

#include <math.h>

#include "../declarations/allvars.h"
#include "grain_collisional_outcomes.h"

/* Composition[] index -> stable GrainOutcomeSpecies enum (used by all
 * per-species lookups in grain_collisional_outcomes.h). The fixed layout is:
 *   Composition[0..2]                           = silicate, carbon, iron (refractory)
 *   Composition[3..GRAIN_NUM_SPECIES-1]         = H2O, CO, CO2 ices */
KOKKOS_INLINE_FUNCTION
inline int grain_evolution_composition_index_to_outcome_kind(int s)
{
    if(s == 0) { return GRAIN_OUTCOME_SPECIES_SILICATE; }
    if(s == 1) { return GRAIN_OUTCOME_SPECIES_CARBON;   }
    if(s == 2) { return GRAIN_OUTCOME_SPECIES_IRON;     }
    if(s == 3) { return GRAIN_OUTCOME_SPECIES_H2O_ICE;  }
    if(s == 4) { return GRAIN_OUTCOME_SPECIES_CO_ICE;   }
    return GRAIN_OUTCOME_SPECIES_CO2_ICE;
}

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

#if (GRAIN_EVOLUTION & (8|16))
/* Bits 3+4 SPUTTER (thermal + non-thermal): sputtering of refractory grain
 * material via ion impact. Same physical process for both bits, differing
 * only in the energy source of the impinging ions:
 *   bit 3 THERM_SPUT  -> ions drawn from a hot gas thermal distribution
 *                        (T_thermal from Gas_InternalEnergy)
 *   bit 4 NTHERM_SPUT -> ions seen by a grain drifting through gas
 *                        (T_drift = (1/2) m_p v_drift^2 / k_B)
 * When both bits are active the rates do not add independently -- they are
 * combined into one effective sputter temperature
 *     T_eff = T_thermal + T_drift
 * which is the standard Tielens (1994) / Caselli+(1997) / Hu+(2019)
 * treatment: the impinging-ion energy distribution is a thermal one offset
 * by the bulk drift KE.
 *
 * Per-superparticle stochastic translation of the same Nozawa+(2006)
 * erosion-rate polynomial fits used by the bin-based fluid module
 * (update_dust_sputtering in solids/ism_dust_chemistry.cc); the polynomial
 * coefficients are shared via grain_collisional_outcomes.h.
 *
 * Physics: sputter shrinks each individual grain (ion-impact ejection of
 * surface atoms), so number-of-grains in the super-particle is conserved and
 * the super-particle Mass scales as Grain_Size^3. Refractory vapor is
 * dropped from the total mass budget at this commit (per the locked Phase
 * 17b plan -- bit 5 COND will reclaim it once the gas-phase
 * VolatileSpecies array is wired in C5). Sputter is energy-neutral against
 * the gas thermal pool by construction (the impinging-ion KE comes from gas
 * thermal/kinetic energy, the rearrangement is implicit), so no
 * DtInternalEnergy back-reaction; latent-heat coupling is reserved for
 * bits 5/6.
 *
 * Composition: weighted average of refractory-species erosion rates;
 * Composition[] mass fractions are NOT updated here (would require species-
 * specific da/dt tracked separately, which breaks the monodisperse
 * super-particle assumption -- not worth it for the small differential
 * sputter rates among silicate/carbon/iron). */
KOKKOS_INLINE_FUNCTION
inline void grain_evolution_apply_sputter(int i, struct particle_data *P, double dt)
{
    if(P[i].Mass <= 0 || P[i].Grain_Size <= 0 || P[i].Gas_Density <= 0) { return; }

    /* Build effective sputter temperature from the active source bits. */
    double T_eff = 0.0;
#if (GRAIN_EVOLUTION & 8)
    /* Thermal contribution. Gas T estimated from the kernel-weighted
     * Gas_InternalEnergy already carried on the grain super-particle (see
     * hydro/density.cc:720) using mu = 0.6 (singly-ionized H+He);
     * sputtering activates only at T > 1e4 K where this is accurate to
     * ~10%. A future refinement could read mu from the host gas cell's
     * chemistry, but the polynomial Y(T) varies far more slowly with T
     * than the 10% T-uncertainty would matter. */
    if(P[i].Gas_InternalEnergy > 0) {
        const double mu_ionized = 0.6;
        T_eff += P[i].Gas_InternalEnergy * mu_ionized * (GAMMA_DEFAULT - 1.0) * U_TO_TEMP_UNITS;
    }
#endif
#if (GRAIN_EVOLUTION & 16)
    /* Drift contribution: relative velocity between grain and local gas
     * (Vel and Gas_Velocity, both in code peculiar units) converts to an
     * effective impinging-ion temperature (Tielens 1994 §4.5;
     * Hu+2019 Eq. 22). Same code-unit -> CGS convention used by the grain
     * drag kernel (solids/grain_drag_gpu.cc:72). */
    Vec3<double> dv_code = P[i].Vel - P[i].Gas_Velocity;
    double v_drift_cgs2 = dv_code.norm_sq() / (All.cf_atime*All.cf_atime) * UNIT_VEL_IN_CGS * UNIT_VEL_IN_CGS;
    T_eff += 0.5 * PROTONMASS_CGS * v_drift_cgs2 / BOLTZMANN_CGS;
#endif
    if(T_eff <= 1.0e4) { return; } /* matches the temp>1e4 gate in update_dust_sputtering */

    /* Composition-weighted erosion rate, refractory species only (ices
     * sublimate via bit 6 long before sputtering matters). */
    double refractory_frac = 0.0;
    for(int s = 0; s < GRAIN_NUM_REFRACTORY_SPECIES; s++) { refractory_frac += P[i].Composition[s]; }
    if(refractory_frac <= 0) { return; }
    double logt = log10(T_eff);
    double Y_sput_eff = 0.0;
    for(int s = 0; s < GRAIN_NUM_REFRACTORY_SPECIES; s++) {
        Y_sput_eff += P[i].Composition[s] * grain_outcomes_sputter_erosion_dadt_per_nH(logt, grain_evolution_composition_index_to_outcome_kind(s));
    }
    Y_sput_eff /= refractory_frac;

    /* da/dt = -Y * 1e-4 * nH * 1e9  [cm/Gyr]; matches update_dust_sputtering
     * scaling (the 1e-4 is um->cm, the 1e9 is yr->Gyr; Y is in um/yr cm^3). */
    double rho_gas_cgs = P[i].Gas_Density * UNIT_DENSITY_IN_CGS * All.cf_a3inv;
    double nH_cgs      = HYDROGEN_MASSFRAC * rho_gas_cgs / PROTONMASS_CGS;
    double dadt_cm_per_Gyr = -Y_sput_eff * 1.0e-4 * nH_cgs * 1.0e9 * All.GrainEvolution_ThermalSputteringScaling;
    double da_cm = dadt_cm_per_Gyr * dt * UNIT_TIME_IN_GYR;

    double a_old = P[i].Grain_Size;
    double a_new = a_old + da_cm;
    /* Floor at the runtime Grain_Size_Min (cgs); under heavy sputtering the
     * per-step shrinkage can outrun the floor, in which case clamp here.
     * A future commit could mark the super-particle for deletion when it
     * hits the floor, similar to gas-cell merge/split. */
    if(a_new < All.Grain_Size_Min) { a_new = All.Grain_Size_Min; }
    if(a_new >= a_old) { return; } /* numerical noise or already at floor */

    double size_ratio = a_new / a_old;
    P[i].Grain_Size = a_new;
    P[i].Mass *= size_ratio * size_ratio * size_ratio; /* M ~ a^3 (number conserved) */
}
#endif /* GRAIN_EVOLUTION & (8|16) */

/* Per-superparticle local step. Dispatches to the active local-operator bits
 * (3|4|5|6 = THERM_SPUT|NTHERM_SPUT|COND|SUBL). */
KOKKOS_INLINE_FUNCTION
inline void grain_evolution_local_step(int i, struct particle_data *P, double dt)
{
#if (GRAIN_EVOLUTION & (8|16))
    grain_evolution_apply_sputter(i, P, dt);
#endif
#if (GRAIN_EVOLUTION & 32)
    /* C5: bit 5 COND (condensation/mantle growth from VolatileSpecies). */
#endif
#if (GRAIN_EVOLUTION & 64)
    /* C6: bit 6 SUBL (sublimation/desorption, inverse of bit 5). */
#endif
#if !(GRAIN_EVOLUTION & (8|16|32|64))
    (void)i; (void)P; (void)dt;
#endif
}

#endif /* GRAIN_EVOLUTION */
#endif /* GRAIN_EVOLUTION_FUNCTIONS_H */
