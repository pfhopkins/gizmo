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
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "grain_collisional_outcomes.h"

/* Composition[] index -> stable GrainOutcomeSpecies enum (used by all
 * per-species lookups in grain_collisional_outcomes.h). The fixed layout is:
 *   Composition[0..2]                           = silicate, carbon, iron (refractory)
 *   Composition[3..GRAIN_NUM_SPECIES-1]         = H2O, CO, CO2 ices */
KOKKOS_INLINE_FUNCTION
int grain_evolution_composition_index_to_outcome_kind(int s)
{
    if(s == 0) { return GRAIN_OUTCOME_SPECIES_SILICATE; }
    if(s == 1) { return GRAIN_OUTCOME_SPECIES_CARBON;   }
    if(s == 2) { return GRAIN_OUTCOME_SPECIES_IRON;     }
    if(s == 3) { return GRAIN_OUTCOME_SPECIES_H2O_ICE;  }
    if(s == 4) { return GRAIN_OUTCOME_SPECIES_CO_ICE;   }
    return GRAIN_OUTCOME_SPECIES_CO2_ICE;
}

#if (GRAIN_EVOLUTION & 7)
/* Pairwise outcome resolver: decides COAG / FRAG / SHAT based on |dv| vs
 * per-species thresholds, given that a collision has fired (the SIDM
 * scatter prob-gate already handled "did a collision happen"). Mutates
 * the local-side accumulators on out (gathered to P[ii] in the host
 * post-pass in gravity/ags_rkern.cc) and atomically mutates P[j].
 *
 * Per-pair routing: SIDM dedup ensures only the lower-ID side processes
 * each pair (sidm_core_flux_functions.h:62), so "local" is the actor.
 * COAG outcome convention: local always absorbs j (mass-conservative;
 * statistical bias toward lower-ID-as-absorber is harmless for stochastic
 * coagulation). FRAG/SHAT (C8/C9): both local and j shrink; symmetric.
 *
 * Thresholds: v_coag from grain_outcomes_v_coag_dominik(); v_shat from
 * grain_outcomes_elastic_props(). C7 uses silicate elastic props as the
 * default species lookup (composition-weighted thresholds are a future
 * refinement). */
template <typename LocalT, typename OutT>
KOKKOS_INLINE_FUNCTION
void grain_evolution_resolve_pairwise(const LocalT &local, int j, struct particle_data *P, const Vec3<double> &dv, OutT &out)
{
    if(local.Mass <= 0 || local.Grain_Size <= 0) { return; }
    if(P[j].Mass <= 0 || P[j].Grain_Size <= 0) { return; }
    double dv_mag = sqrt(dv.norm_sq()) * UNIT_VEL_IN_CGS / All.cf_atime; /* cgs cm/s */

    /* Elastic / breakup thresholds. Use silicate defaults for both pair
     * sides in C7; refining to composition-weighted thresholds is a clean
     * extension that doesn't change the resolver's structural shape. */
    struct GrainOutcomeElasticProps ep = grain_outcomes_elastic_props(GRAIN_OUTCOME_SPECIES_SILICATE);
    double a_i_cm = local.Grain_Size;
    double a_j_cm = (double)P[j].Grain_Size;
    double v_coag = grain_outcomes_v_coag_dominik(a_i_cm, a_j_cm, ep.gamma, ep.E_young, ep.nu_poisson, All.Grain_Internal_Density);
    if(v_coag > ep.v_shat) { v_coag = ep.v_shat; } /* clamp -- match ISMDustChem update_dust_shattering_and_coagulation:1619 */

#if (GRAIN_EVOLUTION & 1)
    /* Bit 0 COAG: low-velocity collision -> stick. Local absorbs j fully.
     * Mass conservation between the pair: M_local += M_j, M_j -> 0.
     * Composition mixing: per-species mass accumulator on out so the
     * host-side post-pass can do the mass-fraction renormalization with
     * full multi-source visibility (multiple j-neighbors of the same i
     * may all be absorbed in one step). */
    if(dv_mag < v_coag * All.GrainEvolution_StickingCoeff) {
        double M_j = (double)P[j].Mass;
        if(M_j <= 0) { return; }
        out.Grain_DeltaCoagMass += M_j;
        for(int s = 0; s < GRAIN_NUM_SPECIES; s++) {
            out.Grain_DeltaCoag_CompositionMass[s] += M_j * (double)P[j].Composition[s];
        }
        /* Atomically zero out j -- it is now fully absorbed. The grain
         * super-particle persists in the array but is inert
         * (Mass = 0 short-circuits all GRAIN_FLUID code paths). */
        Kokkos::atomic_store(&P[j].Mass,       (MyDouble)0.0);
        Kokkos::atomic_store(&P[j].Grain_Size, (MyFloat)0.0);
        return;
    }
#endif

#if (GRAIN_EVOLUTION & 2)
    /* Bit 1 FRAG: intermediate-velocity collision (v_coag <= |dv| < v_shat).
     * Both grains experience mild chipping -- size drops slightly while
     * super-particle mass is preserved (chipped material reinterpreted as
     * smaller grains in the same SP -> N_phys grows implicitly through
     * the M / a^3 ratio).
     *
     * Fragmentation efficiency model: smooth ramp from 0 at v = v_coag to
     * fragmentation_max = 0.1 at v = v_shat (10% mass-equivalent size
     * shrink per collision event near the SHAT threshold). Closed-form
     * monotonic in (dv/v_shat)^2 so the FRAG branch hands off smoothly to
     * SHAT in C9 without a discontinuity.
     *
     * Size update: a_new = a_old * (1 - frag_eff)^(1/3) -- the cube-root
     * reflects the conserved super-particle mass with implicit N growth. */
    if(dv_mag >= v_coag * All.GrainEvolution_StickingCoeff && dv_mag < ep.v_shat) {
        double frag_eff = 0.1 * (dv_mag * dv_mag) / (ep.v_shat * ep.v_shat);
        if(frag_eff > 0.5) { frag_eff = 0.5; }
        double frag_factor = pow(1.0 - frag_eff, 1.0/3.0);
        /* Local i-side accumulator (multiplicative; applied in host
         * post-pass alongside the COAG mass update). */
        out.Grain_DeltaErosionFrac *= frag_factor;
        /* J-side: atomic CAS loop multiplicatively shrinks Grain_Size.
         * Multiple i-neighbors of the same j may FRAG it in the same
         * step; the loop preserves the cumulative product. */
        MyFloat old_a, new_a, returned_a;
        do {
            old_a = Kokkos::atomic_load(&P[j].Grain_Size);
            if(old_a <= 0) { break; }
            new_a = (MyFloat)((double)old_a * frag_factor);
            returned_a = Kokkos::atomic_compare_exchange(&P[j].Grain_Size, old_a, new_a);
        } while(returned_a != old_a);
        return;
    }
#endif

#if (GRAIN_EVOLUTION & 4)
    /* Bit 2 SHAT: high-velocity collision (|dv| >= v_shat). Both grains
     * shatter into many small fragments. In the monodisperse super-
     * particle representation, the fragments stay in the same SP at
     * smaller size (mass-conservative; N_phys grows implicitly through
     * the M / a^3 ratio).
     *
     * Shattering efficiency model: ramp linearly from 0.5 at |dv| = v_shat
     * up to a hard cap at 0.9 (cannot shrink the grain to zero in a
     * single event):
     *   shat_eff = min(0.9, 0.5 * |dv| / v_shat)
     *   shat_factor = (1 - shat_eff)^(1/3)
     * The FRAG -> SHAT transition is intentionally discontinuous at
     * v_shat (FRAG saturates at frag_eff = 0.1; SHAT starts at
     * shat_eff = 0.5) -- this matches the bin-based ISMDustChem model
     * where v_shat is a real threshold, not a smooth crossover. */
    if(dv_mag >= ep.v_shat) {
        double shat_eff = 0.5 * dv_mag / ep.v_shat;
        if(shat_eff > 0.9) { shat_eff = 0.9; }
        double shat_factor = pow(1.0 - shat_eff, 1.0/3.0);
        out.Grain_DeltaErosionFrac *= shat_factor;
        MyFloat old_a, new_a, returned_a;
        do {
            old_a = Kokkos::atomic_load(&P[j].Grain_Size);
            if(old_a <= 0) { break; }
            new_a = (MyFloat)((double)old_a * shat_factor);
            returned_a = Kokkos::atomic_compare_exchange(&P[j].Grain_Size, old_a, new_a);
        } while(returned_a != old_a);
        return;
    }
#endif
    (void)dv_mag; (void)ep;
}
#endif

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
void grain_evolution_apply_sputter(int i, struct particle_data *P, double dt)
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

#if (GRAIN_EVOLUTION & (32|64))
/* Bits 5+6 COND+SUBL: condensation/sublimation of volatile species (H2O,
 * CO, CO2 ices) on grain mantles, with full mass + latent-heat back-
 * reaction to the gas. Same operator handles both directions per species
 * via the sign of the per-species net rate.
 *
 * Per-species rate (Hertz-Knudsen kinetic theory):
 *   For COND: dM/dt = sticking * pi * a^2 * v_th(T,m_v) * rho_v * N_phys
 *     with N_phys = M_super / ((4pi/3) a^3 rho_bulk_grain)
 *     => dM/dt = M_super * 3 * sticking * v_th * rho_v / (4 * a * rho_bulk)
 *     where rho_v = local volatile mass density [cgs] = rho_gas_cgs *
 *     Gas_VolatileSpecies[k]; v_th = sqrt(8 k T / (pi m_v)).
 *   For SUBL (C6): dM/dt is the inverse net flux; same rate prefactor with
 *     rho_v replaced by the equilibrium-vapor pressure / (k T) * m_v.
 *
 * Stability clamp: the per-step COND mass exchange dM is capped at half
 * the available local volatile mass M_v_local = (4pi/3) h^3 * rho_v
 * (kernel-volume estimate). Without this, the GIZMO timestep limiter
 * would only catch the problem after-the-fact via VolatileSpecies going
 * negative on the writeback.
 *
 * Sign convention on the accumulators (zeroed at top of grain_drag_kernel,
 * scattered to gas neighbors by the extended grain_backrx_pair_kernel):
 *   Grain_DeltaVolatileMass[k] > 0 -> COND (mass leaves gas, joins grain)
 *   Grain_DeltaVolatileMass[k] < 0 -> SUBL (mass leaves grain, joins gas)
 *   Grain_DeltaInternalEnergyHeating > 0 -> COND latent release heats gas
 *   Grain_DeltaInternalEnergyHeating < 0 -> SUBL latent absorption cools gas
 *
 * C5b is COND only -- bit 5 fires below T_snow[k] for each ice species.
 * C6 will add the SUBL branch (T > T_snow[k]) symmetrically. */
KOKKOS_INLINE_FUNCTION
void grain_evolution_apply_cond_subl(int i, struct particle_data *P, double dt)
{
    if(P[i].Mass <= 0 || P[i].Grain_Size <= 0 || P[i].Gas_Density <= 0) { return; }
    if(P[i].Gas_InternalEnergy <= 0) { return; }
    if(P[i].KernelRadius <= 0) { return; }

    /* Gas T from kernel-weighted Gas_InternalEnergy. Use mu = 2.3 (cold
     * molecular H2 + He), accurate at the < 200 K regime where ice
     * condensation/sublimation actually drives mantle dynamics. */
    const double mu_molecular = 2.3;
    double T_gas = P[i].Gas_InternalEnergy * mu_molecular * (GAMMA_DEFAULT - 1.0) * U_TO_TEMP_UNITS;
    if(T_gas <= 0) { return; }

    double rho_gas_cgs = (double)P[i].Gas_Density * UNIT_DENSITY_IN_CGS * All.cf_a3inv;
    double a_cm        = (double)P[i].Grain_Size; /* already in cm */
    double rho_bulk    = All.Grain_Internal_Density; /* g/cm^3 */
    double sticking    = (All.GrainEvolution_StickingCoeff > 0) ? All.GrainEvolution_StickingCoeff : 1.0;
    double h_phys_cm     = (double)P[i].KernelRadius * All.cf_atime * UNIT_LENGTH_IN_CGS;
    double V_kernel_cgs  = (4.0 * M_PI / 3.0) * h_phys_cm * h_phys_cm * h_phys_cm;
    double M_super_cgs   = (double)P[i].Mass * UNIT_MASS_IN_CGS;
    double dt_cgs        = dt * UNIT_TIME_IN_CGS;
    double M_old_code    = (double)P[i].Mass;

    /* Per-ice-species net flux. Handles VolatileSpecies[0..2] =
     * H2O/CO/CO2 <-> grain Composition[3..5] mantles. VolatileSpecies[3]
     * (refractory vapor) is intentionally NOT handled here -- there is no
     * source for it yet in Phase 17b (sputter -> vapor coupling reserved
     * for later). dM > 0 = COND (gas->grain), dM < 0 = SUBL (grain->gas).
     * Direction is set by sign of (T_snow - T_gas):
     *   T_gas < T_snow_k -> COND (gas-driven, magnitude limited by
     *                       available volatile in kernel volume)
     *   T_gas > T_snow_k -> SUBL (mantle-driven via Clausius-Clapeyron
     *                       equilibrium-vapor-density excess; magnitude
     *                       limited by available mantle on the grain) */
    double dM_per_species_code[GRAIN_NUM_VOLATILE_SPECIES] = {0};
    double dE_release_total_cgs = 0.0;
    for(int k = 0; k < GRAIN_EVOLUTION_NUM_ICE; k++)
    {
        double T_snow_k = grain_outcomes_ice_snowline_T(k);
        double m_v_amu  = grain_outcomes_ice_molecular_weight(k);
        double m_v_g    = m_v_amu * PROTONMASS_CGS;
        double L_cgs    = grain_outcomes_ice_latent_heat_cgs(k);
        double v_th_cgs = sqrt(8.0 * BOLTZMANN_CGS * T_gas / (M_PI * m_v_g));
        double dM_cgs   = 0.0;

        if(T_gas < T_snow_k) {
#if (GRAIN_EVOLUTION & 32)
            /* COND branch (bit 5). Hertz-Knudsen condensation rate per
             * super-particle, capped at half the locally-available
             * volatile mass per step. */
            if(P[i].Gas_VolatileSpecies[k] <= 0) { continue; }
            double rho_v_cgs = rho_gas_cgs * (double)P[i].Gas_VolatileSpecies[k];
            double dMdt_cgs  = M_super_cgs * 3.0 * sticking * v_th_cgs * rho_v_cgs / (4.0 * a_cm * rho_bulk);
            dM_cgs           = dMdt_cgs * dt_cgs;
            double dM_cap    = 0.5 * V_kernel_cgs * rho_v_cgs;
            if(dM_cgs > dM_cap) { dM_cgs = dM_cap; }
#endif
        } else {
#if (GRAIN_EVOLUTION & 64)
            /* SUBL branch (bit 6). Clausius-Clapeyron equilibrium-vapor-
             * density excess: at T = T_snow the system is in equilibrium
             * with the local volatile field; at T > T_snow the saturation
             * vapor density rises exponentially and the net flux reverses.
             * Reference n_v_sat(T_snow) is taken to equal the current
             * local n_v -- a one-parameter approximation that gives the
             * correct sign and dominant temperature dependence (the exp
             * factor) without requiring per-species P_ref tables. */
            int comp_idx = GRAIN_NUM_REFRACTORY_SPECIES + k;
            if(P[i].Composition[comp_idx] <= 0) { continue; }
            double exp_arg   = L_cgs * m_v_g / BOLTZMANN_CGS * (1.0/T_snow_k - 1.0/T_gas);
            /* Cap exp_arg to avoid overflow; sublimation timescale is
             * exponentially short above T_snow already at exp_arg ~ 30. */
            if(exp_arg > 50.0) { exp_arg = 50.0; }
            double rho_v_local_cgs = rho_gas_cgs * (double)P[i].Gas_VolatileSpecies[k];
            double rho_v_sat_cgs   = rho_v_local_cgs * exp(exp_arg);
            /* If local n_v is itself zero, use a tiny floor so SUBL still
             * proceeds (otherwise mantles never sublimate in vacuum). */
            if(rho_v_sat_cgs <= 0) { rho_v_sat_cgs = 1e-30 * rho_gas_cgs; }
            double dMdt_cgs  = -M_super_cgs * 3.0 * sticking * v_th_cgs * rho_v_sat_cgs / (4.0 * a_cm * rho_bulk);
            dM_cgs           = dMdt_cgs * dt_cgs;
            /* Cap at half the available mantle mass for this species. */
            double M_mantle_k_cgs = M_super_cgs * (double)P[i].Composition[comp_idx];
            double dM_floor_cgs   = -0.5 * M_mantle_k_cgs;
            if(dM_cgs < dM_floor_cgs) { dM_cgs = dM_floor_cgs; }
#endif
        }

        if(dM_cgs == 0) { continue; }
        dM_per_species_code[k] = dM_cgs / UNIT_MASS_IN_CGS;
        /* Latent-heat sign: COND (dM > 0) releases L into gas (warms);
         * SUBL (dM < 0) absorbs L from gas (cools). dE has same sign as dM. */
        dE_release_total_cgs += dM_cgs * L_cgs;
    }

    /* Sum mass change over all species. */
    double dM_total_code = 0.0;
    for(int k = 0; k < GRAIN_EVOLUTION_NUM_ICE; k++) { dM_total_code += dM_per_species_code[k]; }
    if(dM_total_code == 0) { return; }

    /* Update grain Mass + Composition. Number-of-grains stays implicit
     * (mantle change affects M and Composition; the grain core is
     * untouched here -- bit 3 sputter handles refractory loss). */
    double M_new_code = M_old_code + dM_total_code;
    if(M_new_code <= 0) { return; } /* defensive: avoid divide-by-zero */
    for(int s = 0; s < GRAIN_NUM_SPECIES; s++) {
        double M_species_s = M_old_code * (double)P[i].Composition[s];
        if(s >= GRAIN_NUM_REFRACTORY_SPECIES) {
            int ice_idx = s - GRAIN_NUM_REFRACTORY_SPECIES;
            if(ice_idx < GRAIN_EVOLUTION_NUM_ICE) { M_species_s += dM_per_species_code[ice_idx]; }
        }
        if(M_species_s < 0) { M_species_s = 0; } /* defensive: a clamped
            SUBL can momentarily push a species slightly negative */
        P[i].Composition[s] = (MyFloat)(M_species_s / M_new_code);
    }
    P[i].Mass = M_new_code;

    /* Write back-reaction accumulators. Sign convention:
     *   Grain_DeltaVolatileMass[k] > 0 -> COND (gas mass-fraction drops)
     *   Grain_DeltaVolatileMass[k] < 0 -> SUBL (gas mass-fraction rises)
     *   Grain_DeltaInternalEnergyHeating > 0 -> COND latent release heats gas
     *   Grain_DeltaInternalEnergyHeating < 0 -> SUBL latent absorption cools gas */
    for(int k = 0; k < GRAIN_EVOLUTION_NUM_ICE; k++) {
        P[i].Grain_DeltaVolatileMass[k] += (MyFloat)dM_per_species_code[k];
    }
    P[i].Grain_DeltaInternalEnergyHeating += (MyFloat)(dE_release_total_cgs / UNIT_ENERGY_IN_CGS);
}
#endif

/* Per-superparticle local step. Dispatches to the active local-operator bits
 * (3|4|5|6 = THERM_SPUT|NTHERM_SPUT|COND|SUBL). All operators run inside
 * grain_drag_kernel (solids/grain_drag_gpu.cc) and any back-reaction they
 * generate piggybacks on the existing GRAIN_BACKREACTION ghost-writeback
 * infrastructure (mesh/ghost_writeback.cc::ghost_writeback_grainbackrx,
 * extended for the new fields). No parallel pass. */
KOKKOS_INLINE_FUNCTION
void grain_evolution_local_step(int i, struct particle_data *P, double dt)
{
#if (GRAIN_EVOLUTION & (8|16))
    grain_evolution_apply_sputter(i, P, dt);
#endif
#if (GRAIN_EVOLUTION & (32|64))
    grain_evolution_apply_cond_subl(i, P, dt);
#endif
#if !(GRAIN_EVOLUTION & (8|16|32|64))
    (void)i; (void)P; (void)dt;
#endif
}

#endif /* GRAIN_EVOLUTION */
#endif /* GRAIN_EVOLUTION_FUNCTIONS_H */
