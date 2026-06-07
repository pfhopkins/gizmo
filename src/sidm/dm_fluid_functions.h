/* dm_fluid_functions.h — placeholder dark-fluid EOS hook for
 * HYDRO_MULTIFLUID_DM (Type=0+FluidType=FLUID_DM particles).
 *
 * Defaults: trivial adiabatic γ=5/3 EOS, strictly local. Included by
 * eos/eos_functions.h's HYDRO_MULTIFLUID_DM block for set_eos_pressure_impl's
 * dark-fluid early return. No cooling-table dependency.
 *
 * The HYDRO_MULTIFLUID_DM_COOLING cooling chain (atomic + molecular ADM,
 * distilled from gizmo_adm_roy_v2 / Roy et al.) lives in
 * sidm/dm_cooling_functions.h — kept out of this header so broad EOS
 * consumers transitively pulled by rt_functions.h / cooling_functions.h do
 * not also pull DMCoolTables-dependent device-callable inline functions.
 *
 * Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifdef HYDRO_MULTIFLUID_DM

#include "../declarations/multifluid_helpers.h"

/* Trivial dark-fluid EOS: P = (γ-1) ρ u with γ=5/3. Pure ideal-gas adiabat,
 * no chemistry / cooling / radiation coupling. Mirrors the structure of
 * eos.cc::set_eos_pressure's early-pre-cooling branch.
 *
 * KOKKOS_INLINE_FUNCTION (device-clean): body is pure arithmetic on macro
 * constants (GAMMA_DEFAULT, PROTONMASS_CGS, BOLTZMANN_CGS, UNIT_*) plus
 * device-callable cell_data accessors (density_for_energy, InternalEnergyPred).
 * Lets eos_functions.h::set_eos_pressure_impl dispatch to it unconditionally
 * under HYDRO_MULTIFLUID_DM  */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
static KOKKOS_INLINE_FUNCTION void set_dark_eos_pressure(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double gamma_eos_index = GAMMA_DEFAULT;
    double press = (gamma_eos_index - 1) * cell[i].InternalEnergyPred * cell[i].density_for_energy();
    cell[i].Gamma = gamma_eos_index;
    cell[i].Pressure = press;
    double mu_dark = 1.0;
    cell[i].Temperature = (MyFloat)( cell[i].InternalEnergyPred * (gamma_eos_index - 1.0) * mu_dark
                                     * PROTONMASS_CGS / BOLTZMANN_CGS
                                     * UNIT_ENERGY_IN_CGS / UNIT_MASS_IN_CGS );
    cell[i].SoundSpeed = (MyFloat)sqrt(gamma_eos_index * press / cell[i].density_for_energy());
}


/* HYDRO_MULTIFLUID_DM_COOLING cooling chain + do_dark_cooling_for_particle entry
   moved to sidm/dm_cooling_functions.h so that broad EOS consumers (e.g.
   eos/eos_functions.h, transitively pulled by rt_functions.h /
   cooling_functions.h) do NOT pull the DMCoolTables-dependent cooling chain.
   Mirrors the cooling.cc / cooling_functions.h / cooling_tables.h split. */

#endif /* HYDRO_MULTIFLUID_DM */
