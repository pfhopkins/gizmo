/* grain_collisional_outcomes.h -- shared physics-constants header for grain
 * processing routines.
 *
 * Purpose: provide ONE source of truth for the inline physics constants and
 * formulae that are referenced by both the bin-based fluid grain-evolution
 * module (solids/ism_dust_chemistry.cc, GALSF_ISMDUSTCHEM_GRAINSIZEEVO) and
 * the per-superparticle stochastic grain-evolution module
 * (solids/grain_evolution.cc + grain_evolution_functions.h, GRAIN_EVOLUTION,
 * Phase 17b). The two modules are mutually exclusive (different
 * mass-conservation models) but share the underlying microphysics, and we do
 * not want to maintain two copies of polynomial fits, threshold tables, and
 * elastic constants that drift apart.
 *
 * What lives here:
 *   - Nozawa+(2006) thermal-sputtering erosion-rate polynomial fit:
 *         grain_outcomes_sputter_erosion_dadt_per_nH(T, kind)
 *     [returns Y in um/yr cm^3; same form as the inline literals previously
 *      hard-coded in update_dust_sputtering for the GRAINSIZEEVO branch].
 *   - Per-species elastic / breakup parameter table:
 *         grain_outcomes_elastic_props(kind) -> {vshat, P1, gamma, E_young,
 *                                                nu_poisson}
 *     [matches the literal values previously hard-coded in
 *      update_dust_shattering_and_coagulation].
 *   - Dominik & Tielens (1997) coagulation-velocity formula adapted to the
 *     Hirashita & Chen (2023) form (the closed-form expression for v_coag
 *     used in update_dust_shattering_and_coagulation, lines 1618 and 1657).
 *         grain_outcomes_v_coag_dominik(a_i, a_j, gamma, E_young,
 *                                       nu_poisson, bulk_dens)
 *
 * What does NOT live here:
 *   - Anything that mutates particle/cell state. These are pure functions of
 *     their inputs.
 *   - Anything that consults the All.* runtime tables. The caller is
 *     responsible for resolving its own runtime species-index mapping into
 *     the GrainOutcomeSpecies enum below before calling.
 *   - The All.ISMDustChem_*Scaling user knobs. Those are runtime knobs that
 *     stay in the calling code, not microphysics.
 *
 * Bit-identical contract: every literal value here is copied verbatim from
 * the prior inline definitions in solids/ism_dust_chemistry.cc. Refactor of
 * that file to call into this header must not change any answer to the last
 * representable bit.
 *
 * Header-only, KOKKOS_INLINE_FUNCTION-callable, no rdc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GRAIN_COLLISIONAL_OUTCOMES_H
#define GRAIN_COLLISIONAL_OUTCOMES_H

#include <math.h>

/* Stable species enum used as the lookup key for both consumers. ISMDustChem
 * dispatches its runtime spec_indx (== All.ISMDustChem_*_Index) into this
 * enum at the point of call; Phase 17b dispatches its Composition[] index. */
enum GrainOutcomeSpecies {
    GRAIN_OUTCOME_SPECIES_SILICATE = 0,
    GRAIN_OUTCOME_SPECIES_CARBON   = 1,
    GRAIN_OUTCOME_SPECIES_IRON     = 2,
    /* Ice species used by Phase 17b only (bits 5/6); ISMDustChem never
     * dispatches to these. Elastic-props lookup falls back to silicate
     * defaults for ice species (no shattering/coagulation thresholds defined
     * in the literature for cold ice mantles in this regime — the operators
     * that touch ice are condensation/sublimation, not pairwise). */
    GRAIN_OUTCOME_SPECIES_H2O_ICE  = 3,
    GRAIN_OUTCOME_SPECIES_CO_ICE   = 4,
    GRAIN_OUTCOME_SPECIES_CO2_ICE  = 5,
    GRAIN_OUTCOME_SPECIES_DEFAULT  = GRAIN_OUTCOME_SPECIES_SILICATE
};

/* Elastic / breakup parameters. Values copied verbatim from
 * update_dust_shattering_and_coagulation per-species branch. */
struct GrainOutcomeElasticProps {
    double v_shat;       /* shattering threshold (cm/s) */
    double P1;           /* critical pressure (dyn cm^-2) */
    double gamma;        /* surface energy per area (erg cm^-2) */
    double E_young;      /* Young's modulus (dyn cm^-2) */
    double nu_poisson;   /* Poisson's ratio (dimensionless) */
};

KOKKOS_INLINE_FUNCTION
struct GrainOutcomeElasticProps grain_outcomes_elastic_props(int kind)
{
    struct GrainOutcomeElasticProps p;
    /* Defaults match the silicate branch (matches the prior fall-through in
     * update_dust_shattering_and_coagulation). */
    p.v_shat = 2.0E5; p.P1 = 3.0E11; p.gamma = 25.0; p.E_young = 5.4E11; p.nu_poisson = 0.17;
    if(kind == GRAIN_OUTCOME_SPECIES_SILICATE) {
        p.v_shat = 2.7E5; p.P1 = 3.0E11; p.gamma = 25.0; p.E_young = 5.4E11; p.nu_poisson = 0.17;
    } else if(kind == GRAIN_OUTCOME_SPECIES_CARBON) {
        p.v_shat = 1.2E5; p.P1 = 4.0E10; p.gamma = 75.0; p.E_young = 1.0E11; p.nu_poisson = 0.32;
    } else if(kind == GRAIN_OUTCOME_SPECIES_IRON) {
        p.v_shat = 2.2E5; p.P1 = 5.5E10; p.gamma = 3000.0; p.E_young = 2.1E12; p.nu_poisson = 0.27;
    }
    return p;
}

/* Dominik-Tielens / Hirashita-Chen coagulation-velocity formula. Returns
 * v_coag in cm/s. Matches the inline expression in
 * update_dust_shattering_and_coagulation (lines 1618 and 1657 prior to C2):
 *
 *   v_coag = 10 * 2.14 * sqrt((a_i^3 + a_j^3) / (a_i + a_j)^3)
 *               * gamma^(5/6) / (E*^(1/3) * mu^(5/6) * sqrt(rho_bulk))
 *   E* = E_young / (2 * (1 - nu)^2);     mu = a_i * a_j / (a_i + a_j)
 *
 * The All.ISMDustChem_VCoagScaling and any per-call cap against v_shat live
 * at the call site (they are runtime knobs / per-call clamps, not part of
 * the underlying physics formula). */
/* Bit-identical contract: this expression is character-for-character the
 * same as the inline form previously in update_dust_shattering_and_coagulation
 * (lines 1618 and 1657 prior to C2). Do not refactor x*x*x to pow(x,3) or
 * vice versa here without re-validating the FIRE regression. */
KOKKOS_INLINE_FUNCTION
double grain_outcomes_v_coag_dominik(double a_i, double a_j,
                                            double gamma, double E_young,
                                            double nu_poisson, double bulk_dens)
{
    return 10 * 2.14 * sqrt((a_i*a_i*a_i + a_j*a_j*a_j)/pow(a_i+a_j,3))*pow(gamma,5./6.)/(pow((E_young/(2*(1-nu_poisson)*(1-nu_poisson))),1./3.)*pow(a_i*a_j/(a_i + a_j),5./6.)*sqrt(bulk_dens));
}

/* Nozawa+(2006) thermal-sputtering erosion-rate polynomial fits.
 * Returns Y in um/yr cm^3 such that da/dt = -Y * 1e-4 * n_H[cgs] * 1e9
 * gives da/dt in cm/Gyr (this is the conversion the caller applies; the
 * polynomial returns the raw fit value). Coefficients verbatim from the
 * prior inline definitions in update_dust_sputtering. */
/* Per-ice-species microphysics for Phase 17b condensation/sublimation
 * (bits 5/6 of GRAIN_EVOLUTION). Indexed by ice slot:
 *   ICE_INDEX_H2O = 0, ICE_INDEX_CO = 1, ICE_INDEX_CO2 = 2.
 * Refractory vapor (volatile species index 3 in VolatileSpecies[]) is not
 * handled here -- there is no source for it yet in Phase 17b (sputter->
 * vapor coupling is reserved for a later commit). Numerical values from
 * standard cosmochemistry references (Sandford+1988; Fraser+2001;
 * Burke+2010); these match conventions used in protoplanetary-disk
 * chemistry codes. */
enum GrainEvolutionIceIndex {
    GRAIN_EVOLUTION_ICE_H2O = 0,
    GRAIN_EVOLUTION_ICE_CO  = 1,
    GRAIN_EVOLUTION_ICE_CO2 = 2,
    GRAIN_EVOLUTION_NUM_ICE = 3
};

KOKKOS_INLINE_FUNCTION
double grain_outcomes_ice_snowline_T(int ice_index)
{
    /* Equilibrium sublimation/condensation temperature [K] at ~MMSN
     * midplane number densities. */
    if(ice_index == GRAIN_EVOLUTION_ICE_H2O) { return 150.0; }
    if(ice_index == GRAIN_EVOLUTION_ICE_CO)  { return  25.0; }
    if(ice_index == GRAIN_EVOLUTION_ICE_CO2) { return  70.0; }
    return 100.0; /* default */
}
KOKKOS_INLINE_FUNCTION
double grain_outcomes_ice_latent_heat_cgs(int ice_index)
{
    /* Latent heat of sublimation [erg / g of ice]. */
    if(ice_index == GRAIN_EVOLUTION_ICE_H2O) { return 2.83e10; }
    if(ice_index == GRAIN_EVOLUTION_ICE_CO)  { return 2.09e9;  }
    if(ice_index == GRAIN_EVOLUTION_ICE_CO2) { return 5.74e9;  }
    return 1.0e10; /* default */
}
KOKKOS_INLINE_FUNCTION
double grain_outcomes_ice_molecular_weight(int ice_index)
{
    /* Mean molecular weight of the volatile species (in amu). Used to set
     * thermal velocity in the collision-rate prefactor. */
    if(ice_index == GRAIN_EVOLUTION_ICE_H2O) { return 18.0; }
    if(ice_index == GRAIN_EVOLUTION_ICE_CO)  { return 28.0; }
    if(ice_index == GRAIN_EVOLUTION_ICE_CO2) { return 44.0; }
    return 18.0; /* default */
}

/* Bit-identical contract: these polynomials are character-for-character the
 * same as the prior inline carbSput/silSput/ironSput definitions in
 * update_dust_sputtering. Do not refactor pow(logt,k) -> repeated multiply
 * here without re-validating against the original.
 *
 * API note: takes logt = log10(T) rather than T directly. Reason: under
 * -ffast-math the compiler is allowed to reorder the polynomial and FMA
 * differently when log10 is computed inside vs outside the function call,
 * producing LSB-level (~1e-13 relative) drift versus the original inline
 * form. Caller computes log10(T) once and passes it in -- mirrors the
 * original "double logt = log10(temp);" line that preceded the polynomial in
 * update_dust_sputtering. */
KOKKOS_INLINE_FUNCTION
double grain_outcomes_sputter_erosion_dadt_per_nH(double logt, int kind)
{
    if(kind == GRAIN_OUTCOME_SPECIES_CARBON) {
        return pow(10,-226.85 + 133.44*logt - 32.572*pow(logt,2) + 4.0057*pow(logt,3) - 0.24747*pow(logt,4) + 0.0061212*pow(logt,5));
    }
    if(kind == GRAIN_OUTCOME_SPECIES_IRON) {
        return pow(10,-156.88 +  82.110*logt - 18.238*pow(logt,2) + 2.0692*pow(logt,3) - 0.11933*pow(logt,4) + 0.0027788*pow(logt,5));
    }
    /* silicate (and default) */
    return pow(10,-226.95 + 127.94*logt - 29.920*pow(logt,2) + 3.5354*pow(logt,3) - 0.21055*pow(logt,4) + 0.0050362*pow(logt,5));
}

#endif /* GRAIN_COLLISIONAL_OUTCOMES_H */
