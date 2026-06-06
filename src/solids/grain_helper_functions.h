/* grain_helper_functions.h — GPU-callable helpers for GRAIN_COLLISIONS:
 *   - return_grain_cross_section_per_unit_mass_P
 *   - prob_of_grain_interaction_tab
 *
 * Pure KOKKOS_INLINE_FUNCTION forms that take particle_data and the
 * GeoFactorTable as explicit arguments, so the same code path works on
 * CPU tree-walk and the B2 AGSForce GPU kernel. g_geo_tab is pulled in
 * from sidm/sidm_helper_functions.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GRAIN_HELPER_FUNCTIONS_H
#define GRAIN_HELPER_FUNCTIONS_H

#include "../declarations/allvars.h"
#include "../sidm/sidm_helper_functions.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif


#if defined(DM_SIDM) && defined(GRAIN_COLLISIONS)

/* Per-particle cross-section per unit mass. Reads P[i].Grain_Size. */
KOKKOS_INLINE_FUNCTION
double return_grain_cross_section_per_unit_mass_P(int i, const struct particle_data *P_arr)
{
    return All.DM_InteractionCrossSection * 0.75 / (P_arr[i].Grain_Size * All.Grain_Internal_Density);
}


/* Pairwise grain-grain interaction probability (mass-weighted cross section,
   velocity-dependence optional). Mirrors prob_of_grain_interaction()
   exactly but threads P and GeoFactorTable explicitly. */
KOKKOS_INLINE_FUNCTION
double prob_of_grain_interaction_tab(double cx_per_unitmass, double mass,
                                     double r, double h_si,
                                     const Vec3<double> &dV, double dt,
                                     int j_ngb,
                                     const struct particle_data *P_arr,
                                     const MyDouble *GeoFactorTable_arr)
{
    double dVmag = sqrt(dV[0]*dV[0] + dV[1]*dV[1] + dV[2]*dV[2]) / All.cf_atime;
    double rho_eff = 0.5 * (mass + P_arr[j_ngb].Mass) / (h_si*h_si*h_si) * All.cf_a3inv;
    double cx_j = return_grain_cross_section_per_unit_mass_P(j_ngb, P_arr);
    double cx_eff = g_geo_tab(r/h_si, GeoFactorTable_arr)
                    * (mass * cx_per_unitmass + P_arr[j_ngb].Mass * cx_j)
                    / (mass + P_arr[j_ngb].Mass);
    double units = UNIT_SURFDEN_IN_CGS;
    if(All.DM_InteractionVelocityScale > 0) {
        double x = dVmag / All.DM_InteractionVelocityScale;
        cx_eff /= 1 + x*x*x*x;
    }
    return rho_eff * cx_eff * dVmag * dt * units;
}

#endif /* DM_SIDM && GRAIN_COLLISIONS */

#endif /* GRAIN_HELPER_FUNCTIONS_H */
