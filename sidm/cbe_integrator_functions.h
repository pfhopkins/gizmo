/* cbe_integrator_functions.h — GPU-callable CBE (Collisionless Boltzmann
 * Equation) single-sided flux computation. Pure math with no global state.
 *
 * Mirrors do_cbe_flux_computation() in sidm/cbe_integrator.cc exactly, but
 * as a KOKKOS_INLINE_FUNCTION so it can be called both from the CPU
 * tree-walk (via cbe_integrator_flux_functions.h) and from the B2 AGSForce
 * GPU kernel.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef CBE_INTEGRATOR_FUNCTIONS_H
#define CBE_INTEGRATOR_FUNCTIONS_H

#include "../declarations/allvars.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif


#ifdef CBE_INTEGRATOR
KOKKOS_INLINE_FUNCTION
double do_cbe_flux_computation(double moments[CBE_INTEGRATOR_NMOMENTS],
                               double vface_dot_A,
                               double vface[3],
                               double Area[3],
                               double moments_ngb[CBE_INTEGRATOR_NMOMENTS],
                               double fluxes[CBE_INTEGRATOR_NMOMENTS])
{
    double m_inv = 1. / moments[0];
    double v[3], f00_vsig = 1;
    v[0] = m_inv*moments[1]; v[1] = m_inv*moments[2]; v[2] = m_inv*moments[3];
    double vsig = v[0]*Area[0] + v[1]*Area[1] + v[2]*Area[2] - vface_dot_A;
    if(fabs(vsig) <= 0) return 0;
    fluxes[0] = vsig * moments[0];
    for(int k = 1; k < CBE_INTEGRATOR_NMOMENTS; k++) { fluxes[k] = (m_inv * moments[k]) * fluxes[0]; }

#if (CBE_INTEGRATOR_NMOMENTS > 4)
    {
        double dv2 = (v[0]-vface[0])*(v[0]-vface[0]) + (v[1]-vface[1])*(v[1]-vface[1]) + (v[2]-vface[2])*(v[2]-vface[2]);
        double c_eff_over_vsig_A = sqrt(m_inv * (moments[4]+moments[5]+moments[6]) / dv2);
        double SM_vsig = 1 + c_eff_over_vsig_A*c_eff_over_vsig_A/(1 + c_eff_over_vsig_A);
        double f00_SdotA = 1 - c_eff_over_vsig_A / (SM_vsig + c_eff_over_vsig_A);
        f00_vsig = (SM_vsig * (1 + c_eff_over_vsig_A)) / (SM_vsig + c_eff_over_vsig_A);

        double S_dot_A[3];
        S_dot_A[0] = f00_SdotA * (moments[4]*Area[0] + moments[7]*Area[1] + moments[8]*Area[2]);
        S_dot_A[1] = f00_SdotA * (moments[7]*Area[0] + moments[5]*Area[1] + moments[9]*Area[2]);
        S_dot_A[2] = f00_SdotA * (moments[8]*Area[0] + moments[9]*Area[1] + moments[6]*Area[2]);
        fluxes[1] += S_dot_A[0];
        fluxes[2] += S_dot_A[1];
        fluxes[3] += S_dot_A[2];
        fluxes[4] += 2.*v[0]*S_dot_A[0] + fluxes[0]*v[0]*v[0];
        fluxes[5] += 2.*v[1]*S_dot_A[1] + fluxes[0]*v[1]*v[1];
        fluxes[6] += 2.*v[2]*S_dot_A[2] + fluxes[0]*v[2]*v[2];
        fluxes[7] += v[0]*S_dot_A[1] + v[1]*S_dot_A[0] + fluxes[0]*v[0]*v[1];
        fluxes[8] += v[0]*S_dot_A[2] + v[2]*S_dot_A[0] + fluxes[0]*v[0]*v[2];
        fluxes[9] += v[1]*S_dot_A[2] + v[2]*S_dot_A[1] + fluxes[0]*v[1]*v[2];
    }
#else
    (void)vface; (void)moments_ngb;
#endif
    return vsig * f00_vsig;
}
#endif /* CBE_INTEGRATOR */

#endif /* CBE_INTEGRATOR_FUNCTIONS_H */
