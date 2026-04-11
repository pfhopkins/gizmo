/* predict_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementation of
 * evaluate_NH_from_GradRho.  Single source of truth for both CPU and GPU.
 *
 * Proto.h has an inline Vec3<MyFloat> wrapper that forwards to this.
 *
 * Include order: after allvars.h (for All, MyFloat). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double evaluate_NH_from_GradRho(MyFloat gradrho[3], double rkern, double rho, double numngb_ndim, double include_h, int target, struct particle_data *pp)
{
    double gradrho_mag=0;
    if(rho>0)
    {
#ifdef RT_USE_TREECOL_FOR_NH
        gradrho_mag = include_h * rho * rkern / numngb_ndim; if(target>=0) {gradrho_mag += pp[target].SigmaEff;}
#else
        gradrho_mag = sqrt(gradrho[0]*gradrho[0]+gradrho[1]*gradrho[1]+gradrho[2]*gradrho[2]);
        if(gradrho_mag > 0) {gradrho_mag = rho*rho/gradrho_mag;} else {gradrho_mag=0;}
        if(include_h > 0) if(numngb_ndim > 0) gradrho_mag += include_h * rho * rkern / numngb_ndim;
#endif
    }
    return gradrho_mag * All.cf_a2inv;
}
