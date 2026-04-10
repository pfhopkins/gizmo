/* predict_device.h — KOKKOS_INLINE_FUNCTION version of evaluate_NH_from_GradRho
 * for use in the GPU cooling kernel.  The original lives in core/predict.cc
 * which is not compiled by nvcc_wrapper (not in GPU_OBJS).
 *
 * This defines the MyFloat[3] array version.  Proto.h has an inline Vec3<MyFloat>
 * wrapper that forwards to this, so both call patterns work.
 *
 * Include order: after allvars.h (for All, MyFloat). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double evaluate_NH_from_GradRho_impl(const double* gradrho, double rkern, double rho, double numngb_ndim, double include_h, int target, struct particle_data *pp)
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

/* Array version — matches proto.h line 530 declaration */
KOKKOS_INLINE_FUNCTION
double evaluate_NH_from_GradRho(MyFloat gradrho[3], double rkern, double rho, double numngb_ndim, double include_h, int target, struct particle_data *pp)
{
    double grd[3] = {(double)gradrho[0], (double)gradrho[1], (double)gradrho[2]};
    return evaluate_NH_from_GradRho_impl(grd, rkern, rho, numngb_ndim, include_h, target, pp);
}

/* Vec3 overload — matches proto.h line 531 wrapper, but device-callable */
KOKKOS_INLINE_FUNCTION
double evaluate_NH_from_GradRho(const Vec3<MyFloat>& gradrho, double rkern, double rho, double numngb_ndim, double include_h, int target, struct particle_data *pp)
{
    double grd[3] = {(double)gradrho[0], (double)gradrho[1], (double)gradrho[2]};
    return evaluate_NH_from_GradRho_impl(grd, rkern, rho, numngb_ndim, include_h, target, pp);
}
