/* predict_device.h — KOKKOS_INLINE_FUNCTION version of evaluate_NH_from_GradRho
 * for use in the GPU cooling kernel.  The original lives in core/predict.cc
 * which is not compiled by nvcc_wrapper (not in GPU_OBJS).
 *
 * Signature uses Vec3<MyFloat> to match the call sites (particle_data::GradRho
 * is Vec3<MyFloat>).  The original in predict.cc takes MyFloat gradrho[3] which
 * is ABI-compatible but nvcc doesn't match Vec3<MyFloat> to MyFloat[3].
 *
 * Include order: after allvars.h (for All, MyFloat, Vec3). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double evaluate_NH_from_GradRho(const Vec3<MyFloat>& gradrho, double rkern, double rho, double numngb_ndim, double include_h, int target, struct particle_data *pp)
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
