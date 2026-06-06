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

/* calculate_face_area_for_cartesian_mesh — migrated from predict.cc to fix
 * #20011-D (host-only fn called from KOKKOS_INLINE_FUNCTION
 * compute_finitevol_faces template under HYDRO_REGULAR_GRID). Body uses only
 * All.cf_atime (mirror-safe), std::max, fabs — all device-callable.
 * Phase D 2026-05-21. */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
KOKKOS_INLINE_FUNCTION
double calculate_face_area_for_cartesian_mesh(const Vec3<double>& dp, double rinv, double l_side, Vec3<double>& Face_Area_Vec)
{
    Face_Area_Vec = {}; double Face_Area_Norm;
#if (NUMDIMS==1)
    Face_Area_Norm = 1; Face_Area_Vec[0] = Face_Area_Norm * dp[0]/fabs(dp[0]);
#elif (NUMDIMS==2)
    if(fabs(dp[0]) > fabs(dp[1])) {Face_Area_Vec[0] = Face_Area_Norm = DMAX(0.,l_side-fabs(dp[1])) * dp[0]/fabs(dp[0]) * All.cf_atime;} else {Face_Area_Vec[1] = Face_Area_Norm = DMAX(0.,l_side-fabs(dp[0])) * dp[1]/fabs(dp[1]) * All.cf_atime;}
#else
    Vec3<double> dp_abs = {fabs(dp[0]), fabs(dp[1]), fabs(dp[2])};
    int kdir;
    if((dp_abs[0]>=dp_abs[1])&&(dp_abs[0]>=dp_abs[2])) {kdir=0;} else if ((dp_abs[1]>=dp_abs[0])&&(dp_abs[1]>=dp_abs[2])) {kdir=1;} else {kdir=2;}
    Face_Area_Norm=1; for(int k=0;k<3;k++) {if(k!=kdir) {Face_Area_Norm *= DMAX(0.,l_side-dp_abs[k]) * All.cf_atime*All.cf_atime;}}
    Face_Area_Vec[kdir] = Face_Area_Norm * dp[kdir]/fabs(dp[kdir]);
#endif
    return fabs(Face_Area_Norm);
}
#endif

/* Get_Particle_Expected_Area — migrated from predict.cc to fix #20011-D
 * (host-only fn called from KOKKOS_INLINE_FUNCTION compute_finitevol_faces
 * under SLOPE_LIMITER_TOLERANCE==0 — surfaced 2026-05-21 Phase D config 152).
 * Pure-compute function of `h`, dimension-dependent. */
KOKKOS_INLINE_FUNCTION
double Get_Particle_Expected_Area(double h)
{
#if (NUMDIMS == 1)
    return 2;
#endif
#if (NUMDIMS == 2)
    return 2 * M_PI * h;
#endif
#if (NUMDIMS == 3)
    return 4 * M_PI * h * h;
#endif
}
