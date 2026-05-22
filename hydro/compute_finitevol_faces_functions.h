#pragma once

/* calculate_face_area_for_cartesian_mesh is KOKKOS_INLINE_FUNCTION-defined
 * here (Phase D 2026-05-21 #20011-D fix). Include early so all consumers see
 * the device-callable definition before the compute_finitevol_faces template. */
#include "../core/predict_functions.h"

/* compute_finitevol_faces_functions.h -- MFM/MFV face-area construction as a
 * proper function template. Replaces the #include-fragment
 * hydro/compute_finitevol_faces.h.
 *
 * Templated on the caller's local struct and kernel struct types so the five
 * call sites (hydro_core_meshless.h, gradients.cc, gradient_functions.h,
 * mg_gradient_correction.cc, transport_flux_evaluate.h) can pass their own
 * types without wrappers. LocalT must have NV_T; KERNEL_CRK_FACES paths
 * additionally require Tensor_CRK_Face_Corrections on both LocalT and gas_cell_data
 * (same pre-existing constraint as the old fragment).
 *
 * Outputs:
 *   Face_Area_Vec, Face_Area_Norm      (the face geometry)
 *   Vi_inv_corr, Vj_inv_corr           (=1 unless HYDRO_FACE_VOLUME_RECONSTRUCTION_CORRECTION
 *                                       trims them; callers that don't use them
 *                                       can pass a throwaway local)
 *
 * rinv is consumed only in the HYDRO_REGULAR_GRID branch; callers may safely
 * pass 0 when that flag is off.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h included.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef COMPUTE_FINITEVOL_FACES_FUNCTIONS_H
#define COMPUTE_FINITEVOL_FACES_FUNCTIONS_H

template <typename LocalT, typename KernelT>
KOKKOS_INLINE_FUNCTION
void compute_finitevol_faces(
    const LocalT &local,
    const struct gas_cell_data &CPj,
    const KernelT &kernel,
    double rinv,
    double r2,
    double V_i, double V_j,
    double Particle_Size_i, double Particle_Size_j,
    double cnumcrit2,
    Vec3<double> &Face_Area_Vec,
    double &Face_Area_Norm,
    double &Vi_inv_corr,
    double &Vj_inv_corr)
{
    Face_Area_Norm = 0;
    Vi_inv_corr = 1.0;
    Vj_inv_corr = 1.0;

#ifdef HYDRO_REGULAR_GRID
    Face_Area_Norm = calculate_face_area_for_cartesian_mesh(kernel.dp, rinv, Particle_Size_i, Face_Area_Vec);
#endif

#if (defined(HYDRO_MESHLESS_FINITE_MASS) || defined(HYDRO_MESHLESS_FINITE_VOLUME)) && !defined(HYDRO_REGULAR_GRID)
    double wt_i = V_i, wt_j = V_j;
    /* centered face weight when V_i/V_j differ strongly — see Lanson & Vila */
    if((fabs(V_i - V_j) / DMIN(V_i, V_j)) / NUMDIMS > 1.25) {
        wt_i = wt_j = V_i*V_j*(kernel.wk_i + kernel.wk_j) / (V_i*kernel.wk_i + V_j*kernel.wk_j);
    }
    Face_Area_Vec = (kernel.wk_i * wt_i * local.NV_T.matvec(kernel.dp)
                    + kernel.wk_j * wt_j * CPj.NV_T.matvec(kernel.dp))
                   * (All.cf_atime * All.cf_atime);
    Face_Area_Norm = Face_Area_Vec.norm_sq();
    double facenormal_dot_dp = dot(Face_Area_Vec, kernel.dp);

#if defined(KERNEL_CRK_FACES)
    {
        /* Tensor_CRK_Face_Corrections layout: A, B[3], (dA+A*B)[3], (dA.B+A.dB)[3][3] */
        double wk_ij  = 0.5 * (kernel.wk_i + kernel.wk_j);
        double dwk_ij = 0.5 * (kernel.dwk_i + kernel.dwk_j) / (MIN_REAL_NUMBER + kernel.r);
        double Bi_dot_dx = 0, Bj_dot_dx = 0;
        double dAi_etc_dot_dx[3] = {0}, dAj_etc_dot_dx[3] = {0};
        for(int k=0; k<3; k++) {
            Bi_dot_dx += local.Tensor_CRK_Face_Corrections[k+1] * kernel.dp[k];
            Bj_dot_dx -= CPj.Tensor_CRK_Face_Corrections[k+1] * kernel.dp[k];
            for(int k_x=0; k_x<3; k_x++) {
                dAi_etc_dot_dx[k_x] += local.Tensor_CRK_Face_Corrections[7+3*k+k_x] * kernel.dp[k];
                dAj_etc_dot_dx[k_x] -= CPj.Tensor_CRK_Face_Corrections[7+3*k+k_x] * kernel.dp[k];
            }
        }
        Face_Area_Norm = 0;
        facenormal_dot_dp = 0;
        for(int k=0; k<3; k++) {
            double Ai = -V_i*V_j * (  local.Tensor_CRK_Face_Corrections[0] * (1. + Bi_dot_dx) * dwk_ij * (+kernel.dp[k])
                                    + (local.Tensor_CRK_Face_Corrections[4+k] + dAi_etc_dot_dx[k]) * wk_ij);
            double Aj = -V_i*V_j * (  CPj.Tensor_CRK_Face_Corrections[0] * (1. + Bj_dot_dx) * dwk_ij * (-kernel.dp[k])
                                    + (CPj.Tensor_CRK_Face_Corrections[4+k] + dAj_etc_dot_dx[k]) * wk_ij);
            Face_Area_Vec[k] = (Ai - Aj) * All.cf_atime * All.cf_atime;
            Face_Area_Norm += Face_Area_Vec[k] * Face_Area_Vec[k];
            facenormal_dot_dp += Face_Area_Vec[k] * kernel.dp[k];
        }
    }
#endif

    /* ill-conditioned gradient matrix: fall back to RSPH EOM for stability */
    if((CPj.ConditionNumber * CPj.ConditionNumber > 1.0e12 + cnumcrit2) || (facenormal_dot_dp < 0)) {
        Face_Area_Norm = -(wt_i*V_i*kernel.dwk_i + wt_j*V_j*kernel.dwk_j) / kernel.r;
        Face_Area_Norm *= All.cf_atime * All.cf_atime;
        Face_Area_Vec  = kernel.dp * Face_Area_Norm;
        Face_Area_Norm = Face_Area_Norm * Face_Area_Norm * r2;
    }

    Face_Area_Norm = sqrt(Face_Area_Norm);

#if defined(HYDRO_FACE_VOLUME_RECONSTRUCTION_CORRECTION)
    {
        double Vi_phys = V_i / All.cf_a3inv;
        double Vj_phys = V_j / All.cf_a3inv;
        double Vol_min = fabs(0.5 * facenormal_dot_dp) * All.cf_atime / 3.;
        if(Vol_min > DMIN(Vi_phys, Vj_phys)) {
            if(Vol_min > Vi_phys) { Vi_inv_corr = Vi_phys / Vol_min; }
            if(Vol_min > Vj_phys) { Vj_inv_corr = Vj_phys / Vol_min; }
        }
    }
#endif

#if (SLOPE_LIMITER_TOLERANCE == 0)
    {
        double Amax = DMIN(Get_Particle_Expected_Area(Particle_Size_i),
                           Get_Particle_Expected_Area(Particle_Size_j));
        if(Face_Area_Norm > Amax) {
            Face_Area_Vec *= (Amax / Face_Area_Norm);
            Face_Area_Norm = Amax;
        }
    }
#endif
#endif /* HYDRO_MESHLESS_FINITE_MASS || HYDRO_MESHLESS_FINITE_VOLUME */
}

#endif /* COMPUTE_FINITEVOL_FACES_FUNCTIONS_H */
