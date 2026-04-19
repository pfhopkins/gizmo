/* elastic_stress_tensor_force_functions.h -- per-pair elastic-stress force for
 * solids. Replaces the fragment solids/elastic_stress_tensor_force.h.
 *
 * Body guarded by EOS_ELASTIC. Pure i-accumulation into Fluxes.v / Fluxes.p.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h, and
 * system/eigen_symmetric.h (for the GPU-callable eigen_symmetric() routine).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef ELASTIC_STRESS_TENSOR_FORCE_FUNCTIONS_H
#define ELASTIC_STRESS_TENSOR_FORCE_FUNCTIONS_H

#include "../hydro/hydro_pair_types.h"
#include "../system/eigen_symmetric.h"

KOKKOS_INLINE_FUNCTION
void elastic_stress_tensor_force_compute_pair(
    const struct hydro_data_in &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
    const Vec3<MyDouble> &VelPred_j,
    const struct kernel_hydra &kernel,
    double rinv,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double tensile_correction_factor,
    double dt_hydrostep,
    struct Conserved_var_Riemann &Fluxes)
{
#ifdef EOS_ELASTIC
    (void)rinv; (void)dt_hydrostep;  /* only consumed by commented-out limiter in original */

    if(!((local.Mass>0) && (Pj.Mass>0))) { return; }

    double FNormT = Face_Area_Norm;
    double FVec[3] = {Face_Area_Vec[0], Face_Area_Vec[1], Face_Area_Vec[2]};

    double cmag[3] = {0}, v_interface[3];
    double wt_i = -0.5, wt_j = -0.5;
    for(int k=0; k<3; k++) { v_interface[k] = (wt_i*local.Vel[k] + wt_j*VelPred_j[k]) / All.cf_atime; }
    double rho_i = local.Density * All.cf_a3inv;
    double rho_j = CPj.Density * All.cf_a3inv;

    double wt_r = rho_i*local.SoundSpeed * rho_j*CPj.SoundSpeed
                / (rho_i*local.SoundSpeed + rho_j*CPj.SoundSpeed) * FNormT;
    double cT_i = sqrt(All.Tillotson_EOS_params[local.CompositionType][10] / rho_i);
    double cT_j = sqrt(All.Tillotson_EOS_params[CPj.CompositionType][10] / rho_j);
    double wt_t  = rho_i*cT_i * rho_j*cT_j / (rho_i*cT_i + rho_j*cT_j) * FNormT;
    double wt_rt = (wt_r - wt_t) * kernel.vdotr2 * rinv*rinv * All.cf_atime*All.cf_atime;

    for(int ij_switch=0; ij_switch<2; ij_switch++) {
        double nvt[NUMDIMS*NUMDIMS] = {0}, norm_m = 0, wtfac = 0;
        if(ij_switch == 0) { wtfac = wt_i; } else { wtfac = wt_j; }
        for(int k_v=0; k_v<NUMDIMS; k_v++) { for(int j_v=0; j_v<NUMDIMS; j_v++) {
            if(ij_switch == 0) { nvt[NUMDIMS*k_v + j_v] = local.Elastic_Stress_Tensor[k_v][j_v]; }
            else               { nvt[NUMDIMS*k_v + j_v] = CPj.Elastic_Stress_Tensor_Pred[k_v][j_v]; }
            norm_m += nvt[NUMDIMS*k_v + j_v] * nvt[NUMDIMS*k_v + j_v];
        }}
        if(norm_m > MIN_REAL_NUMBER) {
            double eigvals_loc[3] = {0}, eigvecs_loc[9] = {0};
            eigen_symmetric(nvt, eigvals_loc, eigvecs_loc, NUMDIMS);
            for(int k_v=0; k_v<NUMDIMS; k_v++) {
                double eigenval_k = eigvals_loc[k_v], eigenvec_k[3] = {0}, A_dot_v = 0;
                for(int j_v=0; j_v<NUMDIMS; j_v++) {
                    eigenvec_k[j_v] = eigvecs_loc[NUMDIMS*k_v + j_v];
                    A_dot_v += FVec[j_v] * eigenvec_k[j_v];
                }
                double prefac = wtfac * eigenval_k * (A_dot_v * All.cf_a2inv);
#if !defined(HYDRO_MESHLESS_FINITE_VOLUME) && !defined(HYDRO_MESHLESS_FINITE_MASS)
                if(eigenval_k > 0) { prefac *= 1. - tensile_correction_factor; }
#endif
                if(!isnan(eigenval_k)) {
                    for(int j_v=0; j_v<NUMDIMS; j_v++) { cmag[j_v] += prefac * eigenvec_k[j_v]; }
                }
            }
        }
    }
    for(int j_v=0; j_v<3; j_v++) {
        cmag[j_v] -= wt_rt * kernel.dp[j_v] * All.cf_atime + wt_t * kernel.dv[j_v] / All.cf_atime;
    }

    /* zero-energy-mode (hourglass) control: Ganzenmuller 2015 / Kugelstadt 2021 */
    {
        double cT_hg = 0.5 * wt_t / All.cf_atime;
        for(int j_v=0; j_v<3; j_v++) {
            double dv_pred_i = 0, dv_pred_j = 0;
            for(int k_v=0; k_v<3; k_v++) {
                dv_pred_i += local.Gradients.Velocity[j_v][k_v] * kernel.dp[k_v];
                dv_pred_j += CPj.Gradients.Velocity[j_v][k_v] * kernel.dp[k_v];
            }
            double dv_err = kernel.dv[j_v] - 0.5 * (dv_pred_i + dv_pred_j);
            cmag[j_v] -= cT_hg * dv_err;
        }
    }

    double cmag_E = cmag[0]*v_interface[0] + cmag[1]*v_interface[1] + cmag[2]*v_interface[2];

    for(int k=0; k<3; k++) { Fluxes.v[k] += cmag[k]; }
    Fluxes.p += cmag_E;
#endif /* EOS_ELASTIC */
}

#endif /* ELASTIC_STRESS_TENSOR_FORCE_FUNCTIONS_H */
