/* nonideal_mhd_functions.h -- per-pair non-ideal MHD (ohmic / Hall / ambipolar
 * diffusion) flux. Replaces the fragment hydro/nonideal_mhd.h.
 *
 * Body guarded by MHD_NON_IDEAL. Pure i-accumulation into Fluxes.B.
 *
 * Note (from original): not updated for cosmological units. Intended for
 * non-cosmological simulations.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef NONIDEAL_MHD_FUNCTIONS_H
#define NONIDEAL_MHD_FUNCTIONS_H

#include "hydro_pair_types.h"

KOKKOS_INLINE_FUNCTION
void nonideal_mhd_compute_pair(
    const struct hydro_data_in &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
    const Vec3<double> &BPred_j,
    const struct kernel_hydra &kernel,
    double rinv,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double v_hll,
    const Vec3<double> &bhat,
    double bhat_mag,
    double dt_hydrostep,
    struct Conserved_var_Riemann &Fluxes,
    Vec3<double> &bflux_from_nonideal_effects)
{
#ifdef MHD_NON_IDEAL
    (void)bhat_mag;  /* bhat_mag unused here — bhat already carries magnitude */
    if(!((local.Mass > 0) && (Pj.Mass > 0) && (dt_hydrostep > 0))) { return; }

    double eta_i, eta_j, eta_ohmic, eta_hall, eta_ad;
    eta_i = local.Eta_MHD_OhmicResistivity_Coeff; eta_j = CPj.Eta_MHD_OhmicResistivity_Coeff;
    if(eta_i*eta_j > 0) { eta_ohmic = 2*eta_i*eta_j / (eta_i+eta_j); } else { eta_ohmic = 0; }
    eta_i = local.Eta_MHD_HallEffect_Coeff; eta_j = CPj.Eta_MHD_HallEffect_Coeff;
    if(eta_i*eta_j > 0) { eta_hall = 2*eta_i*eta_j / (eta_i+eta_j); } else { eta_hall = 0; }
    eta_i = local.Eta_MHD_AmbiPolarDiffusion_Coeff; eta_j = CPj.Eta_MHD_AmbiPolarDiffusion_Coeff;
    if(eta_i*eta_j > 0) { eta_ad = 2*eta_i*eta_j / (eta_i+eta_j); } else { eta_ad = 0; }

    double eta_max = DMAX(fabs(eta_ohmic), DMAX(fabs(eta_hall), fabs(eta_ad)));
    if(eta_max <= MIN_REAL_NUMBER) { return; }

    Vec3<double> J_current, d_scalar;
    double rinv2 = rinv * rinv;
    Vec3<double> J_direct = {}, grad_dot_x_ij = {};
    for(int k=0; k<3; k++) { d_scalar[k] = local.BPred[k] - BPred_j[k]; }
    for(int k=0; k<3; k++) {
        int k_xyz_A = 0, k_xyz_B = 0;
        if(k == 0) { k_xyz_A = 2; k_xyz_B = 1; }
        if(k == 1) { k_xyz_A = 0; k_xyz_B = 2; }
        if(k == 2) { k_xyz_A = 1; k_xyz_B = 0; }
        double tmp_grad_A = 0.5 * (local.Gradients.B[k_xyz_A][k_xyz_B] + CPj.Gradients.B[k_xyz_A][k_xyz_B]);
        double tmp_grad_B = 0.5 * (local.Gradients.B[k_xyz_B][k_xyz_A] + CPj.Gradients.B[k_xyz_B][k_xyz_A]);
        J_current[k] = tmp_grad_B - tmp_grad_A;
        J_direct[k]  = rinv2 * (kernel.dp[k_xyz_A]*d_scalar[k_xyz_B] - kernel.dp[k_xyz_B]*d_scalar[k_xyz_A]);
        if(J_current[k]*J_direct[k] < 0) { if(fabs(J_direct[k]) > 5.*fabs(J_current[k])) { J_current[k] = 0.0; } }
        for(int k2=0; k2<3; k2++) {
            grad_dot_x_ij[k] += 0.5*(local.Gradients.B[k][k2] + CPj.Gradients.B[k][k2]) * kernel.dp[k2];
        }
    }
    double Jmag = J_current.norm_sq();

    Vec3<double> b_flux = {};
    Vec3<double> JcrossB = cross(J_current, bhat);
    Vec3<double> JcrossBcrossB = cross(JcrossB, bhat);
    if(fabs(eta_ohmic) > 0) { b_flux += -eta_ohmic * J_current; }
    if(fabs(eta_ad)    > 0) { b_flux +=  eta_ad    * JcrossBcrossB; }
    if(fabs(eta_hall)  > 0) { b_flux += -eta_hall  * JcrossB; }

    bflux_from_nonideal_effects = cross(Face_Area_Vec, b_flux);

    double eta_0 = v_hll * kernel.r * All.cf_atime;
    double eta_ohmic_0 = 0, eta_ad_0 = 0, eta_hall_0 = 0, q = 0;
    if(fabs(eta_ohmic) > 0) { q = eta_0/fabs(eta_ohmic); eta_ohmic_0 = eta_0*(0.2 + q)/(0.2 + q + q*q); }
    if(fabs(eta_ad)    > 0) { q = eta_0/fabs(eta_ad);    eta_ad_0    = eta_0*(0.2 + q)/(0.2 + q + q*q); }
    if(fabs(eta_hall)  > 0) { q = eta_0/fabs(eta_hall);  eta_hall_0  = eta_0*(0.2 + q)/(0.2 + q + q*q); if(eta_hall < 0) { eta_hall_0 *= -1; } }
    Vec3<double> b_flux_direct = {}, db_direct;
    Vec3<double> JcrossB_direct      = cross(J_direct, bhat);
    Vec3<double> JcrossBcrossB_direct = cross(JcrossB_direct, bhat);
    if(fabs(eta_ohmic_0) > 0) { b_flux_direct += -eta_ohmic_0 * J_direct; }
    if(fabs(eta_ad_0)    > 0) { b_flux_direct +=  eta_ad_0    * JcrossBcrossB_direct; }
    if(fabs(eta_hall_0)  > 0) { b_flux_direct += -eta_hall_0  * JcrossB_direct; }
    db_direct = cross(Face_Area_Vec, b_flux_direct);

    double db_dot_direct_diff = 0;
    double F_ddiff_prefac = -Face_Area_Norm * rinv * DMAX(eta_ohmic, fabs(eta_hall) + eta_ad);
    Vec3<double> db_direct_diff = F_ddiff_prefac * d_scalar;

    double bfluxmag = bflux_from_nonideal_effects.norm_sq();
    bfluxmag /= (1.e-37 + Jmag * eta_max*eta_max * Face_Area_Norm*Face_Area_Norm);

    for(int k=0; k<3; k++) {
        double d_scalar_tmp = d_scalar[k] - grad_dot_x_ij[k];
        double d_scalar_hll = MINMOD(d_scalar[k], d_scalar_tmp);
        double hll_corr = bfluxmag * hll_correction_fn(d_scalar_hll, 0., 1., eta_max,
                                                       v_hll, Face_Area_Norm, kernel.r, All.cf_atime);
        double db_corr = bflux_from_nonideal_effects[k] + hll_corr;
        bflux_from_nonideal_effects[k] = MINMOD(1.1 * bflux_from_nonideal_effects[k], db_corr);
        if(hll_corr != 0) {
            double db_direct_tmp = fabs(db_direct[k]) * hll_corr / fabs(hll_corr);
            if((bflux_from_nonideal_effects[k] * db_direct_tmp < 0)
               && (fabs(db_direct_tmp) > 1.0*fabs(bflux_from_nonideal_effects[k]))) { bflux_from_nonideal_effects[k] = 0; }
        }
        if((bflux_from_nonideal_effects[k] * db_direct[k] < 0)
           && (fabs(db_direct[k]) > 10.0 * fabs(bflux_from_nonideal_effects[k]))) { bflux_from_nonideal_effects[k] = 0; }

        db_dot_direct_diff += bflux_from_nonideal_effects[k] * db_direct_diff[k];
    }

    if(db_dot_direct_diff < 0) {
        double db_direct_diff_mag2 = db_direct_diff.norm_sq();
        bflux_from_nonideal_effects -= (db_dot_direct_diff / db_direct_diff_mag2) * db_direct_diff;
    }

    Fluxes.B += bflux_from_nonideal_effects;
#endif /* MHD_NON_IDEAL */
}

#endif /* NONIDEAL_MHD_FUNCTIONS_H */
