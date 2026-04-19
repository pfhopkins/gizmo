/* viscosity_functions.h -- per-pair anisotropic/Navier-Stokes viscosity flux.
 *
 * Replaces the #include-fragment hydro/viscosity.h. Body guarded by VISCOSITY.
 * Pure i-accumulation into Fluxes (no j-side writes).
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef VISCOSITY_FUNCTIONS_H
#define VISCOSITY_FUNCTIONS_H

#include "hydro_pair_types.h"

KOKKOS_INLINE_FUNCTION
void viscosity_compute_pair(
    const struct hydro_data_in &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
    const Vec3<MyDouble> &VelPred_j,
    const struct kernel_hydra &kernel,
    double rinv,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double v_hll,
    const Vec3<double> &bhat,
    double bhat_mag,
    double dt_hydrostep,
    struct Conserved_var_Riemann &Fluxes)
{
#ifdef VISCOSITY
    if(!( (((local.Eta_ShearViscosity>MIN_REAL_NUMBER) && (CPj.Eta_ShearViscosity>MIN_REAL_NUMBER)) ||
           ((local.Zeta_BulkViscosity>MIN_REAL_NUMBER) && (CPj.Zeta_BulkViscosity>MIN_REAL_NUMBER))) &&
          (local.Mass>0) && (Pj.Mass>0) )) { return; }

    double a_inv = 1.0 / All.cf_atime;
    Vec3<double> v_interface;
    Vec3<double> cmag;
    double wt_i = 0.5, wt_j = 0.5;
    v_interface = (wt_i*local.Vel + wt_j*VelPred_j) * a_inv;

    double eta = 0.5 * (local.Eta_ShearViscosity + CPj.Eta_ShearViscosity);
    if(eta > 0) { eta = local.Eta_ShearViscosity * CPj.Eta_ShearViscosity / eta; } else { eta = 0; }
    double zeta = 0.5 * (local.Zeta_BulkViscosity + CPj.Zeta_BulkViscosity);
    if(zeta > 0) { zeta = local.Zeta_BulkViscosity * CPj.Zeta_BulkViscosity / zeta; } else { zeta = 0; }
    double viscous_wt_physical = DMAX(eta, zeta);
    wt_i *= -1.0; wt_j *= -1.0;
    int do_non_mhd = 1;

#ifdef MAGNETIC
    Vec3<double> B_interface = {bhat[0]*bhat_mag, bhat[1]*bhat_mag, bhat[2]*bhat_mag};
    double B_interface_mag2 = B_interface.norm_sq();
    double bhat_dot_gradvhat = 1, bhat_dot_gradvhat_direct = 1, grad_v_mag = 0, Bi_proj = 1;
    if(B_interface_mag2 >= 0)
    {
        do_non_mhd = 0;
        double rhs = 0, rhs_direct = 0;
        double one_third = (1.0/3.0) * B_interface_mag2;
        Bi_proj = dot(kernel.dp, B_interface) / B_interface_mag2;
        for(int k_v=0; k_v<3; k_v++)
        {
            double tmp;
            double tmp_kv = B_interface[k_v]*B_interface[k_v] - one_third;
            double b_tensor_dot_face = 0;
            for(int k=0; k<3; k++)
            {
                double grad_v = wt_i*local.Gradients.Velocity[k_v][k] + wt_j*CPj.Gradients.Velocity[k_v][k];
                double grad_direct = -kernel.dv[k_v] * kernel.dp[k] * rinv*rinv;
                grad_v = MINMOD_G(grad_v, grad_direct);
                if(grad_v*grad_direct < 0) { if(fabs(grad_direct) > 2.*fabs(grad_v)) { grad_v = 0.0; } }

                if(k == k_v) { tmp = tmp_kv; } else { tmp = B_interface[k_v]*B_interface[k]; }
                rhs += tmp * grad_v;
                rhs_direct += tmp * kernel.dv[k] * kernel.dp[k_v];
                grad_v_mag += grad_v*grad_v;
                b_tensor_dot_face += tmp * Face_Area_Vec[k];
            }
            cmag[k_v] = b_tensor_dot_face / B_interface_mag2;
        }
        rhs /= B_interface_mag2;
        grad_v_mag = sqrt(grad_v_mag);
        bhat_dot_gradvhat = rhs / grad_v_mag;
        bhat_dot_gradvhat_direct = 3. * rinv * rinv * (rhs_direct / B_interface_mag2);
        cmag *= 3. * eta * rhs;
    }
#endif

    Vec3<double> cmag_dir;
    if(do_non_mhd == 1)
    {
        double divv_i = 0, divv_j = 0;
        for(int k=0; k<3; k++) {
            divv_i += local.Gradients.Velocity[k][k];
            divv_j += CPj.Gradients.Velocity[k][k];
        }
        double divv = (wt_i*divv_i + wt_j*divv_j) * (zeta - eta*(2./3.));
        wt_i *= eta; wt_j *= eta;

        double Pxx = 2*(wt_i*local.Gradients.Velocity[0][0] + wt_j*CPj.Gradients.Velocity[0][0]) + divv;
        double Pyy = 2*(wt_i*local.Gradients.Velocity[1][1] + wt_j*CPj.Gradients.Velocity[1][1]) + divv;
        double Pzz = 2*(wt_i*local.Gradients.Velocity[2][2] + wt_j*CPj.Gradients.Velocity[2][2]) + divv;
        double Pxy = wt_i*(local.Gradients.Velocity[0][1] + local.Gradients.Velocity[1][0]) +
                     wt_j*(CPj.Gradients.Velocity[0][1] + CPj.Gradients.Velocity[1][0]);
        double Pxz = wt_i*(local.Gradients.Velocity[0][2] + local.Gradients.Velocity[2][0]) +
                     wt_j*(CPj.Gradients.Velocity[0][2] + CPj.Gradients.Velocity[2][0]);
        double Pyz = wt_i*(local.Gradients.Velocity[1][2] + local.Gradients.Velocity[2][1]) +
                     wt_j*(CPj.Gradients.Velocity[1][2] + CPj.Gradients.Velocity[2][1]);

        cmag[0] = Pxx*Face_Area_Vec[0] + Pxy*Face_Area_Vec[1] + Pxz*Face_Area_Vec[2];
        cmag[1] = Pxy*Face_Area_Vec[0] + Pyy*Face_Area_Vec[1] + Pyz*Face_Area_Vec[2];
        cmag[2] = Pxz*Face_Area_Vec[0] + Pyz*Face_Area_Vec[1] + Pzz*Face_Area_Vec[2];

        double dv_dir = (zeta - eta*2./3.) * dot(kernel.dv, kernel.dp);
        double Pxx_direct = eta*2.*kernel.dv[0]*kernel.dp[0] + dv_dir;
        double Pyy_direct = eta*2.*kernel.dv[1]*kernel.dp[1] + dv_dir;
        double Pzz_direct = eta*2.*kernel.dv[2]*kernel.dp[2] + dv_dir;
        double Pxy_direct = eta*(kernel.dv[0]*kernel.dp[1] + kernel.dv[1]*kernel.dp[0]);
        double Pxz_direct = eta*(kernel.dv[0]*kernel.dp[2] + kernel.dv[2]*kernel.dp[0]);
        double Pyz_direct = eta*(kernel.dv[2]*kernel.dp[1] + kernel.dv[1]*kernel.dp[2]);
        cmag_dir[0] = (Pxx_direct*Face_Area_Vec[0] + Pxy_direct*Face_Area_Vec[1] + Pxz_direct*Face_Area_Vec[2]) / Face_Area_Norm * kernel.r;
        cmag_dir[1] = (Pxy_direct*Face_Area_Vec[0] + Pyy_direct*Face_Area_Vec[1] + Pyz_direct*Face_Area_Vec[2]) / Face_Area_Norm * kernel.r;
        cmag_dir[2] = (Pxz_direct*Face_Area_Vec[0] + Pyz_direct*Face_Area_Vec[1] + Pzz_direct*Face_Area_Vec[2]) / Face_Area_Norm * kernel.r;
    }

    double rho_i = local.Density * All.cf_a3inv;
    double rho_j = CPj.Density * All.cf_a3inv;
    double rho_ij = 0.5 * (rho_i + rho_j);

    cmag *= All.cf_a2inv;

    for(int k_v=0; k_v<3; k_v++)
    {
#ifdef MAGNETIC
        double b_hll_eff = DMAX(DMIN(1., 3.*bhat_dot_gradvhat*bhat_dot_gradvhat), 0.01);
        double dv_visc = 3. * (B_interface[k_v]*Bi_proj - kernel.dp[k_v]/3.) * bhat_dot_gradvhat_direct * a_inv;
        double thold_ptot_hll = 0.1 * exp(-2. * grad_v_mag * kernel.r / (1.e-30 + fabs(kernel.dv[k_v])));
#else
        double b_hll_eff = 1;
        double dv_visc = cmag_dir[k_v] * rinv * rinv / (All.cf_atime * DMAX(eta, zeta));
        double thold_ptot_hll = 0.03;
#endif
        double hll_tmp = rho_ij * hll_correction_fn(dv_visc, -dv_visc, rho_ij, viscous_wt_physical,
                                                    v_hll, Face_Area_Norm, kernel.r, All.cf_atime);
        double fluxlimiter_absnorm = -DMAX(eta, zeta) * sqrt(b_hll_eff) * Face_Area_Norm * dv_visc * rinv * a_inv;
        double ptot = DMIN(local.Mass, Pj.Mass) * kernel.dv.norm() / (1.e-37 + dt_hydrostep) * a_inv;
        double thold_hll = 0.1 * ptot;
        if(fabs(hll_tmp) > thold_hll) { hll_tmp *= thold_hll / fabs(hll_tmp); }
        hll_tmp *= b_hll_eff;
        if(cmag[k_v] * hll_tmp <= 0) {
            thold_hll = b_hll_eff * thold_ptot_hll * ptot;
            if(fabs(hll_tmp) > thold_hll) { hll_tmp *= thold_hll / fabs(hll_tmp); }
        } else {
            thold_hll = b_hll_eff * DMAX(fabs(0.5*cmag[k_v]), 0.3*thold_ptot_hll*ptot);
            if(fabs(hll_tmp) > thold_hll) { hll_tmp *= thold_hll / fabs(hll_tmp); }
        }
        double cmag_corr = cmag[k_v] + hll_tmp;
        cmag[k_v] = MINMOD(cmag[k_v], cmag_corr);
        double check_for_stability_sign = fluxlimiter_absnorm * cmag[k_v];
        if((check_for_stability_sign < 0) && (fabs(fluxlimiter_absnorm) > HLL_DIFFUSION_OVERSHOOT_FACTOR * fabs(cmag[k_v]))) { cmag[k_v] = 0; }

#if defined(GALSF)
        if(check_for_stability_sign < 0) { cmag[k_v] = 0; }
#endif
    }

    double v_dot_dv = dot(kernel.dv, cmag) * a_inv;
    if(v_dot_dv > 0) {
        cmag = {};
    } else {
        double KE_com = kernel.dv.norm_sq() * All.cf_a2inv;
        KE_com *= 0.25 * (local.Mass + Pj.Mass);
        double dKE_q = fabs(v_dot_dv) * dt_hydrostep / (1.e-40 + KE_com);
        double threshold_tmp = 1.0;
        double lim_corr = 1;
        if(dKE_q > threshold_tmp) { lim_corr = threshold_tmp / dKE_q; }
        cmag *= lim_corr;
    }

    double cmag_E = dot(cmag, v_interface);
    if(dt_hydrostep > 0) {
        double cmag_lim = 0.25 * (local.Mass + Pj.Mass) * v_interface.norm_sq() / dt_hydrostep;
        if(fabs(cmag_E) > cmag_lim) {
            double corr_visc = cmag_lim / fabs(cmag_E);
            cmag *= corr_visc; cmag_E *= corr_visc;
        }
    }
    Fluxes.v += cmag;
    Fluxes.p += cmag_E;
#endif /* VISCOSITY */
}

#endif /* VISCOSITY_FUNCTIONS_H */
