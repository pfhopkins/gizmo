/* conduction_functions.h -- per-pair conduction flux as a proper function.
 *
 * Replaces the #include-fragment hydro/conduction.h. Exposed as
 * KOKKOS_INLINE_FUNCTION so the GPU hydro lambda and the CPU tree-walk can both
 * call it; conduction is pure i-accumulation into Fluxes, no j-side writes.
 *
 * Design pattern (applies to all per-pair physics function ports):
 *   - Callers always call the function unconditionally; when the relevant
 *     physics flag is not defined the body compiles to nothing. That keeps
 *     call-site logic clean (no per-module #ifdef around each invocation).
 *   - Magnetic-field inputs (bhat, bhat_mag) are always in the signature;
 *     the caller passes zero-initialised dummies when MAGNETIC is off, so the
 *     signature itself doesn't need conditional compilation.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h and hydro_pair_types.h already
 * included.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef CONDUCTION_FUNCTIONS_H
#define CONDUCTION_FUNCTIONS_H

#include "hydro_pair_types.h"

KOKKOS_INLINE_FUNCTION
void conduction_compute_pair(
    const struct hydro_data_in &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
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
#ifdef CONDUCTION
    double scalar_i = local.InternalEnergyPred;
    double scalar_j = CPj.InternalEnergyPred;
    double kappa_i  = local.Kappa_Conduction;
    double kappa_j  = CPj.Kappa_Conduction;

    if(!((kappa_i > MIN_REAL_NUMBER) && (kappa_j > MIN_REAL_NUMBER) && (local.Mass > 0) && (Pj.Mass > 0))) { return; }

    double d_scalar = scalar_i - scalar_j;
    double rho_i  = local.Density * All.cf_a3inv;
    double rho_j  = CPj.Density * All.cf_a3inv;
    double rho_ij = 0.5 * (rho_i + rho_j);

    const double *grad_i = local.Gradients.InternalEnergy.data_ptr();
    const double *grad_j = CPj.Gradients.InternalEnergy.data_ptr();

    double flux_wt = rho_ij;
    double diffusion_wt = 0.5 * (kappa_i + kappa_j);
#ifdef COOLING
    diffusion_wt = kappa_i * kappa_j / diffusion_wt;
#endif
    int do_isotropic = 1;
    double b_hll = 1, cmag = 0, wt_i = 0.5, wt_j = 0.5, grad_dot_x_ij = 0;
    Vec3<double> grad_ij;
    for(int k = 0; k < 3; k++)
    {
        double q_grad = wt_i * grad_i[k] + wt_j * grad_j[k];
        double q_direct = d_scalar * kernel.dp[k] * rinv * rinv;
        grad_dot_x_ij += q_grad * kernel.dp[k];
#ifdef MAGNETIC
        grad_ij[k] = MINMOD_G(q_grad, q_direct);
        if(q_grad * q_direct < 0) { if(fabs(q_direct) > 5. * fabs(q_grad)) { grad_ij[k] = 0.0; } }
#else
        grad_ij[k] = MINMOD(q_grad, q_direct);
#endif
    }
    if(bhat_mag > 0)
    {
        do_isotropic = 0;
        double B_interface_dot_grad_T = dot(bhat, grad_ij);
        double grad_mag = grad_ij.norm_sq();
        cmag = dot(bhat, Face_Area_Vec);
        cmag *= B_interface_dot_grad_T;
        if(grad_mag > 0) { grad_mag = sqrt(grad_mag); } else { grad_mag = 1; }
        b_hll = B_interface_dot_grad_T / grad_mag;
        b_hll *= b_hll;
    }
    if(do_isotropic) { cmag = dot(Face_Area_Vec, grad_ij); }
    cmag /= All.cf_atime;

    double d_scalar_tmp = d_scalar - grad_dot_x_ij;
    double d_scalar_hll = MINMOD(d_scalar, d_scalar_tmp);
    double hll_corr = b_hll * flux_wt
        * hll_correction_fn(d_scalar_hll, 0., flux_wt, diffusion_wt,
                            v_hll, Face_Area_Norm, kernel.r, All.cf_atime) / (-diffusion_wt);
    double cmag_corr = cmag + hll_corr;
    cmag = MINMOD(HLL_DIFFUSION_COMPROMISE_FACTOR * cmag, cmag_corr);

    double d_scalar_b = b_hll * d_scalar;
    double f_direct = Face_Area_Norm * d_scalar_b * rinv / All.cf_atime;
    double check_for_stability_sign = f_direct * cmag;
    if((check_for_stability_sign < 0) && (fabs(f_direct) > HLL_DIFFUSION_OVERSHOOT_FACTOR * fabs(cmag))) { cmag = 0; }
    cmag *= -diffusion_wt;

    diffusion_wt = dt_hydrostep * cmag;
    if(fabs(diffusion_wt) > 0)
    {
        double du_ij_cond = DMIN(0.25 * fabs(local.Mass * scalar_i - Pj.Mass * scalar_j),
                                 DMAX(local.Mass * scalar_i, Pj.Mass * scalar_j));
        if(check_for_stability_sign < 0) { du_ij_cond *= 1.e-2; }
        if(fabs(diffusion_wt) > du_ij_cond) { diffusion_wt *= du_ij_cond / fabs(diffusion_wt); }
        Fluxes.p += diffusion_wt / dt_hydrostep;
    }
#endif /* CONDUCTION */
}

#endif /* CONDUCTION_FUNCTIONS_H */
