/* turbulent_diffusion_functions.h -- per-pair turbulent metal diffusion flux.
 *
 * Replaces the #include-fragment turb/turbulent_diffusion.h. Body is guarded
 * by TURB_DIFF_METALS so callers can invoke the function unconditionally; when
 * the flag is off the function compiles to a no-op.
 *
 * Writes out.Dyield[k] (i-side, per-thread accumulator, no atomic needed).
 * j-side writes have been removed — they were gated on the legacy
 * j_is_active_for_fluxes flag which is always 0 in modern GIZMO.
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef TURBULENT_DIFFUSION_FUNCTIONS_H
#define TURBULENT_DIFFUSION_FUNCTIONS_H

#include "../hydro/hydro_pair_types.h"

KOKKOS_INLINE_FUNCTION
void turb_diff_metals_compute_pair(
    const struct hydro_data_in &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
    const struct kernel_hydra &kernel,
    double rinv,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double Face_Density,
    double v_hll,
    double dt_hydrostep,
    double mdot_estimated,
    struct hydro_data_out &out)
{
#ifdef TURB_DIFF_METALS
    if(!((local.Mass>0) && (Pj.Mass>0) && ((local.TD_DiffCoeff>MIN_REAL_NUMBER) || (CPj.TD_DiffCoeff>MIN_REAL_NUMBER)))) { return; }

    double wt_i = 0.5, wt_j = 0.5;
    double diffusion_wt = wt_i*local.TD_DiffCoeff + wt_j*CPj.TD_DiffCoeff;
#ifdef HYDRO_SPH
    diffusion_wt *= 0.5 * (local.Density + CPj.Density) * All.cf_a3inv;
#else
    diffusion_wt *= Face_Density;
#endif
    double massflux = fabs( Face_Area_Norm * diffusion_wt / (DMIN(kernel.h_i, kernel.h_j) * All.cf_atime) * dt_hydrostep / (DMIN(local.Mass, Pj.Mass)) );
    if(massflux > 0.25) { diffusion_wt *= 0.25 / massflux; }

    double rho_i = local.Density * All.cf_a3inv;
    double rho_j = CPj.Density * All.cf_a3inv;
    double rho_ij = 0.5 * (rho_i + rho_j);

    for(int k_species=0; k_species<NUM_METAL_SPECIES; k_species++)
    {
        double cmag = 0.0, grad_dot_x_ij = 0.0;
        double Z_j = Pj.Metallicity[k_species];
        double d_scalar = local.Metallicity[k_species] - Z_j;
        for(int k=0; k<3; k++)
        {
            double grad_direct = d_scalar * kernel.dp[k] * rinv * rinv;
            double grad_ij = grad_direct;
#if !defined(TURB_DIFF_METALS_LOWORDER)
            grad_ij = wt_i*local.Gradients.Metallicity[k_species][k] + wt_j*CPj.Gradients.Metallicity[k_species][k];
#endif
            grad_dot_x_ij += grad_ij * kernel.dp[k];
            grad_ij = MINMOD(grad_ij, grad_direct);
            cmag += Face_Area_Vec[k] * grad_ij;
        }
        cmag /= All.cf_atime;

        double d_scalar_tmp = d_scalar - grad_dot_x_ij;
        double d_scalar_hll = MINMOD(d_scalar, d_scalar_tmp);
        double hll_corr = rho_ij
            * hll_correction_fn(d_scalar_hll, 0., rho_ij, diffusion_wt, v_hll, Face_Area_Norm, kernel.r, All.cf_atime)
            / (-diffusion_wt);
        double cmag_corr = cmag + hll_corr;
        cmag = MINMOD(1.5*cmag, cmag_corr);
        double f_direct = Face_Area_Norm * d_scalar * rinv / All.cf_atime;
        if((f_direct*cmag < 0) && (fabs(f_direct) > HLL_DIFFUSION_OVERSHOOT_FACTOR*fabs(cmag))) { cmag = 0; }

        cmag *= -diffusion_wt * dt_hydrostep;
        if(fabs(cmag) > 0)
        {
            double zlim = 0.25 * DMIN( DMIN(local.Mass, Pj.Mass)*fabs(d_scalar),
                                       DMAX(local.Mass*local.Metallicity[k_species], Pj.Mass*Z_j) );
            if(fabs(cmag) > zlim) { cmag *= zlim / fabs(cmag); }
#ifndef HYDRO_SPH
            double dmet = (Z_j - local.Metallicity[k_species]) * fabs(mdot_estimated) * dt_hydrostep;
            cmag = MINMOD(dmet, cmag);
#endif
            out.Dyield[k_species] += cmag;
        }
    }
#endif /* TURB_DIFF_METALS */
}

#endif /* TURBULENT_DIFFUSION_FUNCTIONS_H */
