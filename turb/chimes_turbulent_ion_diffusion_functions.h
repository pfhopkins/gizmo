/* chimes_turbulent_ion_diffusion_functions.h -- per-pair CHIMES ion/molecule
 * turbulent diffusion. Replaces the fragment turb/chimes_turbulent_ion_diffusion.h.
 *
 * Body guarded by CHIMES_TURB_DIFF_IONS so callers invoke unconditionally. A
 * CHIMES_TOTSIZE fallback (=1) is defined when CHIMES is disabled so that the
 * caller's j-delta buffer has a valid size regardless.
 *
 * Writes:
 *   - out.ChimesIonsYield[k]          (i-side; no atomic)
 *   - ChimesNIons_j_delta[k]          (j-side; amount to *add* to
 *                                     CellP[j].ChimesNIons[k], matches the
 *                                     -cmag semantics of the original fragment)
 *
 * Requires allvars.h, kernel.h, hydro_structs.h, hydro_pair_types.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu); CHIMES variant by Alex Richings.
 */

#ifndef CHIMES_TURBULENT_ION_DIFFUSION_FUNCTIONS_H
#define CHIMES_TURBULENT_ION_DIFFUSION_FUNCTIONS_H

#include "../hydro/hydro_pair_types.h"

#ifndef CHIMES_TOTSIZE
#define CHIMES_TOTSIZE 1
#endif

KOKKOS_INLINE_FUNCTION
void chimes_turb_diff_ions_compute_pair(
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
    struct hydro_data_out &out,
    MyDouble ChimesNIons_j_delta[CHIMES_TOTSIZE])
{
#ifdef CHIMES_TURB_DIFF_IONS
    for(int _k=0; _k<ChimesGlobalVars.totalNumberOfSpecies; _k++) { ChimesNIons_j_delta[_k] = 0; }

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

    for(int k_species=0; k_species<ChimesGlobalVars.totalNumberOfSpecies; k_species++)
    {
        double local_abundance_times_mass = local.ChimesNIons[k_species];
        double external_abundance_times_mass = CPj.ChimesNIons[k_species];
        double d_scalar = (local_abundance_times_mass/local.Mass) - (external_abundance_times_mass/Pj.Mass);

        double cmag = 0.0, grad_dot_x_ij = 0.0;
        for(int k=0; k<3; k++)
        {
            double grad_direct = d_scalar * kernel.dp[k] * rinv * rinv;
            double grad_ij = grad_direct;   /* CHIMES uses LOWORDER only */
            grad_dot_x_ij += grad_ij * kernel.dp[k];
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
                                       DMAX(local_abundance_times_mass, external_abundance_times_mass) );
            if(fabs(cmag) > zlim) { cmag *= zlim / fabs(cmag); }
#ifndef HYDRO_SPH
            double dmet = ((external_abundance_times_mass/Pj.Mass) - (local_abundance_times_mass/local.Mass)) * fabs(mdot_estimated) * dt_hydrostep;
            cmag = MINMOD(dmet, cmag);
#endif
            out.ChimesIonsYield[k_species] += cmag;
            ChimesNIons_j_delta[k_species] = -cmag;
        }
    }
#endif /* CHIMES_TURB_DIFF_IONS */
}

#endif /* CHIMES_TURBULENT_ION_DIFFUSION_FUNCTIONS_H */
