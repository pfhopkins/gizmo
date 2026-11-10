/* turbulent_diffusion_functions.h -- per-pair turbulent metal diffusion flux.
 *
 * Replaces the #include-fragment turb/turbulent_diffusion.h. Body is guarded
 * by TURB_DIFF_METALS so callers can invoke the function unconditionally; when
 * the flag is off the function compiles to a no-op.
 *
 * Writes out.Dyield[k] (i-side, per-thread accumulator, no atomic needed).
 * With TURB_DIFF_METALS_EXACT_CONSERVATION, each side moves half the exchange and
 * books the opposite half into pending_j (CellP[j].Dyield_pending, atomic),
 * which j applies at its next active step; pending_j exists only with the flag.
 * Each side applies the exchange for its own step: the limiters bound the amount
 * over the pair step dt_pair = max(dt_i, dt_j), and cell i receives the fraction
 * dt_i/dt_pair of it.
 * With METALS_EXACT_PAIRWISE_EXCHANGE the shorter step owns the pair (equal steps: half
 * each), computes it over its own step and also writes -exchange to CellP[j].Dyield.
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
    double dt_pair,
    double dt_i,
    double mdot_estimated,
    struct hydro_data_out &out
#ifdef TURB_DIFF_METALS_EXACT_CONSERVATION
    , MyDouble *pending_j
#endif
    )
{
#ifdef TURB_DIFF_METALS
    if(!((local.Mass>0) && (Pj.Mass>0) && ((local.TD_DiffCoeff>MIN_REAL_NUMBER) || (CPj.TD_DiffCoeff>MIN_REAL_NUMBER)))) { return; }
#ifdef METALS_EXACT_PAIRWISE_EXCHANGE
    const double dt_j = get_particle_timestep_in_physical(0, const_cast<struct particle_data *>(&Pj));
    if(!PAIR_EXCHANGE_I_OWNS(dt_i, dt_j) && !PAIR_EXCHANGE_SPLIT(dt_i, dt_j)) { return; } /* j owns the pair and does it at its own step */
    const double owner_share = PAIR_EXCHANGE_I_OWNS(dt_i, dt_j) ? 1.0 : 0.5;
    dt_pair = dt_i; /* the owner's step */
#endif

    double wt_i = 0.5, wt_j = 0.5;
    double diffusion_wt = wt_i*local.TD_DiffCoeff + wt_j*CPj.TD_DiffCoeff;
#ifdef HYDRO_SPH
    diffusion_wt *= 0.5 * (local.Density + CPj.Density) * All.cf_a3inv;
#else
    diffusion_wt *= Face_Density;
#endif
    double massflux = fabs( Face_Area_Norm * diffusion_wt / (DMIN(kernel.h_i, kernel.h_j) * All.cf_atime) * dt_pair / (DMIN(local.Mass, Pj.Mass)) );
    if(massflux > 0.25) { diffusion_wt *= 0.25 / massflux; }

    double rho_i = local.Density * All.cf_a3inv;
    double rho_j = CPj.Density * All.cf_a3inv;
    double rho_ij = 0.5 * (rho_i + rho_j);
    const double dt_fraction_i = (dt_pair > 0) ? dt_i / dt_pair : 0; /* cell i receives the amount for its own step */

#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION) && defined(GALSF_SINGLE_STARS_27_ELEMENT_YIELDS)
    double cmag_slot[NUM_METAL_SPECIES] = {0}; /* total Z takes the sum of the metal slots, applied after the loop */
#endif
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

        cmag *= -diffusion_wt * dt_pair;
        if(fabs(cmag) > 0)
        {
            double zlim = 0.25 * DMIN( DMIN(local.Mass, Pj.Mass)*fabs(d_scalar),
                                       DMAX(local.Mass*local.Metallicity[k_species], Pj.Mass*Z_j) );
            if(fabs(cmag) > zlim) { cmag *= zlim / fabs(cmag); }
#ifndef HYDRO_SPH
            double dmet = (Z_j - local.Metallicity[k_species]) * fabs(mdot_estimated) * dt_pair;
            cmag = MINMOD(dmet, cmag);
#endif
#ifdef TURB_DIFF_METALS_MAX_DONOR_FRACTION_PER_PAIR
            /* the donor loses at most this fraction of its species-k mass per pair (same value from either side) */
            double pair_cap = (TURB_DIFF_METALS_MAX_DONOR_FRACTION_PER_PAIR) * ((cmag > 0) ? Pj.Mass*Z_j : local.Mass*local.Metallicity[k_species]);
            if(fabs(cmag) > pair_cap) { cmag = (cmag > 0) ? pair_cap : -pair_cap; }
#endif
#ifdef TURB_DIFF_METALS_EXACT_CONSERVATION
            /* each side moves half the exchange over its own step; j's share waits in its pending buffer */
            cmag *= 0.5 * dt_fraction_i;
#ifdef GALSF_SINGLE_STARS_27_ELEMENT_YIELDS
            cmag_slot[k_species] = cmag;
#else
            out.Dyield[k_species] += cmag;
            Kokkos::atomic_add(&pending_j[k_species], (MyDouble)(-cmag));
#endif
#else
            /* only cell i is updated, with its own step's share of the pair exchange; j takes its share when it is active.
               So across timebins metal mass is conserved to integration accuracy, as momentum and energy are; only total mass
               has machinery that makes it exact (the MFV mass flux, where the endpoint with the shorter step owns the pair and
               updates both sides). METALS_EXACT_PAIRWISE_EXCHANGE does the same for metals (GitHub issue #35). */
#ifdef METALS_EXACT_PAIRWISE_EXCHANGE
            /* positivity on the pair: the donor gives at most 1/(2 N_ngb) of its metal mass to one partner per step */
            double m_donor = (cmag > 0) ? Pj.Mass*Z_j : local.Mass*local.Metallicity[k_species];
            double m_cap = DMAX(m_donor, 0) / (2.*All.DesNumNgb);
            if(fabs(cmag) > m_cap) { cmag *= m_cap / fabs(cmag); }
            out.Dyield[k_species] += owner_share * cmag;
            Kokkos::atomic_add(const_cast<MyFloat *>(&CPj.Dyield[k_species]), (MyFloat)(-owner_share * cmag)); /* CellP[j] itself */
#else
            out.Dyield[k_species] += cmag * dt_fraction_i;
#endif
#endif
        }
    }
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION) && defined(GALSF_SINGLE_STARS_27_ELEMENT_YIELDS)
    cmag_slot[0] = 0;
    for(int k = 2; k <= NUM_LIVE_SPECIES_FOR_COOLTABLES; k++) {cmag_slot[0] += cmag_slot[k];} /* slot 1 is He */
    for(int k = 0; k < 16; k++) {cmag_slot[0] += cmag_slot[METALS_EXTRA_PASSIVE_SPECIES_OFFSET + k];}
    for(int k = 0; k < NUM_METAL_SPECIES; k++) {
        if(cmag_slot[k] != 0) {out.Dyield[k] += cmag_slot[k]; Kokkos::atomic_add(&pending_j[k], (MyDouble)(-cmag_slot[k]));}
    }
#endif
#endif /* TURB_DIFF_METALS */
}

#endif /* TURBULENT_DIFFUSION_FUNCTIONS_H */
