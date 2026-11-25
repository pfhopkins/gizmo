/* --------------------------------------------------------------------------------- */
/* ... turbulent diffusion (sub-grid) models ...
 *
 *  The basic equations here follow the Smagorinky eddy diffusion model. For SPH, the
 *    discretization comes from Wadsley 2008 & Shen 2010. However, some caution is needed,
 *    for SPH, this relys on the (noisy and zeroth-order inconsistent) SPH second-derivative
 *    operator. So a large kernel is especially useful to minimize the systematic errors.
 *  For MFM/MFV methods, the consistent finite-volume formulation is used, which
 *    greatly minimizes artificial (numerical) diffusion.
 *  In either case, since we solve the diffusion equations explicitly, a stronger timestep
 *    restriction is necessary (since the equations are parabolic); this is in timestep.c.
 *    This is very important (many implementations of these equations in the literature
 *    do not include the appropriate timestep and flux-limiters; that makes the equations
 *    numerically unstable (you can get an answer, but it might be wrong, independent of resolution)
 *
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
{
#ifdef TURB_DIFF_METALS // turbulent diffusion of metals (passive scalar mixing) //
    double td_share = 1, dt_td = dt_hydrostep; /* share of the pair exchange this evaluation books, and the step it is computed over */
#ifdef METALS_EXACT_PAIRWISE_EXCHANGE
    /* pair owner (STL-277): with j inactive i owns the pair; both active: the single HYDRO_PAIR_ONCE evaluation owns it, otherwise
       the shorter step, equal steps half each. The owner computes the exchange over its own step and books it to both cells. */
    if(P[j].KernelRadius <= 0 || local.KernelRadius <= 0) {td_share = 0;} /* the gradient pass does not count such a cell */
    else if(TimeBinActive[P[j].TimeBin]) {
#ifndef HYDRO_PAIR_ONCE
        if(dt_hydrostep_i > dt_hydrostep_j) {td_share = 0;} else if(dt_hydrostep_i == dt_hydrostep_j) {td_share = 0.5;}
#endif
    }
    dt_td = dt_hydrostep_i;
#endif

#ifdef RESOLVED_ISM_FB_AFTER_CLOSING_KICK
    if(!HydroPassRepeatsStep) /* exchange applied directly to Metallicity: not again in the feedback recompute */
#endif
    if((td_share>0)&&(local.Mass>0)&&(P[j].Mass>0)&&((local.TD_DiffCoeff>MIN_REAL_NUMBER)||(CellP[j].TD_DiffCoeff>MIN_REAL_NUMBER)))
    {
        double wt_i=0.5, wt_j=0.5, cmag, d_scalar;
        double diffusion_wt = wt_i*local.TD_DiffCoeff + wt_j*CellP[j].TD_DiffCoeff; // physical
#ifdef HYDRO_SPH
        diffusion_wt *= 0.5*(local.Density + CellP[j].Density)*All.cf_a3inv; // physical
#else
        diffusion_wt *= 0.5*(Riemann_vec.L.rho + Riemann_vec.R.rho)*All.cf_a3inv; // physical; reconstructed face densities averaged, same from either side
#endif
        /* calculate implied mass flux 'across the boundary' to prevent excessively large coefficients */
        double massflux = fabs( Face_Area_Norm * diffusion_wt / (DMIN(kernel.h_i,kernel.h_j)*All.cf_atime) * dt_td / (DMIN(local.Mass,P[j].Mass)) );
        if(massflux > 0.25) {diffusion_wt *= 0.25/massflux;}
        
        int k_species;
        double rho_i = local.Density*All.cf_a3inv, rho_j = CellP[j].Density*All.cf_a3inv, rho_ij = 0.5*(rho_i+rho_j); // physical
#ifdef METALS_EXACT_PAIRWISE_EXCHANGE
        double td_amt[TD_NUM_EXCHANGED]; {int kb; for(kb=0;kb<TD_NUM_EXCHANGED;kb++) {td_amt[kb]=0;}} /* exchange into i, closed and bounded after the loop */
        const int td_nk = TD_NUM_EXCHANGED; /* SGChem tracer slots are not diffused (never applied with METALS_INDIVIDUAL) */
#else
        const int td_nk = NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;
#endif
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) && !defined(METALS_EXACT_PAIRWISE_EXCHANGE)
        /* Buffer the 28 Metallicity-slot fluxes: each slot passes through several
         * INDEPENDENT nonlinear limiters below (MINMODs, overshoot-zeroing, zlim,
         * pair cap), so Σ slot-fluxes ≠ 0 and slot0-flux ≠ Σ metal-fluxes at sharp
         * composition fronts → per-cell ΣX and Z0≡Σmetals drift at 1e-6..1e-5 per
         * step (pisn200 reproducer, 2026-07-04) even though pairwise (global)
         * conservation holds.  We close the system after the loop. */
        double cmag_slotbuf[NUM_METAL_SPECIES]; {int kb; for(kb=0;kb<NUM_METAL_SPECIES;kb++) {cmag_slotbuf[kb]=0;}}
#endif
        for(k_species=0;k_species<td_nk;k_species++)
        {
            cmag = 0.0; double grad_dot_x_ij = 0.0; double Z_j = 0;
            if(k_species < NUM_METAL_SPECIES) {Z_j = P[j].Metallicity[k_species];}
#if defined(GALSF_ISMDUSTCHEM_MODEL)
            if(k_species >= NUM_METAL_SPECIES) {Z_j = return_ismdustchem_species_of_interest_for_diffusion_and_yields(j,k_species,0);}
#endif
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || defined(GALSF_RESOLVEDISM_DUST)
            if(k_species >= NUM_METAL_SPECIES) {double Zr = return_resolvedism_species_for_diffusion(j,k_species); if(Zr >= 0) {Z_j = Zr;}}
#endif
            d_scalar = local.Metallicity[k_species]-Z_j; // physical
            for(k=0;k<3;k++)
            {
                double grad_direct = d_scalar * kernel.dp[k] * rinv*rinv; // 1/code length
                double grad_ij = grad_direct;
#if !defined(TURB_DIFF_METALS_LOWORDER)
                if(k_species < NUM_METAL_SPECIES) {grad_ij = wt_i*local.Gradients.Metallicity[k_species][k] + wt_j*CellP[j].Gradients.Metallicity[k_species][k];} // 1/code length
#endif
                grad_dot_x_ij += grad_ij * kernel.dp[k]; // physical
                grad_ij = MINMOD(grad_ij , grad_direct);
                cmag += Face_Area_Vec[k] * grad_ij; // 1/code length
            }
            cmag /= All.cf_atime; // cmag has units of 1/r -- convert to physical

            double d_scalar_tmp = d_scalar - grad_dot_x_ij; // physical
            double d_scalar_hll = MINMOD(d_scalar , d_scalar_tmp);
            double hll_corr = rho_ij * HLL_correction(d_scalar_hll, 0, rho_ij, diffusion_wt) / (-diffusion_wt); // physical
            double cmag_corr = cmag + hll_corr;
            cmag = MINMOD(1.5*cmag, cmag_corr);
            double f_direct = Face_Area_Norm*d_scalar*rinv/All.cf_atime; // physical
            if((f_direct*cmag < 0) && (fabs(f_direct) > HLL_DIFFUSION_OVERSHOOT_FACTOR*fabs(cmag))) {cmag = 0;}
            
            cmag *= -diffusion_wt * dt_td; // physical
            if(fabs(cmag) > 0)
            {
                double zlim = 0.25 * DMIN( DMIN(local.Mass,P[j].Mass)*fabs(d_scalar) , DMAX(local.Mass*local.Metallicity[k_species] , P[j].Mass*Z_j) );
                if(fabs(cmag)>zlim) {cmag*=zlim/fabs(cmag);}
#ifndef HYDRO_SPH
                double dmet = (Z_j-local.Metallicity[k_species]) * fabs(mdot_estimated) * dt_td;
                cmag = MINMOD(dmet,cmag); // limiter based on mass exchange from MFV HLLC solver //
#endif
#ifdef METALS_EXACT_PAIRWISE_EXCHANGE
                td_amt[k_species] = cmag; continue; /* booked after the loop */
#endif
#ifdef GALSF_RESOLVEDISM
                /* symmetric per-pair donor cap: the donor loses at most 1% of its element-k mass per pair */
                double donor_mass_k = (cmag > 0) ? P[j].Mass * Z_j : local.Mass * local.Metallicity[k_species];
                double pair_cap = 0.01 * donor_mass_k;
                if(fabs(cmag) > pair_cap) cmag = (cmag > 0 ? pair_cap : -pair_cap);
#endif
#ifdef TURB_DIFF_METALS_EXACT_CONSERVATION
                /* each side moves half the exchange over its own timestep; j's share waits in its pending buffer */
                cmag *= 0.5 * dt_hydrostep_i / dt_hydrostep; // limiters above used dt_hydrostep >= dt_hydrostep_i
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                if(k_species < NUM_METAL_SPECIES) {cmag_slotbuf[k_species] = cmag; continue;} /* applied after the closure below */
#endif
                out.Dyield[k_species] += cmag;
                #pragma omp atomic
                CellP[j].Dyield_pending[k_species] -= cmag;
#else
                /* the limiters above bound the exchange over the pair step dt_hydrostep: each side applies the share for its own step */
                out.Dyield[k_species] += FluxCorrectionFactor_to_i * cmag * (dt_hydrostep_i / dt_hydrostep);
                if(j_is_active_for_fluxes) {CellP[j].Dyield[k_species] -= FluxCorrectionFactor_to_j * cmag * (dt_hydrostep_j / dt_hydrostep);}
#endif
            }
        }
#ifdef METALS_EXACT_PAIRWISE_EXCHANGE
        {
            int kk;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
            /* 28-slot closure: the residual of slots 1..27 shared in proportion to |amount| (a slot without exchange takes none), slot 0 =
               sum of the metal slots, so every cell keeps sum X = 1 and Z = sum of its metals */
            double resid = 0, sabs = 0, sum_metal = 0;
            for(kk = 1; kk < NUM_METAL_SPECIES; kk++) {resid += td_amt[kk]; sabs += fabs(td_amt[kk]);}
            if(sabs > 0) {for(kk = 1; kk < NUM_METAL_SPECIES; kk++) {td_amt[kk] -= resid * fabs(td_amt[kk]) / sabs;}}
            for(kk = 3; kk < NUM_METAL_SPECIES; kk++) {sum_metal += td_amt[kk];}
            td_amt[0] = sum_metal;
#endif
            /* debit bound: one factor on the whole pair; a cell gives at most its holding / N per evaluation, N = evaluations that can touch it
               this pass (counted in the gradient pass), so no slot of any cell is overdrawn when the pass is applied */
            double td_s = 1;
            for(kk = 0; kk < TD_NUM_EXCHANGED; kk++) {
                double a = td_share * td_amt[kk], hold, nd;
                if(a == 0) {continue;}
                if(a > 0) { /* j gives */
#if defined(GALSF_RESOLVEDISM_DUST)
                    if(kk >= NUM_METAL_SPECIES) {hold = P[j].Mass * CellP[j].Dust[kk-NUM_METAL_SPECIES];} else
#endif
                    {hold = P[j].Mass * P[j].Metallicity[kk];}
                    nd = CellP[j].TD_NumPairs;
                } else {hold = local.Mass * local.Metallicity[kk]; nd = local.TD_NumPairs;} /* i gives */
                double lim = (1 - 1e-12) * DMAX(hold, 0) / DMAX(nd, 1); /* margin for the sum's round-off */
                if(fabs(a) * td_s > lim) {td_s = lim / fabs(a);}
            }
            for(kk = 0; kk < TD_NUM_EXCHANGED; kk++) {
                double a = td_share * td_s * td_amt[kk];
                if(a == 0) {continue;}
                out.Dyield[kk] += a;
                #pragma omp atomic
                CellP[j].Dyield[kk] -= a;
            }
            #pragma omp atomic write
            CellP[j].TD_Touched = 1;
#ifdef OUTPUT_TURB_DIFF_METALS_BUDGET
            #pragma omp atomic
            TD_EvalsStep += 1;
            if(td_s < 1) {
                #pragma omp atomic
                TD_ScaledStep += 1;}
#endif
#ifdef TEST_PRINT_DIFFUSION_H_CLOSURE
            out.TD_NumEvals += 1;
            #pragma omp atomic
            CellP[j].TD_NumEvals += 1;
#endif
        }
#endif
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) && !defined(METALS_EXACT_PAIRWISE_EXCHANGE)
        /* ---- Closure corrections (2026-07-04), RESOLVEDISM 28-slot layout only
         * ([0]=Ztot, [1]=H, [2]=He, [3..27]=metals; ΣMet[1..27]=1 convention):
         *   (I2) slot0 flux := Σ metal-slot fluxes  → Z0 ≡ Σmetals preserved
         *   (I1) residual of element-slot fluxes absorbed into H (the ~0.7
         *        reservoir; residual is O(1e-6) of the fluxes) → Σ fluxes = 0
         *        → per-cell ΣX = 1 preserved through the unpack.
         * Both ride the same antisymmetric (half-exchange) scatter, so pairwise and
         * global per-element conservation remain bit-exact. */
        {
            int kk; double sum_metal = 0, resid = 0;
            for(kk = 3; kk < NUM_METAL_SPECIES; kk++) {sum_metal += cmag_slotbuf[kk];}
            cmag_slotbuf[0] = sum_metal;
            for(kk = 1; kk < NUM_METAL_SPECIES; kk++) {resid += cmag_slotbuf[kk];}
            cmag_slotbuf[1] -= resid;
            for(kk = 0; kk < NUM_METAL_SPECIES; kk++) {
                if(cmag_slotbuf[kk] != 0) {
                    out.Dyield[kk]              += cmag_slotbuf[kk];
                    #pragma omp atomic
                    CellP[j].Dyield_pending[kk] -= cmag_slotbuf[kk];
                }
            }
        }
#endif
    }
#endif
}
