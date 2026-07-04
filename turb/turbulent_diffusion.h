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
        
    if((local.Mass>0)&&(P[j].Mass>0)&&((local.TD_DiffCoeff>MIN_REAL_NUMBER)||(CellP[j].TD_DiffCoeff>MIN_REAL_NUMBER)))
    {
        double wt_i=0.5, wt_j=0.5, cmag, d_scalar;
        double diffusion_wt = wt_i*local.TD_DiffCoeff + wt_j*CellP[j].TD_DiffCoeff; // physical
#ifdef HYDRO_SPH
        diffusion_wt *= 0.5*(local.Density + CellP[j].Density)*All.cf_a3inv; // physical
#else
        diffusion_wt *= Riemann_out.Face_Density; // physical
#endif
        /* calculate implied mass flux 'across the boundary' to prevent excessively large coefficients */
        double massflux = fabs( Face_Area_Norm * diffusion_wt / (DMIN(kernel.h_i,kernel.h_j)*All.cf_atime) * dt_hydrostep / (DMIN(local.Mass,P[j].Mass)) );
        if(massflux > 0.25) {diffusion_wt *= 0.25/massflux;}
        
        int k_species;
        double rho_i = local.Density*All.cf_a3inv, rho_j = CellP[j].Density*All.cf_a3inv, rho_ij = 0.5*(rho_i+rho_j); // physical
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        /* Buffer the 28 Metallicity-slot fluxes: each slot passes through several
         * INDEPENDENT nonlinear limiters below (MINMODs, overshoot-zeroing, zlim,
         * pair cap), so Σ slot-fluxes ≠ 0 and slot0-flux ≠ Σ metal-fluxes at sharp
         * composition fronts → per-cell ΣX and Z0≡Σmetals drift at 1e-6..1e-5 per
         * step (pisn200 reproducer, 2026-07-04) even though pairwise (global)
         * conservation holds.  We close the system after the loop. */
        double cmag_slotbuf[NUM_METAL_SPECIES]; {int kb; for(kb=0;kb<NUM_METAL_SPECIES;kb++) {cmag_slotbuf[kb]=0;}}
#endif
        for(k_species=0;k_species<NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;k_species++)
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
            
            cmag *= -diffusion_wt * dt_hydrostep; // physical
            if(fabs(cmag) > 0)
            {
                double zlim = 0.25 * DMIN( DMIN(local.Mass,P[j].Mass)*fabs(d_scalar) , DMAX(local.Mass*local.Metallicity[k_species] , P[j].Mass*Z_j) );
                if(fabs(cmag)>zlim) {cmag*=zlim/fabs(cmag);}
#ifndef HYDRO_SPH
                double dmet = (Z_j-local.Metallicity[k_species]) * fabs(mdot_estimated) * dt_hydrostep;
                cmag = MINMOD(dmet,cmag); // limiter based on mass exchange from MFV HLLC solver //
#endif
                /* Symmetric per-pair donor cap: limit |cmag| so the donor cannot
                 * lose more than 10% of its element-k mass in a single pair.  Sign
                 * of cmag tells us the donor: cmag > 0 → element flows from j to i
                 * (donor = j), cmag < 0 → donor = i.  Applied to BOTH sides since
                 * the same cmag is used in Dyield_i and Dyield_j below — strictly
                 * symmetric, preserves per-element pairwise mass conservation. */
                double donor_mass_k = (cmag > 0) ? P[j].Mass * Z_j
                                                 : local.Mass * local.Metallicity[k_species];
                double pair_cap = 0.01 * donor_mass_k;  /* 1% per pair; ~50-80 pairs/cell → ≤80% cumulative drain */
                if(fabs(cmag) > pair_cap) cmag = (cmag > 0 ? pair_cap : -pair_cap);

                /* Dual-walk Dyield_pending scatter: each perspective contributes
                 * 0.5*cmag.  Per pair both walks happen, so totals across walks
                 * apply +cmag to i (via Dyield) and −cmag to j (via Dyield_pending).
                 * The pending buffer lets the j-side write survive across timebins:
                 * an inactive cell j accumulates contributions from active neighbors'
                 * walks until j eventually unpacks.  Bit-exact pair conservation. */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                if(k_species < NUM_METAL_SPECIES) {cmag_slotbuf[k_species] = cmag;} else {
                    out.Dyield[k_species]               += 0.5 * cmag;
                    CellP[j].Dyield_pending[k_species]  -= 0.5 * cmag;
                }
#else
                out.Dyield[k_species]               += 0.5 * cmag;
                CellP[j].Dyield_pending[k_species]  -= 0.5 * cmag;
#endif
            }
        }
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        /* ---- Closure corrections (2026-07-04), RESOLVEDISM 28-slot layout only
         * ([0]=Ztot, [1]=H, [2]=He, [3..27]=metals; ΣMet[1..27]=1 convention):
         *   (I2) slot0 flux := Σ metal-slot fluxes  → Z0 ≡ Σmetals preserved
         *   (I1) residual of element-slot fluxes absorbed into H (the ~0.7
         *        reservoir; residual is O(1e-6) of the fluxes) → Σ fluxes = 0
         *        → per-cell ΣX = 1 preserved through the unpack.
         * Both ride the same antisymmetric ±0.5·cmag scatter, so pairwise and
         * global per-element conservation remain bit-exact. */
        {
            int kk; double sum_metal = 0, resid = 0;
            for(kk = 3; kk < NUM_METAL_SPECIES; kk++) {sum_metal += cmag_slotbuf[kk];}
            cmag_slotbuf[0] = sum_metal;
            for(kk = 1; kk < NUM_METAL_SPECIES; kk++) {resid += cmag_slotbuf[kk];}
            cmag_slotbuf[1] -= resid;
            for(kk = 0; kk < NUM_METAL_SPECIES; kk++) {
                if(cmag_slotbuf[kk] != 0) {
                    out.Dyield[kk]              += 0.5 * cmag_slotbuf[kk];
                    CellP[j].Dyield_pending[kk] -= 0.5 * cmag_slotbuf[kk];
                }
            }
        }
#endif
    }
#endif
}
