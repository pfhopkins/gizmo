#ifdef COSMIC_RAYS
/* --------------------------------------------------------------------------------- */
/* ... real cosmic ray diffusion/streaming evaluation ...
 *
 * This does the cosmic ray diffusion/streaming operations.
 *    Coefficients are calculated in gradients.c. Here they are used to stream/diffuse.
 * The SPH formulation of this is possible (copy from conduction.h, if desired), but
 *    known to be very noisy and systematically problematic, because of the inconsistent
 *    second-derivative operator. So MFM/MFV is strongly recommended, and much more
 *    accurate.
 *
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
{
    double cosmo_unit = All.cf_a3inv;
    double scalar_i = local.CosmicRayPressure * cosmo_unit; // physical units
    double scalar_j = CosmicRayPressure_j * cosmo_unit;
    double kappa_i = fabs(local.CosmicRayDiffusionCoeff); // physical units
    double kappa_j = fabs(SphP[j].CosmicRayDiffusionCoeff);
    double d_scalar = scalar_i - scalar_j;
    
    if(((kappa_i>0)||(kappa_j>0))&&(local.Mass>0)&&(P[j].Mass>0)&&(dt_hydrostep>0)&&(Face_Area_Norm>0)&&(dt_hydrostep>0))
    {
#ifndef COSMIC_RAYS_M1
        // NOT SPH: Now we use the more accurate finite-volume formulation, with the effective faces we have already calculated //
        double *grad_i = local.Gradients.CosmicRayPressure; // units = E/[L_comoving^4]
        double *grad_j = SphP[j].Gradients.CosmicRayPressure;
        double flux_wt = 1;
        double diffusion_wt = 0.5*(kappa_i+kappa_j);
        int do_isotropic = 1;
        double b_hll=1, cmag=0, wt_i=0.5, wt_j=0.5, grad_dot_x_ij = 0;
        double grad_ij[3];
        for(k=0;k<3;k++)
        {
            double q_grad = (wt_i*grad_i[k] + wt_j*grad_j[k]) * cosmo_unit; // units = E/[L_phys^3 * L_comoving]
            double q_direct = d_scalar * kernel.dp[k] * rinv*rinv; // units = E/[L_phys^3 * L_comoving]
            grad_dot_x_ij += q_grad * kernel.dp[k]; // units = E/L_phys^3
#ifdef MAGNETIC
            grad_ij[k] = MINMOD_G(q_grad , q_direct);
            if(q_grad*q_direct < 0) {if(fabs(q_direct) > 5.*fabs(q_grad)) {grad_ij[k] = 0.0;}}
#else
            grad_ij[k] = MINMOD(q_grad , q_direct);
#endif
            // negative coefficient is used here as shorthand for a particle being a local extremum in CR density.
            //  in this case we use a zeroth-order estimate for the flux: more diffusive, but needed to get the gradients resolved
            if((local.CosmicRayDiffusionCoeff<0)||(SphP[j].CosmicRayDiffusionCoeff<0)) {grad_ij[k] = q_direct;}
        }
        
        double grad_mag = 0.0;
        for(k=0;k<3;k++) {grad_mag += grad_ij[k]*grad_ij[k];}
        if(grad_mag > 0) {grad_mag = sqrt(grad_mag);} else {grad_mag=MIN_REAL_NUMBER;}
        if((local.CosmicRayDiffusionCoeff<0)||(SphP[j].CosmicRayDiffusionCoeff<0)) // codes for local maximum: need lower-order gradient estimator
        {
            double gmag_a=0,gmag_b=0,g0=grad_mag*grad_mag;
            for(k=0;k<3;k++) {gmag_a += grad_i[k]*grad_i[k]; gmag_b += grad_j[k]*grad_j[k];}
            double gnew = DMAX(g0,DMAX(gmag_a,gmag_b));
            if(gnew > 0)
            {
                gnew = sqrt(gnew);
                for(k=0;k<3;k++) {grad_ij[k] / grad_mag * gnew;}
                grad_mag = gnew;
            }
        }
#ifdef MAGNETIC
        if(bhat_mag > 0)
        {
            do_isotropic = 0;
            double B_interface_dot_grad_T = 0.0;
            for(k=0;k<3;k++)
            {
                B_interface_dot_grad_T += bhat[k] * grad_ij[k];
                cmag += bhat[k] * Face_Area_Vec[k];
            }
            cmag *= B_interface_dot_grad_T; // L_phys^2 * E/[L_phys^3 * L_comoving] ~ E / (L_phys*L_comoving)
            b_hll = B_interface_dot_grad_T / grad_mag; // dimensionless
            b_hll *= b_hll;
        }
#endif
        if(do_isotropic) {for(k=0;k<3;k++) {cmag += Face_Area_Vec[k] * grad_ij[k];}} // L_phys^2 * E/[L_phys^3 * L_comoving] ~ E / (L_phys*L_comoving)
        cmag /= All.cf_atime; // cmag has units of [E/(L_phys*L_comoving)] -- convert to physical
        
        double check_for_stability_sign = 1; /* if we are using the zeroth-order method, no HLL flux, etc is needed */
        if((local.CosmicRayDiffusionCoeff>=0)&&(SphP[j].CosmicRayDiffusionCoeff>=0))
        {
            /* obtain HLL correction terms for Reimann problem solution */
            double d_scalar_tmp = d_scalar - grad_dot_x_ij; // both in physical units
            double d_scalar_hll = MINMOD(d_scalar , d_scalar_tmp);
            double hll_corr = b_hll * flux_wt * HLL_correction(d_scalar_hll, 0, flux_wt, diffusion_wt) / (-diffusion_wt); // physical units
            double cmag_corr = cmag + hll_corr; // physical units
            cmag = MINMOD(HLL_DIFFUSION_COMPROMISE_FACTOR*cmag, cmag_corr); // physical units
            /* slope-limiter to ensure heat always flows from hot to cold */
            double d_scalar_b = b_hll * d_scalar; // physical units
            double f_direct = Face_Area_Norm * d_scalar_b * (rinv/All.cf_atime); // physical units
            check_for_stability_sign = d_scalar*cmag;
            if((check_for_stability_sign < 0) && (fabs(f_direct) > HLL_DIFFUSION_OVERSHOOT_FACTOR*fabs(cmag))) {cmag = 0;}
        }
        if(check_for_stability_sign<0) {cmag=0;}
        cmag *= -diffusion_wt; /* multiply through coefficient to get flux; all physical units here */
        
        /* follow that with a flux limiter as well */
        diffusion_wt = dt_hydrostep * cmag; // all in physical units //
        if(fabs(diffusion_wt) > 0)
        {
            // enforce a flux limiter for stability (to prevent overshoot) //
            double CR_egy_i = local.CosmicRayPressure*V_i / GAMMA_COSMICRAY_MINUS1; // (E_cr = Volume * (Pressure/(GAMMA_CR-1))) - this is physical units //
            double CR_egy_j = CosmicRayPressure_j*V_j / GAMMA_COSMICRAY_MINUS1;
            double prefac_duij = 0.25, flux_multiplier = 1;
            if((local.CosmicRayDiffusionCoeff<0)||(SphP[j].CosmicRayDiffusionCoeff<0)) {prefac_duij = 0.05;}
            double du_ij_cond = prefac_duij * DMAX(DMIN( fabs(CR_egy_i-CR_egy_j) , DMAX(CR_egy_i , CR_egy_j)) , DMIN(CR_egy_i , CR_egy_j));
            if(diffusion_wt > 0) {du_ij_cond=DMIN(du_ij_cond,0.25*CR_egy_j);} else {du_ij_cond=DMIN(du_ij_cond,0.25*CR_egy_i);} // prevent flux from creating negative values //
            if(fabs(diffusion_wt)>du_ij_cond) {flux_multiplier = du_ij_cond/fabs(diffusion_wt);}
            diffusion_wt *= flux_multiplier;
            Fluxes.CosmicRayPressure += diffusion_wt / dt_hydrostep; // physical units, as needed
        } // if(diffusion_wt > 0)
        
        
#else // COSMIC_RAYS_M1 is active
        
        /* calculate the eigenvalues for the HLLE flux-weighting */
        double cmag=0., flux_norm=0, flux_i[3]={0}, flux_j[3]={0}, thold_hll, V_i_phys = V_i / All.cf_a3inv, V_j_phys = V_j / All.cf_a3inv;
        for(k=0;k<3;k++)
        {
            /* the flux is already known (its explicitly evolved, rather than determined by the gradient of the energy density */
            flux_i[k] = local.CosmicRayFlux[k]/V_i_phys; flux_j[k] = SphP[j].CosmicRayFlux[k]/V_j_phys; // this needs to be in physical units
            double flux_ij = 0.5*(flux_i[k] + flux_j[k]); flux_norm += flux_ij*flux_ij;
            cmag += Face_Area_Vec[k] * flux_ij; // remember, our 'flux' variable is a volume-integral; physical units here//
        }
        flux_norm = sqrt(flux_norm) + MIN_REAL_NUMBER;
        double cos_theta_face_flux = cmag / (Face_Area_Norm * flux_norm); // angle between flux and face vector normal
        if(cos_theta_face_flux < -1) {cos_theta_face_flux=-1;} else {if(cos_theta_face_flux > 1) {cos_theta_face_flux=1;}}
        
        /* add asymptotic-preserving correction so that numerical flux doesn't unphysically dominate in optically thick limit */
#ifdef COSMIC_RAYS_ALFVEN
        double c_hll = 0.5*fabs(face_vel_i-face_vel_j) + COSMIC_RAYS_M1; // physical
        double renormerFAC = cos_theta_face_flux*cos_theta_face_flux;
#else
        double kappa_ij = 0.5 * (kappa_i+kappa_j); // physical
        double L_eff_j = Get_Particle_Size(j)*All.cf_atime; // physical
        double c_light = DMAX( 2.*All.cf_afac3*kernel.vsig , DMIN(COSMIC_RAYS_M1 , kappa_ij/DMIN(Particle_Size_i,L_eff_j)));// * (C/All.UnitVelocity_in_cm_per_s);
        double v_eff_light = DMIN(c_light , kappa_ij / L_eff_j); // physical
        double c_hll = 0.5*fabs(face_vel_i-face_vel_j) + v_eff_light; // physical
        double hll_corr = 1. / (1. + 1.5 * c_light * DMAX(L_eff_j/kappa_j , Particle_Size_i/kappa_i)); // all physical units
        /* q below is a limiter to try and make sure the diffusion speed given by the hll flux doesn't exceed the diffusion speed in the diffusion limit */
        double q = 0.5 * c_hll * (kernel.r * All.cf_atime) / fabs(MIN_REAL_NUMBER + kappa_ij); q = (0.2 + q) / (0.2 + q + q*q); // physical
        double renormerFAC = DMIN(1.,fabs(cos_theta_face_flux*cos_theta_face_flux * q * hll_corr)); // physical
#endif
        
        /* flux-limiter to ensure flow is always down the local gradient [no 'uphill' flow] */
        double f_direct = -Face_Area_Norm * c_hll * (d_scalar/GAMMA_COSMICRAY_MINUS1) * renormerFAC; // simple HLL term for frame moving at 1/2 inter-particle velocity: here not limited //
        if(dt_hydrostep > 0)
        {
            double f_direct_max = 0.25 * fabs(d_scalar) * DMIN(V_i_phys , V_j_phys) / dt_hydrostep;
            if(fabs(f_direct) > f_direct_max) {f_direct *= f_direct_max / fabs(f_direct);}
        } else {f_direct = 0;}
        double sign_c0 = f_direct * cmag;
        if((sign_c0 < 0) && (fabs(f_direct) > fabs(cmag))) {cmag = 0;}
        if(cmag != 0)
        {
            if(f_direct != 0)
            {
                thold_hll = 1.0 * fabs(cmag); // add hll term but flux-limited //
                if(fabs(f_direct) > thold_hll) {f_direct *= thold_hll / fabs(f_direct);}
                cmag += f_direct;
            }
            // enforce a flux limiter for stability (to prevent overshoot) //
            cmag *= dt_hydrostep; // all in physical units //
            double sVi = scalar_i*V_i_phys/GAMMA_COSMICRAY_MINUS1, sVj = scalar_j*V_j_phys/GAMMA_COSMICRAY_MINUS1; // physical units
            thold_hll = DMAX(DMIN(fabs(sVi), fabs(sVj)), DMIN(fabs(sVi-sVj), DMAX(fabs(sVi), fabs(sVj))));
            if(sign_c0 < 0) {thold_hll *= 0.001;} // if opposing signs, restrict this term //
            if(fabs(cmag)>thold_hll) {cmag *= thold_hll/fabs(cmag);}
            cmag /= dt_hydrostep;
            Fluxes.CosmicRayPressure = cmag; // physical, as it needs to be
        } // cmag != 0
        
#ifdef COSMIC_RAYS_ALFVEN
        // pre-compute all the quanitities of interest //
        double A_dot_bhat=0, flux_tmp[2]; int k_j_to_i=0, k_i_to_j=1; for(k=0;k<3;k++) {A_dot_bhat += Face_Area_Vec[k]*bhat[k];}
        if(A_dot_bhat < 0) {k_j_to_i=1; k_i_to_j=0;}
        A_dot_bhat = fabs(A_dot_bhat);
        // assign HLLC fluxes. for this simple (pure energy advection) case, with mass flux already handled, it's just out of one, into another; the hllc dissipation vanishes //
        flux_tmp[k_j_to_i] = +A_dot_bhat * sqrt(kernel.alfven2_j) / V_j_phys;
        flux_tmp[k_i_to_j] = -A_dot_bhat * sqrt(kernel.alfven2_i) / V_i_phys;
        // flux-limit for stability /
        if(flux_tmp[k_j_to_i]*dt_hydrostep > +0.5) {flux_tmp[k_j_to_i] = +0.5/dt_hydrostep;}
        if(flux_tmp[k_i_to_j]*dt_hydrostep < -0.5) {flux_tmp[k_i_to_j] = -0.5/dt_hydrostep;}
        // now just assign these fluxes: written lengthily here to prevent any typos, but trivial assignment //
        Fluxes.CosmicRayAlfvenEnergy[k_j_to_i] += flux_tmp[k_j_to_i] * SphP[j].CosmicRayAlfvenEnergyPred[k_j_to_i];
        Fluxes.CosmicRayAlfvenEnergy[k_i_to_j] += flux_tmp[k_i_to_j] * local.CosmicRayAlfvenEnergy[k_i_to_j];
        for(k=0;k<2;k++) {out.DtCosmicRayAlfvenEnergy[k] += Fluxes.CosmicRayAlfvenEnergy[k];}
        if(j_is_active_for_fluxes) {for(k=0;k<2;k++) {SphP[j].DtCosmicRayAlfvenEnergy[k] -= Fluxes.CosmicRayAlfvenEnergy[k];}}
#endif
        
#endif // COSMIC_RAYS_M1
    } // close check that kappa and particle masses are positive
    // actually assign the fluxes //
    out.DtCosmicRayEnergy += Fluxes.CosmicRayPressure;
    if(j_is_active_for_fluxes) {SphP[j].DtCosmicRayEnergy -= Fluxes.CosmicRayPressure;}
}
#endif // COSMIC_RAYS
