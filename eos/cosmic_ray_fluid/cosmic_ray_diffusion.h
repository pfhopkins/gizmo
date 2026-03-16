#ifdef COSMIC_RAY_FLUID
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
int k_CRegy; double sqrtthreeinv,cosmo_unit,V_i_phys,V_j_phys; sqrtthreeinv=1./sqrt(3.0); cosmo_unit=All.cf_a3inv; V_i_phys=V_i/All.cf_a3inv; V_j_phys=V_j/All.cf_a3inv;
for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
{
    double scalar_i = local.CosmicRayPressure[k_CRegy] * cosmo_unit; // physical units
    double scalar_j = CosmicRayPressure_j[k_CRegy] * cosmo_unit;
    double kappa_i = fabs(local.CosmicRayDiffusionCoeff[k_CRegy]); // physical units
    double kappa_j = fabs(CellP[j].CosmicRayDiffusionCoeff[k_CRegy]);
    double d_scalar = scalar_i - scalar_j;
    double gamma_cr_m1 = GAMMA_COSMICRAY(k_CRegy)-1.;
    
    if(((kappa_i>MIN_REAL_NUMBER)||(kappa_j>MIN_REAL_NUMBER))&&(local.Mass>0)&&(P[j].Mass>0)&&(dt_hydrostep>MIN_REAL_NUMBER)&&(Face_Area_Norm>MIN_REAL_NUMBER))
    {
        /* calculate the eigenvalues for the HLLE flux-weighting */
        double cmag=0., flux_norm=0, flux_i[3]={0}, flux_j[3]={0}, thold_hll;
        for(k=0;k<3;k++)
        {
            /* the flux is already known (its explicitly evolved, rather than determined by the gradient of the energy density */
            flux_i[k] = local.CosmicRayFlux[k_CRegy][k]/V_i_phys; flux_j[k] = CellP[j].CosmicRayFlux[k_CRegy][k]/V_j_phys; // this needs to be in physical units
            double flux_ij = 0.5*(flux_i[k] + flux_j[k]); flux_norm += flux_ij*flux_ij;
            cmag += Face_Area_Vec[k] * flux_ij; // remember, our 'flux' variable is a volume-integral; physical units here//
        }
        flux_norm = sqrt(flux_norm) + MIN_REAL_NUMBER;
        double cos_theta_face_flux = cmag / (Face_Area_Norm * flux_norm); // angle between flux and face vector normal
        if(cos_theta_face_flux < -1) {cos_theta_face_flux=-1;} else {if(cos_theta_face_flux > 1) {cos_theta_face_flux=1;}}
        
        /* add asymptotic-preserving correction so that numerical flux doesn't unphysically dominate in optically thick limit */
        double cr_m1_speed_touse = CRFLUID_REDUCED_C_CODE(k_CRegy);
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        double c_hll = 0.5*fabs(face_vel_i-face_vel_j) + cr_m1_speed_touse; // physical
        double renormerFAC = cos_theta_face_flux*cos_theta_face_flux;
#else
        double kappa_ij = CosmicRayFluid_RSOL_Corrfac(k_CRegy) * 0.5 * (kappa_i+kappa_j); // physical, account for RSOL in effective diffusion speed
        double CRopticaldepth = DMIN(Particle_Size_i,Particle_Size_j)*cr_m1_speed_touse/kappa_ij;
        double reductionfactor = (4.+CRopticaldepth*(4.+3.*CRopticaldepth)) / (4.+CRopticaldepth*(4.+CRopticaldepth*(4.+3.*CRopticaldepth))); // this is just an excellent but simpler/faster approximation to SQRT[1-EXP[-x^2]]/x, which also deals better with small-x limits //
        double reducedcM1 = reductionfactor*cr_m1_speed_touse*sqrtthreeinv; //TK test: correct HLL according Jiang & Oh 2018
        double c_light_eff = DMAX(2.*kernel.vsig, reducedcM1);
        double v_eff_touse = DMIN(c_light_eff , kappa_ij / Particle_Size_j); // physical
        double c_hll = 0.5*fabs(face_vel_i-face_vel_j) + v_eff_touse; // physical
        double hll_corr = 1. / (1. + 1.5 * c_light_eff * DMAX(Particle_Size_j/kappa_j , Particle_Size_i/kappa_i)); // all physical units
        /* q below is a limiter to try and make sure the diffusion speed given by the hll flux doesn't exceed the diffusion speed in the diffusion limit */
        double q = 0.5 * c_hll * (kernel.r * All.cf_atime) / fabs(MIN_REAL_NUMBER + kappa_ij); q = (0.2 + q) / (0.2 + q + q*q); // physical
        double renormerFAC = DMIN(1.,fabs(cos_theta_face_flux*cos_theta_face_flux * q * hll_corr)); // physical
#endif
        
        /* flux-limiter to ensure flow is always down the local gradient [no 'uphill' flow] */
        double f_direct = -Face_Area_Norm * c_hll * (d_scalar/gamma_cr_m1) * renormerFAC; // simple HLL term for frame moving at 1/2 inter-particle velocity: here not limited //
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
            double sVi = scalar_i*V_i_phys/gamma_cr_m1, sVj = scalar_j*V_j_phys/gamma_cr_m1; // physical units
            thold_hll = DMAX(DMIN(fabs(sVi), fabs(sVj)), DMIN(fabs(sVi-sVj), DMAX(fabs(sVi), fabs(sVj))));
            if(sign_c0 < 0) {thold_hll *= 0.001;} // if opposing signs, restrict this term //
            if(fabs(cmag)>thold_hll) {cmag *= thold_hll/fabs(cmag);}
            cmag /= dt_hydrostep;
            Fluxes.CosmicRayPressure[k_CRegy] = cmag; // physical, as it needs to be
        } // cmag != 0
        
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
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
        Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k_j_to_i] += flux_tmp[k_j_to_i] * CellP[j].CosmicRayAlfvenEnergyPred[k_CRegy][k_j_to_i];
        Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k_i_to_j] += flux_tmp[k_i_to_j] * local.CosmicRayAlfvenEnergy[k_CRegy][k_i_to_j];
        for(k=0;k<2;k++) {out.DtCosmicRayAlfvenEnergy[k_CRegy][k] += FluxCorrectionFactor_to_i * Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k];}
        if(j_is_active_for_fluxes) {for(k=0;k<2;k++) {CellP[j].DtCosmicRayAlfvenEnergy[k_CRegy][k] -= FluxCorrectionFactor_to_j * Fluxes.CosmicRayAlfvenEnergy[k_CRegy][k];}}
#endif
                
    } // close check that kappa and particle masses are positive
    // actually assign the fluxes //
    out.DtCosmicRayEnergy[k_CRegy] += FluxCorrectionFactor_to_i * Fluxes.CosmicRayPressure[k_CRegy];
    if(j_is_active_for_fluxes) {CellP[j].DtCosmicRayEnergy[k_CRegy] -= FluxCorrectionFactor_to_j * Fluxes.CosmicRayPressure[k_CRegy];}
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    double CR_number_to_energy_ratio = 0; // ratio of flux of CR number per unit flux of CR energy, follows whichever cell CRs are flowing 'out' of
    if(Fluxes.CosmicRayPressure[k_CRegy] > 0) {CR_number_to_energy_ratio =  CellP[j].Flux_Number_to_Energy_Correction_Factor[k_CRegy] * CellP[j].CosmicRay_Number_in_Bin[k_CRegy] / (CellP[j].CosmicRayEnergy[k_CRegy] + MIN_REAL_NUMBER);} else {CR_number_to_energy_ratio = local.CR_number_to_energy_ratio[k_CRegy];}
    //CR_number_to_energy_ratio *= 0.98; // = if want a simple approximation to the difference b/t diffusivities in energy vs number: multiply by 1 - 0.103021*epsilon, where epsilon = D_diffusion (0.5-ish) - gamma_slope where L_grad[R] ~ R^gamma_slope and kappa_diffusion ~ R^D_diffusion. reasonable approx is something like gamma_slope ~ (0.5-0.75)*D_diffusion [trends towards D_diffusion in eqm]
    out.DtCosmicRay_Number_in_Bin[k_CRegy] += FluxCorrectionFactor_to_i * Fluxes.CosmicRayPressure[k_CRegy] * CR_number_to_energy_ratio;
    if(j_is_active_for_fluxes) {CellP[j].DtCosmicRay_Number_in_Bin[k_CRegy] -= FluxCorrectionFactor_to_j * Fluxes.CosmicRayPressure[k_CRegy] * CR_number_to_energy_ratio;}
#endif
}


out.Face_DivVel_ForAdOps += -(All.cf_a3inv/V_i) * Face_Area_Norm * (Riemann_out.S_M + face_area_dot_vel) / All.cf_a2inv;
if(j_is_active_for_fluxes) {CellP[j].Face_DivVel_ForAdOps -= -(All.cf_a3inv/V_j) * Face_Area_Norm * (Riemann_out.S_M + face_area_dot_vel) / All.cf_a2inv;}


#if defined(CRFLUID_INJECTION_AT_SHOCKS)
/* use a simple pairwise shock-detection algorithm */
double min_shockvel_for_accel = 100., min_machnum_for_accel = 5., machnum_estimator = 0, min_machnum_for_pressurecrit = 1.3, saturation_fraction_for_craccel=(double)(CRFLUID_INJECTION_AT_SHOCKS);
double vdotf2_phys = face_vel_i - face_vel_j, vdotr2_phys = kernel.vdotr2 / (kernel.r * All.cf_atime);
if(All.ComovingIntegrationOn) {vdotr2_phys -= All.cf_hubble_a2 * kernel.r / All.cf_atime;}
if(vdotr2_phys < 0 && vdotf2_phys < 0 && (DMAX(vdotf2_phys, vdotr2_phys)*UNIT_VEL_IN_KMS > min_shockvel_for_accel)) { // particles must be approaching, with some normal to face, above some relative velocity
    if((local.InternalEnergyPred - CellP[j].InternalEnergyPred)*(local.Density - CellP[j].Density) > 0) { // sign of temperature and density jump must be consistent (help filtering contact discontinuities)
        if((local.Pressure - CellP[j].Pressure)*(local.InternalEnergyPred - CellP[j].InternalEnergyPred) > 0) { // consistent pressure and temperature jump
            double gamma_touse = 5./3., m2 = min_machnum_for_accel*min_machnum_for_accel, gplus = gamma_touse+1., gminus = gamma_touse-1., pjump_crit = (2.*gamma_touse*min_machnum_for_pressurecrit*min_machnum_for_pressurecrit - gminus)/gplus, tjump_crit = (2.*gamma_touse*m2 - gminus)*(gminus*m2 + 2.)/(gplus*gplus*m2); // various critical definitions
            double tjump = local.InternalEnergyPred/CellP[j].InternalEnergyPred, pjump = local.Pressure/CellP[j].Pressure; // temperature and pressure ratios used to estimate shock strength
            if(tjump < 1.) {tjump=1./tjump; pjump=1./pjump;} // make sure have correct upwind/downwind assignment
            if((tjump > tjump_crit) && (pjump > pjump_crit)) { // ok, sufficiently strong shock to justify some application of the scalings here, and passes pressure minimum check
                double m2_guess = (16./5.)*tjump, dissipation_fac = (0.5625 - 2.9607/m2_guess), upwind_density = DMIN(local.Density,CellP[j].Density), zeta_obliquity_fac = 1., velforflux = fabs(vdotf2_phys);
#ifdef MAGNETIC
                double cos_t=0,dvmag=0,bmag=0; for(k=0;k<3;k++) {double dv=local.Vel[k]-VelPred_j[k]; cos_t+=bhat[k]*dv; dvmag+=dv*dv; bmag+=bhat[k]*bhat[k];}
                if(dvmag>0 && bmag>0) {cos_t/=sqrt(dvmag*bmag);} else {cos_t=0;}
                double q = (1.-fabs(cos_t))/0.34; zeta_obliquity_fac = exp(-q*q*q);
#endif
                double DtCREgyNewInjection = saturation_fraction_for_craccel * zeta_obliquity_fac * dissipation_fac * 0.5 * upwind_density * velforflux*velforflux*velforflux * Face_Area_Norm;
                if(local.InternalEnergyPred > CellP[j].InternalEnergyPred) {out.DtCREgyNewInjectionFromShocks += FluxCorrectionFactor_to_i * DtCREgyNewInjection;} else {if(j_is_active_for_fluxes) {CellP[j].DtCREgyNewInjectionFromShocks += FluxCorrectionFactor_to_j * DtCREgyNewInjection;}} // do injection upstream
            }}}}
#endif

#endif // COSMIC_RAY_FLUID
