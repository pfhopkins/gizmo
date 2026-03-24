/* --------------------------------------------------------------------------------- */
/* ... non-ideal MHD term evaluation ...
 *
 * For SPH, this relys on the anisoptropic SPH second-derivative
 *  operator. So a large kernel is especially useful to minimize the systematic errors.
 *  For MFM/MFV methods, the consistent finite-volume formulation is used.
 *  In either case, since we solve the diffusion equations explicitly, a stronger timestep
 *  restriction is necessary (since the equations are not strictly hyperbolic); this is in timestep.c
 *
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 *
 * IMPORTANT the code below is not updated with full cosmological units included, as it has only been used for
 *   non-cosmological simulations. Be sure to check carefully to convert to physical units for any cosmological run[s].
 */
/* --------------------------------------------------------------------------------- */
Vec3<double> bflux_from_nonideal_effects={};
if((local.Mass > 0) && (P[j].Mass > 0) && (dt_hydrostep > 0))
{
    // set the effective scalar coefficients //
    double eta_i, eta_j, eta_ohmic, eta_hall, eta_ad;
    eta_i = local.Eta_MHD_OhmicResistivity_Coeff; eta_j = CellP[j].Eta_MHD_OhmicResistivity_Coeff;
    if(eta_i*eta_j>0) {eta_ohmic = 2*eta_i*eta_j / (eta_i+eta_j);} else {eta_ohmic = 0;}
    eta_i = local.Eta_MHD_HallEffect_Coeff; eta_j = CellP[j].Eta_MHD_HallEffect_Coeff;
    if(eta_i*eta_j>0) {eta_hall = 2*eta_i*eta_j / (eta_i+eta_j);} else {eta_hall = 0;}
    eta_i = local.Eta_MHD_AmbiPolarDiffusion_Coeff; eta_j = CellP[j].Eta_MHD_AmbiPolarDiffusion_Coeff;
    if(eta_i*eta_j>0) {eta_ad = 2*eta_i*eta_j / (eta_i+eta_j);} else {eta_ad = 0;}
    
    // only go further if these are non-zero //
    double eta_max = DMAX(fabs(eta_ohmic) , DMAX(fabs(eta_hall), fabs(eta_ad)));
    int k_xyz_A=0, k_xyz_B=0;
    if(eta_max > MIN_REAL_NUMBER)
    {
        // define the current J //
        Vec3<double> J_current, d_scalar; double rinv2 = rinv*rinv;
        Vec3<double> J_direct={}, grad_dot_x_ij={};
        for(k=0;k<3;k++) {d_scalar[k] = local.BPred[k] - BPred_j[k];}
        for(k=0;k<3;k++)
        {
            int k2;
            if(k==0) {k_xyz_A=2; k_xyz_B=1;}
            if(k==1) {k_xyz_A=0; k_xyz_B=2;}
            if(k==2) {k_xyz_A=1; k_xyz_B=0;}
            double tmp_grad_A = 0.5*(local.Gradients.B[k_xyz_A][k_xyz_B] + CellP[j].Gradients.B[k_xyz_A][k_xyz_B]); // construct averaged slopes //
            double tmp_grad_B = 0.5*(local.Gradients.B[k_xyz_B][k_xyz_A] + CellP[j].Gradients.B[k_xyz_B][k_xyz_A]);
            J_current[k] = tmp_grad_B - tmp_grad_A; // determine contribution to J //
            // calculate the 'direct' J needed for stabilizing numerical diffusion terms //
            J_direct[k] = rinv2*(kernel.dp[k_xyz_A]*d_scalar[k_xyz_B]-kernel.dp[k_xyz_B]*d_scalar[k_xyz_A]);
            if(J_current[k]*J_direct[k] < 0) {if(fabs(J_direct[k]) > 5.*fabs(J_current[k])) {J_current[k] = 0.0;}}
            for(k2=0;k2<3;k2++) {grad_dot_x_ij[k] += 0.5*(local.Gradients.B[k][k2]+CellP[j].Gradients.B[k][k2]) * kernel.dp[k2];}
        }
        double Jmag = J_current.norm_sq();
        
        // calculate the actual fluxes : need term = -[eta_O*(J) + eta_A*(-(Jxbhat)xbhat) + (-|eta_H|)*(Jxbhat)], assuming we're passed |eta| (=eta for eta_O,eta_A, but =-eta for eta_H) //
        Vec3<double> b_flux={};
        Vec3<double> bhat_v = {bhat[0], bhat[1], bhat[2]};
        Vec3<double> JcrossB = cross(J_current, bhat_v); // this is Jxbhat
        Vec3<double> JcrossBcrossB = cross(JcrossB, bhat_v); // this is (Jxbhat)x(bhat)
        if(fabs(eta_ohmic)>0) {b_flux += -eta_ohmic * J_current;} // ohmic ~ J
        if(fabs(eta_ad)>0) {b_flux += eta_ad * JcrossBcrossB;} // a.d. ~ (JxB)xB
        if(fabs(eta_hall)>0) {b_flux += -eta_hall * JcrossB;} // hall ~ (JxB)
        
        // calculate dB/dt = Area.cross.flux //
        bflux_from_nonideal_effects = cross(Face_Area_Vec, b_flux);
        
        // ok now construct the numerical fluxes based on the numerical diffusion coefficients and the direct-difference values between elements
        double eta_0 = v_hll * kernel.r * All.cf_atime; // standard numerical diffusivity needed for stabilizing fluxes
        double eta_ohmic_0=0,eta_ad_0=0,eta_hall_0=0,q=0; // now limit the numerical diffusivity to avoid unphysically fast diffusion
        if(fabs(eta_ohmic)>0) {q=eta_0/fabs(eta_ohmic); eta_ohmic_0=eta_0*(0.2 + q)/(0.2 + q + q*q);}
        if(fabs(eta_ad)>0) {q=eta_0/fabs(eta_ad); eta_ad_0=eta_0*(0.2 + q)/(0.2 + q + q*q);}
        if(fabs(eta_hall)>0) {q=eta_0/fabs(eta_hall); eta_hall_0=eta_0*(0.2 + q)/(0.2 + q + q*q); if(eta_hall<0) {eta_hall_0*=-1;}} // since signed, less clear if numerical term should be signed also, or mono-sign here //
        Vec3<double> b_flux_direct={}, db_direct;
        Vec3<double> JcrossB_direct = cross(J_direct, bhat_v);
        Vec3<double> JcrossBcrossB_direct = cross(JcrossB_direct, bhat_v);
        if(fabs(eta_ohmic_0)>0) {b_flux_direct += -eta_ohmic_0 * J_direct;}
        if(fabs(eta_ad_0)>0) {b_flux_direct += eta_ad_0 * JcrossBcrossB_direct;}
        if(fabs(eta_hall_0)>0) {b_flux_direct += -eta_hall_0 * JcrossB_direct;}
        db_direct = cross(Face_Area_Vec, b_flux_direct);

        double db_dot_direct_diff=0, F_ddiff_prefac = -Face_Area_Norm * rinv * DMAX(eta_ohmic, fabs(eta_hall)+eta_ad);
        Vec3<double> db_direct_diff = F_ddiff_prefac * d_scalar; // direct differential for face pointing along dp //
        
        double bfluxmag = bflux_from_nonideal_effects.norm_sq();
        bfluxmag /= (1.e-37 + Jmag * eta_max*eta_max * Face_Area_Norm*Face_Area_Norm);
        
        for(k=0;k<3;k++)
        {
            double d_scalar_tmp = d_scalar[k] - grad_dot_x_ij[k];
            double d_scalar_hll = MINMOD(d_scalar[k] , d_scalar_tmp);
            double hll_corr = bfluxmag * HLL_correction(d_scalar_hll,0.,1.,eta_max);
            double db_corr = bflux_from_nonideal_effects[k] + hll_corr;
            bflux_from_nonideal_effects[k] = MINMOD(1.1*bflux_from_nonideal_effects[k], db_corr);
            if(hll_corr!=0)
            {
                double db_direct_tmp = fabs(db_direct[k]) * hll_corr / fabs(hll_corr);
                if((bflux_from_nonideal_effects[k]*db_direct_tmp < 0) && (fabs(db_direct_tmp) > 1.0*fabs(bflux_from_nonideal_effects[k]))) {bflux_from_nonideal_effects[k]=0;}
            }
            if((bflux_from_nonideal_effects[k]*db_direct[k] < 0) && (fabs(db_direct[k]) > 10.0*fabs(bflux_from_nonideal_effects[k]))) {bflux_from_nonideal_effects[k]=0;}
            
            db_dot_direct_diff += bflux_from_nonideal_effects[k] * db_direct_diff[k];
        }

        // check if projection along direct diffusion direction produces explicitly anti-diffusive behavior, and if so, subtract that term from the flux update //
        if(db_dot_direct_diff < 0) {
            double db_direct_diff_mag2 = db_direct_diff.norm_sq();
            bflux_from_nonideal_effects -= (db_dot_direct_diff / db_direct_diff_mag2) * db_direct_diff;
        }
        
        // -now- we can finally add this to the numerical fluxes //
        Fluxes.B += bflux_from_nonideal_effects;
    }
}
