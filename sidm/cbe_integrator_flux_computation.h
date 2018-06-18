/* ---------------------------------------------------------------------------------
 this is the kernel where the core of the CBE flux computation is performed,
 to calculate the relevant fluxes between faces of partciels, within the gravity routine
 --------------------------------------------------------------------------------- */
{
    int m; double V_i=local.V_i, V_j=get_particle_volume_ags(j), rho_i=local.Mass/V_i*All.cf_a3inv, rho_j=P[j].Mass/V_j*All.cf_a3inv, Face_Area_Vec[3], Face_Area_Norm=0, psi_i, psi_j, vf0_dot_dp=0, vface_guess[3]; // calculate densities (in physical units)
    psi_i=1./(1. + kernel.h_i/kernel.h_j); psi_i=0.5; psi_j=1-psi_i; rho_i*=psi_i; rho_j*=psi_j;
    
    // calculate effective faces and face velocity between elements //
    for(k=0;k<3;k++)
    {
        Face_Area_Vec[k] = -(kernel.wk_i*V_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2]) +
                             kernel.wk_j*V_j * (P[j].NV_T[k][0]*kernel.dp[0] + P[j].NV_T[k][1]*kernel.dp[1] + P[j].NV_T[k][2]*kernel.dp[2])) * All.cf_atime*All.cf_atime; // physical units
        Face_Area_Norm += Face_Area_Vec[k]*Face_Area_Vec[k]; // physical units
        vface_guess[k] = 0.5*(local.Vel[k] + P[j].Vel[k]) / All.cf_atime; // physical units
        vf0_dot_dp += vface_guess[k] * kernel.dp[k];
    }
    Face_Area_Norm = sqrt(Face_Area_Norm);
    
    
    double local_CBE_basis_moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double Pj_CBE_basis_moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    
    for(m=0;m<CBE_INTEGRATOR_NBASIS;m++)
    {
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
        {
            local_CBE_basis_moments[m][k] = local.CBE_basis_moments[m][k];
            Pj_CBE_basis_moments[m][k] = P[j].CBE_basis_moments[m][k];
            if((k>0)&&(k<4))
            {
                local_CBE_basis_moments[m][k] += local_CBE_basis_moments[m][0]*local.Vel[k-1]/All.cf_atime; // physical
                Pj_CBE_basis_moments[m][k] += Pj_CBE_basis_moments[m][0]*P[j].Vel[k-1]/All.cf_atime; // physical
            }
        }
    }
    
    double vface_new[3]={0};
    double theta_i[CBE_INTEGRATOR_NBASIS]={0}, theta_j[CBE_INTEGRATOR_NBASIS]={0}, v_wt_sum=0;
    for(m=0;m<CBE_INTEGRATOR_NBASIS;m++)
    {
        double vi_dot_dp=0, vj_dot_dp=0;
        for(k=0;k<3;k++)
        {
            vi_dot_dp += (local_CBE_basis_moments[m][k+1]/local_CBE_basis_moments[m][0] - vface_guess[k])*kernel.dp[k];
            vj_dot_dp += (Pj_CBE_basis_moments[m][k+1]/Pj_CBE_basis_moments[m][0]  - vface_guess[k])*kernel.dp[k];
        }
        if(vi_dot_dp < 0) {theta_i[m]=1;} // approaching interaction face
        if(vj_dot_dp > 0) {theta_j[m]=1;} // approaching interaction face
        double w0_i = theta_i[m] * rho_i / local.Mass, w0_j = theta_j[m] * rho_j / P[j].Mass;
        v_wt_sum += w0_i*local_CBE_basis_moments[m][0] + w0_j*Pj_CBE_basis_moments[m][0]; // summed weights for interaction
        for(k=0;k<3;k++) {vface_new[k] += w0_i*local_CBE_basis_moments[m][k+1] + w0_j*Pj_CBE_basis_moments[m][k+1];} // summed velocities
    }
    
    // OK, now we've done the easy bit -- we're ready for the actual computations //
    double vface[3] = {0}, fMfac = 0;
    if((v_wt_sum > MIN_REAL_NUMBER) && (v_wt_sum < MAX_REAL_NUMBER))
    {
        for(k=0;k<3;k++) {vface[k] = vface_new[k] / v_wt_sum;} // ensures net mass flux = 0, after accounting for theta_ij weights
        
        // first loop over pairs to determine closest a-to-b, closest b-to-a //
        int matching_basis_j_for_basis_in_i[CBE_INTEGRATOR_NBASIS], matching_basis_i_for_basis_in_j[CBE_INTEGRATOR_NBASIS], m_j;
        double wt_i[CBE_INTEGRATOR_NBASIS], wt_j[CBE_INTEGRATOR_NBASIS], imag_i[CBE_INTEGRATOR_NBASIS], imag_j[CBE_INTEGRATOR_NBASIS], cos_ij, vsig=0;
        for(m=0;m<CBE_INTEGRATOR_NBASIS;m++) // first pass to normalize and initialize quantities for both sides
        {
            double norm_tmp,q0; wt_i[m] = -1000.; wt_j[m] = -1000.; // large negative value (<-1)
            norm_tmp=0; for(k=0;k<3;k++) {q0=local_CBE_basis_moments[m][k+1]-fMfac*vface[k]*local_CBE_basis_moments[m][0]; norm_tmp += q0*q0;} // squared weight of momentum (vectors we'll use for matching below //
            if(norm_tmp > 0) {norm_tmp = 1./sqrt(norm_tmp);} // compute inverse-weight, for use below
            imag_i[m] = norm_tmp; // assign
            norm_tmp=0; for(k=0;k<3;k++) {q0=Pj_CBE_basis_moments[m][k+1]-fMfac*vface[k]*Pj_CBE_basis_moments[m][0]; norm_tmp += q0*q0;} // squared weight of momentum (vectors we'll use for matching below //
            if(norm_tmp > 0) {norm_tmp = 1./sqrt(norm_tmp);} // compute inverse-weight, for use below
            imag_j[m] = norm_tmp; // assign
        }
        for(m=0;m<CBE_INTEGRATOR_NBASIS;m++)
        {
            for(m_j=0;m_j<CBE_INTEGRATOR_NBASIS;m_j++)
            {
                cos_ij=0;
                for(k=0;k<3;k++)
                {
                    double q1 = local_CBE_basis_moments[m][k+1] - fMfac*vface[k]*local_CBE_basis_moments[m][0];
                    double q2 = Pj_CBE_basis_moments[m_j][k+1] - fMfac*vface[k]*Pj_CBE_basis_moments[m][0];
                    cos_ij += q1*q2;
                } // product of vector momenta
                cos_ij *= imag_i[m] * imag_j[m_j]; // ok this is now the dot product p_hat_i_alpha . p_hat_j_beta
                if(cos_ij > wt_i[m]) // better match found for i
                {
                    wt_i[m] = cos_ij; // note the best match so far
                    matching_basis_j_for_basis_in_i[m] = m_j; // assign the basis function to the list
                }
                if(cos_ij > wt_j[m_j]) // better match found for j
                {
                    wt_j[m_j] = cos_ij; // note the best match so far
                    matching_basis_i_for_basis_in_j[m_j] = m; // assign the basis function to the list
                }
            }
        } // for(m=0;m<CBE_INTEGRATOR_NBASIS;m++)
        
        // now loop over each basis for each particle and actually compute fluxes //
        double wt_prefac_i = -rho_i / local.Mass; // normalized the fluxes, no need to re-compute below
        double wt_prefac_j = -rho_j / P[j].Mass;
        double vface_dot_A = vface[0]*Face_Area_Vec[0] + vface[1]*Face_Area_Vec[1] + vface[2]*Face_Area_Vec[2]; // v_face . A_face
        for(m=0;m<CBE_INTEGRATOR_NBASIS;m++)
        {
            int j_m = matching_basis_j_for_basis_in_i[m], i_m = matching_basis_i_for_basis_in_j[m]; // id matched bases for fluxes below //
            //j_m=m; i_m=m;
            // fluxes from "i" side
            double flux[CBE_INTEGRATOR_NMOMENTS]={0}, vsig_i=0, vsig_j=0;
            if(theta_i[m] == 1)
            {
                vsig_i = do_cbe_flux_computation(local_CBE_basis_moments[m] , vface_dot_A, vface, Face_Area_Vec, Pj_CBE_basis_moments[j_m], flux); // moments are physical, these are as well //
                for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
                {
                    flux[k] *= wt_prefac_i; // normalize appropriately
                    out.CBE_basis_moments_dt[m][k] += flux[k]; // flux out of "i"
                    //if(TimeBinActive[P[j].TimeBin]) {P[j].CBE_basis_moments_dt[j_m][k] -= flux[k];} // flux into "j" (if j is active)
                }
            }
            // fluxes from "j" side
            if(theta_j[m] == 1)
            {
                vsig_j = do_cbe_flux_computation(Pj_CBE_basis_moments[m] , vface_dot_A, vface, Face_Area_Vec, local_CBE_basis_moments[i_m], flux); // moments are physical, these are as well //
                for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
                {
                    flux[k] *= wt_prefac_j; // normalize appropriately
                    out.CBE_basis_moments_dt[i_m][k] += flux[k]; // flux out of "i"
                    //if(TimeBinActive[P[j].TimeBin]) {P[j].CBE_basis_moments_dt[m][k] -= flux[k];} // flux into "j" (if j is active)
                } // normalize appropriately
            }
            vsig = DMAX(DMAX(fabs(vsig_i),fabs(vsig_j)),vsig);
        } // for(m=0;m<CBE_INTEGRATOR_NBASIS;m++)
        vsig /= Face_Area_Norm * All.cf_afac3 * All.cf_atime; // into appropriate sound-speed units
        if(vsig > out.AGS_vsig) {out.AGS_vsig = vsig;} // set signal velocity if new value found
        //if(TimeBinActive[P[j].TimeBin]) {if(vsig > PPP[j].AGS_vsig) PPP[j].AGS_vsig = vsig;}
#ifdef WAKEUP
        if(!(TimeBinActive[P[j].TimeBin]) && (All.Time > All.TimeBegin)) {if(vsig > WAKEUP*PPP[j].AGS_vsig) {P[j].wakeup = 1;}}
#endif
    } // v_wt_sum > 0
} // master bracket (for variable protection

