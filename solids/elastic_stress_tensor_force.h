/* --------------------------------------------------------------------------------- */
/* ... elastic stress evaluation. currently a first-order solver. HLL artificial diffusivity
 *      terms not included here because with current formulation what is in the Reimann solver
 *      should already be sufficient, up to tensile check included. But this needs to be
 *      directly tested.
 *
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
{
    if( ((local.Mass>0)&&(P[j].Mass>0)) )
    {
        int k_v, j_v;
        double FNormT = Face_Area_Norm; // use the already-computed face area (MFM: gradient-corrected via NV_T; SPH: standard kernel derivative formulation)
        double FVec[3]={0}; for(k=0;k<3;k++) {FVec[k] = Face_Area_Vec[k];}

        double cmag[3]={0}, v_interface[3], wt_i=-0.5, wt_j=-0.5; // we need a minus sign at some point; its handy to just include it now in the weights //
        for(k=0;k<3;k++) {v_interface[k] = (wt_i*local.Vel[k] + wt_j*VelPred_j[k]) / All.cf_atime;} // physical units //
        double rho_i = local.Density * All.cf_a3inv, rho_j = CellP[j].Density * All.cf_a3inv;

        // terms for HLL fluxes
        double wt_r = rho_i*local.SoundSpeed * rho_j*CellP[j].SoundSpeed / (rho_i*local.SoundSpeed + rho_j*CellP[j].SoundSpeed) * FNormT; // physical
        double cT_i=sqrt(All.Tillotson_EOS_params[local.CompositionType][10]/rho_i), cT_j=sqrt(All.Tillotson_EOS_params[CellP[j].CompositionType][10]/rho_j); // physical
        double wt_t = rho_i*cT_i * rho_j*cT_j / (rho_i*cT_i + rho_j*cT_j) * FNormT; // physical
        double wt_rt = (wt_r - wt_t) * kernel.vdotr2 * rinv*rinv*All.cf_atime*All.cf_atime;
        // evaluate force from deviatoric stress tensor //
        // fancier model where tensile correction is always applied to compressive principle component of the stress tensor [have to solve for eigenvalues of the stress tensor]
        int ij_switch; for(ij_switch=0;ij_switch<2;ij_switch++) {
            double nvt[NUMDIMS*NUMDIMS]={0}, norm_m=0, wtfac=0; if(ij_switch==0) {wtfac=wt_i;} else {wtfac=wt_j;}
            for(k_v=0;k_v<NUMDIMS;k_v++) {for(j_v=0;j_v<NUMDIMS;j_v++) {
            if(ij_switch==0) {nvt[NUMDIMS*k_v + j_v] = local.Elastic_Stress_Tensor[k_v][j_v];} else {nvt[NUMDIMS*k_v + j_v] = CellP[j].Elastic_Stress_Tensor_Pred[k_v][j_v];}
            norm_m += nvt[NUMDIMS*k_v + j_v]*nvt[NUMDIMS*k_v + j_v];}} // initialize auxiliary array to store for feeding to GSL eigen routine
            if(norm_m > MIN_REAL_NUMBER) {
                gsl_matrix_view M = gsl_matrix_view_array(nvt,NUMDIMS,NUMDIMS); gsl_vector *eigvals = gsl_vector_alloc(NUMDIMS); gsl_matrix *eigvecs = gsl_matrix_alloc(NUMDIMS,NUMDIMS);
                gsl_eigen_symmv_workspace *v = gsl_eigen_symmv_alloc(NUMDIMS); gsl_eigen_symmv(&M.matrix, eigvals, eigvecs, v);
                double eigenval_k=0, eigenvec_k[NUMDIMS]={0}, A_dot_v=0, prefac=0;
                for(k_v=0;k_v<NUMDIMS;k_v++) {eigenval_k = gsl_vector_get(eigvals, k_v); A_dot_v = 0;
                    for(j_v=0;j_v<NUMDIMS;j_v++) {eigenvec_k[j_v] = gsl_matrix_get(eigvecs, j_v, k_v); A_dot_v += FVec[j_v]*eigenvec_k[j_v];}
                    prefac = wtfac * eigenval_k * (A_dot_v*All.cf_a2inv); 
#if !defined(HYDRO_MESHLESS_FINITE_VOLUME) && !defined(HYDRO_MESHLESS_FINITE_MASS)
                    if(eigenval_k > 0) {prefac *= 1. - tensile_correction_factor;} /* for SPH, we want to apply the tensile correction to all positive eigenvalues */
#endif
                    if(!isnan(eigenval_k)) {for(j_v=0;j_v<NUMDIMS;j_v++) {cmag[j_v] += prefac * eigenvec_k[j_v];}} // evaluate S.Face = S.[sum of eigenvalues times projection on Face onto each eigenvector]
                }
                gsl_eigen_symmv_free(v); gsl_vector_free(eigvals); gsl_matrix_free(eigvecs);} // free memory
        }
        for(j_v=0;j_v<3;j_v++) {cmag[j_v] -= wt_rt * kernel.dp[j_v]*All.cf_atime + wt_t * kernel.dv[j_v]/All.cf_atime;} // HLL-type fluxes

        /* zero-energy mode (hourglass) control: damp velocity residuals not captured by the local velocity gradient.
            inspired by Ganzenmuller 2015 and Kugelstadt et al. 2021, adapted to rate formulation.
            the velocity difference predicted by the gradient is subtracted, leaving only the sub-grid residual;
            this vanishes identically for any linear velocity field, so it only targets hourglass-type modes. */
        {
            double dv_pred_i, dv_pred_j, dv_err;
            double cT_hg = 0.5 * wt_t / All.cf_atime; // hourglass damping coefficient: 0.5 * impedance-weighted shear-wave factor (same scaling as HLL transverse term)
            for(j_v=0;j_v<3;j_v++) {
                dv_pred_i = 0; dv_pred_j = 0;
                for(k_v=0;k_v<3;k_v++) {
                    dv_pred_i += local.Gradients.Velocity[j_v][k_v] * kernel.dp[k_v]; // (nabla v)_i . (x_i - x_j)
                    dv_pred_j += CellP[j].Gradients.Velocity[j_v][k_v] * kernel.dp[k_v]; // (nabla v)_j . (x_i - x_j)
                }
                dv_err = kernel.dv[j_v] - 0.5*(dv_pred_i + dv_pred_j); // velocity residual (zero for linear fields)
                cmag[j_v] -= cT_hg * dv_err;
            }
        }

        /* // below limiter doesn't actually appear necessary in our tests, thus far; worth more detailed testing the future //
        if(kernel.vdotr2 < 0) // check if particles are approaching
        {
            double astress_dot_r=0, a_dot_r=0; for(k_v=0;k_v<3;k_v++) {astress_dot_r += cmag[k_v]*kernel.dp[k_v]; a_dot_r += Fluxes.v[k_v]*kernel.dp[k_v];}
            if((astress_dot_r < 0) && (a_dot_r+astress_dot_r < 0)) // check if this makes them approach faster (and if -net- term is also making them approach faster)
            {
                double cmag_dir[3]={0}, a_dot_r_alt=0; // check if the face area orientation is the difference
                for(j_v=0;j_v<3;j_v++)
                {
                    for(k_v=0;k_v<3;k_v++) {cmag_dir[j_v] += (wt_i*local.Elastic_Stress_Tensor[j_v][k_v] + wt_j*CellP[j].Elastic_Stress_Tensor[j_v][k_v]) * FNormT*kernel.dp[k_v]/kernel.r;}
                    a_dot_r_alt += kernel.dp[j_v]*cmag[j_v];
                }
                if((a_dot_r_alt >= 0)||(a_dot_r_alt+astress_dot_r>=0)) {for(k_v=0;k_v<3;k_v++) {cmag[k_v]=cmag_dir[k_v];}} else // if this solves it, use this, done!
                {
                    double h_eff = 0.5*(Particle_Size_i + Particle_Size_j); // effective inter-particle spacing around these elements
                    double corrfac = kernel.r/h_eff; corrfac=corrfac*corrfac; corrfac=corrfac/(1+corrfac); // factor which interpolates between 1 if r>>mean spacing, 0 if r<<mean spacing
                    for(k_v=0;k_v<3;k_v++) {cmag[k_v] *= corrfac;} // applies 'artificial stress' term which behaves similarly to Monaghan 2000
                    a_dot_r *= corrfac;
                }
            }
        }
        */
        
        double cmag_E = cmag[0]*v_interface[0] + cmag[1]*v_interface[1] + cmag[2]*v_interface[2];
        /* now add a flux-limiter to prevent overshoot (even when the directions are correct) */
        /*
        if(dt_hydrostep > 0)
        {
            double cmag_lim = 0.25 * (local.Mass + P[j].Mass) * (v_interface[0]*v_interface[0]+v_interface[1]*v_interface[1]+v_interface[2]*v_interface[2]) / dt_hydrostep;
            if(fabs(cmag_E) > cmag_lim)
            {
                double corr_visc = cmag_lim / fabs(cmag_E);
                cmag[0]*=corr_visc; cmag[1]*=corr_visc; cmag[2]*=corr_visc; cmag_E*=corr_visc;
            }
        }
        */
        /* ok now we can finally add this to the numerical fluxes */
        for(k=0;k<3;k++) {Fluxes.v[k] += cmag[k];}
        Fluxes.p += cmag_E;
    } // close check that kappa and particle masses are positive
}
