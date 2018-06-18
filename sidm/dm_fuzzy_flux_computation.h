/* quantum pressure-tensor computation to calculate the relevant fluxes between faces of particles, within the gravity routine */
if((local.Type==1) && (P[j].Type==1)) // only acts between DM particles of type 1 (can easily change if desired)
{
    /* since this isn't a super-expensive calculation, and we need to walk the gravity tree for both 'sides' anyways, we will effectively do this twice each timestep */
    double V_i=local.V_i, V_j=get_particle_volume_ags(j), wt_i=V_i, wt_j=V_j, rho_i=local.Mass/V_i*All.cf_a3inv, rho_j=P[j].Mass/V_j*All.cf_a3inv, Face_Area_Vec[3], Face_Area_Norm=0, vface_i_minus_j=0, dv[3], flux[3]={0}; // calculate densities (in physical units)
    // calculate effective faces and face velocity between elements //
    if((fabs(V_i-V_j)/DMIN(V_i,V_j))/NUMDIMS > 1.25) {wt_i=wt_j=2.*V_i*V_j/(V_i+V_j);} else {wt_i=V_i; wt_j=V_j;} // limiter to prevent faces from going geometrically crazy in disparate limits
    for(k=0;k<3;k++)
    {
        Face_Area_Vec[k] = (kernel.wk_i*wt_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2]) +
                            kernel.wk_j*wt_j * (P[j].NV_T[k][0]*kernel.dp[0] + P[j].NV_T[k][1]*kernel.dp[1] + P[j].NV_T[k][2]*kernel.dp[2])) * All.cf_atime*All.cf_atime; // physical units
        Face_Area_Norm += Face_Area_Vec[k]*Face_Area_Vec[k]; // physical units
        dv[k] = kernel.dv[k] / All.cf_atime; // physical units
        vface_i_minus_j += dv[k] * Face_Area_Vec[k]; // physical units
    }
    Face_Area_Norm = sqrt(Face_Area_Norm); vface_i_minus_j /= Face_Area_Norm;
    // convert everything needed below into physical units //
    double igrad[3], jgrad[3], i2grad[3][3], j2grad[3][3], fac_g = All.cf_a3inv/All.cf_atime, fac_g2 = All.cf_a3inv*All.cf_a2inv, dp[3]; int m;
    for(k=0;k<3;k++)
    {
        dp[k] = kernel.dp[k] * All.cf_atime;
        igrad[k] = fac_g * local.AGS_Gradients_Density[k];
        jgrad[k] = fac_g * P[j].AGS_Gradients_Density[k];
        for(m=0;m<3;m++)
        {
            i2grad[k][m] = fac_g2 * local.AGS_Gradients2_Density[k][m];
            j2grad[k][m] = fac_g2 * P[j].AGS_Gradients2_Density[k][m];
        }
    }
    
    double dt = local.dt_step * All.Timebase_interval/All.cf_hubble_a, m_mean = 0.5*(local.Mass+P[j].Mass), prev_acc = All.G*All.cf_a2inv * P[j].Mass * P[j].OldAcc, AGS_Numerical_QuantumPotential = 0.5*(local.AGS_Numerical_QuantumPotential/V_i + P[j].AGS_Numerical_QuantumPotential/V_j)*All.cf_a3inv, dt_egy_Numerical_QuantumPotential=0;
    double HLLwt = (0.5*(kernel.wk_i/kernel.hinv3_i + kernel.wk_j/kernel.hinv3_j)) * (0.5*(kernel.h_i+kernel.h_j)/kernel.r); HLLwt = 10.*HLLwt*HLLwt; // strong dissipation terms allowed for very-close particles, where second-derivative diverges, otherwise weak (no diffusion) //
    // actually compute the fluxes now, this is the key routine, below //
    do_dm_fuzzy_flux_computation(HLLwt, dt, m_mean, prev_acc, dp, dv, jgrad,  igrad, j2grad, i2grad, rho_j, rho_i, vface_i_minus_j, Face_Area_Vec, flux, AGS_Numerical_QuantumPotential, &dt_egy_Numerical_QuantumPotential);
    out.AGS_Dt_Numerical_QuantumPotential += dt_egy_Numerical_QuantumPotential; for(k=0;k<3;k++) {out.acc[k] += flux[k] / (local.Mass * All.cf_a2inv);} // assign back to particles
} // master bracket (for variable protection)
