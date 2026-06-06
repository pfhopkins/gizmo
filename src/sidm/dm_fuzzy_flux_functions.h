/* dm_fuzzy_flux_functions.h -- per-pair quantum-pressure-tensor flux
 * for DM_FUZZY (fuzzy dark matter) within the adaptive-gravity AGSForce loop.
 *
 * Replaces the fragment sidm/dm_fuzzy_flux_computation.h. Body guarded by
 * DM_FUZZY so the caller (ags_rkern.cc AGSForce_evaluate) can invoke
 * unconditionally. Pure i-accumulation into the caller out struct.
 *
 * Templated on LocalT/KernelT/OutT so the AGSForce xchange-scaffolding
 * structs (unnamed INPUT_STRUCT_NAME / OUTPUT_STRUCT_NAME) plug in directly.
 *
 * Requires allvars.h / proto.h / kernel.h / sidm/dm_fuzzy.h forward decls
 * (for do_dm_fuzzy_flux_computation{,_old} and dm_fuzzy_reconstruct_and_slopelimit).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DM_FUZZY_FLUX_FUNCTIONS_H
#define DM_FUZZY_FLUX_FUNCTIONS_H

#include "../gravity/ags_functions.h"
#include "dm_fuzzy_functions.h"

#ifndef DM_FUZZY_USE_SIMPLER_HLL_SOLVER
#define DM_FUZZY_USE_SIMPLER_HLL_SOLVER 0    /* which HLL solver for DM_FUZZY=0: =1 simpler & more diffusive */
#endif

template <typename LocalT, typename KernelT, typename OutT>
KOKKOS_INLINE_FUNCTION
void dm_fuzzy_flux_compute_pair(
    const LocalT &local,
    int j,
    struct particle_data *P,
    const KernelT &kernel,
    OutT &out)
{
#ifdef DM_FUZZY
    if(!((local.Type == 1) && (P[j].Type == 1))) { return; }

    /* effective face geometry and velocity */
    double V_i = local.V_i;
    double V_j = get_particle_volume_ags_P(j, P);
    double wt_i = V_i, wt_j = V_j;
    double Face_Area_Norm = 0, vface_i_minus_j = 0;
    Vec3<double> Face_Area_Vec = {0,0,0};
    Vec3<double> dv = {0,0,0};
    if((fabs(V_i - V_j) / DMIN(V_i, V_j)) / NUMDIMS > 1.25) {
        wt_i = wt_j = 2. * V_i * V_j / (V_i + V_j);
    }
    for(int k=0; k<3; k++) {
        Face_Area_Vec[k] = (kernel.wk_i * wt_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2])
                          + kernel.wk_j * wt_j * (P[j].NV_T[k][0]*kernel.dp[0] + P[j].NV_T[k][1]*kernel.dp[1] + P[j].NV_T[k][2]*kernel.dp[2])) * All.cf_atime * All.cf_atime;
        Face_Area_Norm += Face_Area_Vec[k] * Face_Area_Vec[k];
        dv[k] = kernel.dv[k] / All.cf_atime;
        vface_i_minus_j += dv[k] * Face_Area_Vec[k];
    }
    Face_Area_Norm = sqrt(Face_Area_Norm);
    vface_i_minus_j /= Face_Area_Norm;

    double fac_g  = All.cf_a3inv / All.cf_atime;
    double fac_g2 = All.cf_a3inv * All.cf_a2inv;
    double igrad[3], jgrad[3], i2grad[3][3], j2grad[3][3];
    Vec3<double> dp = {0,0,0};
    for(int k=0; k<3; k++) {
        dp[k] = kernel.dp[k] * All.cf_atime;
        igrad[k] = fac_g * local.AGS_Gradients_Density[k];
        jgrad[k] = fac_g * P[j].AGS_Gradients_Density[k];
        for(int m=0; m<3; m++) {
            i2grad[k][m] = fac_g2 * local.AGS_Gradients2_Density[k][m];
            j2grad[k][m] = fac_g2 * P[j].AGS_Gradients2_Density[k][m];
        }
    }

#if (DM_FUZZY == 0)
    double prev_acc = All.G * All.cf_a2inv * P[j].Mass * P[j].OldAcc;
    double dt_egy_Numerical_QuantumPotential = 0;
    double m_mean = 0.5 * (local.Mass + P[j].Mass);
    double rho_i  = local.Mass / V_i * All.cf_a3inv;
    double rho_j  = P[j].Mass / V_j * All.cf_a3inv;
    double dt = local.dtime;
    double AGS_Numerical_QuantumPotential = 0.5 * (local.AGS_Numerical_QuantumPotential / V_i
                                                   + P[j].AGS_Numerical_QuantumPotential / V_j) * All.cf_a3inv;
    Vec3<double> flux = {0,0,0};
    double HLLwt = (0.5 * (kernel.wk_i / kernel.hinv3_i + kernel.wk_j / kernel.hinv3_j))
                 * (0.5 * (kernel.h_i + kernel.h_j) / kernel.r);
    HLLwt = 10. * HLLwt * HLLwt;

#if (DM_FUZZY_USE_SIMPLER_HLL_SOLVER == 0)
    do_dm_fuzzy_flux_computation_old(HLLwt, dt, m_mean, prev_acc, dp, dv, jgrad, igrad, j2grad, i2grad,
                                     rho_j, rho_i, vface_i_minus_j, Face_Area_Vec, flux,
                                     AGS_Numerical_QuantumPotential, &dt_egy_Numerical_QuantumPotential);
#else
    do_dm_fuzzy_flux_computation(HLLwt, dt, prev_acc, dv, jgrad, igrad, j2grad, i2grad,
                                 rho_j, rho_i, vface_i_minus_j, Face_Area_Vec, flux,
                                 P[j].AGS_Numerical_QuantumPotential / V_j * All.cf_a3inv,
                                 local.AGS_Numerical_QuantumPotential / V_i * All.cf_a3inv,
                                 &dt_egy_Numerical_QuantumPotential);
#endif
    out.AGS_Dt_Numerical_QuantumPotential += dt_egy_Numerical_QuantumPotential;
    for(int k=0; k<3; k++) { out.acc[k] += flux[k] / (local.Mass * All.cf_a2inv); }

#else  /* DM_FUZZY > 0: scalar-field formulation */
    double h_2m = 0.5 * All.ScalarField_hbar_over_mass;
    /* dm_fuzzy_reconstruct_and_slopelimit leaves the du_R/du_L outputs
       untouched (see behaviour note in dm_fuzzy_functions.h). Zero here
       so GPU and CPU agree deterministically. */
    double Psi_Re_R = 0, Psi_Re_L = 0, d_Psi_Re_R[3] = {0,0,0}, d_Psi_Re_L[3] = {0,0,0}, v_face[3] = {0,0,0};
    double Psi_Im_R = 0, Psi_Im_L = 0, d_Psi_Im_R[3] = {0,0,0}, d_Psi_Im_L[3] = {0,0,0};
    double Flux_Re = 0, Flux_Im = 0, Flux_M = 0;

    for(int k=0; k<3; k++) { v_face[k] = 0.5 * (local.Vel[k] + P[j].Vel[k]) / All.cf_atime; }

    dm_fuzzy_reconstruct_and_slopelimit(&Psi_Re_R, d_Psi_Re_R, &Psi_Re_L, d_Psi_Re_L,
                                        local.AGS_Psi_Re, local.AGS_Gradients_Psi_Re, local.AGS_Gradients2_Psi_Re,
                                        P[j].AGS_Psi_Re_Pred * P[j].AGS_Density / P[j].Mass,
                                        P[j].AGS_Gradients_Psi_Re, P[j].AGS_Gradients2_Psi_Re, dp);

    dm_fuzzy_reconstruct_and_slopelimit(&Psi_Im_R, d_Psi_Im_R, &Psi_Im_L, d_Psi_Im_L,
                                        local.AGS_Psi_Im, local.AGS_Gradients_Psi_Im, local.AGS_Gradients2_Psi_Im,
                                        P[j].AGS_Psi_Im_Pred * P[j].AGS_Density / P[j].Mass,
                                        P[j].AGS_Gradients_Psi_Im, P[j].AGS_Gradients2_Psi_Im, dp);

    double psi2_L = Psi_Re_L*Psi_Re_L + Psi_Im_L*Psi_Im_L;
    double psi2_R = Psi_Re_R*Psi_Re_R + Psi_Im_R*Psi_Im_R;

    for(int k=0; k<3; k++) {
        Flux_Re += 0.5 * Face_Area_Vec[k] * ( -h_2m*(d_Psi_Im_R[k] + d_Psi_Im_L[k]) + v_face[k]*(Psi_Re_R + Psi_Re_L) );
        Flux_Im += 0.5 * Face_Area_Vec[k] * (  h_2m*(d_Psi_Re_R[k] + d_Psi_Re_L[k]) + v_face[k]*(Psi_Im_R + Psi_Im_L) );
        Flux_M  += 0.5 * Face_Area_Vec[k] * ( 2. * h_2m * (Psi_Im_R * d_Psi_Re_R[k] - Psi_Re_R * d_Psi_Im_R[k]
                                                         + Psi_Im_L * d_Psi_Re_L[k] - Psi_Re_L * d_Psi_Im_L[k])
                                             + v_face[k] * (psi2_L + psi2_R) );
    }
    double k_eff   = 0.3 / kernel.r;
    double cs_eff  = 0.5 * h_2m * k_eff;
    double prefac  = 0.5 * Face_Area_Norm * cs_eff;
    Flux_Re += prefac * (Psi_Re_R - Psi_Re_L);
    Flux_Im += prefac * (Psi_Im_R - Psi_Im_L);
    Flux_M  += prefac * (psi2_R - psi2_L);

    out.AGS_Dt_Psi_Re += Flux_Re;
    out.AGS_Dt_Psi_Im += Flux_Im;
    out.AGS_Dt_Psi_Mass += Flux_M;
#endif
#else
    (void)local; (void)j; (void)P; (void)kernel; (void)out;
#endif /* DM_FUZZY */
}

#endif /* DM_FUZZY_FLUX_FUNCTIONS_H */
