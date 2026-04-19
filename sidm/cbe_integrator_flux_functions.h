/* cbe_integrator_flux_functions.h -- per-pair CBE (Collisionless Boltzmann
 * Equation) flux computation for the adaptive-gravity AGSForce loop.
 *
 * Replaces the fragment sidm/cbe_integrator_flux_computation.h. Body guarded
 * by CBE_INTEGRATOR so the caller (ags_rkern.cc AGSForce_evaluate) can invoke
 * unconditionally. Pure i-accumulation into the caller out struct; the only
 * shared-memory writes are the optional P[j].wakeup flag, which is returned
 * as a boolean out in CbeFluxResult and applied atomically by the caller.
 *
 * Stack note: this function stacks several CBE_INTEGRATOR_NBASIS-sized arrays
 * of size up to CBE_INTEGRATOR_NMOMENTS (largest ~ NBASIS*NMOMENTS doubles
 * per side). For the current NBASIS<=8, NMOMENTS<=11 configs this is ~700B
 * per side; GPU-stack-safe. Values verified at the current config; revisit
 * if NBASIS grows > ~32.
 *
 * Requires allvars.h / proto.h and sidm/cbe_integrator.h forward decls
 * (for do_cbe_flux_computation and get_particle_volume_ags).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef CBE_INTEGRATOR_FLUX_FUNCTIONS_H
#define CBE_INTEGRATOR_FLUX_FUNCTIONS_H

#include "../gravity/ags_functions.h"
#include "cbe_integrator_functions.h"

struct CbeFluxResult {
    int set_wakeup_j;  /* 1 if P[j].wakeup should be set to -1 */
};

template <typename LocalT, typename KernelT, typename OutT>
KOKKOS_INLINE_FUNCTION
CbeFluxResult cbe_integrator_flux_compute_pair(
    const LocalT &local,
    int j,
    struct particle_data *P,
    const KernelT &kernel,
    OutT &out,
    const int *timebin_active)
{
    CbeFluxResult r; r.set_wakeup_j = 0;
#ifdef CBE_INTEGRATOR
    double V_i = local.V_i, V_j = get_particle_volume_ags_P(j, P);
    double rho_i = local.Mass / V_i * All.cf_a3inv;
    double rho_j = P[j].Mass / V_j * All.cf_a3inv;
    double psi_i, psi_j;
    psi_i = 0.5; psi_j = 1 - psi_i;
    rho_i *= psi_i; rho_j *= psi_j;

    double Face_Area_Vec[3];
    double Face_Area_Norm = 0;
    double vface_guess[3];
    double vf0_dot_dp = 0;
    for(int k=0; k<3; k++) {
        Face_Area_Vec[k] = -(kernel.wk_i * V_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2])
                           + kernel.wk_j * V_j * (P[j].NV_T[k][0]*kernel.dp[0] + P[j].NV_T[k][1]*kernel.dp[1] + P[j].NV_T[k][2]*kernel.dp[2])) * All.cf_atime * All.cf_atime;
        Face_Area_Norm += Face_Area_Vec[k] * Face_Area_Vec[k];
        vface_guess[k] = 0.5 * (local.Vel[k] + P[j].Vel[k]) / All.cf_atime;
        vf0_dot_dp += vface_guess[k] * kernel.dp[k];
    }
    Face_Area_Norm = sqrt(Face_Area_Norm);

    /* load basis moments, boosted to the physical frame */
    double local_CBE_basis_moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double Pj_CBE_basis_moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
            local_CBE_basis_moments[m][k] = local.CBE_basis_moments[m][k];
            Pj_CBE_basis_moments[m][k]    = P[j].CBE_basis_moments[m][k];
            if((k>0) && (k<4)) {
                local_CBE_basis_moments[m][k] += local_CBE_basis_moments[m][0] * local.Vel[k-1] / All.cf_atime;
                Pj_CBE_basis_moments[m][k]    += Pj_CBE_basis_moments[m][0]    * P[j].Vel[k-1] / All.cf_atime;
            }
        }
    }

    /* center-of-motion frame: determine which bases are approaching, define face velocity */
    double vface_new[3] = {0};
    double theta_i[CBE_INTEGRATOR_NBASIS] = {0};
    double theta_j[CBE_INTEGRATOR_NBASIS] = {0};
    double v_wt_sum = 0;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        double vi_dot_dp = 0, vj_dot_dp = 0;
        for(int k=0; k<3; k++) {
            vi_dot_dp += (local_CBE_basis_moments[m][k+1] / local_CBE_basis_moments[m][0] - vface_guess[k]) * kernel.dp[k];
            vj_dot_dp += (Pj_CBE_basis_moments[m][k+1]    / Pj_CBE_basis_moments[m][0]    - vface_guess[k]) * kernel.dp[k];
        }
        if(vi_dot_dp < 0) { theta_i[m] = 1; }
        if(vj_dot_dp > 0) { theta_j[m] = 1; }
        double w0_i = theta_i[m] * rho_i / local.Mass;
        double w0_j = theta_j[m] * rho_j / P[j].Mass;
        v_wt_sum += w0_i * local_CBE_basis_moments[m][0] + w0_j * Pj_CBE_basis_moments[m][0];
        for(int k=0; k<3; k++) { vface_new[k] += w0_i * local_CBE_basis_moments[m][k+1] + w0_j * Pj_CBE_basis_moments[m][k+1]; }
    }

    double vface[3] = {0};
    if(!((v_wt_sum > MIN_REAL_NUMBER) && (v_wt_sum < MAX_REAL_NUMBER))) { return r; }

    for(int k=0; k<3; k++) { vface[k] = vface_new[k] / v_wt_sum; }

    /* find best-match basis pairs across the two sides */
    int matching_basis_j_for_basis_in_i[CBE_INTEGRATOR_NBASIS];
    int matching_basis_i_for_basis_in_j[CBE_INTEGRATOR_NBASIS];
    double wt_i[CBE_INTEGRATOR_NBASIS], wt_j[CBE_INTEGRATOR_NBASIS];
    double vsig = 0;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        wt_i[m] = wt_j[m] = MAX_REAL_NUMBER;
    }
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        for(int m_j=0; m_j<CBE_INTEGRATOR_NBASIS; m_j++) {
            double cos_ij = 0;
            for(int k=0; k<3; k++) {
                double q1 = local_CBE_basis_moments[m][k+1]   / local_CBE_basis_moments[m][0];
                double q2 = Pj_CBE_basis_moments[m_j][k+1]    / Pj_CBE_basis_moments[m][0];
                cos_ij += (q1 - q2) * (q1 - q2);
            }
            if(cos_ij < wt_i[m])   { wt_i[m] = cos_ij;   matching_basis_j_for_basis_in_i[m]   = m_j; }
            if(cos_ij < wt_j[m_j]) { wt_j[m_j] = cos_ij; matching_basis_i_for_basis_in_j[m_j] = m; }
        }
    }

    /* now compute fluxes */
    double wt_prefac_i = -rho_i / local.Mass;
    double wt_prefac_j = -rho_j / P[j].Mass;
    double vface_dot_A = vface[0]*Face_Area_Vec[0] + vface[1]*Face_Area_Vec[1] + vface[2]*Face_Area_Vec[2];
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        int j_m = matching_basis_j_for_basis_in_i[m];
        int i_m = matching_basis_i_for_basis_in_j[m];
        double flux[CBE_INTEGRATOR_NMOMENTS] = {0};
        double vsig_i = 0, vsig_j = 0;
        if(theta_i[m] == 1) {
            vsig_i = do_cbe_flux_computation(local_CBE_basis_moments[m], vface_dot_A, vface, Face_Area_Vec, Pj_CBE_basis_moments[j_m], flux);
            for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
                flux[k] *= wt_prefac_i;
                out.CBE_basis_moments_dt[m][k] += flux[k];
            }
        }
        if(theta_j[m] == 1) {
            vsig_j = do_cbe_flux_computation(Pj_CBE_basis_moments[m], vface_dot_A, vface, Face_Area_Vec, local_CBE_basis_moments[i_m], flux);
            for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
                flux[k] *= wt_prefac_j;
                out.CBE_basis_moments_dt[i_m][k] += flux[k];
            }
        }
        vsig = DMAX(DMAX(fabs(vsig_i), fabs(vsig_j)), vsig);
    }
    vsig /= Face_Area_Norm * All.cf_atime;
    if(vsig > out.AGS_vsig) { out.AGS_vsig = vsig; }
    if(!(timebin_active[P[j].TimeBin]) && (All.Time > All.TimeBegin)) {
        if(vsig > WAKEUP * P[j].AGS_vsig) { r.set_wakeup_j = 1; }
    }
#else
    (void)local; (void)j; (void)P; (void)kernel; (void)out;
#endif /* CBE_INTEGRATOR */
    return r;
}

#endif /* CBE_INTEGRATOR_FLUX_FUNCTIONS_H */
