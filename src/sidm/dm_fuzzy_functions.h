/* dm_fuzzy_functions.h — GPU-callable DM_FUZZY flux / reconstruction helpers:
 *   - do_dm_fuzzy_flux_computation            (DM_FUZZY_USE_SIMPLER_HLL_SOLVER=1 path)
 *   - do_dm_fuzzy_flux_computation_old        (DM_FUZZY_USE_SIMPLER_HLL_SOLVER=0 path)
 *   - dm_fuzzy_reconstruct_and_slopelimit_sub
 *   - dm_fuzzy_reconstruct_and_slopelimit
 *
 * All four were host-only routines in sidm/dm_fuzzy.cc. They are pure math —
 * no global state apart from All (available on GPU via All_dev) — so they
 * become KOKKOS_INLINE_FUNCTION directly, callable from both the CPU
 * tree-walk and the B2 AGSForce GPU kernel.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DM_FUZZY_FUNCTIONS_H
#define DM_FUZZY_FUNCTIONS_H

#include "../declarations/allvars.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif


#ifdef DM_FUZZY

KOKKOS_INLINE_FUNCTION
void do_dm_fuzzy_flux_computation(double HLLwt, double dt, double prev_a, Vec3<double>& dv,
                                  double GradRho_L[3], double GradRho_R[3],
                                  double GradRho2_L[3][3], double GradRho2_R[3][3],
                                  double rho_L, double rho_R, double dv_Right_minus_Left,
                                  Vec3<double>& Area, Vec3<double>& fluxes,
                                  double AGS_Numerical_QuantumPotential_L,
                                  double AGS_Numerical_QuantumPotential_R,
                                  double *dt_egy_Numerical_QuantumPotential)
{
    double f00 = 0.5 * All.ScalarField_hbar_over_mass; f00 *= f00;
    double gamma_eff = 5./3., PL_dot_AA = 0, PR_dot_AA = 0, rSi = 1./(rho_L+rho_R);
    double QL = 0, QR = 0;
    Vec3<double> PL_dot_A = {0,0,0}, PR_dot_A = {0,0,0};
    double Face_Area_Norm = Area.norm_sq();
    for(int m = 0; m < 3; m++) {
        QL += GradRho2_L[m][m] - 0.5 * GradRho_L[m] * GradRho_L[m] / rho_L;
        QR += GradRho2_R[m][m] - 0.5 * GradRho_R[m] * GradRho_R[m] / rho_R;
    }
    double PN_L = (gamma_eff - 1.) * (AGS_Numerical_QuantumPotential_L - f00 * QL);
    double PN_R = (gamma_eff - 1.) * (AGS_Numerical_QuantumPotential_R - f00 * QR);
    PN_L = DMAX(PN_L, 0); PN_R = DMAX(PN_R, 0);
    for(int m = 0; m < 3; m++) {
        for(int k = 0; k < 3; k++) {
            PL_dot_A[m] += Area[k] * f00 * (GradRho_L[m]*GradRho_L[k]/rho_L - GradRho2_L[m][k]);
            PR_dot_A[m] += Area[k] * f00 * (GradRho_R[m]*GradRho_R[k]/rho_R - GradRho2_R[m][k]);
        }
        fluxes[m] = (rho_L * (PR_dot_A[m] + PN_R*Area[m]) + rho_R * (PL_dot_A[m] + PN_L*Area[m])) * rSi;
    }
    PL_dot_AA = dot(PL_dot_A, Area); PR_dot_AA = dot(PR_dot_A, Area);
    PL_dot_AA /= Face_Area_Norm; PR_dot_AA /= Face_Area_Norm;
    Face_Area_Norm = sqrt(Face_Area_Norm);
    double PL_norm = PL_dot_AA + PN_L, PR_norm = PR_dot_AA + PN_L;
    double cs_L = gamma_eff * PL_norm / rho_L, cs_R = gamma_eff * PR_norm / rho_R;
    double cs = DMAX(cs_L, cs_R), ceff = cs;
    if(dv_Right_minus_Left < 0) ceff += fabs(dv_Right_minus_Left);
    fluxes -= rho_L * rho_R * rSi * ceff * dv_Right_minus_Left * Area;
    double S_M = ((PL_norm - PR_norm)/ceff + 0.5*(rho_R - rho_L)*dv_Right_minus_Left) * rSi;
    *dt_egy_Numerical_QuantumPotential = dot(fluxes, S_M * Area / Face_Area_Norm - 0.5 * dv);
    (void)HLLwt; (void)dt; (void)prev_a;
}


KOKKOS_INLINE_FUNCTION
void do_dm_fuzzy_flux_computation_old(double HLLwt, double dt, double m0, double prev_a,
                                      Vec3<double>& dp, Vec3<double>& dv,
                                      double GradRho_L[3], double GradRho_R[3],
                                      double GradRho2_L[3][3], double GradRho2_R[3][3],
                                      double rho_L, double rho_R, double dv_Right_minus_Left,
                                      Vec3<double>& Area, Vec3<double>& fluxes,
                                      double AGS_Numerical_QuantumPotential,
                                      double *dt_egy_Numerical_QuantumPotential)
{
    if(dt <= 0) return;
    double f00 = 0.5 * All.ScalarField_hbar_over_mass;
    double f2 = f00 * f00, rhoL_i = 1./rho_L, rhoR_i = 1./rho_R, rSi = 1./(rho_L+rho_R);
    *dt_egy_Numerical_QuantumPotential = 0;
    double r2 = dp.norm_sq(); fluxes = {0,0,0};
    if(r2 <= 0) return;
    double r = sqrt(r2), wavespeed = 2.*f00*(M_PI/r);
    if(dv_Right_minus_Left > wavespeed) return;

    double Face_Area_Norm = sqrt(Area.norm_sq());

    for(int m = 0; m < 3; m++) {
        for(int n = 0; n < 3; n++) {
            double QPT_L = f2 * (rhoL_i * GradRho_L[m]*GradRho_L[n] - GradRho2_L[m][n]);
            double QPT_R = f2 * (rhoR_i * GradRho_R[m]*GradRho_R[n] - GradRho2_R[m][n]);
            double P_star = (QPT_L * rho_R + QPT_R * rho_L) * rSi;
            fluxes[m] += Area[n] * P_star;
        }
    }
    double fluxmag = fluxes.norm();
    double fluxmax = 100. * Face_Area_Norm * f2 * 0.5 * (rho_L + rho_R) / (r*r);
    if(fluxmag > fluxmax) fluxes *= fluxmax / fluxmag;

    for(int m = 0; m < 3; m++) {
        double ftmp = (2./3.) * AGS_Numerical_QuantumPotential * Area[m];
        double fmax = 0.5 * m0 * fabs(dv[m]) / dt;
        fmax = DMAX(fmax, 100. * fluxmag);
        fmax = DMAX(DMIN(fmax, 100. * prev_a), fluxmag);
        if(fabs(ftmp) > fmax) ftmp *= fmax / fabs(ftmp);
        *dt_egy_Numerical_QuantumPotential -= 0.5 * ftmp * dv[m];
        fluxes[m] += ftmp;
    }
    fluxmag = fluxes.norm();

    if(dv_Right_minus_Left < 0) {
        double g_kL = sqrt(GradRho_L[0]*GradRho_L[0] + GradRho_L[1]*GradRho_L[1] + GradRho_L[2]*GradRho_L[2]);
        double g_kR = sqrt(GradRho_R[0]*GradRho_R[0] + GradRho_R[1]*GradRho_R[1] + GradRho_R[2]*GradRho_R[2]);
        double k_g1 = 0.5 * (rhoL_i * g_kL + rhoR_i * g_kR);
        double k_eff = 0, k_g2 = 0, k_g3 = 0;
        double k_d1 = fabs(rho_L - rho_R) / (0.5 * (rho_L + rho_R)) / r;
        if(isnan(k_d1)) k_d1 = 0;
        if(isnan(k_g1)) k_g1 = 0;
        double k2L = GradRho2_L[0][0] + GradRho2_L[1][1] + GradRho2_L[2][2];
        double k2R = GradRho2_R[0][0] + GradRho2_R[1][1] + GradRho2_R[2][2];
        k_g2 = sqrt((fabs(k2L) + fabs(k2R)) * rSi);
        k_g3 = 0.5 * sqrt(fabs(k2L - k2R) / (0.5 * (g_kL + g_kR) * r));
        double k_gtan = (fabs(k2L) + fabs(k2R)) / (MIN_REAL_NUMBER + fabs(g_kL) + fabs(g_kR));
        if(isnan(k_g2)) k_g2 = 0;
        if(isnan(k_g3)) k_g3 = 0;
        if(isnan(k_gtan)) k_gtan = 0;
        k_eff = DMIN(DMAX(k_gtan, DMAX(DMAX((3.+HLLwt)*DMAX(k_d1, k_g1), (2.+HLLwt)*k_g2), (1.5+HLLwt)*k_g3)), 1./r);
        if(isnan(k_eff)) k_eff = 0;
        double Pstar = (-dv_Right_minus_Left) * (f00 * k_eff + (-dv_Right_minus_Left)) * rho_L * rho_R * rSi;
        fluxmag = fluxes.norm();
        for(int m = 0; m < 3; m++) {
            double f_dir = Area[m] * Pstar;
            double fmax = 0.5 * m0 * fabs(dv[m]) / dt;
            fmax = DMAX(fmax, 10. * fluxmag);
            fmax = DMAX(DMIN(fmax, 40. * prev_a), fluxmag);
            if(fabs(f_dir) > fmax) f_dir *= fmax / fabs(f_dir);
            fluxes[m] += f_dir;
            *dt_egy_Numerical_QuantumPotential -= 0.5 * f_dir * dv[m];
        }
    }
}


KOKKOS_INLINE_FUNCTION
void dm_fuzzy_reconstruct_and_slopelimit_sub(double *u_R_f, double *u_L_f,
                                             double q_R, const Vec3<double>& dq_R_0,
                                             double q_L, const Vec3<double>& dq_L_0,
                                             const Vec3<double>& dx)
{
    double dq_L = 0; for(int k = 0; k < 3; k++) { dq_L += 0.5 * dx[k] * dq_L_0[k]; }
    double dq_R = 0; for(int k = 0; k < 3; k++) { dq_R -= 0.5 * dx[k] * dq_R_0[k]; }
    double q0 = q_L, u_L = 0, u_R = 0;
    q_L -= q0; q_R -= q0;

    if(dq_L * q_R < 0) dq_L = 0;
    if(dq_R * q_R < 0) dq_R = 0;
    u_L = dq_L; u_R = q_R + dq_R;
    if(q_R > 0) {
        if(u_L > u_R) { double tmp = 0.5*(u_L + u_R); u_L = tmp; u_R = tmp; }
        if(u_L > q_R) u_L = q_R;
        if(u_R > q_R) u_R = q_R;
        if(u_L < 0) u_L = 0;
        if(u_R < 0) u_R = 0;
    } else {
        if(u_L < u_R) { double tmp = 0.5*(u_L + u_R); u_L = tmp; u_R = tmp; }
        if(u_L > 0) u_L = 0;
        if(u_R > 0) u_R = 0;
        if(u_L < q_R) u_L = q_R;
        if(u_R < q_R) u_R = q_R;
    }
    if(q_R == 0) { u_L = 0; u_R = 0; }
    u_L += q0; u_R += q0;
    *u_L_f = u_L; *u_R_f = u_R;
}


/* Behaviour note: the original sidm/dm_fuzzy.cc body writes the
   slope-limited derivatives into the pass-by-value dq_R/dq_L (which are
   never read by the caller) and leaves du_R/du_L untouched. That appears
   to be a latent bug, but we preserve it exactly here to keep
   B2-refactor diffs bit-identical to the pre-refactor CPU tree-walk.
   The gas slope-limited values in u_R/u_L and the pass-through
   d2q_R/d2q_L are all that downstream flux code actually consumes. */
KOKKOS_INLINE_FUNCTION
void dm_fuzzy_reconstruct_and_slopelimit(double *u_R, double du_R[3],
                                         double *u_L, double du_L[3],
                                         double q_R, Vec3<double> dq_R, const Mat3<double>& d2q_R,
                                         double q_L, Vec3<double> dq_L, const Mat3<double>& d2q_L,
                                         const Vec3<double>& dx)
{
    double t_L, t_R;
    dm_fuzzy_reconstruct_and_slopelimit_sub(&t_R, &t_L, q_R, dq_R, q_L, dq_L, dx);
    *u_R = t_R; *u_L = t_L;
    for(int k = 0; k < 3; k++) {
        dm_fuzzy_reconstruct_and_slopelimit_sub(&t_R, &t_L, dq_R[k], d2q_R[k], dq_L[k], d2q_L[k], dx);
        dq_R[k] = t_R; dq_L[k] = t_L;
    }
    (void)du_R; (void)du_L;
}

#endif /* DM_FUZZY */

#endif /* DM_FUZZY_FUNCTIONS_H */
