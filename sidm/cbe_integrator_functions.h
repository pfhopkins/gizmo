/* cbe_integrator_functions.h — GPU-callable CBE (Collisionless Boltzmann
 * Equation) single-sided flux computation. Pure math with no global state.
 *
 * Mirrors do_cbe_flux_computation() in sidm/cbe_integrator.cc exactly, but
 * as a KOKKOS_INLINE_FUNCTION so it can be called both from the CPU
 * tree-walk (via cbe_integrator_flux_functions.h) and from the B2 AGSForce
 * GPU kernel.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef CBE_INTEGRATOR_FUNCTIONS_H
#define CBE_INTEGRATOR_FUNCTIONS_H

#include "../declarations/allvars.h"
#include "../declarations/gpu_rng.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef CBE_INTEGRATOR
/* Per-loop FNV-1a salt for the drift-kick basis-resplit RNG. Mixed into the
 * counter so the CBE drift-kick stream is independent of any other loop that
 * happens to share (Ti_Current, ID, tag). See gpu_rng.h for rationale. */
static constexpr uint64_t CBE_DRIFT_KICK_RNG_SALT = gizmo_loop_rng_salt("cbe_drift_kick");
#endif


#ifdef CBE_INTEGRATOR
KOKKOS_INLINE_FUNCTION
double do_cbe_flux_computation(double moments[CBE_INTEGRATOR_NMOMENTS],
                               double vface_dot_A,
                               double vface[3],
                               double Area[3],
                               double moments_ngb[CBE_INTEGRATOR_NMOMENTS],
                               double fluxes[CBE_INTEGRATOR_NMOMENTS])
{
    double m_inv = 1. / moments[0];
    double v[3], f00_vsig = 1;
    v[0] = m_inv*moments[1]; v[1] = m_inv*moments[2]; v[2] = m_inv*moments[3];
    double vsig = v[0]*Area[0] + v[1]*Area[1] + v[2]*Area[2] - vface_dot_A;
    if(fabs(vsig) <= 0) return 0;
    fluxes[0] = vsig * moments[0];
    for(int k = 1; k < CBE_INTEGRATOR_NMOMENTS; k++) { fluxes[k] = (m_inv * moments[k]) * fluxes[0]; }

#if (CBE_INTEGRATOR_NMOMENTS > 4)
    {
        double dv2 = (v[0]-vface[0])*(v[0]-vface[0]) + (v[1]-vface[1])*(v[1]-vface[1]) + (v[2]-vface[2])*(v[2]-vface[2]);
        double c_eff_over_vsig_A = sqrt(m_inv * (moments[4]+moments[5]+moments[6]) / dv2);
        double SM_vsig = 1 + c_eff_over_vsig_A*c_eff_over_vsig_A/(1 + c_eff_over_vsig_A);
        double f00_SdotA = 1 - c_eff_over_vsig_A / (SM_vsig + c_eff_over_vsig_A);
        f00_vsig = (SM_vsig * (1 + c_eff_over_vsig_A)) / (SM_vsig + c_eff_over_vsig_A);

        double S_dot_A[3];
        S_dot_A[0] = f00_SdotA * (moments[4]*Area[0] + moments[7]*Area[1] + moments[8]*Area[2]);
        S_dot_A[1] = f00_SdotA * (moments[7]*Area[0] + moments[5]*Area[1] + moments[9]*Area[2]);
        S_dot_A[2] = f00_SdotA * (moments[8]*Area[0] + moments[9]*Area[1] + moments[6]*Area[2]);
        fluxes[1] += S_dot_A[0];
        fluxes[2] += S_dot_A[1];
        fluxes[3] += S_dot_A[2];
        fluxes[4] += 2.*v[0]*S_dot_A[0] + fluxes[0]*v[0]*v[0];
        fluxes[5] += 2.*v[1]*S_dot_A[1] + fluxes[0]*v[1]*v[1];
        fluxes[6] += 2.*v[2]*S_dot_A[2] + fluxes[0]*v[2]*v[2];
        fluxes[7] += v[0]*S_dot_A[1] + v[1]*S_dot_A[0] + fluxes[0]*v[0]*v[1];
        fluxes[8] += v[0]*S_dot_A[2] + v[2]*S_dot_A[0] + fluxes[0]*v[0]*v[2];
        fluxes[9] += v[1]*S_dot_A[2] + v[2]*S_dot_A[1] + fluxes[0]*v[1]*v[2];
    }
#else
    (void)vface; (void)moments_ngb;
#endif
    return vsig * f00_vsig;
}


/* GPU-callable per-particle drift-kick update for the CBE integrator.
 * Mirrors do_cbe_drift_kick() in cbe_integrator.cc but takes an explicit
 * particle ref so it runs in both CPU and GPU (Kokkos) contexts.
 * get_random_number() replaced by counter-based gpu_rng (same statistics,
 * deterministic per particle-ID + timestep + loop-domain salt). */
KOKKOS_INLINE_FUNCTION
static void do_cbe_drift_kick_kernel(struct particle_data& pi, double dt)
{
    int j, k;
    double moment[CBE_INTEGRATOR_NMOMENTS]={0}, dmoment[CBE_INTEGRATOR_NMOMENTS]={0}, minv=1./pi.Mass;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
        { moment[k] += pi.CBE_basis_moments[j][k]; dmoment[k] += dt*pi.CBE_basis_moments_dt[j][k]; }
    double biggest_dm = 1.e10;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        double q = (dt*pi.CBE_basis_moments_dt[j][0] - pi.CBE_basis_moments[j][0]*minv*dmoment[0]) / (pi.CBE_basis_moments[j][0] * (1.+minv*dmoment[0]));
        if(!isnan(q)) {if(q < biggest_dm) {biggest_dm=q;}}
    }
    double nfac = 1, threshold_dm = -0.75;
    if(biggest_dm < threshold_dm) {nfac = threshold_dm/biggest_dm;}
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        pi.CBE_basis_moments[j][0] += nfac * (dt*pi.CBE_basis_moments_dt[j][0] - pi.CBE_basis_moments[j][0]*minv*dmoment[0]);
        for(k=1;k<4;k++) {pi.CBE_basis_moments[j][k] += nfac * (dt*pi.CBE_basis_moments_dt[j][k] - pi.CBE_basis_moments[j][0]*minv*dmoment[k]);}
#if (CBE_INTEGRATOR_NMOMENTS > 4)
        for(k=4;k<CBE_INTEGRATOR_NMOMENTS;k++) {pi.CBE_basis_moments[j][k] += nfac * (dt*pi.CBE_basis_moments_dt[j][k]);}
        double eps_tmp = 1.e-8;
        for(k=4;k<7;k++) {if(pi.CBE_basis_moments[j][k] < MIN_REAL_NUMBER) {pi.CBE_basis_moments[j][k]=MIN_REAL_NUMBER;}}
        double xyMax = sqrt(pi.CBE_basis_moments[j][4]*pi.CBE_basis_moments[j][5]) * (1.-eps_tmp);
        double xzMax = sqrt(pi.CBE_basis_moments[j][4]*pi.CBE_basis_moments[j][6]) * (1.-eps_tmp);
        double yzMax = sqrt(pi.CBE_basis_moments[j][5]*pi.CBE_basis_moments[j][6]) * (1.-eps_tmp);
        if(pi.CBE_basis_moments[j][7] > xyMax) {pi.CBE_basis_moments[j][7] = xyMax;}
        if(pi.CBE_basis_moments[j][8] > xzMax) {pi.CBE_basis_moments[j][8] = xzMax;}
        if(pi.CBE_basis_moments[j][9] > yzMax) {pi.CBE_basis_moments[j][9] = yzMax;}
        if(pi.CBE_basis_moments[j][7] < -xyMax) {pi.CBE_basis_moments[j][7] = -xyMax;}
        if(pi.CBE_basis_moments[j][8] < -xzMax) {pi.CBE_basis_moments[j][8] = -xzMax;}
        if(pi.CBE_basis_moments[j][9] < -yzMax) {pi.CBE_basis_moments[j][9] = -yzMax;}
        double crossnorm = 1;
        double detSMatrix_Diag = pi.CBE_basis_moments[j][4]*pi.CBE_basis_moments[j][5]*pi.CBE_basis_moments[j][6];
        double detSMatrix_Cross = 2.*pi.CBE_basis_moments[j][7]*pi.CBE_basis_moments[j][8]*pi.CBE_basis_moments[j][9]
            - (  pi.CBE_basis_moments[j][4]*pi.CBE_basis_moments[j][9]*pi.CBE_basis_moments[j][9]
               + pi.CBE_basis_moments[j][5]*pi.CBE_basis_moments[j][8]*pi.CBE_basis_moments[j][8]
               + pi.CBE_basis_moments[j][6]*pi.CBE_basis_moments[j][7]*pi.CBE_basis_moments[j][7] );
        if(detSMatrix_Diag <= 0) {
            crossnorm=0; for(k=4;k<7;k++) {if(pi.CBE_basis_moments[j][k]<MIN_REAL_NUMBER) {pi.CBE_basis_moments[j][k]=MIN_REAL_NUMBER;}}
        } else if(detSMatrix_Diag + detSMatrix_Cross <= 0) {
            crossnorm = (-detSMatrix_Diag * (1.-eps_tmp)) / detSMatrix_Cross;
        }
        if(crossnorm < 1) {for(k=7;k<10;k++) {pi.CBE_basis_moments[j][k] *= crossnorm;}}
        /* simplify to 1D dispersion along direction of motion */
        if(2==2) {
            double S0 = pi.CBE_basis_moments[j][4]+pi.CBE_basis_moments[j][5]+pi.CBE_basis_moments[j][6], vhat[3]={0}, vmag=0;
            for(k=0;k<3;k++) {vhat[k]=pi.CBE_basis_moments[j][k+1]; vmag+=vhat[k]*vhat[k];}
            if(vmag > 0) {
                vmag = 1./sqrt(vmag); for(k=0;k<3;k++) {vhat[k]*=vmag;}
                pi.CBE_basis_moments[j][4]=S0*vhat[0]*vhat[0]; pi.CBE_basis_moments[j][5]=S0*vhat[1]*vhat[1];
                pi.CBE_basis_moments[j][6]=S0*vhat[2]*vhat[2]; pi.CBE_basis_moments[j][7]=S0*vhat[0]*vhat[1];
                pi.CBE_basis_moments[j][8]=S0*vhat[0]*vhat[2]; pi.CBE_basis_moments[j][9]=S0*vhat[1]*vhat[2];
            }
        }
#endif
    }
    /* split the largest basis into the smallest when one becomes degenerate */
    double mmax=-1, mmin=1.e10*pi.Mass; int jmin=-1,jmax=-1;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    { double m=pi.CBE_basis_moments[j][0]; if(m<mmin){mmin=m;jmin=j;} if(m>mmax){mmax=m;jmax=j;} }
    if((mmin < 1.e-5 * mmax) && (jmin >= 0) && (jmax >= 0) && (All.Time > All.TimeBegin))
    {
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
        {
            double dq = 0.5*pi.CBE_basis_moments[jmax][k];
            if(k>0 && k<4) {
                uint64_t key = (uint64_t)pi.ID ^ ((uint64_t)jmax*65537ULL) ^ ((uint64_t)k*131071ULL);
                uint64_t counter = ((uint64_t)All.Ti_Current << 32) ^ CBE_DRIFT_KICK_RNG_SALT;
                dq *= 1. + 0.001*(gizmo_gpu_rand_double(key, counter) - 0.5);
            }
            pi.CBE_basis_moments[jmax][k] -= dq;
            pi.CBE_basis_moments[jmin][k] += dq;
        }
    }
}

/* GPU-callable per-particle post-gravity finalization for the CBE integrator.
 * Mirrors do_postgravity_cbe_calcs() (originally in cbe_integrator.cc) but
 * takes an explicit particle ref so it runs in both CPU and GPU contexts.
 *
 * Moment ordering: 0, x, y, z, xx, yy, zz, xy, xz, yz
 *
 * Operations (pure i-side):
 *   - Sum dmom_tot across basis functions.
 *   - Fold dmom_tot[1..3] / Mass into pi.GravAccel (cosmological-unit shift).
 *   - Subtract that residual back out of pi.CBE_basis_moments_dt so the net
 *     momentum flux across basis functions is zero to FP precision.
 *   - For NMOMENTS > 4, shift the second-moment derivatives from dT to dS
 *     (subtract the v.v outer-product contribution) and clamp diagonal
 *     components non-negative when dm[0] > 0.
 */
KOKKOS_INLINE_FUNCTION
static void do_cbe_postgravity_kernel(struct particle_data& pi)
{
    int j, k;
    double dmom_tot[CBE_INTEGRATOR_NMOMENTS] = {0};
    double m_inv = 1. / pi.Mass;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {
            dmom_tot[k] += pi.CBE_basis_moments_dt[j][k];
        }
    }
    Vec3<double> dv0 = {m_inv * dmom_tot[1], m_inv * dmom_tot[2], m_inv * dmom_tot[3]};
    pi.GravAccel += dv0 / All.cf_a2inv;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        pi.CBE_basis_moments_dt[j][0] -= pi.CBE_basis_moments[j][0] * (m_inv * dmom_tot[0]);
        for(k=0;k<3;k++) {
            pi.CBE_basis_moments_dt[j][k+1] -= pi.CBE_basis_moments[j][0] * dv0[k];
        }
#if (CBE_INTEGRATOR_NMOMENTS > 4)
        {
            double dS[6] = {0};
            dS[0] = pi.CBE_basis_moments_dt[j][4] - m_inv * (pi.CBE_basis_moments_dt[j][1]*pi.CBE_basis_moments[j][1] + pi.CBE_basis_moments[j][1]*pi.CBE_basis_moments_dt[j][1]) + m_inv*m_inv * pi.CBE_basis_moments_dt[j][0] * pi.CBE_basis_moments[j][1]*pi.CBE_basis_moments[j][1];
            dS[1] = pi.CBE_basis_moments_dt[j][5] - m_inv * (pi.CBE_basis_moments_dt[j][2]*pi.CBE_basis_moments[j][2] + pi.CBE_basis_moments[j][2]*pi.CBE_basis_moments_dt[j][2]) + m_inv*m_inv * pi.CBE_basis_moments_dt[j][0] * pi.CBE_basis_moments[j][2]*pi.CBE_basis_moments[j][2];
            dS[2] = pi.CBE_basis_moments_dt[j][6] - m_inv * (pi.CBE_basis_moments_dt[j][3]*pi.CBE_basis_moments[j][3] + pi.CBE_basis_moments[j][3]*pi.CBE_basis_moments_dt[j][3]) + m_inv*m_inv * pi.CBE_basis_moments_dt[j][0] * pi.CBE_basis_moments[j][3]*pi.CBE_basis_moments[j][3];
            dS[3] = pi.CBE_basis_moments_dt[j][7] - m_inv * (pi.CBE_basis_moments_dt[j][1]*pi.CBE_basis_moments[j][2] + pi.CBE_basis_moments[j][1]*pi.CBE_basis_moments_dt[j][2]) + m_inv*m_inv * pi.CBE_basis_moments_dt[j][0] * pi.CBE_basis_moments[j][1]*pi.CBE_basis_moments[j][2];
            dS[4] = pi.CBE_basis_moments_dt[j][8] - m_inv * (pi.CBE_basis_moments_dt[j][1]*pi.CBE_basis_moments[j][3] + pi.CBE_basis_moments[j][1]*pi.CBE_basis_moments_dt[j][3]) + m_inv*m_inv * pi.CBE_basis_moments_dt[j][0] * pi.CBE_basis_moments[j][1]*pi.CBE_basis_moments[j][3];
            dS[5] = pi.CBE_basis_moments_dt[j][9] - m_inv * (pi.CBE_basis_moments_dt[j][2]*pi.CBE_basis_moments[j][3] + pi.CBE_basis_moments[j][2]*pi.CBE_basis_moments_dt[j][3]) + m_inv*m_inv * pi.CBE_basis_moments_dt[j][0] * pi.CBE_basis_moments[j][2]*pi.CBE_basis_moments[j][3];
            if(pi.CBE_basis_moments_dt[j][0] > 0) { for(k=0;k<3;k++) {dS[k] = DMAX(dS[k], 0.);} }
            for(k=4;k<CBE_INTEGRATOR_NMOMENTS;k++) { pi.CBE_basis_moments_dt[j][k] = dS[k-4]; }
        }
#endif
    }
}

#endif /* CBE_INTEGRATOR */

#endif /* CBE_INTEGRATOR_FUNCTIONS_H */
