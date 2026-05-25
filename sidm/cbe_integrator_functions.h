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

/* --------------------------------------------------------------------------
 * Basis-pair cost and outgoing-side assignment helpers (2026-05-24).
 *
 * Conservation principle: the C step in cbe_integrator_flux_functions.h
 * applies -F to source basis and +F to target basis for every transfer.
 * Global mass/p/T conservation holds regardless of which pairing rule is
 * used; different rules produce different mixing patterns.
 *
 * Only velocity-only cost + greedy outgoing assignment are exposed here
 * (matches the corrected post-Commit-0 inline path). Other cost forms (W2,
 * free-slot-fallback) and assignment forms (Hungarian) will land with
 * their implementations and dimensional/realizability handling intact;
 * see also the 'theta uses dp not Face_Area_Vec' note in
 * cbe_integrator_flux_functions.h.
 * -------------------------------------------------------------------------- */

/* Velocity-only squared cost between two basis components:
 *     C = sum over k of (v_a_k - v_b_k)^2
 * Matches the corrected post-Commit-0 inline path in
 * cbe_integrator_flux_functions.h. Basis-mass denominators floored at
 * MIN_REAL_NUMBER to avoid NaN. */
KOKKOS_INLINE_FUNCTION
double cbe_cost_v_only(const double moments_a[CBE_INTEGRATOR_NMOMENTS],
                       const double moments_b[CBE_INTEGRATOR_NMOMENTS])
{
    double inv_a = 1.0 / DMAX(moments_a[0], MIN_REAL_NUMBER);
    double inv_b = 1.0 / DMAX(moments_b[0], MIN_REAL_NUMBER);
    double c = 0;
    for(int k=0; k<3; k++) {
        double dv = moments_a[k+1] * inv_a - moments_b[k+1] * inv_b;
        c += dv * dv;
    }
    return c;
}


/* Outgoing-side greedy assignment: given an NBASIS x NBASIS cost matrix C,
 * fill beta_of_alpha[m] = argmin_n C[m][n] for each row m. Multiple alphas
 * may collide on the same beta (asymmetric). Matches the corrected
 * post-Commit-0 inline argmin behavior.
 *
 * Call once with C for a-side outgoing, once with C^T for b-side outgoing. */
KOKKOS_INLINE_FUNCTION
void cbe_assign_outgoing_greedy(
    const double C[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NBASIS],
    int beta_of_alpha[CBE_INTEGRATOR_NBASIS])
{
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        double best = MAX_REAL_NUMBER; int best_n = 0;
        for(int n=0; n<CBE_INTEGRATOR_NBASIS; n++) {
            if(C[m][n] < best) { best = C[m][n]; best_n = n; }
        }
        beta_of_alpha[m] = best_n;
    }
}


/* --------------------------------------------------------------------------
 * Per-face root-found face-normal velocity v_F_n (Wave-CBE Commit 3,
 * 2026-05-25). Replaces the bulk-average vface_new computation in
 * cbe_integrator_flux_functions.h with a scalar normal v_F chosen so the
 * basis-summed mass flux across the face is machine-zero. Per harness
 * test_density_wave_root_found this drops per-cell mass drift by ~11
 * orders of magnitude vs the paper-formula bulk average.
 *
 * Residual function. Returns net per-unit-area face mass flux at trial
 * v_F_n, in (rho * v) units. Theta convention: face-normal Ahat (i->j),
 * a-side basis outgoing iff v_alpha_n_i > v_F_n; b-side basis outgoing
 * iff v_alpha_n_j < v_F_n. K_i[m] = m_i[m][0] * rho_i / M_i (>=0) folds
 * the wt_prefac density factor in so the returned residual matches the
 * actual flux update sign-for-sign.
 *
 * Piecewise-linear monotonically increasing in v_F_n: the unique zero
 * is well-defined when the bracket spans it. Bisection is sufficient
 * (codex's "boring, device-safe" pick; no Brent edge cases on Kokkos
 * device headers). Hand-rolled brent can swap in behind the same
 * interface later if profiling shows root-find cost matters.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
double cbe_face_mass_residual_per_unit_area(
    double v_F_n,
    const double v_alpha_n_i[CBE_INTEGRATOR_NBASIS],
    const double v_alpha_n_j[CBE_INTEGRATOR_NBASIS],
    const double K_i[CBE_INTEGRATOR_NBASIS],
    const double K_j[CBE_INTEGRATOR_NBASIS])
{
    double r = 0;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        double dv_i = v_alpha_n_i[m] - v_F_n;
        double dv_j = v_alpha_n_j[m] - v_F_n;
        if(dv_i > 0) r += dv_i * K_i[m];
        if(dv_j < 0) r += dv_j * K_j[m];
    }
    return r;
}


/* Bisection root-find for v_F_n with bracket-widen-up-to-4x and explicit
 * fallback flagged via bracket_ok_out=0. Matches the harness brentq
 * widening loop in cadence; uses bisection instead of Brent for device
 * portability. 40 iters of bisection in a unit bracket converges to
 * ~1e-12, matching the harness xtol. NO silent midpoint -- on failure
 * the caller's analytic fallback v_F is used and the bracket_fail
 * counter is incremented. */
KOKKOS_INLINE_FUNCTION
double cbe_face_solve_v_F_normal(
    const double v_alpha_n_i[CBE_INTEGRATOR_NBASIS],
    const double v_alpha_n_j[CBE_INTEGRATOR_NBASIS],
    const double K_i[CBE_INTEGRATOR_NBASIS],
    const double K_j[CBE_INTEGRATOR_NBASIS],
    double v_F_lo, double v_F_hi,
    double fallback_v_F_n,
    int *bracket_ok_out)
{
    double lo = v_F_lo, hi = v_F_hi;
    double f_lo = cbe_face_mass_residual_per_unit_area(lo, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
    double f_hi = cbe_face_mass_residual_per_unit_area(hi, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
    int bracketed = (f_lo * f_hi <= 0) ? 1 : 0;
    for(int widen = 0; widen < 4 && !bracketed; widen++) {
        double mid  = 0.5 * (lo + hi);
        double half = 4.0 * 0.5 * (hi - lo);
        lo = mid - half; hi = mid + half;
        f_lo = cbe_face_mass_residual_per_unit_area(lo, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
        f_hi = cbe_face_mass_residual_per_unit_area(hi, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
        if(f_lo * f_hi <= 0) { bracketed = 1; break; }
    }
    if(!bracketed) {
        *bracket_ok_out = 0;
        return fallback_v_F_n;
    }
    *bracket_ok_out = 1;
    for(int it=0; it<40; it++) {
        double mid = 0.5 * (lo + hi);
        double f_mid = cbe_face_mass_residual_per_unit_area(mid, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
        if(f_lo * f_mid <= 0) { hi = mid; f_hi = f_mid; }
        else                  { lo = mid; f_lo = f_mid; }
    }
    return 0.5 * (lo + hi);
}


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
