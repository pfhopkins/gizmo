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

/* Wave-CBE Commit 5 (2026-05-26) — SPD repair eigenvalue floor, relative to
 * trace. Applied identically in cell-state drift-kick repair and face-state
 * Q-clamp via cbe_spd_repair_S3x3. 1e-12 is a small relative floor (well
 * below the intended stress scale, but large enough to deterministically
 * exclude strictly-zero / negative eigenvalues from downstream consumers). */
static constexpr double CBE_SPD_RELATIVE_FLOOR = 1.0e-12;
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
 * C0 / C1 exposed velocity-only cost + greedy outgoing assignment (matches
 * the corrected post-C0 inline path). C6a adds trace-W2 cost and the
 * adaptive-free-slot row transform per harness §4.4; these are unused in
 * C6a (no callers; default routing flips in C6c via the compile-time
 * selectors added in C6b). Hungarian assignment is deliberately NOT
 * added in C6 -- harness §4.1 + Phil designate one-sided-nearest (=
 * greedy) the production assignment; Hungarian is a comparator option
 * that may be added in a future commit on explicit request.
 *
 * See also the 'theta uses dp not Face_Area_Vec' note in
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


/* Trace-W^2 squared cost between two basis components (Wave-CBE Commit 6a,
 * 2026-05-27). Rotationally invariant 3D generalization of the harness
 * 1D form  C = (v_a - v_b)^2 + (sqrt(S_a) - sqrt(S_b))^2  (cbe_cosmology
 * python_harness HARNESS_RESULTS_AND_FINDINGS §2.4 / §4.4).
 *
 * Velocity term: sum_k (v_a_k - v_b_k)^2, same float-op ordering as
 * cbe_cost_v_only. NMOMENTS=4 builds (no stress slots) reduce to the
 * exact velocity-only sum and are byte-identical to cbe_cost_v_only.
 *
 * Stress term [#if NMOMENTS >= 7]: trace-of-covariance form
 * tr(S) = sum_k (T_kk/m - v_k^2), with each diagonal piece floored at 0
 * before summation to absorb pre-SPD-repair numerical noise. The cost
 * contribution is (sqrt(tr(S_a)) - sqrt(tr(S_b)))^2 -- one scalar add,
 * one sqrt per side, no matrix sqrt. Rotationally invariant; collapses
 * to the harness 1D form exactly when only one stress component is
 * meaningful. Diagonal-only sum would be coordinate-frame dependent and
 * was rejected during codex+Phil review 2026-05-27.
 *
 * Full multivariate Gaussian Wasserstein-2 (needs matrix sqrt of
 * Sigma_a^{1/2} Sigma_b Sigma_a^{1/2}) is NOT this function; if a
 * comparator is ever wanted, it lands as a separate cbe_cost_w2_full
 * symbol and selector.
 *
 * Unused in C6a (no callers yet); SSOT pair-matching builder in C6b
 * routes to this via the CBE_PAIRING_COST compile-time selector. */
KOKKOS_INLINE_FUNCTION
double cbe_cost_trace_w2(const double moments_a[CBE_INTEGRATOR_NMOMENTS],
                         const double moments_b[CBE_INTEGRATOR_NMOMENTS])
{
    double inv_a = 1.0 / DMAX(moments_a[0], MIN_REAL_NUMBER);
    double inv_b = 1.0 / DMAX(moments_b[0], MIN_REAL_NUMBER);
    double v_a[3], v_b[3];
    double c = 0;
    for(int k=0; k<3; k++) {
        v_a[k] = moments_a[k+1] * inv_a;
        v_b[k] = moments_b[k+1] * inv_b;
        double dv = v_a[k] - v_b[k];
        c += dv * dv;
    }
#if (CBE_INTEGRATOR_NMOMENTS >= 7)
    double tr_S_a = 0, tr_S_b = 0;
    for(int k=0; k<3; k++) {
        double S_a_kk = moments_a[4+k] * inv_a - v_a[k]*v_a[k];
        double S_b_kk = moments_b[4+k] * inv_b - v_b[k]*v_b[k];
        tr_S_a += DMAX(S_a_kk, 0.0);
        tr_S_b += DMAX(S_b_kk, 0.0);
    }
    double dsqrt_S = sqrt(tr_S_a) - sqrt(tr_S_b);
    c += dsqrt_S * dsqrt_S;
#endif
    return c;
}


/* Adaptive free-slot row-fallback transform on an NBASIS x NBASIS cost
 * matrix (Wave-CBE Commit 6a, 2026-05-27). Implements the harness §2.4
 * free-slot principle: when source basis alpha has NO good velocity match
 * on the target side (row min exceeds the adaptive threshold tau =
 * median(C) per call), shrink that row's costs by
 *     target_masses[beta] / (source_masses[alpha] + target_masses[beta] + eps)
 * so empty / lightly-occupied target slots become cheaper destinations.
 * Reduces the perturbation-degrading "overwrite dominant stream" cost and
 * captures the harness Test 3.9 result (one-sided-theta + free_slot_w2
 * preserves the +2 perturbation +78% vs plain w2 or v_only and halves
 * KE drift).
 *
 * Direction is asymmetric. Caller passes (source_masses, target_masses)
 * matching the cost-matrix BUILD direction. For an a->b matrix
 *   C[alpha_on_a][beta_on_b] = cost(Q_a[alpha], Q_b[beta]):
 *     source_masses[i] = Q_a[i][0]        // mass density of a-side basis i
 *     target_masses[j] = Q_b[j][0]        // mass density of b-side basis j
 * The b->a direction (separately built C') requires its own free-slot
 * call with source = Q_b masses, target = Q_a masses. It is NOT a
 * transpose of this transform.
 *
 * fired_count_inout is NULLABLE. When non-null, increments by 1 per
 * alpha-row where the tau test triggered. Used at the flux call site
 * only (in C6c); gradient and BJ-limiter call sites pass NULL since
 * pre-pass matching shares the SSOT helper but is not a flux-pairing
 * decision. The diagnostic semantic: SUM over directional basis rows
 * for which the free-slot fallback transformed a cost-matrix row during
 * flux pairing. Each flux face evaluation builds TWO cost matrices
 * (a->b and b->a); each can fire on up to NBASIS rows, so per-face
 * increment is bounded by 2*NBASIS.
 *
 * tau is computed by insertion-sorting a flat copy of all NBASIS^2
 * entries on the stack (NBASIS <= 8 -> at most 64 doubles; no
 * allocations, no library calls, Kokkos-portable).
 *
 * Unused in C6a (no callers yet); SSOT pair-matching builder in C6b
 * routes to this via the CBE_PAIRING_USE_FREE_SLOT compile-time
 * selector (off in C6b temporary defaults, on in C6c final). */
KOKKOS_INLINE_FUNCTION
void cbe_apply_free_slot_fallback(
    double C[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NBASIS],
    const double source_masses[CBE_INTEGRATOR_NBASIS],
    const double target_masses[CBE_INTEGRATOR_NBASIS],
    int *fired_count_inout)
{
    const int N  = CBE_INTEGRATOR_NBASIS;
    const int NN = CBE_INTEGRATOR_NBASIS * CBE_INTEGRATOR_NBASIS;
    double flat[CBE_INTEGRATOR_NBASIS * CBE_INTEGRATOR_NBASIS];
    int idx = 0;
    for(int m=0; m<N; m++) {
        for(int p=0; p<N; p++) flat[idx++] = C[m][p];
    }
    /* insertion sort -- stable, in-place, branch-friendly at NN<=64. */
    for(int i=1; i<NN; i++) {
        double x = flat[i];
        int j = i;
        while(j > 0 && flat[j-1] > x) { flat[j] = flat[j-1]; j--; }
        flat[j] = x;
    }
    double tau = (NN & 1) ? flat[NN/2]
                          : 0.5 * (flat[NN/2 - 1] + flat[NN/2]);

    for(int m=0; m<N; m++) {
        double row_min = MAX_REAL_NUMBER;
        for(int p=0; p<N; p++) {
            if(C[m][p] < row_min) row_min = C[m][p];
        }
        if(row_min > tau) {
            for(int p=0; p<N; p++) {
                double denom = source_masses[m] + target_masses[p] + MIN_REAL_NUMBER;
                C[m][p] *= target_masses[p] / denom;
            }
            if(fired_count_inout) { (*fired_count_inout)++; }
        }
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
 * Piecewise-linear monotonically DECREASING in v_F_n (each active term
 * loses magnitude as v_F_n rises): R(v_F_lo) >= 0 >= R(v_F_hi) when the
 * bracket spans the unique zero. Bisection is sufficient (codex's
 * "boring, device-safe" pick; no Brent edge cases on Kokkos device
 * headers). Hand-rolled brent can swap in behind the same interface
 * later if profiling shows root-find cost matters.
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
 * portability. Scale-aware termination: stop when (hi-lo) drops below
 * tol_abs + tol_rel * max(|lo|,|hi|), so high-velocity (cosmological)
 * brackets reach machine-eps relative precision rather than absolute
 * 1e-12. Iteration cap = 60 (matches harness brentq xtol/rtol scale).
 * NO silent midpoint -- on bracket failure the caller's analytic
 * fallback v_F is used and bracket_fail_count is incremented. */
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
    const double tol_abs = 1e-12;
    const double tol_rel = 1e-12;
    for(int it=0; it<60; it++) {
        double scale = DMAX(fabs(lo), fabs(hi));
        if((hi - lo) <= tol_abs + tol_rel * scale) break;
        double mid = 0.5 * (lo + hi);
        double f_mid = cbe_face_mass_residual_per_unit_area(mid, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
        if(f_lo * f_mid <= 0) { hi = mid; f_hi = f_mid; }
        else                  { lo = mid; f_lo = f_mid; }
    }
    return 0.5 * (lo + hi);
}


/* --------------------------------------------------------------------------
 * Wave-CBE Commit 4 (2026-05-25) — SSOT conversion from stored basis-frame
 * moments to "flux-frame" Q. The pre-Commit-4 path inlined this in
 * cbe_integrator_flux_compute_pair (cosmology factor + velocity boost at
 * flux_functions.h:45-76, then a hardcoded ψ=0.5 prefactor downstream).
 * Commit 4 hoists the conversion into one helper used by BOTH the
 * gradient pass (when it lands) and the per-pair reconstruction so they
 * cannot drift apart on cf_a3inv / cf_atime / NMOMENTS handling.
 *
 * Output (codex 2026-05-25 wording correction: in this codebase
 * `cf_a3inv * density_code` = physical density in code units, NOT a
 * comoving-frame density — the `a3inv` factor turns the comoving-volume
 * cell-integrated U into a physical-frame density.):
 *   Q[m][0]    = physical-frame mass density of basis m   (code units)
 *   Q[m][1..3] = physical-frame momentum density          = Q[m][0]*v_phys
 *   Q[m][4..9] = physical-frame stress density (only when NMOMENTS==10;
 *                the 3D-second-moment fence in precompiler_logic.h
 *                ensures the 3D layout [4]/[5]/[6]=diag,
 *                [7]/[8]/[9]=off-diag holds).
 *
 * "Flux-frame" = physical-frame momentum baked in (v_phys = Vel/cf_atime
 * folded into the [1..3] slots so do_cbe_flux_computation's vsig matches
 * v_phys directly). NOT a generic U/V conversion. */
KOKKOS_INLINE_FUNCTION
void cbe_build_flux_frame_Q_from_stored_moments(
    const double U[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
    const double Vel_code[3],
    double V_i,
    double cf_a3inv,
    double cf_atime,
    double Q_out[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS])
{
    double inv_V = 1.0 / DMAX(V_i, MIN_REAL_NUMBER);
    double inv_a = 1.0 / cf_atime;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
            Q_out[m][k] = U[m][k] * inv_V * cf_a3inv;
        }
        /* Velocity boost on the momentum slots [1..3]. Mirrors flux_functions.h
         * pre-Commit-4 lines 71-74. */
        for(int k=1; k<4 && k<CBE_INTEGRATOR_NMOMENTS; k++) {
            Q_out[m][k] += Q_out[m][0] * Vel_code[k-1] * inv_a;
        }
    }
}


/* Wave-CBE Commit 5 (2026-05-26) — symmetric 3x3 SPD projection of the
 * stress block of a CBE basis-moment row. Operates in-place on the 6-slot
 * tensor S = [Sxx, Syy, Szz, Sxy, Sxz, Syz] (= moment slots [4..9]).
 * Returns true iff S was modified (eigenvalue floor fired or degenerate-
 * input fallback). *dT_out (when non-null) is assigned trace_after -
 * trace_before; >= 0 by construction (positive eigenvalue floor).
 *
 * Algorithm: 6 fixed sweeps of symmetric 3x3 Jacobi eigendecomposition
 * (Press et al. "Numerical Recipes" formulation; off-diagonals fall to
 * <~1e-12 within 4-5 sweeps for symmetric 3x3, 6 sweeps for safety
 * margin). Eigenvalues are clamped to lambda_floor = CBE_SPD_RELATIVE_FLOOR
 * * max(trace_before, MIN_REAL_NUMBER), then S is reconstructed as
 * V * diag(lambda) * V^T.
 *
 * Degenerate-input policy (codex 2026-05-26): if trace_before is non-finite
 * or strictly non-positive, write isotropic S = MIN_REAL_NUMBER * I and set
 * *dT_out = trace_after (NOT trace_after - trace_before, to keep NaN out
 * of the diagnostic accumulator). Returns true.
 *
 * dP: NOT touched. SPD repair operates only on stress slots; momentum
 * slots [1..3] are not arguments. Caller documents dP=0.0 as an invariant.
 *
 * Used at TWO call sites with identical math (SSOT):
 *  - Cell-state drift-kick repair (do_cbe_drift_kick_kernel, replaces the
 *    Wave-CBE Commit 4-era if(2==2) 1D-collapse stopgap).
 *  - Face-state Q-clamp (cbe_clamp_face_Q, S-tensor portion).
 * Both call sites gate on CBE_INTEGRATOR_NMOMENTS >= 10 since the helper
 * assumes a full 3D second-moment tensor with off-diagonal slots present.
 */
KOKKOS_INLINE_FUNCTION
bool cbe_spd_repair_S3x3(double S[6], double *dT_out)
{
    /* Layout: [0]=Sxx, [1]=Syy, [2]=Szz, [3]=Sxy, [4]=Sxz, [5]=Syz. */
    double trace_before = S[0] + S[1] + S[2];

    /* Degenerate-input fallback: ANY non-finite slot, or non-finite /
     * non-positive trace -> isotropic tiny SPD. Checking the off-diagonals
     * is load-bearing: if Sxy/Sxz/Syz are NaN but the diagonal trace is
     * finite-positive, Jacobi propagates NaN through the eigenvalues,
     * `lambda[k] < lambda_floor` evaluates false for NaN, the floor never
     * fires, and non-finite stress survives into downstream consumers.
     * Codex 2026-05-26. */
    bool all_finite = true;
    for(int q = 0; q < 6; q++) { if(!isfinite(S[q])) { all_finite = false; break; } }
    if(!all_finite || !isfinite(trace_before) || trace_before <= 0) {
        S[0] = MIN_REAL_NUMBER; S[1] = MIN_REAL_NUMBER; S[2] = MIN_REAL_NUMBER;
        S[3] = 0.0; S[4] = 0.0; S[5] = 0.0;
        if(dT_out) *dT_out = 3.0 * MIN_REAL_NUMBER;
        return true;
    }

    /* Full 3x3 symmetric matrix A + identity eigenvector matrix V. */
    double A[3][3] = {{S[0], S[3], S[4]},
                      {S[3], S[1], S[5]},
                      {S[4], S[5], S[2]}};
    double V[3][3] = {{1.0, 0.0, 0.0},
                      {0.0, 1.0, 0.0},
                      {0.0, 0.0, 1.0}};

    /* 6 fixed Jacobi sweeps over the three off-diagonal positions. */
    for(int sweep = 0; sweep < 6; sweep++) {
        for(int pq = 0; pq < 3; pq++) {
            int p, q;
            if     (pq == 0) { p = 0; q = 1; }
            else if(pq == 1) { p = 0; q = 2; }
            else             { p = 1; q = 2; }
            double Apq = A[p][q];
            if(fabs(Apq) < 1.0e-300) continue;
            double theta = (A[q][q] - A[p][p]) / (2.0 * Apq);
            double t;
            if(fabs(theta) > 1.0e15) {
                t = 0.5 / theta;
            } else {
                double sgn = (theta >= 0) ? 1.0 : -1.0;
                t = sgn / (fabs(theta) + sqrt(theta*theta + 1.0));
            }
            double c   = 1.0 / sqrt(1.0 + t*t);
            double s   = t * c;
            double tau = s / (1.0 + c);
            /* Rotate A[p][p], A[q][q], A[p][q]. */
            A[p][p] -= t * Apq;
            A[q][q] += t * Apq;
            A[p][q] = 0.0; A[q][p] = 0.0;
            /* Rotate the remaining row/column r (the one not in {p,q}). */
            for(int r = 0; r < 3; r++) {
                if(r == p || r == q) continue;
                double Arp = A[r][p];
                double Arq = A[r][q];
                A[r][p] = Arp - s * (Arq + tau * Arp);
                A[r][q] = Arq + s * (Arp - tau * Arq);
                A[p][r] = A[r][p];
                A[q][r] = A[r][q];
            }
            /* Rotate eigenvector matrix V. */
            for(int r = 0; r < 3; r++) {
                double Vrp = V[r][p];
                double Vrq = V[r][q];
                V[r][p] = Vrp - s * (Vrq + tau * Vrp);
                V[r][q] = Vrq + s * (Vrp - tau * Vrq);
            }
        }
    }

    /* Eigenvalues = diagonal of converged A. Floor relative to trace_pos. */
    double lambda[3] = {A[0][0], A[1][1], A[2][2]};
    double trace_pos    = DMAX(trace_before, MIN_REAL_NUMBER);
    double lambda_floor = CBE_SPD_RELATIVE_FLOOR * trace_pos;
    bool floored = false;
    for(int k = 0; k < 3; k++) {
        if(lambda[k] < lambda_floor) { lambda[k] = lambda_floor; floored = true; }
    }
    if(!floored) {
        /* Already SPD within floor — no modification, no diagnostic increment. */
        if(dT_out) *dT_out = 0.0;
        return false;
    }

    /* Reconstruct S = V * diag(lambda) * V^T (symmetric, store upper). */
    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for(int i = 0; i < 3; i++) {
        for(int j = i; j < 3; j++) {
            double sum = 0.0;
            for(int k = 0; k < 3; k++) sum += lambda[k] * V[i][k] * V[j][k];
            M[i][j] = sum; M[j][i] = sum;
        }
    }
    S[0] = M[0][0]; S[1] = M[1][1]; S[2] = M[2][2];
    S[3] = M[0][1]; S[4] = M[0][2]; S[5] = M[1][2];

    /* dT >= 0 by construction (eigenvalue floor only adds to the trace),
     * but enforce defensively: roundoff in the V * diag(lambda) * V^T
     * accumulation can produce a tiny negative residual when the floor
     * fires on only one eigenvalue; clamp to 0 to preserve the col-8
     * invariant. Codex 2026-05-26. */
    double dT = (M[0][0] + M[1][1] + M[2][2]) - trace_before;
    if(!isfinite(dT) || dT < 0.0) dT = 0.0;
    if(dT_out) *dT_out = dT;
    return true;
}


/* Density-only face-Q clamp + counter (codex 2026-05-25 #6 + #5). For each
 * basis with Q_face[m][0] <= MIN_REAL_NUMBER, zero the ENTIRE basis row.
 * Rationale: leaving nonzero momentum/stress at zero density would crash
 * do_cbe_flux_computation's m_inv = 1/moments[0] divide. Zeroing the row
 * marks the basis inactive at this face -- the downstream cbe_face_K_and_vn
 * helper gives it K=0, v_n=0 so it contributes nothing to the residual or
 * the flux loop.
 *
 * Wave-CBE Commit 5 (2026-05-26): face-state SPD enforcement on the
 * stress block of each rho-active basis row, via cbe_spd_repair_S3x3.
 * S_clamp_count increments once per basis row whose stress block was
 * modified (eigenvalue floor or degenerate-input fallback). Gated on
 * CBE_INTEGRATOR_NMOMENTS >= 10 since slots [4..9] only exist there. */
KOKKOS_INLINE_FUNCTION
void cbe_clamp_face_Q(
    double Qface[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
    long long *rho_clamp_count,
    long long *S_clamp_count)
{
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(Qface[m][0] <= MIN_REAL_NUMBER) {
            for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) Qface[m][k] = 0.0;
            if(rho_clamp_count) (*rho_clamp_count)++;
        }
#if (CBE_INTEGRATOR_NMOMENTS >= 10)
        else {
            double dT_dump = 0.0;
            if(cbe_spd_repair_S3x3(&Qface[m][4], &dT_dump) && S_clamp_count) {
                (*S_clamp_count)++;
            }
        }
#endif
    }
}


/* Guarded face-Q -> (K[m], v_alpha_n[m]) construction for the root-find +
 * theta gate (codex 2026-05-25 #5). On rows with Qface[m][0] > eps, this is
 * the standard v = momentum/density projection onto Ahat; on clamped-inactive
 * rows (Qface[m][0]==0 after cbe_clamp_face_Q), it returns K=0, v_n=0 so
 * the residual function and the flux loop both naturally skip the basis. */
KOKKOS_INLINE_FUNCTION
void cbe_face_K_and_vn_from_Q(
    const double Qface[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
    const double Ahat[3],
    double K[CBE_INTEGRATOR_NBASIS],
    double v_alpha_n[CBE_INTEGRATOR_NBASIS])
{
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(Qface[m][0] > MIN_REAL_NUMBER) {
            double inv_Q0 = 1.0 / Qface[m][0];
            v_alpha_n[m] = (Qface[m][1]*Ahat[0] + Qface[m][2]*Ahat[1] + Qface[m][3]*Ahat[2]) * inv_Q0;
            K[m]         = Qface[m][0];
        } else {
            v_alpha_n[m] = 0.0;
            K[m]         = 0.0;
        }
    }
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
 * deterministic per particle-ID + timestep + loop-domain salt).
 *
 * Wave-CBE Commit 5 (2026-05-26): the legacy if(2==2) 1D-collapse
 * stopgap is replaced by symmetric 3x3 SPD projection (cbe_spd_repair_S3x3)
 * applied AFTER the existing diagonal/Cauchy-Schwarz/determinant repair
 * chain. *dT_out (nullable) receives the sum over basis components of
 * trace_after - trace_before (>= 0 by construction); caller passes nullptr
 * to skip diagnostic bookkeeping. dP is identically 0 from this kernel
 * (SPD repair only touches stress slots [4..9]). */
KOKKOS_INLINE_FUNCTION
static void do_cbe_drift_kick_kernel(struct particle_data& pi, double dt,
                                     double *dT_out)
{
    double dT_local = 0.0;
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
        /* dt-advance of stress slots, then diagonal lower clamp (valid for
         * both 7-moment diagonal-only and 10-moment full 3D layouts). */
        for(k=4;k<CBE_INTEGRATOR_NMOMENTS;k++) {pi.CBE_basis_moments[j][k] += nfac * (dt*pi.CBE_basis_moments_dt[j][k]);}
        for(k=4;k<7;k++) {if(pi.CBE_basis_moments[j][k] < MIN_REAL_NUMBER) {pi.CBE_basis_moments[j][k]=MIN_REAL_NUMBER;}}
#if (CBE_INTEGRATOR_NMOMENTS >= 10)
        /* Off-diagonal Cauchy-Schwarz + determinant crossnorm + Wave-CBE
         * Commit 5 SPD projection. Only valid for the full 3D second-moment
         * layout — slots [7]/[8]/[9] (Sxy/Sxz/Syz) are absent in 7-moment
         * builds, which rely on the diagonal lower-clamp above for stress
         * regularization. (Codex 2026-05-26: previously this whole block
         * was gated on >4, an out-of-bounds access for 7-moment builds.) */
        double eps_tmp = 1.e-8;
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
        /* Wave-CBE Commit 5 (2026-05-26): symmetric 3x3 SPD projection on
         * (Sxx, Syy, Szz, Sxy, Sxz, Syz) as the final pass after the
         * existing diagonal-lower-clamp + Cauchy-Schwarz + determinant
         * crossnorm chain. Replaces the if(2==2) 1D-collapse stopgap.
         * Diagnostic accumulation gated on dT_out so production builds
         * pay zero bookkeeping overhead (codex 2026-05-26 Strong fix). */
        if(dT_out) {
            double dT_basis = 0.0;
            (void)cbe_spd_repair_S3x3(&pi.CBE_basis_moments[j][4], &dT_basis);
            dT_local += dT_basis;
        } else {
            (void)cbe_spd_repair_S3x3(&pi.CBE_basis_moments[j][4], nullptr);
        }
#endif
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
    if(dT_out) *dT_out = dT_local;
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
