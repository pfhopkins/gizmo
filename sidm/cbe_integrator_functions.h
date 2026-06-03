/* cbe_integrator_functions.h — GPU-callable CBE (Collisionless Boltzmann
 * Equation) per-basis flux + face helpers + drift-kick. Pure math with no
 * global state; called both from the CPU tree-walk path (via
 * cbe_integrator_flux_functions.h) and from the AGSForce GPU kernel as
 * KOKKOS_INLINE_FUNCTION.
 *
 * SSOT: the single per-basis flux implementation lives here as
 * cbe_flux_hllc_vacuum (Wave-CBE Commit 8, 2026-05-30, replaces the
 * pre-fix do_cbe_flux_computation). There is no CPU duplicate in
 * cbe_integrator.cc; the legacy "mirrors do_cbe_flux_computation()"
 * comment that lived here referenced a now-retired pre-GPU-port copy.
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

/* Dimension-aware momentum-slot accessors (2026-06-02). Stored basis-moment
 * row layout is row[0] = mass, row[1..NUMDIMS] = p_x..p_{NUMDIMS-1}, and
 * (under CBE_INTEGRATOR_SECONDMOMENT) row[1+NUMDIMS..NMOMENTS-1] = stress
 * tensor components. NMOMENTS shrinks in 1D/2D so the y/z momentum slots
 * literally don't exist in the array. These helpers read them as 0 and
 * silently drop the write, matching the hydro convention where Vel[1] /
 * Vel[2] etc. are simply zero in low-D. The 3-vector dot products and
 * vector intermediates downstream (dp[3], Vel[3], Face_Area_Vec[3])
 * remain 3-wide so the same flux/update math compiles for any NUMDIMS.
 * Compile-time NUMDIMS branch -> zero overhead at NUMDIMS=3. */
KOKKOS_INLINE_FUNCTION double cbe_basis_p_r(const double *row, int k) {
    return (k < NUMDIMS) ? row[1 + k] : 0.0;
}
KOKKOS_INLINE_FUNCTION void cbe_basis_p_w(double *row, int k, double val) {
    if(k < NUMDIMS) row[1 + k] = val;
}
KOKKOS_INLINE_FUNCTION void cbe_basis_p_a(double *row, int k, double val) {
    if(k < NUMDIMS) row[1 + k] += val;
}
KOKKOS_INLINE_FUNCTION void cbe_basis_p_load_3(const double *row, double p3[3]) {
    p3[0] = cbe_basis_p_r(row, 0);
    p3[1] = cbe_basis_p_r(row, 1);
    p3[2] = cbe_basis_p_r(row, 2);
}
KOKKOS_INLINE_FUNCTION void cbe_basis_p_store_3(double *row, const double p3[3]) {
    cbe_basis_p_w(row, 0, p3[0]);
    cbe_basis_p_w(row, 1, p3[1]);
    cbe_basis_p_w(row, 2, p3[2]);
}
KOKKOS_INLINE_FUNCTION void cbe_basis_v_load_3(const double *row, double v3[3]) {
    /* v = p/m for the basis. Returns zeros if m <= 0. */
    const double m = row[0];
    const double inv_m = (m > 0) ? 1.0 / m : 0.0;
    v3[0] = cbe_basis_p_r(row, 0) * inv_m;
    v3[1] = cbe_basis_p_r(row, 1) * inv_m;
    v3[2] = cbe_basis_p_r(row, 2) * inv_m;
}

/* Symmetric stress-slot helpers (Phil 2026-06-02 — added now so the
 * SECONDMOMENT path can be unfenced for arbitrary D later without
 * rewriting the hot path). Layout per the existing convention:
 *   1D NMOMENTS=3 : [m, p_x, T_xx]                                   -> NSTRESS=1
 *   2D NMOMENTS=6 : [m, p_x, p_y, T_xx, T_yy, T_xy]                  -> NSTRESS=3
 *   3D NMOMENTS=10: [m, p_x, p_y, p_z, T_xx, T_yy, T_zz, T_xy, T_xz, T_yz] -> NSTRESS=6
 * Diagonals occupy 1+NUMDIMS..1+2*NUMDIMS-1; off-diagonals (upper triangle
 * in lex order (a,b) with a<b) follow. cbe_T_idx returns the slot index
 * for symmetric T_{a,b}; cbe_basis_T_r/_w guard the same way the
 * momentum accessors do. SECONDMOMENT non-3D still fenced today; these
 * helpers exist so the call sites that use them are unfence-safe. */
#define CBE_NSTRESS (NUMDIMS * (NUMDIMS + 1) / 2)
KOKKOS_INLINE_FUNCTION int cbe_T_idx(int a, int b) {
    if(a > b) { int tmp = a; a = b; b = tmp; }
    if(a == b) return 1 + NUMDIMS + a;
    const int off = a * (2 * NUMDIMS - a - 1) / 2 + (b - a - 1);
    return 1 + 2 * NUMDIMS + off;
}
KOKKOS_INLINE_FUNCTION double cbe_basis_T_r(const double *row, int a, int b) {
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    return row[cbe_T_idx(a, b)];
#else
    (void)row; (void)a; (void)b;
    return 0.0;
#endif
}
KOKKOS_INLINE_FUNCTION void cbe_basis_T_w(double *row, int a, int b, double val) {
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    row[cbe_T_idx(a, b)] = val;
#else
    (void)row; (void)a; (void)b; (void)val;
#endif
}
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
    /* 3-vector velocity diff; missing components (k>=NUMDIMS) are 0 via
     * cbe_basis_p_r — same dimension-agnostic algebra as hydro. */
    for(int k=0; k<3; k++) {
        double dv = cbe_basis_p_r(moments_a, k) * inv_a - cbe_basis_p_r(moments_b, k) * inv_b;
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
    /* 3-vector velocity diff; missing components zero-padded. */
    for(int k=0; k<3; k++) {
        v_a[k] = cbe_basis_p_r(moments_a, k) * inv_a;
        v_b[k] = cbe_basis_p_r(moments_b, k) * inv_b;
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


/* SSOT pair-matching builder (Wave-CBE Commit 6b, 2026-05-27). Single
 * canonical helper used by the flux body, gradient LSQ accumulator, and
 * BJ limiter to produce basis-pair matchings. Replaces three open-coded
 * (cost-matrix build + greedy assignment) patterns; after wiring, all
 * three call sites are reduced to one call to this helper.
 *
 * Cost: dispatched on the compile-time selector CBE_PAIRING_COST. C6b
 * temporary default = CBE_COST_V_ONLY (byte-compatible with pre-C6b);
 * C6c flips to CBE_COST_TRACE_W2 per harness §4.4.
 *
 * Free-slot: applied iff the compile-time selector
 * CBE_PAIRING_USE_FREE_SLOT is 1. C6b temporary default = 0
 * (byte-compatible); C6c flips to 1 per harness §4.4. Direction is
 * asymmetric: the a->b pass uses source = Q_a masses, target = Q_b
 * masses; the b->a pass (only built when alpha_of_beta_for_b is
 * non-null) builds a fresh cost matrix and uses source = Q_b masses,
 * target = Q_a masses. The b->a matrix is NOT the transpose of the a->b
 * matrix when free-slot is on, because free-slot transforms each row
 * with the directional mass ratio.
 *
 * Assignment: cbe_assign_outgoing_greedy (one-sided-nearest per side).
 * CBE_PAIRING_ASSIGN sentinel exists so a future Hungarian comparator
 * (not part of C6) can be added cleanly; only CBE_ASSIGN_GREEDY is
 * supported in C6.
 *
 * Outputs:
 *   beta_of_alpha_for_a[m] = b-side basis matched as a's outgoing target
 *                            for a-side basis m. ALWAYS written.
 *   alpha_of_beta_for_b[n] = a-side basis matched as b's outgoing target
 *                            for b-side basis n. Optional -- pass NULL if
 *                            only one direction is needed (gradient/BJ
 *                            limiter call sites). Flux passes a real
 *                            pointer (both directions needed).
 *   free_slot_fired_count_inout: nullable counter (see
 *                            cbe_apply_free_slot_fallback). Flux call
 *                            site (in C6c) passes &out.cbe_pairing_free_
 *                            slot_count; gradient and BJ-limiter call
 *                            sites pass NULL since pre-pass matching is
 *                            not a flux-pairing decision. */
KOKKOS_INLINE_FUNCTION
void cbe_build_pair_matching(
    const double Q_a[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
    const double Q_b[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
    int beta_of_alpha_for_a[CBE_INTEGRATOR_NBASIS],
    int alpha_of_beta_for_b[CBE_INTEGRATOR_NBASIS],
    int *free_slot_fired_count_inout)
{
    const int N = CBE_INTEGRATOR_NBASIS;

    /* a->b cost matrix. */
    double C_ab[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NBASIS];
    for(int m=0; m<N; m++) {
        for(int n=0; n<N; n++) {
#if (CBE_PAIRING_COST == CBE_COST_TRACE_W2)
            C_ab[m][n] = cbe_cost_trace_w2(Q_a[m], Q_b[n]);
#else
            C_ab[m][n] = cbe_cost_v_only(Q_a[m], Q_b[n]);
#endif
        }
    }
#if CBE_PAIRING_USE_FREE_SLOT
    {
        double src_masses[CBE_INTEGRATOR_NBASIS];
        double tgt_masses[CBE_INTEGRATOR_NBASIS];
        for(int m=0; m<N; m++) {
            src_masses[m] = Q_a[m][0];
            tgt_masses[m] = Q_b[m][0];
        }
        cbe_apply_free_slot_fallback(C_ab, src_masses, tgt_masses,
                                     free_slot_fired_count_inout);
    }
#else
    (void)free_slot_fired_count_inout;
#endif
    cbe_assign_outgoing_greedy(C_ab, beta_of_alpha_for_a);

    if(alpha_of_beta_for_b) {
        /* b->a cost matrix (separately built; not a transpose of C_ab
         * because free-slot is direction-asymmetric). */
        double C_ba[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NBASIS];
        for(int m=0; m<N; m++) {
            for(int n=0; n<N; n++) {
#if (CBE_PAIRING_COST == CBE_COST_TRACE_W2)
                C_ba[m][n] = cbe_cost_trace_w2(Q_b[m], Q_a[n]);
#else
                C_ba[m][n] = cbe_cost_v_only(Q_b[m], Q_a[n]);
#endif
            }
        }
#if CBE_PAIRING_USE_FREE_SLOT
        {
            double src_masses[CBE_INTEGRATOR_NBASIS];
            double tgt_masses[CBE_INTEGRATOR_NBASIS];
            for(int m=0; m<N; m++) {
                src_masses[m] = Q_b[m][0];
                tgt_masses[m] = Q_a[m][0];
            }
            cbe_apply_free_slot_fallback(C_ba, src_masses, tgt_masses,
                                         free_slot_fired_count_inout);
        }
#endif
        cbe_assign_outgoing_greedy(C_ba, alpha_of_beta_for_b);
    }
}


/* --------------------------------------------------------------------------
 * SSOT scalar c_x = sqrt(gamma_e * n_hat . S . n_hat), gamma_e = 3.
 * Computed from one basis row of stored raw second moments R = <v(x)v>;
 * central stress is S = R - v(x)v. NMOMENTS=4 (no stress stored) returns 0;
 * NMOMENTS>=7 uses diagonal-only (S_kk = R_kk - v_k^2) since off-diagonal
 * raw moments are absent in 7-moment builds; NMOMENTS>=10 uses the full
 * 3D tensor. Negative n_hat.S.n_hat (should not occur post cbe_clamp_face_Q
 * SPD enforcement; defensive) returns 0.
 *
 * Used by:
 *   - cbe_face_K_and_vn_from_Q -- fills per-basis c_x array for downstream
 *     HLLC residual evaluation and bracket-pad sizing.
 *   - cbe_flux_hllc_vacuum -- gets its scalar c_x from here while still
 *     building the per-basis stress contraction S_n[] locally for the
 *     full-tensor flux.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
double cbe_face_normal_stress_speed_from_Qrow(
    const double moments[CBE_INTEGRATOR_NMOMENTS],
    const double n_hat[3])
{
    if(!(moments[0] > MIN_REAL_NUMBER)) return 0;
    const double inv_rho = 1.0 / moments[0];
    /* Always-3-vector velocity. Missing components zero-padded by the helper
     * so 1D/2D builds use the same 3-vector algebra below; the SECONDMOMENT
     * stress block stays 3D-only via the precompiler_logic.h fence. */
    double v[3]; cbe_basis_v_load_3(moments, v);
    double nSn = 0;
#if (CBE_INTEGRATOR_NMOMENTS >= 10)
    {
        const double R[3][3] = {
            { moments[4]*inv_rho, moments[7]*inv_rho, moments[8]*inv_rho },
            { moments[7]*inv_rho, moments[5]*inv_rho, moments[9]*inv_rho },
            { moments[8]*inv_rho, moments[9]*inv_rho, moments[6]*inv_rho }
        };
        for(int k=0; k<3; k++) {
            double s = 0;
            for(int l=0; l<3; l++) s += (R[k][l] - v[k]*v[l]) * n_hat[l];
            nSn += n_hat[k] * s;
        }
    }
#elif (CBE_INTEGRATOR_NMOMENTS >= 7)
    {
        const double S_diag[3] = {
            moments[4]*inv_rho - v[0]*v[0],
            moments[5]*inv_rho - v[1]*v[1],
            moments[6]*inv_rho - v[2]*v[2]
        };
        for(int k=0; k<3; k++) nSn += S_diag[k] * n_hat[k] * n_hat[k];
    }
#endif
    return (nSn > 0) ? sqrt(3.0 * nSn) : 0;
}


/* --------------------------------------------------------------------------
 * SSOT HLLC vacuum mass-flux density per unit face area, per basis. Branches
 * on the source-side outward normal velocity u_out exactly as the full
 * flux solver (cbe_flux_hllc_vacuum) does for its mass slot; extracting it
 * lets the v_F root-find residual and the deposited flux use bit-identical
 * branching, which is the requirement of the strict-root-found policy
 * (basis-summed F_m at v_F == 0 implies cell-summed mass conservation).
 *
 *   u_out >=  c_x         -> rho * u_out         (cold F0 supersonic)
 *  -c_x/3 <  u_out <  c_x -> rho * (3 u_out + c_x) / 4   (subsonic vacuum)
 *   u_out <= -c_x/3       -> 0                   (vacuum, no outflow)
 *
 * Cold limit c_x -> 0: F0 branch for u_out > 0, F = 0 for u_out < 0
 * (recovers cold-F0 cleanly when no stress is stored or n.S.n is zero).
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
double cbe_hllc_mass_flux_per_unit_area(double rho, double u_out, double c_x)
{
    if(u_out >= c_x)              return rho * u_out;
    else if(u_out > -c_x / 3.0)   return 0.25 * rho * (3.0 * u_out + c_x);
    else                          return 0;
}


/* --------------------------------------------------------------------------
 * Per-face root-found face-normal velocity v_F_n (Wave-CBE Commit 3,
 * 2026-05-25; HLLC update Wave-CBE Commit 9, 2026-05-30). Returns net
 * per-unit-area face mass flux at trial v_F_n, summing the HLLC vacuum
 * mass-flux contribution from every basis on both sides.
 *
 * Sign convention matches the deposited flux (Wave-CBE Commit 8): i-side
 * basis outflow (u_out_i = v_alpha_n_i - v_F_n) carries mass in +A_hat,
 * j-side basis outflow (u_out_j = v_F_n - v_alpha_n_j) carries mass in
 * -A_hat. Net flux in +A_hat is therefore
 *
 *   r(v_F_n) = sum_m  F_m_HLLC(K_i[m], u_out_i, c_x_i[m])
 *            - sum_m  F_m_HLLC(K_j[m], u_out_j, c_x_j[m]).
 *
 * Monotone non-increasing in v_F_n (each per-basis F_m_HLLC is monotone
 * non-decreasing in u_out: F0 slope rho, F1 slope 3 rho / 4, F=0 slope 0;
 * u_out_i decreases as v_F_n rises; u_out_j increases as v_F_n rises);
 * bisection in cbe_face_solve_v_F_normal converges to the unique zero.
 * K==0 (rho-clamped-inactive) rows contribute 0 by construction.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
double cbe_face_mass_residual_per_unit_area(
    double v_F_n,
    const double v_alpha_n_i[CBE_INTEGRATOR_NBASIS],
    const double v_alpha_n_j[CBE_INTEGRATOR_NBASIS],
    const double K_i[CBE_INTEGRATOR_NBASIS],
    const double K_j[CBE_INTEGRATOR_NBASIS],
    const double c_x_i[CBE_INTEGRATOR_NBASIS],
    const double c_x_j[CBE_INTEGRATOR_NBASIS])
{
    double r = 0;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        const double u_out_i = v_alpha_n_i[m] - v_F_n;
        const double u_out_j = v_F_n - v_alpha_n_j[m];
        r += cbe_hllc_mass_flux_per_unit_area(K_i[m], u_out_i, c_x_i[m]);
        r -= cbe_hllc_mass_flux_per_unit_area(K_j[m], u_out_j, c_x_j[m]);
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
    const double c_x_i[CBE_INTEGRATOR_NBASIS],
    const double c_x_j[CBE_INTEGRATOR_NBASIS],
    double v_F_lo, double v_F_hi,
    double fallback_v_F_n,
    int *bracket_ok_out)
{
    double lo = v_F_lo, hi = v_F_hi;
    double f_lo = cbe_face_mass_residual_per_unit_area(lo, v_alpha_n_i, v_alpha_n_j, K_i, K_j, c_x_i, c_x_j);
    double f_hi = cbe_face_mass_residual_per_unit_area(hi, v_alpha_n_i, v_alpha_n_j, K_i, K_j, c_x_i, c_x_j);
    int bracketed = (f_lo * f_hi <= 0) ? 1 : 0;
    for(int widen = 0; widen < 4 && !bracketed; widen++) {
        double mid  = 0.5 * (lo + hi);
        double half = 4.0 * 0.5 * (hi - lo);
        lo = mid - half; hi = mid + half;
        f_lo = cbe_face_mass_residual_per_unit_area(lo, v_alpha_n_i, v_alpha_n_j, K_i, K_j, c_x_i, c_x_j);
        f_hi = cbe_face_mass_residual_per_unit_area(hi, v_alpha_n_i, v_alpha_n_j, K_i, K_j, c_x_i, c_x_j);
        if(f_lo * f_hi <= 0) { bracketed = 1; break; }
    }
    if(!bracketed) {
        *bracket_ok_out = 0;
        return fallback_v_F_n;
    }
    *bracket_ok_out = 1;
    /* Tightened tol_rel to 1e-14 (Wave-CBE Commit 9): the strict-root-found
     * policy means the per-face mass residual at v_F is bounded by
     * |F_m_HLLC'(v_F)| * (hi - lo), and |F_m'| ~ sum of active basis K's
     * times Face_Area_Norm. For Hernquist v_F bracket scale ~200 the
     * 1e-12 rel tol left residuals at ~1e-10 (above the col-2 <= 1e-11
     * gate); 1e-14 brings them solidly below. Extra iterations are cheap
     * vs the flux loop. 60-iter cap still allows convergence: 2^60 >> any
     * physically plausible width / 1e-14. */
    const double tol_abs = 1e-12;
    const double tol_rel = 1e-14;
    for(int it=0; it<60; it++) {
        double scale = DMAX(fabs(lo), fabs(hi));
        if((hi - lo) <= tol_abs + tol_rel * scale) break;
        double mid = 0.5 * (lo + hi);
        double f_mid = cbe_face_mass_residual_per_unit_area(mid, v_alpha_n_i, v_alpha_n_j, K_i, K_j, c_x_i, c_x_j);
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
 * folded into the [1..3] slots so cbe_flux_hllc_vacuum's u_out matches
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


/* Wave-CBE Fix #2a (2026-05-30) — symmetric 3x3 SSOT realizability machinery.
 *
 * The three helpers below provide the canonical PSD predicate + projection
 * for CBE basis rows, shared by the Commit-10 cone limiter, the face-state
 * Q-clamp, and the cell-state drift-kick repair. Storage convention: raw
 * moment slots [4..9] hold T_kl = m * R_kl = m * (S_kl + v_k v_l) (raw
 * second-moment density). Central stress S_kl = R_kl - v_k v_l is built
 * locally by the raw-row wrapper before projection.
 *
 * 2a fixes a pre-existing units bug in the prior cbe_spd_repair_S3x3
 * function (Commit 5 introduced; Commit 10 inherited at the face site):
 * both call sites passed raw R slots to a helper that interpreted them
 * as central S. Effective floor was CBE_SPD_RELATIVE_FLOOR * trace(raw R)
 * ~ 1e-12 * m * (trace(S) + v_bulk^2) — a hidden bulk-KE floor on stress.
 * The new raw-row wrapper does the raw <-> central conversion correctly,
 * and the core projection takes an explicit eigenvalue_floor argument so
 * each caller picks its own policy.
 */


/* SSOT 3x3 symmetric PSD test via all principal minors (originally Commit 10
 * Fix #6; moved here in Fix #2a). Returns true iff every principal minor
 * of M is >= -eps_tol * scale, where scale is the natural amplitude for
 * each minor order (trace for diagonal, trace^2 for 2x2, trace^3 for det).
 * eps_tol = 0 gives strict Sylvester-style PSD; eps_tol > 0 absorbs
 * FP-scale near-zero negative roundoff. Cone limiter uses 0 (strict);
 * Fix #2 cell repair uses 1e-12. */
KOKKOS_INLINE_FUNCTION
bool cbe_sym3x3_all_principal_minors_nonneg(const double M[3][3], double eps_tol)
{
    const double trace_pos = DMAX(M[0][0] + M[1][1] + M[2][2], MIN_REAL_NUMBER);
    const double eps_diag  = eps_tol * trace_pos;
    const double eps_2x2   = eps_tol * trace_pos * trace_pos;
    const double eps_det   = eps_tol * trace_pos * trace_pos * trace_pos;
    if(M[0][0] < -eps_diag) return false;
    if(M[1][1] < -eps_diag) return false;
    if(M[2][2] < -eps_diag) return false;
    if(M[0][0]*M[1][1] - M[0][1]*M[0][1] < -eps_2x2) return false;
    if(M[0][0]*M[2][2] - M[0][2]*M[0][2] < -eps_2x2) return false;
    if(M[1][1]*M[2][2] - M[1][2]*M[1][2] < -eps_2x2) return false;
    const double det = M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[1][2])
                     - M[0][1] * (M[0][1]*M[2][2] - M[1][2]*M[0][2])
                     + M[0][2] * (M[0][1]*M[1][2] - M[1][1]*M[0][2]);
    return (det >= -eps_det);
}


/* SSOT realizability predicate for a CBE basis row in raw-slot layout.
 * A row is realizable iff m > 0 AND (NMOMENTS >= 7) M = m*T - p p^T is PSD,
 * where T_kl = U_row[4..9] are the raw second-moment density slots and
 * the PSD test uses cbe_sym3x3_all_principal_minors_nonneg with eps_tol.
 * (Note the M expression: M = m * R_kl * m - p_k p_l = T_kl * m - p_k p_l
 *  with T_kl = m * R_kl already, so M = T_kl * m - p p^T as written.)
 *
 * Consumers: cone limiter (strict, eps_tol = 0); Fix #2b cell repair
 * (FP-scale, eps_tol = CBE_REPAIR_EPS_TOL). */
KOKKOS_INLINE_FUNCTION
bool cbe_basis_row_is_realizable(
    const double Q_row[CBE_INTEGRATOR_NMOMENTS], double eps_tol)
{
    if(!(Q_row[0] > 0) || !isfinite(Q_row[0])) return false;
#if (CBE_INTEGRATOR_NMOMENTS >= 7)
    {
        const double m  = Q_row[0];
        const double p[3] = { Q_row[1], Q_row[2], Q_row[3] };
        double M[3][3];
        M[0][0] = Q_row[4] * m - p[0]*p[0];
        M[1][1] = Q_row[5] * m - p[1]*p[1];
        M[2][2] = Q_row[6] * m - p[2]*p[2];
    #if (CBE_INTEGRATOR_NMOMENTS >= 10)
        M[0][1] = M[1][0] = Q_row[7] * m - p[0]*p[1];
        M[0][2] = M[2][0] = Q_row[8] * m - p[0]*p[2];
        M[1][2] = M[2][1] = Q_row[9] * m - p[1]*p[2];
    #else
        /* NMOMENTS=7: off-diagonal central S absent (Commit 8 convention).
         * Zero M off-diagonals so the full-3x3 PSD helper reduces to the
         * diagonal-only check. */
        M[0][1] = M[1][0] = 0;
        M[0][2] = M[2][0] = 0;
        M[1][2] = M[2][1] = 0;
    #endif
        if(!cbe_sym3x3_all_principal_minors_nonneg(M, eps_tol)) return false;
    }
#endif
    return true;
}


/* CORE: symmetric 3x3 PSD projection of a CENTRAL stress tensor. Operates
 * in-place on the 6-vector S = [Sxx, Syy, Szz, Sxy, Sxz, Syz] (NOT a raw
 * moment row — use cbe_basis_row_project_central_stress_to_PSD for that).
 * Returns true iff S was modified.
 *
 * Algorithm: 6 fixed Jacobi sweeps to diagonalize, clamp eigenvalues to
 * max(lambda_k, eigenvalue_floor), reconstruct S = V * diag(lambda) * V^T.
 *
 * eigenvalue_floor: caller-supplied. Pass 0 for true PSD projection
 * (Fix #2 cell repair); pass CBE_SPD_RELATIVE_FLOOR * trace_central_S_pos
 * for the existing face-clamp floor policy. Fix #2a corrected the
 * pre-existing units bug where this floor was computed from trace(raw R).
 *
 * Degenerate-input fallback: fires ONLY when input is non-finite (NaN/Inf
 * in any S slot). trace_before <= 0 with finite entries is a legitimate
 * central state to project (cold flows naturally give central S near or
 * below zero) — Jacobi handles it, no isotropic-MIN_REAL injection.
 * Codex round 4 explicit. */
KOKKOS_INLINE_FUNCTION
bool cbe_project_central_S_to_PSD(double S[6], double eigenvalue_floor,
                                  double *dT_out)
{
    /* Layout: [0]=Sxx, [1]=Syy, [2]=Szz, [3]=Sxy, [4]=Sxz, [5]=Syz. */
    double trace_before = S[0] + S[1] + S[2];

    /* Degenerate-input fallback (non-finite only). Off-diagonal NaN check
     * is load-bearing per codex 2026-05-26 (Jacobi propagates NaN through
     * eigenvalues silently). */
    bool all_finite = true;
    for(int q = 0; q < 6; q++) { if(!isfinite(S[q])) { all_finite = false; break; } }
    if(!all_finite || !isfinite(trace_before)) {
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

    /* Eigenvalues = diagonal of converged A. Floor with caller-supplied
     * eigenvalue_floor (was: CBE_SPD_RELATIVE_FLOOR * trace_pos baked-in). */
    double lambda[3] = {A[0][0], A[1][1], A[2][2]};
    bool floored = false;
    for(int k = 0; k < 3; k++) {
        if(lambda[k] < eigenvalue_floor) { lambda[k] = eigenvalue_floor; floored = true; }
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

    /* dT enforcement: eigenvalue floor only adds to the trace by
     * construction (>= 0); roundoff in V*diag(lambda)*V^T may produce a
     * tiny negative residual when the floor fires on only one eigenvalue;
     * clamp to 0 to preserve the col-8 invariant. */
    double dT = (M[0][0] + M[1][1] + M[2][2]) - trace_before;
    if(!isfinite(dT) || dT < 0.0) dT = 0.0;
    if(dT_out) *dT_out = dT;
    return true;
}


/* RAW-ROW WRAPPER: takes a CBE raw moment row, converts to central S,
 * projects via cbe_project_central_S_to_PSD, rebuilds raw slots.
 *
 * U_row layout: slots [0..9] = (m, p_x, p_y, p_z, T_xx, T_yy, T_zz,
 * T_xy, T_xz, T_yz) with T_kl = m * R_kl raw second-moment density.
 * Central stress S_kl = R_kl - v_k v_l = T_kl / m - v_k v_l where v = p/m.
 * On modification, rebuilds T_kl_new = m * (S_kl_repaired + v_k v_l).
 *
 * Named explicitly around "raw moment row" to prevent the bug 2a fixes
 * (passing raw slots to a central-S-only API). dT_out is the raw-slot
 * trace delta = m * (central-dT), preserving the dP_sum / dT_sum
 * accumulator semantic Commit 5 wired into cbe_diagnostics col-8.
 *
 * NMOMENTS>=10 only: both current callers (cbe_clamp_face_Q + cell-side
 * drift-kick) gate on >=10 since the central stress 3x3 is only present
 * with the full 3D second-moment layout. NMOMENTS=7 callers (none yet)
 * would need a diagonal-only variant. */
KOKKOS_INLINE_FUNCTION
bool cbe_basis_row_project_central_stress_to_PSD(
    double U_row[CBE_INTEGRATOR_NMOMENTS],
    double eigenvalue_floor,
    double *dT_out)
{
#if (CBE_INTEGRATOR_NMOMENTS >= 10)
    /* Mass-positivity guard: m <= 0 means v = p/m undefined; caller is
     * responsible for rho-clamping upstream (cbe_clamp_face_Q does so via
     * Q_face[m][0] <= MIN_REAL_NUMBER row-zeroing; Fix #2b corrupt-cell
     * guard does so at Step 0). Treat as no-op here. */
    if(!(U_row[0] > 0) || !isfinite(U_row[0])) {
        if(dT_out) *dT_out = 0;
        return false;
    }
    const double m     = U_row[0];
    const double inv_m = 1.0 / m;
    const double v[3]  = { U_row[1]*inv_m, U_row[2]*inv_m, U_row[3]*inv_m };
    /* Build central S in helper's slot order [Sxx, Syy, Szz, Sxy, Sxz, Syz]. */
    double S[6];
    S[0] = U_row[4]*inv_m - v[0]*v[0];
    S[1] = U_row[5]*inv_m - v[1]*v[1];
    S[2] = U_row[6]*inv_m - v[2]*v[2];
    S[3] = U_row[7]*inv_m - v[0]*v[1];
    S[4] = U_row[8]*inv_m - v[0]*v[2];
    S[5] = U_row[9]*inv_m - v[1]*v[2];

    double dT_central = 0;
    bool modified = cbe_project_central_S_to_PSD(S, eigenvalue_floor, &dT_central);

    if(modified) {
        /* Rebuild raw slots: T_kl = m * (S_kl + v_k v_l). */
        U_row[4] = m * (S[0] + v[0]*v[0]);
        U_row[5] = m * (S[1] + v[1]*v[1]);
        U_row[6] = m * (S[2] + v[2]*v[2]);
        U_row[7] = m * (S[3] + v[0]*v[1]);
        U_row[8] = m * (S[4] + v[0]*v[2]);
        U_row[9] = m * (S[5] + v[1]*v[2]);
    }
    if(dT_out) *dT_out = dT_central * m;   /* raw-slot trace delta */
    return modified;
#else
    /* NMOMENTS < 10: no off-diagonal stress slots; caller should not invoke. */
    (void)U_row; (void)eigenvalue_floor;
    if(dT_out) *dT_out = 0;
    return false;
#endif
}


/* COMPAT SHIM (Fix #2a): preserves the OLD pre-2a behavior of
 * cbe_spd_repair_S3x3 for any latent caller not surfaced by the 2a grep.
 * The two known callers (cbe_clamp_face_Q + do_cbe_drift_kick_kernel)
 * are migrated to cbe_basis_row_project_central_stress_to_PSD in 2a.
 * Fix #2b deletes this shim after verifying no other callers remain.
 *
 * Old behavior preserved: trace_before <= 0 (with finite entries) falls
 * to isotropic MIN_REAL_NUMBER * I — this differs from the new core,
 * which projects to PSD via Jacobi. The shim path retains the original
 * fallback for byte-compatibility with any caller depending on it. */
KOKKOS_INLINE_FUNCTION
bool cbe_spd_repair_S3x3(double S[6], double *dT_out)
{
    double trace_before = S[0] + S[1] + S[2];
    bool all_finite = true;
    for(int q = 0; q < 6; q++) { if(!isfinite(S[q])) { all_finite = false; break; } }
    if(!all_finite || !isfinite(trace_before) || trace_before <= 0) {
        S[0] = MIN_REAL_NUMBER; S[1] = MIN_REAL_NUMBER; S[2] = MIN_REAL_NUMBER;
        S[3] = 0.0; S[4] = 0.0; S[5] = 0.0;
        if(dT_out) *dT_out = 3.0 * MIN_REAL_NUMBER;
        return true;
    }
    double trace_pos = DMAX(trace_before, MIN_REAL_NUMBER);
    return cbe_project_central_S_to_PSD(S, CBE_SPD_RELATIVE_FLOOR * trace_pos, dT_out);
}


/* Density-only face-Q clamp + counter (codex 2026-05-25 #6 + #5). For each
 * basis with Q_face[m][0] <= MIN_REAL_NUMBER, zero the ENTIRE basis row.
 * Rationale: leaving nonzero momentum/stress at zero density would crash
 * cbe_flux_hllc_vacuum's inv_rho = 1/moments[0] divide. Zeroing the row
 * marks the basis inactive at this face -- the downstream cbe_face_K_and_vn
 * helper gives it K=0, v_n=0 so it contributes nothing to the residual or
 * the flux loop.
 *
 * Wave-CBE Commit 5 (2026-05-26): face-state SPD enforcement on the
 * stress block of each rho-active basis row.
 *
 * Wave-CBE Fix #2a (2026-05-30): migrated from cbe_spd_repair_S3x3 (which
 * silently treated raw R as if it were central S) to the raw-row wrapper
 * cbe_basis_row_project_central_stress_to_PSD, which correctly converts
 * raw <-> central. The eigenvalue floor is computed from trace(central S)
 * — NOT trace(raw R) which secretly included the bulk-KE contribution
 * v_bulk^2 and acted as a hidden physical floor. Net physics change in
 * cold flows: floor magnitude shrinks dramatically (1e-12 * trace_central_S
 * is much smaller than 1e-12 * (trace_central_S + v_bulk^2) when bulk
 * kinetic dominates). S_clamp_count semantic unchanged.
 *
 * Gated on CBE_INTEGRATOR_NMOMENTS >= 10 since slots [4..9] off-diagonals
 * only exist there. */
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
            /* Compute eigenvalue_floor from central S trace, NOT raw R trace.
             * Mass-positivity guarded above (Qface[m][0] > MIN_REAL_NUMBER). */
            const double m_face = Qface[m][0];
            const double inv_m  = 1.0 / m_face;
            const double v[3] = { Qface[m][1]*inv_m, Qface[m][2]*inv_m, Qface[m][3]*inv_m };
            const double trace_S_central = (Qface[m][4] + Qface[m][5] + Qface[m][6]) * inv_m
                                         - (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            const double trace_S_pos     = DMAX(trace_S_central, MIN_REAL_NUMBER);
            const double eigenvalue_floor = CBE_SPD_RELATIVE_FLOOR * trace_S_pos;
            double dT_dump = 0.0;
            if(cbe_basis_row_project_central_stress_to_PSD(Qface[m], eigenvalue_floor,
                                                            &dT_dump)
               && S_clamp_count) {
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
    double v_alpha_n[CBE_INTEGRATOR_NBASIS],
    double c_x[CBE_INTEGRATOR_NBASIS])
{
    /* Per-basis face state for the HLLC vacuum mass-flux residual + flux
     * loop. K = rho (basis density on the face), v_alpha_n = v . Ahat
     * (normal velocity), c_x = sqrt(gamma_e * Ahat.S.Ahat) (HLLC normal
     * stress speed, gamma_e = 3). Inactive rows (Qface[m][0] <=
     * MIN_REAL_NUMBER post cbe_clamp_face_Q) zero all three outputs so
     * downstream consumers naturally skip them. c_x is computed via the
     * SSOT helper cbe_face_normal_stress_speed_from_Qrow so the wave-speed
     * definition matches what cbe_flux_hllc_vacuum and the residual use. */
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(Qface[m][0] > MIN_REAL_NUMBER) {
            double v_basis[3]; cbe_basis_v_load_3(Qface[m], v_basis);
            v_alpha_n[m] = v_basis[0]*Ahat[0] + v_basis[1]*Ahat[1] + v_basis[2]*Ahat[2];
            K[m]         = Qface[m][0];
            c_x[m]       = cbe_face_normal_stress_speed_from_Qrow(Qface[m], Ahat);
        } else {
            v_alpha_n[m] = 0.0;
            K[m]         = 0.0;
            c_x[m]       = 0.0;
        }
    }
}


KOKKOS_INLINE_FUNCTION
/* Wave-CBE Commit 8 (Fix #1, 2026-05-30) — branched HLLC vacuum one-sided
 * flux. Per-basis flux on one side of a face, in the SOURCE-side outward
 * frame defined by Area_outward.
 *
 * CONTRACT (caller responsibility, NOT verified inside; device code):
 *   - moments[]      Upwind flux-frame Q (cell or face-reconstructed)
 *                    of the SOURCE basis. Layout: [m, p_x, p_y, p_z,
 *                    R_xx, R_yy, R_zz, R_xy, R_xz, R_yz] for NMOMENTS=10;
 *                    [m, p_x, p_y, p_z, R_xx, R_yy, R_zz] for NMOMENTS=7;
 *                    [m, p_x, p_y, p_z] for NMOMENTS=4. Slots [4..9] are
 *                    RAW second moments R_kl = <v_k v_l>, NOT central
 *                    stress S_kl. Central S = R - v⊗v is built locally.
 *   - vface[3]       Face velocity in code units (typically v_F_normal
 *                    times the canonical face unit normal; tangential
 *                    components are silently ignored by this function
 *                    because they cancel in the n_hat contraction).
 *   - Area_outward   SOURCE-side outward face area vector (NOT
 *                    normalized). For a pair sharing one face:
 *                       i-side call: Area_outward = +Face_Area_Vec
 *                       j-side call: Area_outward = -Face_Area_Vec
 *                    The function does NOT infer orientation from
 *                    sign(vsig) — misuse silently inverts the physics.
 *   - fluxes[]       Output, in Area_outward frame, area-integrated
 *                    (units: [Q] × velocity × area).
 *
 * RETURNS: signal speed (|u_out| + c_x) * |Area_outward|. The full HLLC
 * physical wave speed. Caller uses this for CFL and wakeup criteria.
 *
 * BRANCHING: with u_out = (<v> - vface) · n_out and
 * c_x = sqrt(gamma_e * n_out · S · n_out), gamma_e = 3,
 *     u_out >= c_x         -> F0 (cold supersonic outflow)
 *     -c_x/3 < u_out < c_x -> F1 (subsonic vacuum, warm)
 *     u_out <= -c_x/3      -> 0  (no outgoing flux from this basis)
 * F1 prefactor (rho/4)(3 u_out/c_x + 1) goes continuously to F0 at u_out=c_x
 * and to 0 at u_out=-c_x/3. Cold limit S->0: c_x->0, F1 collapses, F0
 * recovers exactly.
 *
 * DEFENSIVE GUARD: if rho is non-positive or non-finite or A_norm_sq <= 0
 * the function zeros the flux and returns 0. Device-safe (no endrun);
 * caller's clamp helpers should already filter these cases. */
KOKKOS_INLINE_FUNCTION
double cbe_flux_hllc_vacuum(const double moments[CBE_INTEGRATOR_NMOMENTS],
                            const double vface[3],
                            const double Area_outward[3],
                            double fluxes[CBE_INTEGRATOR_NMOMENTS])
{
    const double A_norm_sq = Area_outward[0]*Area_outward[0]
                           + Area_outward[1]*Area_outward[1]
                           + Area_outward[2]*Area_outward[2];
    if(!(moments[0] > MIN_REAL_NUMBER) || !isfinite(moments[0]) || !(A_norm_sq > 0)) {
        for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) fluxes[k] = 0;
        return 0;
    }
    const double A_norm  = sqrt(A_norm_sq);
    const double inv_A   = 1.0 / A_norm;
    const double n_hat[3] = { Area_outward[0]*inv_A, Area_outward[1]*inv_A, Area_outward[2]*inv_A };

    const double rho     = moments[0];
    const double inv_rho = 1.0 / rho;
    /* Always-3-vector velocity. Missing components zero-padded so 1D/2D
     * builds use the same 3-vector algebra; SECONDMOMENT stress block
     * stays 3D-only via the precompiler_logic.h fence. */
    double v[3]; cbe_basis_v_load_3(moments, v);
    const double v_n     = v[0]*n_hat[0]    + v[1]*n_hat[1]    + v[2]*n_hat[2];
    const double vF_n    = vface[0]*n_hat[0] + vface[1]*n_hat[1] + vface[2]*n_hat[2];
    const double u_out   = v_n - vF_n;

    /* Central stress contracted with n_hat: S_n_k = (R_kl - v_k v_l) n_hat_l.
     * NMOMENTS=7 has only diagonal raw moments stored, so off-diagonal
     * CENTRAL stress is ABSENT (not zero) — synthesizing -v_k v_l in those
     * slots would invent unphysical off-diagonal stress. Use diagonal-only
     * formula in that branch. Scalar c_x comes from the SSOT helper to
     * keep the wave-speed definition identical to the residual function. */
    double S_n[3] = {0};
#if (CBE_INTEGRATOR_NMOMENTS >= 10)
    {
        const double R[3][3] = {
            { moments[4]*inv_rho, moments[7]*inv_rho, moments[8]*inv_rho },
            { moments[7]*inv_rho, moments[5]*inv_rho, moments[9]*inv_rho },
            { moments[8]*inv_rho, moments[9]*inv_rho, moments[6]*inv_rho }
        };
        for(int k=0; k<3; k++) {
            double s = 0;
            for(int l=0; l<3; l++) s += (R[k][l] - v[k]*v[l]) * n_hat[l];
            S_n[k] = s;
        }
    }
#elif (CBE_INTEGRATOR_NMOMENTS >= 7)
    {
        const double S_diag[3] = {
            moments[4]*inv_rho - v[0]*v[0],
            moments[5]*inv_rho - v[1]*v[1],
            moments[6]*inv_rho - v[2]*v[2]
        };
        for(int k=0; k<3; k++) S_n[k] = S_diag[k] * n_hat[k];
    }
#endif
    const double c_x = cbe_face_normal_stress_speed_from_Qrow(moments, n_hat);

    /* Mass slot via SSOT HLLC helper -- bit-identical to what the v_F
     * root-find residual sums over, so basis-summed F_m at v_F == 0
     * exactly implies cell-summed mass conservation. F = 0 vacuum branch
     * returns short-circuit (no stress / momentum work needed). */
    const double F_m_per_area = cbe_hllc_mass_flux_per_unit_area(rho, u_out, c_x);
    if(F_m_per_area == 0) {
        for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) fluxes[k] = 0;
        return (fabs(u_out) + c_x) * A_norm;
    }
    fluxes[0] = F_m_per_area * A_norm;

    /* prefactor and u_or_c are derived from the helper's branching for
     * use in the momentum + stress slots, which still need the tensor
     * structure that the scalar helper does not return. */
    const double prefactor = (u_out >= c_x) ? rho : 0.25 * rho * (3.0 * u_out / c_x + 1.0);
    const double u_or_c    = (u_out >= c_x) ? u_out : c_x;

    /* Momentum: F_p_k = v_k * F_m + prefactor * |A| * S_n_k. The flux row
     * has only NUMDIMS momentum slots in low-D builds; cbe_basis_p_w silently
     * drops writes to non-existent y/z slots. The S_n[k] math stays 3-vector
     * so the rotation-invariant cold limit (S=0 → fluxes[k+1] = v[k]*F_m)
     * collapses cleanly to the existing 1D code path with v_y=v_z=0. */
    for(int k=0; k<3; k++) {
        cbe_basis_p_w(fluxes, k, v[k] * fluxes[0] + prefactor * A_norm * S_n[k]);
    }

    /* Stress: F_T_kl = R_kl * F_m + prefactor * |A| * (v_k S_n_l + S_n_k v_l).
     * Note the u_or_c-multiplied piece uses RAW R; the cross piece uses
     * central S contracted with n_hat. NMOMENTS=7 has only diagonal slots
     * [4,5,6]; NMOMENTS=10 also has off-diagonal [7,8,9].
     * (Fixes a latent NMOMENTS=7 OOB read in the pre-Commit-8 code.) */
#if (CBE_INTEGRATOR_NMOMENTS >= 10)
    fluxes[4] = moments[4]*inv_rho * fluxes[0] + prefactor * A_norm * 2.0 * v[0] * S_n[0];
    fluxes[5] = moments[5]*inv_rho * fluxes[0] + prefactor * A_norm * 2.0 * v[1] * S_n[1];
    fluxes[6] = moments[6]*inv_rho * fluxes[0] + prefactor * A_norm * 2.0 * v[2] * S_n[2];
    fluxes[7] = moments[7]*inv_rho * fluxes[0] + prefactor * A_norm * (v[0]*S_n[1] + S_n[0]*v[1]);
    fluxes[8] = moments[8]*inv_rho * fluxes[0] + prefactor * A_norm * (v[0]*S_n[2] + S_n[0]*v[2]);
    fluxes[9] = moments[9]*inv_rho * fluxes[0] + prefactor * A_norm * (v[1]*S_n[2] + S_n[1]*v[2]);
#elif (CBE_INTEGRATOR_NMOMENTS >= 7)
    fluxes[4] = moments[4]*inv_rho * fluxes[0] + prefactor * A_norm * 2.0 * v[0] * S_n[0];
    fluxes[5] = moments[5]*inv_rho * fluxes[0] + prefactor * A_norm * 2.0 * v[1] * S_n[1];
    fluxes[6] = moments[6]*inv_rho * fluxes[0] + prefactor * A_norm * 2.0 * v[2] * S_n[2];
#endif

    return (fabs(u_out) + c_x) * A_norm;
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
        /* Momentum slots: indices 1..NUMDIMS only. 1D NMOMENTS=2 has just
         * slot [1]=p_x; 3D NMOMENTS=4/7/10 has [1..3]=p_x,p_y,p_z. */
        for(k=1;k<=NUMDIMS;k++) {pi.CBE_basis_moments[j][k] += nfac * (dt*pi.CBE_basis_moments_dt[j][k] - pi.CBE_basis_moments[j][0]*minv*dmoment[k]);}
#if (CBE_INTEGRATOR_NMOMENTS > 4)
        /* dt-advance of stress slots: indices 1+NUMDIMS..NMOMENTS-1.
         * Currently only fires in 3D SECONDMOMENT (precompiler fence). */
        for(k=1+NUMDIMS;k<CBE_INTEGRATOR_NMOMENTS;k++) {pi.CBE_basis_moments[j][k] += nfac * (dt*pi.CBE_basis_moments_dt[j][k]);}
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
         * the central stress block (Sxx, Syy, Szz, Sxy, Sxz, Syz) as the
         * final pass after the existing diagonal-lower-clamp + Cauchy-
         * Schwarz + determinant crossnorm chain. Replaces the if(2==2)
         * 1D-collapse stopgap.
         *
         * Wave-CBE Fix #2a (2026-05-30): migrated from cbe_spd_repair_S3x3
         * (which silently treated the raw R block at slots [4..9] as if it
         * were central S) to the raw-row wrapper, which correctly converts
         * raw <-> central before/after projection. The eigenvalue floor is
         * now CBE_SPD_RELATIVE_FLOOR * trace(central S), NOT the hidden
         * bulk-KE-inclusive trace(raw R) the old shim used.
         *
         * Note Fix #2b will replace this entire chain (diagonal floor +
         * Cauchy-Schwarz + det crossnorm + this SPD projection + the
         * split-largest below) with a conservative repair operator. 2a
         * preserves the chain shape, fixing only the unit bug. */
        {
            const double m_pi    = pi.CBE_basis_moments[j][0];
            double eigenvalue_floor = 0.0;
            if(m_pi > 0 && isfinite(m_pi)) {
                /* Always-3-vector velocity; missing components zero-padded.
                 * This block is SECONDMOMENT-only (NMOMENTS>=10 outer gate)
                 * → fenced 3D today, so v[1]/v[2] are real. The helper-based
                 * load keeps the call site unfence-safe. */
                double v[3]; cbe_basis_v_load_3(pi.CBE_basis_moments[j], v);
                const double trace_S_central = (pi.CBE_basis_moments[j][4]
                                             +  pi.CBE_basis_moments[j][5]
                                             +  pi.CBE_basis_moments[j][6]) / m_pi
                                             - (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                eigenvalue_floor = CBE_SPD_RELATIVE_FLOOR
                                 * DMAX(trace_S_central, MIN_REAL_NUMBER);
            }
            if(dT_out) {
                double dT_basis = 0.0;
                (void)cbe_basis_row_project_central_stress_to_PSD(
                    pi.CBE_basis_moments[j], eigenvalue_floor, &dT_basis);
                dT_local += dT_basis;
            } else {
                (void)cbe_basis_row_project_central_stress_to_PSD(
                    pi.CBE_basis_moments[j], eigenvalue_floor, nullptr);
            }
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
    /* CBE relative-frame postgravity (storage convention: basis moments are
     * stored relative to the mesh-generating-point velocity P.Vel — i.e.
     *    basis_p_stored[α] = m_α * (v_phys[α] − P.Vel)
     *    v_phys[α]         = basis_p_stored[α]/m_α + P.Vel.
     * Total physical accumulators from the per-basis flux pass are stored in
     * dmom_tot:
     *    dmom_tot[0]   = Σ_α dm_α/dt           (= 0 by MFM mass closure)
     *    dmom_tot[k+1] = Σ_α dp_abs_α/dt       (absolute-frame momentum rate;
     *                                           the flux solver works in
     *                                           absolute frame and the sum
     *                                           is the bulk momentum rate
     *                                           in absolute frame).
     * The bulk acceleration of the mesh-generating point in absolute frame:
     *    a_cbe[k] = ( Σ dp_abs/dt[k+1] − P.Vel[k] * Σ dm/dt ) / M
     * The −V·Σdmdt term is the projection of the COM acceleration onto the
     * mass-weighted relative-frame storage (would vanish if Σdmdt = 0 exactly,
     * but we include it for safety on residual MFM non-exactness). */
    Vec3<double> a_cbe = {
        m_inv * (cbe_basis_p_r(dmom_tot, 0) - pi.Vel[0] * dmom_tot[0]),
        m_inv * (cbe_basis_p_r(dmom_tot, 1) - pi.Vel[1] * dmom_tot[0]),
        m_inv * (cbe_basis_p_r(dmom_tot, 2) - pi.Vel[2] * dmom_tot[0])
    };
    /* Inject the CBE bulk acceleration into P.GravAccel so the regular
     * half-step velocity kick advances P.Vel by a_cbe*dt. This is NOT
     * self-gravity (SELFGRAVITY_OFF leaves this channel zero from gravtree);
     * it is the mesh-generating-point acceleration from CBE flux-sum
     * momentum and is the relative-convention path for advancing the COM. */
    pi.GravAccel += a_cbe / All.cf_a2inv;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        /* Capture RAW per-basis dmdt BEFORE the mass closure modifies it;
         * the per-basis -V*dmdt_raw frame-projection below requires the
         * pre-closure rate. */
        const double dmdt_raw_alpha = pi.CBE_basis_moments_dt[j][0];
        /* Mass closure: enforce Σ_α dmdt_α = 0 exactly (safety net;
         * upstream MFM should already deliver Σdmdt ≈ 0 to machine eps). */
        pi.CBE_basis_moments_dt[j][0] -= pi.CBE_basis_moments[j][0] * (m_inv * dmom_tot[0]);
        /* Per-basis relative-frame momentum conversion. The absolute-frame
         * per-basis momentum rate from the flux pass projects onto the
         * relative-frame stored quantity via
         *    dpdt_rel[α][k] = dpdt_abs[α][k] − P.Vel[k] * dmdt_raw[α] − m_α * a_cbe[k].
         * The −V*dmdt_raw term handles the frame-conversion of momentum
         * carried by mass flux at nonzero bulk velocity; the −m*a_cbe term
         * subtracts the COM-acceleration share so the stored rate is
         * per-basis deviation about the bulk. */
        for(k=0;k<NUMDIMS;k++) {
            pi.CBE_basis_moments_dt[j][k+1] -= pi.Vel[k] * dmdt_raw_alpha
                                            +  pi.CBE_basis_moments[j][0] * a_cbe[k];
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
