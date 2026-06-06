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
 * trace. Applied identically in face-state Q-clamp and the Fix #2b
 * conservative cell-state repair (both via the SSOT helper
 * cbe_basis_row_project_central_stress_to_PSD). 1e-12 is a small relative
 * floor (well below the intended stress scale, but large enough to
 * deterministically exclude strictly-zero / negative eigenvalues from
 * downstream consumers). */
static constexpr double CBE_SPD_RELATIVE_FLOOR = 1.0e-12;

/* Wave-CBE Fix #2b (2026-06-04) tunings for the conservative cell-state
 * realizability repair. EPS_M_ABS / EPS_M_REL: empty-slot mass threshold
 * (m_thresh = max(EPS_M_ABS, EPS_M_REL * M_cell / NBASIS)). EPS_CONS:
 * FP-scale residual acceptance bound (residual norm <= EPS_CONS * cell_scale
 * after donation -> accept as roundoff; larger -> deterministic conservative
 * fallback in Step 7). */
static constexpr double CBE_REPAIR_EPS_M_ABS = 1.0e-30;
static constexpr double CBE_REPAIR_EPS_M_REL = 1.0e-12;
static constexpr double CBE_REPAIR_EPS_CONS  = 1.0e-12;

/* Aggregate per-basis outflow-budget limiter (commit 2 of the limiter pair).
 * FRAC: per-basis outgoing-mass budget as fraction of m_a, capped per drift
 * call. MIN_DONOR_MASS: floor below which a basis is ineligible as a donor
 * AND as a capped-basis recipient of the (1-f)*excess "uncapped remainder"
 * (treated as floor for both ends). 0.9 leaves headroom for the nfac threshold
 * (-0.75) downstream + FP. */
static constexpr double CBE_OUTFLOW_BUDGET_FRAC           = 0.9;
static constexpr double CBE_OUTFLOW_BUDGET_MIN_DONOR_MASS = 1.0e-30;

/* Per-basis outflow-budget limiter status codes (orthogonal to Fix #2b
 * repair codes). 200+ for non-error outcomes; no hard-fails / endruns --
 * donor-limited cases surface via the counter, not abort. */
enum {
    CBE_OUTBUDGET_OK              = 200,   /* nothing exceeded budget          */
    CBE_OUTBUDGET_APPLIED         = 201,   /* at least one basis capped, donors clean */
    CBE_OUTBUDGET_DONOR_LIMITED   = 202,   /* cap applied at fraction f<1; (1-f)*excess unredistributed */
    CBE_OUTBUDGET_NO_DONOR        = 203    /* no eligible donor; this basis a uncapped */
};

/* POD diagnostic counters for the outflow-budget limiter. Allocated only
 * when diagnostics are enabled (matching Fix #2b repair counts gating).
 * NOT promoted to a cbe_diagnostics.txt column (codex 2026-06-04: no schema
 * churn); accessible via the optional CBE_INTEGRATOR_OUTBUDGET_VERBOSE
 * printf path or via debugger. */
struct CbeOutflowBudgetDiag {
    long long n_capped_bases;          /* SUM: bases that hit f_out cap (status APPLIED) */
    long long n_donor_limited;         /* SUM: bisection capped fraction f < 1 (DONOR_LIMITED) */
    long long n_donor_failures;        /* SUM: NO_DONOR + DONOR_LIMITED bookkeeping */
    double    max_pre_out_frac_seen;   /* MAX: pre-cap dt*Udot_out[a][0]/m_a seen */
};

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

/* Active-dimensional symmetric stress workspace helpers (commit 4a, 2026-06-03).
 *
 * Stress math operates on a 3x3 workspace tensor for shape uniformity (matches
 * the 3-vector convention for dp[3]/Vel[3]/Face_Area_Vec[3]), but ONLY the
 * upper-left NUMDIMS x NUMDIMS block carries meaningful values. Off-block
 * entries are kept at exactly zero so that downstream consumers (Jacobi PSD
 * projection, principal-minors realizability test, n.S.n contractions) reduce
 * to the correct active-dim physics without per-D special-casing in callers.
 *
 * SECONDMOMENT non-3D was fenced until this commit; the new helpers carry the
 * dim-aware layout to all stress consumers so the fence at
 * declarations/precompiler_logic.h can be lifted.
 *
 * Semantics:
 *   _load_active(row, T)  -- zero T[3][3], then fill the active D x D upper
 *                            triangle (symmetrized) from row[T_idx(a,b)].
 *   _load_active_scaled(row, scale, T)  -- same but pre-multiplied by `scale`
 *                            (useful for R = T/m).
 *   _store_active(row, T) -- write T_ab for active (a<=b)<NUMDIMS back into
 *                            row[T_idx(a,b)]. Inactive entries ignored.
 *   _trace_active(row, scale) -- sum_{a<NUMDIMS} row[T_idx(a,a)] * scale.
 *
 * Realizability + PSD projection helpers operate on the same 3x3 workspace
 * but ignore inactive rows/cols, so e.g. in 1D the Jacobi sweep is a no-op
 * and only S[0][0] participates in the eigenvalue floor / clamp. */
KOKKOS_INLINE_FUNCTION void cbe_basis_T_load_active(
    const double *row, double T[3][3])
{
    for(int a=0; a<3; a++) for(int b=0; b<3; b++) T[a][b] = 0.0;
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0; a<NUMDIMS; a++) {
        T[a][a] = row[cbe_T_idx(a, a)];
        for(int b=a+1; b<NUMDIMS; b++) {
            const double Tab = row[cbe_T_idx(a, b)];
            T[a][b] = Tab; T[b][a] = Tab;
        }
    }
#else
    (void)row;
#endif
}
KOKKOS_INLINE_FUNCTION void cbe_basis_T_load_active_scaled(
    const double *row, double scale, double T[3][3])
{
    cbe_basis_T_load_active(row, T);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0; a<NUMDIMS; a++) {
        T[a][a] *= scale;
        for(int b=a+1; b<NUMDIMS; b++) { T[a][b] *= scale; T[b][a] = T[a][b]; }
    }
#else
    (void)scale;
#endif
}
KOKKOS_INLINE_FUNCTION void cbe_basis_T_store_active(
    double *row, const double T[3][3])
{
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0; a<NUMDIMS; a++) {
        row[cbe_T_idx(a, a)] = T[a][a];
        for(int b=a+1; b<NUMDIMS; b++) row[cbe_T_idx(a, b)] = T[a][b];
    }
#else
    (void)row; (void)T;
#endif
}
KOKKOS_INLINE_FUNCTION double cbe_basis_T_trace_active(
    const double *row, double scale)
{
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    double tr = 0.0;
    for(int a=0; a<NUMDIMS; a++) tr += row[cbe_T_idx(a, a)];
    return tr * scale;
#else
    (void)row; (void)scale;
    return 0.0;
#endif
}

/* ===== Primitive (ρ, v, S) row helpers — primitive-gradient swap support =====
 *
 * The primitive-gradient swap will make the gradient writer and face
 * reconstruction operate on primitive rows (ρ, v_k, S_kl) rather than moment
 * rows (m, p_k, T_kl). These helpers convert between the two packings; the
 * slot layout of the primitive row matches the existing moment-row packing
 * exactly, slot-for-slot:
 *   slot 0           : ρ (= m)
 *   slots 1..NUMDIMS : v_k = p_k / ρ
 *   stress slots     : S_kl = T_kl / ρ - v_k v_l         (SECONDMOMENT only)
 * The stress slots are indexed by cbe_T_idx() exactly as in the moment row.
 *
 * Identity (codex 2026-06-06, load-bearing for the swap):
 *     ρ · T_kl − p_k p_l ≡ ρ² · S_kl
 * Moment-cone realizability (m·T − p·p ⪰ 0) is therefore exactly equivalent
 * to primitive realizability (ρ ≥ ρ_floor, S ⪰ 0). The face limiter exploits
 * this — see OPEN_cbe_primitive_grad_design.md §6.
 *
 * The persistent storage field is still spelled Gradients_CBE_basis_moments
 * but holds primitive-gradient slots after the swap. The field-name rename
 * is out of scope; see particle_data.h doc and OPEN_cbe_primitive_grad_design.md
 * §3.
 *
 * Phase 0 (2026-06-06) ADDS THESE HELPERS but does NOT use them. No gradient
 * writer / face reader / limiter behavior change in Phase 0. */

/* Single SSOT for "is this row's density active enough to divide by?".
 * Used at every primitive-derivation site (primitive conversion, scratch-row
 * gradient gating, face-clamp consistency check). Matches the existing
 * MIN_REAL_NUMBER threshold used by cbe_clamp_face_Q and the pair-matching
 * cost helpers (cbe_cost_v_only, cbe_cost_trace_w2). Do NOT introduce a
 * second density floor — grep this name to find every site that gates on
 * "active enough to divide". */
KOKKOS_INLINE_FUNCTION
double cbe_rho_active_floor(void) { return MIN_REAL_NUMBER; }

/* Moment row -> primitive row. Both arrays sized [CBE_INTEGRATOR_NMOMENTS]
 * with the slot packing described above. Scratch rows (Q_row[0] below the
 * active-density floor) get prim_row[0] = Q_row[0] (the raw density, preserved
 * for downstream gating) and all v / S slots set to zero — the caller MUST
 * NOT consume those slots without first re-checking the floor. The primitive
 * conversion never divides by a sub-floor density; the explicit zeros prevent
 * NaN/Inf from leaking into downstream consumers.
 *
 * Inactive dimensions (k >= NUMDIMS) are not packed into the row at all
 * (NMOMENTS shrinks in low-D, matching the moment-row convention). The loop
 * bounds use NUMDIMS so the compiler folds away the inactive iterations. */
KOKKOS_INLINE_FUNCTION
void cbe_moments_to_primitives_row(const double Q_row[CBE_INTEGRATOR_NMOMENTS],
                                   double prim_row[CBE_INTEGRATOR_NMOMENTS])
{
    const double rho = Q_row[0];
    prim_row[0] = rho;
    /* Strict `>` matches the existing active-row predicate used by
     * cbe_clamp_face_Q (Q_face[m][0] <= MIN_REAL_NUMBER -> inactive/clamped).
     * SSOT consistency: same edge convention everywhere. */
    const bool active = (rho > cbe_rho_active_floor());
    const double inv_rho = active ? (1.0 / rho) : 0.0;

    double v_tmp[3] = {0.0, 0.0, 0.0};
    for(int k=0; k<NUMDIMS; k++) {
        const double v_k = cbe_basis_p_r(Q_row, k) * inv_rho;
        v_tmp[k] = v_k;
        cbe_basis_p_w(prim_row, k, active ? v_k : 0.0);
    }

#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0; a<NUMDIMS; a++) {
        const double T_aa = cbe_basis_T_r(Q_row, a, a);
        const double S_aa = active ? (T_aa * inv_rho - v_tmp[a] * v_tmp[a]) : 0.0;
        cbe_basis_T_w(prim_row, a, a, S_aa);
        for(int b=a+1; b<NUMDIMS; b++) {
            const double T_ab = cbe_basis_T_r(Q_row, a, b);
            const double S_ab = active ? (T_ab * inv_rho - v_tmp[a] * v_tmp[b]) : 0.0;
            cbe_basis_T_w(prim_row, a, b, S_ab);
        }
    }
#endif
}

/* Primitive row -> moment row. Deterministic, no iteration. The inverse of
 * cbe_moments_to_primitives_row at every site where the primitive row was
 * built from a row with ρ > cbe_rho_active_floor(). At or below the floor
 * the caller must NOT use this helper — see scratch-row handling in
 * OPEN_cbe_primitive_grad_design.md §7 (face reconstruction stays first-
 * order moment for sub-floor rows, bypassing the round-trip).
 *
 * Identity: applying primitives_to_moments after moments_to_primitives is
 * algebraically exact for any row with ρ > floor. FP cancellation can still
 * occur in the forward step (S = T/ρ − vv for cold or warm-small S); the
 * inverse step just multiplies by ρ and adds vv back, so the round-trip
 * recovers the original moment row only up to the cancellation error in S. */
KOKKOS_INLINE_FUNCTION
void cbe_primitives_to_moments_row(const double prim_row[CBE_INTEGRATOR_NMOMENTS],
                                   double Q_row[CBE_INTEGRATOR_NMOMENTS])
{
    const double rho = prim_row[0];
    Q_row[0] = rho;

    double v_tmp[3] = {0.0, 0.0, 0.0};
    for(int k=0; k<NUMDIMS; k++) {
        const double v_k = cbe_basis_p_r(prim_row, k);
        v_tmp[k] = v_k;
        cbe_basis_p_w(Q_row, k, rho * v_k);
    }

#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0; a<NUMDIMS; a++) {
        const double S_aa = cbe_basis_T_r(prim_row, a, a);
        cbe_basis_T_w(Q_row, a, a, rho * (S_aa + v_tmp[a] * v_tmp[a]));
        for(int b=a+1; b<NUMDIMS; b++) {
            const double S_ab = cbe_basis_T_r(prim_row, a, b);
            cbe_basis_T_w(Q_row, a, b, rho * (S_ab + v_tmp[a] * v_tmp[b]));
        }
    }
#endif
}

/* Active-dim symmetric PSD test via principal minors of the upper-left
 * NUMDIMS x NUMDIMS block. Reduces to scalar nonnegativity in 1D, the
 * 2x2 Sylvester chain in 2D, and the full 3x3 chain in 3D. eps_tol is
 * relative to trace_pos (active diagonals only) at each minor order. */
KOKKOS_INLINE_FUNCTION
bool cbe_symNxN_active_principal_minors_nonneg(const double M[3][3], double eps_tol)
{
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    double trace_pos = MIN_REAL_NUMBER;
    for(int a=0; a<NUMDIMS; a++) { trace_pos += M[a][a]; }
    if(trace_pos < MIN_REAL_NUMBER) trace_pos = MIN_REAL_NUMBER;
    const double eps_diag = eps_tol * trace_pos;
    for(int a=0; a<NUMDIMS; a++) { if(M[a][a] < -eps_diag) return false; }
#if (NUMDIMS >= 2)
    const double eps_2x2 = eps_tol * trace_pos * trace_pos;
    for(int a=0; a<NUMDIMS; a++)
        for(int b=a+1; b<NUMDIMS; b++)
            if(M[a][a]*M[b][b] - M[a][b]*M[a][b] < -eps_2x2) return false;
#endif
#if (NUMDIMS >= 3)
    const double eps_det = eps_tol * trace_pos * trace_pos * trace_pos;
    const double det = M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[1][2])
                     - M[0][1] * (M[0][1]*M[2][2] - M[1][2]*M[0][2])
                     + M[0][2] * (M[0][1]*M[1][2] - M[1][1]*M[0][2]);
    if(det < -eps_det) return false;
#endif
    return true;
#else
    (void)M; (void)eps_tol;
    return true;
#endif
}

/* Active-dim symmetric PSD projection in-place on a 3x3 workspace. Outside
 * the active block, entries are assumed zero and are NOT touched (no eigen-
 * value floor on inactive dims -- the trace correction dT_out only counts
 * active diagonals). Algorithm: Jacobi sweep over active (a<b<NUMDIMS) off-
 * diagonals (no-op in 1D), clamp active diagonals to >= eigenvalue_floor,
 * reconstruct via V D V^T on the active block.
 *
 * 3D path: existing 6-sweep over (0,1)/(0,2)/(1,2), matches the pre-commit
 * cbe_project_central_S_to_PSD behavior. 2D: 6 sweeps over (0,1) only --
 * conservative-overkill on a 2x2 (one sweep would suffice; 6 are cheap).
 * 1D: no off-diagonal sweep; floor on S[0][0].
 *
 * Caller provides the eigenvalue_floor (e.g. CBE_SPD_RELATIVE_FLOOR * trace
 * for the face-clamp policy, or 0 for Fix #2 cell repair true-PSD). dT_out
 * (nullable) receives the post-minus-pre active-trace change. Non-finite
 * input triggers the same MIN_REAL_NUMBER fallback on active diagonals as
 * the 3D path used. */
KOKKOS_INLINE_FUNCTION
bool cbe_project_central_S_active_to_PSD(double S[3][3], double eigenvalue_floor,
                                         double *dT_out)
{
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    /* Self-protecting contract: explicitly zero inactive rows/cols on entry
     * so callers that hand in stack workspaces with stale junk cannot
     * pollute the active PSD test or the post-reconstruction store. */
    for(int a=0; a<3; a++) for(int b=0; b<3; b++)
        if(a >= NUMDIMS || b >= NUMDIMS) S[a][b] = 0.0;

    /* Pre-trace over active diagonals. */
    double trace_before = 0.0;
    bool all_finite = true;
    for(int a=0; a<NUMDIMS; a++) {
        if(!isfinite(S[a][a])) { all_finite = false; }
        else trace_before += S[a][a];
        for(int b=a+1; b<NUMDIMS; b++)
            if(!isfinite(S[a][b])) { all_finite = false; }
    }
    if(!all_finite || !isfinite(trace_before)) {
        /* Reset active block to the same MIN_REAL_NUMBER isotropic-floor
         * pattern as the old 3D helper. dT_out reports the post-reset
         * trace; we do NOT compute a delta against a non-finite pre-trace
         * (NaN/Inf would silently poison repair_dT_sum diagnostics). */
        double tr_after = 0.0;
        for(int a=0; a<NUMDIMS; a++) {
            S[a][a] = MIN_REAL_NUMBER;
            tr_after += MIN_REAL_NUMBER;
            for(int b=a+1; b<NUMDIMS; b++) { S[a][b] = 0.0; S[b][a] = 0.0; }
        }
        if(dT_out) *dT_out = tr_after;
        return true;
    }

    /* Working copies for Jacobi: A starts as S on the active block (others 0),
     * V starts as identity on the active block (others 0 -- inactive rows/cols
     * are not rotated). */
    double A[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double V[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for(int a=0; a<NUMDIMS; a++) {
        for(int b=0; b<NUMDIMS; b++) A[a][b] = S[a][b];
        V[a][a] = 1.0;
    }

#if (NUMDIMS >= 2)
    /* 6 fixed Jacobi sweeps over active off-diagonals (a<b<NUMDIMS). */
    for(int sweep=0; sweep<6; sweep++) {
        for(int p=0; p<NUMDIMS; p++) for(int q=p+1; q<NUMDIMS; q++) {
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
            const double c   = 1.0 / sqrt(1.0 + t*t);
            const double s   = t * c;
            const double tau = s / (1.0 + c);
            const double App = A[p][p], Aqq = A[q][q];
            A[p][p] = App - t * Apq;
            A[q][q] = Aqq + t * Apq;
            A[p][q] = 0.0; A[q][p] = 0.0;
            for(int r=0; r<NUMDIMS; r++) {
                if(r == p || r == q) continue;
                const double Arp = A[r][p], Arq = A[r][q];
                A[r][p] = Arp - s * (Arq + tau * Arp);
                A[r][q] = Arq + s * (Arp - tau * Arq);
                A[p][r] = A[r][p]; A[q][r] = A[r][q];
            }
            for(int r=0; r<NUMDIMS; r++) {
                const double Vrp = V[r][p], Vrq = V[r][q];
                V[r][p] = Vrp - s * (Vrq + tau * Vrp);
                V[r][q] = Vrq + s * (Vrp - tau * Vrq);
            }
        }
    }
#endif

    /* Diagonals of A post-sweep are the eigenvalues. Check whether any active
     * eigenvalue lies strictly below eigenvalue_floor: if not, S was already
     * PSD relative to the requested floor and we MUST leave it untouched
     * (do not let V*diag*V^T reconstruction roundoff drift S silently while
     * we report dT=0 / floored=false). Matches the pre-refactor 3D helper
     * semantics. */
    bool floored = false;
    double lambda[3] = {0,0,0};
    for(int a=0; a<NUMDIMS; a++) {
        if(A[a][a] < eigenvalue_floor) { floored = true; lambda[a] = eigenvalue_floor; }
        else                            { lambda[a] = A[a][a]; }
    }
    if(!floored) {
        if(dT_out) *dT_out = 0.0;
        return false;
    }

    /* Reconstruct S_active = V * diag(lambda) * V^T on the active block;
     * inactive entries stay at zero. */
    double trace_after = 0.0;
    for(int a=0; a<NUMDIMS; a++) {
        for(int b=a; b<NUMDIMS; b++) {
            double sum = 0.0;
            for(int k=0; k<NUMDIMS; k++) sum += V[a][k] * lambda[k] * V[b][k];
            S[a][b] = sum;
            if(b > a) S[b][a] = sum;
        }
        trace_after += S[a][a];
    }
    /* Floor cannot mathematically reduce trace; protect repair_dT_sum
     * diagnostics against tiny negative reconstruction roundoff (matches
     * the pre-refactor 3D helper's invariant accounting). */
    if(dT_out) {
        double dT = trace_after - trace_before;
        if(!isfinite(dT) || dT < 0.0) dT = 0.0;
        *dT_out = dT;
    }
    return true;
#else
    (void)S; (void)eigenvalue_floor;
    if(dT_out) *dT_out = 0.0;
    return false;
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
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    /* Active-block trace of central S = sum_{a<NUMDIMS} (T_aa/m - v_a^2),
     * each piece floored at 0 before sqrt. Migrated in commit 4a from
     * hardcoded 4+k indices that only worked in 3D. */
    double tr_S_a = 0, tr_S_b = 0;
    for(int a=0; a<NUMDIMS; a++) {
        double S_a_aa = cbe_basis_T_r(moments_a, a, a) * inv_a - v_a[a]*v_a[a];
        double S_b_aa = cbe_basis_T_r(moments_b, a, a) * inv_b - v_b[a]*v_b[a];
        tr_S_a += DMAX(S_a_aa, 0.0);
        tr_S_b += DMAX(S_b_aa, 0.0);
    }
    double dsqrt_S = sqrt(tr_S_a) - sqrt(tr_S_b);
    c += dsqrt_S * dsqrt_S;
#endif
    return c;
}


/* Free-slot row-fallback transform on an NBASIS x NBASIS cost matrix.
 *
 * Fix #3 from reference_cbe_method_fix_list.md (harness 2026-05-30):
 *
 * For every source basis alpha with nonfloor mass (rho_a[alpha] > eps_rho),
 * apply the cost reweight
 *     C[alpha][beta] *= target_masses[beta] / (source_masses[alpha]
 *                                              + target_masses[beta] + eps_rho)
 * where eps_rho = 1e-8 * (sum_source rho + sum_target rho) is a relative
 * regularizer. The multiplicative prefactor naturally biases the argmin
 * toward empty / lightly-occupied target slots only when no exact-match
 * basis exists (its C_W^2[alpha,beta] ≈ 0 already dominates the argmin).
 *
 * The PRIOR median(C) gate (Wave-CBE Commit 6a) is REMOVED — the harness
 * docstring catches it: row C[alpha] = [1,4,9,16] has median(C)=1 and the
 * gate "1 > 1" is False, so the reweighting silently never fires on the
 * perturbation IC. All three cost choices collapse to plain W^2 and the
 * perturbation routes into the dominant +1 slot instead of the empty -2
 * slot. Always-reweighting (gated only by source mass nonfloor) is the
 * production form per python_harness/cbe1d/pairing.py:cost_free_slot_fallback.
 *
 * The source-mass eps_rho skip is ESSENTIAL: floor-mass sources have
 * v_alpha = p_alpha / m_alpha dominated by FP noise; allowing them to
 * drive routing decisions destabilizes pairing.
 *
 * fired_count_inout is NULLABLE. When non-null, increments by 1 per
 * alpha-row that passed the source-mass gate and applied the reweight.
 * Per-face increment bounded by 2*NBASIS (two cost matrices per face).
 *
 * Direction is asymmetric. Caller passes (source_masses, target_masses)
 * matching the cost-matrix BUILD direction. For an a->b matrix
 *   C[alpha_on_a][beta_on_b] = cost(Q_a[alpha], Q_b[beta]):
 *     source_masses[i] = Q_a[i][0]        // mass density of a-side basis i
 *     target_masses[j] = Q_b[j][0]        // mass density of b-side basis j
 * The b->a direction (separately built C') requires its own free-slot
 * call with source = Q_b masses, target = Q_a masses. It is NOT a
 * transpose of this transform. */
KOKKOS_INLINE_FUNCTION
void cbe_apply_free_slot_fallback(
    double C[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NBASIS],
    const double source_masses[CBE_INTEGRATOR_NBASIS],
    const double target_masses[CBE_INTEGRATOR_NBASIS],
    int *fired_count_inout)
{
    const int N = CBE_INTEGRATOR_NBASIS;
    double sum_rho = 0.0;
    for(int m=0; m<N; m++) { sum_rho += source_masses[m] + target_masses[m]; }
    const double eps_rho = 1.0e-8 * sum_rho;
    for(int m=0; m<N; m++) {
        if(source_masses[m] <= eps_rho) continue;   /* floor-mass source: skip */
        for(int p=0; p<N; p++) {
            const double denom = source_masses[m] + target_masses[p] + eps_rho;
            C[m][p] *= target_masses[p] / denom;
        }
        if(fired_count_inout) { (*fired_count_inout)++; }
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
    /* Always-3-vector velocity. Missing components zero-padded by the helper.
     * Stress block runs only under SECONDMOMENT and loops over the active
     * NUMDIMS x NUMDIMS block via cbe_basis_T_load_active_scaled. Commit 4a
     * migrated the layout to be dim-agnostic; the SECONDMOMENT D!=3 fence
     * was lifted in commit 4b and the frame-conversion bookkeeping was
     * moved (codex 2026-06-04) into the drift-kick absolute round-trip via
     * cbe_relative_row_to_absolute / cbe_absolute_to_relative_row. */
    double v[3]; cbe_basis_v_load_3(moments, v);
    double nSn = 0;
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    {
        /* R = T / m on the active block; inactive entries stay at zero so the
         * n^T S n contraction below restricts to active dimensions of n_hat. */
        double R[3][3];
        cbe_basis_T_load_active_scaled(moments, inv_rho, R);
        for(int k=0; k<NUMDIMS; k++) {
            double s = 0;
            for(int l=0; l<NUMDIMS; l++) s += (R[k][l] - v[k]*v[l]) * n_hat[l];
            nSn += n_hat[k] * s;
        }
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


/* SSOT relative <-> absolute moment transforms (codex 2026-06-04, finite
 * absolute-update round-trip in cbe_drift_kick).
 *
 *   ABSOLUTE-frame moments of basis b:
 *     m_abs    = m_b
 *     p_abs[a] = m_b * <v_a>_b           = p_rel[a] + m_b * V[a]
 *     T_abs[a,b] = m_b * <v_a v_b>_b     = T_rel[a,b] + V[a] p_rel[b] + p_rel[a] V[b] + m_b V[a] V[b]
 *
 *   RELATIVE-frame storage matches the inverse:
 *     p_rel[a]   = p_abs[a]   - m_b * V[a]
 *     T_rel[a,b] = T_abs[a,b] - V[a] p_rel[a,b] - p_rel[a] V[b] - m_b V[a] V[b]   (uses new p_rel)
 *
 * These two helpers are the SSOT used by the new cbe_drift_kick absolute
 * round-trip and by any standalone test code (unit-checked algebra). The row
 * layout is the same in both frames; only the [1..NUMDIMS] and [T_idx] slot
 * VALUES differ. Operations are NUMDIMS-aware and gate the stress block on
 * SECONDMOMENT. */
KOKKOS_INLINE_FUNCTION
void cbe_relative_row_to_absolute(
    const double row_rel[CBE_INTEGRATOR_NMOMENTS],
    const double V[3],
    double *m_abs_out,
    double p_abs_out[3],
    double T_abs_out[3][3])
{
    const double m_b = row_rel[0];
    *m_abs_out = m_b;
    /* Read RELATIVE momentum slots; populate the active dims of p_abs. */
    double p_rel_a[3] = {0,0,0};
    for(int a=0;a<NUMDIMS;a++) p_rel_a[a] = cbe_basis_p_r(row_rel, a);
    for(int a=0;a<3;a++) p_abs_out[a] = 0.0;
    for(int a=0;a<NUMDIMS;a++) p_abs_out[a] = p_rel_a[a] + m_b * V[a];
    /* T_abs = T_rel + V*p_rel + p_rel*V + m*V*V on active block; zero outside. */
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) T_abs_out[a][b] = 0.0;
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0;a<NUMDIMS;a++) {
        T_abs_out[a][a] = cbe_basis_T_r(row_rel, a, a)
                        + 2.0 * V[a] * p_rel_a[a]
                        + m_b * V[a] * V[a];
        for(int b=a+1;b<NUMDIMS;b++) {
            const double T_ab = cbe_basis_T_r(row_rel, a, b)
                              + V[a] * p_rel_a[b] + V[b] * p_rel_a[a]
                              + m_b * V[a] * V[b];
            T_abs_out[a][b] = T_ab; T_abs_out[b][a] = T_ab;
        }
    }
#endif
}

KOKKOS_INLINE_FUNCTION
void cbe_absolute_to_relative_row(
    double m_abs,
    const double p_abs_in[3],
    const double T_abs_in[3][3],
    const double V[3],
    double row_rel_out[CBE_INTEGRATOR_NMOMENTS])
{
    row_rel_out[0] = m_abs;
    /* Compute NEW p_rel first; T_rel update uses NEW p_rel below. */
    double p_rel_new_a[3] = {0,0,0};
    for(int a=0;a<NUMDIMS;a++) {
        p_rel_new_a[a] = p_abs_in[a] - m_abs * V[a];
        cbe_basis_p_w(row_rel_out, a, p_rel_new_a[a]);
    }
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0;a<NUMDIMS;a++) {
        const double T_rel_aa = T_abs_in[a][a]
                              - 2.0 * V[a] * p_rel_new_a[a]
                              - m_abs * V[a] * V[a];
        cbe_basis_T_w(row_rel_out, a, a, T_rel_aa);
        for(int b=a+1;b<NUMDIMS;b++) {
            const double T_rel_ab = T_abs_in[a][b]
                                  - V[a] * p_rel_new_a[b] - V[b] * p_rel_new_a[a]
                                  - m_abs * V[a] * V[b];
            cbe_basis_T_w(row_rel_out, a, b, T_rel_ab);
        }
    }
#else
    (void)T_abs_in;
#endif
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
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        /* Commit 4b (2026-06-03) — relative-frame stress boost. Storage
         * convention from commit 89fb05cc:
         *    p_rel = m * (v_phys - V)
         *    T_rel = m * <(v_phys - V) (v_phys - V)>
         * where V is the mesh-generating-point velocity. The flux solver
         * consumes raw absolute-frame T_abs, related by
         *    T_abs = T_rel + V (x) p_rel + p_rel (x) V + m * V (x) V
         *
         * V here is the SAME Vel_code[a]*inv_a used by the momentum boost
         * below — the FLUX-frame velocity scaling matches the existing
         * momentum-slot convention. p_rel comes from Q_out[m][1..NUMDIMS]
         * BEFORE the momentum boost runs (captured here). The stress boost
         * MUST precede the momentum boost: it reads p_rel from the still-
         * relative-frame momentum slots. */
        {
            const double rho = Q_out[m][0];
            double V_act[3]    = {0,0,0};
            double p_rel_act[3] = {0,0,0};
            for(int a=0; a<NUMDIMS; a++) {
                V_act[a]    = Vel_code[a] * inv_a;
                p_rel_act[a] = cbe_basis_p_r(Q_out[m], a);   /* pre-boost */
            }
            for(int a=0; a<NUMDIMS; a++) {
                /* Diagonal: T_abs_aa = T_rel_aa + 2*V_a*p_rel_a + rho*V_a*V_a. */
                cbe_basis_T_w(Q_out[m], a, a,
                    cbe_basis_T_r(Q_out[m], a, a)
                    + 2.0 * V_act[a] * p_rel_act[a]
                    + rho * V_act[a] * V_act[a]);
                for(int b=a+1; b<NUMDIMS; b++) {
                    /* Off-diagonal: T_abs_ab = T_rel_ab + V_a*p_rel_b
                     *                        + V_b*p_rel_a + rho*V_a*V_b. */
                    cbe_basis_T_w(Q_out[m], a, b,
                        cbe_basis_T_r(Q_out[m], a, b)
                        + V_act[a] * p_rel_act[b]
                        + V_act[b] * p_rel_act[a]
                        + rho * V_act[a] * V_act[b]);
                }
            }
        }
#endif
        /* Velocity boost on the momentum slots [1..NUMDIMS]. cbe_basis_p_a
         * is a no-op for a >= NUMDIMS so the active-only contract holds for
         * any NUMDIMS. Must run AFTER the stress boost above (which reads
         * the still-relative-frame momentum slots). */
        for(int a=0; a<NUMDIMS; a++) {
            cbe_basis_p_a(Q_out[m], a, Q_out[m][0] * Vel_code[a] * inv_a);
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
 * (FP-scale, eps_tol = CBE_REPAIR_EPS_M_REL). */
KOKKOS_INLINE_FUNCTION
bool cbe_basis_row_is_realizable(
    const double Q_row[CBE_INTEGRATOR_NMOMENTS], double eps_tol)
{
    if(!(Q_row[0] > 0) || !isfinite(Q_row[0])) return false;
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    {
        /* Active-dim realizability: M = T * m - p p^T on the upper-left
         * NUMDIMS x NUMDIMS block (zero-padded outside). T loaded via the
         * stress helper so the slot layout is correct for any NUMDIMS;
         * the predicate checks principal minors only of the active block. */
        const double m = Q_row[0];
        double p[3]; cbe_basis_p_load_3(Q_row, p);
        double T[3][3]; cbe_basis_T_load_active(Q_row, T);
        double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for(int a=0; a<NUMDIMS; a++) {
            M[a][a] = T[a][a] * m - p[a]*p[a];
            for(int b=a+1; b<NUMDIMS; b++) {
                const double off = T[a][b] * m - p[a]*p[b];
                M[a][b] = off; M[b][a] = off;
            }
        }
        if(!cbe_symNxN_active_principal_minors_nonneg(M, eps_tol)) return false;
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
 * Commit 4a (2026-06-03): now SECONDMOMENT-gated (replaces NMOMENTS>=10
 * gate) and operates on the active D x D stress block via the new
 * active-dimensional helpers. Inactive rows/cols stay at zero; PSD
 * projection only floors active eigenvalues. */
KOKKOS_INLINE_FUNCTION
bool cbe_basis_row_project_central_stress_to_PSD(
    double U_row[CBE_INTEGRATOR_NMOMENTS],
    double eigenvalue_floor,
    double *dT_out)
{
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
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
    double v[3]; cbe_basis_v_load_3(U_row, v);

    /* Central stress S = T/m - v v^T on the active block. Inactive entries
     * are exactly zero on entry to the projection (cbe_basis_T_load_active_scaled
     * zeros above NUMDIMS, and the v[k]*v[l] correction is only applied
     * on the active loop). */
    double S[3][3];
    cbe_basis_T_load_active_scaled(U_row, inv_m, S);
    for(int a=0; a<NUMDIMS; a++) {
        S[a][a] -= v[a]*v[a];
        for(int b=a+1; b<NUMDIMS; b++) {
            const double off = S[a][b] - v[a]*v[b];
            S[a][b] = off; S[b][a] = off;
        }
    }

    double dT_central = 0;
    bool modified = cbe_project_central_S_active_to_PSD(S, eigenvalue_floor, &dT_central);

    if(modified) {
        /* Rebuild raw slots: T_kl = m * (S_kl + v_k v_l) on active block. */
        for(int a=0; a<NUMDIMS; a++) {
            U_row[cbe_T_idx(a, a)] = m * (S[a][a] + v[a]*v[a]);
            for(int b=a+1; b<NUMDIMS; b++)
                U_row[cbe_T_idx(a, b)] = m * (S[a][b] + v[a]*v[b]);
        }
    }
    if(dT_out) *dT_out = dT_central * m;   /* raw-slot trace delta */
    return modified;
#else
    /* SECONDMOMENT not active: nothing to project. */
    (void)U_row; (void)eigenvalue_floor;
    if(dT_out) *dT_out = 0;
    return false;
#endif
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
 * Commit 4a (2026-06-03): SECONDMOMENT-gated (was NMOMENTS>=10). The stress
 * block runs in any dimension under SECONDMOMENT; trace and projection
 * helpers loop over active NUMDIMS so 1D/2D warm physics participates
 * correctly. */
KOKKOS_INLINE_FUNCTION
void cbe_clamp_face_Q(
    double Qface[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
    long long *rho_clamp_count,
    long long *S_clamp_count)
{
#if defined(CBE_INTEGRATOR_STRICT_FACE_SPD_GUARD) && defined(CBE_INTEGRATOR_SECONDMOMENT)
    /* Fix #9 strict post-clamp face-state realizability guard. Default OFF;
     * enabled by adding CBE_INTEGRATOR_STRICT_FACE_SPD_GUARD to Config.sh.
     * Only meaningful under SECONDMOMENT -- without stress storage,
     * realizability reduces to mass-positivity which the rho>MIN_REAL_NUMBER
     * branch already enforces. Surfaces post-clamp non-realizability that
     * survives the rho-zero branch + the eigenvalue-floor projection --
     * with the cone limiter (Fix #6) upstream this should NEVER fire at
     * non-roundoff amplitude. printf-only + once-per-call latch (the print
     * is device-safe; fflush/endrun are intentionally NOT used here per
     * codex 2026-06-04 and feedback_code_facts #10). */
    bool printed_failure = false;
#endif
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(Qface[m][0] <= MIN_REAL_NUMBER) {
            for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) Qface[m][k] = 0.0;
            if(rho_clamp_count) (*rho_clamp_count)++;
        }
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        else {
            /* Eigenvalue floor from active central-S trace. Active block
             * only -- the prior 3D-hardcoded sum [4]+[5]+[6] silently set
             * a floor proportional to inactive (y/z) diagonals in low-D.
             * Mass-positivity guarded above (Qface[m][0] > MIN_REAL_NUMBER). */
            const double m_face = Qface[m][0];
            const double inv_m  = 1.0 / m_face;
            double v[3]; cbe_basis_v_load_3(Qface[m], v);
            double v_dot_v = 0.0;
            for(int a=0; a<NUMDIMS; a++) v_dot_v += v[a]*v[a];
            const double trace_R_active  = cbe_basis_T_trace_active(Qface[m], inv_m);
            const double trace_S_central = trace_R_active - v_dot_v;
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
#if defined(CBE_INTEGRATOR_STRICT_FACE_SPD_GUARD) && defined(CBE_INTEGRATOR_SECONDMOMENT)
        /* Two-tier tolerant check on rho-active rows (rho-clamped rows were
         * zeroed above and trivially pass). Strict eps=0 then tolerant eps=
         * CBE_REPAIR_EPS_M_REL. Only fail-of-both is reported (filters FP
         * eigenvalue noise). */
        if(!printed_failure && Qface[m][0] > MIN_REAL_NUMBER) {
            if(!cbe_basis_row_is_realizable(Qface[m], /*eps_tol=*/ 0.0)
               && !cbe_basis_row_is_realizable(Qface[m], CBE_REPAIR_EPS_M_REL)) {
                const double m_face_g = Qface[m][0];
                const double inv_m_g  = 1.0 / m_face_g;
                double v_g[3]; cbe_basis_v_load_3(Qface[m], v_g);
                double v_dot_v_g = 0.0;
                for(int a=0; a<NUMDIMS; a++) v_dot_v_g += v_g[a]*v_g[a];
                const double trR_g  = cbe_basis_T_trace_active(Qface[m], inv_m_g);
                const double trSc_g = trR_g - v_dot_v_g;
                /* Device-safe printf; no fflush/endrun (host-only fns) here. */
                printf("[CBE strict face guard (Fix #9)] post-clamp non-realizable: "
                       "basis=%d m=%.3e trace_S_central=%.3e v.v=%.3e\n",
                       m, m_face_g, trSc_g, v_dot_v_g);
                printed_failure = true;
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
    /* Always-3-vector velocity. Missing components zero-padded so the
     * v.n_hat dot product collapses cleanly to NUMDIMS terms when n_hat
     * has only active components (canonical 1D/2D face normals are
     * axis-aligned; inactive n_hat slots would contribute zero against
     * zero v). */
    double v[3]; cbe_basis_v_load_3(moments, v);
    const double v_n     = v[0]*n_hat[0]    + v[1]*n_hat[1]    + v[2]*n_hat[2];
    const double vF_n    = vface[0]*n_hat[0] + vface[1]*n_hat[1] + vface[2]*n_hat[2];
    const double u_out   = v_n - vF_n;

    /* Central stress contracted with n_hat: S_n_k = (R_kl - v_k v_l) n_hat_l.
     * Active-block only (k,l < NUMDIMS). With the dim-aware layout via
     * cbe_basis_T_load_active_scaled, R inactive entries are exactly zero
     * and v inactive components are zero, so the loop bound here is
     * cosmetic in 3D but load-bearing in 1D/2D where there is no
     * physical y/z stress to contract. c_x from the SSOT scalar helper. */
    double S_n[3] = {0};
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    {
        double R[3][3];
        cbe_basis_T_load_active_scaled(moments, inv_rho, R);
        for(int k=0; k<NUMDIMS; k++) {
            double s = 0;
            for(int l=0; l<NUMDIMS; l++) s += (R[k][l] - v[k]*v[l]) * n_hat[l];
            S_n[k] = s;
        }
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

    /* Stress: F_T_ab = R_ab * F_m + prefactor * |A| * (v_a S_n_b + S_n_a v_b).
     * Active block (a,b < NUMDIMS) with 2*v*S_n on diagonals (the v_a S_n_b
     * + S_n_a v_b sum collapses to 2 v_a S_n_a when a==b). The R_ab * F_m
     * piece carries RAW R; the cross piece uses central S contracted with
     * n_hat (S_n). Writes via cbe_basis_T_w so inactive slots are silently
     * untouched -- no OOB at any NUMDIMS. */
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int a=0; a<NUMDIMS; a++) {
        const double R_aa = cbe_basis_T_r(moments, a, a) * inv_rho;
        cbe_basis_T_w(fluxes, a, a,
            R_aa * fluxes[0] + prefactor * A_norm * 2.0 * v[a] * S_n[a]);
        for(int b=a+1; b<NUMDIMS; b++) {
            const double R_ab = cbe_basis_T_r(moments, a, b) * inv_rho;
            cbe_basis_T_w(fluxes, a, b,
                R_ab * fluxes[0] + prefactor * A_norm * (v[a]*S_n[b] + S_n[a]*v[b]));
        }
    }
#endif

    return (fabs(u_out) + c_x) * A_norm;
}


/* Wave-CBE Fix #2b (2026-06-04): conservative cell-state realizability repair.
 *
 * Single SSOT operator that replaces the legacy in-place chain (active-
 * diagonal floor + 3D Cauchy-Schwarz off-diag + det crossnorm + SPD
 * projection + stochastic split-largest) inside do_cbe_drift_kick_kernel.
 * Operates on RELATIVE-FRAME storage (the canonical pi.CBE_basis_moments
 * layout post the 2026-06-04 absolute round-trip).
 *
 * Conservation contract (preserved to roundoff except where explicitly
 * lossy-with-bounded-magnitude):
 *   Sum_b m_b               preserved via donation arithmetic
 *   Sum_b p_rel_b[a]        preserved (active a)
 *   Sum_b T_rel_b[a,b]      preserved (active a <= b, FULL active tensor)
 *   each row leaves realizable: m_b > 0 AND central S PSD in active dims
 *   no stochastic perturbations
 *   no global abort
 *
 * Residual sign convention (used throughout Step 5+):
 *   residual = target_cell_sum_BEFORE_repair - cell_sum_AFTER_local_repairs
 *   donor   += residual_share   =>   cell_sum_after == target_cell_sum
 *
 * Step outline:
 *   0  guard NaN/Inf, M0 <= 0 -> no-op return (upstream NaN bug class).
 *   1  snapshot target cell invariants (M0, P0, T0); V_bulk = P0/M0.
 *   2  identify empty bases (m_b < m_thresh).
 *   3  reset empties to comoving-inert in RELATIVE storage:
 *        m = m_thresh; p_rel = 0; T_rel = 0.
 *      (Codex: do NOT write m*V_bulk into stored slots; storage is
 *      relative-frame. An inert basis comoving with the cell bulk has
 *      p_rel = 0 and T_rel = 0 by definition.)
 *      Special case: if ALL bases are empty, skip reset entirely
 *      (Phil: would over-mass by NBASIS * m_thresh - M0).
 *   4  row-local PSD projection on non-empty bases. NOT cell-conservative;
 *      trace-deltas flow into Step 5 residual.
 *   5  compute residual vs Step 1 target; precompute cell scales.
 *   6  donate residual:
 *        (a) single-donor most-massive realizable; if donor stays
 *            realizable -> done.
 *        (b) multi-donor weighted by m_b at f=1; if all realizable -> done.
 *        (c) continuous bisection on f in [0,1] for the largest realizable
 *            multi-donor fraction; apply at f_lo; unabsorbed = (1-f_lo)*res.
 *   7  bounded acceptance + deterministic mass-positive proportional fallback:
 *        if |unabsorbed| <= EPS_CONS * cell_scale -> accept FP-scale loss
 *        else distribute the FULL unabsorbed residual proportionally over
 *          all bases by current mass fraction (w_b = m_b / M_cur, where
 *          M_cur = Sum m_b at Step 7 entry = M_new + f_eff*res_M). Adds
 *          exactly unab_M to the cell sum -> Sum m_b restored to M0
 *          (similarly for p_rel and T_rel). Mass positivity guaranteed
 *          since m_b_new = m_b * M0 / M_cur > 0 whenever M0 > 0 (Step 0
 *          guard) and M_cur > 0 (all m_b > 0 after Step 6).
 *   8  defensive final realizability sweep. Any non-realizable row after
 *      Step 7 gets row-local PSD projection (preserves m_b and p_b; touches
 *      stress only). Trace-delta bounded by NBASIS * CBE_SPD_RELATIVE_FLOOR *
 *      max_trace_central_S; FP-scale on smooth inputs (empirical).
 *
 * dT_out (nullable): accumulates Sum trace-delta from all PSD projections
 * (Step 4 + Step 8). Same contract as the prior legacy chain so downstream
 * cbe_diagnostics col-8 semantics carry through. */
KOKKOS_INLINE_FUNCTION
static void cbe_repair_cell_conservative_basis_states(
    struct particle_data& pi,
    double *dT_out)
{
    double dT_local = 0.0;

    /* ----- Step 0: input guard. ----- */
    {
        bool any_nonfinite = false;
        double M0_probe = 0.0;
        for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                if(!isfinite(pi.CBE_basis_moments[j][k])) { any_nonfinite = true; break; }
            }
            if(any_nonfinite) break;
            M0_probe += pi.CBE_basis_moments[j][0];
        }
        if(any_nonfinite || !(M0_probe > 0)) {
            if(dT_out) *dT_out = 0.0;
            return;
        }
    }

    /* ----- Step 1: snapshot target cell invariants. ----- */
    double M0 = 0.0;
    double P0[3] = {0.0, 0.0, 0.0};
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        M0 += pi.CBE_basis_moments[j][0];
        for(int a = 0; a < NUMDIMS; a++) P0[a] += cbe_basis_p_r(pi.CBE_basis_moments[j], a);
    }
    const double V_bulk[3] = { P0[0] / M0, P0[1] / M0, P0[2] / M0 };
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    double T0[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        for(int a = 0; a < NUMDIMS; a++) {
            for(int b = a; b < NUMDIMS; b++) {
                T0[a][b] += cbe_basis_T_r(pi.CBE_basis_moments[j], a, b);
            }
        }
    }
#endif

    /* ----- Step 2: identify empties. ----- */
    const double m_thresh = DMAX(CBE_REPAIR_EPS_M_ABS,
                                  CBE_REPAIR_EPS_M_REL * M0 / (double)CBE_INTEGRATOR_NBASIS);
    bool is_empty[CBE_INTEGRATOR_NBASIS];
    int empty_count = 0;
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        is_empty[j] = (pi.CBE_basis_moments[j][0] < m_thresh);
        if(is_empty[j]) empty_count++;
    }

    /* ----- Step 3: reset empties to comoving-inert RELATIVE storage.
     * Inert basis comoving with V_bulk in absolute frame -> relative
     * storage: m = m_thresh, p_rel = 0, T_rel = 0. The cell-sum changes
     * caused by these assignments flow into Step 5 residual. Special-case:
     * if every basis is empty, skip reset (Phil: NBASIS*m_thresh > M0
     * over-masses the cell). ----- */
    if(empty_count > 0 && empty_count < CBE_INTEGRATOR_NBASIS) {
        for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            if(!is_empty[j]) continue;
            pi.CBE_basis_moments[j][0] = m_thresh;
            for(int a = 0; a < NUMDIMS; a++) cbe_basis_p_w(pi.CBE_basis_moments[j], a, 0.0);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
            for(int a = 0; a < NUMDIMS; a++) {
                for(int b = a; b < NUMDIMS; b++) {
                    cbe_basis_T_w(pi.CBE_basis_moments[j], a, b, 0.0);
                }
            }
#endif
        }
    }

    /* ----- Step 4: row-local PSD projection on the (non-empty) goods.
     * Projection is NOT cell-conservative; the trace-deltas it introduces
     * flow into the Step 5 residual which the donor mechanism absorbs. */
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        if(is_empty[j]) continue;
        const double m_b = pi.CBE_basis_moments[j][0];
        if(!(m_b > 0)) continue;
        double v[3]; cbe_basis_v_load_3(pi.CBE_basis_moments[j], v);
        double v_dot_v = 0.0;
        for(int a = 0; a < NUMDIMS; a++) v_dot_v += v[a] * v[a];
        const double trace_R_active  = cbe_basis_T_trace_active(
            pi.CBE_basis_moments[j], 1.0 / m_b);
        const double trace_S_central = trace_R_active - v_dot_v;
        const double trace_S_pos     = DMAX(trace_S_central, MIN_REAL_NUMBER);
        const double eigenvalue_floor = CBE_SPD_RELATIVE_FLOOR * trace_S_pos;
        double dT_basis = 0.0;
        (void)cbe_basis_row_project_central_stress_to_PSD(
            pi.CBE_basis_moments[j], eigenvalue_floor, &dT_basis);
        dT_local += dT_basis;
    }
#endif

    /* ----- Step 5: residual = target - current_after_local_repairs.
     * Donor application later: donor += residual_share -> cell_sum
     * returns to target. ----- */
    double M_new = 0.0;
    double P_new[3] = {0.0, 0.0, 0.0};
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        M_new += pi.CBE_basis_moments[j][0];
        for(int a = 0; a < NUMDIMS; a++) P_new[a] += cbe_basis_p_r(pi.CBE_basis_moments[j], a);
    }
    double res_M    = M0 - M_new;
    double res_P[3] = { P0[0] - P_new[0], P0[1] - P_new[1], P0[2] - P_new[2] };
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    double T_new[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        for(int a = 0; a < NUMDIMS; a++) {
            for(int b = a; b < NUMDIMS; b++) {
                T_new[a][b] += cbe_basis_T_r(pi.CBE_basis_moments[j], a, b);
            }
        }
    }
    double res_T[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for(int a = 0; a < NUMDIMS; a++) {
        for(int b = a; b < NUMDIMS; b++) {
            res_T[a][b] = T0[a][b] - T_new[a][b];
        }
    }
#endif

    /* Cell scales for FP-bounded acceptance test in Step 7. */
    double V_bulk_norm = 0.0;
    for(int a = 0; a < NUMDIMS; a++) V_bulk_norm += V_bulk[a] * V_bulk[a];
    V_bulk_norm = sqrt(V_bulk_norm);
    double max_abs_P0 = 0.0;
    for(int a = 0; a < NUMDIMS; a++) {
        const double ap = fabs(P0[a]); if(ap > max_abs_P0) max_abs_P0 = ap;
    }
    const double cell_scale_M = M0;
    const double cell_scale_P = DMAX(M0 * V_bulk_norm, max_abs_P0);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    double max_abs_T0 = 0.0;
    for(int a = 0; a < NUMDIMS; a++) {
        for(int b = a; b < NUMDIMS; b++) {
            const double at = fabs(T0[a][b]); if(at > max_abs_T0) max_abs_T0 = at;
        }
    }
    const double cell_scale_T = DMAX(M0 * V_bulk_norm * V_bulk_norm, max_abs_T0);
#endif

    /* ----- Step 6: donate residual.
     * (a) single-donor most-massive realizable
     * (b) multi-donor weighted by m_b at f=1
     * (c) continuous bisection on f in [0,1] ----- */
    bool   is_candidate[CBE_INTEGRATOR_NBASIS];
    double cand_m[CBE_INTEGRATOR_NBASIS];
    double cand_m_sum = 0.0;
    int    sd_idx = -1; double sd_m = -1.0;
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        is_candidate[j] = cbe_basis_row_is_realizable(
            pi.CBE_basis_moments[j], CBE_REPAIR_EPS_M_REL);
        cand_m[j]   = is_candidate[j] ? pi.CBE_basis_moments[j][0] : 0.0;
        cand_m_sum += cand_m[j];
        if(is_candidate[j] && pi.CBE_basis_moments[j][0] > sd_m) {
            sd_m = pi.CBE_basis_moments[j][0]; sd_idx = j;
        }
    }

    /* saved[]: per-row pre-donation state for revert / bisection restore. */
    double saved[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) saved[j][k] = pi.CBE_basis_moments[j][k];
    }

    bool single_ok = false;
    if(sd_idx >= 0 && cand_m_sum > 0) {
        pi.CBE_basis_moments[sd_idx][0] += res_M;
        for(int a = 0; a < NUMDIMS; a++)
            cbe_basis_p_a(pi.CBE_basis_moments[sd_idx], a, res_P[a]);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        for(int a = 0; a < NUMDIMS; a++) {
            for(int b = a; b < NUMDIMS; b++) {
                cbe_basis_T_w(pi.CBE_basis_moments[sd_idx], a, b,
                    cbe_basis_T_r(pi.CBE_basis_moments[sd_idx], a, b) + res_T[a][b]);
            }
        }
#endif
        single_ok = cbe_basis_row_is_realizable(
            pi.CBE_basis_moments[sd_idx], CBE_REPAIR_EPS_M_REL);
        if(!single_ok) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                pi.CBE_basis_moments[sd_idx][k] = saved[sd_idx][k];
        }
    }

    double f_eff = single_ok ? 1.0 : 0.0;

    if(!single_ok && cand_m_sum > 0) {
        /* Bisection bracket: f_lo always realizable (start at 0),
         * f_hi might not be. Initial probe at f=1. */
        double f_lo = 0.0, f_hi = 1.0;

        /* apply_at_f(f): restore all candidate rows from saved[], then add
         * (cand_m[j] / cand_m_sum) * f * residual to each candidate. */
        for(int probe = 0; probe < 2 + 40; probe++) {
            double f_try;
            if(probe == 0)      f_try = 1.0;
            else if(probe == 1) { /* if f=1 worked, stop */
                if(f_lo > 0.0) break;
                f_try = 0.5;
            } else {
                f_try = 0.5 * (f_lo + f_hi);
                if((f_hi - f_lo) < 1e-14) break;
            }
            /* Restore candidates. */
            for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                if(!is_candidate[j]) continue;
                for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                    pi.CBE_basis_moments[j][k] = saved[j][k];
            }
            /* Apply weighted share at f_try. */
            for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                if(!is_candidate[j]) continue;
                const double w = cand_m[j] / cand_m_sum;
                pi.CBE_basis_moments[j][0] += w * f_try * res_M;
                for(int a = 0; a < NUMDIMS; a++)
                    cbe_basis_p_a(pi.CBE_basis_moments[j], a, w * f_try * res_P[a]);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
                for(int a = 0; a < NUMDIMS; a++) {
                    for(int b = a; b < NUMDIMS; b++) {
                        cbe_basis_T_w(pi.CBE_basis_moments[j], a, b,
                            cbe_basis_T_r(pi.CBE_basis_moments[j], a, b) + w * f_try * res_T[a][b]);
                    }
                }
#endif
            }
            /* Check realizability of all candidates. */
            bool all_ok = true;
            for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                if(!is_candidate[j]) continue;
                if(!cbe_basis_row_is_realizable(pi.CBE_basis_moments[j], CBE_REPAIR_EPS_M_REL)) {
                    all_ok = false; break;
                }
            }
            if(all_ok) {
                f_lo = f_try;
                if(f_try >= 1.0 - 1e-15) break;
            } else {
                f_hi = f_try;
            }
        }
        /* Final: ensure state reflects f_lo (the largest realizable f). */
        for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            if(!is_candidate[j]) continue;
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                pi.CBE_basis_moments[j][k] = saved[j][k];
            const double w = cand_m[j] / cand_m_sum;
            pi.CBE_basis_moments[j][0] += w * f_lo * res_M;
            for(int a = 0; a < NUMDIMS; a++)
                cbe_basis_p_a(pi.CBE_basis_moments[j], a, w * f_lo * res_P[a]);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
            for(int a = 0; a < NUMDIMS; a++) {
                for(int b = a; b < NUMDIMS; b++) {
                    cbe_basis_T_w(pi.CBE_basis_moments[j], a, b,
                        cbe_basis_T_r(pi.CBE_basis_moments[j], a, b) + w * f_lo * res_T[a][b]);
                }
            }
#endif
        }
        f_eff = f_lo;
    }

    /* ----- Step 7: bounded acceptance + deterministic conservative fallback. */
    if(f_eff < 1.0 - 1e-15) {
        const double unabs_f = 1.0 - f_eff;
        const double unab_M    = unabs_f * res_M;
        const double unab_P[3] = { unabs_f*res_P[0], unabs_f*res_P[1], unabs_f*res_P[2] };
        double abs_unab_P = 0.0;
        for(int a = 0; a < NUMDIMS; a++) abs_unab_P += fabs(unab_P[a]);
        double abs_unab_T = 0.0;
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        double unab_T[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for(int a = 0; a < NUMDIMS; a++) {
            for(int b = a; b < NUMDIMS; b++) {
                unab_T[a][b] = unabs_f * res_T[a][b];
                abs_unab_T  += fabs(unab_T[a][b]);
            }
        }
#endif
        const bool M_fp = (fabs(unab_M) <= CBE_REPAIR_EPS_CONS * cell_scale_M);
        const bool P_fp = (abs_unab_P  <= CBE_REPAIR_EPS_CONS * cell_scale_P);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        const bool T_fp = (abs_unab_T  <= CBE_REPAIR_EPS_CONS * cell_scale_T);
#else
        const bool T_fp = true;
#endif
        if(!(M_fp && P_fp && T_fp)) {
            /* Deterministic mass-positive proportional fallback (codex blocker
             * fix 2026-06-04). Replaces the prior "concentrate the FULL
             * unabsorbed residual on a single donor" path which (a) could
             * leave m_donor <= 0 for large negative unab_M, and (b) silently
             * relied on a row-local PSD projection of unbounded magnitude to
             * clean up T.
             *
             * Codex round-2 correction (2026-06-04): the denominator must be
             * the CURRENT cell mass sum (post-Step-6 donation at f_eff), NOT
             * M_new (the pre-donation sum). At Step 7 entry the cell holds
             *   M_cur = M_new + f_eff * res_M
             * so applying (m_b / M_cur) * unab_M gives Sum w_j = 1 exactly
             * and the proportional distribution adds exactly unab_M to the
             * cell sum -> M_after = M_cur + unab_M = M_new + res_M = M0,
             * the target. Using M_new (the pre-donation sum) instead would
             * scale the donation by M_cur/M_new and break conservation when
             * f_eff > 0.
             *
             * Proportional distribution per-row:
             *   m_b   += (m_b / M_cur) * unab_M
             *   p_b   += (m_b / M_cur) * unab_P
             *   T_b   += (m_b / M_cur) * unab_T
             * preserves Sum m_b, Sum p_b, Sum T_b EXACTLY. Mass positivity:
             *   m_b_new = m_b * (M_cur + unab_M) / M_cur = m_b * M0 / M_cur
             * is positive whenever M0 > 0 (Step 0 guard) AND M_cur > 0
             * (Step 6 produces all candidate m_b > 0 by realizability, and
             * Step 3 reset gives empties m_b = m_thresh > 0). Per-row
             * realizability after proportional shift is NOT guaranteed -- the
             * defensive Step 8 sweep below row-locally projects to PSD if
             * needed. Step 8 projection touches only stress slots (preserves
             * m_b and p_b row-locally); trace-delta is bounded by
             *   NBASIS * CBE_SPD_RELATIVE_FLOOR * max_row_trace_central_S
             * and is FP-scale on smooth inputs (verified empirically by the
             * cold/warm Mac smoke; a non-FP magnitude signals upstream
             * pathological cell state surfaced via the dT_out path). */
            double M_cur = 0.0;
            for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) M_cur += pi.CBE_basis_moments[j][0];
            if(M_cur > 0) {
                for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                    const double w_j = pi.CBE_basis_moments[j][0] / M_cur;
                    pi.CBE_basis_moments[j][0] += w_j * unab_M;
                    for(int a = 0; a < NUMDIMS; a++)
                        cbe_basis_p_a(pi.CBE_basis_moments[j], a, w_j * unab_P[a]);
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
                    for(int a = 0; a < NUMDIMS; a++) {
                        for(int b = a; b < NUMDIMS; b++) {
                            cbe_basis_T_w(pi.CBE_basis_moments[j], a, b,
                                cbe_basis_T_r(pi.CBE_basis_moments[j], a, b) + w_j * unab_T[a][b]);
                        }
                    }
#endif
                }
            }
            (void)abs_unab_P; (void)abs_unab_T;   /* used only in fp-test above */
        }
    }

    /* ----- Step 8: defensive final realizability sweep. Should not fire
     * post Step 7 (donor was already projected). Row-local PSD projection
     * preserves m_b individually; trace-delta accumulated into dT_local. */
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
    for(int j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
        if(cbe_basis_row_is_realizable(pi.CBE_basis_moments[j], CBE_REPAIR_EPS_M_REL))
            continue;
        const double m_b = pi.CBE_basis_moments[j][0];
        if(!(m_b > 0)) continue;
        double dT_last = 0.0;
        (void)cbe_basis_row_project_central_stress_to_PSD(
            pi.CBE_basis_moments[j], 0.0, &dT_last);
        dT_local += dT_last;
    }
#endif

    if(dT_out) *dT_out = dT_local;
}


/* Donor-trial helper for the aggregate outflow-budget limiter (commit 2).
 * Tests whether ALL candidate donors stay row-local realizable when fraction
 * f of the excess is redistributed from basis a. Extracted from a captured
 * lambda inside cbe_apply_basis_outflow_budget per codex 2026-06-05 -- a
 * KOKKOS_INLINE_FUNCTION at file scope is device-portable under CUDA/HIP
 * where a captured C++ lambda inside another KOKKOS_INLINE_FUNCTION is a
 * portability hazard.
 *
 * Inputs:
 *   pi          particle (read-only access to moments + moments_dt + Vel)
 *   dt          drift-kick step length
 *   V_old[3]    canonical frame for the absolute-round-trip (== pi.Vel; EXACT
 *               for row-local realizability by frame-invariance)
 *   is_cand[]   per-basis candidate flag set by the limiter loop
 *   cap_b[]     per-basis capacity weights (mass-floor-deducted)
 *   C_sum       Sum of capacities (> 0 on entry, checked by caller)
 *   excess[]    saved-outflow rates per moment slot (positive for slot 0)
 *   f           trial fraction in [0, 1]
 *
 * Returns true iff every candidate donor's projected post-update state
 * (existing rate + limiter correction * dt, frame-converted via V_old) is
 * mass-positive AND row-local realizable. */
KOKKOS_INLINE_FUNCTION
static bool cbe_outbudget_trial_donors_realizable(
    const struct particle_data& pi,
    double dt,
    const double V_old[3],
    const bool   is_cand[CBE_INTEGRATOR_NBASIS],
    const double cap_b[CBE_INTEGRATOR_NBASIS],
    double C_sum,
    const double excess[CBE_INTEGRATOR_NMOMENTS],
    double f)
{
    for(int b = 0; b < CBE_INTEGRATOR_NBASIS; b++) {
        if(!is_cand[b]) continue;
        const double w_b = cap_b[b] / C_sum;
        double m_b_abs;
        double p_abs[3];
        double T_abs[3][3];
        cbe_relative_row_to_absolute(pi.CBE_basis_moments[b], V_old,
                                      &m_b_abs, p_abs, T_abs);
        double rate_corr[CBE_INTEGRATOR_NMOMENTS];
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            rate_corr[k] = pi.CBE_basis_moments_dt[b][k] - w_b * f * excess[k];
        }
        m_b_abs += dt * rate_corr[0];
        if(!(m_b_abs > 0)) return false;
        for(int aa = 0; aa < NUMDIMS; aa++) {
            p_abs[aa] += dt * cbe_basis_p_r(rate_corr, aa);
        }
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        for(int aa = 0; aa < NUMDIMS; aa++) {
            T_abs[aa][aa] += dt * cbe_basis_T_r(rate_corr, aa, aa);
            for(int bb = aa+1; bb < NUMDIMS; bb++) {
                const double dT_ab = dt * cbe_basis_T_r(rate_corr, aa, bb);
                T_abs[aa][bb] += dT_ab; T_abs[bb][aa] += dT_ab;
            }
        }
#endif
        double row_proj[CBE_INTEGRATOR_NMOMENTS];
        cbe_absolute_to_relative_row(m_b_abs, p_abs, T_abs, V_old, row_proj);
        if(!cbe_basis_row_is_realizable(row_proj, CBE_REPAIR_EPS_M_REL)) return false;
    }
    return true;
}


/* Aggregate per-basis outflow-budget limiter (commit 2 of the limiter pair,
 * 2026-06-04; Phil + codex review). For each basis a whose dt*outflow_rate
 * exceeds f_out * m_a, scale the outflow rates by c = f_out*m_a/dt/Udot_out0
 * and redistribute the saved (1-c)*Udot_out among capacity-weighted donor
 * bases (b != a). Cell-summed dt-accumulator is preserved exactly by
 * construction (every excess[k] added to basis a is subtracted from donors
 * weighted to sum to excess[k]).
 *
 * Inputs read:
 *   pi.CBE_basis_moments[b][0]     current mass per basis
 *   pi.CBE_basis_out_rate_dt[a][k] cumulative outgoing-only flux (stored
 *                                  NEGATIVE; -ledger -> positive out rate)
 *   pi.CBE_basis_moments_dt[b][k]  full net flux rate (existing) -- used by
 *                                  the projected-donor realizability check
 *                                  (codex 2026-06-04: donor check MUST
 *                                  include donor's existing rate, not just
 *                                  the limiter correction)
 *   pi.Vel                         V_old for the frame round-trip
 *
 * Inputs modified:
 *   pi.CBE_basis_moments_dt[][]    a += excess[k]; donors -= w_b * f * excess[k]
 *
 * Frame-consistency for the donor PSD check (codex 2026-06-04, EXACT not
 * approximate): row-local realizability `M = m*T - p p^T PSD` is
 * frame-invariant under affine velocity transforms when the absolute<->relative
 * conversion uses the SAME V on both sides. Using V_old here is therefore
 * the right choice -- not an approximation. The drift-kick will later use
 * V_new = MMV from the post-update MMV; the V_new drift does not affect
 * row-local realizability per the same invariance.
 *
 * Projected donor state INCLUDES donor's existing rate (codex 2026-06-04):
 *   U_abs_proj[b] = U_abs_old[b] + dt * (CBE_basis_moments_dt[b] + apply_to_b)
 * NOT just dt * apply_to_b. This catches the failure where a donor is
 * current-state realizable but already near overdrain from its own flux
 * and the limiter's added load tips it past the realizability edge.
 *
 * Bisection on f: if full donation (f=1) breaks any donor's projected
 * realizability, bisect f down to the largest fraction that keeps all
 * donors realizable. The (1-f)*excess REMAINDER is left uncapped on basis
 * a's net rate (a loses more mass than dm_allowed would ideally permit but
 * still less than uncapped); reported via diag->n_donor_limited. NO silent
 * pretense that the cap fully protected basis a.
 *
 * Failure modes (printf/counter only -- NO endrun, NO new diagnostic
 * columns):
 *   NO_DONOR        : no eligible candidate basis (NBASIS=1 effective or
 *                     all bases at floor). Basis a stays UNCAPPED.
 *   DONOR_LIMITED   : bisection found f<1; partial cap applied + remainder
 *                     uncapped. Counted.
 *
 * Diag is nullable. Returns max-severity status code observed.
 *
 * Per [feedback_never_originate_remote_hpc] and the parallel endrun work,
 * this helper deliberately avoids any new endrun codes (Phil + codex
 * 2026-06-04). */
KOKKOS_INLINE_FUNCTION
static int cbe_apply_basis_outflow_budget(
    struct particle_data& pi,
    double dt,
    struct CbeOutflowBudgetDiag *diag)
{
    int status = CBE_OUTBUDGET_OK;
    if(!(dt > 0)) return status;

    const double V_old[3] = { pi.Vel[0], pi.Vel[1], pi.Vel[2] };
    const double f_out    = CBE_OUTFLOW_BUDGET_FRAC;
    const double m_floor  = CBE_OUTFLOW_BUDGET_MIN_DONOR_MASS;

    /* Donor projected-realizability test is in the standalone
     * cbe_outbudget_trial_donors_realizable helper above (file-scope
     * KOKKOS_INLINE_FUNCTION; device-portable per codex 2026-06-05). */

    for(int a = 0; a < CBE_INTEGRATOR_NBASIS; a++) {
        const double m_a       = pi.CBE_basis_moments[a][0];
        const double Udot_out0 = -pi.CBE_basis_out_rate_dt[a][0]; /* positive outflow rate */
        if(!(m_a > 0) || !(Udot_out0 > 0) || !isfinite(m_a) || !isfinite(Udot_out0)) continue;

        const double dm_out     = dt * Udot_out0;
        const double dm_allowed = f_out * m_a;
        const double pre_of     = dm_out / m_a;
        if(diag && pre_of > diag->max_pre_out_frac_seen) diag->max_pre_out_frac_seen = pre_of;
        if(dm_out <= dm_allowed) continue;   /* basis a within budget */

        const double c = dm_allowed / dm_out;            /* c in (0, 1) */
        const double one_minus_c = 1.0 - c;
        double excess[CBE_INTEGRATOR_NMOMENTS];
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            excess[k] = one_minus_c * (-pi.CBE_basis_out_rate_dt[a][k]);
        }

        /* Build candidate-donor set with capacity weights. Capacity is the
         * conservative v1 form `max(0, m_b - m_floor)` -- documented as not
         * mathematically optimal but the projected-state bisection below is
         * the safety rail. */
        bool   is_cand[CBE_INTEGRATOR_NBASIS];
        double cap_b[CBE_INTEGRATOR_NBASIS];
        double C_sum = 0.0;
        for(int b = 0; b < CBE_INTEGRATOR_NBASIS; b++) {
            is_cand[b] = false; cap_b[b] = 0.0;
            if(b == a) continue;
            const double m_b = pi.CBE_basis_moments[b][0];
            if(!(m_b > m_floor) || !isfinite(m_b)) continue;
            /* current-state quick filter; the real check is the projected one below */
            if(!cbe_basis_row_is_realizable(pi.CBE_basis_moments[b], CBE_REPAIR_EPS_M_REL)) continue;
            const double cap = m_b - m_floor;
            if(!(cap > 0)) continue;
            is_cand[b] = true; cap_b[b] = cap; C_sum += cap;
        }
        if(!(C_sum > 0)) {
            if(diag) diag->n_donor_failures++;
            status = (status > CBE_OUTBUDGET_NO_DONOR) ? status : CBE_OUTBUDGET_NO_DONOR;
            continue;   /* basis a uncapped */
        }

        /* Find largest f in [0,1] with all donors projected-realizable. The
         * trial helper is file-scope KOKKOS_INLINE_FUNCTION (above) for
         * device portability. */
        double f_lo;
        if(cbe_outbudget_trial_donors_realizable(pi, dt, V_old, is_cand, cap_b, C_sum, excess, 1.0)) {
            f_lo = 1.0;
        } else {
            f_lo = 0.0;
            double f_hi = 1.0;
            for(int iter = 0; iter < 40; iter++) {
                const double f_mid = 0.5 * (f_lo + f_hi);
                if(cbe_outbudget_trial_donors_realizable(pi, dt, V_old, is_cand, cap_b, C_sum, excess, f_mid))
                    f_lo = f_mid;
                else
                    f_hi = f_mid;
                if((f_hi - f_lo) < 1e-14) break;
            }
        }

        /* Commit at f=f_lo. The (1-f_lo)*excess REMAINDER stays on basis a's
         * net rate uncapped -- explicitly NOT a hidden full cap. */
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            pi.CBE_basis_moments_dt[a][k] += f_lo * excess[k];
            for(int b = 0; b < CBE_INTEGRATOR_NBASIS; b++) {
                if(!is_cand[b]) continue;
                const double w_b = cap_b[b] / C_sum;
                pi.CBE_basis_moments_dt[b][k] -= w_b * f_lo * excess[k];
            }
        }

        if(f_lo >= 1.0 - 1e-15) {
            if(diag) diag->n_capped_bases++;
            status = (status >= CBE_OUTBUDGET_APPLIED) ? status : CBE_OUTBUDGET_APPLIED;
        } else {
            if(diag) { diag->n_donor_limited++; diag->n_donor_failures++; }
            status = (status > CBE_OUTBUDGET_DONOR_LIMITED) ? status : CBE_OUTBUDGET_DONOR_LIMITED;
        }
    }

    return status;
}


/* GPU-callable per-particle drift-kick update for the CBE integrator.
 * Mirrors do_cbe_drift_kick() in cbe_integrator.cc but takes an explicit
 * particle ref so it runs in both CPU and GPU (Kokkos) contexts.
 *
 * Wave-CBE Fix #2b (2026-06-04): the legacy in-place realizability chain
 * (active-diagonal floor + 3D Cauchy-Schwarz off-diag + det crossnorm +
 * SPD projection + stochastic split-largest) is replaced by a single call
 * to cbe_repair_cell_conservative_basis_states. The 2026-06-04 absolute
 * round-trip (nfac + frame conversion + V_new derivation) is unchanged.
 *
 * *dT_out (nullable) receives the sum-over-bases trace-delta from PSD
 * projections inside the repair helper; caller passes nullptr to skip
 * diagnostic bookkeeping. dP is identically 0 from this kernel. */
KOKKOS_INLINE_FUNCTION
static void do_cbe_drift_kick_kernel(struct particle_data& pi, double dt,
                                     double *dT_out)
{
    /* Codex 2026-06-04 finite absolute-update round-trip.
     *
     * Per particle, per step:
     *   1. Read RELATIVE storage in current pi.Vel frame (V_old).
     *   2. Convert each basis row to ABSOLUTE (m, p_abs, T_abs) via the
     *      SSOT cbe_relative_row_to_absolute helper.
     *   3. Apply dt * dt_accumulator IN ABSOLUTE FRAME (the flux deposited
     *      the rates there; postgrav now leaves them in absolute except for
     *      a mass-closure safety net).
     *   4. Derive V_new from updated mass-weighted-mean basis velocity:
     *         V_new[k] = Σ_b p_abs_new_b[k] / Σ_b m_new_b
     *      This makes Σ p_rel = 0 exact by construction after step 5 and
     *      eliminates the need to inject a_cbe into pi.GravAccel.
     *   5. Convert ABSOLUTE → RELATIVE storage in V_new frame via the SSOT
     *      cbe_absolute_to_relative_row helper.
     *   6. Set pi.Vel = V_new (CBE-induced bulk velocity is now derived,
     *      NOT routed through GravAccel; gravity kicks are independent).
     *   7. Conservative cell-state realizability repair via SSOT helper
     *      cbe_repair_cell_conservative_basis_states (Fix #2b, 2026-06-04).
     *
     * The nfac rate-limit (capping per-basis |dm|/m at 0.75) is preserved
     * from the legacy operator as a robustness safeguard. The legacy
     * mass + momentum closure corrections (per-basis subtraction of
     * m_b/M_total · dmom_tot[k]) are subsumed by step 4: deriving V_new
     * from the updated MMV gives Σ p_rel = 0 by construction; mass closure
     * is already enforced upstream by postgrav.
     *
     * dT_out (nullable) accumulates the sum-over-bases trace-delta from
     * PSD projections inside the Step 7 repair helper (for
     * cbe_diagnostics.txt col-8). dP from this kernel is identically zero
     * (repair preserves Σ p_rel and projection only touches stress slots). */

    /* Aggregate outflow-budget limiter (commit 2, 2026-06-04 Phil + codex).
     * Runs BEFORE nfac so subsequent steps operate on the capped rates.
     * No-op on the cold/warm 1D smoke tests we validate against (basis
     * outflows are well within f_out*m_a); fires only in the scratch-basis
     * runaway regime it was designed for. */
    {
        struct CbeOutflowBudgetDiag obdiag_local{};
        int obstatus = cbe_apply_basis_outflow_budget(pi, dt, &obdiag_local);
#ifdef CBE_INTEGRATOR_OUTBUDGET_VERBOSE
        /* Opt-in verbose smoke verification (codex 2026-06-04): default OFF.
         * Enable by adding CBE_INTEGRATOR_OUTBUDGET_VERBOSE to Config.sh and
         * grep stdout for "[CBE outbudget]" -- zero hits on clean smoke =
         * limiter did not fire. Device-safe printf (no fflush/endrun). */
        if(obdiag_local.n_capped_bases > 0 || obdiag_local.n_donor_limited > 0
                                            || obdiag_local.n_donor_failures > 0) {
            printf("[CBE outbudget] ID=%lld caps=%lld donor_lim=%lld donor_fail=%lld "
                   "max_pre_out_frac=%.3e status=%d\n",
                   (long long)pi.ID,
                   (long long)obdiag_local.n_capped_bases,
                   (long long)obdiag_local.n_donor_limited,
                   (long long)obdiag_local.n_donor_failures,
                   obdiag_local.max_pre_out_frac_seen, obstatus);
        }
#else
        (void)obstatus;
#endif
    }

    /* Step 0 (preserved): nfac rate-limit on per-basis mass change. */
    double dmass_tot = 0;
    for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) dmass_tot += dt * pi.CBE_basis_moments_dt[j][0];
    const double minv = (pi.Mass > 0) ? 1.0 / pi.Mass : 0.0;
    double biggest_dm = 1.e10;
    for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
        const double m_b = pi.CBE_basis_moments[j][0];
        if(!(m_b > 0)) continue;
        const double q = (dt*pi.CBE_basis_moments_dt[j][0] - m_b*minv*dmass_tot)
                       / (m_b * (1.0 + minv*dmass_tot));
        if(!isnan(q) && q < biggest_dm) biggest_dm = q;
    }
    double nfac = 1.0;
    if(biggest_dm < -0.75) nfac = -0.75 / biggest_dm;

    /* Step 1+2: read relative storage and convert each basis to absolute
     * using V_old = current pi.Vel. */
    const double V_old[3] = { pi.Vel[0], pi.Vel[1], pi.Vel[2] };
    double m_b_arr[CBE_INTEGRATOR_NBASIS];
    double p_abs_arr[CBE_INTEGRATOR_NBASIS][3];
    double T_abs_arr[CBE_INTEGRATOR_NBASIS][3][3];
    for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
        cbe_relative_row_to_absolute(pi.CBE_basis_moments[j], V_old,
                                     &m_b_arr[j], p_abs_arr[j], T_abs_arr[j]);
    }

    /* Step 3: apply nfac-limited absolute-frame update. dt_accumulator is
     * in absolute frame post-postgrav (which now does only mass closure). */
    double m_new_total = 0;
    double p_abs_new_total[3] = {0,0,0};
    for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
        m_b_arr[j] += nfac * dt * pi.CBE_basis_moments_dt[j][0];
        m_new_total += m_b_arr[j];
        for(int a=0;a<NUMDIMS;a++) {
            p_abs_arr[j][a] += nfac * dt * cbe_basis_p_r(pi.CBE_basis_moments_dt[j], a);
            p_abs_new_total[a] += p_abs_arr[j][a];
        }
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        for(int a=0;a<NUMDIMS;a++) {
            T_abs_arr[j][a][a] += nfac * dt * cbe_basis_T_r(pi.CBE_basis_moments_dt[j], a, a);
            for(int b=a+1;b<NUMDIMS;b++) {
                const double dT_ab = nfac * dt * cbe_basis_T_r(pi.CBE_basis_moments_dt[j], a, b);
                T_abs_arr[j][a][b] += dT_ab; T_abs_arr[j][b][a] += dT_ab;
            }
        }
#endif
    }

    /* Step 4: derive V_new from new mass-weighted-mean basis velocity. */
    double V_new[3] = {0,0,0};
    if(m_new_total > 0) {
        for(int a=0;a<NUMDIMS;a++) V_new[a] = p_abs_new_total[a] / m_new_total;
    } else {
        for(int a=0;a<3;a++) V_new[a] = V_old[a];   /* defensive — m≤0 cell */
    }

    /* Step 5: convert each basis back to relative storage in V_new frame
     * via SSOT helper. By construction Σ p_rel = 0 to machine eps. */
    for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
        cbe_absolute_to_relative_row(m_b_arr[j], p_abs_arr[j], T_abs_arr[j],
                                     V_new, pi.CBE_basis_moments[j]);
    }

    /* Step 6: pi.Vel <- V_new (CBE-induced bulk velocity, derived not
     * GravAccel-injected). Gravity kicks against pi.GravAccel remain
     * independent and apply only EXTERNAL gravitational acceleration. */
    for(int a=0;a<NUMDIMS;a++) pi.Vel[a] = V_new[a];

    /* Step 7: conservative cell-state realizability repair (Fix #2b).
     * Replaces the legacy in-place chain (active-diagonal floor + 3D
     * Cauchy-Schwarz off-diag + det crossnorm + SPD projection +
     * stochastic split-largest) with the single SSOT helper defined
     * above. Operates on the now-frame-consistent relative storage. */
    cbe_repair_cell_conservative_basis_states(pi, dT_out);
}

/* GPU-callable per-particle post-gravity finalization for the CBE integrator.
 * Codex 2026-06-04: this routine is now minimal — it ONLY enforces basis-
 * mass closure (Σ_α dm_α/dt = 0 to FP) as a safety net. CBE bulk velocity
 * is NOT injected into pi.GravAccel anymore; the absolute round-trip in
 * do_cbe_drift_kick_kernel handles frame conversion and derives V_new from
 * the post-update mass-weighted-mean basis velocity.
 *
 * Moment ordering (unchanged): 0, x, y, z, xx, yy, zz, xy, xz, yz.
 *
 * Operations (pure i-side):
 *   - Sum dmom_tot[0] = Σ_α dm_α/dt across bases.
 *   - Subtract that residual from each basis's dm_α/dt weighted by m_α/M
 *     so the cell-summed mass rate is exactly zero (upstream MFM should
 *     deliver this to FP already; this is the safety net).
 *   - Leave all other dt-accumulator slots (momentum, stress) untouched —
 *     they stay in absolute frame for the drift-kick round-trip to consume.
 */
KOKKOS_INLINE_FUNCTION
static void do_cbe_postgravity_kernel(struct particle_data& pi)
{
    /* Codex 2026-06-04 rewrite. The OLD postgrav (a) injected a_cbe into
     * pi.GravAccel — routing CBE bulk velocity through the gravity kick —
     * and (b) converted dt-accumulator's per-basis dp_abs/dT_abs to relative
     * frame using a continuous-time differential formula. Both were
     * structural bugs: (a) double-counted CBE bulk velocity (see the
     * 2026-06-04 SPIKE A/B), and (b) lacked the O(dt²) finite-step terms
     * needed for V-changing-during-step accuracy.
     *
     * The new design moves the frame conversion into a finite absolute
     * round-trip inside do_cbe_drift_kick_kernel (see SSOT helpers
     * cbe_relative_row_to_absolute / cbe_absolute_to_relative_row above).
     * The flux already deposits dt-accumulator in absolute frame; postgrav
     * leaves those rates alone. CBE-induced bulk velocity is now derived
     * from the post-update mass-weighted-mean basis velocity inside the
     * drift-kick operator and written directly to pi.Vel. pi.GravAccel
     * carries ONLY external (real) gravitational acceleration.
     *
     * The only postgrav job left is mass-closure: enforce Σ_α dm_α/dt = 0
     * exactly as a safety net (upstream MFM should already deliver this
     * to machine eps). This is frame-invariant and is identical in absolute
     * and relative formulations. */
    double dmom_tot_mass = 0;
    for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
        dmom_tot_mass += pi.CBE_basis_moments_dt[j][0];
    }
    if(pi.Mass > 0) {
        const double m_inv = 1.0 / pi.Mass;
        for(int j=0;j<CBE_INTEGRATOR_NBASIS;j++) {
            pi.CBE_basis_moments_dt[j][0] -= pi.CBE_basis_moments[j][0] * (m_inv * dmom_tot_mass);
        }
    }
}

#endif /* CBE_INTEGRATOR */

#endif /* CBE_INTEGRATOR_FUNCTIONS_H */
