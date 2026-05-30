/*
 * sidm/cbe_integrator_gradients.h
 *
 * CBE pre-force gradient module — Wave-CBE Phase 2 commit #5 corrective
 * architecture pivot (2026-05-26, v5). Replaces the earlier scratch-based
 * commits #1-#4 (CbeGradScratch + custom Alltoallv + raw UVM pointer in
 * DeviceContext) with the persistent-particle-field shape used by hydro
 * and DMGrad. Persistent storage holds the spatial gradients of the
 * reconstructed flux-frame basis moments Q = U/V (NOT the gradients of the
 * raw integrated U), per basis, per moment slot, per direction:
 *   double P[i].Gradients_CBE_basis_moments[NBASIS][NMOMENTS][3]
 * (declarations/particle_data.h, gated on CBE_INTEGRATOR_WITHGRADIENTS).
 * The flux body reconstructs Q_face = Q ± psi · (grad_Q · dp) using these.
 *
 * Standard GIZMO ghost import (gizmo_request_filtered_ghost_import_fresh)
 * carries the gradient field naturally as part of P[]. There is no scratch
 * UVM buffer, no custom Alltoallv, and no raw gradient pointer in any
 * DeviceContext. The earlier scratch-based architecture (CbeGradScratch,
 * CbeGradScratchOwner, cbe_grad_import_ghosts, CbeGradientsSpec,
 * CbeBjLimiterSpec) is gone; design history in
 * project memory OPEN_cbe_wave_port_design §"COMMIT #5 v5".
 *
 * Toplevel `CBEGrad_gradient_calc()` runs pre-force (called from
 * core/accel.cc, paralleling DMGrad_gradient_calc) and refreshes
 * P[i].Gradients_CBE_basis_moments for every currently-active particle —
 * active set is AGSForce_isactive(i), same predicate as the force consumer
 * AGSForce_calc. Inactive particles retain their previous-step gradient
 * (hydro semantics). Subgroup partitioning by ags_gravity_kernel_shared_BITFLAG
 * is done at the toplevel via per-bm `args.neighbor_type_mask_override`
 * calls — same pattern as DMGrad.
 *
 * Two passes orchestrated at the toplevel via Aux::loop_iteration:
 *   pass 0  — raw MFM-LSQ gradient (M += w*dp*dp^T, B += w*dQ*(x_j - x_i));
 *             apply_active_writeback solves M^{-1}.B with Tikhonov-style
 *             ill-conditioning guard and writes
 *             P[i].Gradients_CBE_basis_moments.
 *   pass 1  — pairwise BJ-style conservative limiter (per-(m,k) phi from
 *             cbe_bj_phi_pair, min over matched neighbors); writeback
 *             rescales P[i].Gradients_CBE_basis_moments in place by phi.
 *
 * "Pairwise BJ-style conservative limiter": a stricter per-pair matched-
 * basis local-extremum form, NOT the full global-stencil Barth-Jespersen
 * (which takes Q_min/Q_max over the whole neighbor stencil). The name is
 * load-bearing — keep the qualifier.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef CBE_INTEGRATOR_GRADIENTS_H
#define CBE_INTEGRATOR_GRADIENTS_H

/* Kokkos_Core MUST come BEFORE declarations/allvars.h — same convention
 * as ags_force_loop.h / dm_fuzzy_loop.h. */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"

#if defined(CBE_INTEGRATOR_WITHGRADIENTS)

#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"
#include "../gravity/ags_functions.h"       /* get_particle_volume_ags_P */
#include "cbe_integrator_functions.h"       /* cbe_build_flux_frame_Q_from_stored_moments,
                                               cbe_cost_v_only, cbe_assign_outgoing_greedy */
/* NOTE: caller TUs MUST include "../mesh/kernel.h" before this header
 * (kernel.h lacks include guards; same convention as ags_force_loop.h). */

int AGSForce_isactive(int i);   /* forward decl; defined in gravity/ags_force_loop.cc */

/* Toplevel — defined in sidm/cbe_integrator_gradients.cc; also declared
 * (guarded) in core/proto.h. */
void CBEGrad_gradient_calc(void);

/* ============================================================================
 * Spec types.
 * ========================================================================== */

/* Per-call cosmology + the external gradient-pass index. populate_call_scalars
 * copies loop_iteration out of Aux; pair_kernel + apply_active_writeback
 * branch on it. Same shape as DMGradCallScalars. */
struct CBEGradCallScalars {
    NlrCommonScalars common;
    int              loop_iteration;     /* 0 = LSQ; 1 = pairwise BJ-style limiter */
};

/* Per-active local fill. Carries x_i, h_i, V_i, the stored moment array U_i,
 * Vel_i for Q_i construction, AND a by-value snapshot of the persistent
 * gradient row P[i].Gradients_CBE_basis_moments. The gradient snapshot is
 * Mode-B-safe envelope-transit (design invariant I2): pass-1 pair_kernel
 * reads active.local.Gradients_CBE_basis_moments instead of dereferencing
 * peer-rank storage. Populated unconditionally by load_active — pass 0
 * (LSQ) does not consume it, but the cost is fixed-size (NBASIS*NMOMENTS*3
 * doubles, ≤ 960 B worst case) and the unified shape keeps the load_active
 * branch-free. */
struct CBEGradLocalIn {
    double         Mass;
    double         AGS_KernelRadius;
    Vec3<double>   Pos;
    Vec3<double>   Vel;
    double         V_i;
    double         U_i[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    /* By-value snapshot of P[i].Gradients_CBE_basis_moments — written by
     * the previous CBEGrad_gradient_calc pass (or zero-init if first call).
     * Pass 1 limiter reads this; pass 0 LSQ does not consume it. */
    double         Gradients_CBE_basis_moments[CBE_INTEGRATOR_NBASIS]
                                              [CBE_INTEGRATOR_NMOMENTS][3];
};

/* Per-active accumulator. Both passes share one POD:
 *   pass 0 (LSQ):  M (3x3, sum-merge) + B[m][k][d] (sum-merge); phi stays 1.0
 *   pass 1 (row-scalar BJ + cone): phi[m][k] (min-merge, NaN-aware);
 *          M, B stay 0
 * merge_accum unconditionally sums M+B AND mins phi — the unused-pass
 * fields are neutral elements (0 for sum, 1 for min) so the per-pass
 * branch isn't needed in merge_accum.
 *
 * Semantically post Wave-CBE Commit 10 (Fix #6 cone limiter) pass 1 is
 * ROW-SCALAR: a single phi per basis row m is produced per neighbor,
 * and the same value is written to every k in row m. The per-(m,k)
 * storage layout is preserved for ABI stability; after min-merge over
 * neighbors all k in a given row therefore carry the same value, and
 * the per-active writeback rescales the whole row coupled (m, p, T)
 * by that scalar. Mixing different phi across k in a row would move
 * the reconstructed state off the line segment the cone predicate
 * validated and could reintroduce non-realizability.
 *
 * MUST be a POD; zero_accum sets M=0, B=0, phi=1.0 explicitly. */
struct CBEGradOut {
    double M[3][3];
    double B[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS][3];
    double phi[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
};

struct CBEGradActiveState {
    Vec3<double>       pos;
    double             h_search;
    CBEGradLocalIn     local;
    short int          TimeBin;
    CBEGradCallScalars scalars;
    int                origin_local_idx;
    int                origin_rank;
};

/* ============================================================================
 * Inline pair-kernel helpers (KOKKOS_INLINE_FUNCTION; instantiated from the
 * GPU TU that holds the runner instantiation).
 * ========================================================================== */

/* Pass-0 pair body — MFM-LSQ M and B accumulation. */
template <typename NeighborT>
KOKKOS_INLINE_FUNCTION
static void cbe_grad_lsq_pair_kernel_body(const CBEGradActiveState& active,
                                           const NeighborT&          neighbor,
                                           CBEGradOut&               accum)
{
    const int             j   = neighbor.j;
    struct particle_data &Pj  = *neighbor.neighbor_particle;
    const CBEGradLocalIn& L   = active.local;

    if(!(Pj.Mass > 0) || !(Pj.AGS_KernelRadius > 0)) return;

    /* dp = x_i - x_j. The B accumulation uses (-dp) = (x_j - x_i) so
     * apply_active_writeback solves M . grad = sum_j w * dQ * (x_j - x_i),
     * the design-doc convention. M is even in the sign of dp. */
    Vec3<double> dp;
    dp[0] = L.Pos[0] - Pj.Pos[0];
    dp[1] = L.Pos[1] - Pj.Pos[1];
    dp[2] = L.Pos[2] - Pj.Pos[2];
    NEAREST_XYZ(dp[0], dp[1], dp[2], -1);

    const double r2  = dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2];
    const double h_i = L.AGS_KernelRadius;
    if(r2 <= 0) return;
    /* h_i-only acceptance, DESPITE the Spec's symmetric search topology.
     * Matches the raw-gradient semantics of hydro / density / DMGrad:
     * linear consistency of the LSQ estimator relies on the i-kernel
     * weighting + NV_T-style normalisation. The Spec uses symmetric search
     * because pass 1 (pairwise BJ-style limiter) needs the broader face
     * visibility; pass 0 deliberately re-narrows here. Do NOT widen this
     * to a symmetric r <= h_i+h_j (or r <= max(h_i,h_j)) filter. */
    if(r2 >= h_i * h_i) return;

    const double r = sqrt(r2);
    double hinv_i, hinv3_i, hinv4_i;
    kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);

    const double u_i = r * hinv_i;
    double       wk_i, dwk_i;
    kernel_main(u_i, hinv3_i, hinv4_i, &wk_i, &dwk_i, 0);

    /* Volume per neighbor — canonical AGS volume helper. Same call the
     * flux body uses (sidm/cbe_integrator_flux_functions.h). */
    const double V_j = get_particle_volume_ags_P(j, neighbor.P_base);

    const double w = wk_i * V_j;

    /* M_i += w * dp dp^T (outer product even in sign of dp). */
    for(int a = 0; a < 3; a++) {
        for(int b = 0; b < 3; b++) {
            accum.M[a][b] += w * dp[a] * dp[b];
        }
    }

    /* Build Q_i, Q_j with the SSOT helper used by the flux body. */
    const double cf_a3inv = active.scalars.common.cf_a3inv;
    const double cf_atime = active.scalars.common.cf_atime;

    double Q_i[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double Q_j[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    {
        double Vel_i_code[3] = { L.Vel[0], L.Vel[1], L.Vel[2] };
        cbe_build_flux_frame_Q_from_stored_moments(
            L.U_i, Vel_i_code, L.V_i, cf_a3inv, cf_atime, Q_i);
    }
    {
        double Vel_j_code[3] = { (double)Pj.Vel[0], (double)Pj.Vel[1], (double)Pj.Vel[2] };
        cbe_build_flux_frame_Q_from_stored_moments(
            Pj.CBE_basis_moments, Vel_j_code, V_j, cf_a3inv, cf_atime, Q_j);
    }

    /* Basis matching via SSOT helper (Wave-CBE Commit 6b). Uses the same
     * cost function and assignment rule as the limiter pass and the flux
     * body. By design the gradient pre-pass matches on cell-center Q_i /
     * Q_j (Q_face does not exist yet -- it is the output of this and the
     * limiter pass) while the flux body matches on face-reconstructed
     * Qface_i / Qface_j; the per-pair pair identities can therefore
     * differ between the two states, but the matching function itself is
     * shared. This is the architected Q-cell/Q-face split, not a pending
     * fix. Single-direction (a->b only); fired-count counter is NULL. */
    int matched_j_for_i[CBE_INTEGRATOR_NBASIS];
    cbe_build_pair_matching(Q_i, Q_j, matched_j_for_i,
                            /*alpha_of_beta_for_b=*/NULL,
                            /*free_slot_fired_count_inout=*/NULL);

    /* B_i[m][k][e] += w * (Q_j_matched - Q_i)[m][k] * (x_j - x_i)[e]. With
     * dp = x_i - x_j, (x_j - x_i)[e] = -dp[e]. */
    for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
        const int n = matched_j_for_i[m];
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            const double dQ = Q_j[n][k] - Q_i[m][k];
            const double w_dQ = w * dQ;
            accum.B[m][k][0] += w_dQ * (-dp[0]);
            accum.B[m][k][1] += w_dQ * (-dp[1]);
            accum.B[m][k][2] += w_dQ * (-dp[2]);
        }
    }
    (void)j;
}

/* ----------------------------------------------------------------------------
 * Pass-1 helper: pairwise BJ-style phi. Bound the reconstructed delta
 * `predicted` so that Q_face = Q_i + predicted stays between Q_i and
 * Q_j_matched.
 *
 *   - non-finite anywhere       → 0.0
 *   - |predicted| <= tol        → 1.0  (no reconstruction; safe)
 *   - sign(predicted) != sign(dQ) → 0.0 (heads away from neighbor)
 *   - same sign, |pred| > |dQ|  → dQ/predicted          (∈ (0, 1))
 *   - same sign, |pred| <= |dQ| → 1.0                    (no overshoot)
 *
 * Combined: phi = clamp(dQ / predicted, 0, 1) when |predicted| > tol.
 * Caller takes min across (j, m, k).
 * ---------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
static double cbe_bj_phi_pair(double dQ, double predicted)
{
    if(!isfinite(dQ) || !isfinite(predicted)) return 0.0;
    /* Scale-aware tolerance: if |predicted| is below the roundoff floor of
     * |dQ| plus an absolute underflow guard, treat as "no reconstruction
     * in this direction" → phi=1. Prevents spurious phi=0 from sign-noise
     * when grad . dp comes in well below the neighbor-difference scale. */
    const double tol = fabs(dQ) * 1e-14 + MIN_REAL_NUMBER;
    if(fabs(predicted) <= tol) return 1.0;
    const double ratio = dQ / predicted;
    if(!isfinite(ratio)) return 0.0;
    if(ratio < 0.0) return 0.0;
    if(ratio > 1.0) return 1.0;
    return ratio;
}


/* ----------------------------------------------------------------------------
 * SSOT 3x3 symmetric PSD test via all principal minors (Wave-CBE Commit 10,
 * Fix #6). Returns true iff every principal minor of M is >= 0:
 *   - 3 diagonal minors M_kk
 *   - 3 leading 2x2 minors M_kk*M_ll - M_kl^2
 *   - the 3x3 determinant
 * Sylvester-style check sufficient for PSD on a symmetric matrix; the
 * realizability set {S PSD} we care about is closed under intersection
 * with any line in (m, p, T) space (see the convexity argument at
 * cbe_cone_phi_row).
 * ---------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
static bool cbe_sym3x3_all_principal_minors_nonneg(const double M[3][3])
{
    if(M[0][0] < 0) return false;
    if(M[1][1] < 0) return false;
    if(M[2][2] < 0) return false;
    if(M[0][0]*M[1][1] - M[0][1]*M[0][1] < 0) return false;
    if(M[0][0]*M[2][2] - M[0][2]*M[0][2] < 0) return false;
    if(M[1][1]*M[2][2] - M[1][2]*M[1][2] < 0) return false;
    const double det = M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[1][2])
                     - M[0][1] * (M[0][1]*M[2][2] - M[1][2]*M[0][2])
                     + M[0][2] * (M[0][1]*M[1][2] - M[1][1]*M[0][2]);
    return (det >= 0);
}


/* ----------------------------------------------------------------------------
 * Realizability predicate for one basis row at trial phi (Wave-CBE
 * Commit 10, Fix #6). Constructs Q_face = Q_cell_row + phi * g_row and
 * checks the two CBE realizability conditions on the face state:
 *   (1) mass positivity Q_face[0] > MIN_REAL_NUMBER
 *   (2) central stress S = (T*m - p(x)p) / m^2 PSD, S_floor = 0
 * For NMOMENTS=10 (full 3D second moment) condition (2) is the full
 * symmetric-3x3-PSD check via cbe_sym3x3_all_principal_minors_nonneg.
 * For NMOMENTS=7 the off-diagonal central S is absent (Commit 8
 * convention), so the stress 3x3 is diagonal at the row level; zeroing
 * the off-diagonal M entries reduces the PSD check to the three
 * diagonal conditions naturally. NMOMENTS=4 has no stress slot and the
 * predicate reduces to mass positivity.
 *
 * Strict >=0 comparisons (no slack) are conservative: if roundoff trims
 * a few percent off allowable phi the cone over-limits slightly, never
 * under-limits. Tolerance can be added later if profiling shows
 * over-clipping.
 * ---------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
static bool cbe_row_realizable_at_phi(
    const double Q_cell_row[CBE_INTEGRATOR_NMOMENTS],
    const double g_row[CBE_INTEGRATOR_NMOMENTS],
    double phi)
{
    double Q_face[CBE_INTEGRATOR_NMOMENTS];
    for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
        Q_face[k] = Q_cell_row[k] + phi * g_row[k];
    }
    if(!(Q_face[0] > MIN_REAL_NUMBER)) return false;
#if (CBE_INTEGRATOR_NMOMENTS >= 7)
    {
        const double m  = Q_face[0];
        const double p[3] = { Q_face[1], Q_face[2], Q_face[3] };
        double M[3][3];
        M[0][0] = Q_face[4] * m - p[0]*p[0];
        M[1][1] = Q_face[5] * m - p[1]*p[1];
        M[2][2] = Q_face[6] * m - p[2]*p[2];
    #if (CBE_INTEGRATOR_NMOMENTS >= 10)
        M[0][1] = M[1][0] = Q_face[7] * m - p[0]*p[1];
        M[0][2] = M[2][0] = Q_face[8] * m - p[0]*p[2];
        M[1][2] = M[2][1] = Q_face[9] * m - p[1]*p[2];
    #else
        /* NMOMENTS=7: off-diagonal central S absent (not zero -- per
         * Commit 8 convention). Zero the off-diagonal M entries so the
         * full-3x3 PSD helper reduces correctly to diagonal-only. */
        M[0][1] = M[1][0] = 0;
        M[0][2] = M[2][0] = 0;
        M[1][2] = M[2][1] = 0;
    #endif
        if(!cbe_sym3x3_all_principal_minors_nonneg(M)) return false;
    }
#endif
    return true;
}


/* ----------------------------------------------------------------------------
 * Row-scalar realizability cone limiter (Wave-CBE Commit 10, Fix #6).
 *
 * The CBE per-basis realizable set { Q : m > 0, m*T - p(x)p PSD } is
 * convex: by the Schur-complement form, m > 0 AND m*T - p(x)p PSD is
 * equivalent to the block matrix [[T, p], [p^T, m]] being PSD, the
 * intersection of a convex cone with the {m > 0} half-space. Its
 * intersection with the line Q(phi) = Q_cell + phi * g is therefore
 * an interval containing phi = 0 whenever the cell-center row is
 * realizable. Bisection on the predicate cbe_row_realizable_at_phi
 * keeps the largest known-realizable phi as lo; the returned phi is
 * always provably realizable. No monotonicity per individual principal
 * minor is claimed -- the predicate is bisected directly.
 *
 * Defensive: if the cell-center row is itself non-realizable (should
 * not occur post drift-kick repair; if it does, that is a cell-state
 * repair bug, not cone limiter behavior), return 0. Device-safe.
 *
 * S_floor = 0 default. Bracket upper bound is the row-scalar BJ phi
 * for this pair (min over k of per-(m,k) BJ phi); the cone only ever
 * tightens that bound. Scale-aware termination with tol_rel = 1e-14
 * matches the v_F root-find precision; 60-iter cap is plenty for the
 * bracket widths in play.
 * ---------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
static double cbe_cone_phi_row(
    const double Q_cell_row[CBE_INTEGRATOR_NMOMENTS],
    const double g_row[CBE_INTEGRATOR_NMOMENTS],
    double phi_BJ_upper)
{
    if(!(phi_BJ_upper > 0)) return 0;
    if(!cbe_row_realizable_at_phi(Q_cell_row, g_row, 0.0)) return 0;
    if(cbe_row_realizable_at_phi(Q_cell_row, g_row, phi_BJ_upper)) return phi_BJ_upper;
    double lo = 0;
    double hi = phi_BJ_upper;
    for(int it = 0; it < 60; it++) {
        const double scale = DMAX(fabs(lo), fabs(hi));
        if((hi - lo) <= 1e-14 + 1e-14 * scale) break;
        const double mid = 0.5 * (lo + hi);
        if(cbe_row_realizable_at_phi(Q_cell_row, g_row, mid)) lo = mid;
        else                                                  hi = mid;
    }
    return lo;
}


/* Pass-1 pair body — pairwise BJ-style conservative limiter. Reads
 * active.local.Gradients_CBE_basis_moments (by-value snapshot from
 * load_active — Mode-B-safe, NOT a peer-rank dereference). */
template <typename NeighborT>
KOKKOS_INLINE_FUNCTION
static void cbe_grad_bj_pair_kernel_body(const CBEGradActiveState& active,
                                          const NeighborT&          neighbor,
                                          CBEGradOut&               accum)
{
    struct particle_data &Pj  = *neighbor.neighbor_particle;
    const CBEGradLocalIn& L   = active.local;
    const int             j   = neighbor.j;

    if(!(Pj.Mass > 0) || !(Pj.AGS_KernelRadius > 0)) return;

    /* dp = x_i - x_j. Mirrors the LSQ pass and the flux body. */
    Vec3<double> dp;
    dp[0] = L.Pos[0] - Pj.Pos[0];
    dp[1] = L.Pos[1] - Pj.Pos[1];
    dp[2] = L.Pos[2] - Pj.Pos[2];
    NEAREST_XYZ(dp[0], dp[1], dp[2], -1);

    const double r2  = dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2];
    const double h_i = L.AGS_KernelRadius;
    if(r2 <= 0) return;
    const double r   = sqrt(r2);
    const double h_j = (double)Pj.AGS_KernelRadius;

    /* Pair-overlap filter mirrors AgsForceSpec exactly so the limiter sees
     * every face the flux body will subsequently reconstruct. Legacy
     * gravity/ags_force_loop.h:271-278: r <= h_i+h_j under DM_SIDM;
     * r <= max(h_i, h_j) otherwise (symmetric). */
#if defined(DM_SIDM)
    if(r > h_i + h_j) return;
#else
    if(r > h_i && r > h_j) return;
#endif

    /* MFM face-position weight (matches the flux body's reconstruction:
     * psi_i = h_j/(h_i+h_j); face offset on i's side = -psi_i * dp). */
    const double psi_i_denom = h_i + h_j;
    if(!(psi_i_denom > 0)) return;
    const double psi_i = h_j / psi_i_denom;

    /* Volume per neighbor — same helper as gradient and flux bodies. */
    const double V_j = get_particle_volume_ags_P(j, neighbor.P_base);

    /* Build Q_i and Q_j with the SSOT helper — same matching keeps the
     * limiter consistent with the LSQ pass and the flux body. */
    const double cf_a3inv = active.scalars.common.cf_a3inv;
    const double cf_atime = active.scalars.common.cf_atime;

    double Q_i[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double Q_j[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    {
        double Vel_i_code[3] = { L.Vel[0], L.Vel[1], L.Vel[2] };
        cbe_build_flux_frame_Q_from_stored_moments(
            L.U_i, Vel_i_code, L.V_i, cf_a3inv, cf_atime, Q_i);
    }
    {
        double Vel_j_code[3] = { (double)Pj.Vel[0], (double)Pj.Vel[1], (double)Pj.Vel[2] };
        cbe_build_flux_frame_Q_from_stored_moments(
            Pj.CBE_basis_moments, Vel_j_code, V_j, cf_a3inv, cf_atime, Q_j);
    }

    /* Basis matching via SSOT helper (Wave-CBE Commit 6b) — same selector
     * dispatch / cost policy as the LSQ pre-pass and the flux body, so
     * limiter matching uses the same cost function and assignment rule
     * across all consumers. The matched pair from this Q_cell-state call
     * can differ from the flux body's Q_face-state call by design: the
     * gradient/limiter cannot match on Q_face since Q_face is the output
     * of this pass, and the flux must match on the state it actually
     * consumes. Both use the same SSOT helper with the same selectors.
     * Single-direction; counter NULL (not a flux-pairing decision). */
    int matched_j_for_i[CBE_INTEGRATOR_NBASIS];
    cbe_build_pair_matching(Q_i, Q_j, matched_j_for_i,
                            /*alpha_of_beta_for_b=*/NULL,
                            /*free_slot_fired_count_inout=*/NULL);

    /* Pass 1 per-pair limiter — row-scalar phi with cone tightening
     * (Wave-CBE Commit 10, Fix #6). For each basis row m:
     *   1. Compute the existing per-(m, k) BJ phi from cbe_bj_phi_pair.
     *   2. Build g_row[k] = -psi_i * (grad_i[m][k] . dp), the predicted
     *      face delta per k for the same face offset the flux body uses.
     *   3. Row-scalar BJ phi = min_k phi_BJ[k] is the cone bracket upper
     *      bound; the cone bisection only ever tightens it.
     *   4. cbe_cone_phi_row bisects on the realizability predicate
     *      (mass > 0 AND central stress PSD per the Schur-complement
     *      convexity argument) over [0, phi_BJ_row]. Returns the largest
     *      known-realizable phi.
     *   5. The same scalar phi_row_pair is written to every k in row m.
     *      Mixing per-k phi after the cone validated a single-phi point
     *      would move Q_face off the line segment the predicate
     *      validated and could reintroduce non-realizability. */
    for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
        const int n = matched_j_for_i[m];
        double g_row[CBE_INTEGRATOR_NMOMENTS];
        double phi_BJ_row_pair = 1.0;
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            const double gdotdp = L.Gradients_CBE_basis_moments[m][k][0]*dp[0]
                                + L.Gradients_CBE_basis_moments[m][k][1]*dp[1]
                                + L.Gradients_CBE_basis_moments[m][k][2]*dp[2];
            g_row[k] = -psi_i * gdotdp;
            const double dQ        = Q_j[n][k] - Q_i[m][k];
            const double phi_ij_BJ = cbe_bj_phi_pair(dQ, g_row[k]);
            if(phi_ij_BJ < phi_BJ_row_pair) phi_BJ_row_pair = phi_ij_BJ;
        }
        const double phi_row_pair = cbe_cone_phi_row(Q_i[m], g_row, phi_BJ_row_pair);
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            if(phi_row_pair < accum.phi[m][k]) accum.phi[m][k] = phi_row_pair;
        }
    }
    (void)j;
}

/* ============================================================================
 * CBEGradSpec — runner-template Spec contract. Non-iterative; two passes
 * orchestrated by the toplevel via Aux::loop_iteration.
 * ========================================================================== */
struct CBEGradSpec {
    static constexpr const char *loop_name = "cbe_grad";

    /* Symmetric search topology because pass 1 (pairwise BJ-style limiter)
     * needs symmetric neighbor visibility to see every face the flux body
     * will subsequently reconstruct — exactly the role hydro GradientsSpec
     * plays for its slope-limiter / min-max collection. The raw LSQ pass
     * (pass 0) intentionally restricts accumulation to the i-kernel inside
     * the pair body (see cbe_grad_lsq_pair_kernel_body), matching the
     * density / NV_T / hydro raw-gradient semantics (linear consistency of
     * the LSQ estimator relies on the i-kernel normalisation). Per-pass
     * filtering is the right shape; do NOT "fix" pass 0 to symmetric. */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    /* Nominal default — DM, matching DMGradSpec convention. CBE runs on
     * AGS-eligible types alongside AGSForce; the real per-call mask is set
     * at the toplevel via args.neighbor_type_mask_override = bm
     * (ags_gravity_kernel_shared_BITFLAG). Conservative default avoids the
     * footgun of "missed override → all types become neighbors". */
    static constexpr unsigned int            neighbor_type_mask = (1u << 1);  /* DM */
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    static constexpr WritePattern   write_pattern             = WritePattern::ActiveReduceOnly;
    /* SidxCacheKind::None: the per-call DM/SIDM mask varies per bm — same
     * reasoning as DMGradSpec. */
    static constexpr SidxCacheKind  sidx_cache_kind           = SidxCacheKind::None;
    static constexpr bool           uses_ghost_writeback      = false;
    static constexpr bool           uses_ghost_write_detector = false;

    /* Pure-read pair kernel (no j-side writes) → tight oracle. */
    static constexpr double accum_tolerance = 1e-10;

    /* AGSForce_isactive is the SAME predicate the force consumer uses —
     * gradients refresh for exactly the particles whose force will read
     * them this step. Inactive particles retain their previous-step
     * P[i].Gradients_CBE_basis_moments (hydro semantics). */
    static bool is_active(int i) { return AGSForce_isactive(i) != 0; }

    using IterControl    = NotIterative;
    using CallScalars    = CBEGradCallScalars;
    using ActiveData     = CBEGradActiveState;
    using AccumData      = CBEGradOut;
    using DeviceContext  = NeighborLoopDeviceContextBase;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;

    /* NeighborData carries `j` and the P base pointer because the pair
     * body needs get_particle_volume_ags_P(j, P_base). */
    struct NeighborData {
        struct particle_data *neighbor_particle;
        int                   j;
        struct particle_data *P_base;
    };

    /* Aux carries the external gradient-pass index (mirror DMGradSpec).
     * populate_call_scalars copies it into CallScalars; pair_kernel +
     * apply_active_writeback branch on scalars.loop_iteration. */
    struct Aux {
        int loop_iteration;
    };

    static_assert(uses_ghost_writeback == false,
        "CBEGrad is pure i-side accumulation; no j-side writes.");

    /* Host hooks — bodies in sidm/cbe_integrator_gradients.cc. */
    static double      search_radius(const neighbor_loop_args& args,
                                      int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void        apply_active_writeback(const neighbor_loop_args& args,
                                              int active_slot, int i,
                                              const AccumData& accum);
    static void        merge_accum(AccumData& local, const AccumData& peer);
    static double      compare_accum(const AccumData& local, const AccumData& oracle);
    static void        set_oracle_brute_pass(DeviceContext& ctx, bool on);

    /* ---- Device hooks (header-inline; runner instantiates from GPU TUs). ---- */

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        for(int a = 0; a < 3; a++)
            for(int b = 0; b < 3; b++)
                accum.M[a][b] = 0.0;
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                for(int d = 0; d < 3; d++) accum.B[m][k][d] = 0.0;
                /* phi initialized to 1.0 for min-tracking; pass 0 never
                 * touches it so it stays neutral for that pass's merge. */
                accum.phi[m][k] = 1.0;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx,
                                   int /*active_slot*/, int i,
                                   double h_search,
                                   const CallScalars& scalars)
    {
        ActiveData a;
        a.pos[0]           = (double)dctx.P[i].Pos[0];
        a.pos[1]           = (double)dctx.P[i].Pos[1];
        a.pos[2]           = (double)dctx.P[i].Pos[2];
        a.h_search         = h_search;
        a.TimeBin          = dctx.P[i].TimeBin;
        a.scalars          = scalars;
        a.origin_local_idx = i;
        a.origin_rank      = -1;

        CBEGradLocalIn& L = a.local;
        L.Mass             = dctx.P[i].Mass;
        L.AGS_KernelRadius = (double)dctx.P[i].AGS_KernelRadius;
        L.Pos              = a.pos;
        L.Vel[0] = (double)dctx.P[i].Vel[0];
        L.Vel[1] = (double)dctx.P[i].Vel[1];
        L.Vel[2] = (double)dctx.P[i].Vel[2];
        L.V_i    = get_particle_volume_ags_P(i, dctx.P);
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                L.U_i[m][k] = dctx.P[i].CBE_basis_moments[m][k];

        /* Mode-B-safe by-value snapshot of the persistent gradient row
         * (design invariant I2). Pass-1 limiter reads this from
         * active.local; pass-0 LSQ does not consume it but populating
         * unconditionally is fixed-cost and keeps the load-active body
         * branch-free. */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                for(int d = 0; d < 3; d++)
                    L.Gradients_CBE_basis_moments[m][k][d] =
                        dctx.P[i].Gradients_CBE_basis_moments[m][k][d];
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx,
                                       int j,
                                       const IdentitySidecar& /*id*/,
                                       const ActiveData& /*active*/)
    {
        NeighborData n;
        n.neighbor_particle = &dctx.P[j];
        n.j                 = j;
        n.P_base            = dctx.P;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData&   active,
                             const NeighborData& neighbor,
                             AccumData&          accum,
                             NoScatter& /*scatter*/)
    {
        if(active.scalars.loop_iteration <= 0) {
            cbe_grad_lsq_pair_kernel_body(active, neighbor, accum);
        } else {
            cbe_grad_bj_pair_kernel_body (active, neighbor, accum);
        }
    }
};

/* Toplevel runner instantiation. Non-iterative form — paralleling DMGrad. */
extern template void run_neighbor_loop<CBEGradSpec>(const neighbor_loop_args&);

#endif /* CBE_INTEGRATOR_WITHGRADIENTS */
#endif /* CBE_INTEGRATOR_GRADIENTS_H */
