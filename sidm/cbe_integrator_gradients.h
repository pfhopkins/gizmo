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
 *   pass 1 (BJ-style): phi[m][k] (min-merge, NaN-aware); M, B stay 0
 * merge_accum unconditionally sums M+B AND mins phi — the unused-pass
 * fields are neutral elements (0 for sum, 1 for min) so the per-pass
 * branch isn't needed in merge_accum.
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

    /* Basis matching via SSOT helper (Wave-CBE Commit 6b). Same selector
     * dispatch as the flux body, so gradient/limiter and flux see the same
     * basis pairs by construction. Single-direction (a->b only); the
     * fired-count counter is NULL because pre-pass matching is not a
     * flux-pairing decision. */
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

    /* Basis matching via SSOT helper (Wave-CBE Commit 6b) — identical to
     * LSQ + flux by construction (same selector dispatch). */
    int matched_j_for_i[CBE_INTEGRATOR_NBASIS];
    cbe_build_pair_matching(Q_i, Q_j, matched_j_for_i,
                            /*alpha_of_beta_for_b=*/NULL,
                            /*free_slot_fired_count_inout=*/NULL);

    /* Per (m, k): predicted delta from Q_i to Q_face on i's side =
     *   grad_i[m][k] . face_offset_i, face_offset_i = -psi_i * dp.
     * → predicted = -psi_i * (grad_i[m][k] . dp). */
    for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
        const int n = matched_j_for_i[m];
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            const double gdotdp = L.Gradients_CBE_basis_moments[m][k][0]*dp[0]
                                + L.Gradients_CBE_basis_moments[m][k][1]*dp[1]
                                + L.Gradients_CBE_basis_moments[m][k][2]*dp[2];
            const double predicted = -psi_i * gdotdp;
            const double dQ        = Q_j[n][k] - Q_i[m][k];
            const double phi_ij    = cbe_bj_phi_pair(dQ, predicted);
            if(phi_ij < accum.phi[m][k]) accum.phi[m][k] = phi_ij;
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
