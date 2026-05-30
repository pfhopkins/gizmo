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
 * (for cbe_flux_hllc_vacuum and get_particle_volume_ags).
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

    double Face_Area_Vec[3];
    double Face_Area_Norm = 0;
    double vface_guess[3];
    for(int k=0; k<3; k++) {
        Face_Area_Vec[k] = -(kernel.wk_i * V_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2])
                           + kernel.wk_j * V_j * (P[j].NV_T[k][0]*kernel.dp[0] + P[j].NV_T[k][1]*kernel.dp[1] + P[j].NV_T[k][2]*kernel.dp[2])) * All.cf_atime * All.cf_atime;
        Face_Area_Norm += Face_Area_Vec[k] * Face_Area_Vec[k];
        vface_guess[k] = 0.5 * (local.Vel[k] + P[j].Vel[k]) / All.cf_atime;
    }
    Face_Area_Norm = sqrt(Face_Area_Norm);
    if(!(Face_Area_Norm > 0)) { return r; }
    double inv_FAN = 1.0 / Face_Area_Norm;
    double A_hat[3] = { Face_Area_Vec[0]*inv_FAN, Face_Area_Vec[1]*inv_FAN, Face_Area_Vec[2]*inv_FAN };

    /* Wave-CBE Commit 4 (2026-05-25): build "flux-frame" Q on both sides
     * via the SSOT helper so the cosmology / velocity-boost convention
     * matches whatever CBEGradSpec uses for its LSQ pass.
     * Stored basis moments are U-frame (cell-integrated, basis-frame
     * momentum); Q is per-volume comoving density with physical-frame
     * momentum baked in -- exactly what cbe_flux_hllc_vacuum expects. */
    double Q_i[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double Q_j[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    {
        double Vel_i_code[3] = { local.Vel[0], local.Vel[1], local.Vel[2] };
        double Vel_j_code[3] = { P[j].Vel[0],  P[j].Vel[1],  P[j].Vel[2]  };
        cbe_build_flux_frame_Q_from_stored_moments(
            local.CBE_basis_moments, Vel_i_code, V_i, All.cf_a3inv, All.cf_atime, Q_i);
        cbe_build_flux_frame_Q_from_stored_moments(
            P[j].CBE_basis_moments, Vel_j_code, V_j, All.cf_a3inv, All.cf_atime, Q_j);
    }

    /* Face reconstruction. With CBE_INTEGRATOR_WITHGRADIENTS on, the
     * persistent gradient row written by CBEGrad_gradient_calc is read for
     * each side and the MFM face state is built as
     *     Q_face_i[m][k] = Q_i[m][k] - psi_i * grad_Q_i[m][k] . kernel.dp
     *     Q_face_j[m][k] = Q_j[m][k] + psi_j * grad_Q_j[m][k] . kernel.dp
     * with psi_i = h_j / (h_i+h_j) = MFM FACE POSITION weight (NOT a flux
     * share). grad_i is read from local.Gradients_CBE_basis_moments
     * (by-value snapshot in AgsForceSpec::load_active, Mode-B-safe per
     * design invariant I2); grad_j is read directly from
     * P[j].Gradients_CBE_basis_moments (standard P[] ghost transport
     * carries the field onto ghost slots).
     *
     * WITHGRADIENTS off → fall through to Qface = Q (cell-centered),
     * byte-identical to the Phase-1 baseline. */
    double Qface_i[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
    double Qface_j[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    {
        const double psi_i_denom = kernel.h_i + kernel.h_j;
        long long local_nonfinite_count = 0;
        /* AGSForce's pair-overlap filter guarantees both h_i and h_j > 0
         * for any pair reaching this point — psi_i_denom > 0 is structural.
         * This is device code, so we cannot endrun; defense-in-depth is to
         * (a) fall back to first-order Qface = Q rather than NaN the face
         * state, and (b) bump local_nonfinite_count by NBASIS*NMOMENTS so
         * the event is observable when diagnostics are enabled (a clean
         * pair contributes 0; a denom-bad pair contributes the full row). */
        if(psi_i_denom > 0) {
            const double psi_i = kernel.h_j / psi_i_denom;
            const double psi_j = 1.0 - psi_i;
            const double dp[3] = { kernel.dp[0], kernel.dp[1], kernel.dp[2] };
            for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
                for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                    const double gi_dp = local.Gradients_CBE_basis_moments[m][k][0]*dp[0]
                                       + local.Gradients_CBE_basis_moments[m][k][1]*dp[1]
                                       + local.Gradients_CBE_basis_moments[m][k][2]*dp[2];
                    const double gj_dp = P[j].Gradients_CBE_basis_moments[m][k][0]*dp[0]
                                       + P[j].Gradients_CBE_basis_moments[m][k][1]*dp[1]
                                       + P[j].Gradients_CBE_basis_moments[m][k][2]*dp[2];
                    const bool   finite_i = isfinite(gi_dp);
                    const bool   finite_j = isfinite(gj_dp);
                    const double gi_safe  = finite_i ? gi_dp : 0.0;
                    const double gj_safe  = finite_j ? gj_dp : 0.0;
                    if(!finite_i || !finite_j) ++local_nonfinite_count;
                    Qface_i[m][k] = Q_i[m][k] - psi_i * gi_safe;
                    Qface_j[m][k] = Q_j[m][k] + psi_j * gj_safe;
                }
            }
        } else {
            /* Denom <= 0 — should be unreachable. Surface the event by
             * tallying a full row of non-finite contributions, then fall
             * back to first-order Qface so downstream code does not see
             * NaN. */
            local_nonfinite_count = (long long)CBE_INTEGRATOR_NBASIS
                                  * (long long)CBE_INTEGRATOR_NMOMENTS;
            for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
                for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                    Qface_i[m][k] = Q_i[m][k];
                    Qface_j[m][k] = Q_j[m][k];
                }
            }
        }
        /* Tally non-finite events into the diagnostic counter — gated on
         * the diagnostic block (same nesting as the field itself in
         * AgsForceOut). When the diagnostic block is off but WITHGRADIENTS
         * is on, the count is computed but discarded. */
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
        out.cbe_grad_nonfinite_count += local_nonfinite_count;
#else
        (void)local_nonfinite_count;
#endif
    }
#else
    /* Phase-1 fallback — Q_face = Q (cell-centered). Byte-identical
     * baseline when CBE_INTEGRATOR_WITHGRADIENTS is off. */
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
            Qface_i[m][k] = Q_i[m][k];
            Qface_j[m][k] = Q_j[m][k];
        }
    }
#endif
    /* Density clamp + counter (rho slot, all NMOMENTS). Wave-CBE Commit 5
     * (2026-05-26): face-state SPD repair on the stress block of each
     * rho-active basis row (NMOMENTS>=10 only) via cbe_spd_repair_S3x3
     * inside cbe_clamp_face_Q. Counters feed AgsForceOut.cbe_recon_rho_clamp_count
     * (col-5) and cbe_recon_S_clamp_count (col-6). */
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
    cbe_clamp_face_Q(Qface_i, &out.cbe_recon_rho_clamp_count, &out.cbe_recon_S_clamp_count);
    cbe_clamp_face_Q(Qface_j, &out.cbe_recon_rho_clamp_count, &out.cbe_recon_S_clamp_count);
#else
    cbe_clamp_face_Q(Qface_i, (long long*)0, (long long*)0);
    cbe_clamp_face_Q(Qface_j, (long long*)0, (long long*)0);
#endif

    /* Guarded per-basis face-normal velocities + K (= clamped face density).
     * The helper returns K=0, v_n=0 for any basis clamped inactive, so the
     * residual / cost-matrix / flux loop all naturally skip those bases. */
    double v_alpha_n_i[CBE_INTEGRATOR_NBASIS], v_alpha_n_j[CBE_INTEGRATOR_NBASIS];
    double K_i[CBE_INTEGRATOR_NBASIS], K_j[CBE_INTEGRATOR_NBASIS];
    cbe_face_K_and_vn_from_Q(Qface_i, A_hat, K_i, v_alpha_n_i);
    cbe_face_K_and_vn_from_Q(Qface_j, A_hat, K_j, v_alpha_n_j);

    /* Dispersion-based bracket pad (NMOMENTS>4 only; fence guarantees the
     * 3D [4]/[5]/[6]=diag layout). 1D dispersion = sqrt(trace(S)/rho); use
     * the face-state Qface to be consistent with the K, v_n above. */
    double pad = 0;
#if (CBE_INTEGRATOR_NMOMENTS > 4)
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(Qface_i[m][0] > MIN_REAL_NUMBER) {
            double trS_i = (Qface_i[m][4] + Qface_i[m][5] + Qface_i[m][6]) / Qface_i[m][0];
            double sig_i = sqrt(DMAX(trS_i, 0.0));
            if(sig_i > pad) pad = sig_i;
        }
        if(Qface_j[m][0] > MIN_REAL_NUMBER) {
            double trS_j = (Qface_j[m][4] + Qface_j[m][5] + Qface_j[m][6]) / Qface_j[m][0];
            double sig_j = sqrt(DMAX(trS_j, 0.0));
            if(sig_j > pad) pad = sig_j;
        }
    }
#endif
    /* NMOMENTS=4 (or NMOMENTS>4 with all-zero S) pad floor: small fraction
     * of the velocity spread so the bracket has nonzero width even when
     * basis velocities coincide. bracket-widen-4x handles degenerate cases. */
    {
        double v_spread = 0;
        for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
            v_spread = DMAX(v_spread, fabs(v_alpha_n_i[m]));
            v_spread = DMAX(v_spread, fabs(v_alpha_n_j[m]));
        }
        double floor_pad = 1.0e-8 * v_spread + MIN_REAL_NUMBER;
        if(pad < floor_pad) pad = floor_pad;
    }

    /* Bracket from min/max basis normal velocities (both sides, ACTIVE
     * bases only -- K==0 rows have v_n=0 by construction and would
     * artificially squeeze the bracket toward 0). Padded by the dispersion
     * scale above. */
    double v_F_lo = MAX_REAL_NUMBER, v_F_hi = -MAX_REAL_NUMBER;
    int any_active = 0;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(K_i[m] > 0) {
            if(v_alpha_n_i[m] < v_F_lo) v_F_lo = v_alpha_n_i[m];
            if(v_alpha_n_i[m] > v_F_hi) v_F_hi = v_alpha_n_i[m];
            any_active = 1;
        }
        if(K_j[m] > 0) {
            if(v_alpha_n_j[m] < v_F_lo) v_F_lo = v_alpha_n_j[m];
            if(v_alpha_n_j[m] > v_F_hi) v_F_hi = v_alpha_n_j[m];
            any_active = 1;
        }
    }
    /* All-clamped degenerate case: nothing to flux through this face. Skip
     * cleanly (no spurious bracket_fail, no flux, no NaN). */
    if(!any_active) { return r; }
    v_F_lo -= pad; v_F_hi += pad;

    /* Bulk-weighted normal velocity as a root-find fallback (used by
     * cbe_face_solve_v_F_normal only when bisection fails to bracket).
     * Theta active set at vface_guess can be empty; in that case use
     * vface_guess as the fallback carrier. Tangential bulk components
     * are NOT needed for cbe_flux_hllc_vacuum (Fix #1, Commit 8 retired
     * the old SM-dispersion term that consumed them); the final flux
     * function only reads vface . n_hat, which equals v_F_normal under
     * the vface = v_F_normal * A_hat construction below. The full bulk
     * vector is retained here as raw material for Fix #7's analytic
     * fallback rewrite. */
    double v_F_guess = vface_guess[0]*A_hat[0] + vface_guess[1]*A_hat[1] + vface_guess[2]*A_hat[2];
    double vface_bulk[3] = {0};
    double v_wt_sum = 0;
    double theta_i[CBE_INTEGRATOR_NBASIS] = {0};
    double theta_j[CBE_INTEGRATOR_NBASIS] = {0};
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(K_i[m] > 0 && v_alpha_n_i[m] - v_F_guess > 0) {
            double w = K_i[m]; v_wt_sum += w;
            double inv_Q0 = 1.0 / Qface_i[m][0];
            for(int k=0; k<3; k++) vface_bulk[k] += w * Qface_i[m][k+1] * inv_Q0;
        }
        if(K_j[m] > 0 && v_alpha_n_j[m] - v_F_guess < 0) {
            double w = K_j[m]; v_wt_sum += w;
            double inv_Q0 = 1.0 / Qface_j[m][0];
            for(int k=0; k<3; k++) vface_bulk[k] += w * Qface_j[m][k+1] * inv_Q0;
        }
    }
    double vbulk_dot_Ahat;
    if((v_wt_sum > MIN_REAL_NUMBER) && (v_wt_sum < MAX_REAL_NUMBER)) {
        double inv_wt = 1.0 / v_wt_sum;
        vbulk_dot_Ahat = (vface_bulk[0]*A_hat[0] + vface_bulk[1]*A_hat[1] + vface_bulk[2]*A_hat[2]) * inv_wt;
    } else {
        vbulk_dot_Ahat = v_F_guess;
    }

    /* Root-find face-normal v_F. K=0 rows contribute 0 to the residual so
     * the root location depends only on the active-basis set. */
    int bracket_ok = 0;
    double v_F_normal = cbe_face_solve_v_F_normal(
        v_alpha_n_i, v_alpha_n_j, K_i, K_j,
        v_F_lo, v_F_hi, vbulk_dot_Ahat, &bracket_ok);

    /* Face velocity for the flux call: pure normal component along the
     * canonical face unit normal A_hat. cbe_flux_hllc_vacuum reads only
     * vface . n_out (n_out = +A_hat for i-side, -A_hat for j-side), so the
     * tangential component cancels exactly; passing it as zero is clean. */
    const double vface[3] = { v_F_normal * A_hat[0],
                              v_F_normal * A_hat[1],
                              v_F_normal * A_hat[2] };

    /* Final theta gates use the converged v_F_normal. K==0 rows are forced
     * inactive (theta=0) regardless of v_alpha_n. */
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        if(K_i[m] > 0 && v_alpha_n_i[m] - v_F_normal > 0) theta_i[m] = 1;
        if(K_j[m] > 0 && v_alpha_n_j[m] - v_F_normal < 0) theta_j[m] = 1;
    }

    /* Basis-pair matching via SSOT helper (Wave-CBE Commit 6b). C6c flips
     * the selectors to CBE_COST_TRACE_W2 + USE_FREE_SLOT=1 per harness
     * §4.4 (trace-W2 cost + adaptive median-tau free-slot fallback). The
     * free-slot fire-count is accumulated per face into a local int (bound:
     * 2*NBASIS<=16 per face) gated by the diagnostic compile flag, then
     * folded into out.cbe_pairing_free_slot_count for the standard
     * merge/compare/writeback channel that feeds cbe_diagnostics.txt. */
    int matching_basis_j_for_basis_in_i[CBE_INTEGRATOR_NBASIS];
    int matching_basis_i_for_basis_in_j[CBE_INTEGRATOR_NBASIS];
    double vsig = 0;
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
    int free_slot_fired_this_face = 0;
    cbe_build_pair_matching(Qface_i, Qface_j,
                            matching_basis_j_for_basis_in_i,
                            matching_basis_i_for_basis_in_j,
                            &free_slot_fired_this_face);
    out.cbe_pairing_free_slot_count += (long long)free_slot_fired_this_face;
#else
    cbe_build_pair_matching(Qface_i, Qface_j,
                            matching_basis_j_for_basis_in_i,
                            matching_basis_i_for_basis_in_j,
                            /*free_slot_fired_count_inout=*/NULL);
#endif

    /* Flux loop. Wave-CBE Commit 8 (Fix #1, 2026-05-30) — sign convention
     * is now explicit at the call site, not via a wt_prefac trick:
     *   i-side call passes Area_outward = +Face_Area_Vec (i is upwind on
     *     the +A_hat side; F is outflow from i; i LOSES it -> "-= flux").
     *   j-side call passes Area_outward = -Face_Area_Vec (j is upwind on
     *     the -A_hat side; F is outflow from j into i's basis i_m; i
     *     GAINS it -> "+= flux").
     * The HLLC vacuum flux solver branches on the SOURCE-side outward
     * normal velocity u_out; passing the correct Area_outward per side is
     * load-bearing for correctness (silently mis-orienting Area_outward
     * inverts the physics). */
    const double Area_i_out[3] = {  Face_Area_Vec[0],  Face_Area_Vec[1],  Face_Area_Vec[2] };
    const double Area_j_out[3] = { -Face_Area_Vec[0], -Face_Area_Vec[1], -Face_Area_Vec[2] };
    /* matching_basis_j_for_basis_in_i[] is populated by cbe_build_pair_matching
     * for symmetric SSOT but is not consumed locally — matched-pair coupling
     * for the i-side deposit happens via Q_face reconstruction (Fix #5,
     * future commit) reading the matched j-basis through the gradient
     * stencil, NOT through a per-pair flux-call neighbor argument. The
     * j-side outflow path needs i_m to deposit gain into i's basis i_m. */
    (void)matching_basis_j_for_basis_in_i;
    for(int m=0; m<CBE_INTEGRATOR_NBASIS; m++) {
        const int i_m = matching_basis_i_for_basis_in_j[m];
        double flux[CBE_INTEGRATOR_NMOMENTS] = {0};
        double vsig_i = 0, vsig_j = 0;
        if(theta_i[m] == 1) {
            vsig_i = cbe_flux_hllc_vacuum(Qface_i[m], vface, Area_i_out, flux);
            for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
                out.CBE_basis_moments_dt[m][k] -= flux[k];
            }
        }
        if(theta_j[m] == 1) {
            vsig_j = cbe_flux_hllc_vacuum(Qface_j[m], vface, Area_j_out, flux);
            for(int k=0; k<CBE_INTEGRATOR_NMOMENTS; k++) {
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
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
    /* Wave-CBE Commit 3 diagnostic: residual at the converged v_F_normal,
     * converted to dM/dt units by multiplying by Face_Area_Norm. */
    {
        double R_final = cbe_face_mass_residual_per_unit_area(
            v_F_normal, v_alpha_n_i, v_alpha_n_j, K_i, K_j);
        double abs_R_full = fabs(R_final) * Face_Area_Norm;
        if(abs_R_full > out.cbe_face_residual_max) out.cbe_face_residual_max = abs_R_full;
        out.cbe_face_residual_sum += abs_R_full;
        if(!bracket_ok) out.cbe_bracket_fail_count += 1;
    }
#endif
#else
    (void)local; (void)j; (void)P; (void)kernel; (void)out;
#endif /* CBE_INTEGRATOR */
    return r;
}

#endif /* CBE_INTEGRATOR_FLUX_FUNCTIONS_H */
