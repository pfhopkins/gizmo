#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "cbe_integrator_functions.h"   /* cbe_basis_p_r / _w / _a NUMDIMS-aware
                                          momentum-slot helpers used by
                                          cbe_synthesize_cold_default. The
                                          helpers expand to plain inline in
                                          host TUs via the KOKKOS_INLINE_FUNCTION
                                          fallback at the top of the header. */


/*! \file cbe_integrator.cc
 *  \brief startup initialization for the CBE integrator basis moments.
 *         The per-step drift-kick (do_cbe_drift_kick_kernel) and the per-
 *         active post-gravity finalization (do_cbe_postgravity_kernel) now
 *         live in cbe_integrator_functions.h as KOKKOS_INLINE_FUNCTION
 *         kernels, dispatched via sidm/cbe_integrator_gpu.cc.
 *
 *         Per-pair CBE flux (cbe_integrator_flux_compute_pair) is invoked
 *         from the AGSForce iterative neighbor loop (gravity/ags_force_loop.h).
 *
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef CBE_INTEGRATOR


// moment ordering convention: 0, x, y, z, xx, yy, zz, xy, xz, yz
//                             0, 1, 2, 3,  4,  5,  6,  7,  8,  9


/* C7 (2026-05-30) helpers: deterministic Fibonacci-sphere direction
 * sampler + Rodrigues rotation. Used by cbe_synthesize_cold_default()
 * to lay down (NBASIS-1) placeholder-stream directions uniformly on
 * the velocity sphere, rotated so the canonical +z axis aligns with
 * the particle's velocity direction. NO RNG -- the rank-dependent
 * RNG-seeded init this replaces was non-reproducible across MPI
 * decompositions (same particle ID got different basis init under
 * different rank counts; real bug). Fibonacci is rank-independent. */
static void cbe_fibonacci_sphere_dir_canonical(int n, int N, double out[3])
{
    /* Nth of N points on the unit sphere, canonical frame (+z = polar).
     * Standard golden-angle spiral; deterministic. Degenerate N==1 case
     * picks the equator point (any direction is fine for the single
     * placeholder slot at NBASIS==2). */
    const double golden = (1.0 + sqrt(5.0)) / 2.0;
    double y = (N <= 1) ? 0.0 : 1.0 - 2.0 * ((double)n + 0.5) / (double)N;
    double r = sqrt(DMAX(0.0, 1.0 - y*y));
    double theta = 2.0 * M_PI * (double)n / golden;
    out[0] = r * cos(theta);
    out[1] = r * sin(theta);
    out[2] = y;
}

static void cbe_fibonacci_sphere_dir_rotated_to_vhat(int n, int N,
                                                       const double Vel[3], double v0,
                                                       double out[3])
{
    /* Sample direction n of N in canonical frame, then rotate so the
     * canonical +z axis maps to vhat = Vel/|Vel|. Rodrigues' formula.
     * Zero-velocity fallback: stay in canonical frame; harmless because
     * the placeholder mass is tiny (epsilon * P.Mass / (N-1)). */
    cbe_fibonacci_sphere_dir_canonical(n, N, out);
    if(v0 < 1.0e-30) { return; }
    double vhat[3] = {Vel[0]/v0, Vel[1]/v0, Vel[2]/v0};
    double dot = vhat[2];   /* zhat . vhat */
    if(fabs(dot) > 1.0 - 1.0e-12) {
        /* C7 fixup (2026-05-30, codex): antiparallel case (vhat ~ -zhat).
         * Apply a proper 180-deg rotation about the x-axis -- maps +z to
         * -z, leaves +x unchanged. Previous draft negated all three
         * components (an inversion, not a rotation); harmless for the
         * tiny placeholder slot but inconsistent with the Rodrigues
         * intent. */
        if(dot < 0) { out[1] = -out[1]; out[2] = -out[2]; }
        return;
    }
    /* axis = zhat x vhat (axis[2] = 0 always); angle = acos(dot). */
    double axis[3] = {-vhat[1], vhat[0], 0.0};
    double a_norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1]);
    axis[0] /= a_norm; axis[1] /= a_norm;
    double cosA = dot;
    double sinA = sqrt(DMAX(0.0, 1.0 - cosA*cosA));
    double cross_x = axis[1]*out[2] - axis[2]*out[1];
    double cross_y = axis[2]*out[0] - axis[0]*out[2];
    double cross_z = axis[0]*out[1] - axis[1]*out[0];
    double adot = axis[0]*out[0] + axis[1]*out[1];   /* axis[2]==0 */
    double tmp[3] = {
        cosA*out[0] + sinA*cross_x + (1.0-cosA)*adot*axis[0],
        cosA*out[1] + sinA*cross_y + (1.0-cosA)*adot*axis[1],
        cosA*out[2] + sinA*cross_z
    };
    out[0] = tmp[0]; out[1] = tmp[1]; out[2] = tmp[2];
}


/* Synthesize cold-DM default for one particle when the VlasovMoments
 * dataset was absent for its PartType in the IC.
 *
 * RELATIVE-FRAME storage convention: basis moments are stored relative to
 * P.Vel (the mesh-generating-point velocity). Hence the synthesized values
 * are about a per-basis relative velocity v_rel[α] = v_phys[α] − P.Vel:
 *   Slot 0 (dominant):  v_rel = 0  ⇒  basis_p_stored[0] = 0,
 *                                     basis_T_stored[0][k][k] = m_0 * S_floor
 *                                     (off-diag = 0; no v_rel⊗v_rel piece).
 *   Slots 1..N-1 (placeholders): v_rel[α] = dv * dir[α-1]
 *                                with dir a deterministic Fibonacci-sphere
 *                                direction (rotated to v_hat-aligned frame
 *                                if v_hat is meaningful; falls back to a
 *                                fixed basis when |P.Vel|=0). basis_p_stored
 *                                = m_each * v_rel.
 *
 * Slot 0 carries (1-eps) of P[i].Mass; slots 1..N-1 share eps equally.
 * The subsequent do_cbe_initialization() closure enforces Σ basis_p_stored = 0.
 * Units inherit the existing VlasovMoments snapshot-write convention. */
static void cbe_synthesize_cold_default(int i)
{
    const double eps = 1.0e-6;            /* placeholder-stream mass fraction */
    const double S_floor_rel = 1.0e-12;   /* tiny SPD floor relative to v_scale^2 */
    const double dv_rel = 1.0e-3;         /* placeholder v_rel offset relative to v_scale */

    /* v_scale is set from |P.Vel| if nonzero (sets the natural scale for the
     * S floor and the placeholder relative-velocity offset); fallback floor
     * for zero-bulk cells so the cold-default is still SPD. */
    double Vel[3] = {P[i].Vel[0], P[i].Vel[1], P[i].Vel[2]};
    double v2     = Vel[0]*Vel[0] + Vel[1]*Vel[1] + Vel[2]*Vel[2];
    double v0     = (v2 > 0.0) ? sqrt(v2) : 0.0;
    double v_scale = (v0 > 1.0e-30) ? v0 : 1.0e-10;
    double S_floor = S_floor_rel * v_scale * v_scale;

    /* Slot 0 — dominant cold stream at v_rel = 0 (zero relative momentum). */
    {
        double m0 = (1.0 - eps) * P[i].Mass;
        P[i].CBE_basis_moments[0][0] = m0;
        for(int k = 0; k < NUMDIMS; k++) {
            cbe_basis_p_w(P[i].CBE_basis_moments[0], k, 0.0);
        }
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
        /* T_rel_ab = m * (v_rel_a*v_rel_b + S_floor*delta_ab); v_rel = 0 for
         * the dominant cold stream so T_rel_aa = m * S_floor and all
         * off-diagonals are zero. Commit 4a: dim-agnostic via cbe_basis_T_w
         * (was: explicit 3D triangular index block under the SECONDMOMENT
         * fence). */
        for(int a = 0; a < NUMDIMS; a++) {
            cbe_basis_T_w(P[i].CBE_basis_moments[0], a, a, m0 * S_floor);
            for(int b = a + 1; b < NUMDIMS; b++) {
                cbe_basis_T_w(P[i].CBE_basis_moments[0], a, b, 0.0);
            }
        }
#endif
    }

    /* Slots 1..NBASIS-1 — placeholder streams at small v_rel = dv*dir. */
    if(CBE_INTEGRATOR_NBASIS > 1) {
        int N_placeholder = CBE_INTEGRATOR_NBASIS - 1;
        double m_each = eps * P[i].Mass / (double)N_placeholder;
        double dv = dv_rel * v_scale;
        for(int j = 1; j < CBE_INTEGRATOR_NBASIS; j++) {
            double dir[3];
            cbe_fibonacci_sphere_dir_rotated_to_vhat(j-1, N_placeholder, Vel, v0, dir);
            /* RELATIVE-FRAME placeholder velocity: v_rel = dv * dir (NOT
             * absolute Vel + dv*dir). */
            double v_rel[3] = { dv*dir[0], dv*dir[1], dv*dir[2] };
            P[i].CBE_basis_moments[j][0] = m_each;
            for(int k = 0; k < NUMDIMS; k++) {
                cbe_basis_p_w(P[i].CBE_basis_moments[j], k, m_each * v_rel[k]);
            }
#if defined(CBE_INTEGRATOR_SECONDMOMENT)
            /* T_rel_ab = m * (v_rel_a*v_rel_b + S_floor*delta_ab). Commit 4a:
             * dim-agnostic via cbe_basis_T_w over active (a<=b)<NUMDIMS. */
            for(int a = 0; a < NUMDIMS; a++) {
                cbe_basis_T_w(P[i].CBE_basis_moments[j], a, a,
                              m_each * (v_rel[a]*v_rel[a] + S_floor));
                for(int b = a + 1; b < NUMDIMS; b++) {
                    cbe_basis_T_w(P[i].CBE_basis_moments[j], a, b,
                                  m_each * v_rel[a]*v_rel[b]);
                }
            }
#endif
        }
    }
}


/* variable initialization (called once from core/init.cc:844, AFTER
 * read_ic at line 84/87 has populated P[].CBE_basis_moments for any
 * PartTypes whose IC carried the VlasovMoments block; see
 * declarations/allvars.h for CBE_Moments_LoadedFromIC_PType contract). */
void do_cbe_initialization(void)
{
    int i, j, k;
    for(i = 0; i < NumPart; i++)
    {
        /* dt buffer + (under WITHGRADIENTS) gradients: ALWAYS zero,
         * regardless of IC source. Gradients are derived numerics and
         * are NEVER read from IC; CBEGrad_gradient_calc() refreshes
         * them before each force pass. */
        for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                P[i].CBE_basis_moments_dt[j][k] = 0;
            }
        }
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
        for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                for(int d = 0; d < 3; d++) {
                    P[i].Gradients_CBE_basis_moments[j][k][d] = 0;
                }
            }
        }
#endif

        if(CBE_Moments_LoadedFromIC_PType[P[i].Type]) {
            /* IC carried VlasovMoments for this PartType; trust it.
             * Validate finiteness + non-zero mass sum before the
             * conservation pass touches it -- malformed loaded data
             * fails LOUD, not silently synthesized over (would hide
             * a real IC bug; codex 2026-05-30). Per the C6-established
             * migration rule: corruption-class stays on endrun. */
            double mom_tot_check = 0;
            int has_nonfinite = 0;
            for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                    if(!isfinite(P[i].CBE_basis_moments[j][k])) { has_nonfinite = 1; }
                }
                mom_tot_check += P[i].CBE_basis_moments[j][0];
            }
            int mass_bad = (P[i].Mass > 0) &&
                           (!isfinite(mom_tot_check) || mom_tot_check <= 0);
            if(has_nonfinite || mass_bad) {
                printf("CBE: malformed loaded VlasovMoments for particle "
                       "ID=%llu type=%d: nonfinite_slot=%d sum_basis_m0=%g "
                       "P.Mass=%g. Aborting rather than silently "
                       "synthesizing or renormalizing corrupt IC.\n",
                       (unsigned long long) P[i].ID, (int) P[i].Type,
                       has_nonfinite, mom_tot_check, P[i].Mass);
                fflush(stdout);
                endrun(8889);
            }
        } else {
            /* Absent: synthesize cold-DM default. */
            cbe_synthesize_cold_default(i);
        }

        /* Commit 4b (2026-06-03): the prior dim-aware "moment zeroing"
         * blocks for NUMDIMS==1/2 hardcoded the OLD 3D-only slot indices
         * (keep k=0,1,4 in 1D; zero k=3,6,8,9 in 2D) which silently
         * stomped on legitimate stress data once the 4b unfence made
         * SECONDMOMENT reachable in low-D (1D T_xx lives at slot 2,
         * 2D T_xx at slot 3, T_xy at slot 5 per cbe_T_idx). The blocks
         * are also fully redundant now that CBE_INTEGRATOR_NMOMENTS is
         * dim-aware (precompiler_logic.h:142-154): the array literally
         * has exactly the active slots in every (NUMDIMS, SECONDMOMENT)
         * combination, so there is nothing to "trim." Removed. */

        /* Conservation renormalization -- existing two-pass pattern,
         * applies to BOTH loaded and synthesized paths. For an exactly-
         * conservative synthesized basis (which the default helper
         * produces by construction) this is a near-no-op. For loaded
         * moments slightly off due to writer float-precision, a gentle
         * safety net. */
        double mom_tot[CBE_INTEGRATOR_NMOMENTS] = {0};
        for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                mom_tot[k] += P[i].CBE_basis_moments[j][k];
            }
        }
        if(mom_tot[0] > 0) {
            for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                    P[i].CBE_basis_moments[j][k] *= P[i].Mass / mom_tot[0];
                }
            }
        }
        for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) { mom_tot[k] = 0; }
        for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
            for(k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                mom_tot[k] += P[i].CBE_basis_moments[j][k];
            }
        }
        if(P[i].Mass > 0) {
            /* Momentum closure (RELATIVE-FRAME storage convention): enforce
             *     Σ_α basis_p_stored[α][k] = 0  for k = 1..NUMDIMS,
             * i.e. the per-basis momenta are stored relative to P.Vel so
             *     basis_p_stored[α] = m_α * (v_phys[α] − P.Vel)
             * and the absolute-frame bulk momentum is carried entirely by
             * P.Mass * P.Vel. The closure subtracts the residual mass-
             * weighted bulk drift mom_tot[k]/M from each basis row so that
             * the sum-over-bases is exactly zero. This is a per-cell no-op
             * if the IC writer / synthesis already produced relative-frame
             * storage; a safety net otherwise.
             *
             * Slots 1..NUMDIMS only — slot 0 is mass; higher slots are
             * stress / SECONDMOMENT (the second-moment relative-to-
             * absolute boost is handled by the flux-frame helper, NOT
             * here). */
            for(j = 0; j < CBE_INTEGRATOR_NBASIS; j++) {
                for(k = 1; k <= NUMDIMS; k++) {
                    P[i].CBE_basis_moments[j][k] -=
                        P[i].CBE_basis_moments[j][0] *
                        (mom_tot[k] / P[i].Mass);
                }
            }
        }
    }
    return;
}


/* ---------------------------------------------------------------------------
 * Per-output-interval CBE diagnostic counter scaffold (Wave-CBE Commit 2,
 * 2026-05-24). Gated by CBE_INTEGRATOR_OUTPUT_MOREINFO (or the broader
 * OUTPUT_ADDITIONAL_RUNINFO) so production runs can purge the entire
 * diagnostic subsystem by disabling the flag.
 *
 * Host-side per-rank counters accumulated by future Wave-CBE commits
 * (3 = root-found v_F, 4 = gradient/reconstruction, 5 = SPD repair), then
 * MPI-reduced and emitted to FdCbeDiagnostics (cbe_diagnostics.txt; opened
 * in core/begrun.cc under the same gate). Reset at each emit so the values
 * represent the accumulated total since the previous output.
 *
 * Aggregation path note: per-pair updates from inside AGSForce/GPU
 * kernels must go through the existing AgsForceOut accumulator / merge /
 * writeback pattern, not via direct writes to these globals. The pair-loop
 * infrastructure runs reductions onto per-particle out-structs which get
 * merged onto host particles in the post-loop step; the CBE counter
 * updates will hook there. This commit only defines the host-side
 * holding/reduce/emit scaffold; counters stay at zero until populated.
 *
 * No call site is added in this commit (Commit 3 will add the hook from
 * energy_statistics or equivalent once the counter writes are real). The
 * file is opened (with a column-header comment) but stays header-only.
 * --------------------------------------------------------------------------- */
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
struct cbe_step_accumulators {
    /* Populated by Wave-CBE Commit 3 (root-found v_F). NOTE: aggregated
     * over per-pair evaluations (AGSForce visits an i-j pair once when i
     * is active; geometric faces with both endpoints active contribute
     * twice). residual_max is the worst single evaluation; residual_sum
     * is the sum across all pair evaluations on this rank in the
     * interval. */
    double face_mass_flux_residual_max;   /* max |sum_basis F_m * A| over pair evaluations */
    double face_mass_flux_residual_sum;   /* sum |sum_basis F_m * A| over pair evaluations */
    long long bracket_fail_count;         /* root-find bracket-widening failures */
    /* Populated by Wave-CBE Commit 4 (gradient/reconstruction): */
    long long recon_rho_clamp_count;      /* Q-face rho < eps clamps */
    long long recon_S_clamp_count;        /* Q-face Sxx < 0 clamps */
    /* Populated by Wave-CBE Commit 5 (SPD repair): */
    double repair_dP_sum;                 /* sum |dP| introduced by repair */
    double repair_dT_sum;                 /* sum |dT| introduced by repair */
    /* Populated by Wave-CBE Commit 6c (pairing free-slot fallback): SUM
     * over directional basis rows for which the cbe_apply_free_slot_
     * fallback transform fired during flux pairing. Independent of
     * WITHGRADIENTS — emitted as the appended last column of
     * cbe_diagnostics.txt (col-9 in WITHGRADIENTS-off 9-col format;
     * col-10 in WITHGRADIENTS-on 10-col format). 0 in builds with
     * CBE_PAIRING_USE_FREE_SLOT=0 (compile-time selector). */
    long long pairing_free_slot_count;
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    /* Populated by Wave-CBE Commit 4 Phase 2 #5 (face reconstruction):
     * counts non-finite grad·dp events observed per pair in the flux body.
     * Defense-in-depth; Tikhonov regularization in the LSQ pass should
     * keep grads finite under normal physics. Emitted as col-9 of
     * cbe_diagnostics.txt only when WITHGRADIENTS is on; WITHGRADIENTS-off
     * builds emit the original 8-col Phase-1 format. */
    long long grad_nonfinite_count;
#endif
};
static struct cbe_step_accumulators CbeStepAccum;


void cbe_step_diagnostics_reset(void)
{
    CbeStepAccum.face_mass_flux_residual_max = 0;
    CbeStepAccum.face_mass_flux_residual_sum = 0;
    CbeStepAccum.bracket_fail_count          = 0;
    CbeStepAccum.recon_rho_clamp_count       = 0;
    CbeStepAccum.recon_S_clamp_count         = 0;
    CbeStepAccum.repair_dP_sum               = 0;
    CbeStepAccum.repair_dT_sum               = 0;
    CbeStepAccum.pairing_free_slot_count     = 0;
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    CbeStepAccum.grad_nonfinite_count        = 0;
#endif
}


/* Wave-CBE Commit 3 observer: called once per active particle from
 * AgsForceSpec::apply_active_writeback with that particle's merged
 * AgsForceOut.cbe_face_residual_* and cbe_bracket_fail_count. Per-rank
 * step-local aggregation; MPI reduction happens in cbe_step_diagnostics_emit. */
void cbe_step_diagnostics_observe_face(double face_residual_max,
                                        double face_residual_sum,
                                        long long bracket_fail_count)
{
    if(face_residual_max > CbeStepAccum.face_mass_flux_residual_max) {
        CbeStepAccum.face_mass_flux_residual_max = face_residual_max;
    }
    CbeStepAccum.face_mass_flux_residual_sum += face_residual_sum;
    CbeStepAccum.bracket_fail_count          += bracket_fail_count;
}


/* Wave-CBE Commit 4 + Commit 5 observer: per-active particle's merged
 * AgsForceOut.cbe_recon_rho_clamp_count + cbe_recon_S_clamp_count from
 * cbe_clamp_face_Q in the flux body. rho-clamp populated by Commit 4
 * Phase 1; S-clamp populated by Commit 5 face-state SPD repair. */
void cbe_step_diagnostics_observe_recon(long long rho_clamp_count,
                                         long long S_clamp_count)
{
    CbeStepAccum.recon_rho_clamp_count += rho_clamp_count;
    CbeStepAccum.recon_S_clamp_count   += S_clamp_count;
}


#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
/* Wave-CBE Commit 4 Phase 2 #5 observer: per-active particle's merged
 * AgsForceOut.cbe_grad_nonfinite_count from the flux body's per-pair tally
 * of non-finite grad·dp events. */
void cbe_step_diagnostics_observe_grad_nonfinite(long long count)
{
    CbeStepAccum.grad_nonfinite_count += count;
}
#endif


/* Wave-CBE Commit 5 observer (2026-05-26): cell-state SPD-repair drift
 * accumulators from do_cbe_drift_kick_kernel (Site A). Called once per
 * cbe_drift_kick_evaluate_gpu invocation with the host-summed dT_scratch
 * (deterministic sum order; no device atomics).
 *
 * dP is always 0.0 from Site A (SPD repair touches only stress slots
 * [4..9], not momentum [1..3]). Kept as a parameter so any future repair
 * site that touches momentum can plumb in without an API change. Phil's
 * directive: dP=0 is a useful invariant for the diagnostic. */
void cbe_step_diagnostics_observe_repair(double dP, double dT)
{
    CbeStepAccum.repair_dP_sum += dP;
    CbeStepAccum.repair_dT_sum += dT;
}


/* Wave-CBE Commit 6c observer: per-active particle's merged
 * AgsForceOut.cbe_pairing_free_slot_count from the flux body's per-pair
 * tally of free-slot fallback firings. Independent of WITHGRADIENTS;
 * appended as the last column of cbe_diagnostics.txt. */
void cbe_step_diagnostics_observe_pairing_free_slot(long long count)
{
    CbeStepAccum.pairing_free_slot_count += count;
}


void cbe_step_diagnostics_emit(void)
{
    double rmax_local = CbeStepAccum.face_mass_flux_residual_max;
    double rsum_local = CbeStepAccum.face_mass_flux_residual_sum;
    long long brk_local = CbeStepAccum.bracket_fail_count;
    long long rc_local  = CbeStepAccum.recon_rho_clamp_count;
    long long sc_local  = CbeStepAccum.recon_S_clamp_count;
    double dP_local = CbeStepAccum.repair_dP_sum;
    double dT_local = CbeStepAccum.repair_dT_sum;
    double rmax, rsum, dP, dT;
    long long brk, rc, sc;
    MPI_Reduce(&rmax_local, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rsum_local, &rsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&brk_local,  &brk,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rc_local,   &rc,   1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sc_local,   &sc,   1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dP_local,   &dP,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dT_local,   &dT,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    long long fs_local = CbeStepAccum.pairing_free_slot_count, fs;
    MPI_Reduce(&fs_local, &fs, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
    long long gnf_local = CbeStepAccum.grad_nonfinite_count, gnf;
    MPI_Reduce(&gnf_local, &gnf, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    if(ThisTask == 0 && FdCbeDiagnostics != NULL) {
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
        /* 10-col extended format under WITHGRADIENTS. col-9 stays as
         * cbe_grad_nonfinite_count (Commit 4 Phase 2 #5); col-10 is the
         * new cbe_pairing_free_slot_count (Commit 6c). */
        fprintf(FdCbeDiagnostics, "%.16g %.16g %.16g %lld %lld %lld %.16g %.16g %lld %lld\n",
                All.Time, rmax, rsum, brk, rc, sc, dP, dT, gnf, fs);
#else
        /* 9-col format. cols 1-8 unchanged from Phase-1; col-9 is the
         * new cbe_pairing_free_slot_count (Commit 6c). */
        fprintf(FdCbeDiagnostics, "%.16g %.16g %.16g %lld %lld %lld %.16g %.16g %lld\n",
                All.Time, rmax, rsum, brk, rc, sc, dP, dT, fs);
#endif
        fflush(FdCbeDiagnostics);
    }
}
#endif


#endif
