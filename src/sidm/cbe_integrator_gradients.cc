/*
 * sidm/cbe_integrator_gradients.cc
 *
 * CBE pre-force gradient module — Wave-CBE Phase 2 commit #5 corrective
 * architecture pivot (2026-05-26, v5). Out-of-line host hooks for
 * CBEGradSpec plus the toplevel driver CBEGrad_gradient_calc(). Mirrors
 * sidm/dm_fuzzy_loop.cc (DMGradSpec / DMGrad_gradient_calc) — same shape,
 * two passes orchestrated at the toplevel via Aux::loop_iteration.
 *
 * Persistent gradient storage lives on P[i].Gradients_CBE_basis_moments
 * (declarations/particle_data.h, gated on CBE_INTEGRATOR_WITHGRADIENTS).
 * Standard P[]-driven ghost import (gizmo_request_filtered_ghost_import_fresh)
 * carries the field naturally; there is no scratch UVM array and no custom
 * Alltoallv. The earlier scratch architecture (CbeGradScratch,
 * CbeGradScratchOwner, cbe_grad_import_ghosts, CbeGradientsSpec,
 * CbeBjLimiterSpec) is gone.
 *
 * Pass-0 writeback solves M^{-1} . B with the same Tikhonov-style
 * ill-conditioning guard used by hydro NV_T inversion
 * (hydro/density_loop.cc:514-525). Pass-1 writeback rescales the
 * already-persistent field in place by phi[m][k] — it MUST NOT re-zero or
 * re-compute the field (codex guard, 2026-05-26).
 *
 * Primitive-gradient swap (2026-06-06 — OPEN_cbe_primitive_grad_design.md):
 * the stored field now holds primitive gradients (∂ρ, ∂v, ∂S) rather than
 * moment gradients (∂m, ∂p, ∂T), packed in the same slot layout. The
 * writeback math is unchanged (M^{-1} · B, then per-(m,k) rescale by phi)
 * — only the upstream LSQ accumulator B is now built from primitive
 * deltas (see cbe_integrator_gradients.h pair bodies). The field name
 * `Gradients_CBE_basis_moments` is now STALE; rename out of scope.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"   /* MUST precede allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "cbe_integrator_gradients.h"

#if defined(CBE_INTEGRATOR_WITHGRADIENTS)

/* ============================================================================
 * Out-of-line CBEGradSpec hooks.
 * ========================================================================== */

/* search_radius — mirror AgsForceSpec exactly so the symmetric face set the
 * pass-1 limiter walks is a SUPERSET of the AgsForce flux pair set. The
 * pass-0 LSQ pair body internally re-narrows acceptance to r < h_i; widening
 * the search radius here does not perturb pass-0 math. Under DM_SIDM the
 * runner search uses the 3x-inflated radius (matching gravity/ags_force_loop.h
 * pattern); the per-pair physics filter (in the inline pair body) uses
 * un-inflated radii. */
double CBEGradSpec::search_radius(const neighbor_loop_args& /*args*/,
                                   int /*active_slot*/, int i)
{
    const double h = (double)P[i].AGS_KernelRadius;
#if defined(DM_SIDM)
    return 3.0 * h;
#else
    return h;
#endif
}

CBEGradSpec::CallScalars
CBEGradSpec::populate_call_scalars(const neighbor_loop_args& args)
{
    CallScalars s;
    s.common         = nlr_common_scalars_from_all();
    s.loop_iteration = static_cast<const Aux*>(args.aux)->loop_iteration;
    return s;
}

/* apply_active_writeback — pass-gated:
 *   pass 0 (LSQ):       solve M^{-1} . B with Tikhonov-style diagonal-loading
 *                       guard; write P[i].Gradients_CBE_basis_moments.
 *   pass 1 (BJ-style):  rescale the already-persistent gradient by
 *                       phi[m][k] IN PLACE (no re-zero, no recompute). */
void CBEGradSpec::apply_active_writeback(const neighbor_loop_args& args,
                                          int active_slot, int i,
                                          const AccumData& accum)
{
    /* These should be impossible under the runner contract — args.aux is
     * set by CBEGrad_gradient_calc before every per-bm dispatch, and the
     * runner only invokes apply_active_writeback for indices it built into
     * the active list (which is bounds-clamped in toplevel). A failure
     * here is a runner / toplevel bug, not a recoverable physics state;
     * loudly abort rather than silently skipping a gradient. */
    const Aux *aux = static_cast<const Aux*>(args.aux);
    if(aux == nullptr) {
        fprintf(stderr,
                "[CBEGradSpec::apply_active_writeback] args.aux == nullptr "
                "(active_slot=%d, i=%d, NumPart=%d, rank=%d) — runner contract violation\n",
                active_slot, i, NumPart, ThisTask);
        endrun(91420);
    }
    if(i < 0 || i >= NumPart) {
        fprintf(stderr,
                "[CBEGradSpec::apply_active_writeback] i=%d out of [0,%d) "
                "(active_slot=%d, rank=%d) — runner handed out-of-range active index\n",
                i, NumPart, active_slot, ThisTask);
        endrun(91421);
    }

    struct particle_data& Pi = args.P[i];

    if(aux->loop_iteration <= 0) {
        /* ------------------------------------------------------------------
         * Pass 0 — LSQ inversion with Tikhonov-style diagonal-loading guard.
         *
         * Same pattern as hydro NV_T inversion (hydro/density_loop.cc:514-525):
         *   threshold = 10 * CONDITION_NUMBER_DANGER
         *   while cond > threshold:
         *     M += (1.05 * trace/NUMDIMS / threshold) * I on diag
         *     cond_term *= 1.2
         * 50-iter hard cap; if it falls through, zero this row so commit #5's
         * flux body cannot ingest an arbitrarily-large gradient.
         * ------------------------------------------------------------------ */
        double M[3][3];
        double M_inv[3][3];
        for(int a = 0; a < 3; a++)
            for(int b = 0; b < 3; b++)
                M[a][b] = accum.M[a][b];

        const double cond_threshold = 10.0 * CONDITION_NUMBER_DANGER;
        const double trace_initial  = M[0][0] + M[1][1] + M[2][2];
        double       cond_term      = 1.05 * (trace_initial / 3.0) / cond_threshold;
        bool         singular       = false;

        for(int it = 0; it < 50; it++) {
            const double cond = matrix_invert_ndims(M, M_inv);
            if(isfinite(cond) && cond < cond_threshold) {
                bool any_bad = false;
                for(int a = 0; a < 3 && !any_bad; a++)
                    for(int b = 0; b < 3 && !any_bad; b++)
                        if(!isfinite(M_inv[a][b])) any_bad = true;
                if(!any_bad) break;
            }
            if(!isfinite(cond_term) || cond_term <= 0) cond_term = MIN_REAL_NUMBER;
            for(int a = 0; a < 3; a++) M[a][a] += cond_term;
            cond_term *= 1.2;
            if(it == 49) singular = true;
        }

        if(singular) {
            for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
                for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                    for(int d = 0; d < 3; d++)
                        Pi.Gradients_CBE_basis_moments[m][k][d] = 0.0;
            return;
        }

        /* grad[m][k][d] = sum_e M_inv[d][e] * B[m][k][e] */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                for(int d = 0; d < 3; d++) {
                    double g = 0.0;
                    for(int e = 0; e < 3; e++) {
                        g += M_inv[d][e] * accum.B[m][k][e];
                    }
                    Pi.Gradients_CBE_basis_moments[m][k][d] = g;
                }
            }
        }
    } else {
        /* ------------------------------------------------------------------
         * Pass 1 — pairwise BJ-style limiter rescale, IN PLACE.
         *
         * Pi.Gradients_CBE_basis_moments was written by pass 0 (and
         * already imported on ghosts via standard P[] ghost transport when
         * the pass-1 runner call ran). We multiply each (m,k,d) component
         * by phi[m][k] clipped to [0, 1] with non-finite -> 0. Also sanitize
         * the persistent value (non-finite -> 0) so a bad pass-0 entry does
         * not propagate into commit #5's flux body.
         * ------------------------------------------------------------------ */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                double phi = accum.phi[m][k];
                if(!isfinite(phi)) phi = 0.0;
                else if(phi < 0.0) phi = 0.0;
                else if(phi > 1.0) phi = 1.0;
                for(int d = 0; d < 3; d++) {
                    const double g_old = Pi.Gradients_CBE_basis_moments[m][k][d];
                    double g_new = (isfinite(g_old) && isfinite(phi))
                                   ? (g_old * phi) : 0.0;
                    if(!isfinite(g_new)) g_new = 0.0;
                    Pi.Gradients_CBE_basis_moments[m][k][d] = g_new;
                }
            }
        }
    }
}

/* merge_accum — additive for M + B; min-merge for phi (NaN -> 0). Unused-pass
 * fields are at their neutral elements (0 for sum, 1 for min) so this single
 * implementation is correct for both passes. */
void CBEGradSpec::merge_accum(AccumData& local, const AccumData& peer)
{
    for(int a = 0; a < 3; a++)
        for(int b = 0; b < 3; b++)
            local.M[a][b] += peer.M[a][b];
    for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            for(int d = 0; d < 3; d++)
                local.B[m][k][d] += peer.B[m][k][d];
            const double p = peer.phi[m][k];
            double      &L = local.phi[m][k];
            if(!isfinite(p)) { L = 0.0; continue; }
            if(p < L) L = p;
        }
    }
}

/* compare_accum — NaN-aware float-relative across M, B, phi. Any non-finite
 * mismatch returns MAX_REAL_NUMBER so the oracle never silently passes with
 * NaN/inf in any field. */
double CBEGradSpec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    double max_rel = 0.0;
    auto upd = [&](double a, double b) {
        /* Any non-finite on either side is a bug — M/B are bounded under
         * correct execution and phi is bounded in [0, 1]. +inf vs +inf
         * (or NaN vs NaN) is NOT a pass; fail unconditionally. */
        if(!isfinite(a) || !isfinite(b)) {
            max_rel = MAX_REAL_NUMBER;
            return;
        }
        const double denom = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
        const double rel   = std::fabs(a - b) / denom;
        if(rel > max_rel) max_rel = rel;
    };
    for(int a = 0; a < 3; a++)
        for(int b = 0; b < 3; b++)
            upd(local.M[a][b], oracle.M[a][b]);
    for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
        for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
            for(int d = 0; d < 3; d++)
                upd(local.B[m][k][d], oracle.B[m][k][d]);
            upd(local.phi[m][k], oracle.phi[m][k]);
        }
    }
    return max_rel;
}

/* Pure i-side accumulation — no j-side writes to suppress. */
void CBEGradSpec::set_oracle_brute_pass(DeviceContext& /*ctx*/, bool /*on*/) {}

/* ============================================================================
 * CBEGrad_gradient_calc — toplevel. Two passes through the runner, separated
 * so pass 1's standard P[] ghost import sees the persistent
 * P[].Gradients_CBE_basis_moments written by pass 0.
 *
 * Active set: AGSForce_isactive (same predicate as the force consumer).
 * Subgroup partitioning: ags_gravity_kernel_shared_BITFLAG (same as
 *   AGSForce_calc, DMGrad_gradient_calc).
 *
 * Pre-zero: pass 0 only (LSQ accumulator scatter); pass 1 rescales in place.
 * Called from core/accel.cc in the pre-force phase, paralleling
 * DMGrad_gradient_calc.
 * ========================================================================== */
void CBEGrad_gradient_calc(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    const double t00 = my_second();
    PRINT_STATUS(" ..calculating CBE basis-moment gradients\n");

    /* Partition local actives by shared AGS neighbor-type bitmask. bm==0
     * particles (not AGS-kernel-sharing) are dropped, matching DMGrad. */
    std::map<int, std::vector<int>> bm_groups;
    for(int i : ActiveParticleList) {
        if(AGSForce_isactive(i)) {
            const int bm = ags_gravity_kernel_shared_BITFLAG(P[i].Type);
            if(bm > 0 && bm < 64) { bm_groups[bm].push_back(i); }
        }
    }

    /* Global bm-presence union — every rank must iterate the same bm set in
     * the same order so the per-bm run_neighbor_loop calls (collective) stay
     * synchronized. Mirrors DMGrad_gradient_calc / ags_density_loop.cc. */
    uint64_t local_bm_presence = 0;
    for(const auto& kv : bm_groups) { local_bm_presence |= (1ULL << kv.first); }
    uint64_t global_bm_presence = local_bm_presence;
    if(NTask > 1) {
        MPI_Allreduce(&local_bm_presence, &global_bm_presence, 1,
                      MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
    }
    if(global_bm_presence == 0) {
        CPU_Step[CPU_AGSDENSMISC] += timediff(t00, my_second());
        return;   /* no AGSForce-active particles anywhere */
    }
    /* (CPU accounting bucket: parallel DMGrad_gradient_calc — both paths
     * land in CPU_AGSDENSMISC.) */

    /* Local union active list — used for the pass-0 pre-zero. */
    std::vector<int> union_actives;
    for(const auto& kv : bm_groups) {
        union_actives.insert(union_actives.end(), kv.second.begin(), kv.second.end());
    }

    /* Two passes: 0 = raw LSQ -> writes P[].Gradients_CBE_basis_moments;
     *             1 = pairwise BJ-style limiter -> rescales the same field
     *                 in place. Each pass is a fresh set of runner calls,
     *                 so pass 1's standard P[] ghost import sees pass-0's
     *                 freshly-written gradients on ghosts (codex guard,
     *                 2026-05-26). */
    for(int pass = 0; pass < 2; pass++) {
        /* Pre-zero only on pass 0; pass 1 rescales the persistent field. */
        if(pass == 0) {
            for(int i : union_actives) {
                std::memset(P[i].Gradients_CBE_basis_moments, 0,
                            sizeof(P[i].Gradients_CBE_basis_moments));
            }
        }

        /* One runner call per bm in the GLOBAL union, ascending order. A
         * rank with no local actives for a bm still enters with
         * num_active=0 — run_neighbor_loop is collective. */
        for(int bm = 1; bm < 64; bm++) {
            if(!(global_bm_presence & (1ULL << bm))) { continue; }
            auto it = bm_groups.find(bm);
            std::vector<int>* lst = (it != bm_groups.end()) ? &it->second : nullptr;

            CBEGradSpec::Aux aux;
            aux.loop_iteration = pass;

            neighbor_loop_args args = nlr_default_args();
            args.active_list = (lst && !lst->empty()) ? lst->data() : nullptr;
            args.num_active  = lst ? (int)lst->size() : 0;
            args.aux         = &aux;
            args.neighbor_type_mask_override = (unsigned int)bm;
            run_neighbor_loop<CBEGradSpec>(args);
        }
    }

    CPU_Step[CPU_AGSDENSMISC] += timediff(t00, my_second());
}

#endif /* CBE_INTEGRATOR_WITHGRADIENTS */
