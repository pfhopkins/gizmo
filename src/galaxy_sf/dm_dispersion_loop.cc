/* galaxy_sf/dm_dispersion_loop.cc — DMDispersionSpec host hooks and
 * dm_dispersion_finalize_post_runner.
 *
 * Device-callable hooks (pair_kernel, zero_accum, load_active, load_neighbor)
 * live in dm_dispersion_loop.h so the runner instantiates them from GPU TUs.
 * This file owns: is_active, search_radius, populate_call_scalars,
 * apply_active_writeback (endrun sentinel), apply_active_writeback_iterative,
 * after_iter (bisection convergence check), merge_accum, compare_accum,
 * set_oracle_brute_pass, and dm_dispersion_finalize_post_runner.
 *
 * Phase 4 Wave-4 / 3d.D. Replaces disp_density_evaluate_gpu();
 * dm_dispersion_gpu.cc retired by DMDispersionSpec.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"                       /* MUST precede dm_dispersion_loop.h (no include guards) */
#include "dm_dispersion_loop.h"

#ifdef DM_DISPERSION_LOOP_ACTIVE

/* ============================================================================
 * is_active — active-particle predicate.
 *
 * Mirrors legacy disp_density_isactive() exactly:
 *   TimeBin >= 0, Type == 0 (gas only), Mass > 0.
 * The runner's active-set compaction replaces the legacy negative-TimeBin
 * marker; this port does NOT mutate P[i].TimeBin.
 * ========================================================================== */
bool DMDispersionSpec::is_active(int i)
{
    if (P[i].TimeBin < 0) return false;
    if (P[i].Type != 0)   return false;
    if (P[i].Mass <= 0)   return false;
    return true;
}

/* ============================================================================
 * search_radius — per-active radius hook.
 *
 * Returns CellP[i].KernelRadiusDM (the DM-search kernel, distinct from the
 * gas hydro kernel P[i].KernelRadius). Runner stages this into radii_uvm at
 * iter 0; after_iter mutates per-iter via IterResult::new_h_search.
 * ========================================================================== */
double DMDispersionSpec::search_radius(const neighbor_loop_args& args,
                                        int /*active_slot*/, int i)
{
    return (double)args.CellP[i].KernelRadiusDM;
}

/* ============================================================================
 * populate_call_scalars — per-call CallScalars snapshot.
 *
 * Reads through nlr_host_all_ptr() for explicit-intent host access; bare
 * `All.*` here would also be safe under the per-TU AllDeviceMirror scheme
 * (device-pass-only macro), but the named accessor keeps the snapshot
 * intent visible.
 * ========================================================================== */
DMDispersionSpec::CallScalars
DMDispersionSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    CallScalars scalars;
    scalars.common             = nlr_common_scalars_from_all();
    scalars.max_kernel_radius  = nlr_host_all_ptr()->MaxKernelRadius;
    return scalars;
}

/* ============================================================================
 * apply_active_writeback — unreachable sentinel.
 *
 * The runner's SFINAE detector selects apply_active_writeback_iterative for
 * Specs that declare it (nlr_spec_has_apply_active_writeback_iterative_v).
 * This body fires only if that detector breaks; endrun(91201) surfaces it
 * loudly rather than silently producing garbage output.
 * ========================================================================== */
void DMDispersionSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                               int /*active_slot*/, int /*i*/,
                                               const AccumData& /*accum*/)
{
    endrun(91201); /* unreachable: apply_active_writeback_iterative supersedes
                      this for DMDispersionSpec — runner SFINAE detector broken */
}

/* ============================================================================
 * apply_active_writeback_iterative — oracle-safe production writeback.
 *
 * Runner calls this INSTEAD of apply_active_writeback in the post-iter-loop
 * writeback, passing the converged radius and IterScratch. Oracle does NOT
 * invoke writeback at all — its accum lives in a separate accum_oracle_uvm
 * buffer. Therefore writing args.aux here is oracle-safe.
 * ========================================================================== */
void DMDispersionSpec::apply_active_writeback_iterative(
        const neighbor_loop_args& args,
        int active_slot, int /*i*/,
        const AccumData& accum,
        double final_h,
        const IterScratch& /*scratch*/)
{
    auto* aux = static_cast<Aux*>(args.aux);
    aux->per_active_final_accum[active_slot] = accum;
    aux->per_active_final_h[active_slot]     = final_h;
}

/* ============================================================================
 * set_oracle_brute_pass — no-op.
 *
 * No j-side writes in this Spec (uses_ghost_writeback=false), so no
 * suppression is needed during the oracle brute pass. Runner calls without
 * SFINAE guard; body is a no-op.
 * ========================================================================== */
void DMDispersionSpec::set_oracle_brute_pass(DeviceContext& /*ctx*/, bool /*on*/)
{
    /* no-op; dm_dispersion pair_kernel has no j-side writes */
}

/* ============================================================================
 * compare_accum — per-field max relative diff for oracle comparison.
 *
 * All five AccumData fields are double; relative diff uses fmax(1, |a|, |b|)
 * as denominator to avoid divide-by-zero on zero fields.
 * ========================================================================== */
double DMDispersionSpec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    double max_rel = 0.0;
    auto upd = [&](double a, double b) {
        const double denom = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
        const double rel   = std::fabs(a - b) / denom;
        if (rel > max_rel) max_rel = rel;
    };
    upd(local.Ngb,        oracle.Ngb);
    upd(local.DM_Vx,      oracle.DM_Vx);
    upd(local.DM_Vy,      oracle.DM_Vy);
    upd(local.DM_Vz,      oracle.DM_Vz);
    upd(local.DM_VelDisp, oracle.DM_VelDisp);
    upd(local.DM_Rho,     oracle.DM_Rho);
    return max_rel;
}

/* ============================================================================
 * merge_accum — Mode B remote peer merge (all fields additive).
 *
 * All five fields are simple sums; no MIN reductions (unlike density's
 * Sink_TimeBinGasNeighbor). Mismatch vs pair_kernel semantics = silent
 * multi-rank corruption; only oracle validation catches it.
 * ========================================================================== */
void DMDispersionSpec::merge_accum(AccumData& local, const AccumData& peer)
{
    local.Ngb        += peer.Ngb;
    local.DM_Vx      += peer.DM_Vx;
    local.DM_Vy      += peer.DM_Vy;
    local.DM_Vz      += peer.DM_Vz;
    local.DM_VelDisp += peer.DM_VelDisp;
    local.DM_Rho     += peer.DM_Rho;
}

/* ============================================================================
 * after_iter — bisection convergence check + radius update.
 *
 * Literal port of the convergence + radius-adjustment block inside the
 * legacy disp_density() do-while loop (dm_dispersion.cc:102-173), with
 * the runner contract applied:
 *   - READS: accum, ctx.scratch (Left/Right), ctx.h_search_current,
 *     ctx.scalars, ctx.iter_index, ctx.args.P[ctx.i].KernelRadius
 *     (gas hydro kernel, used for maxsoft cap).
 *   - WRITES: ONLY ctx.scratch (Left/Right brackets).
 *   - RETURNS: IterResult{Converged, new_h} or {AdjustRadius, new_h}.
 *
 * NumNgb here is the unweighted count accum.Ngb (each j contributes 1.0),
 * NOT a kernel-weighted quantity like in density's after_iter.
 * ========================================================================== */
IterResult DMDispersionSpec::after_iter(const AfterIterContext<DMDispersionSpec>& ctx,
                                         const AccumData& accum)
{
    const neighbor_loop_args_iterative& args = ctx.args;
    const int    i       = ctx.i;
    const double ngb     = accum.Ngb;
    double       new_h   = ctx.h_search_current;
    DMDispIterScratch& scratch = ctx.scratch;
    const int    iter    = ctx.iter_index;

    /* maxsoft: legacy dm_dispersion.cc:107 */
    const double maxsoft = DMIN(ctx.scalars.max_kernel_radius,
                                10.0 * (double)args.P[i].KernelRadius);

    /* Convergence check (legacy lines 108-117):
     *   redo = (Ngb out of target band) AND (bracket not tight)
     *   Force converge if h >= maxsoft (clamp to maxsoft). */
    int redo_particle = 0;
    if (((ngb < kDesNumNgb - kDesNumNgbDev) || (ngb > kDesNumNgb + kDesNumNgbDev))
        && ((double)scratch.Right - (double)scratch.Left > 0.001 * (double)scratch.Left
            || scratch.Left == 0 || scratch.Right == 0))
    {
        redo_particle = 1;
    }
    if (new_h >= maxsoft) {
        new_h = maxsoft;
        redo_particle = 0;
    }

    if (!redo_particle) {
        return IterResult{IterStatus::Converged, new_h};
    }

    /* Iteration warning (legacy line 127-130) */
    if (iter >= MAXITER - 10) {
        PRINT_WARNING("DM disp: i=%d task=%d ID=%llu Type=%d KernelRadiusDM=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
            i, ThisTask, (unsigned long long)args.P[i].ID, args.P[i].Type,
            new_h, (double)scratch.Left, (double)scratch.Right, ngb,
            (double)scratch.Right - (double)scratch.Left,
            args.P[i].Pos[0], args.P[i].Pos[1], args.P[i].Pos[2]);
    }

    /* Bracket update (legacy lines 124-125) */
    if (ngb < kDesNumNgb - kDesNumNgbDev) {
        if (new_h > (double)scratch.Left) scratch.Left = (float)new_h;
    } else {
        if (scratch.Right > 0) {
            if (new_h < (double)scratch.Right) scratch.Right = (float)new_h;
        } else {
            scratch.Right = (float)new_h;
        }
    }

    /* New radius computation (legacy lines 133-172) */
    if (scratch.Right > 0 && scratch.Left > 0) {
        /* Geometric interpolation between right/left (legacy line 139-150) */
        if (ngb > 1) {
            new_h *= pow(kDesNumNgb / ngb, 1.0 / NUMDIMS);
        } else {
            new_h *= 2.0;
        }
        if ((new_h < (double)scratch.Right) && (new_h > (double)scratch.Left)) {
            new_h = pow(new_h * new_h * new_h * new_h
                        * (double)scratch.Left * (double)scratch.Right, 1.0/6.0);
        } else {
            if (new_h > (double)scratch.Right) new_h = (double)scratch.Right;
            if (new_h < (double)scratch.Left)  new_h = (double)scratch.Left;
            new_h = pow(new_h * (double)scratch.Left * (double)scratch.Right, 1.0/3.0);
        }
    } else {
        /* One-sided bracket (legacy lines 155-172) */
        if (scratch.Right == 0 && scratch.Left == 0) {
            printf("DMDispersionSpec::after_iter: Right==0 && Left==0 && KernelRadiusDM=%g (task=%d)\n", new_h, ThisTask); fflush(stdout);
            endrun(90001011);
            return IterResult{IterStatus::Converged, new_h};   /* graceful: bad-stop set; stop iterating this active with the last valid DM-dispersion radius; drains at runner completion -> phase poll (per-active: NO immediate collective) */
        }
        double fac;
        if (scratch.Right == 0 && scratch.Left > 0) {
            if (ngb > 1) { fac = log(kDesNumNgb / ngb) / NUMDIMS; } else { fac = 1.4; }
            if (ngb < 2*kDesNumNgb && ngb > 0.1*kDesNumNgb) { new_h *= exp(fac); }
            else { new_h *= 1.26; }
        }
        if (scratch.Right > 0 && scratch.Left == 0) {
            if (ngb > 1) { fac = log(kDesNumNgb / ngb) / NUMDIMS; } else { fac = 1.4; }
            fac = DMAX(fac, -1.535);
            if (ngb < 2*kDesNumNgb && ngb > 0.1*kDesNumNgb) { new_h *= exp(fac); }
            else { new_h /= 1.26; }
        }
    }

    return IterResult{IterStatus::AdjustRadius, new_h};
}

/* ============================================================================
 * dm_dispersion_finalize_post_runner — scatter converged AccumData to CellP[].
 *
 * Called once after run_neighbor_loop_iterative returns. Reads fresh cf_atime
 * via nlr_common_scalars_from_all() (not passed through Aux; ensures the
 * cosmological factor is current at finalize time, not frozen at iter-0).
 *
 * Writes all six CellP fields explicitly for every active — no field left
 * at a stale value from a prior step.
 *
 * Variance guard:
 *   variance < -1e-6 * v2_mean  → PRINT_WARNING + endrun (broken accumulator)
 *   -1e-6 * v2_mean <= variance < 0  → clamp to zero (float noise)
 * ========================================================================== */
void dm_dispersion_finalize_post_runner(const neighbor_loop_args_iterative& args)
{
    const NlrCommonScalars common = nlr_common_scalars_from_all();
    auto* aux = static_cast<DMDispersionSpec::Aux*>(args.aux);
    const int N = args.num_active;

    for (int slot = 0; slot < N; ++slot) {
        const int i = args.active_list[slot];
        const DMDispOut& accum    = aux->per_active_final_accum[slot];
        const double     final_h  = aux->per_active_final_h[slot];

        CellP[i].KernelRadiusDM = (MyFloat)final_h;

        if (accum.Ngb > 0) {
            const double inv_ngb  = 1.0 / accum.Ngb;
            const double vx       = accum.DM_Vx * inv_ngb;
            const double vy       = accum.DM_Vy * inv_ngb;
            const double vz       = accum.DM_Vz * inv_ngb;
            const double v2_mean  = accum.DM_VelDisp * inv_ngb;
            const double vmean_sq = vx*vx + vy*vy + vz*vz;
            double variance = v2_mean - vmean_sq;

            if (variance < -1e-6 * v2_mean) {
                PRINT_WARNING("dm_dispersion finalize: negative variance "
                    "i=%d Ngb=%g v2_mean=%g vmean_sq=%g variance=%g\n",
                    i, accum.Ngb, v2_mean, vmean_sq, variance);
                endrun(91203);
            }
            if (variance < 0) variance = 0;

            CellP[i].NumNgbDM   = (MyDouble)accum.Ngb;
            CellP[i].DM_Vx      = (MyDouble)vx;
            CellP[i].DM_Vy      = (MyDouble)vy;
            CellP[i].DM_Vz      = (MyDouble)vz;
            CellP[i].DM_VelDisp = (MyDouble)(sqrt(variance) / (sqrt(3.) * common.cf_atime));
            /* Comoving kernel density → physical via cf_a3inv. */
            CellP[i].DM_Rho     = (MyDouble)(accum.DM_Rho * common.cf_a3inv);
        } else {
            /* No DM neighbors found: zero velocity stats; fall back to
             * gas particle's own velocity for dispersion estimate (legacy
             * dm_dispersion.cc:215-216). */
            CellP[i].NumNgbDM   = 0;
            CellP[i].DM_Vx      = 0;
            CellP[i].DM_Vy      = 0;
            CellP[i].DM_Vz      = 0;
            const double vel_mag = sqrt((double)P[i].Vel[0]*(double)P[i].Vel[0]
                                      + (double)P[i].Vel[1]*(double)P[i].Vel[1]
                                      + (double)P[i].Vel[2]*(double)P[i].Vel[2]);
            CellP[i].DM_VelDisp = (MyDouble)(vel_mag / common.cf_atime);
            CellP[i].DM_Rho     = 0;
        }
    }
}

#endif /* DM_DISPERSION_LOOP_ACTIVE */
