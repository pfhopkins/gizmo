/* galaxy_sf/dm_dispersion_loop.cc — DMDispersionSpec host hooks and
 * dm_dispersion_finalize_post_runner.
 *
 * Device-callable hooks (pair_kernel, zero_accum, load_active, load_neighbor)
 * live in dm_dispersion_loop.h so the runner instantiates them from GPU TUs.
 * This file owns: is_active, search_radius, populate_call_scalars,
 * apply_active_writeback (endrun sentinel), apply_active_writeback_iterative,
 * after_iter (bisection convergence check), merge_accum, and
 * dm_dispersion_finalize_post_runner.
 *
 * Replaces disp_density_evaluate_gpu();
 * dm_dispersion_gpu.cc retired by DMDispersionSpec.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
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
 * apply_active_writeback_iterative — production writeback.
 *
 * Runner calls this INSTEAD of apply_active_writeback in the post-iter-loop
 * writeback, passing the converged radius and IterScratch. That loop runs
 * exactly once per active, which is what makes writing args.aux here
 * single-valued.
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
    return after_iter_body(ctx.args.P, ctx.i, ctx.iter_index, ctx.h_search_current,
                           ctx.scalars, ctx.scratch, accum, ThisTask);
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
