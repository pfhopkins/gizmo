/* turb/difffilter_loop.cc — DiffFilterSpec / DynDiffSpec host hooks + the two
 * toplevel callers for the runner-template port of difffilter_evaluate_gpu /
 * dynamicdiff_evaluate_gpu.
 *
 * Device-callable hooks (pair_kernel, zero_accum, load_active, load_neighbor)
 * live in difffilter_loop.h so the runner instantiates them from GPU TUs.
 * This file owns: is_active, search_radius, populate_call_scalars,
 * apply_active_writeback, merge_accum, symmetric_neighbor_radius_scale,
 * and the toplevels.
 *
 * Replaces turb/difffilter_gpu.cc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"            /* MUST precede difffilter_loop.h (no include guards) */
#include "../mesh/neighbor_loop_runner.h"
#include "../hydro/hydro_corridor.h"   /* mode + external_csr accessors */
#include "difffilter_loop.h"

#ifdef TURB_DIFF_DYNAMIC


/* ============================================================================
 * DiffFilterSpec host hooks.
 * ========================================================================== */

/* is_active — legacy dynamic_diff_vel_calc active filter (gas, live, massive). */
/* is_active — gas + massive. This is character-identical to the predicate the
 * hydro corridor builds its shared row list from, which is what lets this loop
 * consume that list directly. The legacy TimeBin >= 0 test is gone: TimeBin is
 * driven negative only as a transient done-marker inside the density kernel
 * radius iteration, and ActiveParticleList is walked out of the per-bin lists,
 * so an active particle always carries a non-negative bin. Keeping the test
 * would only make this predicate disagree with the corridor's. */
bool DiffFilterSpec::is_active(int i)
{
    if (P[i].Type != 0)   return false;
    if (P[i].Mass <= 0)   return false;
    return true;
}

/* search_radius — wide dynamic-diffusion kernel: fac * P[i].KernelRadius. */
double DiffFilterSpec::search_radius(const neighbor_loop_args& args,
                                     int /*active_slot*/, int i)
{
    return nlr_host_all_ptr()->TurbDynamicDiffFac * (double)args.P[i].KernelRadius;
}

DiffFilterSpec::CallScalars
DiffFilterSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    CallScalars s;
    s.common            = nlr_common_scalars_from_all();
    s.turb_dyn_diff_fac = nlr_host_all_ptr()->TurbDynamicDiffFac;
    return s;
}

/* apply_active_writeback — scatter directly into CellP[i]. The preloop
 * (dynamic_diff_vel_calc_initial_operations_preloop) pre-zeroes Norm_hat /
 * FilterWidth_bar / MaxDistance_for_grad and seeds Velocity_bar = VelPred /
 * TurbDynamicDiffSmoothing, so += / max-merge here matches the legacy
 * scatter in dynamic_diffusion_velocities.cc:82-88. */
void DiffFilterSpec::apply_active_writeback(const neighbor_loop_args& args,
                                            int /*active_slot*/, int i,
                                            const AccumData& accum)
{
    struct gas_cell_data& C = args.CellP[i];
    C.Norm_hat += accum.norm_hat;
    for (int k = 0; k < 3; k++) C.Velocity_bar[k] += accum.velocity_bar_delta[k];
    if (accum.filter_width_bar  > C.FilterWidth_bar)
        C.FilterWidth_bar = (MyFloat)accum.filter_width_bar;
    if (accum.max_dist_for_grad > C.MaxDistance_for_grad)
        C.MaxDistance_for_grad = (MyFloat)accum.max_dist_for_grad;
}


double DiffFilterSpec::symmetric_neighbor_radius_scale()
{
    return nlr_host_all_ptr()->TurbDynamicDiffFac;
}

/* ============================================================================
 * DynDiffSpec host hooks.
 * ========================================================================== */

/* is_active — legacy dynamic_diff_calc filter (gas + massive; NO TimeBin
 * check — distinct from DiffFilterSpec; see dynamic_diffusion.cc:133). */
bool DynDiffSpec::is_active(int i)
{
    if (P[i].Type != 0) return false;
    if (P[i].Mass <= 0) return false;
    return true;
}

double DynDiffSpec::search_radius(const neighbor_loop_args& args,
                                  int /*active_slot*/, int i)
{
    return nlr_host_all_ptr()->TurbDynamicDiffFac * (double)args.P[i].KernelRadius;
}

DynDiffSpec::CallScalars
DynDiffSpec::populate_call_scalars(const neighbor_loop_args& args)
{
    CallScalars s;
    s.common            = nlr_common_scalars_from_all();
    s.turb_dyn_diff_fac = nlr_host_all_ptr()->TurbDynamicDiffFac;
    s.dynamic_iteration = static_cast<const Aux*>(args.aux)->dynamic_iteration;
    return s;
}

/* apply_active_writeback — scatter into the caller's DynamicDiffDataPasser.
 * dynamic_fac[/_const] every iteration; the iter-0-only block gated on
 * aux->dynamic_iteration. Matches dynamic_diffusion.cc:184-210. */
void DynDiffSpec::apply_active_writeback(const neighbor_loop_args& args,
                                         int /*active_slot*/, int i,
                                         const AccumData& accum)
{
    Aux* aux = static_cast<Aux*>(args.aux);
    struct temporary_data_dyndiff& d = aux->dddp[i];

    for (int k = 0; k < 3; k++) {
        for (int v = 0; v < 3; v++) {
            d.dynamic_fac[k][v]       += (MyDouble)accum.dynamic_fac[k][v];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
            d.dynamic_fac_const[k][v] += (MyDouble)accum.dynamic_fac_const[k][v];
#endif
        }
    }

    if (aux->dynamic_iteration == 0) {
        if (accum.filter_width_hat > (double)d.FilterWidth_hat)
            d.FilterWidth_hat = (MyFloat)accum.filter_width_hat;
        d.Dynamic_numerator_hat   += (MyDouble)accum.dynamic_numerator_hat;
        d.Dynamic_denominator_hat += (MyDouble)accum.dynamic_denominator_hat;
        for (int k = 0; k < 3; k++) {
            if (accum.maxima_velocity_hat[k] > d.Maxima.Velocity_hat[k])
                d.Maxima.Velocity_hat[k] = accum.maxima_velocity_hat[k];
            if (accum.minima_velocity_hat[k] < d.Minima.Velocity_hat[k])
                d.Minima.Velocity_hat[k] = accum.minima_velocity_hat[k];
            for (int v = 0; v < 3; v++) {
                d.ProductVelocity_hat[k][v] += (MyDouble)accum.product_velocity_hat[k][v];
                d.GradVelocity_hat[k][v]    += (MyDouble)accum.grad_velocity_hat[k][v];
            }
        }
    }
}


double DynDiffSpec::symmetric_neighbor_radius_scale()
{
    return nlr_host_all_ptr()->TurbDynamicDiffFac;
}

/* ============================================================================
 * Toplevels.
 * ========================================================================== */

/* Runs inside the corridor span, between cellcorrections and the gradients, so
 * it consumes the corridor's shared topology on the same ownership contract as
 * the other consumers. Rows are exact: is_active is the corridor's own row
 * predicate. Reach matches DynDiff's — TurbDynamicDiffFac * h on both sides,
 * with the pair kernel's distance test discarding anything the corridor is
 * over-inclusive about when the factor is below one.
 *
 * Ghost values: the only routine between the corridor's import and this loop is
 * cellcorrections, and it rewrites owner-side Density, which this pair kernel
 * reads on neighbours. So under volume corrections the ghost copies must be
 * resynced first; with volume corrections compiled out nothing has dirtied them
 * and the corridor's import is still current. VelPred, the other field of
 * interest here, is written by feedback above the corridor and so is already
 * current in that import. */
void difffilter_vel_calc_gpu_toplevel(void)
{
#ifdef HYDRO_VOLUME_CORRECTIONS
    if (gizmo_hydro_corridor_external_csr() != nullptr) {
        gizmo_hydro_corridor_refresh_ghost_values("pre_diff_vel");
    }
#endif
    const nlr_external_csr       *corridor_csr  = gizmo_hydro_corridor_external_csr();
    const GizmoHydroCorridorMode  corridor_mode = gizmo_hydro_corridor_get_mode();

    int *active_list_local = nullptr;   /* allocated by nlr_build_active_list in the fallback only */
    int  num_active = 0, num_global_active = 0;

    DiffFilterSpec::Aux aux;   /* unused — DiffFilter scatters straight to CellP */
    neighbor_loop_args args = nlr_default_args();
    args.aux = &aux;

    if (corridor_csr != nullptr) {
        /* Mode A external-CSR path. The row list belongs to the corridor and
         * outlives this call; do NOT free it here. */
        args.active_list       = corridor_csr->active_indices;
        args.num_active        = corridor_csr->num_active;
        args.external_csr      = corridor_csr;
        args.dispatch_override = NlrForceMode::A;
    } else {
        /* Mode B (request-driven, no corridor CSR): build the list here. */
        if (!nlr_build_active_list(DiffFilterSpec::is_active,
                                   &active_list_local, &num_active, &num_global_active,
                                   "difffilter_active_list")) {
            return;   /* no active gas anywhere this step */
        }
        /* Mode A always publishes a corridor view when there is active gas, so
         * reaching here in Mode A is a corridor sequencing bug — fail loudly
         * rather than quietly rebuilding a second topology. */
        if (corridor_mode == GizmoHydroCorridorMode::MODE_A) {
            printf("FATAL: difffilter_vel_calc_gpu_toplevel in Mode A with active gas but no published corridor CSR on task %d.\n", ThisTask);
            fflush(stdout);
            endrun(7318);
        }
        args.active_list = active_list_local;
        args.num_active  = num_active;
        if (corridor_mode == GizmoHydroCorridorMode::MODE_B) {
            args.dispatch_override = NlrForceMode::B;
        }
    }

    run_neighbor_loop<DiffFilterSpec>(args);

    if (active_list_local) nlr_free_active_list(active_list_local);
}

/* This loop runs between the gradient and hydro-force consumers, so the
 * corridor's symmetric gas topology is already built and still live. Consuming
 * it, rather than importing a private ghost pool, is what keeps this loop from
 * tearing the corridor's pool and forcing it to re-import before hydro_force.
 *
 * Rows are exact, not a superset: DynDiffSpec::is_active is Type==0 && Mass>0
 * over ActiveParticleList, character-identical to the predicate the corridor
 * builds its list from, so no per-row gate is needed here.
 *
 * Ghost VALUES must be current, not just the topology, and this loop needs its
 * OWN refresh point: the pair kernel reads j-side Velocity_hat, and
 * hydro_gradient_calc overwrites owner-side Velocity_hat (gradients_loop.cc,
 * Velocity_hat = Velocity_bar * smoothInv) AFTER the corridor's pre-gradients
 * refresh has already run. Consuming the corridor pool without refreshing here
 * would therefore read pre-gradient values on every ghost. Refresh once, at the
 * first iteration: the j-side fields this loop reads are then frozen for the
 * rest of the sequence, because the host work between iterations writes only
 * the Dynamic_, TD_ and Kappa_Conduction fields and this Spec's own writeback
 * targets the caller's passer rather than CellP. The refresh takes the
 * value-only fast path (no intervening loop tears the pool between gradients
 * and here), so it costs no import; a corridor import count above one per step
 * would mean it fell through to the rebuild path. It may republish, so the CSR
 * view is fetched after it, never cached across it.
 *
 * Reach is exact where it matters and safe where it is not. The corridor sizes
 * its import and its symmetric list with DMAX(1, TurbDynamicDiffFac), while
 * this loop searches TurbDynamicDiffFac * h. For fac >= 1 the two agree. For
 * fac < 1 the corridor is over-inclusive, which is harmless because the pair
 * kernel applies its own distance test -- r2 >= h_i^2 && r2 >= h_j^2 with both
 * radii scaled by turb_dyn_diff_fac -- and discards the extra neighbors. */
void dynamicdiff_gpu_toplevel(int dynamic_iteration,
                              struct temporary_data_dyndiff *dddp)
{
    if (dynamic_iteration == 0 && gizmo_hydro_corridor_external_csr() != nullptr) {
        gizmo_hydro_corridor_refresh_ghost_values("pre_dyndiff");
    }
    const nlr_external_csr       *corridor_csr  = gizmo_hydro_corridor_external_csr();
    const GizmoHydroCorridorMode  corridor_mode = gizmo_hydro_corridor_get_mode();

    int *active_list_local = nullptr;   /* allocated by nlr_build_active_list in the fallback only */
    int  num_active = 0, num_global_active = 0;

    DynDiffSpec::Aux aux;
    aux.dddp              = dddp;
    aux.dynamic_iteration = dynamic_iteration;

    neighbor_loop_args args = nlr_default_args();
    args.aux = &aux;

    if (corridor_csr != nullptr) {
        /* Mode A external-CSR path. The row list belongs to the corridor and
         * outlives this call; do NOT free it here. */
        args.active_list       = corridor_csr->active_indices;
        args.num_active        = corridor_csr->num_active;
        args.external_csr      = corridor_csr;
        args.dispatch_override = NlrForceMode::A;
    } else {
        /* Mode B (request-driven, no corridor CSR): build the list here. */
        if (!nlr_build_active_list(DynDiffSpec::is_active,
                                   &active_list_local, &num_active, &num_global_active,
                                   "dyndiff_active_list")) {
            return;   /* no active gas anywhere this step */
        }
        /* Mode A always publishes a corridor view when there is active gas, so
         * reaching here in Mode A is a corridor sequencing bug — fail loudly
         * rather than quietly rebuilding a second topology. */
        if (corridor_mode == GizmoHydroCorridorMode::MODE_A) {
            printf("FATAL: dynamicdiff_gpu_toplevel in Mode A with active gas but no published corridor CSR on task %d.\n", ThisTask);
            fflush(stdout);
            endrun(7317);
        }
        args.active_list = active_list_local;
        args.num_active  = num_active;
        if (corridor_mode == GizmoHydroCorridorMode::MODE_B) {
            args.dispatch_override = NlrForceMode::B;
        }
    }

    run_neighbor_loop<DynDiffSpec>(args);

    if (active_list_local) nlr_free_active_list(active_list_local);
}

#endif /* TURB_DIFF_DYNAMIC */
