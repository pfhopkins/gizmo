/* radiation/rt_source_injection_loop.cc — host hooks + runner wrapper +
 * ghost-writeback manifest for RtSrcInjectionSpec.
 *
 * Inline pair body (rt_source_injection_pair_body), structs, and
 * KOKKOS_INLINE_FUNCTION Spec hooks live in rt_source_injection_loop.h so they
 * inline from device kernels (Mode A) and host walkers (Mode B / brute oracle).
 * This translation unit holds host-only hooks: search_radius,
 * populate_call_scalars (NlrCommonScalars common), populate/cleanup_device_context
 * (UVM staging), apply_active_writeback (no-op), merge_accum, ghost-writeback
 * manifest + lifecycle hooks, compare_accum, rt_source_injection_fill_local
 * (the per-source host pack — SSOT), and the non-template wrapper
 * `rt_source_injection_run_loop` (the only entry point from the host
 * orchestration TU in radiation/rt_source_injection.cc).
 *
 * rt_source_injection. Written by Phil Hopkins
 * (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"      /* per-TU AllDeviceMirror — must precede allvars.h */
#include "../declarations/allvars.h"
#include "../declarations/gpu_numeric_macros.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"                       /* MUST precede rt_source_injection_loop.h (no include guards) */
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_writeback_ops.h"
#include "rt_source_injection_loop.h"
#include "rt_source_injection_loop_api.h"

#ifdef RT_SOURCE_INJECTION

/* ============================================================================
 * ACTIVE PREDICATE — defensive only.
 *
 * The toplevel in rt_source_injection.cc builds the active list directly via
 * the canonical rt_sourceinjection_active_check chain (whose Type-validity
 * SSOT is rt_get_source_luminosity, see rt_utilities.cc:57). This predicate is
 * a runner-template requirement and a belt-and-suspenders re-check; it is
 * NEVER called by nlr_build_active_list for this Spec (toplevel hands the
 * runner a pre-built active list via Aux).
 * ========================================================================== */
bool RtSrcInjectionSpec::is_active(int i)
{
    if (P[i].Type == 0)                        return false;   /* sources are non-gas */
    if (P[i].KernelRadius <= 0)                return false;
    if (P[i].NumNgb <= 0)                      return false;
    if (P[i].Mass <= 0)                        return false;
    if (P[i].KernelSum_Around_RT_Source <= 0)  return false;
    return true;
}

/* ============================================================================
 * SEARCH RADIUS
 *
 * The source's own un-scaled KernelRadius. The pair kernel accepts a gas
 * neighbor when r < h_i OR (under angle-weighting) r < h_j, so discovery must
 * cover both the i-side reach (h_i) and the j-side reach (gas whose own kernel
 * reaches the source). The SYMMETRIC search discovers the j-side (r < h_j)
 * reach independently, so the source radius does NOT need to be inflated to
 * find large-h_j gas. A one-sided search that lacks j-side discovery must
 * over-search to ~3*h_i to approximate it; for sparse sources with large
 * h_i that inflation imports a 27x shell of candidates the pair kernel then
 * rejects, and can engulf a whole distant mass concentration.
 * ========================================================================== */
double RtSrcInjectionSpec::search_radius(const neighbor_loop_args& args,
                                         int active_slot, int i)
{
    const Aux *aux = nlr_aux<RtSrcInjectionSpec>(args);
    double h = 0;
    if (aux != nullptr && aux->host_locals != nullptr) {
        h = (double)aux->host_locals[active_slot].KernelRadius;
    } else {
        h = (double)args.P[i].KernelRadius;   /* fallback; shouldn't fire in production */
    }
    return h;
}

/* ============================================================================
 * CALL SCALARS
 *
 * Minimal: the pair kernel reads All.* directly via the per-TU
 * AllDeviceMirror (see reference_all_mirror_design.md). Only the
 * runner-required NlrCommonScalars snapshot is pre-evaluated host-side.
 * ========================================================================== */
RtSrcInjectionSpec::CallScalars
RtSrcInjectionSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    RtSrcInjCallScalars scalars;
    scalars.common = nlr_common_scalars_from_all();
    return scalars;
}

/* ============================================================================
 * DEVICE-CONTEXT LIFECYCLE (UVM staging of per-active host pack)
 * ========================================================================== */
void RtSrcInjectionSpec::populate_device_context(const neighbor_loop_args& args,
                                                  DeviceContext& ctx)
{
    ctx.oracle_dry_run = false;

    Aux *aux = nlr_aux<RtSrcInjectionSpec>(args);
    const int N = args.num_active;
    if (N <= 0 || aux == nullptr || aux->host_locals == nullptr) {
        ctx.per_active_local = nullptr;
        return;
    }
    RtSrcLocalIn *uvm = (RtSrcLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(RtSrcLocalIn));
    std::memcpy(uvm, aux->host_locals, N * sizeof(RtSrcLocalIn));
    ctx.per_active_local = uvm;
}

void RtSrcInjectionSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                                 DeviceContext& ctx)
{
    if (ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<RtSrcLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
}

/* ============================================================================
 * SOURCE-SIDE HOST WRITEBACK — no-op.
 *
 * The legacy rt_source_injection has no post-runner source-side mutation.
 * P[i].Sink_accreted_photon_energy clearing happens during pre-runner host fill
 * (rt_source_injection_fill_local below), not here.
 * ========================================================================== */
void RtSrcInjectionSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                                 int /*active_slot*/, int /*i*/,
                                                 const AccumData& /*accum*/)
{
    /* intentionally empty */
}

/* ============================================================================
 * MERGE — additive on both telemetry fields.
 * ========================================================================== */
void RtSrcInjectionSpec::merge_accum(AccumData& local_accum, const AccumData& peer_accum)
{
    local_accum.sum_dE     += peer_accum.sum_dE;
    local_accum.pair_count += peer_accum.pair_count;
}

/* ============================================================================
 * HOST-FILL HELPER (SSOT)
 *
 * Per-source host pre-fill. Reads P_host[i] and CellP_host[i]; under
 * RT_REINJECT_ACCRETED_PHOTONS+Type==5 also clears
 * P_host[i].Sink_accreted_photon_energy after capture (deterministic
 * host-serial mutation; safe by construction — called once per active source
 * from the toplevel pre-runner build loop).
 *
 * Logic mirrors the retired rt_src_local_fill in rt_source_injection_gpu.cc
 * verbatim; renamed to rt_source_injection_fill_local for naming consistency
 * with the rest of the runner-port surface (mechfb_local_fill,
 * thermal_fb_local_fill, radfb_rp_local_fill).
 * ========================================================================== */
void rt_source_injection_fill_local(int i,
                                     struct particle_data *P_host,
                                     struct gas_cell_data *CellP_host,
                                     struct RtSrcLocalIn *loc)
{
    loc->Pos                         = P_host[i].Pos;
    loc->KernelRadius                = P_host[i].KernelRadius;
    loc->KernelSum_Around_RT_Source  = P_host[i].KernelSum_Around_RT_Source;

    double lum[N_RT_FREQ_BINS];
    int active_check = rt_get_source_luminosity(i, 0, lum, P_host, CellP_host);

    double dt = 1.;
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
    dt = get_particle_feedback_timestep_in_physical(i, P_host);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if (P_host[i].Type == 5) { dt = P_host[i].dt_since_last_gas_search; }
#endif
#if defined(RT_EVOLVE_FLUX)
    for (int k = 0; k < 3; k++) {
        if (P_host[i].Type == 0) { loc->Vel[k] = CellP_host[i].VelPred[k]; }
        else                     { loc->Vel[k] = P_host[i].Vel[k]; }
    }
#endif
#endif
    for (int k = 0; k < N_RT_FREQ_BINS; k++) {
        if (P_host[i].Type == 0 || active_check == 0) { loc->Luminosity[k] = 0; }
        else                                          { loc->Luminosity[k] = lum[k] * dt; }
    }
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    if (P_host[i].Type == 5 && active_check) {
        loc->Luminosity[N_RT_FREQ_BINS - 1] += P_host[i].Sink_accreted_photon_energy;
        P_host[i].Sink_accreted_photon_energy = 0;
    }
#endif
#if defined(RT_REPROCESS_INJECTED_PHOTONS) && defined(RT_CHEM_PHOTOION)
    loc->Dt = dt;
    if (P_host[i].Type > 0) { loc->Density = P_host[i].DensityAroundParticle; }
    else                    { loc->Density = CellP_host[i].Density; }
#endif
}

/* ============================================================================
 * GHOST-WRITEBACK MANIFEST
 *
 * Translates EVERY #ifdef branch from the retired
 * mesh/ghost_writeback.cc::ghost_writeback_rtsrcinjection (cleanup commit
 * retires the legacy bespoke snapshot-diff). No `#error "<flag> port pending"`
 * placeholders (per feedback_port_all_branches_no_error_endstate.md).
 *
 * GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION intentionally does NOT
 * appear here. The legacy in-kernel atomic_exchange branch used assignment
 * semantics that are incompatible with the additive ghost-writeback bundle;
 * its testproblem boundary condition is now imposed by an owner-side post-
 * runner fixup in rt_source_injection.cc (see
 * rt_source_injection_apply_grain_rdi_boundary_fixup there). Local owner-side
 * mutation; no MPI semantics involved.
 * ========================================================================== */

GHOST_WRITEBACK_BUNDLE_BEGIN(rtsrcinjection)
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
    GHOST_WRITEBACK_GAS_ADD_ARRAY(Rad_Je, N_RT_FREQ_BINS)
#else
    GHOST_WRITEBACK_GAS_ADD_ARRAY(Rad_E_gamma, N_RT_FREQ_BINS)
#ifdef RT_EVOLVE_ENERGY
    GHOST_WRITEBACK_GAS_ADD_ARRAY(Rad_E_gamma_Pred, N_RT_FREQ_BINS)
#endif
#ifdef RT_EVOLVE_INTENSITIES
    GHOST_WRITEBACK_GAS_ADD_2D(Rad_Intensity,      N_RT_FREQ_BINS, N_RT_INTENSITY_BINS)
    GHOST_WRITEBACK_GAS_ADD_2D(Rad_Intensity_Pred, N_RT_FREQ_BINS, N_RT_INTENSITY_BINS)
#endif
#ifdef RT_EVOLVE_FLUX
    GHOST_WRITEBACK_GAS_ADD_VEC3_ARRAY(Rad_Flux,      N_RT_FREQ_BINS)
    GHOST_WRITEBACK_GAS_ADD_VEC3_ARRAY(Rad_Flux_Pred, N_RT_FREQ_BINS)
#endif
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(Vel)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(dp)
    GHOST_WRITEBACK_GAS_ADD_VEC3(VelPred)
#endif
GHOST_WRITEBACK_BUNDLE_END(rtsrcinjection)

/* ghost_write_detector_begin/end: runner default (loop_name = "rtsrcinjection"). */

void RtSrcInjectionSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                                 const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_begin_bundle(rtsrcinjection_ghost_writeback_bundle_ptr());
}

void RtSrcInjectionSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                               const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(rtsrcinjection_ghost_writeback_bundle_ptr());
}

/* ============================================================================
 * DIAGNOSTICS — env-gated oracle compare.
 *
 * AccumData here is TELEMETRY (sum_dE = Σ_pair wk*Σ_k Lum[k]; pair_count). Not
 * a physics surrogate — the real j-side writes are atomic_add direct in the
 * pair kernel, and audited by the per-field ghost-writeback bundle.
 * ========================================================================== */

/* ============================================================================
 * RUNNER WRAPPER (non-template)
 *
 * The only entry point from the host TU (radiation/rt_source_injection.cc).
 * Keeps Kokkos / runner-template instantiation isolated to this GPU TU; the
 * host TU sees only the forward decl in rt_source_injection_loop_api.h.
 *
 * Caller pre-conditions:
 *   - active_src + host_locals are toplevel-owned buffers, valid for the
 *     duration of this call.
 *   - global_num_active > 0 (host TU has done the MPI_Allreduce gate).
 *   - rt_source_injection_initial_operations_preloop has already run for any
 *     gas-source self-emission case.
 * ========================================================================== */
void rt_source_injection_run_loop(const int *active_src,
                                   const struct RtSrcLocalIn *host_locals,
                                   int num_active)
{
    RtSrcInjectionSpec::Aux aux{};
    aux.host_locals     = host_locals;
    aux.host_active_src = active_src;

    /* Non-iterative engine handoff — mirrors thermal_fb_calc in
     * galaxy_sf/thermal_fb.cc. nlr_default_args() supplies P/CellP/num_total
     * and other defaults; we only override active_list, num_active, aux.
     * Subgroup / ghost_safety_factor / num_total wiring is for
     * neighbor_loop_args_iterative (radfb_rp pattern), not us. */
    neighbor_loop_args args = nlr_default_args();
    args.active_list = const_cast<int *>(active_src);
    args.num_active  = num_active;
    args.aux         = &aux;

    run_neighbor_loop<RtSrcInjectionSpec>(args);
}

#endif /* RT_SOURCE_INJECTION */
