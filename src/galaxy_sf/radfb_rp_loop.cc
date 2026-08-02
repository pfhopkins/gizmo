/* galaxy_sf/radfb_rp_loop.cc — host-only hooks for RadFBRPSpec.
 *
 * Inline pair body (radfb_rp_pair_kick), structs, and KOKKOS_INLINE_FUNCTION
 * hooks live in radfb_rp_loop.h so they inline from device kernels (Mode A)
 * and host walkers (Mode B / Brute oracle). This translation unit holds
 * host-only hooks: is_active (defensive cosmo-aware Type filter), per-active
 * radius, per-call scalars capture (NlrCommonScalars + host-precomputed unit
 * factors via nlr_host_all_ptr), populate/cleanup_device_context (UVM staging
 * for per_active_local), reset_per_iter_device_context (sets Aux::iter_index
 * + ctx.iter_index_snapshot — THE single source of truth for iter-gated
 * writeback), apply_active_writeback (source-side jet_momentum_used decrement
 * under STELLAREVOLUTION>2), merge_accum (sum wt_sum + sum jet_momentum_used),
 * after_iter (iter-0 writes ctx.scratch.wt_sum = accum.wt_sum, returns
 * Converged/NeedsMore; iter-1 always Converged — no P/CellP writes, oracle-
 * safe), and after_iter_global (post-iter staging hook, NO physics-side
 * writes; bridges drv.scratch_uvm[sg][slot].wt_sum into
 * drv.ctx.per_active_local[slot].wt_sum and drv.scratch_oracle_uvm[sg][slot].wt_sum
 * into drv.ctx_oracle.per_active_local[slot].wt_sum, then Kokkos::fence(),
 * so iter-1's device kernels read the staged denominator).
 *
 * Also holds: ghost-writeback bundle (PARTICLE_ADD_VEC3 on Vel/dp +
 * GAS_ADD_VEC3 on VelPred — new generic op landed for this port), and the
 * four ghost-writeback hook bodies (all gated by Aux::iter_index — iter 0
 * no-ops on every rank for collective symmetry; iter 1 fires the bundle).
 *
 * Also holds: radfb_rp_local_fill (the CPU stochastic gate — SSOT, called
 * by the toplevel in radfb_local.cc::radiation_pressure_winds_consolidated;
 * NOT called via Spec::is_active because is_active is a bool predicate and
 * the gate produces per-source LocalIn + search-radius payload).
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
#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/constants.h"           /* UNIT_*_IN_* macros (host-side use only) */
#include "../core/proto.h"
#include "../mesh/kernel.h"                       /* MUST precede radfb_rp_loop.h (no include guards) */
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_writeback_ops.h"
#include "../mesh/ghost_symlist_lifecycle.h"   /* gizmo_ghost_safety_factor */
#include "radfb_rp_loop.h"

#ifdef GALSF_FB_FIRE_RT_LOCALRP

/* ============================================================================
 * DEFENSIVE ACTIVE PREDICATE
 *
 * NOT a stochastic gate — that lives in radfb_rp_local_fill (below) and runs
 * in the toplevel BEFORE the runner call. This predicate is a belt-and-
 * suspenders cosmo-aware Type re-check, NEVER invoked via
 * nlr_build_active_list (which would re-evaluate over un-gated
 * ActiveParticleList and re-include non-firing sources). The toplevel hands
 * the runner a pre-built active_src array of already-gated sources.
 * ========================================================================== */

bool RadFBRPSpec::is_active(int i)
{
    /* Host-only function in a GPU TU. Under the per-TU AllDeviceMirror
     * scheme bare `All.*` here is now safe (device-pass-only macro), but
     * the canonical out-of-line accessor is retained as explicit-intent
     * documentation. */
    const struct global_data_all_processes *HostAll = gizmo_host_all_ptr();
    if (HostAll->ComovingIntegrationOn) {
        if (P[i].Type != 4) return false;
    } else {
        if (P[i].Type < 2 || P[i].Type > 4) return false;
    }
    if (P[i].Mass <= 0 || !isfinite((double)P[i].Mass)) return false;
    if (P[i].KernelRadius <= 0) return false;
    if (P[i].DensityAroundParticle <= 0) return false;
    return true;
}

/* ============================================================================
 * HOST HOOKS
 * ========================================================================== */

double RadFBRPSpec::search_radius(const neighbor_loop_args& args,
                                   int active_slot, int /*i*/)
{
    /* Search radius is RtauMax, pre-computed by radfb_rp_local_fill and
     * carried in per-active LocalIn (which was populated host-side in
     * populate_device_context's memcpy from aux->host_locals). */
    Aux *aux = nlr_aux<RadFBRPSpec>(args);
    if (aux == nullptr || aux->host_locals == nullptr) return 0.0;
    return (double)aux->host_locals[active_slot].KernelRadius;
}

/* radfb_rp_build_call_scalars — SSOT for per-call cosmology + unit-conversion
 * struct. Reads All.* exclusively via the canonical nlr_host_all_ptr()
 * accessor (feedback_all_dev_trap_host_side rule). Pair kernel reads NO
 * All.* / UNIT_* macros directly — all such reads route through scalars.
 * (NOTE: rt_kappa, called from inside the pair body, retains its own
 * legacy All.* reach. That is NOT a regression — matches legacy
 * radfb_local_gpu.cc behavior.) */
RadFBRPCallScalars radfb_rp_build_call_scalars(void)
{
    const struct global_data_all_processes *h = nlr_host_all_ptr();
    RadFBRPCallScalars scalars;
    scalars.common = nlr_common_scalars_from_all();

    scalars.rp_renorm      = (double)h->RP_Local_Momentum_Renormalization;

    /* dv cap = 1e4 km/s in code units. Macro UNIT_VEL_IN_KMS expands to
     * (h->UnitVelocity_in_cm_per_s / 1.e5). Pre-evaluate host-side via *h. */
    const double unit_vel_in_kms = h->UnitVelocity_in_cm_per_s / 1.e5;
    scalars.dv_cap_codeunits = 1.0e4 / unit_vel_in_kms;

    scalars.rng_ti_counter = (uint64_t)h->Ti_Current;

#ifdef SINK_WIND_SPAWN
    scalars.spawned_wind_cell_id = h->SpawnedWindCellID;
#endif

    return scalars;
}

RadFBRPSpec::CallScalars
RadFBRPSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    return radfb_rp_build_call_scalars();
}

/* ============================================================================
 * DEVICE-CONTEXT LIFECYCLE
 *
 * populate allocates a UVM array of RadFBRPLocalIn[N] and copies in from
 * args.aux->host_locals. cleanup frees it. The runner separately calls
 * populate for drv.ctx AND drv.ctx_oracle — each gets its own UVM buffer
 * (oracle-safe).
 * ========================================================================== */

void RadFBRPSpec::populate_device_context(const neighbor_loop_args& args,
                                           DeviceContext& ctx)
{
    ctx.oracle_dry_run       = false;
    ctx.iter_index_snapshot  = 0;     /* will be refreshed by reset_per_iter on every iter */

    Aux *aux = nlr_aux<RadFBRPSpec>(args);
    const int N = args.num_active;
    if (N <= 0 || aux == nullptr || aux->host_locals == nullptr) {
        ctx.per_active_local = nullptr;
        return;
    }
    RadFBRPLocalIn *uvm = (RadFBRPLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(RadFBRPLocalIn));
    std::memcpy(uvm, aux->host_locals, N * sizeof(RadFBRPLocalIn));
    /* wt_sum field starts at 0 (carried through from radfb_rp_local_fill).
     * iter-0 device kernel accumulates into accum.wt_sum; iter-0 after_iter
     * stashes that into ctx.scratch.wt_sum; iter-0 after_iter_global bridges
     * scratch→loc.wt_sum before iter-1 dispatches. */
    ctx.per_active_local = uvm;
}

void RadFBRPSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                          DeviceContext& ctx)
{
    if (ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ctx.per_active_local);
        ctx.per_active_local = nullptr;
    }
}

/* ============================================================================
 * PER-ITER RESET — THE single source of truth for iter-gated writeback.
 *
 * Set BEFORE any rank can enter a detector/writeback hook (runner ordering
 * guarantee at mesh/neighbor_loop_runner.cc:4038-4058). Called separately
 * for drv.ctx and drv.ctx_oracle; each ctx gets its own iter_index_snapshot
 * mirror. Aux::iter_index is shared (one Aux per call); since reset for
 * production fires before oracle reset in the runner, the Aux value reflects
 * whichever ctx was reset most recently — but since iter_index is identical
 * for both ctxs at any given outer iter, this is consistent.
 *
 * NO data staging here — that happens in after_iter_global of the PREVIOUS
 * iter (which bridges from drv.scratch_uvm / drv.scratch_oracle_uvm into
 * the corresponding ctx's per_active_local).
 * ========================================================================== */

void RadFBRPSpec::reset_per_iter_device_context(
    const neighbor_loop_args_iterative& args,
    DeviceContext& ctx,
    int iter_index)
{
    ctx.iter_index_snapshot = iter_index;

    Aux *aux = nlr_aux<RadFBRPSpec>(static_cast<const neighbor_loop_args&>(args));
    if (aux != nullptr) {
        aux->iter_index = iter_index;
    }
}

/* ============================================================================
 * AFTER_ITER — writes ctx.scratch.wt_sum on iter 0 (the only allowed write —
 * scratch IS the runner's per-active iterative state; no P/CellP/source-side
 * writes; oracle-safe per mechfb r5+r6+r7).
 *
 * iter 0: stash accum.wt_sum into ctx.scratch.wt_sum (runner pass-specific:
 *         production after_iter writes drv.scratch_uvm[sg][slot]; oracle
 *         after_iter writes drv.scratch_oracle_uvm[sg][slot]; both bridged
 *         into their respective ctx.per_active_local in after_iter_global).
 *         Return NeedsMore if wt_sum > 0; Converged otherwise (source had
 *         no valid neighbors → skip iter 1).
 * iter 1: Converged (always; no third iter).
 * ========================================================================== */

IterResult RadFBRPSpec::after_iter(const AfterIterContext<RadFBRPSpec>& ctx,
                                    const AccumData& accum)
{
    if (ctx.iter_index == 0) {
        ctx.scratch.wt_sum = accum.wt_sum;
        if (accum.wt_sum <= 0) {
            return IterResult{IterStatus::Converged, ctx.h_search_current};
        }
        return IterResult{IterStatus::NeedsMore, ctx.h_search_current};
    }
    /* iter >= 1 — kicks applied; we're done. */
    return IterResult{IterStatus::Converged, ctx.h_search_current};
}

/* ============================================================================
 * AFTER_ITER_GLOBAL — post-iter staging hook. NO physics-side writes.
 *
 * Mutates per_active_local UVM buffers on both drv.ctx and drv.ctx_oracle
 * (the runner allocates them independently). Does NOT touch
 * P[i], CellP[i], or any source/neighbor physics state — this is pure
 * iter-to-iter scalar bridging through the runner's intended per-active
 * iterative state (IterScratch → per_active_local).
 *
 * Fires once per iter on production (post both after_iter loops + compare;
 * mesh/neighbor_loop_runner.cc:4314-4316). For iter 0 only, copies
 * IterScratch.wt_sum (set by after_iter — production: drv.scratch_uvm[sg];
 * oracle: drv.scratch_oracle_uvm[sg]) into the corresponding ctx's
 * per_active_local[slot].wt_sum so iter-1's device kernel reads the staged
 * denominator.
 *
 * Reading from scratch_uvm (the runner's intended per-active iter-state)
 * rather than directly from accum_uvm is the more stable contract — if the
 * runner later changes when/whether accum_uvm is zeroed between hooks,
 * this hook still works. accum.wt_sum was already routed into scratch by
 * after_iter (which is the hook with both accum AND scratch in scope).
 *
 * drv is const-ref. per_active_local-pointee mutation is legal through the
 * const pointer (pointer itself is fixed; pointed-to UVM buffer writable).
 * Same idiom as mechfb_loop.cc::after_iter_global.
 *
 * Kokkos::fence() AFTER host writes — required for SharedSpace/UVM ordering
 * before iter-1's device dispatch reads the staged values.
 *
 * No-op for iter >= 1 (no iter-2 to stage for).
 * ========================================================================== */

void RadFBRPSpec::after_iter_global(const neighbor_loop_args& args,
                                     const NlrIterDriver<RadFBRPSpec>& drv)
{
    (void)args;
    if (drv.iter_index != 0) return;
    if (drv.args.num_subgroups < 1) return;
    const int sg = 0;
    const NlrSubgroup& subgroup = drv.args.subgroups[sg];
    const int N = subgroup.num_active_local;
    if (N <= 0) return;

    bool wrote_any = false;

    /* Bridge production: drv.scratch_uvm[sg][slot].wt_sum
     *                 → drv.ctx.per_active_local[slot].wt_sum */
    if (drv.ctx.per_active_local != nullptr
        && (int)drv.scratch_uvm.size() > sg
        && drv.scratch_uvm[sg] != nullptr) {
        for (int slot = 0; slot < N; slot++) {
            drv.ctx.per_active_local[slot].wt_sum =
                drv.scratch_uvm[sg][slot].wt_sum;
        }
        wrote_any = true;
    }

    /* Bridge oracle: drv.scratch_oracle_uvm[sg][slot].wt_sum
     *             → drv.ctx_oracle.per_active_local[slot].wt_sum.
     * Oracle context has its own scratch trajectory (drv.scratch_oracle_uvm)
     * and its own per_active_local — bridging must be done independently. */
    if (drv.ctx_oracle_initialized
        && drv.ctx_oracle.per_active_local != nullptr
        && (int)drv.scratch_oracle_uvm.size() > sg
        && drv.scratch_oracle_uvm[sg] != nullptr) {
        for (int slot = 0; slot < N; slot++) {
            drv.ctx_oracle.per_active_local[slot].wt_sum =
                drv.scratch_oracle_uvm[sg][slot].wt_sum;
        }
        wrote_any = true;
    }

    /* Fence host writes to UVM before iter-1's device dispatch reads them. */
    if (wrote_any) Kokkos::fence();
}

/* ============================================================================
 * SOURCE-SIDE HOST WRITEBACK
 *
 * Per-active post-runner. Subtracts accum.jet_momentum_used from
 * P_host[i].NewStar_Momentum_For_JetFeedback under
 * GALSF_FB_FIRE_STELLAREVOLUTION > 2. Mirrors legacy radfb_local_gpu.cc:
 * 351-358. No other source-side writes — the legacy CPU code's only
 * post-kernel source mutation is this jet budget decrement.
 *
 * Reads accum.jet_momentum_used; needs no Aux for this op (i is the
 * particle index, P_host[i].NewStar_Momentum_For_JetFeedback is the
 * canonical target). active_slot unused.
 * ========================================================================== */

void RadFBRPSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                          int /*active_slot*/, int i,
                                          const AccumData& accum)
{
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    if (P[i].NewStar_Momentum_For_JetFeedback > 0
        && accum.jet_momentum_used > 0) {
        P[i].NewStar_Momentum_For_JetFeedback -=
            (MyDouble)accum.jet_momentum_used;
        if (P[i].NewStar_Momentum_For_JetFeedback < 0) {
            P[i].NewStar_Momentum_For_JetFeedback = 0;
        }
    }
#else
    (void)i;
    (void)accum;
#endif
}

/* ============================================================================
 * MERGE
 *
 * wt_sum            : additive across peer accumulators (iter-0 reduce).
 * jet_momentum_used : additive across peer accumulators (iter-1 reduce).
 * ========================================================================== */

void RadFBRPSpec::merge_accum(AccumData& local_accum, const AccumData& peer_accum)
{
#define ACCUM_ADD(field)  local_accum.field += peer_accum.field;
    ACCUM_ADD(wt_sum)
    ACCUM_ADD(jet_momentum_used)
#undef ACCUM_ADD
}

/* ============================================================================
 * HOST STOCHASTIC GATE (SSOT) — toplevel calls this per active source pre-
 * runner. Returns 1 if the source passes the gate (fills *loc + *radius);
 * 0 if it doesn't fire this step (skip — not added to runner active list).
 *
 * Ported verbatim from radfb_local_gpu.cc::radfb_rp_local_fill (legacy
 * `static` helper, now public). RNG draw inside the gate retains the
 * legacy single-rank ordering (`get_random_number(P[i].ID + ThisTask + i + 2)`)
 * — that's a source-level stochastic acceptance, NOT a per-pair RNG, so
 * the runner's order-independent per-pair RNG shift (RADFBRP_RNG_SHIFT in
 * radfb_rp_loop.h) doesn't apply here.
 * ========================================================================== */

int radfb_rp_local_fill(int i,
                         struct particle_data *P_host,
                         struct gas_cell_data *CellP_host,
                         struct RadFBRPLocalIn *loc,
                         double *src_radius_out)
{
    /* Host-only function in a GPU TU. Under the per-TU AllDeviceMirror
     * scheme the device-pass macro is host-pass-inert, so bare `All.*`
     * here would read the host extern correctly. The canonical
     * out-of-line accessor is retained as explicit-intent documentation
     * and to give a single named local for UNIT_* pre-evaluation. */
    const struct global_data_all_processes *HostAll = gizmo_host_all_ptr();
    const double unit_mass_in_cgs    = HostAll->UnitMass_in_g / HostAll->HubbleParam;
    const double unit_vel_in_cgs     = HostAll->UnitVelocity_in_cm_per_s;
    const double unit_length_in_cgs  = HostAll->UnitLength_in_cm / HostAll->HubbleParam;
    const double unit_time_in_cgs    = unit_length_in_cgs / unit_vel_in_cgs;
    const double unit_density_in_cgs = unit_mass_in_cgs
                                     / (unit_length_in_cgs * unit_length_in_cgs * unit_length_in_cgs);
    const double unit_length_in_kpc  = unit_length_in_cgs / 3.085678e21;
    const double unit_mass_in_solar  = unit_mass_in_cgs / SOLAR_MASS_CGS;
    const double unit_vel_in_kms     = unit_vel_in_cgs / 1.e5;
    const double unit_density_in_nhcgs = unit_density_in_cgs / PROTONMASS_CGS;

    /* type / mass / age guards — cosmo-aware */
    if (P_host[i].Type != 4) {
        if (!(!HostAll->ComovingIntegrationOn
              && (P_host[i].Type == 2 || P_host[i].Type == 3))) return 0;
    }
    if (P_host[i].Mass <= 0 || !isfinite((double)P_host[i].Mass)) return 0;
    if (P_host[i].DensityAroundParticle <= 0) return 0;

    double star_age = evaluate_stellar_age_Gyr(i);
    if (star_age >= radfb_rp_age_threshold_value()) return 0;

    double dt = get_particle_feedback_timestep_in_physical(i);
    if (dt <= 0 || !isfinite(dt)) return 0;

    /* luminosity */
    double lm_ssp    = evaluate_light_to_mass_ratio(star_age, i);
    double lum_cgs   = lm_ssp * SOLAR_LUM_CGS * (P_host[i].Mass * unit_mass_in_solar);
    double f_lum_ion = particle_ionizing_luminosity_in_cgs(i) / lum_cgs;
    f_lum_ion = DMAX(0., DMIN(1., f_lum_ion));

    double dE_over_c = HostAll->RP_Local_Momentum_Renormalization * lum_cgs
                     * (dt * unit_time_in_cgs) / C_LIGHT_CGS;
    dE_over_c /= (unit_mass_in_cgs * unit_vel_in_cgs);

    /* search radius */
    double rho_phys = P_host[i].DensityAroundParticle * HostAll->cf_a3inv;
    double h_phys   = P_host[i].KernelRadius * HostAll->cf_atime;
    double RtauMax  = P_host[i].KernelRadius
                    * (5.0 + 2.0 * rt_kappa(i, RT_FREQ_BIN_FIRE_UV, P_host, CellP_host)
                       * P_host[i].KernelRadius * P_host[i].DensityAroundParticle * HostAll->cf_a2inv);
    RtauMax = DMAX(1.0 / (unit_length_in_kpc * HostAll->cf_atime),
                   DMIN(10.0 / (unit_length_in_kpc * HostAll->cf_atime), RtauMax));

    /* stochastic gate */
    double v_wind_threshold = 15.0 / unit_vel_in_kms;
#if defined(SINGLE_STAR_SINK_DYNAMICS) && !defined(GALSF_FB_FIRE_STELLAREVOLUTION)
    v_wind_threshold = 0.2 / unit_vel_in_kms;
#endif
    double delta_v = v_wind_threshold;
    double tau_IR  = rt_kappa(i, RT_FREQ_BIN_FIRE_IR, P_host, CellP_host)
                   * rho_phys * h_phys;
    double dv_guess = (dE_over_c / P_host[i].Mass) * (1.0 + tau_IR);
    double prob = dv_guess / delta_v * 2000.0;

#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    double v_grav = DMIN(1.82 * (65.748 / unit_vel_in_kms)
                          * pow(1. + rho_phys * unit_density_in_nhcgs, -0.25),
                         sqrt(HostAll->G * (P_host[i].Mass
                                        + 4.189 * rho_phys * h_phys * h_phys * h_phys)
                              / h_phys));
    delta_v = 2.0 * DMAX(v_grav, v_wind_threshold);
    prob    = dv_guess / delta_v;

    double jet_mom = 0;
    if (P_host[i].Type == 4 && P_host[i].NewStar_Momentum_For_JetFeedback > 0) {
        prob    = 1.0;
        jet_mom = P_host[i].NewStar_Momentum_For_JetFeedback;
    }
    if (prob < 1.0 && prob > 0) dE_over_c /= prob;
#endif

    double p_random = get_random_number(P_host[i].ID + ThisTask + i + 2);
    if (p_random > prob) return 0;     /* doesn't fire this step */

    /* fill struct */
    loc->Pos          = P_host[i].Pos;
    loc->KernelRadius = (MyFloat)RtauMax;
    loc->dE_over_c    = (MyFloat)dE_over_c;
    loc->f_lum_ion    = (MyFloat)f_lum_ion;
    loc->ID           = P_host[i].ID;
    loc->wt_sum       = 0;                /* zero at iter-0 entry; filled by
                                           * after_iter_global before iter 1 */
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    loc->delta_v_imparted_rp = (MyFloat)delta_v;
#endif
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    loc->jet_momentum_tocouple = (MyFloat)jet_mom;
#endif

    *src_radius_out = RtauMax;
    return 1;
}

/* ============================================================================
 * GHOST-WRITEBACK MANIFEST
 *
 * j-side writes from the iter-1 pair kernel:
 *   P[j].Vel       — Vec3<MyDouble> additive (atomic_add at kernel)
 *   P[j].dp        — Vec3<MyDouble> additive (atomic_add at kernel)
 *   CellP[j].VelPred — Vec3<MyDouble> additive (atomic_add at kernel)
 *
 * Bundle uses generic ops:
 *   GHOST_WRITEBACK_PARTICLE_ADD_VEC3 — exists (thermalfb 3e.2)
 *   GHOST_WRITEBACK_GAS_ADD_VEC3      — NEW for radfb_local (added to
 *                                        mesh/ghost_writeback_ops.h in this
 *                                        port). Discipline-rule trio:
 *                                          - legacy line: mesh/ghost_writeback.cc:1118
 *                                          - manifest line: below
 *                                          - validation: MA-N rank-0 records
 *                                            print (Ghost writeback (radfbrp): N).
 * ========================================================================== */

GHOST_WRITEBACK_BUNDLE_BEGIN(radfbrp)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(Vel)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(dp)
    GHOST_WRITEBACK_GAS_ADD_VEC3(VelPred)
GHOST_WRITEBACK_BUNDLE_END(radfbrp)

/* ============================================================================
 * GHOST-WRITEBACK HOOK BODIES — iter-gated.
 *
 * All four read Aux::iter_index (THE single source of truth, set in
 * reset_per_iter_device_context) and early-return on iter_index != 1.
 * Iter 0 has no j-side writes by design (pair kernel only accumulates
 * wt_sum into AccumData); skipping detector + writeback on iter 0 is
 * correct, not just optimization. Decision is uniform on every rank
 * (Aux::iter_index set deterministically by the runner per iter on every
 * rank); collective-symmetric.
 * ========================================================================== */

void RadFBRPSpec::ghost_write_detector_begin(const neighbor_loop_args& args,
                                              const NeighborLoopPlan& /*plan*/)
{
    Aux *aux = nlr_aux<RadFBRPSpec>(args);
    if (aux == nullptr || aux->iter_index != 1) return;
    ::ghost_write_detector_begin("radfbrp");
}

void RadFBRPSpec::ghost_write_detector_end(const neighbor_loop_args& args,
                                            const NeighborLoopPlan& /*plan*/)
{
    Aux *aux = nlr_aux<RadFBRPSpec>(args);
    if (aux == nullptr || aux->iter_index != 1) return;
    ::ghost_write_detector_end();
}

void RadFBRPSpec::ghost_writeback_begin(const neighbor_loop_args& args,
                                         const NeighborLoopPlan& /*plan*/)
{
    Aux *aux = nlr_aux<RadFBRPSpec>(args);
    if (aux == nullptr || aux->iter_index != 1) return;
    ghost_writeback_begin_bundle(radfbrp_ghost_writeback_bundle_ptr());
}

void RadFBRPSpec::ghost_writeback_end(const neighbor_loop_args& args,
                                       const NeighborLoopPlan& /*plan*/)
{
    Aux *aux = nlr_aux<RadFBRPSpec>(args);
    if (aux == nullptr || aux->iter_index != 1) return;
    ghost_writeback_end_bundle(radfbrp_ghost_writeback_bundle_ptr());
}

/* ============================================================================
 * DIAGNOSTICS — env-gated.
 * compare_accum: per-iter byte-walk. Iter 0 cares about wt_sum; iter 1
 * cares about jet_momentum_used. Both summed into the relative residual
 * with a denom-floor of 1.0 (mirrors thermal_fb pattern).
 * ========================================================================== */

/* ============================================================================
 * TOPLEVEL CALLER — radiation_pressure_winds_consolidated
 *
 * Lives HERE (not in radfb_local.cc) so the RP runner-template wire-up stays
 * in the same two-file Spec module (header + .cc). radfb_local.cc stays
 * host-only (STARFORM_OBJS) and keeps the unrelated legacy
 * HII_heating_singledomain + do_the_local_ionization in their own host
 * TU. accel.cc calls
 * `radiation_pressure_winds_consolidated()` via its forward decl in
 * core/proto.h; the linker resolves it here.
 *
 * Replaces the legacy `radfb_local.cc::radiation_pressure_winds_gpu` thin
 * wrapper. Active Type-4 (cosmo-aware: + Types 2/3 non-cosmo) stars scatter
 * UV / IR / jet radiation-pressure kicks into surrounding gas neighbors.
 * Iterative 2-pass inside the runner: iter 0 accumulates Σ h_j² (the kick-
 * normalization denominator) into AccumData; iter 1 applies kicks with the
 * staged denominator. Ghost-writeback bundle (generic Vec3 ops, gated to
 * iter 1 by Aux::iter_index) propagates j-side Vel / VelPred / dp deltas
 * to home ranks in Mode A.
 *
 * Active-list ownership: the stochastic gate (radfb_rp_local_fill above)
 * produces per-source LocalIn + search-radius payload, so we CANNOT use
 * nlr_build_active_list(RadFBRPSpec::is_active, ...) — that would re-evaluate
 * the predicate over un-gated ActiveParticleList and re-include non-firing
 * sources. The toplevel here owns the active-list build: filter +
 * radfb_rp_local_fill per source; pre-built active_src / host_locals arrays
 * handed to the runner via Aux + neighbor_loop_args_iterative.
 * RadFBRPSpec::is_active (above) is a defensive Type re-check only.
 * ========================================================================== */

void radiation_pressure_winds_consolidated(void)
{
    /* Build the gated active list with per-source LocalIn. Mirrors legacy
     * radfb_local_gpu.cc:162-182 (deterministic stochastic gate; only firing
     * sources flow into ActiveData). Search radius is SSOT in
     * RadFBRPLocalIn::KernelRadius (= RtauMax), read by
     * RadFBRPSpec::search_radius via aux->host_locals — no parallel radii
     * vector. radfb_rp_local_fill's `src_radius_out` parameter is retained
     * for API symmetry with the legacy entry point but its return slot is
     * discarded. */
    std::vector<int>             active_src;
    std::vector<RadFBRPLocalIn>  host_locals;
    {
        /* Canonical host-extern accessor; explicit-intent local for the
         * host-side reads below. */
        const struct global_data_all_processes *HostAll = gizmo_host_all_ptr();
        if (HostAll->RP_Local_Momentum_Renormalization > 0)
        {
            const int num_active_global_in = (int)ActiveParticleList.size();
            active_src.reserve(num_active_global_in);
            host_locals.reserve(num_active_global_in);
            for (int a = 0; a < num_active_global_in; a++) {
                int i = ActiveParticleList[a];
                RadFBRPLocalIn loc;
                double radius_discard;
                if (radfb_rp_local_fill(i, P, CellP, &loc, &radius_discard)) {
                    active_src.push_back(i);
                    host_locals.push_back(loc);
                }
            }
        }
    }
    const int num_active = (int)active_src.size();

    /* Collective-symmetry: every rank must enter the runner so MPI
     * collectives inside (ghost_exchange, Allreduce, Alltoallv) stay in
     * lockstep. The runner accepts num_active=0 as a no-op participant.
     * Short-circuit ONLY if globally zero firing sources. */
    int global_num_active = 0;
    MPI_Allreduce(&num_active, &global_num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_num_active == 0) return;

    /* Aux carries non-owning pointers to the toplevel-owned active arrays.
     * iter_index is set by reset_per_iter_device_context on every iter.
     * host_locals is the source pack the runner memcpys into UVM
     * (populate_device_context). host_active_src reserved for parallelism
     * with thermalfb/mechfb Aux patterns. */
    RadFBRPSpec::Aux aux{};
    aux.iter_index      = -1;            /* runner refreshes before any hook */
    aux.host_locals     = host_locals.data();
    aux.host_active_src = active_src.data();

    /* Single gas-only subgroup (one j_type_bitmask = (1u<<0)). May have
     * num_active_local == 0 on this rank — runner is fine, collective-only
     * entry. Pattern matches mechfb_loop.cc::mechfb_run_iterative. */
    NlrSubgroup sg{};
    sg.j_type_bitmask   = (1u << 0);
    sg.active_indices   = active_src.data();
    sg.num_active_local = num_active;

    neighbor_loop_args_iterative args{};
    static_cast<neighbor_loop_args&>(args) = nlr_default_args();
    args.P                   = P;
    args.CellP               = CellP;
    args.num_total           = NumPart;
    args.active_list         = active_src.data();
    args.num_active          = num_active;
    args.aux                 = &aux;
    args.num_subgroups       = 1;
    args.subgroups           = &sg;
    args.ghost_safety_factor = gizmo_ghost_safety_factor();

    run_neighbor_loop_iterative<RadFBRPSpec>(args);

    /* host_locals + active_src are stack-local std::vectors; their lifetime
     * extends through the runner call. apply_active_writeback reads
     * accum.jet_momentum_used per active (post-runner internal pass) and
     * mutates P_host[i].NewStar_Momentum_For_JetFeedback — reads no host
     * buffer. */
}

#else  /* !GALSF_FB_FIRE_RT_LOCALRP */


#endif /* GALSF_FB_FIRE_RT_LOCALRP */
