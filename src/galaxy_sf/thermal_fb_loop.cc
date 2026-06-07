/* galaxy_sf/thermal_fb_loop.cc — host-only hooks for ThermalFBSpec.
 *
 * Inline pair body (thermal_fb_pair_kernel), structs, and KOKKOS_INLINE_FUNCTION
 * hooks live in thermal_fb_loop.h so they inline from device kernels (Mode A)
 * and host walkers (Mode B / Brute oracle). This translation unit holds
 * host-only hooks: is_active, per-active radius, per-call scalars capture
 * (NlrCommonScalars + host-precomputed unit factors via nlr_host_all_ptr),
 * populate/cleanup_device_context (Phase 4.A.0 UVM staging),
 * apply_active_writeback (source-side mass + momentum loss), merge_accum,
 * ghost-writeback manifest + lifecycle hooks, compare_accum, and
 * thermal_fb_local_fill (the per-source host pack — SSOT).
 *
 * Phase 4 / Wave 3 / 3e.2. Written by Phil Hopkins (phopkins@caltech.edu)
 * for GIZMO.
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
#include "../declarations/constants.h"           /* UNIT_*_IN_* macros (host-side use here only) */
#include "../core/proto.h"
#include "../mesh/kernel.h"                       /* MUST precede thermal_fb_loop.h (no include guards) */
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_writeback_ops.h"
#include "../system/gpu_particles_arena.h"  /* gpu_particles_arena_invalidate -- needed for ghost pre-zero pattern */
#include "thermal_fb_loop.h"

#ifdef GALSF_FB_THERMAL

/* particle2in_addFB_fromstars(in, i, fb_loop_iteration) — declared in
 * core/proto.h via stellar_evolution.h. fb_loop_iteration=0 selects the
 * thermal-feedback ejecta branch. */

/* ============================================================================
 * ACTIVE PREDICATE
 * Mirrors addthermalFB_evaluate_active_check in thermal_fb.cc:39-46. SSOT
 * note: when the toplevel filter switches to runner-template, that legacy
 * predicate becomes the only consumer of these conditions; kept inline here
 * to avoid the cross-TU dependency.
 * ========================================================================== */

bool ThermalFBSpec::is_active(int i)
{
    /* Cosmology-aware "is-stellar" gate — mirrors mechanical_fb.cc:75-82.
     * Pre-port thermal_fb.cc gated strictly on Type==4 regardless of cosmo
     * mode, which is a long-standing latent inconsistency with the mechfb
     * convention: in non-cosmological sims, Types 2/3/4 are all valid stars.
     * Fix landed in 3e.2 per directive 5 (Phil confirmed 2026-05-15). */
    if (All.ComovingIntegrationOn) {
        if (P[i].Type != 4) return false;
    } else {
        if (P[i].Type < 2 || P[i].Type > 4) return false;
    }
    if (P[i].KernelRadius <= 0)     return false;
    if (P[i].NumNgb <= 0)           return false;
    if (P[i].SNe_ThisTimeStep <= 0) return false;
    return true;
}

/* ============================================================================
 * HOST HOOKS
 * ========================================================================== */

double ThermalFBSpec::search_radius(const neighbor_loop_args& args,
                                     int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

/* thermal_fb_build_call_scalars — SSOT for the per-call cosmology + unit-
 * conversion struct. Routes `All.*` reads through nlr_host_all_ptr() as
 * stylistic intent-tagging for host-snapshot semantics (under 93897f62 the
 * device-pass redirect is gated so bare host reads are
 * also correct). The kernel reads NO `All.*` and NO `UNIT_*` macros — all
 * such reads route through scalars. SSOT — single owner is
 * Spec::populate_call_scalars. */
ThermalFBCallScalars thermal_fb_build_call_scalars(void)
{
    const struct global_data_all_processes *h = nlr_host_all_ptr();
    ThermalFBCallScalars scalars;
    scalars.common = nlr_common_scalars_from_all();

    /* Host-precompute the four unit-conversion factors. Math is the
     * expansion of the constants.h macros with `All` → `*h`. Per the
     * macro chain:
     *   UNIT_MASS_IN_CGS   = h->UnitMass_in_g / h->HubbleParam
     *   UNIT_VEL_IN_CGS    = h->UnitVelocity_in_cm_per_s
     *   UNIT_LENGTH_IN_CGS = h->UnitLength_in_cm / h->HubbleParam
     *   UNIT_TIME_IN_CGS   = UNIT_LENGTH_IN_CGS / UNIT_VEL_IN_CGS
     *   UNIT_ENERGY_IN_CGS = UNIT_MASS_IN_CGS * UNIT_VEL_IN_CGS^2
     *   UNIT_DENSITY_IN_CGS = UNIT_MASS_IN_CGS / UNIT_LENGTH_IN_CGS^3
     * Constants:
     *   PROTONMASS_CGS, BOLTZMANN_CGS, SECONDS_PER_YEAR  (declarations/constants.h). */
    const double unit_mass_in_cgs   = h->UnitMass_in_g / h->HubbleParam;
    const double unit_vel_in_cgs    = h->UnitVelocity_in_cm_per_s;
    const double unit_length_in_cgs = h->UnitLength_in_cm / h->HubbleParam;
    const double unit_time_in_cgs   = unit_length_in_cgs / unit_vel_in_cgs;
    const double unit_density_in_cgs = unit_mass_in_cgs
                                     / (unit_length_in_cgs * unit_length_in_cgs * unit_length_in_cgs);

    scalars.unit_energy_in_cgs   = unit_mass_in_cgs * unit_vel_in_cgs * unit_vel_in_cgs;
    scalars.unit_density_in_nhcgs = unit_density_in_cgs / PROTONMASS_CGS;
    scalars.u_to_temp_units      = (PROTONMASS_CGS / BOLTZMANN_CGS)
                                 * (scalars.unit_energy_in_cgs / unit_mass_in_cgs);
    scalars.unit_time_in_myr     = unit_time_in_cgs / (1.e6 * SECONDS_PER_YEAR);

    return scalars;
}

/* populate_call_scalars — Spec hook. Forwards to the SSOT helper above. */
ThermalFBSpec::CallScalars
ThermalFBSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    return thermal_fb_build_call_scalars();
}

/* ============================================================================
 * PHASE 4.A.0 DEVICE-CONTEXT LIFECYCLE
 *
 * populate allocates a UVM array of ThermalFBLocalIn[N] and copies in from
 * args.aux->host_locals. cleanup frees it. Mirrors SinkFeedSpec exactly.
 * ========================================================================== */

void ThermalFBSpec::populate_device_context(const neighbor_loop_args& args,
                                             DeviceContext& ctx)
{
    ctx.oracle_dry_run = false;

    Aux *aux = nlr_aux<ThermalFBSpec>(args);
    const int N = args.num_active;
    if (N <= 0 || aux == nullptr || aux->host_locals == nullptr) {
        ctx.per_active_local = nullptr;
        return;
    }
    ThermalFBLocalIn *uvm = (ThermalFBLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(ThermalFBLocalIn));
    std::memcpy(uvm, aux->host_locals, N * sizeof(ThermalFBLocalIn));
    ctx.per_active_local = uvm;
}

void ThermalFBSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                            DeviceContext& ctx)
{
    if (ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<ThermalFBLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
}

/* ============================================================================
 * SOURCE-SIDE HOST WRITEBACK
 *
 * Per-active post-runner. Mirrors the legacy thermal_fb_gpu.cc:236-246:
 *   if (M_coupled > 0 && P[i].Mass > 0):
 *     dp -= M_coupled * Vel
 *     Mass -= M_coupled
 *     clamp negative / NaN to zero
 * Reads Vel from the host-fill pack via args.aux->host_locals (avoid extra
 * P_host indirection on the post-runner per-active loop).
 * ========================================================================== */

void ThermalFBSpec::apply_active_writeback(const neighbor_loop_args& args,
                                            int active_slot, int i,
                                            const AccumData& accum)
{
    const double M_coupled = (double)accum.M_coupled;
    if (M_coupled <= 0)        return;
    if (P[i].Mass <= 0)        return;

    Aux *aux = nlr_aux<ThermalFBSpec>(args);
    const ThermalFBLocalIn& loc = aux->host_locals[active_slot];

    /* All arithmetic stays in MyDouble — Vel and Mass are both declared
     * MyDouble in particle_data, so no narrowing cast (the legacy did the
     * same; codex review 2026-05-15 caught a MyFloat narrowing here). */
    for (int k = 0; k < 3; k++) {
        P[i].dp[k] -= (MyDouble)M_coupled * loc.Vel[k];
    }
    P[i].Mass -= (MyDouble)M_coupled;
    if (P[i].Mass < 0 || P[i].Mass != P[i].Mass) P[i].Mass = 0;
}

/* ============================================================================
 * MERGE
 * Single additive scalar (M_coupled). One line under the macro discipline.
 * ========================================================================== */

void ThermalFBSpec::merge_accum(AccumData& local_accum, const AccumData& peer_accum)
{
#define ACCUM_ADD(field)  local_accum.field += peer_accum.field;
    ACCUM_ADD(M_coupled)
#undef ACCUM_ADD
}

/* ============================================================================
 * HOST-FILL HELPER (SSOT)
 *
 * Same logic as the static `thermal_fb_local_fill` formerly in
 * thermal_fb_gpu.cc:38-60. Now public so the runner-template toplevel
 * (thermal_fb.cc::thermal_fb_calc) AND the legacy `thermal_fb_gpu.cc`
 * call ONE function. Cleanup commit retires `thermal_fb_gpu.cc` and this
 * symbol becomes consumed only by populate_call helpers in this TU.
 * ========================================================================== */

void thermal_fb_local_fill(int i,
                            struct particle_data *P_host,
                            struct gas_cell_data *CellP_host,
                            struct ThermalFBLocalIn *loc)
{
    (void)CellP_host;
    loc->Pos          = P_host[i].Pos;
    loc->Vel          = P_host[i].Vel;
    loc->KernelRadius = P_host[i].KernelRadius;
    loc->wt_sum       = P_host[i].DensityAroundParticle;
    loc->TimeBin      = P_host[i].TimeBin;  /* source TimeBin for receiver wakeup-mark */

    struct addFB_evaluate_data_in_ fb;
    particle2in_addFB_fromstars(&fb, i, 0);
    loc->Msne = fb.Msne;
    loc->Esne = (MyFloat)(0.5 * fb.Msne * fb.SNe_v_ejecta * fb.SNe_v_ejecta);

    double kz = 0, dwk_dummy = 0;
    kernel_main(0.0, 1.0, 1.0, &kz, &dwk_dummy, -1);
    loc->kernel_zero = (MyFloat)kz;

#ifdef METALS
    for (int k = 0; k < NUM_METAL_SPECIES; k++) {
        loc->yields[k] = (MyFloat)fb.yields[k];
    }
#endif
}

/* ============================================================================
 * GHOST-WRITEBACK MANIFEST
 *
 * Adding a new ghost-written field for this loop = ADDING ONE LINE under its
 * physics flag's #ifdef. The scaffold generates snapshot/diff/pack/exchange/
 * apply/cleanup from each manifest line. See mesh/ghost_writeback_ops.h.
 *
 * Coverage matches the legacy ghost_writeback_thermalfb in
 * mesh/ghost_writeback.cc (which the cleanup commit retires):
 *   P[j].Mass                       (always)
 *   P[j].dp                         (always, Vec3<MyDouble>)
 *   CellP[j].Density                (always)
 *   CellP[j].InternalEnergy         (always)
 *   CellP[j].InternalEnergyPred     (always)
 *   CellP[j].MassTrue               (HYDRO_MESHLESS_FINITE_VOLUME)
 *   P[j].Metallicity[NUM_METAL_SPECIES] (METALS)
 *   CellP[j].DelayTimeCoolingSNe    (GALSF_FB_TURNOFF_COOLING)
 *
 * Physics-correctness fix vs legacy: DelayTimeCoolingSNe uses the new
 * GHOST_WRITEBACK_GAS_MAX op (cross-rank max-reduce) to match the device-side
 * atomic_max semantics. The legacy ghost_writeback_thermalfb path applied
 * snapshot-diff additive on this field, which over-shoots when multiple
 * remote ranks each produce post>snap. See OPEN_3d_thermalfb_design.md.
 * ========================================================================== */

GHOST_WRITEBACK_BUNDLE_BEGIN(thermalfb)
    GHOST_WRITEBACK_PARTICLE_ADD(Mass)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(dp)
    GHOST_WRITEBACK_GAS_ADD(Density)
    GHOST_WRITEBACK_GAS_ADD(InternalEnergy)
    GHOST_WRITEBACK_GAS_ADD(InternalEnergyPred)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    GHOST_WRITEBACK_GAS_ADD(MassTrue)
#endif
#ifdef METALS
    GHOST_WRITEBACK_PARTICLE_ADD_ARRAY(Metallicity, NUM_METAL_SPECIES)
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
    GHOST_WRITEBACK_GAS_MAX(DelayTimeCoolingSNe)
#endif
    /* wakeup written via atomic_max(Pj.wakeup, source.TimeBin+1) in
     * thermal_fb_pair_kernel. MAX-reverse-comm semantics match the device
     * side. Pre-zero of imported ghost wakeup happens in ghost_writeback_begin
     * below (same pattern as gravity/ags_force_loop.cc:317-339 -- without
     * pre-zero, an imported ghost wakeup >= the kernel's post-write would
     * silently fail to ship home). */
    GHOST_WRITEBACK_PARTICLE_MAX(wakeup)
GHOST_WRITEBACK_BUNDLE_END(thermalfb)

/* ghost_write_detector_begin/end: runner default (loop_name = "thermalfb"). */

void ThermalFBSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                            const NeighborLoopPlan& /*plan*/)
{
    /* Pre-zero ghost P[j].wakeup BEFORE the bundle snapshot. Same pattern as
     * gravity/ags_force_loop.cc:317-339. Without this, an imported ghost that
     * arrived with a wakeup already >= what the kernel writes would fail to
     * trip the MAX op's "post > snap" predicate, and the kernel write would
     * not ship home. Pre-zeroing forces every kernel-side wakeup write on a
     * ghost to be visible to the MAX reverse-comm. */
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    for(int g = 0; g < num_ghosts; g++) {
        P[num_local + g].wakeup = 0;
    }
    /* Host mutated ghost particles; force the next arena acquire to re-stage P. */
    gpu_particles_arena_invalidate();
    ghost_writeback_begin_bundle(thermalfb_ghost_writeback_bundle_ptr());
}

void ThermalFBSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                          const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(thermalfb_ghost_writeback_bundle_ptr());
}

/* ============================================================================
 * DIAGNOSTICS — env-gated.
 * compare_accum: byte-walk of M_coupled (oracle gate; GIZMO_NLR_ORACLE=1).
 * ========================================================================== */

double ThermalFBSpec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    const double va = (double)local.M_coupled;
    const double vb = (double)oracle.M_coupled;
    /* Denom floor = max(1, |a|, |b|) keeps relative comparisons stable when
     * M_coupled is tiny (e.g. early-step transient): avoids amplifying
     * floating-point noise on small denominators (codex review 2026-05-15). */
    const double denom = std::fmax(1.0, std::fmax(std::fabs(va), std::fabs(vb)));
    return std::fabs(va - vb) / denom;
}

#else  /* !GALSF_FB_THERMAL */


#endif /* GALSF_FB_THERMAL */
