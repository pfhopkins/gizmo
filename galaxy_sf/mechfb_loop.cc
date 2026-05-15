/* mechfb_loop.cc — MechFBSpec method bodies + toplevel-facing helpers
 * for the runner-template port of mechanical_fb_evaluate_gpu
 * (Phase 4 / Wave 3 / 3e.1).
 *
 * Milestone 3 (physics-complete): full Spec contract — real host methods,
 * oracle/state-machine plumbing, ghost_writeback bundle; real mech_fb_local_fill
 * / mech_fb_apply_aws_out / mech_fb_apply_source_mass_out (promoted from
 * static in mechanical_fb_gpu.cc, now SSOT shared by both paths);
 * real reset_per_iter_device_context (repack + mode-machine advance).
 *
 * SSOT design: ../OPEN_3d_mechfb_design.md (v0.4-final, codex-approved).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"                    /* kernel_main / kernel_hinv referenced inline by mechanical_fb_functions.h */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"   /* gizmo_ghost_safety_factor() */

#include "mechfb_loop.h"

#ifdef GALSF_FB_MECHANICAL

/* Host-side superset active-check predicate (mirrors the toplevel filter
 * loop in mechanical_fb.cc:248-251). Declared in mechanical_fb.cc at file
 * scope; forward-declared here. */
int addFB_evaluate_active_check(int i, int fb_loop_iteration);

/* ============================================================================
 * mechfb_compute_num_modes — picks 3/4/5/6 modes per FIRE config.
 * Mirrors the per-mode dispatch count in mechanical_fb_gpu.cc:277-287.
 * ========================================================================== */
int mechfb_compute_num_modes(void) {
    int num_modes = 3;
#if defined(GALSF_FB_FIRE_STELLAREVOLUTION)
    num_modes = 4;
#if defined(GALSF_FB_FIRE_RPROCESS)
    num_modes = 5;
#endif
#if defined(GALSF_FB_FIRE_AGE_TRACERS)
    num_modes = 6;
#endif
#endif
    return num_modes;
}

/* ============================================================================
 * mechfb_populate_aux_initial — fill Aux from caller-provided pointers.
 * Called from mechanical_fb_calc_toplevel before run_neighbor_loop_iterative.
 * ========================================================================== */
void mechfb_populate_aux_initial(MechFBSpec::Aux& aux,
                                 int num_active,
                                 struct MechFBGasDelta *LocalGasMechFBInfoTemp,
                                 int num_local_gas) {
    aux.num_modes = mechfb_compute_num_modes();
    const int modes_canonical[6] = {-2, -1, 0, 1, 2, 3};
    for (int k = 0; k < 6; ++k) aux.modes[k] = modes_canonical[k];
    aux.mode_idx = 0;
    aux.host_locals_scratch.assign(num_active > 0 ? num_active : 0, MechFBLocalIn{});
    aux.LocalGasMechFBInfoTemp = LocalGasMechFBInfoTemp;
    aux.num_local_gas          = num_local_gas;
    aux.n_couplings_thistask   = 0;
}

/* ============================================================================
 * Public mech_fb_* helpers — promoted from `static` in mechanical_fb_gpu.cc
 * (milestone 3) so legacy + runner-port share a single source of truth.
 * ========================================================================== */

void mech_fb_local_fill(int i, int loop_iteration, struct MechFBLocalIn *loc) {
    struct addFB_evaluate_data_in_ fb;
    fb.Pos = P[i].Pos;
    fb.Vel = P[i].Vel;
    double heff = P[i].KernelRadius / (double)P[i].NumNgb;
    fb.V_i = heff * heff * heff;
    fb.KernelRadius = P[i].KernelRadius;
#ifdef METALS
    for (int k = 0; k < NUM_METAL_SPECIES; ++k) fb.yields[k] = 0.0;
#endif
    for (int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; ++k) fb.Area_weighted_sum[k] = P[i].Area_weighted_sum[k];
    fb.Msne = 0; fb.unit_mom_SNe = 0; fb.SNe_v_ejecta = 0;

    if ((P[i].DensityAroundParticle > 0) && (P[i].Mass > 0)) {
        if (loop_iteration < 0) {
            fb.Msne = P[i].Mass;
            fb.unit_mom_SNe = 1.0e-4;
            fb.SNe_v_ejecta = 1.0e-4;
        } else {
            particle2in_addFB_fromstars(&fb, i, loop_iteration);
            fb.unit_mom_SNe = fb.Msne * fb.SNe_v_ejecta;
        }
    }

    loc->Pos = fb.Pos;
    loc->Vel = fb.Vel;
    loc->Msne = fb.Msne;
    loc->KernelRadius = fb.KernelRadius;
    loc->V_i = fb.V_i;
    loc->SNe_v_ejecta = fb.SNe_v_ejecta;
    for (int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; ++k) loc->Area_weighted_sum[k] = fb.Area_weighted_sum[k];
#ifdef METALS
    for (int k = 0; k < NUM_METAL_SPECIES; ++k) loc->yields[k] = fb.yields[k];
#endif
}

void mech_fb_apply_aws_out(const struct MechFBOut *out, int i, int loop_iteration) {
    if (P[i].Mass <= 0) return;
    int kmin = 0, kmax = 7;
    if (loop_iteration == -1) { kmin = 7; kmax = AREA_WEIGHTED_SUM_ELEMENTS; }
    for (int k = kmin; k < kmax; ++k) P[i].Area_weighted_sum[k] = out->Area_weighted_sum[k];
}

void mech_fb_apply_source_mass_out(struct particle_data *P_arr,
                                    struct gas_cell_data *CellP_arr,
                                    int i, MyFloat M_coupled) {
    if (P_arr[i].Mass <= 0) return;
    for (int k = 0; k < 3; ++k) P_arr[i].dp[k] -= M_coupled * P_arr[i].Vel[k];
    P_arr[i].Mass -= M_coupled;
    if (P_arr[i].Mass < 0 || P_arr[i].Mass != P_arr[i].Mass) P_arr[i].Mass = 0;
    if (P_arr[i].Type == 0) CellP_arr[i].Mass = P_arr[i].Mass;
#ifdef SINGLE_STAR_FB_WINDS
    P_arr[i].Sink_Mass -= M_coupled;
    if (P_arr[i].Sink_Mass < 0 || P_arr[i].Sink_Mass != P_arr[i].Sink_Mass) P_arr[i].Sink_Mass = 0;
#endif
}

/* ============================================================================
 * mechfb_repack_per_active_local — host-side per-mode repack of MechFBLocalIn
 * into the SharedSpace per_active_local buffer. Called from
 * reset_per_iter_device_context at the start of every iter.
 *
 * Mirrors mechanical_fb_gpu.cc:294-296 (the per-mode `for(aa) mech_fb_local_fill`
 * inside the legacy mode loop). Reads HOST P[i] — which carries any
 * Area_weighted_sum / Mass updates the previous mode's after_iter_global
 * applied (codex r6 moved per-mode source-side host writes out of after_iter
 * to keep oracle's dual-walk from double-applying).
 * ========================================================================== */
void mechfb_repack_per_active_local(const neighbor_loop_args& args,
                                    MechFBSpec::Aux& aux,
                                    int loop_iteration) {
    const int N = args.num_active;
    if (N <= 0 || args.active_list == nullptr) return;
    if ((int)aux.host_locals_scratch.size() < N) aux.host_locals_scratch.resize(N);
    for (int k = 0; k < N; ++k) {
        const int i = args.active_list[k];
        mech_fb_local_fill(i, loop_iteration, &aux.host_locals_scratch[k]);
    }
}

/* ============================================================================
 * Spec method bodies
 * ========================================================================== */

/* is_active — host-side superset active check.
 *
 * CONTRACT (codex review 2026-05-14): addFB_evaluate_active_check(i, fb_iter)
 * with `fb_iter < 0` is the single canonical "any compiled mechfb mode is
 * active for this source" predicate. The body's per-event branches all
 * accept `fb_loop_iteration < 0` as a wildcard, so this returns true iff at
 * least one of the per-mode active flags would be set for some mode in the
 * compiled mode set. This is the SAME shortcut mechanical_fb.cc uses to
 * build the superset active list (line 253, 268) — keep both call sites in
 * sync; do not replace one with a per-mode loop without updating the other. */
bool MechFBSpec::is_active(int particle_index) {
    return addFB_evaluate_active_check(particle_index, -2) != 0;
}

/* search_radius — per-active radius. Mirrors mechanical_fb_gpu.cc:295's
 * `nl_radii[aa] = (double)P[ii].KernelRadius`. The runner uses the radius
 * to drive Mode A CSR construction + Mode B walk extents; mechfb is not
 * radius-iterative so this value is constant across iters. */
double MechFBSpec::search_radius(const neighbor_loop_args& args,
                                 int /*active_slot*/, int i) {
    return (double)args.P[i].KernelRadius;
}

/* populate_call_scalars — immutable per-call scalars (NlrCommonScalars + a
 * few mechfb-specific All.* snapshots). Reads `All` via nlr_host_all_ptr()
 * per the canonical-accessor rule (feedback_all_dev_trap_host_side.md). */
/* Shared builder for MechFBCallScalars (codex r6 fix). Used by
 *   - MechFBSpec::populate_call_scalars (runner-port path)
 *   - legacy mechanical_fb_evaluate_gpu inside mechanical_fb_gpu.cc
 * so both paths read cosmology / unit-factor / CR-rigidity values from the
 * SAME canonical host All via nlr_host_all_ptr() — no bare All.* inside the
 * kernel-callable helpers regardless of which dispatch path invoked them. */
void mechfb_fill_call_scalars(struct MechFBCallScalars *scalars) {
    if (!scalars) return;
    const struct global_data_all_processes *host_all = nlr_host_all_ptr();
    scalars->common = nlr_common_scalars_from_all();

    /* Precomputed unit factors — replace UNIT_*_IN_* macro reads (which
     * expand to bare All.* and would bind to the per-TU All_dev mirror
     * inside a GPU TU). Derived directly from host_all->Unit* / HubbleParam
     * via the same arithmetic as declarations/constants.h:115-137. */
    const double unit_length_in_cgs = host_all->UnitLength_in_cm / host_all->HubbleParam;
    const double unit_mass_in_cgs   = host_all->UnitMass_in_g    / host_all->HubbleParam;
    const double unit_vel_in_cgs    = host_all->UnitVelocity_in_cm_per_s;
    const double unit_density_in_cgs = unit_mass_in_cgs /
                                        (unit_length_in_cgs * unit_length_in_cgs * unit_length_in_cgs);
    scalars->unit_length_in_kpc    = unit_length_in_cgs / 3.085678e21;
    scalars->unit_density_in_NHcgs = unit_density_in_cgs / PROTONMASS_CGS;
    scalars->unit_energy_in_cgs    = unit_mass_in_cgs * unit_vel_in_cgs * unit_vel_in_cgs;
    scalars->unit_mass_in_solar    = unit_mass_in_cgs / SOLAR_MASS_CGS;
    scalars->unit_vel_in_kms       = unit_vel_in_cgs / 1.0e5;

#ifdef METALS
    scalars->SolarAbundances0 = host_all->SolarAbundances[0];
#endif
#if defined(COSMIC_RAY_FLUID) && defined(CR_DYNAMICAL_INJECTION_IN_SNE)
    scalars->CosmicRay_SNeFraction = host_all->CosmicRay_SNeFraction;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(GALSF_FB_FIRE_STELLAREVOLUTION) && defined(CRFLUID_EVOLVE_SPECTRUM)
    for (int k = 0; k < N_CR_PARTICLE_BINS; ++k) {
        scalars->CR_global_min_rigidity_in_bin   [k] = host_all->CR_global_min_rigidity_in_bin   [k];
        scalars->CR_global_max_rigidity_in_bin   [k] = host_all->CR_global_max_rigidity_in_bin   [k];
        scalars->CR_global_rigidity_at_bin_center[k] = host_all->CR_global_rigidity_at_bin_center[k];
    }
#endif
}

MechFBSpec::CallScalars MechFBSpec::populate_call_scalars(
        const neighbor_loop_args& /*args*/) {
    CallScalars scalars;
    mechfb_fill_call_scalars(&scalars);
    return scalars;
}

/* populate_device_context — allocate SharedSpace per_active_local + bind
 * Aux-owned LocalGasMechFBInfoTemp into ctx. d_gas_iter stays nullptr at
 * milestone 3 single-rank scope: Mode A multi-rank ghost-side writes are
 * guarded by pair_kernel's loud abort (design Appendix D + the abort in
 * mechfb_loop.h). Initial loop_iteration/mode_idx values are set here AND
 * overwritten by reset_per_iter_device_context BEFORE iter 0 — the
 * duplication is intentional belt-and-suspenders. */
void MechFBSpec::populate_device_context(const neighbor_loop_args& args,
                                          DeviceContext& ctx) {
    auto *aux = static_cast<Aux*>(args.aux);

    ctx.mode_idx        = aux ? aux->mode_idx : 0;
    ctx.loop_iteration  = (aux && aux->num_modes > 0) ? aux->modes[ctx.mode_idx] : -2;
    ctx.oracle_dry_run  = false;
    ctx.num_modes       = aux ? aux->num_modes : 3;
    ctx.num_local_gas   = aux ? aux->num_local_gas : 0;
    ctx.n_ghost_alloc   = 0;
    ctx.LocalGasMechFBInfoTemp = aux ? aux->LocalGasMechFBInfoTemp : nullptr;
    ctx.d_gas_iter      = nullptr;

    /* SharedSpace per_active_local — sized to num_active. Contents are filled
     * by reset_per_iter_device_context's mechfb_repack_per_active_local call
     * before every iter; zero-init here is defensive (covers the unlikely
     * case where iter 0 reads before reset fires). */
    const int N = args.num_active;
    if (N > 0) {
        MechFBLocalIn *uvm = (MechFBLocalIn *)
            Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(MechFBLocalIn));
        std::memset(uvm, 0, N * sizeof(MechFBLocalIn));
        ctx.per_active_local = uvm;
    } else {
        ctx.per_active_local = nullptr;
    }
}

/* cleanup_device_context — free SharedSpace per_active_local + d_gas_iter
 * (if Mode A ever allocated it). Idempotent on already-null pointers. */
void MechFBSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                         DeviceContext& ctx) {
    if (ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<MechFBLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
    if (ctx.d_gas_iter) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ctx.d_gas_iter);
        ctx.d_gas_iter   = nullptr;
        ctx.n_ghost_alloc = 0;
    }
    ctx.LocalGasMechFBInfoTemp = nullptr;
}

/* apply_active_writeback — NO-OP for MechFBSpec.
 * All source-side P[i] writes (mode -2/-1 Area_weighted_sum writeback;
 * modes >=0 source mass loss) happen in MechFBSpec::after_iter_global
 * (host-side, production-only, per-iter — codex r6 oracle-safety fix moved
 * them out of after_iter so the runner's dual production+oracle walk doesn't
 * double-apply). This hook stays declared because the Spec contract requires
 * it but the runner's final end-of-call writeback loop has nothing left to do. */
void MechFBSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                         int /*active_slot*/, int /*i*/,
                                         const AccumData& /*accum*/) {
    /* Intentionally empty — see banner. */
}

/* merge_accum — Mode B remote peer accumulator merge. M_coupled adds;
 * Area_weighted_sum[] adds per element. Mirrors the legacy device-side
 * accumulation in mechanical_fb_pair_kernel (myout.Area_weighted_sum[k] +=). */
void MechFBSpec::merge_accum(AccumData& local, const AccumData& peer) {
    local.M_coupled += peer.M_coupled;
    for (int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; ++k) {
        local.Area_weighted_sum[k] += peer.Area_weighted_sum[k];
    }
}

/* set_oracle_brute_pass — gate j-side atomic writes during oracle pass.
 * Mirrors sink_feed pattern: pair_kernel checks ctx.oracle_dry_run before
 * any atomic_add into LocalGasMechFBInfoTemp / d_gas_iter (i-side accum
 * still completes; oracle's separate accum_oracle_uvm receives it). */
void MechFBSpec::set_oracle_brute_pass(DeviceContext& ctx, bool on) {
    ctx.oracle_dry_run = on;
}

/* compare_accum — per-pair-active oracle comparison. Byte-walk-as-doubles
 * pattern (matches sink_feed_loop.cc and ags_density_loop.cc). The
 * AccumData layout (MyFloat M_coupled + MyFloat Area_weighted_sum[12])
 * happens to fall on float boundaries; we walk as float for parity with
 * the original storage. */
double MechFBSpec::compare_accum(const AccumData& local, const AccumData& oracle) {
    /* Codex r5 fix 2026-05-14: pure-relative denom amplifies noise near zero
     * (density port hit the same trap). Use an absolute floor so near-zero
     * fields don't produce spurious ORACLE MISMATCH lines:
     *   denom = max(1, |a|, |b|)
     * Combined with absolute-floor-1 the rel diff degenerates to absolute
     * diff when both values are < 1 in magnitude — appropriate since
     * AccumData fields (Area_weighted_sum components, M_coupled) are
     * dimensionless / kernel-normalized and below-O(1) values are noise. */
    double max_rel = 0.0;
    const MyFloat *pa = reinterpret_cast<const MyFloat*>(&local);
    const MyFloat *pb = reinterpret_cast<const MyFloat*>(&oracle);
    static_assert(sizeof(AccumData) % sizeof(MyFloat) == 0,
        "MechFBSpec::AccumData size must be MyFloat-aligned for byte-walk compare");
    const size_t n = sizeof(AccumData) / sizeof(MyFloat);
    for (size_t k = 0; k < n; ++k) {
        double va = (double)pa[k], vb = (double)pb[k];
        double denom = std::fmax(1.0, std::fmax(std::fabs(va), std::fabs(vb)));
        double rel   = std::fabs(va - vb) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

/* after_iter — STATUS-ONLY (codex r6 fix 2026-05-14).
 *
 * Earlier draft applied per-mode source-side host writes here, but the
 * iterative runner calls after_iter for BOTH the production accum
 * (runner.cc:4128) AND the oracle accum (runner.cc:4189) when oracle is on.
 * Any host mutation would double-apply under GIZMO_NLR_ORACLE=1 — exactly
 * the kind of "passes one path, corrupts another" trap the oracle is
 * supposed to catch. Source-side writes now live in after_iter_global
 * (production-only, runner.cc:4310-4313 — runs once per iter post-oracle).
 *
 * Mechfb is not radius-iterative; each iter corresponds to one fixed mode.
 * Convergence is purely a function of iter_index. h_search is unchanged
 * (IterScratch = NoIterScratch). */
IterResult MechFBSpec::after_iter(const AfterIterContext<MechFBSpec>& ctx,
                                   const AccumData& /*accum*/) {
    auto *aux = static_cast<Aux*>(ctx.args.aux);
    const int num_modes = aux ? aux->num_modes : mechfb_compute_num_modes();
    if (ctx.iter_index >= num_modes - 1) {
        return IterResult{IterStatus::Converged, ctx.h_search_current};
    }
    return IterResult{IterStatus::NeedsMore, ctx.h_search_current};
}

/* after_iter_global — production-only per-iter host hook.
 *
 * Runs ONCE per outer iter AT runner.cc:4310-4313, AFTER both the production
 * and oracle (if enabled) Spec::after_iter passes have completed AND after
 * the oracle 4-thing compare has run. Mutations here apply exactly once per
 * iter regardless of oracle state.
 *
 * Per-iter responsibilities:
 *   - For each subgroup-0 slot, read drv.accum_uvm[0][slot] (production
 *     accum; oracle accum is in a separate buffer and not consulted here).
 *   - mode -2: mech_fb_apply_aws_out  (writes P[i].Area_weighted_sum[0..6])
 *   - mode -1: mech_fb_apply_aws_out  (writes P[i].Area_weighted_sum[7..])
 *   - mode >=0 (gated on accum.M_coupled > 0): mech_fb_apply_source_mass_out
 *     + aux->n_couplings_thistask++
 *
 * Note: `active_set_size[sg]` shrinks after compaction (last iter all
 * Converged → size becomes 0). But accum_uvm[sg][slot] entries persist at
 * their original sg.num_active_local positions (runner.cc:4140-4141 comment),
 * so iterating by sg.num_active_local always reads the correct per-slot
 * accum even on the final iter. */
void MechFBSpec::after_iter_global(const neighbor_loop_args& args,
                                    const NlrIterDriver<MechFBSpec>& drv) {
    /* SFINAE detector requires the args parameter to be `const neighbor_loop_args&`
     * (the base type — runner.h:1584-1591). num_subgroups / subgroups live on
     * the derived neighbor_loop_args_iterative; reach them via drv.args. */
    auto *aux = static_cast<Aux*>(args.aux);
    if (!aux || drv.args.num_subgroups < 1) return;

    const int sg = 0;
    const NlrSubgroup &subgroup = drv.args.subgroups[sg];
    const int N = subgroup.num_active_local;
    if (N <= 0) return;

    const int iter_index = drv.iter_index;
    if (iter_index < 0 || iter_index >= aux->num_modes) return;
    const int mode = aux->modes[iter_index];

    if ((int)drv.accum_uvm.size() <= sg || drv.accum_uvm[sg] == nullptr) return;
    const AccumData *accum_arr = drv.accum_uvm[sg];

    for (int slot = 0; slot < N; ++slot) {
        const int i = subgroup.active_indices[slot];
        /* Per-mode active mask (codex r7 fix) — host-side equivalent of the
         * load_active is_active_this_mode flag the device kernel checks.
         * Without this, mode -2/-1 would overwrite P[i].Area_weighted_sum
         * with the (zero) accum of sources not active for this mode under
         * any future addFB_evaluate_active_check semantics where superset
         * membership doesn't imply mode -2/-1 activity. (Current semantics
         * happen to have all superset members active in modes -2/-1; the
         * guard is robust to future changes and exactly matches legacy.) */
        if (!addFB_evaluate_active_check(i, mode)) continue;

        const AccumData &accum = accum_arr[slot];
        if (mode < 0) {
            mech_fb_apply_aws_out(&accum, i, mode);
        } else if (accum.M_coupled > 0) {
            mech_fb_apply_source_mass_out(P, CellP, i, accum.M_coupled);
            aux->n_couplings_thistask++;
        }
    }
}

/* reset_per_iter_device_context — once per outer iter BEFORE per-subgroup
 * dispatch. Mutable ctx access lets us advance the mode state machine. The
 * runner re-captures ctx into each iter's kernel lambda, so the new
 * loop_iteration / per_active_local become visible to the next launch.
 *
 * Critical (codex r5): without this hook the runner would re-launch with the
 * iter-0 ctx forever — the mode-machine would never advance. */
void MechFBSpec::reset_per_iter_device_context(
        const neighbor_loop_args_iterative& args_iter,
        DeviceContext& ctx, int iter_index) {
    auto *aux = static_cast<Aux*>(args_iter.aux);
    if (!aux) return;

    const int num_modes = aux->num_modes;
    const int idx_clamped = (iter_index >= 0 && iter_index < num_modes) ? iter_index : 0;
    aux->mode_idx       = idx_clamped;
    ctx.mode_idx        = idx_clamped;
    ctx.loop_iteration  = aux->modes[idx_clamped];

    /* Repack host_locals_scratch for the new mode (reads host P[i] post-after_iter
     * writes from the previous iter) and copy into the SharedSpace per_active_local
     * UVM buffer the device kernel reads via load_active. */
    mechfb_repack_per_active_local(args_iter, *aux, ctx.loop_iteration);
    const int N = args_iter.num_active;
    if (N > 0 && ctx.per_active_local != nullptr &&
        (int)aux->host_locals_scratch.size() >= N) {
        std::memcpy(const_cast<MechFBLocalIn*>(ctx.per_active_local),
                    aux->host_locals_scratch.data(),
                    N * sizeof(MechFBLocalIn));
        /* Backend hygiene: fence after host→SharedSpace copy so the freshly
         * repacked per_active_local is visible to the next device launch
         * (codex review 2026-05-14 — non-blocking with current backends but
         * avoids backend-specific reordering ghosts later). */
        Kokkos::fence();
    }
}

/* ============================================================================
 * Ghost-writeback bundle — MILESTONE 3 SCOPE: empty bundle (no callback
 * registered). Single-rank correctness: num_ghosts == 0, runner's begin/end
 * bundle calls are strict no-ops (mesh/ghost_writeback.h:105,109).
 *
 * Mode A multi-rank correctness is INTENTIONALLY NOT YET PROVIDED — the
 * pair_kernel's `j >= num_local_gas` abort (mechfb_loop.h) prevents a silent
 * ghost-delta drop. The milestone 3.5 follow-up is the custom MechFBGasDelta
 * ghost-writeback callback (design Appendix A) + lazy d_gas_iter alloc in
 * reset_per_iter_device_context based on the runner's per-iter num_ghosts.
 *
 * Pattern reference: sinks/sink_swk_loop.cc:194-455 — when milestone 3.5
 * lands, the callback structure mirrors sink_swk's (snapshot/delta_for_ghost/
 * pack/apply/cleanup); the snapshot is no-op because d_gas_iter is zeroed
 * fresh each iter.
 * ========================================================================== */
namespace mechfb_writeback_detail {

static const struct ghost_writeback_bundle empty_bundle = {
    /* callbacks   */ nullptr,
    /* n_callbacks */ 0
};

}  /* namespace mechfb_writeback_detail */

void MechFBSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                        const NeighborLoopPlan& /*plan*/) {
    ghost_writeback_begin_bundle(&mechfb_writeback_detail::empty_bundle);
}

void MechFBSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                      const NeighborLoopPlan& /*plan*/) {
    ghost_writeback_end_bundle(&mechfb_writeback_detail::empty_bundle);
}

/* ============================================================================
 * Toplevel helpers — entry points for mechanical_fb.cc (non-GPU TU).
 * ========================================================================== */

struct MechFBGasDelta *mechfb_alloc_local_gas_delta(int n_gas) {
    const int n = (n_gas > 0) ? n_gas : 1;
    return (struct MechFBGasDelta *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(struct MechFBGasDelta));
}

void mechfb_free_local_gas_delta(struct MechFBGasDelta *p) {
    if (p) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);
}

/* mechfb_zero_local_gas_delta — zero the gas-only portion of the SharedSpace
 * buffer (codex r5 followup 2026-05-14: avoid std::memset on SharedSpace from
 * a non-GPU TU; route through Kokkos so the backend semantic is explicit).
 *
 * Mirrors the legacy zero loop in mechanical_fb.cc:275 — only entries where
 * P[j].Type == 0 are zeroed; non-gas entries are left untouched (saves work
 * on tiny-N steps). Runs on the host because the loop reads global P[]
 * (Type filter), which is not device-resident here. SharedSpace is
 * host-readable so a direct write is correct. */
void mechfb_zero_local_gas_delta(struct MechFBGasDelta *p, int n_gas) {
    if (!p || n_gas <= 0) return;
    Kokkos::fence();  /* ensure any prior device writes are visible */
    for (int j = 0; j < n_gas; ++j) {
        if (P[j].Type == 0) std::memset(&p[j], 0, sizeof(struct MechFBGasDelta));
    }
    Kokkos::fence();  /* ensure host zero is visible to the next device launch */
}

/* mechfb_run_iterative — runner-template dispatch entry for MechFBSpec.
 *
 * Validation matrix (codex r5+r6+r7 review, 2026-05-14):
 *   ✅ SINGLE-RANK Mode A                              — supported.
 *   ✅ SINGLE-RANK Mode B                              — supported.
 *   ✅ SINGLE-RANK Mode B + GIZMO_NLR_ORACLE=1         — supported (oracle-safe
 *      via after_iter status-only + after_iter_global production-only after
 *      oracle compare; threaded MechFBCallScalars; oracle-gated j-side atomics).
 *   ⚠️  Mode A + GIZMO_NLR_ORACLE=1                    — RUNNER-LEVEL STUB.
 *      The iterative runner hard-stubs Mode A oracle paths globally (see the
 *      `run_neighbor_loop_iterative` step 5.b note in mesh/neighbor_loop_runner.cc);
 *      NOT a mechfb-specific restriction. For oracle validation: force Mode B.
 *   ❌ MULTI-RANK Mode A                               — pair_kernel aborts
 *      loudly the first time a ghost-side write would happen
 *      (`j >= num_local_gas`). Custom MechFBGasDelta ghost-writeback callback
 *      + lazy d_gas_iter alloc is the milestone-3.5 follow-up; do NOT
 *      multi-rank Mode A validate until that lands.
 *   ✅ MULTI-RANK Mode B                               — supported by the
 *      collective-symmetry invariant (every rank enters with num_subgroups=1
 *      even if local num_active=0).
 */
void mechfb_run_iterative(int *active_list, int num_active,
                          struct MechFBGasDelta *LocalGasMechFBInfoTemp,
                          int n_gas, int *n_couplings_out) {
    /* Caller (mechanical_fb_calc_toplevel) has already short-circuited the
     * global_num_active == 0 case. A rank reaching here may still have local
     * num_active == 0 (single subgroup with num_active_local=0); the runner's
     * collective-symmetry invariant requires every rank to enter so that
     * Allreduce/Alltoallv calls inside the runner stay in lock-step. */

    /* Build Aux. Host-only per-iter mode state + scratch for per-source repack. */
    MechFBSpec::Aux aux;
    mechfb_populate_aux_initial(aux, num_active, LocalGasMechFBInfoTemp, n_gas);

    /* Single gas-only subgroup (density_build_subgroups pattern). May have
     * num_active_local == 0 — runner is fine with that (collective entry only). */
    NlrSubgroup sg{};
    sg.j_type_bitmask   = (1u << 0);
    sg.active_indices   = active_list;
    sg.num_active_local = num_active;

    neighbor_loop_args_iterative args{};
    static_cast<neighbor_loop_args&>(args) = nlr_default_args();
    args.P                   = P;
    args.CellP               = CellP;
    args.num_total           = NumPart;
    args.active_list         = active_list;
    args.num_active          = num_active;
    args.aux                 = &aux;
    args.num_subgroups       = 1;
    args.subgroups           = &sg;
    args.ghost_safety_factor = gizmo_ghost_safety_factor();

    run_neighbor_loop_iterative<MechFBSpec>(args);

    /* Guardrail #2 (codex r4): explicit fence before host-side scatter
     * (verify_and_assign_local_mechfb_integrals) reads LocalGasMechFBInfoTemp.
     * The runner's post-iter fence covers per-iter writes; the toplevel
     * adds this final fence after the call returns. */
    Kokkos::fence();

    if (n_couplings_out) *n_couplings_out = aux.n_couplings_thistask;
}

#endif /* GALSF_FB_MECHANICAL */
