/* mechfb_loop.cc — MechFBSpec method bodies + toplevel-facing helpers
 * for the runner-template port of mechanical_fb_evaluate_gpu.
 *
 * Full Spec contract: real host methods, state-machine plumbing,
 * ghost_writeback bundle; real mech_fb_local_fill / mech_fb_apply_aws_out
 * / mech_fb_apply_source_mass_out (promoted from `static` when the
 * legacy mechanical_fb_gpu.cc evaluator still existed; now the sole SSOT
 * for these helpers);
 * real reset_per_iter_device_context (repack + mode-machine advance).
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
#include "../declarations/gpu_error_check.h"
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
 * Mirrors the per-mode dispatch count of the retired legacy evaluator.
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
    /* num_local_particles = NumPart on this rank (pre-import). Queried at
     * populate-time once; constant across the 6 iters of this call (mechfb
     * neither adds nor removes particles mid-loop). Threaded into
     * mechfb_target_gas_delta as the upper bound for the local-non-gas gap. */
    aux.num_local_particles    = ghost_get_num_local();

    /* Ghost-side scratch is grown lazily per iter inside
     * reset_per_iter_device_context (when ghost_get_num_ghosts() exceeds
     * current capacity), zeroed via Kokkos::parallel_for, freed in
     * cleanup_device_context (single free path). Initial state: null/zero. */
    aux.d_gas_iter             = nullptr;
    aux.n_ghost_alloc          = 0;
    aux.total_ghost_packs      = 0;
}

/* ============================================================================
 * Public mech_fb_* helpers — promoted from `static` when the
 * legacy mechanical_fb_gpu.cc evaluator still existed; now the sole source
 * of truth, used only by the MechFBSpec runner-port.
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
    loc->TimeBin = P[i].TimeBin;  /* source TimeBin -- for downstream wakeup-mark encoding */
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
 * Mirrors the per-mode `for(aa) mech_fb_local_fill` inside the retired legacy
 * evaluator's mode loop. Reads HOST P[i] — which carries any
 * Area_weighted_sum / Mass updates the previous mode's after_iter_global
 * applied.
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
 * CONTRACT: addFB_evaluate_active_check(i, fb_iter)
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

/* search_radius — per-active radius. Mirrors the retired legacy evaluator's
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
/* Shared builder for MechFBCallScalars. Used by
 * MechFBSpec::populate_call_scalars so cosmology / unit-factor / CR-rigidity
 * values are read from the canonical host All via nlr_host_all_ptr() — no
 * bare All.* inside the kernel-callable helpers. */
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
#if defined(CR_DYNAMICAL_INJECTION_IN_SNE)
    /* Gate must match the struct field gate in mechanical_fb_functions.h:46
     * and the upstream All.CosmicRay_SNeFraction gate in
     * global_data_all_struct.h:408 (defined under
     * COSMIC_RAY_FLUID || COSMIC_RAY_SUBGRID_LEBRON). */
    scalars->CosmicRay_SNeFraction = host_all->CosmicRay_SNeFraction;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(GALSF_FB_FIRE_STELLAREVOLUTION) && defined(CRFLUID_EVOLVE_SPECTRUM)
    for (int k = 0; k < N_CR_PARTICLE_BINS; ++k) {
        scalars->CR_global_min_rigidity_in_bin   [k] = host_all->CR_global_min_rigidity_in_bin   [k];
        scalars->CR_global_max_rigidity_in_bin   [k] = host_all->CR_global_max_rigidity_in_bin   [k];
        scalars->CR_global_rigidity_at_bin_center[k] = host_all->CR_global_rigidity_at_bin_center[k];
    }
#endif
#ifdef SINK_WIND_SPAWN
    scalars->SpawnedWindCellID = host_all->SpawnedWindCellID;
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
 * single-rank scope: Mode A multi-rank ghost-side writes are
 * guarded by pair_kernel's loud abort (see the abort in
 * mechfb_loop.h). Initial loop_iteration/mode_idx values are set here AND
 * overwritten by reset_per_iter_device_context BEFORE iter 0 — the
 * duplication is intentional belt-and-suspenders. */
void MechFBSpec::populate_device_context(const neighbor_loop_args& args,
                                          DeviceContext& ctx) {
    auto *aux = static_cast<Aux*>(args.aux);

    ctx.mode_idx        = aux ? aux->mode_idx : 0;
    ctx.loop_iteration  = (aux && aux->num_modes > 0) ? aux->modes[ctx.mode_idx] : -2;
    ctx.num_modes       = aux ? aux->num_modes : 3;
    ctx.num_local_gas        = aux ? aux->num_local_gas        : 0;
    ctx.num_local_particles  = aux ? aux->num_local_particles  : 0;
    ctx.n_ghost_alloc        = aux ? aux->n_ghost_alloc        : 0;
    ctx.LocalGasMechFBInfoTemp = aux ? aux->LocalGasMechFBInfoTemp : nullptr;
    ctx.d_gas_iter      = aux ? aux->d_gas_iter : nullptr;  /* Aux owns; ctx mirrors */

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

/* cleanup_device_context — SOLE free path for per-call SharedSpace allocations.
 *
 * Owns: ctx.per_active_local (allocated in populate_device_context).
 * Co-owns (via Aux): aux->d_gas_iter (grown lazily in
 *   reset_per_iter_device_context). Aux holds the canonical pointer; the ctx
 *   mirror is cleared but the storage is freed via Aux. mechfb_run_iterative
 *   must NOT free d_gas_iter independently — this is the only path.
 *
 * Idempotent on already-null pointers. */
void MechFBSpec::cleanup_device_context(const neighbor_loop_args& args,
                                         DeviceContext& ctx) {
    if (ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<MechFBLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
    auto *aux = static_cast<Aux*>(args.aux);
    if (aux && aux->d_gas_iter) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(aux->d_gas_iter);
        aux->d_gas_iter   = nullptr;
        aux->n_ghost_alloc = 0;
    }
    ctx.d_gas_iter      = nullptr;
    ctx.n_ghost_alloc   = 0;
    ctx.LocalGasMechFBInfoTemp = nullptr;
}

/* apply_active_writeback — NO-OP for MechFBSpec.
 * All source-side P[i] writes (mode -2/-1 Area_weighted_sum writeback;
 * modes >=0 source mass loss) happen in MechFBSpec::after_iter_global
 * (host-side, per-iter). This hook stays declared because the Spec contract
 * requires it, but the runner's final end-of-call writeback loop has nothing
 * left to do. */
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

/* after_iter — STATUS-ONLY.
 *
 * An earlier draft applied per-mode source-side host writes here. They live
 * in after_iter_global instead: this hook runs inside the runner's convergence
 * pass, which visits only the slots still in the compacted active set, whereas
 * the source-side writes have to see every slot of the subgroup.
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
 * Runs ONCE per outer iter, after the per-active Spec::after_iter pass has
 * completed for every slot. Mutations here therefore apply exactly once per
 * iter.
 *
 * Per-iter responsibilities:
 *   - For each subgroup-0 slot, read drv.accum_uvm[0][slot].
 *   - mode -2: mech_fb_apply_aws_out  (writes P[i].Area_weighted_sum[0..6])
 *   - mode -1: mech_fb_apply_aws_out  (writes P[i].Area_weighted_sum[7..])
 *   - mode >=0 (gated on accum.M_coupled > 0): mech_fb_apply_source_mass_out
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
        /* Per-mode active mask — host-side equivalent of the
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
        }
    }
}

/* ============================================================================
 * Ghost-writeback bundle.
 *
 * Pattern: custom MechFBGasDelta ghost-writeback callback. Modeled on
 * sinks/sink_swk_loop.cc:194-455, but mechfb does NOT need snapshot-diff
 * because Aux::d_gas_iter is itself a per-iter DELTA accumulator (atomic_adds
 * from the kernel are pre-zeroed and self-contained). The wire format
 * collapses to {home_index, MechFBGasDelta delta} so a single struct copy
 * packs and a single mechfb_gas_delta_add applies. The three field-touch
 * helpers (zero/add/nonzero) are the SSOT — adding a new field to
 * MechFBGasDelta means updating ONLY those three helpers.
 *
 * Lifecycle per call (= one mechfb_run_iterative invocation, 6 iters):
 *   reset_per_iter_device_context  (per iter): lazy grow + zero d_gas_iter.
 *   begin_bundle                    (per iter): mirror Aux→s_ctx; runner snapshot.
 *   kernel launch                   (per iter): atomic_adds into d_gas_iter[g].
 *   ghost_writeback_end             (per iter): fence; bundle scan packs nonzeros
 *                                               via mechfb_gas_delta_nonzero+copy;
 *                                               MPI Alltoallv; apply via _add;
 *                                               cleanup_fn zeros d_gas_iter for
 *                                               the next iter.
 *   cleanup_device_context          (end of call): single free of Aux::d_gas_iter.
 *
 * MA-N validation criterion: Aux::total_ghost_packs
 * is incremented in pack_fn via a pointer stored in Ctx, then MPI_Allreduce-
 * summed at end of mechfb_run_iterative. Nonzero proves the path was
 * exercised — "did not abort" alone is not sufficient evidence. */
namespace mechfb_writeback_detail {

/* SSOT field-touch helpers. Gating mirrored 1:1 against MechFBGasDelta in
 * galaxy_sf/mechanical_fb_types.h. Helpers live TU-local because the runner
 * port is the sole consumer — the legacy mechanical_fb_gpu.cc evaluator and
 * the mesh-side ghost_writeback_mechfb wrapper have both been deleted (mechfb
 * physics does not belong in mesh). KOKKOS_INLINE_FUNCTION = host+device
 * callable. */
KOKKOS_INLINE_FUNCTION
static int mechfb_gas_delta_nonzero(const struct MechFBGasDelta *d) {
    /* Invariant (verified in mechanical_fb_pair_kernel and inject_cosmic_rays_into_delta):
     * every j-side write increments N_injected together with any other delta
     * field write. N_injected > 0 is therefore a sufficient predicate for
     * "this gas cell received any deltas this iter". */
    return (d->N_injected > 0) ? 1 : 0;
}

KOKKOS_INLINE_FUNCTION
static void mechfb_gas_delta_zero(struct MechFBGasDelta *d) {
    d->N_injected         = 0;
    d->max_source_wakeup  = 0;  /* MAX identity */
    d->m_injected  = 0;
    d->KE_injected = 0;
    d->TE_injected = 0;
    for (int k = 0; k < 3; ++k) d->p_injected[k] = 0;
    for (int k = 0; k < NUM_METAL_SPECIES; ++k) d->Z_injected[k] = 0;
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    d->Mass_Where_Dust_Shocked = 0;
#endif
#if defined(COSMIC_RAY_FLUID)
    for (int k = 0; k < N_CR_PARTICLE_BINS; ++k) d->CR_energy_injected[k] = 0;
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    for (int k = 0; k < N_CR_PARTICLE_BINS; ++k) d->CR_number_injected[k] = 0;
#endif
    for (int k = 0; k < 3; ++k) d->CR_dir_weighted[k] = 0;
#endif
}

KOKKOS_INLINE_FUNCTION
static void mechfb_gas_delta_add(struct MechFBGasDelta *dst,
                                 const struct MechFBGasDelta *src) {
    dst->N_injected  += src->N_injected;
    dst->max_source_wakeup = IMAX(dst->max_source_wakeup, src->max_source_wakeup);
    dst->m_injected  += src->m_injected;
    dst->KE_injected += src->KE_injected;
    dst->TE_injected += src->TE_injected;
    for (int k = 0; k < 3; ++k) dst->p_injected[k] += src->p_injected[k];
    for (int k = 0; k < NUM_METAL_SPECIES; ++k) dst->Z_injected[k] += src->Z_injected[k];
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    dst->Mass_Where_Dust_Shocked += src->Mass_Where_Dust_Shocked;
#endif
#if defined(COSMIC_RAY_FLUID)
    for (int k = 0; k < N_CR_PARTICLE_BINS; ++k) dst->CR_energy_injected[k] += src->CR_energy_injected[k];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    for (int k = 0; k < N_CR_PARTICLE_BINS; ++k) dst->CR_number_injected[k] += src->CR_number_injected[k];
#endif
    for (int k = 0; k < 3; ++k) dst->CR_dir_weighted[k] += src->CR_dir_weighted[k];
#endif
}

/* Wire format: {home_index, full MechFBGasDelta}. Single struct copy on pack,
 * single mechfb_gas_delta_add on apply — no field-list duplication. */
struct Wire {
    int                   home_index;
    struct MechFBGasDelta delta;
};

/* Callback context. Mirrored from Aux by MechFBSpec::ghost_writeback_begin
 * once per bundle (per iter). All pointers are non-owning. */
struct Ctx {
    struct MechFBGasDelta *d_gas_iter;          /* mirror of aux->d_gas_iter            */
    struct MechFBGasDelta *home_buf;            /* = aux->LocalGasMechFBInfoTemp        */
    int                    num_local_gas;       /* = aux->num_local_gas (apply bound)   */
    int                    num_ghosts;          /* set by snapshot_fn                   */
    long long             *total_ghost_packs;   /* points into aux->total_ghost_packs   */
};
static Ctx s_ctx{nullptr, nullptr, 0, 0, nullptr};

static void snapshot_fn(void * /*vctx — using static s_ctx*/, int num_ghosts, int /*num_local*/) {
    /* Pre-kernel hook. d_gas_iter is zeroed in reset_per_iter_device_context
     * before any kernel launch this iter, so no work to do here besides
     * recording the count. The Kokkos::fence between device writes and the
     * upcoming host bundle scan lives in MechFBSpec::ghost_writeback_end
     * (NOT here — snapshot fires pre-kernel). */
    s_ctx.num_ghosts = num_ghosts;
}

static int delta_for_ghost_fn(void * /*vctx*/, int g, int /*num_local*/) {
    if (s_ctx.d_gas_iter == nullptr) return 0;
    return mechfb_gas_delta_nonzero(&s_ctx.d_gas_iter[g]);
}

static void pack_fn(void * /*vctx*/, int g, int /*num_local*/, void *out) {
    Wire *w = static_cast<Wire*>(out);
    w->home_index = ghost_get_home_index()[g];
    w->delta      = s_ctx.d_gas_iter[g];                /* trivially-copyable */
    if (s_ctx.total_ghost_packs) (*s_ctx.total_ghost_packs)++;
}

static void apply_fn(void * /*vctx*/, const void *in) {
    const Wire *w = static_cast<const Wire*>(in);
    /* Defensive: silent home_index OOB hides provenance bugs. */
    if (s_ctx.home_buf == nullptr) {
        fprintf(stderr, "mechfb apply_fn: s_ctx.home_buf is null on rank %d "
                        "(begin must populate it from Aux).\n", ThisTask);
        endrun(91031);
        return; /* soft-stop + return: home_buf is NULL, the delta-add below would deref NULL */
    }
    if (w->home_index < 0 || w->home_index >= s_ctx.num_local_gas) {
        fprintf(stderr, "mechfb apply_fn: home_index=%d out of [0,%d) "
                        "on rank %d.\n",
                w->home_index, s_ctx.num_local_gas, ThisTask);
        endrun(91032);
        return; /* soft-stop + return: home_index is OOB, the delta-add below would write out of bounds */
    }
    mechfb_gas_delta_add(&s_ctx.home_buf[w->home_index], &w->delta);
}

static void cleanup_fn(void * /*vctx*/) {
    /* Zero d_gas_iter for the next iter (or next pass — defensive against any
     * future flow that wraps multiple kernel passes inside one ghost_writeback
     * pair; today there is exactly one pass per iter). Storage stays
     * allocated; cleanup_device_context is the sole free path. */
    if (s_ctx.d_gas_iter && s_ctx.num_ghosts > 0) {
        MechFBGasDelta *d = s_ctx.d_gas_iter;
        const int n = s_ctx.num_ghosts;
        Kokkos::parallel_for("mechfb_d_gas_iter_zero_cleanup", n,
                              KOKKOS_LAMBDA(int g) {
            mechfb_gas_delta_zero(&d[g]);
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("mechfb_d_gas_iter_zero_cleanup", n);
    }
    s_ctx.num_ghosts = 0;
}

static const struct ghost_writeback_callback callback = {
    sizeof(Wire),
    snapshot_fn,
    delta_for_ghost_fn,
    pack_fn,
    apply_fn,
    cleanup_fn,
    & s_ctx,
};

static const struct ghost_writeback_callback *const raw_cbs[] = { & callback, nullptr };
/* loop_name = nullptr: mechfb already prints its own MA-N evidence via
 * Aux::total_ghost_packs at end of mechfb_run_iterative. Silent here avoids
 * duplicating that line. Future cleanup could retire the bespoke counter
 * and adopt the generic print by setting loop_name = "mechfb". */
static const struct ghost_writeback_bundle bundle = { raw_cbs, 1, nullptr };

}  /* namespace mechfb_writeback_detail */

/* MechFBSpec::ghost_writeback_begin — mirror Aux into the callback's s_ctx
 * BEFORE bundle begin. Aux is reachable here via args.aux per the runner-
 * template contract (DeviceContext is NOT in args). */
void MechFBSpec::ghost_writeback_begin(const neighbor_loop_args& args,
                                        const NeighborLoopPlan& /*plan*/) {
    auto *aux = static_cast<Aux*>(args.aux);
    mechfb_writeback_detail::s_ctx.d_gas_iter        = aux ? aux->d_gas_iter            : nullptr;
    mechfb_writeback_detail::s_ctx.home_buf          = aux ? aux->LocalGasMechFBInfoTemp : nullptr;
    mechfb_writeback_detail::s_ctx.num_local_gas     = aux ? aux->num_local_gas         : 0;
    mechfb_writeback_detail::s_ctx.total_ghost_packs = aux ? &aux->total_ghost_packs    : nullptr;
    /* num_ghosts is set by snapshot_fn inside begin_bundle (it's passed as the
     * runner-side ghost_get_num_ghosts() snapshot). */
    ghost_writeback_begin_bundle(&mechfb_writeback_detail::bundle);
}

/* MechFBSpec::ghost_writeback_end — fence device atomic writes to d_gas_iter
 * before the host-side bundle scan (delta_for_ghost_fn / pack_fn read from
 * the host). The runner does NOT unconditionally fence between the kernel
 * launch and dispatch_ghost_writeback_end — this fence is load-bearing. */
void MechFBSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                      const NeighborLoopPlan& /*plan*/) {
    Kokkos::fence();
    ghost_writeback_end_bundle(&mechfb_writeback_detail::bundle);
}

/* reset_per_iter_device_context — once per outer iter BEFORE per-subgroup
 * dispatch. Mutable ctx access lets us advance the mode state machine. The
 * runner re-captures ctx into each iter's kernel lambda, so the new
 * loop_iteration / per_active_local become visible to the next launch.
 *
 * Critical: without this hook the runner would re-launch with the
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
         * (non-blocking with current backends but avoids backend-specific
         * reordering ghosts later). */
        Kokkos::fence();
    }

    /* Lazy d_gas_iter alloc/grow for this iter's ghost imports.
     *
     * ghost_get_num_ghosts() returns the current rank's imported-ghost count
     * (post-import, valid by the time the runner has issued effective_args).
     * For Mode B / single-rank: returns 0, no allocation; ctx.d_gas_iter stays
     * nullptr. For Mode A multi-rank: grow Aux's buffer if num_ghosts exceeds
     * current capacity, zero it via Kokkos::parallel_for (device-aware), then
     * mirror the pointer + capacity into ctx so load_active / pair_kernel see
     * the right buffer.
     *
     * Aux is the sole owner; cleanup_device_context performs the only free. */
    const int num_ghosts_now = ghost_get_num_ghosts();
    if (num_ghosts_now > aux->n_ghost_alloc) {
        if (aux->d_gas_iter) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(aux->d_gas_iter);
            aux->d_gas_iter = nullptr;
        }
        aux->d_gas_iter = (MechFBGasDelta *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            num_ghosts_now * sizeof(MechFBGasDelta));
        aux->n_ghost_alloc = num_ghosts_now;
    }
    if (num_ghosts_now > 0 && aux->d_gas_iter != nullptr) {
        /* Zero used range (parallel_for + fence; mirrors the SharedSpace
         * style used elsewhere in this TU rather than std::memset, so the
         * backend reorders writes correctly relative to the upcoming device
         * launch that reads/writes via Kokkos::atomic_add). */
        MechFBGasDelta *d = aux->d_gas_iter;
        Kokkos::parallel_for("mechfb_d_gas_iter_zero", num_ghosts_now,
                              KOKKOS_LAMBDA(int g) {
            mechfb_writeback_detail::mechfb_gas_delta_zero(&d[g]);
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("mechfb_d_gas_iter_zero", num_ghosts_now);
    }
    ctx.d_gas_iter    = aux->d_gas_iter;       /* mirror: ctx is non-owning */
    ctx.n_ghost_alloc = aux->n_ghost_alloc;

}

/* mechfb_writeback_detail namespace + MechFBSpec::ghost_writeback_begin/end
 * are defined ABOVE reset_per_iter_device_context (the device-callable
 * mechfb_gas_delta_zero helper inside the namespace is referenced from
 * reset_per_iter's lambda for the per-iter d_gas_iter zero-fill, so the
 * namespace must be in scope at that point). */

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
 * buffer (avoid std::memset on SharedSpace from a non-GPU TU; route through
 * Kokkos so the backend semantic is explicit).
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

/* Persistent capacity-managed gas-delta buffer (grow-only). Replaces the per-step
 * alloc + O(N_gas) zero + free. Invariant (upheld by the caller): the buffer is
 * all-zero between mechfb steps -- verify_and_assign_local_mechfb_integrals
 * re-zeros every drained (N_injected>0) cell via mechfb_reset_one_gas_delta, and
 * only N_injected>0 cells are ever written (mechfb_gas_delta_nonzero SSOT). Being
 * all-zero between steps makes particle reorder (domain decomposition) harmless. */
struct MechFBGasDelta *mechfb_get_persistent_gas_delta(int n_gas) {
    static struct MechFBGasDelta *s_buf = nullptr;
    static int s_cap = 0;
    const int n = (n_gas > 0) ? n_gas : 1;
    if (n > s_cap) {
        if (s_buf) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(s_buf);
        s_buf = (struct MechFBGasDelta *)
            Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("mechfb_gas_delta", n * sizeof(struct MechFBGasDelta));
        Kokkos::fence();
        for (int j = 0; j < n; ++j) mechfb_writeback_detail::mechfb_gas_delta_zero(&s_buf[j]);
        Kokkos::fence();
        s_cap = n;
    }
    return s_buf;
}

/* Host-callable single-cell reset (SSOT mechfb_gas_delta_zero); called on each
 * drained cell to keep the persistent buffer all-zero between steps. */
void mechfb_reset_one_gas_delta(struct MechFBGasDelta *p, int j) {
    mechfb_writeback_detail::mechfb_gas_delta_zero(&p[j]);
}

/* mechfb_run_iterative — runner-template dispatch entry for MechFBSpec.
 *
 * Dispatch coverage:
 *   SINGLE-RANK Mode A and Mode B.
 *   MULTI-RANK Mode A — ghost-side writes land in d_gas_iter (allocated per
 *      iter from ghost_get_num_ghosts()) and are shipped to their home ranks
 *      by the MechFBGasDelta ghost-writeback bundle in this file.
 *      Aux::total_ghost_packs counts the records sent and is the evidence
 *      that cross-rank coupling actually happened.
 *   MULTI-RANK Mode B — held by the collective-symmetry invariant (every rank
 *      enters with num_subgroups=1 even if local num_active=0).
 */
void mechfb_run_iterative(int *active_list, int num_active,
                          struct MechFBGasDelta *LocalGasMechFBInfoTemp,
                          int n_gas) {
    GIZMO_GPU_ENSURE_ALL_FRESH();

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

    /* Guardrail: explicit fence before host-side scatter
     * (verify_and_assign_local_mechfb_integrals) reads LocalGasMechFBInfoTemp.
     * The runner's post-iter fence covers per-iter writes; the toplevel
     * adds this final fence after the call returns. */
    Kokkos::fence();

    /* Validation readout — proves the multi-rank Mode A
     * ghost-writeback path was actually exercised. MA-N pass criterion:
     * total > 0 in at least one mechfb call ("did not abort" is not
     * sufficient evidence). Single-rank / Mode B:
     * always 0 (no ghost imports). Cheap one-shot Allreduce + rank-0 print. */
    {
        long long local_packs = aux.total_ghost_packs;
        long long global_packs = 0;
        MPI_Allreduce(&local_packs, &global_packs, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD);
        if (global_packs > 0 && ThisTask == 0) {
            fprintf(stdout, "[mechfb] ghost-writeback packs this call: total=%lld\n",
                    global_packs);
            fflush(stdout);
        }
    }
}

#endif /* GALSF_FB_MECHANICAL */
