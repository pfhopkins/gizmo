/* mechfb_loop.h — MechFBSpec for the runner-template port of
 * mechanical_fb_evaluate_gpu (Phase 4 / Wave 3 / 3e.1).
 *
 * Milestone 3 (physics-complete): full Spec contract — real pair_kernel
 * (calls mechanical_fb_pair_kernel in mechanical_fb_functions.h), real
 * load_active (builds MechFBLocalIn + MechFBSourceMode + DeviceContext
 * snapshots per active per mode), real reset_per_iter_device_context
 * (repack + mode-machine advance).
 *
 * SSOT design: ../OPEN_3d_mechfb_design.md (v0.4-final, codex-approved).
 *
 * Aux is HOST-ONLY (existing iterative-runner contract; design v0.4-final
 * correction). Mutable per-iter state visible to the device lives in
 * MechFBDeviceContext (extends NeighborLoopDeviceContextBase, captured by
 * value into the device lambda each iter).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. */
#pragma once

#include <vector>

#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"     /* MODE_B_SEARCH_*, MODE_B_RADIUS_* */
#include "../declarations/multifluid_helpers.h"
#include "mechanical_fb_types.h"             /* struct MechFBGasDelta */
/* MechFBCallScalars + shared kernel helpers + host-side inline helpers */
#include "mechanical_fb_functions.h"

#ifdef GALSF_FB_MECHANICAL

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* ============================================================================
 * ActiveData — per-source state visible to the device kernel.
 *
 * Carries:
 *   - The per-source pack (local) used by mechanical_fb_pair_kernel and the
 *     per-source setup helper.
 *   - source_mode, computed once per active in load_active via
 *     mechanical_fb_per_source_setup. Avoids re-running per_source_setup
 *     for every pair.
 *   - DeviceContext field snapshots (LocalGasMechFBInfoTemp / d_gas_iter /
 *     num_local_gas / oracle_dry_run / P_base / CellP_base) — pair_kernel
 *     does NOT receive DeviceContext, so any ctx-derived value the j-side
 *     writes need must ride here.
 *
 * Flat `pos` + `h_search` are required by the runner's
 * collect_candidates_for_remote_queries (Mode B remote candidate query
 * builder reads them via `active.pos[k]` / `active.h_search`); see
 * mesh/neighbor_loop_runner.cc:880-886 comment. They mirror local.Pos /
 * local.KernelRadius and are populated in load_active.
 * ========================================================================== */
struct MechFBActiveState {
    Vec3<MyDouble>         pos;             /* mirrors local.Pos (runner-required flat) */
    MyFloat                h_search;        /* mirrors local.KernelRadius (runner-required flat) */
    MechFBLocalIn          local;           /* per-source pack for current loop_iteration */
    MechFBSourceMode       source_mode;     /* per-source per-mode precomputed state */
    int                    loop_iteration;

    /* Per-mode active flag — populated in load_active via
     * mechanical_fb_star_active_check(i, loop_iteration, dctx.P). Mirrors the
     * legacy `if (!mechanical_fb_star_active_check(...)) return;` guard at the
     * top of mechanical_fb_evaluate_gpu's kernel (line 316). Without it, a
     * source in the toplevel superset that is NOT active for THIS specific
     * mode would still run pair_kernel for every neighbor — for modes 0+ that
     * would atomic-add N_injected even with otherwise-zero deltas (codex r7
     * 2026-05-14). pair_kernel early-returns on !is_active_this_mode;
     * after_iter_global re-checks host-side and skips the per-mode source
     * write for inactive entries. */
    bool                   is_active_this_mode;

    /* Per-call cosmology + physics + unit factors (codex r6 fix). Carried
     * per-active so the pair kernel reads scalars.* instead of bare All.* —
     * follows the density-port pattern (DensitySpec::ActiveData also stores
     * a CallScalars copy). Cost ~200 B per active; trivially copyable. */
    MechFBCallScalars      scalars;

    /* DeviceContext snapshots routed through ActiveData because pair_kernel
     * has no ctx parameter. Trivially copyable; cost is a few pointers + ints
     * per active per iter. */
    struct MechFBGasDelta *LocalGasMechFBInfoTemp;   /* home-side j-write target (SharedSpace) */
    struct MechFBGasDelta *d_gas_iter;               /* ghost-side scratch (Mode A multi-rank only; nullptr when num_ghosts==0) */
    int                    num_local_gas;            /* = N_gas      ; home/non-gas boundary */
    int                    num_local_particles;      /* = NumPart    ; non-gas/imported-ghost boundary */
    bool                   oracle_dry_run;
    struct particle_data  *P_base;                   /* dctx.P; pair_kernel needs full P array for the refactored legacy kernel */
    struct gas_cell_data  *CellP_base;               /* dctx.CellP; nullable when CellP unavailable */
};

/* AccumData = legacy MechFBOut (M_coupled + Area_weighted_sum[]). */
using MechFBAccumData = MechFBOut;

/* ============================================================================
 * NeighborData — j-side per-pair pointer pack. Mechfb writes j-side via
 * atomics into gas_delta (DeviceContext.LocalGasMechFBInfoTemp or d_gas_iter
 * depending on ownership split); the kernel reads CellP[j] only when j is
 * gas. neighbor_cell is gated on (Type==0 && CellP != nullptr) in load_neighbor.
 * ========================================================================== */
struct MechFBNeighborData {
    struct particle_data *neighbor_particle;
    struct gas_cell_data *neighbor_cell;
    int                   neighbor_index;
};

/* ============================================================================
 * MechFBDeviceContext — extends the base ctx with mechfb-specific UVM ptrs
 * and mutable-across-iters state. Trivially copyable (captured by value into
 * the device lambda each iter; runner re-captures between iters).
 * ========================================================================== */
struct MechFBDeviceContext : NeighborLoopDeviceContextBase {
    /* Mutable across iters (set by after_iter_global; runner re-captures). */
    int   loop_iteration;                         /* = modes[mode_idx] */
    int   mode_idx;                               /* 0..num_modes-1 */
    bool  oracle_dry_run;                         /* set via Spec::set_oracle_brute_pass */

    /* Immutable across iters but device-visible (set in populate_device_context). */
    int   num_modes;                              /* 3 / 4 / 5 / 6 per FIRE flags */
    int   num_local_gas;                          /* = N_gas    ; home/non-gas boundary */
    int   num_local_particles;                    /* = NumPart  ; non-gas/imported-ghost boundary */
    int   n_ghost_alloc;                          /* d_gas_iter capacity (Mode A multi-rank only; 0 otherwise) */

    /* SharedSpace pointers (host-allocated by toplevel + populate_device_context). */
    struct MechFBGasDelta      *LocalGasMechFBInfoTemp;  /* SharedSpace, length num_local_gas */
    struct MechFBGasDelta      *d_gas_iter;              /* SharedSpace, length n_ghost_alloc, nullptr in Mode B */
    const struct MechFBLocalIn *per_active_local;        /* SharedSpace, length num_active; repacked per iter */
};

/* ============================================================================
 * MechFBSpec — the runner-template Spec for mechanical feedback.
 *
 * Iterative as a 6-pass state machine over loop_iteration modes
 * {-2, -1, 0, 1, 2, 3} (num_modes truncates to 3/4/5/6 per FIRE flags).
 * Mode -2/-1 are weighting passes; modes >= 0 are coupling passes.
 * after_iter returns NeedsMore until mode_idx == num_modes-1, then Converged.
 * after_iter_global advances mode_idx + repacks per_active_local for the
 * next mode + applies inter-mode source-side host writes (mode -2/-1 →
 * Area_weighted_sum writeback; mode >=0 → source mass loss).
 *
 * mode_a_rebuild_csr_every_iter = false is LOAD-BEARING — preserves the
 * legacy 1-CSR-shared-across-6-modes Mode A optimization (design §F).
 * ========================================================================== */
struct MechFBSpec {
    /* Identity */
    static constexpr const char *loop_name = "mechfb";

    /* Search policy: mirrors legacy mechanical_fb_evaluate_gpu (NGB_SEARCH_SYMMETRIC
     * + gas-only) at mechanical_fb_gpu.cc:270 (gpu_ngb_list_build call). */
    static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int            neighbor_type_mask = (1u << 0);   /* gas only */
    static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;

    /* Write policy. j-side writes happen via uses_ghost_writeback bundle
     * (orthogonal to write_pattern per WritePattern enum docstring). */
    static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::GasOnly;   /* tbm = 1 */
    static constexpr bool           uses_ghost_writeback      = true;
    static constexpr bool           uses_ghost_write_detector = false;

    /* Convergence + iter policy. Mode-machine: max_iters = 6 covers the
     * widest FIRE config (-2..3); accum/radius tolerances are non-radius-
     * iterative and never trigger Converged-by-tolerance (after_iter uses
     * mode_idx to drive the state machine). */
    static constexpr double accum_tolerance  = 1e-8;
    static constexpr double radius_tolerance = 1e-9;
    using                    IterControl = Iterative;
    using                    IterScratch = NoIterScratch;
    static constexpr int     max_iters   = 6;

    /* Mode A CSR policy. mode_a_rebuild_csr_every_iter=false is LOAD-BEARING
     * (design §F): preserves legacy 1-CSR-shared-across-6-modes optimization. */
    static constexpr double mode_a_csr_buffer_factor      = 1.3;
    static constexpr bool   mode_a_rebuild_csr_every_iter = false;

    /* Single j-mask subgroup (gas-only). SupportsSubgroups OMITTED (default
     * false_type — runner runtime-asserts single subgroup at entry). */

    /* Type aliases */
    using CallScalars   = MechFBCallScalars;
    using ActiveData    = MechFBActiveState;
    using AccumData     = MechFBAccumData;
    using NeighborData  = MechFBNeighborData;
    using DeviceContext = MechFBDeviceContext;
    using ScatterData   = NoScatter;

    /* Aux — host-only per-call state. Owned by mechanical_fb_calc_toplevel
     * (lives on the toplevel stack for the duration of the runner call).
     * Mutable per-iter from reset_per_iter_device_context (advances mode_idx
     * + refills host_locals_scratch) and from after_iter_global (per-active
     * n_couplings_thistask increment when accum.M_coupled > 0; production-only
     * per codex r6). */
    struct Aux {
        int                          num_modes;                 /* 3..6 per FIRE flags */
        int                          modes[6];                  /* {-2, -1, 0, 1, 2, 3} */
        int                          mode_idx;                  /* 0..num_modes-1 */
        std::vector<MechFBLocalIn>   host_locals_scratch;       /* [num_active]; repacked each iter */
        struct MechFBGasDelta       *LocalGasMechFBInfoTemp;    /* SharedSpace ptr; owned by toplevel */
        int                          num_local_gas;             /* = N_gas */
        int                          num_local_particles;       /* = NumPart (= ghost_get_num_local()) */
        int                          n_couplings_thistask;      /* accumulated host-side counter */

        /* Milestone 3.5 — ghost-side scratch + ghost-writeback validation counter.
         * Aux is the SOLE OWNER of d_gas_iter: lazy alloc/grow in
         * reset_per_iter_device_context based on ghost_get_num_ghosts(); free
         * via cleanup_device_context (single free path; mechfb_run_iterative
         * does not touch d_gas_iter). DeviceContext mirrors the pointer +
         * capacity for device-side reads. */
        struct MechFBGasDelta       *d_gas_iter;                /* SharedSpace, length n_ghost_alloc; nullptr when num_ghosts==0 */
        int                          n_ghost_alloc;             /* current d_gas_iter capacity */

        /* MA-N validation counter: incremented in callback pack_fn each time a
         * ghost-side MechFBGasDelta entry is shipped. End of mechfb_run_iterative
         * MPI_Allreduce-sums across ranks and rank-0 prints; nonzero proves the
         * new Mode-A-multi-rank ghost-writeback path was actually exercised
         * (codex review 2026-05-14: "did not abort" is not sufficient evidence). */
        long long                    total_ghost_packs;

    };

    /* ====================================================================
     * Host hooks. Bodies in mechfb_loop.cc.
     * ==================================================================== */
    static bool        is_active(int particle_index);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);

    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context (const neighbor_loop_args& args, DeviceContext& ctx);

    /* No-op for mechfb: all source-side P[i] writes happen in after_iter_global
     * (per-iter, inter-mode). Declared because the Spec contract requires it. */
    static void apply_active_writeback(const neighbor_loop_args& args,
                                       int active_slot, int i,
                                       const AccumData& accum);

    static void merge_accum(AccumData& local, const AccumData& peer);

    /* Oracle suppression: flips ctx.oracle_dry_run; pair_kernel gates j-side
     * atomic writes on the flag (i-side accum still completed). */
    static void set_oracle_brute_pass(DeviceContext& ctx, bool on);

    static double compare_accum(const AccumData& local, const AccumData& oracle);

    /* Per-active post-iter hook — STATUS-ONLY (codex r6 fix 2026-05-14).
     * Returns NeedsMore until iter_index >= num_modes-1, then Converged.
     * Must NOT mutate host P[i] / CellP[i] / Aux: the iterative runner calls
     * after_iter twice with oracle enabled (production accum at runner.cc:4128
     * + oracle accum at runner.cc:4189), so any mutation here would double-
     * apply under GIZMO_NLR_ORACLE=1. Source-side host writes live in
     * after_iter_global (production-only; runner.cc:4310-4313). */
    static IterResult after_iter(const AfterIterContext<MechFBSpec>& ctx,
                                 const AccumData& accum);

    /* Production-only per-iter host hook. Walks subgroup-0's actives and
     * applies the current mode's source-side host write to P[i] / CellP[i]
     * from drv.accum_uvm[0][slot]:
     *   mode -2: mech_fb_apply_aws_out for k in [0, 7)
     *   mode -1: mech_fb_apply_aws_out for k in [7, AREA_WEIGHTED_SUM_ELEMENTS)
     *   mode >= 0 (and accum.M_coupled > 0): mech_fb_apply_source_mass_out
     *     + aux->n_couplings_thistask++
     * Runs ONCE per outer iter (production-only — runner.cc:4310-4313 calls
     * it AFTER both production+oracle after_iter passes and after oracle
     * compare), so oracle dual-walk does not double-apply. */
    static void after_iter_global(const neighbor_loop_args& args,
                                  const NlrIterDriver<MechFBSpec>& drv);

    /* Per-iter device-context refresh (Phase 4.B.0 step 2c.1 SFINAE hook).
     * Runner calls this once per outer iter BEFORE per-subgroup dispatch with
     * mutable ctx. Responsibilities:
     *   - set ctx.mode_idx = iter_index, ctx.loop_iteration = aux->modes[iter_index]
     *   - call mechfb_repack_per_active_local for the new mode (reads host P[i]
     *     post-after_iter_global writes, writes MechFBLocalIn into ctx.per_active_local UVM)
     * Critical: without this hook the runner stays on the iter-0 mode forever. */
    static void reset_per_iter_device_context(const neighbor_loop_args_iterative& args,
                                               DeviceContext& ctx,
                                               int iter_index);

    /* Custom ghost-writeback callback (MechFBGasDelta is a parallel array, not
     * a gas_cell_data member — manifest macros don't fit; pattern from
     * sinks/sink_swk_loop.cc:194-455). Milestone 3.5: real callback bundle in
     * mechfb_writeback_detail (mechfb_loop.cc) ships nonzero ghost-side
     * MechFBGasDelta entries to home ranks. ghost_writeback_begin populates
     * the bundle's Ctx from Aux (d_gas_iter, LocalGasMechFBInfoTemp,
     * num_local_gas, total_ghost_packs counter); ghost_writeback_end fences
     * Kokkos so device atomic writes to d_gas_iter are visible to the host
     * bundle scan, then calls _end_bundle which packs/MPI/applies via
     * mechfb_gas_delta_{nonzero,zero,add} helpers (TU-local in mechfb_loop.cc;
     * SSOT for field-touch policy). */
    static void ghost_writeback_begin(const neighbor_loop_args& args, const NeighborLoopPlan& plan);
    static void ghost_writeback_end  (const neighbor_loop_args& args, const NeighborLoopPlan& plan);

    /* ====================================================================
     * Device hooks. Header-inline.
     * ==================================================================== */
    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& accum) {
        accum.M_coupled = 0;
        for (int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; ++k) {
            accum.Area_weighted_sum[k] = 0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx, int active_slot, int i,
                                  double h_search, const CallScalars& scalars) {
        ActiveData a{};
        a.loop_iteration = dctx.loop_iteration;
        if (dctx.per_active_local != nullptr) {
            a.local = dctx.per_active_local[active_slot];
        }
        /* Flat pos / h_search mirror the per-source pack so the runner's Mode B
         * candidate query builder (collect_candidates_for_remote_queries) reads
         * consistent values. Falls back to direct ctx.P[i] when per_active_local
         * isn't staged yet (shouldn't happen in production: reset_per_iter_device_context
         * fires before every iter and repacks). */
        if (dctx.per_active_local != nullptr) {
            a.pos      = a.local.Pos;
            a.h_search = a.local.KernelRadius;
        } else {
            a.pos[0]   = (MyDouble)dctx.P[i].Pos[0];
            a.pos[1]   = (MyDouble)dctx.P[i].Pos[1];
            a.pos[2]   = (MyDouble)dctx.P[i].Pos[2];
            a.h_search = (MyFloat) h_search;
        }
        /* Carry the per-call scalars on the ActiveData so pair_kernel + the
         * refactored mechanical_fb_pair_kernel / inject_cosmic_rays_into_delta
         * read scalars.* (precomputed unit factors, cf_atime/etc., CR rigidity
         * arrays) without touching bare All.*. Density port pattern (codex r6
         * fix). */
        a.scalars = scalars;

        /* Per-mode active mask (codex r7 fix) — mirrors legacy
         * mechanical_fb_evaluate_gpu:316. ctx.P points at the runner-managed
         * particle array (Mode A: arena P_gpu; Mode B: owner-local P), both
         * of which have stable .SNe_ThisTimeStep / .MassReturn_ThisTimeStep /
         * etc. for the duration of the runner call. */
        a.is_active_this_mode =
            (mechanical_fb_star_active_check(i, a.loop_iteration, dctx.P) != 0);

        /* Per-source per-mode precomputed state (Sedov budget, kernel norms, etc.).
         * Computed ONCE per active per iter here; pair_kernel reads via
         * active.source_mode. Takes scalars for All.cf_atime / All.cf_a3inv /
         * All.CosmicRay_SNeFraction / UNIT_*_IN_* reads. */
        mechanical_fb_per_source_setup(a.local, a.loop_iteration, a.scalars, a.source_mode);

        /* Snapshot DeviceContext fields the pair kernel needs but cannot read
         * directly (pair_kernel has no ctx parameter).
         *
         * MPI- AND ORACLE-SAFETY: every field set below is rank-local AND
         * eval-pass-local. They get OVERWRITTEN by
         * MechFBSpec::bind_active_to_eval_context (below) inside the runner's
         * evaluate_pairs_post_drift, once per active per eval pass, using the
         * exact eval ctx in play. So:
         *   - Mode-B-remote PEER eval on a receiver rank uses receiver's ctx
         *     (not sender's stale pointers shipped via MPI envelope).
         *   - Oracle brute-pass uses ctx_oracle_*.oracle_dry_run = true (not
         *     the false captured here at original load_active time), so j-side
         *     atomic writes are suppressed during brute as designed.
         * The values set here matter only for the FIRST eval pass on the
         * active rank in Production mode (the bind hook is idempotent there). */
        a.LocalGasMechFBInfoTemp = dctx.LocalGasMechFBInfoTemp;
        a.d_gas_iter             = dctx.d_gas_iter;
        a.num_local_gas          = dctx.num_local_gas;
        a.num_local_particles    = dctx.num_local_particles;
        a.oracle_dry_run         = dctx.oracle_dry_run;
        a.P_base                 = dctx.P;
        a.CellP_base             = dctx.CellP;

        return a;
    }

    /* Per-eval-pass binding hook (see nlr_spec_has_bind_active_to_eval_context
     * docstring in mesh/neighbor_loop_runner.h). The runner calls this for
     * every active right before its pair_kernel inside evaluate_pairs_post_drift,
     * with the EXACT ctx of that eval pass — so:
     *   1. Mode-B-remote PEER eval: receiver's ctx values overwrite the
     *      sender's rank-local snapshots that arrived via MPI envelope
     *      (fixes the np=2 wind_singlestar SIGSEGV in mechanical_fb_pair_kernel
     *      where rank 0 was using rank 1's P_base etc).
     *   2. Oracle paths (OracleCompare / OracleIterative): the runner copies
     *      ctx into ctx_oracle_* and flips oracle_dry_run via
     *      set_oracle_brute_pass on the COPY. Without per-eval-pass binding,
     *      active.oracle_dry_run would stay at the original ctx's value
     *      (false) on the brute pass, j-side atomic_adds would fire twice,
     *      and the oracle would silently miss the double-coupling. Codex
     *      flagged this 2026-05-20 after the initial rebind-only fix.
     *
     * Source-owned physics fields (pos, h_search, local, source_mode,
     * loop_iteration, is_active_this_mode, scalars) are preserved here — they
     * describe the sender's star / per-call physics and are stable across
     * ranks and eval passes. */
    static void bind_active_to_eval_context(const DeviceContext& eval_ctx,
                                             ActiveData& active) {
        active.LocalGasMechFBInfoTemp = eval_ctx.LocalGasMechFBInfoTemp;
        active.d_gas_iter             = eval_ctx.d_gas_iter;
        active.num_local_gas          = eval_ctx.num_local_gas;
        active.num_local_particles    = eval_ctx.num_local_particles;
        active.oracle_dry_run         = eval_ctx.oracle_dry_run;
        active.P_base                 = eval_ctx.P;
        active.CellP_base             = eval_ctx.CellP;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx, int j,
                                      const IdentitySidecar& id, const ActiveData& active) {
        NeighborData n{};
        n.neighbor_particle = &dctx.P[j];
        n.neighbor_cell     = (dctx.CellP != nullptr && dctx.P[j].Type == 0) ? &dctx.CellP[j] : nullptr;
        n.neighbor_index    = j;
        (void)id; (void)active;
        return n;
    }

    /* pair_kernel — runner-template port of the legacy per-pair body.
     *
     * Delegates the actual physics to mechanical_fb_pair_kernel (signature
     * refactored in milestone 3 to accept home/ghost gas-delta pointers +
     * num_local_gas + oracle_dry_run). The runner-port responsibilities here:
     *   1. Pre-filter (Pj.Type==0 / Mass>0, dp/r2, h_i/h_j gate, r2max gate)
     *      mirroring mechanical_fb_evaluate_gpu's inner neighbor loop.
     *   2. Route j-side writes: home gas → LocalGasMechFBInfoTemp[j];
     *      imported ghost → d_gas_iter[j-num_local_particles] (milestone 3.5
     *      ghost-writeback path; shipped to home ranks via custom bundle callback).
     *   3. Forward to mechanical_fb_pair_kernel with ownership-split + oracle
     *      params from ActiveData. */
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& i_active,
                            const NeighborData& neighbor,
                            AccumData& accum, NoScatter& /*scatter*/) {
        /* Per-mode active mask (codex r7 fix) — equivalent of legacy
         * mechanical_fb_evaluate_gpu:316. Sources in the toplevel superset
         * that are NOT active for THIS specific mode short-circuit the entire
         * pair body. Without this guard, mode>=0 calls for inactive sources
         * still increment N_injected per pair with otherwise-zero deltas. */
        if (!i_active.is_active_this_mode) return;
        if (i_active.local.Msne <= 0) return;
        if (i_active.local.KernelRadius <= 0) return;

        if (neighbor.neighbor_particle == nullptr) return;
        struct particle_data &Pj = *neighbor.neighbor_particle;
        if (Pj.Type != 0) return;
#ifdef HYDRO_MULTIFLUID_DM
        if (Pj.FluidType == FLUID_DM) return; /* skip dark-fluid neighbors */
#endif
        if (Pj.Mass <= 0) return;
        if (neighbor.neighbor_cell == nullptr) return;  /* gas-only safety */

        /* Position delta + r² filters (legacy mechanical_fb_gpu.cc:333-340
         * + mechanical_fb_pair_kernel:239-242). */
        Vec3<double> dp;
        dp[0] = (double)i_active.local.Pos[0] - (double)Pj.Pos[0];
        dp[1] = (double)i_active.local.Pos[1] - (double)Pj.Pos[1];
        dp[2] = (double)i_active.local.Pos[2] - (double)Pj.Pos[2];
        nearest_xyz(dp);
        const double r2 = dp.norm_sq();
        if (r2 <= 0) return;
        const double h2j = (double)Pj.KernelRadius * (double)Pj.KernelRadius;
        if ((r2 > i_active.source_mode.h2) && (r2 > h2j)) return;
        if (r2 > i_active.source_mode.r2max_phys) return;

        /* j-side routing: three-way split via mechfb_target_gas_delta inside
         * mechanical_fb_pair_kernel (mechanical_fb_functions.h). Home gas at
         * j < num_local_gas → LocalGasMechFBInfoTemp[j]; imported ghost at
         * j >= num_local_particles → d_gas_iter[j - num_local_particles];
         * the gas-only mask in load_neighbor rejects local non-gas before
         * we get here (defensive abort fires inside the helper if it doesn't).
         *
         * Milestone 3.5 (multi-rank Mode A): the d_gas_iter buffer is lazily
         * allocated and zeroed each iter by reset_per_iter_device_context
         * based on ghost_get_num_ghosts(); ghost_writeback_begin/end pair
         * (declared below, defined in mechfb_loop.cc) ships nonzero deltas
         * to home ranks via the mechfb_writeback_detail custom callback. */
        mechanical_fb_pair_kernel(
            i_active.local, i_active.source_mode, i_active.loop_iteration,
            neighbor.neighbor_index,
            i_active.P_base, i_active.CellP_base,
            i_active.scalars,
            /*home_gas_delta      */ i_active.LocalGasMechFBInfoTemp,
            /*ghost_gas_delta     */ i_active.d_gas_iter,
            /*num_local_gas       */ i_active.num_local_gas,
            /*num_local_particles */ i_active.num_local_particles,
            /*oracle_dry_run      */ i_active.oracle_dry_run,
            dp, r2, accum);
    }
};

/* ============================================================================
 * Toplevel-facing helpers (free functions; not Spec methods).
 *
 * mechfb_compute_num_modes — picks 3/4/5/6 per FIRE config.
 * mechfb_populate_aux_initial — fills Aux from caller-provided lists at the
 *   start of mechanical_fb_calc_toplevel (before the runner call).
 * mechfb_repack_per_active_local — host-side per-mode repack of MechFBLocalIn
 *   into the SharedSpace per_active_local buffer. Called from
 *   after_iter_global between iters; also called once initially in
 *   populate_device_context before the first iter.
 * ========================================================================== */
int  mechfb_compute_num_modes(void);
void mechfb_populate_aux_initial(MechFBSpec::Aux& aux,
                                 int num_active,
                                 struct MechFBGasDelta *LocalGasMechFBInfoTemp,
                                 int num_local_gas);
void mechfb_repack_per_active_local(const neighbor_loop_args& args,
                                    MechFBSpec::Aux& aux,
                                    int loop_iteration);

#endif /* GALSF_FB_MECHANICAL */

/* NOTE: toplevel helpers callable from non-GPU TUs (mechfb_alloc_local_gas_delta,
 * mechfb_free_local_gas_delta, mechfb_run_iterative) are declared in
 * galaxy_sf/galsf_gpu_decls.h — keep this header GPU-TU-only because it
 * includes mechanical_fb_functions.h whose inline pair_kernel body uses
 * Kokkos:: symbols. */
