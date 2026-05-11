/* mesh/test_iter_harness_loop.h — synthetic iterative-Spec validation
 * harness (Phase 4.B.0 step 3).
 *
 * IterHarnessSpec is a NON-PHYSICS synthetic Spec that exercises the
 * 4.B.0 iterative runner contracts before ags_density (3d.4) ports.
 * Compiled only under GIZMO_NLR_ITER_HARNESS_TEST; invoked at startup
 * via GIZMO_NLR_ITER_HARNESS_RUN=1 (begrun.cc).
 *
 * Design doc: OPEN_phase4_b0_step3_harness_design.md
 *
 * Slot-keyed test pattern in after_iter (validation points 1-5):
 *   slot 0: NeedsMore on iter 0 → Converged on iter 1.
 *   slot 1: AdjustRadius (h × 1.10) on iter 0 → Converged on iter 1.
 *           h × 1.10 exceeds Mode A's 1.05 buffer factor → triggers
 *           cached-CSR rebuild (validation point 8).
 *   slot 2: Converged on iter 0 (single-pass).
 *   slot 3: NeedsMore forever IF GIZMO_NLR_ITER_HARNESS_ABORT_TEST=1,
 *           else Converged on iter 0 (validation point 6 negative test).
 */
#ifndef TEST_ITER_HARNESS_LOOP_H
#define TEST_ITER_HARNESS_LOOP_H

/* Kokkos_Core MUST come BEFORE declarations/allvars.h. allvars.h pulls in
 * declarations/macros.h which #defines `terminate(...)`; that macro mangles
 * the C++ stdlib `<exception>` header's `std::terminate()` declaration that
 * Kokkos_Core.hpp transitively pulls in. Same pattern as sinks/sink_feed_loop.h. */
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"

#ifdef GIZMO_NLR_ITER_HARNESS_TEST
#include "neighbor_loop_runner.h"
#include "mode_b_local_walker.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Env predicates the harness consumes. Defined in test_iter_harness_loop.cc. */
bool gizmo_nlr_iter_harness_run_enabled(void);
bool gizmo_nlr_iter_harness_abort_test_enabled(void);
bool gizmo_nlr_iter_harness_empty_rank_enabled(void);
bool gizmo_nlr_iter_harness_audit_mangle_enabled(void);
bool gizmo_nlr_iter_harness_multi_subgroup_enabled(void);
bool gizmo_nlr_iter_harness_ghost_enabled(void);

/* Harness diagnostic counters (file-scope; reset per invocation). Defined
 * in test_iter_harness_loop.cc. apply_active_writeback fires per particle
 * index; we count by particle index. Range pre-registered by the driver
 * (g_harness_active_indices array); writeback hits look up the index. */
constexpr int IterHarnessMaxActives = 16;
extern int g_harness_writeback_count[IterHarnessMaxActives];
extern int g_harness_active_indices[IterHarnessMaxActives];
extern int g_harness_n_active_total;

/* ============================================================================
 * Compile-gated test telemetry struct (codex step-3 review 2026-05-10).
 *
 * Populated by run_neighbor_loop_iterative<IterHarnessSpec> at end-of-call
 * under GIZMO_NLR_ITER_HARNESS_TEST. Harness driver reads these post-call
 * to assert iterative control-flow VPs (1, 2, 4, 5, 11) without needing
 * friend access to NlrIterDriver or a public std::function callback on
 * the runner's API. Narrow, test-only surface.
 *
 * Reset to zero at start of each iterative call by the runner.
 * ========================================================================== */
struct IterHarnessTelemetry {
    int       last_iter_index;            /* drv.iter_index at end of outer loop */
    int       last_global_active_total;   /* drv.global_active_total */
    int       csr_rebuild_count;
    int       arena_acquire_count;
    int       csr_local_rebuild_count;
    int       final_dummy_jflag_prod;     /* drv.ctx.dummy_jflag — last-iter value */
    int       final_dummy_jflag_oracle;   /* drv.ctx_oracle.dummy_jflag — last-iter value */
    long long oracle_mismatch_count;
    int       oracle_enabled;             /* 1 if drv.oracle_enabled at end of call */
    /* Total pair count across all slots' final (convergence-iter) accums.
     * Used by Stage B step 2 to guard against vacuous oracle-parity passes
     * when the synthetic search radius is too small to find neighbors
     * (codex 2026-05-10 caveat). */
    long long total_pairs_prod;            /* sum accum_uvm[sg][slot].n_pairs */
    long long total_pairs_oracle;          /* sum accum_oracle_uvm[sg][slot].n_pairs */
    /* Per-slot scratch + status snapshot (only sg=0, slots 0..3 — single-subgroup
     * 4-active Stage A test pattern). Stage B extends if multi-subgroup VPs need it. */
    int       scratch_passes[4];           /* drv.scratch_uvm[0][slot].passes_so_far */
    int       final_status_prod[4];        /* drv.status_prod_uvm[0][slot] */
};
extern IterHarnessTelemetry g_iter_harness_telemetry;

/* ============================================================================
 * Per-pair physics types (file scope; trivially copyable). All synthetic —
 * no real physics. slot_id keys the test pattern; iter_seen is harness-
 * observable per-slot per-iter context.
 * ========================================================================== */
struct IterHarnessCallScalars {
    NlrCommonScalars common;
    int              harness_run_id;        /* increments per harness invocation */
};

struct IterHarnessActiveData {
    Vec3<double>             pos;
    MyIDType                 id;
    double                   h_search;
    int                      slot_id;
    int                      iter_seen;
    int                      oracle_brute_pass_active;  /* captured from ctx at load_active time */
    IterHarnessCallScalars   scalars;
};

struct IterHarnessNeighborData {
    Vec3<double> pos;
    MyIDType     id;
    int          j_index;
};

/* AccumData: counts pairs + deterministic accumulator + j-side counter.
 * compare_accum requires production==brute identical. */
struct IterHarnessAccumData {
    int    n_pairs;
    double accum_value;
    int    jflag;
};

/* IterScratch: persists across iters (TRAP 8). */
struct IterHarnessIterScratch {
    int    passes_so_far;
    double last_radius_seen;
};

/* DeviceContext extends base with:
 *   - dummy_jflag:                exercised by reset_per_iter_device_context
 *                                  + after_iter_global readback (VP 11).
 *   - oracle_brute_pass_active:   exercised by set_oracle_brute_pass +
 *                                  pair_kernel suppression (VP 9).
 */
struct IterHarnessDeviceContext : public NeighborLoopDeviceContextBase {
    int  dummy_jflag;
    int  oracle_brute_pass_active;
};

/* ============================================================================
 * Inline pair body (host + device).
 *
 * Deterministic per-pair contribution — identical regardless of search
 * backend (tree vs brute), so production-vs-oracle compare_accum is
 * defined: any divergence = bug.
 *
 * J-side counter: only increments on production walks. set_oracle_brute_pass
 * suppresses it on the brute pass — exercises G9 RAII guard.
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
void iter_harness_pair_body(const IterHarnessActiveData& a,
                                     const IterHarnessNeighborData& n,
                                     IterHarnessAccumData& acc)
{
    if (a.id == n.id) return;     /* self-skip */
    Vec3<double> dx;
    dx[0] = n.pos[0] - a.pos[0];
    dx[1] = n.pos[1] - a.pos[1];
    dx[2] = n.pos[2] - a.pos[2];
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    if (r2 > a.h_search * a.h_search) return;

    acc.n_pairs     += 1;
    /* Slot-id-keyed deterministic contribution (independent of FP order
     * and backend). */
    acc.accum_value += 1.0 / (1.0 + (double)(a.slot_id) + (double)(n.j_index));
    if (!a.oracle_brute_pass_active) {
        acc.jflag += 1;
    }
}

/* ============================================================================
 * IterHarnessSpec — Iterative synthetic Spec.
 * ========================================================================== */
struct IterHarnessSpec {
    /* ============ PHYSICS BLOCK ============ */
    static constexpr const char *loop_name = "iter_harness";
    static constexpr int                    search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int           neighbor_type_mask = 0xFFu;
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;
    static constexpr WritePattern           write_pattern      = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind          sidx_cache_kind    = SidxCacheKind::AllTypes;
    static constexpr double accum_tolerance   = 1e-12;
    static constexpr double radius_tolerance  = 1e-9;
    static constexpr int    max_iters         = 10;
    static constexpr double mode_a_csr_buffer_factor = 1.05;
    static constexpr bool   uses_ghost_write_detector = false;
    static constexpr bool   uses_ghost_writeback      = false;

    using CallScalars   = IterHarnessCallScalars;
    using ActiveData    = IterHarnessActiveData;
    using NeighborData  = IterHarnessNeighborData;
    using AccumData     = IterHarnessAccumData;
    using ScatterData   = NoScatter;
    using IdentityFields = NoIdentity;
    using DeviceContext = IterHarnessDeviceContext;
    using IterControl   = Iterative;
    using IterScratch   = IterHarnessIterScratch;
    using SupportsSubgroups = std::true_type;     /* VP 10/12 multi-subgroup */

    struct Aux { int dummy; };

    /* Active-particle predicate — harness driver provides actives explicitly
     * via nlr_build_active_list-free path; this is a stub. */
    static bool is_active(int /*i*/) { return false; }

    /* ============ Hooks ============ */
    static CallScalars populate_call_scalars(const neighbor_loop_args& /*args*/) {
        CallScalars cs{};
        cs.common.cf_atime = All.cf_atime;
        cs.common.cf_a2inv = All.cf_a2inv;
        cs.common.cf_a3inv = All.cf_a3inv;
        cs.harness_run_id  = 0;
        return cs;
    }

    static double search_radius(const neighbor_loop_args& /*args*/,
                                  int /*slot*/, int /*i*/) {
        /* SYNTHETIC HARNESS RADIUS — codex 2026-05-10 rationale:
         *
         * The harness runs at end-of-begrun, BEFORE set_softenings() has
         * populated All.ForceSoftening[]. Using `All.ForceSoftening[0]`
         * here yielded h=0 → AdjustRadius writeback `h*1.10 = 0` → the
         * Mode A (b.5) buffer-exceedance trigger never fired (0 > 0 is
         * false) → VP 3 / VP 8 rebuild assertions silently failed.
         *
         * Fix: hardcode a deterministic nonzero radius. Constraints:
         *   1. h > 0 (so AdjustRadius creates a nonzero new radius).
         *   2. h * 1.10 must exceed Mode A's 1.05 buffer factor (always
         *      true: 1.10/1.05 = 1.048 > 1).
         *   3. The radius must produce > 0 candidate pairs in the m11i
         *      IC so VP 9 (oracle parity) and VP 11 (j-side counter
         *      symmetry) don't pass vacuously. Codex 2026-05-10 caveat:
         *      h must be large enough to actually exercise pair work.
         *
         * Value `100.0` code units (~150 kpc for m11i internal units
         * where 1 code unit ≈ 1.47 kpc) is comparable to the virial
         * radius of an m11i halo. Particles 0..3 on each rank after
         * domain decomposition can land in arbitrary regions (low-
         * density IGM, halo edges, etc.); a smaller value risked zero
         * neighbors and vacuous oracle parity (verified empirically:
         * h=1.0 gave total_pairs=0 in step 2 first attempt). Brute
         * oracle walks the full O(NumPart) particle set per active
         * regardless of h, so larger h only widens the tree-walker
         * candidate set and doesn't significantly increase brute cost.
         *
         * We deliberately do NOT call set_softenings() earlier just for
         * this test (codex 2026-05-10): changing startup ordering for a
         * synthetic test is the wrong direction. */
        return 100.0;      /* code units; synthetic — see comment above */
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx_base,
                                    int active_slot, int i, double h_search,
                                    const CallScalars& scalars) {
        const IterHarnessDeviceContext& ctx =
            static_cast<const IterHarnessDeviceContext&>(ctx_base);
        ActiveData a{};
        a.pos[0] = ctx.P[i].Pos[0];
        a.pos[1] = ctx.P[i].Pos[1];
        a.pos[2] = ctx.P[i].Pos[2];
        a.id        = ctx.P[i].ID;
        a.h_search  = h_search;
        a.slot_id   = active_slot;      /* slot drives test pattern */
        a.iter_seen = 0;
        a.oracle_brute_pass_active = ctx.oracle_brute_pass_active;
        a.scalars   = scalars;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx_base,
                                        int j, const IdentitySidecar& /*id*/,
                                        const ActiveData& /*active*/) {
        const IterHarnessDeviceContext& ctx =
            static_cast<const IterHarnessDeviceContext&>(ctx_base);
        NeighborData n{};
        n.pos[0]  = ctx.P[j].Pos[0];
        n.pos[1]  = ctx.P[j].Pos[1];
        n.pos[2]  = ctx.P[j].Pos[2];
        n.id      = ctx.P[j].ID;
        n.j_index = j;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& a) {
        a.n_pairs     = 0;
        a.accum_value = 0.0;
        a.jflag       = 0;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& n,
                              AccumData& acc, ScatterData& /*scatter*/) {
        iter_harness_pair_body(a, n, acc);
    }

    static void merge_accum(AccumData& dst, const AccumData& src) {
        dst.n_pairs     += src.n_pairs;
        dst.accum_value += src.accum_value;
        dst.jflag       += src.jflag;
    }

    static double compare_accum(const AccumData& tree, const AccumData& brute) {
        double d = (double)std::abs(tree.n_pairs - brute.n_pairs);
        d       += std::abs(tree.accum_value - brute.accum_value);
        /* jflag is suppressed on the brute pass via set_oracle_brute_pass —
         * brute jflag should always be 0. We compare the NON-jflag fields
         * for parity; jflag invariant ("brute == 0") is checked by the
         * harness driver separately (VP 9). */
        return d;
    }

    static void apply_active_writeback(const neighbor_loop_args& /*args*/,
                                         int /*active_slot*/, int i,
                                         const AccumData& /*accum*/) {
        /* VP 7: count writeback fires per particle index. Map i → slot
         * by linear scan of registered actives (small N, 8-16 max). */
        for (int s = 0; s < g_harness_n_active_total; s++) {
            if (g_harness_active_indices[s] == i) {
                g_harness_writeback_count[s]++;
                return;
            }
        }
    }

    /* ============ DeviceContext hooks ============ */
    static void populate_device_context(const neighbor_loop_args& /*args*/,
                                          DeviceContext& ctx) {
        ctx.dummy_jflag = 0;
        ctx.oracle_brute_pass_active = 0;
    }
    static void cleanup_device_context(const neighbor_loop_args& /*args*/,
                                         DeviceContext& /*ctx*/) {
        /* No owning resources. */
    }
    static void reset_per_iter_device_context(const neighbor_loop_args& /*args*/,
                                                DeviceContext& ctx, int /*iter_index*/) {
        /* VP 11: per-iter zero of the synthetic counter. The 2c.4 ctx_oracle
         * symmetric reset means this runs on BOTH production and oracle
         * ctx per outer iter. */
        ctx.dummy_jflag = 0;
    }
    static void set_oracle_brute_pass(DeviceContext& ctx, bool on) {
        ctx.oracle_brute_pass_active = on ? 1 : 0;
    }

    /* ============ Iterative hooks ============ */
    static IterResult after_iter(const AfterIterContext<IterHarnessSpec>& ctx,
                                   const AccumData& /*accum*/) {
        ctx.scratch.passes_so_far++;
        ctx.scratch.last_radius_seen = ctx.h_search_current;
        const int slot = ctx.subgroup_slot;
        switch (slot) {
            case 0: return (ctx.iter_index >= 1)
                          ? IterResult{IterStatus::Converged,    0.0}
                          : IterResult{IterStatus::NeedsMore,    0.0};
            case 1: return (ctx.iter_index >= 1)
                          ? IterResult{IterStatus::Converged,    0.0}
                          : IterResult{IterStatus::AdjustRadius, ctx.h_search_current * 1.10};
            case 2: return IterResult{IterStatus::Converged, 0.0};
            case 3: return (gizmo_nlr_iter_harness_abort_test_enabled())
                          ? IterResult{IterStatus::NeedsMore, 0.0}
                          : IterResult{IterStatus::Converged, 0.0};
            default: return IterResult{IterStatus::Converged, 0.0};
        }
    }
};

/* ============================================================================
 * IterHarnessGhostSpec — Stage B step 3 (codex 2026-05-10).
 *
 * Iterative spec with `uses_ghost_writeback = true`, exercising the
 * ghost-writeback machinery via existing `PARTICLE_MAX` op on a new
 * compile-gated field `particle_data::NlrIterHarnessJFlag` (declared
 * only under GIZMO_NLR_ITER_HARNESS_TEST in declarations/particle_data.h).
 *
 * Design constraints (codex non-negotiable #3, 2026-05-10):
 *   - Uses ONLY the existing PARTICLE_MAX op (no harness-only new ops).
 *   - Compile-gated end-to-end; production builds see none of this.
 *   - Pair_kernel writes a deterministic value to neighbor P[j] on
 *     production; oracle brute-pass suppresses the write. This makes
 *     ghost-writeback reverse-comm observable AND tests that
 *     set_oracle_brute_pass correctly suppresses j-side writes (which
 *     existing IterHarnessSpec's narrow VP 11 ctx-reset test cannot
 *     prove — codex step-2 review 2026-05-10).
 *
 * Test pattern: same slot-keyed after_iter as IterHarnessSpec so the
 * runner's iterative control-flow is identical. ScatterData = NoScatter
 * (no per-active scratch; j-side write is direct).
 * ========================================================================== */

struct IterHarnessGhostNeighborData {
    Vec3<double>     pos;
    MyIDType         id;
    int              j_index;
    particle_data   *p_ptr;          /* non-const — pair_kernel atomic write target */
    int              oracle_dry_run; /* propagated from ctx.oracle_brute_pass_active */
};

/* Inline pair body for the ghost spec. Deterministic per-pair contribution
 * to AccumData (matches IterHarnessSpec) plus an atomic write to neighbor
 * P[j].NlrIterHarnessJFlag = 1, guarded by !oracle_dry_run. The write
 * targets HOME particles directly (j < num_total_local) and GHOST copies
 * (j >= num_total_local); the ghost-writeback reverse-comm then merges
 * the ghost-copy values back to their owning rank via PARTICLE_MAX. */
KOKKOS_INLINE_FUNCTION
void iter_harness_ghost_pair_body(const IterHarnessActiveData& a,
                                   const IterHarnessGhostNeighborData& n,
                                   IterHarnessAccumData& acc)
{
    if (a.id == n.id) return;
    Vec3<double> dx;
    dx[0] = n.pos[0] - a.pos[0];
    dx[1] = n.pos[1] - a.pos[1];
    dx[2] = n.pos[2] - a.pos[2];
    double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    if (r2 > a.h_search * a.h_search) return;

    acc.n_pairs     += 1;
    acc.accum_value += 1.0 / (1.0 + (double)(a.slot_id) + (double)(n.j_index));
    if (!a.oracle_brute_pass_active) {
        acc.jflag += 1;
        /* j-side write to neighbor's NlrIterHarnessJFlag. atomic_exchange
         * matches the sink_feed pattern (sink_feed_loop.h pair body).
         * With multiple actives potentially writing to the same neighbor,
         * "last writer wins" with value 1 is fine — the PARTICLE_MAX
         * ghost-writeback merges across ranks via max anyway. */
        Kokkos::atomic_exchange(&n.p_ptr->NlrIterHarnessJFlag, 1);
    }
    /* Brute-pass: NO P[j] write. Sets the suppression test up: post-call,
     * P[j].NlrIterHarnessJFlag != 0 only for production-walk neighbors. */
}

struct IterHarnessGhostSpec {
    /* Physics block */
    static constexpr const char *loop_name = "iter_harness_ghost";
    static constexpr int                    search_mode        = MODE_B_SEARCH_SYMMETRIC;
    static constexpr unsigned int           neighbor_type_mask = 0xFFu;
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;
    static constexpr WritePattern           write_pattern      = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind          sidx_cache_kind    = SidxCacheKind::AllTypes;
    static constexpr double accum_tolerance   = 1e-12;
    static constexpr double radius_tolerance  = 1e-9;
    static constexpr int    max_iters         = 10;
    static constexpr double mode_a_csr_buffer_factor = 1.05;
    /* THIS is the distinguishing flag — exercises ghost-writeback machinery. */
    static constexpr bool   uses_ghost_write_detector = true;
    static constexpr bool   uses_ghost_writeback      = true;

    using CallScalars       = IterHarnessCallScalars;
    using ActiveData        = IterHarnessActiveData;
    using NeighborData      = IterHarnessGhostNeighborData;
    using AccumData         = IterHarnessAccumData;
    using ScatterData       = NoScatter;
    using IdentityFields    = NoIdentity;
    using DeviceContext     = IterHarnessDeviceContext;
    using IterControl       = Iterative;
    using IterScratch       = IterHarnessIterScratch;
    using SupportsSubgroups = std::true_type;

    struct Aux { int dummy; };

    static bool is_active(int /*i*/) { return false; }

    /* Lifecycle hooks — definitions in test_iter_harness_loop.cc. */
    static void ghost_write_detector_begin (const struct neighbor_loop_args&,
                                              const struct NeighborLoopPlan&);
    static void ghost_write_detector_end   (const struct neighbor_loop_args&,
                                              const struct NeighborLoopPlan&);
    static void ghost_writeback_begin      (const struct neighbor_loop_args&,
                                              const struct NeighborLoopPlan&);
    static void ghost_writeback_end        (const struct neighbor_loop_args&,
                                              const struct NeighborLoopPlan&);

    static CallScalars populate_call_scalars(const neighbor_loop_args& /*args*/) {
        CallScalars cs{};
        cs.common.cf_atime = All.cf_atime;
        cs.common.cf_a2inv = All.cf_a2inv;
        cs.common.cf_a3inv = All.cf_a3inv;
        cs.harness_run_id  = 1;        /* distinguish from IterHarnessSpec calls */
        return cs;
    }

    static double search_radius(const neighbor_loop_args& /*args*/,
                                  int /*slot*/, int /*i*/) {
        /* The ghost spec needs h LARGE ENOUGH to make rank-0's actives
         * find ghost neighbors imported from rank 1's domain — otherwise
         * the Mode A reverse-comm reverse-comm test passes vacuously
         * (codex 2026-05-10 step-3 review). With m11i's BoxSize ≈ 58480
         * code units and np=2 domain decomp, h=100 (the IterHarnessSpec
         * value) kept all neighbors on rank 0's home side.
         *
         * Use BoxSize / 2 so rank-0 actives are GUARANTEED to import
         * ghosts from rank 1 regardless of where decomp put the active
         * particle indices [0..3]. Memory cost: rank 0 imports a sizable
         * fraction of rank 1's particles as ghosts; tested manageable on
         * Vista with m11i np=2. */
        /* Small deterministic radius matching IterHarnessSpec. For the
         * Mode A reverse-comm test the harness driver COLOCATES rank-0
         * actives at rank-1's P[0].Pos via a one-shot position override
         * (codex 2026-05-10), making h=100 sufficient to import rank-1's
         * target particle as a ghost regardless of decomp geometry. */
        return 100.0;
    }

    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const NeighborLoopDeviceContextBase& ctx_base,
                                    int active_slot, int i, double h_search,
                                    const CallScalars& scalars) {
        const IterHarnessDeviceContext& ctx =
            static_cast<const IterHarnessDeviceContext&>(ctx_base);
        ActiveData a{};
        a.pos[0] = ctx.P[i].Pos[0];
        a.pos[1] = ctx.P[i].Pos[1];
        a.pos[2] = ctx.P[i].Pos[2];
        a.id        = ctx.P[i].ID;
        a.h_search  = h_search;
        a.slot_id   = active_slot;
        a.iter_seen = 0;
        a.oracle_brute_pass_active = ctx.oracle_brute_pass_active;
        a.scalars   = scalars;
        return a;
    }

    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx_base,
                                        int j, const IdentitySidecar& /*id*/,
                                        const ActiveData& /*active*/) {
        const IterHarnessDeviceContext& ctx =
            static_cast<const IterHarnessDeviceContext&>(ctx_base);
        NeighborData n{};
        n.pos[0]          = ctx.P[j].Pos[0];
        n.pos[1]          = ctx.P[j].Pos[1];
        n.pos[2]          = ctx.P[j].Pos[2];
        n.id              = ctx.P[j].ID;
        n.j_index         = j;
        n.p_ptr           = &ctx.P[j];          /* non-const home-or-ghost pointer */
        n.oracle_dry_run  = ctx.oracle_brute_pass_active;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    static void zero_accum(AccumData& a) {
        a.n_pairs = 0; a.accum_value = 0.0; a.jflag = 0;
    }

    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& n,
                              AccumData& acc, ScatterData& /*scatter*/) {
        iter_harness_ghost_pair_body(a, n, acc);
    }

    static void merge_accum(AccumData& dst, const AccumData& src) {
        dst.n_pairs     += src.n_pairs;
        dst.accum_value += src.accum_value;
        dst.jflag       += src.jflag;
    }

    static double compare_accum(const AccumData& tree, const AccumData& brute) {
        double d  = (double)std::abs(tree.n_pairs - brute.n_pairs);
        d        += std::abs(tree.accum_value - brute.accum_value);
        /* Ignore jflag here too — brute writes 0 by construction (suppression). */
        return d;
    }

    static void apply_active_writeback(const neighbor_loop_args& /*args*/,
                                         int /*active_slot*/, int /*i*/,
                                         const AccumData& /*accum*/) {
        /* No per-active home-side writeback. The interesting writeback is
         * the j-side ghost-writeback bundle below, which the runner fires
         * via ghost_writeback_begin/end hooks. */
    }

    /* DeviceContext hooks — identical semantics to IterHarnessSpec. */
    static void populate_device_context(const neighbor_loop_args& /*args*/,
                                          DeviceContext& ctx) {
        ctx.dummy_jflag = 0;
        ctx.oracle_brute_pass_active = 0;
    }
    static void cleanup_device_context(const neighbor_loop_args& /*args*/,
                                         DeviceContext& /*ctx*/) {}
    static void reset_per_iter_device_context(const neighbor_loop_args& /*args*/,
                                                DeviceContext& ctx, int /*iter_index*/) {
        ctx.dummy_jflag = 0;
    }
    static void set_oracle_brute_pass(DeviceContext& ctx, bool on) {
        ctx.oracle_brute_pass_active = on ? 1 : 0;
    }

    static IterResult after_iter(const AfterIterContext<IterHarnessGhostSpec>& ctx,
                                   const AccumData& /*accum*/) {
        /* Same slot-keyed pattern as IterHarnessSpec so the iterative
         * control-flow is identical (preserves VPs 1, 2, 4, 5 semantics
         * indirectly while we exercise ghost writeback). */
        ctx.scratch.passes_so_far++;
        ctx.scratch.last_radius_seen = ctx.h_search_current;
        const int slot = ctx.subgroup_slot;
        switch (slot) {
            case 0: return (ctx.iter_index >= 1)
                          ? IterResult{IterStatus::Converged,    0.0}
                          : IterResult{IterStatus::NeedsMore,    0.0};
            case 1: return (ctx.iter_index >= 1)
                          ? IterResult{IterStatus::Converged,    0.0}
                          : IterResult{IterStatus::AdjustRadius, ctx.h_search_current * 1.10};
            case 2: return IterResult{IterStatus::Converged, 0.0};
            case 3: return IterResult{IterStatus::Converged, 0.0};
            default: return IterResult{IterStatus::Converged, 0.0};
        }
    }
};

/* ============================================================================
 * Harness driver entry point. Invoked once at startup if
 * GIZMO_NLR_ITER_HARNESS_RUN=1; emits HARNESS_VP_NN PASS|FAIL lines and
 * then calls endrun(0) (clean shutdown) so production sim never runs.
 * ========================================================================== */
void run_iter_harness_tests(void);

#endif /* GIZMO_NLR_ITER_HARNESS_TEST */
#endif /* TEST_ITER_HARNESS_LOOP_H */
