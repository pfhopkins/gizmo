/* mesh/neighbor_loop_runner.h — generic neighbor-loop runner for GIZMO.
 *
 * Single source of truth for the neighbor-loop interface every GIZMO physics
 * loop is migrated onto. The interface is the binding contract; new patterns
 * (write modes, iterative loops, identity sidecars) are DECLARED in the
 * interface so callers don't reshape when their needs are added; runtime
 * policy on unsupported features is documented below (TL;DR: Mode A fallback
 * in production; abort if force-Mode-B; static_assert only for internally
 * inconsistent specs).
 *
 * The user-facing copy-pasteable Spec skeleton is below ("NeighborLoopSpec —
 * copy-pasteable skeleton for a new physics loop"). For a worked example,
 * read sinks/sink_env1_loop.h.
 *
 * ============================================================================
 * Three-epoch host/device staging contract
 * ============================================================================
 *
 *   The runner separates per-active staging into three timing points so a
 *   spec can match the byte-exact state epoch the legacy device kernel saw.
 *   For sink_env1 these epochs reproduce the prior sink_environment.cc /
 *   sink_environment_gpu.cc timing exactly.
 *
 *   (1) HOST, pre-arena:
 *         search_radius(args, slot, i) -> double
 *       Reads external state (e.g. P[i].KernelRadius) BEFORE arena_acquire
 *       / drift / freshness. Equivalent to the radii staging that legacy
 *       callers performed in their host loop just after building the
 *       active-particle list. Result is staged into a UVM radii[num_active]
 *       array; Mode A's gpu_ngb_list_build receives it directly, Mode B's
 *       walker reads it per-query.
 *
 *   (2) HOST, pre-arena:
 *         populate_call_scalars(args) -> CallScalars
 *       Captures per-call scalar globals (cosmology factors, gravity
 *       constant, kernel radii, etc.) into a typed POD struct. Captured
 *       once per call; replicated by-value into the device-side lambda
 *       capture so the kernel never reaches into globals. CallScalars
 *       must be std::is_trivially_copyable_v.
 *
 *   (3) DEVICE (or host for Mode B walker), post-NGL-build:
 *         KOKKOS_INLINE_FUNCTION
 *         load_active(ctx, slot, i, h_search, cs) -> ActiveData
 *       Reads ctx.P[i] / ctx.CellP[i] (UVM-resident, post-arena/drift)
 *       and combines with the host-staged h_search and CallScalars.
 *       Runner launches a tiny Kokkos parallel_for that fills
 *       ActiveData[num_active] in UVM. Mode A's pair-kernel launch reads
 *       this array; Mode B/Brute walker calls the SAME function host-side
 *       per query. Same KOKKOS_INLINE_FUNCTION on both paths.
 *
 *   NeighborData is DEVICE-CALLABLE (and host-callable in Mode B):
 *     KOKKOS_INLINE_FUNCTION
 *     load_neighbor(ctx, j, id, active) -> NeighborData
 *     Receives a DeviceContext containing UVM-resident pointers (P, CellP,
 *     plus spec-specific extension if Spec::DeviceContext extends the
 *     base). Same function called from device (Mode A) and host (Mode B,
 *     Brute) — KOKKOS_INLINE_FUNCTION compiles to inline on both.
 *     Implementation must use only device-safe constructs (no std:: I/O,
 *     no host-only helpers); same constraint as the pair_kernel.
 *
 *   Accumulator zeroing:
 *     KOKKOS_INLINE_FUNCTION zero_accum(AccumData& out)
 *     Spec-owned. Generic runner does NOT bake in memset on AccumData;
 *     the spec decides (e.g. memset for POD, structured zeroing for
 *     non-trivial members).
 *
 *   Writebacks are HOST-ONLY:
 *     apply_active_writeback(args, slot, i, accum)
 *     apply_neighbor_writeback(args, j, scatter)   // scatter modes only
 *     After dispatch returns, the runner copies AccumData back to host
 *     (no-op for Mode B/Brute; UVM-coherent for Mode A) and applies
 *     writebacks on the host. Spec is responsible for any further state
 *     scatter (e.g. SinkTempInfo population).
 *
 *   pair_kernel signature (single-source-of-truth lever; contract §"Pair kernel signature"):
 *     KOKKOS_INLINE_FUNCTION
 *     static void pair_kernel(const ActiveData& a,
 *                             const NeighborData& nb,
 *                             AccumData&  active_out,
 *                             ScatterData& nb_out);
 *   Both outputs ALWAYS in the signature. ScatterData=NoScatter for
 *   ActiveReduceOnly compiles to zero-cost. Kernel writes both sides of
 *   any pair interaction in ONE place — never split into "active-side
 *   kernel" + "neighbor-side kernel" (the SPIKE-duplicate trap we are
 *   explicitly killing).
 *
 *   No `int j` parameter. The kernel cannot tell whether nb came from local
 *   P[j], a Mode A ghost, a Mode B remote reply, or brute. Identity arrives
 *   via the spec's NeighborData (ordinary fields) or via the runner-
 *   populated IdentitySidecar passed to load_neighbor (only fields the
 *   spec's IdentityFields opts in to are valid; others are zero).
 *
 *   Self-skip on identity: physics-meaningful fields like P[j].ID for "is
 *   this the same particle I'm asking about?" SHOULD live as a normal field
 *   on NeighborData, loaded by load_neighbor — they're physics, not
 *   identity in the contract sense. IdentityFields is reserved for
 *   transport-aware metadata (owner_rank, local_index for ownership/
 *   tie-breaking when the same i can be reached from multiple paths).
 *
 * ============================================================================
 * Dispatch policy (the three behaviors)
 * ============================================================================
 *
 *   1. Internally inconsistent spec (compile-time): static_assert at the
 *      top of run_neighbor_loop<Spec> rejects (e.g. ActiveReduceOnly with
 *      non-empty ScatterData; iterative loop without after_iter; etc.).
 *      Caller must fix the spec.
 *
 *   2. Force-Mode-B + unimplemented Mode B feature (runtime abort): when
 *      GIZMO_NLR_FORCE_MODEB=1 OR Spec::force_modeb_required = true, and
 *      the dispatch chooses Mode B but the runner doesn't yet implement
 *      that combination, the runner aborts loudly. This is the gate for
 *      unit-test suites forcing the tiny-N path.
 *
 *   3. Default production with unimplemented Mode B feature (runtime
 *      fallback): dispatch falls through to Mode A and logs a one-time
 *      warning identifying the (loop_name, missing-feature) pair. Lets
 *      callers adopt the runner immediately and pick up Mode B as the
 *      runner gains coverage.
 *
 *   GIZMO_NLR_FORCE_MODEA=1 forces Mode A globally regardless of spec or
 *   threshold (regression baseline / sanity).
 *
 * ============================================================================
 * Per-loop spec carries:
 *
 *     - identifier:       loop_name (drives env gates / PHASE0 labels)
 *     - typed payloads:   ActiveData, NeighborData, AccumData, ScatterData
 *     - device context:   DeviceContext (defaults to base; spec extends if needed)
 *     - search spec:      search_mode, neighbor_type_mask, radius_policy,
 *                         search_radius(active)
 *     - writeback spec:   write_pattern + reduction ops via apply_*_writeback
 *     - identity opt-in:  IdentityFields (defaults NoIdentity)
 *     - iter driver:      IterControl after_iter() (defaults NotIterative)
 *     - oracle compare:   compare_accum, accum_tolerance (compare_scatter +
 *                         scatter_tolerance for scatter specs)
 *     - hooks (host):     search_radius, populate_call_scalars,
 *                         apply_active_writeback,
 *                         apply_neighbor_writeback (scatter only),
 *                         populate_device_context (optional extension)
 *     - hooks (device):   load_neighbor, pair_kernel
 *
 * Generic runner does:
 *     - dispatch:  Mode A vs Mode B vs Brute — runtime choice from
 *                  Allreduce'd Nactive vs threshold + force-mode env
 *     - search:    coarse predicate, type mask + per-type hmax pruning
 *     - transport: ghost (Mode A) or peer-to-peer (Mode B)
 *     - drift:     lazy drift_particle on j candidates pre-kernel
 *     - oracle:    GIZMO_<LOOP>_ORACLE=1 → run Mode B + Brute, compare via
 *                  Spec::compare_accum within Spec::accum_tolerance
 *     - timing:    PHASE0 instrumentation per loop_name
 *
 * Architecture binding contract:
 *   ~/.claude/memory/reference_neighbor_loop_contract.md
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
#ifndef GIZMO_NEIGHBOR_LOOP_RUNNER_H
#define GIZMO_NEIGHBOR_LOOP_RUNNER_H

#include <cstdint>
#include <type_traits>
#include "mode_b_local_walker.h"  /* SearchMode, RadiusPolicy already declared */

/* ============================================================================
 * Reused enums:
 *   SearchMode:    MODE_B_SEARCH_ONEWAY, MODE_B_SEARCH_SYMMETRIC
 *   RadiusPolicy:  mode_b_radius_policy_t (MODE_B_RADIUS_GAS_KERNEL, …)
 * ========================================================================== */

/* ============================================================================
 * Writeback specification
 * ========================================================================== */

enum class WritePattern : int {
    ActiveReduceOnly       = 0,  /* AccumData filled; ScatterData == NoScatter */
    SymmetricPairScatter   = 1,  /* both filled */
    NeighborScatter        = 2,  /* AccumData == NoAccum; ScatterData filled */
    GhostWritebackRequired = 3,  /* j may live on remote rank; writeback channel */
};

/* ============================================================================
 * Per-field reduction ops
 * ========================================================================== */

enum class ReductionOp : int {
    AssignAdd = 0, Assign = 1, Max = 2, Min = 3,
    AtomicAdd = 4, AtomicMax = 5, AtomicMin = 6, AtomicCAS = 7,
};

/* ============================================================================
 * RemoteEvalMode (Mode B comm strategy)
 * ========================================================================== */

enum class RemoteEvalMode : int {
    RemoteComputesAccum             = 0,
    ReturnNeighborData              = 1,
    ReturnNeighborDataWithWriteback = 2,
};

/* ============================================================================
 * Identity sidecar
 * ========================================================================== */

struct NoIdentity {};
struct IdField_ID {};
struct IdField_LocalIndex {};
struct IdField_OwnerRank {};

template <typename... Tags>
struct WithFields {};

struct IdentitySidecar {
    unsigned long long id;        /* valid iff IdField_ID requested */
    int                local_idx;  /* valid iff IdField_LocalIndex requested */
    int                owner_rank;/* valid iff IdField_OwnerRank requested.
                                   * Mode A ghost-import metadata MUST carry
                                   * owner_rank if any spec declares this. */
};

/* ============================================================================
 * SidxCacheKind — Mode A SIDX (spatial-index cache) selection
 *
 * Each spec declares which step-persistent SIDX cache its NGL build should
 * reuse. Caches are step-scoped and shared across cooperating callers
 * (e.g. sink_env1, sink_feed, sink_swk all share the all-types cache to
 * amortize the SFC sort + BVH build). The runner resolves the kind to a
 * gpu_spatial_index_t* internally; specs never name the global accessor.
 *
 * 3c.1: only AllTypes is implemented. Other kinds abort with a clear
 * message until the corresponding callers land.
 * ========================================================================== */

enum class SidxCacheKind : int {
    AllTypes = 0,   /* gpu_step_sidx_alltypes_ptr — sink_env1/feed/swk shared */
    /* future: PerSpecMask (3d), None (3f) — implement when first caller arrives. */
};

/* ============================================================================
 * DeviceContext base
 *
 * Specs may extend by typedef-ing `DeviceContext` to a struct that publicly
 * inherits NeighborLoopDeviceContextBase and adds spec-specific UVM pointers.
 * If extension is used, the spec MUST also provide:
 *
 *   static void populate_device_context(const neighbor_loop_args& args,
 *                                       DeviceContext& ctx);
 *
 * The runner detects extension via type-trait check on Spec::DeviceContext
 * and conditionally invokes populate_device_context.
 * ========================================================================== */

struct NeighborLoopDeviceContextBase {
    struct particle_data *P;
    struct gas_cell_data *CellP;
    int                   num_total;
};

/* ============================================================================
 * NlrCommonScalars — cosmology + gravity scalars almost every Spec needs.
 *
 * Specs include this by composition in their CallScalars struct:
 *
 *     struct MyLoopCallScalars {
 *         NlrCommonScalars common;
 *         double           my_loop_specific_scalar;
 *     };
 *
 *     static MyLoopCallScalars populate_call_scalars(...) {
 *         MyLoopCallScalars s;
 *         s.common = nlr_common_scalars_from_all();
 *         s.my_loop_specific_scalar = ...;
 *         return s;
 *     }
 *
 * Trivially copyable — rides the device lambda capture and any cross-rank
 * transport without special handling. Not opt-out: missing-when-needed is
 * silent floating-point corruption, so the cost of always populating these
 * (a few doubles per call) is the safer default.
 *
 * Slightly overcomplete by design: includes fields not strictly needed by
 * every loop, so a new Spec author copying this template gets the obvious
 * cosmology factors without remembering to re-derive them.
 *
 * Unit conversion factors (All.UnitMass_in_g, UnitLength_in_cm, etc.) are
 * NOT in this struct. They are invariant for the run, so a parallel
 * NlrUnitScalars helper can be added when the first loop needs them in a
 * device-side gas-physics helper. Until then, host-side code reads
 * All.Unit* directly.
 * ========================================================================== */
struct NlrCommonScalars {
    double cf_atime;
    double cf_a2inv;
    double cf_a3inv;
    double cf_hubble_a;
    double newton_G;                /* All.G — Newton's gravitational constant in code units */
    double hubble;                  /* All.HubbleParam */
    int    comoving_integration_on;
};

NlrCommonScalars nlr_common_scalars_from_all(void);

/* Forward declaration — full struct neighbor_loop_args is defined later in
 * this header. The DeviceContext trait helpers below need it as a parameter
 * type only (no member access), so the forward decl is sufficient. */
struct neighbor_loop_args;

/* ============================================================================
 * DeviceContext extension trait (Phase 4.A.0)
 *
 * Detects whether a Spec extends DeviceContext beyond NeighborLoopDeviceContextBase.
 * When true, the runner invokes Spec::populate_device_context(args, ctx) after
 * the base members (P, CellP, num_total) are initialised, letting the Spec
 * stash UVM-resident pointers (e.g., per-active host-staged input buffers)
 * into the extended ctx for device-side load_active / load_neighbor / pair_kernel.
 * After the runner's evaluate phase completes, the runner invokes
 * Spec::cleanup_device_context(args, ctx) on every path so the Spec can free
 * any UVM allocations made during populate. cleanup is also detected
 * conditionally — if the Spec doesn't define it (e.g., a derived ctx that only
 * caches an externally-owned pointer), the runner emits no call.
 *
 * When DeviceContext == base (sink_env1 and any Spec that uses base ctx
 * unmodified), the runner skips both populate and cleanup — no behavior change
 * vs. pre-4.A.0.
 *
 * Spec contract: every Spec MUST declare `using DeviceContext = ...;` (either
 * the base or its own derived type). Specs whose DeviceContext derives MUST
 * also provide `static void populate_device_context(const neighbor_loop_args&,
 * DeviceContext&);` and SHOULD provide
 * `static void cleanup_device_context(const neighbor_loop_args&, DeviceContext&);`
 * if populate_device_context allocated owning resources. populate without
 * cleanup is allowed when populate only caches pointers to memory whose
 * lifetime is externally guaranteed (e.g., args.aux-rooted host buffers).
 *
 * Compile-time DeviceContext invariants:
 *   - Spec::DeviceContext MUST publicly derive from NeighborLoopDeviceContextBase
 *     (or BE the base) so the helper templates can pass it as `const Base&`
 *     when convenient.
 *   - Spec::DeviceContext MUST be trivially copyable: the runner captures it
 *     by value into Kokkos device lambdas. No std::vector, no owning smart
 *     pointers, no objects with non-trivial copy / dtor.
 * Both invariants are checked by static_assert at the runner call sites.
 * ========================================================================== */
template <typename Spec>
struct nlr_spec_has_extended_device_context {
    static constexpr bool value =
        !std::is_same<typename Spec::DeviceContext,
                      NeighborLoopDeviceContextBase>::value;
};

template <typename Spec>
constexpr bool nlr_spec_has_extended_device_context_v =
    nlr_spec_has_extended_device_context<Spec>::value;

/* SFINAE detection of optional Spec::cleanup_device_context. Returns false
 * for Specs that don't define it; the runner uses if constexpr to skip the
 * call when absent. */
template <typename Spec, typename = void>
struct nlr_spec_has_cleanup_device_context : std::false_type {};

template <typename Spec>
struct nlr_spec_has_cleanup_device_context<
    Spec,
    std::void_t<decltype(Spec::cleanup_device_context(
        std::declval<const neighbor_loop_args&>(),
        std::declval<typename Spec::DeviceContext&>()))>>
    : std::true_type {};

template <typename Spec>
constexpr bool nlr_spec_has_cleanup_device_context_v =
    nlr_spec_has_cleanup_device_context<Spec>::value;

/* RAII cleanup guard for the runner's DeviceContext. Construct one right
 * after Spec::populate_device_context returns; destruction at function exit
 * (any path) conditionally invokes Spec::cleanup_device_context if the Spec
 * defined one. No-op for base-ctx Specs. */
template <typename Spec>
struct NlrDeviceContextCleanupGuard {
    const neighbor_loop_args& args;
    typename Spec::DeviceContext& ctx;
    NlrDeviceContextCleanupGuard(const neighbor_loop_args& a,
                                  typename Spec::DeviceContext& c)
        : args(a), ctx(c) {}
    NlrDeviceContextCleanupGuard(const NlrDeviceContextCleanupGuard&) = delete;
    NlrDeviceContextCleanupGuard& operator=(const NlrDeviceContextCleanupGuard&) = delete;
    ~NlrDeviceContextCleanupGuard() {
        if constexpr (nlr_spec_has_cleanup_device_context_v<Spec>) {
            Spec::cleanup_device_context(args, ctx);
        }
    }
};

/* ============================================================================
 * Iteration driver
 * ========================================================================== */

struct NotIterative {};

enum class IterStatus : int {
    Converged = 0, Adjust = 1, NeedsMore = 2,
};

struct IterControl {
    IterStatus status;
};

/* ============================================================================
 * Empty marker types — required to be DISTINCT named types (not just any
 * empty struct) so static_asserts can verify exact intent.
 * ========================================================================== */

struct NoScatter {};
struct NoAccum   {};

/* ============================================================================
 * DispatchPath — runtime choice
 * ========================================================================== */

enum class DispatchPath : int {
    ModeA_GPU_NGL    = 0,
    ModeB_HostWalker = 1,
    Brute_Oracle     = 2,
};

/* ============================================================================
 * NeighborLoopSpec — copy-pasteable skeleton for a new physics loop
 *
 * Copy this block verbatim into your loop's <loop_name>_loop.h, replace the
 * physics, and you're done. The runner contract is what's below; everything
 * else is engine. See sinks/sink_env1_loop.h for a worked example.
 *
 * Per-spec layout convention (mirrors sink_env1_loop.h):
 *   PHYSICS BLOCK   — edit when changing the loop's physics
 *   ENGINE APPARATUS — touch only when changing the runner contract itself
 *   DIAGNOSTICS     — env-gated (see GIZMO_*_DIAG / _ORACLE / _SPIKE_*)
 *
 *   struct MyLoopSpec {
 *     // ============ PHYSICS BLOCK ============
 *
 *     // (1) Loop identity (drives env-var prefixes + diagnostic labels)
 *     static constexpr const char *loop_name = "my_loop";
 *
 *     // (2) Search policy
 *     static constexpr int                     search_mode        = MODE_B_SEARCH_SYMMETRIC;
 *     static constexpr unsigned int            neighbor_type_mask = (1u<<0);
 *     static constexpr mode_b_radius_policy_t  radius_policy      = MODE_B_RADIUS_DEFAULT;
 *
 *     // (3) Writeback policy
 *     static constexpr WritePattern   write_pattern   = WritePattern::ActiveReduceOnly;
 *     static constexpr SidxCacheKind  sidx_cache_kind = SidxCacheKind::AllTypes;
 *
 *     // (4) Tolerances
 *     static constexpr double accum_tolerance = 1e-9;
 *
 *     // (5) Active-particle predicate. Caller passes this to nlr_build_active_list.
 *     static bool is_active(int particle_index);
 *
 *     // (6) Per-pair physics types (all trivially copyable; see TRAP 5)
 *     struct CallScalars   { NlrCommonScalars common; ... };
 *     struct ActiveData    { ...; CallScalars scalars; };
 *     struct NeighborData  { const struct particle_data* neighbor_particle; ... };
 *     using  AccumData     = my_loop_accum_t;
 *     using  ScatterData   = NoScatter;        // or your scatter type
 *     using  IdentityFields = NoIdentity;      // see runner header for non-default
 *     using  IterControl   = NotIterative;
 *
 *     // (6b) DeviceContext — REQUIRED. Most Specs declare base unchanged:
 *     using  DeviceContext = NeighborLoopDeviceContextBase;
 *     // To extend (e.g., Specs that need host-staged per-active UVM input
 *     // arrays visible inside load_active / load_neighbor on device):
 *     //   struct MyLoopDeviceContext : NeighborLoopDeviceContextBase {
 *     //       const my_per_active_t *per_active_local;  // UVM
 *     //   };
 *     //   using DeviceContext = MyLoopDeviceContext;
 *     // Then provide:
 *     //   static void populate_device_context(const neighbor_loop_args& args,
 *     //                                       DeviceContext& ctx);
 *     //   static void cleanup_device_context (const neighbor_loop_args& args,
 *     //                                       DeviceContext& ctx);   // optional;
 *     //                                       only required if populate
 *     //                                       allocated owning resources.
 *     // Compile-time invariants enforced by static_assert in the runner:
 *     //   - DeviceContext publicly derives from (or IS) NeighborLoopDeviceContextBase
 *     //   - DeviceContext is trivially copyable (captured by value into Kokkos lambdas)
 *
 *     // (7) Per-active aux passed by caller through neighbor_loop_args::aux
 *     struct Aux { my_loop_accum_t *out_buffer; };
 *
 *     // (8) Per-pair physics body. Forward to a header-inline single-source-of-truth helper if
 *     //     you want the same body callable from device (Mode A) and host
 *     //     (Mode B walker, oracle). See TRAP 3.
 *     KOKKOS_INLINE_FUNCTION
 *     static void pair_kernel(const ActiveData& active, const NeighborData& neighbor,
 *                             AccumData& accum, ScatterData& scatter);
 *
 *     // (9) Per-active and per-call hooks
 *     static double      search_radius(const neighbor_loop_args& args,
 *                                      int active_slot, int i);
 *     static CallScalars populate_call_scalars(const neighbor_loop_args& args);
 *     KOKKOS_INLINE_FUNCTION
 *     static ActiveData  load_active(const NeighborLoopDeviceContextBase& ctx,
 *                                    int active_slot, int i,
 *                                    double h_search, const CallScalars& scalars);
 *     KOKKOS_INLINE_FUNCTION
 *     static NeighborData load_neighbor(const NeighborLoopDeviceContextBase& ctx,
 *                                       int j, const IdentitySidecar& id,
 *                                       const ActiveData& active);
 *     KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& accum);
 *     static void apply_active_writeback(const neighbor_loop_args& args,
 *                                        int active_slot, int i,
 *                                        const AccumData& accum);
 *     static void merge_accum(AccumData& local_accum,
 *                             const AccumData& peer_accum);
 *
 *     // (10) Ghost-side writeback hooks (often empty). Trait-gated.
 *     static constexpr bool uses_ghost_write_detector = true;
 *     static constexpr bool uses_ghost_writeback      = false;
 *     static void ghost_write_detector_begin(const neighbor_loop_args&,
 *                                             const NeighborLoopPlan&);
 *     static void ghost_write_detector_end  (const neighbor_loop_args&,
 *                                             const NeighborLoopPlan&);
 *     // ...writeback variants as needed
 *
 *     // ============ ENGINE APPARATUS ============
 *     // (DeviceContext declaration is part of the REQUIRED block above —
 *     //  see (6b) earlier in this skeleton.)
 *
 *     // Optional: per-loop dispatch-threshold constants. These are
 *     // code-level dispatch policy for the loop (chosen alongside the
 *     // physics), NOT user-tunable runtime parameters. Default 64/64
 *     // when omitted. Tester env overrides exist but are not production
 *     // interface; see runner.cc TESTERS' KNOBS block.
 *     // static constexpr int modeb_threshold_sum = 64;
 *     // static constexpr int modeb_threshold_max = 64;
 *
 *     // ============ DIAGNOSTICS (env-gated) ============
 *     static double compare_accum(const AccumData& local, const AccumData& oracle);
 *     static void   diagnostic_dump_active(const ActiveDumpView<MyLoopSpec>& v);
 *     static void   diagnostic_dump_neighbor_list(const NeighborListDumpView<MyLoopSpec>& v);
 *   };
 *
 * ============================================================================
 * TRAPS — invariants a new Spec author will silently violate without these
 * ============================================================================
 *
 *   TRAP 1: populate_call_scalars is host->device VALUE CAPTURE, not
 *           inter-rank sync. Globals enter the kernel via this function only
 *           — the pair_kernel must NEVER read All.* directly. Threading
 *           CallScalars through ActiveData is what makes Mode A and Mode B
 *           bit-identical (single source of truth for per-call state).
 *
 *   TRAP 2: load_active runs on the device. KOKKOS_INLINE_FUNCTION is the
 *           contract: no std::, no host-only helpers. Same for load_neighbor,
 *           pair_kernel, zero_accum.
 *
 *   TRAP 3: pair_kernel body MUST be header-inlined (in <loop>_loop.h). A
 *           definition in the .cc breaks Kokkos device compilation with
 *           obscure linkage errors.
 *
 *   TRAP 4: merge_accum per-field op MUST match the per-field op pair_kernel
 *           writes. Sum for additive fields, MAX for max-reduced fields.
 *           Mismatch = silent multi-rank corruption (no compile error; only
 *           the oracle catches it).
 *
 *   TRAP 5: CallScalars, ActiveData, NeighborData, AccumData must all be
 *           std::is_trivially_copyable_v. Adding a std::vector member compiles
 *           fine and breaks transport silently in Mode B remote. Runner
 *           static_asserts at the explicit-instantiation site.
 *
 *   TRAP 6: diagnostic_dump_neighbor_list is called once per active, in slot
 *           order 0..num_active-1. State machines that rely on slot ordering
 *           (e.g. first-call-only gates) are safe.
 * ========================================================================== */

/* ============================================================================
 * Public runner entry point
 *
 * Linking contract: function template; implementation in
 * neighbor_loop_runner.cc with EXPLICIT INSTANTIATION per migrated caller:
 *   template void run_neighbor_loop<MyLoopSpec>(const neighbor_loop_args&);
 * Forgetting the instantiation = clean linker error (not silent template bloat).
 * ========================================================================== */

/* ============================================================================
 * neighbor_loop_args — physics inputs to run_neighbor_loop<Spec>.
 *
 * The caller fills these with PHYSICS facts ONLY. The caller does not pick
 * the dispatch path, run any prep, or know which substrate (Mode A GPU NGL,
 * Mode B local, Mode B remote, future paths) will execute. The runner owns
 * all of that.
 *
 * Field semantics:
 *   P, CellP        — particle data view at call time. The runner takes a
 *                     local working copy; on paths that mutate the global
 *                     particle arrays (today: only Mode A's ghost-import
 *                     prep), the runner refreshes its working copy after
 *                     mutation. Callers MUST NOT pass pointers that go
 *                     stale across the call.
 *   num_total       — = NumPart at call time. The runner refreshes this
 *                     internally on paths that grow NumPart via ghost
 *                     import (Mode A).
 *   active_list     — caller-owned active-particle indices, valid for the
 *                     duration of the call. args.active_list[slot] =
 *                     particle index.
 *   num_active      — length of active_list.
 *   aux             — Spec-defined POD pointer for per-active side arrays
 *                     (e.g. SinkEnv1Spec::Aux::per_active_accum). Recover
 *                     the typed pointer with nlr_aux<Spec>(args).
 *   ghost_safety_factor — caller fills unconditionally from
 *                     gizmo_ghost_safety_factor(); the runner uses it only
 *                     on paths that import ghosts (Mode A today). Mode B
 *                     paths ignore it.
 *
 * The caller does NOT compute or stage per-active radii. The runner stages
 * radii once via Spec::search_radius and reuses them for any prep
 * (Mode A) and the chosen walker.
 * ========================================================================== */
struct neighbor_loop_args {
    struct particle_data *P;
    struct gas_cell_data *CellP;
    int    num_total;
    int   *active_list;          /* args.active_list[slot] = particle index */
    int    num_active;
    void  *aux;                  /* spec-defined POD for per-active side arrays;
                                  * recover the typed pointer with nlr_aux<Spec>(args). */
    double ghost_safety_factor;  /* caller fills via gizmo_ghost_safety_factor()
                                  * unconditionally; runner uses ONLY on paths
                                  * that call gizmo_request_filtered_ghost_import. */
};

/* ============================================================================
 * Caller-side helpers (collapse boilerplate; physics stays visible)
 *
 * Pattern for the caller of run_neighbor_loop<Spec>:
 *
 *   int *active_list, num_active, num_global_active;
 *   if (!nlr_build_active_list(MyLoopSpec::is_active, &active_list,
 *                              &num_active, &num_global_active,
 *                              "myloop_active")) {
 *       return;                                       // no active particles anywhere
 *   }
 *   MyLoopAccum *out_buffer = (MyLoopAccum*)mymalloc("myloop_out", ...);
 *   MyLoopSpec::Aux aux{out_buffer};
 *
 *   neighbor_loop_args args = nlr_default_args();
 *   args.active_list = active_list;
 *   args.num_active  = num_active;
 *   args.aux         = &aux;
 *   run_neighbor_loop<MyLoopSpec>(args);
 *
 *   // ...physics scatter from out_buffer into your per-loop temp struct...
 *
 *   myfree(out_buffer);
 *   nlr_free_active_list(active_list);
 * ========================================================================== */

/* Recover the typed Spec::Aux* from neighbor_loop_args.aux, replacing
 * hand-written reinterpret_cast at every hook site. */
template <typename Spec>
typename Spec::Aux *nlr_aux(const neighbor_loop_args& args)
{
    return reinterpret_cast<typename Spec::Aux*>(args.aux);
}

/* Count active particles via predicate, MPI_Allreduce, malloc + fill.
 * Returns false if global count is zero (caller should early-return);
 * otherwise *list_out is mymalloc'd and must be released via
 * nlr_free_active_list. malloc_label is forwarded to mymalloc. */
template <typename ActivePredicate>
bool nlr_build_active_list(ActivePredicate is_active,
                           int **list_out,
                           int  *num_local_active_out,
                           int  *num_global_active_out,
                           const char *malloc_label);

void nlr_free_active_list(int *active_list);

/* Fills the engine fields of neighbor_loop_args from current globals
 * (P, CellP, NumPart, ghost_safety_factor). Caller fills active_list,
 * num_active, aux. Out-of-line so changes to global plumbing are
 * caller-invisible. */
neighbor_loop_args nlr_default_args(void);

/* ============================================================================
 * NeighborLoopPlan — runner-internal execution policy descriptor.
 *
 * `path` is the single-source-of-truth. Properties (needs imported ghosts, may acquire GPU
 * arena, may mutate NumPart, etc.) are derived via the nlr_path_*() helpers
 * below — never stored as separate boolean fields on the plan, so a new
 * path adds cases to the helpers and never new fields to this struct.
 * ========================================================================== */
struct NeighborLoopPlan {
    enum class Path {
        ModeA_GpuNgl,    /* GPU NGL pipeline. Current substrate uses imported
                          * ghosts; freshness is satisfied by today's import
                          * helper (gizmo_request_filtered_ghost_import). The
                          * substrate is what the path REQUIRES; the helper is
                          * one (current) way of satisfying that requirement
                          * — future work may swap helpers without changing
                          * the path. Do not equate Mode A with "global drift". */
        ModeB_Local,     /* Host walker on local pool; no global mutation. */
        ModeB_Remote     /* peer-to-peer request/reply; no global mutation. */
        /* Future paths: add cases here AND to nlr_path_*() predicates below. */
    };
    Path path;

    /* Global active-particle count after MPI_Allreduce. Sentinel: -1 means
     * "not populated" — happens on force-mode paths
     * (GIZMO_NLR_FORCE_MODE{A,B}=1) when GIZMO_PHASE0_DIAG is OFF, since the
     * runner skips the Allreduce in that case to avoid an extra collective.
     * Hooks/consumers MUST check for -1 before assuming this field reflects
     * a real global count. Threshold-dispatch paths and any path with
     * PHASE0_DIAG on always populate it. */
    int  num_active_global;
};

/* Path-derived predicate helpers. Single source of truth: keyed on path. */
bool nlr_path_uses_imported_ghosts(NeighborLoopPlan::Path path);
bool nlr_path_uses_gpu_arena(NeighborLoopPlan::Path path);
bool nlr_path_permits_global_numpart_mutation(NeighborLoopPlan::Path path);
bool nlr_path_uses_lazy_drift(NeighborLoopPlan::Path path);

/* Stable string label for diagnostics (PHASE0_NLR `path=...` field).
 * Stable across the runner's lifetime; future path cases extend this. */
const char *nlr_path_label(NeighborLoopPlan::Path path);

/* ============================================================================
 * run_neighbor_loop<Spec>(args) — the runner.
 *
 * Numerical execution policy single-source-of-truth for one neighbor loop. The caller hands
 * over physics; the runner picks the substrate and runs it.
 *
 *   CALLER OWNS                       RUNNER OWNS
 *   -----------                       -----------
 *   Active set                        Selected execution path
 *   Search predicate (Spec traits)    Whether ghost import runs
 *   Per-loop physics data (Spec)      Drift strategy
 *   Write pattern (Spec trait)        Transport strategy
 *   Ghost safety factor (args field)  Per-active radii staging
 *                                     Lifecycle hooks (detector, writeback)
 *                                     Hard-corridor invariant enforcement
 *
 * Execution path is chosen by the runner from a small enum (Mode A GPU NGL,
 * Mode B local, Mode B remote, future paths). Selection precedence: env
 * force-mode > threshold dispatch on
 * num_active_global vs Spec::modeb_threshold_{sum,max} > Spec defaults.
 * Future paths add a case to the dispatch + the path-predicate helpers
 * (nlr_path_uses_imported_ghosts, nlr_path_uses_gpu_arena, etc.) and never
 * add fields to the plan struct itself — path is the single-source-of-truth.
 *
 * Tiny-N hard-corridor invariants (HARD ABORT on violation, always-on,
 * every build, enforced via global lifecycle counters):
 *
 *   On Mode B (local OR remote) paths, from runner entry to runner exit:
 *     - NO move_particles call (g_global_drift_counter unchanged)
 *     - NO ghost_exchange_impl call (g_ghost_import_counter unchanged)
 *     - NO gpu_particles_arena_acquire call
 *       (g_gpu_arena_acquire_counter unchanged)
 *     - NO change to NumPart
 *
 *   These invariants protect the lazy-drift architecture: Mode B reads
 *   owner-local args.P[j] / args.CellP[j] freely during evaluation and
 *   drifts only touched candidates via mode_b_lazy_drift_candidates. The
 *   invariant is NO GLOBAL MUTATION, not no read. A full global drift in
 *   the prep layer would defeat lazy drift; the corridor enforcement
 *   prevents that regression.
 *
 *   Mode A path retains ghost import + global drift (today's substrate).
 *   That cost is appropriate for large-N where the import + drift amortizes
 *   over millions of pair evaluations. Forced Mode A on tiny-N is
 *   tester/baseline territory: expensive but correct.
 *
 * Path -> required-data matrix (today; future paths extend by adding cases
 * to nlr_path_*() predicates):
 *
 *   ModeA_GpuNgl       : may move_particles, may ghost_exchange_impl,
 *                        may grow NumPart, may acquire GPU arena
 *   ModeB_Local        : none of the above
 *   ModeB_Remote       : none of the above (uses peer-to-peer request/reply
 *                        substrate; peer-local walks; lazy drift on
 *                        touched candidates)
 *
 * Lazy-drift contract inside Mode B:
 *   collect candidates pre-drift -> mode_b_lazy_drift_candidates(touched)
 *   -> evaluate pairs post-drift. drift_particle is per-particle and
 *   short-circuits on time1 == time0. Repeat candidates between queries
 *   in the same call are a fast no-op.
 *
 * radii lifetime: the runner stages a std::vector<double> radii of size
 * args.num_active and passes radii.data() to path-specific functions and
 * (on Mode A) to gizmo_request_filtered_ghost_import_fresh. The pointer
 * is RUNNER-CALL-LIFETIME ONLY. Callees MUST NOT store the pointer; if a
 * callee needs to retain values, it copies. The vector goes out of scope
 * at the end of run_neighbor_loop.
 *
 * Spec lifecycle hooks (two channels, optional, SFINAE-detected;
 * default no-op; per-Spec gating decided per-channel by audit):
 *   detector  : Spec::ghost_write_detector_begin/end<Spec> — audit/debug.
 *   writeback : Spec::ghost_writeback_begin/end<Spec> — physics state
 *               propagation. Spec body is the per-flag #ifdef union of
 *               all enabled physics-flag writebacks for that loop.
 * Whether each channel fires on all paths or only on imported-ghost paths
 * is a per-Spec decision settled by reading the underlying call's actual
 * semantics; no blanket rule.
 *
 * For the full Spec contract (load_active, load_neighbor, pair_kernel,
 * accum/scatter writeback, optional iterative loop semantics), see the
 * Spec-shape comment block above this declaration.
 * ========================================================================== */
template <typename Spec>
void run_neighbor_loop(const neighbor_loop_args& args);

/* ============================================================================
 * Compile-time spec consistency checks (used at the top of run_neighbor_loop):
 *
 *   ActiveReduceOnly       requires std::is_same_v<ScatterData, NoScatter>
 *   NeighborScatter        requires std::is_same_v<AccumData,   NoAccum>
 *   SymmetricPairScatter   requires both non-empty (NOT NoScatter / NoAccum)
 *   GhostWritebackRequired requires both non-empty
 *
 *   IterControl != NotIterative           requires after_iter()
 *   ScatterData != NoScatter              requires apply_neighbor_writeback,
 *                                                  compare_scatter
 *   DeviceContext != NeighborLoopDeviceContextBase
 *                                          requires populate_device_context
 *
 *   POD/device-copy contract (enforced at the top of run_neighbor_loop<Spec>):
 *     std::is_trivially_copyable_v<Spec::CallScalars>  (lambda-captured by value)
 *     std::is_trivially_copyable_v<Spec::ActiveData>   (UVM-staged)
 *     std::is_trivially_copyable_v<Spec::NeighborData> (built per-pair on device)
 *     std::is_trivially_copyable_v<Spec::AccumData>    (UVM-staged)
 * ========================================================================== */

/* ============================================================================
 * Dispatch threshold accessors
 *
 * Production dispatch policy lives in the constexpr Spec::modeb_threshold_sum
 * and Spec::modeb_threshold_max in each NeighborLoopSpec — these are
 * code-level dispatch policy constants for the loop, decided alongside the
 * physics. They are NOT user-tunable runtime parameters.
 *
 * Threshold env vars (GIZMO_NLR_MODEB_THRESHOLD_*, GIZMO_<LOOP>_MODEB_THRESHOLD_*)
 * are TESTERS' KNOBS only — temporary overrides for runner-internal testing.
 * They are NOT part of the production interface, NOT promoted to params.txt
 * or Config.sh, and emit a rank-0 one-shot "tester only" warning when
 * actually consumed. See the TESTERS' KNOBS block in
 * mesh/neighbor_loop_runner.cc for the full env-var listing, resolution
 * precedence, and warning semantics.
 *
 * Resolution precedence (per-loop env > global env > Spec constexpr) is
 * preserved by the _for() lookups; invalid env values silently fall through
 * to the next level (no policy change in B.ii).
 * ========================================================================== */

int  gizmo_nlr_default_modeb_threshold_sum(void);
int  gizmo_nlr_default_modeb_threshold_max(void);
int  gizmo_nlr_modeb_threshold_sum_for(const char *loop_name, int spec_default);
int  gizmo_nlr_modeb_threshold_max_for(const char *loop_name, int spec_default);

/* ============================================================================
 * NLR env config — unified surface for diagnostic, control, and spike vars.
 *
 * Canonical (Pass B.i):
 *   GIZMO_NLR_DIAG=<0|1|2|3>          0=off, 1=PHASE0 timing line per call,
 *                                     2=+dispatch trace, 3=reserved (today
 *                                     equivalent to level 2; rank-0 note)
 *   GIZMO_NLR_FORCE_MODE=A|B          tester force-mode override
 *   GIZMO_NLR_SPIKE_ACCUM_DUMP=1      cross-validation per-active accumulator
 *                                     dump (SPIKE; retire after 3d ports)
 *   GIZMO_NLR_SPIKE_NB_DUMP=1         first-call Mode A neighbor-list dump
 *                                     (SPIKE; retire after 3d ports)
 *   GIZMO_NLR_ORACLE=1                correctness gate (separate concern)
 *   GIZMO_NLR_ORACLE_DUMP=1           field-by-field oracle mismatch dump
 *
 * Aliases (DEPRECATED; accepted for one cycle with rank-0 warning; explicit
 * removal queued for the next cleanup pass after Pass B):
 *   GIZMO_PHASE0_DIAG=1            -> GIZMO_NLR_DIAG=1
 *   GIZMO_NLR_DISPATCH_TRACE=1     -> GIZMO_NLR_DIAG=2
 *   GIZMO_NLR_FORCE_MODEA=1        -> GIZMO_NLR_FORCE_MODE=A
 *   GIZMO_NLR_FORCE_MODEB=1        -> GIZMO_NLR_FORCE_MODE=B
 *   GIZMO_MODE_B_XVAL_DUMP=1       -> GIZMO_NLR_SPIKE_ACCUM_DUMP=1
 *   GIZMO_MODE_B_XVAL_NB_DUMP=1    -> GIZMO_NLR_SPIKE_NB_DUMP=1
 *
 * Conflict policy (collective; all ranks endrun):
 *   GIZMO_NLR_FORCE_MODE set + any old _FORCE_MODE{A,B} set     -> endrun
 *   _FORCE_MODEA=1 AND _FORCE_MODEB=1                           -> endrun
 *   GIZMO_NLR_FORCE_MODE invalid value (not A or B)             -> endrun
 *   GIZMO_NLR_DIAG invalid (non-integer, <0, or >3)             -> endrun
 *
 * Diagnostic alias precedence (NEW WINS, old-set produces ignore note):
 *   GIZMO_NLR_DIAG set + diagnostic alias set -> new wins, alias ignored
 *
 * Warnings: rank-0 only, cached one-shot per env-var key.
 * ========================================================================== */

enum class NlrForceMode { None = 0, A = 1, B = 2 };

int          gizmo_nlr_diag_level(void);              /* 0..3 (3 == 2 today) */
NlrForceMode gizmo_nlr_force_mode(void);              /* None / A / B */
bool         gizmo_nlr_spike_accum_dump_enabled(void);
bool         gizmo_nlr_spike_nb_dump_enabled(void);

/* Convenience adapters — preserved for existing callers, all delegating
 * to the unified API above. */
bool gizmo_nlr_force_mode_b_global(void);
bool gizmo_nlr_force_mode_a_global(void);
bool gizmo_nlr_dispatch_trace_enabled(void);

bool gizmo_nlr_oracle_enabled_global(void);
bool gizmo_nlr_oracle_enabled_for(const char *loop_name);

/* ============================================================================
 * NlrQueryEnvelope — runner-owned transport wrapper for cross-rank queries.
 *
 * Wraps Spec::ActiveData with origin metadata so reply merge does NOT depend
 * on transport ordering (Option C from the 3c.3 design lock-in). Codex
 * invariant: "carrying origin_slot prevents a nasty future silent bug" even
 * though current transport (mode_b_p2p_transport) preserves order — the
 * runner's contract is robust to any future transport substitution.
 *
 * origin_rank kept for diagnostics + sanity asserts (reply receiver can
 * verify envelope.origin_rank == ThisTask).
 *
 * Trivially-copyable iff Spec::ActiveData is — already enforced at the top
 * of run_neighbor_loop<Spec>.
 * ========================================================================== */
template <typename ActiveData>
struct NlrQueryEnvelope {
    int        origin_slot;   /* active_slot on the origin rank (== aa) */
    int        origin_rank;   /* ThisTask of origin rank; for diagnostics */
    ActiveData active;
};

/* Symmetric reply envelope: peer copies {origin_slot, origin_rank} back from
 * the received query envelope into its reply, so the active rank can merge
 * by slot WITHOUT relying on transport ordering. Receive-side asserts
 * origin_rank == ThisTask as a sanity check. Trivially-copyable iff
 * AccumData is — already enforced at the top of run_neighbor_loop<Spec>. */
template <typename AccumData>
struct NlrReplyEnvelope {
    int       origin_slot;
    int       origin_rank;
    AccumData accum;
};

/* ============================================================================
 * Diagnostic views and SFINAE-detected optional Spec hooks (3c.4a)
 *
 * Optional, never part of the physics API. Specs opt in by defining the
 * static method; the runner SFINAE-detects and only calls when present.
 *
 * View structs carry full identity fields (rank, origin_rank, origin_slot,
 * active_slot, path, call_id) even where some are redundant for self/local
 * in 3c.4a. Locking the shape now avoids a future change when peer-side
 * dumps land.
 *
 * `path` strings preserved byte-identical to legacy:
 *   "gpu_ngl" — Mode A GPU NGL path
 *   "mode_b"  — Mode B local OR Mode B remote (active rank's own actives)
 *
 * Accum dump fires for both Mode A and runner-driven Mode B paths. NB dump
 * fires Mode A only; Mode B local NB + UVM-readback diagnostic and Mode B
 * remote peer-side NB dump are deferred (see OPEN_modeb_nb_dump.md in
 * memory).
 *
 * Cached env-gate accessors below are first-use cached; mid-run env
 * changes do NOT take effect, matching this code's cached-env convention.
 * ========================================================================== */

template <typename Spec>
struct ActiveDumpView {
    int rank;
    int origin_rank;
    int origin_slot;
    int active_slot;
    const char *path;
    int call_id;
    const struct neighbor_loop_args *args;
    const typename Spec::ActiveData *active;     /* may be nullptr if spec/path doesn't supply */
    const typename Spec::AccumData  *accum;
};

template <typename Spec>
struct NeighborListDumpView {
    int rank;
    int origin_rank;
    int origin_slot;
    int active_slot;
    const char *path;
    int call_id;
    const struct neighbor_loop_args *args;
    const typename Spec::ActiveData *active;
    const int *candidate_ids;                    /* host-resident; runner-owned, valid only during call */
    int n_candidates;
};

/* Detection traits. SFINAE on the static call expression. */
template <typename Spec, typename = void>
struct spec_has_dump_active : std::false_type {};
template <typename Spec>
struct spec_has_dump_active<Spec, std::void_t<decltype(
    Spec::diagnostic_dump_active(std::declval<const ActiveDumpView<Spec>&>()))>>
    : std::true_type {};

template <typename Spec, typename = void>
struct spec_has_dump_neighbor_list : std::false_type {};
template <typename Spec>
struct spec_has_dump_neighbor_list<Spec, std::void_t<decltype(
    Spec::diagnostic_dump_neighbor_list(std::declval<const NeighborListDumpView<Spec>&>()))>>
    : std::true_type {};

/* Cached env-gate adapters. Read-once-per-process; mid-run env changes
 * do not take effect. All delegate to the unified gizmo_nlr_diag_level /
 * gizmo_nlr_spike_*_enabled accessors documented at the top of this header.
 *
 * Mapping:
 *   gizmo_nlr_xval_dump_enabled()    -> gizmo_nlr_spike_accum_dump_enabled()
 *   gizmo_nlr_xval_nb_dump_enabled() -> gizmo_nlr_spike_nb_dump_enabled()
 *   gizmo_nlr_phase0_diag_enabled()  -> (gizmo_nlr_diag_level() >= 1)
 *
 * The xval_* names are kept on existing call sites for now and will be
 * renamed alongside the env-name retire in a follow-up cleanup pass. */
bool gizmo_nlr_xval_dump_enabled(void);
bool gizmo_nlr_xval_nb_dump_enabled(void);
bool gizmo_nlr_phase0_diag_enabled(void);

/* ============================================================================
 * RunnerStageTimer — timing accumulator populated when GIZMO_PHASE0_DIAG=1
 *
 * Each path function (run_mode_a / run_mode_b_local / run_mode_b_remote)
 * receives a pointer; nullptr means "phase0 off, skip all timing." Path
 * fills only the fields that apply to its dispatch:
 *
 *   path=gpu_ngl         dt_collect (NGL build) + dt_walk_self (pair kernel)
 *                          + dt_writeback. Others = 0.
 *   path=mode_b_local    dt_collect + dt_drift + dt_walk_self + dt_writeback.
 *                          Others = 0.
 *   path=mode_b_remote   all 8 fields. dt_collect = self+peer pre-drift
 *                          collection; dt_walk_self = self_tree evaluate;
 *                          dt_walk_peer = peer_tree evaluate; dt_exchange_q,
 *                          dt_exchange_r = peer-to-peer comm; dt_reduce = reply merge;
 *                          dt_writeback = writeback loop; dt_drift = lazy
 *                          drift on union.
 *
 * dt_total measured wall-clock from runner entry to runner exit. Now
 * includes the runner-owned prep step (dt_prep_import); the caller has
 * been simplified and no longer runs prep before the runner.
 *
 * dt_prep_import is the wall around gizmo_request_filtered_ghost_import_fresh
 * on Mode A paths; 0 on Mode B paths (genuine 0 — the API isn't called).
 * ========================================================================== */

struct RunnerStageTimer {
    double dt_prep_import;   /* Mode A only; 0 on Mode B paths. */
    double dt_collect;
    double dt_drift;
    double dt_walk_self;
    double dt_walk_peer;
    double dt_exchange_q;
    double dt_exchange_r;
    double dt_reduce;
    double dt_writeback;
    double dt_total;
};

/* ============================================================================
 * nlr_build_active_list inline definition.
 *
 * Header-defined because the predicate is templated. Instantiated in the
 * caller's TU, which already pulls in declarations/allvars.h (NumPart,
 * ActiveParticleList) and core/proto.h (mymalloc) via the loop's spec
 * header.
 * ========================================================================== */
template <typename ActivePredicate>
bool nlr_build_active_list(ActivePredicate is_active,
                           int **list_out,
                           int  *num_local_active_out,
                           int  *num_global_active_out,
                           const char *malloc_label)
{
    int num_local = 0;
    for(int i : ActiveParticleList) { if(is_active(i)) num_local++; }
    int num_global = 0;
    MPI_Allreduce(&num_local, &num_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    *num_local_active_out  = num_local;
    *num_global_active_out = num_global;
    if(num_global == 0) { *list_out = nullptr; return false; }

    int alloc_n = (num_local > 0) ? num_local : 1;
    *list_out = (int *) mymalloc(malloc_label, alloc_n * sizeof(int));
    int slot = 0;
    for(int i : ActiveParticleList) {
        if(is_active(i)) { (*list_out)[slot++] = i; }
    }
    return true;
}

#endif /* GIZMO_NEIGHBOR_LOOP_RUNNER_H */
