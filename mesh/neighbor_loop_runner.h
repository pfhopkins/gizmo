/* mesh/neighbor_loop_runner.h — generic neighbor-loop runner for GIZMO.
 *
 * Goal: provide the SSOT interface that every GIZMO neighbor loop will be
 * migrated onto over time. The interface is the binding contract; the runner
 * implementation is built incrementally.
 *
 * Initial coverage (this commit): WritePattern::ActiveReduceOnly with the
 * default RemoteEvalMode::RemoteComputesAccum, single iteration, NoIdentity,
 * NoScatter. Sufficient for sink_env1 migration.
 *
 * Other patterns / modes are DECLARED in the interface so callers don't
 * have to reshape when their needs are added; runtime policy on unsupported
 * features is documented below (TL;DR: Mode A fallback in production; abort
 * if force-Mode-B; static_assert only for internally inconsistent specs).
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
 *         search_radius_host(args, slot, i) -> double
 *       Reads external state (e.g. P[i].KernelRadius) BEFORE arena_acquire
 *       / drift / freshness. Equivalent to the radii staging that legacy
 *       callers performed in their host loop just after building the
 *       active-particle list. Result is staged into a UVM radii[num_active]
 *       array; Mode A's gpu_ngb_list_build receives it directly, Mode B's
 *       walker reads it per-query.
 *
 *   (2) HOST, pre-arena:
 *         populate_call_scalars_host(args) -> CallScalars
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
 *   pair_kernel signature (SSOT lever; contract §"Pair kernel signature"):
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
 *     - hooks (host):     load_active_host, apply_active_writeback,
 *                         apply_neighbor_writeback (scatter only),
 *                         populate_device_context (optional extension)
 *     - hooks (device):   load_neighbor, pair_kernel
 *
 * Generic runner does:
 *     - dispatch:  Mode A vs Mode B vs Brute — runtime choice from
 *                  Allreduce'd Nactive vs threshold + force-mode env
 *     - search:    coarse predicate, type mask + per-type hmax pruning
 *     - transport: ghost (Mode A) or P2P (Mode B)
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
 * Required-shape reference for NeighborLoopSpec
 *
 *   struct MyLoopSpec {
 *     // ---- Required identifiers ----
 *     static constexpr const char* loop_name = "my_loop";
 *
 *     // ---- Required types (all must be std::is_trivially_copyable_v) ----
 *     struct CallScalars   { ... };       // per-call globals; replicated to device
 *     struct ActiveData    { ... };       // device-built post-NGL into UVM array
 *     struct NeighborData  { ... };       // built device-side via load_neighbor
 *     struct AccumData     { ... };       // or NoAccum if NeighborScatter
 *     struct ScatterData   { ... };       // or NoScatter if ActiveReduceOnly
 *     using IdentityFields = NoIdentity;
 *     using IterControl    = NotIterative;
 *
 *     // Optional spec-extended device context. If absent: DeviceContext =
 *     // NeighborLoopDeviceContextBase. If present, must inherit the base
 *     // and the spec MUST also provide populate_device_context (see below).
 *     // using DeviceContext = MyLoopDeviceContext;
 *
 *     // ---- Required search/writeback constexprs ----
 *     static constexpr int           search_mode        = MODE_B_SEARCH_SYMMETRIC;
 *     static constexpr unsigned int  neighbor_type_mask = (1u<<0);
 *     static constexpr mode_b_radius_policy_t radius_policy = MODE_B_RADIUS_DEFAULT;
 *     static constexpr WritePattern  write_pattern      = WritePattern::ActiveReduceOnly;
 *     static constexpr SidxCacheKind sidx_cache_kind    = SidxCacheKind::AllTypes;
 *     // optional override (else inferred from write_pattern):
 *     // static constexpr RemoteEvalMode remote_eval_mode_override =
 *     //     RemoteEvalMode::RemoteComputesAccum;
 *     // optional: force unsupported Mode B features to abort instead of
 *     // falling back to Mode A. Default false.
 *     // static constexpr bool force_modeb_required = false;
 *
 *     // ---- Required oracle compare ----
 *     static double compare_accum(const AccumData& a, const AccumData& b);
 *     static constexpr double accum_tolerance = 1e-9;
 *     // For specs with ScatterData != NoScatter:
 *     // static double compare_scatter(const ScatterData& a, const ScatterData& b);
 *     // static constexpr double scatter_tolerance = 1e-9;
 *
 *     // ---- Required hooks ----
 *
 *     // (Host, pre-arena epoch) Per-active search radius from external
 *     // state (e.g. P[i].KernelRadius). Runner stages radii[num_active]
 *     // for gpu_ngb_list_build (Mode A) and per-query h_search (Mode B).
 *     static double search_radius_host(const struct neighbor_loop_args& args,
 *                                       int active_slot, int i);
 *
 *     // (Host, pre-arena epoch) Capture per-call scalar globals into POD.
 *     // Captured once per call; passed by value into device lambdas.
 *     static CallScalars populate_call_scalars_host(
 *         const struct neighbor_loop_args& args);
 *
 *     // (Device, post-NGL-build epoch) Build ActiveData for slot. Reads
 *     // ctx.P[i] / ctx.CellP[i] (UVM, post-arena/drift) and combines with
 *     // the host-staged h_search + CallScalars. Same function called
 *     // host-side from Mode B/Brute walker.
 *     KOKKOS_INLINE_FUNCTION
 *     static ActiveData load_active(const DeviceContext& ctx,
 *                                    int active_slot, int i,
 *                                    double h_search,
 *                                    const CallScalars& cs);
 *
 *     // (Device or host) Zero-init AccumData. Spec decides (memset for POD,
 *     // structured zeroing for non-trivial). Generic runner never bakes
 *     // in memset on AccumData.
 *     KOKKOS_INLINE_FUNCTION
 *     static void zero_accum(AccumData& out);
 *
 *     // (Device, host-callable) Build NeighborData for j given ctx +
 *     // identity + active.
 *     KOKKOS_INLINE_FUNCTION
 *     static NeighborData load_neighbor(const DeviceContext& ctx, int j,
 *                                       const IdentitySidecar& id,
 *                                       const ActiveData& a);
 *
 *     // (Device, host-callable) The physics. Pure. Both outputs always
 *     // present. ScatterData=NoScatter / AccumData=NoAccum compile to
 *     // zero cost. NEVER split into "active-side" + "neighbor-side"
 *     // kernels (the SPIKE-duplicate trap).
 *     KOKKOS_INLINE_FUNCTION
 *     static void pair_kernel(const ActiveData& a,
 *                             const NeighborData& nb,
 *                             AccumData& active_out,
 *                             ScatterData& nb_out);
 *
 *     // (Host) Scatter per-active accumulator back to sim state.
 *     static void apply_active_writeback(const struct neighbor_loop_args& args,
 *                                        int active_slot, int i,
 *                                        const AccumData& out);
 *
 *     // (Host) Merge a partial AccumData (e.g. from a peer rank's local-pool
 *     // contribution to the same active) into a destination AccumData. Used
 *     // by run_mode_b_remote to combine self-rank self-pair accums with
 *     // per-peer reply accums. Per-field reduction op is spec-determined
 *     // (e.g. sum for additive fields, MAX for DF_mmax_particles). Within a
 *     // single rank's evaluator, accumulation is via repeated pair_kernel
 *     // calls (which already encode the right per-field op) — merge_accum is
 *     // ONLY used at the cross-rank boundary.
 *     static void merge_accum(AccumData& dst, const AccumData& src);
 *
 *     // ---- Conditional hooks ----
 *
 *     // For SymmetricPairScatter / NeighborScatter / GhostWritebackRequired:
 *     // static void apply_neighbor_writeback(const struct neighbor_loop_args& args,
 *     //                                      int j, const ScatterData& s);
 *
 *     // For specs with extended DeviceContext:
 *     // static void populate_device_context(const struct neighbor_loop_args& args,
 *     //                                     DeviceContext& ctx);
 *
 *     // For iterative loops:
 *     // using IterControl = ::IterControl;
 *     // static IterControl after_iter(const AccumData& out, ActiveData& a, int iter);
 *
 *     // Optional dispatch-threshold override:
 *     // static constexpr int modeb_threshold_sum = 64;
 *     // static constexpr int modeb_threshold_max = 64;
 *   };
 * ========================================================================== */

/* ============================================================================
 * Public runner entry point
 *
 * Linking contract: function template; implementation in
 * neighbor_loop_runner.cc with EXPLICIT INSTANTIATION per migrated caller:
 *   template void run_neighbor_loop<MyLoopSpec>(const neighbor_loop_args&);
 * Forgetting the instantiation = clean linker error (not silent template bloat).
 * ========================================================================== */

struct neighbor_loop_args {
    struct particle_data *P;
    struct gas_cell_data *CellP;
    int  num_total;
    int *active_list;     /* args.active_list[slot] = particle index */
    int  num_active;
    void *aux;             /* spec-defined POD for per-active side arrays */
};

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
 * Dispatch threshold + force-mode env gates
 *
 * Hierarchy (per-loop > global > spec constexpr default), resolved at runtime:
 *
 *   Per-loop:  GIZMO_<UPPER_LOOP_NAME>_MODEB_THRESHOLD_SUM / _MAX
 *              e.g. GIZMO_SINK_ENV1_MODEB_THRESHOLD_SUM=128
 *   Global:    GIZMO_NLR_MODEB_THRESHOLD_SUM / _MAX
 *   Default:   Spec::modeb_threshold_sum / _max if present, else 64 / 64
 *
 * The runner queries `gizmo_nlr_modeb_threshold_{sum,max}_for(loop_name)` which
 * walks per-loop → global → caller-supplied default in that order.
 *
 * Legacy precedence — see gizmo_nlr_legacy_modeb_owns_for(): when the legacy
 * SPIKE env (e.g. GIZMO_MODE_B_SINK_ENV1=1) owns Mode B selection for a loop,
 * the runner's threshold dispatch is disabled for that loop (auto-falls to
 * Mode A). GIZMO_NLR_FORCE_MODEB=1 overrides legacy precedence (testers' knob).
 * ========================================================================== */

int  gizmo_nlr_default_modeb_threshold_sum(void);
int  gizmo_nlr_default_modeb_threshold_max(void);
int  gizmo_nlr_modeb_threshold_sum_for(const char *loop_name, int spec_default);
int  gizmo_nlr_modeb_threshold_max_for(const char *loop_name, int spec_default);
bool gizmo_nlr_force_mode_b_global(void);
bool gizmo_nlr_force_mode_a_global(void);
bool gizmo_nlr_dispatch_trace_enabled(void);

bool gizmo_nlr_oracle_enabled_global(void);
bool gizmo_nlr_oracle_enabled_for(const char *loop_name);

/* Legacy SPIKE precedence: returns true if legacy env for the named loop is
 * set to "1", in which case the legacy SPIKE branch in the caller (e.g.
 * sink_environment.cc) owns Mode B selection and the runner must NOT auto-
 * select Mode B via threshold. Force-MODEB env overrides this. */
bool gizmo_nlr_legacy_modeb_owns_for(const char *loop_name);

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
 * fires Mode A only in 3c.4a (Mode B local NB + UVM-readback diagnostic
 * remains in legacy SPIKE branch until 3c.5; Mode B remote peer-side NB
 * dump is deferred — see OPEN_modeb_nb_dump.md in memory). The legacy
 * SPIKE branch self-emits MODEB_XVAL/MODEB_XVAL_NB and bypasses the runner
 * entirely; no double-print at the same call.
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

/* Cached env-gate accessors. Read-once-per-process; mid-run env changes
 * do not take effect. Names preserved byte-identical to legacy:
 *   GIZMO_MODE_B_XVAL_DUMP=1     → per-active accumulator dump
 *   GIZMO_MODE_B_XVAL_NB_DUMP=1  → first-call Mode A NGL neighbor-list dump */
bool gizmo_nlr_xval_dump_enabled(void);
bool gizmo_nlr_xval_nb_dump_enabled(void);

#endif /* GIZMO_NEIGHBOR_LOOP_RUNNER_H */
