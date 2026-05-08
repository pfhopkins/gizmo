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
 * Host / device split
 * ============================================================================
 *
 *   pair_kernel runs on BOTH host (Mode B walker, Brute oracle) AND device
 *   (Mode A GPU lambda). The interface draws a clear line between host-side
 *   packing and device-callable construction/execution.
 *
 *   ActiveData is HOST-PACKED into a UVM-resident array:
 *     load_active_host(args, slot, i, iter) -> ActiveData
 *     The runner stages an ActiveData[num_active] flat array in
 *     GIZMO_KOKKOS_SHARED_SPACE (UVM) before dispatch. Mode A reads it
 *     device-side; Mode B/Brute iterates host-side.
 *
 *   Per-active search radius (host-extracted from ActiveData):
 *     search_radius(active) -> double
 *     The runner extracts a radii[num_active] array via this hook BEFORE
 *     dispatch, so Mode A's gpu_ngb_list_build receives radii (its existing
 *     contract) and Mode B's per-query walker has h_search per active.
 *
 *   NeighborData is DEVICE-CALLABLE:
 *     KOKKOS_INLINE_FUNCTION
 *     load_neighbor(ctx, j, id, active) -> NeighborData
 *     Receives a DeviceContext containing UVM-resident pointers (P, CellP,
 *     plus spec-specific extension if Spec::DeviceContext extends the
 *     base). Same function called from device (Mode A) and host (Mode B,
 *     Brute) — KOKKOS_INLINE_FUNCTION compiles to inline on both.
 *     Implementation must use only device-safe constructs (no std:: I/O,
 *     no host-only helpers); same constraint as the pair_kernel.
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
 *     // ---- Required types ----
 *     struct ActiveData    { ... };       // host-packed; lives in UVM array
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
 *     // Host-only: pack active state for slot.
 *     static ActiveData load_active_host(const struct neighbor_loop_args& args,
 *                                        int active_slot, int i, int iter);
 *
 *     // Host-only: per-active search radius. Runner builds radii[num_active]
 *     // for Mode A's gpu_ngb_list_build and Mode B's per-query h_search.
 *     static double search_radius(const ActiveData& a);
 *
 *     // Device-callable: build NeighborData for j given context + identity + active.
 *     KOKKOS_INLINE_FUNCTION
 *     static NeighborData load_neighbor(const DeviceContext& ctx, int j,
 *                                       const IdentitySidecar& id,
 *                                       const ActiveData& a);
 *
 *     // Device-callable: the physics. Pure. Both outputs always present.
 *     KOKKOS_INLINE_FUNCTION
 *     static void pair_kernel(const ActiveData& a,
 *                             const NeighborData& nb,
 *                             AccumData& active_out,
 *                             ScatterData& nb_out);
 *
 *     // Host-only: scatter per-active accumulator back to sim state.
 *     static void apply_active_writeback(const struct neighbor_loop_args& args,
 *                                        int active_slot, int i,
 *                                        const AccumData& out);
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
 * ========================================================================== */

/* ============================================================================
 * Dispatch threshold + force-mode env gates
 * ========================================================================== */

int  gizmo_nlr_default_modeb_threshold_sum(void);
int  gizmo_nlr_default_modeb_threshold_max(void);
bool gizmo_nlr_force_mode_b_global(void);
bool gizmo_nlr_force_mode_a_global(void);

bool gizmo_nlr_oracle_enabled_global(void);
bool gizmo_nlr_oracle_enabled_for(const char *loop_name);

#endif /* GIZMO_NEIGHBOR_LOOP_RUNNER_H */
