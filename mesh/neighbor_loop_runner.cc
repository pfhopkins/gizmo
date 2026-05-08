/* mesh/neighbor_loop_runner.cc — generic NeighborLoopSpec runner.
 *
 * Current scope (3c.2): WritePattern::ActiveReduceOnly,
 * RemoteEvalMode::RemoteComputesAccum, non-iterative, NoIdentity, NoScatter
 * — sufficient for SinkEnv1Spec migration.
 *
 * Mode A (GPU NGL pipeline): IMPLEMENTED in 3c.1. Stages caller-supplied
 * radii and per-call CallScalars host-side pre-arena, runs
 * gpu_particles_arena_acquire + gpu_ngb_list_build, then a tiny Kokkos
 * parallel_for that calls Spec::load_active to fill ActiveData[] in UVM
 * (same device epoch as the legacy lambda's q-packing), then the parametric
 * pair-kernel parallel_for calling Spec::load_neighbor + Spec::pair_kernel.
 * Both launches go through gizmo_gpu_kernel_launch (project's parallel_for
 * + fence + check_last_error).
 *
 * Mode B local + Brute oracle: IMPLEMENTED in 3c.2 (single-rank only). Same
 * Spec::load_active / Spec::load_neighbor / Spec::pair_kernel — host-side
 * KOKKOS_INLINE_FUNCTION invocation. Lazy-drift function-boundary invariant
 * structurally encoded as collect_candidates_pre_drift →
 * lazy_drift_candidates → evaluate_pairs_post_drift; Brute oracle uses the
 * SAME drift epoch as Mode B. Multi-rank P2P is deferred to 3c.3 — runtime
 * abort guards GIZMO_NLR_FORCE_MODEB on NTask>1.
 *
 * Architecture binding contract:
 *   ~/.claude/memory/reference_neighbor_loop_contract.md
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"          /* MUST precede allvars.h: #define All All_dev for device code */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/gpu_dispatch_templates.h"  /* gizmo_gpu_kernel_launch */

#include "neighbor_loop_runner.h"
#include "gpu_neighbor_list.h"
#include "kernel.h"  /* MUST precede sink_env1_spec.h (kernel_main, NEAREST_XYZ) */
#include "ghost_writeback.h"     /* ghost_get_num_local */
#include "mode_b_local_walker.h" /* mode_b_local_neighbor_walk, brute, lazy_drift */

#include <vector>
#include <cmath>
#include <cctype>

#include "mode_b_p2p_transport.h"  /* mode_b_exchange_queries / _replies */

/* Spec instantiations supported in 3c.1. Each #include declares one Spec
 * type whose explicit template instantiation appears at the bottom of
 * this file. */
#include "../sinks/sink_env1_spec.h"

/* ============================================================================
 * Env-gate functions (3c.2: real implementations).
 *
 * Cached statics — env read once on first call, identical pattern to
 * sink_env1_mode_b_env_enabled (sinks/sink_environment_mode_b.cc:47).
 *
 * Threshold-based dispatch (Allreduce sum/max) is deferred to 3c.3 — needs
 * multi-rank wiring. 3c.2 keeps the threshold getters as 0 (= "no threshold,
 * force-mode is the only selector").
 * ========================================================================== */

/* Global threshold defaults via env override. Returns -1 if env unset, so the
 * caller's hierarchy resolver can fall through to spec/default. */
static int s_global_threshold_sum_cached = -2;
static int s_global_threshold_max_cached = -2;
static int parse_int_env(const char *name)
{
    const char *e = getenv(name);
    if(!e || !e[0]) return -1;
    char *endp = nullptr;
    long v = strtol(e, &endp, 10);
    if(!endp || *endp != '\0' || v < 0 || v > 1000000000) return -1;
    return (int)v;
}
int gizmo_nlr_default_modeb_threshold_sum(void) {
    if(s_global_threshold_sum_cached == -2) {
        s_global_threshold_sum_cached = parse_int_env("GIZMO_NLR_MODEB_THRESHOLD_SUM");
    }
    return s_global_threshold_sum_cached;
}
int gizmo_nlr_default_modeb_threshold_max(void) {
    if(s_global_threshold_max_cached == -2) {
        s_global_threshold_max_cached = parse_int_env("GIZMO_NLR_MODEB_THRESHOLD_MAX");
    }
    return s_global_threshold_max_cached;
}

/* Per-loop env lookup: GIZMO_<UPPER_LOOP_NAME>_MODEB_THRESHOLD_<SUM|MAX>.
 * Same 8-entry intrusive cache pattern as oracle_enabled_for, keyed by
 * Spec::loop_name pointer (constexpr literal). */
static int per_loop_threshold_lookup(const char *loop_name, const char *suffix)
{
    if(!loop_name || !loop_name[0]) return -1;
    char env_name[160];
    int j = 0;
    const char *prefix = "GIZMO_";
    for(int p = 0; prefix[p] && j < 150; p++) env_name[j++] = prefix[p];
    for(int p = 0; loop_name[p] && j < 150; p++) {
        env_name[j++] = (char)toupper((unsigned char)loop_name[p]);
    }
    const char *mid = "_MODEB_THRESHOLD_";
    for(int p = 0; mid[p] && j < 158; p++) env_name[j++] = mid[p];
    for(int p = 0; suffix[p] && j < 159; p++) env_name[j++] = suffix[p];
    env_name[j] = '\0';
    return parse_int_env(env_name);
}

int gizmo_nlr_modeb_threshold_sum_for(const char *loop_name, int spec_default)
{
    int v = per_loop_threshold_lookup(loop_name, "SUM");
    if(v >= 0) return v;
    v = gizmo_nlr_default_modeb_threshold_sum();
    if(v >= 0) return v;
    return spec_default;
}
int gizmo_nlr_modeb_threshold_max_for(const char *loop_name, int spec_default)
{
    int v = per_loop_threshold_lookup(loop_name, "MAX");
    if(v >= 0) return v;
    v = gizmo_nlr_default_modeb_threshold_max();
    if(v >= 0) return v;
    return spec_default;
}

bool gizmo_nlr_dispatch_trace_enabled(void) {
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_NLR_DISPATCH_TRACE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

/* Legacy SPIKE precedence. Per codex amendment: trigger ONLY on
 * GIZMO_MODE_B_<UPPER_LOOP>=1 (e[0]=='1'); merely "set to anything" is
 * surprising. Same intrusive 8-entry cache. */
bool gizmo_nlr_legacy_modeb_owns_for(const char *loop_name)
{
    if(!loop_name || !loop_name[0]) return false;
    struct entry_t { const char *name; int cached; };
    static entry_t cache[8] = {
        {nullptr,0},{nullptr,0},{nullptr,0},{nullptr,0},
        {nullptr,0},{nullptr,0},{nullptr,0},{nullptr,0}};
    for(int k = 0; k < 8; k++) {
        if(cache[k].name == loop_name) return cache[k].cached != 0;
        if(cache[k].name == nullptr) {
            char env_name[160];
            int j = 0;
            const char *prefix = "GIZMO_MODE_B_";
            for(int p = 0; prefix[p] && j < 150; p++) env_name[j++] = prefix[p];
            for(int p = 0; loop_name[p] && j < 158; p++) {
                env_name[j++] = (char)toupper((unsigned char)loop_name[p]);
            }
            env_name[j] = '\0';
            const char *e = getenv(env_name);
            cache[k].name   = loop_name;
            cache[k].cached = (e && e[0] == '1') ? 1 : 0;
            return cache[k].cached != 0;
        }
    }
    return false;
}

bool gizmo_nlr_force_mode_b_global(void) {
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_NLR_FORCE_MODEB");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}
bool gizmo_nlr_force_mode_a_global(void) {
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_NLR_FORCE_MODEA");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}
bool gizmo_nlr_oracle_enabled_global(void) {
    static int cached = -1;
    if(cached < 0) {
        const char *e = getenv("GIZMO_NLR_ORACLE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}
bool gizmo_nlr_oracle_enabled_for(const char *loop_name) {
    /* Per-loop env: GIZMO_<UPPERCASE_LOOP_NAME>_ORACLE. Lookup is per-call
     * but loop_name is a Spec::loop_name constexpr literal, so the env-name
     * derivation is deterministic; we cache via a small intrusive map of up
     * to 8 entries (cheap linear scan). 8 covers every Spec we expect through
     * 3c-3g; bump if a future stage exceeds. */
    if(!loop_name || !loop_name[0]) return false;
    struct entry_t { const char *name; int cached; };
    static entry_t cache[8] = {
        {nullptr,0},{nullptr,0},{nullptr,0},{nullptr,0},
        {nullptr,0},{nullptr,0},{nullptr,0},{nullptr,0}};
    for(int k = 0; k < 8; k++) {
        if(cache[k].name == loop_name) return cache[k].cached != 0;
        if(cache[k].name == nullptr) {
            char env_name[128];
            int j = 0;
            const char *prefix = "GIZMO_";
            for(int p = 0; prefix[p] && j < 120; p++) env_name[j++] = prefix[p];
            for(int p = 0; loop_name[p] && j < 120; p++) {
                env_name[j++] = (char)toupper((unsigned char)loop_name[p]);
            }
            const char *suffix = "_ORACLE";
            for(int p = 0; suffix[p] && j < 127; p++) env_name[j++] = suffix[p];
            env_name[j] = '\0';
            const char *e = getenv(env_name);
            cache[k].name   = loop_name;
            cache[k].cached = (e && e[0] == '1') ? 1 : 0;
            return cache[k].cached != 0;
        }
    }
    /* Cache full — fall through with a one-shot lookup, no caching. */
    return false;
}

/* ============================================================================
 * SIDX cache kind resolver (private to runner; see neighbor_loop_runner.h
 * SidxCacheKind enum doc).
 * ========================================================================== */

static gpu_spatial_index_t* nlr_resolve_sidx_cache(SidxCacheKind k,
                                                    const char *loop_name)
{
    switch(k) {
        case SidxCacheKind::AllTypes:
            return gpu_step_sidx_alltypes_ptr();
    }
    /* Unreachable in 3c.1 — kept exhaustive for compiler warnings on enum
     * additions. Other kinds land alongside their first caller. */
    fprintf(stderr, "neighbor_loop_runner: SidxCacheKind=%d not implemented "
            "for loop '%s' (3c.1 supports only AllTypes)\n",
            (int)k, loop_name ? loop_name : "?");
    fflush(stderr);
    endrun(81030);
    return nullptr;
}

/* ============================================================================
 * Mode B local self-rank helpers (3c.2)
 *
 * These three helpers STRUCTURALLY ENCODE the lazy-drift invariant from the
 * neighbor-loop binding contract:
 *
 *     collect_candidates_pre_drift<Spec>   — search backend (tree or brute)
 *                                            runs against possibly-stale P[j]
 *     lazy_drift_candidates<Spec>          — drift_particle on every j to
 *                                            All.Ti_Current (idempotent;
 *                                            duplicate j's between Mode B and
 *                                            Brute oracle are dedupe-free
 *                                            via drift_particle's
 *                                            time1==time0 early-return)
 *     evaluate_pairs_post_drift<Spec>      — calls Spec::pair_kernel via the
 *                                            same KOKKOS_INLINE_FUNCTION
 *                                            Spec::load_active /
 *                                            Spec::load_neighbor used by
 *                                            run_mode_a (host invocation here)
 *
 * Brute oracle uses the SAME drift epoch as Mode B (we collect both candidate
 * sets before drifting either — see run_mode_b_local_with_oracle). Annotation
 * on each helper makes the ordering enforcement structural.
 *
 * Self-rank only in 3c.2: candidates are local real P[] indices in
 * [0, num_local). Cross-rank P2P comes in 3c.3.
 *
 * Walker buffer sized to num_local — same convention as legacy
 * sinks/sink_environment_mode_b.cc:267-275 (worst-case SYMMETRIC, no h-bound
 * pre-pruning).
 * ========================================================================== */

/* WALK-ONLY. Does NOT mutate P[].Pos/Vel — drift_particle must not be
 * called from inside this helper. (Audited 2026-05-08: walker calls only
 * force_drift_node on tree-internal nodes, which is search-side state.)
 *
 * Per-active variant: walks args.P[i].Pos at radii[aa] for each
 * aa in [0,args.num_active). Self-rank queries.
 */
template <typename Spec>
static void collect_candidates_pre_drift(const neighbor_loop_args& args,
                                          const double *radii,
                                          DispatchPath backend,
                                          std::vector<std::vector<int>>& per_active_cands)
{
    const int N = args.num_active;
    const int num_local = ghost_get_num_local();
    per_active_cands.assign(N, std::vector<int>{});
    if(num_local <= 0) return;
    for(int aa = 0; aa < N; aa++) {
        const int i = args.active_list[aa];
        const double h_q = radii[aa];
        if(h_q <= 0) continue;
        std::vector<int>& cands = per_active_cands[aa];
        cands.assign(num_local, 0);
        double pos_arr[3] = {(double)args.P[i].Pos[0],
                              (double)args.P[i].Pos[1],
                              (double)args.P[i].Pos[2]};
        int n;
        if(backend == DispatchPath::ModeB_HostWalker) {
            n = mode_b_local_neighbor_walk(pos_arr, h_q,
                                            (unsigned int)Spec::neighbor_type_mask,
                                            Spec::search_mode,
                                            Spec::radius_policy,
                                            cands.data(), (int)cands.size());
        } else if(backend == DispatchPath::Brute_Oracle) {
            n = mode_b_local_brute_walk(pos_arr, h_q,
                                         (unsigned int)Spec::neighbor_type_mask,
                                         Spec::search_mode,
                                         Spec::radius_policy,
                                         cands.data(), (int)cands.size());
        } else {
            fprintf(stderr, "neighbor_loop_runner: collect_candidates_pre_drift "
                    "called with non-Mode-B/Brute backend (%d) for loop '%s'\n",
                    (int)backend, Spec::loop_name);
            fflush(stderr);
            endrun(81033);
            return;
        }
        if(n < 0) {
            fprintf(stderr, "neighbor_loop_runner: %s walker overflowed at "
                    "num_local=%d capacity for loop '%s' active_slot=%d. "
                    "Walker bug?\n",
                    (backend == DispatchPath::Brute_Oracle ? "brute" : "tree"),
                    num_local, Spec::loop_name, aa);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cands.resize(n);
    }
}

/* WALK-ONLY. Peer-side variant: walks against the LOCAL pool using each
 * remote query's pos/h_search drawn from peer_actives[k].q.{pos,h_search}.
 * Used for queries received from other ranks via the P2P transport.
 *
 * Note: pulls pos/h_search via Spec::ActiveData::q.{pos,h_search}. This is
 * the sink_env1-shaped contract (other specs migrating later may carry
 * pos/h_search differently); generalizing this access pattern is tracked
 * for stage 3d when the second spec joins. For 3c.3 sink_env1 is the only
 * caller, so the direct field access is acceptable.
 */
template <typename Spec>
static void collect_candidates_for_remote_queries(
    const std::vector<typename Spec::ActiveData>& peer_actives,
    DispatchPath backend,
    std::vector<std::vector<int>>& per_query_cands)
{
    const int K = (int)peer_actives.size();
    const int num_local = ghost_get_num_local();
    per_query_cands.assign(K, std::vector<int>{});
    if(num_local <= 0) return;
    for(int k = 0; k < K; k++) {
        const auto& a = peer_actives[k];
        const double h_q = (double)a.q.h_search;
        if(h_q <= 0) continue;
        std::vector<int>& cands = per_query_cands[k];
        cands.assign(num_local, 0);
        double pos_arr[3] = {(double)a.q.pos[0], (double)a.q.pos[1], (double)a.q.pos[2]};
        int n;
        if(backend == DispatchPath::ModeB_HostWalker) {
            n = mode_b_local_neighbor_walk(pos_arr, h_q,
                                            (unsigned int)Spec::neighbor_type_mask,
                                            Spec::search_mode,
                                            Spec::radius_policy,
                                            cands.data(), (int)cands.size());
        } else if(backend == DispatchPath::Brute_Oracle) {
            n = mode_b_local_brute_walk(pos_arr, h_q,
                                         (unsigned int)Spec::neighbor_type_mask,
                                         Spec::search_mode,
                                         Spec::radius_policy,
                                         cands.data(), (int)cands.size());
        } else {
            fprintf(stderr, "neighbor_loop_runner: collect_candidates_for_remote_queries"
                    " bad backend %d for loop '%s'\n", (int)backend, Spec::loop_name);
            fflush(stderr);
            endrun(81033);
            return;
        }
        if(n < 0) {
            fprintf(stderr, "neighbor_loop_runner: %s walker overflowed for remote "
                    "query at num_local=%d for loop '%s' k=%d. Walker bug?\n",
                    (backend == DispatchPath::Brute_Oracle ? "brute" : "tree"),
                    num_local, Spec::loop_name, k);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cands.resize(n);
    }
}

/* SAME drift epoch contract. Idempotent: drift_particle's time1==time0
 * early-return makes calling this multiple times (e.g. once each on
 * self_tree, self_brute, peer_tree, peer_brute candidate sets) safe. */
template <typename Spec>
static void lazy_drift_candidates(std::vector<std::vector<int>>& per_active_cands)
{
    for(auto& v : per_active_cands) {
        if(!v.empty()) {
            mode_b_lazy_drift_candidates(v.data(), (int)v.size());
        }
    }
}

/* Evaluates pair_kernel POST-DRIFT for a precomputed ActiveData[] paired with
 * candidate lists. Caller is responsible for having frozen actives[] before
 * any drift, and for having drifted the candidate union before calling.
 *
 * Decoupled from args.active_list: works for self-rank (actives built from
 * args.active_list) and peer-rank (actives built from received envelopes).
 * Same KOKKOS_INLINE_FUNCTION Spec::load_neighbor + Spec::pair_kernel either
 * way.
 */
template <typename Spec>
static void evaluate_pairs_post_drift(const NeighborLoopDeviceContextBase& ctx,
                                       const typename Spec::ActiveData *actives,
                                       int N,
                                       const std::vector<std::vector<int>>& per_active_cands,
                                       typename Spec::AccumData *accums)
{
    using NeighborData = typename Spec::NeighborData;
    using ScatterData  = typename Spec::ScatterData;
    for(int aa = 0; aa < N; aa++) {
        Spec::zero_accum(accums[aa]);
        const auto& a = actives[aa];
        ScatterData s{};                              /* NoScatter for ActiveReduceOnly */
        const auto& cands = per_active_cands[aa];
        for(size_t kk = 0; kk < cands.size(); kk++) {
            int j = cands[kk];
            IdentitySidecar id{};                     /* NoIdentity */
            NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
            Spec::pair_kernel(a, nb, accums[aa], s);
        }
    }
}

/* ============================================================================
 * run_mode_b_local<Spec> — single-rank Mode B path.
 *
 * Three-epoch staging contract (matches Mode A timing exactly except step 3
 * runs host-side per query instead of inside a Kokkos kernel):
 *   (1) host pre-drift: Spec::search_radius_host  → radii[num_active]
 *   (2) host pre-drift: Spec::populate_call_scalars_host → CallScalars cs
 *   (3) host post-drift: Spec::load_active per query (KOKKOS_INLINE_FUNCTION
 *                         invoked host-side; same UVM/Pos epoch as Mode A's
 *                         device kernel)
 *
 * No Kokkos kernels in this path → no fence (project rule from contract).
 * No GPU NGL build, no SIDX, no arena_acquire — Mode B's whole point is
 * escaping that overhead for tiny-N. `compare_accum`-gated brute oracle is
 * orchestrated by run_mode_b_local_with_oracle below.
 * ========================================================================== */

/* Helper: build the host-frozen ActiveData[] for self-rank actives. Called
 * BEFORE any drift so the active snapshot matches the legacy pack_query
 * epoch. (Codex nuance: this is NOT the same epoch as Mode A's device-staged
 * load_active post-NGL-build; documented at run_neighbor_loop dispatch site.)
 */
template <typename Spec>
static void build_self_actives_host_pre_drift(
    const neighbor_loop_args& args,
    const NeighborLoopDeviceContextBase& ctx,
    const double *radii,
    const typename Spec::CallScalars& cs,
    typename Spec::ActiveData *actives_out)
{
    const int N = args.num_active;
    for(int aa = 0; aa < N; aa++) {
        actives_out[aa] = Spec::load_active(ctx, aa, args.active_list[aa],
                                             radii[aa], cs);
    }
}

template <typename Spec>
static void run_mode_b_local(const neighbor_loop_args& args)
{
    using ActiveData = typename Spec::ActiveData;
    using AccumData  = typename Spec::AccumData;
    using DeviceCtx  = NeighborLoopDeviceContextBase;

    const int N = args.num_active;
    if(N <= 0) {
        /* Local-zero is legitimate (caller may enter unconditionally if
         * global_num_active > 0 — multi-rank Mode B requires that). */
        return;
    }

    /* (1) Host pre-drift: caller-supplied radii from external state. */
    std::vector<double> radii(N);
    for(int aa = 0; aa < N; aa++) {
        radii[aa] = Spec::search_radius_host(args, aa, args.active_list[aa]);
    }

    /* (2) Host pre-drift: per-call scalar globals into POD. */
    typename Spec::CallScalars cs = Spec::populate_call_scalars_host(args);

    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Freeze active snapshots host-side BEFORE drift. */
    std::vector<ActiveData> actives(N);
    build_self_actives_host_pre_drift<Spec>(args, ctx, radii.data(), cs, actives.data());

    /* Helper layout: collect → drift → evaluate. */
    std::vector<std::vector<int>> cand_modeB;
    collect_candidates_pre_drift<Spec>(args, radii.data(),
                                        DispatchPath::ModeB_HostWalker, cand_modeB);
    lazy_drift_candidates<Spec>(cand_modeB);

    std::vector<AccumData> accums(N);
    evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_modeB, accums.data());

    /* Host writeback — same code path as Mode A's writeback. */
    for(int aa = 0; aa < N; aa++) {
        Spec::apply_active_writeback(args, aa, args.active_list[aa], accums[aa]);
    }
}

/* run_mode_b_local_with_oracle<Spec>: collects BOTH Mode B and Brute candidate
 * sets BEFORE drifting either, then drifts the union (drift_particle is
 * idempotent so two separate lazy_drift calls dedupe naturally). Matches the
 * legacy oracle pattern in sinks/sink_environment_mode_b.cc:314-422.
 *
 * Mode B path is the result; Brute is the comparison. Mismatches print the
 * legacy line shape `[mode_b ORACLE MISMATCH ...]` so existing parser scripts
 * keep working.
 *
 * Brute oracle uses ONLY Mode B / Brute machinery — no GPU NGL, no SIDX, no
 * arena. Codex invariant: "the oracle's pair kernel is the SAME pair_kernel
 * Mode B runs; only the search differs." */
/* Mismatch print cap — shared between local and remote oracle paths so we
 * don't double the cap when both fire (e.g. 3c.3 multi-rank-with-oracle
 * runs the local path under NTask==1 dispatch and the remote path under
 * NTask>1). */
static const long long kMismatchPrintCap = 1024;

template <typename Spec>
static void emit_oracle_mismatch_if_any(int rank, int active_slot,
                                          const typename Spec::AccumData& tree,
                                          const typename Spec::AccumData& brute,
                                          const char *origin_tag,
                                          long long *print_count)
{
    double resid = Spec::compare_accum(tree, brute);
    if(!(resid <= Spec::accum_tolerance)) {  /* NaN-safe */
        if(*print_count < kMismatchPrintCap) {
            fprintf(stderr,
                    "[mode_b ORACLE MISMATCH rank=%d caller=%s origin=%s "
                    "active_slot=%d resid=%g (tol=%g)]\n",
                    rank, Spec::loop_name, origin_tag, active_slot, resid,
                    (double)Spec::accum_tolerance);
            fflush(stderr);
            (*print_count)++;
            if(*print_count == kMismatchPrintCap) {
                fprintf(stderr,
                        "[mode_b ORACLE rank=%d caller=%s] mismatch print cap "
                        "reached (%lld); suppressing further mismatches.\n",
                        rank, Spec::loop_name, (long long)kMismatchPrintCap);
                fflush(stderr);
            }
        }
    }
}

template <typename Spec>
static void run_mode_b_local_with_oracle(const neighbor_loop_args& args)
{
    using ActiveData = typename Spec::ActiveData;
    using AccumData  = typename Spec::AccumData;
    using DeviceCtx  = NeighborLoopDeviceContextBase;

    const int N = args.num_active;
    if(N <= 0) { return; }

    std::vector<double> radii(N);
    for(int aa = 0; aa < N; aa++) {
        radii[aa] = Spec::search_radius_host(args, aa, args.active_list[aa]);
    }
    typename Spec::CallScalars cs = Spec::populate_call_scalars_host(args);

    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Freeze actives BEFORE drift (same snapshot for tree and brute). */
    std::vector<ActiveData> actives(N);
    build_self_actives_host_pre_drift<Spec>(args, ctx, radii.data(), cs, actives.data());

    /* Collect BOTH BEFORE any drift — preserves identical pre-drift state for
     * each search backend. */
    std::vector<std::vector<int>> cand_modeB, cand_brute;
    collect_candidates_pre_drift<Spec>(args, radii.data(),
                                        DispatchPath::ModeB_HostWalker, cand_modeB);
    collect_candidates_pre_drift<Spec>(args, radii.data(),
                                        DispatchPath::Brute_Oracle, cand_brute);

    /* Drift the union. drift_particle's early-return on time1==time0 means
     * order doesn't matter and duplicates are free. */
    lazy_drift_candidates<Spec>(cand_modeB);
    lazy_drift_candidates<Spec>(cand_brute);

    std::vector<AccumData> accums_modeB(N);
    std::vector<AccumData> accums_brute(N);
    evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_modeB, accums_modeB.data());
    evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_brute, accums_brute.data());

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    static long long s_mismatch_print_count = 0;
    for(int aa = 0; aa < N; aa++) {
        emit_oracle_mismatch_if_any<Spec>(rank, aa, accums_modeB[aa],
                                            accums_brute[aa], "self",
                                            &s_mismatch_print_count);
    }

    /* Mode B path is the result. */
    for(int aa = 0; aa < N; aa++) {
        Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_modeB[aa]);
    }
}

/* ============================================================================
 * Mode B remote (multi-rank) helpers (3c.3)
 *
 * STRONG INVARIANT (codex-locked, both oracle and non-oracle paths):
 *   collect-all → drift-union → evaluate-all on each rank.
 *
 * On each rank, the local pool is walked TWICE per call: once for this
 * rank's own self-pair queries (its own actives), and once for queries
 * received from peers. Both candidate sets must be collected pre-drift,
 * then the drift covers their UNION (idempotent so duplicates are free),
 * then evaluation runs post-drift. The non-oracle path does NOT simplify
 * back to per-query drift — single shared structure, oracle off only
 * skips the brute candidate sets and the compare step.
 *
 * Sequence:
 *   stage 1 (active rank) build self radii, cs, frozen actives[]
 *   stage 2 (active rank) build envelopes, ALL peers in broadcast pattern
 *   stage 3 (this rank)   collect self_tree[, self_brute] pre-drift
 *   stage 4 (collective)  exchange queries (P2P) -> recv envelopes
 *   stage 5 (this rank)   flatten envelopes to peer_actives[] + provenance[]
 *   stage 6 (this rank)   collect peer_tree[, peer_brute] pre-drift
 *   stage 7 (this rank)   drift UNION of (self_tree, self_brute, peer_tree,
 *                                          peer_brute)
 *   stage 8 (this rank)   evaluate self_tree[, self_brute]; oracle compare
 *                                  self before any remote merge so prints
 *                                  isolate local-pool walker bugs
 *   stage 9 (this rank)   evaluate peer_tree[, peer_brute]; oracle compare
 *                                  peer-side BEFORE reply transport (peer-
 *                                  rank prints isolate peer-pool bugs)
 *   stage 10 (collective) exchange replies (tree result is what ships)
 *   stage 11 (active rank) merge replies via Spec::merge_accum, ascending
 *                                  rank for FP-reproducible order
 *   stage 12 (active rank) writeback
 *
 * Codex nuance (acknowledged): the host-frozen actives[] match Mode B /
 * legacy pack_query epoch, NOT Mode A's device-staged post-NGL-build
 * epoch. Oracle in this path is Mode B tree vs brute on the same frozen
 * query — it does NOT cross-validate against Mode A's active epoch.
 * ========================================================================== */

template <typename Spec, bool ORACLE>
static void run_mode_b_remote_impl(const neighbor_loop_args& args)
{
    using ActiveData    = typename Spec::ActiveData;
    using AccumData     = typename Spec::AccumData;
    using Envelope      = NlrQueryEnvelope<ActiveData>;
    using ReplyEnvelope = NlrReplyEnvelope<AccumData>;
    using DeviceCtx     = NeighborLoopDeviceContextBase;

    static_assert(std::is_trivially_copyable<Envelope>::value,
        "NlrQueryEnvelope must be trivially-copyable for byte-level MPI transfer");
    static_assert(std::is_trivially_copyable<ReplyEnvelope>::value,
        "NlrReplyEnvelope must be trivially-copyable for byte-level MPI transfer");

    const int N    = args.num_active;     /* may be 0 on this rank; collective entry */
    const int nt   = NTask;
    const int rank = ThisTask;

    /* Stage 1: host pre-drift staging on the active rank. */
    std::vector<double> radii(N > 0 ? N : 0);
    for(int aa = 0; aa < N; aa++) {
        radii[aa] = Spec::search_radius_host(args, aa, args.active_list[aa]);
    }
    typename Spec::CallScalars cs = Spec::populate_call_scalars_host(args);

    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Freeze actives host-side pre-drift (same snapshot used for self-pair
     * AND for ship-to-peers; codex requirement to prevent self/remote epoch
     * skew on the active rank). */
    std::vector<ActiveData> actives(N);
    if(N > 0) {
        build_self_actives_host_pre_drift<Spec>(args, ctx, radii.data(), cs,
                                                  actives.data());
    }

    /* Stage 2: build envelopes per peer (broadcast pattern: every peer
     * receives ALL of this rank's actives). Self-pair handled locally;
     * envelopes for self entry stay empty. */
    std::vector<std::vector<Envelope>> queries_per_peer(nt);
    if(N > 0) {
        for(int p = 0; p < nt; p++) {
            if(p == rank) continue;
            queries_per_peer[p].reserve(N);
            for(int aa = 0; aa < N; aa++) {
                Envelope env;
                env.origin_slot = aa;
                env.origin_rank = rank;
                env.active      = actives[aa];
                queries_per_peer[p].push_back(env);
            }
        }
    }

    /* Stage 3: collect SELF candidate sets PRE-DRIFT. */
    std::vector<std::vector<int>> cand_self_tree, cand_self_brute;
    if(N > 0) {
        collect_candidates_pre_drift<Spec>(args, radii.data(),
                                            DispatchPath::ModeB_HostWalker,
                                            cand_self_tree);
        if(ORACLE) {
            collect_candidates_pre_drift<Spec>(args, radii.data(),
                                                DispatchPath::Brute_Oracle,
                                                cand_self_brute);
        }
    }

    /* Stage 4: exchange queries (collective). Every rank participates even
     * if N == 0 (peers may have queries directed at this rank's pool). */
    auto state = mode_b_exchange_queries<Envelope>(queries_per_peer);

    /* Stage 5: flatten received envelopes and build provenance map.
     * provenance[k] carries:
     *   - source_peer / source_qi: where in state.recv_queries[][] this k
     *     came from (used for unflattening replies back into per-peer arrays).
     *   - origin_slot / origin_rank: copied from the received envelope; ride
     *     into the REPLY envelope so the active rank can merge by slot
     *     without relying on transport ordering (codex Option C — symmetric
     *     query and reply envelopes). */
    std::vector<ActiveData> peer_actives;
    struct Provenance {
        int source_peer; int source_qi;
        int origin_slot; int origin_rank;
    };
    std::vector<Provenance> peer_provenance;
    int total_recv = 0;
    for(int p = 0; p < nt; p++) total_recv += state.recv_counts[p];
    peer_actives.reserve(total_recv);
    peer_provenance.reserve(total_recv);
    for(int p = 0; p < nt; p++) {
        if(p == rank) continue;
        for(int qi = 0; qi < state.recv_counts[p]; qi++) {
            const Envelope& env = state.recv_queries[p][qi];
            /* Sanity: envelope's origin_rank should equal sender p. */
            if(env.origin_rank != p) {
                fprintf(stderr, "[neighbor_loop_runner ABORT rank=%d caller=%s] "
                        "envelope origin_rank=%d but received from peer=%d "
                        "(qi=%d). Transport corruption?\n",
                        rank, Spec::loop_name, env.origin_rank, p, qi);
                fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            peer_actives.push_back(env.active);
            peer_provenance.push_back({p, qi, env.origin_slot, env.origin_rank});
        }
    }

    /* Stage 6: collect PEER candidate sets PRE-DRIFT (against MY local pool). */
    std::vector<std::vector<int>> cand_peer_tree, cand_peer_brute;
    collect_candidates_for_remote_queries<Spec>(peer_actives,
                                                 DispatchPath::ModeB_HostWalker,
                                                 cand_peer_tree);
    if(ORACLE) {
        collect_candidates_for_remote_queries<Spec>(peer_actives,
                                                     DispatchPath::Brute_Oracle,
                                                     cand_peer_brute);
    }

    /* Stage 7: drift the UNION of all candidate sets that touch MY pool. */
    if(N > 0) {
        lazy_drift_candidates<Spec>(cand_self_tree);
        if(ORACLE) lazy_drift_candidates<Spec>(cand_self_brute);
    }
    lazy_drift_candidates<Spec>(cand_peer_tree);
    if(ORACLE) lazy_drift_candidates<Spec>(cand_peer_brute);

    /* Stage 8: evaluate SELF post-drift; oracle compare BEFORE merge. */
    std::vector<AccumData> accums_self(N);
    std::vector<AccumData> accums_self_brute;
    if(N > 0) {
        evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N,
                                          cand_self_tree, accums_self.data());
        if(ORACLE) {
            accums_self_brute.assign(N, AccumData{});
            evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N,
                                              cand_self_brute, accums_self_brute.data());
            static long long s_self_mismatch_count = 0;
            for(int aa = 0; aa < N; aa++) {
                emit_oracle_mismatch_if_any<Spec>(rank, aa, accums_self[aa],
                                                    accums_self_brute[aa], "self",
                                                    &s_self_mismatch_count);
            }
        }
    }

    /* Stage 9: evaluate PEER queries post-drift. Reply per-query is the
     * tree result; oracle compares peer-side before reply transport. */
    const int K = (int)peer_actives.size();
    std::vector<AccumData> peer_replies(K);
    std::vector<AccumData> peer_replies_brute;
    if(K > 0) {
        evaluate_pairs_post_drift<Spec>(ctx, peer_actives.data(), K,
                                          cand_peer_tree, peer_replies.data());
        if(ORACLE) {
            peer_replies_brute.assign(K, AccumData{});
            evaluate_pairs_post_drift<Spec>(ctx, peer_actives.data(), K,
                                              cand_peer_brute, peer_replies_brute.data());
            static long long s_peer_mismatch_count = 0;
            for(int k = 0; k < K; k++) {
                /* Print the ORIGIN active slot (from the query envelope) so
                 * the message identifies which active on which rank asked,
                 * independent of any peer-side flattening order. */
                emit_oracle_mismatch_if_any<Spec>(rank, peer_provenance[k].origin_slot,
                                                    peer_replies[k],
                                                    peer_replies_brute[k],
                                                    "peer", &s_peer_mismatch_count);
            }
        }
    }

    /* Stage 10: build reply envelopes (origin_slot/rank copied from each
     * received query envelope), unflatten into per-peer arrays via the
     * provenance map, then exchange. Reply envelope makes the active-side
     * merge transport-order independent (codex Option C consistency). */
    std::vector<std::vector<ReplyEnvelope>> replies_per_peer(nt);
    for(int p = 0; p < nt; p++) {
        replies_per_peer[p].assign(state.recv_counts[p], ReplyEnvelope{});
    }
    for(int k = 0; k < K; k++) {
        const Provenance& pv = peer_provenance[k];
        ReplyEnvelope& re = replies_per_peer[pv.source_peer][pv.source_qi];
        re.origin_slot = pv.origin_slot;
        re.origin_rank = pv.origin_rank;
        re.accum       = peer_replies[k];
    }

    auto recv_replies = mode_b_exchange_replies<Envelope, ReplyEnvelope>(
        replies_per_peer, state);

    /* Stage 11: merge replies into accums_self by envelope.origin_slot.
     * Pinned deterministic order: ascending peer rank (self contribution
     * already in accums_self from stage 8). Asserts each reply envelope's
     * origin_rank == ThisTask — a transport-corruption sanity check. */
    if(N > 0) {
        for(int p = 0; p < nt; p++) {
            if(p == rank) continue;
            const int q_to_p = state.sent_counts[p];
            for(int qi = 0; qi < q_to_p; qi++) {
                const ReplyEnvelope& re = recv_replies[p][qi];
                if(re.origin_rank != rank) {
                    fprintf(stderr, "[neighbor_loop_runner ABORT rank=%d caller=%s] "
                            "reply envelope origin_rank=%d != ThisTask=%d from "
                            "peer=%d qi=%d. Transport/peer-side corruption?\n",
                            rank, Spec::loop_name, re.origin_rank, rank, p, qi);
                    fflush(stderr);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                const int slot = re.origin_slot;
                if(slot < 0 || slot >= N) {
                    fprintf(stderr, "[neighbor_loop_runner ABORT rank=%d caller=%s] "
                            "reply envelope slot %d out of range [0,%d) from peer %d.\n",
                            rank, Spec::loop_name, slot, N, p);
                    fflush(stderr);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                Spec::merge_accum(accums_self[slot], re.accum);
            }
        }
    }

    /* Stage 12: writeback. */
    for(int aa = 0; aa < N; aa++) {
        Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_self[aa]);
    }
}

template <typename Spec>
static void run_mode_b_remote(const neighbor_loop_args& args) {
    run_mode_b_remote_impl<Spec, /*ORACLE=*/false>(args);
}
template <typename Spec>
static void run_mode_b_remote_with_oracle(const neighbor_loop_args& args) {
    run_mode_b_remote_impl<Spec, /*ORACLE=*/true>(args);
}

/* ============================================================================
 * run_mode_a<Spec> — generic Mode A path through the GPU NGL pipeline.
 *
 * Three-epoch staging contract (see neighbor_loop_runner.h doc):
 *   (1) host pre-arena: Spec::search_radius_host  → radii_uvm[num_active]
 *   (2) host pre-arena: Spec::populate_call_scalars_host → CallScalars cs
 *   (3) device post-NGL-build: Spec::load_active → d_actives[num_active]
 *
 * Both Kokkos launches go through gizmo_gpu_kernel_launch which wraps
 * parallel_for + fence + check_last_error (declarations/gpu_dispatch_templates.h).
 * No additional explicit fence is needed before host-side d_accums readback
 * — the launch helper already fenced.
 * ========================================================================== */

template <typename Spec>
static void run_mode_a(const neighbor_loop_args& args)
{
    using ActiveData   = typename Spec::ActiveData;
    using AccumData    = typename Spec::AccumData;
    using ScatterData  = typename Spec::ScatterData;
    using NeighborData = typename Spec::NeighborData;
    using CallScalars  = typename Spec::CallScalars;
    using DeviceCtx    = NeighborLoopDeviceContextBase;

    const int N = args.num_active;
    if(N <= 0) {
        /* Defensive: caller already early-outs in collective dispatch when
         * global_num_active == 0, but local-zero with global-positive is
         * legitimate (peer rank with no actives still must enter the
         * collective in Mode B; Mode A has no collective work, just no-op). */
        return;
    }

    /* (1) Host, pre-arena: stage caller-supplied radii from external state. */
    double *radii_uvm = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        N * sizeof(double));
    for(int aa = 0; aa < N; aa++) {
        radii_uvm[aa] = Spec::search_radius_host(args, aa, args.active_list[aa]);
    }

    /* (2) Host, pre-arena: capture per-call scalar globals into a POD. */
    CallScalars cs = Spec::populate_call_scalars_host(args);

    /* Arena + freshness (matches sinks/sink_environment_gpu.cc:76,86). The
     * caller is responsible for the args.CellP=NULL-when-no-gas decision;
     * runner does not read All.TotN_gas. */
    GIZMO_GPU_ENSURE_ALL_FRESH(neighbor_loop_runner);
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total, args.P, args.CellP);
    struct particle_data *P_gpu    = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = (args.CellP != nullptr)
                                        ? gpu_particles_arena_CellP() : nullptr;

    /* NGL build using pre-arena radii. SIDX cache resolved from spec. */
    gpu_neighbor_list_t gnl;
    gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                       Spec::loop_name);
    gpu_ngb_list_build(P_gpu, args.num_total,
                       args.active_list, N,
                       Spec::search_mode,
                       (int)Spec::neighbor_type_mask,
                       &gnl, sidx,
                       1.0, radii_uvm, NULL, Spec::loop_name);

    /* UVM-allocate ActiveData[] and AccumData[] arrays. */
    ActiveData *d_actives = (ActiveData *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(ActiveData));
    AccumData *d_accums = (AccumData *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(AccumData));

    /* Build DeviceContext (base only in 3c.1; future specs may extend). */
    DeviceCtx ctx;
    ctx.P         = P_gpu;
    ctx.CellP     = CellP_gpu;
    ctx.num_total = args.num_total;

    /* (3) Device, post-NGL-build: stage ActiveData. Same device epoch as
     * the legacy GPU lambda's q-packing. d_active is gnl-resident. */
    int *d_active_idx = gnl.d_active;
    {
        gizmo_gpu_kernel_launch("nlr_stage_active", N, KOKKOS_LAMBDA(int aa) {
            d_actives[aa] = Spec::load_active(ctx, aa, d_active_idx[aa],
                                              radii_uvm[aa], cs);
        });
    }

    /* Pair-kernel launch — generic over Spec. */
    {
        int *offsets   = gnl.offsets;
        int *neighbors = gnl.neighbors;
        gizmo_gpu_kernel_launch(Spec::loop_name, N, KOKKOS_LAMBDA(int aa) {
            Spec::zero_accum(d_accums[aa]);
            const ActiveData& a = d_actives[aa];
            ScatterData s{};                     /* NoScatter for ActiveReduceOnly */
            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                IdentitySidecar id{};            /* NoIdentity */
                NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                Spec::pair_kernel(a, nb, d_accums[aa], s);
            }
        });
    }
    /* Both launches fenced internally by gizmo_gpu_kernel_launch. UVM is
     * coherent → host can read d_accums directly. */

    /* (4) Host writeback via spec. */
    for(int aa = 0; aa < N; aa++) {
        Spec::apply_active_writeback(args, aa, args.active_list[aa], d_accums[aa]);
    }

    /* Cleanup. SIDX cache pointer passed so the free leaves cached storage
     * intact for sink_feed/sink_swk reuse (matches existing
     * sink_environment_gpu.cc:261 idiom). */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_actives);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
    gpu_ngb_list_free(&gnl, sidx);
    gpu_particles_arena_mark_clean_after_scatter(Spec::loop_name);
}

/* ============================================================================
 * SFINAE: detect optional Spec::modeb_threshold_sum / _max constexpr.
 * If absent, fall back to caller-supplied default.
 * ========================================================================== */

template <typename Spec, typename = void>
struct nlr_has_threshold_sum : std::false_type {};
template <typename Spec>
struct nlr_has_threshold_sum<Spec, decltype((void)Spec::modeb_threshold_sum)>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_threshold_max : std::false_type {};
template <typename Spec>
struct nlr_has_threshold_max<Spec, decltype((void)Spec::modeb_threshold_max)>
    : std::true_type {};

template <typename Spec>
static int nlr_spec_threshold_sum(int fallback) {
    if constexpr (nlr_has_threshold_sum<Spec>::value) {
        return (int)Spec::modeb_threshold_sum;
    } else {
        return fallback;
    }
}
template <typename Spec>
static int nlr_spec_threshold_max(int fallback) {
    if constexpr (nlr_has_threshold_max<Spec>::value) {
        return (int)Spec::modeb_threshold_max;
    } else {
        return fallback;
    }
}

/* ============================================================================
 * Public entry: run_neighbor_loop<Spec>
 * ========================================================================== */

template <typename Spec>
void run_neighbor_loop(const neighbor_loop_args& args)
{
    /* ---- Compile-time spec consistency ---- */

    /* WritePattern ↔ AccumData/ScatterData consistency. */
    static_assert(Spec::write_pattern != WritePattern::ActiveReduceOnly ||
                  std::is_same<typename Spec::ScatterData, NoScatter>::value,
        "ActiveReduceOnly requires ScatterData == NoScatter");
    static_assert(Spec::write_pattern != WritePattern::NeighborScatter ||
                  std::is_same<typename Spec::AccumData, NoAccum>::value,
        "NeighborScatter requires AccumData == NoAccum");

    /* POD/device-copy contract — captured by value into Kokkos lambdas
     * and/or staged into UVM arrays. Trivially-copyable is the binding
     * requirement. */
    static_assert(std::is_trivially_copyable<typename Spec::CallScalars>::value,
        "Spec::CallScalars must be trivially-copyable (lambda capture by value)");
    static_assert(std::is_trivially_copyable<typename Spec::ActiveData>::value,
        "Spec::ActiveData must be trivially-copyable (UVM-staged)");
    static_assert(std::is_trivially_copyable<typename Spec::NeighborData>::value,
        "Spec::NeighborData must be trivially-copyable (built per-pair on device)");
    static_assert(std::is_trivially_copyable<typename Spec::AccumData>::value,
        "Spec::AccumData must be trivially-copyable (UVM-staged)");

    /* ---- Dispatch (3c.3) ----
     *
     * Selection precedence (highest first):
     *   1. GIZMO_NLR_FORCE_MODEA + GIZMO_NLR_FORCE_MODEB both set → endrun.
     *   2. GIZMO_NLR_FORCE_MODEA=1 → Mode A unconditionally.
     *   3. GIZMO_NLR_FORCE_MODEB=1 → Mode B (local if NTask==1, remote else),
     *                                 overrides legacy SPIKE precedence
     *                                 (testers' explicit knob).
     *   4. Legacy SPIKE owns Mode B (e.g. GIZMO_MODE_B_SINK_ENV1=1) →
     *      runner falls to Mode A; the legacy SPIKE branch in the caller
     *      (sink_environment.cc) makes the Mode B decision. Retired in 3c.5.
     *   5. Threshold dispatch: if (sum_active>0 && sum_active<=TS &&
     *      max_active<=TM) → Mode B; else Mode A. Hierarchy of TS/TM:
     *      per-loop env > global env > Spec::modeb_threshold_{sum,max}
     *      constexpr (default 64/64).
     *
     * Codex nuance (active-epoch caveat): Mode B host-frozen actives[] are
     * NOT bit-equivalent to Mode A's device-staged post-NGL-build actives.
     * Oracle in this dispatch is Mode B tree vs Mode B brute on the same
     * frozen query — it does NOT cross-validate Mode A. Mode A vs Mode B
     * active-epoch consistency is a separate concern, deferred.
     */
    const bool force_a   = gizmo_nlr_force_mode_a_global();
    const bool force_b   = gizmo_nlr_force_mode_b_global();
    const bool oracle_on = gizmo_nlr_oracle_enabled_global() ||
                            gizmo_nlr_oracle_enabled_for(Spec::loop_name);
    const bool legacy_owns = gizmo_nlr_legacy_modeb_owns_for(Spec::loop_name);

    if(force_a && force_b) {
        fprintf(stderr, "neighbor_loop_runner: both GIZMO_NLR_FORCE_MODEA and "
                "GIZMO_NLR_FORCE_MODEB set for loop '%s' — these are mutually "
                "exclusive. Pick one.\n", Spec::loop_name);
        fflush(stderr);
        endrun(81034);
    }

    /* One-shot warning when legacy SPIKE owns Mode B selection. */
    if(legacy_owns && !force_b) {
        static int s_legacy_warned[8] = {0,0,0,0,0,0,0,0};
        static const char *s_legacy_warned_name[8] = {nullptr};
        bool already = false;
        int free_slot = -1;
        for(int k = 0; k < 8; k++) {
            if(s_legacy_warned_name[k] == Spec::loop_name) { already = true; break; }
            if(s_legacy_warned_name[k] == nullptr && free_slot < 0) free_slot = k;
        }
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(!already && rank == 0 && free_slot >= 0) {
            s_legacy_warned_name[free_slot] = Spec::loop_name;
            s_legacy_warned[free_slot] = 1;
            fprintf(stderr, "[neighbor_loop_runner caller=%s] legacy SPIKE env "
                    "(GIZMO_MODE_B_<LOOP>=1) is set; runner threshold dispatch "
                    "DISABLED for this loop. Legacy SPIKE owns Mode B selection. "
                    "Will be retired in 3c.5.\n", Spec::loop_name);
            fflush(stderr);
        }
    }

    /* Threshold dispatch. Allreduce sum + max of args.num_active.
     * Skipped when force-mode envs are set (cheap path). When legacy SPIKE
     * owns the loop, runner forces Mode A regardless of threshold. */
    bool select_mode_b = force_b;
    if(!force_a && !force_b && !legacy_owns) {
        int local_act = args.num_active;
        int sum_act = 0, max_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_act, &max_act, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        /* Spec::modeb_threshold_{sum,max} via SFINAE; default 64/64. */
        const int spec_default_sum = nlr_spec_threshold_sum<Spec>(64);
        const int spec_default_max = nlr_spec_threshold_max<Spec>(64);
        const int TS = gizmo_nlr_modeb_threshold_sum_for(Spec::loop_name, spec_default_sum);
        const int TM = gizmo_nlr_modeb_threshold_max_for(Spec::loop_name, spec_default_max);
        select_mode_b = (sum_act > 0) && (sum_act <= TS) && (max_act <= TM);
    }

    /* Optional dispatch trace. Rank-0 only to avoid spam. */
    if(gizmo_nlr_dispatch_trace_enabled()) {
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0) {
            const char *path =
                force_a       ? "mode_a (force)" :
                force_b       ? (NTask > 1 ? "mode_b_remote (force)" : "mode_b_local (force)") :
                legacy_owns   ? "mode_a (legacy_owns)" :
                select_mode_b ? (NTask > 1 ? "mode_b_remote (threshold)" : "mode_b_local (threshold)") :
                                "mode_a (threshold)";
            fprintf(stderr, "[NLR DISPATCH caller=%s path=%s NTask=%d local_active=%d oracle=%d]\n",
                    Spec::loop_name, path, NTask, args.num_active, (int)oracle_on);
            fflush(stderr);
        }
    }

    if(select_mode_b) {
        if(NTask <= 1) {
            if(oracle_on) run_mode_b_local_with_oracle<Spec>(args);
            else          run_mode_b_local<Spec>(args);
        } else {
            if(oracle_on) run_mode_b_remote_with_oracle<Spec>(args);
            else          run_mode_b_remote<Spec>(args);
        }
        return;
    }

    /* Mode A path. Oracle on Mode A is a no-op (oracle compares Mode B vs
     * Brute; no Brute-vs-Mode-A oracle in scope for 3c.x). */
    run_mode_a<Spec>(args);
}

/* ============================================================================
 * Explicit template instantiations — one per migrated caller.
 *
 * Forgetting an instantiation = clean linker error at the call site. Each
 * Spec must opt in here and is implicitly acknowledging the
 * Spec::sidx_cache_kind = AllTypes (the only kind implemented in 3c.1)
 * via its constexpr declaration; cf. nlr_resolve_sidx_cache.
 * ========================================================================== */

template void run_neighbor_loop<SinkEnv1Spec>(const neighbor_loop_args&);

/* Per-TU GPU All-mirror sync function (paired with GIZMO_GPU_ENSURE_ALL_FRESH
 * in run_mode_a). Same idiom as sink_environment_gpu.cc:388. */
GPU_ALL_SYNC_FUNC(neighbor_loop_runner)
