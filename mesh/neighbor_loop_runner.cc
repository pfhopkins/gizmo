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

int gizmo_nlr_default_modeb_threshold_sum(void) { return 0; }
int gizmo_nlr_default_modeb_threshold_max(void) { return 0; }

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

/* SAME drift epoch as Mode B. Idempotent: drift_particle's time1==time0
 * early-return makes calling this twice (e.g. once on Mode B candidates,
 * once on Brute candidates) safe. */
template <typename Spec>
static void lazy_drift_candidates(std::vector<std::vector<int>>& per_active_cands)
{
    for(auto& v : per_active_cands) {
        if(!v.empty()) {
            mode_b_lazy_drift_candidates(v.data(), (int)v.size());
        }
    }
}

template <typename Spec>
static void evaluate_pairs_post_drift(const neighbor_loop_args& args,
                                       const NeighborLoopDeviceContextBase& ctx,
                                       const double *radii,
                                       const typename Spec::CallScalars& cs,
                                       const std::vector<std::vector<int>>& per_active_cands,
                                       typename Spec::AccumData *accums)
{
    using ActiveData   = typename Spec::ActiveData;
    using NeighborData = typename Spec::NeighborData;
    using ScatterData  = typename Spec::ScatterData;
    const int N = args.num_active;
    for(int aa = 0; aa < N; aa++) {
        Spec::zero_accum(accums[aa]);
        const int i = args.active_list[aa];
        ActiveData a = Spec::load_active(ctx, aa, i, radii[aa], cs);
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

template <typename Spec>
static void run_mode_b_local(const neighbor_loop_args& args)
{
    using AccumData = typename Spec::AccumData;
    using DeviceCtx = NeighborLoopDeviceContextBase;

    const int N = args.num_active;
    if(N <= 0) {
        /* Local-zero with global-positive (peer with no actives). 3c.2 is
         * single-rank — this branch is mostly defensive against caller
         * shape; 3c.3 makes it load-bearing for collective P2P. */
        return;
    }

    /* (1) Host pre-drift: caller-supplied radii from external state. */
    std::vector<double> radii(N);
    for(int aa = 0; aa < N; aa++) {
        radii[aa] = Spec::search_radius_host(args, aa, args.active_list[aa]);
    }

    /* (2) Host pre-drift: per-call scalar globals into POD. */
    typename Spec::CallScalars cs = Spec::populate_call_scalars_host(args);

    /* DeviceContext (host-side bytes; ctx.P / ctx.CellP point at the host
     * particle arrays directly — same KOKKOS_INLINE_FUNCTION load_active /
     * load_neighbor that runs on UVM in Mode A). */
    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Helper layout: collect → drift → evaluate. SAME drift epoch as Brute. */
    std::vector<std::vector<int>> cand_modeB;
    collect_candidates_pre_drift<Spec>(args, radii.data(),
                                        DispatchPath::ModeB_HostWalker, cand_modeB);
    lazy_drift_candidates<Spec>(cand_modeB);

    std::vector<AccumData> accums(N);
    evaluate_pairs_post_drift<Spec>(args, ctx, radii.data(), cs, cand_modeB,
                                     accums.data());

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
template <typename Spec>
static void run_mode_b_local_with_oracle(const neighbor_loop_args& args)
{
    using AccumData = typename Spec::AccumData;
    using DeviceCtx = NeighborLoopDeviceContextBase;

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
    evaluate_pairs_post_drift<Spec>(args, ctx, radii.data(), cs, cand_modeB,
                                     accums_modeB.data());
    evaluate_pairs_post_drift<Spec>(args, ctx, radii.data(), cs, cand_brute,
                                     accums_brute.data());

    /* Compare per-active. Mismatch shape kept compatible with legacy
     * sinks/sink_environment_mode_b.cc:374 line for parser reuse. Auto-cap
     * the print volume so a long mismatched run doesn't hose stderr. */
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    static long long s_mismatch_print_count = 0;
    static const long long kMismatchPrintCap = 1024;
    for(int aa = 0; aa < N; aa++) {
        double resid = Spec::compare_accum(accums_modeB[aa], accums_brute[aa]);
        if(!(resid <= Spec::accum_tolerance)) {  /* NaN-safe */
            if(s_mismatch_print_count < kMismatchPrintCap) {
                fprintf(stderr,
                        "[mode_b ORACLE MISMATCH rank=%d caller=%s active_slot=%d "
                        "resid=%g (tol=%g)]\n",
                        rank, Spec::loop_name, aa, resid,
                        (double)Spec::accum_tolerance);
                fflush(stderr);
                s_mismatch_print_count++;
                if(s_mismatch_print_count == kMismatchPrintCap) {
                    fprintf(stderr,
                            "[mode_b ORACLE rank=%d caller=%s] mismatch print cap "
                            "reached (%lld); suppressing further mismatches.\n",
                            rank, Spec::loop_name, (long long)kMismatchPrintCap);
                    fflush(stderr);
                }
            }
        }
    }

    /* Mode B path is the result. */
    for(int aa = 0; aa < N; aa++) {
        Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_modeB[aa]);
    }
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

    /* ---- Dispatch ---- */

    /* 3c.2 dispatch: force-mode envs select between Mode A and Mode B local.
     * Threshold-based dispatch (Allreduce sum/max) is deferred to 3c.3 — it
     * needs multi-rank coordination, and 3c.2 is single-rank only.
     *
     * Decision tree:
     *   GIZMO_NLR_FORCE_MODEA=1 → Mode A unconditionally (regression baseline).
     *                              Oracle is ignored for Mode A in 3c.2 — the
     *                              oracle pairs Mode B against Brute.
     *   GIZMO_NLR_FORCE_MODEB=1 → Mode B local. Oracle (if enabled) runs Brute
     *                              alongside.
     *   default                 → Mode A (3c.2 has no auto Mode B trigger;
     *                              3c.3 adds threshold-based selection).
     *
     * Co-existence with legacy GIZMO_MODE_B_SINK_ENV1 SPIKE: the SPIKE branch
     * in sinks/sink_environment.cc:87 short-circuits BEFORE entering this
     * function, so the two env-gate sets do not overlap. User must enable
     * EITHER legacy SPIKE or new NLR force, not both. 3c.5 retires the
     * legacy SPIKE. */
    const bool force_a = gizmo_nlr_force_mode_a_global();
    const bool force_b = gizmo_nlr_force_mode_b_global();
    const bool oracle_on = gizmo_nlr_oracle_enabled_global() ||
                            gizmo_nlr_oracle_enabled_for(Spec::loop_name);

    if(force_a && force_b) {
        fprintf(stderr, "neighbor_loop_runner: both GIZMO_NLR_FORCE_MODEA and "
                "GIZMO_NLR_FORCE_MODEB set for loop '%s' — these are mutually "
                "exclusive. Pick one.\n", Spec::loop_name);
        fflush(stderr);
        endrun(81034);
    }

    if(force_b) {
        /* HARD GUARD: 3c.2 Mode B is single-rank only. The local walker
         * returns candidates only from [0, ghost_get_num_local()) on this
         * rank — multi-rank would silently drop remote contributions. P2P
         * exchange lands in 3c.3; until then, force-Mode-B on NTask>1 must
         * abort loudly rather than produce wrong physics. */
        int ntask = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &ntask);
        if(ntask > 1) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(rank == 0) {
                fprintf(stderr,
                    "neighbor_loop_runner: GIZMO_NLR_FORCE_MODEB=1 set with "
                    "NTask=%d for loop '%s', but 3c.2 Mode B is single-rank "
                    "only (no remote P2P). Multi-rank Mode B lands in 3c.3. "
                    "Run on -np 1 or unset GIZMO_NLR_FORCE_MODEB.\n",
                    ntask, Spec::loop_name);
                fflush(stderr);
            }
            endrun(81035);
        }
        if(oracle_on) {
            run_mode_b_local_with_oracle<Spec>(args);
        } else {
            run_mode_b_local<Spec>(args);
        }
        return;
    }

    /* Mode A path. Oracle on Mode A is a no-op in 3c.2 (oracle compares Mode B
     * vs Brute; no Brute-vs-Mode-A oracle in scope). */
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
