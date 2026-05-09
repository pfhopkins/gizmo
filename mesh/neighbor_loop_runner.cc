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
 * SAME drift epoch as Mode B. Multi-rank peer-to-peer is deferred to 3c.3 — runtime
 * abort guards GIZMO_NLR_FORCE_MODE=B on NTask>1.
 *
 * Architecture binding contract:
 *   ~/.claude/memory/reference_neighbor_loop_contract.md
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdarg>                                   /* va_list, vfprintf for nlr_warn_once_rank0 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <utility>                                   /* std::declval for SFINAE hook detection */
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"          /* MUST precede allvars.h: #define All All_dev for device code */
#include "../declarations/allvars.h"
#include "../declarations/lifecycle_counters.h"      /* g_global_drift_counter etc. for Mode B corridor */
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/gpu_dispatch_templates.h"  /* gizmo_gpu_kernel_launch */

#include "neighbor_loop_runner.h"
#include "gpu_neighbor_list.h"
#include "kernel.h"  /* MUST precede sink_env1_loop.h (kernel_main, NEAREST_XYZ) */
#include "ghost_writeback.h"             /* ghost_get_num_local */
#include "ghost_symlist_lifecycle.h"     /* gizmo_request_filtered_ghost_import_fresh, ghost_exchange_cleanup */
#include "mode_b_local_walker.h"         /* mode_b_local_neighbor_walk, brute, lazy_drift */

#include <vector>
#include <cmath>
#include <cctype>

#include "mode_b_p2p_transport.h"  /* mode_b_exchange_queries / _replies */

/* Spec instantiations. Each #include declares one Spec type whose explicit
 * template instantiation appears at the bottom of this file. */
#include "../sinks/sink_env1_loop.h"

/* ============================================================================
 * Caller-side helpers (declared in runner.h).
 *
 * Out-of-line so changes to global plumbing (NumPart, P, CellP fetch site,
 * ghost-safety-factor source, etc.) are caller-invisible.
 * ========================================================================== */

NlrCommonScalars nlr_common_scalars_from_all(void)
{
    NlrCommonScalars s;
    s.cf_atime                = All.cf_atime;
    s.cf_a2inv                = All.cf_a2inv;
    s.cf_a3inv                = All.cf_a3inv;
    s.cf_hubble_a             = All.cf_hubble_a;
    s.newton_G                = All.G;
    s.hubble                  = All.HubbleParam;
    s.comoving_integration_on = All.ComovingIntegrationOn;
    return s;
}

neighbor_loop_args nlr_default_args(void)
{
    neighbor_loop_args args;
    args.P                   = P;
    args.CellP               = (All.TotN_gas > 0) ? CellP : nullptr;
    args.num_total           = NumPart;
    args.active_list         = nullptr;       /* caller fills */
    args.num_active          = 0;             /* caller fills */
    args.aux                 = nullptr;       /* caller fills */
    args.ghost_safety_factor = gizmo_ghost_safety_factor();
    return args;
}

void nlr_free_active_list(int *active_list)
{
    if(active_list) myfree(active_list);
}

/* ============================================================================
 * Env-gate functions.
 *
 * Cached statics — env read once on first call. Threshold dispatch is the
 * default Mode A vs Mode B selector; force-mode envs override.
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

/* ============================================================================
 * NLR env config (Pass B.i unified API).
 *
 * Single canonical surface for diagnostic, control, and spike (cross-
 * validation) env vars. Old names are accepted as aliases for one cycle
 * with rank-0 deprecation warnings; explicit retire queued for the next
 * cleanup pass after Pass B.
 *
 * Conflict policy and alias precedence: see the comment block on the
 * declarations at the top of mesh/neighbor_loop_runner.h.
 *
 * All accessors are first-use cached and lock-free (single load per call).
 * Initialization may call endrun on a hard conflict; endrun is collective,
 * and TACC env vars are uniform across ranks, so all ranks reach the same
 * decision and abort together.
 * ========================================================================== */

namespace {

/* One-shot rank-0 warning helper. Cached set keyed by string-literal
 * pointer (so each call site occupies one slot). Cap is generous; if hit,
 * subsequent warnings are silently dropped — they are diagnostics, not
 * correctness gates. */
static void nlr_warn_once_rank0(const char *key, const char *fmt, ...)
{
    if(ThisTask != 0) return;
    static const char *seen[32];
    static int seen_n = 0;
    for(int i = 0; i < seen_n; i++) { if(seen[i] == key) return; }
    if(seen_n < 32) { seen[seen_n++] = key; }
    fprintf(stderr, "[NLR env] ");
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static bool nlr_env_is_one(const char *name) {
    const char *e = getenv(name);
    return (e && e[0] == '1' && e[1] == '\0');
}

/* Initialize diag level. Reads new var first, then aliases. */
static int nlr_init_diag_level(void)
{
    const char *raw = getenv("GIZMO_NLR_DIAG");
    int new_set_level = -1;
    if(raw && raw[0]) {
        char *endp = nullptr;
        long v = strtol(raw, &endp, 10);
        if(!endp || *endp != '\0' || v < 0 || v > 3) {
            if(ThisTask == 0) {
                fprintf(stderr, "[NLR env] FATAL: GIZMO_NLR_DIAG=\"%s\" must be in {0,1,2,3}.\n",
                        raw);
                fflush(stderr);
            }
            endrun(81100);
        }
        new_set_level = (int)v;
    }

    bool old_phase0   = nlr_env_is_one("GIZMO_PHASE0_DIAG");
    bool old_dispatch = nlr_env_is_one("GIZMO_NLR_DISPATCH_TRACE");

    int level;
    if(new_set_level >= 0) {
        level = new_set_level;
        if(old_phase0) {
            nlr_warn_once_rank0("alias_phase0_overridden",
                "GIZMO_PHASE0_DIAG ignored; GIZMO_NLR_DIAG=%d takes precedence.", level);
        }
        if(old_dispatch) {
            nlr_warn_once_rank0("alias_dispatch_overridden",
                "GIZMO_NLR_DISPATCH_TRACE ignored; GIZMO_NLR_DIAG=%d takes precedence.", level);
        }
    } else {
        level = 0;
        if(old_phase0) {
            nlr_warn_once_rank0("alias_phase0_deprecated",
                "GIZMO_PHASE0_DIAG=1 is deprecated; use GIZMO_NLR_DIAG=1 instead.");
            if(level < 1) level = 1;
        }
        if(old_dispatch) {
            nlr_warn_once_rank0("alias_dispatch_deprecated",
                "GIZMO_NLR_DISPATCH_TRACE=1 is deprecated; use GIZMO_NLR_DIAG=2 instead.");
            if(level < 2) level = 2;
        }
    }

    if(level == 3) {
        nlr_warn_once_rank0("level_3_reserved",
            "GIZMO_NLR_DIAG=3: level 3 currently has no extra diagnostics; "
            "reserved for future scalar-only extensions. Behaving as level 2.");
    }

    return level;
}

/* Initialize force mode. Conflict cases endrun (collective). */
static NlrForceMode nlr_init_force_mode(void)
{
    const char *raw_new = getenv("GIZMO_NLR_FORCE_MODE");
    bool new_set = (raw_new && raw_new[0]);
    bool old_a   = nlr_env_is_one("GIZMO_NLR_FORCE_MODEA");
    bool old_b   = nlr_env_is_one("GIZMO_NLR_FORCE_MODEB");

    if(new_set && (old_a || old_b)) {
        if(ThisTask == 0) {
            fprintf(stderr, "[NLR env] FATAL: GIZMO_NLR_FORCE_MODE='%s' is set AND "
                    "old GIZMO_NLR_FORCE_MODEA=%d / GIZMO_NLR_FORCE_MODEB=%d are set. "
                    "Use only one. Old names are deprecated.\n",
                    raw_new, (int)old_a, (int)old_b);
            fflush(stderr);
        }
        endrun(81101);
    }
    if(old_a && old_b) {
        if(ThisTask == 0) {
            fprintf(stderr, "[NLR env] FATAL: GIZMO_NLR_FORCE_MODEA and GIZMO_NLR_FORCE_MODEB "
                    "are both set; mutually exclusive.\n");
            fflush(stderr);
        }
        endrun(81102);
    }

    if(new_set) {
        if(raw_new[0] == 'A' && raw_new[1] == '\0') return NlrForceMode::A;
        if(raw_new[0] == 'B' && raw_new[1] == '\0') return NlrForceMode::B;
        if(ThisTask == 0) {
            fprintf(stderr, "[NLR env] FATAL: GIZMO_NLR_FORCE_MODE=\"%s\" must be 'A' or 'B'.\n",
                    raw_new);
            fflush(stderr);
        }
        endrun(81103);
    }

    if(old_a) {
        nlr_warn_once_rank0("alias_force_modea_deprecated",
            "GIZMO_NLR_FORCE_MODEA=1 is deprecated; use GIZMO_NLR_FORCE_MODE=A instead.");
        return NlrForceMode::A;
    }
    if(old_b) {
        nlr_warn_once_rank0("alias_force_modeb_deprecated",
            "GIZMO_NLR_FORCE_MODEB=1 is deprecated; use GIZMO_NLR_FORCE_MODE=B instead.");
        return NlrForceMode::B;
    }
    return NlrForceMode::None;
}

static bool nlr_init_spike_accum_dump(void)
{
    bool new_set = nlr_env_is_one("GIZMO_NLR_SPIKE_ACCUM_DUMP");
    bool old_set = nlr_env_is_one("GIZMO_MODE_B_XVAL_DUMP");
    if(new_set) {
        if(old_set) {
            nlr_warn_once_rank0("alias_xval_dump_overridden",
                "GIZMO_MODE_B_XVAL_DUMP ignored; GIZMO_NLR_SPIKE_ACCUM_DUMP takes precedence.");
        }
        return true;
    }
    if(old_set) {
        nlr_warn_once_rank0("alias_xval_dump_deprecated",
            "GIZMO_MODE_B_XVAL_DUMP=1 is deprecated; use GIZMO_NLR_SPIKE_ACCUM_DUMP=1 instead.");
        return true;
    }
    return false;
}

static bool nlr_init_spike_nb_dump(void)
{
    bool new_set = nlr_env_is_one("GIZMO_NLR_SPIKE_NB_DUMP");
    bool old_set = nlr_env_is_one("GIZMO_MODE_B_XVAL_NB_DUMP");
    if(new_set) {
        if(old_set) {
            nlr_warn_once_rank0("alias_xval_nb_dump_overridden",
                "GIZMO_MODE_B_XVAL_NB_DUMP ignored; GIZMO_NLR_SPIKE_NB_DUMP takes precedence.");
        }
        return true;
    }
    if(old_set) {
        nlr_warn_once_rank0("alias_xval_nb_dump_deprecated",
            "GIZMO_MODE_B_XVAL_NB_DUMP=1 is deprecated; use GIZMO_NLR_SPIKE_NB_DUMP=1 instead.");
        return true;
    }
    return false;
}

} /* anonymous namespace */

int gizmo_nlr_diag_level(void)
{
    static int cached = -1;
    if(cached < 0) cached = nlr_init_diag_level();
    return cached;
}

NlrForceMode gizmo_nlr_force_mode(void)
{
    static int  cached_int   = -1;          /* -1 = uninit; 0/1/2 = enum */
    if(cached_int < 0) cached_int = (int)nlr_init_force_mode();
    return (NlrForceMode)cached_int;
}

bool gizmo_nlr_spike_accum_dump_enabled(void)
{
    static int cached = -1;
    if(cached < 0) cached = nlr_init_spike_accum_dump() ? 1 : 0;
    return cached != 0;
}

bool gizmo_nlr_spike_nb_dump_enabled(void)
{
    static int cached = -1;
    if(cached < 0) cached = nlr_init_spike_nb_dump() ? 1 : 0;
    return cached != 0;
}

/* Adapters — preserve existing call-site names. */
bool gizmo_nlr_phase0_diag_enabled(void)    { return gizmo_nlr_diag_level() >= 1; }
bool gizmo_nlr_dispatch_trace_enabled(void) { return gizmo_nlr_diag_level() >= 2; }
bool gizmo_nlr_xval_dump_enabled(void)      { return gizmo_nlr_spike_accum_dump_enabled(); }
bool gizmo_nlr_xval_nb_dump_enabled(void)   { return gizmo_nlr_spike_nb_dump_enabled(); }

/* ============================================================================
 * NeighborLoopPlan path predicates — single source of truth keyed on path.
 *
 * New paths (future Mode C, dual-tree large-N, etc.) extend the switch
 * statements below; never add fields to NeighborLoopPlan.
 * ========================================================================== */
bool nlr_path_uses_imported_ghosts(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return true;
        case NeighborLoopPlan::Path::ModeB_Local:  return false;
        case NeighborLoopPlan::Path::ModeB_Remote: return false;
    }
    return false;
}

bool nlr_path_uses_gpu_arena(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return true;
        case NeighborLoopPlan::Path::ModeB_Local:  return false;
        case NeighborLoopPlan::Path::ModeB_Remote: return false;
    }
    return false;
}

bool nlr_path_permits_global_numpart_mutation(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return true;
        case NeighborLoopPlan::Path::ModeB_Local:  return false;
        case NeighborLoopPlan::Path::ModeB_Remote: return false;
    }
    return false;
}

bool nlr_path_uses_lazy_drift(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return false;
        case NeighborLoopPlan::Path::ModeB_Local:  return true;
        case NeighborLoopPlan::Path::ModeB_Remote: return true;
    }
    return false;
}

const char *nlr_path_label(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return "gpu_ngl";
        case NeighborLoopPlan::Path::ModeB_Local:  return "mode_b_local";
        case NeighborLoopPlan::Path::ModeB_Remote: return "mode_b_remote";
    }
    return "unknown";
}

/* ============================================================================
 * StageTimer — internal RAII helper for PHASE0 timing (3c.4b)
 *
 * `target == nullptr` means phase0 is off: ctor and dtor do nothing except
 * a single predictable nullptr branch. NO MPI_Wtime call when off — codex
 * constraint that "MPI_Wtime is not zero overhead" addressed at the call
 * site, not just at the gating env var.
 *
 * Targets are accumulators: multiple StageTimer scopes can target the same
 * field (e.g. Mode B remote's dt_collect spans BOTH self and peer pre-drift
 * collection — two scopes accumulate). dt_total spans the whole runner call
 * and is set explicitly at top-level, not via this helper.
 * ========================================================================== */
namespace {
struct StageTimer {
    double *target;
    double t0;
    explicit StageTimer(double *tgt) : target(tgt), t0(0.0) {
        if(target) t0 = MPI_Wtime();
    }
    ~StageTimer() {
        if(target) *target += MPI_Wtime() - t0;
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;
};
} /* anonymous namespace */

bool gizmo_nlr_force_mode_b_global(void) { return gizmo_nlr_force_mode() == NlrForceMode::B; }
bool gizmo_nlr_force_mode_a_global(void) { return gizmo_nlr_force_mode() == NlrForceMode::A; }
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
 * [0, num_local). Cross-rank peer-to-peer comes in 3c.3.
 *
 * Walker buffer sized to num_local (worst-case SYMMETRIC, no h-bound
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
    /* Per-call ownership: each inner vector owns its capacity for the
     * duration of this call. .assign(N, {}) clobbers any stale residue
     * from a previous call. Stage 4 contract: walker appends via
     * push_back; geometric growth handles any-size match set without
     * imposing the previous full-pool .assign(num_local, 0) cost
     * (~24 MB × N_active on fire_m11i). */
    per_active_cands.assign(N, std::vector<int>{});
    if(num_local <= 0) return;
    for(int aa = 0; aa < N; aa++) {
        const int i = args.active_list[aa];
        const double h_q = radii[aa];
        if(h_q <= 0) continue;
        std::vector<int>& cands = per_active_cands[aa];
        cands.clear();
        if(cands.capacity() == 0) cands.reserve(64); /* small initial; grows geometrically */
        double pos_arr[3] = {(double)args.P[i].Pos[0],
                              (double)args.P[i].Pos[1],
                              (double)args.P[i].Pos[2]};
        if(backend == DispatchPath::ModeB_HostWalker) {
            mode_b_local_neighbor_walk(pos_arr, h_q,
                                        (unsigned int)Spec::neighbor_type_mask,
                                        Spec::search_mode,
                                        Spec::radius_policy,
                                        cands);
        } else if(backend == DispatchPath::Brute_Oracle) {
            mode_b_local_brute_walk(pos_arr, h_q,
                                     (unsigned int)Spec::neighbor_type_mask,
                                     Spec::search_mode,
                                     Spec::radius_policy,
                                     cands);
        } else {
            fprintf(stderr, "neighbor_loop_runner: collect_candidates_pre_drift "
                    "called with non-Mode-B/Brute backend (%d) for loop '%s'\n",
                    (int)backend, Spec::loop_name);
            fflush(stderr);
            endrun(81033);
            return;
        }
    }
}

/* WALK-ONLY. Peer-side variant: walks against the LOCAL pool using each
 * remote query's pos/h_search drawn from peer_actives[k].{pos,h_search}.
 * Used for queries received from other ranks via the peer-to-peer transport.
 *
 * Pulls pos/h_search directly from Spec::ActiveData fields. The current
 * convention is that every Spec exposes `pos` and `h_search` as flat
 * ActiveData fields (matches sink_env1_loop.h template). Generalizing
 * this access pattern (e.g. to a Spec::query_pos/query_h trait pair) is
 * tracked for the runner-template-hardening pass.
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
        const auto& active = peer_actives[k];
        const double h_q = (double)active.h_search;
        if(h_q <= 0) continue;
        std::vector<int>& cands = per_query_cands[k];
        cands.clear();
        if(cands.capacity() == 0) cands.reserve(64);
        double pos_arr[3] = {(double)active.pos[0], (double)active.pos[1], (double)active.pos[2]};
        if(backend == DispatchPath::ModeB_HostWalker) {
            mode_b_local_neighbor_walk(pos_arr, h_q,
                                        (unsigned int)Spec::neighbor_type_mask,
                                        Spec::search_mode,
                                        Spec::radius_policy,
                                        cands);
        } else if(backend == DispatchPath::Brute_Oracle) {
            mode_b_local_brute_walk(pos_arr, h_q,
                                     (unsigned int)Spec::neighbor_type_mask,
                                     Spec::search_mode,
                                     Spec::radius_policy,
                                     cands);
        } else {
            fprintf(stderr, "neighbor_loop_runner: collect_candidates_for_remote_queries"
                    " bad backend %d for loop '%s'\n", (int)backend, Spec::loop_name);
            fflush(stderr);
            endrun(81033);
            return;
        }
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
 *   (1) host pre-drift: Spec::search_radius  → radii[num_active]
 *   (2) host pre-drift: Spec::populate_call_scalars → CallScalars cs
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

/* ============================================================================
 * Diagnostic active-dump emit helper (3c.4a)
 *
 * Iterates per-active and calls Spec::diagnostic_dump_active when the env
 * gate is on AND the spec opts in via the SFINAE-detected hook.
 *
 * Timing invariant: post-Spec::apply_active_writeback, pre-caller-side
 * SinkTempInfo scatter. Fires inside the runner before returning to the
 * caller, so the caller's ghost_write_detector_end / scatter loop have
 * not yet run. (Legacy emit fired post-ghost-detector but pre-scatter; the
 * runner-driven emit fires pre-ghost-detector but still pre-scatter.
 * Accumulator values are byte-identical between the two timing points —
 * writeback is the last write into the per-active accumulator buffer
 * before scatter — so the cross-validation dump line content is unchanged.)
 *
 * Mode A path supplies actives_or_null = d_actives (UVM-resident; host-
 * coherent post-fence). Mode B local/remote supply host-frozen actives.
 * Spec hooks read view.active->{pos,h_search,...} fields when needed.
 * ========================================================================== */
template <typename Spec>
static void runner_emit_active_dumps(const neighbor_loop_args& args,
                                      const typename Spec::AccumData *accums,
                                      const typename Spec::ActiveData *actives_or_null,
                                      const char *path)
{
    if(!gizmo_nlr_xval_dump_enabled()) return;
    if constexpr (spec_has_dump_active<Spec>::value) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        for(int aa = 0; aa < args.num_active; aa++) {
            ActiveDumpView<Spec> v;
            v.rank        = rank;
            v.origin_rank = rank;
            v.origin_slot = aa;
            v.active_slot = aa;
            v.path        = path;
            v.call_id     = 0;
            v.args        = &args;
            v.active      = actives_or_null ? &actives_or_null[aa] : nullptr;
            v.accum       = &accums[aa];
            Spec::diagnostic_dump_active(v);
        }
        std::fflush(stdout);
    }
}

template <typename Spec>
static void run_mode_b_local(const neighbor_loop_args& args, const double *radii,
                             RunnerStageTimer *tim = nullptr)
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

    /* (1) Radii are runner-staged and passed in; pointer is call-lifetime
     * only (do not store). See run_neighbor_loop contract in the header. */

    /* (2) Host pre-drift: per-call scalar globals into POD. */
    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Freeze active snapshots host-side BEFORE drift. */
    std::vector<ActiveData> actives(N);
    build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs, actives.data());

    /* Helper layout: collect → drift → evaluate. */
    std::vector<std::vector<int>> cand_modeB;
    {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        collect_candidates_pre_drift<Spec>(args, radii,
                                            DispatchPath::ModeB_HostWalker, cand_modeB);
    }
    {
        StageTimer t(tim ? &tim->dt_drift : nullptr);
        lazy_drift_candidates<Spec>(cand_modeB);
    }

    std::vector<AccumData> accums(N);
    {
        StageTimer t(tim ? &tim->dt_walk_self : nullptr);
        evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_modeB, accums.data());
    }

    /* Host writeback — same code path as Mode A's writeback. */
    {
        StageTimer t(tim ? &tim->dt_writeback : nullptr);
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], accums[aa]);
        }
    }
    runner_emit_active_dumps<Spec>(args, accums.data(), actives.data(), "mode_b");
}

/* run_mode_b_local_with_oracle<Spec>: collects BOTH Mode B and Brute candidate
 * sets BEFORE drifting either, then drifts the union (drift_particle is
 * idempotent so two separate lazy_drift calls dedupe naturally).
 *
 * Mode B path is the result; Brute is the comparison. Mismatches print the
 * line shape `[mode_b ORACLE MISMATCH ...]` (parser-stable across the runner
 * extraction).
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
static void run_mode_b_local_with_oracle(const neighbor_loop_args& args, const double *radii,
                                         RunnerStageTimer *tim = nullptr)
{
    using ActiveData = typename Spec::ActiveData;
    using AccumData  = typename Spec::AccumData;
    using DeviceCtx  = NeighborLoopDeviceContextBase;

    const int N = args.num_active;
    if(N <= 0) { return; }

    /* Radii runner-staged and passed in; pointer is call-lifetime only. */
    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Freeze actives BEFORE drift (same snapshot for tree and brute). */
    std::vector<ActiveData> actives(N);
    build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs, actives.data());

    /* Collect BOTH BEFORE any drift — preserves identical pre-drift state for
     * each search backend. PHASE0 timing covers ONLY the Mode B path stages,
     * not the Brute oracle (diagnostic-only, not production cost). */
    std::vector<std::vector<int>> cand_modeB, cand_brute;
    {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        collect_candidates_pre_drift<Spec>(args, radii,
                                            DispatchPath::ModeB_HostWalker, cand_modeB);
    }
    collect_candidates_pre_drift<Spec>(args, radii,
                                        DispatchPath::Brute_Oracle, cand_brute);

    /* Drift the union. drift_particle's early-return on time1==time0 means
     * order doesn't matter and duplicates are free. */
    {
        StageTimer t(tim ? &tim->dt_drift : nullptr);
        lazy_drift_candidates<Spec>(cand_modeB);
    }
    lazy_drift_candidates<Spec>(cand_brute);

    std::vector<AccumData> accums_modeB(N);
    std::vector<AccumData> accums_brute(N);
    {
        StageTimer t(tim ? &tim->dt_walk_self : nullptr);
        evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_modeB, accums_modeB.data());
    }
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
    {
        StageTimer t(tim ? &tim->dt_writeback : nullptr);
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_modeB[aa]);
        }
    }
    runner_emit_active_dumps<Spec>(args, accums_modeB.data(), actives.data(), "mode_b");
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
 *   stage 4 (collective)  exchange queries (peer-to-peer) -> recv envelopes
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
static void run_mode_b_remote_impl(const neighbor_loop_args& args, const double *radii,
                                   RunnerStageTimer *tim = nullptr)
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

    /* Radii runner-staged and passed in; pointer is call-lifetime only. */
    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    /* Freeze actives host-side pre-drift (same snapshot used for self-pair
     * AND for ship-to-peers; codex requirement to prevent self/remote epoch
     * skew on the active rank). */
    std::vector<ActiveData> actives(N);
    if(N > 0) {
        build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs,
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
        {
            StageTimer t(tim ? &tim->dt_collect : nullptr);
            collect_candidates_pre_drift<Spec>(args, radii,
                                                DispatchPath::ModeB_HostWalker,
                                                cand_self_tree);
        }
        if(ORACLE) {
            collect_candidates_pre_drift<Spec>(args, radii,
                                                DispatchPath::Brute_Oracle,
                                                cand_self_brute);
        }
    }

    /* Stage 4: exchange queries (collective). Every rank participates even
     * if N == 0 (peers may have queries directed at this rank's pool). */
    auto state = [&]{
        StageTimer t(tim ? &tim->dt_exchange_q : nullptr);
        return mode_b_exchange_queries<Envelope>(queries_per_peer);
    }();

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
    {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        collect_candidates_for_remote_queries<Spec>(peer_actives,
                                                     DispatchPath::ModeB_HostWalker,
                                                     cand_peer_tree);
    }
    if(ORACLE) {
        collect_candidates_for_remote_queries<Spec>(peer_actives,
                                                     DispatchPath::Brute_Oracle,
                                                     cand_peer_brute);
    }

    /* Stage 7: drift the UNION of all candidate sets that touch MY pool. */
    {
        StageTimer t(tim ? &tim->dt_drift : nullptr);
        if(N > 0) lazy_drift_candidates<Spec>(cand_self_tree);
        lazy_drift_candidates<Spec>(cand_peer_tree);
    }
    if(ORACLE) {
        if(N > 0) lazy_drift_candidates<Spec>(cand_self_brute);
        lazy_drift_candidates<Spec>(cand_peer_brute);
    }

    /* Stage 8: evaluate SELF post-drift; oracle compare BEFORE merge. */
    std::vector<AccumData> accums_self(N);
    std::vector<AccumData> accums_self_brute;
    if(N > 0) {
        {
            StageTimer t(tim ? &tim->dt_walk_self : nullptr);
            evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N,
                                              cand_self_tree, accums_self.data());
        }
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
        {
            StageTimer t(tim ? &tim->dt_walk_peer : nullptr);
            evaluate_pairs_post_drift<Spec>(ctx, peer_actives.data(), K,
                                              cand_peer_tree, peer_replies.data());
        }
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

    auto recv_replies = [&]{
        StageTimer t(tim ? &tim->dt_exchange_r : nullptr);
        return mode_b_exchange_replies<Envelope, ReplyEnvelope>(replies_per_peer, state);
    }();

    /* Stage 11: merge replies into accums_self by envelope.origin_slot.
     * Pinned deterministic order: ascending peer rank (self contribution
     * already in accums_self from stage 8). Asserts each reply envelope's
     * origin_rank == ThisTask — a transport-corruption sanity check. */
    {
        StageTimer t(tim ? &tim->dt_reduce : nullptr);
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
    }

    /* Stage 12: writeback. */
    {
        StageTimer t(tim ? &tim->dt_writeback : nullptr);
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_self[aa]);
        }
    }
    runner_emit_active_dumps<Spec>(args, accums_self.data(),
                                    (N > 0) ? actives.data() : nullptr, "mode_b");
}

template <typename Spec>
static void run_mode_b_remote(const neighbor_loop_args& args, const double *radii,
                              RunnerStageTimer *tim = nullptr) {
    run_mode_b_remote_impl<Spec, /*ORACLE=*/false>(args, radii, tim);
}
template <typename Spec>
static void run_mode_b_remote_with_oracle(const neighbor_loop_args& args, const double *radii,
                                          RunnerStageTimer *tim = nullptr) {
    run_mode_b_remote_impl<Spec, /*ORACLE=*/true>(args, radii, tim);
}

/* ============================================================================
 * run_mode_a<Spec> — generic Mode A path through the GPU NGL pipeline.
 *
 * Three-epoch staging contract (see neighbor_loop_runner.h doc):
 *   (1) host pre-arena: Spec::search_radius  → radii_uvm[num_active]
 *   (2) host pre-arena: Spec::populate_call_scalars → CallScalars cs
 *   (3) device post-NGL-build: Spec::load_active → d_actives[num_active]
 *
 * Both Kokkos launches go through gizmo_gpu_kernel_launch which wraps
 * parallel_for + fence + check_last_error (declarations/gpu_dispatch_templates.h).
 * No additional explicit fence is needed before host-side d_accums readback
 * — the launch helper already fenced.
 * ========================================================================== */

template <typename Spec>
static void run_mode_a(const neighbor_loop_args& args, const double *radii,
                       RunnerStageTimer *tim = nullptr)
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

    /* (1) Host, pre-arena: stage caller-supplied radii into UVM for the
     * device-visible NGL build. Source `radii` is runner-staged on host
     * (call-lifetime only); we copy into shared/UVM so gpu_ngb_list_build
     * and the device pair kernel can read the per-active values. */
    double *radii_uvm = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
        N * sizeof(double));
    for(int aa = 0; aa < N; aa++) {
        radii_uvm[aa] = radii[aa];
    }

    /* (2) Host, pre-arena: capture per-call scalar globals into a POD. */
    CallScalars cs = Spec::populate_call_scalars(args);

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
    {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        gpu_ngb_list_build(P_gpu, args.num_total,
                           args.active_list, N,
                           Spec::search_mode,
                           (int)Spec::neighbor_type_mask,
                           &gnl, sidx,
                           1.0, radii_uvm, NULL, Spec::loop_name);
    }

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
        StageTimer t(tim ? &tim->dt_walk_self : nullptr);
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
    {
        StageTimer t(tim ? &tim->dt_writeback : nullptr);
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], d_accums[aa]);
        }
    }

    /* (5) Optional diagnostic dumps (3c.4a). Order matches legacy whole-log
     * shape: NB lines first (legacy emitted from sink_environment_gpu.cc
     * BEFORE returning to sink_environment.cc, where the accumulator dump
     * fired), then accumulator lines. NB dump is Mode A only; the GPU CSR
     * host-copy is allocated only when its env gate is on (zero overhead
     * off). Accumulator dump preserves legacy line shape. */
    if constexpr (spec_has_dump_neighbor_list<Spec>::value) {
        if(gizmo_nlr_xval_nb_dump_enabled() && gnl.total_pairs > 0) {
            std::vector<int> nbrs_host((size_t)gnl.total_pairs);
            gpu_ngb_copy_neighbors_to_host(&gnl, nbrs_host.data());
            int *offsets = gnl.offsets;
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            for(int aa = 0; aa < N; aa++) {
                NeighborListDumpView<Spec> v;
                v.rank          = rank;
                v.origin_rank   = rank;
                v.origin_slot   = aa;
                v.active_slot   = aa;
                v.path          = "gpu_ngl";
                v.call_id       = 1;
                v.args          = &args;
                v.active        = &d_actives[aa];
                v.candidate_ids = nbrs_host.data() + offsets[aa];
                v.n_candidates  = offsets[aa + 1] - offsets[aa];
                Spec::diagnostic_dump_neighbor_list(v);
            }
            std::fflush(stdout);
        }
    }
    runner_emit_active_dumps<Spec>(args, d_accums, d_actives, "gpu_ngl");

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
 * SFINAE: detect optional Spec::uses_* lifecycle traits (default false) and
 * dispatch to the corresponding hook methods (default no-op). The runner
 * gates each hook on (a) the trait being true AND (b) whether the chosen
 * path imports ghosts (per-Spec audit decided imported-ghost-only for the
 * detector + writeback + sidechannel hooks; future Specs may differ).
 * ========================================================================== */

/* Trait detection: Spec::uses_ghost_write_detector / _ghost_writeback /
 * _sidechannel_writeback. Absent ⇒ false. */
template <typename Spec, typename = void>
struct nlr_has_uses_ghost_write_detector : std::false_type {};
template <typename Spec>
struct nlr_has_uses_ghost_write_detector<Spec, decltype((void)Spec::uses_ghost_write_detector)>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_uses_ghost_writeback : std::false_type {};
template <typename Spec>
struct nlr_has_uses_ghost_writeback<Spec, decltype((void)Spec::uses_ghost_writeback)>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_uses_sidechannel_writeback : std::false_type {};
template <typename Spec>
struct nlr_has_uses_sidechannel_writeback<Spec, decltype((void)Spec::uses_sidechannel_writeback)>
    : std::true_type {};

template <typename Spec> static constexpr bool nlr_uses_ghost_write_detector_v() {
    if constexpr (nlr_has_uses_ghost_write_detector<Spec>::value) {
        return Spec::uses_ghost_write_detector;
    } else { return false; }
}
template <typename Spec> static constexpr bool nlr_uses_ghost_writeback_v() {
    if constexpr (nlr_has_uses_ghost_writeback<Spec>::value) {
        return Spec::uses_ghost_writeback;
    } else { return false; }
}
template <typename Spec> static constexpr bool nlr_uses_sidechannel_writeback_v() {
    if constexpr (nlr_has_uses_sidechannel_writeback<Spec>::value) {
        return Spec::uses_sidechannel_writeback;
    } else { return false; }
}

/* Hook-method detection: Spec::ghost_write_detector_begin etc. Absent ⇒
 * runner skips the call (compile-time short-circuit; no runtime cost). */
template <typename Spec, typename = void>
struct nlr_has_hook_gwd_begin : std::false_type {};
template <typename Spec>
struct nlr_has_hook_gwd_begin<Spec,
    decltype(Spec::ghost_write_detector_begin(std::declval<const neighbor_loop_args&>(),
                                              std::declval<const NeighborLoopPlan&>()))>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_hook_gwd_end : std::false_type {};
template <typename Spec>
struct nlr_has_hook_gwd_end<Spec,
    decltype(Spec::ghost_write_detector_end(std::declval<const neighbor_loop_args&>(),
                                             std::declval<const NeighborLoopPlan&>()))>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_hook_gwb_begin : std::false_type {};
template <typename Spec>
struct nlr_has_hook_gwb_begin<Spec,
    decltype(Spec::ghost_writeback_begin(std::declval<const neighbor_loop_args&>(),
                                         std::declval<const NeighborLoopPlan&>()))>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_hook_gwb_end : std::false_type {};
template <typename Spec>
struct nlr_has_hook_gwb_end<Spec,
    decltype(Spec::ghost_writeback_end(std::declval<const neighbor_loop_args&>(),
                                       std::declval<const NeighborLoopPlan&>()))>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_hook_swb_begin : std::false_type {};
template <typename Spec>
struct nlr_has_hook_swb_begin<Spec,
    decltype(Spec::sidechannel_writeback_begin(std::declval<const neighbor_loop_args&>(),
                                               std::declval<const NeighborLoopPlan&>()))>
    : std::true_type {};

template <typename Spec, typename = void>
struct nlr_has_hook_swb_end : std::false_type {};
template <typename Spec>
struct nlr_has_hook_swb_end<Spec,
    decltype(Spec::sidechannel_writeback_end(std::declval<const neighbor_loop_args&>(),
                                             std::declval<const NeighborLoopPlan&>()))>
    : std::true_type {};

/* Dispatch wrappers. Each gates on:
 *   (a) `uses_*` trait true (Spec opted in)
 *   (b) `nlr_path_uses_imported_ghosts(plan.path)` — for SinkEnv1Spec these
 *        hooks are imported-ghost-only per Stage 2c audit. The path gate
 *        IS the policy; the hook trait is the Spec opt-in.
 *   (c) hook method exists (SFINAE; absent ⇒ silent no-op).
 * Future Specs that need a different gating may add their own dispatch
 * helpers; this set covers SinkEnv1Spec E1. */

template <typename Spec>
static void nlr_dispatch_ghost_write_detector_begin(const neighbor_loop_args& args,
                                                    const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_ghost_write_detector_v<Spec>()) {
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_gwd_begin<Spec>::value) {
                Spec::ghost_write_detector_begin(args, plan);
            }
        }
    }
}
template <typename Spec>
static void nlr_dispatch_ghost_write_detector_end(const neighbor_loop_args& args,
                                                  const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_ghost_write_detector_v<Spec>()) {
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_gwd_end<Spec>::value) {
                Spec::ghost_write_detector_end(args, plan);
            }
        }
    }
}
template <typename Spec>
static void nlr_dispatch_ghost_writeback_begin(const neighbor_loop_args& args,
                                               const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_ghost_writeback_v<Spec>()) {
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_gwb_begin<Spec>::value) {
                Spec::ghost_writeback_begin(args, plan);
            }
        }
    }
}
template <typename Spec>
static void nlr_dispatch_ghost_writeback_end(const neighbor_loop_args& args,
                                             const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_ghost_writeback_v<Spec>()) {
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_gwb_end<Spec>::value) {
                Spec::ghost_writeback_end(args, plan);
            }
        }
    }
}
template <typename Spec>
static void nlr_dispatch_sidechannel_writeback_begin(const neighbor_loop_args& args,
                                                     const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_sidechannel_writeback_v<Spec>()) {
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_swb_begin<Spec>::value) {
                Spec::sidechannel_writeback_begin(args, plan);
            }
        }
    }
}
template <typename Spec>
static void nlr_dispatch_sidechannel_writeback_end(const neighbor_loop_args& args,
                                                   const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_sidechannel_writeback_v<Spec>()) {
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_swb_end<Spec>::value) {
                Spec::sidechannel_writeback_end(args, plan);
            }
        }
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

    /* ---- Dispatch ----
     *
     * Selection precedence (highest first):
     *   1. GIZMO_NLR_FORCE_MODE=A → Mode A unconditionally.
     *   2. GIZMO_NLR_FORCE_MODE=B → Mode B (local if NTask==1, remote else).
     *   3. Threshold dispatch: if (sum_active>0 && sum_active<=TS &&
     *      max_active<=TM) → Mode B; else Mode A. Hierarchy of TS/TM:
     *      per-loop env > global env > Spec::modeb_threshold_{sum,max}
     *      constexpr defaults (64/64 today).
     *
     * Force-mode conflict cases (both old and new vars set, both old vars
     * set together, invalid value) are caught and endrun'd centrally inside
     * gizmo_nlr_force_mode(); see the env-config block earlier in this TU.
     *
     * Codex nuance (active-epoch caveat): Mode B host-frozen actives[] are
     * NOT bit-equivalent to Mode A's device-staged post-neighbor-list-build
     * actives. Oracle in this dispatch is Mode B tree vs Mode B brute on the
     * same frozen query — it does NOT cross-validate Mode A. Mode A vs
     * Mode B active-epoch consistency is a separate concern, deferred.
     */
    const NlrForceMode force_mode = gizmo_nlr_force_mode();
    const bool force_a   = (force_mode == NlrForceMode::A);
    const bool force_b   = (force_mode == NlrForceMode::B);
    const bool oracle_on = gizmo_nlr_oracle_enabled_global() ||
                            gizmo_nlr_oracle_enabled_for(Spec::loop_name);

    /* PHASE0 timing scaffolding (3c.4b). Cached env-gate; mid-run env
     * changes do not take effect. When on, ALL MPI_Wtime calls inside the
     * runner are gated; off-path overhead is one branch per StageTimer
     * scope, no MPI_Wtime call. PHASE0_NLR measures only RUNNER-OWNED time:
     * caller-side ghost prep / detector / SinkTempInfo scatter are NOT
     * included. */
    const bool phase0_on = gizmo_nlr_phase0_diag_enabled();
    RunnerStageTimer tim = {};
    RunnerStageTimer *tim_ptr = phase0_on ? &tim : nullptr;
    const double t_runner_start = phase0_on ? MPI_Wtime() : 0.0;

    /* Threshold dispatch. Allreduce sum + max of args.num_active.
     * Skipped when force-mode envs are set (cheap path).
     * PHASE0 num_active_global is captured here when the threshold path
     * already did the Allreduce; on force paths an extra Allreduce is done
     * ONLY when phase0_on (codex constraint #4). */
    bool select_mode_b = force_b;
    int phase0_sum_active = -1;       /* -1 = not yet computed */
    if(!force_a && !force_b) {
        int local_act = args.num_active;
        int sum_act = 0, max_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_act, &max_act, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        phase0_sum_active = sum_act;
        /* Spec::modeb_threshold_{sum,max} via SFINAE; default 64/64. */
        const int spec_default_sum = nlr_spec_threshold_sum<Spec>(64);
        const int spec_default_max = nlr_spec_threshold_max<Spec>(64);
        const int TS = gizmo_nlr_modeb_threshold_sum_for(Spec::loop_name, spec_default_sum);
        const int TM = gizmo_nlr_modeb_threshold_max_for(Spec::loop_name, spec_default_max);
        select_mode_b = (sum_act > 0) && (sum_act <= TS) && (max_act <= TM);
    } else if(phase0_on) {
        /* Force path: dispatch logic skipped the Allreduce. Do it here
         * ONLY for phase0 num_active_global. */
        int local_act = args.num_active;
        int sum_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        phase0_sum_active = sum_act;
    }

    /* Compute the execution plan from the dispatch decision. Path is the
     * single-source-of-truth; predicates derive from it. */
    NeighborLoopPlan plan;
    if(force_a) {
        plan.path = NeighborLoopPlan::Path::ModeA_GpuNgl;
    } else if(force_b || select_mode_b) {
        plan.path = (NTask > 1) ? NeighborLoopPlan::Path::ModeB_Remote
                                : NeighborLoopPlan::Path::ModeB_Local;
    } else {
        plan.path = NeighborLoopPlan::Path::ModeA_GpuNgl;
    }
    plan.num_active_global = phase0_sum_active;   /* may be -1 when phase0_on=false */

    /* Optional dispatch trace. Rank-0 only to avoid spam. */
    if(gizmo_nlr_dispatch_trace_enabled()) {
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0) {
            const char *src =
                force_a       ? "force" :
                force_b       ? "force" :
                                "threshold";
            fprintf(stderr, "[NLR DISPATCH caller=%s path=%s (%s) NTask=%d local_active=%d oracle=%d]\n",
                    Spec::loop_name, nlr_path_label(plan.path), src,
                    NTask, args.num_active, (int)oracle_on);
            fflush(stderr);
        }
    }

    /* ---- Stage radii once ---- */
    /* Computed via Spec::search_radius. Used for any path-conditional
     * prep (Mode A) AND the chosen walker. Pointer is call-lifetime only;
     * see contract in mesh/neighbor_loop_runner.h. */
    std::vector<double> radii(args.num_active);
    for(int aa = 0; aa < args.num_active; ++aa) {
        radii[aa] = Spec::search_radius(args, aa, args.active_list[aa]);
    }

    /* ---- Hard-corridor counter snapshot (always-on, every build) ---- */
    /* Mode B paths must NOT enter move_particles, ghost_exchange_impl, or
     * gpu_particles_arena_acquire, and must NOT mutate NumPart. Counters
     * live in declarations/lifecycle_counters.h, incremented at API entry
     * by the owner TUs. */
    const uint64_t s_drift0 = g_global_drift_counter;
    const uint64_t s_ghost0 = g_ghost_import_counter;
    const uint64_t s_arena0 = g_gpu_arena_acquire_counter;
    const int      s_np0    = NumPart;

    /* ---- Working copy of args; refreshed after path-conditional prep ---- */
    neighbor_loop_args effective_args = args;

    /* ---- Path-conditional prep on imported-ghost paths ---- */
    /* Mode A's substrate today satisfies its freshness + neighbor-pool
     * requirements via gizmo_request_filtered_ghost_import_fresh (full
     * global drift + ghost import). Mode B paths skip this entirely;
     * peer-local pool + lazy candidate drift is sufficient. The dt_prep_import
     * timer is 0 for Mode B paths (genuine 0 — the API isn't called). */
    if(nlr_path_uses_imported_ghosts(plan.path)) {
        StageTimer t_prep(tim_ptr ? &tim_ptr->dt_prep_import : nullptr);
        gizmo_request_filtered_ghost_import_fresh(Spec::loop_name,
                                                   Spec::search_mode,
                                                   Spec::neighbor_type_mask,
                                                   args.active_list,
                                                   args.num_active,
                                                   radii.data(),
                                                   args.ghost_safety_factor);
        /* Ghost import grew NumPart and may have realloc'd P/CellP. Refresh
         * the runner's data view; only paths that imported ghosts read this
         * extended view (Mode B paths use the original args via copy). */
        effective_args.num_total = NumPart;
        effective_args.P         = P;
        effective_args.CellP     = (All.TotN_gas > 0) ? CellP : nullptr;
    }

    /* ---- Spec lifecycle hooks (begin) ---- */
    /* Mode A ordering invariant (codex constraint, Stage 2c audit):
     *   request_filtered_ghost_import_fresh  (above)
     *   ghost_write_detector_begin           (this hook)
     *   ghost_writeback_begin                (this hook; SinkEnv1Spec no-op)
     *   sidechannel_writeback_begin          (this hook; SINGLE_STAR_SINK_DYNAMICS-gated)
     *   <run_mode_a kernel>
     *   sidechannel_writeback_end            (reverse order, below)
     *   ghost_writeback_end
     *   ghost_write_detector_end
     *   ghost_exchange_cleanup               (below)
     *
     * Each dispatch helper checks both the Spec's `uses_*` trait AND the
     * path-imports-ghosts predicate; on Mode B paths the predicate is
     * false and all six hook calls compile to no-ops. */
    nlr_dispatch_ghost_write_detector_begin<Spec>(effective_args, plan);
    nlr_dispatch_ghost_writeback_begin<Spec>(effective_args, plan);
    nlr_dispatch_sidechannel_writeback_begin<Spec>(effective_args, plan);

    /* ---- Path dispatch ---- */
    switch(plan.path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl:
            /* Oracle on Mode A is a no-op (oracle compares Mode B vs Brute;
             * no Brute-vs-Mode-A oracle in scope). */
            run_mode_a<Spec>(effective_args, radii.data(), tim_ptr);
            break;
        case NeighborLoopPlan::Path::ModeB_Local:
            if(oracle_on) run_mode_b_local_with_oracle<Spec>(effective_args, radii.data(), tim_ptr);
            else          run_mode_b_local<Spec>(effective_args, radii.data(), tim_ptr);
            break;
        case NeighborLoopPlan::Path::ModeB_Remote:
            if(oracle_on) run_mode_b_remote_with_oracle<Spec>(effective_args, radii.data(), tim_ptr);
            else          run_mode_b_remote<Spec>(effective_args, radii.data(), tim_ptr);
            break;
    }

    /* ---- Spec lifecycle hooks (end, reverse order) ---- */
    nlr_dispatch_sidechannel_writeback_end<Spec>(effective_args, plan);
    nlr_dispatch_ghost_writeback_end<Spec>(effective_args, plan);
    nlr_dispatch_ghost_write_detector_end<Spec>(effective_args, plan);

    /* ---- Imported-ghost cleanup ---- */
    if(nlr_path_uses_imported_ghosts(plan.path) && NTask > 1) {
        ghost_exchange_cleanup();
    }

    /* ---- Hard-corridor enforcement (Mode B paths) ---- */
    /* HARD ABORT on any counter advance or NumPart change across the path
     * body. Always-on; cheap (4 uint64 compares + one int compare). */
    if(plan.path == NeighborLoopPlan::Path::ModeB_Local ||
       plan.path == NeighborLoopPlan::Path::ModeB_Remote) {
        const bool drift_violation = (g_global_drift_counter      != s_drift0);
        const bool ghost_violation = (g_ghost_import_counter      != s_ghost0);
        const bool arena_violation = (g_gpu_arena_acquire_counter != s_arena0);
        const bool np_violation    = (NumPart != s_np0);
        if(drift_violation || ghost_violation || arena_violation || np_violation) {
            int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            fprintf(stderr,
                    "[NLR CORRIDOR ABORT rank=%d caller=%s path=%s] Mode B path "
                    "violated tiny-N corridor invariant during run_neighbor_loop. "
                    "Counter deltas: drift=%llu ghost=%llu arena=%llu NumPart_pre=%d NumPart_post=%d\n",
                    rank, Spec::loop_name, nlr_path_label(plan.path),
                    (unsigned long long)(g_global_drift_counter - s_drift0),
                    (unsigned long long)(g_ghost_import_counter - s_ghost0),
                    (unsigned long long)(g_gpu_arena_acquire_counter - s_arena0),
                    s_np0, NumPart);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 81036);
        }
    }

    /* ---- PHASE0_NLR emit ---- */
    /* Stable prefix `PHASE0_NLR`. `caller=` is a field, not part of the
     * token, so future Specs keep the parser regex stable. The new
     * dt_prep_import field measures the runner-internal prep wall (Mode A
     * only; 0 on Mode B paths). Other fields per the path-specific
     * documentation in neighbor_loop_runner.h. */
    if(phase0_on) {
        tim.dt_total = MPI_Wtime() - t_runner_start;
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        static long long s_call_id = 0;
        ++s_call_id;
        std::printf("PHASE0_NLR rank=%d caller=%s path=%s call_id=%lld "
                    "num_active_local=%d num_active_global=%d "
                    "dt_prep_import=%.6g "
                    "dt_collect=%.6g dt_drift=%.6g dt_walk_self=%.6g "
                    "dt_walk_peer=%.6g dt_exchange_q=%.6g dt_exchange_r=%.6g "
                    "dt_reduce=%.6g dt_writeback=%.6g dt_total=%.6g\n",
                    rank, Spec::loop_name, nlr_path_label(plan.path), s_call_id,
                    args.num_active, phase0_sum_active,
                    tim.dt_prep_import,
                    tim.dt_collect, tim.dt_drift, tim.dt_walk_self,
                    tim.dt_walk_peer, tim.dt_exchange_q, tim.dt_exchange_r,
                    tim.dt_reduce, tim.dt_writeback, tim.dt_total);
        std::fflush(stdout);
    }
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
