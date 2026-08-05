/* mesh/neighbor_loop_runner.cc — generic NeighborLoopSpec runner.
 *
 * Mode A (GPU NGL pipeline): stages caller-supplied radii and per-call
 * CallScalars host-side pre-arena, runs gpu_particles_arena_acquire +
 * gpu_ngb_list_build (or stages a caller-injected external CSR), then a
 * Kokkos parallel_for calling Spec::load_active to fill ActiveData[] in
 * UVM (same device epoch as the pair walk), then the parametric
 * pair-kernel parallel_for calling Spec::load_neighbor + Spec::pair_kernel.
 * Launches go through gizmo_gpu_kernel_launch (parallel_for + fence +
 * check_last_error).
 *
 * Mode B (request-driven walker, local + cross-rank peer-to-peer) and the
 * host-side invocation with the
 * lazy-drift boundary structurally encoded as collect_candidates_pre_drift
 * -> lazy_drift_candidates -> evaluate_pairs_post_drift; it uses
 * the SAME drift epoch as Mode B.
 *
 * The Spec contract (hard-required members, hooks, invariants) is
 * documented in mesh/neighbor_loop_runner.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
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

#include "../declarations/gpu_all_mirror.h"          /* per-TU AllDeviceMirror + device-pass `All` redirect; include before allvars.h */
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
#include "../gravity/gpu_gravity_tree.h" /* gpu_gravity_soa_drift_certified (drift-cert diagnostic) */
#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <algorithm>   /* nth_element / max_element (coverage percentiles) */
#include <unordered_map>
#include <cmath>

#include "mode_b_p2p_transport.h"  /* ModeBBoundedExchange (query/reply transport) */

/* Spec instantiations. Each #include declares one Spec type whose explicit
 * template instantiation appears at the bottom of this file. */
#include "../sinks/sink_env1_loop.h"
#include "../sinks/sink_feed_loop.h"
#include "../sinks/sink_swk_loop.h"
#if defined(SINK_PARTICLES) && defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
#include "../sinks/sink_env2_loop.h"
#endif

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
#include "../gravity/ags_density_loop.h"
#include "../gravity/ags_force_loop.h"
#endif

#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
#include "../sidm/cbe_integrator_gradients.h"
#endif

#include "../hydro/density_loop.h"

#ifdef GALSF_FB_MECHANICAL
#include "../galaxy_sf/mechfb_loop.h"
#endif

#ifdef GALSF_FB_THERMAL
#include "../galaxy_sf/thermal_fb_loop.h"
#endif

#ifdef HYDRO_VOLUME_CORRECTIONS
#include "../hydro/cellcorrections_loop.h"
#endif

/* GradientsSpec always built (gradients has no master #ifdef gate — every
 * hydro build runs hydro_gradient_calc). */
#include "../hydro/gradients_loop.h"

/* HydroForceSpec always built (every hydro build runs hydro_force). */
#include "../hydro/hydro_force_loop.h"

#ifdef GALSF_FB_FIRE_RT_LOCALRP
#include "../galaxy_sf/radfb_rp_loop.h"
#endif
#ifdef DM_DISPERSION_LOOP_ACTIVE
#include "../galaxy_sf/dm_dispersion_loop.h"
#endif
#ifdef TURB_DIFF_DYNAMIC
#include "../turb/difffilter_loop.h"
#endif

#ifdef DM_FUZZY
#include "../sidm/dm_fuzzy_loop.h"
#endif

#ifdef DO_FLUID_ALTSPECIES_DRAG_CALCULATION
#include "../solids/grain_physics_loop.h"
#endif

#ifdef RT_SOURCE_INJECTION
#include "../radiation/rt_source_injection_loop.h"
#endif

/* ============================================================================
 * Shared NLR utility helpers (used by env-config and threshold blocks below).
 * File-scope static; TU-local linkage. Defined here so the threshold helpers
 * (which are file-scope `extern` for runner.h API) can call them.
 * ========================================================================== */

/* One-shot rank-0 warning helper.
 *
 * Cached set keyed by string CONTENT (strcmp), not pointer identity, so
 * dynamically-constructed keys dedupe correctly per distinct key rather than
 * per category. Costs a 64x192
 * BSS array (~12 KB) and an O(N) lookup per call; both fine for a warning
 * surface.
 *
 * Cap is generous (64 distinct keys); if hit, subsequent warnings are
 * silently dropped — they are diagnostics, not correctness gates. */
static void nlr_warn_once_rank0(const char *key, const char *fmt, ...)
{
    if(ThisTask != 0) return;
    static char seen[64][192];
    static int  seen_n = 0;
    for(int i = 0; i < seen_n; i++) {
        if(strcmp(seen[i], key) == 0) return;
    }
    if(seen_n < 64) {
        size_t n = strlen(key);
        if(n >= sizeof(seen[0])) n = sizeof(seen[0]) - 1;
        memcpy(seen[seen_n], key, n);
        seen[seen_n][n] = '\0';
        seen_n++;
    }
    fprintf(stderr, "[NLR env] ");
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

/* ============================================================================
 * Caller-side helpers (declared in runner.h).
 *
 * Out-of-line so changes to global plumbing (NumPart, P, CellP fetch site,
 * ghost-safety-factor source, etc.) are caller-invisible.
 * ========================================================================== */

/* Thin wrapper over the canonical host accessor; retained for callers
 * that already route through this name. With the per-TU AllDeviceMirror
 * scheme's device-pass-gated `#define All AllDeviceMirror`, bare `All.*`
 * in host code reads the host extern unconditionally, so this
 * indirection is no longer load-bearing — kept for stability of
 * existing call sites. */
const struct global_data_all_processes * nlr_host_all_ptr(void)
{
    return gizmo_host_all_ptr();
}

NlrCommonScalars nlr_common_scalars_from_all(void)
{
    const struct global_data_all_processes *h = nlr_host_all_ptr();
    NlrCommonScalars s;
    s.cf_atime                = h->cf_atime;
    s.cf_a2inv                = h->cf_a2inv;
    s.cf_a3inv                = h->cf_a3inv;
    s.cf_hubble_a             = h->cf_hubble_a;
    s.newton_G                = h->G;
    s.hubble                  = h->HubbleParam;
    s.comoving_integration_on = h->ComovingIntegrationOn;
    return s;
}

neighbor_loop_args nlr_default_args(void)
{
    neighbor_loop_args args;
    args.P                   = P;
    args.CellP               = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;
    args.num_total           = NumPart;
    args.active_list         = nullptr;       /* caller fills */
    args.num_active          = 0;             /* caller fills */
    args.aux                 = nullptr;       /* caller fills */
    args.ghost_safety_factor = gizmo_ghost_safety_factor();
    args.neighbor_type_mask_override = 0;      /* 0 => use Spec::neighbor_type_mask */
    args.external_csr        = nullptr;        /* nullptr => runner builds its own CSR */
    args.dispatch_override   = NlrForceMode::None; /* None => adaptive threshold */
    return args;
}

void nlr_free_active_list(int *active_list)
{
    if(active_list) myfree(active_list);
}

/* ============================================================================
 * Mode-A/B dispatch thresholds
 *
 * Production dispatch policy is the constexpr Spec::modeb_threshold_sum and
 * Spec::modeb_threshold_max in each NeighborLoopSpec — those are code-level
 * dispatch policy constants for the loop, decided alongside the physics.
 *
 * Resolution precedence (first wins):
 *   1. parameterfile NeighborLoopModeBThreshold{Sum,Max} (-1 = unset;
 *      any other value overrides, <= 0 disables Mode B)
 *   2. Spec::modeb_threshold_{sum,max} constexpr (code default)
 *
 * The threshold is settable ONLY from the parameterfile, so the value a run
 * used is recorded in its parameter log.
 * ========================================================================== */

int gizmo_nlr_modeb_threshold_sum_for(const char *loop_name, int spec_default)
{
    (void)loop_name;
    /* Parameterfile override (the supported interface): -1 = unset -> use the
       Spec default; any other value overrides it (<= 0 disables Mode B). */
    { int p = nlr_host_all_ptr()->NeighborLoopModeBThresholdSum; if(p != -1) return p; }
    return spec_default;
}
int gizmo_nlr_modeb_threshold_max_for(const char *loop_name, int spec_default)
{
    (void)loop_name;
    /* Parameterfile override (the supported interface): -1 = unset -> use the
       Spec default; any other value overrides it (<= 0 disables Mode B). */
    { int p = nlr_host_all_ptr()->NeighborLoopModeBThresholdMax; if(p != -1) return p; }
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

/* nlr_warn_once_rank0 is defined at file scope above (near the includes). */

/* Initialize diag level. */
static int nlr_init_diag_level(void)
{
    const char *raw = getenv("GIZMO_NLR_DIAG");
    int level = 0;
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
        level = (int)v;
    }

    if(level == 3) {
        nlr_warn_once_rank0("level_3_reserved",
            "GIZMO_NLR_DIAG=3: level 3 currently has no extra diagnostics; "
            "reserved for future scalar-only extensions. Behaving as level 2.");
    }

    return level;
}



} /* anonymous namespace */

int gizmo_nlr_diag_level(void)
{
    static int cached = -1;
    if(cached < 0) cached = nlr_init_diag_level();
    return cached;
}

/* Adapters — preserve existing call-site names. */
bool gizmo_nlr_phase0_diag_enabled(void)    { return gizmo_nlr_diag_level() >= 1; }
bool gizmo_nlr_dispatch_trace_enabled(void) { return gizmo_nlr_diag_level() >= 2; }

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

/* Caller-owned ghost pool: a caller supplying external_csr owns the live
 * particle+ghost pool the CSR indexes (see the GHOST-POOL OWNERSHIP contract
 * in neighbor_loop_runner.h). The runner must not import (slot renumbering
 * would silently invalidate the CSR) and must not cleanup (the pool outlives
 * this call). Single ownership signal by design — no separate flag that
 * could be set inconsistently with external_csr. */
static inline bool nlr_caller_owns_ghost_pool(const neighbor_loop_args &args)
{
    return args.external_csr != nullptr;
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
 * StageTimer — internal RAII helper for PHASE0 timing
 *
 * `target == nullptr` means phase0 is off: ctor and dtor do nothing except
 * a single predictable nullptr branch. NO MPI_Wtime call when off — "MPI_Wtime is not zero overhead" is
 * addressed at the call site, not just at the gating env var.
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
        case SidxCacheKind::GasOnly:
            return gpu_step_sidx_ptr();
        case SidxCacheKind::None:
            return nullptr;
    }
    /* Unreachable today — kept exhaustive for compiler warnings on enum
     * additions. Other kinds land alongside their first caller. */
    fprintf(stderr, "neighbor_loop_runner: SidxCacheKind=%d not implemented "
            "for loop '%s'\n",
            (int)k, loop_name ? loop_name : "?");
    fflush(stderr);
    endrun(81030);
    return nullptr;
}

/* ============================================================================
 * Mode B local self-rank helpers
 *
 * These three helpers STRUCTURALLY ENCODE the lazy-drift invariant from the
 * neighbor-loop binding contract:
 *
 *     collect_candidates_pre_drift<Spec>   — search backend (tree or brute)
 *                                            runs against possibly-stale P[j]
 *     lazy_drift_candidates<Spec>          — drift_particle on every j to
 *                                            All.Ti_Current (idempotent;
 *                                            duplicate j's between Mode B and
 *                                            are dedupe-free
 *                                            via drift_particle's
 *                                            time1==time0 early-return)
 *     evaluate_pairs_post_drift<Spec>      — calls Spec::pair_kernel via the
 *                                            same KOKKOS_INLINE_FUNCTION
 *                                            Spec::load_active /
 *                                            Spec::load_neighbor used by
 *                                            run_mode_a (host invocation here)
 *
 * Candidate sets are collected before any drift. Annotation
 * on each helper makes the ordering enforcement structural.
 *
 * Self-rank only here: candidates are local real P[] indices in
 * [0, num_local). Cross-rank peer-to-peer comes below.
 *
 * Walker buffer sized to num_local (worst-case SYMMETRIC, no h-bound
 * pre-pruning).
 * ========================================================================== */

/* Mode-B discovery-walk OpenMP threading. The self/receiver walks write into
 * disjoint per-item output slots, so they parallelize over the item index with
 * no shared writes. Threading engages only above a structural work threshold
 * (never a caller name); below it the serial code runs verbatim (tiny-N steps
 * pay nothing). Chunk sizes are separate constants because received-query work
 * variance differs from self-active variance. */
static constexpr int MODEB_OMP_MIN_PER_THREAD = 4;    /* work >= max(64, 4*nthreads) to thread */
static constexpr int MODEB_OMP_CHUNK_ACTIVE   = 16;   /* schedule(dynamic) chunk for self-active loops */
static constexpr int MODEB_OMP_CHUNK_RECV     = 16;   /* schedule(dynamic) chunk for received-query loop */

static inline int nlr_modeb_omp_nthreads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Structural gate: thread iff more than one thread AND enough work to amortize.
 * Reads NOTHING tree- or drift-related; safe to call on any step. */
static inline bool nlr_modeb_use_omp(long long n_items, int nthreads)
{
    if(nthreads <= 1) return false;
    const long long floor_work = (long long)MODEB_OMP_MIN_PER_THREAD * nthreads;
    const long long thresh = (floor_work > 64) ? floor_work : 64;
    return n_items >= thresh;
}

/* Eval-threading policy for evaluate_pairs_post_drift. The production tree eval
 * may thread (BitwiseReadonly or EpsilonAtomic specs, i.e. any non-SerialOnly tier).
 * Passed explicitly at every call site (no default) so production vs
 * reference eval is greppable and can never be silently mis-gated. */
enum class EvalOMPPolicy { AllowProduction, ForceSerialReference };

/* GX_MODEB_EXPORT omp_eval= field: the eval-threading decision, mirroring the
 * evaluate_pairs_post_drift gate exactly (same nlr_modeb_use_omp threshold) so
 * the diagnostic can never disagree with the code path taken. Reports the
 * production self-eval decision (EvalOMPPolicy::AllowProduction at the emit). */
static inline void nlr_modeb_eval_decision_label(char *buf, size_t n,
                                                 ModeBEvalOMP tier, bool is_explicit,
                                                 EvalOMPPolicy policy,
                                                 int nthreads, long long work)
{
    if(!is_explicit)                                  { std::snprintf(buf, n, "serial(missing_trait)"); return; }
    if(policy == EvalOMPPolicy::ForceSerialReference) { std::snprintf(buf, n, "serial(reference)"); return; }
    if(tier == ModeBEvalOMP::SerialOnly)              { std::snprintf(buf, n, "serial(trait_serialonly)"); return; }
    /* tier is BitwiseReadonly or EpsilonAtomic (both thread the production eval) */
    if(nthreads <= 1)                                 { std::snprintf(buf, n, "serial(1thread)"); return; }
    if(nlr_modeb_use_omp(work, nthreads)) {
        if(tier == ModeBEvalOMP::BitwiseReadonly)       std::snprintf(buf, n, "bitwise_readonly(%d)", nthreads);
        else                                            std::snprintf(buf, n, "epsilon_atomic(%d)", nthreads);
    } else                                              std::snprintf(buf, n, "serial(below_threshold)");
}

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
                                          unsigned int neighbor_type_mask,
                                          DispatchPath backend,
                                          std::vector<std::vector<int>>& per_active_cands,
                                          ModeBDriftCounters* drift_ctr_out = nullptr,
                                          int* threads_used_out = nullptr)
{
    if(threads_used_out) *threads_used_out = 0;   /* 0 until threading is decided */
    /* neighbor_type_mask is an explicit caller parameter (mask-threading
     * refactor). Non-iter callers pass Spec::neighbor_type_mask (unchanged
     * behavior); iter dispatch passes sg.j_type_bitmask for per-subgroup walks. */
    const int N = args.num_active;
    const int num_local = ghost_get_num_local();
    /* Per-call ownership: each inner vector owns its capacity for the
     * duration of this call. .assign(N, {}) clobbers any stale residue
     * from a previous call. Walker-append contract: appends via
     * push_back; geometric growth handles any-size match set without
     * imposing the previous full-pool .assign(num_local, 0) cost
     * (~24 MB × N_active on fire_m11i). */
    per_active_cands.assign(N, std::vector<int>{});
    if(num_local <= 0) return;
    if(backend != DispatchPath::ModeB_HostWalker) {
        fprintf(stderr, "neighbor_loop_runner: collect_candidates_pre_drift "
                "called with non-Mode-B backend (%d) for loop '%s'\n",
                (int)backend, Spec::loop_name);
        fflush(stderr);
        endrun(81033);
        return;
    }
    const double jscale = nlr_spec_symmetric_j_radius_scale<Spec>();
    /* Thread the tree walk above the work threshold; below it stays serial. */
    const int nthreads = nlr_modeb_omp_nthreads();
    const bool use_omp = (backend == DispatchPath::ModeB_HostWalker) &&
                         nlr_modeb_use_omp(N, nthreads);
    if(threads_used_out) *threads_used_out = use_omp ? nthreads : 0;
    if(use_omp) {
        /* Each thread mutates only its own per_active_cands[aa] (outer vector
         * pre-sized; no shared push) and its own drift counter (diagnostic
         * only — allocated iff a counter sink was requested). */
        std::vector<ModeBDriftCounters> tctr;
        if(drift_ctr_out) tctr.resize(nthreads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, MODEB_OMP_CHUNK_ACTIVE)
#endif
        for(int aa = 0; aa < N; aa++) {
            const int i = args.active_list[aa];
            const double h_q = radii[aa];
            if(h_q <= 0) continue;
            std::vector<int>& cands = per_active_cands[aa];
            cands.clear();
            if(cands.capacity() == 0) cands.reserve(64);
            double pos_arr[3] = {(double)args.P[i].Pos[0],
                                  (double)args.P[i].Pos[1],
                                  (double)args.P[i].Pos[2]};
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            ModeBDriftCounters* ctr = drift_ctr_out ? &tctr[tid] : nullptr;
            mode_b_local_neighbor_walk(pos_arr, h_q, neighbor_type_mask,
                                       Spec::search_mode, Spec::radius_policy,
                                       cands, jscale, ctr);
        }
        if(drift_ctr_out) for(const auto& c : tctr) drift_ctr_out->add(c);
        return;
    }
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
        mode_b_local_neighbor_walk(pos_arr, h_q, neighbor_type_mask,
                                    Spec::search_mode, Spec::radius_policy,
                                    cands, jscale);
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
    const std::vector<int>& peer_nodelist_flat,   /* K*NODELISTLENGTH; exported start-nodes per query */
    const std::vector<int>& peer_nnodes,          /* K; valid entries per query's NodeList */
    unsigned int neighbor_type_mask,
    DispatchPath backend,
    std::vector<std::vector<int>>& per_query_cands,
    ModeBDriftCounters* drift_ctr_out = nullptr,
    int* threads_used_out = nullptr)
{
    if(threads_used_out) *threads_used_out = 0;   /* 0 until threading is decided */
    /* neighbor_type_mask is an explicit caller parameter (mask-threading
     * refactor). Non-iter callers pass Spec::neighbor_type_mask; iter
     * dispatch passes sg.j_type_bitmask.
     *
     * ModeB_HostWalker resumes the walk from the exported NodeList
     * start-nodes (legacy mode==1). */
    const int K = (int)peer_actives.size();
    const int num_local = ghost_get_num_local();
    per_query_cands.assign(K, std::vector<int>{});
    if(num_local <= 0) return;
    if(backend != DispatchPath::ModeB_HostWalker) {
        fprintf(stderr, "neighbor_loop_runner: collect_candidates_for_remote_queries"
                " bad backend %d for loop '%s'\n", (int)backend, Spec::loop_name);
        fflush(stderr);
        endrun(81033);
        return;
    }
    const double jscale = nlr_spec_symmetric_j_radius_scale<Spec>();
    /* Thread the received-query tree walk above the work threshold; brute
     * below-threshold stays serial (byte-identical). */
    const int nthreads = nlr_modeb_omp_nthreads();
    const bool use_omp = (backend == DispatchPath::ModeB_HostWalker) &&
                         nlr_modeb_use_omp(K, nthreads);
    if(threads_used_out) *threads_used_out = use_omp ? nthreads : 0;
    if(use_omp) {
        std::vector<ModeBDriftCounters> tctr;   /* diagnostic only */
        if(drift_ctr_out) tctr.resize(nthreads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, MODEB_OMP_CHUNK_RECV)
#endif
        for(int k = 0; k < K; k++) {
            const auto& active = peer_actives[k];
            const double h_q = (double)active.h_search;
            if(h_q <= 0) continue;
            std::vector<int>& cands = per_query_cands[k];
            cands.clear();
            if(cands.capacity() == 0) cands.reserve(64);
            double pos_arr[3] = {(double)active.pos[0], (double)active.pos[1], (double)active.pos[2]};
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            ModeBDriftCounters* ctr = drift_ctr_out ? &tctr[tid] : nullptr;
            if(peer_nnodes[k] > 0) {
                mode_b_walk_from_start_nodes(pos_arr, h_q, neighbor_type_mask,
                                             Spec::search_mode, Spec::radius_policy,
                                             &peer_nodelist_flat[(size_t)k * NODELISTLENGTH],
                                             peer_nnodes[k], cands, jscale, ctr);
            } else {
                mode_b_local_neighbor_walk(pos_arr, h_q, neighbor_type_mask,
                                            Spec::search_mode, Spec::radius_policy,
                                            cands, jscale, ctr);
            }
        }
        if(drift_ctr_out) for(const auto& c : tctr) drift_ctr_out->add(c);
        return;
    }
    for(int k = 0; k < K; k++) {
        const auto& active = peer_actives[k];
        const double h_q = (double)active.h_search;
        if(h_q <= 0) continue;
        std::vector<int>& cands = per_query_cands[k];
        cands.clear();
        if(cands.capacity() == 0) cands.reserve(64);
        double pos_arr[3] = {(double)active.pos[0], (double)active.pos[1], (double)active.pos[2]};
        if(peer_nnodes[k] > 0) {
            /* Targeted query: bounded resume from the exported start-nodes. */
            mode_b_walk_from_start_nodes(pos_arr, h_q, neighbor_type_mask,
                                         Spec::search_mode, Spec::radius_policy,
                                         &peer_nodelist_flat[(size_t)k * NODELISTLENGTH],
                                         peer_nnodes[k], cands, jscale);
        } else {
            /* Broadcast query (n_nodes==0): the sender took the broadcast
             * path (uncovered radius policy) — full local
             * walk from root (the prior broadcast behavior). */
            mode_b_local_neighbor_walk(pos_arr, h_q, neighbor_type_mask,
                                        Spec::search_mode, Spec::radius_policy,
                                        cands, jscale);
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
template <typename Spec, typename DeviceCtx>
static void evaluate_pairs_post_drift(const DeviceCtx& ctx,
                                       const typename Spec::ActiveData *actives,
                                       int N,
                                       const std::vector<std::vector<int>>& per_active_cands,
                                       typename Spec::AccumData *accums,
                                       EvalOMPPolicy eval_policy)
{
    using NeighborData = typename Spec::NeighborData;
    using ScatterData  = typename Spec::ScatterData;

    /* Per-active evaluation. Writes ONLY accums[aa] plus call-local scratch, so
     * distinct aa are independent — the invariant the BitwiseReadonly threading
     * below relies on. */
    auto eval_one = [&](int aa) {
        Spec::zero_accum(accums[aa]);
        ScatterData s{};                              /* NoScatter for ActiveReduceOnly */
        const auto& cands = per_active_cands[aa];
        if constexpr (nlr_spec_has_bind_active_to_eval_context_v<Spec>) {
            /* Spec needs per-eval-pass rebinding of rank-local + eval-pass-local
             * fields embedded in ActiveData (P_base / CellP_base / per-rank
             * gas-delta ptrs / per-rank index bounds). Take a local
             * mutable copy, refresh from the eval ctx, then walk. See
             * nlr_spec_has_bind_active_to_eval_context_v in
             * mesh/neighbor_loop_runner.h for the contract. */
            typename Spec::ActiveData a = actives[aa];
            Spec::bind_active_to_eval_context(ctx, a);
            for(size_t kk = 0; kk < cands.size(); kk++) {
                int j = cands[kk];
                IdentitySidecar id{};                 /* NoIdentity */
                NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                Spec::pair_kernel(a, nb, accums[aa], s);
            }
        } else {
            /* No bind hook: walk by const-ref, no copy. */
            const auto& a = actives[aa];
            for(size_t kk = 0; kk < cands.size(); kk++) {
                int j = cands[kk];
                IdentitySidecar id{};                 /* NoIdentity */
                NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                Spec::pair_kernel(a, nb, accums[aa], s);
            }
        }
    };

    /* Two tiers thread the production eval over aa; each aa writes only its own
     * accums[aa] plus call-local scratch (fixed per-active neighbor order):
     *   BitwiseReadonly — no j-side writes at all -> bit-identical regardless of
     *     thread count/schedule.
     *   EpsilonAtomic   — j-side scatter goes through Kokkos::atomic_* (the same
     *     atomics Mode A's kernel already applies over concurrent actives), so
     *     threading only re-orders those atomic updates -> ulp-class
     *     nondeterminism, NOT a new race. The per-active AccumData stays
     *     order-independent (no j-write is read back into the kernel).
     * Only the production path threads; the reference eval
     * (ForceSerialReference) and SerialOnly specs run the serial loop verbatim.
     * Below the work threshold no OpenMP region is entered — the serial path is
     * byte-identical to the unthreaded code. */
    if constexpr (nlr_spec_modeb_eval_omp<Spec>() == ModeBEvalOMP::BitwiseReadonly) {
        static_assert(!Spec::uses_ghost_writeback,
                      "BitwiseReadonly eval tier requires uses_ghost_writeback==false "
                      "(no j-side ghost scatter)");
    }
    const int  eval_nthreads = nlr_modeb_omp_nthreads();
    const bool eval_use_omp  =
        (eval_policy == EvalOMPPolicy::AllowProduction) &&
        (nlr_spec_modeb_eval_omp<Spec>() != ModeBEvalOMP::SerialOnly) &&
        nlr_modeb_use_omp((long long)N, eval_nthreads);

    if(eval_use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, MODEB_OMP_CHUNK_ACTIVE)
#endif
        for(int aa = 0; aa < N; aa++) eval_one(aa);
    } else {
        for(int aa = 0; aa < N; aa++) eval_one(aa);
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
 * escaping that overhead for tiny-N.
 * ========================================================================== */

/* Helper: build the host-frozen ActiveData[] for self-rank actives. Called
 * BEFORE any drift so the active snapshot matches the legacy pack_query
 * epoch. (Note: this is NOT the same epoch as Mode A's device-staged
 * load_active post-NGL-build; documented at run_neighbor_loop dispatch site.)
 */
template <typename Spec, typename DeviceCtx>
static void build_self_actives_host_pre_drift(
    const neighbor_loop_args& args,
    const DeviceCtx& ctx,
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
static void run_mode_b_local(const neighbor_loop_args& args, const double *radii,
                             RunnerStageTimer *tim = nullptr)
{
    using ActiveData = typename Spec::ActiveData;
    using AccumData  = typename Spec::AccumData;
    using DeviceCtx  = typename Spec::DeviceContext;

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
    static_assert(std::is_base_of<NeighborLoopDeviceContextBase, DeviceCtx>::value,
                  "Spec::DeviceContext must publicly derive from NeighborLoopDeviceContextBase");
    static_assert(std::is_trivially_copyable<DeviceCtx>::value,
                  "Spec::DeviceContext must be trivially copyable; the runner captures it by value into Kokkos device lambdas");
    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(args, ctx);
    }
    NlrDeviceContextCleanupGuard<Spec> _nlr_dctx_cleanup_guard(args, ctx);

    /* Freeze active snapshots host-side BEFORE drift. */
    std::vector<ActiveData> actives(N);
    build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs, actives.data());

    /* Helper layout: collect → drift → evaluate. */
    std::vector<std::vector<int>> cand_modeB;
    {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        collect_candidates_pre_drift<Spec>(args, radii,
                                            nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                                            DispatchPath::ModeB_HostWalker, cand_modeB);
    }
    {
        StageTimer t(tim ? &tim->dt_drift : nullptr);
        lazy_drift_candidates<Spec>(cand_modeB);
    }

    std::vector<AccumData> accums(N);
    {
        StageTimer t(tim ? &tim->dt_walk_self : nullptr);
        evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_modeB, accums.data(), EvalOMPPolicy::AllowProduction);
    }

    /* Host writeback — same code path as Mode A's writeback. */
    {
        StageTimer t(tim ? &tim->dt_writeback : nullptr);
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], accums[aa]);
        }
    }
}






/* ============================================================================
 * Mode B remote (multi-rank) helpers
 *
 * STRONG INVARIANT:
 *   collect-all → drift-union → evaluate-all on each rank.
 *
 * On each rank, the local pool is walked TWICE per call: once for this
 * rank's own self-pair queries (its own actives), and once for queries
 * received from peers. Both candidate sets must be collected pre-drift,
 * then the drift covers their UNION (idempotent so duplicates are free),
 * then evaluation runs post-drift.
 *
 * Sequence:
 *   stage 1 (active rank) build self radii, cs, frozen actives[]
 *   stage 2 (active rank) build envelopes, ALL peers in broadcast pattern
 *   stage 3 (this rank)   collect self candidates pre-drift
 *   stage 4 (collective)  exchange queries (peer-to-peer) -> recv envelopes
 *   stage 5 (this rank)   flatten envelopes to peer_actives[] + provenance[]
 *   stage 6 (this rank)   collect peer candidates pre-drift
 *   stage 7 (this rank)   drift the UNION of the self and peer candidate sets
 *   stage 8 (this rank)   evaluate self candidates
 *   stage 9 (this rank)   evaluate peer candidates
 *   stage 10 (collective) exchange replies (tree result is what ships)
 *   stage 11 (active rank) merge replies via Spec::merge_accum, ascending
 *                                  rank for FP-reproducible order
 *   stage 12 (active rank) writeback
 *
 * Note: the host-frozen actives[] match Mode B /
 * legacy pack_query epoch, NOT Mode A's device-staged post-NGL-build
 * epoch — the two active epochs are not bit-equivalent.
 * ========================================================================== */

/* ============================================================================
 * mode_b_remote_evaluate_into_buffer<Spec> — extracted helper.
 *
 * Mechanical refactor of the existing run_mode_b_remote_impl body, extracting
 * stages 1-12 (queries / collect / drift / evaluate / merge / replies / merge)
 * + the env-gated active_dumps diagnostic into a reusable helper. The caller
 * provides the output AccumData buffer; the helper does NOT call
 * Spec::apply_active_writeback. Final-only writeback decision stays with the
 * caller ("let the caller decide whether to call
 * apply_active_writeback").
 *
 * Two callers:
 *   - run_mode_b_remote_impl (non-iterative wrapper): allocates per-call
 *     AccumData buffer, calls helper (which emits dumps internally on its
 *     locally-built actives[] view), then runs apply_active_writeback.
 *   - nlr_iter_dispatch_subgroup_mode_b_remote (iterative dispatch): calls
 *     helper with driver-owned compacted AccumData buffer; runner does
 *     final-only apply_active_writeback after the iter loop. (Per-iter
 *     active_dumps emits are harmless: env-gated, only fire under spike-test
 *     flags.)
 *
 * Epoch order is preserved EXACTLY: build self/peer
 * queries -> collect all candidate sets pre-drift -> exchange -> drift union
 * -> evaluate -> exchange replies -> merge replies
 * in deterministic peer order. No re-derivation; line-for-line move from
 * the old impl.
 * ========================================================================== */
template <typename Spec>
static void mode_b_remote_evaluate_into_buffer(
    const neighbor_loop_args& args,
    const double *radii,
    const typename Spec::CallScalars& cs,
    const typename Spec::DeviceContext& ctx,         /* caller-owned */
    unsigned int neighbor_type_mask,                  /* explicit caller param */
    typename Spec::AccumData *accums_out,             /* size = args.num_active; caller-owned */
    RunnerStageTimer *tim = nullptr)
{
    using ActiveData    = typename Spec::ActiveData;
    using AccumData     = typename Spec::AccumData;
    using DeviceCtx     = typename Spec::DeviceContext;
    using Envelope      = NlrQueryEnvelope<ActiveData>;
    using ReplyEnvelope = NlrReplyEnvelope<AccumData>;

    /* Mode predicates (single source of truth). Constexpr so the dead
     * branches are elided per instantiation. */
    static_assert(std::is_trivially_copyable<Envelope>::value,
        "NlrQueryEnvelope must be trivially-copyable for byte-level MPI transfer");
    static_assert(std::is_trivially_copyable<ReplyEnvelope>::value,
        "NlrReplyEnvelope must be trivially-copyable for byte-level MPI transfer");

    const int N    = args.num_active;     /* may be 0 on this rank; collective entry */
    const int nt   = NTask;
    const int rank = ThisTask;
    /* CallScalars and DeviceContext passed in by caller:
     *   - Iterative: driver-owned via NlrIterDriver, populated once at iter-0 entry.
     *   - Non-iterative wrapper: locally-owned, populated + RAII-cleaned by wrapper.
     * Helper does NOT call populate_call_scalars or populate_device_context. */

    /* Freeze actives host-side pre-drift (same snapshot used for self-pair
     * AND for ship-to-peers; prevents self/remote epoch
     * skew on the active rank). */
    std::vector<ActiveData> actives(N);
    if(N > 0) {
        build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs,
                                                  actives.data());
    }

    /* ---- SELF stages run ONCE, BEFORE the peer round loop. Self candidate
     * collection + drift + evaluation must PRECEDE peer evaluation so the
     * self-before-peer WRITE order (a neighbor j touched by a local active is
     * updated before any remote active's pair sees it) is preserved. Only the
     * PEER work streams in bounded rounds below.
     *
     * ORDERING NOTE — the reorder this introduces vs the prior single-pass form:
     * peer candidate COLLECTION now runs AFTER self EVALUATION (previously it ran
     * before). The peer-candidate walk keys membership on P[j].{Type,Pos,Mass}:
     *   - Type/Pos are NEVER mutated by a pair kernel (only drift writes Pos, and
     *     it is applied identically per round) → position/type membership is
     *     reorder-invariant for ALL specs.
     *   - Mass is the only membership field a pair kernel can change (mass-flux
     *     specs: MFV hydro; feedback add; sink swallow remove). For mass-
     *     PRESERVING specs (density, gradients, MFM hydro_force) the reorder is
     *     EXACTLY neutral. For mass-MUTATING specs a j crossing the Mass>0 gate
     *     during self eval could shift its peer-candidate membership — but those
     *     specs are ALREADY order-dependent, and this stays within that regime,
     *     not a new exactness contract.
     * NO generic j-write-exactness claim is made. Any mass-mutating spec routed
     * through streamed Mode-B at multi-round MUST be re-audited (evrard, the
     * validated case, is mass-preserving MFM). */

    /* The eligibility gate is RETIRED — EVERY Mode-B loop routes via
     * targeted export (per-type node band is cross-rank-fresh + dominates every
     * radius_policy; see mode_b_local_walker.h). Hard-coded true; the `else`
     * broadcast arms below are compile-time-DEAD cleanup debt, NOT a runtime
     * fallback -- pending physical deletion. */
    constexpr bool targeted_export_ok = true;
    const double jscale = nlr_spec_symmetric_j_radius_scale<Spec>();

    /* Targeted-export reverse map: topnode indices are stable between builds →
     * build ONCE, reuse for the fused walk. */
    ModeBTopleafMap topleaf_map;   /* shared read-only map, built once */
    ModeBExportSink export_sink;   /* per-query export sink (write-only during the walk) */
    if constexpr (targeted_export_ok) {
        if(N > 0 && nt > 1) { export_sink.ensure_size(nt); topleaf_map.build(); }
    } else {
        if(rank == 0 && gizmo_nlr_dispatch_trace_enabled()) {
            static bool s_announced = false;
            if(!s_announced) {
                s_announced = true;
                fprintf(stdout, "GX_MODEB_EXPORT rank=0 caller=%s BROADCAST "
                        "(radius_policy not scalar-hmax-dominated; broadcast retained)\n",
                        Spec::loop_name);
                fflush(stdout);
            }
        }
    }

    /* Fused-walk export CSR (targeted specs): per active, its per-peer export
     * node-lists, staged ONCE by the fused self walk and marshalled (no second
     * walk) by the peer round loop below. Active-ordered (csr_rec_off), peer-ascending
     * within an active, node order = walk append order. Sized O(total targeted
     * exports), NOT O(NTask*N). */
    struct FusedExportRec { int peer; int node_off; int n_nodes; };
    std::vector<int> csr_rec_off;                    /* size N+1: active aa -> [off[aa],off[aa+1]) recs */
    std::vector<FusedExportRec> csr_recs;
    std::vector<int> csr_nodes;
    long long diag_csr_bytes = 0; int diag_max_env_per_active = 0;

    /* Discovery-walk threading diagnostics (GX_MODEB_EXPORT, NLR_DIAG>=2 only).
     * The threading itself always runs above the work threshold; only the
     * per-thread drift accounting is diagnostic, so it is fully OFF when the
     * dispatch trace is off (drift_sink == nullptr => walker skips every counter
     * increment, no per-thread tctr allocated) — zero production overhead.
     * drift_certified is the O(1) SoA drift-cert query, read LAZILY the first
     * time a walk actually threads (below-threshold tiny-N calls never touch the
     * stamp); it reports whether the lazy per-node drift branch is provably dead
     * (stale_node_hits MUST be 0 when drift_certified==1). */
    const int modeb_nthreads = nlr_modeb_omp_nthreads();
    const bool nlr_diag_on = gizmo_nlr_dispatch_trace_enabled();
    ModeBDriftCounters drift_ctr_total{};
    ModeBDriftCounters* const drift_sink = nlr_diag_on ? &drift_ctr_total : nullptr;
    int drift_certified = -1;   /* -1 = no threaded walk ran (or diag off) */
    long long diag_omp_self = 0, diag_omp_recv = 0;   /* actual threads used per stage; 0 = serial */
    auto nlr_note_threaded_walk = [&]() {
        if(nlr_diag_on && drift_certified < 0)
            drift_certified = gpu_gravity_soa_drift_certified(All.Ti_Current) ? 1 : 0;
    };

    /* Stage 3: collect SELF candidates PRE-DRIFT. For targeted specs this is the
     * FUSED legacy-mode==0 walk — candidates + export CSR in ONE traversal, keyed
     * on the frozen actives[] snapshot (== the query the receiver walks) so the
     * candidate / export / receiver walks share one query SSOT. Broadcast specs
     * keep the plain candidate walk (they have no export walk to fuse). */
    std::vector<std::vector<int>> cand_self_tree;
    if(N > 0) {
        if constexpr (targeted_export_ok) {
            if(nt > 1) {
                /* want_cands: the tree walk needs candidates; the export CSR
                 * for the round loop is built regardless,
                 * so the fused walk runs with cand_out=nullptr there. */
                const bool want_cands = true;
                if(want_cands) cand_self_tree.assign(N, std::vector<int>{});
                csr_rec_off.assign(N + 1, 0);
                StageTimer t(tim ? &tim->dt_collect : nullptr);
                /* Thread the fused self walk when producing candidates above the
                 * work threshold. Each thread walks its actives into its OWN
                 * export sink + its OWN CSR segment (no shared push, no lock); a
                 * serial prefix-sum then assembles the active-ordered CSR
                 * BYTE-IDENTICALLY to the serial build (per-active walk order
                 * fixed; peers ascending within an active; node order = walk
                 * append order). The export walk (want_cands
                 * false) stays serial. */
                const bool use_omp_self = want_cands && nlr_modeb_use_omp(N, modeb_nthreads);
                if(use_omp_self) {
                    diag_omp_self = modeb_nthreads;
                    nlr_note_threaded_walk();
                    struct AaMeta { int tid; int rec_off; int n_recs; int node_off; int n_nodes; long long env_count; };
                    std::vector<AaMeta> meta(N);
                    std::vector<ModeBExportSink> tsink(modeb_nthreads);
                    for(auto& s : tsink) s.ensure_size(nt);
                    std::vector<std::vector<FusedExportRec>> trecs(modeb_nthreads);
                    std::vector<std::vector<int>> tnodes(modeb_nthreads);
                    std::vector<ModeBDriftCounters> tctr;   /* diagnostic only */
                    if(drift_sink) tctr.resize(modeb_nthreads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, MODEB_OMP_CHUNK_ACTIVE)
#endif
                    for(int aa = 0; aa < N; aa++) {
#ifdef _OPENMP
                        const int tid = omp_get_thread_num();
#else
                        const int tid = 0;
#endif
                        ModeBExportSink& sink = tsink[tid];
                        std::vector<FusedExportRec>& lrecs = trecs[tid];
                        std::vector<int>& lnodes = tnodes[tid];
                        AaMeta& m = meta[aa];
                        m.tid = tid; m.rec_off = (int)lrecs.size(); m.node_off = (int)lnodes.size();
                        m.n_recs = 0; m.n_nodes = 0; m.env_count = 0;
                        const double h_q = (double)actives[aa].h_search;
                        if(h_q <= 0) continue;
                        double pos_arr[3] = {(double)actives[aa].pos[0],
                                             (double)actives[aa].pos[1],
                                             (double)actives[aa].pos[2]};
                        sink.clear_all();
                        std::vector<int>* cand_ptr = &cand_self_tree[aa];
                        if(cand_ptr->capacity() == 0) cand_ptr->reserve(64);
                        mode_b_walk_and_export(pos_arr, h_q, neighbor_type_mask,
                                                Spec::search_mode, Spec::radius_policy,
                                                cand_ptr, topleaf_map, sink, jscale,
                                                drift_sink ? &tctr[tid] : nullptr);
                        for(int p = 0; p < nt; p++) {
                            if(p == rank) continue;
                            const std::vector<int>& nodes = sink.nodes_per_peer[p];
                            const int nn = (int)nodes.size();
                            if(nn == 0) continue;
                            FusedExportRec rec;
                            rec.peer = p; rec.node_off = (int)lnodes.size(); rec.n_nodes = nn;
                            lrecs.push_back(rec);
                            lnodes.insert(lnodes.end(), nodes.begin(), nodes.end());
                            m.n_recs++;
                            m.n_nodes += nn;
                            m.env_count += (nn + NODELISTLENGTH - 1) / NODELISTLENGTH;
                        }
                    }
                    /* Deterministic active-ordered merge. */
                    size_t total_recs = 0, total_nodes = 0;
                    for(int aa = 0; aa < N; aa++) { total_recs += meta[aa].n_recs; total_nodes += meta[aa].n_nodes; }
                    csr_recs.resize(total_recs);
                    csr_nodes.resize(total_nodes);
                    int rec_cursor = 0, node_cursor = 0;
                    for(int aa = 0; aa < N; aa++) {
                        csr_rec_off[aa] = rec_cursor;
                        const AaMeta& m = meta[aa];
                        const std::vector<FusedExportRec>& lrecs = trecs[m.tid];
                        const std::vector<int>& lnodes = tnodes[m.tid];
                        for(int rr = 0; rr < m.n_recs; rr++) {
                            FusedExportRec rec = lrecs[m.rec_off + rr];
                            const int local_node_off = rec.node_off;   /* thread-segment-relative */
                            rec.node_off = node_cursor;
                            for(int q = 0; q < rec.n_nodes; q++)
                                csr_nodes[node_cursor++] = lnodes[local_node_off + q];
                            csr_recs[rec_cursor++] = rec;
                        }
                        if(m.env_count > diag_max_env_per_active)
                            diag_max_env_per_active = (int)m.env_count;
                    }
                    csr_rec_off[N] = rec_cursor;
                    if(drift_sink) for(const auto& c : tctr) drift_ctr_total.add(c);
                } else {
                    for(int aa = 0; aa < N; aa++) {
                        csr_rec_off[aa] = (int)csr_recs.size();
                        const double h_q = (double)actives[aa].h_search;
                        if(h_q <= 0) continue;
                        double pos_arr[3] = {(double)actives[aa].pos[0],
                                             (double)actives[aa].pos[1],
                                             (double)actives[aa].pos[2]};
                        export_sink.clear_all();
                        std::vector<int>* cand_ptr = nullptr;
                        if(want_cands) {
                            cand_ptr = &cand_self_tree[aa];
                            if(cand_ptr->capacity() == 0) cand_ptr->reserve(64);
                        }
                        mode_b_walk_and_export(pos_arr, h_q, neighbor_type_mask,
                                                Spec::search_mode, Spec::radius_policy,
                                                cand_ptr, topleaf_map, export_sink, jscale);
                        /* stage this active's per-peer exports into the CSR */
                        long long env_this_active = 0;
                        for(int p = 0; p < nt; p++) {
                            if(p == rank) continue;
                            const std::vector<int>& nodes = export_sink.nodes_per_peer[p];
                            const int nn = (int)nodes.size();
                            if(nn == 0) continue;
                            FusedExportRec rec;
                            rec.peer = p; rec.node_off = (int)csr_nodes.size(); rec.n_nodes = nn;
                            csr_recs.push_back(rec);
                            csr_nodes.insert(csr_nodes.end(), nodes.begin(), nodes.end());
                            env_this_active += (nn + NODELISTLENGTH - 1) / NODELISTLENGTH;
                        }
                        if(env_this_active > diag_max_env_per_active)
                            diag_max_env_per_active = (int)env_this_active;
                    }
                    csr_rec_off[N] = (int)csr_recs.size();
                }
                diag_csr_bytes = (long long)csr_recs.size() * (long long)sizeof(FusedExportRec)
                               + (long long)csr_nodes.size() * (long long)sizeof(int);
            } else {
                /* single rank: no peers to export to → plain candidate walk. */
                StageTimer t(tim ? &tim->dt_collect : nullptr);
                int tu_self = 0;
                collect_candidates_pre_drift<Spec>(args, radii,
                                                    neighbor_type_mask,
                                                    DispatchPath::ModeB_HostWalker,
                                                    cand_self_tree, drift_sink, &tu_self);
                if(tu_self > 0) { diag_omp_self = tu_self; nlr_note_threaded_walk(); }
            }
        } else {
            StageTimer t(tim ? &tim->dt_collect : nullptr);
            int tu_self = 0;
            collect_candidates_pre_drift<Spec>(args, radii,
                                                neighbor_type_mask,
                                                DispatchPath::ModeB_HostWalker,
                                                cand_self_tree, drift_sink, &tu_self);
            if(tu_self > 0) { diag_omp_self = tu_self; nlr_note_threaded_walk(); }
        }
    }

    /* Self drift (split from the former self+peer union drift; peer candidates
     * drift per-round below). drift_particle is idempotent to All.Ti_Current
     * (constant across the helper), so a j that is both a self- and peer-
     * candidate drifts once — identical to the old combined union drift. */
    {
        StageTimer t(tim ? &tim->dt_drift : nullptr);
        if (N > 0) lazy_drift_candidates<Spec>(cand_self_tree);
    }

    /* Stage 8: evaluate SELF post-drift -> accums_out. */
    if(N > 0) {
        StageTimer t(tim ? &tim->dt_walk_self : nullptr);
        evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N,
                                          cand_self_tree, accums_out, EvalOMPPolicy::AllowProduction);
    }

    /* ---- PEER round loop (streaming). Build a BufferSize-bounded batch of
     * query envelopes from actives[cursor..N), exchange+evaluate+reply, advance
     * cursor, iterate until every rank is drained (legacy do/while +
     * Allreduce(ndone), code_block_xchange_perform_ops.h:9-191). Bounds total
     * in-flight export envelopes so forced-Mode-B is memory-safe at large
     * N_active. This caps the SENDER side only; the full large-N fix needs
     * receiver group staging.
     *
     * ALL Mode-B loops get TARGETED export (gate retired): walk the
     * local tree per active and export ONLY to peers whose remote subtree the query
     * reaches, carrying the exported start-nodes so the receiver resumes a bounded
     * walk. The per-type node band prunes the SYMMETRIC reach and is cross-rank-fresh
     * (via force_update_hmax), so the sender bounds every loop's reach on remote
     * peers. The `else` broadcast arm (n_nodes==0 to all peers) is compile-time-DEAD
     * cleanup debt, pending removal. Self-pair handled above; self entry stays
     * empty. */
    /* targeted_export_ok, jscale and exporter are hoisted above Stage 3 for
     * the fused walk. */
    /* Cap = legacy All.BunchSize analog: BufferSize / (query env + reply env).
     * Our NlrQueryEnvelope IS data_index+data_nodelist+ActiveData fused, so
     * counting envelopes == legacy counting DataIndexTable entries. (No env var;
     * All.BufferSize is the existing parameterfile parameter, default 100MB.) */
    constexpr size_t kReplyBytes = sizeof(ReplyEnvelope);
    const long long kEnvPairBytes = (long long)sizeof(Envelope) + (long long)kReplyBytes;
    long long bunch = ((long long)All.BufferSize * 1024 * 1024) /
                      (kEnvPairBytes > 0 ? kEnvPairBytes : 1);
    if(bunch < 1) bunch = 1;

    long long diag_export_qr = 0, diag_node_appends = 0;   /* scalar export volume (NLR diag) */
    long long diag_rounds = 0, diag_peak_sent = 0;
    long long diag_recv_groups = 0, diag_peak_recv_env = 0, diag_peak_recv_bytes = 0;
    long long eval_peer_work_peak = 0;   /* max per-round peer-eval work K (feeds omp_eval_peer diag) */
    int cursor = 0;
    int ndone  = 0;

    do {
        long long round_env_count = 0;
        std::vector<std::vector<Envelope>> queries_per_peer(nt);

        /* Stage 2 (bounded): fill this round's export batch from actives[cursor..).
         * MARSHAL from the fused walk's export CSR — NO walk here (the fused
         * legacy mode==0 walk already ran in Stage 3). Per active, MEASURE its
         * envelope count from the CSR, then commit-or-stop atomically —
         * measure-then-commit gives legacy's all-or-nothing-per-particle
         * rollback (an active never lands half its chunks in one round). An
         * active whose OWN set exceeds `bunch` ships in a solo oversized round
         * (graceful; loud diag) instead of aborting. */
        if(N > 0 && nt > 1) {
            if constexpr (targeted_export_ok) {
                int aa = cursor;
                for(; aa < N; aa++) {
                    const int r0 = csr_rec_off[aa], r1 = csr_rec_off[aa + 1];
                    /* envelopes this active would add across all peers */
                    long long add = 0;
                    for(int r = r0; r < r1; r++)
                        add += (csr_recs[r].n_nodes + NODELISTLENGTH - 1) / NODELISTLENGTH;
                    if(round_env_count > 0 && round_env_count + add > bunch) break; /* defer to next round */
                    if(round_env_count == 0 && add > bunch) {
                        nlr_warn_once_rank0("modeb_oversize_active",
                            "[mode_b B2a caller=%s] single active's export set (%lld envelopes, "
                            "~%lld bytes) exceeds BufferSize bunch (%lld envelopes); shipping a solo "
                            "oversized round — cap ineffective for this call (raise BufferSize).",
                            Spec::loop_name, add, add * kEnvPairBytes, bunch);
                    }
                    /* commit: chunked envelopes per exported peer (CSR records are peer-ascending). */
                    int rr = r0;
                    for(int p = 0; p < nt; p++) {
                        if(p == rank) continue;
                        if(rr < r1 && csr_recs[rr].peer == p) {
                            const FusedExportRec& rec = csr_recs[rr];
                            const int* nd = &csr_nodes[rec.node_off];
                            const int nn = rec.n_nodes;
                            diag_export_qr++; diag_node_appends += nn;
                            /* Chunk into NODELISTLENGTH-sized records (legacy opens a
                             * fresh export slot when a NodeList fills). Chunks cover
                             * disjoint subtrees → the slot-keyed reply merge sums their
                             * partial results without double counting. All chunks of a
                             * (query,peer) group land in THIS round (all-or-nothing
                             * above), so each group stays contiguous. */
                            for(int c = 0; c < nn; c += NODELISTLENGTH) {
                                Envelope env;
                                env.origin_slot = aa;
                                env.origin_rank = rank;
                                int cnt = 0;
                                for(; cnt < NODELISTLENGTH && (c + cnt) < nn; cnt++) {
                                    env.NodeList[cnt] = nd[c + cnt];
                                }
                                env.n_nodes = cnt;
                                env.reserved_wire_padding = 0;
                                for(int t = cnt; t < NODELISTLENGTH; t++) env.NodeList[t] = -1;
                                env.active = actives[aa];
                                queries_per_peer[p].push_back(env);
                            }
                            rr++;
                            continue;
                        }
                    }
                    round_env_count += add;
                }
                cursor = aa;
            } else {
                /* Broadcast: each active adds exactly (nt-1) envelopes. */
                int aa = cursor;
                for(; aa < N; aa++) {
                    const long long add = (long long)(nt - 1);
                    if(round_env_count > 0 && round_env_count + add > bunch) break;
                    if(round_env_count == 0 && add > bunch) {
                        nlr_warn_once_rank0("modeb_oversize_active_bcast",
                            "[mode_b B2a caller=%s] broadcast active adds %lld envelopes > BufferSize "
                            "bunch (%lld); solo oversized round — cap ineffective (raise BufferSize).",
                            Spec::loop_name, add, bunch);
                    }
                    for(int p = 0; p < nt; p++) {
                        if(p == rank) continue;
                        Envelope env;
                        env.origin_slot = aa;
                        env.origin_rank = rank;
                        env.n_nodes = 0;
                        env.reserved_wire_padding = 0;   /* legit broadcast, matches expected */
                        for(int t = 0; t < NODELISTLENGTH; t++) env.NodeList[t] = -1;
                        env.active = actives[aa];
                        queries_per_peer[p].push_back(env);
                    }
                    round_env_count += add;
                }
                cursor = aa;
            }
        } else {
            cursor = N;   /* nothing to export (N==0 or single rank) */
        }

        diag_rounds++;
        if(round_env_count > diag_peak_sent) diag_peak_sent = round_env_count;

        /* Stage 4: exchange queries. Every rank participates even if it queued
         * 0 this round (peers may target this rank's pool); the Allreduce(ndone)
         * at the round's end keeps every rank's round count equal, so the
         * exchange stays balanced. begin() posts ALL query-payload Isends + ALL
         * reply Irecvs up front; the receiver then stages, evaluates, and
         * answers incoming queries in memory-bounded WHOLE-PEER groups (the
         * legacy import sub-chunk loop, code_block_xchange_perform_ops.h:96-174)
         * instead of materializing every peer's payload at once. Group budget =
         * the same All.BufferSize that sizes the sender bunch. The bound covers
         * TRANSPORT payloads only (envelopes + replies) — candidate vectors and
         * pair-kernel scratch scale with the group's query content, not with
         * NTask. Peers are consumed in ascending rank order, so the
         * concatenated per-group evaluation sequence — and the post-loop reply
         * merge — keep the exact pre-group order (matters for j-writing specs). */
        using XReply = ReplyEnvelope;
        ModeBBoundedExchange<Envelope, XReply> xch;
        {
            StageTimer t(tim ? &tim->dt_exchange_q : nullptr);
            xch.begin(queries_per_peer);
        }
        const size_t group_budget_bytes =
            (size_t)((double)All.BufferSize * 1024.0 * 1024.0);

        std::vector<int> group_peers;
        std::vector<std::vector<Envelope>> group_queries;
        while(true) {
            bool have_group;
            {
                StageTimer t(tim ? &tim->dt_exchange_q : nullptr);
                have_group = xch.next_group(group_budget_bytes, group_peers, group_queries);
            }
            if(!have_group) break;
            diag_recv_groups++;

    /* Stage 5: flatten THIS GROUP's envelopes and build the provenance map.
     * provenance[k] carries:
     *   - source_gidx / source_qi: where in group_queries[][] this k came
     *     from (used for unflattening replies back into per-peer arrays);
     *     source_peer = the peer's rank, for diagnostics.
     *   - origin_slot / origin_rank: copied from the received envelope; ride
     *     into the REPLY envelope so the active rank can merge by slot
     *     without relying on transport ordering (symmetric
     *     query and reply envelopes). */
    std::vector<ActiveData> peer_actives;
    struct Provenance {
        int source_gidx; int source_peer; int source_qi;
        int origin_slot; int origin_rank;
    };
    std::vector<Provenance> peer_provenance;
    /* Parallel to peer_actives[k]: the exported start-node list carried in the
     * received envelope, flattened K*NODELISTLENGTH, so the receiver walk
     * (collect_candidates_for_remote_queries) resumes from those nodes. */
    std::vector<int> peer_nodelist_flat;
    std::vector<int> peer_nnodes;
    size_t total_recv = 0, total_recv_bytes = 0;
    for(size_t gi = 0; gi < group_peers.size(); gi++) {
        total_recv += group_queries[gi].size();
        total_recv_bytes += group_queries[gi].size() * (sizeof(Envelope) + sizeof(XReply));
    }
    if((long long)total_recv > diag_peak_recv_env) diag_peak_recv_env = (long long)total_recv;
    if((long long)total_recv_bytes > diag_peak_recv_bytes) diag_peak_recv_bytes = (long long)total_recv_bytes;
    peer_actives.reserve(total_recv);
    peer_provenance.reserve(total_recv);
    peer_nodelist_flat.reserve(total_recv * NODELISTLENGTH);
    peer_nnodes.reserve(total_recv);
    for(size_t gi = 0; gi < group_peers.size(); gi++) {
        const int p = group_peers[gi];
        for(int qi = 0; qi < (int)group_queries[gi].size(); qi++) {
            const Envelope& env = group_queries[gi][qi];
            /* Sanity: envelope's origin_rank should equal sender p. */
            if(env.origin_rank != p) {
                fprintf(stderr, "[neighbor_loop_runner ABORT rank=%d caller=%s] "
                        "envelope origin_rank=%d but received from peer=%d "
                        "(qi=%d). Transport corruption?\n",
                        rank, Spec::loop_name, env.origin_rank, p, qi);
                fflush(stderr);
                endrun(81220);
            }
            /* Soft bad-stop on corruption but DO NOT skip: the reply array is sized by
             * recv_counts[p], so omitting this qi would leave a default-initialized reply
             * (bogus origin_rank=0 could be merged into slot 0 on the peer). Keep the entry
             * with the transport-consistent origin rank p (== env.origin_rank when clean);
             * the run drains at the next phase poll with reply choreography intact. */
            peer_actives.push_back(env.active);
            peer_provenance.push_back({(int)gi, p, qi, env.origin_slot, p});
            peer_nnodes.push_back(env.n_nodes);
            for(int t = 0; t < NODELISTLENGTH; t++) peer_nodelist_flat.push_back(env.NodeList[t]);
        }
    }
    /* Note: receiver-side binding of peer_actives (and per-eval-pass rebinding
     * of both self actives and peer actives) is performed inside
     * evaluate_pairs_post_drift, gated on
     * nlr_spec_has_bind_active_to_eval_context_v<Spec>. The bind runs once
     * per active per eval pass with the EXACT eval ctx. No standalone
     * post-flatten rebind needed here. */

    /* Stage 6: collect PEER candidate sets PRE-DRIFT (against MY local pool). */
    std::vector<std::vector<int>> cand_peer_tree;
    {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        int tu_recv = 0;
        collect_candidates_for_remote_queries<Spec>(peer_actives,
                                                     peer_nodelist_flat, peer_nnodes,
                                                     neighbor_type_mask,
                                                     DispatchPath::ModeB_HostWalker,
                                                     cand_peer_tree, drift_sink, &tu_recv);
        if(tu_recv > 0) { diag_omp_recv = tu_recv; nlr_note_threaded_walk(); }
    }

    /* Stage 7 (peer): drift THIS round's peer candidate sets (self candidates
     * were drifted once before the round loop). Idempotent to All.Ti_Current. */
    {
        StageTimer t(tim ? &tim->dt_drift : nullptr);
        lazy_drift_candidates<Spec>(cand_peer_tree);
    }

    /* Stage 9: evaluate PEER queries post-drift -> peer_replies, shipped back
     * to the home rank. */
    const int K = (int)peer_actives.size();
    std::vector<AccumData> peer_replies(K);
    if(K > 0) {
        StageTimer t(tim ? &tim->dt_walk_peer : nullptr);
        if((long long)K > eval_peer_work_peak) eval_peer_work_peak = K;
        evaluate_pairs_post_drift<Spec>(ctx, peer_actives.data(), K,
                                          cand_peer_tree, peer_replies.data(), EvalOMPPolicy::AllowProduction);
    }

    /* Stage 10 (per group): build reply envelopes (origin_slot/rank copied from
     * each received query envelope), unflatten into per-peer arrays via the
     * provenance map, then send THIS group's replies — after which the group's
     * buffers are released (the whole point of the group staging). Production
     * replies share one payload type (XReply), so there is one
     * reply exchange per group, never two with identical MPI tags. */
    {
        std::vector<std::vector<XReply>> replies_for_group(group_peers.size());
        for(size_t gi = 0; gi < group_peers.size(); gi++) {
            replies_for_group[gi].assign(group_queries[gi].size(), XReply{});
        }
        for(int k = 0; k < K; k++) {
            const Provenance& pv = peer_provenance[k];
            XReply& re = replies_for_group[pv.source_gidx][pv.source_qi];
            re.origin_slot = pv.origin_slot;
            re.origin_rank = pv.origin_rank;
            re.accum = peer_replies[k];
        }
        StageTimer t(tim ? &tim->dt_exchange_r : nullptr);
        /* send_group_replies posts the reply Isends and waits them in finish();
         * move the group's reply buffers into the exchange so they outlive the Isend. */
        xch.send_group_replies(group_peers, std::move(replies_for_group));
    }
        }   /* end whole-peer group loop */

        /* Stage 11: drain the exchange (remaining query-payload Isends + all
         * pre-posted reply Irecvs, byte-count-asserted), then merge replies
         * into accums_out by envelope.origin_slot. Pinned deterministic order:
         * ascending peer rank, ascending qi — identical to the pre-group
         * single-shot merge (self contribution already in accums_out from
         * stage 8). Asserts each reply envelope's origin_rank == ThisTask. */
        {
            auto recv_replies = [&]{
                StageTimer t(tim ? &tim->dt_exchange_r : nullptr);
                return xch.finish();
            }();
            StageTimer t(tim ? &tim->dt_reduce : nullptr);
            for (int p = 0; p < nt; p++) {
                if (p == rank) continue;
                const int q_to_p = xch.sent_counts[p];
                for (int qi = 0; qi < q_to_p; qi++) {
                    const XReply& re = recv_replies[p][qi];
                    if(re.origin_rank != rank) {
                        fprintf(stderr, "[neighbor_loop_runner ABORT rank=%d caller=%s] "
                                "reply envelope origin_rank=%d != ThisTask=%d from "
                                "peer=%d qi=%d. Transport/peer-side corruption?\n",
                                rank, Spec::loop_name, re.origin_rank, rank, p, qi);
                        fflush(stderr);
                        /* reply exchange already completed; soft bad-stop + skip the corrupt reply. */
                        endrun(81223); continue;
                    }
                    const int slot = re.origin_slot;
                    if(slot < 0 || slot >= N) {
                        fprintf(stderr, "[neighbor_loop_runner ABORT rank=%d caller=%s] "
                                "reply envelope slot %d out of range [0,%d) from peer %d.\n",
                                rank, Spec::loop_name, slot, N, p);
                        fflush(stderr);
                        /* skip: continuing would merge into accums_out[slot] with slot OOB. */
                        endrun(81224); continue;
                    }
                    Spec::merge_accum(accums_out[slot], re.accum);
                }
            }
        }

        /* Termination (legacy 186-191): done when my cursor drained; the SUM
         * Allreduce makes every rank run the SAME number of rounds so the
         * per-round query/reply exchanges stay collective-balanced. Ranks that
         * finished sending keep entering as 0-send receivers for peers still
         * draining. */
        int ndone_flag = (cursor >= N) ? 1 : 0;
        {
            StageTimer t(tim ? &tim->dt_exchange_q : nullptr);
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    } while(ndone < NTask);

    if(gizmo_nlr_dispatch_trace_enabled()) {
        /* peak_recv_env/bytes bound TRANSPORT payloads only (envelopes+replies
         * staged per group) — not candidate vectors or kernel scratch.
         * export_csr_bytes/max_env_per_active bound the fused walk's materialized
         * export CSR (recs+nodes) built once in Stage 3 (targeted specs).
         * nthr = OpenMP threads available; omp_self/omp_recv = threads used for
         * the self / receiver discovery walks (0 = ran serial, below the work
         * threshold). drift_certified = O(1) SoA drift-cert query (1 = the lazy
         * per-node drift branch is provably dead; -1 = not queried on a serial
         * call). The three drift counts are lazy per-node drifts under threading:
         * stale_node_hits = fast-path saw a stale node; lazy_drift_performed =
         * this thread drifted it; lazy_drift_raced = a peer drifted it first.
         * stale_node_hits MUST be 0 on a drift_certified=1 run.
         * omp_eval_self / omp_eval_peer = the production self / peer eval
         * threading decisions, each mirroring the evaluate_pairs_post_drift
         * eval-threading (non-SerialOnly) gate on its OWN work count (N self; peak per-round K
         * peer). Reported separately so peer-eval threading is never hidden
         * behind the self-eval decision (a call can be self-serial/peer-threaded
         * or vice versa). */
        char eval_self_label[40], eval_peer_label[40];
        nlr_modeb_eval_decision_label(eval_self_label, sizeof(eval_self_label),
                                      nlr_spec_modeb_eval_omp<Spec>(),
                                      nlr_spec_modeb_eval_omp_is_explicit_v<Spec>,
                                      EvalOMPPolicy::AllowProduction,
                                      modeb_nthreads, (long long)N);
        nlr_modeb_eval_decision_label(eval_peer_label, sizeof(eval_peer_label),
                                      nlr_spec_modeb_eval_omp<Spec>(),
                                      nlr_spec_modeb_eval_omp_is_explicit_v<Spec>,
                                      EvalOMPPolicy::AllowProduction,
                                      modeb_nthreads, eval_peer_work_peak);
        fprintf(stdout, "GX_MODEB_EXPORT rank=%d caller=%s N=%d export_qr=%lld node_appends=%lld "
                "bunch=%lld env_bytes=%zu reply_bytes=%zu rounds=%lld peak_sent_env=%lld "
                "recv_groups=%lld peak_recv_env=%lld peak_recv_bytes=%lld "
                "export_csr_bytes=%lld max_env_per_active=%d "
                "nthr=%d omp_self=%lld omp_recv=%lld omp_eval_self=%s omp_eval_peer=%s drift_certified=%d "
                "stale_node_hits=%lld lazy_drift_performed=%lld lazy_drift_raced=%lld\n",
                rank, Spec::loop_name, N, diag_export_qr, diag_node_appends,
                bunch, sizeof(Envelope), kReplyBytes, diag_rounds, diag_peak_sent,
                diag_recv_groups, diag_peak_recv_env, diag_peak_recv_bytes,
                diag_csr_bytes, diag_max_env_per_active,
                modeb_nthreads, diag_omp_self, diag_omp_recv,
                eval_self_label, eval_peer_label,
                drift_certified,
                drift_ctr_total.stale_node_hits, drift_ctr_total.lazy_drift_performed,
                drift_ctr_total.lazy_drift_raced);
        fflush(stdout);
    }


    /* End of helper. Caller decides whether to call apply_active_writeback
     * (final-only for iterative; per-call for non-iterative) and whether
     * to emit active_dumps (after writeback for non-iter; deferred for
     * iter until proper diagnostic plumbing lands). */
}

/* ============================================================================
 * run_mode_b_remote_impl<Spec> — non-iterative thin wrapper.
 *
 * Allocates per-call AccumData buffer, calls helper, runs final
 * apply_active_writeback + active_dumps emit. Same epoch order and same
 * writeback per active as the earlier monolithic impl.
 * ========================================================================== */
template <typename Spec>
static void run_mode_b_remote_impl(const neighbor_loop_args& args, const double *radii,
                                   RunnerStageTimer *tim = nullptr)
{
    using AccumData    = typename Spec::AccumData;
    using DeviceCtx    = typename Spec::DeviceContext;

    const int N = args.num_active;

    /* Capture CallScalars per call (non-iter wrapper). */
    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    /* Build per-call DeviceContext + RAII cleanup guard (helper
     * no longer builds its own ctx; caller does). Behavior byte-equivalent
     * to the earlier monolith — guard runs Spec::cleanup_device_context at
     * scope exit, after the writeback loop completes. */
    DeviceCtx ctx;
    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;
    static_assert(std::is_base_of<NeighborLoopDeviceContextBase, DeviceCtx>::value,
                  "Spec::DeviceContext must publicly derive from NeighborLoopDeviceContextBase");
    static_assert(std::is_trivially_copyable<DeviceCtx>::value,
                  "Spec::DeviceContext must be trivially copyable; "
                  "captured by value into Kokkos device lambdas");
    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(args, ctx);
    }
    NlrDeviceContextCleanupGuard<Spec> _nlr_dctx_cleanup_guard(args, ctx);

    /* Caller-owned per-call AccumData buffer; helper writes into it.
     * Explicit Spec::zero_accum per slot: defensive
     * against future evaluate_pairs variants that don't zero internally.
     * Today evaluate_pairs_post_drift calls Spec::zero_accum at line 812
     * before walking candidates, so this outer zero is redundant — but
     * makes the AccumData contract explicit at the caller level (sentinel-
     * bearing AccumData like sink_feed's Sink_PotentialMinimumOfNeighbors
     * needs Spec::zero_accum, NOT default-construction). */
    std::vector<AccumData> accums_self(N);
    for (int aa = 0; aa < N; aa++) {
        Spec::zero_accum(accums_self[aa]);
    }

    /* Helper runs Stages 1-12; writeback is the wrapper's responsibility
     * (preserves the earlier timing). */
    mode_b_remote_evaluate_into_buffer<Spec>(args, radii, cs, ctx,
                                                       nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                                                       (N > 0) ? accums_self.data() : nullptr,
                                                       tim);

    /* Stage 12 final: writeback per active. */
    {
        StageTimer t(tim ? &tim->dt_writeback : nullptr);
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_self[aa]);
        }
    }

}

template <typename Spec>
static void run_mode_b_remote(const neighbor_loop_args& args, const double *radii,
                              RunnerStageTimer *tim = nullptr) {
    run_mode_b_remote_impl<Spec>(args, radii, tim);
}

/* ============================================================================
 * External-CSR staging helpers (hydro corridor support).
 *
 * When args.external_csr is non-null, Mode A skips gpu_ngb_list_build and
 * instead stages the caller's host CSR into Kokkos memory shaped like a
 * gpu_neighbor_list_t — so the rest of run_mode_a is path-agnostic. Only
 * the build site (replaced with this helper) and the free site (the
 * matching helper below) differ between the two paths.
 *
 * The spatial-index fields of gnl (d_tiles / d_bvh / d_pool / ntiles /
 * bvh_root / periodic_flags / box_sizes / box_halves) stay zero/null
 * because the pair_kernel does not read them (it uses nearest_xyz which
 * reads All.BoxSize_* via the AllDeviceMirror).
 *
 * The runner OWNS the SharedSpace/DeviceSpace allocations made here and
 * frees them in nlr_free_external_csr_gnl(). It does NOT free the caller's
 * host buffers (active_indices / offsets / neighbors). Contract: caller
 * keeps host CSR alive for the duration of every run_neighbor_loop call
 * that injects it; the corridor design owns CSR across multiple consumers
 * by holding it in the gizmo_sym_* globals. */
static inline void
nlr_stage_external_csr_into_gnl(const nlr_external_csr *ext,
                                gpu_neighbor_list_t *gnl)
{
    /* zero-init everything; we touch only what we own */
    memset(gnl, 0, sizeof(*gnl));

    gnl->num_active  = ext->num_active;
    gnl->total_pairs = ext->total_pairs;

    const size_t off_bytes = (size_t)(ext->num_active + 1) * sizeof(int64_t);
    const size_t act_bytes = (size_t)(ext->num_active > 0 ? ext->num_active : 1)
                             * sizeof(int);
    const int64_t pairs = ext->total_pairs;
    const size_t nbr_bytes = (size_t)(pairs > 0 ? pairs : 1) * sizeof(int);

    /* offsets and d_active in SharedSpace (UVM) — host memcpy is fine */
    gnl->offsets  = (int64_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_csr_offsets", off_bytes);
    gnl->d_active = (int *)     Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_active", act_bytes);
    memcpy(gnl->offsets,  ext->offsets,          off_bytes);
    memcpy(gnl->d_active, ext->active_indices,   act_bytes);

    /* neighbors in DeviceSpace (GPU HBM) — must deep_copy from host view */
    gnl->neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>("modea_csr_neighbors", nbr_bytes);
    if(pairs > 0) {
        Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_n(ext->neighbors, (size_t)pairs);
        Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_n(gnl->neighbors, (size_t)pairs);
        Kokkos::deep_copy(d_n, h_n);
    }
}

static inline void
nlr_free_external_csr_gnl(gpu_neighbor_list_t *gnl)
{
    if(gnl->neighbors) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(gnl->neighbors);
    if(gnl->d_active)  Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->d_active);
    if(gnl->offsets)   Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gnl->offsets);
    gnl->neighbors = nullptr;
    gnl->d_active  = nullptr;
    gnl->offsets   = nullptr;
    gnl->num_active = 0;
    gnl->total_pairs = 0;
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
 *
 * External CSR injection (args.external_csr != nullptr): the caller has
 * already built a symmetric gas CSR (e.g. corridor's gizmo_sym_*) and we
 * stage it into the gnl shape via nlr_stage_external_csr_into_gnl() instead
 * of calling gpu_ngb_list_build. SidxCacheKind::GasOnly only; other Specs
 * MUST leave args.external_csr null. Existing Specs unaffected.
 * ========================================================================== */

/* Per-active Mode-A staging can be CHUNKED so the PER-ACTIVE arrays
 * (d_actives + d_accums) stay bounded regardless of the rank-active count. This
 * cap bounds ONLY those two arrays: radii_uvm and the gnl CSR (offsets/neighbors)
 * are built ONCE over all N and stay full-N, so a loop whose CSR dominates its
 * per-active PODs is not helped by this cap (for the gradient loop the fat
 * accumulator dominates the CSR, which is why it is the useful customer). The
 * bound is an internal per-rank byte target (not a user knob, not a live
 * free-memory query), sized conservatively so ranks_per_node * cap stays under a
 * node's transient headroom for the target rank counts; it is NOT derived from
 * the live rank count, so a very dense packing may need a lower value -- a
 * rank-count-aware node budget is the eventual fix.
 *
 * SAFETY: the Mode-A arena ALIASES the host P/CellP (gpu_particles_arena_acquire
 * sets arena_P = P_host; there is no snapshot), so a per-chunk writeback IS
 * visible to later chunks' stage/pair reads. The unchunked path ran every pair
 * kernel BEFORE any writeback; chunking interleaves them. It is therefore
 * bitwise-safe ONLY when a Spec's apply_active_writeback writes no field that
 * any pair_kernel reads. That is a per-Spec AND per-config property, NOT implied
 * by "i-side" or "no ghost writeback" -- so it is an explicit opt-in trait
 * (Spec::mode_a_chunked_active_staging) the Spec author sets after auditing it.
 * Absent -> the Spec always stages the full set (K == N). */
static constexpr size_t NLR_MODE_A_STAGING_BYTES_CAP = (size_t)128 * 1024 * 1024;

/* Opt-in Spec trait gating the staging chunker above. Absent -> false. */
template <typename Spec, typename = void>
struct nlr_has_mode_a_chunked_active_staging : std::false_type {};
template <typename Spec>
struct nlr_has_mode_a_chunked_active_staging<Spec, decltype((void)Spec::mode_a_chunked_active_staging)>
    : std::true_type {};
template <typename Spec> static constexpr bool nlr_mode_a_chunked_active_staging_v() {
    if constexpr (nlr_has_mode_a_chunked_active_staging<Spec>::value) {
        return Spec::mode_a_chunked_active_staging;
    } else { return false; }
}

/* Defined below with the other lifecycle-trait helpers; forward-declared here as
 * a defensive backstop so a mis-set opt-in on a ghost-writeback Spec cannot chunk. */
template <typename Spec> static constexpr bool nlr_uses_ghost_writeback_v();

template <typename Spec>
static void run_mode_a(const neighbor_loop_args& args, const double *radii,
                       RunnerStageTimer *tim = nullptr)
{
    using ActiveData   = typename Spec::ActiveData;
    using AccumData    = typename Spec::AccumData;
    using ScatterData  = typename Spec::ScatterData;
    using NeighborData = typename Spec::NeighborData;
    using CallScalars  = typename Spec::CallScalars;
    using DeviceCtx    = typename Spec::DeviceContext;

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
    double *radii_uvm = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_radii",
        N * sizeof(double));
    for(int aa = 0; aa < N; aa++) {
        radii_uvm[aa] = radii[aa];
    }

    /* (2) Host, pre-arena: capture per-call scalar globals into a POD. */
    CallScalars cs = Spec::populate_call_scalars(args);

    /* Arena + freshness (matches sinks/sink_environment_gpu.cc:76,86). The
     * caller is responsible for the args.CellP=NULL-when-no-gas decision;
     * runner does not read All.TotN_gas. */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total, args.P, args.CellP);
    struct particle_data *P_gpu    = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = (args.CellP != nullptr)
                                        ? gpu_particles_arena_CellP() : nullptr;

    /* NGL build using pre-arena radii. SIDX cache resolved from spec.
     * Hydro-corridor external-CSR path: when args.external_csr is non-null
     * the caller has already built a symmetric gas CSR — stage it into gnl
     * shape and skip the build. Contract: GasOnly Specs only. */
    gpu_neighbor_list_t gnl;
    gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                       Spec::loop_name);
    if(args.external_csr != nullptr) {
        /* Runtime checks — cannot be static_assert because that would fire
         * at template instantiation for every NotIterative Spec, including
         * non-GasOnly ones (sink_env1, dm_fuzzy, etc.) whose callers never
         * set external_csr. Compile-time enforcement is impossible since
         * external_csr is a runtime args field, not a Spec constexpr.
         *
         * All checks are UNCONDITIONAL (never diagnostic-gated):
         * external CSR injection is a sharp tool, contract violations
         * cause silent wrong-particle writeback (kernel stages for
         * external_csr->active_indices[aa] but writeback applies
         * d_accums[aa] to args.active_list[aa]). Detect loud always.
         *
         * On violation: soft bad-stop + release the arena and free radii_uvm
         * (both acquired/staged above) + return, skipping the corrupt-CSR
         * staging and the device walk. run_mode_a issues no MPI, so this
         * return cannot desync a peer; the caller's next phase poll drains.
         * else-if chain so a null pointer is never deref'd by a later check
         * (offsets[0] read only once ec->offsets is confirmed non-null). */
        const nlr_external_csr *ec = args.external_csr;
        const char *csr_err = nullptr;
        int         csr_code = 0;
        if(Spec::sidx_cache_kind != SidxCacheKind::GasOnly) { csr_err = "requires GasOnly cache"; csr_code = 7300; }
        else if(ec->num_active != N)                        { csr_err = "num_active mismatch"; csr_code = 7301; }
        else if(!ec->active_indices)                        { csr_err = "null active_indices"; csr_code = 7302; }
        else if(!ec->offsets)                               { csr_err = "null offsets"; csr_code = 7303; }
        else if(ec->total_pairs < 0)                        { csr_err = "negative total_pairs"; csr_code = 7304; }
        else if(ec->total_pairs > 0 && !ec->neighbors)      { csr_err = "null neighbors with total_pairs>0"; csr_code = 7305; }
        else if(N > 0 && ec->offsets[0] != 0)               { csr_err = "offsets[0]!=0"; csr_code = 7306; }
        else if(N > 0 && ec->offsets[N] != ec->total_pairs) { csr_err = "offsets[N]!=total_pairs"; csr_code = 7307; }
        else {
            /* Row order MUST match args.active_list elementwise — otherwise
             * the kernel accumulates for ec->active_indices[aa] but the host
             * writeback re-applies d_accums[aa] to args.active_list[aa]. */
            for(int aa = 0; aa < N; aa++) {
                if(ec->active_indices[aa] != args.active_list[aa]) { csr_err = "row order != active_list"; csr_code = 7308; break; }
                if(ec->offsets[aa+1] < ec->offsets[aa])            { csr_err = "non-monotonic offsets"; csr_code = 7309; break; }
            }
        }
        if(csr_err != nullptr) {
            fprintf(stderr, "[neighbor_loop_runner rank=%d caller=%s] external-CSR contract violation: %s\n",
                    ThisTask, Spec::loop_name, csr_err);
            fflush(stderr);
            endrun(csr_code);
            gpu_particles_arena_release();
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
            return;
        }
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        nlr_stage_external_csr_into_gnl(ec, &gnl);
    } else {
        StageTimer t(tim ? &tim->dt_collect : nullptr);
        /* Active-source-in-pool contract: stage explicit P[active_i].Pos for specs
         * whose active sources may be non-pool (else nullptr keeps the compact
         * fast-path). See neighbor_loop_runner.h. Radii are already explicit. */
        std::vector<double> _nlr_srcpos_storage;
        const double* _nlr_srcpos = nlr_stage_explicit_source_positions<Spec>(
            args.P, args.active_list, N, _nlr_srcpos_storage);
        gpu_ngb_list_build(P_gpu, args.num_total,
                           args.active_list, N,
                           Spec::search_mode,
                           (int)nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                           &gnl, sidx,
                           1.0, radii_uvm, _nlr_srcpos, Spec::loop_name,
                           nlr_spec_symmetric_j_radius_scale<Spec>(),
                           Spec::radius_policy);
    }

    /* Chunk the fat per-active staging so its transient footprint is bounded.
     * K stays at the full N unless (a) the Spec opts into chunked staging (its
     * writeback is disjoint from every pair read -- see the trait contract) AND
     * (b) staging all N rows would exceed the internal per-rank byte cap. The
     * NoScatter / !uses_ghost_writeback conjuncts are defensive backstops (every
     * current Spec is NoScatter; the opt-in trait is the real contract). K
     * degrades to 1 if a single record exceeds the cap. K == N reproduces the
     * unchunked path exactly. */
    const bool chunk_ok =
        nlr_mode_a_chunked_active_staging_v<Spec>()
        && std::is_same<typename Spec::ScatterData, NoScatter>::value
        && !nlr_uses_ghost_writeback_v<Spec>();
    const size_t rec_bytes = sizeof(ActiveData) + sizeof(AccumData);
    int K = N;
    if(chunk_ok && (size_t)N * rec_bytes > NLR_MODE_A_STAGING_BYTES_CAP) {
        size_t k = NLR_MODE_A_STAGING_BYTES_CAP / rec_bytes;
        if(k < 1) { k = 1; }
        if(k < (size_t)N) { K = (int)k; }
    }

    /* UVM-allocate chunk-sized ActiveData[] and AccumData[] arrays. */
    ActiveData *d_actives = (ActiveData *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_active_data", (size_t)K * sizeof(ActiveData));
    AccumData *d_accums = (AccumData *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_accum_data", (size_t)K * sizeof(AccumData));

    /* Build DeviceContext. Specs that extend Spec::DeviceContext beyond
     * NeighborLoopDeviceContextBase get populate_device_context invoked
     * here; base-only Specs skip the call (trait check). */
    DeviceCtx ctx;
    ctx.P         = P_gpu;
    ctx.CellP     = CellP_gpu;
    ctx.num_total = args.num_total;
    static_assert(std::is_base_of<NeighborLoopDeviceContextBase, DeviceCtx>::value,
                  "Spec::DeviceContext must publicly derive from NeighborLoopDeviceContextBase");
    static_assert(std::is_trivially_copyable<DeviceCtx>::value,
                  "Spec::DeviceContext must be trivially copyable; the runner captures it by value into Kokkos device lambdas");
    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(args, ctx);
    }
    NlrDeviceContextCleanupGuard<Spec> _nlr_dctx_cleanup_guard(args, ctx);

    /* (3)-(4) Chunked stage -> pair-kernel -> writeback. Each chunk [c0, c0+n)
     * stages into chunk-local slots [0,n): CSR rows / radii are read at the
     * absolute active index aa = c0 + kk, results land in d_actives[kk] /
     * d_accums[kk], and the writeback re-applies them to args.active_list[aa]
     * before the next chunk reuses the arrays. d_active/offsets/neighbors are
     * gnl-resident (built once over all N). K == N is a single pass identical
     * to the unchunked path. */
    int     *d_active_idx = gnl.d_active;
    int64_t *offsets      = gnl.offsets;
    int     *neighbors    = gnl.neighbors;
    for(int c0 = 0; c0 < N; c0 += K) {
        const int n = (N - c0 < K) ? (N - c0) : K;

        /* stage ActiveData for [c0, c0+n) into chunk-local [0,n). */
        gizmo_gpu_kernel_launch("nlr_stage_active", n, KOKKOS_LAMBDA(int kk) {
            const int aa = c0 + kk;
            d_actives[kk] = Spec::load_active(ctx, aa, d_active_idx[aa],
                                              radii_uvm[aa], cs);
        });

        /* pair-kernel over [c0, c0+n) — generic over Spec. */
        {
            StageTimer t(tim ? &tim->dt_walk_self : nullptr);
            const double t_pair_kernel_start = my_second();
            gizmo_gpu_kernel_launch(Spec::loop_name, n, KOKKOS_LAMBDA(int kk) {
                const int aa = c0 + kk;
                Spec::zero_accum(d_accums[kk]);
                const ActiveData& a = d_actives[kk];
                ScatterData s{};                     /* NoScatter for ActiveReduceOnly */
                int64_t start = offsets[aa], end = offsets[aa + 1];
                for(int64_t nn = start; nn < end; nn++) {
                    int j = neighbors[nn];
                    IdentitySidecar id{};            /* NoIdentity */
                    NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                    Spec::pair_kernel(a, nb, d_accums[kk], s);
                }
            });
            cpu_charge_child(CPU_PAIR_KERNEL, timediff(t_pair_kernel_start, my_second()));
        }
        /* Launches fenced internally by gizmo_gpu_kernel_launch. UVM coherent ->
         * host reads d_accums[0,n) directly. */

        /* (4) host writeback for [c0, c0+n) before the next chunk reuses arrays. */
        {
            StageTimer t(tim ? &tim->dt_writeback : nullptr);
            for(int kk = 0; kk < n; kk++) {
                Spec::apply_active_writeback(args, c0 + kk, args.active_list[c0 + kk], d_accums[kk]);
            }
        }
    }


    /* Cleanup. SIDX cache pointer passed so the free leaves cached storage
     * intact for sink_feed/sink_swk reuse (matches existing
     * sink_environment_gpu.cc:261 idiom). External-CSR path frees only what
     * we staged (gnl offsets/neighbors/d_active); caller owns host CSR. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_actives);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
    if(args.external_csr != nullptr) {
        nlr_free_external_csr_gnl(&gnl);
    } else {
        gpu_ngb_list_free(&gnl, sidx);
    }
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
 * dispatch to the corresponding hook methods (default no-op). Two channels:
 *
 *   detector  : audit/debug — catches illegal kernel writes to imported
 *               ghosts. Trait `uses_ghost_write_detector`.
 *   writeback : physics state propagation — pre-kernel ghost-state snapshot
 *               + post-kernel reverse-comm of any j-side writes. Trait
 *               `uses_ghost_writeback`. Spec body is the per-flag #ifdef
 *               union of all enabled physics-flag writebacks.
 *
 * Each hook is gated on (a) the trait being true AND (b) whether the chosen
 * path imports ghosts (per-Spec audit decided imported-ghost-only for these
 * hooks; future Specs may differ).
 * ========================================================================== */

/* Trait detection: Spec::uses_ghost_write_detector / _ghost_writeback.
 * Absent ⇒ false. */
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

/* Optional label override: Spec::ghost_write_detector_name. Absent ⇒ runner
 * default labels the detector with Spec::loop_name. Used for the two
 * sink_env Specs whose detector labels predate the runner-template loop_name
 * convention and need to be preserved. */
template <typename Spec, typename = void>
struct nlr_has_ghost_write_detector_name : std::false_type {};
template <typename Spec>
struct nlr_has_ghost_write_detector_name<Spec, decltype((void)Spec::ghost_write_detector_name)>
    : std::true_type {};

template <typename Spec>
static constexpr const char *nlr_ghost_write_detector_name()
{
    if constexpr (nlr_has_ghost_write_detector_name<Spec>::value) {
        return Spec::ghost_write_detector_name;
    } else {
        return Spec::loop_name;
    }
}

/* Dispatch wrappers. Each gates on:
 *   (a) `uses_*` trait true (Spec opted in)
 *   (b) `nlr_path_uses_imported_ghosts(plan.path)` — for SinkEnv1Spec these
 *        hooks are imported-ghost-only. The path gate
 *        IS the policy; the hook trait is the Spec opt-in.
 *   (c) hook method exists (SFINAE) — present ⇒ Spec custom hook fires;
 *        absent ⇒ runner default fires (::ghost_write_detector_begin(name)
 *        / ::ghost_write_detector_end()). The dispatcher enforces that
 *        begin/end hook presence is symmetric: defining begin without end
 *        (or vice versa) is a compile error, not a half-default. */

template <typename Spec>
static void nlr_dispatch_ghost_write_detector_begin(const neighbor_loop_args& args,
                                                    const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_ghost_write_detector_v<Spec>()) {
        static_assert(nlr_has_hook_gwd_begin<Spec>::value == nlr_has_hook_gwd_end<Spec>::value,
                      "Spec must define both ghost_write_detector_begin/end or neither");
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_gwd_begin<Spec>::value) {
                Spec::ghost_write_detector_begin(args, plan);
            } else {
                ::ghost_write_detector_begin(nlr_ghost_write_detector_name<Spec>());
            }
        }
    }
}
template <typename Spec>
static void nlr_dispatch_ghost_write_detector_end(const neighbor_loop_args& args,
                                                  const NeighborLoopPlan& plan)
{
    if constexpr (nlr_uses_ghost_write_detector_v<Spec>()) {
        static_assert(nlr_has_hook_gwd_begin<Spec>::value == nlr_has_hook_gwd_end<Spec>::value,
                      "Spec must define both ghost_write_detector_begin/end or neither");
        if(nlr_path_uses_imported_ghosts(plan.path)) {
            if constexpr (nlr_has_hook_gwd_end<Spec>::value) {
                Spec::ghost_write_detector_end(args, plan);
            } else {
                ::ghost_write_detector_end();
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

    /* Active-source-in-pool contract (see neighbor_loop_runner.h). */
    static_assert(nlr_spec_satisfies_source_pool_contract_v<Spec>,
        "Cached-SIDX Spec must declare 'static constexpr bool mode_a_active_sources_in_sidx_pool' "
        "(true = active sources are SIDX-pool members; false = runner stages explicit P[].Pos). "
        "Prevents the stale gas-only-compact source-position bug for non-pool actives.");

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
     *   1. args.dispatch_override = A -> Mode A unconditionally.
     *   2. args.dispatch_override = B -> Mode B (local if NTask==1, remote else).
     *   3. Threshold dispatch: if (sum_active>0 && sum_active<=TS &&
     *      max_active<=TM) → Mode B; else Mode A. Hierarchy of TS/TM:
     *      parameterfile NeighborLoopModeBThreshold{Sum,Max} >
     *      Spec::modeb_threshold_{sum,max} constexpr defaults (64/64 today).
     *      Setting the parameterfile pair above every active count selects
     *      Mode B for the whole run; setting it to 0 selects Mode A.
     *
     * Note (active-epoch caveat): Mode B host-frozen actives[] are
     * NOT bit-equivalent to Mode A's device-staged post-neighbor-list-build
     * actives. The two modes therefore need not agree bit-for-bit on which
     * particles a call treats as active; consistency between them is a
     * property of the dispatch policy, not of this helper.
     */
    /* Dispatch priority: args.dispatch_override > adaptive threshold.
     * The args field is the corridor mode-decision hook (hydro_corridor.cc): when
     * a corridor consumer sets this to force coherent Mode A or Mode B across the
     * whole hydro corridor (cellcorrections/gradients/hydro_force), the per-call
     * override wins so corridor coherence is enforced, not advisory. */
    const NlrForceMode force_mode = args.dispatch_override;
    const bool force_a   = (force_mode == NlrForceMode::A);
    const bool force_b   = (force_mode == NlrForceMode::B);

    /* PHASE0 timing scaffolding. Cached env-gate; mid-run env
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
     * Skipped when the caller supplied a dispatch override (cheap path).
     * PHASE0 num_active_global is captured here when the threshold path
     * already did the Allreduce; on force paths an extra Allreduce is done
     * ONLY when phase0_on. */
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
    } else if(phase0_on || force_b) {
        /* Force path: dispatch logic skipped the Allreduce. Do it here for
         * phase0 num_active_global and for the forced-Mode-B size guard. */
        int local_act = args.num_active;
        int sum_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        phase0_sum_active = sum_act;
    }
    /* Globally-zero-active call: do NO neighbor work. phase0_sum_active is the
     * dispatch Allreduce of active particles (set on the threshold + force-B
     * paths), so it is identical on every rank -> this return is collective-
     * symmetric (all ranks return together, skipping the Mode-A ghost import /
     * writeback / cleanup as a matched set). NOT the banned local num_active==0
     * early return: the condition is GLOBAL. Without it, a zero-active call
     * falls to Mode A and fires a spurious ghost import with nothing to compute.
     * Not fired on the force-A path (phase0_sum_active stays -1 there). The
     * normal PHASE0/dispatch summary is intentionally skipped for such calls; a
     * distinct rank-0 marker (diag-gated) keeps them observable. */
    if(phase0_sum_active == 0) {
        if(phase0_on && ThisTask == 0) {
            std::printf("NLR_ZERO_ACTIVE_NOOP sp=%d caller=%s\n",
                        (int)All.NumCurrentTiStep, Spec::loop_name);
            std::fflush(stdout);
        }
        return;
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
            fprintf(stderr, "[NLR DISPATCH caller=%s path=%s (%s) NTask=%d local_active=%d]\n",
                    Spec::loop_name, nlr_path_label(plan.path), src,
                    NTask, args.num_active);
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
     * timer is 0 for Mode B paths (genuine 0 — the API isn't called).
     *
     * Caller-owned ghost pool (external_csr non-null): the caller imported
     * the pool the CSR indexes and owns its lifetime — skip the import (a
     * fresh import could renumber ghost slots under the CSR) and, below, the
     * cleanup. Contract enforced loudly here; see neighbor_loop_runner.h. */
    if(args.external_csr != nullptr && !nlr_path_uses_imported_ghosts(plan.path)) {
        /* external_csr is a Mode-A-only input; a Mode B dispatch with a CSR
         * supplied means the caller's dispatch_override and CSR provisioning
         * disagree — fail loudly rather than silently ignoring the CSR. */
        fprintf(stderr, "[neighbor_loop_runner rank=%d caller=%s] external_csr supplied but "
                "dispatch selected a Mode B path — caller contract violation.\n",
                ThisTask, Spec::loop_name);
        fflush(stderr);
        endrun(7312);
    }
    if(nlr_path_uses_imported_ghosts(plan.path)) {
        if(nlr_caller_owns_ghost_pool(args)) {
            /* At NTask>1 the CSR references imported ghost slots: require the
             * caller's pool + provenance to be LIVE, and args to have been
             * built AFTER the caller's import (num_total spans the ghosts).
             * At NTask==1 no pool exists (imports early-out) — nothing to
             * verify. ghost_get_num_ghosts()==0 is NOT a liveness signal
             * (also 0 for a live zero-ghost pool); ghost_pool_is_live() is. */
            if(NTask > 1) {
                const char *own_err = nullptr;
                if(!ghost_pool_is_live())                 own_err = "ghost pool not live";
                else if(ghost_get_home_rank()  == nullptr ||
                        ghost_get_home_index() == nullptr) own_err = "ghost provenance maps absent";
                else if(args.num_total != NumPart)         own_err = "args.num_total != NumPart (args built before caller's import?)";
                if(own_err != nullptr) {
                    fprintf(stderr, "[neighbor_loop_runner rank=%d caller=%s] caller-owned ghost-pool "
                            "contract violation: %s\n", ThisTask, Spec::loop_name, own_err);
                    fflush(stderr);
                    endrun(7313);
                }
            }
        } else {
            StageTimer t_prep(tim_ptr ? &tim_ptr->dt_prep_import : nullptr);
            gizmo_request_filtered_ghost_import_fresh(Spec::loop_name,
                                                       Spec::search_mode,
                                                       nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                                                       args.active_list,
                                                       args.num_active,
                                                       radii.data(),
                                                       args.ghost_safety_factor,
                                                       Spec::radius_policy,
                                                       nlr_spec_symmetric_j_radius_scale<Spec>(),
                                                       nlr_spec_supply_band_dominated<Spec>());
            /* Ghost import grew NumPart and may have realloc'd P/CellP. Refresh
             * the runner's data view; only paths that imported ghosts read this
             * extended view (Mode B paths use the original args via copy). */
            effective_args.num_total = NumPart;
            effective_args.P         = P;
            effective_args.CellP     = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;
        }
    }

    /* ---- Spec lifecycle hooks (begin) ---- */
    /* Mode A imported-ghost ordering invariant (two channels: detector +
     * writeback). The Spec's writeback_begin/_end body is the per-flag
     * #ifdef union of all enabled physics-flag writebacks for that loop:
     *
     *   request_filtered_ghost_import_fresh  (above)
     *   ghost_write_detector_begin           (this hook)
     *   ghost_writeback_begin                (this hook; per-flag union)
     *   <run_mode_a kernel>
     *   ghost_writeback_end                  (reverse order, below)
     *   ghost_write_detector_end
     *   ghost_exchange_cleanup               (below)
     *
     * Each dispatch helper checks both the Spec's `uses_*` trait AND the
     * path-imports-ghosts predicate; on Mode B paths the predicate is
     * false and all four hook calls compile to no-ops. */
    nlr_dispatch_ghost_write_detector_begin<Spec>(effective_args, plan);
    nlr_dispatch_ghost_writeback_begin<Spec>(effective_args, plan);

    /* ---- Path dispatch ---- */
    switch(plan.path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl:
            run_mode_a<Spec>(effective_args, radii.data(), tim_ptr);
            break;
        case NeighborLoopPlan::Path::ModeB_Local:
            run_mode_b_local<Spec>(effective_args, radii.data(), tim_ptr);
            break;
        case NeighborLoopPlan::Path::ModeB_Remote:
            run_mode_b_remote<Spec>(effective_args, radii.data(), tim_ptr);
            break;
    }

    /* ---- Spec lifecycle hooks (end, reverse order) ---- */
    nlr_dispatch_ghost_writeback_end<Spec>(effective_args, plan);
    nlr_dispatch_ghost_write_detector_end<Spec>(effective_args, plan);

    /* ---- Imported-ghost cleanup ---- */
    /* Caller-owned pools (external_csr) outlive this call — the caller tears
     * them down at the end of its span; skip cleanup here. See
     * neighbor_loop_runner.h. */
    if(nlr_path_uses_imported_ghosts(plan.path) && NTask > 1) {
        if(!nlr_caller_owns_ghost_pool(args)) {
            ghost_exchange_cleanup();
        }
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
            /* one-shot self-check at runner end (no loop); local PHASE0 emit + return
             * follow, no intervening collective -- soft bad-stop + fall through, drains
             * at the next phase-boundary poll. */
            endrun(81036);
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
 * run_neighbor_loop_iterative<Spec> — iterative entry point.
 *
 * Compile-time spec consistency:
 *   - IterControl == Iterative
 *   - IterScratch trivially copyable
 *   - max_iters >= 1; mode_a_csr_buffer_factor > 1.0
 *   - Spec::after_iter declared and callable (clean diagnostic naming the
 *     missing member if forgot)
 *
 * Runtime checks (fire BEFORE any arena/ghost/session touch):
 *   - num_subgroups >= 1 (caller short-circuits if globally empty)
 *   - num_subgroups > 1 requires Spec::SupportsSubgroups::value
 * ========================================================================== */

/* Helper: compile-time member detector for SupportsSubgroups (default false). */
template <typename Spec, typename = void>
struct nlr_supports_subgroups : std::false_type {};
template <typename Spec>
struct nlr_supports_subgroups<Spec, std::void_t<typename Spec::SupportsSubgroups>>
    : Spec::SupportsSubgroups {};

/* ============================================================================
 * NlrIterDriver<Spec> — constructor + destructor.
 * ========================================================================== */
template <typename Spec>
NlrIterDriver<Spec>::NlrIterDriver(const neighbor_loop_args_iterative& a,
                                   const typename Spec::CallScalars& s)
    : args(a), cs(s), iter_index(0),
      local_active_total(0), global_active_total(-1),
      local_active_per_sg (a.num_subgroups, 0),
      global_active_per_sg(a.num_subgroups, 0),
      scratch_uvm(a.num_subgroups, nullptr),
      accum_uvm  (a.num_subgroups, nullptr),
      radii_uvm  (a.num_subgroups, nullptr),
      active_set_uvm  (a.num_subgroups, nullptr),
      active_set_size (a.num_subgroups, 0),
      /* Mode A iterative cached-CSR state: zero-init per subgroup;
       * UVM allocations land lazily on first Mode A iter dispatch. */
      mode_a_cached_gnl       (a.num_subgroups, gpu_neighbor_list_t{}),
      mode_a_csr_offset_lookup(a.num_subgroups, nullptr),
      mode_a_csr_buffered_h   (a.num_subgroups, nullptr),
      mode_a_csr_valid        (a.num_subgroups, false),
      /* effective_args: copy of base slice; refreshed on Mode A ghost import. */
      effective_args(static_cast<const neighbor_loop_args&>(a))
{
    using IterScratch = typename Spec::IterScratch;
    using AccumData   = typename Spec::AccumData;

    /* Build a base neighbor_loop_args view per subgroup so we can call
     * Spec::search_radius for each (subgroup, slot, particle index) tuple.
     * search_radius is host-only and predates any drift / arena work. */
    for (int sg = 0; sg < args.num_subgroups; sg++) {
        const NlrSubgroup& sgr = args.subgroups[sg];
        const int n = sgr.num_active_local;
        active_set_size[sg] = n;
        if (n <= 0) continue;     /* leave pointers as nullptr; pair_kernel is no-op */

        scratch_uvm[sg]    = (IterScratch *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_iter_scratch", n * sizeof(IterScratch));
        accum_uvm  [sg]    = (AccumData   *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_accum_data", n * sizeof(AccumData));
        radii_uvm  [sg]    = (double      *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_radii", n * sizeof(double));
        active_set_uvm[sg] = (int         *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_active_set", n * sizeof(int));

        /* Byte-zero IterScratch ONCE — persists across iters. */
        std::memset(scratch_uvm[sg], 0, n * sizeof(IterScratch));
        /* AccumData NOT zeroed here; runner zeros via Spec::zero_accum at start of each iter. */

        /* Build a sub-args view for Spec::search_radius. Mirrors the per-
         * subgroup args the iterative dispatch will hand to lower-level
         * helpers (so search_radius signature stays a base neighbor_loop_args&). */
        neighbor_loop_args sub = args;          /* base slice; aux/CellP/etc. carry through */
        sub.active_list = sgr.active_indices;
        sub.num_active  = n;

        for (int slot = 0; slot < n; slot++) {
            radii_uvm[sg][slot]      = Spec::search_radius(sub, slot, sgr.active_indices[slot]);
            active_set_uvm[sg][slot] = slot;     /* {0..n-1} initial */
        }
        local_active_total += n;
    }
}

template <typename Spec>
NlrIterDriver<Spec>::~NlrIterDriver()
{
    /* DeviceContext cleanup: only fire if init actually
     * completed. Stubbed/aborted init paths leave ctx_initialized=false →
     * cleanup_device_context is NOT called, preventing free of unallocated
     * resources. Direct Spec call (NOT NlrDeviceContextCleanupGuard) — the
     * guard is RAII-only and wouldn't compose with conditional init.
     *
     * Ordering: cleanup_device_context fires
     * BEFORE the per-subgroup UVM frees below. Spec::cleanup_device_context
     * MUST only free resources it OWNS (e.g., UVM arrays it allocated in
     * populate_device_context). It MUST NOT assume the driver's per-subgroup
     * UVM arrays (scratch/accum/radii/active_set) remain valid AFTER cleanup
     * returns — those frees happen below. In practice cleanup hooks like
     * sink_feed's only free their own UVM, so this ordering is fine; the
     * note exists to flag the contract for future Specs. */
    if (ctx_initialized) {
        if constexpr (nlr_spec_has_cleanup_device_context_v<Spec>) {
            Spec::cleanup_device_context(args, ctx);
        }
    }

    /* Free Mode A cached CSR/lookup state by POINTER STATE (mode_a_csr_valid=false can mean "allocated but invalid,
     * pending rebuild" if the rebuild trigger fired but rebuild itself
     * hadn't completed yet; check pointers, not the flag).
     *
     * gpu_ngb_list_free passes the SIDX pointer so the step-persistent SIDX
     * cache is preserved across iterative calls (matches sink_env1/feed/swk
     * idiom). */
    {
        gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                             Spec::loop_name);
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (mode_a_cached_gnl[sg].offsets != nullptr ||
                mode_a_cached_gnl[sg].neighbors != nullptr) {
                gpu_ngb_list_free(&mode_a_cached_gnl[sg], sidx);
                mode_a_cached_gnl[sg] = gpu_neighbor_list_t{};
            }
            mode_a_csr_valid[sg] = false;
            if (mode_a_csr_offset_lookup[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(mode_a_csr_offset_lookup[sg]);
                mode_a_csr_offset_lookup[sg] = nullptr;
            }
            if (mode_a_csr_buffered_h[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(mode_a_csr_buffered_h[sg]);
                mode_a_csr_buffered_h[sg] = nullptr;
            }
        }
    }

    /* Arena cleanup ONCE per call (matches non-iter run_mode_a line 1718). */
    if (arena_acquired) {
        gpu_particles_arena_mark_clean_after_scatter(Spec::loop_name);
        arena_acquired = false;
    }

    /* Imported-ghost cleanup ONCE per call (matches non-iter
     * line 2094-2097). Only fires for Mode A multi-rank paths that imported.
     * Runner's Mode A imports an exact-query ghost pool sized to the iter's
     * actives+radii; the caller (e.g. density()) is responsible for any
     * downstream handoff pool it needs AFTER runner-return (post-finalize
     * fresh broad import — not "keep this exact-query pool alive"). */
    if (ghost_import_done && NTask > 1) {
        ghost_exchange_cleanup();
        ghost_import_done = false;
    }

    for (int sg = 0; sg < args.num_subgroups; sg++) {
        if (scratch_uvm[sg])    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(scratch_uvm[sg]);
        if (accum_uvm[sg])      Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(accum_uvm[sg]);
        if (radii_uvm[sg])      Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm[sg]);
        if (active_set_uvm[sg]) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(active_set_uvm[sg]);
    }
}

/* ============================================================================
 * Path-specific DeviceContext init.
 *
 * Mode B: bind to caller's args.P/CellP — Mode B reads owner-local per-particle
 *         state directly (lazy-drift contract). Static_asserts mirror run_mode_b_*
 *         (DeviceContext base + trivially copyable).
 * Mode A: bind to arena-resident gpu_particles_arena_P()/CellP() — Mode A's
 *         GPU NGL build + kernel walk operate on the arena copy. Body lands
 *         alongside the arena_acquire call site.
 * ========================================================================== */
template <typename Spec>
void NlrIterDriver<Spec>::initialize_device_context_mode_b()
{
    using DeviceCtx = typename Spec::DeviceContext;
    static_assert(std::is_base_of<NeighborLoopDeviceContextBase, DeviceCtx>::value,
                  "Spec::DeviceContext must publicly derive from NeighborLoopDeviceContextBase");
    static_assert(std::is_trivially_copyable<DeviceCtx>::value,
                  "Spec::DeviceContext must be trivially copyable; "
                  "captured by value into Kokkos device lambdas");

    /* Double-init guard: accidental call sequences (e.g., path mis-route calling
     * both inits) would silently double-populate / double-cleanup. Catch
     * loudly here. */
    if (ctx_initialized) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[NlrIterDriver<%s>::initialize_device_context_mode_b] FATAL: "
                "ctx_initialized=true on entry. Double-init would orphan resources.\n",
                Spec::loop_name);
            fflush(stderr);
        }
        /* Soft bad-stop + return: the valid first-init ctx is left intact
         * (no re-population). This function issues no MPI, so the early
         * return cannot desync a peer; drains at the next phase poll. */
        endrun(81209);
        return;
    }

    ctx.P         = args.P;
    ctx.CellP     = args.CellP;
    ctx.num_total = args.num_total;

    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(args, ctx);
    }
    ctx_initialized = true;
}

template <typename Spec>
void NlrIterDriver<Spec>::initialize_device_context_mode_a_after_arena()
{
    using DeviceCtx = typename Spec::DeviceContext;
    static_assert(std::is_base_of<NeighborLoopDeviceContextBase, DeviceCtx>::value,
                  "Spec::DeviceContext must publicly derive from NeighborLoopDeviceContextBase");
    static_assert(std::is_trivially_copyable<DeviceCtx>::value,
                  "Spec::DeviceContext must be trivially copyable; "
                  "captured by value into Kokkos device lambdas");

    /* Double-init guard. */
    if (ctx_initialized) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[NlrIterDriver<%s>::initialize_device_context_mode_a_after_arena] FATAL: "
                "ctx_initialized=true on entry. Double-init would orphan resources.\n",
                Spec::loop_name);
            fflush(stderr);
        }
        /* Soft bad-stop + return: the valid first-init ctx is left intact
         * (no re-population). This function issues no MPI, so the early
         * return cannot desync a peer; drains at the next phase poll. */
        endrun(81210);
        return;
    }
    /* Arena must already have been acquired. */
    if (!arena_acquired) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[NlrIterDriver<%s>::initialize_device_context_mode_a_after_arena] FATAL: "
                "arena not acquired. Call acquire_arena_and_init_ctx_mode_a() instead.\n",
                Spec::loop_name);
            fflush(stderr);
        }
        /* Symmetric lifecycle-contract violation. Soft bad-stop + return WITHOUT binding
         * ctx.P to an unacquired arena; the outer runner's nlr:iter_context_init poll
         * drains all ranks before any device dispatch. */
        endrun(81211); return;
    }

    /* Bind to arena-resident P_gpu / CellP_gpu. Use the driver's
     * effective_args: if ghost import ran above, these
     * point at the refreshed POST-IMPORT global P/CellP/NumPart; otherwise
     * they equal the original base args. CellP=NULL is legitimate (gas-free). */
    ctx.P         = gpu_particles_arena_P();
    ctx.CellP     = (effective_args.CellP != nullptr) ? gpu_particles_arena_CellP() : nullptr;
    ctx.num_total = effective_args.num_total;

    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(effective_args, ctx);
    }
    ctx_initialized = true;
}

template <typename Spec>
void NlrIterDriver<Spec>::acquire_arena_and_init_ctx_mode_a()
{
    if (arena_acquired) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[NlrIterDriver<%s>::acquire_arena_and_init_ctx_mode_a] FATAL: "
                "arena already acquired. Single-acquire-per-call contract violated.\n",
                Spec::loop_name);
            fflush(stderr);
        }
        /* Symmetric lifecycle-contract violation. Soft bad-stop + return WITHOUT a second
         * acquire (the first acquire's state stays intact); the outer runner's
         * nlr:iter_context_init poll drains all ranks before any device dispatch. */
        endrun(81212); return;
    }

    /* === (1) Imported-ghost prep ONCE per call ===
     * Mode A requires neighbors that live on peer ranks to be imported as
     * ghosts. Mirrors non-iter run_neighbor_loop lines 2037-2052.
     *
     * Multi-subgroup union semantics: mask = OR of all
     * subgroups' j_type_bitmask; actives = concatenation of all subgroups'
     * full active_indices (initial iter-0 set; later iters' compactions
     * narrow but don't grow); radii = matching concatenation oversized via
     * mode_a_csr_buffer_factor. Single-subgroup case: trivial length-1
     * union = original behavior. Multi-subgroup over-imports in cross-bm
     * cases (deliberate physics/perf tradeoff).
     *
     * On rebuild via rebuild_mode_a_arena_and_ctx_for_current_active_union,
     * the union covers only CURRENT compacted actives (post-Converged-removal)
     * so the rebuild import doesn't over-cover converged slots. */
    if (NTask > 1) {
        std::vector<int>    union_actives;
        std::vector<double> union_radii_oversized;
        unsigned int        mask_union = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            mask_union |= args.subgroups[sg].j_type_bitmask;
            const NlrSubgroup& sgr = args.subgroups[sg];
            for (int k = 0; k < sgr.num_active_local; k++) {
                union_actives.push_back(sgr.active_indices[k]);
                union_radii_oversized.push_back(radii_uvm[sg][k] * Spec::mode_a_csr_buffer_factor);
            }
        }
        const int union_n = (int)union_actives.size();
        gizmo_request_filtered_ghost_import_fresh(Spec::loop_name,
                                                   Spec::search_mode,
                                                   mask_union,
                                                   (union_n > 0) ? union_actives.data() : nullptr,
                                                   union_n,
                                                   (union_n > 0) ? union_radii_oversized.data() : nullptr,
                                                   args.ghost_safety_factor,
                                                   Spec::radius_policy,
                                                   nlr_spec_symmetric_j_radius_scale<Spec>(),
                                                   nlr_spec_supply_band_dominated<Spec>());
        ghost_import_done = true;

        /* Refresh effective_args from globals (ghost import grew NumPart and
         * may have realloc'd P/CellP). Matches non-iter line 2049-2051. */
        effective_args.num_total = NumPart;
        effective_args.P         = P;
        effective_args.CellP     = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;
    }

    /* === (2) Arena acquire ONCE per call, with refreshed args === */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(effective_args.num_total,
                                 effective_args.P, effective_args.CellP);
    arena_acquired = true;

    /* === (3) Bind ctx to arena-resident pointers + populate extended DeviceContext === */
    initialize_device_context_mode_a_after_arena();
}

/* ============================================================================
 * rebuild_mode_a_arena_and_ctx_for_current_active_union — self-sufficient
 * rebuild method (no parameters).
 *
 * Builds the union from driver's own subgroup state. Used by the Mode A
 * outer-iter pre-dispatch invalidation sweep (step 8). Invalidates ALL
 * subgroup CSR caches as part of the lifecycle (pitfall 2: cross-subgroup
 * neighbor indices become stale after arena re-acquire).
 *
 * Union semantics (pitfall 4): mask = OR of all globally-active subgroups'
 * j_type_bitmask; active list = concatenation of current compacted actives
 * across globally-active subgroups; radii = matching concatenation of
 * radii_uvm[sg][slot] * mode_a_csr_buffer_factor. Over-imports in
 * multi-bm cases (each active gets ghosts of all union types instead of
 * just its own mask). Cosmological FIRE typical = single bm-group = no
 * over-import. Documented as deliberate physics/perf tradeoff.
 *
 * Pre-condition: drv.global_active_per_sg has been computed (set by the
 * outer iter loop's per-iter Allreduce). On iter 0 it may all be 0 (not
 * yet Allreduced); caller should NOT invoke this method at iter 0 — the
 * iter-0 ghost import + arena acquire happens in
 * acquire_arena_and_init_ctx_mode_a per the original path.
 * ========================================================================== */
template <typename Spec>
void NlrIterDriver<Spec>::rebuild_mode_a_arena_and_ctx_for_current_active_union()
{
    if (NTask <= 1) {
        /* Single rank: no ghost pool to manage. Just invalidate all CSR
         * caches; per-subgroup dispatch will rebuild local CSRs on first
         * access against the unchanged arena/pool. */
        gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                             Spec::loop_name);
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (mode_a_cached_gnl[sg].offsets != nullptr ||
                mode_a_cached_gnl[sg].neighbors != nullptr) {
                gpu_ngb_list_free(&mode_a_cached_gnl[sg], sidx);
                mode_a_cached_gnl[sg] = gpu_neighbor_list_t{};
            }
            mode_a_csr_valid[sg] = false;
            if (mode_a_csr_offset_lookup[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(mode_a_csr_offset_lookup[sg]);
                mode_a_csr_offset_lookup[sg] = nullptr;
            }
            if (mode_a_csr_buffered_h[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(mode_a_csr_buffered_h[sg]);
                mode_a_csr_buffered_h[sg] = nullptr;
            }
        }
        return;
    }

    /* Build union from current active_set across globally-active subgroups. */
    std::vector<int>    union_actives;
    std::vector<double> union_radii_oversized;
    unsigned int        mask_union = 0;
    for (int sg = 0; sg < args.num_subgroups; sg++) {
        if (global_active_per_sg[sg] <= 0) continue;     /* skip globally-converged */
        mask_union |= args.subgroups[sg].j_type_bitmask;
        const NlrSubgroup& sgr = args.subgroups[sg];
        for (int k = 0; k < active_set_size[sg]; k++) {
            int slot = active_set_uvm[sg][k];
            union_actives.push_back(sgr.active_indices[slot]);
            union_radii_oversized.push_back(radii_uvm[sg][slot] * Spec::mode_a_csr_buffer_factor);
        }
    }

    /* === (0) Invalidate ALL subgroup CSR caches (avoids cross-subgroup
     * staleness after arena teardown). Free by pointer state. */
    {
        gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                             Spec::loop_name);
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (mode_a_cached_gnl[sg].offsets != nullptr ||
                mode_a_cached_gnl[sg].neighbors != nullptr) {
                gpu_ngb_list_free(&mode_a_cached_gnl[sg], sidx);
                mode_a_cached_gnl[sg] = gpu_neighbor_list_t{};
            }
            mode_a_csr_valid[sg] = false;
            if (mode_a_csr_offset_lookup[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(mode_a_csr_offset_lookup[sg]);
                mode_a_csr_offset_lookup[sg] = nullptr;
            }
            if (mode_a_csr_buffered_h[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(mode_a_csr_buffered_h[sg]);
                mode_a_csr_buffered_h[sg] = nullptr;
            }
        }
    }

    /* === (1) cleanup_device_context for extended Specs === */
    if (ctx_initialized) {
        if constexpr (nlr_spec_has_cleanup_device_context_v<Spec>) {
            Spec::cleanup_device_context(effective_args, ctx);
        }
        ctx_initialized = false;
    }

    /* === (2) END current arena view BEFORE mutating the global ghost pool === */
    if (arena_acquired) {
        gpu_particles_arena_mark_clean_after_scatter(Spec::loop_name);
        arena_acquired = false;
    }

    /* === (3) Destroy current ghost pool === */
    if (ghost_import_done) {
        ghost_exchange_cleanup();
        ghost_import_done = false;
    }

    /* === (4) Reimport with UNION radii / actives / mask (deliberate over-import,
     * an accepted tradeoff) === */
    {
        const int union_n = (int)union_actives.size();
        gizmo_request_filtered_ghost_import_fresh(Spec::loop_name,
                                                   Spec::search_mode,
                                                   mask_union,
                                                   (union_n > 0) ? union_actives.data() : nullptr,
                                                   union_n,
                                                   (union_n > 0) ? union_radii_oversized.data() : nullptr,
                                                   args.ghost_safety_factor,
                                                   Spec::radius_policy,
                                                   nlr_spec_symmetric_j_radius_scale<Spec>(),
                                                   nlr_spec_supply_band_dominated<Spec>());
        ghost_import_done = true;
    }

    /* === (5) Refresh effective_args from post-import globals === */
    effective_args.num_total = NumPart;
    effective_args.P         = P;
    effective_args.CellP     = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;

    /* === (6) Fresh arena view of new pool === */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(effective_args.num_total,
                                 effective_args.P, effective_args.CellP);
    arena_acquired = true;

    /* === (7) Rebind ctx + (8) populate extended ctx for NEW pool === */
    initialize_device_context_mode_a_after_arena();

    /* Do NOT re-fire reset_per_iter_device_context
     * here. The outer iter loop's step (a-pre) calls this method BEFORE
     * step (a0), which fires the reset hook exactly once per outer iter
     * over the freshly-populated ctx. Re-firing here would cause a
     * double-reset on rebuild iters and violate the "fires once per outer
     * iter" contract used by the synthetic harness validation. */
}

/* ============================================================================
 * Per-iter, per-subgroup dispatch helpers (Mode B local only).
 *
 * Each helper takes the driver, a subgroup index, and runs ONE iteration's
 * pair_kernel walk for that subgroup writing into driver-owned accum_uvm.
 * apply_active_writeback is NOT called here — that's final-only after the
 * outer iter loop completes.
 *
 * The Mode B local helper composes existing lower-level helpers
 * (build_self_actives_host_pre_drift / collect_candidates_pre_drift /
 * lazy_drift_candidates / evaluate_pairs_post_drift). SSOT preserved with
 * existing run_mode_b_local — same helper chain, just driver-owned output
 * buffer instead of stack vector.
 *
 * Mode A iter lands via the sibling nlr_iter_dispatch_subgroup_mode_a helper.
 *
 * DEVICE CONTEXT LIFETIME:
 * The dispatch helper uses driver-owned drv.ctx (populated once via
 * NlrIterDriver::initialize_device_context_mode_b at iter-0 entry, cleaned
 * once in destructor via cleanup_device_context if extended). Per-iter
 * reset is handled at the outer iter loop via Spec::reset_per_iter_device_context.
 * No per-iter / per-subgroup ctx rebuild here; the older "rebuilds DeviceCtx
 * per-iter + per-subgroup" hack is gone.
 * ========================================================================== */
/* ============================================================================
 * nlr_iter_dispatch_subgroup_mode_b_remote<Spec>.
 *
 * Per-iter Mode B REMOTE dispatch for one subgroup. Composes the same
 * extracted helper (mode_b_remote_evaluate_into_buffer<Spec, false>) the
 * non-iterative wrapper uses — SSOT preserved. The driver-owned compacted
 * AccumData buffer is the only difference. apply_active_writeback NOT
 * called here — that's final-only after the outer iter loop.
 */
template <typename Spec>
static void nlr_iter_dispatch_subgroup_mode_b_remote(NlrIterDriver<Spec>& drv, int sg)
{
    using AccumData = typename Spec::AccumData;

    const NlrSubgroup& sgr = drv.args.subgroups[sg];
    const int n_compacted  = drv.active_set_size[sg];
    /* Note: even if n_compacted == 0 on this rank, the helper MUST be
     * entered because Mode B remote uses collectives (alltoallv exchange
     * of queries / replies). Empty rank participates with N=0. */

    /* Compact active particle indices + radii for the still-active slots. */
    std::vector<int>    active_particle_indices(n_compacted);
    std::vector<double> radii_compacted(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        active_particle_indices[k] = sgr.active_indices[slot];
        radii_compacted[k]         = drv.radii_uvm[sg][slot];
    }

    neighbor_loop_args sub = drv.args;
    sub.active_list = (n_compacted > 0) ? active_particle_indices.data() : nullptr;
    sub.num_active  = n_compacted;

    /* Driver-owned compacted AccumData buffer. Helper writes into this;
     * we scatter back into driver.accum_uvm[sg][slot] for active slots
     * after the helper returns. Explicit Spec::zero_accum per slot:
     * defensive against future evaluate_pairs
     * variants + makes the per-iter AccumData zero contract explicit at
     * the caller level. */
    std::vector<AccumData> accums_compacted(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        Spec::zero_accum(accums_compacted[k]);
    }

    mode_b_remote_evaluate_into_buffer<Spec>(
        sub,
        radii_compacted.data(),
        drv.cs,                          /* driver-owned CallScalars */
        drv.ctx,                         /* driver-owned DeviceContext */
        (unsigned int)sgr.j_type_bitmask, /* per-subgroup mask */
        (n_compacted > 0) ? accums_compacted.data() : nullptr,
        /*tim=*/nullptr);

    /* Scatter compacted accums back into driver-owned per-slot accum_uvm.
     * Slots NOT in active_set_uvm keep their last-evaluated value (the
     * v4.4 invariant: converged slots' final accum persists for
     * apply_active_writeback). */
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        drv.accum_uvm[sg][slot] = accums_compacted[k];
    }
}

/* ============================================================================
 * nlr_iter_dispatch_subgroup_mode_a<Spec>(drv, sg).
 *
 * Per-iter Mode A dispatch for one subgroup. Cached oversized CSR + rebuild
 * trigger preserve legacy density_gpu.cc:140-220 session semantics.
 *
 * Lifecycle assumptions (set up by run_neighbor_loop_iterative + driver):
 *   - drv.acquire_arena_and_init_ctx_mode_a() ran ONCE before any subgroup
 *     dispatch. drv.ctx.P/CellP point at arena-resident P_gpu/CellP_gpu.
 *   - drv.arena_acquired = true; drv.ctx_initialized = true.
 *
 * Per-iter sequence:
 *   1. If !drv.mode_a_csr_valid[sg]: rebuild CSR with current compacted-
 *      active radii × Spec::mode_a_csr_buffer_factor. Frees old gnl/lookup
 *      (passing SIDX). Populates csr_offset_lookup + csr_buffered_h keyed
 *      on subgroup-slot-at-build-time.
 *   2. ghost_write_detector_begin + ghost_writeback_begin (gated on Spec
 *      traits + path → fires for Mode A iter + uses_ghost_writeback=true,
 *      e.g. ags_density wakeup; no-op for non-j-write Specs / harness).
 *   3. Stage d_actives via Spec::load_active (compacted indices).
 *   4. Pair-kernel launch over compacted active set, writing into a
 *      compacted UVM accums buffer.
 *   5. ghost_writeback_end + ghost_write_detector_end.
 *   6. Scatter compacted accums into driver-owned accum_uvm[sg][slot].
 * NO apply_active_writeback — final-only at driver level (post-iter-loop).
 *
 * CSR row-key invariant: rows in
 * cached_gnl are in COMPACTED-AT-BUILD-TIME order. csr_offset_lookup[sg]
 * maps subgroup-slot → compacted-build-time row index. Active-set
 * compaction in later iters does NOT re-key rows; converged slots simply
 * aren't walked. Rebuild creates a fresh lookup and discards the old one.
 *
 * Buffer-exceedance rebuild trigger: handled at outer-iter level AFTER
 * after_iter (post-radii mutation on AdjustRadius). Helper just consumes
 * the csr_valid[sg] flag.
 * ========================================================================== */
template <typename Spec>
static void nlr_iter_dispatch_subgroup_mode_a(NlrIterDriver<Spec>& drv, int sg)
{
    using ActiveData   = typename Spec::ActiveData;
    using AccumData    = typename Spec::AccumData;
    using ScatterData  = typename Spec::ScatterData;
    using NeighborData = typename Spec::NeighborData;

    const NlrSubgroup& sgr = drv.args.subgroups[sg];
    const int n_compacted  = drv.active_set_size[sg];

    /* Collective-symmetry: ghost_writeback_begin/end MUST fire on every rank
     * for uses_ghost_writeback Specs — even
     * empty-actives ranks, otherwise reverse-comm deadlocks. Hoist the
     * dispatch hooks outside the n_compacted<=0 short-circuit. The
     * kernel-side work between begin/end is gated on n_compacted > 0. */
    std::vector<int> active_particle_indices_iter(n_compacted > 0 ? n_compacted : 0);
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        active_particle_indices_iter[k] = sgr.active_indices[slot];
    }
    NeighborLoopPlan plan;
    plan.path              = NeighborLoopPlan::Path::ModeA_GpuNgl;
    plan.num_active_global = drv.global_active_total;
    neighbor_loop_args sub = drv.effective_args;
    sub.active_list = (n_compacted > 0) ? active_particle_indices_iter.data() : nullptr;
    sub.num_active  = n_compacted;

    /* ===== (1) Build / rebuild CSR if invalid =====
     * The reimport/re-arena lifecycle lives
     * EXCLUSIVELY in the outer pre-dispatch union rebuild (run_neighbor_loop_iterative
     * step a-pre), which uses the union mask across all globally-active
     * subgroups. Inside this helper, when !mode_a_csr_valid[sg], the ghost
     * pool / arena / ctx are already correct (handled by the outer sweep);
     * we just need to free + rebuild the local CSR for this subgroup. */
    gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                         Spec::loop_name);
    if (!drv.mode_a_csr_valid[sg] && n_compacted > 0) {
        /* Free old cached CSR / lookup state by POINTER STATE (mode_a_csr_valid=false can mean "allocated
         * but invalid, pending rebuild"; check pointers, not flag). */
        if (drv.mode_a_cached_gnl[sg].offsets != nullptr ||
            drv.mode_a_cached_gnl[sg].neighbors != nullptr) {
            gpu_ngb_list_free(&drv.mode_a_cached_gnl[sg], sidx);
            drv.mode_a_cached_gnl[sg] = gpu_neighbor_list_t{};
        }
        if (drv.mode_a_csr_offset_lookup[sg]) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(drv.mode_a_csr_offset_lookup[sg]);
            drv.mode_a_csr_offset_lookup[sg] = nullptr;
        }
        if (drv.mode_a_csr_buffered_h[sg]) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(drv.mode_a_csr_buffered_h[sg]);
            drv.mode_a_csr_buffered_h[sg] = nullptr;
        }

        /* Build oversized CSR with CURRENT active set (compacted) and
         * oversized radii × buffer_factor. */
        std::vector<int>    active_particle_indices_host_build(n_compacted);
        std::vector<double> radii_buffered_host_build(n_compacted);
        for (int k = 0; k < n_compacted; k++) {
            int slot = drv.active_set_uvm[sg][k];
            active_particle_indices_host_build[k] = sgr.active_indices[slot];
            radii_buffered_host_build[k]          = drv.radii_uvm[sg][slot] * Spec::mode_a_csr_buffer_factor;
        }
        double *radii_oversized_uvm = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_radii",
            n_compacted * sizeof(double));
        for (int k = 0; k < n_compacted; k++) radii_oversized_uvm[k] = radii_buffered_host_build[k];

        /* Active-source-in-pool contract: stage explicit P[active_i].Pos for specs
         * whose active sources may be non-pool (else nullptr keeps the compact
         * fast-path). See neighbor_loop_runner.h. Radii are already explicit. */
        std::vector<double> _nlr_srcpos_storage;
        const double* _nlr_srcpos = nlr_stage_explicit_source_positions<Spec>(
            drv.ctx.P, active_particle_indices_host_build.data(), n_compacted, _nlr_srcpos_storage);
        gpu_ngb_list_build(drv.ctx.P, drv.ctx.num_total,
                           active_particle_indices_host_build.data(), n_compacted,
                           Spec::search_mode,
                           (int)sgr.j_type_bitmask,
                           &drv.mode_a_cached_gnl[sg], sidx,
                           1.0, radii_oversized_uvm, _nlr_srcpos, Spec::loop_name,
                           nlr_spec_symmetric_j_radius_scale<Spec>(),
                           Spec::radius_policy);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_oversized_uvm);

        /* Allocate csr_offset_lookup + csr_buffered_h sized to subgroup max.
         * Lookup keyed on subgroup-slot (invariant):
         *   csr_offset_lookup[slot] = build-time row index (0..n_compacted-1),
         *                              or -1 if slot wasn't in build-time set
         *                              (shouldn't happen — rebuild covers all
         *                              current actives, and converged slots
         *                              never re-enter active_set). */
        const int n_max = sgr.num_active_local;
        drv.mode_a_csr_offset_lookup[sg] = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_csr_lookup",
            n_max * sizeof(int));
        drv.mode_a_csr_buffered_h[sg]    = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_csr_h",
            n_max * sizeof(double));
        for (int s = 0; s < n_max; s++) {
            drv.mode_a_csr_offset_lookup[sg][s] = -1;
            drv.mode_a_csr_buffered_h[sg][s]    = 0.0;
        }
        for (int k = 0; k < n_compacted; k++) {
            int slot = drv.active_set_uvm[sg][k];
            drv.mode_a_csr_offset_lookup[sg][slot] = k;
            drv.mode_a_csr_buffered_h[sg][slot]    = radii_buffered_host_build[k];
        }
        drv.mode_a_csr_valid[sg] = true;
    }

    /* ===== (2) Ghost write detector + writeback begin =====
     * UNCONDITIONAL: fires on every rank for
     * uses_ghost_writeback Specs, INCLUDING empty-actives ranks, to keep
     * MPI reverse-comm collectives in lockstep with non-empty ranks.
     * Plan.path == ModeA_GpuNgl gates the dispatch hooks via
     * nlr_path_uses_imported_ghosts; uses effective_args (post-import). */
    nlr_dispatch_ghost_write_detector_begin<Spec>(sub, plan);
    nlr_dispatch_ghost_writeback_begin<Spec>(sub, plan);

    if (n_compacted > 0) {
        /* ===== (3) Stage d_actives + (4) pair-kernel launch =====
         * CSR ROW LOOKUP USAGE:
         *   slot = active_set[k]
         *   row  = csr_offset_lookup[slot]    (build-time row index)
         *   i    = d_active[row]              (particle index at build-time)
         *   h    = radii_uvm[slot]            (current radius — kernel filter)
         * CSR build uses drv.ctx.num_total (= post-import effective num_total).
         * Walk gnl.offsets[row]..offsets[row+1]. */
        ActiveData *d_actives = (ActiveData *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_active_data",
            n_compacted * sizeof(ActiveData));
        AccumData *d_accums = (AccumData *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("modea_accum_data",
            n_compacted * sizeof(AccumData));
        {
            auto cs_ref = drv.cs;
            const typename Spec::DeviceContext dctx_local = drv.ctx;
            int    *active_set_arr = drv.active_set_uvm[sg];
            int    *csr_lookup     = drv.mode_a_csr_offset_lookup[sg];
            double *radii_arr      = drv.radii_uvm[sg];
            int     *d_active_arr  = drv.mode_a_cached_gnl[sg].d_active;
            int64_t *offsets       = drv.mode_a_cached_gnl[sg].offsets;
            int     *neighbors     = drv.mode_a_cached_gnl[sg].neighbors;

            gizmo_gpu_kernel_launch("nlr_iter_stage_active", n_compacted, KOKKOS_LAMBDA(int k) {
                int slot = active_set_arr[k];
                int row  = csr_lookup[slot];
                int i    = d_active_arr[row];
                double h = radii_arr[slot];
                d_actives[k] = Spec::load_active(dctx_local, slot, i, h, cs_ref);
            });

            const double t_pair_kernel_start = my_second();
            gizmo_gpu_kernel_launch(Spec::loop_name, n_compacted, KOKKOS_LAMBDA(int k) {
                int slot = active_set_arr[k];
                int row  = csr_lookup[slot];
                Spec::zero_accum(d_accums[k]);
                const ActiveData& a = d_actives[k];
                ScatterData s{};
                int64_t start = offsets[row], end = offsets[row + 1];
                for (int64_t nn = start; nn < end; nn++) {
                    int j = neighbors[nn];
                    IdentitySidecar id{};
                    NeighborData nb = Spec::load_neighbor(dctx_local, j, id, a);
                    Spec::pair_kernel(a, nb, d_accums[k], s);
                }
            });
            cpu_charge_child(CPU_PAIR_KERNEL, timediff(t_pair_kernel_start, my_second()));
        }

        /* ===== (6) Scatter compacted accums into driver accum_uvm ===== */
        for (int k = 0; k < n_compacted; k++) {
            int slot = drv.active_set_uvm[sg][k];
            drv.accum_uvm[sg][slot] = d_accums[k];
        }

        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_actives);
    }

    /* ===== (5) Ghost writeback end + detector end (unconditional) ===== */
    nlr_dispatch_ghost_writeback_end<Spec>(sub, plan);
    nlr_dispatch_ghost_write_detector_end<Spec>(sub, plan);
}

template <typename Spec>
static void nlr_iter_dispatch_subgroup_mode_b_local(NlrIterDriver<Spec>& drv, int sg)
{
    using ActiveData  = typename Spec::ActiveData;
    using AccumData   = typename Spec::AccumData;

    const NlrSubgroup& sgr = drv.args.subgroups[sg];
    const int n_compacted  = drv.active_set_size[sg];
    if (n_compacted <= 0) return;     /* All actives in this subgroup converged. */

    /* Build a base sub-args restricted to the still-active slots in this subgroup.
     * This is per-iter — the active_list/num_active reflect the COMPACTED set
     * after Converged compaction from prior iters. */
    std::vector<int> active_particle_indices(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        active_particle_indices[k] = sgr.active_indices[slot];
    }
    neighbor_loop_args sub = drv.args;
    sub.active_list = active_particle_indices.data();
    sub.num_active  = n_compacted;

    /* Per-active radii in compacted order (helpers expect contiguous radii array). */
    std::vector<double> radii_compacted(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        radii_compacted[k] = drv.radii_uvm[sg][drv.active_set_uvm[sg][k]];
    }

    /* DeviceContext: driver-owned. Populated once at iter-0 by
     * NlrIterDriver::initialize_device_context_mode_b(). Per-iter reset hook
     * (if Spec declares it) fires at outer iter level, before this dispatch. */

    /* Stage actives + zero accums host-side (compacted). */
    std::vector<ActiveData> actives_compacted(n_compacted);
    build_self_actives_host_pre_drift<Spec>(sub, drv.ctx, radii_compacted.data(),
                                              drv.cs, actives_compacted.data());

    std::vector<AccumData> accums_compacted(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        Spec::zero_accum(accums_compacted[k]);     /* per-iter zero */
    }

    /* Collect → drift → evaluate (same helper chain as run_mode_b_local).
     * Per-subgroup mask threaded via sgr.j_type_bitmask
     * for multi-subgroup walker support. */
    std::vector<std::vector<int>> cand_modeB;
    collect_candidates_pre_drift<Spec>(sub, radii_compacted.data(),
                                         (unsigned int)sgr.j_type_bitmask,
                                         DispatchPath::ModeB_HostWalker, cand_modeB);
    lazy_drift_candidates<Spec>(cand_modeB);
    evaluate_pairs_post_drift<Spec>(drv.ctx, actives_compacted.data(), n_compacted,
                                      cand_modeB, accums_compacted.data(), EvalOMPPolicy::AllowProduction);

    /* Scatter compacted accums back into driver-owned per-slot accum_uvm.
     * Slots NOT in active_set_uvm keep their stale values (will not be
     * read this iter — after_iter only fires on still-active slots). */
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        drv.accum_uvm[sg][slot] = accums_compacted[k];
    }
}


/* ============================================================================
 * Default on_max_iter_exceeded — runner-supplied bad-stop on max iteration.
 * endrun(1155) is now a soft controlled-stop (post Stage-1 macro flip): the
 * run flags + proceeds with un-converged radii to the next phase-boundary
 * poll, then finalizes cleanly (no MPI_Abort). Matches legacy
 * hydro/density.cc:602 and gravity/ags_rkern.cc:414 stop policy.
 * Specs override via `static void on_max_iter_exceeded(const NlrIterDriver<Spec>&);`
 * only when port's legacy policy differs and Phil approved (rare).
 * ========================================================================== */
template <typename Spec>
static void nlr_default_on_max_iter_exceeded(const NlrIterDriver<Spec>& drv)
{
    if (ThisTask == 0) {
        fprintf(stderr,
            "[%s] FATAL: failed to converge in %d iterations (max_iters=%d). "
            "global_active_total=%d remained.\n",
            Spec::loop_name, drv.iter_index, Spec::max_iters, drv.global_active_total);
        fflush(stderr);
    }
    endrun(1155);
}

/* ============================================================================
 * run_neighbor_loop_iterative<Spec> — body (step 2b).
 * ========================================================================== */
template <typename Spec>
void run_neighbor_loop_iterative(const neighbor_loop_args_iterative& args)
{

    /* ===== Compile-time spec consistency ===== */
    static_assert(std::is_same_v<typename Spec::IterControl, Iterative>,
                  "run_neighbor_loop_iterative requires Spec::IterControl = Iterative. "
                  "NotIterative Specs must call run_neighbor_loop instead.");
    static_assert(nlr_spec_has_after_iter_v<Spec>,
                  "Iterative Spec is missing `static IterResult after_iter(const AfterIterContext<Spec>&, const AccumData&);`. "
                  "Required when IterControl = Iterative.");
    static_assert(std::is_trivially_copyable_v<typename Spec::IterScratch>,
                  "Spec::IterScratch must be std::is_trivially_copyable_v. "
                  "Use `using IterScratch = NoIterScratch;` for the empty form.");
    static_assert(Spec::max_iters >= 1,
                  "Spec::max_iters must be >= 1.");
    /* Active-source-in-pool contract (see neighbor_loop_runner.h). */
    static_assert(nlr_spec_satisfies_source_pool_contract_v<Spec>,
        "Cached-SIDX Spec must declare 'static constexpr bool mode_a_active_sources_in_sidx_pool' "
        "(true = active sources are SIDX-pool members; false = runner stages explicit P[].Pos). "
        "Prevents the stale gas-only-compact source-position bug for non-pool actives.");
    static_assert(Spec::mode_a_csr_buffer_factor > 1.0,
                  "Spec::mode_a_csr_buffer_factor must be > 1.0 "
                  "(legacy DENSITY_H_BUFFER_FACTOR = 1.3).");
    /* TRAP-5 carry-forward: same trivially-copyable
     * checks as run_neighbor_loop. Don't let iterative Specs bypass TRAP 5. */
    static_assert(std::is_trivially_copyable_v<typename Spec::CallScalars>,
                  "Spec::CallScalars must be trivially copyable (lambda-captured by value).");
    static_assert(std::is_trivially_copyable_v<typename Spec::ActiveData>,
                  "Spec::ActiveData must be trivially copyable (UVM-staged).");
    static_assert(std::is_trivially_copyable_v<typename Spec::NeighborData>,
                  "Spec::NeighborData must be trivially copyable (built per-pair on device).");
    static_assert(std::is_trivially_copyable_v<typename Spec::AccumData>,
                  "Spec::AccumData must be trivially copyable (UVM-staged).");

    /* ===== Runtime checks BEFORE any state touch ===== */
    if (args.num_subgroups < 1) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[run_neighbor_loop_iterative<%s>] FATAL: num_subgroups=%d < 1. "
                "Caller must short-circuit when global active total is 0.\n",
                Spec::loop_name, args.num_subgroups);
            fflush(stderr);
        }
        endrun(81200);
        return;   /* before any state touch; symmetric caller-contract failure -> graceful return, drains at next poll */
    }
    if (args.num_subgroups > 1 && !nlr_supports_subgroups<Spec>::value) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[run_neighbor_loop_iterative<%s>] FATAL: num_subgroups=%d > 1 "
                "but Spec did not declare `using SupportsSubgroups = std::true_type;` (TRAP 9).\n",
                Spec::loop_name, args.num_subgroups);
            fflush(stderr);
        }
        endrun(81201);
        return;   /* before any state touch; symmetric caller-contract failure -> graceful return, drains at next poll */
    }
    /* Multi-subgroup walker:
     *   - sg.j_type_bitmask threaded through Mode B local + remote collectors
     *     and Mode A NGL build per-subgroup.
     *   - Mode A multi-subgroup ghost import uses union semantics (mask
     *     OR + concatenation of actives/radii across subgroups). Documented
     *     as deliberate over-import for cross-bm cases.
     *   - Per-subgroup global activity tracking via global_active_per_sg[].
     *   - Pre-dispatch invalidation sweep rebuilds CSR caches once per iter
     *     when any subgroup invalidated.
     *   - The caller must fill subgroups[] from the global_bm_presence union
     *     with identical ordering on every rank, and must place each particle
     *     in at most one subgroup.
     * Per-Spec opt-in still REQUIRED via `using SupportsSubgroups = std::true_type;`
     * (runtime abort 81201 above catches missing trait). */


    /* ===== Path selection at iter 0 (FIXED for whole call) =====
     * Integrates with the canonical override / threshold dispatch
     * (mirrors the selection in run_neighbor_loop). Path is held fixed
     * across all iterations of one iterative call.
     *
     * num_active for threshold uses the UNION across all subgroups (base
     * args.num_active per the doc convention). This selection is single-subgroup-only;
     * multi-subgroup walker mask-threading is handled separately. */
    /* Dispatch priority: args.dispatch_override > adaptive threshold
     * (mirrors the non-iterative site). Iterative Specs (density, ags_density,
     * mechfb, etc.) generally won't set dispatch_override since the
     * corridor mode flows through cellcorrections/gradients/hydro_force; preserved
     * here for completeness and future use. */
    const NlrForceMode force_mode = args.dispatch_override;
    DispatchPath path;
    int forced_modeb_global_active = -1;
    /* Global active-particle sum across all ranks (from the dispatch Allreduce);
     * -1 = not computed (force-A cheap path). Used for the globally-zero-active
     * no-op below. */
    int global_active_sum = -1;
    if (force_mode == NlrForceMode::A) {
        path = DispatchPath::ModeA_GPU_NGL;
    } else if (force_mode == NlrForceMode::B) {
        int local_act = args.num_active;
        MPI_Allreduce(&local_act, &forced_modeb_global_active, 1,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        global_active_sum = forced_modeb_global_active;
        path = DispatchPath::ModeB_HostWalker;
    } else {
        /* Threshold dispatch: Allreduce sum + max of base args.num_active
         * (= union across subgroups). */
        int local_act = args.num_active;
        int sum_act = 0, max_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_act, &max_act, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        global_active_sum = sum_act;
        const int spec_default_sum = nlr_spec_threshold_sum<Spec>(64);
        const int spec_default_max = nlr_spec_threshold_max<Spec>(64);
        const int TS = gizmo_nlr_modeb_threshold_sum_for(Spec::loop_name, spec_default_sum);
        const int TM = gizmo_nlr_modeb_threshold_max_for(Spec::loop_name, spec_default_max);
        bool select_mode_b = (sum_act > 0) && (sum_act <= TS) && (max_act <= TM);
        path = select_mode_b ? DispatchPath::ModeB_HostWalker : DispatchPath::ModeA_GPU_NGL;
    }

    /* Globally-zero-active call: do NO neighbor work. global_active_sum comes
     * from the dispatch Allreduce, so it is identical on every rank -> this
     * return is collective-symmetric (all ranks return together, skipping the
     * ghost import / writeback / cleanup as a matched set). This is NOT the
     * banned local num_active==0 early return: the condition is GLOBAL. Without
     * it, a zero-active call falls to Mode A (sum_act>0 gate fails) and fires a
     * spurious request-driven ghost import with nothing to compute. Not fired on
     * the force-A path (global_active_sum stays -1 there). The normal
     * PHASE0/dispatch summary is intentionally skipped for such calls; a distinct
     * rank-0 marker (diag-gated) keeps them observable. */
    if (global_active_sum == 0) {
        if (gizmo_nlr_phase0_diag_enabled() && ThisTask == 0) {
            std::printf("NLR_ZERO_ACTIVE_NOOP sp=%d caller=%s\n",
                        (int)All.NumCurrentTiStep, Spec::loop_name);
            std::fflush(stdout);
        }
        return;
    }

    if(gizmo_nlr_dispatch_trace_enabled()) {
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0) {
            const char *src =
                (force_mode == NlrForceMode::A || force_mode == NlrForceMode::B)
                ? "force" : "threshold";
            fprintf(stderr,
                    "[NLR ITER DISPATCH caller=%s path=%s (%s) NTask=%d "
                    "local_active=%d forced_modeb_global_active=%d]\n",
                    Spec::loop_name,
                    path == DispatchPath::ModeA_GPU_NGL ? "gpu_ngl" : "mode_b",
                    src, NTask, args.num_active,
                    forced_modeb_global_active);
            fflush(stderr);
        }
    }

    /* ===== CallScalars captured ONCE for whole call ===== */
    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    /* ===== Mode B hard-corridor counter snapshot =====
     * Mode B paths MUST NOT enter move_particles / ghost_exchange_impl /
     * gpu_particles_arena_acquire, and MUST NOT mutate NumPart. Always-on
     * enforcement around the iterative call; mirrors non-iter run_neighbor_loop
     * lines 2018-2026 + 2099-2122. Snapshot here; check post-iter-loop.
     * Mode A paths legitimately advance these counters (one ghost_import +
     * one arena_acquire per call); enforcement skipped. */
    const uint64_t s_drift0 = g_global_drift_counter;
    const uint64_t s_ghost0 = g_ghost_import_counter;
    const uint64_t s_arena0 = g_gpu_arena_acquire_counter;
    const int      s_np0    = NumPart;

    /* ===== Driver lifetime wrapped in inner scope =====
     * Driver destructor runs at scope exit; Mode B hard-corridor check fires
     * AFTER scope exit so the check covers driver cleanup hooks too (defensive
     * — current cleanup_device_context contractually MUST NOT touch
     * drift/ghost/arena globals, but inner-scope wrapping makes the invariant
     * maximally airtight). */
    {
    NlrIterDriver<Spec> drv(args, cs);

    /* ===== Path-specific DeviceContext init =====
     * Mode B: bind ctx.P=args.P/CellP directly (no arena).
     * Mode A: acquire_arena_and_init_ctx_mode_a — arena_acquire ONCE per call
     *         then bind ctx to arena-resident P_gpu/CellP_gpu. */
    if (path == DispatchPath::ModeB_HostWalker) {
        drv.initialize_device_context_mode_b();
    } else if (path == DispatchPath::ModeA_GPU_NGL) {
        drv.acquire_arena_and_init_ctx_mode_a();
    } else {
        /* Future path — not currently reachable. */
        if (ThisTask == 0) {
            fprintf(stderr,
                "[run_neighbor_loop_iterative<%s>] FATAL: unhandled path %d.\n",
                Spec::loop_name, (int)path);
            fflush(stderr);
        }
        /* path is Allreduce-symmetric (see dispatch above) -> all ranks hit this
         * together; soft bad-stop + immediate poll (81203 precedent). */
        endrun(81208); gizmo_exit_bad_stop_if_requested("nlr:unhandled_dispatch");
    }
    /* Drain a soft bad-stop raised inside the path-specific init (Mode-A arena
     * lifecycle 81211/81212 return early without binding ctx) BEFORE any device
     * dispatch. All-rank: Mode-A path is symmetric. */
    gizmo_exit_bad_stop_if_requested("nlr:iter_context_init");

    /* ===== Outer iter loop =====
     *
     * INVARIANT: `accum_uvm[sg][slot]` always
     * holds the most recent evaluated AccumData for that slot. Converged
     * slots are NEVER re-touched after they leave active_set; their final
     * iter's result persists in accum_uvm until the post-loop apply_active_writeback
     * reads it.
     *
     * Per-iter zero of AccumData is therefore the dispatch helper's
     * responsibility (it zeroes via Spec::zero_accum on ONLY the compacted
     * active subset before pair_kernel evaluation; see
     * nlr_iter_dispatch_subgroup_mode_b_local).
     * The runner does NOT zero converged slots — doing so would destroy
     * exactly the result apply_active_writeback needs. The earlier "defensive
     * cross-iter zero of all slots" in this position was a bug. */
    /* Partition-by-subgroup contract: record the
     * iter-0 per-subgroup j_type_bitmask values so the per-iter partition
     * assertion below can verify Spec::active_subgroup_key returns the same
     * key for every active in that subgroup on every iter. Only populated
     * when the Spec opts into actives_partition_by_subgroup. Compile-time
     * no-op for Specs that don't. */
    std::vector<unsigned int> partition_expected_bm_key;
    if constexpr (nlr_spec_actives_partition_by_subgroup_v<Spec>) {
        static_assert(nlr_spec_has_active_subgroup_key_v<Spec>,
                      "Spec::actives_partition_by_subgroup=true requires "
                      "Spec::active_subgroup_key(const DeviceContext&, int, "
                      "const CallScalars&) returning int (= bm key).");
        partition_expected_bm_key.resize(args.num_subgroups);
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            partition_expected_bm_key[sg] = args.subgroups[sg].j_type_bitmask;
        }
    }

    for (drv.iter_index = 0; drv.iter_index < Spec::max_iters; drv.iter_index++) {

        /* mode_a_rebuild_csr_every_iter correctness
         * fallback. When the Spec sets
         * this trait true, force-invalidate every subgroup's Mode A CSR
         * cache at the start of every iter > 0, bypassing the buffer-
         * exceedance trigger entirely. The static_assert on
         * mode_a_csr_buffer_factor > 1.0 is unchanged (factor stays a valid
         * number even when unused). Iter 0 is exempt — initial CSR build
         * happens in the per-subgroup dispatch on first invalidation pass. */
        if constexpr (nlr_spec_mode_a_rebuild_csr_every_iter_v<Spec>) {
            if (path == DispatchPath::ModeA_GPU_NGL && drv.iter_index > 0) {
                for (int sg = 0; sg < args.num_subgroups; sg++) {
                    drv.mode_a_csr_valid[sg] = false;
                }
            }
        }

        /* Partition-by-subgroup debug assertion (DEBUG / GIZMO_NLR_ASSERT_PARTITION
         * only; zero production overhead). Checks each active's active_subgroup_key
         * against the iter-0 j_type_bitmask; a mismatch routes to the graceful
         * controlled stop below. Host-side, before per-iter dispatch. */
#if defined(DEBUG) || defined(GIZMO_NLR_ASSERT_PARTITION)
        if constexpr (nlr_spec_actives_partition_by_subgroup_v<Spec>) {
            int local_partition_bad = 0;
            for (int sg = 0; sg < args.num_subgroups && !local_partition_bad; sg++) {
                /* Skip globally-converged subgroups (iter > 0 only). */
                if (drv.iter_index > 0 && drv.global_active_per_sg[sg] <= 0) continue;
                const int n_compacted = drv.active_set_size[sg];
                const NlrSubgroup& sgr = args.subgroups[sg];
                const unsigned int expect = partition_expected_bm_key[sg];
                for (int k = 0; k < n_compacted; k++) {
                    const int slot = drv.active_set_uvm[sg][k];
                    const int i    = sgr.active_indices[slot];
                    const int key  = Spec::active_subgroup_key(drv.ctx, i, drv.cs);
                    if (static_cast<unsigned int>(key) != expect) {
                        printf("[run_neighbor_loop_iterative<%s>] FATAL: "
                               "actives_partition_by_subgroup key drift. "
                               "sg=%d iter=%d slot=%d i=%d expected_bm=%u got_key=%d. "
                               "active_subgroup_key MUST be a pure function of "
                               "state that does not change across iterations.\n",
                               Spec::loop_name, sg, drv.iter_index, slot, i,
                               expect, key); fflush(stdout);
                        local_partition_bad = 1;
                        break;
                    }
                }
            }
            /* Debug-only: collectivize the per-rank assertion result so all ranks
             * reach the controlled stop together BEFORE the per-iter dispatch
             * collectives below (a lone asserting rank would otherwise desync). */
            int global_partition_bad = local_partition_bad;
            if (NTask > 1) {
                MPI_Allreduce(&local_partition_bad, &global_partition_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            }
            if (global_partition_bad) {
                endrun(90001020);
                gizmo_exit_bad_stop_if_requested("nlr:partition_key_drift");
            }
        }
#endif

        /* (a-pre) Mode A pre-dispatch invalidation sweep.
         * If any subgroup's CSR is invalid (set by the buffer-exceedance
         * trigger in last iter's step b.5), call the no-arg union-rebuild
         * method ONCE before any per-subgroup dispatch this iter. Prevents
         * mixing old/new arena pools across subgroup dispatches in the same
         * iter. Skips iter 0 (initial arena
         * acquire was via acquire_arena_and_init_ctx_mode_a). */
        if (path == DispatchPath::ModeA_GPU_NGL && drv.iter_index > 0) {
            /* The sweep must be COLLECTIVE.
             * mode_a_csr_valid is rank-local; a rank with no actives for a
             * globally-active subgroup never builds CSR locally, so its
             * mode_a_csr_valid[sg] stays false while another rank's is true.
             * Without an Allreduce, one rank enters ghost_exchange_cleanup +
             * reimport collectives while the other skips them => deadlock.
             * Also skip globally-converged subgroups: a sg with
             * global_active_per_sg[sg]==0 may have invalid CSR from an
             * earlier buffer-exceedance trigger but doesn't need a rebuild. */
            int local_needs_rebuild = 0;
            for (int sg = 0; sg < args.num_subgroups; sg++) {
                if (drv.global_active_per_sg[sg] <= 0) continue;
                /* Note: a rank
                 * with zero local actives in this globally-active subgroup
                 * never builds CSR locally, so its mode_a_csr_valid[sg]
                 * stays false. Without this guard, such a rank would vote
                 * "rebuild" every iteration forever — Allreduce would
                 * force a global rebuild that doesn't actually fix the
                 * vote, infinite loop of useless global rebuilds. The rank
                 * still participates collectively in the Allreduce below
                 * (voting 0); it just doesn't request a rebuild for a CSR
                 * it doesn't need. */
                if (drv.active_set_size[sg] <= 0) continue;
                if (!drv.mode_a_csr_valid[sg]) {
                    local_needs_rebuild = 1;
                    break;
                }
            }
            int global_needs_rebuild = local_needs_rebuild;
            if (NTask > 1) {
                MPI_Allreduce(&local_needs_rebuild, &global_needs_rebuild, 1,
                              MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            }
            if (global_needs_rebuild) {
                drv.rebuild_mode_a_arena_and_ctx_for_current_active_union();
            }
        }

        /* (a0) Per-iter Spec hook: reset_per_iter_device_context.
         * Optional. Runs HOST-side on every rank before subgroup dispatch.
         * Use case: ags_density's per_iter_wakeup_detected counter zero.
         * Hook MUST NOT do MPI. */
        if constexpr (nlr_spec_has_reset_per_iter_device_context_v<Spec>) {
            Spec::reset_per_iter_device_context(args, drv.ctx, drv.iter_index);
        }

        /* (a) Per-subgroup dispatch — fixed path for the call. This
         * implements Mode B local (np=1) and Mode B remote (np>1) via the
         * same DispatchPath::ModeB_HostWalker label; the per-subgroup helper
         * picks local vs remote based on NTask. Mode A iter is still
         * hard-stubbed at the path-selection block. */
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            /* Skip globally-converged subgroups. Saves
             * per-iter no-op collectives. On iter 0 global_active_per_sg
             * is still 0 (not yet Allreduced); use local count as proxy +
             * global presence-of-actives (subgroups[] is filled from
             * global_bm_presence by the caller, so any subgroup present
             * has SOME rank with non-zero actives — entering the dispatch
             * is the correct lockstep collective participation). */
            const bool sg_globally_active = (drv.iter_index == 0)
                ? true   /* iter 0: trust caller's subgroups[] (global union) */
                : (drv.global_active_per_sg[sg] > 0);
            if (!sg_globally_active) continue;

            /* (a) Mode B only (Mode A hard-stubbed at outer entry).
             * Helpers are collective on remote — must be
             * entered on every rank regardless of local n_compacted. */
            switch (path) {
                case DispatchPath::ModeB_HostWalker:
                    if (NTask == 1) {
                        nlr_iter_dispatch_subgroup_mode_b_local<Spec>(drv, sg);
                    } else {
                        nlr_iter_dispatch_subgroup_mode_b_remote<Spec>(drv, sg);
                    }
                    break;
                case DispatchPath::ModeA_GPU_NGL:
                    nlr_iter_dispatch_subgroup_mode_a<Spec>(drv, sg);
                    break;
            }
        }

        /* (b) Per-active Spec::after_iter — collect IterStatus, compact
         * active_set per subgroup, mutate radii on AdjustRadius. */
        drv.local_active_total = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            int n_compacted   = drv.active_set_size[sg];
            int write_idx     = 0;
            for (int k = 0; k < n_compacted; k++) {
                int  slot = drv.active_set_uvm[sg][k];
                int  i    = args.subgroups[sg].active_indices[slot];
                AfterIterContext<Spec> ctx{
                    args, sg, slot, i, drv.iter_index,
                    /* h_search_current */ drv.radii_uvm[sg][slot],
                    /* scalars (ref to driver-owned) */ drv.cs,
                    /* scratch (mutable ref) */          drv.scratch_uvm[sg][slot]
                };
                IterResult r = Spec::after_iter(ctx, drv.accum_uvm[sg][slot]);
                switch (r.status) {
                    case IterStatus::Converged:
                        /* Drop slot from compacted active_set; its accum_uvm
                         * stays so apply_active_writeback can read it post-loop. */
                        break;
                    case IterStatus::NeedsMore:
                        drv.active_set_uvm[sg][write_idx++] = slot;
                        break;
                    case IterStatus::AdjustRadius:
                        drv.radii_uvm[sg][slot] = r.new_h_search;
                        drv.active_set_uvm[sg][write_idx++] = slot;
                        break;
                    default:
                        /* Unknown enum = Spec bug. Soft bad-stop + drop the
                         * slot as converged (do not re-queue): the run still
                         * stops at the next poll, but stays lockstep with
                         * peers through this iteration's collectives. The
                         * stderr above surfaces the bug — not masked. */
                        if (ThisTask == 0) {
                            fprintf(stderr,
                                "[run_neighbor_loop_iterative<%s>] FATAL: Spec::after_iter "
                                "returned unknown IterStatus=%d at iter=%d sg=%d slot=%d "
                                "(particle index i=%d). Valid values: 0=Converged, "
                                "1=NeedsMore, 2=AdjustRadius.\n",
                                Spec::loop_name, (int)r.status,
                                drv.iter_index, sg, slot, i);
                            fflush(stderr);
                        }
                        endrun(81206);
                        break;
                }
            }
            drv.active_set_size[sg]     = write_idx;
            drv.local_active_per_sg[sg] = write_idx;
            drv.local_active_total     += write_idx;
        }


        /* (b.5) Mode A buffer-exceedance rebuild trigger.
         * After radii mutation on AdjustRadius, check if any still-active
         * slot's radius now exceeds its buffered radius (built oversized
         * at last build). If so, invalidate that subgroup's cached CSR;
         * next iter's dispatch will rebuild. Mirrors legacy density_gpu.cc:
         * 152-188 rebuild trigger. */
        if (path == DispatchPath::ModeA_GPU_NGL) {
            for (int sg = 0; sg < args.num_subgroups; sg++) {
                if (!drv.mode_a_csr_valid[sg]) continue;
                int n_compacted = drv.active_set_size[sg];
                for (int k = 0; k < n_compacted; k++) {
                    int slot = drv.active_set_uvm[sg][k];
                    if (drv.radii_uvm[sg][slot] > drv.mode_a_csr_buffered_h[sg][slot]) {
                        drv.mode_a_csr_valid[sg] = false;
                        break;
                    }
                }
            }
        }

        /* (c) Spec::after_iter_global — host-only, no-MPI (TRAP 7). */
        if constexpr (nlr_spec_has_after_iter_global_v<Spec>) {
            Spec::after_iter_global(args, drv);
        }

        /* (d) Per-iter MPI Allreduce on PER-SUBGROUP local_active counts.
         * Single array Allreduce-SUM gives per-sg
         * global activity; sum is global_active_total for the break check.
         * Globally-converged subgroups (global_active_per_sg[sg]==0) skip
         * the per-iter dispatch + collective on the next iteration. */
        if (NTask > 1) {
            MPI_Allreduce(drv.local_active_per_sg.data(),
                          drv.global_active_per_sg.data(),
                          args.num_subgroups, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        } else {
            drv.global_active_per_sg = drv.local_active_per_sg;
        }
        drv.global_active_total = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            drv.global_active_total += drv.global_active_per_sg[sg];
        }
        if (drv.global_active_total == 0) break;
    }

    /* ===== Max-iter exceeded check (matches legacy density / ags_density endrun(1155)) ===== */
    if (drv.iter_index >= Spec::max_iters && drv.global_active_total > 0) {
        if constexpr (nlr_spec_has_on_max_iter_exceeded_v<Spec>) {
            Spec::on_max_iter_exceeded(drv);
        } else {
            nlr_default_on_max_iter_exceeded<Spec>(drv);
        }
    }

    /* ===== Final apply_active_writeback (final-only) ===== */
    /* Fires once per active across all subgroups, on the final iteration's
     * accum (whether that iteration was Converged for that slot, or
     * max_iters terminated for everyone). The active_list semantics for
     * apply_active_writeback are the SUBGROUP'S full active_indices —
     * every active particle gets its converged result written back. */
    for (int sg = 0; sg < args.num_subgroups; sg++) {
        const NlrSubgroup& sgr = args.subgroups[sg];
        const int n_total = sgr.num_active_local;
        if (n_total <= 0) continue;
        /* Use effective_args: Mode A iterative refreshed
         * P/CellP/num_total after ghost import; apply_active_writeback hooks
         * may read these for correctness. Mode B leaves effective_args ==
         * base args. */
        neighbor_loop_args sub = drv.effective_args;
        sub.active_list = sgr.active_indices;
        sub.num_active  = n_total;
        for (int slot = 0; slot < n_total; slot++) {
            int i = sgr.active_indices[slot];
            /* If Spec opts into the iterative-variant hook, route the
             * converged radius + IterScratch through it. Specs that don't
             * declare the iterative hook fall through to the original. */
            if constexpr (nlr_spec_has_apply_active_writeback_iterative_v<Spec>) {
                Spec::apply_active_writeback_iterative(
                    sub, slot, i,
                    drv.accum_uvm[sg][slot],
                    drv.radii_uvm[sg][slot],
                    drv.scratch_uvm[sg][slot]);
            } else {
                Spec::apply_active_writeback(sub, slot, i, drv.accum_uvm[sg][slot]);
            }
        }
    }


    }  /* end inner scope: driver destructs HERE */

    /* ===== Mode B hard-corridor enforcement =====
     * Check AFTER final apply_active_writeback AND after driver destruction
     * so the invariant covers the FULL Mode B path including writeback hooks
     * and driver cleanup. Mode A paths legitimately advanced counters; skip
     * enforcement. */
    if (path == DispatchPath::ModeB_HostWalker) {
        const bool drift_violation = (g_global_drift_counter      != s_drift0);
        const bool ghost_violation = (g_ghost_import_counter      != s_ghost0);
        const bool arena_violation = (g_gpu_arena_acquire_counter != s_arena0);
        const bool np_violation    = (NumPart != s_np0);
        if (drift_violation || ghost_violation || arena_violation || np_violation) {
            int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            fprintf(stderr,
                "[NLR_ITER CORRIDOR ABORT rank=%d caller=%s path=mode_b] Mode B "
                "iterative path violated tiny-N corridor invariant during "
                "run_neighbor_loop_iterative. Counter deltas: drift=%llu "
                "ghost=%llu arena=%llu NumPart_pre=%d NumPart_post=%d\n",
                rank, Spec::loop_name,
                (unsigned long long)(g_global_drift_counter      - s_drift0),
                (unsigned long long)(g_ghost_import_counter      - s_ghost0),
                (unsigned long long)(g_gpu_arena_acquire_counter - s_arena0),
                s_np0, NumPart);
            fflush(stderr);
            /* one-shot self-check at iter-runner end (no loop); function returns next,
             * no intervening collective -- soft bad-stop + fall through, drains at the
             * next phase-boundary poll. */
            endrun(81213);
        }
    }
    /* Driver destructor frees per-subgroup UVM allocations on scope exit. */
}

/* ============================================================================
 * Explicit template instantiations — one per migrated caller.
 *
 * Forgetting an instantiation = clean linker error at the call site. Each
 * Spec must opt in here and is implicitly acknowledging its declared
 * Spec::sidx_cache_kind; cf. nlr_resolve_sidx_cache.
 *
 * ========================================================================== */

#ifdef SINK_PARTICLES
template void run_neighbor_loop<SinkEnv1Spec>(const neighbor_loop_args&);
template void run_neighbor_loop<SinkFeedSpec>(const neighbor_loop_args&);
template void run_neighbor_loop<SinkSwkSpec>(const neighbor_loop_args&);
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
template void run_neighbor_loop<SinkEnv2Spec>(const neighbor_loop_args&);
#endif
#endif


/* AgsDensitySpec — first production iterative Spec consumer of the
 * runner + partition-by-subgroup + sticky-call-scope wakeup
 * traits. See gravity/ags_density_loop.{h,cc}. */
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
template void run_neighbor_loop_iterative<AgsDensitySpec>(const neighbor_loop_args_iterative&);
/* AgsForceSpec — single-pass-iterative-shaped Spec for
 * non-gas AGS force loop (max_iters=1, after_iter always Converged); uses
 * the iterative path so the multi-subgroup contract is available. See
 * gravity/ags_force_loop.{h,cc}. */
template void run_neighbor_loop_iterative<AgsForceSpec>(const neighbor_loop_args_iterative&);
#endif

/* DensitySpec — hydro density runner port.
 * Single gas-only subgroup, no after_iter P/CellP writes,
 * uses apply_active_writeback_iterative for single-valued radius
 * channeling. See hydro/density_loop.{h,cc}. */
template void run_neighbor_loop_iterative<DensitySpec>(const neighbor_loop_args_iterative&);

/* MechFBSpec — mechanical-feedback runner port
 * (physics-complete pair kernel + full state-machine Spec
 * contract). 6-mode iterative state machine over loop_iteration
 * {-2,-1,0,1,2,3}; mode_a_rebuild_csr_every_iter=false preserves the legacy
 * 1-CSR-shared-across-6-modes optimization. Mode A multi-rank ghost-side
 * writes hit a Kokkos::abort; supporting them needs a custom MechFBGasDelta
 * ghost-writeback callback + lazy d_gas_iter, which are not implemented. See
 * galaxy_sf/mechfb_loop.{h,cc}. */
#ifdef GALSF_FB_MECHANICAL
template void run_neighbor_loop_iterative<MechFBSpec>(const neighbor_loop_args_iterative&);
#endif

/* ThermalFBSpec — thermal-feedback runner port.
 * Non-iterative scatter (Type-4 stars → gas neighbors); ActiveReduceOnly +
 * manifest-bundle ghost_writeback; sink_feed pattern. See
 * galaxy_sf/thermal_fb_loop.{h,cc}. The Spec definition is gated on
 * GALSF_FB_THERMAL in thermal_fb_loop.h, so the explicit instantiation must
 * sit inside the same #ifdef (non-thermal Configs would
 * otherwise hit an undefined type at this template instantiation site). */
#ifdef GALSF_FB_THERMAL
template void run_neighbor_loop<ThermalFBSpec>(const neighbor_loop_args&);
#endif

/* Cellcorrections corridor: CellcorrectionsSpec — first-pass
 * volume corrections (Volume_1 = sum_j Volume_0[j]^2 wk(r, h_j)).
 * NotIterative GasOnly Spec, no j-side writes, no ghost-writeback;
 * first corridor consumer in the chain. See
 * hydro/cellcorrections_loop.{h,cc}. */
#ifdef HYDRO_VOLUME_CORRECTIONS
template void run_neighbor_loop<CellcorrectionsSpec>(const neighbor_loop_args&);
#endif

/* GradientsSpec — runner port of the legacy `gradient_evaluate_gpu`
 * walker. Broad active list (Type==0 && Mass>0) matching the legacy GPU
 * walker; narrow GasGrad_isactive filter stays at the neighbor side inside
 * gradient_accumulate_neighbor. Symmetric gas-gas topology — hydro
 * corridor consumer (after CellcorrectionsSpec, before HydroForceSpec).
 * See hydro/gradients_loop.{h,cc}. */
template void run_neighbor_loop<GradientsSpec>(const neighbor_loop_args&);

#ifdef MHD_CONSTRAINED_GRADIENT
/* GradientsIterSpec — slim variant for the constrained-gradient iterations
 * (grad_iter>0). Accumulates the slim GasGraddata_out_iter_ (FaceDotB +
 * MIDPOINT PhiGrad) instead of the full GasGraddata_out_. Gated with the same
 * #ifdef that gates the Spec definition (guard the instantiation with the
 * Spec's own gate to avoid an undefined type in non-MHD builds). See
 * hydro/gradients_loop.{h,cc}. */
template void run_neighbor_loop<GradientsIterSpec>(const neighbor_loop_args&);
#endif

/* HydroForceSpec — runner port of the legacy `hydro_evaluate_gpu` walker.
 * Final hydro-corridor consumer (after CellcorrectionsSpec and
 * GradientsSpec). uses_ghost_writeback=true with a snapshot-diff bundle
 * (PARTICLE_MAX(wakeup) + MFV GAS_ADD(dMass)) for Mode A imported-ghost
 * lifecycle; Mode B direct-owner-rank j-writes via request-driven P2P.
 * See hydro/hydro_force_loop.{h,cc}. */
template void run_neighbor_loop<HydroForceSpec>(const neighbor_loop_args&);

/* RadFBRPSpec — local radiation-pressure winds.
 * Iterative 2-pass (iter 0 wt_sum aggregation; iter 1 kick application).
 * Ghost-writeback bundle with PARTICLE_ADD_VEC3 + new GAS_ADD_VEC3 ops.
 * iter-gating via Aux::iter_index (set in reset_per_iter_device_context);
 * inter-iter wt_sum staged through IterScratch by after_iter_global. See
 * galaxy_sf/radfb_rp_loop.{h,cc}. */
#ifdef GALSF_FB_FIRE_RT_LOCALRP
template void run_neighbor_loop_iterative<RadFBRPSpec>(const neighbor_loop_args_iterative&);
#endif

/* DMDispersionSpec — DM velocity dispersion runner port.
 * Gas actives (Type==0) iterate bisection on KernelRadiusDM to enclose
 * 64±48 DM (Type==1) neighbors; accumulates unweighted Vel sums for dispersion.
 * ActiveReduceOnly + SidxCacheKind::None (DM tbm matches neither GasOnly nor
 * AllTypes cache). apply_active_writeback_iterative + Aux finalize pattern
 * mirrors DensitySpec exactly. See galaxy_sf/dm_dispersion_loop.{h,cc}. */
#ifdef DM_DISPERSION_LOOP_ACTIVE
template void run_neighbor_loop_iterative<DMDispersionSpec>(const neighbor_loop_args_iterative&);
#endif

/* CBE gradients corrective architecture pivot
 * (sidm/cbe_integrator_gradients.{h,cc}). CBEGradSpec is non-iterative
 * (paralleling DMGradSpec); the two passes (raw LSQ then pairwise BJ-style
 * conservative limiter) are orchestrated by CBEGrad_gradient_calc() at the
 * toplevel via Aux::loop_iteration. Persistent storage on
 * P[i].Gradients_CBE_basis_moments; standard P[] ghost transport carries
 * gradients across ranks (no scratch, no custom Alltoallv). */
#if defined(CBE_INTEGRATOR_WITHGRADIENTS)
template void run_neighbor_loop<CBEGradSpec>(const neighbor_loop_args&);
#endif

/* RtSrcInjectionSpec — radiation source
 * injection runner port. Non-iterative scatter (non-gas sources → gas); the
 * toplevel builds the active list directly (Aux::host_locals) so
 * nlr_build_active_list is not used. SYMMETRIC search matches the legacy GPU
 * evaluator (correctness-required under RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION).
 * Ghost-writeback bundle uses three new generic ops (GAS_ADD_ARRAY,
 * GAS_ADD_2D, GAS_ADD_VEC3_ARRAY); GRAIN_RDI_TESTPROBLEM boundary condition
 * is imposed by an owner-local post-runner fixup in
 * radiation/rt_source_injection.cc (not inside the pair kernel). See
 * radiation/rt_source_injection_loop.{h,cc}. */
#ifdef RT_SOURCE_INJECTION
template void run_neighbor_loop<RtSrcInjectionSpec>(const neighbor_loop_args&);
#endif

/* difffilter: DiffFilterSpec + DynDiffSpec — TURB_DIFF_DYNAMIC
 * velocity-smoothing + dynamic-Smagorinsky loops. Non-iterative,
 * scaled-symmetric gas-gas (symmetric_neighbor_radius_scale = TurbDynamicDiffFac),
 * pure i-side reduce. See turb/difffilter_loop.{h,cc}. */
#ifdef TURB_DIFF_DYNAMIC
template void run_neighbor_loop<DiffFilterSpec>(const neighbor_loop_args&);
template void run_neighbor_loop<DynDiffSpec>(const neighbor_loop_args&);
#endif

/* dm_fuzzy: DMGradSpec — DM_FUZZY higher-order density-gradient
 * estimator. Non-iterative, one-way DM->DM, fixed radius P[i].AGS_KernelRadius,
 * pure i-side reduce. The toplevel DMGrad_gradient_calc issues one call per
 * (gradient-pass, bm group) with args.neighbor_type_mask_override = bm. See
 * sidm/dm_fuzzy_loop.{h,cc}. */
#ifdef DM_FUZZY
template void run_neighbor_loop<DMGradSpec>(const neighbor_loop_args&);
#endif

/* grain_physics: GrainBackrxSpec + GrainRTGasSpec + GrainRTGrainSpec.
 * Three independent non-iterative Specs for the grain_physics neighbor loops
 * (grain→gas backreaction with ghost writeback; gas→grain
 * RT opacity, pure i-side; grain→gas radiation acceleration,
 * pure i-side). Replaces grain_backrx_evaluate_gpu +
 * interpolate_fluxes_opacities_gasgrains_evaluate_gpu in
 * solids/grain_physics_gpu.cc (retired in cleanup commit). See
 * solids/grain_physics_loop.{h,cc}. */
#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION) && defined(GRAIN_BACKREACTION)
template void run_neighbor_loop<GrainBackrxSpec>(const neighbor_loop_args&);
#endif
#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION) && defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
template void run_neighbor_loop<GrainRTGasSpec>(const neighbor_loop_args&);
template void run_neighbor_loop<GrainRTGrainSpec>(const neighbor_loop_args&);
#endif

