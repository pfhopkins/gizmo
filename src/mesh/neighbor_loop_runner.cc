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
 * -> drift_and_filter_candidates -> evaluate_pairs_post_drift; it uses
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
#include <climits>                                   /* INT_MAX */
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
#include "device_tree_walk.h"          /* the one device traversal; Mode D enters it from the root */
#include "device_packet_walk.h"        /* packets: several queries sharing one traversal (shared with the gravity walk) */
#include "mode_d_export.h"             /* Mode D's device export decision, envelope packing and reply merge */
#include "../declarations/gpu_recorder_claim.h"   /* gx_touched_set_claim_in, the fused walk's touched-set claim */
#include "kernel.h"  /* MUST precede sink_env1_loop.h (kernel_main, NEAREST_XYZ) */
#include "ghost_writeback.h"             /* ghost_get_num_local */
#include "ghost_symlist_lifecycle.h"     /* gizmo_request_filtered_ghost_import_fresh, ghost_exchange_cleanup */
#include "mode_b_local_walker.h"         /* mode_b_local_neighbor_walk, brute */
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
    args.active_call_slot    = nullptr;       /* the caller's list is the call's list */
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
 * Mode-A/B/D dispatch thresholds
 *
 * Mode-B resolution precedence (first wins):
 *   1. parameterfile NeighborLoopModeBThreshold{Sum,Max} (-1 = unset;
 *      any other value overrides, <= 0 disables Mode B)
 *   2. Spec::modeb_threshold_{sum,max} constexpr, where a Spec declares one
 *   3. the default for the loop's search mode (nlr_default_modeb_thresholds)
 *
 * Mode D, on the calls Mode B does not take, needs both counts at or above the
 * parameterfile pair NeighborLoopModeDThreshold{Sum,Max}; unset, that is 0/0,
 * so an eligible one-way loop uses Mode D wherever Mode B is not chosen.
 *
 * The thresholds are settable ONLY from the parameterfile, so the values a run
 * used are recorded in its parameter log.
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

bool nlr_mode_b_rule(int sum_active, int max_active, int default_sum, int default_max)
{
    const int TS = gizmo_nlr_modeb_threshold_sum_for(nullptr, default_sum);
    const int TM = gizmo_nlr_modeb_threshold_max_for(nullptr, default_max);
    return (sum_active > 0) && (sum_active <= TS) && (max_active <= TM);
}

/* The fewest actives some rank must hold before Mode D takes a call: one device batch (GPU_MIN_SOURCES_FOR_WALK_OFFLOAD).
 * When no rank holds that many the request-driven host walker is the faster path: a device batch of fewer queries loses
 * to the host per query (measured on FIRE-dwarf and fire2ics, 3-7x below 4096), and a Mode D iteration's fixed cost --
 * export walks, count exchange, termination -- is then more than Mode B's whole iteration (Poisson box, 128 ranks: the
 * remainder of a density call finishing on Mode B below this took the call from 31-38 to 21.6 ms).  The busiest rank is
 * the test, not an average, so a call whose work sits on a few ranks still gives those ranks the device.  Asked when a
 * call is routed and again after every iteration of an iterative Mode D call (nlr_mode_d_feeds_device). */
static constexpr int MODE_D_MIN_ACTIVES_PER_RANK = GPU_MIN_SOURCES_FOR_WALK_OFFLOAD;

/* Built-in Mode-B thresholds by search mode, used where neither the parameterfile
 * nor the Spec sets one.
 *   One-way loops: Mode B up to 64 active globally (and 64 on any rank); above
 *   that an eligible loop uses Mode D, which needs no ghost import either, once the
 *   busiest rank holds MODE_D_MIN_ACTIVES_PER_RANK (Mode B below that, while Mode B is on).
 *   Symmetric loops: Mode B up to an average of 320 active per rank, with no rank
 *   above 3200.  Measured on Frontier, Mode A's import and list build stop paying
 *   for themselves below a few hundred active per rank.  The per-rank cap is a
 *   separate, absolute guard against imbalance (ten times the 320 cutoff): a step
 *   goes to Mode A when any one rank holds more than 3200, which keeps a hot rank's
 *   request-driven walks and its export rounds bounded. */
void nlr_default_modeb_thresholds(int search_mode, int *threshold_sum, int *threshold_max)
{
    if(search_mode == MODE_B_SEARCH_SYMMETRIC) {
        const long long per_rank_average = 320, per_rank_max = 3200;
        const long long sum = per_rank_average * (long long)NTask;
        *threshold_sum = (sum > INT_MAX) ? INT_MAX : (int)sum;
        *threshold_max = (int)per_rank_max;
    } else {
        *threshold_sum = 64;
        *threshold_max = 64;
    }
}

void gizmo_nlr_report_dispatch_thresholds(void)
{
    if(ThisTask != 0) return;
    int oneway_sum, oneway_max, symm_sum, symm_max, d_sum, d_max;
    nlr_default_modeb_thresholds(MODE_B_SEARCH_ONEWAY, &oneway_sum, &oneway_max);
    nlr_default_modeb_thresholds(MODE_B_SEARCH_SYMMETRIC, &symm_sum, &symm_max);
    oneway_sum = gizmo_nlr_modeb_threshold_sum_for(nullptr, oneway_sum);
    oneway_max = gizmo_nlr_modeb_threshold_max_for(nullptr, oneway_max);
    symm_sum = gizmo_nlr_modeb_threshold_sum_for(nullptr, symm_sum);
    symm_max = gizmo_nlr_modeb_threshold_max_for(nullptr, symm_max);
    gizmo_nlr_moded_thresholds(&d_sum, &d_max);
    printf("Neighbor loops: Mode B at or below summed/per-rank-max active %d/%d (one-way), %d/%d (symmetric); "
           "eligible one-way loops use Mode D at or above %d/%d with at least %d on the busiest rank (below that Mode B when "
           "it is on, else the usual path), Mode A otherwise\n",
           oneway_sum, oneway_max, symm_sum, symm_max, d_sum, d_max, MODE_D_MIN_ACTIVES_PER_RANK);
}

void gizmo_nlr_moded_thresholds(int *threshold_sum, int *threshold_max)
{
    const int p_sum = nlr_host_all_ptr()->NeighborLoopModeDThresholdSum;
    const int p_max = nlr_host_all_ptr()->NeighborLoopModeDThresholdMax;
    /* Unset (-1) = 0: Mode D wherever Mode B is not chosen and the call reaches MODE_D_MIN_ACTIVES_PER_RANK. */
    *threshold_sum = (p_sum == -1) ? 0 : p_sum;
    *threshold_max = (p_max == -1) ? 0 : p_max;
}


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
        case NeighborLoopPlan::Path::ModeD_DeviceFused: return false;
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

/* The iterative form: the subgroups carry caller-built lists (all or none,
 * checked at the iterative entry, so the first subgroup answers for all). */
static inline bool nlr_subgroups_carry_caller_lists(const neighbor_loop_args_iterative &args)
{
    return args.num_subgroups > 0 && args.subgroups[0].external_csr != nullptr;
}

/* A caller that owns the ghost pool a list indexes must hand the runner a live
 * pool with its provenance, and arguments built after its import (num_total
 * spans the ghosts).  At NTask==1 there is no pool to check (imports early-out).
 * ghost_get_num_ghosts()==0 is NOT a liveness signal (also 0 for a live
 * zero-ghost pool); ghost_pool_is_live() is.  Soft stop on a violation. */
static void nlr_check_caller_owned_ghost_pool(const neighbor_loop_args &args, const char *loop_name)
{
    if(NTask <= 1) {return;}
    const char *own_err = nullptr;
    if(!ghost_pool_is_live())                 own_err = "ghost pool not live";
    else if(ghost_get_home_rank()  == nullptr ||
            ghost_get_home_index() == nullptr) own_err = "ghost provenance maps absent";
    else if(args.num_total != NumPart)         own_err = "args.num_total != NumPart (args built before caller's import?)";
    if(own_err != nullptr) {
        fprintf(stderr, "[neighbor_loop_runner rank=%d caller=%s] caller-owned ghost-pool "
                "contract violation: %s\n", ThisTask, loop_name, own_err);
        fflush(stderr);
        endrun(7313);
    }
}

bool nlr_path_uses_gpu_arena(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return true;
        case NeighborLoopPlan::Path::ModeB_Local:  return false;
        case NeighborLoopPlan::Path::ModeB_Remote: return false;
        case NeighborLoopPlan::Path::ModeD_DeviceFused: return true;
    }
    return false;
}

bool nlr_path_permits_global_numpart_mutation(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return true;
        case NeighborLoopPlan::Path::ModeB_Local:  return false;
        case NeighborLoopPlan::Path::ModeB_Remote: return false;
        case NeighborLoopPlan::Path::ModeD_DeviceFused: return false;
    }
    return false;
}

bool nlr_path_uses_lazy_drift(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return false;
        case NeighborLoopPlan::Path::ModeB_Local:  return true;
        case NeighborLoopPlan::Path::ModeB_Remote: return true;
        case NeighborLoopPlan::Path::ModeD_DeviceFused: return true;
    }
    return false;
}

const char *nlr_path_label(NeighborLoopPlan::Path path)
{
    switch(path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl: return "gpu_ngl";
        case NeighborLoopPlan::Path::ModeB_Local:  return "mode_b_local";
        case NeighborLoopPlan::Path::ModeB_Remote: return "mode_b_remote";
        case NeighborLoopPlan::Path::ModeD_DeviceFused: return "mode_d_fused";
    }
    return "unknown";
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
 *     collect_candidates_pre_drift<Spec>   — the tree walk records every
 *                                            eligible particle under the nodes
 *                                            it opens, without reading P[j]'s
 *                                            possibly-stale position
 *     drift_and_filter_candidates<Spec>    — drift_particle on every j to
 *                                            All.Ti_Current (a j already
 *                                            current returns at once), then
 *                                            keep exactly the neighbours at
 *                                            current positions
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

/* WHERE one batch of fused-walk sources runs: device, host threads, or one host
 * core. Every fused walk goes through here -- the source loop and the placement
 * decision exist once, and the five walks that use it (record and evaluate, from
 * the root and resumed from a peer's start nodes, plus the single-rank evaluator)
 * keep their own surrounding protocol and share nothing else. Adding a sixth walk
 * means calling this, not writing another loop.
 *
 * Nothing about the traversal, the leaf policy, the drift, the exchange or the
 * reply protocol varies with the choice -- only the execution space. The body is a
 * `KOKKOS_LAMBDA`, which both device compilers expand to `[=] __host__ __device__`,
 * so one body serves all three arms unchanged.
 *
 * The count is the batch's OWN source count, never a step-level or global one: a
 * rank walking ten sources of its own is a different question from the same rank
 * answering ten thousand imported ones, and on a clustered run those differ by
 * three orders of magnitude within a single call.
 *
 * CAPTURE RESIDENCE -- the one way this can go silently wrong. The body captures by
 * value, so a CallScalars, a device context or a tree view it captures lives INSIDE
 * the closure: on the host stack when called here, device-resident when copied to a
 * launch. A leaf built inside the body therefore points into whichever space is
 * executing, which is what makes all three arms correct. Every other pointer a leaf
 * holds -- particles, accumulators, the touched set, the anomaly word -- is
 * SharedSpace and valid in both. Capturing any of the first group BY REFERENCE
 * would leave a leaf pointing at a host stack object on the device arm, which is a
 * fault or a wrong answer rather than a compile error. Those captures must also
 * stay unmodified for the whole batch, and no pointer into the closure may outlive
 * the call.
 *
 * Two preconditions, both true at present call sites: any device work this batch
 * reads from has already completed (each producer fences at its own launch), and no
 * caller is already inside a parallel region. */
/* Whether nlr_walk_for_sources runs a batch of n on the device; smaller batches run on the host. */
static inline bool nlr_walk_runs_on_device(int n) {return n >= GPU_MIN_SOURCES_FOR_WALK_OFFLOAD;}

template <class F>
static inline void nlr_walk_for_sources(const char *tag, int n, F &&body)
{
    if(n <= 0) {return;}
    if(nlr_walk_runs_on_device(n)) {
        gizmo_gpu_kernel_launch(tag, n, std::forward<F>(body));
        return;
    }
    if(nlr_modeb_use_omp(n, nlr_modeb_omp_nthreads())) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, MODEB_OMP_CHUNK_ACTIVE)
#endif
        for(int kk = 0; kk < n; kk++) {body(kk);}
        return;
    }
    for(int kk = 0; kk < n; kk++) {body(kk);}
}

/* Eval-threading policy for evaluate_pairs_post_drift. The production tree eval
 * may thread (BitwiseReadonly or EpsilonAtomic specs, i.e. any non-SerialOnly tier).
 * Passed explicitly at every call site (no default) so production vs
 * reference eval is greppable and can never be silently mis-gated. */
enum class EvalOMPPolicy { AllowProduction, ForceSerialReference };


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
                                          std::vector<std::vector<int>>& per_active_cands)
{
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
    if(use_omp) {
        /* Each thread mutates only its own per_active_cands[aa] (outer vector
         * pre-sized; no shared push) and its own drift counter (diagnostic
         * only — allocated iff a counter sink was requested). */
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
            mode_b_local_neighbor_walk(pos_arr, h_q, neighbor_type_mask,
                                       Spec::search_mode, Spec::radius_policy,
                                       cands, jscale);
        }
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
    std::vector<std::vector<int>>& per_query_cands)
{
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
    if(use_omp) {
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
            if(peer_nnodes[k] > 0) {
                mode_b_walk_from_start_nodes(pos_arr, h_q, neighbor_type_mask,
                                             Spec::search_mode, Spec::radius_policy,
                                             &peer_nodelist_flat[(size_t)k * NODELISTLENGTH],
                                             peer_nnodes[k], cands, jscale);
            } else {
                mode_b_local_neighbor_walk(pos_arr, h_q, neighbor_type_mask,
                                            Spec::search_mode, Spec::radius_policy,
                                            cands, jscale);
            }
        }
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

/* Drift each query's walk candidates to the current time and keep exactly its neighbours at their
 * current positions (mode_b_drift_and_filter_candidates).  query_at(aa, pos, h_q) must give the query
 * that list was walked with; every query is read before any candidate is drifted, since a query's own
 * particle may be another query's candidate.  Serial, as drift_particle requires; drift_particle
 * returns at once for a particle already current, so one reached by several queries is drifted once.
 * It runs only after every walk has finished (the walks' OpenMP loop ends in its barrier), so no particle
 * is drifted while a walk reads its clock and position -- which is why the walker's leaf reads them
 * plainly.  Drifting candidates while walks are still running would need deduplication across threads
 * and a synchronised clock and position, not just a different load in the leaf. */
template <typename Spec, typename QueryAt>
static void drift_and_filter_candidates(std::vector<std::vector<int>>& per_active_cands,
                                        unsigned int neighbor_type_mask,
                                        QueryAt query_at)
{
    const double jscale = nlr_spec_symmetric_j_radius_scale<Spec>();
    const size_t n_queries = per_active_cands.size();
    std::vector<double> queries(4 * n_queries);   /* pos[3], h_q per query */
    for(size_t aa = 0; aa < n_queries; aa++) {
        if(per_active_cands[aa].empty()) continue;
        query_at((int)aa, &queries[4 * aa], queries[4 * aa + 3]);
    }
    for(size_t aa = 0; aa < n_queries; aa++) {
        std::vector<int>& v = per_active_cands[aa];
        if(v.empty()) continue;
        mode_b_drift_and_filter_candidates(&queries[4 * aa], queries[4 * aa + 3], neighbor_type_mask,
                                           Spec::search_mode, Spec::radius_policy, jscale, v);
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
                                       const typename Spec::CallScalars& cs,
                                       EvalOMPPolicy eval_policy)
{
    using NeighborData = typename Spec::NeighborData;
    using ScatterData  = typename Spec::ScatterData;
    const struct GxMotionTargetSet motion_targets = gx_motion_target_view();

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
                if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_mark(motion_targets, j);}
                IdentitySidecar id{};                 /* NoIdentity */
                NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                Spec::pair_kernel(a, nb, accums[aa], s, cs);
            }
        } else {
            /* No bind hook: walk by const-ref, no copy. */
            const auto& a = actives[aa];
            for(size_t kk = 0; kk < cands.size(); kk++) {
                int j = cands[kk];
                if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_mark(motion_targets, j);}
                IdentitySidecar id{};                 /* NoIdentity */
                NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                Spec::pair_kernel(a, nb, accums[aa], s, cs);
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
        const int slot = args.active_call_slot ? args.active_call_slot[aa] : aa;
        actives_out[aa] = Spec::load_active(ctx, slot, args.active_list[aa],
                                             radii[aa], cs);
    }
}

template <typename Spec>
static void run_mode_b_local(const neighbor_loop_args& args, const double *radii)
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
    /* A hook that could not stage its buffers has already asked for the stop.
     * This path is host-local with no collectives, so returning leaves no peer
     * waiting; the guard above releases whatever the hook did obtain. */
    if(ctx.populate_failed) { return; }

    /* Freeze active snapshots host-side BEFORE drift. */
    std::vector<ActiveData> actives(N);
    build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs, actives.data());

    /* Helper layout: collect → drift → evaluate. */
    std::vector<std::vector<int>> cand_modeB;
    const unsigned int modeb_type_mask = nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask);
    {
        collect_candidates_pre_drift<Spec>(args, radii, modeb_type_mask,
                                            DispatchPath::ModeB_HostWalker, cand_modeB);
    }
    {
        drift_and_filter_candidates<Spec>(cand_modeB, modeb_type_mask, [&](int aa, double *pos, double &h_q) {
            const int i = args.active_list[aa];
            pos[0] = (double)args.P[i].Pos[0]; pos[1] = (double)args.P[i].Pos[1]; pos[2] = (double)args.P[i].Pos[2];
            h_q = radii[aa];
        });
    }

    std::vector<AccumData> accums(N);
    {
        evaluate_pairs_post_drift<Spec>(ctx, actives.data(), N, cand_modeB, accums.data(), cs, EvalOMPPolicy::AllowProduction);
    }

    /* Host writeback — same code path as Mode A's writeback. */
    {
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
/* ============================================================================
 * Answering another rank's queries against this rank's particles.
 *
 * A rank that receives a batch of queries has to return one accumulator per
 * query, and there is more than one way to arrive at them.  The host way walks
 * the local tree per query into a candidate list, drifts whatever the walk
 * touched, and then runs the pair kernel over the list.  A device way walks and
 * evaluates in a single pass and never builds a list at all.
 *
 * The seam is drawn around ALL THREE steps rather than around the evaluation
 * alone, and that is the whole point: an evaluation-only hook would take the
 * candidate list as an argument, so any backend behind it would still have to
 * build one -- which is precisely the cost a fused walk exists to avoid.  Drawn
 * here, "how the neighbours are found" and "whether they are ever written down"
 * are the backend's business rather than the transport's.
 *
 * The transport around this -- envelope exchange, provenance, reply exchange,
 * the bounded round loop -- does not know which backend ran and must not learn.
 * There is one body of MPI choreography for the peer wire and it stays that way.
 * ========================================================================== */
template <typename Spec>
struct NlrPeerAnswerHostWalk {
    using AccumData = typename Spec::AccumData;

    /* The host path, unchanged: collect against the local pool, drift what the
     * walk touched, evaluate. */
    static void answer(const typename Spec::DeviceContext& ctx,
                       const typename Spec::CallScalars& cs,
                       const std::vector<typename Spec::ActiveData>& peer_actives,
                       const std::vector<int>& peer_nodelist_flat,
                       const std::vector<int>& peer_nnodes,
                       unsigned int neighbor_type_mask,
                       std::vector<AccumData>& peer_replies_out)
    {
        /* Stage 6: collect PEER candidate sets PRE-DRIFT (against MY local pool). */
        std::vector<std::vector<int>> cand_peer_tree;
        {
            collect_candidates_for_remote_queries<Spec>(peer_actives,
                                                         peer_nodelist_flat, peer_nnodes,
                                                         neighbor_type_mask,
                                                         DispatchPath::ModeB_HostWalker,
                                                         cand_peer_tree);
            /* Reported rather than recorded here: the threaded-walk note is the
             * transport's own diagnostic, and it lives in a lambda that closes
             * over the transport's locals.  A backend that reached for it would
             * be reaching out of its scope, which is what the first draft of
             * this extraction did. */
        }

        /* Stage 7 (peer): drift THIS round's peer candidate sets (self candidates
         * were drifted once before the round loop). Idempotent to All.Ti_Current. */
        {
            drift_and_filter_candidates<Spec>(cand_peer_tree, neighbor_type_mask, [&](int k, double *pos, double &h_q) {
                pos[0] = (double)peer_actives[k].pos[0]; pos[1] = (double)peer_actives[k].pos[1]; pos[2] = (double)peer_actives[k].pos[2];
                h_q = (double)peer_actives[k].h_search;
            });
        }

        /* Stage 9: evaluate PEER queries post-drift -> peer_replies, shipped back
         * to the home rank. */
        const int K = (int)peer_actives.size();
        if(K > 0) {
            evaluate_pairs_post_drift<Spec>(ctx, peer_actives.data(), K,
                                              cand_peer_tree, peer_replies_out.data(), cs, EvalOMPPolicy::AllowProduction);
        }
    }
};

/* The motion-target set for one call of a loop that writes neighbour
 * velocities: opened before any kernel, raised and closed after the writeback.
 * A loop without the trait touches none of this. */
template <typename Spec>
static void nlr_motion_targets_open(void)
{
    if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {
        const int num_local = ghost_get_num_local();
        if(num_local <= 0) {return;}   /* nothing this rank owns can be marked */
        if(gx_motion_target_ensure(num_local) != 0) {
            /* Without the set no bound can be raised, and a bound not raised
             * under-includes silently. */
            if(ThisTask == 0) {fprintf(stderr, "[%s] FATAL: no memory for the motion-target set.\n", Spec::loop_name); fflush(stderr);}
            endrun(90001040);
            return;
        }
        gx_motion_target_begin_call();
    }
}
template <typename Spec>
static void nlr_motion_targets_close(void)
{
    if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_consume();}
}
/* Opened where a call's inputs are final, closed when the call leaves scope
 * -- after its writeback, whichever return it takes. */
template <typename Spec>
struct NlrMotionTargetScope {
    NlrMotionTargetScope()  {nlr_motion_targets_open<Spec>();}
    ~NlrMotionTargetScope() {nlr_motion_targets_close<Spec>();}
};

/* One rank's queries answered from its own tree, device-resident end to end.
 * Returns false only when the device buffers could not be had, in which case
 * nothing was evaluated and the caller answers on the host instead. */
template <typename Spec>
static bool nlr_mode_d_evaluate_single_rank(const typename Spec::DeviceContext& ctx,
                                            const typename Spec::CallScalars& cs,
                                            const int *active_idx_host,
                                            const int *active_slot_host,
                                            const double *radii_host,
                                            int n,
                                            unsigned int supply_mask,
                                            const GxDeviceTreeView& tree,
                                            typename Spec::AccumData *accums_out);

/* Mode D's device-resident exchange (defined with the device walks below) and the call storage it runs in.  The
 * exchange returns true when it ended in a stop every rank agreed on.  Its actives come from the host (args, radii)
 * or, for a call iterating on the device, from the device list of slots still iterating (ModeDDeviceQueries), and
 * its results go to accums_out on the host, or stay in the call's storage when accums_out is null. */
struct ModeDCallLease;
struct ModeDDeviceQueries {
    const int    *live;             /* [n_live] the call slots still iterating */
    int           n_live;
    int           call_slot_base;   /* the subgroup's first slot in the call's staging */
    const int    *slot_index;       /* [call slots] each slot's particle */
    const double *slot_radius;      /* [call slots] each slot's search radius */
};
template <typename Spec>
static bool mode_d_exchange(ModeDCallLease &lease, const neighbor_loop_args& args, const double *radii,
                            const typename Spec::CallScalars& cs, const typename Spec::DeviceContext& ctx,
                            unsigned int type_mask, typename Spec::AccumData *accums_out,
                            const ModeDDeviceQueries *device_queries = nullptr);

template <typename Spec>
static void mode_b_remote_evaluate_into_buffer(
    const neighbor_loop_args& args,
    const double *radii,
    const typename Spec::CallScalars& cs,
    const typename Spec::DeviceContext& ctx,         /* caller-owned */
    unsigned int neighbor_type_mask,                  /* explicit caller param */
    typename Spec::AccumData *accums_out)             /* size = args.num_active; caller-owned */
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

    /* The queries. One snapshot serves both this rank's own pairs and the copies
     * shipped to peers, so neither can see a different epoch from the other.  An
     * active is already current when a neighbour loop is entered: the sync point
     * drifts every particle in every active time bin to All.Ti_Current
     * (core/run.cc), ActiveParticleList is built from those same bins, and every
     * Spec draws its active list from that list.  Nothing advances All.Ti_Current
     * again inside the step. */
    std::vector<ActiveData> actives_host;
    ActiveData *actives = nullptr;
    if(N > 0) {
        actives_host.resize(N);
        build_self_actives_host_pre_drift<Spec>(args, ctx, radii, cs,
                                                  actives_host.data());
        actives = actives_host.data();
    }

    /* ---- The self stages run ONCE, BEFORE the peer round loop: self candidate
     * collection + drift + evaluation precede peer evaluation, and only the PEER
     * work streams in bounded rounds below.  That order describes this path; it
     * is not a contract other paths keep (Mode A and Mode D evaluate local and
     * received queries together).
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

    /* Every Mode-B loop routes via targeted export: the per-type node band is
     * cross-rank-fresh and dominates every radius_policy (mode_b_local_walker.h). */
    const double jscale = nlr_spec_symmetric_j_radius_scale<Spec>();

    /* Targeted-export reverse map: topnode indices are stable between builds →
     * build ONCE, reuse for the fused walk. */
    ModeBTopleafMap topleaf_map;   /* shared read-only map, built once */
    ModeBExportSink export_sink;   /* per-query export sink (write-only during the walk) */
    if(N > 0 && nt > 1) { export_sink.ensure_size(nt); topleaf_map.build(); }

    /* Fused-walk export CSR (targeted specs): per active, its per-peer export
     * node-lists, staged ONCE by the fused self walk and marshalled (no second
     * walk) by the peer round loop below. Active-ordered (csr_rec_off), peer-ascending
     * within an active, node order = walk append order. Sized O(total targeted
     * exports), NOT O(NTask*N). */
    struct FusedExportRec { int peer; int node_off; int n_nodes; };
    std::vector<int> csr_rec_off;                    /* size N+1: active aa -> [off[aa],off[aa+1]) recs */
    std::vector<FusedExportRec> csr_recs;
    std::vector<int> csr_nodes;

    const int modeb_nthreads = nlr_modeb_omp_nthreads();

    /* Stage 3: collect SELF candidates PRE-DRIFT. For targeted specs this is the
     * FUSED legacy-mode==0 walk — candidates + export CSR in ONE traversal, keyed
     * on the frozen actives[] snapshot (== the query the receiver walks) so the
     * candidate / export / receiver walks share one query SSOT. Broadcast specs
     * keep the plain candidate walk (they have no export walk to fuse). */
    std::vector<std::vector<int>> cand_self_tree;
    if(N > 0) {
                if(nt > 1) {
            /* One walk per active collects its candidates and its export CSR together. */
            cand_self_tree.assign(N, std::vector<int>{});
            csr_rec_off.assign(N + 1, 0);
            /* Thread the fused self walk above the work threshold. Each thread
             * walks its actives into its OWN export sink + its OWN CSR segment
             * (no shared push, no lock); a serial prefix-sum then assembles the
             * active-ordered CSR BYTE-IDENTICALLY to the serial build
             * (per-active walk order fixed; peers ascending within an active;
             * node order = walk append order).
             *
             * The threshold is the only thing that decides this. Whether the
             * walk also collects candidates does not: it changes what the walk
             * records, not how its actives divide between threads, and the
             * per-active work is the traversal either way. This test once also
             * required candidates, back when a walk without them was a
             * validation pass whose speed was irrelevant; a walk without them
             * is now the production path that exports to peers, and it has the
             * most actives of any of them. */
            const bool use_omp_self = nlr_modeb_use_omp(N, modeb_nthreads);
            if(use_omp_self) {
                struct AaMeta { int tid; int rec_off; int n_recs; int node_off; int n_nodes; };
                std::vector<AaMeta> meta(N);
                std::vector<ModeBExportSink> tsink(modeb_nthreads);
                for(auto& s : tsink) s.ensure_size(nt);
                std::vector<std::vector<FusedExportRec>> trecs(modeb_nthreads);
                std::vector<std::vector<int>> tnodes(modeb_nthreads);
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
                    m.n_recs = 0; m.n_nodes = 0;
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
                                            cand_ptr, topleaf_map, sink, jscale);
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
                }
                csr_rec_off[N] = rec_cursor;
            } else {
                for(int aa = 0; aa < N; aa++) {
                    csr_rec_off[aa] = (int)csr_recs.size();
                    const double h_q = (double)actives[aa].h_search;
                    if(h_q <= 0) continue;
                    double pos_arr[3] = {(double)actives[aa].pos[0],
                                         (double)actives[aa].pos[1],
                                         (double)actives[aa].pos[2]};
                    export_sink.clear_all();
                    std::vector<int>* cand_ptr = &cand_self_tree[aa];
                    if(cand_ptr->capacity() == 0) cand_ptr->reserve(64);
                    mode_b_walk_and_export(pos_arr, h_q, neighbor_type_mask,
                                            Spec::search_mode, Spec::radius_policy,
                                            cand_ptr, topleaf_map, export_sink, jscale);
                    /* stage this active's per-peer exports into the CSR */
                    for(int p = 0; p < nt; p++) {
                        if(p == rank) continue;
                        const std::vector<int>& nodes = export_sink.nodes_per_peer[p];
                        const int nn = (int)nodes.size();
                        if(nn == 0) continue;
                        FusedExportRec rec;
                        rec.peer = p; rec.node_off = (int)csr_nodes.size(); rec.n_nodes = nn;
                        csr_recs.push_back(rec);
                        csr_nodes.insert(csr_nodes.end(), nodes.begin(), nodes.end());
                    }
                }
                csr_rec_off[N] = (int)csr_recs.size();
            }
        } else {
            /* single rank: no peers to export to → plain candidate walk. */
            collect_candidates_pre_drift<Spec>(args, radii,
                                                neighbor_type_mask,
                                                DispatchPath::ModeB_HostWalker,
                                                cand_self_tree);
        }
    }

    /* Self drift (split from the former self+peer union drift; peer candidates
     * drift per-round below). drift_particle is idempotent to All.Ti_Current
     * (constant across the helper), so a j that is both a self- and peer-
     * candidate drifts once — identical to the old combined union drift. */
    {
        /* The same query the walk above used: the frozen actives[] on the multi-rank fused walk, the
         * particle and its radius on the single-rank walk. */
        if (N > 0) drift_and_filter_candidates<Spec>(cand_self_tree, neighbor_type_mask, [&](int aa, double *pos, double &h_q) {
            if(nt > 1) {
                pos[0] = (double)actives[aa].pos[0]; pos[1] = (double)actives[aa].pos[1]; pos[2] = (double)actives[aa].pos[2];
                h_q = (double)actives[aa].h_search;
            } else {
                const int i = args.active_list[aa];
                pos[0] = (double)args.P[i].Pos[0]; pos[1] = (double)args.P[i].Pos[1]; pos[2] = (double)args.P[i].Pos[2];
                h_q = radii[aa];
            }
        });
    }

    /* Stage 8: answer THIS rank's own queries -> accums_out, from the list collected above. */
    if(N > 0) {
        evaluate_pairs_post_drift<Spec>(ctx, actives, N,
                                          cand_self_tree, accums_out, cs, EvalOMPPolicy::AllowProduction);
    }

    /* ---- PEER round loop (streaming). Build a CommChunkSize-bounded batch of
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
     * peers. Self-pair handled above; self entry stays empty. */
    /* jscale and the exporter are hoisted above Stage 3 for the fused walk. */
    /* How much this round may carry: one communication chunk, divided by the size of a
     * query and its reply. NlrQueryEnvelope fuses what used to be sent as separate index,
     * node-list and active records, so counting envelopes here counts what the older code
     * counted as export entries. All.CommChunkSize is the parameterfile chunk size. */
    constexpr size_t kReplyBytes = sizeof(ReplyEnvelope);
    const long long kEnvPairBytes = (long long)sizeof(Envelope) + (long long)kReplyBytes;
    long long bunch = ((long long)All.CommChunkSize * 1024 * 1024) /
                      (kEnvPairBytes > 0 ? kEnvPairBytes : 1);
    if(bunch < 1) bunch = 1;

    long long diag_export_qr = 0, diag_node_appends = 0;   /* scalar export volume (NLR diag) */
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
                        "~%lld bytes) exceeds CommChunkSize bunch (%lld envelopes); shipping a solo "
                        "oversized round — cap ineffective for this call (raise CommChunkSize).",
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
            cursor = N;   /* nothing to export (N==0 or single rank) */
        }

        /* Stage 4: exchange queries. Every rank participates even if it queued
         * 0 this round (peers may target this rank's pool); the Allreduce(ndone)
         * at the round's end keeps every rank's round count equal, so the
         * exchange stays balanced. begin() posts ALL query-payload Isends + ALL
         * reply Irecvs up front; the receiver then stages, evaluates, and
         * answers incoming queries in memory-bounded WHOLE-PEER groups (the
         * legacy import sub-chunk loop, code_block_xchange_perform_ops.h:96-174)
         * instead of materializing every peer's payload at once. Group budget =
         * the same All.CommChunkSize that sizes the sender bunch. The bound covers
         * TRANSPORT payloads only (envelopes + replies) — candidate vectors and
         * pair-kernel scratch scale with the group's query content, not with
         * NTask. Peers are consumed in ascending rank order, so the
         * concatenated per-group evaluation sequence — and the post-loop reply
         * merge — keep the exact pre-group order (matters for j-writing specs). */
        using XReply = ReplyEnvelope;
        ModeBBoundedExchange<Envelope, XReply> xch;
        {
            xch.begin(queries_per_peer);
        }
        const size_t group_budget_bytes =
            (size_t)((double)All.CommChunkSize * 1024.0 * 1024.0);

        std::vector<int> group_peers;
        std::vector<std::vector<Envelope>> group_queries;
        while(true) {
            bool have_group;
            {
                have_group = xch.next_group(group_budget_bytes, group_peers, group_queries);
            }
            if(!have_group) break;

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

    /* Stages 6, 7 and 9 -- find this round's peer neighbours, bring them current,
     * and evaluate -- are one backend operation.  See NlrPeerAnswerHostWalk. */
    const int K = (int)peer_actives.size();
    std::vector<AccumData> peer_replies(K);
    NlrPeerAnswerHostWalk<Spec>::answer(ctx, cs, peer_actives,
                                        peer_nodelist_flat, peer_nnodes,
                                        neighbor_type_mask,
                                        peer_replies);
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
        /* send_group_replies posts the reply Isends and waits them in finish();
         * move the group's reply buffers into the exchange so they outlive the Isend. */
        xch.send_group_replies(group_peers, std::move(replies_for_group));
    }
        }   /* end whole-peer group loop */

        /* Stage 11: drain the exchange (remaining query-payload Isends + all
         * pre-posted reply Irecvs, byte-count-asserted), then merge replies
         * into accums_out by envelope.origin_slot. Pinned deterministic order:
         * ascending peer rank, ascending qi — identical to the pre-group
         * single-shot merge (the local contribution is already in accums_out:
         * either the group loop above placed it, or the flush just did).
         * Asserts each reply envelope's origin_rank == ThisTask. */
        {
            auto recv_replies = [&]{
                return xch.finish();
            }();
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
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    } while(ndone < NTask);



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
static void run_mode_b_remote_impl(const neighbor_loop_args& args, const double *radii)
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
    /* ctx.populate_failed is deliberately NOT honoured on this multi-rank leg:
     * returning here would desync the query/reply transport the peers enter
     * below, and answering their queries needs the very buffers that failed.
     * What makes that safe is that the readers tolerate an absent buffer --
     * a Spec whose load_neighbor indexes a hook-owned array must test it
     * first (sink_feed's binary_merge_eligible is the worked example, and it
     * is sized by num_total rather than by the active count, so it is not
     * small). The stop is already requested with the buffer named, and the
     * loop contributes nothing further. */

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
                                                       (N > 0) ? accums_self.data() : nullptr);

    /* Stage 12 final: writeback per active. */
    {
        for(int aa = 0; aa < N; aa++) {
            Spec::apply_active_writeback(args, aa, args.active_list[aa], accums_self[aa]);
        }
    }

}

template <typename Spec>
static void run_mode_b_remote(const neighbor_loop_args& args, const double *radii) {
    run_mode_b_remote_impl<Spec>(args, radii);
}

/* Whether a loop can be answered by the fused device walk at all: a one-way
 * search whose pair kernel may run concurrently, and whose active rebinding
 * the device evaluation does not perform.  Compile-time, so the machinery is
 * instantiated only for loops that can reach it. */
template <typename Spec>
constexpr bool nlr_spec_mode_d_eligible_v =
    (Spec::search_mode == MODE_B_SEARCH_ONEWAY) &&
    (nlr_spec_modeb_eval_omp<Spec>() != ModeBEvalOMP::SerialOnly) &&
    !nlr_spec_has_bind_active_to_eval_context_v<Spec>;

/* Mode D for a single-pass loop: the same transport and the same device walk
 * the iterative driver uses, without the per-iteration compaction.  The
 * particles are read from the device-resident arena, as on Mode A, because
 * that is where the walk evaluates them; the tree was prepared and voted on at
 * the dispatch site, once, before this is entered. */
template <typename Spec>
static void run_mode_d(const neighbor_loop_args& args, const double *radii, ModeDCallLease &lease,
                       const GxDeviceTreeView& tree)
{
    using AccumData = typename Spec::AccumData;
    using DeviceCtx = typename Spec::DeviceContext;

    const int N = args.num_active;

    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total, args.P, args.CellP);

    DeviceCtx ctx;
    ctx.P         = gpu_particles_arena_P();
    ctx.CellP     = (args.CellP != nullptr) ? gpu_particles_arena_CellP() : nullptr;
    ctx.num_total = args.num_total;
    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(args, ctx);
    }

    std::vector<AccumData> accums(N);
    for(int aa = 0; aa < N; aa++) {Spec::zero_accum(accums[aa]);}

    bool evaluated = true, stopped = false;
    {
        /* The guard releases the Spec's context after the writeback below, the
         * order the other wrappers keep. */
        NlrDeviceContextCleanupGuard<Spec> _nlr_dctx_cleanup_guard(args, ctx);
        /* A hook that could not stage its buffers has already asked for the
         * stop. Alone, this rank enters no collective, so it returns here as
         * Mode A does and the request drains at the caller's next poll. With
         * peers it proceeds, for the reason the host transport states at its
         * own population site: returning would desync the query/reply exchange
         * the peers enter, and every reader of a hook-owned buffer tests for
         * its absence first. */
        if(ctx.populate_failed && NTask == 1) {
            gpu_particles_arena_mark_clean_after_scatter(Spec::loop_name);
            return;
        }
        if(NTask > 1) {
            /* Every rank enters the exchange, with or without actives of its own. */
            stopped = mode_d_exchange<Spec>(lease, args, radii, cs, ctx, nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                                  (N > 0) ? accums.data() : nullptr);
        } else if(N > 0) {
            /* Each entry's slot into the call-level staging: carried by a
             * narrowed list, its position otherwise. */
            std::vector<int> slots(N);
            for(int aa = 0; aa < N; aa++) {slots[aa] = args.active_call_slot ? args.active_call_slot[aa] : aa;}
            evaluated = nlr_mode_d_evaluate_single_rank<Spec>(
                ctx, cs, args.active_list, slots.data(), radii, N,
                nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                tree, accums.data());
        }
        /* An exchange that ended in an agreed stop leaves no result to write back. */
        if(evaluated && !stopped) {
            for(int aa = 0; aa < N; aa++) {
                Spec::apply_active_writeback(args, aa, args.active_list[aa], accums[aa]);
            }
        }
    }
    gpu_particles_arena_mark_clean_after_scatter(Spec::loop_name);

    /* Out of device memory is a reason to answer differently, never to answer
     * less: the host walker takes the call.  Only the single-rank shape can get
     * here -- with peers the transport above has already been entered -- so
     * the change of backend is invisible to every other rank. */
    if(!evaluated) {run_mode_b_local<Spec>(args, radii);}
}

/* ============================================================================
 * Caller-built list helpers (corridor support; see struct nlr_external_csr).
 *
 * When a loop is handed a caller-built list (args.external_csr, or
 * NlrSubgroup::external_csr), Mode A skips gpu_ngb_list_build and instead
 * stages the caller's host CSR into Kokkos memory shaped like a
 * gpu_neighbor_list_t -- so the rest of the Mode-A path is unchanged. Only
 * the build site (replaced with this helper) differs between the two paths.
 *
 * The runner OWNS the SharedSpace/DeviceSpace allocations made here and
 * frees them with gpu_ngb_list_free, as for a built list. It does NOT free the caller's
 * host buffers (active_indices / offsets / neighbors). Contract: caller
 * keeps host CSR alive for the duration of every run_neighbor_loop call
 * that injects it; a corridor owns its lists across all of its consumers. */
/* Exhaustion is reported by returning NULL, so a caller NULL-check can
 * controlled-stop with attribution (the label appears in the memory ledger)
 * instead of a hard terminate. */
static void *nlr_shared_alloc_bytes(size_t bytes, const char *label)
{
    if(bytes == 0) { return NULL; }
    return gizmo_gpu_alloc_shared(bytes, label);
}

/* One wording for every runner staging buffer that could not be had. The stop is
 * drained at the next phase boundary; until then the loop simply leaves its actives
 * untouched. Nothing here returns early past a collective — the ghost-writeback
 * pair around the kernel work still fires, so no peer rank is left waiting. */
static void nlr_stop_no_staging_memory(const char *loop_name, const char *what, size_t bytes)
{
    char msg[256];
    snprintf(msg, sizeof(msg),
             "%s: could not allocate %s (%.1f MB); this loop leaves its actives untouched",
             loop_name ? loop_name : "neighbour loop", what,
             (double) bytes / (1024.0 * 1024.0));
    gizmo_request_controlled_stop(7713, msg, __FILE__, __LINE__, __FUNCTION__);
}

/* Whether a caller-built list answers this loop's search (contract on struct
 * nlr_external_csr).  `rows`, `radii` and `type_mask` are the loop's own: its
 * active list, the search radius it gives each row, its effective neighbour-type
 * mask.  Returns nullptr when the list may be used, else the reason, with the
 * stop code in *code.  O(rows), no collective. */
template <typename Spec>
static const char *
nlr_external_csr_mismatch(const nlr_external_csr *ec, const int *rows, int n,
                          const double *radii, unsigned int type_mask, int *code)
{
    constexpr bool wider_ok = nlr_spec_accepts_wider_neighbor_list_v<Spec>;
    const double scale = nlr_spec_symmetric_j_radius_scale<Spec>();
    *code = 7300;
    if(ec->type_mask != type_mask)                    {return "built for other neighbour types";}
    if(ec->search_mode != Spec::search_mode)          {return "built for another search mode";}
    if(nlr_radius_policy_for_mask(ec->radius_policy, type_mask) !=
       nlr_radius_policy_for_mask(Spec::radius_policy, type_mask)) {return "built under another radius policy";}
    if(wider_ok ? !(ec->neighbor_radius_scale >= scale)
                : !(ec->neighbor_radius_scale == scale)) {return "built with another neighbour-radius scale";}
    if(ec->num_active != n)                           {*code = 7301; return "num_active mismatch";}
    if(!ec->active_indices)                           {*code = 7302; return "null active_indices";}
    if(!ec->query_radii)                              {*code = 7302; return "null query_radii";}
    if(!ec->offsets)                                  {*code = 7303; return "null offsets";}
    if(ec->total_pairs < 0)                           {*code = 7304; return "negative total_pairs";}
    if(ec->total_pairs > 0 && !ec->neighbors)         {*code = 7305; return "null neighbors with total_pairs>0";}
    if(n > 0 && ec->offsets[0] != 0)                  {*code = 7306; return "offsets[0]!=0";}
    if(n > 0 && ec->offsets[n] != ec->total_pairs)    {*code = 7307; return "offsets[N]!=total_pairs";}
    for(int aa = 0; aa < n; aa++) {
        if(ec->active_indices[aa] != rows[aa])        {*code = 7308; return "row order != active_list";}
        if(ec->offsets[aa+1] < ec->offsets[aa])       {*code = 7309; return "non-monotonic offsets";}
        if(wider_ok ? !(radii[aa] <= ec->query_radii[aa])
                    : !(radii[aa] == ec->query_radii[aa])) {*code = 7300; return "a row was searched with another radius";}
    }
    if(NTask > 1 && ec->pool_epoch != ghost_provenance_epoch()) {*code = 7313; return "built against another ghost import";}
    return nullptr;
}

static inline bool
nlr_stage_external_csr_into_gnl(const nlr_external_csr *ext,
                                gpu_neighbor_list_t *gnl,
                                const char *loop_name)
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
    gnl->offsets  = (int64_t *) nlr_shared_alloc_bytes(off_bytes, "modea_csr_offsets");
    gnl->d_active = (int *)     nlr_shared_alloc_bytes(act_bytes, "modea_active");
    /* neighbors in DeviceSpace (GPU HBM) — must deep_copy from host view */
    gnl->neighbors = (int *) gizmo_gpu_alloc_device(nbr_bytes, "modea_csr_neighbors");
    /* Report the refusal rather than staging into null: the memcpy below writes
     * the caller's whole CSR, and the pair kernel reads all three. The caller
     * releases what did land and stops. */
    if(!gnl->offsets || !gnl->d_active || !gnl->neighbors) {
        nlr_stop_no_staging_memory(loop_name, "the injected neighbour list",
                                   off_bytes + act_bytes + nbr_bytes);
        return false;
    }
    memcpy(gnl->offsets,  ext->offsets,          off_bytes);
    memcpy(gnl->d_active, ext->active_indices,   act_bytes);

    if(pairs > 0) {
        Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            h_n(ext->neighbors, (size_t)pairs);
        Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            d_n(gnl->neighbors, (size_t)pairs);
        Kokkos::deep_copy(d_n, h_n);
    }
    return true;
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
 * Caller-built list (args.external_csr != nullptr): a corridor has already
 * built the list; once nlr_external_csr_mismatch() finds it answers this
 * loop's search, it is staged into the gnl shape via
 * nlr_stage_external_csr_into_gnl() instead of calling gpu_ngb_list_build.
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

/* The staged i-side ActiveData array is written by the staging kernel and read
 * by the pair kernel, both on the device; no host code reads it. Keeping it in
 * device memory therefore costs nothing in reachability and keeps the staging
 * kernel's writes off the migratable shared path, which on a discrete-memory
 * device is where that kernel's cost lives.
 *
 * The accumulator array is NOT eligible and must stay shared: the host reads it
 * directly, in apply_active_writeback on the single-pass path and in the scatter
 * back into the driver's per-slot accumulators on the iterative path.
 *
 * On a host-only backend the two spaces are the same type, so this is a no-op
 * there by construction -- a local run can show that nothing regressed, it
 * cannot exercise the separation. Same non-throwing contract as above. */
static void *nlr_active_stage_alloc_bytes(size_t bytes, const char *label)
{
    if(bytes == 0) { return NULL; }
    return gizmo_gpu_alloc_device(bytes, label);
}

static void nlr_active_stage_free(void *p)
{
    if(p == NULL) { return; }
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(p);
}

/* ============================================================================
 * TeamRowReduce support: combining lanes' partial accumulators.
 *
 * A Kokkos reducer over Spec::AccumData. Its join IS Spec::merge_accum and its
 * init IS Spec::zero_accum, so the lane combination and the cross-rank Mode-B
 * reply merge are the same algebra by construction rather than by agreement --
 * there is no second copy of the merge rules to drift.
 *
 * This rests on zero_accum being the identity element of merge_accum, which is
 * already required and already holds: Mode-B starts each peer's accumulator from
 * zero_accum before merging, so the pair is a monoid tree-wide. The two cases
 * that could have failed both check out -- DensitySpec::zero_accum sets its
 * sink fields to explicit MIN-reduction sentinels rather than zero, and
 * GradientsSpec byte-zeroes where 0 is a true identity because its Maxima and
 * Minima range over signed deltas, a set that contains the self-delta.
 *
 * join takes the whole accumulator. Decomposing it field-wise would be wrong,
 * not merely slower: SinkFeedSpec merges a coupled minimum-with-payload, taking
 * the peer's position only when the peer's potential is lower, and independent
 * per-field reduction would pair a winning value with a losing payload.
 * ========================================================================== */
template <typename Spec>
struct NlrAccumReducer {
    using reducer          = NlrAccumReducer<Spec>;
    using value_type       = typename Spec::AccumData;
    using result_view_type = Kokkos::View<value_type, Kokkos::AnonymousSpace,
                                          Kokkos::MemoryUnmanaged>;

    /* The lane combination moves an accumulator between threads with a shuffle,
     * and a shuffle of anything larger than one machine word is performed as a
     * whole number of INT-SIZED PIECES: the backends compute that count as
     * sizeof(T) / sizeof(int), truncating. An accumulator whose size is not a
     * whole number of those pieces therefore loses its trailing bytes in the
     * reduction -- silently, with no diagnostic, and only on the device, where
     * whichever field happens to sit last would come back holding whatever the
     * receiving lane had there before.
     *
     * That is a bad failure to leave discoverable only by a wrong answer, so it
     * is a build error instead. If a new accumulator trips this, pad it to a
     * multiple of sizeof(int) rather than reordering its fields to hide the
     * tail -- the next field added would put it back. An accumulator with no
     * members at all is exempt because it carries nothing to lose. */
    static_assert(std::is_empty<value_type>::value ||
                  (sizeof(value_type) % sizeof(int)) == 0,
                  "Spec::AccumData must be a whole number of int-sized pieces: the device lane "
                  "reduction shuffles it as sizeof(AccumData)/sizeof(int) words and would drop "
                  "the remainder. Pad the struct to a multiple of sizeof(int).");

    KOKKOS_INLINE_FUNCTION explicit NlrAccumReducer(value_type& v) : m_value(v) {}

    KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
        Spec::merge_accum(dst, src);
    }
    KOKKOS_INLINE_FUNCTION void init(value_type& v) const { Spec::zero_accum(v); }
    KOKKOS_INLINE_FUNCTION value_type&       reference()         const { return m_value; }
    KOKKOS_INLINE_FUNCTION result_view_type  view()              const { return result_view_type(&m_value); }
    KOKKOS_INLINE_FUNCTION bool              references_scalar() const { return true; }

private:
    value_type& m_value;
};

/* The TeamRowReduce pair kernel, shared by both Mode-A dispatch sites.
 *
 * A named functor rather than a lambda for two reasons. It can be instantiated
 * once up front to size the team and again per chunk to run, which keeps the
 * occupancy query out of the chunk loop; and nvcc forbids defining an extended
 * device lambda inside another lambda, which a per-chunk lambda factory would
 * require.
 *
 * The two sites differ only in how a work item reaches its CSR row. The
 * single-pass site walks the active list directly, so row = chunk_base + i. The
 * iterative site walks a compacted active set into a build-time row index, so
 * row = csr_lookup[active_set[i]]. Passing active_set == nullptr selects the
 * former. Both index the staged actives and accumulators by i.
 */
template <typename Spec, typename DeviceCtx>
struct NlrModeATeamPairKernel {
    using ActiveData   = typename Spec::ActiveData;
    using AccumData    = typename Spec::AccumData;
    using ScatterData  = typename Spec::ScatterData;
    using NeighborData = typename Spec::NeighborData;
    using TeamMember   = typename Kokkos::TeamPolicy<>::member_type;

    DeviceCtx      ctx;
    ActiveData    *d_actives;
    AccumData     *d_accums;
    const int64_t *offsets;
    const int     *neighbors;
    const int     *active_set;   /* nullptr on the single-pass site */
    const int     *csr_lookup;   /* used only when active_set != nullptr */
    int            chunk_base;   /* used only when active_set == nullptr */
    /* One snapshot for the whole call, held by value so the device functor
     * carries it without reaching for a global. */
    typename Spec::CallScalars cs;
    struct GxMotionTargetSet   motion_targets;   /* for a loop that writes neighbour motion */

    KOKKOS_INLINE_FUNCTION void operator()(const TeamMember& team) const {
        const int i   = team.league_rank();
        const int row = (active_set != nullptr) ? csr_lookup[active_set[i]]
                                                : chunk_base + i;
        const ActiveData& a = d_actives[i];
        const int64_t start = offsets[row], end = offsets[row + 1];

        /* An empty or malformed row must produce the zero accumulator and
         * nothing else. The serial walk got this for free -- `for(nn = start;
         * nn < end; ...)` simply runs zero times when end <= start -- but a
         * team range takes a COUNT, and a non-positive one is not something to
         * hand it. Restoring the property explicitly keeps the two assignments
         * equivalent on degenerate rows as well as ordinary ones. */
        const int row_len = (end > start) ? (int)(end - start) : 0;
        if(row_len == 0) {
            Kokkos::single(Kokkos::PerTeam(team), [&]() { Spec::zero_accum(d_accums[i]); });
            return;
        }

        AccumData row_accum;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, row_len),
            [&](int nn, AccumData& lane_accum) {
                ScatterData     s{};
                IdentitySidecar id{};
                if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_mark(motion_targets, neighbors[start + nn]);}
                NeighborData    nb = Spec::load_neighbor(ctx, neighbors[start + nn], id, a);
                Spec::pair_kernel(a, nb, lane_accum, s, cs);
            },
            NlrAccumReducer<Spec>(row_accum));

        /* Every lane leaves the reduction holding the combined value; one
         * publishes it. */
        Kokkos::single(Kokkos::PerTeam(team), [&]() { d_accums[i] = row_accum; });
    }
};

/* Lanes per row for a TeamRowReduce Spec, or 1 to select the flat kernel.
 *
 * Keyed only on structural properties -- execution space, the Spec's assignment
 * policy, its search_mode, and sizeof(AccumData). No caller name appears here,
 * so every ONEWAY loop gets the one-way width and every SYMMETRIC loop the
 * symmetric one with no per-loop work, which is the whole point of resolving it
 * in one place.
 *
 * The Kokkos bound is applied last and is a LEGALITY clamp: it reports the
 * largest team the backend can launch for this functor, and does not shrink as
 * the reduction value grows. The accumulator size is handled separately, and
 * deliberately bluntly, by the fat-accumulator cap. */
template <typename Spec, typename Functor>
static int nlr_mode_a_team_width(const Functor& f)
{
    if constexpr (gizmo_gpu_default_space_is_host()) {
        return 1;
    } else if constexpr (nlr_mode_a_pair_policy<Spec>() == ModeAPairAssignment::RowSerial) {
        return 1;
    } else {
        /* Every input is fixed by the Spec and the kernel type, so this resolves
         * once per (Spec, kernel) rather than once per dispatch: the occupancy
         * query behind it is not worth repeating, and the answer cannot change
         * within a run. */
        static const int resolved = [&]() {
            int target;
            if(NUMDIMS < 3) {
                target = NLR_TEAM_WIDTH_LOWDIM;
            } else {
                target = (Spec::search_mode == MODE_B_SEARCH_ONEWAY)
                         ? NLR_TEAM_WIDTH_ONEWAY : NLR_TEAM_WIDTH_SYMMETRIC;
                if(sizeof(typename Spec::AccumData) > NLR_TEAM_FAT_ACCUM_BYTES) {
                    target = NLR_TEAM_WIDTH_FAT_ACCUM;
                }
            }
            /* A non-positive answer means the backend reports NO launchable
             * team size for this functor, not "no limit" -- fall back to the
             * flat kernel rather than launching at the full target. */
            const int hw = gizmo_gpu_team_size_max(f);
            if(hw <= 0)       { return 1; }
            if(hw < target)   { target = hw; }
            int w = 1;
            while((w << 1) <= target) { w <<= 1; }
            return w;
        }();
        return resolved;
    }
}

template <typename Spec>
static void run_mode_a(const neighbor_loop_args& args, const double *radii)
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
    const size_t radii_uvm_bytes = (size_t) N * sizeof(double);
    double *radii_uvm = (double *) nlr_shared_alloc_bytes(radii_uvm_bytes, "modea_radii");
    if(radii_uvm == NULL) {
        /* Nothing has been acquired yet, so there is nothing to release. */
        nlr_stop_no_staging_memory(Spec::loop_name, "the per-active search radii", radii_uvm_bytes);
        return;
    }
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
     * Caller-built list (args.external_csr non-null): stage it into gnl shape
     * and skip the build, once it is shown to answer this loop's search. */
    gpu_neighbor_list_t gnl;
    bool gnl_borrowed = false;   /* the caller's device list, read in place: never freed here */
    gpu_spatial_index_t *sidx = nlr_resolve_sidx_cache(Spec::sidx_cache_kind,
                                                       Spec::loop_name);
    if(args.external_csr != nullptr) {
        /* Checked at run time, every call, never diagnostic-gated: a list built
         * for another search hands the kernel a wrong neighbour set, and a row
         * order other than args.active_list applies one particle's sums to
         * another.  On a mismatch: soft bad-stop, release the arena and the
         * staged radii, return before staging or walking.  run_mode_a issues no
         * MPI, so the return cannot desync a peer; the next phase poll drains. */
        const nlr_external_csr *ec = args.external_csr;
        int         csr_code = 0;
        const char *csr_err  = nlr_external_csr_mismatch<Spec>(
            ec, args.active_list, N, radii,
            nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask), &csr_code);
        if(csr_err != nullptr) {
            fprintf(stderr, "[neighbor_loop_runner rank=%d caller=%s] external-CSR contract violation: %s\n",
                    ThisTask, Spec::loop_name, csr_err);
            fflush(stderr);
            endrun(csr_code);
            gpu_particles_arena_release();
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
            return;
        }
        const gpu_neighbor_list_t *dev = ec->device;
        if(dev && dev->num_active == ec->num_active && dev->total_pairs == ec->total_pairs && dev->offsets && dev->d_active
           && (dev->total_pairs == 0 || dev->neighbors)) {
            gnl = *dev;
            gnl_borrowed = true;
        } else if(!nlr_stage_external_csr_into_gnl(ec, &gnl, Spec::loop_name)) {
            gpu_ngb_list_free(&gnl);
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
            gpu_particles_arena_release();
            return;
        }
    } else {
        gpu_ngb_list_build(P_gpu, args.num_total,
                           args.active_list, N,
                           Spec::search_mode,
                           (int)nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                           &gnl, sidx,
                           1.0, radii_uvm, nullptr, Spec::loop_name,
                           nlr_spec_symmetric_j_radius_scale<Spec>(),
                           Spec::radius_policy);
    }

    /* A build that ran out of memory or refused its index hands back the empty list,
     * without the row index the kernel reads first, and has already asked for the stop. Release
     * this call's own buffers and return; no MPI has been issued here. */
    if(gnl.d_active == nullptr) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
        if(!gnl_borrowed) {gpu_ngb_list_free(&gnl);}
        gpu_particles_arena_release();
        return;
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

    /* Allocate chunk-sized ActiveData[] and AccumData[] arrays, both through
     * non-throwing allocators. They live in DIFFERENT spaces: the staged
     * actives are device-only (see nlr_active_stage_alloc_bytes), the
     * accumulators must stay host-readable for the writeback below. These are
     * the largest per-active transients (the demonstrated FIF OOM site); an
     * allocation failure here means the corresponding pool is genuinely full
     * (K is byte-capped), so controlled-stop with the buffer named in the
     * ledger rather than a hard terminate. run_mode_a issues no MPI, so the
     * request drains collectively at the caller's next phase poll (same as the
     * external-CSR contract-violation path above). */
    ActiveData *d_actives = (ActiveData *) nlr_active_stage_alloc_bytes((size_t)K * sizeof(ActiveData), "modea_active_data");
    AccumData  *d_accums  = (AccumData  *) nlr_shared_alloc_bytes((size_t)K * sizeof(AccumData),  "modea_accum_data");
    if(d_actives == NULL || d_accums == NULL) {
        if(d_accums)  { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums); }
        nlr_active_stage_free(d_actives);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
        if(!gnl_borrowed) {gpu_ngb_list_free(&gnl);}
        gpu_particles_arena_release();
        gizmo_request_controlled_stop(7710,
            "run_mode_a: Mode-A per-active staging out of memory (modea_active_data is device-resident, "
            "modea_accum_data is host-visible) -- reduce the active set; note that adding ranks per node "
            "does NOT relieve device memory, since the ranks on a node share it",
            __FILE__, __LINE__, __FUNCTION__);
        return;
    }

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
    /* A hook that could not stage its buffers has already asked for the stop.
     * Same exit as the staging failure above: release everything this call
     * obtained and return — this function issues no MPI, so the request drains
     * at the caller's next phase poll. The guard releases the hook's buffers. */
    if(ctx.populate_failed) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
        nlr_active_stage_free(d_actives);   /* device space, not shared -- see nlr_active_stage_alloc_bytes */
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
        if(!gnl_borrowed) {gpu_ngb_list_free(&gnl);}
        gpu_particles_arena_release();
        return;
    }

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

    /* Team width is a property of the Spec and the kernel type, not of a chunk,
     * so it is resolved once here rather than per chunk -- the occupancy query
     * behind it is not something to repeat inside the loop. */
    using TeamKernel = NlrModeATeamPairKernel<Spec, DeviceCtx>;
    int team_width = 1;
    if constexpr (nlr_mode_a_pair_policy<Spec>() == ModeAPairAssignment::TeamRowReduce) {
        TeamKernel probe{ctx, d_actives, d_accums, offsets, neighbors, nullptr, nullptr, 0, cs};
        team_width = nlr_mode_a_team_width<Spec>(probe);
    }

    for(int c0 = 0; c0 < N; c0 += K) {
        const int n = (N - c0 < K) ? (N - c0) : K;

        /* stage ActiveData for [c0, c0+n) into chunk-local [0,n). */
        gizmo_gpu_kernel_launch("nlr_stage_active", n, KOKKOS_LAMBDA(int kk) {
            const int aa = c0 + kk;
            d_actives[kk] = Spec::load_active(ctx, aa, d_active_idx[aa],
                                              radii_uvm[aa], cs);
        });

        /* pair-kernel over [c0, c0+n) — generic over Spec.
         *
         * Two assignments of the same physics. The flat form gives one work item
         * per active particle, which then walks its whole CSR row in sequence.
         * The team form gives one team per active particle whose lanes stride
         * the row together and combine through Spec::merge_accum. Which one runs
         * is decided by nlr_mode_a_team_width from structural properties only;
         * width 1 selects the flat form, and does so through if constexpr, so a
         * width-1 Spec compiles to exactly the kernel it compiled to before
         * teams existed rather than to a one-lane imitation of a team. */
        {
            const double t_pair_kernel_start = my_second();
            const struct GxMotionTargetSet motion_targets = gx_motion_target_view();

            auto flat_kernel = KOKKOS_LAMBDA(int kk) {
                /* Named here, unconditionally, so that the capture happens outside the
                 * `if constexpr` below: a device lambda may not first-capture a variable
                 * inside one, and a Spec that does not write neighbour motion would
                 * otherwise reach the capture only through the discarded branch. */
                const struct GxMotionTargetSet &targets = motion_targets;
                const int aa = c0 + kk;
                Spec::zero_accum(d_accums[kk]);
                const ActiveData& a = d_actives[kk];
                ScatterData s{};                     /* NoScatter for ActiveReduceOnly */
                int64_t start = offsets[aa], end = offsets[aa + 1];
                for(int64_t nn = start; nn < end; nn++) {
                    int j = neighbors[nn];
                    if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_mark(targets, j);}
                    IdentitySidecar id{};            /* NoIdentity */
                    NeighborData nb = Spec::load_neighbor(ctx, j, id, a);
                    Spec::pair_kernel(a, nb, d_accums[kk], s, cs);
                }
            };

            if constexpr (nlr_mode_a_pair_policy<Spec>() == ModeAPairAssignment::TeamRowReduce) {
                if(team_width > 1) {
                    TeamKernel fn{ctx, d_actives, d_accums, offsets, neighbors, nullptr, nullptr, c0, cs, motion_targets};
                    gizmo_gpu_team_kernel_launch(Spec::loop_name, n, team_width, fn);
                } else {
                    gizmo_gpu_kernel_launch(Spec::loop_name, n, flat_kernel);
                }
            } else {
                gizmo_gpu_kernel_launch(Spec::loop_name, n, flat_kernel);
            }
            cpu_charge_child(CPU_PAIR_KERNEL, timediff(t_pair_kernel_start, my_second()));
        }
        /* Launches fenced internally by gizmo_gpu_kernel_launch. UVM coherent ->
         * host reads d_accums[0,n) directly. */

        /* (4) host writeback for [c0, c0+n) before the next chunk reuses arrays. */
        {
            for(int kk = 0; kk < n; kk++) {
                Spec::apply_active_writeback(args, c0 + kk, args.active_list[c0 + kk], d_accums[kk]);
            }
        }
    }


    /* Cleanup. The list holds no index memory, so the cached index stays for
     * the next loop; on the external-CSR path the free releases only what we
     * staged (gnl offsets/neighbors/d_active), the caller owning its host CSR
     * and any device list it lent. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
    nlr_active_stage_free(d_actives);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm);
    if(!gnl_borrowed) {gpu_ngb_list_free(&gnl);}
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

/* The adaptive A/B/D rule, one definition.  Mode B by the Mode-B rule
 * (parameterfile NeighborLoopModeBThreshold{Sum,Max}, else the Spec's
 * modeb_threshold_{sum,max}, else the default for its search mode).  Otherwise Mode D when the loop can
 * be answered by the fused device walk, some rank has work, and both counts
 * meet the Mode-D lower bounds; otherwise Mode A.  Collective inputs, so every
 * rank answers alike.  Nothing here touches Mode D's state: a call answered by
 * A or B pays nothing for D.
 *
 * A call that meets the Mode-D bounds but whose busiest rank has fewer than MODE_D_MIN_ACTIVES_PER_RANK actives
 * goes to Mode B (see that constant), unless Mode B is switched off. */
/* Whether the parameter file leaves Mode B on for this loop: a threshold of 0 switches it off. */
template <typename Spec>
static bool nlr_mode_b_switched_on(void)
{
    int default_sum, default_max;
    nlr_default_modeb_thresholds(Spec::search_mode, &default_sum, &default_max);
    return gizmo_nlr_modeb_threshold_sum_for(nullptr, nlr_spec_threshold_sum<Spec>(default_sum)) > 0 &&
           gizmo_nlr_modeb_threshold_max_for(nullptr, nlr_spec_threshold_max<Spec>(default_max)) > 0;
}

/* The per-iteration reduction of a one-subgroup Mode D call: {summed actives, busiest rank's actives, stop code}, as one
 * element of three long longs, summed / maximum / maximum. */
static void nlr_count_triple_reduce(void *in, void *inout, int *len, MPI_Datatype *)
{
    const long long *a = (const long long *) in;
    long long *b = (long long *) inout;
    for(int k = 0; k < *len; k++, a += 3, b += 3) {b[0] += a[0]; if(a[1] > b[1]) {b[1] = a[1];} if(a[2] > b[2]) {b[2] = a[2];}}
}
static MPI_Datatype nlr_count_triple_type(void)
{
    static MPI_Datatype t = MPI_DATATYPE_NULL;
    if(t == MPI_DATATYPE_NULL) {MPI_Type_contiguous(3, MPI_LONG_LONG, &t); MPI_Type_commit(&t);}
    return t;
}
static MPI_Op nlr_count_triple_op(void)
{
    static MPI_Op op = MPI_OP_NULL;
    if(op == MPI_OP_NULL) {MPI_Op_create(nlr_count_triple_reduce, 1, &op);}
    return op;
}

/* Whether a Mode D call's busiest rank still holds MODE_D_MIN_ACTIVES_PER_RANK actives.  Asked when a call is routed,
 * and again after every iteration of an iterative Mode D call, whose converged actives drop out: a remainder below it
 * finishes on Mode B.  Collective input (a maximum over the ranks), so every rank answers alike.  The parameter file's
 * Mode D thresholds are asked only when the call is routed. */
static bool nlr_mode_d_feeds_device(int max_active)
{
    return max_active >= MODE_D_MIN_ACTIVES_PER_RANK;
}

template <typename Spec>
NlrMode nlr_select_mode(int sum_active, int max_active)
{
    int default_sum, default_max;
    nlr_default_modeb_thresholds(Spec::search_mode, &default_sum, &default_max);
    if(nlr_mode_b_rule(sum_active, max_active, nlr_spec_threshold_sum<Spec>(default_sum), nlr_spec_threshold_max<Spec>(default_max))) {
        return NlrMode::B;
    }
    if constexpr (nlr_spec_mode_d_eligible_v<Spec>) {
        int TDS = 0, TDM = 0;
        gizmo_nlr_moded_thresholds(&TDS, &TDM);
        if(sum_active > 0 && sum_active >= TDS && max_active >= TDM) {
            if(nlr_mode_d_feeds_device(max_active)) {return NlrMode::D;}
            /* Below the floor: Mode B, unless the parameter file has switched Mode B off, in
             * which case the call keeps its usual path. */
            if(nlr_mode_b_switched_on<Spec>()) {return NlrMode::B;}
        }
    }
    return NlrMode::A;
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
                gx_motion_target_set_armed(nlr_spec_writes_neighbour_motion_v<Spec> ? 1 : 0);
                Spec::ghost_writeback_end(args, plan);
                gx_motion_target_set_armed(0);
            }
        }
    }
}
/* ============================================================================
 * Public entry: run_neighbor_loop<Spec>
 * ========================================================================== */

/* ============================================================================
 * Mode D call admission: what a device-resident call holds, and the one owner of it.
 *
 * A Mode D call needs storage for its whole pipeline -- the queries it asks, its export decision, the
 * envelopes it packs, the replies it gets back, the messages it receives -- and every piece is sized from
 * quantities known before the call starts: the actives on this rank, the task count, the top-level leaves and
 * the communication chunk.  So it is decided and allocated before any work, and the call either has every
 * piece or is not a Mode D call: nothing on the exchange's path is allocated or declined in the middle of an
 * exchange, where one rank stopping would leave its peers waiting.
 *
 * One piece may grow after admission: the round arena, which holds what one round sends.  Admission gives it
 * room for one worst-case active; once a call has counted what its actives actually send, it tries to grow
 * the arena to that.  A growth refused keeps the arena as it was and the call simply runs more rounds --
 * every rank reconciles round counts at the round's end anyway -- so it changes neither the answer nor any
 * rank's path.
 *
 * Allocating is the test.  The device storage a call gives back is kept, at its high water, for the next call
 * (allocating and freeing it every call cost ~1 ms of a ~30 ms density call) and released with the tree, which
 * bounds what is kept to one tree's calls; the page-locked host staging is one fixed ring held for the run,
 * because pinning costs far too much to repeat per call (gx_mode_d_staging_acquire).  One task holds none of it:
 * the single-rank path keeps its own.
 * ========================================================================== */

/* The share of the device's free memory a lease may take: the rest stays for what the call
 * still allocates while it runs, so admission can never leave the call's own later
 * allocations to fail in the middle of an exchange. */
static constexpr double MODE_D_DEVICE_FREE_FRACTION = 0.8;

/* The sizes a Mode D call's storage is built from.  Every product is checked in 64-bit arithmetic before it is
 * used. */
struct ModeDWorkspacePlan {
    long long envelopes_per_round;           /* E: query envelopes one round may carry, the chunk / (query + reply) */
    long long worst_envelopes_per_active;    /* the most one active can need: a run per peer plus its node lists */
    long long arena_capacity_limit;          /* the most envelopes the round arena may ever hold: E, within the index range */
    long long arena_min_envelopes;           /* the arena at admission: room for one worst-case active */
    long long segment_records;               /* S: records of one staging message, either kind */
    long long device_bytes;                  /* what admission allocates on the device */
    long long iteration_actives;             /* actives whose iteration state the call keeps on the device (0: the host keeps it) */
    long long scratch_bytes;                 /* one active's iteration scratch */
};

/* a * b, or -1 when either is negative or the product leaves 64 bits. */
static long long mode_d_checked_product(long long a, long long b)
{
    if(a < 0 || b < 0) {return -1;}
    if(a != 0 && b > LLONG_MAX / a) {return -1;}
    return a * b;
}
/* a + b, or -1 when either is negative or the sum leaves 64 bits. */
static long long mode_d_checked_sum(long long a, long long b)
{
    if(a < 0 || b < 0 || b > LLONG_MAX - a) {return -1;}
    return a + b;
}

/* The most envelopes one active can need: it can open every remote top leaf, and each peer's
 * leaves go out in node lists of NODELISTLENGTH -- so the sum over the other tasks of their top
 * leaves, chunked.  Local top leaves send nothing, so on one task this is zero.  O(NTopleaves),
 * paid only by a call that reaches Mode D's admission.  -1 when its O(NTask) tally cannot be had. */
static long long mode_d_worst_envelopes_per_active(void)
{
    if(NTask <= 1 || NTopleaves <= 0) {return 0;}
    std::vector<long long> leaves_of_task;
    try {leaves_of_task.assign((size_t) NTask, 0);} catch(const std::bad_alloc &) {return -1;}
    for(int i = 0; i < NTopleaves; i++) {
        const int t = DomainTask[i];
        if(t >= 0 && t < NTask && t != ThisTask) {leaves_of_task[(size_t) t]++;}
    }
    long long worst = 0;
    for(int t = 0; t < NTask; t++) {worst += (leaves_of_task[(size_t) t] + NODELISTLENGTH - 1) / NODELISTLENGTH;}
    return worst;
}

/* Every message a Mode D call posts is one staging message at most, so MPI's int byte count always holds it. */
static_assert(GX_MODE_D_STAGING_MAX_MESSAGE_BYTES <= (size_t) INT_MAX,
              "a Mode D staging message must fit one MPI byte count");

/* The round arena: everything one round holds, carved out of one allocation for a capacity of C envelopes.  A
 * round's records number at most NODELISTLENGTH per envelope it may send, since an active's envelope bound is
 * never below its node count over NODELISTLENGTH. */
static constexpr int MODE_D_ARENA_ROWS = 8;
struct ModeDRoundArena {
    long long          capacity = 0;   /* C */
    char              *base = nullptr;
    long long          bytes = 0;
    ModeDExportRecord *records = nullptr, *records_spare = nullptr;   /* [NODELISTLENGTH C]: the round's records and the sort's other half */
    int               *scan = nullptr;                                /* [NODELISTLENGTH C]: the sort's scan, and each record's run start */
    int               *first_record = nullptr;                        /* [C]: where each envelope's records start */
    void              *envelopes = nullptr;                           /* [C]: the queries this round sends, in send order */
    int               *run_offsets = nullptr;                         /* [NODELISTLENGTH C]: each query's run list (2 (C + 1) used) */
    int               *envelope_of = nullptr;                         /* [NODELISTLENGTH C]: the envelope each sorted record goes in */
    void              *replies = nullptr;                             /* [C]: the answers to this round's queries, in send order */
};
/* The arena's row offsets for capacity c, and its total; -1 when a size leaves 64 bits. */
static long long mode_d_arena_layout(long long c, size_t query_bytes, size_t reply_bytes, long long offset[MODE_D_ARENA_ROWS])
{
    const long long rec = mode_d_checked_product(NODELISTLENGTH, c);
    const long long size[MODE_D_ARENA_ROWS] = {
        mode_d_checked_product(rec, (long long) sizeof(ModeDExportRecord)), mode_d_checked_product(rec, (long long) sizeof(ModeDExportRecord)),
        mode_d_checked_product(rec, (long long) sizeof(int)), mode_d_checked_product(c, (long long) sizeof(int)),
        mode_d_checked_product(c, (long long) query_bytes),
        mode_d_checked_product(rec, (long long) sizeof(int)), mode_d_checked_product(rec, (long long) sizeof(int)),
        mode_d_checked_product(c, (long long) reply_bytes)};
    long long total = 0;
    for(int k = 0; k < MODE_D_ARENA_ROWS; k++) {
        if(size[k] < 0) {return -1;}
        total = ((total + 63) / 64) * 64;
        offset[k] = total;
        total = mode_d_checked_sum(total, size[k]);
        if(total < 0) {return -1;}
    }
    return total;
}

/* One device buffer of a lease: where it is kept, how many bytes, and its memory-ledger label. */
struct ModeDLeaseBuffer { void **slot; long long bytes; const char *label; };
static constexpr int MODE_D_LEASE_BUFFERS = 22;
static constexpr int MODE_D_LEASE_GROUP_ROW = 18;   /* the first of the received batches' rows: queries 0, replies 0, queries 1, replies 1 */
/* The round's small device record: [0] range end, [1] [2] its record span, then two ints the sort and the cut keep on
 * the device -- a radix pass's split point and the envelope count -- each in one long long slot. */
enum {MODE_D_ROUND_END = 0, MODE_D_ROUND_FIRST_RECORD, MODE_D_ROUND_LAST_RECORD, MODE_D_ROUND_SPLIT, MODE_D_ROUND_ENVELOPES,
      MODE_D_ROUND_TOTAL_BOUND, MODE_D_ROUND_TOTAL_RECORDS, MODE_D_ROUND_INFO};   /* the last two: the export count's totals, read together */
/* The flags a call raises on the device and reads once per round. */
enum {MODE_D_FLAG_WALK = 0, MODE_D_FLAG_FILL, MODE_D_FLAG_MERGE, MODE_D_FLAGS};

/* The device storage the last calls gave back, kept for the next: each lease buffer by its row in the lease's table,
 * and the round arena as raw bytes (its rows are laid out again for the call that takes it, whose record sizes may
 * differ).  Released with the tree and at shutdown (nlr_mode_d_packet_workspace_release). */
struct ModeDLeasePool {
    void      *row[MODE_D_LEASE_BUFFERS] = {nullptr};
    long long  row_bytes[MODE_D_LEASE_BUFFERS] = {0};
    char      *arena = nullptr;
    long long  arena_bytes = 0;
};
static ModeDLeasePool g_mode_d_lease_pool;

/* Give a buffer to the pool's row k; whichever of it and what the row held is smaller is freed. */
static void mode_d_pool_keep(int k, void *ptr, long long bytes)
{
    ModeDLeasePool &pool = g_mode_d_lease_pool;
    if(!ptr) {return;}
    if(pool.row[k] && pool.row_bytes[k] >= bytes) {gizmo_gpu_free_device(ptr); return;}
    gizmo_gpu_free_device(pool.row[k]);
    pool.row[k] = ptr; pool.row_bytes[k] = bytes;
}
/* A buffer of at least `bytes` for row k: the pool's when it is large enough, else a new one (the pool's smaller
 * one is freed).  Null when the device refuses; *got is its size. */
static void *mode_d_pool_take(int k, long long bytes, const char *label, long long *got)
{
    ModeDLeasePool &pool = g_mode_d_lease_pool;
    if(pool.row[k] && pool.row_bytes[k] >= bytes) {
        void *p = pool.row[k]; *got = pool.row_bytes[k];
        pool.row[k] = nullptr; pool.row_bytes[k] = 0;
        return p;
    }
    gizmo_gpu_free_device(pool.row[k]); pool.row[k] = nullptr; pool.row_bytes[k] = 0;
    void *p = gizmo_gpu_alloc_device((size_t) bytes, label);
    *got = p ? bytes : 0;
    return p;
}
/* Device bytes a call must newly allocate beyond what the pool can hand it. */
static long long mode_d_pool_shortfall(const long long *row_bytes, long long arena_bytes)
{
    const ModeDLeasePool &pool = g_mode_d_lease_pool;
    long long fresh = 0;
    for(int k = 0; k < MODE_D_LEASE_BUFFERS; k++) {if(row_bytes[k] > pool.row_bytes[k]) {fresh = mode_d_checked_sum(fresh, row_bytes[k]);}}
    if(arena_bytes > pool.arena_bytes) {fresh = mode_d_checked_sum(fresh, arena_bytes);}
    return fresh;
}
static void mode_d_lease_pool_release(void)
{
    ModeDLeasePool &pool = g_mode_d_lease_pool;
    int any = (pool.arena != nullptr);
    for(int k = 0; k < MODE_D_LEASE_BUFFERS; k++) {any |= (pool.row[k] != nullptr);}
    if(!any) {return;}
    Kokkos::fence("mode_d_lease_pool_release");
    for(int k = 0; k < MODE_D_LEASE_BUFFERS; k++) {gizmo_gpu_free_device(pool.row[k]);}
    gizmo_gpu_free_device(pool.arena);
    pool = ModeDLeasePool{};
}

/* Everything a Mode D call holds, released together.  On any route out of the call -- the vote declining it,
 * an early return, its normal end -- the destructor gives back the storage and retires the touched-set epoch the
 * walk preparation opened.  A recorder left held would refuse every later epoch, which pins the fused walk to its
 * fallback for the rest of the run while looking exactly like the mechanism working.  The top-leaf map is not
 * the call's: it belongs to the tree (gx_top_leaf_map_*). */
struct ModeDCallLease {
    GxDeviceTreeView   tree{};
    GxTopLeafMapView   map{};
    int                epoch_held = 0;
    int                stopped = 0;   /* every rank agreed on a stop in an exchange of this call: no exchange follows */
    ModeDWorkspacePlan plan{};

    /* Device, per active of the call. */
    void      *query_snapshot       = nullptr;   /* [n] ActiveData: the query each active asks, frozen once per iteration */
    int       *active_index         = nullptr;   /* [n] the particle */
    int       *active_slot          = nullptr;   /* [n] its slot in the call's staging */
    double    *active_radius        = nullptr;   /* [n] its search radius */
    int       *active_nodes         = nullptr;   /* [n] remote top leaves it reaches */
    long long *active_node_scan     = nullptr;   /* [n + 1] where its export records start */
    long long *active_envelope_scan = nullptr;   /* [n + 1] its envelope bound, then the scan of the bounds */
    void      *active_accums        = nullptr;   /* [n] AccumData: its own result, then every reply merged in */
    /* Device, fixed. */
    int       *per_peer             = nullptr;   /* [NTask + 1] envelopes per peer this round */
    long long *round_info           = nullptr;   /* [MODE_D_ROUND_INFO] the round's range end and record span, and two device-side ints */
    int       *flags                = nullptr;   /* [MODE_D_FLAGS]: the tail of per_peer's buffer, read with the counts */
    void      *group_queries[2]     = {nullptr, nullptr};   /* [batch_capacity[m]] received queries, two batches in flight */
    void      *group_replies[2]     = {nullptr, nullptr};   /* [batch_capacity[m]] their answers */
    long long  batch_capacity[2]    = {0, 0};               /* S at admission; grown per round to what arrives (mode_d_batch_grow) */
    long long  row_bytes[MODE_D_LEASE_BUFFERS] = {0};       /* each buffer's size as held, for the pool on release */
    ModeDRoundArena arena;

    /* Device, per active of the call, when the call iterates on the device (plan.iteration_actives): each active's
     * particle, search radius, iteration scratch and latest result, by its slot in the call; the slots still iterating,
     * twice (this iteration's list and the next one's); a keep flag per listed slot, and the next list's length. */
    int       *iter_slot_index      = nullptr;
    double    *iter_slot_radius     = nullptr;
    void      *iter_slot_scratch    = nullptr;
    void      *iter_slot_accum      = nullptr;
    int       *iter_live[2]         = {nullptr, nullptr};
    int       *iter_keep            = nullptr;
    int       *iter_count           = nullptr;   /* [2]: the next list's length, the iteration's stop code */
    int        iter_on = 0, iter_which = 0, iter_n_live = 0;   /* host: whether the device holds the state, which list is current, its length */

    ModeDSegmentStream stream;   /* the exchange, reused for every round of the call */
    /* Host, reserved at admission so the exchange allocates nothing: per-peer counts and offsets of a round, and
     * the segments each of the two received batches holds. */
    struct BatchSegment {int source, index, records, offset, status;};
    std::vector<int>          peer_counts, peer_offsets;
    std::vector<BatchSegment> batch_segments[2];

    ModeDCallLease() = default;
    ModeDCallLease(const ModeDCallLease&) = delete;
    ModeDCallLease& operator=(const ModeDCallLease&) = delete;
    ~ModeDCallLease() {release();}

    /* The device buffers, once: the plan sums them, the allocation and the release walk them.  Sizes come from the
     * plan and the record sizes (a negative size means a product left 64 bits); the release needs only the slots. */
    void buffers(ModeDLeaseBuffer rows[MODE_D_LEASE_BUFFERS], long long num_active, size_t active_bytes, size_t accum_bytes,
                 size_t query_bytes, size_t reply_bytes)
    {
        /* One task exchanges nothing and holds none of this. */
        const long long nt = (NTask > 1) ? NTask : 0, n = nt ? num_active : 0, s = plan.segment_records;
        auto bytes = [](long long count, long long each) {return mode_d_checked_product(count, each);};
        const long long n1 = (n > 0) ? n + 1 : 0;
        int k = 0;
        rows[k++] = {&query_snapshot,                  bytes(n, (long long) active_bytes),        "ngl_moded_query_snapshot"};
        rows[k++] = {(void **) &active_index,          bytes(n, (long long) sizeof(int)),         "ngl_moded_active_index"};
        rows[k++] = {(void **) &active_slot,           bytes(n, (long long) sizeof(int)),         "ngl_moded_active_slot"};
        rows[k++] = {(void **) &active_radius,         bytes(n, (long long) sizeof(double)),      "ngl_moded_active_radius"};
        rows[k++] = {(void **) &active_nodes,          bytes(n, (long long) sizeof(int)),         "ngl_moded_active_nodes"};
        rows[k++] = {(void **) &active_node_scan,      bytes(n1, (long long) sizeof(long long)),  "ngl_moded_active_node_scan"};
        rows[k++] = {(void **) &active_envelope_scan,  bytes(n1, (long long) sizeof(long long)),  "ngl_moded_active_envelope_scan"};
        rows[k++] = {&active_accums,                   bytes(n, (long long) accum_bytes),         "ngl_moded_active_accums"};
        rows[k++] = {(void **) &per_peer,              bytes(nt ? nt + 1 + MODE_D_FLAGS : 0, (long long) sizeof(int)), "ngl_moded_per_peer"};
        rows[k++] = {(void **) &round_info,            bytes(nt ? MODE_D_ROUND_INFO : 0, (long long) sizeof(long long)), "ngl_moded_round_info"};
        const long long ni = nt ? plan.iteration_actives : 0;
        rows[k++] = {(void **) &iter_slot_index,       bytes(ni, (long long) sizeof(int)),        "ngl_moded_iter_index"};
        rows[k++] = {(void **) &iter_slot_radius,      bytes(ni, (long long) sizeof(double)),     "ngl_moded_iter_radius"};
        rows[k++] = {&iter_slot_scratch,               bytes(ni, plan.scratch_bytes),             "ngl_moded_iter_scratch"};
        rows[k++] = {&iter_slot_accum,                 bytes(ni, (long long) accum_bytes),        "ngl_moded_iter_accum"};
        rows[k++] = {(void **) &iter_live[0],          bytes(ni, (long long) sizeof(int)),        "ngl_moded_iter_live"};
        rows[k++] = {(void **) &iter_live[1],          bytes(ni, (long long) sizeof(int)),        "ngl_moded_iter_live"};
        rows[k++] = {(void **) &iter_keep,             bytes(ni, (long long) sizeof(int)),        "ngl_moded_iter_keep"};
        rows[k++] = {(void **) &iter_count,            bytes(ni > 0 ? 2 : 0, (long long) sizeof(int)), "ngl_moded_iter_count"};
        for(int m = 0; m < 2; m++) {
            rows[k++] = {&group_queries[m],            bytes(s, (long long) query_bytes),         "ngl_moded_group_queries"};
            rows[k++] = {&group_replies[m],            bytes(s, (long long) reply_bytes),         "ngl_moded_group_replies"};
        }
        static_assert(MODE_D_LEASE_GROUP_ROW + 4 == MODE_D_LEASE_BUFFERS, "the received batches are the table's last four rows");
    }

    /* The arena back to the pool (the larger of it and what the pool holds is kept). */
    void release_arena()
    {
        ModeDLeasePool &pool = g_mode_d_lease_pool;
        if(arena.base) {
            Kokkos::fence("mode_d_arena_release");
            if(pool.arena && pool.arena_bytes >= arena.bytes) {gizmo_gpu_free_device(arena.base);}
            else {gizmo_gpu_free_device(pool.arena); pool.arena = arena.base; pool.arena_bytes = arena.bytes;}
        }
        arena = ModeDRoundArena{};
    }

    void release()
    {
        ModeDLeaseBuffer rows[MODE_D_LEASE_BUFFERS];
        buffers(rows, 0, 0, 0, 0, 0);
        int any = 0;
        for(const ModeDLeaseBuffer &row : rows) {any |= (*row.slot != nullptr);}
        /* Kernels and copies that use this storage are enqueued, not finished: wait for them. */
        if(any) {Kokkos::fence("mode_d_call_lease_release");}
        for(int k = 0; k < MODE_D_LEASE_BUFFERS; k++) {mode_d_pool_keep(k, *rows[k].slot, row_bytes[k]); *rows[k].slot = nullptr; row_bytes[k] = 0;}
        batch_capacity[0] = batch_capacity[1] = 0;
        iter_on = 0; iter_which = 0; iter_n_live = 0;
        flags = nullptr;
        release_arena();
        stream.release_storage();
        std::vector<int>().swap(peer_counts); std::vector<int>().swap(peer_offsets);
        for(int m = 0; m < 2; m++) {std::vector<BatchSegment>().swap(batch_segments[m]);}
        if(epoch_held) {gx_touched_set_retire(GX_TOUCHED_OWNER_FUSED_WALK); epoch_held = 0;}
    }
};

/* Give the round arena room for c envelopes (never less than it has).  The old arena is kept until the new one is
 * granted, so a refusal leaves the call exactly as it was.  Returns 0 when the arena holds c. */
static int mode_d_arena_grow(ModeDCallLease *lease, long long c, size_t query_bytes, size_t reply_bytes)
{
    ModeDRoundArena &a = lease->arena;
    if(c <= a.capacity) {return 0;}
    long long offset[MODE_D_ARENA_ROWS];
    const long long bytes = mode_d_arena_layout(c, query_bytes, reply_bytes, offset);
    if(bytes <= 0) {return 1;}
    ModeDLeasePool &pool = g_mode_d_lease_pool;
    char *base = nullptr; long long held = bytes;
    if(pool.arena && pool.arena_bytes >= bytes) {
        base = pool.arena; held = pool.arena_bytes; pool.arena = nullptr; pool.arena_bytes = 0;
        /* a pooled arena holds as many envelopes as its bytes lay out for this call's record sizes, within its limit */
        long long lo = c, hi = lease->plan.arena_capacity_limit;
        while(lo < hi) {
            const long long mid = lo + (hi - lo + 1) / 2;
            const long long need = mode_d_arena_layout(mid, query_bytes, reply_bytes, offset);
            if(need > 0 && need <= held) {lo = mid;} else {hi = mid - 1;}
        }
        c = lo;
        (void) mode_d_arena_layout(c, query_bytes, reply_bytes, offset);
    }
    else {base = (char *) gizmo_gpu_alloc_device((size_t) bytes, "ngl_moded_round_arena");}
    if(!base) {return 1;}
    if(a.base) {Kokkos::fence("mode_d_arena_grow"); gizmo_gpu_free_device(a.base);}
    a.capacity = c; a.base = base; a.bytes = held;
    a.records = (ModeDExportRecord *) (base + offset[0]); a.records_spare = (ModeDExportRecord *) (base + offset[1]);
    a.scan = (int *) (base + offset[2]); a.first_record = (int *) (base + offset[3]);
    a.envelopes = base + offset[4];
    a.run_offsets = (int *) (base + offset[5]); a.envelope_of = (int *) (base + offset[6]);
    a.replies = base + offset[7];
    return 0;
}

/* Give received-query batch m room for c queries (never less than it has).  The old buffers are kept until both new
 * ones are granted, so a refusal leaves the batch as it was and the round runs in more, smaller batches; rank-local,
 * so it is never a routing change.  Called between rounds, when nothing reads or writes the batch.  Returns 0 when
 * batch m holds c. */
static int mode_d_batch_grow(ModeDCallLease *lease, int m, long long c, size_t query_bytes, size_t reply_bytes)
{
    if(c <= lease->batch_capacity[m]) {return 0;}
    const long long query_total = mode_d_checked_product(c, (long long) query_bytes);
    const long long reply_total = mode_d_checked_product(c, (long long) reply_bytes);
    if(query_total <= 0 || reply_total <= 0) {return 1;}
    void *queries = gizmo_gpu_alloc_device((size_t) query_total, "ngl_moded_group_queries");
    void *replies = queries ? gizmo_gpu_alloc_device((size_t) reply_total, "ngl_moded_group_replies") : nullptr;
    if(!queries || !replies) {gizmo_gpu_free_device(queries); return 1;}
    Kokkos::fence("mode_d_batch_grow");
    gizmo_gpu_free_device(lease->group_queries[m]); gizmo_gpu_free_device(lease->group_replies[m]);
    lease->group_queries[m] = queries; lease->group_replies[m] = replies; lease->batch_capacity[m] = c;
    lease->row_bytes[MODE_D_LEASE_GROUP_ROW + 2 * m] = query_total; lease->row_bytes[MODE_D_LEASE_GROUP_ROW + 2 * m + 1] = reply_total;
    return 0;
}

/* Size a Mode D call.  Returns 0 and fills *plan, or names (in *why) the first bound the call does not meet.
 * Rank-local: the active count differs between ranks, so the verdict is voted.  One task sizes nothing. */
static int mode_d_workspace_plan(long long num_active, size_t query_bytes, size_t reply_bytes,
                                 ModeDWorkspacePlan *plan, const char **why)
{
    *plan = ModeDWorkspacePlan{};
    if(NTask <= 1) {return 0;}
    /* Payloads move through the staging in messages of at most GX_MODE_D_STAGING_MAX_MESSAGE_BYTES, a header
     * and whole records: one record must fit in one. */
    const size_t room = GX_MODE_D_STAGING_MAX_MESSAGE_BYTES - sizeof(ModeDSegmentHeader);
    if(query_bytes > room || reply_bytes > room) {*why = "one query or reply record is larger than a staging message"; return 1;}
    const long long pair_bytes  = (long long) query_bytes + (long long) reply_bytes;
    const long long chunk_bytes = (long long) (All.CommChunkSize * 1024.0 * 1024.0);
    plan->envelopes_per_round = chunk_bytes / pair_bytes;
    plan->worst_envelopes_per_active = mode_d_worst_envelopes_per_active();
    if(plan->worst_envelopes_per_active < 0) {*why = "its per-peer top-leaf tally could not be allocated"; return 1;}
    /* the arena's records are indexed by int */
    const long long index_limit = (long long) INT_MAX / NODELISTLENGTH;
    plan->arena_capacity_limit = (plan->envelopes_per_round < index_limit) ? plan->envelopes_per_round : index_limit;
    if(plan->arena_capacity_limit < plan->worst_envelopes_per_active) {
        *why = "one active's worst-case export does not fit in a communication chunk"; return 1;}
    plan->arena_min_envelopes = (num_active > 0) ? plan->worst_envelopes_per_active : 0;
    const size_t larger = (query_bytes > reply_bytes) ? query_bytes : reply_bytes;
    plan->segment_records = (long long) (room / larger);
    return 0;
}

/* The lease's device total at admission, from its one buffer table and the minimum arena; -1 when a size leaves
 * 64 bits. */
static long long mode_d_lease_device_bytes(ModeDCallLease *lease, long long num_active, size_t active_bytes, size_t accum_bytes,
                                           size_t query_bytes, size_t reply_bytes)
{
    ModeDLeaseBuffer rows[MODE_D_LEASE_BUFFERS];
    lease->buffers(rows, num_active, active_bytes, accum_bytes, query_bytes, reply_bytes);
    long long device = 0;
    for(const ModeDLeaseBuffer &row : rows) {device = (device < 0) ? -1 : mode_d_checked_sum(device, row.bytes);}
    if(lease->plan.arena_min_envelopes > 0 && device >= 0) {
        long long offset[MODE_D_ARENA_ROWS];
        device = mode_d_checked_sum(device, mode_d_arena_layout(lease->plan.arena_min_envelopes, query_bytes, reply_bytes, offset));
    }
    return device;
}

/* Allocate every piece of a planned lease.  Returns 0 when all were granted; the caller releases a partial one. */
static int mode_d_lease_allocate(ModeDCallLease *lease, long long num_active, size_t active_bytes, size_t accum_bytes,
                                 size_t query_bytes, size_t reply_bytes)
{
    if(NTask <= 1) {return 0;}
    int ok = (lease->stream.reserve(NTask) == 0);
    try {
        lease->peer_counts.reserve((size_t) NTask + 1 + MODE_D_FLAGS); lease->peer_offsets.reserve((size_t) NTask + 1);
        for(int m = 0; m < 2; m++) {lease->batch_segments[m].reserve((size_t) lease->plan.segment_records);}
    } catch(const std::bad_alloc &) {ok = 0;}
    ModeDLeaseBuffer rows[MODE_D_LEASE_BUFFERS];
    lease->buffers(rows, num_active, active_bytes, accum_bytes, query_bytes, reply_bytes);
    /* A buffer this call cannot need (zero bytes: no actives) is not allocated at all, and its slot stays null. */
    for(int k = 0; k < MODE_D_LEASE_BUFFERS; k++) {
        if(rows[k].bytes == 0) {continue;}
        *rows[k].slot = mode_d_pool_take(k, rows[k].bytes, rows[k].label, &lease->row_bytes[k]);
        ok = ok && (*rows[k].slot != nullptr);
    }
    lease->flags = lease->per_peer ? lease->per_peer + NTask + 1 : nullptr;
    for(int m = 0; m < 2; m++) {
        /* what the (possibly pooled, larger) buffers hold, never less than one staging segment */
        const long long q_rows = lease->row_bytes[MODE_D_LEASE_GROUP_ROW + 2 * m] / (long long) query_bytes;
        const long long r_rows = lease->row_bytes[MODE_D_LEASE_GROUP_ROW + 2 * m + 1] / (long long) reply_bytes;
        lease->batch_capacity[m] = (lease->group_queries[m] && lease->group_replies[m]) ? ((q_rows < r_rows) ? q_rows : r_rows) : 0;
    }
    if(ok && lease->plan.arena_min_envelopes > 0) {ok = (mode_d_arena_grow(lease, lease->plan.arena_min_envelopes, query_bytes, reply_bytes) == 0);}
    if(ok) {
        char *base[ModeDSegmentStream::ROLES];
        for(int r = 0; r < ModeDSegmentStream::ROLES; r++) {base[r] = gx_mode_d_staging_slab(r, 0);}
        lease->stream.attach(base, GX_MODE_D_STAGING_MAX_MESSAGE_BYTES);
    }
    return ok ? 0 : 1;
}

/* The ring's roles and slots are the stream's. */
static_assert((int) GX_MODE_D_QUERIES_OUT == (int) ModeDSegmentStream::QUERIES_OUT && (int) GX_MODE_D_QUERIES_IN == (int) ModeDSegmentStream::QUERIES_IN &&
              (int) GX_MODE_D_REPLIES_OUT == (int) ModeDSegmentStream::REPLIES_OUT && (int) GX_MODE_D_REPLIES_IN == (int) ModeDSegmentStream::REPLIES_IN,
              "the staging roles and the stream's roles are one numbering");
static_assert((size_t) ModeDSegmentStream::SLOTS * GX_MODE_D_STAGING_MAX_MESSAGE_BYTES == GX_MODE_D_STAGING_SLABS_PER_ROLE * GX_MODE_D_STAGING_SLAB_BYTES,
              "a role's slabs are exactly its slots, contiguous");

/* The collective verdict on a Mode D call.  A rank that cannot hold the call's storage (or the device would not
 * hold the top-leaf map) sends it to Mode B, which needs none of it (unless Mode B is switched off: then, as for a
 * tree refusal, the call keeps its usual path).  A rank whose tree cannot be described to the device keeps the
 * path the call had before Mode D was considered -- what such a refusal has always done -- and that class wins
 * when both occur.  A top-leaf map that does not describe the tree is a defect, and stops the run. */
enum class ModeDAdmission { Admitted, StorageRefused, TreeRefused };

/* Admit a Mode D call, collectively.  Plans, makes sure the persistent staging is held, allocates the whole
 * lease, prepares the tree and the top-leaf map, then votes once.  Every rank returns the same verdict; on any
 * refusal every rank's lease is already released, so the call proceeds holding nothing of Mode D.  If any rank's
 * staging allocation was refused, every rank gives it up and later calls are refused without asking again (the
 * latch is the same on every rank, so that refusal needs no vote); likewise a refused map, until the next build. */
static ModeDAdmission mode_d_admit(ModeDCallLease *lease, long long num_active, size_t active_bytes, size_t accum_bytes,
                                   size_t query_bytes, size_t reply_bytes, const char *caller,
                                   bool iterates_on_device, size_t scratch_bytes)
{
    if(gx_mode_d_staging_unavailable()) {return ModeDAdmission::StorageRefused;}
    const char *why = nullptr;
    int tree_ok = 1, staging_ok = 1, map_granted = 1, map_wellformed = 1;
    int ok = (mode_d_workspace_plan(num_active, query_bytes, reply_bytes, &lease->plan, &why) == 0);
    lease->plan.iteration_actives = iterates_on_device ? num_active : 0;
    lease->plan.scratch_bytes = (long long) scratch_bytes;
    if(ok) {
        lease->plan.device_bytes = mode_d_lease_device_bytes(lease, num_active, active_bytes, accum_bytes, query_bytes, reply_bytes);
        if(lease->plan.device_bytes < 0) {why = "the call's storage size leaves 64-bit range"; ok = 0;}
    }
    /* One task sends no messages, so it holds no staging. */
    if(ok && NTask > 1) {staging_ok = (gx_mode_d_staging_acquire() == 0); if(!staging_ok) {why = "its pinned staging could not be allocated"; ok = 0;}}
    if(ok) {
        ModeDLeaseBuffer rows[MODE_D_LEASE_BUFFERS];
        lease->buffers(rows, num_active, active_bytes, accum_bytes, query_bytes, reply_bytes);
        long long row_bytes[MODE_D_LEASE_BUFFERS], offset[MODE_D_ARENA_ROWS];
        for(int k = 0; k < MODE_D_LEASE_BUFFERS; k++) {row_bytes[k] = rows[k].bytes;}
        const long long arena_bytes = (lease->plan.arena_min_envelopes > 0) ? mode_d_arena_layout(lease->plan.arena_min_envelopes, query_bytes, reply_bytes, offset) : 0;
        const long long fresh = mode_d_pool_shortfall(row_bytes, arena_bytes);
        if(fresh < 0 || !gizmo_device_has_room((size_t) fresh, MODE_D_DEVICE_FREE_FRACTION)) {why = "the device has no room for its workspace"; ok = 0;}
    }
    if(ok && mode_d_lease_allocate(lease, num_active, active_bytes, accum_bytes, query_bytes, reply_bytes) != 0) {why = "its workspace could not be allocated"; ok = 0;}
    /* The tree preparation repairs state and is the expensive half: only a rank holding its
     * storage pays for it.  It reports its own reasons when it declines. */
    if(ok) {tree_ok = (gx_device_fused_walk_prepare(&lease->tree, caller) == 0); lease->epoch_held = tree_ok; ok = tree_ok;}
    if(ok && NTask > 1) {
        const int status = gx_top_leaf_map_acquire(&lease->tree, &lease->map);
        if(status == GX_TOP_LEAF_MAP_REFUSED) {map_granted = 0; why = "the device would not hold its top-leaf map"; ok = 0;}
        if(status == GX_TOP_LEAF_MAP_MALFORMED) {map_wellformed = 0; why = "its top-leaf map does not describe the tree"; ok = 0;}
    }
    int local[5] = {ok, tree_ok, staging_ok, map_granted, map_wellformed}, all[5] = {0, 0, 0, 0, 0};
    MPI_Allreduce(local, all, 5, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!all[4]) {
        /* Every rank is here, in step.  Nothing of Mode D is held across the stop, which drains at once. */
        lease->release();
        gx_top_leaf_map_discard_for_tree();
        if(!map_wellformed) {
            printf("%s: task %d: the domain's top leaves do not describe this rank's tree (a leaf outside the node range, "
                   "not a top-level node, mapped twice, or owned by no task); stopping.\n", caller, ThisTask);
            fflush(stdout);
        }
        endrun(90001041);
        gizmo_exit_bad_stop_if_requested("nlr:mode_d_map");
        return ModeDAdmission::TreeRefused;   /* not reached: the stop above finalizes every rank */
    }
    /* Reported after the vote, so it names where the call actually went; each distinct reason the
     * first time it occurs. */
    if(why) {
        /* Room for every reason this admission can give (twelve); were it ever full, nothing more
         * is printed, so no reason can repeat on every call. */
        static const char *reported_reasons[16] = {nullptr};
        int seen = 0, k = 0;
        for(; k < 16 && reported_reasons[k]; k++) {if(reported_reasons[k] == why) {seen = 1; break;}}
        if(!seen && k < 16) {
            reported_reasons[k] = why;
            printf("%s: task %d declines a device-resident call: %s (%.1f MB device, %.1f MB pinned staging); "
                   "a call declined this way %s on every rank\n",
                   caller, ThisTask, why, (lease->plan.device_bytes > 0 ? lease->plan.device_bytes : 0) / (1024.0 * 1024.0),
                   (double) GX_MODE_D_STAGING_BYTES / (1024.0 * 1024.0),
                   all[1] ? "goes to Mode B (or keeps its usual path if Mode B is switched off)"
                          : "keeps the path it had before Mode D (another task could not describe its tree)");
            fflush(stdout);
        }
    }
    if(all[0]) {return ModeDAdmission::Admitted;}
    lease->release();
    if(!all[2]) {gx_mode_d_staging_mark_unavailable();}
    if(!all[3]) {gx_top_leaf_map_discard_for_tree();}
    return all[1] ? ModeDAdmission::StorageRefused : ModeDAdmission::TreeRefused;
}

/* Helper: compile-time member detector for SupportsSubgroups (default false). */
template <typename Spec, typename = void>
struct nlr_supports_subgroups : std::false_type {};
template <typename Spec>
struct nlr_supports_subgroups<Spec, std::void_t<typename Spec::SupportsSubgroups>>
    : Spec::SupportsSubgroups {};

/* Whether a multi-rank Mode D call of this loop iterates on the device: its after_iter has the device entry point
 * (Spec::after_iter_device), it has no host hook that runs after every iteration over the whole active set
 * (after_iter_global reads the host's per-active state), and it has one subgroup (the device state is one list). */
template <typename Spec, typename = void>
struct nlr_spec_has_after_iter_device : std::false_type {};
template <typename Spec>
struct nlr_spec_has_after_iter_device<Spec, std::void_t<decltype(&Spec::after_iter_device)>> : std::true_type {};
template <typename Spec>
constexpr bool nlr_mode_d_iterates_on_device_v =
    nlr_spec_has_after_iter_device<Spec>::value && !nlr_spec_has_after_iter_global_v<Spec> && !nlr_supports_subgroups<Spec>::value;

template <typename Spec>
static ModeDAdmission mode_d_admit_for(ModeDCallLease *lease, int num_active)
{
    size_t scratch_bytes = 0;
    if constexpr (nlr_mode_d_iterates_on_device_v<Spec>) {scratch_bytes = sizeof(typename Spec::IterScratch);}
    return mode_d_admit(lease, num_active, sizeof(typename Spec::ActiveData), sizeof(typename Spec::AccumData),
                        sizeof(NlrQueryEnvelope<typename Spec::ActiveData>),
                        sizeof(NlrReplyEnvelope<typename Spec::AccumData>), Spec::loop_name,
                        nlr_mode_d_iterates_on_device_v<Spec>, scratch_bytes);
}

template <typename Spec>
void run_neighbor_loop(const neighbor_loop_args& args_in)
{
    /* The one live argument view for this call.  A ghost import can raise the particle
     * capacity and move P[]/CellP[], so the cached base pointers are refreshed in this
     * object below; every later reader sees the current values.  There is deliberately no
     * second, unrefreshed copy in scope -- reading a stale P[] reads freed storage. */
    neighbor_loop_args args = args_in;

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

    /* Threshold dispatch. Allreduce sum + max of args.num_active.
     * Skipped when the caller supplied a dispatch override (cheap path), which
     * is why the global count is not available on every path: it is computed
     * only where something needs it — the threshold decision itself, and the
     * forced-Mode-B size guard. It is never computed merely to report it. */
    bool select_mode_b = force_b;
    /* Mode D is asked for only on the threshold path: an override is honoured
     * as given. */
    NlrMode selected_mode = force_b ? NlrMode::B : NlrMode::A;
    int global_num_active = -1;       /* -1 = not computed on this path */
    if(!force_a && !force_b) {
        int local_act = args.num_active;
        int sum_act = 0, max_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_act, &max_act, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        global_num_active = sum_act;
        selected_mode = nlr_select_mode<Spec>(sum_act, max_act);
        select_mode_b = (selected_mode == NlrMode::B);
    } else if(force_b) {
        /* Forced Mode B skipped the dispatch Allreduce, but its size guard
         * needs the global count, so it is done here. */
        int local_act = args.num_active;
        int sum_act = 0;
        MPI_Allreduce(&local_act, &sum_act, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        global_num_active = sum_act;
    }
    /* Globally-zero-active call: do NO neighbor work. global_num_active is the
     * dispatch Allreduce of active particles (set on the threshold + force-B
     * paths), so it is identical on every rank -> this return is collective-
     * symmetric (all ranks return together, skipping the Mode-A ghost import /
     * writeback / cleanup as a matched set). NOT the banned local num_active==0
     * early return: the condition is GLOBAL. Without it, a zero-active call
     * falls to Mode A and fires a spurious ghost import with nothing to compute.
     * The forced-Mode-A path does not compute the count, so it does not reach
     * this and still pays that import; giving it the saving means giving it a
     * collective it does not otherwise need, which is a separate decision. */
    if(global_num_active == 0) {
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
    plan.num_active_global = global_num_active;   /* -1 on dispatch-override paths */

    /* A loop whose kernel reads neighbour state it is itself changing needs the
     * live particle, which a ghost copy is not (nlr_spec_needs_live_neighbours).
     * It is answered where its neighbours live: on the device below when the
     * tree can be described there, by the host walker otherwise. */
    if constexpr (nlr_spec_needs_live_neighbours_v<Spec>) {
        if(plan.path == NeighborLoopPlan::Path::ModeA_GpuNgl) {
            plan.path = (NTask > 1) ? NeighborLoopPlan::Path::ModeB_Remote
                                    : NeighborLoopPlan::Path::ModeB_Local;
        }
    }
    /* Everything a Mode D call holds, released on every way out of this function (see the
     * lease).  Declared here so it outlives the decision below and the execution further down. */
    ModeDCallLease mode_d_lease;
    /* Mode D takes the calls nlr_select_mode gives it, when every rank is admitted -- the
     * same decision the iterative driver makes, and for the same reasons (stated there).  A
     * caller-owned neighbour list is Mode A by construction.  The admission is collective;
     * the mode was chosen from a collective too, so every rank reaches it.  A refused call
     * holds nothing of Mode D; where it goes is mode_d_admit's verdict. */
    if constexpr (nlr_spec_mode_d_eligible_v<Spec>) {
        if(selected_mode == NlrMode::D && args.external_csr == nullptr) {
            const ModeDAdmission verdict = mode_d_admit_for<Spec>(&mode_d_lease, args.num_active);
            if(verdict == ModeDAdmission::Admitted) {plan.path = NeighborLoopPlan::Path::ModeD_DeviceFused;}
            else if(verdict == ModeDAdmission::StorageRefused && nlr_mode_b_switched_on<Spec>()) {
                plan.path = (NTask > 1) ? NeighborLoopPlan::Path::ModeB_Remote : NeighborLoopPlan::Path::ModeB_Local;}
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
    /* Neighbours whose motion this loop changes are raised when the call ends. */
    NlrMotionTargetScope<Spec> motion_target_scope;

    /* ---- Hard-corridor counter snapshot (always-on, every build) ---- */
    /* Mode B paths must NOT enter move_particles, ghost_exchange_impl, or
     * gpu_particles_arena_acquire, and must NOT mutate NumPart. Counters
     * live in declarations/lifecycle_counters.h, incremented at API entry
     * by the owner TUs. */
    const uint64_t s_drift0 = g_global_drift_counter;
    const uint64_t s_ghost0 = g_ghost_import_counter;
    const uint64_t s_arena0 = g_gpu_arena_acquire_counter;
    const int      s_np0    = NumPart;


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
            nlr_check_caller_owned_ghost_pool(args, Spec::loop_name);
        } else {
            gizmo_request_filtered_ghost_import_fresh(Spec::loop_name,
                                                       Spec::search_mode,
                                                       nlr_effective_neighbor_type_mask(args, Spec::neighbor_type_mask),
                                                       args.active_list,
                                                       args.num_active,
                                                       radii.data(),
                                                       args.ghost_safety_factor,
                                                       Spec::radius_policy,
                                                       nlr_spec_symmetric_j_radius_scale<Spec>());
            /* Ghost import grew NumPart and may have realloc'd P/CellP. Refresh
             * the runner's data view; only paths that imported ghosts read this
             * extended view (Mode B paths use the original args via copy). */
            args.num_total = NumPart;
            args.P         = P;
            args.CellP     = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;
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
    nlr_dispatch_ghost_write_detector_begin<Spec>(args, plan);
    nlr_dispatch_ghost_writeback_begin<Spec>(args, plan);

    /* ---- Path dispatch ---- */
    switch(plan.path) {
        case NeighborLoopPlan::Path::ModeA_GpuNgl:
            run_mode_a<Spec>(args, radii.data());
            break;
        case NeighborLoopPlan::Path::ModeB_Local:
            run_mode_b_local<Spec>(args, radii.data());
            break;
        case NeighborLoopPlan::Path::ModeB_Remote:
            run_mode_b_remote<Spec>(args, radii.data());
            break;
        case NeighborLoopPlan::Path::ModeD_DeviceFused:
            if constexpr (nlr_spec_mode_d_eligible_v<Spec>) {
                run_mode_d<Spec>(args, radii.data(), mode_d_lease, mode_d_lease.tree);
            } else {
                /* Never chosen without the eligibility above. */
                endrun(90001032);
            }
            break;
    }

    /* ---- Spec lifecycle hooks (end, reverse order) ---- */
    nlr_dispatch_ghost_writeback_end<Spec>(args, plan);
    nlr_dispatch_ghost_write_detector_end<Spec>(args, plan);

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

/* ============================================================================
 * NlrIterDriver<Spec> — constructor + destructor.
 * ========================================================================== */
template <typename Spec>
NlrIterDriver<Spec>::NlrIterDriver(neighbor_loop_args_iterative& a,
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
      call_slot_base  (a.num_subgroups, 0),
      /* Mode A iterative cached-CSR state: zero-init per subgroup;
       * UVM allocations land lazily on first Mode A iter dispatch. */
      mode_a_cached_gnl       (a.num_subgroups, gpu_neighbor_list_t{}),
      mode_a_csr_offset_lookup(a.num_subgroups, nullptr),
      mode_a_csr_buffered_h   (a.num_subgroups, nullptr),
      mode_a_csr_valid        (a.num_subgroups, false)
{
    using IterScratch = typename Spec::IterScratch;
    using AccumData   = typename Spec::AccumData;

    /* Build a base neighbor_loop_args view per subgroup so we can call
     * Spec::search_radius for each (subgroup, slot, particle index) tuple.
     * search_radius is host-only and predates any drift / arena work. */
    /* Where each subgroup's entries sit in the call's active_list, which is the
     * layout a Spec's call-level per-active staging is in (NlrSubgroup). A
     * caller that hands in a list which is not the ordered concatenation of its
     * subgroups would have every Spec that stages per-active state read another
     * particle's record, so the identity is checked here rather than assumed. */
    {
        int base = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            const NlrSubgroup& sgr = args.subgroups[sg];
            call_slot_base[sg] = base;
            base += (sgr.num_active_local > 0) ? sgr.num_active_local : 0;
        }
        bool consistent = (base == args.num_active);
        for (int sg = 0; consistent && sg < args.num_subgroups; sg++) {
            const NlrSubgroup& sgr = args.subgroups[sg];
            for (int k = 0; k < sgr.num_active_local; k++) {
                if (args.active_list[call_slot_base[sg] + k] != sgr.active_indices[k]) {consistent = false; break;}
            }
        }
        if (!consistent) {
            fprintf(stderr, "[NlrIterDriver<%s>] FATAL: active_list (%d entries) is not the ordered "
                            "concatenation of the %d subgroups' active_indices; per-active staging "
                            "would be read against the wrong particles.\n",
                    Spec::loop_name, args.num_active, args.num_subgroups);
            fflush(stderr);
            endrun(90001031);
        }
    }

    for (int sg = 0; sg < args.num_subgroups; sg++) {
        const NlrSubgroup& sgr = args.subgroups[sg];
        const int n = sgr.num_active_local;
        active_set_size[sg] = n;
        if (n <= 0) continue;     /* leave pointers as nullptr; pair_kernel is no-op */

        const size_t sg_bytes = (size_t) n * (sizeof(IterScratch) + sizeof(AccumData)
                                              + sizeof(double) + sizeof(int));
        scratch_uvm[sg]    = (IterScratch *) nlr_shared_alloc_bytes((size_t) n * sizeof(IterScratch), "modea_iter_scratch");
        accum_uvm  [sg]    = (AccumData   *) nlr_shared_alloc_bytes((size_t) n * sizeof(AccumData), "modea_accum_data");
        radii_uvm  [sg]    = (double      *) nlr_shared_alloc_bytes((size_t) n * sizeof(double), "modea_radii");
        active_set_uvm[sg] = (int         *) nlr_shared_alloc_bytes((size_t) n * sizeof(int), "modea_active_set");
        /* Without its four arrays this subgroup cannot be walked, so it is left in
         * the same state as one with no actives at all: sized zero, pointers null,
         * which every dispatch and the destructor already handle. The other
         * subgroups keep their state and the run stops at the next phase boundary. */
        if (!scratch_uvm[sg] || !accum_uvm[sg] || !radii_uvm[sg] || !active_set_uvm[sg]) {
            if (scratch_uvm[sg])    { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(scratch_uvm[sg]);    scratch_uvm[sg] = nullptr; }
            if (accum_uvm[sg])      { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(accum_uvm[sg]);      accum_uvm[sg] = nullptr; }
            if (radii_uvm[sg])      { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_uvm[sg]);      radii_uvm[sg] = nullptr; }
            if (active_set_uvm[sg]) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(active_set_uvm[sg]); active_set_uvm[sg] = nullptr; }
            active_set_size[sg] = 0;
            nlr_stop_no_staging_memory(Spec::loop_name, "the per-subgroup iteration state", sg_bytes);
            continue;
        }

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
     * hadn't completed yet; check pointers, not the flag). */
    {
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (mode_a_cached_gnl[sg].offsets != nullptr ||
                mode_a_cached_gnl[sg].neighbors != nullptr) {
                gpu_ngb_list_free(&mode_a_cached_gnl[sg]);
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
     * args: if ghost import ran above, these
     * point at the refreshed POST-IMPORT global P/CellP/NumPart; otherwise
     * they equal the original base args. CellP=NULL is legitimate (gas-free). */
    if (!nlr_args_view_is_live(args)) {
        if (ThisTask == 0) {
            fprintf(stderr,
                "[NlrIterDriver<%s>::initialize_device_context_mode_a_after_arena] FATAL: the "
                "argument view no longer describes the live particle storage (P=%p vs %p, "
                "num_total=%d vs %d). Binding a device context to it would read released memory.\n",
                Spec::loop_name, (void *) args.P, (void *) P, args.num_total, NumPart);
            fflush(stderr);
        }
        /* Soft bad-stop + return WITHOUT binding ctx: same shape as the two lifecycle
         * violations above, drained at the runner's next poll before any dispatch. */
        endrun(81215); return;
    }
    ctx.P         = gpu_particles_arena_P();
    ctx.CellP     = (args.CellP != nullptr) ? gpu_particles_arena_CellP() : nullptr;
    ctx.num_total = args.num_total;

    if constexpr (nlr_spec_has_extended_device_context_v<Spec>) {
        Spec::populate_device_context(args, ctx);
    }
    ctx_initialized = true;
}

/* The union ghost import of a subgrouped Mode-A call: every subgroup's actives
 * at their search radius padded by the Spec's buffer factor, neighbour types =
 * the OR of the subgroup masks, under the Spec's search mode, radius policy and
 * neighbour-radius scale.  One recipe, for the iterative driver and for a caller
 * that imports in a Spec's place (the CBE corridor imports for AgsForceSpec). */
template <typename Spec>
void nlr_import_union_ghosts(unsigned int mask_union, const std::vector<int> &actives,
                             const std::vector<double> &radii, double ghost_safety_factor)
{
    std::vector<double> padded(radii.size());
    for (size_t k = 0; k < radii.size(); k++) {padded[k] = radii[k] * Spec::mode_a_csr_buffer_factor;}
    const int n = (int)actives.size();
    gizmo_request_filtered_ghost_import_fresh(Spec::loop_name, Spec::search_mode, mask_union,
                                               (n > 0) ? actives.data() : nullptr, n,
                                               (n > 0) ? padded.data() : nullptr, ghost_safety_factor,
                                               Spec::radius_policy, nlr_spec_symmetric_j_radius_scale<Spec>());
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
     * so the rebuild import doesn't over-cover converged slots.
     *
     * Subgroups with caller-built lists: the caller imported the pool the
     * lists index and owns it -- no import here (it would renumber the ghost
     * slots under the lists), and ghost_import_done stays false so the call's
     * end leaves the pool alone. */
    if (nlr_subgroups_carry_caller_lists(args)) {
        nlr_check_caller_owned_ghost_pool(args, Spec::loop_name);
    } else if (NTask > 1) {
        std::vector<int>    union_actives;
        std::vector<double> union_radii;
        unsigned int        mask_union = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            mask_union |= args.subgroups[sg].j_type_bitmask;
            const NlrSubgroup& sgr = args.subgroups[sg];
            /* A subgroup whose per-active radii could not be staged contributes
             * nothing to the import. The import itself is collective, so this
             * rank still enters it -- just asking for no ghosts of its own. */
            if (radii_uvm[sg] == nullptr) { continue; }
            for (int k = 0; k < sgr.num_active_local; k++) {
                union_actives.push_back(sgr.active_indices[k]);
                union_radii.push_back(radii_uvm[sg][k]);
            }
        }
        nlr_import_union_ghosts<Spec>(mask_union, union_actives, union_radii, args.ghost_safety_factor);
        ghost_import_done = true;

        /* Refresh args from globals (ghost import grew NumPart and
         * may have realloc'd P/CellP). Matches non-iter line 2049-2051. */
        args.num_total = NumPart;
        args.P         = P;
        args.CellP     = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;
    }

    /* === (2) Arena acquire ONCE per call, with refreshed args === */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total,
                                 args.P, args.CellP);
    arena_acquired = true;

    /* === (3) Bind ctx to arena-resident pointers + populate extended DeviceContext === */
    initialize_device_context_mode_a_after_arena();
}

/* Mode D's context: the same device-resident arrays, WITHOUT the ghost import.
 *
 * Mode A imports because it walks a pool that has to contain the neighbours
 * living on other ranks.  A fused walk never sees a ghost -- the traversal stops
 * at the owned count -- and the queries that need another rank's particles are
 * shipped to that rank instead.  Calling the Mode A entry here would perform the
 * whole collective import and then discard every ghost it fetched, which is the
 * one cost this path exists to avoid.
 *
 * Skipping it is collective-safe because the decision that selected Mode D was
 * itself collective and unanimous: either every rank takes this entry or none
 * does, so no rank is left waiting in an import its peers skipped. */
template <typename Spec>
void NlrIterDriver<Spec>::acquire_arena_and_init_ctx_mode_d()
{
    if (arena_acquired) {return;}

    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total, args.P, args.CellP);
    arena_acquired = true;

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
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (mode_a_cached_gnl[sg].offsets != nullptr ||
                mode_a_cached_gnl[sg].neighbors != nullptr) {
                gpu_ngb_list_free(&mode_a_cached_gnl[sg]);
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
    std::vector<double> union_radii;
    unsigned int        mask_union = 0;
    for (int sg = 0; sg < args.num_subgroups; sg++) {
        if (global_active_per_sg[sg] <= 0) continue;     /* skip globally-converged */
        mask_union |= args.subgroups[sg].j_type_bitmask;
        const NlrSubgroup& sgr = args.subgroups[sg];
        for (int k = 0; k < active_set_size[sg]; k++) {
            int slot = active_set_uvm[sg][k];
            union_actives.push_back(sgr.active_indices[slot]);
            union_radii.push_back(radii_uvm[sg][slot]);
        }
    }

    /* === (0) Invalidate ALL subgroup CSR caches (avoids cross-subgroup
     * staleness after arena teardown). Free by pointer state. */
    {
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (mode_a_cached_gnl[sg].offsets != nullptr ||
                mode_a_cached_gnl[sg].neighbors != nullptr) {
                gpu_ngb_list_free(&mode_a_cached_gnl[sg]);
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
            Spec::cleanup_device_context(args, ctx);
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
    nlr_import_union_ghosts<Spec>(mask_union, union_actives, union_radii, args.ghost_safety_factor);
    ghost_import_done = true;

    /* === (5) Refresh args from post-import globals === */
    args.num_total = NumPart;
    args.P         = P;
    args.CellP     = (gizmo_host_all_ptr()->TotN_gas > 0) ? CellP : nullptr;

    /* === (6) Fresh arena view of new pool === */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    gpu_particles_arena_set_site(Spec::loop_name);
    gpu_particles_arena_acquire(args.num_total,
                                 args.P, args.CellP);
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
 * drift_and_filter_candidates / evaluate_pairs_post_drift). SSOT preserved with
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
    std::vector<int>    call_slots(n_compacted);
    std::vector<double> radii_compacted(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        active_particle_indices[k] = sgr.active_indices[slot];
        call_slots[k]              = drv.call_slot_base[sg] + slot;
        radii_compacted[k]         = drv.radii_uvm[sg][slot];
    }

    neighbor_loop_args sub = drv.args;
    sub.active_list      = (n_compacted > 0) ? active_particle_indices.data() : nullptr;
    sub.active_call_slot = (n_compacted > 0) ? call_slots.data() : nullptr;
    sub.num_active       = n_compacted;

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
        (n_compacted > 0) ? accums_compacted.data() : nullptr);

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
 * Mode D — the walk and the pair kernel in one device pass.
 *
 * Mode A brings the neighbours to the seeker: it imports the particles another
 * rank owns, builds a neighbour list over them, and then evaluates.  Mode D
 * sends the seeker to the neighbours instead, and evaluates where they already
 * live, so the list is never built and the particles are never moved.  What that
 * removes is not arithmetic -- the same pair kernel runs on the same pairs --
 * but the import, the list, and the copies that surround them.
 *
 * The seeker cannot tell where it is.  A query that started on this rank and one
 * that arrived from another are the same object to the walk and to the kernel;
 * the only difference is which accumulator the result lands in, and that is the
 * caller's business, not the kernel's.  This file's half is the local one.
 *
 * Availability is a property of the loop, not of the caller's name: the pair
 * kernel has to be one that only reduces into the seeker's own accumulator, or
 * evaluating it on a rank that does not own the seeker would need writes shipped
 * back.  That is exactly what the eval tier already records, so it is what is
 * asserted, and no loop is named here.
 * ========================================================================== */

/* Whether a fused walk of this loop keeps each reached particle's mass verdict for the rest of the call
 * (gx_candidate_has_mass): only when the loop's pair kernel cannot take a neighbour's mass to zero or below
 * (nlr_spec_keeps_neighbour_mass_positive_v), and only when the walk runs on the device, where the verdicts
 * live.  Otherwise Mass is read from the particle record at every visit. */
template <typename Spec>
static int nlr_mode_d_keeps_mass_verdict(bool on_device)
{
    return (nlr_spec_keeps_neighbour_mass_positive_v<Spec> && on_device) ? 1 : 0;
}


/* The leaf mirror for a fused walk: the device copy of every tree slot's position and type, brought current,
 * when the walk runs on the device and this rank has it; otherwise the null view, and the walk reads the
 * particle records. */
static GxLeafMirrorView nlr_mode_d_leaf_mirror(bool on_device, const GxDeviceTreeView &tree)
{
    return on_device ? gx_leaf_mirror_current(tree.local_particle_slots) : GxLeafMirrorView{};
}

/* Run a fused walk's work items: on the device whatever their number when `device_only` (an admitted
 * multi-rank Mode D call, whose queries and replies live in device memory), else where the batch size puts
 * them (nlr_walk_for_sources). */
template <class F>
static inline void nlr_mode_d_launch(bool device_only, const char *tag, int n, F &&body)
{
    if(n <= 0) {return;}
    if(device_only) {gizmo_gpu_kernel_launch(tag, n, std::forward<F>(body)); return;}
    nlr_walk_for_sources(tag, n, std::forward<F>(body));
}

/* What happens when the walk reaches a particle this rank owns.
 *
 * The traversal decides which indices are reachable and which are ghosts; this
 * decides whether a reachable one is a neighbour, and it applies the same three
 * tests the host walker applies in mesh/mode_b_local_walker.cc -- allowed type,
 * positive mass, and the geometric accept -- through the same shared predicate,
 * so the two cannot answer differently about the same pair.  The reach is the
 * query's alone, which is what makes this the ONEWAY case: no neighbour supply
 * radius is consulted, and none is mirrored to the device to consult. */
template <typename Spec, bool ReadsLeafMirror = false>
struct NlrModeDReduceLeaf {
    const typename Spec::DeviceContext *ctx;
    const typename Spec::ActiveData    *active;
    typename Spec::AccumData           *accum;
    typename Spec::ScatterData         *scatter;
    unsigned int                        supply_mask;
    /* By pointer, not by value: the leaf is built per work item inside the
     * kernel, and a CallScalars is 120-200 B, so copying it into every leaf
     * would put that much in local memory per item for a value every item
     * shares. The pointee is the launching lambda's OWN by-value capture --
     * device-resident -- which is why the leaf must keep being constructed
     * INSIDE the kernel. Hoisting that construction out would leave this
     * pointing at a host stack object and fault on device. */
    const typename Spec::CallScalars   *cs;
    struct GxMotionTargetSet            motion_targets{};   /* for a loop that writes neighbour motion */
    GxLeafMirrorView                    leaf_mirror{};      /* read when ReadsLeafMirror: P[j].Pos and P[j].Type, exact, in device memory */
    struct GxTouchedSet                 mass_verdicts{};    /* this call's mass verdicts, used when use_mass_verdict */
    int                                 use_mass_verdict = 0;   /* nlr_mode_d_keeps_mass_verdict */

    KOKKOS_INLINE_FUNCTION
    void visit(int j, double qx, double qy, double qz, double reach)
    {
        /* Type, then geometry, then mass: type and position from the leaf mirror on a device batch (an exact copy)
         * or from the particle record, mass last, so it is asked only of slots that already pass both and usually
         * answered by this call's verdict.  The tests have no side effects, so the order does not change the
         * accepted set. */
        if constexpr (ReadsLeafMirror) {
            if(!(supply_mask & (1u << (unsigned int)leaf_mirror.type[j]))) {return;}
            const Vec3<MyDouble> &x = leaf_mirror.pos[j];
            if(!gx_pair_accept_wrap_and_test(qx - (double)x[0], qy - (double)x[1], qz - (double)x[2],
                                             reach, 0.0, NGB_SEARCH_ONEWAY)) {return;}
        } else {
            const struct particle_data &Pj = ctx->P[j];
            if(!(supply_mask & (1u << (unsigned int)Pj.Type))) {return;}
            if(!gx_pair_accept_wrap_and_test(qx - (double)Pj.Pos[0],
                                             qy - (double)Pj.Pos[1],
                                             qz - (double)Pj.Pos[2],
                                             reach, 0.0, NGB_SEARCH_ONEWAY)) {return;}
        }
        if(!gx_candidate_has_mass(mass_verdicts, use_mass_verdict, ctx->P, j)) {return;}
        if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_mark(motion_targets, j);}
        IdentitySidecar id{};
        typename Spec::NeighborData nb = Spec::load_neighbor(*ctx, j, id, *active);
        Spec::pair_kernel(*active, nb, *accum, *scatter, *cs);
    }
};

/* One query of a fused Mode-D batch: the batch's own locals [0, n_local) search the
 * whole tree from the root; a received query kk resumes at the start nodes its
 * sender's walk exported to this rank. */
/* How many start nodes a received envelope carries, as a walk may use it: a count off the wire is clamped to
 * the list it came in, and a negative one is none -- never the "walk from the root" a local query takes -- so
 * the recording and the evaluating walk decide the same way about the same query. */
template <typename ActiveData>
KOKKOS_INLINE_FUNCTION int nlr_mode_d_envelope_nodes(const NlrQueryEnvelope<ActiveData> &env)
{
    return (env.n_nodes < 0) ? 0 : ((env.n_nodes > NODELISTLENGTH) ? NODELISTLENGTH : env.n_nodes);
}

template <typename Leaf, typename ActiveData>
KOKKOS_INLINE_FUNCTION
void nlr_mode_d_walk_query(int kk, int n_local, const ActiveData &a, const NlrQueryEnvelope<ActiveData> *received,
                           const GxDeviceTreeView &tree, Leaf &leaf, int *anomaly_d, unsigned int type_mask)
{
    if(kk < n_local) {
        gx_device_tree_walk_from_root((double)a.pos[0], (double)a.pos[1], (double)a.pos[2],
                                      (double)a.h_search, tree, leaf, anomaly_d, type_mask);
        return;
    }
    const NlrQueryEnvelope<ActiveData> &env = received[kk - n_local];
    const int n_start = nlr_mode_d_envelope_nodes(env);
    /* No start nodes: nothing on this rank was exported to this query.  Every query on this wire is TARGETED,
     * so one carrying no start nodes reached nothing here; the host walker skips it rather than walking from
     * anywhere, and so does this, leaving the zeroed accumulator in place, which is the right reply. */
    if(n_start <= 0) {return;}
    gx_device_tree_walk((double)a.pos[0], (double)a.pos[1], (double)a.pos[2], (double)a.h_search,
                        env.NodeList, n_start, tree, leaf, anomaly_d, type_mask);
}


/* What a walk's anomaly report means, for the three sites that stop the run on
 * one.  The states are distinct and so are their causes, so a single message
 * naming only the tree would send the reader looking in the wrong place. */
static const char *nlr_walk_anomaly_text(int code)
{
    switch(code) {
    case GX_WALK_ANOMALY_MALFORMED_TREE:
        return "a query reached an index in the gap between the particle slots and the node base; the tree is malformed";
    case GX_WALK_ANOMALY_TOUCHED_SET_FULL:
        return "the touched-set list was shorter than the distinct set the recording walk put in it";
    case GX_WALK_ANOMALY_RECORDER_OUT_OF_PHASE:
        return "a walk claimed into a recorder epoch its own phase does not own";
    default:
        return "an unrecognised walk anomaly";
    }
}

/* What happens when a RECORDING walk reaches a particle this rank owns.
 *
 * The same traversal as the evaluating leaf above, with the evaluation removed:
 * this one only writes down which local particles the walk touches, so that just
 * those can be brought current before the walk runs again and evaluates them.
 *
 * It records BEFORE the geometric accept, and that is the whole correctness
 * argument rather than an efficiency choice.  The accept test compares positions,
 * and the positions are exactly what has not been brought current yet; a particle
 * that will move into range is rejected here and would then be evaluated on its
 * undrifted position in the second pass, which is worse than not de-fusing at
 * all.  The type and mass tests are safe to apply because a drift changes
 * neither, so they narrow the set without being able to drop anything the second
 * pass can accept.
 *
 * The traversal reads node geometry and the particle-slot links, none of which a
 * drift writes, so the pass that evaluates reaches exactly the set this pass
 * recorded.  That is what makes this complete rather than approximate.
 *
 * It reads no Spec member, so one policy serves every loop the fused backend can
 * take rather than one per Spec. */
struct NlrRecordLeaf {
    const struct particle_data *P;
    struct GxTouchedSet         ts;
    unsigned int                supply_mask;
    int                        *anomaly;
    int                         use_mass_verdict = 0;          /* nlr_mode_d_keeps_mass_verdict */
    const unsigned char        *mirror_type = nullptr;         /* the leaf mirror's types; null = read the particle records */

    KOKKOS_INLINE_FUNCTION
    void visit(int j, double, double, double, double) const
    {
        const int type = mirror_type ? (int)mirror_type[j] : (int)P[j].Type;
        if(!(supply_mask & (1u << (unsigned int)type))) {return;}
        if(!gx_candidate_has_mass(ts, use_mass_verdict, P, j)) {return;}
        /* The one claim.  A full list or an
         * out-of-phase claim is reported rather than dropped, because a dropped index is a
         * particle silently evaluated at a stale position -- the one failure this design
         * must not be able to have quietly. */
        gx_touched_set_claim_in(ts, j, GX_TOUCHED_OWNER_FUSED_WALK, anomaly);
    }
};

/* ============================================================================
 * PACKETS: several queries of one batch sharing one traversal.
 *
 * The flat device walk gives each query one lane and its own walk.  Below about
 * half the device's lanes per query that leaves most of the device idle while each
 * lane chases its own pointer chain, so a batch in that band walks as PACKETS
 * instead: TREE_QUERY_PACKET_SIZE adjacent queries of the batch share one traversal
 * (mesh/device_packet_walk.h, the same mechanics the gravity walk uses), each still
 * deciding every node for itself, and each member then evaluates the leaves its
 * own decisions reached, in the order its own flat walk would have reached them.
 * The evaluating packet keeps one walker, so that order -- and with it every sum --
 * is the same on every run; the team then shares each member's evaluation, which
 * changes the order its partial sums are added in, so a member's result equals its
 * flat walk's to round-off rather than bit for bit.
 *
 * Routing, per batch, on the batch's own query count M on this rank:
 *   - below the device tier (nlr_walk_runs_on_device) the batch runs on the host;
 *   - the RECORDING pass walks as packets while 4 M < lanes, the EVALUATING pass
 *     while 2 M < lanes; each was measured to win on FIRE-dwarf and fire2ics in
 *     that band (record traversal 4096-65k, evaluation 4096-112k) and to lose
 *     above it, so the two are selected independently;
 *   - otherwise the flat device walk.
 * A packet size of 1 walks flat: a packet of one is the flat walk plus bookkeeping.
 *
 * Evaluation through packets is limited to loops whose pair kernel only reduces
 * into the query's own accumulator -- no neighbour-side writes, motion marks,
 * ghost writeback or scatter -- because a packet that gives up is discarded and
 * its members are walked again flat; anything a discarded packet had published
 * would then be applied twice.  That is a structural property of the loop, tested
 * here, and every loop that has it takes packets.  The recording pass serves every
 * loop: its claims are idempotent, so a packet that gives up keeps the claims it
 * made and the flat replay completes the set.
 *
 * A packet gives up, and its members are replayed flat, when its continuations run
 * out of room, when a round makes no progress, or when the start nodes its received
 * members carry cannot be put in one depth-first order.  A malformed tree index is
 * reported through the walk's anomaly word, which stops the run; its packet's members
 * are replayed flat too, so they leave exactly what the flat walk leaves on its way
 * to the same report.
 * ========================================================================== */

#define MODE_D_PACKET_LOCAL_STACK 16      /* continuations the walker keeps itself; the oldest moves to the frontier */
#define MODE_D_PACKET_CHUNK       256     /* leaves recorded between two evaluations by the members: small enough that the
                                            scratch it costs leaves room for ~10 resident packets per compute unit */
#define MODE_D_PACKET_STEPS       64      /* node steps the walker takes per round */
#define MODE_D_PACKET_TEAM_MAX    256     /* members, and so threads, per packet at most */

enum {
    MODE_D_PACKET_FAIL_NONE = 0,
    MODE_D_PACKET_FAIL_MALFORMED,        /* an index of no class, or a start past the tree's storage: fatal (the replay reports it too) */
    MODE_D_PACKET_FAIL_NO_CONTINUATION,  /* local stack and frontier both full */
    MODE_D_PACKET_FAIL_NO_PROGRESS,      /* a round in which the walker neither advanced nor waited on the chunk */
    MODE_D_PACKET_FAIL_START_ORDER,      /* received start nodes not in the top tree's depth-first order */
    MODE_D_PACKET_FAIL_START_ROOM,       /* more distinct received start nodes than the frontier holds */
    MODE_D_PACKET_FAIL_REASONS
};

/* Whether a batch of M queries on this rank walks as packets, for a pass that needs fewer than `ratio`
 * lanes of the device per query to gain from sharing a traversal. */
static inline bool nlr_mode_d_packets_for(int M, int ratio, bool device_only = false)
{
    return TREE_QUERY_PACKET_SIZE > 1 && (device_only || nlr_walk_runs_on_device(M)) &&
           !device_packet_lanes_saturated(M, ratio, gizmo_gpu_lane_count());
}

/* A loop whose evaluation may run in packets: its pair kernel writes nothing but the query's own
 * accumulator, so a discarded packet has published nothing. */
template <typename Spec>
constexpr bool nlr_mode_d_evaluates_in_packets_v =
    (Spec::write_pattern == WritePattern::ActiveReduceOnly) &&
    (nlr_spec_modeb_eval_omp<Spec>() == ModeBEvalOMP::BitwiseReadonly) &&
    !nlr_spec_writes_neighbour_motion_v<Spec> && !Spec::uses_ghost_writeback &&
    std::is_same_v<typename Spec::ScatterData, NoScatter>;

/* The packet walk's device workspace: a give-up flag per batch member, the give-ups by reason, each
 * top-level node's position in the top tree's depth-first order (the order a sender's export walk appends
 * the start nodes it ships, which a packet of received queries orders their union by), and the single-rank
 * path's query buffer.  It persists across calls, grows only when a batch actually walks as packets, and is
 * released where the tree is freed (nlr_mode_d_packet_workspace_release).  A growth the device refuses keeps
 * the old allocation, sends only that batch down the flat walk, and is not asked for again until the next
 * release.  Its labels give it its own line in the memory ledger (ngl-packet, current and peak), and a
 * refused growth is reported once, by the rank it happened on. */
struct NlrModeDPacketWorkspace {
    int *member_failed = nullptr; int member_capacity = 0; int member_refused = 0;
    int *failed_by_reason = nullptr;
    int *top_rank = nullptr; int top_rank_capacity = 0; int top_rank_refused = 0;
    int top_rank_span = 0; int top_rank_current = 0;   /* the map describes the tree as built; cleared by every build */
    void *queries = nullptr; size_t query_bytes = 0; size_t query_refused = 0;
};
static NlrModeDPacketWorkspace g_mode_d_packets;

static void mode_d_lease_pool_release(void);
void nlr_mode_d_packet_workspace_release(void)
{
    mode_d_lease_pool_release();   /* and the device storage Mode D calls gave back */
    NlrModeDPacketWorkspace &w = g_mode_d_packets;
    if(w.member_failed)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.member_failed);}
    if(w.failed_by_reason) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.failed_by_reason);}
    if(w.top_rank)         {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.top_rank);}
    if(w.queries)          {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.queries);}
    w = NlrModeDPacketWorkspace{};
}

/* A growth the device refused: said once (it is not asked for again until the next release), by the rank. */
static void nlr_mode_d_packet_refused(const char *what, size_t bytes)
{
    printf("[Mode D packets] Task %d: the device refused %.1f MB for the packet walk's %s; batches needing more walk flat "
           "until the tree is next freed.\n", ThisTask, (double) bytes / (1024.0 * 1024.0), what);
    fflush(stdout);
}

/* Room for M batch members' give-up flags and the reason tally, or false: the batch then walks flat. */
static bool nlr_mode_d_packet_workspace_reserve(int M)
{
    NlrModeDPacketWorkspace &w = g_mode_d_packets;
    if(!w.failed_by_reason) {
        w.failed_by_reason = (int *) nlr_shared_alloc_bytes((size_t) MODE_D_PACKET_FAIL_REASONS * sizeof(int), "ngl_packet_reasons");
        if(!w.failed_by_reason) {return false;}
    }
    if(w.member_capacity >= M) {return true;}
    if(w.member_refused && M >= w.member_refused) {return false;}
    const int capacity = (M + M / 8 + 1 > M) ? (M + M / 8 + 1) : M;
    int *grown = (int *) nlr_shared_alloc_bytes((size_t) capacity * sizeof(int), "ngl_packet_member_failed");
    if(!grown) {w.member_refused = M; nlr_mode_d_packet_refused("give-up flags", (size_t) capacity * sizeof(int)); return false;}
    if(w.member_failed) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.member_failed);}
    w.member_failed = grown; w.member_capacity = capacity;
    return true;
}

/* Room for a single-rank batch's query buffer, or null: the batch then walks flat. */
static void *nlr_mode_d_packet_query_buffer(size_t bytes)
{
    NlrModeDPacketWorkspace &w = g_mode_d_packets;
    if(w.query_bytes >= bytes) {return w.queries;}
    if(w.query_refused && bytes >= w.query_refused) {return nullptr;}
    const size_t capacity = bytes + bytes / 8;
    void *grown = nlr_shared_alloc_bytes(capacity, "ngl_packet_queries");
    if(!grown) {w.query_refused = bytes; nlr_mode_d_packet_refused("query buffer", capacity); return nullptr;}
    if(w.queries) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.queries);}
    w.queries = grown; w.query_bytes = capacity;
    return w.queries;
}

/* Walk the top-level tree depth-first through the same node arrays the device walk reads, calling visit(kn)
 * on every top-level node in order: an internal top-level node descends to its first child, which is itself
 * top-level, and a top leaf continues at its sibling.  A top leaf's own child link is never followed -- it is
 * the one top-level link that changes between builds (pointer fixups when particles move between chains, and
 * the redirect of a remote top leaf into the imported tree) -- so the order computed after a build stays the
 * order until the next one.  Node indices are NOT depth-first (siblings are numbered before descendants),
 * which is why a rank has to be computed at all. */
template <class Visit>
static void nlr_mode_d_top_tree_depth_first(const GxDeviceTreeView &tree, Visit visit)
{
    int guard = 0;
    for(int no = tree.node_base; no >= tree.node_base && no < tree.foreign_base && guard < tree.node_capacity; guard++) {
        const int kn = no - tree.node_base;
        const unsigned int flags = tree.node_bitflags[kn];
        if(!(flags & (1u << BITFLAG_TOPLEVEL))) {no = tree.node_sibling[kn]; continue;}
        visit(kn);
        no = (flags & (1u << BITFLAG_INTERNAL_TOPLEVEL)) ? tree.node_nextnode[kn] : tree.node_sibling[kn];
    }
}

/* Each top-level node's depth-first rank, indexed by node offset over the span the top-level nodes occupy
 * (the first nodes a build creates); -1 for the other nodes in that span.  Computed from the node arrays the
 * device walk reads, at the first use after a build, and kept until the next build: the top-level links do not
 * change between builds (the imported-tree redirects run inside the build, and the in-place pointer fixups
 * between builds touch particle chains, not the top level).  Kept rather than recomputed per batch because
 * reading those arrays on the host pulls their pages away from the device on a backend without coherent
 * host access.  Null when it cannot be had (the batch walks flat). */
void nlr_mode_d_packet_tree_rebuilt(void) {g_mode_d_packets.top_rank_current = 0;}

static const int *nlr_mode_d_packet_top_rank(const GxDeviceTreeView &tree, int *span_out)
{
    NlrModeDPacketWorkspace &w = g_mode_d_packets;
    if(w.top_rank_current) {*span_out = w.top_rank_span; return w.top_rank;}
    int span = 0;
    nlr_mode_d_top_tree_depth_first(tree, [&](int kn) {if(kn + 1 > span) {span = kn + 1;}});
    if(span <= 0) {return nullptr;}
    if(w.top_rank_capacity < span) {
        if(w.top_rank_refused && span >= w.top_rank_refused) {return nullptr;}
        int *grown = (int *) nlr_shared_alloc_bytes((size_t) span * sizeof(int), "ngl_packet_top_rank");
        if(!grown) {w.top_rank_refused = span; nlr_mode_d_packet_refused("top-level order", (size_t) span * sizeof(int)); return nullptr;}
        if(w.top_rank) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(w.top_rank);}
        w.top_rank = grown; w.top_rank_capacity = span;
    }
    int *rank = w.top_rank;
    for(int kn = 0; kn < span; kn++) {rank[kn] = -1;}
    int r = 0;
    nlr_mode_d_top_tree_depth_first(tree, [&](int kn) {rank[kn] = r++;});
    w.top_rank_span = span; w.top_rank_current = 1;
    *span_out = span;
    return rank;
}

/* Where a packet's team keeps its state, in level-0 scratch: each member's query (position and reach) and
 * accumulator, the frontier, the record chunk, the walker's own continuations and masks, and the counters,
 * last.  The accumulators live here rather than in each member's registers because the walker is member 0:
 * a register accumulator would stay live through the whole traversal, so the walk would carry a member's
 * full state on top of its own -- the largest loops' accumulators then spill.  In scratch, a member's
 * accumulator is in registers only while that member evaluates its records. */
struct NlrModeDPacketPlan {
    size_t queries, accums, partials, frontier, frontier_masks, records, record_masks, local, local_masks, walker_masks, walker_state,
           member_state, counters, bytes;
};

/* The walker's position in the traversal, kept in team scratch between its rounds: it must survive each round
 * boundary and the members' evaluation of the record chunk, and kept in registers it would stay live through
 * that evaluation beside the members' accumulators. */
struct NlrModeDPacketWalkerState {int no, exit, item_live, local_head, local_count;};
static NlrModeDPacketPlan nlr_mode_d_packet_plan(int q, int walkers, int mask_words, int frontier_cap, int chunk_cap, size_t accum_bytes,
                                                 size_t member_state_bytes = 0)
{
    NlrModeDPacketPlan p;
    DevicePacketScratchLayout layout;
    p.queries        = layout.take((size_t) q * 4 * sizeof(double), alignof(double));
    p.accums         = layout.take((size_t) q * accum_bytes, 16);
    /* one partial result per thread while the team evaluates one member's records together */
    p.partials       = chunk_cap ? layout.take((size_t) q * accum_bytes, 16) : 0;
    p.frontier       = layout.take((size_t) frontier_cap * sizeof(device_packet_item_t), alignof(device_packet_item_t));
    p.frontier_masks = layout.take((size_t) frontier_cap * mask_words * sizeof(device_packet_mask_word_t), alignof(device_packet_mask_word_t));
    p.records        = layout.take((size_t) chunk_cap * sizeof(int), alignof(int));
    p.record_masks   = layout.take((size_t) chunk_cap * mask_words * sizeof(device_packet_mask_word_t), alignof(device_packet_mask_word_t));
    /* each walker's own continuations, its item's mask + open mask, and its position between rounds */
    p.local          = layout.take((size_t) walkers * MODE_D_PACKET_LOCAL_STACK * sizeof(device_packet_item_t), alignof(device_packet_item_t));
    p.local_masks    = layout.take((size_t) walkers * MODE_D_PACKET_LOCAL_STACK * mask_words * sizeof(device_packet_mask_word_t), alignof(device_packet_mask_word_t));
    p.walker_masks   = layout.take((size_t) walkers * 2 * mask_words * sizeof(device_packet_mask_word_t), alignof(device_packet_mask_word_t));
    p.walker_state   = layout.take((size_t) walkers * sizeof(NlrModeDPacketWalkerState), alignof(NlrModeDPacketWalkerState));
    /* A client that decides nodes itself keeps each member's running state here; the region exists only for
       such a client, so every other client's scratch request is unchanged. */
    p.member_state   = member_state_bytes ? layout.take((size_t) q * member_state_bytes, alignof(long long)) : 0;
    p.counters       = layout.take(DEVICE_PACKET_COUNTER_SLOTS * sizeof(int), alignof(long long));
    p.bytes = layout.bytes;
    return p;
}

/* One packet per team, thread m < q owning member m.  Threads [0, n_walkers) walk.  The recording client takes
 * one walker per member -- a single walker leaves the other lanes idle through a traversal that is a chain of
 * dependent loads, the shared frontier hands every opened node's sibling to an idle walker, and its claims are
 * idempotent, so their order does not matter.  Every other client keeps one walker: the export count and fill
 * write each member's running state in walk order, and an evaluating client's sums would otherwise depend on how
 * the walkers' record reservations interleave, which differs from run to run.  The members then evaluate each
 * chunk of records together, one member at a time, every thread taking every q-th record, so the members'
 * unequal record counts do not leave lanes idle.  A batch holds its root queries in
 * [0, n_local) and its received queries in [n_local, n_total); a packet never mixes the two, so its entry
 * is either the root or the union of its members' start nodes.
 *
 * The client supplies the batch and what happens at a leaf:
 *   static constexpr bool evaluates;  using Accum;   int n_local, n_total;
 *   GxDeviceTreeView tree; unsigned int type_mask; int *anomaly;
 *   void query(int kk, double *x4) const;                           position and reach
 *   void starts(int kk, const int *&nodes, int &n) const;           a received query's start nodes
 *   evaluating:  void begin(Accum &) const; void evaluate(int kk, int j, const double *x4, Accum &) const;
 *                void merge(Accum &into, const Accum &part) const;  void commit(int kk, const Accum &) const;
 *   recording:   void claim(int j) const; */
template <class Client>
struct NlrModeDPacketWalk {
    using TeamMember = Kokkos::TeamPolicy<>::member_type;
    Client cl;
    DevicePacketTraversal<MODE_D_PACKET_LOCAL_STACK> trav;
    NlrModeDPacketPlan plan;
    int q, n_local_packets;
    int *member_failed, *failed_by_reason;
    const int *top_rank; int top_rank_span;

    /* A node-deciding client records `tag` for every member in `members`, each into its own running state. */
    KOKKOS_INLINE_FUNCTION void record_members(const device_packet_mask_word_t *members, int tag, char *member_states) const
    {
        if constexpr (gx_walk_policy_has_node_hook_v<Client>) {
            using MemberState = typename Client::MemberState;
            MemberState *states = (MemberState *) member_states;
            for(int m = 0; m < q; m++) {if(device_packet_mask_test(members, m)) {cl.record(states[m], tag);}}
        }
    }

    /* The walker: advance its item until its step budget runs out, the chunk is full, it runs out of work,
     * or the packet gives up -- deciding every node the way gx_device_tree_walk_impl decides it, for every
     * member of the item at once. */
    KOKKOS_INLINE_FUNCTION void walk(int resume, int &no, int &exit, int &item_live, int &local_head, int &local_count,
                                     device_packet_mask_word_t *mask, device_packet_mask_word_t *open_mask,
                                     device_packet_item_t *local, device_packet_mask_word_t *lmasks,
                                     const double *queries, device_packet_item_t *frontier, device_packet_mask_word_t *fmasks,
                                     int *records, device_packet_mask_word_t *rmasks, int *ctr, char *member_states) const
    {
        const GxDeviceTreeView &tree = cl.tree;
        const int W = trav.mask_words;
        int budget = trav.steps_per_round, stepped = 0;
        while(1) {
            if(!item_live || no == exit || no < 0) {
                if(trav.next_item(no, exit, mask, item_live, local_head, local_count, local, lmasks, ctr, frontier, fmasks)) {continue;}
                break;
            }
            if(budget <= 0) {break;}   /* back to the round boundary, item intact */
            budget--; stepped++;
            if(no >= tree.particle_slots && no < tree.node_base) {
                Kokkos::atomic_store(cl.anomaly, GX_WALK_ANOMALY_MALFORMED_TREE);
                trav.fail(ctr, MODE_D_PACKET_FAIL_MALFORMED); break;
            }
            if(no < tree.particle_slots) {
                if(no < tree.local_particle_slots) {   /* imported ghosts above the owned locals are stepped over */
                    if constexpr (Client::evaluates) {
                        const int r = trav.record_reserve(ctr);
                        if(r < 0) {break;}   /* chunk full: the members evaluate it, then the walk resumes here */
                        records[r] = no; device_packet_mask_copy(rmasks + (size_t) r * W, mask, W);
                    } else {
                        cl.claim(no);
                    }
                }
                no = tree.nextnode_aux[no];
                continue;
            }
            if(no >= tree.pseudo_start) {   /* pseudo-particle: another rank's subtree, nothing local below it */
                if constexpr (gx_walk_policy_has_node_hook_v<Client>) {
                    const GxNodeDecision pseudo = cl.pseudo_decision(no - tree.pseudo_start);
                    if(pseudo.action == GxNodeAction::Malformed) {
                        Kokkos::atomic_store(cl.anomaly, GX_WALK_ANOMALY_MALFORMED_TREE);
                        trav.fail(ctr, MODE_D_PACKET_FAIL_MALFORMED); break;
                    }
                    if(pseudo.action == GxNodeAction::RecordAndSibling) {record_members(mask, pseudo.tag, member_states);}
                }
                no = tree.nextnode_aux[tree.particle_slots + (no - tree.pseudo_start)];
                continue;
            }
            const int kn = no - tree.node_base;
            if(kn < 0 || kn >= tree.node_capacity) {
                Kokkos::atomic_store(cl.anomaly, GX_WALK_ANOMALY_MALFORMED_TREE);
                trav.fail(ctr, MODE_D_PACKET_FAIL_MALFORMED); break;
            }
            /* a resumed walk ends where it re-enters the top-level tree; a walk from the root descends it */
            if(resume && (tree.node_bitflags[kn] & (1u << BITFLAG_TOPLEVEL))) {item_live = 0; continue;}
            [[maybe_unused]] GxNodeDecision decision{GxNodeAction::Descend, -1};
            if constexpr (gx_walk_policy_has_node_hook_v<Client>) {
                decision = cl.node_decision(kn);
                if(decision.action == GxNodeAction::Malformed) {
                    Kokkos::atomic_store(cl.anomaly, GX_WALK_ANOMALY_MALFORMED_TREE);
                    trav.fail(ctr, MODE_D_PACKET_FAIL_MALFORMED); break;
                }
                if(decision.action == GxNodeAction::Sibling) {no = tree.node_sibling[kn]; continue;}
            }
            double len_eff = (double) tree.node_len[kn];
            if(tree.node_vmax && tree.node_ti && tree.drift_tables_ok) {
                const double dl = motion_bound_widening((double) tree.node_vmax[kn], tree.node_ti[kn], tree.ti_now, &tree.drift_tables);
                if(!motion_bound_widening_is_valid(dl)) {Kokkos::atomic_store(cl.anomaly, GX_WALK_ANOMALY_MALFORMED_TREE);}
                else {len_eff += dl;}
            }
            const double hw = 0.5 * len_eff;
            device_packet_mask_clear(open_mask, W);
            const int pruned = (cl.type_mask && tree.type_mask_trusted && !(NODE_TYPE_PRESENCE(tree.node_bitflags[kn]) & cl.type_mask));
            if(!pruned) {
                for(int m = 0; m < q; m++) {
                    if(!device_packet_mask_test(mask, m)) {continue;}
                    const double *x = queries + 4 * m;
                    if(gx_extended_overlap_wrap_and_test((double) tree.node_center[kn][0] - x[0], (double) tree.node_center[kn][1] - x[1],
                                                         (double) tree.node_center[kn][2] - x[2], hw, hw, hw, x[3])) {device_packet_mask_set(open_mask, m);}
                }
            }
            const int child = tree.node_nextnode[kn], sibling = tree.node_sibling[kn];
            if constexpr (gx_walk_policy_has_node_hook_v<Client>) {
                if(decision.action == GxNodeAction::RecordAndSibling) {record_members(open_mask, decision.tag, member_states); no = sibling; continue;}
            }
            /* nobody opens it, or it is an imported foreign subtree, which holds no locally-owned particles */
            if(!device_packet_mask_any(open_mask, W) || (child >= tree.foreign_base && child < tree.pseudo_start)) {no = sibling; continue;}
            if(!trav.descend(child, sibling, no, exit, mask, open_mask, local_head, local_count, local, lmasks,
                             ctr, frontier, fmasks, MODE_D_PACKET_FAIL_NO_CONTINUATION)) {break;}
        }
        if(stepped) {trav.ctr_add(&ctr[DEVICE_PACKET_CTR_STEPPED], stepped);}
    }

    /* The received members' entries: one item per distinct start node, its mask the members that carry it.
     * Every member's list must be strictly increasing in the top tree's depth-first rank (the order its
     * sender's export walk appended them in), and the union is pushed in decreasing rank, so the walker's
     * newest-first pop takes the items in each member's own order.  Run by thread 0; returns 0 when the
     * packet gave up. */
    KOKKOS_INLINE_FUNCTION int received_entries(int first, int q_eff, device_packet_item_t *frontier, device_packet_mask_word_t *fmasks, int *ctr) const
    {
        const GxDeviceTreeView &tree = cl.tree;
        const int W = trav.mask_words;
        int nu = 0;
        for(int m = 0; m < q_eff; m++) {
            const int *starts; int n_start;
            cl.starts(first + m, starts, n_start);
            int last_rank = -1;
            for(int k = 0; k < n_start; k++) {
                const int start = starts[k];
                const GxStartNode cls = gx_classify_start_node(start, tree);
                if(cls == GxStartNode::EndOfList) {break;}
                if(cls == GxStartNode::Skip) {continue;}
                if(cls == GxStartNode::Malformed) {
                    Kokkos::atomic_store(cl.anomaly, GX_WALK_ANOMALY_MALFORMED_TREE);
                    trav.fail(ctr, MODE_D_PACKET_FAIL_MALFORMED); return 0;
                }
                const int kn = start - tree.node_base;
                if(kn >= top_rank_span || top_rank[kn] < 0 || top_rank[kn] <= last_rank) {   /* not top-level, or out of order */
                    trav.fail(ctr, MODE_D_PACKET_FAIL_START_ORDER); return 0;
                }
                last_rank = top_rank[kn];
                int pos = -1;
                for(int u = 0; u < nu; u++) {if(frontier[u].no == start) {pos = u; break;}}
                if(pos < 0) {
                    if(nu >= trav.frontier_cap) {trav.fail(ctr, MODE_D_PACKET_FAIL_START_ROOM); return 0;}
                    pos = nu++;
                    frontier[pos].no = start; frontier[pos].exit = last_rank;   /* exit holds the rank until the sort */
                    device_packet_mask_clear(fmasks + (size_t) pos * W, W);
                }
                device_packet_mask_set(fmasks + (size_t) pos * W, m);
            }
        }
        /* decreasing rank (insertion sort over a handful of start nodes, each mask moving with its item),
           then each start node becomes its item */
        for(int a = 1; a < nu; a++) {
            const device_packet_item_t it = frontier[a];
            device_packet_mask_word_t it_mask[MODE_D_PACKET_TEAM_MAX / DEVICE_PACKET_MASK_BITS];
            device_packet_mask_copy(it_mask, fmasks + (size_t) a * W, W);
            int b = a - 1;
            while(b >= 0 && frontier[b].exit < it.exit) {
                frontier[b + 1] = frontier[b];
                device_packet_mask_copy(fmasks + (size_t) (b + 1) * W, fmasks + (size_t) b * W, W);
                b--;
            }
            frontier[b + 1] = it;
            device_packet_mask_copy(fmasks + (size_t) (b + 1) * W, it_mask, W);
        }
        for(int u = 0; u < nu; u++) {
            const int kn = frontier[u].no - tree.node_base;
            frontier[u].no = tree.node_nextnode[kn]; frontier[u].exit = tree.node_sibling[kn];   /* open the exported node */
        }
        ctr[DEVICE_PACKET_CTR_FR_PUB] = nu;
        return 1;
    }

    /* A region of the team's scratch.  Each phase takes the regions it uses, so that none is held across the
     * phases that do not. */
    template <class T> KOKKOS_INLINE_FUNCTION static T *region(char *scratch, size_t offset) {return (T *) (scratch + offset);}

    KOKKOS_INLINE_FUNCTION void operator()(const TeamMember &team) const
    {
        const int W = trav.mask_words;
        const int p = team.league_rank(), t = team.team_rank();
        const int resume = (p >= n_local_packets);
        const int first = resume ? cl.n_local + (p - n_local_packets) * q : p * q;
        const int end   = resume ? cl.n_total : cl.n_local;
        const int q_eff = (end - first < q) ? (end - first) : q;
        char *scratch = (char *) team.team_scratch(0).get_shmem(plan.bytes);
        int  *ctr     = region<int>(scratch, plan.counters);

        if(t < q_eff) {cl.query(first + t, region<double>(scratch, plan.queries) + 4 * t);}
        if(t == 0) {for(int c = 0; c < DEVICE_PACKET_COUNTER_SLOTS; c++) {ctr[c] = 0;}}
        team.team_barrier();

        /* Walker 0 holds the packet's entry -- the root, or the received members' start nodes on the frontier --
           and the others start idle and take what is published; all of it set before any walker pops. */
        if(t < trav.n_walkers) {
            device_packet_mask_word_t *mask = region<device_packet_mask_word_t>(scratch, plan.walker_masks) + (size_t) t * 2 * W;
            NlrModeDPacketWalkerState &ws = region<NlrModeDPacketWalkerState>(scratch, plan.walker_state)[t];
            ws.no = -1; ws.exit = -1; ws.item_live = 0; ws.local_head = 0; ws.local_count = 0;
            device_packet_mask_clear(mask, W);
            if(t == 0 && !resume) {
                for(int m = 0; m < q_eff; m++) {device_packet_mask_set(mask, m);}
                ws.no = cl.tree.node_base; ws.exit = -1; ws.item_live = (q_eff > 0);   /* the root: its own overlap test runs */
            } else if(t == 0) {
                received_entries(first, q_eff, region<device_packet_item_t>(scratch, plan.frontier),
                                 region<device_packet_mask_word_t>(scratch, plan.frontier_masks), ctr);   /* a failure is seen at the first round boundary */
            }
        }
        if constexpr (Client::evaluates) {if(t < q_eff) {cl.begin(region<typename Client::Accum>(scratch, plan.accums)[t]);}}
        if constexpr (gx_walk_policy_has_node_hook_v<Client>) {
            if(t < q_eff) {cl.member_begin(first + t, region<typename Client::MemberState>(scratch, plan.member_state)[t]);}
        }
        team.team_barrier();

        while(1) {
            if(t < trav.n_walkers && !ctr[DEVICE_PACKET_CTR_FAILED]) {
                /* the walkers' phase: each one's state from scratch, its traversal, its state back */
                NlrModeDPacketWalkerState &ws = region<NlrModeDPacketWalkerState>(scratch, plan.walker_state)[t];
                device_packet_mask_word_t *mask = region<device_packet_mask_word_t>(scratch, plan.walker_masks) + (size_t) t * 2 * W;
                int no = ws.no, exit = ws.exit, item_live = ws.item_live, local_head = ws.local_head, local_count = ws.local_count;
                walk(resume, no, exit, item_live, local_head, local_count, mask, mask + W,
                     region<device_packet_item_t>(scratch, plan.local) + (size_t) t * MODE_D_PACKET_LOCAL_STACK,
                     region<device_packet_mask_word_t>(scratch, plan.local_masks) + (size_t) t * MODE_D_PACKET_LOCAL_STACK * W,
                     region<double>(scratch, plan.queries),
                     region<device_packet_item_t>(scratch, plan.frontier), region<device_packet_mask_word_t>(scratch, plan.frontier_masks),
                     region<int>(scratch, plan.records), region<device_packet_mask_word_t>(scratch, plan.record_masks), ctr,
                     scratch + plan.member_state);
                if(item_live || local_count > 0) {trav.ctr_add(&ctr[DEVICE_PACKET_CTR_WORKING], 1);}
                ws.no = no; ws.exit = exit; ws.item_live = item_live; ws.local_head = local_head; ws.local_count = local_count;
            }
            const int chunk_full = trav.round_boundary(team, ctr, region<device_packet_item_t>(scratch, plan.frontier),
                                                       region<device_packet_mask_word_t>(scratch, plan.frontier_masks),
                                                       MODE_D_PACKET_FAIL_NO_PROGRESS);
            if(chunk_full < 0) {break;}   /* the packet gave up */
            if constexpr (Client::evaluates) {
                /* The chunk is full, or the traversal has finished: the team evaluates it one member at a time, thread
                   t taking records t, t + q, ...; thread 0 then adds the threads' partial results to the member's own in
                   thread order, so the sum is the same on every run. */
                if(ctr[DEVICE_PACKET_CTR_RECORDS] > 0 && (chunk_full || ctr[DEVICE_PACKET_CTR_DONE])) {
                    const int *records = region<int>(scratch, plan.records);
                    const device_packet_mask_word_t *rmasks = region<device_packet_mask_word_t>(scratch, plan.record_masks);
                    typename Client::Accum *accums   = region<typename Client::Accum>(scratch, plan.accums);
                    typename Client::Accum *partials = region<typename Client::Accum>(scratch, plan.partials);
                    const int n_rec = ctr[DEVICE_PACKET_CTR_RECORDS];
                    for(int m = 0; m < q_eff; m++) {
                        const double *x = region<double>(scratch, plan.queries) + 4 * m;
                        typename Client::Accum acc;   /* in registers for this member's share only */
                        cl.begin(acc);
                        for(int r = t; r < n_rec; r += q) {
                            if(device_packet_mask_test(rmasks + (size_t) r * W, m)) {cl.evaluate(first + m, records[r], x, acc);}
                        }
                        partials[t] = acc;
                        team.team_barrier();
                        if(t == 0) {for(int l = 0; l < q; l++) {cl.merge(accums[m], partials[l]);}}
                        team.team_barrier();
                    }
                    if(t == 0) {ctr[DEVICE_PACKET_CTR_RECORDS] = 0;}
                    team.team_barrier();
                }
            }
            if(ctr[DEVICE_PACKET_CTR_DONE]) {break;}
        }

        if(ctr[DEVICE_PACKET_CTR_FAILED]) {
            const int reason = ctr[DEVICE_PACKET_CTR_FAIL_REASON];
            if(t == 0) {Kokkos::atomic_fetch_add(&failed_by_reason[(reason > 0 && reason < MODE_D_PACKET_FAIL_REASONS) ? reason : 0], 1);}
            if(t < q_eff) {member_failed[first + t] = 1;}   /* replayed flat; a malformed index then raises the flat walk's own report */
            return;
        }
        if constexpr (Client::evaluates) {if(t < q_eff) {cl.commit(first + t, region<typename Client::Accum>(scratch, plan.accums)[t]);}}
        /* Published only by a packet that completed: one that gave up leaves nothing, and its members are replayed from the start. */
        if constexpr (gx_walk_policy_has_node_hook_v<Client>) {if(t < q_eff) {cl.member_commit(first + t, region<typename Client::MemberState>(scratch, plan.member_state)[t]);}}
    }
};

/* The per-member state a node-deciding client keeps in team scratch; none for any other client. */
template <class Client>
constexpr size_t nlr_packet_member_state_bytes(void)
{
    if constexpr (gx_walk_policy_has_node_hook_v<Client>) {return sizeof(typename Client::MemberState);}
    else {return 0;}
}

/* Run a batch as packets.  Returns true with each member's give-up flag in the workspace (the caller replays
 * the flagged members flat), or false when the batch cannot walk as packets here -- no workspace, no legal
 * team, or no depth-first ranks for its received members -- in which case nothing has run and the caller
 * walks the whole batch flat.  Every allocation happens before the launch, so nothing a packet publishes is
 * ever followed by a refusal. */
template <class Client>
static bool nlr_mode_d_packet_run(const Client &cl, const char *label, const int **member_failed_out)
{
    const int M = cl.n_total;
    /* an unfilled view is the flat walk's to report: it stops the run on it */
    if(M <= 0 || cl.tree.local_particle_slots < 0 || !nlr_mode_d_packet_workspace_reserve(M)) {return false;}
    const int *top_rank = nullptr; int top_rank_span = 0;
    if(M > cl.n_local) {top_rank = nlr_mode_d_packet_top_rank(cl.tree, &top_rank_span); if(!top_rank) {return false;}}
    NlrModeDPacketWorkspace &w = g_mode_d_packets;

    NlrModeDPacketWalk<Client> f{cl};
    int q = (TREE_QUERY_PACKET_SIZE < MODE_D_PACKET_TEAM_MAX) ? TREE_QUERY_PACKET_SIZE : MODE_D_PACKET_TEAM_MAX;
    f.trav.steps_per_round = MODE_D_PACKET_STEPS;
    f.trav.chunk_cap = Client::evaluates ? MODE_D_PACKET_CHUNK : 1;   /* a recording packet reserves no records */
    /* The frontier is the single walker's overflow store for its oldest continuations, and the home of a
     * received packet's entries; it is sized for both rather than left at zero. */
    auto shape = [&](int qq) {
        f.q = qq; f.trav.mask_words = device_packet_mask_words(qq);
        f.trav.frontier_cap = (qq * NODELISTLENGTH > 16) ? qq * NODELISTLENGTH : 16;
        f.trav.n_walkers = (gx_walk_policy_has_node_hook_v<Client> || Client::evaluates) ? 1 : qq;   /* only claims are order-free */
        f.plan = nlr_mode_d_packet_plan(qq, f.trav.n_walkers, f.trav.mask_words, f.trav.frontier_cap, Client::evaluates ? MODE_D_PACKET_CHUNK : 0,
                                        Client::evaluates ? sizeof(typename Client::Accum) : 0, nlr_packet_member_state_bytes<Client>());
    };
    /* The packet width this backend can launch is found once and kept: it is a property of the kernel, the
     * backend and the scratch request, none of which change during a run.  A packet wider than the legal
     * team is walked as more, narrower packets, never truncated; 0 means no width of 2 or more launches. */
    static int launch_width = -1;
    if(launch_width < 0) {
        int qq = q;
        while(qq >= 2) {shape(qq); if(gizmo_gpu_team_size_max(f, f.plan.bytes) >= qq) {break;} qq--;}
        launch_width = (qq >= 2) ? qq : 0;
    }
    if(launch_width < 2) {return false;}
    q = launch_width;
    shape(q);
    f.n_local_packets = (cl.n_local + q - 1) / q;
    const int league = f.n_local_packets + (M - cl.n_local + q - 1) / q;
    f.member_failed = w.member_failed; f.failed_by_reason = w.failed_by_reason;
    f.top_rank = top_rank; f.top_rank_span = top_rank_span;
    int *mf = w.member_failed, *fr = w.failed_by_reason;
    gizmo_gpu_kernel_launch("ngl_packet_clear", M, KOKKOS_LAMBDA(int k) {
        mf[k] = 0;
        if(k < MODE_D_PACKET_FAIL_REASONS) {fr[k] = 0;}
    });
    if(M < MODE_D_PACKET_FAIL_REASONS) {for(int k = M; k < MODE_D_PACKET_FAIL_REASONS; k++) {fr[k] = 0;}}
    gizmo_gpu_team_kernel_launch(label, league, q, f.plan.bytes, f);
    /* Members to replay only when some packet gave up for a reason that replays. */
    int replay = 0;
    for(int k = 1; k < MODE_D_PACKET_FAIL_REASONS; k++) {replay += w.failed_by_reason[k];}
    *member_failed_out = replay ? w.member_failed : nullptr;
    return true;
}

/* The evaluating client: the batch's queries as the fused walk holds them, root queries first, and the
 * production leaf at every leaf a member's own decisions reached. */
template <typename Spec, bool Mirror>
struct NlrModeDPacketEvaluate {
    static constexpr bool evaluates = true;
    using Accum = typename Spec::AccumData;
    using ActiveData = typename Spec::ActiveData;
    using Envelope = NlrQueryEnvelope<ActiveData>;
    using Reply = NlrReplyEnvelope<Accum>;
    typename Spec::DeviceContext ctx;
    typename Spec::CallScalars cs;
    const ActiveData *ql = nullptr;                    /* root queries */
    const Envelope *received = nullptr;                /* received queries, read in place: each carries its start nodes */
    Reply *replies = nullptr;                          /* one per received query */
    const double *reach = nullptr;                     /* the search radius when the caller walks with its own; else the query's */
    int n_local = 0, n_total = 0;
    GxDeviceTreeView tree;
    unsigned int type_mask = 0;
    int *anomaly = nullptr;
    struct GxMotionTargetSet motion_targets{};
    GxLeafMirrorView leaf_mirror{};
    struct GxTouchedSet mass_verdicts{};
    int use_mass_verdict = 0;
    Accum *acc_out = nullptr;

    KOKKOS_INLINE_FUNCTION const ActiveData &act(int kk) const {return (kk < n_local) ? ql[kk] : received[kk - n_local].active;}
    KOKKOS_INLINE_FUNCTION void query(int kk, double *x4) const
    {
        const ActiveData &a = act(kk);
        x4[0] = (double) a.pos[0]; x4[1] = (double) a.pos[1]; x4[2] = (double) a.pos[2];
        x4[3] = reach ? reach[kk] : (double) a.h_search;
    }
    KOKKOS_INLINE_FUNCTION void starts(int kk, const int *&nodes, int &n) const
    {
        const Envelope &env = received[kk - n_local];
        nodes = env.NodeList; n = nlr_mode_d_envelope_nodes(env);
    }
    KOKKOS_INLINE_FUNCTION void begin(Accum &acc) const {memset((void *) &acc, 0, sizeof(Accum)); Spec::zero_accum(acc);}
    KOKKOS_INLINE_FUNCTION void evaluate(int kk, int j, const double *x4, Accum &acc) const
    {
        typename Spec::ScatterData sd{};
        NlrModeDReduceLeaf<Spec, Mirror> leaf{&ctx, &act(kk), &acc, &sd, type_mask, &cs};
        leaf.motion_targets = motion_targets;
        if constexpr (Mirror) {leaf.leaf_mirror = leaf_mirror;}
        leaf.mass_verdicts = mass_verdicts; leaf.use_mass_verdict = use_mass_verdict;
        leaf.visit(j, x4[0], x4[1], x4[2], x4[3]);
    }
    KOKKOS_INLINE_FUNCTION void merge(Accum &into, const Accum &part) const {Spec::merge_accum(into, part);}
    KOKKOS_INLINE_FUNCTION void commit(int kk, const Accum &acc) const
    {
        if(kk < n_local) {acc_out[kk] = acc; return;}
        const Envelope &env = received[kk - n_local];
        Reply &r = replies[kk - n_local];
        r.origin_slot = env.origin_slot; r.origin_rank = env.origin_rank; r.accum = acc;
    }
};

/* The recording client: one claim per reached local slot, through the production recording leaf. */
template <class QueryFn>
struct NlrModeDPacketRecord {
    static constexpr bool evaluates = false;
    struct Accum {};
    QueryFn query_fn;
    int n_local = 0, n_total = 0;
    GxDeviceTreeView tree;
    unsigned int type_mask = 0;
    int *anomaly = nullptr;
    NlrRecordLeaf leaf{};

    KOKKOS_INLINE_FUNCTION void query(int kk, double *x4) const
    {
        const int *nodes; int n;
        query_fn(kk, x4[0], x4[1], x4[2], x4[3], nodes, n);
    }
    KOKKOS_INLINE_FUNCTION void starts(int kk, const int *&nodes, int &n) const
    {
        double x, y, z, reach;
        query_fn(kk, x, y, z, reach, nodes, n);
        if(n < 0) {n = 0;}
    }
    KOKKOS_INLINE_FUNCTION void claim(int j) const {leaf.visit(j, 0, 0, 0, 0);}
};

/* Evaluate a batch -- root queries [0, n_local), received ones after -- as packets.  Returns false when it
 * did not run (the caller walks the whole batch flat); true with *replay set to the members to walk flat, or
 * null when every packet completed. */
template <typename Spec>
static bool nlr_mode_d_packet_evaluate(const typename Spec::DeviceContext &ctx, const typename Spec::CallScalars &cs,
                                       const typename Spec::ActiveData *ql,
                                       const NlrQueryEnvelope<typename Spec::ActiveData> *received,
                                       NlrReplyEnvelope<typename Spec::AccumData> *replies, const double *reach, int n_local, int M,
                                       const GxDeviceTreeView &tree, unsigned int type_mask, int *anomaly,
                                       const struct GxMotionTargetSet &motion_targets, const GxLeafMirrorView &leaf_mirror,
                                       const struct GxTouchedSet &mass_verdicts, int use_mass_verdict,
                                       typename Spec::AccumData *acc_out, const int **replay)
{
    auto fill = [&](auto &cl) {
        cl.ctx = ctx; cl.cs = cs; cl.ql = ql; cl.received = received; cl.replies = replies; cl.reach = reach;
        cl.n_local = n_local; cl.n_total = M; cl.tree = tree; cl.type_mask = type_mask; cl.anomaly = anomaly;
        cl.motion_targets = motion_targets; cl.mass_verdicts = mass_verdicts; cl.use_mass_verdict = use_mass_verdict;
        cl.acc_out = acc_out;
    };
    if(leaf_mirror.pos && leaf_mirror.type) {
        NlrModeDPacketEvaluate<Spec, true> cl; fill(cl); cl.leaf_mirror = leaf_mirror;
        return nlr_mode_d_packet_run(cl, Spec::loop_name, replay);
    }
    NlrModeDPacketEvaluate<Spec, false> cl; fill(cl);
    return nlr_mode_d_packet_run(cl, Spec::loop_name, replay);
}

/* Bring current exactly the local particles a fused walk is about to reach.
 *
 * Runs the traversal once with the recording policy, waits for it, advances the
 * distinct set it recorded, and leaves the cursor ready for the next pass.  The
 * evaluation that follows then walks the same tree with the same queries and
 * reaches the same leaves, now current.
 *
 * Called immediately before each evaluation rather than once per call, because
 * the search radius is what the iteration changes: a later pass can reach
 * further than an earlier one.  It costs little to repeat -- a particle advanced
 * for an earlier pass stays current for the rest of the call, and the generation
 * is per call, so the second and later passes record only what is new.
 *
 * One pass covers this rank's own queries and the ones its peers sent, in a
 * single traversal over both.  They ask the same question of the same tree and
 * write the same touched set, and which of the two a given local particle was
 * reached by is an accident of where the domain boundary fell, so separating
 * them bought nothing and cost a launch, a fence and a second drift pass.  The
 * two kinds keep the two ENTRIES the traversal already has -- this rank's root,
 * or the start nodes a peer exported -- and `query` says which this work item
 * takes by yielding a start list or none.
 *
 * `query` is device-callable and yields this work item's position, reach, and
 * either its start-node list or a negative count meaning "walk from the root".
 * Callers keep the two kinds in CONTIGUOUS index ranges, the `n_root` root
 * queries first, so that the entry a work item takes does not alternate between
 * neighbouring lanes and a packet never mixes the two. */
template <class QueryFn>
static void nlr_record_and_drift(const struct particle_data *P,
                                 unsigned int supply_mask,
                                 int use_mass_verdict,
                                 const unsigned char *mirror_type,
                                 const GxDeviceTreeView &tree,
                                 int n, int n_root, int *anomaly,
                                 const char *label,
                                 QueryFn query,
                                 bool device_only = false)
{
    if(n <= 0) {return;}
    /* Nothing to discover when the whole rank is already uniform: something else
     * in the step -- a tree build, a decomposition, an output -- has published
     * the full-drift certificate, and every leaf this walk can reach is current
     * by that proof.  Recording them would be a traversal spent to learn that
     * there is no work, which on an all-active call is the largest traversal of
     * the step.  This is the same early-out the neighbour-list hook applies for
     * the same reason.
     *
     * The local certificate alone is the right test here, unlike there: this
     * path imports no ghosts, and the traversal stops at the owned slots, so
     * there is no imported segment to vouch for.
     *
     * Rank-local, and safe to be: neither this test nor the drift it guards
     * enters a collective, so a rank that skips and a rank that does not still
     * meet at the same place. */
    if(gizmo_full_drift_ti() == All.Ti_Current) {return;}
    const struct GxTouchedSet ts = gx_touched_set_view();
    /* The preparation declines collectively when it cannot hold this, so an
     * absent workspace here is not a state to recover from -- but it must not be
     * walked past either, because the drift that would not happen is a particle
     * evaluated at a stale position.  Reported through the channel the caller
     * already treats as fatal, rather than returned, so this adds no path out of
     * an exchange that a peer is waiting on. */
    if(!ts.seen || !ts.list || !ts.counter) {
        /* in device memory when device_only: raised there */
        if(device_only) {gizmo_gpu_kernel_launch("nlr_mode_d_record_unavailable", 1, KOKKOS_LAMBDA(int) {Kokkos::atomic_store(anomaly, GX_WALK_ANOMALY_TOUCHED_SET_FULL);});}
        else {Kokkos::atomic_store(anomaly, GX_WALK_ANOMALY_TOUCHED_SET_FULL);}
        return;
    }
    /* All is the caller's to have brought current, once for its call or exchange: this runs per evaluation batch. */
    /* In the packet band the batch records as packets; members of a packet that gave up are then recorded by
     * the flat walk below, which completes the set (the claims a discarded packet made are kept: they are
     * deduplicated per call, and the drift consumes the set once). */
    const int *replay = nullptr;
    if(nlr_mode_d_packets_for(n, 4, device_only)) {
        NlrModeDPacketRecord<QueryFn> cl{query};
        cl.n_local = n_root; cl.n_total = n; cl.tree = tree; cl.type_mask = supply_mask; cl.anomaly = anomaly;
        cl.leaf = NlrRecordLeaf{P, ts, supply_mask, anomaly};
        cl.leaf.use_mass_verdict = use_mass_verdict; cl.leaf.mirror_type = mirror_type;
        if(nlr_mode_d_packet_run(cl, "nlr_mode_d_record_packets", &replay) && !replay) {
            gx_touched_set_drift_and_mark(All.Ti_Current);
            return;
        }
    }
    nlr_mode_d_launch(device_only, label, n, KOKKOS_LAMBDA(int kk) {
        if(replay && !replay[kk]) {return;}   /* only the members of packets that gave up */
        double qx = 0, qy = 0, qz = 0, reach = 0;
        const int *start_nodes = nullptr;
        int n_start = -1;                 /* negative: this one walks from the root */
        query(kk, qx, qy, qz, reach, start_nodes, n_start);
        NlrRecordLeaf leaf{P, ts, supply_mask, anomaly};
        leaf.use_mass_verdict = use_mass_verdict;
        leaf.mirror_type = mirror_type;
        if(n_start < 0) {
            gx_device_tree_walk_from_root(qx, qy, qz, reach, tree, leaf, anomaly, supply_mask);
        } else if(n_start > 0) {
            /* Zero start nodes means nothing on this rank was exported to this
             * query, so there is nothing of ours for it to reach -- the same
             * skip the evaluating walk makes at the same data. */
            gx_device_tree_walk(qx, qy, qz, reach, start_nodes, n_start, tree, leaf, anomaly, supply_mask);
        }
    });
    gx_touched_set_drift_and_mark(All.Ti_Current);
}

/* One work item per active: build its query, walk this rank's tree from the
 * root, reduce into its own accumulator.
 *
 * The query is built HERE, on the device, from the resident particle arrays.
 * The host-walker path builds the equivalent on the host, one active at a time,
 * which is right where it runs -- a few actives -- and wrong here, where it
 * would be a serial host pass over the canonical arrays proportional to the
 * active count, on the steps that have the most actives.
 *
 * `radii` rather than the query's own stored radius: the search radius is the
 * runner's, it is what the iteration mutates between passes, and it is what the
 * host walk is given.  Reading it from anywhere else would let the two drift.
 *
 * Ownership is one accumulator per work item, so nothing is shared and nothing
 * needs an atomic.  A traversal that gave several work items a share of one
 * query -- one team per node pair, say -- would not have that property, which is
 * why it is stated rather than assumed. */
template <typename Spec>
static void nlr_mode_d_local_reduce(const typename Spec::DeviceContext &ctx,
                                    const typename Spec::CallScalars &cs,
                                    const int *active_idx,
                                    const int *active_slot,
                                    const double *radii,
                                    int n,
                                    unsigned int supply_mask,
                                    const GxDeviceTreeView &tree,
                                    typename Spec::AccumData *accums_out,
                                    int *anomaly)
{
    using ActiveData  = typename Spec::ActiveData;
    using ScatterData = typename Spec::ScatterData;

    static_assert(nlr_spec_modeb_eval_omp<Spec>() != ModeBEvalOMP::SerialOnly,
                  "Mode D evaluates a seeker's pairs on the rank that owns the neighbours, so a "
                  "pair kernel's neighbour-side writes land on the owner's own particles and need "
                  "no writeback -- but they land from many device lanes at once, so a kernel that "
                  "must run serially (a read-then-write of live neighbour state) cannot be served.");
    static_assert(Spec::search_mode == MODE_B_SEARCH_ONEWAY,
                  "Mode D prunes on the query's reach alone. A symmetric search also needs the "
                  "per-type supply bands, which are not mirrored to the device.");
    static_assert(!nlr_spec_has_bind_active_to_eval_context_v<Spec>,
                  "This Spec rebinds each active to the evaluating context before its pair kernel "
                  "runs, which the host backend does inside evaluate_pairs_post_drift. The fused "
                  "backend evaluates in a device kernel and performs no such rebind, so a received "
                  "active would still carry the sending rank's pointers. Port the rebind into the "
                  "device path before serving this Spec.");

    /* ADMISSION INVARIANT, not checkable here: EVERY ACTIVE THIS BACKEND IS GIVEN
     * IS ALREADY CURRENT AT All.Ti_Current.
     *
     * The three tests above are compile-time properties of the Spec.  This fourth
     * requirement is a property of the LIST the Spec hands in, so it cannot be
     * asserted -- but it is load-bearing in the same way, and this is where a new
     * Spec is admitted, so it is stated here rather than only where the queries
     * get built.
     *
     * It holds today by construction: core/run.cc's sync point drifts every
     * particle in every active time bin to All.Ti_Current, ActiveParticleList is
     * built from those same bins, nlr_build_active_list filters that list, and
     * nothing advances All.Ti_Current again inside the step.
     *
     * What breaks if a future Spec supplies an independent active list: this
     * backend snapshots its queries BEFORE the recording pass -- frozen ActiveData
     * on the transport path, packed envelopes on the peer path -- so a stale
     * active is baked into a snapshot no later drift can repair.  Drifting at the
     * record site would patch only the one site that re-derives its query, and
     * would leave the other two silently wrong while looking defended.  The only
     * correct remedy is to establish currentness BEFORE the first query snapshot
     * or transport packing, as part of admitting that Spec -- a change to this
     * backend's admission, not a defensive line inside the record loop. */


    if(n <= 0) {return;}

    /* The traversal reads All.BoxSize through the wrap macros, which resolve to
     * this unit's mirror; without this the box reads as zero on device and
     * periodic wrapping stops, silently and only on HIP. */
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Record what this walk will reach and bring just those current, before it
     * runs for real.  Same tree, same queries, so the same leaves.  The record
     * tests types on the leaf mirror as it stands; the mirror is brought current
     * again after the drift, whose marks it takes, before the evaluation reads
     * positions from it. */
    const int use_mass_verdict = nlr_mode_d_keeps_mass_verdict<Spec>(nlr_walk_runs_on_device(n));
    const GxLeafMirrorView record_mirror = nlr_mode_d_leaf_mirror(nlr_walk_runs_on_device(n), tree);
    nlr_record_and_drift(
        ctx.P, supply_mask, use_mass_verdict, record_mirror.type, tree, n, n, anomaly, "nlr_mode_d_self_record",
        KOKKOS_LAMBDA(int kk, double &qx, double &qy, double &qz, double &reach,
                      const int *&start_nodes, int &n_start) {
            const ActiveData a_rec = Spec::load_active(ctx, active_slot[kk], active_idx[kk], radii[kk], cs);
            qx = (double)a_rec.pos[0]; qy = (double)a_rec.pos[1]; qz = (double)a_rec.pos[2];
            reach = radii[kk];
            /* Single-rank: every query is this rank's own and walks from the root. */
            start_nodes = nullptr; n_start = -1;
        });

    const struct GxMotionTargetSet motion_targets = gx_motion_target_view();
    const struct GxTouchedSet mass_verdicts = use_mass_verdict ? gx_touched_set_view() : GxTouchedSet{};
    /* As on the multi-rank batch: the leaf mirror on a device batch when this rank has it, else the particle records. */
    const GxLeafMirrorView leaf_mirror = nlr_mode_d_leaf_mirror(nlr_walk_runs_on_device(n), tree);
    /* In the packet band the batch is evaluated by the same packet client as a multi-rank batch, every query a
     * root query, from a query buffer built where the particles are; the flat walk below then evaluates only
     * the members of packets that gave up, or the whole batch when packets did not run.  The buffer is built
     * only once packets are chosen, so a batch outside the band pays nothing for it. */
    bool packets_ran = false;
    const int *replay = nullptr;
    if constexpr (nlr_mode_d_evaluates_in_packets_v<Spec>) {
        if(nlr_mode_d_packets_for(n, 2)) {
            ActiveData *queries = (ActiveData *) nlr_mode_d_packet_query_buffer((size_t) n * sizeof(ActiveData));
            if(queries) {
                gizmo_gpu_kernel_launch("nlr_mode_d_build_packet_queries", n, KOKKOS_LAMBDA(int k) {
                    queries[k] = Spec::load_active(ctx, active_slot[k], active_idx[k], radii[k], cs);
                });
                /* the walk's reach is the runner's radius, as the recording pass and the flat walk take it */
                packets_ran = nlr_mode_d_packet_evaluate<Spec>(ctx, cs, queries, nullptr, nullptr, radii, n, n, tree, supply_mask,
                                                               anomaly, motion_targets, leaf_mirror, mass_verdicts, use_mass_verdict,
                                                               accums_out, &replay);
            }
        }
    }
    if(packets_ran && !replay) {
        /* every packet completed */
    } else if(leaf_mirror.pos && leaf_mirror.type) {
        nlr_walk_for_sources(Spec::loop_name, n, KOKKOS_LAMBDA(int kk) {
            if(replay && !replay[kk]) {return;}
            Spec::zero_accum(accums_out[kk]);
            const int i = active_idx[kk];
            ActiveData  a = Spec::load_active(ctx, active_slot[kk], i, radii[kk], cs);
            ScatterData s{};
            NlrModeDReduceLeaf<Spec, true> leaf{&ctx, &a, &accums_out[kk], &s, supply_mask, &cs};
            leaf.motion_targets = motion_targets;
            leaf.leaf_mirror = leaf_mirror;
            leaf.mass_verdicts = mass_verdicts; leaf.use_mass_verdict = use_mass_verdict;
            gx_device_tree_walk_from_root((double)a.pos[0], (double)a.pos[1], (double)a.pos[2],
                                          radii[kk], tree, leaf, anomaly, supply_mask);
        });
    } else {
        nlr_walk_for_sources(Spec::loop_name, n, KOKKOS_LAMBDA(int kk) {
            if(replay && !replay[kk]) {return;}
            Spec::zero_accum(accums_out[kk]);
            const int i = active_idx[kk];
            ActiveData  a = Spec::load_active(ctx, active_slot[kk], i, radii[kk], cs);
            ScatterData s{};
            NlrModeDReduceLeaf<Spec> leaf{&ctx, &a, &accums_out[kk], &s, supply_mask, &cs};
            leaf.motion_targets = motion_targets;
            leaf.mass_verdicts = mass_verdicts; leaf.use_mass_verdict = use_mass_verdict;
            gx_device_tree_walk_from_root((double)a.pos[0], (double)a.pos[1], (double)a.pos[2],
                                          radii[kk], tree, leaf, anomaly, supply_mask);
        });
    }
}


/* One evaluation launch sequence of an admitted multi-rank Mode D call: this rank's own queries (if any are
 * still unanswered) and a batch of received ones, answered from this rank's tree, on the device, whatever
 * their number -- the queries and the replies live in device memory.  No candidate list is built: the walk
 * evaluates where it lands.  The locals this walk reaches are brought current first by a recording pass over the
 * same queries and a drift of exactly what it recorded; a later batch can reach locals an earlier one did not,
 * which is why every batch discovers rather than relying on an earlier pass.
 *
 * The two kinds occupy contiguous index ranges, locals first, so the entry a work item takes does not alternate
 * between neighbouring lanes.  A local's result is ASSIGNED to its accumulator (a local query is evaluated once
 * per call; the reply merge accumulates on top of it); a received query's result is written into its reply, with
 * the query's own origin, so the rank that asked can check it.  Anomalies are raised in `anomaly`, in device
 * memory, and read by the caller once per round. */
template <typename Spec>
struct NlrModeDBatch {
    using ActiveData = typename Spec::ActiveData;
    using AccumData  = typename Spec::AccumData;
    using Envelope   = NlrQueryEnvelope<ActiveData>;
    using Reply      = NlrReplyEnvelope<AccumData>;

    static void evaluate(const typename Spec::DeviceContext& ctx, const typename Spec::CallScalars& cs,
                         const GxDeviceTreeView& tree, unsigned int neighbor_type_mask,
                         const ActiveData *locals, int n_local, AccumData *local_accums,
                         const Envelope *received, int K, Reply *replies, int *anomaly)
    {
        using ScatterData = typename Spec::ScatterData;
        static_assert(nlr_spec_modeb_eval_omp<Spec>() != ModeBEvalOMP::SerialOnly,
                      "Mode D evaluates a seeker's pairs on the rank that owns the neighbours, so a "
                      "pair kernel's neighbour-side writes land on the owner's own particles and need "
                      "no writeback -- but they land from many device lanes at once, so a kernel that "
                      "must run serially (a read-then-write of live neighbour state) cannot be served.");
        static_assert(Spec::search_mode == MODE_B_SEARCH_ONEWAY,
                      "Mode D prunes on the query's reach alone. A symmetric search also needs the "
                      "per-type supply bands, which are not mirrored to the device.");
        static_assert(!nlr_spec_has_bind_active_to_eval_context_v<Spec>,
                      "This Spec rebinds each active to the evaluating context before its pair kernel "
                      "runs, which the host backend does inside evaluate_pairs_post_drift. The device "
                      "evaluation performs no such rebind, so a received active would still carry the "
                      "sending rank's pointers. Port the rebind into the device path before serving this Spec.");

        const int M = n_local + K;
        if(M <= 0) {return;}
        /* All was brought current by the exchange this batch belongs to, once for the whole exchange. */

        const int use_mass_verdict = nlr_mode_d_keeps_mass_verdict<Spec>(true);
        const GxLeafMirrorView record_mirror = nlr_mode_d_leaf_mirror(true, tree);
        nlr_record_and_drift(
            ctx.P, neighbor_type_mask, use_mass_verdict, record_mirror.type, tree, M, n_local, anomaly, "nlr_mode_d_record",
            KOKKOS_LAMBDA(int kk, double &qx, double &qy, double &qz, double &reach, const int *&start_nodes, int &n_start) {
                const ActiveData &a = (kk < n_local) ? locals[kk] : received[kk - n_local].active;
                qx = (double)a.pos[0]; qy = (double)a.pos[1]; qz = (double)a.pos[2];
                reach = (double)a.h_search;
                if(kk < n_local) {start_nodes = nullptr; n_start = -1;}   /* ours: from the root */
                else {
                    const Envelope &env = received[kk - n_local];
                    start_nodes = env.NodeList; n_start = nlr_mode_d_envelope_nodes(env);
                }
            }, true);

        const struct GxMotionTargetSet motion_targets = gx_motion_target_view();
        const struct GxTouchedSet mass_verdicts = use_mass_verdict ? gx_touched_set_view() : GxTouchedSet{};
        const GxLeafMirrorView leaf_mirror = nlr_mode_d_leaf_mirror(true, tree);
        /* A loop whose evaluation may run in packets takes them in their band; the flat walk then evaluates only the
           members of packets that gave up, or the whole batch when packets did not run. */
        bool packets_ran = false;
        const int *replay = nullptr;
        if constexpr (nlr_mode_d_evaluates_in_packets_v<Spec>) {
            if(nlr_mode_d_packets_for(M, 2, true)) {
                packets_ran = nlr_mode_d_packet_evaluate<Spec>(ctx, cs, locals, received, replies, nullptr, n_local, M, tree,
                                                               neighbor_type_mask, anomaly, motion_targets, leaf_mirror,
                                                               mass_verdicts, use_mass_verdict, local_accums, &replay);
            }
        }
        if(packets_ran && !replay) {return;}   /* every packet completed */
        if(leaf_mirror.pos && leaf_mirror.type) {
            flat<true>(ctx, cs, tree, neighbor_type_mask, locals, n_local, local_accums, received, K, replies, anomaly,
                       replay, motion_targets, leaf_mirror, mass_verdicts, use_mass_verdict);
        } else {
            flat<false>(ctx, cs, tree, neighbor_type_mask, locals, n_local, local_accums, received, K, replies, anomaly,
                        replay, motion_targets, leaf_mirror, mass_verdicts, use_mass_verdict);
        }
    }

    /* The flat device walk: one work item per query (only the replayed members when packets ran). */
    template <bool UsesMirror>
    static void flat(const typename Spec::DeviceContext& ctx, const typename Spec::CallScalars& cs,
                     const GxDeviceTreeView& tree, unsigned int neighbor_type_mask,
                     const ActiveData *locals, int n_local, AccumData *local_accums,
                     const Envelope *received, int K, Reply *replies, int *anomaly, const int *replay,
                     const struct GxMotionTargetSet &motion_targets, const GxLeafMirrorView &leaf_mirror,
                     const struct GxTouchedSet &mass_verdicts, int use_mass_verdict)
    {
        using ScatterData = typename Spec::ScatterData;
        const int M = n_local + K;
            gizmo_gpu_kernel_launch(Spec::loop_name, M, KOKKOS_LAMBDA(int kk) {
                if(replay && !replay[kk]) {return;}   /* its packet completed: that result stands */
                const ActiveData &a = (kk < n_local) ? locals[kk] : received[kk - n_local].active;
                /* The result accumulates in place, in this query's own output (one writer per slot): nothing the size of an
                 * accumulator is held in registers across the walk. */
                AccumData *acc = (kk < n_local) ? &local_accums[kk] : &replies[kk - n_local].accum;
                Spec::zero_accum(*acc);
                ScatterData sc{};
                NlrModeDReduceLeaf<Spec, UsesMirror> leaf{&ctx, &a, acc, &sc, neighbor_type_mask, &cs};
                leaf.motion_targets = motion_targets;
                /* Named here, unconditionally, so that the capture happens outside the `if constexpr`: a device lambda
                 * may not first-capture a variable inside one. */
                const GxLeafMirrorView &mirror = leaf_mirror;
                if constexpr (UsesMirror) {leaf.leaf_mirror = mirror;} else {(void) mirror;}
                leaf.mass_verdicts = mass_verdicts; leaf.use_mass_verdict = use_mass_verdict;
                nlr_mode_d_walk_query(kk, n_local, a, received, tree, leaf, anomaly, neighbor_type_mask);
                if(kk < n_local) {return;}
                const Envelope &env = received[kk - n_local];
                Reply &r = replies[kk - n_local];
                r.origin_slot = env.origin_slot; r.origin_rank = env.origin_rank;
            });
    }
};

static void mode_d_copy_async_to_host(const Kokkos::DefaultExecutionSpace &space, void *host_dst, const void *device_src, size_t bytes)
{
    if(bytes) {Kokkos::deep_copy(space, ModeDHostBytes((char *) host_dst, bytes), ModeDDeviceBytes((char *) device_src, bytes));}
}
static void mode_d_copy_async_to_device(const Kokkos::DefaultExecutionSpace &space, void *device_dst, const void *host_src, size_t bytes)
{
    if(bytes) {Kokkos::deep_copy(space, ModeDDeviceBytes((char *) device_dst, bytes), ModeDHostBytes((char *) host_src, bytes));}
}

/* One export pass over a batch of frozen queries: as packets in their band, the flat device walk otherwise and for
 * the members of packets that gave up (which then start again from nothing: a packet that gave up published
 * nothing).  Device-only, whatever the batch size. */
template <int Pass, class ActiveData>
static void mode_d_export_walk(const ModeDExportStep<Pass> &step, const ActiveData *queries, int n,
                               const GxDeviceTreeView &tree, int *anomaly)
{
    if(n <= 0) {return;}
    bool ran = false;
    const int *replay = nullptr;
    if(nlr_mode_d_packets_for(n, 4, true)) {
        ModeDExportPacketClient<Pass, ActiveData> cl;
        cl.step = step; cl.queries = queries; cl.n_local = n; cl.n_total = n; cl.tree = tree; cl.anomaly = anomaly;
        ran = nlr_mode_d_packet_run(cl, "nlr_mode_d_export_packets", &replay);
    }
    if(ran && !replay) {return;}
    gizmo_gpu_kernel_launch("nlr_mode_d_export", n, KOKKOS_LAMBDA(int kk) {
        if(replay && !replay[kk]) {return;}
        const ActiveData &a = queries[kk];
        ModeDExportFlatPolicy<Pass> policy{&step, {}};
        step.begin(kk, (double) a.h_search, policy.s);
        gx_device_tree_walk_from_root((double) a.pos[0], (double) a.pos[1], (double) a.pos[2], (double) a.h_search,
                                      tree, policy, anomaly, 0u);
        step.commit(kk, policy.s);
    });
}

/* An exclusive scan of n counts into out[0..n], out[n] the total. */
template <class Count, class Total>
static void mode_d_exclusive_scan(const Count *counts, Total *out, int n)
{
    Kokkos::parallel_scan("mode_d_scan", Kokkos::RangePolicy<>(0, n + 1), KOKKOS_LAMBDA(int i, Total &sum, const bool last) {
        const Total v = (i < n) ? (Total) counts[i] : (Total) 0;
        if(last) {out[i] = sum;}
        sum += v;
    });
}

/* ============================================================================
 * mode_d_exchange<Spec> -- one Mode D call's peer exchange (one iteration's, for an iterative loop), resident
 * on the device.  The host moves messages and O(NTask) counts; every query, record, envelope and reply is in
 * device memory and is touched only by device kernels and staging copies.
 *
 *   1. the queries, frozen: one kernel builds every active's query where its particle lives;
 *   2. the export count: what each query sends (mode_d_export.h), and from it each query's slice and envelope
 *      bound; the round arena grows to this iteration's bound if the device allows;
 *   3. rounds, each the largest run of whole actives whose bounds fit the arena: fill the round's records, sort
 *      them by peer, cut them into envelopes in send order, and fix the merge order; exchange counts (an anomaly
 *      on any rank is a negative count, and no payload moves); stream the queries out, answer the queries that
 *      come in, in batches, on the device, and stream the replies back; merge each active's replies onto its own
 *      result in the host's order; agree at the round's end whether everyone is done, or a stop was raised;
 *   4. the results to the caller.
 *
 * The received queries are evaluated in as few batches as the device needs: a batch is evaluated when it is full
 * or when nothing more is coming this round, never merely with what has arrived so far.  This rank's own queries go
 * with the first batch, and alone only when nothing is coming: a separate evaluation would add a whole record, drift and
 * evaluate sequence, and would overlap little, since an evaluation holds the progress loop. */
template <typename Spec>
static bool mode_d_exchange(ModeDCallLease &L, const neighbor_loop_args& args, const double *radii,
                            const typename Spec::CallScalars& cs, const typename Spec::DeviceContext& ctx,
                            unsigned int type_mask, typename Spec::AccumData *accums_out,
                            const ModeDDeviceQueries *device_queries)
{
    using ActiveData = typename Spec::ActiveData;
    using AccumData  = typename Spec::AccumData;
    using Envelope   = NlrQueryEnvelope<ActiveData>;
    using Reply      = NlrReplyEnvelope<AccumData>;
    using Stream     = ModeDSegmentStream;
    static_assert(alignof(Envelope) <= alignof(ModeDSegmentHeader) && alignof(Reply) <= alignof(ModeDSegmentHeader),
                  "records follow the segment header in a staging message");

    const int N = device_queries ? device_queries->n_live : args.num_active, nt = NTask, me = ThisTask;
    const size_t q = sizeof(Envelope), r = sizeof(Reply);
    const int S = (int) L.plan.segment_records;
    const GxDeviceTreeView tree = L.tree;
    ActiveData *snapshot = (ActiveData *) L.query_snapshot;
    AccumData  *accums   = (AccumData *) L.active_accums;
    int        *flags    = L.flags;
    GIZMO_GPU_ENSURE_ALL_FRESH();
    const Kokkos::DefaultExecutionSpace &copy_space = gx_mode_d_copy_space();
    Kokkos::parallel_for("mode_d_flags_clear", Kokkos::RangePolicy<>(0, MODE_D_FLAGS), KOKKOS_LAMBDA(int k) {flags[k] = 0;});

    /* 1. the queries */
    if(N > 0 && device_queries) {
        const int *live = device_queries->live, *index = device_queries->slot_index;
        const double *radius = device_queries->slot_radius;
        const int base = device_queries->call_slot_base;
        gizmo_gpu_kernel_launch("nlr_mode_d_build_queries", N, KOKKOS_LAMBDA(int k) {
            const int slot = live[k];
            snapshot[k] = Spec::load_active(ctx, base + slot, index[slot], radius[slot], cs);
        });
    } else if(N > 0) {
        mode_d_copy_to_device(L.active_index, args.active_list, (size_t) N * sizeof(int));
        mode_d_copy_to_device(L.active_radius, radii, (size_t) N * sizeof(double));
        int *slot = L.active_slot; const int *idx = L.active_index; const double *rad = L.active_radius;
        if(args.active_call_slot) {mode_d_copy_to_device(L.active_slot, args.active_call_slot, (size_t) N * sizeof(int));}
        else {Kokkos::parallel_for("mode_d_slots", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int k) {slot[k] = k;});}
        gizmo_gpu_kernel_launch("nlr_mode_d_build_queries", N, KOKKOS_LAMBDA(int k) {
            snapshot[k] = Spec::load_active(ctx, slot[k], idx[k], rad[k], cs);
        });
    }

    /* 2. the export count, and the arena for it */
    const ModeDExportTopology topo{L.map, me};
    long long totals[2] = {0, 0};   /* envelope bound, records */
    if(N > 0) {
        ModeDExportStep<MODE_D_EXPORT_COUNT> count;
        count.topo = topo; count.out.nodes = L.active_nodes; count.out.envelope_bound = L.active_envelope_scan;
        mode_d_export_walk(count, snapshot, N, tree, flags + MODE_D_FLAG_WALK);
        mode_d_exclusive_scan(L.active_nodes, L.active_node_scan, N);
        long long *bound = L.active_envelope_scan;
        Kokkos::parallel_scan("mode_d_bound_scan", Kokkos::RangePolicy<>(0, N + 1), KOKKOS_LAMBDA(int i, long long &sum, const bool last) {
            const long long v = (i < N) ? bound[i] : 0;
            if(last) {bound[i] = sum;}
            sum += v;
        });
        {
            long long *info = L.round_info; const long long *nscan = L.active_node_scan;
            Kokkos::parallel_for("mode_d_totals", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
                info[MODE_D_ROUND_TOTAL_BOUND] = bound[N]; info[MODE_D_ROUND_TOTAL_RECORDS] = nscan[N];
            });
            mode_d_copy_to_host(totals, L.round_info + MODE_D_ROUND_TOTAL_BOUND, sizeof(totals));
        }
        const long long want = (totals[0] < L.plan.arena_capacity_limit) ? totals[0] : L.plan.arena_capacity_limit;
        (void) mode_d_arena_grow(&L, want, q, r);   /* refused: the arena keeps its size, and the call runs more rounds */
    }

    /* 3. rounds */
    ModeDRoundArena &A = L.arena;
    std::vector<int> &counts = L.peer_counts, &peer_off = L.peer_offsets;
    int a_begin = 0, round = 0, locals_done = (N == 0), stop = 0, stop_code = 0, walk_anomaly = 0;
    while(1) {
        round++;
        int a_end = a_begin, n_round = 0, n_rec = 0;
        long long round_info[3] = {0, 0, 0};
        const ModeDExportRecord *sorted = nullptr;
        ModeDQueryRun *query_runs = nullptr;
        if(a_begin < N) {
            const long long capacity = A.capacity;
            const long long *bound = L.active_envelope_scan, *node_scan = L.active_node_scan;
            long long *info = L.round_info;
            const int first = a_begin;
            /* the largest run of whole actives whose bounds fit: one active always fits (admission) */
            Kokkos::parallel_for("mode_d_round_end", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
                int lo = first + 1, hi = N;
                while(lo < hi) {const int mid = lo + (hi - lo + 1) / 2; if(bound[mid] - bound[first] <= capacity) {lo = mid;} else {hi = mid - 1;}}
                info[0] = lo; info[1] = node_scan[first]; info[2] = node_scan[lo];
            });
            mode_d_copy_to_host(round_info, info, sizeof(round_info));
            a_end = (int) round_info[0]; n_round = a_end - a_begin;
            n_rec = (int) (round_info[2] - round_info[1]);

            ModeDExportStep<MODE_D_EXPORT_FILL> fill;
            fill.topo = topo; fill.first_active = a_begin;
            fill.out.node_offset = L.active_node_scan + a_begin; fill.out.node_count = L.active_nodes + a_begin;
            fill.out.node_offset_base = round_info[1]; fill.out.records = A.records; fill.out.anomaly = flags + MODE_D_FLAG_FILL;
            mode_d_export_walk(fill, snapshot + a_begin, n_round, tree, flags + MODE_D_FLAG_WALK);

            int *round_ints = (int *) (info + MODE_D_ROUND_SPLIT);   /* the split point, then (next slot) the envelope count */
            int *n_env_dev  = (int *) (info + MODE_D_ROUND_ENVELOPES);
            sorted = mode_d_stable_radix(A.records, A.records_spare, A.scan, n_rec,
                                         KOKKOS_LAMBDA(const ModeDExportRecord &x) {return (unsigned int) x.peer;}, mode_d_key_bits(nt), round_ints);
            mode_d_envelope_boundaries(sorted, n_rec, A.scan, A.first_record, A.envelope_of, n_env_dev);
            query_runs = (ModeDQueryRun *) ((sorted == A.records) ? A.records_spare : A.records);   /* the sort's other half is free */
            mode_d_query_runs(sorted, n_rec, A.scan, A.envelope_of, a_begin, n_round, A.run_offsets, query_runs);
            int *per_peer = L.per_peer;
            Kokkos::parallel_for("mode_d_per_peer_clear", Kokkos::RangePolicy<>(0, nt + 1), KOKKOS_LAMBDA(int p) {per_peer[p] = 0;});
            mode_d_pack_envelopes(sorted, n_rec, A.first_record, n_env_dev, snapshot, me, (Envelope *) A.envelopes, per_peer, nt);
        }
        /* the per-peer counts and, after them, the flags, in one read (no counts when this rank sent nothing this round) */
        int host_flags[MODE_D_FLAGS];
        if(a_begin < N) {
            counts.resize((size_t) nt + 1 + MODE_D_FLAGS);   /* within the capacity reserved at admission */
            mode_d_copy_to_host(counts.data(), L.per_peer, ((size_t) nt + 1 + MODE_D_FLAGS) * sizeof(int));
            for(int f = 0; f < MODE_D_FLAGS; f++) {host_flags[f] = counts[(size_t) nt + 1 + f];}
            counts.resize((size_t) nt);
        } else {
            counts.assign((size_t) nt, 0);
            mode_d_copy_to_host(host_flags, flags, sizeof(host_flags));
        }
        if(host_flags[MODE_D_FLAG_WALK] || host_flags[MODE_D_FLAG_FILL]) {
            /* an export this rank cannot vouch for: no payload moves anywhere this round */
            for(int p = 0; p < nt; p++) {counts[(size_t) p] = (p == me) ? 0 : -1;}
            stop_code = host_flags[MODE_D_FLAG_FILL] ? 90001042 : 90001044;
            if(!host_flags[MODE_D_FLAG_FILL]) {walk_anomaly = host_flags[MODE_D_FLAG_WALK];}
        }
        peer_off.assign((size_t) nt + 1, 0);
        for(int p = 0; p < nt; p++) {peer_off[(size_t) p + 1] = peer_off[(size_t) p] + ((counts[(size_t) p] > 0) ? counts[(size_t) p] : 0);}

        Stream &X = L.stream;
        const int anomaly_round = X.begin_round(round, counts.data(), S, q, r);
        if(!anomaly_round) {
            /* What arrives this round is known from the count exchange.  A batch holds what arrives, up to one query
             * per lane of the device (a larger batch adds no parallelism) and never fewer than one staging segment;
             * the second batch is grown only when more arrives than the first holds.  Refused growth leaves a batch
             * as it was: the round then runs in more batches. */
            long long incoming = 0;
            int in_segments = 0, in_taken = 0;
            for(int p = 0; p < nt; p++) {if(p != me) {incoming += X.recv_count(p); in_segments += X.segments_from(p);}}
            {
                const long long lanes = gizmo_gpu_lane_count(), feed = (lanes > S) ? lanes : (long long) S;
                (void) mode_d_batch_grow(&L, 0, (incoming < feed) ? incoming : feed, q, r);
                const long long rest = incoming - L.batch_capacity[0];
                if(rest > 0) {(void) mode_d_batch_grow(&L, 1, (rest < feed) ? rest : feed, q, r);}
                /* Each batch's segment list, so that taking a segment never allocates.  A refusal is not a defect:
                 * a batch whose list is full counts as full below. */
                for(int m = 0; m < 2; m++) {
                    try {L.batch_segments[m].reserve((size_t) in_segments);} catch(const std::bad_alloc &) {}
                }
            }
            /* the queries out: each destination's segments in order */
            int out_peer = 0, out_index = 0;
            auto next_out = [&](void) {
                while(out_peer < nt && (out_peer == me || out_index >= X.segments_to(out_peer))) {out_peer++; out_index = 0;}
                return out_peer < nt;
            };
            /* the batches of received queries: one being filled, and those whose replies are still going out, in order */
            struct Batch {int records = 0, segments_sent = 0, state = 0;};   /* state 0 free, 1 filling, 2 answered */
            Batch batch[2];
            int filling = 0, answered_order[2] = {-1, -1}, n_answered = 0;
            batch[0].state = 1; L.batch_segments[0].clear(); L.batch_segments[1].clear();
            Stream::Segment held{}; int have_held = 0;
            while(1) {
                X.progress();
                /* (a) queries out, as staging slots free: the copies into every free slot are issued, the copy queue is fenced
                 * once, then the sends are posted -- each destination's segments in order */
                {
                    int free_k[Stream::SLOTS], dest[Stream::SLOTS], seg_index[Stream::SLOTS], recs[Stream::SLOTS], n_out = 0;
                    const int n_free = X.free_slots(Stream::QUERIES_OUT, free_k);
                    while(n_out < n_free && next_out()) {
                        recs[n_out] = X.records_in_segment(X.sent_count(out_peer), out_index);
                        const size_t first_env = (size_t) peer_off[(size_t) out_peer] + (size_t) out_index * (size_t) S;
                        mode_d_copy_async_to_host(copy_space, X.payload(Stream::QUERIES_OUT, free_k[n_out]), (char *) A.envelopes + first_env * q, (size_t) recs[n_out] * q);
                        dest[n_out] = out_peer; seg_index[n_out] = out_index;
                        out_index++; n_out++;
                    }
                    if(n_out) {copy_space.fence("mode_d_query_segments_out");}
                    for(int k = 0; k < n_out; k++) {X.send_query_segment(free_k[k], dest[k], seg_index[k], recs[k]);}
                }
                /* (b) received queries into the batch being filled, as many as fit and are ready */
                if(filling >= 0) {
                    std::vector<ModeDCallLease::BatchSegment> &segs = L.batch_segments[filling];
                    int copied = 0, copied_slot[Stream::SLOTS];
                    while(1) {
                        Stream::Segment seg;
                        if(have_held) {seg = held;} else if(!X.next_query_segment(seg)) {break;}
                        /* full: no room for its records, or no room in its (reserved) segment list */
                        if((seg.status == 0 && batch[filling].records + seg.records > L.batch_capacity[filling]) ||
                           segs.size() >= segs.capacity()) {held = seg; have_held = 1; break;}
                        have_held = 0; in_taken++;
                        if(seg.status == 0) {
                            mode_d_copy_async_to_device(copy_space, (char *) L.group_queries[filling] + (size_t) batch[filling].records * q,
                                                        X.payload(Stream::QUERIES_IN, seg.slot), (size_t) seg.records * q);
                            segs.push_back({seg.source, seg.index, seg.records, batch[filling].records, 0});
                            batch[filling].records += seg.records;
                            copied_slot[copied++] = seg.slot;
                        } else {
                            /* answered without being evaluated, so it holds no room in the batch */
                            segs.push_back({seg.source, seg.index, seg.records, -1, 1});
                            X.release(Stream::QUERIES_IN, seg.slot);
                        }
                        if(copied == Stream::SLOTS) {break;}
                    }
                    if(copied) {copy_space.fence("mode_d_query_segments_in"); for(int c = 0; c < copied; c++) {X.release(Stream::QUERIES_IN, copied_slot[c]);}}
                }
                /* (c) a batch of received queries when it is full or nothing more is coming this round, with this rank's
                 * own queries in the first (alone, when nothing is coming) */
                const bool batch_full = (filling >= 0) &&
                    (have_held || batch[filling].records >= L.batch_capacity[filling] ||
                     L.batch_segments[filling].size() >= L.batch_segments[filling].capacity());
                if(filling >= 0 && (batch_full || in_taken == in_segments) && (!locals_done || !L.batch_segments[filling].empty())) {
                    const int K = batch[filling].records;
                    NlrModeDBatch<Spec>::evaluate(ctx, cs, tree, type_mask, locals_done ? nullptr : snapshot, locals_done ? 0 : N, accums,
                                                  (const Envelope *) L.group_queries[filling], K, (Reply *) L.group_replies[filling],
                                                  flags + MODE_D_FLAG_WALK);
                    locals_done = 1;
                    if(!L.batch_segments[filling].empty()) {
                        batch[filling].state = 2; answered_order[n_answered++] = filling;
                        const int other = 1 - filling;
                        filling = (batch[other].state == 0) ? other : -1;
                        if(filling >= 0) {batch[filling] = Batch{}; batch[filling].state = 1; L.batch_segments[filling].clear();}
                    }
                }
                /* (d) replies out, batch by batch in the order they were answered, each batch's segments in order */
                while(n_answered > 0) {
                    const int b = answered_order[0];
                    std::vector<ModeDCallLease::BatchSegment> &segs = L.batch_segments[b];
                    {
                        /* the copies into every free slot, one fence, then the sends, in the batch's segment order */
                        int free_k[Stream::SLOTS], n_out = 0, copied = 0;
                        const int n_free = X.free_slots(Stream::REPLIES_OUT, free_k);
                        const int first = batch[b].segments_sent;
                        while(n_out < n_free && first + n_out < (int) segs.size()) {
                            const ModeDCallLease::BatchSegment &g = segs[(size_t) (first + n_out)];
                            char *dst = X.payload(Stream::REPLIES_OUT, free_k[n_out]);
                            if(g.status == 0) {
                                mode_d_copy_async_to_host(copy_space, dst, (char *) L.group_replies[b] + (size_t) g.offset * r, (size_t) g.records * r);
                                copied = 1;
                            } else {
                                /* the rank that asked skips every reply of an erroneous segment; these only have to be well formed */
                                Reply *rep = (Reply *) dst;
                                for(int k = 0; k < g.records; k++) {rep[k].origin_slot = 0; rep[k].origin_rank = g.source; Spec::zero_accum(rep[k].accum);}
                            }
                            n_out++;
                        }
                        if(copied) {copy_space.fence("mode_d_reply_segments_out");}
                        for(int k = 0; k < n_out; k++) {
                            const ModeDCallLease::BatchSegment &g = segs[(size_t) (first + k)];
                            X.send_reply_segment(free_k[k], g.source, g.index, g.records, g.status);
                        }
                        batch[b].segments_sent += n_out;
                    }
                    if(batch[b].segments_sent < (int) segs.size()) {break;}
                    batch[b] = Batch{}; segs.clear();
                    answered_order[0] = answered_order[1]; answered_order[1] = -1; n_answered--;
                    if(filling < 0) {filling = b; batch[b].state = 1;}
                }
                /* (e) replies in, each to its envelope's place */
                {
                    int copied = 0, copied_slot[Stream::SLOTS];
                    Stream::Segment seg;
                    while(copied < Stream::SLOTS && X.next_reply_segment(seg)) {
                        if(seg.status != 0) {X.release(Stream::REPLIES_IN, seg.slot); continue;}   /* skipped: the stop is already raised */
                        const size_t first_env = (size_t) peer_off[(size_t) seg.source] + (size_t) seg.index * (size_t) S;
                        mode_d_copy_async_to_device(copy_space, (char *) A.replies + first_env * r, X.payload(Stream::REPLIES_IN, seg.slot), (size_t) seg.records * r);
                        copied_slot[copied++] = seg.slot;
                    }
                    if(copied) {copy_space.fence("mode_d_reply_segments_in"); for(int c = 0; c < copied; c++) {X.release(Stream::REPLIES_IN, copied_slot[c]);}}
                }
                if(locals_done && !have_held && n_answered == 0 && (filling < 0 || L.batch_segments[filling].empty()) && X.round_complete()) {break;}
            }
            if(X.errors_seen() && !stop_code) {stop_code = 90001043;}
            if(!X.errors_seen() && n_rec > 0) {
                mode_d_merge_replies<Spec>(query_runs, A.run_offsets, a_begin, n_round, (const Reply *) A.replies, me, accums,
                                           flags + MODE_D_FLAG_MERGE);
            }
            mode_d_copy_to_host(host_flags, flags, sizeof(host_flags));
            if(host_flags[MODE_D_FLAG_MERGE] && !stop_code) {stop_code = 90001043;}
            if(host_flags[MODE_D_FLAG_WALK] && !stop_code) {stop_code = 90001044; walk_anomaly = host_flags[MODE_D_FLAG_WALK];}
        }
        a_begin = a_end;
        /* the round's end: everyone done, or a stop raised anywhere */
        int local[2] = {(a_begin >= N && locals_done) ? 1 : 0, (stop_code || anomaly_round) ? 0 : 1}, all[2] = {0, 0};
        MPI_Allreduce(local, all, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(!all[1]) {stop = 1; L.stopped = 1; break;}
        if(all[0]) {break;}
    }

    /* 4. the results */
    if(N > 0 && accums_out) {mode_d_copy_to_host(accums_out, accums, (size_t) N * sizeof(AccumData));}
    if(stop && stop_code) {
        printf("[%s] task %d: the device-resident exchange stopped (%s).\n", Spec::loop_name, me,
               stop_code == 90001042 ? "an export fill did not match its count"
               : stop_code == 90001043 ? "a reply or segment did not match what was asked" : nlr_walk_anomaly_text(walk_anomaly));
        fflush(stdout);
        endrun(stop_code);
    }
    return stop != 0;
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
    neighbor_loop_args sub = drv.args;
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
            gpu_ngb_list_free(&drv.mode_a_cached_gnl[sg]);
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
        /* This subgroup's list: the caller's (NlrSubgroup::external_csr; the
         * entry admits it only for a single-pass loop), shown to answer this
         * loop's search and then staged, or built here.  A failure on either
         * side leaves the CSR invalid rather than returning: the ghost-writeback
         * pair below fires on every rank, so a return here would leave the peers
         * waiting on a reverse-comm that never comes -- a hang, which is the one
         * outcome worse than the stop itself. */
        bool listed = false;   /* a list with its row index is in place */
        const nlr_external_csr *caller_list = sgr.external_csr;
        if (caller_list != nullptr) {
            /* The list's rows carry the loop's search radius itself: the buffer
             * factor only pads a list the runner builds for an iteration to grow
             * into, and a caller-built list serves one pass. */
            std::vector<double> radii_host_build(n_compacted);
            for (int k = 0; k < n_compacted; k++) {
                radii_host_build[k] = drv.radii_uvm[sg][drv.active_set_uvm[sg][k]];
            }
            int         csr_code = 0;
            const char *csr_err  = nlr_external_csr_mismatch<Spec>(
                caller_list, active_particle_indices_host_build.data(), n_compacted,
                radii_host_build.data(), sgr.j_type_bitmask, &csr_code);
            if (csr_err != nullptr) {
                fprintf(stderr, "[neighbor_loop_runner rank=%d caller=%s subgroup=%d] external-CSR contract violation: %s\n",
                        ThisTask, Spec::loop_name, sg, csr_err);
                fflush(stderr);
                endrun(csr_code);
            } else if (nlr_stage_external_csr_into_gnl(caller_list, &drv.mode_a_cached_gnl[sg], Spec::loop_name)) {
                listed = true;
            } else {
                gpu_ngb_list_free(&drv.mode_a_cached_gnl[sg]);
                drv.mode_a_cached_gnl[sg] = gpu_neighbor_list_t{};
            }
        } else {
        const size_t radii_oversized_bytes = (size_t) n_compacted * sizeof(double);
        double *radii_oversized_uvm = (double *) nlr_shared_alloc_bytes(radii_oversized_bytes, "modea_radii");
        if (radii_oversized_uvm != NULL) {
        for (int k = 0; k < n_compacted; k++) radii_oversized_uvm[k] = radii_buffered_host_build[k];

        gpu_ngb_list_build(drv.ctx.P, drv.ctx.num_total,
                           active_particle_indices_host_build.data(), n_compacted,
                           Spec::search_mode,
                           (int)sgr.j_type_bitmask,
                           &drv.mode_a_cached_gnl[sg], sidx,
                           1.0, radii_oversized_uvm, nullptr, Spec::loop_name,
                           nlr_spec_symmetric_j_radius_scale<Spec>(),
                           Spec::radius_policy);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(radii_oversized_uvm);
        /* A build that ran out of memory hands back the empty list and has already
         * asked for the stop. Its row index is absent, which is exactly what the
         * kernel below reads first, so the CSR must not be marked valid. */
        listed = (drv.mode_a_cached_gnl[sg].d_active != nullptr);
        } else {
            nlr_stop_no_staging_memory(Spec::loop_name, "the oversized per-active search radii",
                                       radii_oversized_bytes);
        }
        }   /* list built here */
        if (listed) {

        /* Allocate csr_offset_lookup + csr_buffered_h sized to subgroup max.
         * Lookup keyed on subgroup-slot (invariant):
         *   csr_offset_lookup[slot] = build-time row index (0..n_compacted-1),
         *                              or -1 if slot wasn't in build-time set
         *                              (shouldn't happen — rebuild covers all
         *                              current actives, and converged slots
         *                              never re-enter active_set). */
        const int n_max = sgr.num_active_local;
        const size_t csr_lookup_bytes = (size_t) n_max * sizeof(int);
        const size_t csr_h_bytes      = (size_t) n_max * sizeof(double);
        drv.mode_a_csr_offset_lookup[sg] = (int *) nlr_shared_alloc_bytes(csr_lookup_bytes, "modea_csr_lookup");
        drv.mode_a_csr_buffered_h[sg]    = (double *) nlr_shared_alloc_bytes(csr_h_bytes, "modea_csr_h");
        if (!drv.mode_a_csr_offset_lookup[sg] || !drv.mode_a_csr_buffered_h[sg]) {
            if (drv.mode_a_csr_offset_lookup[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(drv.mode_a_csr_offset_lookup[sg]);
                drv.mode_a_csr_offset_lookup[sg] = nullptr;
            }
            if (drv.mode_a_csr_buffered_h[sg]) {
                Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(drv.mode_a_csr_buffered_h[sg]);
                drv.mode_a_csr_buffered_h[sg] = nullptr;
            }
            nlr_stop_no_staging_memory(Spec::loop_name, "the CSR row lookup",
                                       csr_lookup_bytes + csr_h_bytes);
        } else {
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
        }   /* CSR row lookup obtained */
        }   /* neighbour list in place */
    }

    /* ===== (2) Ghost write detector + writeback begin =====
     * UNCONDITIONAL: fires on every rank for
     * uses_ghost_writeback Specs, INCLUDING empty-actives ranks, to keep
     * MPI reverse-comm collectives in lockstep with non-empty ranks.
     * Plan.path == ModeA_GpuNgl gates the dispatch hooks via
     * nlr_path_uses_imported_ghosts; uses args (post-import). */
    nlr_dispatch_ghost_write_detector_begin<Spec>(sub, plan);
    nlr_dispatch_ghost_writeback_begin<Spec>(sub, plan);

    /* The CSR is what the kernel walks, so an unbuilt one means no work this
     * iteration -- but the writeback pair above and below still runs. */
    if (n_compacted > 0 && drv.mode_a_csr_valid[sg]) {
        /* ===== (3) Stage d_actives + (4) pair-kernel launch =====
         * CSR ROW LOOKUP USAGE:
         *   slot = active_set[k]
         *   row  = csr_offset_lookup[slot]    (build-time row index)
         *   i    = d_active[row]              (particle index at build-time)
         *   h    = radii_uvm[slot]            (current radius — kernel filter)
         * CSR build uses drv.ctx.num_total (= post-import effective num_total).
         * Walk gnl.offsets[row]..offsets[row+1]. */
        /* Non-throwing on both, matching run_mode_a. The staged actives are
         * device-resident, and device memory is exhausted PER RANK rather than
         * node-wide, so a throw here would escape a single rank while its peers
         * sat in the reverse-comm collectives below -- a hang instead of a
         * reported stop. Failure is handled like a subgroup with nothing left to
         * do: the work below is skipped, the stop is requested, and control falls
         * through to the UNCONDITIONAL writeback_end, so every rank still enters
         * the collectives the same number of times. */
        ActiveData *d_actives = (ActiveData *) nlr_active_stage_alloc_bytes(
            (size_t)n_compacted * sizeof(ActiveData), "modea_active_data");
        AccumData *d_accums = (AccumData *) nlr_shared_alloc_bytes(
            (size_t)n_compacted * sizeof(AccumData), "modea_accum_data");
        if (d_actives == NULL || d_accums == NULL) {
            if (d_accums) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums); }
            nlr_active_stage_free(d_actives);
            gizmo_request_controlled_stop(7711,
                "nlr_iter_dispatch_subgroup_mode_a: per-active staging out of memory "
                "(modea_active_data is device-resident, modea_accum_data is host-visible) "
                "-- reduce the active set; note that adding ranks per node does NOT relieve "
                "device memory, since the ranks on a node share it",
                __FILE__, __LINE__, __FUNCTION__);
        } else {
            auto cs_ref = drv.cs;
            const typename Spec::DeviceContext dctx_local = drv.ctx;
            int    *active_set_arr = drv.active_set_uvm[sg];
            int    *csr_lookup     = drv.mode_a_csr_offset_lookup[sg];
            double *radii_arr      = drv.radii_uvm[sg];
            int     *d_active_arr  = drv.mode_a_cached_gnl[sg].d_active;
            int64_t *offsets       = drv.mode_a_cached_gnl[sg].offsets;
            int     *neighbors     = drv.mode_a_cached_gnl[sg].neighbors;

            const int call_slot_base = drv.call_slot_base[sg];
            gizmo_gpu_kernel_launch("nlr_iter_stage_active", n_compacted, KOKKOS_LAMBDA(int k) {
                int slot = active_set_arr[k];
                int row  = csr_lookup[slot];
                int i    = d_active_arr[row];
                double h = radii_arr[slot];
                d_actives[k] = Spec::load_active(dctx_local, call_slot_base + slot, i, h, cs_ref);
            });

            const double t_pair_kernel_start = my_second();
            const struct GxMotionTargetSet motion_targets = gx_motion_target_view();

            /* Same two assignments as the single-pass site; see the commentary
             * there. The only difference is the extra indirection from the
             * compacted active set to the build-time CSR row. */
            auto flat_kernel = KOKKOS_LAMBDA(int k) {
                /* Captured outside the `if constexpr` below; see the single-pass site. */
                const struct GxMotionTargetSet &targets = motion_targets;
                int slot = active_set_arr[k];
                int row  = csr_lookup[slot];
                Spec::zero_accum(d_accums[k]);
                const ActiveData& a = d_actives[k];
                ScatterData s{};
                int64_t start = offsets[row], end = offsets[row + 1];
                for (int64_t nn = start; nn < end; nn++) {
                    int j = neighbors[nn];
                    if constexpr (nlr_spec_writes_neighbour_motion_v<Spec>) {gx_motion_target_mark(targets, j);}
                    IdentitySidecar id{};
                    NeighborData nb = Spec::load_neighbor(dctx_local, j, id, a);
                    Spec::pair_kernel(a, nb, d_accums[k], s, cs_ref);
                }
            };

            if constexpr (nlr_mode_a_pair_policy<Spec>() == ModeAPairAssignment::TeamRowReduce) {
                using TeamKernel = NlrModeATeamPairKernel<Spec, typename Spec::DeviceContext>;
                TeamKernel fn{dctx_local, d_actives, d_accums, offsets, neighbors,
                              active_set_arr, csr_lookup, 0, cs_ref, motion_targets};
                const int team_width = nlr_mode_a_team_width<Spec>(fn);
                if (team_width > 1) {
                    gizmo_gpu_team_kernel_launch(Spec::loop_name, n_compacted, team_width, fn);
                } else {
                    gizmo_gpu_kernel_launch(Spec::loop_name, n_compacted, flat_kernel);
                }
            } else {
                gizmo_gpu_kernel_launch(Spec::loop_name, n_compacted, flat_kernel);
            }
            cpu_charge_child(CPU_PAIR_KERNEL, timediff(t_pair_kernel_start, my_second()));

            /* ===== (6) Scatter compacted accums into driver accum_uvm ===== */
            for (int k = 0; k < n_compacted; k++) {
                int slot = drv.active_set_uvm[sg][k];
                drv.accum_uvm[sg][slot] = d_accums[k];
            }

            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_accums);
            nlr_active_stage_free(d_actives);
        }
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
    std::vector<int> call_slots(n_compacted);
    for (int k = 0; k < n_compacted; k++) {
        int slot = drv.active_set_uvm[sg][k];
        active_particle_indices[k] = sgr.active_indices[slot];
        call_slots[k]              = drv.call_slot_base[sg] + slot;
    }
    neighbor_loop_args sub = drv.args;
    sub.active_list      = active_particle_indices.data();
    sub.active_call_slot = call_slots.data();
    sub.num_active       = n_compacted;

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
    drift_and_filter_candidates<Spec>(cand_modeB, (unsigned int)sgr.j_type_bitmask, [&](int aa, double *pos, double &h_q) {
        const int i = sub.active_list[aa];
        pos[0] = (double)sub.P[i].Pos[0]; pos[1] = (double)sub.P[i].Pos[1]; pos[2] = (double)sub.P[i].Pos[2];
        h_q = radii_compacted[aa];
    });
    evaluate_pairs_post_drift<Spec>(drv.ctx, actives_compacted.data(), n_compacted,
                                      cand_modeB, accums_compacted.data(), drv.cs, EvalOMPPolicy::AllowProduction);

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
 * nlr_iter_dispatch_subgroup_mode_d<Spec>(drv, sg) — one iteration, one subgroup.
 *
 * Two shapes, chosen by whether this rank has peers.
 *
 * With peers, the queries travel: this rank's own go out, other ranks' come in,
 * and both halves are answered by walking the tree of whichever rank owns the
 * particles.  That runs through the same transport Mode B uses -- same wire,
 * same bounded rounds, same reply merge -- with the fused backend supplying the
 * answers instead of a host walk over a collected list.
 *
 * Alone, there is nothing to exchange, so the transport is skipped entirely and
 * the queries never leave: they are built in the kernel from resident particles
 * rather than up front on the host.
 *
 * There is no ghost pool and no neighbour list on either shape, and their
 * absence is the whole of the difference from Mode A: the same Spec, the same
 * pair kernel, the same accumulator, reached by walking rather than by
 * importing. Staging is reduced rather than gone -- the queries are built where
 * the particles live instead of on the host, but their indices and radii, and
 * the accumulators coming back, are still staged.
 *
 * WHERE DECLINING IS DECIDED.  Whether this rank can answer on the device is
 * settled COLLECTIVELY before dispatch, when the path is chosen, because the
 * three paths enter different collectives and a rank deciding alone would hang
 * rather than answer differently.  The one rank-local fallback left in this
 * function is for a device allocation that fails, and it is reachable only on
 * the single-rank shape, which issues nothing collective -- with peers the
 * function has already returned through the transport before that point.
 * ========================================================================== */
template <typename Spec>
static void nlr_iter_dispatch_subgroup_mode_d(NlrIterDriver<Spec>& drv, int sg)
{
    using AccumData = typename Spec::AccumData;

    const NlrSubgroup& sgr = drv.args.subgroups[sg];
    const int n_compacted  = drv.active_set_size[sg];

    /* ⛔ NO EARLY RETURN ON AN EMPTY LOCAL SET WHEN THERE ARE PEERS.  The peer
     * path below is collective: a rank with nothing of its own still has to
     * enter it, because its peers are waiting to send it queries and to receive
     * their replies.  The local-only path further down may return early, and
     * does, because it issues nothing collective at all. */

    /* The tree was described, and the rank made current, once for the whole
     * call before any discovery round -- not here.  Re-preparing per subgroup
     * per iteration would repeat a decision the path selection has already acted
     * on, and there is nothing to re-establish: nothing this path touches can go
     * stale within the call. */
    const GxDeviceTreeView &tree = drv.mode_d_tree;

    /* Compact the still-active slots, as every other per-iter dispatch does. */
    std::vector<int>    active_particle_indices(n_compacted);
    std::vector<int>    call_slots(n_compacted);
    std::vector<double> radii_compacted(n_compacted);
    for(int k = 0; k < n_compacted; k++) {
        const int slot = drv.active_set_uvm[sg][k];
        active_particle_indices[k] = sgr.active_indices[slot];
        call_slots[k]              = drv.call_slot_base[sg] + slot;
        radii_compacted[k]         = drv.radii_uvm[sg][slot];
    }
    neighbor_loop_args sub = drv.args;
    sub.active_list      = active_particle_indices.data();
    sub.active_call_slot = call_slots.data();
    sub.num_active       = n_compacted;

    /* With peers, the queries travel and the owners answer them: same wire, same
     * reply merge, same bounded rounds as the host walker uses -- only the way a
     * query becomes an accumulator differs, on both halves.  Entered on every
     * rank, including one with no actives of its own. */
    if(NTask > 1) {
        std::vector<AccumData> accums_compacted(n_compacted);
        mode_d_exchange<Spec>(*drv.mode_d_lease, sub, radii_compacted.data(), drv.cs, drv.ctx,
                              (unsigned int)sgr.j_type_bitmask,
                              (n_compacted > 0) ? accums_compacted.data() : nullptr);
        for(int k = 0; k < n_compacted; k++) {
            drv.accum_uvm[sg][drv.active_set_uvm[sg][k]] = accums_compacted[k];
        }
        return;
    }

    /* Single rank: nothing to exchange, so the queries never leave and are built
     * in the kernel from resident particles rather than up front on the host. */
    if(n_compacted <= 0) {return;}

    std::vector<AccumData> accums_compacted(n_compacted);
    const bool evaluated = nlr_mode_d_evaluate_single_rank<Spec>(
        drv.ctx, drv.cs, active_particle_indices.data(), call_slots.data(),
        radii_compacted.data(), n_compacted,
        /* The SUBGROUP's mask, as the peer branch and every other per-subgroup
         * dispatcher use.  A multi-subgroup Spec would otherwise see different
         * neighbours on one rank than on many. */
        (unsigned int)sgr.j_type_bitmask,
        tree, accums_compacted.data());
    if(!evaluated) {
        /* Out of device memory is a reason to answer differently, never a reason
         * to answer less, so the host walker takes this iteration.
         *
         * Reachable on the single-rank shape only -- the peer shape returned
         * above -- which is what keeps this legal: swapping backends mid-call is
         * invisible to peers, and with peers it would be a hang rather than a
         * different answer.  If this branch is ever given a path that has peers,
         * the decision has to move to the collective site where readiness is
         * already decided. */
        nlr_iter_dispatch_subgroup_mode_b_local<Spec>(drv, sg);
        return;
    }
    for(int k = 0; k < n_compacted; k++) {
        drv.accum_uvm[sg][drv.active_set_uvm[sg][k]] = accums_compacted[k];
    }
}

template <typename Spec>
static bool nlr_mode_d_evaluate_single_rank(const typename Spec::DeviceContext& ctx,
                                            const typename Spec::CallScalars& cs,
                                            const int *active_idx_host,
                                            const int *active_slot_host,
                                            const double *radii_host,
                                            int n,
                                            unsigned int supply_mask,
                                            const GxDeviceTreeView& tree,
                                            typename Spec::AccumData *accums_out)
{
    using AccumData = typename Spec::AccumData;
    if(n <= 0) {return true;}

    int       *idx_d     = (int *)    nlr_shared_alloc_bytes((size_t)n * sizeof(int),    "moded_idx");
    int       *slot_d    = (int *)    nlr_shared_alloc_bytes((size_t)n * sizeof(int),    "moded_slot");
    double    *rad_d     = (double *) nlr_shared_alloc_bytes((size_t)n * sizeof(double), "moded_radii");
    AccumData *acc_d     = (AccumData *) nlr_shared_alloc_bytes((size_t)n * sizeof(AccumData), "moded_accum");
    int       *anomaly_d = (int *)    nlr_shared_alloc_bytes(sizeof(int), "moded_anomaly");

    if(!idx_d || !slot_d || !rad_d || !acc_d || !anomaly_d) {
        if(idx_d)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(idx_d);}
        if(slot_d)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(slot_d);}
        if(rad_d)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(rad_d);}
        if(acc_d)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(acc_d);}
        if(anomaly_d) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(anomaly_d);}
        return false;
    }

    for(int k = 0; k < n; k++) {
        idx_d[k]  = active_idx_host[k];
        slot_d[k] = active_slot_host[k];
        rad_d[k]  = radii_host[k];
    }
    *anomaly_d = 0;

    nlr_mode_d_local_reduce<Spec>(ctx, cs, idx_d, slot_d, rad_d, n, supply_mask, tree, acc_d, anomaly_d);

    const int anomaly_seen = *anomaly_d;
    for(int k = 0; k < n; k++) {accums_out[k] = acc_d[k];}

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(idx_d);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(slot_d);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(rad_d);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(acc_d);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(anomaly_d);

    /* The traversal reports the one state the host walk treats as fatal: an
     * index belonging to none of the three classes, which means the tree is
     * malformed.  The host stops the run there and so does this, rather than
     * keep an answer short by an unknown amount.  Requested after the buffers
     * are released, because the stop drains at the next phase boundary rather
     * than here. */
    if(anomaly_seen != 0) {
        if(ThisTask == 0) {
            fprintf(stderr,
                "[%s] FATAL: device walk: %s.\n", Spec::loop_name,
                nlr_walk_anomaly_text(anomaly_seen));
            fflush(stderr);
        }
        endrun(90001025);
    }
    return true;
}

/* ============================================================================
 * A multi-rank Mode D call of a loop that iterates on the device (nlr_mode_d_iterates_on_device_v) keeps each
 * active's iteration state in the call's device storage from its first iteration to its last: the radius, the
 * scratch, the latest result, and the list of slots still iterating.  Per iteration the exchange builds its queries
 * from that list and leaves its results on the device; after_iter runs on the device for every listed slot, the
 * slots that iterate again are compacted, in order, into the other list, and the host reads only how many there are
 * -- the count every rank reduces anyway.  The host's copy of the state (the driver's per-active arrays) is brought
 * up to date only when the call hands over to Mode B or finishes, where the host reads it.
 * ========================================================================== */
template <typename Spec>
static void nlr_mode_d_iteration_begin(NlrIterDriver<Spec>& drv)
{
    ModeDCallLease &L = *drv.mode_d_lease;
    const int n = drv.active_set_size[0];
    L.iter_on = 1; L.iter_which = 0; L.iter_n_live = (n > 0) ? n : 0;
    if(n <= 0) {return;}
    mode_d_copy_to_device(L.iter_slot_index, drv.args.subgroups[0].active_indices, (size_t) n * sizeof(int));
    mode_d_copy_to_device(L.iter_slot_radius, drv.radii_uvm[0], (size_t) n * sizeof(double));
    /* the scratch starts zeroed and the list in slot order, as the driver's own copies do: set where they live */
    char *scratch = (char *) L.iter_slot_scratch; int *live = L.iter_live[0];
    const size_t scratch_bytes = (size_t) n * sizeof(typename Spec::IterScratch);
    Kokkos::parallel_for("nlr_mode_d_iteration_begin", Kokkos::RangePolicy<>(0, (scratch_bytes > (size_t) n) ? (long long) scratch_bytes : (long long) n),
                         KOKKOS_LAMBDA(long long b) {
        if((size_t) b < scratch_bytes) {scratch[b] = 0;}
        if(b < n) {live[b] = (int) b;}
    });
}

template <typename Spec>
static void nlr_mode_d_iteration_end(NlrIterDriver<Spec>& drv)
{
    ModeDCallLease &L = *drv.mode_d_lease;
    if(!L.iter_on) {return;}
    L.iter_on = 0;
    const int n_all = drv.args.subgroups[0].num_active_local;
    if(n_all > 0 && drv.radii_uvm[0]) {
        mode_d_copy_to_host(drv.radii_uvm[0], L.iter_slot_radius, (size_t) n_all * sizeof(double));
        mode_d_copy_to_host(drv.scratch_uvm[0], L.iter_slot_scratch, (size_t) n_all * sizeof(typename Spec::IterScratch));
        mode_d_copy_to_host(drv.accum_uvm[0], L.iter_slot_accum, (size_t) n_all * sizeof(typename Spec::AccumData));
        mode_d_copy_to_host(drv.active_set_uvm[0], L.iter_live[L.iter_which], (size_t) L.iter_n_live * sizeof(int));
    }
    drv.active_set_size[0] = L.iter_n_live;
}

/* One iteration's exchange, the queries and results staying on the device. */
template <typename Spec>
static void nlr_iter_dispatch_subgroup_mode_d_on_device(NlrIterDriver<Spec>& drv, int sg)
{
    ModeDCallLease &L = *drv.mode_d_lease;
    const ModeDDeviceQueries dq{L.iter_live[L.iter_which], L.iter_n_live, drv.call_slot_base[sg], L.iter_slot_index, L.iter_slot_radius};
    neighbor_loop_args sub = drv.args;
    sub.active_list = nullptr; sub.active_call_slot = nullptr; sub.num_active = L.iter_n_live;
    mode_d_exchange<Spec>(L, sub, nullptr, drv.cs, drv.ctx, (unsigned int) drv.args.subgroups[sg].j_type_bitmask, nullptr, &dq);
}

/* after_iter for every listed slot, then the slots that iterate again, in order, into the other list.  Returns the
 * iteration's stop code (0: none): the largest any slot raised, for the caller to agree on with every rank. */
template <typename Spec>
static int nlr_mode_d_after_iter_on_device(NlrIterDriver<Spec>& drv)
{
    using AccumData   = typename Spec::AccumData;
    using IterScratch = typename Spec::IterScratch;
    ModeDCallLease &L = *drv.mode_d_lease;
    const int n = L.iter_n_live;
    int n_next = 0, stop_code = 0;
    if(n > 0) {
        const int *cur = L.iter_live[L.iter_which];
        int *next = L.iter_live[1 - L.iter_which], *keep = L.iter_keep, *count = L.iter_count;
        const int *index = L.iter_slot_index;
        double *radius = L.iter_slot_radius;
        IterScratch *scratch = (IterScratch *) L.iter_slot_scratch;
        AccumData *slot_accum = (AccumData *) L.iter_slot_accum;
        const AccumData *accums = (const AccumData *) L.active_accums;
        struct particle_data *Pp = drv.args.P;
        struct gas_cell_data *Cp = drv.args.CellP;
        const typename Spec::CallScalars cs = drv.cs;
        const int iter = drv.iter_index, this_task = ThisTask;
        Kokkos::parallel_for("nlr_mode_d_iter_flags_clear", Kokkos::RangePolicy<>(0, 2), KOKKOS_LAMBDA(int c) {count[c] = 0;});
        gizmo_gpu_kernel_launch("nlr_mode_d_after_iter", n, KOKKOS_LAMBDA(int k) {
            const int slot = cur[k];
            const AccumData acc = accums[k];
            slot_accum[slot] = acc;
            int code = 0;
            const IterResult res = Spec::after_iter_device(Pp, Cp, index[slot], iter, radius[slot], cs, scratch[slot], acc, this_task, code);
            if(res.status == IterStatus::AdjustRadius) {radius[slot] = res.new_h_search;}
            else if(res.status != IterStatus::Converged && res.status != IterStatus::NeedsMore) {code = 81206;}   /* an unknown status is a Spec defect */
            if(code) {Kokkos::atomic_max(&count[1], code);}
            keep[k] = (res.status == IterStatus::NeedsMore || res.status == IterStatus::AdjustRadius) ? 1 : 0;
        });
        Kokkos::parallel_scan("nlr_mode_d_compact", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int k, int &before, const bool last) {
            if(last) {
                if(keep[k]) {next[before] = cur[k];}
                if(k == n - 1) {*count = before + keep[k];}
            }
            before += keep[k];
        });
        int both[2] = {0, 0};
        mode_d_copy_to_host(both, count, sizeof(both));
        n_next = both[0]; stop_code = both[1];
    }
    L.iter_which = 1 - L.iter_which; L.iter_n_live = n_next;
    drv.active_set_size[0] = n_next; drv.local_active_per_sg[0] = n_next; drv.local_active_total = n_next;
    return stop_code;
}

/* ============================================================================
 * run_neighbor_loop_iterative<Spec> — body (step 2b).
 * ========================================================================== */
template <typename Spec>
void run_neighbor_loop_iterative(const neighbor_loop_args_iterative& args_in)
{
    /* Everything a Mode D call holds, released on every way out, exactly as in the
     * non-iterative runner.  Declared first so it outlives the driver, which walks the tree
     * the lease prepared.  Without the release the touched-set recorder stays held after the
     * first iterative call and every later fused call is refused, which silently pins the
     * whole run to the fallback path. */
    ModeDCallLease mode_d_lease;

    /* The one live argument view for this call.  A ghost import can raise the particle
     * capacity and move P[]/CellP[], so the cached base pointers are refreshed in this
     * object as they change; the driver holds a reference to it, so the runner and every
     * hook read the same, current values.  The caller's own copy is left untouched. */
    neighbor_loop_args_iterative args = args_in;

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
    static_assert(Spec::mode_a_csr_buffer_factor >= 1.0,
                  "Spec::mode_a_csr_buffer_factor must be >= 1.0. It oversizes the CSR so an "
                  "iteration that ENLARGES the search radius can reuse the list instead of "
                  "rebuilding it (legacy DENSITY_H_BUFFER_FACTOR = 1.3). A Spec whose radius is "
                  "fixed for the whole call -- no AdjustRadius -- declares 1.0: a larger value "
                  "there is not a buffer, it is a wider import nothing will ever read.");
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
    /* Caller-built lists (NlrSubgroup::external_csr): every subgroup or none,
     * a single-pass loop only, and Mode A by the caller's own override -- the
     * lists index the caller's ghost pool, so no other path can use them. */
    {
        int with_list = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            if (args.subgroups[sg].external_csr != nullptr) with_list++;
        }
        const char *why = nullptr;
        if (with_list == 0)                                  {}
        else if (with_list != args.num_subgroups)            {why = "only some subgroups carry a caller-built list";}
        else if (Spec::max_iters != 1)                       {why = "an iterating loop cannot use a list built once";}
        else if (args.dispatch_override != NlrForceMode::A)  {why = "caller-built lists need dispatch_override = A";}
        if (why != nullptr) {
            fprintf(stderr, "[run_neighbor_loop_iterative<%s> rank=%d] caller-built list contract violation: %s\n",
                    Spec::loop_name, ThisTask, why);
            fflush(stderr);
            endrun(7310);
            return;   /* before any state touch; symmetric caller-contract failure -> graceful return, drains at next poll */
        }
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


    /* ===== Path selection at iter 0 =====
     * Integrates with the canonical override / threshold dispatch
     * (mirrors the selection in run_neighbor_loop). Path is held across the
     * iterations of one call, with one exception: a Mode D call whose
     * remaining actives fall below Mode D's floor finishes on Mode B (end of
     * the iteration loop).
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
        const NlrMode mode = nlr_select_mode<Spec>(sum_act, max_act);
        path = (mode == NlrMode::B) ? DispatchPath::ModeB_HostWalker : DispatchPath::ModeA_GPU_NGL;
        /* Answered where its neighbours live, never on ghost copies (stated at
         * the single-pass entry, nlr_spec_needs_live_neighbours). */
        if constexpr (nlr_spec_needs_live_neighbours_v<Spec>) {
            path = DispatchPath::ModeB_HostWalker;
        }
        /* Mode D takes the calls nlr_select_mode gives it, for a loop it can
         * serve and every rank can actually answer on the device.  It
         * serves every one-way search whose pair kernel may run concurrently:
         * the neighbours are always the evaluating rank's own particles, so a
         * kernel's neighbour-side writes are local atomics, exactly as they are
         * on the host walker's remote protocol, which this is the device form of.
         *
         * The verdict has to be UNANIMOUS, not merely identical-looking.  Two of
         * the terms are compile-time Spec traits and one is a path already chosen
         * from a collective, so those cannot differ -- but whether a rank's tree
         * can be described to the device is a local fact that genuinely can
         * differ between ranks, and the three paths enter different collectives.
         * A rank deciding alone would not answer differently, it would hang.
         *
         * The preparation runs HERE, once, before any discovery round, and what
         * it establishes holds for every iteration of the call: no particles are
         * imported on this path and the writeback runs after the last iteration,
         * so nothing it certifies can go stale in between.
         *
         * With the peer half in place this now runs at any task count, and the
         * reduction above stops being a formality: a rank that cannot describe
         * its tree to the device pulls the WHOLE call off Mode D, because the
         * alternative is one rank walking while its peers wait for queries that
         * never arrive. */
        /* nlr_select_mode never asks for Mode D on a call with no actives
         * anywhere, and that is not a shortcut: preparing mirrors and sweeps the
         * whole tree, and such a call has no work to justify that -- it is
         * O(Nnodes) on the way to the globally-zero-active no-op below. */
        if(mode == NlrMode::D) {
            /* The lease's prepared tree is handed to the driver so every iteration walks the
             * tree the decision was made on.  A refused call holds nothing of Mode D; where it
             * goes is mode_d_admit's verdict. */
            const ModeDAdmission verdict = mode_d_admit_for<Spec>(&mode_d_lease, args.num_active);
            if(verdict == ModeDAdmission::Admitted) {path = DispatchPath::ModeD_DeviceFused;}
            else if(verdict == ModeDAdmission::StorageRefused && nlr_mode_b_switched_on<Spec>()) {path = DispatchPath::ModeB_HostWalker;}
        }
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
        return;
    }


    /* ===== CallScalars captured ONCE for whole call ===== */
    typename Spec::CallScalars cs = Spec::populate_call_scalars(args);

    /* ===== Mode B hard-corridor counter snapshot =====
     * Mode B paths MUST NOT enter move_particles / ghost_exchange_impl /
     * gpu_particles_arena_acquire, and MUST NOT mutate NumPart. Always-on
     * enforcement around the iterative call; mirrors non-iter run_neighbor_loop
     * lines 2018-2026 + 2099-2122. Snapshot here; check post-iter-loop.
     * Mode A paths legitimately advance these counters (one ghost_import +
     * one arena_acquire per call); enforcement skipped.  A Mode D call that
     * finishes on Mode B takes the snapshot again where it switches. */
    uint64_t s_drift0 = g_global_drift_counter;
    uint64_t s_ghost0 = g_ghost_import_counter;
    uint64_t s_arena0 = g_gpu_arena_acquire_counter;
    int      s_np0    = NumPart;

    /* ===== Driver lifetime wrapped in inner scope =====
     * Driver destructor runs at scope exit; Mode B hard-corridor check fires
     * AFTER scope exit so the check covers driver cleanup hooks too (defensive
     * — current cleanup_device_context contractually MUST NOT touch
     * drift/ghost/arena globals, but inner-scope wrapping makes the invariant
     * maximally airtight). */
    {
    /* Declared before the driver so it closes after the driver has finished
     * (destructors run in reverse): the raise sees the final velocities. */
    NlrMotionTargetScope<Spec> motion_target_scope;
    NlrIterDriver<Spec> drv(args, cs);

    /* ===== Path-specific DeviceContext init =====
     * Mode B: bind ctx.P=args.P/CellP directly (no arena).
     * Mode A: acquire_arena_and_init_ctx_mode_a — arena_acquire ONCE per call
     *         then bind ctx to arena-resident P_gpu/CellP_gpu. */
    if (path == DispatchPath::ModeB_HostWalker) {
        drv.initialize_device_context_mode_b();
    } else if (path == DispatchPath::ModeD_DeviceFused) {
        drv.mode_d_tree = mode_d_lease.tree;
        drv.mode_d_lease = &mode_d_lease;
        drv.acquire_arena_and_init_ctx_mode_d();
        if constexpr (nlr_mode_d_iterates_on_device_v<Spec>) {if(NTask > 1) {nlr_mode_d_iteration_begin<Spec>(drv);}}
    } else if (path == DispatchPath::ModeA_GPU_NGL) {
        /* Both read P/CellP from the device-resident arena; Mode D differs in
         * how it finds the neighbours, not in where the particles live. */
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
        if(!nlr_args_view_is_live(args)) {
            if(ThisTask == 0) {
                fprintf(stderr, "[run_neighbor_loop_iterative<%s>] FATAL: the argument view no longer "
                        "describes the live particle storage at iter=%d (P=%p vs %p, num_total=%d vs %d). "
                        "A capacity change during the call was not reflected here.\n",
                        Spec::loop_name, drv.iter_index, (void *) args.P, (void *) P,
                        args.num_total, NumPart);
                fflush(stderr);
            }
            /* Request the stop and stay in lockstep: this loop body holds MPI collectives,
             * so breaking out on one rank alone would hang its peers.  The stop drains at
             * the next phase boundary. */
            endrun(81214);
        }

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

        /* (a) Per-subgroup dispatch on the call's path (Mode D may have handed over to Mode B). This
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
            /* A stop agreed in an earlier exchange of this call ends the call: no rank enters another. */
            if (drv.mode_d_lease && drv.mode_d_lease->stopped) break;
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
                case DispatchPath::ModeD_DeviceFused:
                    if constexpr (nlr_spec_mode_d_eligible_v<Spec>) {
                        bool on_device = false;
                        if constexpr (nlr_mode_d_iterates_on_device_v<Spec>) {
                            if(drv.mode_d_lease && drv.mode_d_lease->iter_on) {nlr_iter_dispatch_subgroup_mode_d_on_device<Spec>(drv, sg); on_device = true;}
                        }
                        if(!on_device) {nlr_iter_dispatch_subgroup_mode_d<Spec>(drv, sg);}
                    }
                    break;
            }
        }

        /* A stop agreed in this iteration's exchange: every rank leaves the iterations here, before any
         * result of it is read, and the stop drains at the run's next poll. */
        if (drv.mode_d_lease && drv.mode_d_lease->stopped) break;

        /* (b) Per-active Spec::after_iter — collect IterStatus, compact
         * active_set per subgroup, mutate radii on AdjustRadius. */
        bool after_on_device = false;
        int iteration_stop_code = 0;   /* raised by a device iteration's after_iter; agreed below */
        if constexpr (nlr_mode_d_iterates_on_device_v<Spec>) {
            if(drv.mode_d_lease && drv.mode_d_lease->iter_on) {iteration_stop_code = nlr_mode_d_after_iter_on_device<Spec>(drv); after_on_device = true;}
        }
        if(!after_on_device) {
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
        /* A one-subgroup Mode D call reduces, in the same one collective, the busiest rank's actives (its floor) and
         * a device iteration's stop code. */
        int busiest_rank_actives = drv.local_active_total, agreed_stop_code = iteration_stop_code;
        if (NTask > 1 && path == DispatchPath::ModeD_DeviceFused && args.num_subgroups == 1) {
            long long mine[3] = {drv.local_active_per_sg[0], drv.local_active_per_sg[0], iteration_stop_code}, all[3] = {0, 0, 0};
            MPI_Allreduce(mine, all, 1, nlr_count_triple_type(), nlr_count_triple_op(), MPI_COMM_WORLD);
            drv.global_active_per_sg[0] = (int) all[0]; busiest_rank_actives = (int) all[1]; agreed_stop_code = (int) all[2];
        } else if (NTask > 1) {
            MPI_Allreduce(drv.local_active_per_sg.data(),
                          drv.global_active_per_sg.data(),
                          args.num_subgroups, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (path == DispatchPath::ModeD_DeviceFused) {
                MPI_Allreduce(&drv.local_active_total, &busiest_rank_actives, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            }
        } else {
            drv.global_active_per_sg = drv.local_active_per_sg;
        }
        /* A stop any rank's device iteration raised: every rank requests it here, together, and leaves the iterations
         * before another exchange; nothing of this call is written back. */
        if (agreed_stop_code) {
            endrun(agreed_stop_code);
            if (drv.mode_d_lease) {drv.mode_d_lease->stopped = 1;}
            break;
        }
        drv.global_active_total = 0;
        for (int sg = 0; sg < args.num_subgroups; sg++) {
            drv.global_active_total += drv.global_active_per_sg[sg];
        }
        if (drv.global_active_total == 0) break;

        /* A Mode D call whose busiest rank no longer meets Mode D's bound finishes on Mode B, the path a
         * call that small takes from the start: the converged actives have dropped out and the rest cannot feed the
         * device.  One way only, decided from the count every rank has just reduced, so every rank switches at the same
         * iteration.  Mode D's storage is given back here.  Mode D's context already reads the caller's particle arrays
         * (the arena acquire aliases them, and no ghosts were imported), so Mode B continues from the same state through
         * the same context.  With Mode B switched off the call stays on Mode D, as it was routed. */
        if (path == DispatchPath::ModeD_DeviceFused && !nlr_mode_d_feeds_device(busiest_rank_actives) &&
            nlr_mode_b_switched_on<Spec>()) {
            if constexpr (nlr_mode_d_iterates_on_device_v<Spec>) {nlr_mode_d_iteration_end<Spec>(drv);}   /* Mode B reads the host's copy */
            mode_d_lease.release();
            drv.mode_d_lease = nullptr;
            path = DispatchPath::ModeB_HostWalker;
            s_drift0 = g_global_drift_counter;
            s_ghost0 = g_ghost_import_counter;
            s_arena0 = g_gpu_arena_acquire_counter;
            s_np0    = NumPart;
        }
    }

    /* A call that iterated on the device to its end: the state back to the host's copy, which the writeback reads. */
    if constexpr (nlr_mode_d_iterates_on_device_v<Spec>) {
        if(drv.mode_d_lease && !drv.mode_d_lease->stopped) {nlr_mode_d_iteration_end<Spec>(drv);}
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
        /* A Mode D exchange of this call ended in an agreed stop: no result is written back. */
        if (drv.mode_d_lease && drv.mode_d_lease->stopped) break;
        const NlrSubgroup& sgr = args.subgroups[sg];
        const int n_total = sgr.num_active_local;
        /* This loop is bounded by the subgroup's ORIGINAL active count, not by
         * how many are still iterating, so it also has to skip a subgroup whose
         * per-active state could never be staged: there are no results to write
         * back, and the constructor has already asked for the stop. */
        if (n_total <= 0 || drv.accum_uvm[sg] == nullptr) continue;
        /* Use args: Mode A iterative refreshed
         * P/CellP/num_total after ghost import; apply_active_writeback hooks
         * may read these for correctness. Mode B leaves args ==
         * base args. */
        neighbor_loop_args sub = drv.args;
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
/* The CBE corridor decides A/B for CBEGrad + AGSForce together, as AGSForce would. */
template NlrMode nlr_select_mode<AgsForceSpec>(int, int);
template void nlr_import_union_ghosts<AgsForceSpec>(unsigned int, const std::vector<int>&,
                                                     const std::vector<double>&, double);
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

