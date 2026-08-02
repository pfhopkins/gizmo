/* sinks/sink_feed_loop.cc — host-only hooks for SinkFeedSpec.
 *
 * KOKKOS_INLINE_FUNCTION hooks (load_active, load_neighbor, pair_kernel,
 * zero_accum) and the inline pair body (sink_feed_pair_kernel) live in
 * sink_feed_loop.h so they inline from device kernels (Mode A) and host
 * walkers (Mode B). This translation unit holds host-only hooks: per-active
 * radius, per-call scalars capture (with sink_feed-unique RNG shift),
 * populate/cleanup_device_context (UVM staging for per-active
 * SinkFeedLocalIn), apply_active_writeback, merge_accum, and the
 * ghost-writeback manifest + lifecycle hooks.
 *
 * Replaces sinks/sink_feed_gpu.cc + sinks/sink_feed_functions.h.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../declarations/gpu_numeric_macros.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"               /* MUST precede sink_feed_loop.h */
#include "../mesh/ghost_writeback.h"      /* scaffold + detector */
#include "../mesh/ghost_writeback_ops.h"  /* manifest macros */
#include "sink_feed_loop.h"

#ifdef SINK_PARTICLES

/* ============================================================================
 * ACTIVE PREDICATE
 * Mirrors sink_isactive + extra guards from the legacy particle2in
 * (sink_feed_gpu.cc:38–46).
 * ========================================================================== */

int sink_feed_is_active(int i)
{
    if(P[i].Type != 5) return 0;
    if(P[i].Mass <= 0) return 0;
    if(P[i].KernelRadius <= 0) return 0;
    if(P[i].TimeBin < 0) return 0;
    return 1;
}

/* ============================================================================
 * PHYSICS HOOKS
 * ========================================================================== */

double SinkFeedSpec::search_radius(const neighbor_loop_args& args,
                                    int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

/* Per-call scalars + sink_feed-unique RNG shift. The 0xfeed5117ULL XOR
 * shift on top of NumCurrentTiStep makes per-particle draws disjoint
 * across different sink loops in the same timestep — see
 * feedback_rng_loop_uniqueness.md in project memory. */
SinkFeedSpec::CallScalars
SinkFeedSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    /* Uses nlr_host_all_ptr() for explicit host-snapshot intent. The
     * redirect is device-pass-only, so bare All.* in this host
     * hook would also be correct; the accessor documents that these reads
     * run on the host pass. */
    const struct global_data_all_processes *h = nlr_host_all_ptr();
    CallScalars scalars;
    scalars.common                 = nlr_common_scalars_from_all();
    scalars.sink_radius_grav       = SinkParticle_GravityKernelRadius;
#if defined(SINK_WIND_KICK) || defined(SINK_WIND_SPAWN)
    scalars.sink_accreted_fraction = h->Sink_accreted_fraction;
#else
    scalars.sink_accreted_fraction = 0.0;  /* All.Sink_accreted_fraction only exists under
                                              SINK_WIND_KICK || SINK_WIND_SPAWN; pair-body
                                              readers are all under matching #ifdefs, so the
                                              fallback value is never observed. */
#endif
    scalars.rng_step               = (uint64_t)h->NumCurrentTiStep ^ 0xfeed5117ULL;
    return scalars;
}

/* Copy AccumData into args.aux->per_active_accum[active_slot]. The caller
 * (sinks/sink_feed.cc::sink_feed_loop) reads per_active_accum after the
 * runner returns and applies physics-specific reductions (SinkTempInfo
 * additive scatter, P[i] potential-min pair) via the caller-side scatter
 * manifest. */
void SinkFeedSpec::apply_active_writeback(const neighbor_loop_args& args,
                                           int active_slot, int /*i*/,
                                           const AccumData& accum)
{
    Aux *aux = nlr_aux<SinkFeedSpec>(args);
    aux->per_active_accum[active_slot] = accum;
}

/* Per-field merge of a peer's contribution (Mode B remote). Per-field op
 * MUST match the pair_kernel writes. mass_markedswallow_scratch is
 * intentionally NOT in the manifest — per-i scratch, not aggregable
 * across peers. Nothing checks this manifest against pair_kernel at runtime,
 * so drift between them is silent.
 *
 * Adding a new accumulator field for this loop = ONE LINE under the
 * appropriate physics flag's #ifdef. */
void SinkFeedSpec::merge_accum(AccumData& local_accum, const AccumData& peer_accum)
{
#define ACCUM_ADD(field)                local_accum.field += peer_accum.field;
#define ACCUM_ADD_VEC3(field)           for(int k = 0; k < 3; k++) local_accum.field[k] += peer_accum.field[k];
#define ACCUM_MIN_PAIR(value, pos)                                              \
    do {                                                                        \
        if(peer_accum.value < local_accum.value) {                              \
            local_accum.value = peer_accum.value;                               \
            for(int k = 0; k < 3; k++) local_accum.pos[k] = peer_accum.pos[k];  \
        }                                                                       \
    } while(0);

#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
    ACCUM_ADD(Sink_angle_weighted_kernel_sum)
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    ACCUM_MIN_PAIR(Sink_PotentialMinimumOfNeighbors,
                   Sink_PotentialMinimumOfNeighborsPos)
#endif

#undef ACCUM_ADD
#undef ACCUM_ADD_VEC3
#undef ACCUM_MIN_PAIR
}

/* ============================================================================
 * DEVICE CONTEXT LIFECYCLE
 *
 * populate_device_context allocates a UVM array of SinkFeedLocalIn[N] and
 * copies in from args.aux->host_locals (which the caller fills on host
 * before run_neighbor_loop using sink_feed_local_fill semantics). Device-
 * side load_active reads dctx.per_active_local[aa].
 *
 * cleanup_device_context frees the UVM. Runs unconditionally via
 * NlrDeviceContextCleanupGuard at runner exit (any path — Mode A,
 * Mode B local, Mode B remote).
 * ========================================================================== */

void SinkFeedSpec::populate_device_context(const neighbor_loop_args& args,
                                            DeviceContext& ctx)
{
    /* Null all UVM pointers BEFORE any early-return so cleanup is safe on
     * every path. */
    ctx.per_active_local  = nullptr;
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    ctx.binary_merge_eligible = nullptr;
#endif

    Aux *aux = nlr_aux<SinkFeedSpec>(args);
    const int N = args.num_active;
    if(N > 0) {
        SinkFeedLocalIn *uvm = (SinkFeedLocalIn *)
            Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(SinkFeedLocalIn));
        std::memcpy(uvm, aux->host_locals, N * sizeof(SinkFeedLocalIn));
        ctx.per_active_local = uvm;
    }

#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    /* binary_merge_eligible is sized ctx.num_total — same index space as
     * ctx.P[j]. Covers local + imported-ghost slots (Mode-A ghost-import
     * appends ghosts to global P[] BEFORE this hook runs). Gated
     * INDEPENDENTLY of args.num_active: Mode-B-remote responder ranks
     * with N==0 still walk this rank's local pool for peer queries and
     * load_neighbor reads dctx.binary_merge_eligible[j]. The fill loop
     * uses global P[j] (not ctx.P[j]) to remain consistent with
     * is_star_eligible_for_binary_merge_away(j), which itself reads
     * global P[j]. Mode-A post-import invariant (NumPart == ctx.num_total,
     * P grown to include ghosts) guarantees safety. */
    const int n_total = ctx.num_total;
    if(n_total > 0) {
        unsigned char *elig = (unsigned char *)
            Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
                (size_t)n_total * sizeof(unsigned char));
        for(int j = 0; j < n_total; j++) {
            elig[j] = (P[j].Type == 5
                       && is_star_eligible_for_binary_merge_away(j)) ? 1 : 0;
        }
        ctx.binary_merge_eligible = elig;
    }
#endif
}

void SinkFeedSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                           DeviceContext& ctx)
{
    /* Free each pointer independently; null after free so a second
     * cleanup (defensive — runner only invokes cleanup once per dispatch)
     * would be a no-op. */
    if(ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<SinkFeedLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    if(ctx.binary_merge_eligible) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<unsigned char *>(ctx.binary_merge_eligible));
        ctx.binary_merge_eligible = nullptr;
    }
#endif
}

/* ============================================================================
 * IMPORTED-GHOST LIFECYCLE
 *
 * Two channels: write detector + ghost writeback. Runner gates each pair
 * on (a) the corresponding `uses_*` trait being true AND (b)
 * nlr_path_uses_imported_ghosts(plan.path) being true.
 *
 *   - Mode A: detector_begin -> writeback_begin -> kernel ->
 *             writeback_end -> detector_end.  bundle end performs
 *             the snapshot-diff reverse-comm of j-side writes.
 *   - Mode B local + remote: skipped (j-side writes are local on the
 *             rank that owns j; no reverse-comm needed).
 *
 * GHOST WRITEBACK MANIFEST (first port with non-empty bundle).
 * Adding a new ghost-written field for this loop = ADDING ONE LINE under
 * its physics flag's #ifdef. The scaffold (mesh/ghost_writeback.cc)
 * generates snapshot/changed/pack/exchange/apply/cleanup machinery from
 * each manifest line.
 *
 * Operations available: see mesh/ghost_writeback_ops.h.
 * ========================================================================== */

GHOST_WRITEBACK_BUNDLE_BEGIN(sink_feed)
    /* PARTICLE_MAX(SwallowID) — D1/S-SYNC max-ID-wins tiebreak across ranks.
     * Legacy: ghost_writeback.cc:1012. */
    GHOST_WRITEBACK_PARTICLE_MAX(SwallowID)
#ifdef SINK_THERMALFEEDBACK
    /* GAS_ADD(Injected_Sink_Energy) — additive merge over per-rank deltas.
     * Legacy: ghost_writeback.cc:1016. */
    GHOST_WRITEBACK_GAS_ADD(Injected_Sink_Energy)
#endif
GHOST_WRITEBACK_BUNDLE_END(sink_feed)

/* ghost_write_detector_begin/end: runner default (loop_name = "sink_feed"). */

void SinkFeedSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                          const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_begin_bundle(sink_feed_ghost_writeback_bundle_ptr());
}

void SinkFeedSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                        const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(sink_feed_ghost_writeback_bundle_ptr());
}

#else  /* !SINK_PARTICLES */


#endif /* SINK_PARTICLES */
