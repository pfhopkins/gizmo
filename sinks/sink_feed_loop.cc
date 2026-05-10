/* sinks/sink_feed_loop.cc — host-only hooks for SinkFeedSpec.
 *
 * KOKKOS_INLINE_FUNCTION hooks (load_active, load_neighbor, pair_kernel,
 * zero_accum) and the inline pair body (sink_feed_pair_kernel) live in
 * sink_feed_loop.h so they inline from device kernels (Mode A) and host
 * walkers (Mode B / Brute oracle). This translation unit holds host-only
 * hooks: per-active radius, per-call scalars capture (with sink_feed-
 * unique RNG shift), populate/cleanup_device_context (Phase 4.A.0 UVM
 * staging for per-active SinkFeedLocalIn), apply_active_writeback,
 * merge_accum, ghost-writeback manifest + lifecycle hooks, and the
 * env-gated diagnostics block.
 *
 * Replaces sinks/sink_feed_gpu.cc + sinks/sink_feed_functions.h. Phase 4
 * port 3d.1.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../declarations/gpu_all_mirror.h"   /* GPU_ALL_SYNC_FUNC */
#include "../declarations/gpu_numeric_macros.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"               /* MUST precede sink_feed_loop.h */
#include "../mesh/ghost_writeback.h"      /* scaffold + detector */
#include "../mesh/ghost_writeback_ops.h"  /* manifest macros (B.iv + 3d.1 ops) */
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
    CallScalars scalars;
    scalars.common                 = nlr_common_scalars_from_all();
    scalars.sink_radius_grav       = SinkParticle_GravityKernelRadius;
    scalars.sink_accreted_fraction = All.Sink_accreted_fraction;
    scalars.rng_step               = (uint64_t)All.NumCurrentTiStep ^ 0xfeed5117ULL;
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
 * across peers. The oracle (FORCE_MODE=B + ORACLE=1) catches drift between
 * this manifest and pair_kernel writes.
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
 * PHASE 4.A.0 DEVICE CONTEXT LIFECYCLE
 *
 * populate_device_context allocates a UVM array of SinkFeedLocalIn[N] and
 * copies in from args.aux->host_locals (which the caller fills on host
 * before run_neighbor_loop using sink_feed_local_fill semantics). Device-
 * side load_active reads dctx.per_active_local[aa].
 *
 * cleanup_device_context frees the UVM. Runs unconditionally via
 * NlrDeviceContextCleanupGuard at runner exit (any path — Mode A,
 * Mode B local, Mode B remote, oracle).
 * ========================================================================== */

void SinkFeedSpec::populate_device_context(const neighbor_loop_args& args,
                                            DeviceContext& ctx)
{
    /* Oracle dry-run flag defaults off; runner flips on for the brute pass
     * via Spec::set_oracle_brute_pass, which reads/writes a copied ctx. */
    ctx.oracle_dry_run = false;

    Aux *aux = nlr_aux<SinkFeedSpec>(args);
    const int N = args.num_active;
    if(N <= 0) {
        ctx.per_active_local = nullptr;
        return;
    }
    SinkFeedLocalIn *uvm = (SinkFeedLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(N * sizeof(SinkFeedLocalIn));
    std::memcpy(uvm, aux->host_locals, N * sizeof(SinkFeedLocalIn));
    ctx.per_active_local = uvm;
}

void SinkFeedSpec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                           DeviceContext& ctx)
{
    if(ctx.per_active_local) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(
            const_cast<SinkFeedLocalIn *>(ctx.per_active_local));
        ctx.per_active_local = nullptr;
    }
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
 * GHOST WRITEBACK MANIFEST (3d.1 — first port with non-empty bundle).
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

void SinkFeedSpec::ghost_write_detector_begin(const neighbor_loop_args& /*args*/,
                                               const NeighborLoopPlan& /*plan*/)
{
    ::ghost_write_detector_begin("sink_feed");
}

void SinkFeedSpec::ghost_write_detector_end(const neighbor_loop_args& /*args*/,
                                             const NeighborLoopPlan& /*plan*/)
{
    ::ghost_write_detector_end();
}

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

/* ============================================================================
 * DIAGNOSTICS — env-gated.
 * compare_accum: oracle gate, called only when GIZMO_NLR_ORACLE=1.
 * ========================================================================== */

double SinkFeedSpec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    /* Walk byte representation as MyFloat; same approach as sink_env1. The
     * mass_markedswallow_scratch field is per-i scratch and NOT replicated
     * across peers in merge_accum, so single-rank Mode B and Brute see the
     * same value. We include it in the byte walk; if it ever disagrees,
     * the oracle output identifies the field. */
    double max_rel = 0.0;
    /* AccumData mixes double + Vec3<double>; both are "double" at the byte
     * level. Treat as a byte-walk of doubles. */
    const double *pa = reinterpret_cast<const double*>(&local);
    const double *pb = reinterpret_cast<const double*>(&oracle);
    static_assert(sizeof(AccumData) % sizeof(double) == 0,
        "SinkFeedSpec::AccumData must be double-aligned for byte-walk compare");
    const size_t n = sizeof(AccumData) / sizeof(double);
    for(size_t k = 0; k < n; k++) {
        double va = pa[k], vb = pb[k];
        double denom = std::fmax(std::fabs(va), std::fabs(vb));
        double diff  = std::fabs(va - vb);
        double rel   = (denom > 0.0) ? (diff / denom) : diff;
        if(rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

/* GPU All_dev sync stub for cooling.cc's per-TU sync sweep. sink_feed_loop.cc
 * is host-pattern-compiled (NOT in GPU_OBJS), so this expands to a stub on
 * both Mac (OpenMP backend) and Vista (CUDA backend); there is no per-TU
 * All_dev for sink_feed_loop because the device kernel runs inside the
 * runner's TU (mesh/neighbor_loop_runner.cc) and uses that TU's All_dev. */
GPU_ALL_SYNC_FUNC_STUB(sinkfeed)

#else  /* !SINK_PARTICLES */

GPU_ALL_SYNC_FUNC_STUB(sinkfeed)

#endif /* SINK_PARTICLES */
