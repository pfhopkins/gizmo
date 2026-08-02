/* sinks/sink_env1_loop.cc — host-only hooks for SinkEnv1Spec.
 *
 * KOKKOS_INLINE_FUNCTION hooks (load_active, load_neighbor, pair_kernel,
 * zero_accum) and the single source of truth for the pair body live in
 * sink_env1_loop.h so they inline from both device kernels (Mode A) and
 * host walkers (Mode B / Brute oracle). This translation unit holds the
 * host-only hooks: per-active radius, per-call scalars capture, host
 * writebacks (apply_active_writeback, merge_accum), imported-ghost
 * lifecycle hooks, and the env-gated diagnostics block at the bottom.
 *
 * File layout (matches sink_env1_loop.h conventions):
 *   PHYSICS HOOKS                — search_radius, populate_call_scalars,
 *                                   apply_active_writeback, merge_accum
 *   IMPORTED-GHOST LIFECYCLE     — write detector + writeback wrappers
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* Kokkos_Core.hpp must precede allvars.h (its macros may conflict with stdlib
 * names); sink_env1_loop.h emits inline Kokkos here. */
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"               /* MUST precede sink_env1_loop.h */
#include "../mesh/ghost_writeback.h"      /* scaffold + detector */
#include "../mesh/ghost_writeback_ops.h"  /* manifest macros (B.iv) */
#include "sink_env1_loop.h"

#ifdef SINK_PARTICLES

/* ============================================================================
 * PHYSICS HOOKS
 * ========================================================================== */

/* Per-active search radius from external state (P[i].KernelRadius).
 * Pre-arena, pre-drift. The runner stages a radii_uvm[num_active] array
 * from this hook and passes it to gpu_ngb_list_build (Mode A) and Mode
 * B's per-query walker. */
double SinkEnv1Spec::search_radius(const neighbor_loop_args& args,
                                    int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

/* Capture per-call cosmology + gravity scalars into a typed POD. The
 * common cosmology factors come from nlr_common_scalars_from_all() so
 * a Spec author cannot forget them; only loop-specific scalars need
 * explicit assignment here. */
SinkEnv1Spec::CallScalars
SinkEnv1Spec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    CallScalars scalars;
    scalars.common           = nlr_common_scalars_from_all();
    scalars.sink_radius_grav = SinkParticle_GravityKernelRadius;
    return scalars;
}

/* Device-context lifecycle. sink_env1 has no UVM-allocated
 * per-active state (unlike sink_feed/sink_swk), so populate only seeds
 * the oracle dry-run flag and cleanup is a no-op. The pair is declared
 * for symmetric runner-contract pairing. */
void SinkEnv1Spec::populate_device_context(const neighbor_loop_args& /*args*/,
                                            DeviceContext& ctx)
{
}

void SinkEnv1Spec::cleanup_device_context(const neighbor_loop_args& /*args*/,
                                           DeviceContext& /*ctx*/)
{
}

/* Copy AccumData into args.aux->per_active_accum[active_slot]. The caller
 * (sinks/sink_environment.cc) reads per_active_accum in its scatter loop
 * and applies physics-specific reductions into SinkTempInfo. AccumData is
 * POD; the runner already zero-initialized in zero_accum, so this is a
 * straight assignment. */
void SinkEnv1Spec::apply_active_writeback(const neighbor_loop_args& args,
                                           int active_slot, int /*i*/,
                                           const AccumData& accum)
{
    Aux *aux = nlr_aux<SinkEnv1Spec>(args);
    aux->per_active_accum[active_slot] = accum;
}

/* Per-field merge of a peer rank's contribution into a local accumulator.
 * Per-field op MUST match the pair_kernel writes (sum for additive fields,
 * MAX for max-reduced fields, componentwise sum for vec3). Used by
 * run_mode_b_remote at the cross-rank boundary; within a single rank,
 * accumulation is via repeated pair_kernel calls (which already encode
 * the right per-field op).
 *
 * The oracle (Mode B via the parameterfile NeighborLoopModeBThreshold pair,
 * plus GIZMO_NLR_ORACLE=1) catches drift
 * between this manifest and pair_kernel writes — any mismatch surfaces
 * as ORACLE MISMATCH on the next run.
 *
 * Adding a new accumulator field for this loop = ONE LINE under the
 * appropriate physics flag's #ifdef. Operations available below; extend
 * by adding new ACCUM_* defines if a field needs different op semantics. */
void SinkEnv1Spec::merge_accum(AccumData& local_accum, const AccumData& peer_accum)
{
    /* Local op macros — scoped to this function via #undef below. The
     * macros expand to the same statements as the prior hand-written
     * field listing (semantically identical expansion). */
#define ACCUM_ADD(field)       local_accum.field += peer_accum.field;
#define ACCUM_ADD_VEC3(field)  for(int k = 0; k < 3; k++) local_accum.field[k] += peer_accum.field[k];
#define ACCUM_MAX(field)       if(peer_accum.field > local_accum.field) local_accum.field = peer_accum.field;

    ACCUM_ADD(Sink_SurroudingGasInternalEnergy)
    ACCUM_ADD(Mgas_in_Kernel)
    ACCUM_ADD(Mstar_in_Kernel)
    ACCUM_ADD(Malt_in_Kernel)
    ACCUM_ADD_VEC3(Jgas_in_Kernel)
    ACCUM_ADD_VEC3(Jstar_in_Kernel)
    ACCUM_ADD_VEC3(Jalt_in_Kernel)
#ifdef SINK_REPOSITION_ON_POTMIN
    ACCUM_ADD(DF_rms_vel)
    ACCUM_ADD_VEC3(DF_mean_vel)
    ACCUM_MAX(DF_mmax_particles)
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    ACCUM_ADD(Sfr_in_Kernel)
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    ACCUM_ADD_VEC3(Sink_SurroundingGasVel)
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
    ACCUM_ADD_VEC3(Sink_SurroundingGasCOM)
#endif
#if (SINK_GRAVACCRETION == 8)
    ACCUM_ADD(hubber_mdot_bondi_limiter)
    ACCUM_ADD(hubber_mdot_vr_estimator)
    ACCUM_ADD(hubber_mdot_disk_estimator)
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    ACCUM_ADD(mass_to_swallow_edd)
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    ACCUM_ADD_VEC3(angmom_prepass_sum_for_passback)
#endif
#if defined(SINK_RETURN_BFLUX)
    ACCUM_ADD(kernel_norm_topass_in_swallowloop)
#endif

#undef ACCUM_ADD
#undef ACCUM_ADD_VEC3
#undef ACCUM_MAX
}

/* ============================================================================
 * IMPORTED-GHOST LIFECYCLE
 *
 * Two channels: write detector + ghost writeback. The runner gates each
 * pair on (a) the corresponding `uses_*` trait being true AND (b)
 * nlr_path_uses_imported_ghosts(plan.path) being true.
 *
 *   - Mode A (imported-ghost lifecycle): the runner preserves the order
 *     detector_begin -> writeback_begin -> kernel -> writeback_end ->
 *     detector_end. Behavior is byte-identical to the prior caller-side
 *     sequence under all relevant compile configs.
 *   - Mode B paths: the runner skips both pairs. There is no imported-
 *     ghost lifecycle for them to attend to (lazy-drift corridor).
 *
 * GHOST WRITEBACK MANIFEST (Pass B.iv).
 *
 * Below is the entire manifest of ghost-written fields for sink_env1.
 * Adding a new ghost-written field for this loop = ADDING ONE LINE under
 * its physics flag's #ifdef. The scaffold (mesh/ghost_writeback.cc)
 * generates the snapshot, change-predicate, pack, exchange, apply, and
 * cleanup machinery from each manifest line.
 *
 * Operations available: see mesh/ghost_writeback_ops.h.
 *
 * Empty bundle (no flags active): scaffold short-circuits at line 1 of
 * begin_bundle and end_bundle — strict no-op, identical to "writeback
 * absent." Default builds without SINGLE_STAR_SINK_DYNAMICS pay no cost.
 * ========================================================================== */

GHOST_WRITEBACK_BUNDLE_BEGIN(sink_env1)
#ifdef SINGLE_STAR_SINK_DYNAMICS
    GHOST_WRITEBACK_PARTICLE_MIN(SwallowTime)
#endif
    /* future ghost-written fields: add one line per (op, field) under the
     * appropriate physics flag #ifdef. */
GHOST_WRITEBACK_BUNDLE_END(sink_env1)

/* ghost_write_detector_begin/end: runner default (see sink_env1_loop.h
 * ghost_write_detector_name override). */

void SinkEnv1Spec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                          const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_begin_bundle(sink_env1_ghost_writeback_bundle_ptr());
}

void SinkEnv1Spec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                        const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(sink_env1_ghost_writeback_bundle_ptr());
}

#else  /* !SINK_PARTICLES */


#endif /* SINK_PARTICLES */
