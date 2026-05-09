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
 *   DIAGNOSTICS (env-gated)      — compare_accum (PERMANENT), and
 *                                   diagnostic_dump_* (SPIKE / cross-validation
 *                                   probes; scheduled retire after 3d ports)
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
 * The oracle (GIZMO_NLR_FORCE_MODE=B + GIZMO_NLR_ORACLE=1) catches drift
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

void SinkEnv1Spec::ghost_write_detector_begin(const neighbor_loop_args& /*args*/,
                                               const NeighborLoopPlan& /*plan*/)
{
    ::ghost_write_detector_begin("sink_environment");
}

void SinkEnv1Spec::ghost_write_detector_end(const neighbor_loop_args& /*args*/,
                                             const NeighborLoopPlan& /*plan*/)
{
    ::ghost_write_detector_end();
}

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

/* ============================================================================
 * DIAGNOSTICS — env-gated, safe to ignore for physics edits.
 *
 * PERMANENT_DIAGNOSTIC : compare_accum (oracle gate, called only when
 *                       GIZMO_NLR_ORACLE=1)
 *
 * SPIKE_DIAGNOSTIC     : diagnostic_dump_active, diagnostic_dump_neighbor_list
 *                       (cross-validation probes from the Mode B bring-up;
 *                       scheduled retire after 3d ports complete)
 *
 * Canonical env vars (Pass B.i):
 *   GIZMO_NLR_ORACLE=1                gates compare_accum invocation
 *   GIZMO_NLR_ORACLE_DUMP=1           field-by-field oracle mismatch dump
 *   GIZMO_NLR_SPIKE_ACCUM_DUMP=1      per-active accumulator dump (SPIKE)
 *   GIZMO_NLR_SPIKE_NB_DUMP=1         first-call neighbor-list dump (SPIKE)
 * Old names (GIZMO_MODE_B_XVAL_DUMP, GIZMO_MODE_B_XVAL_NB_DUMP) accepted as
 * aliases with rank-0 deprecation warning. See mesh/neighbor_loop_runner.h
 * env-config block for the full alias / conflict policy.
 * ========================================================================== */

/* PERMANENT_DIAGNOSTIC — oracle compare.
 *
 * Returns the maximum relative residual across all AccumData fields. The
 * runner gates calls behind GIZMO_NLR_ORACLE=1 + Spec::accum_tolerance.
 *
 * Implementation: walk the byte representation as MyFloat scalars and
 * compute max relative diff per-field. AccumData layout is exclusively
 * MyFloat scalars and MyFloat[3] arrays (see sinks/sinks_gpu_decls.h);
 * for a future MyFloat=float build this still produces a meaningful
 * relative residual. */
double SinkEnv1Spec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    double max_rel = 0.0;
    size_t max_rel_field = 0;
    const MyFloat *pa = reinterpret_cast<const MyFloat*>(&local);
    const MyFloat *pb = reinterpret_cast<const MyFloat*>(&oracle);
    const size_t n = sizeof(AccumData) / sizeof(MyFloat);
    static_assert(sizeof(AccumData) % sizeof(MyFloat) == 0,
        "SinkEnv1Spec::AccumData must be MyFloat-aligned for byte-walk compare");
    for(size_t k = 0; k < n; k++) {
        double va = (double)pa[k], vb = (double)pb[k];
        double denom = std::fmax(std::fabs(va), std::fabs(vb));
        double diff  = std::fabs(va - vb);
        double rel   = (denom > 0.0) ? (diff / denom) : diff;
        if(rel > max_rel) { max_rel = rel; max_rel_field = k; }
    }
    /* When GIZMO_NLR_ORACLE_DUMP=1 and max_rel exceeds tolerance, dump every
     * nonzero-diff field. Distinguishes summation-reorder roundoff from
     * physics drift. Print cap = 16 calls. */
    static int s_dump_cached = -1;
    if(s_dump_cached < 0) {
        const char *e = std::getenv("GIZMO_NLR_ORACLE_DUMP");
        s_dump_cached = (e && e[0] == '1') ? 1 : 0;
    }
    static int s_dump_count = 0;
    if(s_dump_cached && max_rel > 1e-10 && s_dump_count < 16) {
        std::fprintf(stderr, "[compare_accum DUMP call=%d max_rel=%g (field=%zu) abs=%g local=%g oracle=%g]\n",
                     s_dump_count, max_rel, max_rel_field,
                     std::fabs((double)pa[max_rel_field] - (double)pb[max_rel_field]),
                     (double)pa[max_rel_field], (double)pb[max_rel_field]);
        for(size_t k = 0; k < n; k++) {
            double va = (double)pa[k], vb = (double)pb[k];
            double denom = std::fmax(std::fabs(va), std::fabs(vb));
            double diff  = std::fabs(va - vb);
            double rel   = (denom > 0.0) ? (diff / denom) : diff;
            if(diff > 0.0) {
                std::fprintf(stderr, "  field[%zu] local=%.17g oracle=%.17g abs=%.6g rel=%.6g\n",
                             k, va, vb, diff, rel);
            }
        }
        std::fflush(stderr);
        s_dump_count++;
    }
    return max_rel;
}

/* SPIKE_DIAGNOSTIC — per-active accumulator dump (cross-validation).
 *
 * Stable line shape across the runner's Mode A and Mode B paths. Active
 * line label "MODEB_XVAL" is preserved for parser compatibility; rename to
 * a single GIZMO_NLR_DIAG=<level>-driven label is queued for the
 * runner-template-hardening pass. */
void SinkEnv1Spec::diagnostic_dump_active(const ActiveDumpView<SinkEnv1Spec>& v)
{
    const AccumData& a = *v.accum;
    std::printf("MODEB_XVAL rank=%d caller=sink_env1 active_local=%d "
                "Mgas=%g Mstar=%g Malt=%g IE=%g "
                "Jgas=%g,%g,%g Jstar=%g,%g,%g Jalt=%g,%g,%g\n",
                v.rank, v.active_slot,
                (double)a.Mgas_in_Kernel,
                (double)a.Mstar_in_Kernel,
                (double)a.Malt_in_Kernel,
                (double)a.Sink_SurroudingGasInternalEnergy,
                (double)a.Jgas_in_Kernel[0], (double)a.Jgas_in_Kernel[1], (double)a.Jgas_in_Kernel[2],
                (double)a.Jstar_in_Kernel[0], (double)a.Jstar_in_Kernel[1], (double)a.Jstar_in_Kernel[2],
                (double)a.Jalt_in_Kernel[0], (double)a.Jalt_in_Kernel[1], (double)a.Jalt_in_Kernel[2]);
}

/* SPIKE_DIAGNOSTIC — neighbor-list dump (Mode A only; first call per
 * process). Runner only invokes this when GIZMO_NLR_SPIKE_NB_DUMP=1
 * (or its old alias GIZMO_MODE_B_XVAL_NB_DUMP=1) and the call is on the
 * Mode A path with a live GPU neighbor-list CSR.
 *
 * The runner calls this hook once per active in slot order
 * (0..num_active-1). The first-call gate sets s_fired only after the
 * LAST active of the first call has printed, matching legacy semantics. */
void SinkEnv1Spec::diagnostic_dump_neighbor_list(const NeighborListDumpView<SinkEnv1Spec>& v)
{
    static int s_fired = 0;
    if(s_fired) return;
    const struct neighbor_loop_args& args = *v.args;
    const struct particle_data *P_host = args.P;
    const int ii = args.active_list[v.active_slot];
    const double h_i = (double)v.active->h_search;
    std::printf("MODEB_XVAL_NB rank=%d call=1 active=%d path=%s "
                "h_search=%.17g i_pos=%.17g,%.17g,%.17g "
                "i_vel=%.17g,%.17g,%.17g i_id=%llu n_j=%d\n",
                v.rank, v.active_slot, v.path, h_i,
                (double)P_host[ii].Pos[0], (double)P_host[ii].Pos[1], (double)P_host[ii].Pos[2],
                (double)P_host[ii].Vel[0], (double)P_host[ii].Vel[1], (double)P_host[ii].Vel[2],
                (unsigned long long)P_host[ii].ID, v.n_candidates);
    for(int idx = 0; idx < v.n_candidates; idx++) {
        const int j = v.candidate_ids[idx];
        double dx = (double)P_host[j].Pos[0] - (double)P_host[ii].Pos[0];
        double dy = (double)P_host[j].Pos[1] - (double)P_host[ii].Pos[1];
        double dz = (double)P_host[j].Pos[2] - (double)P_host[ii].Pos[2];
        NEAREST_XYZ(dx, dy, dz, 1);
        const double r2 = dx*dx + dy*dy + dz*dz;
        std::printf("MODEB_XVAL_NB_J rank=%d call=1 active=%d path=%s "
                    "j=%d Type=%d Mass=%.17g KernelRadius=%.17g r2=%.17g "
                    "Pos=%.17g,%.17g,%.17g Vel=%.17g,%.17g,%.17g ID=%llu\n",
                    v.rank, v.active_slot, v.path, j, (int)P_host[j].Type,
                    (double)P_host[j].Mass, (double)P_host[j].KernelRadius, r2,
                    (double)P_host[j].Pos[0], (double)P_host[j].Pos[1], (double)P_host[j].Pos[2],
                    (double)P_host[j].Vel[0], (double)P_host[j].Vel[1], (double)P_host[j].Vel[2],
                    (unsigned long long)P_host[j].ID);
    }
    if(v.active_slot == args.num_active - 1) s_fired = 1;
}

#endif /* SINK_PARTICLES */
