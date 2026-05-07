/* sinks/sink_environment_mode_b.h
 *
 * Mode B SPIKE for sink_environment_loop's GPU NGL+kernel block.
 *
 * STATUS: SPIKE — first Mode B caller in the codebase. Not yet a
 * NeighborLoopSpec-shaped runner; the trapdoor-with-a-lock cadence
 * (see ~/.claude/memory/reference_neighbor_loop_contract.md) requires
 * runner extraction before any second caller.
 *
 * What this file owns:
 *   - sink_env1_query_t: compact typed payload of the active sink's data
 *     that the pair body reads. NEVER ships particle_data& across ranks.
 *   - sink_env1_mode_b_evaluate(): the evaluator. Replaces the
 *     ghost_prep_fresh + sink_environment_evaluate_gpu pair when
 *     GIZMO_MODE_B_SINK_ENV1=1 and the global guard passes.
 *   - sink_env1_pair_kernel_host(): SPIKE DUPLICATE of the GPU lambda body
 *     in sink_environment_gpu.cc:152-275. Identical math; runs on host.
 *     TODO(runner-extraction): consolidate to a shared
 *     KOKKOS_INLINE_FUNCTION used by both GPU lambda and host helper.
 *     This consolidation is REQUIRED before extending Mode B to a second
 *     caller. Until consolidated, GPU<->host bit-equivalence is NOT
 *     proven by the in-call oracle — only Mode B internal consistency.
 *
 * Env flags:
 *   GIZMO_MODE_B_SINK_ENV1=1  — enable the spike path.
 *   GIZMO_MODE_B_ORACLE=1     — every-call brute oracle (Mode B walker
 *                                vs brute walker, same kernel).
 *   GIZMO_MODE_B_XVAL_DUMP=1  — dump nl_outs to a parser-friendly line for
 *                                offline env-on vs env-off cross-validation.
 *   GIZMO_PHASE0_DIAG=1       — emit PHASE0_MODEB_NGL diagnostic line.
 */

#ifndef SINK_ENVIRONMENT_MODE_B_H
#define SINK_ENVIRONMENT_MODE_B_H

#include "../declarations/allvars.h"
#include "sinks_gpu_decls.h"   /* struct sink_env_gpu_out */

#ifdef SINK_PARTICLES

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if GIZMO_MODE_B_SINK_ENV1=1 in environment. Cached static. */
int sink_env1_mode_b_env_enabled(void);

/* Compact active-side query payload. Trivially copyable for byte-level
 * MPI transfer. Conditional fields wrapped in the same #ifdefs as the
 * pair body so payload size matches per build config.
 *
 * h_search MUST come from nl_radii[a] (the radius the existing GPU path
 * was given), not from P[ii].KernelRadius — they happen to be equal in
 * the current code but the pair body is contractually keyed off the
 * caller-supplied radius.
 */
struct sink_env1_query_t {
    Vec3<double>  pos;          /* P[ii].Pos */
    Vec3<double>  vel;          /* P[ii].Vel */
    MyIDType      id;           /* P[ii].ID — for self-skip predicate */
    double        h_search;     /* nl_radii[a] — pair body's h_i */
    double        ags_h;        /* AGS_KernelRadius if defined, else sink_radius_grav */
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
    double        mass;         /* active sink mass (boundedness/Bondi paths) */
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    double        sink_radius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<double>  sink_angmom;  /* Sink_Specific_AngMom */
#endif
    int           origin_local_idx;  /* index into requester's nl_outs[] */
    int           origin_rank;       /* requester's MPI rank */
};

/* Mode B evaluator. COLLECTIVE — every rank must enter even if its
 * num_active==0, because peers receive queries and reply.
 *
 * P, CellP, NumPart, nl_active, num_active, nl_radii, nl_outs match the
 * existing call to sink_environment_evaluate_gpu(). j_type_bitmask is
 * implicit (SINK_NEIGHBOR_BITFLAG); not an arg here because the spike
 * has only one caller.
 *
 * On return: nl_outs[a] is filled with the merged contribution from
 * self-rank + all peer ranks. The host-side scatter (caller's job) is
 * unchanged.
 */
void sink_env1_mode_b_evaluate(struct particle_data *P,
                               struct gas_cell_data *CellP,
                               int *nl_active,
                               int  num_active,
                               const double *nl_radii,
                               struct sink_env_gpu_out *nl_outs);

#ifdef __cplusplus
}
#endif

#endif /* SINK_PARTICLES */

#endif /* SINK_ENVIRONMENT_MODE_B_H */
