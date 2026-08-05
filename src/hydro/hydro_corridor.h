/* hydro/hydro_corridor.h — hydro corridor mode-decision + lifecycle.
 *
 * The hydro corridor is the block of neighbor-loop work spanning density()
 * through hydro_force() whose downstream consumers share the symmetric
 * gas-gas neighbor topology: cellcorrections, gradients and hydro_force
 * always, plus dynamic_diff_vel_calc and dynamic_diff_calc under
 * TURB_DIFF_DYNAMIC.
 * Across this span, positions, KernelRadius, active membership, and the
 * ghost set are invariant: no drift/kick occurs inside the span,
 * KernelRadius is frozen by force_update_hmax() before the first consumer,
 * ActiveParticleList is frozen at step entry, and the ghost set is a
 * function of the former. The one exception is gas cells driven to
 * Mass<=0 mid-span (sink swallow, full SF conversion); consumers guard on
 * Mass>0 and merge_split removes eliminated elements. What DOES change mid-span is ghost FIELD VALUES
 * (cellcorrections writes owner-side Volume_1 that gradients consume;
 * gradients produce CellP.Gradients that hydro_force needs on ghost copies).
 * Sequence in core/accel.cc::compute_hydro_densities_and_forces:
 *
 *   ---- MODE-DECISION ENTRY: gizmo_hydro_corridor_decide_mode() ----
 *       (BEFORE density(). density/ags_density run through the iterative
 *        runner with their own per-call dispatch; the corridor mode does
 *        not steer them.)
 *   density() / ags_density()                  [iterative runner, owns its
 *                                               own iterative CSR — NOT the
 *                                               corridor's shared symmetric
 *                                               gas CSR]
 *   force_update_hmax()                        [tree update; freezes
 *                                               KernelRadius for the rest
 *                                               of the corridor]
 *   rt_source_injection / opacity_interp /
 *     compute_stellar_feedback                 [each runs its own neighbor
 *                                               loop and so its own ghost
 *                                               import, which tears down the
 *                                               single global ghost pool;
 *                                               kept ABOVE the corridor so
 *                                               they cannot tear its pool]
 *   ---- TOPOLOGY BEGIN: gizmo_hydro_corridor_begin() ----
 *       (AFTER density()+force_update_hmax() and after the loops above,
 *        BEFORE cellcorrections: the one place the corridor's shared
 *        symmetric gas CSR is built)
 *   cellcorrections_calc()                     [corridor consumer]
 *   dynamic_diff_vel_calc()                    [TURB_DIFF_DYNAMIC; corridor
 *                                               consumer. Refreshes ghost
 *                                               values first under
 *                                               HYDRO_VOLUME_CORRECTIONS,
 *                                               which dirties Density above]
 *   hydro_gradient_calc()                      [corridor consumer]
 *   mg_gradient_correction_calc()              [MHD_MODIFIED_GRADIENT]
 *   selfshield_local_incident_uv_flux()        [local-only; no ghost reads]
 *   special_rt_feedback_injection()            [local-only; corridor-aware]
 *   dynamic_diff_calc()                        [TURB_DIFF_DYNAMIC; corridor
 *                                               consumer. Refreshes ghost
 *                                               values first: gradients
 *                                               rewrites Velocity_hat, which
 *                                               its pair kernel reads]
 *   hydro_force()                              [corridor consumer —
 *                                               closes the corridor]
 *   ---- CORRIDOR EXIT: gizmo_hydro_corridor_end() ----
 *
 * This file owns the step-scoped mode state, the mode decision logic, and
 * the shared CSR lifecycle + topology guardrails.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef HYDRO_CORRIDOR_H
#define HYDRO_CORRIDOR_H

enum class GizmoHydroCorridorMode : int {
    UNSET  = 0,   /* outside the corridor; never observed by a consumer */
    MODE_A = 1,   /* large-N adaptive: corridor consumers should pick Mode A */
    MODE_B = 2,   /* tiny-N adaptive: corridor consumers should pick Mode B */
};

/* Decide the corridor mode for the current step. Called once per step from
 * core/accel.cc::compute_hydro_densities_and_forces just before density().
 *
 * Decision priority (highest first):
 *   1. TRANSPORT_SUBCYCLE builds require the shared CSR every step -> Mode A.
 *   2. adaptive: Mode B iff sum>0 && sum<=TS && max<=TM, where the summed and
 *      per-rank-max active-gas counts are compared against the optional
 *      NeighborLoopModeBThreshold{Sum,Max} parameters if set (<= 0 forces
 *      Mode A), otherwise the built-in corridor defaults. */
void gizmo_hydro_corridor_decide_mode(void);

/* Read the current corridor mode. Returns UNSET when called outside the
 * corridor span (before decide_mode or after end). */
GizmoHydroCorridorMode gizmo_hydro_corridor_get_mode(void);

/* Begin the corridor's shared topology for this step.
 *
 * Called from core/accel.cc::compute_hydro_densities_and_forces AFTER the
 * pre-corridor loops that own their own ghost imports (RT source injection,
 * gas-grain opacity interpolation, stellar feedback) and BEFORE the first
 * corridor consumer. Placing it after those loops is what stops them tearing
 * the pool this call builds.
 *   MODE_A (any rank count): imports the gas ghost pool (NTask>1) and builds
 *     the shared active-index list + symmetric gas CSR once; consumers
 *     (see the list at the top of this file) consume the CSR via
 *     gizmo_hydro_corridor_external_csr() under the runner's
 *     caller-owned-pool contract (the runner skips its own ghost
 *     import/cleanup — see neighbor_loop_runner.h GHOST-POOL OWNERSHIP).
 *     Mid-span ghost staleness is handled by
 *     gizmo_hydro_corridor_refresh_ghost_values() below, the ONLY function
 *     that may rebuild + republish the corridor CSR view.
 *   MODE_B: builds ONLY the shared active-index list (the request-driven
 *     walkers' row source) and dematerializes leftover ghosts — no import,
 *     no CSR, no arena work.
 *   UNSET: no-op (decide_mode wasn't called — defensive).
 *
 * Publish gating (Mode A): the CSR view is published only when the runner's
 * ownership contract can be satisfied — at NTask>1 that requires a LIVE
 * ghost pool (ghost_pool_is_live()); when no import ran (zero active gas
 * globally), nothing is published and consumers take their nothing-to-do
 * paths.
 * Lifetime: the underlying gizmo_sym_* globals are freed by
 * gizmo_hydro_cleanup_symlist_and_ghosts() after hydro_force();
 * gizmo_hydro_corridor_end() only clears the corridor's file-static
 * pointer view (no extra free). */
void gizmo_hydro_corridor_begin(void);

/* Refresh ghost field values mid-corridor. A refresh is a temporal
 * checkpoint, not a field-selective update: it re-packs whole owner structs
 * along fixed provenance, so the question at each call site is only "has an
 * owner field that a LATER neighbour-side reader consumes been written since
 * the last pack?". The stages, and what each protects:
 *   "pre_diff_vel"     — cellcorrections rewrote Density, which the turbulent
 *                        velocity filter reads on neighbours. HYDRO_VOLUME_
 *                        CORRECTIONS only; nothing else dirties it there.
 *   "pre_gradients"    — cellcorrections' Volume_1/Density, and under
 *                        TURB_DIFF_DYNAMIC the velocity filter's Norm_hat and
 *                        Velocity_bar.
 *   "mhd_cg_iter"      — slope-limited fields between gradient iterations.
 *   "pre_dyndiff"      — gradients rewrote Velocity_hat, which the dynamic
 *                        diffusion pair kernel reads on neighbours.
 *                        TURB_DIFF_DYNAMIC only, first iteration only.
 *   "pre_hydro_force"  — CellP.Gradients.
 * No-op unless the corridor published a
 * Mode-A CSR view. THE ONLY function that may rebuild and republish the
 * corridor CSR. Two paths:
 *   FAST — when the live ghost pool is the very import the CSR was built
 *   from (import-epoch match): value-only in-place refresh via
 *   ghost_refresh_values(); slot identity preserved by construction, CSR
 *   and provenance untouched.
 *   FULL (fail-closed fallback) — pool torn down by an intervening loop's
 *   own ghost lifecycle, epoch mismatch, or any refresh-guard failure:
 *   full teardown + re-import + CSR rebuild + REPUBLISH with new
 *   offsets/neighbors pointers.
 * Because a refresh MAY republish, consumers must RE-FETCH
 * gizmo_hydro_corridor_external_csr() after every refresh (gradients: once
 * per MHD-CG iteration), never cache the view across a refresh.
 * `stage` is a short diagnostic label ("pre_gradients", "mhd_cg_iter",
 * "pre_hydro_force"). */
void gizmo_hydro_corridor_refresh_ghost_values(const char *stage);

/* Accessor for downstream corridor consumers. Returns a non-null pointer
 * iff gizmo_hydro_corridor_begin populated a usable Mode-A CSR for this
 * corridor span. nullptr otherwise: Mode B consumers use the corridor-built
 * active list (request-driven, no CSR); Mode A with active gas and no view
 * is a sequencing bug the consumers abort on (the quiet rebuild-your-own
 * fallback is retired). */
struct nlr_external_csr;
const nlr_external_csr * gizmo_hydro_corridor_external_csr(void);

/* Tear down the corridor at the end of hydro work for this step. Resets
 * the mode to UNSET so any unexpected read between steps is detectable.
 * Also clears the corridor's file-static external_csr view (does NOT
 * free gizmo_sym_*; that stays with gizmo_hydro_cleanup_symlist_and_ghosts). */
void gizmo_hydro_corridor_end(void);

#endif /* HYDRO_CORRIDOR_H */
