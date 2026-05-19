/* hydro/hydro_corridor.h — hydro corridor mode-decision + lifecycle.
 *
 * The hydro corridor is the block of neighbor-loop work spanning density()
 * through hydro_force() whose downstream consumers (cellcorrections,
 * gradients, hydro_force) share the symmetric gas-gas neighbor topology
 * (positions / KernelRadius / active membership / ghost set invariant
 * across this span; see §1.5 of OPEN_3d_hydro_corridor_design.md for the
 * topology audit and the Mass<=0 guardrail). Sequence in
 * core/accel.cc::compute_hydro_densities_and_forces:
 *
 *   ---- MODE-DECISION ENTRY: gizmo_hydro_corridor_decide_mode() ----
 *       (BEFORE density() — so density can later honor the corridor
 *        choice; commit 4 leaves density independent, future commits
 *        wire density-side dispatch to the corridor mode)
 *   density() / ags_density()                  [iterative runner, owns its
 *                                               own iterative CSR — NOT the
 *                                               corridor's shared symmetric
 *                                               gas CSR]
 *   force_update_hmax()                        [tree update; freezes
 *                                               KernelRadius for the rest
 *                                               of the corridor]
 *   ---- (FUTURE COMMIT 5) CSR/TOPOLOGY BEGIN ----
 *       (AFTER density()+force_update_hmax(), BEFORE cellcorrections;
 *        this is where the corridor's shared symmetric gas CSR will be
 *        built — currently still built inside hydro_gradient_calc and
 *        rebuilt by cellcorrections, both to be hoisted here)
 *   cellcorrections_calc()                     [WAVE 5 corridor consumer]
 *   dynamic_diff_vel_calc / rt_source_injection / opacity_interp /
 *     compute_stellar_feedback                 [intervening; dirty ghost
 *                                               field values but not
 *                                               topology — see §1.5]
 *   hydro_gradient_calc()                      [WAVE 5 corridor consumer]
 *   mg_gradient_correction_calc()              [MHD_MODIFIED_GRADIENT]
 *   hydro_force()                              [WAVE 5 corridor consumer
 *                                               — closes the corridor]
 *   ---- CORRIDOR EXIT: gizmo_hydro_corridor_end() ----
 *
 * This file owns the step-scoped mode state, env-var decision logic, and
 * (later commits) the shared CSR build/refresh lifecycle + topology
 * guardrails. In commit 4 the API is STATE+LIFECYCLE infra only; no
 * consumer reads gizmo_hydro_corridor_get_mode() yet. The corridor cannot
 * be truly closed until commit 8 (HydroForceSpec) — see the design doc.
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
 *   1. env GIZMO_HYDRO_CORRIDOR_FORCE_MODE = A | B   (corridor-wide override)
 *   2. env GIZMO_NLR_FORCE_MODE            = A | B   (global runner override)
 *   3. adaptive vs env GIZMO_NLR_MODEB_THRESHOLD_SUM (default if unset = a
 *      sensible large-N threshold; see implementation)
 *
 * Per-loop overrides (GIZMO_GRADIENTS_FORCE_MODE, etc.) are NOT consulted
 * here — they break corridor coherence and will be loudly diagnosed at the
 * consumer site once consumers read corridor mode (commit 5+). */
void gizmo_hydro_corridor_decide_mode(void);

/* Read the current corridor mode. Returns UNSET when called outside the
 * corridor span (before decide_mode or after end). */
GizmoHydroCorridorMode gizmo_hydro_corridor_get_mode(void);

/* Tear down the corridor at the end of hydro work for this step. Resets
 * the mode to UNSET so any unexpected read between steps is detectable. */
void gizmo_hydro_corridor_end(void);

#endif /* HYDRO_CORRIDOR_H */
