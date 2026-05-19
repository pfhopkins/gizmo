/* hydro/hydro_corridor.cc — hydro corridor mode-decision + lifecycle.
 *
 * See hydro/hydro_corridor.h for the API contract and the
 * OPEN_3d_hydro_corridor_design.md design doc for the full corridor
 * sequencing rationale. Commit 4 (this file) lands STATE+LIFECYCLE only;
 * no consumer reads gizmo_hydro_corridor_get_mode() yet.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "hydro_corridor.h"

/* Default adaptive threshold (sum of global_num_active_gas) below which
 * the corridor picks Mode B. Matches the project-wide
 * GIZMO_NLR_MODEB_THRESHOLD_SUM convention used by the runner's per-loop
 * threshold; centralized here so a future tuning lives in one place. */
static constexpr int CORRIDOR_DEFAULT_MODEB_THRESHOLD_SUM = 1000;

/* Step-scoped mode state. Reset to UNSET by gizmo_hydro_corridor_end().
 * File-static so consumers must go through the accessor; no extern reads.
 *
 * Lifetime invariant: UNSET -> {MODE_A, MODE_B} via decide_mode at the top
 * of compute_hydro_densities_and_forces, then back to UNSET via end at the
 * bottom of the same function. Reading UNSET from a consumer is a bug
 * (consumer placed outside the corridor span). */
static GizmoHydroCorridorMode g_corridor_mode = GizmoHydroCorridorMode::UNSET;

/* Helper: parse an "A" / "B" env var into a corridor mode. Returns
 * UNSET when the env var is missing or empty (caller treats this as
 * "no override at this priority level"). Case-insensitive on the first
 * char so "a" and "b" also work.
 *
 * Invalid set-but-nonsense values (e.g. "banana") FATAL via endrun.
 * Silent fallback to adaptive would create exactly the testing-confusion
 * class that ruins mode validation runs — a user setting the env var to
 * the wrong value must learn loudly. */
static GizmoHydroCorridorMode
parse_force_mode_env(const char *env_name)
{
    const char *v = getenv(env_name);
    if(v == nullptr || v[0] == '\0') return GizmoHydroCorridorMode::UNSET;
    /* Accept ONLY the single-char strings "A" or "B" (case-insensitive).
     * "banana" starts with 'b' but is not a valid override; reject. */
    if(v[1] == '\0') {
        char c = v[0];
        if(c == 'A' || c == 'a') return GizmoHydroCorridorMode::MODE_A;
        if(c == 'B' || c == 'b') return GizmoHydroCorridorMode::MODE_B;
    }
    if(ThisTask == 0) {
        printf("FATAL: %s='%s' not recognized (expected 'A' or 'B'). "
               "Unset to fall through to the next priority level (adaptive).\n",
               env_name, v);
        fflush(stdout);
    }
    endrun(7310);
    return GizmoHydroCorridorMode::UNSET; /* unreachable; silences compiler */
}

static int
parse_threshold_env(const char *env_name, int fallback)
{
    const char *v = getenv(env_name);
    if(v == nullptr || v[0] == '\0') return fallback;
    long n = strtol(v, nullptr, 10);
    if(n <= 0) return fallback;
    return (int)n;
}

/* Global active-gas count — collective Allreduce so every rank computes
 * the same mode (corridor coherence requires this). Caller must be inside
 * MPI_COMM_WORLD scope. Cheap (one int Allreduce). */
static int
global_active_gas_count(void)
{
    int local_active_gas = 0;
    for(int ii : ActiveParticleList) {
        if(P[ii].Type == 0 && P[ii].Mass > 0) local_active_gas++;
    }
    int global_active_gas = 0;
    if(NTask > 1) {
        MPI_Allreduce(&local_active_gas, &global_active_gas, 1,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    } else {
        global_active_gas = local_active_gas;
    }
    return global_active_gas;
}

void gizmo_hydro_corridor_decide_mode(void)
{
    GizmoHydroCorridorMode mode = GizmoHydroCorridorMode::UNSET;
    const char *source = "(unset)";

    /* Priority 1: corridor-wide override */
    mode = parse_force_mode_env("GIZMO_HYDRO_CORRIDOR_FORCE_MODE");
    if(mode != GizmoHydroCorridorMode::UNSET) {
        source = "GIZMO_HYDRO_CORRIDOR_FORCE_MODE";
    } else {
        /* Priority 2: global runner override */
        mode = parse_force_mode_env("GIZMO_NLR_FORCE_MODE");
        if(mode != GizmoHydroCorridorMode::UNSET) {
            source = "GIZMO_NLR_FORCE_MODE";
        } else {
            /* Priority 3: adaptive on global active-gas count */
            int global_active_gas = global_active_gas_count();
            int threshold = parse_threshold_env("GIZMO_NLR_MODEB_THRESHOLD_SUM",
                                                CORRIDOR_DEFAULT_MODEB_THRESHOLD_SUM);
            mode = (global_active_gas <= threshold)
                       ? GizmoHydroCorridorMode::MODE_B
                       : GizmoHydroCorridorMode::MODE_A;
            source = "adaptive";
        }
    }

    g_corridor_mode = mode;

    if(ThisTask == 0 && gizmo_verbose_diag()) {
        const char *mode_str = (mode == GizmoHydroCorridorMode::MODE_A) ? "MODE_A"
                             : (mode == GizmoHydroCorridorMode::MODE_B) ? "MODE_B"
                             : "UNSET";
        printf("[CORRIDOR step=%d] mode=%s source=%s\n",
               (int)All.NumCurrentTiStep, mode_str, source);
        fflush(stdout);
    }
}

GizmoHydroCorridorMode gizmo_hydro_corridor_get_mode(void)
{
    return g_corridor_mode;
}

void gizmo_hydro_corridor_end(void)
{
    g_corridor_mode = GizmoHydroCorridorMode::UNSET;
}
