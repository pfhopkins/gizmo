/* Mode B local neighbor walker — SKELETON.
 *
 * Real implementation lands Day 1 of the Phase 0 plan. This file exists so
 * the public surface (mode_b_local_walker.h) compiles into the build and
 * downstream callers can be drafted without waiting on the walker body.
 *
 * Until the walker is implemented:
 *   - mode_b_enabled() returns 0 unless GIZMO_MODE_B_DENSITY=1 in env.
 *   - mode_b_local_neighbor_walk() returns 0 candidates and emits a warn-
 *     once stderr line if Mode B is on. This makes accidental wiring loud.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mode_b_local_walker.h"

int mode_b_enabled(void)
{
    static const char *env = NULL;
    static int cached = -1;
    if(cached < 0) {
        env = getenv("GIZMO_MODE_B_DENSITY");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached;
}

int mode_b_local_neighbor_walk(const double pos[3],
                               double h_q,
                               unsigned int type_mask,
                               int search_mode,
                               int *out_candidates,
                               int out_capacity)
{
    (void)pos; (void)h_q; (void)type_mask; (void)search_mode;
    (void)out_candidates; (void)out_capacity;

    if(mode_b_enabled()) {
        static int warned = 0;
        if(!warned) {
            fprintf(stderr,
                    "[mode_b] WARN: mode_b_local_neighbor_walk called but skeleton "
                    "has no implementation yet. Returning 0 candidates. Wire the "
                    "real walker before testing Day 1 acceptance.\n");
            fflush(stderr);
            warned = 1;
        }
    }
    return 0;
}
