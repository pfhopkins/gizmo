/* step_phases.cc — see step_phases.h */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <utility>
#include <mpi.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "step_phases.h"

namespace {

/* Enabled flag, cached on first lookup. */
int g_enabled_cached = -1;

/* Per-step accumulators, in registration order. */
std::vector<std::pair<std::string, double>> g_buckets;

/* Wallclock at start of current step (set by gizmo_step_phase_step_start). */
double g_step_start_t = 0.0;

int find_bucket(const char *name)
{
    for(size_t i = 0; i < g_buckets.size(); i++) {
        if(g_buckets[i].first == name) return (int)i;
    }
    return -1;
}

} /* anonymous namespace */

extern "C" int gizmo_verbose_diag(void)
{
    if(g_enabled_cached < 0) {
        const char *q = getenv("GIZMO_VERBOSE_DIAG");
        g_enabled_cached = (q && q[0] && q[0] != '0') ? 1 : 0;
    }
    return g_enabled_cached;
}

/* Backwards-compat shim — same gate now. */
extern "C" int gizmo_step_phase_enabled(void) { return gizmo_verbose_diag(); }

extern "C" void gizmo_step_phase_record(const char *name, double dt)
{
    if(!gizmo_step_phase_enabled()) return;
    if(!name) return;
    int idx = find_bucket(name);
    if(idx < 0) {
        g_buckets.emplace_back(std::string(name), dt);
    } else {
        g_buckets[idx].second += dt;
    }
}

extern "C" void gizmo_step_phase_step_start(void)
{
    if(!gizmo_step_phase_enabled()) return;
    g_step_start_t = my_second();
}

extern "C" void gizmo_step_phase_dump(int step)
{
    if(!gizmo_step_phase_enabled()) return;
    if(ThisTask != 0) {g_buckets.clear(); return;}
    double wall = (g_step_start_t > 0) ? (my_second() - g_step_start_t) : 0.0;
    double bucketsum = 0;
    for(size_t i = 0; i < g_buckets.size(); i++) bucketsum += g_buckets[i].second;
    /* Emit step+1 to match the balance.txt sync-point label. (Old behavior
     * printed the local counter which was off-by-one against balance.txt;
     * see feedback_gpu_wrapper_disciplines.md.) */
    printf("[STEP_PHASES step=%d wall=%.3fs bucketsum=%.3fs", step + 1, wall, bucketsum);
    for(size_t i = 0; i < g_buckets.size(); i++) {
        printf(" %s=%.3f", g_buckets[i].first.c_str(), g_buckets[i].second);
    }
    printf("]\n");
    fflush(stdout);
    g_buckets.clear();
    g_step_start_t = 0.0;
}
