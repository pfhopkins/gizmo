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

int find_bucket(const char *name)
{
    for(size_t i = 0; i < g_buckets.size(); i++) {
        if(g_buckets[i].first == name) return (int)i;
    }
    return -1;
}

} /* anonymous namespace */

extern "C" int gizmo_step_phase_enabled(void)
{
    if(g_enabled_cached < 0) {
        const char *q = getenv("GIZMO_STEP_PHASES");
        g_enabled_cached = (q && q[0] && q[0] != '0') ? 1 : 0;
    }
    return g_enabled_cached;
}

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

extern "C" void gizmo_step_phase_dump(int step)
{
    if(!gizmo_step_phase_enabled()) return;
    if(ThisTask != 0) {g_buckets.clear(); return;}
    double total = 0;
    for(size_t i = 0; i < g_buckets.size(); i++) total += g_buckets[i].second;
    printf("[STEP_PHASES step=%d total=%.3fs", step, total);
    for(size_t i = 0; i < g_buckets.size(); i++) {
        printf(" %s=%.3f", g_buckets[i].first.c_str(), g_buckets[i].second);
    }
    printf("]\n");
    fflush(stdout);
    g_buckets.clear();
}
