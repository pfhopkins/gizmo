/* State-hash harness — minimal working implementation.
 *
 * Triggered on GIZMO_STATE_HASH=1. Emits one STATE_HASH line per call,
 * per rank. Bit-exact xor-fold over double bit representations.
 *
 * Cost when off: one cached-bool branch. Cost when on: O(NumPart) per call
 * with simple xor-fold + multiply — call site frequency dominates.
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "state_hash.h"

extern std::vector<int> ActiveParticleList;

int state_hash_enabled(void)
{
    static int cached = -1;
    if(cached < 0) {
        const char *env = getenv("GIZMO_STATE_HASH");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached;
}

static inline uint64_t mix64(uint64_t h, uint64_t v)
{
    /* SplitMix-style mix; cheap and well-distributed. */
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    return h;
}

static inline uint64_t bits_of(double x)
{
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}

void state_hash_record(const char *label, long long step)
{
    if(!state_hash_enabled()) return;

    static long long auto_step = 0;
    if(step < 0) step = ++auto_step;

    uint64_t h_pos = 0xcbf29ce484222325ULL;     /* FNV offset */
    uint64_t h_h   = 0xcbf29ce484222325ULL;
    uint64_t h_act = 0xcbf29ce484222325ULL;

    for(int i = 0; i < NumPart; i++) {
        h_pos = mix64(h_pos, bits_of(P[i].Pos[0]));
        h_pos = mix64(h_pos, bits_of(P[i].Pos[1]));
        h_pos = mix64(h_pos, bits_of(P[i].Pos[2]));
        h_pos = mix64(h_pos, bits_of((double)P[i].Type));
        h_pos = mix64(h_pos, bits_of(P[i].Mass));
        h_h   = mix64(h_h,   bits_of(P[i].KernelRadius));
    }
    for(size_t k = 0; k < ActiveParticleList.size(); k++) {
        h_act = mix64(h_act, (uint64_t)ActiveParticleList[k]);
    }

    uint64_t h_total = mix64(mix64(mix64(h_pos, h_h), h_act),
                             ((uint64_t)NumPart << 32) ^ (uint64_t)All.Ti_Current);

    printf("STATE_HASH rank=%d step=%lld label=%s NumPart=%d NumGas=%d Ti=%lld "
           "h_pos=0x%016llx h_h=0x%016llx h_active=0x%016llx h_total=0x%016llx\n",
           ThisTask, step, label ? label : "?", NumPart, N_gas, (long long)All.Ti_Current,
           (unsigned long long)h_pos, (unsigned long long)h_h,
           (unsigned long long)h_act, (unsigned long long)h_total);
    fflush(stdout);
}
