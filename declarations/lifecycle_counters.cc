/* declarations/lifecycle_counters.cc — single-TU definitions for the
 * lifecycle counters declared in lifecycle_counters.h. Owner TUs (predict,
 * ghost_exchange, gpu_particles_arena) only `extern` and increment.
 * Always-linked via CORE_OBJS to remove any config-dependent link issues. */

#include "lifecycle_counters.h"

extern "C" {
uint64_t g_global_drift_counter      = 0;
uint64_t g_ghost_import_counter      = 0;
uint64_t g_gpu_arena_acquire_counter = 0;
}
