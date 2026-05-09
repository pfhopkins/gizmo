/* declarations/lifecycle_counters.h — global API-entry counters for the
 * tiny-N hard-corridor guard in mesh/neighbor_loop_runner::run_neighbor_loop.
 *
 * Each counter is incremented at the entry of a low-level GIZMO API that
 * Mode B paths must NOT call: a global drift, a ghost import, or a GPU
 * arena acquire. The runner snapshots all three at Mode B entry and HARD
 * ABORTS at exit if any has advanced.
 *
 * Counter semantics: counts API ENTRIES, not work done. An early-out (e.g.
 * NTask <= 1, no-ghost case, time1 == time0 short-circuit) still
 * increments, by design — Mode B violates the corridor by entering the
 * API at all, regardless of whether the API would have done any work.
 *
 * All three counters are defined in declarations/lifecycle_counters.cc
 * (always linked via CORE_OBJS) so non-GPU builds and any future
 * configuration that conditionally omits an owner TU still link cleanly.
 *
 * No upward dependency from owner TUs to the runner: lower-level systems
 * include this neutral header, never the runner header.
 */
#ifndef GIZMO_LIFECYCLE_COUNTERS_H
#define GIZMO_LIFECYCLE_COUNTERS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

extern uint64_t g_global_drift_counter;       /* ++ at top of move_particles            */
extern uint64_t g_ghost_import_counter;       /* ++ at top of ghost_exchange_impl       */
extern uint64_t g_gpu_arena_acquire_counter;  /* ++ at top of gpu_particles_arena_acquire */

#ifdef __cplusplus
}
#endif

#endif /* GIZMO_LIFECYCLE_COUNTERS_H */
