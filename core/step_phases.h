/* step_phases.h — env-gated per-step phase wallclock recorder.
 *
 * Lightweight diagnostic that captures host-side wallclock between named
 * checkpoints across the per-step orchestration. Complements CPU_Step / cpu.txt
 * which only track coarse buckets and miss things like arena memcpy, scatter
 * loops, and host-side aggregation inside misc_hydro.
 *
 * Enabled by environment variable GIZMO_STEP_PHASES=1 (cached on first call).
 * When disabled, all calls are no-ops with negligible overhead (one branch).
 *
 * Usage:
 *   double t = my_second();
 *   foo();
 *   gizmo_step_phase_record("foo", my_second() - t);
 *   ...
 *   gizmo_step_phase_dump(step_number);   // at end of step on rank 0
 *
 * The macro STEP_PHASE_TIME(name, code) wraps the begin/run/record dance.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef STEP_PHASES_H
#define STEP_PHASES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if GIZMO_STEP_PHASES env var is set (cached). 0 otherwise. */
int gizmo_step_phase_enabled(void);

/* Record dt seconds against named bucket. Multiple records to same name
 * accumulate. No-op when disabled. Thread-safe at the step granularity
 * (per-step sequential calls on rank 0). */
void gizmo_step_phase_record(const char *name, double dt);

/* Print accumulated buckets in registration order (rank 0, env-gated)
 * with total/sum line, then clear all buckets for next step. */
void gizmo_step_phase_dump(int step);

#ifdef __cplusplus
}
#endif

/* Convenience macro: time a block of code and record its elapsed wallclock
 * against the named bucket. Single-evaluation-safe; no scope leaking. */
#define STEP_PHASE_TIME(name, code) do {                            \
    if(gizmo_step_phase_enabled()) {                                \
        double _spt = my_second();                                  \
        code;                                                       \
        gizmo_step_phase_record((name), my_second() - _spt);        \
    } else {                                                        \
        code;                                                       \
    }                                                               \
} while(0)

#endif /* STEP_PHASES_H */
