#ifndef GIZMO_SYSTEM_TIMER_GIZMO_H
#define GIZMO_SYSTEM_TIMER_GIZMO_H

/* Main-thread application lifecycle. Initialize and report are collective;
 * shutdown is local and must precede Kokkos finalization. */
void gizmo_timers_initialize();
/* Local hooks: begin a fresh step measurement after CPU-log reporting and
 * capture workload after repartition/active-list construction, before forces. */
void gizmo_timers_step_begin();
void gizmo_timers_step_context();
void gizmo_timers_report(const char *label, long long step, bool force = false);
void gizmo_timers_shutdown();

#endif
