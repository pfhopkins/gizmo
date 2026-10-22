#ifndef GIZMO_SYSTEM_TIMER_GIZMO_H
#define GIZMO_SYSTEM_TIMER_GIZMO_H

/* Main-thread application lifecycle. Initialize and report are collective;
 * shutdown is local and must precede Kokkos finalization. */
void gizmo_timers_initialize();
void gizmo_timers_report(const char *label, long long step, bool force = false);
void gizmo_timers_shutdown();

#endif
