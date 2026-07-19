#ifndef GIZMO_WAKEUP_SIDECAR_H
#define GIZMO_WAKEUP_SIDECAR_H

/* Wakeup dirty sidecar — acceleration index for process_wake_ups().
 *
 * WakeupDirty[i] is a SUPERSET index over the authoritative P[i].wakeup: when
 * WakeupDirtyValid, it is set for every i<NumPart with P[i].wakeup!=0. False
 * positives are allowed (the scan self-clears them); a missed SET is the only
 * correctness hazard. P[i].wakeup stays the sole authoritative wakeup state.
 * The scan reads this contiguous byte array instead of striding fat P[].
 *
 * wakeup_sidecar_mark(i) sets the bit — call it wherever a host write or a
 * whole-struct copy could make P[i].wakeup nonzero (over-marking on a copy of a
 * zero-wakeup particle is a harmless false positive). Device wakeup kernels
 * mark via the base pointer in their DeviceContext. Index remaps (domain decomp
 * / rearrange / subfind) call wakeup_sidecar_invalidate(); the next
 * process_wake_ups rebuilds from P[]. */

#include "../declarations/allvars.h"   /* WakeupDirty, WakeupDirtyValid */

/* Full authoritative rebuild from P[] over [0,NumPart); sets WakeupDirtyValid.
 * Defined in core/timestep.cc. */
void wakeup_sidecar_rebuild(void);

static inline void wakeup_sidecar_mark(int i) { if(WakeupDirty) { WakeupDirty[i] = 1; } }
static inline void wakeup_sidecar_invalidate(void) { WakeupDirtyValid = 0; }

#endif /* GIZMO_WAKEUP_SIDECAR_H */
