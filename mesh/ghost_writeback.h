/* ghost_writeback.h — reverse communication of ghost particle modifications.
 *
 * After a GPU neighbor loop writes to ghost j-particles (via Kokkos atomics),
 * ghost_writeback communicates those modifications back to the home MPI ranks.
 *
 * Usage pattern for each loop that modifies j-particles:
 *   1. ghost_writeback_zero_hydro()       — zero accumulator fields on ghosts
 *   2. [run GPU kernel with atomic j-writes]
 *   3. ghost_writeback_hydro()            — scan, pack, reverse MPI, apply
 *
 * The ghost provenance map (home rank + index per ghost) is built during
 * ghost_exchange() and freed in ghost_exchange_cleanup(). Accessors are
 * in ghost_exchange.cc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GHOST_WRITEBACK_H
#define GHOST_WRITEBACK_H

/* Zero the hydro accumulator fields (dMass, wakeup) on all ghost particles.
 * Call before the hydro force loop so post-kernel values ARE the deltas. */
void ghost_writeback_zero_hydro(void);

/* Scan ghost particles for nonzero hydro modifications (dMass, wakeup),
 * pack into compact delta structs, reverse MPI_Alltoallv to home ranks,
 * and apply the deltas. Call after the hydro force loop, before cleanup. */
void ghost_writeback_hydro(void);

/* Wakeup-only variant, for use around GPU neighbor-list kernels that
 * atomically write P[j].wakeup but no other j-side state (e.g. the AGS
 * density kernel). Usage pattern:
 *   ghost_writeback_zero_wakeup();
 *   [run GPU kernel with atomic P[j].wakeup writes]
 *   ghost_writeback_wakeup();
 * Safe to call in addition to ghost_writeback_hydro on the same timestep —
 * each call zeroes + reverse-communicates independently. */
void ghost_writeback_zero_wakeup(void);
void ghost_writeback_wakeup(void);

/* AGSForce variant: reverse-communicates the full AGS j-side delta set
 *   (Vel[3], dp[3], NInteractions, wakeup) — everything the CPU tree-walk
 * wrote into neighbors inside the AGSForce loop. Zero is called before
 * the GPU kernel; the scatter version runs after, before cleanup. Only
 * the fields that exist in the current build are packed (the dp/
 * NInteractions fields live under #ifdef DM_SIDM). */
void ghost_writeback_zero_agsforce(void);
void ghost_writeback_agsforce(void);

/* SwallowTime variant: reverse-communicates minimum SwallowTime written to
 * ghost particles by the sink_environment GPU kernel (under
 * SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS).  Zero snapshots the
 * pre-kernel values; swallowtime sends any ghost whose SwallowTime decreased
 * back to its home rank and applies a min there.
 * Only compiled when SINGLE_STAR_SINK_DYNAMICS is enabled (field only exists then). */
#ifdef SINGLE_STAR_SINK_DYNAMICS
void ghost_writeback_zero_swallowtime(void);
void ghost_writeback_swallowtime(void);
#endif

/* Accessors from ghost_exchange.cc */
int ghost_get_num_ghosts(void);
int ghost_get_num_local(void);
int *ghost_get_home_rank(void);
int *ghost_get_home_index(void);
int *ghost_get_wb_recv_count(void);
int *ghost_get_wb_recv_disp(void);
int *ghost_get_wb_send_count(void);
int *ghost_get_wb_send_disp(void);

#endif /* GHOST_WRITEBACK_H */
