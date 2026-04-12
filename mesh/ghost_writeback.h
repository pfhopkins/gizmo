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
