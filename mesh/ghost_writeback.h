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

/* ThermalFB variant: snapshot-based reverse communication for
 * addthermalFB j-particle writes (Mass, Density, dp, IE, Metallicity, etc.).
 * Call zero before the GPU kernel; writeback after the scatter memcpy. */
#ifdef GALSF_FB_THERMAL
void ghost_writeback_zero_thermalfb(void);
void ghost_writeback_thermalfb(void);
#endif

/* SinkFeed variant: snapshot-based reverse communication for sink_feed_evaluate
 * j-particle writes (SwallowID, and optionally Injected_Sink_Energy).
 * Call zero before the GPU kernel; writeback after the scatter memcpy. */
#ifdef SINK_PARTICLES
void ghost_writeback_zero_sinkfeed(void);
void ghost_writeback_sinkfeed(void);
#endif

/* MechFB variant: reverse communication for mechanical_fb GPU kernel per-gas
 * MechFBGasDelta accumulator. Unlike the snapshot-based patterns above,
 * the MechFBGasDelta buffer is allocated fresh and zeroed each top-level
 * invocation, so the FULL ghost entry IS the delta — no snapshot needed.
 * Input: ghost_full_buf[num_local .. num_local+num_ghost) holds ghost deltas.
 *        home_buf[0..n_gas) is the home-rank destination (already contains
 *        home-cell deltas from the kernel). MPI reduces ghost→home and
 *        accumulates into home_buf[home_index]. */
#ifdef GALSF_FB_MECHANICAL
struct MechFBGasDelta;
void ghost_writeback_mechfb(struct MechFBGasDelta *ghost_full_buf,
                             struct MechFBGasDelta *home_buf,
                             int n_gas);
#endif

/* Grain backreaction (B7a): snapshot-based reverse communication for
 * grain_backrx_evaluate j-particle writes (Vel, VelPred, dp, Grain_AccelTimeMin[min-update]). */
#if defined(GRAIN_FLUID) && defined(GRAIN_BACKREACTION)
void ghost_writeback_zero_grainbackrx(void);
void ghost_writeback_grainbackrx(void);
#endif

/* SinkSwallow (D1): snapshot-based reverse communication for
 * sink_swallow_and_kick_evaluate j-particle writes. Mass / Vel / dp / VelPred /
 * InternalEnergy(Pred) / MassTrue (additive), B / BPred (additive), plus
 * sink-merger Sink_Mass/Sink_Mdot/Sink_Mass_Reservoir zeroing. */
#ifdef SINK_PARTICLES
void ghost_writeback_zero_sinkswallow(void);
void ghost_writeback_sinkswallow(void);
#endif

/* RadFBRP variant: snapshot-based reverse communication for
 * radiation_pressure_winds GPU kernel j-particle writes (Vel, VelPred, dp). */
#ifdef GALSF_FB_FIRE_RT_LOCALRP
void ghost_writeback_zero_radfbrp(void);
void ghost_writeback_radfbrp(void);
#endif

/* RT source injection variant: snapshot-based reverse communication for
 * rt_source_injection GPU kernel j-gas writes (radiation energy/source fields,
 * optional momentum kicks, and optional flux/intensity fields). */
#ifdef RT_SOURCE_INJECTION
void ghost_writeback_zero_rtsrcinjection(void);
void ghost_writeback_rtsrcinjection(void);
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
