/* gpu_ghostdev_config.h -- where the hydro corridor's imported ghosts may be kept on the device (the contract is in
 * gpu_particles_arena.h, the transport in mesh/ghost_exchange.cc).  Include after allvars.h.
 *
 * GHOSTDEV_COMPILED: the hydro pass's ghost writeback is the wakeup maximum alone (no MFV mass flux) and the pool is
 *   never parked (no transport sub-steps).
 * GHOSTDEV_AUDITED_CONFIG: the corridor configurations whose host reads of the ghost range, and whose owner writes
 *   between the import and the hydro pass, were audited: hydro MFM without a module that adds a corridor pass, a
 *   refresh, or fields of its own to the gradients and hydro passes.  There the device route is on by default and
 *   the pre-hydro refresh may be trimmed to the fields those writes touch. */
#ifndef GIZMO_GPU_GHOSTDEV_CONFIG_H
#define GIZMO_GPU_GHOSTDEV_CONFIG_H

#if defined(GPU_PCIE_OPTIMIZATIONS) && !defined(HYDRO_MESHLESS_FINITE_VOLUME) && !defined(TRANSPORT_SUBCYCLE)
#define GHOSTDEV_COMPILED 1
#if defined(HYDRO_MESHLESS_FINITE_MASS) && !defined(MAGNETIC) && !defined(COSMIC_RAY_FLUID) && !defined(RADTRANSFER) && \
    !defined(HYDRO_VOLUME_CORRECTIONS) && !defined(TURB_DIFF_DYNAMIC) && !defined(TURB_DIFF_METALS) && \
    !defined(TURB_DIFF_ENERGY) && !defined(TURB_DIFF_VELOCITY) && !defined(CONDUCTION) && !defined(VISCOSITY) && \
    !defined(GALSF) && !defined(SINGLE_STAR_SINK_DYNAMICS) && !defined(BLACK_HOLES) && !defined(COOLING) && \
    !defined(METALS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(OUTPUT_SHOCK_MACH_NUMBER) && \
    !defined(ENERGY_ENTROPY_SWITCH_IS_ACTIVE) && !defined(MHD_CONSTRAINED_GRADIENT)
#define GHOSTDEV_AUDITED_CONFIG 1
#endif
#endif

#endif /* GIZMO_GPU_GHOSTDEV_CONFIG_H */
