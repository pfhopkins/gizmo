/* solids/grain_physics_loop_api.h — host-clean API surface between
 * solids/grain_physics.cc (host orchestration) and
 * solids/grain_physics_loop.cc (runner-template Spec hooks + toplevels).
 *
 * Kokkos- and runner-template-free: the host TU includes this header and
 * never sees the full Spec definitions. The three Specs (GrainBackrxSpec,
 * GrainRTGasSpec, GrainRTGrainSpec) instantiate run_neighbor_loop<Spec>
 * inside the GPU TU (solids/grain_physics_loop.cc via
 * mesh/neighbor_loop_runner.cc explicit instantiations).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GRAIN_PHYSICS_LOOP_API_H
#define GRAIN_PHYSICS_LOOP_API_H

/* DO_FLUID_ALTSPECIES_DRAG_CALCULATION widens the gate from GRAIN_FLUID-only
 * to also cover HYDRO_MULTIFLUID_DUST_DRAG / HYDRO_MULTIFLUID_IONNEUTRAL —
 * those builds auto-define GRAIN_BACKREACTION (precompiler_logic.h) and
 * apply_grain_dragforce() in grain_physics.cc calls grain_backrx_calc(). */
#ifdef DO_FLUID_ALTSPECIES_DRAG_CALCULATION

#if defined(GRAIN_BACKREACTION)
/* Replaces grain_backrx_evaluate_gpu + ghost_writeback_{zero_,}grainbackrx.
 * Called from apply_grain_dragforce() in grain_physics.cc after the
 * Grain_AccelTimeMin reset loop. */
void grain_backrx_calc(void);
#endif

#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
/* Replaces interpolate_fluxes_opacities_gasgrains_evaluate_gpu.
 * Called from interpolate_fluxes_opacities_gasgrains() in grain_physics.cc.
 * Zeros J_dust_cell for active gas under MHD_BATTERY_MECHANISMS & 8 before
 * the GrainRTGasSpec run. */
void grain_rt_opacity_calc(void);
#endif

#endif /* DO_FLUID_ALTSPECIES_DRAG_CALCULATION */

#endif /* GRAIN_PHYSICS_LOOP_API_H */
