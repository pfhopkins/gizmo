/* ghost_exchange_typed.h — caller-typed ghost exchange API design (DRAFT).
 *
 * Status: DESIGN SKETCH, no implementation yet. Pending hmax-outlier diagnostic
 * (commit 2026-05-04 evening) to confirm the type-pollution model before we
 * commit the structural rewrite.
 *
 * Problem the API solves:
 *   The current ghost_exchange() builds tiles over every massive particle and
 *   uses every active particle as a request driver. For a hydro density call
 *   with 1 active gas particle, this means:
 *     - tile hmax can be inflated by a co-tile particle of another Type's
 *       KernelRadius (often 100-600× the gas h)
 *     - search_r = max(active_hmax, remote_tile_hmax) becomes huge
 *     - import 50k+ ghosts of all types when ~80 gas neighbors are wanted
 *   Empirically (fire_m11i 2-rank, 1 active gas): 48 680 ghosts received
 *   for a step that needs <100 actual neighbors.
 *
 * Design (one struct, one entry point, ~6 caller wrappers):
 *
 *   struct ghost_exchange_spec_t {
 *       uint32_t request_type_bitmask;   // which Types drive the import
 *       uint32_t supply_type_bitmask;    // which Types fill the pool/tiles
 *       double  (*radius_fn)(int j);     // per-particle effective h
 *       double   safety_factor;          // multiplier on search_r
 *       const char *caller_name;         // for diagnostic prints
 *   };
 *
 *   void ghost_exchange_typed(const ghost_exchange_spec_t *spec);
 *
 * Internal changes from the current monolithic ghost_exchange():
 *   1. Pool building (line ~139): instead of `if(P[i].Mass > 0)`, gate by
 *      `((1u << P[i].Type) & spec->supply_type_bitmask) && P[i].Mass > 0`.
 *      This removes non-supply-Type contamination of tile hmax for hydro
 *      callers in one place.
 *   2. effective_ghost_radius lambda (line ~179): replaced by the caller's
 *      radius_fn. Hydro callers pass a gas-only function that returns
 *      P[j].KernelRadius (no AGS_KernelRadius / KernelRadiusDM mixing).
 *      Sink callers pass a function that returns P[j].KernelRadius for
 *      sinks, ignored for non-sink supply (or supply-only-gas).
 *   3. is_active flag (line ~196): `is_active[i_act] = 1` only if
 *      `(1u << P[i_act].Type) & spec->request_type_bitmask`. Hydro callers
 *      pass mask = 1 (Type 0/cells only); other callers pass the Type mask
 *      appropriate to their physics.
 *
 * Caller routing table:
 *
 *   gizmo_density_prep_ghosts          { req=Type0  sup=Type0  r=KernelRadius }
 *   gizmo_density_redo_ghosts_if_needed{ req=Type0  sup=Type0  r=KernelRadius }
 *   gizmo_gradients_prep_symlist       { req=Type0  sup=Type0  r=KernelRadius }
 *   gizmo_gradients_refresh_symlist    { req=Type0  sup=Type0  r=KernelRadius }
 *   sink_environment_loop              { req=Type5  sup=caller mask r=KernelRadius_sink }
 *   sink_feed_loop                     { req=Type5  sup=caller mask r=KernelRadius_sink }
 *   sink_swallow_and_kick_loop         { req=Type5  sup=caller mask r=KernelRadius_sink }
 *   mech_fb (mechfb_loop)              { req=caller actives sup=Type0 r=source_h }
 *   thermal_fb_evaluate_gpu            { req=caller actives sup=Type0 r=source_h }
 *   HII_heating_singledomain           { req=caller actives sup=Type0 r=R_HII or KernelRadius }
 *   radiation_pressure_winds (gpu)     { req=caller actives sup=Type0 r=source_h }
 *
 * Backwards-compat:
 *   The bare `ghost_exchange(double safety)` becomes a wrapper that calls
 *   ghost_exchange_typed with all-types request and supply masks and the
 *   current effective_ghost_radius lambda. Nothing in the existing call
 *   tree breaks; new typed callers opt in one by one.
 *
 * Type bitmasks (matching P[].Type):
 *   #define GHOST_TYPE_0      (1u << 0)
 *   #define GHOST_TYPE_1      (1u << 1)
 *   #define GHOST_TYPE_2      (1u << 2)
 *   #define GHOST_TYPE_3      (1u << 3)
 *   #define GHOST_TYPE_4      (1u << 4)
 *   #define GHOST_TYPE_5      (1u << 5)
 *   #define GHOST_TYPE_ALL    0x3F
 *
 * Migration order (after diagnostic confirms):
 *   1. Refactor ghost_exchange() body into ghost_exchange_typed() taking
 *      a spec; bare ghost_exchange() becomes the all-types wrapper. No
 *      semantic change.
 *   2. Add typed wrappers: ghost_exchange_for_hydro_density(),
 *      ghost_exchange_for_sinks(), ghost_exchange_for_stellar_fb().
 *   3. Replace one caller at a time in lifecycle.h + galaxy_sf/* + sinks/*.
 *   4. Each replacement validated via single-rank smoke + 2-rank tiny-N
 *      perf check (target: hydro ghost count drops 50000→<1000).
 *   5. Once all hydro callers migrated, the all-types path is only used
 *      by sink/feedback callers — but they should also be migrated.
 *
 * Cost model:
 *   - Pool size shrinks from N_total to N_gas for hydro callers (~50%
 *     for fire_m11i where gas:non-gas is ~1:1).
 *   - Tile hmax for hydro pool drops from 583 (DM-poisoned) to ~kpc
 *     (real gas h). search_r drops similarly. ghost count drops
 *     proportional to (search_r)^3 in the worst case.
 *   - Conservative estimate: 10-100x reduction in ghost imports for
 *     hydro tiny-N. Reduces meta_allgather by ~50% (smaller tile_meta
 *     payload), pack+mpi by ~10-100x.
 *
 * Open questions:
 *   - Should pool build be cached across same-type calls within a step?
 *     (e.g. density+gradient+hydro_force all use same gas pool — worth
 *     not rebuilding tile_meta 3x per step?)
 *   - For sink callers, should "supply_type_bitmask = caller mask" actually
 *     build TWO separate tile lists, one per type, with type-specific h?
 *     Current single-list approach risks contaminating sink pool with
 *     gas h, but gas h ~ sink h numerically (both kpc-scale), so probably
 *     fine in one list.
 */
#ifndef GHOST_EXCHANGE_TYPED_H
#define GHOST_EXCHANGE_TYPED_H

/* This file intentionally has no declarations — design sketch only. The
 * struct + entry point land in ghost_exchange.cc once the diagnostic
 * confirms the type-pollution model. */

#endif /* GHOST_EXCHANGE_TYPED_H */
