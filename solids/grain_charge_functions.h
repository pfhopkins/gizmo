/* grain_charge_functions.h -- canonical grain charge model (Draine & Sutin
 * 1987 / Weingartner & Draine 2001), header-only inlines.
 *
 * Provides:
 *   grain_charge_equilibrium(...)  -- Z_eq(a_grain, n_e, T, J_FUV, zeta_CR):
 *                                     equilibrium charge from balance of
 *                                     collisional electron capture, photo-
 *                                     electric ejection, and CR-secondary
 *                                     charging. Used both as initial
 *                                     condition for the explicitly-evolved
 *                                     Grain_Charge field and as the per-cell
 *                                     analytic charge in the TVA path of the
 *                                     dust battery.
 *
 *   grain_charge_relax_implicit(...) -- single-step time-implicit relaxation
 *                                       of Grain_Charge toward equilibrium,
 *                                       used by solids/grain_charge_gpu.cc.
 *
 * REPLACES the oversimplified inline Z_grain expression that previously
 * lived inside solids/grain_drag_gpu.cc (used for GRAIN_LORENTZFORCE). Both
 * the drag (Lorentz force) and the dust battery now read the per-grain
 * Grain_Charge field populated by grain_charge_gpu().
 *
 * CR-driven secondary charging is proportional to the existing
 * self-contained CR ionization rate helper (see cooling/CR module);
 * NOT a free parameter or a parallel CR model.
 *
 * Header-only KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GRAIN_CHARGE_FUNCTIONS_H
#define GRAIN_CHARGE_FUNCTIONS_H

#if defined(GRAIN_LORENTZFORCE) || (defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8))

/* TODO: grain_charge_equilibrium(grain_size, n_e, T, J_FUV, zeta_CR) ->
   Z_eq (signed integer-equivalent charge number). Implements
   Draine & Sutin 1987 / Weingartner & Draine 2001 equilibrium balance
   with FUV photoelectric source (live RT field if available, else ISRF
   default) and CR-rate-proportional secondary charging (zeta_CR from
   existing cooling/CR helper). */

/* TODO: grain_charge_relax_implicit(Z_old, Z_eq, t_charge, dt) ->
   Z_new via implicit relaxation Z_new = (Z_old + dt/t_charge * Z_eq) /
   (1 + dt/t_charge). t_charge ~ collisional charging time; fast in dense
   gas, slow in low-n environments. */

#endif

#endif /* GRAIN_CHARGE_FUNCTIONS_H */
