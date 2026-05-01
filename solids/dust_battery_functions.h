/* dust_battery_functions.h -- dust-exclusive parts of the magnetic battery.
 *
 * Provides:
 *   battery_alpha_coeffs(...)      -- Soliman, Hopkins & Squire 2025 Eqs. 10-12
 *                                     (alpha_O, alpha_H, alpha_A from local
 *                                     plasma collision rates and densities).
 *   battery_E_dust_TVA(...)        -- E'_bat,d in the terminal-velocity
 *                                     approximation (SHS25 Eqs. 16-18; for
 *                                     simulations without explicit dust).
 *   battery_E_dust_explicit(...)   -- E'_bat,d using the actual dust current
 *                                     J_d summed from grain particles
 *                                     (SHS25 Eq. 9; bit 3 / GRAIN_FLUID).
 *   alpha_to_eta(...)              -- conversion eta = (c^2 / 4 pi) * alpha
 *                                     (textbook conversion between mobility
 *                                     coefficient and diffusivity).
 *
 * The dust-battery EMF E'_bat,d is built per-cell here, then exported to the
 * gradient pass for slope-limited gradient computation; the curl is taken in
 * hydro_toplevel.cc::out2particle_hydra and added as a cell-centered source
 * to DtB. The shared linear-algebra helper nonideal_mhd_assemble_bflux()
 * lives in hydro/nonideal_mhd_functions.h.
 *
 * All entries guarded by MHD_BATTERY_MECHANISMS & (4|8). Header-only,
 * KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Reference: Soliman, Hopkins & Squire 2025, ApJ 985, 55 (arXiv:2410.21461).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DUST_BATTERY_FUNCTIONS_H
#define DUST_BATTERY_FUNCTIONS_H

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (4|8))

#include "grain_charge_functions.h"

/* TODO: battery_alpha_coeffs(n_e, n_ion, n_d, T, B, ...) producing
   (alpha_O, alpha_H, alpha_A) per SHS25 Eqs. 10-12.
   alpha_O = (omega_en omega_+n) / (mu_e omega_+n + mu_+ omega_en)
   alpha_H = ... (Eq. 11)
   alpha_A = ... (Eq. 12)
   with mu_j = n_j q_j^2 / m_j and Omega_j = q_j B / (m_j c). */

/* TODO: battery_E_dust_TVA(...) per SHS25 Eqs. 16-18, picked by ionization
   fraction. Reads delta_a_dg (radiation-pressure differential), grain charge
   (from grain_charge_functions.h), drag stopping time, dust-to-gas ratio. */

/* TODO: battery_E_dust_explicit(J_d, alpha_O, alpha_H, alpha_A, bhat) per
   SHS25 Eq. 9. Calls nonideal_mhd_assemble_bflux() with eta = (c^2/4pi)*alpha. */

#endif /* MHD_BATTERY_MECHANISMS & (4|8) */

#endif /* DUST_BATTERY_FUNCTIONS_H */
