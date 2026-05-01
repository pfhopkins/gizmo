/* dust_battery_functions.h -- dust-exclusive parts of the magnetic battery.
 *
 * Provides:
 *   battery_alpha_coeffs(...)    -- Soliman, Hopkins & Squire 2025 Eqs. 10-12
 *                                   (alpha_O, alpha_H, alpha_A from local
 *                                   plasma collision rates and densities).
 *   battery_E_dust_TVA(...)      -- E'_bat,d in the terminal-velocity
 *                                   approximation (SHS25 Eqs. 16-18; bit 4).
 *   battery_E_dust_explicit(...) -- E'_bat,d using the actual dust current J_d
 *                                   summed from grain particles (SHS25 Eq. 9;
 *                                   bit 8 / GRAIN_FLUID).
 *   alpha_to_eta(...)            -- conversion eta = (c^2 / 4 pi) * alpha
 *                                   (textbook conversion between mobility
 *                                   coefficient and diffusivity).
 *   dust_battery_assemble_E_cell(...) -- per-cell aggregator. Picks bit 4
 *                                   (TVA) or bit 8 (explicit) based on the
 *                                   compile-time MHD_BATTERY_MECHANISMS gate
 *                                   and returns E'_bat,d in cgs statvolt/cm.
 *                                   Called from eos/eos.cc after T_e/n_e and
 *                                   E_battery_T2_cell (radiative) are set;
 *                                   adds to E_battery_T2_cell.
 *
 * The dust-battery EMF E'_bat,d is built per-cell here, then exported to the
 * gradient pass for slope-limited gradient computation; the curl is taken in
 * hydro_toplevel.cc::out2particle_hydra and added as a cell-centered source
 * to DtB. Energy conservation is automatic via the existing per-cell
 * DtInternalEnergy -= dot(B_phys, DtB) hook (battery is conservative, not
 * dissipative). The shared linear-algebra helper nonideal_mhd_assemble_bflux()
 * lives in hydro/nonideal_mhd_functions.h and is reused for the explicit-J_d
 * variant.
 *
 * All entries guarded by MHD_BATTERY_MECHANISMS & (4|8). Header-only,
 * KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * STATUS: scaffolding only. The aggregator returns {0,0,0} and is wired into
 * the Tier-2 accumulator via dust_battery_assemble_E_cell(). Each of the
 * underlying SHS25-equation implementations is a TODO. When implemented, the
 * existing wiring will pick them up automatically.
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
   SHS25 Eq. 9. Calls nonideal_mhd_assemble_bflux() with eta = (c^2/4pi)*alpha,
   then converts the resulting b_flux back to E (b_flux is structurally -E in
   the helper's convention). */

/* TODO: alpha_to_eta() — eta_j = (c^2 / 4 pi) * alpha_j (textbook). */


/* ============================================================================
 * Per-cell aggregator. Sums the active dust-battery contribution into the
 * cell-centered Tier-2 EMF accumulator E_battery_T2_cell. Caller adds the
 * return value to cell[i].E_battery_T2_cell after the radiative builder has
 * run. Returns E'_bat,d in cgs statvolt/cm (= Gauss; Heaviside-Lorentz units
 * matching what the gradient pass expects).
 *
 * Currently a stub returning {0,0,0}. Once the SHS25 implementations above
 * land, replace the stub body with:
 *
 *   if (MHD_BATTERY_MECHANISMS & 8): explicit J_d path
 *     - Read cell.J_dust_cell (populated by a separate density-style pass
 *       that sums grain currents from grain particles in the kernel).
 *     - alpha_{O,H,A} = battery_alpha_coeffs(...).
 *     - Return battery_E_dust_explicit(cell.J_dust_cell, alpha_*, b_hat).
 *
 *   if (MHD_BATTERY_MECHANISMS & 4): TVA path
 *     - Return battery_E_dust_TVA(grain charge, stopping time, delta_a_dg,
 *       dust-to-gas ratio, ...).
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
Vec3<MyDouble> dust_battery_assemble_E_cell(int i, struct gas_cell_data *cell, struct particle_data *pp)
{
    (void)i; (void)cell; (void)pp;
    Vec3<MyDouble> E_dust = {};
    /* TODO: replace stub with calls to battery_E_dust_TVA / battery_E_dust_explicit
       per SHS25 Eqs. 9, 16-18 once those are implemented. */
    return E_dust;
}

#endif /* MHD_BATTERY_MECHANISMS & (4|8) */

#endif /* DUST_BATTERY_FUNCTIONS_H */
