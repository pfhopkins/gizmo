/* battery_functions.h -- gas-side magnetic battery EMFs.
 *
 * Provides the per-cell comoving electric-field contribution E'_battery from:
 *   bit 0: Biermann battery (Kulsrud 1997)               -- battery_E_Biermann()
 *   bit 1: radiative-ionization battery                  -- battery_E_RadIonization()
 *                                                          (Harrison 1973 / Durrive & Langer 2015)
 *   bit 2: dust battery, terminal-velocity / fluid limit -- (delegates to solids/dust_battery_functions.h)
 *   bit 3: dust battery, explicit J_d                    -- (delegates to solids/dust_battery_functions.h)
 *
 * Plus the per-cell aggregator battery_assemble_per_cell_emf() that combines
 * whichever bits are active, and the shared linear-algebra helper
 * nonideal_mhd_assemble_bflux() that both classical non-ideal MHD and the
 * dust-battery contributions call to map (coef_O, coef_H, coef_A, J, B-hat) ->
 * b_flux = -coef_O J - coef_H (J x B^) - coef_A ((J x B^) x B^).
 *
 * The induction-equation source is dB/dt|_battery = -c (grad x E'_battery),
 * realized in the existing per-pair hydro flux as
 *   Fluxes.B += cross(Face_Area_Vec, 0.5*(E'_battery_i + E'_battery_j)),
 * mirroring nonideal_mhd_compute_pair.
 *
 * All entries are guarded by MHD_BATTERY_MECHANISMS. Header-only,
 * KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Reference: Soliman, Hopkins & Squire 2025, ApJ 985, 55 (arXiv:2410.21461)
 * for the dust battery; Kulsrud+ 1997 / Graziani+ 2015 for Biermann; Harrison
 * 1973 / Durrive & Langer 2015 for the radiative-ionization term.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef BATTERY_FUNCTIONS_H
#define BATTERY_FUNCTIONS_H

#include "hydro_pair_types.h"

#ifdef MHD_BATTERY_MECHANISMS

/* TODO: per-cell Biermann electric field E'_Bier = -(1/(e n_e)) grad P_e
   (or, with MHD_BATTERY_BIERMANN_DIRECT_CURL set, the direct-curl form
   dB/dt|_Bier = (c k_B / (e n_e)) grad T_e x grad n_e routed straight to DtB).
   See plan in ~/.claude/plans/proud-exploring-lagoon.md. */

/* TODO: per-cell radiative-ionization electric field E'_RI from sigma_nu * x_n * F_rad.
   Compact form parallel to Biermann; ~10 lines of math. */

/* TODO: shared helper nonideal_mhd_assemble_bflux(coef_O, coef_H, coef_A, J, bhat)
   refactored out of nonideal_mhd_functions.h:71-76 and reused by dust battery. */

/* TODO: per-cell aggregator battery_assemble_per_cell_emf() called from the
   density/gradient post-pass, populating CellP[i].E_battery_cell. */

#endif /* MHD_BATTERY_MECHANISMS */

#endif /* BATTERY_FUNCTIONS_H */
