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

/* TODO: per-cell radiative-ionization electric field E'_RI from sigma_nu * x_n * F_rad.
   Compact form parallel to Biermann; ~10 lines of math. */

/* nonideal_mhd_assemble_bflux() now lives in hydro/nonideal_mhd_functions.h
   (factored out in commit 49d194a3). Dust battery includes that header. */


/* ============================================================================
 * Biermann battery
 *
 * Ohm's-law EMF form:
 *   E'_Bier = -(1/(n_e e)) grad p_e = -(k_B/e) [grad T_e + (T_e/n_e) grad n_e]
 *
 * Verification of the curl: ∇ x E'_Bier = (k_B/(e n_e)) (∇n_e × ∇T_e),
 * giving the canonical Biermann induction
 *   ∂B/∂t|_Bier = -c ∇ x E'_Bier = -(c k_B / (e n_e)) ∇n_e × ∇T_e.
 *
 * Inputs (all physical/cgs):
 *   n_e [cm^-3]     -- cell.n_e()         (populated in eos/eos.cc, commit 5)
 *   T_e [K]         -- cell.T_e()         (idem)
 *   grad n_e        -- cell.Gradients.ElectronNumberDensity
 *                      stored in cgs n_e per *code-unit length*
 *   grad T_e        -- cell.Gradients.ElectronTemperature
 *                      stored in K per *code-unit length*
 *
 * The function returns E'_Bier *in code units suitable for the existing
 * Fluxes.B = cross(Face_Area_Vec, 0.5*(E_i+E_j)) wiring*. The conversion
 * follows from dB/dt = -c ∇ x E and the chain
 *   E_code = (C_LIGHT_CODE / UNIT_B_IN_GAUSS / UNIT_LENGTH_IN_CGS) * E_phys[esu/cm^2 statvolt/cm]
 * where the extra 1/UNIT_LENGTH_IN_CGS is because the gradients are stored
 * per code-length, so grad_phys = grad_code / UNIT_LENGTH_IN_CGS, and that
 * factor appears once (not twice) since E_phys ∝ grad.
 *
 * Cosmological factors (cf_atime, cf_a3inv): NOT yet applied. The whole
 * nonideal+battery cosmological-units pass is its own deliverable, paired
 * with the long-standing comment in nonideal_mhd_functions.h. Until that
 * lands, callers should restrict tests to cf_atime=1 (non-cosmological).
 * ========================================================================== */

KOKKOS_INLINE_FUNCTION
Vec3<double> battery_E_Biermann(int i, struct gas_cell_data *cell)
{
    Vec3<double> E_zero = {0,0,0};
#if (MHD_BATTERY_MECHANISMS & 1)
    const double n_e = cell[i].n_e();
    const double T_e = cell[i].T_e();
    if(!(n_e > 0)) {return E_zero;}

    const Vec3<double> g_ne = cell[i].Gradients.ElectronNumberDensity; /* cgs/length_code */
    const Vec3<double> g_Te = cell[i].Gradients.ElectronTemperature;   /* K/length_code */

    /* prefactor [statvolt/cm] / [K/cm or (cm^-3)/cm * K/cm^-3]: just k_B/e */
    const double kB_over_e = BOLTZMANN_CGS / ELECTRONCHARGE_CGS; /* erg/K / esu = statvolt·cm/K */

    /* E_phys in [statvolt/cm], using grad_phys = grad_code / UNIT_LENGTH_IN_CGS */
    Vec3<double> E_phys;
    {
        const double inv_L = 1.0 / UNIT_LENGTH_IN_CGS;
        const double prefac_1 = -kB_over_e * inv_L;                    /* multiplies grad T_e */
        const double prefac_2 = -kB_over_e * (T_e / n_e) * inv_L;      /* multiplies grad n_e */
        for(int k=0;k<3;k++) {E_phys[k] = prefac_1 * g_Te[k] + prefac_2 * g_ne[k];}
    }
    /* code-unit conversion: dB[code]/dt[code] = -∇_code × E_code with
       E_code = (C_LIGHT_CODE / UNIT_B_IN_GAUSS) * E_phys */
    const double to_code = C_LIGHT_CODE / UNIT_B_IN_GAUSS;
    return to_code * E_phys;
#else
    (void)i; (void)cell;
    return E_zero;
#endif
}

/* ============================================================================
 * Per-cell aggregator. Combines whichever battery bits are active and writes
 * the result to cell[i].E_battery_cell. Called once per active gas cell, after
 * the gradient pass has finalized grad(n_e)/grad(T_e) (and, eventually,
 * grad(other ionization fields) for RI / dust contributions).
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
void battery_assemble_per_cell_emf(int i, struct gas_cell_data *cell)
{
    Vec3<double> E_total = {0,0,0};
#if (MHD_BATTERY_MECHANISMS & 1)
    E_total += battery_E_Biermann(i, cell);
#endif
    /* (MHD_BATTERY_MECHANISMS & 2): RI contribution -- TODO */
    /* (MHD_BATTERY_MECHANISMS & 4): dust TVA, in solids/dust_battery_functions.h -- TODO */
    /* (MHD_BATTERY_MECHANISMS & 8): dust explicit-J_d, idem -- TODO */
    cell[i].E_battery_cell = E_total;
}

#endif /* MHD_BATTERY_MECHANISMS */

#endif /* BATTERY_FUNCTIONS_H */
