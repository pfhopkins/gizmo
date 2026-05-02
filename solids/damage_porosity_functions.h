/* damage_porosity_functions.h -- Phase 17e damage and P-alpha porosity rate
 * equations and yield-criterion modifier. Body guarded by EOS_DAMAGE_POROSITY
 * (bit 0 Grady-Kipp, bit 1 Drucker-Prager, bit 2 P-alpha Jutzi 2008).
 *
 * Stubs in C2; physics wired in C3 (P-alpha) / C4 (Drucker-Prager) / C5
 * (Grady-Kipp).
 *
 * Tillotson_EOS_params slot map for damage/porosity:
 *   [12] = k_Weibull [1/cm^3]   (flaw number density)
 *   [13] = m_Weibull [-]        (Weibull modulus)
 *   [14] = mu_DP    [-]         (Drucker-Prager friction coefficient)
 *   [15] = alpha_0  [-]         (initial distention; 1 = solid)
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DAMAGE_POROSITY_FUNCTIONS_H
#define DAMAGE_POROSITY_FUNCTIONS_H

#include <math.h>
#include <Kokkos_Core.hpp>
#include "../declarations/allvars.h"

#ifdef EOS_DAMAGE_POROSITY

/* convenience accessor for whether a sub-bit is on (compile-time when the
 * runtime flag is a literal in tests; runtime otherwise) */
KOKKOS_INLINE_FUNCTION
int damage_porosity_bit_active(int bit_index)
{
    return ((EOS_DAMAGE_POROSITY) >> bit_index) & 1;
}

/* ---------------------------------------------------------------------------
 * bit 2: P-alpha porosity (Jutzi 2008 quadratic crush curve)
 *
 * dot(alpha) rate from current cell pressure P and distention alpha. Returns
 * 0 outside [P_e, P_s] crush range. Material-specific P_e/P_s pulled from the
 * registry in C3.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double distention_rate_jutzi_palpha(double alpha, double P, double dPdt,
                                    int composition_type)
{
    (void)alpha; (void)P; (void)dPdt; (void)composition_type;
    /* C3 will populate: quadratic crush curve d(alpha)/dt = -2*(alpha-1)*
     *  (P-P_e)/(P_s-P_e)^2 * dP/dt for dP/dt > 0 and alpha > 1 */
    return 0.0;
}

/* ---------------------------------------------------------------------------
 * bit 1: Drucker-Prager pressure-dependent yield extension
 *
 * Given the von Mises yield Y0 and the cell trace I_1 = tr(sigma) (or hydrostatic
 * pressure), return the modified yield strength Y_DP = Y0 + mu_DP * I_1
 * clamped at zero. Wired into elastic_body_update_driftkick yield block in C4.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double apply_drucker_prager(double Y0, double I_1_or_P, int composition_type)
{
    (void)I_1_or_P; (void)composition_type;
    /* C4 will populate: Y_DP = max(0, Y0 + mu_DP * I_1) */
    return Y0;
}

/* ---------------------------------------------------------------------------
 * bit 0: Grady-Kipp scalar damage from Weibull flaw distribution
 *
 * dot(D) growth rate from active-crack count and local strain rate. Returns 0
 * for D >= 1 (fully damaged). Wired in C5; per-pair stress carries (1-D) factor.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double damage_rate_grady_kipp(double D, double active_cracks,
                              double strain_rate_invariant,
                              int composition_type)
{
    (void)D; (void)active_cracks; (void)strain_rate_invariant;
    (void)composition_type;
    /* C5 will populate: dot(D)^(1/3) = (n_act * c_g / 3) using Grady-Kipp
     * crack-growth speed c_g and number of active flaws n_act from k/m Weibull. */
    return 0.0;
}

#endif /* EOS_DAMAGE_POROSITY */

#endif /* DAMAGE_POROSITY_FUNCTIONS_H */
