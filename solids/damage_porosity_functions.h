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
#include "jutzi_crush_curve.h"

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
 * Returns the irreversible-monotone distention alpha given current matrix
 * pressure P_solid (i.e. the Tillotson/ANEOS pressure evaluated at matrix
 * density rho_s = alpha*rho), the previous distention alpha_prev, and the
 * cell's CompositionType. Compaction is one-way: alpha never grows.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double distention_jutzi_palpha_update(double alpha_prev, double P_solid,
                                      int composition_type)
{
    double alpha_0 = All.Tillotson_EOS_params[composition_type][15];
    double P_e     = All.Tillotson_EOS_params[composition_type][16];
    double P_s     = All.Tillotson_EOS_params[composition_type][17];
    double alpha_eq = jutzi_distention_eq8(P_solid, alpha_0, P_e, P_s);
    double alpha_new = (alpha_eq < alpha_prev) ? alpha_eq : alpha_prev;
    if(alpha_new < 1.0) { alpha_new = 1.0; }
    return alpha_new;
}

/* Convenience: is bit 2 (P-alpha porosity) requested for this build? */
#if ((EOS_DAMAGE_POROSITY) & 4)
#define DAMAGE_POROSITY_BIT_PALPHA 1
#else
#define DAMAGE_POROSITY_BIT_PALPHA 0
#endif

/* ---------------------------------------------------------------------------
 * bit 1: Drucker-Prager pressure-dependent yield extension
 *
 * Returns the modified yield strength Y_DP = max(0, Y_0 + mu_DP * P_hydro),
 * where mu_DP is the per-material friction coefficient (slot 14). Reduces
 * to von Mises (Y_0) for mu_DP = 0. P_hydro is the cell hydrostatic
 * pressure (positive in compression). Tensile failure (Y_DP -> 0) handled
 * naturally by the clamp.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double apply_drucker_prager(double Y0, double P_hydro, int composition_type)
{
    double mu_DP = All.Tillotson_EOS_params[composition_type][14];
    double Y_eff = Y0 + mu_DP * P_hydro;
    return (Y_eff > 0.0) ? Y_eff : 0.0;
}

/* Convenience: is bit 1 (Drucker-Prager) requested? */
#if ((EOS_DAMAGE_POROSITY) & 2)
#define DAMAGE_POROSITY_BIT_DRUCKER_PRAGER 1
#else
#define DAMAGE_POROSITY_BIT_DRUCKER_PRAGER 0
#endif

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
