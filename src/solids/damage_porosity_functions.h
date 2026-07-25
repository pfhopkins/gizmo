/* damage_porosity_functions.h -- damage and P-alpha porosity rate
 * equations and yield-criterion modifier. Body guarded by EOS_DAMAGE_POROSITY
 * (bit 0 Grady-Kipp, bit 1 Drucker-Prager, bit 2 P-alpha Jutzi 2008).
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
#include "../declarations/allvars.h"
#include "jutzi_crush_curve.h"

#ifdef EOS_DAMAGE_POROSITY
/* KOKKOS_INLINE_FUNCTION fallback: GPU TUs (cooling.cc et al.) already include
 * <Kokkos_Core.hpp> upstream and get the real `inline __device__ __host__`
 * expansion.  Host-only TUs (solids/elastic_physics.cc — built by mpicxx, not
 * nvcc_wrapper) must NOT pull <Kokkos_Core.hpp> in transitively: under Vista
 * CUDA Kokkos, doing so triggers Kokkos_Setup_Cuda.hpp's `__CUDACC__` precondition
 * and fails the build.  Same fallback pattern as eos/eos_functions.h. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

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
 * bit 0: Grady-Kipp scalar damage (Benz & Asphaug 1995 Eq. 3-6)
 *
 * Each particle tracks an effective crack radius ActiveCracks [code length].
 * At each kick step:
 *   1. Crack grows: A_new = A + c_g * dt  (c_g ≈ 0.4 * c_s)
 *   2. Active-flaw density from Weibull: n_act = k_W * eps_eq^m_W
 *      where eps_eq = sqrt(J2) / mu  (shear-strain proxy from deviatoric stress)
 *   3. D = min(1, n_act * (4π/3) * A_new^3)   (sphere volume * flaw density)
 *   4. Irreversibility: D never decreases, A never decreases.
 *
 * Arguments: D_prev, A_prev (ActiveCracks), J2 (deviatoric 2nd invariant
 * [pressure^2]), mu_shear (shear modulus [pressure]), c_s (sound speed
 * [code vel]), dt (timestep), composition_type.  Outputs via pointers.
 * k_Weibull unit conversion applied here: k_code = k_cgs * UNIT_LENGTH_IN_CGS^3.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
void damage_update_grady_kipp(double D_prev, double A_prev, double J2,
                              double mu_shear, double c_s, double dt,
                              int composition_type,
                              double *D_out, double *A_out)
{
    if(D_prev >= 1.0) { *D_out = 1.0; *A_out = A_prev; return; }

    double k_cgs     = All.Tillotson_EOS_params[composition_type][12];
    double m_weibull = All.Tillotson_EOS_params[composition_type][13];

    /* convert k [1/cm^3] → [1/code_length^3] */
    double Lcgs = UNIT_LENGTH_IN_CGS;
    double k_code = k_cgs * (Lcgs * Lcgs * Lcgs);

    /* crack growth speed and new radius */
    double c_g = 0.4 * c_s;
    double A_new = A_prev + c_g * dt;
    if(A_new < 0.0) { A_new = 0.0; }

    /* Weibull active-flaw density at current equivalent shear strain */
    double eps_eq = (mu_shear > 0.0) ? sqrt(J2) / mu_shear : 0.0;
    double n_act  = (eps_eq > 0.0 && m_weibull > 0.0 && k_code > 0.0)
                    ? k_code * pow(eps_eq, m_weibull) : 0.0;

    /* D from sphere-of-cracks model */
    double D_new = n_act * (4.0 * M_PI / 3.0) * A_new * A_new * A_new;
    if(D_new > 1.0) { D_new = 1.0; }
    if(D_new < D_prev) { D_new = D_prev; }  /* irreversible */

    *D_out = D_new;
    *A_out = A_new;
}

/* Convenience: is bit 0 (Grady-Kipp damage) requested? */
#if ((EOS_DAMAGE_POROSITY) & 1)
#define DAMAGE_POROSITY_BIT_GRADY_KIPP 1
#else
#define DAMAGE_POROSITY_BIT_GRADY_KIPP 0
#endif

#endif /* EOS_DAMAGE_POROSITY */

#endif /* DAMAGE_POROSITY_FUNCTIONS_H */
