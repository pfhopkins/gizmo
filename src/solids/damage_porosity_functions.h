/* damage_porosity_functions.h -- the damage rate equation and the yield-criterion
 * modifier. Body guarded by EOS_DAMAGE_POROSITY (bit 0 fracture damage, bit 1
 * Drucker-Prager). The P-alpha porosity update (bit 2) lives with the Tillotson
 * pressure it modifies (declarations/cell_data.h; crush curve in jutzi_crush_curve.h).
 *
 * The per-material parameters (Weibull k and m, Drucker-Prager mu) come
 * from solid_material_of(CompositionType) (declarations/allvars.h, eos/composition_registry.h).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DAMAGE_POROSITY_FUNCTIONS_H
#define DAMAGE_POROSITY_FUNCTIONS_H

#include <math.h>
#include "../declarations/allvars.h"
#include "../declarations/gpu_rng.h"
#include "../system/eigen_symmetric.h"

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
 * bit 1: Drucker-Prager pressure-dependent yield extension
 *
 * Returns the modified yield strength Y_DP = max(0, Y_0 + mu_DP * P_hydro),
 * where mu_DP is the per-material friction coefficient (DruckerPragerMu). Reduces
 * to von Mises (Y_0) for mu_DP = 0. P_hydro is the cell hydrostatic
 * pressure (positive in compression). Tensile failure (Y_DP -> 0) handled
 * naturally by the clamp.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double apply_drucker_prager(double Y0, double P_hydro, int composition_type)
{
    double mu_DP = solid_material_of(composition_type).DruckerPragerMu;
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
 * bit 0, FractureActivation = shear-activated: simplified Grady-Kipp scalar damage. It follows the sphere-of-cracks form of
 * Benz & Asphaug 1995, but the cracks grow in every particle from the start, flaws are
 * activated by the shear-strain proxy below rather than the tensile strain, and the flaw
 * population is not normalised per particle, so its thresholds differ from that model.
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
 * k_Weibull is held in code units (1/code_length^3), converted once when the materials are resolved.
 * ---------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
void damage_update_grady_kipp(double D_prev, double A_prev, double J2,
                              double mu_shear, double c_s, double dt,
                              int composition_type,
                              double *D_out, double *A_out)
{
    if(D_prev >= 1.0) { *D_out = 1.0; *A_out = A_prev; return; }

    double k_code    = solid_material_of(composition_type).WeibullK; /* [1/code_length^3] */
    double m_weibull = solid_material_of(composition_type).WeibullM;

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

/* ---------------------------------------------------------------------------
 * bit 0, FractureActivation = tensile flaws (the default): tensile-flaw fracture (Benz &
 * Asphaug 1994, 1995; as written in Schaefer et al. 2016, eqs. 20-24 and 45-49).
 *
 * Flaws: an element holds flaws whose activation strains follow the Weibull distribution
 * n(eps) = k eps^m per unit volume, in its fixed flaw volume V (mass x alpha_0 / rho_0, times
 * the out-of-plane extent in 1D/2D), so the thresholds do not move as the element is compressed
 * or stretched. In s = k V eps^m they are the arrivals s_1 < s_2 < ... of a unit-rate Poisson
 * process; each s_j is the sum of j unit exponential draws keyed on the persistent particle identity
 * (ID, child number, generation) and the flaw index, so the thresholds are fixed for the particle, need no storage and do not depend on the
 * domain decomposition. The element holds the flaws with s_j <= s_max, at least one, where
 * s_max = ln(N) N m / M for its material's N elements of total mass M (Benz & Asphaug's N ln N
 * flaws per body, shared out by mass).
 * Activation: eps = sigma_max / ((1-D) E), with sigma_max the largest principal value of the
 * damage-reduced total stress -P I + (1-D) S and E = 9 K mu / (3K + mu) (K the material bulk
 * modulus); flaws with s_j <= k V eps^m activate (n_act, kept in ActiveCracks). Pure compression
 * (sigma_max <= 0) activates nothing, and an element in compression with no active flaw draws
 * nothing.
 * Growth: d(D^(1/3))/dt = n_act c_g / R_s, c_g = 0.4 times the longitudinal wave speed of the
 * damaged material, R_s the radius of the element's current volume; D <= n_act / n_tot, and
 * D never decreases.
 * ---------------------------------------------------------------------------*/
/* Flaws per element: an element whose expected flaw count s_max exceeds TENSILE_FLAWS_MAX_PER_ELEMENT is refused at startup
   (solid_material_check_cells; it would need a mass hundreds of times its material's mean). The per-kick pass stops at ten times that, which only
   an element whose mass has grown since startup can reach: the update then reports failure and the run stops. */
#define TENSILE_FLAWS_MAX_PER_ELEMENT 10000

KOKKOS_INLINE_FUNCTION
double tensile_flaw_spacing(uint64_t identity, int j_flaw) /* s_j - s_(j-1): a unit exponential draw keyed on the particle identity and the flaw index */
{
    static constexpr uint64_t TENSILE_FLAW_RNG_SALT = gizmo_loop_rng_salt("tensile_flaws");
    return -log(1.0 - gizmo_gpu_rand_double(identity, gizmo_rng_event_counter(TENSILE_FLAW_RNG_SALT, 0, (uint64_t)j_flaw)));
}

KOKKOS_INLINE_FUNCTION
int damage_update_tensile_flaws(double D_prev, double n_act_prev, const double stress[3][3], int ndim,
                                double bulk_modulus, double mu_shear, double rho, double volume_now,
                                double flaw_volume, double s_max, uint64_t identity, double dt, int composition_type,
                                double *D_out, double *n_act_out) /* returns 1 (state unchanged) if the element's flaws exceed the per-element bound */
{
    if(!(D_prev > 0)) {D_prev = 0;} if(D_prev > 1) {D_prev = 1;}
    *D_out = D_prev; *n_act_out = n_act_prev;
    double k_code = solid_material_of(composition_type).WeibullK, m_weibull = solid_material_of(composition_type).WeibullM;
    if(D_prev >= 1.0 || !(k_code > 0) || !(m_weibull > 0) || !(volume_now > 0) || !(flaw_volume > 0) || !(rho > 0) || !(mu_shear > 0)) {return 0;}
    double n_stored = (n_act_prev > 0) ? DMIN(n_act_prev, 10. * TENSILE_FLAWS_MAX_PER_ELEMENT) : 0; /* bounded before the conversion: a stored value may come from a file */
    int n_act = (int)(n_stored + 0.5);

    /* how far the largest principal tensile stress has reached into the flaw population */
    double s_now = 0;
    if(bulk_modulus > 0) {
        double A[9] = {0}, eval[3] = {0};
        for(int a = 0; a < ndim; a++) {for(int b = 0; b < ndim; b++) {A[ndim*a + b] = stress[a][b];}}
        eigen_symmetric_values(A, eval, ndim);
        double sigma_max = eval[0];
        for(int a = 1; a < ndim; a++) {if(eval[a] > sigma_max) {sigma_max = eval[a];}}
        if(sigma_max > 0) {
            double E_young = 9. * bulk_modulus * mu_shear / (3. * bulk_modulus + mu_shear);
            double eps = sigma_max / ((1. - D_prev) * E_young);
            s_now = k_code * flaw_volume * pow(eps, m_weibull);
        }
    }
    if(n_act < 1 && !(s_now > 0)) {return 0;} /* nothing to activate and nothing growing */

    /* one pass over the thresholds: the element's flaw total n_tot (s_j <= s_max, at least one) and how many of them s_now has reached */
    int n_tot = 0, n_reached = 0;
    volatile int truncated = 1;
    double s = 0;
    for(int j = 1; j <= 10 * TENSILE_FLAWS_MAX_PER_ELEMENT; j++) {
        s += tensile_flaw_spacing(identity, j);
        if(j > 1 && s > s_max) {truncated = 0; break;}
        n_tot = j;
        if(s <= s_now) {n_reached = j;}
    }
    if(truncated) {return 1;}
    if(n_reached > n_act) {n_act = n_reached;}
    if(n_act > n_tot) {n_act = n_tot;} /* also bounds a stored count from a file */
    *n_act_out = (double)n_act;
    if(n_act < 1) {return 0;}

    /* crack growth from the active flaws */
    double c_L = sqrt((DMAX(bulk_modulus, 0.) + 2. * (1. - 1. / ndim) * (1. - D_prev) * mu_shear) / rho); /* longitudinal speed of the damaged material */
    double R_s = (ndim == 3) ? pow(3. * volume_now / (4. * M_PI), 1. / 3.) : ((ndim == 2) ? sqrt(volume_now / M_PI) : 0.5 * volume_now);
    double D_third = pow(D_prev, 1. / 3.) + n_act * 0.4 * c_L * dt / R_s;
    double D_new = D_third * D_third * D_third, D_cap = (double)n_act / (double)n_tot;
    if(D_new > D_cap) {D_new = D_cap;}
    if(D_new > 1.0) {D_new = 1.0;}
    if(D_new < D_prev) {D_new = D_prev;}
    *D_out = D_new;
    return 0;
}

/* Convenience: is bit 0 (fracture damage) requested? */
#if ((EOS_DAMAGE_POROSITY) & 1)
#define DAMAGE_POROSITY_BIT_FRACTURE 1
#else
#define DAMAGE_POROSITY_BIT_FRACTURE 0
#endif

#endif /* EOS_DAMAGE_POROSITY */

#endif /* DAMAGE_POROSITY_FUNCTIONS_H */
