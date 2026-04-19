/* sidm_helper_functions.h — GPU-callable SIDM helper math:
 *   - g_geo                     geometric-factor spline (reads GeoFactorTable)
 *   - prob_of_interaction       per-pair SIDM scattering probability
 *   - calculate_interact_kick_rng  isotropic post-scatter kick using counter-RNG
 *
 * Each function is a pure KOKKOS_INLINE_FUNCTION: no internal references to
 * global state beyond All (available via the All_dev mirror on GPU), and no
 * GSL RNG. The tabulated geometric factor is threaded in as an explicit
 * pointer; callers pass `GeoFactorTable` (host) or the SharedSpace mirror
 * (device).
 *
 * RNG migration: calculate_interact_kick used to call gsl_rng_uniform twice
 * (for cos_theta and phi). The new `_rng` variant takes a 64-bit key and a
 * 64-bit counter and uses the counter-based RNG in declarations/gpu_rng.h
 * (tag nibble 1 = SIDM scatter direction). The caller chooses key/counter to
 * guarantee (i,j)-symmetric, timestep-reproducible draws on both CPU and GPU.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef SIDM_HELPER_FUNCTIONS_H
#define SIDM_HELPER_FUNCTIONS_H

#include "../declarations/allvars.h"
#include "../declarations/gpu_rng.h"

/* KOKKOS_INLINE_FUNCTION falls back to plain inline outside a GPU TU. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif


#ifdef DM_SIDM

/* Geometric factor g_geo(r/h_si) evaluated against a caller-supplied lookup
   table of length GEOFACTOR_TABLE_LENGTH. The table is populated by
   init_geofactor_table() (host-only, uses GSL integration). */
KOKKOS_INLINE_FUNCTION
double g_geo_tab(double r_over_hsi, const MyDouble *GeoFactorTable_arr)
{
    double u = r_over_hsi / 2.0 * GEOFACTOR_TABLE_LENGTH;
    int i = (int) u;
    if(i >= GEOFACTOR_TABLE_LENGTH) i = GEOFACTOR_TABLE_LENGTH - 1;
    double f;
    if(i <= 1) {
        f = 0.992318 + (GeoFactorTable_arr[0] - 0.992318) * u;
    } else {
        f = GeoFactorTable_arr[i - 1] + (GeoFactorTable_arr[i] - GeoFactorTable_arr[i - 1]) * (u - i);
    }
    return f;
}


/* Pair scattering probability. Vec3 dV is i-minus-j velocity difference
   (code units, same as the caller passes to the CPU tree-walk). The
   `_tab` suffix marks the table-taking variant used on GPU. */
KOKKOS_INLINE_FUNCTION
double prob_of_interaction_tab(double mass, double r, double h_si,
                               const Vec3<double> &dV, double dt,
                               const MyDouble *GeoFactorTable_arr)
{
    double dVmag = sqrt(dV[0]*dV[0] + dV[1]*dV[1] + dV[2]*dV[2]) / All.cf_atime;
    double rho_eff = mass / (h_si*h_si*h_si) * All.cf_a3inv;
    double cx_eff = All.DM_InteractionCrossSection * g_geo_tab(r/h_si, GeoFactorTable_arr);
    double units = UNIT_SURFDEN_IN_CGS;
    if(All.DM_InteractionVelocityScale > 0) {
        double x = dVmag / All.DM_InteractionVelocityScale;
        cx_eff /= 1 + x*x*x*x;
    }
    return rho_eff * cx_eff * dVmag * dt * units;
}


/* Momentum-conserving isotropic scatter kick. Consumes two draws from the
   counter-based RNG, keyed by (rng_key, 2*rng_counter_base) and
   (rng_key, 2*rng_counter_base + 1). Writes to `kick`. `dV` is i-minus-j
   velocity difference (code units). `m_mean` is the pairwise-mean mass
   (unused in the kick formula itself — kept only so the signature matches
   the prior CPU prototype and makes the math path obvious to readers). */
KOKKOS_INLINE_FUNCTION
void calculate_interact_kick_rng(const Vec3<double> &dV, Vec3<double> &kick,
                                 double /*m_mean*/,
                                 uint64_t rng_key, uint64_t rng_counter_base)
{
    double dVmag = (1.0 - All.DM_DissipationFactor) *
                   sqrt(dV[0]*dV[0] + dV[1]*dV[1] + dV[2]*dV[2]);
    if(dVmag < 0) dVmag = 0;
    if(All.DM_KickPerCollision > 0) {
        double v0 = All.DM_KickPerCollision;
        dVmag = sqrt(dVmag*dVmag + v0*v0);
    }
    double u0 = gizmo_gpu_rand_double(rng_key, 2 * rng_counter_base);
    double u1 = gizmo_gpu_rand_double(rng_key, 2 * rng_counter_base + 1);
    double cos_theta = 2.0 * u0 - 1.0;
    double sin_theta_sq = 1.0 - cos_theta * cos_theta;
    double sin_theta = (sin_theta_sq > 0) ? sqrt(sin_theta_sq) : 0.0;
    double phi = u1 * 2.0 * M_PI;
    kick[0] = 0.5 * (dV[0] + dVmag * sin_theta * cos(phi));
    kick[1] = 0.5 * (dV[1] + dVmag * sin_theta * sin(phi));
    kick[2] = 0.5 * (dV[2] + dVmag * cos_theta);
}

#endif /* DM_SIDM */

#endif /* SIDM_HELPER_FUNCTIONS_H */
