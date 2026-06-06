/* hydro_pair_types.h -- shared helpers for per-pair hydro physics sub-modules.
 *
 * Provides:
 *   - HLL_DIFFUSION_COMPROMISE_FACTOR / HLL_DIFFUSION_OVERSHOOT_FACTOR macros
 *   - hll_correction_fn(): pure function replacing the old HLL_correction macro
 *
 * These were previously defined inline inside hydro_functions.h / hydro_evaluate.h
 * (and relied on captured locals), which blocked the per-module physics fragments
 * from being extracted into standalone KOKKOS_INLINE_FUNCTIONs. Centralising them
 * here lets conduction/viscosity/CR-diffusion/RT-explicit/turb-diffusion functions
 * take all their inputs as explicit arguments.
 *
 * Requires allvars.h / kernel.h / hydro_structs.h already included.
 */

#ifndef HYDRO_PAIR_TYPES_H
#define HYDRO_PAIR_TYPES_H

#ifndef HLL_DIFFUSION_COMPROMISE_FACTOR
#ifdef MAGNETIC
#define HLL_DIFFUSION_COMPROMISE_FACTOR 1.1
#else
#define HLL_DIFFUSION_COMPROMISE_FACTOR 1.5
#endif
#endif

#ifndef HLL_DIFFUSION_OVERSHOOT_FACTOR
#if !defined(MAGNETIC) || defined(GALSF) || defined(COOLING) || defined(SINK_PARTICLES)
#define HLL_DIFFUSION_OVERSHOOT_FACTOR  0.005
#else
#define HLL_DIFFUSION_OVERSHOOT_FACTOR  1.0
#endif
#endif

KOKKOS_INLINE_FUNCTION
double hll_correction_fn(double ui, double uj, double wt, double kappa,
                         double v_hll, double Face_Area_Norm, double r, double cf_atime)
{
    double k_hll = v_hll * wt * r * cf_atime / fabs(kappa);
    k_hll = (0.2 + k_hll) / (0.2 + k_hll + k_hll * k_hll);
    return -1.0 * k_hll * Face_Area_Norm * v_hll * (ui - uj);
}

#endif /* HYDRO_PAIR_TYPES_H */
