/* jutzi_crush_curve.h -- Jutzi 2008 quadratic P-alpha crush curve, Eq. 8.
 *
 * Standalone math header with NO GIZMO dependencies (only <math.h>) so that
 * the analytic-crush unit test can include this file without the rest of the
 * GIZMO build. The damage_porosity_functions.h wrapper pulls this in and
 * supplies the per-cell / per-material plumbing.
 *
 * Reference: Jutzi, Benz, Michel 2008, Icarus 198, 242, Eq. 8 (quadratic form).
 *   alpha(P) = 1 + (alpha_0 - 1) * ( (P_s - P) / (P_s - P_e) )^2   for P_e <= P <= P_s
 *   alpha(P) = alpha_0                                              for P <  P_e
 *   alpha(P) = 1                                                    for P >  P_s
 *
 * Distention is irreversible: physical compaction is one-way, so the consumer
 * should always take min(alpha_current, alpha_new).
 */

#ifndef JUTZI_CRUSH_CURVE_H
#define JUTZI_CRUSH_CURVE_H

#include <math.h>

/* Both includers (declarations/cell_data.h and solids/damage_porosity_functions.h)
   reach this header through declarations/allvars.h, which pulls in macros.h and so
   defines GIZMO_GPU_FUNCTION first. This backstop only covers a future direct
   includer, and mirrors macros.h in falling back to host+device under a GPU
   compiler rather than to nothing: a host-only fallback would silently reintroduce
   the "__host__ function called from a __host__ __device__ function" defect that
   only a device compiler diagnoses, and only as a warning. */
#ifndef GIZMO_GPU_FUNCTION
#if (defined(__CUDACC__) || defined(__HIPCC__))
#define GIZMO_GPU_FUNCTION __device__ __host__
#else
#define GIZMO_GPU_FUNCTION
#endif
#endif

/* Returns the equilibrium distention alpha given current pressure P,
 * initial distention alpha_0, elastic-limit pressure P_e, and full-
 * compaction pressure P_s. All in consistent units (e.g. cgs). */
GIZMO_GPU_FUNCTION static inline double jutzi_distention_eq8(double P, double alpha_0,
                                          double P_e, double P_s)
{
    if(!(alpha_0 > 1.0)) { return 1.0; }            /* solid material */
    if(!(P_s > P_e))     { return alpha_0; }        /* degenerate range guard */
    if(P <= P_e)         { return alpha_0; }
    if(P >= P_s)         { return 1.0; }
    double r = (P_s - P) / (P_s - P_e);
    return 1.0 + (alpha_0 - 1.0) * r * r;
}

#endif /* JUTZI_CRUSH_CURVE_H */
