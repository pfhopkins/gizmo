/* elastic_physics_functions.h — KOKKOS_INLINE_FUNCTION inline body for
 * get_negative_pressure_tensilecorrfac, the tensile-instability kernel
 * correction factor used by the hydro pair body under solid-EOS Configs
 * (EOS_TILLOTSON / EOS_ELASTIC / EOS_ANEOS).
 *
 * Why this header: hydro_functions.h:hydro_accumulate_neighbor (a
 * __host__ __device__ pair body, in the corridor hot path) calls this
 * function on the device pass. Without an inline body visible at the call
 * site, the nvc++ device pass sees only the host-only forward declaration
 * in core/proto.h and emits warning #20011-D ("calling a __host__ function
 * from a __host__ __device__ function is not allowed"). That is a silent
 * physics error on GPU.
 *
 * elastic_physics.cc continues to provide the non-inline host external
 * symbol via the standard #undef KOKKOS_INLINE_FUNCTION / re-include
 * pattern; other host call sites (proto.h forward-declared) link against
 * that one strong symbol.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Requires the caller to have already included mesh/kernel.h (provides
 * kernel_main) and declarations/allvars.h (provides All.DesNumNgb).
 * NOT included here because mesh/kernel.h has no include guard and would
 * cause "redefinition" errors when this header is pulled in by hydro_functions.h
 * (which has already included kernel.h via its own include chain). */

#if defined(EOS_ELASTIC) || defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
/* routine to get and define the correction factor needed to prevent tensile instability for negative pressures, for arbitrary kernels & dimensions */
KOKKOS_INLINE_FUNCTION
double get_negative_pressure_tensilecorrfac(double r, double h_i, double h_j)
{
    double dx_ips=0, wk_0=0, dwk_tmp=0, wk_r=0, r_over_heff=0;
#if (NUMDIMS==1)
    dx_ips = 2. / All.DesNumNgb; // 1D inter-node separation for desired NNgb, relative to radius of compact support
#elif(NUMDIMS==2)
    dx_ips = sqrt(M_PI / All.DesNumNgb); // 2D inter-node separation for desired NNgb, relative to radius of compact support
#else
    dx_ips = pow(4.*M_PI/3. / All.DesNumNgb, 1./3.); // 3D inter-node separation for desired NNgb, relative to radius of compact support
#endif
    kernel_main(dx_ips, 1., 1., &wk_0, &dwk_tmp, -1); // use kernels because of their stability properties: here weight for 'mean separation'
    r_over_heff = r / DMAX(h_i, h_j);
    kernel_main(r_over_heff, 1., 1., &wk_r, &dwk_tmp, -1); // here weight for actual half-separation
    return 0.2 * pow(wk_r / wk_0, 4); // correction factor for n=4 from Monaghan et al. 2000, Gray et al. 2001
}
#endif
