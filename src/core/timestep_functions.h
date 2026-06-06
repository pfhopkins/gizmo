/* timestep_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations of
 * timestep utility functions.  Single source of truth for both CPU and GPU.
 *
 * timestep_dilation_factor forwards to return_timestep_dilation_factor on
 * CPU when USE_TIMESTEP_DILATION_FOR_ZOOMS is enabled.  GPU kernels read the
 * host-computed particle cache; the host function remains the physics source
 * of truth.
 *
 * Include order: after allvars.h, proto.h. */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#if defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) && !defined(GIZMO_GPU_COMPILER)
/* Forward declaration so this header is self-contained when included from TUs
 * that don't transitively include proto.h. The full definition lives in
 * core/timestep.cc and is unconditional (it short-circuits to 1 when
 * USE_TIMESTEP_DILATION_FOR_ZOOMS is undefined). */
double return_timestep_dilation_factor(int i, int mode, struct particle_data *pp);
#endif

KOKKOS_INLINE_FUNCTION
double timestep_dilation_factor(int i, int mode, struct particle_data *pp)
{
#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
#ifndef GIZMO_GPU_COMPILER
    return return_timestep_dilation_factor(i, mode, pp);
#else
    if(i < 0) {return 1;}
    if(mode != 0) {endrun(929101); return 1;}
    return pp[i].TimestepDilationFactor;
#endif
#else
    return 1;
#endif
}

KOKKOS_INLINE_FUNCTION
double unit_integertime_in_physical(int i, struct particle_data *pp)
{
    return (All.Timebase_interval / All.cf_hubble_a) * timestep_dilation_factor(i, 0, pp);
}

KOKKOS_INLINE_FUNCTION
double get_physical_timestep_from_timebin(int bin, int i, struct particle_data *pp)
{
    return GET_INTEGERTIME_FROM_TIMEBIN(bin) * unit_integertime_in_physical(i, pp);
}

KOKKOS_INLINE_FUNCTION
double get_particle_timestep_in_physical(int i, struct particle_data *pp)
{
    return pp[i].integertime_step() * unit_integertime_in_physical(i, pp);
}
