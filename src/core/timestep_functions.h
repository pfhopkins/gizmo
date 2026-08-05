/* timestep_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations of
 * timestep utility functions.  Single source of truth for both CPU and GPU.
 *
 * timestep_dilation_factor returns the dilation factor frozen for this particle when its
 * timestep was assigned (core/timestep.cc, get_timestep), so a step's physical landing time
 * cannot mutate while it is being taken.  Identical on CPU and GPU.  The live evaluations,
 * for timestep assignment and for tree nodes, are host-only and live in core/timestep.cc.
 *
 * Include order: after allvars.h, proto.h. */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double timestep_dilation_factor(int i, struct particle_data *pp)
{
#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
    if(i < 0) {return 1;}
    return pp[i].TimestepDilationFactor;
#else
    (void)i; (void)pp; return 1;
#endif
}

KOKKOS_INLINE_FUNCTION
double unit_integertime_in_physical(int i, struct particle_data *pp)
{
    return (All.Timebase_interval / All.cf_hubble_a) * timestep_dilation_factor(i, pp);
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
