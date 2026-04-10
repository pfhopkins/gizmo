/* timestep_device.h — KOKKOS_INLINE_FUNCTION versions of the timestep
 * utility functions needed by do_the_cooling_for_particle on the GPU.
 *
 * cooling.cc includes this header so the GPU cooling kernel can call
 * get_particle_timestep_in_physical without -rdc=true (inlined at the call
 * site from the same TU).
 *
 * Note: USE_TIMESTEP_DILATION_FOR_ZOOMS is not supported on GPU; the dilation
 * factor is always 1 here.  If that feature is ever needed on-device the zoom
 * logic must be ported separately.
 *
 * Include order: must come after allvars.h (for All, integertime macros) and
 * after the particle_data header (for integertime_step()).
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

KOKKOS_INLINE_FUNCTION
double timestep_dilation_factor(int i, int mode, struct particle_data *pp)
{
    /* USE_TIMESTEP_DILATION_FOR_ZOOMS not supported on GPU — always 1 */
    return 1.;
}

KOKKOS_INLINE_FUNCTION
double unit_integertime_in_physical(int i, struct particle_data *pp)
{
    return (All.Timebase_interval / All.cf_hubble_a) * timestep_dilation_factor(i, 0, pp);
}

KOKKOS_INLINE_FUNCTION
double get_particle_timestep_in_physical(int i, struct particle_data *pp)
{
    return pp[i].integertime_step() * unit_integertime_in_physical(i, pp);
}
