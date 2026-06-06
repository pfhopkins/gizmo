/* SSOT for Type=0 fluid-subtype IDs and pair predicates. FluidType partitions
 * the hydro corridor; it does not replace GIZMO particle Type. */

#ifndef MULTIFLUID_HELPERS_H
#define MULTIFLUID_HELPERS_H

#include "macros.h"   /* KOKKOS_INLINE_FUNCTION fallback for non-Kokkos builds */

#ifdef HYDRO_MULTIFLUID

enum FluidID
{
    FLUID_DEFAULT     = 0,
    FLUID_DUST_GRAIN  = 1,   /* Type=0 dust-like fluid, not Type=3 grains */
    FLUID_ION         = 2,
    /* FLUID_NEUTRAL: reserved stub for future multi-species setups; not used
     * in the default HYDRO_MULTIFLUID_IONNEUTRAL build, which puts neutrals in
     * FLUID_DEFAULT by convention (see Template_Config.sh). */
    FLUID_NEUTRAL     = 3,
    FLUID_DM          = 4,   /* dark fluid (e.g. atomic/mirror DM); promotes to Type=3 inert under HYDRO_MULTIFLUID_DM */
    FLUID_TYPE_MAX_ID = 255   /* uint8 ceiling */
};

KOKKOS_INLINE_FUNCTION
bool same_lagrangian_fluid_id(unsigned char ft_i, unsigned char ft_j)
{
    return ft_i == ft_j;
}

KOKKOS_INLINE_FUNCTION
bool cross_lagrangian_fluid_pair_id(unsigned char ft_i, unsigned char ft_j)
{
    return ft_i != ft_j;
}

#endif
#endif /* MULTIFLUID_HELPERS_H */
