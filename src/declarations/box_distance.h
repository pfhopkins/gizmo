#ifndef BOX_DISTANCE_H
#define BOX_DISTANCE_H

/* Generic box nearest-image distance helpers. These are thin inline wrappers over the
 * canonical NEAREST_XYZ / NGB_PERIODIC_BOX_LONG_* macros (declarations/macros.h), so the
 * box-wrapping math (per-axis periodicity, reflect/outflow exclusions, BOX_LONG_*,
 * shearing-box offset, Y-before-X/Z order) has a single source.
 *
 * These carry NO gravity policy: they wrap whenever the box is periodic, exactly like the
 * neighbor/hydro code. The gravity-specific gate (skip wrapping under GRAVITY_NOT_PERIODIC)
 * lives in gravity_box_distance.h. Any consumer needing generic box geometry uses these
 * directly; gravity consumers use the gravity_box_* wrappers.
 *
 * Include AFTER declarations/allvars.h (which pulls in macros.h). The guard below fails the
 * build loudly if those macros are not yet visible. */

#if !defined(NEAREST_XYZ) || !defined(NGB_PERIODIC_BOX_LONG_X) || !defined(boxSize_X) || !defined(boxHalf_X)
#error "box_distance.h requires the box-wrap macros and box-size macros; include declarations/allvars.h (which includes macros.h) before box_distance.h"
#endif

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* In-place signed nearest-image wrap of a displacement (dx,dy,dz). Reproduces NEAREST_XYZ.
 * sign encodes the displacement direction convention (pos[local]-pos[neighbor] vs the
 * reverse); the current wrap math does not branch on it but it is carried for callers and
 * forward compatibility. */
KOKKOS_INLINE_FUNCTION void box_nearest_image(double &dx, double &dy, double &dz, int sign)
{
    (void)sign;
    NEAREST_XYZ(dx, dy, dz, sign);
}

/* Per-axis absolute wrapped distance. Reproduce NGB_PERIODIC_BOX_LONG_{X,Y,Z}. The macros
 * use a local temporary (xtmp); each helper declares its own. The Y variant uses dx for the
 * shearing-box offset, so all three take the full displacement. */
KOKKOS_INLINE_FUNCTION double box_long_abs_x(double dx, double dy, double dz, int sign)
{
    (void)sign; (void)dy; (void)dz;
    double xtmp = 0.0; (void)xtmp;
    return NGB_PERIODIC_BOX_LONG_X(dx, dy, dz, sign);
}
KOKKOS_INLINE_FUNCTION double box_long_abs_y(double dx, double dy, double dz, int sign)
{
    (void)sign; (void)dz;
    double xtmp = 0.0; (void)xtmp;
    return NGB_PERIODIC_BOX_LONG_Y(dx, dy, dz, sign);
}
KOKKOS_INLINE_FUNCTION double box_long_abs_z(double dx, double dy, double dz, int sign)
{
    (void)sign; (void)dx; (void)dy;
    double xtmp = 0.0; (void)xtmp;
    return NGB_PERIODIC_BOX_LONG_Z(dx, dy, dz, sign);
}

#endif /* BOX_DISTANCE_H */
