#ifndef GRAVITY_BOX_DISTANCE_H
#define GRAVITY_BOX_DISTANCE_H

/* Gravity-policy box-distance helpers. These apply the gravity periodicity gate
 * (wrap only when the box is periodic AND gravity is periodic) and then defer to the
 * generic box_* geometry helpers. This is the single home for the gravity wrap policy:
 * the CPU GRAVITY_NEAREST_XYZ / GRAVITY_NGB_PERIODIC_BOX_LONG_* macros and the GPU tree
 * walks both route through these, so CPU and GPU use identical box-distance behavior.
 *
 * Include AFTER declarations/allvars.h (and, in GPU translation units, after the
 * per-TU All-mirror) so the box-size macros resolve on host and device. */

#include "../declarations/box_distance.h"

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* In-place signed nearest-image wrap for gravity. No-op when gravity is non-periodic
 * (matches the empty GRAVITY_NEAREST_XYZ macro). */
KOKKOS_INLINE_FUNCTION void gravity_box_nearest_image(double &dx, double &dy, double &dz, int sign)
{
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
    box_nearest_image(dx, dy, dz, sign);
#else
    (void)dx; (void)dy; (void)dz; (void)sign;
#endif
}

/* Per-axis absolute wrapped distance for gravity. Returns the plain absolute value of the
 * component when gravity is non-periodic (matches GRAVITY_NGB_PERIODIC_BOX_LONG_*). */
KOKKOS_INLINE_FUNCTION double gravity_box_long_abs_x(double dx, double dy, double dz, int sign)
{
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
    return box_long_abs_x(dx, dy, dz, sign);
#else
    (void)dy; (void)dz; (void)sign; return (dx < 0.0) ? -dx : dx;
#endif
}
KOKKOS_INLINE_FUNCTION double gravity_box_long_abs_y(double dx, double dy, double dz, int sign)
{
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
    return box_long_abs_y(dx, dy, dz, sign);
#else
    (void)dx; (void)dz; (void)sign; return (dy < 0.0) ? -dy : dy;
#endif
}
KOKKOS_INLINE_FUNCTION double gravity_box_long_abs_z(double dx, double dy, double dz, int sign)
{
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
    return box_long_abs_z(dx, dy, dz, sign);
#else
    (void)dx; (void)dy; (void)sign; return (dz < 0.0) ? -dz : dz;
#endif
}

#endif /* GRAVITY_BOX_DISTANCE_H */
