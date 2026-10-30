/* domain_key_functions.h
 *
 * The one definition of the position -> integer-coordinate mapping behind every Peano-Hilbert and
 * Morton key in the code.  The domain decomposition decides which rank owns a particle from its key
 * (domain.cc), and the device tree build decides which top-leaf to bucket it under from the key it
 * forms itself (gpu_topology_build.cc).  The two have to agree bit for bit: a particle owned by one
 * rank and bucketed under a top-leaf another rank owns is detached from every rank's tree.  Both
 * sides, and the neighbour-list spatial index, take the mapping from here.
 *
 * Why one definition is necessary but not sufficient.  The mapping is a subtraction, a division and
 * an addition, each correctly rounded, so under a strict floating-point model the expression has no
 * freedom and every copy agrees.  Under -ffast-math the compiler may substitute a reciprocal and fuse
 * the multiply-add, and does so where that pays (a vectorised loop) and not where it does not (a
 * scalar call), so two expansions of the SAME source can differ in the last bit -- which the key
 * extraction then reads as a different cell.  Hence, in addition to this header: the host callers go
 * through one noinline emission (domain_peano_key_of_position in domain.cc, so no host loop can
 * carry its own contracted copy), and every tree build compares a sample of device keys against that
 * host emission (gpu_topology_build.cc) and requests a controlled stop if they disagree.  The
 * compile-time refusal of __FAST_MATH__ in declarations/typedefs.h remains in force.
 *
 * Header-only and Kokkos-free so host translation units can include it: KOKKOS_INLINE_FUNCTION is
 * the real macro when Kokkos_Core.hpp was included first, and plain inline otherwise.
 */

#ifndef GIZMO_DOMAIN_KEY_FUNCTIONS_H
#define GIZMO_DOMAIN_KEY_FUNCTIONS_H

#include <stdint.h>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* The coordinate the key is read from: (pos - corner) / len + 1.0, which lies in [1.0, 2.0) for a
 * position inside the extent, so that its mantissa is the position's fraction of the extent.
 * The sum is clamped, not the fraction: a fraction just below 1.0 can round the sum up to 2.0,
 * whose mantissa is zero -- the first cell instead of the last.  The tests are written so a NaN
 * coordinate takes the lower bound rather than passing through; no extent yet (len <= 0) maps to
 * the lower bound too.  Callers that must know whether a position was inside the extent test it
 * beforehand (position_outside_domain_extent); here it is only made to key somewhere valid. */
KOKKOS_INLINE_FUNCTION double domain_key_scaled_coordinate(double pos, double corner, double len)
{
    double f = (len > 0.0) ? ((pos - corner) / len) : 0.0;
    double s = f + 1.0;
    if(!(s >= 1.0)) {s = 1.0;}
    if(!(s < 2.0))  {s = 0x1.fffffffffffffp0;}   /* the largest double below 2.0 */
    return s;
}

/* The top `bits` mantissa bits of a double in [1.0, 2.0), as the integer coordinate in [0, 2^bits). */
KOKKOS_INLINE_FUNCTION uint64_t domain_key_integer_coordinate(double scaled, int bits)
{
    union {double d; uint64_t ull;} u;
    u.d = scaled;
    return (u.ull & 0xFFFFFFFFFFFFFull) >> (52 - bits);
}

/* Position -> the three integer coordinates every key is built from. */
KOKKOS_INLINE_FUNCTION void domain_key_integer_coordinates(double x, double y, double z,
                                                           double corner_x, double corner_y, double corner_z,
                                                           double len, int bits, uint64_t out[3])
{
    out[0] = domain_key_integer_coordinate(domain_key_scaled_coordinate(x, corner_x, len), bits);
    out[1] = domain_key_integer_coordinate(domain_key_scaled_coordinate(y, corner_y, len), bits);
    out[2] = domain_key_integer_coordinate(domain_key_scaled_coordinate(z, corner_z, len), bits);
}

#endif /* GIZMO_DOMAIN_KEY_FUNCTIONS_H */
