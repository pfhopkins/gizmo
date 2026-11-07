/* domain_key_functions.h
 *
 * The one definition of the position -> integer-coordinate mapping behind every Peano-Hilbert and
 * Morton key in the code.  The domain decomposition decides which rank owns a particle from its key
 * (domain.cc), and the device tree build decides which top-leaf to bucket it under from the key it
 * forms itself (gpu_topology_build.cc).  The two have to agree bit for bit: a particle owned by one
 * rank and bucketed under a top-leaf another rank owns is detached from every rank's tree.  Both
 * sides, and the neighbour-list spatial index, take the mapping from here.
 *
 * Under a strict floating-point model the mapping (a subtraction, a division and an addition, each
 * correctly rounded) has no freedom, so every expansion of this source agrees; a fast model that
 * licenses reciprocals or fused multiply-adds would not, which is why declarations/typedefs.h refuses
 * to compile under __FAST_MATH__.
 *
 * Header-only and Kokkos-free so host translation units can include it.
 */

#ifndef GIZMO_DOMAIN_KEY_FUNCTIONS_H
#define GIZMO_DOMAIN_KEY_FUNCTIONS_H

#include <stdint.h>

/* A private annotation, keyed on the compiler rather than on whether Kokkos was included first, so the functions are
 * device-callable wherever a device compiler sees them and this header never defines Kokkos's own macro. */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define DOMAIN_KEY_FUNCTION __host__ __device__ inline
#else
#define DOMAIN_KEY_FUNCTION inline
#endif

/* The coordinate the key is read from: (pos - corner) / len + 1.0, which lies in [1.0, 2.0) for a
 * position inside the extent, so that its mantissa is the position's fraction of the extent.
 * The sum is clamped, not the fraction: a fraction just below 1.0 can round the sum up to 2.0,
 * whose mantissa is zero -- the first cell instead of the last.  The tests are written so a NaN
 * coordinate takes the lower bound rather than passing through; no extent yet (len <= 0) maps to
 * the lower bound too.  Callers that must know whether a position was inside the extent test it
 * beforehand (position_outside_domain_extent); here it is only made to key somewhere valid. */
DOMAIN_KEY_FUNCTION double domain_key_scaled_coordinate(double pos, double corner, double len)
{
    double f = (len > 0.0) ? ((pos - corner) / len) : 0.0;
    double s = f + 1.0;
    if(!(s >= 1.0)) {s = 1.0;}
    if(!(s < 2.0))  {s = 0x1.fffffffffffffp0;}   /* the largest double below 2.0 */
    return s;
}

/* The top `bits` mantissa bits of a double in [1.0, 2.0), as the integer coordinate in [0, 2^bits). */
DOMAIN_KEY_FUNCTION uint64_t domain_key_integer_coordinate(double scaled, int bits)
{
    union {double d; uint64_t ull;} u;
    u.d = scaled;
    return (u.ull & 0xFFFFFFFFFFFFFull) >> (52 - bits);
}

/* Position -> the three integer coordinates every key is built from. */
DOMAIN_KEY_FUNCTION void domain_key_integer_coordinates(double x, double y, double z,
                                                           double corner_x, double corner_y, double corner_z,
                                                           double len, int bits, uint64_t out[3])
{
    out[0] = domain_key_integer_coordinate(domain_key_scaled_coordinate(x, corner_x, len), bits);
    out[1] = domain_key_integer_coordinate(domain_key_scaled_coordinate(y, corner_y, len), bits);
    out[2] = domain_key_integer_coordinate(domain_key_scaled_coordinate(z, corner_z, len), bits);
}

#undef DOMAIN_KEY_FUNCTION

#endif /* GIZMO_DOMAIN_KEY_FUNCTIONS_H */
