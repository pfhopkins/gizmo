/* gpu_morton_functions.h -- Step 13 Phase 6.5b
 *
 * KOKKOS_INLINE_FUNCTION primitives for Morton-key-based octree topology
 * construction.  These live in a header (rather than a .cc) so they can be
 * called from device kernels in other translation units without rdc/device
 * linking (per GIZMO's header-only GPU design constraint).
 *
 * Layered above gpu_morton.h (which provides the host-callable encode + sort
 * launchers).  This header is the device-side toolkit:
 *
 *   - 21-bit-per-axis Morton bit-spread / interleave (also re-exposed here
 *     so kernels in 6.5c can encode keys inline if they need to).
 *   - LCP (longest common prefix) bit-count between two 63-bit Morton keys.
 *   - Octant extraction at a given octree depth (0 = root).
 *   - 8-way split-range location: given a sorted (index -> key) view and
 *     a range [range_first, range_last), find the 8 child sub-ranges at a
 *     given parent depth d.  Populates child_starts[9] s.t. octant k spans
 *     [child_starts[k], child_starts[k+1]) within the sorted range.
 *
 * No external state, no device global variables, no mutation -- pure
 * device-side helpers.  Safe to include from any GPU TU under
 * OPENMP_GPU_OFFLOAD.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_MORTON_FUNCTIONS_H
#define GIZMO_GPU_MORTON_FUNCTIONS_H

#include <stdint.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>

/* Octree depth limit driven by 63-bit Morton (21 bits per axis = 21 levels).
 * Beyond this depth, all keys in a sub-range that haven't separated must be
 * collocated and will fall through to the RNG branch in 6.5c. */
#define GIZMO_GPU_MORTON_MAX_DEPTH 21

/* 21-bit-per-axis bit spread.  Identical to the static helper in gpu_morton.cc
 * but exposed here for cross-TU device-side use. */
KOKKOS_INLINE_FUNCTION uint64_t gpu_morton_spread21(uint64_t v) {
    v &= 0x1FFFFFull;
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}

KOKKOS_INLINE_FUNCTION uint64_t gpu_morton_encode63(uint32_t x, uint32_t y, uint32_t z) {
    return gpu_morton_spread21((uint64_t)x)
         | (gpu_morton_spread21((uint64_t)y) << 1)
         | (gpu_morton_spread21((uint64_t)z) << 2);
}

/* Count the number of leading bits two Morton keys share, in the high
 * 63 bits.  Returns 63 if a == b in the relevant bit range, 0 if the
 * top bit differs.  Implementation: XOR + count-leading-zeros, biased
 * to ignore bit 63 (we use 63-bit Morton). */
KOKKOS_INLINE_FUNCTION int gpu_morton_lcp_bits(uint64_t a, uint64_t b) {
    uint64_t x = a ^ b;
    if(x == 0) {return 63;}
    /* clz64.  __builtin_clzll is host-only; use Kokkos::bit_cast-free
     * intrinsic via Kokkos::Experimental::countl_zero.  Fall back to a
     * manual loop on backends that lack it. */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    int n = __clzll((unsigned long long)x);
#else
    int n = 0;
    /* Manual leading-zero count for the host backend (Kokkos OpenMP / Serial). */
    if((x & 0xFFFFFFFF00000000ull) == 0) {n += 32; x <<= 32;}
    if((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
    if((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
    if((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
    if((x & 0xC000000000000000ull) == 0) {n +=  2; x <<=  2;}
    if((x & 0x8000000000000000ull) == 0) {n +=  1;}
#endif
    /* Bit 63 (the unused high bit, since Morton is 63-bit) is always 0 in
     * both keys, so n is at least 1; subtract that bias to report shared
     * bits in the meaningful 63-bit range. */
    return (n > 0) ? (n - 1) : 0;
}

/* Return the 3-bit octant index (0..7) of `key` at octree depth `depth`
 * (0 = root, 1 = first split, ...).  Octant bits at depth d occupy Morton
 * positions [62 - 3d, 60 - 3d] (high z, then y, then x in the encoding
 * convention used by gpu_morton_encode63). */
KOKKOS_INLINE_FUNCTION int gpu_morton_octant_at_depth(uint64_t key, int depth) {
    int shift = 62 - 3 * depth;
    if(shift < 0) {return 0;}
    return (int)((key >> shift) & 0x7ull);
}

/* Locate 8-way split boundaries within a sorted Morton-key range.
 *
 *   sorted_idx[range_first..range_last)   -- particle indices in sorted order
 *   keys[i]                               -- Morton key for particle index i
 *   depth                                 -- octree depth of the parent node
 *                                            (octant bits read from depth+1)
 *
 * Output:
 *   child_starts[0..8]  -- offsets relative to range_first.
 *                          Octant k occupies sorted_idx[range_first + child_starts[k]
 *                                                       .. range_first + child_starts[k+1]).
 *                          child_starts[0] = 0; child_starts[8] = range_last - range_first.
 *
 * Implementation: linear scan, O(range_count).  range_count is bounded by
 * the per-topleaf particle count (~thousands), so a sequential scan inside
 * a per-topleaf team thread is the right granularity.  For larger ranges
 * a parallel binary-search variant is possible (Karras section 3.1) but
 * unneeded here.
 *
 * Pre-condition: keys are sorted within the range.  The split level is
 * `depth + 1` (i.e. we read the 3-bit octant for depth+1 from each key). */
KOKKOS_INLINE_FUNCTION void gpu_morton_split_8way(const int       *sorted_idx,
                                                  const uint64_t  *keys,
                                                  int              range_first,
                                                  int              range_last,
                                                  int              depth,
                                                  int              child_starts[9])
{
    int n = range_last - range_first;
    /* Initialize all boundaries to n (= empty trailing octants). */
    for(int k = 0; k < 9; k++) {child_starts[k] = n;}
    child_starts[0] = 0;
    if(n <= 0) {return;}

    int next_octant = 0;
    int child_depth = depth + 1;
    for(int j = 0; j < n; j++) {
        int      idx = sorted_idx[range_first + j];
        uint64_t k   = keys[idx];
        int      oct = gpu_morton_octant_at_depth(k, child_depth);
        /* Sorted order guarantees octant indices are non-decreasing.  Walk the
         * boundary cursor forward, marking each newly-entered octant's start. */
        while(next_octant <= oct) {
            child_starts[next_octant] = j;
            next_octant++;
        }
    }
    /* Fill in any remaining trailing octants with the end-of-range sentinel. */
    while(next_octant <= 8) {
        child_starts[next_octant] = n;
        next_octant++;
    }
}

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_MORTON_FUNCTIONS_H */
