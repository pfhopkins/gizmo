/* gpu_morton_functions.h -- Step 13 Phase 6.5b/6.5c0
 *
 * KOKKOS_INLINE_FUNCTION primitives for Morton-key-based octree topology
 * construction.  These live in a header (rather than a .cc) so they can be
 * called from device kernels in other translation units without rdc/device
 * linking (per GIZMO's header-only GPU design constraint).
 *
 * Phase 6.5c0: upgraded from 63-bit (21 bits/axis) to 128-bit (42 bits/axis)
 * Morton precision to match GIZMO's CPU build (BITS_PER_DIMENSION = 42 in
 * declarations/typedefs.h).  Tree topology is now bit-equivalent to CPU
 * within FP/sort-stability tolerances on every supported simulation regime
 * (typical galaxy, normal high-res ~1e-9, hyper-refinement ~1e-12).
 *
 * The 128-bit Morton is stored as a (hi, lo) pair of uint64_t where each
 * half is a 63-bit Morton key over a 21-bit-per-axis half of the input
 * coordinates:
 *
 *     hi = encode63(x[41:21], y[41:21], z[41:21])    // top 21 bits per axis
 *     lo = encode63(x[20: 0], y[20: 0], z[20: 0])    // bot 21 bits per axis
 *
 * Lexicographic ordering on (hi, lo) is identical to natural Morton ordering
 * on the full 126-bit interleaved code.  Provable bit-positional argument:
 * the top 63 bits of the natural code are exactly the top 63 bits of `hi`;
 * the next 63 bits are exactly `lo`.
 *
 * Provides:
 *   - Morton128 struct with operator< / operator== (lexicographic).
 *   - Morton128Less device-callable functor for sort_by_key comparator.
 *   - 21-bit-per-axis bit-spread / 63-bit interleave helpers (private).
 *   - 128-bit Morton encode from 42-bit-per-axis integer coordinates.
 *   - LCP (longest common prefix) bit count, in [0, 126].
 *   - Octant extraction at a given octree depth (0 = root, max = 41).
 *   - 8-way split-range location within a sorted Morton range.
 *   - IEEE-754-mantissa-bit-trick double->42-bit-int conversion (matches
 *     the host domain_double_to_int).
 *
 * Pure / stateless / header-only -- no device linker requirements.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_MORTON_FUNCTIONS_H
#define GIZMO_GPU_MORTON_FUNCTIONS_H

#include <stdint.h>

#include <Kokkos_Core.hpp>
#include "../declarations/gpu_rng.h"

/* Octree depth limit driven by 126-bit Morton (42 bits per axis = 42 levels).
 * Beyond this depth, all keys in a sub-range that haven't separated must be
 * collocated and will fall through to the RNG branch in 6.5c4. */
#define GIZMO_GPU_MORTON_MAX_DEPTH 42

/* ------------------------- Morton128 storage type ------------------------- */

struct Morton128 {
    uint64_t hi;  /* top 63 useful bits  (positions 63..125 of full code) */
    uint64_t lo;  /* bot 63 useful bits  (positions  0.. 62 of full code) */

    KOKKOS_INLINE_FUNCTION bool operator<(const Morton128 &o) const {
        return (hi < o.hi) || (hi == o.hi && lo < o.lo);
    }
    KOKKOS_INLINE_FUNCTION bool operator==(const Morton128 &o) const {
        return hi == o.hi && lo == o.lo;
    }
    KOKKOS_INLINE_FUNCTION bool operator!=(const Morton128 &o) const {
        return !(*this == o);
    }
    KOKKOS_INLINE_FUNCTION bool operator<=(const Morton128 &o) const {
        return *this < o || *this == o;
    }
};

/* Comparator functor for Kokkos::Experimental::sort_by_key with comparator
 * overload.  Device-callable.  Lexicographic on (hi, lo). */
struct Morton128Less {
    KOKKOS_INLINE_FUNCTION bool operator()(const Morton128 &a, const Morton128 &b) const {
        return a < b;
    }
};

/* ----------------------- private 63-bit helpers --------------------------- */

/* 21-bit-per-axis bit spread: bit i of `v` (i in [0,21)) lands at position 3i. */
KOKKOS_INLINE_FUNCTION uint64_t gpu_morton_spread21(uint64_t v) {
    v &= 0x1FFFFFull;
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}

/* 63-bit Morton encode of 21-bit-per-axis (x, y, z).  Bit layout:
 *   bit 0 = x[0],  bit 1 = y[0],  bit 2 = z[0],  bit 3 = x[1], ...,
 *   bit 60 = x[20], bit 61 = y[20], bit 62 = z[20].  Bit 63 unused. */
KOKKOS_INLINE_FUNCTION uint64_t gpu_morton_encode63(uint32_t x, uint32_t y, uint32_t z) {
    return gpu_morton_spread21((uint64_t)x)
         | (gpu_morton_spread21((uint64_t)y) << 1)
         | (gpu_morton_spread21((uint64_t)z) << 2);
}

/* ------------------------- 128-bit Morton encode -------------------------- */

/* 128-bit Morton encode of 42-bit-per-axis (x, y, z).  Returns Morton128
 * with hi = encode63(top 21 bits of each axis), lo = encode63(bottom 21
 * bits of each axis).  Lexicographic on the result is full-precision
 * Morton ordering. */
KOKKOS_INLINE_FUNCTION Morton128 gpu_morton_encode128(uint64_t x42, uint64_t y42, uint64_t z42) {
    Morton128 k;
    uint32_t xh = (uint32_t)((x42 >> 21) & 0x1FFFFFull);
    uint32_t yh = (uint32_t)((y42 >> 21) & 0x1FFFFFull);
    uint32_t zh = (uint32_t)((z42 >> 21) & 0x1FFFFFull);
    uint32_t xl = (uint32_t)( x42        & 0x1FFFFFull);
    uint32_t yl = (uint32_t)( y42        & 0x1FFFFFull);
    uint32_t zl = (uint32_t)( z42        & 0x1FFFFFull);
    k.hi = gpu_morton_encode63(xh, yh, zh);
    k.lo = gpu_morton_encode63(xl, yl, zl);
    return k;
}

/* IEEE-754 mantissa-bit-trick conversion of a double in [1.0, 2.0) to a
 * 42-bit unsigned integer in [0, 2^42).  Matches host domain_double_to_int
 * with BITS_PER_DIMENSION = 42 (domain/domain.cc:2711).
 *
 * Caller must arrange the input as (Pos - DomainCorner) / DomainLen + 1.0
 * to land in [1.0, 2.0); positions outside the domain box clamp at the
 * caller layer. */
KOKKOS_INLINE_FUNCTION uint64_t gpu_morton_double_to_int42(double d) {
    union { double d; uint64_t ull; } u;
    u.d = d;
    return (u.ull & 0xFFFFFFFFFFFFFull) >> (52 - 42);
}

/* ------------------------- LCP / octant / split --------------------------- */

/* clz64: count leading zeros of a uint64_t. */
KOKKOS_INLINE_FUNCTION int gpu_morton_clz64(uint64_t x) {
    if(x == 0) {return 64;}
#if defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__
    return __clzll((unsigned long long)x);
#else
    int n = 0;
    if((x & 0xFFFFFFFF00000000ull) == 0) {n += 32; x <<= 32;}
    if((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
    if((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
    if((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
    if((x & 0xC000000000000000ull) == 0) {n +=  2; x <<=  2;}
    if((x & 0x8000000000000000ull) == 0) {n +=  1;}
    return n;
#endif
}

/* Count the number of leading bits two 128-bit Morton keys share, in the
 * 126 useful bits.  Returns 126 if a == b (collocated).  Bit 63 of each
 * half is unused (always 0), so we subtract that bias from each half's
 * leading-zero count. */
KOKKOS_INLINE_FUNCTION int gpu_morton_lcp_bits(const Morton128 &a, const Morton128 &b) {
    uint64_t xhi = a.hi ^ b.hi;
    if(xhi != 0) {
        int n_hi = gpu_morton_clz64(xhi);
        return (n_hi > 0) ? (n_hi - 1) : 0;  /* bias for unused bit 63 of hi */
    }
    /* hi matches exactly -> 63 common high bits; check lo. */
    uint64_t xlo = a.lo ^ b.lo;
    if(xlo == 0) {return 126;}
    int n_lo = gpu_morton_clz64(xlo);
    return 63 + ((n_lo > 0) ? (n_lo - 1) : 0);
}

/* Return the 3-bit octant index (0..7) of `key` at octree depth `depth`.
 * depth = 0 reads the topmost 3 bits of the full 126-bit Morton (= top
 * 3 bits of hi at positions 62..60).  Each subsequent depth descends 3
 * bits.  For depth in [0, 21) reads from hi; for [21, 42) reads from lo. */
KOKKOS_INLINE_FUNCTION int gpu_morton_octant_at_depth(const Morton128 &key, int depth) {
    if(depth < 0) {return 0;}
    if(depth < 21) {
        int shift = 60 - 3 * depth;
        return (int)((key.hi >> shift) & 0x7ull);
    }
    if(depth < 42) {
        int shift = 60 - 3 * (depth - 21);
        return (int)((key.lo >> shift) & 0x7ull);
    }
    return 0;
}

/* Locate 8-way split boundaries within a sorted Morton-key range.
 *
 *   sorted_idx[range_first..range_last)   particle indices in sorted order
 *   keys[i]                                Morton128 key for particle index i
 *   split_level                            octree level whose 3 octant bits
 *                                          discriminate the children being
 *                                          produced.  Equivalently: the depth
 *                                          of the parent node whose children
 *                                          we are placing.  level 0 picks
 *                                          among the 8 children of the root
 *                                          (top 3 bits of full 126-bit code).
 *
 * Output:
 *   child_starts[0..8] -- offsets relative to range_first.  Octant k spans
 *                         sorted_idx[range_first + child_starts[k] ..
 *                                    range_first + child_starts[k+1]).
 *                         child_starts[0] = 0; child_starts[8] = range_count.
 *
 * Linear scan, O(range_count).  Pre-condition: keys are sorted within the
 * range. */
KOKKOS_INLINE_FUNCTION void gpu_morton_split_8way(const int       *sorted_idx,
                                                  const Morton128 *keys,
                                                  int              range_first,
                                                  int              range_last,
                                                  int              split_level,
                                                  int              child_starts[9])
{
    int n = range_last - range_first;
    for(int k = 0; k < 9; k++) {child_starts[k] = n;}
    child_starts[0] = 0;
    if(n <= 0) {return;}

    int next_octant = 0;
    for(int j = 0; j < n; j++) {
        int idx = sorted_idx[range_first + j];
        int oct = gpu_morton_octant_at_depth(keys[idx], split_level);
        while(next_octant <= oct) {
            child_starts[next_octant] = j;
            next_octant++;
        }
    }
    while(next_octant <= 8) {
        child_starts[next_octant] = n;
        next_octant++;
    }
}

/* In-place reshuffle of a sorted_idx range using random per-particle octants
 * (collocation handler).  Mirrors CPU forcetree.cc:267-273 random subnode
 * assignment when Nodes[th].len < EPSILON_FOR_TREERND_SUBNODE_SPLITTING *
 * split_scale: each particle gets a random octant from get_random_number(P[i].ID).
 *
 *   sorted_idx_range[0..count) -- in-place: read particle indices, write
 *                                 reshuffled by random octant.
 *   P_id_of(idx)               -- functor returning P[idx].ID for the RNG seed.
 *   counter                    -- RNG counter (e.g. parent_depth) so each
 *                                 BFS level samples an independent stream.
 *   child_starts[0..8]         -- output octant boundaries.
 *
 * Uses a thread-local scratch of fixed size MAX_COLLOC_RANGE.  If the range
 * exceeds this, returns -1 (caller signals failure; user must use a different
 * IC or upgrade scratch budget).  On success returns 0. */
#define GIZMO_GPU_MORTON_COLLOC_SCRATCH 512

template <class IDFunc>
KOKKOS_INLINE_FUNCTION int gpu_morton_split_8way_random_inplace(
    int           *sorted_idx_range,
    int            count,
    IDFunc         id_of,
    uint64_t       counter,
    int            child_starts[9])
{
    if(count <= 0) {
        for(int k = 0; k < 9; k++) {child_starts[k] = 0;}
        return 0;
    }
    if(count > GIZMO_GPU_MORTON_COLLOC_SCRATCH) {return -1;}

    int      idx_buf[GIZMO_GPU_MORTON_COLLOC_SCRATCH];
    unsigned char oct_buf[GIZMO_GPU_MORTON_COLLOC_SCRATCH];
    int      counts[8] = {0,0,0,0,0,0,0,0};

    /* 1. Snapshot input + assign random octant per particle, count occupants. */
    for(int j = 0; j < count; j++) {
        int idx = sorted_idx_range[j];
        idx_buf[j] = idx;
        uint64_t pid = (uint64_t) id_of(idx);
        double   r   = gizmo_gpu_rand_double(pid, counter);
        int      o   = (int)(8.0 * r);
        if(o < 0) {o = 0;}
        if(o > 7) {o = 7;}
        oct_buf[j] = (unsigned char)o;
        counts[o]++;
    }

    /* 2. Exclusive scan -> child_starts. */
    int sum = 0;
    for(int k = 0; k < 8; k++) {
        child_starts[k] = sum;
        sum += counts[k];
    }
    child_starts[8] = sum;

    /* 3. Scatter back into sorted_idx_range using cursors per octant. */
    int cursors[8];
    for(int k = 0; k < 8; k++) {cursors[k] = child_starts[k];}
    for(int j = 0; j < count; j++) {
        int o = (int)oct_buf[j];
        sorted_idx_range[cursors[o]] = idx_buf[j];
        cursors[o]++;
    }
    return 0;
}


#endif /* GIZMO_GPU_MORTON_FUNCTIONS_H */
