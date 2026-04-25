/* gpu_peano_walk_functions.h -- Step 13 Phase 6.5c1
 *
 * KOKKOS_INLINE_FUNCTION primitives for device-side Peano-Hilbert key
 * computation and TopNodes-tree walking.  Mirrors the host implementation
 * in system/peano.cc (peano_and_morton_key) and gravity/forcetree.cc lines
 * 236-249 (TopNodes walk + DomainNodeIndex lookup).
 *
 * Why device-side: Phase 6.5 GPU tree-build kernel needs to assign each
 * particle to its topleaf before sorting and emitting topology.  Doing
 * this on host would be a silent rollback of the Phase 6 goal of
 * eliminating per-tree-build CPU cost.  Per-particle Peano walk is
 * embarrassingly parallel and well-suited to GPU.
 *
 * Lookup tables: subpix3[48][8] and rottable3[48][8] from system/peano.cc
 * are replicated here as static constexpr arrays.  Each is 384 bytes; both
 * fit comfortably in __constant__ / read-only cache on any backend.
 *
 * peano1D type: when BITS_PER_DIMENSION > 21, peanokey is __int128 and
 * peano1D is uint64_t.  __int128 is a standard nvcc/nvc++ extension and
 * works in Kokkos lambdas; we use it directly.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_PEANO_WALK_FUNCTIONS_H
#define GIZMO_GPU_PEANO_WALK_FUNCTIONS_H

#include <stdint.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>

#include "../declarations/typedefs.h"
#include "../declarations/allvars.h"   /* topnode_data definition */
#include "gpu_morton_functions.h"      /* Morton128 + gpu_morton_double_to_int42 */

/* ----------------------- Peano-Hilbert lookup tables ---------------------- */
/* Replicated verbatim from system/peano.cc. */

static constexpr unsigned char gpu_peano_rottable3[48][8] = {
  {36, 28, 25, 27, 10, 10, 25, 27},
  {29, 11, 24, 24, 37, 11, 26, 26},
  {8, 8, 25, 27, 30, 38, 25, 27},
  {9, 39, 24, 24, 9, 31, 26, 26},
  {40, 24, 44, 32, 40, 6, 44, 6},
  {25, 7, 33, 7, 41, 41, 45, 45},
  {4, 42, 4, 46, 26, 42, 34, 46},
  {43, 43, 47, 47, 5, 27, 5, 35},
  {33, 35, 36, 28, 33, 35, 2, 2},
  {32, 32, 29, 3, 34, 34, 37, 3},
  {33, 35, 0, 0, 33, 35, 30, 38},
  {32, 32, 1, 39, 34, 34, 1, 31},
  {24, 42, 32, 46, 14, 42, 14, 46},
  {43, 43, 47, 47, 25, 15, 33, 15},
  {40, 12, 44, 12, 40, 26, 44, 34},
  {13, 27, 13, 35, 41, 41, 45, 45},
  {28, 41, 28, 22, 38, 43, 38, 22},
  {42, 40, 23, 23, 29, 39, 29, 39},
  {41, 36, 20, 36, 43, 30, 20, 30},
  {37, 31, 37, 31, 42, 40, 21, 21},
  {28, 18, 28, 45, 38, 18, 38, 47},
  {19, 19, 46, 44, 29, 39, 29, 39},
  {16, 36, 45, 36, 16, 30, 47, 30},
  {37, 31, 37, 31, 17, 17, 46, 44},
  {12, 4, 1, 3, 34, 34, 1, 3},
  {5, 35, 0, 0, 13, 35, 2, 2},
  {32, 32, 1, 3, 6, 14, 1, 3},
  {33, 15, 0, 0, 33, 7, 2, 2},
  {16, 0, 20, 8, 16, 30, 20, 30},
  {1, 31, 9, 31, 17, 17, 21, 21},
  {28, 18, 28, 22, 2, 18, 10, 22},
  {19, 19, 23, 23, 29, 3, 29, 11},
  {9, 11, 12, 4, 9, 11, 26, 26},
  {8, 8, 5, 27, 10, 10, 13, 27},
  {9, 11, 24, 24, 9, 11, 6, 14},
  {8, 8, 25, 15, 10, 10, 25, 7},
  {0, 18, 8, 22, 38, 18, 38, 22},
  {19, 19, 23, 23, 1, 39, 9, 39},
  {16, 36, 20, 36, 16, 2, 20, 10},
  {37, 3, 37, 11, 17, 17, 21, 21},
  {4, 17, 4, 46, 14, 19, 14, 46},
  {18, 16, 47, 47, 5, 15, 5, 15},
  {17, 12, 44, 12, 19, 6, 44, 6},
  {13, 7, 13, 7, 18, 16, 45, 45},
  {4, 42, 4, 21, 14, 42, 14, 23},
  {43, 43, 22, 20, 5, 15, 5, 15},
  {40, 12, 21, 12, 40, 6, 23, 6},
  {13, 7, 13, 7, 41, 41, 22, 20}
};

static constexpr unsigned char gpu_peano_subpix3[48][8] = {
  {0, 7, 1, 6, 3, 4, 2, 5},
  {7, 4, 6, 5, 0, 3, 1, 2},
  {4, 3, 5, 2, 7, 0, 6, 1},
  {3, 0, 2, 1, 4, 7, 5, 6},
  {1, 0, 6, 7, 2, 3, 5, 4},
  {0, 3, 7, 4, 1, 2, 6, 5},
  {3, 2, 4, 5, 0, 1, 7, 6},
  {2, 1, 5, 6, 3, 0, 4, 7},
  {6, 1, 7, 0, 5, 2, 4, 3},
  {1, 2, 0, 3, 6, 5, 7, 4},
  {2, 5, 3, 4, 1, 6, 0, 7},
  {5, 6, 4, 7, 2, 1, 3, 0},
  {7, 6, 0, 1, 4, 5, 3, 2},
  {6, 5, 1, 2, 7, 4, 0, 3},
  {5, 4, 2, 3, 6, 7, 1, 0},
  {4, 7, 3, 0, 5, 6, 2, 1},
  {6, 7, 5, 4, 1, 0, 2, 3},
  {7, 0, 4, 3, 6, 1, 5, 2},
  {0, 1, 3, 2, 7, 6, 4, 5},
  {1, 6, 2, 5, 0, 7, 3, 4},
  {2, 3, 1, 0, 5, 4, 6, 7},
  {3, 4, 0, 7, 2, 5, 1, 6},
  {4, 5, 7, 6, 3, 2, 0, 1},
  {5, 2, 6, 1, 4, 3, 7, 0},
  {7, 0, 6, 1, 4, 3, 5, 2},
  {0, 3, 1, 2, 7, 4, 6, 5},
  {3, 4, 2, 5, 0, 7, 1, 6},
  {4, 7, 5, 6, 3, 0, 2, 1},
  {6, 7, 1, 0, 5, 4, 2, 3},
  {7, 4, 0, 3, 6, 5, 1, 2},
  {4, 5, 3, 2, 7, 6, 0, 1},
  {5, 6, 2, 1, 4, 7, 3, 0},
  {1, 6, 0, 7, 2, 5, 3, 4},
  {6, 5, 7, 4, 1, 2, 0, 3},
  {5, 2, 4, 3, 6, 1, 7, 0},
  {2, 1, 3, 0, 5, 6, 4, 7},
  {0, 1, 7, 6, 3, 2, 4, 5},
  {1, 2, 6, 5, 0, 3, 7, 4},
  {2, 3, 5, 4, 1, 0, 6, 7},
  {3, 0, 4, 7, 2, 1, 5, 6},
  {1, 0, 2, 3, 6, 7, 5, 4},
  {0, 7, 3, 4, 1, 6, 2, 5},
  {7, 6, 4, 5, 0, 1, 3, 2},
  {6, 1, 5, 2, 7, 0, 4, 3},
  {5, 4, 6, 7, 2, 3, 1, 0},
  {4, 3, 7, 0, 5, 2, 6, 1},
  {3, 2, 0, 1, 4, 5, 7, 6},
  {2, 5, 1, 6, 3, 4, 0, 7}
};

/* ---------------------- Peano + Morton key computation -------------------- */

/* Compute both the Peano-Hilbert key and the Morton (Z-order) key for an
 * integer triplet (x, y, z), each with `bits` bits of precision.  Mirrors
 * system/peano.cc:331 (peano_and_morton_key) bit-for-bit.
 *
 * Caller responsibility: x, y, z in [0, 2^bits).  For the GIZMO build
 * we use bits = BITS_PER_DIMENSION = 42.
 *
 * Output Morton is returned as Morton128 (lexicographic-equivalent to the
 * full 126-bit code, see gpu_morton_functions.h). */
KOKKOS_INLINE_FUNCTION peanokey gpu_peano_and_morton_key(uint64_t  x,
                                                         uint64_t  y,
                                                         uint64_t  z,
                                                         int       bits,
                                                         Morton128 *morton_out)
{
    unsigned char rotation = 0;
    peanokey      key      = 0;
    Morton128     morton   = {0, 0};

    /* mask iterates from high bit (position bits-1) downward.  At each step,
     * append 3 bits to both keys.  For bits = 42 the loop runs 42 times,
     * producing 126-bit keys in 128-bit storage. */
    for(int b = bits - 1; b >= 0; b--) {
        uint64_t mb = ((uint64_t)1) << b;
        unsigned char pix = (unsigned char)(((x & mb) ? 4 : 0)
                                          | ((y & mb) ? 2 : 0)
                                          | ((z & mb) ? 1 : 0));

        /* Peano key: shift up 3, OR in subpix3 lookup. */
        key = (peanokey)(key << 3);
        key = (peanokey)(key | (peanokey)gpu_peano_subpix3[rotation][pix]);
        rotation = gpu_peano_rottable3[rotation][pix];

        /* Morton key (Z-order, no rotation): shift up 3, OR in (z|y|x) bits.
         * Maintain (hi, lo) lexicographically: shift hi left 3 bits, bringing
         * in the top 3 bits of lo; shift lo left 3 bits and OR in pix. */
        uint64_t carry = morton.lo >> (64 - 3);
        morton.hi = (morton.hi << 3) | carry;
        morton.lo = (morton.lo << 3) | (uint64_t)pix;
    }
    if(morton_out) {*morton_out = morton;}
    return key;
}

/* Walk the TopNodes tree using the Peano-Hilbert key and return the
 * topleaf id (0..NTopleaves).  Mirrors gravity/forcetree.cc lines 243-249.
 *
 *   tn         -- device-accessible TopNodes mirror
 *   key        -- particle's Peano-Hilbert key
 *
 * Returns leaf id; caller can index DomainNodeIndex[leaf] for the absolute
 * Nodes[] slot if needed.  Invariant: tn[no].Daughter < 0 marks a leaf
 * in the topnode tree. */
KOKKOS_INLINE_FUNCTION int gpu_topleaf_for_key(const struct topnode_data *tn,
                                               peanokey                   key)
{
    int no = 0;
    while(tn[no].Daughter >= 0) {
        peanokey rel  = (peanokey)(key - tn[no].StartKey);
        peanokey size = tn[no].Size;
        peanokey step = (peanokey)(size / 8);
        int      child= (int)(rel / step);
        no = tn[no].Daughter + child;
    }
    return tn[no].Leaf;
}

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_PEANO_WALK_FUNCTIONS_H */
