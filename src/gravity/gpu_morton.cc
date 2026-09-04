/* gpu_morton.cc
 *
 * 128-bit Morton encoding (42 bits/axis) + parallel sort infrastructure for
 * the GPU tree-build insertion kernel.  See gpu_morton.h for the API and
 * integration plan.
 *
 * Keys store as Morton128 = (uint64_t hi, uint64_t lo)
 * to match GIZMO's CPU build precision (BITS_PER_DIMENSION = 42).  Encoding
 * uses the same IEEE-754 mantissa-bit-trick as the host
 * domain_double_to_int (domain/domain.cc:2711) for bit-equivalent integer
 * coordinate extraction.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_morton.h"
#include "gpu_morton_functions.h"


namespace {

static Morton128 *g_morton_keys     = NULL;  /* SharedSpace, [g_morton_keys_cap] */
static int        g_morton_keys_cap = 0;

}  /* anonymous namespace */

extern "C" struct Morton128 *gpu_morton_keys_acquire(int npart)
{
    if(npart <= 0) {return g_morton_keys;}
    if(g_morton_keys_cap < npart) {
        if(g_morton_keys) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_morton_keys);
            g_morton_keys = NULL;
        }
        g_morton_keys = (Morton128 *) gizmo_gpu_alloc_shared((long)npart * sizeof(Morton128), "treescratch_build_morton");
        if(!g_morton_keys) {
            printf("gpu_morton: keys alloc failed (npart=%d)\n", npart);
            g_morton_keys_cap = 0;
            return NULL;
        }
        g_morton_keys_cap = npart;
    }
    return g_morton_keys;
}

extern "C" int gpu_morton_compute_global_keys(int npart)
{
    if(npart <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    if(!gpu_morton_keys_acquire(npart)) {return 1;}

    /* Acquire the particles arena.  Caller is expected to have an active
     * arena already (this is called from inside the gravity-tree build,
     * which arena-acquires earlier), but the call is idempotent. */
    gpu_particles_arena_set_site("gpu_morton_compute_global_keys");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {
        printf("gpu_morton: P_dev null (npart=%d)\n", npart);
        return 1;
    }

    /* Capture domain bounds.  DomainCorner / DomainLen are host globals not
     * mirrored in All_dev; pass by value into the kernel.  Match host
     * convention from domain.cc:496-498 / forcetree.cc:236-238: feed
     * domain_double_to_int the value (Pos - DomainCorner)/DomainLen + 1.0
     * (in [1.0, 2.0)), and let the IEEE-754 mantissa-bit-trick extract
     * 42 bits of fractional position. */
    const double dc0 = DomainCorner[0];
    const double dc1 = DomainCorner[1];
    const double dc2 = DomainCorner[2];
    const double dlen = DomainLen;
    /* DIVIDE, do not multiply by a precomputed reciprocal. The host computes
     *     domain_double_to_int(((Pos[k] - DomainCorner[k]) / DomainLen) + 1.0)
     * (domain.cc, one rounding). x * (1/d) rounds twice -- once forming the reciprocal, once in the
     * product -- and is a DIFFERENT operation in IEEE-754, needing no -ffast-math to diverge. Measured
     * over 2e5 uniform positions with a representative DomainLen, ~0.004% of particles land on a
     * different 42-bit integer (always +/-1 in the last bit); at 1e7 particles that is ~350 per call.
     * The bit trick in gpu_morton_double_to_int42 is exact and cannot diverge -- only this argument can,
     * which is why the fix belongs here and not there.
     * Today the keys only order particles within a topleaf, so the cost is reproducibility: reordering
     * adjacent particles changes LBVH structure and force summation order, which is what a
     * "bitwise identical CPU vs GPU" claim rests on. If gpu_morton_compute_global_keys is ever wired
     * into the tree build (its comment says that was the intent) the same divergence becomes an
     * ownership disagreement between the CPU domain decomposition and the GPU tree -- the Father[]
     * orphaning failure with a host/device axis. */

    Morton128 *keys = g_morton_keys;
    Kokkos::parallel_for("morton_encode_global", npart, KOKKOS_LAMBDA(int i) {
        double fx = (dlen > 0.0) ? ((P_dev[i].Pos[0] - dc0) / dlen) : 0.0;
        double fy = (dlen > 0.0) ? ((P_dev[i].Pos[1] - dc1) / dlen) : 0.0;
        double fz = (dlen > 0.0) ? ((P_dev[i].Pos[2] - dc2) / dlen) : 0.0;
        /* Clamp into [0, 1) so that (frac + 1.0) lies in [1.0, 2.0).
         * Domain decomp pre-wraps periodic cases; this clamp protects
         * against rare boundary FP drift. */
        if(fx < 0.0) {fx = 0.0;} if(fx >= 1.0) {fx = 0.99999999999999988897;}
        if(fy < 0.0) {fy = 0.0;} if(fy >= 1.0) {fy = 0.99999999999999988897;}
        if(fz < 0.0) {fz = 0.0;} if(fz >= 1.0) {fz = 0.99999999999999988897;}
        uint64_t ix = gpu_morton_double_to_int42(fx + 1.0);
        uint64_t iy = gpu_morton_double_to_int42(fy + 1.0);
        uint64_t iz = gpu_morton_double_to_int42(fz + 1.0);
        keys[i] = gpu_morton_encode128(ix, iy, iz);
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("morton_encode_global", npart);
    return 0;
}

extern "C" int gpu_morton_sort_indices(int count, int *indices_inout)
{
    if(count <= 1) {return 0;}
    if(!g_morton_keys || g_morton_keys_cap <= 0) {
        printf("gpu_morton_sort_indices: keys not populated\n");
        return 1;
    }
    if(!indices_inout) {
        printf("gpu_morton_sort_indices: indices_inout null\n");
        return 1;
    }

    using ExSpace  = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExSpace::memory_space;

    /* Device-local scratch: gather (key, value) pairs and sort. */
    Kokkos::View<Morton128*, MemSpace> keys_work("morton_sort_keys", count);
    Kokkos::View<int*,       MemSpace> vals_work("morton_sort_vals", count);

    int              keys_cap = g_morton_keys_cap;
    const Morton128 *keys_src = g_morton_keys;
    Kokkos::parallel_for("morton_sort_gather", count, KOKKOS_LAMBDA(int j) {
        int idx = indices_inout[j];
        Morton128 k;
        if(idx >= 0 && idx < keys_cap) {
            k = keys_src[idx];
        } else {
            k.hi = 0; k.lo = 0;
        }
        keys_work(j) = k;
        vals_work(j) = idx;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("morton_sort_gather", count);

    /* Comparator overload of sort_by_key: lexicographic on (hi, lo).
     * Falls back from radix to merge-sort on the CUDA backend (still
     * deterministic + stable). */
    Kokkos::Experimental::sort_by_key(ExSpace(), keys_work, vals_work, Morton128Less{});
    Kokkos::fence();

    Kokkos::parallel_for("morton_sort_scatter", count, KOKKOS_LAMBDA(int j) {
        indices_inout[j] = vals_work(j);
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("morton_sort_scatter", count);
    return 0;
}

extern "C" const struct Morton128 *gpu_morton_keys(void)
{
    return g_morton_keys;
}

extern "C" void gpu_morton_release(void)
{
    if(g_morton_keys) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_morton_keys);
        g_morton_keys = NULL;
        g_morton_keys_cap = 0;
    }
}


