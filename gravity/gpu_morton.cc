/* gpu_morton.cc — Step 13 Phase 6.5a
 *
 * Morton encoding + parallel sort infrastructure for the GPU tree-build
 * insertion kernel.  See gpu_morton.h for the API and integration plan.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_morton.h"

#if defined(OPENMP_GPU_OFFLOAD)

namespace {

static uint64_t *g_morton_keys     = NULL;  /* SharedSpace, [g_morton_keys_cap] */
static int       g_morton_keys_cap = 0;

/* 21-bit-per-axis bit-spread for 63-bit Morton interleaving. */
KOKKOS_INLINE_FUNCTION uint64_t morton_spread21(uint64_t v) {
    v &= 0x1FFFFFull;                                    /* keep low 21 bits */
    v = (v | (v << 32)) & 0x1F00000000FFFFull;
    v = (v | (v << 16)) & 0x1F0000FF0000FFull;
    v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v <<  2)) & 0x1249249249249249ull;
    return v;
}

KOKKOS_INLINE_FUNCTION uint64_t morton_encode63(uint32_t x, uint32_t y, uint32_t z) {
    return morton_spread21((uint64_t)x)
         | (morton_spread21((uint64_t)y) << 1)
         | (morton_spread21((uint64_t)z) << 2);
}

}  /* anonymous namespace */

extern "C" int gpu_morton_compute_global_keys(int npart)
{
    if(npart <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH(morton);

    /* Allocate / grow the shared-space key buffer if needed. */
    if(g_morton_keys_cap < npart) {
        if(g_morton_keys) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_morton_keys);
            g_morton_keys = NULL;
        }
        g_morton_keys = (uint64_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            (long)npart * sizeof(uint64_t));
        if(!g_morton_keys) {
            printf("gpu_morton: keys alloc failed (npart=%d)\n", npart);
            g_morton_keys_cap = 0;
            return 1;
        }
        g_morton_keys_cap = npart;
    }

    /* Acquire the particles arena.  Caller is expected to have an active
     * arena already (this is called from inside the gravity-tree build,
     * which arena-acquires earlier), but the call is idempotent. */
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {
        printf("gpu_morton: P_dev null (npart=%d)\n", npart);
        return 1;
    }

    /* Capture domain bounds.  DomainCorner / DomainLen are host globals not
     * mirrored in All_dev; pass by value into the kernel. */
    const double dc0 = DomainCorner[0];
    const double dc1 = DomainCorner[1];
    const double dc2 = DomainCorner[2];
    const double dlen = DomainLen;
    const double inv_dlen = (dlen > 0.0) ? (1.0 / dlen) : 0.0;
    const uint32_t MAX21 = (1u << 21) - 1u;

    uint64_t *keys = g_morton_keys;
    Kokkos::parallel_for("morton_encode_global", npart, KOKKOS_LAMBDA(int i) {
        double rx = (P_dev[i].Pos[0] - dc0) * inv_dlen;
        double ry = (P_dev[i].Pos[1] - dc1) * inv_dlen;
        double rz = (P_dev[i].Pos[2] - dc2) * inv_dlen;
        if(rx < 0.0) {rx = 0.0;} if(rx > 1.0) {rx = 1.0;}
        if(ry < 0.0) {ry = 0.0;} if(ry > 1.0) {ry = 1.0;}
        if(rz < 0.0) {rz = 0.0;} if(rz > 1.0) {rz = 1.0;}
        uint32_t ix = (uint32_t)(rx * (double)MAX21);
        uint32_t iy = (uint32_t)(ry * (double)MAX21);
        uint32_t iz = (uint32_t)(rz * (double)MAX21);
        if(ix > MAX21) {ix = MAX21;}
        if(iy > MAX21) {iy = MAX21;}
        if(iz > MAX21) {iz = MAX21;}
        keys[i] = morton_encode63(ix, iy, iz);
    });
    Kokkos::fence();
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
    Kokkos::View<uint64_t*, MemSpace> keys_work("morton_sort_keys", count);
    Kokkos::View<int*,      MemSpace> vals_work("morton_sort_vals", count);

    int            keys_cap = g_morton_keys_cap;
    const uint64_t *keys_src = g_morton_keys;
    Kokkos::parallel_for("morton_sort_gather", count, KOKKOS_LAMBDA(int j) {
        int idx = indices_inout[j];
        uint64_t k = (idx >= 0 && idx < keys_cap) ? keys_src[idx] : (uint64_t)0;
        keys_work(j) = k;
        vals_work(j) = idx;
    });
    Kokkos::fence();

    Kokkos::Experimental::sort_by_key(ExSpace(), keys_work, vals_work);
    Kokkos::fence();

    Kokkos::parallel_for("morton_sort_scatter", count, KOKKOS_LAMBDA(int j) {
        indices_inout[j] = vals_work(j);
    });
    Kokkos::fence();
    return 0;
}

extern "C" const uint64_t *gpu_morton_keys(void)
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

GPU_ALL_SYNC_FUNC(morton)

#endif /* OPENMP_GPU_OFFLOAD */
