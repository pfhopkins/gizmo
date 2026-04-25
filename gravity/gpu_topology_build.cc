/* gpu_topology_build.cc -- Step 13 Phase 6.5c2
 *
 * GPU tree-build data path: per-particle Peano-walk to topleaf, 128-bit
 * Morton key, parallel histogram + scan + scatter to bucket particles
 * by topleaf, per-topleaf Morton sort.  See gpu_topology_build.h for the
 * API.
 *
 * Topology emission (6.5c3), collocation handling (6.5c4), and overflow
 * retry are added in subsequent commits.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_morton.h"
#include "gpu_morton_functions.h"
#include "gpu_peano_walk.h"
#include "gpu_peano_walk_functions.h"
#include "gpu_topology_build.h"

#if defined(OPENMP_GPU_OFFLOAD)

namespace {

/* SharedSpace scratch -- reused across tree builds, grown on demand. */
static int *g_sorted_idx          = NULL;  /* [npart]              */
static int *g_particle_topleaf    = NULL;  /* [npart]              */
static int *g_topleaf_start       = NULL;  /* [NTopleaves + 1]     */
static int *g_topleaf_count       = NULL;  /* [NTopleaves]         */
static int *g_topleaf_cursor      = NULL;  /* [NTopleaves] -- scatter cursors */
static int  g_npart_cap           = 0;
static int  g_topleaf_cap         = 0;

/* Allocate/grow a SharedSpace int buffer. */
static int *grow_int_buffer(int *buf, int old_cap, int new_cap, const char *name) {
    if(old_cap >= new_cap) {return buf;}
    if(buf) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(buf);}
    buf = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((long)new_cap * sizeof(int));
    if(!buf) {
        printf("gpu_topology_build: %s alloc failed (n=%d)\n", name, new_cap);
    }
    return buf;
}

}  /* anonymous namespace */

extern "C" int gpu_topology_build_data_path(int npart)
{
    if(npart <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH(topobuild);

    /* Acquire dependencies. */
    int rc = gpu_peano_walk_acquire();
    if(rc) {printf("gpu_topology_build: gpu_peano_walk_acquire failed\n"); return rc;}
    Morton128 *keys = gpu_morton_keys_acquire(npart);
    if(!keys) {return 1;}

    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {printf("gpu_topology_build: P_dev null\n"); return 1;}

    const struct topnode_data *tn  = gpu_peano_walk_topnodes();
    const int                 *dni = gpu_peano_walk_domain_node_index();
    if(!tn || !dni) {printf("gpu_topology_build: peano-walk mirrors null\n"); return 1;}
    (void)dni;  /* used by 6.5c3 topology emission to map topleaf -> Nodes[] slot */

    /* Grow per-particle scratch. */
    if(g_npart_cap < npart) {
        g_sorted_idx       = grow_int_buffer(g_sorted_idx,       g_npart_cap, npart, "sorted_idx");
        g_particle_topleaf = grow_int_buffer(g_particle_topleaf, g_npart_cap, npart, "particle_topleaf");
        if(!g_sorted_idx || !g_particle_topleaf) {g_npart_cap = 0; return 1;}
        g_npart_cap = npart;
    }
    /* Grow per-topleaf scratch. */
    if(g_topleaf_cap < NTopleaves + 1) {
        int newcap = NTopleaves + 1;
        g_topleaf_start  = grow_int_buffer(g_topleaf_start,  g_topleaf_cap, newcap, "topleaf_start");
        g_topleaf_count  = grow_int_buffer(g_topleaf_count,  g_topleaf_cap, newcap, "topleaf_count");
        g_topleaf_cursor = grow_int_buffer(g_topleaf_cursor, g_topleaf_cap, newcap, "topleaf_cursor");
        if(!g_topleaf_start || !g_topleaf_count || !g_topleaf_cursor) {g_topleaf_cap = 0; return 1;}
        g_topleaf_cap = newcap;
    }

    int  ntl  = NTopleaves;
    int *pt   = g_particle_topleaf;
    int *tcnt = g_topleaf_count;
    int *tcur = g_topleaf_cursor;
    int *tsta = g_topleaf_start;
    int *sidx = g_sorted_idx;

    /* Capture domain bounds for the encode kernel. */
    const double dc0 = DomainCorner[0];
    const double dc1 = DomainCorner[1];
    const double dc2 = DomainCorner[2];
    const double dlen = DomainLen;
    const double inv_dlen = (dlen > 0.0) ? (1.0 / dlen) : 0.0;
    const int    bits = BITS_PER_DIMENSION;

    /* Zero topleaf bucket counters. */
    Kokkos::parallel_for("topo_zero_counts", ntl, KOKKOS_LAMBDA(int t) {
        tcnt[t] = 0;
        tcur[t] = 0;
    });
    Kokkos::fence();

    /* Kernel 1: per-particle Peano + Morton key compute, TopNodes walk to
     * topleaf id, write Morton key + topleaf id, increment bucket count. */
    Kokkos::parallel_for("topo_keys_and_assign", npart, KOKKOS_LAMBDA(int i) {
        double fx = (P_dev[i].Pos[0] - dc0) * inv_dlen;
        double fy = (P_dev[i].Pos[1] - dc1) * inv_dlen;
        double fz = (P_dev[i].Pos[2] - dc2) * inv_dlen;
        if(fx < 0.0) {fx = 0.0;} if(fx >= 1.0) {fx = 0.99999999999999988897;}
        if(fy < 0.0) {fy = 0.0;} if(fy >= 1.0) {fy = 0.99999999999999988897;}
        if(fz < 0.0) {fz = 0.0;} if(fz >= 1.0) {fz = 0.99999999999999988897;}
        uint64_t ix = gpu_morton_double_to_int42(fx + 1.0);
        uint64_t iy = gpu_morton_double_to_int42(fy + 1.0);
        uint64_t iz = gpu_morton_double_to_int42(fz + 1.0);

        Morton128 m;
        peanokey  pkey = gpu_peano_and_morton_key(ix, iy, iz, bits, &m);
        keys[i] = m;

        int leaf = gpu_topleaf_for_key(tn, pkey);
        pt[i] = leaf;

        Kokkos::atomic_fetch_add(&tcnt[leaf], 1);
    });
    Kokkos::fence();

    /* Kernel 2: exclusive prefix scan to compute topleaf_start[]. */
    Kokkos::parallel_scan("topo_scan", ntl,
        KOKKOS_LAMBDA(int t, int &acc, bool final_pass) {
            int c = tcnt[t];
            if(final_pass) {tsta[t] = acc;}
            acc += c;
        });
    Kokkos::fence();
    /* Sentinel: tsta[NTopleaves] = total particle count. */
    Kokkos::parallel_for("topo_scan_sentinel", 1, KOKKOS_LAMBDA(int /*unused*/) {
        int total = 0;
        for(int t = 0; t < ntl; t++) {total += tcnt[t];}
        tsta[ntl] = total;
    });
    Kokkos::fence();

    /* Kernel 3: scatter -- each particle takes its slot in its topleaf's
     * range via atomic_fetch_add into tcur[]. */
    Kokkos::parallel_for("topo_scatter", npart, KOKKOS_LAMBDA(int i) {
        int leaf = pt[i];
        int slot = Kokkos::atomic_fetch_add(&tcur[leaf], 1);
        sidx[tsta[leaf] + slot] = i;
    });
    Kokkos::fence();

    /* Kernel 4 (per-topleaf sort): host loop over topleaves, dispatching
     * one Morton sort per range.  NTopleaves is typically O(100); each
     * sort is small.  Future optimization: parallel team-policy sort,
     * but correctness-first here. */
    for(int t = 0; t < ntl; t++) {
        int start = g_topleaf_start[t];
        int count = g_topleaf_count[t];
        if(count > 1) {
            int rc2 = gpu_morton_sort_indices(count, g_sorted_idx + start);
            if(rc2) {return rc2;}
        }
    }

    return 0;
}

extern "C" const int *gpu_topology_build_sorted_idx(void)        { return g_sorted_idx;       }
extern "C" const int *gpu_topology_build_topleaf_start(void)     { return g_topleaf_start;    }
extern "C" const int *gpu_topology_build_topleaf_count(void)     { return g_topleaf_count;    }
extern "C" const int *gpu_topology_build_particle_topleaf(void)  { return g_particle_topleaf; }

extern "C" void gpu_topology_build_release(void)
{
    if(g_sorted_idx)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_sorted_idx);       g_sorted_idx       = NULL;}
    if(g_particle_topleaf) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_particle_topleaf); g_particle_topleaf = NULL;}
    if(g_topleaf_start)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_start);    g_topleaf_start    = NULL;}
    if(g_topleaf_count)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_count);    g_topleaf_count    = NULL;}
    if(g_topleaf_cursor)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_cursor);   g_topleaf_cursor   = NULL;}
    g_npart_cap = 0;
    g_topleaf_cap = 0;
}

GPU_ALL_SYNC_FUNC(topobuild)

#endif /* OPENMP_GPU_OFFLOAD */
