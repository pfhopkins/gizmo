/* gpu_peano_walk.cc -- Step 13 Phase 6.5c1
 *
 * Host-side mirror lifecycle for the device Peano walk.  See
 * gpu_peano_walk.h for the API and gpu_peano_walk_functions.h for the
 * device-side primitives.
 *
 * Lifecycle notes:
 *   - TopNodes[] and DomainNodeIndex[] are populated on host as part of
 *     domain decomposition (TopNodes) and force_treebuild (DomainNodeIndex
 *     via force_create_empty_nodes).  We mirror them into SharedSpace
 *     after both are valid -- typically just before the GPU build kernel
 *     fires (Phase 6.5d wires the call site).
 *   - Mirrors are sized to MaxTopNodes + 1 / NTopleaves so the realloc
 *     path is rare.  Free + re-alloc when host arrays grow beyond the
 *     mirror capacity.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gpu_peano_walk.h"
#include "gpu_peano_walk_functions.h"  /* syntax-check device-side header */


namespace {

static struct topnode_data *g_topnodes_dev      = NULL;  /* SharedSpace mirror */
static int                 *g_domain_idx_dev    = NULL;  /* SharedSpace mirror */
static int                  g_topnodes_cap      = 0;     /* allocated entries */
static int                  g_domain_idx_cap    = 0;

}  /* anonymous namespace */

extern "C" int gpu_peano_walk_acquire(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Grow the TopNodes mirror if needed. */
    if(g_topnodes_cap < NTopnodes) {
        if(g_topnodes_dev) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topnodes_dev);
            g_topnodes_dev = NULL;
        }
        long bytes = (long)NTopnodes * (long)sizeof(struct topnode_data);
        g_topnodes_dev = (struct topnode_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bytes);
        if(!g_topnodes_dev) {
            printf("gpu_peano_walk: TopNodes mirror alloc failed (NTopnodes=%d, %ld B)\n",
                   NTopnodes, bytes);
            g_topnodes_cap = 0;
            return 1;
        }
        g_topnodes_cap = NTopnodes;
    }

    /* Grow the DomainNodeIndex mirror if needed. */
    if(g_domain_idx_cap < NTopleaves) {
        if(g_domain_idx_dev) {
            Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_domain_idx_dev);
            g_domain_idx_dev = NULL;
        }
        long bytes = (long)NTopleaves * (long)sizeof(int);
        g_domain_idx_dev = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bytes);
        if(!g_domain_idx_dev) {
            printf("gpu_peano_walk: DomainNodeIndex mirror alloc failed (NTopleaves=%d, %ld B)\n",
                   NTopleaves, bytes);
            g_domain_idx_cap = 0;
            return 1;
        }
        g_domain_idx_cap = NTopleaves;
    }

    /* Copy host -> SharedSpace.  SharedSpace (UVM) makes the data visible
     * on device automatically. */
    if(NTopnodes > 0 && TopNodes) {
        memcpy(g_topnodes_dev, TopNodes, (size_t)NTopnodes * sizeof(struct topnode_data));
    }
    if(NTopleaves > 0 && DomainNodeIndex) {
        memcpy(g_domain_idx_dev, DomainNodeIndex, (size_t)NTopleaves * sizeof(int));
    }
    return 0;
}

extern "C" void gpu_peano_walk_release(void)
{
    if(g_topnodes_dev) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topnodes_dev);
        g_topnodes_dev = NULL;
        g_topnodes_cap = 0;
    }
    if(g_domain_idx_dev) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_domain_idx_dev);
        g_domain_idx_dev = NULL;
        g_domain_idx_cap = 0;
    }
}

extern "C" const struct topnode_data *gpu_peano_walk_topnodes(void)
{
    return g_topnodes_dev;
}

extern "C" const int *gpu_peano_walk_domain_node_index(void)
{
    return g_domain_idx_dev;
}


