/* sfc_tiles.cc — the ghost supply pool: the index members of P[], in P[] order.
 *
 * Membership is sfc_pool_member (sfc_tiles.h), shared with the spatial index build in
 * gpu_neighbor_list.cc; the tiles, rows and BVH are built there.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "sfc_tiles.h"


int build_sfc_supply_pool(struct particle_data *P, int num_total,
                          int type_bitmask, int **pool_indices_out)
{
    /* Ordered compaction in fixed index blocks: count each block's members, place each block at the prefix
       sum of the counts before it, then fill the blocks.  The blocks do not depend on the thread count, so
       the pool is the serial one, in P[] order, whatever runs it. */
    const int block = 65536;
    const int nblocks = (num_total + block - 1) / block;
    const int threaded = gizmo_pcie_opt() && nblocks > 1;
    std::vector<int> start((size_t) nblocks + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(threaded)
#endif
    for(int b = 0; b < nblocks; b++) {
        const int i1 = (b + 1) * block < num_total ? (b + 1) * block : num_total;
        int n = 0;
        for(int i = b * block; i < i1; i++) {if(sfc_pool_member(&P[i], type_bitmask)) {n++;}}
        start[(size_t) b + 1] = n;
    }
    for(int b = 0; b < nblocks; b++) {start[(size_t) b + 1] += start[(size_t) b];}
    const int num_pool = start[(size_t) nblocks];
    if(!pool_indices_out) return num_pool;

    int *pool = (int *) mymalloc("sfc_pool", (num_pool > 0 ? num_pool : 1) * sizeof(int));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(threaded)
#endif
    for(int b = 0; b < nblocks; b++) {
        const int i1 = (b + 1) * block < num_total ? (b + 1) * block : num_total;
        int p = start[(size_t) b];
        for(int i = b * block; i < i1; i++) {if(sfc_pool_member(&P[i], type_bitmask)) {pool[p++] = i;}}
    }
    *pool_indices_out = pool;
    return num_pool;
}
