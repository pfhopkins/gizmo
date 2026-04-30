/* gpu_topology_finalize.cc — Step 13 Phase 6.6
 *
 * Replaces the sibling / father / Father[] outputs of
 * force_update_node_recursive with three GPU passes.  See
 * gpu_topology_finalize.h for the contract.
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
#include "gpu_gravity_tree.h"
#include "gpu_topology_finalize.h"
#include "forcetree.h"


extern "C" int gpu_topology_finalize_father(int n)
{
    if(n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH(topo_finalize_father);

    int MaxPart = All.MaxPart;

    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_topology_finalize_father: SoA null\n"); return 1;}
    if(!soa->suns_backup) {printf("gpu_topology_finalize_father: suns_backup null\n"); return 1;}
    if(!soa->father)      {printf("gpu_topology_finalize_father: father null\n");      return 1;}
    if(!soa->bitflags)    {printf("gpu_topology_finalize_father: bitflags null\n");    return 1;}
    if(!Father)           {printf("gpu_topology_finalize_father: Father[] null\n");    return 1;}

    int          *suns_backup = soa->suns_backup;
    int          *father_soa  = soa->father;
    unsigned int *bitflags_soa = soa->bitflags;
    int          *Father_uvm  = Father;          /* UVM-resident on GPU build */

    /* Init pass: father[k] = -1 for all internal-node slots; Father[i] = -1
     * for all particle slots.  Root (k=0) keeps -1 since it has no parent;
     * unused internal-node slots beyond actual topology also stay -1.  This
     * matches the FUNR contract that any node not visited via DFS keeps
     * its prior (effectively undefined) value — using -1 is strictly safer
     * than uninitialized memory.
     *
     * ALSO clear soa->bitflags[k] = 0 for k in [0, n).  Reason: the prior
     * gpu_topology_writeback_to_aos in force_treebuild_single wrote u.suns
     * into Nodes_base[].u, clobbering u.d.bitflags via the union with
     * arbitrary garbage.  FUNR (now retired) used to reset bitflags via
     *   Nodes[no].u.d.bitflags = multiple_flag    (forcetree.cc:964)
     * so moment_refresh's acquire-then-mask saw clean state.  Without that
     * reset, garbage TOPLEVEL/INTERNAL_TOPLEVEL bits bleed into
     * inside-topleaf nodes via moment_refresh's mask ((u.d.bitflags &
     * (TOPLEVEL|DEPENDS|INTERNAL_TOPLEVEL))) and break force_treeupdate_pseudos.
     * force_flag_localnodes runs later and ORs the correct flags onto the
     * right nodes from this clean baseline. */
    Kokkos::parallel_for("topo_father_init_nodes", n, KOKKOS_LAMBDA(int k) {
        father_soa[k]   = -1;
        bitflags_soa[k] = 0u;
    });
    Kokkos::parallel_for("topo_father_init_parts", MaxPart, KOKKOS_LAMBDA(int i) {
        Father_uvm[i] = -1;
    });
    Kokkos::fence();

    /* Main pass: one thread per internal node.  For each occupied child slot,
     * write that child's father back to this node (absolute index = MaxPart + k).
     * Internal child  -> father_soa[child - MaxPart] = parent_abs.
     * Particle child  -> Father_uvm[child]           = parent_abs.
     * Pseudo-particle children (>= MaxPart + MaxNodes) carry no per-particle
     * Father[] entry (matches FUNR which only sets Father for no < MaxPart). */
    int MaxNodes_ = MaxNodes;
    Kokkos::parallel_for("topo_father_main", n, KOKKOS_LAMBDA(int k) {
        int parent_abs = MaxPart + k;
        long base = (long)k * 8;
        for(int s = 0; s < 8; s++) {
            int c = suns_backup[base + s];
            if(c < 0) {continue;}
            if(c < MaxPart) {
                /* particle */
                Father_uvm[c] = parent_abs;
            } else if(c < MaxPart + MaxNodes_) {
                /* internal node */
                father_soa[c - MaxPart] = parent_abs;
            }
            /* else: pseudo-particle — no Father[] entry */
        }
    });
    Kokkos::fence();

    return 0;
}

extern "C" int gpu_topology_finalize_sibling(int n)
{
    if(n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH(topo_finalize_sibling);

    int MaxPart = All.MaxPart;

    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_topology_finalize_sibling: SoA null\n"); return 1;}
    if(!soa->suns_backup) {printf("gpu_topology_finalize_sibling: suns_backup null\n"); return 1;}
    if(!soa->father)      {printf("gpu_topology_finalize_sibling: father null\n");      return 1;}
    if(!soa->sibling)     {printf("gpu_topology_finalize_sibling: sibling null\n");     return 1;}

    int *suns_backup = soa->suns_backup;
    int *father_soa  = soa->father;
    int *sibling_soa = soa->sibling;

    /* One thread per internal node k (absolute id = MaxPart + k).
     * Walk up via father chain until we find a parent slot whose later
     * positions contain an occupied entry — that's the sibling.  Reaching
     * a parent_abs < MaxPart (i.e. -1 for root) means no sibling -> -1.
     *
     * Cost is O(depth * 8) per thread.  For Phil's clustered FIRE/STARFORGE
     * runs depth maxes around 30-50; per-thread work is bounded and uniform
     * enough for good GPU occupancy.  All reads are from suns_backup and
     * father_soa, both SharedSpace UVM. */
    Kokkos::parallel_for("topo_sibling_walk", n, KOKKOS_LAMBDA(int k) {
        int cur_abs = MaxPart + k;
        int found = -1;
        for(int safety = 0; safety < 64; safety++) {
            int parent_abs = father_soa[cur_abs - MaxPart];
            if(parent_abs < MaxPart) {
                /* Reached root or hit an unset father (defensive). */
                break;
            }
            long pb = (long)(parent_abs - MaxPart) * 8;
            /* Locate cur_abs in parent's suns. */
            int slot = -1;
            for(int s = 0; s < 8; s++) {
                if(suns_backup[pb + s] == cur_abs) {slot = s; break;}
            }
            if(slot < 0) {
                /* Topology corruption — bail. */
                break;
            }
            /* Scan for next occupied slot. */
            for(int j = slot + 1; j < 8; j++) {
                int v = suns_backup[pb + j];
                if(v >= 0) {found = v; break;}
            }
            if(found >= 0) {break;}
            /* Ascend. */
            cur_abs = parent_abs;
        }
        sibling_soa[k] = found;
    });
    Kokkos::fence();

    return 0;
}

extern "C" int gpu_topology_writeback_d_to_aos(int n)
{
    if(n <= 0) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa)             {printf("gpu_topology_writeback_d_to_aos: SoA null\n");     return 1;}
    if(!soa->sibling)    {printf("gpu_topology_writeback_d_to_aos: sibling null\n"); return 1;}
    if(!soa->father)     {printf("gpu_topology_writeback_d_to_aos: father null\n");  return 1;}

    int          *sibling_soa = soa->sibling;
    int          *father_soa  = soa->father;
    struct NODE  *Nodes_uvm   = Nodes_base;   /* UVM on the Kokkos path (6.8d) */

    /* Phase 6.8f: GPU kernel writeback (was host OMP loop).  With Nodes_base
     * UVM-resident, the same parallel writes happen device-side without an
     * extra page-touch on host.  Independent stores per node slot — no
     * synchronization needed.  The legacy CPU tree-walks (gravtree.cc) read
     * Nodes[].u.d.sibling/.father; they pull pages back to host on first
     * post-build touch. */
    Kokkos::parallel_for("topo_writeback_d_to_aos", n, KOKKOS_LAMBDA(int k) {
        Nodes_uvm[k].u.d.sibling = sibling_soa[k];
        Nodes_uvm[k].u.d.father  = father_soa[k];
    });
    Kokkos::fence();
    return 0;
}

extern "C" int gpu_node_reset_ephemeral(int n)
{
    if(n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH(node_reset_ephemeral);

    int MaxPart = All.MaxPart;
    /* Snapshot scalars to locals: KOKKOS_LAMBDA captures via [=] but the
     * GIZMO_GPU_ENSURE_ALL_FRESH guarantee for All_dev is per-call, so use
     * locals to avoid cross-platform device-symbol footguns. */
    integertime ti_current = All.Ti_Current;
    int         glob_flag  = GlobFlag;

    struct NODE    *Nodes_uvm    = Nodes_base;     /* UVM (6.8d) */
    struct extNODE *Extnodes_uvm = Extnodes_base;

    Kokkos::parallel_for("node_reset_ephemeral", n, KOKKOS_LAMBDA(int k) {
        /* k is the SoA index; absolute Nodes[] index is MaxPart + k.
         * Nodes_base/Extnodes_base are the unshifted arrays (Nodes ==
         * Nodes_base - MaxPart), so we index directly with k. */
        Nodes_uvm[k].GravCost          = 0;
        Nodes_uvm[k].Ti_current        = ti_current;
        Extnodes_uvm[k].dp             = {};
        Extnodes_uvm[k].Ti_lastkicked  = ti_current;
        Extnodes_uvm[k].Flag           = glob_flag;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        Extnodes_uvm[k].rt_source_lum_dp = {};
#endif
#ifdef DM_SCALARFIELD_SCREENING
        Extnodes_uvm[k].dp_dm          = {};
#endif
    });
    Kokkos::fence();
    return 0;
}

extern "C" int *gpu_father_alloc(int maxpart)
{
    size_t bytes = (size_t)maxpart * sizeof(int);
    int *p = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bytes);
    return p;
}

extern "C" void gpu_father_free(int *p)
{
    if(p) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);}
}

/* Phase 6.8: generic SharedSpace alloc/free for Nodes_base, Extnodes_base,
 * Nextnode (and any future tree-storage array that needs to be GPU-addressable).
 * Pattern matches gpu_father_alloc/free; struct sizes are computed at call site
 * in forcetree.cc which has the NODE/extNODE definitions in scope. */
extern "C" void *gpu_tree_alloc_bytes(size_t bytes)
{
    if(bytes == 0) {return NULL;}
    return Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(bytes);
}

extern "C" void gpu_tree_free_bytes(void *p)
{
    if(p) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);}
}

GPU_ALL_SYNC_FUNC(topo_finalize_father)
GPU_ALL_SYNC_FUNC(topo_finalize_sibling)
GPU_ALL_SYNC_FUNC(node_reset_ephemeral)

