/* gpu_topology_finalize.cc
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
#include <exception>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
#include "gpu_gravity_tree.h"
#include "gpu_topology_finalize.h"
#include "forcetree.h"


extern "C" int gpu_topology_finalize_father(int n)
{
    if(n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    int tree_base = All.TreeNodeIndexBase;
    int tree_slots = All.TreeParticleSlots;

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
    Kokkos::parallel_for("topo_father_init_parts", tree_slots, KOKKOS_LAMBDA(int i) {
        Father_uvm[i] = -1;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_father_init", n);

    /* Main pass: one thread per internal node.  For each occupied child slot,
     * write that child's father back to this node (absolute index = tree_base + k).
     * Internal child  -> father_soa[child - tree_base] = parent_abs.
     * Particle child  -> Father_uvm[child]           = parent_abs.
     * Pseudo-particle children (>= tree_base + MaxNodes) carry no per-particle
     * Father[] entry; Father[] is written only for real particle slots (no < tree_slots). */
    int MaxNodes_ = MaxNodes;
    Kokkos::parallel_for("topo_father_main", n, KOKKOS_LAMBDA(int k) {
        int parent_abs = tree_base + k;
        long base = (long)k * 8;
        for(int s = 0; s < 8; s++) {
            int c = suns_backup[base + s];
            if(c < 0) {continue;}
            if(c < tree_slots) {
                /* particle */
                Father_uvm[c] = parent_abs;
            } else if(c >= tree_base && c < tree_base + MaxNodes_) {
                /* internal node */
                father_soa[c - tree_base] = parent_abs;
            }
            /* else: pseudo-particle — no Father[] entry */
        }
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_father_main", n);

    return 0;
}

extern "C" int gpu_topology_finalize_sibling(int n)
{
    if(n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    int tree_base = All.TreeNodeIndexBase;

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

    /* One thread per internal node k (absolute id = tree_base + k).
     * Walk up via father chain until we find a parent slot whose later
     * positions contain an occupied entry — that's the sibling.  Reaching
     * a parent_abs < tree_base (i.e. -1 for root) means no sibling -> -1.
     *
     * Cost is O(depth * 8) per thread.  For Phil's clustered FIRE/STARFORGE
     * runs depth maxes around 30-50; per-thread work is bounded and uniform
     * enough for good GPU occupancy.  All reads are from suns_backup and
     * father_soa, both SharedSpace UVM. */
    Kokkos::parallel_for("topo_sibling_walk", n, KOKKOS_LAMBDA(int k) {
        int cur_abs = tree_base + k;
        int found = -1;
        for(int safety = 0; safety < 64; safety++) {
            int parent_abs = father_soa[cur_abs - tree_base];
            if(parent_abs < tree_base) {
                /* Reached root or hit an unset father (defensive). */
                break;
            }
            long pb = (long)(parent_abs - tree_base) * 8;
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
    gizmo_gpu_check_last_error("topo_sibling_walk", n);

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
    struct NODE  *Nodes_uvm   = Nodes_base;   /* UVM on the Kokkos path */

    /* GPU kernel writeback (was host OMP loop).  With Nodes_base
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
    gizmo_gpu_check_last_error("topo_writeback_d_to_aos", n);
    return 0;
}

extern "C" int gpu_node_reset_ephemeral(int n)
{
    if(n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Host-side scalar capture from `All.*` in a GPU TU MUST use the
     * canonical out-of-line accessor, not bare All.* (which resolves to
     * All_dev here and can be stale). Capture once to locals so the
     * Kokkos lambda [=] captures the correct host values. */
    const struct global_data_all_processes *host_all = gizmo_host_all_ptr();
    integertime ti_current = host_all->Ti_Current;
    int         glob_flag  = GlobFlag;

    struct NODE    *Nodes_uvm    = Nodes_base;     /* UVM */
    struct extNODE *Extnodes_uvm = Extnodes_base;

    Kokkos::parallel_for("node_reset_ephemeral", n, KOKKOS_LAMBDA(int k) {
        /* k is the SoA index; absolute Nodes[] index is All.TreeNodeIndexBase + k.
         * Nodes_base/Extnodes_base are the unshifted arrays (Nodes ==
         * Nodes_base - All.TreeNodeIndexBase), so we index directly with k. */
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
    gizmo_gpu_check_last_error("node_reset_ephemeral", n);
    return 0;
}

extern "C" int *gpu_father_alloc(int tree_particle_slots)
{
    size_t bytes = (size_t)tree_particle_slots * sizeof(int);
    if(bytes == 0) {return NULL;}
    /* kokkos_malloc THROWS on host-OOM; catch -> NULL so the caller's NULL-check
       controlled-stop path (force_treeallocate) fires instead of a hard terminate. */
    try { return (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("tree_father", bytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void gpu_father_free(int *p)
{
    if(p) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);}
}

/* Generic SharedSpace alloc/free for Nodes_base, Extnodes_base,
 * Nextnode (and any future tree-storage array that needs to be GPU-addressable).
 * Pattern matches gpu_father_alloc/free; struct sizes are computed at call site
 * in forcetree.cc which has the NODE/extNODE definitions in scope. */
extern "C" void *gpu_tree_alloc_bytes(size_t bytes, const char *label)
{
    if(bytes == 0) {return NULL;}
    /* kokkos_malloc THROWS on host-OOM; catch -> NULL so the caller's NULL-check
       controlled-stop path fires instead of a hard terminate. The label names the
       buffer in the Kokkos allocation stream and any future OOM message. */
    try { return Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label ? label : "tree_alloc", bytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void gpu_tree_free_bytes(void *p)
{
    if(p) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);}
}


