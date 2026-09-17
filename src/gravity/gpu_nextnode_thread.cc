/* gpu_nextnode_thread.cc
 *
 * GPU kernel that recomputes the DFS-pre-order `nextnode` link for each
 * internal node, and `Nextnode` for each particle / pseudo-particle, in
 * parallel.  Replaces the CPU's serial `last`-side-effect threading inside
 * force_update_node_recursive.  FUNR is retired on the GPU build:
 * sibling/father now come from gpu_topology_finalize_{father,sibling}
 * which run before this kernel.  On the non-GPU build FUNR still runs.
 *
 * Algorithm (one thread per internal node N):
 *
 *   Read suns_backup[N][0..7] (snapshotted before the union switch).
 *   Walk the suns left-to-right collecting non-empty entries in order.
 *
 *   * The first non-empty entry is the DFS successor of N itself, so:
 *       nextnode[N] = first_nonempty
 *
 *   * For each non-empty entry E at position j_curr, the "successor after
 *     E's entire subtree" is:
 *       - the next non-empty entry within suns[N], if any
 *       - else sibling[N]   (the value force_update_node_recursive set —
 *         "what comes after N's subtree", which equals what comes after
 *         the last child's subtree)
 *
 *   * Apply this successor to E based on E's type:
 *       - particle (E < TreeParticleSlots):     Nextnode[E]              = succ
 *       - pseudo  (E >= TreeNodeIndexBase + MaxNodes + MaxForeignNodes):
 *             Nextnode[TreeParticleSlots + (E - TreeNodeIndexBase - MaxNodes - MaxForeignNodes)] = succ
 *           (Nextnode[] holds TreeParticleSlots particle slots first, then the pseudo segment;
 *            the foreign-node range [TreeNodeIndexBase+MaxNodes, +MaxForeignNodes) sits below
 *            pseudos in the INDEX space but consumes no Nextnode[] slots -- foreign nodes carry
 *            their own NODE.u.d.nextnode)
 *       - internal node (otherwise):           handled by E's own thread
 *         (E's last descendant gets succ via the recursion of sibling[E]
 *         pointers — already set by force_update_node_recursive)
 *
 * The recursion is implicit: each thread only writes the entries DIRECTLY
 * inside its own parent.  The DFS chain across the whole tree is correct
 * because sibling[E] = "what comes after E's subtree" was set by CPU,
 * and each kernel thread propagates that same value to E's last DFS
 * descendant via its own per-parent pass.
 *
 * Output:
 *   * SoA: soa->nextnode[k]  for internal nodes k in [0..n)
 *   * SoA: soa->nextnode_aux[i]  for particles i in [0..TreeParticleSlots)
 *           and pseudo-particles at slots [TreeParticleSlots..TreeParticleSlots+NTopnodes)
 *   * AoS: Nodes[TreeNodeIndexBase+k].u.d.nextnode  (for the legacy CPU walk path)
 *   * AoS: Nextnode[i]  (for legacy CPU walks; writes covered by host loop
 *           after the device kernel returns)
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
#include "../declarations/gpu_error_check.h"
#include "gpu_gravity_tree.h"
#include "forcetree.h"


extern "C" int gpu_nextnode_thread(void)
{
    if(Numnodestree <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    int n         = Numnodestree;
    int tree_base = All.TreeNodeIndexBase;
    int part_slots = All.TreeParticleSlots;
    int MaxNodes_ = MaxNodes;
    int MaxForeignNodes_ = MaxForeignNodes;    /* LET foreign-node range size */
    int NTopnodes_= NTopnodes;

    /* Acquire SoA — must already be allocated by gpu_nextnode_backup_suns
     * (called earlier from force_treebuild_single), but make sure
     * topology+moments are seeded too: full reseed since CPU just rebuilt. */
    int min_nodes = MaxNodes_ + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_nextnode_thread: SoA null\n"); return 1;}
    if(!soa->suns_backup) {printf("gpu_nextnode_thread: suns_backup null — must call gpu_nextnode_backup_suns first\n"); return 1;}

    /* soa->nextnode_aux is aliased to UVM Nextnode[] (owned by
     * force_treeallocate, sized TreeParticleSlots+NTopnodes+MaxForeignNodes).  Just sanity-check it. */
    int aux_size = part_slots + NTopnodes_ + MaxForeignNodes_;
    if(!soa->nextnode_aux || soa->nextnode_aux_size < aux_size) {
        printf("gpu_nextnode_thread: nextnode_aux alias missing or undersized (have=%d, need=%d)\n",
               soa->nextnode_aux_size, aux_size);
        return 1;
    }

    /* Capture raw SoA pointers for the lambda. */
    int          *suns_backup = soa->suns_backup;
    int          *sibling_soa = soa->sibling;
    int          *nextnode_soa= soa->nextnode;
    int          *aux_soa     = soa->nextnode_aux;

#if TREE_LEAF_BUCKET_SIZE > 1
    /* Where the first malformed leaf chain was found: count, node slot, head particle, the member
     * the chain broke at, and what that member held.  Allocated only when leaves can hold more
     * than one particle, since that is the only case with a chain to malform. */
    enum {CHAIN_FAULT_FIELDS = 5};
    int *chain_fault = (int *) gizmo_gpu_alloc_shared(CHAIN_FAULT_FIELDS * sizeof(int), "treescratch_build_ctr");
    if(!chain_fault) {printf("gpu_nextnode_thread: could not allocate the chain-fault record\n"); return 1;}
    for(int q = 0; q < CHAIN_FAULT_FIELDS; q++) {chain_fault[q] = 0;}
#endif

    Kokkos::parallel_for("nx_thread", n, KOKKOS_LAMBDA(int k) {
        /* k is the SoA index (0..n).  Internal-node id = tree_base + k. */
        long base = (long)k * 8;
        /* Find first non-empty sun. */
        int first = -1, first_pos = -1;
        for(int j = 0; j < 8; j++) {
            int s = suns_backup[base + j];
            if(s >= 0) {first = s; first_pos = j; break;}
        }
        if(first < 0) {
            /* Empty internal node — degenerate; use sibling[N] as fallback. */
            nextnode_soa[k] = sibling_soa[k];
            return;
        }
        nextnode_soa[k] = first;

        /* Walk through non-empty entries in order, threading each entry's
         * post-subtree successor.  For internal-node children we don't
         * write anything (their last descendant is reached via their own
         * thread plus the sibling pointer). */
        int prev_pos = first_pos;
        while(true) {
            /* Find next non-empty position after prev_pos. */
            int next = -1, next_pos = -1;
            for(int j = prev_pos + 1; j < 8; j++) {
                int s = suns_backup[base + j];
                if(s >= 0) {next = s; next_pos = j; break;}
            }
            int prev_id = suns_backup[base + prev_pos];
            int succ = (next >= 0) ? next : sibling_soa[k];
            /* Write successor for prev_id based on its type. */
            if(prev_id < part_slots) {
#if TREE_LEAF_BUCKET_SIZE == 1
                /* particle */
                aux_soa[prev_id] = succ;
#else
                /* A particle slot heads a leaf that may hold several particles: a run the build
                 * threaded together and ended with TREE_LEAF_BUCKET_CHAIN_END.  The successor
                 * belongs on the LAST member, and the interior links must be left alone.
                 *
                 * A chain that does not end in the sentinel means the build and this pass disagree
                 * about the tree.  Threading from a truncated chain would leave the successor on an
                 * interior member, silently shortening every walk that enters the leaf, so the
                 * first thread that sees one records where it happened and the build fails. */
                int last = prev_id, guard = 0, reached_end = 1;
                for(;;) {
                    const int nxt = aux_soa[last];
                    if(nxt == TREE_LEAF_BUCKET_CHAIN_END) {break;}
                    if(nxt < 0 || nxt >= part_slots || ++guard > part_slots) {
                        reached_end = 0;
                        if(Kokkos::atomic_fetch_add(&chain_fault[0], 1) == 0) {
                            chain_fault[1] = k; chain_fault[2] = prev_id;
                            chain_fault[3] = last; chain_fault[4] = nxt;
                        }
                        break;
                    }
                    last = nxt;
                }
                /* Only when the end of the chain was actually found.  `last` is otherwise an
                 * INTERIOR member, and writing the successor there is precisely the truncation
                 * this guard exists to prevent -- the request to stop is drained at a later phase
                 * boundary, so the tree would be walked in that state first.  Leave it untouched
                 * and let the failure above stop the run. */
                if(reached_end) {aux_soa[last] = succ;}
#endif
            } else if(prev_id >= tree_base + MaxNodes_ + MaxForeignNodes_) {
                /* pseudo-particle (the foreign-node range sits below pseudos in the index
                 * space but occupies no slots): the pseudo segment starts after the
                 * part_slots physical particle slots. */
                int idx = part_slots + (prev_id - tree_base - MaxNodes_ - MaxForeignNodes_);
                if(idx >= 0 && idx < part_slots + NTopnodes_) {aux_soa[idx] = succ;}
            }
            /* Foreign nodes (prev_id in [tree_base+MaxNodes, tree_base+MaxNodes+MaxForeignNodes))
             * are not threaded by this kernel — they carry their own NODE.u.d.nextnode pointers
             * directly from LET unpack.  An id between part_slots and tree_base cannot occur in a
             * well-formed tree and matches no branch here, so it writes nothing; that is deliberate.
             * This is a write-side classifier, so dropping such an id is already safe, and the walks
             * that would DEREFERENCE it stop on it explicitly. */
            /* internal node: skip — its own thread sets nextnode_soa, and
             * its last DFS descendant gets `succ` via the chain of
             * sibling pointers (already set by force_update_node_recursive). */
            if(next < 0) {break;}  /* no more non-empty siblings */
            prev_pos = next_pos;
        }
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("nx_thread", n);

#if TREE_LEAF_BUCKET_SIZE > 1
    {
        const int nfault = chain_fault[0], f_slot = chain_fault[1], f_head = chain_fault[2];
        const int f_at = chain_fault[3], f_held = chain_fault[4];
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(chain_fault);
        if(nfault > 0) {
            printf("gpu_nextnode_thread: rank %d found %d malformed leaf chain(s) at a leaf size of %d.\n"
                   "The first was under node %d (slot %d), head particle %d: the chain reached particle %d,\n"
                   "which holds %d instead of a particle below %d or the end marker %d. The build threads\n"
                   "every member of a multi-particle leaf, so a chain that does not end there means the\n"
                   "topology emit and this pass disagree, and the tree is not usable.\n",
                   ThisTask, nfault, (int) TREE_LEAF_BUCKET_SIZE, tree_base + f_slot, f_slot,
                   f_head, f_at, f_held, part_slots, (int) TREE_LEAF_BUCKET_CHAIN_END);
            fflush(stdout);
            return 1;
        }
    }
#endif

    /* Nextnode[] aliases soa->nextnode_aux (same UVM buffer).
     * Internal-node Nodes[].u.d.nextnode writeback runs on the
     * device now that Nodes_base is UVM. */
    struct NODE *Nodes_uvm = Nodes_base;
    Kokkos::parallel_for("nx_writeback_aos", n, KOKKOS_LAMBDA(int k) {
        Nodes_uvm[k].u.d.nextnode = nextnode_soa[k];
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("nx_writeback_aos", n);

    return 0;
}


