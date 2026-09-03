/* mesh/device_tree_walk.h -- the one device tree traversal.
 *
 * A query walks a rank's local tree on the device, and the caller decides what
 * happens when the walk reaches a locally-owned particle.  Everything above
 * that decision -- opening nodes, the periodic overlap test, stepping past
 * imported and pseudo subtrees, recognising a malformed tree -- lives here and
 * is written once.
 *
 * There is exactly one of these.  A second device traversal would mean two
 * copies of the periodic-wrap convention and two copies of the rule for which
 * index classes a walk may follow, in the code where a divergence is hardest to
 * see and most expensive to be wrong about.
 *
 * The caller supplies a leaf policy, an object with
 *
 *     KOKKOS_INLINE_FUNCTION
 *     void visit(int local_index, double qx, double qy, double qz, double reach)
 *
 * called once for every locally-owned particle the walk reaches.  The policy
 * applies its own acceptance test and keeps its own result: the ghost-discovery
 * receiver records the accepted supply slots, while a fused loop evaluates its
 * pair kernel and accumulates.  The traversal itself never looks at particle
 * fields and never decides what a neighbour is.
 *
 * Node geometry comes from the gravity tree's device mirror rather than the
 * managed node arrays, because streaming those from a kernel is memory-bound
 * enough to erase the win.
 *
 * The walk cannot drift a stale node -- that needs a lock -- so it is only
 * legal once the node sweep has certified the tree current and no host lazy
 * drift has happened since.  Callers check that immediately before launching,
 * because a discovery walk earlier in the same exchange can withdraw device
 * legality.
 *
 * Only translation units that compile for the device include this file: the
 * traversal is a Kokkos device function, and the periodic-wrap macros it calls
 * read All.BoxSize, which resolves per translation unit to that unit's All
 * mirror.  Any unit that instantiates this walk must therefore call
 * GIZMO_GPU_ENSURE_ALL_FRESH() at its dispatch boundary, or the box size reads
 * as zero on device and periodic wrapping silently stops working.  Kokkos is
 * included directly rather than through the usual annotation fallback, because
 * this walk calls Kokkos itself: a unit that got the fallback would compile the
 * traversal as host code and then fail on the atomic anyway.
 *
 * WHAT THIS WALK CURRENTLY ASSUMES ABOUT ITS ENTRY POINTS.  It resumes from
 * start nodes another rank's walk reached in this tree, so re-entering the
 * top-level tree ends the branch: the querying rank owns everything above it.
 * A walk that instead started at the root would own those regions itself, and
 * ending the branch there would be wrong for it.  Give this walk an explicit
 * choice at that point before starting one from the root.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef DEVICE_TREE_WALK_H
#define DEVICE_TREE_WALK_H

#include <Kokkos_Core.hpp>

#include "neighbor_list.h"              /* gx_export_envelope_t */
#include "ghost_exchange_functions.h"   /* the canonical-wrap overlap predicate */
#include "../gravity/forcetree.h"       /* BITFLAG_TOPLEVEL */

/* The tree as the device sees it: the mirrored node arrays plus the boundaries
 * that separate the three index classes a walk can encounter.
 *
 * An index below `particle_slots` is a locally-owned particle.  One at or above
 * `node_base` and below `pseudo_start` is a node, of which those at or above
 * `foreign_base` are imported subtrees holding no local particles.  One at or
 * above `pseudo_start` is a pseudo-particle standing for another rank's
 * subtree.  Anything in the gap between the particle slots and the node base
 * belongs to no class at all and means the tree is malformed. */
struct GxDeviceTreeView {
    const Vec3<MyFloat> *node_center;
    const MyFloat       *node_len;
    const int           *node_sibling;
    const int           *node_nextnode;
    const unsigned int  *node_bitflags;
    const int           *nextnode_aux;
    int                  node_base;
    int                  particle_slots;
    int                  node_capacity;
    int                  foreign_base;
    int                  pseudo_start;
};

/* Walk one query against the local tree, resuming from the start nodes the
 * querying rank's own walk reached in this tree.
 *
 * `anomaly` reports the single state the host walk treats as fatal, an index in
 * the gap that belongs to no class.  The host stops the run there, so this walk
 * cannot simply stop stepping: that would truncate the query silently and
 * return a short answer that looks complete.  It records the state instead and
 * the caller reproduces the host's stop. */
template <class LeafPolicy>
KOKKOS_INLINE_FUNCTION
void gx_device_tree_walk(const struct gx_export_envelope_t &env,
                         const GxDeviceTreeView &tree,
                         LeafPolicy &leaf_policy,
                         int *anomaly)
{
    const double qx = env.pos[0], qy = env.pos[1], qz = env.pos[2];
    const double reach = env.h;

    for(int k = 0; k < env.n_nodes; k++) {
        const int start = env.nodes[k];
        if(start < 0) {break;}                   /* -1 terminates the list */
        /* The start list arrived over MPI, so it is validated rather than trusted. */
        if(start < tree.node_base || start >= tree.pseudo_start) {continue;}
        if(start - tree.node_base >= tree.node_capacity) {   /* precondition leaves this unreachable */
            Kokkos::atomic_store(anomaly, 1);
            break;
        }
        int no = tree.node_nextnode[start - tree.node_base];   /* open the exported node */

        while(no >= 0) {
            if(no >= tree.particle_slots && no < tree.node_base) {
                Kokkos::atomic_store(anomaly, 1);   /* malformed tree; caller stops the run */
                break;
            }
            if(no < tree.particle_slots) {
                leaf_policy.visit(no, qx, qy, qz, reach);
                no = tree.nextnode_aux[no];
            } else if(no < tree.pseudo_start) {
                const int kn = no - tree.node_base;
                if(kn < 0 || kn >= tree.node_capacity) {
                    Kokkos::atomic_store(anomaly, 1);
                    break;
                }
                /* Re-entering the top-level tree means this exported branch is
                 * exhausted (the querying rank owns everything above it). */
                if(tree.node_bitflags[kn] & (1u << BITFLAG_TOPLEVEL)) {break;}
                const double hw = 0.5 * (double)tree.node_len[kn];
                const int do_open =
                    gx_extended_overlap_wrap_and_test((double)tree.node_center[kn][0] - qx,
                                                      (double)tree.node_center[kn][1] - qy,
                                                      (double)tree.node_center[kn][2] - qz,
                                                      hw, hw, hw, reach);
                if(do_open) {
                    const int child = tree.node_nextnode[kn];
                    /* An imported foreign subtree holds no locally-owned
                     * particles, so there is nothing below it to find. */
                    no = (child >= tree.foreign_base && child < tree.pseudo_start)
                             ? tree.node_sibling[kn] : child;
                } else {
                    no = tree.node_sibling[kn];
                }
            } else {
                /* Pseudo-particle: another rank's subtree root, nothing local
                 * below it.  Step past exactly as the host walk does. */
                no = tree.nextnode_aux[tree.particle_slots + (no - tree.pseudo_start)];
            }
        }
    }
}

#endif /* DEVICE_TREE_WALK_H */
