/* mesh/device_export_topology.h -- the export rule of a device walk over the top-level tree: at every top leaf
 * another rank owns that the walk opens it records (owner, that leaf's node) and goes no deeper; a top leaf this
 * rank owns it steps past, since nothing below a top leaf can be another rank's.  That is the host export walk's
 * rule (mode_b_local_walker.cc), node for node.  Used by Mode D's export (mode_d_export.h) and by request-driven
 * ghost discovery's sender walk (gpu_neighbor_list.cc gx_device_sender_walk), so the two cannot disagree.
 *
 * Include after <Kokkos_Core.hpp>. */
#ifndef GIZMO_DEVICE_EXPORT_TOPOLOGY_H
#define GIZMO_DEVICE_EXPORT_TOPOLOGY_H

#include <Kokkos_Core.hpp>
#include "device_tree_walk.h"   /* GxNodeDecision */
#include "gpu_neighbor_list.h"  /* GxTopLeafMapView */

/* What the export walk does at a node and at a pseudo-particle: the host export walk's rule. */
struct ModeDExportTopology {
    GxTopLeafMapView map{};
    int this_task = -1;

    KOKKOS_INLINE_FUNCTION GxNodeDecision node_decision(int kn) const
    {
        const int leaf = (kn >= 0 && kn < map.span) ? map.leaf_of_node[kn] : -1;
        if(leaf < 0) {return GxNodeDecision{GxNodeAction::Descend, -1};}
        /* a top leaf another rank owns: send there; one this rank owns: nothing below it can be another rank's */
        if(map.domain_task[leaf] != this_task) {return GxNodeDecision{GxNodeAction::RecordAndSibling, leaf};}
        return GxNodeDecision{GxNodeAction::Sibling, leaf};
    }
    /* A pseudo-particle stands for another rank's top leaf the import did not replace; the host sends to it when
       the walk reaches it. */
    KOKKOS_INLINE_FUNCTION GxNodeDecision pseudo_decision(int leaf) const
    {
        if(leaf >= 0 && leaf < map.n_topleaves && map.domain_task[leaf] != this_task) {
            return GxNodeDecision{GxNodeAction::RecordAndSibling, leaf};
        }
        return GxNodeDecision{GxNodeAction::Descend, -1};
    }
};

#endif /* GIZMO_DEVICE_EXPORT_TOPOLOGY_H */
