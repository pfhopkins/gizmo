#ifndef FORCE_NODE_DRIFT_SYNC_H
#define FORCE_NODE_DRIFT_SYNC_H

/* Release/acquire pairing for a tree node's drift timestamp Nodes[no].Ti_current.
 *
 * SCOPE: this protects ONE race — the host lazy per-node drift inside the threaded
 * Mode-B neighbor walk, where one thread may read a node's geometry while another
 * drifts that same stale node via force_drift_node. force_drift_node writes
 * Ti_current LAST (after every center/len/Extnodes update), so publishing it with a
 * release store and reading it with an acquire load guarantees a thread that
 * observes the fresh Ti_current also observes the fresh geometry: the
 * "Ti_current == All.Ti_Current => geometry fresh" fast path becomes race-free with
 * no atomic node fields and no broad fences (x86-64, ARM64). Both freshness-gating
 * reads on that path acquire: the walk's fast-path check and force_drift_node's own
 * early-return. Other Ti_current writers (gravity tree walk, LET unpack,
 * treebuild/finalize) run in phases not concurrent with the Mode-B walk and are not
 * part of this pairing.
 *
 * GCC/Clang atomic builtins (not std::atomic_ref) so one source compiles under both
 * c++17 (Vista/Frontera) and c++20 (Mac). Requires allvars.h (Nodes, integertime)
 * already in scope. */

static inline integertime modeb_node_ti_current_acquire(int no)
{
    return __atomic_load_n(&Nodes[no].Ti_current, __ATOMIC_ACQUIRE);
}

static inline void force_drift_node_publish_current(int no, integertime time1)
{
    __atomic_store_n(&Nodes[no].Ti_current, time1, __ATOMIC_RELEASE);
}

#endif
