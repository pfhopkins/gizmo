/* mesh/mode_d_export.h -- deciding, on the device, where a device-resident neighbour call must send its queries,
 * and packing them; and merging what comes back.
 *
 * A query reaches particles of other ranks through the top leaves those ranks own.  The export walk descends this
 * rank's copy of the top-level tree from the root, and at every top leaf it opens that another rank owns it records
 * (owner, that leaf's node) and goes no deeper; a top leaf this rank owns it skips, since nothing below a top leaf
 * can be another rank's.  That is the host export walk's rule (mode_b_local_walker.cc), node for node, so the
 * device decides exactly what the host decided whenever both see the same node geometry.  The walk runs twice over
 * one frozen set of queries: once to COUNT what each query sends, once to FILL each query's records into a slice
 * the count reserved for it alone.
 *
 * The records then become envelopes in the order the host has always produced them -- by peer, then query, then
 * the order the walk reached the nodes -- through a stable partition by peer (a least-significant-bit radix sort
 * over the peer's bits), and the replies are merged per query in that same order (the query's own result first,
 * then peers ascending, then each peer's envelopes in order): each query's runs are listed once from the sorted
 * records and the query orders its own short list by peer, so there is no second sort.  Nothing here depends on
 * atomics for ORDER (counts only), nothing sizes a table by the task count, and nothing returns to the host
 * between the fill and the round's one read of its per-peer counts.
 *
 * Include after <Kokkos_Core.hpp> and neighbor_loop_runner.h. */
#ifndef GIZMO_MODE_D_EXPORT_H
#define GIZMO_MODE_D_EXPORT_H

#include <climits>
#include <Kokkos_Core.hpp>
#include "neighbor_list.h"
#include "device_tree_walk.h"
#include "gpu_neighbor_list.h"

/* One record of the export decision: query `active` sends top-level node `node` to `peer`. */
struct ModeDExportRecord { int peer; int active; int node; };
/* One query's records for one peer, after the sort: that peer, the envelope the run starts in, and how many records. */
struct ModeDQueryRun { int peer; int first_envelope; int records; };
static_assert(sizeof(ModeDQueryRun) <= sizeof(ModeDExportRecord), "a query's runs are kept in the sort's spare record buffer");

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

/* A query's running count: the nodes it sends, and how many times the destination changes along the walk.  A run of
 * one destination goes out in envelopes of NODELISTLENGTH nodes, so runs + floor(nodes / NODELISTLENGTH) envelopes
 * always cover the query, however its runs fall. */
struct ModeDExportCount {
    int nodes = 0, runs = 0, last_peer = -1, enabled = 0;
};
/* A query's fill: where its slice starts, how long the count said it is, and how far it has got. */
struct ModeDExportFill {
    long long base = 0;
    int expected = 0, written = 0, active = 0, enabled = 0;
};

/* Where count and fill leave their results.  Every array is indexed by the query's position in the batch. */
struct ModeDExportOutputs {
    int               *nodes = nullptr;          /* count: nodes each query sends */
    long long         *envelope_bound = nullptr; /* count: runs + floor(nodes / NODELISTLENGTH) */
    const long long   *node_offset = nullptr;    /* fill: each query's slice, from the scan of the counts */
    long long          node_offset_base = 0;     /* fill: where this batch's records start in that scan */
    const int         *node_count = nullptr;     /* fill: each query's slice length */
    ModeDExportRecord *records = nullptr;        /* fill */
    int               *anomaly = nullptr;        /* fill: a query that wrote more or fewer than its count */
};

enum {MODE_D_EXPORT_COUNT = 0, MODE_D_EXPORT_FILL = 1};

/* The work both passes share: a query's state at its start, one recorded top leaf, and its result at the end. */
template <int Pass>
struct ModeDExportStep {
    using State = typename std::conditional<Pass == MODE_D_EXPORT_COUNT, ModeDExportCount, ModeDExportFill>::type;
    ModeDExportTopology topo;
    ModeDExportOutputs  out;
    int                 first_active = 0;   /* the batch's queries are actives [first_active, first_active + n) */

    KOKKOS_INLINE_FUNCTION void begin(int kk, double reach, State &s) const
    {
        s = State{};
        /* the host export walk skips a query with no reach: a zero reach could still open the node holding it */
        s.enabled = (reach > 0) ? 1 : 0;
        if constexpr (Pass == MODE_D_EXPORT_FILL) {
            s.active = first_active + kk; s.base = out.node_offset[kk] - out.node_offset_base; s.expected = out.node_count[kk];
        }
    }
    KOKKOS_INLINE_FUNCTION void record(State &s, int leaf) const
    {
        if(!s.enabled) {return;}
        const int peer = topo.map.domain_task[leaf];
        if constexpr (Pass == MODE_D_EXPORT_COUNT) {
            s.nodes++;
            if(peer != s.last_peer) {s.runs++; s.last_peer = peer;}
        } else {
            /* never past its own slice, whatever the walk does: the next query's slice starts there */
            if(s.written < s.expected) {out.records[s.base + s.written] = ModeDExportRecord{peer, s.active, topo.map.domain_node_index[leaf]};}
            s.written++;
        }
    }
    KOKKOS_INLINE_FUNCTION void commit(int kk, const State &s) const
    {
        if constexpr (Pass == MODE_D_EXPORT_COUNT) {
            out.nodes[kk] = s.nodes;
            out.envelope_bound[kk] = (long long) s.runs + (long long) (s.nodes / NODELISTLENGTH);
        } else {
            if(s.written != s.expected) {Kokkos::atomic_store(out.anomaly, 1);}
        }
    }
};

/* The flat walk's policy for one query: the topology decides the nodes, the step keeps the state. */
template <int Pass>
struct ModeDExportFlatPolicy {
    static constexpr bool has_node_hook = true;
    const ModeDExportStep<Pass> *step;
    typename ModeDExportStep<Pass>::State s;
    KOKKOS_INLINE_FUNCTION GxNodeDecision node_decision(int kn) const {return step->topo.node_decision(kn);}
    KOKKOS_INLINE_FUNCTION GxNodeDecision pseudo_decision(int leaf) const {return step->topo.pseudo_decision(leaf);}
    KOKKOS_INLINE_FUNCTION void record(int leaf) {step->record(s, leaf);}
    KOKKOS_INLINE_FUNCTION void visit(int, double, double, double, double) {}   /* below a top leaf the walk never goes */
};

/* The packet walk's client: several queries share one traversal, each member's state in team scratch and published
 * only when its packet completes. */
template <int Pass, class ActiveData>
struct ModeDExportPacketClient {
    static constexpr bool evaluates = false;
    static constexpr bool has_node_hook = true;
    struct Accum {};
    using MemberState = typename ModeDExportStep<Pass>::State;
    ModeDExportStep<Pass> step;
    const ActiveData *queries = nullptr;   /* the batch, frozen */
    int n_local = 0, n_total = 0;
    GxDeviceTreeView tree;
    unsigned int type_mask = 0;            /* no type pruning: the host export walk prunes by reach alone */
    int *anomaly = nullptr;

    KOKKOS_INLINE_FUNCTION void query(int kk, double *x4) const
    {
        const ActiveData &a = queries[kk];
        x4[0] = (double) a.pos[0]; x4[1] = (double) a.pos[1]; x4[2] = (double) a.pos[2]; x4[3] = (double) a.h_search;
    }
    KOKKOS_INLINE_FUNCTION void starts(int, const int *&nodes, int &n) const {nodes = nullptr; n = 0;}
    KOKKOS_INLINE_FUNCTION void claim(int) const {}
    KOKKOS_INLINE_FUNCTION GxNodeDecision node_decision(int kn) const {return step.topo.node_decision(kn);}
    KOKKOS_INLINE_FUNCTION GxNodeDecision pseudo_decision(int leaf) const {return step.topo.pseudo_decision(leaf);}
    KOKKOS_INLINE_FUNCTION void member_begin(int kk, MemberState &s) const {step.begin(kk, (double) queries[kk].h_search, s);}
    KOKKOS_INLINE_FUNCTION void record(MemberState &s, int leaf) const {step.record(s, leaf);}
    KOKKOS_INLINE_FUNCTION void member_commit(int kk, const MemberState &s) const {step.commit(kk, s);}
};

/* ---------------------------------------------------------------------------------------------------------------
 * The stable partition.  Items are sorted by an unsigned key below 2^key_bits, least significant bit first, one bit
 * per pass: each pass sends the items whose bit is clear to the front and the rest after them, both in their existing
 * order, so the result is ordered by key and, among equal keys, by the order the items arrived in.  The key range is
 * the caller's (a peer is below the task count), so the passes are known on the host and each pass's split point
 * stays on the device (`n_clear`, one int): nothing waits for the host.  A bit every key shares moves nothing.
 * `scan` holds n ints.  Returns whichever of the two buffers holds the result. */
template <class Item, class KeyFn>
Item *mode_d_stable_radix(Item *items, Item *spare, int *scan, int n, KeyFn key, int key_bits, int *n_clear)
{
    if(n <= 1) {return items;}
    for(int bit = 0; bit < key_bits; bit++) {
        const Item *src = items;
        Item *dst = spare;
        Kokkos::parallel_scan("mode_d_radix_scan", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i, int &before, const bool last) {
            const int clear = ((key(src[i]) >> bit) & 1u) ? 0 : 1;
            if(last) {scan[i] = before; if(i == n - 1) {*n_clear = before + clear;}}
            before += clear;
        });
        Kokkos::parallel_for("mode_d_radix_place", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i) {
            const int clear = ((key(src[i]) >> bit) & 1u) ? 0 : 1;
            dst[clear ? scan[i] : *n_clear + (i - scan[i])] = src[i];
        });
        spare = items; items = dst;
    }
    return items;
}
/* The bits a key below `bound` needs. */
inline int mode_d_key_bits(long long bound) {int b = 0; for(long long v = bound - 1; v > 0; v >>= 1) {b++;} return b;}

/* Plain byte copies between device storage and the host, complete on return. */
using ModeDDeviceBytes = Kokkos::View<char *, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ModeDHostBytes   = Kokkos::View<char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
/* The exchange's copy queue, made and released with the persistent staging (gpu_neighbor_list.cc); valid while it is held. */
const Kokkos::DefaultExecutionSpace &gx_mode_d_copy_space(void);

inline void mode_d_copy_to_host(void *host_dst, const void *device_src, size_t bytes)
{
    if(bytes == 0) {return;}
    Kokkos::deep_copy(ModeDHostBytes((char *) host_dst, bytes), ModeDDeviceBytes((char *) device_src, bytes));
}
inline void mode_d_copy_to_device(void *device_dst, const void *host_src, size_t bytes)
{
    if(bytes == 0) {return;}
    Kokkos::deep_copy(ModeDDeviceBytes((char *) device_dst, bytes), ModeDHostBytes((char *) host_src, bytes));
}

/* The running maximum of the (peer, query) run starts.  A scan's partial results are joined by its join: a maximum
 * here.  (A lambda scan always joins by addition, which gives a wrong maximum once the range is split across threads.) */
struct ModeDRunStartScan {
    using value_type = int;
    const ModeDExportRecord *rec;
    int *run_start;
    KOKKOS_INLINE_FUNCTION void init(int &v) const {v = 0;}
    KOKKOS_INLINE_FUNCTION void join(int &dst, const int &src) const {if(src > dst) {dst = src;}}
    KOKKOS_INLINE_FUNCTION void operator()(int i, int &latest, const bool last) const
    {
        const int starts = (i == 0 || rec[i].peer != rec[i - 1].peer || rec[i].active != rec[i - 1].active) ? i : 0;
        if(starts > latest) {latest = starts;}
        if(last) {run_start[i] = latest;}
    }
};

/* The round's records, sorted by peer, become envelopes: a new envelope where the (peer, query) pair changes, or
 * where NODELISTLENGTH nodes of one pair have filled one.  `first_record[e]` is each envelope's first record,
 * `envelope_of[i]` the envelope record i goes in, `run_start[i]` where its (peer, query) run starts, and
 * `*n_envelopes` (device) their number. */
inline void mode_d_envelope_boundaries(const ModeDExportRecord *rec, int n, int *run_start, int *first_record, int *envelope_of,
                                       int *n_envelopes)
{
    if(n <= 0) {Kokkos::parallel_for("mode_d_envelope_none", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {*n_envelopes = 0;}); return;}
    /* where each record's (peer, query) run starts: an inclusive maximum over the run starts so far */
    Kokkos::parallel_scan("mode_d_run_start", Kokkos::RangePolicy<>(0, n), ModeDRunStartScan{rec, run_start});
    Kokkos::parallel_scan("mode_d_envelope_first", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i, int &before, const bool last) {
        const int opens = ((i - run_start[i]) % NODELISTLENGTH == 0) ? 1 : 0;
        if(last) {
            if(opens) {first_record[before] = i;}
            envelope_of[i] = before + opens - 1;
            if(i == n - 1) {*n_envelopes = before + opens;}
        }
        before += opens;
    });
}

/* One lane per envelope: the query it carries (from the frozen snapshot), up to NODELISTLENGTH nodes of one
 * (peer, query) run; and a count of envelopes per peer.  Envelopes come out in send order, so each peer's are
 * contiguous.  Launched over the records, which bound the envelopes, so the envelope count need not come to the
 * host first. */
template <class ActiveData>
inline void mode_d_pack_envelopes(const ModeDExportRecord *rec, int n_records, const int *first_record, const int *n_envelopes,
                                  const ActiveData *snapshot, int this_task, NlrQueryEnvelope<ActiveData> *envelopes,
                                  int *per_peer, int n_peers)
{
    if(n_records <= 0) {return;}
    Kokkos::parallel_for("mode_d_pack", Kokkos::RangePolicy<>(0, n_records), KOKKOS_LAMBDA(int e) {
        if(e >= *n_envelopes) {return;}
        const int i0 = first_record[e];
        const ModeDExportRecord r0 = rec[i0];
        NlrQueryEnvelope<ActiveData> &env = envelopes[e];
        int c = 0;
        for(; c < NODELISTLENGTH && i0 + c < n_records && rec[i0 + c].peer == r0.peer && rec[i0 + c].active == r0.active; c++) {
            env.NodeList[c] = rec[i0 + c].node;
        }
        for(int t = c; t < NODELISTLENGTH; t++) {env.NodeList[t] = -1;}
        env.origin_slot = r0.active; env.origin_rank = this_task; env.n_nodes = c; env.reserved_wire_padding = 0;
        env.active = snapshot[r0.active];
        if(r0.peer >= 0 && r0.peer < n_peers) {Kokkos::atomic_add(&per_peer[r0.peer], 1);}   /* likewise, never outside the table */
    });
}

/* Each query's runs, listed once: a (peer, query) run of the sorted records is one entry -- its peer, the envelope it
 * starts in, its length -- placed in the query's own list (`offsets[k]` .. `offsets[k + 1]`, query k of the round).  The
 * order within a list is whatever the placement gave; the merge sorts it.  `offsets` holds 2 (n_round + 1) ints (the
 * second half is the placement's cursors); linear in the records. */
inline void mode_d_query_runs(const ModeDExportRecord *sorted, int n, const int *run_start, const int *envelope_of, int first_active,
                              int n_round, int *offsets, ModeDQueryRun *runs)
{
    int *cursor = offsets + n_round + 1;
    Kokkos::parallel_for("mode_d_runs_clear", Kokkos::RangePolicy<>(0, 2 * (n_round + 1)), KOKKOS_LAMBDA(int k) {offsets[k] = 0;});
    if(n <= 0 || n_round <= 0) {return;}
    /* a run ends where the next record's (peer, query) differs */
    Kokkos::parallel_for("mode_d_runs_count", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i) {
        const int ends = (i == n - 1 || sorted[i + 1].peer != sorted[i].peer || sorted[i + 1].active != sorted[i].active);
        if(ends) {Kokkos::atomic_add(&offsets[sorted[i].active - first_active + 1], 1);}
    });
    Kokkos::parallel_scan("mode_d_runs_offsets", Kokkos::RangePolicy<>(0, n_round + 1), KOKKOS_LAMBDA(int k, int &sum, const bool last) {
        sum += offsets[k];
        if(last) {offsets[k] = sum;}
    });
    Kokkos::parallel_for("mode_d_runs_place", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i) {
        const int ends = (i == n - 1 || sorted[i + 1].peer != sorted[i].peer || sorted[i + 1].active != sorted[i].active);
        if(!ends) {return;}
        const int k = sorted[i].active - first_active, start = run_start[i];
        const int slot = offsets[k] + Kokkos::atomic_fetch_add(&cursor[k], 1);
        runs[slot] = ModeDQueryRun{sorted[i].peer, envelope_of[start], i - start + 1};
    });
}

/* One lane per query of the round: its own result, then each reply to it -- peers ascending, each peer's envelopes
 * in send order, the order the host merges in, which a reduction across lanes would change.  The query first orders
 * its run list by peer (a shell sort in place over its p runs, one per peer, so the order is unique; at most p^2 steps
 * for a query reaching every peer, against the records x peers rescans this replaced); a run's records are cut every
 * NODELISTLENGTH, so its envelopes are consecutive from its first.  A reply that does not name this rank and this query
 * is a broken exchange: it is not merged and `bad` is set. */
template <class Spec>
inline void mode_d_merge_replies(ModeDQueryRun *runs, const int *offsets, int first_active, int n_round,
                                 const NlrReplyEnvelope<typename Spec::AccumData> *replies, int this_task,
                                 typename Spec::AccumData *accums, int *bad)
{
    if(n_round <= 0) {return;}
    Kokkos::parallel_for("mode_d_merge", Kokkos::RangePolicy<>(0, n_round), KOKKOS_LAMBDA(int k) {
        const int a = first_active + k, r0 = offsets[k], n_runs = offsets[k + 1] - offsets[k];
        ModeDQueryRun *list = runs + r0;
        for(int gap = n_runs / 2; gap > 0; gap /= 2) {
            for(int x = gap; x < n_runs; x++) {
                const ModeDQueryRun v = list[x];
                int y = x;
                for(; y >= gap && list[y - gap].peer > v.peer; y -= gap) {list[y] = list[y - gap];}
                list[y] = v;
            }
        }
        typename Spec::AccumData acc = accums[a];
        for(int u = 0; u < n_runs; u++) {
            const int e0 = list[u].first_envelope, n_env = (list[u].records + NODELISTLENGTH - 1) / NODELISTLENGTH;
            for(int e = e0; e < e0 + n_env; e++) {
                const NlrReplyEnvelope<typename Spec::AccumData> &r = replies[e];
                if(r.origin_rank != this_task || r.origin_slot != a) {Kokkos::atomic_store(bad, 1); continue;}
                Spec::merge_accum(acc, r.accum);
            }
        }
        accums[a] = acc;
    });
}

#endif /* GIZMO_MODE_D_EXPORT_H */
