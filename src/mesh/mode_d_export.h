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
 * the order the walk reached the nodes -- through a stable partition by peer, and the replies are merged per query
 * in that same order (the query's own result first, then peers ascending, then each peer's envelopes in order),
 * through a second stable partition, by query.  Both partitions are one primitive, a stable least-significant-bit
 * radix sort, so nothing here depends on atomics for ORDER (counts only) and nothing sizes a table by the task count.
 *
 * Include after <Kokkos_Core.hpp> and neighbor_loop_runner.h. */
#ifndef GIZMO_MODE_D_EXPORT_H
#define GIZMO_MODE_D_EXPORT_H

#include <Kokkos_Core.hpp>
#include "neighbor_list.h"
#include "device_tree_walk.h"
#include "gpu_neighbor_list.h"

/* One record of the export decision: query `active` sends top-level node `node` to `peer`. */
struct ModeDExportRecord { int peer; int active; int node; };

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
 * The stable partition.  Items are sorted by an unsigned key, least significant bit first, one bit per pass: each
 * pass sends the items whose bit is clear to the front and the rest after them, both in their existing order, so
 * the result is ordered by key and, among equal keys, by the order the items arrived in.  A bit that is the same for
 * every key moves nothing and is skipped, so a key range of one value costs no pass.  `scan` holds n ints.  Returns
 * whichever of the two buffers holds the result. */
template <class Item, class KeyFn>
Item *mode_d_stable_radix(Item *items, Item *spare, int *scan, int n, KeyFn key)
{
    if(n <= 1) {return items;}
    unsigned int key_or = 0, key_and = ~0u;
    Kokkos::parallel_reduce("mode_d_radix_bits", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i, unsigned int &o, unsigned int &a) {
        const unsigned int k = key(items[i]); o |= k; a &= k;
    }, Kokkos::BOr<unsigned int>(key_or), Kokkos::BAnd<unsigned int>(key_and));
    const unsigned int varying = key_or ^ key_and;
    for(int bit = 0; bit < 32; bit++) {
        if(!((varying >> bit) & 1u)) {continue;}
        const Item *src = items;
        Item *dst = spare;
        int n_clear = 0;
        Kokkos::parallel_scan("mode_d_radix_scan", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i, int &before, const bool last) {
            const int clear = ((key(src[i]) >> bit) & 1u) ? 0 : 1;
            if(last) {scan[i] = before;}
            before += clear;
        }, n_clear);
        Kokkos::parallel_for("mode_d_radix_place", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i) {
            const int clear = ((key(src[i]) >> bit) & 1u) ? 0 : 1;
            dst[clear ? scan[i] : n_clear + (i - scan[i])] = src[i];
        });
        spare = items; items = dst;
    }
    return items;
}

/* An envelope's position in the merge order: which query it answers (relative to the round's first), and which
 * envelope it is in send order. */
struct ModeDEnvelopeKey { int active; int envelope; };

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
 * where NODELISTLENGTH nodes of one pair have filled one.  Returns the number of envelopes; `first_record[e]` is
 * each envelope's first record.  `run_start` holds n ints. */
inline int mode_d_envelope_boundaries(const ModeDExportRecord *rec, int n, int *run_start, int *first_record)
{
    if(n <= 0) {return 0;}
    /* where each record's (peer, query) run starts: an inclusive maximum over the run starts so far */
    Kokkos::parallel_scan("mode_d_run_start", Kokkos::RangePolicy<>(0, n), ModeDRunStartScan{rec, run_start});
    int n_envelopes = 0;
    Kokkos::parallel_scan("mode_d_envelope_first", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i, int &before, const bool last) {
        const int opens = ((i - run_start[i]) % NODELISTLENGTH == 0) ? 1 : 0;
        if(last && opens) {first_record[before] = i;}
        before += opens;
    }, n_envelopes);
    return n_envelopes;
}

/* One lane per envelope: the query it carries (from the frozen snapshot), up to NODELISTLENGTH nodes of one
 * (peer, query) run, and which query it answers; and a count of envelopes per peer.  Envelopes come out in send
 * order, so each peer's are contiguous. */
template <class ActiveData>
inline void mode_d_pack_envelopes(const ModeDExportRecord *rec, int n_records, const int *first_record, int n_envelopes,
                                  const ActiveData *snapshot, int this_task, NlrQueryEnvelope<ActiveData> *envelopes,
                                  int *envelope_active, int *per_peer)
{
    if(n_envelopes <= 0) {return;}
    Kokkos::parallel_for("mode_d_pack", Kokkos::RangePolicy<>(0, n_envelopes), KOKKOS_LAMBDA(int e) {
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
        envelope_active[e] = r0.active;
        Kokkos::atomic_add(&per_peer[r0.peer], 1);
    });
}

/* The merge order: the round's envelopes grouped by the query they answer, keeping send order (peers ascending,
 * then each peer's envelopes) within one query, and where each query's group starts.  `keys` and `spare` hold
 * n_envelopes each, `scan` n_envelopes ints, `offsets` n_round + 1. Returns the buffer holding the result. */
inline ModeDEnvelopeKey *mode_d_merge_order(const int *envelope_active, int n_envelopes, int first_active, int n_round,
                                            ModeDEnvelopeKey *keys, ModeDEnvelopeKey *spare, int *scan, int *offsets)
{
    Kokkos::parallel_for("mode_d_merge_clear", Kokkos::RangePolicy<>(0, n_round + 1), KOKKOS_LAMBDA(int a) {offsets[a] = 0;});
    Kokkos::parallel_for("mode_d_merge_keys", Kokkos::RangePolicy<>(0, n_envelopes), KOKKOS_LAMBDA(int e) {
        const int a = envelope_active[e] - first_active;
        keys[e] = ModeDEnvelopeKey{a, e};
        Kokkos::atomic_add(&offsets[a + 1], 1);   /* a count: its result is the same in any order */
    });
    Kokkos::parallel_scan("mode_d_merge_offsets", Kokkos::RangePolicy<>(0, n_round + 1), KOKKOS_LAMBDA(int a, int &sum, const bool last) {
        sum += offsets[a];
        if(last) {offsets[a] = sum;}
    });
    return mode_d_stable_radix(keys, spare, scan, n_envelopes, KOKKOS_LAMBDA(const ModeDEnvelopeKey &k) {return (unsigned int) k.active;});
}

/* One lane per query of the round: its own result, then each reply to it in merge order.  The merge is serial per
 * query on purpose: it is the order the host merges in (the query's own result, peers ascending, each peer's
 * envelopes in order), which a reduction across lanes would change.  A reply that does not name this rank and this
 * query is a broken exchange: it is not merged and `bad` is set. */
template <class Spec>
inline void mode_d_merge_replies(const ModeDEnvelopeKey *order, const int *offsets, int first_active, int n_round,
                                 const NlrReplyEnvelope<typename Spec::AccumData> *replies, int this_task,
                                 typename Spec::AccumData *accums, int *bad)
{
    if(n_round <= 0) {return;}
    Kokkos::parallel_for("mode_d_merge", Kokkos::RangePolicy<>(0, n_round), KOKKOS_LAMBDA(int k) {
        const int a = first_active + k;
        typename Spec::AccumData acc = accums[a];
        for(int j = offsets[k]; j < offsets[k + 1]; j++) {
            const NlrReplyEnvelope<typename Spec::AccumData> &r = replies[order[j].envelope];
            if(r.origin_rank != this_task || r.origin_slot != a) {Kokkos::atomic_store(bad, 1); continue;}
            Spec::merge_accum(acc, r.accum);
        }
        accums[a] = acc;
    });
}

#endif /* GIZMO_MODE_D_EXPORT_H */
