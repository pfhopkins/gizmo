/* mesh/mode_b_p2p_transport.h
 *
 * Point-to-point query/reply transport for the request-driven neighbour loops.
 *
 * Contract:
 *   - No physics in this header or its .cc. Pure transport.
 *   - Self-rank is EXCLUDED from the exchange. Caller handles self locally.
 *   - All point-to-point. NO collectives in this transport.
 *
 * Two layers:
 *
 *   BoundedPeerExchangeCore -- the protocol. It owns the per-peer counts and the
 *   MPI request bookkeeping of one round and nothing else: every payload byte
 *   lives in storage the caller owns and hands in as one pointer per peer. It
 *   allocates no payload memory and assumes no memory space; a pointer handed to
 *   it must simply be one this MPI can send from or receive into. Its metadata
 *   vectors keep their capacity, so one core reused across rounds allocates only
 *   while its peer count grows.
 *
 *   ModeBBoundedExchange<TQuery, TReply> -- the typed, vector-owning form Mode B
 *   has always used. It keeps every buffer it is given or makes until finish(),
 *   so its MPI calls, their order and its overlap are exactly those of the
 *   one-class transport it replaced.
 *
 * Choreography per round (every rank, sender and receiver roles at once):
 *   exchange_counts()       counts exchange (all Irecvs, then all Isends, one Waitall).
 *   post_query_sends()      Isend every non-empty query payload; the caller's buffers
 *                           must stay valid and unmodified until finish().
 *   post_reply_recvs()      pre-post every reply Irecv, sized by this rank's own sent
 *                           counts (bounded by its own send cap, not by what others do).
 *   select_group()          accumulate WHOLE peers in ascending rank order until the
 *   receive_group()         next would exceed the budget (always >= 1 peer); then
 *                           post + wait that group's query Irecvs into caller storage.
 *   send_group_replies()    Isend the group's replies from caller storage; returns a
 *                           ticket. wait_group_replies(ticket) completes just that
 *                           group's sends, after which its buffers may be reused --
 *                           the bounded double-buffered pipeline needs this. A caller
 *                           that holds its buffers to the end never calls it, and
 *                           finish() waits them instead.
 *   finish()                Waitall remaining query Isends + reply Isends + reply
 *                           Irecvs; byte-assert every reply's size via MPI_Get_count
 *                           (a peer answering with the wrong count is a broken
 *                           protocol, aborted loudly).
 *
 * Deadlock-freedom: group query-Irecv waits are satisfied by peers' query Isends,
 * which depend only on the counts exchange (always completes); reply-send waits are
 * satisfied by the origins' pre-posted reply Irecvs, posted BEFORE any rank enters
 * its group loop; finish()'s waits come last. No rank blocks on a send before posting
 * all its receives of the matching class -> the dependency graph is acyclic. Each
 * (sender,receiver) pair exchanges one count + one query payload + one reply payload
 * per round, so the tags stay unambiguous (callers drain every request before the
 * next round begins).
 *
 * Payloads go through MPI as bytes, so a typed payload must be trivially copyable.
 */

#ifndef MODE_B_P2P_TRANSPORT_H
#define MODE_B_P2P_TRANSPORT_H

#include <mpi.h>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../declarations/allvars.h"   /* ThisTask, NTask */

class BoundedPeerExchangeCore {
public:
    /* sent_counts[p] = queries this rank sends to peer p (0 for self).
     * recv_counts[p] = queries this rank receives from peer p. */
    std::vector<int> sent_counts;
    std::vector<int> recv_counts;

    /* Size every metadata array for ntask peers. Returns 0 when granted, 1 when memory was
     * refused. After a successful reserve no call on this core allocates: a caller that
     * must not fail inside the protocol reserves before it commits to the exchange. */
    int  reserve(int ntask);
    /* Give the metadata memory back. */
    void release_storage();

    /* Open a round: forget the previous round's requests, exchange counts.
     * counts_to_peer has NTask entries; the self entry is ignored. */
    void exchange_counts(const int *counts_to_peer, double *dt_count_exch = nullptr);

    /* query_to_peer[p] holds sent_counts[p] queries of query_bytes each. */
    void post_query_sends(const void *const *query_to_peer, size_t query_bytes);

    /* reply_from_peer[p] has room for sent_counts[p] replies of reply_bytes each. */
    void post_reply_recvs(void *const *reply_from_peer, size_t reply_bytes);

    /* The next whole-peer group of incoming queries, ascending rank order, while its
     * (query + reply) payload stays within budget_bytes; always >= 1 peer. A single
     * peer's payload is bounded by that sender's own cap, so an oversized one-peer
     * group cannot exceed one sender bunch. false when every peer has been consumed. */
    bool select_group(size_t budget_bytes, size_t query_bytes, size_t reply_bytes,
                      std::vector<int>& group_peers);

    /* Receive the selected group's queries: dst[i] gets recv_counts[group_peers[i]]
     * queries from group_peers[i], in send order. Returns when all have arrived. */
    void receive_group(const std::vector<int>& group_peers, void *const *dst,
                       size_t query_bytes, double *dt_query_recv_wait = nullptr);

    /* Isend the group's replies: src[i] holds recv_counts[group_peers[i]] replies for
     * group_peers[i], one per received query in received order. The storage must stay
     * valid until wait_group_replies(ticket) or finish(). */
    int  send_group_replies(const std::vector<int>& group_peers, const void *const *src,
                            size_t reply_bytes);
    void wait_group_replies(int ticket);

    /* Close the round. Every reply_from_peer buffer then holds its sent_counts[p]
     * replies. */
    void finish(size_t reply_bytes, double *dt_reply_finish_wait = nullptr);

private:
    std::vector<MPI_Request> reqs_count;
    std::vector<MPI_Request> reqs_query_send;
    std::vector<MPI_Request> reqs_reply_recv;
    std::vector<int>         reply_recv_peer;      /* parallel to reqs_reply_recv */
    std::vector<MPI_Status>  reply_recv_status;
    std::vector<MPI_Request> reqs_group_query;
    std::vector<MPI_Request> reqs_reply_send;
    std::vector<int>         reply_send_group_start;   /* ticket -> first request in reqs_reply_send */
    int next_peer = 0;
};

/* A group's reply count differs from the queries it received: a caller bug, stopped hard. */
void bounded_peer_exchange_reply_count_mismatch(int peer, size_t have, int want);

template <typename TQuery, typename TReply>
class ModeBBoundedExchange {
    static_assert(std::is_trivially_copyable<TQuery>::value,
                  "TQuery must be trivially copyable for byte-level MPI transfer");
    static_assert(std::is_trivially_copyable<TReply>::value,
                  "TReply must be trivially copyable for byte-level MPI transfer");

    BoundedPeerExchangeCore core;   /* declared first: the count references below bind to it */

public:
    std::vector<int>& sent_counts = core.sent_counts;
    std::vector<int>& recv_counts = core.recv_counts;

    /* The count references bind to this object's own core: a copy or move would keep them
     * bound to the original. */
    ModeBBoundedExchange() = default;
    ModeBBoundedExchange(const ModeBBoundedExchange&) = delete;
    ModeBBoundedExchange& operator=(const ModeBBoundedExchange&) = delete;

    /* Counts exchange + post all query-payload Isends + all reply Irecvs.
     * `queries_per_peer` is caller-owned and MUST stay alive and unmodified
     * until finish() returns (the Isends read from its storage).
     * queries_per_peer[ThisTask] must be empty. */
    void begin(const std::vector<std::vector<TQuery>>& queries_per_peer,
               double *dt_count_exch = nullptr)
    {
        const int nt = NTask;
        recv_replies.assign(nt, std::vector<TReply>{});
        held_reply_bufs.clear();
        if(nt <= 1) {core.sent_counts.assign(nt, 0); core.recv_counts.assign(nt, 0); return;}
        std::vector<int> counts(nt, 0);
        std::vector<const void *> send_ptr(nt, nullptr);
        std::vector<void *> recv_ptr(nt, nullptr);
        for(int p = 0; p < nt; p++) {
            if(p == ThisTask) continue;
            counts[p] = (int)queries_per_peer[p].size();
            send_ptr[p] = queries_per_peer[p].data();
        }
        core.exchange_counts(counts.data(), dt_count_exch);
        core.post_query_sends(send_ptr.data(), sizeof(TQuery));
        for(int p = 0; p < nt; p++) {
            if(p == ThisTask || core.sent_counts[p] <= 0) continue;
            recv_replies[p].resize(core.sent_counts[p]);
            recv_ptr[p] = recv_replies[p].data();
        }
        core.post_reply_recvs(recv_ptr.data(), sizeof(TReply));
    }

    /* Stage the next whole-peer group of incoming queries (see the core).
     * group_queries[i] holds the envelopes from group_peers[i], in send order. */
    bool next_group(size_t budget_bytes,
                    std::vector<int>& group_peers,
                    std::vector<std::vector<TQuery>>& group_queries,
                    double *dt_query_recv_wait = nullptr)
    {
        group_queries.clear();
        if(!core.select_group(budget_bytes, sizeof(TQuery), sizeof(TReply), group_peers)) {return false;}
        group_queries.resize(group_peers.size());
        std::vector<void *> dst(group_peers.size());
        for(size_t i = 0; i < group_peers.size(); i++) {
            group_queries[i].resize(core.recv_counts[group_peers[i]]);
            dst[i] = group_queries[i].data();
        }
        core.receive_group(group_peers, dst.data(), sizeof(TQuery), dt_query_recv_wait);
        return true;
    }

    /* Post this group's reply Isends and take OWNERSHIP of the reply buffers so the
     * caller can drop its group storage while the sends stay in flight. They are
     * waited once in finish(), so a group's send completion overlaps the round-end
     * reply wait (and later groups' work) instead of serializing at each group
     * boundary; the cost is reply buffers held per round rather than per group.
     * replies_for_group[i] must hold exactly recv_counts[group_peers[i]] entries. */
    void send_group_replies(const std::vector<int>& group_peers,
                            std::vector<std::vector<TReply>>&& replies_for_group)
    {
        std::vector<const void *> src(group_peers.size());
        for(size_t i = 0; i < group_peers.size(); i++) {
            if(replies_for_group[i].size() != (size_t)core.recv_counts[group_peers[i]]) {
                bounded_peer_exchange_reply_count_mismatch(group_peers[i], replies_for_group[i].size(),
                                                           core.recv_counts[group_peers[i]]);
            }
            /* Hold the buffer (its heap block survives the move) BEFORE the Isend reads it. */
            held_reply_bufs.push_back(std::move(replies_for_group[i]));
            src[i] = held_reply_bufs.back().data();
        }
        (void)core.send_group_replies(group_peers, src.data(), sizeof(TReply));
    }

    /* Wait out the round; returns recv_replies[p] with exactly sent_counts[p] entries. */
    std::vector<std::vector<TReply>> finish(double *dt_reply_finish_wait = nullptr)
    {
        core.finish(sizeof(TReply), dt_reply_finish_wait);
        held_reply_bufs.clear();   /* reply sends drained -- buffers free */
        return std::move(recv_replies);
    }

private:
    std::vector<std::vector<TReply>> recv_replies;
    std::vector<std::vector<TReply>> held_reply_bufs;   /* keep-alive for the reply Isends */
};

#endif /* MODE_B_P2P_TRANSPORT_H */
