/* mesh/mode_b_p2p_transport.h
 *
 * Generic typed point-to-point transport for Mode B neighbor loops.
 *
 * Contract (locked 2026-05-07):
 *   - No physics in this header or its .cc. Pure transport.
 *   - Self-rank is EXCLUDED from the exchange. Caller handles self locally.
 *   - All point-to-point. NO collectives in this transport.
 *
 * ModeBBoundedExchange is the single transport primitive (it replaced the
 * earlier one-shot mode_b_exchange_queries / mode_b_exchange_replies pair):
 * a stateful per-round exchange that lets the RECEIVER stage, evaluate, and
 * answer incoming queries in memory-bounded whole-peer groups instead of
 * materializing every peer's payload at once — the receiver-side analog of
 * the legacy import sub-chunking in system/code_block_xchange_perform_ops.h.
 * With a budget larger than the total incoming payload it degenerates to one
 * group == the old one-shot behavior.
 *
 * Choreography per round (every rank, sender and receiver roles at once):
 *   begin()               counts exchange; post ALL query-payload Isends
 *                          (buffers = caller's queries_per_peer, which MUST
 *                          outlive finish()); post ALL reply-payload Irecvs
 *                          (sized sent_counts[p] — bounded by the caller's own
 *                          send cap, NOT by what other ranks do).
 *   next_group(budget)     accumulate WHOLE peers in ascending rank order
 *                          until adding the next peer would exceed budget
 *                          (always >=1 peer); post + wait that group's
 *                          query-payload Irecvs; hand envelopes to caller.
 *   send_group_replies()   Isend the group's replies and Waitall THOSE sends
 *                          immediately, so the caller can free the group's
 *                          buffers before staging the next group. The Waitall
 *                          completes promptly because every origin pre-posted
 *                          its reply Irecvs in begin(); it can still block on
 *                          MPI progress while origins compute — if that shows
 *                          up in timings, defer these Waitalls (poll/test)
 *                          rather than assuming it cannot happen.
 *   finish()               Waitall remaining query-payload Isends + all reply
 *                          Irecvs; byte-assert every reply's size via
 *                          MPI_Get_count (replaces the retired reply-count
 *                          handshake message: same protocol-break loudness,
 *                          one less message per peer per round); return
 *                          recv_replies[p] with exactly sent_counts[p]
 *                          entries.
 *
 * Deadlock-freedom: group query-Irecv waits are satisfied by peers' begin()
 * Isends, which depend only on the counts exchange (always completes);
 * reply-send Waitalls are satisfied by origins' begin() pre-posted Irecvs,
 * posted BEFORE any rank enters its group loop; finish()'s waits come last.
 * No rank blocks on a send before posting all its receives of the matching
 * class -> the dependency graph is acyclic. Each (sender,receiver) pair
 * exchanges one count + one query payload + one reply payload per round, so
 * the tags stay unambiguous (callers drain every request before the next
 * round begins).
 *
 * Payload types must be trivially copyable (bytes go through MPI directly).
 */

#ifndef MODE_B_P2P_TRANSPORT_H
#define MODE_B_P2P_TRANSPORT_H

#include <mpi.h>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "../system/tags.h"
#include "../declarations/allvars.h"   /* ThisTask, NTask */

void gizmo_fatal_hard_exit_reviewed(int code, const char *msg,
                                    const char *file, int line, const char *func);

/* -----------------------------------------------------------------------
 * Diagnostics: total bytes shipped in the last exchange (sum across peers,
 * both directions). For the PHASE0_MODEB_NGL line. Reset by
 * mode_b_p2p_diag_reset(). All thread-unsafe / single-flow.
 * --------------------------------------------------------------------- */
struct mode_b_p2p_diag_t {
    size_t bytes_query_sent;
    size_t bytes_query_recv;
    size_t bytes_reply_sent;
    size_t bytes_reply_recv;
    int    peers_sent_to;       /* # peers with non-zero query send count */
    int    peers_recv_from;     /* # peers with non-zero query recv count */
};
void                  mode_b_p2p_diag_reset(void);
mode_b_p2p_diag_t     mode_b_p2p_diag_snapshot(void);

extern mode_b_p2p_diag_t g_mode_b_p2p_diag;   /* defined in .cc */

template <typename TQuery, typename TReply>
class ModeBBoundedExchange {
    static_assert(std::is_trivially_copyable<TQuery>::value,
                  "TQuery must be trivially copyable for byte-level MPI transfer");
    static_assert(std::is_trivially_copyable<TReply>::value,
                  "TReply must be trivially copyable for byte-level MPI transfer");

public:
    /* sent_counts[p] = queries this rank sent to peer p (0 for self).
     * recv_counts[p] = queries this rank will receive from peer p. */
    std::vector<int> sent_counts;
    std::vector<int> recv_counts;

    /* Counts exchange + post all query-payload Isends + all reply Irecvs.
     * `queries_per_peer` is caller-owned and MUST stay alive and unmodified
     * until finish() returns (the Isends read from its storage).
     * queries_per_peer[ThisTask] must be empty. */
    void begin(const std::vector<std::vector<TQuery>>& queries_per_peer)
    {
        const int nt = NTask;
        sent_counts.assign(nt, 0);
        recv_counts.assign(nt, 0);
        recv_replies.assign(nt, std::vector<TReply>{});
        next_peer = 0;
        if(nt <= 1) return;

        for(int p = 0; p < nt; p++) {
            if(p == ThisTask) continue;
            sent_counts[p] = (int)queries_per_peer[p].size();
        }

        /* Counts. */
        std::vector<MPI_Request> reqs_cnt;
        reqs_cnt.reserve(2 * (nt - 1));
        for(int p = 0; p < nt; p++) {
            if(p == ThisTask) continue;
            MPI_Request rq;
            MPI_Irecv(&recv_counts[p], 1, MPI_INT, p, TAG_MODE_B_QUERY_COUNT,
                      MPI_COMM_WORLD, &rq);
            reqs_cnt.push_back(rq);
        }
        for(int p = 0; p < nt; p++) {
            if(p == ThisTask) continue;
            MPI_Request rq;
            MPI_Isend(&sent_counts[p], 1, MPI_INT, p, TAG_MODE_B_QUERY_COUNT,
                      MPI_COMM_WORLD, &rq);
            reqs_cnt.push_back(rq);
        }
        MPI_Waitall((int)reqs_cnt.size(), reqs_cnt.data(), MPI_STATUSES_IGNORE);

        /* All query-payload Isends (buffers caller-owned; waited in finish). */
        for(int p = 0; p < nt; p++) {
            if(p == ThisTask || sent_counts[p] <= 0) continue;
            const size_t nbytes = (size_t)sent_counts[p] * sizeof(TQuery);
            MPI_Request rq;
            MPI_Isend(queries_per_peer[p].data(), (int)nbytes, MPI_BYTE,
                      p, TAG_MODE_B_QUERY_PAYLOAD, MPI_COMM_WORLD, &rq);
            reqs_query_send.push_back(rq);
            g_mode_b_p2p_diag.bytes_query_sent += nbytes;
            g_mode_b_p2p_diag.peers_sent_to++;
        }

        /* All reply Irecvs, pre-posted so receivers' per-group reply sends can
         * complete (and their buffers be freed) before the round ends. Sized
         * by MY sent counts — bounded by my own send cap. */
        for(int p = 0; p < nt; p++) {
            if(p == ThisTask || sent_counts[p] <= 0) continue;
            recv_replies[p].resize(sent_counts[p]);
            const size_t nbytes = (size_t)sent_counts[p] * sizeof(TReply);
            MPI_Request rq;
            MPI_Irecv(recv_replies[p].data(), (int)nbytes, MPI_BYTE,
                      p, TAG_MODE_B_REPLY_PAYLOAD, MPI_COMM_WORLD, &rq);
            reqs_reply_recv.push_back(rq);
            reply_recv_peer.push_back(p);
            g_mode_b_p2p_diag.bytes_reply_recv += nbytes;
        }
    }

    /* Stage the next whole-peer group of incoming queries. Peers are consumed
     * in ascending rank order; a group grows while its (query + reply) payload
     * stays within budget_bytes, and always contains at least one peer (a
     * single peer's payload is bounded by that sender's own cap, so an
     * oversized single-peer group cannot exceed one sender-bunch). Returns
     * false when every peer has been consumed. group_queries[i] holds the
     * envelopes from group_peers[i], in send order. */
    bool next_group(size_t budget_bytes,
                    std::vector<int>& group_peers,
                    std::vector<std::vector<TQuery>>& group_queries)
    {
        const int nt = NTask;
        group_peers.clear();
        group_queries.clear();
        size_t group_bytes = 0;
        while(next_peer < nt) {
            const int p = next_peer;
            if(p == ThisTask || recv_counts[p] <= 0) { next_peer++; continue; }
            const size_t peer_bytes =
                (size_t)recv_counts[p] * (sizeof(TQuery) + sizeof(TReply));
            if(!group_peers.empty() && group_bytes + peer_bytes > budget_bytes) break;
            group_peers.push_back(p);
            group_bytes += peer_bytes;
            next_peer++;
        }
        if(group_peers.empty()) return false;

        group_queries.resize(group_peers.size());
        std::vector<MPI_Request> reqs;
        reqs.reserve(group_peers.size());
        for(size_t i = 0; i < group_peers.size(); i++) {
            const int p = group_peers[i];
            group_queries[i].resize(recv_counts[p]);
            const size_t nbytes = (size_t)recv_counts[p] * sizeof(TQuery);
            MPI_Request rq;
            MPI_Irecv(group_queries[i].data(), (int)nbytes, MPI_BYTE,
                      p, TAG_MODE_B_QUERY_PAYLOAD, MPI_COMM_WORLD, &rq);
            reqs.push_back(rq);
            g_mode_b_p2p_diag.bytes_query_recv += nbytes;
            g_mode_b_p2p_diag.peers_recv_from++;
        }
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        return true;
    }

    /* Send this group's replies and wait for THOSE sends to complete, so the
     * caller can free/reuse the group's storage immediately. The wait is
     * deadlock-free (matching Irecvs were pre-posted in every origin's
     * begin()) but can still block on MPI progress while origins compute; if
     * profiling shows stalls here, defer these waits to finish() at the cost
     * of holding reply buffers per round instead of per group.
     * replies_for_group[i] must hold exactly recv_counts[group_peers[i]]
     * entries (one reply per received query, in received order). */
    void send_group_replies(const std::vector<int>& group_peers,
                            const std::vector<std::vector<TReply>>& replies_for_group)
    {
        std::vector<MPI_Request> reqs;
        reqs.reserve(group_peers.size());
        for(size_t i = 0; i < group_peers.size(); i++) {
            const int p = group_peers[i];
            if((int)replies_for_group[i].size() != recv_counts[p]) {
                fprintf(stderr, "[mode_b ABORT rank=%d] send_group_replies: %zu replies "
                        "for peer %d, expected %d. Caller bug.\n",
                        ThisTask, replies_for_group[i].size(), p, recv_counts[p]);
                fflush(stderr);
                gizmo_fatal_hard_exit_reviewed(90002003,
                    "REVIEWED_HARD_MID_PROTOCOL: mode_b transport group reply-count mismatch",
                    __FILE__, __LINE__, __FUNCTION__);
            }
            if(recv_counts[p] <= 0) continue;
            const size_t nbytes = (size_t)recv_counts[p] * sizeof(TReply);
            MPI_Request rq;
            MPI_Isend(replies_for_group[i].data(), (int)nbytes, MPI_BYTE,
                      p, TAG_MODE_B_REPLY_PAYLOAD, MPI_COMM_WORLD, &rq);
            reqs.push_back(rq);
            g_mode_b_p2p_diag.bytes_reply_sent += nbytes;
        }
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    /* Wait out the remaining query-payload Isends and all reply Irecvs.
     * Every reply's byte count is asserted against sent_counts[p] *
     * sizeof(TReply) via MPI_Get_count — a peer answering with the wrong
     * count is a broken protocol, aborted loudly (this check replaced the
     * retired reply-count handshake message). Returns recv_replies[p] with
     * exactly sent_counts[p] entries. */
    std::vector<std::vector<TReply>> finish()
    {
        MPI_Waitall((int)reqs_query_send.size(), reqs_query_send.data(),
                    MPI_STATUSES_IGNORE);
        std::vector<MPI_Status> stats(reqs_reply_recv.size());
        MPI_Waitall((int)reqs_reply_recv.size(), reqs_reply_recv.data(),
                    stats.data());
        for(size_t i = 0; i < stats.size(); i++) {
            int got_bytes = 0;
            MPI_Get_count(&stats[i], MPI_BYTE, &got_bytes);
            const int p = reply_recv_peer[i];
            const long long want_bytes = (long long)sent_counts[p] * (long long)sizeof(TReply);
            if((long long)got_bytes != want_bytes) {
                fprintf(stderr, "[mode_b ABORT rank=%d] reply payload from peer %d is "
                        "%d bytes, expected %lld (%d replies x %zu B). Protocol broken.\n",
                        ThisTask, p, got_bytes, want_bytes, sent_counts[p], sizeof(TReply));
                fflush(stderr);
                gizmo_fatal_hard_exit_reviewed(90002003,
                    "REVIEWED_HARD_MID_PROTOCOL: mode_b transport reply-size mismatch",
                    __FILE__, __LINE__, __FUNCTION__);
            }
        }
        reqs_query_send.clear();
        reqs_reply_recv.clear();
        reply_recv_peer.clear();
        return std::move(recv_replies);
    }

private:
    std::vector<std::vector<TReply>> recv_replies;
    std::vector<MPI_Request> reqs_query_send;
    std::vector<MPI_Request> reqs_reply_recv;
    std::vector<int>         reply_recv_peer;   /* parallel to reqs_reply_recv */
    int next_peer = 0;
};

#endif /* MODE_B_P2P_TRANSPORT_H */
