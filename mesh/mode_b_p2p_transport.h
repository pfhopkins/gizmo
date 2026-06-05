/* mesh/mode_b_p2p_transport.h
 *
 * Generic typed point-to-point transport for Mode B tiny-N neighbor loops.
 *
 * Contract (locked 2026-05-07; see ~/.claude/memory/reference_neighbor_loop_contract.md):
 *   - No physics in this header or its .cc. Pure transport.
 *   - Self-rank is EXCLUDED from the exchange. Caller handles self locally.
 *   - Two-stage exchange: count headers first (MPI_Isend/Irecv int per peer),
 *     then payload Isend/Irecv. Waitall in between.
 *   - State object retains per-peer send/recv counts so the reply phase can
 *     assert symmetric sizing without re-deriving it from elsewhere.
 *   - All point-to-point. NO collectives in this transport.
 *
 * Templated on the payload type (TQuery, TReply). Payload types must be
 * trivially copyable / standard layout (we MPI_Isend their bytes directly).
 */

#ifndef MODE_B_P2P_TRANSPORT_H
#define MODE_B_P2P_TRANSPORT_H

#include <mpi.h>
#include <vector>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "../system/tags.h"
#include "../declarations/allvars.h"   /* ThisTask, NTask */

/* -----------------------------------------------------------------------
 * Exchange state — retained between query phase and reply phase.
 * The reply phase asserts incoming reply count from peer p equals
 * sent_counts[p], so reply sizing cannot accidentally drift.
 * --------------------------------------------------------------------- */
template <typename TQuery>
struct mode_b_query_exchange_state_t
{
    /* sent_counts[p] = number of queries this rank sent to peer p
     * (0 for p == ThisTask; self is excluded). */
    std::vector<int> sent_counts;

    /* recv_counts[p] = number of queries this rank received from peer p
     * (0 for p == ThisTask). */
    std::vector<int> recv_counts;

    /* recv_queries[p] = those queries from peer p in the order received.
     * Empty for p == ThisTask. */
    std::vector<std::vector<TQuery>> recv_queries;
};

/* -----------------------------------------------------------------------
 * Phase 1 — exchange queries with all peers (excluding self).
 *
 * my_queries_to_each_peer[p] is the list of queries this rank sends to
 * peer p. my_queries_to_each_peer[ThisTask] MUST be empty (self handled
 * by caller). Returns the state object carrying the recv side + counts.
 *
 * Implementation:
 *   1. Post non-blocking recv for one int (recv_count) from each peer.
 *   2. Post non-blocking send of one int (send_count = my_queries_to_each_peer[p].size())
 *      to each peer.
 *   3. Waitall the count exchange.
 *   4. Post non-blocking recv for recv_count[p] * sizeof(TQuery) bytes
 *      from each peer with non-zero count.
 *   5. Post non-blocking send of payload to each peer with non-zero count.
 *   6. Waitall payload.
 * --------------------------------------------------------------------- */
template <typename TQuery>
mode_b_query_exchange_state_t<TQuery>
mode_b_exchange_queries(const std::vector<std::vector<TQuery>>& my_queries_to_each_peer);

/* -----------------------------------------------------------------------
 * Phase 2 — exchange replies. Requires the state object from phase 1.
 *
 * my_replies_to_each_peer[p] must have exactly state.recv_counts[p] entries
 * (one reply per query I received from peer p). Replies for self
 * (p == ThisTask) MUST be empty — caller handled self via direct function
 * calls, not transport.
 *
 * Returns recv_replies[p] containing exactly state.sent_counts[p] entries
 * (replies to queries I originally sent to peer p).
 *
 * Asserts inbound count from peer p == state.sent_counts[p]. If a peer
 * misbehaves and sends the wrong count, abort loudly — symmetric sizing
 * is a transport invariant.
 * --------------------------------------------------------------------- */
template <typename TQuery, typename TReply>
std::vector<std::vector<TReply>>
mode_b_exchange_replies(
    const std::vector<std::vector<TReply>>& my_replies_to_each_peer,
    const mode_b_query_exchange_state_t<TQuery>& state);

/* -----------------------------------------------------------------------
 * Diagnostics: total bytes shipped in the last query exchange (sum across
 * peers, both directions). For the PHASE0_MODEB_NGL line. Reset by
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

/* -----------------------------------------------------------------------
 * Template implementations — header-included for instantiation.
 * --------------------------------------------------------------------- */

extern mode_b_p2p_diag_t g_mode_b_p2p_diag;   /* defined in .cc */

template <typename TQuery>
mode_b_query_exchange_state_t<TQuery>
mode_b_exchange_queries(const std::vector<std::vector<TQuery>>& my_queries_to_each_peer)
{
    static_assert(std::is_trivially_copyable<TQuery>::value,
                  "TQuery must be trivially copyable for byte-level MPI transfer");

    const int nt = NTask;
    mode_b_query_exchange_state_t<TQuery> state;
    state.sent_counts.assign(nt, 0);
    state.recv_counts.assign(nt, 0);
    state.recv_queries.assign(nt, std::vector<TQuery>{});

    if(nt <= 1) return state;   /* nothing to exchange */

    /* Phase 1a — count exchange. */
    std::vector<MPI_Request> reqs_count_send;
    std::vector<MPI_Request> reqs_count_recv;
    reqs_count_send.reserve(nt - 1);
    reqs_count_recv.reserve(nt - 1);

    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        state.sent_counts[p] = (int)my_queries_to_each_peer[p].size();
    }

    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        MPI_Request rq;
        MPI_Irecv(&state.recv_counts[p], 1, MPI_INT, p, TAG_MODE_B_QUERY_COUNT,
                  MPI_COMM_WORLD, &rq);
        reqs_count_recv.push_back(rq);
    }
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        MPI_Request rq;
        MPI_Isend(&state.sent_counts[p], 1, MPI_INT, p, TAG_MODE_B_QUERY_COUNT,
                  MPI_COMM_WORLD, &rq);
        reqs_count_send.push_back(rq);
    }
    MPI_Waitall((int)reqs_count_recv.size(), reqs_count_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall((int)reqs_count_send.size(), reqs_count_send.data(), MPI_STATUSES_IGNORE);

    /* Phase 1b — payload exchange. Only peers with non-zero count. */
    std::vector<MPI_Request> reqs_pay_send;
    std::vector<MPI_Request> reqs_pay_recv;
    reqs_pay_send.reserve(nt - 1);
    reqs_pay_recv.reserve(nt - 1);

    int peers_sent = 0, peers_recv = 0;
    size_t bytes_sent = 0, bytes_recv = 0;

    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        if(state.recv_counts[p] <= 0) continue;
        state.recv_queries[p].resize(state.recv_counts[p]);
        const size_t nbytes = (size_t)state.recv_counts[p] * sizeof(TQuery);
        MPI_Request rq;
        MPI_Irecv(state.recv_queries[p].data(), (int)nbytes, MPI_BYTE,
                  p, TAG_MODE_B_QUERY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_pay_recv.push_back(rq);
        peers_recv++;
        bytes_recv += nbytes;
    }
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        if(state.sent_counts[p] <= 0) continue;
        const size_t nbytes = (size_t)state.sent_counts[p] * sizeof(TQuery);
        MPI_Request rq;
        MPI_Isend(my_queries_to_each_peer[p].data(), (int)nbytes, MPI_BYTE,
                  p, TAG_MODE_B_QUERY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_pay_send.push_back(rq);
        peers_sent++;
        bytes_sent += nbytes;
    }
    MPI_Waitall((int)reqs_pay_recv.size(), reqs_pay_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall((int)reqs_pay_send.size(), reqs_pay_send.data(), MPI_STATUSES_IGNORE);

    g_mode_b_p2p_diag.bytes_query_sent += bytes_sent;
    g_mode_b_p2p_diag.bytes_query_recv += bytes_recv;
    g_mode_b_p2p_diag.peers_sent_to    += peers_sent;
    g_mode_b_p2p_diag.peers_recv_from  += peers_recv;
    return state;
}

template <typename TQuery, typename TReply>
std::vector<std::vector<TReply>>
mode_b_exchange_replies(
    const std::vector<std::vector<TReply>>& my_replies_to_each_peer,
    const mode_b_query_exchange_state_t<TQuery>& state)
{
    static_assert(std::is_trivially_copyable<TReply>::value,
                  "TReply must be trivially copyable for byte-level MPI transfer");

    const int nt = NTask;
    std::vector<std::vector<TReply>> recv_replies(nt);
    if(nt <= 1) return recv_replies;

    /* Phase 2a — reply count exchange (safety check; peer's reply count
     * MUST equal state.sent_counts[p] from phase 1). Buffer counts in a
     * vector so they outlive the MPI_Isend call. */
    std::vector<int> peer_will_send(nt, 0);                    /* incoming */
    std::vector<int> my_send_back_count(nt, 0);                /* outgoing, stable storage */
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        my_send_back_count[p] = (int)my_replies_to_each_peer[p].size();
    }

    std::vector<MPI_Request> reqs_cnt_recv, reqs_cnt_send;
    reqs_cnt_recv.reserve(nt - 1);
    reqs_cnt_send.reserve(nt - 1);
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        MPI_Request rq;
        MPI_Irecv(&peer_will_send[p], 1, MPI_INT, p, TAG_MODE_B_REPLY_COUNT,
                  MPI_COMM_WORLD, &rq);
        reqs_cnt_recv.push_back(rq);
    }
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        MPI_Request rq;
        MPI_Isend(&my_send_back_count[p], 1, MPI_INT, p, TAG_MODE_B_REPLY_COUNT,
                  MPI_COMM_WORLD, &rq);
        reqs_cnt_send.push_back(rq);
    }
    MPI_Waitall((int)reqs_cnt_recv.size(), reqs_cnt_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall((int)reqs_cnt_send.size(), reqs_cnt_send.data(), MPI_STATUSES_IGNORE);

    /* Sanity: peer p said it'll send peer_will_send[p] replies; we expected
     * state.sent_counts[p]. If they disagree, the protocol is broken. */
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        if(peer_will_send[p] != state.sent_counts[p]) {
            fprintf(stderr,
                    "[mode_b ABORT rank=%d] reply-count mismatch from peer %d: "
                    "peer says %d, state says %d. Protocol broken.\n",
                    ThisTask, p, peer_will_send[p], state.sent_counts[p]);
            fflush(stderr);
            gizmo_emergency_hold_reviewed(1, "mode_b_p2p_transport: reply-count protocol mismatch (mid-protocol)", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    /* Phase 2b — reply payload exchange. */
    std::vector<MPI_Request> reqs_pay_send, reqs_pay_recv;
    reqs_pay_send.reserve(nt - 1);
    reqs_pay_recv.reserve(nt - 1);

    size_t bytes_sent = 0, bytes_recv = 0;
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        if(state.sent_counts[p] <= 0) continue;
        recv_replies[p].resize(state.sent_counts[p]);
        const size_t nbytes = (size_t)state.sent_counts[p] * sizeof(TReply);
        MPI_Request rq;
        MPI_Irecv(recv_replies[p].data(), (int)nbytes, MPI_BYTE,
                  p, TAG_MODE_B_REPLY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_pay_recv.push_back(rq);
        bytes_recv += nbytes;
    }
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        if(state.recv_counts[p] <= 0) continue;
        const size_t nbytes = (size_t)state.recv_counts[p] * sizeof(TReply);
        MPI_Request rq;
        MPI_Isend(my_replies_to_each_peer[p].data(), (int)nbytes, MPI_BYTE,
                  p, TAG_MODE_B_REPLY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_pay_send.push_back(rq);
        bytes_sent += nbytes;
    }
    MPI_Waitall((int)reqs_pay_recv.size(), reqs_pay_recv.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall((int)reqs_pay_send.size(), reqs_pay_send.data(), MPI_STATUSES_IGNORE);

    g_mode_b_p2p_diag.bytes_reply_sent += bytes_sent;
    g_mode_b_p2p_diag.bytes_reply_recv += bytes_recv;
    return recv_replies;
}

#endif /* MODE_B_P2P_TRANSPORT_H */
