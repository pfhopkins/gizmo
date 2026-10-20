/* mesh/mode_b_p2p_transport.cc
 *
 * The query/reply round protocol (BoundedPeerExchangeCore). Payload storage is
 * always the caller's; this file only counts, posts and waits.
 */

#include "mode_b_p2p_transport.h"
#include <climits>
#include <cstdio>
#include <new>

#include "../system/tags.h"

void gizmo_fatal_hard_exit_reviewed(int code, const char *msg,
                                    const char *file, int line, const char *func);

/* MPI counts are int. A payload that does not fit is a protocol failure, never a
 * truncation: every caller bounds its round by the communication chunk, far below this. */
static int payload_bytes_as_int(long long count, size_t elem_bytes, int peer)
{
    const long long nbytes = count * (long long) elem_bytes;
    if(count < 0 || nbytes > (long long) INT_MAX) {
        fprintf(stderr, "[neighbour exchange ABORT rank=%d] %lld payloads of %zu bytes for peer %d "
                "exceed one MPI message. Protocol broken.\n", ThisTask, count, elem_bytes, peer);
        fflush(stderr);
        gizmo_fatal_hard_exit_reviewed(90002003,
            "REVIEWED_HARD_MID_PROTOCOL: neighbour exchange payload exceeds one MPI message",
            __FILE__, __LINE__, __FUNCTION__);
    }
    return (int) nbytes;
}

void bounded_peer_exchange_reply_count_mismatch(int peer, size_t have, int want)
{
    fprintf(stderr, "[mode_b ABORT rank=%d] send_group_replies: %zu replies "
            "for peer %d, expected %d. Caller bug.\n", ThisTask, have, peer, want);
    fflush(stderr);
    gizmo_fatal_hard_exit_reviewed(90002003,
        "REVIEWED_HARD_MID_PROTOCOL: mode_b transport group reply-count mismatch",
        __FILE__, __LINE__, __FUNCTION__);
}

int BoundedPeerExchangeCore::reserve(int ntask)
{
    /* Per round: one count each way and at most one query send, reply receive, group receive
     * and reply send per peer; a group holds at least one peer, so there are at most ntask groups. */
    const size_t n = (size_t) (ntask > 0 ? ntask : 1);
    try {
        sent_counts.reserve(n); recv_counts.reserve(n);
        reqs_count.reserve(2 * n); reqs_query_send.reserve(n); reqs_reply_recv.reserve(n);
        reply_recv_peer.reserve(n); reply_recv_status.reserve(n); reqs_group_query.reserve(n);
        reqs_reply_send.reserve(n); reply_send_group_start.reserve(n);
    } catch(const std::bad_alloc &) {return 1;}
    return 0;
}

void BoundedPeerExchangeCore::release_storage()
{
    std::vector<int>().swap(sent_counts); std::vector<int>().swap(recv_counts);
    std::vector<MPI_Request>().swap(reqs_count); std::vector<MPI_Request>().swap(reqs_query_send);
    std::vector<MPI_Request>().swap(reqs_reply_recv); std::vector<int>().swap(reply_recv_peer);
    std::vector<MPI_Status>().swap(reply_recv_status); std::vector<MPI_Request>().swap(reqs_group_query);
    std::vector<MPI_Request>().swap(reqs_reply_send); std::vector<int>().swap(reply_send_group_start);
}

void BoundedPeerExchangeCore::exchange_counts(const int *counts_to_peer, double *dt_count_exch)
{
    const int nt = NTask;
    sent_counts.assign(nt, 0);
    recv_counts.assign(nt, 0);
    reqs_count.clear();
    reqs_query_send.clear();
    reqs_reply_recv.clear();
    reply_recv_peer.clear();
    reqs_reply_send.clear();
    reply_send_group_start.clear();
    next_peer = 0;
    if(nt <= 1) {return;}

    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        sent_counts[p] = counts_to_peer[p];
    }
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        MPI_Request rq;
        MPI_Irecv(&recv_counts[p], 1, MPI_INT, p, TAG_MODE_B_QUERY_COUNT, MPI_COMM_WORLD, &rq);
        reqs_count.push_back(rq);
    }
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) continue;
        MPI_Request rq;
        MPI_Isend(&sent_counts[p], 1, MPI_INT, p, TAG_MODE_B_QUERY_COUNT, MPI_COMM_WORLD, &rq);
        reqs_count.push_back(rq);
    }
    const double t0 = dt_count_exch ? MPI_Wtime() : 0.0;
    MPI_Waitall((int) reqs_count.size(), reqs_count.data(), MPI_STATUSES_IGNORE);
    if(dt_count_exch) {*dt_count_exch += MPI_Wtime() - t0;}
}

void BoundedPeerExchangeCore::post_query_sends(const void *const *query_to_peer, size_t query_bytes)
{
    for(int p = 0; p < NTask; p++) {
        if(p == ThisTask || sent_counts[p] <= 0) continue;
        const int nbytes = payload_bytes_as_int(sent_counts[p], query_bytes, p);
        MPI_Request rq;
        MPI_Isend(query_to_peer[p], nbytes, MPI_BYTE, p, TAG_MODE_B_QUERY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_query_send.push_back(rq);
    }
}

void BoundedPeerExchangeCore::post_reply_recvs(void *const *reply_from_peer, size_t reply_bytes)
{
    for(int p = 0; p < NTask; p++) {
        if(p == ThisTask || sent_counts[p] <= 0) continue;
        const int nbytes = payload_bytes_as_int(sent_counts[p], reply_bytes, p);
        MPI_Request rq;
        MPI_Irecv(reply_from_peer[p], nbytes, MPI_BYTE, p, TAG_MODE_B_REPLY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_reply_recv.push_back(rq);
        reply_recv_peer.push_back(p);
    }
}

bool BoundedPeerExchangeCore::select_group(size_t budget_bytes, size_t query_bytes, size_t reply_bytes,
                                           std::vector<int>& group_peers)
{
    group_peers.clear();
    size_t group_bytes = 0;
    while(next_peer < NTask) {
        const int p = next_peer;
        if(p == ThisTask || recv_counts[p] <= 0) {next_peer++; continue;}
        const size_t peer_bytes = (size_t) recv_counts[p] * (query_bytes + reply_bytes);
        if(!group_peers.empty() && group_bytes + peer_bytes > budget_bytes) break;
        group_peers.push_back(p);
        group_bytes += peer_bytes;
        next_peer++;
    }
    return !group_peers.empty();
}

void BoundedPeerExchangeCore::receive_group(const std::vector<int>& group_peers, void *const *dst,
                                            size_t query_bytes, double *dt_query_recv_wait)
{
    reqs_group_query.clear();
    for(size_t i = 0; i < group_peers.size(); i++) {
        const int p = group_peers[i];
        const int nbytes = payload_bytes_as_int(recv_counts[p], query_bytes, p);
        MPI_Request rq;
        MPI_Irecv(dst[i], nbytes, MPI_BYTE, p, TAG_MODE_B_QUERY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_group_query.push_back(rq);
    }
    const double t0 = dt_query_recv_wait ? MPI_Wtime() : 0.0;
    MPI_Waitall((int) reqs_group_query.size(), reqs_group_query.data(), MPI_STATUSES_IGNORE);
    if(dt_query_recv_wait) {*dt_query_recv_wait += MPI_Wtime() - t0;}
}

int BoundedPeerExchangeCore::send_group_replies(const std::vector<int>& group_peers, const void *const *src,
                                                size_t reply_bytes)
{
    const int ticket = (int) reply_send_group_start.size();
    reply_send_group_start.push_back((int) reqs_reply_send.size());
    for(size_t i = 0; i < group_peers.size(); i++) {
        const int p = group_peers[i];
        if(recv_counts[p] <= 0) continue;
        const int nbytes = payload_bytes_as_int(recv_counts[p], reply_bytes, p);
        MPI_Request rq;
        MPI_Isend(src[i], nbytes, MPI_BYTE, p, TAG_MODE_B_REPLY_PAYLOAD, MPI_COMM_WORLD, &rq);
        reqs_reply_send.push_back(rq);
    }
    return ticket;
}

void BoundedPeerExchangeCore::wait_group_replies(int ticket)
{
    const int first = reply_send_group_start[ticket];
    const int last  = (ticket + 1 < (int) reply_send_group_start.size()) ? reply_send_group_start[ticket + 1]
                                                                         : (int) reqs_reply_send.size();
    /* Completed requests become MPI_REQUEST_NULL, which finish()'s Waitall passes over. */
    MPI_Waitall(last - first, reqs_reply_send.data() + first, MPI_STATUSES_IGNORE);
}

void BoundedPeerExchangeCore::finish(size_t reply_bytes, double *dt_reply_finish_wait)
{
    const double t0 = dt_reply_finish_wait ? MPI_Wtime() : 0.0;
    MPI_Waitall((int) reqs_query_send.size(), reqs_query_send.data(), MPI_STATUSES_IGNORE);
    /* Reply sends still in flight: waiting them here lets their completion overlap the
     * reply-recv wait below instead of blocking per group. */
    MPI_Waitall((int) reqs_reply_send.size(), reqs_reply_send.data(), MPI_STATUSES_IGNORE);
    reply_recv_status.resize(reqs_reply_recv.size());
    MPI_Waitall((int) reqs_reply_recv.size(), reqs_reply_recv.data(), reply_recv_status.data());
    if(dt_reply_finish_wait) {*dt_reply_finish_wait += MPI_Wtime() - t0;}
    for(size_t i = 0; i < reply_recv_status.size(); i++) {
        int got_bytes = 0;
        MPI_Get_count(&reply_recv_status[i], MPI_BYTE, &got_bytes);
        const int p = reply_recv_peer[i];
        const long long want_bytes = (long long) sent_counts[p] * (long long) reply_bytes;
        if((long long) got_bytes != want_bytes) {
            fprintf(stderr, "[mode_b ABORT rank=%d] reply payload from peer %d is "
                    "%d bytes, expected %lld (%d replies x %zu B). Protocol broken.\n",
                    ThisTask, p, got_bytes, want_bytes, sent_counts[p], reply_bytes);
            fflush(stderr);
            gizmo_fatal_hard_exit_reviewed(90002003,
                "REVIEWED_HARD_MID_PROTOCOL: mode_b transport reply-size mismatch",
                __FILE__, __LINE__, __FUNCTION__);
        }
    }
    reqs_query_send.clear();
    reqs_reply_recv.clear();
    reply_recv_peer.clear();
    reqs_reply_send.clear();
    reply_send_group_start.clear();
}
