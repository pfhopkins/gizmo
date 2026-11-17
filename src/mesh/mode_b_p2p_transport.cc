/* mesh/mode_b_p2p_transport.cc
 *
 * The query/reply round protocol (BoundedPeerExchangeCore). Payload storage is
 * always the caller's; this file only counts, posts and waits.
 */

#include "mode_b_p2p_transport.h"
#include <climits>
#include <cstdio>
#include <cstring>
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
    /* Per round: at most one query send, reply receive, group receive and reply send per peer;
     * a group holds at least one peer, so there are at most ntask groups. */
    const size_t n = (size_t) (ntask > 0 ? ntask : 1);
    try {
        sent_counts.reserve(n); recv_counts.reserve(n);
        reqs_query_send.reserve(n); reqs_reply_recv.reserve(n);
        reply_recv_peer.reserve(n); reply_recv_status.reserve(n); reqs_group_query.reserve(n);
        reqs_reply_send.reserve(n); reply_send_group_start.reserve(n);
    } catch(const std::bad_alloc &) {return 1;}
    return 0;
}

void BoundedPeerExchangeCore::release_storage()
{
    std::vector<int>().swap(sent_counts); std::vector<int>().swap(recv_counts);
    std::vector<MPI_Request>().swap(reqs_query_send);
    std::vector<MPI_Request>().swap(reqs_reply_recv); std::vector<int>().swap(reply_recv_peer);
    std::vector<MPI_Status>().swap(reply_recv_status); std::vector<MPI_Request>().swap(reqs_group_query);
    std::vector<MPI_Request>().swap(reqs_reply_send); std::vector<int>().swap(reply_send_group_start);
}

void BoundedPeerExchangeCore::exchange_counts(const int *counts_to_peer, double *dt_count_exch)
{
    const int nt = NTask;
    sent_counts.assign(nt, 0);
    recv_counts.assign(nt, 0);
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
    /* one collective, rather than a message to and from every rank; this rank's own entries are zero both ways */
    const double t0 = dt_count_exch ? MPI_Wtime() : 0.0;
    MPI_Alltoall(sent_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
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

/* ---------------------------------------------------------------------------------------------------------- */
/* ModeDSegmentStream (see the header). */

static void mode_d_stream_broken(const char *what, int peer, long long a, long long b)
{
    fprintf(stderr, "[Mode D exchange ABORT rank=%d] %s (peer %d: %lld vs %lld). Protocol broken.\n", ThisTask, what, peer, a, b);
    fflush(stderr);
    gizmo_fatal_hard_exit_reviewed(90002004, "REVIEWED_HARD_MID_PROTOCOL: Mode D segment stream", __FILE__, __LINE__, __FUNCTION__);
}

int ModeDSegmentStream::reserve(int ntask)
{
    const size_t n = (size_t) (ntask > 0 ? ntask : 1);
    if(core.reserve(ntask) != 0) {return 1;}
    try {
        next_query_from_.reserve(n); next_reply_from_.reserve(n);
        next_query_to_.reserve(n); next_reply_to_.reserve(n);
    } catch(const std::bad_alloc &) {return 1;}
    return 0;
}

void ModeDSegmentStream::release_storage()
{
    core.release_storage();
    std::vector<int>().swap(next_query_from_); std::vector<int>().swap(next_reply_from_);
    std::vector<int>().swap(next_query_to_); std::vector<int>().swap(next_reply_to_);
}

void ModeDSegmentStream::attach(char *const base[ROLES], size_t slot_bytes)
{
    for(int r = 0; r < ROLES; r++) {base_[r] = base[r];}
    slot_bytes_ = slot_bytes;
}

int ModeDSegmentStream::records_in_segment(int count, int index) const
{
    const long long left = (long long) count - (long long) index * segment_records_;
    return (int) ((left < segment_records_) ? left : segment_records_);
}
int ModeDSegmentStream::segments_to(int p) const
{
    const int c = core.sent_counts[p];
    return (c > 0) ? (c + segment_records_ - 1) / segment_records_ : 0;
}
int ModeDSegmentStream::segments_from(int p) const
{
    const int c = core.recv_counts[p];
    return (c > 0) ? (c + segment_records_ - 1) / segment_records_ : 0;
}

int ModeDSegmentStream::begin_round(int round, const int *counts_to_peer, int segment_records, size_t query_bytes, size_t reply_bytes)
{
    for(int r = 0; r < ROLES; r++) {
        for(int k = 0; k < SLOTS; k++) {if(slots_[r][k].state != 0) {mode_d_stream_broken("a round opened with a slot still in use", ThisTask, r, k);}}
        expected_[r] = posted_[r] = taken_[r] = 0;
    }
    round_ = round; segment_records_ = segment_records;
    record_bytes_[QUERIES_OUT] = record_bytes_[QUERIES_IN] = query_bytes;
    record_bytes_[REPLIES_OUT] = record_bytes_[REPLIES_IN] = reply_bytes;
    core.exchange_counts(counts_to_peer);
    const int nt = NTask;
    next_query_from_.assign(nt, 0); next_reply_from_.assign(nt, 0);
    next_query_to_.assign(nt, 0); next_reply_to_.assign(nt, 0);
    int anomaly = 0;
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) {continue;}
        if(core.sent_counts[p] < 0 || core.recv_counts[p] < 0) {anomaly = 1;}
    }
    if(anomaly) {return 1;}
    for(int p = 0; p < nt; p++) {
        if(p == ThisTask) {continue;}
        expected_[QUERIES_OUT] += segments_to(p);   expected_[REPLIES_IN]  += segments_to(p);
        expected_[QUERIES_IN]  += segments_from(p); expected_[REPLIES_OUT] += segments_from(p);
    }
    post_receives(QUERIES_IN, expected_[QUERIES_IN]);
    post_receives(REPLIES_IN, expected_[REPLIES_IN]);
    return 0;
}

void ModeDSegmentStream::post_receives(int role, int expected_total)
{
    const int tag = (role == QUERIES_IN) ? TAG_MODE_D_QUERY_SEGMENT : TAG_MODE_D_REPLY_SEGMENT;
    for(int k = 0; k < SLOTS && posted_[role] < expected_total; k++) {
        Slot &s = slots_[role][k];
        if(s.state != 0) {continue;}
        MPI_Irecv(base_[role] + (size_t) k * slot_bytes_, (int) slot_bytes_, MPI_BYTE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &s.req);
        s.state = 1; posted_[role]++;
    }
}

void ModeDSegmentStream::progress()
{
    for(int r = 0; r < ROLES; r++) {
        for(int k = 0; k < SLOTS; k++) {
            Slot &s = slots_[r][k];
            if(s.state != 1) {continue;}
            int flag = 0; MPI_Status st;
            MPI_Test(&s.req, &flag, &st);
            if(!flag) {continue;}
            if(r == QUERIES_IN || r == REPLIES_IN) {
                s.state = 2; s.source = st.MPI_SOURCE;
                MPI_Get_count(&st, MPI_BYTE, &s.bytes);
            } else {
                s.state = 0;
            }
        }
    }
    post_receives(QUERIES_IN, expected_[QUERIES_IN]);
    post_receives(REPLIES_IN, expected_[REPLIES_IN]);
}

int ModeDSegmentStream::free_slots(int role, int *out)
{
    int n = 0;
    for(int k = 0; k < SLOTS; k++) {if(slots_[role][k].state == 0) {out[n++] = k;}}
    return n;
}

static void mode_d_stream_send(char *slot_base, int round, int index, int records, int status, size_t record_bytes,
                               int dest, int tag, MPI_Request *req)
{
    ModeDSegmentHeader h; h.round = round; h.source = ThisTask; h.index = index; h.records = records; h.status = status;
    memcpy(slot_base, &h, sizeof(h));
    const size_t bytes = sizeof(ModeDSegmentHeader) + (size_t) records * record_bytes;
    MPI_Isend(slot_base, (int) bytes, MPI_BYTE, dest, tag, MPI_COMM_WORLD, req);
}

void ModeDSegmentStream::send_query_segment(int slot, int dest, int index, int records)
{
    Slot &s = slots_[QUERIES_OUT][slot];
    if(s.state != 0 || index != next_query_to_[dest] || records != records_in_segment(core.sent_counts[dest], index)) {
        mode_d_stream_broken("a query segment sent out of order, or of the wrong size", dest, index, next_query_to_[dest]);
    }
    mode_d_stream_send(base_[QUERIES_OUT] + (size_t) slot * slot_bytes_, round_, index, records, 0, record_bytes_[QUERIES_OUT],
                       dest, TAG_MODE_D_QUERY_SEGMENT, &s.req);
    s.state = 1; next_query_to_[dest]++; posted_[QUERIES_OUT]++;
}

void ModeDSegmentStream::send_reply_segment(int slot, int dest, int index, int records, int status)
{
    Slot &s = slots_[REPLIES_OUT][slot];
    if(s.state != 0 || index != next_reply_to_[dest] || records != records_in_segment(core.recv_counts[dest], index)) {
        mode_d_stream_broken("a reply segment sent out of order, or of the wrong size", dest, index, next_reply_to_[dest]);
    }
    mode_d_stream_send(base_[REPLIES_OUT] + (size_t) slot * slot_bytes_, round_, index, records, status, record_bytes_[REPLIES_OUT],
                       dest, TAG_MODE_D_REPLY_SEGMENT, &s.req);
    s.state = 1; next_reply_to_[dest]++; posted_[REPLIES_OUT]++;
}

/* Hand over the next segment of some sender, in that sender's order.  A completed segment is checked against what
 * its sender must send next: a length that is no header plus whole records is a broken protocol; a header that
 * disagrees is handed over as that sender's next segment, marked erroneous, so it is still answered once and the
 * exchange still drains; a segment ahead of its sender's next is held. */
bool ModeDSegmentStream::next_ready(int role, Segment &out)
{
    std::vector<int> &next_from = (role == QUERIES_IN) ? next_query_from_ : next_reply_from_;
    for(int k = 0; k < SLOTS; k++) {
        Slot &s = slots_[role][k];
        if(s.state != 2 || s.source < 0) {continue;}
        const long long body = (long long) s.bytes - (long long) sizeof(ModeDSegmentHeader);
        if(body <= 0 || body % (long long) record_bytes_[role] != 0 || body / (long long) record_bytes_[role] > segment_records_) {
            mode_d_stream_broken("a segment's length is not a header plus whole records", s.source, s.bytes, segment_records_);
        }
        const int src = s.source;
        const int count = (role == QUERIES_IN) ? core.recv_counts[src] : core.sent_counts[src];
        const int n_seg = (count > 0) ? (count + segment_records_ - 1) / segment_records_ : 0;
        const int cursor = next_from[src];
        if(cursor >= n_seg) {mode_d_stream_broken("a sender sent more segments than it counted", src, cursor, n_seg);}
        ModeDSegmentHeader h; memcpy(&h, base_[role] + (size_t) k * slot_bytes_, sizeof(h));
        const int got = (int) (body / (long long) record_bytes_[role]);
        const int want = records_in_segment(count, cursor);
        const int legal_status = (role == QUERIES_IN) ? (h.status == 0) : (h.status == 0 || h.status == 1);
        const int consistent = (h.source == src && h.round == round_ && h.records == got && legal_status);
        if(consistent && h.index > cursor && h.index < n_seg) {continue;}   /* ahead of its predecessor: held */
        out.slot = k; out.source = src; out.index = cursor; out.records = want;
        out.status = (consistent && h.index == cursor && got == want) ? h.status : 1;
        if(out.status) {errors_++;}
        next_from[src]++; taken_[role]++;
        s.source = -1;   /* handed over; the slot stays the caller's until release() */
        return true;
    }
    return false;
}

void ModeDSegmentStream::release(int role, int slot)
{
    Slot &s = slots_[role][slot];
    s.state = 0; s.source = -1; s.bytes = 0;
}

bool ModeDSegmentStream::round_complete() const
{
    if(posted_[QUERIES_OUT] != expected_[QUERIES_OUT] || posted_[REPLIES_OUT] != expected_[REPLIES_OUT]) {return false;}
    if(taken_[QUERIES_IN] != expected_[QUERIES_IN] || taken_[REPLIES_IN] != expected_[REPLIES_IN]) {return false;}
    for(int r = 0; r < ROLES; r++) {for(int k = 0; k < SLOTS; k++) {if(slots_[r][k].state != 0) {return false;}}}
    return true;
}
