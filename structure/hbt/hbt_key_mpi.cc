/* Distributed key-uniqueness count: route each key to a hash-owner task, sort there, count equal neighbours. */
#include <stdint.h>
#include <limits.h>
#include <vector>
#include <algorithm>
#include "hbt_key_mpi.h"

static inline uint64_t hbt_key_hash(hbt_key_t k)
{
    uint64_t z = k.id * 0x9e3779b97f4a7c15ULL ^ (k.child + 0x632be59bd9b4e019ULL + (k.id << 6) + (k.id >> 2));
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

long long hbt_count_duplicate_keys(const hbt_key_t *keys, long n, MPI_Comm comm)
{
    int ntask, task; MPI_Comm_size(comm, &ntask); MPI_Comm_rank(comm, &task);
    std::vector<long long> cnt(ntask, 0);
    std::vector<int> dest(n > 0 ? n : 1);
    for(long i = 0; i < n; i++) {dest[i] = (int)(hbt_key_hash(keys[i]) % (uint64_t)ntask); cnt[dest[i]]++;}
    int overflow = 0;
    std::vector<int> send_count(ntask), recv_count(ntask), send_disp(ntask, 0), recv_disp(ntask, 0);
    for(int t = 0; t < ntask; t++) {if(cnt[t] > INT_MAX) {overflow = 1;} send_count[t] = (int)cnt[t];}
    MPI_Allreduce(MPI_IN_PLACE, &overflow, 1, MPI_INT, MPI_MAX, comm);
    if(overflow) {return -1;}
    MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, comm);
    long long nsend = 0, nrecv = 0;
    for(int t = 0; t < ntask; t++) {nsend += send_count[t]; nrecv += recv_count[t];}
    overflow = (nsend > INT_MAX || nrecv > INT_MAX);
    MPI_Allreduce(MPI_IN_PLACE, &overflow, 1, MPI_INT, MPI_MAX, comm);
    if(overflow) {return -1;}
    for(int t = 1; t < ntask; t++) {send_disp[t] = send_disp[t - 1] + send_count[t - 1]; recv_disp[t] = recv_disp[t - 1] + recv_count[t - 1];}
    std::vector<hbt_key_t> sendbuf(nsend > 0 ? nsend : 1), recvbuf(nrecv > 0 ? nrecv : 1);
    std::vector<int> fill(send_disp);
    for(long i = 0; i < n; i++) {sendbuf[fill[dest[i]]++] = keys[i];}
    MPI_Datatype key_type; MPI_Type_contiguous(2, MPI_UINT64_T, &key_type); MPI_Type_commit(&key_type);
    MPI_Alltoallv(sendbuf.data(), send_count.data(), send_disp.data(), key_type,
                  recvbuf.data(), recv_count.data(), recv_disp.data(), key_type, comm);
    MPI_Type_free(&key_type);
    std::sort(recvbuf.begin(), recvbuf.begin() + nrecv, [](const hbt_key_t &a, const hbt_key_t &b) {return hbt_key_cmp(a, b) < 0;});
    long long ndup = hbt_count_adjacent_duplicates(recvbuf.data(), (long)nrecv);
    MPI_Allreduce(MPI_IN_PLACE, &ndup, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return ndup;
}
