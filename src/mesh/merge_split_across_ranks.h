/* MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS (included by merge_split.cc): the nearest neighbour that sets how far apart a split's
   daughters are placed is found over the cells of all ranks, so the daughters' positions do not depend on the number of ranks.
   Every cell that is to be split is a query, gathered on all ranks; each rank builds one neighbour list with the queries as
   sources (gpu_ngb_list_build with source positions, over its own particles) and answers with the smallest distance; the owner
   takes the minimum over ranks. The list build is the existing one; the loops here run on the host. */
#include <vector>

namespace msx {

struct SplitQuery {double pos[3], h; int type, task, index;};
struct SplitAnswer {int q; double r2min;};

/* send out[t] to rank t; returns what arrived, by source rank (this rank's own entries at its place) */
template<class T> static std::vector<T> exchange(std::vector<std::vector<T>> &out)
{
    std::vector<int> ns(NTask), nr(NTask);
    for(int t = 0; t < NTask; t++) {
        if(out[t].size() * sizeof(T) > (size_t) 2000000000) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: message to rank %d too large (task %d)\n", t, ThisTask); endrun(97561);}
        ns[t] = (int)(out[t].size() * sizeof(T));
    }
    MPI_Alltoall(ns.data(), 1, MPI_INT, nr.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<MPI_Request> req;
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask || ns[t] == 0) {continue;}
        req.emplace_back(); MPI_Isend(out[t].data(), ns[t], MPI_BYTE, t, 9761, MPI_COMM_WORLD, &req.back());
    }
    std::vector<T> in;
    for(int t = 0; t < NTask; t++) {
        if(t == ThisTask) {in.insert(in.end(), out[t].begin(), out[t].end()); continue;}
        if(nr[t] == 0) {continue;}
        size_t n0 = in.size(); in.resize(n0 + nr[t] / sizeof(T));
        MPI_Recv(in.data() + n0, nr[t], MPI_BYTE, t, 9761, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if(!req.empty()) {MPI_Waitall((int) req.size(), req.data(), MPI_STATUSES_IGNORE);}
    return in;
}

/* all ranks' entries on every rank, in rank order; first[t] = index of rank t's first entry */
template<class T> static std::vector<T> gather_all(const std::vector<T> &loc, std::vector<int> &first)
{
    std::vector<int> nb(NTask), disp(NTask);
    if(loc.size() * sizeof(T) > (size_t) 2000000000) {endrun(97562);}
    int n = (int)(loc.size() * sizeof(T));
    MPI_Allgather(&n, 1, MPI_INT, nb.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long tot = 0; first.assign(NTask + 1, 0);
    for(int t = 0; t < NTask; t++) {disp[t] = (int) tot; first[t] = (int)(tot / sizeof(T)); tot += nb[t]; if(tot > 2000000000LL) {endrun(97563);}}
    first[NTask] = (int)(tot / sizeof(T));
    std::vector<T> all(tot / sizeof(T));
    MPI_Allgatherv(loc.data(), n, MPI_BYTE, all.data(), nb.data(), disp.data(), MPI_BYTE, MPI_COMM_WORLD);
    return all;
}

/* distance from each of this rank's cells in 'cells' to its nearest neighbour of the same type inside its kernel radius, over the
   cells of all ranks; 0 if there is none (or one at the same position). Collective: called on every rank, also with no cell. */
static std::vector<double> split_neighbour_distances(const std::vector<int> &cells)
{
    std::vector<SplitQuery> ql(cells.size());
    for(size_t l = 0; l < cells.size(); l++)
    {
        int i = cells[l]; SplitQuery &q = ql[l]; memset(&q, 0, sizeof(SplitQuery));
        for(int k = 0; k < 3; k++) {q.pos[k] = P[i].Pos[k];}
        q.h = P[i].KernelRadius; q.type = P[i].Type; q.task = ThisTask; q.index = i;
    }
    std::vector<int> qfirst; std::vector<SplitQuery> qall = gather_all(ql, qfirst);
    std::vector<double> r(cells.size(), 0.0);
    const int nq = (int) qall.size();
    if(nq == 0) {return r;}
    /* one list for all queries, over this rank's particles */
    int local_count = ghost_get_num_local(); if(local_count <= 0) {local_count = NumPart;}
    int num_all = local_count + ghost_get_num_ghosts(); if(num_all <= 0) {num_all = NumPart;}
    std::vector<int> source_id(nq); std::vector<double> radius(nq), position(3 * (size_t) nq); int type_mask = 0;
    for(int g = 0; g < nq; g++) {source_id[g] = g; radius[g] = qall[g].h; type_mask |= (1 << qall[g].type); for(int k = 0; k < 3; k++) {position[3 * (size_t) g + k] = qall[g].pos[k];}}
    gpu_particles_arena_acquire(num_all, P, CellP);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    gpu_neighbor_list_t gnl = {};
    gpu_ngb_list_build(P_gpu, num_all, source_id.data(), nq, NGB_SEARCH_ONEWAY, type_mask, &gnl, NULL, 1.0, radius.data(), position.data(), "msplit_ranks");
    std::vector<int> neighbours;
    if(gnl.total_pairs > 0) {neighbours.resize((size_t) gnl.total_pairs); gpu_ngb_copy_neighbors_to_host(&gnl, neighbours.data());}
    std::vector<std::vector<SplitAnswer>> aout(NTask);
    for(int g = 0; g < nq; g++)
    {
        const SplitQuery &q = qall[g]; double r2min = MAX_REAL_NUMBER;
        for(int64_t nn = gnl.offsets[g]; nn < gnl.offsets[g + 1]; nn++)
        {
            int j = neighbours[nn];
            if((j < 0) || (j >= local_count) || (q.task == ThisTask && j == q.index) || (P[j].Type != q.type) || !(P[j].Mass > 0)) {continue;}
            Vec3<double> dp = {q.pos[0] - P[j].Pos[0], q.pos[1] - P[j].Pos[1], q.pos[2] - P[j].Pos[2]};
            nearest_xyz(dp);
            double r2 = dp.norm_sq(); if(r2 >= q.h * q.h) {continue;}
            if(r2 < r2min) {r2min = r2;}
        }
        if(r2min < MAX_REAL_NUMBER) {SplitAnswer a; memset(&a, 0, sizeof(SplitAnswer)); a.q = g; a.r2min = r2min; aout[q.task].push_back(a);}
    }
    gpu_ngb_list_free(&gnl);
    gpu_particles_arena_invalidate();
    std::vector<SplitAnswer> ain = exchange(aout);
    std::vector<double> best(cells.size(), MAX_REAL_NUMBER);
    for(const SplitAnswer &a : ain) {int l = a.q - qfirst[ThisTask]; if(l < 0 || l >= (int) cells.size()) {endrun(97565);} if(a.r2min < best[l]) {best[l] = a.r2min;}}
    for(size_t l = 0; l < cells.size(); l++) {if(best[l] < MAX_REAL_NUMBER && best[l] > 0) {r[l] = sqrt(best[l]);}}
    return r;
}

} /* namespace msx */
