/* MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS (included by merge_split.cc): the nearest neighbour that sets how far apart a split's
   daughters are placed is found over the cells of all ranks, so the daughters' positions do not depend on the number of ranks.
   Every cell that is to be split is a query, gathered on all ranks; each rank builds one neighbour list with the queries as
   sources (gpu_ngb_list_build with source positions, over its own particles) and answers with the smallest distance; the owner
   takes the minimum over ranks. The list build is the existing one; the loops here run on the host.
   The merge partner is found the same way, and the choice does not depend on index order:
   1. best candidate by (mass, IDs, position); a proposal i -> j is accepted iff i's key (mass, IDs, position) is the smallest among
      all proposals touching i or j; the matching is repeated within the pass for the cells that lost their target (a cell
      that took a merge does nothing more in the pass, as a claimed target of the rank-local loop);
   2. landing points (centres of mass) of the accepted merges are gathered: exact duplicates go to the smaller key, and a landing
      point exactly on another cell of any rank is refused;
   3. the merge runs on the target's rank with merge_particles_ij: the absorbed cell's P and CellP records are sent as bytes and
      placed in the free slot behind the gas block for the call; its owner zeroes it on the target's acknowledgement.
   Star particles take the same path with P only. Merges are done before the splits of the pass. Everything here is host code;
   the only device work is the list build, whose neighbours are copied to the host before they are read. */
#include <vector>
#include <set>
#include <array>
#include <algorithm>
#include <type_traits>

#if defined(SINK_WIND_SPAWN) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES) || defined(SINK_RIAF_SUBEDDINGTON_MODEL) || defined(CHIMES) || defined(GENERAL_RELATIVITY)
#error "MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS does not carry the pair state the spawned-wind, star-cluster, RIAF and GR merge clauses need, nor ChimesGasVars"
#endif
#ifndef MERGE_SPLIT_MATCHING_ROUNDS
#define MERGE_SPLIT_MATCHING_ROUNDS 16 /* cap on the matching rounds of one pass */
#endif

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

/* this rank's particles within rad[a] of the point pos[3a..3a+2], as a CSR list on the host (row a = ngb[off[a]..off[a+1])) */
static void search_at_positions(const std::vector<double> &pos, const std::vector<double> &rad, int type_mask, std::vector<int64_t> &off, std::vector<int> &ngb, int &local_count)
{
    const int n = (int) rad.size(); off.assign((size_t) n + 1, 0); ngb.clear();
    local_count = ghost_get_num_local(); if(local_count <= 0) {local_count = NumPart;}
    if(n == 0) {return;}
    int num_all = local_count + ghost_get_num_ghosts(); if(num_all <= 0) {num_all = NumPart;}
    std::vector<int> rows(n); for(int a = 0; a < n; a++) {rows[a] = a;} /* query numbers, not particle indices: radius and position are given */
    gpu_particles_arena_acquire(num_all, P, CellP);
    gpu_neighbor_list_t gnl = {};
    gpu_ngb_list_build(gpu_particles_arena_P(), num_all, rows.data(), n, NGB_SEARCH_ONEWAY, type_mask, &gnl, NULL, 1.0, rad.data(), pos.data(), "msplit_ranks");
    if(gnl.offsets) {for(int a = 0; a <= n; a++) {off[a] = gnl.offsets[a];}} /* shared memory */
    if(gnl.total_pairs > 0) {ngb.resize((size_t) gnl.total_pairs); gpu_ngb_copy_neighbors_to_host(&gnl, ngb.data());} /* device memory: copied */
    gpu_ngb_list_free(&gnl);
    gpu_particles_arena_invalidate();
}

typedef std::remove_reference<decltype(P[0])>::type PartT;
typedef std::remove_reference<decltype(CellP[0])>::type CellT;

struct Key {double mass; unsigned long long id, child; long long gen; double pos[3];};
struct Query {Key key; double pos[3], h, mass; int type, task, index, composition;
#ifdef HYDRO_MULTIFLUID
    decltype(PartT::FluidType) fluid;
#endif
};
struct Answer {int q, found; Key jkey; int jtask, jindex;};
struct Proposal {Key key; int itask, iindex, jindex;};
struct Verdict {int iindex, ok;};
struct XferHead {Key key; int itask, iindex, jindex, type;};
struct Landing {Key key; double pos[3], h; int type, itask, iindex, jtask, jindex;};

static inline bool key_less(const Key &a, const Key &b)
{
    if(a.mass != b.mass) {return a.mass < b.mass;}
    if(a.id != b.id) {return a.id < b.id;}
    if(a.child != b.child) {return a.child < b.child;}
    if(a.gen != b.gen) {return a.gen < b.gen;}
    for(int k = 0; k < 3; k++) {if(a.pos[k] != b.pos[k]) {return a.pos[k] < b.pos[k];}}
    return false;
}
static inline bool key_equal(const Key &a, const Key &b) {return !key_less(a, b) && !key_less(b, a);}
static inline Key key_of(int i)
{
    Key k; memset(&k, 0, sizeof(Key));
    k.mass = P[i].Mass; k.id = (unsigned long long) P[i].ID; k.child = (unsigned long long) P[i].ID_child_number; k.gen = (long long) P[i].ID_generation;
    for(int d = 0; d < 3; d++) {k.pos[d] = P[i].Pos[d];}
    return k;
}

/* this rank's best merge candidate for one query, from its row of the list, with the conditions of the rank-local loop */
static Answer answer_merge_query(const Query &q, int qindex, const int *ngb, int64_t n0, int64_t n1, int local_count, const std::vector<char> &no_target)
{
    Answer a; memset(&a, 0, sizeof(Answer)); a.q = qindex; a.jtask = ThisTask; a.jindex = -1;
    for(int64_t n = n0; n < n1; n++)
    {
        int j = ngb[n];
        if((j < 0) || (j >= local_count) || (q.task == ThisTask && j == q.index) || (P[j].Type != q.type) || !(P[j].Mass > 0) || no_target[j]) {continue;}
        if(!gx_pair_accept_wrap_and_test(q.pos[0] - P[j].Pos[0], q.pos[1] - P[j].Pos[1], q.pos[2] - P[j].Pos[2], q.h, 0.0, NGB_SEARCH_ONEWAY)) {continue;}
        if(!((P[j].Mass >= q.mass) && (q.mass + P[j].Mass < All.MaxMassForParticleSplit))) {continue;}
#ifdef HYDRO_MULTIFLUID
        if(!same_lagrangian_fluid_id(q.fluid, P[j].FluidType)) {continue;}
#endif
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
        if((q.type == 0) && (q.composition != (int) CellP[j].CompositionType)) {continue;} /* never merge two materials */
#endif
        Key kj = key_of(j);
        if(!a.found || key_less(kj, a.jkey)) {a.found = 1; a.jkey = kj; a.jindex = j;}
    }
    return a;
}

/* the merges of one pass. candidates: this rank's merge candidates of the pass (as the rank-local loop selects them);
   no_target[i] != 0: cell i takes no merge (2: it is split in this pass). Returns the merges done on this rank. Collective. */
static int merge_across_ranks(const std::vector<int> &candidates, std::vector<char> &no_target)
{
    const int type_mask = (1 << 0)
#if defined(GALSF)
                        | (1 << 4)
#endif
                        ;
    std::set<std::array<double,3>> claimed; /* landing points of this pass's merges (primary-box image), the same set on every rank */
    int n_merged_pass = 0;
    for(int round = 0; round < MERGE_SPLIT_MATCHING_ROUNDS; round++)
    {
    /* 1. queries: the candidates still there (absorbed cells have mass 0) that have not taken a merge, proposing again until they merge or the pass ends */
    std::vector<Query> ql;
    for(int i : candidates)
    {
        if(!(P[i].Mass > 0) || no_target[i] != 0) {continue;} /* 2: split in this pass; 1: took a merge in an earlier round */
        Query q; memset(&q, 0, sizeof(Query)); q.key = key_of(i); q.h = P[i].KernelRadius; q.mass = P[i].Mass; q.type = P[i].Type; q.task = ThisTask; q.index = i;
        for(int k = 0; k < 3; k++) {q.pos[k] = P[i].Pos[k];}
#ifdef HYDRO_MULTIFLUID
        q.fluid = P[i].FluidType;
#endif
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
        if(P[i].Type == 0) {q.composition = (int) CellP[i].CompositionType;}
#endif
        ql.push_back(q);
    }
    std::vector<int> qfirst; std::vector<Query> qall = gather_all(ql, qfirst);
    if(qall.empty()) {break;} /* the same on every rank */

    /* 2. every rank answers every query from one list built at the query positions; the owner keeps the best answer */
    std::vector<std::vector<Answer>> aout(NTask);
    {
        std::vector<double> qpos(3 * qall.size()), qrad(qall.size()); std::vector<int64_t> off; std::vector<int> ngb; int local_count = 0;
        for(size_t g = 0; g < qall.size(); g++) {for(int k = 0; k < 3; k++) {qpos[3 * g + k] = qall[g].pos[k];} qrad[g] = qall[g].h;}
        search_at_positions(qpos, qrad, type_mask, off, ngb, local_count);
        for(int g = 0; g < (int) qall.size(); g++) {Answer a = answer_merge_query(qall[g], g, ngb.data(), off[g], off[g + 1], local_count, no_target); if(a.found) {aout[qall[g].task].push_back(a);}}
    }
    std::vector<Answer> ain = exchange(aout);
    std::vector<Answer> best(ql.size()); for(size_t l = 0; l < ql.size(); l++) {memset(&best[l], 0, sizeof(Answer));}
    for(const Answer &a : ain)
    {
        int l = a.q - qfirst[ThisTask]; if(l < 0 || l >= (int) ql.size()) {endrun(97565);}
        if(!best[l].found || key_less(a.jkey, best[l].jkey)) {best[l] = a;}
    }

    /* 3. proposals and the order-free matching: W(x) = smallest proposer key among the proposals touching x */
    std::vector<Key> W(NumPart); std::vector<char> hasW(NumPart, 0);
    std::vector<std::vector<Proposal>> pout(NTask);
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(!best[l].found) {continue;}
        Proposal p; memset(&p, 0, sizeof(Proposal)); p.key = ql[l].key; p.itask = ThisTask; p.iindex = ql[l].index; p.jindex = best[l].jindex;
        pout[best[l].jtask].push_back(p);
        W[p.iindex] = p.key; hasW[p.iindex] = 1;
    }
    std::vector<Proposal> pin = exchange(pout);
    for(const Proposal &p : pin) {if(p.jindex < 0 || p.jindex >= NumPart) {endrun(97566);} if(!hasW[p.jindex] || key_less(p.key, W[p.jindex])) {W[p.jindex] = p.key; hasW[p.jindex] = 1;}}
    std::vector<std::vector<Verdict>> vout(NTask);
    for(const Proposal &p : pin) {Verdict v; v.iindex = p.iindex; v.ok = key_equal(p.key, W[p.jindex]) ? 1 : 0; vout[p.itask].push_back(v);}
    std::vector<Verdict> vin = exchange(vout);
    std::vector<char> ok_at_target(NumPart, 0); for(const Verdict &v : vin) {if(v.ok) {ok_at_target[v.iindex] = 1;}}

    /* 4. the accepted proposers send their records to the target's rank */
    const size_t rec = sizeof(XferHead) + sizeof(PartT) + sizeof(CellT);
    std::vector<std::vector<char>> xout(NTask);
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(!best[l].found) {continue;}
        int i = ql[l].index; if(!ok_at_target[i] || !key_equal(ql[l].key, W[i])) {continue;}
        XferHead hd; memset(&hd, 0, sizeof(XferHead)); hd.key = ql[l].key; hd.itask = ThisTask; hd.iindex = i; hd.jindex = best[l].jindex; hd.type = P[i].Type;
        std::vector<char> &b = xout[best[l].jtask]; size_t n0 = b.size(); b.resize(n0 + rec, 0);
        memcpy(b.data() + n0, &hd, sizeof(XferHead)); memcpy((void *)(b.data() + n0 + sizeof(XferHead)), (const void *) &P[i], sizeof(PartT));
        if(P[i].Type == 0) {memcpy((void *)(b.data() + n0 + sizeof(XferHead) + sizeof(PartT)), (const void *) &CellP[i], sizeof(CellT));}
    }
    std::vector<char> xin = exchange(xout);
    const int nx = (int)(xin.size() / rec);
    auto head = [&](int x) {XferHead hd; memcpy(&hd, xin.data() + (size_t) x * rec, sizeof(XferHead)); return hd;};

    /* the absorbed cell at a local index for merge_particles_ij: its own slot, or a copy in the free slot behind the gas block (gas)
       or the particles (others); host memory only, restored before any device work */
    std::vector<char> saved(sizeof(PartT)); int saved_valid = 0;
    auto place = [&](int x) {
        XferHead hd = head(x); if(hd.itask == ThisTask) {return hd.iindex;}
        const char *src = xin.data() + (size_t) x * rec + sizeof(XferHead); int slot;
        if(hd.type == 0) {
            slot = N_gas; saved_valid = (N_gas < NumPart); if(saved_valid) {memcpy(saved.data(), (const void *) &P[slot], sizeof(PartT));}
            memcpy((void *) &P[slot], src, sizeof(PartT)); memcpy((void *) &CellP[slot], src + sizeof(PartT), sizeof(CellT));
        } else {
            slot = NumPart; saved_valid = 0; memcpy((void *) &P[slot], src, sizeof(PartT));
        }
        return slot;
    };
    auto unplace = [&](int x) {
        XferHead hd = head(x); if(hd.itask == ThisTask) {return;}
        if(hd.type == 0 && saved_valid) {memcpy((void *) &P[N_gas], saved.data(), sizeof(PartT));} /* the particle that lives there */
        else {P[(hd.type == 0) ? N_gas : NumPart].Mass = 0;} /* beyond the live particles: leave nothing that looks like one */
        saved_valid = 0;
    };
    /* room for the copy: CellP[N_gas] for gas, and P[N_gas] exists or P[NumPart] is free; without room the merge waits (this rank only, reported) */
    auto has_room = [&](int x) {XferHead hd = head(x); if(hd.itask == ThisTask) {return true;}
        if(hd.type == 0) {return (N_gas < All.MaxPartGas) && ((N_gas < NumPart) || (NumPart < All.MaxPart));} return NumPart < All.MaxPart;};

    /* 5. landing points: duplicates and points on existing cells, decided alike on every rank */
    std::vector<Landing> ll(nx);
    for(int x = 0; x < nx; x++)
    {
        XferHead hd = head(x);
        Landing &L = ll[x]; memset(&L, 0, sizeof(Landing)); L.key = hd.key; L.h = P[hd.jindex].KernelRadius; L.type = hd.type; L.itask = hd.itask; L.iindex = hd.iindex; L.jtask = ThisTask; L.jindex = hd.jindex;
        if(!has_room(x)) {L.type = -1; printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: no free slot on task %d, a merge across ranks waits\n", ThisTask); continue;}
        int slot = place(x);
        Vec3<MyDouble> pm = merged_position(slot, hd.jindex); box_wrap_position_to_primary_image(pm); /* where merge_particles_ij puts the survivor */
        for(int k = 0; k < 3; k++) {L.pos[k] = pm[k];}
        unplace(x);
    }
    std::vector<int> lfirst; std::vector<Landing> lall = gather_all(ll, lfirst);
    const int nl = (int) lall.size(); std::vector<int> order(nl); for(int g = 0; g < nl; g++) {order[g] = g;}
    std::sort(order.begin(), order.end(), [&](int a, int b) {return key_less(lall[a].key, lall[b].key);});
    std::vector<int> refused(nl, 0), refused_glob(nl, 0);
    for(int o = 0; o < nl; o++)
    {
        const Landing &L = lall[order[o]]; if(L.type < 0) {refused[order[o]] = 1; continue;}
        if(!claimed.insert({L.pos[0], L.pos[1], L.pos[2]}).second) {refused[order[o]] = 1;} /* the same on every rank: the smaller key keeps the point */
    }
    { /* a cell of this rank exactly at a landing point (other than the pair itself) */
        std::vector<double> lpos(3 * (size_t) nl), lrad(nl); std::vector<int64_t> off; std::vector<int> ngb; int local_count = 0;
        for(int g = 0; g < nl; g++) {for(int k = 0; k < 3; k++) {lpos[3 * (size_t) g + k] = lall[g].pos[k];} lrad[g] = 1.e-6 * lall[g].h;}
        search_at_positions(lpos, lrad, type_mask, off, ngb, local_count);
        for(int g = 0; g < nl; g++)
        {
            const Landing &L = lall[g]; if(L.type < 0) {continue;}
            for(int64_t n = off[g]; n < off[g + 1]; n++)
            {
                int k = ngb[n];
                if((k < 0) || (k >= local_count) || !(P[k].Mass > 0) || (L.itask == ThisTask && k == L.iindex) || (L.jtask == ThisTask && k == L.jindex)) {continue;}
                Vec3<double> dk = {P[k].Pos[0] - L.pos[0], P[k].Pos[1] - L.pos[1], P[k].Pos[2] - L.pos[2]}; nearest_xyz(dk, -1);
                if(dk.norm_sq() == 0) {refused[g] = 1; break;}
            }
        }
    }
    if(nl > 0) {MPI_Allreduce(refused.data(), refused_glob.data(), nl, MPI_INT, MPI_MAX, MPI_COMM_WORLD);}

    /* 6. the merges, on the target's rank; the owner of an absorbed cell on another rank is told the outcome */
    int n_merged = 0; std::vector<std::vector<Verdict>> dout(NTask);
    for(int x = 0; x < nx; x++)
    {
        XferHead hd = head(x); int did = 0;
        if(!refused_glob[lfirst[ThisTask] + x])
        {
            int slot = place(x);
            did = (merge_particles_ij(slot, hd.jindex, -1.0) == 1) ? 1 : 0;
            unplace(x);
            if(did) {int jj = hd.jindex; gizmo_mark_position_dirty_indices(&jj, 1); n_merged++; no_target[jj] = 1;} /* the survivor moved outside a drift; it is no target again in this pass */
        }
        if(hd.itask != ThisTask) {Verdict v; v.iindex = hd.iindex; v.ok = did; dout[hd.itask].push_back(v);}
    }
    std::vector<Verdict> din = exchange(dout);
    for(const Verdict &v : din)
    { /* what merge_particles_ij does to the absorbed cell; rearrange_particle_sequence removes it */
        if(v.iindex < 0 || v.iindex >= NumPart) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: acknowledgement for index %d outside [0,%d) on task %d\n", v.iindex, NumPart, ThisTask); endrun(97570);}
        if(!v.ok) {continue;}
        int i = v.iindex;
        Vec3<double> p_old_i = P[i].Vel * P[i].Mass;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        if(P[i].Type == 0) {CellP[i].MassTrue = 0; CellP[i].DtMass = 0; CellP[i].dMass = 0;}
#endif
#if defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
        P[i].Time_Of_Last_MergeSplit = All.Time;
#endif
        P[i].Mass = 0; if(P[i].Type == 0) {CellP[i].Mass = P[i].Mass;}
        P[i].dp += P[i].Vel * P[i].Mass - p_old_i;
    }
    n_merged_pass += n_merged;
    int n_round_glob = 0; MPI_Allreduce(&n_merged, &n_round_glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(n_round_glob == 0) {break;} /* nothing merged anywhere in this round */
    } /* rounds */
    return n_merged_pass;
}

} /* namespace msx */
