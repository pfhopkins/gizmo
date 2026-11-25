/* MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS (included by merge_split.cc): the merge partner and the nearest neighbour that sets a
   split's daughter spacing are found over the cells of all ranks, and the merge choice does not depend on index order, so the
   result of a merge/split pass does not depend on the number of ranks.
   1. every candidate of the pass (an active cell that needs a merge or a split) is a query, gathered on all ranks; each rank
      answers all queries with one neighbour list built at the query positions over its own cells (the owner like any other rank).
      No ghost copies are used and none are made: the queries travel, the cells stay where they are;
   2. merge: best candidate by (mass, IDs, position); a proposal i -> j is accepted iff i's key (mass, IDs, position) is the
      smallest among all proposals touching i or j: at most one merge per cell and pass, no chains, losers wait for the next pass;
   3. landing points (centres of mass) of the accepted merges are gathered: exact duplicates go to the smaller key, and a landing
      point exactly on another cell of any rank is refused;
   4. the merge runs on the target's rank with merge_particles_ij: the absorbed cell's P and CellP records are sent as bytes and
      placed in the free slot behind the gas block for the call; its owner zeroes it on the target's acknowledgement.
   Star particles (non-gas) take the same path with P only. Merges are done before the splits of the pass.
   Work and memory: every rank holds and answers the queries of all ranks, O(candidates of the pass over all ranks); a pass
   without candidates costs one gather of the counts. */
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

namespace msx {

typedef std::remove_reference<decltype(P[0])>::type PartT;
typedef std::remove_reference<decltype(CellP[0])>::type CellT;

struct Key {double mass; unsigned long long id, child; long long gen; double pos[3];};
struct Query {Key key; double pos[3], h, mass, vel[3], e_light; int type, kind, task, index; /* kind 1 merge, 2 split */
#ifdef HYDRO_MULTIFLUID
    decltype(PartT::FluidType) fluid;
#endif
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    int composition;
#endif
};
struct Answer {int q, found; double r2min; Key jkey; int jtask, jindex;};
struct Proposal {Key key; int itask, iindex, jindex;};
struct Verdict {int iindex, ok;};
struct XferHead {Key key; int itask, iindex, jindex, type;};
struct Landing {Key key; double pos[3], h; int type, itask, iindex, jtask, jindex;};
struct Split {int index; double r_nearest;};

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

/* send out[t] to rank t; returns what arrived, by source rank (this rank's own entries at its place) */
template<class T> static std::vector<T> exchange(std::vector<std::vector<T>> &out)
{
    std::vector<int> ns(NTask), nr(NTask);
    for(int t = 0; t < NTask; t++) {
        if(out[t].size() * sizeof(T) > (size_t) 2000000000) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: message to rank %d too large (task %d)\n", t, ThisTask); endrun(97561); out[t].clear();}
        ns[t] = (int)(out[t].size() * sizeof(T));
    }
    MPI_Alltoall(ns.data(), 1, MPI_INT, nr.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<MPI_Request> req; req.reserve(NTask);
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

/* all ranks' entries on every rank, in rank order; first[t] = index of rank t's first entry. Nothing but the counts travels when no rank has an entry. */
template<class T> static std::vector<T> gather_all(std::vector<T> loc, std::vector<int> &first)
{
    std::vector<int> nb(NTask), disp(NTask);
    if(loc.size() * sizeof(T) > (size_t) 2000000000) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: too many entries on task %d\n", ThisTask); endrun(97562); loc.clear();}
    int n = (int)(loc.size() * sizeof(T));
    MPI_Allgather(&n, 1, MPI_INT, nb.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long tot = 0; first.assign(NTask + 1, 0);
    for(int t = 0; t < NTask; t++) {tot += nb[t];}
    if(tot == 0) {return std::vector<T>();}
    if(tot > 2000000000LL) {if(ThisTask == 0) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: too many entries over all tasks\n");} endrun(97563); return std::vector<T>();} /* the same on every task */
    tot = 0; for(int t = 0; t < NTask; t++) {disp[t] = (int) tot; first[t] = (int)(tot / sizeof(T)); tot += nb[t];}
    first[NTask] = (int)(tot / sizeof(T));
    std::vector<T> all(tot / sizeof(T));
    MPI_Allgatherv(loc.data(), n, MPI_BYTE, all.data(), nb.data(), disp.data(), MPI_BYTE, MPI_COMM_WORLD);
    return all;
}

/* this rank's cells within rad[a] of the point pos[3a..3a+2], as a CSR list (rows off[a]..off[a+1] of ngb): the search of the
   rank-local code, at given positions instead of at particles */
static void search_at_positions(const std::vector<double> &pos, const std::vector<double> &rad, int type_mask, std::vector<int64_t> &off, std::vector<int> &ngb, int &local_count)
{
    const int n = (int) rad.size(); off.assign((size_t) n + 1, 0); ngb.clear();
    local_count = ghost_get_num_local(); if(local_count <= 0) {local_count = NumPart;}
    if(n == 0) {return;}
    int num_all = local_count + ghost_get_num_ghosts(); if(num_all <= 0) {num_all = NumPart;}
    std::vector<int> rows(n); for(int a = 0; a < n; a++) {rows[a] = a;}
    gpu_particles_arena_acquire(num_all, P, CellP);
    gpu_neighbor_list_t gnl = {};
    gpu_ngb_list_build(gpu_particles_arena_P(), num_all, rows.data(), n, NGB_SEARCH_ONEWAY, type_mask, &gnl, NULL, 1.0, rad.data(), pos.data(), "msplit_ranks");
    if(gnl.offsets && gnl.total_pairs > 0)
    {
        for(int a = 0; a <= n; a++) {off[a] = gnl.offsets[a];}
        ngb.resize((size_t) gnl.total_pairs); gpu_ngb_copy_neighbors_to_host(&gnl, ngb.data());
    }
    gpu_ngb_list_free(&gnl);
    gpu_particles_arena_invalidate();
}

/* this rank's answer to one query from its row of the list: nearest cell (split) or best merge candidate among this rank's cells */
static Answer answer_query(const Query &q, int qindex, const int *ngb, int64_t n0, int64_t n1, int local_count, const std::unordered_set<int> &is_split_query)
{
    Answer a; memset(&a, 0, sizeof(Answer)); a.q = qindex; a.r2min = MAX_REAL_NUMBER; a.jtask = ThisTask; a.jindex = -1;
    const double h2 = q.h * q.h;
    for(int64_t n = n0; n < n1; n++)
    {
        int j = ngb[n];
        if((j < 0) || (j >= local_count) || (q.task == ThisTask && j == q.index) || (P[j].Type != q.type) || !(P[j].Mass > 0)) {continue;}
        Vec3<double> dp = {q.pos[0] - P[j].Pos[0], q.pos[1] - P[j].Pos[1], q.pos[2] - P[j].Pos[2]};
        if(q.kind == 2)
        {
            nearest_xyz(dp);
            double r2 = dp.norm_sq(); if(r2 >= h2) {continue;} /* inside the kernel only, whatever else a list row may hold */
            if(r2 < a.r2min) {a.r2min = r2; a.found = 1;}
            continue;
        }
        nearest_xyz(dp, -1);
        if(dp.norm_sq() >= h2) {continue;}
        if(!((P[j].Mass >= q.mass) && (q.mass + P[j].Mass < All.MaxMassForParticleSplit))) {continue;}
        if(is_split_query.count(j)) {continue;} /* a cell that is split in this pass is no target */
#ifdef MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION_ALL_GAS
        if((q.type == 0) && (P[j].Type == 0))
        { /* merge_kinetic_dissipation_fraction with the proposer's values: it is the lighter cell (mass_j >= mass_i) */
            double dv2 = 0; for(int k = 0; k < 3; k++) {double d = q.vel[k] - P[j].Vel[k]; dv2 += d * d;}
            double ke_com = 0.5 * (q.mass * P[j].Mass / (q.mass + P[j].Mass)) * dv2 * All.cf_a2inv;
            if(ke_com > 0) {double frac = (q.e_light > 0) ? ke_com / (ke_com + q.e_light) : 1; if(frac > MERGE_SPLIT_MAX_KINETIC_DISSIPATION_FRACTION) {continue;}}
        }
#endif
#ifdef GALSF_SINGLE_STARS_FROM_IMF
        if((P[j].Type == 4) && (P[j].StarIdx > 0)) {continue;} /* an IMF-sampled star is never the target of a merger */
#endif
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

/* the whole decision and the merges of one pass, for the candidates src_idx (src_kind 1 = merge, 2 = split; src_radius = search radius).
   splits: the cells to be split with their neighbour distance, in candidate order. Returns the merges done on this rank. */
static int merge_and_find_splits(const std::vector<int> &src_idx, const std::vector<unsigned char> &src_kind, const std::vector<double> &src_radius, int type_mask, std::vector<Split> &splits)
{
    splits.clear();
    /* 1. queries */
    std::vector<Query> ql; std::unordered_set<int> is_split_query;
    for(size_t aa = 0; aa < src_idx.size(); aa++)
    {
        const int i = src_idx[aa]; int kind = src_kind[aa];
        if(kind == 2) {if((P[i].Type == 0) && !((CellP[i].Density > 0) && isfinite(CellP[i].Density))) {continue;} is_split_query.insert(i);} /* a cell without a valid density has no volume to divide */
        Query q; memset(&q, 0, sizeof(Query)); q.key = key_of(i); q.h = src_radius[aa]; q.mass = P[i].Mass; q.type = P[i].Type; q.kind = kind; q.task = ThisTask; q.index = i;
        for(int k = 0; k < 3; k++) {q.pos[k] = P[i].Pos[k]; q.vel[k] = P[i].Vel[k];}
        if(P[i].Type == 0) {double vA = CellP[i].Alfven_speed(); q.e_light = P[i].Mass * (CellP[i].InternalEnergyPred + 0.5 * vA * vA);}
#ifdef HYDRO_MULTIFLUID
        q.fluid = P[i].FluidType;
#endif
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
        if(P[i].Type == 0) {q.composition = (int) CellP[i].CompositionType;}
#endif
        ql.push_back(q);
    }
    std::vector<int> qfirst; std::vector<Query> qall = gather_all(ql, qfirst);
    if(qall.empty()) {return 0;}

    /* 2. every rank answers every query; answers go to the query's owner */
    std::vector<std::vector<Answer>> aout(NTask);
    {
        std::vector<double> qpos(3 * qall.size()), qrad(qall.size()); std::vector<int64_t> off; std::vector<int> ngb; int local_count = 0;
        for(size_t g = 0; g < qall.size(); g++) {for(int k = 0; k < 3; k++) {qpos[3 * g + k] = qall[g].pos[k];} qrad[g] = qall[g].h;}
        search_at_positions(qpos, qrad, type_mask, off, ngb, local_count);
        for(int g = 0; g < (int) qall.size(); g++) {Answer a = answer_query(qall[g], g, ngb.data(), off[g], off[g + 1], local_count, is_split_query); if(a.found) {aout[qall[g].task].push_back(a);}}
    }
    std::vector<Answer> ain = exchange(aout);
    std::vector<Answer> best(ql.size()); for(size_t l = 0; l < ql.size(); l++) {memset(&best[l], 0, sizeof(Answer)); best[l].r2min = MAX_REAL_NUMBER;}
    for(const Answer &a : ain)
    {
        int l = a.q - qfirst[ThisTask]; if(l < 0 || l >= (int) ql.size()) {endrun(97565); continue;}
        if(ql[l].kind == 2) {if(a.r2min < best[l].r2min) {best[l].r2min = a.r2min; best[l].found = 1; best[l].jtask = a.jtask;}}
        else if(!best[l].found || key_less(a.jkey, best[l].jkey)) {best[l] = a;}
    }
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(!(ql[l].kind == 2 && best[l].found && best[l].r2min < MAX_REAL_NUMBER && best[l].r2min > 0)) {continue;}
        Split s; s.index = ql[l].index; s.r_nearest = sqrt(best[l].r2min); splits.push_back(s);
#ifdef TEST_WAKEUP_DUMP
        printf("TEST_SPLIT_NEIGHBOR step %lld ID %llu task %d neighbour_task %d\n", (long long) All.NumCurrentTiStep, ql[l].key.id, ThisTask, best[l].jtask);
#endif
    }

    /* 3. proposals and the order-free matching: W[cell] = smallest key among the proposals touching the cell */
    std::unordered_map<int, Key> W;
    std::vector<std::vector<Proposal>> pout(NTask);
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(ql[l].kind != 1 || !best[l].found) {continue;}
        Proposal p; memset(&p, 0, sizeof(Proposal)); p.key = ql[l].key; p.itask = ThisTask; p.iindex = ql[l].index; p.jindex = best[l].jindex;
        pout[best[l].jtask].push_back(p);
        W[p.iindex] = p.key;
    }
    std::vector<Proposal> pin = exchange(pout);
    for(Proposal &p : pin)
    {
        if(p.jindex < 0 || p.jindex >= NumPart) {endrun(97566); p.jindex = -1; continue;}
        auto w = W.find(p.jindex); if(w == W.end()) {W[p.jindex] = p.key;} else if(key_less(p.key, w->second)) {w->second = p.key;}
    }
    std::vector<std::vector<Verdict>> vout(NTask);
    for(const Proposal &p : pin) {Verdict v; v.iindex = p.iindex; v.ok = (p.jindex >= 0 && key_equal(p.key, W[p.jindex])) ? 1 : 0; vout[p.itask].push_back(v);}
    std::vector<Verdict> vin = exchange(vout);
    std::unordered_set<int> ok_at_target; for(const Verdict &v : vin) {if(v.ok) {ok_at_target.insert(v.iindex);}}

    /* 4. the accepted proposers send their records to the target's rank */
    const size_t rec = sizeof(XferHead) + sizeof(PartT) + sizeof(CellT);
    std::vector<std::vector<char>> xout(NTask); long long n_remote = 0; (void) n_remote;
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(ql[l].kind != 1 || !best[l].found) {continue;}
        int i = ql[l].index; if(!ok_at_target.count(i) || !key_equal(ql[l].key, W[i])) {continue;}
        XferHead hd; memset(&hd, 0, sizeof(XferHead)); hd.key = ql[l].key; hd.itask = ThisTask; hd.iindex = i; hd.jindex = best[l].jindex; hd.type = P[i].Type;
        std::vector<char> &b = xout[best[l].jtask]; size_t n0 = b.size(); b.resize(n0 + rec, 0);
        memcpy(b.data() + n0, &hd, sizeof(XferHead)); memcpy((void *)(b.data() + n0 + sizeof(XferHead)), (const void *) &P[i], sizeof(PartT));
        if(P[i].Type == 0) {memcpy((void *)(b.data() + n0 + sizeof(XferHead) + sizeof(PartT)), (const void *) &CellP[i], sizeof(CellT));}
        if(best[l].jtask != ThisTask) {n_remote++;}
    }
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
    { /* records waiting for a kick name cells by rank and index: none may be pending when a cell changes rank */
        long long loc[2] = {n_remote, pairflux_restart_count()}, glob[2];
        MPI_Allreduce(loc, glob, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(glob[0] > 0 && glob[1] > 0)
        { /* the merges across ranks of this pass wait (those on one rank go ahead, as without the flag); the result of this pass then depends on the rank count */
            if(ThisTask == 0) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: %lld merges across ranks postponed, %lld pair-flux records pending\n", glob[0], glob[1]); fflush(stdout);}
            for(int t = 0; t < NTask; t++) {if(t != ThisTask) {xout[t].clear();}}
        }
    }
#endif
    std::vector<char> xin = exchange(xout);
    const int nx = (int)(xin.size() / rec);
    auto head = [&](int x) {XferHead hd; memcpy(&hd, xin.data() + (size_t) x * rec, sizeof(XferHead)); return hd;};

    /* the absorbed cell at a local index for merge_particles_ij: its own slot, or a copy in the free slot behind the gas block (gas) or the particles (others) */
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
    /* room for the copy: CellP[N_gas] for gas, and P[N_gas] exists or P[NumPart] is free; without room the merge waits (this rank only: rank-dependent, reported) */
    auto has_room = [&](int x) {XferHead hd = head(x); if(hd.itask == ThisTask) {return true;}
        if(hd.type == 0) {return (N_gas < All.MaxPartGas) && ((N_gas < NumPart) || (NumPart < All.MaxPart));} return NumPart < All.MaxPart;};

    /* 5. landing points: duplicates and points on existing cells, decided alike on every rank */
    std::vector<Landing> ll(nx);
    for(int x = 0; x < nx; x++)
    {
        XferHead hd = head(x);
        Landing &L = ll[x]; memset(&L, 0, sizeof(Landing)); L.key = hd.key; L.h = P[hd.jindex].KernelRadius; L.type = hd.type; L.itask = hd.itask; L.iindex = hd.iindex; L.jtask = ThisTask; L.jindex = hd.jindex;
        if(!has_room(x)) {L.type = -1; printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: no free slot on task %d, a merge across ranks waits\n", ThisTask); continue;} /* claims no landing point */
        int slot = place(x);
        Vec3<MyDouble> pm = merged_position(slot, hd.jindex); box_wrap_position_to_primary_image(pm);
        for(int k = 0; k < 3; k++) {L.pos[k] = pm[k];}
        unplace(x);
    }
    std::vector<int> lfirst; std::vector<Landing> lall = gather_all(ll, lfirst);
    const int nl = (int) lall.size(); std::vector<int> order(nl); for(int g = 0; g < nl; g++) {order[g] = g;}
    std::sort(order.begin(), order.end(), [&](int a, int b) {return key_less(lall[a].key, lall[b].key);});
    std::vector<int> refused(nl, 0), refused_glob(nl, 0);
    {
        std::set<std::array<double,3>> claimed;
        for(int o = 0; o < nl; o++)
        {
            const Landing &L = lall[order[o]]; if(L.type < 0) {refused[order[o]] = 1; continue;}
            if(!claimed.insert({L.pos[0], L.pos[1], L.pos[2]}).second) {refused[order[o]] = 1;} /* the same on every rank: the smaller key keeps the point */
        }
    }
    if(nl > 0)
    { /* a cell of this rank exactly at a landing point (other than the pair itself) */
        std::vector<double> lpos(3 * (size_t) nl), lrad(nl); std::vector<int64_t> off; std::vector<int> ngb; int local_count = 0;
        for(int g = 0; g < nl; g++) {for(int k = 0; k < 3; k++) {lpos[3 * g + k] = lall[g].pos[k];} lrad[g] = 1.e-6 * lall[g].h;}
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
        MPI_Allreduce(refused.data(), refused_glob.data(), nl, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    /* 6. the merges, on the target's rank; the owner of an absorbed cell on another rank is told the outcome */
    int n_merged = 0; std::vector<std::vector<Verdict>> dout(NTask);
    for(int x = 0; x < nx; x++)
    {
        XferHead hd = head(x); int did = 0;
        if(!refused_glob[lfirst[ThisTask] + x])
        {
            int slot = place(x);
#ifdef TEST_WAKEUP_DUMP
            {Key kj = key_of(hd.jindex); printf("TEST_MERGE_PAIR step %lld absorbed %llu %llu %lld into %llu %llu %lld ranks %d %d\n", (long long) All.NumCurrentTiStep, hd.key.id, hd.key.child, hd.key.gen, kj.id, kj.child, kj.gen, hd.itask, ThisTask);}
#endif
            did = (merge_particles_ij(slot, hd.jindex, -1) == 1) ? 1 : 0;
            unplace(x);
            if(did) {n_merged++;}
        }
        if(hd.itask != ThisTask) {Verdict v; v.iindex = hd.iindex; v.ok = did; dout[hd.itask].push_back(v);}
    }
    std::vector<Verdict> din = exchange(dout);
    for(const Verdict &v : din)
    { /* what merge_particles_ij does to the absorbed cell; rearrange_particle_sequence removes it */
        if(v.iindex < 0 || v.iindex >= NumPart) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: acknowledgement for index %d outside [0,%d) on task %d\n", v.iindex, NumPart, ThisTask); endrun(97570); continue;}
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
    return n_merged;
}

} /* namespace msx */
