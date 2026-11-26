/* MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS (included by merge_split.cc): the merge partner and the nearest neighbour that sets a
   split's daughter spacing are found over the cells of all ranks, and the merge choice does not depend on index order, so the
   result of a merge/split pass does not depend on the number of ranks.
   1. every active cell that needs a merge or a split is a query, gathered on all ranks; each rank answers with the search the
      rank-local code uses (the owner like any other rank);
   2. merge: best candidate by (mass, IDs, position); a proposal i -> j is accepted iff i's key (mass, IDs, position) is the
      smallest among all proposals touching i or j: at most one merge per cell and pass, no chains, losers wait for the next pass;
   3. landing points (centres of mass) of the accepted merges are gathered: exact duplicates go to the smaller key, and a landing
      point exactly on another cell of any rank is refused;
   4. the merge runs on the target's rank with merge_particles_ij: the absorbed cell's P and CellP records are sent as bytes and
      placed in the free slot behind the gas block for the call; its owner zeroes it on the target's acknowledgement.
   Star particles (non-gas) take the same path with P only. Merges are done before the splits of the pass. */
#include <vector>
#include <algorithm>
#include <type_traits>

#if defined(SINK_WIND_SPAWN) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES) || defined(SINK_RIAF_SUBEDDINGTON_MODEL) || defined(CHIMES)
#error "MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS does not carry the pair state the spawned-wind, star-cluster and RIAF merge clauses need, nor ChimesGasVars"
#endif

#ifndef MERGE_SPLIT_MATCHING_ROUNDS
#define MERGE_SPLIT_MATCHING_ROUNDS 16 /* cap on the matching rounds of one pass */
#endif

namespace msx {

typedef std::remove_reference<decltype(P[0])>::type PartT;
typedef std::remove_reference<decltype(CellP[0])>::type CellT;

struct Key {double mass; unsigned long long id, child; long long gen; double pos[3];};
struct Query {Key key; double pos[3], h, mass, vel[3], e_light; int type, kind, task, index;}; /* kind 1 merge, 2 split */
struct Answer {int q, found; double r2min; Key jkey; unsigned long long jmix; int jtask, jindex;};
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
/* order of the merge candidates of one cell: mass first; among equal masses a fixed scramble of the two cells' IDs, so that cells with
   the same neighbours do not all propose to the same one; the same for any rank count and order */
static inline unsigned long long pair_mix(const Key &i, const Key &j)
{
    unsigned long long z = (i.id * 0x9E3779B97F4A7C15ULL) ^ (i.child * 0xC2B2AE3D27D4EB4FULL) ^ ((unsigned long long) i.gen * 0x165667B19E3779F9ULL);
    z ^= (j.id + 0x9E3779B97F4A7C15ULL) * 0xBF58476D1CE4E5B9ULL; z ^= (j.child + 1ULL) * 0x94D049BB133111EBULL; z ^= ((unsigned long long) j.gen + 2ULL) * 0xD6E8FEB86659FD93ULL;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL; z ^= z >> 27; z *= 0x94D049BB133111EBULL; z ^= z >> 31;
    return z;
}
static inline bool candidate_less(const Key &a, unsigned long long amix, const Key &b, unsigned long long bmix)
{
    if(a.mass != b.mass) {return a.mass < b.mass;}
    if(amix != bmix) {return amix < bmix;}
    return key_less(a, b);
}

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

/* this rank's answer to one query: nearest cell (split) or best merge candidate among this rank's cells */
static Answer answer_query(const Query &q, int qindex, const std::vector<char> &is_split_query, const std::vector<char> &used)
{
    Answer a; memset(&a, 0, sizeof(Answer)); a.q = qindex; a.r2min = MAX_REAL_NUMBER; a.jtask = ThisTask; a.jindex = -1;
    int dummy = 0, startnode = All.MaxPart;
    MyDouble centre[3] = {(MyDouble) q.pos[0], (MyDouble) q.pos[1], (MyDouble) q.pos[2]};
    int numngb = ngb_treefind_variable_targeted(centre, (MyFloat) q.h, -1, &startnode, 0, &dummy, &dummy, 1 << q.type);
    const double h2 = q.h * q.h;
    for(int n = 0; n < numngb; n++)
    {
        int j = Ngblist[n];
        if((j < 0) || (q.task == ThisTask && j == q.index) || (P[j].Type != q.type) || !(P[j].Mass > 0)) {continue;}
        Vec3<double> dp = {q.pos[0] - P[j].Pos[0], q.pos[1] - P[j].Pos[1], q.pos[2] - P[j].Pos[2]};
        if(q.kind == 2)
        {
            nearest_xyz(dp);
            double r2 = dp.norm_sq(); if(r2 >= h2) {continue;} /* inside the kernel only: what else a search returns depends on the tree, and so on the rank count */
            if(r2 < a.r2min) {a.r2min = r2; a.found = 1;}
            continue;
        }
        nearest_xyz(dp, -1);
        if(dp.norm_sq() >= h2) {continue;} /* the search returns whole nodes: test the kernel here, as the rank-local loop does */
        if(!((P[j].Mass >= q.mass) && (q.mass + P[j].Mass < All.MaxMassForParticleSplit))) {continue;}
        if(is_split_query[j] || used[j]) {continue;} /* a cell that is split in this pass, or already took part in one of its merges, is no target */
#ifdef MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION
        if((q.type == 0) && (P[j].Type == 0))
        { /* merge_kinetic_dissipation_fraction with the proposer's values: it is the lighter cell (mass_j >= mass_i) */
            double dv2 = 0; for(int k = 0; k < 3; k++) {double d = q.vel[k] - P[j].Vel[k]; dv2 += d * d;}
            double ke_com = 0.5 * (q.mass * P[j].Mass / (q.mass + P[j].Mass)) * dv2 * All.cf_a2inv;
            if(ke_com > 0) {double frac = (q.e_light > 0) ? ke_com / (ke_com + q.e_light) : 1; if(frac > MERGE_SPLIT_MAX_KINETIC_DISSIPATION_FRACTION) {continue;}}
        }
#endif
#ifdef GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL
        if(P[j].Type==0 && sf_channel_for_cell(j)!=SF_CHANNEL_SSP && (q.mass+P[j].Mass)*UNIT_MASS_IN_SOLAR > (GALSF_RESOLVEDISM_HYBRID_M_SSP)) {continue;} /* a merge never lifts Stella-band gas above M_SSP */
#endif
        Key kj = key_of(j);
        unsigned long long mj = pair_mix(q.key, kj);
        if(!a.found || candidate_less(kj, mj, a.jkey, a.jmix)) {a.found = 1; a.jkey = kj; a.jmix = mj; a.jindex = j;}
    }
    return a;
}

/* the whole decision and the merges of one pass. r_nearest_split[i] > 0: cell i is to be split with that neighbour distance. Returns the merges done on this rank. */
static int merge_and_find_splits(std::vector<double> &r_nearest_split)
{
    r_nearest_split.assign(NumPart, 0.0);
    std::vector<char> is_split_query(NumPart, 0), used(NumPart, 0); /* used: survivor of a merge of this pass: no target any more; if still too light it proposes again (as a target of the rank-local loop can) */
    std::set<std::array<double,3>> claimed; /* landing points of this pass's merges, the same set on every rank */
    int n_merged_pass = 0; long long n_queries_pass = 0; int rounds_done = 0; const double t_pass = MPI_Wtime();
    /* several rounds of the same matching: cells that lost their target in one round, and survivors that are still too light, propose again
       in the next, to a cell not yet in a merge of this pass. Ends when a round merges nothing anywhere, or at the cap. Splits are decided in the first round. */
    for(int round = 0; round < MERGE_SPLIT_MATCHING_ROUNDS; round++)
    {
    /* 1. queries */
    std::vector<Query> ql;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Mass <= 0) {continue;}
#if defined(GALSF)
        if(!(((P[i].Type==0)||(P[i].Type==4))&&(TimeBinActive[P[i].TimeBin]))) {continue;}
#else
        if(!((P[i].Type==0)&&(TimeBinActive[P[i].TimeBin]))) {continue;}
#endif
        int kind = 0;
        if(does_particle_need_to_be_merged(i)) {kind = 1;}
        else if(round == 0 && does_particle_need_to_be_split(i)) {if((P[i].Type == 0) && !((CellP[i].Density > 0) && isfinite(CellP[i].Density))) {continue;} kind = 2; is_split_query[i] = 1;}
        if(!kind) {continue;}
        Query q; memset(&q, 0, sizeof(Query)); q.key = key_of(i); q.h = P[i].KernelRadius; q.mass = P[i].Mass; q.type = P[i].Type; q.kind = kind; q.task = ThisTask; q.index = i;
        for(int k = 0; k < 3; k++) {q.pos[k] = P[i].Pos[k]; q.vel[k] = P[i].Vel[k];}
        if(P[i].Type == 0) {double vA = CellP[i].Alfven_speed(); q.e_light = P[i].Mass * (CellP[i].InternalEnergyPred + 0.5 * vA * vA);}
        ql.push_back(q);
    }
    std::vector<int> qfirst; std::vector<Query> qall = gather_all(ql, qfirst);
    if(qall.empty()) {break;} /* the same on every rank */
    if(!force_tree_is_allocated()) {printf("MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: no tree for the neighbour search (task %d)\n", ThisTask); endrun(97564);}
    n_queries_pass += (long long) qall.size(); rounds_done++;

    /* 2. every rank answers every query; answers go to the query's owner */
    std::vector<std::vector<Answer>> aout(NTask);
    for(int g = 0; g < (int) qall.size(); g++) {Answer a = answer_query(qall[g], g, is_split_query, used); if(a.found) {aout[qall[g].task].push_back(a);}}
    std::vector<Answer> ain = exchange(aout);
    std::vector<Answer> best(ql.size()); for(size_t l = 0; l < ql.size(); l++) {memset(&best[l], 0, sizeof(Answer)); best[l].r2min = MAX_REAL_NUMBER;}
    for(const Answer &a : ain)
    {
        int l = a.q - qfirst[ThisTask]; if(l < 0 || l >= (int) ql.size()) {endrun(97565);}
        if(ql[l].kind == 2) {if(a.r2min < best[l].r2min) {best[l].r2min = a.r2min; best[l].found = 1; best[l].jtask = a.jtask;}}
        else if(!best[l].found || candidate_less(a.jkey, a.jmix, best[l].jkey, best[l].jmix)) {best[l] = a;}
    }
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(!(ql[l].kind == 2 && best[l].found && best[l].r2min < MAX_REAL_NUMBER && best[l].r2min > 0)) {continue;}
        r_nearest_split[ql[l].index] = sqrt(best[l].r2min);
#ifdef TEST_WAKEUP_DUMP
        printf("TEST_SPLIT_NEIGHBOR step %lld ID %llu task %d neighbour_task %d\n", (long long) All.NumCurrentTiStep, ql[l].key.id, ThisTask, best[l].jtask);
#endif
    }

    /* 3. proposals and the order-free matching */
    std::vector<Key> W(NumPart); std::vector<char> hasW(NumPart, 0);
    std::vector<std::vector<Proposal>> pout(NTask);
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(ql[l].kind != 1 || !best[l].found) {continue;}
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
    std::vector<std::vector<char>> xout(NTask); long long n_remote = 0;
    for(size_t l = 0; l < ql.size(); l++)
    {
        if(ql[l].kind != 1 || !best[l].found) {continue;}
        int i = ql[l].index; if(!ok_at_target[i] || !key_equal(ql[l].key, W[i])) {continue;}
        XferHead hd; memset(&hd, 0, sizeof(XferHead)); hd.key = ql[l].key; hd.itask = ThisTask; hd.iindex = i; hd.jindex = best[l].jindex; hd.type = P[i].Type;
        std::vector<char> &b = xout[best[l].jtask]; size_t n0 = b.size(); b.resize(n0 + rec, 0);
        memcpy(b.data() + n0, &hd, sizeof(XferHead)); memcpy(b.data() + n0 + sizeof(XferHead), &P[i], sizeof(PartT));
        if(P[i].Type == 0) {memcpy(b.data() + n0 + sizeof(XferHead) + sizeof(PartT), &CellP[i], sizeof(CellT));}
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
    PartT saved; int saved_valid = 0;
    auto place = [&](int x) {
        XferHead hd = head(x); if(hd.itask == ThisTask) {return hd.iindex;}
        const char *src = xin.data() + (size_t) x * rec + sizeof(XferHead); int slot;
        if(hd.type == 0) {
            slot = N_gas; saved_valid = (N_gas < NumPart); if(saved_valid) {memcpy(&saved, &P[slot], sizeof(PartT));}
            memcpy(&P[slot], src, sizeof(PartT)); memcpy(&CellP[slot], src + sizeof(PartT), sizeof(CellT));
        } else {
            slot = NumPart; saved_valid = 0; memcpy(&P[slot], src, sizeof(PartT));
        }
        return slot;
    };
    auto unplace = [&](int x) {
        XferHead hd = head(x); if(hd.itask == ThisTask) {return;}
        if(hd.type == 0 && saved_valid) {memcpy(&P[N_gas], &saved, sizeof(PartT));} /* the particle that lives there */
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
        Vec3<double> pm = merged_position(slot, hd.jindex);
        for(int k = 0; k < 3; k++) {L.pos[k] = pm[k];}
        unplace(x);
    }
    std::vector<int> lfirst; std::vector<Landing> lall = gather_all(ll, lfirst);
    const int nl = (int) lall.size(); std::vector<int> order(nl); for(int g = 0; g < nl; g++) {order[g] = g;}
    std::sort(order.begin(), order.end(), [&](int a, int b) {return key_less(lall[a].key, lall[b].key);});
    std::vector<int> refused(nl, 0), refused_glob(nl, 0);
    {
        for(int o = 0; o < nl; o++)
        {
            const Landing &L = lall[order[o]]; if(L.type < 0) {refused[order[o]] = 1; continue;}
            std::array<double,3> key = {L.pos[0], L.pos[1], L.pos[2]};
#ifdef BOX_PERIODIC
            const double box3[3] = {boxSize_X, boxSize_Y, boxSize_Z};
            for(int d = 0; d < 3; d++) {key[d] -= floor(key[d] / box3[d]) * box3[d];} /* primary-box image */
#endif
            if(!claimed.insert(key).second) {refused[order[o]] = 1;} /* the same on every rank: the smaller key keeps the point */
        }
    }
    for(int g = 0; g < nl; g++)
    { /* a cell of this rank exactly at the landing point (other than the pair itself) */
        const Landing &L = lall[g]; int dummy = 0, startnode = All.MaxPart; if(L.type < 0) {continue;}
        MyDouble centre[3] = {(MyDouble) L.pos[0], (MyDouble) L.pos[1], (MyDouble) L.pos[2]};
        int numngb = ngb_treefind_variable_targeted(centre, (MyFloat)(1.e-6 * L.h), -1, &startnode, 0, &dummy, &dummy, 1 << L.type);
        for(int n = 0; n < numngb; n++)
        {
            int k = Ngblist[n];
            if((k < 0) || !(P[k].Mass > 0) || (L.itask == ThisTask && k == L.iindex) || (L.jtask == ThisTask && k == L.jindex)) {continue;}
            Vec3<double> dk = {P[k].Pos[0] - L.pos[0], P[k].Pos[1] - L.pos[1], P[k].Pos[2] - L.pos[2]}; nearest_xyz(dk, -1);
            if(dk.norm_sq() == 0) {refused[g] = 1; break;}
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
#ifdef TEST_WAKEUP_DUMP
            {Key kj = key_of(hd.jindex); printf("TEST_MERGE_PAIR step %lld absorbed %llu %llu %lld into %llu %llu %lld ranks %d %d\n", (long long) All.NumCurrentTiStep, hd.key.id, hd.key.child, hd.key.gen, kj.id, kj.child, kj.gen, hd.itask, ThisTask);}
#endif
            did = (merge_particles_ij(slot, hd.jindex) == 1) ? 1 : 0;
            unplace(x);
            if(did) {force_cover_position(hd.jindex); n_merged++; used[hd.jindex] = 1;} /* the centre of mass may lie outside j's node boxes */
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
    if(n_round_glob == 0) {break;} /* nothing merged anywhere in this round: the remaining proposals cannot be served in this pass */
    } /* rounds */
#if defined(TEST_WAKEUP_DUMP) || defined(MERGE_SPLIT_REPORT_PASS_COST)
    { /* test output: size and cost of the pass */
        long long loc = n_merged_pass, glob = 0; MPI_Allreduce(&loc, &glob, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(ThisTask == 0 && n_queries_pass > 0) {printf("TEST_MERGE_SPLIT_PASS step %lld rounds %d queries %lld merges %lld seconds %.6f\n", (long long) All.NumCurrentTiStep, rounds_done, n_queries_pass, glob, MPI_Wtime() - t_pass);}
    }
#else
    (void) n_queries_pass; (void) rounds_done; (void) t_pass;
#endif
    return n_merged_pass;
}

} /* namespace msx */
