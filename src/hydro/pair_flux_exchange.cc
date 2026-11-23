#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"
#include "../mesh/ghost_writeback.h"
#include "../core/wakeup_sidecar.h"
#include "pair_flux_exchange.h"
#include <cstring>
#include <vector>
#include <algorithm>

/* Exact pair-flux exchange across timebins (HYDRO_EXACT_PAIR_FLUX_EXCHANGE; port of stella_dev hydro/pair_flux_exchange.cc onto
   gku's two-sided hydro loop) and the per-sync conserved-totals diagnostic (TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC).
   gku evaluates every pair from each active side, each cell adding its own flux F to its rates. A pair is evaluated at every
   sync where either cell is active; each evaluation's flux gets the pair weights (pair_flux_exchange.h): the evaluator's own
   kicks are corrected to them, and an inactive partner receives -F with the same weights as conserved impulses, after the
   closing and after the opening kick. When both cells are active each evaluation is shared half and half (1/2 F_own W kept,
   -1/2 F_own W to the partner), so the pair is antisymmetric even where the two evaluations differ (a Riemann branch flipping
   between the sides). Corrections for a partner on another rank are sent there. */

#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
static double TestKickSplit[5] = {0}; /* reserved columns (stella_dev kick energy split); not filled on gku */
#endif

#ifdef TEST_PAIRFLUX_ADD_MASS_AND_WAKE
#ifndef TEST_PAIRFLUX_HOOK_MODE
#define TEST_PAIRFLUX_HOOK_MODE 1
#endif
static double TestInjected[8] = {0}; /* mass, momentum, thermal + kinetic energy added by the hook (physical) */
/* test only, at the given step, after the closing kick, gas within 0.2 box of the centre:
   mode 1: inactive cells gain half their mass at fixed velocity and u (a feedback deposit) and are woken
   (merges and splits are driven by the test's ICs, stella_dev 2d3174035/2a0da74a7; stella_dev's mode 4 needs MHD).
   Additions are booked so the test can subtract them. */
void test_pairflux_add_mass_and_wake(void)
{
    if(All.NumCurrentTiStep != TEST_PAIRFLUX_ADD_MASS_AND_WAKE) {return;}
    const int mode = TEST_PAIRFLUX_HOOK_MODE; int n = 0; double a = All.cf_atime;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        int active = TimeBinActive[P[i].TimeBin];
        if(mode == 1 && active) {continue;}
        Vec3<double> dx = P[i].Pos - Vec3<double>{0.5*All.BoxSize, 0.5*All.BoxSize, 0.5*All.BoxSize};
        for(int k = 0; k < 3; k++) {if(dx[k] > 0.5*All.BoxSize) {dx[k] -= All.BoxSize;} if(dx[k] < -0.5*All.BoxSize) {dx[k] += All.BoxSize;}}
        if(dx.norm() > 0.2 * All.BoxSize) {continue;}
        if(mode != 1) {printf("TEST_PAIRFLUX_HOOK_MODE %d is not available on gku (1 only)\n", mode); endrun(97544);}
        double fac = 1.5;
        double m = P[i].Mass, dM = (fac - 1.) * m; Vec3<double> v = P[i].Vel / a;
        P[i].Mass += dM; CellP[i].Mass += dM;
        TestInjected[0] += dM; for(int k = 0; k < 3; k++) {TestInjected[1+k] += dM * v[k];}
        TestInjected[4] += dM * (CellP[i].InternalEnergy + 0.5 * v.norm_sq());
        if(mode == 1 && !active) {P[i].wakeup = -1; NeedToWakeupParticles_local = 1; wakeup_sidecar_mark(i);}
        n++;
    }
    long long nl = n, ng = 0; MPI_Allreduce(&nl, &ng, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("TEST_PAIRFLUX_ADD_MASS_AND_WAKE: mode %d changed %lld cells at step %lld\n", mode, ng, (long long) All.NumCurrentTiStep);}
}
#endif

#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE

struct pairflux_record
{
    int e, r_task, r_index, flags;   /* evaluator (local), partner's home rank and index; flags: 1 partner active at the hydro pass */
    double c2_e, c2_r;               /* closing corrections: pair weight minus what the evaluator's closing kick applied; the partner's weight */
    Vec3<double> mom; double egy;    /* the evaluator's flux F (physical) */
};
struct pairflux_impulse {int i, mode; Vec3<double> mom; double egy;}; /* mode 0 conserved, 1 conserved + predicted */
struct pairflux_corr {int idx, mode; double mom[3], egy;};             /* sent to a partner's rank */
struct pairflux_req {int idx;};
struct pairflux_rep {long long N;};

static std::vector<pairflux_record> PairFluxRecords;  /* from the hydro pass to the opening kick */
static std::vector<int> PairFluxPassActive;           /* gas cells active at that pass (their rates carry its fluxes) */
static int PairFluxClosingKickExpected = 0, PairFluxRearranged = 0, PairFluxTag = 0;
static long long PairFluxNFloor = 0, PairFluxNStar = 0; static double PairFluxEFloor = 0, PairFluxPeakMB = 0;
static int PairFluxPath = 0; static long long PairFluxNFloorPath[5] = {0}; static double PairFluxEFloorPath[5] = {0}; /* closing, wake, opening, mass change, kick */
static std::vector<pairflux_impulse> PairFluxSums;
static double PairFluxHelperPeakMB = 0;
static void pf_note_helper(size_t bytes) {double mb = (double) bytes / (1024.*1024.); if(mb > PairFluxHelperPeakMB) {PairFluxHelperPeakMB = mb;}}

void pairflux_note_kick_floor(double de) {PairFluxNFloorPath[4]++; PairFluxEFloorPath[4] += de;}
void pairflux_note_rearrange(void) {PairFluxRearranged = 1;}
void pairflux_mark_closing_kick_expected(void) {PairFluxClosingKickExpected = 1; PairFluxRearranged = 0;} /* all ranks, before the pass: rearranges up to here are before the records */
void pairflux_reset_rearranged(void) {PairFluxRearranged = 0;}

static double pf_U(void) {return All.Timebase_interval / All.cf_hubble_a;} /* physical time per integer time (no per-particle dilation with this flag) */
static double pf_opening_applied(int i) /* what do_first_halfstep_kick applied to i's rates (0 if it is no longer a kicked gas cell) */
{
    if(P[i].Type != 0 || P[i].Mass <= 0 || !TimeBinActive[P[i].TimeBin]) {return 0;}
    integertime ti = P[i].integertime_step(); return (double)(ti/2) * pf_U();
}
static long long pf_next_eval(int i) /* the cell's next evaluation on the integer timeline */
{
    long long t = (long long) All.Ti_Current, end = (long long)(P[i].Ti_begstep + P[i].integertime_step());
    return pairflux_next_sync_ti(t, end, (long long) GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin));
}

/* the hydro pass's records (in the loop's cleanup, ghosts still present); the previous pass's are dropped */
void pairflux_take_hydro_records(const struct pairflux_device_record *rec, long long n)
{
    std::vector<pairflux_record>().swap(PairFluxRecords); PairFluxRearranged = 0;
    PairFluxPassActive.clear();
    for(int i : ActiveParticleList) {if(P[i].Type == 0 && P[i].Mass > 0) {PairFluxPassActive.push_back(i);}}
    int nl = ghost_get_num_local(), ng = ghost_get_num_ghosts(); int *home_rank = ghost_get_home_rank(), *home_index = ghost_get_home_index();
    if(ng <= 0) {nl = NumPart;}
    long long t = (long long) All.Ti_Current; double U = pf_U();
    PairFluxRecords.resize(n);
    for(long long k = 0; k < n; k++)
    {
        const struct pairflux_device_record &d = rec[k]; pairflux_record &r = PairFluxRecords[k];
        int e = d.e, j = d.j;
        if(e < 0 || e >= NumPart || j < 0 || j >= nl + (ng > 0 ? ng : 0)) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: record (%d, %d) outside the local %d + ghost %d cells on task %d\n", e, j, nl, ng, ThisTask); endrun(97543);}
        r.e = e;
        if(j < nl) {r.r_task = ThisTask; r.r_index = j;} else {r.r_task = home_rank[j - nl]; r.r_index = home_index[j - nl];}
        r.flags = TimeBinActive[P[j].TimeBin] ? 1 : 0;
        r.mom = Vec3<double>{d.mom[0], d.mom[1], d.mom[2]}; r.egy = d.egy;
        r.c2_e = r.c2_r = 0;
        if(PairFluxClosingKickExpected)
        {
            long long S_e = (long long) P[e].Ti_begstep, S_r = (long long) P[j].Ti_begstep, ti_e = (long long) P[e].integertime_step();
            double W2p = (double) pairflux_pair_closing_ti(t, S_e, S_r) * U;
            r.c2_e = W2p - (double) pairflux_kick_closing_ti(ti_e) * U;
            r.c2_r = W2p;
        }
    }
    std::sort(PairFluxRecords.begin(), PairFluxRecords.end(), [](const pairflux_record &x, const pairflux_record &y) { /* the kernel appends in thread order: fix the summation order */
        return (x.e != y.e) ? (x.e < y.e) : ((x.r_task != y.r_task) ? (x.r_task < y.r_task) : (x.r_index < y.r_index));});
    double mb = (double)(PairFluxRecords.capacity() * sizeof(pairflux_record)) / (1024.*1024.); if(mb > PairFluxPeakMB) {PairFluxPeakMB = mb;}
}

/* conserved-form impulse: momentum and total energy (physical); u takes dE minus the exact kinetic change. Not a gas cell any
   more: the momentum goes to the particle and the thermal remainder is booked as locked by star formation. */
static void pairflux_apply_impulse(int i, Vec3<double> dmom, double dE, int mode)
{
    if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: impulse target %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97532);}
    if(P[i].Mass <= 0) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: impulse target %d has no mass (task %d)\n", i, ThisTask); endrun(97533);}
    double m = P[i].Mass, a = All.cf_atime;
    Vec3<double> v = P[i].Vel / a, vn = v + dmom / m;
    double dKE = 0.5 * m * (vn.norm_sq() - v.norm_sq());
    if(P[i].Type != 0)
    {
        P[i].Vel = vn * a; P[i].dp += dmom * a;
        ESTEP_BOOK(EBK_SF_TH_LOCKED, -(dE - dKE)); PairFluxNStar++;
        gravity_note_motion_bound(&i, 1);
        return;
    }
    double du = (dE - dKE) / m, u_target = CellP[i].InternalEnergy + du;
    if(mode == 1) {CellP[i].InternalEnergyPred = DMAX(internal_energy_positivity_catch(CellP[i].InternalEnergyPred, CellP[i].InternalEnergyPred + du, 0), All.MinEgySpec);}
    CellP[i].InternalEnergy = internal_energy_positivity_catch(CellP[i].InternalEnergy, u_target, 0);
    CellP[i].enforce_temperature_floor();
    double de_floor = (CellP[i].InternalEnergy - u_target) * m;
    if(de_floor != 0) {PairFluxNFloor++; PairFluxEFloor += de_floor; PairFluxNFloorPath[PairFluxPath]++; PairFluxEFloorPath[PairFluxPath] += de_floor;}
    P[i].Vel = vn * a; P[i].dp += dmom * a;
    if(mode == 1) {CellP[i].VelPred += dmom * (a / m);}
    gravity_note_motion_bound(&i, 1);
}

/* impulses are summed per target and mode in a fixed order (conserved + predicted first, then by index), then applied once */
static void pf_reduce_sums(void)
{
    pf_note_helper(PairFluxSums.capacity() * sizeof(pairflux_impulse));
    std::stable_sort(PairFluxSums.begin(), PairFluxSums.end(), [](const pairflux_impulse &x, const pairflux_impulse &y) {
        return (x.mode != y.mode) ? (x.mode > y.mode) : (x.i < y.i);});
    size_t w = 0;
    for(size_t k = 0; k < PairFluxSums.size();)
    {
        pairflux_impulse s = PairFluxSums[k]; size_t m = k + 1;
        while(m < PairFluxSums.size() && PairFluxSums[m].i == s.i && PairFluxSums[m].mode == s.mode) {s.mom += PairFluxSums[m].mom; s.egy += PairFluxSums[m].egy; m++;}
        PairFluxSums[w++] = s; k = m;
    }
    PairFluxSums.resize(w);
}
static void pf_add(int i, int mode, Vec3<double> mom, double egy)
{
    if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: target %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97534);}
    pairflux_impulse s; s.i = i; s.mode = mode; s.mom = mom; s.egy = egy; PairFluxSums.push_back(s);
    if(PairFluxSums.size() > (size_t)(2 * NumPart + 4096)) {pf_reduce_sums();} /* bounded by the number of targets */
}
static void pf_apply_all(void)
{
    pf_reduce_sums();
    for(auto &s : PairFluxSums) {pairflux_apply_impulse(s.i, s.mom, s.egy, s.mode);}
    std::vector<pairflux_impulse>().swap(PairFluxSums);
}

/* sparse exchange (non-blocking consensus): send[t] goes to rank t; every rank returns with all messages addressed to it */
template<class T> static void pf_sparse_exchange(std::vector<std::vector<T>> &send, std::vector<T> &recv, std::vector<int> &from)
{
    const int tag = 7100 + (PairFluxTag++ % 64); std::vector<MPI_Request> reqs;
    for(int t = 0; t < NTask; t++)
    {
        if(send[t].empty()) {continue;}
        if(t == ThisTask) {recv.insert(recv.end(), send[t].begin(), send[t].end()); from.insert(from.end(), send[t].size(), ThisTask); continue;}
        if(send[t].size() * sizeof(T) > (size_t) 2000000000) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: %zu entries to one rank exceed one message (task %d)\n", send[t].size(), ThisTask); endrun(97537);}
        reqs.emplace_back(); MPI_Issend(send[t].data(), (int)(send[t].size() * sizeof(T)), MPI_BYTE, t, tag, MPI_COMM_WORLD, &reqs.back());
    }
    MPI_Request bar; int bar_active = 0, done = 0;
    while(!done)
    {
        int flag; MPI_Status st; MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &st);
        if(flag)
        {
            int nb; MPI_Get_count(&st, MPI_BYTE, &nb); size_t n = nb / sizeof(T), old = recv.size(); recv.resize(old + n);
            MPI_Recv(recv.data() + old, nb, MPI_BYTE, st.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); from.insert(from.end(), n, st.MPI_SOURCE);
        }
        if(!bar_active) {int all; MPI_Testall((int)reqs.size(), reqs.data(), &all, MPI_STATUSES_IGNORE); if(all) {MPI_Ibarrier(MPI_COMM_WORLD, &bar); bar_active = 1;}}
        else {MPI_Test(&bar, &done, MPI_STATUS_IGNORE);}
    }
}

/* a particle rearrange on any rank since the hydro pass makes record indices stale */
static void pf_check_records(const char *where)
{
    int loc[2] = {PairFluxRearranged, (PairFluxRecords.size() > 0) ? 1 : 0}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(glob[0] && glob[1]) {if(ThisTask == 0) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: particles rearranged between the hydro pass and %s; record indices are stale\n", where);} endrun(97531);}
}

static pairflux_corr pf_corr(int idx, int mode, Vec3<double> mom, double egy)
{
    pairflux_corr c; c.idx = idx; c.mode = mode; c.egy = egy; for(int k = 0; k < 3; k++) {c.mom[k] = mom[k];}
    return c;
}
/* partners' corrections: summed per target before sending, applied in source-rank order */
static void pf_send_partner_corrections(std::vector<std::vector<pairflux_corr>> &out)
{
    size_t nb = 0;
    for(auto &v : out)
    {
        nb += v.capacity() * sizeof(pairflux_corr);
        std::stable_sort(v.begin(), v.end(), [](const pairflux_corr &x, const pairflux_corr &y) {return (x.idx != y.idx) ? (x.idx < y.idx) : (x.mode < y.mode);});
        size_t w = 0;
        for(size_t k = 0; k < v.size();)
        {
            pairflux_corr s = v[k]; size_t m = k + 1;
            while(m < v.size() && v[m].idx == s.idx && v[m].mode == s.mode) {for(int d = 0; d < 3; d++) {s.mom[d] += v[m].mom[d];} s.egy += v[m].egy; m++;}
            v[w++] = s; k = m;
        }
        v.resize(w);
    }
    pf_note_helper(nb);
    std::vector<pairflux_corr> in; std::vector<int> from;
    pf_sparse_exchange(out, in, from);
    for(auto &v : out) {std::vector<pairflux_corr>().swap(v);}
    std::vector<size_t> ord(in.size()); for(size_t k = 0; k < ord.size(); k++) {ord[k] = k;}
    std::stable_sort(ord.begin(), ord.end(), [&from](size_t x, size_t y) {return from[x] < from[y];}); /* arrival order must not change the sums */
    for(size_t k : ord) {pairflux_corr &c = in[k]; pf_add(c.idx, c.mode, Vec3<double>{c.mom[0], c.mom[1], c.mom[2]}, c.egy);}
}

/* after do_second_halfstep_kick: the evaluators' and inactive partners' closing shares (targets synchronised: Pred = conserved) */
void pairflux_after_closing_kick(void)
{
    if(!PairFluxClosingKickExpected) {return;}
    pf_check_records("the closing kick");
    PairFluxPath = 0;
    std::vector<std::vector<pairflux_corr>> out(NTask);
    for(auto &rec : PairFluxRecords)
    {
        /* inactive partner: the evaluator keeps F W, the partner gets -F W. Active partner (it evaluates the pair too): each side keeps
           1/2 F_own W and sends -1/2 F_own W, so the pair is antisymmetric whatever the two evaluations give */
        double ws = (rec.flags & 1) ? 0.5 : 1.0, ce = rec.c2_e - (1. - ws) * rec.c2_r, cr = ws * rec.c2_r;
        if(ce != 0) {pf_add(rec.e, 1, rec.mom * ce, rec.egy * ce);}
        if(cr != 0) {out[rec.r_task].push_back(pf_corr(rec.r_index, 1, rec.mom * (-cr), rec.egy * (-cr)));}
    }
    pf_send_partner_corrections(out);
    pf_apply_all();
    PairFluxClosingKickExpected = 0;
}

/* after do_first_halfstep_kick: the evaluators' and inactive partners' opening shares, the mass-change completion; the records go */
void pairflux_after_opening_kick(void)
{
    pf_check_records("the opening kick");
    PairFluxPath = 2;
    long long t = (long long) All.Ti_Current; double U = pf_U();
    /* the partners' next evaluations, from their ranks (after any wake-up) */
    std::vector<std::vector<pairflux_req>> rq(NTask); std::vector<long long> slot(PairFluxRecords.size());
    for(auto &rec : PairFluxRecords) {pairflux_req q; q.idx = rec.r_index; rq[rec.r_task].push_back(q);}
    auto lt = [](const pairflux_req &x, const pairflux_req &y) {return x.idx < y.idx;};
    for(auto &v : rq) {std::sort(v.begin(), v.end(), lt); v.erase(std::unique(v.begin(), v.end(), [](const pairflux_req &x, const pairflux_req &y) {return x.idx == y.idx;}), v.end());}
    for(size_t k = 0; k < PairFluxRecords.size(); k++)
    {
        auto &v = rq[PairFluxRecords[k].r_task]; pairflux_req q; q.idx = PairFluxRecords[k].r_index;
        slot[k] = (long long)(std::lower_bound(v.begin(), v.end(), q, lt) - v.begin());
    }
    pf_note_helper(PairFluxRecords.size() * sizeof(long long));
    std::vector<pairflux_req> qin; std::vector<int> qfrom; pf_sparse_exchange(rq, qin, qfrom);
    std::vector<std::vector<pairflux_rep>> rp(NTask);
    for(size_t k = 0; k < qin.size(); k++)
    {
        int i = qin[k].idx; if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: requested index %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97535);}
        pairflux_rep r; r.N = pf_next_eval(i); rp[qfrom[k]].push_back(r);
    }
    std::vector<pairflux_rep> rin; std::vector<int> rfrom; pf_sparse_exchange(rp, rin, rfrom);
    std::vector<long long> off(NTask, -1); /* replies arrive as one block per source rank, in request order */
    for(size_t k = 0; k < rin.size(); k++) {if(off[rfrom[k]] < 0) {off[rfrom[k]] = (long long) k;}}
    std::vector<std::vector<pairflux_corr>> out(NTask);
    for(size_t k = 0; k < PairFluxRecords.size(); k++)
    {
        pairflux_record &rec = PairFluxRecords[k];
        if(off[rec.r_task] < 0) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: no reply from task %d\n", rec.r_task); endrun(97536);}
        long long N_r = rin[off[rec.r_task] + slot[k]].N, N_e = pf_next_eval(rec.e);
        double w1p = (double) pairflux_pair_opening_ti(t, N_e, N_r) * U;
        int r_active = rec.flags & 1; double ws = r_active ? 0.5 : 1.0; /* as at the closing kick */
        double c1_e = ws * w1p - pf_opening_applied(rec.e), c1_r = ws * w1p;
        if(c1_e != 0) {pf_add(rec.e, 0, rec.mom * c1_e, rec.egy * c1_e);}
        if(c1_r != 0) {out[rec.r_task].push_back(pf_corr(rec.r_index, r_active ? 0 : 1, rec.mom * (-c1_r), rec.egy * (-c1_r)));}
    }
    pf_send_partner_corrections(out);
    pf_apply_all();
    /* a cell whose mass changed after the hydro pass got sum(F) w1 m/m_ref from its opening kick: complete it to sum(F) w1 */
    PairFluxPath = 3;
    for(int i : PairFluxPassActive)
    {
        if(i < 0 || i >= NumPart || P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].PairFluxMassRef <= 0) {continue;}
        double f = pf_opening_applied(i) * (1. - P[i].Mass / CellP[i].PairFluxMassRef);
        if(f != 0) {pairflux_apply_impulse(i, CellP[i].PairFluxSumMom * f, CellP[i].PairFluxSumEgy * f, 0);}
    }
    std::vector<pairflux_record>().swap(PairFluxRecords); std::vector<int>().swap(PairFluxPassActive);
    PairFluxClosingKickExpected = 0;
}

/* a woken cell's wake kicks applied its stale rates over dt_wake: take the pair-flux part back out (its partners deliver it) */
void pairflux_wake_kick_correction(int i, double dt_wake, int to_pred)
{
    if(dt_wake == 0 || P[i].Type != 0) {return;}
    double fm = (CellP[i].PairFluxMassRef > 0) ? P[i].Mass / CellP[i].PairFluxMassRef : 1; /* the kick applied rates per m_ref times the current mass */
    PairFluxPath = 1; pairflux_apply_impulse(i, CellP[i].PairFluxSumMom * (-dt_wake * fm), CellP[i].PairFluxSumEgy * (-dt_wake * fm), to_pred);
}

/* restart: the records and the pass's active cells waiting for the opening kick (same NTask, nothing rearranges in between) */
long long pairflux_restart_count(void) {return (long long) PairFluxRecords.size();}
long long pairflux_restart_count_active(void) {return (long long) PairFluxPassActive.size();}
void *pairflux_restart_buffer(long long n_read) {if(n_read >= 0) {PairFluxRecords.assign(n_read, pairflux_record()); PairFluxRearranged = 0;} return (void *) PairFluxRecords.data();}
void *pairflux_restart_buffer_active(long long n_read) {if(n_read >= 0) {PairFluxPassActive.assign(n_read, 0);} return (void *) PairFluxPassActive.data();}
size_t pairflux_record_size(void) {return sizeof(pairflux_record);}

#endif /* HYDRO_EXACT_PAIR_FLUX_EXCHANGE */


#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
/* gas mass, physical momentum, thermal + kinetic energy from the conserved variables, after every sync's kicks (column layout
   of stella_dev; magnetic, Powell and limiter columns stay 0 on gku) */
void print_gas_conserved_totals(void)
{
    static FILE *fd = NULL;
    double s[12] = {0}, c[12] = {0}; /* Neumaier: M, Px, Py, Pz, E_th, E_kin, E_B, sum m|v|, BVx, BVy, BVz, Ncells */
    auto add = [&](int k, double x) {double tt = s[k] + x; if(fabs(s[k]) >= fabs(x)) {c[k] += (s[k] - tt) + x;} else {c[k] += (x - tt) + s[k];} s[k] = tt;};
    double a = All.cf_atime;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        double m = P[i].Mass; Vec3<double> v = P[i].Vel / a;
        add(0, m); add(1, m*v[0]); add(2, m*v[1]); add(3, m*v[2]); add(4, m*CellP[i].InternalEnergy); add(5, 0.5*m*v.norm_sq()); add(7, m*v.norm()); add(11, 1);
    }
    double loc[51] = {0}, glob[51] = {0};
    for(int k = 0; k < 12; k++) {loc[k] = s[k] + c[k];}
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
    loc[12] = (double) PairFluxNFloor; loc[13] = PairFluxEFloor; loc[14] = (double) PairFluxNStar; loc[15] = 0;
    for(int k = 0; k < 5; k++) {loc[16+2*k] = (double) PairFluxNFloorPath[k]; loc[17+2*k] = PairFluxEFloorPath[k];}
    loc[26] = PairFluxPeakMB; loc[27] = 0;
    loc[50] = PairFluxHelperPeakMB;
#endif
    for(int k = 0; k < 5; k++) {loc[28+k] = TestKickSplit[k];}
#ifdef TEST_PAIRFLUX_ADD_MASS_AND_WAKE
    for(int k = 0; k < 5; k++) {loc[33+k] = TestInjected[k];}
#endif
    MPI_Reduce(loc, glob, 26, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[27], &glob[27], 23, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[50], &glob[50], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[26], &glob[26], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    if(!fd)
    {
        char buf[1000]; snprintf(buf, sizeof(buf), "%s/conserved_totals.txt", All.OutputDir);
        fd = fopen(buf, "w");
        fprintf(fd, "# step time M Px Py Pz E_th E_kin E_B sum_m|v| BVx BVy BVz Ncells n_floor E_floor n_to_star 0 [n,E floor: closing wake opening masschange kick] records_peak_MB_per_rank records_spilled [kick energy split: not filled] [added by the test hook: M Px Py Pz E] [hook B*V x y z] [limiters] [Powell work] [B*V to stars] exchange_helpers_peak_MB_per_rank (gas, conserved variables, physical, after the sync's kicks)\n");
    }
    fprintf(fd, "%lld %.17e", (long long) All.NumCurrentTiStep, All.Time);
    for(int k = 0; k < 51; k++) {fprintf(fd, " %.17e", glob[k]);}
    fprintf(fd, "\n"); fflush(fd);
}
#endif
