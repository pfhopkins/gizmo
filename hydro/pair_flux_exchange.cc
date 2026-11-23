#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"
#include "pair_flux_weights.h"
#include <cstring>
#include <algorithm>

/* Exact pair-flux exchange across timebins (HYDRO_EXACT_PAIR_FLUX_EXCHANGE) and the per-sync conserved-totals
   diagnostic (TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC). Each pair is evaluated once, at every sync where either cell is
   active; its flux F gets the pair weights of pair_flux_weights.h on both sides. Both cells apply their own kick weights
   through their rates; a record per pair supplies each side's difference as a conserved impulse, after the closing and
   after the opening kick. Corrections for an evaluator on another rank are sent there. */

#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
static double TestKickSplit[5] = {0}; /* gas hydro part, gas rest (gravity...), gas floor, non-gas kinetic, pair-flux impulses */
void test_kick_energy_split_add(int gas, double hydro, double floor_e, double total)
{
    if(gas) {TestKickSplit[0] += hydro; TestKickSplit[2] += floor_e; TestKickSplit[1] += total - hydro - floor_e;} else {TestKickSplit[3] += total;}
}
void test_impulse_energy_add(double de) {TestKickSplit[4] += de;}
#endif

#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
static double TestLimiter[2] = {0}, TestPowell[4] = {0}, TestDroppedBV[3] = {0}; /* Dedner correction renormalised / face-area closure != 1 (cell-steps); Powell momentum and energy work; B*V sent to a cell that became a star */
void test_count_mhd_limiters(int phicorr_renormalised, int closure_not_one) {TestLimiter[0] += phicorr_renormalised; TestLimiter[1] += closure_not_one;}
void test_powell_work_add(Vec3<double> dmom, double degy) {for(int k = 0; k < 3; k++) {TestPowell[k] += dmom[k];} TestPowell[3] += degy;}
#endif

#ifdef TEST_PAIRFLUX_ADD_MASS_AND_WAKE
#ifndef TEST_PAIRFLUX_HOOK_MODE
#define TEST_PAIRFLUX_HOOK_MODE 1
#endif
static double TestInjected[8] = {0}; /* mass, momentum, thermal + kinetic + magnetic energy, B*V added by the hook (physical; B*V code) */
/* test only, at the given step, after the closing kick, gas within 0.2 box of the centre:
   mode 1: inactive cells gain half their mass at fixed velocity and u (a feedback deposit) and are woken;
   mode 3: every cell gets 2.5 times its mass and is cleared for splitting (cells then split; the split wakes the daughters);
   mode 4: every cell keeps 0.7 of its mass, B, BPred and DtB, as an IMF donor does; inactive ones are woken.
   Additions (negative: removals) are booked so the test can subtract them. */
void test_pairflux_add_mass_and_wake(void)
{
    if(All.NumCurrentTiStep != TEST_PAIRFLUX_ADD_MASS_AND_WAKE) {return;}
    const int mode = TEST_PAIRFLUX_HOOK_MODE; int n = 0; double a = All.cf_atime;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        int active = TimeBinActive[P[i].TimeBin];
        if(mode == 1 && active) {continue;}
        Vec3<double> dx = P[i].Pos - Vec3<double>{0.5*All.BoxSize, 0.5*All.BoxSize, 0.5*All.BoxSize}; nearest_xyz(dx);
        if(dx.norm() > 0.2 * All.BoxSize) {continue;}
        double fac = (mode == 1) ? 1.5 : ((mode == 3) ? 2.5 : 0.7);
        double m = P[i].Mass, dM = (fac - 1.) * m; Vec3<double> v = P[i].Vel / a;
        double eb0 = 0;
#ifdef MAGNETIC
        eb0 = 0.5 * CellP[i].B.norm_sq() * (CellP[i].Density / m) / a;
        Vec3<double> B0 = CellP[i].B;
        if(mode == 4)
        {
            CellP[i].B *= fac; CellP[i].BPred *= fac; CellP[i].DtB *= fac;
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
            CellP[i].PairFluxBScale *= fac;
#endif
        }
        for(int k = 0; k < 3; k++) {TestInjected[5+k] += CellP[i].B[k] - B0[k];}
#endif
        P[i].Mass += dM; CellP[i].Mass += dM;
        if(mode == 3) {P[i].Time_Of_Last_MergeSplit = -1.e30;} /* past GIZMO's split timing gate */
        double eb1 = 0;
#ifdef MAGNETIC
        eb1 = 0.5 * CellP[i].B.norm_sq() * (CellP[i].Density / P[i].Mass) / a;
#endif
        TestInjected[0] += dM; for(int k = 0; k < 3; k++) {TestInjected[1+k] += dM * v[k];}
        TestInjected[4] += dM * (CellP[i].InternalEnergy + 0.5 * v.norm_sq()) + (eb1 - eb0);
        if((mode == 1 || mode == 4) && !active) {P[i].wakeup = -1; NeedToWakeupParticles_local = 1; wakeup_list_note(i);}
        n++;
    }
    long long nl = n, ng = 0; MPI_Allreduce(&nl, &ng, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("TEST_PAIRFLUX_ADD_MASS_AND_WAKE: mode %d changed %lld cells at step %lld\n", mode, ng, (long long) All.NumCurrentTiStep);}
}
#endif

#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE

struct pairflux_record
{
    int r, e_task, e_index, flags;   /* flags: 1 r active at the hydro pass, 2 cross-bin (e had the smaller step) */
    double c2_e, c2_r;               /* closing corrections: pair weight minus what each side's closing kick applied */
    Vec3<double> mom; double egy;    /* the receiver's flux G = -F (physical) */
#ifdef MAGNETIC
    Vec3<double> B, Bded;            /* B*V flux and its Dedner mean-phi part */
#endif
};
struct pairflux_impulse {int i, mode; Vec3<double> mom; double egy; Vec3<double> B, Bded;}; /* mode 0 conserved, 1 conserved + predicted, 2 predicted only */
struct pairflux_corr {int idx, mode; double mom[3], egy, B[3], Bded[3];};                   /* sent to an evaluator's rank */
struct pairflux_req {int idx;};
struct pairflux_rep {long long N; double w1;};

static std::vector<pairflux_record> PairFluxLoopRecords; /* filled during the hydro loop */
static pairflux_record *PairFluxRecords = NULL;           /* held from the end of the hydro pass to the opening kick (counted in MaxMemSize); movable, so no non-movable block may be allocated above it and kept past the opening kick */
static long long PairFluxNRecords = 0, PairFluxCap = 0, PairFluxNSpill = 0; /* reserved capacity of the block; records that did not fit (counted) */
static int PairFluxClosingKickExpected = 0, PairFluxRearranged = 0, PairFluxTag = 0;
static long long PairFluxNFloor = 0, PairFluxNStar = 0; static double PairFluxEFloor = 0, PairFluxPeakMB = 0;
static int PairFluxPath = 0; static long long PairFluxNFloorPath[5] = {0}; static double PairFluxEFloorPath[5] = {0}; /* closing, wake, opening, mass change, kick */
static std::vector<pairflux_impulse> PairFluxSums;
static void pf_reduce_sums(void);
static double PairFluxHelperPeakMB = 0; /* largest size of the exchange helpers outside MaxMemSize (test output) */
static void pf_note_helper(size_t bytes) {double mb = (double) bytes / (1024.*1024.); if(mb > PairFluxHelperPeakMB) {PairFluxHelperPeakMB = mb;}}
/* sum the impulses per (mode, target) in a fixed order: conserved + predicted, conserved, predicted only; then by index */
static void pf_reduce_sums(void)
{
    static const int rank_of_mode[3] = {1, 0, 2};
    pf_note_helper(PairFluxSums.capacity() * sizeof(pairflux_impulse));
    std::stable_sort(PairFluxSums.begin(), PairFluxSums.end(), [](const pairflux_impulse &x, const pairflux_impulse &y) {
        return (rank_of_mode[x.mode] != rank_of_mode[y.mode]) ? (rank_of_mode[x.mode] < rank_of_mode[y.mode]) : (x.i < y.i);});
    size_t w = 0;
    for(size_t k = 0; k < PairFluxSums.size();)
    {
        pairflux_impulse s = PairFluxSums[k]; size_t m = k + 1;
        while(m < PairFluxSums.size() && PairFluxSums[m].i == s.i && PairFluxSums[m].mode == s.mode) {s.mom += PairFluxSums[m].mom; s.egy += PairFluxSums[m].egy; s.B += PairFluxSums[m].B; s.Bded += PairFluxSums[m].Bded; m++;}
        PairFluxSums[w++] = s; k = m;
    }
    PairFluxSums.resize(w);
}

void pairflux_note_kick_floor(double de) {PairFluxNFloorPath[4]++; PairFluxEFloorPath[4] += de;}
void pairflux_note_rearrange(void) {PairFluxRearranged = 1;}
void pairflux_mark_closing_kick_expected(void) {PairFluxClosingKickExpected = 1;}

static double pf_U(void) {return unit_integertime_in_physical(-1);}
static double pf_closing_default(int i) {return (double) pairflux_kick_closing_ti((long long) P[i].integertime_step()) * pf_U();} /* what do_second_halfstep_kick applies */
static double pf_opening_applied(int i) /* what do_first_halfstep_kick applied to i's rates (0 if it is no longer a kicked gas cell) */
{
    if(P[i].Type != 0 || P[i].Mass <= 0 || !TimeBinActive[P[i].TimeBin]) {return 0;}
    integertime ti = P[i].integertime_step(); return (double)(ti/2) * pf_U();
}
static long long pf_step_end(int i) {return (long long)(P[i].Ti_begstep + P[i].integertime_step());}

static void pf_free_records(void) {if(PairFluxRecords) {myfree_movable(PairFluxRecords); PairFluxRecords = NULL;} std::vector<pairflux_record>().swap(PairFluxLoopRecords); PairFluxNRecords = 0; PairFluxCap = 0;}
static pairflux_record *pf_records(void) {return PairFluxRecords ? PairFluxRecords : PairFluxLoopRecords.data();} /* records read from a restart stay in the vector */

/* before the hydro loop: reserve a counted block for the records from the active cells' neighbour numbers; overflow spills */
void pairflux_begin_hydro_pass(void)
{
    pf_free_records(); PairFluxLoopRecords.clear(); PairFluxRearranged = 0;
    double est = 1024; for(int i : ActiveParticleList) {if(P[i].Type == 0) {est += DMAX(P[i].NumNgb, All.DesNumNgb);}}
    PairFluxCap = (long long) est;
    PairFluxRecords = (pairflux_record *) mymalloc_movable(&PairFluxRecords, "PairFluxRecords", PairFluxCap * sizeof(pairflux_record));
}

/* end of the hydro pass (the loop's transfer buffers are freed): resize the block to the records it holds, append any spill */
void pairflux_end_hydro_pass(void)
{
    long long nb = PairFluxNRecords, ns = (long long) PairFluxLoopRecords.size(), n = nb + ns;
    double mb = (double)((PairFluxCap + ns) * sizeof(pairflux_record)) / (1024.*1024.); if(mb > PairFluxPeakMB) {PairFluxPeakMB = mb;}
    if(n == 0) {myfree_movable(PairFluxRecords); PairFluxRecords = NULL; PairFluxCap = 0; return;}
    PairFluxRecords = (pairflux_record *) myrealloc_movable(PairFluxRecords, n * sizeof(pairflux_record));
    if(ns > 0) {memcpy(PairFluxRecords + nb, PairFluxLoopRecords.data(), ns * sizeof(pairflux_record)); PairFluxNSpill += ns;}
    PairFluxNRecords = n; PairFluxCap = n;
    std::vector<pairflux_record>().swap(PairFluxLoopRecords);
}

/* called in the hydro loop on j's rank for every evaluated pair; S_e = start of e's step that ends now */
void pairflux_record_add(int r, int r_active, int crossbin, int e_task, int e_index, long long S_e, long long ti_e, Vec3<double> mom, double egy, Vec3<double> B, Vec3<double> Bded)
{
    pairflux_record rec; rec.r = r; rec.flags = (r_active ? 1 : 0) | (crossbin ? 2 : 0); rec.e_task = e_task; rec.e_index = e_index;
    rec.c2_e = rec.c2_r = 0;
    if(PairFluxClosingKickExpected)
    {
        long long t = (long long) All.Ti_Current, S_r = (long long) P[r].Ti_begstep;
        double W2p = (double) pairflux_pair_closing_ti(t, S_e, S_r) * pf_U();
        rec.c2_e = W2p - (double) pairflux_kick_closing_ti(ti_e) * pf_U();
        rec.c2_r = W2p - (r_active ? pf_closing_default(r) : 0);
    }
    rec.mom = mom; rec.egy = egy;
#ifdef MAGNETIC
    rec.B = B; rec.Bded = Bded;
#else
    (void) B; (void) Bded;
#endif
    if(PairFluxRecords && PairFluxNRecords < PairFluxCap) {PairFluxRecords[PairFluxNRecords++] = rec;} else {PairFluxLoopRecords.push_back(rec);}
}

/* conserved-form impulse: momentum, total energy and B*V (physical); u takes dE minus the exact kinetic and magnetic
   change, plus the Dedner energy term the kick convention returns (B . dB_Dedner). Not a gas cell any more: the momentum
   goes to the particle and the thermal remainder is booked as locked by star formation. */
static void pairflux_apply_impulse(int i, Vec3<double> dmom, double dE, Vec3<double> dB, Vec3<double> dBded, int mode)
{
    if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: impulse target %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97532);}
    if(P[i].Mass <= 0) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: impulse target %d has no mass (task %d)\n", i, ThisTask); endrun(97533);}
    double m = P[i].Mass, a = All.cf_atime;
    if(P[i].Type != 0)
    {
        if(mode == 2) {return;}
        Vec3<double> v = P[i].Vel / a, vn = v + dmom / m;
        double dKE = 0.5 * m * (vn.norm_sq() - v.norm_sq());
        P[i].Vel = vn * a; P[i].dp += dmom * a; force_raise_motion_bound(i);
        ESTEP_BOOK(EBK_SF_TH_LOCKED, -(dE - dKE)); PairFluxNStar++; /* a B*V share to it has no cell to go to (counted with n_to_star) */
#if defined(TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC) && defined(MAGNETIC)
        for(int k = 0; k < 3; k++) {TestDroppedBV[k] += dB[k] / a;}
#endif
#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
        test_impulse_energy_add(dKE);
#endif
        return;
    }
    double dEB = 0, dEded = 0;
#ifdef MAGNETIC
    Vec3<double> dBc = dB / a; double rho_over_m = (CellP[i].Density > 0) ? CellP[i].Density / m : 0; /* E_B = |VB_code|^2 (rho/m) / (2a), as in do_the_kick */
    dEded = dot(CellP[i].Bfield() * All.cf_a2inv, dBded);
    dEB = 0.5 * ((CellP[i].B + dBc).norm_sq() - CellP[i].B.norm_sq()) * rho_over_m / a;
#else
    (void) dB; (void) dBded;
#endif
    if(mode == 2) /* predicted variables only (an active receiver's cross-bin share; its drift leaves the received flux out) */
    {
        Vec3<double> vp = CellP[i].VelPred / a, vpn = vp + dmom / m;
        double du = (dE + dEded - 0.5 * m * (vpn.norm_sq() - vp.norm_sq()) - dEB) / m;
        CellP[i].VelPred = vpn * a;
        CellP[i].InternalEnergyPred = DMAX(internal_energy_positivity_catch(CellP[i].InternalEnergyPred, CellP[i].InternalEnergyPred + du, 0), All.MinEgySpec);
#ifdef MAGNETIC
        CellP[i].BPred += dBc;
#endif
        return;
    }
    Vec3<double> v = P[i].Vel / a, vn = v + dmom / m;
    double dKE = 0.5 * m * (vn.norm_sq() - v.norm_sq());
#ifdef MAGNETIC
    CellP[i].B += dBc; if(mode == 1) {CellP[i].BPred += dBc;}
#endif
    double du = (dE + dEded - dKE - dEB) / m, u_target = CellP[i].InternalEnergy + du;
    if(mode == 1) {CellP[i].InternalEnergyPred = DMAX(internal_energy_positivity_catch(CellP[i].InternalEnergyPred, CellP[i].InternalEnergyPred + du, 0), All.MinEgySpec);}
    CellP[i].InternalEnergy = internal_energy_positivity_catch(CellP[i].InternalEnergy, u_target, 0);
    CellP[i].enforce_temperature_floor();
    double de_floor = (CellP[i].InternalEnergy - u_target) * m;
    if(de_floor != 0)
    {
        PairFluxNFloor++; PairFluxEFloor += de_floor; PairFluxNFloorPath[PairFluxPath]++; PairFluxEFloorPath[PairFluxPath] += de_floor;
#if defined(SGCHEM) && defined(GALSF_RESOLVEDISM_FB)
        CumulKickFloorEnergy += de_floor; /* booked with the kick floor */
#endif
    }
    P[i].Vel = vn * a; P[i].dp += dmom * a;
    if(mode == 1) {CellP[i].VelPred += dmom * (a / m);}
#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
    test_impulse_energy_add(dKE + (CellP[i].InternalEnergy - (u_target - du)) * m); /* thermal + kinetic actually changed */
#endif
    force_raise_motion_bound(i);
}

/* impulses are summed per target and mode, then applied once */
static void pf_add(int i, int mode, Vec3<double> mom, double egy, Vec3<double> B, Vec3<double> Bded)
{
    if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: target %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97534);}
    pairflux_impulse t; t.i = i; t.mode = mode; t.mom = mom; t.egy = egy; t.B = B; t.Bded = Bded; PairFluxSums.push_back(t);
    if(PairFluxSums.size() > (size_t)(2 * N_gas + 4096)) {pf_reduce_sums();} /* bounded by the number of targets, not of records */
}
static void pf_apply_all(void)
{
    pf_reduce_sums();
    for(auto &s : PairFluxSums) {pairflux_apply_impulse(s.i, s.mom, s.egy, s.B, s.Bded, s.mode);}
    std::vector<pairflux_impulse>().swap(PairFluxSums); /* not held between syncs */
}

/* sparse exchange (non-blocking consensus): send[t] goes to rank t; every rank returns with all messages addressed to it */
template<class T> static void pf_sparse_exchange(std::vector<std::vector<T>> &send, std::vector<T> &recv, std::vector<int> &from)
{
    const int tag = 7100 + (PairFluxTag++ % 64); std::vector<MPI_Request> reqs;
    for(int t = 0; t < NTask; t++)
    {
        if(send[t].empty()) {continue;}
        if(t == ThisTask) {recv.insert(recv.end(), send[t].begin(), send[t].end()); from.insert(from.end(), send[t].size(), ThisTask); continue;}
        if(send[t].size() * sizeof(T) > (size_t) 2000000000) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: %zu corrections to one rank exceed one message (task %d)\n", send[t].size(), ThisTask); endrun(97537);}
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
    int any_records = (PairFluxNRecords > 0), loc[2] = {PairFluxRearranged, any_records}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(glob[0] && glob[1]) {if(ThisTask == 0) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: particles rearranged between the hydro pass and %s; record indices are stale\n", where);} endrun(97531);}
}

/* evaluator corrections: local ones summed now, remote ones sent to the evaluator's rank */
static void pf_reduce_corr(std::vector<pairflux_corr> &v)
{
    std::stable_sort(v.begin(), v.end(), [](const pairflux_corr &x, const pairflux_corr &y) {return (x.idx != y.idx) ? (x.idx < y.idx) : (x.mode < y.mode);});
    size_t w = 0;
    for(size_t k = 0; k < v.size();)
    {
        pairflux_corr s = v[k]; size_t m = k + 1;
        while(m < v.size() && v[m].idx == s.idx && v[m].mode == s.mode) {for(int d = 0; d < 3; d++) {s.mom[d] += v[m].mom[d]; s.B[d] += v[m].B[d]; s.Bded[d] += v[m].Bded[d];} s.egy += v[m].egy; m++;}
        v[w++] = s; k = m;
    }
    v.resize(w);
}
static void pf_send_evaluator_corrections(std::vector<std::vector<pairflux_corr>> &out)
{
    size_t nb = 0; for(auto &v : out) {nb += v.capacity() * sizeof(pairflux_corr); pf_reduce_corr(v);}
    pf_note_helper(nb);
    std::vector<pairflux_corr> in; std::vector<int> from;
    pf_sparse_exchange(out, in, from);
    for(auto &v : out) {std::vector<pairflux_corr>().swap(v);}
    pf_note_helper(in.capacity() * (sizeof(pairflux_corr) + sizeof(int)));
    std::vector<size_t> ord(in.size()); for(size_t k = 0; k < ord.size(); k++) {ord[k] = k;}
    std::stable_sort(ord.begin(), ord.end(), [&from](size_t x, size_t y) {return from[x] < from[y];}); /* arrival order must not change the sums */
    for(size_t k : ord)
    {
        pairflux_corr &c = in[k];
        Vec3<double> mom = {c.mom[0], c.mom[1], c.mom[2]}, B = {c.B[0], c.B[1], c.B[2]}, Bd = {c.Bded[0], c.Bded[1], c.Bded[2]};
        pf_add(c.idx, c.mode, mom, c.egy, B, Bd);
    }
}
static pairflux_corr pf_corr(int idx, int mode, Vec3<double> mom, double egy, Vec3<double> B, Vec3<double> Bd)
{
    pairflux_corr c; c.idx = idx; c.mode = mode; c.egy = egy;
    for(int k = 0; k < 3; k++) {c.mom[k] = mom[k]; c.B[k] = B[k]; c.Bded[k] = Bd[k];}
    return c;
}

/* after do_second_halfstep_kick: both sides' closing corrections (all targets are synchronised, so Pred = conserved) */
void pairflux_after_closing_kick(void)
{
    if(!PairFluxClosingKickExpected) {return;}
    pf_check_records("the closing kick");
    PairFluxPath = 0;
    std::vector<std::vector<pairflux_corr>> out(NTask);
    for(long long k = 0; k < PairFluxNRecords; k++)
    {
        pairflux_record &rec = pf_records()[k];
        Vec3<double> B = {}, Bd = {};
#ifdef MAGNETIC
        B = rec.B; Bd = rec.Bded;
#endif
        if(rec.c2_r != 0) {pf_add(rec.r, 1, rec.mom * rec.c2_r, rec.egy * rec.c2_r, B * rec.c2_r, Bd * rec.c2_r);}
        if(rec.c2_e != 0) {out[rec.e_task].push_back(pf_corr(rec.e_index, 1, rec.mom * (-rec.c2_e), rec.egy * (-rec.c2_e), B * (-rec.c2_e), Bd * (-rec.c2_e)));}
    }
    pf_send_evaluator_corrections(out);
    pf_apply_all();
    PairFluxClosingKickExpected = 0;
}

/* after do_first_halfstep_kick: both sides' opening corrections, the mass-change correction, the active receivers'
   predicted share; then the records are freed */
void pairflux_after_opening_kick(void)
{
    pf_check_records("the opening kick");
    PairFluxPath = 2;
    long long t = (long long) All.Ti_Current; double U = pf_U();
    /* the evaluators' new step ends and applied opening weights, from their ranks */
    std::vector<std::vector<pairflux_req>> rq(NTask); std::vector<long long> slot(PairFluxNRecords);
    pairflux_record *R = pf_records();
    for(long long k = 0; k < PairFluxNRecords; k++) {pairflux_req q; q.idx = R[k].e_index; rq[R[k].e_task].push_back(q);}
    for(auto &v : rq) {std::sort(v.begin(), v.end(), [](const pairflux_req &x, const pairflux_req &y) {return x.idx < y.idx;});
                     v.erase(std::unique(v.begin(), v.end(), [](const pairflux_req &x, const pairflux_req &y) {return x.idx == y.idx;}), v.end());} /* one request per distinct evaluator */
    for(long long k = 0; k < PairFluxNRecords; k++)
    {
        auto &v = rq[R[k].e_task]; pairflux_req q; q.idx = R[k].e_index;
        slot[k] = (long long)(std::lower_bound(v.begin(), v.end(), q, [](const pairflux_req &x, const pairflux_req &y) {return x.idx < y.idx;}) - v.begin());
    }
    pf_note_helper(PairFluxNRecords * sizeof(long long));
    std::vector<pairflux_req> qin; std::vector<int> qfrom; pf_sparse_exchange(rq, qin, qfrom);
    std::vector<std::vector<pairflux_rep>> rp(NTask);
    for(size_t k = 0; k < qin.size(); k++)
    {
        int i = qin[k].idx; if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: requested index %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97535);}
        pairflux_rep r; r.N = pf_step_end(i); r.w1 = pf_opening_applied(i); rp[qfrom[k]].push_back(r);
    }
    std::vector<pairflux_rep> rin; std::vector<int> rfrom; pf_sparse_exchange(rp, rin, rfrom);
    std::vector<long long> off(NTask, -1); /* replies arrive as one block per source rank, in request order */
    for(size_t k = 0; k < rin.size(); k++) {if(off[rfrom[k]] < 0) {off[rfrom[k]] = (long long) k;}}
    std::vector<std::vector<pairflux_corr>> out(NTask);
    for(long long k = 0; k < PairFluxNRecords; k++)
    {
        pairflux_record &rec = pf_records()[k];
        if(off[rec.e_task] < 0) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: no reply from task %d\n", rec.e_task); endrun(97536);}
        pairflux_rep &rep = rin[off[rec.e_task] + slot[k]];
        int r_active = rec.flags & 1;
        long long N_r = r_active ? (long long)(t + P[rec.r].integertime_step()) : pairflux_next_sync_ti(t, pf_step_end(rec.r), (long long) GET_INTEGERTIME_FROM_TIMEBIN(P[rec.r].TimeBin));
        double w1p = (double) pairflux_pair_opening_ti(t, rep.N, N_r) * U;
        double c1_r = w1p - (r_active ? pf_opening_applied(rec.r) : 0), c1_e = w1p - rep.w1;
        Vec3<double> B = {}, Bd = {};
#ifdef MAGNETIC
        B = rec.B; Bd = rec.Bded;
#endif
        if(c1_r != 0) {pf_add(rec.r, r_active ? 0 : 1, rec.mom * c1_r, rec.egy * c1_r, B * c1_r, Bd * c1_r);}
        if(r_active && (rec.flags & 2)) {pf_add(rec.r, 2, rec.mom * w1p, rec.egy * w1p, B * w1p, Bd * w1p);}
        if(c1_e != 0) {out[rec.e_task].push_back(pf_corr(rec.e_index, 0, rec.mom * (-c1_e), rec.egy * (-c1_e), B * (-c1_e), Bd * (-c1_e)));}
    }
    pf_send_evaluator_corrections(out);
    pf_apply_all();
    /* a cell whose mass changed after the hydro pass got sum(F) w1 m/m_hydro from its opening kick: complete it to sum(F) w1 */
    PairFluxPath = 3;
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].PairFluxMassRef <= 0) {continue;}
        double w1 = pf_opening_applied(i), f = w1 * (1. - P[i].Mass / CellP[i].PairFluxMassRef);
        Vec3<double> B = {}, Bd = {};
#ifdef MAGNETIC
        double fb = w1 * (1. - CellP[i].PairFluxBScale); B = CellP[i].PairFluxSumB * fb; Bd = CellP[i].PairFluxSumBded * fb; /* DtB was scaled only where the mass change scaled it */
        if(f == 0 && fb == 0) {continue;}
#else
        if(f == 0) {continue;}
#endif
        pairflux_apply_impulse(i, CellP[i].PairFluxSumMom * f, CellP[i].PairFluxSumEgy * f, B, Bd, 0);
    }
    pf_free_records();
    PairFluxClosingKickExpected = 0;
}

/* a woken cell's wake kicks applied its stale rates over dt_wake: take the pair-flux part back out (its partners never got it) */
void pairflux_wake_kick_correction(int i, double dt_wake, int to_pred)
{
    if(dt_wake == 0 || P[i].Type != 0) {return;}
    Vec3<double> B = {}, Bd = {};
#ifdef MAGNETIC
    B = CellP[i].PairFluxSumB * (-dt_wake * CellP[i].PairFluxBScale); Bd = CellP[i].PairFluxSumBded * (-dt_wake * CellP[i].PairFluxBScale); /* DtB carries this factor */
#endif
    double fm = (CellP[i].PairFluxMassRef > 0) ? P[i].Mass / CellP[i].PairFluxMassRef : 1; /* the kick applied rates per m_ref times the current mass */
    PairFluxPath = 1; pairflux_apply_impulse(i, CellP[i].PairFluxSumMom * (-dt_wake * fm), CellP[i].PairFluxSumEgy * (-dt_wake * fm), B, Bd, to_pred);
}

/* restart: the records waiting for the opening kick (same NTask, nothing rearranges between write and read) */
long long pairflux_restart_count(void) {return PairFluxNRecords;}
void *pairflux_restart_buffer(long long n_read)
{
    if(n_read >= 0) {pf_free_records(); PairFluxLoopRecords.resize(n_read); PairFluxNRecords = n_read;} /* not in the movable heap: blocks allocated after a restart would pin it */
    return (void *) pf_records();
}
size_t pairflux_record_size(void) {return sizeof(pairflux_record);}

#endif /* HYDRO_EXACT_PAIR_FLUX_EXCHANGE */


#ifdef TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC
/* gas mass, physical momentum, thermal + kinetic (+ magnetic) energy from the conserved variables, after every sync's kicks */
void print_gas_conserved_totals(void)
{
    static FILE *fd = NULL;
    double s[12] = {0}, c[12] = {0}; /* Neumaier: M, Px, Py, Pz, E_th, E_kin, E_B, sum m|v|, BVx, BVy, BVz, Ncells */
    auto add = [&](int k, double x) {double t = s[k] + x; if(fabs(s[k]) >= fabs(x)) {c[k] += (s[k] - t) + x;} else {c[k] += (x - t) + s[k];} s[k] = t;};
    double a = All.cf_atime;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        double m = P[i].Mass; Vec3<double> v = P[i].Vel / a;
        add(0, m); add(1, m*v[0]); add(2, m*v[1]); add(3, m*v[2]); add(4, m*CellP[i].InternalEnergy); add(5, 0.5*m*v.norm_sq()); add(7, m*v.norm()); add(11, 1);
#ifdef MAGNETIC
        double rho_over_m = (CellP[i].Density > 0) ? CellP[i].Density / m : 0;
        add(6, 0.5 * CellP[i].B.norm_sq() * rho_over_m / a); add(8, CellP[i].B[0]); add(9, CellP[i].B[1]); add(10, CellP[i].B[2]);
#endif
    }
    double loc[51] = {0}, glob[51];
    for(int k = 0; k < 12; k++) {loc[k] = s[k] + c[k];}
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
    loc[12] = (double) PairFluxNFloor; loc[13] = PairFluxEFloor; loc[14] = (double) PairFluxNStar; loc[15] = 0;
    for(int k = 0; k < 5; k++) {loc[16+2*k] = (double) PairFluxNFloorPath[k]; loc[17+2*k] = PairFluxEFloorPath[k];}
    loc[26] = PairFluxPeakMB; loc[27] = (double) PairFluxNSpill;
#endif
    for(int k = 0; k < 5; k++) {loc[28+k] = TestKickSplit[k];}
#ifdef TEST_PAIRFLUX_ADD_MASS_AND_WAKE
    for(int k = 0; k < 5; k++) {loc[33+k] = TestInjected[k];}
    for(int k = 0; k < 3; k++) {loc[38+k] = TestInjected[5+k];}
#endif
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
    loc[50] = PairFluxHelperPeakMB;
#endif
    loc[41] = TestLimiter[0]; loc[42] = TestLimiter[1]; for(int k = 0; k < 4; k++) {loc[43+k] = TestPowell[k];} for(int k = 0; k < 3; k++) {loc[47+k] = TestDroppedBV[k];}
    MPI_Reduce(loc, glob, 26, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[27], &glob[27], 23, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[50], &glob[50], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[26], &glob[26], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    if(!fd)
    {
        char buf[1000]; snprintf(buf, sizeof(buf), "%s/conserved_totals.txt", All.OutputDir);
        fd = fopen(buf, "w");
        fprintf(fd, "# step time M Px Py Pz E_th E_kin E_B sum_m|v| BVx BVy BVz Ncells n_floor E_floor n_to_star 0 [n,E floor: closing wake opening masschange kick] records_peak_MB_per_rank records_spilled [cumulative kick energy: gas hydro, gas other (gravity), gas floor, non-gas kinetic, pair-flux impulses] [added by the test hook: M Px Py Pz E] [hook B*V x y z] [Dedner correction renormalised, face-area closure != 1 (cell-steps)] [Powell work: momentum x y z, energy] [B*V to cells that became stars x y z] exchange_helpers_peak_MB_per_rank (gas, conserved variables, physical, after the sync's kicks)\n");
    }
    fprintf(fd, "%lld %.17e", (long long) All.NumCurrentTiStep, All.Time);
    for(int k = 0; k < 51; k++) {fprintf(fd, " %.17e", glob[k]);}
    fprintf(fd, "\n"); fflush(fd);
}
#endif
