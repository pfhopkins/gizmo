#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"
#include "pair_flux_weights.h"
#include <cstring>

/* Exact pair-flux exchange across timebins (HYDRO_EXACT_PAIR_FLUX_EXCHANGE) and the per-sync conserved-totals
   diagnostic (TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC). Each pair is evaluated once, at every sync where either cell is
   active; its flux F gets the pair weights of pair_flux_weights.h on both sides. Both cells apply their own kick weights
   through their rates; a record per pair supplies each side's difference as a conserved impulse, after the closing and
   after the opening kick. Corrections for an evaluator on another rank are sent there. */

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
static pairflux_record *PairFluxRecords = NULL;           /* held from the end of the hydro pass to the opening kick (counted in MaxMemSize) */
static long long PairFluxNRecords = 0;
static int PairFluxClosingKickExpected = 0, PairFluxRearranged = 0, PairFluxTag = 0;
static long long PairFluxNFloor = 0, PairFluxNStar = 0; static double PairFluxEFloor = 0, PairFluxPeakMB = 0;
static int PairFluxPath = 0; static long long PairFluxNFloorPath[5] = {0}; static double PairFluxEFloorPath[5] = {0}; /* closing, wake, opening, mass change, kick */
static std::vector<pairflux_impulse> PairFluxSums; static std::vector<int> PairFluxSlot;

void pairflux_note_kick_floor(double de) {PairFluxNFloorPath[4]++; PairFluxEFloorPath[4] += de;}
void pairflux_note_rearrange(void) {PairFluxRearranged = 1;}
void pairflux_mark_closing_kick_expected(void) {PairFluxClosingKickExpected = 1;}

static double pf_U(void) {return unit_integertime_in_physical(-1);}
static double pf_closing_default(int i) {integertime ti = All.Ti_Current - P[i].Ti_begstep; return (double)(ti - ti/2) * pf_U();} /* what do_second_halfstep_kick applies */
static double pf_opening_applied(int i) /* what do_first_halfstep_kick applied to i's rates (0 if it is no longer a kicked gas cell) */
{
    if(P[i].Type != 0 || P[i].Mass <= 0 || !TimeBinActive[P[i].TimeBin]) {return 0;}
    integertime ti = P[i].integertime_step(); return (double)(ti/2) * pf_U();
}
static long long pf_step_end(int i) {return (long long)(P[i].Ti_begstep + P[i].integertime_step());}

static void pf_free_records(void) {if(PairFluxRecords) {myfree_movable(PairFluxRecords); PairFluxRecords = NULL;} std::vector<pairflux_record>().swap(PairFluxLoopRecords); PairFluxNRecords = 0;}
static pairflux_record *pf_records(void) {return PairFluxRecords ? PairFluxRecords : PairFluxLoopRecords.data();} /* records read from a restart stay in the vector */

void pairflux_begin_hydro_pass(void) {pf_free_records(); PairFluxLoopRecords.clear(); PairFluxRearranged = 0;}

/* end of the hydro pass: move the records into GIZMO's memory budget */
void pairflux_end_hydro_pass(void)
{
    PairFluxNRecords = (long long) PairFluxLoopRecords.size();
    if(PairFluxNRecords > 0)
    {
        PairFluxRecords = (pairflux_record *) mymalloc_movable(&PairFluxRecords, "PairFluxRecords", PairFluxNRecords * sizeof(pairflux_record));
        memcpy(PairFluxRecords, PairFluxLoopRecords.data(), PairFluxNRecords * sizeof(pairflux_record));
    }
    double mb = (double)(PairFluxLoopRecords.capacity() * sizeof(pairflux_record)) / (1024.*1024.); if(mb > PairFluxPeakMB) {PairFluxPeakMB = mb;}
    std::vector<pairflux_record>().swap(PairFluxLoopRecords);
}

/* called in the hydro loop on j's rank for every evaluated pair; S_e = start of e's step that ends now */
void pairflux_record_add(int r, int r_active, int crossbin, int e_task, int e_index, long long S_e, Vec3<double> mom, double egy, Vec3<double> B, Vec3<double> Bded)
{
    pairflux_record rec; rec.r = r; rec.flags = (r_active ? 1 : 0) | (crossbin ? 2 : 0); rec.e_task = e_task; rec.e_index = e_index;
    rec.c2_e = rec.c2_r = 0;
    if(PairFluxClosingKickExpected)
    {
        long long t = (long long) All.Ti_Current, S_r = (long long) P[r].Ti_begstep;
        double W2p = (double) pairflux_pair_closing_ti(t, S_e, S_r) * pf_U(), de = (double)(t - S_e);
        rec.c2_e = W2p - (de - (long long)de/2) * pf_U();
        rec.c2_r = W2p - (r_active ? pf_closing_default(r) : 0);
    }
    rec.mom = mom; rec.egy = egy;
#ifdef MAGNETIC
    rec.B = B; rec.Bded = Bded;
#else
    (void) B; (void) Bded;
#endif
    PairFluxLoopRecords.push_back(rec);
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
        ESTEP_BOOK(EBK_SF_TH_LOCKED, -(dE - dKE)); PairFluxNStar++;
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
    force_raise_motion_bound(i);
}

/* impulses are summed per target and mode, then applied once */
static void pf_add(int i, int mode, Vec3<double> mom, double egy, Vec3<double> B, Vec3<double> Bded)
{
    if(i < 0 || i >= NumPart) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: target %d outside [0,%d) on task %d\n", i, NumPart, ThisTask); endrun(97534);}
    if((int)PairFluxSlot.size() < 3*NumPart) {PairFluxSlot.resize(3*NumPart, -1);}
    int &s = PairFluxSlot[3*i + mode];
    if(s < 0) {pairflux_impulse t; t.i = i; t.mode = mode; t.mom = {}; t.egy = 0; t.B = {}; t.Bded = {}; s = (int)PairFluxSums.size(); PairFluxSums.push_back(t);}
    PairFluxSums[s].mom += mom; PairFluxSums[s].egy += egy; PairFluxSums[s].B += B; PairFluxSums[s].Bded += Bded;
}
static void pf_apply_all(void)
{
    for(int mode : {1, 0, 2}) {for(auto &t : PairFluxSums) {if(t.mode == mode) {pairflux_apply_impulse(t.i, t.mom, t.egy, t.B, t.Bded, mode);}}}
    for(auto &t : PairFluxSums) {PairFluxSlot[3*t.i + t.mode] = -1;}
    PairFluxSums.clear();
}

/* sparse exchange (non-blocking consensus): send[t] goes to rank t; every rank returns with all messages addressed to it */
template<class T> static void pf_sparse_exchange(std::vector<std::vector<T>> &send, std::vector<T> &recv, std::vector<int> &from)
{
    const int tag = 7100 + (PairFluxTag++ % 64); std::vector<MPI_Request> reqs;
    for(int t = 0; t < NTask; t++)
    {
        if(send[t].empty()) {continue;}
        if(t == ThisTask) {recv.insert(recv.end(), send[t].begin(), send[t].end()); from.insert(from.end(), send[t].size(), ThisTask); continue;}
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
static void pf_send_evaluator_corrections(std::vector<std::vector<pairflux_corr>> &out)
{
    std::vector<pairflux_corr> in; std::vector<int> from;
    pf_sparse_exchange(out, in, from);
    for(auto &c : in)
    {
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
    for(long long k = 0; k < PairFluxNRecords; k++) {slot[k] = (long long) rq[R[k].e_task].size(); pairflux_req q; q.idx = R[k].e_index; rq[R[k].e_task].push_back(q);}
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
        long long N_r = r_active ? (long long)(t + P[rec.r].integertime_step()) : pf_step_end(rec.r);
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
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].PairFluxMassRef <= 0 || P[i].Mass == CellP[i].PairFluxMassRef) {continue;}
        double f = pf_opening_applied(i) * (1. - P[i].Mass / CellP[i].PairFluxMassRef);
        Vec3<double> B = {}, Bd = {};
#ifdef MAGNETIC
        B = CellP[i].PairFluxSumB * f; Bd = CellP[i].PairFluxSumBded * f;
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
    B = CellP[i].PairFluxSumB * (-dt_wake); Bd = CellP[i].PairFluxSumBded * (-dt_wake);
#endif
    PairFluxPath = 1; pairflux_apply_impulse(i, CellP[i].PairFluxSumMom * (-dt_wake), CellP[i].PairFluxSumEgy * (-dt_wake), B, Bd, to_pred);
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
    double loc[28] = {0}, glob[28];
    for(int k = 0; k < 12; k++) {loc[k] = s[k] + c[k];}
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
    loc[12] = (double) PairFluxNFloor; loc[13] = PairFluxEFloor; loc[14] = (double) PairFluxNStar; loc[15] = 0;
    for(int k = 0; k < 5; k++) {loc[16+2*k] = (double) PairFluxNFloorPath[k]; loc[17+2*k] = PairFluxEFloorPath[k];}
    loc[26] = PairFluxPeakMB; loc[27] = 0;
#endif
    MPI_Reduce(loc, glob, 26, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc[26], &glob[26], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    if(!fd)
    {
        char buf[1000]; snprintf(buf, sizeof(buf), "%s/conserved_totals.txt", All.OutputDir);
        fd = fopen(buf, "w");
        fprintf(fd, "# step time M Px Py Pz E_th E_kin E_B sum_m|v| BVx BVy BVz Ncells n_floor E_floor n_to_star 0 [n,E floor: closing wake opening masschange kick] records_peak_MB_per_rank (gas, conserved variables, physical, after the sync's kicks)\n");
    }
    fprintf(fd, "%lld %.17e", (long long) All.NumCurrentTiStep, All.Time);
    for(int k = 0; k < 27; k++) {fprintf(fd, " %.17e", glob[k]);}
    fprintf(fd, "\n"); fflush(fd);
}
#endif
