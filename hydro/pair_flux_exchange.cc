#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/forcetree.h"

/* Exact pair-flux exchange across timebins (HYDRO_EXACT_PAIR_FLUX_EXCHANGE) and the per-sync conserved-totals
   diagnostic (TEST_PRINT_GAS_CONSERVED_TOTALS_EACH_SYNC). Scheme: docs in Template_Config.sh and
   reports/exact_pair_flux.md. Each pair is evaluated once, by the cell with the smaller step (e); its closing and
   opening kick weights W2(e), w1(e) set the pair's time weights. The other cell (r) gets the difference between
   that and what its own rates apply, as conserved impulses after the closing and the opening kick. */

#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE

struct pairflux_record
{
    int r, e_task, e_index, r_active;
    double c2;                       /* closing-kick share coefficient, W2(e) - [r active] W2(r) */
    Vec3<double> mom; double egy;    /* the receiver's flux G = -F (physical) */
#ifdef MAGNETIC
    Vec3<double> B;
#endif
};
static std::vector<pairflux_record> PairFluxRecords;
static int PairFluxClosingKickExpected = 0;  /* set before the main-loop hydro pass: its records carry closing-kick shares */
static long long PairFluxNFloor = 0, PairFluxNLost = 0, PairFluxNNegW2 = 0; /* diagnostics */
static double PairFluxEFloor = 0;
static std::vector<int> PairFluxSlot;       /* receiver -> aggregation slot, -1 when unused */

/* closing-kick weight of i's pair fluxes: the time since its delivered coverage ends (= dt_old/2 without wake-ups) */
double pairflux_closing_weight(int i)
{
    integertime ti = P[i].integertime_step(); long long cov = CellP[i].Ti_PairFluxCoveredP1;
    double U = unit_integertime_in_physical(i);
    if(cov <= 0) {return (double)(ti - ti/2) * U;}
    return (double)(All.Ti_Current - (cov - 1)) * U;
}

/* the weight do_second_halfstep_kick applies to i's rates */
static double pairflux_default_closing_weight(int i) {integertime ti = P[i].integertime_step(); return (double)(ti - ti/2) * unit_integertime_in_physical(i);}

/* the weight do_first_halfstep_kick applied to i's rates (0 if it no longer is a gas cell that was kicked) */
static double pairflux_opening_weight(int i)
{
    if(i < 0 || i >= NumPart || P[i].Type != 0 || P[i].Mass <= 0 || !TimeBinActive[P[i].TimeBin]) {return 0;}
    integertime ti = P[i].integertime_step(); return (double)(ti/2) * unit_integertime_in_physical(i);
}

static int PairFluxRecordsInvalid = 0;     /* the particle array was rearranged while records were held */
void pairflux_begin_hydro_pass(void) {PairFluxRecords.clear(); PairFluxRecordsInvalid = 0;}
void pairflux_note_rearrange(void) {if(!PairFluxRecords.empty()) {PairFluxRecordsInvalid = 1;}}
static void pairflux_check_records(const char *where)
{
    if(PairFluxRecordsInvalid && !PairFluxRecords.empty()) {printf("HYDRO_EXACT_PAIR_FLUX_EXCHANGE: particles rearranged between the hydro pass and %s; record indices are stale\n", where); endrun(97531);}
}
void pairflux_mark_closing_kick_expected(void) {PairFluxClosingKickExpected = 1;}

void pairflux_record_add(int r, int r_active, int e_task, int e_index, double W2_e, Vec3<double> mom, double egy, Vec3<double> B)
{
    pairflux_record rec; rec.r = r; rec.r_active = r_active; rec.e_task = e_task; rec.e_index = e_index;
    rec.c2 = PairFluxClosingKickExpected ? (W2_e - (r_active ? pairflux_closing_weight(r) : 0)) : 0;
    rec.mom = mom; rec.egy = egy;
#ifdef MAGNETIC
    rec.B = B;
#endif
    PairFluxRecords.push_back(rec);
}

/* conserved-form impulse: momentum, total energy and B*V (physical); u takes dE minus the exact kinetic and magnetic change.
   to_pred: the predicted variables get the same change (the cell responds now, as the Riemann solver will see it) */
static void pairflux_apply_impulse(int i, Vec3<double> dmom, double dE, Vec3<double> dB, int to_pred)
{
    if(P[i].Type != 0 || P[i].Mass <= 0) {PairFluxNLost++; return;}
    double m = P[i].Mass, a = All.cf_atime;
    Vec3<double> v = P[i].Vel / a, vn = v + dmom / m;
    double dKE = 0.5 * m * (vn.norm_sq() - v.norm_sq()), dEB = 0;
#ifdef MAGNETIC
    Vec3<double> dBc = dB / a; double rho_over_m = (CellP[i].Density > 0) ? CellP[i].Density / P[i].Mass : 0; /* E_B = |VB_code|^2 (rho/m) / (2a), as in do_the_kick */
    dEB = 0.5 * ((CellP[i].B + dBc).norm_sq() - CellP[i].B.norm_sq()) * rho_over_m / a;
    CellP[i].B += dBc; if(to_pred) {CellP[i].BPred += dBc;}
#else
    (void) dB;
#endif
    double du = (dE - dKE - dEB) / m, u_target = CellP[i].InternalEnergy + du;
    if(to_pred) {CellP[i].InternalEnergyPred = DMAX(internal_energy_positivity_catch(CellP[i].InternalEnergyPred, CellP[i].InternalEnergyPred + du, 0), All.MinEgySpec);}
    CellP[i].InternalEnergy = internal_energy_positivity_catch(CellP[i].InternalEnergy, u_target, 0);
    CellP[i].enforce_temperature_floor();
    double de_floor = (CellP[i].InternalEnergy - u_target) * m;
    if(de_floor != 0)
    {
        PairFluxNFloor++; PairFluxEFloor += de_floor;
#if defined(SGCHEM) && defined(GALSF_RESOLVEDISM_FB)
        CumulKickFloorEnergy += de_floor; /* booked with the kick floor */
#endif
    }
    P[i].Vel = vn * a; P[i].dp += dmom * a;
    if(to_pred) {CellP[i].VelPred += dmom * (a / m);}
    force_raise_motion_bound(i);
}

/* sum impulses per receiver, then apply each once */
struct pairflux_sum {int i, to_pred; Vec3<double> mom; double egy; Vec3<double> B;};
static std::vector<pairflux_sum> PairFluxSums;
static void pairflux_sum_add(int i, int to_pred, Vec3<double> mom, double egy, Vec3<double> B)
{
    if(i < 0 || i >= NumPart) {PairFluxNLost++; return;}
    if((int)PairFluxSlot.size() < NumPart) {PairFluxSlot.resize(NumPart, -1);}
    int s = PairFluxSlot[i];
    if(s < 0) {pairflux_sum t; t.i = i; t.to_pred = to_pred; t.mom = {}; t.egy = 0; t.B = {}; s = PairFluxSlot[i] = (int)PairFluxSums.size(); PairFluxSums.push_back(t);}
    PairFluxSums[s].mom += mom; PairFluxSums[s].egy += egy; PairFluxSums[s].B += B;
    if(!to_pred) {PairFluxSums[s].to_pred = 0;}
}
static void pairflux_sums_apply(void)
{
    for(auto &t : PairFluxSums) {pairflux_apply_impulse(t.i, t.mom, t.egy, t.B, t.to_pred); PairFluxSlot[t.i] = -1;}
    PairFluxSums.clear();
}

/* after do_second_halfstep_kick: own closing weight W2 (contiguous coverage) and the receivers' closing shares */
void pairflux_after_closing_kick(void)
{
    if(!PairFluxClosingKickExpected) {return;}
    pairflux_check_records("the closing kick");
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        double dw = pairflux_closing_weight(i) - pairflux_default_closing_weight(i);
        if(dw < 0) {PairFluxNNegW2++;}
        if(dw == 0) {continue;}
        Vec3<double> B = {};
#ifdef MAGNETIC
        B = CellP[i].PairFluxSumB * dw;
#endif
        pairflux_sum_add(i, 1, CellP[i].PairFluxSumMom * dw, CellP[i].PairFluxSumEgy * dw, B);
    }
    for(auto &rec : PairFluxRecords)
    {
        if(rec.c2 == 0) {continue;}
        Vec3<double> B = {};
#ifdef MAGNETIC
        B = rec.B * rec.c2;
#endif
        pairflux_sum_add(rec.r, 1, rec.mom * rec.c2, rec.egy * rec.c2, B); /* active r: Pred = conserved just after its closing kick */
    }
    pairflux_sums_apply();
    PairFluxClosingKickExpected = 0;
}

/* after do_first_halfstep_kick: receivers' opening shares (remote evaluators' weights fetched), then coverage advanced */
void pairflux_after_opening_kick(void)
{
    pairflux_check_records("the opening kick");
    int n = (int) PairFluxRecords.size();
    std::vector<int> send_count(NTask, 0), recv_count(NTask, 0), send_off(NTask, 0), recv_off(NTask, 0), pos(n, -1);
    for(int k = 0; k < n; k++) {if(PairFluxRecords[k].e_task != ThisTask) {send_count[PairFluxRecords[k].e_task]++;}}
    MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int nsend = 0, nrecv = 0;
    for(int t = 0; t < NTask; t++) {send_off[t] = nsend; nsend += send_count[t]; recv_off[t] = nrecv; nrecv += recv_count[t];}
    std::vector<int> req(nsend > 0 ? nsend : 1), fill(NTask, 0), inc(nrecv > 0 ? nrecv : 1);
    for(int k = 0; k < n; k++) {int t = PairFluxRecords[k].e_task; if(t != ThisTask) {pos[k] = send_off[t] + fill[t]++; req[pos[k]] = PairFluxRecords[k].e_index;}}
    MPI_Alltoallv(req.data(), send_count.data(), send_off.data(), MPI_INT, inc.data(), recv_count.data(), recv_off.data(), MPI_INT, MPI_COMM_WORLD);
    std::vector<double> rep(nrecv > 0 ? nrecv : 1), w_remote(nsend > 0 ? nsend : 1);
    for(int k = 0; k < nrecv; k++) {rep[k] = pairflux_opening_weight(inc[k]);}
    MPI_Alltoallv(rep.data(), recv_count.data(), recv_off.data(), MPI_DOUBLE, w_remote.data(), send_count.data(), send_off.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int k = 0; k < n; k++)
    {
        pairflux_record &rec = PairFluxRecords[k];
        double w1_e = (rec.e_task == ThisTask) ? pairflux_opening_weight(rec.e_index) : w_remote[pos[k]];
        double c1 = w1_e - (rec.r_active ? pairflux_opening_weight(rec.r) : 0);
        if(c1 == 0) {continue;}
        Vec3<double> B = {};
#ifdef MAGNETIC
        B = rec.B * c1;
#endif
        pairflux_sum_add(rec.r, !rec.r_active, rec.mom * c1, rec.egy * c1, B); /* active r: opening-kick type change, prediction as GIZMO's */
    }
    pairflux_sums_apply();
    for(int i : ActiveParticleList) {if(P[i].Type == 0) {CellP[i].Ti_PairFluxCoveredP1 = P[i].Ti_begstep + P[i].integertime_step()/2 + 1;}}
    std::vector<pairflux_record>().swap(PairFluxRecords); /* free: a full sync can hold many records */
    PairFluxClosingKickExpected = 0;
}

/* a woken cell's wake kicks applied its stale rates over dt_wake: take the pair-flux part back out (its partners never got it) */
void pairflux_wake_kick_correction(int i, double dt_wake, int to_pred)
{
    if(dt_wake == 0 || P[i].Type != 0) {return;}
    Vec3<double> B = {};
#ifdef MAGNETIC
    B = CellP[i].PairFluxSumB * (-dt_wake);
#endif
    pairflux_apply_impulse(i, CellP[i].PairFluxSumMom * (-dt_wake), CellP[i].PairFluxSumEgy * (-dt_wake), B, to_pred);
}

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
    double loc[16], glob[16];
    for(int k = 0; k < 12; k++) {loc[k] = s[k] + c[k];}
#ifdef HYDRO_EXACT_PAIR_FLUX_EXCHANGE
    loc[12] = (double) PairFluxNFloor; loc[13] = PairFluxEFloor; loc[14] = (double) PairFluxNLost; loc[15] = (double) PairFluxNNegW2;
#else
    loc[12] = loc[13] = loc[14] = loc[15] = 0;
#endif
    MPI_Reduce(loc, glob, 16, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    if(!fd)
    {
        char buf[1000]; snprintf(buf, sizeof(buf), "%s/conserved_totals.txt", All.OutputDir);
        fd = fopen(buf, "w");
        fprintf(fd, "# step time M Px Py Pz E_th E_kin E_B sum_m|v| BVx BVy BVz Ncells n_floor E_floor n_lost n_negW2 (gas, conserved variables, physical, after the sync's kicks)\n");
    }
    fprintf(fd, "%lld %.17e", (long long) All.NumCurrentTiStep, All.Time);
    for(int k = 0; k < 16; k++) {fprintf(fd, " %.17e", glob[k]);}
    fprintf(fd, "\n"); fflush(fd);
}
#endif
