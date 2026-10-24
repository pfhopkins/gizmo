#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <sys/stat.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <functional>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
#include "resolvedism_stellar_tables.h"
#include "resolvedism_star_age.h"
#endif
#include "stella_star_mergers.h"
#include "rumi_lib.h"

/*! \file rumi_coupling.cc
 *  \brief Type-4 stars integrated by the collisional N-body code RUMI++ (STARS_INTEGRATED_BY_RUMI), as a BRIDGE:
 *
 *  - Member stars share one timebin (rumi_limit_timesteps). Their host kicks carry the field of everything but stars:
 *    the tree drops star sources for star targets (TREE_STAR_STAR_EXCLUDED_TYPE).
 *  - At the start of the common step (after the first half kick) the members go to the RUMI++ ranks, RUMI++ integrates
 *    their mutual gravity over the step and the end state comes back (rumi_integrate). The stars drift on the chord to
 *    that end point, so the gas sees them on the chord; after the drift ending the step they land on RUMI++'s state
 *    (rumi_finish_drift). Outputs inside a step get RUMI++'s state there (the step is split at the output time).
 *  - Stars born mid-step (not yet members) feel every star by an exact direct sum (rumi_nonmember_star_gravity) on
 *    their own KDK steps, which are cut to end at the members' next common sync, where they join.
 *  - RUMI++ keeps the members' state; GIZMO's changes reach it as differences against what it returned: gas kicks
 *    (their work is booked: the energy ledger), box wraps, mass changes in place (set_mass, books E_sse), new or
 *    removed stars (a new set). Mergers in RUMI++ come back as Stella merger products (stella_star_mergers.cc).
 *  - G is GIZMO's (All.G), converted to pc, Msun, km/s at each step. RUMI++ runs on tasks 0..RumiNumRanks-1 (its
 *    distributed path on a sub-communicator), Kokkos OpenMP threads or one GPU per RUMI++ rank.
 */

#ifdef STARS_INTEGRATED_BY_RUMI

/* separations wrapped only when gravity itself is periodic (as forcetree.cc) */
#if (defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)) || defined(GRAVITY_PERIODIC_XY_IMAGES)
#define GRAVITY_NEAREST_XYZ(x,y,z,sign) NEAREST_XYZ(x,y,z,sign)
#else
#define GRAVITY_NEAREST_XYZ(x,y,z,sign)
#endif

#define RUMI_COUPLING_API_VERSION 2 /* rumi_lib.h interface these calls are written for (Makefile checks librumi.a) */
static_assert(RUMI_LIB_API_VERSION == RUMI_COUPLING_API_VERSION, "rumi_lib.h is not the RUMI++ interface version rumi_coupling.cc expects");
#ifndef RUMI_LIB_ID_MK
#define RUMI_LIB_ID_MK "unknown"
#endif

namespace {

/* a star's identity: stars spawned from one gas cell share ID and differ in child number / generation */
struct StarKey {
    unsigned long long id, child, gen;
    bool operator<(const StarKey &o) const {return id != o.id ? id < o.id : (child != o.child ? child < o.child : gen < o.gen);}
    bool operator==(const StarKey &o) const {return id == o.id && child == o.child && gen == o.gen;}
};
struct StarKeyHash {size_t operator()(const StarKey &k) const {return std::hash<unsigned long long>()(k.id * 0x9E3779B97F4A7C15ULL ^ (k.child << 20) ^ k.gen);}};
inline StarKey star_key(int i) {return StarKey{(unsigned long long)P[i].ID, (unsigned long long)P[i].ID_child_number, (unsigned long long)P[i].ID_generation};}
struct RumiStar { StarKey ID; int task, index, kw; integertime ti_step; double m, pos[3], vel[3], r_rsun; };
struct RumiBack { StarKey ID; int index, has_mid; double m, pos[3], vel[3], mpos[3], mvel[3]; };
struct Plan { int rebuild, n_new, do_mass, split, nsl; double dt, G, dt1; };   /* task 0 -> RUMI++ ranks, per step */

double UL = 1, UM = 1, UV = 1, UT = 1;   /* GIZMO code unit in pc, Msun, km/s, pc/(km/s) */
MPI_Comm RumiComm = MPI_COMM_NULL;       /* tasks 0..RumiNumRanks-1 */
int rumi_rank = 0;                        /* this task runs RUMI++ */
/* task 0 coupling state (saved in restart files) */
std::vector<StarKey> slot_id;             /* library slot -> star */
std::unordered_map<StarKey, int, StarKeyHash> slot_of;
std::vector<double> last_m, last_x, last_v; /* by slot, GIZMO units, the state last handed back (mass: GIZMO's own) */
std::vector<double> slot_m_r;             /* by slot, the mass RUMI++ holds */
long long n_steps = 0, n_split = 0, n_merge = 0, n_direct = 0;
double E0_led = 0; int have_E0 = 0, replay_written = 0;
double tw_limit = 0, tw_gather = 0, tw_lib = 0, tw_scatter = 0, tw_direct = 0;
FILE *FdRumi = NULL;

inline int is_rumi_star(int i) {return (P[i].Type == 4) && (P[i].Mass > 0);}

double rumi_G(void) {return All.G * UL * UV * UV / UM;} /* G [L^3 M^-1 T^-2] in pc^3 Msun^-1 (km/s)^2 */

double star_radius_rsun(int i) /* as KETJU's contact radius: Stella tables, else the ZAMS fit */
{
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
    if(!P[i].sampled || P[i].MstarSampleIMF[0] <= 0) return 0;
    double logM = log10(P[i].MstarSampleIMF[0]), logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
    double age_tab = resolvedism_star_age_yr(i) - stellar_t_PMS(logM, logZ);
    if(age_tab >= stellar_lifetime(logM, logZ)) return 0;
    return pow(10.0, stellar_log_R_cm(logM, logZ, log10(DMAX(age_tab, 100.0)))) / 6.957e10;
#else
    double M = P[i].Mass * UNIT_MASS_IN_SOLAR;
    return (M < 1) ? pow(M, 0.8) : pow(M, 0.57);
#endif
}

void move_to_bin(int i, int new_bin) /* as ketju_coupling.cc move_particle_timebin */
{
    int old_bin = P[i].TimeBin;
    if(old_bin == new_bin) return;
    TimeBinCount[old_bin]--;
    if(PrevInTimeBin[i] >= 0) {NextInTimeBin[PrevInTimeBin[i]] = NextInTimeBin[i];} else {FirstInTimeBin[old_bin] = NextInTimeBin[i];}
    if(NextInTimeBin[i] >= 0) {PrevInTimeBin[NextInTimeBin[i]] = PrevInTimeBin[i];} else {LastInTimeBin[old_bin] = PrevInTimeBin[i];}
    if(FirstInTimeBin[new_bin] >= 0) {PrevInTimeBin[FirstInTimeBin[new_bin]] = i;}
    NextInTimeBin[i] = FirstInTimeBin[new_bin];
    PrevInTimeBin[i] = -1;
    FirstInTimeBin[new_bin] = i;
    if(LastInTimeBin[new_bin] < 0) {LastInTimeBin[new_bin] = i;}
    TimeBinCount[new_bin]++;
    P[i].TimeBin = new_bin;
}

void rumi_atexit(void) {rumi_lib::kokkos_finalize();}

/* RUMI_COUPLING_CHECK: report coincident bodies (debug) */
void check_coincident(const char *where, int n, const double *x, const double *m, const std::vector<StarKey> &ids)
{
    static int on = getenv("RUMI_COUPLING_CHECK") != NULL; if(!on) return;
    std::vector<int> o(n); for(int k = 0; k < n; k++) {o[k] = k;}
    std::sort(o.begin(), o.end(), [&](int a, int b) {return x[3*a] < x[3*b];});
    for(int a = 0; a < n; a++) for(int b = a + 1; b < n && x[3*o[b]] - x[3*o[a]] < 1e-9; b++) {
        int i = o[a], j = o[b]; if(m && (m[i] <= 0 || m[j] <= 0)) continue;
        double d2 = 0; for(int q = 0; q < 3; q++) {double d = x[3*i+q] - x[3*j+q]; d2 += d*d;}
        if(d2 < 1e-18) {printf("RUMI_COUPLING_CHECK %s t=%.10e: ids %llu.%llu %llu.%llu r=%.3e\n", where, All.Time, ids[i].id, ids[i].child, ids[j].id, ids[j].child, sqrt(d2));}
    }
}

/* the members' group: whether any exists, whether it starts a step now, and its current step's end (all tasks agree) */
void member_status(int &exist, int &sync, long long &ti_end)
{
    int loc[2] = {0, 0}; long long te = LLONG_MAX;
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i) || !P[i].RumiMember) continue;
        loc[0] = 1;
        if(TimeBinActive[P[i].TimeBin]) {loc[1] = 1;}
        else {long long e = (long long)P[i].Ti_begstep + (long long)P[i].dt_step; if(e < te) {te = e;}}
    }
    int glob[2]; MPI_Allreduce(loc, glob, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &te, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    exist = glob[0]; sync = glob[1]; ti_end = te;
}

/* host coupling state <-> bytes (restart files) */
template <class V> void put(std::vector<char> &b, const V &v) {const char *p = (const char *)&v; b.insert(b.end(), p, p + sizeof(V));}
template <class V> void put_vec(std::vector<char> &b, const std::vector<V> &v) {put(b, (long long)v.size()); const char *p = (const char *)v.data(); b.insert(b.end(), p, p + v.size() * sizeof(V));}
template <class V> void get(const std::vector<char> &b, size_t &o, V &v) {std::memcpy(&v, b.data() + o, sizeof(V)); o += sizeof(V);}
template <class V> void get_vec(const std::vector<char> &b, size_t &o, std::vector<V> &v) {long long n; get(b, o, n); v.resize(n); if(n) {std::memcpy(v.data(), b.data() + o, n * sizeof(V));} o += n * sizeof(V);}

} // namespace


const char *rumi_lib_id_string(void) {return RUMI_LIB_ID_MK;}

/* task 0, with each printed ETOT_INCODE: the energy RUMI++ collisions/mergers changed (-E_merge), in code units, over |E0| */
void rumi_note_total_energy(double E0)
{
    if(ThisTask != 0) return;
    double bk[7]; rumi_lib::books(bk);
    double dE = -bk[2] / (UM * UV * UV);
    printf("RUMI ENERGY: t=%.6f collisions-dE=%.6e kick-work=%.6e (|E0| units)\n", All.Time, (E0 != 0) ? dE / fabs(E0) : 0.0, (E0 != 0) ? rumi_lib::ledger().W_kick / (UM * UV * UV) / fabs(E0) : 0.0);
}

/* begrun: units; on the RUMI++ ranks Kokkos (after MPI_Init), the sub-communicator and the RUMI++ configuration */
void rumi_init(void)
{
    if(All.ComovingIntegrationOn) {if(ThisTask == 0) {printf("STARS_INTEGRATED_BY_RUMI: comoving integration is not supported\n");} endrun(9301);}
    UL = UNIT_LENGTH_IN_PC; UM = UNIT_MASS_IN_SOLAR; UV = UNIT_VEL_IN_KMS;
    if(All.RumiLengthUnitPc > 0) {UL = All.RumiLengthUnitPc;} /* e.g. G = 1 test units mapped 1:1 onto pc, Msun, km/s */
    if(All.RumiMassUnitMsun > 0) {UM = All.RumiMassUnitMsun;}
    if(All.RumiVelUnitKms > 0) {UV = All.RumiVelUnitKms;}
    UT = UL / UV;
    int nr = All.RumiNumRanks; if(nr < 1) {nr = 1;} if(nr > NTask) {nr = NTask;}
    rumi_rank = (ThisTask < nr);
    MPI_Comm_split(MPI_COMM_WORLD, rumi_rank ? 0 : MPI_UNDEFINED, ThisTask, &RumiComm);
    int ok = 1;
    if(rumi_rank) {
        if(rumi_lib::kokkos_initialize(All.RumiNumThreads, All.RumiDeviceId) != 0) {ok = 0;}
        else {
            atexit(rumi_atexit); /* endrun(0) calls MPI_Finalize and exit directly */
            if(nr > 1 && rumi_lib::attach_comm(MPI_Comm_c2f(RumiComm)) != 0) {ok = 0;}
            if(ok && rumi_lib::configure(All.RumiConfigFile, ThisTask == 0) != 0) {ok = 0;}
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(!ok) {endrun(9302);}
    if(ThisTask == 0) {
        printf("RUMI++: %s (built against %s), Kokkos %s on %d rank(s); G = %.10e pc^3/Msun (km/s)^2 (All.G = %.10e); units pc %.10g Msun %.10g km/s %.10g\n",
               rumi_lib::build_id(), RUMI_LIB_ID_MK, rumi_lib::backend_name(), nr, rumi_G(), All.G, UL, UM, UV);
        if(strcmp(rumi_lib::build_id(), RUMI_LIB_ID_MK) != 0) {printf("RUMI++: the linked library is not the one the Makefile checked\n"); endrun(9307);}
        mkdir(All.OutputDir, 02755); /* begrun runs this before open_outputfiles creates OutputDir */
        char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi.txt", All.OutputDir);
        FdRumi = fopen(buf, RestartFlag == 0 ? "w" : "a");
        if(!FdRumi) {printf("RUMI++: cannot open %s\n", buf); endrun(9308);}
        {fprintf(FdRumi, "# %s\n# step time N E_stars E_books ledger_residual/|E_ref| E_ref W_kick W_kick_abs E_set (RUMI++ units: pc Msun km/s) | books escape gw merge sse | wall s: limit gather advance rebuild h2d d2h scatter direct | mergers splits\n", rumi_lib::build_id()); fflush(FdRumi);}
    }
}

void rumi_finalize(void)
{
    if(!rumi_rank) return;
    if(ThisTask == 0) {
        const rumi_lib::Timers &T = rumi_lib::timers(); const rumi_lib::Ledger &L = rumi_lib::ledger();
        printf("RUMI++ totals: %lld steps, %lld set rebuilds, %lld mass updates, %lld split steps, %lld mergers; wall s: limit %.3f gather %.3f rumi %.3f (advance %.3f rebuild %.3f h2d %.3f d2h %.3f) scatter %.3f direct %.3f\n",
               n_steps, T.n_rebuild, T.n_set_mass, n_split, n_merge, tw_limit, tw_gather, tw_lib, T.advance, T.rebuild, T.h2d, T.d2h, tw_scatter, tw_direct);
        printf("RUMI ENERGY: kick work net %.6e abs %.6e set changes %.6e (RUMI++ units)\n", L.W_kick, L.W_kick_abs, L.E_set);
    }
    if(All.RumiReplayDump) {char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi_replay_final.txt", All.OutputDir); rumi_lib::dump_ic(buf);}
    if(FdRumi) {fclose(FdRumi); FdRumi = NULL;}
    rumi_lib::kokkos_finalize();
}


/* after find_timesteps. Members starting a step now (or, before the first step, every synchronised star): one bin,
   the smallest any of them chose, RumiStarMaxTimestep, and not past the end of a non-member's current step (it joins
   there). Non-members starting a step while the members are mid-step: steps that end at the members' next sync. */
void rumi_limit_timesteps(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    double t0 = MPI_Wtime();
    int exist, sync; long long ti_grp;
    member_status(exist, sync, ti_grp);
    const int group_step = (!exist || sync);
    int bmin = TIMEBINS; long long ti_ns = LLONG_MAX;
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i)) continue;
        if(TimeBinActive[P[i].TimeBin]) {if(P[i].TimeBin > 0 && P[i].TimeBin < bmin) {bmin = P[i].TimeBin;}}
        else if(!P[i].RumiMember) {long long te = (long long)P[i].Ti_begstep + (long long)P[i].dt_step; if(te < ti_ns) {ti_ns = te;}}
    }
    int bmin_g; long long ti_ns_g;
    MPI_Allreduce(&bmin, &bmin_g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ti_ns, &ti_ns_g, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    if(bmin_g >= TIMEBINS) {tw_limit += MPI_Wtime() - t0; CPU_Step[CPU_RUMI] += measure_time(); return;}
    int bin = bmin_g;
    if(group_step) {
        if(All.RumiStarMaxTimestep > 0) {
            long long ti_cap = (long long)(All.RumiStarMaxTimestep / All.Timebase_interval);
            int bcap = get_timestep_bin((integertime)DMAX(ti_cap, 2)); if(bcap < bin) {bin = bcap;}
        }
        if(ti_ns_g < LLONG_MAX && ti_ns_g > (long long)All.Ti_Current) {int bns = get_timestep_bin((integertime)(ti_ns_g - (long long)All.Ti_Current)); if(bns < bin) {bin = bns;}}
        while(bin > 0 && !TimeBinActive[bin]) {bin--;}
        if(bin > 0) {
            integertime ti_bin = GET_INTEGERTIME_FROM_TIMEBIN(bin);
            for(int i = 0; i < NumPart; i++) {if(is_rumi_star(i) && TimeBinActive[P[i].TimeBin]) {move_to_bin(i, bin); P[i].dt_step = ti_bin;}}
        }
    } else if(ti_grp < LLONG_MAX && ti_grp > (long long)All.Ti_Current) { /* joiners: end no later than the members' sync */
        int bg = get_timestep_bin((integertime)(ti_grp - (long long)All.Ti_Current));
        for(int i = 0; i < NumPart; i++) {
            if(!is_rumi_star(i) || P[i].RumiMember || !TimeBinActive[P[i].TimeBin]) continue;
            int b = P[i].TimeBin; if(b > bg) {b = bg;} while(b > 0 && !TimeBinActive[b]) {b--;}
            if(b > 0 && b != P[i].TimeBin) {move_to_bin(i, b); P[i].dt_step = GET_INTEGERTIME_FROM_TIMEBIN(b);}
        }
    }
    tw_limit += MPI_Wtime() - t0; CPU_Step[CPU_RUMI] += measure_time();
}


/* after do_first_halfstep_kick: when the members start a common step, the synchronised stars (members and joiners)
   go through RUMI++ over that step */
void rumi_integrate(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    double t0 = MPI_Wtime();
    int exist, sync; long long ti_grp;
    member_status(exist, sync, ti_grp);
    if(exist && !sync) {CPU_Step[CPU_RUMI] += measure_time(); return;} /* members mid-step: joiners stay on their KDK steps */
    std::vector<RumiStar> mine;
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i) || !TimeBinActive[P[i].TimeBin]) continue;
        if(!P[i].RumiMember) { /* joining now: its opening half kick carried the direct star-star force, which RUMI++ integrates from here */
            integertime ts = P[i].integertime_step();
            double dtk = get_gravkick_factor(P[i].Ti_begstep, P[i].Ti_begstep + ts / 2, i, 0);
            Vec3<double> dv = P[i].RumiDirectAcc * (All.G * dtk);
            for(int k = 0; k < 3; k++) {P[i].Vel[k] -= dv[k]; P[i].dp[k] -= P[i].Mass * dv[k];}
            P[i].RumiDirectAcc = {};
        }
        RumiStar s; s.ID = star_key(i); s.task = ThisTask; s.index = i; s.ti_step = GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin);
        s.m = P[i].Mass; for(int k = 0; k < 3; k++) {s.pos[k] = P[i].Pos[k]; s.vel[k] = P[i].Vel[k];}
        s.r_rsun = star_radius_rsun(i); s.kw = (P[i].Mass * UNIT_MASS_IN_SOLAR < 0.7) ? 0 : 1;
        mine.push_back(s);
    }
    int nl = (int)mine.size(), ntot = 0;
    MPI_Allreduce(&nl, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(ntot == 0) {CPU_Step[CPU_RUMI] += measure_time(); return;}
    std::vector<int> cnt(NTask), disp(NTask);
    int nb = nl * (int)sizeof(RumiStar);
    MPI_Gather(&nb, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<RumiStar> all;
    if(ThisTask == 0) {int o = 0; for(int t = 0; t < NTask; t++) {disp[t] = o; o += cnt[t];} all.resize(ntot);}
    MPI_Gatherv(mine.data(), nb, MPI_BYTE, all.data(), cnt.data(), disp.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime(); tw_gather += t1 - t0;

    /* ---- task 0 plans the step: deltas of the stars RUMI++ has, mass changes, new or removed stars ---- */
    Plan plan = {0, 0, 0, 0, 0, 0, 0, 0};
    const int nsl = (int)slot_id.size();
    std::vector<int> star_of_slot(nsl, -1);
    std::vector<double> dx, dv, mnew, rnew;
    if(ThisTask == 0) {
        std::sort(all.begin(), all.end(), [](const RumiStar &a, const RumiStar &b) {return a.ID < b.ID;});
        if(getenv("RUMI_COUPLING_CHECK")) {std::vector<double> gx(3*ntot); std::vector<StarKey> gid(ntot); for(int k = 0; k < ntot; k++) {gid[k] = all[k].ID; for(int j = 0; j < 3; j++) {gx[3*k+j] = all[k].pos[j];}} check_coincident("gathered", ntot, gx.data(), NULL, gid);}
        const integertime ti_step = all[0].ti_step;
        for(const RumiStar &s : all) {if(s.ti_step != ti_step) {printf("RUMI++: stars on different bins in one step\n"); endrun(9303);}}
        plan.dt = ti_step * All.Timebase_interval * UT; plan.G = rumi_G(); plan.nsl = nsl;
        for(int k = 0; k < ntot; k++) {auto it = slot_of.find(all[k].ID); if(it == slot_of.end()) {plan.n_new++;} else {star_of_slot[it->second] = k;}}
        for(int q = 0; q < nsl; q++) {if(star_of_slot[q] < 0 && last_m[q] > 0) {plan.rebuild = 1;}} /* a star left */
        if(plan.n_new > 0 || nsl == 0) {plan.rebuild = 1;}
        dx.assign(3*nsl, 0.0); dv.assign(3*nsl, 0.0); mnew.assign(nsl, 0.0); rnew.assign(nsl, 0.0);
        for(int q = 0; q < nsl; q++) {
            int k = star_of_slot[q]; if(k < 0) continue;
            double d[3]; for(int j = 0; j < 3; j++) {d[j] = all[k].pos[j] - last_x[3*q+j];}
            NEAREST_XYZ(d[0], d[1], d[2], 1);
            for(int j = 0; j < 3; j++) {dx[3*q+j] = d[j] * UL; dv[3*q+j] = (all[k].vel[j] - last_v[3*q+j]) * UV;}
            mnew[q] = (all[k].m == last_m[q]) ? slot_m_r[q] : all[k].m * UM; rnew[q] = all[k].r_rsun;
            if(all[k].m != last_m[q]) {plan.do_mass = 1; last_m[q] = all[k].m; slot_m_r[q] = mnew[q];}
        }
        if(nsl > 0) {plan.do_mass = 1;} /* radii age every step; set_mass skips what did not change */
        /* an output inside the step: split it there, so the snapshot gets RUMI++'s state (as KETJU's 95ea45b5) */
        if(All.Ti_nextoutput > All.Ti_Current && All.Ti_nextoutput < All.Ti_Current + ti_step) {plan.split = 1; plan.dt1 = (All.Ti_nextoutput - All.Ti_Current) * All.Timebase_interval * UT;}
    }

    /* ---- RUMI++ ranks: the plan, then the library calls (collective; host arrays read on task 0) ---- */
    std::vector<double> m_mid, x_mid, v_mid, m_out, x_out, v_out;
    std::vector<int> mk, mg; std::vector<double> mp;
    if(rumi_rank) {
        MPI_Bcast(&plan, sizeof(Plan), MPI_BYTE, 0, RumiComm);
        if(plan.nsl > 0) { /* slot bookkeeping lives on task 0: every RUMI++ rank follows the plan */
            rumi_lib::shift(ThisTask == 0 ? dx.data() : nullptr);
            rumi_lib::kick(ThisTask == 0 ? dv.data() : nullptr);           /* the gas work on the stars, booked */
            if(plan.do_mass) {rumi_lib::set_mass(ThisTask == 0 ? mnew.data() : nullptr, ThisTask == 0 ? rnew.data() : nullptr, plan.G);}
        }
        if(plan.rebuild) { /* new set: the library's state for the stars it has, GIZMO's for the others */
            int ns = rumi_lib::n_slots();
            std::vector<double> lm(ns), lx(3*ns), lv(3*ns);
            if(ns > 0) {rumi_lib::get_state(lm.data(), lx.data(), lv.data());}
            std::vector<long long> id; std::vector<double> m, x, v, r; std::vector<int> kw;
            if(ThisTask == 0) {
                id.resize(ntot); m.resize(ntot); x.resize(3*ntot); v.resize(3*ntot); r.resize(ntot); kw.resize(ntot);
                for(int k = 0; k < ntot; k++) {
                    const RumiStar &s = all[k]; id[k] = (long long)s.ID.id; kw[k] = s.kw; r[k] = s.r_rsun;
                    auto it = slot_of.find(s.ID);
                    if(it != slot_of.end() && lm[it->second] > 0) {
                        int q = it->second; m[k] = lm[q];
                        for(int j = 0; j < 3; j++) {x[3*k+j] = lx[3*q+j]; v[3*k+j] = lv[3*q+j];}
                    } else {
                        m[k] = s.m * UM; for(int j = 0; j < 3; j++) {x[3*k+j] = s.pos[j] * UL; v[3*k+j] = s.vel[j] * UV;}
                    }
                }
            }
            if(rumi_lib::set_particles(ntot, id.data(), m.data(), x.data(), v.data(), kw.data(), r.data(), plan.G, plan.dt) != 0) {printf("RUMI++: set_particles failed\n"); endrun(9304);}
            if(ThisTask == 0) {
                slot_id.resize(ntot); slot_of.clear();
                for(int k = 0; k < ntot; k++) {slot_id[k] = all[k].ID; slot_of[all[k].ID] = k;}
                last_m.assign(ntot, 0); last_x.assign(3*ntot, 0); last_v.assign(3*ntot, 0); slot_m_r = m;
                for(int k = 0; k < ntot; k++) {last_m[k] = all[k].m;}
            }
        }
        if(ThisTask == 0 && All.RumiReplayDump && !replay_written) { /* the state, G, step and E_ref of the first advance: a standalone replay */
            char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi_replay.txt", All.OutputDir);
            FILE *f = fopen(buf, "w"); if(f) {fprintf(f, "G_OVERRIDE = %.17g\nDT_MAX_CODE = %.17g\nE_REF = %.17g\n", plan.G, plan.dt, rumi_lib::e_ref()); fclose(f);}
        }
        if(All.RumiReplayDump && !replay_written) {char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi_replay_ic.txt", All.OutputDir); rumi_lib::dump_ic(buf); replay_written = 1;}
        if(getenv("RUMI_COUPLING_CHECK") && ThisTask == 0) {int q = rumi_lib::n_slots(); std::vector<double> cm(q), cx(3*q), cv(3*q); rumi_lib::get_state(cm.data(), cx.data(), cv.data()); check_coincident("before", q, cx.data(), cm.data(), slot_id);}
        const int ns = rumi_lib::n_slots();
        if(plan.split) {
            if(rumi_lib::advance(plan.dt1, plan.G) != 0) {printf("RUMI++: advance failed\n"); endrun(9305);}
            m_mid.resize(ns); x_mid.resize(3*ns); v_mid.resize(3*ns);
            rumi_lib::get_state(m_mid.data(), x_mid.data(), v_mid.data());
            if(rumi_lib::advance(plan.dt - plan.dt1, plan.G) != 0) {printf("RUMI++: advance failed\n"); endrun(9305);}
            n_split++;
        } else if(rumi_lib::advance(plan.dt, plan.G) != 0) {printf("RUMI++: advance failed\n"); endrun(9305);}
        m_out.resize(ns); x_out.resize(3*ns); v_out.resize(3*ns);
        rumi_lib::get_state(m_out.data(), x_out.data(), v_out.data());
        rumi_lib::merger_events(mk, mg, mp);
        n_steps++;
        const int print_now = All.RumiPrintInterval > 0 && (n_steps % All.RumiPrintInterval == 1 || All.RumiPrintInterval == 1);
        double E = 0, Eb = 0;
        if(print_now) {rumi_lib::energy(plan.G, &E, &Eb);} /* collective over the RUMI++ ranks */
        if(ThisTask == 0) {
            check_coincident("after", ns, x_out.data(), m_out.data(), slot_id);
            const rumi_lib::Ledger &L = rumi_lib::ledger();
            double resid = E + Eb - L.W_kick - L.E_set; /* constant up to the integration error */
            if(print_now && !have_E0) {E0_led = resid; have_E0 = 1;}
            if(print_now) {
                double bk[7]; rumi_lib::books(bk); const rumi_lib::Timers &T = rumi_lib::timers();
                if(FdRumi) {fprintf(FdRumi, "%lld %.10e %d %.16e %.6e %.6e %.6e %.6e %.6e %.6e | %.3e %.3e %.3e %.3e | %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f | %lld %lld\n",
                                    n_steps, All.Time, ns, E, Eb, (resid - E0_led) / fabs(rumi_lib::e_ref()), rumi_lib::e_ref(), L.W_kick, L.W_kick_abs, L.E_set,
                                    bk[0], bk[1], bk[2], bk[3], tw_limit, tw_gather, T.advance, T.rebuild, T.h2d, T.d2h, tw_scatter, tw_direct, n_merge, n_split); fflush(FdRumi);}
            }
        }
    }
    /* task 0: the results by star, back to the owners in task order */
    std::vector<RumiBack> back;
    std::vector<int> bcnt(NTask, 0), bdisp(NTask, 0);
    std::vector<star_merger_action> actions;
    if(ThisTask == 0) {
        const int ns = (int)slot_id.size();
        std::vector<char> m_new(ns, 0); /* RUMI++ changed this mass (merger): GIZMO takes it, else GIZMO keeps its own bits */
        for(int q = 0; q < ns; q++) {
            if(m_out[q] != slot_m_r[q]) {m_new[q] = 1; slot_m_r[q] = m_out[q]; last_m[q] = (m_out[q] > 0) ? m_out[q] / UM : 0;}
            for(int j = 0; j < 3; j++) {last_x[3*q+j] = x_out[3*q+j] / UL; last_v[3*q+j] = v_out[3*q+j] / UV;}
        }
        std::unordered_map<int, int> star_of; for(int k = 0; k < ntot; k++) {star_of[slot_of[all[k].ID]] = k;}
        for(size_t e = 0; e < mk.size(); e++) { /* mergers: Stella bookkeeping (stella_star_mergers.cc) */
            auto ik = star_of.find(mk[e]), ig = star_of.find(mg[e]);
            if(ik == star_of.end() || ig == star_of.end()) continue;
            const RumiStar &sk = all[ik->second], &sg = all[ig->second];
            star_merger_action a; memset(&a, 0, sizeof(a));
            a.id_merged = (MyIDType)sg.ID.id; a.id_survivor = (MyIDType)sk.ID.id; a.type_merged = a.type_survivor = 4;
            a.task_merged = sg.task; a.task_survivor = sk.task; a.idx_merged = sg.index; a.idx_survivor = sk.index;
            a.M_merged = sg.m; a.M_survivor_old = sk.m; a.M_survivor_new = mp[e] / UM;
            actions.push_back(a); n_merge++;
            printf("RUMI++ MERGER: ID %llu (M=%g) merged into ID %llu (M=%g->%g)\n", sg.ID.id, sg.m, sk.ID.id, sk.m, a.M_survivor_new);
        }
        back.resize(ntot);
        std::vector<int> order(ntot); for(int k = 0; k < ntot; k++) {order[k] = k;}
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return all[a].task < all[b].task;});
        for(int q = 0; q < ntot; q++) {
            int k = order[q], sl = slot_of[all[k].ID]; RumiBack &b = back[q];
            b.ID = all[k].ID; b.index = all[k].index; b.m = m_new[sl] ? last_m[sl] : -1; b.has_mid = 0;
            for(int j = 0; j < 3; j++) {b.pos[j] = last_x[3*sl+j]; b.vel[j] = last_v[3*sl+j];}
            if(plan.split && m_mid[sl] > 0) {b.has_mid = 1; for(int j = 0; j < 3; j++) {b.mpos[j] = x_mid[3*sl+j] / UL; b.mvel[j] = v_mid[3*sl+j] / UV;}}
            bcnt[all[k].task] += (int)sizeof(RumiBack);
        }
        int o = 0; for(int t = 0; t < NTask; t++) {bdisp[t] = o; o += bcnt[t];}
    }
    double t2 = MPI_Wtime(); tw_lib += t2 - t1;
    std::vector<RumiBack> got(nl);
    MPI_Scatterv(back.data(), bcnt.data(), bdisp.data(), MPI_BYTE, got.data(), nl * (int)sizeof(RumiBack), MPI_BYTE, 0, MPI_COMM_WORLD);
    int mass_changed_local = 0;
    for(const RumiBack &b : got) {
        int i = b.index;
        if(!(star_key(i) == b.ID)) {printf("RUMI++: index mismatch ID %llu vs %llu on task %d\n", (unsigned long long)P[i].ID, b.ID.id, ThisTask); endrun(9306);}
        if(b.m >= 0 && b.m != P[i].Mass) {P[i].Mass = b.m; mass_changed_local = 1;} /* RUMI++ mergers (0: merged away) */
        integertime ti_step = GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin);
        double dt_drift = get_drift_factor(P[i].Ti_current, P[i].Ti_current + ti_step, i, 0);
        Vec3<double> d; for(int j = 0; j < 3; j++) {d[j] = b.pos[j] - P[i].Pos[j];}
        NEAREST_XYZ(d[0], d[1], d[2], 1);
        for(int j = 0; j < 3; j++) {P[i].RumiTargetPos[j] = b.pos[j]; P[i].RumiTrueVel[j] = b.vel[j]; P[i].RumiStartVel[j] = P[i].Vel[j];}
        P[i].RumiMidTi = -1;
        if(b.has_mid) {P[i].RumiMidTi = All.Ti_nextoutput; for(int j = 0; j < 3; j++) {P[i].RumiMidPos[j] = b.mpos[j]; P[i].RumiMidVel[j] = b.mvel[j];}}
        Vec3<double> vnew;
        if(dt_drift > 0) {for(int j = 0; j < 3; j++) {vnew[j] = d[j] / dt_drift;}}
        else {for(int j = 0; j < 3; j++) {vnew[j] = b.vel[j]; P[i].Pos[j] = b.pos[j];}}
        for(int j = 0; j < 3; j++) {P[i].dp[j] += P[i].Mass * (vnew[j] - P[i].Vel[j]); P[i].Vel[j] = vnew[j];} /* node velocities follow (force_update_tree) */
        P[i].RumiTiStart = P[i].Ti_current; P[i].RumiTiEnd = P[i].Ti_current + ti_step;
        P[i].RumiIntegrated = 1; P[i].RumiMember = 1;
    }
    /* mergers: every task applies the same list (victims' and survivors' owners exchange data) */
    int na = (int)actions.size(); MPI_Bcast(&na, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(na > 0) {
        actions.resize(na);
        MPI_Bcast(actions.data(), na * (int)sizeof(star_merger_action), MPI_BYTE, 0, MPI_COMM_WORLD);
        star_mergers_apply(actions, "RUMI++");
        mass_changed_local = 1;
    }
    int mc = 0; MPI_Allreduce(&mass_changed_local, &mc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(mc) {TreeMomentsStaleFlag = 1;}
    tw_scatter += MPI_Wtime() - t2;
    CPU_Step[CPU_RUMI] += measure_time();
}


/* after the drift to a sync point: stars ending their RUMI++ step land on its position and take its velocity */
void rumi_finish_drift(void)
{
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || !P[i].RumiIntegrated || P[i].RumiTiEnd != All.Ti_Current) continue;
        double d[3]; for(int j = 0; j < 3; j++) {d[j] = P[i].Pos[j] - P[i].RumiTargetPos[j];}
        double w[3] = {d[0], d[1], d[2]}; NEAREST_XYZ(w[0], w[1], w[2], 1);
        double err2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2], s2 = 0; for(int j = 0; j < 3; j++) {s2 += P[i].RumiTargetPos[j] * P[i].RumiTargetPos[j];}
        if(err2 > 1e-20 * (s2 + 1e-30)) {printf("RUMI++ WARNING: star %llu drifted %g off its RUMI++ position (task %d)\n", (unsigned long long)P[i].ID, sqrt(err2), ThisTask);}
        for(int j = 0; j < 3; j++) {P[i].Pos[j] = (d[j] == w[j]) ? P[i].RumiTargetPos[j] : P[i].RumiTargetPos[j] + (d[j] - w[j]);} /* keep a box wrap the drift made */
        for(int j = 0; j < 3; j++) {P[i].dp[j] += P[i].Mass * (P[i].RumiTrueVel[j] - P[i].Vel[j]); P[i].Vel[j] = P[i].RumiTrueVel[j];}
        P[i].RumiIntegrated = 0;
    }
}


/* around savepositions: stars on a RUMI++ step write RUMI++'s state at the output, not the chord: its end state, its
   start velocity, or the state of a step split at the output; the second call swaps back (exactly) */
void rumi_swap_output_state(void)
{
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || !P[i].RumiIntegrated) continue;
        if(P[i].RumiTiEnd == All.Ti_nextoutput) {
            for(int j = 0; j < 3; j++) {
                MyDouble x = P[i].Pos[j]; P[i].Pos[j] = P[i].RumiTargetPos[j]; P[i].RumiTargetPos[j] = x;
                MyDouble v = P[i].Vel[j]; P[i].Vel[j] = P[i].RumiTrueVel[j]; P[i].RumiTrueVel[j] = v;
            }
        } else if(P[i].RumiTiStart == All.Ti_nextoutput) {
            for(int j = 0; j < 3; j++) {MyDouble v = P[i].Vel[j]; P[i].Vel[j] = P[i].RumiStartVel[j]; P[i].RumiStartVel[j] = v;}
        } else if(P[i].RumiMidTi == All.Ti_nextoutput) {
            for(int j = 0; j < 3; j++) {
                MyDouble x = P[i].Pos[j]; P[i].Pos[j] = P[i].RumiMidPos[j]; P[i].RumiMidPos[j] = x;
                MyDouble v = P[i].Vel[j]; P[i].Vel[j] = P[i].RumiMidVel[j]; P[i].RumiMidVel[j] = v;
            }
        }
    }
}


/* in gravity_tree, before the G factor: active stars not yet members feel every other star exactly (the tree leaves
   star sources out for all star targets). Softened as the tree's sink pairs (larger of the two kernel radii). The
   members feel the joiners only once these join, at the next common sync (at most one star step later). */
void rumi_nonmember_star_gravity(void)
{
    int nt = 0;
    for(int ii = 0; ii < (int)ActiveParticleList.size(); ii++) {int i = ActiveParticleList[ii]; if(is_rumi_star(i) && !P[i].RumiMember) {nt++;}}
    int ntg = 0; MPI_Allreduce(&nt, &ntg, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(ntg == 0) return;
    double t0 = MPI_Wtime();
    struct Src {double pos[3], m, h; StarKey key;};
    std::vector<Src> loc;
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i)) continue;
        if(P[i].Ti_current != All.Ti_Current) {drift_particle(i, All.Ti_Current);}
        Src s; for(int k = 0; k < 3; k++) {s.pos[k] = P[i].Pos[k];} s.m = P[i].Mass; s.h = ForceSoftening_KernelRadius(i); s.key = star_key(i);
        loc.push_back(s);
    }
    int n = (int)loc.size(), nb = n * (int)sizeof(Src);
    std::vector<int> cnt(NTask), disp(NTask);
    MPI_Allgather(&nb, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int o = 0; for(int t = 0; t < NTask; t++) {disp[t] = o; o += cnt[t];}
    std::vector<Src> src(o / sizeof(Src));
    MPI_Allgatherv(loc.data(), nb, MPI_BYTE, src.data(), cnt.data(), disp.data(), MPI_BYTE, MPI_COMM_WORLD);
    for(int ii = 0; ii < (int)ActiveParticleList.size(); ii++) {
        int i = ActiveParticleList[ii];
        if(!is_rumi_star(i) || P[i].RumiMember) continue;
        StarKey me = star_key(i); double hi = ForceSoftening_KernelRadius(i);
        Vec3<double> acc = {}; double pot = 0; (void)pot;
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
        SymmetricTensor2<double> tidal = {};
#endif
        for(const Src &s : src) {
            if(s.key == me || s.m <= 0) continue;
            Vec3<double> dr; for(int k = 0; k < 3; k++) {dr[k] = s.pos[k] - P[i].Pos[k];}
            GRAVITY_NEAREST_XYZ(dr[0], dr[1], dr[2], -1);
            double r2 = dr.norm_sq(); if(r2 <= 0) continue;
            double r = sqrt(r2), h = DMAX(hi, s.h), fa, f2, fp;
            if(r >= h) {fa = s.m / (r2 * r); f2 = 3.0 * s.m / (r2 * r2 * r); fp = -s.m / r;}
            else {double h_inv = 1. / h, h3_inv = h_inv * h_inv * h_inv, u = r * h_inv; fa = s.m * kernel_gravity(u, h_inv, h3_inv, 1); f2 = s.m * kernel_gravity(u, h_inv, h3_inv, 2); fp = s.m * kernel_gravity(u, h_inv, h3_inv, -1);}
            acc += fa * dr; pot += fp; (void)f2;
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            tidal[0][0] += (-fa + dr[0]*dr[0]*f2); tidal[0][1] += dr[0]*dr[1]*f2; tidal[0][2] += dr[0]*dr[2]*f2;
            tidal[1][1] += (-fa + dr[1]*dr[1]*f2); tidal[1][2] += dr[1]*dr[2]*f2; tidal[2][2] += (-fa + dr[2]*dr[2]*f2);
#endif
        }
        P[i].GravAccel += acc; P[i].RumiDirectAcc = acc;
#ifdef EVALPOTENTIAL
        P[i].Potential += pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
        P[i].tidal_tensorps[0][0] += tidal[0][0]; P[i].tidal_tensorps[0][1] += tidal[0][1]; P[i].tidal_tensorps[0][2] += tidal[0][2];
        P[i].tidal_tensorps[1][1] += tidal[1][1]; P[i].tidal_tensorps[1][2] += tidal[1][2]; P[i].tidal_tensorps[2][2] += tidal[2][2];
#endif
        n_direct++;
    }
    tw_direct += MPI_Wtime() - t0;
}


/* restart files: the RUMI++ library state (bodies, subsystems, books, ledger) and this file's task-0 state, beside
   GIZMO's; a run continued from them is bit-identical to the uninterrupted one */
void rumi_save_state(void)
{
    if(!rumi_rank) return;
    std::vector<char> b;
    if(ThisTask == 0) {
        put_vec(b, slot_id); put_vec(b, last_m); put_vec(b, last_x); put_vec(b, last_v); put_vec(b, slot_m_r);
        put(b, n_steps); put(b, n_split); put(b, n_merge); put(b, n_direct); put(b, E0_led); put(b, have_E0); put(b, replay_written);
    }
    char dir[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(dir, sizeof(dir), "%s/restartfiles/rumi", All.OutputDir);
    if(rumi_lib::save_state(dir, b) != 0) {printf("RUMI++: save_state failed\n"); endrun(9308);}
}

void rumi_load_state(void)
{
    if(!rumi_rank) return;
    std::vector<char> b;
    char dir[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(dir, sizeof(dir), "%s/restartfiles/rumi", All.OutputDir);
    if(rumi_lib::load_state(dir, b) != 0) {printf("RUMI++: load_state failed (%s)\n", dir); endrun(9309);}
    if(ThisTask == 0 && !b.empty()) {
        size_t o = 0;
        get_vec(b, o, slot_id); get_vec(b, o, last_m); get_vec(b, o, last_x); get_vec(b, o, last_v); get_vec(b, o, slot_m_r);
        get(b, o, n_steps); get(b, o, n_split); get(b, o, n_merge); get(b, o, n_direct); get(b, o, E0_led); get(b, o, have_E0); get(b, o, replay_written);
        slot_of.clear(); for(size_t k = 0; k < slot_id.size(); k++) {slot_of[slot_id[k]] = (int)k;}
        printf("RUMI++: restarted from %s, %zu slots, step %lld\n", dir, slot_id.size(), n_steps);
    }
}

#endif /* STARS_INTEGRATED_BY_RUMI */
