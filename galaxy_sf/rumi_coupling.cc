#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <functional>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
#include "resolvedism_stellar_tables.h"
#include "resolvedism_star_age.h"
#endif
#include "rumi_lib.h"

/*! \file rumi_coupling.cc
 *  \brief Type-4 stars integrated by the collisional N-body code RUMI++ (STARS_INTEGRATED_BY_RUMI), as a BRIDGE:
 *
 *  - Every star is on one common timebin (rumi_limit_timesteps). Its host kicks (do_first/second_halfstep_kick) carry
 *    the field of everything but stars: the tree drops star sources for star targets (TREE_STAR_STAR_EXCLUDED_TYPE).
 *  - At the start of the common step (after the first half kick) all stars go to task 0, RUMI++ integrates their
 *    mutual gravity over the step, and the end state comes back (rumi_integrate). The stars then drift on the chord
 *    to that end point (velocity = displacement / step), so the gas sees them on the chord during the step; after the
 *    drift that ends the step they land on RUMI++'s position and get its velocity (rumi_finish_drift).
 *  - Stars act on the gas through the tree as usual. G is GIZMO's (All.G), converted to pc, Msun, km/s at each step.
 *  - RUMI++ keeps the stars' state between steps; GIZMO's changes (gas kicks, mass loss, new stars) reach it as
 *    differences against what it returned, so a star-only run carries RUMI++'s state with no unit round trip.
 *  - Task 0 runs RUMI++ (Kokkos: OpenMP threads or one GPU); the other tasks wait for the scatter.
 */

#ifdef STARS_INTEGRATED_BY_RUMI

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
struct RumiBack { StarKey ID; int index; double m, pos[3], vel[3]; };

double UL = 1, UM = 1, UV = 1, UT = 1;   /* GIZMO code unit in pc, Msun, km/s, pc/(km/s) */
std::vector<StarKey> slot_id;             /* task 0: library slot -> star */
std::unordered_map<StarKey, int, StarKeyHash> slot_of;
std::vector<double> last_m, last_x, last_v; /* task 0: by slot, GIZMO units, the state last handed back */
std::vector<double> slot_m_r;             /* task 0: by slot, the mass RUMI++ holds (changes only by its mergers) */
long long n_steps = 0;
double E0_lib = 0;
int have_E0 = 0, replay_written = 0;
double tw_gather = 0, tw_lib = 0, tw_scatter = 0, tw_limit = 0;
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

} // namespace


/* begrun: units, and on task 0 Kokkos (after MPI_Init) and the RUMI++ configuration */
void rumi_init(void)
{
    if(All.ComovingIntegrationOn) {if(ThisTask == 0) {printf("STARS_INTEGRATED_BY_RUMI: comoving integration is not supported\n");} endrun(9301);}
    UL = UNIT_LENGTH_IN_PC; UM = UNIT_MASS_IN_SOLAR; UV = UNIT_VEL_IN_KMS; UT = UL / UV;
    int ok = 1;
    if(ThisTask == 0) {
        if(rumi_lib::kokkos_initialize(All.RumiNumThreads, All.RumiDeviceId) != 0) {ok = 0;}
        else {
            atexit(rumi_atexit); /* endrun(0) calls MPI_Finalize and exit directly */
            if(rumi_lib::configure(All.RumiConfigFile, 1) != 0) {ok = 0;}
        }
        if(ok) {
            printf("RUMI++: Kokkos %s on task 0; G = %.10e pc^3/Msun (km/s)^2 (GIZMO All.G = %.10e); units pc %.10g Msun %.10g km/s %.10g\n",
                   rumi_lib::backend_name(), rumi_G(), All.G, UL, UM, UV);
            char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi.txt", All.OutputDir);
            FdRumi = fopen(buf, RestartFlag == 0 ? "w" : "a");
            if(FdRumi) {fprintf(FdRumi, "# step time N E_stars E_books dE/|E0| (RUMI++ units: pc Msun km/s) | wall s: limit gather advance rebuild h2d d2h scatter | books escape gw merge sse pn sd tidal, mergers since the last set\n"); fflush(FdRumi);}
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(!ok) {endrun(9302);}
}

void rumi_finalize(void)
{
    if(ThisTask != 0) return;
    const rumi_lib::Timers &T = rumi_lib::timers();
    printf("RUMI++ totals: %lld steps, %lld set rebuilds; wall s: limit %.3f gather %.3f rumi %.3f (advance %.3f rebuild %.3f h2d %.3f d2h %.3f) scatter %.3f\n",
           n_steps, T.n_rebuild, tw_limit, tw_gather, tw_lib, T.advance, T.rebuild, T.h2d, T.d2h, tw_scatter);
    if(All.RumiReplayDump) {char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi_replay_final.txt", All.OutputDir); rumi_lib::dump_ic(buf);}
    if(FdRumi) {fclose(FdRumi); FdRumi = NULL;}
    rumi_lib::kokkos_finalize();
}


/* after find_timesteps: every synchronised star onto one bin, the smallest any of them chose (and RumiStarMaxTimestep);
   while some stars are mid-step (e.g. new stars), the synchronised ones step at most to the next end of those, so
   the sets merge there */
void rumi_limit_timesteps(void)
{
    double t0 = MPI_Wtime();
    int bmin = TIMEBINS; long long ti_ns = LLONG_MAX;
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i)) continue;
        if(TimeBinActive[P[i].TimeBin]) {if(P[i].TimeBin > 0 && P[i].TimeBin < bmin) {bmin = P[i].TimeBin;}}
        else {long long te = (long long)P[i].Ti_begstep + (long long)P[i].dt_step; if(te < ti_ns) {ti_ns = te;}}
    }
    int bmin_g; long long ti_ns_g;
    MPI_Allreduce(&bmin, &bmin_g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ti_ns, &ti_ns_g, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    if(bmin_g >= TIMEBINS) {tw_limit += MPI_Wtime() - t0; return;}
    int bin = bmin_g;
    if(All.RumiStarMaxTimestep > 0) {
        long long ti_cap = (long long)(All.RumiStarMaxTimestep / All.Timebase_interval);
        int bcap = get_timestep_bin((integertime)DMAX(ti_cap, 2)); if(bcap < bin) {bin = bcap;}
    }
    if(ti_ns_g < LLONG_MAX && ti_ns_g > (long long)All.Ti_Current) {int bns = get_timestep_bin((integertime)(ti_ns_g - (long long)All.Ti_Current)); if(bns < bin) {bin = bns;}}
    while(bin > 0 && !TimeBinActive[bin]) {bin--;}
    if(bin <= 0) {tw_limit += MPI_Wtime() - t0; return;}
    integertime ti_bin = GET_INTEGERTIME_FROM_TIMEBIN(bin);
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i) || !TimeBinActive[P[i].TimeBin]) continue;
        move_to_bin(i, bin);
        P[i].dt_step = ti_bin;
    }
    tw_limit += MPI_Wtime() - t0;
}


/* after do_first_halfstep_kick: the synchronised stars (all of them, normally) go through RUMI++ over their step */
void rumi_integrate(void)
{
    double t0 = MPI_Wtime();
    std::vector<RumiStar> mine;
    for(int i = 0; i < NumPart; i++) {
        if(!is_rumi_star(i) || !TimeBinActive[P[i].TimeBin]) continue;
        RumiStar s; s.ID = star_key(i); s.task = ThisTask; s.index = i; s.ti_step = GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin);
        s.m = P[i].Mass; for(int k = 0; k < 3; k++) {s.pos[k] = P[i].Pos[k]; s.vel[k] = P[i].Vel[k];}
        s.r_rsun = star_radius_rsun(i); s.kw = (P[i].Mass * UNIT_MASS_IN_SOLAR < 0.7) ? 0 : 1;
        mine.push_back(s);
    }
    int nl = (int)mine.size(), ntot = 0;
    MPI_Allreduce(&nl, &ntot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(ntot == 0) return;
    std::vector<int> cnt(NTask), disp(NTask);
    int nb = nl * (int)sizeof(RumiStar);
    MPI_Gather(&nb, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<RumiStar> all;
    if(ThisTask == 0) {int o = 0; for(int t = 0; t < NTask; t++) {disp[t] = o; o += cnt[t];} all.resize(ntot);}
    MPI_Gatherv(mine.data(), nb, MPI_BYTE, all.data(), cnt.data(), disp.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime(); tw_gather += t1 - t0;

    std::vector<RumiBack> back;
    std::vector<int> bcnt(NTask, 0), bdisp(NTask, 0);
    if(ThisTask == 0) {
        std::sort(all.begin(), all.end(), [](const RumiStar &a, const RumiStar &b) {return a.ID < b.ID;});
        const int n = ntot;
        if(getenv("RUMI_COUPLING_CHECK")) {std::vector<double> gx(3*n); std::vector<StarKey> gid(n); for(int k = 0; k < n; k++) {gid[k] = all[k].ID; for(int j = 0; j < 3; j++) {gx[3*k+j] = all[k].pos[j];}} check_coincident("gathered", n, gx.data(), NULL, gid);}
        const integertime ti_step = all[0].ti_step;
        for(const RumiStar &s : all) {if(s.ti_step != ti_step) {printf("RUMI++: stars on different bins in one step\n"); endrun(9303);}}
        const double dt = ti_step * All.Timebase_interval, G = rumi_G();
        /* same set: every star has a slot, and the slots without a star are bodies RUMI++ merged away (mass 0) */
        const int nsl = (int)slot_id.size();
        std::vector<int> star_of_slot(nsl, -1);
        int same = (nsl > 0);
        for(int k = 0; same && k < n; k++) {auto it = slot_of.find(all[k].ID); if(it == slot_of.end()) {same = 0;} else {star_of_slot[it->second] = k;}}
        for(int q = 0; same && q < nsl; q++) {if(star_of_slot[q] < 0 && last_m[q] > 0) {same = 0;}}
        int mass_changed = 0;
        if(same) {for(int q = 0; q < nsl; q++) {int k = star_of_slot[q]; if(k >= 0 && all[k].m != last_m[q]) {mass_changed = 1; break;}}}
        /* GIZMO's changes since the last step (gas kicks, box wrap) as differences, in RUMI++ units */
        auto deltas = [&](const RumiStar &s, int k, double *dx, double *dv) {
            double d[3]; for(int j = 0; j < 3; j++) {d[j] = s.pos[j] - last_x[3*k+j];}
            NEAREST_XYZ(d[0], d[1], d[2], 1);
            for(int j = 0; j < 3; j++) {dx[j] = d[j] * UL; dv[j] = (s.vel[j] - last_v[3*k+j]) * UV;}
        };
        if(same && !mass_changed) {
            std::vector<double> dx(3*nsl, 0.0), dv(3*nsl, 0.0);
            for(int q = 0; q < nsl; q++) {int k = star_of_slot[q]; if(k >= 0) {deltas(all[k], q, &dx[3*q], &dv[3*q]);}}
            rumi_lib::shift(dx.data()); rumi_lib::kick(dv.data());
        } else { /* new set: the library's state for the stars it has (plus GIZMO's changes), GIZMO's for the others */
            int ns = rumi_lib::n_slots();
            std::vector<double> lm(ns), lx(3*ns), lv(3*ns);
            if(ns > 0) {rumi_lib::get_state(lm.data(), lx.data(), lv.data());}
            std::vector<long long> id(n); std::vector<double> m(n), x(3*n), v(3*n), r(n); std::vector<int> kw(n);
            for(int k = 0; k < n; k++) {
                const RumiStar &s = all[k]; id[k] = (long long)s.ID.id; kw[k] = s.kw; r[k] = s.r_rsun;
                auto it = slot_of.find(s.ID);
                if(it != slot_of.end() && lm[it->second] > 0) {
                    int q = it->second; double dx[3], dv[3]; deltas(s, q, dx, dv);
                    m[k] = (s.m == last_m[q]) ? lm[q] : s.m * UM;
                    for(int j = 0; j < 3; j++) {x[3*k+j] = lx[3*q+j] + dx[j]; v[3*k+j] = lv[3*q+j] + dv[j];}
                } else {
                    m[k] = s.m * UM; for(int j = 0; j < 3; j++) {x[3*k+j] = s.pos[j] * UL; v[3*k+j] = s.vel[j] * UV;}
                }
            }
            if(rumi_lib::set_particles(n, id.data(), m.data(), x.data(), v.data(), kw.data(), r.data(), G, dt * UT) != 0) {printf("RUMI++: set_particles failed\n"); endrun(9304);}
            slot_id.resize(n); slot_of.clear();
            for(int k = 0; k < n; k++) {slot_id[k] = all[k].ID; slot_of[all[k].ID] = k;}
            last_m.assign(n, 0); last_x.assign(3*n, 0); last_v.assign(3*n, 0); slot_m_r = m;
            for(int k = 0; k < n; k++) {last_m[k] = all[k].m;}
        }
        if(All.RumiReplayDump && !replay_written) { /* the state, G and step of the first advance: a standalone RUMI++ replay */
            char buf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(buf, sizeof(buf), "%s/rumi_replay_ic.txt", All.OutputDir);
            rumi_lib::dump_ic(buf);
            snprintf(buf, sizeof(buf), "%s/rumi_replay.txt", All.OutputDir);
            FILE *f = fopen(buf, "w"); if(f) {fprintf(f, "G_OVERRIDE = %.17g\nDT_MAX_CODE = %.17g\n", G, dt * UT); fclose(f);}
            replay_written = 1;
        }
        if(getenv("RUMI_COUPLING_CHECK")) {int q = rumi_lib::n_slots(); std::vector<double> cm(q), cx(3*q), cv(3*q); rumi_lib::get_state(cm.data(), cx.data(), cv.data()); check_coincident(same && !mass_changed ? "before(same)" : "before(rebuilt)", q, cx.data(), cm.data(), slot_id);}
        if(rumi_lib::advance(dt * UT, G) != 0) {printf("RUMI++: advance failed\n"); endrun(9305);}
        const int ns = (int)slot_id.size();
        std::vector<double> m(ns), x(3*ns), v(3*ns);
        rumi_lib::get_state(m.data(), x.data(), v.data());
        check_coincident("after", ns, x.data(), m.data(), slot_id);
        std::vector<char> m_new(ns, 0); /* RUMI++ changed this mass (merger): GIZMO takes it, else GIZMO keeps its own bits */
        for(int q = 0; q < ns; q++) {
            if(m[q] != slot_m_r[q]) {m_new[q] = 1; slot_m_r[q] = m[q]; last_m[q] = (m[q] > 0) ? m[q] / UM : 0;}
            for(int j = 0; j < 3; j++) {last_x[3*q+j] = x[3*q+j] / UL; last_v[3*q+j] = v[3*q+j] / UV;}
        }
        n_steps++;
        if(All.RumiPrintInterval > 0 && (n_steps % All.RumiPrintInterval == 1 || All.RumiPrintInterval == 1)) {
            double E, Eb; rumi_lib::energy(G, &E, &Eb);
            if(!have_E0) {E0_lib = E + Eb; have_E0 = 1;}
            const rumi_lib::Timers &T = rumi_lib::timers();
            double bk[7]; rumi_lib::books(bk);
            if(FdRumi) {fprintf(FdRumi, "%lld %.10e %d %.16e %.6e %.6e | %.3f %.3f %.3f %.3f %.3f %.3f %.3f | books %.3e %.3e %.3e %.3e %.3e %.3e %.3e mergers %d\n", n_steps, All.Time, n, E, Eb, (E + Eb - E0_lib) / fabs(E0_lib),
                                tw_limit, tw_gather, T.advance, T.rebuild, T.h2d, T.d2h, tw_scatter, bk[0], bk[1], bk[2], bk[3], bk[4], bk[5], bk[6], rumi_lib::n_mergers()); fflush(FdRumi);}
        }
        /* back to the owners, in task order */
        back.resize(n);
        std::vector<int> order(n); for(int k = 0; k < n; k++) {order[k] = k;}
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return all[a].task < all[b].task;});
        for(int q = 0; q < n; q++) {
            int k = order[q], sl = slot_of[all[k].ID]; RumiBack &b = back[q]; b.ID = all[k].ID; b.index = all[k].index; b.m = m_new[sl] ? last_m[sl] : -1;
            for(int j = 0; j < 3; j++) {b.pos[j] = last_x[3*sl+j]; b.vel[j] = last_v[3*sl+j];}
            bcnt[all[k].task] += (int)sizeof(RumiBack);
        }
        int o = 0; for(int t = 0; t < NTask; t++) {bdisp[t] = o; o += bcnt[t];}
    }
    double t2 = MPI_Wtime(); tw_lib += t2 - t1;
    std::vector<RumiBack> got(nl);
    MPI_Scatterv(back.data(), bcnt.data(), bdisp.data(), MPI_BYTE, got.data(), nb / (int)sizeof(RumiStar) * (int)sizeof(RumiBack), MPI_BYTE, 0, MPI_COMM_WORLD);
    int mass_changed_local = 0;
    for(const RumiBack &b : got) {
        int i = b.index;
        if(!(star_key(i) == b.ID)) {printf("RUMI++: index mismatch ID %llu vs %llu on task %d\n", (unsigned long long)P[i].ID, b.ID.id, ThisTask); endrun(9306);}
        if(b.m >= 0 && b.m != P[i].Mass) {P[i].Mass = b.m; mass_changed_local = 1;} /* RUMI++ mergers (0: merged away, removed at the next rearrangement) */
        integertime ti_step = GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin);
        double dt_drift = get_drift_factor(P[i].Ti_current, P[i].Ti_current + ti_step, i, 0);
        Vec3<double> d; for(int j = 0; j < 3; j++) {d[j] = b.pos[j] - P[i].Pos[j];}
        NEAREST_XYZ(d[0], d[1], d[2], 1);
        for(int j = 0; j < 3; j++) {P[i].RumiTargetPos[j] = b.pos[j]; P[i].RumiTrueVel[j] = b.vel[j]; P[i].RumiStartVel[j] = P[i].Vel[j];}
        Vec3<double> vnew;
        if(dt_drift > 0) {for(int j = 0; j < 3; j++) {vnew[j] = d[j] / dt_drift;}}
        else {for(int j = 0; j < 3; j++) {vnew[j] = b.vel[j]; P[i].Pos[j] = b.pos[j];}}
        for(int j = 0; j < 3; j++) {P[i].dp[j] += P[i].Mass * (vnew[j] - P[i].Vel[j]); P[i].Vel[j] = vnew[j];} /* node velocities follow (force_update_tree) */
        P[i].RumiTiStart = P[i].Ti_current; P[i].RumiTiEnd = P[i].Ti_current + ti_step;
        P[i].RumiIntegrated = 1;
    }
    int mc = 0; MPI_Allreduce(&mass_changed_local, &mc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(mc) {TreeReconstructFlag = 1;}
    tw_scatter += MPI_Wtime() - t2;
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


/* around savepositions: stars whose RUMI++ step ends at the output write RUMI++'s position and velocity, stars whose
   step starts there their start velocity, not the chord ones (outputs inside a step stay on the chord); the second
   call swaps back (exactly) */
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
        }
    }
}

#endif /* STARS_INTEGRATED_BY_RUMI */
