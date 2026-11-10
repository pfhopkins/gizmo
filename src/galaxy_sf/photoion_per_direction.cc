/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/photoion_per_direction.cc -- host side of the per-direction photoionisation (Specs in photoion_per_direction.h).
 * Each ionising single star solves on its own active steps, as gizmo2017 (step at most PhotoionMaxTimestep, no floor). At a
 * solve every solving source finds its fronts alone (search sphere grown or refined as stella_dev) and claims its cells;
 * then, in rounds, sources share the bills of cells claimed by several (fronts only advance), until no front moves. Sources
 * not solving now keep their last fronts (front table) and count as claims; those fronts do not move. Cells stay ionised
 * until the next multiple of the cap step, where every source solves, so an inactive source's region cannot drop out;
 * a solving source releases cells it left. Reference: stella_dev galaxy_sf/resolvedism_photoion.cc. */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "single_star_record.h"
#include "photoion_per_direction.h"

#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
#if !defined(GALSF_SINGLE_STARS_FROM_IMF) || !defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES)
#error "GALSF_FB_PHOTOIONIZATION_PER_DIRECTION: the sources are single stars (GALSF_SINGLE_STARS_FROM_IMF, GALSF_SINGLE_STARS_EVOLUTION_TABLES)"
#endif
#ifdef GALSF_FB_FIRE_RT_HIIHEATING
#error "GALSF_FB_PHOTOIONIZATION_PER_DIRECTION replaces the HII-region loop (GALSF_FB_FIRE_RT_HIIHEATING): switch on one of them"
#endif

#define PI_MAX_PASSES 20
#define PI_ALPHA_B 2.6e-13   /* case B at 1e4 K [cm^3/s] */
#define PI_TOLERANCE 0.01    /* a pixel with less than this fraction of its budget unspent counts as converged */

/* A_c by local gas index: grown when needed, zero between solves (the drain empties what was touched) */
static long long *pi_claim = nullptr;
static int pi_claim_cap = 0;
static long long *pi_claim_sums(int n_gas)
{
    const int n = (n_gas > 0) ? n_gas : 1;
    if(n > pi_claim_cap) {
        long long *q = (long long *) gizmo_gpu_alloc_shared((size_t) n * sizeof(long long), "photoion_claim_sums");
        if(!q) {gizmo_request_controlled_stop(7901, "per-direction photoionisation: could not grow the claim buffer", __FILE__, __LINE__, __FUNCTION__); return nullptr;}
        Kokkos::fence();
        std::memset(q, 0, (size_t) n * sizeof(long long));
        if(pi_claim) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pi_claim);}
        pi_claim = q; pi_claim_cap = n;
    }
    return pi_claim;
}

static int pi_any_active = -1;  /* any ionising source active on this step, from the sync point (-1: not known yet) */
static long long pi_calls = 0;  /* this module's reductions and runner calls since the last log row */
void photoion_set_any_active(int any) {pi_any_active = any;}

/* grows a shared array, keeping its first n_keep entries; 0 if out of memory */
template <typename T> static int pi_grow(T *&p, int& cap, int n, int n_keep, const char *name)
{
    if(n <= cap) {return 1;}
    T *q = (T *) gizmo_gpu_alloc_shared((size_t) n * sizeof(T), name);
    if(!q) {gizmo_request_controlled_stop(7904, "per-direction photoionisation: could not grow the front table", __FILE__, __LINE__, __FUNCTION__); return 0;}
    Kokkos::fence();
    if(p) {std::memcpy((void *)q, (void *)p, (size_t) n_keep * sizeof(T)); Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(p);}
    p = q; cap = n;
    return 1;
}

/* the front table: last fronts of the live sources, the same on every rank */
static PiFrontRecord *pi_tab = nullptr;
static int pi_tab_n = 0, pi_tab_cap = 0;
static int *pi_flist = nullptr; /* the partners of each source of a solve */
static int pi_flist_cap = 0;
static int pi_tab_row(MyIDType id)
{
    PiFrontRecord *e = std::lower_bound(pi_tab, pi_tab + pi_tab_n, id, [](const PiFrontRecord& a, MyIDType b) {return a.id < b;});
    return (e < pi_tab + pi_tab_n && e->id == id) ? (int)(e - pi_tab) : -1;
}
int photoion_front_table_size(void) {return pi_tab_n;}
size_t photoion_front_record_bytes(void) {return sizeof(PiFrontRecord);}
void *photoion_front_table(int n) {if(n > 0 && !pi_grow(pi_tab, pi_tab_cap, n, 0, "photoion_front_table")) {n = 0;} pi_tab_n = n; return pi_tab;} /* for the restart files */

/* histogram pool: partial histograms of the sources whose cells live on this rank, by slot; grown when a pass fills it */
static int *pi_pool_slot = nullptr, *pi_pool_g = nullptr, *pi_pool_ctl = nullptr; /* ctl: slots used, full */
static long long *pi_pool_cost = nullptr;
static double *pi_pool_rmin = nullptr;
static int pi_pool_cap = 0, pi_pool_ng = 0;
static void pi_pool_reset(int n_g, int cap)
{
    if(cap > pi_pool_cap || n_g > pi_pool_ng || !pi_pool_ctl) {
        for(void *q : {(void *)pi_pool_slot, (void *)pi_pool_g, (void *)pi_pool_ctl, (void *)pi_pool_cost, (void *)pi_pool_rmin}) {if(q) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(q);}}
        pi_pool_cap = DMAX(cap, pi_pool_cap); pi_pool_ng = DMAX(n_g, pi_pool_ng);
        size_t nb = (size_t) pi_pool_cap * PI_NPIX * PI_NBIN;
        pi_pool_slot = (int *) gizmo_gpu_alloc_shared((size_t) pi_pool_ng * sizeof(int), "photoion_pool_slot");
        pi_pool_g = (int *) gizmo_gpu_alloc_shared((size_t) pi_pool_cap * sizeof(int), "photoion_pool_g");
        pi_pool_ctl = (int *) gizmo_gpu_alloc_shared(2 * sizeof(int), "photoion_pool_ctl");
        pi_pool_cost = (long long *) gizmo_gpu_alloc_shared(nb * sizeof(long long), "photoion_pool_cost");
        pi_pool_rmin = (double *) gizmo_gpu_alloc_shared(nb * sizeof(double), "photoion_pool_rmin");
        if(!pi_pool_slot || !pi_pool_g || !pi_pool_ctl || !pi_pool_cost || !pi_pool_rmin) {gizmo_request_controlled_stop(7903, "per-direction photoionisation: could not allocate the histogram pool", __FILE__, __LINE__, __FUNCTION__); return;}
        Kokkos::fence();
        for(int g = 0; g < pi_pool_ng; g++) {pi_pool_slot[g] = -1;}
        for(size_t k = 0; k < nb; k++) {pi_pool_cost[k] = 0; pi_pool_rmin[k] = PI_RMIN_EMPTY;}
        pi_pool_ctl[0] = pi_pool_ctl[1] = 0;
    }
}
/* empties the slots a pass used (all-zero between passes) */
static void pi_pool_clear(void)
{
    int n = DMIN(pi_pool_ctl[0], pi_pool_cap);
    for(int s = 0; s < n; s++) {
        if(pi_pool_g[s] >= 0) {pi_pool_slot[pi_pool_g[s]] = -1;}
        pi_pool_g[s] = -1;
        for(int k = 0; k < PI_NPIX * PI_NBIN; k++) {pi_pool_cost[(size_t)s * PI_NPIX * PI_NBIN + k] = 0; pi_pool_rmin[(size_t)s * PI_NPIX * PI_NBIN + k] = PI_RMIN_EMPTY;}
    }
    pi_pool_ctl[0] = pi_pool_ctl[1] = 0;
}

/* ionising photon rate of source i [1/s] */
static double photoion_source_rate(int i)
{
    if(P[i].StarIdx <= 0 || P[i].Mass <= 0) {return 0;}
#ifdef TEST_PHOTOIONIZATION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestPhotoionFixedRate > 0) {return (single_star_ionizing_onset_yr(i) >= 0) ? All.TestPhotoionFixedRate : 0;} /* every star of at least 8 Msun, at any age */
#endif
    return All.HIIRegion_fLum_Coupled * single_star_ionizing_photon_rate(i);
}

/* the integer times of the cells and the table, when TimeMax grows on a restart */
void photoion_halve_timeline(void)
{
    for(int i = 0; i < N_gas; i++) {CellP[i].PhotoionizedUntil /= 2;}
    for(int t = 0; t < pi_tab_n; t++) {pi_tab[t].ti /= 2;}
}

/* 1 if an ionising source on this rank is active on this step (rides the reduction at the sync point) */
int photoion_active_sources_local(void)
{
    for(int n = 0; n < TIMEBINS; n++) {
        if(!TimeBinActive[n]) {continue;}
        for(int i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i]) {if(P[i].StarIdx > 0 && photoion_source_rate(i) > 0) {return 1;}}
    }
    return 0;
}

/* the cap bin: the largest bin whose step is within PhotoionMaxTimestep and MaxSizeTimestep. No source steps longer, so
   all live sources are active at each multiple of its step */
static int photoion_timebin(void)
{
    double dt = DMIN(All.PhotoionMaxTimestep / UNIT_TIME_IN_YR, All.MaxSizeTimestep) * All.cf_hubble_a;
    if(dt > 0.5 * TIMEBASE * All.Timebase_interval) {dt = 0.5 * TIMEBASE * All.Timebase_interval;} /* before the integer conversion: a run shorter than the step overflows it */
    integertime ti = (integertime)(dt / All.Timebase_interval);
    int b = 1;
    while(b + 1 < TIMEBINS && ((integertime)1 << (b + 1)) <= ti) {b++;}
    return b;
}

/* largest timestep of single star p [code]: an ionising source at most PhotoionMaxTimestep (it solves on its active
   steps; a cap, can be smaller, no floor); a star that will ionise steps onto its onset age */
double photoion_max_timestep(int p)
{
    double t_on = single_star_ionizing_onset_yr(p);
    if(t_on < 0) {return 0;}
    if(photoion_source_rate(p) > 0) {
        double dt = All.PhotoionMaxTimestep / UNIT_TIME_IN_YR;
#ifdef TEST_PHOTOIONIZATION_ONE_STAR_ON_QUARTER_STEP
        if(All.TestPhotoionQuarterStepStarID > 0 && P[p].ID == (MyIDType) All.TestPhotoionQuarterStepStarID) {dt *= 0.25;}
#endif
        return dt;
    }
    double age = evaluate_stellar_age_Gyr(p) * 1.e9;
    if(age < t_on) {return DMAX(t_on - age, 10.) / UNIT_TIME_IN_YR;}
    return 0;
}

/* ---- Spec hooks ---- */

bool PiSpecBase::is_active(int i) {return (P[i].StarIdx > 0);}

double PiSpecBase::search_radius(const neighbor_loop_args& args, int active_slot, int)
{
    return nlr_aux<PiHistSpec>(args)->host_src[active_slot].R;
}

PiCallScalars PiSpecBase::populate_call_scalars(const neighbor_loop_args& args)
{
    PiCallScalars s;
    s.common = nlr_common_scalars_from_all();
    s.bill_fac = PI_ALPHA_B * HYDROGEN_MASSFRAC * HYDROGEN_MASSFRAC / (PROTONMASS_CGS * PROTONMASS_CGS) * All.cf_a3inv * UNIT_DENSITY_IN_CGS * UNIT_MASS_IN_CGS;
#ifdef GALSF_FB_PHOTOIONIZATION_RECOMBINATION_AT_CELL_TEMPERATURE
    for(int k = 0; k < PI_ALPHA_NT; k++) {s.alpha_ratio[k] = chem_case_b_rate(pow(10., 4. + 0.1 * k)) / chem_case_b_rate(1.e4);}
#endif
    Aux *aux = nlr_aux<PiHistSpec>(args);
    s.stamp = aux ? aux->stamp : 0; s.share = aux ? aux->share : 0;
    s.claim = pi_claim; s.n_claim = DMIN(N_gas, pi_claim_cap);
    s.pool_slot = pi_pool_slot; s.pool_g = pi_pool_g; s.pool_n = pi_pool_ctl; s.pool_full = pi_pool_ctl ? pi_pool_ctl + 1 : nullptr;
    s.pool_cost = pi_pool_cost; s.pool_rmin = pi_pool_rmin; s.pool_cap = pi_pool_cap;
    s.tab = pi_tab; s.flist = pi_flist;
    return s;
}

void PiSpecBase::populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx)
{
    Aux *aux = nlr_aux<PiHistSpec>(args);
    const int N = args.num_active;
    ctx.per_active_local = nullptr;
    if(N <= 0 || aux == nullptr || aux->host_src == nullptr) {return;}
    PiSourceIn *uvm = (PiSourceIn *) gizmo_gpu_alloc_shared((size_t) N * sizeof(PiSourceIn), NULL);
    if(!uvm) {
        ctx.populate_failed = 1;
        gizmo_request_controlled_stop(7902, "per-direction photoionisation: could not stage the sources", __FILE__, __LINE__, __FUNCTION__);
        return;
    }
    std::memcpy(uvm, aux->host_src, (size_t) N * sizeof(PiSourceIn));
    ctx.per_active_local = uvm;
}

void PiSpecBase::cleanup_device_context(const neighbor_loop_args&, DeviceContext& ctx)
{
    if(ctx.per_active_local) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(const_cast<PiSourceIn *>(ctx.per_active_local)); ctx.per_active_local = nullptr;}
}

void PiStampSpec::apply_active_writeback(const neighbor_loop_args& args, int slot, int, const AccumData& a)
{
    ((PiLedgerOut *) nlr_aux<PiStampSpec>(args)->out)[slot] = a;
}

template <typename Spec>
static void pi_run(std::vector<int>& list, std::vector<PiSourceIn>& in, PiSpecBase::Aux& aux)
{
    aux.host_src = in.empty() ? nullptr : in.data();
    neighbor_loop_args args = nlr_default_args();
    args.active_list = list.empty() ? nullptr : list.data(); args.num_active = (int) list.size(); args.aux = &aux;
    pi_calls++; run_neighbor_loop<Spec>(args);
}

/* ---- the front of each pixel ---- */

struct pi_src {int i, g, prev, f0, nf; double Q, R; int refine, open; double front[PI_NPIX], mark[PI_NPIX], held[PI_NPIX];};

/* per pixel, where the summed bills cross the budget (fractional front, for diagnostics and refinement) and the marking
   front, rounded up through the innermost cell of the crossing bin (stella_dev v6). Returns 1 if the source searches
   again (sphere grown, or refined if allow_shrink), 0 if its fronts are final. */
static int pi_analyse(pi_src& s, const PiHistOut& h, int allow_iter, int allow_shrink, double max_r)
{
    const long long S = (long long) PI_BIN_UNIT;
    double R = s.R, dr = R / PI_NBIN, front_max = 0, cum_min = MAX_REAL_NUMBER;
    int n_uncrossed = 0;
    for(int k = 0; k < PI_NPIX; k++) {
        long long cum = 0; double f = -1, fm = -1;
        for(int b = 0; b < PI_NBIN; b++) {
            long long c = h.cost[k * PI_NBIN + b];
            if(c <= 0) {continue;}
            if(cum + c >= S) {
                f = ((double)b + (double)(S - cum) / (double)c) * dr;
                double rmin_b = h.rmin[k * PI_NBIN + b];
                fm = (rmin_b < PI_RMIN_EMPTY && rmin_b >= f) ? rmin_b * (1.0 + 1.0e-9) : f;
                break;
            }
            cum += c;
        }
        if(f < 0) { /* budget not used up inside the sphere */
            if(cum <= 0) {f = 0; n_uncrossed++; cum_min = 0;}
            else if((double)(S - cum) <= PI_TOLERANCE * (double)S) {f = R;}
            else {f = R; n_uncrossed++; if((double)cum < cum_min) {cum_min = (double)cum;}}
            fm = f;
        }
        s.front[k] = f; s.mark[k] = fm; if(f > front_max) {front_max = f;}
    }
    if(n_uncrossed > 0 && allow_iter && !(max_r > 0 && R >= 0.999 * max_r)) { /* grow from the worst pixel, [1.25, 2.5] */
        double grow = 2.0;
        if(cum_min > 0 && cum_min < MAX_REAL_NUMBER) {grow = DMIN(DMAX(pow((double)S / cum_min, 1.0/3.0), 1.25), 2.5);}
        R *= grow; if(max_r > 0 && R > max_r) {R = max_r;}
        s.R = R; return 1;
    }
    if(n_uncrossed == 0 && allow_iter && allow_shrink && s.refine < 2 && front_max > 0 && front_max < 0.5 * R) {s.R = 1.3 * front_max; s.refine++; return 1;}
    return 0;
}

static PiSourceIn pi_source_in(const pi_src& s)
{
    PiSourceIn x; std::memset(&x, 0, sizeof(x));
    for(int k = 0; k < 3; k++) {x.Pos[k] = P[s.i].Pos[k];}
    x.R = s.R; x.S_pix = s.Q / PI_NPIX;
    for(int k = 0; k < PI_NPIX; k++) {x.held[k] = s.held[k]; x.mark[k] = s.mark[k];}
    x.g = s.g; x.prev = s.prev; x.f0 = s.f0; x.nf = s.nf;
    return x;
}

/* histograms of the sources S[which[..]] (global offsets off[rank]): the cells' ranks fill their pools, then send the
   non-empty bins to the sources' ranks (Alltoall, Alltoallv); integer sums and minima, so the order of arrival is free */
struct pi_bin {int g, k; long long cost; double rmin;};
static void pi_histograms(std::vector<pi_src>& S, const std::vector<int>& which, int share, const std::vector<long long>& off, std::vector<PiHistOut>& H)
{
    std::vector<int> list; std::vector<PiSourceIn> in;
    for(int a : which) {list.push_back(S[a].i); in.push_back(pi_source_in(S[a]));}
    for(int full = 1; full; ) {
        PiSpecBase::Aux aux = {nullptr, nullptr, 0, share};
        pi_run<PiHistSpec>(list, in, aux);
        int need[2] = {pi_pool_ctl[1], pi_pool_ctl[0]}, need_all[2];
        pi_calls++; MPI_Allreduce(need, need_all, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        full = need_all[0];
        if(full) {pi_pool_clear(); pi_pool_reset((int) off[NTask], 2 * need_all[1]);} /* grow and repeat the pass */
    }
    std::vector<std::vector<pi_bin>> out(NTask);
    for(int s = 0; s < DMIN(pi_pool_ctl[0], pi_pool_cap); s++) {
        int g = pi_pool_g[s]; if(g < 0) {continue;}
        int dest = (int)(std::upper_bound(off.begin(), off.end(), (long long) g) - off.begin()) - 1;
        for(int k = 0; k < PI_NPIX * PI_NBIN; k++) {
            size_t q = (size_t)s * PI_NPIX * PI_NBIN + k;
            if(pi_pool_cost[q] > 0 || pi_pool_rmin[q] < PI_RMIN_EMPTY) {out[dest].push_back({g, k, pi_pool_cost[q], pi_pool_rmin[q]});}
        }
    }
    pi_pool_clear();
    std::vector<int> sc(NTask), rc(NTask), sd(NTask), rd(NTask);
    for(int t = 0; t < NTask; t++) {sc[t] = (int)(out[t].size() * sizeof(pi_bin));}
    pi_calls++; MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<char> sbuf, rbuf; int ns = 0, nr = 0;
    for(int t = 0; t < NTask; t++) {sd[t] = ns; ns += sc[t]; rd[t] = nr; nr += rc[t];}
    sbuf.resize(DMAX(ns, 1)); rbuf.resize(DMAX(nr, 1));
    for(int t = 0; t < NTask; t++) {if(sc[t]) {std::memcpy(&sbuf[sd[t]], out[t].data(), sc[t]);}}
    pi_calls++; MPI_Alltoallv(sbuf.data(), sc.data(), sd.data(), MPI_BYTE, rbuf.data(), rc.data(), rd.data(), MPI_BYTE, MPI_COMM_WORLD);
    std::vector<int> pos(S.size(), -1);
    for(size_t a = 0; a < which.size(); a++) {pos[S[which[a]].g - off[ThisTask]] = (int) a;}
    H.assign(which.size(), PiHistOut());
    for(PiHistOut& h : H) {for(int k = 0; k < PI_NPIX * PI_NBIN; k++) {h.cost[k] = 0; h.rmin[k] = PI_RMIN_EMPTY;}}
    const pi_bin *rb = (const pi_bin *) rbuf.data();
    for(size_t m = 0; m < nr / sizeof(pi_bin); m++) {
        int a = pos[rb[m].g - off[ThisTask]]; if(a < 0) {continue;}
        H[a].cost[rb[m].k] = pi_sat_add(H[a].cost[rb[m].k], rb[m].cost);
        if(rb[m].rmin < H[a].rmin[rb[m].k]) {H[a].rmin[rb[m].k] = rb[m].rmin;}
    }
}

/* claim the cells between each source's held and marking fronts; they are held from now on */
static void pi_claim_all(std::vector<pi_src>& S)
{
    std::vector<int> list; std::vector<PiSourceIn> in;
    for(pi_src& s : S) {list.push_back(s.i); in.push_back(pi_source_in(s));}
    PiSpecBase::Aux aux = {nullptr, nullptr, 0, 0};
    pi_run<PiClaimSpec>(list, in, aux);
    for(pi_src& s : S) {for(int k = 0; k < PI_NPIX; k++) {s.held[k] = s.mark[k];}}
}

/* a table row counts until the first multiple of the cap step after its solve (where the source solves again, or is gone) */
static integertime pi_row_end(const PiFrontRecord& e, integertime dti) {return (e.ti / dti + 1) * dti;}

struct pi_sphere {MyIDType id; Vec3<double> Pos; double R;};
static void pi_counts_in_bytes(const std::vector<long long>& cnt, const std::vector<long long>& off, size_t rec, std::vector<int>& n, std::vector<int>& d)
{
    n.resize(NTask); d.resize(NTask);
    for(int t = 0; t < NTask; t++) {n[t] = (int)(cnt[t] * rec); d[t] = (int)(off[t] * rec);}
}

/* the partners of each solving source: live table rows not solving now whose fronts can reach it. Every rank computes the
   same list (it travels as offsets in the source records). Returns the number of partner links. */
static int pi_find_partners(std::vector<pi_src>& S, const std::vector<long long>& cnt, const std::vector<long long>& off, integertime dti)
{
    const int n_tot = (int) off[NTask];
    std::vector<pi_sphere> mine(S.size()), all(n_tot);
    for(size_t a = 0; a < S.size(); a++) {
        pi_sphere& m = mine[a]; m.id = P[S[a].i].ID; m.R = 2.0 * S[a].R; /* rounds can grow a sphere */
        for(int k = 0; k < 3; k++) {m.Pos[k] = P[S[a].i].Pos[k];}
        if(S[a].prev >= 0) {Vec3<double> d; for(int k = 0; k < 3; k++) {d[k] = m.Pos[k] - pi_tab[S[a].prev].Pos[k];} m.R = DMAX(m.R, sqrt(d.norm_sq()) + pi_tab[S[a].prev].R);} /* and the cells it may release */
    }
    std::vector<int> nb, db; pi_counts_in_bytes(cnt, off, sizeof(pi_sphere), nb, db);
    pi_calls++; MPI_Allgatherv(mine.data(), (int)(mine.size() * sizeof(pi_sphere)), MPI_BYTE, all.data(), nb.data(), db.data(), MPI_BYTE, MPI_COMM_WORLD);
    std::vector<char> solving(pi_tab_n, 0);
    for(const pi_sphere& s : all) {int r = pi_tab_row(s.id); if(r >= 0) {solving[r] = 1;}}
    std::vector<int> fl, f0(n_tot + 1, 0);
    for(int g = 0; g < n_tot; g++) {
        f0[g] = (int) fl.size();
        for(int t = 0; t < pi_tab_n; t++) {
            if(solving[t] || pi_row_end(pi_tab[t], dti) <= All.Ti_Current) {continue;}
            Vec3<double> d; for(int k = 0; k < 3; k++) {d[k] = all[g].Pos[k] - pi_tab[t].Pos[k];}
            nearest_xyz(d);
            double reach = all[g].R + pi_tab[t].R;
            if(d.norm_sq() < reach * reach) {fl.push_back(t);}
        }
    }
    f0[n_tot] = (int) fl.size();
    if(!pi_grow(pi_flist, pi_flist_cap, DMAX(f0[n_tot], 1), 0, "photoion_partners")) {return 0;}
    if(!fl.empty()) {std::memcpy(pi_flist, fl.data(), fl.size() * sizeof(int));}
    for(pi_src& s : S) {s.f0 = f0[s.g]; s.nf = f0[s.g + 1] - f0[s.g];}
    return f0[n_tot];
}

/* the solved fronts go into every rank's table; rows that reached their end go */
static void pi_table_update(const std::vector<pi_src>& S, const std::vector<PiLedgerOut>& L, const std::vector<long long>& cnt, const std::vector<long long>& off, integertime dti, double *ledger_ext)
{
    std::vector<PiFrontRecord> mine(S.size()), all(off[NTask]);
    for(size_t a = 0; a < S.size(); a++) {
        PiFrontRecord& e = mine[a]; const pi_src& s = S[a];
        e.id = P[s.i].ID; e.ti = All.Ti_Current; e.R = 0; e.S_pix = s.Q / PI_NPIX; e.ledger = L[a].bill / s.Q;
        for(int k = 0; k < 3; k++) {e.Pos[k] = P[s.i].Pos[k];}
        for(int k = 0; k < PI_NPIX; k++) {e.mark[k] = s.mark[k]; e.R = DMAX(e.R, 1.0001 * s.mark[k]);} /* the reach of its fronts */
    }
    std::vector<int> nb, db; pi_counts_in_bytes(cnt, off, sizeof(PiFrontRecord), nb, db);
    pi_calls++; MPI_Allgatherv(mine.data(), (int)(mine.size() * sizeof(PiFrontRecord)), MPI_BYTE, all.data(), nb.data(), db.data(), MPI_BYTE, MPI_COMM_WORLD);
    ledger_ext[0] = MAX_REAL_NUMBER; ledger_ext[1] = 0;
    for(const PiFrontRecord& e : all) {ledger_ext[0] = DMIN(ledger_ext[0], e.ledger); ledger_ext[1] = DMAX(ledger_ext[1], e.ledger);}
    auto by_id = [](const PiFrontRecord& a, const PiFrontRecord& b) {return a.id < b.id;};
    std::sort(all.begin(), all.end(), by_id);
    std::vector<PiFrontRecord> keep;
    for(int t = 0; t < pi_tab_n; t++) {
        if(pi_row_end(pi_tab[t], dti) > All.Ti_Current && !std::binary_search(all.begin(), all.end(), pi_tab[t], by_id)) {keep.push_back(pi_tab[t]);}
    }
    all.insert(all.end(), keep.begin(), keep.end());
    std::sort(all.begin(), all.end(), by_id);
    pi_tab_n = (int) all.size(); /* the capacity is reserved at the start of the solve */
    if(pi_tab_n > 0) {std::memcpy((void *)pi_tab, (void *)all.data(), (size_t) pi_tab_n * sizeof(PiFrontRecord));}
}

/* solve for the sources active on this step. Collectives: the source count; per pass the runner call and one reduction;
   off the cap step the partner spheres; claims; per sharing round the runner calls and one reduction; stamp, drain; the
   table; the log row. A step without an active source returns here (pi_any_active, from the sync point). */
void photoionization_per_direction(void)
{
    if(All.HIIRegion_fLum_Coupled <= 0 || All.Time <= 0 || pi_any_active == 0) {return;} /* quiet: no work, no collective */
    const int b_pi = photoion_timebin();
    const integertime dti = (integertime)1 << b_pi;
    const double t_start = MPI_Wtime();
    const double max_r = All.PhotoionMaxSearchRadius / (UNIT_LENGTH_IN_KPC * All.cf_atime);
    std::vector<pi_src> S;
    for(int i : ActiveParticleList) {
        if(P[i].StarIdx <= 0) {continue;}
        double Q = photoion_source_rate(i);
        if(Q <= 0) {continue;}
        pi_src s; std::memset(&s, 0, sizeof(s)); s.i = i; s.Q = Q; s.open = 1;
        s.prev = pi_tab_row(P[i].ID);
        if(s.prev >= 0 && pi_row_end(pi_tab[s.prev], dti) < All.Ti_Current) {s.prev = -1;} /* its cells expired */
        for(int k = 0; k < PI_NPIX; k++) {s.held[k] = -1;}
        double nH = (P[i].DensityAroundParticle > 0) ? HYDROGEN_MASSFRAC * P[i].DensityAroundParticle * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS : 1.0;
        double R = pow(3.0 * Q / (4.0 * M_PI * PI_ALPHA_B * nH * nH), 1.0/3.0) / (UNIT_LENGTH_IN_CGS * All.cf_atime); /* the mean-density guess */
        R = DMAX(R, P[i].KernelRadius); if(R <= 0) {R = All.ForceSoftening[P[i].Type];}
        if(max_r > 0 && R > max_r) {R = max_r;}
        s.R = R; S.push_back(s);
    }
    long long n_loc = (long long) S.size(), n_tot = 0;
    std::vector<long long> cnt(NTask), off(NTask + 1, 0);
    pi_calls++; MPI_Allgather(&n_loc, 1, MPI_LONG_LONG, cnt.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD); /* the global source indices */
    for(int t = 0; t < NTask; t++) {off[t + 1] = off[t] + cnt[t];}
    n_tot = off[NTask];
    for(size_t a = 0; a < S.size(); a++) {S[a].g = (int)(off[ThisTask] + (long long) a);}
    if(n_tot == 0) {return;}
    int no_buffer = (pi_claim_sums(N_gas) == nullptr) || !pi_grow(pi_tab, pi_tab_cap, pi_tab_n + (int) n_tot, pi_tab_n, "photoion_front_table"), any_no_buffer = 0;
    pi_calls++; MPI_Allreduce(&no_buffer, &any_no_buffer, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(any_no_buffer) {return;}
    pi_pool_reset((int) n_tot, DMAX(pi_pool_cap, 16));
    PRINT_STATUS(" ..per-direction photoionisation around %lld single stars", n_tot);

    /* fronts */
    int passes = 0, any_open = 1;
    for(; passes < PI_MAX_PASSES && any_open; passes++) {
        std::vector<int> which; std::vector<PiHistOut> H;
        for(size_t a = 0; a < S.size(); a++) {if(S[a].open) {which.push_back((int)a);}}
        pi_histograms(S, which, 0, off, H);
        int n_open = 0;
        for(size_t a = 0; a < which.size(); a++) {
            pi_src& s = S[which[a]];
            s.open = pi_analyse(s, H[a], passes < PI_MAX_PASSES - 1, 1, max_r);
            n_open += s.open;
        }
        pi_calls++; MPI_Allreduce(&n_open, &any_open, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    /* sharing rounds: held cells bill q_sc / A_c, others q_sc / (A_c + q_sc); bills only fall, so fronts only advance.
       Unspent photons escape: they are not handed to another source. */
    int n_links = (All.Ti_Current % dti != 0 && pi_tab_n > 0) ? pi_find_partners(S, cnt, off, dti) : 0; /* on the cap step all live sources solve */
    pi_claim_all(S);
    int rounds = 0, moved = (n_tot > 1 || n_links > 0);
    for(; rounds < All.PhotoionMaxRounds && moved; rounds++) {
        std::vector<int> which; std::vector<PiHistOut> H;
        for(size_t a = 0; a < S.size(); a++) {which.push_back((int)a);}
        pi_histograms(S, which, 1, off, H);
        int moved_local = 0;
        for(size_t a = 0; a < S.size(); a++) {
            pi_src& s = S[a];
            moved_local |= pi_analyse(s, H[a], 1, 0, max_r); /* the sphere grew */
            for(int k = 0; k < PI_NPIX; k++) {if(s.mark[k] > s.held[k]) {moved_local = 1;} else {s.mark[k] = s.held[k];}}
        }
        pi_claim_all(S);
        pi_calls++; MPI_Allreduce(&moved_local, &moved, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    /* stamp until the next multiple of the cap step: each source solves by then, so the regions of inactive ones stay; a
       source's cells it left are released; drain */
    const integertime stamp = (All.Ti_Current / dti + 1) * dti;
    std::vector<int> list; std::vector<PiSourceIn> in;
    for(pi_src& s : S) {
        if(s.prev >= 0) {const PiFrontRecord& e = pi_tab[s.prev]; Vec3<double> d; for(int k = 0; k < 3; k++) {d[k] = (double)P[s.i].Pos[k] - e.Pos[k];} nearest_xyz(d); s.R = DMAX(s.R, sqrt(d.norm_sq()) + e.R);} /* reach its old cells */
        list.push_back(s.i); in.push_back(pi_source_in(s));
    }
    std::vector<PiLedgerOut> L(S.size() + 1);
    PiSpecBase::Aux aux = {nullptr, L.data(), stamp, 0};
    pi_run<PiStampSpec>(list, in, aux);
    pi_run<PiDrainSpec>(list, in, aux);
    double ledger_ext[2];
    pi_table_update(S, L, cnt, off, dti, ledger_ext);

    /* log row (HIIheating.txt): solving sources, Q/X, their cells, mass and mean distance (a shared cell by share), recombination / Q */
    double loc[5] = {0, 0, 0, 0, 0}, tot[5], ext[3] = {0, -MAX_REAL_NUMBER, 0}, ext_tot[3]; /* front max, -(front min), wall time */
    for(size_t a = 0; a < S.size(); a++) {
        loc[0] += S[a].Q; loc[1] += L[a].bill; loc[2] += L[a].n_new; loc[3] += L[a].m_new; loc[4] += L[a].mr_new;
        for(int k = 0; k < PI_NPIX; k++) {ext[0] = DMAX(ext[0], S[a].front[k]); ext[1] = DMAX(ext[1], -S[a].front[k]);}
    }
    pi_calls++; MPI_Reduce(loc, tot, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    ext[2] = MPI_Wtime() - t_start;
    pi_calls++; MPI_Reduce(ext, ext_tot, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        double to_kpc = All.cf_atime * UNIT_LENGTH_IN_KPC;
        fprintf(FdHIIHeating, "%.16g %lld %g %g %g %g %.8g %d %d %.10g %.10g %lld %.4g %.6g %.6g\n", All.Time, n_tot, tot[0] / HYDROGEN_MASSFRAC, tot[2], tot[3] * UNIT_MASS_IN_SOLAR,
                (tot[3] > 0) ? tot[4] / tot[3] * to_kpc : 0., (tot[0] > 0) ? tot[1] / tot[0] : 0., rounds, passes, -ext_tot[1] * to_kpc, ext_tot[0] * to_kpc, pi_calls, ext_tot[2], ledger_ext[0], ledger_ext[1]);
        fflush(FdHIIHeating);
    }
    pi_calls = 0;
}

#endif /* GALSF_FB_PHOTOIONIZATION_PER_DIRECTION */
