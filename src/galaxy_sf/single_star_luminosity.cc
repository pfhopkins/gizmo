/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Band luminosities of IMF-sampled single stars from the stellar-evolution tables at the star's current age, in the
   disjoint bands PE 8-13.6, LW 11.2-13.6 (inside PE), 6-8, NUV 3.444-6, OPT 0.4133-3.444 eV: as gravity-tree radiation
   sources (GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) and for the radiation-pressure budget at the star
   (GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS). The lookups are the KOKKOS_INLINE_FUNCTIONs of single_star_luminosity_functions.h; the
   functions here call them with the host tables and records. */
#include <math.h>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h: a Kokkos TU (the light stages are kernels) */
#include "../declarations/allvars.h"
#include "../core/proto.h"
/* the kernels here evaluate the device source cores (the star's age, rt_get_source_luminosity): opened before their headers */
#ifdef GRAVTREE_SOURCE_LAZY_SUPPORTED
#define GRAVTREE_SOURCE_DEVICE_TU
#elif defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES)
#error "single-star luminosity kernels need the device source cores (GRAVTREE_SOURCE_LAZY_SUPPORTED; not with GR_TABULATED_COSMOLOGY*)"
#endif
#include "single_star_record.h"
#include "stellar_evolution_tables.h"
#include "single_star_luminosity_functions.h"
#include "single_star_checks.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../system/loop_routes.h"   /* OUTPUT_LOOP_ROUTES */
#if defined(GALSF_SINGLE_STARS_LIGHT_REFRESHES_TREE) || defined(GALSF_SINGLE_STARS_LIGHT_UPDATES_NODE_COPIES)
/* the light kernels call rt_get_source_luminosity: its device bodies are opened here, as gravity/gpu_gravtree.cc does */
#include "../radiation/rt_functions.h"
#endif
#ifdef SGCHEM
#include "../cooling/sgchem/sgchem_dust_to_gas.h"
#endif

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS) || defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS)

/* L[SINGLE_STAR_BAND_*] [erg/s] and the bolometric luminosity (single_star_band_luminosities_core) */
int single_star_band_luminosities_cgs(int i, double *L, double *L_bol)
{
    const int on = single_star_band_luminosities_core(stellar_tables_host_view(), SINGLE_STAR_RECORD_HOST_VIEW(), P, i, L, L_bol);
#if defined(TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS) && defined(GALSF_FB_MECHANICAL) && !defined(TEST_SINGLE_STARS_LUMINOSITY_PINNED) && !defined(TEST_SINGLE_STARS_RADIATION_OFF)
    if(on) {
        int s = single_star_slot(P[i].StarIdx);
        SINGLE_STAR_CHECK("lifetime_consistent", !(SingleStars.event_bits[s] & (SINGLE_STAR_EVENT_SN_DONE | SINGLE_STAR_EVENT_AGB_DONE)), "ID=%llu M=%g radiates after its death", (unsigned long long)P[i].ID, SingleStars.M_drawn[s]);
        SINGLE_STAR_CHECK_RAN("lifetime_consistent", 1);
    }
#endif
    return on;
}
#endif

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS)
/* as rt_get_source_luminosity: fills lum[] [code units], returns 1 for a shining star (pp is P here) */
int single_star_rt_band_luminosities(int i, int mode, double *lum, struct particle_data *pp)
{
    return single_star_rt_band_luminosities_core(stellar_tables_host_view(), SINGLE_STAR_RECORD_HOST_VIEW(), i, mode, lum, pp);
}
#endif

#ifdef SINGLE_STAR_ION_PHOTON_ENERGY_FROM_TABLES
/* mean photon energy of the one ionising bin = sum L_ion / sum Q over the living stars above 8 Msun, from the tables at
   their current ages: the stars' hardness instead of a fixed blackbody's. Emission and the cells' rates and heating read
   the same number, so the photon count is kept. Each star's share is renewed on its own active steps and kept in its
   record; the rank sums are not tied to the particles, so stars may move between ranks, and a step on which every
   particle is active counts anew (stars that merged or left are dropped then) */
static double ion_light_sum[2];
static int ion_light_sum_built = 0;

static void single_star_ion_light(int i, double *L, double *Q)
{
    double b[SINGLE_STAR_NBANDS], L_bol; *L = *Q = 0;
    int s = single_star_slot(P[i].StarIdx);
    if(SingleStars.M_drawn[s] <= 8. || !single_star_band_luminosities_cgs(i, b, &L_bol)) {return;}
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double l = single_star_band_lum_cgs(stellar_log_L_ion_tot(logM, logZ, log10(DMAX(single_star_age_yr(i) - stellar_t_PMS(logM, logZ), 100.))));
    if(l > 0 && b[SINGLE_STAR_BAND_Q] > 0) {*L = l; *Q = b[SINGLE_STAR_BAND_Q];}
}

/* the active stars' shares into the rank sums; 1 on the first call of a run: the bin's energy is not set yet */
int single_star_ion_light_count_active(void)
{
    int first = !ion_light_sum_built;
    if(first)
    {
        ion_light_sum[0] = ion_light_sum[1] = 0; ion_light_sum_built = 1; /* restart files carry the shares, other starts have all particles active */
        for(int s = 0; s < SingleStars.capacity; s++) {if(SingleStars.used[s]) {ion_light_sum[0] += SingleStars.L_ion_counted[s]; ion_light_sum[1] += SingleStars.Q_ion_counted[s];}}
    }
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
    {
        ion_light_sum[0] = ion_light_sum[1] = 0;
        for(int s = 0; s < SingleStars.capacity; s++) {if(SingleStars.used[s]) {SingleStars.L_ion_counted[s] = SingleStars.Q_ion_counted[s] = 0;}}
    }
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 4 || P[i].StarIdx <= 0) {continue;}
        int s = single_star_slot(P[i].StarIdx); double L, Q;
        single_star_ion_light(i, &L, &Q);
        ion_light_sum[0] += L - SingleStars.L_ion_counted[s]; ion_light_sum[1] += Q - SingleStars.Q_ion_counted[s];
        SingleStars.L_ion_counted[s] = L; SingleStars.Q_ion_counted[s] = Q;
    }
    return first;
}

/* collective: one reduction; without living ionising stars the energy stays */
void single_star_ion_photon_energy_update(void)
{
    double sum[2]; MPI_Allreduce(ion_light_sum, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(!(sum[0] > 0 && sum[1] > 0)) {return;}
    double e = DMIN(DMAX(sum[0] / sum[1] / ELECTRONVOLT_IN_ERGS, 13.6), 100.);
    All.rt_nu_eff_eV[RT_FREQ_BIN_H0] = e;
    All.rt_ion_G_HI[RT_FREQ_BIN_H0] = (e - 13.6) * ELECTRONVOLT_IN_ERGS / UNIT_ENERGY_IN_CGS;
}
#endif

#ifdef GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS
/* photon momentum a single star hands to the gas this step [g cm/s]: band luminosities absorbed
   over the kernel column at the star, f_b = 1 - exp(-sigma_b DGR N_H) with sigma = 2.0 (PE + 6-8), 1.3 (NUV), 0.5 (OPT)
   x 1e-21 cm^2, LW not counted (inside PE), capped at L_bol; plus the IR boost tau_IR = 5 DGR Sigma [cm^2/g] on the
   absorbed light. GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE adds 6-8 eV to the NUV term.
   Stars below SINGLE_STAR_RADPRESSURE_MIN_MASS push nothing */
#define SINGLE_STAR_RADPRESSURE_MIN_MASS 2.0 /* [Msun] drawn mass */
double single_star_radpressure_momentum_cgs(int i, double dt_cgs, double *tau_IR)
{
    double L[SINGLE_STAR_NBANDS], L_bol;
    *tau_IR = 0;
    if(P[i].StarIdx <= 0 || SingleStars.M_drawn[single_star_slot(P[i].StarIdx)] < SINGLE_STAR_RADPRESSURE_MIN_MASS) {return 0;}
    if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return 0;}
#ifdef SINGLE_STAR_RADPRESSURE_LIVE_DUST
#ifndef SGCHEM_LIVE_DUST_MILKY_WAY
#define SGCHEM_LIVE_DUST_MILKY_WAY 0.01 /* Milky Way dust mass fraction, without SGCHEM (sgchem_dust_to_gas.h) */
#endif
    double dgr = DMAX(DMIN(SingleStars.dust_ambient[single_star_slot(P[i].StarIdx)] / SGCHEM_LIVE_DUST_MILKY_WAY, 10.), 0.); /* kernel mean of the dust around the star */
#elif defined(SGCHEM)
    double dgr = DMAX(sgchem_dust_to_gas_ratio(P[i].Metallicity[0]), 0.);
#else
    double dgr = DMAX(DMIN(P[i].Metallicity[0] / All.SolarAbundances[0], 10.), 0.);
#endif
    double rho = P[i].DensityAroundParticle * All.cf_a3inv, h = P[i].KernelRadius * All.cf_atime;
    double N_H = rho * UNIT_DENSITY_IN_NHCGS * h * UNIT_LENGTH_IN_CGS, Sigma = rho * UNIT_DENSITY_IN_CGS * h * UNIT_LENGTH_IN_CGS;
    double f_uv = 1.0 - exp(-2.0e-21 * dgr * N_H), f_nuv = 1.0 - exp(-1.3e-21 * dgr * N_H), f_opt = 1.0 - exp(-0.5e-21 * dgr * N_H);
    double L_nuv = L[SINGLE_STAR_BAND_NUV];
#ifdef GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
    L_nuv += L[SINGLE_STAR_BAND_68];
#endif
    double L_abs = DMIN(f_uv * (L[SINGLE_STAR_BAND_PE] + L[SINGLE_STAR_BAND_68]) + f_nuv * L_nuv + f_opt * L[SINGLE_STAR_BAND_OPT], L_bol);
    *tau_IR = 5.0 * dgr * Sigma;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    return L_abs * (*tau_IR) * dt_cgs / C_LIGHT_CGS; /* direct term: per cell, single_star_radpressure_per_cell */
#else
    return L_abs * (1.0 + *tau_IR) * dt_cgs / C_LIGHT_CGS;
#endif
}
#endif

/* GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL: the absorbed sum and the per-cell kick are Kokkos kernels in radfb_rp_loop.cc */

#if defined(GALSF_SINGLE_STARS_LIGHT_REFRESHES_TREE) || defined(GALSF_SINGLE_STARS_LIGHT_UPDATES_NODE_COPIES)
/* The light scans and stores below are Kokkos kernels over the particles, on the device or the host execution space
   (gizmo_gpu_stage_on_device). A star's band light is rt_get_source_luminosity there: the device lookup on the device, the
   host lookup on the host (equal to round-off). The stored light (tree_lum) and the light a scan compares with it come from
   the same space as long as the particle count keeps its side of the offload count. */

/* tree_lum of every star among the first n particles: its light as of now */
template<class Space> static void single_star_tree_lum_fill(int n)
{
    struct particle_data *kp = P; struct gas_cell_data *kc = CellP; struct single_star_tree_lum *tl = SingleStars.tree_lum;
    gizmo_gpu_kernel_launch<Space>("single_star_tree_lum", n, KOKKOS_LAMBDA(int i) {
        if(kp[i].Type != 4 || kp[i].StarIdx <= 0) {return;}
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, kp, kc);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {tl[kp[i].StarIdx - 1].L[b] = (float) lum[b];}
    });
}
static void single_star_tree_lum_fill_all(void)
{
    const int n = NumPart, on_device = gizmo_gpu_stage_on_device(n);
    if(n <= 0) {return;}
    if(on_device) {GIZMO_GPU_ENSURE_ALL_FRESH(); single_star_tree_lum_fill<Kokkos::DefaultExecutionSpace>(n);}
    else {single_star_tree_lum_fill<Kokkos::DefaultHostExecutionSpace>(n);}
    loop_route_count("single_star_tree_lum", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
}
#endif

#ifdef GALSF_SINGLE_STARS_LIGHT_UPDATES_NODE_COPIES /* opt-in (Uli 2026-11-19, GKS-95 option c); default: bounded staleness */
#include "../gravity/forcetree.h"
#include "../gravity/gpu_pseudo_update.h"
#include "../gravity/gravtree_moment_kernel.h"   /* moment_light_centre_weight */
int let_light_wire_of(int R, int key);
int let_light_foreign_node(int sender, int wire);

/* one light change for one node copy: kind 0 = a top-level node (the same index on every rank; applied up its chain to the
   root), kind 1 = the receiver's imported copy of one of the sender's nodes or star leaves (wire index in the sender's import) */
struct single_star_light_change {int target, kind; double dl[N_RT_FREQ_BINS], x[3], v[3];};
static std::vector<int> light_pending; /* local stars whose light the nodes do not hold yet */
static std::vector<float> light_pending_lum; /* their light as the scan evaluated it, N_RT_FREQ_BINS per star */
static double light_stats[4];          /* exchanges, changed stars, bytes sent, seconds (this rank, cumulative) */
static double light_scan_seconds;      /* the scans before every gravity call (this rank, cumulative) */

/* after a whole gravity-tree build: the nodes (and every rank's imported copies) hold each star's light as of now */
void single_star_tree_lum_set_from_build(void)
{
    light_pending.clear(); light_pending_lum.clear();
    single_star_tree_lum_fill_all();
}

/* the stars among the first n particles whose light differs from tree_lum (single_star_light_exchange_scan), in index order,
   with their light as of now: one kernel marks, a prefix sum places them, a second kernel evaluates the pending ones */
template<class Space> static int single_star_light_scan_on(int n, double tol)
{
    struct particle_data *kp = P; struct gas_cell_data *kc = CellP; const struct single_star_tree_lum *tl = SingleStars.tree_lum;
    int *mark = (int *) gizmo_gpu_alloc_shared((size_t) (n + 1) * sizeof(int), "single_star_light_scan");
    if(!mark) {printf("single_star_light_exchange_scan: rank %d has no shared memory for %d particles\n", ThisTask, n); fflush(stdout); endrun(7823);}
    gizmo_gpu_kernel_launch<Space>("single_star_light_scan", n, KOKKOS_LAMBDA(int i) {
        mark[i] = 0;
        if(kp[i].Type != 4 || kp[i].StarIdx <= 0) {return;}
        const int s = kp[i].StarIdx - 1;
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, kp, kc);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            const double l0 = tl[s].L[b], lb = (float) lum[b];
            if(((l0 == 0) != (lb == 0)) || fabs(lb - l0) > tol * DMAX(fabs(lb), fabs(l0))) {mark[i] = 1; break;}
        }
    });
    int n_pending = 0;
    Kokkos::parallel_scan("single_star_light_place", Kokkos::RangePolicy<Space>(0, n), KOKKOS_LAMBDA(int i, int &off, const bool final) {
        const int m = mark[i];
        if(final && m) {mark[i] = off + 1;} /* 1 + the place in the list; 0 = not pending */
        off += m;
    }, n_pending);
    Kokkos::fence();
    int *list = (int *) gizmo_gpu_alloc_shared((size_t) (n_pending + 1) * sizeof(int), "single_star_light_scan");
    float *lum_out = (float *) gizmo_gpu_alloc_shared((size_t) (n_pending + 1) * N_RT_FREQ_BINS * sizeof(float), "single_star_light_scan");
    if(!list || !lum_out) {printf("single_star_light_exchange_scan: rank %d has no shared memory for %d pending stars\n", ThisTask, n_pending); fflush(stdout); endrun(7823);}
    gizmo_gpu_kernel_launch<Space>("single_star_light_list", n, KOKKOS_LAMBDA(int i) {if(mark[i] > 0) {list[mark[i] - 1] = i;}});
    gizmo_gpu_kernel_launch<Space>("single_star_light_pending", n_pending, KOKKOS_LAMBDA(int k) {
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(list[k], 1, lum, kp, kc);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {lum_out[(size_t) k * N_RT_FREQ_BINS + b] = (float) lum[b];}
    });
    light_pending.assign(list, list + n_pending);
    light_pending_lum.assign(lum_out, lum_out + (size_t) n_pending * N_RT_FREQ_BINS);
    gizmo_gpu_free_shared(mark); gizmo_gpu_free_shared(list); gizmo_gpu_free_shared(lum_out);
    return n_pending;
}

/* local stars whose band light differs from what the nodes hold by more than tol of their own in some band, or switched a band
   on or off (tol 0: any difference); returns their number. No collective. */
int single_star_light_exchange_scan(double tol)
{
    light_pending.clear(); light_pending_lum.clear();
    if(!force_tree_is_allocated()) {return 0;}
    const double t0 = MPI_Wtime();
    const int n = (NumPart < All.TreeParticleSlots) ? NumPart : All.TreeParticleSlots, on_device = gizmo_gpu_stage_on_device(n);
    if(n > 0) {
        if(on_device) {GIZMO_GPU_ENSURE_ALL_FRESH(); single_star_light_scan_on<Kokkos::DefaultExecutionSpace>(n, tol);}
        else {single_star_light_scan_on<Kokkos::DefaultHostExecutionSpace>(n, tol);}
        loop_route_count("single_star_light_scan", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
    }
    light_scan_seconds += MPI_Wtime() - t0;
    return (int) light_pending.size();
}

/* add one change to node `no` (local, top-level or imported copy) at the current time: band luminosities, and the
   luminosity-weighted centre and velocity re-weighted with the star's position and velocity; the walk's mirror follows */
static void single_star_light_apply(int no, const struct single_star_light_change *c)
{
    if(Nodes[no].Ti_current != All.Ti_Current) {force_drift_node(no, All.Ti_Current);}
    const double L0 = moment_light_centre_weight(&Nodes[no].stellar_lum[0]);
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {
        double l = (double) Nodes[no].stellar_lum[b] + c->dl[b];
        Nodes[no].stellar_lum[b] = (MyFloat) ((l > 0) ? l : 0);
    }
    const double L1 = moment_light_centre_weight(&Nodes[no].stellar_lum[0]);
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    if(L1 > 0) {
        const double h = Nodes[no].len; /* guard against cancellation when the dominant source goes dark: stay near the node */
        for(int k = 0; k < 3; k++) {
            double sk = (L0 * (double) Nodes[no].rt_source_lum_s[k] + (L1 - L0) * c->x[k]) / L1;
            sk = DMIN(DMAX(sk, (double) Nodes[no].center[k] - h), (double) Nodes[no].center[k] + h);
            Nodes[no].rt_source_lum_s[k] = (MyFloat) sk;
            Extnodes[no].rt_source_lum_vs[k] = (MyFloat) ((L0 * (double) Extnodes[no].rt_source_lum_vs[k] + (L1 - L0) * c->v[k]) / L1);
        }
    } else {
        for(int k = 0; k < 3; k++) {Nodes[no].rt_source_lum_s[k] = Nodes[no].center[k]; Extnodes[no].rt_source_lum_vs[k] = 0;}
    }
#endif
    gpu_mirror_put_node_light(no);
}

static inline int single_star_top_level_node(int no) {return no >= All.TreeNodeIndexBase && no - All.TreeNodeIndexBase < NTopnodes;}

/* collective, on every rank when any rank has pending stars (or before a node-moment refresh): each pending star's light change
   is added to its rank's nodes below the top level, sent to the ranks holding imported copies of those nodes or of the star's
   leaf, and to every rank for the top-level nodes above it; all ranks apply the top-level changes in rank order, so their
   copies of the top tree stay identical. The nodes then hold every pending star's current light. */
void single_star_light_exchange(void)
{
    const double t0 = MPI_Wtime();
    std::vector<std::vector<struct single_star_light_change>> out(NTask);
    for(size_t k = 0; k < light_pending.size(); k++) {
        const int p = light_pending[k];
        int s = single_star_slot(P[p].StarIdx);
        struct single_star_light_change c; memset(&c, 0, sizeof(c));
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            const float lb = light_pending_lum[k * N_RT_FREQ_BINS + b]; /* the scan's evaluation (same space as tree_lum) */
            c.dl[b] = (double) lb - (double) SingleStars.tree_lum[s].L[b];
            SingleStars.tree_lum[s].L[b] = lb;
        }
        const double dt = get_drift_factor(P[p].Ti_current, All.Ti_Current, p, 0);
        for(int k = 0; k < 3; k++) {c.x[k] = (double) P[p].Pos[k] + (double) P[p].Vel[k] * dt; c.v[k] = (double) P[p].Vel[k];}
        for(int R = 0; R < NTask; R++) {
            if(R == ThisTask) {continue;}
            const int j = let_light_wire_of(R, -1 - p);
            if(j >= 0) {c.kind = 1; c.target = j; out[R].push_back(c);}
        }
        int no = Father[p];
        while(no >= All.TreeNodeIndexBase && no < All.TreeNodeIndexBase + MaxNodes) {
            for(int R = 0; R < NTask; R++) {
                if(R == ThisTask) {continue;}
                const int j = let_light_wire_of(R, no);
                if(j >= 0) {c.kind = 1; c.target = j; out[R].push_back(c);}
            }
            if(single_star_top_level_node(no)) {c.kind = 0; c.target = no; for(int R = 0; R < NTask; R++) {out[R].push_back(c);} break;}
            single_star_light_apply(no, &c);
            no = Nodes[no].u.d.father;
        }
    }
    std::vector<int> scount(NTask), rcount(NTask), soff(NTask), roff(NTask);
    long long nsend = 0, nrecv = 0;
    for(int R = 0; R < NTask; R++) {scount[R] = (int) (out[R].size() * sizeof(struct single_star_light_change)); soff[R] = (int) nsend; nsend += scount[R];}
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for(int R = 0; R < NTask; R++) {roff[R] = (int) nrecv; nrecv += rcount[R];}
    std::vector<char> sbuf(nsend > 0 ? nsend : 1), rbuf(nrecv > 0 ? nrecv : 1);
    for(int R = 0; R < NTask; R++) {if(scount[R] > 0) {memcpy(sbuf.data() + soff[R], out[R].data(), scount[R]);}}
    MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_BYTE, rbuf.data(), rcount.data(), roff.data(), MPI_BYTE, MPI_COMM_WORLD);
    long long missing = 0;
    for(int r = 0; r < NTask; r++) {
        const int n = rcount[r] / (int) sizeof(struct single_star_light_change);
        for(int m = 0; m < n; m++) {
            struct single_star_light_change c; memcpy(&c, rbuf.data() + roff[r] + (size_t) m * sizeof(c), sizeof(c));
            if(c.kind == 1) {
                const int no = let_light_foreign_node(r, c.target);
                if(no < 0) {missing++; continue;}
                single_star_light_apply(no, &c);
            } else {
                for(int no = c.target; single_star_top_level_node(no); no = Nodes[no].u.d.father) {single_star_light_apply(no, &c);}
            }
        }
    }
    if(missing) {printf("SINGLE_STAR_LIGHT_EXCHANGE: rank %d received %lld changes for imported nodes it does not hold\n", ThisTask, missing); fflush(stdout); endrun(90001101);}
    const double here[3] = {(double) light_pending.size(), (double) (nsend - scount[ThisTask]), MPI_Wtime() - t0};
    light_stats[0] += 1; for(int k = 0; k < 3; k++) {light_stats[k + 1] += here[k];}
    double sum[2] = {0, 0}, tloc[2] = {here[2], light_scan_seconds}, tmax[2] = {0, 0};
    MPI_Reduce(here, sum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(tloc, tmax, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("SINGLE_STAR_LIGHT_EXCHANGE: t=%.10g stars=%.0f bytes=%.0f seconds_max=%.3e scan_seconds_cumulative_max=%.3e\n", All.Time, sum[0], sum[1], tmax[0], tmax[1]);}
    light_pending.clear(); light_pending_lum.clear();
}

/* this rank's cumulative exchange cost: calls, changed stars, bytes sent to other ranks, seconds */
void single_star_light_exchange_stats(double *out4) {for(int k = 0; k < 4; k++) {out4[k] = light_stats[k];}}
#endif

#ifdef GALSF_SINGLE_STARS_LIGHT_REFRESHES_TREE /* opt-in: by default the nodes keep the light of the last build (trunk's bounded staleness) */
/* before a gravity call: refresh the tree node moments (luminosities) if a single star started to shine, started to ionise or went
   dark since the last call, or its band luminosity moved by more than SINGLE_STAR_TREE_LUM_TOL of its own since the nodes last
   saw it; reduced in gravity_tree, so no collective here. A switch or a drift sets 2: on several ranks gravity_tree then
   rebuilds, which re-sends the node light to the ranks holding copies (one rank: a refresh) */
void single_star_light_switch_check(void)
{
    static double t_last = -1; double t0 = t_last; t_last = All.Time;
    if(t0 < 0 || All.Time <= t0) {return;}
    if(!force_tree_is_allocated() || TreeReconstructFlag) {return;} /* a build is due: it sees every star as it is now */
    const double dt_yr = evaluate_time_since_t_initial_in_Gyr(t0) * 1.e9, mcut = All.IMFSampleStellarMassCut;
    const int n = NumPart, on_device = gizmo_gpu_stage_on_device(n);
    if(n <= 0) {return;}
    if(on_device) {GIZMO_GPU_ENSURE_ALL_FRESH();} /* the light and the age read All */
    const StellarTablesView T = on_device ? All.StellarTablesDev : stellar_tables_host_view();
    const SingleStarRecordView R = SINGLE_STAR_RECORD_HOST_VIEW();
    struct particle_data *kp = P; struct gas_cell_data *kc = CellP; const struct single_star_tree_lum *tl = SingleStars.tree_lum;
#if defined(GALSF_FB_MECHANICAL)
    const uint8_t *ev = SingleStars.event_bits;
#endif
    /* a star that switched a band on or off, drifted beyond the tolerance, or passed an on/off age since the last call */
    auto stale = KOKKOS_LAMBDA(int i, int &n_stale) {
        if(kp[i].Type != 4 || kp[i].StarIdx <= 0 || kp[i].Mass <= 0) {return;}
        const int s = kp[i].StarIdx - 1;
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, kp, kc);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            const double l0 = tl[s].L[b], lb = (float) lum[b]; /* as stored */
            if((l0 == 0) != (lb == 0)) {n_stale++; return;} /* a band switched on or off */
            if(fabs(lb - l0) > SINGLE_STAR_TREE_LUM_TOL * DMAX(fabs(lb), fabs(l0))) {n_stale++; return;}
        }
        if(R.M_drawn[s] <= mcut) {return;}
#if defined(GALSF_FB_MECHANICAL)
        if(ev[s] & (SINGLE_STAR_EVENT_SN_DONE | SINGLE_STAR_EVENT_AGB_DONE)) {return;}
#endif
        double logM = log10(R.M_drawn[s]), logZ = log10(DMAX(R.Z_birth[s], 1e-10));
        double age = single_star_age_yr_core(kp, i), t_pms = stbl_interp2d(T, T.t_PMS, logM, logZ), t_on = single_star_ionizing_onset_yr_core(T, R, kp, i);
        double t_switch[3] = {t_pms, t_pms + stbl_interp2d(T, T.lifetime_yr, logM, logZ), t_on};
        for(int k = 0; k < 3; k++) {if(t_switch[k] > 0 && age - dt_yr < t_switch[k] && t_switch[k] <= age) {n_stale++; return;}}
    };
    int n_stale = 0;
    if(on_device) {gizmo_gpu_kernel_launch_reduce("single_star_light_switch", n, stale, n_stale);}
    else {gizmo_gpu_kernel_launch_reduce<Kokkos::DefaultHostExecutionSpace>("single_star_light_switch", n, stale, n_stale);}
    loop_route_count("single_star_light_switch", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
    if(n_stale > 0) {TreeMomentsStaleFlag = 2;}
}

/* after a gravity call: if it built the tree or refreshed the node moments, the node luminosities are the stars' current ones */
void single_star_tree_lum_store(void)
{
    if(!TreeMomentsRenewed) {return;}
    single_star_tree_lum_fill_all();
}
#endif
