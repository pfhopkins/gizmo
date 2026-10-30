/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/photoion_per_direction.h -- Specs of the per-direction Stromgren photoionisation (GALSF_FB_PHOTOIONIZATION_PER_DIRECTION).
 * Each source splits Q over 12 HEALPix pixels (NSIDE 1); per pixel the front is where the cells' recombination bills,
 * summed outward, use up Q/12. histogram: per pixel and radial bin, the summed bills (integer units, so independent of
 * order, threads and ranks) and the innermost cell radius, kept in a pool on the cell's rank and sent to the source's rank
 * in one exchange (not in the reply records); with sharing, a cell claimed by several sources bills each its
 * share q_sc / A_c. claim: A_c += q_sc for cells newly inside a source's fronts. stamp: cells inside the fronts are
 * ionised until the next solve. drain: A_c = 0. All run where the cell lives; A_c is a side buffer by local gas index,
 * zero between solves. Design note: design_photoionisation_pixelised.md. */
#ifndef PHOTOION_PER_DIRECTION_H
#define PHOTOION_PER_DIRECTION_H
#include <Kokkos_Core.hpp>
#include "../declarations/allvars.h"
#ifdef GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/mode_b_local_walker.h"

#define PI_NPIX 12
#define PI_NBIN 64
#define PI_BIN_UNIT 1099511627776.0           /* 2^40 histogram units per pixel budget */
#define PI_BIN_MAX  4611686018427387904LL     /* 2^62: saturation of a bin */
#define PI_RMIN_EMPTY 1.0e300

struct PiSourceIn {
    Vec3<double> Pos;
    double R;                     /* search radius (comoving) */
    double S_pix;                 /* Q/12 [1/s] */
    double held[PI_NPIX];         /* fronts of the cells it has claimed; -1 = none */
    double mark[PI_NPIX];         /* marking fronts */
    int g;                        /* global index of the source in this solve */
};
struct PiCallScalars {
    NlrCommonScalars common;
    double bill_fac;              /* bill [1/s] = bill_fac * rho_code * m_code */
    integertime stamp;            /* stamp: ionised until this time */
    long long *claim;             /* A_c by local gas index */
    int n_claim;
    int share;                    /* histogram: shared bills */
    int *pool_slot, *pool_g, *pool_n, *pool_full; /* histogram pool: slot of global source g (-1: none), g of a slot, slots used, overflow */
    long long *pool_cost;         /* [slot][PI_NPIX * PI_NBIN] */
    double *pool_rmin;
    int pool_cap;
};
struct PiActiveState {Vec3<double> pos; double h_search; PiSourceIn src;};
struct PiDeviceContext : NeighborLoopDeviceContextBase {const PiSourceIn *per_active_local;};
struct PiNeighborData {struct particle_data *neighbor_particle; struct gas_cell_data *neighbor_cell; int j;};
struct PiHistOut {long long cost[PI_NPIX * PI_NBIN]; double rmin[PI_NPIX * PI_NBIN];}; /* one source, assembled on its rank */
struct PiCountOut {long long n;};
struct PiLedgerOut {double bill, n_new, m_new, mr_new;}; /* bills of its cells; cells newly stamped, their mass, mass x radius */

/* HEALPix NSIDE 1 ring pixel of direction (x, y, z) (Gorski et al. 2005) */
KOKKOS_INLINE_FUNCTION
static int pi_healpix_pixel(double x, double y, double z)
{
    double r = sqrt(x*x + y*y + z*z);
    if(r <= 0) {return 0;}
    z /= r;
    double phi = atan2(y, x); if(phi < 0) {phi += 2.0 * M_PI;}
    double tt = phi / (0.5 * M_PI), za = fabs(z);
    if(za <= 2.0/3.0) {
        long jp = (long)(0.5 + tt - 0.75 * z), jm = (long)(0.5 + tt + 0.75 * z);
        long ir = 2 + jp - jm, ip = ((jp + jm + (1 - (ir & 1))) / 2) % 4;
        return (int)((ir - 1) * 4 + ip);
    }
    double tp = tt - (long)tt, tmp = sqrt(3.0 * (1.0 - za));
    long ir = (long)(tp * tmp) + (long)((1.0 - tp) * tmp) + 1, ip = ((long)(tt * ir)) % (4 * ir);
    return (z > 0) ? (int)(2 * ir * (ir - 1) + ip) : (int)(12 - 2 * ir * (ir + 1) + ip);
}

/* cell j seen from source s: radius, pixel, bill alpha_B n_e n_H V [1/s]; 0 outside the search sphere */
KOKKOS_INLINE_FUNCTION
static int pi_cell(const PiSourceIn& s, const struct particle_data& Pj, const struct gas_cell_data *Cj, int j, const PiCallScalars& cs,
                   double *r, int *pix, double *bill, long long *q)
{
    if(Pj.Type != 0 || Cj == nullptr || Pj.Mass <= 0 || j >= cs.n_claim) {return 0;}
#ifdef HYDRO_MULTIFLUID_DM
    if(Pj.FluidType == FLUID_DM) {return 0;}
#endif
    Vec3<double> d;
    for(int k = 0; k < 3; k++) {d[k] = (double)Pj.Pos[k] - s.Pos[k];}
    nearest_xyz(d);
    double r2 = d.norm_sq();
    if(r2 <= 0 || r2 >= s.R * s.R) {return 0;}
    *r = sqrt(r2);
    *pix = pi_healpix_pixel(d[0], d[1], d[2]);
    *bill = cs.bill_fac * (double)Cj->Density * (double)Pj.Mass;
    if(!(*bill > 0)) {return 0;}
    double h = (double)Pj.Get_Particle_Size(), w = ldexp(s.S_pix * h * h / (4.0 * (r2 + h * h)) / *bill, 32); /* photons it delivers per bill, x 2^32 */
    *q = (w >= 18014398509481984.0) ? 18014398509481984LL : ((w < 1.0) ? 1LL : (long long)llround(w)); /* claim weight in [1, 2^54] */
    return 1;
}

KOKKOS_INLINE_FUNCTION static long long pi_sat_add(long long a, long long b) {long long c = a + b; return (c > PI_BIN_MAX) ? PI_BIN_MAX : c;}
KOKKOS_INLINE_FUNCTION static void pi_atomic_sat_add(long long *x, long long b)
{
    long long old = Kokkos::atomic_load(x);
    while(true) {
        long long want = pi_sat_add(old, b);
        if(want == old) {return;}
        long long seen = Kokkos::atomic_compare_exchange(x, old, want);
        if(seen == old) {return;}
        old = seen;
    }
}
/* the pool slot of global source g on this rank, claimed at first use; -1 if the pool is full */
KOKKOS_INLINE_FUNCTION static int pi_pool_claim(const PiCallScalars& cs, int g)
{
    int s = Kokkos::atomic_load(&cs.pool_slot[g]);
    if(s >= 0) {return s;}
    int n = Kokkos::atomic_fetch_add(cs.pool_n, 1);
    if(n >= cs.pool_cap) {Kokkos::atomic_store(cs.pool_full, 1); return -1;}
    int seen = Kokkos::atomic_compare_exchange(&cs.pool_slot[g], -1, n);
    if(seen == -1) {cs.pool_g[n] = g; return n;}
    return seen; /* another thread claimed it; slot n stays empty */
}

struct PiSpecBase {
    static constexpr int                    search_mode        = MODE_B_SEARCH_ONEWAY;
    static constexpr unsigned int           neighbor_type_mask = (1u << 0);
    static constexpr mode_b_radius_policy_t radius_policy      = MODE_B_RADIUS_DEFAULT;
    static constexpr WritePattern  write_pattern   = WritePattern::ActiveReduceOnly;
    static constexpr SidxCacheKind sidx_cache_kind = SidxCacheKind::GasOnly;
    static constexpr bool needs_live_neighbours     = true;  /* the stamp writes the live cell */
    static constexpr bool uses_ghost_writeback      = false; /* evaluated where the cell lives */
    static constexpr bool uses_ghost_write_detector = false;
    static constexpr bool keeps_neighbour_mass_positive = true;
    using CallScalars    = PiCallScalars;
    using ActiveData     = PiActiveState;
    using NeighborData   = PiNeighborData;
    using DeviceContext  = PiDeviceContext;
    using ScatterData    = NoScatter;
    using IdentityFields = NoIdentity;
    using IterControl    = NotIterative;
    struct Aux {const PiSourceIn *host_src; void *out; integertime stamp; int share;};
    static bool        is_active(int i);
    static double      search_radius(const neighbor_loop_args& args, int active_slot, int i);
    static CallScalars populate_call_scalars(const neighbor_loop_args& args);
    static void populate_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    static void cleanup_device_context(const neighbor_loop_args& args, DeviceContext& ctx);
    KOKKOS_INLINE_FUNCTION
    static ActiveData load_active(const DeviceContext& dctx, int active_slot, int i, double h_search, const CallScalars&) {
        ActiveData a{};
        if(dctx.per_active_local != nullptr) {a.src = dctx.per_active_local[active_slot];}
        else {for(int k = 0; k < 3; k++) {a.src.Pos[k] = (double)dctx.P[i].Pos[k];} a.src.R = h_search;}
        a.pos = a.src.Pos; a.h_search = a.src.R;
        return a;
    }
    KOKKOS_INLINE_FUNCTION
    static NeighborData load_neighbor(const DeviceContext& dctx, int j, const IdentitySidecar&, const ActiveData&) {
        NeighborData n{};
        n.neighbor_particle = &dctx.P[j];
        n.neighbor_cell = (dctx.CellP != nullptr && dctx.P[j].Type == 0) ? &dctx.CellP[j] : nullptr;
        n.j = j;
        return n;
    }
};

struct PiHistSpec : PiSpecBase {
    static constexpr const char *loop_name = "photoion_per_direction_histogram";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* integer adds and minima into the pool: bitwise in any order */
    using AccumData = PiCountOut; /* the histogram stays in the pool of the cell's rank, out of the reply record */
    static void apply_active_writeback(const neighbor_loop_args&, int, int, const AccumData&) {}
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.n = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.n += p.n;}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        double r, bill; int pix; long long q;
        if(!pi_cell(a.src, *nb.neighbor_particle, nb.neighbor_cell, nb.j, cs, &r, &pix, &bill, &q)) {return;}
        if(cs.share) { /* its share: held cells q / A_c, others as if it joined */
            long long A = Kokkos::atomic_load(&cs.claim[nb.j]);
            if(r > a.src.held[pix]) {A += q;}
            if(A > 0) {bill *= (double)q / (double)A;}
        }
        double u = bill / a.src.S_pix * PI_BIN_UNIT;
        long long iu = (u >= (double)PI_BIN_MAX) ? PI_BIN_MAX : (long long)llround(u);
        int b = (int)(r * PI_NBIN / a.src.R); if(b >= PI_NBIN) {b = PI_NBIN - 1;}
        int s = pi_pool_claim(cs, a.src.g);
        if(s < 0) {return;}
        size_t k = (size_t)s * PI_NPIX * PI_NBIN + pix * PI_NBIN + b;
        pi_atomic_sat_add(&cs.pool_cost[k], iu);
        Kokkos::atomic_min(&cs.pool_rmin[k], r);
        out.n++;
    }
};

/* stamp: cells inside the marking fronts are ionised until cs.stamp (AtomicMax: order-free); ledger of their bills */
struct PiStampSpec : PiSpecBase {
    static constexpr const char *loop_name = "photoion_per_direction_stamp";
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::EpsilonAtomic; /* integer max: bitwise in any order */
    using AccumData = PiLedgerOut;
    static void apply_active_writeback(const neighbor_loop_args& args, int active_slot, int i, const AccumData& accum);
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {a.bill = a.n_new = a.m_new = a.mr_new = 0;}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {l.bill += p.bill; l.n_new += p.n_new; l.m_new += p.m_new; l.mr_new += p.mr_new;}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        double r, bill; int pix; long long q;
        if(!pi_cell(a.src, *nb.neighbor_particle, nb.neighbor_cell, nb.j, cs, &r, &pix, &bill, &q)) {return;}
        if(r > a.src.mark[pix]) {return;}
        long long A = Kokkos::atomic_load(&cs.claim[nb.j]);
        out.bill += (A > 0) ? bill * (double)q / (double)A : bill; /* its share: the shares of a cell sum to its bill */
        integertime old = Kokkos::atomic_fetch_max(&nb.neighbor_cell->PhotoionizedUntil, cs.stamp);
        if(old < cs.stamp) {double m = (double)nb.neighbor_particle->Mass; out.n_new += 1; out.m_new += m; out.mr_new += m * r;}
    }
};

/* claim: A_c += q_sc for cells between the held and the marking front (integer adds: order-free) */
struct PiClaimSpec : PiStampSpec {
    static constexpr const char *loop_name = "photoion_per_direction_claim";
    static void apply_active_writeback(const neighbor_loop_args&, int, int, const AccumData&) {}
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData&, NoScatter&, const CallScalars& cs) {
        double r, bill; int pix; long long q;
        if(!pi_cell(a.src, *nb.neighbor_particle, nb.neighbor_cell, nb.j, cs, &r, &pix, &bill, &q)) {return;}
        if(r > a.src.held[pix] && r <= a.src.mark[pix]) {Kokkos::atomic_add(&cs.claim[nb.j], q);}
    }
};

/* drain: A_c = 0 inside the search sphere */
struct PiDrainSpec : PiClaimSpec {
    static constexpr const char *loop_name = "photoion_per_direction_drain";
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData&, NoScatter&, const CallScalars& cs) {
        double r, bill; int pix; long long q;
        if(!pi_cell(a.src, *nb.neighbor_particle, nb.neighbor_cell, nb.j, cs, &r, &pix, &bill, &q)) {return;}
        Kokkos::atomic_store(&cs.claim[nb.j], 0LL);
    }
};

#endif /* GALSF_FB_PHOTOIONIZATION_PER_DIRECTION */
#endif /* PHOTOION_PER_DIRECTION_H */
