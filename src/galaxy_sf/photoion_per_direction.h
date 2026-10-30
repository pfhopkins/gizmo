/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/photoion_per_direction.h -- Specs of the per-direction Stromgren photoionisation (GALSF_FB_PHOTOIONIZATION_PER_DIRECTION).
 * Each source splits Q over 12 HEALPix pixels (NSIDE 1); per pixel the front is where the cells' recombination bills,
 * summed outward, use up Q/12. histogram: per pixel and radial bin, the summed bills (integer units, so independent of
 * order, threads and ranks) and the innermost cell radius. stamp: cells inside the fronts are ionised until the next
 * solve. Both run where the cell lives. Design note: design_photoionisation_pixelised.md. */
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
    double mark[PI_NPIX];         /* marking fronts (stamp) */
};
struct PiCallScalars {
    NlrCommonScalars common;
    double bill_fac;              /* bill [1/s] = bill_fac * rho_code * m_code */
    integertime stamp;            /* stamp: ionised until this time */
};
struct PiActiveState {Vec3<double> pos; double h_search; PiSourceIn src;};
struct PiDeviceContext : NeighborLoopDeviceContextBase {const PiSourceIn *per_active_local;};
struct PiNeighborData {struct particle_data *neighbor_particle; struct gas_cell_data *neighbor_cell; int j;};
struct PiHistOut {long long cost[PI_NPIX * PI_NBIN]; double rmin[PI_NPIX * PI_NBIN];};
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
static int pi_cell(const PiSourceIn& s, const struct particle_data& Pj, const struct gas_cell_data *Cj, const PiCallScalars& cs,
                   double *r, int *pix, double *bill)
{
    if(Pj.Type != 0 || Cj == nullptr || Pj.Mass <= 0) {return 0;}
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
    return (*bill > 0);
}

KOKKOS_INLINE_FUNCTION static long long pi_sat_add(long long a, long long b) {long long c = a + b; return (c > PI_BIN_MAX) ? PI_BIN_MAX : c;}

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
    struct Aux {const PiSourceIn *host_src; void *out; integertime stamp;};
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
    static constexpr ModeBEvalOMP modeb_eval_omp = ModeBEvalOMP::BitwiseReadonly;
    using AccumData = PiHistOut;
    static void apply_active_writeback(const neighbor_loop_args& args, int active_slot, int i, const AccumData& accum);
    KOKKOS_INLINE_FUNCTION static void zero_accum(AccumData& a) {for(int k = 0; k < PI_NPIX * PI_NBIN; k++) {a.cost[k] = 0; a.rmin[k] = PI_RMIN_EMPTY;}}
    KOKKOS_INLINE_FUNCTION static void merge_accum(AccumData& l, const AccumData& p) {
        for(int k = 0; k < PI_NPIX * PI_NBIN; k++) {l.cost[k] = pi_sat_add(l.cost[k], p.cost[k]); if(p.rmin[k] < l.rmin[k]) {l.rmin[k] = p.rmin[k];}}
    }
    KOKKOS_INLINE_FUNCTION
    static void pair_kernel(const ActiveData& a, const NeighborData& nb, AccumData& out, NoScatter&, const CallScalars& cs) {
        double r, bill; int pix;
        if(!pi_cell(a.src, *nb.neighbor_particle, nb.neighbor_cell, cs, &r, &pix, &bill)) {return;}
        double u = bill / a.src.S_pix * PI_BIN_UNIT;
        long long iu = (u >= (double)PI_BIN_MAX) ? PI_BIN_MAX : (long long)llround(u);
        int b = (int)(r * PI_NBIN / a.src.R); if(b >= PI_NBIN) {b = PI_NBIN - 1;}
        out.cost[pix * PI_NBIN + b] = pi_sat_add(out.cost[pix * PI_NBIN + b], iu);
        if(r < out.rmin[pix * PI_NBIN + b]) {out.rmin[pix * PI_NBIN + b] = r;}
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
        double r, bill; int pix;
        if(!pi_cell(a.src, *nb.neighbor_particle, nb.neighbor_cell, cs, &r, &pix, &bill)) {return;}
        if(r > a.src.mark[pix]) {return;}
        out.bill += bill;
        integertime old = Kokkos::atomic_fetch_max(&nb.neighbor_cell->PhotoionizedUntil, cs.stamp);
        if(old < cs.stamp) {double m = (double)nb.neighbor_particle->Mass; out.n_new += 1; out.m_new += m; out.mr_new += m * r;}
    }
};

#endif /* GALSF_FB_PHOTOIONIZATION_PER_DIRECTION */
#endif /* PHOTOION_PER_DIRECTION_H */
