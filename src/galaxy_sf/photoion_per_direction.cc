/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* galaxy_sf/photoion_per_direction.cc -- host side of the per-direction photoionisation (Specs in photoion_per_direction.h).
 * Ionising single stars share one cadence (PhotoionMaxTimestep); at each solve every source finds its fronts (search
 * sphere grown or refined as stella_dev). Reference: stella_dev galaxy_sf/resolvedism_photoion.cc. */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
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

/* ionising photon rate of source i [1/s] */
static double photoion_source_rate(int i)
{
    if(P[i].StarIdx <= 0 || P[i].Mass <= 0) {return 0;}
    return All.HIIRegion_fLum_Coupled * single_star_ionizing_photon_rate(i);
}

/* the photoionisation timebin: the largest bin whose step is within PhotoionMaxTimestep and MaxSizeTimestep */
static int photoion_timebin(void)
{
    double dt = DMIN(All.PhotoionMaxTimestep / UNIT_TIME_IN_YR, All.MaxSizeTimestep);
    integertime ti = (integertime)(dt / All.Timebase_interval);
    int b = 1;
    while(b + 1 < TIMEBINS && ((integertime)1 << (b + 1)) <= ti) {b++;}
    return b;
}

/* largest timestep of single star p [code]: an ionising source keeps to the photoionisation bin; a star that will
   ionise steps onto its onset age */
double photoion_max_timestep(int p)
{
    double t_on = single_star_ionizing_onset_yr(p);
    if(t_on < 0) {return 0;}
    if(photoion_source_rate(p) > 0) {return (double)((integertime)1 << photoion_timebin()) * All.Timebase_interval;}
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

void PiHistSpec::apply_active_writeback(const neighbor_loop_args& args, int slot, int, const AccumData& a)
{
    ((PiHistOut *) nlr_aux<PiHistSpec>(args)->out)[slot] = a;
}

template <typename Spec>
static void pi_run(std::vector<int>& list, std::vector<PiSourceIn>& in, PiSpecBase::Aux& aux)
{
    aux.host_src = in.empty() ? nullptr : in.data();
    neighbor_loop_args args = nlr_default_args();
    args.active_list = list.empty() ? nullptr : list.data(); args.num_active = (int) list.size(); args.aux = &aux;
    run_neighbor_loop<Spec>(args);
}

/* ---- the front of each pixel ---- */

struct pi_src {int i; double Q, R; int refine, open; double front[PI_NPIX], mark[PI_NPIX];};

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
    return x;
}

/* solve when the photoionisation bin is active. Collectives: the source count; per pass the runner call and one
   reduction; the log row. */
void photoionization_per_direction(void)
{
    if(All.HIIRegion_fLum_Coupled <= 0 || All.Time <= 0) {return;}
    const int b_pi = photoion_timebin();
    if(!TimeBinActive[b_pi]) {return;}
    const double max_r = All.PhotoionMaxSearchRadius / (UNIT_LENGTH_IN_KPC * All.cf_atime);
    std::vector<pi_src> S;
    for(int i : ActiveParticleList) {
        if(P[i].StarIdx <= 0) {continue;}
        double Q = photoion_source_rate(i);
        if(Q <= 0) {continue;}
        pi_src s; std::memset(&s, 0, sizeof(s)); s.i = i; s.Q = Q; s.open = 1;
        double nH = (P[i].DensityAroundParticle > 0) ? HYDROGEN_MASSFRAC * P[i].DensityAroundParticle * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS : 1.0;
        double R = pow(3.0 * Q / (4.0 * M_PI * PI_ALPHA_B * nH * nH), 1.0/3.0) / (UNIT_LENGTH_IN_CGS * All.cf_atime); /* the mean-density guess */
        R = DMAX(R, P[i].KernelRadius); if(R <= 0) {R = All.ForceSoftening[P[i].Type];}
        if(max_r > 0 && R > max_r) {R = max_r;}
        s.R = R; S.push_back(s);
    }
    long long n_loc = (long long) S.size(), n_tot = 0;
    MPI_Allreduce(&n_loc, &n_tot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(n_tot == 0) {return;}
    PRINT_STATUS(" ..per-direction photoionisation around %lld single stars", n_tot);

    /* fronts */
    int passes = 0, any_open = 1;
    for(; passes < PI_MAX_PASSES && any_open; passes++) {
        std::vector<int> list, which; std::vector<PiSourceIn> in;
        for(size_t a = 0; a < S.size(); a++) {if(S[a].open) {list.push_back(S[a].i); in.push_back(pi_source_in(S[a])); which.push_back((int)a);}}
        std::vector<PiHistOut> H(in.size() + 1);
        PiSpecBase::Aux aux = {nullptr, H.data()};
        pi_run<PiHistSpec>(list, in, aux);
        int n_open = 0;
        for(size_t a = 0; a < in.size(); a++) {
            pi_src& s = S[which[a]];
            s.open = pi_analyse(s, H[a], passes < PI_MAX_PASSES - 1, 1, max_r);
            n_open += s.open;
        }
        MPI_Allreduce(&n_open, &any_open, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    /* log row (HIIheating.txt): sources, Q/X, (ionised cells: columns 4-7), passes, smallest and largest front */
    double loc[5] = {0, 0, 0, 0, 0}, tot[5], ext[2] = {0, -MAX_REAL_NUMBER}, ext_tot[2]; /* front max, -(front min) */
    for(size_t a = 0; a < S.size(); a++) {
        loc[0] += S[a].Q;
        for(int k = 0; k < PI_NPIX; k++) {ext[0] = DMAX(ext[0], S[a].front[k]); ext[1] = DMAX(ext[1], -S[a].front[k]);}
    }
    MPI_Reduce(loc, tot, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ext, ext_tot, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        double to_kpc = All.cf_atime * UNIT_LENGTH_IN_KPC;
        fprintf(FdHIIHeating, "%.16g %lld %g %g %g %g %.8g %d %d %.10g %.10g\n", All.Time, n_tot, tot[0] / HYDROGEN_MASSFRAC, tot[2], tot[3] * UNIT_MASS_IN_SOLAR,
                (tot[3] > 0) ? tot[4] / tot[3] * to_kpc : 0., (tot[0] > 0) ? tot[1] / tot[0] : 0., 0, passes, -ext_tot[1] * to_kpc, ext_tot[0] * to_kpc);
        fflush(FdHIIHeating);
    }
}

#endif /* GALSF_FB_PHOTOIONIZATION_PER_DIRECTION */
