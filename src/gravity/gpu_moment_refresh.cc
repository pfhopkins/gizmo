/* gpu_moment_refresh.cc — Step 13 Phase 6.2
 *
 * GPU implementation of the local-tree moment refresh that mirrors the
 * CPU code in force_refresh_node_moments (gravity/forcetree.cc:3826).
 * The CPU body is a 4-step pass (zero / particle-accumulate / bottom-up
 * propagate / normalize) over Nodes[MaxPart .. MaxPart+Numnodestree).
 *
 * On the GPU we use 5 device kernels with dependency-counter atomics so
 * the bottom-up walk parallelises:
 *   1) zero  : zero all moment scratch + init pending[]=0.
 *   2) count : each child node atomically increments its parent's
 *              pending counter (counts internal-node children).
 *   3) parts : each particle atomically accumulates into Father[i]'s
 *              scratch slot. Uses precomputed per-particle RT / sink /
 *              CR arrays (mirrors the gpu_gravtree.cc Phase 2-A..D
 *              pattern).
 *   4) walk  : each thread starts at node k. If pending[k] != 0 it
 *              aborts (a descendant will reach this node). Otherwise it
 *              walks up via father, atomic-adding moments to the parent
 *              and atomic-decrementing the parent's pending counter; if
 *              the previous value was 1, this thread now owns the
 *              parent and continues with curr=parent.
 *   5) norm  : divide mass-weighted sums by mass (or fall back to node
 *              centre if mass==0); set/clear BITFLAG_MULTIPLEPARTICLES.
 *
 * After the kernels finish, gpu_moment_writeback_to_aos copies the SoA
 * back into Nodes[]/Extnodes[] (mechanical inverse of seed_node_), so
 * the CPU pseudo path (force_exchange_pseudodata +
 * force_treeupdate_pseudos) reads identical AoS values.
 *
 * All conditional payloads from force_refresh_node_moments are handled:
 *   GRAVTREE_CALCULATE_GAS_MASS_IN_NODE / RT_USE_GRAVTREE (+CHIMES) /
 *   RT_SEPARATELY_TRACK_LUMPOS / SINK_PHOTONMOMENTUM /
 *   COSMIC_RAY_SUBGRID_LEBRON / SINK_CALC_DISTANCES (with the nested
 *   N_SINK / sink_vel / sink_acc / MaxFeedbackVel) /
 *   ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION / DM_SCALARFIELD_SCREENING.
 *
 * Per the Phase 6.2 design directive (handoff_step13_phase62_design.md),
 * payload-list expansions are factored into KOKKOS_INLINE_FUNCTION
 * helpers so each payload appears at most a handful of times; the X-style
 * directives `MOMENT_FOR_EACH_*` keep zero / accum / norm / writeback in
 * lock-step.
 *
 * Memory model notes
 *   * Moment accumulators live in `Kokkos::View<...>` allocations on
 *     `Kokkos::DefaultExecutionSpace` (= device-local scratch). This avoids
 *     the MI250X HIPManaged atomic penalty noted in the Phase 6 plan.
 *     `Kokkos::deep_copy` flushes results into the SharedSpace SoA before
 *     the host-side AoS writeback.
 *   * Per-particle precomputed inputs (RT lum, sink lum, CR injection)
 *     live in SharedSpace, mirroring gpu_gravtree.cc:1216+.
 *   * Father[i] is mirrored into a SharedSpace int* once per call.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "forcetree.h"
#include "gravtree_moment_kernel.h"

#ifdef RT_USE_GRAVTREE
#include "../radiation/rt_functions.h"
#endif


/* ------------------------------------------------------------------ */
/* Atomic-max helper for floating-point types using a CAS loop.       */
/* Kokkos::atomic_max for floats/doubles is backend-dependent; CAS is */
/* portable across CUDA/HIP/OpenMP-host. Inlined as KOKKOS_INLINE.    */
/* ------------------------------------------------------------------ */
template <typename T>
KOKKOS_INLINE_FUNCTION static void
atomic_max_fp(T *addr, T val)
{
    T old = *addr;
    while(val > old) {
        T prev = Kokkos::atomic_compare_exchange(addr, old, val);
        if(prev == old) {return;}
        old = prev;
    }
}

/* ------------------------------------------------------------------ */
/* Vec3 atomic add (component-wise).                                  */
/* ------------------------------------------------------------------ */
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION static void
atomic_add_vec3(Vec3<T> *addr, const Vec3<U>& val)
{
    Kokkos::atomic_add(&(*addr)[0], (T)val[0]);
    Kokkos::atomic_add(&(*addr)[1], (T)val[1]);
    Kokkos::atomic_add(&(*addr)[2], (T)val[2]);
}

/* Atomic write policy for the shared moment construction kernel (gravtree_moment_kernel.h). Kept in
 * this TU because the header carries no Kokkos dependency; gpu_moment_refresh injects it as the Ops
 * template argument while the topnode re-sum / force_add_element_to_tree use the header's plain policy. */
struct moment_atomic_ops {
    template <class T> KOKKOS_INLINE_FUNCTION static void add(T *dst, T v)  { Kokkos::atomic_add(dst, v); }
    template <class T> KOKKOS_INLINE_FUNCTION static void add_vec3(Vec3<T> *dst, const Vec3<T>& v) { atomic_add_vec3<T, T>(dst, v); }
    template <class T> KOKKOS_INLINE_FUNCTION static void fmax(T *dst, T v) { atomic_max_fp<T>(dst, v); }
    KOKKOS_INLINE_FUNCTION static void add_long(long *dst, long v) { Kokkos::atomic_add(dst, v); }
    KOKKOS_INLINE_FUNCTION static void add_int (int  *dst, int  v) { Kokkos::atomic_add(dst, v); }
};

/* ------------------------------------------------------------------ */
/* GPU-callable accessor for the cached force-softening kernel radius. */
/* Single source of truth: compute_force_softening_kernel_radius() in  */
/* forcetree.cc; populated by compute_all_force_softening() at startup */
/* and at the start of every gravity_tree() call.                      */
/* ------------------------------------------------------------------ */
KOKKOS_INLINE_FUNCTION static double
gpu_force_softening_kernelradius(const struct particle_data *Pp, int p)
{
    return Pp[p].ForceSoftening;
}

/* =================================================================== */
/* Per-particle precomputed inputs (RT, sink, CR). Same pattern as     */
/* gpu_gravtree.cc Phase 2-A..D. Allocated in SharedSpace so the device */
/* kernel can read them, but populated on host because the CPU helpers  */
/* (rt_get_source_luminosity, sink_lum_bol, cr_get_source_injection_rate)*/
/* are not GPU-callable.                                                */
/* =================================================================== */
namespace {
struct precomputed_t {
#ifdef RT_USE_GRAVTREE
    MyFloat *src_lum;             /* [NumPart * N_RT_FREQ_BINS] */
#ifdef CHIMES_STELLAR_FLUXES
    double  *src_lum_G0;          /* [NumPart * CHIMES_LOCAL_UV_NBINS] */
    double  *src_lum_ion;         /* [NumPart * CHIMES_LOCAL_UV_NBINS] */
#endif
#endif
#ifdef SINK_PHOTONMOMENTUM
    MyFloat       *bh_lum;        /* [NumPart] */
    Vec3<MyFloat> *bh_angle;      /* [NumPart] */
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat *cr_inject;           /* [NumPart] */
#endif
};

static int precompute_alloc_(precomputed_t& pre, int N)
{
#ifdef RT_USE_GRAVTREE
    long sz = (long)N * N_RT_FREQ_BINS * sizeof(MyFloat);
    pre.src_lum = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
    if(!pre.src_lum) {printf("gpu_moment_refresh: src_lum alloc failed\n"); endrun(913301); return 1;}
    memset(pre.src_lum, 0, sz);
#ifdef CHIMES_STELLAR_FLUXES
    long szc = (long)N * CHIMES_LOCAL_UV_NBINS * sizeof(double);
    pre.src_lum_G0  = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
    pre.src_lum_ion = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
    if(!pre.src_lum_G0 || !pre.src_lum_ion) {printf("gpu_moment_refresh: CHIMES alloc failed\n"); endrun(913302); return 1;}
    memset(pre.src_lum_G0,  0, szc);
    memset(pre.src_lum_ion, 0, szc);
#endif
    for(int p = 0; p < N; p++) {
        if(P[p].Mass <= 0) {continue;}
        double lum[N_RT_FREQ_BINS];
#ifdef CHIMES_STELLAR_FLUXES
        double lum_G0[CHIMES_LOCAL_UV_NBINS], lum_ion[CHIMES_LOCAL_UV_NBINS];
        int active_check = rt_get_source_luminosity_chimes(p, 1, lum, lum_G0, lum_ion, P, CellP);
#else
        int active_check = rt_get_source_luminosity(p, 1, lum, P, CellP);
#endif
        if(active_check) {
            int kf;
            for(kf = 0; kf < N_RT_FREQ_BINS; kf++) {
                pre.src_lum[(long)p * N_RT_FREQ_BINS + kf] = (MyFloat) lum[kf];
            }
#ifdef CHIMES_STELLAR_FLUXES
            for(kf = 0; kf < CHIMES_LOCAL_UV_NBINS; kf++) {
                pre.src_lum_G0[(long)p * CHIMES_LOCAL_UV_NBINS + kf]  = lum_G0[kf];
                pre.src_lum_ion[(long)p * CHIMES_LOCAL_UV_NBINS + kf] = lum_ion[kf];
            }
#endif
        }
    }
#endif

#ifdef SINK_PHOTONMOMENTUM
    long sz_lum  = (long)N * sizeof(MyFloat);
    long sz_ang  = (long)N * sizeof(Vec3<MyFloat>);
    pre.bh_lum   = (MyFloat *)       Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_lum);
    pre.bh_angle = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_ang);
    if(!pre.bh_lum || !pre.bh_angle) {printf("gpu_moment_refresh: bh_lum alloc failed\n"); endrun(913303); return 1;}
    memset(pre.bh_lum,   0, sz_lum);
    memset(pre.bh_angle, 0, sz_ang);
    for(int p = 0; p < N; p++) {
        if(P[p].Type != 5 || P[p].Mass <= 0) {continue;}
        if(P[p].DensityAroundParticle <= 0 || P[p].Sink_Mdot <= 0) {continue;}
        double bhlum = sink_lum_bol(P[p].Sink_Mdot, P[p].Sink_Mass, p);
        pre.bh_lum[p] = (MyFloat) bhlum;
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
        pre.bh_angle[p] = P[p].Sink_Specific_AngMom;
#else
        pre.bh_angle[p] = P[p].GradRho;
#endif
    }
#endif

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    long sz_cr = (long)N * sizeof(MyFloat);
    pre.cr_inject = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_cr);
    if(!pre.cr_inject) {printf("gpu_moment_refresh: cr_inject alloc failed\n"); endrun(913304); return 1;}
    memset(pre.cr_inject, 0, sz_cr);
    for(int p = 0; p < N; p++) {
        if(P[p].Type != 0 || P[p].Mass <= 0) {continue;}
        pre.cr_inject[p] = (MyFloat) cr_get_source_injection_rate(p, P, CellP);
    }
#endif
    return 0;
}

static void precompute_free_(precomputed_t& pre)
{
#ifdef RT_USE_GRAVTREE
    if(pre.src_lum) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pre.src_lum); pre.src_lum = NULL;}
#ifdef CHIMES_STELLAR_FLUXES
    if(pre.src_lum_G0)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pre.src_lum_G0);  pre.src_lum_G0  = NULL;}
    if(pre.src_lum_ion) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pre.src_lum_ion); pre.src_lum_ion = NULL;}
#endif
#endif
#ifdef SINK_PHOTONMOMENTUM
    if(pre.bh_lum)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pre.bh_lum);   pre.bh_lum   = NULL;}
    if(pre.bh_angle) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pre.bh_angle); pre.bh_angle = NULL;}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    if(pre.cr_inject) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(pre.cr_inject); pre.cr_inject = NULL;}
#endif
}
} /* anonymous namespace */

/* ------------------------------------------------------------------ */
/* Father[i] mirror in SharedSpace. Allocated and populated once per   */
/* gpu_moment_refresh call.                                             */
/* ------------------------------------------------------------------ */
static int *father_mirror_alloc_(int N)
{
    int *fmirror = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((long)N * sizeof(int));
    if(!fmirror) {printf("gpu_moment_refresh: Father mirror alloc failed\n"); endrun(913305); return NULL;}
    for(int i = 0; i < N; i++) {fmirror[i] = Father[i];}
    return fmirror;
}

/* =================================================================== */
/* Phase 6.7.0 refactor: bundle scratch Views + factor per-payload     */
/* reductions into KOKKOS_INLINE_FUNCTION helpers.  Phase 6.7c's       */
/* gpu_moment_refresh_topnodes() reuses the exact same helpers for the */
/* topnode-subtree ancestor re-sum, avoiding ~250 lines of duplicated  */
/* #ifdef-guarded payload logic.  All behavior here is bit-identical   */
/* to the prior inline implementation; this is purely a SoT refactor.  */
/* =================================================================== */
namespace {

using MrExSpace  = Kokkos::DefaultExecutionSpace;
using MrMemSpace = MrExSpace::memory_space;

struct mr_scratch_t {
    Kokkos::View<int*,           MrMemSpace> pending;
    Kokkos::View<MyGravFloat*,   MrMemSpace> mass;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> s;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> vs;
    Kokkos::View<long*,          MrMemSpace> Npart;
    Kokkos::View<MyGravFloat*,   MrMemSpace> hmax;
    Kokkos::View<MyGravFloat*,   MrMemSpace> vmax;
    Kokkos::View<MyGravFloat*,   MrMemSpace> divVmax;
    Kokkos::View<MyGravFloat*,   MrMemSpace> maxsoft;
    Kokkos::View<unsigned int*,  MrMemSpace> bitflags;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    Kokkos::View<MyGravFloat*,   MrMemSpace> gasmass;
#endif
#ifdef RT_USE_GRAVTREE
    Kokkos::View<MyGravFloat*,   MrMemSpace> stellar_lum;  /* [n * N_RT_FREQ_BINS] */
#ifdef CHIMES_STELLAR_FLUXES
    Kokkos::View<double*,        MrMemSpace> chimes_G0;    /* [n * CHIMES_LOCAL_UV_NBINS] */
    Kokkos::View<double*,        MrMemSpace> chimes_ion;
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> rt_s;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> rt_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
    Kokkos::View<MyGravFloat*,   MrMemSpace> sink_lum;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> sink_lum_grad;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    Kokkos::View<MyGravFloat*,   MrMemSpace> cr_inject;
#endif
#ifdef SINK_CALC_DISTANCES
    Kokkos::View<MyGravFloat*,   MrMemSpace> sink_mass;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Kokkos::View<int*,           MrMemSpace> N_SINK;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> sink_vel;
#endif
#if defined(SPECIAL_POINT_MOTION)
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> sink_acc;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    Kokkos::View<MyGravFloat*,   MrMemSpace> max_fbvel;
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    Kokkos::View<MyGravFloat*,   MrMemSpace> tidal;        /* [n * 6] */
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Kokkos::View<MyGravFloat*,   MrMemSpace> mass_dm;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> s_dm;
    Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace> vs_dm;
#endif
};

/* Allocate all scratch Views sized to n internal nodes. */
static void mr_scratch_alloc_(mr_scratch_t& scr, int n)
{
    scr.pending  = Kokkos::View<int*,         MrMemSpace>("mr_pending",  n);
    scr.mass     = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_mass",     n);
    scr.s        = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_s",  n);
    scr.vs       = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_vs", n);
    scr.Npart    = Kokkos::View<long*,        MrMemSpace>("mr_Npart",    n);
    scr.hmax     = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_hmax",     n);
    scr.vmax     = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_vmax",     n);
    scr.divVmax  = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_divVmax",  n);
    scr.maxsoft  = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_maxsoft",  n);
    scr.bitflags = Kokkos::View<unsigned int*,MrMemSpace>("mr_bitflags", n);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    scr.gasmass  = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_gasmass",  n);
#endif
#ifdef RT_USE_GRAVTREE
    scr.stellar_lum = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_stellar_lum", (long)n * N_RT_FREQ_BINS);
#ifdef CHIMES_STELLAR_FLUXES
    scr.chimes_G0  = Kokkos::View<double*, MrMemSpace>("mr_chimes_G0",  (long)n * CHIMES_LOCAL_UV_NBINS);
    scr.chimes_ion = Kokkos::View<double*, MrMemSpace>("mr_chimes_ion", (long)n * CHIMES_LOCAL_UV_NBINS);
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    scr.rt_s  = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_rt_s",  n);
    scr.rt_vs = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_rt_vs", n);
#endif
#ifdef SINK_PHOTONMOMENTUM
    scr.sink_lum      = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_sink_lum",      n);
    scr.sink_lum_grad = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_sink_lum_grad", n);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    scr.cr_inject = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_cr_inj", n);
#endif
#ifdef SINK_CALC_DISTANCES
    scr.sink_mass = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_sink_mass", n);
    scr.sink_pos  = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_sink_pos",  n);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    scr.N_SINK   = Kokkos::View<int*,         MrMemSpace>("mr_N_SINK",   n);
    scr.sink_vel = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_sink_vel", n);
#endif
#if defined(SPECIAL_POINT_MOTION)
    scr.sink_acc = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_sink_acc", n);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    scr.max_fbvel = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_max_fbvel", n);
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    scr.tidal = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_tidal", (long)n * 6);
#endif
#ifdef DM_SCALARFIELD_SCREENING
    scr.mass_dm = Kokkos::View<MyGravFloat*, MrMemSpace>("mr_mass_dm", n);
    scr.s_dm    = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_s_dm", n);
    scr.vs_dm   = Kokkos::View<Vec3<MyGravFloat>*, MrMemSpace>("mr_vs_dm", n);
#endif
}

/* Adapters between this venue's scratch Views and the shared moment kernel's POD interface
 * (gravtree_moment_kernel.h).  mr_ref_ points a moment_node_ref at one node's scratch slots (the
 * array payloads as their per-node base); mr_child_accum_ reads a settled child slot into a by-value
 * accumulator for the bottom-up propagate.  Pure pointer/loads — bit-identical to the prior inline
 * scratch accesses. */
KOKKOS_INLINE_FUNCTION
static moment_node_ref<MyGravFloat> mr_ref_(const mr_scratch_t& scr, int k)
{
    moment_node_ref<MyGravFloat> r;
    r.mass     = &scr.mass(k);
    r.s        = &scr.s(k);
    r.vs       = &scr.vs(k);
    r.Npart    = &scr.Npart(k);
    r.hmax     = &scr.hmax(k);
    r.vmax     = &scr.vmax(k);
    r.divVmax  = &scr.divVmax(k);
    r.maxsoft  = &scr.maxsoft(k);
    r.bitflags = &scr.bitflags(k);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    r.gasmass  = &scr.gasmass(k);
#endif
#ifdef RT_USE_GRAVTREE
    r.stellar_lum = &scr.stellar_lum((long)k * N_RT_FREQ_BINS);
#ifdef CHIMES_STELLAR_FLUXES
    r.chimes_G0  = &scr.chimes_G0 ((long)k * CHIMES_LOCAL_UV_NBINS);
    r.chimes_ion = &scr.chimes_ion((long)k * CHIMES_LOCAL_UV_NBINS);
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    r.rt_s  = &scr.rt_s(k);
    r.rt_vs = &scr.rt_vs(k);
#endif
#ifdef SINK_PHOTONMOMENTUM
    r.sink_lum      = &scr.sink_lum(k);
    r.sink_lum_grad = &scr.sink_lum_grad(k);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    r.cr_inject = &scr.cr_inject(k);
#endif
#ifdef SINK_CALC_DISTANCES
    r.sink_mass = &scr.sink_mass(k);
    r.sink_pos  = &scr.sink_pos(k);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    r.N_SINK   = &scr.N_SINK(k);
    r.sink_vel = &scr.sink_vel(k);
#endif
#if defined(SPECIAL_POINT_MOTION)
    r.sink_acc = &scr.sink_acc(k);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    r.max_fbvel = &scr.max_fbvel(k);
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    r.tidal = &scr.tidal((long)k * 6);
#endif
#ifdef DM_SCALARFIELD_SCREENING
    r.mass_dm = &scr.mass_dm(k);
    r.s_dm    = &scr.s_dm(k);
    r.vs_dm   = &scr.vs_dm(k);
#endif
    return r;
}

KOKKOS_INLINE_FUNCTION
static moment_node_accum<MyGravFloat> mr_child_accum_(const mr_scratch_t& scr, int curr)
{
    moment_node_accum<MyGravFloat> c = {};
    c.mass    = scr.mass(curr);
    c.s       = scr.s(curr);
    c.vs      = scr.vs(curr);
    c.Npart   = scr.Npart(curr);
    c.hmax    = scr.hmax(curr);
    c.vmax    = scr.vmax(curr);
    c.divVmax = scr.divVmax(curr);
    c.maxsoft = scr.maxsoft(curr);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    c.gasmass = scr.gasmass(curr);
#endif
#ifdef RT_USE_GRAVTREE
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {c.stellar_lum[b] = scr.stellar_lum((long)curr * N_RT_FREQ_BINS + b);}
#ifdef CHIMES_STELLAR_FLUXES
    for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
        c.chimes_G0 [b] = scr.chimes_G0 ((long)curr * CHIMES_LOCAL_UV_NBINS + b);
        c.chimes_ion[b] = scr.chimes_ion((long)curr * CHIMES_LOCAL_UV_NBINS + b);
    }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    c.rt_s  = scr.rt_s(curr);
    c.rt_vs = scr.rt_vs(curr);
#endif
#ifdef SINK_PHOTONMOMENTUM
    c.sink_lum      = scr.sink_lum(curr);
    c.sink_lum_grad = scr.sink_lum_grad(curr);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    c.cr_inject = scr.cr_inject(curr);
#endif
#ifdef SINK_CALC_DISTANCES
    c.sink_mass = scr.sink_mass(curr);
    c.sink_pos  = scr.sink_pos(curr);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    c.N_SINK   = scr.N_SINK(curr);
    c.sink_vel = scr.sink_vel(curr);
#endif
#if defined(SPECIAL_POINT_MOTION)
    c.sink_acc = scr.sink_acc(curr);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    c.max_fbvel = scr.max_fbvel(curr);
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    for(int kk = 0; kk < 6; kk++) {c.tidal[kk] = scr.tidal((long)curr * 6 + kk);}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    c.mass_dm = scr.mass_dm(curr);
    c.s_dm    = scr.s_dm(curr);
    c.vs_dm   = scr.vs_dm(curr);
#endif
    return c;
}

/* Helper: zero per-node payload accumulators at slot k.  Bitflags
 * preserves the topology bits (TOPLEVEL/INTERNAL_TOPLEVEL/DEPENDS_
 * ON_LOCAL_ELEMENT) and clears the rest, matching the CPU saved_
 * bitflags mask in force_refresh_node_moments (forcetree.cc:3841). */
KOKKOS_INLINE_FUNCTION
static void mr_zero_payloads_(const mr_scratch_t& scr, int k, unsigned int saved_bitflags)
{
    moment_accum_zero<MyGravFloat>(mr_ref_(scr, k), saved_bitflags);
}

/* Helper: bottom-up child→parent propagation.  Atomically adds curr's
 * accumulated moments + max-reduces hmax/vmax/divVmax/maxsoft into kp.
 * Used by Kernel 4 (walk-up); reused by Phase 6.7c for topnode-subtree
 * ancestor re-sum after MPI pseudodata exchange. */
KOKKOS_INLINE_FUNCTION
static void mr_propagate_to_parent_(const mr_scratch_t& scr, int curr, int kp)
{
    moment_accum_add_child_raw<moment_atomic_ops, MyGravFloat>(mr_ref_(scr, kp), mr_child_accum_(scr, curr));
}

/* Finalize a node: turn the mass-weighted sums into stored normalized payloads + patch
 * BITFLAG_MULTIPLEPARTICLES.  `center` is the node's geometric center (COM fallback when the weight
 * is zero).  Delegates to the shared moment_finalize (gravtree_moment_kernel.h); the scratch
 * bitflags carry the topology bits, so the multiple-particles bit is patched in place via the ref. */
KOKKOS_INLINE_FUNCTION
static void mr_normalize_payloads_(const mr_scratch_t& scr, int k, const Vec3<MyFloat>& center)
{
    moment_finalize<MyGravFloat>(mr_ref_(scr, k),
                                 Vec3<MyGravFloat>{(MyGravFloat) center[0],
                                                   (MyGravFloat) center[1],
                                                   (MyGravFloat) center[2]});
}

/* Pending-counter init launcher.  For each k in [0..n_iter), atomically
 * increments pending[father(k) - MaxPart] when father falls within
 * [range_lo, range_hi).  Phase 6.7c passes a restricted iteration
 * range + filter for the topnode-subtree variant. */
static void mr_pending_init_launch_(Kokkos::View<int*, MrMemSpace> pending,
                                     int *father_soa, int MaxPart,
                                     int n_iter, int range_lo, int range_hi)
{
    Kokkos::parallel_for("mr_pending_init", n_iter, KOKKOS_LAMBDA(int k) {
        int f = father_soa[k];
        if(f >= range_lo && f < range_hi) {
            Kokkos::atomic_inc(&pending(f - MaxPart));
        }
    });
}

} /* anonymous namespace (Phase 6.7.0 helpers) */

/* =================================================================== */
/*  Main entry: gpu_moment_refresh                                      */
/* =================================================================== */
extern "C" int gpu_moment_refresh(int active_root_node)
{
    (void) active_root_node; /* Phase 9 hook; unused for now */
    if(Numnodestree <= 0) {return 0;}
    /* Defensive idempotent sync of AllDevice before any device-side All.* read. */
    GIZMO_GPU_ENSURE_ALL_FRESH();

    int n          = Numnodestree;       /* number of internal nodes [0..n) in SoA */
    int MaxPart    = All.MaxPart;
    int N          = NumPart;

    /* ---------------- 1. arenas / SoA / per-particle precompute -------- */
    gpu_particles_arena_set_site("gpu_moment_refresh(pre-precompute)");
    gpu_particles_arena_acquire(N, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {printf("gpu_moment_refresh: arena P_dev null\n"); return 1;}

    /* Tree SoA storage must already be allocated; the GPU build pipeline
     * (gpu_nextnode_backup_suns + topology emit/finalize kernels) populated
     * topology fields (sibling, nextnode, father, center, len, bitflags)
     * directly in the SoA before this kernel runs.  We overwrite the
     * moments here. */
    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_moment_refresh: SoA null\n"); return 1;}

    precomputed_t pre = {};
    if(precompute_alloc_(pre, N) != 0) {precompute_free_(pre); return 1;}   /* soft bad-stop: no kernel launch on NULL precompute buffers; caller (forcetree) drains at next poll */
    int *fmirror = father_mirror_alloc_(N);
    if(!fmirror) {precompute_free_(pre); return 1;}   /* soft bad-stop: no kernel launch on NULL Father mirror */

    /* ---------------- 2. device-local scratch (Phase 6.7.0) ------------ */
    /* Bundled into mr_scratch_t so the per-payload helpers (zero /
     * propagate / normalize / pending_init) above can be reused unchanged
     * by gpu_moment_refresh_topnodes() in Phase 6.7c. */
    mr_scratch_t scr;
    mr_scratch_alloc_(scr, n);

    /* SoA pointers captured into the lambdas (raw pointers; SharedSpace is
     * accessible from device). The SoA `father`/`bitflags` carry topology
     * already seeded by gpu_gravity_tree_acquire above. */
    int          *father_soa   = soa->father;
    unsigned int *bitflags_soa = soa->bitflags;
    Vec3<MyFloat> *center_soa  = soa->center;

    /* ---------------- Kernel 1: zero scratch + pending ----------------- */
    /* pending is initialized to 1 (resident-thread slot); Kernel 2 adds
     * one more for each internal-node child.  In Kernel 4 each thread
     * claims its starting node with atomic_fetch_sub; only the last to
     * arrive (prev==1) proceeds upward.  The +1 prevents the race where
     * a late resident thread and the bottom-up walk both see pending==0
     * and double-count the node. */
    Kokkos::deep_copy(scr.pending, 1);
    Kokkos::parallel_for("mr_zero_payloads", n, KOKKOS_LAMBDA(int k) {
        mr_zero_payloads_(scr, k, bitflags_soa[k]);
    });

    /* ---------------- Kernel 2: pending counter init ------------------- */
    mr_pending_init_launch_(scr.pending, father_soa, MaxPart, n,
                             MaxPart, MaxPart + n);

    /* ---------------- Kernel 3: particle pass -------------------------- */
    double maxKR = All.MaxKernelRadius;
#ifdef SINK_PHOTONMOMENTUM
    MyFloat       *bh_lum_   = pre.bh_lum;
    Vec3<MyFloat> *bh_angle_ = pre.bh_angle;
#endif
#ifdef RT_USE_GRAVTREE
    MyFloat *src_lum_ = pre.src_lum;
#ifdef CHIMES_STELLAR_FLUXES
    double  *src_lum_G0_  = pre.src_lum_G0;
    double  *src_lum_ion_ = pre.src_lum_ion;
#endif
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat *cr_inj_ = pre.cr_inject;
#endif
    Kokkos::parallel_for("mr_part_accum", N, KOKKOS_LAMBDA(int i) {
        int f = fmirror[i];
        if(f < MaxPart || f >= MaxPart + n) {return;}
        int k = f - MaxPart;
        const struct particle_data *pa = &P_dev[i];

        /* Load this leaf's fields + per-particle precomputes into the shared kernel's POD (native
         * double; weighted products are formed at double precision inside the helper).  The gates and
         * formulas live in moment_source_from_particle. */
        moment_particle_src<MyGravFloat> ps = {};
        ps.mass = pa->Mass;
        for(int kk = 0; kk < 3; kk++) {ps.pos[kk] = pa->Pos[kk]; ps.vel[kk] = pa->Vel[kk];}
        ps.type              = (int) pa->Type;
        ps.kernel_radius     = pa->KernelRadius;
        ps.max_kernel_radius = maxKR;
        ps.force_softening   = gpu_force_softening_kernelradius(P_dev, i);
        ps.particle_divvel   = pa->Particle_DivVel;
#if defined(SINK_ALPHADISK_ACCRETION) && defined(RT_USE_TREECOL_FOR_NH)
        ps.sink_mass_reservoir = pa->Sink_Mass_Reservoir;
#endif
#ifdef RT_USE_GRAVTREE
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {ps.src_lum[b] = src_lum_[(long)i * N_RT_FREQ_BINS + b];}
#ifdef CHIMES_STELLAR_FLUXES
        for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
            ps.src_lum_G0 [b] = src_lum_G0_ [(long)i * CHIMES_LOCAL_UV_NBINS + b];
            ps.src_lum_ion[b] = src_lum_ion_[(long)i * CHIMES_LOCAL_UV_NBINS + b];
        }
#endif
#endif
#ifdef SINK_PHOTONMOMENTUM
        ps.bh_lum = bh_lum_[i];
        for(int kk = 0; kk < 3; kk++) {ps.bh_angle[kk] = bh_angle_[i][kk];}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        ps.cr_inject = cr_inj_[i];
#endif
#if defined(SPECIAL_POINT_MOTION)
        for(int kk = 0; kk < 3; kk++) {ps.acc_prevstep[kk] = pa->Acc_Total_PrevStep[kk];}
#endif
#if defined(SINK_CALC_DISTANCES) && defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
        ps.max_feedback_vel = pa->MaxFeedbackVel;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        for(int kk = 0; kk < 6; kk++) {ps.tidal_prevstep[kk] = pa->tidal_tensorps_prevstep.data[kk];}
#endif

        moment_accum_add_particle<moment_atomic_ops, MyGravFloat>(mr_ref_(scr, k), ps);
    });

    /* ---------------- Kernel 4: bottom-up walk via father chain -------- */
    Kokkos::parallel_for("mr_walk_up", n, KOKKOS_LAMBDA(int k0) {
        if(Kokkos::atomic_fetch_sub(&scr.pending(k0), 1) != 1) {return;}
        int curr = k0;
        while(true) {
            int f = father_soa[curr];
            if(f < MaxPart || f >= MaxPart + n) {return;}
            int kp = f - MaxPart;
            mr_propagate_to_parent_(scr, curr, kp);
            int prev = Kokkos::atomic_fetch_sub(&scr.pending(kp), 1);
            if(prev != 1) {return;}
            curr = kp;
        }
    });

    /* ---------------- Kernel 5: normalize ------------------------------ */
    Kokkos::parallel_for("mr_normalize", n, KOKKOS_LAMBDA(int k) {
        mr_normalize_payloads_(scr, k, center_soa[k]);
    });

    Kokkos::fence();
    gizmo_gpu_check_last_error("mr_moment_refresh", n);

    /* ---------------- Bulk deep-copy scratch → SharedSpace SoA -------- */
    /* Wrap raw SoA pointers in unmanaged Views and deep_copy. */
    using UV_unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ShSpace = Kokkos::View<MyGravFloat*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShVec3  = Kokkos::View<Vec3<MyGravFloat>*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShU32   = Kokkos::View<unsigned int*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShLong  = Kokkos::View<long*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShDbl   = Kokkos::View<double*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShInt   = Kokkos::View<int*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;

    Kokkos::deep_copy(ShSpace(soa->mass,    n), scr.mass);
    Kokkos::deep_copy(ShVec3 (soa->s,       n), scr.s);
    Kokkos::deep_copy(ShVec3 (soa->node_vs, n), scr.vs);
    Kokkos::deep_copy(ShLong (soa->N_part,  n), scr.Npart);
    Kokkos::deep_copy(ShSpace(soa->hmax,    n), scr.hmax);
    Kokkos::deep_copy(ShSpace(soa->vmax,    n), scr.vmax);
    Kokkos::deep_copy(ShSpace(soa->divVmax, n), scr.divVmax);
    Kokkos::deep_copy(ShSpace(soa->maxsoft, n), scr.maxsoft);
    Kokkos::deep_copy(ShU32  (soa->bitflags,n), scr.bitflags);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    Kokkos::deep_copy(ShSpace(soa->gasmass, n), scr.gasmass);
#endif
#ifdef RT_USE_GRAVTREE
    Kokkos::deep_copy(ShSpace(soa->stellar_lum, (long)n * N_RT_FREQ_BINS), scr.stellar_lum);
#ifdef CHIMES_STELLAR_FLUXES
    Kokkos::deep_copy(ShDbl(soa->chimes_stellar_lum_G0,  (long)n * CHIMES_LOCAL_UV_NBINS), scr.chimes_G0);
    Kokkos::deep_copy(ShDbl(soa->chimes_stellar_lum_ion, (long)n * CHIMES_LOCAL_UV_NBINS), scr.chimes_ion);
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Kokkos::deep_copy(ShVec3(soa->rt_source_lum_s,  n), scr.rt_s);
    Kokkos::deep_copy(ShVec3(soa->rt_source_lum_vs, n), scr.rt_vs);
#endif
#ifdef SINK_PHOTONMOMENTUM
    Kokkos::deep_copy(ShSpace(soa->sink_lum,      n), scr.sink_lum);
    Kokkos::deep_copy(ShVec3 (soa->sink_lum_grad, n), scr.sink_lum_grad);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    Kokkos::deep_copy(ShSpace(soa->cr_injection, n), scr.cr_inject);
#endif
#ifdef SINK_CALC_DISTANCES
    Kokkos::deep_copy(ShSpace(soa->sink_mass, n), scr.sink_mass);
    Kokkos::deep_copy(ShVec3 (soa->sink_pos,  n), scr.sink_pos);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Kokkos::deep_copy(ShInt  (soa->N_SINK,    n), scr.N_SINK);
    Kokkos::deep_copy(ShVec3 (soa->sink_vel,  n), scr.sink_vel);
#endif
#if defined(SPECIAL_POINT_MOTION)
    Kokkos::deep_copy(ShVec3 (soa->sink_acc,  n), scr.sink_acc);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    Kokkos::deep_copy(ShSpace(soa->MaxFeedbackVel, n), scr.max_fbvel);
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    Kokkos::deep_copy(ShSpace(soa->tidal_tensorps, (long)n * 6), scr.tidal);
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Kokkos::deep_copy(ShSpace(soa->mass_dm, n), scr.mass_dm);
    Kokkos::deep_copy(ShVec3 (soa->s_dm,    n), scr.s_dm);
    Kokkos::deep_copy(ShVec3 (soa->vs_dm,   n), scr.vs_dm);
#endif

    Kokkos::fence();

    /* ---------------- AoS write-back ---------------------------------- */
    gpu_moment_writeback_to_aos(n);

    /* ---------------- cleanup ----------------------------------------- */
    if(fmirror) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(fmirror);}
    precompute_free_(pre);

    return 0;
}

/* =================================================================== */
/* Bulk SoA → Nodes[]/Extnodes[] AoS write-back.  Runs on host         */
/* (host-side mutation of the AoS arrays so the CPU pseudo path picks  */
/* up the new moments).  SharedSpace pages may need a Kokkos::fence    */
/* before this; callers are responsible.                                */
/* =================================================================== */
extern "C" void gpu_moment_writeback_to_aos(int n)
{
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {return;}
    int MaxPart = All.MaxPart;

    for(int k = 0; k < n; k++) {
        int no = MaxPart + k;
        Nodes[no].u.d.mass     = (MyFloat) soa->mass[k];
        Nodes[no].u.d.s        = Vec3<MyFloat>{(MyFloat) soa->s[k][0],
                                                (MyFloat) soa->s[k][1],
                                                (MyFloat) soa->s[k][2]};
        Nodes[no].u.d.bitflags = soa->bitflags[k];
        Nodes[no].N_part       = soa->N_part[k];
        Nodes[no].maxsoft      = (MyFloat) soa->maxsoft[k];
        /* GravCost / Ti_current: matches CPU step-1 reset, but we only
         * touch them in the same way force_refresh_node_moments did
         * before our dispatch was wired in. They are reset on the host
         * before this routine is called (see forcetree.cc step 1). */
        Extnodes[no].vs       = Vec3<MyFloat>{(MyFloat) soa->node_vs[k][0],
                                               (MyFloat) soa->node_vs[k][1],
                                               (MyFloat) soa->node_vs[k][2]};
        Extnodes[no].hmax     = (MyFloat) soa->hmax[k];
        Extnodes[no].vmax     = (MyFloat) soa->vmax[k];
        Extnodes[no].divVmax  = (MyFloat) soa->divVmax[k];
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        Nodes[no].gasmass     = (MyFloat) soa->gasmass[k];
#endif
#ifdef RT_USE_GRAVTREE
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            Nodes[no].stellar_lum[b] = (MyFloat) soa->stellar_lum[(long)k * N_RT_FREQ_BINS + b];
        }
#ifdef CHIMES_STELLAR_FLUXES
        for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
            Nodes[no].chimes_stellar_lum_G0 [b] = soa->chimes_stellar_lum_G0 [(long)k * CHIMES_LOCAL_UV_NBINS + b];
            Nodes[no].chimes_stellar_lum_ion[b] = soa->chimes_stellar_lum_ion[(long)k * CHIMES_LOCAL_UV_NBINS + b];
        }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        Nodes[no].rt_source_lum_s   = Vec3<MyFloat>{(MyFloat) soa->rt_source_lum_s [k][0],
                                                     (MyFloat) soa->rt_source_lum_s [k][1],
                                                     (MyFloat) soa->rt_source_lum_s [k][2]};
        Extnodes[no].rt_source_lum_vs = Vec3<MyFloat>{(MyFloat) soa->rt_source_lum_vs[k][0],
                                                       (MyFloat) soa->rt_source_lum_vs[k][1],
                                                       (MyFloat) soa->rt_source_lum_vs[k][2]};
#endif
#ifdef SINK_PHOTONMOMENTUM
        Nodes[no].sink_lum      = (MyFloat) soa->sink_lum[k];
        Nodes[no].sink_lum_grad = Vec3<MyFloat>{(MyFloat) soa->sink_lum_grad[k][0],
                                                 (MyFloat) soa->sink_lum_grad[k][1],
                                                 (MyFloat) soa->sink_lum_grad[k][2]};
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        Nodes[no].cr_injection = (MyFloat) soa->cr_injection[k];
#endif
#ifdef SINK_CALC_DISTANCES
        Nodes[no].sink_mass = (MyFloat) soa->sink_mass[k];
        Nodes[no].sink_pos  = Vec3<MyFloat>{(MyFloat) soa->sink_pos[k][0],
                                             (MyFloat) soa->sink_pos[k][1],
                                             (MyFloat) soa->sink_pos[k][2]};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
        Nodes[no].sink_vel  = Vec3<MyFloat>{(MyFloat) soa->sink_vel[k][0],
                                             (MyFloat) soa->sink_vel[k][1],
                                             (MyFloat) soa->sink_vel[k][2]};
#endif
#if defined(SPECIAL_POINT_MOTION)
        Nodes[no].sink_acc  = Vec3<MyFloat>{(MyFloat) soa->sink_acc[k][0],
                                             (MyFloat) soa->sink_acc[k][1],
                                             (MyFloat) soa->sink_acc[k][2]};
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        Nodes[no].N_SINK    = soa->N_SINK[k];
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
        Nodes[no].MaxFeedbackVel = (MyFloat) soa->MaxFeedbackVel[k];
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        for(int kk = 0; kk < 6; kk++) {
            Nodes[no].tidal_tensorps_prevstep.data[kk] = (MyFloat) soa->tidal_tensorps[(long)k * 6 + kk];
        }
#endif
#ifdef DM_SCALARFIELD_SCREENING
        Nodes[no].mass_dm = (MyFloat) soa->mass_dm[k];
        Nodes[no].s_dm    = Vec3<MyFloat>{(MyFloat) soa->s_dm[k][0],
                                            (MyFloat) soa->s_dm[k][1],
                                            (MyFloat) soa->s_dm[k][2]};
        Extnodes[no].vs_dm = Vec3<MyFloat>{(MyFloat) soa->vs_dm[k][0],
                                            (MyFloat) soa->vs_dm[k][1],
                                            (MyFloat) soa->vs_dm[k][2]};
#endif
    }
}


