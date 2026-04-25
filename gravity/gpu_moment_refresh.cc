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

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "forcetree.h"

#ifdef RT_USE_GRAVTREE
#include "../radiation/rt_functions.h"
#endif

#if defined(OPENMP_GPU_OFFLOAD)

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

/* ------------------------------------------------------------------ */
/* GPU-callable mirror of ForceSoftening_KernelRadius (a strict       */
/* subset — CPU implementation has a few extra branches we don't hit  */
/* in the moment-accumulation context). For nodes the legacy CPU code */
/* uses ForceSoftening_KernelRadius(i) so we mirror exactly the gpu_  */
/* gravtree helper below to keep parity.                              */
/* ------------------------------------------------------------------ */
KOKKOS_INLINE_FUNCTION static double
gpu_force_softening_kernelradius(const struct particle_data *Pp, int p)
{
#ifdef GALSF_MERGER_STARCLUSTER_PARTICLES
    if(Pp[p].Type == 4) {return Pp[p].StarParticleEffectiveSize;}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if((1 << Pp[p].Type) & (ADAPTIVE_GRAVSOFT_FORALL)) {return Pp[p].AGS_KernelRadius;}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(SELFGRAVITY_OFF)
    if(Pp[p].Type == 0) {
#if defined(ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT)
        double cap = ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT / All.cf_atime;
        return (Pp[p].KernelRadius < cap) ? Pp[p].KernelRadius : cap;
#else
        return Pp[p].KernelRadius;
#endif
    }
#endif
    return All.ForceSoftening[Pp[p].Type];
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

static void precompute_alloc_(precomputed_t& pre, int N)
{
#ifdef RT_USE_GRAVTREE
    long sz = (long)N * N_RT_FREQ_BINS * sizeof(MyFloat);
    pre.src_lum = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
    if(!pre.src_lum) {printf("gpu_moment_refresh: src_lum alloc failed\n"); endrun(913301);}
    memset(pre.src_lum, 0, sz);
#ifdef CHIMES_STELLAR_FLUXES
    long szc = (long)N * CHIMES_LOCAL_UV_NBINS * sizeof(double);
    pre.src_lum_G0  = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
    pre.src_lum_ion = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
    if(!pre.src_lum_G0 || !pre.src_lum_ion) {printf("gpu_moment_refresh: CHIMES alloc failed\n"); endrun(913302);}
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
    if(!pre.bh_lum || !pre.bh_angle) {printf("gpu_moment_refresh: bh_lum alloc failed\n"); endrun(913303);}
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
    if(!pre.cr_inject) {printf("gpu_moment_refresh: cr_inject alloc failed\n"); endrun(913304);}
    memset(pre.cr_inject, 0, sz_cr);
    for(int p = 0; p < N; p++) {
        if(P[p].Type != 0 || P[p].Mass <= 0) {continue;}
        pre.cr_inject[p] = (MyFloat) cr_get_source_injection_rate(p, P, CellP);
    }
#endif
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
    if(!fmirror) {printf("gpu_moment_refresh: Father mirror alloc failed\n"); endrun(913305);}
    for(int i = 0; i < N; i++) {fmirror[i] = Father[i];}
    return fmirror;
}

/* =================================================================== */
/*  Main entry: gpu_moment_refresh                                      */
/* =================================================================== */
extern "C" int gpu_moment_refresh(int active_root_node)
{
    (void) active_root_node; /* Phase 9 hook; unused for now */
    if(Numnodestree <= 0) {return 0;}

    int n          = Numnodestree;       /* number of internal nodes [0..n) in SoA */
    int MaxPart    = All.MaxPart;
    int N          = NumPart;

    /* ---------------- 1. arenas / SoA / per-particle precompute -------- */
    gpu_particles_arena_acquire(N, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {printf("gpu_moment_refresh: arena P_dev null\n"); return 1;}

    /* Tree SoA must already be allocated/seeded with topology (CPU build
     * just ran, mark_all_dirty was called by force_treebuild — the next
     * acquire reseeds everything from AoS). Topology fields (sibling,
     * nextnode, father, center, len, bitflags) come from the CPU-built
     * Nodes[]; we overwrite the moments. */
    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_moment_refresh: SoA null\n"); return 1;}

    precomputed_t pre = {};
    precompute_alloc_(pre, N);
    int *fmirror = father_mirror_alloc_(N);

    /* ---------------- 2. device-local scratch Views -------------------- */
    using ExSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExSpace::memory_space;
    Kokkos::View<int*, MemSpace>             pending("mr_pending", n);
    Kokkos::View<MyGravFloat*, MemSpace>     scr_mass("mr_mass", n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_s("mr_s", n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_vs("mr_vs", n);
    Kokkos::View<long*, MemSpace>            scr_Npart("mr_Npart", n);
    Kokkos::View<MyGravFloat*, MemSpace>     scr_hmax("mr_hmax", n);
    Kokkos::View<MyGravFloat*, MemSpace>     scr_vmax("mr_vmax", n);
    Kokkos::View<MyGravFloat*, MemSpace>     scr_divVmax("mr_divVmax", n);
    Kokkos::View<MyGravFloat*, MemSpace>     scr_maxsoft("mr_maxsoft", n);
    Kokkos::View<unsigned int*, MemSpace>    scr_bitflags("mr_bitflags", n);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    Kokkos::View<MyGravFloat*, MemSpace>     scr_gasmass("mr_gasmass", n);
#endif
#ifdef RT_USE_GRAVTREE
    Kokkos::View<MyGravFloat*, MemSpace>     scr_stellar_lum("mr_stellar_lum", (long)n * N_RT_FREQ_BINS);
#ifdef CHIMES_STELLAR_FLUXES
    Kokkos::View<double*, MemSpace>          scr_chimes_G0 ("mr_chimes_G0",  (long)n * CHIMES_LOCAL_UV_NBINS);
    Kokkos::View<double*, MemSpace>          scr_chimes_ion("mr_chimes_ion", (long)n * CHIMES_LOCAL_UV_NBINS);
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_rt_s ("mr_rt_s",  n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_rt_vs("mr_rt_vs", n);
#endif
#ifdef SINK_PHOTONMOMENTUM
    Kokkos::View<MyGravFloat*, MemSpace>       scr_sink_lum     ("mr_sink_lum", n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_sink_lum_grad("mr_sink_lum_grad", n);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    Kokkos::View<MyGravFloat*, MemSpace>       scr_cr_inject("mr_cr_inj", n);
#endif
#ifdef SINK_CALC_DISTANCES
    Kokkos::View<MyGravFloat*, MemSpace>       scr_sink_mass("mr_sink_mass", n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_sink_pos ("mr_sink_pos",  n);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Kokkos::View<int*, MemSpace>               scr_N_SINK   ("mr_N_SINK",  n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_sink_vel ("mr_sink_vel", n);
#endif
#if defined(SPECIAL_POINT_MOTION)
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_sink_acc ("mr_sink_acc", n);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    Kokkos::View<MyGravFloat*, MemSpace>       scr_max_fbvel("mr_max_fbvel", n);
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    Kokkos::View<MyGravFloat*, MemSpace>       scr_tidal("mr_tidal", (long)n * 6);
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Kokkos::View<MyGravFloat*, MemSpace>       scr_mass_dm("mr_mass_dm", n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_s_dm   ("mr_s_dm",  n);
    Kokkos::View<Vec3<MyGravFloat>*, MemSpace> scr_vs_dm  ("mr_vs_dm", n);
#endif

    /* SoA pointers captured into the lambdas (raw pointers; SharedSpace is
     * accessible from device). The SoA `father`/`bitflags` carry topology
     * already seeded by gpu_gravity_tree_acquire above. */
    int          *father_soa = soa->father;
    unsigned int *bitflags_soa = soa->bitflags;
    Vec3<MyFloat> *center_soa = soa->center;

    /* ---------------- Kernel 1: zero scratch + pending ----------------- */
    Kokkos::deep_copy(pending,      0);
    Kokkos::deep_copy(scr_mass,     (MyGravFloat) 0);
    Kokkos::deep_copy(scr_Npart,    (long) 0);
    Kokkos::deep_copy(scr_hmax,     (MyGravFloat) 0);
    Kokkos::deep_copy(scr_vmax,     (MyGravFloat) 0);
    Kokkos::deep_copy(scr_divVmax,  (MyGravFloat) 0);
    Kokkos::deep_copy(scr_maxsoft,  (MyGravFloat) 0);
    /* Vec3 views need element-wise zeroing; use parallel_for. */
    Kokkos::parallel_for("mr_zero_vecs", n, KOKKOS_LAMBDA(int k) {
        scr_s(k) = Vec3<MyGravFloat>{};
        scr_vs(k) = Vec3<MyGravFloat>{};
        /* preserve TOPLEVEL/DEPENDS_ON_LOCAL_ELEMENT/INTERNAL_TOPLEVEL bits;
         * clear everything else (matches CPU saved_bitflags mask at
         * forcetree.cc:3841). */
        unsigned int saved = bitflags_soa[k];
        scr_bitflags(k) = saved & ((1u << BITFLAG_TOPLEVEL) |
                                    (1u << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT) |
                                    (1u << BITFLAG_INTERNAL_TOPLEVEL));
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        scr_gasmass(k) = 0;
#endif
#ifdef RT_USE_GRAVTREE
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {scr_stellar_lum((long)k * N_RT_FREQ_BINS + b) = 0;}
#ifdef CHIMES_STELLAR_FLUXES
        for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
            scr_chimes_G0 ((long)k * CHIMES_LOCAL_UV_NBINS + b) = 0;
            scr_chimes_ion((long)k * CHIMES_LOCAL_UV_NBINS + b) = 0;
        }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        scr_rt_s (k) = Vec3<MyGravFloat>{};
        scr_rt_vs(k) = Vec3<MyGravFloat>{};
#endif
#ifdef SINK_PHOTONMOMENTUM
        scr_sink_lum     (k) = 0;
        scr_sink_lum_grad(k) = Vec3<MyGravFloat>{};
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        scr_cr_inject(k) = 0;
#endif
#ifdef SINK_CALC_DISTANCES
        scr_sink_mass(k) = 0;
        scr_sink_pos (k) = Vec3<MyGravFloat>{};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        scr_N_SINK  (k) = 0;
        scr_sink_vel(k) = Vec3<MyGravFloat>{};
#endif
#if defined(SPECIAL_POINT_MOTION)
        scr_sink_acc(k) = Vec3<MyGravFloat>{};
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
        scr_max_fbvel(k) = 0;
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        for(int kk = 0; kk < 6; kk++) {scr_tidal((long)k * 6 + kk) = 0;}
#endif
#ifdef DM_SCALARFIELD_SCREENING
        scr_mass_dm(k) = 0;
        scr_s_dm   (k) = Vec3<MyGravFloat>{};
        scr_vs_dm  (k) = Vec3<MyGravFloat>{};
#endif
    });

    /* ---------------- Kernel 2: pending counter init ------------------- */
    /* For each node k in [0..n), if k's father is an internal node, atomic-
     * inc pending[father - MaxPart]. This counts internal-node children. */
    Kokkos::parallel_for("mr_pending_init", n, KOKKOS_LAMBDA(int k) {
        int f = father_soa[k];
        if(f >= MaxPart && f < MaxPart + n) {
            Kokkos::atomic_inc(&pending(f - MaxPart));
        }
    });

    /* ---------------- Kernel 3: particle pass -------------------------- */
    double cf_atime = All.cf_atime;
    double maxKR    = All.MaxKernelRadius;
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

        Kokkos::atomic_add(&scr_mass(k), (MyGravFloat) pa->Mass);
        atomic_add_vec3<MyGravFloat, MyDouble>(&scr_s(k),  pa->Mass * pa->Pos);
        atomic_add_vec3<MyGravFloat, MyFloat >(&scr_vs(k), pa->Mass * pa->Vel);
        Kokkos::atomic_add(&scr_Npart(k), (long) 1);

        MyFloat vmax_p = 0;
        for(int kk = 0; kk < 3; kk++) {MyFloat v = pa->Vel[kk]; if(v < 0) {v = -v;} if(v > vmax_p) {vmax_p = v;}}
        atomic_max_fp<MyGravFloat>(&scr_vmax(k), (MyGravFloat) vmax_p);

        double soft_p = gpu_force_softening_kernelradius(P_dev, i);
        atomic_max_fp<MyGravFloat>(&scr_maxsoft(k), (MyGravFloat) soft_p);
#ifdef SINGLE_STAR_SINK_DYNAMICS
        if(pa->Type == 5) {
            atomic_max_fp<MyGravFloat>(&scr_maxsoft(k), (MyGravFloat) pa->KernelRadius);
        }
#endif

        if(pa->Type == 0) {
            double htmp = (maxKR < pa->KernelRadius) ? maxKR : pa->KernelRadius;
            atomic_max_fp<MyGravFloat>(&scr_hmax(k),    (MyGravFloat) htmp);
            atomic_max_fp<MyGravFloat>(&scr_divVmax(k), (MyGravFloat) pa->Particle_DivVel);
        }

#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        if(pa->Type == 0) {Kokkos::atomic_add(&scr_gasmass(k), (MyGravFloat) pa->Mass);}
#if defined(SINK_ALPHADISK_ACCRETION) && defined(RT_USE_TREECOL_FOR_NH)
        if(pa->Type == 5) {Kokkos::atomic_add(&scr_gasmass(k), (MyGravFloat) pa->Sink_Mass_Reservoir);}
#endif
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        Kokkos::atomic_add(&scr_cr_inject(k), (MyGravFloat) cr_inj_[i]);
#endif
#ifdef RT_USE_GRAVTREE
        {
            double l_sum = 0;
            for(int b = 0; b < N_RT_FREQ_BINS; b++) {
                double lb = src_lum_[(long)i * N_RT_FREQ_BINS + b];
                Kokkos::atomic_add(&scr_stellar_lum((long)k * N_RT_FREQ_BINS + b), (MyGravFloat) lb);
                l_sum += lb;
            }
#ifdef CHIMES_STELLAR_FLUXES
            for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
                Kokkos::atomic_add(&scr_chimes_G0 ((long)k * CHIMES_LOCAL_UV_NBINS + b),
                                   src_lum_G0_ [(long)i * CHIMES_LOCAL_UV_NBINS + b]);
                Kokkos::atomic_add(&scr_chimes_ion((long)k * CHIMES_LOCAL_UV_NBINS + b),
                                   src_lum_ion_[(long)i * CHIMES_LOCAL_UV_NBINS + b]);
            }
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            if(l_sum > 0) {
                atomic_add_vec3<MyGravFloat, MyDouble>(&scr_rt_s (k), l_sum * pa->Pos);
                atomic_add_vec3<MyGravFloat, MyFloat >(&scr_rt_vs(k), l_sum * pa->Vel);
            }
#endif
        }
#endif
#ifdef SINK_PHOTONMOMENTUM
        if(pa->Type == 5 && pa->Mass > 0 && bh_lum_[i] > 0) {
            Kokkos::atomic_add(&scr_sink_lum(k), (MyGravFloat) bh_lum_[i]);
            atomic_add_vec3<MyGravFloat, MyFloat>(&scr_sink_lum_grad(k),
                                                   ((MyFloat) bh_lum_[i]) * bh_angle_[i]);
        }
#endif
#ifdef SINK_CALC_DISTANCES
        if(pa->Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES) {
            Kokkos::atomic_add(&scr_sink_mass(k), (MyGravFloat) pa->Mass);
            atomic_add_vec3<MyGravFloat, MyDouble>(&scr_sink_pos(k), pa->Mass * pa->Pos);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
            Kokkos::atomic_add(&scr_N_SINK(k), 1);
            atomic_add_vec3<MyGravFloat, MyFloat>(&scr_sink_vel(k), pa->Mass * pa->Vel);
#endif
#ifdef SPECIAL_POINT_MOTION
            atomic_add_vec3<MyGravFloat, MyDouble>(&scr_sink_acc(k), pa->Mass * pa->Acc_Total_PrevStep);
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
            atomic_max_fp<MyGravFloat>(&scr_max_fbvel(k), (MyGravFloat) pa->MaxFeedbackVel);
#endif
        }
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        for(int kk = 0; kk < 6; kk++) {
            Kokkos::atomic_add(&scr_tidal((long)k * 6 + kk),
                               (MyGravFloat)(pa->Mass * pa->tidal_tensorps_prevstep.data[kk]));
        }
#endif
#ifdef DM_SCALARFIELD_SCREENING
        if(pa->Type != 0) {
            Kokkos::atomic_add(&scr_mass_dm(k), (MyGravFloat) pa->Mass);
            atomic_add_vec3<MyGravFloat, MyDouble>(&scr_s_dm (k), pa->Mass * pa->Pos);
            atomic_add_vec3<MyGravFloat, MyFloat >(&scr_vs_dm(k), pa->Mass * pa->Vel);
        }
#endif
    });

    /* ---------------- Kernel 4: bottom-up walk via father chain -------- */
    /* Each thread starts at node k. Precondition: pending[k] is the count
     * of internal-node children. Leaves (pending==0) walk up first; each
     * step decrements the parent's pending and the LAST decrementer wins
     * the right to continue (CAS-like ownership transfer). */
    Kokkos::parallel_for("mr_walk_up", n, KOKKOS_LAMBDA(int k0) {
        int curr = k0;
        while(true) {
            if(Kokkos::atomic_fetch_add(&pending(curr), 0) != 0) {return;}
            int f = father_soa[curr];
            if(f < MaxPart || f >= MaxPart + n) {return;}
            int kp = f - MaxPart;

            /* Add curr's accumulated moments to parent kp. */
            Kokkos::atomic_add(&scr_mass(kp), scr_mass(curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_s(kp),  scr_s(curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_vs(kp), scr_vs(curr));
            Kokkos::atomic_add(&scr_Npart(kp), scr_Npart(curr));
            atomic_max_fp<MyGravFloat>(&scr_hmax(kp),    scr_hmax(curr));
            atomic_max_fp<MyGravFloat>(&scr_vmax(kp),    scr_vmax(curr));
            atomic_max_fp<MyGravFloat>(&scr_divVmax(kp), scr_divVmax(curr));
            atomic_max_fp<MyGravFloat>(&scr_maxsoft(kp), scr_maxsoft(curr));
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            Kokkos::atomic_add(&scr_gasmass(kp), scr_gasmass(curr));
#endif
#ifdef RT_USE_GRAVTREE
            for(int b = 0; b < N_RT_FREQ_BINS; b++) {
                Kokkos::atomic_add(&scr_stellar_lum((long)kp * N_RT_FREQ_BINS + b),
                                   scr_stellar_lum((long)curr * N_RT_FREQ_BINS + b));
            }
#ifdef CHIMES_STELLAR_FLUXES
            for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
                Kokkos::atomic_add(&scr_chimes_G0 ((long)kp * CHIMES_LOCAL_UV_NBINS + b),
                                   scr_chimes_G0 ((long)curr * CHIMES_LOCAL_UV_NBINS + b));
                Kokkos::atomic_add(&scr_chimes_ion((long)kp * CHIMES_LOCAL_UV_NBINS + b),
                                   scr_chimes_ion((long)curr * CHIMES_LOCAL_UV_NBINS + b));
            }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_rt_s (kp), scr_rt_s (curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_rt_vs(kp), scr_rt_vs(curr));
#endif
#ifdef SINK_PHOTONMOMENTUM
            Kokkos::atomic_add(&scr_sink_lum(kp), scr_sink_lum(curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_sink_lum_grad(kp), scr_sink_lum_grad(curr));
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            Kokkos::atomic_add(&scr_cr_inject(kp), scr_cr_inject(curr));
#endif
#ifdef SINK_CALC_DISTANCES
            Kokkos::atomic_add(&scr_sink_mass(kp), scr_sink_mass(curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_sink_pos(kp), scr_sink_pos(curr));
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
            Kokkos::atomic_add(&scr_N_SINK(kp), scr_N_SINK(curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_sink_vel(kp), scr_sink_vel(curr));
#endif
#ifdef SPECIAL_POINT_MOTION
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_sink_acc(kp), scr_sink_acc(curr));
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
            if(scr_sink_mass(curr) > 0) {
                atomic_max_fp<MyGravFloat>(&scr_max_fbvel(kp), scr_max_fbvel(curr));
            }
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            for(int kk = 0; kk < 6; kk++) {
                Kokkos::atomic_add(&scr_tidal((long)kp * 6 + kk), scr_tidal((long)curr * 6 + kk));
            }
#endif
#ifdef DM_SCALARFIELD_SCREENING
            Kokkos::atomic_add(&scr_mass_dm(kp), scr_mass_dm(curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_s_dm (kp), scr_s_dm (curr));
            atomic_add_vec3<MyGravFloat, MyGravFloat>(&scr_vs_dm(kp), scr_vs_dm(curr));
#endif

            /* Decrement the parent's pending counter. The thread that takes
             * the counter from 1 → 0 wins the right to continue walking up
             * with curr=parent. Other threads abort. */
            int prev = Kokkos::atomic_fetch_sub(&pending(kp), 1);
            if(prev != 1) {return;}
            curr = kp;
        }
    });

    /* ---------------- Kernel 5: normalize ------------------------------ */
    Kokkos::parallel_for("mr_normalize", n, KOKKOS_LAMBDA(int k) {
        MyGravFloat mass = scr_mass(k);
        if(mass > 0) {
            MyGravFloat inv = (MyGravFloat) 1 / mass;
            scr_s(k)  = scr_s(k)  * inv;
            scr_vs(k) = scr_vs(k) * inv;
        } else {
            scr_s(k) = Vec3<MyGravFloat>{(MyGravFloat) center_soa[k][0],
                                          (MyGravFloat) center_soa[k][1],
                                          (MyGravFloat) center_soa[k][2]};
            scr_vs(k) = Vec3<MyGravFloat>{};
        }
        if(scr_Npart(k) > 1) {scr_bitflags(k) |=  (1u << BITFLAG_MULTIPLEPARTICLES);}
        else                  {scr_bitflags(k) &= ~(1u << BITFLAG_MULTIPLEPARTICLES);}
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        {
            double l_tot = 0;
            for(int b = 0; b < N_RT_FREQ_BINS; b++) {l_tot += scr_stellar_lum((long)k * N_RT_FREQ_BINS + b);}
            if(l_tot > 0) {
                MyGravFloat inv_l = (MyGravFloat) (1.0 / l_tot);
                scr_rt_s (k) = scr_rt_s (k) * inv_l;
                scr_rt_vs(k) = scr_rt_vs(k) * inv_l;
            } else {
                scr_rt_s (k) = Vec3<MyGravFloat>{(MyGravFloat) center_soa[k][0],
                                                  (MyGravFloat) center_soa[k][1],
                                                  (MyGravFloat) center_soa[k][2]};
                scr_rt_vs(k) = Vec3<MyGravFloat>{};
            }
        }
#endif
#ifdef SINK_PHOTONMOMENTUM
        if(scr_sink_lum(k) > 0) {
            scr_sink_lum_grad(k) = scr_sink_lum_grad(k) * ((MyGravFloat) 1 / scr_sink_lum(k));
        } else {
            scr_sink_lum_grad(k) = Vec3<MyGravFloat>{0, 0, 1};
        }
#endif
#ifdef SINK_CALC_DISTANCES
        if(scr_sink_mass(k) > 0) {
            MyGravFloat invsm = (MyGravFloat) 1 / scr_sink_mass(k);
            scr_sink_pos(k) = scr_sink_pos(k) * invsm;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
            scr_sink_vel(k) = scr_sink_vel(k) * invsm;
#endif
#ifdef SPECIAL_POINT_MOTION
            scr_sink_acc(k) = scr_sink_acc(k) * invsm;
#endif
        }
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        if(mass > 0) {
            MyGravFloat inv_m = (MyGravFloat) (1.0 / ((double)mass + MIN_REAL_NUMBER));
            for(int kk = 0; kk < 6; kk++) {scr_tidal((long)k * 6 + kk) *= inv_m;}
        }
#endif
#ifdef DM_SCALARFIELD_SCREENING
        if(scr_mass_dm(k) > 0) {
            MyGravFloat inv_dm = (MyGravFloat) 1 / scr_mass_dm(k);
            scr_s_dm (k) = scr_s_dm (k) * inv_dm;
            scr_vs_dm(k) = scr_vs_dm(k) * inv_dm;
        } else {
            scr_s_dm (k) = Vec3<MyGravFloat>{(MyGravFloat) center_soa[k][0],
                                              (MyGravFloat) center_soa[k][1],
                                              (MyGravFloat) center_soa[k][2]};
            scr_vs_dm(k) = Vec3<MyGravFloat>{};
        }
#endif
    });

    Kokkos::fence();

    /* ---------------- Bulk deep-copy scratch → SharedSpace SoA -------- */
    /* Wrap raw SoA pointers in unmanaged Views and deep_copy. */
    using UV_unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ShSpace = Kokkos::View<MyGravFloat*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShVec3  = Kokkos::View<Vec3<MyGravFloat>*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShU32   = Kokkos::View<unsigned int*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShLong  = Kokkos::View<long*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShDbl   = Kokkos::View<double*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;
    using ShInt   = Kokkos::View<int*, GIZMO_KOKKOS_SHARED_SPACE, UV_unmanaged>;

    Kokkos::deep_copy(ShSpace(soa->mass,    n), scr_mass);
    Kokkos::deep_copy(ShVec3 (soa->s,       n), scr_s);
    Kokkos::deep_copy(ShVec3 (soa->node_vs, n), scr_vs);
    Kokkos::deep_copy(ShLong (soa->N_part,  n), scr_Npart);
    Kokkos::deep_copy(ShSpace(soa->hmax,    n), scr_hmax);
    Kokkos::deep_copy(ShSpace(soa->vmax,    n), scr_vmax);
    Kokkos::deep_copy(ShSpace(soa->divVmax, n), scr_divVmax);
    Kokkos::deep_copy(ShSpace(soa->maxsoft, n), scr_maxsoft);
    Kokkos::deep_copy(ShU32  (soa->bitflags,n), scr_bitflags);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    Kokkos::deep_copy(ShSpace(soa->gasmass, n), scr_gasmass);
#endif
#ifdef RT_USE_GRAVTREE
    Kokkos::deep_copy(ShSpace(soa->stellar_lum, (long)n * N_RT_FREQ_BINS), scr_stellar_lum);
#ifdef CHIMES_STELLAR_FLUXES
    Kokkos::deep_copy(ShDbl(soa->chimes_stellar_lum_G0,  (long)n * CHIMES_LOCAL_UV_NBINS), scr_chimes_G0);
    Kokkos::deep_copy(ShDbl(soa->chimes_stellar_lum_ion, (long)n * CHIMES_LOCAL_UV_NBINS), scr_chimes_ion);
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Kokkos::deep_copy(ShVec3(soa->rt_source_lum_s,  n), scr_rt_s);
    Kokkos::deep_copy(ShVec3(soa->rt_source_lum_vs, n), scr_rt_vs);
#endif
#ifdef SINK_PHOTONMOMENTUM
    Kokkos::deep_copy(ShSpace(soa->sink_lum,      n), scr_sink_lum);
    Kokkos::deep_copy(ShVec3 (soa->sink_lum_grad, n), scr_sink_lum_grad);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    Kokkos::deep_copy(ShSpace(soa->cr_injection, n), scr_cr_inject);
#endif
#ifdef SINK_CALC_DISTANCES
    Kokkos::deep_copy(ShSpace(soa->sink_mass, n), scr_sink_mass);
    Kokkos::deep_copy(ShVec3 (soa->sink_pos,  n), scr_sink_pos);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Kokkos::deep_copy(ShInt  (soa->N_SINK,    n), scr_N_SINK);
    Kokkos::deep_copy(ShVec3 (soa->sink_vel,  n), scr_sink_vel);
#endif
#if defined(SPECIAL_POINT_MOTION)
    Kokkos::deep_copy(ShVec3 (soa->sink_acc,  n), scr_sink_acc);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    Kokkos::deep_copy(ShSpace(soa->MaxFeedbackVel, n), scr_max_fbvel);
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    Kokkos::deep_copy(ShSpace(soa->tidal_tensorps, (long)n * 6), scr_tidal);
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Kokkos::deep_copy(ShSpace(soa->mass_dm, n), scr_mass_dm);
    Kokkos::deep_copy(ShVec3 (soa->s_dm,    n), scr_s_dm);
    Kokkos::deep_copy(ShVec3 (soa->vs_dm,   n), scr_vs_dm);
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
/* Bulk SoA → Nodes[]/Extnodes[] AoS write-back. Mechanical inverse of  */
/* seed_node_ in gpu_gravity_tree.cc. Runs on host (host-side mutation  */
/* of the AoS arrays so the CPU pseudo path picks up the new moments).  */
/* SharedSpace pages may need a Kokkos::fence before this; callers are  */
/* responsible.                                                          */
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

#endif /* OPENMP_GPU_OFFLOAD */
