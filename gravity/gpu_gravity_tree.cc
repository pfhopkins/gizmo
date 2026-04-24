/* gpu_gravity_tree.cc — Step 13 Phase 3
 *
 * See gpu_gravity_tree.h for design notes.
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

/* GPU All mirror precedes allvars.h so nvc++ sees `All` (=All_dev) for
 * eagerly-parsed templates referencing it. Matches density_gpu.cc include order. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "gpu_gravity_tree.h"

#ifdef OPENMP_GPU_OFFLOAD

static struct gpu_gravity_tree_soa_t soa_ = {0};
static int soa_capacity_ = 0;
static int soa_valid_    = 0;

/* Phase 6.0: per-node dirty tracking. dirty_[k]==1 means Nodes[MaxPart+k]
 * needs its SoA slot re-copied on the next acquire(). any_dirty_ is a
 * fast-path scalar so acquire() can early-exit when nothing changed.
 * dirty_count_ is for diagnostics only. */
static unsigned char *dirty_    = NULL;
static int            any_dirty_   = 0;
static int            dirty_count_ = 0;

static void free_arrays_(void)
{
    if(soa_.center)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.center);   soa_.center   = NULL;}
    if(soa_.len)      {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.len);      soa_.len      = NULL;}
    if(soa_.s)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.s);        soa_.s        = NULL;}
    if(soa_.mass)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.mass);     soa_.mass     = NULL;}
    if(soa_.sibling)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sibling);  soa_.sibling  = NULL;}
    if(soa_.nextnode) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.nextnode); soa_.nextnode = NULL;}
    if(soa_.bitflags) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.bitflags); soa_.bitflags = NULL;}
    if(soa_.maxsoft)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.maxsoft);  soa_.maxsoft  = NULL;}
    if(soa_.N_part)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.N_part);   soa_.N_part   = NULL;}
    if(soa_.nextnode_aux) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.nextnode_aux); soa_.nextnode_aux = NULL;}
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    if(soa_.gasmass)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.gasmass);        soa_.gasmass        = NULL;}
#endif
#ifdef RT_USE_GRAVTREE
    if(soa_.stellar_lum)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.stellar_lum);    soa_.stellar_lum    = NULL;}
#ifdef CHIMES_STELLAR_FLUXES
    if(soa_.chimes_stellar_lum_G0)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.chimes_stellar_lum_G0);  soa_.chimes_stellar_lum_G0  = NULL;}
    if(soa_.chimes_stellar_lum_ion) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.chimes_stellar_lum_ion); soa_.chimes_stellar_lum_ion = NULL;}
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    if(soa_.rt_source_lum_s)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.rt_source_lum_s);  soa_.rt_source_lum_s  = NULL;}
    if(soa_.rt_source_lum_vs) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.rt_source_lum_vs); soa_.rt_source_lum_vs = NULL;}
#endif
#ifdef SINK_PHOTONMOMENTUM
    if(soa_.sink_lum)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sink_lum);       soa_.sink_lum       = NULL;}
    if(soa_.sink_lum_grad)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sink_lum_grad);  soa_.sink_lum_grad  = NULL;}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    if(soa_.cr_injection)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.cr_injection);   soa_.cr_injection   = NULL;}
#endif
#ifdef SINK_CALC_DISTANCES
    if(soa_.sink_mass)      {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sink_mass);      soa_.sink_mass      = NULL;}
    if(soa_.sink_pos)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sink_pos);       soa_.sink_pos       = NULL;}
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
    if(soa_.sink_vel)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sink_vel);       soa_.sink_vel       = NULL;}
#endif
#if defined(SPECIAL_POINT_MOTION)
    if(soa_.sink_acc)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sink_acc);       soa_.sink_acc       = NULL;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    if(soa_.N_SINK)         {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.N_SINK);         soa_.N_SINK         = NULL;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    if(soa_.MaxFeedbackVel) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.MaxFeedbackVel); soa_.MaxFeedbackVel = NULL;}
#endif
#endif
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
    if(soa_.node_vs)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.node_vs);        soa_.node_vs        = NULL;}
#endif
    if(dirty_) {free(dirty_); dirty_ = NULL;}
    any_dirty_   = 0;
    dirty_count_ = 0;
    soa_.nnodes = 0;
    soa_.nextnode_aux_size = 0;
}

static int alloc_arrays_(int n)
{
    soa_.center   = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    soa_.len      = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.s        = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    soa_.mass     = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.sibling  = (int           *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.nextnode = (int           *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.bitflags = (unsigned int  *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(unsigned int));
    soa_.maxsoft  = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.N_part   = (long          *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(long));
    if(!soa_.center || !soa_.len || !soa_.s || !soa_.mass || !soa_.sibling ||
       !soa_.nextnode || !soa_.bitflags || !soa_.maxsoft || !soa_.N_part) {
        printf("gpu_gravity_tree: kokkos_malloc failed for %d nodes\n", n);
        return 0;
    }
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    soa_.gasmass = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    if(!soa_.gasmass) {printf("gpu_gravity_tree: gasmass alloc failed (%d)\n", n); return 0;}
#endif
#ifdef RT_USE_GRAVTREE
    soa_.stellar_lum = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * N_RT_FREQ_BINS * sizeof(MyFloat));
    if(!soa_.stellar_lum) {printf("gpu_gravity_tree: stellar_lum alloc failed (%d)\n", n); return 0;}
#ifdef CHIMES_STELLAR_FLUXES
    soa_.chimes_stellar_lum_G0  = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * CHIMES_LOCAL_UV_NBINS * sizeof(double));
    soa_.chimes_stellar_lum_ion = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * CHIMES_LOCAL_UV_NBINS * sizeof(double));
    if(!soa_.chimes_stellar_lum_G0 || !soa_.chimes_stellar_lum_ion) {printf("gpu_gravity_tree: chimes lum alloc failed (%d)\n", n); return 0;}
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    soa_.rt_source_lum_s  = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    soa_.rt_source_lum_vs = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    if(!soa_.rt_source_lum_s || !soa_.rt_source_lum_vs) {printf("gpu_gravity_tree: rt_source_lum alloc failed (%d)\n", n); return 0;}
#endif
#ifdef SINK_PHOTONMOMENTUM
    soa_.sink_lum      = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.sink_lum_grad = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    if(!soa_.sink_lum || !soa_.sink_lum_grad) {printf("gpu_gravity_tree: sink_lum alloc failed (%d)\n", n); return 0;}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    soa_.cr_injection = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    if(!soa_.cr_injection) {printf("gpu_gravity_tree: cr_injection alloc failed (%d)\n", n); return 0;}
#endif
#ifdef SINK_CALC_DISTANCES
    soa_.sink_mass = (MyFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.sink_pos  = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    if(!soa_.sink_mass || !soa_.sink_pos) {printf("gpu_gravity_tree: sink_mass/pos alloc failed (%d)\n", n); return 0;}
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
    soa_.sink_vel = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    if(!soa_.sink_vel) {printf("gpu_gravity_tree: sink_vel alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SPECIAL_POINT_MOTION)
    soa_.sink_acc = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    if(!soa_.sink_acc) {printf("gpu_gravity_tree: sink_acc alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    soa_.N_SINK = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    if(!soa_.N_SINK) {printf("gpu_gravity_tree: N_SINK alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    soa_.MaxFeedbackVel = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    if(!soa_.MaxFeedbackVel) {printf("gpu_gravity_tree: MaxFeedbackVel alloc failed (%d)\n", n); return 0;}
#endif
#endif
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
    soa_.node_vs = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    if(!soa_.node_vs) {printf("gpu_gravity_tree: node_vs alloc failed (%d)\n", n); return 0;}
#endif
    dirty_ = (unsigned char *) calloc((size_t) n, sizeof(unsigned char));
    if(!dirty_) {printf("gpu_gravity_tree: dirty_ alloc failed (%d)\n", n); return 0;}
    return 1;
}

static inline void seed_node_(int k, struct NODE *Nodes_host, struct extNODE *Extnodes_host)
{
    /* Copy the fields the walk reads for a single node index k. Used by both
     * the full-seed loop (fresh allocation) and the partial-seed pass (dirty
     * nodes only). SharedSpace pages migrate device-side on first kernel
     * touch. The Vec3 fields (center, s) live inside Nodes[] as Vec3<MyFloat>;
     * straight assignment works because the SoA mirror uses the same type. */
    {
        soa_.center[k]   = Nodes_host[k].center;
        soa_.len[k]      = Nodes_host[k].len;
        soa_.s[k]        = Nodes_host[k].u.d.s;
        soa_.mass[k]     = Nodes_host[k].u.d.mass;
        soa_.sibling[k]  = Nodes_host[k].u.d.sibling;
        soa_.nextnode[k] = Nodes_host[k].u.d.nextnode;
        soa_.bitflags[k] = Nodes_host[k].u.d.bitflags;
        soa_.maxsoft[k]  = Nodes_host[k].maxsoft;
        soa_.N_part[k]   = Nodes_host[k].N_part;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        soa_.gasmass[k]  = Nodes_host[k].gasmass;
#endif
#ifdef RT_USE_GRAVTREE
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            soa_.stellar_lum[k * N_RT_FREQ_BINS + b] = Nodes_host[k].stellar_lum[b];
        }
#ifdef CHIMES_STELLAR_FLUXES
        for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
            soa_.chimes_stellar_lum_G0 [k * CHIMES_LOCAL_UV_NBINS + b] = Nodes_host[k].chimes_stellar_lum_G0[b];
            soa_.chimes_stellar_lum_ion[k * CHIMES_LOCAL_UV_NBINS + b] = Nodes_host[k].chimes_stellar_lum_ion[b];
        }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        soa_.rt_source_lum_s[k]  = Nodes_host[k].rt_source_lum_s;
        if(Extnodes_host) {soa_.rt_source_lum_vs[k] = Extnodes_host[k].rt_source_lum_vs;}
#endif
#ifdef SINK_PHOTONMOMENTUM
        soa_.sink_lum[k]      = Nodes_host[k].sink_lum;
        soa_.sink_lum_grad[k] = Nodes_host[k].sink_lum_grad;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        soa_.cr_injection[k]  = Nodes_host[k].cr_injection;
#endif
#ifdef SINK_CALC_DISTANCES
        soa_.sink_mass[k] = Nodes_host[k].sink_mass;
        soa_.sink_pos[k]  = Nodes_host[k].sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
        soa_.sink_vel[k]  = Nodes_host[k].sink_vel;
#endif
#if defined(SPECIAL_POINT_MOTION)
        soa_.sink_acc[k]  = Nodes_host[k].sink_acc;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        soa_.N_SINK[k]    = Nodes_host[k].N_SINK;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
        soa_.MaxFeedbackVel[k] = Nodes_host[k].MaxFeedbackVel;
#endif
#endif
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
        if(Extnodes_host) {soa_.node_vs[k] = Extnodes_host[k].vs;}
#endif
    }
}

/* Seed every node in [0..n) and mark all clean. Used on fresh allocation. */
static void seed_full_(int n, struct NODE *Nodes_host, struct extNODE *Extnodes_host)
{
    for(int k = 0; k < n; k++) {
        seed_node_(k, Nodes_host, Extnodes_host);
        if(dirty_) {dirty_[k] = 0;}
    }
    soa_.nnodes    = n;
    any_dirty_     = 0;
    dirty_count_   = 0;
}

/* Seed only nodes marked dirty in [0..n). Used after per-node mark_dirty
 * fires (e.g. pre-walk drift loop touched some nodes but not others). */
static void seed_dirty_(int n, struct NODE *Nodes_host, struct extNODE *Extnodes_host)
{
    if(!dirty_) {seed_full_(n, Nodes_host, Extnodes_host); return;}
    for(int k = 0; k < n; k++) {
        if(dirty_[k]) {
            seed_node_(k, Nodes_host, Extnodes_host);
            dirty_[k] = 0;
        }
    }
    soa_.nnodes    = n;
    any_dirty_     = 0;
    dirty_count_   = 0;
}

extern "C" void gpu_gravity_tree_acquire(int min_nodes,
                                          struct NODE    *Nodes_host,
                                          struct extNODE *Extnodes_host)
{
    if(min_nodes <= 0) {min_nodes = 1;}

    if(soa_capacity_ >= min_nodes && soa_.center) {
        if(soa_valid_ && !any_dirty_) {
            /* Mirror fully in sync — fast path, no copy. */
            return;
        }
        /* Partial reseed for dirty nodes only (or full reseed if the mirror
         * was hard-invalidated via soa_valid_=0). */
        if(Nodes_host) {
            if(!soa_valid_) {seed_full_(min_nodes, Nodes_host, Extnodes_host);}
            else             {seed_dirty_(min_nodes, Nodes_host, Extnodes_host);}
        }
        soa_valid_ = 1;
        return;
    }

    /* Need fresh allocation: capacity grew or first acquire. */
    free_arrays_();
    if(!alloc_arrays_(min_nodes)) {endrun(913101);}
    soa_capacity_ = min_nodes;
    if(Nodes_host) {seed_full_(min_nodes, Nodes_host, Extnodes_host);}
    soa_valid_ = 1;
}

extern "C" void gpu_gravity_tree_mark_dirty(int no)
{
    /* no is an absolute Nodes[] index (>= All.MaxPart). Convert to the SoA
     * [0..capacity) range. Out-of-range, pre-allocation, or stale-invalidate
     * states are silently ignored — the next acquire() will DTRT either way. */
    if(!dirty_ || soa_capacity_ <= 0) {return;}
    int k = no - All.MaxPart;
    if(k < 0 || k >= soa_capacity_) {return;}
    if(dirty_[k] == 0) {
        dirty_[k] = 1;
        dirty_count_++;
    }
    any_dirty_ = 1;
}

extern "C" void gpu_gravity_tree_mark_all_dirty(void)
{
    /* Topology rebuild (force_treebuild) or full moment refresh
     * (force_refresh_node_moments) — every slot needs re-copy. Use the
     * stronger soa_valid_=0 signal so acquire() takes the full-seed path. */
    soa_valid_   = 0;
    any_dirty_   = 1;
    if(dirty_ && soa_capacity_ > 0) {
        memset(dirty_, 1, (size_t) soa_capacity_);
        dirty_count_ = soa_capacity_;
    }
}

extern "C" void gpu_gravity_tree_invalidate(void)
{
    /* Back-compat alias. Removed in Phase 6.8. */
    gpu_gravity_tree_mark_all_dirty();
}

extern "C" int gpu_gravity_tree_dirty_count(void)
{
    return any_dirty_ ? dirty_count_ : 0;
}

extern "C" void gpu_gravity_tree_release(void)
{
    free_arrays_();
    soa_capacity_ = 0;
    soa_valid_    = 0;
}

extern "C" void gpu_gravity_tree_set_nextnode(int n, int *Nextnode_host)
{
    if(n <= 0) {return;}
    if(soa_.nextnode_aux_size < n) {
        if(soa_.nextnode_aux) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.nextnode_aux);}
        soa_.nextnode_aux = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
        if(!soa_.nextnode_aux) {
            printf("gpu_gravity_tree_set_nextnode: kokkos_malloc failed (n=%d)\n", n);
            endrun(913102);
        }
        soa_.nextnode_aux_size = n;
    }
    memcpy(soa_.nextnode_aux, Nextnode_host, n * sizeof(int));
}

extern "C" struct gpu_gravity_tree_soa_t *gpu_gravity_tree_soa(void) {return soa_valid_ ? &soa_ : NULL;}
extern "C" int gpu_gravity_tree_capacity(void)                       {return soa_capacity_;}
extern "C" int gpu_gravity_tree_valid(void)                          {return soa_valid_;}

#endif /* OPENMP_GPU_OFFLOAD */
