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

#include <Kokkos_Core.hpp>

/* GPU All mirror precedes allvars.h so nvc++ sees `All` (=All_dev) for
 * eagerly-parsed templates referencing it. Matches density_gpu.cc include order. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/gpu_error_check.h"
#include "gpu_gravity_tree.h"


static struct gpu_gravity_tree_soa_t soa_ = {0};
static int soa_capacity_ = 0;
static int soa_valid_    = 0;

static void free_arrays_(void)
{
    if(soa_.center)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.center);   soa_.center   = NULL;}
    if(soa_.len)      {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.len);      soa_.len      = NULL;}
    if(soa_.s)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.s);        soa_.s        = NULL;}
    if(soa_.mass)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.mass);     soa_.mass     = NULL;}
    if(soa_.sibling)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.sibling);  soa_.sibling  = NULL;}
    if(soa_.nextnode) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.nextnode); soa_.nextnode = NULL;}
    if(soa_.father)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.father);   soa_.father   = NULL;}
    if(soa_.bitflags) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.bitflags); soa_.bitflags = NULL;}
    if(soa_.maxsoft)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.maxsoft);  soa_.maxsoft  = NULL;}
    if(soa_.N_part)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.N_part);   soa_.N_part   = NULL;}
    /* Phase 6.8e: nextnode_aux is an alias to Nextnode[] (UVM, owned by
     * forcetree.cc).  Do NOT free or clear here — the alias persists across
     * SoA realloc cycles and is set/cleared exclusively by
     * gpu_gravity_tree_alias_nextnode().  free_arrays_ runs whenever capacity
     * changes, but the alias outlives that. */
    if(soa_.suns_backup)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.suns_backup);  soa_.suns_backup  = NULL;}
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
    /* Unconditional Extnodes mirrors (Phase 6.1a). */
    if(soa_.node_vs)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.node_vs);        soa_.node_vs        = NULL;}
    if(soa_.hmax)           {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.hmax);           soa_.hmax           = NULL;}
    if(soa_.vmax)           {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.vmax);           soa_.vmax           = NULL;}
    if(soa_.divVmax)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.divVmax);        soa_.divVmax        = NULL;}
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    if(soa_.tidal_tensorps) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.tidal_tensorps); soa_.tidal_tensorps = NULL;}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    if(soa_.mass_dm)        {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.mass_dm);        soa_.mass_dm        = NULL;}
    if(soa_.s_dm)           {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.s_dm);           soa_.s_dm           = NULL;}
    if(soa_.vs_dm)          {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.vs_dm);          soa_.vs_dm          = NULL;}
#endif
    soa_.nnodes = 0;
    /* Phase 6.8e: do NOT touch nextnode_aux / nextnode_aux_size here — the
     * alias is owned by force_treeallocate and outlives SoA realloc cycles. */
}

static int alloc_arrays_(int n)
{
    /* Phase 6.1b: moment/force fields retyped as MyGravFloat (flag-gated:
     * float when GIZMO_MIXED_PRECISION_GRAVITY set, double otherwise).
     * Geometric (center, len) stays MyFloat — opening-criterion precision. */
    soa_.center   = (Vec3<MyFloat>     *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyFloat>));
    soa_.len      = (MyFloat           *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyFloat));
    soa_.s        = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    soa_.mass     = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.sibling  = (int               *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.nextnode = (int               *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.father   = (int               *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    soa_.bitflags = (unsigned int      *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(unsigned int));
    soa_.maxsoft  = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.N_part   = (long              *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(long));
    soa_.suns_backup = (int             *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((long)n * 8 * sizeof(int));
    if(!soa_.center || !soa_.len || !soa_.s || !soa_.mass || !soa_.sibling ||
       !soa_.nextnode || !soa_.father || !soa_.bitflags || !soa_.maxsoft || !soa_.N_part ||
       !soa_.suns_backup) {
        printf("gpu_gravity_tree: kokkos_malloc failed for %d nodes\n", n);
        return 0;
    }
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    soa_.gasmass = (MyGravFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    if(!soa_.gasmass) {printf("gpu_gravity_tree: gasmass alloc failed (%d)\n", n); return 0;}
#endif
#ifdef RT_USE_GRAVTREE
    soa_.stellar_lum = (MyGravFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * N_RT_FREQ_BINS * sizeof(MyGravFloat));
    if(!soa_.stellar_lum) {printf("gpu_gravity_tree: stellar_lum alloc failed (%d)\n", n); return 0;}
#ifdef CHIMES_STELLAR_FLUXES
    /* CHIMES stays double — chemistry tolerances. */
    soa_.chimes_stellar_lum_G0  = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * CHIMES_LOCAL_UV_NBINS * sizeof(double));
    soa_.chimes_stellar_lum_ion = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * CHIMES_LOCAL_UV_NBINS * sizeof(double));
    if(!soa_.chimes_stellar_lum_G0 || !soa_.chimes_stellar_lum_ion) {printf("gpu_gravity_tree: chimes lum alloc failed (%d)\n", n); return 0;}
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    soa_.rt_source_lum_s  = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    soa_.rt_source_lum_vs = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.rt_source_lum_s || !soa_.rt_source_lum_vs) {printf("gpu_gravity_tree: rt_source_lum alloc failed (%d)\n", n); return 0;}
#endif
#ifdef SINK_PHOTONMOMENTUM
    soa_.sink_lum      = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.sink_lum_grad = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_lum || !soa_.sink_lum_grad) {printf("gpu_gravity_tree: sink_lum alloc failed (%d)\n", n); return 0;}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    soa_.cr_injection = (MyGravFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    if(!soa_.cr_injection) {printf("gpu_gravity_tree: cr_injection alloc failed (%d)\n", n); return 0;}
#endif
#ifdef SINK_CALC_DISTANCES
    soa_.sink_mass = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.sink_pos  = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_mass || !soa_.sink_pos) {printf("gpu_gravity_tree: sink_mass/pos alloc failed (%d)\n", n); return 0;}
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
    soa_.sink_vel = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_vel) {printf("gpu_gravity_tree: sink_vel alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SPECIAL_POINT_MOTION)
    soa_.sink_acc = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_acc) {printf("gpu_gravity_tree: sink_acc alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    soa_.N_SINK = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(int));
    if(!soa_.N_SINK) {printf("gpu_gravity_tree: N_SINK alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    soa_.MaxFeedbackVel = (MyGravFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    if(!soa_.MaxFeedbackVel) {printf("gpu_gravity_tree: MaxFeedbackVel alloc failed (%d)\n", n); return 0;}
#endif
#endif
    /* Unconditional Extnodes mirrors (Phase 6.1a; retyped 6.1b). */
    soa_.node_vs = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    soa_.hmax    = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.vmax    = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.divVmax = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    if(!soa_.node_vs || !soa_.hmax || !soa_.vmax || !soa_.divVmax) {
        printf("gpu_gravity_tree: unconditional extnode mirrors alloc failed (n=%d)\n", n);
        return 0;
    }
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    soa_.tidal_tensorps = (MyGravFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * 6 * sizeof(MyGravFloat));
    if(!soa_.tidal_tensorps) {printf("gpu_gravity_tree: tidal_tensorps alloc failed (%d)\n", n); return 0;}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    soa_.mass_dm = (MyGravFloat       *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(MyGravFloat));
    soa_.s_dm    = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    soa_.vs_dm   = (Vec3<MyGravFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.mass_dm || !soa_.s_dm || !soa_.vs_dm) {
        printf("gpu_gravity_tree: DM_SCALARFIELD mirrors alloc failed (n=%d)\n", n); return 0;
    }
#endif
    return 1;
}

/* Phase 7.a: seed_node_ / seed_dirty_ / dirty_[] / mark_dirty / dirty_count_
 * are all gone.  The GPU build pipeline (gpu_nextnode_backup_suns ->
 * gpu_topology_emit_bfs -> gpu_topology_finalize_father ->
 * gpu_topology_finalize_sibling -> gpu_moment_refresh ->
 * gpu_nextnode_thread) populates the SoA end-to-end inside kernels; the GPU
 * drift kernel (gpu_force_drift_nodes in gpu_force_drift.cc) writes both
 * UVM AoS and SoA mirrors in one pass.  No host-side AoS->SoA reseed path
 * remains.  acquire() is now a pure capacity-grow / pointer-grab. */

extern "C" void gpu_gravity_tree_acquire(int min_nodes,
                                          struct NODE    * /*Nodes_host*/,
                                          struct extNODE * /*Extnodes_host*/)
{
    if(min_nodes <= 0) {min_nodes = 1;}

    /* Phase 9.2-pre: extend SoA capacity to cover the LET foreign-node range
     * [MaxNodes, MaxNodes+MaxForeignNodes).  Foreign nodes installed by
     * let_unpack_and_install are scattered into SoA at slot_base + j with
     * absolute index = MaxPart + MaxNodes + slot, so SoA index =
     * (MaxNodes + slot).  Non-GPU builds have MaxForeignNodes==0; GPU
     * builds require positive LET headroom. */
    if(MaxForeignNodes > 0) {
        min_nodes += MaxForeignNodes;
    }

    if(soa_capacity_ >= min_nodes && soa_.center) {
        return;   /* fast path: capacity already sufficient */
    }

    /* Capacity grew (or first call): allocate.  No seeding -- the build
     * pipeline will populate.  soa_valid_ stays 1 because callers downstream
     * of the build kernels expect a populated SoA; the kernels write before
     * any reader. */
    free_arrays_();
    if(!alloc_arrays_(min_nodes)) {endrun(913101); soa_capacity_ = 0; soa_valid_ = 0; return;}  /* soft bad-stop: leave SoA invalid (soa() returns NULL; callers NULL-check); drains at next poll */
    soa_capacity_ = min_nodes;
    soa_valid_    = 1;
}

extern "C" void gpu_gravity_tree_release(void)
{
    free_arrays_();
    soa_capacity_ = 0;
    soa_valid_    = 0;
    gpu_force_drift_release();
}

extern "C" void gpu_gravity_tree_alias_nextnode(int *Nextnode_host, int n)
{
    /* Phase 6.8e: alias soa->nextnode_aux to the SharedSpace Nextnode[] owned
     * by force_treeallocate.  No separate buffer, no per-walk memcpy.  Called
     * once per (allocate / free) cycle from forcetree.cc. */
    soa_.nextnode_aux      = Nextnode_host;
    soa_.nextnode_aux_size = n;
}

extern "C" struct gpu_gravity_tree_soa_t *gpu_gravity_tree_soa(void) {return soa_valid_ ? &soa_ : NULL;}
extern "C" int gpu_gravity_tree_capacity(void)                       {return soa_capacity_;}
extern "C" int gpu_gravity_tree_valid(void)                          {return soa_valid_;}

/* Phase 6.4: snapshot Nodes_base[k].u.suns[0..7] for k in [0..n) into the
 * SoA's suns_backup buffer.  Called from force_treebuild_single right
 * before force_update_node_recursive overwrites the union with the d struct.
 * If the SoA hasn't been allocated yet (first acquire after a fresh
 * allocation), set up minimal scaffolding so the buffer exists.  Otherwise
 * just refresh the contents from current AoS. */
extern "C" void gpu_nextnode_backup_suns(int n)
{
    if(n <= 0) {return;}
    /* Allocate at MaxNodes+1 (full-tree capacity), not just n (=Numnodestree).
     * If we only allocated at n, the subsequent gpu_gravity_tree_acquire(MaxNodes+1)
     * inside gpu_nextnode_thread would see soa_capacity_ < MaxNodes+1, call
     * free_arrays_(), and destroy the suns_backup we store below — corrupting
     * the nextnode kernel's input on the first treebuild. */
    int cap = (MaxNodes > 0) ? MaxNodes + 1 : n;
    /* Phase 9.2-pre: include LET foreign-node range. */
    if(MaxForeignNodes > 0) {cap += MaxForeignNodes;}
    if(soa_capacity_ < cap || !soa_.suns_backup) {
        free_arrays_();
        if(!alloc_arrays_(cap)) {endrun(913401); soa_capacity_ = 0; soa_valid_ = 0; return;}  /* soft bad-stop: leave SoA invalid before the suns_backup seed reads NULL; drains at next poll */
        soa_capacity_ = cap;
        soa_valid_ = 1;   /* SoA is populated end-to-end by the build kernels;
                           * no AoS-seed needed.  Keep valid so acquire() fast-paths. */
    }
    /* Seed the topnode range (centers / lengths / suns_backup)
     * directly from UVM AoS into the SoA.  Nodes_base / Extnodes_base are
     * SharedSpace (6.8d); read/write happens inside the lambda.  emit_bfs
     * reads soa->center / soa->len for the topnode range to compute new
     * child centers — without this seed it would read garbage. */
    struct NODE             *Nodes_uvm   = Nodes_base;
    int                     *suns_soa    = soa_.suns_backup;
    Vec3<MyFloat>           *center_soa  = soa_.center;
    MyFloat                 *len_soa     = soa_.len;
    Kokkos::parallel_for("seed_topnode_geom", n, KOKKOS_LAMBDA(int k) {
        for(int j = 0; j < 8; j++) {
            suns_soa[(long)k * 8 + j] = Nodes_uvm[k].u.suns[j];
        }
        center_soa[k] = Nodes_uvm[k].center;
        len_soa[k]    = Nodes_uvm[k].len;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("seed_topnode_geom", n);
}

