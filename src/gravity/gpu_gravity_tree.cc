/* gpu_gravity_tree.cc
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
#include "forcetree.h"   /* force_treebuild_generation() — SoA-drift stamp invalidation key */


static struct gpu_gravity_tree_soa_t soa_ = {0};
static int soa_capacity_ = 0;
static int soa_valid_    = 0;

/* SoA node-geometry drift-freshness stamp.  The SoA node centers/len are
 * drift-refreshed to a given Ti by gpu_force_drift_nodes; this stamp records
 * WHEN that last succeeded, so a consumer can certify the device geometry is
 * current WITHOUT relying on "the gravity walk probably ran earlier".
 * Keyed on both the target Ti and the treebuild generation
 * (a rebuild repopulates node geometry even at unchanged capacity).  -1 = not
 * certified.  Invalidated on realloc/free/rebuild. */
static integertime g_soa_drift_ti  = -1;
static long        g_soa_drift_gen = -1;

static void gpu_gravity_soa_invalidate_drift_stamp_(void) { g_soa_drift_ti = -1; g_soa_drift_gen = -1; }

static void free_arrays_(void)
{
    /* SSOT: freeing any SoA buffer invalidates drift certification — covers every
     * free_arrays_ caller (acquire realloc, release, gpu_nextnode_backup_suns). */
    gpu_gravity_soa_invalidate_drift_stamp_();
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
    if(soa_.foreign_leaf_tag)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.foreign_leaf_tag);  soa_.foreign_leaf_tag  = NULL;}
    if(soa_.foreign_leaf_type) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.foreign_leaf_type); soa_.foreign_leaf_type = NULL;}
    if(soa_.foreign_leaf_zeta) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.foreign_leaf_zeta); soa_.foreign_leaf_zeta = NULL;}
    if(soa_.foreign_leaf_soft) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(soa_.foreign_leaf_soft); soa_.foreign_leaf_soft = NULL;}
    soa_.foreign_leaf_cap = 0;
    /* nextnode_aux is an alias to Nextnode[] (UVM, owned by
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
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
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
    /* Unconditional Extnodes mirrors. */
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
    /* Do NOT touch nextnode_aux / nextnode_aux_size here — the
     * alias is owned by force_treeallocate and outlives SoA realloc cycles. */
}

/* kokkos_malloc throws when it cannot serve the request, so the failure branches below --
 * which return 0 so the caller can request a controlled stop -- are unreachable if it is
 * called directly: the run aborts on the exception instead.  Returning NULL is what those
 * branches, and the transactional rollback in gpu_gravity_tree_grow_foreign, are written
 * against.  Nothing is caught when the allocation succeeds. */
static void *tree_soa_alloc(size_t bytes)
{
    try { return Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("gravity_tree_soa", bytes); }
    catch(const std::exception &) { return NULL; }
}

static int alloc_arrays_(int n)
{
    /* moment/force fields retyped as MyGravFloat (flag-gated:
     * float when GIZMO_MIXED_PRECISION_GRAVITY set, double otherwise).
     * Geometric (center, len) stays MyFloat — opening-criterion precision. */
    soa_.center   = (Vec3<MyFloat>     *) tree_soa_alloc(n * sizeof(Vec3<MyFloat>));
    soa_.len      = (MyFloat           *) tree_soa_alloc(n * sizeof(MyFloat));
    soa_.s        = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    soa_.mass     = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.sibling  = (int               *) tree_soa_alloc(n * sizeof(int));
    soa_.nextnode = (int               *) tree_soa_alloc(n * sizeof(int));
    soa_.father   = (int               *) tree_soa_alloc(n * sizeof(int));
    soa_.bitflags = (unsigned int      *) tree_soa_alloc(n * sizeof(unsigned int));
    soa_.maxsoft  = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.N_part   = (long              *) tree_soa_alloc(n * sizeof(long));
    soa_.suns_backup = (int             *) tree_soa_alloc((long)n * 8 * sizeof(int));
    if(!soa_.center || !soa_.len || !soa_.s || !soa_.mass || !soa_.sibling ||
       !soa_.nextnode || !soa_.father || !soa_.bitflags || !soa_.maxsoft || !soa_.N_part ||
       !soa_.suns_backup) {
        printf("gpu_gravity_tree: kokkos_malloc failed for %d nodes\n", n);
        return 0;
    }
    /* Foreign-leaf identity sidecar mirror -- sized by the foreign storage that exists, NOT n, and
     * indexed by foreign_slot = no-(TreeNodeIndexBase+MaxNodes) by both the scatter and the walk.  The
     * capacity requests below add the same count, so this runs with the current one. */
    soa_.foreign_leaf_cap = (AllocatedForeignNodes > 0) ? AllocatedForeignNodes : 0;
    if(soa_.foreign_leaf_cap > 0) {
        soa_.foreign_leaf_tag  = (int     *) tree_soa_alloc((size_t)soa_.foreign_leaf_cap * sizeof(int));
        soa_.foreign_leaf_type = (int     *) tree_soa_alloc((size_t)soa_.foreign_leaf_cap * sizeof(int));
        soa_.foreign_leaf_zeta = (MyFloat *) tree_soa_alloc((size_t)soa_.foreign_leaf_cap * sizeof(MyFloat));
        soa_.foreign_leaf_soft = (MyFloat *) tree_soa_alloc((size_t)soa_.foreign_leaf_cap * sizeof(MyFloat));
        if(!soa_.foreign_leaf_tag || !soa_.foreign_leaf_type || !soa_.foreign_leaf_zeta || !soa_.foreign_leaf_soft) {
            printf("gpu_gravity_tree: foreign_leaf sidecar alloc failed (%d)\n", soa_.foreign_leaf_cap); return 0;
        }
    } else {
        soa_.foreign_leaf_tag = NULL; soa_.foreign_leaf_type = NULL; soa_.foreign_leaf_zeta = NULL; soa_.foreign_leaf_soft = NULL;
    }
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    soa_.gasmass = (MyGravFloat *) tree_soa_alloc(n * sizeof(MyGravFloat));
    if(!soa_.gasmass) {printf("gpu_gravity_tree: gasmass alloc failed (%d)\n", n); return 0;}
#endif
#ifdef RT_USE_GRAVTREE
    soa_.stellar_lum = (MyGravFloat *) tree_soa_alloc(n * N_RT_FREQ_BINS * sizeof(MyGravFloat));
    if(!soa_.stellar_lum) {printf("gpu_gravity_tree: stellar_lum alloc failed (%d)\n", n); return 0;}
#ifdef CHIMES_STELLAR_FLUXES
    /* CHIMES stays double — chemistry tolerances. */
    soa_.chimes_stellar_lum_G0  = (double *) tree_soa_alloc(n * CHIMES_LOCAL_UV_NBINS * sizeof(double));
    soa_.chimes_stellar_lum_ion = (double *) tree_soa_alloc(n * CHIMES_LOCAL_UV_NBINS * sizeof(double));
    if(!soa_.chimes_stellar_lum_G0 || !soa_.chimes_stellar_lum_ion) {printf("gpu_gravity_tree: chimes lum alloc failed (%d)\n", n); return 0;}
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    soa_.rt_source_lum_s  = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    soa_.rt_source_lum_vs = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.rt_source_lum_s || !soa_.rt_source_lum_vs) {printf("gpu_gravity_tree: rt_source_lum alloc failed (%d)\n", n); return 0;}
#endif
#ifdef SINK_PHOTONMOMENTUM
    soa_.sink_lum      = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.sink_lum_grad = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_lum || !soa_.sink_lum_grad) {printf("gpu_gravity_tree: sink_lum alloc failed (%d)\n", n); return 0;}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    soa_.cr_injection = (MyGravFloat *) tree_soa_alloc(n * sizeof(MyGravFloat));
    if(!soa_.cr_injection) {printf("gpu_gravity_tree: cr_injection alloc failed (%d)\n", n); return 0;}
#endif
#ifdef SINK_CALC_DISTANCES
    soa_.sink_mass = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.sink_pos  = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_mass || !soa_.sink_pos) {printf("gpu_gravity_tree: sink_mass/pos alloc failed (%d)\n", n); return 0;}
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    soa_.sink_vel = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_vel) {printf("gpu_gravity_tree: sink_vel alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SPECIAL_POINT_MOTION)
    soa_.sink_acc = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.sink_acc) {printf("gpu_gravity_tree: sink_acc alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    soa_.N_SINK = (int *) tree_soa_alloc(n * sizeof(int));
    if(!soa_.N_SINK) {printf("gpu_gravity_tree: N_SINK alloc failed (%d)\n", n); return 0;}
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    soa_.MaxFeedbackVel = (MyGravFloat *) tree_soa_alloc(n * sizeof(MyGravFloat));
    if(!soa_.MaxFeedbackVel) {printf("gpu_gravity_tree: MaxFeedbackVel alloc failed (%d)\n", n); return 0;}
#endif
#endif
    /* Unconditional Extnodes mirrors (retyped for mixed-precision gravity). */
    soa_.node_vs = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    soa_.hmax    = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.vmax    = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.divVmax = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    if(!soa_.node_vs || !soa_.hmax || !soa_.vmax || !soa_.divVmax) {
        printf("gpu_gravity_tree: unconditional extnode mirrors alloc failed (n=%d)\n", n);
        return 0;
    }
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    soa_.tidal_tensorps = (MyGravFloat *) tree_soa_alloc(n * 6 * sizeof(MyGravFloat));
    if(!soa_.tidal_tensorps) {printf("gpu_gravity_tree: tidal_tensorps alloc failed (%d)\n", n); return 0;}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    soa_.mass_dm = (MyGravFloat       *) tree_soa_alloc(n * sizeof(MyGravFloat));
    soa_.s_dm    = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    soa_.vs_dm   = (Vec3<MyGravFloat> *) tree_soa_alloc(n * sizeof(Vec3<MyGravFloat>));
    if(!soa_.mass_dm || !soa_.s_dm || !soa_.vs_dm) {
        printf("gpu_gravity_tree: DM_SCALARFIELD mirrors alloc failed (n=%d)\n", n); return 0;
    }
#endif
    return 1;
}

/* Bytes this mirror needs per tree node, for the startup memory projection, which has to run
 * before any tree exists.  Only the fields present in EVERY build are counted, so the answer is
 * a deliberate under-estimate and the projection it feeds reports trouble only when the run
 * cannot fit whatever a configuration adds on top of them.  Kept next to alloc_arrays_ so the
 * two are read and edited together. */
extern "C" size_t gpu_gravity_tree_bytes_per_node(void)
{
    return sizeof(Vec3<MyFloat>)                 /* center   */
         + sizeof(MyFloat)                       /* len      */
         + sizeof(Vec3<MyGravFloat>)             /* s        */
         + sizeof(MyGravFloat)                   /* mass     */
         + 3 * sizeof(int)                       /* sibling, nextnode, father */
         + sizeof(unsigned int)                  /* bitflags */
         + sizeof(MyGravFloat)                   /* maxsoft  */
         + sizeof(long)                          /* N_part   */
         + 8 * sizeof(int)                       /* suns_backup */
         + sizeof(Vec3<MyGravFloat>)             /* node_vs  */
         + 3 * sizeof(MyGravFloat);              /* hmax, vmax, divVmax */
}

/* seed_node_ / seed_dirty_ / dirty_[] / mark_dirty / dirty_count_
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

    /* Extend SoA capacity to cover the foreign-node storage that currently exists,
     * [MaxNodes, MaxNodes+AllocatedForeignNodes).  Foreign nodes installed by
     * let_unpack_and_install are scattered into SoA at slot_base + j with
     * absolute index = TreeNodeIndexBase + MaxNodes + slot, so SoA index =
     * (MaxNodes + slot).  This is zero until the exchange has counted this rank's
     * import and gpu_gravity_tree_grow_foreign has extended the mirror -- deliberately
     * NOT the index ceiling MaxForeignNodes, which every rank shares and which would
     * reserve the whole run the worst rank's import. */
    if(AllocatedForeignNodes > 0) {
        min_nodes += AllocatedForeignNodes;
    }

    if(soa_capacity_ >= min_nodes && soa_.center) {
        return;   /* fast path: capacity already sufficient */
    }

    /* Capacity grew (or first call): allocate.  No seeding -- the build
     * pipeline will populate.  soa_valid_ stays 1 because callers downstream
     * of the build kernels expect a populated SoA; the kernels write before
     * any reader. */
    free_arrays_();   /* invalidates the drift stamp (SSOT inside free_arrays_) */
    /* Soft bad-stop, drained at the next poll.  Release what the failed attempt did allocate
     * rather than carrying it: an empty mirror is what the rest of this file reads as "nothing
     * here yet", so soa() gives NULL to the callers that check it and the foreign-grow path
     * takes its first-allocation branch instead of mistaking a stale pointer for a live mirror. */
    if(!alloc_arrays_(min_nodes)) {endrun(913101); free_arrays_(); soa_capacity_ = 0; soa_valid_ = 0; return;}
    soa_capacity_ = min_nodes;
    soa_valid_    = 1;
}

/* Extend the mirror to `min_nodes` slots WITHOUT losing what it already holds.
 *
 * acquire() above may throw the mirror away and reallocate, because it only ever runs before
 * the build kernels repopulate it.  This one runs after them: the local tree is finished and
 * lives in these arrays, and all that is being added is room for the foreign nodes about to be
 * installed.  So the existing contents are copied across.
 *
 * Transactional: the live arrays are replaced only once the new set is fully allocated and
 * copied.  If allocation fails the old set is still installed and intact, and the caller gets a
 * failure to report -- never a half-swapped mirror or a freed pointer.
 *
 * The field list below mirrors alloc_arrays_ entry for entry, including the #ifdef guards; the
 * two are kept adjacent so they can be read side by side.  Only the prefix already in use is
 * copied -- the foreign range is written by gpu_scatter_foreign_to_soa right after this returns.
 * Returns 1 on success, 0 on failure. */
extern "C" int gpu_gravity_tree_grow_foreign(int min_nodes)
{
    /* Enough node slots is not enough on its own: the sidecar mirror is sized separately, from
     * AllocatedForeignNodes, so it has to be checked on its own terms or a grow that only widened
     * the foreign range would leave the scatter silently dropping leaf identities. */
    if(min_nodes <= soa_capacity_ && soa_.center
       && soa_.foreign_leaf_cap >= AllocatedForeignNodes) {return 1;}   /* already large enough */
    if(!soa_.center)
    {
        /* Nothing has been allocated yet, so there is nothing to preserve and this is simply the
         * first allocation.  Reached when the build pipeline had no nodes to seed. */
        if(!alloc_arrays_(min_nodes)) {return 0;}
        soa_capacity_ = min_nodes;
        soa_valid_    = 1;
        return 1;
    }

    const struct gpu_gravity_tree_soa_t old_soa = soa_;
    const int n_old = (soa_capacity_ < min_nodes) ? soa_capacity_ : min_nodes;

    /* Hand alloc_arrays_ a BLANK set to fill.  It returns at the first field it cannot allocate,
     * leaving every later field as it found it -- so if it were given the live set, the rollback
     * below would free the arrays the tree is still using and then reinstall those freed pointers.
     * Starting blank means the rollback can only ever free what this call itself allocated.
     * The Nextnode alias is not owned here and is carried across untouched. */
    {
        struct gpu_gravity_tree_soa_t blank = {0};
        blank.nextnode_aux      = soa_.nextnode_aux;
        blank.nextnode_aux_size = soa_.nextnode_aux_size;
        soa_ = blank;
    }
    if(!alloc_arrays_(min_nodes))
    {
        free_arrays_();        /* drop the partially-allocated new set, and only that */
        soa_ = old_soa;        /* the tree's mirror is untouched and still current */
        return 0;
    }
    const struct gpu_gravity_tree_soa_t new_soa = soa_;

#define GIZMO_SOA_COPY(field, count) \
    do { if(new_soa.field && old_soa.field) {memcpy(new_soa.field, old_soa.field, (size_t)(count) * sizeof(*new_soa.field));} } while(0)
    GIZMO_SOA_COPY(center,   n_old);
    GIZMO_SOA_COPY(len,      n_old);
    GIZMO_SOA_COPY(s,        n_old);
    GIZMO_SOA_COPY(mass,     n_old);
    GIZMO_SOA_COPY(sibling,  n_old);
    GIZMO_SOA_COPY(nextnode, n_old);
    GIZMO_SOA_COPY(father,   n_old);
    GIZMO_SOA_COPY(bitflags, n_old);
    GIZMO_SOA_COPY(maxsoft,  n_old);
    GIZMO_SOA_COPY(N_part,   n_old);
    GIZMO_SOA_COPY(suns_backup, (long) n_old * 8);
    /* foreign_leaf_* are deliberately NOT copied: they describe the previous import, which the
     * scatter overwrites for every slot it installs. */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    GIZMO_SOA_COPY(gasmass, n_old);
#endif
#ifdef RT_USE_GRAVTREE
    GIZMO_SOA_COPY(stellar_lum, (long) n_old * N_RT_FREQ_BINS);
#ifdef CHIMES_STELLAR_FLUXES
    GIZMO_SOA_COPY(chimes_stellar_lum_G0,  (long) n_old * CHIMES_LOCAL_UV_NBINS);
    GIZMO_SOA_COPY(chimes_stellar_lum_ion, (long) n_old * CHIMES_LOCAL_UV_NBINS);
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    GIZMO_SOA_COPY(rt_source_lum_s,  n_old);
    GIZMO_SOA_COPY(rt_source_lum_vs, n_old);
#endif
#ifdef SINK_PHOTONMOMENTUM
    GIZMO_SOA_COPY(sink_lum,      n_old);
    GIZMO_SOA_COPY(sink_lum_grad, n_old);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    GIZMO_SOA_COPY(cr_injection, n_old);
#endif
#ifdef SINK_CALC_DISTANCES
    GIZMO_SOA_COPY(sink_mass, n_old);
    GIZMO_SOA_COPY(sink_pos,  n_old);
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    GIZMO_SOA_COPY(sink_vel, n_old);
#endif
#if defined(SPECIAL_POINT_MOTION)
    GIZMO_SOA_COPY(sink_acc, n_old);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    GIZMO_SOA_COPY(N_SINK, n_old);
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    GIZMO_SOA_COPY(MaxFeedbackVel, n_old);
#endif
#endif
    GIZMO_SOA_COPY(node_vs, n_old);
    GIZMO_SOA_COPY(hmax,    n_old);
    GIZMO_SOA_COPY(vmax,    n_old);
    GIZMO_SOA_COPY(divVmax, n_old);
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    GIZMO_SOA_COPY(tidal_tensorps, (long) n_old * 6);
#endif
#ifdef DM_SCALARFIELD_SCREENING
    GIZMO_SOA_COPY(mass_dm, n_old);
    GIZMO_SOA_COPY(s_dm,    n_old);
    GIZMO_SOA_COPY(vs_dm,   n_old);
#endif
#undef GIZMO_SOA_COPY

    /* The copies above are host stores into SharedSpace that device kernels will read; fence
     * before the old buffers go away, for the same reason the scatter path fences. */
    Kokkos::fence();

    soa_ = old_soa;    /* release the superseded set through the one function that knows it */
    free_arrays_();
    soa_ = new_soa;
    soa_capacity_ = min_nodes;
    soa_valid_    = 1;
    return 1;
}

extern "C" void gpu_gravity_tree_release(void)
{
    free_arrays_();   /* invalidates the drift stamp (SSOT inside free_arrays_) */
    soa_capacity_ = 0;
    soa_valid_    = 0;
    gpu_force_drift_release();
    gpu_moment_refresh_release();
}

/* Record node geometry drifted to `ti` (snapshot current treebuild gen). Set by
 * the drift sweep's success path; read by the read-only certification query
 * below, which is its only consumer. */
extern "C" void gpu_gravity_soa_mark_drift_certified(integertime ti)
{
    g_soa_drift_ti  = ti;
    g_soa_drift_gen = force_treebuild_generation();
}

/* Pure O(1) read-only certification query (no drift, no node loop). */
extern "C" int gpu_gravity_soa_drift_certified(integertime ti)
{
    return (g_soa_drift_ti == ti && g_soa_drift_gen == force_treebuild_generation()) ? 1 : 0;
}

extern "C" void gpu_gravity_tree_alias_nextnode(int *Nextnode_host, int n)
{
    /* Alias soa->nextnode_aux to the SharedSpace Nextnode[] owned
     * by force_treeallocate.  No separate buffer, no per-walk memcpy.  Called
     * once per (allocate / free) cycle from forcetree.cc. */
    soa_.nextnode_aux      = Nextnode_host;
    soa_.nextnode_aux_size = n;
}

extern "C" struct gpu_gravity_tree_soa_t *gpu_gravity_tree_soa(void) {return soa_valid_ ? &soa_ : NULL;}
extern "C" int gpu_gravity_tree_capacity(void)                       {return soa_capacity_;}
extern "C" int gpu_gravity_tree_valid(void)                          {return soa_valid_;}

/* Snapshot Nodes_base[k].u.suns[0..7] for k in [0..n) into the
 * SoA's suns_backup buffer.  Called from force_treebuild_single right
 * before force_update_node_recursive overwrites the union with the d struct.
 * If the SoA hasn't been allocated yet (first acquire after a fresh
 * allocation), set up minimal scaffolding so the buffer exists.  Otherwise
 * just refresh the contents from current AoS. */
extern "C" void gpu_nextnode_backup_suns(int n)
{
    if(n <= 0) {return;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Allocate at MaxNodes+1 (full-tree capacity), not just n (=Numnodestree).
     * If we only allocated at n, the subsequent gpu_gravity_tree_acquire(MaxNodes+1)
     * inside gpu_nextnode_thread would see soa_capacity_ < MaxNodes+1, call
     * free_arrays_(), and destroy the suns_backup we store below — corrupting
     * the nextnode kernel's input on the first treebuild. */
    int cap = (MaxNodes > 0) ? MaxNodes + 1 : n;
    /* Include whatever foreign-node storage exists.  At build time that is normally none:
     * the import is counted, and the mirror extended to fit it, only once the tree is built. */
    if(AllocatedForeignNodes > 0) {cap += AllocatedForeignNodes;}
    if(soa_capacity_ < cap || !soa_.suns_backup) {
        free_arrays_();
        if(!alloc_arrays_(cap)) {endrun(913401); soa_capacity_ = 0; soa_valid_ = 0; return;}  /* soft bad-stop: leave SoA invalid before the suns_backup seed reads NULL; drains at next poll */
        soa_capacity_ = cap;
        soa_valid_ = 1;   /* SoA is populated end-to-end by the build kernels;
                           * no AoS-seed needed.  Keep valid so acquire() fast-paths. */
    }
    /* Seed the topnode range (centers / lengths / suns_backup)
     * directly from UVM AoS into the SoA.  Nodes_base / Extnodes_base are
     * SharedSpace; read/write happens inside the lambda.  emit_bfs
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

