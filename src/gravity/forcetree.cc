#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "forcetree.h"               /* GIZMO_EWALD_EN + Ewald-table accessor decls */
#include "gravtree_force_kernel.h"   /* shared CPU/GPU accepted-source contribution physics (SSOT) */
#include "gravtree_moment_kernel.h"  /* shared node moment/payload construction physics (SSOT); plain primitives only here */
#include "gravtree_ewald.h"          /* shared CPU/GPU Ewald image-correction trilinear interp (SSOT) */
#include "pm_highres_region.h"       /* pmforce_is_particle_high_res SSOT (device-callable) */
#include "let_data.h"   /* Phase 9.1b: LET wire format + per-rank payload structs (compile-only here; consumers will land in 9.1c-e) */
#include "../mesh/gpu_neighbor_list.h" /* gizmo_mark_kernel_radius_dirty_indices */
#include "../mesh/nlr_radius_policy.h" /* SSOT helper for force_hmax_per_type_particle_radius */
#ifdef SUBFIND
#include "../structure/subfind/subfind.h"
#endif
#include "gpu_gravity_tree.h"
#include "gpu_peano_walk.h"
#include "gpu_topology_build.h"
#include "gpu_topology_finalize.h"
#include "gpu_pseudo_update.h"

/*! \file forcetree.c
 *  \brief gravitational tree and code for Ewald correction
 *
 *  This file contains the computation of the gravitational force by means
 *  of a tree. The type of tree implemented is a geometrical oct-tree,
 *  starting from a cube encompassing all particles. This cube is
 *  automatically found in the domain decomposition, which also splits up
 *  the global "top-level" tree along node boundaries, moving the particles
 *  of different parts of the tree to separate processors. Tree nodes can
 *  be dynamically updated in drift/kick operations to avoid having to
 *  reconstruct the tree every timestep.
 */
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * substantially (condensed, new feedback routines added, many different
 * types of walk and calculations added, structures in memory changed,
 * switched options for nodes, optimizations, new physics modules and
 * calcutions, and new variable/memory conventions added)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 * Mike Grudic has also made major revisions to code the Hermitian calculations and binary timestepping.
 */


/* Compute the force-softening kernel radius for one particle.
 *
 * Single source of truth for the per-particle softening logic.  All call sites
 * (CPU walk, GPU walk, tree-build split-scale, etc.) read the cached result via
 * ForceSoftening_KernelRadius() / gpu_force_softening_kernel_radius(); this
 * routine is the only place that performs the actual computation.
 *
 * The cache is refreshed once per gravity_tree() call by compute_all_force_softening()
 * (active-particle loop) and seeded over all particles at startup in init.c.  Inputs
 * (KernelRadius, AGS_KernelRadius, tidal_tensor_mag_prev, StarParticleEffectiveSize)
 * only mutate when the particle is active, so cached values for inactive particles
 * remain correct between active steps. */
double compute_force_softening_kernel_radius(int p)
{
#ifdef GALSF_MERGER_STARCLUSTER_PARTICLES
    if(P[p].Type == 4) {return P[p].StarParticleEffectiveSize;} // this variable is defined in force softening terms
    //if(P[p].Type == 4) {return All.ForceSoftening[4] * pow(P[p].Mass / (0.5*(All.MaxMassForParticleSplit/3.01+All.MinMassForParticleMerger/0.49)),0.333);} // alternative 'adaptive' version for constant-resolution runs
    //if(P[p].Type == 4) {return All.ForceSoftening[4] * pow(P[p].Mass*UNIT_MASS_IN_SOLAR / (GALSF_MERGER_STARCLUSTER_PARTICLES),0.333);}
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if((1 << P[p].Type) & (ADAPTIVE_GRAVSOFT_FORALL)) {return P[p].AGS_KernelRadius;}
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(SELFGRAVITY_OFF) /* softening scale still appears in timestep criterion for problems without self-gravity, so set it adaptively */
#ifdef ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT
    if(P[p].Type == 0) {return DMIN(P[p].KernelRadius, ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT/All.cf_atime);}
#else
    if(P[p].Type == 0) {return P[p].KernelRadius;}
#endif
#endif

#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
    if(P[p].Type == 4) {return All.ForceSoftening[P[p].Type] * DMIN(100., DMAX(1., pow(P[p].Mass*UNIT_MASS_IN_SOLAR/100. , 0.33)));}
#endif

#if defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION) /* still playing with criterion below, highly experimental for now */
    if((1 << P[p].Type) & (ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)) {if((P[p].tidal_tensor_mag_prev>0) && (All.Time>All.TimeBegin)) {return DMIN(1.e2*All.ForceSoftening[P[p].Type] , DMAX(All.ForceSoftening[P[p].Type] , All.ForceSoftening[P[p].Type] + 1.25 * pow( (All.DesNumNgb * All.G * P[p].Mass / P[p].tidal_tensor_mag_prev) , 1./3. )));} else {return 100.*All.ForceSoftening[P[p].Type];}}
#endif

    return All.ForceSoftening[P[p].Type]; // this is the default if nothing was active above
}

/* Public accessor: returns the cached value populated by compute_all_force_softening().
 * Both CPU and GPU walks read this same value, so the softening logic above lives in
 * exactly one place. */
double ForceSoftening_KernelRadius(int p)
{
    return P[p].ForceSoftening;
}

/* Refresh the per-particle ForceSoftening cache.  Called from gravity_tree() at
 * the start of every walk dispatch (active particles only) and from init.c during
 * startup (all particles).  Inputs to compute_force_softening_kernel_radius() only
 * change for active particles within a timestep, so an active-particle pass is
 * sufficient for steady-state operation; the init pass seeds inactive particles
 * loaded from the IC file or spawned mid-run. */
void compute_all_force_softening(int mode)
{
    /* mode = 0 : active particles only (FirstActiveParticle list)
     * mode = 1 : every particle in [0, NumPart) -- used at startup / after restart  */
    if(mode == 1)
    {
        int i;
#pragma omp parallel for schedule(static)
        for(i = 0; i < NumPart; i++) {P[i].ForceSoftening = compute_force_softening_kernel_radius(i);}
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int ii = 0; ii < (int)ActiveParticleList.size(); ii++)
        {
            int i = ActiveParticleList[ii];
            P[i].ForceSoftening = compute_force_softening_kernel_radius(i);
        }
    }
}



/*! auxiliary variable used to set-up non-recursive walk */
static int last;

/* NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE is defined globally in
 * precompiler_logic.h (via allvars.h) so the CPU and GPU tree walks share the
 * same leaf-opening criterion. */

/*! length of look-up table for short-range force kernel in TreePM algorithm */
#define NTAB GRAVTREE_SHORTRANGE_NTAB   /* table length owned by gravtree_force_kernel.h (shared with the GPU walk) */
/*! variables for short-range lookup table.  Non-static so the GPU gravity
 *  walk in gpu_gravtree.cc can read them via extern declarations.  Sized at
 *  NTAB floats = 4 KB each — fine to leave in host memory on Kokkos OMP;
 *  for true device offload they will need mirroring (Phase 4 follow-up). */
float shortrange_table[NTAB], shortrange_table_potential[NTAB];
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
/* Non-static so the GPU walk in gpu_gravtree.cc can read this via extern (mirrors
 * the shortrange_table / shortrange_table_potential pattern above). */
float shortrange_table_tidal[NTAB];
#endif
/*! toggles after first tree-memory allocation, has only influence on log-files */
static int first_flag = 0;
static int tree_allocated_flag = 0;

#ifdef BOX_PERIODIC
/*! Size of 3D look-up table for Ewald correction force */
#define EN  64
/*! 3D look-up table for Ewald correction to force and potential. Only one octant is stored, the rest constructed by using the symmetry of the problem */
static MyFloat fcorrx[EN + 1][EN + 1][EN + 1];
static MyFloat fcorry[EN + 1][EN + 1][EN + 1];
static MyFloat fcorrz[EN + 1][EN + 1][EN + 1];
static MyFloat potcorr[EN + 1][EN + 1][EN + 1];
static double fac_intp;
#if !defined(GRAVITY_NOT_PERIODIC)
/* Accessor for gpu_gravtree.cc — see forcetree.h. */
void gizmo_get_ewald_tables(const MyFloat **fcorrx_out, const MyFloat **fcorry_out,
                            const MyFloat **fcorrz_out, const MyFloat **potcorr_out,
                            double *fac_intp_out)
{
    *fcorrx_out  = &fcorrx[0][0][0];
    *fcorry_out  = &fcorry[0][0][0];
    *fcorrz_out  = &fcorrz[0][0][0];
    *potcorr_out = &potcorr[0][0][0];
    *fac_intp_out = fac_intp;
}
#endif
#endif


/* Gravity box-wrap policy is centralized in the shared helpers (single source used by both
 * the CPU tree walk here and the GPU walks). The gravity gate (wrap only for periodic box
 * AND periodic gravity, else no-op/abs) lives inside gravity_box_distance.h. The GRAVITY_*
 * macros below are thin wrappers so the existing call sites are unchanged. */
#include "gravity_box_distance.h"
#include "gravtree_opening.h"
#define GRAVITY_NEAREST_XYZ(x,y,z,sign)             gravity_box_nearest_image(x,y,z,sign)
#define GRAVITY_NGB_PERIODIC_BOX_LONG_X(x,y,z,sign) gravity_box_long_abs_x(x,y,z,sign)
#define GRAVITY_NGB_PERIODIC_BOX_LONG_Y(x,y,z,sign) gravity_box_long_abs_y(x,y,z,sign)
#define GRAVITY_NGB_PERIODIC_BOX_LONG_Z(x,y,z,sign) gravity_box_long_abs_z(x,y,z,sign)

/*! This function is a driver routine for constructing the gravitational
 *  oct-tree, which is done by calling a small number of other functions.
 */
/*! Mode B per-type hmax host-side re-seed.
 *
 *  Why this exists: gpu_moment_refresh() writes the scalar Extnodes[no].hmax
 *  to AoS but not the per-type bands Extnodes[no].hmax_per_type[].  The GPU
 *  SoA intentionally does not carry per-type bands -- their only consumer is
 *  this host-side Mode B walker (mesh/mode_b_local_walker.cc) -- so the GPU
 *  moment path bypasses the host moment loop that would otherwise seed them.
 *  Without this pass every full force_treebuild and every
 *  force_refresh_node_moments would leave the bands at zero, and Mode B's
 *  SYMMETRIC walker reading zero bands would over-prune (collapse to ONEWAY)
 *  -- exactly the oracle mismatch observed on fire_m11i.
 *
 *  Behavior: zero all internal-node bands, leaf-seed each particle's
 *  conservative radius into Father[i]'s band, then bottom-up max-over-children
 *  (cheap host loop ~O(NumPart + Nnodes)).  Caller-restriction: must run AFTER
 *  gpu_moment_refresh has populated Father[] / Nodes[].u.d.father, since the
 *  bottom-up step walks the father chain.
 */
/* Conservative per-particle radius (node-prune upper bound) for
 * Extnodes[no].hmax_per_type[Type] band seeding. See forcetree.h docstring.
 * Used by every site that grows or seeds a per-type band:
 * force_refresh_hmax_per_type_host, force_update_node_recursive,
 * force_refresh_node_moments, force_update_hmax (forcetree_update.cc),
 * force_add_element_to_tree.
 *
 * SSOT: routes through nlr_particle_symmetric_radius_capped — no
 * AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE branch here.  MODE_B_RADIUS_ALL_SOURCES
 * is the conservative union (KernelRadius / AGS_KernelRadius / ForceSoftening
 * across all types).  Capping at All.MaxKernelRadius applies to kernel radii
 * only; ForceSoftening is uncapped so the band dominates the leaf-policy reach
 * even when FS > All.MaxKernelRadius (codex 2026-06-07). */
double force_hmax_per_type_particle_radius(int i)
{
    return nlr_particle_symmetric_radius_capped(P[i],
                                                MODE_B_RADIUS_ALL_SOURCES,
                                                (double)All.MaxKernelRadius);
}

static void force_refresh_hmax_per_type_host(int Numnodestree)
{
    /* Step 1: zero per-type bands in all internal nodes. */
    for(int no = All.MaxPart; no < All.MaxPart + Numnodestree; no++) {
        for(int t = 0; t < 6; t++) Extnodes[no].hmax_per_type[t] = 0;
    }
    /* Step 2: leaf seed — Father[i] only (per-particle to its immediate
     * parent), conservative across every leaf-policy-selectable source. */
    for(int i = 0; i < NumPart; i++) {
        int no = Father[i];
        if(no < 0) continue;
        struct particle_data *pa = &P[i];
        if(pa->Mass <= 0) continue;
        double htmp = force_hmax_per_type_particle_radius(i);
        int t = (int)pa->Type;
        if(htmp > Extnodes[no].hmax_per_type[t]) Extnodes[no].hmax_per_type[t] = (MyFloat)htmp;
    }
    /* Step 3: bottom-up max-over-children via father chain. Children always
     * allocated at higher indices than parents, so reverse iteration gives
     * children-before-parent order without an explicit DAG sort. */
    for(int no = All.MaxPart + Numnodestree - 1; no >= All.MaxPart; no--) {
        int father = Nodes[no].u.d.father;
        if(father < All.MaxPart || father >= All.MaxPart + Numnodestree) continue;
        for(int t = 0; t < 6; t++) {
            if(Extnodes[no].hmax_per_type[t] > Extnodes[father].hmax_per_type[t]) {
                Extnodes[father].hmax_per_type[t] = Extnodes[no].hmax_per_type[t];
            }
        }
    }
}

/* Gravity-tree freshness generations (see forcetree.h). Plain host counters,
 * SSOT in this TU; force_update_hmax (forcetree_update.cc) bumps the hmax one
 * via force_bump_hmax_refresh_generation(). */
static long g_force_treebuild_generation = 0;
static long g_force_hmax_refresh_generation = 0;
long force_treebuild_generation(void)        { return g_force_treebuild_generation; }
long force_hmax_refresh_generation(void)      { return g_force_hmax_refresh_generation; }
void force_bump_hmax_refresh_generation(void) { g_force_hmax_refresh_generation++; }

int force_treebuild(int npart, struct unbind_data *mp)
{
    int flag;
    /* Adaptive LET foreign-arena retry: force_treebuild owns the rebuild loop (same
     * idiom as the TreeAllocFactor-overflow retry below). On a retryable LET overflow
     * we grow the adaptive floor and rebuild the whole tree from scratch -- the foreign
     * arena is contiguous inside Nodes_base, so growing it means reallocating + rebuilding.
     * One rebuild per ratchet event suffices (the in-call domain is fixed, so the rebuilt
     * LET needs the same capacity and the 1.5x headroom covers it); the bound is a backstop. */
    int let_retry = 0;
    const int LET_MAX_RETRY = 3;
let_build_attempt:
    /* Phase 9.6: reset force_add_element insertion counter at each full build. */
    ForceAddElementToTree_CallsSinceBuild = 0;
    do
    {
        Numnodestree = force_treebuild_single(npart, mp);
        MPI_Allreduce(&Numnodestree, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if(flag == -1)
        {
            force_treefree();
            if(ThisTask == 0) {printf("Increasing TreeAllocFactor=%g", All.TreeAllocFactor);}
            All.TreeAllocFactor *= 1.15;
            if(ThisTask == 0) {printf(" new value=%g\n", All.TreeAllocFactor);}
            force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
            /* drain a tree-alloc UVM OOM before force_treebuild_single re-runs on a NULL-based
             * tree. Symmetric: all ranks enter this block together (flag from Allreduce(MIN)). */
            gizmo_exit_bad_stop_if_requested("gravtree:treeallocate");
        }
    }
    while(flag == -1);
    /* GPU finalize stage replaces force_update_node_recursive's
     * sibling/father/Father[] outputs.  Order matters:
     *   1. finalize_father: writes soa->father for all internal nodes
     *      (covers topnodes — emit_bfs only set inside-topleaf), and
     *      writes Father[i] for every particle child.  Must run before
     *      moment_refresh (which reads soa->father in its dependency walk).
     *   2. finalize_sibling: writes soa->sibling for all internal nodes.
     *      Must run before nextnode_thread (which reads soa->sibling).
     *   3. moment_refresh: writes moments + Extnodes/N_part/maxsoft/bitflags.
     *   4. nextnode_thread: writes nextnode + Nextnode[] from suns_backup.
     *   5. writeback_d_to_aos: pushes soa->sibling/father into AoS u.d for
     *      legacy CPU walks.  Clobbers u.suns via union, but suns_backup
     *      in SoA is the truth.  Runs last so prior steps reading SoA see
     *      consistent state.  topnode-range center/len was already
     *      pulled into SoA by gpu_nextnode_backup_suns inside
     *      force_treebuild_single. */
    /* The GPU tree-finalize steps below are rank-local; on failure they set
     * a soft bad-stop and fall through the matched, topology-driven pseudo/
     * LET collectives, which drain at the gravtree:after_treebuild poll
     * before the GPU gravity walk reads any moments. */
    if(gpu_topology_finalize_father(Numnodestree)  != 0) {endrun(90000065);}
    if(gpu_topology_finalize_sibling(Numnodestree) != 0) {endrun(90000066);}
    /* Phase 6.8f: GPU kernel resets GravCost + ephemeral fields for all
     * nodes.  On the CPU path FUNR does this work inline; on the GPU path
     * FUNR is retired (6.6) so the kernel takes its place.  Replaces a
     * host loop over Numnodestree -- the worst sparse-active scaling. */
    if(gpu_node_reset_ephemeral(Numnodestree) != 0) {endrun(90000067);}
    if(gpu_moment_refresh(-1) != 0) {endrun(90000068);}
    if(gpu_nextnode_thread() != 0) {endrun(90000069);}
    if(gpu_topology_writeback_d_to_aos(Numnodestree) != 0) {endrun(90000070);}
    /* Mode B: GPU moment refresh writes scalar hmax but not per-type bands;
     * re-seed those host-side now. MUST run AFTER gpu_topology_writeback_d_to_aos
     * because force_refresh_hmax_per_type_host's Step 3 propagation walks via
     * Nodes[no].u.d.father, which is only valid post-writeback (the SoA→AoS
     * writeback overwrites the union slot from the build-time u.suns layout). */
    force_refresh_hmax_per_type_host(Numnodestree);
    /* Phase 6.7a: set TOPLEVEL/INTERNAL_TOPLEVEL/DEPENDS bitflags in SoA
     * (and mirror to AoS for force_exchange_pseudodata / force_treeupdate_pseudos
     * which still run on CPU in 6.7a). */
    if(gpu_force_flag_localnodes() != 0) {endrun(90000071);}
    /* Phase 10.3 (B): post the pseudo-data Iallgathervs first, then run the
     * LET MPI round concurrently, then wait/unpack pseudo-data and resum.
     * LET pack reads only LOCAL Nodes/Extnodes (which are already valid from
     * gpu_moment_refresh above) and does not depend on foreign topleaves; so
     * the two MPI exchanges can overlap.  Latency drops from sum to max of
     * the two collectives' wall-times. */
    force_exchange_pseudodata_issue();
    /* LET exchange returns a typed status + (on overflow) the foreign-node capacity the
     * receiver needed. The pseudodata Iallgatherv posted just above must still be
     * completed regardless, so no early return here. */
    long long foreign_needed = 0;
    let_exchange_status_t let_status = let_run_exchange(&foreign_needed);
    int pseudo_status = force_exchange_pseudodata_complete();

    /* Classify globally (worst status wins). A send-buffer malloc failure
     * (LET_PACK_OOM), a malformed exchange (LET_UNPACK_INTERNAL), or a pseudodata
     * failure are HARD -- graceful stop, no retry (a bigger arena cannot fix them).
     * A LET_OVERFLOW_RETRYABLE on any rank triggers an arena grow + full rebuild.
     * Deciding globally keeps every rank on the same path (downstream + retry are
     * collective). */
    int overflow_local = (let_status == LET_OVERFLOW_RETRYABLE);
    int hardfail_local = (let_status == LET_PACK_OOM) || (let_status == LET_UNPACK_INTERNAL) || (pseudo_status != 0);
    int overflow_any = 0, hardfail_any = 0;
    long long need_max = 0;
    MPI_Allreduce(&overflow_local, &overflow_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&hardfail_local, &hardfail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&foreign_needed, &need_max, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if(hardfail_any)
    {
        /* Non-retryable. The failing rank records the cause; pseudodata failures
         * already soft-stopped inside force_exchange_pseudodata_complete(). Skip the
         * foreign-moment scatter/finalize/resum; drains at gravtree:after_treebuild. */
        if(let_status == LET_PACK_OOM || let_status == LET_UNPACK_INTERNAL) {endrun(90000072);}
    }
    else if(overflow_any)
    {
        if(let_retry >= LET_MAX_RETRY)
        {
            /* Should not happen: one rebuild per ratchet suffices (fixed in-call domain
             * + 1.5x headroom). Each overflowing rank reports its own need, then stop. */
            if(overflow_local)
                printf("LET foreign-arena overflow persists after %d retries on rank=%d: needed %lld nodes > MaxForeignNodes=%d (MaxNodes=%d). Stopping.\n",
                       LET_MAX_RETRY, ThisTask, foreign_needed, MaxForeignNodes, MaxNodes);
            fflush(stdout);
            endrun(90000062);   /* graceful drain; skip the foreign-moment steps */
        }
        else
        {
            long long want = (long long) ceil(1.5 * (double) need_max);
            if(want > RuntimeMinLETForeignNodes) {RuntimeMinLETForeignNodes = want;}
            if(ThisTask == 0)
                printf("LET foreign arena too small (needed up to %lld nodes); growing adaptive floor to %lld and rebuilding the tree (retry %d/%d).\n",
                       need_max, RuntimeMinLETForeignNodes, let_retry + 1, LET_MAX_RETRY);
            fflush(stdout);
            let_retry++;
            force_treefree();
            force_treeallocate((int) (All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
            /* drain a tree-alloc UVM OOM before rebuilding on a NULL-based tree (symmetric) */
            gizmo_exit_bad_stop_if_requested("gravtree:treeallocate");
            goto let_build_attempt;   /* rebuild the whole tree with the larger arena */
        }
    }
    else
    {
        /* Success: foreign moments complete. Scatter AoS→SoA, finalize LET
         * completeness, then re-sum ancestor topnode moments. */
        if(gpu_scatter_pseudo_to_soa() != 0)    {endrun(90000073);}
        let_finalize_unredirected_foreign_topleaves();
        if(gpu_topnode_moment_resum() != 0)     {endrun(90000074);}
    }
    TimeOfLastTreeConstruction = All.Time;
    g_force_treebuild_generation++;   /* topology + Father[] + node structure changed */
    return Numnodestree;
}



/*! Constructs the gravitational oct-tree.
 *
 *  The index convention for accessing tree nodes is the following: the
 *  indices 0...NumPart-1 reference single particles, the indices
 *  All.MaxPart.... All.MaxPart+nodes-1 reference tree nodes. `Nodes_base'
 *  points to the first tree node, while `nodes' is shifted such that
 *  nodes[All.MaxPart] gives the first tree node. Finally, node indices
 *  with values 'All.MaxPart + MaxNodes + MaxForeignNodes' and larger indicate "pseudo
 *  particles", i.e. multipole moments of top-level nodes that lie on
 *  different CPUs. If such a node needs to be opened, the corresponding
 *  particle must be exported to that CPU. The 'Extnodes' structure
 *  parallels that of 'Nodes'. Its information is only needed for the hydro
 *  part of the computation. (The data is split onto these two structures
 *  as a tuning measure.  If it is merged into 'Nodes' a somewhat bigger
 *  size of the nodes also for gravity would result, which would reduce
 *  cache utilization slightly.
 */
int force_treebuild_single(int npart, struct unbind_data *mp)
{
    int i, j, k, subnode = 0, shift, parent, numnodes, rep, nfree, th, nn, no;
    struct NODE *nfreep;
    MyFloat lenhalf;
    peanokey key, morton, th_key, *morton_list;
    
    /* create an empty root node  */
    nfree = All.MaxPart;        /* index of first free node */
    nfreep = &Nodes[nfree];    /* select first node */
    nfreep->len = DomainLen;
    nfreep->center = {(MyFloat)DomainCenter[0], (MyFloat)DomainCenter[1], (MyFloat)DomainCenter[2]};
    for(j = 0; j < 8; j++) {nfreep->u.suns[j] = -1;}
    numnodes = 1;
    nfreep++;
    nfree++;
    
    /* create a set of empty nodes corresponding to the top-level domain grid. We need to generate these nodes first to make sure that we have a
     * complete top-level tree which allows the easy insertion of the pseudo-particles at the right place */
    
    /* Root topnode 0 maps to the root tree node All.MaxPart; children are mapped
     * inside the recursion (top-leaf router geometry SSOT, H0). */
    TopNodeNodeIndex[0] = All.MaxPart;
    if(TopNodes[0].Daughter < 0) {
        /* Degenerate root-is-leaf (single top-cell): the recursion below sets no
         * children, so set the leaf's DomainNodeIndex explicitly to keep both maps
         * complete + consistent. */
        DomainNodeIndex[TopNodes[0].Leaf] = All.MaxPart;
    }
    if(force_create_empty_nodes(All.MaxPart, 0, 1, 0, 0, 0, &numnodes, &nfree) < 0) {return -1;}
    /* H0 post-build validation: every topnode must map to a valid Nodes[] slot. */
    {
        const int node_lo = All.MaxPart, node_hi = All.MaxPart + MaxNodes;
        for(int tnchk = 0; tnchk < NTopnodes; tnchk++) {
            if(TopNodeNodeIndex[tnchk] < node_lo || TopNodeNodeIndex[tnchk] >= node_hi) {
                printf("force_treebuild: TopNodeNodeIndex[%d]=%d out of range [%d,%d) (NTopnodes=%d) — top-leaf router map incomplete\n",
                       tnchk, TopNodeNodeIndex[tnchk], node_lo, node_hi, NTopnodes);
                endrun(91561);
            }
        }
    }
    /* if a high-resolution region in a global tree is used, we need to generate an additional set empty nodes to make sure that we have a complete top-level tree for the high-resolution inset */

    /* Step 13 Phase 6.5d: GPU tree-build replaces the per-particle CPU
     * insertion loop for inside-topleaf topology.  Order on GPU compile:
     *   1. force_insert_pseudo_particles (modifies foreign-topleaf u.suns).
     *   2. Acquire SoA + Peano-walk mirrors.
     *   3. gpu_topology_build_data_path: per-particle Peano walk + Morton
     *      sort within each topleaf.
     *   4. gpu_topology_emit_bfs: BFS from each topleaf root, emits
     *      inside-topleaf internal-node topology into SoA suns_backup,
     *      center, len, father.
     *   5. Writeback inside-topleaf range AoS u.suns/center/len.  Phase 6.6
     *      retired force_update_node_recursive; this writeback now exists so
     *      that any non-GPU CPU consumer of AoS u.suns sees complete topology
     *      between force_treebuild_single and the final
     *      gpu_topology_writeback_d_to_aos in force_treebuild.
     *
     * On overflow (rc=1), return -1 so force_treebuild's outer loop grows
     * TreeAllocFactor and retries -- same contract as the CPU path. */
    {
        force_insert_pseudo_particles();

        /* Phase 6.8a: the old mark_all_dirty + acquire pair triggered seed_full_
         * to copy AoS topnode center/len into SoA before BFS.  That seeding now
         * happens inside gpu_nextnode_backup_suns below (single GPU kernel reads
         * UVM AoS, writes SoA suns_backup + center + len for [0..numnodes)). */
        if(gpu_peano_walk_acquire() != 0) {return -1;}
        /* Snapshot topnode u.suns -> SoA suns_backup.  At this point u.suns
         * for intermediate topnodes is populated by force_create_empty_nodes;
         * force_insert_pseudo_particles set u.suns[0] for foreign topleafs;
         * local topleaf u.suns are uninitialized (BFS will overwrite their
         * suns_backup entries with the local particle subtree topology). */
        gpu_nextnode_backup_suns(numnodes);

        if(gpu_topology_build_data_path(npart, mp) != 0) {return -1;}
        int new_numnodes = numnodes;
        int rc = gpu_topology_emit_bfs(numnodes, &new_numnodes);
        if(rc == 1) {
            /* MaxNodes overflow -- bail out, force_treebuild retries with
             * larger TreeAllocFactor (line 149-151 of this file). */
            return -1;
        }
        if(rc != 0) {
            printf("force_treebuild_single: gpu_topology_emit_bfs failed rc=%d\n", rc);
            return -1;
        }
        int topnode_end = numnodes;
        numnodes = new_numnodes;

        /* Writeback GPU-built suns / center / len to AoS for the FULL range
         * (0..numnodes), not just the new BFS nodes (topnode_end..numnodes).
         * The topleaf nodes (0..topnode_end) have their soa->suns_backup entries
         * updated by BFS to hold the local-particle subtree root indices.
         * Without flushing these back to AoS, force_update_node_recursive reads
         * the stale (uninitialized) topleaf suns, never reaches local particles,
         * and Father[i] is never set -- causing an infinite loop in
         * setup_smoothinglengths which walks Nodes[Father[i]].u.d.father. */
        if(gpu_topology_writeback_to_aos(0, numnodes) != 0) {return -1;}
    }

    /* now compute the multipole moments recursively */
    last = -1;
    /* Phase 6.6: force_update_node_recursive retired on GPU build.  The GPU
     * finalize stage in force_treebuild (gpu_topology_finalize_father,
     * gpu_topology_finalize_sibling, gpu_moment_refresh, gpu_nextnode_thread)
     * now produces all of FUNR's outputs (sibling, father, Father[], moments,
     * nextnode).  The Nextnode[last]=-1 tail fixup is redundant: sibling
     * for the root is -1, which propagates through the DFS chain in
     * gpu_nextnode_thread to give the last DFS particle Nextnode[] = -1
     * automatically.  The second gpu_nextnode_backup_suns is removed too:
     * snapshot #1 (in the GPU build block above) plus emit_bfs's direct
     * SoA writes give complete suns_backup coverage, and nothing clobbers
     * AoS u.suns until gpu_topology_writeback_d_to_aos at the very end of
     * force_treebuild (by which point all SoA readers are done). */

    return numnodes;
}



/*! This function recursively creates a set of empty tree nodes which
 *  corresponds to the top-level tree for the domain grid. This is done to
 *  ensure that this top-level tree is always "complete" so that we can easily
 *  associate the pseudo-particles of other CPUs with tree-nodes at a given
 *  level in the tree, even when the particle population is so sparse that
 *  some of these nodes are actually empty.
 */
int force_create_empty_nodes(int no, int topnode, int bits, peano1D x, peano1D y, peano1D z, int *nodecount,
                              int *nextfree)
{
    int i, j, k, n, sub, count;
    MyFloat lenhalf;

    if(TopNodes[topnode].Daughter >= 0)
    {
        for(i = 0; i < 2; i++)
            for(j = 0; j < 2; j++)
                for(k = 0; k < 2; k++)
                {
                    sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);

                    count = i + 2 * j + 4 * k;

                    Nodes[no].u.suns[count] = *nextfree;

                    /* H0 (top-leaf router geometry SSOT): map this child topnode
                     * (PH offset Daughter+sub) to the physical Nodes[] slot
                     * (*nextfree) whose exact center/len is set just below.  NOTE
                     * the topnode child offset `sub` is Peano-Hilbert while the
                     * Nodes suns index `count` is Morton i+2j+4k — they differ, so
                     * the router MUST read geometry via this map, never derive it. */
                    TopNodeNodeIndex[TopNodes[topnode].Daughter + sub] = *nextfree;

                    lenhalf = 0.25 * Nodes[no].len;
                    Nodes[*nextfree].len = 0.5 * Nodes[no].len;
                    Nodes[*nextfree].center[0] = Nodes[no].center[0] + (2 * i - 1) * lenhalf;
                    Nodes[*nextfree].center[1] = Nodes[no].center[1] + (2 * j - 1) * lenhalf;
                    Nodes[*nextfree].center[2] = Nodes[no].center[2] + (2 * k - 1) * lenhalf;

                    for(n = 0; n < 8; n++)
                        Nodes[*nextfree].u.suns[n] = -1;

                    if(TopNodes[TopNodes[topnode].Daughter + sub].Daughter == -1)
                    {
                        DomainNodeIndex[TopNodes[TopNodes[topnode].Daughter + sub].Leaf] = *nextfree;
                        /* H0 SSOT consistency: a leaf topnode's router slot must equal
                         * its DomainNodeIndex entry (both are *nextfree here). Guards
                         * against future desync of the two maps. */
                        if(TopNodeNodeIndex[TopNodes[topnode].Daughter + sub] !=
                           DomainNodeIndex[TopNodes[TopNodes[topnode].Daughter + sub].Leaf])
                        {
                            printf("force_create_empty_nodes: TopNodeNodeIndex/DomainNodeIndex mismatch "
                                   "(child topnode %d, leaf %d)\n",
                                   TopNodes[topnode].Daughter + sub,
                                   TopNodes[TopNodes[topnode].Daughter + sub].Leaf);
                            endrun(91560);
                        }
                    }

                    *nextfree = *nextfree + 1;
                    *nodecount = *nodecount + 1;

                    if((*nodecount) >= MaxNodes)
                    {
                        printf("task %d: maximum number MaxNodes=%d of tree-nodes reached."
                               "MaxTopNodes=%d NTopnodes=%d NTopleaves=%d nodecount=%d\n",
                               ThisTask, MaxNodes, MaxTopNodes, NTopnodes, NTopleaves, *nodecount);
                        printf("in create empty nodes\n");
                        if(All.TreeAllocFactor > 5.0)
                        {
                            dump_particles();
                            endrun(11);
                        }
                        return -1; /* signal to caller to retry with larger TreeAllocFactor */
                    }

                    if(force_create_empty_nodes(*nextfree - 1, TopNodes[topnode].Daughter + sub,
                                             bits + 1, 2 * x + i, 2 * y + j, 2 * z + k, nodecount, nextfree) < 0)
                        return -1;
                }
    }
    return 0;
}



/*! this function inserts pseudo-particles which will represent the mass
 *  distribution of the other CPUs. Initially, the mass of the
 *  pseudo-particles is set to zero, and their coordinate is set to the
 *  center of the domain-cell they correspond to. These quantities will be
 *  updated later on.
 */
void force_insert_pseudo_particles(void)
{
    int i, index;
    
    for(i = 0; i < NTopleaves; i++)
    {
        index = DomainNodeIndex[i];
        
        if(DomainTask[i] != ThisTask)
            Nodes[index].u.suns[0] = All.MaxPart + MaxNodes + MaxForeignNodes + i;    /* Phase 9: pseudo-particles live above the foreign-node range */
    }
}






/*! Pseudo-particle exchange wire format.  Lifted to file scope (Phase 10.3) so
 *  the issue/complete halves of force_exchange_pseudodata can share the type
 *  across the LET overlap window. */
struct DomainNODE
    {
        Vec3<MyFloat> s;
        Vec3<MyFloat> vs;
        MyFloat mass;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        MyFloat gasmass;
#endif
        MyFloat hmax;
        MyFloat vmax;
        MyFloat divVmax;
        long N_part;
        MyFloat maxsoft;
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        MyFloat cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
        MyFloat stellar_lum[N_RT_FREQ_BINS];
#ifdef CHIMES_STELLAR_FLUXES
        double chimes_stellar_lum_G0[CHIMES_LOCAL_UV_NBINS];
        double chimes_stellar_lum_ion[CHIMES_LOCAL_UV_NBINS];
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        Vec3<MyFloat> rt_source_lum_s;
        Vec3<MyFloat> rt_source_lum_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
        MyFloat sink_lum; Vec3<MyFloat> sink_lum_grad;
#endif
#ifdef SINK_CALC_DISTANCES
        MyFloat sink_mass;
        Vec3<MyFloat> sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        int N_SINK;
        Vec3<MyFloat> sink_vel;
#ifdef SPECIAL_POINT_MOTION
        Vec3<MyFloat> sink_acc;
#endif
#ifdef  SINGLE_STAR_FB_TIMESTEPLIMIT
        MyFloat MaxFeedbackVel;
#endif
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        SymmetricTensor2<MyFloat> tidal_tensorps_prevstep;
#endif
#ifdef DM_SCALARFIELD_SCREENING
        Vec3<MyFloat> s_dm;
        Vec3<MyFloat> vs_dm;
        MyFloat mass_dm;
#endif
        unsigned int bitflags;
#ifdef PAD_STRUCTURES
        int pad[3];
#endif
    };

/*! Phase 10.3: state shared between force_exchange_pseudodata_issue() and
 *  ..._complete() for the (B) non-blocking-overlap pattern.  let_run_exchange
 *  runs concurrently with the pseudodata Iallgathervs in the GPU build path. */
static struct DomainNODE *DomainMoment_pending = NULL;
static MPI_Request *pseudo_requests_pending = NULL;
static int *pseudo_recvcounts_pending = NULL;
static int *pseudo_recvoffset_pending = NULL;
static int  pseudo_n_requests_pending = 0;

void force_exchange_pseudodata_issue(void)
{
    int i, no, m;
    /* Re-entrant issue() (a prior issue had no matching complete()): soft
     * bad-stop + return BEFORE allocating/posting a second Iallgatherv, so the
     * already-pending exchange is left intact for its complete(). No poll here
     * (a nonblocking collective may be outstanding); drains at a later poll. */
    if(DomainMoment_pending != NULL) {endrun(90000075); return;}

    DomainMoment_pending = (struct DomainNODE *) mymalloc("DomainMoment", NTopleaves * sizeof(struct DomainNODE));
    struct DomainNODE *DomainMoment = DomainMoment_pending;

    for(m = 0; m < MULTIPLEDOMAINS; m++)
        for(i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
            i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++)
        {
            no = DomainNodeIndex[i];
            
            /* read out the multipole moments from the local base cells */
            DomainMoment[i].s = Nodes[no].u.d.s;
            DomainMoment[i].vs = Extnodes[no].vs;
            DomainMoment[i].mass = Nodes[no].u.d.mass;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            DomainMoment[i].gasmass = Nodes[no].gasmass;
#endif
            DomainMoment[i].hmax = Extnodes[no].hmax;
            DomainMoment[i].vmax = Extnodes[no].vmax;
            DomainMoment[i].divVmax = Extnodes[no].divVmax;
            DomainMoment[i].bitflags = Nodes[no].u.d.bitflags;
            DomainMoment[i].N_part = Nodes[no].N_part;
            DomainMoment[i].maxsoft = Nodes[no].maxsoft;
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            DomainMoment[i].cr_injection = Nodes[no].cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
            int k; for(k=0;k<N_RT_FREQ_BINS;k++) {DomainMoment[i].stellar_lum[k] = Nodes[no].stellar_lum[k];}
#ifdef CHIMES_STELLAR_FLUXES
            for (k = 0; k < CHIMES_LOCAL_UV_NBINS; k++)
            {
                DomainMoment[i].chimes_stellar_lum_G0[k] = Nodes[no].chimes_stellar_lum_G0[k];
                DomainMoment[i].chimes_stellar_lum_ion[k] = Nodes[no].chimes_stellar_lum_ion[k];
            }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            DomainMoment[i].rt_source_lum_s = Nodes[no].rt_source_lum_s;
            DomainMoment[i].rt_source_lum_vs = Extnodes[no].rt_source_lum_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
            DomainMoment[i].sink_lum = Nodes[no].sink_lum;
            DomainMoment[i].sink_lum_grad = Nodes[no].sink_lum_grad;
#endif
#ifdef SINK_CALC_DISTANCES
            DomainMoment[i].sink_mass = Nodes[no].sink_mass;
            DomainMoment[i].sink_pos = Nodes[no].sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
            DomainMoment[i].sink_vel = Nodes[no].sink_vel;
            DomainMoment[i].N_SINK = Nodes[no].N_SINK;
#ifdef SPECIAL_POINT_MOTION
            DomainMoment[i].sink_acc = Nodes[no].sink_acc;
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
            DomainMoment[i].MaxFeedbackVel = Nodes[no].MaxFeedbackVel;
#endif
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            DomainMoment[i].tidal_tensorps_prevstep = Nodes[no].tidal_tensorps_prevstep;
#endif
#ifdef DM_SCALARFIELD_SCREENING
            DomainMoment[i].s_dm = Nodes[no].s_dm;
            DomainMoment[i].mass_dm = Nodes[no].mass_dm;
            DomainMoment[i].vs_dm = Extnodes[no].vs_dm;
#endif
        }

    /* Phase 10.3: post one MPI_Iallgatherv per MULTIPLEDOMAINS slice; the requests
     * are stored in static pseudo_requests_pending and waited on in _complete().
     * Per-slice recvcounts/recvoffset arrays must remain valid until Wait, so
     * we allocate one set per slice and free them all in _complete(). */
    pseudo_n_requests_pending = MULTIPLEDOMAINS;
    pseudo_requests_pending = (MPI_Request *) mymalloc("pseudo_requests",
                                  MULTIPLEDOMAINS * sizeof(MPI_Request));
    pseudo_recvcounts_pending = (int *) mymalloc("pseudo_recvcounts",
                                  MULTIPLEDOMAINS * NTask * sizeof(int));
    pseudo_recvoffset_pending = (int *) mymalloc("pseudo_recvoffset",
                                  MULTIPLEDOMAINS * NTask * sizeof(int));
    for(m = 0; m < MULTIPLEDOMAINS; m++)
    {
        int *rc = pseudo_recvcounts_pending + m * NTask;
        int *ro = pseudo_recvoffset_pending + m * NTask;
        for(int recvTask = 0; recvTask < NTask; recvTask++)
        {
            rc[recvTask] =
                (DomainEndList[recvTask * MULTIPLEDOMAINS + m] -
                 DomainStartList[recvTask * MULTIPLEDOMAINS + m] + 1)
                * sizeof(struct DomainNODE);
            ro[recvTask] = DomainStartList[recvTask * MULTIPLEDOMAINS + m]
                           * sizeof(struct DomainNODE);
        }
        MPI_Iallgatherv(MPI_IN_PLACE, rc[ThisTask], MPI_BYTE,
                        &DomainMoment[0], rc, ro, MPI_BYTE, MPI_COMM_WORLD,
                        &pseudo_requests_pending[m]);
    }
}

/*! Waits on the Iallgathervs posted by force_exchange_pseudodata_issue() and
 *  unpacks the received topleaf moments into the AoS Nodes_base / Extnodes_base.
 *  GPU build path calls let_run_exchange() in between to overlap MPI; CPU and
 *  refresh paths call the sync wrapper force_exchange_pseudodata() below. */
int force_exchange_pseudodata_complete(void)
{
    /* Unmatched complete (pending==NULL = complete without a matching issue, or a
     * double-complete): symmetric control-flow invariant. Soft bad-stop + status-return
     * (1) so the caller skips the foreign-moment scatter/finalize/resum -- which would run
     * on un-exchanged moments and could itself emergency-hold in let_finalize -- and
     * drains at the next poll. */
    if(DomainMoment_pending == NULL) {endrun(90000076); return 1;}
    struct DomainNODE *DomainMoment = DomainMoment_pending;

    MPI_Waitall(pseudo_n_requests_pending, pseudo_requests_pending, MPI_STATUSES_IGNORE);

    /* Free request/count buffers (LIFO order: ro, rc, requests). */
    myfree(pseudo_recvoffset_pending);
    myfree(pseudo_recvcounts_pending);
    myfree(pseudo_requests_pending);
    pseudo_requests_pending = NULL;
    pseudo_recvcounts_pending = NULL;
    pseudo_recvoffset_pending = NULL;
    pseudo_n_requests_pending = 0;

    int i, no, m, ta;
    for(ta = 0; ta < NTask; ta++)
        if(ta != ThisTask)
            for(m = 0; m < MULTIPLEDOMAINS; m++)
                for(i = DomainStartList[ta * MULTIPLEDOMAINS + m]; i <= DomainEndList[ta * MULTIPLEDOMAINS + m]; i++)
                {
                    no = DomainNodeIndex[i];

                    Nodes[no].u.d.s = DomainMoment[i].s;
                    Extnodes[no].vs = DomainMoment[i].vs;
                    Nodes[no].u.d.mass = DomainMoment[i].mass;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                    Nodes[no].gasmass = DomainMoment[i].gasmass;
#endif
                    Extnodes[no].hmax = DomainMoment[i].hmax;
                    Extnodes[no].vmax = DomainMoment[i].vmax;
                    Extnodes[no].divVmax = DomainMoment[i].divVmax;
                    Nodes[no].N_part = DomainMoment[i].N_part;
                    Nodes[no].u.d.bitflags = (Nodes[no].u.d.bitflags & (~((1 << BITFLAG_MULTIPLEPARTICLES)))) | (DomainMoment[i].bitflags & ((1 << BITFLAG_MULTIPLEPARTICLES)));
                    Nodes[no].maxsoft = DomainMoment[i].maxsoft;
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                    Nodes[no].cr_injection = DomainMoment[i].cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
                    int k; for(k=0;k<N_RT_FREQ_BINS;k++) {Nodes[no].stellar_lum[k] = DomainMoment[i].stellar_lum[k];}
#ifdef CHIMES_STELLAR_FLUXES
                    for (k = 0; k < CHIMES_LOCAL_UV_NBINS; k++)
                    {
                        Nodes[no].chimes_stellar_lum_G0[k] = DomainMoment[i].chimes_stellar_lum_G0[k];
                        Nodes[no].chimes_stellar_lum_ion[k] = DomainMoment[i].chimes_stellar_lum_ion[k];
                    }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                    Nodes[no].rt_source_lum_s = DomainMoment[i].rt_source_lum_s;
                    Extnodes[no].rt_source_lum_vs = DomainMoment[i].rt_source_lum_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
                    Nodes[no].sink_lum = DomainMoment[i].sink_lum;
                    Nodes[no].sink_lum_grad = DomainMoment[i].sink_lum_grad;
#endif
#ifdef SINK_CALC_DISTANCES
                    Nodes[no].sink_mass = DomainMoment[i].sink_mass;
                    Nodes[no].sink_pos = DomainMoment[i].sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
                    Nodes[no].sink_vel = DomainMoment[i].sink_vel;
                    Nodes[no].N_SINK = DomainMoment[i].N_SINK;
#ifdef SPECIAL_POINT_MOTION
                    Nodes[no].sink_acc = DomainMoment[i].sink_acc;
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
                    Nodes[no].MaxFeedbackVel = DomainMoment[i].MaxFeedbackVel;
#endif
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
                    Nodes[no].tidal_tensorps_prevstep = DomainMoment[i].tidal_tensorps_prevstep;
#endif
#ifdef DM_SCALARFIELD_SCREENING
                    Nodes[no].s_dm = DomainMoment[i].s_dm;
                    Nodes[no].mass_dm = DomainMoment[i].mass_dm;
                    Extnodes[no].vs_dm = DomainMoment[i].vs_dm;
#endif
                }

    myfree(DomainMoment);
    DomainMoment_pending = NULL;
    return 0;
}

/*! Synchronous wrapper preserving the pre-Phase-10.3 API for the CPU and
 *  refresh code paths (which do not have a LET round to overlap with).
 *  Returns the complete() status (nonzero = unmatched; caller skips dependent
 *  pseudo-update work and drains at its poll). */
int force_exchange_pseudodata(void)
{
    force_exchange_pseudodata_issue();
    return force_exchange_pseudodata_complete();
}



/*! This function updates the top-level tree after the multipole moments of
 *  the pseudo-particles have been updated.
 */
void force_treeupdate_pseudos(int no)
{
    int j, p, count_particles, multiple_flag;
    MyFloat hmax, vmax;
    MyFloat hmax_per_type[6];
    MyFloat divVmax;
    Vec3<MyFloat> s, vs; MyFloat mass;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    MyFloat gasmass = 0;
#endif

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    double cr_injection = 0;
#endif
#ifdef RT_USE_GRAVTREE
    MyFloat stellar_lum[N_RT_FREQ_BINS]={0};
#ifdef CHIMES_STELLAR_FLUXES
    double chimes_stellar_lum_G0[CHIMES_LOCAL_UV_NBINS]={0}, chimes_stellar_lum_ion[CHIMES_LOCAL_UV_NBINS]={0};
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyFloat> rt_source_lum_s, rt_source_lum_vs;
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Vec3<MyFloat> s_dm, vs_dm; MyFloat mass_dm;
#endif
    
    MyFloat maxsoft;
    
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    rt_source_lum_s = rt_source_lum_vs = {};
#endif
#ifdef SINK_PHOTONMOMENTUM
    MyFloat sink_lum = 0; Vec3<MyFloat> sink_lum_grad = {};
#endif
#ifdef SINK_CALC_DISTANCES
    MyFloat sink_mass=0;
    Vec3<MyFloat> sink_pos_times_mass = {};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Vec3<MyFloat> sink_mom = {};
    int N_SINK = 0;
#ifdef SPECIAL_POINT_MOTION
    Vec3<MyFloat> sink_force = {};
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    MyFloat max_feedback_vel=0;
#endif
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    SymmetricTensor2<MyFloat> tidal_tensorps_prevstep = {};
#endif
#ifdef DM_SCALARFIELD_SCREENING
    mass_dm = 0;
    s_dm = vs_dm = {};
#endif
    mass = 0;
    s = vs = {};
    hmax = 0;
    for(int t = 0; t < 6; t++) hmax_per_type[t] = 0;
    vmax = 0;
    divVmax = 0;
    count_particles = 0;
    maxsoft = 0;

    p = Nodes[no].u.d.nextnode;
    
    for(j = 0; j < 8; j++)    /* since we are dealing with top-level nodes, we now that there are 8 consecutive daughter nodes */
    {
        if(p >= All.MaxPart && p < All.MaxPart + MaxNodes)    /* internal node */
        {
            if(Nodes[p].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL)) {force_treeupdate_pseudos(p);}
            
            mass += (Nodes[p].u.d.mass);
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            gasmass += Nodes[p].gasmass;
#endif
            s += Nodes[p].u.d.mass * Nodes[p].u.d.s;
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            cr_injection += Nodes[p].cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
            int k; for(k=0;k<N_RT_FREQ_BINS;k++) {stellar_lum[k] += (Nodes[p].stellar_lum[k]);}
#ifdef CHIMES_STELLAR_FLUXES
            for (k = 0; k < CHIMES_LOCAL_UV_NBINS; k++)
            {
                chimes_stellar_lum_G0[k] += Nodes[p].chimes_stellar_lum_G0[k];
                chimes_stellar_lum_ion[k] += Nodes[p].chimes_stellar_lum_ion[k];
            }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            double l_tot=0; for(k=0;k<N_RT_FREQ_BINS;k++) {l_tot += (Nodes[p].stellar_lum[k]);}
            rt_source_lum_s += l_tot * Nodes[p].rt_source_lum_s;
            rt_source_lum_vs += l_tot * Extnodes[p].rt_source_lum_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
            sink_lum += Nodes[p].sink_lum;
            sink_lum_grad += Nodes[p].sink_lum * Nodes[p].sink_lum_grad;
#endif
#ifdef SINK_CALC_DISTANCES
            sink_mass += Nodes[p].sink_mass;
            sink_pos_times_mass += Nodes[p].sink_mass * Nodes[p].sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
            N_SINK += Nodes[p].N_SINK;
            sink_mom += Nodes[p].sink_mass * Nodes[p].sink_vel;
#ifdef SPECIAL_POINT_MOTION
            sink_force += Nodes[p].sink_mass * Nodes[p].sink_acc;
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
            if(Nodes[p].sink_mass > 0) {max_feedback_vel = DMAX(max_feedback_vel, Nodes[p].MaxFeedbackVel);}
#endif
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            {int k; for(k=0;k<6;k++) {tidal_tensorps_prevstep.data[k] += Nodes[p].u.d.mass * Nodes[p].tidal_tensorps_prevstep.data[k];}}
#endif
#ifdef DM_SCALARFIELD_SCREENING
            mass_dm += (Nodes[p].mass_dm);
            s_dm += Nodes[p].mass_dm * Nodes[p].s_dm;
            vs_dm += Nodes[p].mass_dm * Extnodes[p].vs_dm;
#endif
            vs += Nodes[p].u.d.mass * Extnodes[p].vs;
            
            if(Extnodes[p].hmax > hmax) {hmax = Extnodes[p].hmax;}
            for(int t = 0; t < 6; t++) {
                if(Extnodes[p].hmax_per_type[t] > hmax_per_type[t]) {hmax_per_type[t] = Extnodes[p].hmax_per_type[t];}
            }
            if(Extnodes[p].vmax > vmax) {vmax = Extnodes[p].vmax;}
            if(Extnodes[p].divVmax > divVmax) {divVmax = Extnodes[p].divVmax;}
            if(Nodes[p].u.d.mass > 0) {count_particles += Nodes[p].N_part;} // saved, so directly add
            if(Nodes[p].maxsoft > maxsoft) {maxsoft = Nodes[p].maxsoft;}
        }
        else
        {   /* invalid node type: soft bad-stop + break before the corrupt
             * Nodes[p].sibling deref below; drains at a gravtree poll */
            endrun(90000077);
            break;
        }

        p = Nodes[p].u.d.sibling;
    }
    
    if(mass)
    {
        s /= mass;
        vs /= mass;
    }
    else
    {
        s = Nodes[no].center;
        vs = {};
    }
    
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    double l_tot=0; int kfreq; for(kfreq=0;kfreq<N_RT_FREQ_BINS;kfreq++) {l_tot += stellar_lum[kfreq];}
    if(l_tot)
    {
        rt_source_lum_s /= l_tot;
        rt_source_lum_vs /= l_tot;
    }
    else
    {
        rt_source_lum_s = Nodes[no].center;
        rt_source_lum_vs = {};
    }
#endif
#ifdef SINK_PHOTONMOMENTUM
    if(sink_lum)
    {
        sink_lum_grad /= sink_lum;
    }
    else
    {
        sink_lum_grad = {0, 0, 1};
    }
#endif
#ifdef DM_SCALARFIELD_SCREENING
    if(mass_dm)
    {
        s_dm /= mass_dm;
        vs_dm /= mass_dm;
    }
    else
    {
        s_dm = Nodes[no].center;
        vs_dm = {};
    }
#endif


    Nodes[no].u.d.s = s;
    Extnodes[no].vs = vs;
    Nodes[no].u.d.mass = mass;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    Nodes[no].gasmass = gasmass;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    Nodes[no].cr_injection = cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
    int k; for(k=0;k<N_RT_FREQ_BINS;k++) {Nodes[no].stellar_lum[k] = stellar_lum[k];}
#ifdef CHIMES_STELLAR_FLUXES
    for (k = 0; k < CHIMES_LOCAL_UV_NBINS; k++)
    {
        Nodes[no].chimes_stellar_lum_G0[k] = chimes_stellar_lum_G0[k];
        Nodes[no].chimes_stellar_lum_ion[k] = chimes_stellar_lum_ion[k];
    }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Nodes[no].rt_source_lum_s = rt_source_lum_s;
    Extnodes[no].rt_source_lum_vs = rt_source_lum_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
    Nodes[no].sink_lum = sink_lum;
    Nodes[no].sink_lum_grad = sink_lum_grad;
#endif
#ifdef SINK_CALC_DISTANCES
    Nodes[no].sink_mass = sink_mass;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Nodes[no].N_SINK = N_SINK;
#endif
    if(sink_mass > 0)
    {
        Nodes[no].sink_pos = sink_pos_times_mass / sink_mass;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        Nodes[no].sink_vel = sink_mom / sink_mass;
#if defined(SPECIAL_POINT_MOTION)
        Nodes[no].sink_acc = sink_force / sink_mass;
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
        Nodes[no].MaxFeedbackVel = max_feedback_vel;
#endif
#endif
    }
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    {MyFloat inv_mass = 1.0/(mass+MIN_REAL_NUMBER); int k; for(k=0;k<6;k++) {Nodes[no].tidal_tensorps_prevstep.data[k] = tidal_tensorps_prevstep.data[k] * inv_mass;}}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Nodes[no].s_dm = s_dm;
    Nodes[no].mass_dm = mass_dm;
    Extnodes[no].vs_dm = vs_dm;
#endif
    
    Extnodes[no].hmax = hmax;
    for(int t = 0; t < 6; t++) Extnodes[no].hmax_per_type[t] = hmax_per_type[t];
    Extnodes[no].vmax = vmax;
    Extnodes[no].divVmax = divVmax;
    Extnodes[no].Flag = GlobFlag;
    Nodes[no].N_part = count_particles; // record
    if(count_particles > 1) {multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);} else {multiple_flag = 0;}
    Nodes[no].u.d.bitflags &= (~((1 << BITFLAG_MULTIPLEPARTICLES)));    /* this clears the bits */
    Nodes[no].u.d.bitflags |= multiple_flag;
    Nodes[no].maxsoft = maxsoft;
    /* Phase 7.a: no GPU SoA hook here — force_treeupdate_pseudos is the CPU
     * pseudo-path; on GPU builds it is replaced by gpu_topnode_moment_resum
     * (gpu_pseudo_update.cc) which writes the SoA directly.  This function
     * compiles on both, but the GPU build never calls it. */
}



/*! This function flags nodes in the top-level tree that are dependent on
 *  local particle data.
 */
void force_flag_localnodes(void)
{
    int no, i, m;
    
    /* mark all top-level nodes */
    
    for(i = 0; i < NTopleaves; i++)
    {
        no = DomainNodeIndex[i];
        
        while(no >= 0)
        {
            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {break;}
            
            Nodes[no].u.d.bitflags |= (1 << BITFLAG_TOPLEVEL);
            
            no = Nodes[no].u.d.father;
        }
        
        /* mark also internal top level nodes */
        
        no = DomainNodeIndex[i];
        no = Nodes[no].u.d.father;
        
        while(no >= 0)
        {
            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL)) {break;}
            
            Nodes[no].u.d.bitflags |= (1 << BITFLAG_INTERNAL_TOPLEVEL);
            
            no = Nodes[no].u.d.father;
        }
    }
    
    /* mark top-level nodes that contain local particles */
    
    for(m = 0; m < MULTIPLEDOMAINS; m++)
        for(i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
            i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++)
        {
            no = DomainNodeIndex[i];
            
            if(DomainTask[i] != ThisTask) {endrun(90000078); continue;} /* soft bad-stop + skip the foreign entry instead of mismarking its DEPENDS bitflags; drains at a gravtree poll */
            
            while(no >= 0)
            {
                if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT)) {break;}
                
                Nodes[no].u.d.bitflags |= (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT);
                
                no = Nodes[no].u.d.father;
            }
        }
}


/*! When a new additional resolution element is created, we can put it into the
 *  tree at the position of the spawning element. This is possible
 *  because the Nextnode[] array essentially describes the full tree walk as a
 *  link list. Multipole moments of tree nodes need not be changed (the new
 *  particle inherits the parent's position so mass+CoM are preserved at the
 *  insertion site; the 9.6 ForceAddElementToTree_CallsSinceBuild guardrail
 *  bounds drift to ancestor nodes between full rebuilds).
 *
 *  Phase 10.2 (Crossing 4 retirement, scope-α): Father/Nextnode/Extnodes are
 *  UVM (SharedSpace) since Phase 6.6/6.8d/6.8e, so the CPU mutations below
 *  are GPU-visible without a copy.  We additionally mirror the parent node's
 *  hmax/vmax/len into the SoA walk view, because the next gpu_force_drift_nodes
 *  early-outs when the parent's Ti_current already matches All.Ti_Current
 *  (insertions between drifts at the same Ti_current would otherwise leave
 *  the SoA stale until the next full rebuild).
 */
void force_add_element_to_tree(int iparent, int ichild)
{
    int father = Father[iparent];
    int no = Nextnode[iparent];
    Nextnode[iparent] = ichild; // insert new particle into linked list
    Nextnode[ichild] = no; // order correctly
    Father[ichild] = father; // set parent node to be the same
    // update parent node properties [maximum softening, speed] for opening criteria
    MyFloat new_hmax = DMAX(Extnodes[father].hmax, (MyFloat) moment_gas_hmax_from_kernelradius(P[iparent].KernelRadius, All.MaxKernelRadius));
    Extnodes[father].hmax = new_hmax;
    /* Mode B per-type incremental update — only the band corresponding to
     * iparent's type. Other bands unchanged (correct: this insertion adds
     * one particle, only its type's band can grow). Conservative across every
     * leaf-policy-selectable source via the shared helper. */
    {
        int ptype = (int)P[iparent].Type;
        double htmp = force_hmax_per_type_particle_radius(iparent);
        if(htmp > Extnodes[father].hmax_per_type[ptype]) {
            Extnodes[father].hmax_per_type[ptype] = (MyFloat)htmp;
        }
    }
    double new_vmax = moment_vmax_running_max(Extnodes[father].vmax, P[ichild].Vel[0], P[ichild].Vel[1], P[ichild].Vel[2]);
    Extnodes[father].vmax = (MyFloat) new_vmax;

    /* Phase 10.2 (α): keep SoA walk-mirror coherent with the AoS Extnodes
     * change above.  hmax and vmax are read by the walk's opening criteria
     * (and vmax drives bbox expansion in subsequent drifts via Nodes[].len).
     * Indexing matches the local-or-foreign convention from gpu_force_drift_nodes:
     * SoA slot k = father - All.MaxPart for both local nodes (k < MaxNodes) and
     * foreign nodes (MaxNodes <= k < MaxNodes + MaxForeignNodes). */
    {
        struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
        int k_soa = father - All.MaxPart;
        if(soa && k_soa >= 0 && k_soa < MaxNodes + MaxForeignNodes) {
            if(soa->hmax) {soa->hmax[k_soa] = (MyGravFloat) new_hmax;}
            if(soa->vmax) {soa->vmax[k_soa] = (MyGravFloat) new_vmax;}
        }
    }

    /* Phase 9.6 diagnostic: each insertion stales the LET / pseudo-particle
     * moments shipped on the last full build.  Mass+CoM remain conserved at
     * the insertion site, but ancestor topnodes (and any rank's foreign view
     * of them) carry the pre-insertion moments until the next rebuild. */
    ForceAddElementToTree_CallsSinceBuild++;
}



/*! This routine computes the gravitational force for a given local
 *  particle, or for a particle in the communication buffer. Depending on
 *  the value of TypeOfOpeningCriterion, either the geometrical BH
 *  cell-opening criterion, or the `relative' opening criterion is used.
 */
/*! The modern version of this routine handles both the PM-grid and non-PM
 *  cases, unlike the previous version (which used two, redundant, algorithms)
 */
/*! In the TreePM algorithm, the tree is walked only locally around the
 *  target coordinate.  Tree nodes that fall outside a box of half
 *  side-length Rcut= PM_RCUT*PM_ASMTH*MeshSize can be discarded. The short-range
 *  potential is modified by a complementary error function, multiplied
 *  with the Newtonian form. The resulting short-range suppression compared
 *  to the Newtonian force is tabulated, because looking up from this table
 *  is faster than recomputing the corresponding factor, despite the
 *  memory-access panelty (which reduces cache performance) incurred by the
 *  table.
 */
int force_treeevaluate(int target, int *exportflag, int *exportnodecount, int *exportindex)
{
    struct NODE *nop = 0;
    int no, ptype, ninteractions=0, nexp, task, maxPart = All.MaxPart;
    long bunchSize = All.BunchSize; int maxNodes = MaxNodes; int maxForeignNodes = MaxForeignNodes; integertime ti_Current = All.Ti_Current;    /* Phase 9: maxForeignNodes shifts pseudo-particle range above the foreign-node range */
    double soft, r2, mass, r, fac_accel, h=0, h_p=0, xtmp, aold; xtmp=0; soft=0;
    Vec3<double> pos, dr; Vec3<MyDouble> acc = {};
    double pmass;
    double zeta=0, zeta_sec=0; int ptype_sec=-1;
#ifdef RT_USE_TREECOL_FOR_NH
    double angular_bin_size = 4*M_PI / RT_USE_TREECOL_FOR_NH, treecol_angular_bins[RT_USE_TREECOL_FOR_NH] = {0};
#endif
#if defined(COMPUTE_JERK_IN_GRAVTREE) || defined(SINK_DYNFRICTION_FROMTREE)
    Vec3<double> dv;
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    double gasmass;
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
    Vec3<double> jerk = {};
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(COMPUTE_JERK_IN_GRAVTREE) || defined(SINK_DYNFRICTION_FROMTREE)
    Vec3<double> vel;
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
    double r_source, r_target, center[3]={0};
#ifdef BOX_PERIODIC
    center[0] = 0.5 * boxSize_X; center[1] = 0.5 * boxSize_Y; center[2] = 0.5 * boxSize_Z;
#endif
#endif
    int tabindex = 0;   /* unconditional; computed + consumed only under PMGRID */
#ifdef PMGRID
    double rcut, asmth, asmthfac, rcut2; rcut = All.Rcut[0]; asmth = All.Asmth[0];
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
    MyFloat tree_mass = 0;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    int i1, i2; double fac2_tidal, fac_tidal; SymmetricTensor2<MyDouble> tidal_tensorps;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    double cr_injection = 0;
#endif
#ifdef RT_USE_GRAVTREE
    double mass_stellarlum[N_RT_FREQ_BINS]; int k_freq; for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++) {mass_stellarlum[k_freq]=0;}
#ifdef CHIMES_STELLAR_FLUXES
    double chimes_mass_stellarlum_G0[CHIMES_LOCAL_UV_NBINS]={0}, chimes_mass_stellarlum_ion[CHIMES_LOCAL_UV_NBINS]={0}, chimes_flux_G0[CHIMES_LOCAL_UV_NBINS]={0}, chimes_flux_ion[CHIMES_LOCAL_UV_NBINS]={0};
#endif
    Vec3<double> d_stellarlum = {}; int valid_gas_particle_for_rt = 0;
#ifdef RT_OTVET
    SymmetricTensor2<double> RT_ET[N_RT_FREQ_BINS]={};
#endif
#endif
#ifdef SINK_PHOTONMOMENTUM
    double mass_sinklumwt_forradfb=0; // convert bh luminosity to our tree units
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    double incident_flux_uv=0, incident_flux_euv=0;
#endif
#ifdef SINK_COMPTON_HEATING
    double incident_flux_agn=0;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    double SubGrid_CosmicRayEnergyDensity = 0;
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
    double Rad_E_gamma[N_RT_FREQ_BINS]={0};
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
    Vec3<double> Rad_Flux[N_RT_FREQ_BINS]; {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {Rad_Flux[kf] = {};}}
#endif
#ifdef SINK_CALC_DISTANCES
    grav_sink_prox_accum_t sink_prox; grav_sink_prox_accum_init(sink_prox); /* nearest-sink + single-star timestep/binary accumulators (gravtree_force_kernel.h) */
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Vec3<double> d_dm = {}; double mass_dm = 0;
#endif
#if defined(SINK_DYNFRICTION_FROMTREE)
    double sink_mass = 0, m_j_eff_for_df = 0;
#endif
    double fac_pot = 0;   /* unconditional; a dead 0 when !EVALPOTENTIAL (consumed only under that gate) */
#ifdef EVALPOTENTIAL
    MyDouble pot; pot = 0;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    tidal_tensorps = {};
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    double tidal_zeta=0; SymmetricTensor2<MyFloat> i_zeta_tidal_tensorps_prevstep, j_zeta_tidal_tensorps_prevstep;
    i_zeta_tidal_tensorps_prevstep=P[target].tidal_tensorps_prevstep;
#endif
#endif
    
    pos = P[target].Pos;
    ptype = P[target].Type;
    soft = ForceSoftening_KernelRadius(target);
    aold = All.ErrTolForceAcc * P[target].OldAcc;
    pmass = P[target].Mass;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(COMPUTE_JERK_IN_GRAVTREE) || defined(SINK_DYNFRICTION_FROMTREE)
    vel = P[target].Vel;
#endif
#if defined(SINK_DYNFRICTION_FROMTREE)
    if(ptype == 5) {sink_mass = P[target].Sink_Mass;}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
    grav_target_select_soft_and_zeta(ptype, P[target].AGS_zeta, soft, zeta);
#endif
#if defined(PMGRID) && defined(PM_PLACEHIGHRESREGION)
    if(pmforce_is_particle_high_res(ptype, P[target].Pos)) {rcut = All.Rcut[1]; asmth = All.Asmth[1];}
#endif


    if(pmass<=0) {return 0;} /* quick check if particle has mass: if not, we won't deal with it */
    int AGS_kernel_shared_BITFLAG = ags_gravity_kernel_shared_BITFLAG(ptype); // determine allowed particle types for correction terms for adaptive gravitational softening terms
#ifdef PMGRID
    rcut2 = rcut * rcut; asmthfac = grav_pm_asmthfac(asmth);
#endif
    /* read-only PM short-range config for the shared force helpers (empty when !PMGRID;
     * built once per target after the PM_PLACEHIGHRESREGION rcut/asmth override above). */
    grav_pm_shortrange_t pm{};
#ifdef PMGRID
    pm.rcut = rcut; pm.rcut2 = rcut2; pm.asmthfac = asmthfac; pm.shortrange_tab = shortrange_table;
#ifdef EVALPOTENTIAL
    pm.shortrange_pot_tab = shortrange_table_potential;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    pm.shortrange_tidal_tab = shortrange_table_tidal;
#endif
#endif
#ifdef RT_USE_GRAVTREE
    valid_gas_particle_for_rt = grav_target_valid_gas_for_rt(ptype, soft, pmass);
#if defined(RT_LEBRON) && !defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
    double fac_stellum[N_RT_FREQ_BINS];
    if(valid_gas_particle_for_rt)
    {
        double kappa_eff[N_RT_FREQ_BINS]; int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {kappa_eff[kf] = rt_kappa(-1,kf, P, CellP);} // rt_kappa is in physical code units (needs the walk's particle pointers, so evaluated here)
        grav_target_rt_fac_stellum(soft, pmass, kappa_eff, fac_stellum);
    }
#endif
#endif
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
    double m_enc_in_rcrit = 0, r_for_total_menclosed = grav_target_menc_radius(soft); /* baseline Rcrit_min applied in the helper, otherwise we get statistics that are very noisy */
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    /* per-target CR gate + max stream time, hoisted from the accumulation (the time helper is host-only; value is interaction-independent) */
    int cr_active_gate = (All.Time > All.TimeBegin) ? 1 : 0; double cr_t_max = 0;
    if(cr_active_gate) {cr_t_max = DMIN(1., evaluate_time_since_t_initial_in_Gyr(All.TimeBegin))/UNIT_TIME_IN_GYR;}
#endif
    
    
    no = maxPart;        /* root node */

    while(no >= 0)   /* outer loop runs once: the mode-1 imported-NodeList iteration is retired */
    {
        while(no >= 0)
        {
            h=soft; h_p=-1; /* initialize h and h_p, for use below: make sure to do so at the top of each iteration */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            gasmass=0; /* reset per interaction: non-gas leaf sources carry NO gas mass. Without this, a
                        * non-gas leaf inherits the stale gasmass of an earlier source (or garbage before the
                        * first assignment) and the TREECOL column estimate fabricates contributions through
                        * dark-matter/star particles. Nodes assign unconditionally below; only gas leaves
                        * (and the sink alpha-disk reservoir, where enabled) carry gas mass. */
#endif
            
            if(no < maxPart) /* this is a particle, we will use it */
            {
                /* the index of the node is the index of the particle */
                if(P[no].Ti_current != ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_particledriftforce_)
#endif
                    {
                        if(P[no].Ti_current != ti_Current) {
                            drift_particle(no, ti_Current);
                            gizmo_mark_kernel_radius_dirty_indices(&no, 1);
                        }
                    }
                }
                dr = P[no].Pos - pos;
                GRAVITY_NEAREST_XYZ(dr[0],dr[1],dr[2],-1);
                r2 = dr.norm_sq();
                mass = P[no].Mass;
                
#ifdef GRAVITY_SPHERICAL_SYMMETRY
                r_source = grav_spherical_symmetry_r_from_center(P[no].Pos[0],P[no].Pos[1],P[no].Pos[2],center[0],center[1],center[2]);
#endif
#if defined(COMPUTE_JERK_IN_GRAVTREE) || defined(SINK_DYNFRICTION_FROMTREE)
                dv = P[no].Vel - vel;
#endif
#if defined(SINK_DYNFRICTION_FROMTREE)
                m_j_eff_for_df = mass;
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                if(P[no].Type == 0) {gasmass = P[no].Mass;}
#if defined(SINK_ALPHADISK_ACCRETION) && defined(RT_USE_TREECOL_FOR_NH)
                if(P[no].Type == 5) {gasmass = P[no].Sink_Mass_Reservoir;} // gas at the inner edge of a disk should not see a hole due to the sink
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
                j_zeta_tidal_tensorps_prevstep=P[no].tidal_tensorps_prevstep;
#endif
                
                /* only proceed if the mass is positive and there is separation! */
                if((r2 > 0) && (mass > 0))
                {
                    
#ifdef SINK_CALC_DISTANCES
                    /* nearest-sink + single-star timestep/binary tracking via the shared helper (gravtree_force_kernel.h) */
                    grav_sink_prox_target_t prox_target = {}; prox_target.ptype = ptype; prox_target.pmass = pmass; prox_target.soft = soft;
#if defined(SINGLE_STAR_TIMESTEPPING)
                    prox_target.vel = vel;
#endif
                    grav_sink_prox_leaf_src_t prox_src = {}; prox_src.src_type = P[no].Type; prox_src.src_mass = P[no].Mass; prox_src.motion.vel = P[no].Vel;
#if defined(SPECIAL_POINT_MOTION) || defined(SPECIAL_POINT_WEIGHTED_MOTION)
                    prox_src.motion.acc = P[no].Acc_Total_PrevStep;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
                    prox_src.motion.max_feedback_vel = P[no].MaxFeedbackVel;
#endif
                    grav_sink_prox_leaf_accumulate(r2, dr, prox_target, prox_src, sink_prox);
#endif // SINK_CALC_DISTANCES

#ifdef COSMIC_RAY_SUBGRID_LEBRON
                    cr_injection = cr_get_source_injection_rate(no, P, CellP);
#endif

#ifdef RT_USE_GRAVTREE
                    if(valid_gas_particle_for_rt)    /* we have a (valid) gas particle as target */
                    {
                        d_stellarlum=dr;
                        double lum[N_RT_FREQ_BINS];
#ifdef CHIMES_STELLAR_FLUXES
                        double chimes_lum_G0[CHIMES_LOCAL_UV_NBINS], chimes_lum_ion[CHIMES_LOCAL_UV_NBINS];
                        int active_check = rt_get_source_luminosity_chimes(no,1,lum, chimes_lum_G0, chimes_lum_ion, P, CellP);
#else
                        int active_check = rt_get_source_luminosity(no,1,lum, P, CellP);
#endif
                        int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {if(active_check) {mass_stellarlum[kf]=lum[kf];} else {mass_stellarlum[kf]=0;}}
#ifdef CHIMES_STELLAR_FLUXES
                        for(kf = 0; kf < CHIMES_LOCAL_UV_NBINS; kf++)
                        {
                            if(active_check) {chimes_mass_stellarlum_G0[kf] = chimes_lum_G0[kf]; chimes_mass_stellarlum_ion[kf] = chimes_lum_ion[kf];} else {chimes_mass_stellarlum_G0[kf] = 0; chimes_mass_stellarlum_ion[kf] = 0;}
                        }
#endif
#ifdef SINK_PHOTONMOMENTUM
                        mass_sinklumwt_forradfb=0;
                        if(P[no].Type == 5)
                        {
                            double bhlum_t = sink_lum_bol(P[no].Sink_Mdot, P[no].Sink_Mass, no);
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
                            mass_sinklumwt_forradfb = sink_fb_angleweight(bhlum_t, P[no].Sink_Specific_AngMom, dr[0],dr[1],dr[2]);
#else
                            mass_sinklumwt_forradfb = sink_fb_angleweight(bhlum_t, P[no].GradRho, dr[0],dr[1],dr[2]);
#endif
                        }
#endif
                    }
#endif // RT_USE_GRAVTREE
                    
#ifdef DM_SCALARFIELD_SCREENING
                    if(ptype != 0) {if(P[no].Type == 1) {d_dm = dr; mass_dm = mass;} else {d_dm = {}; mass_dm = 0;}} /* we have a dark matter particle as target */
#endif
                    
                    h_p = ForceSoftening_KernelRadius(no);
                    ptype_sec=P[no].Type; zeta_sec=0; /* set secondary softening and zeta term */
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                    if(ptype_sec==0) {zeta_sec=P[no].AGS_zeta;}
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
                    zeta_sec=P[no].AGS_zeta;
#endif
                } // closes (if((r2 > 0) && (mass > 0))) check
                
            }
            else /* we have an  internal node */
            {
                if(no >= maxPart + maxNodes + maxForeignNodes) /* pseudo particle (Phase 9: foreign-node range below pseudos) -- this will not be used for calculations below, but needs to be parsed here */
                {
                    /* LET-incompleteness DETECTOR (not an export system: the MPI round-trip is
                     * retired). Reaching a non-empty pseudo means this target's gravity is not
                     * covered by the local LET; record it so Nexport>0 and gravity_tree() can
                     * raise a graceful controlled-stop ("raise LETAllocFactor"). The
                     * DataIndexTable/DataNodeList entries are never shipped -- they only count. */
                    if(exportflag[task = DomainTask[no - (maxPart + maxNodes + maxForeignNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }
                    if(exportnodecount[task] == NODELISTLENGTH)
                    {
                        int exitFlag = 0;
#ifdef _OPENMP
#pragma omp critical(_nexportforce_)
#endif
                        {
                            if(Nexport >= bunchSize)
                            {
                                /* out of buffer space. Need to discard work for this particle and interrupt */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport++;
                            }
                        }
                        if(exitFlag) {return -1;} /* buffer has filled -- important that only this and other buffer-full conditions return the negative condition for the routine */
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }
                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                    DomainNodeIndex[no - (maxPart + maxNodes + maxForeignNodes)];
                    if(exportnodecount[task] < NODELISTLENGTH) {DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;}
                    no = Nextnode[no - maxNodes - maxForeignNodes];
                    continue;
                }
                /* ok we have an internal node on the local processor, need to decide if we open it and go further or keep it */
                nop = &Nodes[no];
                int in_foreign = (no >= maxPart + maxNodes && no < maxPart + maxNodes + maxForeignNodes);
                /* LET guard: if a foreign node has nextnode < 0 (unreplaced -1 sentinel from
                 * unpack, meaning this is the last topleaf with no sibling), opening the node
                 * would immediately exit the while(no >= 0) walk and skip its force contribution.
                 * Force multipole treatment instead so the node's contribution is accumulated. */
                int foreign_force_multipole = (in_foreign && (nop->u.d.nextnode < 0));

                /* C1: foreign-leaf identity lookup (host sidecar; foreign_slot = no-(maxPart+maxNodes),
                 * EXPLICIT and bounds-checked -- not the node index no-maxPart). */
                int    fl_tag = 0, fl_type = -1;
                double fl_zeta = 0.0, fl_soft = 0.0;
                if(in_foreign && ForeignLeafTag) {
                    int fs = no - (maxPart + maxNodes);
                    if(fs >= 0 && fs < MaxForeignNodes) {
                        fl_tag  = ForeignLeafTag[fs];
                        fl_type = ForeignLeafType[fs];
                        fl_zeta = (double) ForeignLeafZeta[fs];
                        fl_soft = (double) ForeignLeafSoft[fs];
                    }
                }

                mass = nop->u.d.mass;
                if(mass <= 0) /* nothing in the node */
                {
                    no = nop->u.d.sibling;
                    continue;
                }
                //if(nop->N_part <= 1)
                if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    if(mass) /* open cell */
                    {
                        if(!foreign_force_multipole)
                        {
                            no = nop->u.d.nextnode;
                            continue;
                        }
                    }
                }
                if(nop->Ti_current != ti_Current) // add this so that threads arriving here after the the node has been drifted do not have to enter critical at all!
                {
#ifdef _OPENMP
#pragma omp critical(_nodedriftforce_)
#endif
                    {
                        if(nop->Ti_current != ti_Current) {force_drift_node(no, ti_Current);}
                    }
                }

                dr = nop->u.d.s - pos;
                GRAVITY_NEAREST_XYZ(dr[0],dr[1],dr[2],-1);
                r2 = dr.norm_sq();
                /* Acceptance geometry via the shared predicate (gravtree_opening.h), the single home
                 * for the node opening decision. The caller owns the wrapped dr/r2 (also used below
                 * for the accepted-node force) and the foreign-multipole policy; the predicate is
                 * foreign-blind geometry. PM short-range cull, neighbour sphere-box / softening-open,
                 * the angular and relative opening criteria, and the sink-direct gate all live in the
                 * predicate. */
                {
                    double cen0 = nop->center[0] - pos[0];
                    double cen1 = nop->center[1] - pos[1];
                    double cen2 = nop->center[2] - pos[2];
#ifdef PMGRID
                    double pred_rcut = rcut, pred_rcut2 = rcut2;
#else
                    double pred_rcut = 0.0, pred_rcut2 = 0.0;
#endif
#ifdef GRAVITY_HYBRID_OPENING_CRIT
                    int pred_is_first_step = (All.Ti_Current == 0 && RestartFlag != 1);
#else
                    int pred_is_first_step = 0;
#endif
#if (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES)) && defined(SINGLE_STAR_DIRECT_GRAVITY_RADIUS)
                    int pred_n_sink = (int)nop->N_SINK;
#else
                    int pred_n_sink = 0;
#endif
                    gravtree_open_t pred = gravtree_open_decision_from_distances(
                        r2, cen0, cen1, cen2, soft, h, aold, ptype,
                        nop->len, mass, nop->maxsoft,
                        pred_rcut, pred_rcut2, pred_n_sink, pred_is_first_step);
                    /* Foreign LET policy (mirrors the GPU walk exactly).  A foreign tagged leaf
                     * (fl_tag==1) is a TERMINAL leaf source: its nextnode is the DFS CONTINUATION after
                     * the leaf, NOT a child to descend.  A predicate OPEN on it must mean "accept this
                     * already-leaf source with leaf semantics" (C1 restores the leaf identity below),
                     * never "descend".  foreign_force_multipole (the nextnode<0 sentinel) is the other
                     * terminal that must be accepted.  Both fall through to the accept path; only a
                     * genuine descendable node takes nextnode.  (Pre-fix the descend guard keyed on
                     * foreign_force_multipole alone, so a tagged leaf whose continuation resolved to a
                     * valid >=0 slot was OPENED -> advanced to its continuation -> its mass was never
                     * summed; the dropped foreign-leaf mass is rank-asymmetric, breaking force
                     * reciprocity / momentum conservation at np>=2 under adaptive softening.) */
                    int foreign_real_leaf = (in_foreign && fl_tag == 1);
                    int must_accept_foreign_terminal = foreign_force_multipole || foreign_real_leaf;
                    if(pred == GRAV_SKIP_NODE) {no = nop->u.d.sibling; continue;}
                    if(pred == GRAV_OPEN_NODE && !must_accept_foreign_terminal) {no = nop->u.d.nextnode; continue;}
                    /* C1 permanent invariant guard (predicate-keyed; mirrors the GPU walk): a
                     * predicate-OPEN foreign terminal forced to multipole that is NOT a tagged real
                     * leaf is an unopenable aggregate that would silently downgrade leaf-sensitive
                     * physics. Hard-surface (controlled stop) until C2/owner-continuation exists. */
                    if(pred == GRAV_OPEN_NODE && foreign_force_multipole && fl_tag != 1) {
                        printf("[GRAV-INVARIANT VIOLATION rank=%d] CPU walk: predicate-OPEN foreign-terminal "
                               "node %d accepted as multipole but not a tagged real leaf (unopenable "
                               "aggregate in leaf-sensitive support); C2/owner-continuation required. Stopping.\n",
                               ThisTask, no);
                        fflush(stdout); endrun(90000087);
                    }
                }

                /* ok we will be using this node, can now set variables that depend on it */
                h_p = nop->maxsoft;
                zeta_sec = 0; ptype_sec = -1; /* set secondary softening and zeta terms */
                /* C1: a tagged real foreign single-particle leaf is consumed with particle-leaf
                 * secondary semantics -- restore the Type + AGS_zeta the node moment cannot carry,
                 * via the shared seam (identical to the GPU walk). */
                if(fl_tag == 1) { grav_apply_foreign_leaf_identity(fl_tag, fl_type, fl_zeta, fl_soft, &ptype_sec, &zeta_sec, &h_p); }
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                gasmass = nop->gasmass;
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
                r_source = grav_spherical_symmetry_r_from_center(nop->u.d.s[0],nop->u.d.s[1],nop->u.d.s[2],center[0],center[1],center[2]);
#endif
#if defined(COMPUTE_JERK_IN_GRAVTREE) || defined(SINK_DYNFRICTION_FROMTREE)
                dv = Extnodes[no].vs - vel;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                cr_injection = nop->cr_injection;
#endif
                
#ifdef RT_USE_GRAVTREE
                if(valid_gas_particle_for_rt)    /* we have a (valid) gas particle as target */
                {
                    int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {mass_stellarlum[kf] = nop->stellar_lum[kf];}
#ifdef CHIMES_STELLAR_FLUXES
                    for(kf = 0; kf < CHIMES_LOCAL_UV_NBINS; kf++)
                    {
                        chimes_mass_stellarlum_G0[kf] = nop->chimes_stellar_lum_G0[kf];
                        chimes_mass_stellarlum_ion[kf] = nop->chimes_stellar_lum_ion[kf];
                    }
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                    d_stellarlum = nop->rt_source_lum_s - pos;
                    GRAVITY_NEAREST_XYZ(d_stellarlum[0],d_stellarlum[1],d_stellarlum[2],-1);
#else
                    d_stellarlum = dr;
#endif
#ifdef SINK_PHOTONMOMENTUM
                    mass_sinklumwt_forradfb = sink_fb_angleweight(nop->sink_lum, nop->sink_lum_grad, d_stellarlum[0],d_stellarlum[1],d_stellarlum[2]);
#endif
                }
#endif // RT_USE_GRAVTREE
                
#ifdef DM_SCALARFIELD_SCREENING
                if(ptype != 0) {d_dm = nop->s_dm - pos; mass_dm = nop->mass_dm;} else {d_dm = {}; mass_dm = 0;} /* we have a dark matter particle as target */
#endif
#if defined(SINK_DYNFRICTION_FROMTREE)
                m_j_eff_for_df = (nop->u.d.mass) / (nop->N_part);
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
                j_zeta_tidal_tensorps_prevstep=nop->tidal_tensorps_prevstep;
#endif
                
#ifdef SINK_CALC_DISTANCES // NOTE: moved this to AFTER the checks for node opening, because we only want to record BH positions from the nodes that actually get used for the force calculation - MYG
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
                grav_sink_prox_node_specialweighted(r2, Extnodes[no].vs, ptype, sink_prox);
#endif
                if(nop->sink_mass > 0)        /* found a node with non-zero BH mass */
                {
                    Vec3<double> sink_dr = nop->sink_pos - pos;  /* SHEA:  now using sink_pos instead of center */
                    GRAVITY_NEAREST_XYZ(sink_dr[0],sink_dr[1],sink_dr[2],-1);
                    grav_sink_prox_target_t prox_target = {}; prox_target.ptype = ptype; prox_target.pmass = pmass; prox_target.soft = soft;
#if defined(SINGLE_STAR_TIMESTEPPING)
                    prox_target.vel = vel;
#endif
                    grav_sink_prox_node_src_t prox_src = {}; prox_src.sink_mass = nop->sink_mass;
#if defined(SINGLE_STAR_FIND_BINARIES)
                    prox_src.n_sink = (int)nop->N_SINK;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
                    prox_src.motion.vel = nop->sink_vel;
#endif
#if defined(SPECIAL_POINT_MOTION)
                    prox_src.motion.acc = nop->sink_acc;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
                    prox_src.motion.max_feedback_vel = nop->MaxFeedbackVel;
#endif
                    grav_sink_prox_node_accumulate(r2, sink_dr, prox_src, prox_target, sink_prox);
                }
#endif // SINK_CALC_DISTANCES
                
            } /* ok we've completed all the opening criteria -- we will keep this node or particle as-is */
            
            
            
            
            if((r2 > 0) && (mass > 0)) // only go forward if mass positive and there is separation -- this is check for the whole block below, which should no include 'self' terms
            {
                r = sqrt(r2);
                /* pair-wise gravity terms (Newtonian/softened selection, softening symmetrization,
                 * AGS zeta corrections) via the shared contribution kernel (gravtree_force_kernel.h),
                 * the single home for the pair physics on both walks. */
                {
                    grav_force_pair_t pair_out = grav_force_pair(r, r2, mass, h, h_p, ptype, ptype_sec, pmass,
                                                                 zeta, zeta_sec, AGS_kernel_shared_BITFLAG);
                    fac_accel = pair_out.fac_accel;
#ifdef EVALPOTENTIAL
                    fac_pot = pair_out.fac_pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
                    fac_tidal = pair_out.fac_tidal; fac2_tidal = pair_out.fac2_tidal;
#endif
                }
                
                
#ifdef PMGRID
                tabindex = grav_pm_shortrange_tabindex(asmthfac, r);
                if(grav_pm_shortrange_in_range(tabindex))
#endif // PMGRID //
                {
#ifdef PMGRID
                    grav_force_apply_pm_truncation(pm, tabindex, fac_pot, fac_accel);
#endif
#ifdef EVALPOTENTIAL
                    pot += (fac_pot);
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC) && !defined(PMGRID)
                    pot += (mass * ewald_pot_corr(dr[0], dr[1], dr[2]));
#endif
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
                    r_target = grav_spherical_symmetry_r_from_center(pos[0],pos[1],pos[2],center[0],center[1],center[2]); // distance of target point from box center
                    grav_spherical_symmetry_force_override(r_source, r_target, h, mass, center[0],center[1],center[2], pos[0],pos[1],pos[2], dr, fac_accel);
#endif

                    /* actually add the accelerations, now that we've corrected for the ewald and other terms */
                    acc += fac_accel * dr;
                    
                    
#if defined(SINK_DYNFRICTION_FROMTREE)
                    grav_sink_dynfriction_accumulate(dr, dv, fac_accel, mass, sink_mass, m_j_eff_for_df, ptype, acc);
#endif
                    
                    
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION /* 'correction' terms for variable smoothing lengths (analogous to the ags-zeta terms); shared helper */
                    grav_ags_tidal_criterion_accumulate(r, r2, dr, mass, h, h_p, ptype, ptype_sec, fac_tidal, fac2_tidal,
                                                        i_zeta_tidal_tensorps_prevstep, j_zeta_tidal_tensorps_prevstep, tidal_zeta, acc);
#endif


#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
#ifdef GRAVITY_SPHERICAL_SYMMETRY
                    fac2_tidal = grav_spherical_symmetry_fac2_tidal_override(r_source, r_target, h, mass);
#endif
                    grav_tidal_tensor_accumulate(dr, fac_tidal, fac2_tidal, pm, tabindex, tidal_tensorps);
#endif // COMPUTE_TIDAL_TENSOR_IN_GRAVTREE //
#ifdef COMPUTE_JERK_IN_GRAVTREE
                    grav_jerk_accumulate(dv, dr, fac_accel, fac2_tidal, ptype, jerk);
#endif
                } // closes TABINDEX<NTAB
                
                ninteractions++;
                
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
                if(r < r_for_total_menclosed) {m_enc_in_rcrit += mass;}
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
                tree_mass += mass;
#endif
#ifdef RT_USE_TREECOL_FOR_NH
                grav_treecol_accumulate(dr, r, fac_accel, gasmass, mass, angular_bin_size, treecol_angular_bins);
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                grav_cr_lebron_accumulate(ptype, r, soft, cr_injection, cr_active_gate, cr_t_max, pm, SubGrid_CosmicRayEnergyDensity);
#endif
#ifdef RT_USE_GRAVTREE
                if(valid_gas_particle_for_rt)    /* we have a (valid) gas particle as target; payload formulas in the shared helper */
                {
                    grav_rt_src_t rt_src = {}; rt_src.d_stellarlum = d_stellarlum; rt_src.soft = soft; rt_src.mass_stellarlum = mass_stellarlum;
#ifdef CHIMES_STELLAR_FLUXES
                    rt_src.chimes_mass_stellarlum_G0 = chimes_mass_stellarlum_G0; rt_src.chimes_mass_stellarlum_ion = chimes_mass_stellarlum_ion;
#endif
#ifdef SINK_PHOTONMOMENTUM
                    rt_src.mass_sinklumwt_forradfb = mass_sinklumwt_forradfb;
#endif
#if defined(RT_LEBRON) && !defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
                    rt_src.fac_stellum = fac_stellum;
#endif
                    grav_rt_accum_t rt_accum = {};
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
                    rt_accum.Rad_E_gamma = Rad_E_gamma;
#endif
#ifdef CHIMES_STELLAR_FLUXES
                    rt_accum.chimes_flux_G0 = chimes_flux_G0; rt_accum.chimes_flux_ion = chimes_flux_ion;
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
                    rt_accum.incident_flux_uv = &incident_flux_uv; rt_accum.incident_flux_euv = &incident_flux_euv;
#endif
#ifdef SINK_COMPTON_HEATING
                    rt_accum.incident_flux_agn = &incident_flux_agn;
#endif
#ifdef RT_OTVET
                    rt_accum.RT_ET = RT_ET;
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
                    rt_accum.Rad_Flux = Rad_Flux;
#endif
                    grav_rt_payload_accumulate(rt_src, rt_accum, acc);
                } // closes if(valid_gas_particle_for_rt)

#endif // RT_USE_GRAVTREE


#ifdef DM_SCALARFIELD_SCREENING
                if(ptype != 0)    /* we have a dark matter particle as target */
                {
                    grav_dm_scalarfield_accumulate(d_dm, mass_dm, h, pm, acc);
                } // closes if(ptype != 0)
#endif // DM_SCALARFIELD_SCREENING //
                
            } // closes (if((r2 > 0) && (mass > 0))) check
            
            
            /* advance for used nodes: note this used to be above, now handled down here so we can use the 'no/nop' structures above */
            if(no < maxPart) {
                if(TakeLevel >= 0) {P[no].GravCost[TakeLevel] += 1.0;} /* node was used */
                no = Nextnode[no];
            } else {
                if(TakeLevel >= 0) {nop->GravCost += 1.0;}
                no = nop->u.d.sibling;
            }
            
        } // closes inner (while(no>=0)) check
    } // closes outer (while(no>=0)) check


    /* store result at the proper place (local target only; the imported-particle export path is retired) */
    {
        P[target].GravAccel = acc;
#ifdef RT_USE_TREECOL_FOR_NH
        int k; for(k=0; k < RT_USE_TREECOL_FOR_NH; k++) P[target].ColumnDensityBins[k] = treecol_angular_bins[k];
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
        P[target].TreeMass = tree_mass;
#endif
#ifdef RT_OTVET
        if(valid_gas_particle_for_rt) {int k; for(k=0;k<N_RT_FREQ_BINS;k++) {CellP[target].ET[k] = RT_ET[k];}} else {if(P[target].Type==0) {int k; for(k=0;k<N_RT_FREQ_BINS;k++) {CellP[target].ET[k] = {};}}}
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
        if(valid_gas_particle_for_rt) {CellP[target].Rad_Flux_UV = incident_flux_uv;}
        if(valid_gas_particle_for_rt) {CellP[target].Rad_Flux_EUV = incident_flux_euv;}
#endif
#ifdef CHIMES_STELLAR_FLUXES
        if(valid_gas_particle_for_rt)
        {
            int kc; for (kc = 0; kc < CHIMES_LOCAL_UV_NBINS; kc++) {CellP[target].Chimes_G0[kc] = chimes_flux_G0[kc]; CellP[target].Chimes_fluxPhotIon[kc] = chimes_flux_ion[kc];}
        }
#endif
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
        P[target].MencInRcrit = m_enc_in_rcrit;
#endif
#ifdef SINK_COMPTON_HEATING
        if(valid_gas_particle_for_rt) {CellP[target].Rad_Flux_AGN = incident_flux_agn;}
#endif
#if defined(COSMIC_RAY_SUBGRID_LEBRON)
        if(P[target].Type==0) {CellP[target].SubGrid_CosmicRayEnergyDensity = SubGrid_CosmicRayEnergyDensity;}
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
        if(valid_gas_particle_for_rt) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {CellP[target].Rad_E_gamma[kf] = Rad_E_gamma[kf];}}
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
        if(valid_gas_particle_for_rt) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {CellP[target].Rad_Flux[kf] = Rad_Flux[kf];}}
#endif
#ifdef EVALPOTENTIAL
        P[target].Potential = pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
        P[target].tidal_tensorps = tidal_tensorps;
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        P[target].tidal_zeta = tidal_zeta;
#endif
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
        P[target].GravJerk = jerk;
#endif
#ifdef SINK_CALC_DISTANCES
        P[target].Min_Distance_to_Sink = sqrt( sink_prox.Min_Distance_to_Sink2 );
        P[target].Min_xyz_to_Sink = sink_prox.Min_xyz_to_Sink;   /* remember, dr = x_SINK - myx */
#ifdef SPECIAL_POINT_MOTION
        {
            P[target].vel_of_nearest_special = sink_prox.vel_of_nearest_special;
            P[target].acc_of_nearest_special = sink_prox.acc_of_nearest_special;
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
            P[target].weight_sum_for_special_point_smoothing = sink_prox.weight_sum_for_special_point_smoothing; /* weighted sum needed */
#endif
        }
#endif
#ifdef SINGLE_STAR_FIND_BINARIES
        P[target].is_in_a_binary=0; P[target].Min_Sink_OrbitalTime=sink_prox.Min_Sink_OrbitalTime; //orbital time for binary
        if (sink_prox.Min_Sink_OrbitalTime<MAX_REAL_NUMBER)
        {
            P[target].is_in_a_binary=1; P[target].comp_Mass=sink_prox.comp_Mass; //mass of binary companion
            P[target].comp_dx = sink_prox.comp_dx; P[target].comp_dv = sink_prox.comp_dv;
        }
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
        P[target].Min_Sink_Approach_Time = sqrt(sink_prox.Min_Sink_Approach_Time);
        P[target].Min_Sink_Freefall_time = sqrt(sqrt(sink_prox.Min_Sink_Freefall_time)/All.G);
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
        P[target].Min_Sink_FeedbackTime = sqrt(sink_prox.Min_Sink_FeedbackTime);
#endif
#endif
#endif // SINK_CALC_DISTANCES
    }

    return ninteractions;
}





#ifdef BOX_PERIODIC
/*! This function computes the Ewald correction, and is needed if periodic
 *  boundary conditions together with a pure tree algorithm are used. Note
 *  that the ordinary tree walk does not carry out this correction directly
 *  as it was done in Gadget-1.1. Instead, the tree is walked a second
 *  time. This is actually faster because the "Ewald-Treewalk" can use a
 *  different opening criterion than the normal tree walk. In particular,
 *  the Ewald correction is negligible for particles that are very close,
 *  but it is large for particles that are far away (this is quite
 *  different for the normal direct force). So we can here use a different
 *  opening criterion. Sufficient accuracy is usually obtained if the node
 *  length has dropped to a certain fraction ~< 0.25 of the
 *  BoxLength. However, we may only short-cut the interaction list of the
 *  normal full Ewald tree walk if we are sure that the whole node and all
 *  daughter nodes "lie on the same side" of the periodic boundary,
 *  i.e. that the real tree walk would not find a daughter node or particle
 *  that was mapped to a different nearest neighbour position when the tree
 *  walk would be further refined.
 */
int force_treeevaluate_ewald_correction(int target, int *exportflag, int *exportnodecount, int *exportindex)
{
    struct NODE *nop = 0;
    int signx, signy, signz, nexp, openflag, task, no, cost;
    double mass, r2, u;   /* u: scratch in the periodic-boundary shortcut; interp locals now in the SSOT helper */
    double boxsize, boxhalf, aold, xtmp; xtmp=0;
    Vec3<double> pos, dr;
    Vec3<MyDouble> acc = {};

    boxsize = All.BoxSize;
    boxhalf = 0.5 * All.BoxSize;

    cost = 0;
    pos = P[target].Pos;
    aold = All.ErrTolForceAcc * P[target].OldAcc;

    no = All.MaxPart;        /* root node */

    while(no >= 0)   /* outer loop runs once: the mode-1 imported-NodeList iteration is retired */
    {
        while(no >= 0)
        {
            if(no < All.MaxPart)    /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */
                if(P[no].Ti_current != All.Ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_particledriftewald_)
#endif
                    {
                        if(P[no].Ti_current != All.Ti_Current) {
                            drift_particle(no, All.Ti_Current);
                            gizmo_mark_kernel_radius_dirty_indices(&no, 1);
                        }
                    }
                }

                dr = P[no].Pos - pos;
                mass = P[no].Mass;
            }
            else            /* we have an  internal node */
            {
                if(no >= All.MaxPart + MaxNodes + MaxForeignNodes)    /* pseudo particle (Phase 9: foreign-node range below pseudos) */
                {
                    /* LET-incompleteness DETECTOR (the MPI export round-trip is retired): reaching a
                     * non-empty pseudo means this target's Ewald correction is not covered by the local
                     * LET; record it so Nexport>0 triggers gravity_tree()'s graceful controlled-stop. */
                    if(exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes + MaxForeignNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if(exportnodecount[task] == NODELISTLENGTH)
                    {
                        int exitFlag = 0;
#ifdef _OPENMP
#pragma omp critical(_nexportewald_)
#endif
                        {
                            if(Nexport >= All.BunchSize)
                            {
                                /* out if buffer space. Need to discard work for this particle and interrupt */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport++;
                            }
                        }
                        if(exitFlag) {return -1;} /* buffer has filled -- important that only this and other buffer-full conditions return the negative condition for the routine */

                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }

                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] = DomainNodeIndex[no - (All.MaxPart + MaxNodes + MaxForeignNodes)];

                    if(exportnodecount[task] < NODELISTLENGTH) {DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;}
                    no = Nextnode[no - MaxNodes - MaxForeignNodes];
                    continue;
                }

                nop = &Nodes[no];

                //if(nop->N_part <= 1) /* open cell */
                if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    no = nop->u.d.nextnode;
                    continue;
                }
                if(nop->Ti_current != All.Ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_nodedriftewald_)
#endif
                    {
                        if(nop->Ti_current != All.Ti_Current) {force_drift_node(no, All.Ti_Current);}
                    }
                }

                mass = nop->u.d.mass;
                dr = nop->u.d.s - pos;
            }
            GRAVITY_NEAREST_XYZ(dr[0],dr[1],dr[2],-1);

            if(no < All.MaxPart)
            {no = Nextnode[no];}
            else            /* we have an internal node. Need to check opening criterion */
            {
                openflag = 0;
                r2 = dr.norm_sq();
                if(r2 <= 0) {r2=MIN_REAL_NUMBER;}
                if(All.ErrTolTheta)    /* check Barnes-Hut opening criterion */
                {
                    if(nop->len * nop->len > r2 * All.ErrTolTheta * All.ErrTolTheta)
                    {
                        openflag = 1;
                    }
                }
#ifndef GRAVITY_HYBRID_OPENING_CRIT
                else        /* check relative opening criterion */
#else
                    if(!(All.Ti_Current == 0 && RestartFlag != 1))
#endif
                    {
                        if(mass * nop->len * nop->len > r2 * r2 * aold)
                        {
                            openflag = 1;
                        }
                        else
                        {
                            if(GRAVITY_NGB_PERIODIC_BOX_LONG_X(nop->center[0] - pos[0], nop->center[1] - pos[1], nop->center[2] - pos[2], -1) < 0.60 * nop->len)
                            {
                                if(GRAVITY_NGB_PERIODIC_BOX_LONG_Y(nop->center[0] - pos[0], nop->center[1] - pos[1], nop->center[2] - pos[2], -1) < 0.60 * nop->len)
                                {
                                    if(GRAVITY_NGB_PERIODIC_BOX_LONG_Z(nop->center[0] - pos[0], nop->center[1] - pos[1], nop->center[2] - pos[2], -1) < 0.60 * nop->len)
                                    {
                                        openflag = 1;
                                    }
                                }
                            }
                        }
                    }

                if(openflag)
                {
                    /* now we check if we can avoid opening the cell */

                    u = nop->center[0] - pos[0];
                    if(u > boxhalf) {u -= boxsize;}
                    if(u < -boxhalf) {u += boxsize;}
                    if(fabs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[1] - pos[1];
                    if(u > boxhalf) {u -= boxsize;}
                    if(u < -boxhalf) {u += boxsize;}
                    if(fabs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[2] - pos[2];
                    if(u > boxhalf) {u -= boxsize;}
                    if(u < -boxhalf) {u += boxsize;}
                    if(fabs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                    
                    /* if the cell is too large, we need to refine it further */
                    if(nop->len > 0.20 * boxsize)
                    {
                        /* cell is too large */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                
                no = nop->u.d.sibling;    /* ok, node can be used */
            }
            
            /* compute the Ewald correction force */

            if(dr[0] < 0)
            {
                dr[0] = -dr[0];
                signx = +1;
            }
            else
            {signx = -1;}
            if(dr[1] < 0)
            {
                dr[1] = -dr[1];
                signy = +1;
            }
            else
            {signy = -1;}
            if(dr[2] < 0)
            {
                dr[2] = -dr[2];
                signz = +1;
            }
            else
            {signz = -1;}
            /* trilinear interp of the Ewald force octant tables via the shared SSOT helper
             * (gravtree_ewald.h): the index + 8 weights are built once from |dr| and applied to
             * all three force tables; the odd-force per-component signs stay here. */
            grav_ewald_interp_weights ew = grav_ewald_interp_setup(dr[0], dr[1], dr[2], fac_intp);
            acc[0] += mass * signx * grav_ewald_interp_apply(&fcorrx[0][0][0], ew);
            acc[1] += mass * signy * grav_ewald_interp_apply(&fcorry[0][0][0], ew);
            acc[2] += mass * signz * grav_ewald_interp_apply(&fcorrz[0][0][0], ew);
            cost++;
        }
        
    }

    /* add the result at the proper place (local target only; the imported-particle export path is retired) */
    P[target].GravAccel += acc;

    return cost;
}
#endif // #ifdef BOX_PERIODIC //









#ifdef SUBFIND
/* Local-only Barnes-Hut potential walk for SUBFIND unbinding. Foreign-rank mass is
 * supplied by the locally-installed LET nodes (LET is mandatory and force_treebuild
 * hard-stops on an incomplete LET), so no export/import round-trip is needed: each
 * rank fully computes its own particles' DM_Potential. Reaching a pseudo-particle
 * would mean the LET is incomplete -- a hard correctness failure, since the export
 * fallback is retired -- so we request a controlled stop and abandon the walk
 * (return nonzero; the caller drains at its next collective poll before any
 * potential-derived logic). */
int subfind_force_treeevaluate_potential(int target)
{
    struct NODE *nop = 0;
    MyDouble pot = 0;
    int no = All.MaxPart;    /* root node */
    double r2, dx, dy, dz, mass, r, u, h, h_inv, pos_x, pos_y, pos_z;

    pos_x = P[target].Pos[0];
    pos_y = P[target].Pos[1];
    pos_z = P[target].Pos[2];
    h = ForceSoftening_KernelRadius(target); h_inv = 1.0 / h;

    while(no >= 0)
    {
        if(no < All.MaxPart)    /* single particle: node index is the particle index */
        {
            dx = P[no].Pos[0] - pos_x;
            dy = P[no].Pos[1] - pos_y;
            dz = P[no].Pos[2] - pos_z;
            mass = P[no].Mass;
        }
        else
        {
            if(no >= All.MaxPart + MaxNodes + MaxForeignNodes)    /* pseudo particle */
            {
                /* LET supplies complete foreign coverage in all allowed builds; the
                 * export fallback is retired. A pseudo here means the LET is incomplete. */
                endrun(90000080);    /* graceful stop request; does NOT return -- abandon this walk */
                return 1;
            }

            nop = &Nodes[no];
            mass = nop->u.d.mass;
            if(!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if(mass) {no = nop->u.d.nextnode; continue;}    /* open cell */
            }

            dx = nop->u.d.s[0] - pos_x;
            dy = nop->u.d.s[1] - pos_y;
            dz = nop->u.d.s[2] - pos_z;
        }
        GRAVITY_NEAREST_XYZ(dx,dy,dz,-1);
        r2 = dx * dx + dy * dy + dz * dz;
        if(no < All.MaxPart)
        {
            no = Nextnode[no];
        }
        else            /* internal node: check Barnes-Hut opening criterion */
        {
            double ErrTolThetaSubfind = All.ErrTolTheta;
            if(nop->len * nop->len > r2 * ErrTolThetaSubfind * ErrTolThetaSubfind)
            {
                if(mass) {no = nop->u.d.nextnode; continue;}    /* open cell */
            }
            no = nop->u.d.sibling;    /* node can be used */
        }

        r = sqrt(r2);
        if(r >= h)
        {pot += (-mass / r);}
        else
        {
            u = r * h_inv;
            pot += ( mass * kernel_gravity(u, h_inv, 1, -1) );
        }
    }

    P[target].u.DM_Potential = pot;
    return 0;
}
#endif // SUBFIND //




/*! This function allocates the memory used for storage of the tree and of
 *  auxiliary arrays needed for tree-walk and link-lists.  Usually,
 *  maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 *  tree for up to maxpart particles.
 */
void force_treeallocate(int maxnodes, int maxpart)
{
    int i;
    size_t bytes;
    double allbytes = 0, allbytes_topleaves = 0;
    double u;
    
    tree_allocated_flag = 1;
    DomainNodeIndex = (int *) mymalloc("DomainNodeIndex", bytes = NTopleaves * sizeof(int));
    allbytes_topleaves += bytes;
    /* Top-leaf-router geometry SSOT (allocated immediately after DomainNodeIndex
     * so the mymalloc LIFO free order in force_treefree is simply the reverse).
     * Sized for ALL topnodes (internal + leaf); populated in
     * force_create_empty_nodes. */
    TopNodeNodeIndex = (int *) mymalloc("TopNodeNodeIndex", bytes = (NTopnodes > 0 ? NTopnodes : 1) * sizeof(int));
    allbytes_topleaves += bytes;
    for(i = 0; i < NTopnodes; i++) TopNodeNodeIndex[i] = -1;  /* sentinel: post-build validation requires all populated */
    MaxNodes = maxnodes;
    /* Phase 9 LET: foreign-node headroom in Nodes_base/Extnodes_base/Nextnode.
     * Index map (single source of truth):
     *   [0,                                     MaxPart)                                  -> particles
     *   [MaxPart,                               MaxPart+MaxNodes)                         -> local tree nodes
     *   [MaxPart+MaxNodes,                      MaxPart+MaxNodes+MaxForeignNodes)         -> foreign tree nodes (LET unpack)
     *   [MaxPart+MaxNodes+MaxForeignNodes,      MaxPart+MaxNodes+MaxForeignNodes+NTopleaves) -> pseudo-particles
     * SoA slot for any node index `no` (local OR foreign): idx = no - MaxPart (same formula).
     * MaxForeignNodes = ceil(LETAllocFactor * (MaxNodes + synth_overhead)) where synth_overhead accounts
     * for synthesized particle leaves (one entry per particle that is a direct child of an essential
     * multi-particle node).  Synthesis overhead ≤ NumPart_per_rank per received rank; using
     * 2 × (All.MaxPart / PartAllocFactor) ≈ 2 × TotNumPart / NTask as headroom covers NTask=2
     * worst case (both ranks overlap entirely) while staying modest for large NTask.
     * Numforeignnodes (current count, <= MaxForeignNodes) is reset on each LET exchange.
     * On non-GPU builds the foreign range is empty (MaxForeignNodes = 0); legacy export path is used. */
    {
        double synth_overhead = 2.0 * (double)All.MaxPart / (double)All.PartAllocFactor;
        long long base = (long long) ceil(All.LETAllocFactor * ((double)MaxNodes + synth_overhead));
        /* Take the larger of the parameter-derived floor and the runtime adaptive
         * floor (ratcheted by force_treebuild on a retryable LET overflow). This is
         * the SOLE place MaxForeignNodes is derived. */
        long long want = (base > RuntimeMinLETForeignNodes) ? base : RuntimeMinLETForeignNodes;
        if(want < 0) want = 0;
        if(want > (long long)INT_MAX)
        {
            printf("force_treeallocate: LET foreign-node capacity %lld exceeds INT_MAX "
                   "(MaxNodes=%d, RuntimeMinLETForeignNodes=%lld). Stopping.\n",
                   want, MaxNodes, RuntimeMinLETForeignNodes);
            fflush(stdout);
            endrun(90000082);   /* graceful drain; same family as the foreign-arena UVM OOM below */
            want = (long long)INT_MAX;
        }
        MaxForeignNodes = (int) want;
    }
    if(MaxForeignNodes < 0) {MaxForeignNodes = 0;}
    Numforeignnodes = 0;
    /* Phase 10.5: FOF/SUBFIND/twopoint pseudo-particle threshold ported from
     * `MaxPart+MaxNodes` to `MaxPart+MaxNodes+MaxForeignNodes` (matches the
     * forcetree.cc/let_pack.cc convention).  FOF/SUBFIND build their own local
     * trees and never call `let_run_exchange()`, so during their walks
     * Numforeignnodes==0 and the foreign range is empty; the threshold update
     * is correct in both regimes (LET-active gravity walks and LET-inactive
     * halo-finding walks).  The Phase 9 LET-FOF/SUBFIND startup gate is
     * therefore retired. */
    long long total_node_slots = (long long) MaxNodes + (long long) MaxForeignNodes + 1LL;
    /* Phase 6.8d: Nodes_base / Extnodes_base live in SharedSpace (UVM) so GPU
     * kernels can read/write them directly.  Same pattern as Father[] (6.6) and
     * Nextnode[] (6.8e below).  Skip mymalloc accounting; kokkos_malloc has its
     * own.  Phase 9: extended by MaxForeignNodes for foreign-tree storage. */
    bytes = (size_t) total_node_slots * sizeof(struct NODE);
    Nodes_base = (struct NODE *) gpu_tree_alloc_bytes(bytes);
    if(!Nodes_base)
    {
        printf("failed to allocate %d local + %d foreign (LETAllocFactor=%g) tree-nodes (%g MB) in SharedSpace.\n",
               MaxNodes, MaxForeignNodes, All.LETAllocFactor, bytes / (1024.0 * 1024.0));
        /* UVM OOM (per-rank). Soft bad-stop + return BEFORE `Nodes = Nodes_base - MaxPart`
         * so the wild alias is never formed; Nodes_base stays NULL for the caller's check.
         * DomainNodeIndex + tree_allocated_flag are already set (partial allocation) -- safe
         * not because nothing was allocated, but because the caller immediately polls (all-rank
         * sites) or NULL-checks + skips payload (restart turn) before any tree use. */
        endrun(90000082);
        return;
    }
    bytes = (size_t) total_node_slots * sizeof(struct extNODE);
    Extnodes_base = (struct extNODE *) gpu_tree_alloc_bytes(bytes);
    if(!Extnodes_base)
    {
        printf("failed to allocate %d local + %d foreign tree-extnodes (%g MB) in SharedSpace.\n",
               MaxNodes, MaxForeignNodes, bytes / (1024.0 * 1024.0));
        /* UVM OOM (per-rank). Soft bad-stop + return BEFORE `Extnodes = Extnodes_base - MaxPart`
         * so the wild alias is never formed; Extnodes_base stays NULL for the caller's check. */
        endrun(90000083);
        return;
    }
    Nodes = Nodes_base - All.MaxPart;
    Extnodes = Extnodes_base - All.MaxPart;
    /* Phase 6.8e: Nextnode also in SharedSpace; soa->nextnode_aux is aliased to
     * this pointer (no separate buffer, no per-walk memcpy).
     * Phase 9: indexed via Nextnode[no - MaxNodes - MaxForeignNodes] for pseudo-particles
     * after the foreign range; foreign nodes carry their next-sibling pointers in NODE.u.d
     * directly so they don't consume Nextnode[] slots, but we extend the buffer so the
     * pseudo-particle range stays in bounds after the index shift. */
    long long nextnode_slots = (long long) maxpart + (long long) NTopnodes + (long long) MaxForeignNodes;
    bytes = (size_t) nextnode_slots * sizeof(int);
    Nextnode = (int *) gpu_tree_alloc_bytes(bytes);
    if(!Nextnode)
    {
        printf("Failed to allocate %lld 'Nextnode' slots (%g MB) in SharedSpace\n",
               nextnode_slots, bytes / (1024.0 * 1024.0));
        /* UVM OOM (per-rank). Soft bad-stop + return BEFORE gpu_gravity_tree_alias_nextnode()
         * so a NULL Nextnode is never registered; Nextnode stays NULL for the caller's check. */
        endrun(90000084);
        return;
    }
    gpu_gravity_tree_alias_nextnode(Nextnode, (int) nextnode_slots);
    /* Phase 6.6: Father[] is UVM (SharedSpace) so the GPU father kernel can
     * write into it directly and host readers (setup_smoothinglengths etc.)
     * page-fault on touch.  No per-tree-build deep_copy needed.  Skip the
     * mymalloc accounting; the matching gpu_father_free lives in force_treefree. */
    bytes = (size_t)maxpart * sizeof(int);
    Father = gpu_father_alloc(maxpart);
    if(!Father)
    {
        printf("Failed to allocate %d spaces for 'Father' array (%g MB) in SharedSpace\n",
               maxpart, bytes / (1024.0 * 1024.0));
        /* UVM OOM (per-rank). Soft bad-stop + return BEFORE any Father[i] write/use;
         * Father stays NULL for the caller's check. */
        endrun(90000085);
        return;
    }
    /* C1: foreign-leaf identity sidecar.  Allocated in SharedSpace via the foreign-node arena
     * allocator (gpu_tree_alloc_bytes), the SAME discipline as Nodes_base/Extnodes_base/Nextnode --
     * freed in force_treefree, re-allocated on every force_treebuild retry.  Sized MaxForeignNodes,
     * indexed by foreign_slot = no - (MaxPart+MaxNodes).  Zero-initialized so every slot defaults to
     * leaf_tag=0 (node) before any LET exchange installs real foreign leaves. */
    if(MaxForeignNodes > 0)
    {
        ForeignLeafTag  = (int *)     gpu_tree_alloc_bytes((size_t) MaxForeignNodes * sizeof(int));
        ForeignLeafType = (int *)     gpu_tree_alloc_bytes((size_t) MaxForeignNodes * sizeof(int));
        ForeignLeafZeta = (MyFloat *) gpu_tree_alloc_bytes((size_t) MaxForeignNodes * sizeof(MyFloat));
        ForeignLeafSoft = (MyFloat *) gpu_tree_alloc_bytes((size_t) MaxForeignNodes * sizeof(MyFloat));
        if(!ForeignLeafTag || !ForeignLeafType || !ForeignLeafZeta || !ForeignLeafSoft)
        {
            printf("Failed to allocate %d foreign-leaf sidecar slots.\n", MaxForeignNodes);
            endrun(90000086);
            return;
        }
        memset(ForeignLeafTag,  0, (size_t) MaxForeignNodes * sizeof(int));
        memset(ForeignLeafType, 0, (size_t) MaxForeignNodes * sizeof(int));
        memset(ForeignLeafZeta, 0, (size_t) MaxForeignNodes * sizeof(MyFloat));
        memset(ForeignLeafSoft, 0, (size_t) MaxForeignNodes * sizeof(MyFloat));
    }
    else { ForeignLeafTag = ForeignLeafType = NULL; ForeignLeafZeta = ForeignLeafSoft = NULL; }

    /* Don't add to allbytes — kokkos_malloc accounting is separate. */
    if(first_flag == 0)
    {
        first_flag = 1;
        if(ThisTask == 0)
        {
            printf
            ("Allocated %g MByte for tree, and %g Mbyte for top-leaves.  (presently allocated %g MB)\n",
             allbytes / (1024.0 * 1024.0), allbytes_topleaves / (1024.0 * 1024.0),
             AllocatedBytes / (1024.0 * 1024.0));
            /* Step 13 Phase 2: gravity-node sizing audit. Lists active payload #ifdefs
             * that inflate NODE/extNODE; informs the compact-node variants in Phase 6. */
            printf("Gravity tree node sizes: sizeof(NODE)=%zu B, sizeof(extNODE)=%zu B; "
                   "MyGravFloat=%zu B (mixed-precision gravity %s). Active payload flags:",
                   sizeof(struct NODE), sizeof(struct extNODE), sizeof(MyGravFloat),
#ifdef GIZMO_MIXED_PRECISION_GRAVITY
                   "ON"
#else
                   "OFF"
#endif
                   );
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            printf(" GRAVTREE_CALCULATE_GAS_MASS_IN_NODE");
#endif
#ifdef RT_USE_GRAVTREE
            printf(" RT_USE_GRAVTREE");
#endif
#ifdef CHIMES_STELLAR_FLUXES
            printf(" CHIMES_STELLAR_FLUXES");
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            printf(" RT_SEPARATELY_TRACK_LUMPOS");
#endif
#ifdef SINK_PHOTONMOMENTUM
            printf(" SINK_PHOTONMOMENTUM");
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            printf(" COSMIC_RAY_SUBGRID_LEBRON");
#endif
#ifdef SINK_CALC_DISTANCES
            printf(" SINK_CALC_DISTANCES");
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            printf(" ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION");
#endif
#ifdef DM_SCALARFIELD_SCREENING
            printf(" DM_SCALARFIELD_SCREENING");
#endif
            printf("\n");
        }
        for(i = 0; i < NTAB; i++)
        {
            u = 3.0 / NTAB * (i + 0.5);
            shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
            shortrange_table_potential[i] = erfc(u);
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            shortrange_table_tidal[i] = 4.0 * u * u * u / sqrt(M_PI) * exp(-u * u);
#endif
        }
    }
}


/*! This function frees the memory allocated for the tree, i.e. it frees
 *  the space allocated by the function force_treeallocate().
 */
void force_treefree(void)
{
    if(tree_allocated_flag)
    {
        /* Phase 6.8d/e: SharedSpace (UVM) frees for GPU-addressable tree
         * storage.  Order is reverse-of-alloc (LIFO discipline preserved for
         * the residual mymalloc'd DomainNodeIndex). */
        if(Father)        {gpu_father_free(Father); Father = NULL;}
        gpu_gravity_tree_alias_nextnode(NULL, 0);  /* clear SoA alias before free */
        if(Nextnode)      {gpu_tree_free_bytes(Nextnode);      Nextnode      = NULL;}
        if(Extnodes_base) {gpu_tree_free_bytes(Extnodes_base); Extnodes_base = NULL;}
        if(Nodes_base)    {gpu_tree_free_bytes(Nodes_base);    Nodes_base    = NULL;}
        /* C1: free the foreign-leaf sidecar (SharedSpace, same allocator as the foreign-node arena). */
        if(ForeignLeafTag)  {gpu_tree_free_bytes(ForeignLeafTag);  ForeignLeafTag  = NULL;}
        if(ForeignLeafType) {gpu_tree_free_bytes(ForeignLeafType); ForeignLeafType = NULL;}
        if(ForeignLeafZeta) {gpu_tree_free_bytes(ForeignLeafZeta); ForeignLeafZeta = NULL;}
        if(ForeignLeafSoft) {gpu_tree_free_bytes(ForeignLeafSoft); ForeignLeafSoft = NULL;}
        myfree(TopNodeNodeIndex);   /* LIFO: allocated right after DomainNodeIndex, so freed right before it */
        myfree(DomainNodeIndex);
        tree_allocated_flag = 0;
    }
}





/*! This function dumps some of the basic particle data to a file. In case
 *  the tree construction fails, it is called just before the run
 *  terminates with an error message. Examination of the generated file may
 *  then give clues to what caused the problem.
 */
void dump_particles(void)
{
    FILE *fd;
    char buffer[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    int i;
    
    snprintf(buffer, DEFAULT_PATH_BUFFERSIZE_TOUSE, "particles%d.dat", ThisTask);
    fd = fopen(buffer, "w");
    my_fwrite(&NumPart, 1, sizeof(int), fd);
    for(i = 0; i < NumPart; i++)
        my_fwrite(&P[i].Pos[0], 3, sizeof(MyFloat), fd);
    for(i = 0; i < NumPart; i++)
        my_fwrite(&P[i].Vel[0], 3, sizeof(MyFloat), fd);
    for(i = 0; i < NumPart; i++)
        my_fwrite(&P[i].ID, 1, sizeof(int), fd);
    fclose(fd);
}



#ifdef BOX_PERIODIC

/*! This function initializes tables with the correction force and the
 *  correction potential due to the periodic images of a point mass located
 *  at the origin. These corrections are obtained by Ewald summation. (See
 *  Hernquist, Bouchet, Suto, ApJS, 1991, 75, 231) The correction fields
 *  are used to obtain the full periodic force if periodic boundaries
 *  combined with the pure tree algorithm are used. For the TreePM
 *  algorithm, the Ewald correction is not used.
 *
 *  The correction fields are stored on disk once they are computed. If a
 *  corresponding file is found, they are loaded from disk to speed up the
 *  initialization.  The Ewald summation is done in parallel, i.e. the
 *  processors share the work to compute the tables if needed.
 */
void ewald_init(void)
{
#ifndef SELFGRAVITY_OFF
    int i, j, k, beg, len, size, n, task, count;
    double x[3], force[3];
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    FILE *fd;
    
    if(ThisTask == 0) {printf("Initializing Ewald correction...\n");}
    
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "ewald_spc_table_%d_dbl.dat", EN);
    if((fd = fopen(buf, "r")))
    {
        my_fread(&fcorrx[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        my_fread(&fcorry[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        my_fread(&fcorrz[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        my_fread(&potcorr[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
        fclose(fd);
    }
    else
    {
        if(ThisTask == 0) {printf("\nNo Ewald tables in file `%s' found.\nRecomputing them...\n", buf);}
        
        /* ok, let's recompute things. Actually, we do that in parallel. */
        
        size = (EN + 1) * (EN + 1) * (EN + 1) / NTask;
        beg = ThisTask * size;
        len = size;
        if(ThisTask == (NTask - 1))
            len = (EN + 1) * (EN + 1) * (EN + 1) - beg;
        for(i = 0, count = 0; i <= EN; i++)
            for(j = 0; j <= EN; j++)
                for(k = 0; k <= EN; k++)
                {
                    n = (i * (EN + 1) + j) * (EN + 1) + k;
                    if(n >= beg && n < (beg + len))
                    {
                        if((count % (len / 20)) == 0) {PRINT_STATUS("%4.1f percent done", count / (len / 100.0));}
                        x[0] = 0.5 * ((double) i) / EN;
                        x[1] = 0.5 * ((double) j) / EN;
                        x[2] = 0.5 * ((double) k) / EN;
                        ewald_force(i, j, k, x, force);
                        fcorrx[i][j][k] = force[0];
                        fcorry[i][j][k] = force[1];
                        fcorrz[i][j][k] = force[2];
                        if(i + j + k == 0)
                            potcorr[i][j][k] = 2.8372975;
                        else
                            potcorr[i][j][k] = ewald_psi(x);
                        count++;
                    }
                }
        
        for(task = 0; task < NTask; task++)
        {
            beg = task * size;
            len = size;
            if(task == (NTask - 1))
                len = (EN + 1) * (EN + 1) * (EN + 1) - beg;
            MPI_Bcast(&fcorrx[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
            MPI_Bcast(&fcorry[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
            MPI_Bcast(&fcorrz[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
            MPI_Bcast(&potcorr[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MPI_COMM_WORLD);
        }
        
        if(ThisTask == 0)
        {
            printf("\nwriting Ewald tables to file `%s'\n", buf);
            if((fd = fopen(buf, "w")))
            {
                my_fwrite(&fcorrx[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&fcorry[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&fcorrz[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&potcorr[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                fclose(fd);
            }
        }
    }
    
    fac_intp = 2 * EN / All.BoxSize;
    for(i = 0; i <= EN; i++)
        for(j = 0; j <= EN; j++)
            for(k = 0; k <= EN; k++)
            {
                potcorr[i][j][k] /= All.BoxSize;
                fcorrx[i][j][k] /= All.BoxSize * All.BoxSize;
                fcorry[i][j][k] /= All.BoxSize * All.BoxSize;
                fcorrz[i][j][k] /= All.BoxSize * All.BoxSize;
            }
    
    if(ThisTask == 0) {printf(" ..initialization of periodic boundaries finished.\n");}
#endif // #ifndef SELFGRAVITY_OFF
}


/*! This function looks up the correction potential due to the infinite
 *  number of periodic particle/node images. We here use tri-linear
 *  interpolation to get it from the precomputed table, which contains
 *  one octant around the target particle at the origin. The other
 *  octants are obtained from it by exploiting symmetry properties.
 */
double ewald_pot_corr(double dx, double dy, double dz)
{
    /* trilinear interp of the potential octant table via the shared SSOT helper (gravtree_ewald.h) */
    grav_ewald_interp_weights w = grav_ewald_interp_setup(dx, dy, dz, fac_intp);
    return grav_ewald_interp_apply(&potcorr[0][0][0], w);
}



/*! This function computes the potential correction term by means of Ewald
 *  summation.
 */
double ewald_psi(double x[3])
{
    double alpha, psi;
    double r, sum1, sum2, hdotx;
    double dx[3];
    int i, n[3], h[3], h2;
    
    alpha = 2.0;
    for(n[0] = -4, sum1 = 0; n[0] <= 4; n[0]++)
        for(n[1] = -4; n[1] <= 4; n[1]++)
            for(n[2] = -4; n[2] <= 4; n[2]++)
            {
                for(i = 0; i < 3; i++)
                    dx[i] = x[i] - n[i];
                r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                sum1 += erfc(alpha * r) / r;
            }
    
    for(h[0] = -4, sum2 = 0; h[0] <= 4; h[0]++)
        for(h[1] = -4; h[1] <= 4; h[1]++)
            for(h[2] = -4; h[2] <= 4; h[2]++)
            {
                hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
                h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
                if(h2 > 0)
                    sum2 += 1 / (M_PI * h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * cos(2 * M_PI * hdotx);
            }
    
    r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    psi = M_PI / (alpha * alpha) - sum1 - sum2 + 1 / r;
    return psi;
}


/*! This function computes the force correction term (difference between full
 *  force of infinite lattice and nearest image) by Ewald summation.
 */
void ewald_force(int iii, int jjj, int kkk, double x[3], double force[3])
{
    double alpha, r2;
    double r, val, hdotx, dx[3];
    int i, h[3], n[3], h2;
    
    alpha = 2.0;
    for(i = 0; i < 3; i++)
        force[i] = 0;
    if(iii == 0 && jjj == 0 && kkk == 0)
        return;
    r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    for(i = 0; i < 3; i++)
        force[i] += x[i] / (r2 * sqrt(r2));
    for(n[0] = -4; n[0] <= 4; n[0]++)
        for(n[1] = -4; n[1] <= 4; n[1]++)
            for(n[2] = -4; n[2] <= 4; n[2]++)
            {
                for(i = 0; i < 3; i++)
                    dx[i] = x[i] - n[i];
                r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                val = erfc(alpha * r) + 2 * alpha * r / sqrt(M_PI) * exp(-alpha * alpha * r * r);
                for(i = 0; i < 3; i++)
                    force[i] -= dx[i] / (r * r * r) * val;
            }
    
    for(h[0] = -4; h[0] <= 4; h[0]++)
        for(h[1] = -4; h[1] <= 4; h[1]++)
            for(h[2] = -4; h[2] <= 4; h[2]++)
            {
                hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
                h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
                if(h2 > 0)
                {
                    val = 2.0 / ((double) h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * sin(2 * M_PI * hdotx);
                    for(i = 0; i < 3; i++)
                        force[i] -= h[i] * val;
                }
            }
}
#endif // #ifdef BOX_PERIODIC //


/*! Refresh tree node moments without rebuilding the tree structure. Uses bottom-up accumulation
 *  via Father[] pointers instead of u.suns[] (which are destroyed after the initial tree build
 *  since they share a union with u.d). Nodes are processed from high to low index, which gives
 *  bottom-up order since children are always allocated with higher indices than parents.
 *  Use this when particle properties (mass, type, luminosity) have changed but particles haven't
 *  moved, e.g. after star formation or sink SN events. */
void force_refresh_node_moments(void)
{
    int i, k, no;
    PRINT_STATUS("Refreshing tree node moments (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));

    /* Phase 6.2: GPU moment-refresh kernel computes local-tree node
     * moments + writes back to AoS. After this returns, Nodes[] /
     * Extnodes[] are in the same state CPU steps 1-4 below would
     * produce, so the CPU pseudo-particle path can run unchanged. */
    {
        /* Reset GravCost/Ti_current/Flag/Ti_lastkicked/dp/dp_dm/dp_stellarlum
         * fields that the GPU kernel does not own. These mirror the
         * non-moment lines in CPU step 1 (forcetree.cc:3837..3848). */
        for(no = All.MaxPart; no < All.MaxPart + Numnodestree; no++) {
            Nodes[no].GravCost = 0;
            Nodes[no].Ti_current = All.Ti_Current;
            Extnodes[no].dp = {};
            Extnodes[no].Ti_lastkicked = All.Ti_Current;
            Extnodes[no].Flag = GlobFlag;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            Extnodes[no].rt_source_lum_dp = {};
#endif
#ifdef DM_SCALARFIELD_SCREENING
            Extnodes[no].dp_dm = {};
#endif
        }
        /* Rank-local GPU refresh steps: on failure set a soft bad-stop and
         * fall through force_exchange_pseudodata (matched, topology-driven);
         * the gravtree:after_refresh_moments poll drains before the walk. */
        if(gpu_moment_refresh(-1) != 0)          {endrun(90000086);}
        /* Mode B: re-seed per-type bands; gpu_moment_refresh wrote scalar
         * hmax to AoS but not per-type. Without this, hmax_per_type[] are
         * left at zero by the GPU bypass and Mode B's SYMMETRIC walker
         * over-prunes (oracle mismatches). */
        force_refresh_hmax_per_type_host(Numnodestree);
        if(gpu_force_flag_localnodes() != 0)     {endrun(90000087);}
        int pseudo_status = force_exchange_pseudodata();
        /* skip dependent pseudo-update on an unmatched complete (soft bad-stop set);
         * drains at gravtree:after_refresh_moments. */
        if(!pseudo_status) {
            if(gpu_scatter_pseudo_to_soa() != 0)     {endrun(90000088);}
            if(gpu_topnode_moment_resum() != 0)      {endrun(90000089);}
        }
        PRINT_STATUS(" ..tree node moments refreshed (GPU).");
        return;
    }
}
