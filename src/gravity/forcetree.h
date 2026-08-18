#ifndef FORCETREE_H
#define FORCETREE_H

#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif

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


#define BITFLAG_TOPLEVEL                   0
#define BITFLAG_DEPENDS_ON_LOCAL_ELEMENT   1
#define BITFLAG_INTERNAL_TOPLEVEL          6
#define BITFLAG_MULTIPLEPARTICLES          7
#define BITFLAG_NODEHASBEENKICKED          8
#define BITFLAG_INSIDE_LINKINGLENGTH       9

void force_update_tree(void);
void force_refresh_node_moments(void);

void force_flag_localnodes(void);

void *gravity_primary_loop(void *p);

int force_treeevaluate(int target, int *exportflag, int *exportnodecount, int *exportindex);
int force_treeevaluate_ewald_correction(int target, int *exportflag, int *exportnodecount, int *exportindex);
void force_drift_node(int no, integertime time1);

/*! Single home for the host-vs-device routing decision of the gravity walk and the
 *  dynamic tree update. Two independent reasons to answer yes, and BOTH are part of the
 *  contract:
 *    - this rank has fewer than All.GravityHostWalkBelowActive active candidates, so the
 *      host walk (which drifts nodes lazily as it opens them) is cheaper than the device
 *      walk plus the all-node drift it requires;
 *    - a host lazy drift has ALREADY happened at the current time, in which case the host
 *      keeps ownership for the rest of the time step whatever the count says. The device
 *      sweep skips nodes already at its target time and so cannot refresh their mirror,
 *      so once any node is drifted lazily, no device walk may run at that time again.
 *  The second clause is what keeps a repeated same-time evaluation (a Hermite correction
 *  pass, the opening-criterion re-walk) from reading stale node geometry. */
int gravity_walk_route_to_host(long long n_local_active);

/*! Time at which a host lazy node drift was last actually performed on this rank
 *  (-1 = never). The device node-drift sweep skips nodes already at its target time, so
 *  it must never run at a time the host has already drifted to: it would leave those
 *  nodes' SoA mirror holding pre-drift geometry. Read by gpu_force_drift_nodes as an
 *  invariant check, not as a control input. */
integertime force_host_lazy_drift_ti(void);
void force_tree_discardpartials(void);
void force_treeupdate_pseudos(int);
void force_update_pseudoparticles(void);
void force_kick_node(int i, Vec3<MyDouble>& dv);
void force_dynamic_update(void);
void force_dynamic_update_node(int no, int mode, MyFloat *minbound, MyFloat *maxbound);
void force_update_hmax(void);
void force_update_hmax_of_node(int no, int mode);
void force_finish_kick_nodes(void);
int force_create_empty_nodes(int no, int topnode, int bits, peano1D x, peano1D y, peano1D z, int *nodecount, int *nextfree);
int  force_exchange_pseudodata(void);          /* returns complete() status: nonzero = unmatched (caller skips dependent pseudo-update) */
void force_exchange_pseudodata_issue(void);    /* split for non-blocking overlap with LET */
int  force_exchange_pseudodata_complete(void); /* pair to _issue; nonzero = unmatched (pending==NULL) */
void force_insert_pseudo_particles(void);
void force_add_element_to_tree(int igas, int istar);

void   force_costevaluate(void);
int    force_getcost_single(void);
int    force_getcost_quadru(void);
void   force_resetcost(void);
void   force_setupnonrecursive(int no);
/* foreign_node_slots_exact: LET foreign-node CAPACITY to allocate verbatim; negative derives it,
 * which is what every normal caller wants.  Only the restart read passes a value: the node pointers
 * it is about to deserialize encode the writer's capacity (pseudo-particles start at
 * TreeNodeIndexBase + MaxNodes + MaxForeignNodes), so the reader has to reproduce it exactly. */
void   force_treeallocate(int maxnodes, int tree_particle_slots, int foreign_node_slots_exact = -1);
/* Conservative per-particle radius used to seed Extnodes[no].hmax_per_type[Type]
 * bands. Mode B SYMMETRIC tree-prune reads these bands as an upper bound on
 * any leaf-policy-selectable reach for that type, so the band must dominate
 * every per-particle radius source a Spec's radius_policy can pick at the
 * leaf (P[i].KernelRadius, P[i].AGS_KernelRadius when defined, P[i].ForceSoftening).
 * Leaf-level Mode B predicate / Mode A compact_xyzh still apply the exact policy;
 * over-opening here is correct (extra candidates filter at leaf). Capped at
 * All.MaxKernelRadius to match the legacy band semantics. */
double force_hmax_per_type_particle_radius(int i);

/* Monotonic gravity-tree freshness generations.  treebuild_generation bumps on
 * every successful force_treebuild (topology + Father[] + node structure changed);
 * hmax_refresh_generation bumps at the end of force_update_hmax (ancestor node
 * boxes re-drifted + per-type bands re-seeded after density).  Consumers that
 * cache anything derived from the tree geometry (e.g. the ghost-route fine band)
 * key on BOTH plus All.Ti_Current; a mismatch means rebuild / fail-closed.  These
 * are NOT a substitute for the per-data epoch keys, only the tree-side half. */
long   force_treebuild_generation(void);
long   force_hmax_refresh_generation(void);
void   force_bump_hmax_refresh_generation(void);   /* called by force_update_hmax (separate TU) */

int    force_treebuild(int npart, struct unbind_data *mp);
int    force_treebuild_single(int npart, struct unbind_data *mp);
int    force_treeevaluate_direct(int target, int mode);
void   force_treefree(void);
int    force_tree_is_allocated(void);   /* nonzero while tree storage is held */

/*! Give the foreign-node range storage for the `foreign_needed` nodes this rank is about to
 *  receive, once the LET exchange has counted them.  force_treeallocate leaves that storage
 *  at zero because the count is not knowable when the tree is allocated.  Call once per tree
 *  build, before the first foreign node is installed; a rank importing nothing passes 0 and
 *  does nothing.  Leaves the tree untouched and returns nonzero if the memory is not there. */
int    force_tree_grow_foreign_storage(long long foreign_needed);

/*! Can a particle created at this index be carried by the tree that is standing right now?
 *  Father[] and Nextnode[]'s particle segment span All.TreeParticleSlots entries and are NOT resized
 *  between rebuilds, so a slot being available in P[] does not by itself mean the live tree can index
 *  it: inserting past those arrays would write out of bounds, and every later pass that reads a
 *  particle's parent would read out of bounds too.  Creation sites ask this in addition to their
 *  existing capacity checks, and decline the way they already decline when storage is short.
 *  A tree built by force_treeallocate sizes these arrays from All.MaxPartExpandable, the run's
 *  ceiling on the particle capacity, so the answer is currently yes for every index P[] can hold and
 *  this costs one comparison.  It is kept rather than deleted because it states the requirement at
 *  the sites that depend on it: if the ceiling ever became something a capacity could reach, these
 *  are exactly the places that must decline.  With no tree standing the answer is also yes, because
 *  there is nothing to outgrow -- not a licence to insert into a tree that is not there, which
 *  force_add_element_to_tree refuses on its own. */
static inline int gizmo_particle_index_fits_live_tree(int index)
{
    return (!force_tree_is_allocated()) || (index < All.TreeParticleSlots);
}
void   force_update_node(int no, int flag);
void   force_update_size_of_parent_node(int no);

void   dump_particles(void);

/* mesh/ngb.cc retired in Step 5 Phase D5: ngb_treebuild/ngb_treefind_* all dead on the Kokkos path.
   ngb_treebuild() callers replaced with force_treebuild(NumPart, NULL) directly. */

#ifdef BOX_PERIODIC
/* Ewald octant table size. Declared under BOX_PERIODIC to match `EN` in forcetree.cc: the tables
 * + the CPU Ewald functions + the shared interp helper (gravtree_ewald.h) all compile under
 * BOX_PERIODIC (even under GRAVITY_NOT_PERIODIC, where the correction is compiled-but-dead). */
#define GIZMO_EWALD_EN 64
#endif
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
/* Ewald correction table accessor. Returns flat pointers of length (EN+1)^3
 * to the four static look-up tables (fcorrx/y/z/potcorr) inside forcetree.cc,
 * plus fac_intp (= 2*EN/All.BoxSize). Used by gpu_gravtree.cc to mirror the
 * tables into SharedSpace once after ewald_init(). */
void gizmo_get_ewald_tables(const MyFloat **fcorrx_out,
                            const MyFloat **fcorry_out,
                            const MyFloat **fcorrz_out,
                            const MyFloat **potcorr_out,
                            double *fac_intp_out);
#endif

#endif



