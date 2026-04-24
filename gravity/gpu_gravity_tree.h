/* gpu_gravity_tree.h — Step 13 Phase 3
 *
 * GPU-resident SoA mirror of the gravity tree (NODE / extNODE arrays).
 *
 * Build still happens on CPU (force_treebuild → Nodes_base AoS); this layer
 * provides the SoA mirror so a GPU walk kernel (Phase 4) can do coalesced
 * reads. Retired in Phase 6 once the tree builds directly on the GPU.
 *
 * Lifetime: acquire() takes Nodes_host + Extnodes_host pointers and a
 * capacity. If the SoA already mirrors the latest CPU tree, returns the
 * cached pointers without re-copying. Otherwise reseeds. invalidate() marks
 * the mirror stale (call after force_treebuild, force_update_node_recursive,
 * domain decomp). release() frees the SharedSpace storage.
 *
 * Fields mirrored: the subset the walk reads. center/len for opening, s/mass
 * for force, sibling/nextnode for traversal, bitflags for opening type,
 * maxsoft for adaptive softening. Optional payloads (RT_USE_GRAVTREE,
 * SINK_*, tidal tensor, DM_SCALARFIELD_SCREENING) are NOT mirrored yet —
 * Phase 4 will extend per-payload as the walk needs them, gated by the same
 * #ifdefs as the AoS NODE definition.
 *
 * Memory model: GIZMO_KOKKOS_SHARED_SPACE (UVM) so host writes during
 * acquire are visible to device kernels without explicit deep_copy. Per-field
 * SoA arrays (Vec3 fields kept as Vec3 arrays for now; can split into x/y/z
 * if profiling shows coalescing benefit).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GIZMO_GPU_GRAVITY_TREE_H
#define GIZMO_GPU_GRAVITY_TREE_H

struct NODE;
struct extNODE;

#ifdef OPENMP_GPU_OFFLOAD

#ifdef __cplusplus
extern "C" {
#endif

/* SoA view exposed to GPU kernels. All pointers live in SharedSpace; indices
 * match the AoS Nodes[] convention (callers index by [no - All.MaxPart] when
 * using the base array, or by absolute Nodes[] index after applying the
 * NTopnodes offset — Phase 4 will pick a convention and lock it in). */
struct gpu_gravity_tree_soa_t {
    /* Geometric / opening criterion */
    Vec3<MyFloat>  *center;     /* geometric center of node */
    MyFloat        *len;        /* sidelength */
    /* Multipole (currently monopole) */
    Vec3<MyFloat>  *s;          /* center of mass */
    MyFloat        *mass;       /* total mass */
    /* Walk traversal */
    int            *sibling;
    int            *nextnode;
    unsigned int   *bitflags;
    /* Force kernel */
    MyFloat        *maxsoft;
    long           *N_part;
    int             nnodes;     /* number of valid entries */
    /* Nextnode[] mirror — used for particle-level traversal: when the walk
     * lands on `no < MaxPart`, advance via nextnode_aux[no]. Sized for
     * MaxPart + NTopnodes so the pseudo-particle region is addressable
     * (though Tier-1 GPU walk aborts on pseudo-particle rather than
     * following it). */
    int            *nextnode_aux;
    int             nextnode_aux_size;

    /* --- Phase 2-I optional payloads: gated by the same flags as the AoS
     *     NODE definition in allvars.h. Each block is present iff the host
     *     NODE carries the field; downstream Phase 2-A/B/C/D walk kernels
     *     consume these. Fields absent from the SoA under current Tier 1c
     *     configs (evrard_forall) → bitwise unchanged. */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    MyFloat        *gasmass;          /* [nnodes] */
#endif
#ifdef RT_USE_GRAVTREE
    MyFloat        *stellar_lum;      /* flat [nnodes * N_RT_FREQ_BINS] */
#ifdef CHIMES_STELLAR_FLUXES
    double         *chimes_stellar_lum_G0;  /* flat [nnodes * CHIMES_LOCAL_UV_NBINS] */
    double         *chimes_stellar_lum_ion; /* flat [nnodes * CHIMES_LOCAL_UV_NBINS] */
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyFloat>  *rt_source_lum_s;  /* from NODE */
    Vec3<MyFloat>  *rt_source_lum_vs; /* from extNODE */
#endif
#ifdef SINK_PHOTONMOMENTUM
    MyFloat        *sink_lum;
    Vec3<MyFloat>  *sink_lum_grad;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat        *cr_injection;
#endif
#ifdef SINK_CALC_DISTANCES
    MyFloat        *sink_mass;
    Vec3<MyFloat>  *sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
    Vec3<MyFloat>  *sink_vel;
#endif
#if defined(SPECIAL_POINT_MOTION)
    Vec3<MyFloat>  *sink_acc;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    int            *N_SINK;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    MyFloat        *MaxFeedbackVel;
#endif
#endif
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
    Vec3<MyFloat>  *node_vs;          /* mirror of Extnodes[].vs for dyn-friction */
#endif
};

/* Acquire SoA mirror sized for at least min_nodes. Reuses existing storage
 * when capacity suffices, and does a per-node partial reseed for any node
 * marked dirty since the last acquire. Nothing is copied for clean nodes.
 * On first allocation or capacity growth, all nodes are marked dirty and
 * the full [0..min_nodes) range is seeded.
 *
 * Nodes_host and Extnodes_host must be the *base* pointers (= Nodes_base /
 * Extnodes_base); dirty indices [0..min_nodes) are read. Pass NULL for the
 * host pointers only on first allocation if you want to defer the copy. */
void gpu_gravity_tree_acquire(int min_nodes,
                              struct NODE    *Nodes_host,
                              struct extNODE *Extnodes_host);

/* Mirror Nextnode[0..n) into the SoA's nextnode_aux field. Callers must call
 * acquire() before invoking this, so the node arrays exist. Separate from
 * acquire() because Nextnode has different size and lifetime semantics. */
void gpu_gravity_tree_set_nextnode(int n, int *Nextnode_host);

/* Mark a single node dirty (absolute Nodes[] index, i.e. >= All.MaxPart).
 * Next acquire() will re-copy just this node's fields. Called from the
 * pre-walk drift loop in gpu_gravtree.cc for each node whose Ti_current
 * advanced (force_drift_node mutated s/len/vs/hmax/etc). No-op if the
 * index is out of range or the SoA has not yet been allocated. */
void gpu_gravity_tree_mark_dirty(int no);

/* Mark every node dirty. Called after force_treebuild (topology rebuild
 * invalidates everything) and force_refresh_node_moments (all moments
 * zeroed + re-accumulated). */
void gpu_gravity_tree_mark_all_dirty(void);

/* Back-compat alias — identical to mark_all_dirty. Retained for callers
 * that haven't been updated. Will be removed in Phase 6.8 once the Phase-3
 * stopgap is retired. */
void gpu_gravity_tree_invalidate(void);

/* Free SharedSpace storage. Called at shutdown (and from force_treefree if
 * the tree is being torn down without a follow-up rebuild — but typically
 * force_treefree just invalidates and lets the next acquire reuse). */
void gpu_gravity_tree_release(void);

/* Accessors. Returns NULL pointers / 0 capacity when not held or stale. */
struct gpu_gravity_tree_soa_t *gpu_gravity_tree_soa(void);
int gpu_gravity_tree_capacity(void);
int gpu_gravity_tree_valid(void);

/* Diagnostic — returns the count of dirty nodes queued for the next acquire
 * reseed. Used by 6.0 baseline benchmarks to measure reseed volume per call. */
int gpu_gravity_tree_dirty_count(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_GPU_OFFLOAD */

#endif /* GIZMO_GPU_GRAVITY_TREE_H */
