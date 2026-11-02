/* gpu_topology_build.cc
 *
 * GPU tree-build data path: per-particle Peano-walk to topleaf, 128-bit
 * Morton key, parallel histogram + scan + scatter to bucket particles
 * by topleaf, per-topleaf Morton sort.  See gpu_topology_build.h for the
 * API.
 *
 * Topology emission, collocation handling, and overflow retry are
 * handled elsewhere.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_morton.h"
#include "gpu_morton_functions.h"
#include "gpu_peano_walk.h"
#include "gpu_peano_walk_functions.h"
#include "gpu_gravity_tree.h"
#include "gpu_topology_build.h"


namespace {

/* SharedSpace scratch -- reused across tree builds, grown on demand. */
static int *g_sorted_idx          = NULL;  /* [npart]              */
static int *g_particle_topleaf    = NULL;  /* [npart]              */
static int *g_topleaf_start       = NULL;  /* [NTopleaves + 1]     */
static int *g_topleaf_count       = NULL;  /* [NTopleaves]         */
static int  g_npart_cap           = 0;
static int  g_topleaf_cap         = 0;

/* Every counter that device threads update with atomics lives in DEVICE memory, never in SharedSpace.
 * A managed page need not be device-resident when a kernel runs -- the host touches these pages
 * between kernels -- and where the platform reports cudaDevAttrHostNativeAtomicSupported == 0 (PCIe
 * H100s among them) device atomics on a page that is not device-resident lose updates.  They do so
 * silently: the bucket scatter handed two particles one slot, leaving one in two top-leaf ranges and
 * one in none (pfhopkins/gizmo#83), and a prefetch to the device is only a hint the driver may undo.
 * Plain loads and stores are unaffected, so arrays that are only written or read stay in SharedSpace.
 * Still on SharedSpace here: error flags (a lost increment only under-counts a reported error) and the
 * node-length maximum in gpu_topology_grow_retained_paths, which is not yet moved. */
static int *g_topleaf_count_dev   = NULL;  /* [NTopleaves] bucket counts, accumulated here, then copied */
static int *g_topleaf_cursor_dev  = NULL;  /* [NTopleaves] scatter cursors */
static int  g_topleaf_dev_cap     = 0;
enum {TOPO_CTR_BFS_A = 0, TOPO_CTR_BFS_B, TOPO_CTR_BFS_NODES, TOPO_CTR_BFS_FAIL, TOPO_CTR_LEAF_CHAIN,
      TOPO_CTR_PREP_CROSSED, TOPO_CTR_PREP_UNRECOVERED, TOPO_CTR_PREP_OUTSIDE, TOPO_NCOUNTERS};
static int *g_counters_dev        = NULL;  /* [TOPO_NCOUNTERS] */

static int *topo_counters_dev(void)
{
    if(!g_counters_dev) {g_counters_dev = (int *) gizmo_gpu_alloc_device(TOPO_NCOUNTERS * sizeof(int), "treescratch_build_counters");}
    return g_counters_dev;
}
static void topo_counters_put(int first, const int *vals, int n)
{
    Kokkos::deep_copy(Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(g_counters_dev + first, n),
                      Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(vals, n));
}
static void topo_counters_get(int first, int *vals, int n)
{
    Kokkos::fence();
    Kokkos::deep_copy(Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(vals, n),
                      Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged> >(g_counters_dev + first, n));
}

/* Subset-build slot->particle map. force_treebuild(npart, mp) may request a
 * tree over an arbitrary subset of P[] (mp[i].index, e.g. SUBFIND per-species
 * density or collective unbinding). The build pipeline is slot-indexed (slot i
 * in 0..npart-1); g_slot_to_particle[slot] gives the real P[] index. Identity
 * (mp==NULL, full-tree builds) leaves g_slot_map_active=0 and the map unused.
 * Scratch-owned + reset per build: set definitively in gpu_topology_build_data_path
 * before it can be consumed, stays valid through gpu_topology_emit_bfs (incl.
 * overflow/retry), freed in gpu_topology_build_release. */
/* One record per tree leaf holding more than one particle, produced while the tree is emitted and
 * consumed by the three passes that finish the build: the members' Father[] entries, the capture of
 * the leaf's traversal successor, and the write of the member chain itself.  Holding the leaf's
 * range rather than its particles keeps this to four integers per leaf, and the head and tail are
 * recovered from the range when they are needed.
 *
 * Scratch, not tree state: it is reset at the start of every emit (a build that is retried must not
 * see the previous attempt's records) and released once the chain has been written, before anything
 * can insert into the tree.  A leaf's membership is therefore never described anywhere that could go
 * stale, which is the same reason no member count is kept on the node.
 *
 * A leaf recorded here holds at least two particles, so there can be no more than half as many
 * records as there are particle slots. */
#if TREE_LEAF_BUCKET_SIZE > 1
struct LeafChainRecord {
    int range_first;   /* first member, as an index into the build's sorted order */
    int count;         /* members, always >= 2 */
    int parent_abs;    /* the node they hang from, as an absolute tree index */
    int successor;     /* where the walk goes after this leaf; filled in after threading */
};
static struct LeafChainRecord *g_leaf_chain = NULL;
static int   g_leaf_chain_count = 0;    /* records written by the last emit; counted on the device (TOPO_CTR_LEAF_CHAIN) */
static int   g_leaf_chain_cap = 0;
#endif

#if TREE_LEAF_BUCKET_SIZE > 1
static inline bool leaf_members_run_flat(void)
{
    /* Below this many members a team per leaf costs more in idle lanes than the division saves. */
    const int lanes_worth_dividing = 16;
    return gizmo_gpu_default_space_is_host() || (TREE_LEAF_BUCKET_SIZE < lanes_worth_dividing);
}
#endif

static int *g_slot_to_particle    = NULL;  /* [npart]; real index per build slot */
static int  g_slot_cap            = 0;     /* own capacity: lazily grown only for subset builds */
static int  g_slot_map_active     = 0;     /* 1 iff this build is a non-identity subset */

/* Retained attachment.  A particle whose CURRENT geometric topleaf belongs to another rank cannot be
 * bucketed there: the pseudo-particle exchange overwrites that node afterwards, and the subtree
 * holding the particle is left unreachable from the root, so its mass enters no rank's multipole
 * moments.  Such a particle is instead kept under the topleaf it hung from in the standing tree,
 * which this rank does own.  These record which slots needed that, so the two later stages -- clamping
 * the key into the retained leaf, and growing that leaf's path to cover the true position -- touch
 * only those particles rather than rescanning every particle. */
static int *g_retained_slots      = NULL;  /* [g_retained_n] re-attached particle slots */
static int  g_retained_cap        = 0;
static int  g_retained_n          = 0;
/* npart whose keys and leaves are already computed and whose retained attachments are already
 * applied, or -1.  Positions, TopNodes and the retained attachment do not change while a build
 * retries for a larger arena, so the work is done once and every attempt reuses it. */
static int  g_prepared_npart      = -1;
#if TREE_LEAF_BUCKET_SIZE > 1
/* How many particles the sorted order above currently describes.  A build may cover an arbitrary
 * subset of the particles, so this is not the tree's capacity and not the scratch's capacity: it is
 * the extent that leaf records are checked against.  Only leaves holding several particles are
 * described by range, so this is not kept when they cannot occur. */
static int  g_sorted_npart        = 0;
#endif

/* Allocate/grow a SharedSpace int buffer. `label` is a stable string literal
   (the memory ledger classifies allocations by label; a stack buffer must not be
   relied on to survive into the free callback). */
static int *grow_int_buffer(int *buf, int old_cap, int new_cap, const char *label) {
    if(old_cap >= new_cap) {return buf;}
    if(buf) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(buf);}
    buf = (int *) gizmo_gpu_alloc_shared((long)new_cap * sizeof(int), label);
    if(!buf) {
        printf("gpu_topology_build: %s alloc failed (n=%d)\n", label, new_cap);
    }
    return buf;
}

/* The one place a position becomes a (Peano, Morton) key pair.  Used by the bucketing kernel, by the
 * retained-attachment recovery, and by the clamp, so the three cannot drift apart. */
KOKKOS_INLINE_FUNCTION peanokey topo_key_of_position_(double x, double y, double z,
                                                      double dc0, double dc1, double dc2,
                                                      double dlen, int bits, Morton128 *morton_out)
{
    double fx = (dlen > 0.0) ? ((x - dc0) / dlen) : 0.0;
    double fy = (dlen > 0.0) ? ((y - dc1) / dlen) : 0.0;
    double fz = (dlen > 0.0) ? ((z - dc2) / dlen) : 0.0;
    /* Clamp the sum, not the fraction: the key is the mantissa of (frac + 1.0), and a fraction just
     * under 1.0 can carry that sum up to 2.0, whose mantissa is zero -- the first cell instead of the
     * last.  See gpu_morton.cc. */
    double sx = fx + 1.0, sy = fy + 1.0, sz = fz + 1.0;
    if(!(sx >= 1.0)) {sx = 1.0;} if(!(sx < 2.0)) {sx = 0x1.fffffffffffffp0;}
    if(!(sy >= 1.0)) {sy = 1.0;} if(!(sy < 2.0)) {sy = 0x1.fffffffffffffp0;}
    if(!(sz >= 1.0)) {sz = 1.0;} if(!(sz < 2.0)) {sz = 0x1.fffffffffffffp0;}
    Morton128 m;
    peanokey pkey = gpu_peano_and_morton_key(gpu_morton_double_to_int42(sx),
                                             gpu_morton_double_to_int42(sy),
                                             gpu_morton_double_to_int42(sz), bits, &m);
    *morton_out = m;
    return pkey;
}

/* Everything the bucketing and the pre-build retained-attachment stage both need: the device mirrors,
 * the particle arena, the Morton key buffer, and the scratch sized to this build.  Factored out so the
 * two entry points acquire identically rather than by two copies of the same sequence. */
struct topo_build_ctx {
    Morton128                 *keys;
    struct particle_data      *P_dev;
    const struct topnode_data *tn;
    const int                 *dni;
};

static int topo_acquire_(int npart, const struct unbind_data *mp, const char *site, struct topo_build_ctx *ctx)
{
    /* Reset the subset map state first: any early-return failure below then
     * leaves an identity (safe) build, never a stale subset map from a prior
     * build that gpu_topology_emit_bfs could consume. */
    g_slot_map_active = 0;
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Acquire dependencies. */
    int rc = gpu_peano_walk_acquire();
    if(rc) {printf("gpu_topology_build: gpu_peano_walk_acquire failed\n"); return rc;}
    ctx->keys = gpu_morton_keys_acquire(npart);
    if(!ctx->keys) {return 1;}

    if((long long) npart * 8 >= (long long) NumPart) {gizmo_particles_prefetch_P_to_device(site);}   /* a build over most of the particles: move them in bulk (it reads no cell data) */
    gpu_particles_arena_set_site(site);
    gpu_particles_arena_acquire(NumPart, P, CellP);
    ctx->P_dev = gpu_particles_arena_P();
    if(!ctx->P_dev) {printf("gpu_topology_build: P_dev null\n"); return 1;}

    ctx->tn  = gpu_peano_walk_topnodes();
    ctx->dni = gpu_peano_walk_domain_node_index();
    if(!ctx->tn || !ctx->dni) {printf("gpu_topology_build: peano-walk mirrors null\n"); return 1;}

    /* Grow per-particle scratch. */
    if(g_npart_cap < npart) {
        g_sorted_idx       = grow_int_buffer(g_sorted_idx,       g_npart_cap, npart, "treescratch_build_sorted_idx");
        g_particle_topleaf = grow_int_buffer(g_particle_topleaf, g_npart_cap, npart, "treescratch_build_particle_topleaf");
        if(!g_sorted_idx || !g_particle_topleaf) {g_npart_cap = 0; return 1;}
        g_npart_cap = npart;
    }
#if TREE_LEAF_BUCKET_SIZE > 1
    /* The particle count THIS build sorts.  Set here, where the sorted order is established, so it
     * is right for a subset build too: the retained-attachment prepass runs only for a whole-tree
     * build, so a bound taken from there is absent exactly when a group or subset tree is built. */
    g_sorted_npart = npart;
#endif

    /* Subset build ONLY: lazily allocate (own capacity) + stage the real-particle
     * index per slot. Identity/full builds (mp==NULL) keep g_slot_to_particle NULL
     * and g_slot_map_active=0 (set at function entry) -> slot==real, zero extra cost.
     * SharedSpace is host-writable; consumed on device by Kernel 1 + emit_bfs. */
    if(mp) {
        if(g_slot_cap < npart) {
            g_slot_to_particle = grow_int_buffer(g_slot_to_particle, g_slot_cap, npart, "treescratch_build_slot_to_particle");
            if(!g_slot_to_particle) {g_slot_cap = 0; return 1;}
            g_slot_cap = npart;
        }
        for(int s = 0; s < npart; s++) {g_slot_to_particle[s] = mp[s].index;}
        g_slot_map_active = 1;
    }
    /* Grow per-topleaf scratch. */
    if(g_topleaf_cap < NTopleaves + 1) {
        int newcap = NTopleaves + 1;
        g_topleaf_start  = grow_int_buffer(g_topleaf_start,  g_topleaf_cap, newcap, "treescratch_build_topleaf_start");
        g_topleaf_count  = grow_int_buffer(g_topleaf_count,  g_topleaf_cap, newcap, "treescratch_build_topleaf_count");
        if(!g_topleaf_start || !g_topleaf_count) {g_topleaf_cap = 0; return 1;}
        g_topleaf_cap = newcap;
    }
    if(g_topleaf_dev_cap < NTopleaves + 1) {
        const int newcap = NTopleaves + 1;
        gizmo_gpu_free_device(g_topleaf_count_dev);
        gizmo_gpu_free_device(g_topleaf_cursor_dev);
        g_topleaf_count_dev  = (int *) gizmo_gpu_alloc_device((size_t) newcap * sizeof(int), "treescratch_build_topleaf_count_dev");
        g_topleaf_cursor_dev = (int *) gizmo_gpu_alloc_device((size_t) newcap * sizeof(int), "treescratch_build_topleaf_cursor_dev");
        if(!g_topleaf_count_dev || !g_topleaf_cursor_dev) {printf("gpu_topology_build: device bucket counters alloc failed (n=%d)\n", newcap); g_topleaf_dev_cap = 0; return 1;}
        g_topleaf_dev_cap = newcap;
    }
    if(!topo_counters_dev()) {printf("gpu_topology_build: device counter block alloc failed\n"); return 1;}
    return 0;
}

/* The original per-range sort: one Morton sort per top-leaf range, each a gather, a merge sort and a scatter
 * with its own allocations and fences (~9 launches and ~6 synchronizations per range, ~170 ranges per rank). */
static int topo_sort_ranges_one_by_one(int ntl)
{
    for(int t = 0; t < ntl; t++) {
        int start = g_topleaf_start[t];
        int count = g_topleaf_count[t];
        if(count > 1) {
            int rc2 = gpu_morton_sort_indices(count, g_sorted_idx + start);
            if(rc2) {return rc2;}
        }
    }
    return 0;
}

/* GIZMO_K1_SEGSORT: 0 restores the per-range loop (same-binary comparison); default on. */
static int topo_segsort_enabled(void)
{
    static int enabled = -1;
    if(enabled < 0) {
        const char *env = getenv("GIZMO_K1_SEGSORT");
        enabled = (env && env[0]) ? (atoi(env) != 0) : 1;
        if(ThisTask == 0) {printf("Tree build: top-leaf ranges Morton-sorted in one pass: %s.\n", enabled ? "on" : "off"); fflush(stdout);}
    }
    return enabled;
}
/* GIZMO_K1_SORT_VERIFY=1: every build sorts both ways and compares the orders (keeps the one-pass one). */
static int topo_segsort_verify_wanted(void)
{
    static int wanted = -1;
    if(wanted < 0) {const char *env = getenv("GIZMO_K1_SORT_VERIFY"); wanted = (env && env[0]) ? (atoi(env) != 0) : 0;}
    return wanted;
}

/* Sort key of the one-pass sort: the range, then the Morton key, then the position in the bucketed order. */
struct TopoRangeKey {
    Morton128 m;
    int range;
    int pos;
};
struct TopoRangeKeyLess {
    KOKKOS_INLINE_FUNCTION bool operator()(const TopoRangeKey &a, const TopoRangeKey &b) const {
        if(a.range != b.range) {return a.range < b.range;}
        if(a.m < b.m) {return true;}
        if(b.m < a.m) {return false;}
        return a.pos < b.pos;
    }
};
static TopoRangeKey *g_rangesort_keys = NULL;   /* device, [g_rangesort_cap], grow-only */
static int          *g_rangesort_vals = NULL;
static int           g_rangesort_cap  = 0;

/* All top-leaf ranges sorted at once.  The ranges are contiguous and in increasing leaf order, so sorting the
 * whole bucketed order by (range, key) keeps every element inside its range and orders each range by key; the
 * position as the last key makes ties keep their bucketed order, which is what the per-range merge sort (stable)
 * does with them.  The result is therefore the per-range order exactly (GIZMO_K1_SORT_VERIFY checks it).  The
 * range of a position is the last leaf whose start does not exceed it, so empty ranges are skipped.  A position
 * whose index is outside the key array takes key zero, as in gpu_morton_sort_indices. */
static int topo_sort_ranges_one_pass(int ntl, int total)
{
    if(total <= 1 || ntl <= 0) {return 0;}
    if(g_rangesort_cap < total) {
        gizmo_gpu_free_device(g_rangesort_keys);
        gizmo_gpu_free_device(g_rangesort_vals);
        g_rangesort_keys = (TopoRangeKey *) gizmo_gpu_alloc_device((size_t) total * sizeof(TopoRangeKey), "treescratch_build_rangesort_keys");
        g_rangesort_vals = (int *) gizmo_gpu_alloc_device((size_t) total * sizeof(int), "treescratch_build_rangesort_vals");
        if(!g_rangesort_keys || !g_rangesort_vals) {
            printf("gpu_topology_build: range-sort scratch alloc failed (n=%d)\n", total);
            gizmo_gpu_free_device(g_rangesort_keys); gizmo_gpu_free_device(g_rangesort_vals);
            g_rangesort_keys = NULL; g_rangesort_vals = NULL; g_rangesort_cap = 0;
            return 1;
        }
        g_rangesort_cap = total;
    }
    using ExSpace  = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExSpace::memory_space;
    using UM = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<TopoRangeKey*, MemSpace, UM> keys(g_rangesort_keys, total);
    Kokkos::View<int*, MemSpace, UM> vals(g_rangesort_vals, total);
    const Morton128 *mk = gpu_morton_keys();
    const int keys_cap = gpu_morton_keys_capacity();
    const int *tsta = g_topleaf_start;
    int *sidx = g_sorted_idx;
    Kokkos::parallel_for("topo_rangesort_gather", Kokkos::RangePolicy<ExSpace>(0, total), KOKKOS_LAMBDA(int j) {
        int lo = 0, hi = ntl - 1;   /* last t in [0, ntl) with tsta[t] <= j */
        while(lo < hi) {
            const int mid = lo + (hi - lo + 1) / 2;
            if(tsta[mid] <= j) {lo = mid;} else {hi = mid - 1;}
        }
        const int idx = sidx[j];
        TopoRangeKey k;
        if(idx >= 0 && idx < keys_cap) {k.m = mk[idx];} else {k.m.hi = 0; k.m.lo = 0;}
        k.range = lo;
        k.pos = j;
        keys(j) = k;
        vals(j) = idx;
    });
    try {
        Kokkos::Experimental::sort_by_key(ExSpace(), keys, vals, TopoRangeKeyLess{});
    } catch(const std::bad_alloc &) {
        printf("gpu_topology_build: range-sort working space refused (n=%d)\n", total);
        return 1;
    }
    Kokkos::parallel_for("topo_rangesort_scatter", Kokkos::RangePolicy<ExSpace>(0, total), KOKKOS_LAMBDA(int j) {
        sidx[j] = vals(j);
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_rangesort", total);
    return 0;
}

}  /* anonymous namespace */

extern "C" int gpu_topology_build_data_path(int npart, const struct unbind_data *mp)
{
    if(npart <= 0) {g_slot_map_active = 0; return 0;}
    struct topo_build_ctx ctx;
    int rc = topo_acquire_(npart, mp, "gpu_topology_build_data_path", &ctx);
    if(rc) {return rc;}
    Morton128                 *keys  = ctx.keys;
    struct particle_data      *P_dev = ctx.P_dev;
    const struct topnode_data *tn    = ctx.tn;

    int  ntl  = NTopleaves;
    int *pt   = g_particle_topleaf;
    int *tcnt = g_topleaf_count;
    int *dcnt = g_topleaf_count_dev;
    int *tcur = g_topleaf_cursor_dev;
    int *tsta = g_topleaf_start;
    int *sidx = g_sorted_idx;

    /* Capture domain bounds for the encode kernel. */
    const double dc0 = DomainCorner[0];
    const double dc1 = DomainCorner[1];
    const double dc2 = DomainCorner[2];
    const double dlen = DomainLen;
    const int    bits = BITS_PER_DIMENSION;

    /* Kernel 1: per-particle Peano + Morton key compute, TopNodes walk to
     * topleaf id, write Morton key + topleaf id.  Geometry is read from the real
     * particle (slot->particle map for subset builds); keys[]/pt[] stay slot-indexed.
     * Skipped when the pre-build stage already computed these for the same particles:
     * it also applied the retained attachments, which cannot be recomputed here because
     * the standing tree they were recovered from no longer exists. */
    const int *stp = g_slot_map_active ? g_slot_to_particle : NULL;
    if(!(mp == NULL && g_prepared_npart == npart)) {
        Kokkos::parallel_for("topo_keys_and_assign", npart, KOKKOS_LAMBDA(int i) {
            int real = stp ? stp[i] : i;
            Morton128 m;
            peanokey pkey = topo_key_of_position_(P_dev[real].Pos[0], P_dev[real].Pos[1], P_dev[real].Pos[2],
                                                  dc0, dc1, dc2, dlen, bits, &m);
            keys[i] = m;
            pt[i] = gpu_topleaf_for_key(tn, pkey);
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("topo_keys_and_assign", npart);
    }

    /* Retained attachment: a particle kept under the topleaf it hung from in the standing tree takes
     * its key from the point where its true position is clamped into that leaf's cube.  The cube is
     * the nominal one here -- force_create_empty_nodes has just rebuilt the top tree and nothing has
     * grown it yet -- and clamping rather than regenerating keeps the whole Morton/BFS path untouched:
     * the particle sorts and descends exactly as one that really sat at that point would.  The true
     * position is what the moments and the node bounds use; only the key is taken from the clamped
     * point.  Idempotent, so a build that retries for a larger arena simply redoes it. */
    if(mp == NULL && g_prepared_npart == npart && g_retained_n > 0) {
        const int *dni = ctx.dni;
        const int *ret = g_retained_slots;
        const int  nret = g_retained_n;
        struct NODE *Nodes_uvm = Nodes;
        /* How far inside the face the clamped point has to land.  A key keeps the top 42 bits of the
         * mantissa (gpu_morton_double_to_int42), so one cell of key space is DomainLen * 2^-42 of
         * position, and a point merely a rounding step below the upper face still encodes into the
         * NEXT cell -- i.e. into the neighbouring top-leaf.  Two cells of margin puts it clear of that
         * and of the rounding in forming the fraction, while being geometrically nothing: a top-leaf
         * is many orders of magnitude wider than a key cell. */
        const double clamp_backoff = 2.0 * dlen / 4398046511104.0;   /* 2^42 */
        /* Bounds for the leaf's node in THIS tree.  The attachment was checked against the standing
         * tree's DomainNodeIndex before the build; that array has since been refilled by
         * force_create_empty_nodes, and TreeNodeIndexBase and MaxNodes can both have moved with it,
         * so the earlier check says nothing about the index dereferenced here. */
        const int tbase = All.TreeNodeIndexBase, maxn = MaxNodes;
        /* [0] clamped outside their leaf, [1] leaf with no usable node here, [2]/[3] the first such
         * leaf and index, so the report names a case instead of only counting them. */
        int *bad = (int *) gizmo_gpu_alloc_shared(4 * sizeof(int), "treescratch_build_ctr");
        if(!bad) {printf("gpu_topology_build: retained clamp counter alloc failed\n"); return 1;}
        bad[0] = 0; bad[1] = 0; bad[2] = -1; bad[3] = -1;
        Kokkos::parallel_for("topo_retained_clamp", nret, KOKKOS_LAMBDA(int j) {
            int i = ret[j];
            int leaf = pt[i];
            int nd = (leaf >= 0 && leaf < ntl) ? dni[leaf] : -1;
            if(nd < tbase || nd >= tbase + maxn) {
                if(Kokkos::atomic_fetch_add(&bad[1], 1) == 0) {bad[2] = leaf; bad[3] = nd;}
                return;
            }
            Vec3<double> sep = {(double)P_dev[i].Pos[0] - (double)Nodes_uvm[nd].center[0],
                                (double)P_dev[i].Pos[1] - (double)Nodes_uvm[nd].center[1],
                                (double)P_dev[i].Pos[2] - (double)Nodes_uvm[nd].center[2]};
            nearest_xyz(sep, -1);
            double inner = 0.5 * (double)Nodes_uvm[nd].len - clamp_backoff;
            if(!(inner > 0.0)) {inner = 0.0;}   /* a leaf narrower than the margin: the centre is inside */
            /* Clamping each axis on its own sends every particle that lies outside ALL THREE faces
             * to the same interior corner (+-inner, +-inner, +-inner), so an arbitrary number of
             * them collapse onto one point and take an IDENTICAL key.  The range then looks
             * perfectly collocated to the builder when the particles are nowhere near each other:
             * the split finds one occupied child, the node cannot subdivide, and the chain descends
             * a level at a time burning a node per level until the softening floor hands it to the
             * randomized path.  Scaling the whole separation instead keeps the direction, so those
             * particles land on distinct points of the face and sort apart as they should.
             *
             * Two saturated axes collapse the same way whenever the third coordinate is shared --
             * the pair goes to (+-inner, +-inner) and only the unshared axis could have told them
             * apart.  So the projection runs whenever at least two axes saturate, over just those
             * axes: the saturated components are scaled by the largest of themselves, which lands
             * that one on the face and the rest inside, while every unsaturated component is left
             * exactly as it was.  With three saturated axes this is the same arithmetic as before.
             *
             * One saturated axis is NOT covered and cannot be by this means: with a single
             * component to scale, every particle on that side of the face still maps to +-inner,
             * and a population sharing the other two coordinates -- a line -- still collapses.
             * That case needs the other two coordinates to differ, which is what the common
             * crossing has. */
            const double ax = fabs(sep[0]), ay = fabs(sep[1]), az = fabs(sep[2]);
            const int sat0 = (ax > inner), sat1 = (ay > inner), sat2 = (az > inner);
            if(sat0 + sat1 + sat2 >= 2) {
                double mx = 0.0;
                if(sat0 && ax > mx) {mx = ax;}
                if(sat1 && ay > mx) {mx = ay;}
                if(sat2 && az > mx) {mx = az;}
                const double scale = (mx > 0.0) ? (inner / mx) : 0.0;
                if(sat0) {sep[0] *= scale;}
                if(sat1) {sep[1] *= scale;}
                if(sat2) {sep[2] *= scale;}
            } else {
                for(int d = 0; d < 3; d++) {
                    if(sep[d] >  inner) {sep[d] =  inner;}
                    if(sep[d] < -inner) {sep[d] = -inner;}
                }
            }
            Morton128 m;
            peanokey pkey = topo_key_of_position_((double)Nodes_uvm[nd].center[0] + sep[0],
                                                  (double)Nodes_uvm[nd].center[1] + sep[1],
                                                  (double)Nodes_uvm[nd].center[2] + sep[2],
                                                  dc0, dc1, dc2, dlen, bits, &m);
            keys[i] = m;
            if(gpu_topleaf_for_key(tn, pkey) != leaf) {Kokkos::atomic_fetch_add(&bad[0], 1);}
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("topo_retained_clamp", nret);
        int nbad = bad[0], nbadnode = bad[1], badleaf = bad[2], badnode = bad[3];
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(bad);
        if(nbadnode) {
            printf("gpu_topology_build: task %d holds %d of %d retained particles whose top-leaf has no "
                   "node in the tree just built (first: leaf %d -> node %d, valid range [%d,%d)); the "
                   "attachment was checked against the standing tree, whose DomainNodeIndex this build "
                   "has already replaced.\n",
                   ThisTask, nbadnode, nret, badleaf, badnode, tbase, tbase + maxn);
            endrun(91570);
            return 1;
        }
        if(nbad) {
            printf("gpu_topology_build: task %d clamped %d of %d retained particles outside the top-leaf "
                   "they were clamped into; the key and the top-tree geometry disagree.\n",
                   ThisTask, nbad, nret);
            endrun(91563);
            return 1;
        }
    }

    /* Bucket counts, taken from the final topleaf assignment rather than fused into the
     * kernel above: a retained attachment changes which bucket a particle belongs to
     * after its key has been computed. */
    Kokkos::parallel_for("topo_zero_counts", ntl, KOKKOS_LAMBDA(int t) {
        dcnt[t] = 0;
        tcur[t] = 0;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_zero_counts", ntl);

    Kokkos::parallel_for("topo_count_buckets", npart, KOKKOS_LAMBDA(int i) {
        Kokkos::atomic_fetch_add(&dcnt[pt[i]], 1);
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_count_buckets", npart);
    /* The counts are read on the host (the sort loop below) and by later device passes, so they are
     * published to the shared array with plain stores once accumulated. */
    Kokkos::parallel_for("topo_publish_counts", ntl, KOKKOS_LAMBDA(int t) {
        tcnt[t] = dcnt[t];
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_publish_counts", ntl);

    /* Kernel 2: exclusive prefix scan to compute topleaf_start[]. */
    Kokkos::parallel_scan("topo_scan", ntl,
        KOKKOS_LAMBDA(int t, int &acc, bool final_pass) {
            int c = tcnt[t];
            if(final_pass) {tsta[t] = acc;}
            acc += c;
        });
    Kokkos::fence();
    /* Sentinel: tsta[NTopleaves] = total particle count. */
    Kokkos::parallel_for("topo_scan_sentinel", 1, KOKKOS_LAMBDA(int /*unused*/) {
        int total = 0;
        for(int t = 0; t < ntl; t++) {total += tcnt[t];}
        tsta[ntl] = total;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_scan_sentinel", 1);

    /* Kernel 3: scatter -- each particle takes its slot in its topleaf's
     * range via atomic_fetch_add into tcur[]. */
    Kokkos::parallel_for("topo_scatter", npart, KOKKOS_LAMBDA(int i) {
        int leaf = pt[i];
        int slot = Kokkos::atomic_fetch_add(&tcur[leaf], 1);
        sidx[tsta[leaf] + slot] = i;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_scatter", npart);

    /* Kernel 4: Morton order within each top-leaf range. */
    const int verify = topo_segsort_verify_wanted();
    if(!topo_segsort_enabled() && !verify) {return topo_sort_ranges_one_by_one(ntl);}
    const int total = tsta[ntl];
    std::vector<int> before, one_pass;
    if(verify) {before.assign(sidx, sidx + total);}
    int rc2 = topo_sort_ranges_one_pass(ntl, total);
    if(rc2 || !verify) {return rc2;}
    one_pass.assign(sidx, sidx + total);
    memcpy(sidx, before.data(), (size_t) total * sizeof(int));
    rc2 = topo_sort_ranges_one_by_one(ntl);
    if(rc2) {return rc2;}
    long ndiff = 0;
    for(int j = 0; j < total; j++) {if(sidx[j] != one_pass[(size_t) j]) {ndiff++;}}
    memcpy(sidx, one_pass.data(), (size_t) total * sizeof(int));   /* the run continues on the one-pass order */
    if(ndiff || ThisTask == 0) {
        printf("GIZMO_K1_SORT_VERIFY: task %d, %d particles in %d top-leaf ranges: %ld positions differ\n",
               ThisTask, total, ntl, ndiff);
        fflush(stdout);
    }
    return 0;
}

/* Pre-build stage, run while the standing tree is still intact.  For every particle it computes the
 * key and the topleaf its CURRENT position falls in; for those whose topleaf belongs to another rank
 * it recovers the topleaf they hung from in the standing tree and records them, so the rest of the
 * build keeps them local.
 *
 * topology_valid says whether the standing tree's Father[] links still describe these particles.  When
 * it is false the links are not consulted at all and the caller is told how many crossers there are, so
 * it can restore geometric ownership before building.
 *
 * The recovery needs no chain walk and no reverse map: node centres are fixed by the octree
 * subdivision and never move (only node lengths grow), so the node a particle hung under lies inside
 * its top leaf, and that leaf is simply the one the node's own centre keys into.
 *
 * Leaves the keys and leaves prepared for the following gpu_topology_build_data_path().  Returns 0 on
 * success; *n_crossed_out and *n_unrecovered_out are this rank's counts. */
extern "C" int gpu_topology_prepare_retained_attachment(int npart, int topology_valid,
                                                        long *n_crossed_out, long *n_unrecovered_out,
                                                        long *n_outside_extent_out)
{
    if(n_crossed_out)     {*n_crossed_out = 0;}
    if(n_unrecovered_out) {*n_unrecovered_out = 0;}
    if(n_outside_extent_out) {*n_outside_extent_out = 0;}
    g_prepared_npart = -1;
    g_retained_n     = 0;
    if(npart <= 0) {return 0;}

    struct topo_build_ctx ctx;
    int rc = topo_acquire_(npart, NULL, "gpu_topology_prepare_retained_attachment", &ctx);
    if(rc) {return rc;}

    const int *dtask = gpu_peano_walk_domain_task();
    if(!dtask) {printf("gpu_topology_build: DomainTask mirror null\n"); return 1;}

    Morton128                 *keys  = ctx.keys;
    struct particle_data      *P_dev = ctx.P_dev;
    const struct topnode_data *tn    = ctx.tn;
    int *pt    = g_particle_topleaf;
    const int *dni = ctx.dni;
    int *stage = g_sorted_idx;   /* free until the bucket scatter runs, and large enough by construction */

    const double dc0 = DomainCorner[0], dc1 = DomainCorner[1], dc2 = DomainCorner[2];
    const double dlen = DomainLen;
    const int    bits = BITS_PER_DIMENSION;
    const int    ntl  = NTopleaves;
    const int    me   = ThisTask;
    const int    tbase = All.TreeNodeIndexBase, maxn = MaxNodes;
    const int    use_standing_tree = (topology_valid && Father && Nodes_base) ? 1 : 0;
    struct NODE *Nodes_uvm = Nodes;
    const int   *father    = Father;

    int *ctr = g_counters_dev + TOPO_CTR_PREP_CROSSED;   /* device memory: see g_counters_dev */
    {const int zero3[3] = {0, 0, 0}; topo_counters_put(TOPO_CTR_PREP_CROSSED, zero3, 3);}

    Kokkos::parallel_for("topo_keys_and_assign", npart, KOKKOS_LAMBDA(int i) {
        /* Tested before the key is formed: outside the extent the key would name another cell. */
        if(position_outside_domain_extent(P_dev[i].Pos[0], P_dev[i].Pos[1], P_dev[i].Pos[2], dc0, dc1, dc2, dlen))
            {Kokkos::atomic_fetch_add(&ctr[2], 1); return;}
        Morton128 m;
        peanokey pkey = topo_key_of_position_(P_dev[i].Pos[0], P_dev[i].Pos[1], P_dev[i].Pos[2],
                                              dc0, dc1, dc2, dlen, bits, &m);
        keys[i] = m;
        int leaf = gpu_topleaf_for_key(tn, pkey);
        pt[i] = leaf;
        if(leaf < 0 || leaf >= ntl || dtask[leaf] == me) {return;}   /* the fast path: this rank owns it */
        stage[Kokkos::atomic_fetch_add(&ctr[0], 1)] = i;
        if(!use_standing_tree) {return;}
        int f = father[i];
        if(f < tbase || f >= tbase + maxn) {Kokkos::atomic_fetch_add(&ctr[1], 1); return;}
        Morton128 mf;
        peanokey fkey = topo_key_of_position_((double)Nodes_uvm[f].center[0],
                                              (double)Nodes_uvm[f].center[1],
                                              (double)Nodes_uvm[f].center[2],
                                              dc0, dc1, dc2, dlen, bits, &mf);
        int retained = gpu_topleaf_for_key(tn, fkey);
        if(retained < 0 || retained >= ntl || dtask[retained] != me) {Kokkos::atomic_fetch_add(&ctr[1], 1); return;}
        /* The leaf has to have a node in this tree before anything is attached to it: the clamp that
         * follows, and the growth pass after the build, both index Nodes[] with exactly this value.
         * An attachment naming a leaf whose node lies outside the tree is not usable, so it is
         * reported with the others that cannot be recovered rather than retained. */
        const int retained_node = dni[retained];
        if(retained_node < tbase || retained_node >= tbase + maxn) {Kokkos::atomic_fetch_add(&ctr[1], 1); return;}
        pt[i] = retained;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_keys_and_assign", npart);

    int prep[3]; topo_counters_get(TOPO_CTR_PREP_CROSSED, prep, 3);
    int n_crossed = prep[0], n_unrecovered = prep[1], n_outside_extent = prep[2];
    if(n_crossed_out)     {*n_crossed_out = (long) n_crossed;}
    if(n_unrecovered_out) {*n_unrecovered_out = (long) n_unrecovered;}
    if(n_outside_extent_out) {*n_outside_extent_out = (long) n_outside_extent;}

    if(n_crossed > 0 && use_standing_tree && n_unrecovered == 0) {
        if(g_retained_cap < n_crossed) {
            g_retained_slots = grow_int_buffer(g_retained_slots, g_retained_cap, n_crossed, "treescratch_build_retained_slots");
            if(!g_retained_slots) {g_retained_cap = 0; return 1;}
            g_retained_cap = n_crossed;
        }
        memcpy(g_retained_slots, stage, (size_t) n_crossed * sizeof(int));
        g_retained_n = n_crossed;
    }
    g_prepared_npart = npart;
    return 0;
}

/* Drop a prepared plan.  Called for any build the pre-build stage does not cover, so a later build
 * cannot consume keys and attachments computed for a different particle set. */
extern "C" void gpu_topology_forget_prepared(void)
{
    g_prepared_npart = -1;
    g_retained_n     = 0;
}

/* Grow the nodes holding a retained particle so their cubes cover where it actually is.  A retained
 * particle sits outside the nominal cube of the leaf it was kept in, and a node whose stated length
 * does not bound its contents can be accepted by an opening test that should have opened it.  This is
 * the same tolerance the tree already carries between rebuilds, where force_drift_node grows len by
 * the distance the node's contents can have moved.
 *
 * Runs after gpu_topology_finalize_father, which is what establishes the particle Father[] links, and
 * before the moments and the pseudo-particle exchange, which is what carries the grown top-leaf length
 * to the other ranks.  Walks the node's own SoA father links, because the AoS union still holds the
 * build-time suns layout at this point. */
extern "C" int gpu_topology_grow_retained_paths(void)
{
    if(g_retained_n <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->len || !soa->father) {printf("gpu_topology_grow_retained_paths: SoA not ready\n"); return 1;}
    const int *dni = gpu_peano_walk_domain_node_index();
    if(!dni) {printf("gpu_topology_grow_retained_paths: peano-walk mirror null\n"); return 1;}

    gpu_particles_arena_set_site("gpu_topology_grow_retained_paths");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {printf("gpu_topology_grow_retained_paths: P_dev null\n"); return 1;}

    const int *ret = g_retained_slots;
    const int  nret = g_retained_n;
    const int  tbase = All.TreeNodeIndexBase, maxn = MaxNodes;
    const int  ntl = NTopleaves;
    struct NODE *Nodes_uvm = Nodes;
    const int   *father    = Father;
    int         *pt        = g_particle_topleaf;
    MyFloat     *soa_len   = soa->len;
    const int   *soa_father = soa->father;

    int *unreached = (int *) gizmo_gpu_alloc_shared(sizeof(int), "treescratch_build_ctr");
    if(!unreached) {printf("gpu_topology_grow_retained_paths: counter alloc failed\n"); return 1;}
    *unreached = 0;

    Kokkos::parallel_for("topo_retained_grow", nret, KOKKOS_LAMBDA(int j) {
        int i = ret[j];
        /* Same index, same reason as the clamp: bound the leaf before reading the map. */
        const int lf = pt[i];
        int leaf_node = (lf >= 0 && lf < ntl) ? dni[lf] : -1;
        int no = father[i];
        volatile int reached_leaf = 0;
        for(int guard = 0; guard < GIZMO_GPU_MORTON_MAX_DEPTH + 8; guard++) {
            if(no < tbase || no >= tbase + maxn) {break;}
            Vec3<double> sep = {(double)P_dev[i].Pos[0] - (double)Nodes_uvm[no].center[0],
                                (double)P_dev[i].Pos[1] - (double)Nodes_uvm[no].center[1],
                                (double)P_dev[i].Pos[2] - (double)Nodes_uvm[no].center[2]};
            nearest_xyz(sep, -1);
            double reach = fabs(sep[0]);
            if(fabs(sep[1]) > reach) {reach = fabs(sep[1]);}
            if(fabs(sep[2]) > reach) {reach = fabs(sep[2]);}
            /* Unconditional: several retained particles can share a node, so reading the length
             * first to decide whether to raise it would be a plain load racing with their atomic
             * writes.  The maximum is already a no-op when the node is wide enough, and this path
             * only runs for particles that crossed a top-leaf boundary. */
            const MyFloat need = (MyFloat)(2.0 * reach);
            Kokkos::atomic_max(&Nodes_uvm[no].len,   need);
            Kokkos::atomic_max(&soa_len[no - tbase], need);
            if(no == leaf_node) {reached_leaf = 1; break;}
            no = soa_father[no - tbase];
        }
        /* The retained top-leaf is an ancestor of this particle's node by construction, so failing to
         * arrive at it means the node chain is not what the build just emitted.  It matters because
         * that leaf is the one whose length is exchanged: left ungrown, every other rank would bound
         * it by its nominal cube while it holds a particle outside. */
        if(!reached_leaf) {Kokkos::atomic_fetch_add(unreached, 1);}
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_retained_grow", nret);
    const int n_unreached = *unreached;
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(unreached);
    if(n_unreached) {
        printf("gpu_topology_grow_retained_paths: task %d failed to reach the retained top-leaf for %d of "
               "%d particles; their leaves would be published at a length that does not bound them.\n",
               ThisTask, n_unreached, nret);
        return 1;
    }
    return 0;
}

/* ---------------- 6.5c3 BFS topology emission ---------------------- */

namespace {

/* Per-internal-node BFS unit. */
struct BfsItem {
    int parent_soa;   /* parent's SoA index (= Nodes[] absolute - tree_base) */
    int range_first;  /* particle range [range_first, range_last) in sorted_idx */
    int range_last;
    int parent_depth; /* octree depth of parent (root = 0); split bits read at this level */
};

}  /* anonymous namespace */


extern "C" int gpu_topology_emit_bfs(int start_node_index, int *new_node_count_out)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(new_node_count_out) {*new_node_count_out = start_node_index;}

    /* Dependencies. */
    int             *sidx = (int *) gpu_topology_build_sorted_idx();  /* writable for collocation reshuffle */
    const int       *tsta = gpu_topology_build_topleaf_start();
    const int       *tcnt = gpu_topology_build_topleaf_count();
    const Morton128 *keys = gpu_morton_keys();
    const int       *dni  = gpu_peano_walk_domain_node_index();
    if(!sidx || !tsta || !tcnt || !keys || !dni) {
        printf("gpu_topology_emit_bfs: data-path / peano-walk scratch null\n");
        return 3;
    }

    /* For collocation RNG: need P_dev[].ID lookup inside the kernel. */
    gpu_particles_arena_set_site("gpu_topology_emit_bfs");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {printf("gpu_topology_emit_bfs: P_dev null\n"); return 3;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_topology_emit_bfs: SoA null\n"); return 3;}

    /* Subset slot->particle map (set by gpu_topology_build_data_path; NULL for
     * identity/full builds). Used to translate sidx slots to real P[] indices
     * at the two real-particle boundaries below: the collocation ID read and
     * the leaf emission into suns. keys[sidx[*]] + the split helpers stay slot-
     * indexed. Father[]/Nextnode[] inherit real indices via the leaf slots. */
    const int *stp = g_slot_map_active ? g_slot_to_particle : NULL;

    /* Capture SoA pointers for kernel use. */
    Vec3<MyFloat> *soa_center  = soa->center;
    integertime   *soa_ti      = soa->node_ti;
    const integertime ti_build = All.Ti_Current;   /* the build stamps every node here */
    MyFloat       *soa_len     = soa->len;
    int           *soa_father  = soa->father;
    int           *soa_suns    = soa->suns_backup;
    if(!soa_center || !soa_len || !soa_father || !soa_suns) {
        printf("gpu_topology_emit_bfs: SoA core fields not allocated\n");
        return 3;
    }
#if TREE_LEAF_BUCKET_SIZE > 1
    /* Records for the leaves that hold more than one particle.  A recorded leaf has at least two
     * members, so half the particle slots is an exact bound and the allocation never has to grow
     * mid-build.  The count is reset here rather than where the memory is taken, so that a build
     * which is retried after an overflow starts from an empty list instead of appending to the
     * abandoned attempt's records. */
    {
        /* Sized to the particles THIS build covers, which for a subset build is far fewer than the
         * tree's capacity; a recorded leaf holds at least two of them, so half is an exact bound. */
        if(g_sorted_npart <= 0) {
            printf("gpu_topology_emit_bfs: rank %d has no sorted order to describe leaves against\n", ThisTask);
            fflush(stdout);
            return 3;
        }
        const int cap_needed = g_sorted_npart / 2 + 1;
        if(g_leaf_chain_cap < cap_needed) {
            if(g_leaf_chain) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_leaf_chain);}
            g_leaf_chain = (struct LeafChainRecord *) gizmo_gpu_alloc_shared(
                (long) cap_needed * sizeof(struct LeafChainRecord), "treescratch_build_leaf_chain");
            g_leaf_chain_cap = g_leaf_chain ? cap_needed : 0;
        }
        if(!g_leaf_chain) {
            printf("gpu_topology_emit_bfs: rank %d could not reserve %d leaf records; the tree is not\n"
                   "built rather than retried, because this is an allocation failure and not a shortage\n"
                   "of tree nodes.\n", ThisTask, cap_needed);
            fflush(stdout);
            return 3;
        }
        g_leaf_chain_count = 0;
    }
    struct LeafChainRecord *chain_out = g_leaf_chain;
    int *chain_n   = g_counters_dev + TOPO_CTR_LEAF_CHAIN;   /* device memory: see g_counters_dev */
    int  chain_cap = g_leaf_chain_cap;
#endif

    int ntl       = NTopleaves;
    int max_nodes = MaxNodes;
    int tree_base  = All.TreeNodeIndexBase;

    /* The worklist sizes, the node counter and the failure flag are taken by device atomics, so they
     * live in device memory (see g_counters_dev); the host reads them with an explicit copy, h[] below,
     * once per level.  Slots BFS_A and BFS_B alternate as the current and the next worklist size. */
    if(!topo_counters_dev()) {printf("gpu_topology_emit_bfs: counter alloc failed\n"); return 3;}
    int h[TOPO_CTR_LEAF_CHAIN + 1] = {0, 0, start_node_index, 0, 0};
    topo_counters_put(TOPO_CTR_BFS_A, h, TOPO_CTR_LEAF_CHAIN + 1);
    int cur = TOPO_CTR_BFS_A;   /* which slot holds the current worklist's size */
    int *sz_curr = g_counters_dev + TOPO_CTR_BFS_A;
    int *sz_next = g_counters_dev + TOPO_CTR_BFS_B;
    int *ncount  = g_counters_dev + TOPO_CTR_BFS_NODES;
    int *fail    = g_counters_dev + TOPO_CTR_BFS_FAIL;

    /* Worklist Views in device-local memory.  Sized at max_nodes upper
     * bound; in practice the worklist size at any one level is <= live
     * leaf count, much smaller. */
    using ExSpace  = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExSpace::memory_space;
    Kokkos::View<BfsItem*, MemSpace> wl_a("bfs_wl_a", max_nodes);
    Kokkos::View<BfsItem*, MemSpace> wl_b("bfs_wl_b", max_nodes);

    /* Initial population: for each topleaf with >= 1 particle, push a
     * BfsItem.  Compute parent_depth from the topleaf's len. */
    const double dlen = DomainLen;
    Kokkos::parallel_for("topo_bfs_init", ntl, KOKKOS_LAMBDA(int t) {
        int count = tcnt[t];
        if(count < 1) {return;}
        int parent_abs = dni[t];
        int parent_soa = parent_abs - tree_base;
        if(parent_soa < 0 || parent_soa >= max_nodes) {return;}

        /* Topleaf depth = log2(DomainLen / topleaf_len), exact integer. */
        double len_d = (double)soa_len[parent_soa];
        int depth = 0;
        if(len_d > 0.0) {
            double r = dlen / len_d;
            while(r > 1.5 && depth < 64) {depth++; r *= 0.5;}
        }

        BfsItem w;
        w.parent_soa   = parent_soa;
        w.range_first  = tsta[t];
        w.range_last   = tsta[t] + count;
        w.parent_depth = depth;
        int slot = Kokkos::atomic_fetch_add(sz_curr, 1);
        if(slot < max_nodes) {wl_a(slot) = w;}
        else {Kokkos::atomic_fetch_max(fail, 1);}
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_bfs_init", ntl);
    topo_counters_get(TOPO_CTR_BFS_A, h, TOPO_CTR_LEAF_CHAIN + 1);

    if(h[TOPO_CTR_BFS_FAIL]) {
        printf("gpu_topology_emit_bfs: worklist overflow at init\n");
        return h[TOPO_CTR_BFS_FAIL];
    }


    /* BFS loop: each iteration processes wl_curr, populates wl_next. */
    Kokkos::View<BfsItem*, MemSpace> wl_curr = wl_a;
    Kokkos::View<BfsItem*, MemSpace> wl_next = wl_b;
    int level_guard = 0;
    while(h[cur] > 0 && h[TOPO_CTR_BFS_FAIL] == 0 && level_guard < GIZMO_GPU_MORTON_MAX_DEPTH + 4) {
        int curr_size = h[cur];
        {const int zero = 0; topo_counters_put(TOPO_CTR_BFS_A + TOPO_CTR_BFS_B - cur, &zero, 1);}

        Kokkos::parallel_for("topo_bfs_level", curr_size, KOKKOS_LAMBDA(int i) {
            BfsItem w = wl_curr(i);

            /* Read parent geometry from SoA. */
            Vec3<MyFloat> pc = soa_center[w.parent_soa];
            MyFloat       pl = soa_len[w.parent_soa];
            MyFloat       lh = (MyFloat)(0.25 * (double)pl);  /* offset of child center from parent center */
            MyFloat       cl = (MyFloat)(0.5  * (double)pl);  /* child len */

            /* 8-way split.  Default: Morton bits at split level = parent_depth.
             * Collocation: if the range is fully collocated (LCP >= 126),
             * Morton-bit split would put everything in one octant -> infinite
             * recursion.  Reshuffle by random per-particle octant instead,
             * matching CPU forcetree.cc:267-273 semantics.  Each BFS level
             * uses parent_depth as the RNG counter, so re-rolls every level. */
            int child_starts[9];
            int range_count = w.range_last - w.range_first;
            int do_random = 0;
            if(range_count > 1) {
                int lo_idx = sidx[w.range_first];
                int hi_idx = sidx[w.range_last - 1];
                int lcp = gpu_morton_lcp_bits(keys[lo_idx], keys[hi_idx]);
                if(lcp >= 126) {
                    /* Collocated range (keys identical). Randomizing octants HERE —
                     * at the isolation depth, node len ~ the local interparticle
                     * separation — detaches the descendants' cubes from the true
                     * particle positions by up to ~len, which silently breaks every
                     * geometric cube-prune walk (neighbor export, gravity opening).
                     * The CPU build (forcetree.cc:223) randomizes ONLY once
                     * len < EPSILON_FOR_TREERND_SUBNODE_SPLITTING * softening, so
                     * the mislocation is bounded by a scale no physics query can
                     * resolve. Mirror that floor: ABOVE it, keep the deterministic
                     * key split (identical keys -> one occupied child; the chain
                     * descends, len halves, depth advances — no infinite recursion).
                     * Randomize only below the floor, or when the key bits are
                     * exhausted (mislocation then <= box/2^42, negligible).
                     * ForceSoftening not yet cached (first build) reads as 0 ->
                     * floor 0 -> falls through to the bit-exhaustion arm: safe. */
                    double split_scale = 0;
                    for(int j = w.range_first; j < w.range_last; j++) {
                        int sj = sidx[j];
                        double fs = (double)P_dev[stp ? stp[sj] : sj].ForceSoftening;
                        if(j == w.range_first || fs < split_scale) {split_scale = fs;}
                    }
                    if((double)cl < EPSILON_FOR_TREERND_SUBNODE_SPLITTING * split_scale
                       || w.parent_depth >= GIZMO_GPU_MORTON_MAX_DEPTH) {do_random = 1;}
                }
            }
            if(do_random) {
                /* Inner lambda must NOT use KOKKOS_FUNCTION (extended __host__ __device__
                 * lambda) -- nvcc forbids nesting extended lambdas inside each other.
                 * Plain capture is implicitly __device__ inside the outer KOKKOS_LAMBDA. */
                auto id_of = [P_dev, stp] (int idx) -> uint64_t {
                    return (uint64_t) P_dev[stp ? stp[idx] : idx].ID;
                };
                gpu_morton_split_8way_random_inplace(
                    sidx + w.range_first, range_count, id_of,
                    (uint64_t) w.parent_depth, child_starts);
            } else {
                gpu_morton_split_8way(sidx, keys, w.range_first, w.range_last,
                                      w.parent_depth, child_starts);
            }

            for(int k = 0; k < 8; k++) {
                int rf = w.range_first + child_starts[k];
                int rl = w.range_first + child_starts[k+1];
                int cnt = rl - rf;
                int slot_value = -1;

#if TREE_LEAF_BUCKET_SIZE == 1
                if(cnt == 1) {
#else
                if(cnt >= 1 && cnt <= TREE_LEAF_BUCKET_SIZE) {
#endif
                    /* Terminal leaf: this child holds few enough particles that subdividing it
                     * further costs more tree than it saves work, so it is not split at all.  The
                     * slot carries the FIRST particle; at a leaf size above one the rest are
                     * linked behind it once the leaf's successor is known, so a walk reaches them
                     * exactly as it reaches any run of particles and no walker learns a new node
                     * kind.  The member count is NOT stored on the tree, because particles are
                     * inserted into a live tree without a rebuild and any cached count would go
                     * stale; the record written below describes this build only and is retired
                     * before anything can insert.
                     *
                     * At TREE_LEAF_BUCKET_SIZE == 1 the test above is `cnt == 1` and nothing but
                     * the slot is written: the historical single-particle leaf, exactly. */
                    slot_value = stp ? stp[sidx[rf]] : sidx[rf];
#if TREE_LEAF_BUCKET_SIZE > 1
                    if(cnt > 1) {
                        /* Record the leaf and move on.  Linking the members here would make one
                         * thread walk the whole leaf while its neighbours handle one particle each,
                         * which is the imbalance this tree is being changed to avoid; the links are
                         * written later, one thread per member. */
                        const int slot = Kokkos::atomic_fetch_add(chain_n, 1);
                        if(slot < chain_cap) {
                            chain_out[slot].range_first = rf;
                            chain_out[slot].count       = cnt;
                            chain_out[slot].parent_abs  = tree_base + w.parent_soa;
                            chain_out[slot].successor   = -1;
                        }
                    }
#endif
                } else if(cnt > 1) {
                    /* Allocate a new internal node from the device counter.
                     * Collocation in the sub-range is OK -- the next BFS
                     * level on this child will trigger the random reshuffle
                     * at its parent boundary check. */
                    int new_soa = Kokkos::atomic_fetch_add(ncount, 1);
                    if(new_soa >= max_nodes) {
                        Kokkos::atomic_fetch_max(fail, 1);
                        continue;
                    }
                    /* Initialize child node geometry + suns + father. */
                    Vec3<MyFloat> cc;
                    cc[0] = pc[0] + ((k & 1) ? lh : -lh);
                    cc[1] = pc[1] + ((k & 2) ? lh : -lh);
                    cc[2] = pc[2] + ((k & 4) ? lh : -lh);
                    soa_center[new_soa] = cc;
                    soa_len[new_soa]    = cl;
                    if(soa_ti) {soa_ti[new_soa] = ti_build;}   /* pairs with the length */
                    soa_father[new_soa] = w.parent_soa + tree_base;
                    long sb = (long)new_soa * 8;
                    for(int s = 0; s < 8; s++) {soa_suns[sb + s] = -1;}

                    slot_value = tree_base + new_soa;

                    /* Push child BFS entry. */
                    BfsItem nw;
                    nw.parent_soa   = new_soa;
                    nw.range_first  = rf;
                    nw.range_last   = rl;
                    nw.parent_depth = w.parent_depth + 1;
                    int slot = Kokkos::atomic_fetch_add(sz_next, 1);
                    if(slot < max_nodes) {wl_next(slot) = nw;}
                    else {Kokkos::atomic_fetch_max(fail, 1);}
                }
                /* Write parent.suns[k]. */
                long pb = (long)w.parent_soa * 8;
                soa_suns[pb + k] = slot_value;
            }
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("topo_bfs_level", curr_size);

        /* Swap worklists. */
        Kokkos::View<BfsItem*, MemSpace> tmp = wl_curr; wl_curr = wl_next; wl_next = tmp;
        int *tmpp = sz_curr; sz_curr = sz_next; sz_next = tmpp;
        cur = TOPO_CTR_BFS_A + TOPO_CTR_BFS_B - cur;
        level_guard++;
        topo_counters_get(TOPO_CTR_BFS_A, h, TOPO_CTR_LEAF_CHAIN + 1);
    }
#if TREE_LEAF_BUCKET_SIZE > 1
    g_leaf_chain_count = h[TOPO_CTR_LEAF_CHAIN];
#endif


#if TREE_LEAF_BUCKET_SIZE > 1
    /* Check every record before anything indexes through one.  These replaced the chain guards that
     * used to sit in the passes themselves, so they are now the only thing standing between a
     * miswritten range and a device read at an arbitrary offset; checking here means neither
     * consumer has to, and neither is the first place a bad range would be noticed.  Reported with
     * the offending record rather than as a count, because one bad range is a build fault and the
     * rest tell you nothing further. */
    if(g_leaf_chain_count >= 0 && g_leaf_chain_count <= g_leaf_chain_cap) {
        const int nrec_chk = g_leaf_chain_count;
        const struct LeafChainRecord *chk = g_leaf_chain;
        const int *sidx_chk  = g_sorted_idx;
        const int *stp_chk   = g_slot_map_active ? g_slot_to_particle : NULL;
        const int  np_chk    = g_sorted_npart;
        const int  slots_chk = All.TreeParticleSlots;
        const int  node_lo   = tree_base, node_hi = tree_base + max_nodes;
        int *badrec = (int *) gizmo_gpu_alloc_shared(2 * sizeof(int), "treescratch_build_ctr");
        if(!badrec) {
            printf("gpu_topology_emit_bfs: could not allocate the record check\n");
            return 3;
        }
        badrec[0] = -1; badrec[1] = 0;
        Kokkos::parallel_for("leaf_chain_check", nrec_chk, KOKKOS_LAMBDA(int r) {
            const struct LeafChainRecord d = chk[r];
            int why = 0;
            if(d.count < 2)                                   {why = 1;}
            else if(d.range_first < 0 || d.range_first + d.count > np_chk) {why = 2;}
            else if(d.parent_abs < node_lo || d.parent_abs >= node_hi)     {why = 3;}
            else {
                for(int m = 0; m < d.count && !why; m++) {
                    const int s = sidx_chk[d.range_first + m];
                    if(s < 0 || s >= np_chk) {why = 4; break;}
                    const int pp = stp_chk ? stp_chk[s] : s;
                    if(pp < 0 || pp >= slots_chk) {why = 5; break;}
                }
            }
            if(why) {if(Kokkos::atomic_fetch_add(&badrec[1], 1) == 0) {badrec[0] = r * 8 + why;}}
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("leaf_chain_check", nrec_chk);
        const int nbad = badrec[1];
        /* Whichever thread won the counter writes this, so a nonzero count should always carry a
         * record; clamp anyway, because the alternative is indexing the reason table at -1. */
        const int first = (badrec[0] >= 0) ? badrec[0] : 0;
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(badrec);
        if(nbad > 0) {
            static const char *why_text[6] = {"", "fewer than two members", "a range outside the particles this build covers",
                                              "a parent outside the local nodes", "a sorted-order entry out of range",
                                              "a member outside the tree's particle slots"};
            printf("gpu_topology_emit_bfs: rank %d wrote %d unusable leaf record(s); the first is record %d,\n"
                   "which has %s. Nothing is indexed through these, so the tree is not built.\n",
                   ThisTask, nbad, first / 8, why_text[first % 8]);
            fflush(stdout);
            return 3;
        }
    }
    if(g_leaf_chain_count > g_leaf_chain_cap) {
        printf("gpu_topology_emit_bfs: rank %d found %d multi-particle leaves but reserved room for %d.\n"
               "Each holds at least two particles, so this cannot happen for a well-formed tree; the\n"
               "build stops rather than leaving part of the tree unthreaded.\n",
               ThisTask, g_leaf_chain_count, g_leaf_chain_cap);
        fflush(stdout);
        return 3;
    }
#endif
    int rc = (h[TOPO_CTR_BFS_FAIL] == 0) ? 0 : h[TOPO_CTR_BFS_FAIL];
    int new_total = h[TOPO_CTR_BFS_NODES];
    int work_left = h[cur];
    if(new_node_count_out) {*new_node_count_out = new_total;}

    /* Ask whether work was LEFT, not whether the last allowed level was used: a breadth-first walk
     * that finishes exactly on that level leaves an empty worklist and has not exceeded anything.
     * Testing the level alone reported a completed build as a failure, which the caller now treats
     * as fatal rather than retrying, so the distinction has to be right. */
    if(work_left > 0 && level_guard >= GIZMO_GPU_MORTON_MAX_DEPTH + 4 && rc == 0) {
        printf("gpu_topology_emit_bfs: BFS exceeded depth guard (level=%d) with work still pending -- a node is not subdividing\n", level_guard);
        return 4;
    }
    return rc;
}
extern "C" int gpu_topology_writeback_to_aos(int first_soa_idx, int last_soa_idx)
{
    if(last_soa_idx <= first_soa_idx) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {return 1;}
    Vec3<MyFloat> *soa_center  = soa->center;
    MyFloat       *soa_len     = soa->len;
    int           *soa_suns    = soa->suns_backup;
    if(!soa_center || !soa_len || !soa_suns) {return 1;}

    /* GPU kernel writeback (was host OMP loop).  Nodes_base is
     * UVM, so device-side writes work directly.  All stores are
     * independent per-k. */
    struct NODE *Nodes_uvm = Nodes_base;
    int range = last_soa_idx - first_soa_idx;
    int base_k = first_soa_idx;
    Kokkos::parallel_for("topo_writeback_to_aos", range, KOKKOS_LAMBDA(int j) {
        int k = base_k + j;
        long sb = (long)k * 8;
        Nodes_uvm[k].u.suns[0] = soa_suns[sb + 0];
        Nodes_uvm[k].u.suns[1] = soa_suns[sb + 1];
        Nodes_uvm[k].u.suns[2] = soa_suns[sb + 2];
        Nodes_uvm[k].u.suns[3] = soa_suns[sb + 3];
        Nodes_uvm[k].u.suns[4] = soa_suns[sb + 4];
        Nodes_uvm[k].u.suns[5] = soa_suns[sb + 5];
        Nodes_uvm[k].u.suns[6] = soa_suns[sb + 6];
        Nodes_uvm[k].u.suns[7] = soa_suns[sb + 7];
        Nodes_uvm[k].len       = soa_len[k];
        Nodes_uvm[k].center    = soa_center[k];
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_writeback_to_aos", range);
    return 0;
}

extern "C" const int *gpu_topology_build_sorted_idx(void)        { return g_sorted_idx;       }
extern "C" const int *gpu_topology_build_topleaf_start(void)     { return g_topleaf_start;    }
extern "C" const int *gpu_topology_build_topleaf_count(void)     { return g_topleaf_count;    }
extern "C" const int *gpu_topology_build_particle_topleaf(void)  { return g_particle_topleaf; }

/* The two passes below finish a multi-particle leaf, and both divide a leaf across threads rather
 * than giving one thread the whole leaf.  That division is the point: a leaf may hold anything from
 * two particles to several thousand, and walking a long one in a single thread while its neighbours
 * handle one particle each reproduces, inside the build, the same imbalance the leaves exist to
 * remove.
 *
 * Whether that division is worth making depends on how long a leaf can be, which the leaf size fixes
 * at compile time.  At a small cap a leaf holds a handful of particles and a flat loop over the
 * leaves is both simpler and faster than giving each one a team of idle lanes; past that the team is
 * what keeps a long leaf off a single thread.  A host backend always takes the flat form: its leaves
 * are already spread across threads and there are no lanes to divide them among.
 *
 * Neither pass may run at a leaf size of one: there is nothing to divide, and the ordinary passes
 * already do the work. */

/* Give every particle in a multi-particle leaf the node it hangs from.  The moment pass reads
 * Father[] per particle, so a member left pointing at an older node would contribute its mass to the
 * wrong node with nothing visible in the output.  Runs after Father[] is cleared and before the
 * moments are accumulated. */
extern "C" int gpu_leaf_chain_assign_fathers(void)
{
#if TREE_LEAF_BUCKET_SIZE == 1
    return 0;
#else
    if(Numnodestree <= 0) {return 0;}
    if(!g_leaf_chain) {return 0;}
    const int nrec = g_leaf_chain_count;
    if(nrec <= 0) {return 0;}
    if(!Father)     {printf("gpu_leaf_chain_assign_fathers: Father[] null\n");     return 1;}
    if(!g_sorted_idx) {printf("gpu_leaf_chain_assign_fathers: sorted order gone\n"); return 1;}

    const struct LeafChainRecord *rec = g_leaf_chain;
    const int *sidx  = g_sorted_idx;
    const int *stp   = g_slot_map_active ? g_slot_to_particle : NULL;
    const int  slots = All.TreeParticleSlots;
    int *Father_uvm  = Father;
    int *bad = (int *) gizmo_gpu_alloc_shared(sizeof(int), "treescratch_build_ctr");
    if(!bad) {printf("gpu_leaf_chain_assign_fathers: could not allocate the fault flag\n"); return 1;}
    *bad = 0;

    if(leaf_members_run_flat()) {
        Kokkos::parallel_for("leaf_chain_fathers", nrec, KOKKOS_LAMBDA(int r) {
            const struct LeafChainRecord d = rec[r];
            for(int m = 0; m < d.count; m++) {
                const int s = sidx[d.range_first + m];
                const int p = stp ? stp[s] : s;
                if(p < 0 || p >= slots) {Kokkos::atomic_fetch_max(bad, 1);}
                else {Father_uvm[p] = d.parent_abs;}
            }
        });
    } else {
        Kokkos::TeamPolicy<> policy(nrec, Kokkos::AUTO, 1);
        Kokkos::parallel_for("leaf_chain_fathers", policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
                const struct LeafChainRecord d = rec[team.league_rank()];
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, d.count), [&](const int m) {
                    const int s = sidx[d.range_first + m];
                    const int p = stp ? stp[s] : s;
                    if(p < 0 || p >= slots) {Kokkos::atomic_fetch_max(bad, 1);}
                    else {Father_uvm[p] = d.parent_abs;}
                });
            });
    }
    Kokkos::fence();
    gizmo_gpu_check_last_error("leaf_chain_fathers", nrec);

    const int fault = *bad;
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(bad);
    if(fault) {
        printf("gpu_leaf_chain_assign_fathers: rank %d recorded a leaf member outside the tree's\n"
               "particle slots (0..%d), so the range written for that leaf does not describe\n"
               "particles and the tree is not built.\n", ThisTask, slots - 1);
        fflush(stdout);
        return 1;
    }
    return 0;
#endif
}

/* Link the members of every multi-particle leaf, once each leaf's traversal successor is known.
 *
 * The threading pass treats a leaf as the single particle its slot names, and writes that leaf's
 * successor onto it, exactly as it did before leaves could hold more than one particle.  So the
 * successor is read back off the head here BEFORE any link is written, and only then are the members
 * joined and the successor moved to the last of them.  Doing both in one pass would race: the thread
 * writing the head's link destroys the value the thread handling the tail still has to read.
 *
 * This must run before the tree is exported.  The export routine enumerates a leaf by following
 * these same links, so exporting an unlinked leaf would ship its first particle and silently leave
 * out the rest. */
extern "C" int gpu_leaf_chain_materialize(void)
{
#if TREE_LEAF_BUCKET_SIZE == 1
    return 0;
#else
    if(Numnodestree <= 0) {return 0;}
    if(!g_leaf_chain) {return 0;}
    const int nrec = g_leaf_chain_count;
    if(nrec <= 0) {return 0;}
    if(!g_sorted_idx) {printf("gpu_leaf_chain_materialize: sorted order gone\n"); return 1;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    const int slots = All.TreeParticleSlots;
    if(!soa || !soa->nextnode_aux || soa->nextnode_aux_size < slots) {
        printf("gpu_leaf_chain_materialize: the particle successor array is missing or too small\n");
        return 1;
    }
    struct LeafChainRecord *rec = g_leaf_chain;
    int *aux = soa->nextnode_aux;
    const int *sidx = g_sorted_idx;
    const int *stp  = g_slot_map_active ? g_slot_to_particle : NULL;

    /* 1. take each leaf's successor off its head, before anything overwrites it */
    Kokkos::parallel_for("leaf_chain_capture", nrec, KOKKOS_LAMBDA(int r) {
        const int s = sidx[rec[r].range_first];
        rec[r].successor = aux[stp ? stp[s] : s];
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("leaf_chain_capture", nrec);

    /* 2. join the members, successor onto the last */
    int *bad = (int *) gizmo_gpu_alloc_shared(sizeof(int), "treescratch_build_ctr");
    if(!bad) {printf("gpu_leaf_chain_materialize: could not allocate the fault flag\n"); return 1;}
    *bad = 0;
    if(leaf_members_run_flat()) {
        Kokkos::parallel_for("leaf_chain_link", nrec, KOKKOS_LAMBDA(int r) {
            const struct LeafChainRecord d = rec[r];
            for(int m = 0; m < d.count; m++) {
                const int s = sidx[d.range_first + m];
                const int p = stp ? stp[s] : s;
                if(p < 0 || p >= slots) {Kokkos::atomic_fetch_max(bad, 1); continue;}
                if(m + 1 < d.count) {const int sn = sidx[d.range_first + m + 1]; aux[p] = stp ? stp[sn] : sn;}
                else {aux[p] = d.successor;}
            }
        });
    } else {
        Kokkos::TeamPolicy<> policy(nrec, Kokkos::AUTO, 1);
        Kokkos::parallel_for("leaf_chain_link", policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
                const struct LeafChainRecord d = rec[team.league_rank()];
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, d.count), [&](const int m) {
                    const int s = sidx[d.range_first + m];
                    const int p = stp ? stp[s] : s;
                    if(p < 0 || p >= slots) {Kokkos::atomic_fetch_max(bad, 1); return;}
                    if(m + 1 < d.count) {const int sn = sidx[d.range_first + m + 1]; aux[p] = stp ? stp[sn] : sn;}
                    else {aux[p] = d.successor;}
                });
            });
    }
    Kokkos::fence();
    gizmo_gpu_check_last_error("leaf_chain_link", nrec);

    const int fault = *bad;
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(bad);
    if(fault) {
        printf("gpu_leaf_chain_materialize: rank %d recorded a leaf member outside the tree's particle\n"
               "slots (0..%d); the tree is not built.\n", ThisTask, slots - 1);
        fflush(stdout);
        return 1;
    }

    /* These records describe this build only, and they are released here rather than kept for the
     * next one: nothing downstream can then consult a leaf's membership, which is what keeps a later
     * insertion into the live tree correct, and the memory does not sit held for the rest of the run
     * on a code whose tree is already the thing competing for it. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_leaf_chain);
    g_leaf_chain     = NULL;
    g_leaf_chain_cap = 0;
    g_leaf_chain_count  = 0;
    return 0;
#endif
}

extern "C" void gpu_topology_build_release(void)
{
    if(g_sorted_idx)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_sorted_idx);       g_sorted_idx       = NULL;}
    if(g_particle_topleaf) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_particle_topleaf); g_particle_topleaf = NULL;}
    if(g_topleaf_start)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_start);    g_topleaf_start    = NULL;}
    if(g_topleaf_count)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_count);    g_topleaf_count    = NULL;}
    gizmo_gpu_free_device(g_topleaf_count_dev);  g_topleaf_count_dev  = NULL;
    gizmo_gpu_free_device(g_topleaf_cursor_dev); g_topleaf_cursor_dev = NULL;
    g_topleaf_dev_cap = 0;
    gizmo_gpu_free_device(g_counters_dev);       g_counters_dev       = NULL;
    if(g_slot_to_particle) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_slot_to_particle); g_slot_to_particle = NULL;}
    if(g_retained_slots)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_retained_slots);   g_retained_slots   = NULL;}
#if TREE_LEAF_BUCKET_SIZE > 1
    if(g_leaf_chain)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_leaf_chain);       g_leaf_chain       = NULL;}
    g_leaf_chain_count = 0;
    g_leaf_chain_cap = 0;
#endif
    g_slot_map_active = 0;
    g_slot_cap = 0;
    g_retained_cap = 0;
    g_retained_n = 0;
    g_prepared_npart = -1;
    g_npart_cap = 0;
    g_topleaf_cap = 0;
}


