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

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
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
static int *g_topleaf_cursor      = NULL;  /* [NTopleaves] -- scatter cursors */
static int  g_npart_cap           = 0;
static int  g_topleaf_cap         = 0;

/* Subset-build slot->particle map. force_treebuild(npart, mp) may request a
 * tree over an arbitrary subset of P[] (mp[i].index, e.g. SUBFIND per-species
 * density or collective unbinding). The build pipeline is slot-indexed (slot i
 * in 0..npart-1); g_slot_to_particle[slot] gives the real P[] index. Identity
 * (mp==NULL, full-tree builds) leaves g_slot_map_active=0 and the map unused.
 * Scratch-owned + reset per build: set definitively in gpu_topology_build_data_path
 * before it can be consumed, stays valid through gpu_topology_emit_bfs (incl.
 * overflow/retry), freed in gpu_topology_build_release. */
static int *g_slot_to_particle    = NULL;  /* [npart]; real index per build slot */
static int  g_slot_cap            = 0;     /* own capacity: lazily grown only for subset builds */
static int  g_slot_map_active     = 0;     /* 1 iff this build is a non-identity subset */

/* Allocate/grow a SharedSpace int buffer. `label` is a stable string literal
   (the memory ledger classifies allocations by label; a stack buffer must not be
   relied on to survive into the free callback). */
static int *grow_int_buffer(int *buf, int old_cap, int new_cap, const char *label) {
    if(old_cap >= new_cap) {return buf;}
    if(buf) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(buf);}
    buf = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label, (long)new_cap * sizeof(int));
    if(!buf) {
        printf("gpu_topology_build: %s alloc failed (n=%d)\n", label, new_cap);
    }
    return buf;
}

}  /* anonymous namespace */

extern "C" int gpu_topology_build_data_path(int npart, const struct unbind_data *mp)
{
    /* Reset the subset map state first: any early-return failure below then
     * leaves an identity (safe) build, never a stale subset map from a prior
     * build that gpu_topology_emit_bfs could consume. */
    g_slot_map_active = 0;
    if(npart <= 0) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* Acquire dependencies. */
    int rc = gpu_peano_walk_acquire();
    if(rc) {printf("gpu_topology_build: gpu_peano_walk_acquire failed\n"); return rc;}
    Morton128 *keys = gpu_morton_keys_acquire(npart);
    if(!keys) {return 1;}

    gpu_particles_arena_set_site("gpu_topology_build_data_path");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {printf("gpu_topology_build: P_dev null\n"); return 1;}

    const struct topnode_data *tn  = gpu_peano_walk_topnodes();
    const int                 *dni = gpu_peano_walk_domain_node_index();
    if(!tn || !dni) {printf("gpu_topology_build: peano-walk mirrors null\n"); return 1;}
    (void)dni;  /* used by 6.5c3 topology emission to map topleaf -> Nodes[] slot */

    /* Grow per-particle scratch. */
    if(g_npart_cap < npart) {
        g_sorted_idx       = grow_int_buffer(g_sorted_idx,       g_npart_cap, npart, "treescratch_build_sorted_idx");
        g_particle_topleaf = grow_int_buffer(g_particle_topleaf, g_npart_cap, npart, "treescratch_build_particle_topleaf");
        if(!g_sorted_idx || !g_particle_topleaf) {g_npart_cap = 0; return 1;}
        g_npart_cap = npart;
    }

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
        g_topleaf_cursor = grow_int_buffer(g_topleaf_cursor, g_topleaf_cap, newcap, "treescratch_build_topleaf_cursor");
        if(!g_topleaf_start || !g_topleaf_count || !g_topleaf_cursor) {g_topleaf_cap = 0; return 1;}
        g_topleaf_cap = newcap;
    }

    int  ntl  = NTopleaves;
    int *pt   = g_particle_topleaf;
    int *tcnt = g_topleaf_count;
    int *tcur = g_topleaf_cursor;
    int *tsta = g_topleaf_start;
    int *sidx = g_sorted_idx;

    /* Capture domain bounds for the encode kernel. */
    const double dc0 = DomainCorner[0];
    const double dc1 = DomainCorner[1];
    const double dc2 = DomainCorner[2];
    const double dlen = DomainLen;
    const double inv_dlen = (dlen > 0.0) ? (1.0 / dlen) : 0.0;
    const int    bits = BITS_PER_DIMENSION;

    /* Zero topleaf bucket counters. */
    Kokkos::parallel_for("topo_zero_counts", ntl, KOKKOS_LAMBDA(int t) {
        tcnt[t] = 0;
        tcur[t] = 0;
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_zero_counts", ntl);

    /* Kernel 1: per-particle Peano + Morton key compute, TopNodes walk to
     * topleaf id, write Morton key + topleaf id, increment bucket count.
     * Geometry is read from the real particle (slot->particle map for subset
     * builds); keys[]/pt[] stay slot-indexed. */
    const int *stp = g_slot_map_active ? g_slot_to_particle : NULL;
    Kokkos::parallel_for("topo_keys_and_assign", npart, KOKKOS_LAMBDA(int i) {
        int real = stp ? stp[i] : i;
        double fx = (P_dev[real].Pos[0] - dc0) * inv_dlen;
        double fy = (P_dev[real].Pos[1] - dc1) * inv_dlen;
        double fz = (P_dev[real].Pos[2] - dc2) * inv_dlen;
        if(fx < 0.0) {fx = 0.0;} if(fx >= 1.0) {fx = 0.99999999999999988897;}
        if(fy < 0.0) {fy = 0.0;} if(fy >= 1.0) {fy = 0.99999999999999988897;}
        if(fz < 0.0) {fz = 0.0;} if(fz >= 1.0) {fz = 0.99999999999999988897;}
        uint64_t ix = gpu_morton_double_to_int42(fx + 1.0);
        uint64_t iy = gpu_morton_double_to_int42(fy + 1.0);
        uint64_t iz = gpu_morton_double_to_int42(fz + 1.0);

        Morton128 m;
        peanokey  pkey = gpu_peano_and_morton_key(ix, iy, iz, bits, &m);
        keys[i] = m;

        int leaf = gpu_topleaf_for_key(tn, pkey);
        pt[i] = leaf;

        Kokkos::atomic_fetch_add(&tcnt[leaf], 1);
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("topo_keys_and_assign", npart);

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

    /* Kernel 4 (per-topleaf sort): host loop over topleaves, dispatching
     * one Morton sort per range.  NTopleaves is typically O(100); each
     * sort is small.  Future optimization: parallel team-policy sort,
     * but correctness-first here. */
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

/* ---------------- 6.5c3 BFS topology emission ---------------------- */

namespace {

/* Per-internal-node BFS unit. */
struct BfsItem {
    int parent_soa;   /* parent's SoA index (= Nodes[] absolute - MaxPart) */
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
    MyFloat       *soa_len     = soa->len;
    int           *soa_father  = soa->father;
    int           *soa_suns    = soa->suns_backup;
    if(!soa_center || !soa_len || !soa_father || !soa_suns) {
        printf("gpu_topology_emit_bfs: SoA core fields not allocated\n");
        return 3;
    }

    int ntl       = NTopleaves;
    int max_nodes = MaxNodes;
    int max_part  = All.MaxPart;

    /* SharedSpace single-int counters: easy host/device coordination. */
    int *sz_curr = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("treescratch_build_ctr", sizeof(int));
    int *sz_next = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("treescratch_build_ctr", sizeof(int));
    int *ncount  = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("treescratch_build_ctr", sizeof(int));
    int *fail    = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>("treescratch_build_ctr", sizeof(int));
    if(!sz_curr || !sz_next || !ncount || !fail) {
        printf("gpu_topology_emit_bfs: counter alloc failed\n");
        if(sz_curr) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(sz_curr);
        if(sz_next) Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(sz_next);
        if(ncount)  Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ncount);
        if(fail)    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(fail);
        return 3;
    }
    *sz_curr = 0; *sz_next = 0; *ncount = start_node_index; *fail = 0;

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
        int parent_soa = parent_abs - max_part;
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

    if(*fail) {
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(sz_curr);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(sz_next);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ncount);
        int rc = *fail; Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(fail);
        printf("gpu_topology_emit_bfs: worklist overflow at init\n");
        return rc;
    }

    /* BFS loop: each iteration processes wl_curr, populates wl_next. */
    Kokkos::View<BfsItem*, MemSpace> wl_curr = wl_a;
    Kokkos::View<BfsItem*, MemSpace> wl_next = wl_b;
    int level_guard = 0;
    while(*sz_curr > 0 && *fail == 0 && level_guard < GIZMO_GPU_MORTON_MAX_DEPTH + 4) {
        int curr_size = *sz_curr;
        *sz_next = 0;

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
                int rrc = gpu_morton_split_8way_random_inplace(
                    sidx + w.range_first, range_count, id_of,
                    (uint64_t) w.parent_depth, child_starts);
                if(rrc < 0) {
                    /* Range too large for thread-local scratch (rare).
                     * Signal failure; would need a global-scratch path. */
                    Kokkos::atomic_fetch_max(fail, 5);
                    return;
                }
            } else {
                gpu_morton_split_8way(sidx, keys, w.range_first, w.range_last,
                                      w.parent_depth, child_starts);
            }

            for(int k = 0; k < 8; k++) {
                int rf = w.range_first + child_starts[k];
                int rl = w.range_first + child_starts[k+1];
                int cnt = rl - rf;
                int slot_value = -1;

                if(cnt == 1) {
                    /* Single particle: store its REAL P[] index directly as a leaf
                     * (subset builds map slot->particle; identity: slot==real). */
                    slot_value = stp ? stp[sidx[rf]] : sidx[rf];
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
                    soa_father[new_soa] = w.parent_soa + max_part;
                    long sb = (long)new_soa * 8;
                    for(int s = 0; s < 8; s++) {soa_suns[sb + s] = -1;}

                    slot_value = max_part + new_soa;

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
        level_guard++;
    }

    int rc = (*fail == 0) ? 0 : *fail;
    int new_total = *ncount;
    if(new_node_count_out) {*new_node_count_out = new_total;}

    /* After ping-pong swaps, sz_curr / sz_next still refer to the two
     * originally-allocated pointers (in some order); free both. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(sz_curr);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(sz_next);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ncount);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(fail);

    if(level_guard >= GIZMO_GPU_MORTON_MAX_DEPTH + 4 && rc == 0) {
        printf("gpu_topology_emit_bfs: BFS exceeded depth guard (level=%d) -- possible infinite recursion\n", level_guard);
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

extern "C" void gpu_topology_build_release(void)
{
    if(g_sorted_idx)       {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_sorted_idx);       g_sorted_idx       = NULL;}
    if(g_particle_topleaf) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_particle_topleaf); g_particle_topleaf = NULL;}
    if(g_topleaf_start)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_start);    g_topleaf_start    = NULL;}
    if(g_topleaf_count)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_count);    g_topleaf_count    = NULL;}
    if(g_topleaf_cursor)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_topleaf_cursor);   g_topleaf_cursor   = NULL;}
    if(g_slot_to_particle) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_slot_to_particle); g_slot_to_particle = NULL;}
    g_slot_map_active = 0;
    g_slot_cap = 0;
    g_npart_cap = 0;
    g_topleaf_cap = 0;
}


