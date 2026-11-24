/* gpu_force_update.cc
 *
 * GPU replacement for force_update_tree().
 * Propagates per-particle momentum kicks (P[i].dp) through the tree via three
 * GPU-accelerated stages + host-side MPI:
 *
 *   Stage 1  bring current the nodes the kick will touch: the chains above the
 *            kicked elements (claim pass + gpu_device_node_list_bring_current), or every
 *            node (gpu_force_drift_nodes) when the update is a large fraction of
 *            the rank (TreeUpdateFullSweep_ActiveFraction).
 *   Stage 2  gpu_force_kick_kernel      — per-active-particle Father-chain walk;
 *                                         atomic-accumulates dp into Extnodes[no].dp,
 *                                         atomic-maxes vmax, sets NODEHASBEENKICKED,
 *                                         fills UVM DomainList buffer.
 *   Stage 3  force_finish_kick_nodes()  — unchanged CPU code; does MPI Allgatherv
 *                                         of changed domain nodes and applies received
 *                                         kicks to ancestor chain (all UVM, CPU-safe).
 *
 * RT_SEPARATELY_TRACK_LUMPOS: rt_get_source_luminosity() is not GPU-callable;
 * rt_source_lum_dp is pre-computed on host into a UVM buffer before kernel launch.
 * DM_SCALARFIELD_SCREENING: dp_dm is computed in-kernel (Type != 0 check).
 *
 * DomainList is allocated as SharedSpace (UVM) so the GPU kernel writes and
 * force_finish_kick_nodes reads without copies.  The global DomainList pointer
 * is temporarily redirected to the UVM buffer for the duration of the call.
 *
 * P[i].dp zeroing: the kick kernel zeros Pp[i].dp for every active particle
 * it walks.  The arena is P itself on every build, so no host pass follows.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_error_check.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "../mesh/ghost_writeback.h"   /* ghost_get_num_local: the rank's own element count */
#include "forcetree.h"
#include "../core/timestep_functions.h"   /* dilation, for the motion bound; particle_radius_growth_rate */
#include "ags_functions.h"                /* particle_radius_drifts_with_divergence_P */
#include "../mesh/nlr_radius_policy.h"    /* force_hmax_per_type_radius */
#include "../declarations/gpu_dispatch_templates.h"   /* gizmo_gpu_default_space_is_host */


/* Atomic max for MyFloat via 64-bit CAS (MyFloat = double in GIZMO typedefs). */
static_assert(sizeof(MyFloat) == sizeof(uint64_t),
              "gpu_atomic_max_float: MyFloat must be 64-bit");
/* Atomic max for MyGravFloat, the SoA mirror's type.
 *
 * ⛔ gpu_atomic_max_myfloat below CANNOT be reused on soa->vmax: it static_asserts
 * MyFloat is 64-bit and does a 64-bit CAS, while MyGravFloat is `float` under
 * GRAVITY_MIXED_PRECISION (typedefs.h:46) and `double` otherwise. The
 * mismatch is invisible in the default build and breaks only in that config, which
 * is exactly the kind of trap a single-config test never finds.
 *
 * Kokkos::atomic_max is used where it exists for the type; the CAS loop is the
 * portable fallback and is correct for both widths because it compares in the
 * VALUE domain, not the bit domain (vmax >= 0 always -- it is a running max of
 * |velocity| -- so no negative-float ordering hazard arises). */
KOKKOS_INLINE_FUNCTION static void
gpu_atomic_max_gravfloat(MyGravFloat* addr, MyGravFloat val)
{
    MyGravFloat old = *addr;
    while(val > old) {
        const MyGravFloat prev = Kokkos::atomic_compare_exchange(addr, old, val);
        if(prev == old) {break;}
        old = prev;
    }
}

KOKKOS_INLINE_FUNCTION static void
gpu_atomic_max_myfloat(MyFloat* addr, MyFloat val)
{
    uint64_t val_bits, old_bits;
    memcpy(&val_bits, &val, sizeof(MyFloat));
    MyFloat old = *addr;
    while(val > old) {
        memcpy(&old_bits, &old, sizeof(MyFloat));
        uint64_t prev = Kokkos::atomic_compare_exchange(
            reinterpret_cast<uint64_t*>(addr), old_bits, val_bits);
        if(prev == old_bits) break;
        memcpy(&old, &prev, sizeof(MyFloat));
    }
}

/* Raise *addr to val if val is larger; 1 if it did.  The host's atomic_max_double, on the device. */
KOKKOS_INLINE_FUNCTION static int
gpu_atomic_raise_myfloat(MyFloat* addr, MyFloat val)
{
    uint64_t val_bits, old_bits;
    memcpy(&val_bits, &val, sizeof(MyFloat));
    MyFloat old = *addr;
    while(val > old) {
        memcpy(&old_bits, &old, sizeof(MyFloat));
        const uint64_t prev = Kokkos::atomic_compare_exchange(reinterpret_cast<uint64_t*>(addr), old_bits, val_bits);
        if(prev == old_bits) {return 1;}
        memcpy(&old, &prev, sizeof(MyFloat));
    }
    return 0;
}

/* =========================================================================
 * gpu_force_update_tree — drop-in GPU replacement for force_update_tree().
 * ========================================================================= */
extern "C" void gpu_force_update_tree(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    PRINT_STATUS("Kick-subroutine will prepare for dynamic update of tree (GPU)");

    GlobFlag++;
    DomainNumChanged = 0;

    /* One shared-space allocation carved into every buffer this call needs, rather
     * than one allocation per buffer: on a unified-memory device each allocation
     * carries page-registration cost, so the count matters more than the size.
     * Sized from the active list before the drift, since the drift only ever
     * shrinks how much of it is used. The widest type is placed first so the
     * following integer regions stay naturally aligned. */
    const int ntop = (NTopleaves > 0) ? NTopleaves : 1;
    const int n_active_cap = (int) ActiveParticleList.size();
    const int n_active_alloc = (n_active_cap > 0) ? n_active_cap : 1;
    const integertime ti_now = gizmo_host_ti_current();

    /* Stage 1 has two forms.  The kick below touches only the chains of parent nodes above the
     * elements whose kicks it propagates, so only those nodes have to be current before it runs:
     * bringing exactly them current costs O(chains), where sweeping every node costs O(tree) on
     * every call however few elements are active.  The full sweep is still taken when this rank's
     * update is a large fraction of its elements: then the chains reach much of the tree anyway,
     * and a full sweep also certifies the whole tree current for the walks that follow, which
     * otherwise each bring current whatever they open.  The fraction takes the larger of the
     * kicks propagated here (the step just closed) and the coming step's count, because the first
     * sizes this call's work and the second the work of the walks that would use the certificate.
     * A tree already current at this time (built at it, or swept) needs neither.
     *
     * The chain form also brings every top-level node current.  The top tree is shared with the
     * other ranks: after the kick, force_finish_kick_nodes adds their changes to it and drifts
     * each node it touches on the host.  A host drift at this time would mark the step as one in
     * which the host has advanced nodes, which turns the device sweep and the device receiver walk
     * away for the rest of it; with the whole top tree already current, those drifts do nothing. */
    const int tree_current = gpu_gravity_tree_nodes_current_at(ti_now) ? 1 : 0;
    const int n_local_elements = ghost_get_num_local();
    const long long n_update = (n_active_cap > NumForceUpdateAtSyncPoint) ? (long long) n_active_cap : (long long) NumForceUpdateAtSyncPoint;
    const double update_fraction = (double) n_update / (double) ((n_local_elements > 0) ? n_local_elements : 1);
    const int bring_chains_only = (!tree_current && update_fraction < All.TreeUpdateFullSweep_ActiveFraction) ? 1 : 0;
    /* The claim list: each node is claimed at most once, so it can never need more than the
     * local node count; below that it is sized from the active count times a depth that ordinary
     * trees stay under.  That depth is not a bound -- a longer chain overflows the list, which is
     * detected and answered by the full sweep -- so it decides only how often the cheap form is
     * used, never whether the result is right. */
    const size_t claim_depth = 48;
    const int n_top = bring_chains_only ? NTopnodes : 0;
    size_t claim_cap = 0;
    if(bring_chains_only) {
        const size_t by_chains = (size_t) n_active_cap * claim_depth + (size_t) n_top;
        claim_cap = (by_chains < (size_t) Numnodestree) ? by_chains : (size_t) Numnodestree;
    }
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    const size_t rt_bytes = (size_t) n_active_alloc * 3 * sizeof(MyDouble);
#else
    const size_t rt_bytes = 0;
#endif
    /* flags: claim cursor, claim overflow, a kicked node found not current; then the claim list and
     * the top-level node indices it starts from */
    const size_t int_bytes = ((size_t) ntop + 1 + (size_t) n_active_alloc + 3 + claim_cap + (size_t) n_top) * sizeof(int);
    char *scratch_dev = (char *) gizmo_gpu_alloc_shared(rt_bytes + int_bytes, "force_update_scratch");
    /* Refused scratch is handled the way a failed node drift already is below:
     * this rank reports zero changed nodes and still enters the all-rank
     * exchange, which is legal -- force_finish_kick_nodes only reads
     * DomainList on ranks whose own count is nonzero. */
    const bool scratch_ok = (scratch_dev != NULL);
    if(!scratch_ok) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "tree-force update: could not stage %d active particles and %d domain "
                 "slots (%.1f MB); the tree forces are not refreshed",
                 n_active_cap, ntop, (double)(rt_bytes + int_bytes) / (1024.0 * 1024.0));
        gizmo_request_controlled_stop(7732, msg, __FILE__, __LINE__, __FUNCTION__);
    }
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    MyDouble *rt_lum_dp_dev = (MyDouble *) scratch_dev;
#endif
    /* Offset only a base that exists: advancing a refused (NULL) base is undefined,
     * and a compiler may read it as proof the base is non-NULL and drop the
     * scratch_ok guards below. Refused scratch leaves these NULL, which no path
     * reads -- !scratch_ok forces num_active to 0 and exits through finish_mpi.
     * They are still declared here, ahead of that goto, which may not jump over
     * an initialization. */
    int *domain_list_dev  = scratch_ok ? (int *) (scratch_dev + rt_bytes) : NULL;
    int *domain_count_dev = scratch_ok ? domain_list_dev + ntop : NULL;
    int *active_dev       = scratch_ok ? domain_count_dev + 1 : NULL;
    int *flags_dev        = scratch_ok ? active_dev + n_active_alloc : NULL;
    int *claim_list_dev   = scratch_ok ? flags_dev + 3 : NULL;
    int *top_nodes_dev    = scratch_ok ? claim_list_dev + claim_cap : NULL;

    if(scratch_ok) {
        domain_count_dev[0] = 0;
        flags_dev[0] = flags_dev[1] = flags_dev[2] = 0;
        DomainList = domain_list_dev;   /* redirect global ptr to UVM buffer */
        /* Copy active-particle index list into its slice of the scratch buffer. */
        if(n_active_cap > 0) {memcpy(active_dev, ActiveParticleList.data(), n_active_cap * sizeof(int));}
        if(n_top > 0) {memcpy(top_nodes_dev, TopNodeNodeIndex, n_top * sizeof(int));}
    }

    /* Re-acquire particles arena (invalidated at end of gpu_gravtree_walk_primary).
     * On UVM systems this is a same-pointer re-registration (cheap). */
    gpu_particles_arena_set_site("gpu_force_update_domainlist");
    gpu_particles_arena_acquire(NumPart, P, CellP);

    /* Stage 1: bring the nodes the kick will touch current at Ti_Current -- the kick chains alone,
     * or every node (see the choice above).  Uses an out-of-line host accessor for the host-side
     * Ti_Current read. */
    /* Soft bad-stop on node-drift failure: flag it, then route through the
     * existing num_active<=0 -> finish_mpi path. This keeps the failing rank on
     * the SAME all-rank force_finish_kick_nodes() Allgatherv as its peers
     * (collective-symmetric, no deadlock), skips the update kernel on stale/
     * un-drifted node state, and drains at the next phase-boundary poll -- with
     * NO MPI_Abort. (A direct `goto finish_mpi` from here is ill-formed: it would
     * jump over the t_fut_drift_nodes initialization, which finish_mpi uses.) */
    int drift_rc = 0;
    if(scratch_ok && !tree_current) {
        if(bring_chains_only) {
            /* Claim each chain node once.  The lane that stamps a node first lists it and goes on
             * upward; a lane reaching a node already stamped stops, since the node and everything
             * above it is the stamper's.  The lanes past the active elements each claim one
             * top-level node the same way, so the top tree joins the list without duplicates.  A
             * fresh stamp value, so no earlier use of the flag can read as a claim, and the kick
             * below takes another for its own top-level claims. */
            GlobFlag++;
            const int gclaim = GlobFlag;
            const int claim_cap_i = (int) claim_cap;
            int            *Fa = Father;
            struct NODE    *No = Nodes;
            struct extNODE *Ex = Extnodes;
            int *claim_cursor = flags_dev, *claim_overflow = flags_dev + 1, *claim_list = claim_list_dev;
            const int *act = active_dev, *top = top_nodes_dev;
            const int n_chains = n_active_cap;
            Kokkos::parallel_for("gpu_force_claim_kick_chains", n_active_cap + n_top, KOKKOS_LAMBDA(const int idx) {
                if(idx >= n_chains) {
                    const int no = top[idx - n_chains];
                    if(no < 0 || Kokkos::atomic_exchange(&Ex[no].Flag, gclaim) == gclaim) {return;}
                    const int slot = Kokkos::atomic_fetch_add(claim_cursor, 1);
                    if(slot < claim_cap_i) {claim_list[slot] = no;}
                    else {Kokkos::atomic_store(claim_overflow, 1);}
                    return;
                }
                int no = Fa[act[idx]];
                while(no >= 0) {
                    if(Kokkos::atomic_exchange(&Ex[no].Flag, gclaim) == gclaim) {break;}
                    const int slot = Kokkos::atomic_fetch_add(claim_cursor, 1);
                    if(slot < claim_cap_i) {claim_list[slot] = no;}
                    else {Kokkos::atomic_store(claim_overflow, 1);}
                    if(No[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {break;}
                    no = No[no].u.d.father;
                }
            });
            Kokkos::fence();
            gizmo_gpu_check_last_error("gpu_force_claim_kick_chains", n_active_cap + n_top);
            /* A chain that outran the list, or a list the drift could not take, is answered by the
               full sweep, which brings every node the list would have and more. */
            if(flags_dev[1] || gpu_device_node_list_bring_current(claim_list_dev, flags_dev[0], All.TreeNodeIndexBase,
                                                           gpu_gravity_tree_capacity(), ti_now) != 0) {
                drift_rc = gpu_force_drift_nodes(ti_now);
            }
        } else {
            drift_rc = gpu_force_drift_nodes(ti_now);
        }
    }
    const bool drift_ok = scratch_ok && (drift_rc == 0);
    if(scratch_ok && !drift_ok) { endrun(929703); }
    double t_fut_drift_nodes = my_second();

    /* same count the scratch was sized from, not a second read of the list */
    const int num_active = drift_ok ? n_active_cap : 0;
    if(num_active <= 0) {
        DomainNumChanged = 0;
        goto finish_mpi;
    }

    {
        struct particle_data *Pp   = gpu_particles_arena_P();
        struct gas_cell_data *Cp   = gpu_particles_arena_CellP();
        int                  *Fa   = Father;    /* UVM pointer */
        struct NODE          *No   = Nodes;     /* UVM shifted pointer */
        struct extNODE       *Ex   = Extnodes;  /* UVM pointer */
        /* The walk's vmax mirror, captured by value for the kernel. Bounded by the
           mirror that EXISTS (MaxNodes + this rank's installed foreign range), not
           by the run-wide index ceiling. */
        struct gpu_gravity_tree_soa_t *soa_u = gpu_gravity_tree_soa();
        MyGravFloat          *soa_vmax   = (soa_u ? soa_u->vmax : NULL);
        const int             tree_base_soa = All.TreeNodeIndexBase;
        const int             soa_vmax_n    = gpu_gravity_tree_capacity();   /* the ALLOCATION, not the index range */
        unsigned int         *soa_bitflags  = (soa_u ? soa_u->bitflags : NULL);
        integertime          *soa_node_ti   = (soa_u ? soa_u->node_ti : NULL);
        int                  *kick_not_current = flags_dev + 2;
        const unsigned int    kicked_bit    = (1u << BITFLAG_NODEHASBEENKICKED);
        if(bring_chains_only) {GlobFlag++;}   /* the claim pass used the previous value */
        int                   gflag = GlobFlag;
        integertime           ti_cur = ti_now;   /* captured by value into the device lambda */

#ifdef RT_SEPARATELY_TRACK_LUMPOS
        /* Pre-compute rt_source_lum_dp per active particle on CPU (not GPU-callable). */
        for(int idx = 0; idx < num_active; idx++) {
            int i = ActiveParticleList[idx];
            double lum[N_RT_FREQ_BINS];
            int active_check = rt_get_source_luminosity(i, -1, lum, P, CellP);
            Vec3<MyDouble> dp_i = P[i].dp;
            Vec3<MyDouble> rt_dp = active_check ? dp_i : Vec3<MyDouble>{};
            rt_lum_dp_dev[idx*3+0] = rt_dp[0];
            rt_lum_dp_dev[idx*3+1] = rt_dp[1];
            rt_lum_dp_dev[idx*3+2] = rt_dp[2];
        }
#endif

        /* Stage 2: GPU kick kernel — per-active-particle Father-chain walk. */
        Kokkos::parallel_for("gpu_force_kick", num_active,
            KOKKOS_LAMBDA(const int idx) {
                const int i = active_dev[idx];

                /* Read and zero P[i].dp (the arena is P itself on every build). */
                Vec3<MyDouble> dp = Pp[i].dp;
                Pp[i].dp = Vec3<MyDouble>{};

                /* How fast this particle can now move, for the ancestors' widening bound. */
                const MyFloat vmax = (MyFloat) particle_motion_speed_bound(i, Pp, Cp);

#ifdef RT_SEPARATELY_TRACK_LUMPOS
                Vec3<MyDouble> rt_dp = { rt_lum_dp_dev[idx*3+0],
                                          rt_lum_dp_dev[idx*3+1],
                                          rt_lum_dp_dev[idx*3+2] };
#endif
#ifdef DM_SCALARFIELD_SCREENING
                Vec3<MyDouble> dp_dm = (Pp[i].Type != 0) ? dp : Vec3<MyDouble>{};
#endif
#ifdef SINK_NODE_MOTION_TRACKED
                /* same predicate as the host mirror in force_kick_node, and the same one the
                   moment builders use to decide what sink_mass/sink_pos sum over */
                Vec3<MyDouble> sink_dp = (Pp[i].Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES) ? dp : Vec3<MyDouble>{};
#endif

                /* Walk Father chain, accumulating kicks. */
                int no = Fa[i];
                while(no >= 0) {
                    /* Every node this walk kicks must stand at the kick time, in its canonical copy
                       and in the mirror the device walks read: a kick added to a node behind it
                       would later be folded in over the wrong interval. Stage 1 guarantees it;
                       this is the check that it did. */
                    {
                        const int kk_now = no - tree_base_soa;
                        if(No[no].Ti_current != ti_cur ||
                           !soa_node_ti || kk_now < 0 || kk_now >= soa_vmax_n || soa_node_ti[kk_now] != ti_cur) {
                            Kokkos::atomic_store(kick_not_current, 1);
                            break;
                        }
                    }
                    /* dp accumulation (atomic since multiple particles share ancestors). */
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].dp[k], dp[k]);
                    }
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].rt_source_lum_dp[k], rt_dp[k]);
                    }
#endif
#ifdef DM_SCALARFIELD_SCREENING
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].dp_dm[k], dp_dm[k]);
                    }
#endif
#ifdef SINK_NODE_MOTION_TRACKED
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].sink_dp[k], sink_dp[k]);
                    }
#endif
                    gpu_atomic_max_myfloat(&Ex[no].vmax, vmax);
                    /* The walk's mirror, raised with the AoS. This is the DEVICE kick
                       route; the host route raises it in force_kick_node. Both then
                       reach force_finish_kick_nodes, which raises the merged
                       top-level value. */
                    if(soa_vmax) {
                        const int kk_soa = no - tree_base_soa;
                        if(kk_soa >= 0 && kk_soa < soa_vmax_n) {
                            gpu_atomic_max_gravfloat(&soa_vmax[kk_soa], (MyGravFloat) vmax);
                            /* and that the node holds a pending kick; the bit only rises in this
                               phase, so a node already marked needs no atomic */
                            if(soa_bitflags && !(soa_bitflags[kk_soa] & kicked_bit)) {
                                Kokkos::atomic_fetch_or(&soa_bitflags[kk_soa], kicked_bit);
                            }
                        }
                    }
                    Kokkos::atomic_fetch_or(&No[no].u.d.bitflags,
                                            (unsigned int)(1 << BITFLAG_NODEHASBEENKICKED));
                    Ex[no].Ti_lastkicked = ti_cur;

                    if(No[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {
                        /* Deduplicate: claim with atomic_exchange on Flag. */
                        int old_flag = Kokkos::atomic_exchange(&Ex[no].Flag, gflag);
                        if(old_flag != gflag) {
                            int slot = Kokkos::atomic_fetch_add(domain_count_dev, 1);
                            if(slot < ntop) { domain_list_dev[slot] = no; }
                        }
                        break;
                    }
                    no = No[no].u.d.father;
                }
            });
        Kokkos::fence();
        gizmo_gpu_check_last_error("gpu_force_kick", num_active);
        if(flags_dev[2]) {
            printf("Task=%d gpu_force_update_tree: a node on a kick chain was not current at the kick time; its kick would be folded over the wrong interval\n", ThisTask);
            fflush(stdout);
            endrun(929705);
        }

        /* The kernel clamps its writes to the DomainList slice, so a count past
         * that slice would hand the exchange indices it never wrote. One claim
         * per distinct top-level node makes this unreachable; say so loudly
         * rather than pass on a count the buffer does not back. */
        DomainNumChanged = domain_count_dev[0];
        if(DomainNumChanged > ntop) {
            printf("Task=%d gpu_force_update_tree: %d changed top-level nodes exceeds the %d-entry list\n",
                   ThisTask, DomainNumChanged, ntop);
            fflush(stdout);
            DomainNumChanged = ntop;
            endrun(929704);
        }
    }

finish_mpi:
    /* Stage 3: host-side MPI Allgatherv + ancestor apply.
     * force_finish_kick_nodes reads DomainList (now UVM), DomainNumChanged, and
     * writes to Extnodes/Nodes (UVM) — all coherent after Kokkos::fence() above. */
    force_finish_kick_nodes();

    /* Restore global DomainList to NULL and release the one scratch allocation. */
    DomainList = NULL;
    if(scratch_dev) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(scratch_dev);}

    PRINT_STATUS(" ..Tree has been updated dynamically (GPU)");

}




/* =========================================================================
 * gpu_force_update_hmax_local -- the local part of force_update_hmax on the device.
 *
 * The host form drifts each active element's chain of parent nodes up to the first node already current, then
 * raises the chain's kernel-length bands (scalar hmax, divVmax, the six per-type bands) while they grow, listing
 * each top-level node reached.  Here the same three steps run on the device, so the host writes none of those
 * node pages between device updates:
 *   1. claim: each chain, up to its first node current at this time, is listed once (a node stamped by another
 *      lane is that lane's from there up) -- the set the host's serial drift visits;
 *   2. the listed nodes brought current by the device drift (gpu_device_node_list_bring_current), the same
 *      per-node arithmetic as the host's force_drift_node;
 *   3. the band raise with atomics, the changed top-level nodes listed once (an atomic stamp), and the listed
 *      nodes' mirrors published again, so they carry the raised bands as the host route's later publication would.
 * The bands are maxima, so the result does not depend on the order the lanes run in; the changed-node list is a
 * set, and every reader of it (the pack, the exchange, the apply) is order-independent.
 * ========================================================================= */
static char *g_hmax_update_scratch = NULL;

extern "C" int gpu_force_update_hmax_available(void)
{
    return gizmo_gpu_default_space_is_host() ? 0 : 1;
}

extern "C" void gpu_force_update_hmax_release(void)
{
    DomainList = NULL;
    if(g_hmax_update_scratch) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(g_hmax_update_scratch); g_hmax_update_scratch = NULL;}
}

extern "C" int gpu_force_update_hmax_local(struct gpu_hmax_reach_record *reach_records)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();

    /* the device drift and the mirror it publishes need the walk's mirror of the tree */
    const struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->len) {return 1;}
    const int n_active = (int) ActiveParticleList.size();
    const int ntop = (NTopleaves > 0) ? NTopleaves : 1;
    const int n_alloc = (n_active > 0) ? n_active : 1;
    const integertime ti_now = gizmo_host_ti_current();
    /* The claim list never needs more than the local node count; below it, the active count times a depth ordinary
       trees stay under.  A longer chain overflows it, and the update then takes the host form, before any node has
       changed. */
    const size_t claim_depth = 48;
    size_t claim_cap = (size_t) n_alloc * claim_depth;
    if(claim_cap > (size_t) Numnodestree) {claim_cap = (size_t) Numnodestree;}
    if(claim_cap < 1) {claim_cap = 1;}
    const size_t rec_bytes = reach_records ? (size_t) n_alloc * sizeof(struct gpu_hmax_reach_record) : 0;
    /* changed-node list, its count, the active list, claim cursor + overflow flag, the claim list */
    const size_t int_bytes = ((size_t) ntop + 1 + (size_t) n_alloc + 2 + claim_cap) * sizeof(int);
    char *scratch = (char *) gizmo_gpu_alloc_shared(rec_bytes + int_bytes, "hmax_update_scratch");
    if(!scratch) {return 1;}
    struct gpu_hmax_reach_record *records_dev = (struct gpu_hmax_reach_record *) scratch;
    int *domain_list_dev  = (int *) (scratch + rec_bytes);
    int *domain_count_dev = domain_list_dev + ntop;
    int *active_dev       = domain_count_dev + 1;
    int *flags_dev        = active_dev + n_alloc;
    int *claim_list_dev   = flags_dev + 2;
    domain_count_dev[0] = 0;
    flags_dev[0] = flags_dev[1] = 0;
    if(n_active > 0) {memcpy(active_dev, ActiveParticleList.data(), (size_t) n_active * sizeof(int));}

    gpu_particles_arena_set_site("gpu_force_update_hmax");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *Pp = gpu_particles_arena_P();
    int            *Fa = Father;
    struct NODE    *No = Nodes;
    struct extNODE *Ex = Extnodes;

    /* 1. claim the chains */
    GlobFlag++;
    const int gclaim = GlobFlag;
    const int claim_cap_i = (int) claim_cap;
    {
        int *claim_cursor = flags_dev, *claim_overflow = flags_dev + 1, *claim_list = claim_list_dev;
        const int *act = active_dev;
        const integertime ti = ti_now;
        Kokkos::parallel_for("gpu_hmax_claim_chains", n_active, KOKKOS_LAMBDA(const int idx) {
            const int i = act[idx];
            if(!(Pp[i].Mass > 0)) {return;}
            int no = Fa[i];
            while(no >= 0) {
                if(No[no].Ti_current == ti) {break;}
                if(Kokkos::atomic_exchange(&Ex[no].Flag, gclaim) == gclaim) {break;}
                const int slot = Kokkos::atomic_fetch_add(claim_cursor, 1);
                if(slot < claim_cap_i) {claim_list[slot] = no;}
                else {Kokkos::atomic_store(claim_overflow, 1);}
                no = No[no].u.d.father;
            }
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("gpu_hmax_claim_chains", n_active);
    }
    if(flags_dev[1]) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(scratch); return 1;}   /* nothing has changed yet */
    g_hmax_update_scratch = scratch;
    DomainList = domain_list_dev;
    DomainNumChanged = 0;
    const int n_claimed = flags_dev[0];

    /* 2. bring the claimed nodes current.  A failure past this point has changed nodes, so it is not answered by the
          host form: it stops the run, and this rank still joins the exchange with nothing to send. */
    if(gpu_device_node_list_bring_current(claim_list_dev, n_claimed, All.TreeNodeIndexBase, gpu_gravity_tree_capacity(), ti_now) != 0)
    {
        printf("Task=%d force_update_hmax (device): the active chains could not be brought current\n", ThisTask); fflush(stdout);
        endrun(929711);
        return 0;
    }

    /* 3. raise the bands */
    GlobFlag++;
    const int gband = GlobFlag;
    {
        const int *act = active_dev;
        int *dlist = domain_list_dev, *dcount = domain_count_dev;
        struct gpu_hmax_reach_record *rec = reach_records ? records_dev : NULL;
        const double max_kernel_radius = (double) All.MaxKernelRadius;
        Kokkos::parallel_for("gpu_hmax_raise_bands", Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, n_active, Kokkos::ChunkSize(1)),
                             KOKKOS_LAMBDA(const int idx) {
            const int i = act[idx];
            if(!(Pp[i].Mass > 0)) {if(rec) {rec[idx].valid = 0;} return;}
            const double divVel = particle_radius_growth_rate(i, Pp);
            const int per_type_band = (int) Pp[i].Type;
            const double per_type_htmp = force_hmax_per_type_radius(Pp[i], max_kernel_radius);
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            const int scalar_eligible = 1;
            const double kernrad_temp = (Pp[i].Type == 0) ? (double) Pp[i].KernelRadius : (double) Pp[i].AGS_KernelRadius;
            const double htmp = DMIN(kernrad_temp, max_kernel_radius);
#else
            const int scalar_eligible = (Pp[i].Type == 0);
            const double htmp = (Pp[i].Type == 0) ? DMIN((double) Pp[i].KernelRadius, max_kernel_radius) : 0.0;
#endif
            const int divv_eligible = particle_radius_drifts_with_divergence_P(i, Pp);
            int no = Fa[i];
            while(no >= 0) {
                /* as the host form: climb only while something grows */
                const int per_type_grew = gpu_atomic_raise_myfloat(&Ex[no].hmax_per_type[per_type_band], (MyFloat) per_type_htmp);
                int scalar_grew = scalar_eligible ? gpu_atomic_raise_myfloat(&Ex[no].hmax, (MyFloat) htmp) : 0;
                if(divv_eligible && gpu_atomic_raise_myfloat(&Ex[no].divVmax, (MyFloat) divVel)) {scalar_grew = 1;}
                if(!(scalar_grew || per_type_grew)) {break;}
                if(No[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {
                    if(Kokkos::atomic_exchange(&Ex[no].Flag, gband) != gband) {
                        const int slot = Kokkos::atomic_fetch_add(dcount, 1);
                        if(slot < ntop) {dlist[slot] = no;}
                    }
                    break;
                }
                no = No[no].u.d.father;
            }
            if(rec) {
                int leaf = Fa[i];
                while(leaf >= 0 && !(No[leaf].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))) {leaf = No[leaf].u.d.father;}
                rec[idx].reach = per_type_htmp;
                rec[idx].growth = divv_eligible ? divVel : 0.0;
                rec[idx].leaf_node = leaf;
                rec[idx].valid = 1;
            }
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("gpu_hmax_raise_bands", n_active);
    }
    if(gpu_device_node_list_publish(claim_list_dev, n_claimed, All.TreeNodeIndexBase, gpu_gravity_tree_capacity()) != 0)
    {
        printf("Task=%d force_update_hmax (device): the raised nodes' mirrors could not be published\n", ThisTask); fflush(stdout);
        endrun(929712);
    }

    DomainNumChanged = domain_count_dev[0];
    if(DomainNumChanged > ntop)
    {
        printf("Task=%d force_update_hmax (device): %d changed top-level nodes exceeds the %d-entry list\n", ThisTask, DomainNumChanged, ntop);
        fflush(stdout);
        DomainNumChanged = ntop;
        endrun(929713);
    }
    /* the records, one contiguous copy to the host */
    if(reach_records && n_active > 0)
    {
        Kokkos::View<struct gpu_hmax_reach_record *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> to(reach_records, (size_t) n_active);
        Kokkos::View<const struct gpu_hmax_reach_record *, GIZMO_KOKKOS_SHARED_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> from(records_dev, (size_t) n_active);
        Kokkos::deep_copy(to, from);
    }
    return 0;
}
