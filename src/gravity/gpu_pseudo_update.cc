/* gpu_pseudo_update.cc — Step 13 Phase 6.7
 *
 * GPU/SoA-native replacements for the three CPU end-game stages in
 * force_treebuild and force_refresh_node_moments.  See
 * gpu_pseudo_update.h for the per-function contract.
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
#include "gpu_gravity_tree.h"
#include "gpu_pseudo_update.h"
#include "forcetree.h"
#include "gravtree_moment_kernel.h"


/* ===================================================================== */
/* Phase 6.7a — gpu_force_flag_localnodes                                */
/* ===================================================================== */

extern "C" int gpu_force_flag_localnodes(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(NTopleaves <= 0) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->bitflags || !soa->father) {
        printf("gpu_force_flag_localnodes: SoA not ready\n");
        return 1;
    }

    int MaxPart = All.MaxPart;
    unsigned int *bitflags_soa = soa->bitflags;
    int          *father_soa   = soa->father;

    /* Copy DomainNodeIndex[0..NTopleaves) to device-side storage.
     * NTopleaves is small (≤ MaxTopNodes, typically a few hundred to a few
     * thousand) so the copy cost is negligible. */
    int n_leaves = NTopleaves;
    using ExSp = Kokkos::DefaultExecutionSpace;
    using MemSp = ExSp::memory_space;
    Kokkos::View<int*, Kokkos::HostSpace> dom_h(
        Kokkos::ViewAllocateWithoutInitializing("ps_dom_h"), n_leaves);
    for(int i = 0; i < n_leaves; i++) {dom_h[i] = DomainNodeIndex[i];}
    Kokkos::View<int*, MemSp> dom_d(
        Kokkos::ViewAllocateWithoutInitializing("ps_dom_d"), n_leaves);
    Kokkos::deep_copy(dom_d, dom_h);

    /* Collect DomainNodeIndex values for topleaves owned by ThisTask.
     * These are exactly the leaves over which the DEPENDS_ON_LOCAL_ELEMENT
     * walk should start. */
    int n_local = 0;
    for(int m = 0; m < MULTIPLEDOMAINS; m++) {
        int s = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
        int e = DomainEndList[ThisTask * MULTIPLEDOMAINS + m];
        if(e >= s) {n_local += e - s + 1;}
    }
    Kokkos::View<int*, MemSp> local_d;
    if(n_local > 0) {
        Kokkos::View<int*, Kokkos::HostSpace> local_h(
            Kokkos::ViewAllocateWithoutInitializing("ps_local_h"), n_local);
        int idx = 0;
        for(int m = 0; m < MULTIPLEDOMAINS; m++) {
            for(int i = DomainStartList[ThisTask * MULTIPLEDOMAINS + m];
                    i <= DomainEndList[ThisTask * MULTIPLEDOMAINS + m]; i++) {
                local_h[idx++] = DomainNodeIndex[i];
            }
        }
        local_d = Kokkos::View<int*, MemSp>(
            Kokkos::ViewAllocateWithoutInitializing("ps_local_d"), n_local);
        Kokkos::deep_copy(local_d, local_h);
    }

    constexpr unsigned int TOPLEVEL_BIT = (1u << BITFLAG_TOPLEVEL);
    constexpr unsigned int INTERNAL_BIT = (1u << BITFLAG_INTERNAL_TOPLEVEL);
    constexpr unsigned int DEPENDS_BIT  = (1u << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT);

    /* Pass 1: BITFLAG_TOPLEVEL — walk up from each DomainNodeIndex[i].
     * Mirrors CPU: while(no >= 0) { if(already_set) break; set; no = father; }
     * father_soa[k] stores the ABSOLUTE parent index (>= MaxPart) or -1 for
     * the root, so `k_abs < MaxPart` is the termination condition. */
    Kokkos::parallel_for("flag_toplevel", n_leaves, KOKKOS_LAMBDA(int i) {
        int k_abs = dom_d(i);
        while(k_abs >= MaxPart) {
            int k = k_abs - MaxPart;
            unsigned int old = Kokkos::atomic_fetch_or(&bitflags_soa[k], TOPLEVEL_BIT);
            if(old & TOPLEVEL_BIT) {break;}   /* all ancestors already set */
            k_abs = father_soa[k];
        }
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("flag_toplevel", n_leaves);

    /* Pass 2: BITFLAG_INTERNAL_TOPLEVEL — walk from *parent* of each
     * DomainNodeIndex[i] to root.  Marks all strict ancestors of topleaves
     * as internal-toplevel.  Topleaves themselves get only TOPLEVEL (pass 1),
     * not INTERNAL_TOPLEVEL. */
    Kokkos::parallel_for("flag_internal_toplevel", n_leaves, KOKKOS_LAMBDA(int i) {
        int k_abs = dom_d(i);
        if(k_abs < MaxPart) {return;}
        k_abs = father_soa[k_abs - MaxPart];   /* start one level above topleaf */
        while(k_abs >= MaxPart) {
            int k = k_abs - MaxPart;
            unsigned int old = Kokkos::atomic_fetch_or(&bitflags_soa[k], INTERNAL_BIT);
            if(old & INTERNAL_BIT) {break;}
            k_abs = father_soa[k];
        }
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("flag_internal_toplevel", n_leaves);

    /* Pass 3: BITFLAG_DEPENDS_ON_LOCAL_ELEMENT — walk from each topleaf
     * owned by ThisTask to root. */
    if(n_local > 0) {
        Kokkos::parallel_for("flag_depends_local", n_local, KOKKOS_LAMBDA(int j) {
            int k_abs = local_d(j);
            while(k_abs >= MaxPart) {
                int k = k_abs - MaxPart;
                unsigned int old = Kokkos::atomic_fetch_or(&bitflags_soa[k], DEPENDS_BIT);
                if(old & DEPENDS_BIT) {break;}
                k_abs = father_soa[k];
            }
        });
        Kokkos::fence();
        gizmo_gpu_check_last_error("flag_depends_local", n_local);
    }

    /* AoS writeback: mirror bitflags_soa[0..NTopnodes) → Nodes[].u.d.bitflags.
     * Required so that force_exchange_pseudodata (still CPU in 6.7a/b) reads
     * the correct bitflags from AoS.  Inside-topleaf bitflags were already
     * written by gpu_moment_writeback_to_aos at the end of gpu_moment_refresh;
     * only the topnode range [0, NTopnodes) needs patching here. */
    for(int k = 0; k < NTopnodes; k++) {
        Nodes[All.MaxPart + k].u.d.bitflags = bitflags_soa[k];
    }

    return 0;
}

/* ===================================================================== */
/* Phase 6.7b — gpu_scatter_pseudo_to_soa                                */
/* ===================================================================== */

extern "C" int gpu_scatter_pseudo_to_soa(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(NTopleaves <= 0) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_scatter_pseudo_to_soa: SoA null\n"); return 1;}

    int MaxPart = All.MaxPart;

    /* Copy the just-exchanged AoS foreign pseudo moments → SoA for all
     * foreign-rank topleaves.  This is O(NTopleaves) and runs on the host;
     * the SoA lives in UVM so host writes are visible after a Kokkos::fence.
     * Local (ThisTask) topleaves were already correct in SoA from
     * gpu_moment_refresh; we skip them here. */
    for(int ta = 0; ta < NTask; ta++) {
        if(ta == ThisTask) {continue;}
        for(int m = 0; m < MULTIPLEDOMAINS; m++) {
            for(int i = DomainStartList[ta * MULTIPLEDOMAINS + m];
                    i <= DomainEndList[ta * MULTIPLEDOMAINS + m]; i++) {
                int no = DomainNodeIndex[i];
                int k  = no - MaxPart;
                /* Scalar moment fields */
                soa->mass[k]    = (MyGravFloat) Nodes[no].u.d.mass;
                soa->N_part[k]  = Nodes[no].N_part;
                soa->maxsoft[k] = (MyGravFloat) Nodes[no].maxsoft;
                soa->hmax[k]    = (MyGravFloat) Extnodes[no].hmax;
                soa->vmax[k]    = (MyGravFloat) Extnodes[no].vmax;
                soa->divVmax[k] = (MyGravFloat) Extnodes[no].divVmax;
                /* Merge the MULTIPLEPARTICLES bit from the exchanged data;
                 * preserve all topology bits already set by 6.7a. */
                soa->bitflags[k] = (soa->bitflags[k] & ~(1u << BITFLAG_MULTIPLEPARTICLES))
                                 | (Nodes[no].u.d.bitflags & (1u << BITFLAG_MULTIPLEPARTICLES));
                /* Vec3 moment fields */
                soa->s[k]        = {(MyGravFloat) Nodes[no].u.d.s[0],
                                    (MyGravFloat) Nodes[no].u.d.s[1],
                                    (MyGravFloat) Nodes[no].u.d.s[2]};
                soa->node_vs[k]  = {(MyGravFloat) Extnodes[no].vs[0],
                                    (MyGravFloat) Extnodes[no].vs[1],
                                    (MyGravFloat) Extnodes[no].vs[2]};
                /* Optional payload fields */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
                soa->gasmass[k]  = (MyGravFloat) Nodes[no].gasmass;
#endif
#ifdef RT_USE_GRAVTREE
                for(int b = 0; b < N_RT_FREQ_BINS; b++) {
                    soa->stellar_lum[(long)k * N_RT_FREQ_BINS + b] =
                        (MyGravFloat) Nodes[no].stellar_lum[b];
                }
#ifdef CHIMES_STELLAR_FLUXES
                for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
                    soa->chimes_stellar_lum_G0 [(long)k * CHIMES_LOCAL_UV_NBINS + b] =
                        Nodes[no].chimes_stellar_lum_G0[b];
                    soa->chimes_stellar_lum_ion[(long)k * CHIMES_LOCAL_UV_NBINS + b] =
                        Nodes[no].chimes_stellar_lum_ion[b];
                }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                soa->rt_source_lum_s[k]  = {(MyGravFloat) Nodes[no].rt_source_lum_s[0],
                                             (MyGravFloat) Nodes[no].rt_source_lum_s[1],
                                             (MyGravFloat) Nodes[no].rt_source_lum_s[2]};
                soa->rt_source_lum_vs[k] = {(MyGravFloat) Extnodes[no].rt_source_lum_vs[0],
                                             (MyGravFloat) Extnodes[no].rt_source_lum_vs[1],
                                             (MyGravFloat) Extnodes[no].rt_source_lum_vs[2]};
#endif
#ifdef SINK_PHOTONMOMENTUM
                soa->sink_lum[k]      = (MyGravFloat) Nodes[no].sink_lum;
                soa->sink_lum_grad[k] = {(MyGravFloat) Nodes[no].sink_lum_grad[0],
                                          (MyGravFloat) Nodes[no].sink_lum_grad[1],
                                          (MyGravFloat) Nodes[no].sink_lum_grad[2]};
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                soa->cr_injection[k] = (MyGravFloat) Nodes[no].cr_injection;
#endif
#ifdef SINK_CALC_DISTANCES
                soa->sink_mass[k] = (MyGravFloat) Nodes[no].sink_mass;
                soa->sink_pos[k]  = {(MyGravFloat) Nodes[no].sink_pos[0],
                                      (MyGravFloat) Nodes[no].sink_pos[1],
                                      (MyGravFloat) Nodes[no].sink_pos[2]};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
                soa->N_SINK[k]    = Nodes[no].N_SINK;
                soa->sink_vel[k]  = {(MyGravFloat) Nodes[no].sink_vel[0],
                                      (MyGravFloat) Nodes[no].sink_vel[1],
                                      (MyGravFloat) Nodes[no].sink_vel[2]};
#ifdef SPECIAL_POINT_MOTION
                soa->sink_acc[k]  = {(MyGravFloat) Nodes[no].sink_acc[0],
                                      (MyGravFloat) Nodes[no].sink_acc[1],
                                      (MyGravFloat) Nodes[no].sink_acc[2]};
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
                soa->MaxFeedbackVel[k] = (MyGravFloat) Nodes[no].MaxFeedbackVel;
#endif
#endif
#endif
#ifdef DM_SCALARFIELD_SCREENING
                soa->mass_dm[k]  = (MyGravFloat) Nodes[no].mass_dm;
                soa->s_dm[k]     = {(MyGravFloat) Nodes[no].s_dm[0],
                                     (MyGravFloat) Nodes[no].s_dm[1],
                                     (MyGravFloat) Nodes[no].s_dm[2]};
                soa->vs_dm[k]    = {(MyGravFloat) Extnodes[no].vs_dm[0],
                                     (MyGravFloat) Extnodes[no].vs_dm[1],
                                     (MyGravFloat) Extnodes[no].vs_dm[2]};
#endif
            }
        }
    }
    return 0;
}

/* ===================================================================== */
/* Phase 6.7c — gpu_topnode_moment_resum                                 */
/* ===================================================================== */

/* Read one (already finalized, i.e. normalized) child topnode's moments out of the SoA into the
 * shared moment kernel's by-value accumulator, casting to the plain MyFloat venue type.  Mirrors the
 * per-child SoA reads the accumulate loop used to do inline; moment_source_from_child_normalized then
 * re-weights the COM-vector payloads by the child's own mass / luminosity / sink-mass. */
static moment_node_accum<MyFloat> topnode_child_accum_(const struct gpu_gravity_tree_soa_t *soa, int pk)
{
    moment_node_accum<MyFloat> c = {};
    c.mass    = (MyFloat) soa->mass[pk];
    c.s       = Vec3<MyFloat>{(MyFloat)soa->s[pk][0], (MyFloat)soa->s[pk][1], (MyFloat)soa->s[pk][2]};
    c.vs      = Vec3<MyFloat>{(MyFloat)soa->node_vs[pk][0], (MyFloat)soa->node_vs[pk][1], (MyFloat)soa->node_vs[pk][2]};
    c.Npart   = soa->N_part[pk];
    c.hmax    = (MyFloat) soa->hmax[pk];
    c.vmax    = (MyFloat) soa->vmax[pk];
    c.divVmax = (MyFloat) soa->divVmax[pk];
    c.maxsoft = (MyFloat) soa->maxsoft[pk];
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    c.gasmass = (MyFloat) soa->gasmass[pk];
#endif
#ifdef RT_USE_GRAVTREE
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {c.stellar_lum[b] = (MyFloat) soa->stellar_lum[(long)pk * N_RT_FREQ_BINS + b];}
#ifdef CHIMES_STELLAR_FLUXES
    for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
        c.chimes_G0 [b] = soa->chimes_stellar_lum_G0 [(long)pk * CHIMES_LOCAL_UV_NBINS + b];
        c.chimes_ion[b] = soa->chimes_stellar_lum_ion[(long)pk * CHIMES_LOCAL_UV_NBINS + b];
    }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    c.rt_s  = Vec3<MyFloat>{(MyFloat)soa->rt_source_lum_s[pk][0],  (MyFloat)soa->rt_source_lum_s[pk][1],  (MyFloat)soa->rt_source_lum_s[pk][2]};
    c.rt_vs = Vec3<MyFloat>{(MyFloat)soa->rt_source_lum_vs[pk][0], (MyFloat)soa->rt_source_lum_vs[pk][1], (MyFloat)soa->rt_source_lum_vs[pk][2]};
#endif
#ifdef SINK_PHOTONMOMENTUM
    c.sink_lum      = (MyFloat) soa->sink_lum[pk];
    c.sink_lum_grad = Vec3<MyFloat>{(MyFloat)soa->sink_lum_grad[pk][0], (MyFloat)soa->sink_lum_grad[pk][1], (MyFloat)soa->sink_lum_grad[pk][2]};
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    c.cr_inject = (double) soa->cr_injection[pk];
#endif
#ifdef SINK_CALC_DISTANCES
    c.sink_mass = (MyFloat) soa->sink_mass[pk];
    c.sink_pos  = Vec3<MyFloat>{(MyFloat)soa->sink_pos[pk][0], (MyFloat)soa->sink_pos[pk][1], (MyFloat)soa->sink_pos[pk][2]};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    c.N_SINK   = soa->N_SINK[pk];
    c.sink_vel = Vec3<MyFloat>{(MyFloat)soa->sink_vel[pk][0], (MyFloat)soa->sink_vel[pk][1], (MyFloat)soa->sink_vel[pk][2]};
#endif
#if defined(SPECIAL_POINT_MOTION)
    c.sink_acc = Vec3<MyFloat>{(MyFloat)soa->sink_acc[pk][0], (MyFloat)soa->sink_acc[pk][1], (MyFloat)soa->sink_acc[pk][2]};
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    c.max_fbvel = (MyFloat) soa->MaxFeedbackVel[pk];
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    for(int t = 0; t < 6; t++) {c.tidal[t] = (MyFloat) soa->tidal_tensorps[(long)pk * 6 + t];}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    c.mass_dm = (MyFloat) soa->mass_dm[pk];
    c.s_dm    = Vec3<MyFloat>{(MyFloat)soa->s_dm[pk][0], (MyFloat)soa->s_dm[pk][1], (MyFloat)soa->s_dm[pk][2]};
    c.vs_dm   = Vec3<MyFloat>{(MyFloat)soa->vs_dm[pk][0], (MyFloat)soa->vs_dm[pk][1], (MyFloat)soa->vs_dm[pk][2]};
#endif
    return c;
}

/* Recursive SoA-native replacement for force_treeupdate_pseudos.
 * Reads moments from soa->* for the 8 children (topnode tree children)
 * of `no_abs`, re-sums them, and writes the result back to soa->[no_k].
 * Also writes back to AoS Nodes[no_abs] / Extnodes[no_abs] for
 * any CPU code that may read them.
 *
 * Called only for INTERNAL_TOPLEVEL nodes (the caller checks the bit
 * before descending).  Topleaf topnodes are never passed here; their
 * moments are read directly by the caller and accumulated. */
static void topnode_resum_node_(int no_abs,
                                struct gpu_gravity_tree_soa_t *soa,
                                int MaxPart, int MaxNodes_)
{
    int no_k = no_abs - MaxPart;   /* 0-based SoA index of `no_abs` */

    /* --- accumulate from 8 topnode children ------------------------- */
    MyFloat mass = 0;
    Vec3<MyFloat> s = {}, vs = {};
    MyFloat hmax = 0, vmax = 0, divVmax = 0, maxsoft = 0;
    long count_particles = 0;
    int multiple_flag = 0;

#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    MyFloat gasmass = 0;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    double cr_injection = 0;
#endif
#ifdef RT_USE_GRAVTREE
    MyFloat stellar_lum[N_RT_FREQ_BINS] = {};
#ifdef CHIMES_STELLAR_FLUXES
    double chimes_G0[CHIMES_LOCAL_UV_NBINS] = {}, chimes_ion[CHIMES_LOCAL_UV_NBINS] = {};
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyFloat> rt_s = {}, rt_vs = {};
#endif
#ifdef SINK_PHOTONMOMENTUM
    MyFloat sink_lum = 0;
    Vec3<MyFloat> sink_lum_grad = {};
#endif
#ifdef SINK_CALC_DISTANCES
    MyFloat sink_mass = 0;
    Vec3<MyFloat> sink_pos = {};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    Vec3<MyFloat> sink_vel = {};
    int N_SINK = 0;
#ifdef SPECIAL_POINT_MOTION
    Vec3<MyFloat> sink_acc = {};
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    MyFloat max_fbvel = 0;
#endif
#endif
#endif
#ifdef DM_SCALARFIELD_SCREENING
    MyFloat mass_dm = 0;
    Vec3<MyFloat> s_dm = {}, vs_dm = {};
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    SymmetricTensor2<MyFloat> tidal_tensorps_prevstep = {};
#endif

    /* Walk 8 children via nextnode + sibling.  For INTERNAL_TOPLEVEL topnodes,
     * nextnode points to the first of 8 topnode children; sibling links them.
     * Mirrors the force_treeupdate_pseudos loop (forcetree.cc:1292-1360). */
    /* Point the shared moment kernel's ref at this node's local accumulators. The accumulate loop
     * below adds each (normalized) child's re-weighted contribution through moment_accum_add_child_
     * normalized; the normalize + store blocks that follow are untouched (finalize unifies in c2). */
    moment_node_ref<MyFloat> acc_ref = {};
    acc_ref.mass     = &mass;
    acc_ref.s        = &s;
    acc_ref.vs       = &vs;
    acc_ref.Npart    = &count_particles;
    acc_ref.hmax     = &hmax;
    acc_ref.vmax     = &vmax;
    acc_ref.divVmax  = &divVmax;
    acc_ref.maxsoft  = &maxsoft;
    acc_ref.bitflags = NULL;   /* multiple-particles bit handled at store, not accumulate */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    acc_ref.gasmass  = &gasmass;
#endif
#ifdef RT_USE_GRAVTREE
    acc_ref.stellar_lum = &stellar_lum[0];
#ifdef CHIMES_STELLAR_FLUXES
    acc_ref.chimes_G0  = &chimes_G0[0];
    acc_ref.chimes_ion = &chimes_ion[0];
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    acc_ref.rt_s  = &rt_s;
    acc_ref.rt_vs = &rt_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
    acc_ref.sink_lum      = &sink_lum;
    acc_ref.sink_lum_grad = &sink_lum_grad;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    acc_ref.cr_inject = &cr_injection;
#endif
#ifdef SINK_CALC_DISTANCES
    acc_ref.sink_mass = &sink_mass;
    acc_ref.sink_pos  = &sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
    acc_ref.N_SINK   = &N_SINK;
    acc_ref.sink_vel = &sink_vel;
#endif
#if defined(SPECIAL_POINT_MOTION)
    acc_ref.sink_acc = &sink_acc;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    acc_ref.max_fbvel = &max_fbvel;
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    acc_ref.tidal = &tidal_tensorps_prevstep.data[0];
#endif
#ifdef DM_SCALARFIELD_SCREENING
    acc_ref.mass_dm = &mass_dm;
    acc_ref.s_dm    = &s_dm;
    acc_ref.vs_dm   = &vs_dm;
#endif

    int p = soa->nextnode[no_k];   /* first child (absolute index) */
    for(int j = 0; j < 8; j++) {
        if(p < MaxPart || p >= MaxPart + MaxNodes_) {endrun(6767); return;}  /* soft bad-stop: skip OOB SoA read; partial resum drains at next poll */
        int pk = p - MaxPart;

        /* Recurse if this child is also an internal topnode. */
        if(soa->bitflags[pk] & (1u << BITFLAG_INTERNAL_TOPLEVEL)) {
            topnode_resum_node_(p, soa, MaxPart, MaxNodes_);
        }

        /* Accumulate this (now-finalized) child's re-weighted moments. */
        moment_node_accum<MyFloat> child = topnode_child_accum_(soa, pk);
        moment_accum_add_child_normalized<moment_plain_ops, MyFloat>(acc_ref, child);

        p = soa->sibling[pk];   /* advance to next sibling */
    }

    /* Normalize the mass-weighted sums in place (COM-style fields divide, tidal keeps its
     * reciprocal-multiply); the sink-distance COM accumulators (sink_pos / sink_vel / sink_acc) are
     * normalized in place here, so the stores below write them directly. acc_ref.bitflags is null;
     * the multiple-particles bit is set just below and applied to both the SoA and AoS bitflags. */
    moment_finalize<MyFloat>(acc_ref, Vec3<MyFloat>{(MyFloat)Nodes[no_abs].center[0],
                                                    (MyFloat)Nodes[no_abs].center[1],
                                                    (MyFloat)Nodes[no_abs].center[2]});

    if(count_particles > 1) {multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);}

    /* --- write SoA -------------------------------------------------- */
    soa->mass[no_k]    = (MyGravFloat) mass;
    soa->s[no_k]       = {(MyGravFloat)s[0], (MyGravFloat)s[1], (MyGravFloat)s[2]};
    soa->node_vs[no_k] = {(MyGravFloat)vs[0], (MyGravFloat)vs[1], (MyGravFloat)vs[2]};
    soa->N_part[no_k]  = count_particles;
    soa->hmax[no_k]    = (MyGravFloat) hmax;
    soa->vmax[no_k]    = (MyGravFloat) vmax;
    soa->divVmax[no_k] = (MyGravFloat) divVmax;
    soa->maxsoft[no_k] = (MyGravFloat) maxsoft;
    soa->bitflags[no_k] = (soa->bitflags[no_k] & ~(1u << BITFLAG_MULTIPLEPARTICLES))
                        | (unsigned int) multiple_flag;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    soa->gasmass[no_k] = (MyGravFloat) gasmass;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    soa->cr_injection[no_k] = (MyGravFloat) cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {
        soa->stellar_lum[(long)no_k * N_RT_FREQ_BINS + b] = (MyGravFloat) stellar_lum[b];
    }
#ifdef CHIMES_STELLAR_FLUXES
    for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
        soa->chimes_stellar_lum_G0 [(long)no_k * CHIMES_LOCAL_UV_NBINS + b] = chimes_G0[b];
        soa->chimes_stellar_lum_ion[(long)no_k * CHIMES_LOCAL_UV_NBINS + b] = chimes_ion[b];
    }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    soa->rt_source_lum_s[no_k]  = {(MyGravFloat)rt_s[0],
                                    (MyGravFloat)rt_s[1],
                                    (MyGravFloat)rt_s[2]};
    soa->rt_source_lum_vs[no_k] = {(MyGravFloat)rt_vs[0],
                                    (MyGravFloat)rt_vs[1],
                                    (MyGravFloat)rt_vs[2]};
#endif
#ifdef SINK_PHOTONMOMENTUM
    soa->sink_lum[no_k]      = (MyGravFloat) sink_lum;
    soa->sink_lum_grad[no_k] = {(MyGravFloat)sink_lum_grad[0],
                                  (MyGravFloat)sink_lum_grad[1],
                                  (MyGravFloat)sink_lum_grad[2]};
#endif
#ifdef SINK_CALC_DISTANCES
    soa->sink_mass[no_k] = (MyGravFloat) sink_mass;
    if(sink_mass > 0) {
        soa->sink_pos[no_k] = {(MyGravFloat)sink_pos[0], (MyGravFloat)sink_pos[1], (MyGravFloat)sink_pos[2]};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        soa->N_SINK[no_k] = N_SINK;
        soa->sink_vel[no_k] = {(MyGravFloat)sink_vel[0], (MyGravFloat)sink_vel[1], (MyGravFloat)sink_vel[2]};
#ifdef SPECIAL_POINT_MOTION
        soa->sink_acc[no_k] = {(MyGravFloat)sink_acc[0], (MyGravFloat)sink_acc[1], (MyGravFloat)sink_acc[2]};
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
        soa->MaxFeedbackVel[no_k] = (MyGravFloat) max_fbvel;
#endif
#endif
    }
#endif
#ifdef DM_SCALARFIELD_SCREENING
    soa->mass_dm[no_k] = (MyGravFloat) mass_dm;
    soa->s_dm[no_k]    = {(MyGravFloat)s_dm[0], (MyGravFloat)s_dm[1], (MyGravFloat)s_dm[2]};
    soa->vs_dm[no_k]   = {(MyGravFloat)vs_dm[0], (MyGravFloat)vs_dm[1], (MyGravFloat)vs_dm[2]};
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    for(int t = 0; t < 6; t++) {soa->tidal_tensorps[(long)no_k * 6 + t] = (MyGravFloat) tidal_tensorps_prevstep.data[t];}
#endif

    /* --- AoS writeback (mirrors gpu_moment_writeback_to_aos for topnodes) */
    Nodes[no_abs].u.d.mass     = mass;
    Nodes[no_abs].u.d.s        = s;
    Nodes[no_abs].N_part       = count_particles;
    Nodes[no_abs].maxsoft      = maxsoft;
    Nodes[no_abs].u.d.bitflags = (Nodes[no_abs].u.d.bitflags
                                   & ~(1u << BITFLAG_MULTIPLEPARTICLES))
                                | (unsigned int) multiple_flag;
    Extnodes[no_abs].vs        = vs;
    Extnodes[no_abs].hmax      = hmax;
    Extnodes[no_abs].vmax      = vmax;
    Extnodes[no_abs].divVmax   = divVmax;
    Extnodes[no_abs].Flag      = GlobFlag;
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
    Nodes[no_abs].gasmass      = gasmass;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    Nodes[no_abs].cr_injection = (MyFloat) cr_injection;
#endif
#ifdef RT_USE_GRAVTREE
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {
        Nodes[no_abs].stellar_lum[b] = stellar_lum[b];
    }
#ifdef CHIMES_STELLAR_FLUXES
    for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
        Nodes[no_abs].chimes_stellar_lum_G0[b]  = chimes_G0[b];
        Nodes[no_abs].chimes_stellar_lum_ion[b] = chimes_ion[b];
    }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Nodes[no_abs].rt_source_lum_s       = rt_s;
    Extnodes[no_abs].rt_source_lum_vs   = rt_vs;
#endif
#ifdef SINK_PHOTONMOMENTUM
    Nodes[no_abs].sink_lum      = sink_lum;
    Nodes[no_abs].sink_lum_grad = sink_lum_grad;
#endif
#ifdef SINK_CALC_DISTANCES
    Nodes[no_abs].sink_mass = sink_mass;
    if(sink_mass > 0) {
        Nodes[no_abs].sink_pos = sink_pos;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        Nodes[no_abs].N_SINK   = N_SINK;
        Nodes[no_abs].sink_vel = sink_vel;
#ifdef SPECIAL_POINT_MOTION
        Nodes[no_abs].sink_acc = sink_acc;
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
        Nodes[no_abs].MaxFeedbackVel = max_fbvel;
#endif
#endif
    }
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Nodes[no_abs].mass_dm     = mass_dm;
    Nodes[no_abs].s_dm        = s_dm;
    Extnodes[no_abs].vs_dm    = vs_dm;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    Nodes[no_abs].tidal_tensorps_prevstep = tidal_tensorps_prevstep;
#endif
}

extern "C" int gpu_topnode_moment_resum(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(NTopnodes <= 0) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->bitflags || !soa->nextnode || !soa->sibling) {
        printf("gpu_topnode_moment_resum: SoA not ready\n");
        return 1;
    }

    int MaxPart  = All.MaxPart;
    int MaxNodes_ = MaxNodes;

    /* Start the recursive resum from the tree root (absolute index MaxPart,
     * SoA index 0).  The root is always INTERNAL_TOPLEVEL when NTask > 1.
     * With NTask == 1 the root is the single topleaf — no resum needed. */
    int root_abs = All.MaxPart;
    int root_k   = 0;
    if(soa->bitflags[root_k] & (1u << BITFLAG_INTERNAL_TOPLEVEL)) {
        topnode_resum_node_(root_abs, soa, MaxPart, MaxNodes_);
    }

    return 0;
}

/* ===================================================================== */
/* Phase 9.2-pre — gpu_scatter_foreign_to_soa                            */
/*                                                                        */
/* Mirror foreign-node AoS Nodes[]/Extnodes[] fields into SoA for the     */
/* range [slot_base_abs, slot_base_abs + count).  Called from             */
/* let_unpack_and_install once per sender after the AoS install +         */
/* pointer-remap pass.  Walk reads ALL of these fields from SoA, so we    */
/* must cover every conditional payload that gpu_scatter_pseudo_to_soa    */
/* covers.  Topology fields (sibling/nextnode/father/bitflags) come       */
/* directly from the AoS (already remapped to absolute indices in         */
/* let_unpack_and_install Pass 1).                                        */
/* ===================================================================== */
extern "C" int gpu_scatter_foreign_to_soa(int slot_base_abs, int count)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(count <= 0) {return 0;}

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {printf("gpu_scatter_foreign_to_soa: SoA null\n"); return 1;}

    int MaxPart_ = All.MaxPart;

    for(int j = 0; j < count; j++) {
        int no = slot_base_abs + j;
        int k  = no - MaxPart_;     /* SoA index = (MaxNodes + slot_local) */

        /* Geometric */
        soa->center[k] = Nodes[no].center;
        soa->len[k]    = Nodes[no].len;

        /* Moment scalars */
        soa->mass[k]    = (MyGravFloat) Nodes[no].u.d.mass;
        soa->N_part[k]  = Nodes[no].N_part;
        soa->maxsoft[k] = (MyGravFloat) Nodes[no].maxsoft;
        soa->hmax[k]    = (MyGravFloat) Extnodes[no].hmax;
        soa->vmax[k]    = (MyGravFloat) Extnodes[no].vmax;
        soa->divVmax[k] = (MyGravFloat) Extnodes[no].divVmax;
        soa->bitflags[k] = Nodes[no].u.d.bitflags;

        /* Moment vectors */
        soa->s[k]        = {(MyGravFloat) Nodes[no].u.d.s[0],
                            (MyGravFloat) Nodes[no].u.d.s[1],
                            (MyGravFloat) Nodes[no].u.d.s[2]};
        soa->node_vs[k]  = {(MyGravFloat) Extnodes[no].vs[0],
                            (MyGravFloat) Extnodes[no].vs[1],
                            (MyGravFloat) Extnodes[no].vs[2]};

        /* Topology (already in absolute indices after Pass 1 remap) */
        soa->sibling[k]  = Nodes[no].u.d.sibling;
        soa->nextnode[k] = Nodes[no].u.d.nextnode;
        soa->father[k]   = Nodes[no].u.d.father;

#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        soa->gasmass[k]  = (MyGravFloat) Nodes[no].gasmass;
#endif
#ifdef RT_USE_GRAVTREE
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            soa->stellar_lum[(long)k * N_RT_FREQ_BINS + b] =
                (MyGravFloat) Nodes[no].stellar_lum[b];
        }
#ifdef CHIMES_STELLAR_FLUXES
        for(int b = 0; b < CHIMES_LOCAL_UV_NBINS; b++) {
            soa->chimes_stellar_lum_G0 [(long)k * CHIMES_LOCAL_UV_NBINS + b] =
                Nodes[no].chimes_stellar_lum_G0[b];
            soa->chimes_stellar_lum_ion[(long)k * CHIMES_LOCAL_UV_NBINS + b] =
                Nodes[no].chimes_stellar_lum_ion[b];
        }
#endif
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        soa->rt_source_lum_s[k]  = {(MyGravFloat) Nodes[no].rt_source_lum_s[0],
                                     (MyGravFloat) Nodes[no].rt_source_lum_s[1],
                                     (MyGravFloat) Nodes[no].rt_source_lum_s[2]};
        soa->rt_source_lum_vs[k] = {(MyGravFloat) Extnodes[no].rt_source_lum_vs[0],
                                     (MyGravFloat) Extnodes[no].rt_source_lum_vs[1],
                                     (MyGravFloat) Extnodes[no].rt_source_lum_vs[2]};
#endif
#ifdef SINK_PHOTONMOMENTUM
        soa->sink_lum[k]      = (MyGravFloat) Nodes[no].sink_lum;
        soa->sink_lum_grad[k] = {(MyGravFloat) Nodes[no].sink_lum_grad[0],
                                  (MyGravFloat) Nodes[no].sink_lum_grad[1],
                                  (MyGravFloat) Nodes[no].sink_lum_grad[2]};
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        soa->cr_injection[k] = (MyGravFloat) Nodes[no].cr_injection;
#endif
#ifdef SINK_CALC_DISTANCES
        soa->sink_mass[k] = (MyGravFloat) Nodes[no].sink_mass;
        soa->sink_pos[k]  = {(MyGravFloat) Nodes[no].sink_pos[0],
                              (MyGravFloat) Nodes[no].sink_pos[1],
                              (MyGravFloat) Nodes[no].sink_pos[2]};
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
        soa->N_SINK[k]    = Nodes[no].N_SINK;
        soa->sink_vel[k]  = {(MyGravFloat) Nodes[no].sink_vel[0],
                              (MyGravFloat) Nodes[no].sink_vel[1],
                              (MyGravFloat) Nodes[no].sink_vel[2]};
#ifdef SPECIAL_POINT_MOTION
        soa->sink_acc[k]  = {(MyGravFloat) Nodes[no].sink_acc[0],
                              (MyGravFloat) Nodes[no].sink_acc[1],
                              (MyGravFloat) Nodes[no].sink_acc[2]};
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
        soa->MaxFeedbackVel[k] = (MyGravFloat) Nodes[no].MaxFeedbackVel;
#endif
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
        for(int t = 0; t < 6; t++) {
            soa->tidal_tensorps[(long)k * 6 + t] = (MyGravFloat) Nodes[no].tidal_tensorps_prevstep.data[t];
        }
#endif
#ifdef DM_SCALARFIELD_SCREENING
        soa->mass_dm[k] = (MyGravFloat) Nodes[no].mass_dm;
        soa->s_dm[k]    = {(MyGravFloat) Nodes[no].s_dm[0],
                           (MyGravFloat) Nodes[no].s_dm[1],
                           (MyGravFloat) Nodes[no].s_dm[2]};
        soa->vs_dm[k]   = {(MyGravFloat) Extnodes[no].vs_dm[0],
                           (MyGravFloat) Extnodes[no].vs_dm[1],
                           (MyGravFloat) Extnodes[no].vs_dm[2]};
#endif
    }
    return 0;
}

extern "C" int gpu_set_soa_nextnode(int abs_idx, int new_nextnode)
{
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->nextnode) {return 1;}
    int k = abs_idx - All.MaxPart;
    if(k < 0) {return 1;}
    soa->nextnode[k] = new_nextnode;
    return 0;
}

