/* sink_feed_gpu.cc — GPU neighbor-loop port for sink_feed_evaluate (C3).
 *
 * Active Type-5 (sink) particles scatter accretion events and thermal feedback
 * into surrounding neighbor particles via a Kokkos parallel_for kernel.
 * This replaces the CPU tree-walk inside sink_feed_loop() when
 * GPU translation unit (Kokkos/nvcc_wrapper).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/ghost_symlist_lifecycle.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../mesh/ghost_writeback.h"

#if defined(SINK_PARTICLES)

#include "sink_feed_functions.h"

/* active-check: mirrors sink_isactive + extra guards from particle2in */
static int sink_feed_is_active(int i, struct particle_data *P_host)
{
    if(P_host[i].Type != 5) return 0;
    if(P_host[i].Mass <= 0) return 0;
    if(P_host[i].KernelRadius <= 0) return 0;
    /* TimeBin >= 0 check: sink_isactive requires P[i].TimeBin >= 0 */
    if(P_host[i].TimeBin < 0) return 0;
    return 1;
}

/* Fill SinkFeedLocalIn for source particle i from CPU global arrays */
static void sink_feed_local_fill(int i,
                                  struct particle_data *P_host,
                                  struct gas_cell_data *CellP_host,
                                  struct SinkFeedLocalIn *loc)
{
    (void)CellP_host;
    int j_tempinfo = P_host[i].IndexMapToTempStruc;

    loc->Pos          = P_host[i].Pos;
    loc->Vel          = P_host[i].Vel;
    loc->KernelRadius = P_host[i].KernelRadius;
    loc->Mass         = P_host[i].Mass;
    loc->Sink_Mass    = P_host[i].Sink_Mass;
    loc->ID           = P_host[i].ID;
    loc->Density      = P_host[i].DensityAroundParticle;
    loc->Mdot         = P_host[i].Sink_Mdot;
    loc->Dt           = (MyFloat)get_particle_feedback_timestep_in_physical(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    loc->Dt           = P_host[i].dt_since_last_gas_search;
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    loc->SinkRadius   = P_host[i].SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    loc->AGS_KernelRadius = (MyFloat)ForceSoftening_KernelRadius(i);
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    loc->Sink_Mass_Reservoir = P_host[i].Sink_Mass_Reservoir;
#endif
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM
    loc->Jgas_in_Kernel = P_host[i].Sink_Specific_AngMom;
#else
    loc->Jgas_in_Kernel = SinkTempInfo[j_tempinfo].Jgas_in_Kernel;
#endif
#endif
#ifdef SINK_GRAVCAPTURE_GAS
    loc->mass_to_swallow_edd = SinkTempInfo[j_tempinfo].mass_to_swallow_edd;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
    {
        double meddington  = sink_eddington_mdot((double)P_host[i].Sink_Mass);
        double medd_max    = All.SinkEddingtonFactor * meddington * (double)loc->Dt;
        double edd_factor  = (medd_max > 0) ? ((double)SinkTempInfo[j_tempinfo].mass_to_swallow_edd / medd_max) : 1e30;
        loc->edd_p         = (MyFloat)((edd_factor > 0) ? (1.0 / edd_factor) : 1.0);
    }
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    loc->Sink_AccretionDeficit = P_host[i].Sink_AccretionDeficit;
#endif
#ifdef SINK_THERMALFEEDBACK
    loc->thermal_energy = (MyFloat)(sink_lum_bol((double)P_host[i].Sink_Mdot,
                                                  (double)P_host[i].Sink_Mass, -1)
                                    * (double)loc->Dt);
#endif
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    loc->Sink_eligible_for_binary_merge_away = is_star_eligible_for_binary_merge_away(i);
#endif
}


/* ================================================================
   GPU sink-feed evaluator (mark-for-accretion pass)
   ----------------------------------------------------------------
   Kernel writes (host-visible, by index):
     i-side: out_arr[aa] (per-source SinkFeedOut: angle-weighted-kernel-sum,
                          potential-min position, etc.)
     j-side (neighbors, indexed by j = neighbors[nn]):
       P_gpu[j].SwallowID                     — Kokkos::atomic_exchange
       CellP_gpu[j].Injected_Sink_Energy      — Kokkos::atomic_add

   Sparse-scatter target: walk neighbors[] CSR; for each touched j scatter
   the two fields above. The full memcpy(P_host,P_gpu,...) +
   memcpy(CellP_host,CellP_gpu,...) at lines ~282-283 are oversized for
   what the kernel actually writes. Replace with sparse two-field scatter.
   ================================================================ */
void sink_feed_evaluate_gpu(struct particle_data *P_host,
                              struct gas_cell_data *CellP_host,
                              int num_total,
                              int *i_active_host, int num_active,
                              const double *src_radii_host)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(sinkfeed);

    /* Caller supplies LOCAL active sources (ActiveParticleList + sink_isactive).
       Multi-rank correctness: gizmo_density_prep_ghosts is an MPI collective so
       every rank must agree on whether to call it. MPI_Allreduce num_src first;
       if no rank has any source, every rank skips ghost_prep + the rest. */
    int num_src = num_active;
    int global_num_src = num_src;
    MPI_Allreduce(&num_src, &global_num_src, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(global_num_src == 0) { return; } /* nothing to do anywhere; ghost_writeback's writes would be no-ops */

    /* Prep ghosts */
    int imported_ghosts = gizmo_explicit_query_prep_ghosts_fresh(
        "sink_feed", NGB_SEARCH_SYMMETRIC, (unsigned int)SINK_NEIGHBOR_BITFLAG,
        i_active_host, num_src, src_radii_host, gizmo_ghost_safety_factor());

    /* Fill per-source input structs (1-element backstop when num_src==0) */
    std::vector<struct SinkFeedLocalIn> src_local(num_src > 0 ? num_src : 1);
    for(int a = 0; a < num_src; a++) {
        sink_feed_local_fill(i_active_host[a], P_host, CellP_host, &src_local[a]);
    }

    /* Copy P and CellP to SharedSpace */
    int num_all = ghost_get_num_local() + ghost_get_num_ghosts();
    if(num_all <= 0) num_all = num_total;

    /* Wrapper fast-path: no source sinks → no kernel writes.  Skip the 12.4M-
     * iter softening_kernel_radius_host[] CPU loop, the same-size sink_binary
     * loop, arena_acquire (1.4s slow path), big SharedSpace allocs, kernel,
     * scatter.  Preserve ghost_writeback collectives for multi-rank
     * consistency (each self-guards on NTask<=1 || num_ghosts<=0). */
    if(num_src == 0) {
        ghost_write_detector_begin("sink_feed");
        ghost_writeback_zero_sinkfeed();
        ghost_writeback_sinkfeed();
        ghost_write_detector_end();
        if(imported_ghosts) ghost_exchange_cleanup();
        return;
    }

    std::vector<double> softening_kernel_radius_host(num_all);
    for(int n = 0; n < num_all; n++) {
        softening_kernel_radius_host[n] = ForceSoftening_KernelRadius(n);
    }
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    std::vector<int> sink_binary_merge_eligible_host(num_all, 0);
    for(int n = 0; n < num_all; n++) {
        if(P_host[n].Mass > 0 && P_host[n].Type == 5) {
            sink_binary_merge_eligible_host[n] = is_star_eligible_for_binary_merge_away(n);
        }
    }
#endif

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_set_site("sink_feed_evaluate_gpu");
    gpu_particles_arena_acquire(num_all, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    double *d_softening_kernel_radius = (double *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_all * sizeof(double));
    memcpy(d_softening_kernel_radius, softening_kernel_radius_host.data(), num_all * sizeof(double));
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    int *d_sink_binary_merge_eligible = (int *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_all * sizeof(int));
    memcpy(d_sink_binary_merge_eligible, sink_binary_merge_eligible_host.data(), num_all * sizeof(int));
#else
    int *d_sink_binary_merge_eligible = NULL;
#endif

    /* Snapshot ghost fields before kernel */
    ghost_write_detector_begin("sink_feed");
    ghost_writeback_zero_sinkfeed();

    /* Copy per-source input to SharedSpace (1-element backstop when num_src==0) */
    int alloc_n = (num_src > 0) ? num_src : 1;
    struct SinkFeedLocalIn *d_local = (struct SinkFeedLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct SinkFeedLocalIn));
    if(num_src > 0) memcpy(d_local, src_local.data(), num_src * sizeof(struct SinkFeedLocalIn));

    /* Output: per-source accumulators */
    struct SinkFeedOut *d_out = (struct SinkFeedOut *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct SinkFeedOut));
    for(int a = 0; a < num_src; a++) {
        memset(&d_out[a], 0, sizeof(struct SinkFeedOut));
#ifdef SINK_REPOSITION_ON_POTMIN
        d_out[a].Sink_PotentialMinimumOfNeighbors = (double)SINK_MINPOTVALUE_INIT;
#endif
    }

    /* RNG step counter: use NumCurrentTiStep as counter base */
    uint64_t rng_step = (uint64_t)All.NumCurrentTiStep;

    /* Build neighbor list: Type-5 sources → SINK_NEIGHBOR_BITFLAG types.
     * Use the all-types step-persistent SIDX cache (sink_env1/feed/swk all
     * share; saves ~1.5s per step on every sink-active step). Sink kernels
     * use two-sided predicates, so this must be symmetric. */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_all,
                       i_active_host, num_src,
                       NGB_SEARCH_SYMMETRIC, SINK_NEIGHBOR_BITFLAG,
                       &gnl, gpu_step_sidx_alltypes_ptr(),
                       1.0, src_radii_host, NULL, "sink_feed");

    PRINT_STATUS("  GPU sink_feed: %d sources, j_bitmask=%d, %d pairs",
                 num_src, SINK_NEIGHBOR_BITFLAG, gnl.total_pairs);

    /* Launch kernel */
    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        struct SinkFeedLocalIn *local_arr = d_local;
        struct SinkFeedOut     *out_arr   = d_out;
        struct particle_data   *kp = P_gpu;
        struct gas_cell_data   *kc = CellP_gpu;
        double *softening_kernel_radius = d_softening_kernel_radius;
        int *sink_binary_merge_eligible = d_sink_binary_merge_eligible;

        Kokkos::parallel_for("sink_feed", num_src, KOKKOS_LAMBDA(int aa) {
            const struct SinkFeedLocalIn& loc = local_arr[aa];
            if(loc.KernelRadius <= 0 || loc.Mass <= 0) return;

            struct SinkFeedOut myout;
            memset(&myout, 0, sizeof(struct SinkFeedOut));
#ifdef SINK_REPOSITION_ON_POTMIN
            myout.Sink_PotentialMinimumOfNeighbors = (double)SINK_MINPOTVALUE_INIT;
#endif
            double mass_markedswallow = 0.;

            int start = offsets[aa], end = offsets[aa+1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0) continue;

                Vec3<double> dpos = kp[j].Pos - loc.Pos;
                Vec3<double> dvel = kp[j].Vel - loc.Vel;
                nearest_xyz(dpos, -1);
                NGB_SHEARBOX_BOUNDARY_VELCORR_(loc.Pos, kp[j].Pos, dvel, -1);
                double r2 = dpos.norm_sq();
                if(r2 <= 0) continue;

                sink_feed_pair_kernel(loc, j, kp, kc, r2, dpos, dvel,
                                      myout, mass_markedswallow, rng_step,
                                      softening_kernel_radius,
                                      sink_binary_merge_eligible);
            }

            /* Write i-side outputs — no race: each aa writes to its own slot */
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
            out_arr[aa].Sink_angle_weighted_kernel_sum +=
                myout.Sink_angle_weighted_kernel_sum;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
            if(myout.Sink_PotentialMinimumOfNeighbors <
               out_arr[aa].Sink_PotentialMinimumOfNeighbors) {
                out_arr[aa].Sink_PotentialMinimumOfNeighbors    =
                    myout.Sink_PotentialMinimumOfNeighbors;
                out_arr[aa].Sink_PotentialMinimumOfNeighborsPos =
                    myout.Sink_PotentialMinimumOfNeighborsPos;
            }
#endif
        });
    }
    Kokkos::fence();
    gizmo_gpu_check_last_error("sink_feed", num_src);

    /* Row 4b of arena-scope sweep: sparse scatter over CSR neighbors[] —
     * replaces former full memcpy(P_host, P_gpu, num_all*...) and
     * memcpy(CellP_host, CellP_gpu, num_all*...). Per the kernel-writes
     * audit (commit a06e30ca), sink_feed_pair_kernel writes ONLY:
     *   P_gpu[j].SwallowID                (Kokkos::atomic_exchange)
     *   CellP_gpu[j].Injected_Sink_Energy (Kokkos::atomic_add, SINK_THERMALFEEDBACK only)
     * Walk the CSR; idempotent assignment makes duplicates correct.
     *
     * gnl.neighbors lives in DEVICE_SPACE (CudaSpace) on GH200 — deep-copy
     * once to a host buffer, then iterate. */
    if(gnl.total_pairs > 0) {
        std::vector<int> gnl_neighbors_host(gnl.total_pairs);
        gpu_ngb_copy_neighbors_to_host(&gnl, gnl_neighbors_host.data());
        for(int idx = 0; idx < gnl.total_pairs; idx++) {
            int j = gnl_neighbors_host[idx];
            P_host[j].SwallowID = P_gpu[j].SwallowID;
#ifdef SINK_THERMALFEEDBACK
            CellP_host[j].Injected_Sink_Energy = CellP_gpu[j].Injected_Sink_Energy;
#endif
        }
    }

    /* Ghost writeback: propagate j-particle deltas to home ranks */
    ghost_writeback_sinkfeed();
    ghost_write_detector_end();

    /* Apply per-source outputs */
    std::vector<struct SinkFeedOut> h_out(num_src);
    memcpy(h_out.data(), d_out, num_src * sizeof(struct SinkFeedOut));

    for(int a = 0; a < num_src; a++) {
        int i = i_active_host[a];
        int j_tempinfo = P_host[i].IndexMapToTempStruc;
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
        SinkTempInfo[j_tempinfo].Sink_angle_weighted_kernel_sum +=
            h_out[a].Sink_angle_weighted_kernel_sum;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
        if(h_out[a].Sink_PotentialMinimumOfNeighbors < P_host[i].Sink_PotentialMinimumOfNeighbors) {
            P_host[i].Sink_PotentialMinimumOfNeighbors    = h_out[a].Sink_PotentialMinimumOfNeighbors;
            P_host[i].Sink_PotentialMinimumOfNeighborsPos = h_out[a].Sink_PotentialMinimumOfNeighborsPos;
        }
#endif
    }

    /* Preserve shared all-types SIDX for sink_swk reuse. */
    gpu_ngb_list_free(&gnl, gpu_step_sidx_alltypes_ptr());
    /* Full P+CellP scatter syncs arena→host; arena stays consistent.
     * ghost_writeback_sinkfeed() above already invalidates so the next
     * acquire re-syncs from any host changes it applied. */
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_softening_kernel_radius);
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_sink_binary_merge_eligible);
#endif

    if(imported_ghosts) { ghost_exchange_cleanup(); }
}

GPU_ALL_SYNC_FUNC(sinkfeed)

#else

void sink_feed_evaluate_gpu(struct particle_data *p,
                              struct gas_cell_data *cp,
                              int num_total,
                              int *i_active_host, int num_active,
                              const double *src_radii_host)
{
    (void)p; (void)cp; (void)num_total; (void)i_active_host; (void)num_active; (void)src_radii_host;
}
GPU_ALL_SYNC_FUNC_STUB(sinkfeed)

#endif /* SINK_PARTICLES */
