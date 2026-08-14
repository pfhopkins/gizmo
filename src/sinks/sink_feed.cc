/*! \file sink_feed.c
*  \brief This is where particles are marked for gas accretion.
*/
/* Stdlib + Kokkos must precede any project header (allvars.h macros may
 * conflict with stdlib names). */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/neighbor_loop_runner.h"
#include "sink_functions.h"
#include "sink_feed_loop.h"

/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/

#ifdef SINK_PARTICLES // top-level flag


/* Host-side fill of SinkFeedLocalIn for source particle i. Mirrors the
 * legacy sink_feed_local_fill from sinks/sink_feed_gpu.cc:49-107
 * verbatim. SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES now wired: per-i
 * eligibility is filled here (per-j eligibility is filled in
 * sink_feed_loop.cc::populate_device_context). */
static void sink_feed_fill_local(int i, struct SinkFeedLocalIn *loc)
{
    int j_tempinfo = P[i].IndexMapToTempStruc;

    loc->Pos          = P[i].Pos;
    loc->Vel          = P[i].Vel;
    loc->KernelRadius = P[i].KernelRadius;
    loc->Mass         = P[i].Mass;
    loc->Sink_Mass    = P[i].Sink_Mass;
    loc->ID           = P[i].ID;
    loc->Density      = P[i].DensityAroundParticle;
    loc->Mdot         = P[i].Sink_Mdot;
    loc->Dt           = (MyFloat)get_particle_feedback_timestep_in_physical(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    loc->Dt           = P[i].dt_since_last_gas_search;
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    loc->SinkRadius   = P[i].SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    loc->AGS_KernelRadius = (MyFloat)ForceSoftening_KernelRadius(i);
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    loc->Sink_Mass_Reservoir = P[i].Sink_Mass_Reservoir;
#endif
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM
    loc->Jgas_in_Kernel = P[i].Sink_Specific_AngMom;
#else
    loc->Jgas_in_Kernel = SinkTempInfo[j_tempinfo].Jgas_in_Kernel;
#endif
#endif
#ifdef SINK_GRAVCAPTURE_GAS
    loc->mass_to_swallow_edd = SinkTempInfo[j_tempinfo].mass_to_swallow_edd;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
    {
        double meddington  = sink_eddington_mdot((double)P[i].Sink_Mass);
        double medd_max    = All.SinkEddingtonFactor * meddington * (double)loc->Dt;
        double edd_factor  = (medd_max > 0) ? ((double)SinkTempInfo[j_tempinfo].mass_to_swallow_edd / medd_max) : 1e30;
        loc->edd_p         = (MyFloat)((edd_factor > 0) ? (1.0 / edd_factor) : 1.0);
    }
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    loc->Sink_AccretionDeficit = P[i].Sink_AccretionDeficit;
#endif
#ifdef SINK_THERMALFEEDBACK
    /* The particle index is required, not optional: with a protostellar-evolution
     * model the luminosity is a cached per-star quantity read from the particle,
     * and the accretion rate and mass arguments are not used at all. */
    loc->thermal_energy = (MyFloat)(sink_lum_bol((double)P[i].Sink_Mdot,
                                                  (double)P[i].Sink_Mass, i)
                                    * (double)loc->Dt);
#endif
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    /* Per-i eligibility for the binary-merge-away criterion. Mirrors the
     * legacy sink_feed.cc:46 fill on the CPU path. Per-j eligibility is
     * filled by sink_feed_loop.cc::populate_device_context over the
     * full ctx.num_total range. */
    loc->Sink_eligible_for_binary_merge_away =
        is_star_eligible_for_binary_merge_away(i);
#endif
}


/* Caller-side scatter manifest for sink_feed. Per-loop physics; the
 * runner doesn't see this. Operations must match the pair_kernel's
 * per-field write semantics field-for-field — drift here is silent
 * since it lives downstream of the runner's correctness checks.
 *
 * Adding a new scattered field for this loop = ONE LINE under the
 * appropriate physics flag's #ifdef.
 *
 * Two scatter targets, deliberately not unified into one abstraction
 * (don't collapse merge_accum / scatter / ghost-writeback into one):
 * SinkTempInfo[j_tempinfo] for additive aggregates, P[i] direct for
 * the potential-min value+pos pair. */
static void sink_feed_scatter_to_temp_info(const int *active_list,
                                            int num_active,
                                            const struct SinkFeedOut *per_active_accum)
{
    for(int a = 0; a < num_active; a++) {
        int i = active_list[a];
        int t = P[i].IndexMapToTempStruc;

#define SCATTER_ADD(dst, src_field)        dst += per_active_accum[a].src_field;
#define SCATTER_MIN_PAIR(dst_v, dst_p, src_v, src_p)                            \
    do {                                                                        \
        if(per_active_accum[a].src_v < (dst_v)) {                               \
            (dst_v) = per_active_accum[a].src_v;                                \
            for(int k = 0; k < 3; k++) (dst_p)[k] = per_active_accum[a].src_p[k]; \
        }                                                                       \
    } while(0);

#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
        SCATTER_ADD(SinkTempInfo[t].Sink_angle_weighted_kernel_sum,
                    Sink_angle_weighted_kernel_sum)
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
        SCATTER_MIN_PAIR(P[i].Sink_PotentialMinimumOfNeighbors,
                         P[i].Sink_PotentialMinimumOfNeighborsPos,
                         Sink_PotentialMinimumOfNeighbors,
                         Sink_PotentialMinimumOfNeighborsPos)
#endif

#undef SCATTER_ADD
#undef SCATTER_MIN_PAIR
        (void)t; /* silence unused if all scatter ops are gated out */
    }
}


void sink_feed_loop(void)
{
    CPU_Step[CPU_SINK_FEEDSWK] += measure_time();

    /* ---------- engine: build active list ---------- */
    int *active_list = nullptr;
    int  num_active = 0, num_global_active = 0;
    if(!nlr_build_active_list(SinkFeedSpec::is_active,
                               &active_list, &num_active, &num_global_active,
                               "sinkfeed_active_list")) {
        CPU_Step[CPU_SINK_FEEDSWK] += measure_time();
        return;
    }

    /* ---------- physics: per-active accumulator + per-active host-fill input ---------- */
    int alloc_n = (num_active > 0) ? num_active : 1;
    struct SinkFeedOut *per_active_accum = (struct SinkFeedOut *)
        mymalloc("sinkfeed_per_active_accum", alloc_n * sizeof(struct SinkFeedOut));
    struct SinkFeedLocalIn *host_locals = (struct SinkFeedLocalIn *)
        mymalloc("sinkfeed_host_locals", alloc_n * sizeof(struct SinkFeedLocalIn));

    for(int a = 0; a < num_active; a++) {
        sink_feed_fill_local(active_list[a], &host_locals[a]);
    }

    SinkFeedSpec::Aux aux;
    aux.per_active_accum = per_active_accum;
    aux.host_locals      = host_locals;

    /* ---------- engine: hand off to runner ----------
     * Inside run_neighbor_loop: Mode A path runs detector_begin →
     * writeback_begin (snapshot ghost fields) → kernel → writeback_end
     * (snapshot-diff reverse-comm to home ranks) → detector_end. Mode B
     * paths skip the bundle. populate_device_context stages host_locals
     * into UVM ctx.per_active_local before the device kernel;
     * cleanup_device_context frees it on exit. */
    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    args.aux         = &aux;
    run_neighbor_loop<SinkFeedSpec>(args);

    /* ---------- physics: caller scatter into SinkTempInfo + P[i] ----------
     * Order-preserved per legacy sink_feed_gpu.cc:320 (ghost_writeback)
     * → :327-340 (apply per-source outputs). The runner's bundle end
     * (Mode A) fires inside run_neighbor_loop above, so by the time
     * we reach this scatter the j-side reverse-comm has already landed.
     */
    myfree(host_locals);
    sink_feed_scatter_to_temp_info(active_list, num_active, per_active_accum);

    /* ---------- engine: free + return ---------- */
    myfree(per_active_accum);
    nlr_free_active_list(active_list);

    CPU_Step[CPU_SINK_FEEDSWK] += measure_time();
}


#endif // top-level flag
