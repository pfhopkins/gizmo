/*! \file sink_feed.c
*  \brief This is where particles are marked for gas accretion.
*/
/* Stdlib + Kokkos MUST come before any project header (allvars.h pulls
 * macros.h which #defines `terminate(...)` and would mangle the C++
 * <exception> declarations Kokkos transitively pulls in). Same pattern
 * as sinks/sink_environment_gpu.cc and mesh/neighbor_loop_runner.cc. */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <Kokkos_Core.hpp>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"  /* ghost_get_num_local */
#include "../mesh/neighbor_loop_runner.h"
#include "sink_functions.h"
#include "sink_feed_loop.h"

/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/

#ifdef SINK_PARTICLES // top-level flag


#ifdef GIZMO_NLR_JSIDE_HASH_TEST
/* TRANSIENT j-side validation harness (per OPEN_3d_sinkfeed_design.md §E.2;
 * codex invariant 9). Compile-flag gated; remove or move engine-facing
 * before final commit. Dumps collision-resistant hashes of P[j].SwallowID
 * and CellP[j].Injected_Sink_Energy across all owner-local j's
 * (num_local — excludes imported ghosts). Used to compare runner Mode A
 * vs Mode B vs legacy on the same step. Safe at any path; fires once
 * per sink_feed_loop call. */
static inline uint64_t jside_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline uint64_t jside_rotl64(uint64_t x, int k) {
    return (x << (k & 63)) | (x >> ((64 - k) & 63));
}
static void jside_hash_dump(const char *path_label)
{
    /* Walk owner-local j's (skip ghosts). NumPart includes both; ghost
     * counters split it. Be conservative: use NumPart - num_ghosts. */
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = NumPart - num_ghosts;
    if(num_local < 0) num_local = NumPart;

    long sw_count_nonzero_loc = 0;
    unsigned long long sw_max_loc = 0;
    uint64_t sw_pair_hash_loc = 0;
#ifdef SINK_THERMALFEEDBACK
    long ie_count_nonzero_loc = 0;
    double ie_sum_loc = 0.0, ie_abs_sum_loc = 0.0;
#endif
    for(int j = 0; j < num_local; j++) {
        unsigned long long sw = (unsigned long long)P[j].SwallowID;
        if(sw != 0) {
            sw_count_nonzero_loc++;
            if(sw > sw_max_loc) sw_max_loc = sw;
        }
        uint64_t h_id = jside_splitmix64((uint64_t)P[j].ID);
        uint64_t h_sw = jside_splitmix64((uint64_t)sw);
        sw_pair_hash_loc ^= (h_id ^ jside_rotl64(h_sw, 1));
#ifdef SINK_THERMALFEEDBACK
        if(P[j].Type == 0) {
            double v = (double)CellP[j].Injected_Sink_Energy;
            if(v != 0.0) ie_count_nonzero_loc++;
            ie_sum_loc     += v;
            ie_abs_sum_loc += (v >= 0.0 ? v : -v);
        }
#endif
    }
    long sw_count_nonzero = 0;
    unsigned long long sw_max = 0;
    uint64_t sw_pair_hash = 0;
    MPI_Allreduce(&sw_count_nonzero_loc, &sw_count_nonzero, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sw_max_loc, &sw_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&sw_pair_hash_loc, &sw_pair_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
#ifdef SINK_THERMALFEEDBACK
    long ie_count_nonzero = 0;
    double ie_sum = 0.0, ie_abs_sum = 0.0;
    MPI_Allreduce(&ie_count_nonzero_loc, &ie_count_nonzero, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ie_sum_loc,           &ie_sum,           1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ie_abs_sum_loc,       &ie_abs_sum,       1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(ThisTask == 0) {
        std::printf("JSIDE_HASH sink_feed step=%lld path=%s "
                    "sw_count_nonzero=%ld sw_max=0x%llx sw_pair_hash=0x%016lx",
                    (long long)All.NumCurrentTiStep, path_label,
                    sw_count_nonzero, sw_max,
                    (unsigned long)sw_pair_hash);
#ifdef SINK_THERMALFEEDBACK
        std::printf(" ie_count_nonzero=%ld ie_sum=%a ie_abs_sum=%a",
                    ie_count_nonzero, ie_sum, ie_abs_sum);
#endif
        std::printf("\n");
        std::fflush(stdout);
    }
}
#endif /* GIZMO_NLR_JSIDE_HASH_TEST */


/* Host-side fill of SinkFeedLocalIn for source particle i. Mirrors the
 * legacy sink_feed_local_fill from sinks/sink_feed_gpu.cc:49–107
 * verbatim, with one cleanup: the SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
 * fill is gated out by the corresponding compile guard in
 * sinks/sink_feed_loop.h (eligibility array isn't yet wired through the
 * runner DeviceContext extension). */
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
    loc->thermal_energy = (MyFloat)(sink_lum_bol((double)P[i].Sink_Mdot,
                                                  (double)P[i].Sink_Mass, -1)
                                    * (double)loc->Dt);
#endif
}


/* Caller-side scatter manifest for sink_feed. Per-loop physics; the
 * runner doesn't see this. Operations match the pair_kernel per-field
 * write semantics (the oracle protects merge_accum specifically; this
 * scatter manifest must agree on op semantics field-for-field — drift is
 * silent because it lives downstream of the runner's oracle gate).
 *
 * Adding a new scattered field for this loop = ONE LINE under the
 * appropriate physics flag's #ifdef.
 *
 * Two scatter targets, deliberately not unified into one abstraction
 * (codex invariant 8: don't collapse merge_accum / scatter / ghost-
 * writeback into one): SinkTempInfo[j_tempinfo] for additive aggregates,
 * P[i] direct for the potential-min value+pos pair. */
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
     * paths skip the bundle. Phase 4.A.0 populate_device_context
     * stages host_locals into UVM ctx.per_active_local before the
     * device kernel; cleanup_device_context frees it on exit. */
    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    args.aux         = &aux;
    run_neighbor_loop<SinkFeedSpec>(args);

#ifdef GIZMO_NLR_JSIDE_HASH_TEST
    /* TRANSIENT — see jside_hash_dump comment above. Path label captured
     * via env var so a single binary can label each runner-driven path. */
    {
        const char *path_label = std::getenv("GIZMO_NLR_FORCE_MODE");
        if(!path_label || !path_label[0]) path_label = "default";
        jside_hash_dump(path_label);
    }
#endif

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
