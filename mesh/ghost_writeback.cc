/* ghost_writeback.cc — reverse communication of ghost particle modifications.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "ghost_writeback.h"
#include "../system/gpu_particles_arena.h"
#include "../system/mpi_alltoallv_typed.h"
#ifdef GALSF_FB_MECHANICAL
#include "../galaxy_sf/mechanical_fb_types.h"  /* for struct MechFBGasDelta */
#endif


/* ============================================================================
 * Generic ghost-writeback scaffold (Pass B.iv + B.iv.1 bundle-level exchange).
 *
 * begin_bundle: per-callback snapshot when num_ghosts>0; exactly-once
 *               arena_invalidate when num_ghosts>0. Strict no-op when
 *               n_callbacks==0 (no snapshot, no invalidate, no MPI, no
 *               cleanup).
 *
 * end_bundle:   exactly-once register_writeback (always, when bundle
 *               non-empty); BUNDLE-LEVEL single-pass count + pack +
 *               ONE Alltoall + ONE Alltoallv + apply (NTask>1 only);
 *               per-callback cleanup; exactly-once arena_invalidate
 *               (NTask>1 only). Strict no-op when n_callbacks==0.
 *
 * MPI-call invariant: total MPI calls per non-trivial end_bundle is O(1)
 * in the number of callbacks, NOT O(n_callbacks). All callbacks in a bundle
 * share one Alltoall (count exchange) and one gizmo_mpi_alltoallv_typed
 * (data exchange); the receiver demultiplexes by canonical callback order.
 * This makes it safe to put dozens-to-hundreds of writeback fields in one
 * Spec's manifest (mechFB, chemistry, etc.) without ballooning MPI cost.
 *
 * Per-rank send-buffer layout (rank-major, callback-minor):
 *
 *     [rank 0 segment][rank 1 segment]...[rank NTask-1 segment]
 *
 *     each rank-t segment = concatenation in canonical bundle order:
 *       [ cb0 records to t | cb1 records to t | ... | cb(N-1) records to t ]
 *
 * Both sender and receiver derive offsets from the per-callback per-rank
 * count matrix (exchanged by the count Alltoall). Canonical bundle order
 * is identical across ranks because every rank assembles the same
 * manifest; the per-process bundle pointer addresses differ but the
 * ordered list of callbacks they describe is the same.
 *
 * Mode B paths (local + remote): bundle hooks are NOT called by the runner
 * — j-side writes happen on the rank that owns j (Mode B local: same rank
 * as walker; Mode B remote: peer rank running the walker on its own pool),
 * so no reverse-comm is needed. The runner's nlr_path_uses_imported_ghosts
 * predicate gates the begin_bundle/end_bundle calls to Mode A only.
 *
 * Legacy invalidation timing preserved exactly:
 *   begin: arena_invalidate only when num_ghosts > 0
 *   end:   arena_invalidate only when NTask > 1
 *   end:   register_writeback always (when bundle non-empty)
 * ========================================================================== */

void ghost_writeback_begin_bundle(const struct ghost_writeback_bundle *bundle)
{
    if (!bundle || bundle->n_callbacks == 0) return;  /* strict no-op */
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if (num_ghosts <= 0) return;                       /* legacy gate */
    for (int c = 0; c < bundle->n_callbacks; c++) {
        const ghost_writeback_callback *cb = bundle->callbacks[c];
        cb->snapshot(cb->ctx, num_ghosts, num_local);
    }
    gpu_particles_arena_invalidate();
}

void ghost_writeback_end_bundle(const struct ghost_writeback_bundle *bundle)
{
    if (!bundle || bundle->n_callbacks == 0) return;  /* strict no-op */

    /* Always register the writeback when the bundle is non-empty, even on
     * NTask<=1 short-circuit. Matches legacy ghost_writeback_swallowtime's
     * unconditional ghost_write_detector_register_writeback() call. */
    ghost_write_detector_register_writeback();

    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    const int N = bundle->n_callbacks;

    if (NTask <= 1) {
        /* No reverse-comm needed; still let each callback free its snap. */
        for (int c = 0; c < N; c++) {
            const ghost_writeback_callback *cb = bundle->callbacks[c];
            cb->cleanup(cb->ctx);
        }
        return;  /* NO arena_invalidate on NTask<=1 (matches legacy). */
    }

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();
    (void)home_index;  /* used by callbacks; some need it indirectly */

    /* ---- Per-callback per-rank count matrix.
     * send_count[c * NTask + t] = number of cb-c records to send to rank t. */
    int *send_count = (int *) calloc((size_t)N * NTask, sizeof(int));

    /* ---- SINGLE COUNT PASS over ghosts (not per-callback). For each ghost
     * and each callback, ask whether it contributes; accumulate into the
     * count matrix. Hot path: N predicate calls per ghost. */
    for (int g = 0; g < num_ghosts; g++) {
        int t = home_rank[g];
        for (int c = 0; c < N; c++) {
            const ghost_writeback_callback *cb = bundle->callbacks[c];
            if (cb->delta_for_ghost(cb->ctx, g, num_local)) {
                send_count[c * NTask + t]++;
            }
        }
    }

    /* Local record count = total records this rank will send. Captured here
     * (before send_count is freed below) for the optional rank-0 traffic
     * print after the apply pass. Global SUM == global received (matter
     * conservation in the alltoallv), so one MPI_Reduce gives both numbers. */
    long long local_records_sent = 0;
    for (int c = 0; c < N; c++) {
        for (int t = 0; t < NTask; t++) {
            local_records_sent += send_count[c * NTask + t];
        }
    }

    /* ---- Per-rank send byte counts and displacements. */
    int *send_bytes     = (int *) calloc(NTask, sizeof(int));
    for (int c = 0; c < N; c++) {
        const size_t ds = bundle->callbacks[c]->delta_size;
        for (int t = 0; t < NTask; t++) {
            send_bytes[t] += send_count[c * NTask + t] * (int)ds;
        }
    }
    int *send_byte_disp = (int *) malloc(NTask * sizeof(int));
    send_byte_disp[0] = 0;
    for (int t = 1; t < NTask; t++) send_byte_disp[t] = send_byte_disp[t-1] + send_bytes[t-1];
    int total_send_bytes = send_byte_disp[NTask-1] + send_bytes[NTask-1];

    /* ---- Per-rank per-callback OFFSET within the rank's segment.
     * cb_off[c * NTask + t] = start of cb-c records inside rank-t's segment
     * (relative to send_byte_disp[t]). */
    int *cb_off = (int *) calloc((size_t)N * NTask, sizeof(int));
    for (int t = 0; t < NTask; t++) {
        int running = 0;
        for (int c = 0; c < N; c++) {
            cb_off[c * NTask + t] = running;
            running += send_count[c * NTask + t] * (int)bundle->callbacks[c]->delta_size;
        }
    }

    /* ---- Single send buffer, packed in canonical (rank, callback) order. */
    char *send_buf = (char *) malloc((size_t)(total_send_bytes > 0 ? total_send_bytes : 1));

    /* ---- SINGLE PACK PASS over ghosts. Per-(c, t) progress counters track
     * how many cb-c records have already been packed for rank t inside its
     * segment. */
    int *pack_progress = (int *) calloc((size_t)N * NTask, sizeof(int));
    for (int g = 0; g < num_ghosts; g++) {
        int t = home_rank[g];
        for (int c = 0; c < N; c++) {
            const ghost_writeback_callback *cb = bundle->callbacks[c];
            if (cb->delta_for_ghost(cb->ctx, g, num_local)) {
                const size_t ds = cb->delta_size;
                int p = pack_progress[c * NTask + t]++;
                size_t offset = (size_t)send_byte_disp[t]
                              + (size_t)cb_off[c * NTask + t]
                              + (size_t)p * ds;
                cb->pack_delta(cb->ctx, g, num_local, send_buf + offset);
            }
        }
    }
    free(pack_progress);
    free(cb_off);

    /* ---- ONE MPI_Alltoall on the count matrix. Each rank sends N integers
     * (one per callback) to every other rank, packaged as "what I'm going
     * to send to you, broken down by callback." Receiver gets the per-cb
     * counts of records arriving FROM each rank. */
    int *alltoall_send = (int *) malloc((size_t)NTask * N * sizeof(int));
    int *alltoall_recv = (int *) malloc((size_t)NTask * N * sizeof(int));
    for (int t = 0; t < NTask; t++) {
        for (int c = 0; c < N; c++) {
            alltoall_send[t * N + c] = send_count[c * NTask + t];
        }
    }
    MPI_Alltoall(alltoall_send, N, MPI_INT, alltoall_recv, N, MPI_INT, MPI_COMM_WORLD);
    /* Untranspose to recv_count[c * NTask + t] = # cb-c records arriving from rank t. */
    int *recv_count = (int *) calloc((size_t)N * NTask, sizeof(int));
    for (int t = 0; t < NTask; t++) {
        for (int c = 0; c < N; c++) {
            recv_count[c * NTask + t] = alltoall_recv[t * N + c];
        }
    }
    free(alltoall_send); free(alltoall_recv);

    /* ---- Per-rank recv byte counts and displacements. */
    int *recv_bytes     = (int *) calloc(NTask, sizeof(int));
    for (int c = 0; c < N; c++) {
        const size_t ds = bundle->callbacks[c]->delta_size;
        for (int t = 0; t < NTask; t++) {
            recv_bytes[t] += recv_count[c * NTask + t] * (int)ds;
        }
    }
    int *recv_byte_disp = (int *) malloc(NTask * sizeof(int));
    recv_byte_disp[0] = 0;
    for (int t = 1; t < NTask; t++) recv_byte_disp[t] = recv_byte_disp[t-1] + recv_bytes[t-1];
    int total_recv_bytes = recv_byte_disp[NTask-1] + recv_bytes[NTask-1];

    char *recv_buf = (char *) malloc((size_t)(total_recv_bytes > 0 ? total_recv_bytes : 1));

    /* ---- ONE gizmo_mpi_alltoallv_typed on the byte stream (size = 1). */
    gizmo_mpi_alltoallv_typed(send_buf, send_bytes, send_byte_disp,
                              recv_buf, recv_bytes, recv_byte_disp,
                              1, MPI_COMM_WORLD);
    free(send_buf); free(send_bytes); free(send_byte_disp); free(send_count);

    /* ---- SINGLE APPLY PASS over received bytes. For each rank-t segment
     * (in canonical rank order), demultiplex by callback in canonical
     * bundle order using the per-cb-per-rank recv_count. */
    for (int t = 0; t < NTask; t++) {
        size_t cb_base = (size_t)recv_byte_disp[t];
        for (int c = 0; c < N; c++) {
            const ghost_writeback_callback *cb = bundle->callbacks[c];
            const size_t ds = cb->delta_size;
            int n_records = recv_count[c * NTask + t];
            for (int r = 0; r < n_records; r++) {
                cb->apply_delta(cb->ctx, recv_buf + cb_base + (size_t)r * ds);
            }
            cb_base += (size_t)n_records * ds;
        }
    }

    free(recv_buf); free(recv_bytes); free(recv_byte_disp); free(recv_count);

    /* Per-callback cleanup, then exactly-once arena_invalidate. */
    for (int c = 0; c < N; c++) {
        const ghost_writeback_callback *cb = bundle->callbacks[c];
        cb->cleanup(cb->ctx);
    }
    gpu_particles_arena_invalidate();

    /* Optional rank-0 traffic print. Backwards-compatible: bundles without a
     * loop_name (e.g. manually-constructed swallowtime_singleton_bundle)
     * stay silent. Uses MPI-summed global record count so rank 0 reports
     * total traffic, not just its own share — avoids false-fail on multi-
     * rank MA-N validation when rank 0 is purely a receiver. */
    if (bundle->loop_name != nullptr) {
        long long global_records = 0;
        MPI_Reduce(&local_records_sent, &global_records, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        if (ThisTask == 0 && global_records > 0) {
            printf("  Ghost writeback (%s): %lld records exchanged (global, sent==received)\n",
                   bundle->loop_name, global_records);
            fflush(stdout);
        }
    }
}


/* Compact delta struct for hydro j-writes.
 * Contains only the fields that hydro_force writes to j-particles. */
struct ghost_delta_hydro_t {
    int home_index;         /* P[]/CellP[] index on the home rank */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble dMass;         /* additive: CellP[j].dMass */
#endif
    short int wakeup;       /* max: P[j].wakeup */
};


void ghost_writeback_zero_hydro(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    for(int g = 0; g < num_ghosts; g++)
    {
        int j = num_local + g;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP[j].dMass = 0;
#endif
        P[j].wakeup = 0;
    }
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}


void ghost_writeback_hydro(void)
{
    ghost_write_detector_register_writeback();
    if(NTask <= 1) return; /* single rank: no ghosts, no communication needed */

    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();
    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();
    int *wb_recv_count = ghost_get_wb_recv_count();
    int *wb_recv_disp  = ghost_get_wb_recv_disp();
    int *wb_send_count = ghost_get_wb_send_count();
    int *wb_send_disp  = ghost_get_wb_send_disp();

    /* All ranks must participate in MPI collectives below, even with 0 ghosts.
       If provenance map is unavailable (ghost_exchange was skipped), all ranks
       must still agree — use zero-length communication. */

    /* Count how many deltas to send to each rank (only ghosts with modifications) */
    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++)
    {
        int j = num_local + g;
        int modified = 0;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        if(CellP[j].dMass != 0) modified = 1;
#endif
        if(P[j].wakeup != 0) modified = 1;
        if(modified) delta_send_count[home_rank[g]]++;
    }

    /* Build send displacements */
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1];
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    /* Pack deltas */
    struct ghost_delta_hydro_t *send_buf = (struct ghost_delta_hydro_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_hydro_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    for(int g = 0; g < num_ghosts; g++)
    {
        int j = num_local + g;
        int modified = 0;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        if(CellP[j].dMass != 0) modified = 1;
#endif
        if(P[j].wakeup != 0) modified = 1;
        if(!modified) continue;

        int task = home_rank[g];
        int off = pack_offset[task]++;
        send_buf[off].home_index = home_index[g];
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        send_buf[off].dMass = CellP[j].dMass;
#endif
        send_buf[off].wakeup = P[j].wakeup;
    }
    free(pack_offset);

    /* Exchange counts: each rank tells every other how many deltas it's sending */
    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);

    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1];
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    /* Exchange deltas via MPI_Alltoallv */
    struct ghost_delta_hydro_t *recv_buf = (struct ghost_delta_hydro_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_hydro_t));

    gizmo_mpi_alltoallv_typed(send_buf, delta_send_count, delta_send_disp,
                              recv_buf, delta_recv_count, delta_recv_disp,
                              sizeof(struct ghost_delta_hydro_t), MPI_COMM_WORLD);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    /* Apply received deltas to home particles */
    int wakeups_applied = 0;
    for(int d = 0; d < total_recv; d++)
    {
        int idx = recv_buf[d].home_index;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP[idx].dMass += recv_buf[d].dMass;
#endif
        if(recv_buf[d].wakeup > P[idx].wakeup) {
            P[idx].wakeup = recv_buf[d].wakeup;
            wakeups_applied++;
        }
    }
    if(wakeups_applied > 0) NeedToWakeupParticles_local = 1;

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (hydro): sent %d deltas, received %d deltas, %d wakeups applied\n",
               total_send, total_recv, wakeups_applied);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    gpu_particles_arena_invalidate(); /* host CellP/P deltas applied; arena stale */
}


/* --- AGSForce variant --------------------------------------------------- */

/* Reverse-communicates the j-side AGSForce delta set:
 *   - Vel delta (ADD)    [always present — AGS core carries Vel field]
 *   - dp delta  (ADD)    [populated under DM_SIDM]
 *   - NInteractions  (ADD)  [DM_SIDM only]
 *   - wakeup          (MAX, hydro-convention TimeBin+1) [CBE + SIDM wakeup condition]
 *
 * Zero version: snapshots ghost Vel/dp/NInteractions so post-kernel values
 * ARE the per-timestep deltas. Wakeup is zeroed outright. */

struct ghost_delta_agsforce_t {
    int home_index;
    double dVel[3];
    double ddp[3];
#if defined(DM_SIDM)
    long unsigned int dNInteractions;
#endif
    short int wakeup;
};

static Vec3<MyDouble> *agsforce_ghost_Vel0    = NULL;
static Vec3<MyDouble> *agsforce_ghost_dp0     = NULL;
#if defined(DM_SIDM)
static long unsigned int *agsforce_ghost_NInt0 = NULL;
#endif


void ghost_writeback_zero_agsforce(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    agsforce_ghost_Vel0 = (Vec3<MyDouble> *) malloc(num_ghosts * sizeof(Vec3<MyDouble>));
    agsforce_ghost_dp0  = (Vec3<MyDouble> *) malloc(num_ghosts * sizeof(Vec3<MyDouble>));
#if defined(DM_SIDM)
    agsforce_ghost_NInt0 = (long unsigned int *) malloc(num_ghosts * sizeof(long unsigned int));
#endif
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        agsforce_ghost_Vel0[g] = P[j].Vel;
        agsforce_ghost_dp0[g]  = P[j].dp;
#if defined(DM_SIDM)
        agsforce_ghost_NInt0[g] = P[j].NInteractions;
#endif
        P[j].wakeup = 0;
    }
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}


void ghost_writeback_agsforce(void)
{
    ghost_write_detector_register_writeback();
    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();

    /* MULTI-RANK COLLECTIVE CORRECTNESS: only NTask<=1 short-circuits the MPI
     * calls below. num_ghosts==0 OR snapshot-arrays==NULL: participate in the
     * collective with empty buffers; skip the per-ghost diff/pack loops. With
     * narrow-supply migrated callers, ghost imports are asymmetric (one rank
     * has them, another doesn't), so the local count cannot gate a collective. */
    if(NTask <= 1) {
        if(agsforce_ghost_Vel0) { free(agsforce_ghost_Vel0); agsforce_ghost_Vel0 = NULL; }
        if(agsforce_ghost_dp0)  { free(agsforce_ghost_dp0);  agsforce_ghost_dp0 = NULL; }
#if defined(DM_SIDM)
        if(agsforce_ghost_NInt0) { free(agsforce_ghost_NInt0); agsforce_ghost_NInt0 = NULL; }
#endif
        return;
    }
    int have_snap = (agsforce_ghost_Vel0 != NULL && agsforce_ghost_dp0 != NULL && num_ghosts > 0);

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    if(have_snap) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
            int modified = 0;
            if(P[j].Vel[0] != agsforce_ghost_Vel0[g][0] ||
               P[j].Vel[1] != agsforce_ghost_Vel0[g][1] ||
               P[j].Vel[2] != agsforce_ghost_Vel0[g][2]) modified = 1;
            if(P[j].dp[0] != agsforce_ghost_dp0[g][0] ||
               P[j].dp[1] != agsforce_ghost_dp0[g][1] ||
               P[j].dp[2] != agsforce_ghost_dp0[g][2]) modified = 1;
#if defined(DM_SIDM)
            if(P[j].NInteractions != agsforce_ghost_NInt0[g]) modified = 1;
#endif
            if(P[j].wakeup != 0) modified = 1;
            if(modified) delta_send_count[home_rank[g]]++;
        }
    }
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_agsforce_t *send_buf = (struct ghost_delta_agsforce_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_agsforce_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    if(have_snap) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
            int modified = 0;
            double dVx = P[j].Vel[0] - agsforce_ghost_Vel0[g][0];
            double dVy = P[j].Vel[1] - agsforce_ghost_Vel0[g][1];
            double dVz = P[j].Vel[2] - agsforce_ghost_Vel0[g][2];
            double dpx = P[j].dp[0]  - agsforce_ghost_dp0[g][0];
            double dpy = P[j].dp[1]  - agsforce_ghost_dp0[g][1];
            double dpz = P[j].dp[2]  - agsforce_ghost_dp0[g][2];
            if(dVx != 0 || dVy != 0 || dVz != 0) modified = 1;
            if(dpx != 0 || dpy != 0 || dpz != 0) modified = 1;
#if defined(DM_SIDM)
            long unsigned int dNI = P[j].NInteractions - agsforce_ghost_NInt0[g];
            if(dNI != 0) modified = 1;
#endif
            if(P[j].wakeup != 0) modified = 1;
            if(!modified) continue;

            int task = home_rank[g];
            int off = pack_offset[task]++;
            send_buf[off].home_index = home_index[g];
            send_buf[off].dVel[0] = dVx; send_buf[off].dVel[1] = dVy; send_buf[off].dVel[2] = dVz;
            send_buf[off].ddp[0]  = dpx; send_buf[off].ddp[1]  = dpy; send_buf[off].ddp[2]  = dpz;
#if defined(DM_SIDM)
            send_buf[off].dNInteractions = dNI;
#endif
            send_buf[off].wakeup = P[j].wakeup;
        }
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_agsforce_t *recv_buf = (struct ghost_delta_agsforce_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_agsforce_t));

    gizmo_mpi_alltoallv_typed(send_buf, delta_send_count, delta_send_disp,
                              recv_buf, delta_recv_count, delta_recv_disp,
                              sizeof(struct ghost_delta_agsforce_t), MPI_COMM_WORLD);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    int wakeups_applied = 0;
    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        P[idx].Vel[0] += recv_buf[d].dVel[0];
        P[idx].Vel[1] += recv_buf[d].dVel[1];
        P[idx].Vel[2] += recv_buf[d].dVel[2];
        P[idx].dp[0]  += recv_buf[d].ddp[0];
        P[idx].dp[1]  += recv_buf[d].ddp[1];
        P[idx].dp[2]  += recv_buf[d].ddp[2];
#if defined(DM_SIDM)
        P[idx].NInteractions += recv_buf[d].dNInteractions;
#endif
        if(recv_buf[d].wakeup != 0 && recv_buf[d].wakeup > P[idx].wakeup) {
            P[idx].wakeup = recv_buf[d].wakeup;
            wakeups_applied++;
        }
    }
    if(wakeups_applied > 0) { NeedToWakeupParticles_local = 1; }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (agsforce): sent %d deltas, received %d deltas, %d wakeups applied\n",
               total_send, total_recv, wakeups_applied);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);

    if(agsforce_ghost_Vel0) { free(agsforce_ghost_Vel0); agsforce_ghost_Vel0 = NULL; }
    if(agsforce_ghost_dp0)  { free(agsforce_ghost_dp0);  agsforce_ghost_dp0 = NULL; }
#if defined(DM_SIDM)
    if(agsforce_ghost_NInt0) { free(agsforce_ghost_NInt0); agsforce_ghost_NInt0 = NULL; }
#endif
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}


/* --- SwallowTime variant ------------------------------------------------- */
/* SwallowTime exists only when SINGLE_STAR_SINK_DYNAMICS is enabled.
 *
 * Pass B.iv: the legacy hand-written zero_swallowtime / swallowtime bodies
 * are replaced by thin singleton-bundle wrappers around the generic
 * ghost-writeback scaffold (begin_bundle / end_bundle). The op
 * (PARTICLE_MIN on particle_data::SwallowTime) and its callback set are
 * generated by the gw_detail::ParticleMinOp template in
 * mesh/ghost_writeback_ops.h.
 *
 * Behavior is byte-equivalent to the prior implementation for the
 * single-callback case (one Alltoallv, same delta layout, same gating
 * on num_ghosts and NTask, same exactly-once detector-register +
 * arena_invalidate).
 *
 * These public names are retained as compatibility wrappers for any
 * non-Spec caller. The SinkEnv1Spec ghost-writeback hooks DO NOT call
 * them — they call ghost_writeback_begin_bundle / _end_bundle directly
 * with the manifest-generated bundle. New physics flags should add a
 * GHOST_WRITEBACK_*(field) line to the appropriate Spec's bundle, NOT
 * a new public function alongside this one. */
#ifdef SINGLE_STAR_SINK_DYNAMICS

#include "ghost_writeback_ops.h"  /* gw_detail::ParticleMinOp template */

namespace {
static const ghost_writeback_callback *const swallowtime_singleton_cbs[] = {
    & gw_detail::ParticleMinOp<MyFloat, &particle_data::SwallowTime>::callback
};
static const ghost_writeback_bundle swallowtime_singleton_bundle = {
    swallowtime_singleton_cbs, 1, nullptr /* loop_name: silent print */
};
} /* anonymous namespace */

void ghost_writeback_zero_swallowtime(void) {
    ghost_writeback_begin_bundle(&swallowtime_singleton_bundle);
}

void ghost_writeback_swallowtime(void) {
    ghost_writeback_end_bundle(&swallowtime_singleton_bundle);
}

#endif /* SINGLE_STAR_SINK_DYNAMICS */


/* ThermalFB snapshot-based ghost_writeback_thermalfb / ghost_writeback_zero_thermalfb
 * retired in 3e.2 cleanup; replaced by generic bundle in thermal_fb_loop.cc. */




/* --- MechFB variant ------------------------------------------------------ */
/* Reverse communication for mechanical_fb GPU kernel per-gas delta accumulator.
 * MechFBGasDelta buffer is fresh-zeroed per top-level invocation, so the full
 * ghost entry IS the delta — no pre-kernel snapshot needed.
 *
 * Input: ghost_full_buf is the full device-side buffer (shared memory) of size
 *        num_local + num_ghost; entries [num_local..num_local+num_ghost) hold
 *        ghost-side deltas that need to ship home. home_buf is the caller's
 *        N_gas-sized home-rank buffer (already contains home-cell deltas from
 *        the kernel). We pack non-trivial ghost entries, Alltoallv them to
 *        home ranks, and accumulate into home_buf[home_index]. */

#ifdef GALSF_FB_MECHANICAL

struct ghost_delta_mechfb_t {
    int home_index;
    int N_injected;
    double m_injected, p_injected[3], KE_injected, TE_injected;
    double Z_injected[NUM_METAL_SPECIES];
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    double Mass_Where_Dust_Shocked;
#endif
#if defined(COSMIC_RAY_FLUID)
    double CR_energy_injected[N_CR_PARTICLE_BINS];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    double CR_number_injected[N_CR_PARTICLE_BINS];
#endif
    double CR_dir_weighted[3];
#endif
};

void ghost_writeback_mechfb(struct MechFBGasDelta *ghost_full_buf,
                             struct MechFBGasDelta *home_buf,
                             int n_gas)
{
    ghost_write_detector_register_writeback();
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    /* MULTI-RANK CORRECTNESS: do NOT early-return on num_ghosts<=0 alone.
     * The MPI_Alltoall + Alltoallv below are collective; every rank that
     * passes the caller's global-active-zero gate (e.g. mech_fb_gpu.cc:152)
     * must enter them in lock-step, even when its local ghost set is
     * empty. With a narrow supply mask (e.g. mech_fb_v1 supply=GAS),
     * ghost imports are strictly asymmetric (rank with sources gets
     * imports; rank with only sinks-of-imports gets none) — early-return
     * on num_ghosts==0 deadlocked rank-with-imports waiting for the other
     * rank's collectives. Now: NTask<=1 still returns; num_ghosts==0
     * runs the function with empty buffers, sending zeros into the
     * collective and returning when the no-op count loop ends. */
    if(NTask <= 1) return;

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    /* First pass: count non-trivial ghost entries per home-rank.
     * num_ghosts==0 → loop body never executes → all delta_send_count[t]==0. */
    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        if(ghost_full_buf[j].N_injected > 0) { delta_send_count[home_rank[g]]++; }
    }

    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_mechfb_t *send_buf = (struct ghost_delta_mechfb_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_mechfb_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        if(ghost_full_buf[j].N_injected <= 0) continue;
        int task = home_rank[g];
        int off  = pack_offset[task]++;
        send_buf[off].home_index   = home_index[g];
        send_buf[off].N_injected   = ghost_full_buf[j].N_injected;
        send_buf[off].m_injected   = ghost_full_buf[j].m_injected;
        send_buf[off].p_injected[0] = ghost_full_buf[j].p_injected[0];
        send_buf[off].p_injected[1] = ghost_full_buf[j].p_injected[1];
        send_buf[off].p_injected[2] = ghost_full_buf[j].p_injected[2];
        send_buf[off].KE_injected  = ghost_full_buf[j].KE_injected;
        send_buf[off].TE_injected  = ghost_full_buf[j].TE_injected;
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {
            send_buf[off].Z_injected[k] = ghost_full_buf[j].Z_injected[k];
        }
#if defined(GALSF_ISMDUSTCHEM_MODEL)
        send_buf[off].Mass_Where_Dust_Shocked = ghost_full_buf[j].Mass_Where_Dust_Shocked;
#endif
#if defined(COSMIC_RAY_FLUID)
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            send_buf[off].CR_energy_injected[k] = ghost_full_buf[j].CR_energy_injected[k];
        }
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            send_buf[off].CR_number_injected[k] = ghost_full_buf[j].CR_number_injected[k];
        }
#endif
        send_buf[off].CR_dir_weighted[0] = ghost_full_buf[j].CR_dir_weighted[0];
        send_buf[off].CR_dir_weighted[1] = ghost_full_buf[j].CR_dir_weighted[1];
        send_buf[off].CR_dir_weighted[2] = ghost_full_buf[j].CR_dir_weighted[2];
#endif
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_mechfb_t *recv_buf = (struct ghost_delta_mechfb_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_mechfb_t));

    gizmo_mpi_alltoallv_typed(send_buf, delta_send_count, delta_send_disp,
                              recv_buf, delta_recv_count, delta_recv_disp,
                              sizeof(struct ghost_delta_mechfb_t), MPI_COMM_WORLD);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    /* Accumulate received deltas into home_buf */
    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        if(idx < 0 || idx >= n_gas) continue;  /* safety */
        home_buf[idx].N_injected   += recv_buf[d].N_injected;
        home_buf[idx].m_injected   += recv_buf[d].m_injected;
        home_buf[idx].p_injected[0] += recv_buf[d].p_injected[0];
        home_buf[idx].p_injected[1] += recv_buf[d].p_injected[1];
        home_buf[idx].p_injected[2] += recv_buf[d].p_injected[2];
        home_buf[idx].KE_injected  += recv_buf[d].KE_injected;
        home_buf[idx].TE_injected  += recv_buf[d].TE_injected;
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {
            home_buf[idx].Z_injected[k] += recv_buf[d].Z_injected[k];
        }
#if defined(GALSF_ISMDUSTCHEM_MODEL)
        home_buf[idx].Mass_Where_Dust_Shocked += recv_buf[d].Mass_Where_Dust_Shocked;
#endif
#if defined(COSMIC_RAY_FLUID)
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            home_buf[idx].CR_energy_injected[k] += recv_buf[d].CR_energy_injected[k];
        }
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
            home_buf[idx].CR_number_injected[k] += recv_buf[d].CR_number_injected[k];
        }
#endif
        home_buf[idx].CR_dir_weighted[0] += recv_buf[d].CR_dir_weighted[0];
        home_buf[idx].CR_dir_weighted[1] += recv_buf[d].CR_dir_weighted[1];
        home_buf[idx].CR_dir_weighted[2] += recv_buf[d].CR_dir_weighted[2];
#endif
    }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (mechfb): sent %d deltas, received %d deltas\n",
               total_send, total_recv);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}

#endif /* GALSF_FB_MECHANICAL */


/* --- Grain backreaction (B7a) -------------------------------------------- */
/* Snapshot-based reverse communication for grain_backrx_evaluate j-writes:
 * Vel, VelPred, dp (additive) and Grain_AccelTimeMin (min update). */

#if defined(GRAIN_FLUID) && defined(GRAIN_BACKREACTION)

struct ghost_delta_grainbackrx_t {
    int home_index;
    MyDouble dVel[3];
    MyDouble dVelPred[3];
    MyFloat  ddp[3];
    MyFloat  Grain_AccelTimeMin;   /* absolute value (min-apply at home) */
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
    /* Phase 17b COND/SUBL: additive deltas for the gas-side fields the
     * grain_backrx pair kernel mutates under (GRAIN_EVOLUTION & 96). */
    MyFloat  dVolatileSpecies[GRAIN_NUM_VOLATILE_SPECIES];
    MyDouble dInternalEnergy;
    MyDouble dInternalEnergyPred;
#endif
};

struct ghost_snap_grainbackrx_t {
    MyDouble Vel[3];
    MyDouble VelPred[3];
    MyFloat  dp[3];
    MyFloat  Grain_AccelTimeMin;
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
    MyFloat  VolatileSpecies[GRAIN_NUM_VOLATILE_SPECIES];
    MyDouble InternalEnergy;
    MyDouble InternalEnergyPred;
#endif
};

static struct ghost_snap_grainbackrx_t *grainbackrx_snap = NULL;


void ghost_writeback_zero_grainbackrx(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    grainbackrx_snap = (struct ghost_snap_grainbackrx_t *)
        malloc(num_ghosts * sizeof(struct ghost_snap_grainbackrx_t));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        auto& s = grainbackrx_snap[g];
        s.Vel[0] = P[j].Vel[0]; s.Vel[1] = P[j].Vel[1]; s.Vel[2] = P[j].Vel[2];
        s.VelPred[0] = CellP[j].VelPred[0]; s.VelPred[1] = CellP[j].VelPred[1]; s.VelPred[2] = CellP[j].VelPred[2];
        s.dp[0] = P[j].dp[0]; s.dp[1] = P[j].dp[1]; s.dp[2] = P[j].dp[2];
        s.Grain_AccelTimeMin = P[j].Grain_AccelTimeMin;
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
        for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { s.VolatileSpecies[kv] = CellP[j].VolatileSpecies[kv]; }
        s.InternalEnergy     = CellP[j].InternalEnergy;
        s.InternalEnergyPred = CellP[j].InternalEnergyPred;
#endif
    }
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}


void ghost_writeback_grainbackrx(void)
{
    ghost_write_detector_register_writeback();
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();

    /* Multi-rank collective correctness: see ghost_writeback_agsforce. */
    if(NTask <= 1) {
        if(grainbackrx_snap) { free(grainbackrx_snap); grainbackrx_snap = NULL; }
        return;
    }
    int have_snap = (grainbackrx_snap != NULL && num_ghosts > 0);

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    if(have_snap) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
            auto& s = grainbackrx_snap[g];
            int modified = (P[j].Vel[0] != s.Vel[0] || P[j].Vel[1] != s.Vel[1] || P[j].Vel[2] != s.Vel[2]
                         || P[j].Grain_AccelTimeMin != s.Grain_AccelTimeMin);
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
            if(!modified) {
                if(CellP[j].InternalEnergy != s.InternalEnergy) { modified = 1; }
                else { for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { if(CellP[j].VolatileSpecies[kv] != s.VolatileSpecies[kv]) { modified = 1; break; } } }
            }
#endif
            if(modified) delta_send_count[home_rank[g]]++;
        }
    }

    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1];
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_grainbackrx_t *send_buf = (struct ghost_delta_grainbackrx_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_grainbackrx_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    if(have_snap) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
            auto& s = grainbackrx_snap[g];
            int modified = (P[j].Vel[0] != s.Vel[0] || P[j].Vel[1] != s.Vel[1] || P[j].Vel[2] != s.Vel[2]
                         || P[j].Grain_AccelTimeMin != s.Grain_AccelTimeMin);
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
            if(!modified) {
                if(CellP[j].InternalEnergy != s.InternalEnergy) { modified = 1; }
                else { for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { if(CellP[j].VolatileSpecies[kv] != s.VolatileSpecies[kv]) { modified = 1; break; } } }
            }
#endif
            if(!modified) continue;
            int task = home_rank[g];
            int off  = pack_offset[task]++;
            send_buf[off].home_index = home_index[g];
            send_buf[off].dVel[0] = P[j].Vel[0] - s.Vel[0];
            send_buf[off].dVel[1] = P[j].Vel[1] - s.Vel[1];
            send_buf[off].dVel[2] = P[j].Vel[2] - s.Vel[2];
            send_buf[off].dVelPred[0] = CellP[j].VelPred[0] - s.VelPred[0];
            send_buf[off].dVelPred[1] = CellP[j].VelPred[1] - s.VelPred[1];
            send_buf[off].dVelPred[2] = CellP[j].VelPred[2] - s.VelPred[2];
            send_buf[off].ddp[0] = P[j].dp[0] - s.dp[0];
            send_buf[off].ddp[1] = P[j].dp[1] - s.dp[1];
            send_buf[off].ddp[2] = P[j].dp[2] - s.dp[2];
            send_buf[off].Grain_AccelTimeMin = P[j].Grain_AccelTimeMin; /* absolute (min-apply) */
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
            for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { send_buf[off].dVolatileSpecies[kv] = CellP[j].VolatileSpecies[kv] - s.VolatileSpecies[kv]; }
            send_buf[off].dInternalEnergy     = CellP[j].InternalEnergy     - s.InternalEnergy;
            send_buf[off].dInternalEnergyPred = CellP[j].InternalEnergyPred - s.InternalEnergyPred;
#endif
        }
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1];
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_grainbackrx_t *recv_buf = (struct ghost_delta_grainbackrx_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_grainbackrx_t));

    gizmo_mpi_alltoallv_typed(send_buf, delta_send_count, delta_send_disp,
                              recv_buf, delta_recv_count, delta_recv_disp,
                              sizeof(struct ghost_delta_grainbackrx_t), MPI_COMM_WORLD);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        P[idx].Vel[0]        += recv_buf[d].dVel[0];
        P[idx].Vel[1]        += recv_buf[d].dVel[1];
        P[idx].Vel[2]        += recv_buf[d].dVel[2];
        CellP[idx].VelPred[0] += recv_buf[d].dVelPred[0];
        CellP[idx].VelPred[1] += recv_buf[d].dVelPred[1];
        CellP[idx].VelPred[2] += recv_buf[d].dVelPred[2];
        P[idx].dp[0]         += recv_buf[d].ddp[0];
        P[idx].dp[1]         += recv_buf[d].ddp[1];
        P[idx].dp[2]         += recv_buf[d].ddp[2];
        if(recv_buf[d].Grain_AccelTimeMin < P[idx].Grain_AccelTimeMin) {
            P[idx].Grain_AccelTimeMin = recv_buf[d].Grain_AccelTimeMin;
        }
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
        for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { CellP[idx].VolatileSpecies[kv] += recv_buf[d].dVolatileSpecies[kv]; }
        CellP[idx].InternalEnergy     += recv_buf[d].dInternalEnergy;
        CellP[idx].InternalEnergyPred += recv_buf[d].dInternalEnergyPred;
#endif
    }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (grainbackrx): sent %d deltas, received %d deltas\n",
               total_send, total_recv);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    free(grainbackrx_snap); grainbackrx_snap = NULL;
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}

#endif /* GRAIN_FLUID && GRAIN_BACKREACTION */



/* --- RadFBRP variant (radiation_pressure_winds GPU) ------------------------ */

#ifdef GALSF_FB_FIRE_RT_LOCALRP

struct ghost_delta_radfbrp_t {
    int home_index;
    MyDouble dVel[3];
    MyDouble dVelPred[3];
    MyDouble ddp[3];
};

struct ghost_snap_radfbrp_t {
    MyDouble Vel[3];
    MyDouble VelPred[3];
    MyDouble dp[3];
};

static struct ghost_snap_radfbrp_t *radfbrp_snap = NULL;


void ghost_writeback_zero_radfbrp(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    radfbrp_snap = (struct ghost_snap_radfbrp_t *)
        malloc(num_ghosts * sizeof(struct ghost_snap_radfbrp_t));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        auto& s = radfbrp_snap[g];
        s.Vel[0] = P[j].Vel[0]; s.Vel[1] = P[j].Vel[1]; s.Vel[2] = P[j].Vel[2];
        s.VelPred[0] = CellP[j].VelPred[0]; s.VelPred[1] = CellP[j].VelPred[1]; s.VelPred[2] = CellP[j].VelPred[2];
        s.dp[0] = P[j].dp[0]; s.dp[1] = P[j].dp[1]; s.dp[2] = P[j].dp[2];
    }
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}


void ghost_writeback_radfbrp(void)
{
    ghost_write_detector_register_writeback();
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();

    /* Multi-rank collective correctness: see ghost_writeback_agsforce. */
    if(NTask <= 1) {
        if(radfbrp_snap) { free(radfbrp_snap); radfbrp_snap = NULL; }
        return;
    }
    int have_snap = (radfbrp_snap != NULL && num_ghosts > 0);

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    if(have_snap) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
            auto& s = radfbrp_snap[g];
            int modified = (P[j].Vel[0] != s.Vel[0] || P[j].Vel[1] != s.Vel[1] || P[j].Vel[2] != s.Vel[2]);
            if(modified) delta_send_count[home_rank[g]]++;
        }
    }

    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1];
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_radfbrp_t *send_buf = (struct ghost_delta_radfbrp_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_radfbrp_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    if(have_snap) {
        for(int g = 0; g < num_ghosts; g++) {
            int j = num_local + g;
            auto& s = radfbrp_snap[g];
            int modified = (P[j].Vel[0] != s.Vel[0] || P[j].Vel[1] != s.Vel[1] || P[j].Vel[2] != s.Vel[2]);
            if(!modified) continue;
            int task = home_rank[g];
            int off  = pack_offset[task]++;
            send_buf[off].home_index = home_index[g];
            send_buf[off].dVel[0] = P[j].Vel[0] - s.Vel[0];
            send_buf[off].dVel[1] = P[j].Vel[1] - s.Vel[1];
            send_buf[off].dVel[2] = P[j].Vel[2] - s.Vel[2];
            send_buf[off].dVelPred[0] = CellP[j].VelPred[0] - s.VelPred[0];
            send_buf[off].dVelPred[1] = CellP[j].VelPred[1] - s.VelPred[1];
            send_buf[off].dVelPred[2] = CellP[j].VelPred[2] - s.VelPred[2];
            send_buf[off].ddp[0] = P[j].dp[0] - s.dp[0];
            send_buf[off].ddp[1] = P[j].dp[1] - s.dp[1];
            send_buf[off].ddp[2] = P[j].dp[2] - s.dp[2];
        }
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1];
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_radfbrp_t *recv_buf = (struct ghost_delta_radfbrp_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_radfbrp_t));

    gizmo_mpi_alltoallv_typed(send_buf, delta_send_count, delta_send_disp,
                              recv_buf, delta_recv_count, delta_recv_disp,
                              sizeof(struct ghost_delta_radfbrp_t), MPI_COMM_WORLD);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        P[idx].Vel[0]          += recv_buf[d].dVel[0];
        P[idx].Vel[1]          += recv_buf[d].dVel[1];
        P[idx].Vel[2]          += recv_buf[d].dVel[2];
        CellP[idx].VelPred[0]  += recv_buf[d].dVelPred[0];
        CellP[idx].VelPred[1]  += recv_buf[d].dVelPred[1];
        CellP[idx].VelPred[2]  += recv_buf[d].dVelPred[2];
        P[idx].dp[0]           += recv_buf[d].ddp[0];
        P[idx].dp[1]           += recv_buf[d].ddp[1];
        P[idx].dp[2]           += recv_buf[d].ddp[2];
    }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (radfbrp): sent %d deltas, received %d deltas\n",
               total_send, total_recv);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    free(radfbrp_snap); radfbrp_snap = NULL;
    gpu_particles_arena_invalidate(); /* host CellP/P updated; arena stale */
}

#endif /* GALSF_FB_FIRE_RT_LOCALRP */


/* --- RT source injection variant ------------------------------------------ */

#ifdef RT_SOURCE_INJECTION

struct ghost_delta_rtsrcinjection_t {
    int home_index;
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
    MyFloat dRad_Je[N_RT_FREQ_BINS];
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
    MyFloat dRad_E_gamma[N_RT_FREQ_BINS];
#ifdef RT_EVOLVE_ENERGY
    MyFloat dRad_E_gamma_Pred[N_RT_FREQ_BINS];
#endif
#ifdef RT_EVOLVE_INTENSITIES
    MyFloat dRad_Intensity[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
    MyFloat dRad_Intensity_Pred[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
#endif
#ifdef RT_EVOLVE_FLUX
    MyFloat dRad_Flux[N_RT_FREQ_BINS][3];
    MyFloat dRad_Flux_Pred[N_RT_FREQ_BINS][3];
#endif
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
    MyDouble dVel[3];
    MyDouble dVelPred[3];
    MyDouble ddp[3];
#endif
};

struct ghost_snap_rtsrcinjection_t {
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
    MyFloat Rad_Je[N_RT_FREQ_BINS];
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
    MyFloat Rad_E_gamma[N_RT_FREQ_BINS];
#ifdef RT_EVOLVE_ENERGY
    MyFloat Rad_E_gamma_Pred[N_RT_FREQ_BINS];
#endif
#ifdef RT_EVOLVE_INTENSITIES
    MyFloat Rad_Intensity[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
    MyFloat Rad_Intensity_Pred[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
#endif
#ifdef RT_EVOLVE_FLUX
    MyFloat Rad_Flux[N_RT_FREQ_BINS][3];
    MyFloat Rad_Flux_Pred[N_RT_FREQ_BINS][3];
#endif
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
    MyDouble Vel[3];
    MyDouble VelPred[3];
    MyDouble dp[3];
#endif
};

static struct ghost_snap_rtsrcinjection_t *rtsrcinjection_snap = NULL;

void ghost_writeback_zero_rtsrcinjection(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(NTask <= 1) return;

    rtsrcinjection_snap = (struct ghost_snap_rtsrcinjection_t *)
        malloc((num_ghosts > 0 ? num_ghosts : 1) * sizeof(struct ghost_snap_rtsrcinjection_t));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        auto& s = rtsrcinjection_snap[g];
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
            s.Rad_Je[k] = CellP[j].Rad_Je[k];
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
            s.Rad_E_gamma[k] = CellP[j].Rad_E_gamma[k];
#ifdef RT_EVOLVE_ENERGY
            s.Rad_E_gamma_Pred[k] = CellP[j].Rad_E_gamma_Pred[k];
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int kv = 0; kv < N_RT_INTENSITY_BINS; kv++) {
                s.Rad_Intensity[k][kv] = CellP[j].Rad_Intensity[k][kv];
                s.Rad_Intensity_Pred[k][kv] = CellP[j].Rad_Intensity_Pred[k][kv];
            }
#endif
#ifdef RT_EVOLVE_FLUX
            for(int kv = 0; kv < 3; kv++) {
                s.Rad_Flux[k][kv] = CellP[j].Rad_Flux[k][kv];
                s.Rad_Flux_Pred[k][kv] = CellP[j].Rad_Flux_Pred[k][kv];
            }
#endif
#endif
        }
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
        for(int kv = 0; kv < 3; kv++) {
            s.Vel[kv] = P[j].Vel[kv];
            s.VelPred[kv] = CellP[j].VelPred[kv];
            s.dp[kv] = P[j].dp[kv];
        }
#endif
    }
    gpu_particles_arena_invalidate();
}

void ghost_writeback_rtsrcinjection(void)
{
    ghost_write_detector_register_writeback();
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();

    if(!rtsrcinjection_snap || NTask <= 1) {
        if(rtsrcinjection_snap) { free(rtsrcinjection_snap); rtsrcinjection_snap = NULL; }
        return;
    }

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        auto& s = rtsrcinjection_snap[g];
        int modified = 0;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
            if(CellP[j].Rad_Je[k] != s.Rad_Je[k]) modified = 1;
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
            if(CellP[j].Rad_E_gamma[k] != s.Rad_E_gamma[k]) modified = 1;
#ifdef RT_EVOLVE_ENERGY
            if(CellP[j].Rad_E_gamma_Pred[k] != s.Rad_E_gamma_Pred[k]) modified = 1;
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int kv = 0; kv < N_RT_INTENSITY_BINS; kv++) {
                if(CellP[j].Rad_Intensity[k][kv] != s.Rad_Intensity[k][kv]) modified = 1;
                if(CellP[j].Rad_Intensity_Pred[k][kv] != s.Rad_Intensity_Pred[k][kv]) modified = 1;
            }
#endif
#ifdef RT_EVOLVE_FLUX
            for(int kv = 0; kv < 3; kv++) {
                if(CellP[j].Rad_Flux[k][kv] != s.Rad_Flux[k][kv]) modified = 1;
                if(CellP[j].Rad_Flux_Pred[k][kv] != s.Rad_Flux_Pred[k][kv]) modified = 1;
            }
#endif
#endif
        }
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
        for(int kv = 0; kv < 3; kv++) {
            if(P[j].Vel[kv] != s.Vel[kv]) modified = 1;
            if(CellP[j].VelPred[kv] != s.VelPred[kv]) modified = 1;
            if(P[j].dp[kv] != s.dp[kv]) modified = 1;
        }
#endif
        if(modified) delta_send_count[home_rank[g]]++;
    }

    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1];
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_rtsrcinjection_t *send_buf = (struct ghost_delta_rtsrcinjection_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_rtsrcinjection_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        auto& s = rtsrcinjection_snap[g];
        int modified = 0;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
            if(CellP[j].Rad_Je[k] != s.Rad_Je[k]) modified = 1;
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
            if(CellP[j].Rad_E_gamma[k] != s.Rad_E_gamma[k]) modified = 1;
#ifdef RT_EVOLVE_ENERGY
            if(CellP[j].Rad_E_gamma_Pred[k] != s.Rad_E_gamma_Pred[k]) modified = 1;
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int kv = 0; kv < N_RT_INTENSITY_BINS; kv++) {
                if(CellP[j].Rad_Intensity[k][kv] != s.Rad_Intensity[k][kv]) modified = 1;
                if(CellP[j].Rad_Intensity_Pred[k][kv] != s.Rad_Intensity_Pred[k][kv]) modified = 1;
            }
#endif
#ifdef RT_EVOLVE_FLUX
            for(int kv = 0; kv < 3; kv++) {
                if(CellP[j].Rad_Flux[k][kv] != s.Rad_Flux[k][kv]) modified = 1;
                if(CellP[j].Rad_Flux_Pred[k][kv] != s.Rad_Flux_Pred[k][kv]) modified = 1;
            }
#endif
#endif
        }
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
        for(int kv = 0; kv < 3; kv++) {
            if(P[j].Vel[kv] != s.Vel[kv]) modified = 1;
            if(CellP[j].VelPred[kv] != s.VelPred[kv]) modified = 1;
            if(P[j].dp[kv] != s.dp[kv]) modified = 1;
        }
#endif
        if(!modified) continue;
        int task = home_rank[g];
        int off  = pack_offset[task]++;
        send_buf[off].home_index = home_index[g];
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
            send_buf[off].dRad_Je[k] = CellP[j].Rad_Je[k] - s.Rad_Je[k];
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
            send_buf[off].dRad_E_gamma[k] = CellP[j].Rad_E_gamma[k] - s.Rad_E_gamma[k];
#ifdef RT_EVOLVE_ENERGY
            send_buf[off].dRad_E_gamma_Pred[k] = CellP[j].Rad_E_gamma_Pred[k] - s.Rad_E_gamma_Pred[k];
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int kv = 0; kv < N_RT_INTENSITY_BINS; kv++) {
                send_buf[off].dRad_Intensity[k][kv] = CellP[j].Rad_Intensity[k][kv] - s.Rad_Intensity[k][kv];
                send_buf[off].dRad_Intensity_Pred[k][kv] = CellP[j].Rad_Intensity_Pred[k][kv] - s.Rad_Intensity_Pred[k][kv];
            }
#endif
#ifdef RT_EVOLVE_FLUX
            for(int kv = 0; kv < 3; kv++) {
                send_buf[off].dRad_Flux[k][kv] = CellP[j].Rad_Flux[k][kv] - s.Rad_Flux[k][kv];
                send_buf[off].dRad_Flux_Pred[k][kv] = CellP[j].Rad_Flux_Pred[k][kv] - s.Rad_Flux_Pred[k][kv];
            }
#endif
#endif
        }
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
        for(int kv = 0; kv < 3; kv++) {
            send_buf[off].dVel[kv] = P[j].Vel[kv] - s.Vel[kv];
            send_buf[off].dVelPred[kv] = CellP[j].VelPred[kv] - s.VelPred[kv];
            send_buf[off].ddp[kv] = P[j].dp[kv] - s.dp[kv];
        }
#endif
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1];
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_rtsrcinjection_t *recv_buf = (struct ghost_delta_rtsrcinjection_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_rtsrcinjection_t));

    gizmo_mpi_alltoallv_typed(send_buf, delta_send_count, delta_send_disp,
                              recv_buf, delta_recv_count, delta_recv_disp,
                              sizeof(struct ghost_delta_rtsrcinjection_t), MPI_COMM_WORLD);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {
#if !defined(RT_INJECT_PHOTONS_DISCRETELY)
            CellP[idx].Rad_Je[k] += recv_buf[d].dRad_Je[k];
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
            CellP[idx].Rad_E_gamma[k] += recv_buf[d].dRad_E_gamma[k];
#ifdef RT_EVOLVE_ENERGY
            CellP[idx].Rad_E_gamma_Pred[k] += recv_buf[d].dRad_E_gamma_Pred[k];
#endif
#ifdef RT_EVOLVE_INTENSITIES
            for(int kv = 0; kv < N_RT_INTENSITY_BINS; kv++) {
                CellP[idx].Rad_Intensity[k][kv] += recv_buf[d].dRad_Intensity[k][kv];
                CellP[idx].Rad_Intensity_Pred[k][kv] += recv_buf[d].dRad_Intensity_Pred[k][kv];
            }
#endif
#ifdef RT_EVOLVE_FLUX
            for(int kv = 0; kv < 3; kv++) {
                CellP[idx].Rad_Flux[k][kv] += recv_buf[d].dRad_Flux[k][kv];
                CellP[idx].Rad_Flux_Pred[k][kv] += recv_buf[d].dRad_Flux_Pred[k][kv];
            }
#endif
#endif
        }
#if defined(RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION) && !defined(RT_DISABLE_RAD_PRESSURE)
        for(int kv = 0; kv < 3; kv++) {
            P[idx].Vel[kv] += recv_buf[d].dVel[kv];
            CellP[idx].VelPred[kv] += recv_buf[d].dVelPred[kv];
            P[idx].dp[kv] += recv_buf[d].ddp[kv];
        }
#endif
    }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (rtsrcinjection): sent %d deltas, received %d deltas\n",
               total_send, total_recv);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    free(rtsrcinjection_snap); rtsrcinjection_snap = NULL;
    gpu_particles_arena_invalidate();
}

#endif /* RT_SOURCE_INJECTION */


/* --- Ghost-write detector (debug builds only) ----------------------------- */

#ifdef GIZMO_GPU_ARENA_DEBUG

static struct particle_data *gwd_P_snap = NULL;
static struct gas_cell_data *gwd_CellP_snap = NULL;
static int gwd_n_snap = 0;
static int gwd_local_at_snap = 0;
static int gwd_wb_count_at_snap = 0;
static int gwd_wb_count = 0;
static const char *gwd_kernel = "(none)";
static int gwd_active = 0;

void ghost_write_detector_register_writeback(void) { gwd_wb_count++; }

/* Re-snapshot ghost P/CellP at the current state. Called by gpu_ngb_list_build
 * after its lazy-drift loop updates Ti_current + Pos on touched ghost slots —
 * those updates are NOT a "kernel write" the detector should flag (they're
 * caller-side predicted-state setup), so we move the comparison baseline to
 * the post-drift state. The remaining detector window (kernel + scatter) then
 * catches any genuine kernel-side writes that need a writeback. */
void ghost_write_detector_resnapshot_after_lazy_drift(void)
{
    if(!gwd_active) return;
    if(gwd_n_snap <= 0) return;
    int local = ghost_get_num_local();
    int n = ghost_get_num_ghosts();
    /* If layout changed mid-window we already can't compare — leave the
     * snapshot alone; the original alarm logic handles it. */
    if(local != gwd_local_at_snap || n != gwd_n_snap) return;
    memcpy(gwd_P_snap,    P     + local, n * sizeof(struct particle_data));
    memcpy(gwd_CellP_snap, CellP + local, n * sizeof(struct gas_cell_data));
}

void ghost_write_detector_begin(const char *kernel_name)
{
    if(NTask <= 1) return;
    int n = ghost_get_num_ghosts();
    if(n <= 0) return;
    int local = ghost_get_num_local();
    if(gwd_active) {
        /* Nested begin without an end — programmer error. */
        fprintf(stderr, "[ghost_write_detector] nested begin('%s') while '%s' still active\n",
                kernel_name ? kernel_name : "(null)", gwd_kernel);
        endrun(91234);
    }
    gwd_P_snap = (struct particle_data *) malloc(n * sizeof(struct particle_data));
    gwd_CellP_snap = (struct gas_cell_data *) malloc(n * sizeof(struct gas_cell_data));
    memcpy(gwd_P_snap,    P     + local, n * sizeof(struct particle_data));
    memcpy(gwd_CellP_snap, CellP + local, n * sizeof(struct gas_cell_data));
    gwd_n_snap = n;
    gwd_local_at_snap = local;
    gwd_wb_count_at_snap = gwd_wb_count;
    gwd_kernel = kernel_name ? kernel_name : "(unnamed)";
    gwd_active = 1;
}

void ghost_write_detector_end(void)
{
    if(!gwd_active) return;
    int wb_ran = (gwd_wb_count > gwd_wb_count_at_snap);
    if(!wb_ran) {
        /* Use the local count captured at begin(); if it changed, ghost layout
         * was rebuilt mid-window — treat as a write-detection failure. */
        int local_now = ghost_get_num_local();
        int n_now = ghost_get_num_ghosts();
        int n_diff = 0, first_diff = -1;
        if(local_now == gwd_local_at_snap && n_now == gwd_n_snap) {
            for(int g = 0; g < gwd_n_snap; g++) {
                if(memcmp(&P[gwd_local_at_snap + g],     &gwd_P_snap[g],     sizeof(struct particle_data)) != 0 ||
                   memcmp(&CellP[gwd_local_at_snap + g], &gwd_CellP_snap[g], sizeof(struct gas_cell_data)) != 0) {
                    n_diff++; if(first_diff < 0) first_diff = g;
                }
            }
        } else {
            n_diff = -1; /* layout changed — can't compare */
        }
        if(n_diff != 0) {
            fprintf(stderr, "[ghost_write_detector] kernel '%s' modified ghost particles "
                    "but no ghost_writeback_* was called.\n"
                    "  ghost_count_at_begin=%d, local_at_begin=%d, ghost_count_now=%d, local_now=%d, "
                    "first_differing_ghost_index=%d, n_diff=%d\n"
                    "  → either add a ghost_writeback call, or confirm this kernel is read-only on j.\n",
                    gwd_kernel, gwd_n_snap, gwd_local_at_snap, n_now, local_now,
                    first_diff, n_diff);
            /* Byte-level diff for the first differing slot — narrows down
             * which struct field the kernel mutated. */
            if(first_diff >= 0) {
                const unsigned char *cur_P = (const unsigned char *) &P[gwd_local_at_snap + first_diff];
                const unsigned char *snap_P = (const unsigned char *) &gwd_P_snap[first_diff];
                size_t sz = sizeof(struct particle_data);
                size_t first_byte = (size_t)-1, last_byte = 0, diff_bytes = 0;
                for(size_t b = 0; b < sz; b++) {
                    if(cur_P[b] != snap_P[b]) {
                        if(first_byte == (size_t)-1) first_byte = b;
                        last_byte = b; diff_bytes++;
                    }
                }
                fprintf(stderr, "  P[]: diff at bytes [%zu..%zu], %zu bytes total within sizeof(particle_data)=%zu\n",
                        first_byte, last_byte, diff_bytes, sz);
                if(first_byte != (size_t)-1) {
                    fprintf(stderr, "  P[].first_byte = 0x%02x → 0x%02x (snap → now)\n",
                            snap_P[first_byte], cur_P[first_byte]);
                }
                const unsigned char *cur_C = (const unsigned char *) &CellP[gwd_local_at_snap + first_diff];
                const unsigned char *snap_C = (const unsigned char *) &gwd_CellP_snap[first_diff];
                size_t sc = sizeof(struct gas_cell_data);
                size_t first_byteC = (size_t)-1, last_byteC = 0, diff_bytesC = 0;
                for(size_t b = 0; b < sc; b++) {
                    if(cur_C[b] != snap_C[b]) {
                        if(first_byteC == (size_t)-1) first_byteC = b;
                        last_byteC = b; diff_bytesC++;
                    }
                }
                fprintf(stderr, "  CellP[]: diff at bytes [%zu..%zu], %zu bytes total within sizeof(gas_cell_data)=%zu\n",
                        first_byteC, last_byteC, diff_bytesC, sc);
            }
            endrun(91234);
        }
    }
    free(gwd_P_snap);     gwd_P_snap = NULL;
    free(gwd_CellP_snap); gwd_CellP_snap = NULL;
    gwd_n_snap = 0;
    gwd_active = 0;
}

#endif /* GIZMO_GPU_ARENA_DEBUG */
