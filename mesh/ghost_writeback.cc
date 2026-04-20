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
}


void ghost_writeback_hydro(void)
{
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

    int delta_size = sizeof(struct ghost_delta_hydro_t);
    int *send_bytes = (int *) malloc(NTask * sizeof(int));
    int *recv_bytes = (int *) malloc(NTask * sizeof(int));
    int *send_bdisp = (int *) malloc(NTask * sizeof(int));
    int *recv_bdisp = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        send_bytes[t] = delta_send_count[t] * delta_size;
        recv_bytes[t] = delta_recv_count[t] * delta_size;
        send_bdisp[t] = delta_send_disp[t] * delta_size;
        recv_bdisp[t] = delta_recv_disp[t] * delta_size;
    }

    MPI_Alltoallv(send_buf, send_bytes, send_bdisp, MPI_BYTE,
                  recv_buf, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);

    free(send_bytes); free(recv_bytes); free(send_bdisp); free(recv_bdisp);
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
}


/* --- Wakeup-only variant ------------------------------------------------- */

struct ghost_delta_wakeup_t {
    int home_index;
    short int wakeup;
};


void ghost_writeback_zero_wakeup(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();
    if(num_ghosts <= 0) return;
    for(int g = 0; g < num_ghosts; g++) { P[num_local + g].wakeup = 0; }
}


void ghost_writeback_wakeup(void)
{
    if(NTask <= 1) return;

    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();
    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        if(P[num_local + g].wakeup != 0) { delta_send_count[home_rank[g]]++; }
    }
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_wakeup_t *send_buf = (struct ghost_delta_wakeup_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_wakeup_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        if(P[j].wakeup == 0) continue;
        int task = home_rank[g];
        int off = pack_offset[task]++;
        send_buf[off].home_index = home_index[g];
        send_buf[off].wakeup = P[j].wakeup;
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_wakeup_t *recv_buf = (struct ghost_delta_wakeup_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_wakeup_t));

    int delta_size = sizeof(struct ghost_delta_wakeup_t);
    int *send_bytes = (int *) malloc(NTask * sizeof(int));
    int *recv_bytes = (int *) malloc(NTask * sizeof(int));
    int *send_bdisp = (int *) malloc(NTask * sizeof(int));
    int *recv_bdisp = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        send_bytes[t] = delta_send_count[t] * delta_size;
        recv_bytes[t] = delta_recv_count[t] * delta_size;
        send_bdisp[t] = delta_send_disp[t] * delta_size;
        recv_bdisp[t] = delta_recv_disp[t] * delta_size;
    }
    MPI_Alltoallv(send_buf, send_bytes, send_bdisp, MPI_BYTE,
                  recv_buf, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);
    free(send_bytes); free(recv_bytes); free(send_bdisp); free(recv_bdisp);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    int wakeups_applied = 0;
    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        if(recv_buf[d].wakeup > P[idx].wakeup) {
            P[idx].wakeup = recv_buf[d].wakeup;
            wakeups_applied++;
        }
    }
    if(wakeups_applied > 0) { NeedToWakeupParticles_local = 1; }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (wakeup): sent %d deltas, received %d deltas, %d wakeups applied\n",
               total_send, total_recv, wakeups_applied);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
}


/* --- AGSForce variant --------------------------------------------------- */

/* Reverse-communicates the j-side AGSForce delta set:
 *   - Vel delta (ADD)    [always present — AGS core carries Vel field]
 *   - dp delta  (ADD)    [populated under DM_SIDM]
 *   - NInteractions  (ADD)  [DM_SIDM only]
 *   - wakeup          (MAX in {0,-1}) [CBE + SIDM wakeup condition]
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
}


void ghost_writeback_agsforce(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local = ghost_get_num_local();

    if(NTask <= 1 || num_ghosts <= 0) {
        if(agsforce_ghost_Vel0) { free(agsforce_ghost_Vel0); agsforce_ghost_Vel0 = NULL; }
        if(agsforce_ghost_dp0)  { free(agsforce_ghost_dp0);  agsforce_ghost_dp0 = NULL; }
#if defined(DM_SIDM)
        if(agsforce_ghost_NInt0) { free(agsforce_ghost_NInt0); agsforce_ghost_NInt0 = NULL; }
#endif
        return;
    }

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
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
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_agsforce_t *send_buf = (struct ghost_delta_agsforce_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_agsforce_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

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
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_agsforce_t *recv_buf = (struct ghost_delta_agsforce_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_agsforce_t));

    int delta_size = sizeof(struct ghost_delta_agsforce_t);
    int *send_bytes = (int *) malloc(NTask * sizeof(int));
    int *recv_bytes = (int *) malloc(NTask * sizeof(int));
    int *send_bdisp = (int *) malloc(NTask * sizeof(int));
    int *recv_bdisp = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        send_bytes[t] = delta_send_count[t] * delta_size;
        recv_bytes[t] = delta_recv_count[t] * delta_size;
        send_bdisp[t] = delta_send_disp[t] * delta_size;
        recv_bdisp[t] = delta_recv_disp[t] * delta_size;
    }
    MPI_Alltoallv(send_buf, send_bytes, send_bdisp, MPI_BYTE,
                  recv_buf, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);
    free(send_bytes); free(recv_bytes); free(send_bdisp); free(recv_bdisp);
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
}


/* --- SwallowTime variant ------------------------------------------------- */
/* SwallowTime exists only when SINGLE_STAR_SINK_DYNAMICS is enabled. */
#ifdef SINGLE_STAR_SINK_DYNAMICS

struct ghost_delta_swallowtime_t {
    int home_index;
    MyFloat SwallowTime;
};

static MyFloat *swallowtime_ghost0 = NULL;


void ghost_writeback_zero_swallowtime(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    swallowtime_ghost0 = (MyFloat *) malloc(num_ghosts * sizeof(MyFloat));
    for(int g = 0; g < num_ghosts; g++) {
        swallowtime_ghost0[g] = P[num_local + g].SwallowTime;
    }
}


void ghost_writeback_swallowtime(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();

    if(!swallowtime_ghost0 || NTask <= 1 || num_ghosts <= 0) {
        if(swallowtime_ghost0) { free(swallowtime_ghost0); swallowtime_ghost0 = NULL; }
        return;
    }

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        if(P[num_local + g].SwallowTime < swallowtime_ghost0[g]) { delta_send_count[home_rank[g]]++; }
    }
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_swallowtime_t *send_buf = (struct ghost_delta_swallowtime_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_swallowtime_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        if(P[j].SwallowTime >= swallowtime_ghost0[g]) continue;
        int task = home_rank[g];
        int off = pack_offset[task]++;
        send_buf[off].home_index = home_index[g];
        send_buf[off].SwallowTime = P[j].SwallowTime;
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_swallowtime_t *recv_buf = (struct ghost_delta_swallowtime_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_swallowtime_t));

    int delta_size = sizeof(struct ghost_delta_swallowtime_t);
    int *send_bytes = (int *) malloc(NTask * sizeof(int));
    int *recv_bytes = (int *) malloc(NTask * sizeof(int));
    int *send_bdisp = (int *) malloc(NTask * sizeof(int));
    int *recv_bdisp = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        send_bytes[t] = delta_send_count[t] * delta_size;
        recv_bytes[t] = delta_recv_count[t] * delta_size;
        send_bdisp[t] = delta_send_disp[t] * delta_size;
        recv_bdisp[t] = delta_recv_disp[t] * delta_size;
    }
    MPI_Alltoallv(send_buf, send_bytes, send_bdisp, MPI_BYTE,
                  recv_buf, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);
    free(send_bytes); free(recv_bytes); free(send_bdisp); free(recv_bdisp);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        if(recv_buf[d].SwallowTime < P[idx].SwallowTime) { P[idx].SwallowTime = recv_buf[d].SwallowTime; }
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    free(swallowtime_ghost0); swallowtime_ghost0 = NULL;
}

#endif /* SINGLE_STAR_SINK_DYNAMICS */


/* --- ThermalFB variant --------------------------------------------------- */
/* Snapshot-based reverse communication for addthermalFB_evaluate j-writes.
 * Fields written per-pair: Mass, Density, dp[3], InternalEnergy,
 * InternalEnergyPred, and optionally MassTrue, Metallicity[], DelayTimeCoolingSNe. */

#ifdef GALSF_FB_THERMAL

struct ghost_delta_thermalfb_t {
    int home_index;
    MyDouble dMass;
    MyDouble dDensity;
    MyDouble ddp[3];
    MyDouble dInternalEnergy;
    MyDouble dInternalEnergyPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble dMassTrue;
#endif
#ifdef METALS
    MyDouble dMetallicity[NUM_METAL_SPECIES];
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
    MyDouble dDelayCool;
#endif
};

/* Snapshot storage */
struct ghost_snap_thermalfb_t {
    MyDouble Mass;
    MyDouble Density;
    MyDouble dp[3];
    MyDouble InternalEnergy;
    MyDouble InternalEnergyPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble MassTrue;
#endif
#ifdef METALS
    MyDouble Metallicity[NUM_METAL_SPECIES];
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
    MyDouble DelayCool;
#endif
};

static struct ghost_snap_thermalfb_t *thermalfb_snap = NULL;


void ghost_writeback_zero_thermalfb(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    thermalfb_snap = (struct ghost_snap_thermalfb_t *)
        malloc(num_ghosts * sizeof(struct ghost_snap_thermalfb_t));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        auto& s = thermalfb_snap[g];
        s.Mass              = P[j].Mass;
        s.Density           = CellP[j].Density;
        s.dp[0] = P[j].dp[0]; s.dp[1] = P[j].dp[1]; s.dp[2] = P[j].dp[2];
        s.InternalEnergy     = CellP[j].InternalEnergy;
        s.InternalEnergyPred = CellP[j].InternalEnergyPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        s.MassTrue = CellP[j].MassTrue;
#endif
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) { s.Metallicity[k] = P[j].Metallicity[k]; }
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
        s.DelayCool = CellP[j].DelayTimeCoolingSNe;
#endif
    }
}


void ghost_writeback_thermalfb(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();

    if(!thermalfb_snap || NTask <= 1 || num_ghosts <= 0) {
        if(thermalfb_snap) { free(thermalfb_snap); thermalfb_snap = NULL; }
        return;
    }

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        if(P[j].Mass != thermalfb_snap[g].Mass) { delta_send_count[home_rank[g]]++; }
    }
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_thermalfb_t *send_buf = (struct ghost_delta_thermalfb_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_thermalfb_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        if(P[j].Mass == thermalfb_snap[g].Mass) continue;

        int task = home_rank[g];
        int off  = pack_offset[task]++;
        auto& s  = thermalfb_snap[g];
        send_buf[off].home_index        = home_index[g];
        send_buf[off].dMass             = P[j].Mass             - s.Mass;
        send_buf[off].dDensity          = CellP[j].Density       - s.Density;
        send_buf[off].ddp[0]            = P[j].dp[0]             - s.dp[0];
        send_buf[off].ddp[1]            = P[j].dp[1]             - s.dp[1];
        send_buf[off].ddp[2]            = P[j].dp[2]             - s.dp[2];
        send_buf[off].dInternalEnergy    = CellP[j].InternalEnergy     - s.InternalEnergy;
        send_buf[off].dInternalEnergyPred = CellP[j].InternalEnergyPred - s.InternalEnergyPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        send_buf[off].dMassTrue         = CellP[j].MassTrue      - s.MassTrue;
#endif
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {
            send_buf[off].dMetallicity[k] = P[j].Metallicity[k] - s.Metallicity[k];
        }
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
        send_buf[off].dDelayCool        = CellP[j].DelayTimeCoolingSNe - s.DelayCool;
#endif
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_thermalfb_t *recv_buf = (struct ghost_delta_thermalfb_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_thermalfb_t));

    int delta_size = sizeof(struct ghost_delta_thermalfb_t);
    int *send_bytes = (int *) malloc(NTask * sizeof(int));
    int *recv_bytes = (int *) malloc(NTask * sizeof(int));
    int *send_bdisp = (int *) malloc(NTask * sizeof(int));
    int *recv_bdisp = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        send_bytes[t] = delta_send_count[t] * delta_size;
        recv_bytes[t] = delta_recv_count[t] * delta_size;
        send_bdisp[t] = delta_send_disp[t] * delta_size;
        recv_bdisp[t] = delta_recv_disp[t] * delta_size;
    }
    MPI_Alltoallv(send_buf, send_bytes, send_bdisp, MPI_BYTE,
                  recv_buf, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);
    free(send_bytes); free(recv_bytes); free(send_bdisp); free(recv_bdisp);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        P[idx].Mass                     += recv_buf[d].dMass;
        if(P[idx].Mass < 0 || P[idx].Mass != P[idx].Mass) P[idx].Mass = 0;
        CellP[idx].Density              += recv_buf[d].dDensity;
        P[idx].dp[0]                    += recv_buf[d].ddp[0];
        P[idx].dp[1]                    += recv_buf[d].ddp[1];
        P[idx].dp[2]                    += recv_buf[d].ddp[2];
        CellP[idx].InternalEnergy       += recv_buf[d].dInternalEnergy;
        CellP[idx].InternalEnergyPred   += recv_buf[d].dInternalEnergyPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP[idx].MassTrue             += recv_buf[d].dMassTrue;
#endif
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {
            P[idx].Metallicity[k] += recv_buf[d].dMetallicity[k];
        }
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
        CellP[idx].DelayTimeCoolingSNe  += recv_buf[d].dDelayCool;
#endif
    }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (thermalfb): sent %d deltas, received %d deltas\n",
               total_send, total_recv);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    free(thermalfb_snap); thermalfb_snap = NULL;
}

#endif /* GALSF_FB_THERMAL */


/* --- SinkFeed variant ---------------------------------------------------- */
/* Snapshot-based reverse communication for sink_feed_evaluate j-writes.
 * Fields: P[j].SwallowID (set-if-zero at home), and optionally
 * CellP[j].Injected_Sink_Energy (additive, SINK_THERMALFEEDBACK). */

#ifdef SINK_PARTICLES

struct ghost_delta_sinkfeed_t {
    int home_index;
    MyIDType SwallowID;
#ifdef SINK_THERMALFEEDBACK
    MyFloat dInjected_Sink_Energy;
#endif
};

struct ghost_snap_sinkfeed_t {
    MyIDType SwallowID;
#ifdef SINK_THERMALFEEDBACK
    MyFloat Injected_Sink_Energy;
#endif
};

static struct ghost_snap_sinkfeed_t *sinkfeed_snap = NULL;


void ghost_writeback_zero_sinkfeed(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();
    if(num_ghosts <= 0) return;

    sinkfeed_snap = (struct ghost_snap_sinkfeed_t *)
        malloc(num_ghosts * sizeof(struct ghost_snap_sinkfeed_t));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        sinkfeed_snap[g].SwallowID = P[j].SwallowID;
#ifdef SINK_THERMALFEEDBACK
        sinkfeed_snap[g].Injected_Sink_Energy = CellP[j].Injected_Sink_Energy;
#endif
    }
}


void ghost_writeback_sinkfeed(void)
{
    int num_ghosts = ghost_get_num_ghosts();
    int num_local  = ghost_get_num_local();

    if(!sinkfeed_snap || NTask <= 1 || num_ghosts <= 0) {
        if(sinkfeed_snap) { free(sinkfeed_snap); sinkfeed_snap = NULL; }
        return;
    }

    int *home_rank  = ghost_get_home_rank();
    int *home_index = ghost_get_home_index();

    int *delta_send_count = (int *) calloc(NTask, sizeof(int));
    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        int changed = (P[j].SwallowID != sinkfeed_snap[g].SwallowID);
#ifdef SINK_THERMALFEEDBACK
        changed |= (CellP[j].Injected_Sink_Energy != sinkfeed_snap[g].Injected_Sink_Energy);
#endif
        if(changed) { delta_send_count[home_rank[g]]++; }
    }
    int *delta_send_disp = (int *) malloc(NTask * sizeof(int));
    delta_send_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_send_disp[t] = delta_send_disp[t-1] + delta_send_count[t-1]; }
    int total_send = delta_send_disp[NTask-1] + delta_send_count[NTask-1];

    struct ghost_delta_sinkfeed_t *send_buf = (struct ghost_delta_sinkfeed_t *)
        malloc((total_send > 0 ? total_send : 1) * sizeof(struct ghost_delta_sinkfeed_t));
    int *pack_offset = (int *) malloc(NTask * sizeof(int));
    memcpy(pack_offset, delta_send_disp, NTask * sizeof(int));

    for(int g = 0; g < num_ghosts; g++) {
        int j = num_local + g;
        int changed = (P[j].SwallowID != sinkfeed_snap[g].SwallowID);
#ifdef SINK_THERMALFEEDBACK
        changed |= (CellP[j].Injected_Sink_Energy != sinkfeed_snap[g].Injected_Sink_Energy);
#endif
        if(!changed) continue;

        int task = home_rank[g];
        int off  = pack_offset[task]++;
        send_buf[off].home_index = home_index[g];
        send_buf[off].SwallowID  = P[j].SwallowID;
#ifdef SINK_THERMALFEEDBACK
        send_buf[off].dInjected_Sink_Energy =
            CellP[j].Injected_Sink_Energy - sinkfeed_snap[g].Injected_Sink_Energy;
#endif
    }
    free(pack_offset);

    int *delta_recv_count = (int *) calloc(NTask, sizeof(int));
    MPI_Alltoall(delta_send_count, 1, MPI_INT, delta_recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    int *delta_recv_disp = (int *) malloc(NTask * sizeof(int));
    delta_recv_disp[0] = 0;
    for(int t = 1; t < NTask; t++) { delta_recv_disp[t] = delta_recv_disp[t-1] + delta_recv_count[t-1]; }
    int total_recv = delta_recv_disp[NTask-1] + delta_recv_count[NTask-1];

    struct ghost_delta_sinkfeed_t *recv_buf = (struct ghost_delta_sinkfeed_t *)
        malloc((total_recv > 0 ? total_recv : 1) * sizeof(struct ghost_delta_sinkfeed_t));

    int delta_size = sizeof(struct ghost_delta_sinkfeed_t);
    int *send_bytes = (int *) malloc(NTask * sizeof(int));
    int *recv_bytes = (int *) malloc(NTask * sizeof(int));
    int *send_bdisp = (int *) malloc(NTask * sizeof(int));
    int *recv_bdisp = (int *) malloc(NTask * sizeof(int));
    for(int t = 0; t < NTask; t++) {
        send_bytes[t] = delta_send_count[t] * delta_size;
        recv_bytes[t] = delta_recv_count[t] * delta_size;
        send_bdisp[t] = delta_send_disp[t] * delta_size;
        recv_bdisp[t] = delta_recv_disp[t] * delta_size;
    }
    MPI_Alltoallv(send_buf, send_bytes, send_bdisp, MPI_BYTE,
                  recv_buf, recv_bytes, recv_bdisp, MPI_BYTE, MPI_COMM_WORLD);
    free(send_bytes); free(recv_bytes); free(send_bdisp); free(recv_bdisp);
    free(send_buf); free(delta_send_count); free(delta_send_disp);

    for(int d = 0; d < total_recv; d++) {
        int idx = recv_buf[d].home_index;
        if(recv_buf[d].SwallowID > 0 && P[idx].SwallowID == 0) {
            P[idx].SwallowID = recv_buf[d].SwallowID;
        }
#ifdef SINK_THERMALFEEDBACK
        CellP[idx].Injected_Sink_Energy += recv_buf[d].dInjected_Sink_Energy;
#endif
    }

    if(ThisTask == 0 && (total_send > 0 || total_recv > 0)) {
        printf("  Ghost writeback (sinkfeed): sent %d deltas, received %d deltas\n",
               total_send, total_recv);
        fflush(stdout);
    }

    free(recv_buf); free(delta_recv_count); free(delta_recv_disp);
    free(sinkfeed_snap); sinkfeed_snap = NULL;
}

#endif /* SINK_PARTICLES */
