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
