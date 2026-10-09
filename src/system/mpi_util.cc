/** \file
    MPI utility functions.
*/
/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * in part (cleaned up, some routines re-organized and consolidated and a 
 * couple others added, and updated in various places to properly interact with newer
 * libraries and compilers) by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"


/* Persistent node-local (shared-memory) communicator and the counts derived from
   it. MPI_COMM_WORLD is split once by shared-memory node and the result is kept for
   the run, so anything that needs node-scoped grouping (node count, per-node memory
   aggregation) uses the same single communicator rather than re-splitting. */
MPI_Comm GizmoNodeComm      = MPI_COMM_NULL;
int      GizmoNodeRankOfTask = 0;   /* this task's rank within its node */
int      GizmoRanksThisNode  = 1;   /* MPI tasks sharing this node */
int      GizmoNodeCount      = 1;   /* number of distinct shared-memory nodes */

/** Build the persistent node-local communicator and its derived counts. The first
    (initializing) call is COLLECTIVE over MPI_COMM_WORLD -- it does an Allreduce to
    count nodes -- so it must run on all ranks; it is invoked once at startup. Later
    calls are local no-ops. Not for arbitrary subset-of-ranks use. */
void gizmo_node_comm_init(void)
{
    if(GizmoNodeComm != MPI_COMM_NULL) {return;}
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &GizmoNodeComm);
    gizmo_mpi_set_failfast_errhandler(GizmoNodeComm);  /* fail-fast, not MPI's default wedge-prone abort */
    MPI_Comm_rank(GizmoNodeComm, &GizmoNodeRankOfTask);
    MPI_Comm_size(GizmoNodeComm, &GizmoRanksThisNode);
    int is_node_lead = (GizmoNodeRankOfTask == 0) ? 1 : 0;
    MPI_Allreduce(&is_node_lead, &GizmoNodeCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

/** Number of unique -nodes- (shared memory machine structures), not MPI tasks, on
    which we are running. Machine-independent internal check for memory allocation
    purposes; derived from the persistent node-local communicator. */
int getNodeCount(void)
{
    gizmo_node_comm_init();
    return GizmoNodeCount;
}



int MPI_Sizelimited_Sendrecv(void *sendbuf0, size_t sendcount, MPI_Datatype sendtype,
                             int dest, int sendtag, void *recvbuf0, size_t recvcount,
                             MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                             MPI_Status *status)
{
    int iter = 0, size_sendtype, size_recvtype, send_now, recv_now;
    char *sendbuf = (char *)sendbuf0;
    char *recvbuf = (char *)recvbuf0;

    if(dest != source) gizmo_fatal_hard_exit_reviewed(90002004, "REVIEWED_HARD_MID_PROTOCOL: mpi_util dest!=source invariant (mid-collective exchange, no symmetric poll)", __FILE__, __LINE__, __FUNCTION__);
    
    MPI_Type_size(sendtype, &size_sendtype);
    MPI_Type_size(recvtype, &size_recvtype);
    
    if(dest == ThisTask)
    {
        memcpy(recvbuf, sendbuf, recvcount * size_recvtype);
        return 0;
    }
    
    size_t count_limit = (((long long)All.CommChunkSize)*1024LL * 1024LL) / size_sendtype;
    size_t count_limit_intmax = INT_MAX;
    if(count_limit_intmax < count_limit) {count_limit = count_limit_intmax;}
    
    while(sendcount > 0 || recvcount > 0)
    {
        if(sendcount > count_limit)
        {
            send_now = count_limit;
            iter++;
        }
        else
            send_now = sendcount;
        
        if(recvcount > count_limit)
            recv_now = count_limit;
        else
            recv_now = recvcount;
        
        MPI_Sendrecv(sendbuf, send_now, sendtype, dest, sendtag,
                     recvbuf, recv_now, recvtype, source, recvtag, comm, status);
        
        sendcount -= send_now;
        recvcount -= recv_now;
        
        sendbuf += send_now * size_sendtype;
        recvbuf += recv_now * size_recvtype;
    }
    return 0;
}


/* MPI must not receive directly into the particle storage (P, CellP) where that storage is CUDA managed memory
 * (gizmo_particle_storage_needs_staged_mpi_receive): such writes can be lost.  There, receives meant for the
 * particle storage land in one host buffer, allocated once at start-up, and are copied in from it, a chunk of at
 * most PARTICLE_RECEIVE_STAGING_BYTES at a time.  Everywhere else nothing is allocated and each call makes
 * exactly the call it replaces.  Only for receives into the particle storage; other buffers keep their own
 * MPI calls. */
#define PARTICLE_RECEIVE_STAGING_BYTES ((size_t)64 * 1024 * 1024)
static char *ParticleReceiveStaging = NULL;

void particle_receive_staging_init(void)
{
    if(ParticleReceiveStaging || !gizmo_particle_storage_needs_staged_mpi_receive()) {return;}
    ParticleReceiveStaging = (char *) malloc(PARTICLE_RECEIVE_STAGING_BYTES);
    if(!ParticleReceiveStaging) {
        printf("task %d: could not allocate the %g MB host buffer that particle-storage receives are staged through\n",
               ThisTask, (double) PARTICLE_RECEIVE_STAGING_BYTES / (1024.0 * 1024.0));
        fflush(stdout);
        gizmo_request_controlled_stop(7744, "particle_receive_staging_init: no host buffer for staged particle-storage receives",
                                      __FILE__, __LINE__, __FUNCTION__);
    }
}

/* A pairwise exchange whose receive buffer is in the particle storage.  size_limited selects which call it stands
 * in for where no staging is needed: MPI_Sizelimited_Sendrecv (1) or a plain MPI_Sendrecv (0).  Where staging is
 * needed both partners run this same staged loop, so their chunks match. */
int MPI_Sendrecv_into_particle_storage(void *sendbuf0, size_t sendcount, MPI_Datatype sendtype,
                                       int dest, int sendtag, void *recvbuf0, size_t recvcount,
                                       MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                                       MPI_Status *status, int size_limited)
{
    if(!gizmo_particle_storage_needs_staged_mpi_receive()) {
        if(size_limited) {
            return MPI_Sizelimited_Sendrecv(sendbuf0, sendcount, sendtype, dest, sendtag, recvbuf0, recvcount,
                                            recvtype, source, recvtag, comm, status);
        }
        return MPI_Sendrecv(sendbuf0, (int) sendcount, sendtype, dest, sendtag, recvbuf0, (int) recvcount,
                            recvtype, source, recvtag, comm, status);
    }
    if(dest == ThisTask) {   /* a copy within this rank: host memcpy, no MPI write into the storage */
        return MPI_Sizelimited_Sendrecv(sendbuf0, sendcount, sendtype, dest, sendtag, recvbuf0, recvcount,
                                        recvtype, source, recvtag, comm, status);
    }
    if(!ParticleReceiveStaging) {
        gizmo_fatal_hard_exit_reviewed(90002040, "REVIEWED_HARD_MID_PROTOCOL: staged particle-storage receive without its "
                                       "host buffer (particle_receive_staging_init did not run)", __FILE__, __LINE__, __FUNCTION__);
    }
    int size_sendtype, size_recvtype;
    MPI_Type_size(sendtype, &size_sendtype);
    MPI_Type_size(recvtype, &size_recvtype);
    if(size_sendtype != size_recvtype) {   /* both partners chunk by one count, which must mean the same bytes */
        gizmo_fatal_hard_exit_reviewed(90002041, "REVIEWED_HARD_MID_PROTOCOL: staged particle-storage receive with send and "
                                       "receive types of different sizes", __FILE__, __LINE__, __FUNCTION__);
    }
    size_t count_limit = (((long long)All.CommChunkSize) * 1024LL * 1024LL) / size_sendtype;
    if(count_limit > (size_t) INT_MAX) {count_limit = INT_MAX;}
    const size_t staging_limit = PARTICLE_RECEIVE_STAGING_BYTES / (size_t) size_recvtype;
    if(count_limit > staging_limit) {count_limit = staging_limit;}
    if(count_limit < 1) {count_limit = 1;}
    char *sendbuf = (char *) sendbuf0;
    char *recvbuf = (char *) recvbuf0;
    while(sendcount > 0 || recvcount > 0)
    {
        const int send_now = (int) ((sendcount > count_limit) ? count_limit : sendcount);
        const int recv_now = (int) ((recvcount > count_limit) ? count_limit : recvcount);
        MPI_Sendrecv(sendbuf, send_now, sendtype, dest, sendtag,
                     ParticleReceiveStaging, recv_now, recvtype, source, recvtag, comm, status);
        if(recv_now > 0) {memcpy(recvbuf, ParticleReceiveStaging, (size_t) recv_now * (size_t) size_recvtype);}
        sendcount -= send_now;
        recvcount -= recv_now;
        sendbuf += (size_t) send_now * (size_t) size_sendtype;
        recvbuf += (size_t) recv_now * (size_t) size_recvtype;
    }
    return 0;
}

