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

