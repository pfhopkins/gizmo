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


/** Function to calculate the number of unique -nodes- (shared memory machine structures), not MPI tasks, on which we are running.
    often this is specified by global variables by the runtime commands or job scheduler but that is machine-dependent, so we want a
    way to check internally for some memory allocation purposes. */
int getNodeCount(void)
{
    int rank, is_rank0, nodes;
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    gizmo_mpi_set_failfast_errhandler(shmcomm);  /* fail-fast, not MPI's default wedge-prone abort */
    MPI_Comm_rank(shmcomm, &rank);
    is_rank0 = (rank == 0) ? 1 : 0;
    MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_free(&shmcomm);
    return nodes;
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
    
    size_t count_limit = (((long long)All.BufferSize)*1024LL * 1024LL) / size_sendtype;
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

