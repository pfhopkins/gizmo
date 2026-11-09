/* LD_PRELOAD interposer: counts MPI collectives on rank 0, prints the total per call at MPI_Finalize */
#include <mpi.h>
#include <stdio.h>
static long long C[10];
#define WRAP(i, name, decl, args) int name decl {C[i]++; return P##name args;}
WRAP(0, MPI_Allreduce, (const void *a, void *b, int c, MPI_Datatype t, MPI_Op o, MPI_Comm m), (a, b, c, t, o, m))
WRAP(1, MPI_Allgather, (const void *a, int c, MPI_Datatype t, void *b, int d, MPI_Datatype u, MPI_Comm m), (a, c, t, b, d, u, m))
WRAP(2, MPI_Allgatherv, (const void *a, int c, MPI_Datatype t, void *b, const int *d, const int *e, MPI_Datatype u, MPI_Comm m), (a, c, t, b, d, e, u, m))
WRAP(3, MPI_Bcast, (void *a, int c, MPI_Datatype t, int r, MPI_Comm m), (a, c, t, r, m))
WRAP(4, MPI_Reduce, (const void *a, void *b, int c, MPI_Datatype t, MPI_Op o, int r, MPI_Comm m), (a, b, c, t, o, r, m))
WRAP(5, MPI_Gather, (const void *a, int c, MPI_Datatype t, void *b, int d, MPI_Datatype u, int r, MPI_Comm m), (a, c, t, b, d, u, r, m))
WRAP(6, MPI_Alltoall, (const void *a, int c, MPI_Datatype t, void *b, int d, MPI_Datatype u, MPI_Comm m), (a, c, t, b, d, u, m))
WRAP(7, MPI_Barrier, (MPI_Comm m), (m))
WRAP(8, MPI_Comm_create_group, (MPI_Comm m, MPI_Group g, int t, MPI_Comm *n), (m, g, t, n))
WRAP(9, MPI_Comm_split, (MPI_Comm m, int c, int k, MPI_Comm *n), (m, c, k, n))
int MPI_Finalize(void)
{
    int r; long long t = 0; PMPI_Comm_rank(MPI_COMM_WORLD, &r);
    for(int i = 0; i < 10; i++) {t += C[i];}
    if(r == 0) {printf("MPI_COLLECTIVES total %lld allreduce %lld allgather %lld allgatherv %lld bcast %lld reduce %lld create_group %lld\n", t, C[0], C[1], C[2], C[3], C[4], C[8]); fflush(stdout);}
    return PMPI_Finalize();
}
