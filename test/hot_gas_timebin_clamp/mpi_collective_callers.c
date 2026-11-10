/* LD_PRELOAD interposer: MPI collectives counted on rank 0 by return address, printed at MPI_Finalize (resolve with addr2line) */
#include <mpi.h>
#include <stdio.h>
#define NMAX 4096
static void *A[NMAX]; static long long N[NMAX]; static int nA;
static void note(void *ra) {for(int i = 0; i < nA; i++) {if(A[i] == ra) {N[i]++; return;}} if(nA < NMAX) {A[nA] = ra; N[nA++] = 1;}}
#define WRAP(name, decl, args) int name decl {note(__builtin_return_address(0)); return P##name args;}
WRAP(MPI_Allreduce, (const void *a, void *b, int c, MPI_Datatype t, MPI_Op o, MPI_Comm m), (a, b, c, t, o, m))
WRAP(MPI_Allgather, (const void *a, int c, MPI_Datatype t, void *b, int d, MPI_Datatype u, MPI_Comm m), (a, c, t, b, d, u, m))
WRAP(MPI_Allgatherv, (const void *a, int c, MPI_Datatype t, void *b, const int *d, const int *e, MPI_Datatype u, MPI_Comm m), (a, c, t, b, d, e, u, m))
WRAP(MPI_Bcast, (void *a, int c, MPI_Datatype t, int r, MPI_Comm m), (a, c, t, r, m))
WRAP(MPI_Reduce, (const void *a, void *b, int c, MPI_Datatype t, MPI_Op o, int r, MPI_Comm m), (a, b, c, t, o, r, m))
WRAP(MPI_Gather, (const void *a, int c, MPI_Datatype t, void *b, int d, MPI_Datatype u, int r, MPI_Comm m), (a, c, t, b, d, u, r, m))
WRAP(MPI_Gatherv, (const void *a, int c, MPI_Datatype t, void *b, const int *d, const int *e, MPI_Datatype u, int r, MPI_Comm m), (a, c, t, b, d, e, u, r, m))
WRAP(MPI_Alltoall, (const void *a, int c, MPI_Datatype t, void *b, int d, MPI_Datatype u, MPI_Comm m), (a, c, t, b, d, u, m))
WRAP(MPI_Alltoallv, (const void *a, const int *c, const int *e, MPI_Datatype t, void *b, const int *d, const int *f, MPI_Datatype u, MPI_Comm m), (a, c, e, t, b, d, f, u, m))
WRAP(MPI_Barrier, (MPI_Comm m), (m))
int MPI_Finalize(void)
{
    int r; PMPI_Comm_rank(MPI_COMM_WORLD, &r);
    if(r == 0) {for(int i = 0; i < nA; i++) {printf("MPI_CALLER %p %lld\n", A[i], N[i]);} fflush(stdout);}
    return PMPI_Finalize();
}
