/* mpi_alltoallv_typed.h — overflow-safe wrapper around MPI_Alltoallv.
 *
 * Background. Classic MPI_Alltoallv takes int* count/displ arrays. With
 * MPI_BYTE that means any per-peer payload above INT_MAX (2.1 GB) overflows
 * the count argument and the call aborts with MPI_ERR_COUNT. fire_m11i at
 * 12.4M particles on 2 ranks (~6M particles per rank, sizeof(struct
 * particle_data) ~ 400 B → ~2.4 GB per peer in ghost_exchange) trips this.
 *
 * Fix. Build a contiguous MPI datatype matching the per-element struct, then
 * pass element counts/displs (still int*, but in element units rather than
 * bytes). This lifts the int-overflow ceiling from "2.1 GB per peer" to
 * "2.1 G *elements* per peer" — sizeof(struct)*INT_MAX bytes, effectively
 * unbounded for any realistic problem.
 *
 * MPI 3 portable. Same wire layout, same allocation pattern at the caller —
 * the contiguous datatype is byte-equivalent to MPI_BYTE × elem_size, just
 * with a different counting unit. */

#ifndef MPI_ALLTOALLV_TYPED_H
#define MPI_ALLTOALLV_TYPED_H

#include <mpi.h>
#include <stddef.h>

/* sendbuf / recvbuf       — same buffers as MPI_Alltoallv(MPI_BYTE)
 * sendcount[t], recvcount[t]   — element counts (NOT bytes) for peer t
 * senddisp[t], recvdisp[t]     — element displacements (NOT byte offsets)
 * elem_size                    — sizeof of the struct/element being shipped
 * comm                         — MPI communicator
 *
 * Caller must size the send/recv buffers as elem_size * sum(counts) bytes.
 * Caller's count/disp arrays are pre-existing element-unit arrays — the
 * "*sizeof(struct)" multiplications used to populate parallel byte arrays
 * become unnecessary at the call site. */
static inline void gizmo_mpi_alltoallv_typed(const void *sendbuf,
                                             const int  *sendcount,
                                             const int  *senddisp,
                                             void       *recvbuf,
                                             const int  *recvcount,
                                             const int  *recvdisp,
                                             size_t      elem_size,
                                             MPI_Comm    comm)
{
    MPI_Datatype t;
    MPI_Type_contiguous((int)elem_size, MPI_BYTE, &t);
    MPI_Type_commit(&t);
    MPI_Alltoallv((void *)sendbuf, (int *)sendcount, (int *)senddisp, t,
                  recvbuf,         (int *)recvcount, (int *)recvdisp, t,
                  comm);
    MPI_Type_free(&t);
}

#endif /* MPI_ALLTOALLV_TYPED_H */
