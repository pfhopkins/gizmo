/* Global per-type group offsets: local scan plus an exclusive scan of the per-rank totals. */
#include "hbt_catalogue_mpi.h"

void hbt_group_type_offsets_global(const int *lentype, long ngroups, long long *offsettype, long long *tot_global, MPI_Comm comm)
{
    long long loc[HBT_NTYPES], base[HBT_NTYPES];
    int rank; MPI_Comm_rank(comm, &rank);
    hbt_group_type_totals(lentype, ngroups, loc);
    MPI_Exscan(loc, base, HBT_NTYPES, MPI_LONG_LONG, MPI_SUM, comm);
    if(rank == 0) {for(int t = 0; t < HBT_NTYPES; t++) {base[t] = 0;}} /* MPI_Exscan leaves rank 0 undefined */
    hbt_group_type_offsets(lentype, ngroups, base, offsettype);
    MPI_Allreduce(loc, tot_global, HBT_NTYPES, MPI_LONG_LONG, MPI_SUM, comm);
}
