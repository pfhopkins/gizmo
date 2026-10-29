/* Driver for test_kport3_io_probes.py: asks the SnapFormat 1/2 record check whether one file with the given per-type
   counts fits the formats' 32-bit block markers, around the boundary the positions block sets. Prints FIT lines, then DONE. */
#include <mpi.h>
#include <cstdio>
#include <climits>
#include "declarations/allvars.h"
#include "core/proto.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    for(int t = 0; t < 6; t++) {All.MassTable[t] = 0;}
    const long long pos_bytes = get_bytes_per_blockelement(IO_POS, 0);
    for(int fmt = 1; fmt <= 2; fmt++)
    {
        All.SnapFormat = fmt;
        const long long marker = (fmt == 2) ? 2 * (long long) sizeof(int) : 0;
        const long long nmax = (INT_MAX - marker) / pos_bytes;   /* most gas cells whose positions record fits */
        const long long cases[][6] = {{1000, 0, 0, 0, 0, 0}, {nmax, 0, 0, 0, 0, 0}, {nmax + 1, 0, 0, 0, 0, 0},
                                      {nmax / 2 + 1, nmax / 2 + 1, 0, 0, 0, 0}, {3000000000LL, 0, 0, 0, 0, 0}};
        const char *names[] = {"small", "at_limit", "above_limit", "two_types_above", "above_int32"};
        for(int k = 0; k < 5; k++)
            {printf("FIT fmt=%d case=%s n0=%lld n1=%lld fits=%d\n", fmt, names[k], cases[k][0], cases[k][1], binary_snapshot_file_fits(cases[k], 0));}
    }
    printf("DONE pos_bytes=%lld\n", pos_bytes);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
