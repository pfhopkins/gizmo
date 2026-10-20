/* Driver for test_treecol_nongas.py: GIZMO's startup up to the first gravity pass, then prints each gas cell's
   TreeCol column-density bins as "COL id bin value". */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "declarations/allvars.h"
#include "core/proto.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    strcpy(ParameterFile, argv[1]);
    RestartFlag = 0; RestartSnapNum = -1;
    for(int i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] = CPU_Step[i] = 0;}
    CPUThisRun = 0; WallclockTime = my_second();
    begrun();
    domain_Decomposition(0, 0, 0); /* as run() does before the first force computation */
    set_non_standard_physics_for_current_time();
    compute_grav_accelerations();
    for(int i = 0; i < NumPart; i++)
        if(P[i].Type == 0) {for(int k = 0; k < RT_USE_TREECOL_FOR_NH; k++) {printf("COL %llu %d %.17g\n", (unsigned long long)P[i].ID, k, (double)P[i].ColumnDensityBins[k]);}}
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
