/* Driver for test_treecol_shielding_bounded.py: GIZMO's startup up to the first gravity pass (as treecol_nongas), then prints
   the TreeCol gas-column pixels of the cells with ID 1 and the last ID as "COL id pixel value" and the kernel radius as "H id h". */
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
    domain_Decomposition(0, 0, 0);
    set_non_standard_physics_for_current_time();
    compute_grav_accelerations();
    MyIDType idmax = 0; for(int i = 0; i < NumPart; i++) {if(P[i].ID > idmax) {idmax = P[i].ID;}}
    for(int i = 0; i < NumPart; i++)
        if(P[i].Type == 0 && (P[i].ID == 1 || P[i].ID == idmax)) {
            printf("H %llu %.17g\n", (unsigned long long)P[i].ID, (double)P[i].KernelRadius);
            for(int k = 0; k < RT_USE_TREECOL_FOR_NH; k++) {printf("COL %llu %d %.17g\n", (unsigned long long)P[i].ID, k, (double)TREECOL_BINS(i)[TREECOL_GAS][k]);}
        }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
