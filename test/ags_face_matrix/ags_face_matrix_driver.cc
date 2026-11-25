/* Driver for test_ags_face_matrix.py: GIZMO's startup, which ends with the adaptive-softening kernel loop, then
   for each type-1 particle prints the stored face matrix P[].NV_T (9 values) and the neighbour moment matrix
   sum_j W_ij dx dx^T recomputed here by direct summation (xx xy xz yy yz zz), as "NVT id ...". */
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "mesh/kernel.h"

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
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 1) {continue;}
        double h = P[i].AGS_KernelRadius, hinv, hinv3, hinv4, M[3][3] = {{0}};
        kernel_hinv(h, &hinv, &hinv3, &hinv4);
        for(int j = 0; j < NumPart; j++)
        {
            double d[3], wk, dwk;
            for(int k = 0; k < 3; k++) {d[k] = P[i].Pos[k] - P[j].Pos[k];}
            NEAREST_XYZ(d[0], d[1], d[2], 1);
            double r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            if(r >= h) {continue;}
            kernel_main(r * hinv, hinv3, hinv4, &wk, &dwk, 0);
            for(int a = 0; a < 3; a++) {for(int b = 0; b < 3; b++) {M[a][b] += wk * d[a] * d[b];}}
        }
        printf("NVT %llu", (unsigned long long)P[i].ID);
        for(int a = 0; a < 3; a++) {for(int b = 0; b < 3; b++) {printf(" %.17g", (double)P[i].NV_T[a][b]);}}
        printf(" %.17g %.17g %.17g %.17g %.17g %.17g\n", M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
