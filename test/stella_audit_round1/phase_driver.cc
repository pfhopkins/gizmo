/* Driver for test_tables_phase_matched.py (STL-391/392/411): GIZMO's startup (loads the stellar tables), then for one star
   (argv[2] = M [Msun], argv[3] = Z) the code's own lookups over its life:
     PHASEQ  lifetime integral of the ionising photon rate, midpoint rule on N phase samples of the interpolated lifetime
     PHASEVW v_wind at the death age (the death flush reads it there)
     PHASEWC cumulative wind mass at the death age (what the wind closure has delivered at death) */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "galaxy_sf/resolvedism_stellar_tables.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    if(argc < 4) {fprintf(stderr, "usage: %s params M Z\n", argv[0]); return 1;}
    strcpy(ParameterFile, argv[1]);
    RestartFlag = 0; RestartSnapNum = -1;
    for(int i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] = CPU_Step[i] = 0;}
    CPUThisRun = 0; WallclockTime = my_second();
    begrun();
    if(ThisTask == 0) {
        double M = atof(argv[2]), Z = atof(argv[3]), logM = log10(M), logZ = log10(Z);
        double lt = stellar_lifetime(logM, logZ);
        const int N = 20000;
        double q = 0;
        for(int k = 0; k < N; k++) {double age = (k + 0.5) / N * lt; q += pow(10., stellar_log_Q_ion(logM, logZ, log10(DMAX(age, 100.0))));}
        q *= lt / N;
        double la_end = log10(DMAX(lt, 100.0)), w = 0;
        for(int k = 0; k < STBL_NELEM; k++) {w += stellar_elem_ej_wind_cumulative(logM, logZ, la_end, k);}
        printf("PHASEQ M=%.17g Z=%.17g lt=%.17g N=%d Qint=%.17g\n", M, Z, lt, N, q);
        printf("PHASEVW v=%.17g\n", stellar_v_wind(logM, logZ, la_end));
        printf("PHASEWC w=%.17g\n", w);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
