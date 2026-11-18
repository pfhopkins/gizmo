/* Driver for test_stella_fb_conservation.py::test_death_ejecta_table: GIZMO's startup (loads the stellar tables), then the
   death ejecta the feedback deposits (stellar_death_ejecta) on a dense (M, Z) scan off and on the table grid.
   M_death = M_init - lifetime winds, as the wind channel leaves the star at death. One line per sample:
     EJ logM logZ rem_type M_death rem sum_m min_m
   then a summary line. */
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
    strcpy(ParameterFile, argv[1]);
    RestartFlag = 0; RestartSnapNum = -1;
    for(int i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] = CPU_Step[i] = 0;}
    CPUThisRun = 0; WallclockTime = my_second();
    begrun();
    if(ThisTask == 0) {
        const int NMS = 1500; /* log-spaced 0.9..300 Msun: AGB/WD, ECSN, CCSN, failed, PPISN, PISN, direct collapse */
        const double lM0 = log10(0.9), lM1 = log10(300.);
        for(int iz = 0; iz < 2 * STBL_NZ - 1; iz++) { /* grid values and midpoints */
            double logZ = (iz % 2 == 0) ? StellarTbl.log_Z[iz / 2] : 0.5 * (StellarTbl.log_Z[iz / 2] + StellarTbl.log_Z[iz / 2 + 1]);
            for(int im = 0; im <= NMS; im++) {
                double logM = lM0 + (lM1 - lM0) * im / NMS, M = pow(10., logM);
                double log_end = log10(DMAX(stellar_lifetime(logM, logZ), 100.0)), w = 0;
#ifdef GALSF_RESOLVEDISM_WINDS
                for(int k = 0; k < STBL_NELEM; k++) {w += stellar_elem_ej_wind_cumulative(logM, logZ, log_end, k);}
#endif
                double M_death = M - w, m[STBL_NELEM], sum = 0, mn = 1e300;
                double rem = stellar_death_ejecta(logM, logZ, M_death, m);
                for(int k = 0; k < STBL_NELEM; k++) {sum += m[k]; mn = DMIN(mn, m[k]);}
                printf("EJ %.10f %.10f %d %.17g %.17g %.17g %.17g\n", logM, logZ, stellar_remnant_type(logM, logZ), M_death, rem, sum, mn);
            }
        }
        /* grid stars: the remnant fraction is the table's own */
        for(int iz = 0; iz < STBL_NZ; iz++) {
            for(int im = 0; im < STBL_NM; im++) {
                double logM = StellarTbl.log_M[im], logZ = StellarTbl.log_Z[iz], m[STBL_NELEM], ej = 0;
                int rt = StellarTbl.remnant_type[iz * STBL_NM + im];
                const double *tab = (rt == REM_WD) ? StellarTbl.elem_ej_AGB_mass : StellarTbl.elem_ej_SN_mass;
                for(int k = 0; k < STBL_NELEM; k++) {ej += tab[(iz * STBL_NM + im) * STBL_NELEM + k];}
                double rt_mass = (rt == REM_PISN) ? 0 : StellarTbl.remnant_mass[iz * STBL_NM + im];
                double rem = stellar_death_ejecta(logM, logZ, 1.0, m); /* per unit mass at death */
                printf("GRID %.10f %.10f %d %.17g %.17g\n", logM, logZ, rt, rem, (rt_mass + ej > 0) ? rt_mass / (rt_mass + ej) : 1.0);
            }
        }
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
