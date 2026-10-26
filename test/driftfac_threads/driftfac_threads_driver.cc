/* Driver for test_driftfac_threads.py: calls the comoving drift and gravity-kick factors from every OpenMP thread at
   once and compares each result with the serial value for the same interval. The drift routines call these
   concurrently (move_particles, and the neighbour-search lazy drift under two different critical sections), so any
   state shared between calls must not leak one interval's value into another's. Prints PASS|FAIL lines, then DONE. */
#include <mpi.h>
#include <cstdio>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "declarations/allvars.h"
#include "core/proto.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    All.ComovingIntegrationOn = 1;
    All.TimeBegin = 0.01; All.TimeMax = 1.0;
    All.OmegaMatter = 0.3; All.OmegaLambda = 0.7; All.OmegaRadiation = 0; All.Hubble_H0_CodeUnits = 0.1;
    All.Timebase_interval = (log(All.TimeMax) - log(All.TimeBegin)) / TIMEBASE;
    init_drift_table();

    /* a few distinct intervals, so successive calls keep replacing whatever the previous call computed */
    const int K = 4;
    integertime t0[K], t1[K];
    double ref_drift[K], ref_kick[K];
    for(int k = 0; k < K; k++) {t0[k] = (TIMEBASE / 8) * k; t1[k] = t0[k] + TIMEBASE / 16 + 12345 * (k + 1);}
    for(int k = 0; k < K; k++) {ref_drift[k] = get_drift_factor(t0[k], t1[k], -1, 0); ref_kick[k] = get_gravkick_factor(t0[k], t1[k], -1, 0);}

    const long long n_calls = 20000000;
    long long bad_drift = 0, bad_kick = 0;
    int n_threads = 1;
#pragma omp parallel reduction(+:bad_drift,bad_kick)
    {
#ifdef _OPENMP
#pragma omp single
        n_threads = omp_get_num_threads();
#endif
#pragma omp for schedule(static)
        for(long long m = 0; m < n_calls; m++)
        {
            int k = (int)((m + m / 3) % K);
            if(get_drift_factor(t0[k], t1[k], -1, 0) != ref_drift[k]) {bad_drift++;}
            if(get_gravkick_factor(t0[k], t1[k], -1, 0) != ref_kick[k]) {bad_kick++;}
        }
    }
    printf("threads=%d calls=%lld\n", n_threads, n_calls);
    printf("%s threads_ran %d\n", n_threads > 1 ? "PASS" : "FAIL", n_threads);
    printf("%s get_drift_factor mismatches=%lld\n", bad_drift ? "FAIL" : "PASS", bad_drift);
    printf("%s get_gravkick_factor mismatches=%lld\n", bad_kick ? "FAIL" : "PASS", bad_kick);
    printf("DONE\n");
    MPI_Finalize();
    return 0;
}
