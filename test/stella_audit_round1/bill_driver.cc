/* Driver for test_photoion_bill.py (STL-409): the photoionisation recombination bill of one ionised cell as the walk computes it
   (resolvedism_photoion_recomb_bill), for a few densities and masses, with the build's hydrogen fraction. No startup needed. */
#include <mpi.h>
#include <cstdio>
#include "declarations/allvars.h"
#include "core/proto.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const double alpha = 2.6e-13, rho[3] = {1.0e-24, 1.7e-22, 3.0e-20}, M[3] = {1.989e33, 3.0e32, 2.0e34};
    for(int a = 0; a < 3; a++) {for(int b = 0; b < 3; b++) {
        printf("BILL alpha=%.17g rho=%.17g M=%.17g X=%.17g mp=%.17g bill=%.17g\n", alpha, rho[a], M[b], (double)HYDROGEN_MASSFRAC, (double)PROTONMASS_CGS,
               resolvedism_photoion_recomb_bill(alpha, rho[a], M[b]));}}
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
