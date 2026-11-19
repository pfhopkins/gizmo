/* Driver for test_sgchem_unified_fractions.py: begrun() on the small Stella box, one cooling call per gas cell, then for each cell
   "EL Ne abe_rhs x_ion x_HII ne_thermal" (stored electrons, net-5 RHS electrons H+ + (C - CO) + Si, Get_Gas_Ionized_Fraction, H+,
   ThermalProperties' electrons) and "XH HYDROGEN_MASSFRAC". */
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "cooling/cooling.h"
#include "cooling/sgchem/chem_interface.h"
#include "cooling/sgchem/sgchem_consts.h"

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
    printf("XH %.17g\n", (double)HYDROGEN_MASSFRAC);
    double dt = 1.0e3 * SECONDS_PER_YEAR / UNIT_TIME_IN_CGS; /* 1 kyr */
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 0) continue;
        P[i].dt_step = (integertime) llround(dt / All.Timebase_interval);
        do_the_cooling_for_particle(i, P, CellP);
        struct ChemInputs in; memset(&in, 0, sizeof(in)); in.dt_code = dt; in.u_code = CellP[i].InternalEnergy;
        chem_fill_inputs(i, &in);
        double abe_rhs = CellP[i].TracAbund[IHP] + (in.abundc - CellP[i].TracAbund[ICO]) + in.abundsi;
        double mu = 1, ne = 1, nh0 = 0, nhp = 0, nHe0 = 0, nHeII = 0, nHepp = 0;
        ThermalProperties(CellP[i].InternalEnergy, CellP[i].Density * All.cf_a3inv, i, &mu, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, P, CellP);
        printf("EL %.17g %.17g %.17g %.17g %.17g\n", (double)CellP[i].Ne, abe_rhs, Get_Gas_Ionized_Fraction(i, P, CellP), (double)CellP[i].TracAbund[IHP], ne);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
