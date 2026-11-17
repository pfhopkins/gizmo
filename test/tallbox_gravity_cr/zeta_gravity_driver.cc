/* Driver shared by test_tallbox_gravity_cr.py and test_sgchem_cr_floor.py: startup and one gravity pass, then
   "PROBE id z az pot" for type-3 particles, "HU z az tzz phi dAz/dz -dPhi/dz" from hu_tallbox_field and "EPOT3" (tallbox builds), and the chemistry's
   cosmic-ray ionisation rate "ZETA zeta Z" of gas cell 0 with its TreeCol gas columns set to argv[2] g/cm^2 (if given). */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "cooling/sgchem/chem_interface.h"

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
    printf("G %.17g\n", All.G);
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 3) {printf("PROBE %llu %.17g %.17g %.17g\n", (unsigned long long)P[i].ID, P[i].Pos[2] - boxHalf_Z, P[i].GravAccel[2], P[i].Potential);}}
#if defined(GALSF_RESOLVEDISM_TALLBOX) && defined(HU_TALLBOX_FIELD_DEFINED)
    for(int k = -40; k <= 40; k++) {
        double z = 0.25 * k, d = 1e-4, az, tzz, phi, azp, azm, phip, phim, t_;
        hu_tallbox_field(z, &az, &tzz, &phi); hu_tallbox_field(z + d, &azp, &t_, &phip); hu_tallbox_field(z - d, &azm, &t_, &phim);
        printf("HU %.17g %.17g %.17g %.17g %.17g %.17g\n", z, az, tzz, phi, (azp - azm) / (2 * d), -(phip - phim) / (2 * d));
    }
    compute_global_quantities_of_system();
    double epot_probes = 0; for(int i = 0; i < NumPart; i++) {if(P[i].Type == 3) {epot_probes += 0.5 * P[i].Mass * P[i].Potential + P[i].Mass * analytic_external_potential(i);}}
    printf("EPOT3 %.17g %.17g\n", SysState.EnergyPotComp[3], epot_probes);
#endif
    int c = -1; for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0) {c = i; break;}}
    if(c >= 0) {
        if(argc > 2) {double sig = atof(argv[2]) / (UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv); for(int k = 0; k < RT_USE_TREECOL_FOR_NH; k++) {TREECOL_BINS(c)[TREECOL_GAS][k] = sig;}}
        struct ChemInputs in; memset(&in, 0, sizeof(in));
        in.dt_code = 1e-6; in.u_code = CellP[c].InternalEnergy;
        chem_fill_inputs(c, &in);
        printf("ZETA %.17g %.17g\n", in.cosmic_ray_ion_rate, (double)P[c].Metallicity[0] / All.SolarAbundances[0]);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
