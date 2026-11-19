/* Driver shared by test_tallbox_gravity_cr.py and test_sgchem_cr_floor.py: startup and one gravity pass, then
   "PROBE id z az pot" for type-3 particles, "HU z az tzz phi dAz/dz -dPhi/dz" from hu_tallbox_field and "EPOT3" (tallbox builds), and the chemistry's
   cosmic-ray ionisation rate, Z/Zsun, G0 and CosmicRayIonRate "ZETA zeta Z G0 rate" of gas cell 0 with its TreeCol gas columns set to argv[2] g/cm^2 (if given). */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "cooling/sgchem/chem_interface.h"
#ifdef GALSF_RESOLVEDISM_TALLBOX
void GravAccel_HuTallbox(void); /* gravity/analytic_gravity.h, compiled into gravtree.o */
#endif

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
    printf("EPOT3 %.17g\n", SysState.EnergyPotComp[3]);
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 3) {printf("PPOT %llu %.17g %.17g %.17g\n", (unsigned long long)P[i].ID, P[i].Pos[2] - boxHalf_Z, P[i].Mass, P[i].Potential);}}
    for(int i = 0; i < NumPart; i++) {if(P[i].Mass > 0) {printf("PALL %.17g %.17g %.17g\n", P[i].Pos[2] - boxHalf_Z, P[i].Mass, P[i].Potential);}}
    energy_budget_gravity_report(); /* BUDGET: "Energy (synced,grav) ... E_pot=" */
#ifdef OUTPUT_ENERGY_STEP_TRACE
    energy_step_trace_report(); /* step trace: SYNC line with E_pot */
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    { /* what one call of the production routine adds to each probe's tidal tensor */
        std::vector<double> t0(NumPart); for(int i = 0; i < NumPart; i++) {t0[i] = P[i].tidal_tensorps[2][2];}
        GravAccel_HuTallbox();
        for(int i = 0; i < NumPart; i++) {if(P[i].Type == 3) {double az_, tzz_, phi_; hu_tallbox_field(P[i].Pos[2] - boxHalf_Z, &az_, &tzz_, &phi_); printf("TZZ %.17g %.17g %.17g\n", P[i].Pos[2] - boxHalf_Z, P[i].tidal_tensorps[2][2] - t0[i], tzz_);}}
    }
#endif
#endif
#if defined(GHOSDT_G0_CR) && defined(HU21_SIGMA_SFR0)
    {double m = HU21_SIGMA_SFR0 * 3.0e7 * (boxSize_X * UNIT_LENGTH_IN_KPC) * (boxSize_Y * UNIT_LENGTH_IN_KPC) / UNIT_MASS_IN_SOLAR; /* Sigma_SFR,0 over 30 Myr */
     double f_save = All.FactorG0; All.FactorG0 = (m / 0.03) / (All.Mgas0 / 4.0); /* as sfr_eff.cc */
     printf("IUV1 %.17g\n", ghosdt_hu21_sigma_sfr_ratio()); All.FactorG0 = f_save;}
#endif
    int c = -1; for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0) {c = i; break;}}
    if(c >= 0) {
        if(argc > 2) {double sig = atof(argv[2]) / (UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv); for(int k = 0; k < RT_USE_TREECOL_FOR_NH; k++) {TREECOL_BINS(c)[TREECOL_GAS][k] = sig;}}
        struct ChemInputs in; memset(&in, 0, sizeof(in));
        in.dt_code = 1e-6; in.u_code = CellP[c].InternalEnergy;
        chem_fill_inputs(c, &in);
        double fe = 0;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        fe = P[c].Metallicity[MET_OF(ELEM_Fe)] / resolvedism_solar_massfrac[ELEM_Fe];
#endif
        printf("ZETA %.17g %.17g %.17g %.17g %.17g\n", in.cosmic_ray_ion_rate, (double)P[c].Metallicity[0] / All.SolarAbundances[0], in.G0, All.CosmicRayIonRate, fe);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
