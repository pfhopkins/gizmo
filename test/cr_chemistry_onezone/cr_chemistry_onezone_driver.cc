/* Driver for test_cr_chemistry_onezone.py: begrun() on a small Stella box builds a real CHEMCOOL cell; do_the_cooling_for_particle
   then evolves copies of it to equilibrium at fixed density and fixed cosmic-ray energy density (reset every step), and prints one
   row per (nH, zeta) point: equilibrium state, the CR heating rate from the chemistry, the analytic rate, and the CR fluid loss rate. */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "cooling/chemcool/chemcool_consts.h"

static FILE *out;
static int c = -1;
static struct particle_data p0;
static struct gas_cell_data c0;

/* code CR energy of cell c for a CR energy density e_cgs [erg cm^-3] */
static double ecr_code(double e_cgs) {return e_cgs * (P[c].Mass / CellP[c].Density) / UNIT_PRESSURE_IN_CGS;}

static void run_point(double nH, double zeta, double t_end_myr, double dt_myr)
{
    P[c] = p0; CellP[c] = c0;
    CellP[c].Density = nH * (1. + 4.*ABHE) * PROTONMASS_CGS / UNIT_DENSITY_IN_CGS; /* yn = rho / ((1+4 ABHE) m_p), chem_interface.cc */
    double T0 = 50., mu = 2.3;
    CellP[c].InternalEnergy = CellP[c].InternalEnergyPred = T0 / (mu * (GAMMA_DEFAULT-1.) * U_TO_TEMP_UNITS);
    double dt = dt_myr * (1e6*SECONDS_PER_YEAR) / UNIT_TIME_IN_CGS;
    P[c].dt_step = (integertime) llround(dt / All.Timebase_interval);
    CellP[c].DtInternalEnergy = 0;
    double e_cgs = (zeta / CR_IONRATE_SOLCIRC) * CR_ENDEN_SOLCIRC * 1.602e-12; /* zeta = 3e-17 e_CR / (1.8 eV cm^-3), as SGChem */
    int nstep = (int) llround(t_end_myr / dt_myr);
    double loss_frac = 0;
    for(int s = 0; s < nstep; s++)
    {
        double e0 = ecr_code(e_cgs);
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {CellP[c].CosmicRayEnergy[k] = CellP[c].CosmicRayEnergyPred[k] = e0 / N_CR_PARTICLE_BINS;}
        do_the_cooling_for_particle(c, P, CellP);
        double e1 = 0; for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {e1 += CellP[c].CosmicRayEnergy[k];}
        loss_frac = 1. - e1 / e0;
    }
    double yn = nH;
    double xh2 = CellP[c].TracAbund[IH2], xhp = CellP[c].TracAbund[IHP], xhi = 1. - 2.*xh2 - xhp;
    double gam_cr = -CellP[c].Lambda[10] * yn * yn; /* chemistry R11 [erg cm^-3 s^-1], heating > 0 */
    fprintf(out, "ROW nH=%.6e zeta_in=%.6e zeta_cell=%.6e T=%.6e xH2=%.6e xHp=%.6e xHI=%.6e ne=%.6e gamCR=%.6e ecr=%.6e lossfrac=%.6e dt_s=%.6e\n",
            nH, zeta, CellP[c].CR_ionization_rate, CellP[c].Temp, xh2, xhp, xhi, CellP[c].Ne, gam_cr, e_cgs, loss_frac, dt_myr * (1e6*SECONDS_PER_YEAR));
    fflush(out);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    if(argc < 2) {fprintf(stderr, "usage: %s params.txt\n", argv[0]); return 1;}
    strcpy(ParameterFile, argv[1]);
    RestartFlag = 0; RestartSnapNum = -1;
    for(int i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] = CPU_Step[i] = 0;}
    CPUThisRun = 0; WallclockTime = my_second();
    begrun();
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0) {c = i; break;}}
    if(c < 0) {fprintf(stderr, "no gas cell\n"); MPI_Abort(MPI_COMM_WORLD, 1);}
    p0 = P[c]; c0 = CellP[c];

    fflush(stdout);
    out = fdopen(dup(fileno(stdout)), "w");
    {FILE *w = tmpfile(); if(!w || dup2(fileno(w), fileno(stdout)) < 0) {fprintf(stderr, "could not divert stdout\n"); return 1;}}

    const double nHs[] = {1e2, 1e3, 1e4}, zetas[] = {3e-17, 3e-16};
    for(double z : zetas) {for(double n : nHs) {run_point(n, z, 30., 0.05);}}
    fprintf(out, "DONE\n");
    fflush(out);
    MPI_Finalize();
    return 0;
}
