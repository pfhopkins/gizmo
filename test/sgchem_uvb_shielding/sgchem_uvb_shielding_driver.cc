/* Driver for test_sgchem_uvb_shielding.py: begrun() on a small comoving box at z = 6.5 (FG20 TREECOOL) builds a real net-17 cell;
   do_the_cooling_for_particle then evolves copies of it with prescribed TreeCol gas columns and prints one row per point:
   FAC   (shielded build only) band attenuation factors for given HI, HeI columns;
   THIN  zero columns (must match the build without shielding bit for bit);
   SPH   a cell at radius r in a uniform sphere, columns along the 12 HEALPix directions (radial axis = pixel axis). */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "cooling/sgchem/sgchem_consts.h"
#include "cooling/sgchem/chem_interface.h"

static FILE *out;
static int c = -1;
static struct particle_data p0;
static struct gas_cell_data c0;

static void evolve(double nH, const double colH[NPIX], double t_end_myr, double dt_myr, const char *tag, double r_over_R, double Nmin)
{
    P[c] = p0; CellP[c] = c0;
    CellP[c].Density = nH * (1. + 4.*ABHE) * PROTONMASS_CGS / UNIT_DENSITY_IN_CGS / All.cf_a3inv; /* yn = rho / ((1+4 ABHE) m_p), chem_interface.cc */
    double T0 = 100., mu = 1.22;
    CellP[c].InternalEnergy = CellP[c].InternalEnergyPred = T0 / (mu * (GAMMA_DEFAULT-1.) * U_TO_TEMP_UNITS);
    CellP[c].DtInternalEnergy = 0;
    CellP[c].TracAbund[IH2] = 1e-6; CellP[c].TracAbund[IHP] = 1e-4; CellP[c].TracAbund[IHEP] = 0; CellP[c].TracAbund[IHEPP] = 0;
    for(int s = 0; s < TREECOL_NSPECIES; s++) {for(int p = 0; p < NPIX; p++) {TREECOL_BINS(c)[s][p] = 0;}}
    for(int p = 0; p < NPIX; p++) {TREECOL_BINS(c)[TREECOL_GAS][p] = colH[p] * (1. + 4.*ABHE) * PROTONMASS_CGS / (UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv);}
    double dt = dt_myr * (1e6*SECONDS_PER_YEAR) / UNIT_TIME_IN_CGS;
    P[c].dt_step = (integertime) llround(dt / unit_integertime_in_physical(c, P));
    int nstep = (int) llround(t_end_myr / dt_myr);
    for(int s = 0; s < nstep; s++) {do_the_cooling_for_particle(c, P, CellP);}
    fprintf(out, "%s nH=%.6e r=%.6e Nmin=%.6e T=%.17g xHp=%.17g xHep=%.17g xHepp=%.17g xH2=%.17g\n", tag, nH, r_over_R, Nmin,
            (double)CellP[c].Temp, (double)CellP[c].TracAbund[IHP], (double)CellP[c].TracAbund[IHEP], (double)CellP[c].TracAbund[IHEPP], (double)CellP[c].TracAbund[IH2]);
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
    double g[6]; get_uvb_rates(&g[0], &g[1], &g[2], &g[3], &g[4], &g[5]);
    fprintf(out, "UVB z=%.6f gHI=%.6e gHeI=%.6e gHeII=%.6e eHI=%.6e eHeI=%.6e eHeII=%.6e\n", 1./All.Time - 1., g[0], g[1], g[2], g[3], g[4], g[5]);

#ifdef SGCHEM_UVB_TREECOL_SHIELDING
    for(int ihe = 0; ihe < 2; ihe++) {
        for(int k = -1; k <= 28; k++) {
            double NHI = (k < 0) ? 0 : pow(10., 15. + 0.25 * k), N[3] = {NHI, ihe * ABHE * NHI, 0}, fg[3], fq[3];
            chem_uvb_shield_factors(N, fg, fq);
            fprintf(out, "FAC NHI=%.6e NHeI=%.6e fgHI=%.17g fgHeI=%.17g fgHeII=%.17g fqHI=%.17g fqHeI=%.17g fqHeII=%.17g\n", N[0], N[1], fg[0], fg[1], fg[2], fq[0], fq[1], fq[2]);
        }
    }
#endif

    const double nHs[] = {0.1, 1., 10., 100.};
    double col[NPIX];
    for(double n : nHs) {for(int p = 0; p < NPIX; p++) {col[p] = 0;} evolve(n, col, 30., 0.05, "THIN", 0, 0);}

    /* sphere with centre column n R = 1e20 cm^-2; HEALPix NSIDE=1 pixel centres have cos(angle to the radial axis) = 2/3, 0, -2/3 (4 each) */
    const double mus[3] = {2./3., 0., -2./3.};
    for(double n : nHs) {
        double R = 1e20 / n;
        for(int k = 0; k <= 17; k++) {
            double depth = (k < 17) ? pow(10., 16. + 0.25 * k) / n : R; /* R - r, depth column 1e16 ... 1e20 */
            double r = R - depth, Nmin = 1e300;
            for(int p = 0; p < NPIX; p++) {
                double mu = mus[p / 4], l = -r * mu + sqrt(R*R - r*r * (1. - mu*mu));
                col[p] = n * l; Nmin = fmin(Nmin, col[p]);
            }
            evolve(n, col, 30., 0.05, "SPH", r / R, Nmin);
        }
    }
    fprintf(out, "DONE\n");
    fflush(out);
    MPI_Finalize();
    return 0;
}
