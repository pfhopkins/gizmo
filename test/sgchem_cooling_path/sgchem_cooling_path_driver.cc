/* Driver for test_sgchem_cooling_path.py: begrun() on a small Stella IC builds a real SGCHEM cell; do_the_cooling_for_particle
   then runs on copies of it in pairs of calls that differ in one input, and prints PASS|FAIL for the steps the standard cooling
   path owns: sink thermal energy, unsplit hydro heating, the electron abundance. */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "cooling/sgchem/sgchem_consts.h"

static FILE *out;
static int n_fail = 0, n_check = 0;
static int c = -1;
static struct particle_data p0;
static struct gas_cell_data c0;

static void check(const char *name, bool ok, double got, double want)
{
    n_check++; if(!ok) {n_fail++;}
    fprintf(out, "%s %s got=%.17g want=%.17g\n", ok ? "PASS" : "FAIL", name, got, want);
}

/* the template cell at nH=1, T=8000 K, stepped once */
static double cool_once(double injected, double heating_code, int unsplit, double dt_code)
{
    P[c] = p0; CellP[c] = c0;
    double X_H = 1. - p0.Metallicity[0] - p0.Metallicity[2];
    CellP[c].Density = 1.0 * PROTONMASS_CGS / X_H / UNIT_DENSITY_IN_CGS;
    double mu = 4. / (1. + 3.*X_H);
    CellP[c].InternalEnergy = CellP[c].InternalEnergyPred = 8000. / (mu * (GAMMA_DEFAULT-1.) * U_TO_TEMP_UNITS);
    CellP[c].Temp = 8000.; CellP[c].Ne = -1; /* sentinel: the cooling path must set it */
    P[c].dt_step = (integertime) llround(dt_code / All.Timebase_interval);
    CellP[c].Injected_Sink_Energy = injected;
    CellP[c].DtInternalEnergy = heating_code;
    CellP[c].CoolingIsOperatorSplitThisTimestep = unsplit ? 0 : 1;
    do_the_cooling_for_particle(c, P, CellP);
    return CellP[c].InternalEnergy;
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

    double dt = 1.0e3 * SECONDS_PER_YEAR / UNIT_TIME_IN_CGS; /* 1 kyr */
    double u_ref = cool_once(0, 0, 0, dt);
    check("reference:finite", std::isfinite(u_ref) && u_ref > 0, u_ref, 0);

    /* electron abundance: the chemistry's electrons are returned as Ne (C, Si singly ionized, plus He+, He++) */
    double ne_min = CellP[c].TracAbund[IHP];
    check("ne:set_from_chemistry", CellP[c].Ne >= ne_min && CellP[c].Ne < 2.0, CellP[c].Ne, ne_min);

    /* sink thermal energy: added after the solve, exactly E/m, then consumed */
    double E = 0.3 * u_ref * P[c].Mass;
    double u_sink = cool_once(E, 0, 0, dt);
    check("sink_energy:deposited_exactly", fabs((u_sink - u_ref) - E / P[c].Mass) <= 1e-12 * u_sink, u_sink - u_ref, E / P[c].Mass);
    check("sink_energy:consumed", CellP[c].Injected_Sink_Energy == 0, CellP[c].Injected_Sink_Energy, 0);

    /* unsplit hydro heating: the kick skips it on such steps, so the cooling path must apply it */
    double heat = 0.3 * u_ref / dt; /* code specific energy per code time */
    double u_unsplit0 = cool_once(0, 0, 1, dt), u_unsplit = cool_once(0, heat, 1, dt);
    check("unsplit_heating:applied", (u_unsplit - u_unsplit0) > 0.5 * heat * dt, u_unsplit - u_unsplit0, heat * dt);
    check("unsplit_heating:consumed", CellP[c].DtInternalEnergy == 0, CellP[c].DtInternalEnergy, 0);

    fprintf(out, "DONE checks=%d failed=%d\n", n_check, n_fail);
    fflush(out);
    MPI_Finalize();
    return n_fail;
}
