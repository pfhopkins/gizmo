/* Driver for test_sgchem_cooling_path.py: begrun() on a small Stella IC builds a real SGCHEM cell; do_the_cooling_for_particle
   then runs on copies of it in pairs of calls that differ in one input, and prints PASS|FAIL for the steps the standard cooling
   path owns: the solver's energy handed back, the electron abundance, sink thermal energy (SINK_THERMALFEEDBACK) and hydro
   heating (applied and consumed when unsplit; left to the kick under COOLING_OPERATOR_SPLIT). */
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

/* the template cell at nH=1, T=8000 K */
static void set_template(double injected, double heating_code, int unsplit, double dt_code)
{
    (void)injected; (void)unsplit; /* unused without sinks / under the split */
    P[c] = p0; CellP[c] = c0;
    double X_H = 1. - p0.Metallicity[0] - p0.Metallicity[2];
    CellP[c].Density = 1.0 * PROTONMASS_CGS / X_H / UNIT_DENSITY_IN_CGS;
    double mu = 4. / (1. + 3.*X_H);
    CellP[c].InternalEnergy = CellP[c].InternalEnergyPred = 8000. / (mu * (GAMMA_DEFAULT-1.) * U_TO_TEMP_UNITS);
    CellP[c].Temp = 8000.; CellP[c].Ne = -1; /* sentinel: the cooling path must set it */
    P[c].dt_step = (integertime) llround(dt_code / All.Timebase_interval);
#ifdef SINK_THERMALFEEDBACK
    CellP[c].Injected_Sink_Energy = injected;
#endif
    CellP[c].DtInternalEnergy = heating_code;
#ifndef COOLING_OPERATOR_SPLIT
    CellP[c].CoolingIsOperatorSplitThisTimestep = unsplit ? 0 : 1;
#endif
}

/* the template cell stepped once through the cooling path */
static double cool_once(double injected, double heating_code, int unsplit, double dt_code)
{
    set_template(injected, heating_code, unsplit, dt_code);
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

    /* hand-back: the cooling path stores exactly what the SGCHEM solve returns for the same cell and step */
    double u_pred = CellP[c].InternalEnergyPred;
    set_template(0, 0, 0, dt);
    double u_in = CellP[c].InternalEnergy;
    double u_solver = DMAX(sgchem_cool_cell(c, u_in, get_particle_timestep_in_physical(c)), All.MinEgySpec);
    check("handback:energy_is_solver_result", u_ref == u_solver, u_ref, u_solver);
    check("handback:pred_is_energy", u_pred == u_ref, u_pred, u_ref);
    check("handback:solver_changed_energy", u_solver != u_in, u_solver, u_in);

#ifdef SINK_THERMALFEEDBACK
    /* sink thermal energy: added after the solve, exactly E/m, then consumed */
    double E = 0.3 * u_ref * P[c].Mass;
    double u_sink = cool_once(E, 0, 0, dt);
    check("sink_energy:deposited_exactly", fabs((u_sink - u_ref) - E / P[c].Mass) <= 1e-12 * u_sink, u_sink - u_ref, E / P[c].Mass);
    check("sink_energy:consumed", CellP[c].Injected_Sink_Energy == 0, CellP[c].Injected_Sink_Energy, 0);

#endif
    double heat = 0.3 * u_ref / dt; /* code specific energy per code time */
#ifndef COOLING_OPERATOR_SPLIT
    /* unsplit hydro heating: the kick skips it on such steps, so the cooling path must apply it */
    double u_unsplit0 = cool_once(0, 0, 1, dt), u_unsplit = cool_once(0, heat, 1, dt);
    check("unsplit_heating:applied", (u_unsplit - u_unsplit0) > 0.5 * heat * dt, u_unsplit - u_unsplit0, heat * dt);
    check("unsplit_heating:consumed", CellP[c].DtInternalEnergy == 0, CellP[c].DtInternalEnergy, 0);
#else
    /* split: the kick applies the hydro heating, so the cooling path neither applies nor consumes it */
    double u_split0 = cool_once(0, 0, 0, dt), u_split = cool_once(0, heat, 0, dt);
    check("split_heating:not_applied", u_split == u_split0, u_split - u_split0, 0);
    check("split_heating:kept", CellP[c].DtInternalEnergy == heat, CellP[c].DtInternalEnergy, heat);
#endif

    fprintf(out, "DONE checks=%d failed=%d\n", n_check, n_fail);
    fflush(out);
    MPI_Finalize();
    return n_fail;
}
