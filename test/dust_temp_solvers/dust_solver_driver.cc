/* Driver for test_dust_temp_solvers.py: calls GIZMO's two equilibrium dust-temperature solvers on given
   cell states, and brute-force scans each solver's own residual for an independent reference root set.

   Linked against the objects of a full GIZMO build in src/ (everything except core/main.o), so the code under
   test is exactly the production code (the solvers live in radiation/rt_functions.h; rt_utilities.o carries
   their host symbols):
     A  rt_eqm_dust_temp()  (RT kick side, residual dust_dEdt), absorption rate assembled as a half-step kick
     B  rt_ir_lambdadust()  (implicit cooling side, residual dust_dE_cooling)

   Input, one case per line on stdin ('#' comments ignored):
     id nH T Trad uIR uOPT uNUV uPE uH0 Ne fH2 Zsol mass dt Tdprev LamIRgas [LamNUV]
       nH [cm^-3], temperatures [K], band energy densities u* [eV cm^-3], Ne free electrons per H,
       fH2 molecular mass fraction, Zsol metallicity [solar], mass and dt [code units],
       Tdprev the cell's previous Dust_Temperature (the solvers' initial guess), LamIRgas/LamNUV gas-phase
       cooling already deposited in those bins [erg s^-1 cm^3]
   Output, one line per case on stdout:
     id T TdA nrootA rootAlo rootAhi TdB nrootB rootBlo rootBhi
   where nroot/rootlo/roothi are the number of sign changes of the residual on a log grid in Tdust and the
   lowest/highest bisected root (-1 if none). GIZMO's own stdout chatter is diverted to a temp file.

   Built without -ffast-math so the scan's isfinite() checks work. */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "declarations/allvars.h"
#include "core/proto.h"

#define NGRID 600
static const double TD_LO = 1.0, TD_HI = MAX_DUST_TEMP;

struct Roots {int n; double lo, hi;};

/* build provenance normally recorded by core/main.cc; the snapshot writer references it */
unsigned long long executable_fnv1a64(void) {return 0;}
long long executable_size_in_bytes(void) {return -1;}

/* scan f(Td) on a log grid and bisect every sign change */
template <class F> static Roots scan_roots(F f)
{
    Roots r = {0, -1, -1};
    double x_prev = TD_LO, f_prev = f(x_prev);
    for(int j = 1; j <= NGRID; j++)
    {
        double x = TD_LO * pow(TD_HI/TD_LO, (double)j/NGRID), fx = f(x);
        if(std::isfinite(f_prev) && std::isfinite(fx) && ((f_prev < 0 && fx > 0) || (f_prev > 0 && fx < 0)))
        {
            double a = x_prev, b = x, fa = f_prev;
            for(int it = 0; it < 60; it++) {double m = sqrt(a*b), fm = f(m); if((fm < 0) == (fa < 0)) {a = m; fa = fm;} else {b = m;}}
            double root = sqrt(a*b);
            if(r.n == 0) {r.lo = root;}
            r.hi = root; r.n++;
        }
        x_prev = x; f_prev = fx;
    }
    return r;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    gizmo_node_comm_init();
    gizmo_kokkos_initialize(argc, argv); /* All and the memory pool sit behind the Kokkos runtime */
    if(argc < 2) {fprintf(stderr, "usage: %s params.txt < cases.txt\n", argv[0]); return 1;}

    /* the subset of begrun() the solvers depend on */
    memset(&All, 0, sizeof(All));
    RestartFlag = 0;
    strncpy(ParameterFile, argv[1], sizeof(ParameterFile)-1);
    read_parameter_file(ParameterFile);
    gizmo_size_memory_arena();
    mymalloc_init();
    set_units();
    set_cosmo_factors_for_current_time();
    All.Time = All.TimeBegin;
    gizmo_gpu_sync_all();
    InitCool();
#ifdef RT_CHEM_PHOTOION
    rt_get_sigma();
#endif
    rt_define_effective_frequencies_in_bands();
    All.Timebase_interval = (All.TimeMax - All.TimeBegin) / TIMEBASE;
    /* solar pattern as init.cc sets it for this build (COOL_METAL_LINES_BY_SPECIES, FIRE-3 abundances) */
    {const double sa[11] = {0.0142, 0.27030, 2.53e-3, 7.41e-4, 6.13e-3, 1.34e-3, 7.57e-4, 7.12e-4, 3.31e-4, 6.87e-5, 1.38e-3};
     for(int j = 0; j < NUM_METAL_SPECIES; j++) {All.SolarAbundances[j] = (j < 11) ? sa[j] : 0;}}
    All.Time = 0.5 * All.TimeMax; /* mid-run: rt_ir_lambdadust special-cases All.Time==0 */

    fflush(stdout);
    FILE *out = fdopen(dup(fileno(stdout)), "w");
    {FILE *w = tmpfile(); if(!w || dup2(fileno(w), fileno(stdout)) < 0) {fprintf(stderr, "could not divert stdout\n"); return 1;}}

    struct particle_data pp[1]; struct gas_cell_data cell[1];
    const double eV_to_code = ELECTRONVOLT_IN_ERGS / UNIT_PRESSURE_IN_CGS;
    char line[4096]; long id;
    while(fgets(line, sizeof(line), stdin))
    {
        if(line[0] == '#' || line[0] == '\n') {continue;}
        double nH, T, Trad, uIR, uOPT, uNUV, uPE, uH0, Ne, fH2, Zsol, mass, dt, Tdprev, LamIRgas, LamNUV = 0;
        int nread = sscanf(line, "%ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &id, &nH, &T, &Trad, &uIR, &uOPT, &uNUV, &uPE, &uH0, &Ne, &fH2, &Zsol, &mass, &dt, &Tdprev, &LamIRgas, &LamNUV);
        if(nread < 16) {fprintf(stderr, "bad line: %s", line); return 1;}

        memset(pp, 0, sizeof(pp)); memset(cell, 0, sizeof(cell));
        pp[0].Type = 0; pp[0].Mass = cell[0].Mass = mass;
        cell[0].Density = nH * PROTONMASS_CGS / HYDROGEN_MASSFRAC / UNIT_DENSITY_IN_CGS;
        for(int j = 0; j < NUM_METAL_SPECIES; j++) {pp[0].Metallicity[j] = (j < 11) ? Zsol * All.SolarAbundances[j] : 0;}
        if(NUM_METAL_SPECIES > 1) {pp[0].Metallicity[1] = (1. - HYDROGEN_MASSFRAC) + (All.SolarAbundances[1] - (1. - HYDROGEN_MASSFRAC)) * Zsol;} /* He as init.cc scales it */
        pp[0].NumNgb = 32; pp[0].KernelRadius = 2.*cbrt(mass/cell[0].Density);
        integertime ti = (integertime)(dt / All.Timebase_interval); if(ti < 1) {ti = 1;}
        pp[0].dt_step = ti; {int b = 0; while(((integertime)1 << (b+1)) <= ti && b < TIMEBINS-1) {b++;} pp[0].TimeBin = b;}
        cell[0].Temperature = T; cell[0].Ne = Ne;
        cell[0].HII = DMIN(Ne, 1.); cell[0].HI = 1. - cell[0].HII; cell[0].MolecularMassFraction = fH2;
        {double g = cell[0].gamma_eos_value(); cell[0].InternalEnergy = cell[0].InternalEnergyPred = DMAX(T - 1., 0.) / (0.59 * (g - 1.) * U_TO_TEMP_UNITS);} /* so rt_kappa_adaptive_IR_band's crude Tgas estimate equals T */
        double vol = mass / cell[0].Density;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {cell[0].Rad_E_gamma[k] = MIN_REAL_NUMBER; cell[0].Lambda_RadiativeCooling_toRHDBins[k] = 0; cell[0].Rad_Je[k] = 0; cell[0].Dt_Rad_E_gamma[k] = 0;}
        cell[0].Rad_E_gamma[RT_FREQ_BIN_INFRARED] += uIR * eV_to_code * vol;
        cell[0].Rad_E_gamma[RT_FREQ_BIN_OPTICAL_NIR] += uOPT * eV_to_code * vol;
        cell[0].Rad_E_gamma[RT_FREQ_BIN_NUV] += uNUV * eV_to_code * vol;
        cell[0].Rad_E_gamma[RT_FREQ_BIN_PHOTOELECTRIC] += uPE * eV_to_code * vol;
        cell[0].Rad_E_gamma[RT_FREQ_BIN_H0] += uH0 * eV_to_code * vol;
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {cell[0].Rad_E_gamma_Pred[k] = cell[0].Rad_E_gamma[k];}
        cell[0].Radiation_Temperature = Trad; cell[0].Dust_Temperature = Tdprev;
        cell[0].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] = LamIRgas;
        cell[0].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_NUV] = LamNUV;

        /* A: absorption rate as rt_update_driftkick assembles it in a half-step kick (dt/2; the IR band also sees
           the energy the other bands donated to it earlier in the band loop) */
        double E_abs_tot_toIR = 0, dt_entr = 0.5 * get_particle_timestep_in_physical(0, pp), e_donated = 0;
        for(int k = 0; k < N_RT_FREQ_BINS; k++)
        {
            if(k == RT_FREQ_BIN_INFRARED || rt_get_donation_target_bin(k) != RT_FREQ_BIN_INFRARED) {continue;}
            double a0 = rt_absorption_rate(0, k, pp, cell); if(a0*dt_entr > 50.) {a0 = 50./dt_entr;}
            double de_abs = cell[0].Rad_E_gamma[k] * (1. - exp(-a0*dt_entr));
            E_abs_tot_toIR += de_abs / (MIN_REAL_NUMBER + dt_entr); e_donated += de_abs;
        }
        double a0_IR = rt_absorption_rate(0, RT_FREQ_BIN_INFRARED, pp, cell);
        double total_absorption_rate = E_abs_tot_toIR + fabs(a0_IR) * (cell[0].Rad_E_gamma[RT_FREQ_BIN_INFRARED] + e_donated);
        double absrate = total_absorption_rate * (cell[0].Density * All.cf_a3inv / cell[0].Mass) * C_LIGHT_CODE / C_LIGHT_CODE_REDUCED;
        double fdustmet_init = return_dust_to_metals_ratio_vs_solar(0, DMIN(DMAX(Tdprev, 1.), MAX_DUST_TEMP), pp, cell);
        Roots rA = scan_roots([&](double Td) {return dust_dEdt(0, T, Td, absrate, fdustmet_init, pp, cell);});
        double TdA = rt_eqm_dust_temp(0, T, absrate, pp, cell);

        /* B: the residual is scanned with the pre-solve cell state, which is what the solver starts from */
        cell[0].Dust_Temperature = Tdprev;
        Roots rB = scan_roots([&](double Td) {double d1, d2; return dust_dE_cooling(0, T, Td, &d1, &d2, pp, cell);});
        cell[0].Dust_Temperature = Tdprev;
        rt_ir_lambdadust(0, T, pp, cell);
        double TdB = cell[0].Dust_Temperature;

        fprintf(out, "%ld %.17g %.17g %d %.17g %.17g %.17g %d %.17g %.17g\n", id, T, TdA, rA.n, rA.lo, rA.hi, TdB, rB.n, rB.lo, rB.hi);
        fflush(out);
    }
    gizmo_kokkos_finalize();
    MPI_Finalize();
    return 0;
}
