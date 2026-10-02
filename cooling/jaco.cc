/*
 * jaco.cc -- GIZMO glue for the jaco code-generated microphysics network.
 *
 * Packs a gas cell into the generated SolveVars/Params (gizmo_to_jaco), calls the implicit solver
 * (jaco_solver.cc, no GIZMO dependencies), and writes the answer back (jaco_to_gizmo). Also
 * builds the CIE table that seeds the ions, and reports per-step solver statistics.
 */

#include "../core/proto.h"
#include "../declarations/allvars.h"
#include "microphysics_func_jac.h"
#include <math.h>
#include <string.h>
#include "jaco_solver.h"
/* NaN/Inf check — defined in jaco_util.cc to prevent LTO from optimizing it away */
extern "C" int jaco_isfinite(double x);
#define JACO_ABUNDANCE_FLOOR 1e-20

#ifdef JACO

/* ---- CIE lookup table for initial ion abundance guesses ---- */

#define CIE_TABLE_N 200
#define CIE_TABLE_LOG_TMIN 2.5 /* 316 K */
#define CIE_TABLE_LOG_TMAX 6.5 /* 3.16e6 K */
static double cie_log_T[CIE_TABLE_N];
static double cie_xHp[CIE_TABLE_N];
static double cie_xHep[CIE_TABLE_N];
static double cie_xHepp[CIE_TABLE_N];
static int cie_table_initialized = 0;

/* Interpolate from the CIE table in log T */
static double cie_interp(const double *table, double logT) {
    double f = (logT - CIE_TABLE_LOG_TMIN) / (CIE_TABLE_LOG_TMAX - CIE_TABLE_LOG_TMIN) * (CIE_TABLE_N - 1);
    if (f <= 0)
        return table[0];
    if (f >= CIE_TABLE_N - 1)
        return table[CIE_TABLE_N - 1];
    int i = (int)f;
    double t = f - i;
    return (1 - t) * table[i] + t * table[i + 1];
}

/* Build the CIE table by sweeping T downward with continuation, solving the chemistry at each
   fixed T with the solver's fixed-T chemistry solve. Called once from InitCool(). */
void jaco_build_cie_table(void) {
    if (cie_table_initialized)
        return;

    /* Default params for CIE (low density, no radiation) */
    struct JacoSolverSettings set;
    jaco_solver_default_settings(&set);
    Params pr = {};
    pr.n_Htot = 1.0;
    pr.Delta_t = 1e15;
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL) || defined(JACO_MODEL_KWH)
    pr.y = 0.0994;
#endif
#if defined(JACO_MODEL_STARFORGE)
    pr.ISRF = 1.0;
    pr.N_H = 1e20;
    pr.G_0 = 1.0;
    pr.Td = 15.0;
    pr.Z_d = 1.0;
    pr.f_d = 1.0;
    pr.grad_v = 1e-14;
    pr.Delta_x = 3e18;
    pr.x_C_tot = 2.1e-4;
    pr.x_N = 6.8e-5;
    pr.x_Ne = 8.5e-5;
    pr.x_Mg = 3.2e-5;
    pr.x_Si = 3.2e-5;
    pr.x_S = 1.3e-5;
    pr.x_Ca = 2.2e-6;
    pr.x_Fe = 2.5e-5;
    pr.x_O_tot = 4.9e-4;
#elif defined(JACO_MODEL_KWH)
    pr.Delta_x = 3e18;
    pr.grad_v = 1e-14;
#elif defined(JACO_MODEL_PRIMORDIAL)
    pr.Td = 15.0;
    pr.Z_d = 1.0;
    pr.f_d = 1.0;
    pr.Delta_x = 3e18;
    pr.grad_v = 1e-14;
#endif

#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL) || defined(JACO_MODEL_KWH)
    /* Sweep with continuation, from HIGH T to LOW T. The (0.9, 0.01, 0.01)
       guess is in the basin of attraction of the physical solution at high T;
       sweeping downward, the previous (mostly-ionized) solution stays in the
       physical basin as T decreases. Sweeping upward from low T tends to fall
       into the trivial all-zero fixed point. */
    double xHp = 0.99, xHep = 1e-4, xHepp = 0.099;
    for (int i = CIE_TABLE_N - 1; i >= 0; i--) {
        double logT = CIE_TABLE_LOG_TMIN + (CIE_TABLE_LOG_TMAX - CIE_TABLE_LOG_TMIN) * i / (CIE_TABLE_N - 1);
        double T = pow(10.0, logT);
        cie_log_T[i] = logT;

        SolveVars sv = {};
        sv.T = T;
        sv.x_Hplus = xHp;
        sv.x_Heplus = xHep;
        sv.x_Heplusplus = xHepp;
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
        sv.x_H_2 = 1e-20;
#endif
        sv.u = jaco_T_to_u(T, &sv, &pr, NULL);
        pr.u_initial = sv.u;
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
        pr.x_H_2_initial = sv.x_H_2;
#endif

        int nfeval = 0;
        if (jaco_solve_chemistry(&sv, &pr, &set, &nfeval)) {
            printf("jaco_build_cie_table: chemistry failed at logT=%g (T=%g)\n", logT, T);
            jaco_print_state(stdout, "  state:", &sv, &pr);
            endrun(11);
        }

        cie_xHp[i] = sv.x_Hplus;
        cie_xHep[i] = sv.x_Heplus;
        cie_xHepp[i] = sv.x_Heplusplus;
        /* Continuation */
        xHp = sv.x_Hplus;
        xHep = sv.x_Heplus;
        xHepp = sv.x_Heplusplus;
    }
    if (ThisTask == 0) {
        printf("JACO: built CIE lookup table (%d points, logT %.1f–%.1f)\n", CIE_TABLE_N, CIE_TABLE_LOG_TMIN,
               CIE_TABLE_LOG_TMAX);
        FILE *fd = fopen("jaco_cie_table.dat", "w");
        if (fd) {
            fprintf(fd, "# logT x_Hplus x_Heplus x_Heplusplus\n");
            for (int i = 0; i < CIE_TABLE_N; i++)
                fprintf(fd, "%.6e %.6e %.6e %.6e\n", cie_log_T[i], cie_xHp[i], cie_xHep[i], cie_xHepp[i]);
            fclose(fd);
            printf("JACO: CIE table written to jaco_cie_table.dat\n");
        }
    }
#endif /* JACO_MODEL_STARFORGE || JACO_MODEL_PRIMORDIAL || JACO_MODEL_KWH */
    cie_table_initialized = 1;
}

/* ---- GIZMO interface layer ---- */

void gizmo_to_jaco(int i, SolveVars *sv, Params *pr, struct particle_data *pp, struct gas_cell_data *cell) {
    double dtime = get_particle_timestep_in_physical(i, pp);
    double Delta_t = dtime * UNIT_TIME_IN_CGS;
    set_PdV_work_heatingrate(i, dtime, pp, cell);
    double n_Htot = cell[i].nHcgs();

    /* --- Variables common to all models --- */
    sv->u = cell[i].InternalEnergy * UNIT_SPECEGY_IN_CGS;
    pr->u_initial = sv->u;
    pr->n_Htot = n_Htot;
    pr->Delta_t = Delta_t;
    pr->pdv_work = (cell[i].CoolingIsOperatorSplitThisTimestep == 0) ? cell[i].DtInternalEnergy * n_Htot : 0;

    /* --- Model-specific parameter packing --- */
#ifdef JACO_MODEL_WIND_COMPARISON
    pr->x_H = 1.;
#elif defined(JACO_MODEL_KWH)
    /* Primordial H/He cooling model. Minimal params: y, C_2, plus T and ion solve vars. */
    {
        double X_H = HYDROGEN_MASSFRAC;
#ifdef METALS
        X_H = 1.0 - pp[i].Metallicity[0];
        if (NUM_METAL_SPECIES >= 10)
            X_H -= pp[i].Metallicity[1];
#endif
        double Y_He = (1.0 - X_H) * 0.25;
        pr->y = Y_He / X_H;
    }
    /* Cell geometry for C_2 clumping factor (derived_param in the model). */
    {
        double dx_code = pp[i].Get_Particle_Size() * All.cf_atime;
        double grad_v = cell[i].velocity_gradient_norm();
        pr->Delta_x = dx_code * UNIT_LENGTH_IN_CGS;
        pr->grad_v = DMAX(1e-30, grad_v * UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS);
    }
    sv->T = cell[i].Temperature;
    {
        double logT = log10(DMAX(sv->T, 10.));
        sv->x_Hplus = cie_interp(cie_xHp, logT);
        sv->x_Heplus = cie_interp(cie_xHep, logT);
        sv->x_Heplusplus = cie_interp(cie_xHepp, logT);
    }
#elif defined(JACO_MODEL_PRIMORDIAL)
    /* Primordial H/He + H2 chemistry model. KWH base plus H2 chemistry params. */
    {
        double X_H = HYDROGEN_MASSFRAC;
#ifdef METALS
        X_H = 1.0 - pp[i].Metallicity[0];
        if (NUM_METAL_SPECIES >= 10)
            X_H -= pp[i].Metallicity[1];
#endif
        double Y_He = (1.0 - X_H) * 0.25;
        pr->y = Y_He / X_H;
    }
    /* C_2, C_3 are derived_params computed from T, grad_v, Delta_x in the model. */
    /* Dust and geometry — minimal defaults (no radiation field) */
    pr->Td = 15.0;
    pr->Z_d = 1.0;
    pr->f_d = 1.0;
    {
        double dx_code = pp[i].Get_Particle_Size() * All.cf_atime;
        double grad_v = cell[i].velocity_gradient_norm();
        pr->Delta_x = dx_code * UNIT_LENGTH_IN_CGS;
        pr->grad_v = DMAX(1e-30, grad_v * UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS);
    }
    sv->T = cell[i].Temperature;
    {
        double logT = log10(DMAX(sv->T, 10.));
        sv->x_Hplus = cie_interp(cie_xHp, logT);
        sv->x_Heplus = cie_interp(cie_xHep, logT);
        sv->x_Heplusplus = cie_interp(cie_xHepp, logT);
    }
    {
        double fmol = DMIN(DMAX(cell[i].MolecularMassFraction, 0), 1.0);
        sv->x_H_2 = DMAX(JACO_ABUNDANCE_FLOOR, 0.5 * fmol);
        pr->x_H_2_initial = sv->x_H_2;
    }
#elif defined(JACO_MODEL_STARFORGE)
    /* Hydrogen mass fraction and helium abundance by number */
    double X_H = HYDROGEN_MASSFRAC;
#ifdef METALS
    X_H = 1.0 - pp[i].Metallicity[0]; /* X = 1 - Z */
    if (NUM_METAL_SPECIES >= 10) {
        X_H -= pp[i].Metallicity[1]; /* X = 1 - Y - Z */
    }
#endif
    double Y_He = (1.0 - X_H) * 0.25; /* He number fraction per H = (1-X)/(4X) but stored as y = n_He/n_H */
    pr->y = Y_He / X_H;

    /* Metal abundances (per H nucleus) from metallicity array */
    double Z_solar = All.SolarAbundances[0];
    double Z_met = 0;
#ifdef METALS
    Z_met = pp[i].Metallicity[0];
    if (NUM_METAL_SPECIES >= 10) {
        /* Metallicity indices: [0]=Z, [1]=He, [2]=C, [3]=N, [4]=O, [5]=Ne, [6]=Mg, [7]=Si, [8]=S, [9]=Ca, [10]=Fe
           Convert mass fractions to number abundances per H: x_s = (X_s / m_s) / (X_H / m_H) = (X_s / m_s_amu) * (1 /
           X_H) */
        double inv_XH = 1.0 / X_H;
        pr->x_C_tot = pp[i].Metallicity[2] / 12.0 * inv_XH;
        pr->x_N = pp[i].Metallicity[3] / 14.0 * inv_XH;
        pr->x_O_tot = pp[i].Metallicity[4] / 16.0 * inv_XH;
        pr->x_Ne = pp[i].Metallicity[5] / 20.0 * inv_XH;
        pr->x_Mg = pp[i].Metallicity[6] / 24.0 * inv_XH;
        pr->x_Si = pp[i].Metallicity[7] / 28.0 * inv_XH;
        pr->x_S = pp[i].Metallicity[8] / 32.0 * inv_XH;
        pr->x_Ca = pp[i].Metallicity[9] / 40.0 * inv_XH;
        pr->x_Fe = pp[i].Metallicity[10] / 56.0 * inv_XH;
    } else {
        /* Scale all elemental abundances from total metallicity assuming solar ratios */
        double Zfac = (Z_solar > 0) ? Z_met / Z_solar : 0;
        pr->x_C_tot = 2.1e-4 * Zfac;
        pr->x_N = 6.8e-5 * Zfac;
        pr->x_O_tot = 4.9e-4 * Zfac;
        pr->x_Ne = 8.5e-5 * Zfac;
        pr->x_Mg = 3.2e-5 * Zfac;
        pr->x_Si = 3.2e-5 * Zfac;
        pr->x_S = 1.3e-5 * Zfac;
        pr->x_Ca = 2.2e-6 * Zfac;
        pr->x_Fe = 2.5e-5 * Zfac;
    }
#else
    pr->x_C_tot = pr->x_N = pr->x_O_tot = pr->x_Ne = pr->x_Mg = pr->x_Si = pr->x_S = pr->x_Ca = pr->x_Fe = 0;
#endif

    /* Initial T guess: use stored temperature from previous step. */
    sv->T = cell[i].Temperature;

    /* Ion abundances: interpolate from pre-computed CIE table for a physically
       correct initial guess at any T. This avoids cold-start issues where
       stored Ne is zero or stale after shock heating. */
    double logT = log10(DMAX(sv->T, 10.));
    sv->x_Hplus = cie_interp(cie_xHp, logT);
    sv->x_Heplus = cie_interp(cie_xHep, logT);
    sv->x_Heplusplus = cie_interp(cie_xHepp, logT);

    /* H2: use stored MolecularMassFraction from previous step */
    double fmol = DMIN(DMAX(cell[i].MolecularMassFraction, 0), 1.0);
    sv->x_H_2 = DMAX(JACO_ABUNDANCE_FLOOR, fmol);

    pr->x_H_2_initial = sv->x_H_2;

    /* H-, C+, CO are now fully inlined by jaco codegen as fixed_species with
       symbolic expressions. No params to set here. */

    /* Dust: solar-normalized dust abundance and sublimation factor */
    double Zd_solar = (Z_solar > 0) ? Z_met / Z_solar : 1.0;
    pr->Z_d = DMAX(1e-4, Zd_solar);
    pr->f_d = 1.0; /* no sublimation correction for now */

    /* Dust temperature: use stored value if available, otherwise compute equilibrium estimate */
#ifdef RT_INFRARED
    pr->Td = cell[i].Dust_Temperature;
#else
    double shieldfac_Td = return_uvb_shieldfac(i, 0, n_Htot, log10(DMAX(sv->T, 10.)), cell);
    pr->Td = 10; // get_equilibrium_dust_temperature_estimate(i, shieldfac_Td, sv->T, pp, cell);
#endif

    /* Radiation field and cosmic rays */
    pr->G_0 = 1.0; /* Habing units; will be overridden below if RT available */
    pr->ISRF = 1.0;
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
    double shieldfac = return_uvb_shieldfac(i, 0, n_Htot, log10(DMAX(sv->T, 10.)), cell);
    pr->G_0 = get_FUV_G0(i, shieldfac, 0, pp, cell);
#endif

    /* Column density, cell size, and velocity gradient */
    double dx_code = pp[i].Get_Particle_Size() * All.cf_atime; /* physical cell size in code units */
    double grad_v = cell[i].velocity_gradient_norm();          /* velocity gradient Frobenius norm in code units */
    pr->Delta_x = dx_code * UNIT_LENGTH_IN_CGS;
    pr->N_H = evaluate_NH_from_GradRho(pp[i].GradRho, pp[i].KernelRadius, cell[i].Density, pp[i].NumNgb, 1, i, pp) *
              UNIT_SURFDEN_IN_CGS / PROTONMASS_CGS;
    pr->grad_v =
        DMAX(1e-30, grad_v * UNIT_VEL_IN_CGS /
                        UNIT_LENGTH_IN_CGS); /* CGS s^-1, floored to avoid division by zero in LVG expressions */

    /* Cosmological redshift for inverse Compton cooling */
    pr->z = All.ComovingIntegrationOn ? (1.0 / All.Time - 1.0) : 0;

#endif /* JACO_MODEL_STARFORGE */
}

void jaco_to_gizmo(int i, const SolveVars *sv, const Params *pr, struct particle_data *pp, struct gas_cell_data *cell) {
    cell[i].InternalEnergy = sv->u / UNIT_SPECEGY_IN_CGS;
    cell[i].InternalEnergyPred = cell[i].InternalEnergy;
    cell[i].Temperature = sv->T;

#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
    /* Write solved species back BEFORE set_eos_pressure, which needs Ne and MolecularMassFraction for gamma */
    double xHp = sv->x_Hplus, xH2 = sv->x_H_2;
    double xHep = sv->x_Heplus, xHepp = sv->x_Heplusplus;
    cell[i].Ne = xHp + xHep + 2.0 * xHepp;
    double xH0 = DMAX(1.0 - xHp - 2.0 * xH2, 0);
    cell[i].MolecularMassFraction = 2.0 * xH2;
    cell[i].MolecularMassFraction_perNeutralH = (xH0 > 0) ? cell[i].MolecularMassFraction / xH0 : 0;
#elif defined(JACO_MODEL_KWH)
    cell[i].Ne = sv->x_Hplus + sv->x_Heplus + 2.0 * sv->x_Heplusplus;
#endif

    set_eos_pressure(i, pp, cell); /* use standard GIZMO EOS for pressure/sound speed */
#ifndef COOLING_OPERATOR_SPLIT
    if (cell[i].CoolingIsOperatorSplitThisTimestep == 0) {
        cell[i].DtInternalEnergy = 0;
    }
#endif

    /* Sanity check GIZMO-side variables */
    if (!jaco_isfinite(cell[i].InternalEnergy) || !jaco_isfinite(cell[i].Pressure) ||
        !jaco_isfinite(cell[i].Temperature) || cell[i].InternalEnergy <= 0 || cell[i].Pressure <= 0 ||
        cell[i].Temperature <= 0) {
        printf("JACO FATAL: bad value after jaco_to_gizmo\n");
        printf("  InternalEnergy=%.6e Pressure=%.6e Temperature=%.6e\n", (double)cell[i].InternalEnergy,
               (double)cell[i].Pressure, (double)cell[i].Temperature);
        printf("  Ne=%.6e MolecularMassFraction=%.6e Density=%.6e\n", (double)cell[i].Ne,
               (double)cell[i].MolecularMassFraction, (double)cell[i].Density);
        printf("  SolveVars:");
        for (int k = 0; k < N_VARS; k++)
            printf(" [%d]=%.6e", k, sv->data[k]);
        printf("\n  Params:");
        for (int k = 0; k < N_PARAMS; k++)
            printf(" [%d]=%.6e", k, pr->data[k]);
        printf("\n");
        endrun(778);
    }
}

/* ---- Per-step solver statistics ----
   Accumulated over the cells call_jaco() solves in one cooling pass; reported and reset by
   jaco_report_solve_stats(), which every rank must call (MPI collective) once per pass. */
enum { JS_CELLS, JS_TIER1, JS_TIER2, JS_TIER3, JS_FAILED, JS_FEVALS, JS_RESYNC, JS_PINNED, JS_NANJ, JS_NANF,
       JS_T1_1, JS_T1_2, JS_T1_3, JS_T1_4, JS_T1_MORE, JS_N };
static long jaco_stats[JS_N];
static int jaco_stats_max_nfeval = 0;

static void jaco_stats_add(const struct JacoSolveInfo *info) {
    int bin[4] = {JS_TIER1, JS_TIER2, JS_TIER3, JS_FAILED};
    int t1 = info->nfeval - info->nfeval_fd; /* evaluations Newton asked for, not finite-difference columns */
#ifdef _OPENMP
#pragma omp critical(jaco_stats_update)
#endif
    {
        jaco_stats[JS_CELLS]++;
        jaco_stats[bin[info->tier - 1]]++;
        jaco_stats[JS_FEVALS] += info->nfeval;
        jaco_stats[JS_RESYNC] += info->resynced_T0;
        jaco_stats[JS_PINNED] += info->pinned;
        jaco_stats[JS_NANJ] += info->n_nonfinite_jac > 0;
        jaco_stats[JS_NANF] += info->n_nonfinite_F > 0;
        if (info->tier == JACO_TIER_NEWTON) jaco_stats[t1 <= 4 ? JS_T1_1 + t1 - 1 : JS_T1_MORE]++;
        if (info->nfeval > jaco_stats_max_nfeval) jaco_stats_max_nfeval = info->nfeval;
    }
}

/* Called at the end of each cooling pass on every rank: prints the MPI-reduced statistics on
   rank 0 and resets the counters. */
void jaco_report_solve_stats(void) {
    long global[JS_N];
    int global_max = 0;
    MPI_Reduce(jaco_stats, global, JS_N, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&jaco_stats_max_nfeval, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ThisTask == 0 && global[JS_CELLS] > 0) {
        double n = (double)global[JS_CELLS];
        printf("jaco solve stats: %ld cells | tier1 %ld (%.2f%%) tier2 %ld tier3 %ld FAILED %ld | nfeval mean %.2f max %d | "
               "tier-1 evals 1:%ld 2:%ld 3:%ld 4:%ld >4:%ld | T0 resynced %ld, pinned %ld, non-finite J %ld F %ld\n",
               global[JS_CELLS], global[JS_TIER1], 100.0 * global[JS_TIER1] / n, global[JS_TIER2], global[JS_TIER3],
               global[JS_FAILED], global[JS_FEVALS] / n, global_max, global[JS_T1_1], global[JS_T1_2], global[JS_T1_3],
               global[JS_T1_4], global[JS_T1_MORE], global[JS_RESYNC], global[JS_PINNED], global[JS_NANJ], global[JS_NANF]);
        fflush(stdout);
    }
    memset(jaco_stats, 0, sizeof(jaco_stats));
    jaco_stats_max_nfeval = 0;
}

void call_jaco(struct particle_data *p, struct gas_cell_data *c) {
    double dtime = get_particle_timestep_in_physical(0, p);
    if (dtime == 0)
        return;
    SolveVars sv = {};
    Params pr = {};
    gizmo_to_jaco(0, &sv, &pr, p, c);

    struct JacoSolverSettings set;
    jaco_solver_default_settings(&set);
    set.T_min = (All.MinGasTemp > 0) ? All.MinGasTemp : 1.0;
    set.u_min = All.MinEgySpec * UNIT_SPECEGY_IN_CGS;
    struct JacoSolveInfo info;
    const SolveVars sv_in = sv;
    if (jaco_solve(&sv, &pr, &set, &info)) {
        printf("JACO FATAL: every solver tier failed on task %d (nfeval=%d)\n", ThisTask, info.nfeval);
        jaco_print_state(stdout, "  input state:", &sv_in, &pr);
        set.verbose = 2; /* replay with a trace of every iterate */
        sv = sv_in;
        jaco_solve(&sv, &pr, &set, &info);
        fflush(stdout);
        endrun(10);
    }
    jaco_to_gizmo(0, &sv, &pr, p, c);
    jaco_stats_add(&info);
}
#endif
