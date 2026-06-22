/*
 * jaco.cc — Implicit backward-Euler microphysics solver for GIZMO.
 *
 * This file is a static template distributed with the jaco codegen package.
 * It provides the Newton-Raphson solver loop and the GIZMO integration layer.
 * The physics (RHS + Jacobian) is supplied by the codegen-produced
 * microphysics_func_jac(), which is called as a black box.
 *
 * The solver uses SolveVars and Params unions (defined in microphysics_func_jac.h)
 * that provide both named field access (sv->T, pr->n_Htot) and indexed array
 * access (sv->data[i]) via anonymous structs. This eliminates index-mismatch
 * bugs while allowing generic loops over variables.
 *
 * Recovery strategy on convergence failure:
 *   1. Full Newton steps (careful_steps=1). Fast when it works.
 *   2. Damped Newton with 30-step ramp (careful_steps=30). Prevents overshoot.
 *   3. Bisection restart (careful_steps=31). Resets to geometric mean of
 *      temperature bounds, then damped Newton again.
 *   4. If all attempts exhaust MAXITER, accept the state if the residual is
 *      small (relaxed tolerance), otherwise abort.
 */

#include "../core/proto.h"
#include "../declarations/allvars.h"
#include "microphysics_func_jac.h"
#include <math.h>
#include <string.h>
extern "C" {
#include <gsl/gsl_linalg.h>
}
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

/* Build the CIE table by sweeping T with continuation from the CIE test solver.
   Called once from jaco_init_tables(). Uses the compiled microphysics_func_jac
   with T fixed (identity rows for u/T) and the same Newton solver as the unit test. */
void jaco_build_cie_table(void) {
    if (cie_table_initialized)
        return;

    /* Default params for CIE (low density, no radiation) */
    Params pr = {};
    pr.n_Htot = 1.0;
    pr.Delta_t = 1e15;
    pr.y = 0.0994;
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

        /* Newton solve at fixed T: zero the u/T equations */
        SolveVars func, dsv;
        double jac[N_VARS][N_VARS];
        const int CIE_MAXITER = 100;
        int converged = 0;
        for (int iter = 0; iter < CIE_MAXITER; iter++) {
            microphysics_func_jac(&sv, &pr, &func, jac);
            func.data[0] = 0;
            func.data[1] = 0;
            for (int j = 0; j < N_VARS; j++) {
                jac[0][j] = 0;
                jac[1][j] = 0;
            }
            jac[0][0] = 1;
            jac[1][1] = 1;

            double jf[N_VARS * N_VARS], rhs[N_VARS];
            for (int ii = 0; ii < N_VARS; ii++) {
                rhs[ii] = -func.data[ii];
                for (int jj = 0; jj < N_VARS; jj++)
                    jf[ii * N_VARS + jj] = jac[ii][jj];
            }
            gsl_matrix_view A = gsl_matrix_view_array(jf, N_VARS, N_VARS);
            gsl_vector_view b = gsl_vector_view_array(rhs, N_VARS);
            gsl_vector_view x = gsl_vector_view_array(dsv.data, N_VARS);
            gsl_vector *tau = gsl_vector_alloc(N_VARS);
            gsl_linalg_QR_decomp(&A.matrix, tau);
            gsl_linalg_QR_solve(&A.matrix, tau, &b.vector, &x.vector);
            gsl_vector_free(tau);

            double fac = fmin(1.0, (double)(iter + 1) / 10.0);
            converged = 1;
            for (int k = 0; k < N_VARS; k++) {
                sv.data[k] += fac * dsv.data[k];
                if (fabs(dsv.data[k]) > 1e-6 * (fabs(sv.data[k]) + 1e-6))
                    converged = 0;
            }
            sv.T = fmax(10.0, fmin(1e10, sv.T));
            for (int k = 2; k < N_VARS; k++)
                sv.data[k] = fmax(JACO_ABUNDANCE_FLOOR, fmin(1.0, sv.data[k]));
            sv.u = jaco_T_to_u(sv.T, &sv, &pr, NULL);
            if (converged)
                break;
        }
        if (!converged) {
            printf("jaco_build_cie_table: failed to converge at logT=%g (T=%g) after %d iterations\n", logT, T,
                   CIE_MAXITER);
            printf("  sv:");
            for (int k = 0; k < N_VARS; k++)
                printf(" %.4e", sv.data[k]);
            printf("\n");
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

    cie_table_initialized = 1;
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

/* ---- Per-timestep solver statistics ----
   Accumulated across all particles processed by call_jaco() during one cooling pass,
   reset and reported at the start of each call to jaco_report_solve_stats(). */
struct JacoSolveStats {
    long n_solves;        /* number of call_jaco() invocations with nfeval>0 */
    long n_fevals;        /* total function evaluations */
    long n_expensive;     /* solves with nfeval > 10 */
    int  max_nfeval;      /* worst-case nfeval from a single particle */
};
static struct JacoSolveStats jaco_stats = {0, 0, 0, 0};

/* Called once at the end of cooling_parent_routine() to print per-timestep solver
   statistics (MPI-reduced) and reset the counters. */
void jaco_report_solve_stats(void) {
    long local[3] = {jaco_stats.n_solves, jaco_stats.n_fevals, jaco_stats.n_expensive};
    long global[3] = {0, 0, 0};
    int local_max = jaco_stats.max_nfeval, global_max = 0;
    MPI_Reduce(local, global, 3, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ThisTask == 0 && global[0] > 0) {
        double mean = (double)global[1] / (double)global[0];
        printf("jaco solve stats: %ld particles, %ld fevals (mean=%.2f, max=%d), %ld expensive (>10)\n",
               global[0], global[1], mean, global_max, global[2]);
        fflush(stdout);
    }
    jaco_stats.n_solves = 0;
    jaco_stats.n_fevals = 0;
    jaco_stats.n_expensive = 0;
    jaco_stats.max_nfeval = 0;
}

void call_jaco(struct particle_data *p, struct gas_cell_data *c) {
    double dtime = get_particle_timestep_in_physical(0, p);
    if (dtime == 0)
        return;
    SolveVars sv = {};
    Params pr = {};
    gizmo_to_jaco(0, &sv, &pr, p, c);
    int nfeval = jaco_solve(&sv, &pr);
    jaco_to_gizmo(0, &sv, &pr, p, c);

#ifdef _OPENMP
#pragma omp atomic update
    jaco_stats.n_solves++;
#pragma omp atomic update
    jaco_stats.n_fevals += nfeval;
    if (nfeval > 10) {
#pragma omp atomic update
        jaco_stats.n_expensive++;
    }
#pragma omp critical
    { if (nfeval > jaco_stats.max_nfeval) jaco_stats.max_nfeval = nfeval; }
#else
    jaco_stats.n_solves++;
    jaco_stats.n_fevals += nfeval;
    if (nfeval > 10) jaco_stats.n_expensive++;
    if (nfeval > jaco_stats.max_nfeval) jaco_stats.max_nfeval = nfeval;
#endif
}

/* ---- Newton-Raphson solver ---- */

/* Apply physical bounds. Returns 1 if any variable was clamped, 0 otherwise. */
static int jaco_clamp(SolveVars *sv) {
    int clamped = 0;
    double T_old = sv->T, u_old = sv->u;
    sv->T = DMAX(1.0, DMIN(1e10, sv->T));
    sv->u = DMAX(All.MinEgySpec * UNIT_SPECEGY_IN_CGS, DMIN(1e17, sv->u));
    if (sv->T != T_old || sv->u != u_old) clamped = 1;
    for (int k = 2; k < N_VARS; k++) {
        double old = sv->data[k];
        sv->data[k] = DMAX(JACO_ABUNDANCE_FLOOR, DMIN(1.0, sv->data[k]));
        if (sv->data[k] != old) clamped = 1;
    }
    return clamped;
}

/* Returns 1 if the solver has not yet converged, 0 if converged. */
static int iter_condition(const SolveVars *sv, const SolveVars *dsv, double tol) {
    for (int i = 0; i < N_VARS; i++) {
        /* NaN in state or step is never converged */
        if (!jaco_isfinite(sv->data[i]) || !jaco_isfinite(dsv->data[i]))
            return 1;

        /* Abundances pinned at the floor with a negative step are constrained, not unconverged */
        if (i >= 2 && sv->data[i] <= JACO_ABUNDANCE_FLOOR && dsv->data[i] <= 0)
            continue;

        /* For abundances, add an absolute tolerance so that tiny species
           (e.g. x_Heplus ~ 1e-13 with dsv ~ 1e-16) count as converged */
        double abstol = 0;
        if (i >= 2)
            abstol = JACO_ABUNDANCE_FLOOR;

        if (fabs(dsv->data[i]) > tol * fabs(sv->data[i]) + abstol)
            return 1;
    }
    return 0;
}

/* Solve the CIE subsystem at fixed T: equilibrate ion abundances only (rows 2+).
   Replaces the u and T equations with identity rows so only chemistry updates. */
static void solve_CIE_subsystem(SolveVars *sv, const Params *pr, double tol, int maxiter);

/* Solve A * x = b via QR decomposition. */
static void qr_solve(double A[N_VARS][N_VARS], double b[N_VARS], double x[N_VARS]) {
    double A_flat[N_VARS * N_VARS];
    for (int i = 0; i < N_VARS; i++)
        for (int j = 0; j < N_VARS; j++)
            A_flat[i * N_VARS + j] = A[i][j];

    gsl_matrix_view Am = gsl_matrix_view_array(A_flat, N_VARS, N_VARS);
    gsl_vector_view bv = gsl_vector_view_array(b, N_VARS);
    gsl_vector_view xv = gsl_vector_view_array(x, N_VARS);
    gsl_vector *tau = gsl_vector_alloc(N_VARS);
    gsl_linalg_QR_decomp(&Am.matrix, tau);
    gsl_linalg_QR_solve(&Am.matrix, tau, &bv.vector, &xv.vector);
    gsl_vector_free(tau);
}

/* Solve the CIE subsystem at fixed T (ion equilibrium only). Used as an
   operator-splitting step within the outer Newton loop to decouple chemistry
   from thermal dynamics. */
static void solve_CIE_subsystem(SolveVars *sv, const Params *pr, double tol, int maxiter) {
    SolveVars func, dsv;
    double jac[N_VARS][N_VARS];
    for (int iter = 0; iter < maxiter; iter++) {
        microphysics_func_jac(sv, pr, &func, jac);
        /* Zero out u and T rows — only solve the chemistry equations */
        func.data[0] = 0;
        func.data[1] = 0;
        for (int j = 0; j < N_VARS; j++) {
            jac[0][j] = 0;
            jac[1][j] = 0;
        }
        jac[0][0] = 1;
        jac[1][1] = 1;

        /* Check chemistry convergence (rows 2+) */
        int converged = 1;
        for (int k = 2; k < N_VARS; k++) {
            if (fabs(func.data[k]) > tol * (fabs(sv->data[k]) * pr->n_Htot / pr->Delta_t + 1e-20))
                converged = 0;
        }
        if (converged)
            return;

        double rhs[N_VARS];
        for (int i = 0; i < N_VARS; i++)
            rhs[i] = -func.data[i];
        qr_solve(jac, rhs, dsv.data);

        for (int k = 2; k < N_VARS; k++)
            sv->data[k] = DMAX(JACO_ABUNDANCE_FLOOR, DMIN(1.0, sv->data[k] + dsv.data[k]));
    }
}

/* One Newton step: solve J * dsv = -func, then sv += dsv and clamp. */
static void jaco_newton_step(SolveVars *sv, SolveVars *dsv, const SolveVars *func, double jac[N_VARS][N_VARS]) {
    double rhs[N_VARS];
    for (int i = 0; i < N_VARS; i++)
        rhs[i] = -func->data[i];

    qr_solve(jac, rhs, dsv->data);

    for (int k = 0; k < N_VARS; k++)
        sv->data[k] += dsv->data[k];

    jaco_clamp(sv);
}

/* Run Newton iterations with damping factor that ramps from 1/careful_steps to 1.
   Returns number of iterations taken, or -1 if MAXITER reached without convergence.
   If verbose, prints all iterates to stdout.
   If limit_T_step, restrict the T step so T changes by no more than a factor 1.1
   per iteration — a trust-region-like safeguard for stiff low-density cases.
   *nfeval is incremented by the number of microphysics_func_jac calls. */
static int jaco_newton_loop(SolveVars *sv, const Params *pr, double tol, int careful_steps, int verbose,
                            int limit_T_step, int *nfeval) {
    SolveVars func, dsv;
    double jac[N_VARS][N_VARS];

    for (int i = 0; i < N_VARS; i++)
        dsv.data[i] = MAX_REAL_NUMBER;

    /* In the fallback stages (damping or T-step limiting), sync u with T via the
       EOS so the func[u] residual is always zero. Without this, a stale u from
       the hydro step produces a huge EOS residual that drives Newton's T step
       even when we're trying to cautiously approach a solution. */
    int sync_u_to_T = (careful_steps > 1) || limit_T_step;

    for (int iter = 0; iter < MAXITER; iter++) {
        if (!iter_condition(sv, &dsv, tol))
            return iter;

        if (sync_u_to_T)
            sv->u = jaco_T_to_u(sv->T, sv, pr, NULL);

        microphysics_func_jac(sv, pr, &func, jac);
        if (nfeval) (*nfeval)++;

        double fac = fmin(1.0, ((double)iter + 1) / careful_steps);
        double rhs[N_VARS];
        for (int i = 0; i < N_VARS; i++)
            rhs[i] = -func.data[i];
        qr_solve(jac, rhs, dsv.data);

        /* Bail immediately if QR produced NaN (singular Jacobian) */
        {
            int has_nan = 0;
            for (int k = 0; k < N_VARS; k++)
                if (!jaco_isfinite(dsv.data[k])) { has_nan = 1; break; }
            if (has_nan) return -1;
        }

        if (verbose) {
            printf("  iter=%d sv:", iter);
            for (int k = 0; k < N_VARS; k++)
                printf(" %.4e", sv->data[k]);
            printf("  func:");
            for (int k = 0; k < N_VARS; k++)
                printf(" %.4e", func.data[k]);
            printf("  dsv:");
            for (int k = 0; k < N_VARS; k++)
                printf(" %.4e", dsv.data[k]);
            printf("\n");
            /* Decompose dsv[T] into contributions from each func component:
               dsv = -J^{-1} func, so dsv[T] = sum_i -(J^{-1})[T][i] * func[i].
               Compute each contribution by zeroing all but one func entry. */
            printf("    dsv[T] contributions (zeroing all but func[i]):");
            for (int i = 0; i < N_VARS; i++) {
                double rhs2[N_VARS] = {0};
                rhs2[i] = -func.data[i];
                double dsv2[N_VARS];
                double jac_copy[N_VARS][N_VARS];
                for (int a = 0; a < N_VARS; a++)
                    for (int b = 0; b < N_VARS; b++)
                        jac_copy[a][b] = jac[a][b];
                qr_solve(jac_copy, rhs2, dsv2);
                printf(" [%d]=%.4e", i, dsv2[1]);
            }
            printf("\n");
        }

        /* Apply damping factor from careful_steps */
        for (int k = 0; k < N_VARS; k++)
            dsv.data[k] *= fac;

        /* Trust-region limit on T: cap |dT/T| to 0.1 (i.e. factor-of-1.1 change per iter) */
        if (limit_T_step) {
            double max_dT = 0.1 * fabs(sv->T);
            if (fabs(dsv.data[IDX_T]) > max_dT) {
                double scale = max_dT / fabs(dsv.data[IDX_T]);
                for (int k = 0; k < N_VARS; k++)
                    dsv.data[k] *= scale;
            }
        }

        /* Fraction-to-the-boundary: scale dsv so that no abundance falls below
           its floor, and the conservation-derived neutral abundances (x_H, x_He)
           stay non-negative. Required because Newton doesn't see these constraints
           and can produce unphysical states where neutral abundances go negative,
           which causes cooling rates to flip sign and T to run away. */
        {
            const double tau = 0.99; /* land at most 99% of the way to the boundary */
            double alpha = 1.0;

            /* Per-species non-negativity: x_k + alpha*dsv_k >= (1-tau)*x_k.
               Skip variables already at the floor — those are active-set constraints
               handled by the post-step clamp; letting them drive alpha would freeze
               the whole solver when one species sits at its floor. */
            for (int k = 2; k < N_VARS; k++) {
                if (sv->data[k] <= 2.0 * JACO_ABUNDANCE_FLOOR) continue;
                if (dsv.data[k] < 0) {
                    double max_step = -tau * sv->data[k] / dsv.data[k];
                    if (max_step < alpha) alpha = max_step;
                }
            }

#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
            /* Neutral H budget: x_H = 1 - x_Hp - 2*x_H2 >= 0.
               dsv[x_H] = -(dsv[x_Hp] + 2*dsv[x_H2]) */
            {
                double xH_now = 1.0 - sv->x_Hplus - 2.0 * sv->x_H_2;
                double dxH = -(dsv.x_Hplus + 2.0 * dsv.x_H_2);
                if (dxH < 0 && xH_now > 0) {
                    double max_step = -tau * xH_now / dxH;
                    if (max_step < alpha) alpha = max_step;
                }
            }
#elif defined(JACO_MODEL_KWH)
            /* Neutral H budget: x_H = 1 - x_Hp >= 0 */
            {
                double xH_now = 1.0 - sv->x_Hplus;
                double dxH = -dsv.x_Hplus;
                if (dxH < 0 && xH_now > 0) {
                    double max_step = -tau * xH_now / dxH;
                    if (max_step < alpha) alpha = max_step;
                }
            }
#endif

#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL) || defined(JACO_MODEL_KWH)
            /* Neutral He budget: x_He = y - x_Hep - x_Hepp >= 0 */
            {
                double xHe_now = pr->y - sv->x_Heplus - sv->x_Heplusplus;
                double dxHe = -(dsv.x_Heplus + dsv.x_Heplusplus);
                if (dxHe < 0 && xHe_now > 0) {
                    double max_step = -tau * xHe_now / dxHe;
                    if (max_step < alpha) alpha = max_step;
                }
            }
#endif

            /* Only scale the abundance rows (k>=2). Don't let chemistry floors
               shrink u/T steps — thermal dynamics should evolve at full damping,
               while chemistry stays within physical bounds via alpha. */
            if (alpha < 1.0) {
                for (int k = 2; k < N_VARS; k++)
                    dsv.data[k] *= alpha;
            }
        }

        for (int k = 0; k < N_VARS; k++)
            sv->data[k] += dsv.data[k];

        /* If any variable hit its physical bounds, the step was truncated by the
           clamp and dsv no longer reflects the actual change. Reset dsv to large
           values so iter_condition forces at least one more iteration to verify
           convergence at the clamped state, rather than falsely declaring success
           based on the pre-clamp step.
           Exception: abundances that were already at the floor before the step
           and got pushed further below (a normal active-set constraint) should
           NOT trigger a global dsv reset — iter_condition's floor-skip handles them. */
        {
            SolveVars sv_pre = *sv; /* state before clamp but after step */
            int clamped_nontrivially = jaco_clamp(sv);
            if (clamped_nontrivially) {
                /* Check if the only clamping was abundances already at the floor */
                int nontrivial = 0;
                if (sv->T != sv_pre.T || sv->u != sv_pre.u) nontrivial = 1;
                for (int k = 2; k < N_VARS && !nontrivial; k++) {
                    if (sv->data[k] != sv_pre.data[k] && sv_pre.data[k] > 2.0 * JACO_ABUNDANCE_FLOOR)
                        nontrivial = 1; /* clamped a variable that wasn't already at the floor */
                }
                if (nontrivial)
                    for (int k = 0; k < N_VARS; k++)
                        dsv.data[k] = MAX_REAL_NUMBER;
            }
        }

    }
    return -1;
}

/* Try to solve a single implicit step with the 2-stage fallback strategy.
   Returns: nfeval on success, -1 on hard failure. */
static int jaco_solve_single(SolveVars *sv, Params *pr, double tol, int *nfeval_out) {
    const SolveVars sv0 = *sv;
    int nfeval = 0;
    int converged = 0;

    /* Stage 1: undamped Newton */
    *sv = sv0;
    if (jaco_newton_loop(sv, pr, tol, 1, 0, 0, &nfeval) >= 0)
        converged = 1;

    /* Stage 2: damped Newton with EOS sync.
       Also sync u_initial to u(T_initial) so the BDF term starts at zero —
       otherwise a u/T inconsistency from the hydro step creates a constant
       offset that no amount of dt-halving can resolve. */
    // if (!converged) {
    //     *sv = sv0;
    //     /* Reset to a fully self-consistent initial state: CIE ions for stored T,
    //        then u = u(T, ions). This eliminates the u/T inconsistency from the
    //        hydro step that prevents convergence even at tiny sub-timesteps. */
    //     double logT = log10(DMAX(sv->T, 10.));
    //     sv->x_Hplus = cie_interp(cie_xHp, logT);
    //     sv->x_Heplus = cie_interp(cie_xHep, logT);
    //     sv->x_Heplusplus = cie_interp(cie_xHepp, logT);
    //     pr->u_initial = jaco_T_to_u(sv->T, sv, pr, NULL);
    //     sv->u = pr->u_initial;
    //     if (jaco_newton_loop(sv, pr, tol, 30, 0, 1, &nfeval) >= 0)
    //         converged = 1;
    // }

    /* Always report nfeval to the caller, even on failure */
    if (nfeval_out) *nfeval_out += nfeval;

    if (!converged) {
        printf("jaco_solve_single FAILED: T0=%g T=%g n=%g dt=%g nfeval=%d\n",
               sv0.T, sv->T, pr->n_Htot, pr->Delta_t, nfeval);
        /* Replay stage 2 verbose to see what's happening */
        static int _replay_count = 0;
        if (_replay_count < 1) {
            *sv = sv0;
            // double logT = log10(DMAX(sv->T, 10.));
            // sv->x_Hplus = cie_interp(cie_xHp, logT);
            // sv->x_Heplus = cie_interp(cie_xHep, logT);
            // sv->x_Heplusplus = cie_interp(cie_xHepp, logT);
            // pr->u_initial = jaco_T_to_u(sv->T, sv, pr, NULL);
            // sv->u = pr->u_initial;
            jaco_newton_loop(sv, pr, tol, 1, 1, 0, NULL);
            fflush(stdout);
            _replay_count++;
        }
    return -1;
    }

    /* Announce any suspiciously hot result */
    if (sv->T > 2e4)
        printf("jaco WARNING: T=%g > 2e4 after solve (T0=%g n=%g dt=%g nfeval=%d)\n",
               sv->T, sv0.T, pr->n_Htot, pr->Delta_t, nfeval);

    return nfeval;
}

int jaco_solve(SolveVars *sv, const Params *pr_in, double tol) {
    Params pr = *pr_in;
    const SolveVars sv0 = *sv;
    int nfeval_total = 0;

    /* Attempt the full timestep. On failure, halve the sub-step size and
       subcycle until the full interval is covered. */
    double dt_remaining = pr.Delta_t;
    double dt_sub = dt_remaining;
    const int MAX_SUBSTEPS = 256;

    for (int substep = 0; substep < MAX_SUBSTEPS && dt_remaining > 1e-30 * pr_in->Delta_t; substep++) {
        /* Set up sub-step params */
        SolveVars sv_save = *sv;
        pr.Delta_t = dt_sub;
        pr.u_initial = sv->u;
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
        pr.x_H_2_initial = sv->x_H_2;
#endif

        int ret = jaco_solve_single(sv, &pr, tol, &nfeval_total);

        if (ret >= 0) {
            /* Sub-step succeeded — advance time */
            nfeval_total += ret;
            dt_remaining -= dt_sub;
            dt_sub = DMIN(dt_sub * 2.0, dt_remaining);
            continue;
        }

        /* Failure: halve the sub-step and retry from saved state */
        *sv = sv_save;
        dt_sub *= 0.5;
        printf("jaco_solve: subcycling dt_sub halved to %g (n=%g T=%g dt=%g)\n",
               dt_sub, pr_in->n_Htot, sv->T, pr_in->Delta_t);

        if (dt_sub < pr_in->Delta_t * 1e-10) {
            printf("jaco_solve: subcycle dt_sub too small (n=%g T0=%g T=%g dt_sub=%g dt=%g nfeval=%d)\n",
                   pr_in->n_Htot, sv0.T, sv->T, dt_sub, pr_in->Delta_t, nfeval_total);
            fflush(stdout);
            endrun(10);
        }
    }

    return nfeval_total;
}
#endif
