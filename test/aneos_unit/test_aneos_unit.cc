/*
 * Standalone unit test for the ANEOS (SESAME tabulated EOS) module.
 *
 * This test:
 *   1. Generates an ideal-gas SESAME table file analytically
 *   2. Loads it with aneos_read_table()
 *   3. Tests direct (rho, T) lookups against analytic values
 *   4. Tests T-inversion from (rho, u) via aneos_compute()
 *   5. Tests derived quantities (Cv, Gruneisen parameter)
 *   6. Tests boundary clamping behavior
 *
 * Compile:
 *   mpicxx -std=c++17 -DEOS_ANEOS -DGIZMO_config_H -I../.. \
 *     test_aneos_unit.cc ../../eos/aneos.cc -o test_aneos_unit -lm
 *
 * Run:
 *   ./test_aneos_unit
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EOS_ANEOS
#define GIZMO_config_H
#include "../../eos/aneos.h"

/* ---- Test parameters ---- */
static const double GAMMA = 5.0 / 3.0;
static const double CV = 1.0e7;  /* erg/g/K — specific heat at constant volume */

/* For an ideal gas with constant Cv:
 *   u = Cv * T
 *   P = (gamma - 1) * rho * u = (gamma - 1) * rho * Cv * T
 *   S = Cv * ln(T) - (gamma-1)*Cv * ln(rho)  + const  (entropy, up to additive constant)
 *   cs = sqrt(gamma * P / rho) = sqrt(gamma * (gamma-1) * Cv * T)
 *   Gruneisen = gamma - 1
 */

static double analytic_P(double rho, double T) { return (GAMMA - 1.0) * rho * CV * T; }
static double analytic_u(double T) { return CV * T; }
static double analytic_cs(double T) { return sqrt(GAMMA * (GAMMA - 1.0) * CV * T); }
/* Entropy with arbitrary offset to keep S > 0 in our table range */
static double analytic_S(double rho, double T) {
    return CV * log(T) - (GAMMA - 1.0) * CV * log(rho) + 1.0e8;
}

/* ---- Table generation ---- */

static const char *TABLE_FILE = "test_ideal_gas.sesame";
static const int NRHO = 100;
static const int NT = 100;
static const double LOGRHO_MIN = -2.0;  /* 0.01 g/cm^3 */
static const double LOGRHO_MAX = 2.0;   /* 100 g/cm^3 */
static const double LOGT_MIN = 2.0;     /* 100 K */
static const double LOGT_MAX = 6.0;     /* 1e6 K */

static int generate_ideal_gas_table(const char *fname)
{
    FILE *fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "Cannot create %s\n", fname); return 1; }

    int mat_id = 9999;
    int ncols = 6; /* rho T P u S cs */
    fprintf(fp, "%d %d %d %d\n", mat_id, NRHO, NT, ncols);

    /* Write density grid */
    for (int i = 0; i < NRHO; i++) {
        double lr = LOGRHO_MIN + i * (LOGRHO_MAX - LOGRHO_MIN) / (NRHO - 1);
        fprintf(fp, "%.15e\n", pow(10.0, lr));
    }

    /* Write temperature grid */
    for (int j = 0; j < NT; j++) {
        double lT = LOGT_MIN + j * (LOGT_MAX - LOGT_MIN) / (NT - 1);
        fprintf(fp, "%.15e\n", pow(10.0, lT));
    }

    /* Write 2D data: rho T P u S cs */
    for (int i = 0; i < NRHO; i++) {
        double lr = LOGRHO_MIN + i * (LOGRHO_MAX - LOGRHO_MIN) / (NRHO - 1);
        double rho = pow(10.0, lr);
        for (int j = 0; j < NT; j++) {
            double lT = LOGT_MIN + j * (LOGT_MAX - LOGT_MIN) / (NT - 1);
            double T = pow(10.0, lT);
            double P = analytic_P(rho, T);
            double u = analytic_u(T);
            double S = analytic_S(rho, T);
            double cs = analytic_cs(T);
            fprintf(fp, "%.15e %.15e %.15e %.15e %.15e %.15e\n",
                    rho, T, P, u, S, cs);
        }
    }

    fclose(fp);
    return 0;
}

/* ---- Test helpers ---- */

static int n_tests = 0;
static int n_pass = 0;
static int n_fail = 0;

static void check(const char *name, double got, double expected, double rtol)
{
    n_tests++;
    double rel_err = (expected != 0) ? fabs(got - expected) / fabs(expected) : fabs(got);
    if (rel_err <= rtol) {
        n_pass++;
    } else {
        n_fail++;
        fprintf(stderr, "  FAIL: %s: got %.6e, expected %.6e, rel_err=%.2e (tol=%.2e)\n",
                name, got, expected, rel_err, rtol);
    }
}

/* ---- Tests ---- */

static void test_table_loading(void)
{
    printf("--- Test: Table loading ---\n");
    int ierr = aneos_read_table(TABLE_FILE, 0);
    check("aneos_read_table returns 0", (double)ierr, 0.0, 0.0);
    check("table loaded flag", (double)ANEOS_Tables[0].loaded, 1.0, 0.0);
    check("nrho", (double)ANEOS_Tables[0].nrho, (double)NRHO, 0.0);
    check("nT", (double)ANEOS_Tables[0].nT, (double)NT, 0.0);
    check("has_csound", (double)ANEOS_Tables[0].has_csound, 1.0, 0.0);
    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

static void test_direct_lookup(void)
{
    printf("--- Test: Direct (rho, T) lookup ---\n");
    /* Test at grid points (should be exact up to log-space interpolation) */
    double test_rhos[] = {0.1, 1.0, 10.0, 50.0};
    double test_Ts[] = {300.0, 1000.0, 10000.0, 500000.0};
    int n_rho = 4, n_T = 4;

    /* Tolerance: bilinear in log-log introduces some error for quantities
       that aren't perfectly log-linear. For an ideal gas, P ~ rho*T is
       perfectly log-linear, so P should be very accurate.
       u ~ T is log-linear in T only. cs ~ sqrt(T) is log-linear. */
    double rtol_P = 0.02;   /* P = (g-1)*rho*Cv*T is bilinear in log(rho)+log(T) — good */
    double rtol_u = 0.02;   /* u = Cv*T — log-linear in log(T) */
    double rtol_cs = 0.02;  /* cs = sqrt(g*(g-1)*Cv*T) — log-linear in log(T) */

    for (int i = 0; i < n_rho; i++) {
        for (int j = 0; j < n_T; j++) {
            double rho = test_rhos[i], T = test_Ts[j];
            double P_got, u_got, S_got, cs_got;
            int phase_got;
            int ierr = aneos_lookup_rhoT(0, rho, T, &P_got, &u_got, &S_got, &cs_got, &phase_got);

            char label[128];
            snprintf(label, sizeof(label), "lookup P(rho=%.1f, T=%.0f)", rho, T);
            check(label, P_got, analytic_P(rho, T), rtol_P);

            snprintf(label, sizeof(label), "lookup u(rho=%.1f, T=%.0f)", rho, T);
            check(label, u_got, analytic_u(T), rtol_u);

            snprintf(label, sizeof(label), "lookup cs(rho=%.1f, T=%.0f)", rho, T);
            check(label, cs_got, analytic_cs(T), rtol_cs);

            /* Check ierr */
            snprintf(label, sizeof(label), "lookup ierr(rho=%.1f, T=%.0f)", rho, T);
            check(label, (double)ierr, 0.0, 0.0);
        }
    }
    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

static void test_temperature_inversion(void)
{
    printf("--- Test: Temperature inversion (rho, u) -> T ---\n");
    double test_rhos[] = {0.1, 1.0, 10.0, 50.0};
    double test_Ts[] = {300.0, 1000.0, 10000.0, 500000.0};
    int n_rho = 4, n_T = 4;
    double rtol_T = 0.01;  /* 1% tolerance on recovered T */
    double rtol_P = 0.03;  /* slightly looser on P since it compounds T error */

    for (int i = 0; i < n_rho; i++) {
        for (int j = 0; j < n_T; j++) {
            double rho = test_rhos[i], T_true = test_Ts[j];
            double u_in = analytic_u(T_true);

            /* Start with a deliberately bad initial guess */
            double T_guess = 5000.0;
            double P_out, cs_out, S_out, cv_out, grun_out;
            int phase_out;

            int ierr = aneos_compute(0, rho, u_in, &T_guess,
                                     &P_out, &cs_out, &S_out, &cv_out, &grun_out, &phase_out);

            char label[128];
            snprintf(label, sizeof(label), "inversion T(rho=%.1f, T_true=%.0f)", rho, T_true);
            check(label, T_guess, T_true, rtol_T);

            snprintf(label, sizeof(label), "inversion P(rho=%.1f, T_true=%.0f)", rho, T_true);
            check(label, P_out, analytic_P(rho, T_true), rtol_P);

            snprintf(label, sizeof(label), "inversion ierr(rho=%.1f, T_true=%.0f)", rho, T_true);
            check(label, (double)ierr, 0.0, 0.0);
        }
    }
    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

static void test_derived_quantities(void)
{
    printf("--- Test: Derived quantities (Cv, Gruneisen) ---\n");
    double rho = 1.0, T_true = 10000.0;
    double u_in = analytic_u(T_true);
    double T_guess = T_true;  /* good initial guess */
    double P_out, cs_out, S_out, cv_out, grun_out;
    int phase_out;

    aneos_compute(0, rho, u_in, &T_guess,
                  &P_out, &cs_out, &S_out, &cv_out, &grun_out, &phase_out);

    /* Cv should be our constant CV */
    check("Cv", cv_out, CV, 0.05);

    /* Gruneisen = gamma - 1 for ideal gas */
    check("Gruneisen", grun_out, GAMMA - 1.0, 0.05);

    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

static void test_boundary_clamping(void)
{
    printf("--- Test: Boundary clamping ---\n");
    /* Query outside table bounds — should clamp without crashing */
    double P_out, u_out, S_out, cs_out;
    int phase_out;

    /* Very low density */
    int ierr = aneos_lookup_rhoT(0, 1e-10, 1000.0, &P_out, &u_out, &S_out, &cs_out, &phase_out);
    check("low-rho clamp ierr", (double)ierr, 0.0, 0.0);
    check("low-rho clamp P > 0", (P_out > 0) ? 1.0 : 0.0, 1.0, 0.0);

    /* Very high density */
    ierr = aneos_lookup_rhoT(0, 1e10, 1000.0, &P_out, &u_out, &S_out, &cs_out, &phase_out);
    check("high-rho clamp ierr", (double)ierr, 0.0, 0.0);
    check("high-rho clamp P > 0", (P_out > 0) ? 1.0 : 0.0, 1.0, 0.0);

    /* Very low temperature */
    ierr = aneos_lookup_rhoT(0, 1.0, 1.0, &P_out, &u_out, &S_out, &cs_out, &phase_out);
    check("low-T clamp ierr", (double)ierr, 0.0, 0.0);

    /* Very high temperature */
    ierr = aneos_lookup_rhoT(0, 1.0, 1e12, &P_out, &u_out, &S_out, &cs_out, &phase_out);
    check("high-T clamp ierr", (double)ierr, 0.0, 0.0);

    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

static void test_inversion_with_good_guess(void)
{
    printf("--- Test: T-inversion with cached (good) guess ---\n");
    /* Simulates the typical runtime case where the previous timestep's T
       is close to the current T */
    double rho = 5.0, T_true = 50000.0;
    double u_in = analytic_u(T_true);
    double T_guess = T_true * 1.01;  /* 1% off — should converge in 1-2 iterations */
    double P_out, cs_out, S_out, cv_out, grun_out;
    int phase_out;

    aneos_compute(0, rho, u_in, &T_guess,
                  &P_out, &cs_out, &S_out, &cv_out, &grun_out, &phase_out);

    check("good-guess T recovery", T_guess, T_true, 0.001);
    check("good-guess P", P_out, analytic_P(rho, T_true), 0.01);

    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

static void test_invalid_material(void)
{
    printf("--- Test: Invalid material index ---\n");
    double T_guess = 1000.0;
    double P, cs, S, cv, grun;
    int phase;

    int ierr = aneos_compute(99, 1.0, 1e10, &T_guess, &P, &cs, &S, &cv, &grun, &phase);
    check("invalid mat returns error", (ierr != 0) ? 1.0 : 0.0, 1.0, 0.0);

    ierr = aneos_compute(-1, 1.0, 1e10, &T_guess, &P, &cs, &S, &cv, &grun, &phase);
    check("negative mat returns error", (ierr != 0) ? 1.0 : 0.0, 1.0, 0.0);

    printf("  %d/%d passed\n\n", n_pass, n_tests);
}

/* ---- Main ---- */

int main(void)
{
    printf("=== ANEOS Unit Test Suite ===\n\n");

    /* Step 1: Generate the ideal-gas table */
    printf("Generating ideal-gas SESAME table: %s\n", TABLE_FILE);
    int ierr = generate_ideal_gas_table(TABLE_FILE);
    if (ierr) { fprintf(stderr, "FATAL: could not generate table\n"); return 1; }
    printf("Done.\n\n");

    /* Step 2: Run tests */
    test_table_loading();
    test_direct_lookup();
    test_temperature_inversion();
    test_derived_quantities();
    test_boundary_clamping();
    test_inversion_with_good_guess();
    test_invalid_material();

    /* Cleanup */
    aneos_cleanup();
    remove(TABLE_FILE);

    /* Summary */
    printf("=== Summary: %d/%d tests passed", n_pass, n_tests);
    if (n_fail > 0)
        printf(", %d FAILED", n_fail);
    printf(" ===\n");

    return (n_fail > 0) ? 1 : 0;
}
