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
 *     test_aneos_unit.cc ../../src/eos/aneos.cc -o test_aneos_unit -lm
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
#include "../../src/eos/aneos.h"

/* aneos.cc puts the tables in memory the host and the device can both read.
   The real entry points for that live in the GPU arena, which needs Kokkos,
   MPI and the global state this test is built to run without, so they are
   served here from the heap instead. The lookups under test do not depend on
   where the table sits, and aneos.cc zeroes what it allocates itself. */
extern "C" void *gizmo_gpu_alloc_shared(size_t nbytes, const char *label) { (void)label; return malloc(nbytes); }
extern "C" void gpu_particles_uvm_free(void *ptr) { free(ptr); }

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

/* ---- Irregular-grid fixture ----
 * Real tables (e.g. the Stewart M-ANEOS releases) space their axes unevenly. This
 * fixture uses deliberately irregular axes (adjacent spacings differing by up to a
 * factor of ~2000) and fields that are bilinear in (log10 rho, log10 T), which a
 * correct lookup reproduces exactly at every point, knot or not. Entropy is a
 * signed linear field with a zero on a knot. Phase = 100*irho + iT. */

static const char *IRR_TABLE_FILE = "test_irregular.sesame";
static const double IRR_LR[] = {-3.0, -2.9, -2.85, -1.0, 0.0, 0.001, 0.002, 0.5, 2.0, 2.05, 4.0};
static const double IRR_LT[] = {0.0, 0.003, 1.0, 1.5, 1.51, 3.0, 5.0, 5.001, 7.0};
static const int IRR_NR = sizeof(IRR_LR) / sizeof(IRR_LR[0]);
static const int IRR_NT = sizeof(IRR_LT) / sizeof(IRR_LT[0]);

static double irr_logP(double lr, double lT)  { return 1.0 + lr + lT + 0.1 * lr * lT; }
static double irr_logu(double lr, double lT)  { return 8.0 + 0.2 * lr + lT + 0.05 * lr * lT; }
static double irr_logcs(double lr, double lT) { return 5.0 + 0.5 * lT - 0.1 * lr; }
static double irr_S(double lr, double lT)     { return 1.0e7 * (lT - 2.0 * lr - 1.0); }

/* per-density lowest valid temperature rows written by write_irregular_table when asked (one per IRR_LR) */
static const int IRR_LOW[] = {0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2};

/* writes the fixture; 'defect' plants one inconsistency for the refusal tests:
   0 none, 1 density axis not increasing, 2 a row's T differs from the axis, 3 a non-positive axis value,
   4 a lowest valid row outside [0, nT-2]. ncols = 7 (all columns) or 5 (no sound speed or phase, so the code
   estimates cs). low_block = 1 appends the per-density lowest-valid-row block (IRR_LOW). */
static int write_irregular_table_block(const char *fname, int defect, int ncols, int low_block)
{
    FILE *fp = fopen(fname, "w");
    if(!fp) { fprintf(stderr, "Cannot create %s\n", fname); return 1; }
    fprintf(fp, "%d %d %d %d\n", 4242, IRR_NR, IRR_NT, ncols);
    for(int i = 0; i < IRR_NR; i++) {
        double rho = pow(10.0, IRR_LR[i]);
        if(defect == 1 && i == 5) rho = pow(10.0, IRR_LR[3]);
        if(defect == 3 && i == 0) rho = 0.0;
        fprintf(fp, "%.15e\n", rho);
    }
    for(int j = 0; j < IRR_NT; j++) fprintf(fp, "%.15e\n", pow(10.0, IRR_LT[j]));
    for(int i = 0; i < IRR_NR; i++) {
        for(int j = 0; j < IRR_NT; j++) {
            double lr = IRR_LR[i], lT = IRR_LT[j];
            double T_row = pow(10.0, lT);
            if(defect == 2 && i == 2 && j == 3) T_row *= 1.001;
            fprintf(fp, "%.15e %.15e %.15e %.15e %.15e", pow(10.0, lr), T_row,
                    pow(10.0, irr_logP(lr, lT)), pow(10.0, irr_logu(lr, lT)), irr_S(lr, lT));
            if(ncols == 7) fprintf(fp, " %.15e %d", pow(10.0, irr_logcs(lr, lT)), 100 * i + j);
            fprintf(fp, "\n");
        }
    }
    if(low_block) {
        fprintf(fp, "lowest_valid_T_index\n");
        for(int i = 0; i < IRR_NR; i++) fprintf(fp, "%d ", (defect == 4 && i == 3) ? IRR_NT - 1 : IRR_LOW[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}

static int write_irregular_table(const char *fname, int defect, int ncols) {return write_irregular_table_block(fname, defect, ncols, 0);}

/* The lookup this module used before irregular axes were supported: a single mean
   spacing per axis. Kept here only to prove the fixture detects that defect. */
static double old_uniform_index_logP(const struct aneos_table *tbl, double lr, double lT)
{
    double d_r = (tbl->logrho_max - tbl->logrho_min) / (tbl->nrho - 1), d_T = (tbl->logT_max - tbl->logT_min) / (tbl->nT - 1);
    double fi = (lr - tbl->logrho_min) / d_r, fj = (lT - tbl->logT_min) / d_T;
    int i = (int)fi, j = (int)fj;
    if(i > tbl->nrho - 2) i = tbl->nrho - 2;
    if(j > tbl->nT - 2) j = tbl->nT - 2;
    double t = fi - i, u = fj - j; int nT = tbl->nT;
    return (1-t)*(1-u)*tbl->log_pressure[i*nT+j] + t*(1-u)*tbl->log_pressure[(i+1)*nT+j]
         + (1-t)*u*tbl->log_pressure[i*nT+j+1] + t*u*tbl->log_pressure[(i+1)*nT+j+1];
}

/* absolute-tolerance check, for a field that can be zero or change sign */
static void check_abs(const char *name, double got, double expected, double atol)
{
    n_tests++;
    if(fabs(got - expected) <= atol) { n_pass++; }
    else { n_fail++; fprintf(stderr, "  FAIL: %s: got %.9e, expected %.9e, |diff|=%.2e (atol=%.2e)\n", name, got, expected, fabs(got - expected), atol); }
}

static void irr_check_point(const char *what, double lr, double lT)
{
    double P, u, S, cs; int phase; char label[160];
    int ierr = aneos_lookup_rhoT(1, pow(10.0, lr), pow(10.0, lT), &P, &u, &S, &cs, &phase);
    snprintf(label, sizeof(label), "%s ierr (lr=%g, lT=%g)", what, lr, lT); check(label, (double)ierr, 0.0, 0.0);
    snprintf(label, sizeof(label), "%s P (lr=%g, lT=%g)", what, lr, lT);  check(label, P, pow(10.0, irr_logP(lr, lT)), 1e-10);
    snprintf(label, sizeof(label), "%s u (lr=%g, lT=%g)", what, lr, lT);  check(label, u, pow(10.0, irr_logu(lr, lT)), 1e-10);
    snprintf(label, sizeof(label), "%s cs (lr=%g, lT=%g)", what, lr, lT); check(label, cs, pow(10.0, irr_logcs(lr, lT)), 1e-10);
    snprintf(label, sizeof(label), "%s S (lr=%g, lT=%g)", what, lr, lT);  check_abs(label, S, irr_S(lr, lT), 1e-3);
}

static void test_irregular_grid(void)
{
    printf("--- Test: Irregular-grid table ---\n");
    if(write_irregular_table(IRR_TABLE_FILE, 0, 7)) { check("irregular table written", 0.0, 1.0, 0.0); return; }
    int ierr = aneos_read_table(IRR_TABLE_FILE, 1);
    check("irregular table loads", (double)ierr, 0.0, 0.0);
    if(ierr) return;

    /* every knot */
    for(int i = 0; i < IRR_NR; i++) for(int j = 0; j < IRR_NT; j++) irr_check_point("knot", IRR_LR[i], IRR_LT[j]);

    /* interiors of narrow, wide and mixed cells */
    const double pts[][2] = {{-2.95, 0.0015}, {-2.875, 0.5}, {-1.7, 1.505}, {0.0005, 1.505}, {0.0015, 4.0},
                             {0.25, 5.0005}, {1.3, 6.2}, {2.03, 0.7}, {3.1, 2.2}, {-0.4, 1.2}};
    const int npts = sizeof(pts) / sizeof(pts[0]);
    for(int k = 0; k < npts; k++) irr_check_point("interior", pts[k][0], pts[k][1]);

    /* signed entropy: negative, zero and positive at knots and inside a cell */
    double P, u, S, cs; int phase;
    aneos_lookup_rhoT(1, pow(10.0, 0.0), pow(10.0, 1.0), &P, &u, &S, &cs, &phase); check_abs("S = 0 at knot (0,1)", S, 0.0, 1e-3);
    aneos_lookup_rhoT(1, pow(10.0, 0.5), pow(10.0, 0.0), &P, &u, &S, &cs, &phase); check("S < 0 at knot (0.5,0)", (S < 0) ? 1.0 : 0.0, 1.0, 0.0);
    aneos_lookup_rhoT(1, pow(10.0, -3.0), pow(10.0, 0.0), &P, &u, &S, &cs, &phase); check("S > 0 at knot (-3,0)", (S > 0) ? 1.0 : 0.0, 1.0, 0.0);
    aneos_lookup_rhoT(1, pow(10.0, 0.25), pow(10.0, 1.2), &P, &u, &S, &cs, &phase); check_abs("S inside cell, sign change", S, irr_S(0.25, 1.2), 1e-3);

    /* clamping: outside either axis the value at the nearest edge */
    aneos_lookup_rhoT(1, 1e-10, pow(10.0, 1.2), &P, &u, &S, &cs, &phase); check("clamp low rho", P, pow(10.0, irr_logP(-3.0, 1.2)), 1e-10);
    aneos_lookup_rhoT(1, 1e10, pow(10.0, 1.2), &P, &u, &S, &cs, &phase);  check("clamp high rho", P, pow(10.0, irr_logP(4.0, 1.2)), 1e-10);
    aneos_lookup_rhoT(1, 1.0, 0.5, &P, &u, &S, &cs, &phase);             check("clamp low T", P, pow(10.0, irr_logP(0.0, 0.0)), 1e-10);
    aneos_lookup_rhoT(1, 1.0, 1e9, &P, &u, &S, &cs, &phase);             check("clamp high T", P, pow(10.0, irr_logP(0.0, 7.0)), 1e-10);

    /* phase: nearest knot on either side of a cell midpoint (rho cell [0, 0.001], T cell [1.5, 1.51]) */
    aneos_lookup_rhoT(1, pow(10.0, 0.0004), pow(10.0, 1.504), &P, &u, &S, &cs, &phase); check("phase lower-lower", (double)phase, 100.0 * 4 + 3, 0.0);
    aneos_lookup_rhoT(1, pow(10.0, 0.0006), pow(10.0, 1.504), &P, &u, &S, &cs, &phase); check("phase upper rho", (double)phase, 100.0 * 5 + 3, 0.0);
    aneos_lookup_rhoT(1, pow(10.0, 0.0004), pow(10.0, 1.506), &P, &u, &S, &cs, &phase); check("phase upper T", (double)phase, 100.0 * 4 + 4, 0.0);

    /* (rho, u) -> T inversion inside irregular cells, from a poor guess */
    for(int k = 0; k < npts; k++) {
        double lr = pts[k][0], lT = pts[k][1], T_guess = 3.0e3, cv, grun; char label[160];
        aneos_compute(1, pow(10.0, lr), pow(10.0, irr_logu(lr, lT)), &T_guess, &P, &cs, &S, &cv, &grun, &phase);
        snprintf(label, sizeof(label), "inversion T (lr=%g, lT=%g)", lr, lT); check(label, T_guess, pow(10.0, lT), 1e-6);
        snprintf(label, sizeof(label), "inversion P (lr=%g, lT=%g)", lr, lT); check(label, P, pow(10.0, irr_logP(lr, lT)), 1e-5);
        snprintf(label, sizeof(label), "inversion S (lr=%g, lT=%g)", lr, lT); check_abs(label, S, irr_S(lr, lT), 1e-1);
        /* Cv and Gruneisen are exact slopes of the interpolant: for these bilinear fields,
           dlog u/dlog T = 1 + 0.05 lr and dlog P/dlog T = 1 + 0.1 lr */
        double rho = pow(10.0, lr), Tt = pow(10.0, lT), uu = pow(10.0, irr_logu(lr, lT)), PP = pow(10.0, irr_logP(lr, lT));
        double cv_exp = uu / Tt * (1.0 + 0.05 * lr), grun_exp = (PP / Tt * (1.0 + 0.1 * lr)) / (rho * cv_exp);
        snprintf(label, sizeof(label), "Cv exact slope (lr=%g, lT=%g)", lr, lT); check(label, cv, cv_exp, 1e-6);
        snprintf(label, sizeof(label), "Gruneisen exact slope (lr=%g, lT=%g)", lr, lT); check(label, grun, grun_exp, 1e-6);
    }

    /* knot convention: at an interior knot (the stored axis value) the locator selects the interval
       above it (t = 0); at the last knot, the interval below it (t = 1) */
    for(int k = 0; k < IRR_NT; k++) {
        struct aneos_axis_location loc = aneos_locate_T(&ANEOS_Tables[1], ANEOS_Tables[1].logT[k]); char label[96];
        snprintf(label, sizeof(label), "T knot %d selects interval", k); check(label, (double)loc.i, (double)((k < IRR_NT - 1) ? k : IRR_NT - 2), 0.0);
        snprintf(label, sizeof(label), "T knot %d fraction", k); check_abs(label, loc.t, (k < IRR_NT - 1) ? 0.0 : 1.0, 1e-12);
    }
    for(int k = 0; k < IRR_NR; k++) {
        struct aneos_axis_location loc = aneos_locate_rho(&ANEOS_Tables[1], ANEOS_Tables[1].logrho[k]); char label[96];
        snprintf(label, sizeof(label), "rho knot %d selects interval", k); check(label, (double)loc.i, (double)((k < IRR_NR - 1) ? k : IRR_NR - 2), 0.0);
    }

    /* sound-speed estimate when the table has no cs column: cs^2 = (dP/drho)_T + T (dP/dT)^2 / (rho^2 Cv),
       with dlog P/dlog rho = 1 + 0.1 lT */
    write_irregular_table(IRR_TABLE_FILE, 0, 5);
    check("5-column irregular table loads", (double)aneos_read_table(IRR_TABLE_FILE, 3), 0.0, 0.0);
    for(int k = 0; k < npts; k++) {
        double lr = pts[k][0], lT = pts[k][1], T_guess = pow(10.0, lT), cv, grun; char label[160];
        aneos_compute(3, pow(10.0, lr), pow(10.0, irr_logu(lr, lT)), &T_guess, &P, &cs, &S, &cv, &grun, &phase);
        double rho = pow(10.0, lr), Tt = pow(10.0, lT), uu = pow(10.0, irr_logu(lr, lT)), PP = pow(10.0, irr_logP(lr, lT));
        double cv_exp = uu / Tt * (1.0 + 0.05 * lr), dPdT = PP / Tt * (1.0 + 0.1 * lr), dPdrho = PP / rho * (1.0 + 0.1 * lT);
        snprintf(label, sizeof(label), "cs estimate (lr=%g, lT=%g)", lr, lT); check(label, cs, sqrt(dPdrho + Tt * dPdT * dPdT / (rho * rho * cv_exp)), 1e-6);
    }

    /* detector proof: the old single-spacing index misses the fixture */
    double worst = 0;
    for(int k = 0; k < npts; k++) {
        double err = fabs(old_uniform_index_logP(&ANEOS_Tables[1], pts[k][0], pts[k][1]) - irr_logP(pts[k][0], pts[k][1]));
        if(err > worst) worst = err;
    }
    printf("  old uniform-index formula: worst |error| in log10 P = %.3f dex\n", worst);
    check("old uniform-index formula fails the fixture", (worst > 0.1) ? 1.0 : 0.0, 1.0, 0.0);

    /* the reader refuses inconsistent tables */
    const char *why[] = {"", "non-increasing density axis", "row T not matching the axis", "non-positive axis value"};
    for(int defect = 1; defect <= 3; defect++) {
        char label[160];
        write_irregular_table(IRR_TABLE_FILE, defect, 7);
        snprintf(label, sizeof(label), "refuses table with %s", why[defect]);
        check(label, (aneos_read_table(IRR_TABLE_FILE, 2) != 0) ? 1.0 : 0.0, 1.0, 0.0);
    }

    /* per-density lowest valid temperature: below it every quantity is the state at that row. The bilinear fields
       are exact on the interpolant, so the expected values are the analytic ones at the floor. */
    write_irregular_table_block(IRR_TABLE_FILE, 0, 7, 1);
    check("table with a lowest-valid-row block loads", (double)aneos_read_table(IRR_TABLE_FILE, 4), 0.0, 0.0);
    {
        /* lr 0.25 lies between knots 6 and 7 (rows 0 and 2): floor at IRR_LT[2] = 1.0; lr -2.875 between knots 1 and 2
           (rows 2 and 4): floor at IRR_LT[4] = 1.51 */
        const double cases[][3] = {{0.25, 0.5, 1.0}, {0.25, 3.0, 3.0}, {-2.875, 1.2, 1.51}, {-2.875, 6.0, 6.0}};
        for(int k = 0; k < 4; k++) {
            double lr = cases[k][0], lT = cases[k][1], lT_eff = cases[k][2], T_guess = 3.0e3, cv, grun; char label[160];
            aneos_lookup_rhoT(4, pow(10.0, lr), pow(10.0, lT), &P, &u, &S, &cs, &phase);
            snprintf(label, sizeof(label), "lowest valid T: lookup P (lr=%g, lT=%g)", lr, lT); check(label, P, pow(10.0, irr_logP(lr, lT_eff)), 1e-10);
            snprintf(label, sizeof(label), "lowest valid T: lookup cs (lr=%g, lT=%g)", lr, lT); check(label, cs, pow(10.0, irr_logcs(lr, lT_eff)), 1e-10);
            aneos_compute(4, pow(10.0, lr), pow(10.0, irr_logu(lr, lT)), &T_guess, &P, &cs, &S, &cv, &grun, &phase);
            snprintf(label, sizeof(label), "lowest valid T: inversion T (lr=%g, lT=%g)", lr, lT); check(label, T_guess, pow(10.0, lT_eff), 1e-6);
            snprintf(label, sizeof(label), "lowest valid T: inversion P (lr=%g, lT=%g)", lr, lT); check(label, P, pow(10.0, irr_logP(lr, lT_eff)), 1e-5);
            snprintf(label, sizeof(label), "lowest valid T: inversion S (lr=%g, lT=%g)", lr, lT); check_abs(label, S, irr_S(lr, lT_eff), 1e-1);
        }
    }
    write_irregular_table_block(IRR_TABLE_FILE, 4, 7, 1);
    check("refuses a lowest valid row outside [0, nT-2]", (aneos_read_table(IRR_TABLE_FILE, 5) != 0) ? 1.0 : 0.0, 1.0, 0.0);
    remove(IRR_TABLE_FILE);
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
    test_irregular_grid();

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
