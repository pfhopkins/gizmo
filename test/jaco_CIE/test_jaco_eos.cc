/* Unit test for jaco EOS functions: T_to_u and u_to_T must be inverses,
   and T_to_u must return positive values for all positive T inputs. */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../cooling/microphysics_func_jac.h"

/* Declarations from jaco_eos.cc */
double jaco_T_to_u(double T, const SolveVars *sv, const Params *pr, double *cv_out);
double jaco_u_to_T(double u, const SolveVars *sv, const Params *pr);

/* Fill SolveVars/Params with physically reasonable values for a given ionization state */
void fill_test_state(SolveVars *sv, Params *pr, double T, double xHp, double xH2) {
    for (int i = 0; i < N_VARS; i++) sv->data[i] = 0;
    for (int i = 0; i < N_PARAMS; i++) pr->data[i] = 0;

    sv->T = T;
    sv->u = 0; /* will be set by T_to_u */
    sv->x_Hplus = xHp;
    sv->x_Heplus = 1e-20;
    sv->x_Heplusplus = 1e-20;

    pr->n_Htot = 100.0;       /* 100 cm^-3 */
    pr->y = 0.0994;           /* He/H number ratio for Y=0.27 */
#if defined(JACO_MODEL_STARFORGE)
    sv->x_H_2 = xH2;
    pr->x_C_tot = 2.1e-4;
    pr->x_O_tot = 4.9e-4;
    pr->x_N = 6.8e-5;
    pr->x_Ne = 8.5e-5;
    pr->x_Mg = 3.2e-5;
    pr->x_Si = 3.2e-5;
    pr->x_S = 1.3e-5;
    pr->x_Ca = 2.2e-6;
    pr->x_Fe = 2.5e-5;
#elif defined(JACO_MODEL_KWH)
    (void)xH2; /* unused for kwh — no H_2 species */
    pr->C_2 = 1.0;
#endif
}

int main() {
    int failures = 0;
    int tests = 0;
    double tol = 1e-6;

    /* Test grid: T from 1 K to 1e9 K, various ionization/molecular states */
    double T_values[] = {1.0, 2.73, 10.0, 100.0, 300.0, 1000.0, 3000.0, 8000.0,
                         1e4, 3e4, 1e5, 1e6, 1e7, 1e8, 1e9};
    int n_T = sizeof(T_values) / sizeof(T_values[0]);

    struct { double xHp; double xH2; const char *label; } states[] = {
        {1e-20, 1e-20, "neutral atomic"},
        {0.5,   1e-20, "half ionized"},
        {0.99,  1e-20, "fully ionized"},
        {1e-20, 0.49,  "fully molecular"},
        {0.3,   0.1,   "mixed"},
    };
    int n_states = sizeof(states) / sizeof(states[0]);

    printf("=== jaco_T_to_u positivity test ===\n");
    for (int s = 0; s < n_states; s++) {
        for (int t = 0; t < n_T; t++) {
            SolveVars sv;
            Params pr;
            double T = T_values[t];
            fill_test_state(&sv, &pr, T, states[s].xHp, states[s].xH2);

            double cv;
            double u = jaco_T_to_u(T, &sv, &pr, &cv);
            tests++;

            if (u <= 0 || !isfinite(u)) {
                printf("  FAIL: T=%.2e state='%s' -> u=%.6e (must be positive)\n",
                       T, states[s].label, u);
                failures++;
            }
            if (cv <= 0 || !isfinite(cv)) {
                printf("  FAIL: T=%.2e state='%s' -> cv=%.6e (must be positive)\n",
                       T, states[s].label, cv);
                failures++;
            }
        }
    }
    printf("Positivity: %d/%d passed\n\n", tests - failures, tests);

    printf("=== jaco_T_to_u / jaco_u_to_T inverse test ===\n");
    int inv_failures = 0;
    int inv_tests = 0;
    for (int s = 0; s < n_states; s++) {
        for (int t = 0; t < n_T; t++) {
            SolveVars sv;
            Params pr;
            double T = T_values[t];
            fill_test_state(&sv, &pr, T, states[s].xHp, states[s].xH2);

            double u = jaco_T_to_u(T, &sv, &pr, NULL);
            if (u <= 0 || !isfinite(u)) continue; /* skip broken cases */

            sv.u = u;
            double T_recovered = jaco_u_to_T(u, &sv, &pr);
            inv_tests++;

            double rel_err = fabs(T_recovered - T) / T;
            if (rel_err > tol) {
                printf("  FAIL: T=%.2e state='%s' u=%.6e -> T_recovered=%.6e rel_err=%.2e\n",
                       T, states[s].label, u, T_recovered, rel_err);
                inv_failures++;
            }
        }
    }
    printf("Inverse: %d/%d passed (tol=%.0e)\n\n", inv_tests - inv_failures, inv_tests, tol);

    int total_failures = failures + inv_failures;
    if (total_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TOTAL FAILURES\n", total_failures);
    }
    return total_failures > 0 ? 1 : 0;
}
