/* Model-agnostic test of the jaco implicit solver (cooling/jaco_solver.cc), for models outside the STARFORGE family
 * (test_jaco_solver.cc drives those in depth). The solver reads everything model-specific from the generated header,
 * so it must run on any model: sweep n_Htot x T0 x dt x pdv_work with the species seeded at a tenth of their ceiling
 * (within the budgets), and check every answer independently: finite, u = u(T,x), abundances and budgets in bounds,
 * the residual of the generated system small, the outputs finite. Exits non-zero on any failure.
 *
 *   make -C test/jaco_solver MODEL=wind_comparison
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "jaco_solver.h"

void jaco_init_tables(const char *dir);

static const double M_PER_H = 2.34e-24, X_ATOL = 1e-14, CHECK_TOL = 1e-5;
static const int budget_n = JACO_N_BUDGETS;
struct Budget { double total; int total_param; int nterm; int k[JACO_BUDGET_MAX_TERMS]; double w[JACO_BUDGET_MAX_TERMS]; };
static const struct Budget budgets[] = JACO_BUDGETS_INIT;
static const double var_ceiling[N_VARS] = JACO_VAR_CEILING_INIT, var_floor[N_VARS] = JACO_VAR_FLOOR_INIT;

static void params(Params *pr, double n, double dt) {
    memset(pr, 0, sizeof(*pr));
    pr->n_Htot = n;
    pr->Delta_t = dt;
#ifdef JACO_HAS_PARAM_x_H
    pr->x_H = 1.0;
#endif
#ifdef JACO_HAS_PARAM_y
    pr->y = 0.0994;
#endif
}

static double budget_value(const struct Budget *b, const SolveVars *sv, const Params *pr) {
    double v = b->total_param < 0 ? b->total : pr->data[b->total_param];
    for (int t = 0; t < b->nterm; t++) v -= b->w[t] * sv->data[b->k[t]];
    return v;
}

static int check(const SolveVars *sv, const Params *pr, const JacoSolverSettings *set, char *why) {
    SolveVars F;
    double J[N_VARS][N_VARS];
    microphysics_func_jac(sv, pr, &F, J);
    for (int i = 0; i < N_VARS; i++)
        if (!isfinite(F.data[i]) || !isfinite(sv->data[i])) return sprintf(why, "non-finite state or residual [%d]", i);
    double u_eos = jaco_T_to_u(sv->T, sv, pr, NULL);
    if (fabs(sv->u - u_eos) > 1e-12 * u_eos) return sprintf(why, "u=%g != u(T,x)=%g", sv->u, u_eos);
    for (int k = 2; k < N_VARS; k++)
        if (sv->data[k] < var_floor[k] || sv->data[k] > var_ceiling[k]) return sprintf(why, "abundance [%d]=%g out of bounds", k, sv->data[k]);
    for (int b = 0; b < budget_n; b++)
        if (budget_value(&budgets[b], sv, pr) < 0) return sprintf(why, "budget %d negative", b);
    if (sv->T < set->T_min || sv->T > set->T_max || sv->u < set->u_min) return sprintf(why, "outside the floor/ceiling T=%g", sv->T);
    double sT = M_PER_H * pr->n_Htot * sv->u / pr->Delta_t + fabs(J[IDX_T][IDX_T]) * sv->T, r = F.T / (CHECK_TOL * sT);
    int at_floor = sv->T <= set->T_min * (1 + 1e-4) || sv->u <= set->u_min * (1 + 1e-4), at_ceiling = sv->T >= set->T_max * (1 - 1e-9);
    if ((at_floor ? r : at_ceiling ? -r : fabs(r)) > 1) return sprintf(why, "energy residual %.3g x tolerance", r);
    for (int k = 2; k < N_VARS; k++) {
        double rk = F.data[k] / (CHECK_TOL * (fabs(J[k][k]) * (sv->data[k] + X_ATOL) + 1e-300));
        if ((sv->data[k] <= var_floor[k] * (1 + 1e-9) ? rk : fabs(rk)) > 1) return sprintf(why, "species [%d] residual %.3g x tolerance", k, rk);
    }
    Outputs out;
    microphysics_outputs(sv, pr, &out);
    for (int k = 0; k < N_OUTPUTS; k++)
        if (!isfinite(out.data[k])) return sprintf(why, "non-finite output [%d]", k);
    return 0;
}

int main(int argc, char **argv) {
    jaco_init_tables(argc > 1 ? argv[1] : ".");
    JacoSolverSettings set;
    jaco_solver_default_settings(&set);
    set.T_min = 10;
    const double ns[] = {1e-2, 1, 1e2, 1e4, 1e6}, dts[] = {1e7, 1e11, 1e15, 1e20}, pdvs[] = {0, 1e-24, -1e-26};
    long n = 0, nfail = 0, tiers[5] = {0, 0, 0, 0, 0};
    for (double nH : ns)
        for (double dt : dts)
            for (double pdv : pdvs)
                for (int i = 0; i <= 14; i++) {
                    double T0 = pow(10., 1 + 0.5 * i);
                    Params pr;
                    params(&pr, nH, dt);
                    pr.pdv_work = pdv * nH;
                    SolveVars sv = {};
                    sv.T = T0;
                    for (int k = 2; k < N_VARS; k++) sv.data[k] = 0.1 * fmin(1, var_ceiling[k]);
                    for (int b = 0; b < budget_n; b++) /* scale into the budgets */
                        if (budget_value(&budgets[b], &sv, &pr) < 0)
                            for (int t = 0; t < budgets[b].nterm; t++) sv.data[budgets[b].k[t]] *= 0.1;
                    jaco_initial_from_state(&sv, &pr);
                    sv.u = pr.u_initial = jaco_T_to_u(T0, &sv, &pr, NULL);
                    SolveVars in = sv;
                    JacoSolveInfo info;
                    char why[256] = "";
                    int rc = jaco_solve(&sv, &pr, &set, &info);
                    n++;
                    tiers[info.tier]++;
                    if (rc || check(&sv, &pr, &set, why)) {
                        nfail++;
                        printf("FAIL n=%g T0=%g dt=%g pdv=%g: %s\n", nH, T0, dt, pdv, rc ? "every tier failed" : why);
                        jaco_print_state(stdout, "  input state:", &in, &pr);
                    }
                }
    printf("%ld cases: tier1 %ld tier2 %ld tier3 %ld FAILED %ld; %ld check failures (N_VARS %d, %d time-dependent species, %d budgets, "
           "%d outputs)\n%s\n", n, tiers[1], tiers[2], tiers[3], tiers[4], nfail, N_VARS, JACO_N_TD_SPECIES, JACO_N_BUDGETS, N_OUTPUTS,
           nfail ? "FAILED" : "PASSED");
    return nfail ? 1 : 0;
}
