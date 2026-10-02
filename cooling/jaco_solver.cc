/*
 * jaco_solver.cc -- implicit (backward-Euler) solver for the jaco microphysics network.
 *
 * Undamped Newton with QR (GSL), clamping and fraction-to-the-boundary on the abundances; on
 * failure the step is subcycled by halving. Pure numerics: no GIZMO dependencies.
 */
#include <math.h>
#include <string.h>
#include "jaco_solver.h"
extern "C" {
#include <gsl/gsl_linalg.h>
}

#define JACO_TOL_DEFAULT 1e-6
#define JACO_MAXITER 150
#define JACO_HUGE 1e56

void jaco_solver_default_settings(struct JacoSolverSettings *set) {
    set->T_min = 1.0;
    set->T_max = 1e10;
    set->u_min = 0;
    set->tol = JACO_TOL_DEFAULT;
    set->fd_jacobian = 0;
    set->verbose = 0;
}

/* Apply physical bounds. Returns 1 if any variable was clamped, 0 otherwise. */
static int jaco_clamp(SolveVars *sv, const struct JacoSolverSettings *set) {
    int clamped = 0;
    double T_old = sv->T, u_old = sv->u;
    sv->T = fmax(set->T_min, fmin(set->T_max, sv->T));
    sv->u = fmax(set->u_min, fmin(1e17, sv->u));
    if (sv->T != T_old || sv->u != u_old) clamped = 1;
    for (int k = 2; k < N_VARS; k++) {
        double old = sv->data[k];
        sv->data[k] = fmax(JACO_ABUNDANCE_FLOOR, fmin(1.0, sv->data[k]));
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

/* Solve the chemistry rows at fixed T (identity rows for u and T) by Newton with a 10-step
   damping ramp, as the CIE table builder did. Returns 0 on success. */
int jaco_solve_chemistry(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, int *nfeval) {
    SolveVars func, dsv;
    double jac[N_VARS][N_VARS];
    (void)set;
    for (int iter = 0; iter < 100; iter++) {
        microphysics_func_jac(sv, pr, &func, jac);
        if (nfeval) (*nfeval)++;
        func.data[0] = 0;
        func.data[1] = 0;
        for (int j = 0; j < N_VARS; j++) {
            jac[0][j] = 0;
            jac[1][j] = 0;
        }
        jac[0][0] = 1;
        jac[1][1] = 1;
        double rhs[N_VARS];
        for (int i = 0; i < N_VARS; i++)
            rhs[i] = -func.data[i];
        qr_solve(jac, rhs, dsv.data);

        double fac = fmin(1.0, (double)(iter + 1) / 10.0);
        int converged = 1;
        for (int k = 0; k < N_VARS; k++) {
            sv->data[k] += fac * dsv.data[k];
            if (fabs(dsv.data[k]) > 1e-6 * (fabs(sv->data[k]) + 1e-6))
                converged = 0;
        }
        sv->T = fmax(10.0, fmin(1e10, sv->T));
        for (int k = 2; k < N_VARS; k++)
            sv->data[k] = fmax(JACO_ABUNDANCE_FLOOR, fmin(1.0, sv->data[k]));
        sv->u = jaco_T_to_u(sv->T, sv, pr, NULL);
        if (converged)
            return 0;
    }
    return 1;
}

/* Undamped Newton with fraction-to-the-boundary on the abundances. Returns the number of
   iterations, or -1 if JACO_MAXITER is reached or the linear solve returns NaN (singular). */
static int jaco_newton_loop(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, int verbose,
                            int *nfeval) {
    SolveVars func, dsv;
    double jac[N_VARS][N_VARS];
    const double tol = set->tol;

    for (int i = 0; i < N_VARS; i++)
        dsv.data[i] = JACO_HUGE;

    for (int iter = 0; iter < JACO_MAXITER; iter++) {
        if (!iter_condition(sv, &dsv, tol))
            return iter;

        microphysics_func_jac(sv, pr, &func, jac);
        if (nfeval) (*nfeval)++;

        double rhs[N_VARS];
        for (int i = 0; i < N_VARS; i++)
            rhs[i] = -func.data[i];
        qr_solve(jac, rhs, dsv.data);

        for (int k = 0; k < N_VARS; k++)
            if (!jaco_isfinite(dsv.data[k])) return -1;

        if (verbose) {
            printf("  iter=%d sv:", iter);
            for (int k = 0; k < N_VARS; k++) printf(" %.4e", sv->data[k]);
            printf("  func:");
            for (int k = 0; k < N_VARS; k++) printf(" %.4e", func.data[k]);
            printf("  dsv:");
            for (int k = 0; k < N_VARS; k++) printf(" %.4e", dsv.data[k]);
            printf("\n");
        }

        /* Fraction-to-the-boundary on the abundances: no species below its floor and the neutral
           H/He budgets non-negative. Species already at the floor are left to the clamp. */
        {
            const double tau = 0.99;
            double alpha = 1.0;
            for (int k = 2; k < N_VARS; k++) {
                if (sv->data[k] <= 2.0 * JACO_ABUNDANCE_FLOOR) continue;
                if (dsv.data[k] < 0) {
                    double max_step = -tau * sv->data[k] / dsv.data[k];
                    if (max_step < alpha) alpha = max_step;
                }
            }
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
            {
                double xH_now = 1.0 - sv->x_Hplus - 2.0 * sv->x_H_2;
                double dxH = -(dsv.x_Hplus + 2.0 * dsv.x_H_2);
                if (dxH < 0 && xH_now > 0) alpha = fmin(alpha, -tau * xH_now / dxH);
            }
#elif defined(JACO_MODEL_KWH)
            {
                double xH_now = 1.0 - sv->x_Hplus;
                double dxH = -dsv.x_Hplus;
                if (dxH < 0 && xH_now > 0) alpha = fmin(alpha, -tau * xH_now / dxH);
            }
#endif
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL) || defined(JACO_MODEL_KWH)
            {
                double xHe_now = pr->y - sv->x_Heplus - sv->x_Heplusplus;
                double dxHe = -(dsv.x_Heplus + dsv.x_Heplusplus);
                if (dxHe < 0 && xHe_now > 0) alpha = fmin(alpha, -tau * xHe_now / dxHe);
            }
#endif
            if (alpha < 1.0)
                for (int k = 2; k < N_VARS; k++) dsv.data[k] *= alpha;
        }

        for (int k = 0; k < N_VARS; k++)
            sv->data[k] += dsv.data[k];

        /* A clamp that truncated a step not already at the floor forces another iteration */
        {
            SolveVars sv_pre = *sv;
            if (jaco_clamp(sv, set)) {
                int nontrivial = (sv->T != sv_pre.T || sv->u != sv_pre.u);
                for (int k = 2; k < N_VARS && !nontrivial; k++)
                    if (sv->data[k] != sv_pre.data[k] && sv_pre.data[k] > 2.0 * JACO_ABUNDANCE_FLOOR) nontrivial = 1;
                if (nontrivial)
                    for (int k = 0; k < N_VARS; k++) dsv.data[k] = JACO_HUGE;
            }
        }
    }
    return -1;
}

int jaco_solve(SolveVars *sv, const Params *pr_in, const struct JacoSolverSettings *set, struct JacoSolveInfo *info) {
    struct JacoSolveInfo dummy;
    if (!info) info = &dummy;
    memset(info, 0, sizeof(*info));
    Params pr = *pr_in;
    const SolveVars sv0 = *sv;
    int nfeval = 0, nsub = 0, first = 1;

    /* Attempt the full step; on failure halve the substep and subcycle until the interval is covered */
    double dt_remaining = pr.Delta_t, dt_sub = dt_remaining;
    while (dt_remaining > 1e-30 * pr_in->Delta_t) {
        SolveVars sv_save = *sv;
        pr.Delta_t = dt_sub;
        pr.u_initial = sv->u;
#if defined(JACO_MODEL_STARFORGE) || defined(JACO_MODEL_PRIMORDIAL)
        pr.x_H_2_initial = sv->x_H_2;
#endif
        int ret = jaco_newton_loop(sv, &pr, set, set->verbose >= 2, &nfeval);
        if (first) {
            info->nfeval_tier1 = nfeval;
            info->tier1_status = ret >= 0 ? 0 : -4;
            first = 0;
        }
        if (ret >= 0) {
            dt_remaining -= dt_sub;
            dt_sub = fmin(dt_sub * 2.0, dt_remaining);
            nsub++;
            continue;
        }
        *sv = sv_save;
        dt_sub *= 0.5;
        if (set->verbose) printf("jaco_solve: subcycling dt_sub halved to %g (n=%g T=%g dt=%g)\n", dt_sub, pr_in->n_Htot, sv->T, pr_in->Delta_t);
        if (dt_sub < pr_in->Delta_t * 1e-10) {
            info->nfeval = nfeval;
            info->tier = JACO_TIER_FAILED;
            *sv = sv0;
            return 1;
        }
    }
    info->nfeval = nfeval;
    info->n_substeps = nsub;
    info->tier = (nsub == 1) ? JACO_TIER_NEWTON : JACO_TIER_SUBCYCLE;
    return 0;
}

void jaco_print_state(FILE *fp, const char *label, const SolveVars *sv, const Params *pr) {
    fprintf(fp, "%s\n  SolveVars (IDX_* order):", label);
    for (int k = 0; k < N_VARS; k++) fprintf(fp, " [%d]=%.17g", k, sv->data[k]);
    fprintf(fp, "\n  Params (PARAM_* order):");
    for (int k = 0; k < N_PARAMS; k++) fprintf(fp, " [%d]=%.17g", k, pr->data[k]);
    fprintf(fp, "\n");
}
