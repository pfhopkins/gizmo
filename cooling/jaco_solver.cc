/*
 * jaco_solver.cc -- tiered implicit (backward-Euler) solver for the jaco microphysics network.
 *
 * The generated system F(v) = 0 on v = (u, T, x_2 .. x_{N-1}) is
 *   F_0 = u(T,x) - u                                       EOS; u enters no other row
 *   F_1 = heat(T,x) + pdv_work - rho (u(T,x) - u_initial)/Delta_t      [erg cm^-3 s^-1]
 *   F_k = net production of species k: steady state, or with a backward-Euler term for the
 *         time-dependent species                                         [cm^-3 s^-1]
 * u is a slave of the EOS: every iterate sets u = u(T,x), so F_0 = 0 and Newton works on (T, x)
 * (u is eliminated by a Schur complement, which is exact for any model of this form).
 *
 * Tiers, cheapest first; each runs only if the previous one failed:
 *   0  consistent start: if u(T0,x0) disagrees with u_initial, T0 = u^-1(u_initial, x0).
 *   1  Newton polish: few iterations, floored species with no tendency to grow are pinned and
 *      dropped from the linear system, fraction-to-the-boundary on T and the neutral budgets,
 *      backtracking on Deuflhard's natural monotonicity test. Fails fast on a singular or
 *      non-finite system.
 *   2  1-D rootfind in T: chemistry solved at fixed T, R(T) = energy residual; geometric
 *      bracketing outward from T0 then Brent in ln T, then a short Newton polish. If R keeps its
 *      sign down to the floor (or up to the ceiling) the answer is the floor (ceiling).
 *   3  subcycling: halve the substep until tiers 1-2 succeed on each piece.
 * A non-finite residual rejects that step or trial point. A non-finite Jacobian column (the
 * generated derivative of a rate that underflowed to zero can be 0/0 although the rate itself is
 * fine) is replaced by finite differences of F if settings.fd_jacobian, else it fails the tier.
 * Both are counted in JacoSolveInfo so model defects stay visible.
 *
 * What the solver knows about the model beyond F and its Jacobian (which species are time-dependent
 * and the parameters holding their start-of-step values, abundance floors, ceilings and scales,
 * charges, and the budgets that bound the eliminated abundances) comes from the generated header.
 */
#include <math.h>
#include <string.h>
#include "jaco_solver.h"

/* ---- tolerances and limits ---- */
#define JACO_TOL_DEFAULT 1e-6        /* relative; Newton converges quadratically, so tight costs ~1 extra iteration */
#define JACO_X_ATOL 1e-14            /* absolute abundance tolerance: far below anything that affects heating or the EOS */
#define JACO_NORM_XMIN 1e-6          /* abundance scale in the step-acceptance norm: trace species must not dominate it */
#define JACO_EOS_START_TOL 1e-3      /* tier 0: re-derive T0 when |u(T0,x0) - u_initial| exceeds this fraction of u_initial */
#define JACO_MASS_PER_H 2.34e-24     /* ~1.4 m_p: mass per H nucleus, only used to scale the energy residual */
#define JACO_TIER1_MAXITER 8         /* tier-1 Newton steps; a consistent start needs 1-3 */
#define JACO_MAX_BACKTRACK 3         /* line search tries alpha = 1, 1/2, 1/4, 1/8 of the bounded step */
#define JACO_TAU_T 0.9               /* T may fall by at most this fraction per step (a factor 10) */
#define JACO_TAU_BUDGET 0.99         /* a neutral H/He budget may shrink by at most this fraction per step */
#define JACO_PIVOT_TOL 1e-13         /* LU pivot below this fraction of its column: singular to working precision */
#define JACO_FD_REL 1e-7             /* finite-difference step, ~sqrt(machine epsilon) */
#define JACO_FD_XMIN 1e-10           /* smallest abundance scale for a finite-difference step */
#define JACO_CHEM_MAXITER 25         /* fixed-T chemistry Newton steps (tier 2 starts each solve warm) */
#define JACO_PTC_MAXITER 6           /* Newton steps per pseudo-transient stage */
#define JACO_PTC_MAXSTAGES 60        /* pseudo-transient stages before the fixed-T chemistry gives up */
#define JACO_PTC_GROW 10.0           /* pseudo-timestep growth per successful stage */
#define JACO_BRACKET_FAC0 1.1        /* tier 2: first bracketing factor in T, as in DoCooling */
#define JACO_BRACKET_GROW 1.5        /* tier 2: growth of that factor per step (4 decades in ~8 steps) */
#define JACO_BRACKET_MAXITER 60
#define JACO_BRENT_MAXITER 100
#define JACO_POLISH_MAXITER 8        /* tier 2: Newton steps from the rootfind answer, whose nested time-dependent species are resolved to tol of themselves, not of their budgets */
#define JACO_VERIFY_FAC 10.0         /* tier-2 acceptance slack: T is bracketed to tol, the energy row then moves with the total dR/dT */
#define JACO_SUBCYCLE_MIN_FRAC 1e-6  /* tier 3 gives up below this fraction of Delta_t */
#define JACO_SUBCYCLE_MAXSTEPS 2000

void jaco_solver_default_settings(struct JacoSolverSettings *set) {
    set->T_min = 1.0;
    set->T_max = 1e10;
    set->u_min = 0;
    set->tol = JACO_TOL_DEFAULT;
    set->fd_jacobian = 0; /* a model that returns a non-finite Jacobian is a bug to fix, not to absorb */
    set->verbose = 0;
}

struct Counters {
    int nfeval;
    int nfeval_fd;
    int n_nonfinite_jac;
    int n_nonfinite_F;
    int fd_jacobian;
};

struct Eval {
    SolveVars F;
    double J[N_VARS][N_VARS];
};

/* pseudo-transient term -n_Htot (x_k - anchor_k) inv_tau added to every species row */
struct PTC {
    double inv_tau;
    SolveVars anchor;
};

enum { NEWTON_OK = 0, NEWTON_NONFINITE = -1, NEWTON_SINGULAR = -2, NEWTON_LINESEARCH = -3, NEWTON_MAXITER = -4 };

/* outcome of a tier or of the tier-2 search; PINNED: the answer sits at the temperature/energy floor or ceiling */
enum Outcome { OUTCOME_FAILED, OUTCOME_SOLVED, OUTCOME_PINNED };

/* which variables a Newton solve may move: T, steady-state species, time-dependent species */
enum { SOLVE_T = 1, SOLVE_IONS = 2, SOLVE_TD = 4, SOLVE_ALL = 7 };

#ifndef JACO_HAS_SOLVER_METADATA
#error "microphysics_func_jac.h carries no solver metadata: regenerate it with a jaco that emits it"
#endif

/* layout the solver relies on: v = (u, T, species...), and the unions' data[] alias their named fields */
static_assert(IDX_u == 0 && IDX_T == 1, "the solver assumes v = (u, T, species...)");
static_assert(sizeof(SolveVars) == N_VARS * sizeof(double) && sizeof(Params) == N_PARAMS * sizeof(double),
              "SolveVars and Params must be exactly their data[] arrays");

/* ---- model metadata, from the generated header ----
   Time-dependent species carry a backward-Euler term towards their start-of-step value pr->data[param]; all other
   species are steady state. Abundances live in [var_floor, var_ceiling]; absolute abundance tolerances are fractions of
   var_scale. Conserved budgets bound the eliminated abundances: total - sum_t w[t] x[k[t]] >= 0, the total a constant
   or a parameter (e.g. x_H0 = 1 - x_H+ - 2 x_H2). The step bound, the projection, the finite-difference direction, the
   start clamps, the ionized seed and the time-dependent species' caps all read budget_table through budget_value or
   budget_cap, so they agree on a budget to the last bit. */
static constexpr int var_time_dependent[N_VARS] = JACO_VAR_TIME_DEPENDENT_INIT;
static constexpr double var_floor[N_VARS] = JACO_VAR_FLOOR_INIT;
static constexpr double var_ceiling[N_VARS] = JACO_VAR_CEILING_INIT;
static constexpr double var_scale[N_VARS] = JACO_VAR_SCALE_INIT;
static constexpr int var_charge[N_VARS] = JACO_VAR_CHARGE_INIT;
static constexpr int var_initial_param[N_VARS] = JACO_VAR_INITIAL_PARAM_INIT;

static int is_time_dependent(int k) { return var_time_dependent[k]; }
/* at or below this a species counts as floored */
static double floor_pin(int k) { return var_floor[k] * (1 + 1e-9); }

struct TDSpecies {
    int k;     /* IDX_ of the species */
    int param; /* PARAM_ of its start-of-step value */
};
static constexpr struct TDSpecies td_species[] = JACO_TD_SPECIES_INIT;
#define N_TD JACO_N_TD_SPECIES

struct Budget {
    double total;    /* the total if total_param < 0 */
    int total_param; /* else the PARAM_ index holding it */
    int nterm;
    int k[JACO_BUDGET_MAX_TERMS];    /* IDX_ of the species */
    double w[JACO_BUDGET_MAX_TERMS]; /* their weights */
};
static constexpr struct Budget budget_table[] = JACO_BUDGETS_INIT;
#define N_BUDGETS JACO_N_BUDGETS

static constexpr bool metadata_valid() {
    if (!var_time_dependent[IDX_T] || var_initial_param[IDX_T] != PARAM_u_initial || var_time_dependent[IDX_u]) return false;
    int ntd = 0;
    for (int k = 2; k < N_VARS; k++) {
        if (!(var_floor[k] >= 0 && var_ceiling[k] > var_floor[k] && var_scale[k] > 0)) return false;
        if (var_time_dependent[k]) {
            if (ntd >= N_TD || td_species[ntd].k != k || td_species[ntd].param != var_initial_param[k]) return false;
            if (td_species[ntd].param < 0 || td_species[ntd].param >= N_PARAMS) return false;
            ntd++;
        } else if (var_initial_param[k] >= 0) return false;
    }
    return ntd == N_TD;
}
static_assert(metadata_valid(), "solver metadata: T must be time-dependent with u_initial, species start at index 2, "
                                "and the time-dependent species table must match the per-variable flags");

static constexpr bool budget_table_valid() {
    for (int b = 0; b < N_BUDGETS; b++) {
        const struct Budget &B = budget_table[b];
        if (B.total_param >= N_PARAMS || B.nterm < 1 || B.nterm > JACO_BUDGET_MAX_TERMS) return false;
        for (int t = 0; t < B.nterm; t++)
            if (B.k[t] < 2 || B.k[t] >= N_VARS || !(B.w[t] > 0)) return false;
    }
    return true;
}
static_assert(budget_table_valid(), "budget table: totals must be parameters or constants, terms species with positive weights");

static double budget_total(const struct Budget *b, const Params *pr) { return b->total_param < 0 ? b->total : pr->data[b->total_param]; }

/* The budget's value. Not inlined: under -ffast-math, inlined copies of one expression may round
   differently, and the step bound and the projection must agree on its sign. */
__attribute__((noinline)) static double budget_value(const struct Budget *b, const SolveVars *sv, const Params *pr) {
    double v = budget_total(b, pr);
    for (int t = 0; t < b->nterm; t++) v -= b->w[t] * sv->data[b->k[t]];
    return v;
}

/* weight of species j in the budget (0 if absent) */
static double budget_weight(const struct Budget *b, int j) {
    double w = 0;
    for (int t = 0; t < b->nterm; t++)
        if (b->k[t] == j) w += b->w[t];
    return w;
}

/* Largest x_j every budget containing it allows, the other species fixed. */
static double budget_cap(const SolveVars *sv, const Params *pr, int j) {
    double cap = HUGE_VAL;
    for (int b = 0; b < N_BUDGETS; b++) {
        const struct Budget *B = &budget_table[b];
        double wj = budget_weight(B, j), rest = budget_total(B, pr);
        if (wj == 0) continue;
        for (int t = 0; t < B->nterm; t++)
            if (B->k[t] != j) rest -= B->w[t] * sv->data[B->k[t]];
        cap = fmin(cap, rest / wj);
    }
    return cap;
}

/* Trim the larger contributor until every budget is non-negative as computed in floating point (a fully
   ionized budget sits within an ulp of zero). */
static void trim_budgets(SolveVars *sv, const Params *pr) {
    for (int b = 0; b < N_BUDGETS; b++) {
        const struct Budget *B = &budget_table[b];
        while (budget_value(B, sv, pr) < 0) {
            int t = 0;
            for (int s2 = 1; s2 < B->nterm; s2++)
                if (B->w[s2] * sv->data[B->k[s2]] > B->w[t] * sv->data[B->k[t]]) t = s2;
            double rest = budget_total(B, pr);
            for (int s2 = 0; s2 < B->nterm; s2++)
                if (s2 != t) rest -= B->w[s2] * sv->data[B->k[s2]];
            sv->data[B->k[t]] = nextafter(fmin(sv->data[B->k[t]], rest / B->w[t]), 0);
        }
    }
}

static const char *newton_status(int s) {
    switch (s) {
    case NEWTON_OK: return "converged";
    case NEWTON_NONFINITE: return "non-finite F";
    case NEWTON_SINGULAR: return "singular";
    case NEWTON_LINESEARCH: return "line search";
    case NEWTON_MAXITER: return "max iterations";
    }
    return "?";
}

/* Clamp the starting state (and the time-dependent species' initial values) into the physical
   region: T in [T_min, T_max], abundances in [floor, ceiling], budgets positive with a margin. In each budget the
   steady-state species are scaled into it (a single one is clamped), then the time-dependent ones take what remains;
   a start-of-step value is clamped to what its budgets allow on their own. */
static void sanitize(SolveVars *sv, Params *pr, const struct JacoSolverSettings *set) {
    if (!jaco_isfinite(sv->T)) sv->T = set->T_min;
    sv->T = fmax(set->T_min, fmin(set->T_max, sv->T));
    for (int k = 2; k < N_VARS; k++) {
        if (!jaco_isfinite(sv->data[k]) || sv->data[k] < var_floor[k]) sv->data[k] = var_floor[k];
        if (sv->data[k] > var_ceiling[k]) sv->data[k] = var_ceiling[k];
    }
    const double margin = 1e-10;
    for (int b = 0; b < N_BUDGETS; b++) {
        const struct Budget *B = &budget_table[b];
        double cap = budget_total(B, pr) * (1 - margin), ss = 0;
        int nss = 0;
        for (int t = 0; t < B->nterm; t++)
            if (!is_time_dependent(B->k[t])) {
                ss += B->w[t] * sv->data[B->k[t]];
                nss++;
            }
        for (int t = 0; t < B->nterm; t++) {
            int k = B->k[t];
            if (is_time_dependent(k)) continue;
            if (nss == 1)
                sv->data[k] = fmin(sv->data[k], cap / B->w[t]);
            else if (ss > cap)
                sv->data[k] = fmax(var_floor[k], sv->data[k] * cap / ss);
        }
        for (int t = 0; t < B->nterm; t++) {
            int k = B->k[t];
            if (!is_time_dependent(k)) continue;
            double rest = cap;
            for (int t2 = 0; t2 < B->nterm; t2++)
                if (t2 != t) rest -= B->w[t2] * sv->data[B->k[t2]];
            sv->data[k] = fmax(var_floor[k], fmin(sv->data[k], rest / B->w[t]));
        }
    }
    for (int t = 0; t < N_TD; t++) {
        int k = td_species[t].k;
        double cap = var_ceiling[k];
        for (int b = 0; b < N_BUDGETS; b++) {
            double w = budget_weight(&budget_table[b], k);
            if (w > 0) cap = fmin(cap, budget_total(&budget_table[b], pr) / w);
        }
        pr->data[td_species[t].param] = fmax(var_floor[k], fmin(pr->data[td_species[t].param], cap));
    }
    /* start-of-step values that overfill a budget together are scaled into it */
    for (int b = 0; b < N_BUDGETS; b++) {
        const struct Budget *B = &budget_table[b];
        double sum = 0, cap = budget_total(B, pr) * (1 - margin);
        int ntd = 0;
        for (int t = 0; t < B->nterm; t++)
            if (is_time_dependent(B->k[t])) {
                sum += B->w[t] * pr->data[var_initial_param[B->k[t]]];
                ntd++;
            }
        if (ntd < 2 || sum <= cap) continue;
        for (int t = 0; t < B->nterm; t++)
            if (is_time_dependent(B->k[t])) {
                double *x0 = &pr->data[var_initial_param[B->k[t]]];
                *x0 = fmax(var_floor[B->k[t]], *x0 * cap / sum);
            }
    }
}

/* Replace the steady-state abundances in each budget by a nearly fully ionized state (time-dependent species kept):
   what the time-dependent species leave, 99.9% of it, nearly all on the most charged species. From above, Newton
   descends to the physical ionization balance; from below it can be drawn to the spurious x_e -> 0 root of collisional
   ionization, which is proportional to n_e. Returns 0 if no budget has a steady-state species. */
static int ionized_seed(SolveVars *sv, const Params *pr) {
    int seeded = 0;
    for (int b = 0; b < N_BUDGETS; b++) {
        const struct Budget *B = &budget_table[b];
        double rest = budget_total(B, pr);
        int nss = 0, top = -1;
        for (int t = 0; t < B->nterm; t++) {
            int k = B->k[t];
            if (is_time_dependent(k)) {
                rest -= B->w[t] * sv->data[k];
            } else {
                nss++;
                if (top < 0 || var_charge[k] > var_charge[B->k[top]]) top = t;
            }
        }
        for (int t = 0; t < B->nterm; t++) {
            if (is_time_dependent(B->k[t])) continue;
            double frac = (t == top) ? (nss == 1 ? 0.999 : 0.998) : 1e-3 / (nss - 1);
            sv->data[B->k[t]] = frac * rest / B->w[t];
        }
        seeded |= nss > 0;
    }
    return seeded;
}

/* ---- evaluation ---- */

/* Set u = u(T,x) and evaluate F and J. Non-finite Jacobian columns are replaced by one-sided
   finite differences of F (u held fixed, so they are the same partials the generated code
   returns) if c->fd_jacobian. Returns 0 if F and J are finite. */
static int evaluate(SolveVars *sv, const Params *pr, struct Eval *e, struct Counters *c) {
    sv->u = jaco_T_to_u(sv->T, sv, pr, NULL);
    microphysics_func_jac(sv, pr, &e->F, e->J);
    c->nfeval++;
    for (int i = 0; i < N_VARS; i++)
        if (!jaco_isfinite(e->F.data[i])) {
            c->n_nonfinite_F++;
            return -1;
        }
    int counted = 0;
    for (int j = 0; j < N_VARS; j++) {
        int bad = 0;
        for (int i = 0; i < N_VARS; i++)
            if (!jaco_isfinite(e->J[i][j])) bad = 1;
        if (!bad) continue;
        if (!counted) c->n_nonfinite_jac++;
        counted = 1;
        if (j == IDX_u || !c->fd_jacobian) return -1;
        double h;
        if (j == IDX_T) {
            h = -JACO_FD_REL * sv->T; /* backward keeps T > 0 */
        } else {
            h = JACO_FD_REL * fmax(sv->data[j], JACO_FD_XMIN * var_scale[j]);
            for (int b = 0; b < N_BUDGETS; b++)
                if (budget_value(&budget_table[b], sv, pr) - budget_weight(&budget_table[b], j) * h < 0) h = -fmin(h, 0.5 * sv->data[j]);
        }
        SolveVars sp = *sv;
        sp.data[j] += h;
        SolveVars Fp;
        double Jp[N_VARS][N_VARS];
        microphysics_func_jac(&sp, pr, &Fp, Jp);
        c->nfeval++;
        c->nfeval_fd++;
        for (int i = 0; i < N_VARS; i++) {
            e->J[i][j] = (Fp.data[i] - e->F.data[i]) / h;
            if (!jaco_isfinite(e->J[i][j])) return -1;
        }
    }
    return 0;
}

static void add_ptc(const SolveVars *sv, const Params *pr, const struct PTC *ptc, struct Eval *e) {
    for (int k = 2; k < N_VARS; k++) {
        e->F.data[k] -= pr->n_Htot * (sv->data[k] - ptc->anchor.data[k]) * ptc->inv_tau;
        e->J[k][k] -= pr->n_Htot * ptc->inv_tau;
    }
}

/* ---- dense linear solve ---- */

/* LU factors of a reduced Jacobian, kept so that the simplified Newton correction at a trial point
   reuses them. Fixed-size: no allocation in the per-cell hot path. */
struct LU {
    int n;
    double M[N_VARS][N_VARS]; /* L (unit diagonal, below) and U of the scaled, row-permuted matrix */
    double rs[N_VARS];        /* row equilibration factors, in original row order */
    double cs[N_VARS];        /* column scales */
    int perm[N_VARS];         /* perm[k] = original row in position k */
};

/* Factor A diag(cs) after equilibrating its rows, with partial pivoting. The rows mix
   erg cm^-3 s^-1 with cm^-3 s^-1 and the columns mix K with abundances near 1e-20, so the
   singularity test compares each pivot with the largest entry of its column in the equilibrated
   matrix, which is invariant to both row and column scaling. Returns -1 if a pivot falls below
   JACO_PIVOT_TOL by that measure (singular to working precision) or anything is non-finite. */
static int lu_factor(struct LU *lu, int n, double A[N_VARS][N_VARS], const double *cs) {
    double colmax[N_VARS];
    lu->n = n;
    for (int i = 0; i < n; i++) {
        double rmax = 0;
        lu->cs[i] = cs[i];
        lu->perm[i] = i;
        for (int j = 0; j < n; j++) {
            lu->M[i][j] = A[i][j] * cs[j];
            if (!jaco_isfinite(lu->M[i][j])) return -1;
            rmax = fmax(rmax, fabs(lu->M[i][j]));
        }
        if (!(rmax > 0)) return -1;
        lu->rs[i] = 1 / rmax;
        for (int j = 0; j < n; j++) lu->M[i][j] *= lu->rs[i];
    }
    for (int j = 0; j < n; j++) {
        colmax[j] = 0;
        for (int i = 0; i < n; i++) colmax[j] = fmax(colmax[j], fabs(lu->M[i][j]));
    }
    for (int k = 0; k < n; k++) {
        int p = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(lu->M[i][k]) > fabs(lu->M[p][k])) p = i;
        if (!(fabs(lu->M[p][k]) >= JACO_PIVOT_TOL * colmax[k])) return -1;
        if (p != k) {
            for (int j = 0; j < n; j++) {
                double t = lu->M[k][j];
                lu->M[k][j] = lu->M[p][j];
                lu->M[p][j] = t;
            }
            int t = lu->perm[k];
            lu->perm[k] = lu->perm[p];
            lu->perm[p] = t;
        }
        for (int i = k + 1; i < n; i++) {
            double f = lu->M[i][k] /= lu->M[k][k];
            for (int j = k + 1; j < n; j++) lu->M[i][j] -= f * lu->M[k][j];
        }
    }
    return 0;
}

/* d = A^-1 b with the factors; returns -1 if the result is non-finite */
static int lu_solve(const struct LU *lu, const double *b, double *d) {
    int n = lu->n;
    double y[N_VARS];
    for (int k = 0; k < n; k++) {
        double s = b[lu->perm[k]] * lu->rs[lu->perm[k]];
        for (int j = 0; j < k; j++) s -= lu->M[k][j] * y[j];
        y[k] = s;
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = y[i];
        for (int j = i + 1; j < n; j++) s -= lu->M[i][j] * y[j];
        y[i] = s / lu->M[i][i];
    }
    for (int i = 0; i < n; i++) {
        d[i] = y[i] * lu->cs[i];
        if (!jaco_isfinite(d[i])) return -1;
    }
    return 0;
}

/* ---- damped Newton with an active set ---- */

/* Clamp into bounds. The step bound keeps the budgets positive in exact arithmetic, trim_budgets
   also as computed. */
static void project(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set) {
    sv->T = fmax(set->T_min, fmin(set->T_max, sv->T));
    for (int k = 2; k < N_VARS; k++)
        if (sv->data[k] < var_floor[k]) sv->data[k] = var_floor[k];
    trim_budgets(sv, pr);
}

static void trace(const char *tag, int it, const SolveVars *sv, const struct Eval *e) {
    printf("    %s it=%d v:", tag, it);
    for (int k = 0; k < N_VARS; k++) printf(" %.6e", sv->data[k]);
    printf("  F:");
    for (int k = 0; k < N_VARS; k++) printf(" %.3e", e->F.data[k]);
    printf("\n");
}

/* trial = sv + alpha d on the variables idx, projected into bounds */
static void take_step(SolveVars *trial, const SolveVars *sv, const int *idx, int n, const double *d, double alpha,
                      const Params *pr, const struct JacoSolverSettings *set) {
    *trial = *sv;
    for (int a = 0; a < n; a++) trial->data[idx[a]] += alpha * d[a];
    project(trial, pr, set);
}

/* right-hand side -F of the reduced system on rows idx, with u eliminated (Schur complement) */
static void reduced_rhs(const struct Eval *e, const int *idx, int n, double *b) {
    for (int a = 0; a < n; a++) {
        int i = idx[a];
        b[a] = -e->F.data[i] + e->J[i][IDX_u] * e->F.data[IDX_u] / e->J[IDX_u][IDX_u];
    }
}

/* Residual scale of row i: energy rho u/dt + |dF_T/dT| T (a relative energy error, or the T
   change that would remove the residual when cooling is stiff); species |dF_k/dx_k| (x_k + atol). */
static double row_scale(const SolveVars *sv, const Params *pr, const struct Eval *e, int i) {
    if (i == IDX_T) return JACO_MASS_PER_H * pr->n_Htot * sv->u / pr->Delta_t + fabs(e->J[IDX_T][IDX_T]) * sv->T;
    return fabs(e->J[i][i]) * (sv->data[i] + JACO_X_ATOL * var_scale[i]) + 1e-300;
}

static double scaled_norm(const double *d, const double *cs, int n) {
    double s = 0;
    for (int a = 0; a < n; a++) s += (d[a] / cs[a]) * (d[a] / cs[a]);
    return sqrt(s);
}

/* Newton on the variables selected by mask (SOLVE_*). A floored species is pinned (row and
   column dropped) unless its net production is positive.
   Step acceptance is Deuflhard's natural monotonicity test: the simplified Newton correction at
   the trial point, -J^-1 F(trial) with the current factors, must be smaller (in variables scaled
   by their magnitudes, abundances floored at JACO_NORM_XMIN) than the Newton correction by
   1 - alpha/4; else alpha is halved. Being
   invariant to row scaling, it does not reject good steps because a fast, tiny species' residual
   jumped in absolute terms, which a raw residual-norm test does.
   Converged when, over the free variables, the step is below tol (relative, plus JACO_X_ATOL for
   abundances) AND every residual is below tol times its row_scale. The returned state is the last
   evaluated one, so its residual is the one tested. On failure sv holds the last accepted iterate. */
static int newton(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, int mask, int maxiter,
                  const struct PTC *ptc, struct Counters *c, SolveVars *F_out) {
    struct Eval e, et;
    struct LU lu;
    if (evaluate(sv, pr, &e, c)) return NEWTON_NONFINITE;
    if (ptc) add_ptc(sv, pr, ptc, &e);
    const double tol = set->tol;
    for (int it = 0;; it++) {
        if (set->verbose >= 2) trace((mask & SOLVE_T) ? "newton" : (mask & SOLVE_TD) ? "chem" : "ions", it, sv, &e);
        int idx[N_VARS], n = 0;
        if (mask & SOLVE_T) idx[n++] = IDX_T;
        for (int k = 2; k < N_VARS; k++)
            if ((mask & (is_time_dependent(k) ? SOLVE_TD : SOLVE_IONS)) && (sv->data[k] > floor_pin(k) || e.F.data[k] > 0))
                idx[n++] = k;
        if (n == 0) {
            if (F_out) *F_out = e.F;
            return NEWTON_OK;
        }

        double cs[N_VARS], w[N_VARS], b[N_VARS], d[N_VARS], A[N_VARS][N_VARS];
        int res_ok = 1;
        for (int a = 0; a < n; a++) {
            int i = idx[a];
            cs[a] = (i == IDX_T) ? sv->T : sv->data[i] + JACO_X_ATOL * var_scale[i];
            w[a] = (i == IDX_T) ? sv->T : sv->data[i] + JACO_NORM_XMIN * var_scale[i];
            if (fabs(e.F.data[i]) > tol * row_scale(sv, pr, &e, i)) res_ok = 0;
            for (int a2 = 0; a2 < n; a2++)
                A[a][a2] = e.J[i][idx[a2]] - e.J[i][IDX_u] * e.J[IDX_u][idx[a2]] / e.J[IDX_u][IDX_u];
        }
        reduced_rhs(&e, idx, n, b);
        if (lu_factor(&lu, n, A, cs) || lu_solve(&lu, b, d)) return NEWTON_SINGULAR;

        /* convergence: projected full step small and residual small */
        int step_ok = 1;
        {
            SolveVars full;
            take_step(&full, sv, idx, n, d, 1.0, pr, set);
            for (int a = 0; a < n; a++) {
                int i = idx[a];
                double atol = (i == IDX_T) ? 0 : JACO_X_ATOL * var_scale[i];
                if (fabs(full.data[i] - sv->data[i]) > tol * fabs(sv->data[i]) + atol) step_ok = 0;
            }
        }
        if (step_ok && res_ok) {
            if (F_out) *F_out = e.F;
            return NEWTON_OK;
        }
        if (it >= maxiter) return NEWTON_MAXITER;

        /* bounded step: T may fall by at most JACO_TAU_T, neutral budgets shrink by at most JACO_TAU_BUDGET */
        double alpha = 1;
        for (int a = 0; a < n; a++)
            if (idx[a] == IDX_T && d[a] < 0) alpha = fmin(alpha, JACO_TAU_T * sv->T / -d[a]);
        for (int bb = 0; bb < N_BUDGETS; bb++) {
            const struct Budget *B = &budget_table[bb];
            double db = 0, bval = budget_value(B, sv, pr);
            for (int a = 0; a < n; a++) db += -budget_weight(B, idx[a]) * d[a];
            if (db < 0 && bval > 0) alpha = fmin(alpha, JACO_TAU_BUDGET * bval / -db);
        }

        double norm_d = scaled_norm(d, w, n);
        int accepted = 0;
        SolveVars trial;
        for (int bt = 0; bt <= JACO_MAX_BACKTRACK; bt++, alpha *= 0.5) {
            take_step(&trial, sv, idx, n, d, alpha, pr, set);
            if (evaluate(&trial, pr, &et, c)) continue;
            if (ptc) add_ptc(&trial, pr, ptc, &et);
            double bt_rhs[N_VARS], dbar[N_VARS];
            reduced_rhs(&et, idx, n, bt_rhs);
            if (lu_solve(&lu, bt_rhs, dbar)) continue;
            if (scaled_norm(dbar, w, n) <= (1 - 0.25 * alpha) * norm_d) {
                accepted = 1;
                break;
            }
        }
        if (!accepted) return NEWTON_LINESEARCH;
        *sv = trial;
        e = et;
    }
}

/* ---- tier 2 ---- */

/* 1-D rootfinding with warm starts: every evaluation solves an inner system starting from the
   state of the nearest previously evaluated point. */
#define JACO_ROOT_HISTORY 32
struct RootCtx {
    int (*eval)(double y, SolveVars *x, struct RootCtx *ctx, double *val);
    const Params *pr;
    const struct JacoSolverSettings *set;
    struct Counters *c;
    int level; /* the time-dependent species a td_bracketed search is over */
    int nhist, next;
    double yh[JACO_ROOT_HISTORY];
    SolveVars xh[JACO_ROOT_HISTORY];
};

static void root_init(struct RootCtx *ctx, int (*eval)(double, SolveVars *, struct RootCtx *, double *), const Params *pr,
                      const struct JacoSolverSettings *set, struct Counters *c) {
    ctx->eval = eval;
    ctx->pr = pr;
    ctx->set = set;
    ctx->c = c;
    ctx->level = 0;
    ctx->nhist = 0;
    ctx->next = 0;
}

static void root_remember(struct RootCtx *ctx, double y, const SolveVars *x) {
    int slot = ctx->nhist < JACO_ROOT_HISTORY ? ctx->nhist++ : (ctx->next++ % JACO_ROOT_HISTORY);
    ctx->yh[slot] = y;
    ctx->xh[slot] = *x;
}

/* evaluate at y from the nearest remembered state (or *x if none); the solved state goes to *x */
static int root_eval(struct RootCtx *ctx, double y, SolveVars *x, double *val) {
    int best = -1;
    for (int i = 0; i < ctx->nhist; i++)
        if (best < 0 || fabs(ctx->yh[i] - y) < fabs(ctx->yh[best] - y)) best = i;
    if (best >= 0) *x = ctx->xh[best];
    if (ctx->eval(y, x, ctx, val)) return -1;
    root_remember(ctx, y, x);
    return 0;
}

/* Brent's method on [a, b] (fa, fb of opposite sign; xa, xb the solved states there) to
   |dy| <= ytol. The root and its solved state are returned. */
static int brent(struct RootCtx *ctx, double a, double fa, const SolveVars *xa, double b, double fb, const SolveVars *xb,
                 double ytol, double *root, SolveVars *x_root) {
    SolveVars sa = *xa, sb = *xb, sc = sa;
    double cc = a, fc = fa, dd = b - a, ee = dd;
    for (int it = 0; it < JACO_BRENT_MAXITER; it++) {
        if ((fb > 0) == (fc > 0)) {
            cc = a;
            fc = fa;
            sc = sa;
            dd = ee = b - a;
        }
        if (fabs(fc) < fabs(fb)) {
            a = b; fa = fb; sa = sb;
            b = cc; fb = fc; sb = sc;
            cc = a; fc = fa; sc = sa;
        }
        double tol1 = 2e-16 * fabs(b) + 0.5 * ytol, xm = 0.5 * (cc - b);
        if (fabs(xm) <= tol1 || fb == 0) {
            *root = b;
            *x_root = sb;
            return 0;
        }
        if (fabs(ee) >= tol1 && fabs(fa) > fabs(fb)) {
            double s = fb / fa, p, q;
            if (a == cc) {
                p = 2 * xm * s;
                q = 1 - s;
            } else {
                double qq = fa / fc, r = fb / fc;
                p = s * (2 * xm * qq * (qq - r) - (b - a) * (r - 1));
                q = (qq - 1) * (r - 1) * (s - 1);
            }
            if (p > 0) q = -q;
            p = fabs(p);
            if (2 * p < fmin(3 * xm * q - fabs(tol1 * q), fabs(ee * q))) {
                ee = dd;
                dd = p / q;
            } else {
                dd = xm;
                ee = dd;
            }
        } else {
            dd = xm;
            ee = dd;
        }
        a = b;
        fa = fb;
        sa = sb;
        b += (fabs(dd) > tol1) ? dd : (xm > 0 ? tol1 : -tol1);
        if (root_eval(ctx, b, &sb, &fb)) return -1;
    }
    return -1;
}

/* Steady-state species at fixed T (and fixed time-dependent species): Newton; if that fails,
   pseudo-transient continuation from the same start, the pseudo-timestep growing from a tenth of
   the fastest chemical timescale until the pseudo-time term is negligible. */
static int ions_at_fixed(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c,
                         SolveVars *F_out) {
    SolveVars start = *sv;
    int st = newton(sv, pr, set, SOLVE_IONS, JACO_CHEM_MAXITER, NULL, c, F_out);
    if (st == NEWTON_OK) return 0;
    *sv = start;
    if (ionized_seed(sv, pr) && newton(sv, pr, set, SOLVE_IONS, JACO_CHEM_MAXITER, NULL, c, F_out) == NEWTON_OK) return 0;
    if (set->verbose) printf("  jaco ions at T=%g: Newton %s, trying pseudo-transient continuation\n", sv->T, newton_status(st));
    *sv = start;
    struct Eval e;
    if (evaluate(sv, pr, &e, c)) return -1;
    double rate_max = 0;
    for (int k = 2; k < N_VARS; k++) {
        double r = fabs(e.J[k][k]) / pr->n_Htot;
        if (!is_time_dependent(k) && jaco_isfinite(r)) rate_max = fmax(rate_max, r);
    }
    if (!(rate_max > 0)) return -1;
    struct PTC ptc;
    ptc.inv_tau = 10 * rate_max;
    for (int stage = 0; stage < JACO_PTC_MAXSTAGES; stage++) {
        ptc.anchor = *sv;
        SolveVars trial = *sv;
        if (newton(&trial, pr, set, SOLVE_IONS, JACO_PTC_MAXITER, &ptc, c, NULL) != NEWTON_OK) {
            ptc.inv_tau *= JACO_PTC_GROW * JACO_PTC_GROW; /* retreat to a smaller pseudo-step */
            continue;
        }
        *sv = trial;
        ptc.inv_tau /= JACO_PTC_GROW;
        /* done once the pseudo-time term is negligible against the current rates (they change as
           the state moves, e.g. recombination slows as x_e falls) */
        if (evaluate(sv, pr, &e, c)) return -1;
        int negligible = 1;
        for (int k = 2; k < N_VARS; k++)
            if (!is_time_dependent(k) && (sv->data[k] > floor_pin(k) || e.F.data[k] > 0) &&
                pr->n_Htot * ptc.inv_tau > 1e-3 * fabs(e.J[k][k]))
                negligible = 0;
        if (negligible) break;
    }
    st = newton(sv, pr, set, SOLVE_IONS, JACO_CHEM_MAXITER, NULL, c, F_out);
    if (st == NEWTON_OK) return 0;
    if (set->verbose) printf("  jaco ions at T=%g failed after continuation: %s\n", sv->T, newton_status(st));
    /* Last, from the floor: where nothing ionizes the gas the root is at x = 0, and from an ionized start Newton only
       halves the ions per step when the free electrons are the ions themselves (F ~ -x^2). */
    *sv = start;
    for (int k = 2; k < N_VARS; k++)
        if (!is_time_dependent(k)) sv->data[k] = var_floor[k];
    return newton(sv, pr, set, SOLVE_IONS, JACO_CHEM_MAXITER, NULL, c, F_out) == NEWTON_OK ? 0 : -1;
}

/* Largest abundance of time-dependent species k: its ceiling and its budgets, the other species fixed */
static double td_upper(const SolveVars *x, const Params *pr, int k) {
    return fmin(var_ceiling[k], budget_cap(x, pr, k)) * (1 - 1e-9);
}

static int td_bracketed(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c, int level);

/* At fixed T, with the time-dependent species before `level` fixed, solve the others: the next time-dependent species
   by its own bracketed search, or, once all of them are fixed, the steady-state species. F_out gets the full residual
   at the solution. */
static int td_inner(SolveVars *x, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c, int level,
                    SolveVars *F_out) {
    if (level >= N_TD) return ions_at_fixed(x, pr, set, c, F_out);
    if (td_bracketed(x, pr, set, c, level)) return -1;
    struct Eval e;
    if (evaluate(x, pr, &e, c)) return -1;
    *F_out = e.F;
    return 0;
}

/* residual of the time-dependent species ctx->level at abundance exp(y), the species after it solved there */
static int td_eval(double y, SolveVars *x, struct RootCtx *ctx, double *val) {
    const int k = td_species[ctx->level].k;
    x->data[k] = fmax(var_floor[k], fmin(exp(y), td_upper(x, ctx->pr, k)));
    SolveVars F;
    if (td_inner(x, ctx->pr, ctx->set, ctx->c, ctx->level + 1, &F)) return -1;
    *val = F.data[k];
    return 0;
}

/* Time-dependent species `level` at fixed T as a scalar backward-Euler equation, the species after it solved at each
   trial: walk geometrically from its start-of-step value in the direction its net rate points, to the first sign
   change (the root the time evolution reaches first; e.g. H2 self-shielding makes its equation non-monotonic, with
   several roots), then Brent in ln x. At the floor the residual is >= 0 (nothing destroys a species that is not
   there, and the backward-Euler term pulls it up) and at its cap <= 0, so a root or a pinned end exists. */
static int td_bracketed(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c, int level) {
    const int k = td_species[level].k;
    struct RootCtx ctx;
    root_init(&ctx, td_eval, pr, set, c);
    ctx.level = level;
    const double y_floor = log(var_floor[k]);
    SolveVars xa = *sv;
    double ya = log(fmax(var_floor[k], fmin(pr->data[td_species[level].param], td_upper(sv, pr, k)))), Ga;
    if (root_eval(&ctx, ya, &xa, &Ga)) return -1;
    int up = Ga > 0;
    double fac = log(2.0), yb = ya, Gb = Ga;
    SolveVars xb = xa;
    for (int it = 0; it < JACO_BRACKET_MAXITER && Ga != 0; it++) {
        double y_hi = log(td_upper(&xa, pr, k));
        if ((up && ya >= y_hi) || (!up && ya <= y_floor)) {
            *sv = xa; /* the net rate keeps its sign to the bound: the bound is the answer */
            return 0;
        }
        yb = up ? fmin(y_hi, ya + fac) : fmax(y_floor, ya - fac);
        if (root_eval(&ctx, yb, &xb, &Gb)) return -1;
        if ((Gb > 0) != (Ga > 0) || Gb == 0) break;
        ya = yb;
        Ga = Gb;
        xa = xb;
        fac *= 2;
    }
    if (Ga == 0) {
        *sv = xa;
        return 0;
    }
    if ((Gb > 0) == (Ga > 0) && Gb != 0) return -1;
    double y;
    if (brent(&ctx, ya, Ga, &xa, yb, Gb, &xb, set->tol, &y, sv)) return -1;
    return 0;
}

/* Chemistry at fixed T: Newton on all species; if that fails, the time-dependent species as nested
   bracketed scalar problems around the steady-state ones (or, without time-dependent species,
   the steady-state fallback directly). F_out gets the full residual at the returned state. */
static int chemistry_at_T(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c,
                          SolveVars *F_out) {
    SolveVars start = *sv;
    int st = newton(sv, pr, set, SOLVE_IONS | SOLVE_TD, JACO_CHEM_MAXITER, NULL, c, F_out);
    if (st == NEWTON_OK) return 0;
    *sv = start;
    if (ionized_seed(sv, pr) && newton(sv, pr, set, SOLVE_IONS | SOLVE_TD, JACO_CHEM_MAXITER, NULL, c, F_out) == NEWTON_OK)
        return 0;
    if (set->verbose) printf("  jaco chemistry at T=%g: Newton %s, falling back\n", sv->T, newton_status(st));
    *sv = start;
    if (N_TD == 0) return ions_at_fixed(sv, pr, set, c, F_out);
    if (td_bracketed(sv, pr, set, c, 0)) return -1;
    /* residual at the returned state, which the bracketed solve evaluated last there */
    struct Eval e;
    if (evaluate(sv, pr, &e, c)) return -1;
    if (F_out) *F_out = e.F;
    return 0;
}

/* Energy residual at T with the chemistry solved there, signed as in DoCooling: R > 0 means T is
   too high. x is the warm start in and the chemistry solution out. */
static int energy_residual(double T, SolveVars *x, const Params *pr, const struct JacoSolverSettings *set,
                           struct Counters *c, double *R) {
    x->T = T;
    SolveVars F;
    if (chemistry_at_T(x, pr, set, c, &F)) return -1;
    *R = -F.T;
    return 0;
}

static int energy_eval(double y, SolveVars *x, struct RootCtx *ctx, double *val) {
    return energy_residual(exp(y), x, ctx->pr, ctx->set, ctx->c, val);
}

/* Move x to the energy floor: the T where u(T, x) = u_min (or T_min), with the chemistry solved
   there; a fixed point because the EOS depends on the abundances. R is the energy residual there. */
static int pin_to_floor(SolveVars *x, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c,
                        double *R) {
    for (int it = 0; it < 6; it++) {
        double Tf = set->T_min;
        if (set->u_min > 0) Tf = fmax(Tf, jaco_u_to_T(set->u_min, x, pr));
        int settled = it > 0 && fabs(Tf - x->T) <= set->tol * Tf;
        if (energy_residual(Tf, x, pr, set, c, R)) return -1;
        if (settled) return 0;
    }
    return 0;
}

/* Tier-2 search. On success sv holds the answer, PINNED if it is the floor or ceiling. */
static enum Outcome rootfind_T(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c) {
    struct RootCtx ctx;
    root_init(&ctx, energy_eval, pr, set, c);
    const double T_hi = set->T_max;
    SolveVars xa = *sv;
    double Ta = fmax(set->T_min, fmin(T_hi, sv->T)), Ra;
    if (root_eval(&ctx, log(Ta), &xa, &Ra)) return OUTCOME_FAILED;
    int a_at_floor = 0;
    if (xa.u < set->u_min) {
        if (pin_to_floor(&xa, pr, set, c, &Ra)) return OUTCOME_FAILED;
        Ta = xa.T;
        root_remember(&ctx, log(Ta), &xa);
        a_at_floor = 1;
    }
    if (Ra == 0) {
        *sv = xa;
        return OUTCOME_SOLVED;
    }

    /* bracket: walk geometrically in the direction R points to, stopping at the floor or ceiling */
    int down = Ra > 0;
    double fac = JACO_BRACKET_FAC0, Tb = Ta, Rb = Ra;
    SolveVars xb = xa;
    int bracketed = 0;
    for (int it = 0; it < JACO_BRACKET_MAXITER; it++) {
        if ((down && (a_at_floor || Ta <= set->T_min)) || (!down && Ta >= T_hi)) {
            *sv = xa; /* R kept its sign all the way to the limit: the limit is the answer */
            if (set->verbose) printf("  jaco tier 2: pinned at T=%g (R=%g)\n", Ta, Ra);
            return OUTCOME_PINNED;
        }
        Tb = down ? fmax(set->T_min, Ta / fac) : fmin(T_hi, Ta * fac);
        if (root_eval(&ctx, log(Tb), &xb, &Rb)) return OUTCOME_FAILED;
        int b_at_floor = 0;
        if (down && xb.u < set->u_min) {
            if (pin_to_floor(&xb, pr, set, c, &Rb)) return OUTCOME_FAILED;
            Tb = xb.T;
            root_remember(&ctx, log(Tb), &xb);
            b_at_floor = 1;
        }
        if (Rb == 0) {
            *sv = xb;
            return OUTCOME_SOLVED;
        }
        if ((Rb > 0) != (Ra > 0)) {
            bracketed = 1;
            break;
        }
        Ta = Tb;
        Ra = Rb;
        xa = xb;
        a_at_floor = b_at_floor;
        fac *= JACO_BRACKET_GROW;
    }
    if (!bracketed) return OUTCOME_FAILED;

    double y;
    if (brent(&ctx, log(Ta), Ra, &xa, log(Tb), Rb, &xb, set->tol, &y, sv)) return OUTCOME_FAILED;

    /* polish on the full system; keep the rootfind answer if the polish does not converge */
    SolveVars pol = *sv;
    if (newton(&pol, pr, set, SOLVE_ALL, JACO_POLISH_MAXITER, NULL, c, NULL) == NEWTON_OK) *sv = pol;
    return OUTCOME_SOLVED;
}

/* ---- driver ---- */

/* An answer within tol below the energy floor is moved onto it exactly: T is raised, abundances
   fixed, until u(T, x) >= u_min as evaluated (Newton in T, then ulp steps for round-off). The
   floor fixed point and the EOS inversion only reach it to their tolerances. SOLVED if the answer
   was not below, PINNED if it was moved, FAILED if it was further below. */
static enum Outcome onto_floor(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set) {
    double cv;
    sv->u = jaco_T_to_u(sv->T, sv, pr, &cv);
    if (sv->u >= set->u_min) return OUTCOME_SOLVED;
    if (sv->u < set->u_min * (1 - set->tol)) return OUTCOME_FAILED;
    for (int it = 0; it < 100 && sv->u < set->u_min; it++) {
        double T_new = sv->T + (set->u_min - sv->u) / cv;
        sv->T = fmin(set->T_max, fmax(T_new, nextafter(sv->T, set->T_max)));
        sv->u = jaco_T_to_u(sv->T, sv, pr, &cv);
    }
    return sv->u >= set->u_min ? OUTCOME_PINNED : OUTCOME_FAILED;
}

/* Acceptance test of a tier-2 answer from a fresh evaluation: every free row within
   JACO_VERIFY_FAC tol of its row_scale; a floored species must not be net-produced, and a
   temperature pinned at the floor (ceiling) must have an energy residual pointing below (above). */
static int verify(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c, int pinned) {
    struct Eval e;
    if (evaluate(sv, pr, &e, c)) return -1;
    const double lim = JACO_VERIFY_FAC * set->tol;
    double r = e.F.T / row_scale(sv, pr, &e, IDX_T);
    if (pinned ? (sv->T >= set->T_max ? r < -lim : r > lim) : fabs(r) > lim) return -1;
    for (int k = 2; k < N_VARS; k++) {
        double rk = e.F.data[k] / row_scale(sv, pr, &e, k);
        if (sv->data[k] <= floor_pin(k) ? rk > lim : fabs(rk) > lim) return -1;
    }
    return 0;
}

/* Tier 1: Newton from the start, moved onto the energy floor if just below it. *status: 0, a
   NEWTON_ code, or -5 (converged below the energy floor). */
static enum Outcome tier1_newton(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c,
                                 int *status) {
    int st = newton(sv, pr, set, SOLVE_ALL, JACO_TIER1_MAXITER, NULL, c, NULL);
    enum Outcome o = (st == NEWTON_OK) ? onto_floor(sv, pr, set) : OUTCOME_FAILED;
    *status = (st == NEWTON_OK && o == OUTCOME_FAILED) ? -5 : st;
    return o;
}

/* Tier 2: rootfind in T, onto the energy floor, then the acceptance test. */
static enum Outcome tier2_rootfind(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c) {
    enum Outcome r = rootfind_T(sv, pr, set, c);
    if (r == OUTCOME_FAILED) return r;
    enum Outcome f = onto_floor(sv, pr, set);
    if (f == OUTCOME_FAILED) return f;
    int pinned = (r == OUTCOME_PINNED || f == OUTCOME_PINNED);
    if (verify(sv, pr, set, c, pinned)) {
        if (set->verbose) printf("  jaco tier 2 answer T=%g failed the acceptance test\n", sv->T);
        return OUTCOME_FAILED;
    }
    return pinned ? OUTCOME_PINNED : OUTCOME_SOLVED;
}

/* Tiers 1 and 2 over one (sub)step; *tier is set on success, sv left at the start on failure. */
static enum Outcome solve_step(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct Counters *c,
                               int *tier, int *nfeval_tier1, int *tier1_status) {
    SolveVars start = *sv;
    int status;
    enum Outcome o = tier1_newton(sv, pr, set, c, &status);
    if (nfeval_tier1) *nfeval_tier1 = c->nfeval;
    if (tier1_status) *tier1_status = status;
    if (o != OUTCOME_FAILED) {
        *tier = JACO_TIER_NEWTON;
        return o;
    }
    if (set->verbose)
        printf("  jaco tier 1 failed (%s) T0=%g n=%g dt=%g; tier 2\n", status == -5 ? "below energy floor" : newton_status(status),
               start.T, pr->n_Htot, pr->Delta_t);
    *sv = start;
    o = tier2_rootfind(sv, pr, set, c);
    if (o != OUTCOME_FAILED) {
        *tier = JACO_TIER_ROOTFIND;
        return o;
    }
    if (set->verbose) printf("  jaco tier 2 failed T0=%g n=%g dt=%g\n", start.T, pr->n_Htot, pr->Delta_t);
    *sv = start;
    return OUTCOME_FAILED;
}

int jaco_solve(SolveVars *sv, const Params *pr_in, const struct JacoSolverSettings *set, struct JacoSolveInfo *info) {
    struct JacoSolveInfo dummy;
    if (!info) info = &dummy;
    memset(info, 0, sizeof(*info));
    struct Counters c = {0, 0, 0, 0, set->fd_jacobian};
    const SolveVars sv_in = *sv;
    Params pr = *pr_in;
    sanitize(sv, &pr, set);

    /* tier 0: start from the temperature the initial energy implies */
    double u0 = jaco_T_to_u(sv->T, sv, &pr, NULL);
    if (!(fabs(u0 - pr.u_initial) <= JACO_EOS_START_TOL * pr.u_initial)) {
        double T0 = jaco_u_to_T(fmax(pr.u_initial, set->u_min), sv, &pr);
        if (jaco_isfinite(T0) && T0 > 0) {
            sv->T = fmax(set->T_min, fmin(set->T_max, T0));
            info->resynced_T0 = 1;
        }
    }
    const SolveVars start = *sv;

    int tier = 0;
    enum Outcome o = solve_step(sv, &pr, set, &c, &tier, &info->nfeval_tier1, &info->tier1_status);
    if (o != OUTCOME_FAILED) {
        info->tier = tier;
        info->pinned = (o == OUTCOME_PINNED);
        info->nfeval = c.nfeval;
        info->nfeval_fd = c.nfeval_fd;
        info->n_nonfinite_jac = c.n_nonfinite_jac;
        info->n_nonfinite_F = c.n_nonfinite_F;
        return 0;
    }

    /* tier 3: subcycle */
    if (set->verbose) printf("  jaco tier 3: subcycling dt=%g\n", pr.Delta_t);
    SolveVars cur = start;
    Params ps = pr;
    const double dt = pr.Delta_t;
    double t_done = 0, dt_sub = 0.5 * dt;
    int nsub = 0, ok = 1, pinned = 0;
    while (t_done < dt * (1 - 1e-12)) {
        if (nsub >= JACO_SUBCYCLE_MAXSTEPS || dt_sub < JACO_SUBCYCLE_MIN_FRAC * dt) {
            ok = 0;
            break;
        }
        ps.Delta_t = fmin(dt_sub, dt - t_done);
        if (nsub > 0) {
            ps.u_initial = cur.u;
            jaco_initial_from_state(&cur, &ps);
        }
        SolveVars trial = cur;
        int t2;
        enum Outcome o2 = solve_step(&trial, &ps, set, &c, &t2, NULL, NULL);
        if (o2 != OUTCOME_FAILED) {
            cur = trial;
            t_done += ps.Delta_t;
            nsub++;
            pinned |= (o2 == OUTCOME_PINNED);
            dt_sub *= 2;
        } else {
            dt_sub *= 0.5;
        }
    }
    info->nfeval = c.nfeval;
    info->nfeval_fd = c.nfeval_fd;
    info->n_nonfinite_jac = c.n_nonfinite_jac;
    info->n_nonfinite_F = c.n_nonfinite_F;
    info->n_substeps = nsub;
    if (!ok) {
        info->tier = JACO_TIER_FAILED;
        *sv = sv_in;
        return 1;
    }
    *sv = cur;
    info->tier = JACO_TIER_SUBCYCLE;
    info->pinned = pinned;
    return 0;
}

void jaco_initial_from_state(const SolveVars *sv, Params *pr) {
    for (int t = 0; t < N_TD; t++) pr->data[td_species[t].param] = sv->data[td_species[t].k];
}

int jaco_solve_chemistry(SolveVars *sv, const Params *pr_in, const struct JacoSolverSettings *set, int *nfeval) {
    struct Counters c = {0, 0, 0, 0, set->fd_jacobian};
    Params pr = *pr_in;
    sanitize(sv, &pr, set);
    int rc = chemistry_at_T(sv, &pr, set, &c, NULL);
    if (nfeval) *nfeval += c.nfeval;
    return rc;
}

void jaco_print_state(FILE *fp, const char *label, const SolveVars *sv, const Params *pr) {
    fprintf(fp, "%s\n  SolveVars (IDX_* order):", label);
    for (int k = 0; k < N_VARS; k++) fprintf(fp, " [%d]=%.17g", k, sv->data[k]);
    fprintf(fp, "\n  Params (PARAM_* order):");
    for (int k = 0; k < N_PARAMS; k++) fprintf(fp, " [%d]=%.17g", k, pr->data[k]);
    fprintf(fp, "\n");
}

void jaco_print_outputs(FILE *fp, const char *label, const Outputs *out) {
    fprintf(fp, "%s", label);
#ifdef JACO_HAS_OUTPUTS
    static const char *names[N_OUTPUTS] = JACO_OUTPUT_NAMES, *units[N_OUTPUTS] = JACO_OUTPUT_UNITS;
    for (int k = 0; k < N_OUTPUTS; k++) fprintf(fp, " %s=%.10g [%s]", names[k], out->data[k], units[k]);
#else
    (void)out;
    fprintf(fp, " (the model has no outputs)");
#endif
    fprintf(fp, "\n");
}
