/* Standalone test of the jaco implicit solver (cooling/jaco_solver.cc) on the STARFORGE model.
 *
 *   1. replays cells that broke the previous solver in a gmc_cooling run;
 *   2. sweeps n_Htot x T0 x dt x pdv_work x (u0 consistent / +-10%) x x_H2 seed, plus a "warm"
 *      re-solve of every answer with u_initial raised 1% (what the next timestep of a settled cell
 *      looks like), and reports failures, tier counts and nfeval percentiles;
 *   3. checks every answer independently: residual of the generated system at the returned state,
 *      u = u(T,x), abundances and neutral budgets in bounds; and, where the energy backward-Euler
 *      term is negligible (dt >= 1e13 s), that T is within 3% of a root of heat(T) + pdv_work = 0
 *      with the chemistry at steady state, located by its own bisection.
 * Exits non-zero on any failure.
 *
 *   make -C test/jaco_solver
 *
 * Debug modes (after the table directory argument, e.g. ./build/test_jaco_solver build ...):
 *   case n T0 dt pdv ufac seed [verbose] [warm]    one sweep-style solve with an iterate trace
 *   gscan n dt pdv seed Tlo Thi npts               heat + pdv with steady-state chemistry vs T
 *   replay file [which] [verbose] [jac]            re-solve states dumped by jaco_print_state
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "jaco_solver.h"

#ifndef JACO_MODEL_STARFORGE
#error "test_jaco_solver drives the STARFORGE model (MODEL=starforge)"
#endif

void jaco_init_tables(const char *dir);

static const double M_PER_H = 2.34e-24; /* mass per H nucleus used to express the energy residual relative to rho u/dt */
static const double X_ATOL = 1e-14;     /* absolute abundance scale of the residual check */
static const double CHECK_TOL = 1e-5;   /* residual check: 10x the solver tolerance */
static const double EQ_GATE = 0.01;     /* equilibrium check applies when the backward-Euler term shifts T by < 1% (linearized) */
static const double EQ_TOL = 0.03;      /* ... and then T must be within 3% of an equilibrium root */

static void starforge_params(Params *pr, double n, double dt) {
    memset(pr, 0, sizeof(*pr));
    pr->n_Htot = n;
    pr->Delta_t = dt;
    pr->y = 0.0994;
    pr->ISRF = 1.0;
    pr->G_0 = 1.0;
    pr->Td = 10.0;
    pr->Z_d = 1.0;
    pr->f_d = 1.0;
    pr->Delta_x = 3e18;
    pr->N_H = n * pr->Delta_x;
    pr->grad_v = 1e-14;
    pr->x_C_tot = 2.1e-4;
    pr->x_N = 6.8e-5;
    pr->x_Ne = 8.5e-5;
    pr->x_Mg = 3.2e-5;
    pr->x_Si = 3.2e-5;
    pr->x_S = 1.3e-5;
    pr->x_Ca = 2.2e-6;
    pr->x_Fe = 2.5e-5;
    pr->x_O_tot = 4.9e-4;
    pr->z = 0;
}

/* ---- CIE seed table, as jaco_build_cie_table in cooling/jaco.cc ---- */
#define CIE_N 200
#define CIE_LOGTMIN 2.5
#define CIE_LOGTMAX 6.5
static double cie_xHp[CIE_N], cie_xHep[CIE_N], cie_xHepp[CIE_N];

static int build_cie_table(const JacoSolverSettings *set) {
    Params pr;
    starforge_params(&pr, 1.0, 1e15);
    pr.Td = 15.0;
    pr.N_H = 1e20;
    double xHp = 0.99, xHep = 1e-4, xHepp = 0.099;
    for (int i = CIE_N - 1; i >= 0; i--) {
        SolveVars sv = {};
        sv.T = pow(10.0, CIE_LOGTMIN + (CIE_LOGTMAX - CIE_LOGTMIN) * i / (CIE_N - 1));
        sv.x_Hplus = xHp;
        sv.x_Heplus = xHep;
        sv.x_Heplusplus = xHepp;
        sv.x_H_2 = JACO_ABUNDANCE_FLOOR;
        pr.x_H_2_initial = sv.x_H_2;
        if (jaco_solve_chemistry(&sv, &pr, set, NULL)) {
            printf("CIE table: chemistry failed at T=%g\n", sv.T);
            return 1;
        }
        cie_xHp[i] = xHp = sv.x_Hplus;
        cie_xHep[i] = xHep = sv.x_Heplus;
        cie_xHepp[i] = xHepp = sv.x_Heplusplus;
    }
    return 0;
}

static double cie_interp(const double *table, double logT) {
    double f = (logT - CIE_LOGTMIN) / (CIE_LOGTMAX - CIE_LOGTMIN) * (CIE_N - 1);
    if (f <= 0) return table[0];
    if (f >= CIE_N - 1) return table[CIE_N - 1];
    int i = (int)f;
    double t = f - i;
    return (1 - t) * table[i] + t * table[i + 1];
}

/* seeds as gizmo_to_jaco makes them: ions from the CIE table at the cached T, H2 from the cell */
static void seed(SolveVars *sv, double T0, double xH2) {
    double logT = log10(fmax(T0, 10.));
    sv->T = T0;
    sv->x_Hplus = cie_interp(cie_xHp, logT);
    sv->x_Heplus = cie_interp(cie_xHep, logT);
    sv->x_Heplusplus = cie_interp(cie_xHepp, logT);
    sv->x_H_2 = xH2;
}

/* sweep seed variants: CIE ions with little or much H2, or fully ionized (x_H+ = 1 exactly) */
enum { SEED_ATOMIC, SEED_MOLECULAR, SEED_IONIZED, N_SEEDS };
static const char *seed_name[N_SEEDS] = {"CIE+xH2=1e-6", "CIE+xH2=0.25", "x_H+=1"};
static void seed_variant(SolveVars *sv, double T0, int variant, double y) {
    if (variant == SEED_IONIZED) {
        sv->T = T0;
        sv->x_Hplus = 1.0;
        sv->x_Heplus = JACO_ABUNDANCE_FLOOR;
        sv->x_Heplusplus = y;
        sv->x_H_2 = JACO_ABUNDANCE_FLOOR;
    } else
        seed(sv, T0, variant == SEED_ATOMIC ? 1e-6 : 0.25);
}

/* ---- independent check of an answer ---- */

static double T_floor_of(const SolveVars *sv, const Params *pr, const JacoSolverSettings *set) {
    return fmax(set->T_min, jaco_u_to_T(set->u_min, sv, pr));
}

/* d F_k / d v_k: the generated value, or a finite difference where the model returns a non-finite one */
static double diag_derivative(const SolveVars *sv, const Params *pr, const SolveVars *F, double J[N_VARS][N_VARS], int k) {
    if (isfinite(J[k][k])) return J[k][k];
    SolveVars sp = *sv, Fp;
    double Jp[N_VARS][N_VARS];
    double h = (k == IDX_T) ? -1e-7 * sv->T : 1e-7 * fmax(sv->data[k], 1e-10);
    sp.data[k] += h;
    microphysics_func_jac(&sp, pr, &Fp, Jp);
    return (Fp.data[k] - F->data[k]) / h;
}

/* 0 if OK; otherwise writes the reason. *worst gets the largest residual / (CHECK_TOL * scale). */
static int check_answer(const SolveVars *sv, const Params *pr, const JacoSolverSettings *set, char *why, double *worst) {
    SolveVars F;
    double J[N_VARS][N_VARS];
    microphysics_func_jac(sv, pr, &F, J);
    *worst = 0;
    for (int i = 0; i < N_VARS; i++)
        if (!isfinite(F.data[i]) || !isfinite(sv->data[i])) {
            sprintf(why, "non-finite state or residual [%d]", i);
            return 1;
        }
    double u_eos = jaco_T_to_u(sv->T, sv, pr, NULL);
    if (fabs(sv->u - u_eos) > 1e-12 * u_eos) {
        sprintf(why, "u=%g != u(T,x)=%g", sv->u, u_eos);
        return 1;
    }
    for (int k = 2; k < N_VARS; k++)
        if (sv->data[k] < JACO_ABUNDANCE_FLOOR || sv->data[k] > 1) {
            sprintf(why, "abundance [%d]=%g out of bounds", k, sv->data[k]);
            return 1;
        }
    double xH0 = 1 - sv->x_Hplus - 2 * sv->x_H_2, xHe0 = pr->y - sv->x_Heplus - sv->x_Heplusplus;
    if (xH0 < 0 || xHe0 < 0) {
        sprintf(why, "neutral budget negative: x_H0=%g x_He0=%g", xH0, xHe0);
        return 1;
    }
    if (sv->T < set->T_min || sv->T > set->T_max || sv->u < set->u_min * (1 - 1e-9)) {
        sprintf(why, "T=%g or u=%g outside the floor/ceiling", sv->T, sv->u);
        return 1;
    }
    for (int k = 1; k < N_VARS; k++) J[k][k] = diag_derivative(sv, pr, &F, J, k);
    /* energy row: |F_T| small relative to rho u/dt + |dF_T/dT| T, unless T is at a limit and F_T points past it */
    double sT = M_PER_H * pr->n_Htot * sv->u / pr->Delta_t + fabs(J[IDX_T][IDX_T]) * sv->T;
    double r = F.T / (CHECK_TOL * sT);
    int at_floor = sv->T <= T_floor_of(sv, pr, set) * (1 + 1e-4) || sv->u <= set->u_min * (1 + 1e-4);
    int at_ceiling = sv->T >= set->T_max * (1 - 1e-9);
    double rT = at_floor ? fmax(r, 0) : at_ceiling ? fmax(-r, 0) : fabs(r);
    *worst = rT;
    if (rT > 1) {
        sprintf(why, "energy residual %.3g x tolerance (F_T=%g scale=%g%s)", rT, F.T, sT, at_floor ? ", at floor" : "");
        return 1;
    }
    for (int k = 2; k < N_VARS; k++) {
        double sk = fabs(J[k][k]) * (sv->data[k] + X_ATOL) + 1e-300;
        double rk = F.data[k] / (CHECK_TOL * sk);
        if (sv->data[k] <= JACO_ABUNDANCE_FLOOR * (1 + 1e-9)) rk = fmax(rk, 0); /* floored: must not want to grow */
        rk = fabs(rk);
        *worst = fmax(*worst, rk);
        if (rk > 1) {
            sprintf(why, "species [%d] residual %.3g x tolerance (F=%g x=%g)", k, rk, F.data[k], sv->data[k]);
            return 1;
        }
    }
    return 0;
}

/* ---- reference equilibrium: roots of G(T) = heat(T) + pdv_work with steady-state ions ---- */

/* x_H2 keeps its backward-Euler term (it is time-dependent in the system being checked); only the
   energy backward-Euler term is removed, by setting u_initial = u(T, x). */
static int G_of_T(double T, SolveVars *x, const Params *pr, const JacoSolverSettings *set, double *G, double *u) {
    x->T = T;
    if (jaco_solve_chemistry(x, pr, set, NULL)) return 1;
    Params p = *pr;
    p.u_initial = x->u;
    SolveVars F;
    double J[N_VARS][N_VARS];
    microphysics_func_jac(x, &p, &F, J);
    *G = F.T;
    *u = x->u;
    return isfinite(*G) ? 0 : 1;
}

/* Where the energy backward-Euler term is negligible, T must sit on a root of G. The root is
   located directly: a sign change of G among T(1 +- {0.1, 1, 3}%), refined by bisection.
   *shift: the T offset (relative) the backward-Euler term causes, linearized at T; infinite if
   the thermal relaxation time exceeds dt/100 (then dt is too short to reach equilibrium).
   Returns 0 and the relative distance to the root in *dist, or 1 if no root within EQ_TOL. */
static int near_root(const SolveVars *sol, const Params *pr, const JacoSolverSettings *set, double *shift, double *dist) {
    const double offs[7] = {-0.03, -0.01, -0.001, 0, 0.001, 0.01, 0.03};
    double G[7], u;
    SolveVars xs[7];
    for (int i = 0; i < 7; i++) {
        xs[i] = *sol;
        if (G_of_T(sol->T * (1 + offs[i]), &xs[i], pr, set, &G[i], &u)) return 2;
    }
    double dGdT = (G[5] - G[1]) / (0.02 * sol->T);
    *shift = fabs(M_PER_H * pr->n_Htot * (sol->u - pr->u_initial) / pr->Delta_t) / (fabs(dGdT) * sol->T);
    /* and the thermal relaxation time there must be short: near a tangency of G there is no
       nearby root however small the backward-Euler term */
    double t_relax = M_PER_H * pr->n_Htot * (sol->u / sol->T) / fabs(dGdT);
    if (t_relax > 1e-2 * pr->Delta_t) *shift = HUGE_VAL;
    *dist = HUGE_VAL;
    for (int i = 0; i < 6; i++) {
        if ((G[i] > 0) == (G[i + 1] > 0) && G[i] != 0) continue;
        double lo = log(sol->T * (1 + offs[i])), hi = log(sol->T * (1 + offs[i + 1])), Glo = G[i];
        SolveVars xm = xs[i];
        for (int bsc = 0; bsc < 40 && hi - lo > 1e-8; bsc++) {
            double mid = 0.5 * (lo + hi), Gm;
            if (G_of_T(exp(mid), &xm, pr, set, &Gm, &u)) return 2;
            if ((Gm > 0) == (Glo > 0)) {
                lo = mid;
                Glo = Gm;
            } else
                hi = mid;
        }
        *dist = fmin(*dist, fabs(exp(0.5 * (lo + hi)) / sol->T - 1));
    }
    return *dist <= EQ_TOL ? 0 : 1;
}

/* ---- statistics ---- */

struct Stats {
    long n = 0, tier[5] = {0, 0, 0, 0, 0}, fail = 0, check_fail = 0, resync = 0, pinned = 0;
    long nanjac_cells = 0, nanjac_evals = 0, nanF_cells = 0, fail_nanjac = 0, fail_nanF = 0, fd_evals = 0;
    long t1hist[6] = {0, 0, 0, 0, 0, 0}; /* tier-1 answers by nfeval: 1,2,3,4,5-8,>8 */
    long eq_gated = 0, eq_match = 0, eq_mismatch = 0, eq_ref_fail = 0;
    long cat[8] = {0, 0, 0, 0, 0, 0, 0, 0}; /* check failures: energy, species 2..5, bounds/EOS, other */
    long t1fail[6] = {0, 0, 0, 0, 0, 0};   /* tier-1 failure reasons, by -status */
    double worst = 0, seconds = 0;
    std::vector<int> nfeval;
    void add(int rc, const JacoSolveInfo &info) {
        n++;
        if (rc) fail++;
        tier[info.tier]++;
        resync += info.resynced_T0;
        pinned += info.pinned;
        nanjac_cells += info.n_nonfinite_jac > 0;
        nanjac_evals += info.n_nonfinite_jac;
        nanF_cells += info.n_nonfinite_F > 0;
        if (rc) {
            fail_nanjac += info.n_nonfinite_jac > 0;
            fail_nanF += info.n_nonfinite_F > 0;
        }
        nfeval.push_back(info.nfeval);
        fd_evals += info.nfeval_fd;
        if (info.tier1_status < 0 && info.tier1_status >= -5) t1fail[-info.tier1_status]++;
        if (info.tier == JACO_TIER_NEWTON) {
            int f = info.nfeval - info.nfeval_fd; /* evaluations the algorithm asked for, not FD columns */
            t1hist[f <= 4 ? f - 1 : (f <= 8 ? 4 : 5)]++;
        }
    }
    int pct(double p) {
        if (nfeval.empty()) return 0;
        std::vector<int> v = nfeval;
        std::sort(v.begin(), v.end());
        return v[(size_t)fmin(v.size() - 1, floor(p * v.size()))];
    }
    void print(const char *name) {
        double mean = 0;
        for (int f : nfeval) mean += f;
        mean /= fmax(1, (double)nfeval.size());
        printf("%-10s %7ld | tier1 %7ld (%5.1f%%) tier2 %6ld tier3 %4ld FAILED %ld | check failures %ld (worst %.2g of tol)\n", name, n,
               tier[1], 100.0 * tier[1] / fmax(1, n), tier[2], tier[3], fail, check_fail, worst);
        printf("%-10s nfeval mean %.2f p50 %d p90 %d p99 %d max %d | tier-1 evals excl. FD 1:%ld 2:%ld 3:%ld 4:%ld 5-8:%ld >8:%ld\n", "", mean,
               pct(0.5), pct(0.9), pct(0.99), pct(1.0), t1hist[0], t1hist[1], t1hist[2], t1hist[3], t1hist[4], t1hist[5]);
        printf("%-10s T0 resynced %ld, pinned %ld, %.1f us/solve\n", "", resync, pinned, 1e6 * seconds / fmax(1, n));
        printf("%-10s cells meeting a non-finite generated Jacobian %ld (%ld evaluations; FD columns are %.0f%% of all fevals), non-finite residual %ld\n", "",
               nanjac_cells, nanjac_evals, 100.0 * fd_evals / fmax(1.0, mean * (double)nfeval.size()), nanF_cells);
        if (fail)
            printf("%-10s of the %ld failed cells: %ld met a non-finite Jacobian, %ld a non-finite residual\n", "", fail, fail_nanjac,
                   fail_nanF);
        printf("%-10s tier-1 failures: non-finite %ld, singular %ld, line search %ld, iteration budget %ld, below floor %ld\n", "",
               t1fail[1], t1fail[2], t1fail[3], t1fail[4], t1fail[5]);
        if (check_fail)
            printf("%-10s check failures by row: energy %ld, x_H+ %ld, x_He+ %ld, x_He++ %ld, x_H2 %ld, bounds/EOS %ld\n", "", cat[0], cat[1],
                   cat[2], cat[3], cat[4], cat[6]);
        if (eq_gated || eq_ref_fail)
            printf("%-10s equilibrium: %ld gated, %ld within %.0f%%, %ld MISMATCHED, %ld reference failures\n", "", eq_gated, eq_match,
                   100 * EQ_TOL, eq_mismatch, eq_ref_fail);
    }
};

/* failure dumps: at most a few per category, so that one systematic problem cannot hide others */
static int report_quota = 4;
static const char *rep_cat[32];
static int rep_n[32], n_cats = 0;

static void report_failure(const char *what, const SolveVars *in, const Params *pr, const JacoSolverSettings *set, const char *why) {
    char cat[24];
    int len = 0; /* category = the reason up to its first digit or '(' */
    while (why[len] && len < 23 && !(why[len] >= '0' && why[len] <= '9') && why[len] != '(' && why[len] != '=') len++;
    snprintf(cat, sizeof(cat), "%.*s", len, why);
    int c = 0;
    while (c < n_cats && strcmp(rep_cat[c], cat)) c++;
    if (c == n_cats) {
        if (n_cats == 32) return;
        rep_cat[n_cats] = strdup(cat);
        rep_n[n_cats++] = 0;
    }
    if (report_quota > 0) printf("FAILED CASE (%s): %s\n", what, why);
    if (rep_n[c]++ >= report_quota) return;
    printf("FAIL (%s): %s\n", what, why);
    jaco_print_state(stdout, "  input state:", in, pr);
    JacoSolverSettings v = *set;
    v.verbose = 1;
    SolveVars sv = *in;
    JacoSolveInfo info;
    jaco_solve(&sv, pr, &v, &info);
}

static double now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* solve + check; returns the solver rc, sets *bad if the check failed */
static int solve_checked(SolveVars *sv, const Params *pr, const JacoSolverSettings *set, Stats &st, const char *what,
                         JacoSolveInfo *info) {
    SolveVars in = *sv;
    double t0 = now();
    int rc = jaco_solve(sv, pr, set, info);
    st.seconds += now() - t0;
    st.add(rc, *info);
    char why[256];
    if (rc) {
        report_failure(what, &in, pr, set, "every tier failed");
        return rc;
    }
    double worst;
    if (check_answer(sv, pr, set, why, &worst)) {
        st.check_fail++;
        if (!strncmp(why, "energy", 6))
            st.cat[0]++;
        else if (!strncmp(why, "species [", 9))
            st.cat[why[9] - '0' - 1]++;
        else
            st.cat[6]++;
        report_failure(what, &in, pr, set, why);
    }
    st.worst = fmax(st.worst, worst);
    return 0;
}

/* ---- replays ---- */

struct Replay {
    const char *name;
    double T0, n, dt, xHp, xHep, xHepp, xH2, u_factor, N_H, pdv;
};

static int run_replays(const JacoSolverSettings *set) {
    /* smoke-run cells; u_factor multiplies u(T0,x0) to give u_initial (1.01374 is the logged
       u_initial = 9.6314e10 against u(T0,x0) = 9.5009e10). xHp < 0: CIE seeds. N_H < 0: n Delta_x.
       pdv = -1e-22..-1e-21 (strong expansion cooling) reproduces the legacy solver's endrun. The last
       rows probe the model's known non-finite points: cold gas, and x_H+ = 1 exactly. */
    const Replay cases[] = {
        {"992K consistent", 992.239, 219.081, 4.82812e11, 2.7042e-3, 1e-20, 1e-20, 1.1217e-4, 1.0, -1, 0},
        {"992K u0 +1.4%", 992.239, 219.081, 4.82812e11, 2.7042e-3, 1e-20, 1e-20, 1.1217e-4, 1.01374, -1, 0},
        {"992K +1.4% pdv-", 992.239, 219.081, 4.82812e11, 2.7042e-3, 1e-20, 1e-20, 1.1217e-4, 1.01374, 1e22, -1e-22},
        {"102K dt=3597s", 102.292, 246.838, 3597.23, -1, 0, 0, 1.534e-4, 1.0, 1e22, -1e-22},
        {"102K dt=4.8e11", 102.292, 246.838, 4.82812e11, -1, 0, 0, 1.534e-4, 1.0, 1e22, -1e-22},
        {"97K stuck state", 96.9555, 246.838, 28.1034, 3.413726e-06, 1e-20, 1e-20, 1.534010e-04, 1.0, 1e22, -1e-22},
        {"745K legacy endrun", 744.921, 246.838, 4.82812e11, 2.7e-3, 1e-20, 1e-20, 1e-4, 1.0, 1e22, -1e-22},
        {"745K pdv=-1e-21", 744.921, 246.838, 4.82812e11, 2.7e-3, 1e-20, 1e-20, 1e-4, 1.0, 1e22, -1e-21},
        {"10K n=1e3", 10.0, 1e3, 4.82812e11, -1, 0, 0, 0.3, 1.0, -1, 0},
        {"10K n=1e5", 10.0, 1e5, 1e10, -1, 0, 0, 0.45, 1.0, -1, 0},
        {"x_H+=1 1e4K", 1e4, 1.0, 1e11, 1.0, 1e-20, 0.0994, 1e-20, 1.0, -1, 0},
        {"x_H+=1 1e6K", 1e6, 1.0, 1e11, 1.0, 1e-20, 0.0994, 1e-20, 1.0, -1, 0},
        {"x_H+=1 100K n=1e4", 100.0, 1e4, 1e11, 1.0, 1e-20, 0.0994, 1e-20, 1.0, -1, 0},
        /* collisionally ionized gas whose cached ions sit near the neutral fixed point of the
           ionization balance (x_e -> 0); the answer must be the ionized balance */
        {"near-neutral 3e4K", 3e4, 0.15, 1e10, 1e-4, 1e-20, 1e-20, 1e-6, 1.0, -1, 0},
        {"near-neutral 3e4K 1e12", 3e4, 0.15, 1e12, 1e-4, 1e-20, 1e-20, 1e-6, 1.0, -1, 0},
        {"floored ions 2e4K", 2e4, 0.1, 1e11, 1e-20, 1e-20, 1e-20, 1e-6, 1.0, -1, 0},
        {"floored ions 5e4K", 5e4, 0.2, 1e11, 1e-20, 1e-20, 1e-20, 1e-6, 1.0, -1, 0},
        /* the H- rate fits branch at 6000 K */
        {"6000K n=1", 6000.0, 1.0, 1e11, -1, 0, 0, 1e-6, 1.0, -1, 0},
        {"6000K n=1e3", 6000.0, 1e3, 1e9, -1, 0, 0, 1e-4, 1.0, -1, 0},
    };
    int nfail = 0;
    /* stale-Ne cell as gizmo_to_jaco treats it: u from the cached near-neutral ions at T0, ions
       re-seeded from CIE at T0, T0 kept, so u(T0, x0) exceeds u_initial */
    {
        const double T0s[] = {2e4, 3e4, 5e4};
        printf("\n== stale-Ne cells (u from x_H+ = 1e-4 at T0, CIE ions seeded at T0) ==\n");
        for (double T0 : T0s) {
            Params pr;
            starforge_params(&pr, 0.15, 1e11);
            SolveVars sv = {};
            seed(&sv, T0, 1e-6);
            SolveVars neutral = sv;
            neutral.x_Hplus = 1e-4;
            neutral.x_Heplus = neutral.x_Heplusplus = 1e-20;
            pr.u_initial = jaco_T_to_u(T0, &neutral, &pr, NULL);
            sv.u = pr.u_initial;
            pr.x_H_2_initial = sv.x_H_2;
            SolveVars in = sv;
            JacoSolveInfo info;
            int rc = jaco_solve(&sv, &pr, set, &info);
            char why[256] = "ok";
            double worst = 0;
            int bad = rc ? 1 : check_answer(&sv, &pr, set, why, &worst);
            printf("T0=%g u(T0,x_CIE)/u0=%.3f: tier %d nfeval %d T=%.5g x_H+=%.3g %s\n", T0, jaco_T_to_u(T0, &in, &pr, NULL) / pr.u_initial,
                   info.tier, info.nfeval, sv.T, sv.x_Hplus, bad ? why : "ok");
            if (bad) {
                nfail++;
                report_failure("stale Ne", &in, &pr, set, rc ? "every tier failed" : why);
            }
        }
    }
    printf("\n== replays ==\n");
    printf("%-20s %9s %9s %10s | %4s %6s %10s %9s %9s %5s %5s %s\n", "case", "T0", "n", "dt", "tier", "nfeval", "T", "x_H2", "x_H+",
           "nanJ0", "nanF0", "check");
    for (const Replay &c : cases) {
        Params pr;
        starforge_params(&pr, c.n, c.dt);
        if (c.N_H > 0) pr.N_H = c.N_H;
        pr.pdv_work = c.pdv;
        SolveVars sv = {};
        if (c.xHp < 0)
            seed(&sv, c.T0, c.xH2);
        else {
            sv.T = c.T0;
            sv.x_Hplus = c.xHp;
            sv.x_Heplus = c.xHep;
            sv.x_Heplusplus = c.xHepp;
            sv.x_H_2 = c.xH2;
        }
        pr.u_initial = c.u_factor * jaco_T_to_u(c.T0, &sv, &pr, NULL);
        sv.u = pr.u_initial;
        pr.x_H_2_initial = sv.x_H_2;
        /* non-finite entries of the generated residual and Jacobian at the raw starting state */
        SolveVars F;
        double J[N_VARS][N_VARS];
        microphysics_func_jac(&sv, &pr, &F, J);
        int nan_j = 0, nan_f = 0;
        for (int i = 0; i < N_VARS; i++) {
            nan_f += !isfinite(F.data[i]);
            for (int j = 0; j < N_VARS; j++) nan_j += !isfinite(J[i][j]);
        }
        SolveVars in = sv;
        JacoSolveInfo info;
        int rc = jaco_solve(&sv, &pr, set, &info);
        char why[256] = "ok";
        double worst = 0;
        int bad = rc ? 1 : check_answer(&sv, &pr, set, why, &worst);
        if (rc) strcpy(why, "every tier failed");
        printf("%-20s %9.4g %9.4g %10.4g | %4d %6d %10.5g %9.3g %9.3g %5d %5d %s%s (nan J evals %d)\n", c.name, c.T0, c.n, c.dt,
               info.tier, info.nfeval, sv.T, sv.x_H_2, sv.x_Hplus, nan_j, nan_f, bad ? "FAIL: " : "", why, info.n_nonfinite_jac);
        if (bad) {
            nfail++;
            report_failure(c.name, &in, &pr, set, why);
        }
    }
    return nfail;
}

/* ---- sweep ---- */

static const double sweep_n[] = {1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
static const double sweep_dt[] = {1e3, 1e5, 1e7, 1e9, 1e11, 1e13};
static const double DT_EQ = 1e20; /* effectively infinite: the answer must be an equilibrium */
static const double sweep_ufac[] = {1.0, 1.1, 0.9};
static const int sweep_pdv_sign[] = {0, 1, -1};

/* |pdv_work| ~ the net heating/cooling rate at T0 with the seeds (floored at 1e-27 n^2) */
static double sweep_pdv(double n, double T0, int variant, int sign) {
    if (!sign) return 0;
    Params p0;
    starforge_params(&p0, n, 1e10);
    SolveVars s0 = {};
    seed_variant(&s0, T0, variant, p0.y);
    s0.x_Hplus = fmin(s0.x_Hplus, 1 - 1e-10); /* x_H+ = 1 exactly is a known 0/0 in the model */
    s0.u = p0.u_initial = jaco_T_to_u(T0, &s0, &p0, NULL);
    p0.x_H_2_initial = s0.x_H_2;
    SolveVars F;
    double J[N_VARS][N_VARS];
    microphysics_func_jac(&s0, &p0, &F, J);
    double r = isfinite(F.T) ? fabs(F.T) : 0;
    return sign * fmax(r, 1e-27 * n * n);
}

/* full = 1: checks, equilibrium references and warm re-solves; full = 0: solve and count only */
static void run_sweep(const JacoSolverSettings *set, int full, Stats &sweep, Stats &warm, Stats &eq) {
    std::vector<double> T0s = {3.0};
    for (int i = 2; i <= 14; i++) T0s.push_back(pow(10., 0.5 * i));
    T0s.push_back(6000.0); /* branch point of the H- rate fits */
    for (double n : sweep_n)
        for (int variant = 0; variant < N_SEEDS; variant++)
            for (double T0 : T0s)
                for (int ps : sweep_pdv_sign) {
                    double pdv = sweep_pdv(n, T0, variant, ps);
                    for (int idt = 0; idt <= 6; idt++) {
                        double dt = idt < 6 ? sweep_dt[idt] : DT_EQ;
                        Params pr;
                        starforge_params(&pr, n, dt);
                        pr.pdv_work = pdv;
                        SolveVars s0 = {};
                        seed_variant(&s0, T0, variant, pr.y);
                        pr.x_H_2_initial = s0.x_H_2;
                        int check_eq = full && dt >= 1e13;
                        for (double uf : sweep_ufac) {
                            SolveVars sv = s0;
                            SolveVars s_eos = s0;
                            s_eos.x_Hplus = fmin(s_eos.x_Hplus, 1 - 1e-10);
                            pr.u_initial = uf * jaco_T_to_u(T0, &s_eos, &pr, NULL);
                            sv.u = pr.u_initial;
                            Stats &st = (idt < 6) ? sweep : eq;
                            char what[200];
                            snprintf(what, sizeof(what), "n=%g T0=%g dt=%g pdv=%g ufac=%g seed=%s", n, T0, dt, pdv, uf, seed_name[variant]);
                            JacoSolveInfo info;
                            if (!full) {
                                SolveVars in = sv;
                                double t0 = now();
                                int rc = jaco_solve(&sv, &pr, set, &info);
                                st.seconds += now() - t0;
                                st.add(rc, info);
                                (void)in;
                                continue;
                            }
                            long fails_before = st.fail + st.check_fail;
                            if (solve_checked(&sv, &pr, set, st, what, &info)) continue;
                            if (check_eq && !info.pinned && st.fail + st.check_fail == fails_before) {
                                double shift, dist;
                                int r = near_root(&sv, &pr, set, &shift, &dist);
                                if (r == 2)
                                    st.eq_ref_fail++;
                                else if (shift < EQ_GATE) {
                                    st.eq_gated++;
                                    if (r == 0)
                                        st.eq_match++;
                                    else {
                                        st.eq_mismatch++;
                                        char why[200];
                                        snprintf(why, sizeof(why), "equilibrium: no root of heat+pdv within %.0f%% of T=%g (shift %.2g)",
                                                 100 * EQ_TOL, sv.T, shift);
                                        SolveVars in = s0;
                                        in.u = pr.u_initial;
                                        report_failure(what, &in, &pr, set, why);
                                    }
                                }
                            }
                            /* the next step of a settled cell: start from the answer with 1% more energy */
                            if (idt < 6) {
                                Params pw = pr;
                                pw.u_initial = 1.01 * sv.u;
                                pw.x_H_2_initial = sv.x_H_2;
                                SolveVars sw = sv;
                                JacoSolveInfo iw;
                                solve_checked(&sw, &pw, set, warm, what, &iw);
                            }
                        }
                    }
                }
}

int main(int argc, char **argv) {
    jaco_init_tables(argc > 1 ? argv[1] : ".");
    JacoSolverSettings set;
    jaco_solver_default_settings(&set);
    set.T_min = 2.73;   /* GIZMO's MinGasTemp; the model is not finite below ~1.6 K */
    set.T_max = 1e10;
    set.u_min = 2.75e8; /* GIZMO's MinEgySpec for MinGasTemp = 2.73 K */
    if (build_cie_table(&set)) return 1;

    if (argc > 3 && !strcmp(argv[2], "replay")) {
        /* re-solve states printed by jaco_print_state (e.g. a GIZMO failure dump):
           replay <file> [which (-1 = all)] [verbose] [jac: compare the generated Jacobian with central differences] */
        FILE *fp = fopen(argv[3], "r");
        if (!fp) return 1;
        int which = argc > 4 ? atoi(argv[4]) : -1, k = 0;
        set.verbose = argc > 5 ? atoi(argv[5]) : 0;
        char line[16384];
        SolveVars sv = {};
        Params pr = {};
        int have_sv = 0;
        while (fgets(line, sizeof(line), fp)) {
            double *dst = NULL;
            int n = 0;
            if (strstr(line, "SolveVars (")) dst = sv.data, n = N_VARS;
            if (strstr(line, "Params (")) dst = pr.data, n = N_PARAMS;
            if (!dst) continue;
            char *q = line;
            for (int i = 0; i < n; i++) {
                q = strchr(q, '=');
                if (!q) return 2;
                dst[i] = strtod(q + 1, &q);
            }
            if (dst == sv.data) {
                have_sv = 1;
                continue;
            }
            if (!have_sv) continue;
            have_sv = 0;
            if (which >= 0 && k++ != which) continue;
            if (argc > 6 && !strcmp(argv[6], "jac")) {
                /* generated Jacobian against central differences at the dumped state */
                SolveVars x = sv, F0;
                double J[N_VARS][N_VARS], Jp[N_VARS][N_VARS];
                x.u = jaco_T_to_u(x.T, &x, &pr, NULL);
                microphysics_func_jac(&x, &pr, &F0, J);
                for (int j = 1; j < N_VARS; j++) {
                    double h = 1e-6 * (j == IDX_T ? x.T : x.data[j]); /* relative, so abundances stay positive */
                    SolveVars xp = x, xm = x, Fp, Fm;
                    xp.data[j] += h;
                    xm.data[j] -= h;
                    microphysics_func_jac(&xp, &pr, &Fp, Jp);
                    microphysics_func_jac(&xm, &pr, &Fm, Jp);
                    for (int i = 1; i < N_VARS; i++) {
                        double fd = (Fp.data[i] - Fm.data[i]) / (2 * h);
                        printf("  J[%d][%d] gen %+.6e fd %+.6e rel %.2e\n", i, j, J[i][j], fd, fabs(J[i][j] - fd) / (fabs(fd) + 1e-300));
                    }
                }
            }
            SolveVars x = sv;
            JacoSolveInfo info;
            int rc = jaco_solve(&x, &pr, &set, &info);
            char why[256] = "ok";
            double worst = 0;
            if (!rc) check_answer(&x, &pr, &set, why, &worst);
            printf("replay n=%.4g T0=%.6g dt=%.4g pdv=%.3g: rc=%d tier=%d nfeval=%d (tier-1 status %d) T=%.8g xHp=%.4g xH2=%.4g check %s\n",
                   pr.n_Htot, sv.T, pr.Delta_t, pr.pdv_work, rc, info.tier, info.nfeval, info.tier1_status, x.T, x.x_Hplus, x.x_H_2, why);
        }
        return 0;
    }

    if (argc > 2 && !strcmp(argv[2], "gscan")) {
        /* G(T) = heat + pdv with steady-state chemistry on a fine grid: gscan n dt pdv seed Tlo Thi npts */
        double n = atof(argv[3]), dt = atof(argv[4]), pdv = atof(argv[5]);
        int variant = atoi(argv[6]);
        double Tlo = atof(argv[7]), Thi = atof(argv[8]);
        int np = atoi(argv[9]);
        Params pr;
        starforge_params(&pr, n, dt);
        pr.pdv_work = pdv;
        SolveVars x = {};
        seed_variant(&x, Thi, variant, pr.y);
        pr.x_H_2_initial = x.x_H_2;
        for (int i = 0; i < np; i++) {
            double T = Thi * pow(Tlo / Thi, (double)i / (np - 1)), G, u;
            int bad = G_of_T(T, &x, &pr, &set, &G, &u);
            printf("T=%.6e G=%+.6e bad=%d xHp=%.6e xHep=%.4e xHepp=%.4e xH2=%.4e\n", T, G, bad, x.x_Hplus, x.x_Heplus, x.x_Heplusplus, x.x_H_2);
        }
        return 0;
    }

    if (argc > 2 && !strcmp(argv[2], "case")) {
        /* one verbose solve: case n T0 dt pdv ufac seed [verbose] [warm] */
        double n = atof(argv[3]), T0 = atof(argv[4]), dt = atof(argv[5]), pdv = atof(argv[6]), uf = atof(argv[7]);
        int variant = atoi(argv[8]);
        set.verbose = argc > 9 ? atoi(argv[9]) : 2;
        int do_warm = argc > 10 ? atoi(argv[10]) : 0;
        Params pr;
        starforge_params(&pr, n, dt);
        pr.pdv_work = pdv;
        SolveVars sv = {};
        seed_variant(&sv, T0, variant, pr.y);
        pr.x_H_2_initial = sv.x_H_2;
        SolveVars s_eos = sv;
        s_eos.x_Hplus = fmin(s_eos.x_Hplus, 1 - 1e-10);
        pr.u_initial = uf * jaco_T_to_u(T0, &s_eos, &pr, NULL);
        sv.u = pr.u_initial;
        JacoSolveInfo info;
        JacoSolverSettings quiet = set;
        if (do_warm) quiet.verbose = 0;
        int rc = jaco_solve(&sv, &pr, &quiet, &info);
        if (do_warm && !rc) {
            pr.u_initial = 1.01 * sv.u;
            pr.x_H_2_initial = sv.x_H_2;
            rc = jaco_solve(&sv, &pr, &set, &info);
        }
        char why[256] = "ok";
        double worst = 0;
        if (!rc) check_answer(&sv, &pr, &set, why, &worst);
        printf("rc=%d tier=%d nfeval=%d nonfinite_jac=%d pinned=%d T=%.9g xHp=%.4g xHep=%.4g xHepp=%.4g xH2=%.4g check: %s\n", rc, info.tier,
               info.nfeval, info.n_nonfinite_jac, info.pinned, sv.T, sv.x_Hplus, sv.x_Heplus, sv.x_Heplusplus, sv.x_H_2, why);
        return 0;
    }

    int nfail = run_replays(&set);

    Stats sweep, warm, eq;
    printf("\n== sweep ==\n");
    run_sweep(&set, 1, sweep, warm, eq);

    /* the same sweep without finite-difference repair of non-finite generated Jacobians, to show
       which cells depend on it */
    JacoSolverSettings nofd = set;
    nofd.fd_jacobian = 0;
    Stats sweep_nofd, warm_nofd, eq_nofd;
    report_quota = 0; /* no failure dumps for this pass */
    run_sweep(&nofd, 0, sweep_nofd, warm_nofd, eq_nofd);

    printf("\n== summary (solver tol %g, residual check at %g; %zu n x %d seeds x 15 T0 x 3 pdv x 3 u0) ==\n", set.tol, CHECK_TOL,
           sizeof(sweep_n) / sizeof(sweep_n[0]), (int)N_SEEDS);
    sweep.print("sweep");
    warm.print("warm");
    eq.print("dt=1e20");
    printf("-- without FD repair of non-finite Jacobians (informational) --\n");
    sweep_nofd.print("sweep");
    eq_nofd.print("dt=1e20");
    long bad = sweep.fail + sweep.check_fail + sweep.eq_mismatch + warm.fail + warm.check_fail + eq.fail + eq.check_fail +
               eq.eq_mismatch + sweep.eq_ref_fail + eq.eq_ref_fail + nfail;
    printf("\n%s: %ld failures (replays %d)\n", bad ? "FAILED" : "PASSED", bad, nfail);
    return bad ? 1 : 0;
}
