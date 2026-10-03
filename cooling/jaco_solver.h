/*
 * jaco_solver.h -- implicit (backward-Euler) solver for the jaco code-generated microphysics network.
 *
 * Pure numerics: depends only on the generated RHS/Jacobian (microphysics_func_jac) and EOS
 * (jaco_T_to_u, jaco_u_to_T), never on GIZMO globals, so it can be driven standalone
 * (test/jaco_solver).
 */
#pragma once
#include <stdio.h>
#include "microphysics_func_jac.h"

#ifndef JACO_ABUNDANCE_FLOOR
#define JACO_ABUNDANCE_FLOOR 1e-20 /* seed floor for host code; the solver clamps at the generated per-species floors */
#endif

/* from the generated jaco_eos.cc / jaco_util.cc */
double jaco_T_to_u(double T, const SolveVars *sv, const Params *pr, double *cv_out);
double jaco_u_to_T(double u, const SolveVars *sv, const Params *pr);
extern "C" int jaco_isfinite(double x);

struct JacoSolverSettings {
    double T_min;  /* temperature floor [K] */
    double T_max;  /* temperature ceiling [K] */
    double u_min;  /* specific internal energy floor [erg/g] */
    double tol;    /* relative convergence tolerance on the solution (step and scaled residual) */
    int fd_jacobian; /* 1: replace non-finite generated Jacobian columns by finite differences of F;
                        0: a non-finite Jacobian fails the tier like a non-finite residual */
    int verbose;   /* 0 silent; 1 report tier escalations; 2 also trace every Newton iterate */
};

enum JacoTier { JACO_TIER_NEWTON = 1, JACO_TIER_ROOTFIND = 2, JACO_TIER_SUBCYCLE = 3, JACO_TIER_FAILED = 4 };

struct JacoSolveInfo {
    int tier;          /* JacoTier that produced the answer */
    int nfeval;        /* microphysics_func_jac calls, all tiers, including finite-difference columns */
    int nfeval_fd;     /* of which finite-difference columns */
    int nfeval_tier1;  /* of which spent in the first tier-1 attempt */
    int tier1_status;  /* outcome of that attempt: 0 converged, -1 non-finite residual, -2 singular,
                          -3 line search, -4 iteration budget, -5 converged below the energy floor */
    int n_nonfinite_jac; /* evaluations whose generated Jacobian had non-finite entries (repaired if fd_jacobian) */
    int n_nonfinite_F;   /* evaluations whose generated residual was non-finite (that step or trial is rejected) */
    int resynced_T0;   /* tier 0 re-derived the starting T from u_initial */
    int pinned;        /* the temperature is pinned at its floor or ceiling */
    int n_substeps;    /* tier-3 substeps taken */
};

void jaco_solver_default_settings(struct JacoSolverSettings *set);

/* Advance (sv, pr) over pr->Delta_t. On entry sv holds the starting guess (T and species seeds)
   and pr->u_initial the energy at the start of the step; on return sv holds the solution, with
   sv->u = u(T, x) exactly. Returns 0 on success, nonzero if every tier failed (sv then holds the
   starting state). info may be NULL. */
int jaco_solve(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, struct JacoSolveInfo *info);

/* Set the start-of-step values of the time-dependent species (pr->x_<name>_initial) to their abundances in sv. */
void jaco_initial_from_state(const SolveVars *sv, Params *pr);

/* Solve the chemistry rows (indices >= 2) at the fixed temperature sv->T, starting from the
   abundances in sv; sets sv->u = u(T, x). Returns 0 on success. *nfeval is incremented. */
int jaco_solve_chemistry(SolveVars *sv, const Params *pr, const struct JacoSolverSettings *set, int *nfeval);

/* Print every solve variable and parameter (by IDX_ and PARAM_ index) for failure diagnostics. */
void jaco_print_state(FILE *fp, const char *label, const SolveVars *sv, const Params *pr);

/* Print the model's outputs (microphysics_outputs) by name and units. */
void jaco_print_outputs(FILE *fp, const char *label, const Outputs *out);
