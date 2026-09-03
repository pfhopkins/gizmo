#ifndef ANEOS_H
#define ANEOS_H

#include "../GIZMO_config.h"

#ifdef EOS_ANEOS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Device annotation, derived here rather than assumed from an includer. Every
   real include path reaches this header after declarations/macros.h, but a
   header whose bodies only compile when something else was included first is a
   defect this module has been bitten by before, so the two-line derivation is
   repeated verbatim from macros.h. The GPU-compiler fallback must be
   host+device: a host-only `inline` here would be re-declared host+device once
   Kokkos redefines the macro, which clang rejects. */
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIZMO_GPU_COMPILER)
#define GIZMO_GPU_COMPILER
#endif
#ifndef KOKKOS_INLINE_FUNCTION
#if defined(GIZMO_GPU_COMPILER)
#define KOKKOS_INLINE_FUNCTION __host__ __device__ inline
#else
#define KOKKOS_INLINE_FUNCTION inline
#endif
#endif

#ifndef ANEOS_MAX_MATERIALS
#define ANEOS_MAX_MATERIALS 7
#endif

/* Storage for one ANEOS material table (SESAME-format, pre-computed rho-T grid) */
struct aneos_table {
    int    mat_id;        /* SESAME material number */
    int    nrho;          /* number of density grid points */
    int    nT;            /* number of temperature grid points */
    int    loaded;        /* 0 = empty slot, 1 = loaded */

    /* Grid axes in log10 space (CGS units: rho in g/cm^3, T in K) */
    double *logrho;       /* log10(rho), length nrho */
    double *logT;         /* log10(T),   length nT   */

    /* Tabulated 2D arrays, row-major [irho * nT + iT], stored in log10 of CGS values */
    double *log_pressure; /* log10(P [dyn/cm^2]) */
    double *log_energy;   /* log10(u [erg/g])    */
    double *log_entropy;  /* log10(S [erg/g/K])  */
    double *log_csound;   /* log10(cs [cm/s])    */
    int    *phase;        /* phase flag (integer, no interpolation) */
    int     has_csound;   /* 1 if sound speed column present in table */
    int     has_phase;    /* 1 if phase column present in table */

    /* Grid bounds and spacing */
    double logrho_min, logrho_max;
    double logT_min, logT_max;
    double d_logrho;      /* uniform spacing in log10(rho) */
    double d_logT;        /* uniform spacing in log10(T)   */
    double inv_d_logrho;  /* 1/d_logrho for fast indexing */
    double inv_d_logT;    /* 1/d_logT   for fast indexing */
};

/* Global table array -- one per material slot.
   The descriptors AND every array they point at live in memory both the host
   and the device can read, allocated once by aneos_read_table and never copied
   afterwards: the lookups below run inside a per-particle Newton iteration, so
   a table that had to be staged per call would cost more to move than the
   solve it serves. This pointer itself is an ordinary host global -- device
   code reaches the descriptors through the pointer it is handed, never through
   this symbol. */
extern struct aneos_table *ANEOS_Tables;
extern int ANEOS_Num_Tables_Loaded;

/* ---- Table lifecycle: host only (file I/O), in aneos.cc ---- */

/* Read a SESAME-format table file into slot mat_index.
   Call once per material from the main thread at startup. */
int aneos_read_table(const char *filename, int mat_index);

/* Free all allocated table memory. */
int aneos_cleanup(void);

/* ---- Lookups: device-callable, whole bodies below ----
   Each takes the descriptor array explicitly. The host entry points at the
   bottom of this header supply the global; a kernel supplies the pointer it
   was given, because a device body that reached for the global would read the
   wrong memory with nothing to say so. */

/* Bilinear interpolation on a uniform log-log grid.
   Returns interpolated value of log10(Q) at (lr, lT).
   The table stores log10 of the quantity in log_qty[irho * nT + iT]. */
static KOKKOS_INLINE_FUNCTION double aneos_bilinear_interp(const struct aneos_table *tbl, const double *log_qty,
                               double lr, double lT)
{
    /* Clamp to table bounds */
    if(lr < tbl->logrho_min) lr = tbl->logrho_min;
    if(lr > tbl->logrho_max) lr = tbl->logrho_max;
    if(lT < tbl->logT_min) lT = tbl->logT_min;
    if(lT > tbl->logT_max) lT = tbl->logT_max;

    /* Find cell indices */
    double fi = (lr - tbl->logrho_min) * tbl->inv_d_logrho;
    double fj = (lT - tbl->logT_min) * tbl->inv_d_logT;
    int i = (int)fi;
    int j = (int)fj;

    /* Clamp to valid cell range */
    if(i < 0) i = 0;
    if(i >= tbl->nrho - 1) i = tbl->nrho - 2;
    if(j < 0) j = 0;
    if(j >= tbl->nT - 1) j = tbl->nT - 2;

    /* Local coordinates in [0,1] */
    double t = fi - i;
    double u = fj - j;
    if(t < 0.0) t = 0.0; if(t > 1.0) t = 1.0;
    if(u < 0.0) u = 0.0; if(u > 1.0) u = 1.0;

    /* Four corner values */
    int nT = tbl->nT;
    double f00 = log_qty[i * nT + j];
    double f10 = log_qty[(i+1) * nT + j];
    double f01 = log_qty[i * nT + (j+1)];
    double f11 = log_qty[(i+1) * nT + (j+1)];

    return (1.0 - t) * (1.0 - u) * f00 + t * (1.0 - u) * f10
         + (1.0 - t) * u * f01 + t * u * f11;
}

/* Nearest-neighbor phase lookup */
static KOKKOS_INLINE_FUNCTION int aneos_phase_lookup(const struct aneos_table *tbl, double lr, double lT)
{
    if(!tbl->has_phase) return 0;

    if(lr < tbl->logrho_min) lr = tbl->logrho_min;
    if(lr > tbl->logrho_max) lr = tbl->logrho_max;
    if(lT < tbl->logT_min) lT = tbl->logT_min;
    if(lT > tbl->logT_max) lT = tbl->logT_max;

    double fi = (lr - tbl->logrho_min) * tbl->inv_d_logrho;
    double fj = (lT - tbl->logT_min) * tbl->inv_d_logT;
    int i = (int)(fi + 0.5);
    int j = (int)(fj + 0.5);
    if(i < 0) i = 0; if(i >= tbl->nrho) i = tbl->nrho - 1;
    if(j < 0) j = 0; if(j >= tbl->nT) j = tbl->nT - 1;
    return tbl->phase[i * tbl->nT + j];
}

/* Internal helper: get log10(u) at given (log10_rho, log10_T) */
static KOKKOS_INLINE_FUNCTION double aneos_get_log_u(const struct aneos_table *tbl, double lr, double lT)
{
    return aneos_bilinear_interp(tbl, tbl->log_energy, lr, lT);
}

/* Internal helper: get du/dT numerically at given (log10_rho, log10_T).
   Returns du/dT in CGS (erg/g/K). Uses centered finite difference on the
   bilinear interpolant in log-log space. */
static KOKKOS_INLINE_FUNCTION double aneos_get_du_dT(const struct aneos_table *tbl, double lr, double lT)
{
    double dlT = 1.0e-4; /* step in log10(T) */
    double lT_lo = lT - dlT;
    double lT_hi = lT + dlT;

    /* Clamp to bounds */
    if(lT_lo < tbl->logT_min) { lT_lo = tbl->logT_min; lT_hi = lT_lo + 2.0 * dlT; }
    if(lT_hi > tbl->logT_max) { lT_hi = tbl->logT_max; lT_lo = lT_hi - 2.0 * dlT; }

    double log_u_lo = aneos_bilinear_interp(tbl, tbl->log_energy, lr, lT_lo);
    double log_u_hi = aneos_bilinear_interp(tbl, tbl->log_energy, lr, lT_hi);

    double u_lo = pow(10.0, log_u_lo);
    double u_hi = pow(10.0, log_u_hi);
    double T_lo = pow(10.0, lT_lo);
    double T_hi = pow(10.0, lT_hi);

    double du_dT = (u_hi - u_lo) / (T_hi - T_lo);
    return du_dT;
}

/* Internal helper: get dP/dT numerically */

static KOKKOS_INLINE_FUNCTION double aneos_get_dP_dT(const struct aneos_table *tbl, double lr, double lT)
{
    double dlT = 1.0e-4;
    double lT_lo = lT - dlT;
    double lT_hi = lT + dlT;
    if(lT_lo < tbl->logT_min) { lT_lo = tbl->logT_min; lT_hi = lT_lo + 2.0 * dlT; }
    if(lT_hi > tbl->logT_max) { lT_hi = tbl->logT_max; lT_lo = lT_hi - 2.0 * dlT; }

    double P_lo = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_pressure, lr, lT_lo));
    double P_hi = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_pressure, lr, lT_hi));
    double T_lo = pow(10.0, lT_lo);
    double T_hi = pow(10.0, lT_hi);
    return (P_hi - P_lo) / (T_hi - T_lo);
}

/* ---- Direct (rho, T) lookup ---- */

KOKKOS_INLINE_FUNCTION
int aneos_lookup_rhoT_P(const struct aneos_table *tables, int mat_index,
                      double rho_cgs, double T_cgs,
                      double *press_cgs, double *u_cgs,
                      double *entropy_cgs, double *cs_cgs,
                      int *phase_out)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) return 1;
    const struct aneos_table *tbl = &tables[mat_index];
    if(!tbl->loaded) return 1;

    double lr = log10(rho_cgs > 0 ? rho_cgs : 1e-30);
    double lT = log10(T_cgs > 0 ? T_cgs : 1.0);

    if(press_cgs)   *press_cgs   = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_pressure, lr, lT));
    if(u_cgs)       *u_cgs       = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_energy, lr, lT));
    if(entropy_cgs) *entropy_cgs = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_entropy, lr, lT));
    if(cs_cgs) {
        if(tbl->has_csound)
            *cs_cgs = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_csound, lr, lT));
        else
            *cs_cgs = 0;
    }
    if(phase_out) *phase_out = aneos_phase_lookup(tbl, lr, lT);

    return 0;
}

KOKKOS_INLINE_FUNCTION
int aneos_compute_P(const struct aneos_table *tables, int mat_index,
                  double rho_cgs, double u_cgs,
                  double *T_guess,
                  double *press_cgs,
                  double *cs_cgs,
                  double *entropy_cgs,
                  double *cv_cgs,
                  double *gruneisen,
                  int    *phase_out)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) return 1;
    const struct aneos_table *tbl = &tables[mat_index];
    if(!tbl->loaded) return 1;

    double lr = log10(rho_cgs > 0 ? rho_cgs : 1e-30);

    /* Clamp log(rho) to table */
    if(lr < tbl->logrho_min) lr = tbl->logrho_min;
    if(lr > tbl->logrho_max) lr = tbl->logrho_max;

    double u_target = u_cgs;
    if(u_target <= 0) u_target = 1e-30;

    /* --- Newton-Raphson to invert u(rho, T) = u_target for T --- */
    double lT, T;
    int converged = 0;
    int max_iter = 50;
    double tol = 1.0e-8;

    /* Initial guess from T_guess (cached particle temperature) */
    T = (T_guess && *T_guess > 0) ? *T_guess : 1000.0;
    lT = log10(T);

    /* Clamp initial guess to table */
    if(lT < tbl->logT_min) lT = tbl->logT_min;
    if(lT > tbl->logT_max) lT = tbl->logT_max;

    /* Bracketing for bisection fallback */
    double lT_lo = tbl->logT_min;
    double lT_hi = tbl->logT_max;

    /* Check bracket: evaluate u at the endpoints */
    double u_lo = pow(10.0, aneos_get_log_u(tbl, lr, lT_lo));
    double u_hi = pow(10.0, aneos_get_log_u(tbl, lr, lT_hi));

    /* If u_target is outside the table range, clamp */
    if(u_target <= u_lo) {
        lT = lT_lo; converged = 1;
    } else if(u_target >= u_hi) {
        lT = lT_hi; converged = 1;
    }

    for(int iter = 0; iter < max_iter && !converged; iter++) {
        double log_u_here = aneos_get_log_u(tbl, lr, lT);
        double u_here = pow(10.0, log_u_here);
        double resid = u_here - u_target;

        /* Check convergence */
        if(fabs(resid) < tol * fabs(u_target)) { converged = 1; break; }

        /* Update bracket */
        if(resid < 0) { lT_lo = lT; } else { lT_hi = lT; }

        /* Newton step: du/dT */
        double du_dT = aneos_get_du_dT(tbl, lr, lT);
        double lT_new;

        if(du_dT > 0 && fabs(du_dT) > 1e-50) {
            double dT = -resid / du_dT;
            double T_cur = pow(10.0, lT);
            double T_new = T_cur + dT;

            /* Guard against negative or extreme overshoot */
            if(T_new <= 0 || log10(T_new) < lT_lo || log10(T_new) > lT_hi) {
                /* Bisection fallback */
                lT_new = 0.5 * (lT_lo + lT_hi);
            } else {
                lT_new = log10(T_new);
            }
        } else {
            /* du/dT <= 0 or negligible: bisection fallback (phase boundary region) */
            lT_new = 0.5 * (lT_lo + lT_hi);
        }

        lT = lT_new;
    }

    if(!converged) {
        /* Fallback: use best bracket midpoint */
        lT = 0.5 * (lT_lo + lT_hi);
    }

    T = pow(10.0, lT);
    if(T_guess) *T_guess = T;

    /* --- Look up all quantities at the converged (rho, T) --- */
    if(press_cgs) *press_cgs = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_pressure, lr, lT));
    if(entropy_cgs) *entropy_cgs = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_entropy, lr, lT));

    if(cs_cgs) {
        if(tbl->has_csound)
            *cs_cgs = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_csound, lr, lT));
        else {
            /* Estimate cs from thermodynamic identity: cs^2 = (dP/drho)_S
               Approximate with (dP/drho)_T + T/rho^2 * (dP/dT)^2 / Cv */
            double du_dT = aneos_get_du_dT(tbl, lr, lT);
            double Cv_local = (du_dT > 0) ? du_dT : 1e-10;

            /* dP/drho at constant T */
            double dlr = 1.0e-4;
            double lr_lo = lr - dlr, lr_hi = lr + dlr;
            if(lr_lo < tbl->logrho_min) { lr_lo = tbl->logrho_min; lr_hi = lr_lo + 2.0 * dlr; }
            if(lr_hi > tbl->logrho_max) { lr_hi = tbl->logrho_max; lr_lo = lr_hi - 2.0 * dlr; }
            double P_rlo = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_pressure, lr_lo, lT));
            double P_rhi = pow(10.0, aneos_bilinear_interp(tbl, tbl->log_pressure, lr_hi, lT));
            double rho_lo = pow(10.0, lr_lo), rho_hi = pow(10.0, lr_hi);
            double dP_drho_T = (P_rhi - P_rlo) / (rho_hi - rho_lo);

            double dP_dT_rho = aneos_get_dP_dT(tbl, lr, lT);
            double rho_here = pow(10.0, lr);
            double cs2 = dP_drho_T + T * dP_dT_rho * dP_dT_rho / (rho_here * rho_here * Cv_local);
            *cs_cgs = (cs2 > 0) ? sqrt(cs2) : 0.0;
        }
    }

    /* Cv = du/dT at constant rho */
    if(cv_cgs) {
        double du_dT = aneos_get_du_dT(tbl, lr, lT);
        *cv_cgs = (du_dT > 0) ? du_dT : 0.0;
    }

    /* Gruneisen parameter: Gamma = (1/rho) * (dP/du)_rho = (dP/dT) / (rho * du/dT) */
    if(gruneisen) {
        double du_dT = aneos_get_du_dT(tbl, lr, lT);
        double dP_dT = aneos_get_dP_dT(tbl, lr, lT);
        double rho_here = pow(10.0, lr);
        if(du_dT > 0 && rho_here > 0)
            *gruneisen = dP_dT / (rho_here * du_dT);
        else
            *gruneisen = 0.0;
    }

    if(phase_out) *phase_out = aneos_phase_lookup(tbl, lr, lT);

    return 0;
}

/* ---- Host entry points ----
   Same signatures the module has always exported, so no call site changes.
   These are the only readers of the ANEOS_Tables global. */

/* Main EOS call: given (rho, u) in CGS, compute all thermodynamic quantities.
   T_guess is in/out: pass the previous temperature as initial guess,
   returns the converged temperature.
   phase_out may be NULL if not needed. Returns 0 on success. Thread-safe. */
inline int aneos_compute(int mat_index,
                         double rho_cgs, double u_cgs,
                         double *T_guess,
                         double *press_cgs,
                         double *cs_cgs,
                         double *entropy_cgs,
                         double *cv_cgs,
                         double *gruneisen,
                         int    *phase_out)
{
    return aneos_compute_P(ANEOS_Tables, mat_index, rho_cgs, u_cgs, T_guess,
                           press_cgs, cs_cgs, entropy_cgs, cv_cgs, gruneisen, phase_out);
}

/* Direct lookup from (rho, T) in CGS, no temperature inversion.
   Available for diagnostics and internal use. Thread-safe. */
inline int aneos_lookup_rhoT(int mat_index,
                             double rho_cgs, double T_cgs,
                             double *press_cgs, double *u_cgs,
                             double *entropy_cgs, double *cs_cgs,
                             int *phase_out)
{
    return aneos_lookup_rhoT_P(ANEOS_Tables, mat_index, rho_cgs, T_cgs,
                               press_cgs, u_cgs, entropy_cgs, cs_cgs, phase_out);
}

#endif /* EOS_ANEOS */
#endif /* ANEOS_H */
