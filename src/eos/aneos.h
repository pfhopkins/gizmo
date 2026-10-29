#ifndef ANEOS_H
#define ANEOS_H

#ifndef GIZMO_config_H
#include "../GIZMO_config.h"
#endif

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

/* Storage for one ANEOS material table (SESAME-format, pre-computed rho-T grid).
   The axes may be spaced arbitrarily -- real tables concentrate their points where
   the EOS changes fastest -- so every lookup brackets a query against the actual
   axis values. A coarse map per axis bounds that search. */
struct aneos_table {
    int    mat_id;        /* SESAME material number */
    int    nrho;          /* number of density grid points */
    int    nT;            /* number of temperature grid points */
    int    loaded;        /* 0 = empty slot, 1 = loaded */

    /* Grid axes in log10 space (CGS units: rho in g/cm^3, T in K), strictly increasing */
    double *logrho;       /* log10(rho), length nrho */
    double *logT;         /* log10(T),   length nT   */

    /* Tabulated 2D arrays, row-major [irho * nT + iT] */
    double *log_pressure; /* log10(P [dyn/cm^2]) */
    double *log_energy;   /* log10(u [erg/g])    */
    double *entropy;      /* S [erg/g/K], signed: its zero point is a convention, so it is stored and interpolated as a value */
    double *log_csound;   /* log10(cs [cm/s])    */
    int    *phase;        /* phase flag (integer, no interpolation) */
    int     has_csound;   /* 1 if sound speed column present in table */
    int     has_phase;    /* 1 if phase column present in table */
    int    *lowest_valid_T; /* optional, length nrho: per density, the lowest temperature row the EOS is valid at (NULL = row 0 everywhere) */

    /* Grid bounds */
    double logrho_min, logrho_max;
    double logT_min, logT_max;

    /* Coarse search maps: nmap equal-width bins span each axis, and map[b] is the lowest
       interval bin b touches, so the interval enclosing any point of bin b lies in
       [map[b], map[b+1]]. Built once at load; the size scales with the axis length,
       never with its smallest spacing. */
    int     nmap_rho, nmap_T;
    int    *map_rho;      /* length nmap_rho + 1 */
    int    *map_T;        /* length nmap_T + 1   */
    double  inv_binwidth_rho, inv_binwidth_T;
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

/* A position on one table axis: the lower knot of the enclosing interval and the
   fractional position inside it, t = (x - x_i) / (x_{i+1} - x_i). */
struct aneos_axis_location {
    int    i;
    double t;
};

/* Locate x (clamped to the axis range) on a strictly increasing axis. The coarse map
   gives the short range of candidate intervals and a binary search inside it finds the
   enclosing one; the last two loops step at most an interval to absorb rounding in the
   bin index, so the bracket is exact. */
static KOKKOS_INLINE_FUNCTION struct aneos_axis_location aneos_locate(const double *axis, int n,
                               const int *map, int nmap, double inv_binwidth, double x)
{
    if(x < axis[0]) x = axis[0];
    if(x > axis[n - 1]) x = axis[n - 1];
    int b = (int)((x - axis[0]) * inv_binwidth);
    if(b < 0) b = 0;
    if(b > nmap - 1) b = nmap - 1;
    int lo = map[b], hi = map[b + 1];
    while(lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if(axis[mid] <= x) {lo = mid;} else {hi = mid - 1;}
    }
    while(lo > 0 && axis[lo] > x) {lo--;}
    while(lo < n - 2 && axis[lo + 1] <= x) {lo++;}
    struct aneos_axis_location loc;
    loc.i = lo;
    loc.t = (x - axis[lo]) / (axis[lo + 1] - axis[lo]);
    if(loc.t < 0.0) loc.t = 0.0;
    if(loc.t > 1.0) loc.t = 1.0;
    return loc;
}

static KOKKOS_INLINE_FUNCTION struct aneos_axis_location aneos_locate_rho(const struct aneos_table *tbl, double lr)
{
    return aneos_locate(tbl->logrho, tbl->nrho, tbl->map_rho, tbl->nmap_rho, tbl->inv_binwidth_rho, lr);
}

static KOKKOS_INLINE_FUNCTION struct aneos_axis_location aneos_locate_T(const struct aneos_table *tbl, double lT)
{
    return aneos_locate(tbl->logT, tbl->nT, tbl->map_T, tbl->nmap_T, tbl->inv_binwidth_T, lT);
}

/* Bilinear interpolation of a tabulated quantity (stored as log10 of the value, or as
   the value itself for entropy) at located density and temperature positions. */
static KOKKOS_INLINE_FUNCTION double aneos_interp(const struct aneos_table *tbl, const double *qty,
                               struct aneos_axis_location r, struct aneos_axis_location T)
{
    int nT = tbl->nT, i = r.i, j = T.i;
    double t = r.t, u = T.t;
    return (1.0 - t) * (1.0 - u) * qty[i * nT + j] + t * (1.0 - u) * qty[(i + 1) * nT + j]
         + (1.0 - t) * u * qty[i * nT + (j + 1)] + t * u * qty[(i + 1) * nT + (j + 1)];
}

/* Phase flag of the nearest knot (in log10 rho, log10 T); a point exactly midway takes the upper knot */
static KOKKOS_INLINE_FUNCTION int aneos_phase_at(const struct aneos_table *tbl,
                               struct aneos_axis_location r, struct aneos_axis_location T)
{
    if(!tbl->has_phase) return 0;
    int i = r.i + ((r.t >= 0.5) ? 1 : 0);
    int j = T.i + ((T.t >= 0.5) ? 1 : 0);
    return tbl->phase[i * tbl->nT + j];
}

/* Exact log-slopes of the interpolant inside the located cell: d log10(Q) / d log10(T) and
   d log10(Q) / d log10(rho). At a knot the interpolant has no unique slope; the interval the
   locator selected (the one above the knot, except at the last knot) decides it. With
   Q = 10^(interpolated value), dQ/dT = (Q/T) * d log10(Q)/d log10(T), and likewise for rho. */
static KOKKOS_INLINE_FUNCTION double aneos_dlogQ_dlogT(const struct aneos_table *tbl, const double *log_qty,
                               struct aneos_axis_location r, struct aneos_axis_location T)
{
    int nT = tbl->nT, i = r.i, j = T.i;
    double dq = (1.0 - r.t) * (log_qty[i * nT + j + 1] - log_qty[i * nT + j])
              + r.t * (log_qty[(i + 1) * nT + j + 1] - log_qty[(i + 1) * nT + j]);
    return dq / (tbl->logT[j + 1] - tbl->logT[j]);
}

static KOKKOS_INLINE_FUNCTION double aneos_dlogQ_dlogrho(const struct aneos_table *tbl, const double *log_qty,
                               struct aneos_axis_location r, struct aneos_axis_location T)
{
    int nT = tbl->nT, i = r.i, j = T.i;
    double dq = (1.0 - T.t) * (log_qty[(i + 1) * nT + j] - log_qty[i * nT + j])
              + T.t * (log_qty[(i + 1) * nT + j + 1] - log_qty[i * nT + j + 1]);
    return dq / (tbl->logrho[i + 1] - tbl->logrho[i]);
}

/* The lowest valid log10(T) inside the located density interval: the higher of its two columns' bounds, so the
   bilinear cell never reaches below either. Without per-density bounds this is the bottom of the table. */
static KOKKOS_INLINE_FUNCTION double aneos_logT_floor(const struct aneos_table *tbl, struct aneos_axis_location r)
{
    if(!tbl->lowest_valid_T) return tbl->logT_min;
    int k = tbl->lowest_valid_T[r.i], k1 = tbl->lowest_valid_T[r.i + 1];
    return tbl->logT[(k1 > k) ? k1 : k];
}

/* The specific internal energy at that lowest valid temperature, at the located density: the lowest energy the
   equation of state is valid for there [erg/g] */
static KOKKOS_INLINE_FUNCTION double aneos_energy_at_logT_floor(const struct aneos_table *tbl, struct aneos_axis_location r)
{
    return pow(10.0, aneos_interp(tbl, tbl->log_energy, r, aneos_locate_T(tbl, aneos_logT_floor(tbl, r))));
}

/* The same at a density [g/cm^3] (clamped to the table, as the energy inversion clamps it); 0 if the material's table is not loaded */
KOKKOS_INLINE_FUNCTION
double aneos_min_specific_energy(const struct aneos_table *tables, int mat_index, double rho_cgs)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) return 0;
    const struct aneos_table *tbl = &tables[mat_index];
    if(!tbl->loaded) return 0;
    double lr = log10(rho_cgs > 0 ? rho_cgs : 1e-30);
    if(lr < tbl->logrho_min) lr = tbl->logrho_min;
    if(lr > tbl->logrho_max) lr = tbl->logrho_max;
    return aneos_energy_at_logT_floor(tbl, aneos_locate_rho(tbl, lr));
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

    struct aneos_axis_location r = aneos_locate_rho(tbl, log10(rho_cgs > 0 ? rho_cgs : 1e-30));
    double lT = log10(T_cgs > 0 ? T_cgs : 1.0), lT_floor = aneos_logT_floor(tbl, r);
    struct aneos_axis_location T = aneos_locate_T(tbl, (lT < lT_floor) ? lT_floor : lT); /* below the valid region: the state at its edge */

    if(press_cgs)   *press_cgs   = pow(10.0, aneos_interp(tbl, tbl->log_pressure, r, T));
    if(u_cgs)       *u_cgs       = pow(10.0, aneos_interp(tbl, tbl->log_energy, r, T));
    if(entropy_cgs) *entropy_cgs = aneos_interp(tbl, tbl->entropy, r, T);
    if(cs_cgs) {
        if(tbl->has_csound)
            *cs_cgs = pow(10.0, aneos_interp(tbl, tbl->log_csound, r, T));
        else
            *cs_cgs = 0;
    }
    if(phase_out) *phase_out = aneos_phase_at(tbl, r, T);

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

    /* density is fixed through the temperature solve: locate it once */
    struct aneos_axis_location r = aneos_locate_rho(tbl, lr);

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

    /* Bracketing for bisection fallback: the valid temperature range at this density */
    double lT_lo = aneos_logT_floor(tbl, r);
    double lT_hi = tbl->logT_max;

    /* Clamp initial guess to it */
    if(lT < lT_lo) lT = lT_lo;
    if(lT > lT_hi) lT = lT_hi;

    /* Check bracket: evaluate u at the endpoints */
    double u_lo = aneos_energy_at_logT_floor(tbl, r);
    double u_hi = pow(10.0, aneos_interp(tbl, tbl->log_energy, r, aneos_locate_T(tbl, lT_hi)));

    /* If u_target is outside the valid range, clamp: below it, every quantity is that of the state at the lower edge */
    if(u_target <= u_lo) {
        lT = lT_lo; converged = 1;
    } else if(u_target >= u_hi) {
        lT = lT_hi; converged = 1;
    }

    for(int iter = 0; iter < max_iter && !converged; iter++) {
        struct aneos_axis_location Tl = aneos_locate_T(tbl, lT);
        double u_here = pow(10.0, aneos_interp(tbl, tbl->log_energy, r, Tl));
        double resid = u_here - u_target;

        /* Check convergence */
        if(fabs(resid) < tol * fabs(u_target)) { converged = 1; break; }

        /* Update bracket */
        if(resid < 0) { lT_lo = lT; } else { lT_hi = lT; }

        /* Newton step: du/dT, the exact slope of the interpolant in this cell */
        double du_dT = (u_here / pow(10.0, lT)) * aneos_dlogQ_dlogT(tbl, tbl->log_energy, r, Tl);
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
    struct aneos_axis_location Tloc = aneos_locate_T(tbl, lT);
    if(press_cgs) *press_cgs = pow(10.0, aneos_interp(tbl, tbl->log_pressure, r, Tloc));
    if(entropy_cgs) *entropy_cgs = aneos_interp(tbl, tbl->entropy, r, Tloc);

    /* du/dT and dP/dT at the converged state, each evaluated once if any output needs it */
    volatile int need_cs_estimate = (cs_cgs && !tbl->has_csound); /* volatile: nvc++ folds plain int gates in device code */
    double du_dT = 0, dP_dT = 0;
    double rho_here = pow(10.0, lr);
    if(need_cs_estimate || cv_cgs || gruneisen) {du_dT = (pow(10.0, aneos_interp(tbl, tbl->log_energy, r, Tloc)) / T) * aneos_dlogQ_dlogT(tbl, tbl->log_energy, r, Tloc);}
    if(need_cs_estimate || gruneisen) {dP_dT = (pow(10.0, aneos_interp(tbl, tbl->log_pressure, r, Tloc)) / T) * aneos_dlogQ_dlogT(tbl, tbl->log_pressure, r, Tloc);}

    if(cs_cgs) {
        if(tbl->has_csound)
            *cs_cgs = pow(10.0, aneos_interp(tbl, tbl->log_csound, r, Tloc));
        else {
            /* Estimate cs from thermodynamic identity: cs^2 = (dP/drho)_S
               Approximate with (dP/drho)_T + T/rho^2 * (dP/dT)^2 / Cv */
            double Cv_local = (du_dT > 0) ? du_dT : 1e-10;

            /* dP/drho at constant T, the exact slope of the interpolant in this cell */
            double dP_drho_T = (pow(10.0, aneos_interp(tbl, tbl->log_pressure, r, Tloc)) / rho_here) * aneos_dlogQ_dlogrho(tbl, tbl->log_pressure, r, Tloc);

            double cs2 = dP_drho_T + T * dP_dT * dP_dT / (rho_here * rho_here * Cv_local);
            *cs_cgs = (cs2 > 0) ? sqrt(cs2) : 0.0;
        }
    }

    /* Cv = du/dT at constant rho */
    if(cv_cgs) *cv_cgs = (du_dT > 0) ? du_dT : 0.0;

    /* Gruneisen parameter: Gamma = (1/rho) * (dP/du)_rho = (dP/dT) / (rho * du/dT) */
    if(gruneisen) *gruneisen = (du_dT > 0 && rho_here > 0) ? dP_dT / (rho_here * du_dT) : 0.0;

    if(phase_out) *phase_out = aneos_phase_at(tbl, r, Tloc);

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
