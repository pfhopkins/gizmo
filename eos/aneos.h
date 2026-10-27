#ifndef ANEOS_H
#define ANEOS_H

#ifndef GIZMO_config_H /* a standalone build (test/aneos_unit) defines it to compile without a Config */
#include "../GIZMO_config.h"
#endif

#ifdef EOS_ANEOS

#include <stdio.h>
#include <stdlib.h>

#ifndef ANEOS_MAX_MATERIALS
#define ANEOS_MAX_MATERIALS 7
#endif

/* Storage for one ANEOS material table (SESAME-format, pre-computed rho-T grid).
   The axes may be spaced arbitrarily (real tables concentrate their points where
   the EOS changes fastest), so every lookup brackets a query against the actual
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

/* Global table array -- one per material slot */
extern struct aneos_table ANEOS_Tables[ANEOS_MAX_MATERIALS];
extern int ANEOS_Num_Tables_Loaded;

/* ---- Public API ---- */

/* Read a SESAME-format table file into slot mat_index.
   Call once per material from the main thread at startup. */
int aneos_read_table(const char *filename, int mat_index);

/* Free all allocated table memory. */
int aneos_cleanup(void);

/* Main EOS call: given (rho, u) in CGS, compute all thermodynamic quantities.
   T_guess is in/out: pass the previous temperature as initial guess,
   returns the converged temperature.
   phase_out may be NULL if not needed. Returns 0 on success. Thread-safe. */
int aneos_compute(int mat_index,
                  double rho_cgs, double u_cgs,
                  double *T_guess,
                  double *press_cgs,
                  double *cs_cgs,
                  double *entropy_cgs,
                  double *cv_cgs,
                  double *gruneisen,
                  int    *phase_out);

/* Direct lookup from (rho, T) in CGS, no temperature inversion.
   Available for diagnostics and internal use. Thread-safe. */
int aneos_lookup_rhoT(int mat_index,
                      double rho_cgs, double T_cgs,
                      double *press_cgs, double *u_cgs,
                      double *entropy_cgs, double *cs_cgs,
                      int *phase_out);

/* A position on one table axis: the lower knot of the enclosing interval and the
   fractional position inside it, t = (x - x_i) / (x_{i+1} - x_i). */
struct aneos_axis_location {
    int    i;
    double t;
};

/* Locate x (clamped to the axis range) on a strictly increasing axis. The coarse map
   gives the short range of candidate intervals and a binary search inside it finds the
   enclosing one; the last two loops step at most an interval to absorb rounding in the
   bin index, so the bracket is exact. At an interior knot this selects the interval
   above it (t = 0), at the last knot the one below it (t = 1). */
static inline struct aneos_axis_location aneos_locate(const double *axis, int n,
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

static inline struct aneos_axis_location aneos_locate_rho(const struct aneos_table *tbl, double lr)
{
    return aneos_locate(tbl->logrho, tbl->nrho, tbl->map_rho, tbl->nmap_rho, tbl->inv_binwidth_rho, lr);
}

static inline struct aneos_axis_location aneos_locate_T(const struct aneos_table *tbl, double lT)
{
    return aneos_locate(tbl->logT, tbl->nT, tbl->map_T, tbl->nmap_T, tbl->inv_binwidth_T, lT);
}

#endif /* EOS_ANEOS */
#endif /* ANEOS_H */
