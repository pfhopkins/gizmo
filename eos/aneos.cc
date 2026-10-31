#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "aneos.h"

#ifdef EOS_ANEOS

/* ============================================================
   ANEOS: SESAME-format tabulated equation of state

   Tables are 2D grids in (log10 rho, log10 T) with columns:
     rho [g/cm^3], T [K], P [dyn/cm^2], u [erg/g], S [erg/g/K],
     optionally cs [cm/s] and phase (int).

   Interpolation is bilinear in (log10 rho, log10 T) on the table's own, arbitrarily
   spaced axes: log10 of P, u and cs, and entropy as a value (it may be negative).
   Phase uses nearest-neighbor lookup (no interpolation on integers).

   Temperature inversion (rho, u) -> T uses Newton-Raphson with
   bisection fallback.
   ============================================================ */

struct aneos_table ANEOS_Tables[ANEOS_MAX_MATERIALS];
int ANEOS_Num_Tables_Loaded = 0;

/* ---- Internal helpers ---- */

/* Bilinear interpolation of a tabulated quantity (stored as log10 of the value, or as
   the value itself for entropy) at located density and temperature positions. */
static double aneos_interp(const struct aneos_table *tbl, const double *qty,
                           struct aneos_axis_location r, struct aneos_axis_location T)
{
    int nT = tbl->nT, i = r.i, j = T.i;
    double t = r.t, u = T.t;
    return (1.0 - t) * (1.0 - u) * qty[i * nT + j] + t * (1.0 - u) * qty[(i + 1) * nT + j]
         + (1.0 - t) * u * qty[i * nT + (j + 1)] + t * u * qty[(i + 1) * nT + (j + 1)];
}

/* Phase flag of the nearest knot (in log10 rho, log10 T); a point exactly midway takes the upper knot */
static int aneos_phase_at(const struct aneos_table *tbl, struct aneos_axis_location r, struct aneos_axis_location T)
{
    if(!tbl->has_phase) return 0;
    int i = r.i + ((r.t >= 0.5) ? 1 : 0);
    int j = T.i + ((T.t >= 0.5) ? 1 : 0);
    return tbl->phase[i * tbl->nT + j];
}

/* Exact log-slopes of the interpolant inside the located cell: d log10(Q) / d log10(T) and
   d log10(Q) / d log10(rho). At a knot the locator's choice of interval decides the slope.
   With Q = 10^(interpolated value), dQ/dT = (Q/T) * d log10(Q)/d log10(T), and likewise for rho. */
static double aneos_dlogQ_dlogT(const struct aneos_table *tbl, const double *log_qty,
                                struct aneos_axis_location r, struct aneos_axis_location T)
{
    int nT = tbl->nT, i = r.i, j = T.i;
    double dq = (1.0 - r.t) * (log_qty[i * nT + j + 1] - log_qty[i * nT + j])
              + r.t * (log_qty[(i + 1) * nT + j + 1] - log_qty[(i + 1) * nT + j]);
    return dq / (tbl->logT[j + 1] - tbl->logT[j]);
}

static double aneos_dlogQ_dlogrho(const struct aneos_table *tbl, const double *log_qty,
                                  struct aneos_axis_location r, struct aneos_axis_location T)
{
    int nT = tbl->nT, i = r.i, j = T.i;
    double dq = (1.0 - T.t) * (log_qty[(i + 1) * nT + j] - log_qty[i * nT + j])
              + T.t * (log_qty[(i + 1) * nT + j + 1] - log_qty[i * nT + j + 1]);
    return dq / (tbl->logrho[i + 1] - tbl->logrho[i]);
}

/* The lowest valid log10(T) inside the located density interval: the higher of its two columns' bounds, so the
   bilinear cell never reaches below either. Without per-density bounds this is the bottom of the table. */
static double aneos_logT_floor(const struct aneos_table *tbl, struct aneos_axis_location r)
{
    if(!tbl->lowest_valid_T) return tbl->logT_min;
    int k = tbl->lowest_valid_T[r.i], k1 = tbl->lowest_valid_T[r.i + 1];
    return tbl->logT[(k1 > k) ? k1 : k];
}

/* The specific internal energy at that lowest valid temperature, at the located density [erg/g]: the lowest energy the
   equation of state is valid for there */
static double aneos_energy_at_logT_floor(const struct aneos_table *tbl, struct aneos_axis_location r)
{
    return pow(10.0, aneos_interp(tbl, tbl->log_energy, r, aneos_locate_T(tbl, aneos_logT_floor(tbl, r))));
}


/* ---- Table I/O ---- */

/* Read a SESAME-format ASCII table. Expected format:
   Line 1: mat_id  nrho  nT  ncols
   Lines 2..(1+nrho): density values [g/cm^3] (one per line or space-separated)
   Next nT values: temperature values [K]
   Then nrho*nT rows, each with ncols columns:
     rho  T  P  u  S  [cs]  [phase]
   ncols=5: (rho, T, P, u, S)
   ncols=6: (rho, T, P, u, S, cs)
   ncols=7: (rho, T, P, u, S, cs, phase)

   The axes must be positive and strictly increasing; their spacing is arbitrary.
   Each row repeats its (rho, T), which must match the axes to ANEOS_AXIS_MATCH_RTOL
   (a table writer may print the rows with fewer digits than the axes). A table
   failing any check is refused rather than read as something it is not.

   An optional trailing block, the word lowest_valid_T_index and then one row index
   per density, marks a cold corner with no valid states: below that row a lookup
   returns the state at the row. */
#define ANEOS_AXIS_MATCH_RTOL 1.0e-8

/* Free every array a table slot holds and mark the slot empty */
static void aneos_free_table_arrays(struct aneos_table *tbl)
{
    double **dbl[] = {&tbl->logrho, &tbl->logT, &tbl->log_pressure, &tbl->log_energy, &tbl->entropy, &tbl->log_csound};
    for(int k = 0; k < (int)(sizeof(dbl) / sizeof(dbl[0])); k++) {free(*dbl[k]); *dbl[k] = NULL;}
    int **ints[] = {&tbl->phase, &tbl->map_rho, &tbl->map_T, &tbl->lowest_valid_T};
    for(int k = 0; k < (int)(sizeof(ints) / sizeof(ints[0])); k++) {free(*ints[k]); *ints[k] = NULL;}
    tbl->loaded = 0;
}

/* refuse the table: report why, release what was allocated, close the file, and return the error code */
static int aneos_refuse(FILE *fp, struct aneos_table *tbl, const char *filename, const char *why, int index)
{
    fprintf(stderr, "ANEOS: refusing table \"%s\": %s (index %d)\n", filename, why, index);
    aneos_free_table_arrays(tbl);
    if(fp) {fclose(fp);}
    return 1;
}

/* Build the coarse search map of one axis: nmap equal bins, map[b] = the lowest
   interval (largest k <= n-2 with axis[k] <= left edge of bin b). */
static void aneos_build_axis_map(const double *axis, int n, int nmap, int *map, double *inv_binwidth)
{
    double width = (axis[n - 1] - axis[0]) / nmap;
    *inv_binwidth = 1.0 / width;
    int k = 0;
    for(int b = 0; b <= nmap; b++) {
        double edge = (b == nmap) ? axis[n - 1] : axis[0] + b * width;
        while(k < n - 2 && axis[k + 1] <= edge) {k++;}
        map[b] = k;
    }
}

/* largest relative departure of the axis spacing from its mean (0 for a uniform axis) */
static double aneos_axis_nonuniformity(const double *axis, int n)
{
    double mean = (axis[n - 1] - axis[0]) / (n - 1), worst = 0;
    for(int k = 0; k < n - 1; k++) {double d = fabs((axis[k + 1] - axis[k]) / mean - 1.0); if(d > worst) {worst = d;}}
    return worst;
}

int aneos_read_table(const char *filename, int mat_index)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) {
        fprintf(stderr, "ANEOS: mat_index %d out of range [0, %d)\n", mat_index, ANEOS_MAX_MATERIALS);
        return 1;
    }

    FILE *fp = fopen(filename, "r");
    if(!fp) {
        fprintf(stderr, "ANEOS: could not open table file \"%s\"\n", filename);
        return 1;
    }

    struct aneos_table *tbl = &ANEOS_Tables[mat_index];
    aneos_free_table_arrays(tbl); /* a slot read again releases what it held */
    memset(tbl, 0, sizeof(*tbl));

    /* Read header */
    int ncols;
    if(fscanf(fp, "%d %d %d %d", &tbl->mat_id, &tbl->nrho, &tbl->nT, &ncols) != 4) {return aneos_refuse(fp, tbl, filename, "header unreadable", 0);}

    int nrho = tbl->nrho, nT = tbl->nT;
    if(nrho < 2 || nT < 2) {return aneos_refuse(fp, tbl, filename, "fewer than two points on an axis", (nrho < 2) ? nrho : nT);}
    /* lookups index the 2D arrays with int arithmetic (irho * nT + iT); the search maps hold 4n+1 entries */
    if(nrho > INT_MAX / 4 - 1 || nT > INT_MAX / 4 - 1) {return aneos_refuse(fp, tbl, filename, "axis too long for the search map", (nrho > nT) ? nrho : nT);}
    if((long long)nrho * (long long)nT > (long long)INT_MAX) {return aneos_refuse(fp, tbl, filename, "nrho*nT exceeds the int index range", nrho);}
    if(ncols < 5 || ncols > 7) {return aneos_refuse(fp, tbl, filename, "ncols must be 5, 6 or 7", ncols);}
    int ntot = nrho * nT;

    tbl->has_csound = (ncols >= 6);
    tbl->has_phase  = (ncols >= 7);
    tbl->nmap_rho = 4 * nrho;
    tbl->nmap_T   = 4 * nT;

    tbl->logrho  = (double *)calloc(nrho, sizeof(double));
    tbl->logT    = (double *)calloc(nT, sizeof(double));
    tbl->map_rho = (int *)calloc(tbl->nmap_rho + 1, sizeof(int));
    tbl->map_T   = (int *)calloc(tbl->nmap_T + 1, sizeof(int));
    tbl->log_pressure = (double *)calloc(ntot, sizeof(double));
    tbl->log_energy   = (double *)calloc(ntot, sizeof(double));
    tbl->entropy      = (double *)calloc(ntot, sizeof(double));
    if(tbl->has_csound) {tbl->log_csound = (double *)calloc(ntot, sizeof(double));}
    if(tbl->has_phase)  {tbl->phase = (int *)calloc(ntot, sizeof(int));}
    if(!tbl->logrho || !tbl->logT || !tbl->map_rho || !tbl->map_T || !tbl->log_pressure || !tbl->log_energy || !tbl->entropy
       || (tbl->has_csound && !tbl->log_csound) || (tbl->has_phase && !tbl->phase)) {return aneos_refuse(fp, tbl, filename, "could not allocate the table", ntot);}

    /* Read the axes: positive, finite, strictly increasing. They are held as the values read
       until the rows have been checked against them, then converted to log10 in place. */
    for(int i = 0; i < nrho; i++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {return aneos_refuse(fp, tbl, filename, "density axis unreadable", i);}
        if(!(isfinite(val) && val > 0)) {return aneos_refuse(fp, tbl, filename, "density axis value not positive and finite", i);}
        if(i > 0 && !(val > tbl->logrho[i - 1])) {return aneos_refuse(fp, tbl, filename, "density axis not strictly increasing", i);}
        tbl->logrho[i] = val;
    }
    for(int j = 0; j < nT; j++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {return aneos_refuse(fp, tbl, filename, "temperature axis unreadable", j);}
        if(!(isfinite(val) && val > 0)) {return aneos_refuse(fp, tbl, filename, "temperature axis value not positive and finite", j);}
        if(j > 0 && !(val > tbl->logT[j - 1])) {return aneos_refuse(fp, tbl, filename, "temperature axis not strictly increasing", j);}
        tbl->logT[j] = val;
    }

    /* Read 2D data: nrho*nT rows, each with ncols columns
       Row order: outer loop over rho (slow), inner loop over T (fast)
       Columns: rho T P u S [cs] [phase] */
    for(int i = 0; i < nrho; i++) {
        for(int j = 0; j < nT; j++) {
            int idx = i * nT + j;
            double rho_val, T_val, P_val, u_val, S_val, cs_val = 0;
            int phase_val = 0, nread;

            if(ncols == 5) {
                nread = fscanf(fp, "%lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val);
            } else if(ncols == 6) {
                nread = fscanf(fp, "%lf %lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val);
            } else {
                nread = fscanf(fp, "%lf %lf %lf %lf %lf %lf %d", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val, &phase_val);
            }
            if(nread != ncols) {return aneos_refuse(fp, tbl, filename, "data row unreadable", idx);}
            if(!(fabs(rho_val / tbl->logrho[i] - 1.0) <= ANEOS_AXIS_MATCH_RTOL)) {return aneos_refuse(fp, tbl, filename, "data row density does not match the density axis", idx);}
            if(!(fabs(T_val / tbl->logT[j] - 1.0) <= ANEOS_AXIS_MATCH_RTOL)) {return aneos_refuse(fp, tbl, filename, "data row temperature does not match the temperature axis", idx);}
            if(!(isfinite(P_val) && isfinite(u_val) && isfinite(S_val) && isfinite(cs_val))) {return aneos_refuse(fp, tbl, filename, "non-finite value in data row", idx);}

            /* P, u and cs are interpolated in log10; zero/negative values are floored */
            tbl->log_pressure[idx] = (P_val > 0) ? log10(P_val) : -30.0;
            tbl->log_energy[idx]   = (u_val > 0) ? log10(u_val) : -30.0;
            tbl->entropy[idx]      = S_val;
            if(tbl->has_csound)
                tbl->log_csound[idx] = (cs_val > 0) ? log10(cs_val) : -30.0;
            if(tbl->has_phase)
                tbl->phase[idx] = phase_val;
        }
    }
    /* Optional trailing block: per density, the index of the lowest valid temperature row */
    char block_name[64];
    if(fscanf(fp, " %63s", block_name) == 1) {
        if(strcmp(block_name, "lowest_valid_T_index") != 0) {return aneos_refuse(fp, tbl, filename, "unrecognised block after the data rows", 0);}
        tbl->lowest_valid_T = (int *)calloc(nrho, sizeof(int));
        if(!tbl->lowest_valid_T) {return aneos_refuse(fp, tbl, filename, "could not allocate the lowest-valid-row block", nrho);}
        for(int i = 0; i < nrho; i++) {
            int k;
            if(fscanf(fp, "%d", &k) != 1) {return aneos_refuse(fp, tbl, filename, "lowest-valid-row block unreadable", i);}
            if(k < 0 || k > nT - 2) {return aneos_refuse(fp, tbl, filename, "lowest valid row outside [0, nT-2]", i);}
            tbl->lowest_valid_T[i] = k;
        }
    }
    fclose(fp);
    for(int i = 0; i < nrho; i++) {tbl->logrho[i] = log10(tbl->logrho[i]);}
    for(int j = 0; j < nT; j++) {tbl->logT[j] = log10(tbl->logT[j]);}
    for(int i = 1; i < nrho; i++) {if(!(tbl->logrho[i] > tbl->logrho[i - 1])) {return aneos_refuse(NULL, tbl, filename, "density axis points too close to resolve in log10", i);}}
    for(int j = 1; j < nT; j++) {if(!(tbl->logT[j] > tbl->logT[j - 1])) {return aneos_refuse(NULL, tbl, filename, "temperature axis points too close to resolve in log10", j);}}

    /* Grid bounds and search maps */
    tbl->logrho_min = tbl->logrho[0];
    tbl->logrho_max = tbl->logrho[nrho - 1];
    tbl->logT_min   = tbl->logT[0];
    tbl->logT_max   = tbl->logT[nT - 1];
    aneos_build_axis_map(tbl->logrho, nrho, tbl->nmap_rho, tbl->map_rho, &tbl->inv_binwidth_rho);
    aneos_build_axis_map(tbl->logT, nT, tbl->nmap_T, tbl->map_T, &tbl->inv_binwidth_T);

    tbl->loaded = 1;

    double nonuni_rho = aneos_axis_nonuniformity(tbl->logrho, nrho), nonuni_T = aneos_axis_nonuniformity(tbl->logT, nT);
    printf("ANEOS: loaded material %d from \"%s\" (%d x %d grid, %d columns%s)\n",
           tbl->mat_id, filename, nrho, nT, ncols, tbl->lowest_valid_T ? ", per-density lowest valid temperature" : "");
    printf("  log10(rho) range: [%.3f, %.3f] %s,  log10(T) range: [%.3f, %.3f] %s\n",
           tbl->logrho_min, tbl->logrho_max, (nonuni_rho < 1.0e-6) ? "uniform" : "irregular",
           tbl->logT_min, tbl->logT_max, (nonuni_T < 1.0e-6) ? "uniform" : "irregular");

    if(mat_index >= ANEOS_Num_Tables_Loaded)
        ANEOS_Num_Tables_Loaded = mat_index + 1;

    return 0;
}


int aneos_cleanup(void)
{
    for(int k = 0; k < ANEOS_MAX_MATERIALS; k++) {aneos_free_table_arrays(&ANEOS_Tables[k]);}
    ANEOS_Num_Tables_Loaded = 0;
    return 0;
}


/* ---- Direct (rho, T) lookup ---- */

int aneos_lookup_rhoT(int mat_index,
                      double rho_cgs, double T_cgs,
                      double *press_cgs, double *u_cgs,
                      double *entropy_cgs, double *cs_cgs,
                      int *phase_out)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) return 1;
    const struct aneos_table *tbl = &ANEOS_Tables[mat_index];
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


double aneos_min_specific_energy(int mat_index, double rho_cgs)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) return 0;
    const struct aneos_table *tbl = &ANEOS_Tables[mat_index];
    if(!tbl->loaded) return 0;
    double lr = log10(rho_cgs > 0 ? rho_cgs : 1e-30);
    if(lr < tbl->logrho_min) lr = tbl->logrho_min;
    if(lr > tbl->logrho_max) lr = tbl->logrho_max;
    return aneos_energy_at_logT_floor(tbl, aneos_locate_rho(tbl, lr));
}


/* ---- Temperature inversion: solve u_table(rho, T) = u_target for T ---- */

int aneos_compute(int mat_index,
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
    const struct aneos_table *tbl = &ANEOS_Tables[mat_index];
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
    int need_cs_estimate = (cs_cgs && !tbl->has_csound);
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


#endif /* EOS_ANEOS */
