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

   Interpolation is bilinear in log-log space for P, u, S, cs.
   Phase uses nearest-neighbor lookup (no interpolation on integers).

   Temperature inversion (rho, u) -> T uses Newton-Raphson with
   bisection fallback.
   ============================================================ */

struct aneos_table ANEOS_Tables[ANEOS_MAX_MATERIALS];
int ANEOS_Num_Tables_Loaded = 0;

/* ---- Internal helpers ---- */

/* Bilinear interpolation on a uniform log-log grid.
   Returns interpolated value of log10(Q) at (lr, lT).
   The table stores log10 of the quantity in log_qty[irho * nT + iT]. */
static double bilinear_interp(const struct aneos_table *tbl, const double *log_qty,
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
static int phase_lookup(const struct aneos_table *tbl, double lr, double lT)
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

   Grid must be uniform in log10 space. */
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
    memset(tbl, 0, sizeof(*tbl));

    /* Read header */
    int ncols;
    if(fscanf(fp, "%d %d %d %d", &tbl->mat_id, &tbl->nrho, &tbl->nT, &ncols) != 4) {
        fprintf(stderr, "ANEOS: error reading header from \"%s\"\n", filename);
        fclose(fp); return 1;
    }

    int nrho = tbl->nrho, nT = tbl->nT, ntot = nrho * nT;
    if(nrho < 2 || nT < 2) {
        fprintf(stderr, "ANEOS: table too small: nrho=%d nT=%d\n", nrho, nT);
        fclose(fp); return 1;
    }
    if(ncols < 5 || ncols > 7) {
        fprintf(stderr, "ANEOS: unexpected ncols=%d (expected 5, 6, or 7)\n", ncols);
        fclose(fp); return 1;
    }

    tbl->has_csound = (ncols >= 6);
    tbl->has_phase  = (ncols >= 7);

    /* Allocate grid arrays */
    tbl->logrho = (double *)calloc(nrho, sizeof(double));
    tbl->logT   = (double *)calloc(nT, sizeof(double));

    /* Allocate 2D arrays */
    tbl->log_pressure = (double *)calloc(ntot, sizeof(double));
    tbl->log_energy   = (double *)calloc(ntot, sizeof(double));
    tbl->log_entropy  = (double *)calloc(ntot, sizeof(double));
    if(tbl->has_csound)
        tbl->log_csound = (double *)calloc(ntot, sizeof(double));
    if(tbl->has_phase)
        tbl->phase = (int *)calloc(ntot, sizeof(int));

    /* Read density grid */
    for(int i = 0; i < nrho; i++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "ANEOS: error reading density grid at index %d\n", i);
            fclose(fp); return 1;
        }
        tbl->logrho[i] = log10(val);
    }

    /* Read temperature grid */
    for(int j = 0; j < nT; j++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "ANEOS: error reading temperature grid at index %d\n", j);
            fclose(fp); return 1;
        }
        tbl->logT[j] = log10(val);
    }

    /* Read 2D data: nrho*nT rows, each with ncols columns
       Row order: outer loop over rho (slow), inner loop over T (fast)
       Columns: rho T P u S [cs] [phase] */
    for(int i = 0; i < nrho; i++) {
        for(int j = 0; j < nT; j++) {
            int idx = i * nT + j;
            double rho_val, T_val, P_val, u_val, S_val, cs_val = 0;
            int phase_val = 0;

            if(ncols == 5) {
                if(fscanf(fp, "%lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val) != 5) {
                    fprintf(stderr, "ANEOS: error reading data row irho=%d iT=%d\n", i, j);
                    fclose(fp); return 1;
                }
            } else if(ncols == 6) {
                if(fscanf(fp, "%lf %lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val) != 6) {
                    fprintf(stderr, "ANEOS: error reading data row irho=%d iT=%d\n", i, j);
                    fclose(fp); return 1;
                }
            } else {
                if(fscanf(fp, "%lf %lf %lf %lf %lf %lf %d", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val, &phase_val) != 7) {
                    fprintf(stderr, "ANEOS: error reading data row irho=%d iT=%d\n", i, j);
                    fclose(fp); return 1;
                }
            }

            /* Store in log10 space. Handle zero/negative values by flooring. */
            tbl->log_pressure[idx] = (P_val > 0) ? log10(P_val) : -30.0;
            tbl->log_energy[idx]   = (u_val > 0) ? log10(u_val) : -30.0;
            tbl->log_entropy[idx]  = (S_val > 0) ? log10(S_val) : -30.0;
            if(tbl->has_csound)
                tbl->log_csound[idx] = (cs_val > 0) ? log10(cs_val) : -30.0;
            if(tbl->has_phase)
                tbl->phase[idx] = phase_val;
        }
    }
    fclose(fp);

    /* Compute grid metadata */
    tbl->logrho_min = tbl->logrho[0];
    tbl->logrho_max = tbl->logrho[nrho - 1];
    tbl->logT_min   = tbl->logT[0];
    tbl->logT_max   = tbl->logT[nT - 1];
    tbl->d_logrho   = (tbl->logrho_max - tbl->logrho_min) / (nrho - 1);
    tbl->d_logT     = (tbl->logT_max - tbl->logT_min) / (nT - 1);
    tbl->inv_d_logrho = 1.0 / tbl->d_logrho;
    tbl->inv_d_logT   = 1.0 / tbl->d_logT;

    tbl->loaded = 1;

    printf("ANEOS: loaded material %d from \"%s\" (%d x %d grid, %d columns)\n",
           tbl->mat_id, filename, nrho, nT, ncols);
    printf("  log10(rho) range: [%.3f, %.3f],  log10(T) range: [%.3f, %.3f]\n",
           tbl->logrho_min, tbl->logrho_max, tbl->logT_min, tbl->logT_max);

    if(mat_index >= ANEOS_Num_Tables_Loaded)
        ANEOS_Num_Tables_Loaded = mat_index + 1;

    return 0;
}


int aneos_cleanup(void)
{
    for(int k = 0; k < ANEOS_MAX_MATERIALS; k++) {
        struct aneos_table *tbl = &ANEOS_Tables[k];
        if(!tbl->loaded) continue;
        free(tbl->logrho);       tbl->logrho = NULL;
        free(tbl->logT);         tbl->logT = NULL;
        free(tbl->log_pressure); tbl->log_pressure = NULL;
        free(tbl->log_energy);   tbl->log_energy = NULL;
        free(tbl->log_entropy);  tbl->log_entropy = NULL;
        if(tbl->log_csound) { free(tbl->log_csound); tbl->log_csound = NULL; }
        if(tbl->phase)      { free(tbl->phase);      tbl->phase = NULL; }
        tbl->loaded = 0;
    }
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

    double lr = log10(rho_cgs > 0 ? rho_cgs : 1e-30);
    double lT = log10(T_cgs > 0 ? T_cgs : 1.0);

    if(press_cgs)   *press_cgs   = pow(10.0, bilinear_interp(tbl, tbl->log_pressure, lr, lT));
    if(u_cgs)       *u_cgs       = pow(10.0, bilinear_interp(tbl, tbl->log_energy, lr, lT));
    if(entropy_cgs) *entropy_cgs = pow(10.0, bilinear_interp(tbl, tbl->log_entropy, lr, lT));
    if(cs_cgs) {
        if(tbl->has_csound)
            *cs_cgs = pow(10.0, bilinear_interp(tbl, tbl->log_csound, lr, lT));
        else
            *cs_cgs = 0;
    }
    if(phase_out) *phase_out = phase_lookup(tbl, lr, lT);

    return 0;
}


/* ---- Temperature inversion: solve u_table(rho, T) = u_target for T ---- */

/* Internal helper: get log10(u) at given (log10_rho, log10_T) */
static inline double get_log_u(const struct aneos_table *tbl, double lr, double lT)
{
    return bilinear_interp(tbl, tbl->log_energy, lr, lT);
}

/* Internal helper: get du/dT numerically at given (log10_rho, log10_T).
   Returns du/dT in CGS (erg/g/K). Uses centered finite difference on the
   bilinear interpolant in log-log space. */
static double get_du_dT(const struct aneos_table *tbl, double lr, double lT)
{
    double dlT = 1.0e-4; /* step in log10(T) */
    double lT_lo = lT - dlT;
    double lT_hi = lT + dlT;

    /* Clamp to bounds */
    if(lT_lo < tbl->logT_min) { lT_lo = tbl->logT_min; lT_hi = lT_lo + 2.0 * dlT; }
    if(lT_hi > tbl->logT_max) { lT_hi = tbl->logT_max; lT_lo = lT_hi - 2.0 * dlT; }

    double log_u_lo = bilinear_interp(tbl, tbl->log_energy, lr, lT_lo);
    double log_u_hi = bilinear_interp(tbl, tbl->log_energy, lr, lT_hi);

    double u_lo = pow(10.0, log_u_lo);
    double u_hi = pow(10.0, log_u_hi);
    double T_lo = pow(10.0, lT_lo);
    double T_hi = pow(10.0, lT_hi);

    double du_dT = (u_hi - u_lo) / (T_hi - T_lo);
    return du_dT;
}

/* Internal helper: get dP/dT numerically */
static double get_dP_dT(const struct aneos_table *tbl, double lr, double lT)
{
    double dlT = 1.0e-4;
    double lT_lo = lT - dlT;
    double lT_hi = lT + dlT;
    if(lT_lo < tbl->logT_min) { lT_lo = tbl->logT_min; lT_hi = lT_lo + 2.0 * dlT; }
    if(lT_hi > tbl->logT_max) { lT_hi = tbl->logT_max; lT_lo = lT_hi - 2.0 * dlT; }

    double P_lo = pow(10.0, bilinear_interp(tbl, tbl->log_pressure, lr, lT_lo));
    double P_hi = pow(10.0, bilinear_interp(tbl, tbl->log_pressure, lr, lT_hi));
    double T_lo = pow(10.0, lT_lo);
    double T_hi = pow(10.0, lT_hi);
    return (P_hi - P_lo) / (T_hi - T_lo);
}


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
    double u_lo = pow(10.0, get_log_u(tbl, lr, lT_lo));
    double u_hi = pow(10.0, get_log_u(tbl, lr, lT_hi));

    /* If u_target is outside the table range, clamp */
    if(u_target <= u_lo) {
        lT = lT_lo; converged = 1;
    } else if(u_target >= u_hi) {
        lT = lT_hi; converged = 1;
    }

    for(int iter = 0; iter < max_iter && !converged; iter++) {
        double log_u_here = get_log_u(tbl, lr, lT);
        double u_here = pow(10.0, log_u_here);
        double resid = u_here - u_target;

        /* Check convergence */
        if(fabs(resid) < tol * fabs(u_target)) { converged = 1; break; }

        /* Update bracket */
        if(resid < 0) { lT_lo = lT; } else { lT_hi = lT; }

        /* Newton step: du/dT */
        double du_dT = get_du_dT(tbl, lr, lT);
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
    if(press_cgs) *press_cgs = pow(10.0, bilinear_interp(tbl, tbl->log_pressure, lr, lT));
    if(entropy_cgs) *entropy_cgs = pow(10.0, bilinear_interp(tbl, tbl->log_entropy, lr, lT));

    if(cs_cgs) {
        if(tbl->has_csound)
            *cs_cgs = pow(10.0, bilinear_interp(tbl, tbl->log_csound, lr, lT));
        else {
            /* Estimate cs from thermodynamic identity: cs^2 = (dP/drho)_S
               Approximate with (dP/drho)_T + T/rho^2 * (dP/dT)^2 / Cv */
            double du_dT = get_du_dT(tbl, lr, lT);
            double Cv_local = (du_dT > 0) ? du_dT : 1e-10;

            /* dP/drho at constant T */
            double dlr = 1.0e-4;
            double lr_lo = lr - dlr, lr_hi = lr + dlr;
            if(lr_lo < tbl->logrho_min) { lr_lo = tbl->logrho_min; lr_hi = lr_lo + 2.0 * dlr; }
            if(lr_hi > tbl->logrho_max) { lr_hi = tbl->logrho_max; lr_lo = lr_hi - 2.0 * dlr; }
            double P_rlo = pow(10.0, bilinear_interp(tbl, tbl->log_pressure, lr_lo, lT));
            double P_rhi = pow(10.0, bilinear_interp(tbl, tbl->log_pressure, lr_hi, lT));
            double rho_lo = pow(10.0, lr_lo), rho_hi = pow(10.0, lr_hi);
            double dP_drho_T = (P_rhi - P_rlo) / (rho_hi - rho_lo);

            double dP_dT_rho = get_dP_dT(tbl, lr, lT);
            double rho_here = pow(10.0, lr);
            double cs2 = dP_drho_T + T * dP_dT_rho * dP_dT_rho / (rho_here * rho_here * Cv_local);
            *cs_cgs = (cs2 > 0) ? sqrt(cs2) : 0.0;
        }
    }

    /* Cv = du/dT at constant rho */
    if(cv_cgs) {
        double du_dT = get_du_dT(tbl, lr, lT);
        *cv_cgs = (du_dT > 0) ? du_dT : 0.0;
    }

    /* Gruneisen parameter: Gamma = (1/rho) * (dP/du)_rho = (dP/dT) / (rho * du/dT) */
    if(gruneisen) {
        double du_dT = get_du_dT(tbl, lr, lT);
        double dP_dT = get_dP_dT(tbl, lr, lT);
        double rho_here = pow(10.0, lr);
        if(du_dT > 0 && rho_here > 0)
            *gruneisen = dP_dT / (rho_here * du_dT);
        else
            *gruneisen = 0.0;
    }

    if(phase_out) *phase_out = phase_lookup(tbl, lr, lT);

    return 0;
}


#endif /* EOS_ANEOS */
