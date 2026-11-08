/* Equation-of-state state census: classifies (rho, u) states against one tabulated (ANEOS / SESAME) material with GIZMO's own
   lookup and inversion (src/eos/aneos.{h,cc}, compiled in), so it sees exactly what a run computes for those states.
   Built as a shared library and driven from Python (census.py) on chunks of snapshot cells; see build.sh. */
#include <math.h>
#include <stdlib.h>
#include "src/eos/aneos.h"
extern "C" void *gizmo_gpu_alloc_shared(size_t nbytes, const char *label) {(void)label; return malloc(nbytes);}
extern "C" void gpu_particles_uvm_free(void *ptr) {free(ptr);}

/* census flags (independent bits) */
#define C_RHO_BELOW       (1 << 0)   /* density below the table */
#define C_RHO_ABOVE       (1 << 1)   /* density above the table */
#define C_U_AT_FLOOR      (1 << 2)   /* energy at or below the lowest valid temperature's at this density */
#define C_U_AT_TOP        (1 << 3)   /* energy at or above the hottest row's */
#define C_NEAR_FLOOR      (1 << 4)   /* u < 1.25 u_floor, or the interpolation cell holds the floor row */
#define C_NEAR_TOP        (1 << 5)   /* u > 0.8 u_top */
#define C_NEAR_RHO_LOW    (1 << 6)   /* inside the table, within log10(1/0.9) dex of its lowest density */
#define C_NEAR_RHO_HIGH   (1 << 7)   /* inside the table, within log10(1.1) dex of its highest density */
#define C_FOLD_IN         (1 << 8)   /* energy inside its density interval's fold range (the runtime scans the whole column there) */
#define C_FOLD_NEAR       (1 << 9)   /* outside it, within log10(1.25) dex */
#define C_ROOTS_MULTI     (1 << 10)  /* the whole-column scan finds several temperatures with this energy */
#define C_PLATEAU         (1 << 11)  /* u(T) flat to the solve tolerance at the energy: T not determined */
#define C_FOLD_GATE_MISS  (1 << 12)  /* several roots OUTSIDE the fold range: the runtime gate would not have looked (a defect) */
#define C_SCANNED         (1 << 13)  /* the whole-column scan ran for this cell */
#define C_FLOOR_IS_BLOCK  (1 << 14)  /* the validity floor here is a table floor row above the coldest row (not the table's bottom) */

#define NEAR_RHO_LOW_DEX   0.045757490560675115  /* log10(1/0.9) */
#define NEAR_RHO_HIGH_DEX  0.041392685158225077  /* log10(1.1) */
#define NEAR_ENERGY_FACTOR 1.25
#define FOLD_NEAR_DEX      0.096910013008056406  /* log10(1.25) */

extern "C" int census_load(const char *path, int slot) {return aneos_read_table(path, slot);}

/* nrho, nT, logrho_min, logrho_max, logT_min, logT_max, has_floor_block, n_fold_intervals */
extern "C" void census_table_info(int slot, double *out)
{
    const struct aneos_table *t = &ANEOS_Tables[slot];
    int n_fold = 0;
    if(t->fold_logu) {for(int i = 0; i < t->nrho - 1; i++) {if(t->fold_logu[2 * i] <= t->fold_logu[2 * i + 1]) {n_fold++;}}}
    out[0] = t->nrho; out[1] = t->nT; out[2] = t->logrho_min; out[3] = t->logrho_max; out[4] = t->logT_min; out[5] = t->logT_max;
    out[6] = (t->lowest_valid_T != NULL); out[7] = n_fold;
}

/* states at (rho, T) for building known positives: u [erg/g] and P [dyn/cm^2] */
extern "C" void census_rhoT(int slot, long n, const double *rho, const double *T, double *u, double *P)
{
    for(long k = 0; k < n; k++) {aneos_lookup_rhoT_P(ANEOS_Tables, slot, rho[k], T[k], &P[k], &u[k], NULL, NULL, NULL);}
}

/* per density: the validity-floor energy and the hottest row's energy [erg/g] (density clamped to the table, as the inversion does) */
extern "C" void census_energy_bounds(int slot, long n, const double *rho, double *u_floor, double *u_top)
{
    const struct aneos_table *t = &ANEOS_Tables[slot];
    for(long k = 0; k < n; k++)
    {
        double lr = log10(rho[k] > 0 ? rho[k] : 1e-30); if(lr < t->logrho_min) {lr = t->logrho_min;} if(lr > t->logrho_max) {lr = t->logrho_max;}
        struct aneos_axis_location r = aneos_locate_rho(t, lr);
        u_floor[k] = aneos_energy_at_logT_floor(t, r);
        u_top[k] = pow(10.0, aneos_row_value(t, t->log_energy, r, t->nT - 1));
    }
}

/* Classify n states (CGS). T_guess seeds the production inversion as the run seeds it. scan_all[k] != 0 forces the whole-column scan for
   cell k (fold-range and fold-near cells are always scanned). Outputs per cell: census flags, the inversion's own report flags, T and P from
   the production call, distances (floor_rows: rows above the floor row in table cells; dlnT_floor: ln(T / T_floor); u_floor_ratio: u / u_floor;
   u_top_ratio: u / u_top; fold_margin: dex from the fold range, negative inside, 99 where the interval has none; dex_lo / dex_hi: log10 rho
   above the lowest / below the highest table density), root count of the scan (-1: not scanned), plateau T width (dex) and P spread. */
extern "C" void census_classify(int slot, long n, const double *rho, const double *u, const double *T_guess, const unsigned char *scan_all,
                                int *flags, int *runtime_flags, double *T_out, double *P_out, double *floor_rows, double *dlnT_floor,
                                double *u_floor_ratio, double *u_top_ratio, double *fold_margin, double *dex_lo, double *dex_hi,
                                int *n_roots, double *plateau_width, double *plateau_spread)
{
    const struct aneos_table *t = &ANEOS_Tables[slot];
    const double tol = ANEOS_INVERSION_TOLERANCE, tol_log = log10(1.0 + tol);
    for(long k = 0; k < n; k++)
    {
        int f = 0;
        double lr = log10(rho[k] > 0 ? rho[k] : 1e-30);
        dex_lo[k] = lr - t->logrho_min; dex_hi[k] = t->logrho_max - lr;
        if(dex_lo[k] < 0) {f |= C_RHO_BELOW;} else if(dex_lo[k] < NEAR_RHO_LOW_DEX) {f |= C_NEAR_RHO_LOW;}
        if(dex_hi[k] < 0) {f |= C_RHO_ABOVE;} else if(dex_hi[k] < NEAR_RHO_HIGH_DEX) {f |= C_NEAR_RHO_HIGH;}

        /* the production call */
        struct aneos_inversion_report rep;
        double Tg = T_guess[k], P = 0;
        aneos_compute_P(ANEOS_Tables, slot, rho[k], u[k], &Tg, &P, NULL, NULL, NULL, NULL, NULL, &rep);
        runtime_flags[k] = rep.flags; T_out[k] = Tg; P_out[k] = P;
        double pw = (rep.flags & ANEOS_INVERSION_PLATEAU) ? rep.plateau_logT_width : 0, ps = (rep.flags & ANEOS_INVERSION_PLATEAU) ? rep.plateau_P_spread : 0;
        if(rep.flags & ANEOS_INVERSION_PLATEAU) {f |= C_PLATEAU;}

        /* floor and top, at the clamped density the inversion used */
        double lrc = (lr < t->logrho_min) ? t->logrho_min : ((lr > t->logrho_max) ? t->logrho_max : lr);
        struct aneos_axis_location r = aneos_locate_rho(t, lrc);
        int j_floor = aneos_floor_row(t, r);
        if(j_floor > 0) {f |= C_FLOOR_IS_BLOCK;}
        double u_floor = aneos_energy_at_logT_floor(t, r), u_top = pow(10.0, aneos_row_value(t, t->log_energy, r, t->nT - 1));
        double lu = log10(u[k] > 0 ? u[k] : 1e-30), lT = log10(Tg);
        struct aneos_axis_location Tl = aneos_locate_T(t, lT);
        u_floor_ratio[k] = u[k] / u_floor; u_top_ratio[k] = u[k] / u_top;
        floor_rows[k] = (Tl.i + Tl.t) - j_floor; dlnT_floor[k] = log(10.0) * (lT - t->logT[j_floor]);
        /* at the floor / top: the energy is at or beyond that end value and the run takes the end state; an energy beyond it that the
           column still holds is solved there by the run (recovered), so it is not at the floor or top */
        const int recovered = (rep.flags & ANEOS_INVERSION_RECOVERED) != 0;
        if(u[k] <= u_floor * (1.0 + tol) && !recovered) {f |= C_U_AT_FLOOR;}
        else if(u[k] < NEAR_ENERGY_FACTOR * u_floor || Tl.i == j_floor) {f |= C_NEAR_FLOOR;}
        if(u[k] >= u_top * (1.0 - tol) && !recovered) {f |= C_U_AT_TOP;}
        else if(u[k] > u_top / NEAR_ENERGY_FACTOR) {f |= C_NEAR_TOP;}

        /* fold range of the density interval: the runtime's own gate decides FOLD_IN */
        fold_margin[k] = 99;
        if(t->fold_logu && t->fold_logu[2 * r.i] <= t->fold_logu[2 * r.i + 1])
        {
            double lo = t->fold_logu[2 * r.i], hi = t->fold_logu[2 * r.i + 1];
            fold_margin[k] = (lu < lo) ? lo - lu : ((lu > hi) ? lu - hi : -fmin(lu - lo, hi - lu));
            if(fabs(lu - 0.5 * (lo + hi)) <= 0.5 * (hi - lo) + tol_log) {f |= C_FOLD_IN;}
            else if(fold_margin[k] < FOLD_NEAR_DEX) {f |= C_FOLD_NEAR;}
        }

        /* the whole-column scan */
        n_roots[k] = -1;
        if(scan_all[k] || (f & (C_FOLD_IN | C_FOLD_NEAR)))
        {
            struct aneos_inversion_report scan;
            scan.flags = 0; scan.n_roots = 0; scan.plateau_logT_width = 0; scan.plateau_P_spread = 0;
            double lT_root = 0;
            n_roots[k] = aneos_column_roots(t, r, lu, tol_log, &lT_root, &scan);
            f |= C_SCANNED;
            if(n_roots[k] >= 2) {f |= C_ROOTS_MULTI; if(!(f & C_FOLD_IN)) {f |= C_FOLD_GATE_MISS;}}
            if(scan.flags & ANEOS_INVERSION_PLATEAU)
            {
                f |= C_PLATEAU;
                if(scan.plateau_logT_width > pw) {pw = scan.plateau_logT_width;}
                if(scan.plateau_P_spread > ps) {ps = scan.plateau_P_spread;}
            }
        }
        plateau_width[k] = pw; plateau_spread[k] = ps;
        flags[k] = f;
    }
}
