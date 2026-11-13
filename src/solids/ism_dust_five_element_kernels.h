/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Dust processes of the five-element ISM dust model on plain doubles (yr, cm^-3, K, Msun): carbon and FeMgSiO4 silicate,
   held as dust in C, O, Mg, Si, Fe (index order below), one grain size; no GIZMO state, host and device. */
#ifndef ISM_DUST_FIVE_ELEMENT_KERNELS_H
#define ISM_DUST_FIVE_ELEMENT_KERNELS_H
#include "ism_dust_hu_kernels.h" /* Nozawa+06 thermal sputtering fits */

#define DUST5_A_UM             0.1   /* grain radius [micron] */
#define DUST5_STICK_MIN        1e-4  /* no growth below this sticking */
#define DUST5_SEED_DGR         0.01  /* fresh-start dust-to-gas ratio at Z = DUST5_SEED_ZSUN */
#define DUST5_SEED_ZSUN        0.014
#define DUST5_SEED_C_FRAC      0.3   /* fresh-start split: carbon, the rest silicate */
#define DUST5_SEED_MAXDEPL_C   0.5   /* at most this of the cell's C in seed dust */
#define DUST5_SEED_MAXDEPL_SIL 0.9   /* at most this of the cell's O, Mg, Si, Fe in seed silicate */
#define DUST5_MCKEE_EFF        0.4   /* McKee 1989 destruction efficiency in the swept mass */
#define DUST5_MCKEE_MSUN       2460. /* mass swept above 100 km/s per 1e51 erg at n = 1 cm^-3 (McKee 1989, Cioffi+88) */

/* d[], z[]: 0 C, 1 O, 2 Mg, 3 Si, 4 Fe; frac[]: silicate mass fractions of O, Mg, Si, Fe */

/* Hirashita 2000 */
DUSTK_INLINE double dust5_sticking(double T, double T_dust) {return 1.0 / (1.0 + 0.04 * sqrt(T + T_dust));}

/* thermal sputtering: carbon and silicate fall as exp(-3 n_H Y(T) dt / a) */
DUSTK_INLINE void dust5_sputter(double d[5], double nH, double T, double dt_yr)
{
    double rC = 3.0 * nH / DUST5_A_UM * dustk_Yth_C(T), rS = 3.0 * nH / DUST5_A_UM * dustk_Yth_Sil(T);
    if(rC > 0) {d[0] *= exp(-dt_yr * rC);}
    if(rS > 0) {double f = exp(-dt_yr * rS); for(int k = 1; k < 5; k++) {d[k] *= f;}}
}

/* growth from the gas phase (Dwek 1998 rates as gizmo2017, Hirashita sticking): carbon from C, silicate keyed on Si with the
   stoichiometric O, Mg, Fe, each capped by its gas phase; returns the dust mass fraction grown */
DUSTK_INLINE double dust5_grow(double d[5], const double z[5], const double frac[4], double nH, double T, double T_dust, double dt_yr)
{
    double S = dust5_sticking(T, T_dust), total = d[0] + d[1] + d[2] + d[3] + d[4], g[5], grown = 0;
    if(!(S > DUST5_STICK_MIN && total > 0)) {return 0;}
    for(int k = 0; k < 5; k++) {g[k] = fmax(z[k] - d[k], 0);}
    double fC = (z[0] > 0) ? fmin(d[0] / z[0], 1.0) : 1.0, fSi = (z[3] > 0) ? fmin(d[3] / z[3], 1.0) : 1.0, base = nH / DUST5_A_UM * sqrt(T) * S;
    double rC = 2.15e-12 * base * (g[0] / 2.07e-3) * (1.0 - fC), rSi = 2.01e-12 * base * (g[3] / 6.83e-4) * (1.0 - fSi); /* [1/yr] */
    if(rC > 0 && g[0] > 0 && d[0] > 0) {double dC = fmin(d[0] * expm1(dt_yr * rC), g[0]); d[0] += dC; grown += dC;}
    if(rSi > 0 && g[3] > 0 && d[3] > 0) {
        double dSil = fmin(d[3] * expm1(dt_yr * rSi), g[3]) / frac[2];
        for(int k = 1; k < 5; k++) {dSil = fmin(dSil, g[k] / frac[k - 1]);}
        if(dSil > 0) {for(int k = 1; k < 5; k++) {d[k] += dSil * frac[k - 1];} grown += dSil;}
    }
    return grown;
}

/* dust inside its element pool; silicate scaled as a whole */
DUSTK_INLINE void dust5_clamp(double d[5], const double z[5])
{
    for(int k = 0; k < 5; k++) {if(d[k] < 0) {d[k] = 0;}}
    d[0] = fmin(d[0], fmax(z[0], 0));
    double f = 1;
    for(int k = 1; k < 5; k++) {if(d[k] > 0) {f = fmin(f, fmax(z[k], 0) / d[k]);}}
    if(f < 1) {for(int k = 1; k < 5; k++) {d[k] *= f;}}
}

/* fresh start: dust-to-gas ratio 0.01 Z / 0.014, 30% carbon, 70% silicate, capped by the cell's own elements */
DUSTK_INLINE void dust5_seed(double d[5], const double z[5], const double frac[4], double Z)
{
    double dgr = DUST5_SEED_DGR * fmax(Z, 0) / DUST5_SEED_ZSUN, sil = (1.0 - DUST5_SEED_C_FRAC) * dgr;
    for(int k = 1; k < 5; k++) {sil = fmin(sil, DUST5_SEED_MAXDEPL_SIL * fmax(z[k], 0) / frac[k - 1]);}
    d[0] = fmin(DUST5_SEED_C_FRAC * dgr, DUST5_SEED_MAXDEPL_C * fmax(z[0], 0));
    for(int k = 1; k < 5; k++) {d[k] = sil * frac[k - 1];}
}

/* fraction of a cell's dust a supernova destroys (McKee 1989): efficiency x swept mass / cell mass, at most 1 */
DUSTK_INLINE double dust5_swept_mass_destroyed_fraction(double n0, double E51, double M_cell_msun)
{
    if(!(n0 > 0 && M_cell_msun > 0 && E51 > 0)) {return 0;}
    return fmin(DUST5_MCKEE_EFF * DUST5_MCKEE_MSUN * E51 / pow(n0, 0.1) / M_cell_msun, 1.0);
}

#endif
