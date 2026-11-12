/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef RESOLVEDISM_POPIII_H
#define RESOLVEDISM_POPIII_H
/* Pop III star formation (GALSF_RESOLVEDISM_POPIII_STARS): pristine-gas switch, SF criterion and IMF.
 * Pure functions, shared by sfr_eff.cc and the unit/one-zone tests.
 * Pop III stars use the lowest stellar-table row (Z = 1e-4 absolute) until stella-tables has a Z = 0 row. */
#include <math.h>

struct popIII_sf_params {
    double Zcrit_abs;       /* gas below this total Z (absolute) is pristine */
    double nH_min;          /* n_H threshold [cm^-3] */
    double xH2_min;         /* minimum n(H2)/n_H */
    double Mreservoir_msun; /* Jeans mass must be below the mass the accretion walk may gather [Msun] */
};

static inline int popIII_gas_is_pristine(double Z_abs, double Zcrit_abs) {return Z_abs < Zcrit_abs;}

/* pristine cell may form a Pop III star: not ionised, dense, converging, molecular enough, Jeans-unstable at the reservoir scale */
static inline int popIII_sf_allowed(double nH, double divv_phys, double xH2, double MJ_msun, int ionized, const struct popIII_sf_params *p)
{
    if(ionized) {return 0;}
    return (nH > p->nH_min) && (divv_phys < 0) && (xH2 > p->xH2_min) && (MJ_msun < p->Mreservoir_msun);
}

enum {POPIII_IMF_CHARACTERISTIC_MASS = 0, POPIII_IMF_LOG_FLAT = 1};

/* number IMF dN/dlnM (unnormalised): M^-1.3 exp[-(Mchar/M)^1.6] (Wise+12a form) or flat in ln M */
static inline double popIII_imf_dNdlnM(double M, int type, double Mchar)
{
    if(type == POPIII_IMF_LOG_FLAT) {return 1.0;}
    return pow(M, -1.3) * exp(-pow(Mchar / M, 1.6));
}

/* mass-weighted draw (density in ln M proportional to M dN/dlnM), so the LYRA 1/M acceptance yields dN/dlnM; uniform() returns [0,1) */
template<class U> static inline double popIII_draw_mass_weighted(U &&uniform, int type, double Mchar, double Mlo, double Mhi)
{
    if(type == POPIII_IMF_LOG_FLAT) {return Mlo + uniform() * (Mhi - Mlo);} /* M dN/dlnM = M: uniform in M */
    double Mpk = Mchar * pow(1.6 / 0.3, 1.0 / 1.6); /* maximum of M^-0.3 exp[-(Mchar/M)^1.6] */
    if(Mpk < Mlo) {Mpk = Mlo;}
    if(Mpk > Mhi) {Mpk = Mhi;}
    double gmax = Mpk * popIII_imf_dNdlnM(Mpk, type, Mchar), lnlo = log(Mlo), lnhi = log(Mhi), M;
    do {M = exp(lnlo + uniform() * (lnhi - lnlo));} while(uniform() * gmax > M * popIII_imf_dNdlnM(M, type, Mchar));
    return M;
}

#endif
