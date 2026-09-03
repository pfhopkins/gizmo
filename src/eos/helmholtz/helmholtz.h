/* C++ port of the Timmes & Swesty (2000) Helmholtz free energy EOS.
   Ported from the Fortran90 implementation by Frank Timmes, wrapped by David Radice.
   This version is thread-safe (no global mutable state) and GPU-buffer-safe
   (all inputs/outputs are pure value types).

   References: Timmes & Swesty, ApJS 126, 501 (2000)
               Cox & Giuli, "Principles of Stellar Structure" (1968), Chapter 24
*/

#ifndef EOS_HELMHOLTZ_H
#define EOS_HELMHOLTZ_H

#include "../../GIZMO_config.h"

#ifdef EOS_HELMHOLTZ

#include <cmath>
#include <cstdio>
#include <cstdlib>

/* DMAX/DMIN/IMAX/IMIN and the device annotation come from here rather than from
   whatever happened to be included first: the bodies below are compiled into
   device translation units, and a header whose bodies only work when an
   includer preloaded its macros is a defect this module has already paid for. */
#include "../../declarations/macros.h"


/* table dimensions — must match the helm_table.dat file being read.
   standard table: imax=271, jmax=101 (log10(rho*Ye) from -12 to 15, log10(T) from 3 to 13) */
#define HELM_IMAX 271
#define HELM_JMAX 101

/* ---------------------------------------------------------------------------
   Physical constants (CGS) — from Timmes' helm_const.dek (CODATA 2006)
   --------------------------------------------------------------------------- */
namespace helm_constants {
    constexpr double pi      = 3.1415926535897932384;
    constexpr double g_grav  = 6.6742867e-8;
    constexpr double h_planck= 6.6260689633e-27;
    constexpr double hbar    = 0.5 * h_planck / pi;
    constexpr double qe      = 4.8032042712e-10;
    constexpr double avo     = 6.0221417930e23;
    constexpr double clight  = 2.99792458e10;
    constexpr double kerg    = 1.380650424e-16;
    constexpr double ev2erg  = 1.60217648740e-12;
    constexpr double kev     = kerg / ev2erg;
    constexpr double amu     = 1.66053878283e-24;
    constexpr double mn      = 1.67492721184e-24;
    constexpr double mp      = 1.67262163783e-24;
    constexpr double me      = 9.1093821545e-28;
    constexpr double rbohr   = hbar * hbar / (me * qe * qe);
    constexpr double fine    = qe * qe / (hbar * clight);
    constexpr double hion    = 13.605698140;
    constexpr double ssol    = 5.67051e-5;
    constexpr double asol    = 4.0 * ssol / clight;
    constexpr double weinlam = h_planck * clight / (kerg * 4.965114232);
    constexpr double weinfre = 2.821439372 * kerg / h_planck;
    constexpr double rhonuc  = 2.342e14;

    /* derived constants used in helmeos */
    constexpr double sioncon = (2.0 * pi * amu * kerg) / (h_planck * h_planck);
    constexpr double forth   = 4.0 / 3.0;
    constexpr double forpi   = 4.0 * pi;
    constexpr double kergavo = kerg * avo;
    constexpr double ikavo   = 1.0 / kergavo;
    constexpr double asoli3  = asol / 3.0;
    constexpr double light2  = clight * clight;
    constexpr double third   = 1.0 / 3.0;
    constexpr double esqu    = qe * qe;

    /* Coulomb correction fit coefficients (Yakovlev & Shalybkov 1989) */
    constexpr double a1_coul = -0.898004;
    constexpr double b1_coul =  0.96786;
    constexpr double c1_coul =  0.220703;
    constexpr double d1_coul = -0.86097;
    constexpr double e1_coul =  2.5269;
    constexpr double a2_coul =  0.29561;
    constexpr double b2_coul =  1.9885;
    constexpr double c2_coul =  0.288675;
}


/* ---------------------------------------------------------------------------
   Table storage — read-only after initialization, safe for concurrent access.
   --------------------------------------------------------------------------- */
struct HelmTable {
    /* electron free energy and its 8 partial derivatives */
    double f[HELM_IMAX][HELM_JMAX];
    double fd[HELM_IMAX][HELM_JMAX];     /* d/d(rho*ye) */
    double ft[HELM_IMAX][HELM_JMAX];     /* d/dT */
    double fdd[HELM_IMAX][HELM_JMAX];
    double ftt[HELM_IMAX][HELM_JMAX];
    double fdt[HELM_IMAX][HELM_JMAX];
    double fddt[HELM_IMAX][HELM_JMAX];
    double fdtt[HELM_IMAX][HELM_JMAX];
    double fddtt[HELM_IMAX][HELM_JMAX];

    /* pressure derivative with density */
    double dpdf[HELM_IMAX][HELM_JMAX];
    double dpdfd[HELM_IMAX][HELM_JMAX];
    double dpdft[HELM_IMAX][HELM_JMAX];
    double dpdfdt[HELM_IMAX][HELM_JMAX];

    /* electron chemical potential */
    double ef[HELM_IMAX][HELM_JMAX];
    double efd[HELM_IMAX][HELM_JMAX];
    double eft[HELM_IMAX][HELM_JMAX];
    double efdt[HELM_IMAX][HELM_JMAX];

    /* electron+positron number density */
    double xf[HELM_IMAX][HELM_JMAX];
    double xfd[HELM_IMAX][HELM_JMAX];
    double xft[HELM_IMAX][HELM_JMAX];
    double xfdt[HELM_IMAX][HELM_JMAX];

    /* density and temperature grid points */
    double d[HELM_IMAX];    /* ye*rho grid (CGS) */
    double t[HELM_JMAX];    /* temperature grid (K) */

    /* grid spacings and their inverses (precomputed) */
    double tlo, thi, tstp, tstpi;
    double dlo, dhi, dstp, dstpi;

    double dt_sav[HELM_JMAX],  dti_sav[HELM_JMAX],  dt2_sav[HELM_JMAX],  dt2i_sav[HELM_JMAX],  dt3i_sav[HELM_JMAX];
    double dd_sav[HELM_IMAX],  ddi_sav[HELM_IMAX],  dd2_sav[HELM_IMAX],  dd2i_sav[HELM_IMAX],  dd3i_sav[HELM_IMAX];

    int initialized;
};


/* ---------------------------------------------------------------------------
   Input/output structs — pure value types, GPU-buffer-safe.
   --------------------------------------------------------------------------- */

/* Full input to the Helmholtz EOS (all quantities in CGS). */
struct HelmInput {
    double rho;         /* density [g/cm^3] */
    double temp;        /* temperature [K] */
    double abar;        /* mean atomic weight [amu] */
    double ye;          /* electron fraction Ye = Z/A */
};

/* Full output from the Helmholtz EOS (all quantities in CGS). */
struct HelmResult {
    /* totals */
    double ptot;        /* total pressure [erg/cm^3] */
    double etot;        /* total specific internal energy [erg/g] */
    double stot;        /* total specific entropy [erg/g/K] */
    double temp;        /* temperature [K] (set by from_temp; converged value from from_energy) */
    double cv;          /* specific heat at constant volume [erg/g/K] */
    double cp;          /* specific heat at constant pressure [erg/g/K] */
    double csound;      /* relativistic sound speed [cm/s] */
    double gam1;        /* first adiabatic exponent */

    /* electron-positron chemical potential */
    double etaele;      /* normalized electron chemical potential eta_e */

    /* derivatives (first order) */
    double dpresdd;     /* dP/drho */
    double dpresdt;     /* dP/dT */
    double denerdt;     /* de/dT = cv */

    /* individual contributions (for diagnostics) */
    double prad, pion, pele, pcoul;
    double erad, eion, eele, ecoul;
};


/* ---------------------------------------------------------------------------
   Public API

   Reading the table is file I/O and stays on the host, in helmholtz.cc.
   Everything else is arithmetic over the const table and is defined below, in
   this header, so a device kernel can call it: without relocatable device code
   a kernel cannot resolve a call into another translation unit, so the body has
   to be visible where it is used. The evaluation is a Newton iteration over
   table lookups, i.e. exactly the per-element work that makes staging a batch
   to the device worth its cost.
   --------------------------------------------------------------------------- */

/* Read the Helmholtz table from file. Returns 0 on success, nonzero on failure.
   Must be called once before any helm_eos calls. The table is stored in *tab. */
int helm_read_table(const char *filename, HelmTable *tab);

/* The live table, owned by eos/eos_interface.cc. Host-only: call it before a
   dispatch and let the kernel capture the pointer, exactly as the cooling
   tables are handed out. Null until eos_init has run. */
const HelmTable *helm_table_view(void);

/* =========================================================================
   Range queries
   ========================================================================= */

KOKKOS_INLINE_FUNCTION
void helm_get_temp_range(const HelmTable *tab, double *tmin, double *tmax)
{
    *tmin = tab->t[0];
    *tmax = tab->t[HELM_JMAX - 1];
}

KOKKOS_INLINE_FUNCTION
void helm_get_rhoye_range(const HelmTable *tab, double *rhoye_min, double *rhoye_max)
{
    *rhoye_min = tab->d[0];
    *rhoye_max = tab->d[HELM_IMAX - 1];
}

/* =========================================================================
   Quintic Hermite polynomial basis functions and their derivatives
   ========================================================================= */

static KOKKOS_INLINE_FUNCTION double psi0(double z) {
    return z*z*z * (z * (-6.0*z + 15.0) - 10.0) + 1.0;
}
static KOKKOS_INLINE_FUNCTION double dpsi0(double z) {
    return z*z * (z * (-30.0*z + 60.0) - 30.0);
}
static KOKKOS_INLINE_FUNCTION double ddpsi0(double z) {
    return z * (z * (-120.0*z + 180.0) - 60.0);
}

static KOKKOS_INLINE_FUNCTION double psi1(double z) {
    return z * (z*z * (z * (-3.0*z + 8.0) - 6.0) + 1.0);
}
static KOKKOS_INLINE_FUNCTION double dpsi1(double z) {
    return z*z * (z * (-15.0*z + 32.0) - 18.0) + 1.0;
}
static KOKKOS_INLINE_FUNCTION double ddpsi1(double z) {
    return z * (z * (-60.0*z + 96.0) - 36.0);
}

static KOKKOS_INLINE_FUNCTION double psi2(double z) {
    return 0.5*z*z * (z * (z * (-z + 3.0) - 3.0) + 1.0);
}
static KOKKOS_INLINE_FUNCTION double dpsi2(double z) {
    return 0.5*z * (z * (z * (-5.0*z + 12.0) - 9.0) + 2.0);
}
static KOKKOS_INLINE_FUNCTION double ddpsi2(double z) {
    return 0.5 * (z * (z * (-20.0*z + 36.0) - 18.0) + 2.0);
}

/* Cubic Hermite basis functions */
static KOKKOS_INLINE_FUNCTION double xpsi0(double z) {
    return z*z * (2.0*z - 3.0) + 1.0;
}
static KOKKOS_INLINE_FUNCTION double xdpsi0(double z) {
    return z * (6.0*z - 6.0);
}
static KOKKOS_INLINE_FUNCTION double xpsi1(double z) {
    return z * (z * (z - 2.0) + 1.0);
}
static KOKKOS_INLINE_FUNCTION double xdpsi1(double z) {
    return z * (3.0*z - 4.0) + 1.0;
}


/* =========================================================================
   Biquintic Hermite interpolation (6x6 = 36 coefficients)
   ========================================================================= */
static KOKKOS_INLINE_FUNCTION double h5eval(const double fi[36],
        double w0t, double w1t, double w2t, double w0mt, double w1mt, double w2mt,
        double w0d, double w1d, double w2d, double w0md, double w1md, double w2md)
{
    return fi[0]  * w0d*w0t   + fi[1]  * w0md*w0t
         + fi[2]  * w0d*w0mt  + fi[3]  * w0md*w0mt
         + fi[4]  * w0d*w1t   + fi[5]  * w0md*w1t
         + fi[6]  * w0d*w1mt  + fi[7]  * w0md*w1mt
         + fi[8]  * w0d*w2t   + fi[9]  * w0md*w2t
         + fi[10] * w0d*w2mt  + fi[11] * w0md*w2mt
         + fi[12] * w1d*w0t   + fi[13] * w1md*w0t
         + fi[14] * w1d*w0mt  + fi[15] * w1md*w0mt
         + fi[16] * w2d*w0t   + fi[17] * w2md*w0t
         + fi[18] * w2d*w0mt  + fi[19] * w2md*w0mt
         + fi[20] * w1d*w1t   + fi[21] * w1md*w1t
         + fi[22] * w1d*w1mt  + fi[23] * w1md*w1mt
         + fi[24] * w2d*w1t   + fi[25] * w2md*w1t
         + fi[26] * w2d*w1mt  + fi[27] * w2md*w1mt
         + fi[28] * w1d*w2t   + fi[29] * w1md*w2t
         + fi[30] * w1d*w2mt  + fi[31] * w1md*w2mt
         + fi[32] * w2d*w2t   + fi[33] * w2md*w2t
         + fi[34] * w2d*w2mt  + fi[35] * w2md*w2mt;
}

/* Bicubic Hermite interpolation (4x4 = 16 coefficients) */
static KOKKOS_INLINE_FUNCTION double h3eval(const double fi[16],
        double w0t, double w1t, double w0mt, double w1mt,
        double w0d, double w1d, double w0md, double w1md)
{
    return fi[0]  * w0d*w0t   + fi[1]  * w0md*w0t
         + fi[2]  * w0d*w0mt  + fi[3]  * w0md*w0mt
         + fi[4]  * w0d*w1t   + fi[5]  * w0md*w1t
         + fi[6]  * w0d*w1mt  + fi[7]  * w0md*w1mt
         + fi[8]  * w1d*w0t   + fi[9]  * w1md*w0t
         + fi[10] * w1d*w0mt  + fi[11] * w1md*w0mt
         + fi[12] * w1d*w1t   + fi[13] * w1md*w1t
         + fi[14] * w1d*w1mt  + fi[15] * w1md*w1mt;
}

/* =========================================================================
   Core EOS evaluation from (rho, T, abar, Ye)
   Pure function of the const table + value-type input.
   ========================================================================= */

KOKKOS_INLINE_FUNCTION
int helm_eos_from_temp(const HelmTable *tab, const HelmInput *in, HelmResult *out)
{
    using namespace helm_constants;   /* was a file-scope directive in helmholtz.cc */
    const double temp = in->temp;
    const double den  = in->rho;
    const double abar = in->abar;
    const double ye   = DMAX(1.0e-16, in->ye);
    const double zbar = ye * abar;

    const double ytot1 = 1.0 / abar;
    const double deni  = 1.0 / den;
    const double tempi = 1.0 / temp;
    const double kt    = kerg * temp;
    const double ktinv = 1.0 / kt;

    /* table coordinate: ye * den */
    const double din = ye * den;

    /* bounds checking */
    if (temp > tab->t[HELM_JMAX - 1] || temp < tab->t[0] ||
        din  > tab->d[HELM_IMAX - 1] || din  < tab->d[0]) {
        return 1; /* out of range */
    }

    /* ---- Radiation contribution ---- */
    double prad    = asoli3 * temp * temp * temp * temp;
    double dpraddt = 4.0 * prad * tempi;

    double erad    = 3.0 * prad * deni;
    double deraddd = -erad * deni;
    double deraddt = 3.0 * dpraddt * deni;

    double srad    = (prad * deni + erad) * tempi;

    /* ---- Ion contribution (ideal gas) ---- */
    double xni     = avo * ytot1 * den;
    double dxnidd  = avo * ytot1;
    double dxnida  = -xni * ytot1;

    double pion    = xni * kt;
    double dpiondd = dxnidd * kt;
    double dpiondt = xni * kerg;
    double dpionda = dxnida * kt;

    double eion    = 1.5 * pion * deni;
    double deiondd = (1.5 * dpiondd - eion) * deni;
    double deiondt = 1.5 * dpiondt * deni;
    double deionda = 1.5 * dpionda * deni;

    /* Sackur-Tetrode ion entropy */
    double x = abar * abar * sqrt(abar) * deni / avo;
    double s = sioncon * temp;
    double z = x * s * sqrt(s);
    double y = log(z);

    double sion = (pion * deni + eion) * tempi + kergavo * ytot1 * y;

    /* ---- Electron-positron section via table interpolation ---- */

    /* hash locate temperature and density in table */
    int jat = (int)((log10(temp) - tab->tlo) * tab->tstpi);
    jat = IMAX(0, IMIN(jat, HELM_JMAX - 2));
    int iat = (int)((log10(din) - tab->dlo) * tab->dstpi);
    iat = IMAX(0, IMIN(iat, HELM_IMAX - 2));

    /* load the 36 free energy table values at this grid cell */
    double fi[36];
    fi[0]  = tab->f[iat][jat];        fi[1]  = tab->f[iat+1][jat];
    fi[2]  = tab->f[iat][jat+1];      fi[3]  = tab->f[iat+1][jat+1];
    fi[4]  = tab->ft[iat][jat];       fi[5]  = tab->ft[iat+1][jat];
    fi[6]  = tab->ft[iat][jat+1];     fi[7]  = tab->ft[iat+1][jat+1];
    fi[8]  = tab->ftt[iat][jat];      fi[9]  = tab->ftt[iat+1][jat];
    fi[10] = tab->ftt[iat][jat+1];    fi[11] = tab->ftt[iat+1][jat+1];
    fi[12] = tab->fd[iat][jat];       fi[13] = tab->fd[iat+1][jat];
    fi[14] = tab->fd[iat][jat+1];     fi[15] = tab->fd[iat+1][jat+1];
    fi[16] = tab->fdd[iat][jat];      fi[17] = tab->fdd[iat+1][jat];
    fi[18] = tab->fdd[iat][jat+1];    fi[19] = tab->fdd[iat+1][jat+1];
    fi[20] = tab->fdt[iat][jat];      fi[21] = tab->fdt[iat+1][jat];
    fi[22] = tab->fdt[iat][jat+1];    fi[23] = tab->fdt[iat+1][jat+1];
    fi[24] = tab->fddt[iat][jat];     fi[25] = tab->fddt[iat+1][jat];
    fi[26] = tab->fddt[iat][jat+1];   fi[27] = tab->fddt[iat+1][jat+1];
    fi[28] = tab->fdtt[iat][jat];     fi[29] = tab->fdtt[iat+1][jat];
    fi[30] = tab->fdtt[iat][jat+1];   fi[31] = tab->fdtt[iat+1][jat+1];
    fi[32] = tab->fddtt[iat][jat];    fi[33] = tab->fddtt[iat+1][jat];
    fi[34] = tab->fddtt[iat][jat+1];  fi[35] = tab->fddtt[iat+1][jat+1];

    /* normalized coordinates within the grid cell */
    double xt  = DMAX((temp - tab->t[jat]) * tab->dti_sav[jat], 0.0);
    double xd  = DMAX((din  - tab->d[iat]) * tab->ddi_sav[iat], 0.0);
    double mxt = 1.0 - xt;
    double mxd = 1.0 - xd;

    /* quintic basis functions for temperature */
    double si0t  =  psi0(xt);
    double si1t  =  psi1(xt) * tab->dt_sav[jat];
    double si2t  =  psi2(xt) * tab->dt2_sav[jat];
    double si0mt =  psi0(mxt);
    double si1mt = -psi1(mxt) * tab->dt_sav[jat];
    double si2mt =  psi2(mxt) * tab->dt2_sav[jat];

    /* quintic basis functions for density */
    double si0d  =  psi0(xd);
    double si1d  =  psi1(xd) * tab->dd_sav[iat];
    double si2d  =  psi2(xd) * tab->dd2_sav[iat];
    double si0md =  psi0(mxd);
    double si1md = -psi1(mxd) * tab->dd_sav[iat];
    double si2md =  psi2(mxd) * tab->dd2_sav[iat];

    /* first derivatives of quintic basis */
    double dsi0t  =  dpsi0(xt) * tab->dti_sav[jat];
    double dsi1t  =  dpsi1(xt);
    double dsi2t  =  dpsi2(xt) * tab->dt_sav[jat];
    double dsi0mt = -dpsi0(mxt) * tab->dti_sav[jat];
    double dsi1mt =  dpsi1(mxt);
    double dsi2mt = -dpsi2(mxt) * tab->dt_sav[jat];

    double dsi0d  =  dpsi0(xd) * tab->ddi_sav[iat];
    double dsi1d  =  dpsi1(xd);
    double dsi2d  =  dpsi2(xd) * tab->dd_sav[iat];
    double dsi0md = -dpsi0(mxd) * tab->ddi_sav[iat];
    double dsi1md =  dpsi1(mxd);
    double dsi2md = -dpsi2(mxd) * tab->dd_sav[iat];

    /* second derivatives for temperature only (needed for df_tt -> cv) */
    double ddsi0t  =  ddpsi0(xt)  * tab->dt2i_sav[jat];
    double ddsi1t  =  ddpsi1(xt)  * tab->dti_sav[jat];
    double ddsi2t  =  ddpsi2(xt);
    double ddsi0mt =  ddpsi0(mxt) * tab->dt2i_sav[jat];
    double ddsi1mt = -ddpsi1(mxt) * tab->dti_sav[jat];
    double ddsi2mt =  ddpsi2(mxt);

    /* evaluate the free energy and its derivatives */
    double free_e = h5eval(fi,
            si0t,  si1t,  si2t,  si0mt,  si1mt,  si2mt,
            si0d,  si1d,  si2d,  si0md,  si1md,  si2md);

    double df_d = h5eval(fi,
            si0t,  si1t,  si2t,  si0mt,  si1mt,  si2mt,
            dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    double df_t = h5eval(fi,
            dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
            si0d,   si1d,   si2d,   si0md,   si1md,   si2md);

    double df_tt = h5eval(fi,
            ddsi0t,  ddsi1t,  ddsi2t,  ddsi0mt,  ddsi1mt,  ddsi2mt,
            si0d,    si1d,    si2d,    si0md,    si1md,    si2md);

    double df_dt = h5eval(fi,
            dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt,
            dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md);


    /* --- Now switch to cubic Hermite for dpdf, ef, xf tables --- */

    /* cubic basis functions for temperature */
    si0t  = xpsi0(xt);
    si1t  = xpsi1(xt) * tab->dt_sav[jat];
    si0mt = xpsi0(mxt);
    si1mt = -xpsi1(mxt) * tab->dt_sav[jat];

    /* cubic basis functions for density */
    si0d  = xpsi0(xd);
    si1d  = xpsi1(xd) * tab->dd_sav[iat];
    si0md = xpsi0(mxd);
    si1md = -xpsi1(mxd) * tab->dd_sav[iat];

    /* cubic derivatives */
    dsi0t  =  xdpsi0(xt) * tab->dti_sav[jat];
    dsi1t  =  xdpsi1(xt);
    dsi0mt = -xdpsi0(mxt) * tab->dti_sav[jat];
    dsi1mt =  xdpsi1(mxt);

    dsi0d  =  xdpsi0(xd) * tab->ddi_sav[iat];
    dsi1d  =  xdpsi1(xd);
    dsi0md = -xdpsi0(mxd) * tab->ddi_sav[iat];
    dsi1md =  xdpsi1(mxd);


    /* pressure derivative with density (bicubic) */
    double fi16[16];
    fi16[0]  = tab->dpdf[iat][jat];      fi16[1]  = tab->dpdf[iat+1][jat];
    fi16[2]  = tab->dpdf[iat][jat+1];    fi16[3]  = tab->dpdf[iat+1][jat+1];
    fi16[4]  = tab->dpdft[iat][jat];     fi16[5]  = tab->dpdft[iat+1][jat];
    fi16[6]  = tab->dpdft[iat][jat+1];   fi16[7]  = tab->dpdft[iat+1][jat+1];
    fi16[8]  = tab->dpdfd[iat][jat];     fi16[9]  = tab->dpdfd[iat+1][jat];
    fi16[10] = tab->dpdfd[iat][jat+1];   fi16[11] = tab->dpdfd[iat+1][jat+1];
    fi16[12] = tab->dpdfdt[iat][jat];    fi16[13] = tab->dpdfdt[iat+1][jat];
    fi16[14] = tab->dpdfdt[iat][jat+1];  fi16[15] = tab->dpdfdt[iat+1][jat+1];

    double dpepdd = h3eval(fi16,
            si0t, si1t, si0mt, si1mt,
            si0d, si1d, si0md, si1md);
    dpepdd = DMAX(ye * dpepdd, 1.0e-30);


    /* electron chemical potential (bicubic) */
    fi16[0]  = tab->ef[iat][jat];      fi16[1]  = tab->ef[iat+1][jat];
    fi16[2]  = tab->ef[iat][jat+1];    fi16[3]  = tab->ef[iat+1][jat+1];
    fi16[4]  = tab->eft[iat][jat];     fi16[5]  = tab->eft[iat+1][jat];
    fi16[6]  = tab->eft[iat][jat+1];   fi16[7]  = tab->eft[iat+1][jat+1];
    fi16[8]  = tab->efd[iat][jat];     fi16[9]  = tab->efd[iat+1][jat];
    fi16[10] = tab->efd[iat][jat+1];   fi16[11] = tab->efd[iat+1][jat+1];
    fi16[12] = tab->efdt[iat][jat];    fi16[13] = tab->efdt[iat+1][jat];
    fi16[14] = tab->efdt[iat][jat+1];  fi16[15] = tab->efdt[iat+1][jat+1];

    double etaele_val = h3eval(fi16,
            si0t, si1t, si0mt, si1mt,
            si0d, si1d, si0md, si1md);

    double detadt = h3eval(fi16,
            dsi0t, dsi1t, dsi0mt, dsi1mt,
            si0d,  si1d,  si0md,  si1md);


    /* electron+positron number density (bicubic) — not needed for outputs but used diagnostically */
    /* (skipped — only etaele is stored in output) */


    /* electron-positron thermodynamic quantities from the free energy */
    x = din * din;
    double pele_val = x * df_d;
    double dpepdt   = x * df_dt;
    /* dpepdd already computed via bicubic above (more stable at high T / low rho) */
    s = dpepdd / ye - 2.0 * din * df_d;
    double dpepda = -ytot1 * (2.0 * pele_val + s * din);
    double dpepdz =  den * ytot1 * (2.0 * din * df_d + s);

    double sele_val = -df_t * ye;
    double dsepdt   = -df_tt * ye;
    double dsepdd   = -df_dt * ye * ye;

    double eele_val = ye * free_e + temp * sele_val;
    double deepdt   = temp * dsepdt;
    double deepdd   = ye * ye * df_d + temp * dsepdd;


    /* ---- Coulomb corrections (Yakovlev & Shalybkov 1989) ---- */
    double pcoul = 0.0, ecoul = 0.0, scoul = 0.0;
    double dpcouldd = 0.0, dpcouldt = 0.0;
    double decouldd = 0.0, decouldt = 0.0;

    {
        z = forth * pi;
        s = z * xni;
        double dsdd = z * dxnidd;

        double lami    = 1.0 / pow(s, third);
        double inv_lami = 1.0 / lami;
        z = -third * lami;
        double lamidd = z * dsdd / s;

        double plasg   = zbar * zbar * esqu * ktinv * inv_lami;
        z = -plasg * inv_lami;
        double plasgdd = z * lamidd;
        double plasgdt = -plasg * ktinv * kerg;
        double plasgdz = 2.0 * plasg / zbar;

        if (plasg >= 1.0) {
            /* strong coupling (eqs 82, 85, 86, 87) */
            x = pow(plasg, 0.25);
            y = avo * ytot1 * kerg;
            ecoul = y * temp * (a1_coul*plasg + b1_coul*x + c1_coul/x + d1_coul);
            pcoul = third * den * ecoul;
            scoul = -y * (3.0*b1_coul*x - 5.0*c1_coul/x
                        + d1_coul * (log(plasg) - 1.0) - e1_coul);

            y = avo * ytot1 * kt * (a1_coul + 0.25/plasg*(b1_coul*x - c1_coul/x));
            decouldd = y * plasgdd;
            decouldt = y * plasgdt + ecoul / temp;

            y = third * den;
            dpcouldd = third * ecoul + y * decouldd;
            dpcouldt = y * decouldt;

        } else {
            /* weak coupling (eqs 102, 103, 104) */
            x = plasg * sqrt(plasg);
            y = pow(plasg, b2_coul);
            z = c2_coul * x - third * a2_coul * y;
            pcoul = -pion * z;
            ecoul = 3.0 * pcoul * deni;
            scoul = -avo / abar * kerg * (c2_coul*x - a2_coul*(b2_coul-1.0)/b2_coul*y);

            s = 1.5*c2_coul*x/plasg - third*a2_coul*b2_coul*y/plasg;
            dpcouldd = -dpiondd*z - pion*s*plasgdd;
            dpcouldt = -dpiondt*z - pion*s*plasgdt;

            s = 3.0 * deni;
            decouldd = s * dpcouldd - ecoul * deni;
            decouldt = s * dpcouldt;
        }

        /* if Coulomb corrections make totals negative, zero them out */
        x = prad + pion + pele_val + pcoul;
        y = erad + eion + eele_val + ecoul;
        z = srad + sion + sele_val + scoul;
        if (x <= 0.0 || y <= 0.0 || z <= 0.0) {
            pcoul = 0.0; ecoul = 0.0; scoul = 0.0;
            dpcouldd = 0.0; dpcouldt = 0.0;
            decouldd = 0.0; decouldt = 0.0;
        }
    }


    /* ---- Sum all contributions ---- */
    double pgas = pion + pele_val + pcoul;
    double egas = eion + eele_val + ecoul;

    double dpgasdd = dpiondd + dpepdd + dpcouldd;
    double dpgasdt = dpiondt + dpepdt + dpcouldt;
    double degasdd = deiondd + deepdd + decouldd;
    double degasdt = deiondt + deepdt + decouldt;

    double pres = prad + pgas;
    double ener = erad + egas;
    double entr = srad + sion + sele_val + scoul;

    double dpresdd = dpgasdd;                 /* dpraddd = 0 */
    double dpresdt = dpraddt + dpgasdt;
    double denerdd = deraddd + degasdd;
    double denerdt = deraddt + degasdt;

    /* ---- Derived thermodynamic quantities ---- */
    double zz   = pres * deni;
    double zzi  = den / pres;
    double chit = temp / pres * dpresdt;
    double chid = dpresdd * zzi;
    double cv_val = denerdt;
    x = zz * chit / (temp * cv_val);
    double gam1_val = chit * x + chid;
    double cp_val = cv_val * gam1_val / chid;
    z = 1.0 + (ener + light2) * zzi;
    double sound = clight * sqrt(gam1_val / z);


    /* ---- Fill output ---- */
    out->ptot    = pres;
    out->etot    = ener;
    out->stot    = entr;
    out->temp    = temp;
    out->cv      = cv_val;
    out->cp      = cp_val;
    out->csound  = sound;
    out->gam1    = gam1_val;
    out->etaele  = etaele_val;

    out->dpresdd = dpresdd;
    out->dpresdt = dpresdt;
    out->denerdt = denerdt;

    out->prad  = prad;   out->pion  = pion;   out->pele  = pele_val;  out->pcoul = pcoul;
    out->erad  = erad;   out->eion  = eion;   out->eele  = eele_val;  out->ecoul = ecoul;

    return 0;
}

/* =========================================================================
   Temperature inversion: given (rho, eps, abar, ye), find T via Newton.
   Mirrors the logic in helm_wrap.f90 helm_eos_e_c.
   ========================================================================= */

KOKKOS_INLINE_FUNCTION
int helm_eos_from_energy(const HelmTable *tab, double rho, double eps,
                         double abar, double ye, double temp_guess,
                         HelmResult *out)
{
    const int maxit = 100;
    const double prec = 1.0e-10;

    double tmin, tmax;
    helm_get_temp_range(tab, &tmin, &tmax);

    /* set up initial guess */
    HelmInput in;
    in.rho  = rho;
    in.abar = abar;
    in.ye   = ye;
    in.temp = temp_guess;

    /* clamp initial guess to table range */
    if (in.temp < tmin || in.temp > tmax) {
        /* fallback: ideal gas guess from gamma=5/3 electron gas */
        in.temp = (2.0/3.0) * abar * eps * helm_constants::me / helm_constants::kerg;
        in.temp = DMAX(tmin, DMIN(in.temp, tmax));
    }

    int ierr = helm_eos_from_temp(tab, &in, out);
    if (ierr) return ierr;

    double delta = fabs(out->etot - eps);
    if (delta < prec * fabs(eps)) return 0;

    for (int it = 0; it < maxit; it++) {
        if (out->cv <= 0.0) return 2; /* degenerate cv */
        double dt = -(out->etot - eps) / out->cv;
        in.temp += dt;
        in.temp = DMAX(tmin, DMIN(in.temp, tmax));

        ierr = helm_eos_from_temp(tab, &in, out);
        if (ierr) return ierr;

        delta = fabs(out->etot - eps);
        if (delta < prec * fabs(eps)) return 0;
    }

    /* One call, not two: this body now compiles for the device, where stderr
       does not exist and a multi-part message from concurrent work items would
       interleave into nonsense. Same condition, same numbers. */
    printf("helm_eos_from_energy: Newton iteration did not converge"
           " (rel err = %e, required = %e)\n", delta/fabs(eps), prec);
    return 3;
}

#endif /* EOS_HELMHOLTZ */
#endif /* EOS_HELMHOLTZ_H */
