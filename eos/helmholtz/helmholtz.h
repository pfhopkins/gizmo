/* C++ port of the Timmes & Swesty (2000) Helmholtz free energy EOS.
   Ported from the Fortran90 implementation by Frank Timmes, wrapped by David Radice.
   This version is thread-safe (no global mutable state) and GPU-buffer-safe
   (all inputs/outputs are pure value types).

   References: Timmes & Swesty, ApJS 126, 501 (2000)
               Cox & Giuli, "Principles of Stellar Structure" (1968), Chapter 24
*/

#ifndef EOS_HELMHOLTZ_H
#define EOS_HELMHOLTZ_H

#ifdef EOS_HELMHOLTZ

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
   --------------------------------------------------------------------------- */

/* Read the Helmholtz table from file. Returns 0 on success, nonzero on failure.
   Must be called once before any helm_eos calls. The table is stored in *tab. */
int helm_read_table(const char *filename, HelmTable *tab);

/* Evaluate the EOS at a single point. Thread-safe: only reads from *tab (const).
   Returns 0 on success, nonzero if inputs are out of table range. */
int helm_eos_from_temp(const HelmTable *tab, const HelmInput *in, HelmResult *out);

/* Invert for temperature given (rho, eps, abar, ye) using Newton iteration.
   Returns 0 on success, nonzero if iteration fails to converge. */
int helm_eos_from_energy(const HelmTable *tab, double rho, double eps,
                         double abar, double ye, double temp_guess,
                         HelmResult *out);

/* Query table ranges */
void helm_get_temp_range(const HelmTable *tab, double *tmin, double *tmax);
void helm_get_rhoye_range(const HelmTable *tab, double *rhoye_min, double *rhoye_max);

#endif /* EOS_HELMHOLTZ */
#endif /* EOS_HELMHOLTZ_H */
