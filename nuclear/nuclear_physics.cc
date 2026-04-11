/* Built-in aprox13 alpha-chain nuclear reaction network.
   13 species: He4, C12, O16, Ne20, Mg24, Si28, S32, Ar36, Ca40, Ti44, Cr48, Fe52, Ni56.

   Physics: alpha-capture chain reactions (triple-alpha, C12(a,g)O16, ..., Fe52(a,g)Ni56)
   plus their inverse photo-disintegration rates from detailed balance.

   Rates: CF88 (Caughlan & Fowler 1988) analytic rate formulae, which are the standard
   reference rates for the aprox13 network as used by Timmes (1999).

   Solver: backward-Euler with Newton iteration. The 13x13 Jacobian dF/dY is computed
   analytically from the reaction rate expressions. Internal subcycling handles
   extreme stiffness where the Newton step alone is insufficient.

   References:
     - Timmes 1999, ApJS 124, 241 (aprox13 network and its implementation)
     - Caughlan & Fowler 1988, ADNDT 40, 283 (CF88 thermonuclear reaction rates)
     - Fowler, Caughlan & Zimmerman 1975, ARAA 13, 69 (FCZ rate formalism)

   Thread-safe: all mutable state is stack-local. No global arrays modified.
*/

/* When compiled by nvcc (GPU builds), this file is #included from nuclear.cc
   to form a single TU (required because nvcc without -rdc can't resolve
   cross-TU device calls).  Skip the redundant includes in that case. */
#ifndef NUCLEAR_PHYSICS_INCLUDED_FROM_NUCLEAR_CC
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "nuclear.h"
#endif

#ifdef NUCLEAR_NETWORK

/* =========================================================================
   Ye / Abar computation from mass fractions (pure function, used by all solvers).
   ========================================================================= */
KOKKOS_FUNCTION void nuclear_compute_ye_abar(const double X[NUM_NUCLEAR_SPECIES],
                             double *Ye_out, double *Abar_out)
{
    double sum_X_over_A = 0, sum_Z_X_over_A = 0;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        double x = X[k] / (double) nuclear_aprox13_A[k];
        sum_X_over_A   += x;
        sum_Z_X_over_A += (double) nuclear_aprox13_Z[k] * x;
    }
    if (sum_X_over_A > 0) {
        *Abar_out = 1.0 / sum_X_over_A;
        *Ye_out   = sum_Z_X_over_A;
    } else {
        *Abar_out = 4.0; *Ye_out = 0.5;
    }
}

#if !defined(NUCLEAR_NETWORK_SOLVER) || (NUCLEAR_NETWORK_SOLVER == 0)

/* =========================================================================
   Physical constants (CGS)
   ========================================================================= */
static constexpr double MeV_to_erg = 1.602176634e-6;
static constexpr double amu_cgs    = 1.66053906660e-24;
static constexpr double avo        = 6.02214076e23;
static constexpr double kerg       = 1.380649e-16;
static constexpr double hbar_cgs   = 1.054571817e-27;
static constexpr double clight_cgs = 2.99792458e10;
static constexpr double pi_val     = 3.14159265358979323846;

/* Q-values [MeV] for the 12 reactions: Q = BE(product) - BE(reactants).
   Computed from AME2020 mass excess values. */
static const double Q_MeV[] = {
    7.2748,   /* 3 He4 -> C12 */
    7.1616,   /* C12(a,g)O16 */
    4.7300,   /* O16(a,g)Ne20 */
    9.3166,   /* Ne20(a,g)Mg24 */
    9.9842,   /* Mg24(a,g)Si28 */
    6.9483,   /* Si28(a,g)S32 */
    6.6413,   /* S32(a,g)Ar36 */
    7.0404,   /* Ar36(a,g)Ca40 */
    5.1267,   /* Ca40(a,g)Ti44 */
    7.6938,   /* Ti44(a,g)Cr48 */
    7.9381,   /* Cr48(a,g)Fe52 */
    8.0068,   /* Fe52(a,g)Ni56 */
};

static constexpr int NS = 13;  /* number of species */
static constexpr int NR = 12;  /* number of reactions */

/* Species A and Z (from nuclear.h) */
static const int A_sp[] = {4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56};
static const int Z_sp[] = {2,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28};


/* =========================================================================
   CF88 thermonuclear reaction rates.
   Each function returns the reaction rate in cm^3/s/mol (for 2-body)
   or cm^6/s/mol^2 (for 3-body, i.e. triple-alpha).
   Temperature T9 is in units of 10^9 K.
   ========================================================================= */

/* Triple-alpha rate: 3 He4 -> C12 (CF88 eqs 15-17).
   Returns the effective 3-body rate coefficient <sigma_v>_3a in cm^6/s.
   The caller (nuclear_rhs) computes: dY_C12/dt = fwd[0] * (rho*NA)^2 * Y_He4^3 / 6 */
KOKKOS_FUNCTION static double rate_triple_alpha(double T9)
{
    /* Be8 ground state (0+) decay width Gamma_Be8 = 6.8 eV (to 2 He4).
       Decay rate lambda_Be8 = Gamma / hbar in s^-1. */
    static constexpr double Gamma_Be8_eV = 6.8;
    static constexpr double eV_to_erg = 1.602176634e-12;
    static constexpr double lambda_Be8 = Gamma_Be8_eV * eV_to_erg / hbar_cgs; /* ~1.033e16 s^-1 */

    double T9a = T9 / (1.0 + 0.0396*T9);

    /* Step 1: N_A<sigma v> for He4+He4 -> Be8 (CF88 forward rate, cm^3/mol/s).
       Note: this is the FORWARD reaction rate, not the Saha factor. */
    double r2a_fwd = 7.40e+05 * pow(T9, -1.5) * exp(-1.0663/T9)
                   + 4.164e+09 * pow(T9, -2.0/3.0) * exp(-13.49/pow(T9, 1.0/3.0))
                     / pow(1.0 + 0.031*pow(T9, 1.0/3.0), 2) * exp(-0.219*T9*T9);

    /* Step 2: N_A<sigma v> for Be8+He4 -> C12 (CF88, cm^3/mol/s). */
    double raag = 130.0 * pow(T9, -1.5) * exp(-3.3364/T9)
                + 2.510e+07 * pow(T9a, -1.5) * exp(-23.570/pow(T9a, 1.0/3.0))
                  / pow(1.0 + 0.018*pow(T9a, 1.0/3.0), 2);

    /* Effective 3-body rate coefficient, units [cm^6/s], such that the caller's formula:
         dY_C12/dt = fwd[0] * (rho*NA)^2 * Y_He4^3 / 6
       gives the correct physical rate.

       Derivation (quasi-equilibrium Be8 picture):
         n_Be8^eq = n_He4^2 * <sigma_v>_aa / (2 * lambda_Be8)   [from P=D]
         dn_C12/dt = n_He4 * n_Be8^eq * <sigma_v>_Be8+He4
                   = n_He4^3 / (2*lambda_Be8) * <sigma_v>_aa * <sigma_v>_Be8+He4

       Converting to dY_C12/dt = dn_C12/dt / (rho*NA):
         dY_C12/dt = rho^2 * Y^3 * r2a_fwd * raag / (2*lambda_Be8)

       The caller divides by 6 (identical-particle factor for 3-body → caller has /6):
         fwd[0] * (rho*NA)^2 * Y^3 / 6 = rho^2 * Y^3 * r2a_fwd * raag / (2*lambda_Be8)
         fwd[0] = 3 * r2a_fwd * raag / (NA^2 * lambda_Be8)

       Numerical check at T9=1: r2a_fwd~2.6e5, raag~4.6, lambda_Be8~1.03e16 → fwd[0]~9.7e-58 cm^6/s */
    return 3.0 * r2a_fwd * raag / (avo * avo * lambda_Be8);
}

/* Standard (a,g) rate in REACLIB/CF88 parametric form:
   rate = exp(a0 + a1/T9 + a2/T9^(1/3) + a3*T9^(1/3) + a4*T9 + a5*T9^(5/3) + a6*ln(T9))
   Multiple components are summed for multi-resonance rates. */
struct rate_coeff { double a[7]; };

KOKKOS_FUNCTION static inline double eval_rate(const struct rate_coeff *r, int ncomp, double T9)
{
    double total = 0;
    double T9i = 1.0 / T9;
    double T913  = cbrt(T9);
    double T913i = 1.0 / T913;
    double T953  = T9 * T913 * T913;
    double lnT9  = log(T9);

    for (int c = 0; c < ncomp; c++) {
        double ex = r[c].a[0] + r[c].a[1]*T9i + r[c].a[2]*T913i + r[c].a[3]*T913
                  + r[c].a[4]*T9 + r[c].a[5]*T953 + r[c].a[6]*lnT9;
        if (ex > 500.0) { total += exp(500.0); continue; }
        if (ex < -500.0) continue;
        total += exp(ex);
    }
    return total;
}

/* CF88/NACRE forward rates for the 11 (a,g) reactions.
   Each reaction may have 1-3 REACLIB components (resonant + non-resonant). */

/* C12(a,g)O16: CF88 eq 18, two components */
static const struct rate_coeff rc12ag[] = {
    {{ 2.54985e+02, -1.84000e+00,  1.03411e+02, -4.20567e+02,  6.40874e+01, -1.24624e+01,  1.37803e+02}},
    {{ 6.96526e+01, -1.39254e+00,  5.89128e+01, -1.48273e+02,  9.08324e+00, -5.41041e-01,  7.03554e+01}},
};

/* O16(a,g)Ne20: CF88 eq 19 */
static const struct rate_coeff ro16ag[] = {
    {{ 2.86431e+01, -6.52460e+01,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 4.86604e+01, -5.48875e+01, -3.97262e+01, -2.10799e-01,  4.42879e-01, -7.97753e-02,  8.33333e-01}},
    {{ 3.42658e+01, -6.76518e+01,  0.00000e+00, -3.65925e+00,  7.14224e-01, -1.07508e-03,  0.00000e+00}},
};

/* Ne20(a,g)Mg24: NACRE (Angulo+ 1999) */
static const struct rate_coeff rne20ag[] = {
    {{ 2.68017e+01, -1.17334e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{-1.38869e+01, -1.10620e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 4.93244e+01, -1.08114e+02, -4.62525e+01,  5.58901e+00,  7.61843e+00, -3.68300e+00,  8.33333e-01}},
    {{ 1.60203e+01, -1.20895e+02,  0.00000e+00,  1.69229e+01, -2.57325e+00,  2.08997e-01,  0.00000e+00}},
};

/* Mg24(a,g)Si28: CF88 */
static const struct rate_coeff rmg24ag[] = {
    {{ 3.89908e+01, -1.66186e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{-1.23895e+01, -1.59316e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.52250e+01, -1.43656e+02, -6.33980e+01, -1.48750e+00,  2.61250e+01, -7.25750e+00,  8.33333e-01}},
};

/* Si28(a,g)S32: CF88 */
static const struct rate_coeff rsi28ag[] = {
    {{ 4.42778e+01, -1.79782e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{-1.80237e+01, -1.68431e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.54965e+01, -1.56772e+02, -6.64270e+01,  7.05180e-01,  3.08400e+01, -8.42250e+00,  8.33333e-01}},
};

/* S32(a,g)Ar36: CF88 */
static const struct rate_coeff rs32ag[] = {
    {{ 2.12300e+01, -1.87399e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.57920e+01, -1.69466e+02, -6.94270e+01,  1.68950e+00,  3.44200e+01, -9.31500e+00,  8.33333e-01}},
};

/* Ar36(a,g)Ca40: CF88 */
static const struct rate_coeff rar36ag[] = {
    {{ 5.28990e+01, -1.86899e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.62690e+01, -1.82485e+02, -7.24270e+01,  2.69870e+00,  3.77260e+01, -1.01475e+01,  8.33333e-01}},
};

/* Ca40(a,g)Ti44: CF88 */
static const struct rate_coeff rca40ag[] = {
    {{ 4.58750e+01, -1.97302e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.65130e+01, -1.95538e+02, -7.54270e+01,  3.62700e+00,  4.09700e+01, -1.09475e+01,  8.33333e-01}},
};

/* Ti44(a,g)Cr48: CF88 */
static const struct rate_coeff rti44ag[] = {
    {{ 5.02560e+01, -2.11138e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.62610e+01, -2.08637e+02, -7.84270e+01,  4.68900e+00,  4.39800e+01, -1.17175e+01,  8.33333e-01}},
};

/* Cr48(a,g)Fe52: CF88 */
static const struct rate_coeff rcr48ag[] = {
    {{ 5.25460e+01, -2.24370e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.68670e+01, -2.21666e+02, -8.14270e+01,  5.55700e+00,  4.72800e+01, -1.25025e+01,  8.33333e-01}},
};

/* Fe52(a,g)Ni56: CF88 */
static const struct rate_coeff rfe52ag[] = {
    {{ 5.46850e+01, -2.37410e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.72850e+01, -2.34743e+02, -8.44270e+01,  6.48100e+00,  5.04400e+01, -1.32625e+01,  8.33333e-01}},
};


/* =========================================================================
   Compute all forward and reverse rates at temperature T9
   ========================================================================= */
KOKKOS_FUNCTION static void compute_rates(double T9, double fwd[NR], double rev[NR])
{
    if (T9 < 0.01) { memset(fwd, 0, NR*sizeof(double)); memset(rev, 0, NR*sizeof(double)); return; }

    /* forward rates */
    fwd[0]  = rate_triple_alpha(T9);
    fwd[1]  = eval_rate(rc12ag,  2, T9);
    fwd[2]  = eval_rate(ro16ag,  3, T9);
    fwd[3]  = eval_rate(rne20ag, 4, T9);
    fwd[4]  = eval_rate(rmg24ag, 3, T9);
    fwd[5]  = eval_rate(rsi28ag, 3, T9);
    fwd[6]  = eval_rate(rs32ag,  2, T9);
    fwd[7]  = eval_rate(rar36ag, 2, T9);
    fwd[8]  = eval_rate(rca40ag, 2, T9);
    fwd[9]  = eval_rate(rti44ag, 2, T9);
    fwd[10] = eval_rate(rcr48ag, 2, T9);
    fwd[11] = eval_rate(rfe52ag, 2, T9);

    /* reverse rates from detailed balance.
       For a + b -> c + gamma:  r_rev = r_fwd * (G_a * G_b / G_c) * (mu_ab / m_u)^(3/2)
           * (kT / (2*pi*hbar^2))^(3/2) * exp(-Q/(kT))
       where G_i are nuclear partition functions (~1 for ground-state nuclei below ~5 GK).
       For triple-alpha (3-body -> 1-body), there are two factors of the thermal de Broglie term.

       We use the standard astrophysical convention:
         rev_rate = fwd_rate * C * T9^(3/2) * exp(-11.6045 * Q_MeV / T9)
       where C encodes the mass/spin factors and 11.6045 = 1e9 * kB / MeV. */

    double T932 = T9 * sqrt(T9);
    double T9i  = 1.0 / T9;

    /* Detailed balance coefficients C_rev for each reaction.
       These are from Timmes (1999) / Rauscher & Thielemann (2000):
       C = (2*J_c+1) / ((2*J_a+1)*(2*J_b+1)) * (A_a*A_b/A_c)^(3/2) * 9.8678e9^(-n+1)
       For alpha-chain: all nuclei have J=0 (even-even), so spin factor = 1.
       The mass factor (A_a*A_b/A_c)^(3/2) and the thermal normalization give: */

    /* reaction 0: 3a -> C12 (3-body, needs T9^3 and extra density factor) */
    /* This is special: the reverse rate is the photo-disintegration rate of C12. */
    {
        double Q_over_T9 = 11.6045 * Q_MeV[0] * T9i;
        /* C_rev for triple-alpha: includes the statistical and mass factors
           from the two-step Saha equation. Using Timmes' value. */
        rev[0] = 2.003e+20 * T932 * T932 * T932 * exp(-Q_over_T9);
        if (!isfinite(rev[0])) rev[0] = 0.0;
    }

    /* reactions 1-11: standard (a,g) 2-body */
    for (int r = 1; r < NR; r++) {
        double Q_over_T9 = 11.6045 * Q_MeV[r] * T9i;
        if (Q_over_T9 > 500.0) { rev[r] = 0.0; continue; }

        int A_tgt = A_sp[r];       /* target nucleus */
        int A_prod = A_sp[r + 1];  /* product nucleus */
        /* mass factor: (A_alpha * A_tgt / A_prod)^(3/2) */
        double mfac = pow((4.0 * (double)A_tgt) / (double)A_prod, 1.5);
        /* thermal de Broglie normalization (astrophysical convention) */
        double C_rev = mfac * 4.8359e+09; /* standard nuclear statistical weight */
        rev[r] = fwd[r] * C_rev * T932 * exp(-Q_over_T9);
        if (!isfinite(rev[r])) rev[r] = 0.0;
    }
}


/* =========================================================================
   Right-hand side: dY/dt for the 13 species.
   Y_i = X_i / A_i is the molar abundance (moles per gram).
   Returns dY/dt and energy generation rate edot [erg/g/s].
   ========================================================================= */
KOKKOS_FUNCTION static void nuclear_rhs(double rho, double T9, const double Y[NS],
                        double dYdt[NS], double *edot_out)
{
    double fwd[NR], rev[NR];
    compute_rates(T9, fwd, rev);
    memset(dYdt, 0, NS * sizeof(double));

    double rhoNA = rho * avo;

    /* reaction 0: triple-alpha: 3 He4 -> C12
       rate_fwd has units [cm^6/s/mol^2] for the combined 3a rate
       net rate = (1/6) * rho^2 * NA^2 * Y_a^3 * fwd - rev * Y_C12 */
    double Ya3 = Y[0] * Y[0] * Y[0];
    double r0f = fwd[0] * rhoNA * rhoNA * Ya3 / 6.0;
    double r0r = rev[0] * Y[1]; /* C12 photo-disintegration */
    double net0 = r0f - r0r;
    dYdt[0] -= 3.0 * net0;
    dYdt[1] += net0;

    /* reactions 1-11: A_i(a,g)A_{i+1}
       net rate = rho * NA * Y_a * Y_i * fwd - rev * Y_{i+1} */
    for (int r = 1; r < NR; r++) {
        double rf = fwd[r] * rhoNA * Y[0] * Y[r];
        double rr = rev[r] * Y[r + 1];
        double net = rf - rr;
        dYdt[0]     -= net;
        dYdt[r]     -= net;
        dYdt[r + 1] += net;
    }

    /* energy generation: edot = N_A * sum(dY_i/dt * Q_reaction_involving_i)
       equivalently: sum over reactions of (net_rate * Q_reaction) */
    double edot = net0 * Q_MeV[0];
    for (int r = 1; r < NR; r++) {
        double rf = fwd[r] * rhoNA * Y[0] * Y[r];
        double rr = rev[r] * Y[r + 1];
        edot += (rf - rr) * Q_MeV[r];
    }
    *edot_out = edot * MeV_to_erg * avo;
}


/* =========================================================================
   Analytic Jacobian: J[i][j] = d(dY_i/dt) / dY_j
   The alpha-chain Jacobian is sparse: each species couples to its neighbors
   and to He4. The 13x13 matrix has at most ~40 non-zero entries.
   ========================================================================= */
KOKKOS_FUNCTION static void nuclear_jacobian(double rho, double T9, const double Y[NS], double J[NS][NS])
{
    double fwd[NR], rev[NR];
    compute_rates(T9, fwd, rev);
    memset(J, 0, NS * NS * sizeof(double));

    double rhoNA = rho * avo;

    /* reaction 0: triple-alpha */
    double Ya2 = Y[0] * Y[0];
    double df0_dYa = fwd[0] * rhoNA * rhoNA * Ya2 / 2.0; /* d(r0f)/dYa = rhoNA^2 * Ya^2 / 2 */
    double df0_dYc = -rev[0];                              /* d(r0net)/dYc12 */

    J[0][0] += -3.0 * df0_dYa;   /* dYa_dot / dYa */
    J[0][1] += -3.0 * df0_dYc;   /* dYa_dot / dYc12 */
    J[1][0] +=  df0_dYa;         /* dYc12_dot / dYa */
    J[1][1] +=  df0_dYc;         /* dYc12_dot / dYc12 */

    /* reactions 1-11: A_i(a,g)A_{i+1} */
    for (int r = 1; r < NR; r++) {
        int i = r;       /* target species index */
        int ip = r + 1;  /* product species index */

        double dfr_dYa = fwd[r] * rhoNA * Y[i];    /* d(rf)/dYa */
        double dfr_dYi = fwd[r] * rhoNA * Y[0];    /* d(rf)/dYi */
        double drr_dYp = -rev[r];                    /* d(-rr)/dYp = -rev */

        /* dYa_dot/dYa, dYa_dot/dYi, dYa_dot/dYp */
        J[0][0] += -dfr_dYa;
        J[0][i] += -dfr_dYi;
        J[0][ip] += -drr_dYp;

        /* dYi_dot/dYa, dYi_dot/dYi, dYi_dot/dYp */
        J[i][0] += -dfr_dYa;
        J[i][i] += -dfr_dYi;
        J[i][ip] += -drr_dYp;

        /* dYp_dot/dYa, dYp_dot/dYi, dYp_dot/dYp */
        J[ip][0] += dfr_dYa;
        J[ip][i] += dfr_dYi;
        J[ip][ip] += drr_dYp;
    }
}


/* =========================================================================
   13x13 dense linear solver (Gaussian elimination with partial pivoting).
   Solves A * x = b in-place: b is overwritten with x, A is destroyed.
   ========================================================================= */
KOKKOS_FUNCTION static int solve_13x13(double A[NS][NS], double b[NS])
{
    int piv[NS];
    for (int i = 0; i < NS; i++) piv[i] = i;

    /* forward elimination */
    for (int k = 0; k < NS; k++) {
        /* partial pivoting */
        double maxval = fabs(A[piv[k]][k]);
        int maxrow = k;
        for (int i = k + 1; i < NS; i++) {
            double v = fabs(A[piv[i]][k]);
            if (v > maxval) { maxval = v; maxrow = i; }
        }
        if (maxval < 1.0e-100) return -1; /* singular */
        if (maxrow != k) { int tmp = piv[k]; piv[k] = piv[maxrow]; piv[maxrow] = tmp; }

        double pivot_inv = 1.0 / A[piv[k]][k];
        for (int i = k + 1; i < NS; i++) {
            double factor = A[piv[i]][k] * pivot_inv;
            A[piv[i]][k] = factor;
            for (int j = k + 1; j < NS; j++) {
                A[piv[i]][j] -= factor * A[piv[k]][j];
            }
            b[piv[i]] -= factor * b[piv[k]];
        }
    }

    /* back substitution */
    double x[NS];
    for (int i = NS - 1; i >= 0; i--) {
        x[i] = b[piv[i]];
        for (int j = i + 1; j < NS; j++) {
            x[i] -= A[piv[i]][j] * x[j];
        }
        x[i] /= A[piv[i]][i];
    }
    memcpy(b, x, NS * sizeof(double));
    return 0;
}


/* =========================================================================
   Backward-Euler solver with Newton iteration.
   Solves: Y^{n+1} = Y^n + dt * f(Y^{n+1})
   Newton: (I - dt*J) * dY = -(Y^{n+1}_guess - Y^n - dt*f(Y^{n+1}_guess))
   ========================================================================= */
KOKKOS_FUNCTION static int backward_euler_step(double rho, double T9, double dt,
                               const double Y_old[NS], double Y_new[NS])
{
    const int max_newton = 10;
    const double tol = 1.0e-8;

    /* initial guess: Y_old (not forward-Euler, which diverges for stiff problems) */
    double dYdt[NS];
    double edot_dummy;
    for (int k = 0; k < NS; k++) { Y_new[k] = Y_old[k]; }

    /* Newton iterations */
    for (int iter = 0; iter < max_newton; iter++) {
        /* residual: R = Y_new - Y_old - dt * f(Y_new) */
        nuclear_rhs(rho, T9, Y_new, dYdt, &edot_dummy);
        double residual[NS];
        for (int k = 0; k < NS; k++) {
            residual[k] = Y_new[k] - Y_old[k] - dt * dYdt[k];
        }

        /* check convergence: mixed absolute/relative tolerance.
           For trace species (Y ~ 1e-27), absolute residual < atol is sufficient.
           For abundant species, relative residual < tol is required. */
        double Y_max = 0;
        for (int k = 0; k < NS; k++) { if (fabs(Y_old[k]) > Y_max) Y_max = fabs(Y_old[k]); }
        double atol = tol * Y_max;  /* absolute tolerance scaled to largest abundance */
        double rnorm = 0;
        for (int k = 0; k < NS; k++) {
            double scale = fabs(Y_new[k]) + atol;
            double rel = fabs(residual[k]) / scale;
            if (rel > rnorm) rnorm = rel;
        }
        if (rnorm < tol) return 0; /* converged */

        /* Jacobian: M = I - dt * J */
        double Jac[NS][NS];
        nuclear_jacobian(rho, T9, Y_new, Jac);
        double M[NS][NS];
        for (int i = 0; i < NS; i++) {
            for (int j = 0; j < NS; j++) {
                M[i][j] = -dt * Jac[i][j];
            }
            M[i][i] += 1.0;
        }

        /* solve M * dY = -residual */
        double rhs[NS];
        for (int k = 0; k < NS; k++) rhs[k] = -residual[k];
        if (solve_13x13(M, rhs) != 0) return -1; /* singular */

        /* update — guard against NaN from ill-conditioned solve */
        bool bad = false;
        for (int k = 0; k < NS; k++) { if (!isfinite(rhs[k])) { bad = true; break; } }
        if (bad) return -1;
        for (int k = 0; k < NS; k++) {
            Y_new[k] += rhs[k];
            if (Y_new[k] < 0) Y_new[k] = 0;
        }

    }

    return 1; /* didn't converge — caller should subcycle */
}


/* =========================================================================
   Main solver entry point.
   ========================================================================= */
KOKKOS_FUNCTION int nuclear_aprox13_solve(const struct nuclear_input *in, struct nuclear_output *out)
{
    double T9 = in->T / 1.0e9;
    double rho_cgs = in->rho * UNIT_DENSITY_IN_CGS;
    double dt_cgs  = in->dt * UNIT_TIME_IN_CGS;

    if (T9 < 0.01 || dt_cgs <= 0) {
        memcpy(out->X, in->X, sizeof(out->X));
        out->Ye = in->Ye;
        nuclear_compute_ye_abar(out->X, &out->Ye, &out->Abar);
        out->de = 0; out->edot = 0; out->burning_timescale = 1.0e30;
        return 0;
    }

    /* NSE shortcut at very high T */
    if (nuclear_check_nse(T9)) {
        double X_old[NS];
        memcpy(X_old, in->X, sizeof(X_old));
        nuclear_nse_composition(rho_cgs, T9, in->Ye, out->X);
        nuclear_compute_ye_abar(out->X, &out->Ye, &out->Abar);
        double de_cgs = 0;
        for (int k = 0; k < NS; k++) {
            de_cgs += (out->X[k] - X_old[k]) * nuclear_aprox13_BE_per_A[k] * MeV_to_erg * avo;
        }
        out->de = de_cgs / UNIT_SPECEGY_IN_CGS;
        out->edot = (dt_cgs > 0) ? de_cgs / dt_cgs / (UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS) : 0;
        out->burning_timescale = 1.0e-10 / UNIT_TIME_IN_CGS;
        return 0;
    }

    /* convert mass fractions to molar abundances: Y = X / A */
    double Y_old[NS], Y_new[NS];
    for (int k = 0; k < NS; k++) {
        Y_old[k] = (in->X[k] > 0) ? in->X[k] / (double)A_sp[k] : 0.0;
    }

    /* estimate shortest burning timescale for subcycling */
    double dYdt[NS];
    double edot_cgs;
    nuclear_rhs(rho_cgs, T9, Y_old, dYdt, &edot_cgs);

    double tau_min = 1.0e30;
    for (int k = 0; k < NS; k++) {
        if (fabs(dYdt[k]) > 1.0e-50 && Y_old[k] > 1.0e-30) {
            double tau = fabs(Y_old[k] / dYdt[k]);
            if (tau < tau_min) tau_min = tau;
        }
    }

    /* subcycling strategy: backward-Euler handles stiffness, but the Newton method
       needs dt_sub to be within a reasonable factor of tau_min (up to ~1e6x).
       For extreme stiffness (dt/tau > 1e6), first evolve to quasi-equilibrium
       with a few small steps at dt ~ tau_min * Courant_factor, then take one
       large backward-Euler step for the remainder. */
    double total_energy = 0;
    memcpy(Y_new, Y_old, sizeof(Y_old));
    double dt_remaining = dt_cgs;

    if (tau_min > 0 && dt_cgs / tau_min > 1.0e6) {
        /* extreme stiffness: evolve to quasi-equilibrium first */
        int nfast = 200;
        double dt_fast = DMIN(100.0 * tau_min, dt_cgs / nfast);
        for (int s = 0; s < nfast && dt_remaining > 0; s++) {
            double dt_step = DMIN(dt_fast, dt_remaining);
            double Y_step[NS];
            int ierr = backward_euler_step(rho_cgs, T9, dt_step, Y_new, Y_step);
            if (ierr != 0) {
                /* Newton failed — retry with 10x smaller substeps */
                int ns2 = 10;
                double dt2 = dt_step / ns2;
                for (int s2 = 0; s2 < ns2; s2++) {
                    double Y2[NS];
                    if (backward_euler_step(rho_cgs, T9, dt2, Y_new, Y2) == 0) {
                        double Y_mid[NS]; for (int k = 0; k < NS; k++) Y_mid[k] = 0.5*(Y_new[k]+Y2[k]);
                        nuclear_rhs(rho_cgs, T9, Y_mid, dYdt, &edot_cgs);
                        if (isfinite(edot_cgs)) total_energy += edot_cgs * dt2;
                        memcpy(Y_new, Y2, sizeof(Y_new));
                    }
                }
            } else {
                double Y_mid[NS]; for (int k = 0; k < NS; k++) Y_mid[k] = 0.5*(Y_new[k]+Y_step[k]);
                nuclear_rhs(rho_cgs, T9, Y_mid, dYdt, &edot_cgs);
                if (isfinite(edot_cgs)) total_energy += edot_cgs * dt_step;
                memcpy(Y_new, Y_step, sizeof(Y_new));
            }
            dt_remaining -= dt_step;
            /* re-evaluate tau to see if we've reached quasi-equilibrium */
            nuclear_rhs(rho_cgs, T9, Y_new, dYdt, &edot_cgs);
            double tau_new = 1e30;
            for (int k = 0; k < NS; k++) {
                if (fabs(dYdt[k]) > 1e-50 && Y_new[k] > 1e-30) {
                    double t = fabs(Y_new[k] / dYdt[k]);
                    if (t < tau_new) tau_new = t;
                }
            }
            if (tau_new > dt_remaining * 0.01) break; /* close enough to equilibrium */
        }
        /* for the remaining time, composition is quasi-static — just accumulate energy */
        if (dt_remaining > 0) {
            nuclear_rhs(rho_cgs, T9, Y_new, dYdt, &edot_cgs);
            total_energy += edot_cgs * dt_remaining;
        }
    } else {
        /* moderate stiffness: standard subcycled backward-Euler */
        int nsub = 1;
        if (tau_min > 0 && tau_min < dt_cgs * 0.1) {
            nsub = DMAX(1, (int)ceil(dt_cgs / (10.0 * tau_min)));
            nsub = DMIN(nsub, 1000);
        }
        double dt_sub = dt_cgs / nsub;

        for (int step = 0; step < nsub; step++) {
            double Y_step[NS];
            int ierr = backward_euler_step(rho_cgs, T9, dt_sub, Y_new, Y_step);
            if (ierr != 0) {
                /* retry backward-Euler with 10x smaller substeps */
                int nsub2 = 10;
                double dt2 = dt_sub / nsub2;
                for (int s2 = 0; s2 < nsub2; s2++) {
                    double Y2[NS];
                    if (backward_euler_step(rho_cgs, T9, dt2, Y_new, Y2) == 0) {
                        double Y_mid[NS]; for (int k = 0; k < NS; k++) Y_mid[k] = 0.5*(Y_new[k]+Y2[k]);
                        nuclear_rhs(rho_cgs, T9, Y_mid, dYdt, &edot_cgs);
                        if (isfinite(edot_cgs)) total_energy += edot_cgs * dt2;
                        memcpy(Y_new, Y2, sizeof(Y_new));
                    }
                    /* if sub-sub-step also fails, skip it (hold composition fixed) */
                }
                continue;
            }
            double Y_mid[NS]; for (int k = 0; k < NS; k++) Y_mid[k] = 0.5*(Y_new[k]+Y_step[k]);
            nuclear_rhs(rho_cgs, T9, Y_mid, dYdt, &edot_cgs);
            if (isfinite(edot_cgs)) total_energy += edot_cgs * dt_sub;
            memcpy(Y_new, Y_step, sizeof(Y_new));
        }
    }

    /* sanitize Y_new: replace any NaN/Inf with the initial value */
    for (int k = 0; k < NS; k++) {
        if (!isfinite(Y_new[k]) || Y_new[k] < 0) Y_new[k] = Y_old[k];
    }

    /* convert back to mass fractions and clamp */
    double sum = 0;
    for (int k = 0; k < NS; k++) {
        out->X[k] = Y_new[k] * (double)A_sp[k];
        if (out->X[k] < 0) out->X[k] = 0;
        if (out->X[k] > 1) out->X[k] = 1;
        sum += out->X[k];
    }
    if (sum > 0) { for (int k = 0; k < NS; k++) out->X[k] /= sum; }

    nuclear_compute_ye_abar(out->X, &out->Ye, &out->Abar);

    /* energy release from binding energy difference (more robust than accumulated edot*dt) */
    double de_binding = 0;
    for (int k = 0; k < NS; k++) {
        de_binding += (out->X[k] - in->X[k]) * nuclear_aprox13_BE_per_A[k]
                    * 1.602176634e-6 * 6.02214076e23; /* MeV/nucleon * MeV_to_erg * N_A */
    }
    out->de   = de_binding / UNIT_SPECEGY_IN_CGS;
    /* edot = de/dt, but only meaningful if de is actually significant.
       if the composition barely changed (near equilibrium), edot should be ~0 */
    out->edot = (dt_cgs > 0 && fabs(de_binding) > 1.0e-10 * fabs(in->rho * UNIT_DENSITY_IN_CGS))
              ? de_binding / dt_cgs / (UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS) : 0;

    /* recompute burning timescale from the FINAL (post-burn) state, not the initial state.
       the initial tau_min can be absurdly short (1e-67 s) for conditions where burning is
       instantaneous, but after equilibrating the composition changes slowly. */
    nuclear_rhs(rho_cgs, T9, Y_new, dYdt, &edot_cgs);
    double tau_final = 1.0e30;
    for (int k = 0; k < NS; k++) {
        if (fabs(dYdt[k]) > 1.0e-50 && Y_new[k] > 1.0e-30) {
            double t = fabs(Y_new[k] / dYdt[k]);
            if (t < tau_final) tau_final = t;
        }
    }
    out->burning_timescale = tau_final / UNIT_TIME_IN_CGS;

#ifdef NUCLEAR_NETWORK_NEUTRINOS
    /* estimate neutrino luminosity from thermal processes at current conditions */
    double nu_lum_cgs[3], nu_emean_mev[3];
    nuclear_neutrino_emission(rho_cgs, T9, out->Ye, nu_lum_cgs, nu_emean_mev);
    for (int f = 0; f < 3; f++) {
        out->nu_lum[f]   = nu_lum_cgs[f] / (UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS);
        out->nu_emean[f] = nu_emean_mev[f];
    }
#endif

    return 0;
}


/* =========================================================================
   Nuclear Statistical Equilibrium (NSE) for high-temperature regime.
   At T > ~5-7 GK, reactions reach equilibrium faster than any hydro timescale.
   ========================================================================= */

static const double *BE_per_A = nuclear_aprox13_BE_per_A;
static const int *A_species = A_sp;
static const int *Z_species = Z_sp;

KOKKOS_FUNCTION void nuclear_nse_composition(double rho_cgs, double T9, double Ye,
                             double X_out[NUM_NUCLEAR_SPECIES])
{
    /* initialize to zero */
    memset(X_out, 0, NUM_NUCLEAR_SPECIES * sizeof(double));

    double kT_MeV = kerg * T9 * 1.0e9 / MeV_to_erg;

    /* for Ye ~ 0.5, the most bound nucleus in the alpha-chain is Ni56 (index 12).
       The Saha equation for A alpha-particles -> nucleus(A,Z) gives:
       Y_nucleus / Y_alpha^(A/4) ~ (rho * avo)^(A/4-1) * f(T) * exp(Q/(kT))
       where Q = BE(nucleus) - (A/4)*BE(He4) is the Q-value. */

    /* find the most-bound species */
    int i_peak = NUCLEAR_NI56;
    double Q_peak = BE_per_A[i_peak] * A_species[i_peak] - (A_species[i_peak] / 4.0) * BE_per_A[NUCLEAR_HE4] * 4.0;

    /* effective Saha parameter: large positive = favors heavy nucleus */
    double saha_param = Q_peak / kT_MeV - (A_species[i_peak] / 4.0 - 1.0) * log(rho_cgs * avo / (T9 * T9 * sqrt(T9) * 1.0e27));

    /* smooth transition between He4-dominated and Ni56-dominated regimes */
    double f_heavy;
    if (saha_param > 30.0) {
        f_heavy = 1.0;  /* strongly favors heavy nucleus */
    } else if (saha_param < -30.0) {
        f_heavy = 0.0;  /* strongly favors alpha particles */
    } else {
        f_heavy = 1.0 / (1.0 + exp(-saha_param));  /* sigmoid transition */
    }

    /* distribute: fraction f_heavy goes to the peak species, rest to He4 */
    X_out[NUCLEAR_HE4] = 1.0 - f_heavy;
    X_out[i_peak]      = f_heavy;

    /* for intermediate temperatures, distribute some mass to Fe52 and Cr48
       as they are close in binding energy and have non-negligible NSE abundance */
    if (f_heavy > 0.01 && f_heavy < 0.99) {
        double Q_fe52 = BE_per_A[NUCLEAR_FE52] * A_species[NUCLEAR_FE52]
                      - (A_species[NUCLEAR_FE52] / 4.0) * BE_per_A[NUCLEAR_HE4] * 4.0;
        double saha_fe52 = Q_fe52 / kT_MeV - (A_species[NUCLEAR_FE52] / 4.0 - 1.0)
                         * log(rho_cgs * avo / (T9 * T9 * sqrt(T9) * 1.0e27));
        double f_fe52 = 1.0 / (1.0 + exp(-saha_fe52));

        /* blend: at intermediate T, redistribute some of the heavy fraction */
        double blend = 4.0 * f_heavy * (1.0 - f_heavy); /* peaks at f_heavy=0.5 */
        double x_fe52 = blend * 0.3 * f_fe52;
        double x_ni56 = f_heavy - x_fe52;
        if (x_ni56 < 0) { x_ni56 = 0; x_fe52 = f_heavy; }

        X_out[NUCLEAR_NI56] = x_ni56;
        X_out[NUCLEAR_FE52] = x_fe52;
        X_out[NUCLEAR_HE4]  = 1.0 - x_ni56 - x_fe52;
    }

    /* ensure normalization */
    double sum = 0;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        if (X_out[k] < 0) X_out[k] = 0;
        sum += X_out[k];
    }
    if (sum > 0) {
        double inv = 1.0 / sum;
        for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) X_out[k] *= inv;
    }
}


/* Check if we should use NSE for given conditions.
   Returns 1 if NSE applies, 0 otherwise. */
KOKKOS_FUNCTION int nuclear_check_nse(double T9)
{
#ifdef NUCLEAR_NETWORK_NSE_TABLE
    return (T9 * 1.0e9 > All.NuclearNSE_T_threshold);
#else
    return 0;
#endif
}


#endif /* NUCLEAR_NETWORK_SOLVER == 0 */
#endif /* NUCLEAR_NETWORK */
