/* Built-in aprox13 alpha-chain nuclear reaction network — header-only.
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

   Header-only: all functions are KOKKOS_INLINE_FUNCTION; static helpers have
   internal linkage so each including TU gets its own copy without ODR conflicts.
   This matches cooling/cooling_functions.h and the codebase's standard
   header-only device-callable pattern (no nvcc cross-TU device-call linking).
*/
#ifndef NUCLEAR_PHYSICS_FUNCTIONS_H
#define NUCLEAR_PHYSICS_FUNCTIONS_H

#include <cmath>
#include <cstring>
#include "../declarations/allvars.h"
#include "nuclear.h"

#ifdef NUCLEAR_NETWORK

/* =========================================================================
   Ye / Abar computation from mass fractions (pure function, used by all solvers).
   ========================================================================= */
KOKKOS_INLINE_FUNCTION void nuclear_compute_ye_abar(const double X[NUM_NUCLEAR_SPECIES],
                             double *Ye_out, double *Abar_out)
{
    double sum_X_over_A = 0, sum_Z_X_over_A = 0;
    for (int k = 0; k < NUM_NUCLEAR_SPECIES; k++) {
        double x = X[k] / (double) nuclear_aprox13_A(k);
        sum_X_over_A   += x;
        sum_Z_X_over_A += (double) nuclear_aprox13_Z(k) * x;
    }
    if (sum_X_over_A > 0) {
        *Abar_out = 1.0 / sum_X_over_A;
        *Ye_out   = sum_Z_X_over_A;
    } else {
        *Abar_out = 4.0; *Ye_out = 0.5;
    }
}

/* =========================================================================
   Physical constants (CGS) shared by every solver and by the neutrino block.
   These sit outside the alpha-chain namespace and outside the solver-0 gate
   because the neutrino routines below are available for all solver settings.
   Namespaced rather than global: this header is pulled in transitively by the
   RT chain, and names this short (avo, kerg) exist in other modules' constant
   sets — eos/helmholtz/helmholtz.h has its own, used via a using-directive.
   ========================================================================= */
namespace nuclear_constants {
inline constexpr double MeV_to_erg = 1.602176634e-6;
inline constexpr double avo        = 6.02214076e23;
inline constexpr double kerg       = 1.380649e-16;
} /* namespace nuclear_constants */


#ifdef NUCLEAR_NETWORK_NEUTRINOS
/* =========================================================================
   Neutrino transport physics for the three neutrino RT bands (electron,
   anti-electron, heavy flavor).

   The key neutrino-matter interaction processes are:
     - Charged-current absorption on free nucleons:
         nu_e + n -> p + e^-    (increases Ye)
         nu_ebar + p -> n + e^+  (decreases Ye)
     - Neutral-current scattering on nucleons and nuclei (all flavors)
     - Electron-positron pair annihilation (at high T)

   Cross-sections scale as sigma ~ (E_nu / m_e c^2)^2 * sigma_0
   where sigma_0 = G_F^2 * m_e^2 * c^4 / (pi * hbar^4 * c^4) ~ 1.76e-44 cm^2

   References:
     - Bruenn 1985, ApJS 58, 771 (neutrino opacity formulae)
     - Burrows, Reddy & Thompson 2006, NPA 777, 356 (neutrino-matter interactions)
     - Horowitz 2002, PRD 65, 043001 (neutrino-nucleon cross-sections)

   Thread-safe: all functions are pure (no global mutable state). The opacity
   and absorbed-fraction routines are called from rt_kappa/rt_absorb_frac_albedo
   in radiation/rt_functions.h, so they must be device-callable and available
   for every NUCLEAR_NETWORK_SOLVER setting.
   ========================================================================= */
namespace nuclear_neutrino_constants {
/* fundamental constants */
inline constexpr double sigma_0   = 1.76e-44;    /* weak cross-section scale [cm^2] */
inline constexpr double m_e_MeV   = 0.51099895;  /* electron mass [MeV] */
inline constexpr double m_n_MeV   = 939.56542;   /* neutron mass [MeV] */
inline constexpr double delta_np  = 1.29333;     /* neutron-proton mass difference [MeV] */

/* weak interaction coupling constants */
inline constexpr double cv = 1.0;    /* vector coupling (nu_e on neutrons) */
inline constexpr double ca = 1.26;   /* axial-vector coupling */
inline constexpr double cv2_ca2 = cv*cv + 3.0*ca*ca;  /* = 1 + 3*1.26^2 ~ 5.76 */

/* neutral-current coupling for protons and neutrons (sin^2(theta_W) ~ 0.231) */
inline constexpr double sin2tw = 0.2312;
inline constexpr double gv_n = -0.5;
inline constexpr double ga_n = -ca / 2.0;
inline constexpr double gv_p = 0.5 - 2.0 * sin2tw;
inline constexpr double ga_p = ca / 2.0;
} /* namespace nuclear_neutrino_constants */


/* ---------------------------------------------------------------------------
   Neutrino opacity [code units: Length^2/Mass].
   Called from rt_kappa() for neutrino frequency bins.

   The opacity is dominated by charged-current absorption on free nucleons
   for electron-flavor neutrinos, and by neutral-current scattering for
   all flavors. The cross-section depends on the mean neutrino energy.
   --------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION double nuclear_neutrino_opacity(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
{
    using namespace nuclear_constants;
    using namespace nuclear_neutrino_constants;
    double Ye = 0.5;
#ifdef EOS_CARRIES_YE
    Ye = cell[i].Ye;
#endif
    double Xn = 1.0 - Ye; /* neutron mass fraction (approximate for free nucleons) */
    double Xp = Ye;        /* proton mass fraction */

    /* mean neutrino energy [MeV] — from the nuclear burning step, or default */
    double E_nu = 10.0; /* default 10 MeV if not set */
    int flavor = 0;
    if (k_freq == RT_FREQ_BIN_NU_E) flavor = 0;
    else if (k_freq == RT_FREQ_BIN_NU_EBAR) flavor = 1;
    else if (k_freq == RT_FREQ_BIN_NU_X) flavor = 2;

    if (cell[i].NeutrinoMeanEnergy[flavor] > 0) {
        E_nu = cell[i].NeutrinoMeanEnergy[flavor];
    }

    double E_ratio_sq = (E_nu / m_e_MeV) * (E_nu / m_e_MeV);
    double kappa = 0; /* opacity in cm^2/g */

    if (k_freq == RT_FREQ_BIN_NU_E) {
        /* charged-current: nu_e + n -> p + e^- */
        /* sigma_cc = sigma_0 * (1 + 3*g_A^2)/4 * (E_nu + delta_np)^2 / m_e^2 */
        double E_eff = E_nu + delta_np;
        double sigma_cc = sigma_0 * cv2_ca2 / 4.0 * (E_eff / m_e_MeV) * (E_eff / m_e_MeV);
        kappa += Xn * avo * sigma_cc;

        /* neutral-current scattering on neutrons */
        double sigma_nc_n = sigma_0 / 4.0 * (gv_n*gv_n + 3.0*ga_n*ga_n) * E_ratio_sq;
        kappa += Xn * avo * sigma_nc_n;

    } else if (k_freq == RT_FREQ_BIN_NU_EBAR) {
        /* charged-current: nu_ebar + p -> n + e^+ */
        double E_eff = E_nu - delta_np;
        if (E_eff > 0) {
            double sigma_cc = sigma_0 * cv2_ca2 / 4.0 * (E_eff / m_e_MeV) * (E_eff / m_e_MeV);
            kappa += Xp * avo * sigma_cc;
        }

        /* neutral-current scattering on protons */
        double sigma_nc_p = sigma_0 / 4.0 * (gv_p*gv_p + 3.0*ga_p*ga_p) * E_ratio_sq;
        kappa += Xp * avo * sigma_nc_p;

    } else if (k_freq == RT_FREQ_BIN_NU_X) {
        /* heavy-flavor (mu, tau): only neutral-current */
        double sigma_nc_n = sigma_0 / 4.0 * (gv_n*gv_n + 3.0*ga_n*ga_n) * E_ratio_sq;
        double sigma_nc_p = sigma_0 / 4.0 * (gv_p*gv_p + 3.0*ga_p*ga_p) * E_ratio_sq;
        kappa += (Xn * sigma_nc_n + Xp * sigma_nc_p) * avo;
    }

    /* convert from cm^2/g to code units [Length^2/Mass] */
    return kappa / UNIT_SURFDEN_IN_CGS;
}


/* ---------------------------------------------------------------------------
   Neutrino absorption fraction (1 - albedo).
   Neutrino "scattering" (neutral-current) changes direction but not energy
   significantly at these energies. For the RT solver, we treat the
   charged-current processes as pure absorption and the NC as scattering.
   --------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION double nuclear_neutrino_absorb_frac(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* for electron-flavor neutrinos, absorption dominates (CC >> NC at nuclear densities).
       for heavy-flavor, only NC scattering occurs — effectively albedo=1 for transport
       but the deposited energy is small. return a conservative fraction. */
    if (k_freq == RT_FREQ_BIN_NU_E || k_freq == RT_FREQ_BIN_NU_EBAR) {
        return 0.9; /* dominated by CC absorption, small NC scattering correction */
    }
    return 0.2; /* heavy-flavor: mostly NC scattering, small energy deposition */
}


/* ---------------------------------------------------------------------------
   Ye feedback from neutrino absorption.
   When neutrinos are absorbed, the electron fraction changes:
     nu_e + n -> p + e^-     : delta_Ye = +1/A per absorption
     nu_ebar + p -> n + e^+  : delta_Ye = -1/A per absorption

   This is called from the RT chemistry update to modify Ye based on
   the absorbed neutrino energy density.
   --------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION void nuclear_neutrino_ye_feedback(int i, double dt_code,
                                   struct particle_data *pp, struct gas_cell_data *cell)
{
    using namespace nuclear_constants;
#ifdef EOS_CARRIES_YE
    double rho_cgs = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
    double dt_cgs = dt_code * UNIT_TIME_IN_CGS;

    /* absorbed energy in each neutrino band [code units] is stored in Rad_E_gamma
       after the RT step. The Ye change is proportional to the number of absorptions:
       dYe/dt = (L_abs_nue - L_abs_nuebar) / (rho * Abar * E_nu) * N_A */

    /* estimate from the neutrino luminosity and opacity:
       absorption rate per unit mass = kappa_abs * c * E_rad / volume */
    double Ye_new = cell[i].Ye;

    /* nu_e absorption: increases Ye */
    if (cell[i].NeutrinoLuminosity[0] > 0 && cell[i].NeutrinoMeanEnergy[0] > 0) {
        double E_nu_erg = cell[i].NeutrinoMeanEnergy[0] * MeV_to_erg;
        double kappa_abs = nuclear_neutrino_opacity(i, RT_FREQ_BIN_NU_E, pp, cell) * UNIT_SURFDEN_IN_CGS;
        double n_absorb_rate = kappa_abs * cell[i].NeutrinoLuminosity[0] * UNIT_SPECEGY_IN_CGS / E_nu_erg;
        Ye_new += n_absorb_rate * dt_cgs / (avo * rho_cgs);
    }

    /* nu_ebar absorption: decreases Ye */
    if (cell[i].NeutrinoLuminosity[1] > 0 && cell[i].NeutrinoMeanEnergy[1] > 0) {
        double E_nu_erg = cell[i].NeutrinoMeanEnergy[1] * MeV_to_erg;
        double kappa_abs = nuclear_neutrino_opacity(i, RT_FREQ_BIN_NU_EBAR, pp, cell) * UNIT_SURFDEN_IN_CGS;
        double n_absorb_rate = kappa_abs * cell[i].NeutrinoLuminosity[1] * UNIT_SPECEGY_IN_CGS / E_nu_erg;
        Ye_new -= n_absorb_rate * dt_cgs / (avo * rho_cgs);
    }

    /* clamp to physical range */
    if (Ye_new < 0.0) Ye_new = 0.0;
    if (Ye_new > 1.0) Ye_new = 1.0;
    cell[i].Ye = Ye_new;
#endif
}


/* ---------------------------------------------------------------------------
   Estimate neutrino luminosity and mean energy from nuclear burning conditions.
   Called by the network solver to populate the nu_lum[] and nu_emean[] fields
   in nuclear_output. This is a local, pure function of (rho, T, Ye, edot).

   Physics: at temperatures relevant for nuclear burning (T > 1 GK), the
   dominant neutrino emission processes are:
     - Thermal pair annihilation: e+ e- -> nu nu_bar (all flavors)
     - Plasmon decay: gamma* -> nu nu_bar
     - Photo-neutrinos: e- gamma -> e- nu nu_bar
   The total neutrino luminosity scales roughly as T^9 at high T (pair process)
   and is partitioned ~equally among the 3 flavors for pair/plasmon processes,
   with electron-flavor enhanced by charged-current processes.

   We use the Itoh et al. (1996) fitting formulae for pair and plasmon rates.
   --------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION void nuclear_neutrino_emission(double rho_cgs, double T9, double Ye,
                               double nu_lum[3], double nu_emean[3])
{
    using namespace nuclear_constants;
    nu_lum[0] = nu_lum[1] = nu_lum[2] = 0;
    nu_emean[0] = nu_emean[1] = nu_emean[2] = 0;

    if (T9 < 0.5) return; /* negligible below ~0.5 GK */

    double T_K = T9 * 1.0e9;
    double kT_MeV = kerg * T_K / MeV_to_erg;

    /* pair annihilation: Q_pair ~ 5e33 * (T/1e10)^9 erg/g/s (Itoh+ 1996 approximate) */
    double T10 = T9 / 10.0;
    double T10_9 = pow(T10, 9.0);
    double Q_pair = 5.0e33 * T10_9 / (rho_cgs + 1.0e-30); /* erg/g/s, per unit mass */
    /* at very low density this overestimates; suppress by density factor */
    Q_pair *= rho_cgs / (rho_cgs + 1.0e6); /* smooth suppression below ~1e6 g/cc */

    /* plasmon decay: Q_plasmon ~ 2e32 * (T/1e10)^9 * (rho/1e9)^(1/3) */
    double Q_plasmon = 2.0e32 * T10_9 * cbrt(rho_cgs / 1.0e9);
    Q_plasmon *= rho_cgs / (rho_cgs + 1.0e4);

    double Q_total = Q_pair + Q_plasmon; /* erg/g/s */

    /* flavor partition: pair and plasmon processes produce all flavors roughly equally,
       but electron-flavor gets ~40% extra from electron captures at high density */
    double f_e = 0.40;  /* electron neutrino fraction */
    double f_ebar = 0.30;
    double f_x = 0.30;  /* mu + tau combined (split equally) */

    nu_lum[0] = Q_total * f_e;      /* nu_e */
    nu_lum[1] = Q_total * f_ebar;   /* nu_ebar */
    nu_lum[2] = Q_total * f_x;      /* nu_x (mu+tau) */

    /* mean neutrino energies: <E_nu> ~ 3.15 * kT for thermal spectrum (Fermi-Dirac) */
    /* electron neutrinos are slightly more energetic due to CC processes */
    nu_emean[0] = 3.5 * kT_MeV;   /* ~12 MeV at T=3 GK */
    nu_emean[1] = 3.15 * kT_MeV;  /* ~11 MeV at T=3 GK */
    nu_emean[2] = 2.5 * kT_MeV;   /* ~8 MeV at T=3 GK (heavy flavor, less energetic) */
}

#endif /* NUCLEAR_NETWORK_NEUTRINOS */


#if !defined(NUCLEAR_NETWORK_SOLVER) || (NUCLEAR_NETWORK_SOLVER == 0)

/* =========================================================================
   Physical constants (CGS) used only by the built-in alpha-chain network
   ========================================================================= */
namespace nuclear_aprox13_internal {
using namespace nuclear_constants;  /* shared CGS constants (avo, kerg, MeV_to_erg) */
inline constexpr double amu_cgs    = 1.66053906660e-24;
inline constexpr double hbar_cgs   = 1.054571817e-27;
inline constexpr double clight_cgs = 2.99792458e10;
inline constexpr double pi_val     = 3.14159265358979323846;

inline constexpr int NS = 13;  /* number of species */
inline constexpr int NR = 12;  /* number of reactions */
} /* namespace nuclear_aprox13_internal */

/* Q-values [MeV] for the 12 reactions: Q = BE(product) - BE(reactants).
   Computed from AME2020 mass excess values. Accessor with function-local
   constexpr — see nuclear.h header comment on the #20091-D fix. */
KOKKOS_INLINE_FUNCTION double Q_MeV(int r) {
    constexpr double data[] = {
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
    return data[r];
}

/* Species A and Z (from nuclear.h) — accessor with function-local constexpr,
 * see nuclear.h header comment on the #20091-D fix. */
KOKKOS_INLINE_FUNCTION int A_sp(int k) {
    constexpr int data[] = {4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56};
    return data[k];
}
KOKKOS_INLINE_FUNCTION int Z_sp(int k) {
    constexpr int data[] = {2,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28};
    return data[k];
}


/* =========================================================================
   CF88 thermonuclear reaction rates.
   Each function returns the reaction rate in cm^3/s/mol (for 2-body)
   or cm^6/s/mol^2 (for 3-body, i.e. triple-alpha).
   Temperature T9 is in units of 10^9 K.
   ========================================================================= */

/* Triple-alpha rate: 3 He4 -> C12 (CF88 eqs 15-17).
   Returns the effective 3-body rate coefficient <sigma_v>_3a in cm^6/s.
   The caller (nuclear_rhs) computes: dY_C12/dt = fwd[0] * (rho*NA)^2 * Y_He4^3 / 6 */
KOKKOS_INLINE_FUNCTION static double rate_triple_alpha(double T9)
{
    using namespace nuclear_aprox13_internal;
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
       gives the correct physical rate. */
    return 3.0 * r2a_fwd * raag / (avo * avo * lambda_Be8);
}

/* Standard (a,g) rate in REACLIB/CF88 parametric form:
   rate = exp(a0 + a1/T9 + a2/T9^(1/3) + a3*T9^(1/3) + a4*T9 + a5*T9^(5/3) + a6*ln(T9))
   Multiple components are summed for multi-resonance rates. */
struct rate_coeff { double a[7]; };

KOKKOS_INLINE_FUNCTION static double eval_rate(const struct rate_coeff *r, int ncomp, double T9)
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
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rc12ag[] = {
    {{ 2.54985e+02, -1.84000e+00,  1.03411e+02, -4.20567e+02,  6.40874e+01, -1.24624e+01,  1.37803e+02}},
    {{ 6.96526e+01, -1.39254e+00,  5.89128e+01, -1.48273e+02,  9.08324e+00, -5.41041e-01,  7.03554e+01}},
};

/* O16(a,g)Ne20: CF88 eq 19 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff ro16ag[] = {
    {{ 2.86431e+01, -6.52460e+01,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 4.86604e+01, -5.48875e+01, -3.97262e+01, -2.10799e-01,  4.42879e-01, -7.97753e-02,  8.33333e-01}},
    {{ 3.42658e+01, -6.76518e+01,  0.00000e+00, -3.65925e+00,  7.14224e-01, -1.07508e-03,  0.00000e+00}},
};

/* Ne20(a,g)Mg24: NACRE (Angulo+ 1999) */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rne20ag[] = {
    {{ 2.68017e+01, -1.17334e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{-1.38869e+01, -1.10620e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 4.93244e+01, -1.08114e+02, -4.62525e+01,  5.58901e+00,  7.61843e+00, -3.68300e+00,  8.33333e-01}},
    {{ 1.60203e+01, -1.20895e+02,  0.00000e+00,  1.69229e+01, -2.57325e+00,  2.08997e-01,  0.00000e+00}},
};

/* Mg24(a,g)Si28: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rmg24ag[] = {
    {{ 3.89908e+01, -1.66186e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{-1.23895e+01, -1.59316e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.52250e+01, -1.43656e+02, -6.33980e+01, -1.48750e+00,  2.61250e+01, -7.25750e+00,  8.33333e-01}},
};

/* Si28(a,g)S32: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rsi28ag[] = {
    {{ 4.42778e+01, -1.79782e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{-1.80237e+01, -1.68431e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.54965e+01, -1.56772e+02, -6.64270e+01,  7.05180e-01,  3.08400e+01, -8.42250e+00,  8.33333e-01}},
};

/* S32(a,g)Ar36: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rs32ag[] = {
    {{ 2.12300e+01, -1.87399e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.57920e+01, -1.69466e+02, -6.94270e+01,  1.68950e+00,  3.44200e+01, -9.31500e+00,  8.33333e-01}},
};

/* Ar36(a,g)Ca40: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rar36ag[] = {
    {{ 5.28990e+01, -1.86899e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.62690e+01, -1.82485e+02, -7.24270e+01,  2.69870e+00,  3.77260e+01, -1.01475e+01,  8.33333e-01}},
};

/* Ca40(a,g)Ti44: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rca40ag[] = {
    {{ 4.58750e+01, -1.97302e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.65130e+01, -1.95538e+02, -7.54270e+01,  3.62700e+00,  4.09700e+01, -1.09475e+01,  8.33333e-01}},
};

/* Ti44(a,g)Cr48: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rti44ag[] = {
    {{ 5.02560e+01, -2.11138e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.62610e+01, -2.08637e+02, -7.84270e+01,  4.68900e+00,  4.39800e+01, -1.17175e+01,  8.33333e-01}},
};

/* Cr48(a,g)Fe52: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rcr48ag[] = {
    {{ 5.25460e+01, -2.24370e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.68670e+01, -2.21666e+02, -8.14270e+01,  5.55700e+00,  4.72800e+01, -1.25025e+01,  8.33333e-01}},
};

/* Fe52(a,g)Ni56: CF88 */
GIZMO_GPU_DEVICE static constexpr struct rate_coeff rfe52ag[] = {
    {{ 5.46850e+01, -2.37410e+02,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00}},
    {{ 6.72850e+01, -2.34743e+02, -8.44270e+01,  6.48100e+00,  5.04400e+01, -1.32625e+01,  8.33333e-01}},
};


/* =========================================================================
   Compute all forward and reverse rates at temperature T9
   ========================================================================= */
KOKKOS_INLINE_FUNCTION static void compute_rates(double T9, double fwd[nuclear_aprox13_internal::NR], double rev[nuclear_aprox13_internal::NR])
{
    using namespace nuclear_aprox13_internal;
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

    /* reverse rates from detailed balance (see original .cc commentary). */

    double T932 = T9 * sqrt(T9);
    double T9i  = 1.0 / T9;

    /* reaction 0: 3a -> C12 (3-body, needs T9^3 and extra density factor) */
    {
        double Q_over_T9 = 11.6045 * Q_MeV(0) * T9i;
        rev[0] = 2.003e+20 * T932 * T932 * T932 * exp(-Q_over_T9);
        if (!isfinite(rev[0])) rev[0] = 0.0;
    }

    /* reactions 1-11: standard (a,g) 2-body */
    for (int r = 1; r < NR; r++) {
        double Q_over_T9 = 11.6045 * Q_MeV(r) * T9i;
        if (Q_over_T9 > 500.0) { rev[r] = 0.0; continue; }

        int A_tgt = A_sp(r);       /* target nucleus */
        int A_prod = A_sp(r + 1);  /* product nucleus */
        double mfac = pow((4.0 * (double)A_tgt) / (double)A_prod, 1.5);
        double C_rev = mfac * 4.8359e+09;
        rev[r] = fwd[r] * C_rev * T932 * exp(-Q_over_T9);
        if (!isfinite(rev[r])) rev[r] = 0.0;
    }
}


/* =========================================================================
   Right-hand side: dY/dt for the 13 species.
   ========================================================================= */
KOKKOS_INLINE_FUNCTION static void nuclear_rhs(double rho, double T9, const double Y[nuclear_aprox13_internal::NS],
                        double dYdt[nuclear_aprox13_internal::NS], double *edot_out)
{
    using namespace nuclear_aprox13_internal;
    double fwd[NR], rev[NR];
    compute_rates(T9, fwd, rev);
    memset(dYdt, 0, NS * sizeof(double));

    double rhoNA = rho * avo;

    /* reaction 0: triple-alpha: 3 He4 -> C12 */
    double Ya3 = Y[0] * Y[0] * Y[0];
    double r0f = fwd[0] * rhoNA * rhoNA * Ya3 / 6.0;
    double r0r = rev[0] * Y[1]; /* C12 photo-disintegration */
    double net0 = r0f - r0r;
    dYdt[0] -= 3.0 * net0;
    dYdt[1] += net0;

    /* reactions 1-11: A_i(a,g)A_{i+1} */
    for (int r = 1; r < NR; r++) {
        double rf = fwd[r] * rhoNA * Y[0] * Y[r];
        double rr = rev[r] * Y[r + 1];
        double net = rf - rr;
        dYdt[0]     -= net;
        dYdt[r]     -= net;
        dYdt[r + 1] += net;
    }

    /* energy generation */
    double edot = net0 * Q_MeV(0);
    for (int r = 1; r < NR; r++) {
        double rf = fwd[r] * rhoNA * Y[0] * Y[r];
        double rr = rev[r] * Y[r + 1];
        edot += (rf - rr) * Q_MeV(r);
    }
    *edot_out = edot * MeV_to_erg * avo;
}


/* =========================================================================
   Analytic Jacobian: J[i][j] = d(dY_i/dt) / dY_j
   ========================================================================= */
KOKKOS_INLINE_FUNCTION static void nuclear_jacobian(double rho, double T9, const double Y[nuclear_aprox13_internal::NS], double J[nuclear_aprox13_internal::NS][nuclear_aprox13_internal::NS])
{
    using namespace nuclear_aprox13_internal;
    double fwd[NR], rev[NR];
    compute_rates(T9, fwd, rev);
    memset(J, 0, NS * NS * sizeof(double));

    double rhoNA = rho * avo;

    /* reaction 0: triple-alpha */
    double Ya2 = Y[0] * Y[0];
    double df0_dYa = fwd[0] * rhoNA * rhoNA * Ya2 / 2.0;
    double df0_dYc = -rev[0];

    J[0][0] += -3.0 * df0_dYa;
    J[0][1] += -3.0 * df0_dYc;
    J[1][0] +=  df0_dYa;
    J[1][1] +=  df0_dYc;

    /* reactions 1-11: A_i(a,g)A_{i+1} */
    for (int r = 1; r < NR; r++) {
        int i = r;
        int ip = r + 1;

        double dfr_dYa = fwd[r] * rhoNA * Y[i];
        double dfr_dYi = fwd[r] * rhoNA * Y[0];
        double drr_dYp = -rev[r];

        J[0][0] += -dfr_dYa;
        J[0][i] += -dfr_dYi;
        J[0][ip] += -drr_dYp;

        J[i][0] += -dfr_dYa;
        J[i][i] += -dfr_dYi;
        J[i][ip] += -drr_dYp;

        J[ip][0] += dfr_dYa;
        J[ip][i] += dfr_dYi;
        J[ip][ip] += drr_dYp;
    }
}


/* =========================================================================
   13x13 dense linear solver (Gaussian elimination with partial pivoting).
   ========================================================================= */
KOKKOS_INLINE_FUNCTION static int solve_13x13(double A[nuclear_aprox13_internal::NS][nuclear_aprox13_internal::NS], double b[nuclear_aprox13_internal::NS])
{
    using namespace nuclear_aprox13_internal;
    int piv[NS];
    for (int i = 0; i < NS; i++) piv[i] = i;

    for (int k = 0; k < NS; k++) {
        double maxval = fabs(A[piv[k]][k]);
        int maxrow = k;
        for (int i = k + 1; i < NS; i++) {
            double v = fabs(A[piv[i]][k]);
            if (v > maxval) { maxval = v; maxrow = i; }
        }
        if (maxval < 1.0e-100) return -1;
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
   ========================================================================= */
KOKKOS_INLINE_FUNCTION static int backward_euler_step(double rho, double T9, double dt,
                               const double Y_old[nuclear_aprox13_internal::NS], double Y_new[nuclear_aprox13_internal::NS])
{
    using namespace nuclear_aprox13_internal;
    const int max_newton = 10;
    const double tol = 1.0e-8;

    double dYdt[NS];
    double edot_dummy;
    for (int k = 0; k < NS; k++) { Y_new[k] = Y_old[k]; }

    for (int iter = 0; iter < max_newton; iter++) {
        nuclear_rhs(rho, T9, Y_new, dYdt, &edot_dummy);
        double residual[NS];
        for (int k = 0; k < NS; k++) {
            residual[k] = Y_new[k] - Y_old[k] - dt * dYdt[k];
        }

        double Y_max = 0;
        for (int k = 0; k < NS; k++) { if (fabs(Y_old[k]) > Y_max) Y_max = fabs(Y_old[k]); }
        double atol = tol * Y_max;
        double rnorm = 0;
        for (int k = 0; k < NS; k++) {
            double scale = fabs(Y_new[k]) + atol;
            double rel = fabs(residual[k]) / scale;
            if (rel > rnorm) rnorm = rel;
        }
        if (rnorm < tol) return 0;

        double Jac[NS][NS];
        nuclear_jacobian(rho, T9, Y_new, Jac);
        double M[NS][NS];
        for (int i = 0; i < NS; i++) {
            for (int j = 0; j < NS; j++) {
                M[i][j] = -dt * Jac[i][j];
            }
            M[i][i] += 1.0;
        }

        double rhs[NS];
        for (int k = 0; k < NS; k++) rhs[k] = -residual[k];
        if (solve_13x13(M, rhs) != 0) return -1;

        bool bad = false;
        for (int k = 0; k < NS; k++) { if (!isfinite(rhs[k])) { bad = true; break; } }
        if (bad) return -1;
        for (int k = 0; k < NS; k++) {
            Y_new[k] += rhs[k];
            if (Y_new[k] < 0) Y_new[k] = 0;
        }
    }

    return 1;
}


/* =========================================================================
   NSE composition + check (forward-declared so nuclear_aprox13_solve can call).
   Defined just below.
   ========================================================================= */
KOKKOS_INLINE_FUNCTION void nuclear_nse_composition(double rho_cgs, double T9, double Ye,
                             double X_out[NUM_NUCLEAR_SPECIES]);
KOKKOS_INLINE_FUNCTION int nuclear_check_nse(double T9);


/* =========================================================================
   Main solver entry point.
   ========================================================================= */
KOKKOS_INLINE_FUNCTION int nuclear_aprox13_solve(const struct nuclear_input *in, struct nuclear_output *out)
{
    using namespace nuclear_aprox13_internal;
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
            de_cgs += (out->X[k] - X_old[k]) * nuclear_aprox13_BE_per_A(k) * MeV_to_erg * avo;
        }
        out->de = de_cgs / UNIT_SPECEGY_IN_CGS;
        out->edot = (dt_cgs > 0) ? de_cgs / dt_cgs / (UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS) : 0;
        out->burning_timescale = 1.0e-10 / UNIT_TIME_IN_CGS;
        return 0;
    }

    /* convert mass fractions to molar abundances: Y = X / A */
    double Y_old[NS], Y_new[NS];
    for (int k = 0; k < NS; k++) {
        Y_old[k] = (in->X[k] > 0) ? in->X[k] / (double)A_sp(k) : 0.0;
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
            nuclear_rhs(rho_cgs, T9, Y_new, dYdt, &edot_cgs);
            double tau_new = 1e30;
            for (int k = 0; k < NS; k++) {
                if (fabs(dYdt[k]) > 1e-50 && Y_new[k] > 1e-30) {
                    double t = fabs(Y_new[k] / dYdt[k]);
                    if (t < tau_new) tau_new = t;
                }
            }
            if (tau_new > dt_remaining * 0.01) break;
        }
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
                }
                continue;
            }
            double Y_mid[NS]; for (int k = 0; k < NS; k++) Y_mid[k] = 0.5*(Y_new[k]+Y_step[k]);
            nuclear_rhs(rho_cgs, T9, Y_mid, dYdt, &edot_cgs);
            if (isfinite(edot_cgs)) total_energy += edot_cgs * dt_sub;
            memcpy(Y_new, Y_step, sizeof(Y_new));
        }
    }

    /* sanitize Y_new */
    for (int k = 0; k < NS; k++) {
        if (!isfinite(Y_new[k]) || Y_new[k] < 0) Y_new[k] = Y_old[k];
    }

    /* convert back to mass fractions and clamp */
    double sum = 0;
    for (int k = 0; k < NS; k++) {
        out->X[k] = Y_new[k] * (double)A_sp(k);
        if (out->X[k] < 0) out->X[k] = 0;
        if (out->X[k] > 1) out->X[k] = 1;
        sum += out->X[k];
    }
    if (sum > 0) { for (int k = 0; k < NS; k++) out->X[k] /= sum; }

    nuclear_compute_ye_abar(out->X, &out->Ye, &out->Abar);

    /* energy release from binding energy difference */
    double de_binding = 0;
    for (int k = 0; k < NS; k++) {
        de_binding += (out->X[k] - in->X[k]) * nuclear_aprox13_BE_per_A(k)
                    * 1.602176634e-6 * 6.02214076e23;
    }
    out->de   = de_binding / UNIT_SPECEGY_IN_CGS;
    out->edot = (dt_cgs > 0 && fabs(de_binding) > 1.0e-10 * fabs(in->rho * UNIT_DENSITY_IN_CGS))
              ? de_binding / dt_cgs / (UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS) : 0;

    /* recompute burning timescale from the FINAL state */
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
   ========================================================================= */

/* Aliases retired with the Phase C accessor refactor — call
 * nuclear_aprox13_BE_per_A(k) / A_sp(k) / Z_sp(k) directly. */

KOKKOS_INLINE_FUNCTION void nuclear_nse_composition(double rho_cgs, double T9, double Ye,
                             double X_out[NUM_NUCLEAR_SPECIES])
{
    using namespace nuclear_aprox13_internal;
    memset(X_out, 0, NUM_NUCLEAR_SPECIES * sizeof(double));

    double kT_MeV = kerg * T9 * 1.0e9 / MeV_to_erg;

    /* find the most-bound species */
    int i_peak = NUCLEAR_NI56;
    double Q_peak = nuclear_aprox13_BE_per_A(i_peak) * A_sp(i_peak) - (A_sp(i_peak) / 4.0) * nuclear_aprox13_BE_per_A(NUCLEAR_HE4) * 4.0;

    double saha_param = Q_peak / kT_MeV - (A_sp(i_peak) / 4.0 - 1.0) * log(rho_cgs * avo / (T9 * T9 * sqrt(T9) * 1.0e27));

    double f_heavy;
    if (saha_param > 30.0) {
        f_heavy = 1.0;
    } else if (saha_param < -30.0) {
        f_heavy = 0.0;
    } else {
        f_heavy = 1.0 / (1.0 + exp(-saha_param));
    }

    X_out[NUCLEAR_HE4] = 1.0 - f_heavy;
    X_out[i_peak]      = f_heavy;

    if (f_heavy > 0.01 && f_heavy < 0.99) {
        double Q_fe52 = nuclear_aprox13_BE_per_A(NUCLEAR_FE52) * A_sp(NUCLEAR_FE52)
                      - (A_sp(NUCLEAR_FE52) / 4.0) * nuclear_aprox13_BE_per_A(NUCLEAR_HE4) * 4.0;
        double saha_fe52 = Q_fe52 / kT_MeV - (A_sp(NUCLEAR_FE52) / 4.0 - 1.0)
                         * log(rho_cgs * avo / (T9 * T9 * sqrt(T9) * 1.0e27));
        double f_fe52 = 1.0 / (1.0 + exp(-saha_fe52));

        double blend = 4.0 * f_heavy * (1.0 - f_heavy);
        double x_fe52 = blend * 0.3 * f_fe52;
        double x_ni56 = f_heavy - x_fe52;
        if (x_ni56 < 0) { x_ni56 = 0; x_fe52 = f_heavy; }

        X_out[NUCLEAR_NI56] = x_ni56;
        X_out[NUCLEAR_FE52] = x_fe52;
        X_out[NUCLEAR_HE4]  = 1.0 - x_ni56 - x_fe52;
    }

    /* normalize */
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


/* Check if we should use NSE for given conditions. */
KOKKOS_INLINE_FUNCTION int nuclear_check_nse(double T9)
{
#ifdef NUCLEAR_NETWORK_NSE_TABLE
    return (T9 * 1.0e9 > All.NuclearNSE_T_threshold);
#else
    (void)T9;
    return 0;
#endif
}


#endif /* NUCLEAR_NETWORK_SOLVER == 0 */
#endif /* NUCLEAR_NETWORK */

#endif /* NUCLEAR_PHYSICS_FUNCTIONS_H */
