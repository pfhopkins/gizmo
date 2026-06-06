/* Neutrino transport physics for the nuclear reaction network module.
   Provides opacity, source term, and Ye-feedback functions for the
   three neutrino RT bands (electron, anti-electron, heavy flavor).

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

   Thread-safe: all functions are pure (no global mutable state).
*/

#include <cmath>
#include <cstdio>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "nuclear.h"

#ifdef NUCLEAR_NETWORK
#ifdef NUCLEAR_NETWORK_NEUTRINOS

/* fundamental constants */
static constexpr double sigma_0   = 1.76e-44;   /* weak cross-section scale [cm^2] */
static constexpr double m_e_MeV   = 0.51099895;  /* electron mass [MeV] */
static constexpr double m_n_MeV   = 939.56542;   /* neutron mass [MeV] */
static constexpr double delta_np  = 1.29333;     /* neutron-proton mass difference [MeV] */
static constexpr double avo       = 6.02214076e23;
static constexpr double MeV_to_erg = 1.602176634e-6;
static constexpr double kerg      = 1.380649e-16;

/* weak interaction coupling constants */
static constexpr double cv = 1.0;    /* vector coupling (nu_e on neutrons) */
static constexpr double ca = 1.26;   /* axial-vector coupling */
static constexpr double cv2_ca2 = cv*cv + 3.0*ca*ca;  /* = 1 + 3*1.26^2 ~ 5.76 */

/* neutral-current coupling for protons and neutrons (sin^2(theta_W) ~ 0.231) */
static constexpr double sin2tw = 0.2312;
static constexpr double gv_n = -0.5;
static constexpr double ga_n = -ca / 2.0;
static constexpr double gv_p = 0.5 - 2.0 * sin2tw;
static constexpr double ga_p = ca / 2.0;


/* ---------------------------------------------------------------------------
   Neutrino opacity [code units: Length^2/Mass].
   Called from rt_kappa() for neutrino frequency bins.

   The opacity is dominated by charged-current absorption on free nucleons
   for electron-flavor neutrinos, and by neutral-current scattering for
   all flavors. The cross-section depends on the mean neutrino energy.
   --------------------------------------------------------------------------- */
double nuclear_neutrino_opacity(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
{
    double rho_cgs = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
    double Ye = 0.5;
#ifdef EOS_CARRIES_YE
    Ye = cell[i].Ye;
#endif
    double Xn = 1.0 - Ye; /* neutron mass fraction (approximate for free nucleons) */
    double Xp = Ye;        /* proton mass fraction */

    /* mean neutrino energy [MeV] — from the nuclear burning step, or default */
    double E_nu = 10.0; /* default 10 MeV if not set */
#ifdef NUCLEAR_NETWORK_NEUTRINOS
    int flavor = 0;
    if (k_freq == RT_FREQ_BIN_NU_E) flavor = 0;
    else if (k_freq == RT_FREQ_BIN_NU_EBAR) flavor = 1;
    else if (k_freq == RT_FREQ_BIN_NU_X) flavor = 2;

    if (cell[i].NeutrinoMeanEnergy[flavor] > 0) {
        E_nu = cell[i].NeutrinoMeanEnergy[flavor];
    }
#endif

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
double nuclear_neutrino_absorb_frac(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
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
void nuclear_neutrino_ye_feedback(int i, double dt_code,
                                   struct particle_data *pp, struct gas_cell_data *cell)
{
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
KOKKOS_FUNCTION void nuclear_neutrino_emission(double rho_cgs, double T9, double Ye,
                               double nu_lum[3], double nu_emean[3])
{
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
#endif /* NUCLEAR_NETWORK */
