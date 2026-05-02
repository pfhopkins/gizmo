/* grain_charge_functions.h -- canonical grain charge model.
 *
 * Provides the equilibrium grain charge Z_grain (signed, in units of e) used
 * by GRAIN_LORENTZFORCE and the dust battery (MHD_BATTERY_MECHANISMS bits 4|8).
 *
 *   grain_charge_equilibrium_simple(a_grain_micron, n_eff, T, n_e, ngr_ngas,
 *                                   zeta_cr, m_ion_in_mp)
 *     -> Z_grain (signed). Closed-form Draine & Sutin 1987 fit, with the same
 *        three-branch ψ approximation used in eos.cc::calculate_and_assign_
 *        nonideal_mhd_coefficients. Two paths:
 *          (a) chemistry-driven (n_e known): solve for ψ given the ratio
 *              R = x_e * (e²/(a kT)) / (n_grain/n_neutral) of negative charge
 *              in electrons vs dust.
 *          (b) cosmic-ray-driven (CR ionization equilibrium with grain
 *              recombination dominant): solve ψ via the alternative balance
 *              determined by zeta_cr.
 *        The function picks (a) if n_e > 0, (b) otherwise.
 *
 * Caps (Draine & Sutin 1987 Eqs. 4.1, 4.2):
 *   Z_min = -1 - 0.7  (a / Angstrom)            [electron field emission]
 *   Z_max = +1 + 21   (a / Angstrom)            [ion field emission, V_emit~30 V]
 *
 * Reference: Draine & Sutin 1987, ApJ 320, 803.
 *
 * Header-only KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GRAIN_CHARGE_FUNCTIONS_H
#define GRAIN_CHARGE_FUNCTIONS_H

#if defined(GRAIN_LORENTZFORCE) || (defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (4|8)))

/* --------------------------------------------------------------------------
 * Equilibrium grain charge Z (signed, in units of e). Lifted from the inline
 * block in eos.cc::calculate_and_assign_nonideal_mhd_coefficients which has
 * been validated in production. Returns Z capped at the DS87 field-emission
 * limits.
 *
 * Inputs:
 *   a_grain_micron   grain radius in microns
 *   n_eff_cgs        gas number density (proton-equivalent) [cm^-3]
 *   T_K              gas temperature [K]
 *   n_e_cgs          electron number density [cm^-3]; if <=0, falls back to
 *                    the cosmic-ray equilibrium path
 *   ngr_ngas         number of grains per neutral (dimensionless)
 *   zeta_cr_cgs      cosmic-ray ionization rate [s^-1]; ignored if n_e_cgs > 0
 *   m_ion_in_mp      ion effective mass in proton masses (e.g. 24.3 for Mg)
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
double grain_charge_equilibrium_simple(double a_grain_micron, double n_eff_cgs, double T_K,
                                       double n_e_cgs, double ngr_ngas, double zeta_cr_cgs,
                                       double m_ion_in_mp)
{
    if(!(a_grain_micron > 0) || !(T_K > 0) || !(n_eff_cgs > 0) || !(ngr_ngas > 0)) return 0;
    const double ag01 = a_grain_micron / 0.1;
    const double psi_prefac = 167.1 / (ag01 * T_K);                     /* e^2 / (a k_B T) */
    const double y = sqrt(m_ion_in_mp * PROTONMASS_CGS / ELECTRONMASS_CGS);
    double psi = 0;

    if(n_e_cgs > 0) {
        /* chemistry-driven path: R = ratio of negative charge in e- vs dust */
        const double R = n_e_cgs * psi_prefac / (n_eff_cgs * ngr_ngas + MIN_REAL_NUMBER);
        const double psi_0 = -3.787124454911839; /* DS87 fit, large-R asymptote */
        if(R > 100.0) { psi = psi_0; }
        else if(R < 0.002) { psi = R * (1.0 - y) / (1.0 + 2.0*y*R); }
        else { psi = psi_0 / (1.0 + pow(R / 0.18967, -0.5646)); }
    } else if(zeta_cr_cgs > 0) {
        /* CR-driven path: alpha = coefficient of the recombination balance */
        const double k0 = 1.95e-4 * ag01 * ag01 * sqrt(T_K);
        const double m_neutral = 1.0; /* same convention as eos.cc */
        const double alpha = zeta_cr_cgs * psi_prefac / (ngr_ngas * ngr_ngas * k0 * (n_eff_cgs / m_neutral) + MIN_REAL_NUMBER);
        const double psi_0 = 0.5188025 - 0.804386 * log(y);
        if(alpha < 0.002) { psi = alpha * (1.0 - y) / (1.0 + alpha * (1.0 + y)); }
        else if(alpha < 10.0) { psi = psi_0 / (1.0 + 0.027 / alpha); }
        else { psi = psi_0; }
    }

    double Z = psi / psi_prefac;

    /* DS87 field-emission caps: a_grain in Angstroms = 1e4 * (a in microns) */
    const double a_Angstrom = 1.0e4 * a_grain_micron;
    const double Z_min = -1.0 - 0.7 * a_Angstrom;        /* electron field emission */
    const double Z_max = +1.0 + 21.0 * a_Angstrom;       /* ion field emission, V_emit ~30 V */
    if(Z < Z_min) Z = Z_min;
    if(Z > Z_max) Z = Z_max;
    return Z;
}

#endif

#endif /* GRAIN_CHARGE_FUNCTIONS_H */
