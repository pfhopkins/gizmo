/* two_temperature_functions.h -- two-temperature plasma module.
 *
 * Evolves the electron temperature T_e independently from the ion/gas
 * temperature T_i. Primary state is cell.u_e_cell = (3/2) n_e k_B T_e / rho,
 * the specific electron internal energy per gas mass [code units]. T_e_cell
 * is a derived cache. The total InternalEnergy remains the single conserved
 * hydro variable; this module only re-partitions it locally and (in C3+)
 * routes radiative cooling/Compton/conduction to T_e.
 *
 * Currently implements only the Spitzer e-i Coulomb equilibration source.
 * Because ν_eq dt can be >> 1 in cluster cores (stiff), we solve it
 * ANALYTICALLY: at fixed total u, the temperature difference T_e - T_i decays
 * exponentially with rate α = (1 + n_e/n_i) ν_eq. The (1 + n_e/n_i) factor
 * comes from total internal energy conservation:
 *   d(T_e - T_i)/dt = -(1 + n_e/n_i) (T_e - T_i) ν_eq.
 *
 * Cosmology: u inputs are physical specific energy in code units; rho_phys
 * is obtained by callers as Density * cf_a3inv * UNIT_DENSITY_IN_CGS. T_e,
 * T_i, n_e, n_i are all physical cgs. cf_atime=1 for now; the full
 * comoving-units pass for cooling+battery+2T is a separate addition.
 *
 * References:
 *   Spitzer 1962, "Physics of Fully Ionized Gases" 2nd ed., §5.3
 *   NRL Plasma Formulary 2019 p.34 (energy-transfer rate, Coulomb log)
 *   Stepney 1983, MNRAS 202, 467 (Eq. 14 prefactor)
 *
 * Header-only, KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef TWO_TEMPERATURE_FUNCTIONS_H
#define TWO_TEMPERATURE_FUNCTIONS_H

#ifdef TWO_TEMPERATURE_PLASMA

/* ============================================================================
 * Coulomb logarithm for electron-ion energy transfer (Z=1).
 * NRL Plasma Formulary 2019 p.34. Valid when T_e >= T_i m_e/m_i, which holds
 * in every regime of interest here (T_i/T_e < ~1800 always satisfied).
 * Floor at 1 mirrors the existing cooling.cc style.
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
double two_temp_coulomb_log(double T_e, double n_e)
{
    if(!(T_e > 0) || !(n_e > 0)) {return 1.;}
    const double T_eV = T_e * BOLTZMANN_CGS / ELECTRONVOLT_IN_ERGS;
    double lnL;
    if(T_eV >= 10.) {
        lnL = 24. - log(sqrt(n_e) / T_e);
    } else {
        lnL = 23. - log(sqrt(n_e) / pow(T_e, 1.5));
    }
    return DMAX(lnL, 1.);
}

/* ============================================================================
 * Energy-equilibration rate ν_eq [s^-1] for hydrogen plasma (Z=1, A=1).
 *
 * NRL Plasma Formulary 2019 p.37 "Approximate transport coefficients":
 *   τ_eq^{ei}(s) = 5.87e8 · A_i · T_e[eV]^{3/2} / (n_e[cm^-3] · Z² · lnΛ)
 *
 * Convert to K (T_eV = 8.617e-5 · T_K, so T_eV^{3/2} = (8.617e-5)^{1.5} · T_K^{1.5}
 * = 8.00e-7 · T_K^{1.5}):
 *   τ_eq(s) ≈ 470 · T_K^{1.5} / (n_e · lnΛ)
 *   ν_eq    ≈ 2.127e-3 · n_e · lnΛ / T_K^{1.5}
 *
 * Cross-check: at T_e=1e7 K, n_e=1e-3 cm^-3, lnΛ=38:
 *   τ_eq = 470 * 3.16e10 / 0.038 = 3.9e14 s ≈ 1.24e7 yr
 * which matches Sarazin 1988 cluster equilibration timescales.
 *
 * For mixed H+He plasma, the exact prefactor sum is Σ_s (n_s Z_s²/A_s) lnΛ_s.
 * For primordial fully-ionized (X=0.76, Y=0.24): Σ = 0.82 ρ/m_p, vs n_e =
 * 0.88 ρ/m_p; using n_e here is a ~7% overestimate of the rate. Acceptable
 * for now; refine once mu_meanwt becomes available at the call site.
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
double two_temp_nu_eq(double T_e, double n_e)
{
    if(!(T_e > 0) || !(n_e > 0)) {return 0;}
    const double lnL = two_temp_coulomb_log(T_e, n_e);
    return 2.127e-3 * n_e * lnL / pow(T_e, 1.5);
}

/* ============================================================================
 * u_e <-> T_e conversions. u_e is specific (per gas mass), in code units.
 *   u_e_phys[erg/g] = (3/2) n_e[cm^-3] k_B T_e[K] / rho_phys[g/cm^3]
 *   u_e_code        = u_e_phys / UNIT_SPECEGY_IN_CGS
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
double two_temp_T_e_from_u_e(double u_e_code, double rho_phys, double n_e_phys)
{
    if(!(n_e_phys > 0) || !(rho_phys > 0)) {return 0;}
    const double u_e_phys = u_e_code * UNIT_SPECEGY_IN_CGS;
    return (2./3.) * rho_phys * u_e_phys / (n_e_phys * BOLTZMANN_CGS);
}

KOKKOS_INLINE_FUNCTION
double two_temp_u_e_from_T_e(double T_e, double rho_phys, double n_e_phys)
{
    if(!(rho_phys > 0)) {return 0;}
    const double u_e_phys = 1.5 * n_e_phys * BOLTZMANN_CGS * T_e / rho_phys;
    return u_e_phys / UNIT_SPECEGY_IN_CGS;
}

/* ============================================================================
 * Per-cell analytic e-i equilibration step.
 *
 * Inputs (all physical cgs except where noted):
 *   u_total_code -- total specific internal energy [code units]; UNCHANGED here
 *   u_e_old_code -- specific electron internal energy before the substep [code]
 *   n_e_phys, n_i_phys, rho_phys -- physical cgs
 *   dt_phys      -- substep duration [s]
 *
 * Returns: u_e_new [code units]. Total u is conserved by construction.
 *
 * Math (closed form):
 *   T_total = (2/3) ρ u_total / ((n_e + n_i) k_B)         single-T equiv.
 *   T_e_n   = T_e_from_u_e(u_e_old)
 *   T_i_n   = ((n_e + n_i) T_total - n_e T_e_n) / n_i     conservation
 *   α       = (1 + n_e/n_i) ν_eq(T_e_n, n_e)
 *   ΔT(t)   = (T_e_n - T_i_n) exp(-α t)
 *   T_e_new = T_total + (n_i/(n_e+n_i)) ΔT(dt)
 *
 * Verifies: n_e T_e_new + n_i T_i_new = (n_e + n_i) T_total at all dt. ✓
 *
 * Numerical guards: ν_eq dt > 50 -> exp underflow -> T_e_new = T_total.
 * T_e floor = max(1 K, 0.01 T_total) to prevent unphysical negative T from
 * roundoff under extreme stiffness.
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
double two_temp_relax_step(double u_total_code, double u_e_old_code,
                           double n_e_phys, double n_i_phys, double rho_phys,
                           double dt_phys)
{
    if(!(n_e_phys > 0) || !(n_i_phys > 0) || !(rho_phys > 0) || !(dt_phys > 0)) {
        return u_e_old_code;
    }
    const double u_total_phys = u_total_code * UNIT_SPECEGY_IN_CGS;
    const double T_total = (2./3.) * rho_phys * u_total_phys
                         / ((n_e_phys + n_i_phys) * BOLTZMANN_CGS);
    if(!(T_total > 0)) {return u_e_old_code;}

    const double T_e_n = two_temp_T_e_from_u_e(u_e_old_code, rho_phys, n_e_phys);
    const double T_i_n = ((n_e_phys + n_i_phys) * T_total - n_e_phys * T_e_n)
                       / n_i_phys;

    const double alpha = (1. + n_e_phys / n_i_phys) * two_temp_nu_eq(T_e_n, n_e_phys);
    const double exponent = alpha * dt_phys;
    const double exp_factor = (exponent > 50.) ? 0. : exp(-exponent);
    const double DeltaT = (T_e_n - T_i_n) * exp_factor;

    double T_e_new = T_total + (n_i_phys / (n_e_phys + n_i_phys)) * DeltaT;

    const double T_e_floor = DMAX(1., 0.01 * T_total);
    if(T_e_new < T_e_floor) {T_e_new = T_e_floor;}

    return two_temp_u_e_from_T_e(T_e_new, rho_phys, n_e_phys);
}

#endif /* TWO_TEMPERATURE_PLASMA */
#endif /* TWO_TEMPERATURE_FUNCTIONS_H */
