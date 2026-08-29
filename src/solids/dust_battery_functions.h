/* dust_battery_functions.h -- dust-exclusive parts of the magnetic battery.
 *
 * Provides:
 *   build_plasma_state_for_battery(...) -- assembles the SHS25 mu/Omega/omega
 *                                          state from the existing GIZMO cell
 *                                          chemistry / RT / grain helpers.
 *                                          (Same physics already in
 *                                          calculate_and_assign_nonideal_mhd_
 *                                          coefficients in eos.cc -- see
 *                                          SHS25 §2.4 for the proof that
 *                                          alpha_O/H/A in SHS25 Eqs. 10-12 are
 *                                          identical to the existing Wardle
 *                                          coefficients in the J_d -> 0 limit.)
 *   battery_E_dust_TVA(...)             -- E'_bat,d in the terminal-velocity
 *                                          approximation (SHS25 Eq. 17,
 *                                          mostly-neutral form recommended for
 *                                          cosmological/CNM regimes; bit 4).
 *   battery_E_dust_explicit(...)        -- E'_bat,d using the actual dust
 *                                          current J_d summed from grain
 *                                          particles (SHS25 Eq. 9; bit 8 /
 *                                          GRAIN_FLUID).
 *   dust_battery_assemble_E_cell(...)   -- per-cell aggregator. Picks bit 4
 *                                          (TVA) or bit 8 (explicit) based on
 *                                          MHD_BATTERY_MECHANISMS gate.
 *                                          Returns E'_bat,d in cgs statvolt/cm.
 *                                          Called from eos/eos.cc.
 *
 * Reference: Soliman, Hopkins & Squire 2025, ApJ 985, 55 (arXiv:2410.21461).
 *
 * Numerical prefactors and rate-coefficient values match the existing
 * non-ideal MHD module (eos.cc::calculate_and_assign_nonideal_mhd_coefficients):
 *   omega_+n collisions: Pinto & Galli 2008  (~1.57e-9 cm^3/s rate coefficient)
 *   omega_en collisions: Pinto & Galli 2008  (~6.21e-9 cm^3/s for T~100K branch)
 *   omega_dn collisions: Epstein drag, ~7.90e-6 a01^2 sqrt(T/m_n)/(m_n+m_d)
 *   grain charge:        Draine & Sutin 1987 closed-form fit (in
 *                        grain_charge_functions.h).
 *
 * Header-only, KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DUST_BATTERY_FUNCTIONS_H
#define DUST_BATTERY_FUNCTIONS_H

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (4|8))

#include "grain_charge_functions.h"
#include "../hydro/battery_alpha_coeffs.h"

/* --------------------------------------------------------------------------
 * Build the SHS25 plasma state from the existing per-cell chemistry/RT/dust
 * inputs. Mirrors the structure used in eos.cc::calculate_and_assign_
 * nonideal_mhd_coefficients (same numerical prefactors, same fallbacks for
 * pre-cooling / non-COOLING / non-RT runs). Output is in cgs.
 *
 * Inputs read from cell:
 *   T (gas temperature, via gamma_eos / U_TO_TEMP_UNITS)
 *   n_e (from cooling Ne, or CR-equilibrium fallback)
 *   B_Gauss (cell magnetic field magnitude)
 *   metallicity (for f_dust/gas via return_dust_to_metals_ratio_vs_solar)
 *
 * Default representative grain size: 0.1 micron (matches the existing
 * non-ideal MHD module default; can be relaxed when explicit grain size
 * distributions become available via RT_OPACITY_FROM_EXPLICIT_GRAINS).
 *
 * Returns also q_d (signed grain charge in esu) and m_grain (cgs grams)
 * via the out-pointer dust_state, since the TVA expression below needs
 * them separately from the plasma state itself.
 * -------------------------------------------------------------------------- */
struct dust_battery_aux_state
{
    double q_d_esu;           /* signed grain charge in esu                    */
    double m_grain_cgs;       /* grain mass in g                               */
    double n_d_cgs;           /* grain number density [cm^-3]                  */
    double n_p_cgs;           /* ion number density [cm^-3]                    */
    double n_n_cgs;           /* neutral number density [cm^-3]                */
    double T_K;               /* gas temperature [K]                           */
    double B_Gauss;           /* B field magnitude [G]                         */
    double m_ion_grams;       /* representative ion mass [g]                   */
};

KOKKOS_INLINE_FUNCTION
struct sh25_plasma_state build_plasma_state_for_battery(int i, struct gas_cell_data *cell, struct particle_data *pp,
                                                         struct dust_battery_aux_state &aux)
{
    struct sh25_plasma_state s = {};
    aux = dust_battery_aux_state{};

    /* --- Default plasma + dust state (matches eos.cc nonideal-MHD defaults) --- */
    double mean_molecular_weight = 2.38;             /* H2 + He + metals */
    double a_grain_micron = 0.1;
    double f_dustgas = 0.01;
    double m_ion_in_mp = 24.3;                       /* Mg dominates */
    double zeta_cr = Get_CosmicRayIonizationRate_cgs(i, pp, cell);

    /* --- Temperature, mu, and electron fraction --- */
    double T_K = cell[i].gas_temperature_from_u(cell[i].InternalEnergyPred); /* composition from the cooling solve, rather than the local estimate above */
    double n_e_cgs = 0;
#ifdef COOLING
    {
        mean_molecular_weight = cell[i].MeanMolecularWeight; /* the composition the cooling solve determined, rather than a local estimate from Ne */
        T_K = cell[i].gas_temperature_from_u(cell[i].InternalEnergyPred);
        n_e_cgs = cell[i].Ne * HYDROGEN_MASSFRAC * mean_molecular_weight
                  * (cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS);
    }
#endif
#ifdef METALS
    f_dustgas = 0.5 * pp[i].Metallicity[0] * return_dust_to_metals_ratio_vs_solar(i, 0, pp, cell);
#endif

    /* --- Densities, masses, B --- */
    const double rho_cgs = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
    const double n_eff = rho_cgs / PROTONMASS_CGS;
    const double m_neutral_g = mean_molecular_weight * PROTONMASS_CGS;
    const double m_grain_g = 7.51e9 * pow(a_grain_micron / 0.1, 3.0) * PROTONMASS_CGS / PROTONMASS_CGS;
    /* note 7.51e9 above is mass in proton masses; convert to grams for clarity */
    const double m_grain_cgs = 7.51e9 * pow(a_grain_micron / 0.1, 3.0) * PROTONMASS_CGS;
    const double ngr_ngas = (mean_molecular_weight / (m_grain_cgs / PROTONMASS_CGS)) * f_dustgas;
    const double n_d_cgs = ngr_ngas * (n_eff / mean_molecular_weight);
    const double m_ion_g = m_ion_in_mp * PROTONMASS_CGS;

    double B_Gauss = sqrt(DMAX(0.0, cell[i].Bfield().norm_sq())) * All.cf_a2inv * UNIT_B_IN_GAUSS;

    /* --- Grain charge (signed, in units of e) --- */
    const double Z_grain = grain_charge_equilibrium_simple(a_grain_micron, n_eff, T_K,
                                                           n_e_cgs, ngr_ngas, zeta_cr, m_ion_in_mp);
    const double q_d_esu = Z_grain * ELECTRONCHARGE_CGS;
    const double q_d_signed = -ELECTRONCHARGE_CGS * Z_grain * ((Z_grain >= 0) ? 1.0 : -1.0); /* magnitude convention; not used downstream */
    (void)q_d_signed;

    /* If electron density wasn't set above (no COOLING), use the CR-equilibrium
       n_e from the existing non-ideal MHD logic for consistency. */
    if(n_e_cgs <= 0 && zeta_cr > 0) {
        const double ag01 = a_grain_micron / 0.1;
        const double k0 = 1.95e-4 * ag01 * ag01 * sqrt(T_K);
        const double psi_prefac = 167.1 / (ag01 * T_K);
        const double psi = Z_grain * psi_prefac;
        const double k_e = k0 * exp(psi);
        n_e_cgs = zeta_cr / (ngr_ngas * k_e + MIN_REAL_NUMBER);
    }
    const double n_p_cgs = n_e_cgs - Z_grain * n_d_cgs;     /* charge balance */
    const double n_n_cgs = DMAX(0.0, n_eff - n_e_cgs - n_p_cgs);

    /* --- Collision frequencies (omega_jn = rho_n <sigma v>_jn / (m_j + m_n))
       using the same rate coefficients already in eos.cc nonideal MHD routine.
       Note: those expressions are written in normalized form; convert. --- */
    const double rho_n_cgs = n_n_cgs * m_neutral_g;
    /* electron-neutral: ~6.21e-9 cm^3/s * (T/100)^0.65 (Pinto & Galli 2008) */
    const double sigma_v_en = 6.21e-9 * pow(DMAX(T_K, 1.0) / 100.0, 0.65);
    /* ion-neutral: ~1.57e-9 cm^3/s (Pinto & Galli 2008) */
    const double sigma_v_pn = 1.57e-9;
    /* dust-neutral: Epstein drag, ~7.90e-6 a01^2 sqrt(T/m_n) / (m_n + m_d) (in scaled units) */
    const double ag01 = a_grain_micron / 0.1;
    const double sigma_v_dn = 7.90e-6 * ag01 * ag01 * sqrt(DMAX(T_K, 1.0) / mean_molecular_weight);
    /* convert from "per (m_n + m_target) in proton masses" form back to cgs */
    s.omega_en = rho_n_cgs * sigma_v_en / (ELECTRONMASS_CGS + m_neutral_g);
    s.omega_pn = rho_n_cgs * sigma_v_pn / (m_ion_g + m_neutral_g);
    s.omega_dn = rho_n_cgs * sigma_v_dn / (m_grain_cgs + m_neutral_g);

    /* --- Mobilities (mu = n q^2 / m) --- */
    const double q_e2 = ELECTRONCHARGE_CGS * ELECTRONCHARGE_CGS;
    s.mu_e = n_e_cgs * q_e2 / ELECTRONMASS_CGS;
    s.mu_p = n_p_cgs * q_e2 / m_ion_g;                       /* singly-ionized */
    s.mu_d = n_d_cgs * (q_d_esu * q_d_esu) / m_grain_cgs;

    /* --- Cyclotron frequencies (signed) --- */
    const double inv_c = 1.0 / C_LIGHT_CGS;
    s.Omega_e = (-ELECTRONCHARGE_CGS) * B_Gauss / (ELECTRONMASS_CGS * C_LIGHT_CGS); /* electron has -e */
    s.Omega_p = ELECTRONCHARGE_CGS * B_Gauss / (m_ion_g * C_LIGHT_CGS);
    s.Omega_d = q_d_esu * B_Gauss / (m_grain_cgs * C_LIGHT_CGS);
    (void)inv_c;

    /* --- Aux state for downstream consumers --- */
    aux.q_d_esu     = q_d_esu;
    aux.m_grain_cgs = m_grain_cgs;
    aux.n_d_cgs     = n_d_cgs;
    aux.n_p_cgs     = n_p_cgs;
    aux.n_n_cgs     = n_n_cgs;
    aux.T_K         = T_K;
    aux.B_Gauss     = B_Gauss;
    aux.m_ion_grams = m_ion_g;
    return s;
}


/* --------------------------------------------------------------------------
 * delta a_dg : differential acceleration on dust vs gas from radiation
 * pressure, summed over RT bands.
 *   delta_a_d,g = sum_k [(kappa_dust[k] - kappa_gas[k]) * F_rad[k] / c]
 * Per-band gas/dust split:
 *   - IR band (RT_FREQ_BIN_INFRARED): use rt_kappa_adaptive_IR_band's
 *     existing dust/gas-only separation (the ONLY band for which the IR
 *     helper applies).
 *   - Photoionizing bands (RT_FREQ_BIN_H0, He0, He1, He2): assume
 *     kappa_total ~ kappa_gas (HI/HeI/HeII photoionization dominates), so
 *     kappa_dust - kappa_gas ~ -kappa_total.
 *   - All other bands: assume kappa_total ~ kappa_dust, so
 *     kappa_dust - kappa_gas ~ +kappa_total.
 * cell.Rad_Kappa[k] holds the total per-band kappa in code units, so we use
 * that as kappa_total for the non-IR cases without redundantly calling rt_kappa.
 * Returns vector (cgs cm/s^2).
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
Vec3<double> dust_battery_delta_a_dg(int i, struct gas_cell_data *cell, struct particle_data *pp)
{
    Vec3<double> da = {0,0,0};
#if defined(RT_EVOLVE_FLUX) && defined(RADTRANSFER)
    if(pp[i].Mass <= 0 || cell[i].Density <= 0) return da;
    /* F_rad in physical cgs: cell.Rad_Flux is stored as F * V_phys in code
       units; vol_inv = Density * cf_a3inv / Mass = 1/V_phys_code. */
    const double vol_inv_code = cell[i].Density * All.cf_a3inv / pp[i].Mass;
    const double rad_flux_to_cgs = vol_inv_code * UNIT_FLUX_IN_CGS;
    const double kappa_code_to_cgs = (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS) / UNIT_MASS_IN_CGS;

    for(int k = 0; k < N_RT_FREQ_BINS; k++) {
        double kappa_diff_cgs = 0; /* kappa_dust - kappa_gas, cm^2/g */
#ifdef RT_INFRARED
        if(k == RT_FREQ_BIN_INFRARED) {
            /* IR: use the dust/gas separation in rt_kappa_adaptive_IR_band */
            const double Tdust = cell[i].Dust_Temperature;
            const double Trad  = cell[i].Radiation_Temperature;
            const double kappa_total_code = rt_kappa_adaptive_IR_band(i, Tdust, Trad,  0,  0, pp, cell);
            const double kappa_gas_code   = rt_kappa_adaptive_IR_band(i, Tdust, Trad, -1, -1, pp, cell);
            const double kappa_dust_code  = DMAX(0.0, kappa_total_code - kappa_gas_code);
            kappa_diff_cgs = (kappa_dust_code - kappa_gas_code) * kappa_code_to_cgs;
        } else
#endif
        {
            /* Non-IR: ionizing bands ~ all gas; everything else ~ all dust */
            const double kappa_total_cgs = cell[i].Rad_Kappa[k] * kappa_code_to_cgs;
            int is_ionizing = 0;
#ifdef RT_CHEM_PHOTOION
            if(k == RT_FREQ_BIN_H0) is_ionizing = 1;
#ifdef RT_CHEM_PHOTOION_HE
            if(k == RT_FREQ_BIN_He0 || k == RT_FREQ_BIN_He1 || k == RT_FREQ_BIN_He2) is_ionizing = 1;
#endif
#endif
            kappa_diff_cgs = is_ionizing ? -kappa_total_cgs : +kappa_total_cgs;
        }
        for(int kd = 0; kd < 3; kd++) {
            da[kd] += kappa_diff_cgs * cell[i].Rad_Flux[k][kd] * rad_flux_to_cgs / C_LIGHT_CGS;
        }
    }
#else
    (void)i; (void)cell; (void)pp;
#endif
    return da;
}


/* --------------------------------------------------------------------------
 * battery_E_dust_TVA -- E'_bat,d in the terminal-velocity approximation,
 * cool/mostly-neutral form (SHS25 Eq. 17, recommended for cosmological/CNM
 * regimes):
 *
 *   E'_bat,d = -(m_d delta_a_dg / q_d)
 *              * mu_d * omega_en * omega_pn
 *              / [ mu_d omega_en omega_pn + mu_e omega_pn omega_dn
 *                + mu_p omega_en omega_dn ]
 *
 * Returns E in cgs statvolt/cm. Vector form: parallel to delta_a_dg.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
Vec3<double> battery_E_dust_TVA(const struct sh25_plasma_state &s,
                                const struct dust_battery_aux_state &aux,
                                const Vec3<double> &delta_a_dg)
{
    Vec3<double> E = {0,0,0};
    if(!(fabs(aux.q_d_esu) > 0)) return E;

    const double num = s.mu_d * s.omega_en * s.omega_pn;
    const double den = s.mu_d * s.omega_en * s.omega_pn
                     + s.mu_e * s.omega_pn * s.omega_dn
                     + s.mu_p * s.omega_en * s.omega_dn;
    if(!(den > 0)) return E;

    const double prefac = -(aux.m_grain_cgs / aux.q_d_esu) * (num / den);
    for(int k = 0; k < 3; k++) E[k] = prefac * delta_a_dg[k];
    return E;
}


/* --------------------------------------------------------------------------
 * battery_E_dust_explicit -- E'_bat,d for a given dust current J_d
 * (SHS25 Eq. 9):
 *
 *   E'_bat,d = -alpha_O J_d  -  alpha_H (J_d x b_hat)  -  alpha_A (J_d x b_hat) x b_hat
 *
 * Inputs in cgs: J_d [esu/cm^2/s], B_Gauss for b_hat. Uses sh25_alpha_coeffs.
 * Returns E in cgs statvolt/cm.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
Vec3<double> battery_E_dust_explicit(const struct sh25_alpha &a,
                                     const Vec3<double> &J_d_cgs,
                                     const Vec3<double> &B_vec_Gauss)
{
    Vec3<double> E = {0,0,0};
    const double Bmag = sqrt(B_vec_Gauss.norm_sq());
    Vec3<double> bhat = {0,0,0};
    if(Bmag > 0) for(int k=0;k<3;k++) bhat[k] = B_vec_Gauss[k] / Bmag;

    /* term 1: -alpha_O J_d */
    for(int k=0;k<3;k++) E[k] += -a.alpha_O * J_d_cgs[k];

    /* term 2: -alpha_H (J_d x b_hat) */
    const Vec3<double> JxB = cross(J_d_cgs, bhat);
    for(int k=0;k<3;k++) E[k] += -a.alpha_H * JxB[k];

    /* term 3: -alpha_A (J_d x b_hat) x b_hat */
    const Vec3<double> JxBxB = cross(JxB, bhat);
    for(int k=0;k<3;k++) E[k] += -a.alpha_A * JxBxB[k];

    return E;
}


/* ============================================================================
 * Per-cell aggregator. Picks the appropriate dust-battery branch (TVA or
 * explicit) based on the compile-time MHD_BATTERY_MECHANISMS gate, returns
 * the dust-battery EMF E'_bat,d in cgs statvolt/cm. Caller adds the return
 * value to cell[i].E_battery_T2_cell.
 * ========================================================================== */
KOKKOS_INLINE_FUNCTION
Vec3<MyDouble> dust_battery_assemble_E_cell(int i, struct gas_cell_data *cell, struct particle_data *pp)
{
    Vec3<MyDouble> E_dust = {0,0,0};
    if(pp[i].Mass <= 0 || cell[i].Density <= 0) return E_dust;

    /* Plasma + dust state: shared between TVA and explicit branches */
    struct dust_battery_aux_state aux;
    const struct sh25_plasma_state s = build_plasma_state_for_battery(i, cell, pp, aux);

#if (MHD_BATTERY_MECHANISMS & 8)
    /* Explicit-J_d path. cell.J_dust_cell is populated by a separate grain->gas
       pass (see solids/grain_physics_gpu.cc); if zero, this returns zero. */
    {
        const struct sh25_alpha a = sh25_alpha_coeffs(s);
        Vec3<double> J_d_cgs = {0,0,0};
        for(int k=0;k<3;k++) J_d_cgs[k] = (double)cell[i].J_dust_cell[k];
        const Vec3<double> B_vec_Gauss = {
            (double)cell[i].Bfield()[0] * All.cf_a2inv * UNIT_B_IN_GAUSS,
            (double)cell[i].Bfield()[1] * All.cf_a2inv * UNIT_B_IN_GAUSS,
            (double)cell[i].Bfield()[2] * All.cf_a2inv * UNIT_B_IN_GAUSS,
        };
        const Vec3<double> E_expl = battery_E_dust_explicit(a, J_d_cgs, B_vec_Gauss);
        for(int k=0;k<3;k++) E_dust[k] = (MyDouble)E_expl[k];
    }
#elif (MHD_BATTERY_MECHANISMS & 4)
    /* Terminal-velocity approximation path (no explicit dust evolution). */
    {
        const Vec3<double> da = dust_battery_delta_a_dg(i, cell, pp);
        const Vec3<double> E_tva = battery_E_dust_TVA(s, aux, da);
        for(int k=0;k<3;k++) E_dust[k] = (MyDouble)E_tva[k];
    }
#endif

    return E_dust;
}

#endif /* MHD_BATTERY_MECHANISMS & (4|8) */

#endif /* DUST_BATTERY_FUNCTIONS_H */
