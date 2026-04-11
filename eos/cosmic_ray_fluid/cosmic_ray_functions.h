/* cosmic_ray_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations
 * of CR utility functions called from GPU kernels (cooling, nuclear, etc.).
 * This is THE single source of truth — no separate _device.h copies.
 *
 * cosmic_ray_utilities.cc includes this header for its remaining host-only
 * functions that call these.  cooling.cc includes it via proto.h or directly.
 *
 * Include order: after allvars.h, after cell_data.h (for Bfield_microGauss, Urad_eVcm3). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef COSMIC_RAY_FLUID

KOKKOS_INLINE_FUNCTION double cosmicrayfluid_rsol_corrfac(int k) {
#if defined(CRFLUID_ALT_RSOL_FORM)
    return ((CRFLUID_REDUCED_C_CODE(k))/(C_LIGHT_CODE));
#else
    return 1.0;
#endif
}

KOKKOS_INLINE_FUNCTION int return_CRbin_CR_species_ID(int k_CRegy) {
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    return CR_species_ID_in_bin[k_CRegy];
#endif
#if (N_CR_PARTICLE_BINS == 2)
    if(k_CRegy==0) {return 1;} else {return -1;}
#endif
    return 1;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_CR_charge_in_e(int target, int k_CRegy) {
#if (N_CR_PARTICLE_BINS > 2)
    return CR_global_charge_in_bin[k_CRegy];
#endif
#if (N_CR_PARTICLE_BINS == 2)
    if(k_CRegy==0) {return 1;} else {return -1;}
#endif
    return 1;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_CR_rigidity_in_GV(int target, int k_CRegy) {
    double R = 1;
#if (N_CR_PARTICLE_BINS == 2)
    double Rv[2]={1.8, 0.6}; R=Rv[k_CRegy]; // approximate peak energies of each from Cummings et al. 2016 Fig 15
#endif
#if (N_CR_PARTICLE_BINS > 2)
#ifndef GIZMO_GPU_COMPILER
    if(target >= 0) {R=CR_return_mean_rigidity_in_bin_in_GV(target,k_CRegy, CellP);} else {R=CR_global_rigidity_at_bin_center[k_CRegy];} // this is pre-defined globally for this bin list
#else
    R=CR_global_rigidity_at_bin_center[k_CRegy]; // GPU fallback: spectral-slope-weighted mean not available on device
#endif
#endif
    return R;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_CRmass_in_mp(int target, int k_CRegy) {
    int species = return_CRbin_CR_species_ID(k_CRegy);
    if(species < 0) {return 0.000544618;} // electrons or positrons
    if(species == 1) {return 1.0;}  // p
    if(species == 7) {return 1.0;}  // pbar
    if(species == 2) {return 10.8;} // B
    if(species == 3) {return 12.0;} // C
    if(species == 4) {return 9.0;}  // Be7-9
    if(species == 5) {return 10.;}  // Be10
    if(species == 6) {return 14.8;} // CNO bin
    double Z_abs = fabs(return_CRbin_CR_charge_in_e(target, k_CRegy));
    if(Z_abs > 1.5) {return 2.*Z_abs;} else {return 1;}
}

KOKKOS_INLINE_FUNCTION double return_CRbin_beta_factor(int target, int k_CRegy) {
    double m_cr_mp = return_CRbin_CRmass_in_mp(target, k_CRegy);
    double q = return_CRbin_CR_rigidity_in_GV(target, k_CRegy) * 1.06579 * fabs(return_CRbin_CR_charge_in_e(target, k_CRegy)) / m_cr_mp;
    double gamma = sqrt(1.+q*q), beta = q/gamma;
    return beta;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_gamma_factor(int target, int k_CRegy) {
    double m_cr_mp = return_CRbin_CRmass_in_mp(target, k_CRegy);
    double q = return_CRbin_CR_rigidity_in_GV(target, k_CRegy) * 1.06579 * fabs(return_CRbin_CR_charge_in_e(target, k_CRegy)) / m_cr_mp;
    return sqrt(1.+q*q);
}

KOKKOS_INLINE_FUNCTION double return_CRbin_kinetic_energy_in_GeV(int target, int k_CRegy) {
    double m_cr_mp = return_CRbin_CRmass_in_mp(target, k_CRegy);
    double R_GV = return_CRbin_CR_rigidity_in_GV(target, k_CRegy), Z = fabs(return_CRbin_CR_charge_in_e(target, k_CRegy));
    double q = R_GV*Z*1.06579/m_cr_mp;
    double gamma = sqrt(1.+q*q), beta = q/gamma;
    double KE_fac = DMAX(0.,(1.-sqrt(DMAX(0.,1.-beta*beta)))) / DMAX(beta,MIN_REAL_NUMBER); if(beta < 0.01) {KE_fac = 0.5*beta;}
    return R_GV * Z * KE_fac;
}

#if !defined(CRFLUID_EVOLVE_SPECTRUM)
KOKKOS_INLINE_FUNCTION void CR_cooling_and_losses(int target, double n_elec, double nHcgs, double dtime_cgs, struct particle_data *pp, struct gas_cell_data *cell) {
    if(dtime_cgs <= 0) {return;}
    int k_CRegy; double f_ion = DMAX(DMIN(Get_Gas_Ionized_Fraction(target, pp, cell), 1.), 0.);
    double a_hadronic = 6.37e-16, b_coulomb_ion_per_GeV = 3.09e-16*(n_elec + 0.57*(1.-f_ion))*HYDROGEN_MASSFRAC;
    for(k_CRegy=0; k_CRegy<N_CR_PARTICLE_BINS; k_CRegy++) {
        double CR_coolrate, Z, species_ID; CR_coolrate=0; Z=fabs(return_CRbin_CR_charge_in_e(target,k_CRegy)); species_ID=return_CRbin_CR_species_ID(k_CRegy);
        if(species_ID > 0) {
#if (N_CR_PARTICLE_BINS > 2)
            double E_GeV=return_CRbin_kinetic_energy_in_GeV(target,k_CRegy), beta=return_CRbin_beta_factor(target,k_CRegy);
            CR_coolrate += b_coulomb_ion_per_GeV * ((Z*Z)/(beta*E_GeV)) * nHcgs;
            if(E_GeV>=0.28) {CR_coolrate += a_hadronic * nHcgs;}
#else
            CR_coolrate = (0.87*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * nHcgs;
#endif
        } else {
            double E_GeV=return_CRbin_kinetic_energy_in_GeV(target,k_CRegy), E_rest=0.000511, gamma=1.+E_GeV/E_rest;
            CR_coolrate += n_elec * nHcgs * 1.39e-16 * DMAX(log(2.*gamma)-0.33,0);
            double b_muG = cell[target].Bfield_microGauss(), U_mag_ev=0.0248342*b_muG*b_muG, U_rad_ev = cell[target].Urad_eVcm3();
            CR_coolrate += 5.2e-20 * gamma * (U_mag_ev + U_rad_ev);
        }
        CR_coolrate *= cosmicrayfluid_rsol_corrfac(k_CRegy);
        double q_CR_cool = exp(-CR_coolrate * dtime_cgs); if(CR_coolrate * dtime_cgs > 20.) {q_CR_cool = 0;}
        cell[target].CosmicRayEnergyPred[k_CRegy] *= q_CR_cool; cell[target].CosmicRayEnergy[k_CRegy] *= q_CR_cool;
        cell[target].CosmicRayFlux[k_CRegy] *= q_CR_cool; cell[target].CosmicRayFluxPred[k_CRegy] *= q_CR_cool;
    }
}
#endif /* !CRFLUID_EVOLVE_SPECTRUM */

#endif /* COSMIC_RAY_FLUID */


/* ========================================================================
 * Functions below are available regardless of COSMIC_RAY_FLUID — they have
 * internal branches for COSMIC_RAY_SUBGRID_LEBRON, RT_ISRF_BACKGROUND, etc.
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION double Get_CosmicRayEnergyDensity_cgs(int i, struct particle_data *pp, struct gas_cell_data *cell) {
    if(i<0) {return 0;}
#ifdef COSMIC_RAY_FLUID
    double u_cr=0; int k; for(k=0;k<N_CR_PARTICLE_BINS;k++) {u_cr += cell[i].CosmicRayEnergyPred[k];}
    return u_cr * (cell[i].Density*All.cf_a3inv / pp[i].Mass) * UNIT_PRESSURE_IN_CGS;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    return cell[i].SubGrid_CosmicRayEnergyDensity*All.cf_a3inv * UNIT_PRESSURE_IN_CGS;
#endif
#ifdef RT_ISRF_BACKGROUND
    double column = evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i,pp) * UNIT_SURFDEN_IN_CGS, sigma_0=2.23e-3;
    double u_cr_0 = sqrt(All.InterstellarRadiationFieldStrength) * 1.6e-12;
    if(column < sigma_0) {return u_cr_0;} else {
        double atten_fac = exp(DMAX(-column/100.,-90)) * (sigma_0/(column+MIN_REAL_NUMBER));
        return atten_fac * u_cr_0;
    }
#endif
    return 1.6e-12;
}

KOKKOS_INLINE_FUNCTION double Get_CosmicRayIonizationRate_cgs(int i, struct particle_data *pp, struct gas_cell_data *cell) {
    double zeta_cr = 0;
#if defined(COSMIC_RAY_FLUID) && (N_CR_PARTICLE_BINS > 2)
    double ecr_units=(cell[i].Density*All.cf_a3inv/pp[i].Mass)*UNIT_PRESSURE_IN_CGS; int k;
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        double T_GeV=return_CRbin_kinetic_energy_in_GeV(-1,k), beta=return_CRbin_beta_factor(-1,k), Z=return_CRbin_CR_charge_in_e(-1,k), gamma=return_CRbin_gamma_factor(-1,k);
        zeta_cr += 3.43e-18 * (Z*Z/T_GeV) * ((1.-0.069*beta*beta+0.14*log(beta*gamma))/beta) * (cell[i].CosmicRayEnergyPred[k]*ecr_units); // cross sections from standard Bethe-Blocke formulation, valid at all CR energies we consider explicitly
    }
#else
    zeta_cr = 1.e-5 * Get_CosmicRayEnergyDensity_cgs(i, pp, cell); // scales following Cummings et al. 2016 to 1.6e-17 per eV/cm^3
#if !defined(COSMIC_RAY_FLUID) && !defined(COSMIC_RAY_SUBGRID_LEBRON) && !defined(RT_ISRF_BACKGROUND)
    double prefac_CR=1.; if(All.ComovingIntegrationOn) {double rhofac = (cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS) / (1000.*COSMIC_BARYON_DENSITY_CGS); if(rhofac < 0.2) {prefac_CR=0;} else {if(rhofac > 200.) {prefac_CR=1;} else {prefac_CR=exp(-1./(rhofac*rhofac));}}}
    zeta_cr *= prefac_CR; // in cosmological runs where we're not following CRs in any sense, turn off CR ionization for any gas with density unless it's >1000 times the cosmic mean density
#endif
#endif
#ifdef METALS
    zeta_cr += 1.e-21 * pp[i].Metallicity[0]/All.SolarAbundances[0]; // include radioactive decay of K-40 and other species, which scales with metallicity
#if (NUM_METAL_SPECIES > 10)
    zeta_cr += 1.e-19 * pp[i].Metallicity[10]/All.SolarAbundances[10]; // Contribution from short-lived radioisotopes, here scaling to Fe abundance - Adams et al 2014 Apj 789 86
#endif
#endif
    return zeta_cr;
}

/* cosmic ray heating of gas, from Guo & Oh 2008, following Mannheim & Schlickeiser 1994.
 We assume all the electron losses go into radiation (and ionization), as the electron coulomb losses into gas are lower than protons by factor of energy and me/mp.
 For protons, we assume 1/6 of the hadronic losses (based on branching ratios) and all of the Coulomb losses thermalize.
 Do want to make sure that the rates we assume here are consistent with those used in the CR cooling routine above. */
KOKKOS_INLINE_FUNCTION double CR_gas_heating(int target, double n_elec, double nH0, double nHcgs, struct particle_data *pp, struct gas_cell_data *cell) {
#if defined(CRFLUID_ALT_DISABLE_LOSSES)
    return 0;
#endif
    double a_hadronic, b_coulomb_ion_per_GeV, f_heat_hadronic;
    a_hadronic = 6.37e-16; b_coulomb_ion_per_GeV = 3.09e-16*(n_elec + 0.57*nH0)*HYDROGEN_MASSFRAC; f_heat_hadronic=1./6.; /* some coefficients; a_hadronic is the default coefficient, b_coulomb_ion_per_GeV the default divided by GeV, b/c we need to divide the energy per CR. note there is an extra factor in principle for the ionization term here compared to its version in the CR losses module above: this represents the fraction of CR energy going into the thermal energy of the gas, as opposed to ionization energy, but this is close to unity */
#if defined(COSMIC_RAY_FLUID) || defined(COSMIC_RAY_SUBGRID_LEBRON)
#if (N_CR_PARTICLE_BINS > 2)
    double e_heat=0, e_CR_units_0=(cell[target].Density*All.cf_a3inv/pp[target].Mass) * UNIT_PRESSURE_IN_CGS / nHcgs; int k_CRegy;
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        double e_cr_units = cell[target].CosmicRayEnergyPred[k_CRegy] * e_CR_units_0;
        if(return_CRbin_CR_species_ID(k_CRegy) > 0)
        {
            double E_GeV = return_CRbin_kinetic_energy_in_GeV(target,k_CRegy), beta = return_CRbin_beta_factor(target,k_CRegy), Z=fabs(return_CRbin_CR_charge_in_e(target,k_CRegy));
            double T_eff_fullion = 0.59*(5./3.-1.)*U_TO_TEMP_UNITS*cell[target].InternalEnergyPred, xm = 0.0286*sqrt(T_eff_fullion/2.e6);
            e_heat += b_coulomb_ion_per_GeV * ((Z*Z*beta*beta)/((beta*beta*beta+xm*xm*xm)*E_GeV)) * e_cr_units; // all protons Coulomb-heat, can be rapid for low-E
            if(E_GeV>=0.28) {e_heat += f_heat_hadronic * a_hadronic * e_cr_units;} // only GeV CRs or higher trigger above threshold for collisions
        }
    }
    return e_heat;
#else
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * Get_CosmicRayEnergyDensity_cgs(target, pp, cell) / nHcgs; /* for N<=2, assume a universal spectral shape, the factor here corrects for the fraction above-threshold for hadronic interactions, and 0.53 likewise for averaging  */
#endif
#elif defined(COOL_LOW_TEMPERATURES) // no CR module, but low-temperature cooling is on, we account for the CRs as a heating source, assuming a MW-like background scaled cosmologically to avoid over-heating IGM at high redshifts //
    double prefac_CR=1.; if(All.ComovingIntegrationOn) {double rhofac = (PROTONMASS_CGS*nHcgs/HYDROGEN_MASSFRAC) / (1000.*COSMIC_BARYON_DENSITY_CGS); if(rhofac < 0.2) {prefac_CR=0;} else {if(rhofac > 200.) {prefac_CR=1;} else {prefac_CR=exp(-1./(rhofac*rhofac));}}} // in cosmological runs, turn off CR heating for any gas with density unless it's >1000 times the cosmic mean density
#ifdef RT_ISRF_BACKGROUND
    prefac_CR *= Get_CosmicRayEnergyDensity_cgs(target, pp, cell)/1.6e-12; // prescription for scaling here assumes ISRF ~ sigma_SFR but zeta_CR ~ t_depletion^-1 ~ sigma_SFR^0.5 assuming the Kennicutt 1998 relation
#endif
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * (1.6e-12*prefac_CR) / (1.e-2 + nHcgs); // assume MW-like CR background modulated by above factor (1.6e-12*prefac_CR)=eCR_cgs here //
#else
    return 0;
#endif
}
