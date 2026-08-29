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

#if defined(CRFLUID_EVOLVE_SPECTRUM)
KOKKOS_INLINE_FUNCTION double CR_return_effective_number_in_bin_in_codeunits(int target, int k_bin, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION double CR_return_spectral_slope_target(int target, int k_bin, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION double CR_return_mean_rigidity_in_bin_in_GV(int target, int k_bin, struct gas_cell_data *cell);
#endif

/* Forward decl: defined ~L103. The CRFLUID_REDUCED_C_CODE(k) macro
 * (declarations/precompiler_logic.h:850) expands to return_CRbin_M1speed(k),
 * used by cosmicrayfluid_rsol_corrfac immediately below (under
 * CRFLUID_ALT_RSOL_FORM) and by several other inline-body call sites
 * earlier in this header than the definition. Phase D fix 2026-05-21. */
KOKKOS_INLINE_FUNCTION double return_CRbin_M1speed(int k_CRegy);

KOKKOS_INLINE_FUNCTION double cosmicrayfluid_rsol_corrfac(int k) {
#if defined(CRFLUID_ALT_RSOL_FORM)
    return ((CRFLUID_REDUCED_C_CODE(k))/(C_LIGHT_CODE));
#else
    return 1.0;
#endif
}

KOKKOS_INLINE_FUNCTION int return_CRbin_CR_species_ID(int k_CRegy) {
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    return All.CR_species_ID_in_bin[k_CRegy];
#endif
#if (N_CR_PARTICLE_BINS == 2)
    if(k_CRegy==0) {return 1;} else {return -1;}
#endif
    return 1;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_CR_charge_in_e(int target, int k_CRegy) {
#if (N_CR_PARTICLE_BINS > 2)
    return All.CR_global_charge_in_bin[k_CRegy];
#endif
#if (N_CR_PARTICLE_BINS == 2)
    if(k_CRegy==0) {return 1;} else {return -1;}
#endif
    return 1;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_CR_rigidity_in_GV(int target, int k_CRegy, struct gas_cell_data *cell) {
    double R = 1;
#if (N_CR_PARTICLE_BINS == 2)
    double Rv[2]={1.8, 0.6}; R=Rv[k_CRegy]; // approximate peak energies of each from Cummings et al. 2016 Fig 15
#endif
#if (N_CR_PARTICLE_BINS > 2)
    if(target >= 0) {R=CR_return_mean_rigidity_in_bin_in_GV(target,k_CRegy, cell);} else {R=All.CR_global_rigidity_at_bin_center[k_CRegy];} // this is pre-defined globally for this bin list
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

KOKKOS_INLINE_FUNCTION double return_CRbin_beta_factor(int target, int k_CRegy, struct gas_cell_data *cell) {
    double m_cr_mp = return_CRbin_CRmass_in_mp(target, k_CRegy);
    double q = return_CRbin_CR_rigidity_in_GV(target, k_CRegy, cell) * 1.06579 * fabs(return_CRbin_CR_charge_in_e(target, k_CRegy)) / m_cr_mp;
    double gamma = sqrt(1.+q*q), beta = q/gamma;
    return beta;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_gamma_factor(int target, int k_CRegy, struct gas_cell_data *cell) {
    double m_cr_mp = return_CRbin_CRmass_in_mp(target, k_CRegy);
    double q = return_CRbin_CR_rigidity_in_GV(target, k_CRegy, cell) * 1.06579 * fabs(return_CRbin_CR_charge_in_e(target, k_CRegy)) / m_cr_mp;
    return sqrt(1.+q*q);
}

KOKKOS_INLINE_FUNCTION double return_CRbin_kinetic_energy_in_GeV(int target, int k_CRegy, struct gas_cell_data *cell) {
    double m_cr_mp = return_CRbin_CRmass_in_mp(target, k_CRegy);
    double R_GV = return_CRbin_CR_rigidity_in_GV(target, k_CRegy, cell), Z = fabs(return_CRbin_CR_charge_in_e(target, k_CRegy));
    double q = R_GV*Z*1.06579/m_cr_mp;
    double gamma = sqrt(1.+q*q), beta = q/gamma;
    double KE_fac = DMAX(0.,(1.-sqrt(DMAX(0.,1.-beta*beta)))) / DMAX(beta,MIN_REAL_NUMBER); if(beta < 0.01) {KE_fac = 0.5*beta;}
    return R_GV * Z * KE_fac;
}

KOKKOS_INLINE_FUNCTION double gamma_eos_of_crs_in_bin(int k_CRegy)
{
    return (4. + 1./return_CRbin_gamma_factor(-1,k_CRegy,nullptr)) / 3.;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_M1speed(int k_CRegy)
{
    double vmax = CRFLUID_SPEEDOFLIGHT_REDUCTION * C_LIGHT_CODE;
#if defined(CRFLUID_ALT_VARIABLE_RSOL) && (N_CR_PARTICLE_BINS > 1)
    double R = All.CR_global_rigidity_at_bin_center[k_CRegy];
    double f = All.CosmicRayDiffusionCoeff * UNIT_LENGTH_IN_KPC * pow(R , 0.8);
    if(f > vmax) {return f;}
#endif
    return vmax;
}

KOKKOS_INLINE_FUNCTION double Get_Gas_CosmicRayPressure(int i, int k_CRegy, struct gas_cell_data *cell)
{
    if((cell[i].Mass > 0) && (cell[i].Density > 0) && (cell[i].CosmicRayEnergyPred[k_CRegy] > 0))
    {
        return (GAMMA_COSMICRAY(k_CRegy)-1.) * (cell[i].CosmicRayEnergyPred[k_CRegy] * cell[i].Density) / cell[i].Mass;
    }
    return 0;
}

KOKKOS_INLINE_FUNCTION double return_CRbin_nuplusminus_asymmetry(int i, int k_CRegy, struct gas_cell_data *cell)
{
    return 1; // default to unity; basically always true with reasonable SC growth rates, even for extremely low ionization fractions //
}

/*<! closure function needed for arbitrarily anisotropic CR distribution function, from Hopkins '21 */
KOKKOS_INLINE_FUNCTION double return_cosmic_ray_anisotropic_closure_function_threechi(int target, int k_CRegy, struct gas_cell_data *cell)
{
    double ecr,ecrv,f,mu1_2,mu2; ecr=cell[target].CosmicRayEnergyPred[k_CRegy];
    double fluxmag2 = cell[target].CosmicRayFluxPred[k_CRegy].norm_sq();
#if defined(CRFLUID_ALT_RSOL_FORM)
    double v_eff_cr = CRFLUID_REDUCED_C_CODE(k_CRegy); // universal reduction factor
#else
    double kappa=cell[target].CosmicRayDiffusionCoeff[k_CRegy], Lgrad_inv=cell[target].Gradients.CosmicRayPressure[k_CRegy].norm_sq();
    Lgrad_inv = (sqrt(Lgrad_inv) / Get_Gas_CosmicRayPressure(target, k_CRegy, cell)) / All.cf_atime;
    double v_eff_cr = DMIN(DMAX(CRFLUID_REDUCED_C_CODE(k_CRegy) , kappa*Lgrad_inv) , C_LIGHT_CODE); // more complicated factor b/c transport not universally slowed-down
#endif
    ecrv=ecr*v_eff_cr; f=fluxmag2/(ecrv*ecrv); mu1_2=DMAX(0,DMIN(1,f));
    if(!isfinite(mu1_2) || fluxmag2 < MIN_REAL_NUMBER || ecrv < MIN_REAL_NUMBER || !isfinite(f) || f < MIN_REAL_NUMBER || !isfinite(ecrv) || !isfinite(fluxmag2)) {mu1_2=0;} else {if(isfinite(f) && f > 1) {mu1_2=1;}} // safety checks for initialization where may have 0's, etc.
    mu1_2 = DMAX(0,DMIN(1,mu1_2)); // one more safety check
    mu2 = (3.+4.*mu1_2) / (5.+2.*sqrt(4.-3.*mu1_2)); // actual closure relation
    return 3.*(1.-mu2)/2.; // definition of chi variable
}

/* calculate the -ion- Alfven speed in a given element, relevant for very short-wavelength modes with frequency larger than the ion-neutral collision timescale (relevant for CRs in particular) */
KOKKOS_INLINE_FUNCTION double Get_Gas_ion_Alfven_speed_i(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
#if !defined(MAGNETIC)
    return cell[i].thermal_soundspeed(); // if no B-fields, just assume Alfven speed equal to thermal sound speed
#endif
    double vA = cell[i].Alfven_speed(); // normal ideal-MHD Alfven speed
    double f_ion = Get_Gas_Ionized_Fraction(i, pp, cell);
    double mu_eff_ion = 1. + (24.305 - 1.)/(1. + pow(f_ion/1.e-3,2)); // -very- crude approximation to transition to heavy-ion dominance at very low ion fractions
    vA /= sqrt(1.e-15 + mu_eff_ion * f_ion); // Alfven speed of interest is that of the ions alone, not the ideal MHD Alfven speed [corrected for weight with crude approximation above - note that in grain-charge dominated regime, this becomes deeply ambiguous] //
    return vA;
}

KOKKOS_INLINE_FUNCTION double CR_get_streaming_loss_rate_coefficient(int target, int k_CRegy, struct particle_data *pp, struct gas_cell_data *cell)
{
    double streamfac = 0;
    if((cell[target].CosmicRayEnergy[k_CRegy] <= MIN_REAL_NUMBER) || (cell[target].CosmicRayEnergy[k_CRegy] <= MIN_REAL_NUMBER)) {return 0;}
    double vA = Get_Gas_ion_Alfven_speed_i(target, pp, cell); /* define Alfven speed, which is what appears in the proper formulation here */
    double v_flux_eff=cell[target].CosmicRayFluxPred[k_CRegy].norm_sq(); int k; // need magnitude of flux vector
    if(v_flux_eff > 0) {v_flux_eff=sqrt(v_flux_eff) / (MIN_REAL_NUMBER + cell[target].CosmicRayEnergyPred[k_CRegy]);} else {v_flux_eff=0;} // effective speed of CRs = |F|/E
    int target_for_cr_gamma = target; // if this = -1, use the gamma factor at the bin-center for evaluating this, if this = target, use the mean gamma of the bin, weighted by the CR energy -- won't give exactly the same result here
    double gamma_0=return_CRbin_gamma_factor(target_for_cr_gamma,k_CRegy, cell), gamma_fac=gamma_0/(gamma_0-1.), beta_fac=return_CRbin_beta_factor(target_for_cr_gamma,k_CRegy, cell); // lorentz factor here, needed in next line, because the loss term here scales with -total- energy, not kinetic energy
    if(beta_fac<0.1) {gamma_fac=2./(beta_fac*beta_fac) -0.5 - 0.125*beta_fac*beta_fac;} // avoid accidental nan
    streamfac = (vA * (beta_fac*beta_fac) / fabs(3.*cell[target].CosmicRayDiffusionCoeff[k_CRegy])) * ((gamma_fac) * return_CRbin_nuplusminus_asymmetry(target,k_CRegy, cell) * v_flux_eff/cosmicrayfluid_rsol_corrfac(k_CRegy) - (3.*(GAMMA_COSMICRAY(k_CRegy)-1.) + (gamma_fac)) * vA * (2./3.) * return_cosmic_ray_anisotropic_closure_function_threechi(target,k_CRegy, cell)); // this is (vA/[3kappa])*(F - 2*chifac*vA*(ecr+3*Pcr))/ecr, using the 'full F' [corrected back from rsol, b/c rsol correction moves outside this for loss terms]
    return streamfac; // probably want to limit to make sure above doesn't take on too extreme a value... also above, initially only had positive term since this removes energy from CRs when streaming super-Alfvenically, but when streaming sub-Alfvenically, could this become a source term with energy going into CRs? seems problematic if vA very high, but then scattering would work inefficiently... so plausible, but really need to be careful again about magnitude...
}

/* estimate amount by which flux of CRs has been reduced relative to solution with c_reduced = c_true, for RSOL with M1 */
KOKKOS_INLINE_FUNCTION double evaluate_cr_transport_reductionfactor(int target, int k_CRegy, int mode, struct gas_cell_data *cell)
{
#if defined(CRFLUID_ALT_RSOL_FORM)
    return cosmicrayfluid_rsol_corrfac(k_CRegy); // uniform reduction factor for all terms
#else
    double kappa = cell[target].CosmicRayDiffusionCoeff[k_CRegy]; /* diffusion coefficient [physical units] */
    double fluxmag=0, Bmag=0, gradmag=0, Lgrad=0, veff=0, P0=Get_Gas_CosmicRayPressure(target, k_CRegy, cell);
    Vec3<double> B_vec = cell[target].CosmicRayFluxPred[k_CRegy]; /* default projection direction is flux itself */
#ifdef MAGNETIC
    B_vec = cell[target].Bfield();
#endif
    Bmag = B_vec.norm_sq(); fluxmag = dot(Vec3<double>(cell[target].CosmicRayFluxPred[k_CRegy]), B_vec); gradmag = dot(Vec3<double>(cell[target].Gradients.CosmicRayPressure[k_CRegy]), B_vec);
    if(Bmag>0) {fluxmag=fabs(fluxmag)/sqrt(Bmag); gradmag=fabs(gradmag)/sqrt(Bmag);}
    if(gradmag>0) {Lgrad = All.cf_atime * P0 / gradmag;}
    if(fluxmag>0 && cell[target].CosmicRayEnergyPred[k_CRegy] > MIN_REAL_NUMBER) {veff = fluxmag / cell[target].CosmicRayEnergyPred[k_CRegy];}
    double vmax = CRFLUID_REDUCED_C_CODE(k_CRegy);
    int use_injectionmod=0;
    if(mode==0) {use_injectionmod=1;} else {if(return_CRbin_CR_species_ID(k_CRegy) < 0) {use_injectionmod=1;}} // for injection, or leptons, where loss=injection at high-RGV (high-diffcoeff, so high veff), more accurate to match suppression factors this way
    if(use_injectionmod) {veff = vmax;} else {veff = DMIN(veff, vmax);} // we're injecting, so the relevant speed here is just the injection speed (note we speed-limit flux to this for timestepping reasons)
    if(use_injectionmod) {Lgrad = 5./UNIT_LENGTH_IN_KPC;} // set initial gradient length to a constant to reduce noise [looking at newer tests, don't -really- need this, but doesn't hurt either]
    double v_max = DMIN( C_LIGHT_CODE , kappa / (MIN_REAL_NUMBER + Lgrad) ); // attempt at a limiter function here to determine if being flux-limited in the equations below //
    double RSOL_over_v_desired = veff / (MIN_REAL_NUMBER + v_max);
    if(isfinite(RSOL_over_v_desired) && (RSOL_over_v_desired > 0) && (RSOL_over_v_desired < MAX_REAL_NUMBER)) {if(RSOL_over_v_desired < 1) {return RSOL_over_v_desired;}}
    return 1;
#endif
}

/* Forward decl: defined L~267. Called from CR_energy_spectrum_injection_fraction below
   under #if (N_CR_PARTICLE_BINS > 2) — definition site is later in this header. */
KOKKOS_INLINE_FUNCTION double return_CRbin_kinetic_energy_in_GeV_binvalsNRR(int k_CRegy);

/* routine which determines the fraction of injected CR energy per 'bin' of CR energy.
    source type [0=SNe; 1=stellar winds; 2-4=unused; 5=sink/AGN]; shock_vel in code units;
    return_index_in_bin=0 returns the energy fraction, =1 returns the injection spectral slope. */
KOKKOS_INLINE_FUNCTION double CR_energy_spectrum_injection_fraction(
    int k_CRegy, int source_type, double shock_vel, int return_index_in_bin,
    int target, struct particle_data *pp, struct gas_cell_data *cell)
{
    double f_bin = 1./N_CR_PARTICLE_BINS;
#if (N_CR_PARTICLE_BINS > 1)
#if (N_CR_PARTICLE_BINS == 2)
    double f_bin_v[2]={0.95 , 0.05}; f_bin=f_bin_v[k_CRegy];
#endif
#if (N_CR_PARTICLE_BINS > 2)
    double f_elec = 0.02;
    double inj_slope = 4.25;
    double R_break_e = 1.0;
    double inj_slope_lowE_e = 4.2;
#if !defined(CRFLUID_ALT_RSOL_FORM)
    inj_slope_lowE_e=4.25;
#endif
    double R=return_CRbin_CR_rigidity_in_GV(-1,k_CRegy,cell); int species=return_CRbin_CR_species_ID(k_CRegy);
    if(species > -200 && R < R_break_e) {inj_slope = inj_slope_lowE_e;}
    double EGeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy);
    f_bin = EGeV * pow(R/R_break_e , 3.-inj_slope) * log(All.CR_global_max_rigidity_in_bin[k_CRegy] / All.CR_global_min_rigidity_in_bin[k_CRegy]);
    if(return_index_in_bin) {return 2.-inj_slope;}
    double f_norm = 1.e-20;
    if(species == -1) {f_norm = f_elec;}
    if(species == +1) {f_norm = 1.-f_elec;}
    if(species == -2) {f_norm = 1.e-10 * f_elec;}
    if(species > 1 && species != 7)
    {
        double Zfac=0, Zfac_ISM=pp[target].Metallicity[0]/All.SolarAbundances[0], mu_wt=return_CRbin_CRmass_in_mp(-1,k_CRegy), Z_cr=fabs(return_CRbin_CR_charge_in_e(-1,k_CRegy)), Mism_over_Mej=1;
        Zfac = Zfac_ISM;
        if(source_type == 1) {Zfac = (Mism_over_Mej*Zfac_ISM + 1.4*DMIN(Zfac_ISM,1.))/(Mism_over_Mej*HYDROGEN_MASSFRAC + HYDROGEN_MASSFRAC);}
        if(source_type == 0) {if(shock_vel>5000./UNIT_VEL_IN_KMS) {Zfac=(Mism_over_Mej*Zfac_ISM + 9.70)/(Mism_over_Mej*HYDROGEN_MASSFRAC + 0.025);} else {Zfac=(Mism_over_Mej*Zfac_ISM + 13.645)/(Mism_over_Mej*HYDROGEN_MASSFRAC + 0.441);}}
        if(species == 2) {Zfac *= 3.7e-9;}
        if(species == 3) {Zfac *= 2.4e-3;}
        if(species == 4) {Zfac *= 1.4e-10;}
        if(species == 5) {Zfac *= 1.4e-20;}
        if(species == 6) {Zfac *= 0.0094;}
        f_norm = Zfac * pow(mu_wt/Z_cr , inj_slope-3.) / mu_wt;
    }
    f_bin *= f_norm;
#endif
#endif
    if(return_index_in_bin) {return 0;}
    return f_bin;
}


#if defined(CRFLUID_EVOLVE_SPECTRUM)

/* quick boolean to determine if we should use relativistic or non-relativistic scalings for a given bin, in our cooling subroutines where we need this to be true across the bin */
KOKKOS_INLINE_FUNCTION int CR_check_if_bin_is_nonrelativistic(int k_bin) // relativistic binflag: all e-, protons/nuclei with R_GV > 1.87655 * A/Z = 1.87655 for protons : need this globally
{
    if(return_CRbin_CR_species_ID(k_bin) <= 0) {return 0;} // assume e-, e+ relativistic here
    else {
        double m_cr_mp = return_CRbin_CRmass_in_mp(-1,k_bin), Zabs = fabs(return_CRbin_CR_charge_in_e(-1,k_bin)); // mass in proton masses, charge in e
        if(All.CR_global_rigidity_at_bin_center[k_bin] < 1.87655*m_cr_mp/Zabs) {return 1;} else {return 0;} // use a simple momentum criterion, dividing exactly for protons as desired here
    }
    //if((All.CR_species_ID_in_bin[k_bin] > 0) && (All.CR_global_rigidity_at_bin_center[k_bin] < 1.87655)) {return 1;} // simpler version if just using e- and p; for now, only protons with division at p=2m0*c, so NR and R expressions equate, qualify as non-relativistic
    return 0; // default to relativistic otherwise
}

/* routine to return bin-centered energy using the slope approximation used in the fast cooling sub-cycling, of each bin being in the relativistic or non-relativistic regime */
KOKKOS_INLINE_FUNCTION double return_CRbin_kinetic_energy_in_GeV_binvalsNRR(int k_CRegy)
{
    double R_GV = All.CR_global_rigidity_at_bin_center[k_CRegy];
    double Zabs = fabs(return_CRbin_CR_charge_in_e(-1,k_CRegy));
    if(CR_check_if_bin_is_nonrelativistic(k_CRegy))
    {
        double fac = 0.5328928536929374; // converts from R_GV to E in GeV for E = p^2/(2m), assuming Z=1, m=mp
        double m_cr_mp = return_CRbin_CRmass_in_mp(-1,k_CRegy); // mass in proton masses
        return fac * R_GV*R_GV * (Zabs*Zabs / m_cr_mp); // E in GeV in non-relativistic limit
    }
    return R_GV * Zabs; // E in GeV in relativistic limit
}

/* input value 'R' = ratio of total CR energy in the bin ('e_tot') to the total CR number times the energy of the CRs with the bin-centered rigidity ('n_tot' x 'E_cr_bin_center_list'), which is a dimensionless function of the slope, whether the bin is relativistic or not, and the bin edges relative to the bin center */
KOKKOS_INLINE_FUNCTION double CR_return_slope_from_number_and_energy_in_bin(double energy_in_code_units, double number_effective_in_code_units, double bin_centered_energy_in_GeV, int k_bin)
{
    if((energy_in_code_units <= 0.) || isnan(energy_in_code_units) || (number_effective_in_code_units <= 0.) || isnan(number_effective_in_code_units)) {return All.CR_global_slope_lut[k_bin][0];}
    double R = energy_in_code_units / (number_effective_in_code_units * bin_centered_energy_in_GeV + MIN_REAL_NUMBER);
    int n_table = N_CR_SPECTRUM_LUT; // table size
    double xm = All.CR_global_min_rigidity_in_bin[k_bin] / All.CR_global_rigidity_at_bin_center[k_bin], xp = All.CR_global_max_rigidity_in_bin[k_bin] / All.CR_global_rigidity_at_bin_center[k_bin], xm_e = xm, xp_e = xp;
    if(CR_check_if_bin_is_nonrelativistic(k_bin)) {xm_e=xm*xm; xp_e=xp*xp;} // sets bounds that this value can possibly obtain
    if(R >= xp_e) {return All.CR_global_slope_lut[k_bin][n_table-1];} // set to maximum
    if(R <= xm_e) {return All.CR_global_slope_lut[k_bin][0];} // set to minimum
    double n_interp = (log(R / xm_e) / log(xp_e / xm_e)) * (n_table-1); // fraction of the way between min and max in our log-spaced table, returns 0-1
    int n0 = (int) floor(n_interp), n1=n0+1;
    if(n1 > n_table-1) {n1=n_table-1;}
    double slope_gamma = All.CR_global_slope_lut[k_bin][n0] + (n_interp-(double)n0) * (All.CR_global_slope_lut[k_bin][n1]-All.CR_global_slope_lut[k_bin][n0]);
    // check for specific bad values that will nan out and avoid them - this introduces really minimal errors
    double tol = 0.0001, badval; // tolerance around the bad values
    badval=-1; if(fabs(slope_gamma - badval) < tol) {if(slope_gamma<badval) {slope_gamma=badval-tol;} else {slope_gamma=badval+tol;}}
    badval=-2; if(fabs(slope_gamma - badval) < tol) {if(slope_gamma<badval) {slope_gamma=badval-tol;} else {slope_gamma=badval+tol;}}
    badval=-3; if(fabs(slope_gamma - badval) < tol) {if(slope_gamma<badval) {slope_gamma=badval-tol;} else {slope_gamma=badval+tol;}}
    return slope_gamma; // ok, safe to return
}

/* return number of CRs in bin, in our strange code units, where the 'number' has units of E_code / GeV. to convert to real number need to multiply by this but that can cause lots of overflow issues so we work with this unit */
KOKKOS_INLINE_FUNCTION double CR_return_effective_number_in_bin_in_codeunits(int target, int k_bin, struct gas_cell_data *cell)
{
    return cell[target].CosmicRay_Number_in_Bin[k_bin];
}

/* return the CR bin spectral slope, depending on what we use as our evolved variable -- note this is the slope of the 1D CR distribution function, df/dp ~ p^alpha, so e.g. N_cr = integral[dp * df/dp] */
KOKKOS_INLINE_FUNCTION double CR_return_spectral_slope_target(int target, int k_bin, struct gas_cell_data *cell)
{
    //return cell[target].CosmicRay_PwrLaw_Slopes_in_Bin[k_bin]; // evolving slopes directly
    return CR_return_slope_from_number_and_energy_in_bin(cell[target].CosmicRayEnergy[k_bin], cell[target].CosmicRay_Number_in_Bin[k_bin], return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_bin), k_bin); // calculate if not evolving directly
}

/* return mean rigidity in GV per CR for CRs in bin */
KOKKOS_INLINE_FUNCTION double CR_return_mean_rigidity_in_bin_in_GV(int target, int k_bin, struct gas_cell_data *cell)
{
    double slope = CR_return_spectral_slope_target(target, k_bin, cell);
    double gamma_one = 1. + slope;
    double R0 = All.CR_global_rigidity_at_bin_center[k_bin], xm = All.CR_global_min_rigidity_in_bin[k_bin] / R0, xp = All.CR_global_max_rigidity_in_bin[k_bin] / R0, xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one);
    double gamma_fac = (gamma_one/(gamma_one+1.)) * (xp_gamma_one*xp - xm_gamma_one*xm) / (xp_gamma_one - xm_gamma_one); // dimensionless term from weighted integral over the bin
    return R0 * gamma_fac; // returns value appropriately weighted over the bin
}

/* return solution for the distance in dimensionless terms from the bin edge for CRs which can propagate to the edge with the dimensionless rate factor given */
KOKKOS_INLINE_FUNCTION double CR_return_new_bin_edge_from_rate(double rate_dt_dimless, double x_m_bin, double x_p_bin, int loss_mode, int NR_key, double additional_variable_dummy)
{
    if(loss_mode <= 0) {return 0;} // hadronic+catastrophic: should never be called [just should never get to this point since hadronic should be done]

    double x_e = 0;
    if(loss_mode == 1 || loss_mode == 4) // adiabatic+brems+streaming
        {if(rate_dt_dimless < 0) {x_e = x_p_bin * exp(rate_dt_dimless);} else {x_e = x_m_bin * exp(rate_dt_dimless);}} // return xp' or xm' depending on sign: solution to dx/dtau=(+/-)x

    if(loss_mode == 2) // coulomb+ionization
    {
        if(NR_key) // non-relativistic Coulomb+ionization terms
            {x_e = pow(x_m_bin*x_m_bin*x_m_bin + 3.*rate_dt_dimless , 1./3.);} // solution to dx/dtau=-1/x^2
        else // relativistic Coulomb+ionization terms
            {x_e = x_m_bin + rate_dt_dimless;} // solution to dx/dtau=-1
    }

    if(loss_mode == 3) // IC+synchrotron
        {if(rate_dt_dimless*x_m_bin < 1.) {x_e = x_m_bin / (1. - rate_dt_dimless * x_m_bin);} else {x_e = x_p_bin;}} // solution for IC+synchrotron: dx/dtau=-x^2

    if(loss_mode == 5) // re-acceleration
        {x_e = pow( DMAX(pow(x_p_bin, additional_variable_dummy) + rate_dt_dimless, 0.), 1./additional_variable_dummy);} // solution to dlnx/dtau = x^-delta; here using additional_variable_dummy = delta_diffcoeff[k]

    if(x_e < x_m_bin) {x_e=x_m_bin;}
    if(x_e > x_p_bin) {x_e=x_p_bin;}
    return x_e; // catch
}

/* integrand needed for numerical evaluation of non-relativistic Coulomb loss terms; this should be G in int_x0^x1 [G] dlnx_old = E_new, so G = x_old * (dN/dx)_old * E_new, in dimensionless bin-centered units: it's crucial here that 'slope' input is ~df/dlogx = "gamma_one" in the notation above, as opposed to "slope_gamma" */
KOKKOS_INLINE_FUNCTION double CR_coulomb_energy_integrand(double x, double tau, double slope)
{
    return pow(x,slope) * pow(x*x*x - 3.*tau , 2./3.);
}

/* integrand needed for numerical evaluation of re-acceleration energy gain terms; this should be G in int_x0^x1 [G] dlnx_old = E_new, so G = x_old * (dN/dx)_old * E_new, in dimensionless bin-centered units: it's crucial here that 'slope' input is ~df/dlogx = "gamma_one" in the notation above, as opposed to "slope_gamma" */
KOKKOS_INLINE_FUNCTION double CR_reaccel_energy_integrand(double x, double tau, double slope, double delta_slope, int NR_key)
{
    double R_slope = 1./delta_slope;
    if(NR_key) {R_slope *= 2.;}
    return pow(x,slope) * pow( pow(x, delta_slope) + tau , R_slope );
}

/* integrand needed for numerical evaluation of Compton loss terms; this should be G in int_x0^x1 [G] dlnx_old = E_new, so G = x_old * (dN/dx)_old * E_new, in dimensionless bin-centered units: it's crucial here that 'slope' input is ~df/dlogx = "gamma_one" in the notation above, as opposed to "slope_gamma" */
KOKKOS_INLINE_FUNCTION double CR_compton_energy_integrand(double x, double tau, double slope)
{
    return pow(x,slope+1.) / (1. + tau*x);
}

/* this routine does the CR cooling/losses and "heating"/re-acceleration for multi-bin CR spectra: i.e. exchanging CR number
    between bins in the multi-bin approximation and modifying the spectral slope within each bin */
KOKKOS_INLINE_FUNCTION void CR_cooling_and_losses_multibin(int target, double n_elec, double nHcgs, double dtime_cgs, int mode_driftkick, struct particle_data *pp, struct gas_cell_data *cell)
{
    /*! loss_mode:
     0 - hadronic+catastrophic: simply remove energy from the bin (N and E decrease together, preserving spectral slope)
     1 - adiabatic+bremstrahhlung: pure multiplicative: Edot ~ E (so instantaneously conserves slope, shifts pmin,p0,pmax, need to calculate flux)
     2 - coulomb+ionization: scale identically, for low-E protons important, messy dependence, similar to e.g. fitting function from Girichidis, dp/dt ~ (1 + p^(-1.9))
     3 - inverse compton+synchrotron: Edot ~ E^2, also pdot~p^2 [ultra-rel b/c e-]: modifies slope
     */
    if(dtime_cgs <= 0 || nHcgs <= 0) {return;} /* catch */

    /* define a bunch of general-use variables below */
    //double *bin_slopes; bin_slopes = cell[target].CosmicRay_PwrLaw_Slopes_in_Bin; // if directly evolving slope
    double *ntot_evolved; ntot_evolved = cell[target].CosmicRay_Number_in_Bin; double bin_slopes[N_CR_PARTICLE_BINS]; // if directly evolving number
    double *Ucr, Ucr_tot=0; if(mode_driftkick==0) {Ucr=cell[target].CosmicRayEnergy;} else {Ucr=cell[target].CosmicRayEnergyPred;}
    int k; for(k=0;k<N_CR_PARTICLE_BINS;k++) {Ucr_tot+=Ucr[k];} // check total energy since some fluid cells can have no CRs
    if(Ucr_tot < MIN_REAL_NUMBER) {return;} // catch - nothing to do here //
    double t=0, dt=0, bbGv=0, E_rest_e_GeV=0.000511, f_ion=DMAX(DMIN(Get_Gas_Ionized_Fraction(target, pp, cell),1.),0.), b_muG=cell[target].Bfield_microGauss(), U_mag_ev=0.0248342*b_muG*b_muG, U_rad_ev=cell[target].Urad_eVcm3();

#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
    double vA=cell[target].Alfven_speed(), vA_ion, vA_touse, kappa_max=0, kappa_i[N_CR_PARTICLE_BINS], delta_diffcoeff[N_CR_PARTICLE_BINS], reaccel_coeff[N_CR_PARTICLE_BINS]; vA_ion=vA/sqrt(f_ion); vA_touse=vA;
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {kappa_i[k] = cell[target].CosmicRayDiffusionCoeff[k]; if(kappa_i[k]>kappa_max) {kappa_max=kappa_i[k];}} // will use diffusion coefficient below
    double reaccel_coeff_0 = 2. * vA_touse*vA_touse / UNIT_TIME_IN_CGS; // below we'll adopt the standard ansatz that the momentum-space diffusion coefficient is related to the spatial coefficient by the usual heuristic expression Dxx*Dpp = <dv^2>
    if(All.Time <= All.TimeBegin || kappa_max*UNIT_LENGTH_IN_CGS*UNIT_VEL_IN_CGS < 1.e25) {reaccel_coeff_0 = 0;} // make sure kappa is initialized and has a physically meaningful value where the prescription here makes sense, or else this will give nonsense values
    reaccel_coeff_0 = DMAX(-0.9/dtime_cgs, DMIN(0.9/dtime_cgs, reaccel_coeff_0)); // our courant-type condition on the timestep should ensure this as well, but just in case, we need to enforce a condition here //
#endif

    //double adiabatic_coeff_divv = (1./3.) * (cell[target].Face_DivVel_ForAdOps*All.cf_a2inv) / UNIT_TIME_IN_CGS ; // coefficient for adiabatic work [compression/expansion terms]. convert to physical units [a2inv], and then cgs for units here. SIGN is flipped from usual convention since we assume convention where positive coefficients = losses, for convenience with everything else below.
    double adiabatic_coeff[N_CR_PARTICLE_BINS], adiabatic_coeff_divv = (1./3.) * (pp[target].Particle_DivVel*All.cf_a2inv) / UNIT_TIME_IN_CGS; // need to use full DivVel to get the right answer here for CR EOS scaling at high-p
    if(All.ComovingIntegrationOn) {adiabatic_coeff_divv += (1./3.) * (3.*All.cf_hubble_a) / UNIT_TIME_IN_CGS;} // adiabatic term from Hubble expansion (needed for cosmological integrations. also converted to physical, cgs, and sign convention we use here.
    // here we calculate the anisotropic stress correction terms, as needed to properly evaluate the CR energy loss. note we don't need to do this for the 'post hydro' correction step since that is the correction for the isotropic pressure used in the Riemann problem; the additional terms here come entirely outside of that.
#if defined(MAGNETIC)
    Vec3<double> bhat = cell[target].Bfield() * All.cf_a2inv;
    double Bmag2=bhat.norm_sq(); if(Bmag2>0) {bhat/=sqrt(Bmag2);}
    bbGv = dot(bhat, cell[target].Gradients.Velocity.matvec(bhat)) * All.cf_a2inv / UNIT_TIME_IN_CGS; // bhat bhat : grad u -> needed to obtain double-dot-product appropriately
    bbGv = DMAX(-0.9/dtime_cgs, DMIN(0.9/dtime_cgs , bbGv)); // our timestep limiter should ensure this, but problem is it responds to the -previous- timestep, so we need to impose an additional check here to prevent a numerical divergence when/if the gradients are inaccurate
#endif
    double adiabatic_min = -0.5*pp[target].Mass*DMAX(DMIN(cell[target].InternalEnergyPred,cell[target].InternalEnergy)-All.MinEgySpec,0.) / (Ucr_tot*dtime_cgs + MIN_REAL_NUMBER); if(adiabatic_coeff_divv < adiabatic_min) {adiabatic_coeff_divv = adiabatic_min;} // limit adiabatic -gains- of CRs (careful about sign convention here, negative means gain!) as this leads to too-large thermal losses, prevented by limiters in our step computing the exchange between CRs and gas in adiabatic calc above //
    adiabatic_coeff_divv = DMAX(-0.9/dtime_cgs, DMIN(0.9/dtime_cgs , adiabatic_coeff_divv)); // our timestep limiter should ensure this, but problem is it responds to the -previous- timestep, so we need to impose an additional check here to prevent a numerical divergence when/if the gradients are inaccurate

    double Ucr_i[N_CR_PARTICLE_BINS], A_wt[N_CR_PARTICLE_BINS], Z[N_CR_PARTICLE_BINS], x_m[N_CR_PARTICLE_BINS], x_p[N_CR_PARTICLE_BINS], R0[N_CR_PARTICLE_BINS], E_GeV[N_CR_PARTICLE_BINS], bin_centered_rate_coeff[N_CR_PARTICLE_BINS], streaming_coeff[N_CR_PARTICLE_BINS], brems_coeff[N_CR_PARTICLE_BINS], M1SpeedCorrFac[N_CR_PARTICLE_BINS]={1}; int NR_key[N_CR_PARTICLE_BINS];
    double hadronic_coeff = 6.37e-16 * nHcgs; // coefficient for hadronic/catastrophic interactions: dEtot/dt = -(coeff) * Etot, or dPtot/dt = -(coeff) * Ptot (since all p effected are in rel limit, and works by deleting N not by lowering individual E. Mannheim & Schlickeiser 1994
    double coulomb_coeff = 3.09e-16 * nHcgs * ((n_elec + 0.57*(1.-f_ion))*HYDROGEN_MASSFRAC); // default Coulomb+ionization (the two scale nearly-identically) normalization divided by GeV, b/c we need to divide the energy per CR. needs to be multiplied by ((Z*Z)/(beta*E_GeV)). Mannheim & Schlickeiser 1994
    double brems_coeff_0 = 1.39e-16  * n_elec * nHcgs; // coefficient for Bremsstrahlung [following Blumenthal & Gould, 1970]: dEkin/dt=4*alpha_finestruct*r_classical_elec^2*c * SUM[n_Z,ion * Z * (Z+1) * (ln[2*gamma_elec]-1/3) * E_kin . this needs to be multiplied by [DMAX(log(2.*gamma)-0.33,0)]; becomes dE/dt = -(coeff) * E, or dP/dt = -(coeff) * P [since all e- in rel limit].
    double synchIC_coeff_0 = 5.2e-20 * (U_mag_ev + U_rad_ev); // synchrotron and inverse compton scale as dE/dt=(4/3)*sigma_Thompson*c*gamma_elec^2*(U_mag+U_rad), where U_mag and U_rad are the magnetic and radiation energy densities, respectively. Ignoring Klein-Nishina corrections here, as they are negligible at <40 GeV and only a ~15% correction up to ~1e5 GeV. U_mag_ev=(B^2/8pi)/(eV/cm^(-3)), here; U_rad=U_rad/(eV/cm^-3). needs to be multiplied by gamma
    double e_ion_coeff = 3.60e-16 * nHcgs * (1.-f_ion), e_coulomb_coeff = 6.40e-16 * nHcgs * n_elec; // electron coulomb + ionization terms - note very similar to proton ionization (slightly different normalization b/c of log terms, and always in relativistic limit. see e.g. Ginzburg and Syrovatskii, 1964; Gould and Burbidge, 1965, Ramaty and Lingenfelter, 1966. note coulomb coeff has a very weak ~0.01*log[E_cr] scaling, but this scaling is basically offset entirely by the beta-dependence of the scaling at energies of interest, and is very weak, so we ignore it. e.g. Gould 72: Edot = (3/2)*sigma_T*ne*me*c^3/(beta^2) * (log[me*c^2*beta*sqrt[gamma-1]/(hbar*sqrt(4pi*e^2*ne/me))] + log[2]*(beta^2 / 2 + 1/gamma) + 1/2 + (gamma-1)^2/(16*gamma^2)

    double dt_min=dtime_cgs, dt_min_e=dt_min, dt_min_p=dt_min, dt_tmp, CourFac=0.4; // courant-like factor for use in subcycling here //
    int sign_flip_adiabatic_terms = 0, sign_key_for_adiabatic_loop = 1; // key that tells us if the adiabatic+brems+streaming terms have a strong sign flip, in which case we need to do 2 loops instead of 1
    double min_R_e=MAX_REAL_NUMBER, min_R_p=MAX_REAL_NUMBER, max_R_e=MIN_REAL_NUMBER, max_R_p=MIN_REAL_NUMBER; // record the minimum/max bin for e and p, for use in timestepping and limiting below (need to know if we're in the smallest bin)
    for(k=0;k<N_CR_PARTICLE_BINS;k++) /* initialize a bunch of variables for the different bins that we'll refer to below */
    {
        Ucr_i[k] = Ucr[k]; // save initial energy for reference at the end of this loop
        Z[k] = return_CRbin_CR_charge_in_e(-1,k); // want bin-centered values, so give index = -1
        R0[k] = All.CR_global_rigidity_at_bin_center[k]; // want bin-centered values, so give index = -1
        E_GeV[k] = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k); // want bin-centered values, so give index = -1
        NR_key[k] = CR_check_if_bin_is_nonrelativistic(k); // key to decide whether to use relativistic or non-relativistic scalings
        A_wt[k] = return_CRbin_CRmass_in_mp(-1,k); // return the proper nuclear weight here, in units of the proton mass
        x_m[k] = All.CR_global_min_rigidity_in_bin[k] / R0[k]; // ratio of min-to-mid-bin CR rigidity or momentum, used for scaling everything below
        x_p[k] = All.CR_global_max_rigidity_in_bin[k] / R0[k]; // ratio of max-to-mid-bin CR rigidity or momentum, used for scaling everything below
        bin_slopes[k] = CR_return_slope_from_number_and_energy_in_bin(Ucr[k], ntot_evolved[k], E_GeV[k], k); // initialize slopes to use below from LUT, if not directly evolving them
        if(All.CR_species_ID_in_bin[k] < 0) {if(R0[k]<min_R_e) {min_R_e=R0[k];}} else {if(R0[k]<min_R_p) {min_R_p=R0[k];}} // record minimum rigidity to use in limits applied below
        if(All.CR_species_ID_in_bin[k] < 0) {if(R0[k]>max_R_e) {max_R_e=R0[k];}} else {if(R0[k]>max_R_p) {max_R_p=R0[k];}} // record minimum rigidity to use in limits applied below
        double three_chi = return_cosmic_ray_anisotropic_closure_function_threechi(target,k, cell); // get the closure function
        adiabatic_coeff[k] = three_chi*adiabatic_coeff_divv + (1.-three_chi)*bbGv; // allows for energy+species-dependent CR anisotropy

        if(All.CR_species_ID_in_bin[k] < 0) {brems_coeff[k] = brems_coeff_0 * DMAX(log(2.*(1.+E_GeV[k]/E_rest_e_GeV))-0.33,0);} else {brems_coeff[k]=0;}
        streaming_coeff[k] = CR_get_streaming_loss_rate_coefficient(target,k, pp, cell) / UNIT_TIME_IN_CGS;

        M1SpeedCorrFac[k] = evaluate_cr_transport_reductionfactor(target, k, 1, cell); /* implement PFH correction term, similar to RHD with RSOL, to account for RSOL in residence time for attenuation */

        // calculate the timestep limit from all possible bins, maximum step. note no constraint from hadronic here b/c cannot 'cross the bin'
        bin_centered_rate_coeff[k] = 0;
        double adiab_brem_coeff = adiabatic_coeff[k] + streaming_coeff[k] + brems_coeff[k]; bin_centered_rate_coeff[k]+=adiab_brem_coeff; // do constraint from adiabatic + Bremsstrahlung
        if(adiab_brem_coeff < 0) {sign_key_for_adiabatic_loop=-1;} // note the net sign of this term for use below
        if(k>0) {if(adiab_brem_coeff * (adiabatic_coeff[k-1] + streaming_coeff[k-1] + brems_coeff[k-1]) < 0) {sign_flip_adiabatic_terms = 1;}} // have a sign flip, and its not negligible in magnitude //
        if(All.CR_species_ID_in_bin[k] < 0) // e- or e+ (leptonic): do constraint from synchrotron + IC
        {
            double IC_sync_coeff = (E_GeV[k]/E_rest_e_GeV) * synchIC_coeff_0; bin_centered_rate_coeff[k]+=IC_sync_coeff;
            double ion_coeff = (e_coulomb_coeff + (1.+0.07*log(E_GeV[k])) * e_ion_coeff) / R0[k]; bin_centered_rate_coeff[k]+=ion_coeff; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
        } else { // p: do constraint from Coulomb + ionization
            if(NR_key[k]==1) // bin is in non-relativistic limit, use those expressions
            {
                double Coul_coeff = ((0.88 * A_wt[k]*A_wt[k]) / (R0[k]*R0[k]*R0[k] * fabs(Z[k]))) * coulomb_coeff; bin_centered_rate_coeff[k]+=Coul_coeff; // non-relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc. should be multiplied by atomic weight A^2 as well.
            } else { // bin is in relativistic limit, use those expressions
                double Coul_coeff = (fabs(Z[k]/R0[k])) * coulomb_coeff; bin_centered_rate_coeff[k]+=Coul_coeff; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
            }
        }
    }

    if(sign_flip_adiabatic_terms==1) {sign_key_for_adiabatic_loop=-1;} // have sign-flips, so adiabatic term -must- have the oppose sign. otherwise -no- sign flips, so just follow the last sign recorded above

#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
    for(k=0;k<N_CR_PARTICLE_BINS;k++) // additional variables need to be initialized here, after the previous loop definitions
    {
        int minbin_flag=0, maxbin_flag=0;
        if(k>0) {if(All.CR_species_ID_in_bin[k-1] != All.CR_species_ID_in_bin[k]) {minbin_flag=1;}} else {minbin_flag=1;} // previous bin doesn't exist or has opposite sign: lowest-E bin for type
        if(k<N_CR_PARTICLE_BINS-1) {if(All.CR_species_ID_in_bin[k+1] != All.CR_species_ID_in_bin[k]) {maxbin_flag=1;}} else {maxbin_flag=1;} // next bin doesn't exist or has opposite sign: highest-E bin for type
        /* need to estimate the log-slope of the spatial diffusion coefficient vs momentum: delta = dln[kappa] / dln[R], for each species type */
        if(minbin_flag) {
            delta_diffcoeff[k] = log((kappa_i[k+1] + MIN_REAL_NUMBER) / (kappa_i[k] + MIN_REAL_NUMBER)) / log((R0[k+1] + MIN_REAL_NUMBER) / (R0[k] + MIN_REAL_NUMBER)); // linear estimate from next bin
        } else if(maxbin_flag) {
            delta_diffcoeff[k] = log((kappa_i[k] + MIN_REAL_NUMBER) / (kappa_i[k-1] + MIN_REAL_NUMBER)) / log((R0[k] + MIN_REAL_NUMBER) / (R0[k-1] + MIN_REAL_NUMBER)); // linear estimate from previous bin
        } else {
            double y_m = log((kappa_i[k-1] + MIN_REAL_NUMBER) / (kappa_i[k] + MIN_REAL_NUMBER)), y_p = log((kappa_i[k+1] + MIN_REAL_NUMBER) / (kappa_i[k] + MIN_REAL_NUMBER));
            double x_m = log((R0[k-1] + MIN_REAL_NUMBER) / (R0[k] + MIN_REAL_NUMBER)), x_p = log((R0[k+1] + MIN_REAL_NUMBER) / (R0[k] + MIN_REAL_NUMBER));
            delta_diffcoeff[k] = (y_p * x_m*x_m - y_m * x_p*x_p) / (x_m * x_p * (x_m - x_p)); // second-order log-slope estimate
        }
        double delta_limit=0.0315059; delta_diffcoeff[k] = DMIN(DMAX(delta_diffcoeff[k], delta_limit),2.-delta_limit); // need to restrict delta value for the quasi-linear theory model below to make any sense (otherwise just gives unphysical answers because relevant integrals all diverge)
        double alpha_denom = delta_diffcoeff[k] * (4.-delta_diffcoeff[k]) * (4.-delta_diffcoeff[k]*delta_diffcoeff[k]); // quasi-linear isotropic turbulent-type assumption for coefficient relating Dpp and Dxx
        reaccel_coeff[k] = -delta_diffcoeff[k] * reaccel_coeff_0 / ((kappa_i[k] + MIN_REAL_NUMBER) * (alpha_denom + MIN_REAL_NUMBER)); // up to a factor of (R/R0)^(-delta) * (1 - slope_gamma/2), this gives the rate in 1/seconds of the effective momentum-space diffusion. note sign here: generally implies energy gain.
        if(fabs(reaccel_coeff[k]) > fabs(delta_diffcoeff[k]*adiabatic_coeff[k])) {reaccel_coeff[k] *= fabs(delta_diffcoeff[k]*adiabatic_coeff[k]) / fabs(reaccel_coeff[k]);}
        bin_centered_rate_coeff[k] += reaccel_coeff[k] * (1. - 0.5*DMIN(bin_slopes[k], 2.)); // limit this to ensure we don't miss if this should be larger
    }
#endif


    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        if((Ucr[k] < 1.e-10 * Ucr_tot) || (bin_centered_rate_coeff[k]==0)) {continue;} // don't bother with timestep limits if the bin contains totally negligible fraction of CR energy
        if(All.ComovingIntegrationOn) {if(All.Time < 0.12) {if(All.CR_species_ID_in_bin[k] < 0) {if(R0[k] >= 0.999*max_R_e) {continue;}}}} // don't need to apply the same timestep in the smallest bin, so long as regulated by the bin above it, since we'll just cool to 0 here in finite time and that's ok
        double abs_bin_coeff_limit = 0.05 * fabs(bin_centered_rate_coeff[k]) * M1SpeedCorrFac[k]; // set threshold for fraction of rate where we need to worry about detailed subcycling: sub-dominant processes not important here. find few percent works well here.
        double adiab_brem_coeff = fabs(adiabatic_coeff[k] + streaming_coeff[k] + brems_coeff[k]) * M1SpeedCorrFac[k]; // do constraint from adiabatic + Bremsstrahlung
        if(adiab_brem_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * log(x_p[k]/x_m[k]) / adiab_brem_coeff; if(All.CR_species_ID_in_bin[k]<0) {dt_min_e=DMIN(dt_min_e, dt_tmp);} else {dt_min_p=DMIN(dt_min_p, dt_tmp);}}
        if(All.CR_species_ID_in_bin[k] < 0) // e- or e+ (leptonic): do constraint from synchrotron + IC
        {
           double IC_sync_coeff = (E_GeV[k]/E_rest_e_GeV) * synchIC_coeff_0 * M1SpeedCorrFac[k];
           if(IC_sync_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * (1./x_m[k] - 1./x_p[k]) / IC_sync_coeff; dt_min_e=DMIN(dt_min_e, dt_tmp);}
           double ion_coeff = (e_coulomb_coeff + (1.+0.07*log(E_GeV[k])) * e_ion_coeff) / R0[k] * M1SpeedCorrFac[k]; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
           if(ion_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * DMIN(x_p[k]-x_m[k], x_m[k]) / ion_coeff; dt_min_e=DMIN(dt_min_e, dt_tmp);}
        } else { // p: do constraint from Coulomb + ionization
           if(NR_key[k]==1) // bin is in non-relativistic limit, use those expressions
           {
               double Coul_coeff = ((0.88 * A_wt[k]*A_wt[k]) / (R0[k]*R0[k]*R0[k] * fabs(Z[k]))) * coulomb_coeff * M1SpeedCorrFac[k]; // non-relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc. should be multiplied by atomic weight A^2 as well.
               if((All.CR_species_ID_in_bin[k] == 1) || (E_GeV[k]/A_wt[k] > 1.)) {if(Coul_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * (x_p[k]*x_p[k]*x_p[k] - x_m[k]*x_m[k]*x_m[k]) / (3.*Coul_coeff); dt_min_p=DMIN(dt_min_p, dt_tmp);}}
           } else { // bin is in relativistic limit, use those expressions
               double Coul_coeff = (fabs(Z[k])/R0[k]) * coulomb_coeff * M1SpeedCorrFac[k]; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
               if(Coul_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * DMIN(x_p[k]-x_m[k], x_m[k]) / Coul_coeff; dt_min_p=DMIN(dt_min_p, dt_tmp);}
           }
        }
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
        double rcoeff_bin = fabs(reaccel_coeff[k]) * (1.-0.5*DMIN(bin_slopes[k],1.)) * M1SpeedCorrFac[k]; // limit b/c this might change in step, then estimate rate coefficient at bin center
        if(rcoeff_bin > abs_bin_coeff_limit) {dt_tmp = CourFac * (pow(x_p[k],delta_diffcoeff[k]) - pow(x_m[k],delta_diffcoeff[k])) / rcoeff_bin;} // set timestep limit
        if(All.CR_species_ID_in_bin[k]<0) {dt_min_e=DMIN(dt_min_e, dt_tmp);} else {dt_min_p=DMIN(dt_min_p, dt_tmp);} // applies to both e and p
#endif
    }
    if(All.ComovingIntegrationOn) {if(Ucr_tot < 1.e-6*pp[target].Mass*cell[target].InternalEnergy) {dt_min_e*=10.; dt_min_p*=10.; dt_min_e=DMAX(dt_min_e,0.01*dtime_cgs); dt_min_p=DMAX(dt_min_p,0.01*dtime_cgs);}} // allow larger slope errors when the CR energy is a negligible fraction of total

    double dt_target = DMIN(dtime_cgs, DMIN(dt_min_e,dt_min_p)); // timescale for subcycling, using constraint above. limit for extreme cases where we might run into expense problems
    if(dt_target < 1.e-4*dtime_cgs)
    {
        if(dt_target < 1.e-8*dtime_cgs)
        {
            printf("WARNING: timestep for subcycling wants to exceed limit: dt_min_e/p=%g/%g dt_tot=%g \n",dt_min_e,dt_min_p,dtime_cgs);
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
            printf(" ID=%llu mode=%d nH=%g ne=%g vA=%g vA_ion=%g dt=%g Utot=%g hadronic_coeff=%g coulomb_coeff=%g brems_coeff_0=%g synchIC_coeff_0=%g (U_mag_eV=%g U_rad_eV=%g) dtmin=%g signflip=%d signkey=%d \n",(unsigned long long)pp[target].ID,mode_driftkick,nHcgs,n_elec,vA,vA_ion,dtime_cgs,Ucr_tot,hadronic_coeff,coulomb_coeff,brems_coeff_0,synchIC_coeff_0,U_mag_ev,U_rad_ev,DMIN(dt_min_e,dt_min_p),sign_flip_adiabatic_terms,sign_key_for_adiabatic_loop);
            for(k=0;k<N_CR_PARTICLE_BINS;k++) {printf("  k=%d U=%g Z=%g R0=%g E=%g NR=%d xm=%g xp=%g slope=%g kappa=%g adiabatic_coeff=%g brem_c=%g stream_c=%g reacc_c=%g delta_DxxSlope=%g IC_sync_c=%g el_ion_coul_c=%g p_ion_coul_c_R/NR=%g/%g binratec=%g \n",k,Ucr[k],Z[k],R0[k],E_GeV[k],NR_key[k],x_m[k],x_p[k],bin_slopes[k],kappa_i[k],adiabatic_coeff[k],brems_coeff[k],streaming_coeff[k],reaccel_coeff[k],delta_diffcoeff[k],(E_GeV[k]/E_rest_e_GeV) * synchIC_coeff_0,(e_coulomb_coeff + (1.+0.07*log(E_GeV[k])) * e_ion_coeff) / R0[k], (fabs(Z[k])/R0[k]) * coulomb_coeff,0.88/ (R0[k]*R0[k]*R0[k] * fabs(Z[k])) * coulomb_coeff,bin_centered_rate_coeff[k]);}
#else
            printf(" ID=%llu mode=%d nH=%g ne=%g dt=%g Utot=%g hadronic_coeff=%g coulomb_coeff=%g brems_coeff_0=%g synchIC_coeff_0=%g (U_mag_eV=%g U_rad_eV=%g) dtmin=%g signflip=%d signkey=%d \n",(unsigned long long)pp[target].ID,mode_driftkick,nHcgs,n_elec,dtime_cgs,Ucr_tot,hadronic_coeff,coulomb_coeff,brems_coeff_0,synchIC_coeff_0,U_mag_ev,U_rad_ev,DMIN(dt_min_e,dt_min_p),sign_flip_adiabatic_terms,sign_key_for_adiabatic_loop);
            for(k=0;k<N_CR_PARTICLE_BINS;k++) {printf("  k=%d U=%g Z=%g R0=%g E=%g NR=%d xm=%g xp=%g slope=%g adiabatic_coeff=%g bremc=%g strmc=%g binratec=%g \n",k,Ucr[k],Z[k],R0[k],E_GeV[k],NR_key[k],x_m[k],x_p[k],bin_slopes[k],adiabatic_coeff[k],brems_coeff[k],streaming_coeff[k],bin_centered_rate_coeff[k]);}
#endif
        }
        double dt_min_tmp = 1.e-4 * dtime_cgs; // enforce this since our integrators can deal with it, just at some loss of accuracy
        if(dt_target < dt_min_tmp) {dt_min_e=DMAX(dt_min_e, dt_min_tmp); dt_min_p=DMAX(dt_min_p, dt_min_tmp);}
    }

    int cr_species_key; for(cr_species_key=0;cr_species_key<N_CR_PARTICLE_SPECIES;cr_species_key++) // loop over whether we consider nuclei or electrons, first //
    {
        int species_ID=All.CR_species_ID_active_list[cr_species_key]; int n_active=0, bins_sorted[N_CR_PARTICLE_BINS]; double R0_bins[N_CR_PARTICLE_BINS];
        for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(All.CR_species_ID_in_bin[k]==species_ID) {bins_sorted[n_active]=k; R0_bins[n_active]=R0[k]; n_active++;}} // bin is valid: charge matches that desired
        if(n_active<=0) {continue;} // nothing to do here
        if(n_active<N_CR_PARTICLE_BINS) {for(k=n_active;k<N_CR_PARTICLE_BINS;k++) {R0_bins[k]=MAX_REAL_NUMBER;}} // set a dummy value here for sorting purposes below
        //qsort(bins_sorted, n_active, sizeof(int), compare_CR_rigidity_for_sort); // sort on energies from smallest-to-largest [this is hard-coded by requiring the list go in monotonic increasing order for e and p, regardless of how the e and p are themselves ordered //

        t = 0; // reset this before we enter the time integration loop below! //
        if(species_ID<0) {dt_target = DMIN(dtime_cgs, dt_min_e);} else {dt_target = DMIN(dtime_cgs, dt_min_p);} // allow separate timesteps for e and p, since there is no direct exchange between them in the subcycle below
        while(t < dtime_cgs)
        {
            dt = DMIN(dt_target , dtime_cgs - t); // set the subcycle step size
            if(dt <= 0) {break;} // we have reached the end of the timestep - exit loop

            int loss_mode; // this will determine which type[s] of losses [differentiated by their qualitative scalings] we are considering //
            for(loss_mode=0;loss_mode<=5;loss_mode++)
            {
                if(loss_mode==0) {if(species_ID==-1) {continue;}} // loss-mode=0 [Hadronic] uses only nuclei and positrons, skip to next in loop
                if(loss_mode==3) {if(species_ID>0) {continue;}} // loss-mode=3 [Compton+Synchrotron] uses only electrons+positrons
                if(loss_mode==4) {if(sign_flip_adiabatic_terms == 0) {continue;}} // adiabatic+streaming+brems term walked in same order, so we don't need to do an additional loop here
#if !defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
                if(loss_mode==5) {continue;}
#endif

                int order = -1; // default to -descending- energy order [for energy-loss]. but for adiabatic terms may need to use -ascending- order if energy increases
                if(loss_mode==1) {if(sign_key_for_adiabatic_loop < 0) {order = 1;}} // adiabatic: if net energy -gain- in this step [only step where its possible], then switch to ascending order
                if(loss_mode==5) {order = 1;} // reacceleration always produces increasing energies (in the parameter space of interest)

                double dn_flux=0, de_flux=0;
                for(k=0;k<n_active;k++)
                {
                    int j = bins_sorted[k]; // target bin for flux, will move 'up the ladder' passing fluxes to higher energies
                    if(order == -1) {j = bins_sorted[n_active-1-k];} // will move 'down the ladder' passing fluxes to lower energies

                    if(loss_mode==0) // hadronic+catastrophic losses.
                    {
#if 1 // (CRFLUID_EVOLVE_SPECTRUM == 2) // now even the simpler network has secondary e-, important for dense regions synchrotron
                        /* this is where we also include losses from fragmentation and radioactive decay, for heavier nuclei */
                        double frag_coeff=All.CR_frag_coeff[j]*nHcgs, rad_coeff=All.CR_rad_decay_coeff[j], total_catastrophic_coeff=frag_coeff+rad_coeff;
                        if(total_catastrophic_coeff > 0) // have some losses here, account for those
                        {
                            double fac_n=DMIN(total_catastrophic_coeff*dt*M1SpeedCorrFac[j], 60.), fac_e=fac_n; // loss rate for number & energy [equal if loss rate is energy-independent. correction term is small: ~0.95 if loss rate ~1/E [e.g. radioactive], or ~1.05 if rate ~E
                            /*
                            double fac_e=DMIN(total_catastrophic_coeff*dt*M1SpeedCorrFac[j], 60.); // loss rate for energy
                            double slope_gamma=bin_slopes[j], xm=x_m[j], xp=x_p[j], xm_g=pow(xm,slope_gamma), xp_g=pow(xp,slope_gamma), xm_g1=xm_g*xm, xp_g1=xp_g*xp, xm_g2=xm_g1*xm, xp_g2=xp_g1*xp, ecorrfac=1;
                            ecorrfac = (slope_gamma*(2.+slope_gamma))/((1.+slope_gamma)*(1.+slope_gamma)) * ((xm_g1-xp_g1)*(xm_g1-xp_g1)) / ((xm_g-xp_g)*(xm_g2-xp_g2));
                            if(ecorrfac>0 && isfinite(ecorrfac)) {fac_e *= ecorrfac;}
                            */
                            if(All.CR_secondary_target_bin[j][0] > -2) /* now check for whether or not there are any secondary products */
                            {
                                double dfac_e, dfac_n; if(fac_e<0.07) {dfac_e=fac_e-0.5*fac_e*fac_e+fac_e*fac_e*fac_e/6.;} else {dfac_e=1.-exp(-fac_e);}
                                if(fac_n<0.07) {dfac_n=fac_n-0.5*fac_n*fac_n+fac_n*fac_n*fac_n/6.;} else {dfac_n=1.-exp(-fac_n);}
                                double slope_inj = bin_slopes[j]; // slope dN/dp of CRs doing the injection -- should be conserved in production
                                int m; for(m=0;m<N_CR_PARTICLE_SPECIES;m++) {
                                    int j_s = All.CR_secondary_target_bin[j][m]; /* destination bin for secondary product 'm' */
                                    if(j_s < -1) {break;} /* no more secondaries exist, cease this loop */
                                    double secondary_coeff = All.CR_frag_secondary_coeff[j][m]*nHcgs; /* rate of secondary production via fragmentation scales with this */
                                    if(m==0) {secondary_coeff += rad_coeff;} /* 100% of radioactive products go into the first secondary bin */
                                    if(j_s < 0 || secondary_coeff <= 0) {continue;} /* no production in this particular bin/channel */
                                    double frac_secondary = DMAX(0.,DMIN(1., secondary_coeff / total_catastrophic_coeff)); /* restrict to sensible bounds */
                                    if(All.CR_species_ID_in_bin[j_s] < 0) {frac_secondary *= 1./HYDROGEN_MASSFRAC;} // crude correction for He secondary e-/e+ production terms

                                    double U_donor = frac_secondary*dfac_e*Ucr[j] * DMAX(1.,A_wt[j_s])/DMAX(1.,A_wt[j]); // need to account for the different total energy assuming fixed energy per nucleon here
                                    if(All.CR_species_ID_in_bin[j_s] < 0) {U_donor *= 0.1;} // secondary e+/e- from protons (pion decay) get ~0.1 original p energy -- needs to match assumption above
                                    if(All.CR_species_ID_in_bin[j_s] == 7) {U_donor *= 0.08;} // pbar get ~0.08 original p energy -- needs to match assumption above
                                    double N_donor = frac_secondary*dfac_n*ntot_evolved[j]; // absolute number being transferred between bins

                                    if(All.CR_species_ID_in_bin[j]==6 && All.CR_species_ID_in_bin[j_s]==5) {U_donor *= 0.9;} // 10Be assumption needs tiny correction b/c of mean molecular weight of CNO bin putting it slightly in the wrong place (giving problematic slopes)
                                    int split_two_bin=0, j2=-1, js2=-1; /* for some species where we have a big energy jump in the parent and not secondary (e.g. hadrons -> leptons) we get 'jumps' in the spectrum, which produce artificial features; attempt to smooth these out by distributing over a pair of bins */
                                    if((All.CR_species_ID_in_bin[j_s]<0 || All.CR_species_ID_in_bin[j_s]==7) && (k<n_active-1) && k>0) {
                                        j2=bins_sorted[n_active-1-(k+1)]; if(All.CR_species_ID_in_bin[j2]==All.CR_species_ID_in_bin[j]) {
                                            js2=All.CR_secondary_target_bin[j2][m]; if(All.CR_species_ID_in_bin[j_s]==All.CR_species_ID_in_bin[js2]) {
                                                if(js2 < j_s-1) {split_two_bin=1; U_donor *= 0.5; N_donor *= 0.5;}}}}

                                    /* instead of conserving U and N separately, which can cause problems in this step with unphysical slopes owing to discreteness, conserve U and dN/dp */
                                    double E_GeV_s=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(j_s),egy_slopemode_s=1,xm_s=All.CR_global_min_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s],xp_s=All.CR_global_max_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s],xm_e_s=xm_s, xp_e_s=xp_s;
                                    if(CR_check_if_bin_is_nonrelativistic(j_s)) {egy_slopemode_s=2; xm_e_s=xm_s*xm_s; xp_e_s=xp_s*xp_s;} // values needed to scale from slope injected to number and back
                                    double gamma_one_s=slope_inj+1., xm_gamma_one_s=pow(xm_s,gamma_one_s), xp_gamma_one_s=pow(xp_s,gamma_one_s); // variables below
                                    N_donor = (U_donor/E_GeV_s) * ((gamma_one_s + egy_slopemode_s) / (gamma_one_s)) * (xp_gamma_one_s - xm_gamma_one_s) / (xp_gamma_one_s*xp_e_s - xm_gamma_one_s*xm_e_s); // injected number in bin

                                    Ucr[j_s] += U_donor; // update energy in secondary bin
                                    ntot_evolved[j_s] += N_donor; // update number in secondary bin
                                    bin_slopes[j_s] = CR_return_slope_from_number_and_energy_in_bin(Ucr[j_s],ntot_evolved[j_s],E_GeV[j_s],j_s); // get the updated slope for this bin

                                    if(split_two_bin==1) {j_s -= 1;
                                        double facU=0.5/0.5; U_donor *= facU; //facN=0.5/0.5; /* share over a second bin */
                                        Ucr[j_s] += U_donor;
                                        /* repeat exercise to obtain N_donor */
                                        E_GeV_s=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(j_s); egy_slopemode_s=1; xm_s=All.CR_global_min_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s]; xp_s=All.CR_global_max_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s]; xm_e_s=xm_s; xp_e_s=xp_s;
                                        if(CR_check_if_bin_is_nonrelativistic(j_s)) {egy_slopemode_s=2; xm_e_s=xm_s*xm_s; xp_e_s=xp_s*xp_s;} // values needed to scale from slope injected to number and back
                                        gamma_one_s=slope_inj+1.; xm_gamma_one_s=pow(xm_s,gamma_one_s); xp_gamma_one_s=pow(xp_s,gamma_one_s); // variables below
                                        N_donor = (U_donor/E_GeV_s) * ((gamma_one_s + egy_slopemode_s) / (gamma_one_s)) * (xp_gamma_one_s - xm_gamma_one_s) / (xp_gamma_one_s*xp_e_s - xm_gamma_one_s*xm_e_s); // injected number in bin

                                        //ntot_evolved[j_s] += facN * N_donor;
                                        ntot_evolved[j_s] += N_donor; // update number in secondary bin
                                        bin_slopes[j_s] = CR_return_slope_from_number_and_energy_in_bin(Ucr[j_s],ntot_evolved[j_s],E_GeV[j_s],j_s);
                                    }

                                    if(split_two_bin==0 && All.CR_species_ID_in_bin[j]==1 && k==0 && (All.CR_species_ID_in_bin[j_s]<0 || All.CR_species_ID_in_bin[j_s]==7)) { /* now code extending the CR spectrum of secondary production to energies higher than our max limit, assuming continued power-law extrapolation of the CR spectrum */
                                        double Rx0=All.CR_global_rigidity_at_bin_center[j_s], U00=U_donor, xm_0=All.CR_global_min_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s], xp_0=All.CR_global_max_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s];
                                        int spec_0=All.CR_species_ID_in_bin[j_s], slope_0=2.+slope_inj; slope_0=DMAX(-4.,DMIN(slope_0,0.)); if(spec_0==7) {slope_0=DMIN(slope_0,-0.7);}
                                        j_s++;
                                        while(j_s<N_CR_PARTICLE_BINS && All.CR_species_ID_in_bin[j_s]==spec_0) {
                                            double Rx1=All.CR_global_rigidity_at_bin_center[j_s];
                                            E_GeV_s=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(j_s); egy_slopemode_s=1; xm_s=All.CR_global_min_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s]; xp_s=All.CR_global_max_rigidity_in_bin[j_s]/All.CR_global_rigidity_at_bin_center[j_s]; xm_e_s=xm_s; xp_e_s=xp_s;
                                            if(CR_check_if_bin_is_nonrelativistic(j_s)) {egy_slopemode_s=2; xm_e_s=xm_s*xm_s; xp_e_s=xp_s*xp_s;} // values needed to scale from slope injected to number and back
                                            gamma_one_s=slope_inj+1.; xm_gamma_one_s=pow(xm_s,gamma_one_s); xp_gamma_one_s=pow(xp_s,gamma_one_s); // variables below
                                            U_donor = U00 * pow(Rx1/Rx0,slope_0) * log(xp_s/xm_s)/log(xp_0/xm_0);
                                            N_donor = (U_donor/E_GeV_s) * ((gamma_one_s + egy_slopemode_s) / (gamma_one_s)) * (xp_gamma_one_s - xm_gamma_one_s) / (xp_gamma_one_s*xp_e_s - xm_gamma_one_s*xm_e_s); // injected number in bin
                                            Ucr[j_s] += U_donor;
                                            ntot_evolved[j_s] += N_donor;
                                            bin_slopes[j_s] = CR_return_slope_from_number_and_energy_in_bin(Ucr[j_s],ntot_evolved[j_s],E_GeV[j_s],j_s); j_s++;
                                        }}
                                }
                            }
                            Ucr[j] *= exp(-fac_e); ntot_evolved[j] *= exp(-fac_n);
                        }
#else
                        if(E_GeV[j] > 0.28) {double fac=exp(-DMIN(hadronic_coeff*dt*M1SpeedCorrFac[j], 60.)); Ucr[j]*=fac; ntot_evolved[j]*=fac;} // only >~GeV trigger threshold for collisions //
#endif
                        dn_flux=0; de_flux=0; // make sure these are zero'd for next step
                        continue; // these -destroy- CRs, decreasing N and E identically [ignoring secondary production for now], leaving slope un-modified and -no- bin-to-bin fluxes: easy to solve, just modify total energy+number identically in the bin //
                    }

                    double etot = Ucr[j]; // total energy in bin, one of our key evolved variables
                    double E_bin_NtoE = E_GeV[j]; // use this to normalize the number in a convenient unit (just to keep easier to track, units are arbitrary). save it so we don't forget below.
                    double slope_gamma = bin_slopes[j]; // get the initial slope if we evolve it and can pass it as a conserved quantity
                    double ntot = ntot_evolved[j]; // total number in bin [in our effective units] //

                    double xm = x_m[j], xp = x_p[j], gamma_one = slope_gamma+1., xe_gamma_one = 0, xm_gamma_one = 0, xp_gamma_one = 0; // just for convenience below
                    /* // old mode below, where we directly evolve slope and derived number from it
                    double ntot = 0; // total number in bin. we use the slope we obtain to convert between number and energy, given the bin-centered values
                    xm_gamma_one = pow(xm, gamma_one); xp_gamma_one = pow(xp, gamma_one); // variables needed for conversion below
                    if(NR_key[j]) {ntot = ((gamma_one + 2.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm);} // non-relativistic map between E and N
                        else {ntot = ((gamma_one + 1.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp - xm_gamma_one*xm);} // relativistic map between E and N
                    ntot *= (etot / E_bin_NtoE); // dimensional normalization. note the units are arbitrary for the E_bin_NtoE term, as long as we are consistent [we can evolve constant x N, instead of N, for convenience]
                    */

                    double dn_flux_fromprevbin = dn_flux, de_flux_fromprevbin = de_flux; // fluxes coming from the last bin. note these have -already- been cooled, so don't want to double-count that operation, hence we add them at the end of the sub-step
                    double extra_var_topass_for_xe = 0; // dummy variable to pass to subroutine below
                    dn_flux = 0; de_flux = 0; // reset these, they will be re-defined below but in case we skip the relevant loop they need to be zeroed
                    if((etot <= 0 && de_flux_fromprevbin <= 0) || (ntot <= 0 && dn_flux_fromprevbin <= 0)) {continue;} // no CRs to actually work with here [nothing injected yet]!

                    double rate_prefac = 0; // coefficient for the bin-centered loss rate from this mode [-negative- loss rate = energy-gain]
                    if(loss_mode==1 || loss_mode==4) // adiabatic + brems + streaming
                    {
                        if(loss_mode==1) {rate_prefac = adiabatic_coeff[j];} // adiabatic always in mode=1
                        if((sign_flip_adiabatic_terms==0) || (loss_mode==4)) {rate_prefac += brems_coeff[j] + streaming_coeff[j];} // no sign-flip, or in mode=4 so we do add the strictly-loss terms
                    }
                    if(loss_mode==2) // coulomb + ion
                    {
                        if(All.CR_species_ID_in_bin[j] < 0) // electron or positron ionization losses here
                        {
                            rate_prefac = (e_coulomb_coeff + (1.+0.07*log(E_GeV[j])) * e_ion_coeff) / R0[j]; // always in relativistic limit here. 1/R0 is b/c this coefficient is defined normalized to the bin center in GeV, to make the equations dimensionless
                        } else { // proton or nuclei ionization + Coulomb losses here
                            rate_prefac = (fabs(Z[j])/R0[j]) * coulomb_coeff; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
                            if(NR_key[j]) {double A=A_wt[j]; rate_prefac *= 0.88 *A*A / (R0[j]*R0[j]*fabs(Z[j]*Z[j]));} // non-relativistic expression (times constant (A/(R0*Z))^2 //
                        }
                    }
                    if(loss_mode==3) {rate_prefac = (E_GeV[j]/E_rest_e_GeV) * synchIC_coeff_0;} // IC + synch
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
                    if(loss_mode==5) {if(slope_gamma>2.) {rate_prefac=0;} else {rate_prefac=reaccel_coeff[j]*(1.-0.5*slope_gamma); extra_var_topass_for_xe=delta_diffcoeff[j];}} // we will treat gamma as constant over this integration sub-step
#endif

                    int do_cooling_ops = 1; // option to skip the cooling subcycle part of the step here:
                    if(fabs(rate_prefac) < 0.01*fabs(bin_centered_rate_coeff[j])+MIN_REAL_NUMBER) {do_cooling_ops=0;} // this loss mode is negligible here, skip it [pure optimization]
                    if(fabs(bin_centered_rate_coeff[j])*dt*M1SpeedCorrFac[j] < 1.e-16 * Ucr[j]) {do_cooling_ops=0;} // total loss rate is so small we won't be able to track it to floating-point accuracy, skip it
                    if(loss_mode==1 || loss_mode==4) {if((rate_prefac < 0 && order == -1) || (rate_prefac > 0 && order == 1)) {do_cooling_ops=0;}} // adiabatic: make sure we are operating in correct order (will ignore bin if we have a sign switch, which is ok, since it must be nearly-null in that case

                    if(do_cooling_ops) // enter the 'cooling' (bin-to-bin flux calculations) portion of the subcycle
                    {
                        double rate_dt = rate_prefac * dt * M1SpeedCorrFac[j]; // dimensionless step size with rate prefactor times timestep
                        double x_to_edge = CR_return_new_bin_edge_from_rate(rate_dt, xm, xp, loss_mode, NR_key[j], extra_var_topass_for_xe); // returns the dimensionless 'x' representing the particles furthest from bin edge that 'reach' the edge by end of this sub-step
                        gamma_one = slope_gamma+1.; xe_gamma_one = pow(x_to_edge, gamma_one); xm_gamma_one = pow(xm, gamma_one); xp_gamma_one = pow(xp, gamma_one);

                        double dn_lostfrombin = 0; // calculate the number flux integrating out to that new bin edge
                        if(rate_prefac < 0) {dn_lostfrombin = ntot * (1 - xe_gamma_one / xp_gamma_one) / (1 - xm_gamma_one / xp_gamma_one);}
                            else {dn_lostfrombin = ntot * (xe_gamma_one / xm_gamma_one - 1.) / (xp_gamma_one / xm_gamma_one - 1.);}

                        double etot_final_inbin = etot; // calculate the final energy of all particles still inside bin bounds at end of the sub-step
                        double etot_final_allparticlesfrombin = etot; // calculate the final energy of all particles that began sub-step within bin bounds [difference is net flux of energy out-of-bin]

                        if(loss_mode==1 || loss_mode==4) // adiabatic + brems
                        {
                            double norm_fac = exp(-rate_dt), x1, x0, x1_gamma_one, x0_gamma_one; // now make sure we flip the sign convention back to normal for rate
                            if(rate_dt < 0) {x1=x_to_edge; x1_gamma_one=xe_gamma_one; x0=xm; x0_gamma_one=xm_gamma_one;} else {x1=xp; x1_gamma_one=xp_gamma_one; x0=x_to_edge; x0_gamma_one=xe_gamma_one;} // this is everything that remains in bin. so if decaying, runs from x_m' to xp; if growing [rate < 0] runs from xm to x_p'
                            if(NR_key[j])
                            {
                                norm_fac *= norm_fac; // exp(2*rate*dt) because of KE going as p^2
                                etot_final_inbin = etot * norm_fac * (x1_gamma_one*x1*x1 - x0_gamma_one*x0*x0) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm); // extra power of x from extra power of p in KE
                            } else {
                                etot_final_inbin = etot * norm_fac * (x1_gamma_one*x1 - x0_gamma_one*x0) / (xp_gamma_one*xp - xm_gamma_one*xm); // relativistic limit expression
                            }
                            etot_final_allparticlesfrombin = etot * norm_fac; // easy because all particles lose the same fractional energy
                        }
                        if(loss_mode==2) // coulomb + ion
                        {
                            if(NR_key[j])
                            {
                                double norm_fac = etot * (gamma_one+2.) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm); // get the pre-factor for the integral from numbers we already have
                                double xmin_remains = pow(3.*rate_dt, 1./3.); // sets absolute minimum value of x for which any energy remains at all, needs to be integrated only from this bound upwards
                                int n_int=10; double xmin=DMAX(xm,xmin_remains), u_int=0, u_int_xe=0, dlnx=log(xp/xm)/((double)n_int), exp_fac=exp(dlnx/2.), x=xmin*exp_fac, f0=CR_coulomb_energy_integrand(xmin,rate_dt,gamma_one), f1, exp_fac_2=exp_fac*exp_fac, x0=xmin, x1;
                                while(x < xp)
                                {
                                    x1 = DMIN(x*exp_fac, xp);
                                    f1 = CR_coulomb_energy_integrand( x1 , rate_dt, gamma_one);
                                    double u_int_fac = log(x1/x0) * (f1 - f0) / (log((f1+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);
                                    u_int += u_int_fac;
                                    if(x0 < x_to_edge) {if(x1 > x_to_edge) {double f1_e=CR_coulomb_energy_integrand(x_to_edge,rate_dt,gamma_one); u_int_xe+=log(x_to_edge/x0)*(f1_e-f0)/(log((f1_e+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);} else {u_int_xe+=u_int_fac;}}
                                    x *= exp_fac_2; f0 = f1; x0 = x1;
                                }
                                etot_final_inbin = (u_int - u_int_xe) * norm_fac; etot_final_allparticlesfrombin = u_int * norm_fac; // normalize these appropriately
                            } else {
                                double d_egy_fac = rate_dt * ntot * E_bin_NtoE; // re-normalizes ntot correctly back in our chosen units to the -same- units as etot, for use below
                                etot_final_inbin = etot * (xp_gamma_one*xp - xe_gamma_one*x_to_edge) / (xp_gamma_one*xp - xm_gamma_one*xm)
                                    - d_egy_fac * (xp_gamma_one - xe_gamma_one) / (xp_gamma_one - xm_gamma_one); // relativistic limit expression: accounts both for particles leaving the bin [first term] and total energy lost by all particles that stay in the bin [second term]
                                etot_final_allparticlesfrombin = etot - d_egy_fac; // easy because all particles lose the same absolute energy
                            }
                        }
                        if(loss_mode==3) // IC + synch
                        {
                            double norm_fac = etot * (gamma_one+1.) / (xp_gamma_one*xp - xm_gamma_one*xm); // get the pre-factor for the integral from numbers we already have
                            int n_int=10; double u_int=0, u_int_xe=0, dlnx=log(xp/xm)/((double)n_int), exp_fac=exp(dlnx/2.), x=xm*exp_fac, f0=CR_compton_energy_integrand(xm,rate_dt,gamma_one), f1, exp_fac_2=exp_fac*exp_fac, x0=xm, x1;
                            while(x < xp)
                            {
                                x1 = DMIN(x*exp_fac, xp);
                                f1 = CR_compton_energy_integrand( x1 , rate_dt, gamma_one);
                                double u_int_fac = log(x1/x0) * (f1 - f0) / (log((f1+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);
                                u_int += u_int_fac;
                                if(x0 < x_to_edge) {if(x1 > x_to_edge) {double f1_e=CR_compton_energy_integrand(x_to_edge,rate_dt,gamma_one); u_int_xe+=log(x_to_edge/x0)*(f1_e-f0)/(log((f1_e+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);} else {u_int_xe+=u_int_fac;}}
                                x *= exp_fac_2; f0 = f1; x0 = x1;
                            }
                            etot_final_inbin = (u_int - u_int_xe) * norm_fac; etot_final_allparticlesfrombin = u_int * norm_fac; // normalize these appropriately
                        }
                        if(loss_mode==5) // diffusive re-acceleration
                        {
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
                            double norm_fac = etot * (gamma_one+1.) / (xp_gamma_one*xp - xm_gamma_one*xm); // get the pre-factor for the integral from numbers we already have
                            if(NR_key[j]) {norm_fac = etot * (gamma_one+2.) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm);} // use correct NR form
                            double delta_j = delta_diffcoeff[j], rate_dt_abs = fabs(rate_dt); // value of delta-slope needed below
                            int n_int=10; double xmin=xm, u_int=0, u_int_xe=0, dlnx=log(xp/xm)/((double)n_int), exp_fac=exp(dlnx/2.), x=xmin*exp_fac, f0=CR_reaccel_energy_integrand(xmin,rate_dt_abs,gamma_one,delta_j,NR_key[j]), f1, exp_fac_2=exp_fac*exp_fac, x0=xmin, x1;
                            while(x < xp)
                            {
                                x1 = DMIN(x*exp_fac, xp);
                                f1 = CR_reaccel_energy_integrand( x1 , rate_dt_abs , gamma_one , delta_j , NR_key[j]);
                                double u_int_fac = log(x1/x0) * (f1 - f0) / (log((f1+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);
                                u_int += u_int_fac;
                                if(x1 > x_to_edge) {if(x0 < x_to_edge) {double f0_e=CR_reaccel_energy_integrand(x_to_edge, rate_dt_abs, gamma_one, delta_j , NR_key[j]); u_int_xe += log(x1/x_to_edge)*(f1-f0_e)/(log((f1+MIN_REAL_NUMBER)/(f0_e+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);} else {u_int_xe+=u_int_fac;}} // // integrate from x_edge to f1, or x0+x1 both > x_to_edge, so whole bin is outside range
                                x *= exp_fac_2; f0 = f1; x0 = x1;
                            }
                            etot_final_inbin = (u_int - u_int_xe) * norm_fac; etot_final_allparticlesfrombin = u_int * norm_fac; // normalize these appropriately
#endif
                        }

                        dn_lostfrombin = DMIN(dn_lostfrombin , (1.0-1.e-8)*ntot); // limit to prevent 0's which will give nan's
                        etot_final_inbin = DMAX(etot_final_inbin , 1.e-8*etot); // limit to prevent 0's which will give nan's
                        dn_flux = dn_lostfrombin; // set total outgoing flux of number of particles to next bin
                        de_flux = DMAX(etot_final_allparticlesfrombin - etot_final_inbin, 0); // determine total outgoing flux of energy to next bin

                        ntot -= dn_lostfrombin; // update number
                        etot = etot_final_inbin; // update energy
                    } // close the bin-to-bin flux calculation portion of the subcycle

                    ntot += dn_flux_fromprevbin; // update the bin with the -already cooled- CRs from the previous bin
                    etot += de_flux_fromprevbin; // update the bin with the -already cooled- CRs from the previous bin
                    if(ntot > 0 && etot > 0) {slope_gamma = CR_return_slope_from_number_and_energy_in_bin(etot, ntot, E_bin_NtoE, j);} else {slope_gamma=0;} // get the updated slope for this bin

                    ntot_evolved[j] = ntot; // set finalized updated number in bin
                    Ucr[j] = etot; // set finalized updated energy in bin
                    bin_slopes[j] = slope_gamma; // set finalized updated slope [or N, if that's the conserved quantity we evolve]
                } // loop over active bins
            } // loop over loss mode
            t += dt; // update time
        } // loop over charge
    } // loop over time

    if(mode_driftkick==0) {for(k=0;k<N_CR_PARTICLE_BINS;k++) {double fac=1; if(Ucr_i[k]>0 && Ucr[k]>0) {fac=DMAX(1.e-10,DMIN(1.e10,Ucr[k]/Ucr_i[k]));} cell[target].CosmicRayEnergyPred[k] *= fac;}}
    /* update fluxes. these will self-adapt quickly but should feel losses as well, so we do that here */
    int kdir; for(k=0;k<N_CR_PARTICLE_BINS;k++) {double fac=1; if(Ucr_i[k]>0 && Ucr[k]>0) {fac=DMAX(1.e-2,DMIN(1.e2,Ucr[k]/Ucr_i[k]));}
        for(kdir=0;kdir<3;kdir++) {if(mode_driftkick==0) {cell[target].CosmicRayFlux[k][kdir] *= fac; cell[target].CosmicRayFluxPred[k][kdir] *= fac;} else {cell[target].CosmicRayFluxPred[k][kdir] *= fac;}}}
    return; // all done!
}

#endif /* CRFLUID_EVOLVE_SPECTRUM */

#if !defined(CRFLUID_EVOLVE_SPECTRUM)
KOKKOS_INLINE_FUNCTION void CR_cooling_and_losses(int target, double n_elec, double nHcgs, double dtime_cgs, struct particle_data *pp, struct gas_cell_data *cell) {
    if(dtime_cgs <= 0) {return;}
    int k_CRegy; double f_ion = DMAX(DMIN(Get_Gas_Ionized_Fraction(target, pp, cell), 1.), 0.);
    double a_hadronic = 6.37e-16, b_coulomb_ion_per_GeV = 3.09e-16*(n_elec + 0.57*(1.-f_ion))*HYDROGEN_MASSFRAC;
    for(k_CRegy=0; k_CRegy<N_CR_PARTICLE_BINS; k_CRegy++) {
        double CR_coolrate, Z, species_ID; CR_coolrate=0; Z=fabs(return_CRbin_CR_charge_in_e(target,k_CRegy)); species_ID=return_CRbin_CR_species_ID(k_CRegy);
        if(species_ID > 0) {
#if (N_CR_PARTICLE_BINS > 2)
            double E_GeV=return_CRbin_kinetic_energy_in_GeV(target,k_CRegy, cell), beta=return_CRbin_beta_factor(target,k_CRegy, cell);
            CR_coolrate += b_coulomb_ion_per_GeV * ((Z*Z)/(beta*E_GeV)) * nHcgs;
            if(E_GeV>=0.28) {CR_coolrate += a_hadronic * nHcgs;}
#else
            CR_coolrate = (0.87*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * nHcgs;
#endif
        } else {
            double E_GeV=return_CRbin_kinetic_energy_in_GeV(target,k_CRegy, cell), E_rest=0.000511, gamma=1.+E_GeV/E_rest;
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
#else /* CRFLUID_EVOLVE_SPECTRUM is defined — forward to multibin solver */
KOKKOS_INLINE_FUNCTION void CR_cooling_and_losses(int target, double n_elec, double nHcgs, double dtime_cgs, struct particle_data *pp, struct gas_cell_data *cell) {
    CR_cooling_and_losses_multibin(target, n_elec, nHcgs, dtime_cgs, 0, pp, cell);
}
#endif /* !CRFLUID_EVOLVE_SPECTRUM */



/* Atomic accumulation into a CR field of a cell that several sources can deposit
   into at once (the sink and feedback injection paths).  A host OpenMP pragma is
   silently ignored by the device pass, so the two backends are selected here
   explicitly; the device branch needs Kokkos, and a translation unit that lacks
   it fails to compile rather than quietly losing the atomicity. */
template <class T>
KOKKOS_INLINE_FUNCTION void cr_atomic_add(T *ptr, double val)
{
#if (defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__)
    Kokkos::atomic_add(ptr, (T)val);
#else
    #pragma omp atomic
    *ptr += (T)val;
#endif
}

/* Defined below, outside the COSMIC_RAY_FLUID block: it is also used without it. */
KOKKOS_INLINE_FUNCTION double CR_calculate_adiabatic_gasCR_exchange_term(int i, double dt_entr,
    double gamma_minus_eCR_tmp, int mode, struct particle_data *pp, struct gas_cell_data *cell);

/* utility routine which handles the numerically-necessary parts of the CR 'injection' for you; here 'injection_velocity' should be in physical (not comoving) units */
KOKKOS_INLINE_FUNCTION void inject_cosmic_rays(double CR_energy_to_inject, double injection_velocity, int source_type, int target, double *dir, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(CR_energy_to_inject <= 0) {return;}
    double f_injected[N_CR_PARTICLE_BINS]; f_injected[0]=1; int k_CRegy;
#if (N_CR_PARTICLE_BINS > 1) /* add a couple steps to make sure injected energy is always normalized properly! */
    double sum_in=0.0; for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]=CR_energy_spectrum_injection_fraction(k_CRegy,source_type,injection_velocity,0,target, pp, cell); sum_in+=f_injected[k_CRegy];}
    if(sum_in>0.0) {for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]/=sum_in;}} else {for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]=1./N_CR_PARTICLE_BINS;}}
#endif
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        double dEcr = evaluate_cr_transport_reductionfactor(target, k_CRegy, 0, cell) * CR_energy_to_inject * f_injected[k_CRegy]; // normalized properly to sum to unity, and account for RSOL in injection rate [akin to RHD treatment]
        if(dEcr <= 0) {continue;}
#if defined(CRFLUID_EVOLVE_SPECTRUM) // update the evolved slopes with the injection spectrum slope: do a simple energy-weighted mean for the updated/mixed slope here
        double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy), egy_slopemode = 1, xm = All.CR_global_min_rigidity_in_bin[k_CRegy] / All.CR_global_rigidity_at_bin_center[k_CRegy], xp = All.CR_global_max_rigidity_in_bin[k_CRegy] / All.CR_global_rigidity_at_bin_center[k_CRegy], xm_e=xm, xp_e=xp; // values needed for bin injection parameters
        if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {egy_slopemode=2; xm_e=xm*xm; xp_e=xp*xp;} // values needed to scale from slope injected to number and back
        double slope_inj = CR_energy_spectrum_injection_fraction(k_CRegy,source_type,injection_velocity,1,target, pp, cell); // spectral slope of injected CRs
#if defined(CRFLUID_ALT_RSOL_FORM) && defined(CRFLUID_ALT_VARIABLE_RSOL) // want to correct injection slope if we're modulating injection with a variable Psi-type rsol function or variable-rsol
        if(k_CRegy>0) {int spec_0=return_CRbin_CR_species_ID(k_CRegy), spec_m=return_CRbin_CR_species_ID(k_CRegy-1), spec_p=-200; if(spec_m==spec_0) {
            double rfac_0=evaluate_cr_transport_reductionfactor(target,k_CRegy,0, cell), rfac_m=evaluate_cr_transport_reductionfactor(target,k_CRegy-1,0, cell), rfac_p=rfac_0, R_0=All.CR_global_rigidity_at_bin_center[k_CRegy], R_m=All.CR_global_rigidity_at_bin_center[k_CRegy-1], R_p=R_0;
            if(k_CRegy<N_CR_PARTICLE_BINS-1) {spec_p=return_CRbin_CR_species_ID(k_CRegy+1);}
            if(spec_p==spec_0) {rfac_p=evaluate_cr_transport_reductionfactor(target,k_CRegy+1,0, cell); R_p=All.CR_global_rigidity_at_bin_center[k_CRegy+1];
                double xm=log(R_m/R_0),xp=log(R_p/R_0),qm=log(rfac_m/rfac_0),qp=log(rfac_p/rfac_0); slope_inj += (qm*xm + qp*xp) / (xm*xm + xp*xp);} else {slope_inj += log(rfac_0/rfac_m) / log(R_0/R_m);}}} // not clear if actually improves accuracy by substantial margin here, vs letting code self-adjust in next timestep
#endif
        double gamma_one = slope_inj + 1., xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one); // variables below
        double ntot_inj = (dEcr / E_GeV) * ((gamma_one + egy_slopemode) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp_e - xm_gamma_one*xm_e); // injected number in bin
        cr_atomic_add(&cell[target].CosmicRay_Number_in_Bin[k_CRegy], ntot_inj); // simply update injected number. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
#endif
        cr_atomic_add(&cell[target].CosmicRayEnergy[k_CRegy], dEcr); // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        cr_atomic_add(&cell[target].CosmicRayEnergyPred[k_CRegy], dEcr); // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        double dir_mag=0, flux_mag=dEcr * CRFLUID_REDUCED_C_CODE(k_CRegy); Vec3<double> dir_to_use={}; int k;
#ifdef MAGNETIC
        Vec3<double> Bdir=cell[target].BPred;
        double B_dot_dir=0; for(k=0;k<3;k++) {B_dot_dir+=dir[k]*Bdir[k];} // the 'default' direction is projected onto B
        dir_to_use = B_dot_dir * Bdir; // launch -along- B, projected [with sign determined] by the intially-desired direction
#else
        dir_to_use = {dir[0], dir[1], dir[2]}; // launch in the 'default' direction
        dir_mag = dir_to_use.norm_sq();
        if(dir_mag <= 0) {dir_to_use[0]=0; dir_to_use[1]=0; dir_to_use[2]=1; dir_mag=1;}
        for(k=0;k<3;k++) {
            double dflux=flux_mag*dir_to_use[k]/sqrt(dir_mag);
            cr_atomic_add(&cell[target].CosmicRayFlux[k_CRegy][k], dflux); // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
            cr_atomic_add(&cell[target].CosmicRayFluxPred[k_CRegy][k], dflux); // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        }
#endif
    }
    return;
}


/* routine to do the drift/kick operations for CRs: mode=0 is kick, mode=1 is drift */
#if !defined(CRFLUID_EVOLVE_SCATTERINGWAVES)
KOKKOS_INLINE_FUNCTION double CosmicRay_Update_DriftKick(int i, double dt_entr, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(dt_entr <= 0) {return 0;} // no update

    int k_CRegy;
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        int k; double eCR, u0; k=0; if(mode==0) {eCR=cell[i].CosmicRayEnergy[k_CRegy]; u0=cell[i].InternalEnergy;} else {eCR=cell[i].CosmicRayEnergyPred[k_CRegy]; u0=cell[i].InternalEnergyPred;} // initial energy
        if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code
        if(eCR < 0) {eCR=0;} // limit to physical values
        double closure_f1, closure_f2; closure_f1=1, closure_f2=0; // prefactors for below
        double three_chi = return_cosmic_ray_anisotropic_closure_function_threechi(i,k_CRegy, cell); // 3*chi = 3*(1-<mu^2>)/2 closure function //
        closure_f1 = 3.-2.*three_chi; closure_f2 = 1.-three_chi; // prefactors for both terms below //

        // this is the exact solution for the CR flux-update equation over a finite timestep dt: it needs to be solved this way [implicitly] as opposed to explicitly for dt because in the limit of dt_cr_dimless being large, the problem exactly approaches the diffusive solution
        Vec3<double> DtCosmicRayFlux={}, flux={}, CR_veff={}; double CR_vmag=0, q_cr=0, cr_speed=CRFLUID_REDUCED_C_CODE(k_CRegy), rsol_correction_factor=cosmicrayfluid_rsol_corrfac(k_CRegy), V_i=pp[i].Mass/cell[i].Density, P0_cr, fac_for_DtCosmicRayFlux; P0_cr=Get_Gas_CosmicRayPressure(i, k_CRegy, cell);
        cr_speed = DMAX(cell[i].MaxSignalVel , DMIN(CRFLUID_REDUCED_C_CODE(k_CRegy) , 10.*fabs(cell[i].CosmicRayDiffusionCoeff[k_CRegy])/(pp[i].Get_Particle_Size()*All.cf_atime)));
        fac_for_DtCosmicRayFlux = -rsol_correction_factor * fabs(cell[i].CosmicRayDiffusionCoeff[k_CRegy]) * V_i / (GAMMA_COSMICRAY(k_CRegy)-1.);
        DtCosmicRayFlux = cell[i].Gradients.CosmicRayPressure[k_CRegy];
#ifdef MAGNETIC // do projection onto field lines
        Vec3<double> bhat={}, B0={}; double Bmag2=0, Bmag=0, bbGB=0, DtCRDotBhat=0;
        if(mode==0) {B0=cell[i].B/V_i;} else {B0=cell[i].BPred/V_i;}
        DtCRDotBhat = dot(DtCosmicRayFlux, B0); Bmag2 = B0.norm_sq(); bhat = B0;
        if(Bmag2 > 0) {Bmag=sqrt(Bmag2); bhat /= Bmag;}
        bbGB = -dot(bhat, cell[i].Gradients.B.matvec(bhat)) / Bmag;
        DtCosmicRayFlux = fac_for_DtCosmicRayFlux * (closure_f1*DtCRDotBhat*B0/Bmag2 + bhat*(closure_f2*P0_cr*bbGB));
#endif
        double v_Alfven = three_chi * Get_Gas_ion_Alfven_speed_i(i, pp, cell) * return_CRbin_nuplusminus_asymmetry(i,k_CRegy, cell); /* define naive streaming and Alfven speeds */
        double dt_f_m=DtCosmicRayFlux.norm_sq();
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        double flux_diff=sqrt(dt_f_m), flux_stream=fabs(rsol_correction_factor*v_Alfven*(GAMMA_COSMICRAY(k_CRegy)*eCR)); // estimate contribution to flux from both diffusive and streaming components
        double frac_diff=flux_diff/(flux_diff+flux_stream); // fraction of flux from diffusive term
        double alpha_v=0.,alpha_qN=0.,alpha_qE=1.,alpha_nu=-0.6,alpha_L=0.,alpha_f0=CR_return_spectral_slope_target(i,k_CRegy, cell); // values of coefficients: replace hard-coded alpha_nu with lookup to actual function numerically, below
        double xi = All.CR_global_max_rigidity_in_bin[k_CRegy] / All.CR_global_min_rigidity_in_bin[k_CRegy]; // bin width in our units
        int kCR_p=k_CRegy, kCR_m=k_CRegy-1; // want two neighboring bins with same species
        if(k_CRegy<N_CR_PARTICLE_BINS-1) {if(All.CR_species_ID_in_bin[k_CRegy+1]==All.CR_species_ID_in_bin[k_CRegy]) {kCR_m++; kCR_p++;}} // check if can use this and next, or use this and below
        double xi_pm = sqrt((All.CR_global_min_rigidity_in_bin[kCR_p]*All.CR_global_max_rigidity_in_bin[kCR_p])/(All.CR_global_min_rigidity_in_bin[kCR_m]*All.CR_global_max_rigidity_in_bin[kCR_m])); // bin ratio to next bin for numerical derivative (being careful to follow our convention of defining these at the geometric mean)
        double beta_k =return_CRbin_beta_factor(-1,k_CRegy,cell), beta_p=return_CRbin_beta_factor(-1,kCR_p,cell), beta_m=return_CRbin_beta_factor(-1,kCR_m,cell); // get beta factors needed to go between scattering rates and diffusivities
        alpha_nu = log((beta_p*beta_p/cell[i].CosmicRayDiffusionCoeff[kCR_p]) / (beta_m*beta_m/cell[i].CosmicRayDiffusionCoeff[kCR_m])) / log(xi_pm); // numerically calculate the slope of the scattering-rate dependence for any functional form
        if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {alpha_v=1.; alpha_qE=2.;} // correct to non-relativistic values as needed
        if(beta_k<1. && beta_k>0.) {double one_minus_beta2=1.-beta_k*beta_k; alpha_v=one_minus_beta2*one_minus_beta2; alpha_qE=1.+sqrt(one_minus_beta2);} // these are exact in terms of beta, so good approx here using bin-centered beta values
        alpha_L = -0.5*alpha_nu; // this is an approximate model, since usually in steady state we end up with alpha_L roughly following this scaling -- but note that to leading order in the most important quantity here which is the -ratio- of omega_n to omega_e, the alpha_L term factors out //
        double alpha_mu = alpha_v - (alpha_nu + 0*alpha_L); // use value of alpha-mu for diffusive equilibrium, the regime where this term matters [alpha_L term zero'd here because we're taking really the ratio of omega_1 over omega_delta, more like omega_kappa in the reference]
        double flux_n_over_e_factor_approx = 1. + ((alpha_qN-alpha_qE)*(alpha_v+alpha_mu)/12.)*log(xi)*log(xi); // approximate series expansion, should use full expressions here
        double c0_a=1.+DMAX(DMIN(alpha_f0,0.),-6.)-alpha_L, c0_b=c0_a+2.*alpha_v-alpha_nu, c0_c=-alpha_v+0.5*alpha_nu, ln_xi=log(xi), c0_a_e=c0_a+alpha_qE, c0_b_e=c0_b+alpha_qE, c0_a_n=c0_a+alpha_qN, c0_b_n=c0_b+alpha_qN; // define a bunch of the coefficients we'll need
        double omega_k_e = (c0_a_e/c0_b_e) * ((exp(ln_xi*c0_b_e)-1.)/(exp(ln_xi*c0_a_e)-1.)) * exp(ln_xi*c0_c); // this is the exact value for the omega_e term we need here
        double omega_k_n = (c0_a_n/c0_b_n) * ((exp(ln_xi*c0_b_n)-1.)/(exp(ln_xi*c0_a_n)-1.)) * exp(ln_xi*c0_c); // this is the exact value for the omega_n term we need here
        if(omega_k_e>0.1 && omega_k_e<2. && isfinite(omega_k_e)) {DtCosmicRayFlux *= omega_k_e;} // correct the energy flux (what we evolve by default) by its omega [this absolute correction is less important than the relative correction below, but since we have it, let's use it]
        double flux_n_over_e_factor = omega_k_n / omega_k_e; // exact value
        if((flux_n_over_e_factor<0) || (!isfinite(flux_n_over_e_factor))) {flux_n_over_e_factor = flux_n_over_e_factor_approx;}
        double flux_n_over_e_factor_modulated = 1. + (flux_n_over_e_factor-1.) * frac_diff;
        cell[i].Flux_Number_to_Energy_Correction_Factor[k_CRegy] = DMAX(DMIN(flux_n_over_e_factor_modulated, 2.0), 0.5); // equilibrium streaming solution is alpha_mu->-alpha_v such that bin-centered is exact, so mean correction applies only to flux 'portion' of this
#endif
        if(dt_f_m>0) {DtCosmicRayFlux *= (1.0 + rsol_correction_factor * v_Alfven * (GAMMA_COSMICRAY(k_CRegy) * eCR) / sqrt(dt_f_m));} // (tilde[c]/c) * v_a * (ecr+Pcr), in same direction as gradient wants to 'push' naturally [natural direction of F]

        if(mode==0) {flux=cell[i].CosmicRayFlux[k_CRegy];} else {flux=cell[i].CosmicRayFluxPred[k_CRegy];}
#ifdef MAGNETIC // do projection onto field lines
        double fluxmag=flux.norm_sq(), fluxdot=dot(flux, B0);
        if(fluxmag>0) {fluxmag=sqrt(fluxmag);} else {fluxmag=0;}
        if(fluxdot<0) {fluxmag*=-1;} // points down-field
        // before acting on the 'stiff' sub-system, account for the 'extra' advection term that accounts for 'twisting' of B: note more careful derivation shows this is sub-leading order in v/c, should not be included here
        //double fac_bv=0; for(k=0;k<3;k++) {fac_bv += All.cf_a2inv * bhat[k] * (bhat[0]*cell[i].Gradients.Velocity[k][0] + bhat[1]*cell[i].Gradients.Velocity[k][1] + bhat[2]*cell[i].Gradients.Velocity[k][2]);}
        //if(All.ComovingIntegrationOn) {fac_bv += All.cf_hubble_a;} // adds cosmological/hubble flow term here [not included in peculiar velocity gradient]
        //fluxmag *= exp(-DMAX(-2.,DMIN(2.,rsol_correction_factor*fac_bv*dt_entr))); // limit factor for change here, should be small given Courant factor, then update flux term accordingly, before next step -- acts like a mod of the divv term //
        if(Bmag2>0) {flux = (fluxmag / sqrt(Bmag2)) * B0;} // re-assign to be along field
#endif
        int target_for_CR_beta_factor = i; // if this =1, use energy-weighted mean value in bin for CR beta, otherwise if =-1, use median point of bin
        target_for_CR_beta_factor = -1;
        double beta_fac = return_CRbin_beta_factor(target_for_CR_beta_factor,k_CRegy,cell); // velocity beta, to account for non-relativistic CRs
        double dt_cr_dimless = dt_entr * beta_fac*beta_fac * cr_speed*cr_speed * (1./3.) / (MIN_REAL_NUMBER + fabs(cell[i].CosmicRayDiffusionCoeff[k_CRegy] * rsol_correction_factor));
        dt_cr_dimless = DMIN(dt_cr_dimless , 0.1); // arbitrary limiter here for some additional numerical stability
        if((dt_cr_dimless > 0)&&(dt_cr_dimless < 20.)) {q_cr = exp(-dt_cr_dimless);} // factor for CR interpolation
        flux = q_cr*flux + (1.-q_cr)*DtCosmicRayFlux; // updated flux
        CR_veff = flux/(eCR+MIN_REAL_NUMBER); CR_vmag = CR_veff.norm_sq(); // effective streaming speed
        if((CR_vmag <= 0) || (isnan(CR_vmag))) // check for valid numbers
        {
            flux = {}; CR_veff = {}; // zero if invalid
        } else {
            double CR_vmax = CRFLUID_REDUCED_C_CODE(k_CRegy); // enforce a hard upper limit here, though shouldn't be needed with modern formulation
            CR_vmag = sqrt(CR_vmag); if(CR_vmag > CR_vmax) {flux *= CR_vmax/CR_vmag; CR_veff *= CR_vmax/CR_vmag;} // limit flux to free-streaming speed [as with RT]
        }
        if(mode==0) {cell[i].CosmicRayFlux[k_CRegy]=flux;} else {cell[i].CosmicRayFluxPred[k_CRegy]=flux;}

        /* update scalar CR energy. first update the CR energies from fluxes. since this is positive-definite, some additional care is needed */
        double dCR_dt = cell[i].DtCosmicRayEnergy[k_CRegy], eCR_tmp = eCR;
        double dCR = dCR_dt*dt_entr, dCRmax = 1.e10*(eCR_tmp+MIN_REAL_NUMBER);
#if defined(GALSF)
        dCRmax = DMAX(2.0*eCR_tmp , 0.1*u0*pp[i].Mass);
#endif
        if(dCR > dCRmax) {dCR=dCRmax;} // don't allow excessively large values
        if(dCR < -eCR_tmp) {dCR=-eCR_tmp;} // don't allow it to go negative
        double eCR_0, eCR_00; eCR_00 = eCR_tmp; eCR_tmp += dCR; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy]=eCR_tmp;} else {cell[i].CosmicRayEnergyPred[k_CRegy]=eCR_tmp;} // updated energy
        eCR_0 = eCR_tmp; // save this value for below

#if defined(CRFLUID_EVOLVE_SPECTRUM)
        // add update for CR number if evolved explicitly //
        if(mode==0) // only update on kicks, since we worth with a drift-conserved slope determining the ratio of N and E
        {
            double dN = cell[i].DtCosmicRay_Number_in_Bin[k_CRegy]*dt_entr, n0 = cell[i].CosmicRay_Number_in_Bin[k_CRegy], n_new = n0+dN;
            double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy), xm = All.CR_global_min_rigidity_in_bin[k] / All.CR_global_rigidity_at_bin_center[k], xp = All.CR_global_max_rigidity_in_bin[k] / All.CR_global_rigidity_at_bin_center[k], xm_e=xm, xp_e=xp;
            if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {xm_e = xm*xm; xp_e = xp*xp;} // extra power of p in energy equation accounted for here, all that's needed
            double N_min = eCR_tmp / (E_GeV * xp_e * (1.-1.e-4)); // even with arbitrarily large slopes we cannot exceed this limit: all CRs 'piled up' at highest energy
            double N_max = eCR_tmp / (E_GeV * xm_e * (1.+1.e-4)); // even with arbitrarily large slopes we cannot exceed this limit: all CRs 'piled up' at lowest energy
            n_new = DMIN(DMAX(n_new,N_min),N_max); if((n_new<0) || (isnan(n_new))) {n_new=0;}
            cell[i].CosmicRay_Number_in_Bin[k_CRegy] = n_new; // alright, updated CR number for evolution equations
        }
#endif

#if defined(COOLING_OPERATOR_SPLIT)
        /* now need to account for the adiabatic heating/cooling of the 'fluid', here, with gamma=GAMMA_COSMICRAY(k_CRegy) */
        double dCR_div = CR_calculate_adiabatic_gasCR_exchange_term(i, dt_entr, (GAMMA_COSMICRAY(k_CRegy)-1.)*eCR_tmp, mode, pp, cell); // this will handle the update below - separate subroutine b/c we want to allow it to appear in a couple different places
        double uf = DMAX(u0 - dCR_div/pp[i].Mass , All.MinEgySpec); // final updated value of internal energy per above
        if(mode==0) {cell[i].InternalEnergy = uf;} else {cell[i].InternalEnergyPred = uf;} // update gas
#if !defined(CRFLUID_EVOLVE_SPECTRUM)
        if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy] += dCR_div;} else {cell[i].CosmicRayEnergyPred[k_CRegy] += dCR_div;} // update CRs: note if explicitly evolving spectrum, this is done separately below //
#endif
#endif

    } // loop over CR bins complete

#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    if(cell[i].DtCREgyNewInjectionFromShocks > 0) /* now perform the actual CR injection using the rates estimated in the hydro solver */
    {
        Vec3<double> dir = -cell[i].Gradients.Pressure; /* initial flux direction down pressure gradient */
        inject_cosmic_rays(cell[i].DtCREgyNewInjectionFromShocks * dt_entr, 1000./UNIT_VEL_IN_KMS, 2, i, &dir[0], pp, cell); /* inject the energy */
        cell[i].DtCREgyNewInjectionFromShocks = 0; // reset to nil, we've successfully injected the energy
    }
#endif
#if defined(SINK_CR_INJECTION_AT_TERMINATION)
    if(cell[i].Sink_CR_Energy_Available_For_Injection > 0) {
        /* need to determine whether or not sufficient deceleration has occurred in order to inject CRs from our 'reservoir */
        double vmag = (pp[i].Vel / All.cf_atime).norm_sq(); int k; /* we will base this on a simple estimate of the velocity and how much things have decelerated */
        if(vmag>0) {vmag=sqrt(vmag);}
        double v_outflow_fast_forinjection = All.Sink_outflow_velocity;
#ifdef SINK_TEST_WIND_MIXED_FASTSLOW
        v_outflow_fast_forinjection = (SINK_TEST_WIND_MIXED_FASTSLOW)/UNIT_VEL_IN_KMS;
#endif
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
        v_outflow_fast_forinjection = 0.05 * C_LIGHT_CODE;
#endif
        if((pp[i].ID != All.SpawnedWindCellID) || (vmag < ((double)(SINK_CR_INJECTION_AT_TERMINATION))*v_outflow_fast_forinjection)) {
            Vec3<double> dir = -cell[i].Gradients.Pressure; /* initial flux direction down pressure gradient */
            inject_cosmic_rays(cell[i].Sink_CR_Energy_Available_For_Injection, v_outflow_fast_forinjection, 5, i, &dir[0], pp, cell); /* inject the energy */
            cell[i].Sink_CR_Energy_Available_For_Injection = 0;  // reset its value to nil, now that it has been injected
        }
    }
#endif

    return 1;
}
#endif


#if defined(CRFLUID_EVOLVE_SCATTERINGWAVES)

/*! To Do: with new RSOL scheme, needs some RSOL factors more carefully placed, here.
    generally can be updated to be a bit more flexible
    to handle newer damping rate estimates more modularly, and to deal with extinsic turbulence as well
    (which just appears as a source term in the e_A equations) */

/* routine to do the drift/kick operations for CRs: mode=0 is kick, mode=1 is drift */
KOKKOS_INLINE_FUNCTION double CosmicRay_Update_DriftKick(int i, double dt_entr, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    int k_CRegy;
    if(dt_entr <= 0) {return 0;} // no update
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {

    int k; double eCR, u0; k=0; if(mode==0) {eCR=cell[i].CosmicRayEnergy[k_CRegy]; u0=cell[i].InternalEnergy;} else {eCR=cell[i].CosmicRayEnergyPred[k_CRegy]; u0=cell[i].InternalEnergyPred;} // initial energy
    if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code
    if(eCR < 0) {eCR=0;} // limit to physical values

    // now update all scalar fields (CR energies and Alfvenic energies, if those are followed) from fluxes and adiabatic terms //
    int q_whichupdate, q_N_updates = 3; // update the Alfvenic energy terms from (0) advection with gas [already solved], (1) their fluxes, and (2) their adiabatic terms. this should be basically identical to the CR term.
    for(q_whichupdate=0; q_whichupdate<q_N_updates; q_whichupdate++)
    {
        // first update the CR energies from fluxes. since this is positive-definite, some additional care is needed //
        double dCR_dt = cell[i].DtCosmicRayEnergy[k_CRegy], gamma_eff = GAMMA_COSMICRAY(k_CRegy), eCR_tmp = eCR;
        if(q_whichupdate>0) {dCR_dt=cell[i].DtCosmicRayAlfvenEnergy[k_CRegy][q_whichupdate-1]; gamma_eff=(3./2.); if(mode==0) {eCR_tmp=cell[i].CosmicRayAlfvenEnergy[k_CRegy][q_whichupdate-1];} else {eCR_tmp=cell[i].CosmicRayAlfvenEnergyPred[k_CRegy][q_whichupdate-1];}}
        double dCR = dCR_dt*dt_entr, dCRmax = 1.e10*(eCR_tmp+MIN_REAL_NUMBER);
        if(dCR > dCRmax) {dCR=dCRmax;} // don't allow excessively large values
        if(dCR < -eCR_tmp) {dCR=-eCR_tmp;} // don't allow it to go negative
	    double eCR_00 = eCR_tmp; eCR_tmp += dCR; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        if(q_whichupdate==0) {if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy]=eCR_tmp;} else {cell[i].CosmicRayEnergyPred[k_CRegy]=eCR_tmp;}} // updated energy
        if(q_whichupdate>0) {if(mode==0) {cell[i].CosmicRayAlfvenEnergy[k_CRegy][q_whichupdate-1]=eCR_tmp;} else {cell[i].CosmicRayAlfvenEnergyPred[k_CRegy][q_whichupdate-1]=eCR_tmp;}} // updated energy
        // now need to account for the adiabatic heating/cooling of the 'fluid', here, with gamma=gamma_eff //
        double eCR_0 = eCR_tmp, d_div = (-(gamma_eff-1.) * cell[i].Face_DivVel_ForAdOps*All.cf_a2inv) * dt_entr;
        if(All.ComovingIntegrationOn) {d_div += (-3.*(gamma_eff-1.) * All.cf_hubble_a) * dt_entr;} /* adiabatic term from Hubble expansion (needed for cosmological integrations */
        double dCR_div = DMIN(eCR_tmp*d_div , 0.5*u0*pp[i].Mass); // limit so don't take away all the gas internal energy [to negative values]
        if(dCR_div + eCR_tmp < 0) {dCR_div = -eCR_tmp;} // check against energy going negative
        eCR_tmp += dCR_div; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        dCR_div = eCR_tmp - eCR_0; // actual change that is going to be applied
        if(dCR_div < -0.5*pp[i].Mass*u0) {dCR_div=-0.5*pp[i].Mass*u0;} // before re-coupling, ensure this will not cause negative energies
        if(dCR_div < -0.9*eCR_00) {dCR_div=-0.9*eCR_00;} // before re-coupling, ensure this will not cause negative energies
        if(q_whichupdate==0) {if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy] += dCR_div; cell[i].InternalEnergy -= dCR_div/pp[i].Mass;} else {cell[i].CosmicRayEnergyPred[k_CRegy] += dCR_div; cell[i].InternalEnergyPred -= dCR_div/pp[i].Mass;}}
        if(q_whichupdate>0) {if(mode==0) {cell[i].CosmicRayAlfvenEnergy[k_CRegy][q_whichupdate-1] += dCR_div; cell[i].InternalEnergy -= dCR_div/pp[i].Mass;} else {cell[i].CosmicRayAlfvenEnergyPred[k_CRegy][q_whichupdate-1] += dCR_div; cell[i].InternalEnergyPred -= dCR_div/pp[i].Mass;}}
    }

    int target_bin_centering_for_CR_quantities = i; // if this = i, evaluate quantities like R_GV at the CR-energy weighted mean of the bin, if =-1, evaluate them at the bin center instead: important for some subtle effects especially if using numerical derivatives for correction terms
    double E_CRs_Gev=return_CRbin_CR_rigidity_in_GV(target_bin_centering_for_CR_quantities,k_CRegy,cell), Z_charge_CR=fabs(return_CRbin_CR_charge_in_e(i,k_CRegy)), M_cr_mp=return_CRbin_CRmass_in_mp(i,k_CRegy); // charge and energy and resonant Alfven wavenumber (in gyro units) of the CR population we're evolving

    // ok, the updates from [0] advection w gas, [1] fluxes, [2] adiabatic, [-] catastrophic (in cooling.c) are all set, just need exchange terms b/t CR and Alfven //
    double EPSILON_SMALL = 1.e-77; // want a very small number here
    Vec3<double> bhat, flux; double Bmag=0, Bmag_Gauss, clight_code=C_LIGHT_CODE, Omega_gyro, eA[2], vA_code, vA2_c2, E_B, fac, flux_G, fac_Omega, f_CR, f_CR_dot_B, cs_thermal, r_turb_driving, G_ion_neutral=0, G_turb_plus_linear_landau=0, G_nonlinear_landau_prefix=0;
    double ne=1, f_ion=1, nh0=0, nhp=0, temperature, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, rho_cgs=rho*UNIT_DENSITY_IN_CGS;
#ifdef COOLING
    temperature = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, pp, cell); // get thermodynamic properties
    f_ion = DMIN(DMAX(DMAX(DMAX(1-nh0, nhp), ne/1.2), 1.e-8), 1.); // account for different measures above (assuming primordial composition)
#endif
    for(k=0;k<3;k++) {bhat[k] = (mode==0) ? cell[i].B[k] : cell[i].BPred[k];} // grab whichever B field we need for our mode
    if(mode==0) {eCR=cell[i].CosmicRayEnergy[k_CRegy]; u0=cell[i].InternalEnergy;} else {eCR=cell[i].CosmicRayEnergyPred[k_CRegy]; u0=cell[i].InternalEnergyPred;} // initial energy
    if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforce the usual minimum thermal energy the code requires
    for(k=0;k<2;k++) {if(mode==0) {eA[k]=cell[i].CosmicRayAlfvenEnergy[k_CRegy][k];} else {eA[k]=cell[i].CosmicRayAlfvenEnergyPred[k_CRegy][k];}} // Alfven energy
    for(k=0;k<3;k++) {flux[k] = (mode==0) ? cell[i].CosmicRayFlux[k_CRegy][k] : cell[i].CosmicRayFluxPred[k_CRegy][k];} // load flux
    f_CR = flux.norm(); f_CR_dot_B = dot(bhat, flux); // compute the magnitude of the flux density
    if(f_CR_dot_B<0) {f_CR*=-1;} // initialize the flux density variable from the previous timestep, appropriately signed with respect to the b-field
    Bmag = bhat.norm(); // compute magnitude
    bhat /= (EPSILON_SMALL+Bmag); // now it's bhat we have here
    Bmag *= cell[i].Density/pp[i].Mass * All.cf_a2inv; // convert to actual B in physical units
    E_B = 0.5*Bmag*Bmag * (pp[i].Mass/(cell[i].Density*All.cf_a3inv)); // B-field energy (energy density times volume, for ratios with energies above)
    double Eth_0 = EPSILON_SMALL + 1.e-8 * pp[i].Mass*u0; // set minimum magnetic energy relative to thermal (maximum plasma beta ~ 1e8) to prevent nasty divergences
    if(E_B < Eth_0) {Bmag = sqrt(2.*Eth_0/((pp[i].Mass/(cell[i].Density*All.cf_a3inv))));} // enforce this maximum beta for purposes of "B" to insert below
    E_B = 0.5*Bmag*Bmag * (pp[i].Mass/(cell[i].Density*All.cf_a3inv)); // B-field energy (energy density times volume, for ratios with energies above)
    Bmag_Gauss = Bmag * UNIT_B_IN_GAUSS; // turn it into Gauss
    Omega_gyro = (8987.34 * Bmag_Gauss * (Z_charge_CR/E_CRs_Gev)) * UNIT_TIME_IN_CGS; // gyro frequency of the CR population we're evolving, converted to physical code units //
    double vA_noion = cell[i].Alfven_speed(); // Alfven speed in code units [recall B units such that there is no 4pi here]
    vA_code = Get_Gas_ion_Alfven_speed_i(i, pp, cell); // include ionization appropriately for small-scale modes
    cs_thermal = sqrt(cell[i].soundspeed2_from_u(u0)); // thermal sound speed at appropriate drift-time [in code units, physical]
    vA2_c2 = vA_code*vA_code / (clight_code*clight_code); // Alfven speed vs speed of light
    fac_Omega = (3.*M_PI/16.) * Omega_gyro * (1.+2.*vA2_c2); // factor which will be used heavily below
    /* for turbulent (anisotropic and linear landau) damping terms: need to know the turbulent driving scale: assume a cascade with a driving length equal to the pressure gradient scale length */
    r_turb_driving = cell[i].Gradients.Pressure.norm_sq(); // compute gradient magnitude
    r_turb_driving = DMAX( cell[i].Pressure / (EPSILON_SMALL + sqrt(r_turb_driving)) , pp[i].Get_Particle_Size() ) * All.cf_atime; // maximum of gradient scale length or resolution scale
    double k_turb = 1./r_turb_driving, k_L = Omega_gyro / clight_code;

    // before acting on the 'stiff' sub-system, account for the 'extra' advection term that accounts for 'twisting' of B:
    fac=0; for(k=0;k<3;k++) {fac += All.cf_a2inv * bhat[k] * (bhat[0]*cell[i].Gradients.Velocity[k][0] + bhat[1]*cell[i].Gradients.Velocity[k][1] + bhat[2]*cell[i].Gradients.Velocity[k][2]);}
    if(All.ComovingIntegrationOn) {fac += All.cf_hubble_a;} // adds cosmological/hubble flow term here [not included in peculiar velocity gradient]
    fac *= -dt_entr; if(!isfinite(fac)) {fac=0;} else {if(fac>2.) {fac=2.;} else {if(fac<-2.) {fac=-2.;}}} // limit factor for change here, should be small given Courant factor
    f_CR *= exp(fac); // update flux term accordingly, before next step //

    // because the equations below will very much try to take things to far-too-small values for numerical precision, we need to define a bunch of sensible bounds for values to allow, to prevent divergences, but also enforce conservation
    // calculate minimum eA,eCR to enforce; needed because if eA is identically zero, nothing can get amplified, and it will always be zero. but for large enough seed to amplify, results should not depend on seed //
    eA[0]=DMAX(eA[0],0); eA[1]=DMAX(eA[1],0); eCR=DMAX(eCR,0); // enforce non-negative energies
    double Min_Egy=0, e_tot=0, e_tot_new=0, fmax=0; e_tot = eCR + eA[0] + eA[1] + EPSILON_SMALL; // sum total energy, enforce positive-definite: will use this to ensure total energy conservation when enforcing minima below
    {
        double h=pp[i].Get_Particle_Size()*All.cf_atime; int k2; for(k=0;k<3;k++) {for(k2=0;k2<3;k2++) {Min_Egy+=cell[i].Gradients.B[k][k2]*cell[i].Gradients.B[k][k2];}}
        Min_Egy=h*sqrt(Min_Egy/9.)*All.cf_a2inv; Min_Egy=DMIN(Min_Egy,Bmag); r_turb_driving=DMAX(h,r_turb_driving); Min_Egy=DMIN(Min_Egy,Bmag*pow(h/r_turb_driving,1./3.)); Min_Egy=Min_Egy*pow(DMIN(clight_code/Omega_gyro,DMIN(h,r_turb_driving))/h,1./3.); // Min_Egy is now magnetic field extrap to r_gyro
        Min_Egy = 0.5 * (Min_Egy*Min_Egy) * pp[i].Mass/(cell[i].Density*All.cf_a3inv); // magnetic energy at this scale, from the above //
        double epsilon = 1.e-15; Min_Egy *= epsilon; // minimum energy is a tiny fraction of B at the dissipation scale
        if(Min_Egy <= 0 || !isfinite(Min_Egy)) {Min_Egy = 1.e-15*eCR;} // if this minimum-energy calculation failed, enforce a tiny fraction of the CR energy
        if(Min_Egy <= 0 || !isfinite(Min_Egy)) {Min_Egy = 1.e-15*pp[i].Mass*u0;} // if this minimum-energy calculation failed, enforce a tiny fraction of the thermal energy
        if(Min_Egy <= 0) {Min_Egy = EPSILON_SMALL;} // if this still failed, simply enforce a tiny positive-definite value
    }
    eCR=DMAX(eCR,Min_Egy); eA[0]=DMAX(eA[0],Min_Egy); eA[1]=DMAX(eA[1],Min_Egy); // enforce

    // ok, now all the advection and adiabatic operations should be complete. they are split above.
    //  what remains is the stiff, coupled subsystem of wave growth+damping, which needs to be treated
    //  more carefully or else we get very large over/under-shoots

    // first define some convenient units and dimensionless quantities, and enforce limits on values of input quantities
    double cr_speed = CRFLUID_REDUCED_C_CODE(k_CRegy);
    double eCR_0 = 1.e-6*(E_B + pp[i].Mass*u0) + eCR + eA[0] + eA[1] + fabs(f_CR/cr_speed); // this can be anything, just need a normalization for the characteristic energy scale of the problem //
    double ceff2_va2=(cr_speed*cr_speed)/(vA_code*vA_code), t0=1./(fac_Omega*(eCR_0/E_B)*vA2_c2), gammCR=GAMMA_COSMICRAY(k_CRegy), f_unit=vA_code*eCR_0, volume=pp[i].Mass/(cell[i].Density*All.cf_a3inv); // factors used below , and for units
    double x_e=eCR/eCR_0, x_f=f_CR/f_unit, x_up=eA[0]/eCR_0, x_um=eA[1]/eCR_0, dtau=dt_entr/t0; e_tot/=eCR_0; Min_Egy/=eCR_0; // initial values in relevant units
    Min_Egy=DMAX(DMIN(Min_Egy,DMIN(x_e,DMIN(x_up,x_um))),EPSILON_SMALL); if(!isfinite(Min_Egy)) {Min_Egy=EPSILON_SMALL;} // enforce positive-definite-ness
    // we can more robustly define a minimum and maximum e_A by reference to a minimum and maximum 'effective diffusivity' over which it is physically meaningful, and numerically possible to evolve them
    double ref_diffusivity = 4.4e26 / (UNIT_VEL_IN_CGS * UNIT_LENGTH_IN_CGS); // define a unit diffusivity in code units for reference below
    double xkappa_min = DMAX(vA_code*vA_code*t0/(3.e8*ref_diffusivity) , EPSILON_SMALL); // maximum diffusivity ~1e35, but be non-zero
    double xkappa_max = DMAX(DMIN(vA_code*vA_code*t0/(3.e-8*ref_diffusivity) , 0.5*E_B/eCR_0), xkappa_min); // minimum diffusivity at ~1e19, but cannot have more energy in eAp+eAm than total magnetic energy! (equations below assume -small- fraction of E_B in eA!, or growth rates non-linearly modified)
    if(e_tot < Min_Egy || !isfinite(e_tot)) {e_tot = Min_Egy;} // enforce minima/maxima
    if(x_e   < Min_Egy || !isfinite(x_e)  ) {x_e   = Min_Egy;} // enforce minima/maxima
    if(x_um<EPSILON_SMALL || !isfinite(x_um)) {x_um=EPSILON_SMALL;} else {if(x_um>xkappa_max) {x_um=xkappa_max;}} // enforce minima/maxima
    if(x_up<EPSILON_SMALL || !isfinite(x_up)) {x_up=EPSILON_SMALL;} else {if(x_up>xkappa_max) {x_up=xkappa_max;}} // enforce minima/maxima
    if(x_um+x_up<xkappa_min) {fac=xkappa_min/(x_um+x_up); x_um*=fac; x_up*=fac;} // only want to enforce -sum- having effective diffusivity, not both
    e_tot_new=x_e+x_um+x_up; x_e*=e_tot/e_tot_new; x_up*=e_tot/e_tot_new; x_um*=e_tot/e_tot_new; // check energy after limit-enforcement
    fmax = x_e*sqrt(ceff2_va2); if(!isfinite(x_f)) {x_f=0;} else {if(x_f>fmax) {x_f=fmax;} else {if(x_f<-fmax) {x_f=-fmax;}}} // check for flux maximum/minimum

    // calculate the dimensionless flux source term for the stiff part of the equations
    flux_G = dot(bhat, cell[i].Gradients.CosmicRayPressure[k_CRegy]); // b.gradient[P] -- flux source term
    double psifac = flux_G * (vA_code*t0) / (eCR/volume); // this gives the strength of the gradient source term, should remain fixed over stiff part of loop

    // calculate the wave-damping rates (again in appropriate dimensionless units)
    /* ion-neutral damping: need thermodynamic information (neutral fractions, etc) to compute self-consistently */
    G_ion_neutral = (5.77e-11 * (rho_cgs/PROTONMASS_CGS) * nh0 * sqrt(temperature)) * UNIT_TIME_IN_CGS / sqrt(M_cr_mp); // need to get thermodynamic quantities [neutral fraction, temperature in Kelvin] to compute here -- // G_ion_neutral = (xiH + xiHe); // xiH = nH * siH * sqrt[(32/9pi) *kB*T*mH/(mi*(mi+mH))]. converted to -physical- code units

    int i1,i2; double v2_t=0,dv2_t=0,b2_t=0,db2_t=0,x_LL,M_A,h0,fturb_multiplier=1; // factor which will represent which cascade model we are going to use
    for(i1=0;i1<3;i1++)
    {
        v2_t += cell[i].VelPred[i1]*cell[i].VelPred[i1];
        for(i2=0;i2<3;i2++) {dv2_t += cell[i].Gradients.Velocity[i1][i2]*cell[i].Gradients.Velocity[i1][i2]; db2_t += cell[i].Gradients.B[i1][i2]*cell[i].Gradients.B[i1][i2];}
    }
    b2_t = cell[i].Bfield().norm_sq();
    v2_t=sqrt(v2_t); b2_t=sqrt(b2_t); dv2_t=sqrt(dv2_t); db2_t=sqrt(db2_t); dv2_t/=All.cf_atime; db2_t/=All.cf_atime; b2_t*=All.cf_a2inv; db2_t*=All.cf_a2inv; v2_t/=All.cf_atime; dv2_t/=All.cf_atime; h0=pp[i].Get_Particle_Size()*All.cf_atime; // physical units
    M_A = h0*(EPSILON_SMALL + dv2_t) / (EPSILON_SMALL + vA_noion); M_A = DMAX(M_A , h0*(EPSILON_SMALL + db2_t) / (EPSILON_SMALL + b2_t)); M_A = DMAX( EPSILON_SMALL , M_A ); // proper calculation of the local Alfven Mach number
    x_LL = clight_code / (Omega_gyro * h0); x_LL=DMAX(x_LL,EPSILON_SMALL); k_turb = 1./h0; // scale at which turbulence is being measured here //
    fturb_multiplier = pow(M_A,3./2.); // corrects to Alfven scale, for correct estimate according to Farmer and Goldreich, Lazarian, etc.
    if(M_A<1.) {fturb_multiplier*=DMIN(sqrt(M_A),pow(M_A,7./6.)/pow(x_LL,1./6.));} else {fturb_multiplier*=DMIN(1.,1./(pow(M_A,1./2.)*pow(x_LL,1./6.)));} /* Lazarian+ 2016 multi-part model for where the resolved scales lie on the cascade */
    G_turb_plus_linear_landau = (vA_noion + sqrt(M_PI/16.)*cs_thermal) * sqrt(k_turb*k_L) * fturb_multiplier; // linear Landau + turbulent (both have same form, assume k_turb from cascade above)

    G_nonlinear_landau_prefix = (sqrt(M_PI)/8.) * (1./E_B) * (cs_thermal*k_L); // non-linear Landau damping (will be multiplied by eA)
    double gamma_in_t_ll = (G_ion_neutral + G_turb_plus_linear_landau) * t0; // dimensionless now and appropriate code units
    double gamma_nll = G_nonlinear_landau_prefix * eCR_0 * t0; // dimensionless now and appropriate code units


    // now we are ready to actually integrate these equations, in a numerically-stable manner, with protection from over/under-shooting
    double dtau_max = 1.e-5;
    double dx_e,dx_f,dx_up,dx_um,x_e_0=x_e,x_f_0=x_f,x_up_0=x_up,x_um_0=x_um,dtaux=0.,efmax=50.,expfac;
    double x_e_prev,x_f_prev,x_up_prev,x_um_prev,n_eqm_loops=1.; fmax=1./EPSILON_SMALL; // (need to set initial fmax to large value)
    long n_iter=0, n_iter_max=100000; // sets the maximum number of sub-cycles which we will allow below for any sub-process
    while(1)
    {
        /* here's the actual set of remaining stiff equations to be solved
            dx_e  = gammCR*(x_up+x_um)*x_e + (x_um-x_up)*x_f;                     // deCR_dt
            dx_f  = -ceff2_va2*(psifac + (x_um-x_up)*x_e + (x_up+x_um)*x_f);      // df_dt
            dx_up = -x_up*(gammCR*x_e + gamma_in_t_ll + gamma_nll*x_up - x_f);    // deAp_dt
            dx_um = -x_um*(gammCR*x_e + gamma_in_t_ll + gamma_nll*x_um + x_f);    // deAm_dt
        */
        x_e_prev=x_e; x_f_prev=x_f; x_up_prev=x_up; x_um_prev=x_um; // reset values at the beginning of the loop (these will be cycled multiple times below)

        // for eqm: if psi>0, f<0, um->grows, up->pure-damping //
        double f_eqm, up_eqm, um_eqm, tinv_u, tinv_f;
        double q_tmp = (gammCR-1.)*x_e + gamma_in_t_ll, q_inner = (4.*gamma_nll*fabs(psifac)) / (q_tmp*q_tmp);
        if(q_inner < 1.e-4) {q_inner=q_inner/2.;} else {q_inner=sqrt(1.+q_inner)-1.;}
        double x_nonzero = q_tmp * q_inner / (2.*gamma_nll); if(fabs(gamma_nll) < EPSILON_SMALL) {x_nonzero = fabs(psifac)/q_tmp;}
        double x_f_magnitude = x_e + fabs(psifac) / x_nonzero;
        if(psifac > 0)
        {
            up_eqm=xkappa_min; um_eqm=x_nonzero; f_eqm=-x_f_magnitude;
            tinv_u = EPSILON_SMALL + fabs(gammCR*x_e + gamma_in_t_ll + gamma_nll*x_um + x_f);
        } else {
            um_eqm=xkappa_min; up_eqm=x_nonzero; f_eqm=+x_f_magnitude;
            tinv_u = EPSILON_SMALL + fabs(gammCR*x_e + gamma_in_t_ll + gamma_nll*x_up - x_f);
        }
        tinv_f = fabs( ceff2_va2*(psifac + (x_um-x_up)*x_e + (x_up+x_um)*x_f) ) * (1./(EPSILON_SMALL + fabs(f_eqm)) + 1./(EPSILON_SMALL+fabs(x_f)));
        double t_eqm = 1./(tinv_u + tinv_f); // timescale to approach equilibrium solution

        // set timestep (steadily  growing from initial conservative value ) //
        dtaux = dtau; if(dtaux > dtau) {dtaux=dtau;}
        if(dtaux > dtau_max) {dtaux = dtau_max;}
        dtau_max *= 2.; if(dtaux > 10.) {dtaux=10.;}

        double jump_fac = 0.5; // fraction towards equilibrium to 'jump' each time
        //if(dtaux >= 0.33*jump_fac*t_eqm)
        if(dtau >= jump_fac*t_eqm)
        {
            // timestep is larger than the timescale to approach the equilibrium solution,
            //  so move the systems towards equilibrium, strictly
            //
            if(dtaux > jump_fac*t_eqm) {dtaux = jump_fac*t_eqm;} else {jump_fac = dtaux/t_eqm;} // initial 'step' is small fraction of equilibrium
            dtaux = n_eqm_loops * t_eqm; jump_fac = 1. + (jump_fac-1.)/n_eqm_loops; n_eqm_loops*=1.1; // each sub-cycle consecutively in eqm, allow longer step
            if((x_f<=-fmax && f_eqm<=-fmax) || (x_f>=+fmax && f_eqm>=+fmax)) {t_eqm=1./EPSILON_SMALL; jump_fac=1.; dtaux=dtau;} // if slamming into limits, terminate cycle with big step
            if(f_eqm > 0)
            {
                x_up = exp( log(x_up)*(1.-jump_fac) + log(up_eqm)*jump_fac );
                if(x_f > 0) {x_f = +exp( log(fabs(x_f))*(1.-jump_fac) + log(fabs(f_eqm))*jump_fac );} else {
                    if(fabs(x_f)<10.*fabs(f_eqm)) {x_f=x_f*(1.-jump_fac)+f_eqm*jump_fac;} else {
                        x_f = -exp( log(fabs(x_f))*(1.-jump_fac) + log(fabs(f_eqm))*jump_fac );}}
            } else {
                x_um = exp( log(x_um)*(1.-jump_fac) + log(um_eqm)*jump_fac );
                if(x_f < 0) {x_f = -exp( log(fabs(x_f))*(1.-jump_fac) + log(fabs(f_eqm))*jump_fac );} else {
                    if(fabs(x_f)<10.*fabs(f_eqm)) {x_f=x_f*(1.-jump_fac)+f_eqm*jump_fac;} else {
                        x_f = +exp( log(fabs(x_f))*(1.-jump_fac) + log(fabs(f_eqm))*jump_fac );}}
            }

        } else {

            // timestep is smaller than the timescale to approach equilibrium, so integrate directly,
            //  but we will still use a fully implicit backwards-Euler type scheme for the two 'stiffest'
            //  components of the system (namely, the flux and eA term corresponding to the multiplicative direction)
            //  [the other terms, e.g. the damped energy change and the CR energy change, can be dealt with after]
            //
            double x_dum=0, x_out=0; n_eqm_loops=1.; // (if we enter this,  we need to terminate the parent loop above)
            if(f_eqm>0) {x_dum=x_um;} else {x_dum=x_up;}
            double q0 = 1.+ceff2_va2*dtaux*x_dum, g00 = gammCR*x_e + gamma_in_t_ll, psi00 = psifac + x_dum*x_e;
            double a_m1 = -x_up_prev/dtaux, a_0 = g00 + 1./dtaux , a_1 = gamma_nll, c2dt = ceff2_va2*dtaux;
            if(f_eqm<0) {psi00=psifac-x_dum*x_e; a_1=-gamma_nll; a_0=-(g00 + 1./dtaux); a_m1=x_um_prev/dtaux;}
            double d0 = -a_m1*q0, c0 = x_f_prev - a_0*q0 - c2dt*(a_m1 + psi00), b0 = -a_1*q0 - c2dt*(a_0-x_e), a0 = -a_1*c2dt;
            if(f_eqm<0) {b0 = -a_1*q0 - c2dt*(a_0+x_e);}
            if(fabs(a0) < EPSILON_SMALL)
            {
                if(fabs(b0) < EPSILON_SMALL)
                {
                    x_out = fabs(d0/c0); // linear solve
                } else {
                    d0/=c0; b0/=c0; c0=fabs(4.*b0*d0); if(c0<1.e-4) {c0*=0.5;} else {c0=sqrt(1.+c0)-1.;}
                    x_out = c0/(2.*fabs(b0)); // quadratic solve
                }
            } else {
                // cubic solve
                double p0=-b0/(3.*a0), q0=p0*p0*p0 + (b0*c0-3.*a0*d0)/(6.*a0*a0), r0=c0/(3.*a0), f0=r0-p0*p0, g0=q0*q0 + f0*f0*f0;
                if(g0 >= 0.)
                {
                    g0=sqrt(g0); a0=q0+g0; b0=q0-g0;
                    q0=pow(fabs(a0),1./3.); if(a0<0) {q0*=-1.;}
                    r0=pow(fabs(b0),1./3.); if(b0<0) {r0*=-1.;}
                } else {
                    g0=sqrt(-g0); a0=sqrt(q0*q0+g0*g0); b0=atan(g0/q0); r0=0.; q0=2.*a0*cos(b0/3.);
                }
                x_out = fabs(p0 + q0 + r0);
            }
            x_f = a_m1/x_out + a_0 + a_1*x_out;
            if(f_eqm>0) {x_up=x_out;} else {x_um=x_out;}

        }

        // now deal with the non-stiff part of the equations, namely the other Alfven-energy component + CR energy
        if(f_eqm > 0) // do the evolution for the eA term -not- involved in the stiff part of the equations //
        {
            double g0 = gammCR*x_e + gamma_in_t_ll + x_f; // pure-damping for um
            if(g0 > 0) {
                expfac=g0*dtaux; if(expfac>efmax) {expfac=efmax;}
                if(expfac>1.e-6) {expfac=exp(expfac)-1.;}
                x_um /= (1. + (1.+gamma_nll*x_um/g0)*expfac);
            } else {x_um -= x_um*(g0 + gamma_nll*x_up)*dtaux;} // (linear if x_f hasn't behaved yet)
        } else {
            double g0 = gammCR*x_e + gamma_in_t_ll - x_f; // pure-damping for up
            if(g0 > 0) {
                expfac=g0*dtaux; if(expfac>efmax) {expfac=efmax;}
                if(expfac>1.e-6) {expfac=exp(expfac)-1.;}
                x_up /= (1. + (1.+gamma_nll*x_up/g0)*expfac);
            } else {x_up -= x_up*(g0 + gamma_nll*x_up)*dtaux;} // (linear if x_f hasn't behaved yet)
        }

        // calculate total-energy damping (needed for deriving change in e_cr, which is then given by energy conservation) //
        double x_um_eff=0.5*(x_um+x_um_prev), x_up_eff=0.5*(x_up+x_up_prev), x_f_eff=0.5*(x_f+x_f_prev); // effective values for use in damping rates below
        expfac=gamma_in_t_ll*dtaux; if(expfac>efmax) {expfac=efmax;}
        if(expfac>1.e-6) {expfac=exp(expfac)-1.;}
        double de_damp = x_um_eff/(1.+1./(expfac*(1.+gamma_nll*x_um_eff/gamma_in_t_ll))) +
                         x_up_eff/(1.+1./(expfac*(1.+gamma_nll*x_up_eff/gamma_in_t_ll))); // energy loss to thermalized wave-damping
        if(!isfinite(de_damp)) {de_damp=0;}
        double e_tot = DMAX(x_um_prev,0) + DMAX(x_up_prev,0) + DMAX(x_e_prev,0) - de_damp; // total energy (less damping) before step
        double x_e_egycon = DMAX(e_tot-(x_up+x_um), Min_Egy);
        expfac = gammCR*(x_up_eff+x_um_eff)*dtaux; double x_numer = x_e_prev + dtaux*(x_um_eff-x_up_eff)*x_f_eff;
        if(expfac<0.9 && x_numer>0.) {x_e=x_numer/(1.-expfac);} else {
            if(expfac*x_e_prev+x_numer > 0.01*x_e) {x_e=expfac*x_e_prev+x_numer;} else {x_e*=0.01;}}
        if(fabs(x_e_egycon-x_e_prev) < fabs(x_e-x_e_prev)) {x_e=x_e_egycon;}
        if(e_tot < Min_Egy || !isfinite(e_tot)) {e_tot = Min_Egy;} // enforce minima/maxima
        if(x_e   < Min_Egy || !isfinite(x_e)  ) {x_e   = Min_Egy;} // enforce minima/maxima
        if(x_um<EPSILON_SMALL || !isfinite(x_um)) {x_um=EPSILON_SMALL;} else {if(x_um>xkappa_max) {x_um=xkappa_max;}} // enforce minima/maxima
        if(x_up<EPSILON_SMALL || !isfinite(x_up)) {x_up=EPSILON_SMALL;} else {if(x_up>xkappa_max) {x_up=xkappa_max;}} // enforce minima/maxima
        if(x_um+x_up<xkappa_min) {expfac=xkappa_min/(x_um+x_up); x_um*=expfac; x_up*=expfac;} // only want to enforce -sum- having effective diffusivity, not both
        double e_tot_new=x_e+x_um+x_up; x_e*=e_tot/e_tot_new; x_up*=e_tot/e_tot_new; x_um*=e_tot/e_tot_new; // check energy after limit-enforcement
	    fmax = x_e*sqrt(ceff2_va2); if(!isfinite(x_f)) {x_f=0;} else {if(x_f>fmax) {x_f=fmax;} else {if(x_f<-fmax) {x_f=-fmax;}}} // check for flux maximum/minimum

        // calculate change in parameters to potentially break the cycle here
        dx_e  = (x_e - x_e_prev) / (EPSILON_SMALL + x_e + x_e_prev);
        dx_up = (x_up - x_up_prev) / (EPSILON_SMALL + x_up + x_up_prev);
        dx_um = (x_um - x_um_prev) / (EPSILON_SMALL + x_um + x_um_prev);
        dx_f  = (x_f - x_f_prev) / (EPSILON_SMALL + fabs(x_f) + fabs(x_f_prev));
        double dx_max = sqrt(dx_e*dx_e + dx_up*dx_up + dx_um*dx_um + dx_f*dx_f); // sum in quadrature
        if(!isfinite(dx_max)) {dx_max=1;} // enforce validity for check below

        if((n_iter > 0) && (n_iter % 10000 == 0)) // print diagnostics if the convergence is happening slowly
        {
            PRINT_WARNING("niter/max=%ld/%ld dtau/step=%g/%g d_params=%g (init/previous/now) xeCR=%g/%g/%g xeAp=%g/%g/%g xeAm=%g/%g/%g xflux=%g/%g/%g ceff2_va2=%g damp_g_intll=%g damp_g_nll=%g psi_gradientfac=%g heat_term=%g xkappa_min/max=%g/%g egy_min=%g flux_max=%g",
                n_iter,n_iter_max,dtau,dtaux,dx_max,x_e_0,x_e_prev,x_e,x_up_0,x_up_prev,x_up,x_um_0,x_um_prev,x_um,x_f_0,x_f_prev,x_f,ceff2_va2,gamma_in_t_ll,gamma_nll,psifac,(x_e_0+x_up_0+x_um_0)-(x_e+x_up+x_um),xkappa_min,xkappa_max,Min_Egy,fmax);
        }
        dtau -= dtaux; // subtract the time we've already integrated from the total timestep
        n_iter++; // count the number of iterations
        if(dtau <= 0.) break; // we have reached the end of the integration time for our sub-stepping. we are done!
        if(dx_max <= 1.e-3*dtaux/dtau) break; // the values of -all- the parameters are changing by much less than the floating-point errors. we are done!
        if(n_iter > n_iter_max) break; // we have reached the maximum allowed number of iterations. we give up!
    }

    // ok! done with the main integration/sub-cycle loop, now just do various clean-up operations
    double thermal_heating = eCR_0 * ((x_e_0+x_up_0+x_um_0)-(x_e+x_up+x_um)); // net thermalized energy from damping terms
    if(thermal_heating < 0 || !isfinite(thermal_heating)) // if this is less than zero (from residual floating-point error), then the energy goes up, which shouldnt happen: set to zero
    {
        e_tot=x_e_0+x_up_0+x_um_0; e_tot_new = x_e+x_up+x_um; // initial and final energies should be equal in this case
        x_e *= e_tot/e_tot_new; x_up *= e_tot/e_tot_new; x_um *= e_tot/e_tot_new; // enforce that equality
        thermal_heating=0; // set thermal change to nil
    }
    eCR=eCR_0*x_e; eA[0]=eCR_0*x_up; eA[1]=eCR_0*x_um; f_CR=f_unit*x_f; Min_Egy*=eCR_0; xkappa_min*=eCR_0; xkappa_max*=eCR_0; // re-assign dimensional quantities

    // assign the updated values back to the resolution elements, finally!
    if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy]=eCR;} else {cell[i].CosmicRayEnergyPred[k_CRegy]=eCR;} // CR energy
    for(k=0;k<2;k++) {if(mode==0) {cell[i].CosmicRayAlfvenEnergy[k_CRegy][k]=eA[k];} else {cell[i].CosmicRayAlfvenEnergyPred[k_CRegy][k]=eA[k];}} // Alfven energy
    {Vec3<double> flux_out = bhat*f_CR; if(mode==0) {for(k=0;k<3;k++) {cell[i].CosmicRayFlux[k_CRegy][k]=flux_out[k];}} else {for(k=0;k<3;k++) {cell[i].CosmicRayFluxPred[k_CRegy][k]=flux_out[k];}}} // assign to flux vector
    if(mode==0) {cell[i].InternalEnergy+=thermal_heating/pp[i].Mass;} else {cell[i].InternalEnergyPred+=thermal_heating/pp[i].Mass;} // heating term from damping
    cell[i].CosmicRayDiffusionCoeff[k_CRegy] = 1. / (fac_Omega*((eA[0]+eA[1])/E_B)/(clight_code*clight_code)); // effective diffusion coefficient in code units


    } // complete loop over CR bins
    return 1; // exit
}
#endif /* CRFLUID_EVOLVE_SCATTERINGWAVES */

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
    if(i < 0) {return 1.e-17;} /* return standard LISM background value here since we are not in a gas cell evaluating actual CR physics */
    double zeta_cr = 0;
#if defined(COSMIC_RAY_FLUID) && (N_CR_PARTICLE_BINS > 2)
    double ecr_units=(cell[i].Density*All.cf_a3inv/pp[i].Mass)*UNIT_PRESSURE_IN_CGS; int k;
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        double T_GeV=return_CRbin_kinetic_energy_in_GeV(-1,k,cell), beta=return_CRbin_beta_factor(-1,k,cell), Z=return_CRbin_CR_charge_in_e(-1,k), gamma=return_CRbin_gamma_factor(-1,k,cell);
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
            double E_GeV = return_CRbin_kinetic_energy_in_GeV(target,k_CRegy, cell), beta = return_CRbin_beta_factor(target,k_CRegy, cell), Z=fabs(return_CRbin_CR_charge_in_e(target,k_CRegy));
            double T_eff_fullion = MEAN_MOLECULAR_WEIGHT_IONIZED*(GAMMA_DEFAULT-1.)*U_TO_TEMP_UNITS*cell[target].InternalEnergyPred, /* deliberately the fully-ionized value, as the name says */ xm = 0.0286*sqrt(T_eff_fullion/2.e6);
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


/* ======================================================================
 * Subgrid-LEBRON cosmic-ray source injection (gravity-tree source payload).
 * Device + host single source of truth; host externals in
 * cosmic_ray_utilities.cc. Depends on stellar-age + sink-CR-efficiency cores.
 * ====================================================================== */
#if defined(COSMIC_RAY_SUBGRID_LEBRON) && (defined(GRAVTREE_SOURCE_HOST_OWNER_TU) || (defined(GRAVTREE_SOURCE_DEVICE_TU) && defined(GRAVTREE_SOURCE_LAZY_SUPPORTED)))
#include "../../galaxy_sf/stellar_evolution_functions.h"
#include "../../sinks/sink_functions.h"

KOKKOS_INLINE_FUNCTION double cr_get_source_shieldfac(int i, struct particle_data *pp, struct gas_cell_data *cell);

KOKKOS_INLINE_FUNCTION double cr_get_source_injection_rate(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double Edot = 0;
#ifdef GALSF
#ifdef GALSF_FB_MECHANICAL
    if(pp[i].Type == 4)
    {
        double star_age=evaluate_stellar_age_Gyr_core(i, pp), RSNe=0, agemin=0.003401, agebrk=0.01037, agemax=0.03753;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        agemin=0.0037; agebrk=0.7e-2; agemax=0.044; double f1=3.9e-4, f2=5.1e-4, f3=1.8e-4; // inputs for newer SNe rate (and newer Ia rate below)
        if(star_age<agemin) {RSNe=0;} else if(star_age<=agebrk) {RSNe=f1*pow(star_age/agemin,log(f2/f1)/log(agebrk/agemin));} else if(star_age<=agemax) {RSNe=f2*pow(star_age/agebrk,log(f3/f2)/log(agemax/agebrk));} else {RSNe=0;} // core-collapse; updated with same stellar evolution models for wind mass loss [see there for references]. simple 2-part power-law provides extremely-accurate fit. models predict a totally negligible metallicity-dependence.
        double t_Ia_min=agemax, norm_Ia=1.6e-3; if(star_age>t_Ia_min) {RSNe += norm_Ia * 7.94e-5 * pow(star_age,-1.1) / fabs(pow(t_Ia_min/0.1,-0.1) - 0.61);} // Ia DTD following Maoz & Graur 2017, ApJ, 848, 25
        //if(star_age < 0.04) {RSNe = 3.0e-4;} else {RSNe = DMIN(3.e-4 , RSNe);} /* replace this with a 'time smoothed' version over the last ~100+ Myr */
#else
        if(star_age>agemin) {if(star_age<=agebrk) {RSNe=5.408e-4;} else {if(star_age<=agemax) {RSNe=2.516e-4;}}} // core-collapse rate [super-simple 2-piece constant] //
        if(star_age>agemax) {RSNe=5.3e-8 + 1.6e-5*exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));} // Ia (prompt Gaussian+delay, Manucci+06)
#endif
        Edot = All.CosmicRay_SNeFraction * (RSNe*UNIT_TIME_IN_MYR) * (pp[i].Mass*UNIT_MASS_IN_SOLAR) * (1.0e51/UNIT_ENERGY_IN_CGS);
    }
#endif
#ifdef SINK_PARTICLES
    if(pp[i].Type == 5) {
        double mdot_eff = pp[i].Sink_Mdot; // code units
        mdot_eff = DMIN( mdot_eff , pp[i].Sink_Mass / (100./UNIT_TIME_IN_MYR) ); // if time-averaging over ~Gyr, can't have time-averaged injection rate above Mbh/<t> more or less (modulo order-one corrections for all this)
        Edot = evaluate_sink_cosmicray_efficiency_core(pp[i].Sink_Mdot,pp[i].Sink_Mass,i) * mdot_eff * C_LIGHT_CODE*C_LIGHT_CODE; // injection in code units
    }
#endif
#endif
    if(Edot > 0) {return Edot * cr_get_source_shieldfac(i, pp, cell);} else {return 0;}
}


KOKKOS_INLINE_FUNCTION double cr_get_source_shieldfac(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double cr_atten_fac = 1;
    if(pp[i].KernelRadius > 0 && pp[i].NumNgb > 0 && All.Time > All.TimeBegin)
    {
        double dx=pp[i].KernelRadius/pp[i].NumNgb, rho; // code units
        Vec3<double> gradrho = pp[i].GradRho;
        if(pp[i].Type==0) {rho=cell[i].Density;} else {rho=pp[i].DensityAroundParticle;}
        if(rho > 0)
        {
            double gradrho_mag = gradrho.norm();
            if(gradrho_mag > 0) {dx += rho/gradrho_mag;} // code units
            double R_loss = ((6.37 + 3.09)*1.e-16*UNIT_TIME_IN_CGS) * (rho*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS); // physical units
            double psi_loss_i = (R_loss / All.CosmicRay_Subgrid_Vstream_0) / sqrt(1. + R_loss*All.CosmicRay_Subgrid_Kappa_0/(All.CosmicRay_Subgrid_Vstream_0*All.CosmicRay_Subgrid_Vstream_0)); // physical units
            double dtau = 0.5 * psi_loss_i * (dx*All.cf_atime); // physical units in dx, so dimensionless here
            cr_atten_fac = exp(-DMIN(dtau, 50.));
        }
    }
    return cr_atten_fac;
}

#endif /* COSMIC_RAY_SUBGRID_LEBRON && (GRAVTREE_SOURCE_HOST_OWNER_TU || (GRAVTREE_SOURCE_DEVICE_TU && GRAVTREE_SOURCE_LAZY_SUPPORTED)) */

/* subroutine to calculate which part of the adiabatic PdV work from the RP gets assigned to the CRs vs the gas; since the CRs are always smooth by definition under this operation this follows simply from the local cell divergence and the effective CR eos */
KOKKOS_INLINE_FUNCTION double CR_calculate_adiabatic_gasCR_exchange_term(int i, double dt_entr, double gamma_minus_eCR_tmp, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    double u0, d_CR; if(mode==0) {u0=cell[i].InternalEnergy;} else {u0=cell[i].InternalEnergyPred;} // initial energy
    if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code

    double divv_p=-dt_entr*pp[i].Particle_DivVel*All.cf_a2inv, divv_f=divv_p, divv_u=0; // get locally-estimated gas velocity divergence for cells - if using non-Lagrangian method, need to modify. take negative of this [for sign of change to energy] and multiply by timestep
#ifdef COSMIC_RAY_FLUID
    divv_f=-dt_entr*cell[i].Face_DivVel_ForAdOps*All.cf_a2inv;
#endif
    if(All.ComovingIntegrationOn) {double divv_h=-dt_entr*(3.*All.cf_hubble_a); divv_p+=divv_h; divv_f+=divv_h;} // include hubble-flow terms
    double P_cr = gamma_minus_eCR_tmp * cell[i].Density * All.cf_a3inv / pp[i].Mass, P_tot = cell[i].Pressure * All.cf_a3inv; // define the pressure from CRs and total pressure (physical units)
#ifdef MAGNETIC
    double B2 = (cell[i].Bfield() * All.cf_a2inv).norm_sq();
    P_tot += 0.5*B2; // add magnetic pressure [B^2/2], in physical code units, since it contributes to the PdV work but not included in 'pressure' total above
#endif
    double fac_P = DMAX(0, DMIN(1, P_cr/(P_tot + 1.e-10*P_cr + MIN_REAL_NUMBER))); // fraction of total pressure from CRs
    double Ui = u0 * pp[i].Mass; // factor for multiplication below, and initial thermal energy
    double dtI_hydro = cell[i].DtInternalEnergy * pp[i].Mass * dt_entr; // change given by hydro-step computed delta_InternalEnergy
    double min_IEgy = pp[i].Mass * All.MinEgySpec; // minimum internal energy - in total units -

    if(divv_p*dtI_hydro > 0 || divv_f*dtI_hydro > 0) // same sign from hydro and from smooth-flow-estimator, suggests we are in a smooth flow, so we'll use stronger assumptions about the effective 'entropy' here
    {
        if(divv_p*dtI_hydro <= 0) {divv_u=divv_f;} // if divv_f agrees in sign here, use it
        if(divv_f*dtI_hydro <= 0) {divv_u=divv_p;} // if divv_p agrees in sign here, use it
        if(divv_p*divv_f > 0) {if(fabs(divv_p) > fabs(divv_f)) {divv_u=divv_p;} else {divv_u=divv_f;}} // if both agree in sign here, use -larger- since more accurately captures CR-dominated limit
        d_CR = gamma_minus_eCR_tmp * divv_u; // expected PdV CR energy change
        if(fabs(d_CR) > fabs(dtI_hydro)) {d_CR = dtI_hydro;} // do not allow this to exceed the sum (since all terms have the same sign here, in a well-ordered smooth flow)
        if(fabs(d_CR) < fac_P*fabs(dtI_hydro)) {d_CR = fac_P*dtI_hydro;} // but also do not allow CR term to be -below- CR pressure fraction times total term, since that should be attributed to the CR (as this is all a quasi-adiabatic term)
    } else { // both divv terms agree with each other, but dis-agree with the sign of the total change. can't assume anything about smoothness-of-the-flow
        if(fabs(divv_p) > fabs(divv_f)) {divv_u=divv_f;} else {divv_u=divv_p;} // pick the divv estimator with the smaller absolute magnitude, since it deviates
        d_CR = gamma_minus_eCR_tmp * divv_u; // expected PdV CR energy change
        double f_limiter, fac_test=fabs(d_CR)/fabs(dtI_hydro); if(fac_test>fac_P) {d_CR*=fac_P/fac_test;} // don't let CR change exceed their pressure fraction
        if(d_CR > 0) {if(Ui <= min_IEgy) {f_limiter = 1.e-20;} else {f_limiter=0.5;} // gas will be 'cooled', limit so don't overshoot when Pcr is large
            if(d_CR > f_limiter*(Ui-min_IEgy)) {d_CR = f_limiter*(Ui-min_IEgy);} // limit fractional loss to gas
        } else {f_limiter = 1000.; if(fabs(d_CR)>f_limiter*Ui) {d_CR=-f_limiter*Ui;}} // gas will be heated, limit fractional gain
    }
#if defined(CRFLUID_EVOLVE_SPECTRUM) && !defined(COOLING_OPERATOR_SPLIT)
    cell[i].Face_DivVel_ForAdOps = -d_CR / (All.cf_a2inv * gamma_minus_eCR_tmp * dt_entr + MIN_REAL_NUMBER); // this is the 'effective' divergence here (in code units) which matches exactly the change in CR energy when the above limiters etc are applied. we can save this for use in the other CR subroutines
#endif
    return d_CR; // return final value
}

