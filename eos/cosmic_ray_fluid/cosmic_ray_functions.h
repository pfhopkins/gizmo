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

KOKKOS_INLINE_FUNCTION double return_CRbin_CR_rigidity_in_GV(int target, int k_CRegy) {
    double R = 1;
#if (N_CR_PARTICLE_BINS == 2)
    double Rv[2]={1.8, 0.6}; R=Rv[k_CRegy]; // approximate peak energies of each from Cummings et al. 2016 Fig 15
#endif
#if (N_CR_PARTICLE_BINS > 2)
#ifndef GIZMO_GPU_COMPILER
    if(target >= 0) {R=CR_return_mean_rigidity_in_bin_in_GV(target,k_CRegy, CellP);} else {R=All.CR_global_rigidity_at_bin_center[k_CRegy];} // this is pre-defined globally for this bin list
#else
    R=All.CR_global_rigidity_at_bin_center[k_CRegy]; // GPU fallback: spectral-slope-weighted mean not available on device
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

KOKKOS_INLINE_FUNCTION double gamma_eos_of_crs_in_bin(int k_CRegy)
{
    return (4. + 1./return_CRbin_gamma_factor(-1,k_CRegy)) / 3.;
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
    double gamma_0=return_CRbin_gamma_factor(target_for_cr_gamma,k_CRegy), gamma_fac=gamma_0/(gamma_0-1.), beta_fac=return_CRbin_beta_factor(target_for_cr_gamma,k_CRegy); // lorentz factor here, needed in next line, because the loss term here scales with -total- energy, not kinetic energy
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
#else /* CRFLUID_EVOLVE_SPECTRUM is defined — forward to multibin solver */
KOKKOS_INLINE_FUNCTION void CR_cooling_and_losses(int target, double n_elec, double nHcgs, double dtime_cgs, struct particle_data *pp, struct gas_cell_data *cell) {
    CR_cooling_and_losses_multibin(target, n_elec, nHcgs, dtime_cgs, 0, pp, cell);
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
