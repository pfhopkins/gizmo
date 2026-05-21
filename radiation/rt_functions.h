/* rt_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations of RT
 * utility functions used by the cooling chain.  Single source of truth:
 * included by both cooling/cooling.cc (GPU kernel) and
 * radiation/rt_utilities.cc (host path).
 *
 * Contains: dust_planck_mean_opacity, blackbody_lum_frac,
 *   rt_kappa_adaptive_IR_band, dust_dEdt, rt_eqm_dust_temp,
 *   rt_irband_egydensity_in_band.
 *
 * Include order: after allvars.h and proto.h (for struct types, All, and
 * forward declarations of gas_dust_heating_coeff, ThermalProperties, etc.).
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#include "../system/bracketed_rootfind_functions.h"
/* rt_kappa and other device-callable functions below reference
 * return_dust_to_metals_ratio_vs_solar (defined as KOKKOS_INLINE_FUNCTION in
 * eos/eos_functions.h). Without this include, TUs that pull rt_functions.h
 * but not eos_functions.h see only the core/proto.h __host__-only forward
 * declaration, and nvc++ emits warning #20011-D (silent physics error on the
 * device pass). Self-contained header avoids per-caller pre-include burden. */
#include "../eos/eos_functions.h"

/* ========================================================================
 * dust_planck_mean_opacity — Semenov 2003 table interpolation
 * (moved from radiation/rt_dust_opacity.cc)
 * ======================================================================== */

/* Returns the Planck-mean dust opacity tabulated for the Semenov 2003 5-layered
 porous shell dust model.

 Parameters
 ----------
 Trad: Radiation temperature in K
 Tdust: Dust temperature in K

 Returns
 -------
 kappa_dust: Planck-mean dust opacity in cm^2/g assuming Solar metallicity
 */
KOKKOS_INLINE_FUNCTION
MyFloat dust_planck_mean_opacity(MyFloat Trad, MyFloat Tdust) {
    static constexpr int N_TRAD_loc = 15;
    static constexpr int N_TDUST_loc = 5;
    static constexpr MyFloat Tdust_zones[5] = {160, 275, 425, 680, 1500};
    static constexpr MyFloat logTrad_table[15] = {0.0,
                                     0.2857142857142857,
                                     0.5714285714285714,
                                     0.8571428571428571,
                                     1.1428571428571428,
                                     1.4285714285714284,
                                     1.7142857142857142,
                                     2.0,
                                     2.2857142857142856,
                                     2.571428571428571,
                                     2.8571428571428568,
                                     3.142857142857143,
                                     3.4285714285714284,
                                     3.714285714285714,
                                     4.0};
    static constexpr MyFloat log_kappadust_table[5][15] = {
        {-1.909515215033498, -1.5017616295543856, -1.2610211141587906,
         -1.0643751130254193, -0.7661794028048912, -0.2485164276172981,
         0.3936319485109052, 0.7185651396015793, 1.003536500936941,
         1.0703750048744685, 1.185414318657744, 1.4334392971521788,
         1.6154100043688104, 1.833331149478953, 2.2402919406422592},
        {-2.0584711507666156, -1.639523808954471, -1.3937562378746238,
         -1.217712452271845, -1.0035307263948003, -0.6537979484512214,
         -0.14444336531252724, 0.33781494089162734, 0.6204502553977504,
         0.7073792380378322, 0.817126602196553, 1.1620592165098858,
         1.4775329387335934, 1.7314724407802422, 2.122441393841703},
        {-2.092756656625387, -1.6688435106176933, -1.4197982288972344,
         -1.2491484838263678, -1.0526101054461765, -0.7239962293949143,
         -0.21897626943630635, 0.26791745490238816, 0.5498747399044497,
         0.6380011931553505, 0.7545581854919681, 1.1204828950445178,
         1.447717175478172, 1.6975481925473666, 2.066531977172171},
        {-2.466846959771108, -1.9983094494283768, -1.7094791411944892,
         -1.5454814504892613, -1.4403397374848839, -1.2856048236415571,
         -0.8297121104893529, -0.28151883033918534, 0.004258175582704628,
         0.12669094376131687, 0.31518820522515606, 0.8267768233252955,
         1.2296194436753016, 1.4691152621947785, 1.6952767195239868},
        {-3.7448342907548136, -3.298656729562377, -2.8766563714426696,
         -2.4999445785716605, -2.1658033063307927, -1.8743193706624732,
         -1.5588128589881036, -1.0606192296411485, -0.33965876540303136,
         0.5368140402181846, 1.1915541410857275, 1.4384491935887818,
         1.4792273276721044, 1.5197919949174217, 1.6603749111299055},
    };

    MyFloat logTmax = logTrad_table[N_TRAD_loc - 1], logTmin = logTrad_table[0];
    MyFloat logT = log10(Trad);

    int Tdust_idx;
    for (Tdust_idx = 0; Tdust_idx < N_TDUST_loc; Tdust_idx++) {
        if (Tdust < Tdust_zones[Tdust_idx]) {
            break;
        }
    }
    if (Tdust_idx >= N_TDUST_loc) { // Tdust exceeds table max
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS) || defined(GALSF_ISMDUSTCHEM_MODEL) || defined(RT_INFRARED) || (defined(COOL_LOW_TEMPERATURES) && !defined(SINGLE_STAR_SINK_DYNAMICS))
        Tdust_idx = N_TDUST_loc-1; // Tdust is hot but we have explicit grain model and need IR opacity to calculate dust-to-gas ratios, so return max table value
#else
        return MIN_REAL_NUMBER; // Tdust is hot so dust is sublimated; return smol value
#endif
    }

    if (logT >= logTmax) {
        return pow(10., log_kappadust_table[Tdust_idx][N_TRAD_loc - 1]);
    }
    if (logT <= logTmin) {
        return pow(10., log_kappadust_table[Tdust_idx][0]);
    }

    MyFloat dlogT = logTrad_table[1] - logTrad_table[0];
    int Trad_idx = (int)(N_TRAD_loc - 1) * logT / (logTmax - logTmin);
    MyFloat wt1 = 1 - (logT - logTrad_table[Trad_idx]) / dlogT, wt2 = 1 - wt1;
    MyFloat log_kappa = wt1 * log_kappadust_table[Tdust_idx][Trad_idx] +
                        wt2 * log_kappadust_table[Tdust_idx][Trad_idx + 1];
    return pow(10., log_kappa);
}


/* ========================================================================
 * blackbody_lum_frac — fraction of blackbody SED in a photon energy band
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

/* returns the fraction of a blackbody SED in a given photon energy band - accurate to <1% over all wavelengths
   E_lower - lower end of the energy band in eV
   E_upper - upper end of the energy band in eV
   T_eff - effective blackbody temperature of the SED, in K
*/
KOKKOS_INLINE_FUNCTION
double blackbody_lum_frac(double E_lower, double E_upper, double T_eff)
{
    double k_B = 8.617e-5; // Boltzmann constant in eV/K
    double x1 = E_lower / (k_B * T_eff), x2 = E_upper / (k_B * T_eff), f_lower, f_upper;
    if(x1 < 3.40309){
      f_lower = (131.4045728599595*x1*x1*x1)/(2560. + x1*(960. + x1*(232. + 39.*x1))); // approximation of integral of Planck function from 0 to x1, valid for x1 << 1
    } else {
      f_lower = 1 - (0.15398973382026504*(6. + x1*(6. + x1*(3. + x1))))*exp(-DMIN(x1,40.)); // approximation of Planck integral for large x
    }
    if(x2 < 3.40309){
      f_upper = (131.4045728599595*x2*x2*x2)/(2560. + x2*(960. + x2*(232. + 39.*x2))); // approximation of integral of Planck function from 0 to x2, valid for x2 << 1
    } else {
      f_upper = 1 - (0.15398973382026504*(6. + x2*(6. + x2*(3. + x2))))*exp(-DMIN(x2,40.)); // approximation of Planck integral for large x
    }
    double df = f_upper - f_lower;
    if(df<=0) {if(x2<=x1) {return 0;} else {if(x1>4.) {if(x1<120.) {return 0.15398973382026504*(6.+x1*(6.+x1*(3.+x1)))*exp(-DMIN(x1,120.));} else {return 2.e-47;}}}}
    return DMAX(df, 0);
}


/* ========================================================================
 * rt_kappa_adaptive_IR_band — full Semenov + gas-phase opacity
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
double rt_kappa_adaptive_IR_band(int i, double T_dust, double Trad, int do_emission_absorption_scattering_opacity, int dust_or_gas_opacity_only_flag, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(do_emission_absorption_scattering_opacity==1) {Trad = T_dust;} // if we want the emissivity then we assume radiation emitted at T_dust
    double fac=UNIT_SURFDEN_IN_CGS, x = 4.*log10(Trad) - 8., kappa=0, T_dust_opacitytable = T_dust; // needed for fitting functions to opacities (may come up with cheaper function later)
    double dx_excess=0; if(x > 7.) {dx_excess=x-7.; x=7.;} // cap for maximum temperatures at which fit-functions should be used //
    //if(x < -4.) {x=-4.;} // cap for minimum temperatures at which fit functions below should be used //
    double Zfac = 1.0, dust_to_metals_vs_standard = return_dust_to_metals_ratio_vs_solar(i, T_dust, pp, cell); // avoid call to return_dust_to_metals_ratio_vs_solar to avert circular dependency
#ifdef METALS
    if(i>=0) {Zfac = pp[i].Metallicity[0]/All.SolarAbundances[0];}
#endif

    if(dust_or_gas_opacity_only_flag >= 0) // dust opacities
    {
        // use fancy detailed fit with composition varying by dust temperature
        /* opacities are from tables of Semenov et al 2003; we use their 'standard'
         model, for each -dust- temperature range (which gives a different dust composition,
         hence different wavelength-dependent specific opacity). We then integrate to
         get the Rosseland-mean opacity for the given dust composition, assuming
         the radiation is a blackbody with the specified -radiation- temperature.
         We adopt their 'porous 5-layered sphere' model for dust composition.
         We use simple fitting functions to the full tabulated data: however, note that
         (because the blackbody assumption smoothes fine structure in the opacities),
         the deviations from the fit functions are much smaller than the deviations owing
         to different grain composition choices (porous/non, composite/non, 5-layer/aggregated/etc)
         in Semenov et al's paper */
        kappa = dust_planck_mean_opacity(Trad, T_dust_opacitytable);
#ifdef RADTRANSFER
        if((do_emission_absorption_scattering_opacity==1) || (do_emission_absorption_scattering_opacity==-1)) {
            kappa *= (1.-0.5/(1.+((725.*725.)/(1.+Trad*Trad)))); /* rough interpolation for dust depending on the radiation temperature: high Trad, this is 1/2, low Trad, gets closer to unity */
        } /* multiply by (1-albedo) because absorption depends only on albedo, and emission cross section depends only on kappa_absorption */
#endif
        kappa *= Zfac*dust_to_metals_vs_standard; // the above are all dust opacities, so they scale with dust content per our usual expressions
    }

    if(dust_or_gas_opacity_only_flag <= 0) // non-dust (e.g. gas-phase) IR opacities
    {
        /* this is an approximate result for a wide range of low-to-high-temperature opacities -not- from the dust phase, but provides a pretty good fit from 1.5e3 - 1.0e9 K, and valid at O(1) level down to <10 K, with updates from PFH in Sept 2022 */
        double x_elec = 1., zmetals = 0.014;
#if defined(COOLING) && !defined(CHIMES)
        x_elec = cell[i].Ne; // actual free electron fraction
#endif
#ifdef METALS
        zmetals = pp[i].Metallicity[0];
#endif
        double f_neutral_approx = DMAX(0., 1.-x_elec); /* approximate neutral fraction (good enough for us for what we need below) */
        double f_free_metals_approx = zmetals * DMAX(0, 1.-0.5*dust_to_metals_vs_standard); /* metal mass fraction times the free (not locked in dust abundance), assuming the default solar scaling is 1/2 */
        double Tgas=1. + 0.59*(cell[i].gamma_eos_value()-1.)*U_TO_TEMP_UNITS*cell[i].InternalEnergyPred, rho_cgs = cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS; /* crude estimate of gas temperature to use with scalings below, and gas density in cgs */
        double k_electron = 0.4 * HYDROGEN_MASSFRAC * x_elec / ((1. + 2.7e11*rho_cgs/(Tgas*Tgas)) * (1. + pow(Trad/4.5e8, 0.86))); /* Thompson scattering (non-relativistic), scaling with free electron fraction [remembering that in our units, x_elec is n_e/n_H_nuclei, not scaled to total nuclear number]; includes corrections for partial degeneracy at low gas temperatures from Buchler et al. 1976, and Klein-Nishina terms at high radiation temperatures >1e9 */
        double k_molecular = 0.1 * (f_free_metals_approx + 3.e-9) * f_neutral_approx; /* molecular line opacities, which should only dominate at low-temperatures in the fits below, but are not really assumed to extrapolate to the very low densities we apply this to here; this works ok comparing e.g. Lenzuni, Chernoff & Salpeter 1991 ApJS 76 759L [opacities for metal free gases], using the 3e-9 to represent the H2 molecular opacity (really low, only here for completeness) */
#if defined(COOL_MOLECFRAC_NONEQM)
        k_molecular *= cell[i].MolecularMassFraction;
#endif
        double k_Kramers = 4.0e25 * (1.+HYDROGEN_MASSFRAC) * (f_free_metals_approx * exp(-DMIN(1.5e5/Trad,40.)) + 0.001*x_elec) * rho_cgs / (Trad*Trad*Trad*sqrt(Tgas)); /* free-free, bound-free, bound-bound transitions. bound-bound is small except at discrete wavelengths, so in a mean for a broad-band like we have here, is negligible. the 0.001 term is free-free, independent of metallicity, but note the power of the free electron fraction. the bound-free depends on metal ions here by assumption, specifically those not locked in dust, being ionized -- hence the exponential suppression at low radiation temperatures where the bound states cannot be ionized. the overall Tgas dependence here comes from the sound speed, the Trad from the wavelength (1/nu^3) dependence of the opacity */
        double k_effective_Fe = 1.5e20 * f_free_metals_approx * rho_cgs / (Trad*Trad) * exp(-DMIN(pow(0.8e4/Trad,4),40.)) * exp(-DMIN(pow(Trad/0.7e6,2),40.)); /* crude approximation to the iron line-blanketing opacity calculations from Jiang et al. 2015+2016 */
        k_Kramers += k_effective_Fe;
        double k_Rayleigh = f_neutral_approx * DMIN(5.e-19 * pow(Trad,4) , 0.2*(1.+ HYDROGEN_MASSFRAC)); /* rayleigh scattering from atomic gas [caps at thompson, much lower at low-T here] */
#ifdef COOLING
#ifdef RT_CHEM_PHOTOION
        double x_Hp = cell[i].HII, x_H0 = cell[i].HI;
#else
        double u_in=cell[i].InternalEnergy, rho_in=cell[i].Density*All.cf_a3inv, mu=1, ne=1, nHI=0, nHII=0, nHeI=1, nHeII=0, nHeIII=0;
        double temp = ThermalProperties(u_in, rho_in, i, &mu, &ne, &nHI, &nHII, &nHeI, &nHeII, &nHeIII, pp, cell);
        double x_Hp = nHII, x_H0 = nHI;
#endif
        double x_Hminus = 4.e-10 * Tgas * x_elec * x_H0 / ((1. + x_Hp*300. + x_elec*1000.*(Tgas/1.3e4)*(Tgas/1.3e4)/(1.+(Tgas/1.3e4)*(Tgas/1.3e4)) + 4.e-17*1.) * (1. + Tgas/3.e4)); /* H- abundance: see series of equations in our non-equilbrium molecular solver (from e.g. Glover and Jappsen 2007 and other sources), with simple but accurate enough for our purposes replacements to make it quick to compute these to the needed accuracy for our purposes. note we need the free-electron fraction, neutral fraction, and free proton fraction. these denominator terms quantify differences from the idealized scaling assumed here, which assumes an idealized scaling of xH0~1~constant and near-vanishing xHp and x_e, for lower temperatures. last term assumes a constant photon-to-baryon ratio for scaling to different environments */
        double k_Hminus_bf = 4.2e7 * pow(8760./Trad, 1.5) * exp(-DMIN(8760./Trad,40.)); /* bound-free H- opacity, from using the fitting functions in John 1988 [A&A, 193, 189], integrating over the Planck function for a flux-mean opacity (Rosseland mean ill-defined here because need all components since this vanishes outside certain ranges) */
        double phi_hm = DMIN(Tgas/5040.,2.), k_Hminus_ff = 1.9e6 * pow(8760./Trad, 2) * exp(-DMIN(8760./Trad,40.)) * (0.6-2.5*sqrt(phi_hm)+2.5*phi_hm+2.7*phi_hm*sqrt(phi_hm)); /* free-free H- opacity, mixing the fits from John and references in Lenzuni, Chernoff, & Salpeter, but re-calculated for arbitrary radiation vs gas temperature. note this will appear to give differences from their opacities, the main difference comes not from this expression (which is simplified) but from the different x_H- and x_e, which owes to a very different chain of expressions, which give a quite different result in the end. */
        double k_Hminus = x_Hminus * (k_Hminus_bf + k_Hminus_ff); /* add both together */
#else
        double k_Hminus = 1.1e-25 * sqrt((zmetals + 1.e-5) * rho_cgs) * pow(Tgas,7.7) * exp(-DMIN(8760./Trad,40.)); /* negative H- ion opacity (this is a fit for stellar atmospheres, which has a very strong temp dependence because of implicit free-electron and H- scaling with T, but that's not as useful for us since we're tracking the chemistry we need here) */
#endif
        double k_radiative = k_molecular + k_Kramers + k_Hminus + k_electron + k_Rayleigh; /* we don't want a rosseland mean here given our band divisions (already kramers and H- and molecular are rosseleand-mean-ized in fact within themselves), but here different sources should add linearly for a flux-mean */
        if((do_emission_absorption_scattering_opacity==1) || (do_emission_absorption_scattering_opacity==-1)) {k_radiative -= k_electron;} /* here we just want absorption/emission, not scattering opacity, so we do not include the free electron term */
        kappa += k_radiative; /* add this to the dust opacity we have already calculated above */
    }

    return kappa * fac;
}


/* ========================================================================
 * dust_dEdt — volumetric de/dt for dust energy balance root-finding
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
double dust_dEdt(int i, double T, double Tdust, double dust_absorption_rate, double fdustmet_init, struct particle_data *pp, struct gas_cell_data *cell)
{
    double nHcgs = HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS * cell[i].Density * All.cf_a3inv / PROTONMASS_CGS;    /* hydrogen number dens in cgs units */
    double fac_emission = 4.*5.67e-5/(UNIT_PRESSURE_IN_CGS*UNIT_VEL_IN_CGS)*cell[i].Density*All.cf_a3inv; // in code units
    double LambdaDust_fac = 0;
#ifdef COOLING
    if(T>0) {LambdaDust_fac = gas_dust_heating_coeff(i,T,Tdust, pp, cell) * nHcgs * nHcgs /(UNIT_PRESSURE_IN_CGS/UNIT_TIME_IN_CGS);}
#endif
    double kappa_emission = rt_kappa_adaptive_IR_band(i, Tdust, Tdust, 1, 1, pp, cell);
    double dust_emission = fac_emission * kappa_emission * pow(Tdust,4);
#if defined(COOLING) && !defined(RT_INFRARED) // if we aren't doing RT self-consistently, approximate outward radiative transport rate in optically-thick regime
    double column = evaluate_NH_from_GradRho(cell[i].Gradients.Density,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i, pp);
    double tau = column * kappa_emission;
    dust_emission /= (1 + tau*tau); // e.g. Masunaha & Inutsuka 1999, Rafikov 2007
#endif
    double fac_abs = 1.; /* this will rescale the estimated absorption by the new dust-to-gas ratio */
    if(fdustmet_init > 0.) {fac_abs = return_dust_to_metals_ratio_vs_solar(i,Tdust, pp, cell) / fdustmet_init;}
    double result_dedt = LambdaDust_fac * (T-Tdust) + fac_abs*dust_absorption_rate - dust_emission;
#ifdef GIZMO_DEBUG_RT_COOLING
    if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[DUST_DEDT] ID=%llu T=%.4e Td=%.4e nH=%.4e LDfac=%.6e kap_em=%.6e d_em=%.6e d_abs=%.6e fac_abs=%.4e result=%.6e\n",
        (unsigned long long)pp[i].ID, T, Tdust, nHcgs, LambdaDust_fac, kappa_emission, dust_emission, dust_absorption_rate, fac_abs, result_dedt);}
#endif
    return result_dedt;
}


/* ========================================================================
 * rt_eqm_dust_temp — equilibrium dust temperature solver
 * (moved from radiation/rt_utilities.cc, inside #ifdef COOLING)
 * ======================================================================== */

#ifdef COOLING
KOKKOS_INLINE_FUNCTION
double rt_eqm_dust_temp(int i, double T, double dust_absorption_rate, struct particle_data *pp, struct gas_cell_data *cell)
{
    double T_old, T_lower=0, T_upper=MAX_REAL_NUMBER, T_secant, Tdust_guess, Tdust, dEdt, dEdt_upper, dEdt_lower, fac, dEdt_guess, scalefac;
    double Tmax=1e10; // upper-bound dust temperature above which we definitely don't believe our detailed (tiny) dust abundance
    Tmax = MAX_DUST_TEMP; // this is now a global variable
    /* First we come up with a reasonable guess for the dust temp based on available info */
#ifdef RT_INFRARED
    Tdust_guess = DMIN(DMAX(cell[i].Dust_Temperature, 1.), MAX_DUST_TEMP); // previous dust temperature should be a good guess
#else // case where we don't have a pre-computed dust temp, use asymptotic limits to get a good guess
    double Zfac = 1.0;
#ifdef METALS
    if(i>=0) {Zfac = pp[i].Metallicity[0]/All.SolarAbundances[0];}
#endif
    double rho_c_arad_fac = (4.*5.67e-5)/(UNIT_VEL_IN_CGS*UNIT_PRESSURE_IN_CGS)*cell[i].Density*All.cf_a3inv; // a c rho in code units
    Tdust_guess = sqrt(cbrt(100 * dust_absorption_rate/(rho_c_arad_fac * (0.1*UNIT_SURFDEN_IN_CGS) * Zfac)));  // guess neglecting gas-dust coupling term and assuming a beta=2 emission opacity law kappa = 0.1 cm^2/g Z (T/10K)^2
    Tdust_guess = DMAX(Tdust_guess, sqrt(sqrt(dust_absorption_rate / (rho_c_arad_fac * (5.*UNIT_SURFDEN_IN_CGS) * Zfac)))); // account for how opacity tops out around 5 Z cm^2/g
#ifdef COOLING // account for gas-dust coupling
    double nHcgs = HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS * cell[i].Density * All.cf_a3inv / PROTONMASS_CGS;    /* hydrogen number dens in cgs units */
    double LambdaDust_fac = gas_dust_heating_coeff(i,T,Tdust_guess, pp, cell) * nHcgs * nHcgs /(UNIT_PRESSURE_IN_CGS/UNIT_TIME_IN_CGS);
    double Tdust_coupled = T - rho_c_arad_fac * rt_kappa_adaptive_IR_band(i,T,T,1,0, pp, cell) * pow(T,4) / (LambdaDust_fac+MIN_REAL_NUMBER); // bound for the gas-dust coupled regime assuming T ~ Td
    Tdust_guess = DMAX(Tdust_coupled, Tdust_guess);
#endif
#endif // end non-RT case for guess
    /* We now have our initial guess */
    if(T==0) {return Tdust_guess;} // if just calling for a rough estimate this is good enough

    Tdust = Tdust_guess;
    int n_iter=0;
    double fdustmet_init = return_dust_to_metals_ratio_vs_solar(i, Tdust, pp, cell); /* need this for reference but can't let it change over iterations */
    dEdt_guess = dEdt = dust_dEdt(i,T,Tdust_guess,dust_absorption_rate,fdustmet_init, pp, cell);

    if(dEdt==0){return Tdust_guess;}
    /* bracketing the dust temperature */
    if(dEdt < 0)
    {
	scalefac = 0.9;
	T_upper = DMIN(Tmax,Tdust), dEdt_upper = dEdt_guess;
	while(dEdt<0) {
	    Tdust *= scalefac;
	    dEdt = dust_dEdt(i,T,Tdust,dust_absorption_rate,fdustmet_init, pp, cell);
        if(dEdt==0){return Tdust;}
	    scalefac *= 0.9;
	    n_iter++;
	}
	T_lower = Tdust, dEdt_lower = dEdt;
    } else {
	T_lower = Tdust, dEdt_lower = dEdt_guess;
	scalefac = 1.1;
	while(dEdt>0 && Tdust < Tmax) {
	    Tdust *= scalefac; Tdust = DMIN(Tdust,Tmax);
	    dEdt = dust_dEdt(i,T,Tdust,dust_absorption_rate,fdustmet_init, pp, cell);
        if(dEdt==0){return Tdust;}
	    scalefac *= 1.1;
	    n_iter++;
	    }
	    T_upper = Tdust, dEdt_upper = dEdt;
    }
    if(T_upper==Tmax && dEdt_upper > 0) {return Tmax;}
    if(T_lower>=Tmax) {return Tmax;}

    /* secant-method Tdust iteration (a Brent-style bracketed rootfind here was
       tested but became unreliable in hyper-zoom-in runs with dust near max
       temperature; the secant method below is more robust for this case). */
    T_old = Tdust; double dEdt_old = dEdt; Tdust = Tdust_guess; dEdt = dEdt_guess; // For our second guess we take the backeting value opposite of the initial guess.
    double dT_dustgas = T-Tdust;
#ifdef GIZMO_DEBUG_RT_COOLING
    if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[TDUST_ITER] ID=%llu T=%.8e Tdust_guess=%.8e T_lower=%.8e T_upper=%.8e dEdt_guess=%.6e bracket_iters=%d\n", (unsigned long long)pp[i].ID, T, Tdust_guess, T_lower, T_upper, dEdt_guess, n_iter);}
#endif
    do  // secant method iterations with bisection as a backup; usually converges to machine epsilon in 4-5 iterations
    {
        dT_dustgas = T - Tdust;
        T_secant = Tdust - dEdt * (Tdust - T_old) / (dEdt - dEdt_old);
        T_secant = DMAX(DMIN(T_secant,T_upper),T_lower);
        dEdt_old = dEdt;
        dEdt = dust_dEdt(i,T,T_secant,dust_absorption_rate,fdustmet_init, pp, cell);
        fac = fabs(T_secant - Tdust)/(MIN_REAL_NUMBER+fabs(Tdust-T_old)); //fabs(dEdt)/(MIN_REAL_NUMBER+fabs(dEdt_old));
        if(fac < 0.5) { // accept the secant iteration if it is converging more rapidly
            T_old=Tdust;
            Tdust=T_secant;
#ifdef GIZMO_DEBUG_RT_COOLING
            if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[TDUST_ITER] ID=%llu iter=%d SECANT Tdust=%.10e dEdt=%.6e fac=%.4e T_lo=%.8e T_hi=%.8e\n", (unsigned long long)pp[i].ID, n_iter, Tdust, dEdt, fac, T_lower, T_upper);}
#endif
        } else { // if secant isn't working do bisection iteration instead; guaranteed to reduce the error
            T_old = Tdust;
            Tdust = sqrt(T_lower*T_upper);
            dEdt = dust_dEdt(i,T,Tdust,dust_absorption_rate,fdustmet_init, pp, cell);
            fac = 0.5;
#ifdef GIZMO_DEBUG_RT_COOLING
            if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[TDUST_ITER] ID=%llu iter=%d BISECT Tdust=%.10e dEdt=%.6e T_lo=%.8e T_hi=%.8e\n", (unsigned long long)pp[i].ID, n_iter, Tdust, dEdt, T_lower, T_upper);}
#endif
        }
        if(dEdt>0) {T_lower=Tdust;} else {T_upper=Tdust;} // either way, update upper and lower bounds
        n_iter++;
        if(n_iter > MAXITER-10) {
            PRINT_WARNING("Warning: Dust temperature iteration converging slowly: ID=%lld iter=%d T=%g Tdust=%g Tdust_guess=%g T_upper=%g T_lower=%g dEdt=%g fac=%g.\n",(long long)(long long)i /* particle index */,n_iter,T,Tdust,Tdust_guess, T_upper, T_lower,dEdt, fac);
            if(n_iter > MAXITER){break;}
        }
    } while(fabs(dT_dustgas - (T-Tdust)) > 1.e-3 * fabs(T-Tdust)); // sufficient to converge dust cooling to 10^-3 tolerance, at this point uncertainties in dust properties will dominate the error budget
#ifdef GIZMO_DEBUG_RT_COOLING
    if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[TDUST_ITER] ID=%llu FINAL Tdust=%.10e n_iter=%d\n", (unsigned long long)pp[i].ID, Tdust, n_iter);}
#endif

    return Tdust;
}


/* ========================================================================
 * get_equilibrium_dust_temperature_estimate — three-component CMB+ISRF+IR
 * equilibrium dust temperature estimator.
 *
 * Moved here (from cooling/cooling.cc) so eos_functions.h's
 * return_dust_to_metals_ratio_vs_solar can call it from device passes without
 * triggering nvc++ warning #20011-D (silent-physics on GPU). Body identical
 * to the prior cooling.cc copy; rt_kappa / rt_kappa_adaptive_IR_band /
 * rt_eqm_dust_temp are defined ABOVE in this same header so the inline body
 * has them in scope. evaluate_NH_from_GradRho is GIZMO_GPU_FUNCTION via
 * predict_functions.h and remains device-callable across TUs.
 *
 * cooling.cc still calls this function; the forward declarations in
 * cooling.cc need to be retired alongside this move (handled in same commit).
 * rt_utilities.cc remains the owner of the non-inline host external symbol
 * via its existing #undef KOKKOS_INLINE_FUNCTION re-include of rt_functions.h.
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
double get_equilibrium_dust_temperature_estimate(int i, double shielding_factor_for_exgalbg, double T, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(RT_INFRARED)
    if(i >= 0) {return cell[i].Dust_Temperature;} // this is pre-computed -- simply return it
#endif
    /* simple three-component model [can do fancier] with cmb, dust, high-energy photons */
    double e_CMB=0.262*All.cf_a3inv/All.cf_atime, T_cmb=2.73/All.cf_atime; // CMB [energy in eV/cm^3, T in K]
    double e_IR=0.31, Tdust_ext=DMAX(30.,T_cmb); // Milky way ISRF from Draine (2011), assume peak of dust emission at ~100 microns
    double e_HiEgy=0.66, T_hiegy=5800.; // Milky way ISRF from Draine (2011), assume peak of stellar emission at ~0.6 microns [can still have hot dust, this effect is pretty weak]
#ifdef RT_ISRF_BACKGROUND
    e_IR *= All.InterstellarRadiationFieldStrength; e_HiEgy *= All.InterstellarRadiationFieldStrength; // need to re-scale the assumed ISRF components
    if(!All.ComovingIntegrationOn){
	e_CMB *= pow(1.+All.RadiationBackgroundRedshift,4);
 	T_cmb *= (1.+All.RadiationBackgroundRedshift);
    }
#endif

    if(i >= 0)
    {
#ifdef SINGLE_STAR_SINK_DYNAMICS // treatment using direct dust temperature solver accounting for absorption and gas-dust coupling - want this when capturing the dynamics of dense collapsing cores
	double absorption_rate=0, vol_inv = cell[i].Density * All.cf_a3inv / cell[i].Mass, fac_abs = C_LIGHT_CODE * cell[i].Density * All.cf_a3inv;
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY) // we have information about individual radiation bands and their opacities; use these to compute dust absorption rate
	for(int k=0;k<N_RT_FREQ_BINS;k++){
	    if(RT_BAND_IS_IONIZING(k)) {continue;} // skip ionizing bands where the dust cross section is not accounted for
	    absorption_rate += fac_abs * rt_kappa(i,k, pp, cell) * cell[i].Rad_E_gamma_Pred[k] * vol_inv;
	}
#endif
	absorption_rate += (e_CMB/UNIT_PRESSURE_IN_EV) * fac_abs * rt_kappa_adaptive_IR_band(i,T_cmb,T_cmb,0,1, pp, cell); // CMB absorption; assume cloud is optically-thin to the CMB
#if defined(RT_ISRF_BACKGROUND) // account for additional optical + IR radiation field with extinction
	double column = evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i,pp); // column density in code units
	double kappa_IR = rt_kappa_adaptive_IR_band(i,20.,20.,0,1, pp, cell); // assume Trad=20 for IR dust opacity
	double Zfac = 1.;
#ifdef METALS
	Zfac = pp[i].Metallicity[0]/All.SolarAbundances[0];
#endif
	double kappa_opt = 180. * Zfac * UNIT_SURFDEN_IN_CGS;
	double tau_opt = kappa_opt*column;
	e_HiEgy += 7.8e-3 * pow(All.cf_atime,3.9)/(1.+pow(DMAX(-1.+1./All.cf_atime,0.001)/1.7,4.4)); // extragalactic UV/optical background
	absorption_rate += fac_abs * kappa_opt * (e_HiEgy/UNIT_PRESSURE_IN_EV) * exp(DMAX(-tau_opt,-100));
	absorption_rate += fac_abs * kappa_IR * ((-0.5*expm1(DMAX(-tau_opt,-100)) * e_HiEgy + e_IR)/UNIT_PRESSURE_IN_EV); // this assumes absorbed ONIR photons are reradiated into IR, factor of 0.5 assumes 1/2 of reradiated IR photons do not go deeper into the cloud
#endif
	// OK now we have our dust absorption rate, let's call the solver
	double Tdust = rt_eqm_dust_temp(i, T, absorption_rate, pp, cell);
	return Tdust;
#endif // SINGLE_STAR_SINK_DYNAMICS

#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY) // use actual explicitly-evolved radiation field, if possible
        e_HiEgy=0; e_IR = 0; int k; double E_tot_to_evol_eVcgs = (cell[i].Density*All.cf_a3inv/cell[i].Mass) * UNIT_PRESSURE_IN_EV;
        for(k=0;k<N_RT_FREQ_BINS;k++) {e_HiEgy+=cell[i].Rad_E_gamma_Pred[k];}
#if defined(GALSF_FB_FIRE_RT_LONGRANGE)
        e_IR += cell[i].Rad_E_gamma_Pred[RT_FREQ_BIN_FIRE_IR]; // note IR
#endif
#if defined(RT_INFRARED)
        e_IR += cell[i].Rad_E_gamma_Pred[RT_FREQ_BIN_INFRARED]; Tdust_ext = cell[i].Radiation_Temperature; // note IR [irrelevant b/c of call above, but we'll keep this as a demo]
#endif
        e_HiEgy -= e_IR; // don't double-count the IR component flagged above //
        e_IR *= E_tot_to_evol_eVcgs; e_HiEgy *= E_tot_to_evol_eVcgs;
#endif
    }
    e_HiEgy += shielding_factor_for_exgalbg * 7.8e-3 * pow(All.cf_atime,3.9)/(1.+pow(DMAX(-1.+1./All.cf_atime,0.001)/1.7,4.4)); // this comes from the cosmic optical+UV backgrounds. small correction, so treat simply, and ignore when self-shielded.
    double Tdust_eqm = 10.; // arbitrary initial value //
    if(Tdust_ext*e_IR < 1.e-10 * (T_cmb*e_CMB + T_hiegy*e_HiEgy)) { // IR term is totally negligible [or zero exactly], use simpler expression assuming constant temperature for it to avoid sensitivity to floating-pt errors //
        Tdust_eqm = 2.92 * pow(Tdust_ext*e_IR + T_cmb*e_CMB + T_hiegy*e_HiEgy, 1./5.); // approximate equilibrium temp assuming Q~1/lambda [beta=1 opacity law], assuming background IR temp is a fixed constant [relevant in IR-thin limit, but we don't know T_rad, so this is a guess anyways]
    } else { // IR term is not vanishingly small. we will assume the IR radiation temperature is equal to the local Tdust. lacking any direct evolution of that field, this is a good proxy, and exact in the locally-IR-optically-thick limit. in the locally-IR-thin limit it slightly under-estimates Tdust, but usually in that limit the other terms dominate anyways, so this is pretty safe //
        double T0=2.92, q=pow(T0*e_IR,0.25), y=(T_cmb*e_CMB + T_hiegy*e_HiEgy)/(T0*e_IR*q); if(y<=1) {Tdust_eqm=T0*q*(0.8+sqrt(0.04+0.1*y));} else {double y5=pow(y,0.2), y5_3=y5*y5*y5, y5_4=y5_3*y5; Tdust_eqm=T0*q*(1.+15.*y5_4+sqrt(1.+30.*y5_4+25.*y5_4*y5_4))/(20.*y5_3);} // this gives an extremely accurate and exactly-joined solution to the full quintic equation assuming T_rad_IR=T_dust
    }
#if defined(OUTPUT_DUST_TEMPERATURE) && (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    cell[i].Dust_Temperature = DMAX(DMIN(Tdust_eqm , 2000.) , 1.);
#endif
    return DMAX(DMIN(Tdust_eqm , 2000.) , 1.); // limit at sublimation temperature or some very low temp //
}
#endif /* COOLING */


/* ========================================================================
 * rt_irband_egydensity_in_band — photon energy density in a frequency band
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
double rt_irband_egydensity_in_band(int i, double E_lower, double E_upper, struct gas_cell_data *cell)
{
#if defined(RT_INFRARED)
    double u_gamma = cell[i].Rad_E_gamma[RT_FREQ_BIN_INFRARED] * (cell[i].Density*All.cf_a3inv/cell[i].Mass) * blackbody_lum_frac(E_lower, E_upper, cell[i].Radiation_Temperature);
    if(!isfinite(u_gamma) || (u_gamma<0)) {u_gamma = 0;}
    return u_gamma;
#else
    return 0;
#endif
}


/* ========================================================================
 * get_min_allowed_dustIRrad_temperature — minimum dust/radiation temperature
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

#ifdef RT_INFRARED
KOKKOS_INLINE_FUNCTION
double get_min_allowed_dustIRrad_temperature(void)
{
#if defined(GALSF)
    return DMAX(All.MinGasTemp, 2.73/All.cf_atime);
#endif
    return MIN_REAL_NUMBER;
}
#endif


/* ========================================================================
 * rt_kappa — total opacity for a given RT frequency bin
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
KOKKOS_INLINE_FUNCTION
double rt_kappa(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
{

#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
#ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION /* special test problem implementation */
    return cell[i].Interpolated_Opacity[k_freq] + 1.e-3 * All.Dust_to_Gas_Mass_Ratio*0.75*All.Grain_Q_at_MaxGrainSize/((All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS)*(All.Grain_Size_Max/UNIT_LENGTH_IN_CGS)); /* enforce minimum; note kappa in code units here so need to convert appropriately */
#endif
    return MIN_REAL_NUMBER + cell[i].Interpolated_Opacity[k_freq]; /* this is calculated in a different routine, just return it now */
#endif

#ifdef RT_CHEM_PHOTOION
    /* opacity to ionizing radiation for Petkova & Springel bands. note cooling.c or rt_update_chemistry is where ionization is actually calculated */
    double nH_over_Density = HYDROGEN_MASSFRAC / PROTONMASS_CGS * UNIT_MASS_IN_CGS;
    double kappa = nH_over_Density * (cell[i].HI + MIN_REAL_NUMBER) * All.rt_ion_sigma_HI[k_freq]; // note this is designed for specific applications: does not include dust, or free-free, or free-electron scattering contributions here, all of which can be important.
#if defined(RT_CHEM_PHOTOION_HE) && defined(RT_PHOTOION_MULTIFREQUENCY)
    kappa += nH_over_Density * ((cell[i].HeI + MIN_REAL_NUMBER) * All.rt_ion_sigma_HeI[k_freq] + (cell[i].HeII + MIN_REAL_NUMBER) * All.rt_ion_sigma_HeII[k_freq]);
    if(k_freq==RT_FREQ_BIN_He0)  {return kappa;}
    if(k_freq==RT_FREQ_BIN_He1)  {return kappa;}
    if(k_freq==RT_FREQ_BIN_He2)  {return kappa;}
#endif
    if(k_freq==RT_FREQ_BIN_H0)  {return kappa;}
#endif

#if defined(RT_HARD_XRAY) || defined(RT_SOFT_XRAY) || defined(RT_PHOTOELECTRIC) || defined (GALSF_FB_FIRE_RT_LONGRANGE) || defined(RT_NUV) || defined(RT_OPTICAL_NIR) || defined(RT_LYMAN_WERNER) || defined(RT_INFRARED) || defined(RT_FREEFREE)
    double fac = UNIT_SURFDEN_IN_CGS, Zfac, dust_to_metals_vs_standard, kappa_HHe; /* units */
    Zfac = 1.0; kappa_HHe=0.35; // assume solar metallicity, simple Thompson cross-section limit for various processes below
    dust_to_metals_vs_standard = return_dust_to_metals_ratio_vs_solar(i,0, pp, cell); // many of the dust opacities below will need this as the dimensionless dust-to-metals ratio normalized to the canonical Solar value of ~1/2
#ifdef METALS
    if(i>=0) {Zfac = pp[i].Metallicity[0]/All.SolarAbundances[0];}
#endif
#if defined(COOLING) && !defined(CHIMES)
    if(i>=0) {kappa_HHe=0.02 + 0.35*cell[i].Ne;}
#endif

#ifdef RT_FREEFREE /* pure (grey, non-relativistic) Thompson scattering opacity + free-free absorption opacity. standard expressions here from Rybicki & Lightman. */
    if(k_freq==RT_FREQ_BIN_FREEFREE)
    {
        double T_eff=0.59*(cell[i].gamma_eos_value()-1.)*U_TO_TEMP_UNITS*cell[i].InternalEnergyPred, rho=cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS; // we're assuming fully-ionized gas with a simple equation-of-state here, nothing fancy, to get the temperature //
        double kappa_abs = 1.e30*rho*pow(T_eff,-3.5);
        return (0.35 + kappa_abs) * fac;
    }
#endif
#ifdef RT_HARD_XRAY
    /* opacity comes from H+He (Thompson) + metal ions. expressions here for metal ions come from integrating over the standard Morrison & McCammmon 1983 metal cross-sections for a standard slope gamma in the band */
    if(k_freq==RT_FREQ_BIN_HARD_XRAY) {return (0.53 + 0.27*Zfac) * fac;}
#endif
#ifdef RT_SOFT_XRAY
    /* opacity comes from H+He (Thompson) + metal ions. expressions here for metal ions come from integrating over the standard Morrison & McCammmon 1983 metal cross-sections for a standard slope gamma in the band */
    if(k_freq==RT_FREQ_BIN_SOFT_XRAY) {return (127. + 50.0*Zfac) * fac;}
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    /* three-band (UV, OPTICAL, IR) approximate spectra for stars as used in the FIRE (Hopkins et al.) models. mean opacities here come from integrating over the Hopkins 2004 (Pei 1992) opacities versus wavelength for the large bands here, using a luminosity-weighted mean stellar spectrum from the same starburst99 models used to compute the stellar feedback */
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    // Use either MW (FIRE-3 default) or SMC (FIRE-2 default) opacities depending on the evolved local dust population composition
    // If silicate mass / carbonaceous mass >= 5 use SMC opacities, else MW opacities.
    double sil_to_C = 0;
    if (cell[i].ISMDustChem_Dust_Metal[0]>0 && cell[i].ISMDustChem_Dust_Species[1]>0)
    {sil_to_C = (cell[i].ISMDustChem_Dust_Metal[0] - cell[i].ISMDustChem_Dust_Species[1])/cell[i].ISMDustChem_Dust_Species[1];} // Everything that isn't carbonaceous dust is silicates for our purpose
    else {sil_to_C = 100;}
    if (sil_to_C >= 5)
    {
        if(k_freq==RT_FREQ_BIN_FIRE_UV)  {return DMAX(kappa_HHe, 1800.*(1.e-2 + Zfac*dust_to_metals_vs_standard)) * fac;}
        if(k_freq==RT_FREQ_BIN_FIRE_OPT) {return DMAX(kappa_HHe, 180.*(1.e-3 + Zfac*dust_to_metals_vs_standard)) * fac;}
        if(k_freq==RT_FREQ_BIN_FIRE_IR)  {return DMAX(kappa_HHe, 10*(1.e-3 + Zfac*dust_to_metals_vs_standard)) * fac;}
    }
#endif
    if(k_freq==RT_FREQ_BIN_FIRE_UV)  {return DMAX(kappa_HHe, 800.*(1.e-2 + Zfac*dust_to_metals_vs_standard)) * fac;}
    if(k_freq==RT_FREQ_BIN_FIRE_OPT) {return DMAX(kappa_HHe, 180.*(1.e-3 + Zfac*dust_to_metals_vs_standard)) * fac;}
    if(k_freq==RT_FREQ_BIN_FIRE_IR)  {return DMAX(kappa_HHe, 6.5*(1.e-3 + Zfac*dust_to_metals_vs_standard)) * fac;}
#endif
    if(k_freq==RT_FREQ_BIN_FIRE_UV)  {return (1800.) * fac;}
    if(k_freq==RT_FREQ_BIN_FIRE_OPT) {return (180.)  * fac;}
    if(k_freq==RT_FREQ_BIN_FIRE_IR)  {return (10.) * fac * (0.1 + Zfac);}
#endif
#ifdef RT_PHOTOELECTRIC
    if(k_freq==RT_FREQ_BIN_PHOTOELECTRIC) {return DMAX(kappa_HHe, 720.*DMAX(1.e-4,Zfac*dust_to_metals_vs_standard)) * fac;}
#endif
#ifdef RT_LYMAN_WERNER
    if(k_freq==RT_FREQ_BIN_LYMAN_WERNER) {return DMAX(kappa_HHe, 900.*Zfac*dust_to_metals_vs_standard) * fac;}
#endif
#ifdef RT_NUV
    if(k_freq==RT_FREQ_BIN_NUV) {return DMAX(kappa_HHe, 480.*Zfac*dust_to_metals_vs_standard) * fac;}
#endif
#ifdef RT_OPTICAL_NIR
    if(k_freq==RT_FREQ_BIN_OPTICAL_NIR) {return DMAX(kappa_HHe, 180.*Zfac*dust_to_metals_vs_standard) * fac;}
#endif
#ifdef RT_INFRARED
    /* IR with dust opacity */
    double T_min = get_min_allowed_dustIRrad_temperature();
    if(k_freq==RT_FREQ_BIN_INFRARED)
    {
        if(isnan(cell[i].Dust_Temperature)) {PRINT_WARNING("\n NaN dust temperature for cell-ID=%llu  \n", (unsigned long long) (long long)i /* particle index */); cell[i].Dust_Temperature = 1.e4;}
        if(isnan(cell[i].Radiation_Temperature)) {PRINT_WARNING("\n NaN gas temperature for cell-ID=%llu  \n", (unsigned long long) (long long)i /* particle index */);}
        if(cell[i].Dust_Temperature<=T_min) {cell[i].Dust_Temperature=T_min;}
        if(cell[i].Radiation_Temperature<=T_min) {cell[i].Radiation_Temperature=T_min;}
        double T_dust_em = cell[i].Dust_Temperature;
        double Trad = cell[i].Radiation_Temperature;
        if((Trad <= 0) || (T_dust_em<=0)) {PRINT_WARNING("\n Cell-ID=%llu  has T_rad=%g and T_dust=%g\n", (unsigned long long) (long long)i /* particle index */, Trad, T_dust_em);}
        return rt_kappa_adaptive_IR_band(i,T_dust_em,Trad,0,0, pp, cell);
    }
#endif
#endif

#ifdef NUCLEAR_NETWORK_NEUTRINOS
    if(k_freq==RT_FREQ_BIN_NU_E || k_freq==RT_FREQ_BIN_NU_EBAR || k_freq==RT_FREQ_BIN_NU_X) {
        extern double nuclear_neutrino_opacity(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell);
        return nuclear_neutrino_opacity(i, k_freq, pp, cell);
    }
#endif

    return 0;
}


/* ========================================================================
 * rt_absorb_frac_albedo — absorbed fraction = 1 - albedo
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
double rt_absorb_frac_albedo(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS) && defined(RT_GENERIC_USER_FREQ)
    if(k_freq==RT_FREQ_BIN_GENERIC_USER_FREQ) {return DMAX(1.e-6, DMIN(1.0 - 1.e-6, All.Grain_Absorbed_Fraction_vs_Total_Extinction));}
#endif

#ifdef RT_CHEM_PHOTOION
    if(k_freq==RT_FREQ_BIN_H0)  {return 1.-1.e-6;} /* negligible scattering for ionizing radiation */
#if defined(RT_CHEM_PHOTOION_HE) && defined(RT_PHOTOION_MULTIFREQUENCY)
    if(k_freq==RT_FREQ_BIN_He0 || k_freq==RT_FREQ_BIN_He1 || k_freq==RT_FREQ_BIN_He2)  {return 1.-1.e-6;}
#endif
#endif

#if defined(RT_HARD_XRAY) || defined(RT_SOFT_XRAY) || defined(RT_INFRARED) /* these have mixed opacities from dust(assume albedo=1/2), ionization(albedo=0), and Thompson (albedo=1) */
    double fac; fac = UNIT_SURFDEN_IN_CGS; /* units */
#ifdef RT_HARD_XRAY
    if(k_freq==RT_FREQ_BIN_HARD_XRAY) {return 1.-0.5*(0. + DMIN(1.,0.35*fac/rt_kappa(i,k_freq, pp, cell)));}
#endif
#ifdef RT_SOFT_XRAY
    if(k_freq==RT_FREQ_BIN_SOFT_XRAY) {return 1.-0.5*(0. + DMIN(1.,0.35*fac/rt_kappa(i,k_freq, pp, cell)));}
#endif
#ifdef RT_INFRARED
    if(k_freq==RT_FREQ_BIN_INFRARED) {return rt_kappa_adaptive_IR_band(i,cell[i].Dust_Temperature,cell[i].Radiation_Temperature,-1,0, pp, cell) / (MIN_REAL_NUMBER + rt_kappa_adaptive_IR_band(i,cell[i].Dust_Temperature,cell[i].Radiation_Temperature,0,0, pp, cell));}
#endif
#endif

#ifdef RT_FREEFREE
    if(k_freq==RT_FREQ_BIN_FREEFREE)
    {
        double T_eff=0.59*(cell[i].gamma_eos_value()-1.)*U_TO_TEMP_UNITS*cell[i].InternalEnergyPred, rho=cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS, kappa_abs = 1.e30*rho*pow(T_eff,-3.5);
        return kappa_abs / (0.35 + kappa_abs);
    }
#endif

#ifdef NUCLEAR_NETWORK_NEUTRINOS
    if(k_freq==RT_FREQ_BIN_NU_E || k_freq==RT_FREQ_BIN_NU_EBAR || k_freq==RT_FREQ_BIN_NU_X) {
        extern double nuclear_neutrino_absorb_frac(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell);
        return nuclear_neutrino_absorb_frac(i, k_freq, pp, cell);
    }
#endif

    return 0.5; /* default to assuming kappa_scattering = kappa_absorption (pretty reasonable for dust at most wavelengths) */
}


/* ========================================================================
 * rt_absorption_rate — photon absorption rate (absorptions per unit time per photon)
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
double rt_absorption_rate(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* should be equal to (c_reduced * Kappa_opacity * rho) */
    return (C_LIGHT_CODE_REDUCED) * rt_absorb_frac_albedo(i,k_freq, pp, cell) * (rt_kappa(i,k_freq, pp, cell) * cell[i].Density*All.cf_a3inv);
}

#endif /* RADTRANSFER || RT_USE_GRAVTREE */


/* ========================================================================
 * rt_get_donation_target_bin — which bin absorbed radiation is re-emitted into
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

KOKKOS_INLINE_FUNCTION
int rt_get_donation_target_bin(int bin)
{
    int donation_target_bin = -1;
#if defined(RT_CHEM_PHOTOION) && defined(RT_OPTICAL_NIR)
    if(bin==RT_FREQ_BIN_H0) {donation_target_bin=RT_FREQ_BIN_OPTICAL_NIR;}
#ifdef RT_PHOTOION_MULTIFREQUENCY
    if(bin==RT_FREQ_BIN_He0) {donation_target_bin=RT_FREQ_BIN_OPTICAL_NIR;}
    if(bin==RT_FREQ_BIN_He1) {donation_target_bin=RT_FREQ_BIN_OPTICAL_NIR;}
    if(bin==RT_FREQ_BIN_He2) {donation_target_bin=RT_FREQ_BIN_OPTICAL_NIR;}
#endif
#endif
#if defined(RT_PHOTOELECTRIC) && defined(RT_INFRARED)
    if(bin==RT_FREQ_BIN_PHOTOELECTRIC) {donation_target_bin=RT_FREQ_BIN_INFRARED;}
#endif
#if defined(RT_NUV) && defined(RT_INFRARED)
    if(bin==RT_FREQ_BIN_NUV) {donation_target_bin=RT_FREQ_BIN_INFRARED;}
#endif
#if defined(RT_OPTICAL_NIR) && defined(RT_INFRARED)
    if(bin==RT_FREQ_BIN_OPTICAL_NIR) {donation_target_bin=RT_FREQ_BIN_INFRARED;}
#endif
    return donation_target_bin;
}


/* ========================================================================
 * dust_dE_cooling — dust energy balance for RT_INFRARED solver
 * (moved from radiation/rt_utilities.cc)
 * ======================================================================== */

#ifdef RT_INFRARED
KOKKOS_INLINE_FUNCTION
double dust_dE_cooling(int i, double Tgas, double Tdust, double* Tdust_fixedpoint_1, double* Tdust_fixedpoint_2, struct particle_data *pp, struct gas_cell_data *cell){
    double dt = get_particle_timestep_in_physical(i, pp);
#ifdef TRANSPORT_SUBCYCLE_COOLING
    dt *= All.Transport_Subcycle_dt_fraction; /* cooling is called N times per hydro step, each with dt/N — projections here must match */
#endif
    double nHcgs = HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS * cell[i].Density * All.cf_a3inv / PROTONMASS_CGS;
    double lambda_to_dErad = (C_LIGHT_CODE_REDUCED/C_LIGHT_CODE) * nHcgs * nHcgs * (dt*UNIT_TIME_IN_CGS) / (cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS) / (UNIT_SPECEGY_IN_CGS) * cell[i].Mass; /* need to account for RSOL factors in emission/absorption rates */
    
    double dust_absorption_nonIR = 0;
    for(int k=0; k < N_RT_FREQ_BINS; k++){
#ifdef RT_CHEM_PHOTOION
        if(RT_BAND_IS_IONIZING(k)) {continue;} /* gas-phase absorption */
#endif
        if(k==RT_FREQ_BIN_INFRARED) {continue;} /* this is only counting up non-IR contributions, e.g. nebular NUV */
        double e_final = cell[i].Rad_E_gamma[k] + cell[i].Lambda_RadiativeCooling_toRHDBins[k] * lambda_to_dErad;
        e_final = DMAX(0,e_final); // check against overshoot into negative values
        double absrate_k = rt_absorption_rate(i, k, pp, cell) * dt; // this needs to be positive to sensible behavior here
        if(absrate_k > 0) {dust_absorption_nonIR += e_final * fabs(expm1(-absrate_k));}
    }
    double alpha_gd = 0;
#ifdef COOLING
    alpha_gd = gas_dust_heating_coeff(i,Tgas,Tdust, pp, cell);
#endif
    double LambdaDust = alpha_gd * (Tgas-Tdust);
    double de_IR_dust = LambdaDust * lambda_to_dErad; // equates to *net* emission of radiation by dust (emission - absorption)
    double LambdaIR_gas = cell[i].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED];
    double de_IR_gas = LambdaIR_gas * lambda_to_dErad; // net emission by gas
    
    double kappa_dust_emission = rt_kappa_adaptive_IR_band(i, Tdust, Tdust, 1,1, pp, cell);
    double fac_emission = 4.*5.67e-5/(UNIT_PRESSURE_IN_CGS*UNIT_VEL_IN_CGS)*cell[i].Mass*(C_LIGHT_CODE_REDUCED/C_LIGHT_CODE)*dt;
    double dust_emission = fac_emission*kappa_dust_emission*pow(Tdust,4); // *total* dust emission
    
    double T_IR_0 = cell[i].Radiation_Temperature;
    double Tmax = DMAX(DMAX(Tgas, Tdust),T_IR_0), Tmin = DMIN(DMIN(Tgas,Tdust), T_IR_0);
    double e_IR_0 = cell[i].Rad_E_gamma[RT_FREQ_BIN_INFRARED];
    double e_IR_final = DMAX(e_IR_0 + de_IR_dust + de_IR_gas, 0);
    double T_IR_final = e_IR_final / (DMAX(e_IR_0/T_IR_0,0) + DMAX(de_IR_gas / Tgas,0) + DMAX(de_IR_dust / Tdust,0) + MIN_REAL_NUMBER);
    T_IR_final = DMAX(Tmin, DMIN(T_IR_final, Tmax));
#ifdef COOLING
    cell[i].Radiation_Temperature_CoolingWeighted = T_IR_final;
#endif
    double dE_dust = 0; // now count up the energy changes in the dust for us to solve for 0
    double dust_absorption = dust_absorption_nonIR;
    dust_absorption += e_IR_final * C_LIGHT_CODE_REDUCED * rt_kappa_adaptive_IR_band(i, Tdust, T_IR_final,-1,1, pp, cell) * cell[i].Density*All.cf_a3inv * dt;
    double result = LambdaDust * lambda_to_dErad + dust_absorption - dust_emission;

    double Tdust_fixed1_tmp = Tgas + (dust_absorption - dust_emission)/(alpha_gd*lambda_to_dErad + MIN_REAL_NUMBER);
    double Tdust_fixed2_tmp = sqrt(sqrt(DMAX(0,LambdaDust * lambda_to_dErad + dust_absorption)/(fac_emission * kappa_dust_emission + MIN_REAL_NUMBER)));
    
    *Tdust_fixedpoint_1 = DMAX(Tdust_fixed1_tmp, 0);
    *Tdust_fixedpoint_2 = DMAX(Tdust_fixed2_tmp, 0);
    return result;
}

/* rt_ir_lambdadust — dust cooling rate solver for RT_INFRARED
 * (moved from radiation/rt_utilities.cc) */
KOKKOS_INLINE_FUNCTION
double rt_ir_lambdadust(int i, double T, struct particle_data *pp, struct gas_cell_data *cell){
    double Tdust, T_lower, T_upper, dE, dE1, dE2, dE_lower, dE_upper, dE_guess, dTdust_tol=1e-6;
    double Tdust_fixedpoint_1, Tdust_fixedpoint_2, dummy;
    auto dust_dE_rootfn = [&](double dTdust) {
        return dust_dE_cooling(i, T, T+dTdust, &Tdust_fixedpoint_1, &Tdust_fixedpoint_2, pp, cell);
    };
    if((All.Time==0 )|| (!isfinite(cell[i].Dust_Temperature))) {Tdust=T;} else {Tdust = DMIN(MAX_DUST_TEMP, cell[i].Dust_Temperature);}

    dE = dE_guess = dust_dE_rootfn(Tdust-T);
    if(Tdust_fixedpoint_1 <= 0) {Tdust_fixedpoint_1 = T;}
    if(Tdust_fixedpoint_1 > MAX_DUST_TEMP) {Tdust_fixedpoint_1 = MAX_DUST_TEMP;}
    if(Tdust_fixedpoint_2 > MAX_DUST_TEMP) {Tdust_fixedpoint_2 = MAX_DUST_TEMP;}
    if(Tdust_fixedpoint_1 > 0 && Tdust_fixedpoint_1 <= MAX_DUST_TEMP) {dE1 =  dust_dE_cooling(i, T, Tdust_fixedpoint_1, &dummy, &dummy, pp, cell);} else {dE1 = MAX_REAL_NUMBER;}
    if(Tdust_fixedpoint_2 > 0 && Tdust_fixedpoint_2 <= MAX_DUST_TEMP) {dE2 =  dust_dE_cooling(i, T, Tdust_fixedpoint_2, &dummy, &dummy, pp, cell);} else {dE2 = MAX_REAL_NUMBER;}

    double fixedpoint_error = DMIN(fabs(Tdust-Tdust_fixedpoint_2), fabs(Tdust-Tdust_fixedpoint_1))/Tdust; 
    if(fabs(dE1) < fabs(dE)){Tdust = Tdust_fixedpoint_1; dE=dE_guess=dE1;}
    if(fabs(dE2) < fabs(dE)){Tdust = Tdust_fixedpoint_2; dE=dE_guess=dE2;}
    
    int n_iter = 0;
    if(dE < 0)
    {
        double scalefac = DMAX(0.9, 1-fixedpoint_error);
        T_upper = Tdust;
        dE_upper = dE_guess; 
        while(dE < 0) {
            Tdust *= scalefac; 
            dE = dust_dE_rootfn(Tdust-T);
            if(dE==0){break;}
            scalefac *= 0.9; 
            n_iter++;
        }
        T_lower = Tdust, dE_lower = dE;
    } else {
        T_lower = Tdust, dE_lower = dE_guess;
        double scalefac = DMIN(1.1, 1+fixedpoint_error);
        while(dE > 0 && Tdust < MAX_DUST_TEMP) {
            Tdust *= scalefac; Tdust = DMIN(Tdust,MAX_DUST_TEMP);
            dE = dust_dE_rootfn(Tdust-T); 
            if(dE==0){break;}
            scalefac *= 1.1; 
            n_iter++;
        }
        T_upper = Tdust, dE_upper = dE;
    }     
    if(T_upper>=MAX_DUST_TEMP && dE_upper > 0) {cell[i].Dust_Temperature = MAX_DUST_TEMP; return 0;}

    if(dE_lower * dE_upper > 0) {PRINT_WARNING("Failed to bracket Tdust solution for ID=%lld T=%g T_lower=%g T_upper=%g dE_lower=%g dE_upper=%g\n", (long long)(long long)i /* particle index */, T, T_lower,T_upper, dE_lower, dE_upper);}

#ifdef GIZMO_DEBUG_RT_COOLING
    if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[LAMBDADUST] ID=%llu T=%.8e Tdust_bracket=[%.8e,%.8e] dE_bracket=[%.6e,%.6e] n_bracket=%d\n", (unsigned long long)pp[i].ID, T, T_lower, T_upper, dE_lower, dE_upper, n_iter);}
#endif
    if(dE!=0){
        BrentRootfindResult rfr = brent_rootfind(dust_dE_rootfn, T_lower-T, T_upper-T, dE_lower, dE_upper, dTdust_tol, 0., MAXITER);
        Tdust = rfr.x + T;
#ifdef GIZMO_DEBUG_RT_COOLING
        if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[LAMBDADUST] ID=%llu rootfind Tdust=%.10e iter=%d\n", (unsigned long long)pp[i].ID, Tdust, rfr.iter);}
#endif
    }
    double LambdaDust = 0;
#ifdef COOLING
    LambdaDust = gas_dust_heating_coeff(i,T,Tdust, pp, cell) * (T-Tdust);
#endif
#ifdef GIZMO_DEBUG_RT_COOLING
    if(pp[i].ID == 1 || pp[i].ID == 100 || pp[i].ID == 1000) {printf("[LAMBDADUST] ID=%llu FINAL Tdust=%.10e LambdaDust=%.10e Trad_CW=%.10e\n", (unsigned long long)pp[i].ID, Tdust, LambdaDust, cell[i].Radiation_Temperature_CoolingWeighted);}
#endif
    cell[i].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] += LambdaDust;
    cell[i].Dust_Temperature = Tdust;
    return LambdaDust;
}
#endif /* RT_INFRARED */


/* ========================================================================
 * slab_averaging_function — (1 - exp(-x))/x, smooth fit valid all x.
 * Used by hydro_toplevel.cc + rt_update_driftkick + cooling. Pure function.
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
double slab_averaging_function(double x)
{
    /* this fitting function is accurate to ~0.1% at all x, and extrapolates to the correct limits without the pathological behaviors of (1-exp(-x))/x for very small/large x */
    return (1.00000000000 + x*(0.21772719088733913 + x*(0.047076512011644776 + x*0.005068307557496351))) /
           (1.00000000000 + x*(0.71772719088733920 + x*(0.239273440788647680 + x*(0.046750496137263675 + x*0.005068307557496351))));
}


/* ========================================================================
 * check_if_absorbed_photons_can_be_reemitted_into_same_band — compile-time
 * gate per band controlling whether RP can exceed absorbed photon momentum.
 * Pure function. Independent of RADTRANSFER (called from hydro/cooling
 * paths outside the RADTRANSFER guard).
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
int check_if_absorbed_photons_can_be_reemitted_into_same_band(int kfreq)
{
    int checker = 0; // default to no, but this isn't always true
#if defined(GALSF_FB_FIRE_RT_LONGRANGE)
    if(kfreq==RT_FREQ_BIN_FIRE_IR) {checker=1;} // skip
#endif
#if defined(RT_FREEFREE)
    if(kfreq==RT_FREQ_BIN_FREEFREE) {checker=1;} // skip
#endif
#if defined(RT_GENERIC_USER_FREQ)
    if(kfreq==RT_FREQ_BIN_GENERIC_USER_FREQ) {checker=1;} // skip
#endif
#if defined(RT_INFRARED)
    if(kfreq==RT_FREQ_BIN_INFRARED) {checker=1;} // skip
#endif
    return checker;
}


#ifdef RADTRANSFER
/* ========================================================================
 * rt_eddington_update_calculation — compute the Eddington tensor under
 * whichever closure is active (M1, FLD, or default isotropic). OTVET path
 * returns immediately because that closure is built elsewhere (gravity).
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
void rt_eddington_update_calculation(int j, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef RT_OTVET
    return; /* eddington tensor is calculated elsewhere [in the gravity subroutine]: don't mess with it here! */
#endif
#if defined(RT_M1) && !defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
    /* calculate the eddington tensor with the M1 closure */
    int k_freq, k; double fmag_j, V_j_inv = cell[j].Density / pp[j].Mass; Vec3<double> n_flux_j;
    for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
    {
        n_flux_j = {};
        Vec3<double> flux_vol = cell[j].Rad_Flux[k_freq] * V_j_inv;
        fmag_j = flux_vol.norm_sq();
        if(fmag_j <= 0) {fmag_j=0;} else {fmag_j=sqrt(fmag_j); n_flux_j = flux_vol/fmag_j;}
        double f_chifac = fmag_j / (MIN_REAL_NUMBER + C_LIGHT_CODE_REDUCED * cell[j].Rad_E_gamma[k_freq] * V_j_inv);
        if(f_chifac < 0) {f_chifac=0;}
        if(fmag_j <= 0) {f_chifac = 0;}
        // restrict values of f_chifac to physical range.
        double f_min = 0.01, f_max = 0.9999;
        if(f_chifac < f_min) {f_chifac = f_min;}
        if(f_chifac > f_max) {f_chifac = f_max;}
        double chi_j = (3.+4.*f_chifac*f_chifac) / (5. + 2.*sqrt(4. - 3.*f_chifac*f_chifac));
        double chifac_iso_j = 0.5 * (1.-chi_j);
        double chifac_n_j = 0.5 * (3.*chi_j-1.);
        for(k=0;k<3;k++) for(int k2=k;k2<3;k2++) {cell[j].ET[k_freq][k][k2] = chifac_n_j * n_flux_j[k]*n_flux_j[k2];}
        cell[j].ET[k_freq][0][0] += chifac_iso_j; cell[j].ET[k_freq][1][1] += chifac_iso_j; cell[j].ET[k_freq][2][2] += chifac_iso_j;
    }
    return;
#endif
#ifdef RT_FLUXLIMITEDDIFFUSION
    /* always assume the isotropic eddington tensor */
    int k_freq; for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++) {cell[j].ET[k_freq].set_isotropic(1./3.);}
    return;
#endif

    /* if nothing is set, default to guess the isotropic eddington tensor */
    {int k_freq; for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++) {cell[j].ET[k_freq].set_isotropic(1./3.);}}
    return;
}
#endif /* RADTRANSFER */


#if defined(RT_ISRF_BACKGROUND) && defined(RADTRANSFER)
/* ========================================================================
 * get_background_isrf_urad — ISRF energy density per band, code units.
 * Reads All.RadiationBackgroundRedshift, All.InterstellarRadiationFieldStrength.
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
void get_background_isrf_urad(int i, double *urad){
    int k;
    for(k = 0; k < N_RT_FREQ_BINS; k++)
    {
        urad[k] = MIN_REAL_NUMBER;
#ifdef RT_INFRARED
	double fac_uCMB = 1.;
	fac_uCMB = pow(1+All.RadiationBackgroundRedshift, 4);
        if(k==RT_FREQ_BIN_INFRARED){urad[k] = (All.InterstellarRadiationFieldStrength * 0.39 + fac_uCMB * 0.26) * ELECTRONVOLT_IN_ERGS / UNIT_PRESSURE_IN_CGS;} // 0.33 eV/cm^3 is dust emission peak, 0.26 is CMB - note how this bin actually lumps the two together
#endif
#ifdef RT_OPTICAL_NIR
        if(k==RT_FREQ_BIN_OPTICAL_NIR){urad[k] = All.InterstellarRadiationFieldStrength * 0.54 * ELECTRONVOLT_IN_ERGS / UNIT_PRESSURE_IN_CGS;} // stellar emission
#endif
#ifdef RT_NUV
        if(k==RT_FREQ_BIN_NUV){urad[k] = All.InterstellarRadiationFieldStrength * 0.024 * ELECTRONVOLT_IN_ERGS / UNIT_PRESSURE_IN_CGS;} // stellar emission
#endif
#ifdef RT_PHOTOELECTRIC
        if(k==RT_FREQ_BIN_PHOTOELECTRIC){urad[k] = All.InterstellarRadiationFieldStrength * 1.7  / UNIT_EGY_DENSITY_IN_HABING;} // Draine 1978 value = 1.7 Habing
#endif
    }
}


/* ========================================================================
 * background_isrf_cmb_Teff — energy-weighted effective temperature of the
 * background ISRF+CMB sum (for the lumped IR band).
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
double background_isrf_cmb_Teff(void){
    // Returns the energy-weighted effective temperature of the background ISRF that has equivalent average photon energy to the sum of the ISRF and CMB
    // Necessary because current IR band treatment lumps both radiation fields together
    double urad_ISRF_CGS_eV = All.InterstellarRadiationFieldStrength * 0.39, Trad_ISRF = 100.;
#ifdef RT_INFRARED
    Trad_ISRF = DMIN(All.InitRadiationTemp, 100.);
#endif
    double fac_TCMB= 1.+All.RadiationBackgroundRedshift, fac_uCMB = pow(fac_TCMB,4);
    double urad_CMB_CGS_eV = fac_uCMB * 0.262, Trad_CMB = 2.73 * fac_TCMB;
    return (urad_ISRF_CGS_eV * Trad_ISRF + urad_CMB_CGS_eV * Trad_CMB) / (urad_ISRF_CGS_eV + urad_CMB_CGS_eV); // weighting by SED energy
}


/* ========================================================================
 * rt_apply_boundary_conditions — at outer 10% of box, clamp Rad_E_gamma and
 * (IR) Radiation/Dust temperatures to the background ISRF+CMB.
 * Calls get_background_isrf_urad + background_isrf_cmb_Teff above.
 * ======================================================================== */
KOKKOS_INLINE_FUNCTION
void rt_apply_boundary_conditions(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double urad[N_RT_FREQ_BINS]; int k, k_dir;
    get_background_isrf_urad(i, urad);
    // if we are within 10% of the box length of the edge:
    if(DMAX(DMAX(pp[i].Pos[0],pp[i].Pos[1]),pp[i].Pos[2]) > 0.9*All.BoxSize || DMIN(DMIN(pp[i].Pos[0],pp[i].Pos[1]),pp[i].Pos[2]) < 0.1*All.BoxSize)
    {
        for(k = 0; k < N_RT_FREQ_BINS; k++)
        {
            cell[i].Rad_E_gamma[k] = urad[k] * cell[i].Mass/(cell[i].Density * All.cf_a3inv);
#ifdef RT_EVOLVE_FLUX
            for(k_dir = 0; k_dir < 3; k_dir++){cell[i].Rad_Flux[k][k_dir] = 0;}
#endif
#ifdef RT_INFRARED
            if(k==RT_FREQ_BIN_INFRARED) {
                cell[i].Radiation_Temperature = background_isrf_cmb_Teff();
                cell[i].Dust_Temperature = DMIN(All.InitRadiationTemp,100.);
            }
#endif
        }
    } else {
        for(k = 0; k < N_RT_FREQ_BINS; k++){cell[i].Rad_E_gamma[k] = DMAX(cell[i].Rad_E_gamma[k], MIN_REAL_NUMBER);}
    }
}
#endif /* RT_ISRF_BACKGROUND && RADTRANSFER */


