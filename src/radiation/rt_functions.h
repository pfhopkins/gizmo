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
/* Neutrino opacity / absorbed-fraction / Ye-feedback bodies, called below from
   rt_kappa, rt_absorb_frac_albedo and rt_update_driftkick. Defined inline there
   so the device pass has a definition rather than an unresolvable extern call. */
#include "../nuclear/nuclear_physics_functions.h"

/* Forward decl: rt_kappa is defined at ~L522, but called earlier in this
 * file from rt_eqm_dust_temp (under SINGLE_STAR_SINK_DYNAMICS + RADTRANSFER,
 * e.g. SSP_HYBRID Config). cooling.cc includes rt_functions.h BEFORE
 * proto.h (see cooling.cc:15-20 comment on nvcc execution-space attribute
 * precedence), so the proto.h:829 host-only forward decl isn't visible at
 * the call site either. Same gate as the definition.
 * Phase D fix 2026-05-21 (Config 31 SINGLE_STAR_AND_SSP_HYBRID_MODEL_DEFAULTS). */
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
KOKKOS_INLINE_FUNCTION double rt_kappa(int i, int k_freq, struct particle_data *pp, struct gas_cell_data *cell);
#endif

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
    Trad = DMAX(Trad, MIN_REAL_NUMBER); // guard against non-positive Trad reaching log10() below: negative values cause log10=NaN, which propagates into an unguarded table-index cast further down (dust_planck_mean_opacity) and segfaults
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
        double Tgas=1. + cell[i].gas_temperature_from_u(cell[i].InternalEnergyPred), rho_cgs = cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS; /* crude estimate of gas temperature to use with scalings below, and gas density in cgs */
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
        double u_in=cell[i].InternalEnergy, rho_in=cell[i].Density*All.cf_a3inv, mu=1, ne=1, nHI=0, nHII=0;
        double temp = ThermalProperties(u_in, rho_in, i, &mu, &ne, &nHI, &nHII, pp, cell);
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
	double Tdust_floor = 2.73 / All.cf_atime; // CMB temperature: dust cannot radiatively cool below the ambient radiation bath, so floor the bracket search here
	T_upper = DMIN(Tmax,Tdust), dEdt_upper = dEdt_guess;
	while(dEdt<0 && Tdust > Tdust_floor && n_iter < MAXITER) {
	    Tdust *= scalefac; Tdust = DMAX(Tdust,Tdust_floor);
	    dEdt = dust_dEdt(i,T,Tdust,dust_absorption_rate,fdustmet_init, pp, cell);
        if(dEdt==0){return Tdust;}
	    scalefac *= 0.9;
	    n_iter++;
	}
	if(dEdt < 0) { // could not bracket downward: equilibrium dust temp is at/below the radiation-bath floor, or we hit the iteration cap -- return the floored value (warn only on the cap, the floor is a physical outcome)
	    if(n_iter >= MAXITER) {PRINT_WARNING("Dust temperature bracketing (cooling side) failed to converge: ID=%lld iter=%d T=%g Tdust=%g Tfloor=%g dEdt=%g.\n",(long long)i,n_iter,T,Tdust,Tdust_floor,dEdt);}
	    return DMAX(Tdust,Tdust_floor);
	}
	T_lower = Tdust, dEdt_lower = dEdt;
    } else {
	T_lower = Tdust, dEdt_lower = dEdt_guess;
	scalefac = 1.1;
	while(dEdt>0 && Tdust < Tmax && n_iter < MAXITER) {
	    Tdust *= scalefac; Tdust = DMIN(Tdust,Tmax);
	    dEdt = dust_dEdt(i,T,Tdust,dust_absorption_rate,fdustmet_init, pp, cell);
        if(dEdt==0){return Tdust;}
	    scalefac *= 1.1;
	    n_iter++;
	    }
	    if(n_iter >= MAXITER && dEdt > 0) {PRINT_WARNING("Dust temperature bracketing (heating side) failed to converge: ID=%lld iter=%d T=%g Tdust=%g dEdt=%g.\n",(long long)i,n_iter,T,Tdust,dEdt); return DMIN(Tdust,Tmax);}
	    T_upper = Tdust, dEdt_upper = dEdt;
    }
    if(T_upper==Tmax && dEdt_upper > 0) {return Tmax;}
    if(T_lower>=Tmax) {return Tmax;}

    /* secant-method Tdust iteration (a Brent-style bracketed rootfind here was
       tested but became unreliable in hyper-zoom-in runs with dust near max
       temperature; the secant method below is more robust for this case). */
    T_old = Tdust; double dEdt_old = dEdt; Tdust = Tdust_guess; dEdt = dEdt_guess; // For our second guess we take the backeting value opposite of the initial guess.
    double dT_dustgas = T-Tdust;
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
        } else { // if secant isn't working do bisection iteration instead; guaranteed to reduce the error
            T_old = Tdust;
            Tdust = sqrt(T_lower*T_upper);
            dEdt = dust_dEdt(i,T,Tdust,dust_absorption_rate,fdustmet_init, pp, cell);
            fac = 0.5;
        }
        if(dEdt>0) {T_lower=Tdust;} else {T_upper=Tdust;} // either way, update upper and lower bounds
        n_iter++;
        if(n_iter > MAXITER-10) {
            PRINT_WARNING("Warning: Dust temperature iteration converging slowly: ID=%lld iter=%d T=%g Tdust=%g Tdust_guess=%g T_upper=%g T_lower=%g dEdt=%g fac=%g.\n",(long long)(long long)i /* particle index */,n_iter,T,Tdust,Tdust_guess, T_upper, T_lower,dEdt, fac);
            if(n_iter > MAXITER){break;}
        }
    } while(fabs(dT_dustgas - (T-Tdust)) > 1.e-3 * fabs(T-Tdust)); // sufficient to converge dust cooling to 10^-3 tolerance, at this point uncertainties in dust properties will dominate the error budget

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
        double T_eff=cell[i].gas_temperature_from_u(cell[i].InternalEnergyPred), rho=cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS; // temperature from the composition the cooling solve stored, or the ionized fallback where none was, nothing fancy, to get the temperature //
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
        double T_eff=cell[i].gas_temperature_from_u(cell[i].InternalEnergyPred), rho=cell[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS, kappa_abs = 1.e30*rho*pow(T_eff,-3.5);
        return kappa_abs / (0.35 + kappa_abs);
    }
#endif

#ifdef NUCLEAR_NETWORK_NEUTRINOS
    if(k_freq==RT_FREQ_BIN_NU_E || k_freq==RT_FREQ_BIN_NU_EBAR || k_freq==RT_FREQ_BIN_NU_X) {
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

    if(dE!=0){
        BrentRootfindResult rfr = brent_rootfind(dust_dE_rootfn, T_lower-T, T_upper-T, dE_lower, dE_upper, dTdust_tol, 0., MAXITER);
        Tdust = rfr.x + T;
    }
    double LambdaDust = 0;
#ifdef COOLING
    LambdaDust = gas_dust_heating_coeff(i,T,Tdust, pp, cell) * (T-Tdust);
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


#ifdef RADTRANSFER
/***********************************************************************************************************/
KOKKOS_INLINE_FUNCTION
void rt_update_driftkick(int i, double dt_entr, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(RT_EVOLVE_ENERGY) || defined(RT_EVOLVE_INTENSITIES)
    int kf, k_tmp; double total_erad_emission_minus_absorption = 0;
#if defined(RT_EVOLVE_INTENSITIES)
    for(kf=0;kf<N_RT_FREQ_BINS;kf++) {cell[i].Rad_E_gamma[kf]=0; for(k_tmp=0;k_tmp<N_RT_INTENSITY_BINS;k_tmp++) {cell[i].Rad_E_gamma[kf]+=RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity[kf][k_tmp];}}
    double de_emission_minus_absorption_saved[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS]; // save this for use below
#endif
#ifdef RT_INFRARED
    double E_abs_tot_toIR = 0;/* energy absorbed in other bands is transfered to IR, by default: track it here */
    double vol_inv_phys=(cell[i].Density*All.cf_a3inv/cell[i].Mass);
#if !defined(COOLING) // if cooling is active, don't reset this here, because it needs to include the gas coupling term which will be self-consistently calculated there
    double Rad_E_gamma_tot = 0; // dust temperature defined by total radiation energy density //
    {int j; for(j=0;j<N_RT_FREQ_BINS;j++) {Rad_E_gamma_tot += cell[i].Rad_E_gamma[j];}}
    double a_rad_inverse=C_LIGHT_CGS/(4.*5.67e-5), u_gamma = Rad_E_gamma_tot * vol_inv_phys * UNIT_PRESSURE_IN_CGS; // photon energy density in CGS //
    double Dust_Temperature_4 = u_gamma * a_rad_inverse; // estimated effective temperature of local rad field in equilibrium with dust emission. note that for our definitions, rad energy density has its 'normal' value independent of RSOL, so Tdust should as well; emission -and- absorption are both lower by a factor of RSOL, but these cancel in the Tdust4 here //
    cell[i].Dust_Temperature = sqrt(sqrt(Dust_Temperature_4)); // just set this to the local radiation equilibrium temperature
#endif
    double T_min = get_min_allowed_dustIRrad_temperature();
    if(cell[i].Dust_Temperature <= T_min) {cell[i].Dust_Temperature = T_min;} // dust temperature shouldn't be below CMB
    double IRBand_opacity_fraction_from_gas_absorption = 0; // needed below to know what fraction is immediately re-radiated or not
    double T_gas = cell[i].Dust_Temperature;
#ifdef COOLING
    T_gas = cell[i].temperature();
#endif
#endif

    for(k_tmp=0; k_tmp<N_RT_FREQ_BINS; k_tmp++)
    {
        kf = k_tmp; // band evaluated on this pass; swapped just below if IR is not the final bin
#ifdef RT_INFRARED
        /* IR must be evaluated last, after absorption from the other bands has been summed
           into E_abs_tot_toIR. IR is the final bin in every band layout the code currently
           builds, so this swap does nothing today; it keeps the ordering correct if a future
           layout puts another band after it. Exchange the positions of IR and the last bin. */
        if(RT_FREQ_BIN_INFRARED < N_RT_FREQ_BINS-1) {
            if(kf == RT_FREQ_BIN_INFRARED) {kf = N_RT_FREQ_BINS-1;}
            else if(kf == N_RT_FREQ_BINS-1) {kf = RT_FREQ_BIN_INFRARED;}
        }
#endif
#if defined(RT_EVOLVE_INTENSITIES)
        int k_angle; for(k_angle=0;k_angle<N_RT_INTENSITY_BINS;k_angle++)
#endif
        {
            double e0, dt_e_gamma_band=0, total_de_dt=0, a0_abs = -rt_absorption_rate(i,kf, pp, cell);
#if defined(RT_EVOLVE_INTENSITIES)
            if(mode==0) {e0 = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity[kf][k_angle];} else {e0 = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity_Pred[kf][k_angle];}
            dt_e_gamma_band = RT_INTENSITY_BINS_DOMEGA*cell[i].Dt_Rad_Intensity[kf][k_angle];
#else
            if(mode==0) {e0 = cell[i].Rad_E_gamma[kf];} else {e0 = cell[i].Rad_E_gamma_Pred[kf];}
            dt_e_gamma_band = cell[i].Dt_Rad_E_gamma[kf];
#endif
#ifdef RT_COMOVING
            double ET_dotdot_GradVcom = cell[i].ET[kf].double_contract(cell[i].Gradients.Velocity);
            double VolP_dotdot_GradV = e0 * ET_dotdot_GradVcom * All.cf_a2inv; // convert to physical units and multiply by radiation energy density to get into appropriate units
            dt_e_gamma_band += (C_LIGHT_CODE_REDUCED/C_LIGHT_CODE) * (-VolP_dotdot_GradV); // account for RSOL term here as usual
#endif
            total_de_dt = cell[i].Rad_Je[kf] + dt_e_gamma_band;

#ifdef RT_INFRARED
            if(kf == RT_FREQ_BIN_INFRARED)
            {
                if((mode==0) && (dt_e_gamma_band!=0) && (dt_entr>0)) // only update temperatures on kick operations //
                {
                    // advected radiation changes temperature of radiation field, before absorption //
                    double dE_fac = dt_e_gamma_band * dt_entr; // change in energy from advection
                    double dTE_fac = cell[i].Dt_Rad_E_gamma_T_weighted_IR * dt_entr; // T-weighted change from advection
                    double dE_abs = -e0 * (1. - exp(a0_abs*dt_entr)); // change in energy from absorption
                    double T_max = DMAX(cell[i].Radiation_Temperature , dE_fac / dTE_fac);
                    double rfac=1; if(dE_fac < -0.5*(e0+dE_abs)) {rfac=fabs(0.5*(e0+dE_abs))/fabs(dE_fac);} else {if(dE_fac > 0.5*e0) {rfac=0.5*e0/dE_fac;}}
                    dE_fac*=rfac; dTE_fac*=rfac; // limit temperature change from advection to prevent spurious divergences

                    cell[i].Radiation_Temperature = (e0 + dE_fac) / (MIN_REAL_NUMBER + DMAX(0., e0 / cell[i].Radiation_Temperature + dTE_fac));
                    cell[i].Radiation_Temperature = DMIN(cell[i].Radiation_Temperature, T_max);
                    cell[i].Radiation_Temperature = DMAX(cell[i].Radiation_Temperature, DMAX(T_min, MIN_REAL_NUMBER)); // numerator above can go negative in extreme dynamic-range regimes; floor before use below to avoid log10(negative)=NaN propagating into the opacity table lookup
                    a0_abs = -rt_absorption_rate(i,kf, pp, cell); // update absorption rate using the new radiation temperature //
                }
                double total_absorption_rate = E_abs_tot_toIR + fabs(a0_abs)*e0; // add the summed absorption and equate to dust emission //
#ifdef COOLING
                /* the dust temperature is stiffly coupled to the gas and radiation temperatures, so it is advanced
                   with them on kicks; a drift reads the value the last kick left rather than re-solving it from a
                   drifted energy with the other two held fixed. */
                if((mode==0) && (dt_entr>0)) {cell[i].Dust_Temperature = rt_eqm_dust_temp(i, T_gas, total_absorption_rate * vol_inv_phys * C_LIGHT_CODE / C_LIGHT_CODE_REDUCED, pp, cell);}
#endif
                if(cell[i].Dust_Temperature < T_min) {cell[i].Dust_Temperature = T_min;}
                double Tdust_eff = cell[i].Dust_Temperature, Trad_eff = cell[i].Radiation_Temperature;
                double kappa_gas = rt_kappa_adaptive_IR_band(i,Tdust_eff,Trad_eff,-1,-1, pp, cell), kappa_total = rt_kappa_adaptive_IR_band(i,Tdust_eff,Trad_eff,0,0, pp, cell);
                IRBand_opacity_fraction_from_gas_absorption = kappa_gas / (kappa_total + MIN_REAL_NUMBER); /* gas absorption opacity only, relative to total opacity (all sources+scattering) */
                total_de_dt = E_abs_tot_toIR + cell[i].Rad_Je[kf] + dt_e_gamma_band;
                if((mode==0) && (Tdust_eff <= MAX_DUST_TEMP)) // only update temperatures on kick operations and Tdust is meaningful //
                {
                    /* dust absorption and re-emission brings T_rad towards T_dust: */
                    double T_max = DMAX(DMAX(Trad_eff , Tdust_eff), T_gas); /* should not exceed either initial temperature */
                    double frac_unabsorbed = exp(-fabs(a0_abs*dt_entr));
                    double e_absorbed = E_abs_tot_toIR*dt_entr + e0*(1.-frac_unabsorbed);
                    double e_unabsorbed = e0*frac_unabsorbed + DMAX(dt_e_gamma_band*dt_entr, 0); /* include incoming radiation, which has already been used to corect Trad, but outgoing doesn't modify T */
                    double etot_em_gas = e_absorbed * IRBand_opacity_fraction_from_gas_absorption + cell[i].Rad_Je[kf]*dt_entr;
                    double etot_em_dust = e_absorbed * (1.-IRBand_opacity_fraction_from_gas_absorption);
                    double Trad_new = (e_unabsorbed + etot_em_gas + etot_em_dust + MIN_REAL_NUMBER) / (MIN_REAL_NUMBER + e_unabsorbed/Trad_eff + etot_em_gas/DMAX(Trad_eff,T_gas) + etot_em_dust/Tdust_eff);
                    cell[i].Radiation_Temperature = DMIN( Trad_new , T_max );
                }
                if(cell[i].Radiation_Temperature < T_min) {cell[i].Radiation_Temperature = T_min;} // radiation temperature shouldn't be below CMB
            }
#endif

            /*---------------------------------------------------------------------------------------------------
             the following block is for absorption and special behavior where
             photons absorbed in one band are re-radiated [or up/down-scattered] into another.
             this must be hard-coded to maintain conservation (as opposed to treated as a source term)
             -----------------------------------------------------------------------------------------------------*/
            if(fabs(a0_abs)*dt_entr > 50.) {a0_abs *= 50./(fabs(a0_abs)*dt_entr);}
            double abs_0 = DMAX(0,fabs(a0_abs)*dt_entr); double slabfac = slab_averaging_function(abs_0); double e_abs_0=exp(-abs_0); if(abs_0>100.) {e_abs_0=0;}
            /* since we're taking exponentials and inverses of some large numbers here, need to be careful not to let floating point errors cause nan's */
            if((dt_entr <= 0.)||(a0_abs >= 0.)||(abs_0 <= 0.)) {abs_0=0.; slabfac=e_abs_0=1.;} else {if(abs_0 < 1.e-5) {slabfac=1.-0.5*abs_0; e_abs_0 = 1.-abs_0;} else {if(abs_0 > 100.) {slabfac = 1./abs_0; e_abs_0 = 0.;}}}
            double e0_postabs = e0*e_abs_0, de_postabs = total_de_dt * dt_entr * slabfac, f_min = 0.01;
            if(e0_postabs+de_postabs < f_min*e0_postabs) {slabfac *= fabs((1.-f_min)*e0_postabs)/(fabs(de_postabs)+MIN_REAL_NUMBER);}

            double ef = e0 * e_abs_0 + total_de_dt * dt_entr * slabfac; // gives exact solution for dE/dt = -E*abs + de , the 'reduction factor' appropriately suppresses the source term //
#ifdef RT_INFRARED
            if(isnan(ef)) {PRINT_WARNING("\n ef energy prediction is NaN for cell-ID=%llu, e0=%g e_abs_0=%g abs_0=%g a0_abs=%g total_de_dt=%g dt_entr=%g slabfac=%g Trad=%g Tdust=%g\n", (unsigned long long) (long long)i /* particle index */,e0, e_abs_0,abs_0, a0_abs, total_de_dt,dt_entr,slabfac,cell[i].Radiation_Temperature,cell[i].Dust_Temperature);}
#else
            if(isnan(ef)) {PRINT_WARNING("\n ef energy prediction is NaN for cell-ID=%llu, e0=%g e_abs_0=%g abs_0=%g a0_abs=%g total_de_dt=%g dt_entr=%g slabfac=%g\n", (unsigned long long) (long long)i /* particle index */,e0, e_abs_0,abs_0, a0_abs, total_de_dt,dt_entr,slabfac);}
#endif
            if(ef < 0) {ef=0;}
            double de_abs = e0 + total_de_dt * dt_entr - ef; // energy removed by absorption alone
            double de_emission_minus_absorption = (ef - DMAX(0, (e0 + dt_e_gamma_band * dt_entr * slabfac))); // total change, relative to what we would get with just advection (positive = net energy increase in the gas)
            if((dt_entr <= 0) || (de_abs <= 0)) {de_abs = 0;}

#if defined(RT_RAD_PRESSURE_FORCES) && defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_EVOLVE_FLUX) && !defined(RT_RADPRESSURE_IN_HYDRO)
            // for OTVET/FLD methods, need to apply radiation pressure term here so can limit this b/c just based on a gradient which is not flux-limited [as in hydro operators] //
            {
                double radacc[3]={0}, rmag=0, vel_i[3], L_particle = pp[i].Get_Particle_Size()*All.cf_atime; // particle effective size/slab thickness
                double Sigma_particle = cell[i].Mass / (M_PI*L_particle*L_particle), abs_per_kappa_dt = C_LIGHT_CODE_REDUCED * (cell[i].Density*All.cf_a3inv) * dt_entr; // effective surface density through particle & fractional absorption over timestep
                double f_kappa_abs = rt_absorb_frac_albedo(i,kf, pp, cell); // get albedo, we'll need this below
                double slabfac_rp=1; if(check_if_absorbed_photons_can_be_reemitted_into_same_band(kf)==0) {slabfac_rp=slab_averaging_function(f_kappa_abs*cell[i].Rad_Kappa[kf]*Sigma_particle) * slab_averaging_function(f_kappa_abs*cell[i].Rad_Kappa[kf]*abs_per_kappa_dt);} // reduction factor for absorption over dt
                int kx; for(kx=0;kx<3;kx++)
                {
                    radacc[kx] = -dt_entr * slabfac_rp * cell[i].flux_limiter(kf) * (cell[i].Gradients.Rad_E_gamma_ET[kf][kx] / cell[i].Density) / All.cf_atime; // naive radiation-pressure calc for FLD methods [physical units]
                    rmag += radacc[kx]*radacc[kx]; // compute magnitude
                    if(mode==0) {vel_i[kx]=(C_LIGHT_CODE_REDUCED/C_LIGHT_CODE)*pp[i].Vel[kx]/All.cf_atime;} else {vel_i[kx]=(C_LIGHT_CODE_REDUCED/C_LIGHT_CODE)*cell[i].VelPred[kx]/All.cf_atime;} // [for comoving] note this is the 'effective' u appearing in the RHD equations for an RSOL, care needed with these factors!
                }
                if(rmag > 0)
                {
                    rmag = sqrt(rmag); for(kx=0;kx<3;kx++) {radacc[kx] /= rmag;} // normalize
                    double rmag_max = de_abs / (cell[i].Mass * C_LIGHT_CODE_REDUCED * (MIN_REAL_NUMBER + f_kappa_abs)); // limit magnitude to absorbed photon momentum
                    if(check_if_absorbed_photons_can_be_reemitted_into_same_band(kf)==0) {if(rmag > rmag_max) {rmag=rmag_max;}}
#if defined(RT_ENABLE_R15_GRADIENTFIX)
                    rmag = rmag_max; // set to maximum (optically thin limit)
#endif
                    double work_band = 0;
                    for(kx=0;kx<3;kx++)
                    {
                        double radacc_eff = radacc[kx] * rmag; // re-normalize according to the criterion above
                        work_band += vel_i[kx] * radacc_eff * cell[i].Mass; // PdV work done by photons [absorbed ones are fully-destroyed, so their loss of energy and momentum is already accounted for by their deletion in this limit //
                        if(mode==0) {double dv_rt=radacc_eff*All.cf_atime; pp[i].Vel[kx]+=dv_rt; pp[i].dp[kx]+=dv_rt*cell[i].Mass;} else {cell[i].VelPred[kx] += radacc_eff * All.cf_atime;}
                    }
                    double d_egy_rad = (2.*f_kappa_abs-1.)*work_band , d_egy_int = -2.*f_kappa_abs*work_band * (C_LIGHT_CODE/C_LIGHT_CODE_REDUCED); // correct for rsol factor above which reduced vel_i by rsol; -only- add back this term for gas
                    if(mode==0) {cell[i].InternalEnergy += d_egy_int;} else {cell[i].InternalEnergyPred += d_egy_int;}
#if defined(RT_EVOLVE_INTENSITIES)
                    {int k_q; for(k_q=0;k_q<N_RT_INTENSITY_BINS;k_q++) {if(mode==0) {cell[i].Rad_Intensity[kf][k_q]+=d_egy_rad/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[kf][k_q]+=d_egy_rad/RT_INTENSITY_BINS_DOMEGA;}}}
#else
                    if(mode==0) {cell[i].Rad_E_gamma[kf]+=d_egy_rad;} else {cell[i].Rad_E_gamma_Pred[kf]+=d_egy_rad;}
#endif
                }
            }
#endif

            int donation_target_bin = rt_get_donation_target_bin(kf); // frequency into which the photons will be deposited, if any //
#ifdef RT_INFRARED
            if((donation_target_bin == RT_FREQ_BIN_INFRARED) && (kf != RT_FREQ_BIN_INFRARED)) {E_abs_tot_toIR += de_abs/(MIN_REAL_NUMBER + dt_entr);} /* donor bin is yourself in the IR - some self-absorption is re-emitted, but this is handled explicitly below, so don't need to include it in sum here */
            if(kf==RT_FREQ_BIN_INFRARED) {
#ifdef COOLING
                ef += de_abs*(1.-IRBand_opacity_fraction_from_gas_absorption); /* update: assume a fraction de_abs * IRBand_opacity_fraction_from_gas_absorption is absorbed by the gas, which will not be instantly re-emitted here, but later in the cooling subroutines */
                if(mode==0) {cell[i].DtInternalEnergy += (de_abs * IRBand_opacity_fraction_from_gas_absorption) / ((MIN_REAL_NUMBER + dt_entr) * cell[i].Mass);} /* this fraction absorbed by gas goes into a heating rate which can be balanced implicitly in the cooling function later */
#else
                ef = e0 + total_de_dt * dt_entr; // previous version: assumes all self-absorption is re-emitted
#endif
            } /* donor bin is yourself in the IR - just need to decide what to do with the photons */
#endif
            // isotropically re-emit the donated radiation into the target bin[s] //
#if defined(RT_EVOLVE_INTENSITIES)
            // this is the leading-order (isotropic) emission-absorption step, i.e. the psi_a * (j_e - I) term in the intensity equation. solved by the methods above to deal generically with stiff emission-absorption problems, re-used below if needed //
            if(donation_target_bin >= 0) {int k_q; for(k_q=0;k_q<N_RT_INTENSITY_BINS;k_q++) {if(mode==0) {cell[i].Rad_Intensity[donation_target_bin][k_q] += de_abs/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[donation_target_bin][k_q] += de_abs/RT_INTENSITY_BINS_DOMEGA;}}}
            if(ef < 0) {ef=0;}
            if(mode==0) {cell[i].Rad_Intensity[kf][k_angle] = ef/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[kf][k_angle] = ef/RT_INTENSITY_BINS_DOMEGA;}
#else
            if(donation_target_bin >= 0) {if(mode==0) {cell[i].Rad_E_gamma[donation_target_bin] += de_abs;} else {cell[i].Rad_E_gamma_Pred[donation_target_bin] += de_abs;}}
            if(ef < 0) {ef=0;}

            if(mode==0) {cell[i].Rad_E_gamma[kf] = ef;} else {cell[i].Rad_E_gamma_Pred[kf] = ef;}
#endif

#if defined(RT_EVOLVE_FLUX)
            int k_dir; double f_mag=0, E_rad_forflux=0, vdot_h[3]={0}, vel_i[3]={0}, DeltaFluxEff[3]={0}, rho=cell[i].Density*All.cf_a3inv; E_rad_forflux=0.5*(e0+ef); // use energy density averaged over this update for the operation below
            for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {vel_i[k_dir]=RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS*pp[i].Vel[k_dir]/All.cf_atime;} else {vel_i[k_dir]=RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS*cell[i].VelPred[k_dir]/All.cf_atime;}} // need gas velocity at this time [effective v - note RSOL terms]
            double teqm_inv = cell[i].Rad_Kappa[kf] * rho * C_LIGHT_CODE_REDUCED + MIN_REAL_NUMBER; // physical code units of 1/time, defines characteristic timescale for coming to equilibrium flux. see notes for CR second-order module for details. //
            {Vec3<double> v_i{vel_i[0],vel_i[1],vel_i[2]}; Vec3<double> vdh = E_rad_forflux * (v_i + cell[i].ET[kf].matvec(v_i)); vdot_h[0]=vdh[0]; vdot_h[1]=vdh[1]; vdot_h[2]=vdh[2];} // calculate P_rad term and eI term, multiply by radiation energy //
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR // definitely favor this for greater accuracy and reduced noise //
            for(k_dir=0;k_dir<3;k_dir++) {DeltaFluxEff[k_dir] -= (cell[i].Mass/rho) * (C_LIGHT_CODE_REDUCED*C_LIGHT_CODE_REDUCED/teqm_inv) * cell[i].Gradients.Rad_E_gamma_ET[kf][k_dir]*All.cf_a3inv/All.cf_atime;} // here we compute the nabla.pressure_gradient_tensor term from gradients directly, and use this in the next step after multiplying the flux equation by (tilde[c]^2/dt_eqm_inv) and working in dimensionless time units
#else
            for(k_dir=0;k_dir<3;k_dir++) {DeltaFluxEff[k_dir] += (cell[i].Dt_Rad_Flux[kf][k_dir]/teqm_inv);} // the nabla.pressure_gradient_tensor is computed in the finite-volume solver, here
#endif
            for(k_dir=0;k_dir<3;k_dir++) {DeltaFluxEff[k_dir] += vdot_h[k_dir];} // add the 'enthalpy advection' term here, vdot_h = Erad v.(e*I + P_rad)

            double tau=dt_entr*teqm_inv, f00=exp(-tau), f11=1.-f00;
            if(tau > 0 && isfinite(tau))
            {
                if(tau <= 0.04) {f11=tau-0.5*tau*tau+tau*tau*tau/6.; f00=1.-f11;} // some limits to prevent small/large number problems here
                if(!isfinite(f00) || !isfinite(f11)) {f00=1.; f11=0.;} // some limits to prevent small/large number problems here
                if(tau >= 20.) {f00=DMAX(0.,DMIN(1.e-11,f00)); f11=1.-f00;} // some limits to prevent small/large number problems here
                for(k_dir=0;k_dir<3;k_dir++)
                {
                    double flux_0; if(mode==0) {flux_0 = cell[i].Rad_Flux[kf][k_dir];} else {flux_0 = cell[i].Rad_Flux_Pred[kf][k_dir];}
                    flux_0 += vel_i[k_dir] * de_emission_minus_absorption; // add Lorentz term from net energy injected by absorption and re-emission (effectively, we operator-split this term and solve it -BEFORE- going to the next step). note our definitions of vel_i above ensure this is zero when the rhd is comoving-frame
                    double flux_f = flux_0 * f00 + DeltaFluxEff[k_dir] * f11; // exact solution for dE/dt = -E*abs + de , the 'reduction factor' appropriately suppresses the source term //
                    if(mode==0) {cell[i].Rad_Flux[kf][k_dir] = flux_f;} else {cell[i].Rad_Flux_Pred[kf][k_dir] = flux_f;}
                    f_mag += flux_f*flux_f; // magnitude of flux vector
                }
                if(f_mag > 0) // limit the flux according the physical (optically thin) maximum //
                {
                    f_mag=sqrt(f_mag); double fmag_max = C_LIGHT_CODE_REDUCED * ef; // maximum flux should be optically-thin limit: e_gamma*c: here allow some tolerance for numerical leapfrogging in timestepping. should be the RSOL here, although in principle equations can allow exceeding this if we have reached equilibrium, it really violates the M1 closure assumptions. see discussion in Skinner+Ostriker 2013 or Levermore et al. 1984
                    if(f_mag > fmag_max) {for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {cell[i].Rad_Flux[kf][k_dir] *= fmag_max/f_mag;} else {cell[i].Rad_Flux_Pred[kf][k_dir] *= fmag_max/f_mag;}}}
#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
                    if(pp[i].Pos[2]<=0.1) {if(mode==0) {cell[i].Rad_Flux[kf][0]=cell[i].Rad_Flux[kf][1]=0; cell[i].Rad_Flux[kf][2]=fmag_max;} else {cell[i].Rad_Flux_Pred[kf][0]=cell[i].Rad_Flux_Pred[kf][1]=0; cell[i].Rad_Flux_Pred[kf][2]=fmag_max;}}
#endif
                }
            }
#endif
            total_erad_emission_minus_absorption += de_emission_minus_absorption; // add to cumulative sum for back-reaction to gas
#if defined(RT_EVOLVE_INTENSITIES)
            de_emission_minus_absorption_saved[kf][k_angle] = de_emission_minus_absorption; // save this for use below; indexed by BAND (kf), matching the read below
#endif
        } // clause for radiation angle [needed for evolving intensities]
    } // loop over frequencies

#if defined(RT_EVOLVE_INTENSITIES)
    if(dt_entr > 0) { // none of this is worth doing if we don't have a finite timestep here
    for(kf=0;kf<N_RT_FREQ_BINS;kf++)
    {
        int k,k_om; double rho=cell[i].Density*All.cf_a3inv, ceff=C_LIGHT_CODE_REDUCED, ctrue=C_LIGHT_CODE, teq_inv=cell[i].Rad_Kappa[kf]*rho*ceff, f_a=rt_absorb_frac_albedo(i,kf, pp, cell), f_s=1.-f_a, b_dot_n[N_RT_INTENSITY_BINS]={0}, beta_2=0.; Vec3<double> beta;
        int n_iter = 1 + (int)(DMIN(DMAX(4. , dt_entr/teq_inv), 1000.)); // number of iterations to subcycle everything below //
        double dt=dt_entr/n_iter, tau=dt*teq_inv, i0[N_RT_INTENSITY_BINS]={0}, invfourpi=1./(4.*M_PI), J, b_dot_H, b2_dot_K; int i_iter;
        for(i_iter=0; i_iter<n_iter; i_iter++)
        {
            double egy_0=0, egy_f=0; Vec3<double> flux_0={}, flux_f={}; // compute total change over sub-cycle, to update gas properties
            // load all the gas and intensity properties we need [all can change on the subcycle so some re-computing here]
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if(mode==0) {i0[k_om] = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity[kf][k_om];} else {i0[k_om] = RT_INTENSITY_BINS_DOMEGA*cell[i].Rad_Intensity_Pred[kf][k_om];}}
            if(mode==0) {beta=pp[i].Vel/(All.cf_atime*ctrue);} else {beta=cell[i].VelPred/(All.cf_atime*ctrue);} // need gas velocity at this time; with equations written this way, the 'beta' term is the -true- beta, so we have to use the true SOL
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {b_dot_n[k_om]=0; for(k=0;k<3;k++) {b_dot_n[k_om]+=All.Rad_Intensity_Direction[k_om][k]*beta[k];}}
            beta_2 = beta.norm_sq();
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {egy_0+=i0[k_om]; for(k=0;k<3;k++) {flux_0[k]+=All.Rad_Intensity_Direction[k_om][k]*i0[k_om];}}
            J=0,b_dot_H=0,b2_dot_K=0; for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {J+=i0[k_om]*invfourpi; b_dot_H+=b_dot_n[k_om]*i0[k_om]*invfourpi; b2_dot_K+=b_dot_n[k_om]*b_dot_n[k_om]*i0[k_om]*invfourpi;}

            // isotropic terms that change total energy in bin (part of the 'work term' for the photon momentum): this includes the beta.beta*(J+K) and beta.H terms
            double work = (1. * (f_s-f_a)*(beta_2*J + b2_dot_K) - 2.*f_s*b_dot_H) * tau; // will be shared isotropically.
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if((work>0) || (i0[k_om]<=0)) {i0[k_om]+=work;} else {i0[k_om]/=(1-work/i0[k_om]);}} // gaurantees linearized sum is still correct, and symmetric with positive changes, but can't get negative energies. shared isotropically.

            // isotropic scattering term [scattering * (J - I) term in the intensity equation] [recall, our general update for the 'energy term' for absorption and emission above already took care of the psi_a*(j_em - I) term in the intensity equation
            J=0; for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {J+=i0[k_om]*invfourpi;} // prepare to calculate isotropic scattering term
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {i0[k_om] = J + (i0[k_om]-J)*exp(-f_s*tau);} // isotropic scattering conserving total energy over step

            // flux 'boost' and 'beaming' terms (go as n.beta). note we replace je -> je-I + I, and use the fact that we have solved already for the time-integral of (psi_a*(je-I)*dt) = de_emission_minus_absorption_saved, which can be re-used here, in average form <psi_a*(je-I)> = dE/dt --> just make sure the units are correct! because we're working in dimensionless units below, we should divide by tau, to be working in the same delta-units here: these are the 3 n.beta * [ psi_a*(j_em-J_nu)*(creduced/c)^2 + (psi_a+psi_s)*J_nu) ] in the intensity equation
            double fboost[N_RT_INTENSITY_BINS], fboost_avg=0, fboost_avg_p=0, fboost_avg_m=0; // calculate flux 'boost' terms
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {fboost[k_om] = 3.*b_dot_n[k_om] * ((de_emission_minus_absorption_saved[kf][k_om]/tau) + ((f_a+f_s)*J)); fboost_avg += fboost[k_om]/N_RT_INTENSITY_BINS;} // pre-calculate to get mean value, will divide out
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {work=(fboost[k_om]-fboost_avg)*tau; if((work>0) || (i0[k_om]<=0)) {fboost[k_om]=work; fboost_avg_p+=fboost[k_om];} else {fboost[k_om]=work/(1.-work/i0[k_om]); fboost_avg_m+=fboost[k_om];}} // zero total energy change at linear order ensured by subtracting out sum here; non-linearization ensures i0 cannot be negative, but does allow second-order dt work term to appear, that's ok for now
            if(fboost_avg_p>0 && fboost_avg_m<0) {double fc=-fboost_avg_m/fboost_avg_p; fboost_avg_m=(1.+fc)/(1.+fc*fc); fboost_avg_p=fc*fboost_avg_m;} else {fboost_avg_m=fboost_avg_p=0;} // // these re-weight to gaurantee the non-linear sum is identically zero while preserving positive-definite behavior
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if(fboost[k_om]>0) {i0[k_om]+=fboost_avg_p*fboost[k_om];} else {i0[k_om]+=fboost_avg_m*fboost[k_om];}} // alright done!

            // flux work term, allowed to both do work and be asymmetric so just need to ensure it retains positive-definite intensities: n.beta * (psi_a+psi_s) * I term in intensity equation
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {work=b_dot_n[k_om]*(f_a+f_s)*i0[k_om] * tau; if((work>0) || (i0[k_om]<=0)) {i0[k_om]+=work;} else {i0[k_om]/=(1-work/i0[k_om]);}}

            // ok -now- calculate the net change in momentum and energy, for updating the gas quantities
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {egy_f+=i0[k_om]; for(k=0;k<3;k++) {flux_f[k]+=All.Rad_Intensity_Direction[k_om][k]*i0[k_om];}}
            Vec3<double> dv_gas = -(flux_f-flux_0)/(ceff*cell[i].Mass); double ke_gas_0=0, ke_gas_f=0, v0g=0, u0=0;
            {Vec3<double> v0_gas = ctrue*beta; ke_gas_0 = v0_gas.norm_sq(); ke_gas_f = (v0_gas+dv_gas).norm_sq();} // note everything is volume-integrated, accounted for above, and we defined flux for convience without the c, so just one power of c here.
            double d_ke_gas = 0.5*(ke_gas_f - ke_gas_0)*cell[i].Mass, de_gas=-(ctrue/ceff)*(egy_f-egy_0), de_gas_internal=(de_gas-d_ke_gas)/cell[i].Mass; // note ctrue/ceff factor here, accounting for rsol difference in gas heating/cooling rates vs RHD
            if(mode==0) {u0=cell[i].InternalEnergy;} else {u0=cell[i].InternalEnergyPred;} // for updating gas internal energy (work terms, after subtracting kinetic energy changes)
            if(de_gas_internal<=-0.9*u0) {de_gas_internal = DMIN(de_gas_internal/(1.-de_gas_internal/u0), -0.9*u0);} // just a catch to avoid negative energies (will break energy conservation if you are slamming into it, however!

            // assign everything back to the appropriate variables after update
            if(mode==0) {auto dv_kick=dv_gas*All.cf_atime; pp[i].Vel+=dv_kick; pp[i].dp+=dv_kick*cell[i].Mass;} else {cell[i].VelPred += dv_gas*All.cf_atime;} // update gas velocities (radiation pressure forces here)
            if(mode==0) {cell[i].InternalEnergy += de_gas_internal;} else {cell[i].InternalEnergyPred += de_gas_internal;} // update gas internal energy (work terms, after subtracting kinetic energy changes)
            for(k_om=0;k_om<N_RT_INTENSITY_BINS;k_om++) {if(mode==0) {cell[i].Rad_Intensity[kf][k_om] = i0[k_om]/RT_INTENSITY_BINS_DOMEGA;} else {cell[i].Rad_Intensity_Pred[kf][k_om] = i0[k_om]/RT_INTENSITY_BINS_DOMEGA;}} // update intensities (all of the above)
            cell[i].Rad_E_gamma[kf]=egy_f; // set this every time this subroutine is called, so it is accessible everywhere else //
        } // loop over iterations
    } // loop over frequencies
    } // finite timestep requirement
#else
    double mom_fac = 1. - RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS * total_erad_emission_minus_absorption / (cell[i].Mass * C_LIGHT_CODE_REDUCED*C_LIGHT_CODE_REDUCED); // back-reaction on gas from emission, which is isotropic in the fluid frame but anisotropic in the lab frame. this effect is only important in actually semi-relativistic problems so we use "real" C here, not a RSOL, and match the corresponding term above in the radiation flux equation (if that is evolved explicitly). careful checking-through gives the single termm here, not both
    if(fabs(mom_fac - 1) > 0.1) {printf("WARNING: Large radiation backreaction for cell %d (mom_fac=%g), check the RT solver stability if this is not a relativistic problem.\n",i,mom_fac);}
    {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {pp[i].dp[k_dir]+=pp[i].Vel[k_dir]*(mom_fac-1.)*cell[i].Mass; pp[i].Vel[k_dir]*=mom_fac;} else {cell[i].VelPred[k_dir] *= mom_fac;}}}
#endif

    if(mode > 0) {rt_eddington_update_calculation(i, pp, cell);} /* update the eddington tensor (if we calculate it) as well */

#ifdef RT_ISRF_BACKGROUND
    if(mode==0) {rt_apply_boundary_conditions(i, pp, cell);} /* if we have any special boundary conditions (e.g. fixed ISRF at box edge) apply this here */
#endif
#ifdef NUCLEAR_NETWORK_NEUTRINOS
    if(mode==0) { /* after RT kick, update Ye from neutrino absorption */
        nuclear_neutrino_ye_feedback(i, dt_entr, pp, cell);
    }
#endif
#endif
}
#endif /* RADTRANSFER */





/* ======================================================================
 * Gravity-tree source-luminosity payload helpers (RT source spectrum,
 * ionizing luminosity, chimes fluxes). Device + host single source of
 * truth; host externals emitted by radiation/rt_utilities.cc.
 * ====================================================================== */
#if defined(GRAVTREE_SOURCE_HOST_OWNER_TU) || (defined(GRAVTREE_SOURCE_DEVICE_TU) && defined(GRAVTREE_SOURCE_LAZY_SUPPORTED))
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)

#include "../galaxy_sf/stellar_evolution_functions.h"
#include "../sinks/sink_functions.h"
#include "../core/predict_functions.h"

#define SET_ACTIVE_RT_CHECK() if(mode<0) {return 1;} else {active_check=1;}

/* forward declarations so the dispatcher can call the band helpers below */
KOKKOS_INLINE_FUNCTION int rt_get_source_luminosity(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION int rt_get_lum_band_stellarpopulation(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION int rt_get_lum_band_agn(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION int rt_get_lum_band_singlestar(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void rt_get_lum_gas(int target, double *je, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION double stellar_lum_in_band(int i, double E_lower, double E_upper, struct particle_data *pp, struct gas_cell_data *cell);

#if defined(GALSF_FB_FIRE_RT_HIIHEATING) || (defined(RT_CHEM_PHOTOION) && defined(GALSF))
KOKKOS_INLINE_FUNCTION double particle_ionizing_luminosity_in_cgs_core(long i, struct particle_data *pp)
{
    if(pp[i].Mass <= 0 || !isfinite(pp[i].Mass)) {return 0;}
    if(is_particle_single_star_eligible_core(i, pp)) /* SINGLE STAR VERSION: use effective temperature as a function of stellar mass and size to get ionizing photon production */
    {
#ifdef SINGLE_STAR_SINK_DYNAMICS
        double l_sol=sink_lum_bol_core(0,pp[i].Mass,i,pp)*(UNIT_LUM_IN_SOLAR), m_sol=pp[i].Mass*UNIT_MASS_IN_SOLAR, r_sol=pow(m_sol,0.738); // L/Lsun, M/Msun, R/Rsun
        double T_eff=5780.*pow(l_sol/(r_sol*r_sol),0.25), x0=157800./T_eff, fion=0; // ZAMS effective temperature; x0=h*nu/kT for nu>13.6 eV; fion=fraction of blackbody emitted above x0
        if(x0 < 30.) {double q=18./(x0*x0) + 1./(8. + x0 + 20.*exp(-x0/10.)); fion = exp(-1./q);} // accurate to <10% for a Planck spectrum to x0>30, well into vanishing flux //
        return fion * l_sol * SOLAR_LUM_CGS; // return value in cgs, as desired for this routine [l_sol is in L_sun, by definition above] //
#endif
    }
    else /* STELLAR POPULATION VERSION: use updated SB99 tracks: including rotation, new mass-loss tracks, etc. */
    {
        if(pp[i].Type != 5)
        {
            double lm_ssp=0, star_age=evaluate_stellar_age_Gyr_core(i, pp), t0=0.0035, tmax=0.02;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
            tmax=0.15; lm_ssp=evaluate_light_to_mass_ratio_core(star_age,i,pp); if(star_age<t0) {lm_ssp*=0.5;} else {lm_ssp*=0.5*pow(star_age/t0,-2.9);} /* slightly revised fit scales simply with Lbol [easier to modify]; see same references for stellar wind mass-loss rates; and extends to later ages (though most comes out at <100 Myr) */
#else
            if(star_age < t0) {lm_ssp=500.;} else {double log_age=log10(star_age/t0); lm_ssp=470.*pow(10.,-2.24*log_age-4.2*log_age*log_age) + 60.*pow(10.,-3.6*log_age);}
#endif
            if(star_age < 0.033) {lm_ssp *= 1.e-4 + calculate_relative_light_to_mass_ratio_from_imf_core(star_age,i,1,pp);}
            if(star_age >= tmax) {return 0;} // skip since old stars don't contribute
            return lm_ssp * SOLAR_LUM_CGS * (pp[i].Mass*UNIT_MASS_IN_SOLAR); // converts to cgs luminosity [lm_ssp is in Lsun/Msun, here]
        } // (pp[i].Type != 5)
#ifdef SINK_HII_HEATING /* AGN template: light-to-mass ratio L(>13.6ev)/Mparticle in Lsun/Msun, above is dNion/dt = 5.5e54 s^-1 (Lbol/1e45 erg/s) */
        if(pp[i].Type == 5) {return 0.18 * sink_lum_bol_core(pp[i].Sink_Mdot,pp[i].Mass,i,pp) * UNIT_LUM_IN_CGS;}
#endif
    }
    return 0; // catch
}
#endif /* GALSF_FB_FIRE_RT_HIIHEATING || (RT_CHEM_PHOTOION && GALSF) */

KOKKOS_INLINE_FUNCTION int rt_get_source_luminosity(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(!((1 << pp[i].Type) & (RT_SOURCES))) {return 0;}; // boolean test of whether i is a source or not - end if not a valid source particle
    if(pp[i].Mass <= 0 || !isfinite(pp[i].Mass)) {return 0;} // reject invalid particles scheduled for deletion (P[].Mass is the SSOT for particle mass; CellP[].Mass is gas-only and is uninitialized for non-gas types, which would silently filter Type 4/5 sources)
    int active_check = 0; // default to inactive //

#if defined(GALSF)
#if defined(SINGLE_STAR_SINK_DYNAMICS)
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(pp[i].Type == 5)
    {   // here we have type=5, treat as a sink particle/single-star
        if(pp[i].ProtoStellarStage != 7) {active_check += rt_get_lum_band_singlestar(i,mode,lum, pp, cell); /* stars and protostars */} else {active_check += rt_get_lum_band_agn(i,mode,lum, pp, cell); /* relics */}
    } else { // otherwise, treat as a stellar population particle
        active_check += rt_get_lum_band_stellarpopulation(i,mode,lum, pp, cell); // get luminosities for star particles assuming they represent IMF-averaged populations
    }
#else
    active_check += rt_get_lum_band_singlestar(i,mode,lum, pp, cell); // get luminosities for individual star/sink particles assuming they are protostars or stars
#endif
#else
    active_check += rt_get_lum_band_stellarpopulation(i,mode,lum, pp, cell); // get luminosities for star particles assuming they represent IMF-averaged populations
#if defined(SINK_PARTICLES)
    active_check += rt_get_lum_band_agn(i,mode,lum, pp, cell); // get luminosities for BH/sink particles assuming they represent AGN
#endif
#endif
#endif
    if(mode < 0 && active_check) {return 1;} // if got a positive answer already, that's all we are checking here, we are done


#if defined(RT_CHEM_PHOTOION) && !defined(GALSF) /* Hydrogen and Helium ionizing bands; this is an idealized test-problem version implementation */
    if(pp[i].Type==4)
    {
        SET_ACTIVE_RT_CHECK(); double l_ion=All.IonizingLuminosityPerSolarMass_cgs * (pp[i].Mass * UNIT_MASS_IN_SOLAR) / UNIT_LUM_IN_CGS; // flux from star particles according to mass (P[].Mass is SSOT for stars; CellP[].Mass is gas-only)
#ifdef RT_ILIEV_TEST1
        l_ion = 5.0e48 * (13.6*ELECTRONVOLT_IN_ERGS) / UNIT_LUM_IN_CGS; // 5e48 in ionizing photons per second -- constant for idealized test problem //
#endif
        lum[RT_FREQ_BIN_H0] = l_ion; // default to all flux into single-band
#if defined(RT_PHOTOION_MULTIFREQUENCY)
        int i_vec[4] = {RT_FREQ_BIN_H0, RT_FREQ_BIN_He0, RT_FREQ_BIN_He1, RT_FREQ_BIN_He2}; // these will all be the same if not using multi-frequency module //
        int k; for(k=0;k<4;k++) {lum[i_vec[k]] = l_ion * All.rt_ion_precalc_stellar_luminosity_fraction[i_vec[k]];} // assign flux appropriately according to pre-tabulated result //
#endif
    }
#endif


#if defined(RT_GENERIC_USER_FREQ)   /* example code to be modified as-needed for custom RT problems */
    if(pp[i].Type == 4) // set this to whichever type you want to use for the specific sources in this band
    {
        SET_ACTIVE_RT_CHECK(); // flag that tells the code that indeed this particle should be active!
        lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = 0; // set the actual luminosity here for your test problem!
#ifdef GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION /* assume special units for this problem, and that total mass of 'sources' is 1 */
        double m_total_expected = 1; // assume total mass of sources is 1, and we want to weight such that fractional emission per source is equal to their mass fraction
        double A_base = boxSize_X * boxSize_X; // area of the base of the box used for scaling to get the desired flux
#if (NUMDIMS == 3)
        A_base = boxSize_X * boxSize_Y;
#endif
        lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = (pp[i].Mass/1.) * All.Vertical_Grain_Accel * C_LIGHT_CODE * ((All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS)*(All.Grain_Size_Max/UNIT_LENGTH_IN_CGS)) * A_base / (0.75*All.Grain_Q_at_MaxGrainSize); // special behavior for particular test of stratified boxes compared to explicit dust opacities (P[].Mass is SSOT for non-gas)
#endif
    }
#endif


#ifdef RADTRANSFER
    if(pp[i].Type == 0) /* generic sub routines for gas as a source term, should go at the very end of this routine */
    {
        SET_ACTIVE_RT_CHECK(); rt_get_lum_gas(i,lum, pp, cell); // optionally re-distributes cooling flux as a blackbody; but also where bands like free-free reside //
        int k; for(k=0;k<N_RT_FREQ_BINS;k++) {lum[k] += cell[i].Rad_Je[k];}
    }
#endif

#ifdef NUCLEAR_NETWORK_NEUTRINOS
    /* neutrino emission from nuclear burning (gas cells only) */
    if(pp[i].Type == 0) {
        lum[RT_FREQ_BIN_NU_E]    += cell[i].NeutrinoLuminosity[0];
        lum[RT_FREQ_BIN_NU_EBAR] += cell[i].NeutrinoLuminosity[1];
        lum[RT_FREQ_BIN_NU_X]    += cell[i].NeutrinoLuminosity[2];
        SET_ACTIVE_RT_CHECK();
    }
#endif

    /* need to renormalize ALL sources for reduced speed of light */
    {int k; for(k=0;k<N_RT_FREQ_BINS;k++) {lum[k] *= (C_LIGHT_CODE_REDUCED/C_LIGHT_CODE);}}
    return active_check;
}


/***********************************************************************************************************/
/* calculate the opacity for use in radiation transport operations [in physical code units = Length^2/Mass]. this should
    be a total extinction opacity, i.e. kappa = kappa_scattering + kappa_absorption */
/***********************************************************************************************************/
/* rt_kappa, rt_absorb_frac_albedo: definitions now in rt_functions.h */




/* subroutine for 'rt_get_source_luminosity', with identical variables, for cases where the radiation
    represents IMF-averaged stellar populations, i.e. the sort of thing which would be used in galaxy simulations.
 */
KOKKOS_INLINE_FUNCTION int rt_get_lum_band_stellarpopulation(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(!is_galsf_stellar_candidate_type(pp[i].Type, All.ComovingIntegrationOn)) {return 0;} // only star-type particles act in this subroutine //
    if(pp[i].Mass <= 0 || !isfinite(pp[i].Mass)) {return 0;} // (P[].Mass is SSOT for particle mass; CellP[].Mass is gas-only and uninitialized for stars)
    int active_check = 0; // default to inactive //
#if defined(GALSF) /* basically none of these modules make sense without the GALSF module active */
    double star_age = evaluate_stellar_age_Gyr_core(i, pp), m_sol = pp[i].Mass * UNIT_MASS_IN_SOLAR; // (P[].Mass is SSOT for stellar mass; CellP[].Mass is zero for Type 4)
    if((star_age<=0) || isnan(star_age)) {return 0;} // calculate stellar age, will be used below, and catch for bad values
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    if(star_age > 0.1) {return 0;} // old optimization, not really needed with how we do this now //
#endif


#if defined(GALSF_FB_FIRE_RT_LONGRANGE) /* three-band (UV, OPTICAL, IR) approximate spectra for stars as used in the FIRE (Hopkins et al.) models */
    SET_ACTIVE_RT_CHECK();
    double f_uv=All.PhotonMomentum_fUV, f_op=All.PhotonMomentum_fOPT;
    double L = evaluate_light_to_mass_ratio_core(star_age, i, pp) * m_sol / UNIT_LUM_IN_SOLAR; if(L<=0 || isnan(L)) {L=0;}
    double sigma_eff = evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,pp[i].DensityAroundParticle,pp[i].NumNgb,0,i,pp); if((sigma_eff <= 0)||(isnan(sigma_eff))) {sigma_eff=0;} // sigma here is in code units
    if(star_age <= 0.0025) {f_op=0.09;} else {if(star_age <= 0.006) {f_op=0.09*(1+((star_age-0.0025)/0.004)*((star_age-0.0025)/0.004));} else {f_op=1-0.8410937/(1+sqrt((star_age-0.006)/0.3));}}
    /* note that the metallicity doing attenuation is the -gas- opacity around the star, while here we only know the stellar metallicity,
        so we use this as a guess, but this could substantially under-estimate opacities for old stars in MW-like galaxies. But for young stars (which dominate) this is generally ok. */
    double tau_uv = sigma_eff*rt_kappa(i,RT_FREQ_BIN_FIRE_UV, pp, cell); double tau_op = sigma_eff*rt_kappa(i,RT_FREQ_BIN_FIRE_OPT, pp, cell); // kappa returned in code units
    f_uv = (1-f_op)*(All.PhotonMomentum_fUV + (1-All.PhotonMomentum_fUV)/(1+0.8*tau_uv+0.85*tau_uv*tau_uv));
    f_op *= All.PhotonMomentum_fOPT + (1-All.PhotonMomentum_fOPT)/(1+0.8*tau_op+0.85*tau_op*tau_op); /* this is a fitting function for tau_disp~0.22 'tail' w. exp(-tau) 'core', removes expensive functions [f_uv = (1-f_op)*(All.PhotonMomentum_fUV + (1-All.PhotonMomentum_fUV)*exp(-tau_uv)); f_op *= All.PhotonMomentum_fOPT + (1-All.PhotonMomentum_fOPT)*exp(-tau_op);]
     :: accounting for leakage for P(tau) ~ exp(-|logtau/tau0|/sig), following Hopkins et al. 2011, we have: [f_uv = (1-f_op)*(All.PhotonMomentum_fUV + (1-All.PhotonMomentum_fUV)/ (1 + pow(tau_uv,1./(4.*tau_disp))/(3.*tau_disp) + pow(2.*tau_disp*tau_uv,1./tau_disp))); f_op *= All.PhotonMomentum_fOPT + (1-All.PhotonMomentum_fOPT)/ (1 + pow(tau_op,1./(4.*tau_disp))/(3.*tau_disp) + pow(2.*tau_disp*tau_op,1./tau_disp));] */
    lum[RT_FREQ_BIN_FIRE_UV]  = L * f_uv;
    lum[RT_FREQ_BIN_FIRE_OPT] = L * f_op;
    lum[RT_FREQ_BIN_FIRE_IR]  = L * (1-f_uv-f_op);
#endif

#if defined(RT_INFRARED) /* can add direct infrared sources, but default to no direct IR (just re-emitted light) */
    lum[RT_FREQ_BIN_INFRARED] = 0; //default to no direct IR (just re-emitted light)
#endif

#if defined(RT_OPTICAL_NIR) /* Optical-NIR approximate spectra for stars as used in the FIRE (Hopkins et al.) models */
    SET_ACTIVE_RT_CHECK();
    double f_op=0; if(star_age <= 0.0025) {f_op=0.09;} else {
        if(star_age <= 0.006) {f_op=0.09*(1+((star_age-0.0025)/0.004)*((star_age-0.0025)/0.004));} else {f_op=1-0.8410937/(1+sqrt((star_age-0.006)/0.3));}}
    lum[RT_FREQ_BIN_OPTICAL_NIR] = f_op * evaluate_light_to_mass_ratio_core(star_age, i, pp) * m_sol / UNIT_LUM_IN_SOLAR;
#endif

#if defined(RT_NUV) /* Near-UV approximate spectra (UV/optical spectra, sub-photo-electric, but high-opacity) for stars as used in the FIRE (Hopkins et al.) models */
    SET_ACTIVE_RT_CHECK();
#if !defined(RT_OPTICAL_NIR)
    double f_op=0; if(star_age <= 0.0025) {f_op=0.09;} else {
        if(star_age <= 0.006) {f_op=0.09*(1+((star_age-0.0025)/0.004)*((star_age-0.0025)/0.004));} else {f_op=1-0.8410937/(1+sqrt((star_age-0.006)/0.3));}}
#endif
    lum[RT_FREQ_BIN_NUV] = (1-f_op) * evaluate_light_to_mass_ratio_core(star_age, i, pp) * m_sol / UNIT_LUM_IN_SOLAR;
#endif

#if defined(RT_PHOTOELECTRIC) /* photo-electric bands (8-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    SET_ACTIVE_RT_CHECK();
    double l_band_pe, x_age_pe = star_age / 3.4e-3; // converts to code units, and defines age relative to convenient break time
    if(x_age_pe <= 1) {l_band_pe = 1.07e36 * (1.+x_age_pe*x_age_pe) * m_sol / UNIT_LUM_IN_CGS;}
        else {l_band_pe = 2.14e36 / (x_age_pe * sqrt(x_age_pe)) * m_sol / UNIT_LUM_IN_CGS;} // 0.1 solar, with nebular. very weak metallicity dependence, with slightly slower decay in time for lower-metallicity pops; effect smaller than binaries
    lum[RT_FREQ_BIN_PHOTOELECTRIC] = l_band_pe; // band luminosity //
#endif

#if defined(RT_LYMAN_WERNER)  /* lyman-werner bands (11.2-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    SET_ACTIVE_RT_CHECK();
    double l_band_lw, x_age_lw = star_age / 3.4e-3; // converts to code units, and defines age relative to convenient break time
    if(x_age_lw <= 1) {l_band_lw = 0.429e36 * (1.+x_age_lw*x_age_lw) * m_sol / UNIT_LUM_IN_CGS;}
        else {l_band_lw = 0.962e36 * pow(x_age_lw,-1.6) * exp(-x_age_lw/117.6) * m_sol / UNIT_LUM_IN_CGS;} // 0.1 solar, with nebular. very weak metallicity dependence, with slightly slower decay in time for lower-metallicity pops; effect smaller than binaries
    lum[RT_FREQ_BIN_LYMAN_WERNER] = l_band_lw; // band luminosity //
#endif

#if defined(RT_CHEM_PHOTOION)   /* Hydrogen and Helium ionizing bands */
    SET_ACTIVE_RT_CHECK();
    double l_ion = particle_ionizing_luminosity_in_cgs_core(i, pp) / UNIT_LUM_IN_CGS; /* calculate ionizing flux based on actual stellar or BH physics */
    lum[RT_FREQ_BIN_H0] = l_ion; // default to putting everything into a single band //
#if defined(RT_PHOTOION_MULTIFREQUENCY)
    int i_vec[4] = {RT_FREQ_BIN_H0, RT_FREQ_BIN_He0, RT_FREQ_BIN_He1, RT_FREQ_BIN_He2}; // these will all be the same if not using multi-frequency module //
    int k; for(k=0;k<4;k++) {lum[i_vec[k]] = l_ion * All.rt_ion_precalc_stellar_luminosity_fraction[i_vec[k]];} // assign flux appropriately according to pre-tabulated result //
#endif
#endif

#if defined(RT_HARD_XRAY) || defined(RT_SOFT_XRAY) /* soft and hard X-rays for e.g. Compton heating by X-ray binaries */
    SET_ACTIVE_RT_CHECK();
    double L_HMXBs=0; if(star_age > 0.01) {L_HMXBs = 1.0e29 / (star_age*star_age);}
#if defined(RT_SOFT_XRAY)
    lum[RT_FREQ_BIN_SOFT_XRAY] = (8.2e27 + 0.4*L_HMXBs) * m_sol / UNIT_LUM_IN_CGS; // LMXBs+HMXBs
#endif
#if defined(RT_HARD_XRAY)
    lum[RT_FREQ_BIN_HARD_XRAY] = (6.3e27 + 0.6*L_HMXBs) * m_sol / UNIT_LUM_IN_CGS; // LMXBs+HMXBs
#endif
#endif

#endif
    return active_check;
}



/* subroutine for 'rt_get_source_luminosity', with identical variables, for cases where the radiation
   represents flux from sink particles representing AGN, i.e. the sort of thing which would be used in galaxy simulations.
*/
KOKKOS_INLINE_FUNCTION int rt_get_lum_band_agn(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(pp[i].Type != 5) {return 0;} // only go forward for BH-type particles
    if(pp[i].Mass <= 0 || !isfinite(pp[i].Mass)) {return 0;} // (P[].Mass is the dynamical-mass SSOT; CellP[].Mass is gas-only and uninitialized for Type 5 sinks)
    int active_check = 0; // default to inactive //
#if defined(SINK_PARTICLES)
    double l_bol = sink_lum_bol_core(pp[i].Sink_Mdot,pp[i].Sink_Mass,i,pp); if(l_bol <= 0) {return 0;} // no accretion luminosity -- no point in going further! (sink_lum_bol takes the accreted sink mass — P[].Sink_Mass — not the dynamical particle mass)
    // corrections below follow  Shen, PFH, et al. 2020 to account for alpha-ox and template spectrum to get AGN set in different bands as a function of bolometric luminosity. functional form very similar to Hopkins, Richards, & Hernquist 2007, but updated values. //
    double lbol_lsun = l_bol * UNIT_LUM_IN_SOLAR, R_opt_xr; // luminosity in physical code units //
    double f_xr_0=0.0461795, R_xr_opt = pow(lbol_lsun/1.e10,0.026) / (0.0455713 + 0.140974*pow(lbol_lsun/1.e10,0.304)), Rfxr=R_xr_opt*f_xr_0; // x-ray to optical ratio normalized to its value at Lbol=1e13 solar
    if(Rfxr > 0.5) {R_xr_opt /= pow(1.+Rfxr*Rfxr*Rfxr*Rfxr, 0.25);} // this just prevents unphysical divergences
    R_opt_xr = (1.-R_xr_opt*f_xr_0) / (1.-f_xr_0); // this corrects the IR/optical/UV portion of the spectrum

#if defined(RT_INFRARED) /* special mid-through-far infrared band, which includes IR radiation temperature evolution */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_INFRARED] = 0.273 * R_opt_xr * l_bol;
#endif
#if defined(RT_OPTICAL_NIR) /* Optical-NIR approximate spectra for stars as used in the FIRE (Hopkins et al.) models; from 0.41-3.4 eV */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_OPTICAL_NIR] = 0.181 * R_opt_xr * l_bol;
#endif
#if defined(RT_NUV) /* Near-UV approximate spectra (UV/optical spectra, sub-photo-electric, but high-opacity) for stars as used in the FIRE (Hopkins et al.) models; from 3.4-8 eV */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_NUV] = 0.141 * R_opt_xr * l_bol;
#endif
#ifdef RT_PHOTOELECTRIC /* photo-electric bands (8-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_PHOTOELECTRIC] = 0.117 * R_opt_xr * l_bol; // broad band here [note can 2x-count with LW because that is a sub-band, but include it b/c need to total for dust PE heating
#endif
#ifdef RT_LYMAN_WERNER  /* lyman-werner bands (11.2-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_LYMAN_WERNER] = 0.0443 * R_opt_xr * l_bol;
#endif
#if defined(RT_CHEM_PHOTOION)   /* Hydrogen and Helium ionizing bands */
    SET_ACTIVE_RT_CHECK();
#if defined(RT_PHOTOION_MULTIFREQUENCY)
    lum[RT_FREQ_BIN_H0]  = 0.1130 * R_opt_xr * l_bol; // total ionizing flux: 13.6-24.6
    lum[RT_FREQ_BIN_He0] = 0.0820 * R_opt_xr * l_bol; // total ionizing flux: 24.6-54.5
    lum[RT_FREQ_BIN_He1] = 0.0111 * R_opt_xr * l_bol; // total ionizing flux: 54.5-70
    lum[RT_FREQ_BIN_He2] = 0.0243 * R_opt_xr * l_bol; // total ionizing flux: 70-500
#else
    lum[RT_FREQ_BIN_H0] = 0.230 * R_opt_xr * l_bol; // total ionizing flux
#endif
#endif
#if defined(RT_SOFT_XRAY) /* soft x-ray 0.5-2 keV band, for compton heating */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_SOFT_XRAY] = 0.00803 * R_xr_opt * l_bol;
#endif
#if defined(RT_HARD_XRAY) /* hard x-ray 2-10+ keV band, for compton heating; since used for that we include some higher-frequence radiation as well */
    SET_ACTIVE_RT_CHECK(); lum[RT_FREQ_BIN_HARD_XRAY] = 2.33 * 0.0113 * R_xr_opt * l_bol; // [2.33 factor is extrapolating to include -ultra-hard- X-rays beyond 10keV, useful for some Compton heating estimates]
#endif
    /* note: once account for 2x-counting of LW and PE bands above, this adds up to almost the entire Lbol, but fraction ~ 0.0236 * R_xr_opt * l_bol remains, divided between radio [radio-quiet agn here] and gamma-rays */


#endif
    return active_check;
}



/* subroutine for 'rt_get_source_luminosity', with identical variables, for cases where the radiation
   represents flux from individual sink or star particles representing individual stars or protostars,
   i.e. the sort of thing which would be used in star or planet formation simulations.
*/
KOKKOS_INLINE_FUNCTION int rt_get_lum_band_singlestar(int i, int mode, double *lum, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(pp[i].Type < 4) {return 0;} // only go forward with star or sink-type particles
    if(pp[i].Mass <= 0 || !isfinite(pp[i].Mass)) {return 0;} // (P[].Mass is the dynamical-mass SSOT for stars/sinks; CellP[].Mass is gas-only and uninitialized for non-gas)
    int active_check = 0, k; // default to inactive //

#if defined(RT_INFRARED) /* special mid-through-far infrared band, which includes IR radiation temperature evolution */
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_INFRARED; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
#endif
#if defined(RT_OPTICAL_NIR) /* Optical-NIR approximate spectra for stars as used in the FIRE (Hopkins et al.) models; from 0.41-3.4 eV */
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_OPTICAL_NIR; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
#endif
#if defined(RT_NUV) /* Near-UV approximate spectra (UV/optical spectra, sub-photo-electric, but high-opacity) for stars as used in the FIRE (Hopkins et al.) models; from 3.4-8 eV */
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_NUV; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
#endif
#ifdef RT_PHOTOELECTRIC /* photo-electric bands (8-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_PHOTOELECTRIC; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell); // broad band here [note can 2x-count with LW because that is a sub-band, but include it b/c need to total for dust PE heating
#endif
#ifdef RT_LYMAN_WERNER  /* lyman-werner bands (11.2-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_LYMAN_WERNER; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
#endif
#if defined(GALSF_FB_FIRE_RT_LONGRANGE) && defined(RADTRANSFER) /* set of FIRE default bands, if used here for stars as well, though currently not cross-linked with some of the other physics */
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_FIRE_UV; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
    k=RT_FREQ_BIN_FIRE_OPT; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
    k=RT_FREQ_BIN_FIRE_IR; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell);
#endif
#if defined(RT_CHEM_PHOTOION)   /* Hydrogen and Helium ionizing bands */
    SET_ACTIVE_RT_CHECK();
#if defined(RT_PHOTOION_MULTIFREQUENCY)
    int i_vec[4] = {RT_FREQ_BIN_H0, RT_FREQ_BIN_He0, RT_FREQ_BIN_He1, RT_FREQ_BIN_He2}; // these will all be the same if not using multi-frequency module //
    for(k=0;k<4;k++) {lum[i_vec[k]] = stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[i_vec[k]],All.RHD_bins_nu_max_ev[i_vec[k]], pp, cell);} // integrate between band boundaries, defined in global 'nu' in eV
#else
    SET_ACTIVE_RT_CHECK(); k=RT_FREQ_BIN_H0; lum[k]=stellar_lum_in_band(i,All.RHD_bins_nu_min_ev[k],All.RHD_bins_nu_max_ev[k], pp, cell); // total ionizing flux
#ifdef RT_STARBENCH_TEST
    lum[RT_FREQ_BIN_H0] = 1e49 * (All.rt_nu_eff_eV[RT_FREQ_BIN_H0]*ELECTRONVOLT_IN_ERGS) / UNIT_LUM_IN_CGS;
#endif
#endif
#endif

#ifdef RT_SOFT_XRAY
    /* currently assume zero here, need to add function here if desired from XRBs or coronal activity, b/c model assumes a thermal spectrum which will give null */
#endif
#ifdef RT_HARD_XRAY
    /* currently assume zero here, need to add function here if desired from XRBs or coronal activity, b/c model assumes a thermal spectrum which will give null */
#endif
#ifdef RT_FREEFREE
    /* negligible free-free emissivity from stars here */
#endif
#ifdef RT_GENERIC_USER_FREQ
    /* code whatever is desired */
#endif

    return active_check;
}

KOKKOS_INLINE_FUNCTION void rt_get_lum_gas(int target, double *je, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef RT_FREEFREE
    int k = RT_FREQ_BIN_FREEFREE;
    double t_eff = cell[target].gas_temperature_from_u(cell[target].InternalEnergyPred); // temperature from the composition the cooling solve stored, or the ionized fallback where none was, nothing fancy, to get the temperature //
    je[k] = rt_absorb_frac_albedo(target, k, pp, cell) * rt_kappa(target,k, pp, cell) * cell[target].Mass * ((4. * 5.67e-5) * t_eff*t_eff*t_eff*t_eff) / UNIT_FLUX_IN_CGS; // blackbody emissivity (Kirchoff's law): account for albedo [absorption opacity], and units //
#endif
}

KOKKOS_INLINE_FUNCTION double stellar_lum_in_band(int i, double E_lower, double E_upper, struct particle_data *pp, struct gas_cell_data *cell)
{
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    double r_sol = pp[i].ProtoStellarRadius_inSolar, l_sol = pp[i].StarLuminosity_Solar; // both protostellar models: cadence-owned luminosity + radius, read stale-by-contract
#elif defined(SINGLE_STAR_SINK_DYNAMICS) // no protostellar-evolution model: generic fits based on mass
    double l_sol=sink_lum_bol_core(0,pp[i].Sink_Mass,i,pp)*UNIT_LUM_IN_SOLAR, m_sol=pp[i].Sink_Mass*UNIT_MASS_IN_SOLAR, r_sol=pow(m_sol,0.738); // L/Lsun, M/Msun, R/Rsun. (Under SINGLE_STAR_SINK_DYNAMICS the relevant mass for the sink luminosity + radius scaling is the accreted sink mass — P[].Sink_Mass — not the dynamical particle mass and definitely not CellP[].Mass which is gas-only.)
#else
    double l_sol=1., r_sol=1.; // nothing usefully defined for the above - default to solar-type stars //
#endif
    double T_eff = 5780. * pow(l_sol/(r_sol*r_sol), 0.25);
    double f = blackbody_lum_frac(E_lower, E_upper, T_eff);
    return f * l_sol / UNIT_LUM_IN_SOLAR;
}

#if defined(CHIMES_STELLAR_FLUXES) && (defined(RADTRANSFER) || defined(RT_USE_GRAVTREE))
KOKKOS_INLINE_FUNCTION double chimes_G0_luminosity(double stellar_age, double stellar_mass) // age in Myr, mass in Msol, return value in Habing units * cm^2
{
  double zeta = 6.5006802e29;
  if (stellar_age < 4.07) {return stellar_mass * exp(89.67 + (0.172 * pow(stellar_age, 0.916)));}
    else {return stellar_mass * zeta * pow(1773082.52 / stellar_age, 1.667) * pow(1.0 + pow(stellar_age / 1773082.52, 28.164), 1.64824);}
}

KOKKOS_INLINE_FUNCTION double chimes_ion_luminosity(double stellar_age, double stellar_mass) // age in Myr, mass in Msol, return value in s^-1
{
  double zeta = 3.2758118e21;
  if (stellar_age < 3.71) {return stellar_mass * exp(107.21 + (0.111 * pow(stellar_age, 0.974)));}
    else {return stellar_mass * zeta * pow(688952.27 / stellar_age, 4.788) * pow(1.0 + pow(stellar_age / 688952.27, 1.124), -17017.50356);}
}

KOKKOS_INLINE_FUNCTION int rt_get_source_luminosity_chimes(int i, int mode, double *lum, double *chimes_lum_G0, double *chimes_lum_ion, struct particle_data *pp, struct gas_cell_data *cell)
{
    int value_to_return = 0;
    value_to_return = rt_get_source_luminosity(i, mode, lum, pp, cell); // call routine as normal for all bands, before adding chimes-specific details
    if( is_galsf_stellar_candidate_type(pp[i].Type, All.ComovingIntegrationOn) && (pp[i].Mass>0) && (pp[i].KernelRadius>0) ) // (P[].Mass is SSOT for stars; CellP[].Mass is gas-only)
    {
        int age_bin, j; double age_Myr=1000.*evaluate_stellar_age_Gyr_core(i, pp), log_age_Myr=log10(age_Myr), stellar_mass=pp[i].Mass*UNIT_MASS_IN_SOLAR;
        if(log_age_Myr < CHIMES_LOCAL_UV_AGE_LOW) {age_bin = 0;} else if (log_age_Myr < CHIMES_LOCAL_UV_AGE_MID) {age_bin = (int) floor(((log_age_Myr - CHIMES_LOCAL_UV_AGE_LOW) / CHIMES_LOCAL_UV_DELTA_AGE_LOW) + 1);} else {
            age_bin = (int) floor((((log_age_Myr - CHIMES_LOCAL_UV_AGE_MID) / CHIMES_LOCAL_UV_DELTA_AGE_HI) + ((CHIMES_LOCAL_UV_AGE_MID - CHIMES_LOCAL_UV_AGE_LOW) / CHIMES_LOCAL_UV_DELTA_AGE_LOW)) + 1);
            if (age_bin > CHIMES_LOCAL_UV_NBINS - 1) {age_bin = CHIMES_LOCAL_UV_NBINS - 1;}}

        for(j=0;j<CHIMES_LOCAL_UV_NBINS;j++) {chimes_lum_G0[j]=0; chimes_lum_ion[j]=0;}
        chimes_lum_G0[age_bin] = chimes_G0_luminosity(age_Myr,stellar_mass) * All.Chimes_f_esc_G0;
        chimes_lum_ion[age_bin] = chimes_ion_luminosity(age_Myr,stellar_mass) * All.Chimes_f_esc_ion;
    }
    return value_to_return;
}
#endif

#undef SET_ACTIVE_RT_CHECK
#endif /* RADTRANSFER || RT_USE_GRAVTREE */
#endif /* GRAVTREE_SOURCE_HOST_OWNER_TU || (GRAVTREE_SOURCE_DEVICE_TU && GRAVTREE_SOURCE_LAZY_SUPPORTED) */
