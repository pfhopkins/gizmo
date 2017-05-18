#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"
#ifdef PTHREADS_NUM_THREADS
#include <pthread.h>
#endif
#ifdef PTHREADS_NUM_THREADS
extern pthread_mutex_t mutex_nexport;
extern pthread_mutex_t mutex_partnodedrift;
#define LOCK_NEXPORT     pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT   pthread_mutex_unlock(&mutex_nexport);
#else
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#endif

/*! \file rt_utilities.c
 *  \brief useful functions for radiation modules
 *
 *  This file contains a variety of useful functions having to do with radiation in different modules
 *    A number of the radiative transfer subroutines and more general mass-to-light ratio calculations
 *    will refer to these routines.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */



/***********************************************************************************************************
 *
 * ROUTINES IN THIS BLOCK MUST BE MODIFIED FOR NEW MODULES USING DIFFERENT WAVEBANDS/PHYSICS
 *
 *  (these routines depend on compiler-time choices for which frequencies will be followed, and the 
 *    physics used to determine things like the types of source particles, source luminosities, 
 *    and how opacities are calculated)
 *
 ***********************************************************************************************************/



#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)

/***********************************************************************************************************/
/* routine which returns the luminosity for the desired source particles, as a function of whatever the user desires, in the relevant bands */
/***********************************************************************************************************/
int rt_get_source_luminosity(int i, double sigma_0, double *lum)
{
    int active_check = 0;
    

#if defined(RADTRANSFER) && (defined(GALSF))
    /* restrict the star particles acting as sources, because they need their own sub-loops */
    if( ((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3)))) && P[i].Mass>0 && PPP[i].Hsml>0 )
    {
        double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
	    if((star_age > 0.1)||(star_age <= 0)||(isnan(star_age))) return 0;
    }
#endif

    
#ifdef GALSF_FB_RT_PHOTONMOMENTUM
    /* three-band (UV, OPTICAL, IR) approximate spectra for stars as used in the FIRE (Hopkins et al.) models */
    if( ((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3)))) && P[i].Mass>0 && PPP[i].Hsml>0 )
    {
        if(sigma_0<0) {return 1;} active_check = 1;
        double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
        double L = P[i].Mass * evaluate_l_over_m_ssp(star_age) * calculate_relative_light_to_mass_ratio_from_imf(i);
        if((L<=0)||(star_age<=0)||(isnan(star_age))||(isnan(L))) {L=0; star_age=0;}
        double f_uv, f_op;
#ifndef RT_FIRE_FIX_SPECTRAL_SHAPE
        double sigma_eff = sigma_0 * evaluate_NH_from_GradRho(P[i].GradRho,PPP[i].Hsml,P[i].DensAroundStar,PPP[i].NumNgb,0);
        if((sigma_eff <= 0)||(isnan(sigma_eff))) {sigma_eff=0;}
        if(star_age <= 0.0025) {f_op=0.09;} else {
            if(star_age <= 0.006) {f_op=0.09*(1+((star_age-0.0025)/0.004)*((star_age-0.0025)/0.004));
            } else {f_op=1-0.8410937/(1+sqrt((star_age-0.006)/0.3));}}
        
        double tau_uv = sigma_eff*KAPPA_UV; double tau_op = sigma_eff*KAPPA_OP;
        f_uv = (1-f_op)*(All.PhotonMomentum_fUV + (1-All.PhotonMomentum_fUV)/(1+0.8*tau_uv+0.85*tau_uv*tau_uv));
        f_op *= All.PhotonMomentum_fOPT + (1-All.PhotonMomentum_fOPT)/(1+0.8*tau_op+0.85*tau_op*tau_op);
        /*
         // above is a fitting function for tau_disp~0.22 'tail' w. exp(-tau) 'core', removes expensive functions
         f_uv = (1-f_op)*(All.PhotonMomentum_fUV + (1-All.PhotonMomentum_fUV)*exp(-tau_uv));
         f_op *= All.PhotonMomentum_fOPT + (1-All.PhotonMomentum_fOPT)*exp(-tau_op);
         // leakage for P(tau) ~ exp(-|logtau/tau0|/sig):
         f_uv = (1-f_op)*(All.PhotonMomentum_fUV + (1-All.PhotonMomentum_fUV)/ (1 + pow(tau_uv,1./(4.*tau_disp))/(3.*tau_disp) + pow(2.*tau_disp*tau_uv,1./tau_disp)));
         f_op *= All.PhotonMomentum_fOPT + (1-All.PhotonMomentum_fOPT)/ (1 + pow(tau_op,1./(4.*tau_disp))/(3.*tau_disp) + pow(2.*tau_disp*tau_op,1./tau_disp));
         */
#else
        f_uv = All.PhotonMomentum_fUV;
        f_op = All.PhotonMomentum_fOPT;
#endif
        lum[RT_FREQ_BIN_FIRE_UV]  = L * f_uv;
        lum[RT_FREQ_BIN_FIRE_OPT] = L * f_op;
        lum[RT_FREQ_BIN_FIRE_IR]  = L * (1-f_uv-f_op);
    }
#endif

#if defined(RT_INFRARED)
    /* can add direct infrared sources, but default to no direct IR (just re-emitted light) */
    if((1 << P[i].Type) & (RT_SOURCES))
    {
        lum[RT_FREQ_BIN_INFRARED] = 0.0;
    }
#endif
    
#if defined(RT_OPTICAL_NIR)
    /* Optical-NIR approximate spectra for stars as used in the FIRE (Hopkins et al.) models */
    if((1 << P[i].Type) & (RT_SOURCES)) // check if the particle falls into one of the allowed source types
    {
        if( ((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3)))) && P[i].Mass>0 && PPP[i].Hsml>0 )
        {
            if(sigma_0<0) {return 1;} active_check = 1;
            double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge), f_op=0;
            if(star_age <= 0.0025) {f_op=0.09;} else {
                if(star_age <= 0.006) {f_op=0.09*(1+((star_age-0.0025)/0.004)*((star_age-0.0025)/0.004));
                } else {f_op=1-0.8410937/(1+sqrt((star_age-0.006)/0.3));}}
            double fac = 3.95e33 * (P[i].Mass * All.UnitMass_in_g / SOLAR_MASS) * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs); // converts to code units
            lum[RT_FREQ_BIN_OPTICAL_NIR] = f_op * fac * evaluate_l_over_m_ssp(star_age) * calculate_relative_light_to_mass_ratio_from_imf(i);
        }
    }
#endif

#ifdef RT_PHOTOELECTRIC
    /* photo-electric bands (8-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    if((1 << P[i].Type) & (RT_SOURCES)) // check if the particle falls into one of the allowed source types
    {
        if( ((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3)))) && P[i].Mass>0 && PPP[i].Hsml>0 )
        {
            if(sigma_0<0) {return 1;} active_check = 1;
            double fac = (P[i].Mass * All.UnitMass_in_g / SOLAR_MASS) * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs); // converts to code units
            //double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge); 
            //double l_band = 2.14e36 / sqrt(1. + pow(star_age/4.e-3,3.6)) * fac; // solar tracks, no nebular
            double l_band, x_age = evaluate_stellar_age_Gyr(P[i].StellarAge) / 3.4e-3;
            if(x_age <= 1) 
            { 
                l_band = 1.07e36 * (1.+x_age*x_age) * fac;
            } else {
                l_band = 2.14e36 / (x_age * sqrt(x_age)) * fac;
            } // 0.1 solar, with nebular. very weak metallicity dependence, with slightly slower decay in time for lower-metallicity pops; effect smaller than binaries
            lum[RT_FREQ_BIN_PHOTOELECTRIC] = l_band; // band luminosity //
        }
    }
#endif
    

#ifdef RT_LYMAN_WERNER
    /* lyman-werner bands (8-13.6 eV, specifically): below is from integrating the spectra from STARBURST99 with the Geneva40 solar-metallicity + lower tracks */
    if((1 << P[i].Type) & (RT_SOURCES)) // check if the particle falls into one of the allowed source types
    {
        if( ((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3)))) && P[i].Mass>0 && PPP[i].Hsml>0 )
        {
            if(sigma_0<0) {return 1;} active_check = 1;
            double fac = (P[i].Mass * All.UnitMass_in_g / SOLAR_MASS) * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs); // converts to code units
            double l_band, x_age = evaluate_stellar_age_Gyr(P[i].StellarAge) / 3.4e-3;
            if(x_age <= 1) 
            { 
                l_band = 0.429e36 * (1.+x_age*x_age) * fac;
            } else {
                l_band = 0.962e36 * pow(x_age,-1.6) * exp(-x_age/117.6) * fac;
            } // 0.1 solar, with nebular. very weak metallicity dependence, with slightly slower decay in time for lower-metallicity pops; effect smaller than binaries
            lum[RT_FREQ_BIN_LYMAN_WERNER] = l_band; // band luminosity //
        }
    }
#endif
        
    
#if defined(RT_CHEM_PHOTOION)
    /* Hydrogen and Helium ionizing bands */
    if((1 << P[i].Type) & (RT_SOURCES)) // check if the particle falls into one of the allowed source types
    {
        lum[RT_FREQ_BIN_H0] = 0; // begin from zero //
        double fac = 0;
#if defined(GALSF) || defined(GALSF_FB_HII_HEATING)
        /* calculate ionizing flux based on actual stellar or BH physics */
        if( ((P[i].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[i].Type == 2)||(P[i].Type==3)))) && P[i].Mass>0 && PPP[i].Hsml>0 )
        {
            if(sigma_0<0) {return 1;} active_check=1;
            fac += particle_ionizing_luminosity_in_cgs(i) * (All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs));
        }
#else
#ifdef RT_ILIEV_TEST1
        if(P[i].Type==4) {if(sigma_0<0) {return 1;} active_check=1; fac += 5.0e48 * (13.6*ELECTRONVOLT_IN_ERGS) * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs);} // 5e48 in ionizing photons per second //
#else
        if(P[i].Type==4) {if(sigma_0<0) {return 1;} active_check=1; fac += (P[i].Mass * All.UnitMass_in_g / SOLAR_MASS) * All.IonizingLuminosityPerSolarMass_cgs * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs);} // flux from star particles according to mass
#endif
#endif // GALSF else
#if defined(RT_PHOTOION_MULTIFREQUENCY)
        // we should have pre-tabulated how much luminosity gets assigned to each different waveband according to the following function //
        lum[RT_FREQ_BIN_He0]=lum[RT_FREQ_BIN_He1]=lum[RT_FREQ_BIN_He2]=0;
        int k; for(k=0;k<N_RT_FREQ_BINS;k++) {lum[k] += fac * precalc_stellar_luminosity_fraction[k];}
#else
        lum[RT_FREQ_BIN_H0] += fac;
#endif
    }
#endif // RT_CHEM_PHOTOION



#if defined(RT_HARD_XRAY) || defined(RT_SOFT_XRAY)
    if((1 << P[i].Type) & (RT_SOURCES)) // check if the particle falls into one of the allowed source types
    {
#if defined(RT_HARD_XRAY) 
            lum[RT_FREQ_BIN_HARD_XRAY] = 0; // LMXBs+HMXBs
#endif
#if defined(RT_SOFT_XRAY) 
            lum[RT_FREQ_BIN_SOFT_XRAY] = 0; // LMXBs+HMXBs
#endif
#if defined(BLACK_HOLES)
        if(P[i].Type == 5) 
        {
            if(sigma_0<0) {return 1;} active_check=1;
            double lbol = bh_lum_bol(P[i].BH_Mdot,P[i].Mass,i);
            double lbol_lsun = lbol / (SOLAR_LUM * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs));
            double bol_corr = 0;
#if defined(RT_HARD_XRAY) 
            bol_corr = 0.43 * (10.83 * pow(lbol_lsun/1.e10,0.28) + 6.08 * pow(lbol_lsun/1.e10,-0.02)); // 0.5 for -ALL- hard-x-ray, 1.0 prefactor for just 2-10 keV
            lum[RT_FREQ_BIN_HARD_XRAY] = lbol / bol_corr; // typical bolometric correction from Hopkins, Richards, & Hernquist 2007 
#endif
#if defined(RT_SOFT_XRAY) 
            bol_corr = 17.87 * pow(lbol_lsun/1.e10,0.28) + 10.0 * pow(lbol_lsun/1.e10,-0.02);
            lum[RT_FREQ_BIN_SOFT_XRAY] = lbol / bol_corr; // typical bolometric correction from Hopkins, Richards, & Hernquist 2007 
#endif
        }
#endif
        if(P[i].Type == 4) 
        {
            if(sigma_0<0) {return 1;} active_check=1;
            double fac = (P[i].Mass * All.UnitMass_in_g / SOLAR_MASS) * All.UnitTime_in_s / (All.HubbleParam * All.UnitEnergy_in_cgs); // converts to code units
            double L_HMXBs = 0.0; 
#ifdef GALSF
            double star_age = evaluate_stellar_age_Gyr(P[i].StellarAge);
            if(star_age > 0.01) {L_HMXBs = 1.0e29 / (star_age*star_age);}
#endif
#if defined(RT_HARD_XRAY) 
            lum[RT_FREQ_BIN_HARD_XRAY] = (6.3e27 + 0.6*L_HMXBs) * fac; // LMXBs+HMXBs
#endif
#if defined(RT_SOFT_XRAY) 
            lum[RT_FREQ_BIN_HARD_XRAY] = (8.2e27 + 0.4*L_HMXBs) * fac; // LMXBs+HMXBs
#endif
        }
    }
#endif // RT_HARD_XRAY

    
#ifdef RADTRANSFER
    /* generic sub routines for gas as a source term */
    if((1 << P[i].Type) & (RT_SOURCES)) // check if the particle falls into one of the allowed source types
    {
        if(P[i].Type == 0)
        {
            if(sigma_0<0) {return 1;} active_check=1; // active //
            rt_get_lum_gas(i,lum); /* optionally re-distributes cooling flux as a blackbody */
            int k; for(k=0;k<N_RT_FREQ_BINS;k++) {lum[k] += SphP[i].Je[k];}        
        }
    }
#endif
    
    /* need to renormalize ALL sources for reduced speed of light */
    {int k; for(k=0;k<N_RT_FREQ_BINS;k++) {lum[k] *= RT_SPEEDOFLIGHT_REDUCTION;}}
    return active_check;
}




/***********************************************************************************************************/
/* calculate the opacity for use in radiation transport operations [in physical code units = L^2/M] */
/***********************************************************************************************************/
double rt_kappa(int i, int k_freq)
{    
    
#ifdef RT_CHEM_PHOTOION
    /* opacity to ionizing radiation for Petkova & Springel bands. note rt_update_chemistry is where ionization is actually calculated */
    double nH_over_Density = HYDROGEN_MASSFRAC / PROTONMASS * All.UnitMass_in_g / All.HubbleParam;
    double kappa = nH_over_Density * (SphP[i].HI + MIN_REAL_NUMBER) * rt_sigma_HI[k_freq];
#if defined(RT_CHEM_PHOTOION_HE) && defined(RT_PHOTOION_MULTIFREQUENCY)
    kappa += nH_over_Density * ((SphP[i].HeI + MIN_REAL_NUMBER) * rt_sigma_HeI[k_freq] + (SphP[i].HeII + MIN_REAL_NUMBER) * rt_sigma_HeII[k_freq]);
    if(k_freq==RT_FREQ_BIN_He0)  {return kappa;}
    if(k_freq==RT_FREQ_BIN_He1)  {return kappa;}
    if(k_freq==RT_FREQ_BIN_He2)  {return kappa;}
#endif
    if(k_freq==RT_FREQ_BIN_H0)  {return kappa;}
#endif


#if defined(RT_HARD_XRAY) || defined(RT_SOFT_XRAY) || defined(RT_PHOTOELECTRIC) || defined (GALSF_FB_RT_PHOTONMOMENTUM) || defined(RT_OPTICAL_NIR) || defined(RT_LYMAN_WERNER) || defined(RT_INFRARED)
    double fac = All.UnitMass_in_g * All.HubbleParam / (All.UnitLength_in_cm * All.UnitLength_in_cm); /* units */
    double Zfac = 1.0; // assume solar metallicity 
#ifdef METALS
    Zfac = P[i].Metallicity[0]/All.SolarAbundances[0];
#endif
#ifdef RT_HARD_XRAY
    /* opacity comes from H+He (Thompson) + metal ions */
    if(k_freq==RT_FREQ_BIN_HARD_XRAY) {return (0.53 + 0.27*Zfac) * fac;}
#endif
#ifdef RT_SOFT_XRAY
    /* opacity comes from H+He (Thompson) + metal ions */
    if(k_freq==RT_FREQ_BIN_SOFT_XRAY) {return (127. + 50.0*Zfac) * fac;}
#endif
#ifdef GALSF_FB_RT_PHOTONMOMENTUM
    /* three-band (UV, OPTICAL, IR) approximate spectra for stars as used in the FIRE (Hopkins et al.) models */
    if(k_freq==RT_FREQ_BIN_FIRE_UV)  {return KAPPA_UV * fac;}
    if(k_freq==RT_FREQ_BIN_FIRE_OPT) {return KAPPA_OP * fac;}
    if(k_freq==RT_FREQ_BIN_FIRE_IR)  {return KAPPA_IR * fac;}
#endif
#ifdef RT_PHOTOELECTRIC
    /* opacity comes primarily from dust (ignoring H2 molecular opacities here) */
    if(k_freq==RT_FREQ_BIN_PHOTOELECTRIC) {return 2000. * DMAX(1.e-4,Zfac) * fac;}
#endif
#ifdef RT_LYMAN_WERNER
    /* opacity from molecular H2 and dust (dominant at higher-metallicity) should be included */
    if(k_freq==RT_FREQ_BIN_LYMAN_WERNER) {return 2400.*Zfac * fac;} // just dust term for now
#endif
#ifdef RT_OPTICAL_NIR
    /* opacity comes primarily from dust */
    if(k_freq==RT_FREQ_BIN_OPTICAL_NIR) {return 180.*Zfac * fac;}
#endif
#ifdef RT_INFRARED
    /* IR with dust opacity */
    if(k_freq==RT_FREQ_BIN_INFRARED)
    {
        if(isnan(SphP[i].Dust_Temperature) || SphP[i].Dust_Temperature<=0) {SphP[i].Dust_Temperature=DMAX(10.,2.73/All.cf_atime);} // reset baseline
        if(isnan(SphP[i].Radiation_Temperature) || SphP[i].Radiation_Temperature<=0) {SphP[i].Radiation_Temperature=DMAX(10.,2.73/All.cf_atime);} // reset baseline
        
        double T_dust_em = SphP[i].Dust_Temperature; // dust temperature in K //
        double Trad = SphP[i].Radiation_Temperature; // radiation temperature in K //
        if(Trad <= 0) {Trad = 5600.;}
        double kappa = 0.0; 
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
        if(T_dust_em < 1500.) // < 1500 K, dust is present
        {
            double x = 4.*log10(Trad) - 8.; // needed for fitting functions to opacities (may come up with cheaper function later)
            double dx_excess=0; if(x > 7.) {dx_excess=x-7.; x=7.;} // cap for maximum temperatures at which fit-functions should be used //
            if(x < -4.) {x=-4.;} // cap for minimum temperatures at which fit functions below should be used //
            if(T_dust_em < 160.) // Tdust < 160 K (all dust constituents present)
            {
                kappa = exp(0.72819004 + 0.75142468*x - 0.07225763*x*x - 0.01159257*x*x*x + 0.00249064*x*x*x*x);
            } else if(T_dust_em < 275.) { // 160 < Tdust < 275 (no ice present)
                kappa = exp(0.16658241 + 0.70072926*x - 0.04230367*x*x - 0.01133852*x*x*x + 0.0021335*x*x*x*x);
            } else if(T_dust_em < 425.) { // 275 < Tdust < 425 (no ice or volatile organics present)
                kappa = exp(0.03583845 + 0.68374146*x - 0.03791989*x*x - 0.01135789*x*x*x + 0.00212918*x*x*x*x);        
            } else if(T_dust_em < 680.) { // 425 < Tdust < 680 (silicates, iron, & troilite present)
                kappa = exp(-0.76576135 + 0.57053532*x - 0.0122809*x*x - 0.01037311*x*x*x + 0.00197672*x*x*x*x);
            } else { // 680 < Tdust < 1500 (silicates & iron present)
                kappa = exp(-2.23863222 + 0.81223269*x + 0.08010633*x*x + 0.00862152*x*x*x - 0.00271909*x*x*x*x);
            }
            if(dx_excess > 0) {kappa *= exp(0.57*dx_excess);} // assumes kappa scales linearly with temperature (1/lambda) above maximum in fit; pretty good approximation //
            kappa *= Zfac; // the above are all dust opacities, so they scale with metallicity
        }
        kappa += 0.35; // Thompson scattering
        return kappa * fac; // convert units and return
    }
#endif
#endif
    
    
    return 0;
}


#endif // #if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)






/***********************************************************************************************************
 *
 * ROUTINES WHICH DO NOT NEED TO BE MODIFIED SHOULD GO BELOW THIS BREAK
 *
 *  (these routines may depend on the RT solver or other numerical choices, but below here, place routines 
 *    which shouldn't needed to be hard-coded for different assumptions about the bands of the RT module, etc)
 *
 ***********************************************************************************************************/



/***********************************************************************************************************/
/* rate of photon absorption [absorptions per unit time per photon]: this, times the timestep dt, times the photon energy density E,
    gives the change in the energy density from absorptions (the sink term) */
/***********************************************************************************************************/
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
double rt_absorption_rate(int i, int k_freq)
{
    /* should be equal to (C * Kappa_opacity * rho) */
    return RT_SPEEDOFLIGHT_REDUCTION * (C/All.UnitVelocity_in_cm_per_s) * rt_kappa(i, k_freq) * SphP[i].Density*All.cf_a3inv;
}
#endif 




#ifdef RADTRANSFER

/***********************************************************************************************************/
/* returns the photon diffusion coefficient = fluxlimiter * c_light / (kappa_opacity * density)  [physical units] */
/***********************************************************************************************************/
double rt_diffusion_coefficient(int i, int k_freq)
{
    double c_light = (C / All.UnitVelocity_in_cm_per_s) * RT_SPEEDOFLIGHT_REDUCTION;
    return SphP[i].Lambda_FluxLim[k_freq] * c_light / (1.e-37 + SphP[i].Kappa_RT[k_freq] * SphP[i].Density*All.cf_a3inv);
}



/***********************************************************************************************************/
/* calculate the eddington tensor according to the M1 formalism (for use with that solver, obviously) */
/***********************************************************************************************************/
void rt_eddington_update_calculation(int j)
{
#ifdef RT_M1
    int k_freq, k;
    double c_light = RT_SPEEDOFLIGHT_REDUCTION * (C/All.UnitVelocity_in_cm_per_s);
    double n_flux_j[3], fmag_j, V_j_inv = SphP[j].Density / P[j].Mass;
    for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
    {
        n_flux_j[0]=n_flux_j[1]=n_flux_j[2]=0;
        double flux_vol[3]; for(k=0;k<3;k++) {flux_vol[k] = SphP[j].Flux[k_freq][k] * V_j_inv;}
        fmag_j = 0; for(k=0;k<3;k++) {fmag_j += flux_vol[k]*flux_vol[k];}
        if(fmag_j <= 0) {fmag_j=0;} else {fmag_j=sqrt(fmag_j); for(k=0;k<3;k++) {n_flux_j[k]=flux_vol[k]/fmag_j;}}
        double f_chifac = RT_SPEEDOFLIGHT_REDUCTION * fmag_j / (c_light * SphP[j].E_gamma[k_freq] * V_j_inv);
        f_chifac = fmag_j / (1.e-37 + c_light * SphP[j].E_gamma[k_freq] * V_j_inv);
        if(f_chifac < 0) {f_chifac=0;}
        if(fmag_j <= 0) {f_chifac = 0;}
        // restrict values of f_chifac to physical range.  
        //   [optional, not here]: impose additional condition: if optically thick locally, use isotropic tensor
        double f_min = 0.01, f_max = 0.99;
        if((f_chifac < f_min) || (isnan(f_chifac))) {f_chifac = f_min;}
        if(f_chifac > f_max) {f_chifac = f_max;}
        double chi_j = (3.+4.*f_chifac*f_chifac) / (5. + 2.*sqrt(4. - 3.*f_chifac*f_chifac));
        double chifac_iso_j = 0.5 * (1.-chi_j);
        double chifac_n_j = 0.5 * (3.*chi_j-1.);
        for(k=0;k<6;k++)
        {
            if(k<3)
            {
                SphP[j].ET[k_freq][k] = chifac_iso_j + chifac_n_j * n_flux_j[k]*n_flux_j[k];
            } else {
                if(k==3) {SphP[j].ET[k_freq][k] = chifac_n_j * n_flux_j[0]*n_flux_j[1];} // recall, for ET: 0=xx,1=yy,2=zz,3=xy,4=yz,5=xz
                if(k==4) {SphP[j].ET[k_freq][k] = chifac_n_j * n_flux_j[1]*n_flux_j[2];}
                if(k==5) {SphP[j].ET[k_freq][k] = chifac_n_j * n_flux_j[0]*n_flux_j[2];}
            }
        }
    }
#endif
}




/***********************************************************************************************************/
/*
  routine which does the drift/kick operations on radiation quantities. separated here because we use a non-trivial
    update to deal with potentially stiff absorption terms (could be done more rigorously with something fully implicit in this 
    step, in fact). 
    mode = 0 == kick operation (update the conserved quantities)
    mode = 1 == predict/drift operation (update the predicted quantities)
 */
/***********************************************************************************************************/
void rt_update_driftkick(int i, double dt_entr, int mode)
{
#if defined(RT_EVOLVE_NGAMMA)
    int kf, k_tmp; 
#ifdef RT_INFRARED
    double E_abs_tot = 0;/* energy absorbed in other bands is transfered to IR, by default: track it here */
    double c_light = (C / All.UnitVelocity_in_cm_per_s) * RT_SPEEDOFLIGHT_REDUCTION;
    double E_gamma_tot = 0; // dust temperature defined by total radiation energy density //
    {int j; for(j=0;j<N_RT_FREQ_BINS;j++) {E_gamma_tot += SphP[i].E_gamma[j];}}
    double u_gamma = E_gamma_tot * (SphP[i].Density*All.cf_a3inv/P[i].Mass) * All.UnitPressure_in_cgs * All.HubbleParam*All.HubbleParam; // photon energy density in CGS //
    double Dust_Temperature_4 = All.UnitVelocity_in_cm_per_s * c_light * u_gamma / (4. * 5.67e-5); // estimated effective temperature of local rad field in equilibrium with dust emission //
    Dust_Temperature_4 /= RT_SPEEDOFLIGHT_REDUCTION*RT_SPEEDOFLIGHT_REDUCTION;
    SphP[i].Dust_Temperature = sqrt(sqrt(Dust_Temperature_4));
#endif
    for(k_tmp=0; k_tmp<N_RT_FREQ_BINS; k_tmp++)
    {
        kf = k_tmp; // normal loop 
#ifdef RT_INFRARED
        // need to do IR last after summing absorption from other bands //
        if(RT_FREQ_BIN_INFRARED < N_RT_FREQ_BINS-1) {if(kf == RT_FREQ_BIN_INFRARED) {kf = N_RT_FREQ_BINS-1;} if(kf == N_RT_FREQ_BINS-1) {kf = RT_FREQ_BIN_INFRARED;}}
#endif
        double e0;
        if(mode==0) {e0 = SphP[i].E_gamma[kf];} else {e0 = SphP[i].E_gamma_Pred[kf];}
        double a0 = -rt_absorption_rate(i,kf);
        double total_de_dt = SphP[i].Je[kf] + SphP[i].Dt_E_gamma[kf];
#ifdef RT_INFRARED
        if(kf == RT_FREQ_BIN_INFRARED)
        {
            double T_cmb = 2.73 / All.cf_atime; // don't let dust or radiation temperatures drop below T_cmb //
            if((mode==0) && (SphP[i].Dt_E_gamma[kf]!=0) && (dt_entr>0)) // only update temperatures on kick operations //
            {
                // advected radiation changes temperature of radiation field, before absorption //
                double dE_fac = SphP[i].Dt_E_gamma[kf] * dt_entr; // change in energy from advection
                double dTE_fac = SphP[i].Dt_E_gamma_T_weighted_IR * dt_entr; // T-weighted change from advection
                double dE_abs = -e0 * (1. - exp(a0*dt_entr)); // change in energy from absorption
                double rfac=1; if(dE_fac < -0.5*(e0+dE_abs)) {rfac=fabs(0.5*(e0+dE_abs))/fabs(dE_fac);} else {if(dE_fac > 0.5*e0) {rfac=0.5*e0/dE_fac;}}
                dE_fac*=rfac; dTE_fac*=rfac; // limit temperature change from advection to prevent spurious divergences
                
                double T_max = DMAX(SphP[i].Radiation_Temperature , dE_fac / dTE_fac);
                SphP[i].Radiation_Temperature = (e0 + dE_fac) / (MIN_REAL_NUMBER + DMAX(0., e0 / SphP[i].Radiation_Temperature + dTE_fac));
                SphP[i].Radiation_Temperature = DMIN(SphP[i].Radiation_Temperature, T_max);
                a0 = -rt_absorption_rate(i,kf); // update absorption rate using the new radiation temperature //
            }
            double total_emission_rate = E_abs_tot + fabs(a0)*e0 + SphP[i].Je[kf]; // add the summed absorption as emissivity here //
            total_de_dt = total_emission_rate + SphP[i].Dt_E_gamma[kf];
            if(fabs(a0)>0)
            {
                Dust_Temperature_4 = total_emission_rate * (SphP[i].Density*All.cf_a3inv/P[i].Mass) / (4. * (MIN_REAL_NUMBER + fabs(a0)) / c_light); // flux units
                Dust_Temperature_4 *= (All.UnitPressure_in_cgs * All.HubbleParam * All.HubbleParam * All.UnitVelocity_in_cm_per_s) / (5.67e-5); // convert to cgs
                Dust_Temperature_4 /= RT_SPEEDOFLIGHT_REDUCTION*RT_SPEEDOFLIGHT_REDUCTION;
                SphP[i].Dust_Temperature = sqrt(sqrt(Dust_Temperature_4));
                if(SphP[i].Dust_Temperature < T_cmb) {SphP[i].Dust_Temperature = T_cmb;} // dust temperature shouldn't be below CMB
            }
            if(mode==0) // only update temperatures on kick operations //
            {
                // dust absorption and re-emission brings T_rad towards T_dust: //
                double dE_abs = -e0 * (1. - exp(a0*dt_entr)); // change in energy from absorption
                double T_max = DMAX(SphP[i].Radiation_Temperature , SphP[i].Dust_Temperature); // should not exceed either initial temperature //
                SphP[i].Radiation_Temperature = (e0 + dE_abs + total_emission_rate*dt_entr) / ((e0 + dE_abs) / SphP[i].Radiation_Temperature + total_emission_rate*dt_entr / SphP[i].Dust_Temperature);
                SphP[i].Radiation_Temperature = DMIN(SphP[i].Radiation_Temperature, T_max);
            }
            if(SphP[i].Radiation_Temperature < T_cmb) {SphP[i].Radiation_Temperature = T_cmb;} // radiation temperature shouldn't be below CMB
        }
#endif
        
        /*---------------------------------------------------------------------------------------------------
            the following block is for absorption and special behavior where 
            photons absorbed in one band are re-radiated [or up/down-scattered] into another. 
            this must be hard-coded to maintain conservation (as opposed to treated as a source term) 
          -----------------------------------------------------------------------------------------------------*/
        if(fabs(a0)*dt_entr > 50.) {a0 *= 50./(fabs(a0)*dt_entr);}
        double abs_0 = DMAX(0,fabs(a0)*dt_entr); double slabfac = slab_averaging_function(abs_0); double e_abs_0=exp(-abs_0); if(abs_0>100.) {e_abs_0=0;}
        /* since we're taking exponentials and inverses of some large numbers here, need to be careful not to let floating point errors cause nan's */
        if((dt_entr <= 0.)||(a0 >= 0.)||(abs_0 <= 0.)) {abs_0=0.; slabfac=e_abs_0=1.;} else {if(abs_0 < 1.e-5) {slabfac=1.-0.5*abs_0; e_abs_0 = 1.-abs_0;} else {if(abs_0 > 100.) {slabfac = 1./abs_0; e_abs_0 = 0.;}}}
        double e0_postabs = e0*e_abs_0, de_postabs = total_de_dt * dt_entr * slabfac, f_min = 0.01;
        if(e0_postabs+de_postabs < f_min*e0_postabs) {slabfac *= fabs((1.-f_min)*e0_postabs)/(fabs(de_postabs)+MIN_REAL_NUMBER);}

        double ef = e0 * e_abs_0 + total_de_dt * dt_entr * slabfac; // gives exact solution for dE/dt = -E*abs + de , the 'reduction factor' appropriately suppresses the source term //
        if((ef < 0)||(isnan(ef))) {ef=0;}
        double de_abs = e0 + total_de_dt * dt_entr - ef; // energy removed by absorption alone
        if((dt_entr <= 0) || (de_abs <= 0)) {de_abs = 0;}
        int donor_bin = -1; // frequency into which the photons will be deposited, if any //
#if defined(RT_CHEM_PHOTOION) && defined(RT_INFRARED)
        /* assume absorbed ionizing photons are re-emitted via recombination into optical-NIR bins. valid if recombination time is fast. 
            more accurately, this should be separately calculated in the cooling rates, and gas treated as a source */
        if(kf==RT_FREQ_BIN_H0) {donor_bin=RT_FREQ_BIN_INFRARED;}
#ifdef RT_PHOTOION_MULTIFREQUENCY
        if(kf==RT_FREQ_BIN_He0) {donor_bin=RT_FREQ_BIN_INFRARED;}
        if(kf==RT_FREQ_BIN_He1) {donor_bin=RT_FREQ_BIN_INFRARED;}
        if(kf==RT_FREQ_BIN_He2) {donor_bin=RT_FREQ_BIN_INFRARED;}        
#endif
#endif
#if defined(RT_PHOTOELECTRIC) && defined(RT_INFRARED)
        /* this is direct dust absorption, re-radiated in IR */
        if(kf==RT_FREQ_BIN_PHOTOELECTRIC) {donor_bin=RT_FREQ_BIN_INFRARED;}
#endif
#if defined(RT_OPTICAL_NIR) && defined(RT_INFRARED)
        /* this is direct dust absorption, re-radiated in IR */
        if(kf==RT_FREQ_BIN_OPTICAL_NIR) {donor_bin=RT_FREQ_BIN_INFRARED;}
#endif
#ifdef RT_INFRARED
        /* donor bin is yourself in the IR - all self-absorption is re-emitted */
        if(donor_bin==RT_FREQ_BIN_INFRARED) {E_abs_tot += de_abs/(MIN_REAL_NUMBER + dt_entr);}
        if(kf==RT_FREQ_BIN_INFRARED) {ef = e0 + total_de_dt * dt_entr;} 
#endif
        if(donor_bin >= 0) {if(mode==0) {SphP[i].E_gamma[donor_bin] += de_abs;} else {SphP[i].E_gamma_Pred[donor_bin] += de_abs;}}
        if((ef < 0)||(isnan(ef))) {ef=0;}
        if(mode==0) {SphP[i].E_gamma[kf] = ef;} else {SphP[i].E_gamma_Pred[kf] = ef;}

#if defined(RT_EVOLVE_FLUX)
        int k_dir; double f_mag=0;
        for(k_dir=0;k_dir<3;k_dir++)
        {
            double flux_0;
            if(mode==0) {flux_0 = SphP[i].Flux[kf][k_dir];} else {flux_0 = SphP[i].Flux_Pred[kf][k_dir];}
            double flux_f = flux_0 * e_abs_0 + SphP[i].Dt_Flux[kf][k_dir] * dt_entr * slabfac; // gives exact solution for dE/dt = -E*abs + de , the 'reduction factor' appropriately suppresses the source term //
            if(mode==0) {SphP[i].Flux[kf][k_dir] = flux_f;} else {SphP[i].Flux_Pred[kf][k_dir] = flux_f;}
            f_mag += flux_f*flux_f; // magnitude of flux vector
        }
        if(f_mag > 0)
        {
            // limit the flux according the physical (optically thin) maximum //
            f_mag=sqrt(f_mag); double fmag_max = 1.1 * (C/All.UnitVelocity_in_cm_per_s) * ef; // maximum flux should be optically-thin limit: e_gamma/C: here allow some tolerance for numerical leapfrogging in timestepping
            if(f_mag > fmag_max) {for(k_dir=0;k_dir<3;k_dir++) {if(mode==0) {SphP[i].Flux[kf][k_dir] *= fmag_max/f_mag;} else {SphP[i].Flux_Pred[kf][k_dir] *= fmag_max/f_mag;}}}
        }
#endif
    }
    if(mode > 0) {rt_eddington_update_calculation(i);} /* update the eddington tensor (if we calculate it) as well */
#endif
}



#endif





#ifdef RADTRANSFER
/***********************************************************************************************************/
/* this function initializes some of the variables we need */
/***********************************************************************************************************/
void rt_set_simple_inits(void)
{
    int i; for(i = 0; i < N_gas; i++)
    {
        if(P[i].Type == 0)
        {
            int k;
#ifdef RT_INFRARED
            SphP[i].Dust_Temperature = DMAX(10.,2.73/All.cf_atime); // in K, floor = CMB temperature or 10K
            SphP[i].Radiation_Temperature = SphP[i].Dust_Temperature;
            SphP[i].Dt_E_gamma_T_weighted_IR = 0;
#endif
#ifdef RT_RAD_PRESSURE_OUTPUT
            for(k=0;k<3;k++) {SphP[i].RadAccel[k]=0;}
#endif
#ifdef RT_CHEM_PHOTOION
            SphP[i].HII = MIN_REAL_NUMBER;
            SphP[i].HI = 1.0 - SphP[i].HII;
            SphP[i].Ne = SphP[i].HII;
#ifdef RT_CHEM_PHOTOION_HE
            double fac = (1-HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
            SphP[i].HeIII = MIN_REAL_NUMBER * fac;
            SphP[i].HeII = MIN_REAL_NUMBER * fac;
            SphP[i].HeI = (1.0 - SphP[i].HeII - SphP[i].HeIII) * fac;
            SphP[i].Ne += SphP[i].HeII + 2.0 * SphP[i].HeIII;
#endif
#endif
            for(k = 0; k < N_RT_FREQ_BINS; k++)
            {
                SphP[i].E_gamma[k] = MIN_REAL_NUMBER;
                SphP[i].ET[k][0]=SphP[i].ET[k][1]=SphP[i].ET[k][2]=1./3.; SphP[i].ET[k][3]=SphP[i].ET[k][4]=SphP[i].ET[k][5]=0;
                SphP[i].Je[k] = 0;
                SphP[i].Kappa_RT[k] = rt_kappa(i,k);
                SphP[i].Lambda_FluxLim[k] = 1;
#ifdef RT_EVOLVE_NGAMMA
                SphP[i].E_gamma_Pred[k] = SphP[i].E_gamma[k];
                SphP[i].Dt_E_gamma[k] = 0;
#endif
#ifdef RT_EVOLVE_FLUX
                int k_dir;
                for(k_dir=0;k_dir<3;k_dir++)
                {
                    SphP[i].Flux_Pred[k][k_dir] = SphP[i].Flux[k][k_dir] = SphP[i].Dt_Flux[k][k_dir] = 0;
                }
#endif
            }
        }
    }
}
#endif







/***********************************************************************************************************/
/* optional routine to distribute cooling or other sources from gas: currently empty */
/***********************************************************************************************************/
void rt_get_lum_gas(int target, double *je)
{
}



/***********************************************************************************************************/
/* below returns (1-exp(-x))/x , which is needed for averages through slabs and over time for radiative quantities. here for convenience */
/***********************************************************************************************************/
double slab_averaging_function(double x)
{
    /* this fitting function is accurate to ~0.1% at all x, and extrapolates to the correct limits without the pathological behaviors of (1-exp(-x))/x for very small/large x */
    return (1.00000000000 + x*(0.21772719088733913 + x*(0.047076512011644776 + x*0.005068307557496351))) /
           (1.00000000000 + x*(0.71772719088733920 + x*(0.239273440788647680 + x*(0.046750496137263675 + x*0.005068307557496351))));
}



