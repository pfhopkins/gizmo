#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_eigen.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* Routines for models that require stellar evolution: luminosities, mass loss, SNe rates, etc.
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifdef GALSF


/* function to say whether a given particle should be treated as a "single star" or as -eligible to become- a single star (if it is gas), or
    whether it should be treated as a stellar population with some total mass, with IMF-integrated properties */
int is_particle_single_star_eligible(long i)
{
#if defined(SINGLE_STAR_SINK_DYNAMICS)
    if(P[i].Type == 0 || P[i].Type == 5) // only type=0 or type=5 (sinks) are eligible in our models here to be 'single star' candidates
    {
#if defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL) // here's the interesting regime, where we have some criterion for deciding which cells are eligible for 'single-star' status
        if(P[i].Type == 5) {return 1;} // all type-5 elements are assumed sinks
        if(P[i].Type == 0) {if(P[i].Mass*UNIT_MASS_IN_SOLAR > (SINGLE_STAR_AND_SSP_HYBRID_MODEL)) {return 0;} else {return 1;}} // use a simple mass threshold to decide which model we will use, specified by using this as a compile-time parameter
#else
        return 1; // no hybrid model, so all particles satisfying these criteria are automatically single-star eligible
#endif
    }
#endif
    return 0; // catch - default to non-single-star (SSP), unless satisfy some of the criteria above
}


/* return the light-to-mass ratio [in units of Lsun/Msun] of a star or stellar population with a given age; used throughout the code below */
double evaluate_light_to_mass_ratio(double stellar_age_in_gyr, int i)
{
    if(is_particle_single_star_eligible(i)) // SINGLE-STAR VERSION: calculate single-star luminosity (and convert to solar luminosity-to-mass ratio, which this output assumes)
    {
#ifdef SINGLE_STAR_SINK_DYNAMICS
        double m0=P[i].Mass; if(P[i].Type == 5) {m0=P[i].Sink_Mass;}
        return calculate_individual_stellar_luminosity(0, m0, i) / m0 * (UNIT_LUM_IN_SOLAR) / (UNIT_MASS_IN_SOLAR);
#endif
    }
    else // STELLAR-POPULATION VERSION: compute integrated mass-to-light ratio of an SSP
    {
        double lum=1; if(stellar_age_in_gyr < 0.01) {lum=1000;} // default to a dumb imf-averaged 'young/high-mass' vs 'old/low-mass' distinction
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION // fit to updated SB99 tracks: including rotation, new mass-loss tracks, etc.
        if(stellar_age_in_gyr < 0.0035) {lum=1136.59;} else {double log_age=log10(stellar_age_in_gyr/0.0035); lum=1500.*pow(10.,-1.8*log_age+0.3*log_age*log_age-0.025*log_age*log_age*log_age);}
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        double t1=0.0012, t2=0.0037, f1=800., f2=1100.*pow(Z_for_stellar_evol(i),-0.1), tx=log10(stellar_age_in_gyr/t2), t_g=log10(stellar_age_in_gyr/1.2)/0.05;
        if(stellar_age_in_gyr<=t1) {lum=f1;} else if(stellar_age_in_gyr<=t2) {lum=f1*pow(stellar_age_in_gyr/t1,log(f2/f1)/log(t2/t1));} else {lum=f2*pow(10.,-1.82*tx+0.42*tx*tx-0.07*tx*tx*tx)*(1.+1.2*exp(-0.5*t_g*t_g));}
#endif
#endif
#ifdef GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF
        lum *= calculate_relative_light_to_mass_ratio_from_imf(stellar_age_in_gyr,i,-1); // account for IMF variation model [if used; currently must be custom set as desired for modules]
#endif
        return lum;
    }
    return 0; // catch
}


/* subroutine to calculate luminosity of an individual star, according to accretion rate,
    mass, age, etc. Modify your assumptions about main-sequence evolution here. ONLY relevant for SINGLE-STAR inputs. */
double calculate_individual_stellar_luminosity(double mdot, double mass, long i)
{
#if !defined(SINGLE_STAR_SINK_DYNAMICS)
    return 0; /* not defined */
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 2) /* this is pre-calculated, simply return it */
    return P[i].StarLuminosity_Solar / UNIT_LUM_IN_SOLAR;
#endif
    /* if above flags not defined, estimate accretion + main-sequence luminosity as simply as possible */
    double lum=0, lum_sol=0, c_code = C_LIGHT_CODE, m_solar = mass * UNIT_MASS_IN_SOLAR;
    double rad_eff_protostar = 5.0e-7; /* if below the deuterium burning limit, just use the potential energy efficiency at the surface of a jupiter-density object */
    if(m_solar < 0.012) {rad_eff_protostar = 5.e-8 * pow(m_solar/0.00095,2./3.);}
    lum = rad_eff_protostar * mdot * c_code*c_code;
    if(m_solar >= 0.012) /* now for pre-main sequence and main sequence, need to also check the mass-luminosity relation */
    {
        if(m_solar < 0.43) {lum_sol = 0.185 * m_solar*m_solar;}
        else if(m_solar < 2.) {lum_sol = m_solar*m_solar*m_solar*m_solar;}
        else if(m_solar < 53.9) {lum_sol = 1.5 * m_solar*m_solar*m_solar * sqrt(m_solar);}
        else {lum_sol = 32000. * m_solar;}
    }
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 1) // now, account for pre-main sequence evolution and calculate accretion luminosity using protostellar radius
    if(i > 0) {if(P[i].Type == 5) {
        double eps_protostar=1.0, T4000_4 = pow(m_solar , 0.55), l_kh = 0.2263 * P[i].ProtoStellarRadius_inSolar*P[i].ProtoStellarRadius_inSolar * T4000_4; // protostellar temperature along Hayashi track and luminosity from KH contraction
        lum = DMAX(lum_sol,l_kh) / UNIT_LUM_IN_SOLAR + eps_protostar * (All.G * P[i].Mass / (P[i].ProtoStellarRadius_inSolar / UNIT_LENGTH_IN_SOLAR)) * mdot; // assume GM/r liberated per unit mass. Note we need radius in code units here since everything else in 'lum' is code-units as well. for pre-ms evolution, if Hayashi-temp luminosity exceeds MS luminosity, use it. otherwise use main sequence luminosity, and assume the star is moving along the Henyey track
        P[i].StarLuminosity_Solar = lum * UNIT_LUM_IN_SOLAR; //store total luminosity of the star in solar units
    }}
#endif
    return lum;

}


/* return the light-to-mass ratio, for the IMF of a given particle, relative to the Chabrier/Kroupa IMF.
    ONLY relevant for STELLAR POPULATION integrated inputs. "mode" denotes if we're interested in very massive stars (or other special behaviors): for now -1=bolometric, +1=very massive */
double calculate_relative_light_to_mass_ratio_from_imf(double stellar_age_in_gyr, int i, int mode)
{
#ifdef GALSF_SFR_IMF_VARIATION // fitting function from David Guszejnov's IMF calculations (ok for Mturnover in range 0.01-100) for how mass-to-light ratio varies with IMF shape/effective turnover mass
    double log_mimf = log10(P[i].IMF_Mturnover);
    return (0.051+0.042*(log_mimf+2)+0.031*(log_mimf+2)*(log_mimf+2)) / 0.31;
#endif
#ifdef GALSF_SFR_IMF_SAMPLING // account for IMF sampling model if not evolving individual stars
    double mu = 0.0115 * P[i].Mass * UNIT_MASS_IN_SOLAR; // 1 O-star per 100 Msun [more exactly calculated here as number of stars per solar mass with mass > 8 Msun, from our adopted Kroupa IMF from 0.01-100 Msun]
    double t=stellar_age_in_gyr*1000.,t1=3.7,t2=7.,t3=44.,a0=0.13,mu_min=1.e-3*mu;
    if(t>t3) {mu*=0;} else {if(t>t2) {mu*=(1.-a0)*(1.-(t-t2)/(t3-t2));} else {if(t>t1) {mu*=1.-a0*(t-t1)/(t2-t1);}}} // expectation value is declining with time, so 'effective multiplier' is larger
    if(mode > 0) {
        return P[i].IMF_NumMassiveStars / DMAX(mu,mu_min); // scales just with the number of massive stars
    } else {
        if(t>=t3) {return 1;} else {return 0.01 + P[i].IMF_NumMassiveStars / DMAX(mu,mu_min);} // scales with a minimum and a long-timescale baseline
    }
#endif
    return 1; // Chabrier or Kroupa IMF //
}


#if defined(GALSF_FB_FIRE_RT_HIIHEATING) || (defined(RT_CHEM_PHOTOION) && defined(GALSF))
/* routine to compute the -ionizing- luminosity coming from either individual stars or an SSP */
double particle_ionizing_luminosity_in_cgs(long i)
{
    if(P[i].Mass <= 0 || !isfinite(P[i].Mass)) {return 0;}
    if(is_particle_single_star_eligible(i)) /* SINGLE STAR VERSION: use effective temperature as a function of stellar mass and size to get ionizing photon production */
    {
#ifdef SINGLE_STAR_SINK_DYNAMICS
        double l_sol=sink_lum_bol(0,P[i].Mass,i)*(UNIT_LUM_IN_SOLAR), m_sol=P[i].Mass*UNIT_MASS_IN_SOLAR, r_sol=pow(m_sol,0.738); // L/Lsun, M/Msun, R/Rsun
        double T_eff=5780.*pow(l_sol/(r_sol*r_sol),0.25), x0=157800./T_eff, fion=0; // ZAMS effective temperature; x0=h*nu/kT for nu>13.6 eV; fion=fraction of blackbody emitted above x0
        if(x0 < 30.) {double q=18./(x0*x0) + 1./(8. + x0 + 20.*exp(-x0/10.)); fion = exp(-1./q);} // accurate to <10% for a Planck spectrum to x0>30, well into vanishing flux //
        return fion * l_sol * SOLAR_LUM_CGS; // return value in cgs, as desired for this routine [l_sol is in L_sun, by definition above] //
#endif
    }
    else /* STELLAR POPULATION VERSION: use updated SB99 tracks: including rotation, new mass-loss tracks, etc. */
    {
        if(P[i].Type != 5)
        {
            double lm_ssp=0, star_age=evaluate_stellar_age_Gyr(i), t0=0.0035, tmax=0.02;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
            tmax=0.15; lm_ssp=evaluate_light_to_mass_ratio(star_age,i); if(star_age<t0) {lm_ssp*=0.5;} else {lm_ssp*=0.5*pow(star_age/t0,-2.9);} /* slightly revised fit scales simply with Lbol [easier to modify]; see same references for stellar wind mass-loss rates; and extends to later ages (though most comes out at <100 Myr) */
#else
            if(star_age < t0) {lm_ssp=500.;} else {double log_age=log10(star_age/t0); lm_ssp=470.*pow(10.,-2.24*log_age-4.2*log_age*log_age) + 60.*pow(10.,-3.6*log_age);}
#endif
            if(star_age < 0.033) {lm_ssp *= 1.e-4 + calculate_relative_light_to_mass_ratio_from_imf(star_age,i,1);}
            if(star_age >= tmax) {return 0;} // skip since old stars don't contribute
            return lm_ssp * SOLAR_LUM_CGS * (P[i].Mass*UNIT_MASS_IN_SOLAR); // converts to cgs luminosity [lm_ssp is in Lsun/Msun, here]
        } // (P[i].Type != 5)
#ifdef SINK_HII_HEATING /* AGN template: light-to-mass ratio L(>13.6ev)/Mparticle in Lsun/Msun, above is dNion/dt = 5.5e54 s^-1 (Lbol/1e45 erg/s) */
        if(P[i].Type == 5) {return 0.18 * sink_lum_bol(P[i].Sink_Mdot,P[i].Mass,i) * UNIT_LUM_IN_CGS;}
#endif
    }
    return 0; // catch
}
#endif



/* this routine tells the feedback algorithms what to 'inject' when a stellar feedback event occurs.
    you must define the mass, velocity (which defines the momentum and energy), and metal content (yields)
    of the ejecta for the event[s] of interest. Mass [Msne] and velocity [SNe_v_ejecta] should
    be in code units. yields[k] should be defined for all metal species [k], and in dimensionless units
    (mass fraction of the ejecta in that species). */
void particle2in_addFB_fromstars(struct addFB_evaluate_data_in_ *in, int i, int fb_loop_iteration)
{
#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
    if(fb_loop_iteration == 0) {particle2in_addFB_SNe(in,i);}
    if(fb_loop_iteration == 1) {particle2in_addFB_winds(in,i);}
    if(fb_loop_iteration == 2) {particle2in_addFB_Rprocess(in,i);}
    if(fb_loop_iteration == 3) {particle2in_addFB_ageTracer(in,i);}
    return;
#endif
    if(P[i].SNe_ThisTimeStep<=0) {in->Msne=0; return;} // no event
    // 'dummy' example model assumes all SNe are identical with IMF-averaged properties from the AGORA model (Kim et al., 2016 ApJ, 833, 202)
    in->Msne = P[i].SNe_ThisTimeStep * (14.8/UNIT_MASS_IN_SOLAR); // assume every SNe carries 14.8 solar masses (IMF-average)
    in->SNe_v_ejecta = 2607. / UNIT_VEL_IN_KMS; // assume ejecta are ~2607 km/s [KE=1e51 erg, for M=14.8 Msun], which is IMF-averaged
#ifdef SINGLE_STAR_SINK_DYNAMICS // if single-star exploding or returning mass, use its actual mass & assumed energy to obtain the velocity
    if(is_particle_single_star_eligible(i))
    {
        in->Msne = DMIN(1.,P[i].SNe_ThisTimeStep) * P[i].Mass; // mass fraction of star being returned this timestep
        in->SNe_v_ejecta = sqrt(2.*(1.e51/UNIT_ENERGY_IN_CGS)/P[i].Mass); // for SNe [total return], simple v=sqrt(2E/m)should be fine without relativistic corrections
        if(P[i].SNe_ThisTimeStep<1) {double m_msun=P[i].Mass*UNIT_MASS_IN_SOLAR; in->SNe_v_ejecta = (616. * sqrt((1.+0.1125*m_msun)/(1.+0.0125*m_msun)) * pow(m_msun,0.131)) / UNIT_VEL_IN_KMS;} // scaling from size-mass relation+eddington factor, assuming line-driven winds //
    }
#endif
#ifdef METALS
    int k; for(k=0;k<NUM_METAL_SPECIES;k++) {in->yields[k]=0.178*All.SolarAbundances[k]/All.SolarAbundances[0];} // assume a universal solar-type yield with ~2.63 Msun of metals
    if(NUM_LIVE_SPECIES_FOR_COOLTABLES>=10) {in->yields[1] = 0.4;} // (catch for Helium, which the above scaling would give bad values for)
#endif
#endif
}


/* this routine calculates the event rates for different types of mechanical/thermal feedback
    algorithms. things like SNe rates, which determine when energy/momentum/mass are injected, should go here.
    you can easily modify this to accomodate any form of thermal or mechanical feedback/injection of various
    quantities from stars. */
double mechanical_fb_calculate_eventrates(int i, double dt)
{
#if defined(GALSF_FB_MECHANICAL) && defined(GALSF_FB_FIRE_STELLAREVOLUTION) // FIRE-specific stellar population version: separate calculation for SNe, stellar mass loss, R-process injection //
    double SNe_rate = mechanical_fb_calculate_eventrates_SNe(i,dt);
    mechanical_fb_calculate_eventrates_Winds(i,dt);
    mechanical_fb_calculate_eventrates_Rprocess(i,dt);
    mechanical_fb_calculate_eventrates_Agetracers(i,dt);
    return SNe_rate;
#endif

#if defined(SINGLE_STAR_SINK_DYNAMICS) && !defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) /* SINGLE-STAR version: simple implementation of single-star wind mass-loss and SNe rates */
    if(is_particle_single_star_eligible(i))
    {
        double m_sol,l_sol; m_sol=P[i].Mass*UNIT_MASS_IN_SOLAR; l_sol=sink_lum_bol(0,P[i].Mass,i)*UNIT_LUM_IN_SOLAR;
#ifdef SINGLE_STAR_FB_WINDS
        double gam=DMIN(0.5,3.2e-5*l_sol/m_sol), alpha=0.5+0.4/(1.+16./m_sol), q0=(1.-alpha)*gam/(1.-gam), k0=1./30.; // Eddington factor (~L/Ledd for winds), capped at 1/2 for sanity reasons, approximate scaling for alpha factor with stellar type (weak effect)
        P[i].SNe_ThisTimeStep = DMIN(0.5, (2.338 * alpha * pow(l_sol,7./8.) * pow(m_sol,0.1845) * (1./q0) * pow(q0*k0,1./alpha) / m_sol) * (dt*UNIT_TIME_IN_GYR)); // Castor, Abbot, & Klein scaling
#endif
#ifdef SINGLE_STAR_FB_SNE
        double t_lifetime_Gyr = 10.*(m_sol/l_sol) + 0.003; /* crude estimate of main-sequence lifetime, capped at 3 Myr*/
        if(evaluate_stellar_age_Gyr(i) >= t_lifetime_Gyr) {P[i].SNe_ThisTimeStep=1;}
#endif
        return 1;
    }
#endif

#ifdef GALSF_FB_THERMAL /* STELLAR-POPULATION version: pure thermal feedback: assumes AGORA model (Kim et al., 2016 ApJ, 833, 202) where everything occurs at 5Myr exactly */
    if(P[i].SNe_ThisTimeStep != 0) {P[i].SNe_ThisTimeStep=-1; return 0;} // already had an event, so this particle is "done"
    if(evaluate_stellar_age_Gyr(i) < 0.005) {return 0;} // enforce age limit of 5 Myr
    P[i].SNe_ThisTimeStep = P[i].Mass*UNIT_MASS_IN_SOLAR / 91.; // 1 event per 91 solar masses
    return 1;
#endif

#ifdef GALSF_FB_MECHANICAL /* STELLAR-POPULATION version: mechanical feedback: 'dummy' example model below assumes a constant SNe rate for t < 30 Myr, then nothing. experiment! */
    double star_age = evaluate_stellar_age_Gyr(i);
    if(star_age < 0.03)
    {
        double RSNe = 3.e-4; // assume a constant rate ~ 3e-4 SNe/Myr/solar mass for t = 0-30 Myr //
        double p = RSNe * (P[i].Mass*UNIT_MASS_IN_SOLAR) * (dt*UNIT_TIME_IN_MYR); // unit conversion factor
        double n_sn_0=(float)floor(p); p-=n_sn_0; if(get_random_number(P[i].ID+6) < p) {n_sn_0++;} // determine if SNe occurs
        P[i].SNe_ThisTimeStep = n_sn_0; // assign to particle
        return RSNe;
    }
#endif

    return 0;
}



#if defined(GALSF_FB_MECHANICAL) && defined(GALSF_FB_FIRE_STELLAREVOLUTION)
/* functions below contain pre-calculation of event rates and energetics, masses, etc, for FIRE mechanical feedback modules */

double mechanical_fb_calculate_eventrates_SNe(int i, double dt)
{
#if defined(SINGLE_STAR_SINK_DYNAMICS) && (!defined(SINGLE_STAR_FB_SNE) || defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)) /* no single-star module to use here, for these flags its in the spawn routine */
    if(is_particle_single_star_eligible(i)) {return 0;}
#endif
    
    if(All.SNe_Energy_Renormalization <= 0) {return 0;}
    double star_age = evaluate_stellar_age_Gyr(i), RSNe=0, agemin=0.003401, agebrk=0.01037, agemax=0.03753; // some ages in Gyr used below
    /* here we are determining an expected SNe rate, so SNe occur stochastically but with an age dependence in the population */
    // calculate: NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from the energy curve [do this so we are gauranteed to get the correct SNe energy] //
#if (GALSF_FB_FIRE_STELLAREVOLUTION == 1) || (GALSF_FB_FIRE_STELLAREVOLUTION == 2)
    if(star_age>agemin) {if(star_age<=agebrk) {RSNe=5.408e-4;} else {if(star_age<=agemax) {RSNe=2.516e-4;}}} // core-collapse rate [super-simple 2-piece constant] //
    if(star_age>agemax) {RSNe=5.3e-8 + 1.6e-5*exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));} // Ia (prompt Gaussian+delay, Manucci+06)
#elif (GALSF_FB_FIRE_STELLAREVOLUTION > 2) 
    agemin=0.0037; agebrk=0.7e-2; agemax=0.044; double f1=3.9e-4, f2=5.1e-4, f3=1.8e-4;
    if(star_age<agemin) {RSNe=0;} else if(star_age<=agebrk) {RSNe=f1*pow(star_age/agemin,log(f2/f1)/log(agebrk/agemin));}
        else if(star_age<=agemax) {RSNe=f2*pow(star_age/agebrk,log(f3/f2)/log(agemax/agebrk));} else {RSNe=0;} // core-collapse; updated with same stellar evolution models for wind mass loss [see there for references]. simple 2-part power-law provides extremely-accurate fit. models predict a totally negligible metallicity-dependence.
    double t_Ia_min=agemax, norm_Ia=1.6e-3; // t_Ia_min = delay time to first Ia, in Gyr; norm_Ia = Hubble-time integrated number of Ia's per solar mass
    if(star_age>t_Ia_min) {RSNe += norm_Ia * 7.94e-5 * pow(star_age,-1.1) / fabs(pow(t_Ia_min/0.1,-0.1) - 0.61);} // Ia DTD following Maoz & Graur 2017, ApJ, 848, 25
#endif

    double renorm = calculate_relative_light_to_mass_ratio_from_imf(star_age,i,1); // account for higher # of O-stars with a different IMF
    if(star_age<agemax) {RSNe *= renorm;}
#ifdef GALSF_SFR_IMF_SAMPLING
    if(star_age>agemax && P[i].IMF_NumMassiveStars>0) {RSNe += 2.516e-4*renorm;} // account for un-exploded O-stars
#endif
    double p = RSNe * (P[i].Mass*UNIT_MASS_IN_SOLAR) * (dt*UNIT_TIME_IN_MYR); // unit conversion factor
    double n_sn_0=(float)floor(p); p-=n_sn_0; if(get_random_number(P[i].ID+6) < p) {n_sn_0++;} // determine if SNe occurs
#ifdef GALSF_SFR_IMF_SAMPLING
    if(star_age<agemax && P[i].IMF_NumMassiveStars<n_sn_0) {n_sn_0=P[i].IMF_NumMassiveStars;} // limit to number of O-stars for SNe //
#endif
    P[i].SNe_ThisTimeStep = n_sn_0; // assign to particle
    return RSNe;
}


void mechanical_fb_calculate_eventrates_Rprocess(int i, double dt)
{
#ifdef GALSF_FB_FIRE_RPROCESS
    /* we'll use the maximum rate here, then in the -yields- setting, 'cut' each down to its sub-population */
    double star_age = evaluate_stellar_age_Gyr(i);
    if(star_age>=0.003) // rate is zero at <3e6 yr
    {
        double p = 3.0e-5 / (1000.*star_age) * (P[i].Mass*UNIT_MASS_IN_SOLAR) * (dt*UNIT_TIME_IN_MYR); // Nsne/Myr for a 1 Msun population
        double n_sn_0=(float)floor(p); p-=n_sn_0; if(get_random_number(P[i].ID + 7) < p) {n_sn_0++;} // if > 1, this cuts that part off so we get appropriate n > 1 solution
        P[i].RProcessEvent_ThisTimeStep = n_sn_0; // assign event
    }
#endif
}


#ifdef GALSF_FB_FIRE_AGE_TRACERS
/* Finds the age tracer bin (passive scalar) corresponding to the given star's age (in Myr) */
int get_age_tracer_bin(double age)
{
    int index = -1; int too_old_flag = -9;
#ifdef GALSF_FB_FIRE_AGE_TRACERS_CUSTOM
    /* Bins are custom with arbitrary sizes - perform a search */
    if(age < All.AgeTracerTimeBins[0]) {index=0;} // initial bin
        else if(age >= All.AgeTracerTimeBins[NUM_AGE_TRACERS]) {return too_old_flag;} /* do nothing here and continue */
        else {index = binarySearch(All.AgeTracerTimeBins,age,0,NUM_AGE_TRACERS+1,NUM_AGE_TRACERS+1); /* find the bin */
            if(age < All.AgeTracerTimeBins[index]) {PRINT_WARNING("Age tracer binary search not working %d  %f  %f  %f",index, age,All.AgeTracerTimeBins[index],All.AgeTracerTimeBins[index+1]); endrun(8888);}}
#else
    /* Bins are log-spaced we do not need to perform a search */
    double binstart=log10(All.AgeTracerBinStart), binend=log10(All.AgeTracerBinEnd), log_bin_dt=(binend - binstart)/((double)NUM_AGE_TRACERS);
    if(age <= All.AgeTracerBinStart) {index=0;} /* initial bin */
        else if(age >= All.AgeTracerBinEnd) {return too_old_flag;} /* do nothing here and continue [likely only in test problems] */
        else {index=(int) floor((log10(age)-binstart)/log_bin_dt);} /* find bin */
#endif
    if(index<0) {PRINT_WARNING("Age tracer binary search not working index=%d age=%f",index,age); endrun(8888);} // exit if invalid result
    return index; // return valid bin
}


/* Returns the start time, in Myr, of the age bin with index "k" */
double get_age_tracer_bin_start_time(int k)
{
    if(k<0) {return 0;} // always return 0 for arbitrarily low indices
    if(k>NUM_AGE_TRACERS) {return get_age_tracer_bin_start_time(NUM_AGE_TRACERS);} // should never happen-set to same value as last value here
#ifdef GALSF_FB_FIRE_AGE_TRACERS_CUSTOM
    return All.AgeTracerTimeBins[k]; // use pre-tabulated bin times, in Myr
#else
    return All.AgeTracerBinStart * pow(All.AgeTracerBinEnd/All.AgeTracerBinStart , ((double)k)/((double)NUM_AGE_TRACERS)); // uniform log-spaced bins
#endif
}
#endif


/* function to determine if deposition into age-tracer bins occurs this time step */
void mechanical_fb_calculate_eventrates_Agetracers(int i, double dt)
{
#ifdef GALSF_FB_FIRE_AGE_TRACERS
    P[i].AgeDeposition_ThisTimeStep=0; // reset
    if(P[i].Type != 4) {return;} // only new stars are eligible sources
    double rate_normalization = All.AgeTracerRateNormalization; // determines the fraction of time spent depositing tracers, or rate-per-particle
    if(rate_normalization == 0 || rate_normalization <= -1) {P[i].AgeDeposition_ThisTimeStep=1; return;} // rate_normalization = 0 means deposit every time step
    double stellar_age_myr = evaluate_stellar_age_Gyr(i)*1000.; int k = get_age_tracer_bin(stellar_age_myr); // get age in Myr and bin for age
    double bin_dt_myr = get_age_tracer_bin_start_time(k+1)-get_age_tracer_bin_start_time(k); // get bin duration in Myr
    if(rate_normalization <= 0) // there is a constant probability of event -per timestep-, with 'amount returned' depending on timestep duration
    {
        if(dt*UNIT_TIME_IN_MYR/bin_dt_myr > fabs(rate_normalization)) {P[i].AgeDeposition_ThisTimeStep=1;} else  // if dt is large compared to bin spacing, might 'miss' an event, so we want to always make an event and return
            {if(get_random_number(P[i].ID + 3) < fabs(rate_normalization)) {P[i].AgeDeposition_ThisTimeStep=1./fabs(rate_normalization);}} // rate limit according to the return fraction this represents fraction of time star should be depositing; if selected, deposit age tracer and use this scaling to increase the normalization of the tracer field
    } else { // there is a constant probability per -time- (target 'number of events per bin per particle'), with 'amount returned' constant
        double p=fabs(rate_normalization)*(dt*UNIT_TIME_IN_MYR/bin_dt_myr), n_events=(float)floor(p); p-=n_events; if(get_random_number(P[i].ID + 3) < p) {n_events++;} // if > 1, this cuts that part off so we get appropriate n > 1 solution
        P[i].AgeDeposition_ThisTimeStep = n_events / fabs(rate_normalization); // assign event
    }
#endif
    return;
}


void mechanical_fb_calculate_eventrates_Winds(int i, double dt)
{
    double D_RETURN_FRAC=1.e-15, p=0;
    
    if(is_particle_single_star_eligible(i)) /* SINGLE-STAR VERSION: single-star wind mass-loss rates */
    {
#if defined(SINGLE_STAR_FB_WINDS)
        double fire_wind_rel_mass_res = 1e-4; /* relative mass resolution of winds, essentially the wind will get spawned in packets of fire_wind_rel_mass_res*(gas_mass_resolution) mass */
        D_RETURN_FRAC = (fire_wind_rel_mass_res * P[i].Sink_Formation_Mass) / P[i].Mass;
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION /* for 'fancy' multi-stage modules, have a separate subroutine to compute this */
        if(P[i].wind_mode != 2) {return;} /* only some eligible particles have winds in this module */
        p = single_star_wind_mdot(i,0) * dt / P[i].Mass; /* actual mdot from its own subroutine, given in code units */
#else /* otherwise use standard scaling from e.g. Castor, Abbot, & Klein */
        double m_sol=P[i].Mass*UNIT_MASS_IN_SOLAR, l_sol=sink_lum_bol(0,P[i].Mass,i)*UNIT_LUM_IN_SOLAR; /* luminosity in solar */
        double gam=DMIN(0.5,3.2e-5*l_sol/m_sol), alpha=0.5+0.4/(1.+16./m_sol), q0=(1.-alpha)*gam/(1.-gam), k0=1./30.; // Eddington factor (~L/Ledd for winds), capped at 1/2 for sanity reasons, approximate scaling for alpha factor with stellar type (weak effect)
        p = (2.338 * alpha * pow(L_sol,7./8.) * pow(M_sol,0.1845) * (1./q0) * pow(q0*k0,1./alpha) / m_sol) * (dt*UNIT_TIME_IN_GYR); // mdot in M_sun/Gyr, times dt
#endif
#endif
        p=1.-exp(-p);
    }
    else /* STELLAR POPULATION VERSION: now do stellar-population-averaged inputs */
    {
        D_RETURN_FRAC = 0.01; // fraction of particle mass to return on a recycling step //
        if(All.StellarMassLoss_Rate_Renormalization <= 0) {return;}
        double star_age=evaluate_stellar_age_Gyr(i), ZZ=Z_for_stellar_evol(i);
#if (GALSF_FB_FIRE_STELLAREVOLUTION == 1)
        if(star_age<=0.001){p=4.76317;} else {if(star_age<=0.0035){p=4.76317*pow(10.,1.838*(0.79+log10(ZZ))*(log10(star_age)-(-3.00)));} else {
            if(star_age<=0.1){p=29.4*pow(star_age/0.0035,-3.25)+0.0041987;} else {p=0.41987*pow(star_age,-1.1)/(12.9-log(star_age));}}} // normalized  to give expected return fraction from stellar winds alone (~17%)
#elif (GALSF_FB_FIRE_STELLAREVOLUTION == 2)
        double t_agb = 0.1;
#if defined(GALSF_ISMDUSTCHEM_MODEL)
        t_agb = 0.03753; // dust routines slightly adjust the wind rates so that prompt AGB winds start at the same time SNe II end instead of at 100 Myr. Overall has neglible effects to energetics but a large amount of dust can be produced by these massive AGBs early on in a stellar population's life
#endif
        if(star_age<=0.001){p=4.76317*ZZ;} else {if(star_age<=0.0035){p=4.76317*ZZ*pow(10.,1.838*(0.79+log10(ZZ))*(log10(star_age)-(-3.00)));} else {
            if(star_age<=t_agb){p=29.4*pow(star_age/0.0035,-3.25)+0.0041987;} else {p=0.41987*pow(star_age,-1.1)/(12.9-log(star_age));}}} // normalized  to give expected return fraction from stellar winds alone (~17%)
#elif (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        /* updated fit. separates the more robust line-driven winds [massive-star-dominated] component, and -very- uncertain AGB. extremely good fits to updated STARBURST99 result for a 3-part Kroupa IMF (0.3,1.3,2.3 slope, 0.01-0.08-0.5-100 Msun, 8-120 SNe/BH cutoff, wind model evolution, Geneva v40 [rotating, Geneva 2013 updated tracks, at all metallicities available, ~0.1-1 solar], sampling times 1e4-2e10 yr at high resolution */
        double f1=3.*pow(ZZ,0.87), f2=20.*pow(ZZ,0.45), f3=0.6*ZZ, t1=0.0017, t2=0.004, t3=0.02, t=star_age; /* fit parameters for 'massive star' mass-loss */
        if(t<=t1) {p=f1;} else if(t<=t2) {p=f1*pow(t/t1,log(f2/f1)/log(t2/t1));} else if(t<=t3) {p=f2*pow(t/t2,log(f3/f2)/log(t3/t2));} else {p=f3*pow(t/t3,-3.1);} /* piecewise continuous function linking constant early and rapid late decay */
        // double f_agb=0.01, t_agb=1.; p += f_agb/((1. + pow(t/t_agb,1.1)) * (1. + 0.01/(t/t_agb))); /* add AGB component. note that essentially no models [any of the SB99 geneva or padova tracks, or NuGrid, or recent other MESA models] predict a significant dependence on metallicity (that shifts slightly when the 'bump' occurs, but not the overall loss rate), so this term is effectively metallicity-independent */
        double f_agb=0.1, t_agb=0.8, x_agb=t_agb/DMAX(t,1.e-4); x_agb*=x_agb; p += f_agb * pow(x_agb,0.8) * (exp(-DMIN(50.,x_agb*x_agb*x_agb)) + 1./(100. + x_agb)); /* May 28, 2022: re-fit AGB component based on inputs from Caleb Choban: previous model for FIRE-3 doesn't make sense for stars 1.5-4 Msol, can't possibly give sub-Chandrasekhar WDs or the correct initial-final mass relation. Here we re-fit the AGB mass loss for the Geneva v00 (rotating models show too little), 2013 tracks, 1x solar, times at same resolution as above, using the 'Empirical' wind prescription in STARBURST99. Have validated that nothing else, including wind specific energies, changes - only the mass-loss rates */
#endif
        if(star_age < 0.033) {p *= 0.01 + calculate_relative_light_to_mass_ratio_from_imf(star_age,i,1);} // late-time independent of massive stars
        p *= All.StellarMassLoss_Rate_Renormalization * (dt*UNIT_TIME_IN_GYR); // fraction of particle mass expected to return in the timestep //
        p = 1.0 - exp(-p); // need to account for p>1 cases //
    }
    
    double n_wind_0=(double)floor(p/D_RETURN_FRAC); p-=n_wind_0*D_RETURN_FRAC; // if p >> return frac, should have > 1 event, so we inject the correct wind mass
    P[i].MassReturn_ThisTimeStep += n_wind_0*D_RETURN_FRAC; // add this in, then determine if there is a 'remainder' to be added as well
    if(get_random_number(P[i].ID + 5) < p/D_RETURN_FRAC) {P[i].MassReturn_ThisTimeStep += D_RETURN_FRAC;} // add the 'remainder' stochastically
}






void particle2in_addFB_Rprocess(struct addFB_evaluate_data_in_ *in, int i)
{
#ifdef GALSF_FB_FIRE_RPROCESS
    if(P[i].RProcessEvent_ThisTimeStep<=0) {in->Msne=0; return;} // no event
    int k; double star_age=evaluate_stellar_age_Gyr(i), p=get_random_number(P[i].ID + 8), pcrit, tcrit;
    for(k=0;k<NUM_RPROCESS_SPECIES;k++)
    {
        if(k==0) {tcrit=0.03;  pcrit=0.3333333333;}  // model 0: tmin > 3e7 yr, rate = 1e-5
        if(k==1) {tcrit=0.003; pcrit=0.3333333333;}  // model 1: tmin > 3e6 yr, rate = 1e-5
        if(k==2) {tcrit=0.03;  pcrit=0.03333333333;} // model 2: tmin > 3e7 yr, rate = 1e-6
        if(k==3) {tcrit=0.003; pcrit=0.03333333333;} // model 3: tmin > 3e6 yr, rate = 1e-6
        if((star_age>=tcrit)&&(p<=pcrit)) {in->yields[1+NUM_LIVE_SPECIES_FOR_COOLTABLES+k]=1;} // units irrelevant//
    }
    in->Msne = 0.01 * (double)P[i].RProcessEvent_ThisTimeStep / ((double)(UNIT_MASS_IN_SOLAR)); // mass ejected ~0.01*M_sun; only here for bookkeeping //
#endif
}




/* Routine injects passive scalar tracer field for each star particle according to the age of that particle for post processing stellar
  abundances. Finds the appropriate time bin(s) to inject into given the star's age and the timestep. Normalization of these fields is
  somewhat arbitrary, but normalizing such that each star deposits 1 unit in each time bin (regardless of size) per solar mass. Thus,
  scaling is simple (just multiply field by mass of metal species produced per solar mass of star formation to get an abundance) */
void particle2in_addFB_ageTracer(struct addFB_evaluate_data_in_ *in, int i)
{
#ifdef GALSF_FB_FIRE_AGE_TRACERS
    if(P[i].Type != 4) {return;} // only new stars are eligible for events
    if(P[i].AgeDeposition_ThisTimeStep<=0) {return;} // no event

    double M_norm = (P[i].Mass*UNIT_MASS_IN_SOLAR) * P[i].AgeDeposition_ThisTimeStep; /* AJE: may need to switch to normalizing over logbin spacing if bins are large too avoid small number issues  - this requires undoing the normalization in post */
    double age_myr = evaluate_stellar_age_Gyr(i) * 1000.; int k = get_age_tracer_bin(age_myr); // get age in myr and corresponding tracer bin
    if(k==-9) {printf("Stellar age greater than maximum allows in AGE_TRACERS bins\n"); return;} // error trap for age > max bin
    double dt=GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i)*UNIT_TIME_IN_MYR, dt_half=0.5*dt, age_initial=age_myr-dt_half, age_final=age_myr+dt_half; // get particle timestep in Myr, and age at beginning/end of timestep centered on us
    if(dt <= 0) {return;} // no event possible - must have arrived here in error
    int k_age_start = 1+NUM_LIVE_SPECIES_FOR_COOLTABLES+NUM_RPROCESS_SPECIES; // first index of tracers

    // now deposit tracer fields, with check if multiple bins are crossed
    double t_start=get_age_tracer_bin_start_time(k), t_end=get_age_tracer_bin_start_time(k+1), bin_dt=t_end-t_start; // bin edges in Myr
    double amount_returned_normalizer = dt / bin_dt; // mode when AgeTracerRateNormalization <= 0 :: there is a constant probability of event -per timestep-, with 'amount returned' depending on timestep duration
    if(All.AgeTracerRateNormalization > 0) {amount_returned_normalizer = 1.;} // mode when AgeTracerRateNormalization < 0 :: there is a constant probability per -time- (target 'number of events per bin per particle'), with 'amount returned' constant
    M_norm *= amount_returned_normalizer; // re-normalize the total amount returned, appropriate for our timestepping scheme
    if(((k>0) && (age_initial<t_start)) || ((k<NUM_AGE_TRACERS) && (age_final>t_end))) // timestep crosses multiple -valid- bins
    {
        double age=age_initial, age_step=0; if(age_initial < t_start) {k--;} // step back a bin, set variables
        while((age<age_final) && (k<=NUM_AGE_TRACERS)) { // loop over active bins to set yields in each bin
            bin_dt = get_age_tracer_bin_start_time(k+1) - get_age_tracer_bin_start_time(k); age_step = DMIN(t_end, age_final); // get the step to the end of bin
            in->yields[k+k_age_start] += ((age_step - age) / dt) * M_norm; // do fractional / full injection for each bin we cross
            age = age_step; k++;}
    } else {in->yields[k+k_age_start] += M_norm;} // normalization is somewhat arbitrary, but choosing "1" unit per bin per solar mass of star for convenience
    in->Msne = 1.e-9 / UNIT_MASS_IN_SOLAR; // small number just to be nonzero
#endif
    return;
}



void particle2in_addFB_SNe(struct addFB_evaluate_data_in_ *in, int i)
{
    int k; if(P[i].SNe_ThisTimeStep<=0) {in->Msne=0; return;} // no event
    double t_gyr = evaluate_stellar_age_Gyr(i); int SNeIaFlag=0; if(t_gyr > 0.03753) {SNeIaFlag=1;}; /* assume SNe before critical time are core-collapse, later are Ia */
    double Msne=10.5; if(SNeIaFlag) {Msne=1.4;} // average ejecta mass for single event (normalized to give total mass loss correctly)
#ifdef METALS
    double yields[NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION]={0}; get_SNe_yields(yields,i,t_gyr,SNeIaFlag,&Msne); // call subroutine to collect the yields
    for(k=0;k<NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;k++) {in->yields[k]=yields[k];} // assign them back to the vector
#endif
    in->Msne = P[i].SNe_ThisTimeStep * (Msne/UNIT_MASS_IN_SOLAR); // total mass in code units
#ifdef SINGLE_STAR_SINK_DYNAMICS
    if(is_particle_single_star_eligible(i)) {in->Msne = P[i].Mass;} // conserve mass and destroy star completely
#endif
    double SNeEgy = All.SNe_Energy_Renormalization*P[i].SNe_ThisTimeStep * 1.0e51/UNIT_ENERGY_IN_CGS; // assume each SNe has 1e51 erg
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    if(SNeIaFlag==0) {double z_eff = P[i].Metallicity[10]/All.SolarAbundances[10]; if(z_eff < 1) {SNeEgy *= pow(z_eff + 1.e-5 , -0.12);}} // updated to use same metallicity used for stellar evolution, rather than total metallicity, if this derives from pre-explosion winds, etc, for consistency
#if (FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT > 1)
    if(i>0) {double z0 = P[i].Metallicity[0]/All.SolarAbundances[0];
#if (FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT > 2)
        SNeEgy *= pow(z0/0.1 + 1.e-3 , -0.2);
#else
        SNeEgy *= pow(z0/0.1 + 1.e-3 , -0.1);
#endif
    }
#endif
#endif
    in->SNe_v_ejecta = sqrt(2.0*SNeEgy/(in->Msne)); // v_ej in code units
#ifdef SNE_NONSINK_SPAWN
    if(P[i].Type == 4) {
        double frac_spawned = 0.5;
        P[i].unspawned_wind_mass += frac_spawned * (in->Msne);
        in->Msne *= 1.-frac_spawned;
        double n_unspawned = P[i].unspawned_wind_mass / ((SINK_WIND_SPAWN)*target_mass_for_wind_spawning(i)); // number of spawned gas cells that can be made from the mass in the reservoir
        if(n_unspawned> Max_Unspawned_MassUnits_fromSink) {Max_Unspawned_MassUnits_fromSink = n_unspawned;} // track the maximum integer number of elements this sink could spawn
    } // designate some fraction of the SNe ejecta for spawning new high-res cells, instead of coupling to pre-existing neighbor cells //
#endif
}


#ifdef METALS
void get_SNe_yields(double *yields, int i, double t_gyr, int SNeIaFlag, double *Msne)
{
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2) || (defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && defined(SINGLE_STAR_FB_SNE))
#if !defined(SINGLE_STAR_FB_SNE) || defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL)
    if(!is_particle_single_star_eligible(i)) {if(t_gyr > 0.044) {SNeIaFlag=1;} else {SNeIaFlag=0;}} /* match to rates tabulation above to determine if Ia or CC */
#endif
    if(SNeIaFlag==1) {*Msne=1.4;} else {*Msne=8.72;} /* default to a mean mass for Ia vs CC SNe; for updated table of SNe rates and energetics, this is the updated mean mass per explosion to give the correct total SNe mass */
    int k; for(k=0;k<NUM_METAL_SPECIES;k++) {yields[k]=P[i].Metallicity[k];} /* initialize to surface abundances */
    if(NUM_METAL_SPECIES>=10) { // All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe
        if(SNeIaFlag) { /* SNIa :: from Iwamoto et al. 1999; 'W7' models: total ejecta mass = Msne = 1.4. yields below are -fractional- */
            yields[0]=1; /* total metal mass (species below, + residuals primarily in Ar, Cr, Mn, Ni) */ yields[1]=0; /*He*/
            /* adopted yield: mean of W7 and WDD2 in Mori+2018. other models included below for reference in comments */
            yields[2]=1.76e-2; yields[3]=2.10e-06; yields[4]=7.36e-2; yields[5]=2.02e-3; yields[6]=6.21e-3; yields[7]=1.46e-1; yields[8]=7.62e-2; yields[9]=1.29e-2; yields[10]=5.58e-1; // arguably better obs calibration vs LN/NL papers
            /* Iwamoto et al. 1999, 'W7' model */ //yields[2]=3.50e-2; yields[3]=8.57e-07; yields[4]=1.02e-1; yields[5]=3.21e-3; yields[6]=6.14e-3; yields[7]=1.11e-1; yields[8]=6.21e-2; yields[9]=8.57e-3; yields[10]=5.31e-1; // old, modestly disfavored albeit for species like Mn not here
            /* updated W7 in Nomoto+Leung 18 review */ //yields[2]=3.71e-2; yields[3]=7.79e-10; yields[4]=1.32e-1; yields[5]=3.11e-3; yields[6]=3.07e-3; yields[7]=1.19e-1; yields[8]=5.76e-2; yields[9]=8.21e-3; yields[10]=5.73e-1; // not significantly different from updated W7 below, bit more of an outlier and review tables seem a bit unreliable (typos, etc)
            /* mean of new yields for W7 + WDD2 in Leung+Nomoto+18 */ //yields[2]=1.54e-2; yields[3]=1.24e-08; yields[4]=8.93e-2; yields[5]=2.41e-3; yields[6]=3.86e-3; yields[7]=1.34e-1; yields[8]=7.39e-2; yields[9]=1.19e-2; yields[10]=5.54e-1; // not significantly different from updated W7 below, bit more of an outlier and review tables seem a bit unreliable (typos, etc)
            // W7   [Mori+18] [3.42428571e-02, 4.16428571e-06, 9.68571429e-02, 2.67928571e-03, 7.32857143e-03, 1.25296429e-01, 5.65937143e-02, 8.09285714e-03, 5.68700000e-01] -- absolute yield in solar // WDD2 [Mori+18] [9.70714286e-04, 2.36285714e-08, 5.04357143e-02, 1.35621429e-03, 5.10112857e-03, 1.65785714e-01, 9.57078571e-02, 1.76928571e-02, 5.47890000e-01] -- absolute yield in solar
            /* updated W7 in Leung+Nomoto+18 */ //yields[2]=1.31e-2; yields[3]=7.59e-10; yields[4]=9.29e-2; yields[5]=1.79e-3; yields[6]=2.82e-3; yields[7]=1.06e-1; yields[8]=5.30e-2; yields[9]=6.27e-3; yields[10]=5.77e-1; // seems bit low in Ca/Fe, less plausible if those dominated by Ia's
            /* Seitenzahl et al. 2013, model N100 [favored] */ //yields[2]=2.17e-3; yields[3]=2.29e-06; yields[4]=7.21e-2; yields[5]=2.55e-3; yields[6]=1.10e-2; yields[7]=2.05e-1; yields[8]=8.22e-2; yields[9]=1.05e-2; yields[10]=5.29e-1; // very high Si, seems bit less plausible vs other models here
            /* new benchmark model in Leung+Nomoto+18 [closer to WDD2 in lighter elements, to W7 in heavier elements] */ //yields[2]=1.21e-3; yields[3]=1.40e-10; yields[4]=4.06e-2; yields[5]=1.29e-4; yields[6]=7.86e-4; yields[7]=1.68e-1; yields[8]=8.79e-2; yields[9]=1.28e-2; yields[10]=6.14e-1; // arguably better theory motivation vs Mori+ combination
        } else { /* Core collapse :: temporary new time-dependent fits */
            double t=t_gyr, tmin=0.0037, tbrk=0.0065, tmax=0.044, Mmax=35., Mbrk=10., Mmin=6.; // numbers for interpolation of ejecta masses [must be careful here that this integrates to the correct -total- ejecta mass]
            /* note these break times: tmin=3.7 Myr corresponds to the first explosions (Eddington-limited lifetime of the most massive stars),
             tbrk=6.5 Myr to the end of this early phase, stars with ZAMS mass ~30+ Msun here. curve flattens both from IMF but also b/c mass-loss less efficient. tmax=44 Myr to the last explosion determined by lifetime of 8 Msun stars */
            if(t<=tbrk) {*Msne=Mmax*pow(t/tmin, log(Mbrk/Mmax)/log(tbrk/tmin));} else {*Msne=Mbrk*pow(t/tbrk, log(Mmin/Mbrk)/log(tmax/tbrk));} // power-law interpolation of ejecta mass from initial to final value over duration of CC phase
            int i_tvec = 5; /* number of entries */ double tvec[5]={3.7, 8., 18., 30., 44.}; /* time in Myr */
            double fvec[10][5] = {
                {4.61e-01, 3.30e-01, 3.58e-01, 3.65e-01, 3.59e-01}, // He [IMF-mean y=3.67e-01]  [note have to remove normal solar correction and take care with winds]
                {2.37e-01, 8.57e-03, 1.69e-02, 9.33e-03, 4.47e-03}, // C  [IMF-mean y=3.08e-02]  [note care needed in fitting out winds: wind=6.5e-3, ejecta_only=1.0e-3]
                {1.07e-02, 3.48e-03, 3.44e-03, 3.72e-03, 3.50e-03}, // N  [IMF-mean y=4.47e-03]  [some care needed with winds, but not as essential]
                {9.53e-02, 1.02e-01, 9.85e-02, 1.73e-02, 8.20e-03}, // O  [IMF-mean y=7.26e-02]  [reasonable - generally IMF-integrated alpha-element total mass-yields lower vs fire-2 by factor ~0.7 or so]
                {2.60e-02, 2.20e-02, 1.93e-02, 2.70e-03, 2.75e-03}, // Ne [IMF-mean y=1.58e-02]  [roughly a hybrid of fit direct to ejecta and fit to all mass as above, truncating at highest masses]
                {2.89e-02, 1.25e-02, 5.77e-03, 1.03e-03, 1.03e-03}, // Mg [IMF-mean y=9.48e-03]  [fit directly on ejecta and ignore mass-fraction rescaling since that's not reliable at early times: this gives a reasonable number. important to note that early SNe dominate Mg here, quite strongly]
                {4.12e-04, 7.69e-03, 8.73e-03, 2.23e-03, 1.18e-03}, // Si [IMF-mean y=4.53e-03]  [lots comes from 1a's, so low here isn't an issue]
                {3.63e-04, 5.61e-03, 5.49e-03, 1.26e-03, 5.75e-04}, // S  [IMF-mean y=3.01e-03]  [more from Ia's]
                {4.28e-05, 3.21e-04, 6.00e-04, 1.84e-04, 9.64e-05}, // Ca [IMF-mean y=2.77e-04]  [Ia]
                {5.46e-04, 2.18e-03, 1.08e-02, 4.57e-03, 1.83e-03}  // Fe [IMF-mean y=4.11e-03]  [Ia]
            }; // compare nomoto '06: y = [He: 3.69e-1, C: 1.27e-2, N: 4.56e-3, O: 1.11e-1, Ne: 3.81e-2, Mg: 9.40e-3, Si: 8.89e-3, S: 3.78e-3, Ca: 4.36e-4, Fe: 7.06e-3]
            /* ok now use the fit parameters above for the piecewise power-law components to define the yields at each time */
            double t_myr=t_gyr*1000.; int i_t=-1; for(k=0;k<i_tvec;k++) {if(t_myr>tvec[k]) {i_t=k;}}
            for(k=0;k<10;k++) {int i_y = k + 1; if(i_t<0) {yields[i_y]=fvec[k][0];} else if(i_t>=i_tvec-1) {yields[i_y]=fvec[k][i_tvec-1];} else {yields[i_y] = fvec[k][i_t] * pow(t_myr/tvec[i_t] , log(fvec[k][i_t+1]/fvec[k][i_t]) / log(tvec[i_t+1]/tvec[i_t]));}}
            /* sum heavy element yields to get the 'total Z' yield here, multiplying by a small correction term to account for trace species not explicitly followed above [mean for CC] */
            yields[0]=0; for(k=2;k<=10;k++) {yields[0] += 1.0144 * yields[k];} // assume here that there is some trace species proportional to each species, not really correct but since it's such a tiny correction this is pretty negligible //
        }
    }
#else
    if(NUM_METAL_SPECIES>=10) { // All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe
        if(SNeIaFlag) { /* SNIa */ /* from Iwamoto et al. 1999; 'W7' models */
            yields[0]=1;/* total metal mass */ yields[1]=0.0;/*He*/ yields[2]=0.035;/*C*/ yields[3]=8.57e-7;/*N*/ yields[4]=0.102;/*O*/
            yields[5]=0.00321;/*Ne*/ yields[6]=0.00614;/*Mg*/ yields[7]=0.111;/*Si*/ yields[8]=0.0621;/*S*/ yields[9]=0.00857;/*Ca*/ yields[10]=0.531;/*Fe*/
        } else { /* SNII (IMF-averaged... may not be the best approx on short timescales..., Nomoto 2006 (arXiv:0605725). here divided by ejecta mass total. */
            yields[0]=0.19;/*Z [total metal mass]*/ yields[1]=0.369;/*He*/ yields[2]=0.0127;/*C*/ yields[3]=0.00456;/*N*/ yields[4]=0.111;/*O*/
            yields[5]=0.0381;/*Ne*/ yields[6]=0.00940;/*Mg*/ yields[7]=0.00889;/*Si*/ yields[8]=0.00378;/*S*/ yields[9]=0.000436;/*Ca*/ yields[10]=0.00706;/*Fe*/
            if(P[i].Metallicity[0]<0.033) {yields[3]*=P[i].Metallicity[0]/All.SolarAbundances[0];} else {yields[3]*=1.65;} /* metal-dependent yields: N scaling is strongly dependent on initial metallicity of the star */
            yields[0] += yields[3]-0.00456; // augment total metal mass for this correction //
        }}
    if(NUM_METAL_SPECIES==3 || NUM_METAL_SPECIES==4) {if(SNeIaFlag) {yields[0]=1; yields[1]=0.00614; yields[2]=0.531; /* All Z, Mg, Fe in total mass (SnIa) */} else {yields[0]=2.0; yields[1]=0.12; yields[2]=0.0741; /* SnII (per-SNe IMF-weighted averages) */}}
    if(NUM_METAL_SPECIES==1) {if(SNeIaFlag) {yields[0]=1;} else {yields[0]=0.19;}} /* toy model to use if only evolving one species */
    int k; for(k=0;k<=NUM_LIVE_SPECIES_FOR_COOLTABLES;k++) {yields[k]=yields[k]*(1.-P[i].Metallicity[0]) + (P[i].Metallicity[k]-All.SolarAbundances[k]);} /* add a check to allow for larger abundances in the progenitor stars (usually irrelevant) */
    if(SNeIaFlag) {if(NUM_METAL_SPECIES>=10) {yields[1]=0.0;}} // no He yield for Ia SNe //
#endif // fire-2 vs fire-2.x yields clause
#ifdef STARFORGE_FEEDBACK_TRACERS
    for(k=0;k<NUM_STARFORGE_FEEDBACK_TRACERS;k++) {yields[NUM_METAL_SPECIES-NUM_STARFORGE_FEEDBACK_TRACERS+k]=0;} yields[NUM_METAL_SPECIES-NUM_STARFORGE_FEEDBACK_TRACERS+2]=1; // this is 'fully' sne material, so mark as such here, so it is noted for all wind routines [whichever form of the wind subroutine we actually use, otherwise it would only appear in the jet version]
#endif
    // just a catch to prevent un-physical yields, need to do this here before handing them off to other routines which may use the yields (currently just the dust routines)
    for(k=0;k<NUM_METAL_SPECIES;k++) {yields[k]=DMIN(1.,DMAX(0.,yields[k]));}
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    ISMDustChem_get_SNe_dust_yields(yields,i,t_gyr,SNeIaFlag,*Msne); // get dust yields
#endif
}
#endif





void particle2in_addFB_winds(struct addFB_evaluate_data_in_ *in, int i)
{
    int k; if(P[i].MassReturn_ThisTimeStep<=0) {in->Msne=0; return;} // no event
    
    /* STELLAR POPULATION-AVERAGED VERSION: calculate wind kinetic luminosity + internal energy (hot winds from O-stars, slow from AGB winds) */
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    double star_age = evaluate_stellar_age_Gyr(i), E_wind_tscaling=0.0013, age_agbthreshold=0.1;
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    age_agbthreshold = 0.03753; // shift the age threshold here to earlier onset for AGB stars because we need to follow their early enrichment. does lower the energetics a bit here, but this is a small effect
#endif
    if(star_age <= age_agbthreshold) {E_wind_tscaling=0.0013 + 16.0/(1+pow(star_age/0.0025,1.4)+pow(star_age/0.01,5.0));} // stellar population age dependence of specific wind energy, in units of an effective internal energy/temperature
    in->SNe_v_ejecta = sqrt(2.0 * (All.StellarMassLoss_Energy_Renormalization * E_wind_tscaling * (3.0e7/((5./3.-1.)*U_TO_TEMP_UNITS)))); // get the actual wind velocity (multiply specific energy by units, user-set normalization, and convert)
#elif (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    double star_age=evaluate_stellar_age_Gyr(i), Z=Z_for_stellar_evol(i), f0=pow(Z,0.12); /* setup: updated fit here uses the same stellar evolution models/tracks as used to compute mass-loss rates. see those for references here. */
    in->SNe_v_ejecta = sqrt(All.StellarMassLoss_Energy_Renormalization) * f0 * (3000./(1.+pow(star_age/0.003,2.5)) + 600./(1.+pow(sqrt(Z)*star_age/0.05,6.)+pow(Z/0.2,1.5)) + 30.) / UNIT_VEL_IN_KMS; /* interpolates smoothly from OB winds through AGB, also versus Z */
#endif
    
#if defined(SINGLE_STAR_FB_WINDS) /* SINGLE-STAR VERSION: instead of a stellar population, this is wind from a single star */
    if(is_particle_single_star_eligible(i))
    {
        double m_msun=P[i].Mass*UNIT_MASS_IN_SOLAR;
        in->SNe_v_ejecta = (616. * sqrt((1.+0.1125*m_msun)/(1.+0.0125*m_msun)) * pow(m_msun,0.131)) / UNIT_VEL_IN_KMS; // scaling from size-mass relation+eddington factor, assuming line-driven winds //
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
        in->SNe_v_ejecta = single_star_wind_velocity(i); /* for fancy models, wind velocity in subroutine, based on Leitherer 1992 and stellar evolutions tage, size, etc. */
#endif
    }
#endif

#ifdef METALS /* assume track initial metallicity; turn on COOL_METAL_LINES_BY_SPECIES for more detailed tracking of light elements */
    double yields[NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION]={0}; get_wind_yields(yields, i);
    for(k=0;k<NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;k++) {in->yields[k]=yields[k];} // assign them back to the vector
#endif
    in->Msne = P[i].Mass * P[i].MassReturn_ThisTimeStep; // mass (in code units) returned
}


#ifdef METALS
void get_wind_yields(double *yields, int i)
{
    int k; for(k=0;k<NUM_METAL_SPECIES;k++) {yields[k]=P[i].Metallicity[k];} // return surface abundances, to leading order //
#if defined(SINGLE_STAR_FB_WINDS)
    if(is_particle_single_star_eligible(i))
    {
#ifdef STARFORGE_FEEDBACK_TRACERS
        for(k=0;k<NUM_STARFORGE_FEEDBACK_TRACERS;k++) {yields[NUM_METAL_SPECIES-NUM_STARFORGE_FEEDBACK_TRACERS+k]=0;} yields[NUM_METAL_SPECIES-NUM_STARFORGE_FEEDBACK_TRACERS+1]=1; // this is 'fully' wind material, so mark as such here, so it is noted for all wind routines [whichever form of the wind subroutine we actually use, otherwise it would only appear in the jet version]
#endif
#if !defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL)
        return; /* nothing more complicated gets modeled here, just use initial surface abundances */
#endif
    }
#endif
    if(NUM_METAL_SPECIES>=10) /* All, then He,C,N,O,Ne,Mg,Si,S,Ca,Fe ; follow AGB/O star yields in more detail for the light elements */
    {
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        /* everything except He and CNO and S-process is well-approximated by surface abundances. and CNO is conserved to high accuracy in sum for secondary production */
        double f_H_0=1.-(yields[0]+yields[1]), f_He_0=yields[1], f_C_0=yields[2], f_N_0=yields[3], f_O_0=yields[4], f_CNO_0=f_C_0+f_N_0+f_O_0+MIN_REAL_NUMBER, y; // define initial H, He, CNO fraction
        double t = evaluate_stellar_age_Gyr(i), z_sol; z_sol = f_CNO_0 / (All.SolarAbundances[2]+All.SolarAbundances[3]+All.SolarAbundances[4]); // stellar population age in Gyr, and solar-scaled CNO abundance
        // model He production : this scales off of the fraction of H in IC: y here represents the yield of He produced by burning H, scales off availability
        {double t1=0.0028, t2=0.01, t3=2.3, t4=3.0, y1=0.4*DMIN(pow(z_sol+1.e-3,0.6),2.), y2=0.08, y3=0.07, y4=0.042;
            if(t<t1) {y=y1*pow(t/t1,3);} else if(t<t2) {y=y1*pow(t/t1,log(y2/y1)/log(t2/t1));} else if(t<t3) {y=y2*pow(t/t2,log(y3/y2)/log(t3/t2));} else if(t<t4) {y=y3*pow(t/t3,log(y4/y3)/log(t4/t3));} else {y=y4;}}
        yields[1] = f_He_0 + y * f_H_0; // y above //
        // model secondary N production in CNO cycle: scales off of initial fraction of CNO: y here represents fraction of CO mass converted to -additional- N
        {double t1=0.001, t2=0.0028, t3=0.05, t4=1.9, t5=14.0, y1=0.2*DMAX(1.e-4,DMIN(z_sol*z_sol,0.9)), y2=0.68*DMIN(pow(z_sol+1.e-3,0.1),0.9), y3=0.4, y4=0.23, y5=0.065;
            if(t<t1) {y=y1*pow(t/t1,3.5);} else if(t<t2) {y=y1*pow(t/t1,log(y2/y1)/log(t2/t1));} else if(t<t3) {y=y2*pow(t/t2,log(y3/y2)/log(t3/t2));} else if(t<t4) {y=y3*pow(t/t3,log(y4/y3)/log(t4/t3));} else if(t<t5) {y=y4*pow(t/t4,log(y5/y4)/log(t5/t4));} else {y=y5;}}
        y=DMAX(0.,DMIN(1.,y)); double frac_loss_from_C = 0.5, floss_CO = y * (f_C_0 + f_O_0), floss_C = DMIN(frac_loss_from_C * floss_CO, 0.99*f_C_0), floss_O = floss_CO - floss_C;
        yields[3] = f_N_0 + floss_CO; yields[2] = f_C_0 - floss_C; yields[4] = f_O_0 - floss_O; // convert mass from CO to N, conserving exactly total CNO mass
        // model primary C production: scales off initial H+He, generally small compared to loss fraction above in SB99, large in some other models, very small for early OB winds
        {double t1=0.005, t2=0.04, t3=10., y1=1.e-6, y2=0.001, y3=0.005;
            if(t<t1) {y=y1*pow(t/t1,3);} else if(t<t2) {y=y1*pow(t/t1,log(y2/y1)/log(t2/t1));} else if(t<t3) {y=y2*pow(t/t2,log(y3/y2)/log(t3/t2));} else {y=y3;}}
        double y_H_to_C = (1.-(yields[0]+yields[1])) * y, y_He_to_C = f_He_0 * y; // simply multiple initial He by this factor to get final production
        yields[1] -= y_He_to_C; yields[2] += y_H_to_C + y_He_to_C; // transfer this mass fraction from H+He to C; gives stable results if 0 < f_He_0_to_C < 1
        // model S-process production: currently no S-process tracers -explicitly- followed, so we skip this step
        yields[0]=0.0; for(k=2;k<=NUM_LIVE_SPECIES_FOR_COOLTABLES;k++) {yields[0]+=yields[k];} // finally, add up metals [not He!] to get actual metal yield
#else
        /* the interesting species are He & CNO: below is based on a compilation of van den Hoek & Groenewegen 1997, Marigo 2001, Izzard 2004 */
        yields[1]=0.36; /*He*/ yields[2]=0.016; /*C*/ yields[3]=0.0041; /*N*/ yields[4]=0.0118; /*O*/
        if(P[i].Metallicity[0]<0.033) {yields[4] *= P[i].Metallicity[0]/All.SolarAbundances[0];} else {yields[4] *= 1.65;} // metal-dependent yields: O scaling is strongly dependent on initial metallicity of the star //
        yields[0]=0.0; for(k=2;k<=NUM_LIVE_SPECIES_FOR_COOLTABLES;k++) {yields[0]+=yields[k];}
#endif
    } else {yields[0]=0.032; for(k=1;k<NUM_METAL_SPECIES;k++) {yields[k]=0.0;}} /* if <10 species, adopt toy model for simple enrichment [not using extended networks] */
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    // just a catch to prevent un-physical yields, need to do this here before handing them off to other routines which may use the yields (currently just the dust routines)
    for(k=0;k<NUM_METAL_SPECIES;k++) {yields[k]=DMIN(1.,DMAX(0.,yields[k]));}    
    ISMDustChem_get_wind_dust_yields(yields,i); // get dust yields
#endif
}
#endif


#endif // GALSF_FB_MECHANICAL+GALSF_FB_FIRE_STELLAREVOLUTION



double Z_for_stellar_evol(int i)
{
    if(i<0) {return 1;}
#ifdef METALS
    double Z_solar = P[i].Metallicity[0]/All.SolarAbundances[0]; // use total metallicity
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2) && defined(COOL_METAL_LINES_BY_SPECIES)
    int i_Fe=10; Z_solar = P[i].Metallicity[i_Fe]/All.SolarAbundances[i_Fe]; // use Fe, specifically, for computing stellar properties, as its most relevant here. MAKE SURE this is set to the correct abundance in the list, to match Fe!
#endif
    return DMIN(DMAX(Z_solar,0.01),3.); // stellar evolution tables here are not arbitrarily extrapolable, so this is bounded //
#else
    return 1; // metals not evolved, return unity
#endif
}


#if defined(GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF)
void update_stellarnumber_and_timedistribofstarformation(void)
{
    int i; for (int i : ActiveParticleList)
    {
        if(P[i].Type == 4 && P[i].Mass > 0)
        {
            double dt_since_form_code = evaluate_time_since_t_initial_in_Gyr(P[i].StellarAge) / UNIT_TIME_IN_GYR; // time since spawn in code [physical] units
            if((dt_since_form_code < P[i].TimeDistribOfStarFormation) && (P[i].TimeDistribOfStarFormation > 0)) // candidate for 'spawning'
            {
                double dt_remaining = P[i].TimeDistribOfStarFormation - dt_since_form_code; // time remaining to spawn
                double dt_timestep = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i); // timestep being taken [code units]
                double d_tau = DMAX(DMIN( dt_timestep , dt_remaining) , 0) / P[i].TimeDistribOfStarFormation; // effective step size in dimensionless units
                double f_highz=0.0115, f_m; f_m = f_highz; // 1/mass in solar per O-star number
#ifdef FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT // SIMPLE_POPTHREE_MODEL
                double f_lowz=2., zmin=-7, zmax=-5, z0=log10(P[i].Metallicity[0]/0.01 + 1.e-15);
#if (FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT > 3)
                zmin=-5.; zmax=-3.;
#endif
                if(z0<=zmin) {f_m=f_lowz;} else if(z0<zmax) {f_m=f_lowz + (f_highz-f_lowz)*(z0-zmin)/(zmax-zmin);}
#endif
                double n_expected_total = f_m * P[i].Mass * UNIT_MASS_IN_SOLAR; // default f_m above is 1 O-star per 100 Msun [more exactly calculated here as number of stars per solar mass with mass > 8 Msun, from our adopted Kroupa IMF from 0.01-100 Msun]
                double dn_expected_in_dt = n_expected_total * d_tau; // expected number to spawn in this particular interval
                gsl_rng *random_generator_for_massivestars; // allocate rng
                random_generator_for_massivestars = gsl_rng_alloc(gsl_rng_ranlxd1); // setup generator
                gsl_rng_set(random_generator_for_massivestars, P[i].ID + 49531 + All.NumCurrentTiStep); // seed random number
                unsigned int n_to_add = gsl_ran_poisson(random_generator_for_massivestars, dn_expected_in_dt); // actually draw the number
                if(n_to_add > 0) // hey, new stars!
                {
                    P[i].IMF_WeightedMeanStellarFormationTime = (P[i].IMF_NumMassiveStars * P[i].StellarAge + n_to_add * All.Time) / (P[i].IMF_NumMassiveStars + n_to_add); // update the effective stellar age (so e.g. if we lose stars, this can keep updating)
                    P[i].IMF_NumMassiveStars += n_to_add; // add this to the number of massive stars that we are tracking explicitly
                }
            }
        }
    }
}
#endif


#ifdef METALS
void get_jet_yields(double *yields, int i) {
    int k; for(k=0;k<NUM_METAL_SPECIES;k++) {yields[k]=P[i].Metallicity[k];} /* return surface abundances, to leading order */
#ifdef SNE_NONSINK_SPAWN
    double t_gyr=evaluate_stellar_age_Gyr(i); int SNeIaFlag=0; if(t_gyr>0.03753) {SNeIaFlag=1;}; /* assume SNe before critical time are core-collapse, later are Ia */
    if(P[i].Type==4) {double Msne; get_SNe_yields(yields,i,t_gyr,SNeIaFlag,&Msne);}
#endif
#ifdef STARFORGE_FEEDBACK_TRACERS
    for(k=0;k<NUM_STARFORGE_FEEDBACK_TRACERS;k++) {yields[NUM_METAL_SPECIES-NUM_STARFORGE_FEEDBACK_TRACERS+k]=0;} yields[NUM_METAL_SPECIES-NUM_STARFORGE_FEEDBACK_TRACERS+0]=1; // this is 'fully' jet material, so mark as such here, so it is noted for all wind routines [whichever form of the wind subroutine we actually use, otherwise it would only appear in the jet version]
#endif
}
#endif



#ifdef SINGLE_STAR_FB_JETS
/* Calculates the launch velocity of jets, in physical (not comoving) units */
double single_star_jet_velocity(int n)
{
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) /* use the fancy stellar evolution modules to calculate these for stars or protostars */
    return (All.Sink_outflow_jetlaunchvelscaling * sqrt(All.G * P[n].Sink_Mass / (P[n].ProtoStellarRadius_inSolar / UNIT_LENGTH_IN_SOLAR))); // we use the flag as a multiplier times the Kepler velocity at the protostellar radius. Really we'd want v_kick = v_kep * m_accreted / m_kicked to get the right momentum
#else
    return (All.Sink_outflow_jetlaunchvelscaling * sqrt(All.G * P[n].Sink_Mass / (10. / UNIT_LENGTH_IN_SOLAR))); // we use the flag as a multiplier times the Kepler velocity at the protostellar radius. Really we'd want v_kick = v_kep * m_accreted / m_kicked to get the right momentum; without a better guess, assume fiducial protostellar radius of 10*Rsun, as in Federrath 2014
#endif
}
#endif



#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) /* begins large block of 'fancy' protostar-through-MS stellar evolution models */

/* 'top-level' function to update the size (and other properties like effective temperature) of accreting protostars along relevant stellar evolution tracks */
void singlestar_subgrid_protostellar_evolution_update_track(int n, double dm, double dt)
{
    if(P[n].Type != 5) {return;} // only type-5 particles are eligible here
    
#if (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 1)
    /* this is the simple version written by Phil: intentionally simplified PS evolution tracks, designed to make it easy to understand and model the evolution and reduce un-necessary complications */
    double lum_sol = 0.0, m_initial = DMAX(1.e-37 , (P[n].Sink_Mass - dm)), mu = DMAX(0, dm/m_initial), m_solar = P[n].Sink_Mass*UNIT_MASS_IN_SOLAR, T4000_4 = pow(m_solar, 0.55); // m_initial = mass before the accretion, mu = relative mass accreted, m_solar = mass in solar units, T4000_4 = (temperature/4000K)^4 along Hayashi track
    if(m_solar > 0.012) // below this limit, negligible luminosity //
    {
        if(m_solar < 0.43) {lum_sol = 0.185 * m_solar*m_solar;} else if(m_solar < 2.) {lum_sol = m_solar*m_solar*m_solar*m_solar;}
        else if(m_solar < 53.9) {lum_sol = 1.5 * m_solar*m_solar*m_solar * sqrt(m_solar);} else {lum_sol = 32000. * m_solar;}
    }
    double R_Hayashi_Henyey = 2.1 * sqrt(lum_sol / T4000_4); // size below which, at the temperature above, contraction must occur along the Henyey track at constant luminosity
    double t_R_evol = 0, contraction_factor = 0; // timescale for contraction
    if(P[n].ProtoStellarRadius_inSolar <= R_Hayashi_Henyey)
    {// currently on Henyey track, contracting at constant Luminosity
        t_R_evol = 1.815e7 * m_solar*m_solar / (P[n].ProtoStellarRadius_inSolar * lum_sol) / (UNIT_TIME_IN_YR); // contraction timescale
        contraction_factor = 1. / (1. + dt/t_R_evol);
    } else {// currently on Hayashi track, contracting at constant Temperature
        t_R_evol = 8.021e7 * m_solar*m_solar / (P[n].ProtoStellarRadius_inSolar*P[n].ProtoStellarRadius_inSolar*P[n].ProtoStellarRadius_inSolar * T4000_4) / (UNIT_TIME_IN_YR); // contraction timescale
        contraction_factor = 1. / pow(1 + 3.*dt/t_R_evol, 1./3.);
    }
    double r_new = DMAX(10. * m_solar, 5.24 * pow(m_solar, 1./3)), R_main_sequence_ignition; // r_new = size of newly-formed protostar; R_main_sequence_ignition = main sequence radius - where contraction should halt
    if (m_solar < 0.012) {r_new =  5.24 * pow(m_solar, 1./3);} // constant density for Jupiter-type objects
    P[n].ProtoStellarRadius_inSolar = (P[n].ProtoStellarRadius_inSolar * contraction_factor + r_new * mu) / (1. + mu); // new size (contraction + accretion both accounted for)
    if(m_solar <= 1) {R_main_sequence_ignition = pow(m_solar,0.8);} else {R_main_sequence_ignition = pow(m_solar,0.57);}
    if(P[n].ProtoStellarRadius_inSolar <= R_main_sequence_ignition)
    {
        P[n].ProtoStellarRadius_inSolar = R_main_sequence_ignition; P[n].ProtoStellarStage = 5; // using same notation for MS as SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 1
    }
    
#elif (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 2) /* Protostellar evolution model based on the ORION version, see Offner 2009 Appendix B */
    double fk = 1.; // fraction of kinetic energy accreta has when it arrives at the accretion shock, relative to freefall from infinity
    double f_acc = 1.; // fraction of accretion power that is radiated away (the rest is advected into the star)
    double max_rel_dr = 0.01; // maximum relative change in radius per step, if the change over a single timestep is larger than this than we subcycle the evolution of the stellar radius
    double mass = P[n].Sink_Mass; // mass of star/protostar at the end of the timestep
    double mass_D = P[n].Mass_D; // amount of D in the protostar
    double mdot = P[n].Sink_Mdot; // accretion rate, shorter to write it this way
    double mdot_m_solar_per_year = mdot * UNIT_MASS_IN_SOLAR / UNIT_TIME_IN_YR; // accretion rate in msolar/yr
    double m_solar = mass * UNIT_MASS_IN_SOLAR; // mass in units of Msun
    double m_initial = DMAX(1.e-37 , (mass - dm)); // mass before accretion
    int stage = P[n].ProtoStellarStage; /* what stage of stellar evolution the particle is in 0: pre collapse, 1: no burning, 2: fixed Tc burnig, 3: variable Tc burning, 4: shell burning, 5: main sequence, 6: SNe, 7: relic, see Offner 2009 Appendix B*/
    if (stage == 0) {// set the radius for the pre-collapse phase according to Eq B1 in Offner 2009, this overwrites the original prescription from sfr_eff.c
        P[n].ProtoStellarRadius_inSolar = DMAX(2.5 * pow(mdot_m_solar_per_year/1e-5,1./3), 2.);
    }
    double r = P[n].ProtoStellarRadius_inSolar / UNIT_LENGTH_IN_SOLAR; // star radius in code units
    int stage_increase = 0;
    double lum_Hayashi = ps_lum_Hayashi_BB(mass, r); // blackbody radiation assuming the star follows the Hayashi track
    double lum_MS = ps_lum_MS(mass); // luminosity of main sequence star of m mass
    double lum_int = DMAX(lum_Hayashi, lum_MS); // luminosity from the stellar interior
    double lum_acc = f_acc*fk*All.G*mass*mdot/r; // accretion luminosity
    double lum_I = ps_lum_I(mdot); // luminosity needed to ionize the accreted material
    double t_KH = All.G*mass*mass/r/lum_int; // Kelvin-Helmholtz time
    double t_acc = mass/(mdot + MIN_REAL_NUMBER);
    double n_ad, ag, rhoc, Pc, Tc, beta, dlogbeta_dlogm, lum_D, dm_D, rel_dr;
    int loop_subcycle=0;
    int n_subcycle=(int) (dt/(0.1*DMIN(All.G * mass * mass / r / (lum_int+lum_acc+lum_I), t_acc)) + 1); // subcycle timestep criterion taking the shorter of the kelvin-helmholtz or accretion timescales
    if(stage < 5) { // not a main sequence star
        if (stage >= 1) { // we only evolve those that are beyond the pre-collapse phase
            double dm_curr = dm/n_subcycle; double dt_curr = dt/n_subcycle;
            mass = m_initial; // value at the beginning o the timestep
            for(loop_subcycle=0; loop_subcycle<n_subcycle; loop_subcycle++){
                // Note: this subcycling does not update stage, so a protostar could overshoot
                // Evolve mass
                double dm_rel = dm_curr/mass;
                mass += dm_curr; // increase mass first
                // Get properties for stellar evolution
                lum_Hayashi = ps_lum_Hayashi_BB(mass, r); // blackbody radiation assuming the star follows the Hayashi track
                lum_MS = ps_lum_MS(mass); // luminosity of main sequence star of m mass
                lum_int = DMAX(lum_Hayashi, lum_MS); // luminosity from the stellar interior
                lum_acc = f_acc*fk*All.G*mass*mdot/r; // accretion luminosity
                t_KH = All.G*mass*mass/r/lum_int; // Kelvin-Helmholtz time
                t_acc = mass/(mdot + MIN_REAL_NUMBER);
                n_ad = ps_polytropic_index(stage, mdot, mass); // get polytropic index. Note: ORION does not seem to update this, but I think it is worthwhile as mdot can vary over time
                ag = 3.0/(5.0-n_ad); //shorthand
                rhoc = ps_rhoc(mass, n_ad, r); // central density
                Pc = ps_Pc(mass, n_ad, r); // central pressure
                Tc = ps_Tc(rhoc,Pc); // central temperature
                beta = ps_beta(mass, n_ad, rhoc, Pc); // mean ratio of gas pressure to total pressure
                dlogbeta_dlogm = ps_dlogbeta_dlogm(mass, r, n_ad, beta, rhoc, Pc); // d log beta/ d log m
                // Calculate luminosity from D burning
                lum_D = 0; // luminosity from D burning
                dm_D = dm_curr; // by default we burn no D (stage 1)
                if (stage==2){ // burning at fixed Tc, lum_D set to keep the central temperature constant
                    double dlogbetaperbetac_dlogm = ps_dlogbetaperbetac_dlogm(mass, r, n_ad, beta, rhoc, Pc, Tc); // ratio of gas pressure to total pressure at the center
                    lum_D = lum_int + lum_acc + lum_I + (All.G*mass*mdot/r) * ( 1.-fk-0.5*ag*beta * (1.+dlogbetaperbetac_dlogm) ); // Eq B8 of Offner 2009
                    // Change in available deuterium mass
                    dm_D = dm_curr - dt_curr * (lum_D*UNIT_LUM_IN_SOLAR/15.) * (1e-5) / (UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR) ;
                } else if(stage>2){
                    // burning all accreted D for stages above 2
                    lum_D = (15./UNIT_LUM_IN_SOLAR) * (mdot_m_solar_per_year/(1e-5));
                    dm_D = 0; // all new D is burned
                    mass_D = 0; // no D left in protostar
                }
                // Let's evolve the stellar radius
                //		double dlogr_dlogm = 2 - 2/(ag*beta)*(1-fk) + dlogbeta_dlogm - 2*r/(ag*beta*All.G*mass*mdot) * (lum_int+lum_acc+lum_I-lum_D); // Nakano 2000 Eq (8); this gives the slope of the M vs. R diagram
                //		rel_dr = dlogr_dlogm * dm_rel; // multiply by dlogm = dm/m to get dlogr
                double rel_dr = (2 - 2/(ag*beta)*(1-fk) + dlogbeta_dlogm) * dm_rel - 2*r*dt_curr/(ag*beta*All.G*mass*mass) * (lum_int+lum_acc+lum_I-lum_D);
                r *= exp(rel_dr); // interpolates properly to large timestep, but subcycling should limit it
                mass_D += dm_D;
            }
            
            // Update stellar properties
            P[n].ProtoStellarRadius_inSolar = r * UNIT_LENGTH_IN_SOLAR;
            P[n].Mass_D = mass_D;
            // Debug message
            // Check whether the star can progress to the next state
            // move from "no burn" to "burning at fixed Tc" phase when central temperature gets high enough for D ignition
            if ( (stage==1) && (Tc >= 1.5e6) && ((All.Time-P[n].StellarAge) > DMAX(3.*dt, 1e-4/(UNIT_TIME_IN_MYR)) ) ){ // further check that the sink has been promoted at least a couple of timesteps and 100 yr ago, so that we don't start D burning immediately after forming the sink (relevant in low res cases)
                stage_increase = 1; // particle qualifies to the "fixed Tc burn" phase
            }
            // move from "burning at fixed Tc" to "variable Tc burn" phase when D runs out
            if ( (stage==2) && (P[n].Mass_D <= 0) ){
                P[n].Mass_D = 0;
                stage_increase = 1; // particle qualifies to the "variable Tc burn" phase
            }
            // move from "variable Tc burn" to "shell burn" phase when radiation becomes strong enough to form a radiative zone
            if ( (stage==3) && (t_KH/t_acc < 3.) ){
                stage_increase = 1; // particle qualifies to the "shell burn" phase
                P[n].ProtoStellarRadius_inSolar *= 3; //star swells due to the formation of radiative barrier
            }
            // move from "shell burn" to "main sequence" phase when the radius reaches the main sequence radius
            if ( (stage<5) && (P[n].ProtoStellarRadius_inSolar <= ps_radius_MS_in_solar(mass)) ){
                stage_increase = 1; // particle qualifies to become a ZAMS star
                P[n].ProtoStellarRadius_inSolar = ps_radius_MS_in_solar(mass);
            }
        }
        else{ // the protostar is in the "pre-collapse" state, no internal evolution, just check if it can be promoted to the next stage
            P[n].Mass_D = P[n].Sink_Mass; //no D burned so far
            if (m_solar >= 0.01){
                stage_increase = 1; // particle qualifies to the "no burning stage"
            }
        }
        if(stage_increase) {
            P[n].ProtoStellarStage += stage_increase;
            P[n].StellarAge = All.Time; // store the time of the last promotion
            if(P[n].ProtoStellarStage == 5) {P[n].ZAMS_Mass = P[n].Sink_Mass;} // store the mass at which we reached the main sequence
        } //increase evolutionary stage if the particle satisfies the requirements
    }
    else{ // for main sequence stars or SNe or relics
        if(P[n].ProtoStellarStage == 5) { // MS stars
            P[n].ProtoStellarRadius_inSolar = ps_radius_MS_in_solar(mass); // update the mass if the mass changes (unlikely)
#ifdef SINGLE_STAR_FB_SNE
            // Check age and see if we need to blow up this star
            double age_Gyr = evaluate_stellar_age_Gyr(n);
            if ( age_Gyr > stellar_lifetime_in_Gyr(n) ) {
                P[n].ProtoStellarStage = 6; // time to explode
                P[n].Mass_final = P[n].Sink_Mass; // record the final mass the star had
                double n_stars_within_star_particle = 1; // allow for compact binary representations that might evolve differently
#if defined(SINK_COUNTPROGS) && defined(SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES)
                n_stars_within_star_particle = (double)P[i].Sink_CountProgs; // this was our tracker of binary consolidation events
#endif
                if(P[n].Sink_Mass/n_stars_within_star_particle <= 1.4/UNIT_MASS_IN_SOLAR) {P[n].ProtoStellarStage = 7;} // should collapse to WD, no explosion
#ifdef SINGLE_STAR_RELICS
                // if(P[n].Sink_Mass < 20./UNIT_MASS_IN_SOLAR) {P[n].Sink_Mass = 1.4;} // Collapse to NS of mass 1.4 Msun. Still maintain ProtoStellarStage = 6 -- handled in SNe routines now, at -end- of SNe
                if(P[n].Sink_Mass >= 20./UNIT_MASS_IN_SOLAR) {P[n].ProtoStellarStage = 7;} // direct-collapse to relic (BH), skipping SNe entirely
#endif
#if defined(SINK_ALPHADISK_ACCRETION)
                P[n].Sink_Mass_Reservoir = 0; // probably does not matter, but let's make sure these don't cause issues
                P[n].Mass = P[n].Sink_Mass;
#endif
                // Save properties of SN progenitor.
                char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE]; FILE *FdSinkSNDetails;
                snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s%s", All.OutputDir, "SN_details.txt");
                FdSinkSNDetails = fopen(buf, "a");
                fprintf(FdSinkSNDetails, "%.16g %llu %g %g %g %g %g %g %g \n", All.Time, (unsigned long long)P[n].ID, P[n].Sink_Mass, P[n].Pos[0], P[n].Pos[1],P[n].Pos[2],P[n].Vel[0], P[n].Vel[1],P[n].Vel[2]); fflush(FdSinkSNDetails);
                fclose(FdSinkSNDetails);
            }
#endif
        }
    }
    // Calculate the luminosity of the star
    /*********************************************/
    /* Power based parametrization from ORION (2 params)*/
    // lum_acc = facc * fk * All.G*mass*mdot/r; // luminosity radiated at the accretion shock
    // lum_disk = (1.-fk) * All.G*mass*mdot/r; // luminosity released by material that traverses the inner disk
    // P[n].StarLuminosity_Solar = (lum_acc + lum_disk + lum_int) * UNIT_LUM_IN_SOLAR ; //luminosity of the star
    /*********************************************/
    
    P[n].StarLuminosity_Solar = ((f_acc*fk + (1-fk)) * All.G*mass*mdot/r + lum_int) * UNIT_LUM_IN_SOLAR; // disk + accretion shock + internal luminosity (=MAX(L_MS,L_Hayashi))
    
    if(P[n].ProtoStellarStage == 0) {P[n].StarLuminosity_Solar = 0;} // no luminosity yet
    if(P[n].ProtoStellarStage == 7) {
        P[n].StarLuminosity_Solar = 0; // default to no luminosity for safety
#ifdef SINGLE_STAR_RELICS
        if(P[n].Mass_final > 1.4) { // only interested here in the systems with -compact- remnants, for this calculation
            P[n].ProtoStellarRadius_inSolar = 6. * (2.12e-6 * P[n].Sink_Mass*UNIT_MASS_IN_SOLAR); // relics 'radius' = 6 * (G*Mbh/c^2), for ISCO
            double eps_r = evaluate_sink_radiative_efficiency(mdot,mass,n); // get compact remnant radiative efficiency
            P[n].StarLuminosity_Solar = eps_r * (1.44e13 * mdot_m_solar_per_year); // radiative efficiency times Mdot*c^2, in units of Lsolar
        }
#endif
    }
#endif //end of SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 2
    
#if defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    P[n].MaxFeedbackVel = single_star_feedback_velocity_fortimestep(n);
#endif
    
    return;
}

#if defined(SINGLE_STAR_FB_WINDS)
/* Let's get the wind mass loss rate for MS stars. n is the index of the particle (P[n]). mode is 1 when called by the wind spawning routine (sink.c) and 2 if called by the FIRE wind module (in this file, mechanical_fb_calculate_eventrates_Winds). The function decides which type of wind feedback is appropriate for the current star and will only give a nonzero mdot to one of these */
double single_star_wind_mdot(int n, int set_mode) { //if set_mode is zero then the wind mode is not changed by calling this function
    if(P[n].Type != 5) {return 0;} // only type-5 particles are eligible here

    double minimum_stellarmass_for_winds_solar  = 2.0;  // minimum stellar mass allowed to have winds
    int    model_wolf_rayet_phase_explicitly    = 1;    // assumes that O stars turn into WR stars at the end of their lifetime, increasing their mass loss rate
    double n_particles_for_discrete_wind_spawn  = 1e-1; // parameter for switching between wind spawning and just depositing momentum to nearby gas (FIRE winds) -- particle number required to trigger 'explicit' spawn module. Setting it to 0 ensures that we always spawn winds, while a high value (e.g. 1e6) ensures we always use the FIRE wind module

    double wind_mass_loss_rate=0; //mass loss rate in code units
    if(P[n].ProtoStellarStage != 5) {return 0;}
    double m_solar = P[n].Mass * UNIT_MASS_IN_SOLAR; // mass in units of Msun
    if (m_solar < minimum_stellarmass_for_winds_solar){return 0.0;} //no winds for low mass stars
    // winds are for MS only: we are assuming that METALS are also on
    double ZZ = P[n].Metallicity[0]/All.SolarAbundances[0]; //relative metallicity to solar
    double logmdot_wind; // log10(Mdot / (Msun/yr))

    // phenomenological prescription: "de Jager / 3" model from Smith 2014, with limiter for "weak-wind problem"
    double L = ps_lum_MS(P[n].Sink_Mass) * UNIT_LUM_IN_SOLAR, R = P[n].ProtoStellarRadius_inSolar; double Teff = 5780.*sqrt(sqrt(L/(R*R)));
    logmdot_wind = -6 + 1.5 * log10(P[n].StarLuminosity_Solar / 1e6) + 0.69 * log10(ZZ); // "de Jager / 3"
#if (SINGLE_STAR_FB_WINDS > 0)
    double vinf_over_vesc = single_star_wind_velocity(n)/single_star_escape_speed(n);
#endif 
#if (SINGLE_STAR_FB_WINDS & 1) // least significant bit toggles Vink 2001 model
    if(Teff > 2.25e4) { // hot side of the bistability jump
	    double logT40k = log10(Teff/4e4);
	    logmdot_wind = -6.697 + 2.194 * log10(L/1e5) 	    
	    - 1.313 * log10(m_solar/30) 
	    - 1.226 * log10(vinf_over_vesc/2)
	    + 0.933 * logT40k 
	    - 10.92 * logT40k * logT40k 
	    + 0.85 * log10(ZZ); // Sahahit arXiv:2205.09125 Eq. 10
    } else {
	    logmdot_wind = -6.688 + 2.21 * log10(L/1e5) 	    
	    - 1.339 * log10(m_solar/30) 
	    - 1.601 * log10(vinf_over_vesc/2)
	    + 1.07 * log10(Teff/2e4)
	    + 0.85 * log10(ZZ); // Vink 2001,  arXiv:astro-ph/0101509
    }
#else // default STARFORGE phenomenological prescription: "de Jager / 3" model from Smith 2014, with limiter for "weak-wind problem"
    logmdot_wind = -6 + 1.5 * log10(P[n].StarLuminosity_Solar / 1e6) + 0.69 * log10(ZZ);
#endif
    logmdot_wind = DMIN(-7.65 + 2.9*log10(P[n].StarLuminosity_Solar/ 1e5), logmdot_wind); // weak-wind problem

    if(model_wolf_rayet_phase_explicitly) {if(evaluate_stellar_age_Gyr(n) > (stellar_lifetime_in_Gyr(n)-singlestar_WR_lifetime_Gyr(n))){logmdot_wind += 1.;}} //Our star is in the WR phase, for now use the simple prescription of having 10x higher wind loss rates based on Smith 2014

#if (SINGLE_STAR_FB_WINDS & 2) // next-to-least significant bit toggles Eddington factor-dependent mass loss recipe 
    double logmdot_wind_high = -8.445 + 4.77* log10(L/1e5)
	- 3.99 * log10(m_solar/30) 
	- 1.226 * log10(vinf_over_vesc/2)
	+ 0.761 * log10(ZZ);  // Sahahit arXiv:2205.09125 Eq. 13
    logmdot_wind = DMAX(logmdot_wind, logmdot_wind_high);
#endif
    wind_mass_loss_rate = pow(10.0,logmdot_wind) / (UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR); //reducing the rate to be more in line with observations, see Nathan Smith 2014, conversion to code units from Msun/yr

    // let's deal with the case of undefined wind mode (just promoted to MS or restart from snapshot)
    if ( set_mode && (wind_mass_loss_rate>0) ) {
        // let's calculate N_wind = Mdot_wind * t_wind / dm_wind, where t_wind is solved from: Mdot_wind * t_wind = material swept up = 4/3 pi rho (v_wind*t_wind)^3
        double v_wind = single_star_wind_velocity(n);
        double t_wind =sqrt( wind_mass_loss_rate * (3.0/(4.0*M_PI*P[n].DensityAroundParticle)) / (v_wind*v_wind*v_wind));
        double N_wind = wind_mass_loss_rate * t_wind / target_mass_for_wind_spawning(n);

        int old_wind_mode = P[n].wind_mode;
        if (N_wind >= n_particles_for_discrete_wind_spawn){
            P[n].wind_mode = 1; // we can spawn enough particles per wind time
        } else{
            P[n].wind_mode = 2; // we can't spawn enough particles per wind time, switching to FIRE wind module to reduce burstiness
        }
#ifdef SINGLE_STAR_FB_JETS
        double spawning_min_wind_jet_mom_ratio = 10.0; // if winds are much more powerful than jets ( (wind momentum injection/jet momentum injection) > this value) then we can safely spawn the winds and neglect the jets if we want to
        if ( (P[n].wind_mode == 1) && (P[n].Sink_Mdot>0) ){ // we want to spawn winds but we have jets too
            double jet_mom_inj = single_star_jet_velocity(n) * P[n].Sink_Mdot;
            double wind_mom_inj = v_wind * wind_mass_loss_rate;
            if (spawning_min_wind_jet_mom_ratio < (spawning_min_wind_jet_mom_ratio * jet_mom_inj) ){ P[n].wind_mode = 2;} //w e switch back to the FIRE wind injection so that we can spawn the jet and have winds at the same time
        }
#endif
        if (old_wind_mode != P[n].wind_mode){
            printf("Wind mode change for star %llu to %d at %g. Mdot_wind %g\n",(unsigned long long)P[n].ID,P[n].wind_mode,All.Time, wind_mass_loss_rate);
        }
    }
    return wind_mass_loss_rate;
}

double singlestar_WR_lifetime_Gyr(int n) { // calculate lifetime for star in Wolf-Rayet Phase
    double m_solar = P[n].Mass * UNIT_MASS_IN_SOLAR; // mass in units of Msun
    double ZZ = P[n].Metallicity[0]/All.SolarAbundances[0]; // relative metallicity to solar
    if (m_solar<=20.0){return 0.;} // no WR phase below that
    return DMAX(0., 1.5e-3 * DMIN(1., ((m_solar-20.)/80.)) * pow(ZZ,0.5) ); // using prescription based on Fig 7 from Meynet & Maeder 2005, all >10 Msun star spend the end of their lifetime as WR
}


double single_star_wind_velocity(int n) { /* Let's get the wind velocity for MS stars */
    double T_eff = 5814.33 * pow( P[n].StarLuminosity_Solar/(P[n].ProtoStellarRadius_inSolar*P[n].ProtoStellarRadius_inSolar), 0.25 ); // effective temperature in K
    double v_esc = single_star_escape_speed(n); // surface escape velocity - wind escape velocity should be O(1) factor times this, factor given below
    if(T_eff < 1.25e4){ return 0.7 * v_esc;}  // Lamers 1995
    else if (T_eff < 2.1e4) {return 1.3 * v_esc;}
    else {return 2.6 * v_esc;}
}

double single_star_escape_speed(int n){
    return (617.7 / UNIT_VEL_IN_KMS) * sqrt(P[n].Mass*UNIT_MASS_IN_SOLAR / P[n].ProtoStellarRadius_inSolar);
}
#endif // SINGLE_STAR_FB_WINDS


#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
/* Computes the maximum signal velocity of _any_ feedback mechanism emanating from a star (jets, winds, radiation, SNe), as a worst-case for e.g. timestepping stability purposes */
double single_star_feedback_velocity_fortimestep(int n) {
#ifdef SELFGRAVITY_OFF
    return 0; 
#endif    
    if(P[n].Type != 5) {return 0;}
    double v_fb, force, h, rho, v_shell; v_fb=0; force=0; h=Get_Particle_Size(n); rho=P[n].DensityAroundParticle; v_shell=0;
#ifdef SINGLE_STAR_FB_WINDS
    double v_wind = single_star_wind_velocity(n);
    double mdot = single_star_wind_mdot(n, 0);
    force += mdot * v_wind;
    double Lwind = 0.5 * mdot * v_wind * v_wind;
    // estimate the velocity of a wind shell swept up on the scale of a single resolution element in the similarity solution R ~ (L/rho)^1/3 t^(3/5) - use this if slower than v_wind (ie the free-expansion phase is unresolved)
    if(P[n].DensityAroundParticle) {
      v_shell = 0.38 * cbrt(Lwind/(rho*h*h));
    } else {v_shell = MAX_REAL_NUMBER;}
    v_fb = DMAX(v_fb, DMIN(v_shell, v_wind));
#endif
#if defined(RADTRANSFER) || defined(SINGLE_STAR_FB_LOCAL_RP)
    v_shell = 0;
    if(P[n].DensityAroundParticle) {
      force += sink_lum_bol(P[n].Sink_Mdot, P[n].Sink_Mass, n) / C_LIGHT_CODE;
      v_shell = sqrt(0.053 * force / rho) / h; // terminal velocity assuming momentum-conserving solution R ~ (F / rho)^(1/4) t^1/2 = h
    } else {v_shell = 0;}
    v_fb = DMAX(v_fb, v_shell);
#endif
#ifdef SINGLE_STAR_FB_SNE
    if(P[n].ProtoStellarStage == 6) {v_fb = DMAX(v_fb, single_star_SN_velocity(n));}
#endif    
#ifdef RADTRANSFER
    v_fb = DMAX(v_fb, C_LIGHT_CODE_REDUCED(n));  // produces a timestep criterion redundant with the RSOL CFL condition, but can be important if running fancy timestepping hacks that violate CFL under special circumstances
#endif
    if(All.Ti_Current == 0 && RestartFlag == 0) v_fb = DMAX(1e3 / UNIT_VEL_IN_KMS, v_fb); // for idealized box problems
    return v_fb;
}

#endif


double stellar_lifetime_in_Gyr(int n) { //Estimate lifetime of star, using simple MS approximation t/Gyr ~ 9.6 M/L in solar
    double m_solar = P[n].Mass * UNIT_MASS_IN_SOLAR, lum = P[n].StarLuminosity_Solar; // mass in units of Msun
#if (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 2)
    lum = ps_lum_MS(P[n].ZAMS_Mass) * UNIT_LUM_IN_SOLAR; // just calculate the ZAMS luminosity
#endif
    return 9.6*(m_solar/lum) + 0.0034; // gives ~10Gyr for solar-type stars, ~40Myr for 8msun ZAMS, and asymptotes to 3.7 Myr at high mass [assuming those stars are near-Eddington-limited, where the 9.6*(M/L) term becomes -> 0.3 Myr
}


double single_star_relic_SN_mass(int n) { // Estimate mass of the relic that should be left behind after a SNe or other ejecta event
#ifdef SINGLE_STAR_RELICS
    return 1.4 / UNIT_MASS_IN_SOLAR;
#endif
    return 0;
}


#if defined(SINGLE_STAR_FB_SNE)
double single_star_SN_velocity(int n) { // Initial velocity of SNe ejecta: 10^51 erg/SN, distributed evenly among the mass
    return sqrt(1.0 * (2e51/UNIT_ENERGY_IN_CGS)/P[n].Mass_final); //simple v=sqrt(2E/m)should be fine without relativistic corrections
}
#endif


#if (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 2) /* Functions for protosteller evolution model based on Offner 2009 */
/* Calculate the mean ratio of the gas pressure to the gas+radiation pressure, either by solving the Eddington quartic (for n_ad=3, Eq B5) or by using tabulated values, based on Offner 2009, code taken from ORION */
double ps_beta(double m, double n_ad, double rhoc, double Pc) {
    double mass=m*UNIT_MASS_IN_SOLAR;//in units of solar mass
    double Pc_cgs=Pc*All.cf_a3inv*UNIT_PRESSURE_IN_CGS, rhoc_cgs=rhoc*All.cf_a3inv*UNIT_DENSITY_IN_CGS;
    if(n_ad==3.0) {// In this case we solve the Eddington quartic, P_c^3 = (3/a) (k / (mu mH))^4 (1 - beta) / beta^4 rho_c^4 for beta
        int JMAX=40, j; double BETAMIN=1.0e-4, BETAMAX=1.0, TOL=1.0e-7, dx, f, fmid, xmid, rtb, x1=BETAMIN, x2=BETAMAX, coef=3.0/7.56e-15*pow(BOLTZMANN_CGS*rhoc_cgs/(0.613*PROTONMASS_CGS),4);
        f = pow(Pc_cgs,3) - coef * (1.0-x1)/pow(x1,4); fmid = pow(Pc_cgs,3) - coef * (1.0-x2)/pow(x2,4); rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
        for (j=1;j<=JMAX;j++) {
          xmid=rtb+(dx *= 0.5); fmid = pow(Pc_cgs,3) - coef * (1.0-xmid)/pow(xmid,4);
          if(fmid <= 0.0) rtb=xmid;
          if(fabs(dx) < TOL*fabs(xmid) || fmid == 0.0) return rtb;
        }
        printf("ps_beta: bisection solve failed to converge"); return(-1);
    } else {
        // For n != 3, we use a table lookup. The values of beta have been pre-computed with mathematica. The table goes from M=5 to 50 solar masses in steps of 2.5 M_sun, and from n=1.5 to n=3 in steps of 0.5. We should never call this routine with M > 50 Msun, since by then the star should be fully on the main sequence.
        double MTABMIN=5.0, MTABMAX=50.0, MTABSTEP=2.5, NTABMIN=1.5, NTABMAX=3.0, NTABSTEP=0.5;
        //double MBETMIN=0.1; if (mass < MBETMIN){return (1.0+ 0.25*log(mass/MBETMIN)/log(0.01/MBETMIN) );}  // Setting from Offner+Mckee2011, not sure why, does not make much sense above 1, probably to fit to previous results. I made it change continously to avoid big drops in R at 0.1 Msun, value adjusted from 1.15
        if (mass < MTABMIN){return (1.0);}  // Set beta = 1 for M < 5 Msun
        if ((mass >= MTABMAX) || (n_ad >= NTABMAX)) {printf("WARNING: protostellar mass or polytropic index outside of the range of interpolation for beta, m: %g n_ad %g\n",m, n_ad);}
	mass = DMIN(mass, MTABMAX); n_ad = DMIN(n_ad, NTABMAX);
        static double betatab[19][4] = {{0.98785, 0.988928, 0.98947, 0.989634}, {0.97438, 0.976428, 0.977462, 0.977774}, {0.957927, 0.960895, 0.962397, 0.962846},
          {0.939787, 0.943497, 0.945369, 0.945922}, {0.92091, 0.925151, 0.927276, 0.927896}, {0.901932, 0.906512, 0.908785, 0.909436}, {0.883254, 0.888017, 0.890353, 0.891013},
          {0.865111, 0.86994, 0.872277, 0.872927}, {0.847635, 0.852445, 0.854739, 0.855367}, {0.830886, 0.835619, 0.837842, 0.838441}, {0.814885, 0.8195, 0.821635, 0.822201},
          {0.799625, 0.804095, 0.806133, 0.806664}, {0.785082, 0.789394, 0.791328, 0.791825}, {0.771226, 0.775371, 0.777202, 0.777665}, {0.758022, 0.761997, 0.763726, 0.764156},
          {0.745433, 0.749238, 0.750869, 0.751268}, {0.733423, 0.73706, 0.738596, 0.738966}, {0.721954, 0.725429, 0.726874, 0.727216}, {0.710993, 0.714311, 0.715671, 0.715987}};
        // Locate ourselves on the table and do a linear interpolation
        int midx = (int) floor((mass-MTABMIN)/MTABSTEP), nidx = (int) floor((n_ad-NTABMIN)/NTABSTEP);
        double mwgt = (mass-(MTABMIN+midx*MTABSTEP)) / MTABSTEP, nwgt = (n_ad-(NTABMIN+nidx*NTABSTEP)) / NTABSTEP;
        return ( betatab[midx][nidx]*(1.0-mwgt)*(1.0-nwgt) + betatab[midx+1][nidx]*mwgt*(1.0-nwgt) + betatab[midx][nidx+1]*(1.0-mwgt)*nwgt + betatab[midx+1][nidx+1]*mwgt*nwgt );
    }
}
/* Sets the polytropic index for pre burning protostars based on Eq B2 from Offner 2009 */
double ps_polytropic_index_func(double mdot) {
    double mdot_m_solar_per_year = mdot * UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR; // accretion rate in msolar/yr
    //return ( 5.0 - 3/(1.475+0.07*log10(mdot_m_solar_per_year)) );
    if (mdot_m_solar_per_year < 1.0e-5){
        return (2.5);
    }else{
        return(2.5+0.25*log10(mdot_m_solar_per_year*1.0e5));
    }
}
/* Sets the polytropic index for protostars based on Appendix B of Offner 2009 */
double ps_polytropic_index(int stage, double mdot, double mass){
    double n_ad;
    if(mass * UNIT_MASS_IN_SOLAR > 40.){return 3.0;} // force n=3 polytrop for very massive stars even pre-main sequence, because tables for n != 3 don't extrapolate there
    switch(stage) {
        case 0: n_ad = ps_polytropic_index_func(mdot); break;
        case 1: n_ad = ps_polytropic_index_func(mdot); break;
        case 2: n_ad = 1.75; break;
        case 3: n_ad = 1.75; break;
        case 4: n_ad = 3.0; break;
        case 5: n_ad = 3.0; break;
    }
    if(n_ad > 3.0) {n_ad=3.0;}
    if(n_ad < 1.75) {n_ad=1.75;}
    return n_ad;
}
/* Calculate central temperature for protostar by solving Pc = rho_c*kb*Tc/(mu*mH)+1/3*a*Tc^4 using bisection, based on Offner 2009 Eq B14, code taken from ORION */
double ps_Tc(double rhoc, double Pc) {
    int JMAX=40; double TOL=1.0e-7; // max number of iterations, and error tolerance, respectively
    double Pc_cgs=Pc*All.cf_a3inv*UNIT_PRESSURE_IN_CGS, rhoc_cgs=rhoc*All.cf_a3inv*UNIT_DENSITY_IN_CGS, Tgas=Pc_cgs*0.613*PROTONMASS_CGS/(BOLTZMANN_CGS*rhoc_cgs), Trad=pow(3*Pc_cgs/7.56e-15, 0.25), dx, f, fmid, xmid, rtb, x1=0, x2; int j;
    x2 = (Trad > Tgas) ? 2*Trad : Tgas;
    f = Pc_cgs - rhoc_cgs*BOLTZMANN_CGS*x1/(0.613*PROTONMASS_CGS) - 7.56e-15*pow(x1,4)/3.0;
    fmid=Pc_cgs - rhoc_cgs*BOLTZMANN_CGS*x2/(0.613*PROTONMASS_CGS) - 7.56e-15*pow(x2,4)/3.0;
    rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
    for (j=1;j<=JMAX;j++) {
        xmid=rtb+(dx *= 0.5);
        fmid = Pc_cgs - rhoc_cgs*BOLTZMANN_CGS*xmid/(0.613*PROTONMASS_CGS) - 7.56e-15*pow(xmid,4)/3.0;
        if (fmid <= 0.0) rtb=xmid;
        if (fabs(dx) < TOL*fabs(xmid) || fmid == 0.0) return rtb;
    }
    printf("ps_Tc: bisection solve didn't converge, P_c = %e, rho_c = %e, Tgas = %e Trad = %e",Pc_cgs, rhoc_cgs, Tgas, Trad); return(-1);
}
/* Calculate central density for protostar using a pre-computed table for fixed mass, radius and polytropic index, based on Offner 2009, table and code taken from ORION */
double ps_rhoc(double m, double n_ad, double r) { /* Use Tabulated values of rho_c/rho_mean for n=1.5 to 3.1 in intervals of 0.1*/
    static double rhofactab[] = {0.166931, 0.14742, 0.129933, 0.114265, 0.100242, 0.0877, 0.0764968, 0.0665109, 0.0576198, 0.0497216, 0.0427224, 0.0365357, 0.0310837, 0.0262952, 0.0221057, 0.0184553, 0.01529};
    int itab = (int) floor((n_ad-1.5)/0.1); double wgt = (n_ad - (1.5 + 0.1*itab)) / 0.1, rhofac = rhofactab[itab]*(1.0-wgt) + rhofactab[itab+1]*wgt;
    return (m / (4.0/3.0*M_PI*r*r*r) / rhofac);
}
/* Calculate central pressure for protostar using a pre-computed table for fixed mass, radius and polytropic index, based on Offner 2009, table and code taken from ORION */
double ps_Pc(double m, double n_ad, double r) {
    static double pfactab[] = {0.770087, 0.889001, 1.02979, 1.19731, 1.39753, 1.63818, 1.92909, 2.2825, 2.71504, 3.24792, 3.90921, 4.73657, 5.78067, 7.11088, 8.82286, 11.0515, 13.9885};
    int itab = (int) floor((n_ad-1.5)/0.1); double wgt = (n_ad - (1.5 + 0.1*itab)) / 0.1, pfac = pfactab[itab]*(1.0-wgt) + pfactab[itab+1]*wgt;
    return (pfac * All.G * m*m/(r*r*r*r));
}
/* Calculate the mean ratio of the gas pressure to the gas+radiation pressure at the center, based on Offner 2009, code taken from ORION */
double ps_betac(double rhoc, double Pc, double Tc) {
    double Pc_cgs=Pc*All.cf_a3inv*UNIT_PRESSURE_IN_CGS, rhoc_cgs=rhoc*All.cf_a3inv*UNIT_DENSITY_IN_CGS;
    return (rhoc_cgs*BOLTZMANN_CGS*Tc/(0.613*PROTONMASS_CGS) / Pc_cgs);
}
/* Calculate dlog(beta)/dlog(m) by taking a numerical derivative, based on Offner 2009, code taken from ORION */
double ps_dlogbeta_dlogm(double m, double r, double n_ad, double beta, double rhoc, double Pc) {
    double eps=0.01, epspone=1.+eps, rhoc2 = ps_rhoc( m*epspone , n_ad, r), Pc2 = ps_Pc( m*epspone , n_ad, r), beta2 = ps_beta( m*epspone, n_ad, rhoc2, Pc2); //slight imprecision here as we do not update the radius
    return (beta2-beta) / (eps*beta);
}
/* Calculate dlog(beta/betac)/dlog(m) by taking a numerical derivative, based on Offner 2009, code taken from ORION */
double ps_dlogbetaperbetac_dlogm(double m, double r, double n_ad, double beta, double rhoc, double Pc, double Tc) {
    double eps=0.01, epspone=1.+eps, betac = ps_betac(rhoc, Pc, Tc), rhoc2 = ps_rhoc( m*epspone , n_ad, r), Pc2 = ps_Pc( m*epspone , n_ad, r); //slight imprecision here as we do not update the radius
    double Tc2 = ps_Tc(rhoc2, Pc2), beta2 = ps_beta( m*epspone, n_ad, rhoc2, Pc2), betac2 = ps_betac(rhoc2, Pc2, Tc2);
    return (betac*beta2/betac2 - beta) / (eps*beta);
}
/* Calculate the luminosity [in code units] required to ionize the infalling material, based on Offner 2009 */
double ps_lum_I(double mdot) {
    double mdot_m_solar_per_year = mdot * UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR; // accretion rate in msolar/yr
    return (2.5/UNIT_LUM_IN_SOLAR) * ( mdot_m_solar_per_year/(1e-5));
}
/* Calculate the blackbody luminosity [in code units] of the star following the Hayashi track */
double ps_lum_Hayashi_BB(double m, double r) {
    double m_solar = m * UNIT_MASS_IN_SOLAR, T4000_4 = pow(m_solar , 0.55), r_solar = r * UNIT_LENGTH_IN_SOLAR; // protostellar temperature along Hayashi track
    T4000_4 = 0.316406; //ORION prescription T=3000K
    return 0.2263 * r_solar * r_solar * T4000_4 / UNIT_LUM_IN_SOLAR; // luminosity from KH contraction
}
/* Calculate the luminosity of a main sequence star in code units: ORION version: fitting formulas of Tout et al (1996) */
double ps_lum_MS(double m) {
    double m_solar = m * UNIT_MASS_IN_SOLAR;
    if(m_solar <= 0.1) {return 0;} // zero luminosity below some threshold for fits, otherwise use formula below //
    return ((0.39704170*pow(m_solar,5.5) + 8.52762600*pow(m_solar,11)) / (0.00025546+pow(m_solar,3)+5.43288900*pow(m_solar,5) + 5.56357900*pow(m_solar,7) + 0.78866060*pow(m_solar,8)+0.00586685*pow(m_solar,9.5))) / UNIT_LUM_IN_SOLAR;
}
/* Calculate the radius of a main sequence star in solar units: ORION version: fitting formulas of Tout et al (1996) */
double ps_radius_MS_in_solar(double m) {
    double m_solar = m * UNIT_MASS_IN_SOLAR;
    return (1.71535900*pow(m_solar,2.5)+6.59778800*pow(m_solar,6.5)+10.08855000*pow(m_solar,11)+1.01249500*pow(m_solar,19)+0.07490166*pow(m_solar,19.5)) /
        (0.01077422+3.08223400*pow(m_solar,2)+17.84778000*pow(m_solar,8.5)+pow(m_solar,18.5)+0.00022582*pow(m_solar,19.5));
}
#endif // (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION == 2) /* end functions for protosteller evolution model based on Offner 2009 */


#endif //end of protostellar evolution functions


#ifdef SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT
void single_star_SN_init_directions(void){
    /* routine to initialize the distribution of spawned wind particles during SNe. This is essentially a copy of the function rt_init_intensity_directions() in rt_utilities.c */
    int n_polar = SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT;
    if(n_polar < 1) {printf("Number of SN ejecta particles is invalid (<1). Terminating.\n"); endrun(53463431);}
    double mu[n_polar]; int i,j,k,l,n=0,n_oct=n_polar*(n_polar+1)/2;
    double SN_Ejecta_Direction_tmp[n_oct][3];
    for(j=0;j<n_polar;j++) {mu[j] = sqrt( (j + 1./6.) / (n_polar - 1./2.) );}
    for(i=0;i<n_polar;i++)
    {
        for(j=0;j<n_polar-i;j++)
        {
            k=n_polar-1-i-j;
            SN_Ejecta_Direction_tmp[n][0]=mu[i]; SN_Ejecta_Direction_tmp[n][1]=mu[j]; SN_Ejecta_Direction_tmp[n][2]=mu[k];
            n++;
        }
    }
    n=0;
    for(i=0;i<2;i++)
    {
        double sign_x = 1 - 2*i;
        for(j=0;j<2;j++)
        {
            double sign_y = 1 - 2*j;
            for(k=0;k<2;k++)
            {
                double sign_z = 1 - 2*k;
                for(l=0;l<n_oct;l++)
                {
                    All.SN_Ejecta_Direction[n][0] = SN_Ejecta_Direction_tmp[l][0] * sign_x;
                    All.SN_Ejecta_Direction[n][1] = SN_Ejecta_Direction_tmp[l][1] * sign_y;
                    All.SN_Ejecta_Direction[n][2] = SN_Ejecta_Direction_tmp[l][2] * sign_z;
                    n++;
                }
            }
        }
    }
}
#endif






#endif /* GALSF */
