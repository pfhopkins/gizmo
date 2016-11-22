#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../allvars.h"
#include "../proto.h"

/*!
 *  routines for star formation in cosmological simulations
 */
/*
 * This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 *   It was based on a similar file in GADGET3 by Volker Springel (volker.springel@h-its.org),
 *   but the physical modules for star formation and feedback have been
 *   replaced, and the algorithm is mostly new to GIZMO.
 */


#ifdef GALSF // master switch for compiling the routines below //

#define WindInitialVelocityBoost 1.0 // (optional) boost velocity coupled (fixed momentum)

#if defined(GALSF_SFR_IMF_VARIATION) || defined(GALSF_SFR_IMF_SAMPLING)
/* function to determine what the IMF of a new star particle will be, based 
    on the gas properties of the particle out of which it forms */
void assign_imf_properties_from_starforming_gas(int i)
{
    
#ifdef GALSF_SFR_IMF_VARIATION
    double h = Get_Particle_Size(i) * All.cf_atime;
    double cs = Particle_effective_soundspeed_i(i) * All.cf_afac3; // actual sound speed in the simulation: might be unphysically high for SF conditions!
    cs = (1.9e4 / All.UnitVelocity_in_cm_per_s); // set to a minimum cooling temperature, for the actual star-forming conditions. for now, just use a constant //
    double dv2_abs = 0; /* calculate local velocity dispersion (including hubble-flow correction) in physical units */
    // squared norm of the trace-free symmetric [shear] component of the velocity gradient tensor //
    dv2_abs = ((1./2.)*((SphP[i].Gradients.Velocity[1][0]+SphP[i].Gradients.Velocity[0][1])*(SphP[i].Gradients.Velocity[1][0]+SphP[i].Gradients.Velocity[0][1]) +
                        (SphP[i].Gradients.Velocity[2][0]+SphP[i].Gradients.Velocity[0][2])*(SphP[i].Gradients.Velocity[2][0]+SphP[i].Gradients.Velocity[0][2]) +
                        (SphP[i].Gradients.Velocity[2][1]+SphP[i].Gradients.Velocity[1][2])*(SphP[i].Gradients.Velocity[2][1]+SphP[i].Gradients.Velocity[1][2])) +
               (2./3.)*((SphP[i].Gradients.Velocity[0][0]*SphP[i].Gradients.Velocity[0][0] +
                         SphP[i].Gradients.Velocity[1][1]*SphP[i].Gradients.Velocity[1][1] +
                         SphP[i].Gradients.Velocity[2][2]*SphP[i].Gradients.Velocity[2][2]) -
                        (SphP[i].Gradients.Velocity[1][1]*SphP[i].Gradients.Velocity[2][2] +
                         SphP[i].Gradients.Velocity[0][0]*SphP[i].Gradients.Velocity[1][1] +
                         SphP[i].Gradients.Velocity[0][0]*SphP[i].Gradients.Velocity[2][2]))) * All.cf_a2inv*All.cf_a2inv;
    double M_sonic = cs*cs*cs*cs / (All.G * dv2_abs * h);
    M_sonic *= All.UnitMass_in_g / All.HubbleParam / (1.989e33); // sonic mass in solar units //
    P[i].IMF_Mturnover = DMAX(0.01,DMIN(M_sonic,100.));
    P[i].IMF_Mturnover = 2.0; // 'normal' IMF in our definitions
    
    
    /* now we need to record all the properties we care to save about the star-forming gas, for the sake of later use: */
    int j,k;
    double NH = evaluate_NH_from_GradRho(P[i].GradRho,PPP[i].Hsml,SphP[i].Density,PPP[i].NumNgb,1);
    double dv2abs_tot = 0; /* calculate complete velocity dispersion (including hubble-flow correction) in physical units */
    for(j=0;j<3;j++)
    {
        for(k=0;k<3;k++)
        {
            double vt = SphP[i].Gradients.Velocity[j][k]*All.cf_a2inv; /* physical velocity gradient */
            if(All.ComovingIntegrationOn) {if(j==k) {vt += All.cf_hubble_a;}} /* add hubble-flow correction */
            dv2abs_tot += vt*vt;
        }
    }
    double acc=0,vel=0;
    for(k=0;k<3;k++)
    {
        double acc_tmp = P[i].GravAccel[k];
#ifdef PMGRID
        acc_tmp += P[i].GravPM[k];
#endif
        acc_tmp *= All.cf_a2inv;
        acc += acc_tmp * acc_tmp;
        vel += SphP[i].VelPred[k]*SphP[i].VelPred[k];
    }
    double b_mag = 0;
#ifdef MAGNETIC
    double gizmo2gauss = 4.*M_PI*All.UnitPressure_in_cgs*All.HubbleParam*All.HubbleParam;
    for(k=0;k<3;k++) {b_mag += Get_Particle_BField(i,k)*Get_Particle_BField(i,k) * gizmo2gauss;}
#endif
    double rad_flux_uv = 1;
#ifdef GALSF_FB_LOCAL_UV_HEATING
    rad_flux_uv = SphP[i].RadFluxUV;
#endif
    double cr_energy_density = 0;
#ifdef COSMIC_RAYS
    cr_energy_density = SphP[i].CosmicRayEnergyPred * SphP[i].Density * All.cf_a3inv / P[i].Mass;
#endif

    P[i].IMF_FormProps[0] = P[i].IMF_Mturnover; // IMF turnover mass as defined above
    P[i].IMF_FormProps[1] = SphP[i].Density * All.cf_a3inv; // density
    P[i].IMF_FormProps[2] = SphP[i].InternalEnergyPred; // thermal internal energy (use to calculate temperature)
    P[i].IMF_FormProps[3] = Particle_effective_soundspeed_i(i) * All.cf_afac3; // sound speed (not trivially related to temperature if CRs, etc included)
    P[i].IMF_FormProps[4] = sqrt(dv2_abs); // shear velocity gradient (norm of shear gradient tensor)
    P[i].IMF_FormProps[5] = h; // particle length/size (inter-particle spacing)
    P[i].IMF_FormProps[6] = NH; // local gas surface density (our usual estimator) in the cloud where the particle formed
    P[i].IMF_FormProps[7] = sqrt(dv2abs_tot) * h; // total rms/turbulent velocity dispersion
    P[i].IMF_FormProps[8] = sqrt(acc); // gravitational acceleration
    P[i].IMF_FormProps[9] = sqrt(vel); // total velocity (use with acceleration to estimate shear omega, etc)
    P[i].IMF_FormProps[10] = sqrt(b_mag) * All.cf_a2inv; // magnetic field strength |B|
    P[i].IMF_FormProps[11] = rad_flux_uv; // incident UV flux normalized to MW 'canonical' (Habing) field value
    P[i].IMF_FormProps[12] = cr_energy_density; // cosmic ray energy density (if CRs are enabled)
    
#endif 
    
    
#ifdef GALSF_SFR_IMF_SAMPLING
    gsl_rng *random_generator_for_massivestars;
    random_generator_for_massivestars = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(random_generator_for_massivestars, P[i].ID+121);
    double mu = 0.01 * P[i].Mass * All.UnitMass_in_g / All.HubbleParam / (1.989e33); // 1 O-star per 100 Msun
    unsigned int kk = gsl_ran_poisson(random_generator_for_massivestars, mu);
    P[i].IMF_NumMassiveStars = (double)kk;
#endif
    
}
#endif


/* return the light-to-mass ratio, for the IMF of a given particle, relative to the Chabrier/Kroupa IMF which 
    is otherwise (for all purposes) our 'default' choice */
double calculate_relative_light_to_mass_ratio_from_imf(int i)
{
#ifdef SINGLE_STAR_FORMATION
    double unit_lsun_msun = (All.UnitEnergy_in_cgs / (All.UnitTime_in_s * SOLAR_LUM)) / (All.UnitMass_in_g / (All.HubbleParam * SOLAR_MASS));
    return bh_lum_bol(0, P[i].Mass, i) / P[i].Mass * unit_lsun_msun;
#endif
#ifdef GALSF_SFR_IMF_VARIATION
    /* more accurate version from David Guszejnov's IMF calculations (ok for Mturnover in range 0.01-100) */
    double log_mimf = log10(P[i].IMF_Mturnover);
    return (0.051+0.042*(log_mimf+2)+0.031*(log_mimf+2)*(log_mimf+2)) / 0.31;
    // return pow(P[i].IMF_Mturnover/1.0,0.35);
#endif
#ifdef GALSF_SFR_IMF_SAMPLING
    double mu = 0.01 * P[i].Mass * All.UnitMass_in_g / All.HubbleParam / (1.989e33); // 1 O-star per 100 Msun
    double age = evaluate_stellar_age_Gyr(P[i].StellarAge);
    if(age > 0.003) {mu *= 0.326 * (0.003 / age);} // expectation value is declining with time, so 'effective multiplier' is larger
    return P[i].IMF_NumMassiveStars / mu;
#endif
    return 1; // Chabrier or Kroupa IMF //
    // return 0.5; // Salpeter IMF down to 0.1 solar //
}


/* return the stellar age in Gyr for a given labeled age, needed throughout for stellar feedback */
double evaluate_stellar_age_Gyr(double stellar_tform)
{
    double age,a0,a1,a2,x0,x1,x2;
    if(All.ComovingIntegrationOn)
    {
        a0 = stellar_tform;
        a2 = All.Time;
        if(fabs(1-(All.Omega0+All.OmegaLambda))<=0.01)
        {
            /* use exact solution for flat universe */
            x0 = (All.Omega0/(1-All.Omega0))/(a0*a0*a0);
            x2 = (All.Omega0/(1-All.Omega0))/(a2*a2*a2);
            age = (2./(3.*sqrt(1-All.Omega0)))*log(sqrt(x0*x2)/((sqrt(1+x2)-1)*(sqrt(1+x0)+1)));
            age *= 1./All.Hubble_H0_CodeUnits;
        } else {
            /* use simple trap rule integration */
            a1 = 0.5*(a0+a2);
            x0 = 1./(a0*hubble_function(a0));
            x1 = 1./(a1*hubble_function(a1));
            x2 = 1./(a2*hubble_function(a2));
            age = (a2-a0)*(x0+4.*x1+x2)/6.;
        }
    } else {
        /* time variable is simple time, when not in comoving coordinates */
        age=All.Time-stellar_tform;
    }
    age *= 0.001*All.UnitTime_in_Megayears/All.HubbleParam; // convert to absolute Gyr
    if((age<=1.e-5)||(isnan(age))) {age=1.e-5;}
    return age;
}


/* return the (solar-scaled) light-to-mass ratio of an SSP with a given age; used throughout */
double evaluate_l_over_m_ssp(double stellar_age_in_gyr)
{
#ifdef SINGLE_STAR_FORMATION
    return 1;
#endif
    // original SB99 tracks
    /*
    if(stellar_age_in_gyr < 0.0029)
    {
        return 1136.59;
    } else {
        double log_age = log10(stellar_age_in_gyr)-(-2.2681);
        return 478.63*pow(10.,-1.3625*log_age+0.115765*log_age*log_age);
        // could replace with piecewise linear functions; if this call in forcetree gets expensive //
    }
    */
    // updated SB99 tracks: including rotation, new mass-loss tracks, etc.
    if(stellar_age_in_gyr < 0.0035)
    {
        return 1136.59;
    } else {
        double log_age = log10(stellar_age_in_gyr/0.0035);
        return 1500.*pow(10.,-1.8*log_age+0.3*log_age*log_age-0.025*log_age*log_age*log_age);
    }
    return 0;
}





/* simple routine to determine density thresholds and other common units for SF routines */
void set_units_sfr(void)
{
    All.OverDensThresh = All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G);
    All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / (HYDROGEN_MASSFRAC * All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam);
    
#ifdef GALSF_EFFECTIVE_EQS
    double meanweight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: assuming NEUTRAL GAS */
    All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
    All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */
    All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
    All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
#endif // GALSF_EFFECTIVE_EQS
}


/* Routine to actually determine the SFR assigned to an individual gas particle at each time */
double get_starformation_rate(int i)
{
    double rateOfSF,tsfr,y; y=0;
    int flag;
#ifdef GALSF_EFFECTIVE_EQS
    double factorEVP, egyhot, ne, tcool, x, cloudmass;
#endif
#ifdef GALSF_SUBGRID_WINDS
    if(SphP[i].DelayTime > 0)
    return 0;
#endif
    
#ifdef BH_WIND_SPAWN
    if(P[i].ID == All.AGNWindID) return 0;
#endif

    flag = 1;			/* default is normal cooling */
    if(SphP[i].Density*All.cf_a3inv >= All.PhysDensThresh)
    flag = 0;
    if(All.ComovingIntegrationOn)
    if(SphP[i].Density < All.OverDensThresh)
    flag = 1;
    if((flag == 1)||(P[i].Mass<=0))
    return 0;
    
    
    tsfr = sqrt(All.PhysDensThresh / (SphP[i].Density * All.cf_a3inv)) * All.MaxSfrTimescale;
    if(tsfr<=0) return 0;
    
#ifndef GALSF_EFFECTIVE_EQS
    /* 'normal' sfr from density law above */
    rateOfSF = P[i].Mass / tsfr;
#else
    factorEVP = pow(SphP[i].Density * All.cf_a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;
    egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
    ne = SphP[i].Ne;
    tcool = GetCoolingTime(egyhot, SphP[i].Density * All.cf_a3inv, &ne, i);
    y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
    cloudmass = x * P[i].Mass;
    rateOfSF = (1 - All.FactorSN) * cloudmass / tsfr;
    
    update_internalenergy_for_galsf_effective_eos(i,tcool,tsfr,x,rateOfSF); // updates entropies for the effective equation-of-state //
#endif // GALSF_EFFECTIVE_EQS
    
    
#ifdef GALSF_SFR_MOLECULAR_CRITERION
    /* Krumholz & Gnedin fitting function for f_H2 as a function of local properties */
    double tau_fmol = evaluate_NH_from_GradRho(P[i].GradRho,PPP[i].Hsml,SphP[i].Density,PPP[i].NumNgb,1);
    tau_fmol *= (0.1 + P[i].Metallicity[0]/All.SolarAbundances[0]);
    if(tau_fmol>0) {
        tau_fmol *= 434.78 * All.UnitDensity_in_cgs * All.UnitLength_in_cm * All.HubbleParam;
        y = 0.756 * (1 + 3.1*pow(P[i].Metallicity[0]/All.SolarAbundances[0],0.365));
        y = log(1 + 0.6*y + 0.01*y*y) / (0.6*tau_fmol);
        y = 1 - 0.75*y/(1 + 0.25*y);
        if(y<0) y=0; if(y>1) y=1;
        rateOfSF *= y;
    } // if(tau_fmol>0)
#endif // GALSF_SFR_MOLECULAR_CRITERION
    
    
#ifdef GALSF_SFR_VIRIAL_SF_CRITERION
    double dv2abs = 0; /* calculate local velocity dispersion (including hubble-flow correction) in physical units */
    int j,k;
    for(j=0;j<3;j++)
        for(k=0;k<3;k++)
        {
            double vt = SphP[i].Gradients.Velocity[j][k]*All.cf_a2inv; /* physical velocity gradient */
            if(All.ComovingIntegrationOn)
                if(j==k)
                    vt += All.cf_hubble_a; /* add hubble-flow correction */
            dv2abs += vt*vt;
        }
    /* add thermal support, although it is almost always irrelevant */
    double cs_eff = Particle_effective_soundspeed_i(i);
    double k_cs = cs_eff / (Get_Particle_Size(i)*All.cf_atime);
#ifdef SINGLE_STAR_FORMATION
    double press_grad_length = 0;
    for(j=0;j<3;j++) {press_grad_length += SphP[i].Gradients.Pressure[k]*SphP[i].Gradients.Pressure[k];}
    press_grad_length = All.cf_atime * DMAX(Get_Particle_Size(i) , SphP[i].Pressure / (1.e-37 + sqrt(press_grad_length))); 
    k_cs = cs_eff / press_grad_length;
#ifdef MAGNETIC
    double bmag=0; for(k=0;k<3;k++) {bmag+=Get_Particle_BField(i,k)*Get_Particle_BField(i,k);}
    double cs_b = sqrt(cs_eff*cs_eff + bmag/SphP[i].Density);
    k_cs = cs_b / (Get_Particle_Size(i)*All.cf_atime);
#endif
#endif
#if !defined(COSMIC_RAYS)
    dv2abs += 2.*k_cs*k_cs; // account for thermal pressure with standard Jeans criterion (k^2*cs^2 vs 4pi*G*rho) //
#endif
    
    //double alpha_vir = 0.2387 * dv2abs / (All.G * SphP[i].Density * All.cf_a3inv); // coefficient here was for old form, with only divv information
    double alpha_vir = dv2abs / (8. * M_PI * All.G * SphP[i].Density * All.cf_a3inv); // 1/4 or 1/8 ? //

    
#if !(EXPAND_PREPROCESSOR_(GALSF_SFR_VIRIAL_SF_CRITERION) == 1)
    /* the above macro checks if GALSF_SFR_VIRIAL_SF_CRITERION has been assigned a numerical value */
#if (GALSF_SFR_VIRIAL_SF_CRITERION > 0) || (GALSF_SFR_VIRIAL_SF_CRITERION == 2)
    if(alpha_vir < 1.0)
    {
        /* check if Jeans mass is remotely close to solar; if not, dont allow it to form 'stars' */
        double q = cs_eff * All.UnitVelocity_in_cm_per_s / (0.2e5);
        double q2 = SphP[i].Density * All.cf_a3inv * All.UnitDensity_in_cgs * All.HubbleParam*All.HubbleParam / (HYDROGEN_MASSFRAC*1.0e3*PROTONMASS);
        double MJ_solar = 2.*q*q*q/sqrt(q2);
        double MJ_crit = 1000.;
#ifdef SINGLE_STAR_FORMATION
        MJ_crit = 1.e4;
#endif
        if(MJ_solar > MJ_crit) {alpha_vir = 100.;}
    }
#endif
#if (GALSF_SFR_VIRIAL_SF_CRITERION > 1)
    if(alpha_vir >= 1.0) {rateOfSF *= 0.0;}
#endif
#endif
    if((alpha_vir<1.0)||(SphP[i].Density*All.cf_a3inv>100.*All.PhysDensThresh)) {rateOfSF *= 1.0;} else {rateOfSF *= 0.0015;}
    // PFH: note the latter flag is an arbitrary choice currently set -by hand- to prevent runaway densities from this prescription! //
    
    //  if( divv>=0 ) rateOfSF=0; // restrict to convergent flows (optional) //
    //  rateOfSF *= 1.0/(1.0 + alpha_vir); // continuous cutoff w alpha_vir instead of sharp (optional) //
#endif // GALSF_SFR_VIRIAL_SF_CRITERION
    
#ifdef SINGLE_STAR_FORMATION
    rateOfSF *= 1.0e5; // make sink formation guaranteed to happen, where it can
    // restrict to convergent flows //
    {
        int k; double divv=0; for(k=0;k<3;k++) {divv += SphP[i].Gradients.Velocity[k][k] * All.cf_a2inv;}
        if(All.ComovingIntegrationOn) {divv += 3.*All.cf_hubble_a;}
        if(divv >= 0) {rateOfSF=0;} 
    }
    if(SphP[i].Density_Relative_Maximum_in_Kernel > 0) {rateOfSF=0;} // restrict to local density/potential maxima //
#ifdef BH_CALC_DISTANCES
    if(P[i].min_dist_to_bh < PPP[i].Hsml) {rateOfSF=0;} // restrict to particles without a sink in their kernel //
#endif
#endif // SINGLE_STAR_FORMATION 
    
    return rateOfSF;
}



#ifdef GALSF_EFFECTIVE_EQS
/* compute the 'effective eos' cooling/heating, including thermal feedback sources, here */
void update_internalenergy_for_galsf_effective_eos(int i, double tcool, double tsfr, double x, double rateOfSF)
{
    double dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
    double dtime = dt / All.cf_hubble_a; /*  the actual time-step */
    double factorEVP = pow(SphP[i].Density * All.cf_a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;
    double trelax = tsfr * (1 - x) / x / (All.FactorSN * (1 + factorEVP));
    double egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
    double egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
    double egycurrent = SphP[i].InternalEnergy;

#if defined(BH_THERMALFEEDBACK)
    if((SphP[i].Injected_BH_Energy > 0) && (P[i].Mass>0))
    {
        egycurrent += SphP[i].Injected_BH_Energy / P[i].Mass;
        if(egycurrent > egyeff)
        {
            tcool = GetCoolingTime(egycurrent, SphP[i].Density * All.cf_a3inv, &ne, i);
            if(tcool < trelax && tcool > 0) trelax = tcool;
        }
        SphP[i].Injected_BH_Energy = 0;
    }
#endif // defined(BH_THERMALFEEDBACK)

    /* now update the thermal variables */
    SphP[i].InternalEnergy = (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax));
    SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
    SphP[i].Pressure = get_pressure(i);
    //SphP[i].dInternalEnergy = 0;
    SphP[i].DtInternalEnergy = 0; /* HERE, it's ok, b/c effective EOS is designed to model new pressure even under compressions, 
                                 (since we're zero'ing the second-half-step from the hydro step) */
}
#endif // GALSF_EFFECTIVE_EQS //




/* master routine for star formation. for 'effective equation of state' models for star-forming gas, this also updates their effective EOS parameters */
void star_formation_parent_routine(void)
{
  int i, bin, flag, stars_spawned, tot_spawned, stars_converted, tot_converted, number_of_stars_generated;
  unsigned int bits;
  double dtime, mass_of_star, p, prob, rate_in_msunperyear, sfrrate, totsfrrate;
  double sum_sm, total_sm, sm=0, rate, sum_mass_stars, total_sum_mass_stars;
#if defined(BH_POPIII_SEEDS) || defined(SINGLE_STAR_FORMATION)
  int num_bhformed=0, tot_bhformed=0;
  double GradRho;
  GradRho=0;
#endif
#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) && defined(GALSF_FB_RPWIND_LOCAL)
  double total_n_wind,total_m_wind,total_mom_wind,total_prob_kick,avg_v_kick,momwt_avg_v_kick,avg_taufac;
  double totMPI_n_wind,totMPI_m_wind,totMPI_mom_wind,totMPI_prob_kick,totMPI_avg_v,totMPI_pwt_avg_v,totMPI_taufac;
  total_n_wind=total_m_wind=total_mom_wind=total_prob_kick=avg_v_kick=momwt_avg_v_kick=avg_taufac=0;
  totMPI_n_wind=totMPI_m_wind=totMPI_mom_wind=totMPI_prob_kick=totMPI_avg_v=totMPI_pwt_avg_v=totMPI_taufac=0;
#endif
    
    for(bin = 0; bin < TIMEBINS; bin++) {if(TimeBinActive[bin]) {TimeBinSfr[bin] = 0;}}
  stars_spawned = stars_converted = 0; sum_sm = sum_mass_stars = 0;

  for(bits = 0; GALSF_GENERATIONS > (1 << bits); bits++);

  for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
      if((P[i].Type == 0)&&(P[i].Mass>0))
	{
        SphP[i].Sfr = 0; flag = 1; /* will be reset below if flag==0, but default to flag = 1 (non-eligible) */
        dtime = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a; /*  the actual time-step */
        
        /* check whether an initial (not fully-complete!) conditions for star formation are fulfilled for a given particle */
        if(SphP[i].Density * All.cf_a3inv >= All.PhysDensThresh) {flag = 0;} // if sufficiently dense, go forward into SF routine //
        if(All.ComovingIntegrationOn) {if(SphP[i].Density < All.OverDensThresh) flag = 1;} // (additional density check for cosmological runs) //

#ifdef GALSF_SUBGRID_WINDS
        if(SphP[i].DelayTime > 0) {flag=1; SphP[i].DelayTime -= dtime;} /* no star formation for particles in the wind; update our wind delay-time calculations */
        if((SphP[i].DelayTime<0) || (SphP[i].Density*All.cf_a3inv < All.WindFreeTravelDensFac*All.PhysDensThresh)) {SphP[i].DelayTime=0;}
#endif
#ifdef GALSF_TURNOFF_COOLING_WINDS
        if(SphP[i].DelayTimeCoolingSNe > 0) {flag=1; SphP[i].DelayTimeCoolingSNe -= dtime;} /* no star formation for particles in the wind; update our wind delay-time calculations */
#endif
        
        
    if((flag == 0)&&(dtime>0)&&(P[i].TimeBin))		/* active star formation (upon start-up, we need to protect against dt==0) */
	    {
          sm = get_starformation_rate(i) * dtime; // expected stellar mass formed this timestep
            // (this also updates entropies for the effective equation-of-state model) //
	      p = sm / P[i].Mass;
	      sum_sm += P[i].Mass * (1 - exp(-p));
            

        /* Alright, now we consider the actual gas-to-star particle conversion and associated steps */

	      /* the upper bits of the gas particle ID store how many stars this gas particle gas already generated */
	      if(bits == 0)
            number_of_stars_generated = 0;
	      else
            number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType) * 8 - bits));

	      mass_of_star = P[i].Mass / (GALSF_GENERATIONS - number_of_stars_generated);
            if(number_of_stars_generated >= GALSF_GENERATIONS-1) mass_of_star=P[i].Mass;

          SphP[i].Sfr = sm / dtime *
            (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
	      if(dtime>0) TimeBinSfr[P[i].TimeBin] += SphP[i].Sfr;

          prob = P[i].Mass / mass_of_star * (1 - exp(-p));
        
#if defined(METALS) && defined(GALSF_EFFECTIVE_EQS) // does instantaneous enrichment //
            double w = get_random_number(P[i].ID);
            P[i].Metallicity[0] += w * All.SolarAbundances[0] * (1 - exp(-p));
            if(NUM_METAL_SPECIES>=10)
            {
                int k;
                for(k=1;k<NUM_METAL_SPECIES;k++) {P[i].Metallicity[k] += w * All.SolarAbundances[k] * (1 - exp(-p));}
            }
#endif
            
        if(get_random_number(P[i].ID + 1) < prob)	/* ok, make a star */
		{

#ifdef BH_POPIII_SEEDS
            /* before making a star, assess whether or not we can instead make a BH seed particle */
            p=0;
            if ( (SphP[i].Density*All.cf_a3inv > All.PhysDensThresh) && (P[i].Metallicity[0]/All.SolarAbundances[0] < 0.1) )
            {
                GradRho = evaluate_NH_from_GradRho(P[i].GradRho,PPP[i].Hsml,SphP[i].Density,PPP[i].NumNgb,1);
                GradRho *= All.UnitDensity_in_cgs * All.UnitLength_in_cm * All.HubbleParam;
                /* surface dens in g/cm^2; threshold for bound cluster formation in our experiments is ~2 g/cm^2 (10^4 M_sun/pc^2) */
                if (GradRho > 0.1)
                {
                    /* now calculate probability of forming a BH seed particle */
                    p = 0.0004; /* ratio of BH mass formed to stellar mass for Z~0.01 Zsun population */
                    p *= (P[i].Mass / All.SeedBlackHoleMass); /* resolves resolution-dependence by making p=massfrac */
                    p *= (1-exp(-GradRho/1.0)) * exp(-(P[i].Metallicity[0]/All.SolarAbundances[0])/0.01);
                    /* want to add factors to control this probability in zoom-in runs */
                } // (y>2)
            } // (above density threshold and below metallicity threshold)
            if(get_random_number(P[i].ID + 2) < p)
            {
                /* make a BH particle */
                P[i].Type = 5;
                TimeBinCountSph[P[i].TimeBin]--;
                num_bhformed++;
                Stars_converted++;
                stars_converted++;
                P[i].StellarAge = All.Time;
                P[i].BH_Mass = All.SeedBlackHoleMass;
                if(p>1) P[i].BH_Mass *= p; /* assume multiple seeds in particle merge */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                P[i].Mass = SphP[i].MassTrue + SphP[i].dMass;
#endif
#ifdef BH_ALPHADISK_ACCRETION
                P[i].BH_Mass_AlphaDisk = 0;
#endif
#ifdef BH_COUNTPROGS
                P[i].BH_CountProgs = 1;
#endif
                P[i].BH_Mdot = 0;
#ifdef BH_PHOTONMOMENTUM
                P[i].BH_disk_hr = 0.333333;
#endif
                P[i].DensAroundStar = SphP[i].Density;
#ifdef BH_BUBBLES
                P[i].BH_Mass_bubbles = All.SeedBlackHoleMass;
                P[i].BH_Mass_ini = All.SeedBlackHoleMass;
#endif
#ifdef UNIFIED_FEEDBACK
                P[i].BH_Mass_radio = All.SeedBlackHoleMass;
#endif
            } else {
#endif /* closes ifdef(BH_POPIII_SEEDS) */ 

            /* ok, we're going to make a star! */
#if defined(GALSF_SFR_IMF_VARIATION) || defined(GALSF_SFR_IMF_SAMPLING)
            /* if we're allowing for a variable IMF, this is where we will 
                calculate the IMF properties produced from the gas forming stars */
            assign_imf_properties_from_starforming_gas(i);
#endif
                
            if(number_of_stars_generated == (GALSF_GENERATIONS - 1))
		    {
		      /* here we turn the gas particle itself into a star */
		      Stars_converted++;
		      stars_converted++;
		      sum_mass_stars += P[i].Mass;

		      P[i].Type = 4;
		      TimeBinCountSph[P[i].TimeBin]--;
		      TimeBinSfr[P[i].TimeBin] -= SphP[i].Sfr;

		      P[i].StellarAge = All.Time;
#ifdef DO_DENSITY_AROUND_STAR_PARTICLES
                P[i].DensAroundStar = SphP[i].Density;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                P[i].Mass = SphP[i].MassTrue + SphP[i].dMass;
#endif
                

#ifdef SINGLE_STAR_FORMATION
                P[i].Type = 5;
                num_bhformed++;
                P[i].BH_Mass = All.SeedBlackHoleMass;
#ifdef BH_ALPHADISK_ACCRETION
                P[i].BH_Mass_AlphaDisk = DMAX(0, P[i].Mass-P[i].BH_Mass);
#endif
#ifdef BH_COUNTPROGS
                P[i].BH_CountProgs = 1;
#endif
                P[i].BH_Mdot = 0;
#ifdef BH_PHOTONMOMENTUM
                P[i].BH_disk_hr = 0.333333;
#endif
                P[i].DensAroundStar = SphP[i].Density;
#ifdef SINGLE_STAR_PROMOTION
                P[i].ProtoStellarAge = All.Time; // record the proto-stellar age instead of age
                P[i].ProtoStellar_Radius = 100. * (P[i].Mass * All.UnitMass_in_g / All.HubbleParam / SOLAR_MASS);
                //P[i].PreMainSeq_Tracker = 1;
#endif
#endif // SINGLE_STAR_FORMATION
                
		    } /* closes final generation from original gas particle */
		  else
		    {
		      /* here we spawn a new star particle */

		      if(NumPart + stars_spawned >= All.MaxPart)
			{
			  printf
			    ("On Task=%d with NumPart=%d we try to spawn %d particles. Sorry, no space left...(All.MaxPart=%d)\n",
			     ThisTask, NumPart, stars_spawned, All.MaxPart);
			  fflush(stdout);
			  endrun(8888);
			}

		      P[NumPart + stars_spawned] = P[i];
		      P[NumPart + stars_spawned].Type = 4;
#ifdef DO_DENSITY_AROUND_STAR_PARTICLES
              P[NumPart + stars_spawned].DensAroundStar = SphP[i].Density;
#endif
		      NextActiveParticle[NumPart + stars_spawned] = FirstActiveParticle;
		      FirstActiveParticle = NumPart + stars_spawned;
		      NumForceUpdate++;

		      TimeBinCount[P[NumPart + stars_spawned].TimeBin]++;
		      PrevInTimeBin[NumPart + stars_spawned] = i;
		      NextInTimeBin[NumPart + stars_spawned] = NextInTimeBin[i];
		      if(NextInTimeBin[i] >= 0)
                  PrevInTimeBin[NextInTimeBin[i]] = NumPart + stars_spawned;
		      NextInTimeBin[i] = NumPart + stars_spawned;
		      if(LastInTimeBin[P[i].TimeBin] == i)
                  LastInTimeBin[P[i].TimeBin] = NumPart + stars_spawned;

		      P[i].ID += ((MyIDType) 1 << (sizeof(MyIDType) * 8 - bits));

		      P[NumPart + stars_spawned].Mass = mass_of_star;
		      P[i].Mass -= P[NumPart + stars_spawned].Mass;
              if(P[i].Mass<0) P[i].Mass=0;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
              SphP[i].MassTrue -= P[NumPart + stars_spawned].Mass;
              if(SphP[i].MassTrue<0) SphP[i].MassTrue=0;
#endif
		      sum_mass_stars += P[NumPart + stars_spawned].Mass;
		      P[NumPart + stars_spawned].StellarAge = All.Time;
		      force_add_star_to_tree(i, NumPart + stars_spawned);

		      stars_spawned++;
		    }
#ifdef BH_POPIII_SEEDS
            } /* closes else for decision to make a BH particle */
#endif
		}

#if defined(METALS) && defined(GALSF_EFFECTIVE_EQS) // does instantaneous enrichment //
	    if(P[i].Type == 0)	/* to protect using a particle that has been turned into a star */
        {
            P[i].Metallicity[0] += (1 - w) * All.SolarAbundances[0] * (1 - exp(-p));
            if(NUM_METAL_SPECIES>=10)
            {
                int k;
                for(k=1;k<NUM_METAL_SPECIES;k++) {P[i].Metallicity[k] += (1-w) * All.SolarAbundances[k] * (1 - exp(-p));}
            }
        }
#endif
        } // closes check of flag==0 for star-formation operation

#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) || defined(GALSF_SUBGRID_WINDS) || defined(GALSF_SUBGRID_VARIABLEVELOCITY) || defined(GALSF_SUBGRID_VARIABLEVELOCITY_DM_DISPSERSION)
        if( (flag==0 || All.ComovingIntegrationOn==0) &&
           (P[i].Mass>0) && (P[i].Type==0) && (dtime>0) && (All.Time>0) )
        {
            double pvtau_return[4];
            assign_wind_kick_from_sf_routine(i,sm,dtime,pvtau_return);
#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) && defined(GALSF_FB_RPWIND_LOCAL) // values tabulated below purely for bookkeeping //
            if(pvtau_return[0]>0)
            {
                total_n_wind+=pvtau_return[0];
                total_m_wind+=P[i].Mass;
                total_mom_wind+=P[i].Mass*pvtau_return[2];
                total_prob_kick+=pvtau_return[1]/dtime;
                avg_v_kick+=pvtau_return[2];
                momwt_avg_v_kick+=pvtau_return[1]*pvtau_return[2]/dtime;
                avg_taufac+=log(pvtau_return[3]);
            }
#endif
        }
#endif

	} /* End of If Type = 0 */
    } /* end of main loop over active particles, huzzah! */



#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) && defined(GALSF_FB_RPWIND_LOCAL)
if(All.WindMomentumLoading)
{
    MPI_Reduce(&total_n_wind, &totMPI_n_wind, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_m_wind, &totMPI_m_wind, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_mom_wind, &totMPI_mom_wind, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_prob_kick, &totMPI_prob_kick, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_v_kick, &totMPI_avg_v, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&momwt_avg_v_kick, &totMPI_pwt_avg_v, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_taufac, &totMPI_taufac, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#ifdef IO_REDUCED_MODE
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
    if(ThisTask == 0)
    {
        if(totMPI_n_wind>0)
        {
            totMPI_avg_v /= totMPI_n_wind;
            totMPI_pwt_avg_v /= totMPI_prob_kick;
            totMPI_taufac /= totMPI_n_wind;
            totMPI_taufac = exp(totMPI_taufac);
            printf("Momentum Wind Feedback: Time=%g Nkicked=%g Mkicked=%g Momkicks=%g dPdtkick=%g V_avg=%g V_momwt_avg=%g Tau_avg=%g \n",
                   All.Time,totMPI_n_wind,totMPI_m_wind,totMPI_mom_wind,totMPI_prob_kick,totMPI_avg_v,
                   totMPI_pwt_avg_v,totMPI_taufac);
            fprintf(FdMomWinds, "%lg %g %g %g %g %g %g %g \n",
                    All.Time,totMPI_n_wind,totMPI_m_wind,totMPI_mom_wind,totMPI_prob_kick,totMPI_avg_v,
                    totMPI_pwt_avg_v,totMPI_taufac);
#ifndef IO_REDUCED_MODE
            fflush(stdout);
#endif
            fflush(FdMomWinds); // can flush because in reduced mode, only written on highest timesteps
        }
    }
}
#endif /* GALSF_FB_RPWIND_DO_IN_SFCALC */

    
#if defined(BH_POPIII_SEEDS) || defined(SINGLE_STAR_FORMATION)
  MPI_Allreduce(&num_bhformed, &tot_bhformed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(tot_bhformed > 0)
  {
      if(ThisTask==0)
      {
#ifndef IO_REDUCED_MODE
      printf("BH/Sink formation: %d gas particles converted into BHs \n",tot_bhformed);
#endif
      }
      All.TotBHs += tot_bhformed;
  } // if(tot_bhformed > 0)
#endif

  MPI_Allreduce(&stars_spawned, &tot_spawned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&stars_converted, &tot_converted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(tot_spawned > 0 || tot_converted > 0)
    {
      if(ThisTask == 0)
	{
#ifndef IO_REDUCED_MODE
	  printf("SFR: spawned %d stars, converted %d gas particles into stars\n",
		 tot_spawned, tot_converted);
#endif
    }
      All.TotNumPart += tot_spawned;
      All.TotN_gas -= tot_converted;
      NumPart += stars_spawned;
      /* Note: N_gas is only reduced once rearrange_particle_sequence is called */
      /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    } //(tot_spawned > 0 || tot_converted > 0)

  for(bin = 0, sfrrate = 0; bin < TIMEBINS; bin++)
    if(TimeBinCount[bin])
      sfrrate += TimeBinSfr[bin];

#ifdef IO_REDUCED_MODE
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
#endif
    {
        MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
        {
            if(All.TimeStep > 0)
                rate = total_sm / (All.TimeStep / (All.cf_atime*All.cf_hubble_a));
            else
                rate = 0;
            /* convert to solar masses per yr */
            rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
            fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars);
            fflush(FdSfr); // can flush it, because only occuring on master steps anyways
        } // thistask==0
    }

    if(tot_converted+tot_spawned > 0) {rearrange_particle_sequence();}

    CPU_Step[CPU_COOLINGSFR] += measure_time();
} /* end of main sfr_cooling routine!!! */





#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) || defined(GALSF_SUBGRID_WINDS) || defined(GALSF_SUBGRID_VARIABLEVELOCITY) || defined(GALSF_SUBGRID_VARIABLEVELOCITY_DM_DISPSERSION)
void assign_wind_kick_from_sf_routine(int i, double sm, double dtime, double pvtau_return[4])
{
    int j;
    double v,p,prob;
    double norm, dir[3];
#ifdef GALSF_FB_RPWIND_LOCAL
    double tau_IR=0;
    double m_gas_kernel,h,m_gas_kernel_i,vq;
    double h_kernel_i;
    double m_st_kernel,m_st_kernel_i,l_st_kernel,rho_to_launch;
#ifdef GALSF_FB_RPWIND_FROMSTARS
    MyDouble *pos;
    double star_age,log_age,lm_ssp;
    double unittime_in_gyr = 0.001*All.UnitTime_in_Megayears/All.HubbleParam;
    int startnode,numngb_inbox,n,dummy;
#endif
#endif
#ifdef GALSF_FB_RPWIND_FROMCLUMPS
    int ngb_run_cntr,kmin,k;
    double dmin1w,dmax1w,dmax2w;
#endif
#ifdef GALSF_WINDS_ISOTROPIC
    double theta, phi;
#endif

#ifdef GALSF_SUBGRID_WINDS
    /* this is the simple, old standard wind model, with constant velocity & loading with SFR */
    p = All.WindEfficiency * sm / P[i].Mass;
    v = sqrt(2 * All.WindEnergyFraction*All.FactorSN*All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency);
    prob = 1 - exp(-p);
#endif
    

#ifdef GALSF_SUBGRID_DMDISPERSION
    /* wind model where launching scales with halo/galaxy bulk properties (as in Vogelsberger's simulations) */
    if(SphP[i].DM_VelDisp > 0 && sm > 0)
    {
        double wind_energy, wind_momentum, wind_mass;
        v = All.VariableWindVelFactor * SphP[i].DM_VelDisp;  /* physical wind velocity */
        //      if(v < 50.0) v = 50.0;
        wind_momentum = sm * All.VariableWindSpecMomentum;
        wind_energy = sm * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN);
        wind_mass = (wind_energy + sqrt(wind_energy * wind_energy + v * v * wind_momentum * wind_momentum)) / (v * v);
        /* wind mass for this particle, assuming the wind is first given the energy wind_energy and then the momentum wind_momentum */
        p = wind_mass / P[i].Mass;
    }
    else
    {
        v = 0;
        p = 0;
    }
    prob = 1 - exp(-p);
#endif
    
    
    
#ifdef GALSF_SUBGRID_VARIABLEVELOCITY
    /* wind model where launching scales with halo/galaxy bulk properties (as in Romeel's simulations) */
    if(SphP[i].HostHaloMass > 0 && sm > 0)
    {
        double HaloConcentrationNorm = 9.;  /* concentration c0 of a halo of unit mass */
        double HaloConcentrationSlope = -0.15;  /* slope n of mass concentration relation, namely c = c0 * M_200,crit^n */

        double r200c, v_esc, c_halo, wind_energy, wind_momentum, wind_mass;
        double rhocrit = 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G);
        rhocrit *= All.Omega0/All.cf_a3inv + (1-All.Omega0-All.OmegaLambda)/All.cf_a2inv + All.OmegaLambda; /* physical critical density at redshift z */

        r200c = pow(SphP[i].HostHaloMass / (4 * M_PI / 3.0 * 200 * rhocrit), 1.0 / 3.0);	/* physical r_200,crit value, assuming FoF mass = M_200,crit */
        v_esc = sqrt(All.G * SphP[i].HostHaloMass / r200c);	/* physical circular velocity at r_200,crit */
        c_halo = HaloConcentrationNorm * pow(SphP[i].HostHaloMass, HaloConcentrationSlope);
        v_esc *= sqrt(2 * c_halo / (log(1 + c_halo) - c_halo / (1 + c_halo)));	/* physical escape velocity of halo */
        v = All.VariableWindVelFactor * v_esc;	/* physical wind velocity */
        
        wind_momentum = sm * All.VariableWindSpecMomentum;
        wind_energy = sm * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN);
        
        wind_mass = (wind_energy + sqrt(wind_energy * wind_energy + v * v * wind_momentum * wind_momentum)) / (v * v);
        /* wind mass for this particle, assuming the wind is first given the energy wind_energy and then the momentum wind_momentum */
        p = wind_mass / P[i].Mass;
    }
    else
    {
        v = 0;
        p = 0;
    }
    prob = 1 - exp(-p);
#endif
    
    
    
#ifdef GALSF_FB_RPWIND_LOCAL
    /* revised winds model (blowing winds out from local dense clumps) */
    double unitmass_in_e10solar = All.UnitMass_in_g/SOLAR_MASS/1.0e10/All.HubbleParam;
    double unitlength_in_kpc = All.UnitLength_in_cm/3.086e21/All.HubbleParam;
    double unitrho_in_e10solar_kpc3 = unitmass_in_e10solar/(unitlength_in_kpc*unitlength_in_kpc*unitlength_in_kpc);
    double unitvel_in_km_s = All.UnitVelocity_in_cm_per_s/1.0e5;
    if(All.WindMomentumLoading<=0)
    {
        p=v=0;
    } else {
        m_st_kernel=l_st_kernel=m_st_kernel_i=h_kernel_i=m_gas_kernel_i=m_gas_kernel=0;
        
#if !defined(GALSF_FB_RPWIND_FROMSTARS)
        /* we only weakly revise the model here, to scale velocities with density estimate */
        m_gas_kernel=0; h=PPP[i].Hsml;
        rho_to_launch = SphP[i].Density*All.cf_a3inv;
        v = WindInitialVelocityBoost * (45.0/unitvel_in_km_s) * pow(rho_to_launch*unitrho_in_e10solar_kpc3, 0.25);
        if(v<0.01/unitvel_in_km_s) v=0.01/unitvel_in_km_s;
        p = 116 * sm / (P[i].Mass*(v/unitvel_in_km_s));
#endif
        
#ifdef GALSF_FB_RPWIND_FROMSTARS
        /* here we have to do a loop to search for nearby stars */
        Ngblist = (int *) mymalloc("Ngblist",NumPart * sizeof(int));
        startnode = All.MaxPart;
        dummy=0;h=0;numngb_inbox=0;
        h=3.0*PPP[i].Hsml;
        pos=P[i].Pos;
        m_st_kernel=0; l_st_kernel=0; //l_st_kernel_nonrad=0;
        do {
            numngb_inbox = ngb_treefind_variable_threads_targeted(pos, h, -1, &startnode, 0, &dummy, &dummy, &dummy, Ngblist, 16); // search for particles of type 4: 2^4=16
            
            /* searches for all new stars inside h */
            if(numngb_inbox>0)
            {
                for(n=0; n<numngb_inbox; n++)
                {
                    j = Ngblist[n];
                    star_age = evaluate_stellar_age_Gyr(P[j].StellarAge);
                    m_st_kernel += P[j].Mass;
                    lm_ssp = evaluate_l_over_m_ssp(star_age) * calculate_relative_light_to_mass_ratio_from_imf(j);
                    l_st_kernel += lm_ssp*P[j].Mass;
                } /* for(n=0; n<numngb_inbox; n++) */
            } /* if(numngb_inbox>0) */
        } while(startnode >= 0);
        
        if(l_st_kernel>0) {
            m_gas_kernel_i = NORM_COEFF*SphP[i].Density*PPP[i].Hsml*PPP[i].Hsml*PPP[i].Hsml;
            rho_to_launch = SphP[i].Density*a3inv;
            m_st_kernel_i=m_st_kernel*(PPP[i].Hsml*PPP[i].Hsml*PPP[i].Hsml/(h*h*h));
            h_kernel_i=PPP[i].Hsml;
            
            /* calculate gas mass in same kernel properly to appropriately share the luminosity */
            m_gas_kernel=0; pos=P[i].Pos;
            startnode = All.MaxPart; dummy=0;
            do {
                numngb_inbox = ngb_treefind_variable_targeted(pos,h,-1,&startnode,0,&dummy,&dummy,1); // search for gas: 2^0=1
                if(numngb_inbox>0) for(n=0; n<numngb_inbox; n++) m_gas_kernel+=P[Ngblist[n]].Mass;
            } while(startnode >= 0);
            //printf("wind h %g numngb %d m_gas_kernel %g \n",h,numngb_inbox,m_gas_kernel);
            if(m_gas_kernel>0) {
                m_gas_kernel_i=m_gas_kernel;
                m_st_kernel_i=m_st_kernel;
                h_kernel_i=h;
            }
            rho_to_launch *= (1 + m_st_kernel_i/m_gas_kernel_i);
        }
        
#ifdef GALSF_FB_RPWIND_FROMCLUMPS
        /* ok, here we do a chained search of neighbors to find the local density peak,
         and use that to define the clump center (and escape velocity wrt that center) */
        
        vq=1; /* use as flag for whether to continue search iteration for a new density peak */
        k=i;  /* start at current particle */
        kmin=i; dmin1w=SphP[k].Density*1000.0;
        ngb_run_cntr=0; h=0;
        do {
            startnode = All.MaxPart; dummy=0;
            h=3.0*PPP[k].Hsml; pos=P[k].Pos;
            dmax1w=SphP[k].Density; vq=0;
            do {
                numngb_inbox=ngb_treefind_variable_targeted(pos,h,-1,&startnode,0,&dummy,&dummy,1); // search for gas: 2^0=1
                if(numngb_inbox>0) {
                    for(n=0; n<numngb_inbox; n++) {
                        j = Ngblist[n];
                        dmax2w=SphP[j].Density;
                        if((ngb_run_cntr==0)||(kmin==i)) {
                            if((dmax2w<dmin1w)&&(j != i)) {
                                dmin1w=dmax2w;
                                kmin=j;
                            }
                        } /* looks for nearby low-density particles to ID density gradient */
                        if(dmax2w>dmax1w)
                        {
                            dmax1w=dmax2w;
                            k=j;
                            vq=1; /* more dense particle found, continue chain */
                        }
                    } /* for(n=0; n<numngb_inbox; n++) */
                } /* if(numngb_inbox>0) */
            }while(startnode>=0);
            ngb_run_cntr++;
            h=0; for(j=0;j<3;j++) h+=(P[i].Pos[j]-P[k].Pos[j])*(P[i].Pos[j]-P[k].Pos[j]);
            h=sqrt(h);
        }while((vq==1)&&(ngb_run_cntr<20)&&(h<=20.0*PPP[i].Hsml));
        
        if(k==i)
        {
            m_gas_kernel = 8.*NORM_COEFF*SphP[i].Density*PPP[i].Hsml*PPP[i].Hsml*PPP[i].Hsml;
            h = 2.0*PPP[i].Hsml;
        }
        else
        {
            h=0;
            for(j=0;j<3;j++) h+=(P[i].Pos[j]-P[k].Pos[j])*(P[i].Pos[j]-P[k].Pos[j]);
            h=sqrt(h);
            if(h<2.0*PPP[i].Hsml) h=2.0*PPP[i].Hsml;
            if(h<2.0*PPP[k].Hsml) h=2.0*PPP[k].Hsml;
            
            m_gas_kernel=0;
            startnode = All.MaxPart; dummy=0;
            do {
                pos=P[k].Pos;
                numngb_inbox = ngb_treefind_variable_targeted(pos,h,-1,&startnode,0,&dummy,&dummy,1); // search for gas: 2^0=1
                if(numngb_inbox>0) for(n=0; n<numngb_inbox; n++) m_gas_kernel+=P[Ngblist[n]].Mass;
            } while(startnode >= 0);
        }
        
        float stcom_pos[3],cl_st_lum=0;
        m_st_kernel=0; //cl_st_lum_nonrad=0;
        startnode = All.MaxPart; dummy=0;
        do {
            pos=P[k].Pos;
            numngb_inbox = ngb_treefind_variable_threads_targeted(pos, h, -1, &startnode, 0, &dummy, &dummy, &dummy, Ngblist, 16); // search for particles of type 4: 2^4=16
            stcom_pos[0]=0.0;stcom_pos[1]=0.0;stcom_pos[2]=0.0;
            if(numngb_inbox>0) { for(n=0; n<numngb_inbox; n++) {
                star_age = evaluate_stellar_age_Gyr(P[Ngblist[n]].StellarAge);
                vq = 10.1+500*pow(m_st_kernel_i*unitmass_in_e10solar,0.62)*
                pow(rho_to_launch*unitrho_in_e10solar_kpc3,-0.5);
                if(star_age <= vq) {
                    m_st_kernel+=P[Ngblist[n]].Mass;
                    lm_ssp = evaluate_l_over_m_ssp(star_age) * calculate_relative_light_to_mass_ratio_from_imf(Ngblist[n]);
                    cl_st_lum+=lm_ssp*P[Ngblist[n]].Mass;
                    
                    stcom_pos[0]+=P[Ngblist[n]].Pos[0]*lm_ssp*P[Ngblist[n]].Mass;
                    stcom_pos[1]+=P[Ngblist[n]].Pos[1]*lm_ssp*P[Ngblist[n]].Mass;
                    stcom_pos[2]+=P[Ngblist[n]].Pos[2]*lm_ssp*P[Ngblist[n]].Mass;
                }
            }
                stcom_pos[0]/=cl_st_lum;
                stcom_pos[1]/=cl_st_lum;
                stcom_pos[2]/=cl_st_lum;
                if(cl_st_lum>0&&m_gas_kernel>0) {
                    if(cl_st_lum/m_gas_kernel > l_st_kernel/m_gas_kernel_i) {
                        l_st_kernel=cl_st_lum;
                        m_gas_kernel_i=m_gas_kernel;
                        //l_st_kernel_nonrad=cl_st_lum_nonrad;
                    }
                }
            }
        } while(startnode >= 0);
#endif // GALSF_FB_RPWIND_FROMCLUMPS
        
        /* alright, now do calculations on the results to determine speed & loading of kicks */
        vq= WindInitialVelocityBoost*sqrt(All.G*(m_gas_kernel_i+m_st_kernel_i)/(h_kernel_i*All.cf_atime));
        v = WindInitialVelocityBoost*sqrt(All.G*(m_gas_kernel+m_st_kernel)/(h*All.cf_atime));
        if(vq>v) v=vq;
        /* compare the velocity from the central star cluster, using the observed cluster size-mass relation */
        if((m_st_kernel==0)&&(m_st_kernel_i>0)) m_st_kernel=m_st_kernel_i;
        vq=WindInitialVelocityBoost*(65.748/unitvel_in_km_s)*pow(m_st_kernel*unitmass_in_e10solar/0.0001,0.25);
        vq=vq*1.82;
        /* this corresponds to =G M_star/R_e for a 10^6 Msun cluster, scaling upwards from there;
         note that All.WindEnergyFraction will boost appropriately; should be at least sqrt(2), if want
         full velocities; in fact for Hernquist profile, the mass-weighted V_esc=1.82 times this */
        if (vq>v) v=vq;
        if(v<0.01/unitvel_in_km_s) v=0.01/unitvel_in_km_s;
        if(v>1000./unitvel_in_km_s) v=1000./unitvel_in_km_s;
        
        /* and now their mass-loading */
        /* winds driven by young stars, rather than proportional to SFR */
        p = 20.52*(dtime/unittime_in_gyr)*(l_st_kernel/m_gas_kernel_i)/(v/unitvel_in_km_s);
        
        myfree(Ngblist); // free neighbor lists used for wind kick searches
#endif // GALSF_FB_RPWIND_FROMSTARS
        
        
        // now calculate the 'boost factor' for the local properties //
        tau_IR = 1.91*SphP[i].Density*All.cf_a3inv*PPP[i].Hsml*All.cf_atime;
        if((m_gas_kernel>0)&&(h>0)) {
            vq=0.48*m_gas_kernel/(h*h*All.cf_atime*All.cf_atime);
            if(vq>tau_IR) tau_IR=vq;
        }
        tau_IR *= KAPPA_IR * All.UnitDensity_in_cgs*All.HubbleParam*All.UnitLength_in_cm;
        tau_IR *= (0.1+P[i].Metallicity[0]/All.SolarAbundances[0]);
        p *= (1.0 + All.WindMomentumLoading*tau_IR); //p += p_nonrad; // now nonrad is explicitly tracked in SNe/gasreturn
    } // closes(All.WindMomentumLoading>0)
    prob=p; if(prob>1) v*=p;
#endif // GALSF_FB_RPWIND_LOCAL
    
    
    
#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) && defined(GALSF_FB_RPWIND_LOCAL) // set values to return to previous step (for tabulation) //
    pvtau_return[0]=0.0;
#endif
#ifdef GALSF_FB_RPWIND_CONTINUOUS
    prob = 2;
#endif
    if(get_random_number(P[i].ID + 2) < prob)	/* ok, make the particle go into the wind */
    {
#ifdef GALSF_FB_RPWIND_CONTINUOUS
        v *= p;
#endif
        
#if defined(GALSF_FB_RPWIND_DO_IN_SFCALC) && defined(GALSF_FB_RPWIND_LOCAL) // set values to return to previous step (for tabulation) //
        pvtau_return[0]=1.0;
        pvtau_return[1]=p;
        pvtau_return[2]=v;
        pvtau_return[3]=tau_IR;
#endif
        
        // determine the wind acceleration orientation //
#ifdef GALSF_FB_RPWIND_FROMSTARS
        for(j=0;j<3;j++) dir[j]=-P[i].GradRho[j]; // default is along opacity gradient //
#endif
        
#if defined(GALSF_WINDS_POLAR) || !defined(GALSF_FB_RPWIND_FROMSTARS) // polar wind (defined by acel.cross.vel)
        dir[0] = P[i].GravAccel[1] * P[i].Vel[2] - P[i].GravAccel[2] * P[i].Vel[1];
        dir[1] = P[i].GravAccel[2] * P[i].Vel[0] - P[i].GravAccel[0] * P[i].Vel[2];
        dir[2] = P[i].GravAccel[0] * P[i].Vel[1] - P[i].GravAccel[1] * P[i].Vel[0];
        if(get_random_number(P[i].ID + 5) < 0.5) {for(j=0;j<3;j++) dir[j]=-dir[j];}
#endif
        
#ifdef GALSF_FB_RPWIND_FROMCLUMPS // in this case wind is directed from the local clump center //
        if(i != k)
        {
            for(j=0;j<3;j++) dir[j]=P[i].Pos[j]-P[k].Pos[j];
        } else {
            for(j=0;j<3;j++) dir[j]=P[kmin].Pos[j]-P[i].Pos[j];
        }
#endif
        
#ifdef GALSF_WINDS_ISOTROPIC // here the direction is random //
        theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
        phi = 2 * M_PI * get_random_number(P[i].ID + 4);
        dir[0] = sin(theta) * cos(phi); dir[1] = sin(theta) * sin(phi); dir[2] = cos(theta);
        if(get_random_number(P[i].ID + 5) < 0.5) {for(j=0;j<3;j++) dir[j]=-dir[j];}
#endif
        
        // now actually do the kick for the wind //
        for(j=0,norm=0;j<3;j++) norm+=dir[j]*dir[j];
        if(norm>0) {norm=sqrt(norm);} else {dir[0]=dir[1]=0; dir[2]=norm=1;}
        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] += v * All.cf_atime * dir[j]/norm;
            SphP[i].VelPred[j] += v * All.cf_atime * dir[j]/norm;
        }
#ifdef GALSF_SUBGRID_WINDS
            SphP[i].DelayTime = All.WindFreeTravelMaxTimeFactor / All.cf_hubble_a;
#endif
    } /* if(get_random_number(P[i].ID + 2) < prob) */
}
#endif // defined(GALSF_FB_RPWIND_DO_IN_SFCALC) || defined(GALSF_SUBGRID_WINDS) || defined(GALSF_SUBGRID_VARIABLEVELOCITY) //




#if defined(GALSF_EFFECTIVE_EQS)
/* Routine to initialize quantities needed for the Spingel & Hernquist effective equation of state */
void init_clouds(void)
{
  double A0, dens, tcool, ne, coolrate, egyhot, x, u4, meanweight;
  double tsfr, y, peff, fac, neff, egyeff, factorEVP, thresholdStarburst;
#ifdef COOL_METAL_LINES_BY_SPECIES
  int k; double Z[NUM_METAL_SPECIES]; for(k=0;k<NUM_METAL_SPECIES;k++) Z[k]=0;
#endif

  if(All.PhysDensThresh == 0)
    {
      A0 = All.FactorEVP;
      egyhot = All.EgySpecSN / A0;
      meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));	/* note: assuming FULL ionization */
      u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
      u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
      dens = 1.0e6 * 3 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G);

      if(All.ComovingIntegrationOn)
	{
	  All.Time = 1.0;	/* to be guaranteed to get z=0 rate */
	  set_cosmo_factors_for_current_time();
	  IonizeParams();
	}

      ne = 1.0;
      SetZeroIonization();
      tcool = GetCoolingTime(egyhot, dens, &ne, -1);
      coolrate = egyhot / tcool / dens;
      x = (egyhot - u4) / (egyhot - All.EgySpecCold);

      All.PhysDensThresh = x / pow(1 - x, 2) * (All.FactorSN * All.EgySpecSN - (1 -
						      All.FactorSN) * All.EgySpecCold) / (All.MaxSfrTimescale * coolrate);

      if(ThisTask == 0)
	{
	  printf("\nA0= %g  \n", A0);
	  printf("Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n", All.PhysDensThresh,
		 All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
	  printf("EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n\n", x);
	  printf("tcool=%g dens=%g egyhot=%g\n", tcool, dens, egyhot);
	}

      dens = All.PhysDensThresh * 10;
      do
	{
	  tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
	  factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
	  egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

	  ne = 0.5;
      tcool = GetCoolingTime(egyhot, dens, &ne, -1);

	  y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
	  x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
	  egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
	  peff = GAMMA_MINUS1 * dens * egyeff;
	  fac = 1 / (log(dens * 1.025) - log(dens));
	  dens *= 1.025;
	  neff = -log(peff) * fac;
	  tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
	  factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
	  egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

	  ne = 0.5;
      tcool = GetCoolingTime(egyhot, dens, &ne, -1);

	  y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
	  x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
	  egyeff = egyhot * (1 - x) + All.EgySpecCold * x;
	  peff = GAMMA_MINUS1 * dens * egyeff;
	  neff += log(peff) * fac;
	}
      while(neff > 4.0 / 3);

      thresholdStarburst = dens;
#ifdef MODIFIEDBONDI
      All.BlackHoleRefDensity = thresholdStarburst;
      All.BlackHoleRefSoundspeed = sqrt(GAMMA * GAMMA_MINUS1 * egyeff);
#endif

#ifndef IO_REDUCED_MODE
      if(ThisTask == 0)
	{
	  printf("Run-away sets in for dens=%g\n", thresholdStarburst);
	  printf("Dynamic range for quiescent star formation= %g\n", thresholdStarburst / All.PhysDensThresh);
	  fflush(stdout);
	}
#endif

      if(All.ComovingIntegrationOn)
	{
	  All.Time = All.TimeBegin;
	  set_cosmo_factors_for_current_time();
	  IonizeParams();
	}

#ifdef WINDS
#ifndef GALSF_FB_RPWIND_LOCAL
      if(All.WindEfficiency > 0)
	if(ThisTask == 0)
	  printf("Windspeed: %g\n",
		 sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency));
#endif
#endif
    }
}
#endif // GALSF_EFFECTIVE_EQS //



#endif // GALSF


