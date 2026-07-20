/* stellar_evolution_functions.h — Canonical implementations of the stellar
 * age / luminosity / metallicity helper cores shared by host and device.
 * The host externals in stellar_evolution.cc and sfr_eff.cc are one-line
 * wrappers around these cores (passing the global P/CellP arrays), so each
 * formula exists once; device callers pass explicit particle-array pointers.
 *
 * Body visibility: a device translation unit that will evaluate the source
 * payload on-device defines GRAVTREE_SOURCE_DEVICE_TU before its includes and,
 * when GRAVTREE_SOURCE_LAZY_SUPPORTED holds (every gravity-tree source-payload
 * helper enabled in this build is device-callable), sees the bodies as device
 * inlines; translation units that own the host wrapper externals define
 * GRAVTREE_SOURCE_HOST_OWNER_TU before their includes and always see the bodies
 * (host-inline), whatever the compiler. TUs that define neither marker never
 * parse the bodies as device code.
 *
 * Under a protostellar-evolution model, StarLuminosity_Solar is cadence-owned
 * by the stellar-evolution update; these cores READ it and never write particle
 * state, so they are side-effect-free on the device lazy path.
 *
 * Caller must include allvars.h and proto.h first (proto.h has no include
 * guard, standard convention).
 */
#ifndef STELLAR_EVOLUTION_FUNCTIONS_H
#define STELLAR_EVOLUTION_FUNCTIONS_H

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#include "../gravity/cosmology_functions.h"

#ifdef GALSF

#if defined(GRAVTREE_SOURCE_HOST_OWNER_TU) || (defined(GRAVTREE_SOURCE_DEVICE_TU) && defined(GRAVTREE_SOURCE_LAZY_SUPPORTED))

/* function to say whether a given particle should be treated as a "single star" or as -eligible to become- a single star (if it is gas), or
    whether it should be treated as a stellar population with some total mass, with IMF-integrated properties */
KOKKOS_INLINE_FUNCTION int is_particle_single_star_eligible_core(long i, struct particle_data *pp)
{
#if defined(SINGLE_STAR_SINK_DYNAMICS)
    if(pp[i].Type == 0 || pp[i].Type == 5) // only type=0 or type=5 (sinks) are eligible in our models here to be 'single star' candidates
    {
#if defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL) // here's the interesting regime, where we have some criterion for deciding which cells are eligible for 'single-star' status
        if(pp[i].Type == 5) {return 1;} // all type-5 elements are assumed sinks
        if(pp[i].Type == 0) {if(pp[i].Mass*UNIT_MASS_IN_SOLAR > (SINGLE_STAR_AND_SSP_HYBRID_MODEL)) {return 0;} else {return 1;}} // use a simple mass threshold to decide which model we will use, specified by using this as a compile-time parameter
#else
        return 1; // no hybrid model, so all particles satisfying these criteria are automatically single-star eligible
#endif
    }
#endif
    return 0; // catch - default to non-single-star (SSP), unless satisfy some of the criteria above
}


/* subroutine to calculate luminosity of an individual star, according to accretion rate,
    mass, age, etc. Modify your assumptions about main-sequence evolution here. ONLY relevant for SINGLE-STAR inputs. */
KOKKOS_INLINE_FUNCTION double calculate_individual_stellar_luminosity_core(double mdot, double mass, long i, struct particle_data *pp)
{
#if !defined(SINGLE_STAR_SINK_DYNAMICS)
    return 0; /* not defined */
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    /* both protostellar-evolution models: luminosity is cadence-owned by the stellar-evolution update
       (singlestar_subgrid_protostellar_evolution_update_track); read the cached value, stale-by-contract */
    return pp[i].StarLuminosity_Solar / UNIT_LUM_IN_SOLAR;
#else
    /* single-star without a protostellar-evolution model: accretion + main-sequence estimate */
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
    (void)lum_sol;
    return lum;
#endif
}


/* return the metallicity (in solar units) used by the stellar evolution fits; bounded because the tables are not arbitrarily extrapolable */
KOKKOS_INLINE_FUNCTION double Z_for_stellar_evol_core(int i, struct particle_data *pp)
{
    if(i<0) {return 1;}
#ifdef METALS
    double Z_solar = pp[i].Metallicity[0]/All.SolarAbundances[0]; // use total metallicity
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2) && defined(COOL_METAL_LINES_BY_SPECIES)
    int i_Fe=10; Z_solar = pp[i].Metallicity[i_Fe]/All.SolarAbundances[i_Fe]; // use Fe, specifically, for computing stellar properties, as its most relevant here. MAKE SURE this is set to the correct abundance in the list, to match Fe!
#endif
    return DMIN(DMAX(Z_solar,0.01),3.); // stellar evolution tables here are not arbitrarily extrapolable, so this is bounded //
#else
    return 1; // metals not evolved, return unity
#endif
}


/* return the light-to-mass ratio, for the IMF of a given particle, relative to the Chabrier/Kroupa IMF.
    ONLY relevant for STELLAR POPULATION integrated inputs. "mode" denotes if we're interested in very massive stars (or other special behaviors): for now -1=bolometric, +1=very massive */
KOKKOS_INLINE_FUNCTION double calculate_relative_light_to_mass_ratio_from_imf_core(double stellar_age_in_gyr, int i, int mode, struct particle_data *pp)
{
#ifdef GALSF_SFR_IMF_VARIATION // fitting function from David Guszejnov's IMF calculations (ok for Mturnover in range 0.01-100) for how mass-to-light ratio varies with IMF shape/effective turnover mass
    double log_mimf = log10(pp[i].IMF_Mturnover);
    return (0.051+0.042*(log_mimf+2)+0.031*(log_mimf+2)*(log_mimf+2)) / 0.31;
#endif
#ifdef GALSF_SFR_IMF_SAMPLING // account for IMF sampling model if not evolving individual stars
    double mu = 0.0115 * pp[i].Mass * UNIT_MASS_IN_SOLAR; // 1 O-star per 100 Msun [more exactly calculated here as number of stars per solar mass with mass > 8 Msun, from our adopted Kroupa IMF from 0.01-100 Msun]
    double t=stellar_age_in_gyr*1000.,t1=3.7,t2=7.,t3=44.,a0=0.13,mu_min=1.e-3*mu;
    if(t>t3) {mu*=0;} else {if(t>t2) {mu*=(1.-a0)*(1.-(t-t2)/(t3-t2));} else {if(t>t1) {mu*=1.-a0*(t-t1)/(t2-t1);}}} // expectation value is declining with time, so 'effective multiplier' is larger
    if(mode > 0) {
        return pp[i].IMF_NumMassiveStars / DMAX(mu,mu_min); // scales just with the number of massive stars
    } else {
        if(t>=t3) {return 1;} else {return 0.01 + pp[i].IMF_NumMassiveStars / DMAX(mu,mu_min);} // scales with a minimum and a long-timescale baseline
    }
#endif
    return 1; // Chabrier or Kroupa IMF //
}


/* stellar age in Gyr from the tracked formation time */
KOKKOS_INLINE_FUNCTION double evaluate_stellar_age_Gyr_core(long i, struct particle_data *pp)
{
    double tform_code = pp[i].StellarAge; // formation time as tracked in-code
#if defined(GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF)
    if(pp[i].Type==4) {tform_code = pp[i].IMF_WeightedMeanStellarFormationTime;} // use this 'effective' age for this module, to reflect the spread-out duration of SF
#endif
    double age = evaluate_time_since_t_initial_in_Gyr_core(tform_code);
    age = DMAX(age, 1.e-5); // set a floor for some routines
    return age;
}


/* return the light-to-mass ratio [in units of Lsun/Msun] of a star or stellar population with a given age; used throughout the code below */
KOKKOS_INLINE_FUNCTION double evaluate_light_to_mass_ratio_core(double stellar_age_in_gyr, int i, struct particle_data *pp)
{
    if(is_particle_single_star_eligible_core(i, pp)) // SINGLE-STAR VERSION: calculate single-star luminosity (and convert to solar luminosity-to-mass ratio, which this output assumes)
    {
#ifdef SINGLE_STAR_SINK_DYNAMICS
        double m0=pp[i].Mass; if(pp[i].Type == 5) {m0=pp[i].Sink_Mass;}
        return calculate_individual_stellar_luminosity_core(0, m0, i, pp) / m0 * (UNIT_LUM_IN_SOLAR) / (UNIT_MASS_IN_SOLAR);
#endif
    }
    else // STELLAR-POPULATION VERSION: compute integrated mass-to-light ratio of an SSP
    {
        double lum=1; if(stellar_age_in_gyr < 0.01) {lum=1000;} // default to a dumb imf-averaged 'young/high-mass' vs 'old/low-mass' distinction
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION // fit to updated SB99 tracks: including rotation, new mass-loss tracks, etc.
        if(stellar_age_in_gyr < 0.0035) {lum=1136.59;} else {double log_age=log10(stellar_age_in_gyr/0.0035); lum=1500.*pow(10.,-1.8*log_age+0.3*log_age*log_age-0.025*log_age*log_age*log_age);}
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        double t1=0.0012, t2=0.0037, f1=800., f2=1100.*pow(Z_for_stellar_evol_core(i, pp),-0.1), tx=log10(stellar_age_in_gyr/t2), t_g=log10(stellar_age_in_gyr/1.2)/0.05;
        if(stellar_age_in_gyr<=t1) {lum=f1;} else if(stellar_age_in_gyr<=t2) {lum=f1*pow(stellar_age_in_gyr/t1,log(f2/f1)/log(t2/t1));} else {lum=f2*pow(10.,-1.82*tx+0.42*tx*tx-0.07*tx*tx*tx)*(1.+1.2*exp(-0.5*t_g*t_g));}
#endif
#endif
#ifdef GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF
        lum *= calculate_relative_light_to_mass_ratio_from_imf_core(stellar_age_in_gyr,i,-1,pp); // account for IMF variation model [if used; currently must be custom set as desired for modules]
#endif
        return lum;
    }
    return 0; // catch
}

/* particle_ionizing_luminosity_in_cgs_core lives in radiation/rt_functions.h
 * (with the RT source-spectrum helpers): it needs sink_lum_bol_core, and
 * keeping it there avoids a stellar<->sink header cycle. */

#endif /* GRAVTREE_SOURCE_HOST_OWNER_TU || (GRAVTREE_SOURCE_DEVICE_TU && GRAVTREE_SOURCE_LAZY_SUPPORTED) */

#endif /* GALSF */

#endif /* STELLAR_EVOLUTION_FUNCTIONS_H */
