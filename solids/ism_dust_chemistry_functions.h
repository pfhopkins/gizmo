/* ism_dust_chemistry_functions.h — KOKKOS_INLINE_FUNCTION migrations of
 * functions originally in solids/ism_dust_chemistry.cc that are called from
 * __host__ __device__ inline contexts (cooling/cooling.cc kernel CoolingRate,
 * galaxy_sf/mechanical_fb_functions.h mechanical_fb_pair_kernel).
 *
 * Migrated 2026-05-21 (Phase D config 109 FIRE_PHYSICS_DEFAULTS=3 +
 * GALSF_ISMDUSTCHEM_MODEL=2): the .cc-defined versions are plain `double`
 * host-only, so calling them from KOKKOS_INLINE_FUNCTION triggered
 * #20011-D (silent-physics-on-GPU). Bodies are pure compute + struct field
 * reads + All.* (mirror-safe under 93897f62) + standard libm — all
 * device-callable.
 *
 * Include order: after allvars.h (for All, struct field types). */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#if defined(GALSF_ISMDUSTCHEM_MODEL)

/* Approximate dust cooling via electron-dust collisions for MRN sized dust in
 * plasmas from Dwek(1987)+Dwek&Werner(1981). Surpasses metal-line cooling
 * for >10^6 K (even without considering dust depletion), but overpredicts
 * dust cooling for <10^7 K since cooling is dominated by small grains which
 * should be destroyed via sputtering. */
KOKKOS_INLINE_FUNCTION
double Lambda_Dust_HighTemperature_Gas_ISM(int target, double T, double n_elec,
                                            struct particle_data *pp,
                                            struct gas_cell_data *cell)
{
    if(target<0 || T<1.e5) {return 0;} // dust cooling << metal-line cooling below 10^5 K
    if(cell[target].ISMDustChem_Dust_Metal[0] <= 0) {return 0;}
    // rho_c (gm cm^-3) grain solid density (intermediate between silicate and carbonaceous), a3 (cm^3) average grain volume for MRN grain size distribution with a=4-250nm (i.e. integrate a^3 dn/da with dn/da normalize to unity), Havg (erg s^−1 cm^3) average heating rate for a dust grain assuming MRN size distribution by incident electrons
    double rho_c=3., a3=2.21e-18, h_frac = 1-(pp[target].Metallicity[0]+pp[target].Metallicity[1]);
    double Havg, coolrate;
    if (T>=7.17E7) {Havg=1.43E-11;}
    else if (T>=2.39E7) {Havg=-2.07E-12+1.23E-16*pow(T,0.745)+2.10E-17*pow(T,0.75)-1.07E-17*pow(T,0.88);}
    else if (T>=4.55E6) {Havg=-2.07E-12+1.70E-17*pow(T,0.745)+3.96E-17*pow(T,0.75)-5.44E-23*pow(T,1.5);}
    else if (T>=1.52E6) {Havg=-1.06E-16*pow(T,0.745)+1.86E-17*pow(T,0.75)+1.56E-17*pow(T,0.88)-5.44E-23*pow(T,1.5);}
    else {Havg=3.76E-22*pow(T,1.5);}
    // Lambda/nH^2 cooling rate (ergs s^-1 cm^3) same as rest of cooling routine (note n_elec is the ratio of electron to H densities)
    coolrate = (3.*cell[target].ISMDustChem_Dust_Metal[0]*PROTONMASS_CGS)/(4.*M_PI*rho_c*h_frac)*n_elec*(Havg/a3);
    if(!isfinite(coolrate)) {coolrate=0;}
    return coolrate;
}

/* return the mass of gas shocked by an SNe in which dust can be destroyed */
KOKKOS_INLINE_FUNCTION
double ISMDustChem_Return_Mass_Where_Dust_Shocked(double rho_cell_in_code_units,
                                                   double Esne51_into_cell,
                                                   double mass_preshock_in_code_units,
                                                   double Z_cell)
{
    double vs7=1., local_n0=rho_cell_in_code_units*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS; // dust destruction efficiency, minimum gas shock velocity in ~10^7 cm/s which destroys dust, and number density around SNe
    double mass_shocked_in_code_units; // mass shocked to 100 km/s which destroys dust. use the weights to distribute shocked mass across the neighboring gas particles
#if (GALSF_ISMDUSTCHEM_MODEL & 16)  || (GALSF_ISMDUSTCHEM_MODEL & 32)
    /* From detailed SNR simulations in Kirchschlager+ 2022/24 */
    // TBD

    /* From fits in Yamasawa+ 2011 */
    mass_shocked_in_code_units = 1535 * Esne51_into_cell / (pow(local_n0, 0.202) * pow(Z_cell/All.SolarAbundances[0]+0.039,0.298) * UNIT_MASS_IN_SOLAR);
#else
    /* Simple radiative SNR case from McKee 1989 and Cioffi 1988 */
    mass_shocked_in_code_units = 2460 * Esne51_into_cell / (pow(local_n0, 0.1) * pow(vs7, 9./7.) * UNIT_MASS_IN_SOLAR);
#endif

    return DMIN(mass_shocked_in_code_units * All.ISMDustChem_SNeGasClearedOfDustScaling, mass_preshock_in_code_units); // mass shocked limited to the entire mass of the gas particle
}

#endif // GALSF_ISMDUSTCHEM_MODEL
