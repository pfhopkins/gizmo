#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "./cooling.h"
#if defined(RT_CHEM_PHOTOION)
#include "../radiation/rt_functions.h" /* the ionizing photon density per band, shared with the default network */
#endif

#ifdef COOL_GRACKLE
#include <grackle.h>
#define ENDRUNVAL 91234

/* Interface to Grackle 3 (its local API). The run-time parameters and the rate tables live in this file's own chemistry_data and
   chemistry_data_storage: InitGrackle sets them once and Grackle's solvers only read them afterwards, so the host cooling threads
   share them. Each call builds its own one-cell field data, and its own copy of the units carrying the current scale factor. */
static chemistry_data GrackleChemistry;
static chemistry_data_storage GrackleRates;

//
// 'mode' -- tells the routine what to do
//
//     0 == solve chemistry and assign new abundances
//     1 == calculate and return cooling time
//     2 == calculate and return temperature
//     3 == calculate and return pressure
//     4 == calculate and return gamma (only valid when COOL_GRACKLE_CHEMISTRY>0)
//
#if defined(RT_CHEM_PHOTOION)
/* the transported ionizing radiation as the default network uses it: per ionizing band, a photo-ionization rate sigma*c*n_gamma
   [1/s] and a photo-heating rate G*sigma*c*n_gamma [erg/s] per absorber, for HI, HeI and HeII (in that order) */
static void grackle_rt_photo_rates(int target, double ionization[3], double heating[3])
{
    double c_light = C_LIGHT_CGS / UNIT_LENGTH_IN_CGS; /* times a cross section [code length^2] and a photon density [code length^-3]: per second */
    for(int s = 0; s < 3; s++) {ionization[s] = heating[s] = 0;}
    for(int k = 0; k < N_RT_FREQ_BINS; k++)
    {
        if(!RT_BAND_IS_IONIZING(k)) {continue;}
        double photon_flux = c_light * rt_ionizing_band_photon_number_density(target, k, CellP);
        if(All.rt_ion_G_HI[k] > 0) {ionization[0] += All.rt_ion_sigma_HI[k] * photon_flux; heating[0] += All.rt_ion_G_HI[k] * All.rt_ion_sigma_HI[k] * photon_flux;}
        if(All.rt_ion_G_HeI[k] > 0) {ionization[1] += All.rt_ion_sigma_HeI[k] * photon_flux; heating[1] += All.rt_ion_G_HeI[k] * All.rt_ion_sigma_HeI[k] * photon_flux;}
        if(All.rt_ion_G_HeII[k] > 0) {ionization[2] += All.rt_ion_sigma_HeII[k] * photon_flux; heating[2] += All.rt_ion_G_HeII[k] * All.rt_ion_sigma_HeII[k] * photon_flux;}
    }
    for(int s = 0; s < 3; s++) {heating[s] *= UNIT_ENERGY_IN_CGS;}
}
#endif

/* the metal mass fraction grackle is given: grackle's species carry the hydrogen of the metal-free mass only (its consistency step
   rescales them to X_H*(density - metal density)), so this also sets what every quantity per hydrogen nucleus is taken against */
static double grackle_metal_mass_fraction(int target)
{
#ifdef METALS
    return P[target].Metallicity[0];
#else
    return 0.02;
#endif
}

#if (COOL_GRACKLE_CHEMISTRY > 0)
/* fill the shared cache from the species grackle holds for the cell -- at start-up, from the IC or snapshot, before grackle has solved
   it. the electron and neutral fractions (and the ionized one, with the radiation chemistry) are the cell's own start-up values */
void grackle_cache_species_state(int target)
{
    double density = CellP[target].Density, hydrogen_nuclei_density = HYDROGEN_MASSFRAC * (density - density * grackle_metal_mass_fraction(target)), molecular_mass_fraction = 0;
#if defined(RT_CHEM_PHOTOION)
    double ionized_hydrogen = CellP[target].HII;
#else
    double ionized_hydrogen = DMAX(0, DMIN(1, density * CellP[target].grHII / hydrogen_nuclei_density));
#endif
#if (COOL_GRACKLE_CHEMISTRY >= 2)
    molecular_mass_fraction = DMAX(0, DMIN(1, (density * CellP[target].grH2I + density * CellP[target].grH2II) / hydrogen_nuclei_density));
#endif
    CellP[target].store_solved_chemistry(CellP[target].InternalEnergy, CellP[target].MeanMolecularWeight, CellP[target].Gamma, CellP[target].Ne, CellP[target].HI, ionized_hydrogen, molecular_mass_fraction);
}
#endif

double CallGrackle(double u_old, double rho, double dt, double ne_guess, int target, int mode)
{
    gr_float returnval = 0.0;
    code_units units = All.GrackleUnits; units.a_value = All.cf_atime; /* the current scale factor sets the UV-background redshift */

    int grid_dimension[3] = {1,1,1}, grid_start[3] = {0,0,0}, grid_end[3] = {0,0,0}; /* one cell, no ghost zones */
    grackle_field_data fields;
    gr_initialize_field_data(&fields);
    fields.grid_rank = 3; fields.grid_dimension = grid_dimension; fields.grid_start = grid_start; fields.grid_end = grid_end;

    gr_float density, metal_density, energy, velx, vely, velz;
    gr_float cooling_time, temperature, pressure, gamma;
    velx          = CellP[target].VelPred[0];
    vely          = CellP[target].VelPred[1];
    velz          = CellP[target].VelPred[2];
    density       = rho;
    energy        = u_old;
    metal_density = density * grackle_metal_mass_fraction(target);
    fields.density = &density; fields.internal_energy = &energy; fields.metal_density = &metal_density;
    fields.x_velocity = &velx; fields.y_velocity = &vely; fields.z_velocity = &velz;

#if (COOL_GRACKLE_CHEMISTRY >  0) // non-tabular
    gr_float ne_density;
    gr_float HI_density, HII_density, HM_density;
    gr_float HeI_density, HeII_density, HeIII_density;
    gr_float H2I_density, H2II_density;
    gr_float DI_density, DII_density, HDI_density;
    gr_float tiny = 1.0e-20;

    // Atomic
    gr_float hydrogen_nuclei_density = HYDROGEN_MASSFRAC * (density - metal_density);
    ne_density    = hydrogen_nuclei_density * ne_guess; /* grackle's electron density is n_e*m_H; ne_guess is per hydrogen nucleus, as read back below */

    HI_density    = density * CellP[target].grHI;  //initialized with HYDROGEN_MASSFRAC
    HII_density   = density * CellP[target].grHII;
    HM_density    = density * CellP[target].grHM;

    HeI_density   = density * CellP[target].grHeI;
    HeII_density  = density * CellP[target].grHeII;
    HeIII_density = density * CellP[target].grHeIII;

    H2I_density  = density * tiny;
    H2II_density = density * tiny;
    DI_density   = density * tiny;
    DII_density  = density * tiny;
    HDI_density  = density * tiny;

#if (COOL_GRACKLE_CHEMISTRY >= 2) // Atomic+(H2+H2I+H2II)
    H2I_density  = density * CellP[target].grH2I;
    H2II_density = density * CellP[target].grH2II;
    /* H2 self-shielding length: the Sobolev-type column of the in-code networks over the density, both proper (Grackle's
       grid-cell option cannot describe a particle) */
    gr_float H2_shielding_length = evaluate_NH_from_GradRho(CellP[target].Gradients.Density, P[target].KernelRadius, CellP[target].Density, P[target].NumNgb, 1, target, P) / density;
    fields.H2_self_shielding_length = &H2_shielding_length;
#endif

#if (COOL_GRACKLE_CHEMISTRY >= 3) // Atomic+(H2+H2I+H2II)+(DI+DII+HD)
    DI_density   = density * CellP[target].grDI;
    DII_density  = density * CellP[target].grDII;
    HDI_density  = density * CellP[target].grHDI;
#endif
    fields.e_density = &ne_density;
    fields.HI_density = &HI_density; fields.HII_density = &HII_density; fields.HM_density = &HM_density;
    fields.HeI_density = &HeI_density; fields.HeII_density = &HeII_density; fields.HeIII_density = &HeIII_density;
    fields.H2I_density = &H2I_density; fields.H2II_density = &H2II_density;
    fields.DI_density = &DI_density; fields.DII_density = &DII_density; fields.HDI_density = &HDI_density;

#if defined(RT_CHEM_PHOTOION)
    /* the ionizing radiation the code transports, given to grackle exactly as the default network uses it: per
       ionizing band, a photo-ionization rate sigma*c*n_gamma and a photo-heating rate G*sigma*c*n_gamma per absorber.
       grackle's own heating input is per neutral hydrogen atom, so the helium photo-heating goes in as a volumetric
       rate. every field is supplied (zero where nothing feeds it): grackle reads them all once radiation is on */
    gr_float rt_ionization_HI = 0, rt_ionization_HeI = 0, rt_ionization_HeII = 0, rt_heating_per_HI = 0, rt_H2_dissociation = 0, rt_heating_helium_volumetric = 0;
    {
        double ionization[3], heating[3]; grackle_rt_photo_rates(target, ionization, heating);
        rt_ionization_HI = ionization[0] * UNIT_TIME_IN_CGS; rt_ionization_HeI = ionization[1] * UNIT_TIME_IN_CGS; rt_ionization_HeII = ionization[2] * UNIT_TIME_IN_CGS; /* grackle wants them per its time unit */
        rt_heating_per_HI = heating[0]; /* erg/s per neutral hydrogen atom */
#if defined(RT_CHEM_PHOTOION_HE)
        double helium_atoms_per_gram_of_species = density * UNIT_DENSITY_IN_CGS / (4. * PROTONMASS_CGS); /* times a helium mass fraction: number density */
        rt_heating_helium_volumetric = helium_atoms_per_gram_of_species * (CellP[target].grHeI * heating[1] + CellP[target].grHeII * heating[2]); /* erg/s/cm^3 */
#endif
#if (COOL_GRACKLE_CHEMISTRY >= 2) && defined(RT_LYMAN_WERNER)
        /* H2 photo-dissociation by the transported Lyman-Werner band (11.2-13.6 eV), through grackle's own conversion of a
           mean intensity J21 [1e-21 erg/s/cm^2/Hz/sr] to a dissociation rate, 1.38e-12 J21 per second */
        double u_LW = CellP[target].Rad_E_gamma[RT_FREQ_BIN_LYMAN_WERNER] * (CellP[target].Density*All.cf_a3inv/CellP[target].Mass) * UNIT_PRESSURE_IN_CGS; /* erg/cm^3 */
        double delta_nu_LW = (13.6 - 11.2) * ELECTRONVOLT_IN_ERGS / PLANCK_CGS; /* Hz */
        double J21 = (u_LW / delta_nu_LW) * C_LIGHT_CGS / (4. * M_PI) / 1.e-21;
        rt_H2_dissociation = 1.38e-12 * DMAX(J21, 0) * UNIT_TIME_IN_CGS;
#endif
    }
    fields.RT_HI_ionization_rate = &rt_ionization_HI; fields.RT_HeI_ionization_rate = &rt_ionization_HeI; fields.RT_HeII_ionization_rate = &rt_ionization_HeII;
    fields.RT_heating_rate = &rt_heating_per_HI; fields.RT_H2_dissociation_rate = &rt_H2_dissociation;
#if defined(RT_CHEM_PHOTOION_HE)
    fields.volumetric_heating_rate = &rt_heating_helium_volumetric;
#endif
#endif
#endif // non-tabular
#if defined(RT_CHEM_PHOTOION) && !(COOL_GRACKLE_CHEMISTRY > 0)
    /* the tables carry no species, so the radiation chemistry evolves the ionization states (rt_update_chemistry); the
       photo-heating of those absorbers reaches grackle as a volumetric heating rate */
    gr_float rt_heating_volumetric = 0;
    {
        double ionization[3], heating[3]; grackle_rt_photo_rates(target, ionization, heating);
        double hydrogen_nuclei_cgs = HYDROGEN_MASSFRAC * density * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS; /* the species there are per hydrogen nucleus */
        rt_heating_volumetric = hydrogen_nuclei_cgs * CellP[target].HI * heating[0];
#if defined(RT_CHEM_PHOTOION_HE)
        rt_heating_volumetric += hydrogen_nuclei_cgs * (CellP[target].HeI * heating[1] + CellP[target].HeII * heating[2]);
#endif
    }
    fields.volumetric_heating_rate = &rt_heating_volumetric;
#endif

    int status = GR_FAIL;
    switch(mode) {
        case 0: status = local_solve_chemistry(&GrackleChemistry, &GrackleRates, &units, &fields, dt); break; //solve chemistry & update values
        case 1: status = local_calculate_cooling_time(&GrackleChemistry, &GrackleRates, &units, &fields, &cooling_time); break;
        case 2: status = local_calculate_temperature(&GrackleChemistry, &GrackleRates, &units, &fields, &temperature); break;
        case 3: status = local_calculate_pressure(&GrackleChemistry, &GrackleRates, &units, &fields, &pressure); break;
#if (COOL_GRACKLE_CHEMISTRY >  0)
        case 4: status = local_calculate_gamma(&GrackleChemistry, &GrackleRates, &units, &fields, &gamma); break;
#endif
    }
    if(status != GR_SUCCESS)
    {
        fprintf(stderr, "Error in Grackle (CallGrackle mode %d, target %d).\n", mode, target);
        endrun(ENDRUNVAL);
        /* Soft stop: the grackle call failed, so its output vars are
         * unchanged/uninitialized. Return a safe sentinel rather than
         * writing failed chemistry back to CellP or returning garbage;
         * drains at the next cooling-phase poll. */
        return (mode == 0) ? u_old : 0.0;
    }

    /* after a solve, the temperature and adiabatic index of the new state from grackle itself, cached below with the mean molecular
       weight of the same (u, T, gamma) under the relation the cache owns (gas_temperature_from_u), so every reader agrees with grackle */
    gr_float T_solved = 0, gamma_solved = GrackleChemistry.Gamma;
    if(mode == 0)
    {
        if(local_calculate_temperature(&GrackleChemistry, &GrackleRates, &units, &fields, &T_solved) != GR_SUCCESS) {status = GR_FAIL;}
#if (COOL_GRACKLE_CHEMISTRY >  0)
        if(local_calculate_gamma(&GrackleChemistry, &GrackleRates, &units, &fields, &gamma_solved) != GR_SUCCESS) {status = GR_FAIL;}
#endif
        if(status != GR_SUCCESS)
        {
            fprintf(stderr, "Error in Grackle (temperature/gamma of the solved state, target %d).\n", target);
            endrun(ENDRUNVAL); return u_old; /* soft stop, as above */
        }
    }
    const double mu_solved = (energy > 0 && gamma_solved > 1) ? T_solved / ((gamma_solved - 1.) * U_TO_TEMP_UNITS * energy) : MEAN_MOLECULAR_WEIGHT_DEFAULT;

    switch(mode) {
        case 0:
#if (COOL_GRACKLE_CHEMISTRY >  0)
            // Assign variables back
            CellP[target].grHI    = HI_density    / density;
            CellP[target].grHII   = HII_density   / density;
            CellP[target].grHM    = HM_density    / density;

            CellP[target].grHeI   = HeI_density   / density;
            CellP[target].grHeII  = HeII_density  / density;
            CellP[target].grHeIII = HeIII_density / density;

            /* save the state other modules read back, from the species grackle just solved. the
               gr* fields above are fractions of the TOTAL density; the electron and neutral
               fractions the rest of the code uses are per hydrogen nucleus, of the hydrogen grackle holds, and the
               molecular fraction is the fraction of that hydrogen's mass in H2, as for every other network. the
               temperature, adiabatic index and mean molecular weight are grackle's own (above), which
               count every species it tracks, molecules included -- no chemical network is consulted */
            CellP[target].store_solved_chemistry(energy, mu_solved, gamma_solved, ne_density / hydrogen_nuclei_density, DMAX(0, DMIN(1, HI_density / hydrogen_nuclei_density)),
                                                 DMAX(0, DMIN(1, HII_density / hydrogen_nuclei_density)), DMAX(0, DMIN(1, (H2I_density + H2II_density) / hydrogen_nuclei_density)));
#if defined(RT_CHEM_PHOTOION)
#if defined(RT_CHEM_PHOTOION_HE)
            CellP[target].HeI = DMAX(0, (HeI_density / 4.) / hydrogen_nuclei_density);
            CellP[target].HeII = DMAX(0, (HeII_density / 4.) / hydrogen_nuclei_density);
            CellP[target].HeIII = DMAX(0, (HeIII_density / 4.) / hydrogen_nuclei_density);
#endif
#endif

#if (COOL_GRACKLE_CHEMISTRY >= 2) // Atomic+(H2+H2I+H2II)
            CellP[target].grH2I   = H2I_density   / density;
            CellP[target].grH2II  = H2II_density  / density;
#endif

#if (COOL_GRACKLE_CHEMISTRY >= 3) // Atomic+(H2+H2I+H2II)+(DI+DII+HD)
            CellP[target].grDI    = DI_density    / density;
            CellP[target].grDII   = DII_density   / density;
            CellP[target].grHDI   = HDI_density   / density;
#endif
#else // tabular
            /* The temperature, adiabatic index and mean molecular weight are grackle's (above): its
               tables carry the mean molecular weight of the gas they solved. They hand back no
               species, though, so the free electron and neutral fractions other modules read are
               a documented guess -- a simple collisional ionization fraction at grackle's
               temperature. This deliberately does NOT call the chemical network in cooling.cc --
               that is a different network from the one the tables just solved, so its answer
               would not describe this gas. Use a chemistry mode if you need real abundances. */
            {
            CellP[target].Gamma = gamma_solved;
            CellP[target].MeanMolecularWeight = mu_solved;
            CellP[target].Temperature = T_solved;
#ifndef RT_CHEM_PHOTOION /* with it, the radiation chemistry evolves the hydrogen ionization and the electrons, which the tables do not: they stay its */
            double f_ion_guess = 0;
            if(T_solved > 1000.) {f_ion_guess = exp(-15000./T_solved);}
            CellP[target].Ne = f_ion_guess;
            CellP[target].HI = DMAX(0, 1. - f_ion_guess);
#endif
            }
#endif // COOL_GRACKLE_CHEMISTRY
            returnval = energy;
            break;
        case 1: returnval = cooling_time; break;
        case 2: returnval = temperature; break;
        case 3: returnval = pressure; break;
        case 4: returnval = gamma; break;
    } //end switch

    return returnval;
}




//Initialize Grackle
void InitGrackle(void)
{
    grackle_version version = get_grackle_version();
    if(atoi(version.version) < 3)
    {
        if(ThisTask == 0) {printf("COOL_GRACKLE needs Grackle 3 or later; the linked library is version %s.\n", version.version);}
        endrun(ENDRUNVAL); gizmo_exit_bad_stop_if_requested("grackle:version"); return;
    }
    if(gr_check_consistency() != GR_SUCCESS) {fprintf(stderr, "Error in gr_check_consistency: grackle.h does not match the linked Grackle library.\n"); endrun(ENDRUNVAL); gizmo_exit_bad_stop_if_requested("grackle:consistency"); return;}
    grackle_verbose = 0;
    // Enable output
    if(ThisTask == 0) grackle_verbose = 1;

    // First, set up the units system.
    // These are conversions from code units to cgs.
    All.GrackleUnits.comoving_coordinates = 0; //All.ComovingIntegrationOn; // 1 if cosmological sim, 0 if not
    All.GrackleUnits.density_units        = UNIT_DENSITY_IN_CGS;
    All.GrackleUnits.length_units         = UNIT_LENGTH_IN_CGS;
    All.GrackleUnits.time_units           = UNIT_TIME_IN_CGS;
    All.GrackleUnits.velocity_units       = UNIT_VEL_IN_CGS;
    All.GrackleUnits.a_units              = 1.0; // units for the expansion factor
    // Set initial expansion factor (for internal units).
    // Set expansion factor to 1 for non-cosmological simulation.
    All.GrackleUnits.a_value              = 1.0; if(All.ComovingIntegrationOn) {All.GrackleUnits.a_value = All.TimeBegin;}

    // Second, create a chemistry object for parameters and rate data.
    if(local_initialize_chemistry_parameters(&GrackleChemistry) != GR_SUCCESS) {fprintf(stderr, "Error in local_initialize_chemistry_parameters.\n"); endrun(ENDRUNVAL); gizmo_exit_bad_stop_if_requested("grackle:set_default_params"); return;}
    // Third, set parameter values for chemistry & cooling
    int three_body_rate=0, metal_cooling=0, h2_on_dust=0, photoelectric_heating=0, cmb_temperature_floor=1, UVbackground=1, Compton_xray_heating=1, cie_cooling=0, h2_optical_depth_approximation=0, LWbackground_intensity=0, LWbackground_sawtooth_suppression=0, use_grackle=1, with_radiative_cooling=1, primordial_chemistry=0, dust_chemistry=0, H2_self_shielding=0, self_shielding_method=3; double photoelectric_heating_rate=8.5e-26, Gamma_touse=5./3.; dust_chemistry=0;

    /* optional flags:: */
    three_body_rate = 0; /* Flag to control which three-body H2 formation rate is used.
                          0: Abel, Bryan & Norman (2002), 1: Palla, Salpeter & Stahler (1983), 2: Cohen & Westberg (1983), 3: Flower & Harris (2007), 4: Glover (2008).
                          These are discussed in Turk et. al. (2011). Default: 0. */
    cmb_temperature_floor = 1; // Flag to enable an effective CMB temperature floor. This is implemented by subtracting the value of the cooling rate at TCMB from the total cooling rate. Default: 1.
    UVbackground = 1; // Flag to enable a UV background. If enabled, the cooling table to be used must be specified with the grackle_data_file parameter. Default: 0.
    Compton_xray_heating = 1; // Flag to enable Compton heating from an X-ray background following Madau & Efstathiou (1999). Default: 0.
    cie_cooling = 0; // Flag to enable H2 collision-induced emission cooling from Ripamonti & Abel (2004). Default: 0.
    h2_optical_depth_approximation = 0; // Flag to enable H2 cooling attenuation from Ripamonti & Abel (2004). Default: 0
    LWbackground_intensity = 0; // Rad_Intensity of a constant Lyman-Werner H2 photo-dissociating radiation field, in units of 10-21 erg s-1 cm-2 Hz-1 sr-1. Default: 0.
    LWbackground_sawtooth_suppression = 0; // Flag to enable suppression of Lyman-Werner flux due to Lyman-series absorption (giving a sawtooth pattern), taken from Haiman & Abel, & Rees (2000). Default: 0.
    self_shielding_method=3; // Switch to enable approximate self-shielding from the UV background (see user guide for details); needs a table carrying the shielding cross sections (Grackle's *_shielded.h5 tables)
#ifdef RT_DISABLE_UV_BACKGROUND
    UVbackground = 0; self_shielding_method = 0; // the UV background is off for the whole code, so Grackle has none either, and nothing to shield it from
#endif
#ifdef METALS
    metal_cooling = 1; // Flag to enable metal cooling using the Cloudy tables. If enabled, the cooling table to be used must be specified with the grackle_data_file parameter. Default: 0.
    dust_chemistry = 1; // Flag to control additional dust cooling and chemistry processes. Default: 0 (no dust). 1: adds the following processes: photo-electric heating, electron recombination on dust, H2 formation on dust.
    h2_on_dust = 0; // Flag to enable H2 formation on dust grains, dust cooling, and dust-gas heat transfer follow Omukai (2000). This assumes that the dust to gas ratio scales with the metallicity. Default: 0.
    // Flag to enable a spatially uniform heating term approximating photo-electric heating from dust from Tasker & Bryan (2008). Default: 0.
    // If photoelectric_heating enabled, photoelectric_heating_rate is the heating rate in units of erg cm-3 s-1. Default: 8.5e-26.
    //   (Caution: this tends to heat gas even at extremely high densities to ~3000 K, when it should be entirely self-shielding)
    photoelectric_heating = 1; // photo-electric on [but not adjusted to local background, beware!]
    photoelectric_heating_rate = 8.5e-26; // default rate normalization
#endif
    /* fixed flags:: */
    use_grackle = 1; // Flag to activate the grackle machinery
    with_radiative_cooling = 1; // Flag to include radiative cooling and actually update the thermal energy during the chemistry solver. If off, the chemistry species will still be updated. The most common reason to set this to off is to iterate the chemistry network to an equilibrium state. Default: 1.
    Gamma_touse = GAMMA_DEFAULT; // The ratio of specific heats for an ideal gas. A direct calculation for the molecular component is used if primordial_chemistry > 1. Default: 5/3.
    primordial_chemistry = 0; // Flag to control which primordial chemistry network is used (set by Config file). 0 = fully tabulated cooling
#ifdef COOL_GRACKLE_CHEMISTRY
    primordial_chemistry = COOL_GRACKLE_CHEMISTRY;
#endif
    if(primordial_chemistry >= 2) {H2_self_shielding = 2;} // H2 self-shielding from both the UV background dissociation rate and the H2 dissociation rate, over the Sobolev-type length CallGrackle supplies

    GrackleChemistry.grackle_data_file = All.GrackleDataFile; // Path to the data file containing the metal cooling and UV background tables
    GrackleChemistry.three_body_rate = three_body_rate;
    GrackleChemistry.metal_cooling = metal_cooling;
    GrackleChemistry.h2_on_dust = h2_on_dust;
    GrackleChemistry.photoelectric_heating = photoelectric_heating;
    GrackleChemistry.photoelectric_heating_rate = photoelectric_heating_rate;
    GrackleChemistry.cmb_temperature_floor = cmb_temperature_floor;
    GrackleChemistry.UVbackground = UVbackground;
    GrackleChemistry.Compton_xray_heating = Compton_xray_heating;
    GrackleChemistry.cie_cooling = cie_cooling;
    GrackleChemistry.h2_optical_depth_approximation = h2_optical_depth_approximation;
    GrackleChemistry.LWbackground_intensity = LWbackground_intensity;
    GrackleChemistry.LWbackground_sawtooth_suppression = LWbackground_sawtooth_suppression;
    GrackleChemistry.use_grackle = use_grackle;
    GrackleChemistry.with_radiative_cooling = with_radiative_cooling;
    GrackleChemistry.Gamma = Gamma_touse;
    GrackleChemistry.primordial_chemistry = primordial_chemistry;
    GrackleChemistry.dust_chemistry = dust_chemistry;
    GrackleChemistry.H2_self_shielding = H2_self_shielding;
    GrackleChemistry.self_shielding_method = self_shielding_method;
    /* the hydrogen mass fraction the species are reckoned in is the code's; the tabular mode must keep its tables' own, so it is left to grackle there */
    if(primordial_chemistry > 0) {GrackleChemistry.HydrogenFractionByMass = HYDROGEN_MASSFRAC;}
#if defined(RT_CHEM_PHOTOION)
    /* the transported ionizing radiation drives grackle's photo-ionization and photo-heating (CallGrackle supplies the rates);
       the helium photo-heating goes in through grackle's volumetric heating input */
    if(primordial_chemistry > 0) {GrackleChemistry.use_radiative_transfer = 1;}
#if defined(RT_CHEM_PHOTOION_HE)
    if(primordial_chemistry > 0) {GrackleChemistry.use_volumetric_heating_rate = 1;}
#endif
    if(primordial_chemistry == 0) {GrackleChemistry.use_volumetric_heating_rate = 1;} /* tabular: all the photo-heating goes in this way */
#endif

    // Finally, initialize the chemistry object.
    if(local_initialize_chemistry_data(&GrackleChemistry, &GrackleRates, &All.GrackleUnits) != GR_SUCCESS)
    {
        fprintf(stderr, "Error in local_initialize_chemistry_data with GrackleDataFile=%s (the UV-background self-shielding needs one of Grackle's *_shielded.h5 tables).\n", All.GrackleDataFile);
        endrun(ENDRUNVAL); gizmo_exit_bad_stop_if_requested("grackle:init_chem_data"); return;
    }
    if(ThisTask == 0) {printf("Grackle Initialized: version %s, primordial_chemistry=%d, gr_float of %d bytes\n", version.version, primordial_chemistry, (int)sizeof(gr_float));}
}

#endif  //COOL_GRACKLE
