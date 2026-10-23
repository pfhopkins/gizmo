#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "./cooling.h"

#ifdef COOL_GRACKLE
#include <grackle.h>
#define ENDRUNVAL 91234

/* Interface to Grackle 3.x (tested with 3.4.1; build Grackle with its CMake build, see the COOL_GRACKLE notes in the
 * documentation). One-cell grackle_field_data and the local_* functions on a chemistry_data/chemistry_data_storage pair
 * private to this file, so none of Grackle's global state (grackle_data, grackle_rates) is touched. Only the Grackle 3 API is
 * supported: InitGrackle stops the run unless COOL_GRACKLE_APIVERSION=3 and the linked Grackle library is >= 3.0.
 * CallGrackle mode: 0 = solve chemistry and assign new abundances, 1 = cooling time, 2 = temperature, 3 = pressure, 4 = gamma.
 */
#include <hdf5.h>

static chemistry_data GrackleChemistry;      /* run-time parameters (set once in InitGrackle) */
static chemistry_data_storage GrackleRates;  /* rate and cooling tables */

/* Inputs are proper (physical) code units, as everywhere in the cooling routines; ne_guess is the free-electron abundance per H nucleus
   (cell.Ne). target indexes pp/cell, which in the cooling loop are compacted copies of P/CellP (copied back afterwards), so the
   species must be read from and written to those arrays, never P/CellP directly. If ne_eval is given, mode 0 writes the updated electron abundance there (DoCooling passes it on to CellP.Ne). target<0 is
   a cell-less call (e.g. the effective-EOS tables), which uses primordial neutral abundances and solar-scaled metals. */
double CallGrackle(double u_old, double rho, double dt, double ne_guess, int target, int mode, double *ne_eval, struct particle_data *pp, struct gas_cell_data *cell)
{
    gr_float returnval = 0.0;
    code_units units = All.GrackleUnits; /* local copy: a_value follows the current scale factor (sets the UV-background redshift) */
    units.a_value = All.cf_atime / units.a_units;

    int grid_dimension[3] = {1,1,1}, grid_start[3] = {0,0,0}, grid_end[3] = {0,0,0}; /* one cell, no ghost zones */
    grackle_field_data fields;
    if(gr_initialize_field_data(&fields) != GR_SUCCESS) {fprintf(stderr, "Error in gr_initialize_field_data.\n"); endrun(ENDRUNVAL);}
    fields.grid_rank = 3; fields.grid_dimension = grid_dimension; fields.grid_start = grid_start; fields.grid_end = grid_end;
    fields.grid_dx = 0; /* only used by H2_self_shielding=1 (Cartesian grids); we use =2 with a Sobolev length below */

    gr_float density = rho, energy = u_old, velx = 0, vely = 0, velz = 0; /* velocities are not used by the solver */
    fields.density = &density; fields.internal_energy = &energy;
    fields.x_velocity = &velx; fields.y_velocity = &vely; fields.z_velocity = &velz;
    gr_float metal_density = 0.02 * density;
#ifdef METALS
    if(target >= 0) {metal_density = density * pp[target].Metallicity[0];}
#endif
    fields.metal_density = &metal_density;

#if (COOL_GRACKLE_CHEMISTRY > 0) // non-tabular: Grackle evolves the species (mass densities) and the electrons
    gr_float tiny = 1.0e-20, xH = HYDROGEN_MASSFRAC;
    gr_float HI_density = xH*density, HII_density = tiny*density, HM_density = tiny*density;
    gr_float HeI_density = (1-xH)*density, HeII_density = tiny*density, HeIII_density = tiny*density;
    gr_float e_density = ne_guess * xH * density; /* Grackle's e_density is n_e*m_H (code density units) = Ne * X_H * rho */
    if(target >= 0)
    {
        HI_density = density * cell[target].grHI; HII_density = density * cell[target].grHII; HM_density = density * cell[target].grHM;
        HeI_density = density * cell[target].grHeI; HeII_density = density * cell[target].grHeII; HeIII_density = density * cell[target].grHeIII;
    }
    fields.HI_density = &HI_density; fields.HII_density = &HII_density; fields.HM_density = &HM_density;
    fields.HeI_density = &HeI_density; fields.HeII_density = &HeII_density; fields.HeIII_density = &HeIII_density;
    fields.e_density = &e_density;
#if (COOL_GRACKLE_CHEMISTRY >= 2) // Atomic+(H2+H2I+H2II)
    gr_float H2I_density = tiny*density, H2II_density = tiny*density, H2_shielding_length = 0;
    if(target >= 0)
    {
        H2I_density = density * cell[target].grH2I; H2II_density = density * cell[target].grH2II;
        /* Sobolev-type length L = Sigma/rho with the same column estimator as the in-code networks (proper code units) */
        H2_shielding_length = evaluate_NH_from_GradRho(cell[target].Gradients.Density, pp[target].KernelRadius, cell[target].Density, pp[target].NumNgb, 1, target, pp) / density;
    }
    fields.H2I_density = &H2I_density; fields.H2II_density = &H2II_density;
    fields.H2_self_shielding_length = &H2_shielding_length;
#endif
#if (COOL_GRACKLE_CHEMISTRY >= 3) // Atomic+(H2+H2I+H2II)+(DI+DII+HD)
    gr_float DI_density = GrackleChemistry.DeuteriumToHydrogenRatio * xH * density, DII_density = tiny*density, HDI_density = tiny*density;
    if(target >= 0) {DI_density = density * cell[target].grDI; DII_density = density * cell[target].grDII; HDI_density = density * cell[target].grHDI;}
    fields.DI_density = &DI_density; fields.DII_density = &DII_density; fields.HDI_density = &HDI_density;
#endif
#endif // COOL_GRACKLE_CHEMISTRY > 0

    switch(mode) {
        case 0: //solve chemistry & update values
            if(local_solve_chemistry(&GrackleChemistry, &GrackleRates, &units, &fields, dt) != GR_SUCCESS) {
                fprintf(stderr, "Error in local_solve_chemistry: target=%d rho=%g u=%g dt=%g\n", target, rho, u_old, dt); endrun(ENDRUNVAL);}
#if (COOL_GRACKLE_CHEMISTRY > 0)
            if(target >= 0)
            {
                cell[target].grHI = HI_density / density; cell[target].grHII = HII_density / density; cell[target].grHM = HM_density / density;
                cell[target].grHeI = HeI_density / density; cell[target].grHeII = HeII_density / density; cell[target].grHeIII = HeIII_density / density;
#if (COOL_GRACKLE_CHEMISTRY >= 2)
                cell[target].grH2I = H2I_density / density; cell[target].grH2II = H2II_density / density;
#endif
#if (COOL_GRACKLE_CHEMISTRY >= 3)
                cell[target].grDI = DI_density / density; cell[target].grDII = DII_density / density; cell[target].grHDI = HDI_density / density;
#endif
            }
            if(ne_eval) {*ne_eval = e_density / (xH * density);}
#else // tabular: no species or electron field in Grackle; derive Ne from the mean molecular weight of Grackle's (Cloudy) tables,
      //   mu = k T / ((gamma-1) m_H u), with 1/mu = X (1 + Ne) + (1-X)/4 for X = Grackle's tabular H mass fraction. (The in-code network
      //   used here before is not initialized when COOL_GRACKLE is on.) RT_CHEM_PHOTOION species are not updated in this mode.
            if(ne_eval)
            {
                gr_float temperature = 0;
                if(local_calculate_temperature(&GrackleChemistry, &GrackleRates, &units, &fields, &temperature) != GR_SUCCESS) {fprintf(stderr, "Error in local_calculate_temperature.\n"); endrun(ENDRUNVAL);}
                double xHt = GrackleChemistry.HydrogenFractionByMass, mu = BOLTZMANN_CGS * temperature / ((GrackleChemistry.Gamma-1) * PROTONMASS_CGS * energy * UNIT_SPECEGY_IN_CGS);
                *ne_eval = DMAX(0, (1./mu - xHt - (1-xHt)/4.) / xHt);
            }
#endif
            returnval = energy;
            break;
        case 1: //cooling time (code units; negative for net cooling, as Grackle returns it)
            {gr_float cooling_time = 0;
            if(local_calculate_cooling_time(&GrackleChemistry, &GrackleRates, &units, &fields, &cooling_time) != GR_SUCCESS) {fprintf(stderr, "Error in local_calculate_cooling_time.\n"); endrun(ENDRUNVAL);}
            returnval = cooling_time;}
            break;
        case 2: //temperature [K]
            {gr_float temperature = 0;
            if(local_calculate_temperature(&GrackleChemistry, &GrackleRates, &units, &fields, &temperature) != GR_SUCCESS) {fprintf(stderr, "Error in local_calculate_temperature.\n"); endrun(ENDRUNVAL);}
            returnval = temperature;}
            break;
        case 3: //pressure (code units)
            {gr_float pressure = 0;
            if(local_calculate_pressure(&GrackleChemistry, &GrackleRates, &units, &fields, &pressure) != GR_SUCCESS) {fprintf(stderr, "Error in local_calculate_pressure.\n"); endrun(ENDRUNVAL);}
            returnval = pressure;}
            break;
        case 4: //gamma
            {gr_float gamma = GAMMA_DEFAULT;
            if(local_calculate_gamma(&GrackleChemistry, &GrackleRates, &units, &fields, &gamma) != GR_SUCCESS) {fprintf(stderr, "Error in local_calculate_gamma.\n"); endrun(ENDRUNVAL);}
            returnval = gamma;}
            break;
    } //end switch
    return returnval;
}


/* does the Grackle table contain the UV-background shielding cross sections (required by self_shielding_method>0)? */
static int GrackleTableHasShielding(const char *fname)
{
    int found = 0;
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file >= 0)
    {
        if(H5Lexists(file, "/UVBRates", H5P_DEFAULT) > 0 && H5Lexists(file, "/UVBRates/CrossSections", H5P_DEFAULT) > 0) {found = 1;}
        H5Fclose(file);
    }
    return found;
}

//Initialize Grackle (Grackle >= 3 API)
void InitGrackle(void)
{
#if !defined(COOL_GRACKLE_APIVERSION) || (COOL_GRACKLE_APIVERSION < 3)
    if(ThisTask == 0) {printf("COOL_GRACKLE requires COOL_GRACKLE_APIVERSION=3 (Grackle 3.x): the older Grackle APIs are no longer supported. Set COOL_GRACKLE_APIVERSION=3 in Config.sh and link Grackle >= 3.0.\n");}
    endrun(ENDRUNVAL);
#endif
    if(atoi(get_grackle_version().version) < 3) {if(ThisTask == 0) {printf("COOL_GRACKLE requires Grackle >= 3.0, but the linked library is version %s.\n", get_grackle_version().version);} endrun(ENDRUNVAL);}
    grackle_verbose = 0; if(ThisTask == 0) {grackle_verbose = 1;} // output from the root task only
    if(gr_check_consistency() != GR_SUCCESS) {fprintf(stderr, "Error in gr_check_consistency: grackle.h does not match the linked Grackle library.\n"); endrun(ENDRUNVAL);}

    // units: conversions from (proper) code units to cgs. The cooling routines pass proper quantities, so comoving_coordinates=0 and
    //   a_units=1 (required by Grackle in that case); a_value is updated to the current scale factor in every call (UV-background redshift).
    All.GrackleUnits.comoving_coordinates = 0;
    All.GrackleUnits.density_units        = UNIT_DENSITY_IN_CGS;
    All.GrackleUnits.length_units         = UNIT_LENGTH_IN_CGS;
    All.GrackleUnits.time_units           = UNIT_TIME_IN_CGS;
    All.GrackleUnits.velocity_units       = UNIT_VEL_IN_CGS;
    All.GrackleUnits.a_units              = 1.0;
    All.GrackleUnits.a_value              = 1.0; if(All.ComovingIntegrationOn) {All.GrackleUnits.a_value = All.TimeBegin;}

    if(local_initialize_chemistry_parameters(&GrackleChemistry) != GR_SUCCESS) {fprintf(stderr, "Error in local_initialize_chemistry_parameters.\n"); endrun(ENDRUNVAL);}
    chemistry_data *gc = &GrackleChemistry;
    gc->use_grackle = 1; // Flag to activate the grackle machinery
    gc->with_radiative_cooling = 1; // include radiative cooling and actually update the thermal energy during the chemistry solver
    gc->grackle_data_file = All.GrackleDataFile; // Path to the data file containing the metal cooling and UV background tables
    gc->Gamma = GAMMA_DEFAULT; // ratio of specific heats for an ideal gas (computed directly for the molecular component if primordial_chemistry > 1)
    gc->primordial_chemistry = 0; // 0 = fully tabulated cooling (set by Config file)
#ifdef COOL_GRACKLE_CHEMISTRY
    gc->primordial_chemistry = COOL_GRACKLE_CHEMISTRY;
#endif
    gc->UVbackground = 1; // UV background from the grackle_data_file table
    gc->cmb_temperature_floor = 1; // effective CMB temperature floor
    gc->Compton_xray_heating = 1; // Compton heating from an X-ray background (Madau & Efstathiou 1999)
    gc->three_body_rate = 0; // Abel, Bryan & Norman (2002)
    gc->cie_cooling = 0; gc->h2_optical_depth_approximation = 0; gc->LWbackground_intensity = 0; gc->LWbackground_sawtooth_suppression = 0;
    if(gc->primordial_chemistry > 0)
    {
        gc->HydrogenFractionByMass = HYDROGEN_MASSFRAC; // same X_H as the rest of the code (for =0 Grackle requires the value of its Cloudy tables)
        /* approximate self-shielding of HI+HeI from the UV background, HeII photoionization ignored (Rahmati et al. 2013). Grackle 3 refuses
           this unless the table carries the shielding cross sections (the *_shielded.h5 tables), so it is enabled only for those tables */
        if(GrackleTableHasShielding(All.GrackleDataFile)) {gc->self_shielding_method = 3;}
        else if(ThisTask == 0) {printf("Grackle: %s has no /UVBRates/CrossSections, so UV-background self-shielding is off (self_shielding_method=0); use e.g. CloudyData_UVB=HM2012_shielded.h5 to enable it\n", All.GrackleDataFile);}
    }
    if(gc->primordial_chemistry > 1) {gc->H2_self_shielding = 2;} // H2 self-shielding with a user-supplied length (Sobolev length set in CallGrackle); =1 is only valid for Cartesian grids
#ifdef METALS
    gc->metal_cooling = 1; // metal cooling using the Cloudy tables in grackle_data_file
    if(gc->primordial_chemistry > 0) {gc->dust_chemistry = 1;} // dust processes (photo-electric heating, recombination on dust, H2 on dust for primordial_chemistry>1); dust-to-gas ratio scales with Z
    gc->photoelectric_heating = 1; // spatially uniform photo-electric heating [not adjusted to the local radiation field, beware!]
    gc->photoelectric_heating_rate = 8.5e-26; // erg cm^-3 s^-1 (Tasker & Bryan 2008)
#endif

    if(local_initialize_chemistry_data(gc, &GrackleRates, &All.GrackleUnits) != GR_SUCCESS) {fprintf(stderr, "Error in local_initialize_chemistry_data (GrackleDataFile=%s).\n", All.GrackleDataFile); endrun(ENDRUNVAL);}
    if(ThisTask == 0)
    {
        grackle_version v = get_grackle_version();
        printf("Grackle Initialized: version %s (%s %s), primordial_chemistry=%d metal_cooling=%d dust_chemistry=%d, gr_float=%d bytes, table %s\n",
               v.version, v.branch, v.revision, gc->primordial_chemistry, gc->metal_cooling, gc->dust_chemistry, (int)sizeof(gr_float), All.GrackleDataFile);
    }
}


#endif  //COOL_GRACKLE
