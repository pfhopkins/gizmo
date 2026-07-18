/* ism_dust_chemistry_functions.h -- Canonical KOKKOS_INLINE_FUNCTION
 * implementations of update_dust_processes and its full hot-path call graph.
 *
 * SSOT: included by both ism_dust_chemistry.cc (which provides the host
 * external symbols via the #undef KOKKOS_INLINE_FUNCTION pattern, mirroring
 * eos.cc/eos_functions.h) and -- in Phase 2 chunk 2 -- cooling/cooling.cc so
 * the GPU post-cooling-tail kernel can call these inline.
 *
 * Chunk 1 scope: header migration only. NO call-site changes. The cooling-
 * tail call path remains host-only via the same external symbol that existed
 * pre-migration. See OPEN_post_cooling_kernel_phase2.md.
 *
 * Diagnostic helpers (ISMDustChemEvo_check_bins_after_update,
 * ISMDustChemEvo_check_yields_before_update) are NOT migrated -- they have
 * zero callers in the deferred cooling hot path and remain host-only in
 * ism_dust_chemistry.cc.
 *
 * Include order: after allvars.h (struct types + All) and proto.h.
 */
#ifndef ISM_DUST_CHEMISTRY_FUNCTIONS_H
#define ISM_DUST_CHEMISTRY_FUNCTIONS_H

/* Fallback for non-Kokkos builds. Must appear BEFORE any include that
 * uses KOKKOS_INLINE_FUNCTION, especially grain_collisional_outcomes.h. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
#ifndef KOKKOS_FUNCTION
#define KOKKOS_FUNCTION
#endif

/* Self-sufficient dependency includes so any TU that pulls this header gets
 * every symbol the migrated bodies use. Originally Chunk 1 leaned on
 * ism_dust_chemistry.cc's own include block; that left cooling.cc (Phase 2
 * chunk 2) missing gpu_rng.h for the shattering/coagulation RNG path under
 * GALSF_ISMDUSTCHEM_GRAINSIZEEVO. */
#include "../declarations/gpu_rng.h"   /* gizmo_gpu_rand_double */
#include "grain_collisional_outcomes.h"

#if defined(GALSF_ISMDUSTCHEM_MODEL)

/* Dust constants/macros -- moved here from ism_dust_chemistry.cc so the
 * inline functions below see them at every include site. The non-migrated
 * Initialize_ISMDustChem_Particle_Variables in the .cc also uses
 * GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC; the .cc includes this header before
 * any function body that needs the constants. */
#define ACCRETION_T_CUTOFF 300  /* gas-dust accretion cutoff temp (K); also used as cutoff for density enhancements of dust-dust coagulation */
#if (GALSF_ISMDUSTCHEM_MODEL & 8)
#define GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC 0.7 /* fraction of iron-dust mass locked as inclusions in silicates */
#else
#define GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC 0 /* no iron inclusions tracked */
#endif
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
#define MAXIMUM_SUBCYCLE_STEPS 200
#define ACC_SPUT_SUBCYCLE_PARAMETER 0.3
#define SHAT_COAG_SUBCYCLE_PARAMETER 0.1
#define COAGULATION_DENSITY_ENHANCEMENT 2000
#endif

/* Forward declarations for every migrated function (Chunk 1 + Phase D
 * helpers). The inline bodies below reference each other regardless of
 * source-order; proto.h intentionally does NOT carry plain host prototypes
 * for these (device-callability trap for Chunk 2's post_cooling_tail kernel
 * — a plain `void foo(...)` proto.h decl seen first would force device
 * lambda calls to bind host-only). All forward decls tagged
 * KOKKOS_INLINE_FUNCTION so the eventual inline definitions match. */
KOKKOS_INLINE_FUNCTION int ismdustchem_spec_indx_to_outcome_kind(int spec_indx);
KOKKOS_INLINE_FUNCTION void ISMDustChem_get_elem_yields_from_species_yields(double *dust_yields, double *species_yields);
KOKKOS_INLINE_FUNCTION void ISMDustChem_get_species_properties(int spec_indx, double *dust_atomic_weight, double *bulk_dens);
KOKKOS_INLINE_FUNCTION void ISMDustChem_get_species_key_elem(int spec_indx, double *dust_metallicity, int *key_elem, double *key_num_atoms, double *key_mass);
KOKKOS_INLINE_FUNCTION double Lambda_Dust_HighTemperature_Gas_ISM(int target, double T, double n_elec, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION double ISMDustChem_Return_Mass_Where_Dust_Shocked(double rho_cell_in_code_units, double Esne51_into_cell, double mass_preshock_in_code_units, double Z_cell);
KOKKOS_INLINE_FUNCTION void update_dust_processes(int i, double dtime_gyr, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void ISMDustChem_update_iron_inclusions(int i, struct particle_data *pp, struct gas_cell_data *cell);
#if !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
KOKKOS_INLINE_FUNCTION void update_dense_molecular_fields(int i, double temp, double rho, double nh0, double ne, struct particle_data *pp, struct gas_cell_data *cell);
#endif
KOKKOS_INLINE_FUNCTION void update_dust_accretion(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void update_dust_sputtering(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void update_dust_shattering_and_coagulation(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void update_dust_photodestruction(int i, double dtime_gyr, struct gas_cell_data *cell);
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
KOKKOS_INLINE_FUNCTION double get_ISMDustChemEvo_bin_mass(int i, int j, int k, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void update_ISMDustChemEvo_bin_number_and_slope(int i, int j, int k, double number_in_bin, double mass_in_bin, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void check_for_slope_limiting(int k, double bulk_dens, double *number_in_bin, double *slope_in_bin, double mass_in_bin);
KOKKOS_INLINE_FUNCTION void ISMDustChemEvo_update_bins_given_grain_size_change(int i, int j, double *bin_da, double mass_limit, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION void ISMDustChemEvo_update_bins_given_mass_change(int i, int j, double *bin_dM, double bulk_dens, struct gas_cell_data *cell);
KOKKOS_INLINE_FUNCTION double ISMDustChemEvo_explicit_shat_coag_poly(double ail, double aiu, double aic, double ajl, double aju, double ajc, double Ni, double si, double Nj, double sj);
KOKKOS_INLINE_FUNCTION double ISMDustChemEvo_fast_shat_coag_poly(int i, int spec_indx, int bin_i, int bin_j, struct gas_cell_data *cell);
void ISMDustChemEvo_precompute_poly_coeffs(void); /* host-only: runs once at init, before any device dispatch */
KOKKOS_INLINE_FUNCTION void ISMDustChemEvo_get_new_bin_N_and_slope_given_mass_change(double *bin_dM, double *bin_M, double *bin_N, double *bin_slope, double *new_bin_N, double *new_bin_slope, double bulk_dens);
#endif

/* === Phase D 2026-05-21 migrations (commit 45602b53) — preserved here unchanged.
 *     These are the original two helpers Phase D moved out of ism_dust_chemistry.cc
 *     to clear #20011-D ("host function called from __host__ __device__"). Bodies
 *     untouched. The Chunk 1 expansion below adds the rest of the
 *     update_dust_processes call graph to the same header. === */

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
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
    /* From detailed SNR simulations in Schaffler+ 2025 */
    // TBD
    /* From fits in Yamasawa+ 2011 */
    mass_shocked_in_code_units = 1535 * Esne51_into_cell / (pow(local_n0, 0.202) * pow(Z_cell/All.SolarAbundances[0]+0.039,0.298) * UNIT_MASS_IN_SOLAR);
#else
    /* Simple radiative SNR case from McKee 1989 and Cioffi 1988 */
    mass_shocked_in_code_units = 2460 * Esne51_into_cell / (pow(local_n0, 0.1) * pow(vs7, 9./7.) * UNIT_MASS_IN_SOLAR);
#endif

    return DMIN(mass_shocked_in_code_units * All.ISMDustChem_SNeGasClearedOfDustScaling, mass_preshock_in_code_units); // mass shocked limited to the entire mass of the gas particle
}

/* === Chunk 1 migrations (function bodies verbatim from ism_dust_chemistry.cc) === */

KOKKOS_INLINE_FUNCTION
int ismdustchem_spec_indx_to_outcome_kind(int spec_indx)
{
    if(spec_indx == All.ISMDustChem_Sil_Index)        { return GRAIN_OUTCOME_SPECIES_SILICATE; }
    if(spec_indx == All.ISMDustChem_Carb_Index)       { return GRAIN_OUTCOME_SPECIES_CARBON;   }
    if(spec_indx == All.ISMDustChem_FreeIron_Index ||
       spec_indx == All.ISMDustChem_InclIron_Index)   { return GRAIN_OUTCOME_SPECIES_IRON;     }
    return GRAIN_OUTCOME_SPECIES_DEFAULT;
}

KOKKOS_INLINE_FUNCTION
void ISMDustChem_get_elem_yields_from_species_yields(double *dust_yields, double *species_yields)
{
    int k,j,spec_indx;
    for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
        /******** SILICATE ********/
        if (spec_indx==All.ISMDustChem_Sil_Index) {
            for (j=0;j<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;j++) {dust_yields[All.ISMDustChem_SilicateMetallicityFieldIndexTable[j]] += species_yields[k] * All.ISMDustChem_SilicateNumberOfAtomsTable[j] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[j]] / All.ISMDustChem_EffectiveSilicateDustAtomicWeight;}
        }
        /******** CARBONACEOUS ********/
        else if (spec_indx==All.ISMDustChem_Carb_Index) {
            dust_yields[2] += species_yields[k];
        }
        /******** METALLIC IRON ********/
        else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {
            dust_yields[10] += species_yields[k];
        }
        /******** O RESERVOIR ********/
        else if (spec_indx==All.ISMDustChem_ORes_Index) {
            dust_yields[4] += species_yields[k];
        }
    }

    for(k=1;k<NUM_ISMDUSTCHEM_ELEMENTS;k++)  dust_yields[0] += dust_yields[k]; // Total fraction of yields that is dust
}

KOKKOS_INLINE_FUNCTION
void ISMDustChem_get_species_properties(int spec_indx, double *dust_atomic_weight, double *bulk_dens)
{
    *dust_atomic_weight = 1; *bulk_dens = 1;
    /******** SILICATE ********/
    if (spec_indx==All.ISMDustChem_Sil_Index) {
        *dust_atomic_weight = All.ISMDustChem_EffectiveSilicateDustAtomicWeight;
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[spec_indx];
    }
    /******** CARBONACEOUS ********/
    else if (spec_indx==All.ISMDustChem_Carb_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[2];
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[spec_indx];
    }
    /******** METALLIC IRON ********/
    else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[10];
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[All.ISMDustChem_FreeIron_Index]; // both free and inclusion iron use the metallic-iron bulk density (FreeIron slot); InclIron_Index is a species-field index (4), not an index into the size-3 SpeciesBulkDens array
    }
    /******** O RESERVOIR ********/
    else if (spec_indx==All.ISMDustChem_ORes_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[4];
        *bulk_dens = 1; // O res if a bucket for excess oxygen and doesn't correspond to an actual chemical species but setting to one here for safety
    }
}

KOKKOS_INLINE_FUNCTION
void ISMDustChem_get_species_key_elem(int spec_indx, double *dust_metallicity, int *key_elem, double *key_num_atoms, double *key_mass)
{
    int k;
    /* Entry-time defaults. Per codex 2026-05-27: on CUDA the non-silicate
     * branches below NEVER explicitly write *key_num_atoms; on host that's
     * fine because the default-init below persists, but nvc++ aggressive
     * device-side optimization can elide the store as dead-code if it
     * concludes the only downstream writer is the silicate branch. The
     * resulting register-uninitialized read gives garbage species_yields[]
     * in update_dust_accretion -> DustSpecies diverges first and
     * catastrophically (matches Vista t2 oracle signature: step 88,
     * DustSpecies=1.5e+01 while DustMetal/Source still ~5e-13). Defensive
     * per-branch reassignment below makes the function independent of the
     * entry-time init surviving any optimization. The value 1.0 is the
     * physically-correct atoms-per-formula-unit count for carbon,
     * free iron, and the O reservoir; preserves
     * pre-Phase-2-port behaviour exactly. */
    *key_elem = -1; *key_num_atoms = 1.0; *key_mass = 1.0;
    /******** SILICATE ********/
    if (spec_indx==All.ISMDustChem_Sil_Index) {
        double sil_elem_abunds[GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES] = {0.};
        for(k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++) {if (All.ISMDustChem_SilicateNumberOfAtomsTable[k] > 0) {*key_elem = k; break;}} // start with first element in silicates
        for(k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++)
        {
            if (All.ISMDustChem_SilicateNumberOfAtomsTable[k] <= 0) {continue;} // if no atoms of this element in silicate composition then skip
            int index = All.ISMDustChem_SilicateMetallicityFieldIndexTable[k];
            sil_elem_abunds[k] = dust_metallicity[index] / All.ISMDustChem_AtomicMassTable[index];
            // If an element is missing nothing else to do
            if (sil_elem_abunds[k] <= 0) {*key_elem = -1; return;}
            else if (sil_elem_abunds[*key_elem] / All.ISMDustChem_SilicateNumberOfAtomsTable[*key_elem] > sil_elem_abunds[k] / All.ISMDustChem_SilicateNumberOfAtomsTable[k]) *key_elem = k;
        }
        *key_num_atoms = All.ISMDustChem_SilicateNumberOfAtomsTable[*key_elem];
        *key_elem = All.ISMDustChem_SilicateMetallicityFieldIndexTable[*key_elem];
        *key_mass = All.ISMDustChem_AtomicMassTable[*key_elem];
    }
    /******** CARBONACEOUS ********/
    else if (spec_indx==All.ISMDustChem_Carb_Index) {if (dust_metallicity[2]>0) {*key_elem=2; *key_num_atoms=1.0; *key_mass=All.ISMDustChem_AtomicMassTable[*key_elem];}}
    /******** METALLIC IRON ********/
    else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {if (dust_metallicity[10]>0) {*key_elem=10; *key_num_atoms=1.0; *key_mass=All.ISMDustChem_AtomicMassTable[*key_elem];}}
    /******** O RESERVOIR ********/
    else if (spec_indx==All.ISMDustChem_ORes_Index) {if (dust_metallicity[4]>0) {*key_elem=4; *key_num_atoms=1.0; *key_mass=All.ISMDustChem_AtomicMassTable[*key_elem];}}
}

KOKKOS_INLINE_FUNCTION
void update_dust_processes(int i, double dtime_gyr, struct particle_data *pp, struct gas_cell_data *cell)
{
    int k; double ne=1, nh0=0, nHe0, nHepp, nhp, nHeII, temp, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
    temp = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, pp, cell);
    rho*=UNIT_DENSITY_IN_CGS;

#if !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    // Choban+22 version of the code tracks a simplified dense molecular gas and CO fraction
    update_dense_molecular_fields(i,temp,rho,nh0,ne, pp, cell);
#endif

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO) && !defined(COOL_MOLECFRAC_NONEQM)
    // Need local mach number for grain size evolution routines which is taken from nonequilibrium H2 calculations
    // If nonequilibrium H2 not used then need to calculate the mach number here
    double dx_cell = pp[i].Get_Particle_Size() * All.cf_atime; // cell size
    double v_thermal_rms = 0.111*sqrt(temp); // sqrt(3*kB*T/2*mp), in km/s; RMS thermal speed of molecular H2
    double gradv = cell[i].velocity_gradient_norm();
    double dv_turb = gradv * dx_cell * UNIT_VEL_IN_KMS; // delta-velocity across cell in km/s
    cell[i].ISMDustChem_MachNumber = dv_turb / (v_thermal_rms/sqrt(3.));
#endif

    if (cell[i].ISMDustChem_Dust_Metal[0] <= 0) {return;} // No dust so nothing more to do

    update_dust_accretion(i,dtime_gyr,temp,rho, pp, cell);

    // If gas cell has been recently shocked by SNe, delay accounting for destruction and shattering to avoid double counting
    if(cell[i].ISMDustChem_DelayTimeSNeSputtering > 0) {cell[i].ISMDustChem_DelayTimeSNeSputtering = DMAX(0,cell[i].ISMDustChem_DelayTimeSNeSputtering-dtime_gyr);}
    else {
        update_dust_sputtering(i,dtime_gyr,temp,rho, pp, cell);

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        update_dust_shattering_and_coagulation(i,dtime_gyr,temp,rho, pp, cell);

        update_dust_photodestruction(i,dtime_gyr, cell);
#endif
    }
}

#if !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
KOKKOS_INLINE_FUNCTION
void update_dense_molecular_fields(int i, double temp, double rho, double nh0, double ne, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* Choban+22 version for FIRE-2.
     * Calculate H2 fraction to determine whether gas is in the CNM/diffuse MC or dense MC phase.
     * Gas-dust accretion is assumed to have Coloumb enhancing in CNM/diffuse MC due to dust grain charge and ionized metal species.
     * In dense MC we assume no enhancing due to neutral metal species.
     * Also use dense MC fraction to determine when C is locked up in CO and thuse unavailable for gas-dust accretion.
     * We assume C rapidly converts to CO once the gas is sufficently molecular.
     */
    double fH2=0., new_ISMDustChem_MassFractionInDenseMolecular=0.; // mass fraction of gas that is H2 and gas in dense MC phase
    double NH2 = 1.5E21; // cm^-2 Column density of H2 needed to be in dense MC phase (this is a tuned value but falls within observed range for rapid C->CO conversion)
    double l_depth, x_dens; // depth into cloud to reach NH2 and radial fraction of cloud in dense MC phase
    /* Pass `pp` explicitly: proto.h declares evaluate_NH_from_GradRho with
     * `struct particle_data *pp = P` default, where P is the host extern.
     * On device, P is undefined -> nvcc error "identifier 'P' is undefined
     * in device code". Mac OMP tolerated the default; Vista CUDA caught it.
     * Surface-and-fix Phase 2 chunk 3 sweep, 2026-05-27. */
    double surface_density = evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i,pp) * UNIT_SURFDEN_IN_CGS; // converts to cgs
    // shielding length giving effective radius of gas particle
    double l_shield = surface_density / rho;
    fH2 = Get_Gas_Molecular_Mass_Fraction(i, temp, nh0, ne, 0., pp, cell);
    if (fH2 > 0)
    {
        double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS;
        l_depth = 2.*NH2 / (fH2*nHcgs);
        x_dens = (l_shield-l_depth)/l_shield;
        x_dens = DMIN(1,DMAX(0,x_dens));
        new_ISMDustChem_MassFractionInDenseMolecular = pow(x_dens,3);
        new_ISMDustChem_MassFractionInDenseMolecular = DMIN(new_ISMDustChem_MassFractionInDenseMolecular,fH2); // Maximum dense molecular fraction set by total molecular fraction
    }
    // Only need to update CO if there is C present
    if (pp[i].Metallicity[2]>0)
    {
        // If dense MC has shrunk, reduce the C in CO by the fraction it has shrunk
        if (new_ISMDustChem_MassFractionInDenseMolecular < cell[i].ISMDustChem_MassFractionInDenseMolecular) {cell[i].ISMDustChem_C_in_CO *= new_ISMDustChem_MassFractionInDenseMolecular/cell[i].ISMDustChem_MassFractionInDenseMolecular;}
        // If dense MC has grown, increase the C in CO by the newly add volume of remaining gas-phase C if any is left
        else
        {
            if (pp[i].Metallicity[2]-cell[i].ISMDustChem_Dust_Metal[2]-cell[i].ISMDustChem_C_in_CO > 0.)
            {
                cell[i].ISMDustChem_C_in_CO += (new_ISMDustChem_MassFractionInDenseMolecular-cell[i].ISMDustChem_MassFractionInDenseMolecular) * ((pp[i].Metallicity[2]-cell[i].ISMDustChem_Dust_Metal[2])-cell[i].ISMDustChem_C_in_CO) / (1.-cell[i].ISMDustChem_MassFractionInDenseMolecular);
            }
        }
    }
    else {cell[i].ISMDustChem_C_in_CO = 0.;}

    cell[i].ISMDustChem_MassFractionInDenseMolecular = new_ISMDustChem_MassFractionInDenseMolecular;
}
#endif

KOKKOS_INLINE_FUNCTION
void update_dust_accretion(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell)
{
    int j,k,spec_indx;
    double dF; // change in fraction of element condensed into dust
    double growth_timescale, t_ref, T_ref, avg_grain_radius;
    double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS] = {0.0};
    int source = 0;

    /* Restrict accretion only to MC environments by assuming sticking efficiency of 1 for
     * T <= 300K and 0 otherwise. Check the three main dust species that can form through
     * accretion in the ISM silicates, carbon, and metallic iron.
     */
    double T_cutoff; // K, temperature below which accretion is allowed
    double bulk_dens; //  mass density of the dust condensed phase (g cm^-3)
    double dust_atomic_weight; // atomic weight of one formula unit of dust species
    double species_yields[NUM_ISMDUSTCHEM_SPECIES] = {0.0};
    int key_elem;  // index of least abundant element needed to create the dust species under consideration
    double key_mass, key_num_atoms; // atomic mass of key element number of atoms in dust species
    double gas_Z[NUM_ISMDUSTCHEM_ELEMENTS]; // gas-phase metallicity of all elements (i.e. not in dust)
    double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS;
    // Determine gas-phase metallicity for each element, use this to determine the key element for each dust species
    for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {gas_Z[k]=pp[i].Metallicity[k] - cell[i].ISMDustChem_Dust_Metal[k];}

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    T_cutoff=ACCRETION_T_CUTOFF*All.ISMDustChem_AccretionTcutoffScaling;
    int k_cycle, n_subcycle;
    double dadt_ref = 1.91249E-4; // reference change in grain size in cm/Gyr assuming purely hard-sphere type encounters
    double bin_da[NUM_ISMDUSTCHEM_SIZE_BINS] = {0.};
    double key_depl, key_num_dens, mass_limit; // Limit for given species mass based on metallicity
    double sigma_squared, nH_max, nH_dense=1E3, eff_clump_factor, temp_clump_factor; // variance of log-normal density profile, max density for accretion, min density for dense mol gas, gas-dust clumping factor, and temp clumping factor
    double da1dt, dt_acc, dt_subcycle, a1_width; // grain size change in smallest bin, subcycle timestep fraction, and timestep for subcycle
    double Coulomb_enhancement[NUM_ISMDUSTCHEM_SIZE_BINS] = {0.}; // enhancement from grain charging
    double log_a_nano, fdense; // grain size in nm, dense gas mass fraction
    double D_small, D_large, a_min=0.001*1E-4, a_mid=0.01*1E-4; // small and large grain Coulomb enhanceent factor, 0.01 micron grain size cutoff between small and large

    sigma_squared = log(1+(0.5*cell[i].ISMDustChem_MachNumber)*(0.5*cell[i].ISMDustChem_MachNumber));
    temp_clump_factor = 1/sqrt(1+(0.5*cell[i].ISMDustChem_MachNumber)*(0.5*cell[i].ISMDustChem_MachNumber));
    // Assuming sticking efficiency of zero for Teff > 300 K
    // Use clumping factor here to account for gas with high mach numbers which will have effective temperatures below the cutoff
    if (temp * temp_clump_factor <= T_cutoff)
    {
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            ISMDustChem_get_species_key_elem(spec_indx, gas_Z, &key_elem, &key_num_atoms, &key_mass);

            // The maximum density for clumping enhancement and polynomial coefficients for Coulomb enhancement. Accretion halts at high densities due to C->CO conversion for carbonaceous grains or ice freeze out for all other grains. The densities for these two processes are slightly different.
            nH_max = 1E4; // This is default except for carbonaceous grains
            if (spec_indx==All.ISMDustChem_Sil_Index) {D_small=10; D_large=0.5;}
            else if (spec_indx==All.ISMDustChem_Carb_Index) {D_small=3; D_large=0;nH_max = 1E3;}
            else if (spec_indx==All.ISMDustChem_FreeIron_Index) {D_small=20; D_large=1;}
            else{D_small=1; D_large=1;}
            // Determine clumping factors for density and temperature
            // Note this an effective clumping factor which accounts for the turn off of accretion past a maximum density.
            // nH_max is set either by the typical C to CO critical density for carbonaceous dust or density at which
            // molecules freeze-out onto dust for all other species.
            eff_clump_factor = exp(sigma_squared)/2*erfc((3*sigma_squared/2 - log(nH_max/nHcgs))/(sqrt(2*sigma_squared)));
            // need all the constituent elements for the given dust species to grow
            if (key_elem != -1) {
                key_depl = cell[i].ISMDustChem_Dust_Metal[key_elem]/pp[i].Metallicity[key_elem];
                key_num_dens = rho * pp[i].Metallicity[key_elem] * (1-key_depl)/ (key_mass * PROTONMASS_CGS);
                ISMDustChem_get_species_properties(spec_indx, &dust_atomic_weight, &bulk_dens);

                // calculate Coulomb enhancement for each grain size bin
                for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {
                    // simple scaling function between expected Coulomb enhancement of small and large grains
                    // from Weingartner & Draine 2001
                    if (All.ISMDustChem_GrainBinCenters[j] <= a_min) {Coulomb_enhancement[j] = D_small;}
                    else if (All.ISMDustChem_GrainBinCenters[j] <= a_mid) {Coulomb_enhancement[j] = ((D_large-D_small)/log10(a_mid/a_min)) * log10(All.ISMDustChem_GrainBinCenters[j]/a_min) + D_small;}
                    else {Coulomb_enhancement[j] = D_large;}
                }

                // Check if we need to subcycle the timesteps by limiting the grain size change by the smallest grain size bin
                // Smallest bin width
                a1_width=All.ISMDustChem_GrainBinEdges[1]-All.ISMDustChem_GrainBinEdges[0];
                // Change in grain size for smallest bin
                da1dt = dadt_ref * (dust_atomic_weight / (key_num_atoms * sqrt(key_mass))) * key_num_dens * sqrt(temp * temp_clump_factor) / bulk_dens * Coulomb_enhancement[0] * eff_clump_factor * All.ISMDustChem_DustAccretionScaling; // change in cm/Gyr
                // Check if we need to subcycle timesteps
                dt_acc = ACC_SPUT_SUBCYCLE_PARAMETER*a1_width/da1dt;
                if (dt_acc < dtime_gyr) {n_subcycle = IMIN(MAXIMUM_SUBCYCLE_STEPS,ceil(dtime_gyr/dt_acc)); dt_subcycle = dtime_gyr/n_subcycle;}
                else {n_subcycle = 1; dt_subcycle = dtime_gyr;}

                mass_limit = pp[i].Metallicity[key_elem] * dust_atomic_weight / (key_num_atoms * key_mass) * cell[i].Mass * UNIT_MASS_IN_CGS;
                double init_species_mass = cell[i].ISMDustChem_Dust_Species[k]*cell[i].Mass*UNIT_MASS_IN_CGS, final_species_mass=0;
                for (k_cycle=0;k_cycle<n_subcycle;k_cycle++) {
                    // Need to caculate the change in grain size for every time step since the key element abundance decreases as the dust grows
                    if (k_cycle !=0) {
                        key_depl *= final_species_mass / init_species_mass;
                        if (key_depl>=1) {break;} // Catch case where we run out of key element to grow dust
                        key_num_dens = rho * pp[i].Metallicity[key_elem] * (1-key_depl) / (key_mass * PROTONMASS_CGS);
                        init_species_mass = final_species_mass;
                    }
                    for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {
                        bin_da[j] = dt_subcycle * dadt_ref * (dust_atomic_weight / (key_num_atoms * sqrt(key_mass))) * key_num_dens * sqrt(temp * temp_clump_factor) / bulk_dens * Coulomb_enhancement[j] * eff_clump_factor * All.ISMDustChem_DustAccretionScaling; // change in cm
                    }
                    ISMDustChemEvo_update_bins_given_grain_size_change(i, k, bin_da, mass_limit, cell);

                    // Get new total mass of dust species so we can update depletion in next timestep
                    final_species_mass=0;
                    for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {final_species_mass += get_ISMDustChemEvo_bin_mass(i,k,j, cell);}
                }
                // Get the final new species fractions
                for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {species_yields[k] += get_ISMDustChemEvo_bin_mass(i,k,j, cell);}
                species_yields[k] /= (cell[i].Mass * UNIT_MASS_IN_CGS); // Convert to mass fractions
            }
            else {
                species_yields[k] = cell[i].ISMDustChem_Dust_Species[k];
            }
        }
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);
        // Update dust yields and creation source
        cell[i].ISMDustChem_Dust_Source[source] += DMAX(0,dust_yields[0] - cell[i].ISMDustChem_Dust_Metal[0]);
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) cell[i].ISMDustChem_Dust_Species[k] = species_yields[k];
        for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k] = dust_yields[k];}
#else
    T_cutoff = ACCRETION_T_CUTOFF*All.ISMDustChem_AccretionTcutoffScaling;
    double max_num_dens; // max number density of key element assuming all of element is in gas
    double key_elem_DZ, key_gas_Z=0; // key element fraction locked in dust and mass fraction in the gas phase.
    // reference accretion timescales for ionized (with Coulomb enhancment) and neutral (no enhancement) gas-phase metals
    double t_ref_CNM, t_ref_MC; // reference timescales CNM and MC
    // Assuming sticking efficiency of zero for T > 300 K
    if (temp <= T_cutoff)
    {
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            ISMDustChem_get_species_key_elem(spec_indx, gas_Z, &key_elem, &key_num_atoms, &key_mass);
            t_ref = 0;
            // if missing an element all done here
            if (key_elem == -1) {key_gas_Z=0;}
            // Set some dust species specific values first
            else if (spec_indx==All.ISMDustChem_Sil_Index) {
                t_ref_CNM = 0.252E-3;   // Gyr
                t_ref_MC = 1.38E-3;     // Gyr
                t_ref = (t_ref_CNM * t_ref_MC) / (cell[i].ISMDustChem_MassFractionInDenseMolecular * t_ref_CNM + (1.-cell[i].ISMDustChem_MassFractionInDenseMolecular) * t_ref_MC) / All.ISMDustChem_DustAccretionScaling;
                key_elem_DZ = cell[i].ISMDustChem_Dust_Metal[key_elem] / pp[i].Metallicity[key_elem];
                key_gas_Z = gas_Z[key_elem];
            }
            else if (spec_indx==All.ISMDustChem_Carb_Index) {
                if (cell[i].ISMDustChem_MassFractionInDenseMolecular < 1.) {
                    // Since the transition between C+ -> C -> CO is quick, assume C+ -> CO so carbon dust only grows in CNM environments
                    // Also need to take into account C in CO reduces the maximum amount of carbon dust which can be formed
                    t_ref_CNM = 1.54E-3; // Gyr
                    t_ref = t_ref_CNM / (1.-cell[i].ISMDustChem_MassFractionInDenseMolecular) / All.ISMDustChem_DustAccretionScaling;
                    // Need to account for C locked in CO
                    key_elem_DZ = cell[i].ISMDustChem_Dust_Metal[key_elem] / (pp[i].Metallicity[key_elem] - cell[i].ISMDustChem_C_in_CO);
                    key_gas_Z = gas_Z[key_elem] - cell[i].ISMDustChem_C_in_CO;
                }
            }
            else if (spec_indx==All.ISMDustChem_FreeIron_Index) {
                // nano-particle sized or MRN-sized iron
                if(GALSF_ISMDUSTCHEM_MODEL & 8) {t_ref_CNM = 1.66E-6; t_ref_MC = 0.139E-3;} // iron is nano-sized when the iron-inclusions species (bit 8 under the current model numbering) is tracked
                else {t_ref_CNM = 0.252E-3; t_ref_MC = 1.38E-3;} // Gyr
                t_ref = (t_ref_CNM * t_ref_MC) / (cell[i].ISMDustChem_MassFractionInDenseMolecular * t_ref_CNM + (1.-cell[i].ISMDustChem_MassFractionInDenseMolecular) * t_ref_MC) / All.ISMDustChem_DustAccretionScaling;
                key_elem_DZ = cell[i].ISMDustChem_Dust_Metal[key_elem] / pp[i].Metallicity[key_elem];
                key_gas_Z = gas_Z[key_elem];
            }
            // O reservior is a special case
            else if (spec_indx==All.ISMDustChem_ORes_Index) {
                /* Observed O depletions (Jenkins 2009) cannot be explained by silicate dust alone. So
                * throw extra oxygen into a reservoir to better match observations given O depletions vs
                * number density from Whittet (2010). We assume that this reservoir only holds as much O
                * as would be needed to match this trend scaled with the amount of silicate dust vs
                * the maximum allowable amount of silicate dust in the gas. So if the maximum amount
                * of silicate dust has formed than the O depletions should exactly match with
                * observations. This scaling allows for some variability in bursty environments.
                */
                double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS;    /* hydrogen number dens in cgs units */
                double D_O = 1. - 0.65441 / pow(nHcgs,0.103725);        /* expected fractional O depletion (upper limit) */
                double max_O_in_sil;                                    /* max O depletion due to silicates */
                        double extra_O=0.;                                         /* extra O that needs to be depleted to match observations */
                double frac_of_sil;                                     /* fraction of maximum amount of silicate present in gas */
                double O_in_CO;                                         /* mass fraction of O in CO, sets max for D_O */
                O_in_CO = cell[i].ISMDustChem_C_in_CO * All.ISMDustChem_AtomicMassTable[4] / All.ISMDustChem_AtomicMassTable[2] / pp[i].Metallicity[4];
                D_O = DMAX(0.,DMIN(D_O, 1.-O_in_CO)); // set depletion upper limit to O in CO
                int sil_indx = All.ISMDustChem_Sil_Index;
                // Now determine maximum possible silicate dust based on the least abundant element
                // This roughly scales with the fraction of the key element (usually Si) depleted into dust
                ISMDustChem_get_species_key_elem(sil_indx, pp[i].Metallicity, &key_elem, &key_num_atoms, &key_mass);
                ISMDustChem_get_species_properties(sil_indx, &dust_atomic_weight, &bulk_dens);
                // No extra O if there is no silicate dust
                if (key_elem != -1) {
                    frac_of_sil = cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[sil_indx]] / (pp[i].Metallicity[key_elem] * dust_atomic_weight/(key_num_atoms * key_mass));
                    max_O_in_sil = pp[i].Metallicity[key_elem] * ((All.ISMDustChem_SilicateNumberOfAtomsTable[0] * All.ISMDustChem_AtomicMassTable[4])/(key_num_atoms * key_mass));
                    extra_O = frac_of_sil * D_O * pp[i].Metallicity[4] - max_O_in_sil - cell[i].ISMDustChem_Dust_Species[k];
                    if (extra_O>0) {species_yields[k] = extra_O;}
                }
            }

            // check to make sure we have all the constituent elements for the given dust species and it is allowed to grow
            if (t_ref > 0 && key_gas_Z > 0) {
                ISMDustChem_get_species_properties(spec_indx, &dust_atomic_weight, &bulk_dens);
                max_num_dens = rho * pp[i].Metallicity[key_elem] / (key_mass * PROTONMASS_CGS);
                growth_timescale = t_ref * (key_num_atoms * sqrt(key_mass) / dust_atomic_weight) * bulk_dens / max_num_dens / sqrt(temp);
                // change in dust condensation for key element
                dF = dtime_gyr * (1. - key_elem_DZ) * cell[i].ISMDustChem_Dust_Metal[key_elem] / growth_timescale;
                // Check in case we use up the rest of the remaining metal in the gas phase and deal with unphysical values
                dF = DMAX(0.,DMIN(key_gas_Z,dF));
                species_yields[k] = dF * (dust_atomic_weight / (key_num_atoms * key_mass));
            }
        }

        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);

        // Update dust yields and creation source
        if (dust_yields[0] != 0.)
        {
            // update dust source
            cell[i].ISMDustChem_Dust_Source[source] += dust_yields[0];
            for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) cell[i].ISMDustChem_Dust_Species[k] += species_yields[k];
            // update new dust mass
            for (k = 0; k < NUM_ISMDUSTCHEM_ELEMENTS; k++) {cell[i].ISMDustChem_Dust_Metal[k] += dust_yields[k];}
            if(GALSF_ISMDUSTCHEM_MODEL & 8) { // Update amount of free-flying iron and iron inclusions since some of the free-flying particles become inclusions in silicates. Scales with local amount of silicates
                ISMDustChem_update_iron_inclusions(i, pp, cell);
            }
        }
#endif
    }  // if (temp <= 300)
}

KOKKOS_INLINE_FUNCTION
void update_dust_sputtering(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell)
{
    // Sputtering timescales are negligable for cool gas
    if (temp>1E4) {
        int k,j,spec_indx;
        double dF; // change in fraction of element condensed into dust
        double sputter_timescale, t_ref, T_ref, avg_grain_radius;
        double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS] = {0.0};

#ifndef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
        double species_yields[NUM_ISMDUSTCHEM_SPECIES] = {0.0};
        T_ref = 2E6; /* K */ t_ref = 0.17; /* Gyr */

        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            avg_grain_radius = 0.032; /* um */
            // If assuming nano-particle iron, need to use different grain size (nano when iron-inclusions species, bit 8 under the current model numbering, is tracked)
            if ((GALSF_ISMDUSTCHEM_MODEL & 8) && spec_indx==All.ISMDustChem_FreeIron_Index) {avg_grain_radius = 0.0032; }
            sputter_timescale = t_ref * (avg_grain_radius / 0.1) / (rho*1E27) * (pow((T_ref/ temp), 2.5) + 1.) /  All.ISMDustChem_ThermalSputteringScaling;
            dF = - dtime_gyr * (cell[i].ISMDustChem_Dust_Species[k] / (sputter_timescale / 3.));
            dF = DMAX(-cell[i].ISMDustChem_Dust_Species[k],DMIN(0,dF)); // can't destroy more dust then there is available
            species_yields[k] += dF;
        }
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);

        // Update dust yields and creation source and deal with rounding errors when all dust is destroyed
        if (dust_yields[0] != 0.)
        {
            // Assume all dust sources are destroyed evenly
            for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {cell[i].ISMDustChem_Dust_Source[k] *= (1.+dust_yields[0]/cell[i].ISMDustChem_Dust_Metal[0]);}
            // Update new dust mass
            for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) cell[i].ISMDustChem_Dust_Species[k] += species_yields[k];
            // If all dust (silicates, carbonaceous, and free-flying iron) is destroyed zero everything to avoid rounding error
            int all_dest = 1;
            for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {if(cell[i].ISMDustChem_Dust_Species[k]>0 && All.ISMDustChem_TrackedSpeciesIDTable[k]!=All.ISMDustChem_InclIron_Index) {all_dest = 0; break;}}
            if (all_dest)
            {
                for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k] = 0;}
                for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {cell[i].ISMDustChem_Dust_Source[k] = 0.;}
                for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {cell[i].ISMDustChem_Dust_Species[k] = 0.;}
            }
            else {
            for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k] = DMAX(0,cell[i].ISMDustChem_Dust_Metal[k]+dust_yields[k]);}
            if(GALSF_ISMDUSTCHEM_MODEL & 8) { // Update amount of free-flying iron and iron inclusions since some of the inclusions are released as silicates are sputtered. This scales with local amount of silicates. Note if all free-flying dust is destroyed then we assume all iron inclusions are also destroyed
                    ISMDustChem_update_iron_inclusions(i, pp, cell);
                }
            }
        }
#else
        int k_cycle, n_subcycle;
        double species_yields[NUM_ISMDUSTCHEM_SPECIES] = {0.0};
        double Y_sput, dadt, dust_formula_mass, clumping_factor;
        double logt = log10(temp);
        double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS;    /* hydrogen number dens in cgs units */
        double bin_da[NUM_ISMDUSTCHEM_SIZE_BINS];
        double dt_sput, dt_subcycle, a1_width;
        clumping_factor = 1+0.5*0.5 * cell[i].ISMDustChem_MachNumber*cell[i].ISMDustChem_MachNumber;

        // Sputtering erosion rates (change in grain size per nH) for silicates, carbonaceous, and metallic iron dust from polynomial fits to Nozawa+(2006). Y=(da/dt)/nH (um/yr cm^3)
        // This is the change in grain radius over time which is independant of grain size.
        // Polynomial fits live in solids/grain_collisional_outcomes.h.

        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            Y_sput = 0;
            if (spec_indx==All.ISMDustChem_Sil_Index)            { Y_sput = grain_outcomes_sputter_erosion_dadt_per_nH(logt, GRAIN_OUTCOME_SPECIES_SILICATE); }
            else if (spec_indx==All.ISMDustChem_Carb_Index)      { Y_sput = grain_outcomes_sputter_erosion_dadt_per_nH(logt, GRAIN_OUTCOME_SPECIES_CARBON);   }
            else if (spec_indx==All.ISMDustChem_FreeIron_Index)  { Y_sput = grain_outcomes_sputter_erosion_dadt_per_nH(logt, GRAIN_OUTCOME_SPECIES_IRON);     }

            if (Y_sput > 0) {
                dadt = DMIN(0, -Y_sput * 1E-4 * nHcgs * 1E9 * clumping_factor *  All.ISMDustChem_ThermalSputteringScaling); // cm/Gyr
                // Find smallest bin with grains
                a1_width = All.ISMDustChem_GrainBinEdges[1]-All.ISMDustChem_GrainBinEdges[0];
                for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {
                    if (cell[i].ISMDustChem_Dust_NumberInBin[k][j] > 0) {
                        a1_width=All.ISMDustChem_GrainBinEdges[j+1]-All.ISMDustChem_GrainBinEdges[j];
                        break;
                    }
                }
                // Check if we need to subcycle the timesteps given the change in grain size
                dt_sput = fabs(ACC_SPUT_SUBCYCLE_PARAMETER*a1_width/dadt);
                if (dt_sput < dtime_gyr) {n_subcycle = IMIN(MAXIMUM_SUBCYCLE_STEPS,ceil(dtime_gyr/dt_sput)); dt_subcycle = dtime_gyr/n_subcycle;}
                else {{n_subcycle = 1; dt_subcycle = dtime_gyr;}}

                for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {bin_da[j] = dadt*dt_subcycle;}
                for (k_cycle=0;k_cycle<n_subcycle;k_cycle++) {
                ISMDustChemEvo_update_bins_given_grain_size_change(i, k, bin_da, 0, cell);
                }
            // Get the new species fractions
            for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {species_yields[k] += get_ISMDustChemEvo_bin_mass(i,k,j, cell);}
            species_yields[k] /= (cell[i].Mass * UNIT_MASS_IN_CGS); // Convert to mass fraction
            }
        }

        // Determine new dust element fractions and creation sources
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);

        // Assume all dust creation sources are destroyed equally
        for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {cell[i].ISMDustChem_Dust_Source[k] *= dust_yields[0]/cell[i].ISMDustChem_Dust_Metal[0];}
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {cell[i].ISMDustChem_Dust_Species[k] = species_yields[k];}
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k] = dust_yields[k];}
#endif // size evo
    } // temperature cutoff
}

KOKKOS_INLINE_FUNCTION
void update_dust_shattering_and_coagulation(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO

    // Gas cell volume (cm^-3), relative velocity between colliding grains (cm/s), mass of shattered grains (g), i, j, k grain velocities (cm/s), mach factor for grain velocities, cos theta for angle of impact between two grains
    double Vcell, vikrel, vkjrel, mshat, vgri, vgrk, vgrj, Mach=cell[i].ISMDustChem_MachNumber, cos_imp_angle, b_time_Mach, clumping_factor;
    double vgr[NUM_ISMDUSTCHEM_SIZE_BINS], vrel[NUM_ISMDUSTCHEM_SIZE_BINS][NUM_ISMDUSTCHEM_SIZE_BINS];
    double m_bin[NUM_ISMDUSTCHEM_SIZE_BINS]; // typical grain mass per bin, depends only on bulk_dens+bin geometry (fixed per species)
    double vcoag_cache[NUM_ISMDUSTCHEM_SIZE_BINS][NUM_ISMDUSTCHEM_SIZE_BINS]; // grain_outcomes_v_coag_dominik is symmetric in its two size args, precompute once per species
    double poly[NUM_ISMDUSTCHEM_SIZE_BINS][NUM_ISMDUSTCHEM_SIZE_BINS]; // interaction-rate polynomial, rebuilt each subcycle since it depends on current bin N/slope state
    double nH_cgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS; // hydrogen number dens in cgs units
    // Dust physical properties
    // shattering and coagulation thresholds (cm/s), critical pressure (dyn cm^-2), surface energy per area (dyn cm^-2), Poisson's ratio (dyn cm^-2), Young's modulus
    double vshat, vcoag, P1, gamma, nu_poisson, E_young;
    double ailower, aiupper, aicenter, ajlower, ajupper, ajcenter, aklower, akupper, akcenter;
    double mlost_shat, mgained_shat, mlost_coag, mgained_coag, total_mgained, total_mlost, miavg, mkj_shat, mkj_coag;
    double mk, mj, mej, phi, Eimp, QDstar, afmax, afmin, m_inj, mremnant, aremnant, aaggregate;
    int k, bin_i, bin_j, bin_k, spec_indx;
    int k_cycle, n_subcycle;
    double tau_coll, dMdt_moved, dNdt_moved, dt_subcycle=dtime_gyr, total_N;
    // additional clumping factor for coagulation following a power-law, used only for coagulation and not shattering
    double enh_factor=1, nH_min=0.1, nH_max=100;
    double enh_power=log10(COAGULATION_DENSITY_ENHANCEMENT * All.ISMDustChem_CoagDensityEnhancementScaling)/log10(nH_max/nH_min);
    b_time_Mach = 0.5*Mach;
    clumping_factor = 1+b_time_Mach*b_time_Mach;
    Vcell = (cell[i].Mass*UNIT_MASS_IN_CGS)/rho; // cm^3

    // Coagulation is efficient in dense MC gas (nH~10^4) which is beyond typical FIRE resolutions.
    // To overcome this we artificially enhance the density of cool gas, using the same temperature
    // cutoff as accretion, a power law density enhancement factor depending on the density, and an
    // assumed Mach number of 1 for dense gas to lower grain velocities.
    double T_cutoff = ACCRETION_T_CUTOFF*All.ISMDustChem_AccretionTcutoffScaling;
    if (temp / sqrt(clumping_factor) <= T_cutoff) {
        if (nH_cgs <= nH_min) {enh_factor=1;}
        else if (nH_cgs <= nH_max) {enh_factor = pow(nH_cgs/nH_min, enh_power);}
        else {enh_factor = COAGULATION_DENSITY_ENHANCEMENT * All.ISMDustChem_CoagDensityEnhancementScaling;}
        nH_cgs *= enh_factor;
        Vcell /= enh_factor;
        // Need to curtail Mach number for grain velocities to be below coagulation threshold
        Mach=1;
    }
    uint64_t dust_rng_key = (uint64_t)pp[i].ID, dust_rng_ctr = (uint64_t)(11 + All.NumCurrentTiStep);

    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {
        if (cell[i].ISMDustChem_Dust_Species[k] <= 0) {continue;} // No dust nothin to do
        double dM[NUM_ISMDUSTCHEM_SIZE_BINS]={0};
        spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
        double bulk_dens, dust_atomic_weight;
        ISMDustChem_get_species_properties(spec_indx, &dust_atomic_weight, &bulk_dens);
        dMdt_moved=0; dNdt_moved=0; total_N=0;

        {
            struct GrainOutcomeElasticProps __ep = grain_outcomes_elastic_props(ismdustchem_spec_indx_to_outcome_kind(spec_indx));
            vshat = __ep.v_shat; P1 = __ep.P1; gamma = __ep.gamma; E_young = __ep.E_young; nu_poisson = __ep.nu_poisson;
        }

        // Calculate turbulence driven grain velocities for each bin following prescription in Hirashita & Chen (2023)
        // Note we use the rms nH
        for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
            // Assuming supersonic turbulence for determing grain velocities
            // This is typically ~1 dex lower than assuming Kolmogorov turbulence and ultimately supresses shattering and sometimes enhancing coagulation
            vgr[bin_i] = All.ISMDustChem_GrainVelocityScaling * 0.066E5*(Mach/3)*(All.ISMDustChem_GrainBinCenters[bin_i]/1E-4)*pow(sqrt(clumping_factor)*nH_cgs/1E3,-0.5)*(bulk_dens/3.5); // cm/s
            // Assuming Kolmogorov turbulence for determing grain velocities
            //vgr[bin_i] = All.ISMDustChem_GrainVelocityScaling * 0.32E5*(Mach/3)*pow(All.ISMDustChem_GrainBinCenters[bin_i]/1E-4,0.5)*pow((temp/sqrt(clumping_factor))/100,0.25)*pow(sqrt(clumping_factor)*nH_cgs/1E3,-0.25)*pow(bulk_dens/3.5,0.5); // cm/s
        }

        // Calculate relative velocities between grain bins given random impact angle
        // Cant randomize for each term in each sum since this would change vrel for interactions between the same 2 bins

        // WARNING
        // For some reason using the global random number generator leads to undefined behaviour in the sim at large.
        // In my case the mechanical feedback routine is producing very large injection masses that don't match up with the expected yields!
        // Using a new random number generator fixes this for now. Will need to investigate.
        //cos_imp_angle = 2.0*(get_random_number((MyIDType) (pp[i].ID+5+k))-0.5);
        cos_imp_angle = 2.0*gizmo_gpu_rand_double(dust_rng_key, dust_rng_ctr++)-1.0;

        for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
            double aic_i = All.ISMDustChem_GrainBinCenters[bin_i];
            m_bin[bin_i] = 4./3.*M_PI*bulk_dens*aic_i*aic_i*aic_i; // typical grain mass in this bin
            for (bin_k=bin_i;bin_k<NUM_ISMDUSTCHEM_SIZE_BINS;bin_k++) { // vrel + vcoag are symmetric in (bin_i,bin_k): compute upper triangle once, mirror
                double aic_k = All.ISMDustChem_GrainBinCenters[bin_k];
                double vr = sqrt(vgr[bin_i]*vgr[bin_i] + vgr[bin_k]*vgr[bin_k] - 2*vgr[bin_i]*vgr[bin_k]*cos_imp_angle); // cm/s
                vrel[bin_i][bin_k] = vr; vrel[bin_k][bin_i] = vr;
                double vc = All.ISMDustChem_VCoagScaling * grain_outcomes_v_coag_dominik(aic_i, aic_k, gamma, E_young, nu_poisson, bulk_dens); // cm/s
                if (vc > vshat) vc = vshat; // Rare cases where coagualation threshold is higher than shattering threshold
                vcoag_cache[bin_i][bin_k] = vc; vcoag_cache[bin_k][bin_i] = vc;
            }
        }

        // This makes a precursory pass to determine if timestep subcycling is needed given the timestep. If subcycling is needed it sets n_subcycle and loops over them, if no subcycling is needed it updates the grain size bins and ends.
        n_subcycle=0;
        k_cycle=0;
        while (k_cycle <= n_subcycle) {
            dMdt_moved=0; dNdt_moved=0; total_N=0;
            // rebuild the interaction-rate polynomial cache each subcycle (depends on current per-bin N/slope state, which ISMDustChemEvo_update_bins_given_mass_change updates between subcycles)
            for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
                for (bin_j=0;bin_j<NUM_ISMDUSTCHEM_SIZE_BINS;bin_j++) {
                    poly[bin_i][bin_j] = ISMDustChemEvo_fast_shat_coag_poly(i, k, bin_i, bin_j, cell);
                }
            }
            for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
                ailower = All.ISMDustChem_GrainBinEdges[bin_i], aiupper = All.ISMDustChem_GrainBinEdges[bin_i+1], aicenter=All.ISMDustChem_GrainBinCenters[bin_i];
                miavg = m_bin[bin_i];
                mlost_shat = 0; mgained_shat = 0; mlost_coag = 0; mgained_coag = 0;

                for (bin_k=0;bin_k<NUM_ISMDUSTCHEM_SIZE_BINS;bin_k++) {
                    akcenter=All.ISMDustChem_GrainBinCenters[bin_k];
                    mk = m_bin[bin_k];

                    vikrel = vrel[bin_i][bin_k];
                    // Mass lost from bin i due to shattering collisions with grains in bin k
                    if (vikrel > vshat) {mlost_shat += All.ISMDustChem_ShatteringScaling * vikrel * poly[bin_i][bin_k];}
                    vcoag = vcoag_cache[bin_i][bin_k];
                    // Mass lost from bin i due to coagulating collisions with grains in bin k
                    if (vikrel <= vcoag) {mlost_coag += All.ISMDustChem_CoagulationScaling * (vikrel) * poly[bin_i][bin_k];}
                    for (bin_j=0;bin_j<NUM_ISMDUSTCHEM_SIZE_BINS;bin_j++) {
                        ajcenter=All.ISMDustChem_GrainBinCenters[bin_j];
                        mj = m_bin[bin_j]; // Typical mass of grains in bin j
                        vkjrel = vrel[bin_k][bin_j];

                        // Calculate the mass of grains injected into bin i
                        // Mass gained in bin i due to shattering collisions between grains in bin k and bin j producing fragments
                        if (vkjrel > vshat) {
                            Eimp = 0.5*(mk*mj/(mk+mj))*vkjrel*vkjrel; // impact energy betwen grains
                            QDstar = P1/(2*bulk_dens); // specific impact energy that causes more than 1/2 of mk to be disrupted
                            phi = Eimp/(mk*QDstar);
                            mej = phi/(1+phi)*mk; // total mass of material ejected and shattered from mk
                            // Assume dn_frag/da = C_frag * a^-3.3 with max and min size fragments, where C_frag is a normalization factor to recover the total ejecta mass mej
                            afmax = pow(0.02*mej/mk,1./3.)*akcenter; afmin = 0.01*afmax;
                            // Determine if there are any fragments moving into this bin based on integration bounds
                            double aintlower = ailower, aintupper = aiupper;
                            // No injected fragments of grains in this bin
                            if (afmin > aiupper || afmax < ailower) {mkj_shat=0;}
                            // Injected mass from fragments of grain into bin k
                            else {
                                // Deal with shattered grains smaller than the minimum bin by injection them into the minimum bin
                                if (bin_i==0 && afmin<ailower) {aintlower=afmin;}
                                // Catch the edges of the injected grain sizes
                                if (afmin > ailower) {aintlower = afmin;}
                                if (afmax < aiupper) {aintupper = afmax;}
                                mkj_shat = (pow(aintupper,0.7) - pow(aintlower,0.7))/(pow(afmax,0.7) - pow(afmin,0.7))*mej;
                            }
                            // Check to injected mass from remnant (if there is any) of grain in bin k after shattering
                            if (mej < mk) {
                                aremnant = DMAX(0,pow(akcenter*akcenter*akcenter - mej/(4./3.*M_PI*bulk_dens),1./3.));
                                mremnant = DMAX(0,mk-mej);
                                if (aremnant > ailower && aremnant <= aiupper) {mkj_shat += mremnant;}
                            }
                            mgained_shat += All.ISMDustChem_ShatteringScaling * vkjrel * mkj_shat * poly[bin_k][bin_j];
                        }
                        vcoag = vcoag_cache[bin_k][bin_j];
                        // Mass gained in bin i due to coagulating collisions between grains in bin k and bin j producing aggregate grains
                        if (vkjrel <= vcoag) {
                            aaggregate = pow((mk + mj)/(4*M_PI/3*bulk_dens),1./3.);
                            if (aaggregate < aiupper && aaggregate >= ailower) {mkj_coag = (mk + mj)/2;} // Counted twice so divide by 2
                            else {mkj_coag = 0;}
                            mgained_coag += All.ISMDustChem_CoagulationScaling * (vkjrel) * mkj_coag * poly[bin_k][bin_j];
                        }
                    }
                }
                // Note change in Vcell due to coagulation density enhancement only applied to coagulation mass change
                total_mlost = (mlost_coag+mlost_shat)*M_PI*miavg; // units of g/s cm^3
                total_mgained = (mgained_coag+mgained_shat)*M_PI; // units of g/s cm^3
                dM[bin_i] = (total_mgained-total_mlost)*clumping_factor/Vcell*dt_subcycle*1E9*SECONDS_PER_YEAR; // grams
                // Keep track of the net mass and number grains moved out of bins for time step subcycling check
                if (k_cycle==0) {
                    total_N += cell[i].ISMDustChem_Dust_NumberInBin[k][bin_i];
                    if (dM[bin_i] < 0) {
                        dMdt_moved-=dM[bin_i]/dt_subcycle;
                        dNdt_moved-=dM[bin_i]/miavg/dt_subcycle;
                    }
                }
            }
            // Determine if we need to subcycle timesteps if either the number or mass of grains moved out of bins is greater than epsilon_cycle fraction of the total in the particle
            if (k_cycle == 0) {
                if (dMdt_moved == 0) {break;} // If no dust moved nothing to do here
                tau_coll = SHAT_COAG_SUBCYCLE_PARAMETER * DMIN(fabs(cell[i].ISMDustChem_Dust_Species[k]*cell[i].Mass*UNIT_MASS_IN_CGS / dMdt_moved),fabs(total_N / dNdt_moved)); // Gyr
                // No sub cycling needed so we can finish
                if (tau_coll > dtime_gyr) {
                    ISMDustChemEvo_update_bins_given_mass_change(i, k, dM, bulk_dens, cell);
                }
                // Sub cycling is needed so we need to determine the number of subcycles and the timestep for each subcycle
                else {
                    n_subcycle = DMIN(MAXIMUM_SUBCYCLE_STEPS,ceil(dtime_gyr/tau_coll));
                    dt_subcycle = dtime_gyr/n_subcycle;
                }
            }
            // We are subcyling so only need to update the bins
            else {ISMDustChemEvo_update_bins_given_mass_change(i, k, dM, bulk_dens, cell);}
            k_cycle++;
        }
    }
#endif
}

KOKKOS_INLINE_FUNCTION
void update_dust_photodestruction(int i, double dtime_gyr, struct gas_cell_data *cell)
{
    // still in development so off by default
    // current implementation is too effective at destroying dust
#if defined(DUSTPHOTODESTRUCTION_TURNON)
    // If gas has been recently photoionized then dust should also be destroyed via photodestruction
    // The delay time is zeroed after recombination so it serves as a useful tracker of HII regions
    if (cell[i].DelayTimeHII != 0) {
        double dt_limited = DMIN(dtime_gyr, 0.01);// PDRs dont last longer than ~10 Myr so limit the timestep in case we dont time resolve the recombination
        // Largest grain size photodestroyed (5 nm) and typical photodestruction timescale (5 Myr)
        double a_pd = 5E-7, tau_pd = 5E-3;
        int k, j, k_cycle, n_subcycle;
        double dadt, a1_width, dt_pd, dt_subcycle;
        double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS] = {0.0};
        double species_yields[NUM_ISMDUSTCHEM_SPECIES] = {0.0};
        double bin_da[NUM_ISMDUSTCHEM_SIZE_BINS] = {0.0};
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {
            dadt = - All.ISMDustChem_PhotodestructionScaling * a_pd / tau_pd; // cm/Gyr
            // Find smallest bin with grains
            for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {
                if (cell[i].ISMDustChem_Dust_NumberInBin[k][j] > 0) {
                    a1_width=All.ISMDustChem_GrainBinEdges[j+1]-All.ISMDustChem_GrainBinEdges[j];
                    break;
                }
            }
            // Check if we need to subcycle the timesteps given the change in grain size
            dt_pd = fabs(ACC_SPUT_SUBCYCLE_PARAMETER*a1_width/dadt);
            if (dt_pd < dt_limited) {n_subcycle = IMIN(MAXIMUM_SUBCYCLE_STEPS,ceil(dt_limited/dt_pd)); dt_subcycle = dt_limited/n_subcycle;}
            else {{n_subcycle = 1; dt_subcycle = dt_limited;}}

            for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {
                if (All.ISMDustChem_GrainBinCenters[j] <= a_pd) {
                    // If the grain size is smaller than the max photodestruction size then it can be photodestroyed
                    bin_da[j] = dadt * dt_subcycle; // cm/Gyr
                }
            }
            for (k_cycle=0;k_cycle<n_subcycle;k_cycle++) {
                ISMDustChemEvo_update_bins_given_grain_size_change(i, k, bin_da, 0, cell);
            }
            // Get the new species fractions
            for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {species_yields[k] += get_ISMDustChemEvo_bin_mass(i,k,j, cell);}
            species_yields[k] /= (cell[i].Mass * UNIT_MASS_IN_CGS); // Convert to mass fraction
        }
        // Determine new dust element fractions and creation sources
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);

        // Assume all dust creation sources are destroyed equally
        for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {cell[i].ISMDustChem_Dust_Source[k] *= dust_yields[0]/cell[i].ISMDustChem_Dust_Metal[0];}
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {cell[i].ISMDustChem_Dust_Species[k] = species_yields[k];}
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k] = dust_yields[k];}
    }
#endif
}

KOKKOS_INLINE_FUNCTION
void ISMDustChem_update_iron_inclusions(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double frac_of_max_sil, incl_frac;
    int key_elem; double key_mass, key_num_atoms, dust_atomic_weight, bulk_dens;
    int sil_indx = All.ISMDustChem_Sil_Index, incl_indx = All.ISMDustChem_InclIron_Index, free_indx = All.ISMDustChem_FreeIron_Index;
    ISMDustChem_get_species_key_elem(sil_indx, pp[i].Metallicity, &key_elem, &key_num_atoms, &key_mass);
    ISMDustChem_get_species_properties(sil_indx, &dust_atomic_weight, &bulk_dens);
    if (key_elem==-1) {frac_of_max_sil=0;}
    else {frac_of_max_sil = cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[sil_indx]] / (pp[i].Metallicity[key_elem] * dust_atomic_weight/(key_num_atoms * key_mass));}
    incl_frac = DMAX(DMIN(GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC*frac_of_max_sil,GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC),0.);
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[free_indx]] = (1.-incl_frac) * cell[i].ISMDustChem_Dust_Metal[10];
    cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[incl_indx]] = incl_frac * cell[i].ISMDustChem_Dust_Metal[10];
}

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)

KOKKOS_INLINE_FUNCTION
double get_ISMDustChemEvo_bin_mass(int i, int j, int k, struct gas_cell_data *cell)
{
    if(cell[i].ISMDustChem_Dust_NumberInBin[j][k]<=0) {return 0;} // no grains
    double alower = All.ISMDustChem_GrainBinEdges[k], aupper = All.ISMDustChem_GrainBinEdges[k+1], acenter=All.ISMDustChem_GrainBinCenters[k];
    double bulk_dens, dust_atomic_weight;
    ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[j], &dust_atomic_weight, &bulk_dens);
    return DMAX(0, 4*M_PI*bulk_dens/3*((cell[i].ISMDustChem_Dust_NumberInBin[j][k]/(4*(aupper-alower))-cell[i].ISMDustChem_Dust_SlopeInBin[j][k]*acenter/4)*(pow(aupper,4)-pow(alower,4))+cell[i].ISMDustChem_Dust_SlopeInBin[j][k]/5*(pow(aupper,5)-pow(alower,5))));
}

KOKKOS_INLINE_FUNCTION
void update_ISMDustChemEvo_bin_number_and_slope(int i, int j, int k, double number_in_bin, double mass_in_bin, struct gas_cell_data *cell)
{
    double slope_in_bin;
    // Check if there is dust in the bin
    if (number_in_bin>0 && mass_in_bin>0) {

        double alower = All.ISMDustChem_GrainBinEdges[k], aupper = All.ISMDustChem_GrainBinEdges[k+1], acenter=All.ISMDustChem_GrainBinCenters[k];
        double bulk_dens, dust_atomic_weight;
        ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[j], &dust_atomic_weight, &bulk_dens);
        slope_in_bin = (3*mass_in_bin/(4*M_PI*bulk_dens)-number_in_bin/(4*(aupper-alower))*(pow(aupper,4)-pow(alower,4))) / ((pow(aupper,5)-pow(alower,5))/5-acenter/4*(pow(aupper,4)-pow(alower,4)));
        check_for_slope_limiting(k, bulk_dens, &number_in_bin, &slope_in_bin, mass_in_bin);
    }
    else {
        number_in_bin = 0;
        slope_in_bin = 0;
    }
    // Assign new number and slope
    cell[i].ISMDustChem_Dust_NumberInBin[j][k] = number_in_bin;
    cell[i].ISMDustChem_Dust_SlopeInBin[j][k] = slope_in_bin;
}

KOKKOS_INLINE_FUNCTION
void check_for_slope_limiting(int k, double bulk_dens, double *number_in_bin, double *slope_in_bin, double mass_in_bin)
{
    double alower = All.ISMDustChem_GrainBinEdges[k], aupper = All.ISMDustChem_GrainBinEdges[k+1], acenter=All.ISMDustChem_GrainBinCenters[k];
    double lower_edge, upper_edge;

    lower_edge = *number_in_bin / (aupper-alower) + *slope_in_bin * (alower-acenter);
    upper_edge = *number_in_bin / (aupper-alower) + *slope_in_bin * (aupper-acenter);
    // Large slopes can cause negative values at bin edges which are unphysical, so correct if needed.
    if (lower_edge < 0 || upper_edge < 0) {
        // To fix, conserve the mass of the bin but change the slope and number of grains so the
        // grain size distribution is zero (to machine accuracy) at the edge
        if (lower_edge < 0) {
            *number_in_bin = 15*(alower-acenter)*mass_in_bin/(M_PI*bulk_dens*(alower-aupper)*(pow(alower,3)+2*pow(alower,2)*aupper+3*alower*pow(aupper,2)+4*pow(aupper,3)));
            *slope_in_bin = 15*mass_in_bin/(M_PI*bulk_dens*(pow(alower,5)-5*alower*pow(aupper,4)+4*pow(aupper,5)));
        }
        else if (upper_edge < 0) {
            *number_in_bin = 15*(acenter-aupper)*mass_in_bin/(M_PI*bulk_dens*(alower-aupper)*(4*pow(alower,3)+3*pow(alower,2)*aupper+2*alower*pow(aupper,2)+pow(aupper,3)));
            *slope_in_bin = -15*mass_in_bin/(M_PI*bulk_dens*(4*pow(alower,5)-5*aupper*pow(alower,4)+pow(aupper,5)));
        }
    }
}

KOKKOS_INLINE_FUNCTION
void ISMDustChemEvo_update_bins_given_grain_size_change(int i, int j, double *bin_da, double mass_limit, struct gas_cell_data *cell)
{
    int l, m, m_start, m_stop;
    double x1, x2, l_low_edge, l_high_edge, m_low_edge, m_high_edge,m_center, bulk_dens, dust_atomic_weight, da, sign=0;
    double new_bin_masses[NUM_ISMDUSTCHEM_SIZE_BINS+2]={0.}, new_bin_numbers[NUM_ISMDUSTCHEM_SIZE_BINS+2]={0.};
    ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[j], &dust_atomic_weight, &bulk_dens);
    // Determine sign of change
    for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {sign += bin_da[l];}
    // Done if no grain sizes change or no dust
    if (sign==0 || cell[i].ISMDustChem_Dust_Species[j] <=0) {return;}
    // Below we determine the number of grains from bin m move into bin l given the change in grain sizes
    for(l=-1;l<NUM_ISMDUSTCHEM_SIZE_BINS+1;l++) {

        // Deal with edge cases when a grain grows beyond max grain size or shrinks below min grain size
        if (l==-1) {l_low_edge=0; l_high_edge = All.ISMDustChem_GrainBinEdges[l+1];} // can't have a grain shrink below zero
        else if (l==NUM_ISMDUSTCHEM_SIZE_BINS) {l_low_edge = All.ISMDustChem_GrainBinEdges[l]; l_high_edge = 1E10;} // grains can grow to inf size, so use an arbitrarily large size here
        else {l_low_edge = All.ISMDustChem_GrainBinEdges[l]; l_high_edge = All.ISMDustChem_GrainBinEdges[l+1];}

        // The sign of da lets us know if only bins above (larger grains shrinking) or below (smaller grains growing) should be considered
        if (sign<0) {
            if (l == -1) {m_start=0;}
            else {m_start=l;}
            m_stop=NUM_ISMDUSTCHEM_SIZE_BINS;}
        else {
            m_start=0;
            if (l == NUM_ISMDUSTCHEM_SIZE_BINS) {m_stop=NUM_ISMDUSTCHEM_SIZE_BINS;}
            else {m_stop=l+1;}
        }

        for(m=m_start;m<m_stop;m++) {
            da = bin_da[m];
            m_low_edge = All.ISMDustChem_GrainBinEdges[m]; m_high_edge = All.ISMDustChem_GrainBinEdges[m+1];
            m_center = All.ISMDustChem_GrainBinCenters[m];
            x1 = DMAX(m_low_edge,l_low_edge-da);
            x2 = DMIN(m_high_edge,l_high_edge-da);
            // If bins m and l overlap given the grain size change then update bin l grain number and mass
            // For a set grain size change a maximum of 2 bins can overlap (the bin itself and one other)
            if (x2>x1) {
                new_bin_numbers[l+1] += (cell[i].ISMDustChem_Dust_NumberInBin[j][m] * (x2-x1))/(m_high_edge-m_low_edge) + cell[i].ISMDustChem_Dust_SlopeInBin[j][m] * (1/2.*(x2*x2-x1*x1)-m_center*(x2-x1));
                // define these short hands since the mass equation is quite long
                double fm_x1 = pow(x1,5)/5 + (3*da - m_center)/4*pow(x1,4) + da*(da-m_center)*pow(x1,3) + da*da*(da-3*m_center)/2*x1*x1 - da*da*da*m_center*x1;
                double fm_x2 = pow(x2,5)/5 + (3*da - m_center)/4*pow(x2,4) + da*(da-m_center)*pow(x2,3) + da*da*(da-3*m_center)/2*x2*x2 - da*da*da*m_center*x2;
                new_bin_masses[l+1] += 4*M_PI*bulk_dens/3*(cell[i].ISMDustChem_Dust_NumberInBin[j][m] / (4*(m_high_edge-m_low_edge)) * (pow(x2+da,4)-pow(x1+da,4)) + cell[i].ISMDustChem_Dust_SlopeInBin[j][m]*(fm_x2-fm_x1));
            }

        }
    }

    // Check to make sure growing grains don't use up more metals than there are available.
    // If this happens shift all the grain bin masses down (i.e. shrink the grains a little)
    if (sign>0) {
        double total_mass = 0;
        for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS+2;l++) {total_mass += new_bin_masses[l];}
        if (total_mass>mass_limit) {
            for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS+2;l++) {new_bin_masses[l] *= mass_limit/total_mass;}
        }
    }

    // Update bin numbers and slopes given new numbers and masses and deal with edge case bins
    for(l=-1;l<NUM_ISMDUSTCHEM_SIZE_BINS+1;l++) {
        if (l !=-1 && l != NUM_ISMDUSTCHEM_SIZE_BINS) {update_ISMDustChemEvo_bin_number_and_slope(i,j,l,new_bin_numbers[l+1],new_bin_masses[l+1], cell);}
        // (case l = NUM_ISMDUSTCHEM_SIZE_BINS) for grains which grow beyond the max grain size we redistribute them back into the last grain size bin in a mass conserving manner
        else if (l==NUM_ISMDUSTCHEM_SIZE_BINS && new_bin_masses[NUM_ISMDUSTCHEM_SIZE_BINS+1]>0 && new_bin_numbers[NUM_ISMDUSTCHEM_SIZE_BINS+1]>0) {
            double avg_size, new_avg_size, new_total_mass, rebinned_number, last_bin_num, last_bin_mass, last_bin_slope;
            double new_slope_in_bin, new_number_in_bin;
            m = NUM_ISMDUSTCHEM_SIZE_BINS-1;
            m_low_edge = All.ISMDustChem_GrainBinEdges[m]; m_high_edge = All.ISMDustChem_GrainBinEdges[m+1];
            m_center = All.ISMDustChem_GrainBinCenters[m];
            last_bin_num = cell[i].ISMDustChem_Dust_NumberInBin[j][m];
            last_bin_slope = cell[i].ISMDustChem_Dust_SlopeInBin[j][m];
            last_bin_mass = get_ISMDustChemEvo_bin_mass(i,j,m, cell);
            // 1: average grain size in last bin before any rebinning
            if (cell[i].ISMDustChem_Dust_NumberInBin[j][m]>0) {
                avg_size = (m_high_edge*m_high_edge-m_low_edge*m_low_edge)/(2*(m_high_edge-m_low_edge))+last_bin_slope/last_bin_num * ((pow(m_high_edge,3)-pow(m_low_edge,3))/3 - (m_high_edge*m_high_edge-m_low_edge*m_low_edge)*m_center/2);
            }
            else {avg_size = 0;}
            // 2: number of grains to be rebinned by shrinking grains to the max grain size but conserving mass
            rebinned_number = new_bin_masses[l+1]/(4./3.*M_PI*bulk_dens*pow(m_high_edge,3));
            // 3: new average grain size in last bin after shrinking rebinned grains
            new_avg_size = (last_bin_num*avg_size + rebinned_number*m_high_edge) / (last_bin_num + rebinned_number);
            // 4: new total mass after we shift all excess mass back into last bin
            new_total_mass =  last_bin_mass + new_bin_masses[l+1];
            // 5 : new number and slope for last bin. Solved by assuming total mass before and after rebinning is conserved and determining the new average grain size
            // in the last bin when you assume the rebinned mass are all grains of the maximum grain size.
            // This results in a mess of an equation, but substituting m_center = (m_low_edge+m_high_edge)/2 makes it simpler.
            new_slope_in_bin = (-45*new_total_mass*(m_low_edge + m_high_edge - 2*new_avg_size)) /
            (pow(m_low_edge - m_high_edge,3)*(2*(m_low_edge + m_high_edge)*(pow(m_low_edge,2) + 3*m_low_edge*m_high_edge + pow(m_high_edge,2)) -
            3*(3*pow(m_low_edge,2) + 4*m_low_edge*m_high_edge + 3*pow(m_high_edge,2))*new_avg_size)*M_PI*bulk_dens);
            new_number_in_bin = (-15*new_total_mass)/(2.*(2*(m_low_edge + m_high_edge)*(pow(m_low_edge,2) + 3*m_low_edge*m_high_edge
                + pow(m_high_edge,2)) - 3*(3*pow(m_low_edge,2) + 4*m_low_edge*m_high_edge + 3*pow(m_high_edge,2))*new_avg_size)*M_PI*bulk_dens);
            // make sure to limit the new slope if necessary
            check_for_slope_limiting(m, bulk_dens, &new_number_in_bin, &new_slope_in_bin, new_total_mass);
            // Assign new number and slope
            cell[i].ISMDustChem_Dust_NumberInBin[j][m] = new_number_in_bin;
            cell[i].ISMDustChem_Dust_SlopeInBin[j][m] = new_slope_in_bin;
        }
        // (case l = -1) for grains which shrink below the min grain size we assume they are fully destroyed so nothing to do here
    }
}

KOKKOS_INLINE_FUNCTION
void ISMDustChemEvo_update_bins_given_mass_change(int i, int j, double *bin_dM, double bulk_dens, struct gas_cell_data *cell)
{
    int bin_i;
    double a_avg_new, a_avg_old, a_avg_inj, m_avg_inj, N_add, ailower, aiupper, aicenter, new_mass_in_bin, new_slope_in_bin, new_number_in_bin;
    double total_dM=0, total_pos_dM=0, total_neg_dM=0;
    double bin_M[NUM_ISMDUSTCHEM_SIZE_BINS], new_bin_slope[NUM_ISMDUSTCHEM_SIZE_BINS], new_bin_N[NUM_ISMDUSTCHEM_SIZE_BINS];
    // First ensure total mass change is zero by limiting dM
    for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
        bin_M[bin_i] = get_ISMDustChemEvo_bin_mass(i,j,bin_i, cell); // Will use this later on
        if (bin_M[bin_i] <= 0 && bin_dM[bin_i] < 0) {bin_dM[bin_i] = 0;} // catch cases where dM should be zero if mass is zero
        else if (bin_dM[bin_i] < -bin_M[bin_i]) {bin_dM[bin_i] = -bin_M[bin_i];}  // limit mass loss to total mass in bin
        if (bin_dM[bin_i]>0) {total_pos_dM+=bin_dM[bin_i];}
        else if (bin_dM[bin_i]<0) {total_neg_dM+=bin_dM[bin_i];}
    }
    total_dM = total_pos_dM+total_neg_dM;

    for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
        if (total_dM > 0 && bin_dM[bin_i]>0) {bin_dM[bin_i] *= (1 - total_dM/total_pos_dM);}
        else if (total_dM < 0 && bin_dM[bin_i]<0)  {bin_dM[bin_i] *= (1 - total_dM/total_neg_dM);}
    }

    // Now update bin number and slope
    ISMDustChemEvo_get_new_bin_N_and_slope_given_mass_change(bin_dM, bin_M, cell[i].ISMDustChem_Dust_NumberInBin[j], cell[i].ISMDustChem_Dust_SlopeInBin[j], new_bin_N, new_bin_slope, bulk_dens);
    for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
        cell[i].ISMDustChem_Dust_NumberInBin[j][bin_i] = new_bin_N[bin_i];
        cell[i].ISMDustChem_Dust_SlopeInBin[j][bin_i] = new_bin_slope[bin_i];
    }
}

KOKKOS_INLINE_FUNCTION
double ISMDustChemEvo_explicit_shat_coag_poly(double ail, double aiu, double aic, double ajl, double aju, double ajc, double Ni, double si, double Nj, double sj)
{
    /* Interaction rate between grains of bin_i and bin_j.
       An ugly polynomial but it's analytically solvable. Every term contains exactly one of
       {Ni,si} and one of {Nj,sj}: this is bilinear in (Ni,si) vs (Nj,sj), so the geometry-only
       coefficients can be precomputed once (ISMDustChemEvo_precompute_poly_coeffs) and combined
       per-particle via a 4-term dot product (ISMDustChemEvo_fast_shat_coag_poly) instead of
       re-evaluating this full expression every call. */
    double Iij = (12*(2*aiu*aiu + 3*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju))*Ni*Nj +
     6*aiu*(-2*aic*(2*aiu*aiu + 3*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju)) +
        aiu*(3*aiu*aiu + 4*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju)))*Nj*si +
     6*(-3*ajl*ajl*ajl*ajl + 2*aiu*aiu*(2*ajc - ajl - aju)*(ajl - aju) + 3*aju*aju*aju*aju + 4*ajc*(ajl*ajl*ajl - aju*aju*aju) +
        aiu*(-4*ajl*ajl*ajl + 4*aju*aju*aju + 6*ajc*(ajl - aju)*(ajl + aju)))*Ni*sj +
     aiu*(-6*aic*(ajl*(4*aiu*aiu*ajc - 2*aiu*(aiu - 3*ajc)*ajl + 4*(-aiu + ajc)*ajl*ajl - 3*ajl*ajl*ajl) -
           4*aiu*aiu*ajc*aju + 2*aiu*(aiu - 3*ajc)*aju*aju + 4*(aiu - ajc)*aju*aju*aju + 3*aju*aju*aju*aju) +
        aiu*(-9*ajl*ajl*ajl*ajl + 9*aiu*aiu*(2*ajc - ajl - aju)*(ajl - aju) + 9*aju*aju*aju*aju +
           12*ajc*(ajl*ajl*ajl - aju*aju*aju) + 8*aiu*(-2*ajl*ajl*ajl + 2*aju*aju*aju + 3*ajc*(ajl - aju)*(ajl + aju))))*si*sj
      - 9*ail*ail*ail*ail*si*(2*Nj + (2*ajc - ajl - aju)*(ajl - aju)*sj) +
     4*ail*ail*ail*si*(6*(aic - ajl - aju)*Nj + (ajl - aju)*
         (6*aic*ajc - 3*aic*(ajl + aju) - 6*ajc*(ajl + aju) + 4*(ajl*ajl + ajl*aju + aju*aju))*sj) +
     6*ail*(ajl*(6*Ni*Nj + 4*aic*aju*Nj*si) - 3*aic*ajl*ajl*ajl*ajl*si*sj - 4*ajl*ajl*ajl*(Ni - aic*ajc*si)*sj +
        2*aiu*Ni*(2*Nj + (2*ajc - ajl - aju)*(ajl - aju)*sj) + ajl*ajl*(4*aic*Nj*si + 6*ajc*Ni*sj) +
        aju*(6*Ni*Nj + 4*aic*aju*Nj*si + aju*(-6*ajc*Ni + 4*aju*Ni - 4*aic*ajc*aju*si + 3*aic*aju*aju*si)*sj)) +
     3*ail*ail*(8*Ni*Nj + 4*(2*ajc - ajl - aju)*(ajl - aju)*Ni*sj +
        si*(-4*ajl*ajl*Nj - 4*ajl*aju*Nj - 4*ajc*ajl*ajl*ajl*sj + 3*ajl*ajl*ajl*ajl*sj +
           aju*aju*(-4*Nj + (4*ajc - 3*aju)*aju*sj) +
           4*aic*(3*ajl*Nj + 3*ajc*ajl*ajl*sj - 2*ajl*ajl*ajl*sj + aju*(3*Nj + aju*(-3*ajc + 2*aju)*sj)))))/72.;

    return Iij;
}

/* ISMDustChemEvo_precompute_poly_coeffs (host-only, writes All.* config, runs once at init) is
   defined in ism_dust_chemistry.cc, NOT here — this header is included non-inline by 5 different
   .cc files (merge_split.cc, io.cc, cooling.cc, hydro_toplevel.cc, ism_dust_chemistry.cc itself),
   so a non-KOKKOS_INLINE_FUNCTION definition here would emit a strong symbol in every one of them
   (multiple-definition link error). See ism_dust_chemistry.cc for the TU that owns this symbol. */

KOKKOS_INLINE_FUNCTION
double ISMDustChemEvo_fast_shat_coag_poly(int i, int spec_indx, int bin_i, int bin_j, struct gas_cell_data *cell)
{
    double Ni = cell[i].ISMDustChem_Dust_NumberInBin[spec_indx][bin_i], si = cell[i].ISMDustChem_Dust_SlopeInBin[spec_indx][bin_i];
    double Nj = cell[i].ISMDustChem_Dust_NumberInBin[spec_indx][bin_j], sj = cell[i].ISMDustChem_Dust_SlopeInBin[spec_indx][bin_j];
    return DMAX(0, All.ISMDustChem_C_NiNj[bin_i][bin_j]*Ni*Nj + All.ISMDustChem_C_Njsi[bin_i][bin_j]*si*Nj
                 + All.ISMDustChem_C_Nisj[bin_i][bin_j]*Ni*sj + All.ISMDustChem_C_sisj[bin_i][bin_j]*si*sj); // Limit to 0 to avoid negative values due to rounding errors
}

KOKKOS_INLINE_FUNCTION
void ISMDustChemEvo_get_new_bin_N_and_slope_given_mass_change(double *bin_dM, double *bin_M, double *bin_N, double *bin_slope, double *new_bin_N, double *new_bin_slope, double bulk_dens)
{
    int bin_i;
    double a_avg_new, a_avg_old, a_avg_inj, m_avg_inj, N_add, ailower, aiupper, aicenter, new_mass_in_bin;
    // Since we only have a change in mass, we make some assumptions for the average grain size in each bin to solve for bin number and slope.
    for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
        // No change in mass then no change in bin N or slope
        if (bin_dM[bin_i] == 0) {
            new_bin_slope[bin_i] = bin_slope[bin_i];
            new_bin_N[bin_i] = bin_N[bin_i];
        }
        // All the mass removed from the bin then set bin number and slope to zero
        else if (bin_dM[bin_i] <= -bin_M[bin_i]) {
            new_bin_slope[bin_i] = 0;
            new_bin_N[bin_i] = 0;
        }
        else {
            ailower = All.ISMDustChem_GrainBinEdges[bin_i], aiupper = All.ISMDustChem_GrainBinEdges[bin_i+1], aicenter=All.ISMDustChem_GrainBinCenters[bin_i];
            // 1: Get new average grain size in the bin
            // If mass is injected determine average injected size assuming dn/da_inj ~ a^-3.3
            if (bin_dM[bin_i]>0) {
                // Assume injected grains have an average size and mass
                a_avg_inj = 1.77*(pow(aiupper,-1.3) - pow(ailower,-1.3))/(pow(aiupper,-2.3) - pow(ailower,-2.3));
                m_avg_inj = (4*M_PI*bulk_dens/3)*(-3.286*(pow(aiupper,0.7) - pow(ailower,0.7)))/(pow(aiupper,-2.3) - pow(ailower,-2.3));
                N_add = bin_dM[bin_i]/m_avg_inj;
                // Assume preexisting grains in bin have the same average size
                if (bin_N[bin_i]>0) {
                    a_avg_old = (aiupper*aiupper-ailower*ailower)/(2*(aiupper-ailower)) + bin_slope[bin_i]/bin_N[bin_i] * ((pow(aiupper,3)-pow(ailower,3))/3 - (aiupper*aiupper-ailower*ailower)*aicenter/2);
                    a_avg_new = (bin_N[bin_i] * a_avg_old + N_add * a_avg_inj) / (bin_N[bin_i] + N_add);
                }
                else {a_avg_new = a_avg_inj;}
            }
            // If mass is lost, assume average grain size stays the same
            else {a_avg_new = (aiupper*aiupper-ailower*ailower)/(2*(aiupper-ailower)) + bin_slope[bin_i]/bin_N[bin_i] * ((pow(aiupper,3)-pow(ailower,3))/3 - (aiupper*aiupper-ailower*ailower)*aicenter/2);
            }
            // 2: Determine new mass
            new_mass_in_bin = bin_M[bin_i] + bin_dM[bin_i];
            // 3 : Determine new bin number and slope from new mass and average grain size. Same as what's done for edge case rebinning.
            new_bin_slope[bin_i] = (-45*new_mass_in_bin*(ailower + aiupper - 2*a_avg_new)) /
            (pow(ailower - aiupper,3)*(2*(ailower + aiupper)*(pow(ailower,2) + 3*ailower*aiupper + pow(aiupper,2)) -
            3*(3*pow(ailower,2) + 4*ailower*aiupper + 3*pow(aiupper,2))*a_avg_new)*M_PI*bulk_dens);
            new_bin_N[bin_i] = (-15*new_mass_in_bin)/(2.*(2*(ailower + aiupper)*(pow(ailower,2) + 3*ailower*aiupper
            + pow(aiupper,2)) - 3*(3*pow(ailower,2) + 4*ailower*aiupper + 3*pow(aiupper,2))*a_avg_new)*M_PI*bulk_dens);

            // make sure to limit the new slope if necessary
            check_for_slope_limiting(bin_i, bulk_dens, &new_bin_N[bin_i], &new_bin_slope[bin_i], new_mass_in_bin);
        }
    }
}

#endif /* GALSF_ISMDUSTCHEM_GRAINSIZEEVO */

#endif /* GALSF_ISMDUSTCHEM_MODEL */
#endif /* ISM_DUST_CHEMISTRY_FUNCTIONS_H */
