#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/gpu_rng.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "grain_collisional_outcomes.h"


/* Single source of truth for update_dust_processes and its full hot-path
 * call graph lives in ism_dust_chemistry_functions.h. This .cc retains only
 * host-only helpers (Initialize_*, SNe/wind/AGB yield helpers, renormalize,
 * the three diagnostic check_* functions) plus the eos.cc-style non-inline
 * include below that emits the host external symbols for the migrated set.
 *
 * No call-site changes anywhere; the host call chain
 * (cooling_parent_routine -> host scatter loop ->
 * finish_cooling_host_deferred_dust_updates -> update_dust_processes)
 * resolves to the same external symbol it did before. */
#if defined(GALSF_ISMDUSTCHEM_MODEL)
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "ism_dust_chemistry_functions.h"
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif /* GALSF_ISMDUSTCHEM_MODEL */


/* This module collects the live ism dust chemistry modules developed by Caleb Choban in Choban et al., 2022/25.
    Written by C. Choban, reorganized and collected by PFH.
 */

#if defined(GALSF_ISMDUSTCHEM_MODEL)


#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
/* host-only: computes the 4 bin-geometry-only coefficients (via superposition at the 4 unit
   vectors, using the KOKKOS_INLINE_FUNCTION ISMDustChemEvo_explicit_shat_coag_poly from
   ism_dust_chemistry_functions.h) so ISMDustChemEvo_fast_shat_coag_poly can replace the O(N_bins^3)
   per-particle polynomial evaluation with an O(1) 4-term dot product against cached coefficients.
   Runs once, called from Initialize_ISMDustChem_Global_Variables() below, before any
   gizmo_gpu_sync_all() call pushes the populated All struct to device. Symbol owned by this TU
   (ism_dust_chemistry.cc) only -- do not add a definition to ism_dust_chemistry_functions.h,
   which is included non-inline by 5 different .cc files. */
void ISMDustChemEvo_precompute_poly_coeffs(void)
{
    int bin_i, bin_j;
    for (bin_i=0; bin_i<NUM_ISMDUSTCHEM_SIZE_BINS; bin_i++) {
        double ail = All.ISMDustChem_GrainBinEdges[bin_i], aiu = All.ISMDustChem_GrainBinEdges[bin_i+1], aic = All.ISMDustChem_GrainBinCenters[bin_i];
        for (bin_j=0; bin_j<NUM_ISMDUSTCHEM_SIZE_BINS; bin_j++) {
            double ajl = All.ISMDustChem_GrainBinEdges[bin_j], aju = All.ISMDustChem_GrainBinEdges[bin_j+1], ajc = All.ISMDustChem_GrainBinCenters[bin_j];
            /* Iij is bilinear in (Ni,si) vs (Nj,sj):
             * Iij = C_NiNj*Ni*Nj + C_Njsi*si*Nj + C_Nisj*Ni*sj + C_sisj*si*sj */
            All.ISMDustChem_C_NiNj[bin_i][bin_j] = ISMDustChemEvo_explicit_shat_coag_poly(ail,aiu,aic, ajl,aju,ajc, 1,0, 1,0);
            All.ISMDustChem_C_Njsi[bin_i][bin_j] = ISMDustChemEvo_explicit_shat_coag_poly(ail,aiu,aic, ajl,aju,ajc, 0,1, 1,0);
            All.ISMDustChem_C_Nisj[bin_i][bin_j] = ISMDustChemEvo_explicit_shat_coag_poly(ail,aiu,aic, ajl,aju,ajc, 1,0, 0,1);
            All.ISMDustChem_C_sisj[bin_i][bin_j] = ISMDustChemEvo_explicit_shat_coag_poly(ail,aiu,aic, ajl,aju,ajc, 0,1, 0,1);
        }
    }
}
#endif


/* Intializes global dust variables at startup of runs */
void Initialize_ISMDustChem_Global_Variables()
{
    int j;
    /* atomic mass for each element in metallicity field, and some other variables. these always need to be initialized */
    All.ISMDustChem_AtomicMassTable[0] = 1.01;    // H
    All.ISMDustChem_AtomicMassTable[1] = 4.0;     // He
    All.ISMDustChem_AtomicMassTable[2] = 12.01;   // C
    All.ISMDustChem_AtomicMassTable[3] = 14;      // N
    All.ISMDustChem_AtomicMassTable[4] = 15.99;   // O
    All.ISMDustChem_AtomicMassTable[5] = 20.2;    // Ne
    All.ISMDustChem_AtomicMassTable[6] = 24.305;  // Mg
    All.ISMDustChem_AtomicMassTable[7] = 28.086;  // Si
    All.ISMDustChem_AtomicMassTable[8] = 32.065;  // S
    All.ISMDustChem_AtomicMassTable[9] = 40.078;  // Ca
    All.ISMDustChem_AtomicMassTable[10] = 55.845; // Fe
    All.ISMDustChem_SNeSputteringShutOffTime = 0.3E-3; // Destruction of dust due to SNe thermal sputtering ends around 0.3 Myr after SNe (from idealized SNe in Hu+2019)
    // Fiducial olivine-pyroxene silicate dust composition with olivine fraction = 0.63 and Mg frac = 0.65. If using iron nanoparticles assume iron is always present for silicate structure in the form of iron inclusions. index in metallicity field for elements which make up silicate dust (O,Mg,Si)
    All.ISMDustChem_SilicateMetallicityFieldIndexTable[0] = 4;
    All.ISMDustChem_SilicateMetallicityFieldIndexTable[1] = 6;
    All.ISMDustChem_SilicateMetallicityFieldIndexTable[2] = 7;
    All.ISMDustChem_SilicateMetallicityFieldIndexTable[3] = 10;
    // number of O, Mg, Si, and Fe in one formula unit of silicate dust
    All.ISMDustChem_SilicateNumberOfAtomsTable[0] = 3.63;
    All.ISMDustChem_SilicateNumberOfAtomsTable[1] = 1.06;
    All.ISMDustChem_SilicateNumberOfAtomsTable[2] = 1.;
    All.ISMDustChem_SilicateNumberOfAtomsTable[3] = 0.571;
    if (GALSF_ISMDUSTCHEM_SILICATE_COMPOSITION & 2) {All.ISMDustChem_SilicateNumberOfAtomsTable[0] += 2;} // add 2 more O atoms for silicates to account for excess O depletions with no known carrier
    if (GALSF_ISMDUSTCHEM_SILICATE_COMPOSITION & 4) {All.ISMDustChem_SilicateNumberOfAtomsTable[3] += 1;} // add extra Fe to better match Fe depletions
    if (GALSF_ISMDUSTCHEM_SILICATE_COMPOSITION & 8) {All.ISMDustChem_SilicateNumberOfAtomsTable[3] = 0;} // remove all Fe from silicates (use with separate metallic iron species)
    All.ISMDustChem_EffectiveSilicateDustAtomicWeight = 0.; for(j=0;j<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;j++) {All.ISMDustChem_EffectiveSilicateDustAtomicWeight += All.ISMDustChem_SilicateNumberOfAtomsTable[j] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[j]];}

    All.ISMDustChem_Sil_Index = 0;
    All.ISMDustChem_Carb_Index = 1;
    All.ISMDustChem_FreeIron_Index = 2;
    All.ISMDustChem_ORes_Index = 3;
    All.ISMDustChem_InclIron_Index = 4;

    // Internal densities for main dust species
    All.ISMDustChem_SpeciesBulkDens[All.ISMDustChem_Sil_Index]=3.13; // g cm^-3
    All.ISMDustChem_SpeciesBulkDens[All.ISMDustChem_Carb_Index]=2.25;
    All.ISMDustChem_SpeciesBulkDens[All.ISMDustChem_FreeIron_Index]=7.86;

    for (j=0;j<NUM_ISMDUSTCHEM_SPECIES_IDS;j++) {All.ISMDustChem_SpeciesFieldIndexTable[j] = -1;}
    // silicates and carbonaceous dust are always tracked
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Sil_Index] = 0;
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Carb_Index] = 1;
    All.ISMDustChem_TrackedSpeciesIDTable[0]=All.ISMDustChem_Sil_Index;
    All.ISMDustChem_TrackedSpeciesIDTable[1]=All.ISMDustChem_Carb_Index;
    j=2; /* start index for any additional species beyond silicates and carbonaceous dust which are always tracked */
    if (GALSF_ISMDUSTCHEM_MODEL & 2)
    {
        All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_FreeIron_Index] = j;
        All.ISMDustChem_TrackedSpeciesIDTable[j]=All.ISMDustChem_FreeIron_Index;
        j++;
    }
    if (GALSF_ISMDUSTCHEM_MODEL & 4)
    {
        All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_ORes_Index] = j;
        All.ISMDustChem_TrackedSpeciesIDTable[j]=All.ISMDustChem_ORes_Index;
        j++;
    }
    if (GALSF_ISMDUSTCHEM_MODEL & 8)
    {
        All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_InclIron_Index] = j;
        All.ISMDustChem_TrackedSpeciesIDTable[j]=All.ISMDustChem_InclIron_Index;
        j++;
    }
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
    All.ISMDustChem_GrainBinSize = pow(10,log10(All.ISMDustChem_Grain_Size_Max/All.ISMDustChem_Grain_Size_Min)/NUM_ISMDUSTCHEM_SIZE_BINS);
    for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS+1;j++) {All.ISMDustChem_GrainBinEdges[j] = pow(All.ISMDustChem_GrainBinSize,j)*All.ISMDustChem_Grain_Size_Min;}
    for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {All.ISMDustChem_GrainBinCenters[j] = (All.ISMDustChem_GrainBinEdges[j+1]+All.ISMDustChem_GrainBinEdges[j])/2.;}
    ISMDustChemEvo_precompute_poly_coeffs(); /* precompute coag/shat polynomial coefficients following bin edges/centers init above; must run before gizmo_gpu_sync_all() pushes All to device */
#endif
}


/* initialize values of particle fields for startup of runs */
void Initialize_ISMDustChem_Particle_Variables(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    int j,k;
    /* only initialize these on a new run or snapshot restart without dust */
#if defined(IO_DUST_NOT_IN_ICFILE)
    if(RestartFlag == 0 || RestartFlag == 2) {
#else
    if(RestartFlag == 0) {
#endif
        cell[i].ISMDustChem_DelayTimeSNeSputtering = 0;
#if !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        cell[i].ISMDustChem_C_in_CO = cell[i].ISMDustChem_MassFractionInDenseMolecular = 0.;
#endif
        double temp_cutoff=1E5, ne=1, nh0=0, nhp=0, temp, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
        temp = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, pp, cell);
        if(All.Initial_ISMDustChem_Depletion > 0 && temp < temp_cutoff)
        {
            for(j=0;j<NUM_ISMDUSTCHEM_ELEMENTS;j++) {cell[i].ISMDustChem_Dust_Metal[j] = 0.;}
            double sil_mass_frac=0., spec_indx;
            for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
                spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[j];
                if (spec_indx==All.ISMDustChem_Sil_Index) {
                    // Silicate dust
                    cell[i].ISMDustChem_Dust_Metal[7] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[7]; // Set Si depletion
                    sil_mass_frac+=cell[i].ISMDustChem_Dust_Metal[7];
                    for(k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++) {
                        // Set element depletions for all other elements in silicates given initial Si depletion
                        if(All.ISMDustChem_SilicateMetallicityFieldIndexTable[k] != 7 && All.ISMDustChem_SilicateNumberOfAtomsTable[k] > 0) { // if this element is in silicate composition and not Si itself then set depletion based on Si depletion and silicate stoichiometry
                            cell[i].ISMDustChem_Dust_Metal[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]] += cell[i].ISMDustChem_Dust_Metal[7] / (All.ISMDustChem_SilicateNumberOfAtomsTable[2] * All.ISMDustChem_AtomicMassTable[7]) * (All.ISMDustChem_SilicateNumberOfAtomsTable[k] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]]);
                            sil_mass_frac += cell[i].ISMDustChem_Dust_Metal[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]];
                        }
                    }
                    cell[i].ISMDustChem_Dust_Species[j] = sil_mass_frac;
                }
                else if (spec_indx==All.ISMDustChem_Carb_Index) {
                    // Carbonaceous dust
                    cell[i].ISMDustChem_Dust_Metal[2] = DMIN(pp[i].Metallicity[2],sil_mass_frac/All.Initial_ISMDustChem_SiliconToCarbonRatio);
                    cell[i].ISMDustChem_Dust_Species[j] = cell[i].ISMDustChem_Dust_Metal[2];
                }
                else if (spec_indx==All.ISMDustChem_FreeIron_Index) {
                    // Free-flying iron
                    cell[i].ISMDustChem_Dust_Metal[10] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[10];
                    cell[i].ISMDustChem_Dust_Species[j] = (1.-GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC)*cell[i].ISMDustChem_Dust_Metal[10];
                }
                else if (spec_indx==All.ISMDustChem_InclIron_Index) {
                    cell[i].ISMDustChem_Dust_Species[j] = GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC*cell[i].ISMDustChem_Dust_Metal[10];
                }
            }
            for (j=1;j<NUM_ISMDUSTCHEM_ELEMENTS;j++) {cell[i].ISMDustChem_Dust_Metal[0] += cell[i].ISMDustChem_Dust_Metal[j];}
            for (j=0;j<NUM_ISMDUSTCHEM_SOURCES;j++) {cell[i].ISMDustChem_Dust_Source[j] = 0.;}
            cell[i].ISMDustChem_Dust_Source[2] = cell[i].ISMDustChem_Dust_Metal[0];  // Assume initial dust population is from SNe II
        }
        else
        {
            for (j=0;j<NUM_ISMDUSTCHEM_ELEMENTS;j++) {cell[i].ISMDustChem_Dust_Metal[j] = 0.;}
            for (j=0;j<NUM_ISMDUSTCHEM_SOURCES;j++) {cell[i].ISMDustChem_Dust_Source[j] = 0.;}
            for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {cell[i].ISMDustChem_Dust_Species[j] = 0.;}
        }
#if (defined(RADTRANSFER) && defined(RT_INFRARED)) || (defined(OUTPUT_DUST_TEMPERATURE) && (GALSF_FB_FIRE_STELLAREVOLUTION > 2))
        cell[i].Dust_Temperature = DMIN(All.InitGasTemp,100.);
#endif
    }
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    Initialize_ISMDustChemEvo_Particle_Variables(i, pp, cell);
#endif
}

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
void Initialize_ISMDustChemEvo_Particle_Variables(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    int j,k,l;
#if defined(IO_DUST_NOT_IN_ICFILE)
    if(RestartFlag == 0 || RestartFlag == 2) {
#else
    if(RestartFlag == 0) {
#endif
        cell[i].ISMDustChem_MachNumber = 0;
        if(All.Initial_ISMDustChem_Depletion > 0 && cell[i].ISMDustChem_Dust_Metal[0] > 0)
        {
            // Assume MRN powerlaw size distribution
            double powerlaw = -3.5;
            for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
                double bulk_dens, dust_atomic_weight;
                ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[j], &dust_atomic_weight, &bulk_dens);
                // Determine normalization constant for grain size distribution given total mass of dust species
                double C_norm = (cell[i].ISMDustChem_Dust_Species[j]*cell[i].Mass*UNIT_MASS_IN_CGS)*(12+3*powerlaw) / (4 * M_PI * bulk_dens * (pow(All.ISMDustChem_Grain_Size_Max,4+powerlaw)-pow(All.ISMDustChem_Grain_Size_Min,4+powerlaw)));
                for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {
                    double alower = All.ISMDustChem_GrainBinEdges[k], aupper = All.ISMDustChem_GrainBinEdges[k+1];
                    double mass_in_bin, number_in_bin;
                    mass_in_bin = 4*M_PI*bulk_dens/(3*(4+powerlaw))*C_norm*(pow(aupper,4+powerlaw)-pow(alower,4+powerlaw));
                    number_in_bin = C_norm/(powerlaw+1)*(pow(aupper,powerlaw+1) - pow(alower,powerlaw+1));
                    update_ISMDustChemEvo_bin_number_and_slope(i,j,k,number_in_bin,mass_in_bin, cell);
                }
            }
        }
        else
        {
            for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {cell[i].ISMDustChem_Dust_NumberInBin[j][k]=0;cell[i].ISMDustChem_Dust_SlopeInBin[j][k]=0;}}
        }
    }
#if !defined(IO_DUST_NOT_IN_ICFILE)
    // Simulations track the number and slope and not the mass of dust in each bin, but only the mass and number are
    // saved in the snapshot. Need to recalculate the slope from the number and mass in each bin here
    if(RestartFlag == 2) {
        for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
            double bin_number, bin_mass;
            for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {
                bin_number = cell[i].ISMDustChem_Dust_NumberInBin[j][k];
                bin_mass = cell[i].ISMDustChem_Dust_SlopeInBin[j][k];
                update_ISMDustChemEvo_bin_number_and_slope(i,j,k,bin_number,bin_mass, cell);
            }
        }
    }
#endif
}
#endif


/* Converts species yields/mass fractions to element yields/mass fractions */


// Get general properties of given dust species


// Determine the key element for the given dust species (i.e. the least abudant element that comprises the dust species)
//  key_elem will be -1 if you are missing an element


/* Lambda_Dust_HighTemperature_Gas_ISM migrated to
 * solids/ism_dust_chemistry_functions.h as KOKKOS_INLINE_FUNCTION
 * (Phase D 2026-05-21 #20011-D fix — called from CoolingRate which is
 * KOKKOS_INLINE_FUNCTION). */


/* routine to give yields for dust for different types of SNe (Ia & II) followed in-code */
void ISMDustChem_get_SNe_dust_yields(double *yields, int i, double t_gyr, int SNeIaFlag, double Msne, struct particle_data *pp, struct gas_cell_data *cell)
{
    double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS]={0}, species_yields[NUM_ISMDUSTCHEM_SPECIES]={0}; double SNeIa_age = 0.03753; int j,k,spec_indx,source_key=1;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    SNeIa_age =  0.044;
#endif
    if(t_gyr < SNeIa_age) {source_key=2;} // 1=1a, 2=II
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY]=0;} // initialize yields to null
    double SNeII_sil_cond = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.2),
           SNeII_C_cond   = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.2),
           SNeII_Fe_cond  = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.2),
           SNeI_Fe_cond   = DMIN(1,All.ISMDustChem_SNeIaDustScaling*0.005),
           SNeII_cond;
    double key_num_atoms,key_mass,dust_atomic_weight,bulk_dens;
    int key_elem;
    // For each dust species find the key element and condense a fraction of that element into dust
    if(t_gyr < SNeIa_age)
    {
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            ISMDustChem_get_species_key_elem(spec_indx, yields, &key_elem, &key_num_atoms, &key_mass);
            // check to make sure we have all the constituant elements for the given dust species
            if (key_elem != -1) {
                if (spec_indx==All.ISMDustChem_Sil_Index) {SNeII_cond=SNeII_sil_cond;}
                else if (spec_indx==All.ISMDustChem_Carb_Index) {SNeII_cond=SNeII_C_cond;}
                else if (spec_indx==All.ISMDustChem_FreeIron_Index) {SNeII_cond=SNeII_Fe_cond;}
                else {SNeII_cond=0;}

                ISMDustChem_get_species_properties(spec_indx, &dust_atomic_weight, &bulk_dens);
                species_yields[k] = SNeII_cond * yields[key_elem] * dust_atomic_weight / (key_num_atoms * key_mass);
            }
        }
    }
    else
    {
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            if (spec_indx==All.ISMDustChem_FreeIron_Index) {
                // Only a little bit of metallic iron dust from SNIa
                species_yields[k] = SNeI_Fe_cond * yields[10];
    }
        }
    }

    ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY]=dust_yields[k];}
    yields[ISMDUSTCHEM_DUST_SOURCE_OFFSET_IN_METALLICITY+source_key] = dust_yields[0]; // total yield goes to the source term of this type
    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+ISMDUSTCHEM_DUST_SPECIES_OFFSET_IN_METALLICITY]=species_yields[k];}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    ISMDustChemEvo_get_SNe_dust_grain_size_yields(yields,i,SNeIaFlag,Msne, pp, cell); // get dust grain size/mass yields
#endif
}


/* routine to give the dust yields for AGB winds (currently no dust yield assumed for stars younger than AGB age from continuous mass-loss, i.e. O/B winds) */
void ISMDustChem_get_wind_dust_yields(double *yields, int i, struct gas_cell_data *cell)
{
    double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS]={0}, species_yields[NUM_ISMDUSTCHEM_SPECIES]={0}; int j,k,spec_indx,source_key=3;
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY]=0;} // initialize yields to null
    double transition_age = 0.03753, star_age = evaluate_stellar_age_Gyr(i); // Assume AGB dust production stars at SNe II to SNe Ia transition. This limits AGB stars with mass < ~8 solar masses
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    transition_age =  0.044;
#endif
    if(star_age <= transition_age) {return;} // no yield here if too young, otherwise continue

    // Simple AGB yields routine based solely on C to O ratios
    // double condens_eff = DMIN(1,All.ISMDustChem_AGBDustScaling*0.8);
    // if((yields[2]/All.ISMDustChem_AtomicMassTable[2])/(yields[4]/All.ISMDustChem_AtomicMassTable[4]) > 1.0) // AGB stars with abundace ratio C/O > 1 only produce carbonacous dust
    // {
    //     dust_yields[2] = yields[2] - 0.75*yields[4]; dust_yields[0] = dust_yields[2]; // C
    // } else { // AGB stars with abundance C/O < 1 produce general silicate dust
    //     dust_yields[6] = condens_eff * yields[6]; // Mg
    //     dust_yields[7] = condens_eff * yields[7]; // Si
    //     dust_yields[10] = condens_eff * yields[10]; // Fe
    //     dust_yields[4] = 16 * (dust_yields[6]/All.ISMDustChem_AtomicMassTable[6] + dust_yields[7]/All.ISMDustChem_AtomicMassTable[7] + dust_yields[10]/All.ISMDustChem_AtomicMassTable[10]); // O
    //     // Check to make sure we dont produce too much O dust given the leftover O dust not in CO
    //     if (dust_yields[4] > yields[4]-(4./3.*yields[2])) {dust_yields[4] = yields[4]-(4./3.*yields[2]);}
    //     for(k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {dust_yields[0]+=dust_yields[k];}
    // }
    // for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY]=dust_yields[k];}
    // yields[ISMDUSTCHEM_DUST_SOURCE_OFFSET_IN_METALLICITY+source_key] = dust_yields[0]; // total yield goes to the source term of this type
    // return; // end routine
    double dt,Z,elem_yield,wind_rate;
    dt=get_particle_feedback_timestep_in_physical(i, P)*UNIT_TIME_IN_GYR;
    Z = Z_for_stellar_evol(i);
    // Take difference in cumulative dust production between start and end time to get estimate of instantaneous dust injection rate (M_solar/Gyr)
    double total_dust=0;
    for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
        species_yields[k] = (cumulative_AGB_dust_returns(spec_indx,(star_age+dt)*1E3,Z)-cumulative_AGB_dust_returns(spec_indx,star_age*1E3,Z))/dt;
        species_yields[k] = DMAX(0.,All.ISMDustChem_AGBDustScaling*species_yields[k]); // Deal with unphysical values which can result near boundaries in fit
        total_dust+=species_yields[k];
    }
    // All done if no dust is produced
    if (total_dust>0.)
    {
        // Now convert from instantaneous dust injection rates to dust yields using instantaneous wind rate
        wind_rate=0.41987*pow(star_age,-1.1)/(12.9-log(star_age));
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        double f_agb=0.1, t_agb=0.8, x_agb=t_agb/DMAX(star_age,1.e-4); x_agb*=x_agb; wind_rate = f_agb * pow(x_agb,0.8) * (exp(-DMIN(50.,x_agb*x_agb*x_agb)) + 1./(100. + x_agb)); /* only need AGB component for FIRE-3 */
#endif
        if(star_age < 0.033) {wind_rate *= 0.01 + calculate_relative_light_to_mass_ratio_from_imf(star_age,i,1);} // late-time independent of massive stars
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
        wind_rate *= All.StellarMassLoss_Rate_Renormalization;
#endif
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {species_yields[k] = DMAX(0.,species_yields[k]/wind_rate);}
        // Now check to make sure there are enough metals in the wind to produce the dust since the metal and dust yields are calculated separately
        // If not renorm dust species which are made up of the element in question
        // First need to add up the total dust yields for each element from the tracked dust species
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);
        for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {
            // More of this element is used up by dust then is available in the wind!
            if (dust_yields[k]>yields[k]) {
                // Check each dust species so that we decrease the yields for all dust species which are composed of the given element
                // Since we do this all in one go, if multiple elements all from one dust species are over the limit then we will decrease that
                // dust species yield multiple times instead of once, but this shouldn't have much of an effect.
                for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
                    spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[j];
                    // C
                    if (k==2 && (spec_indx==All.ISMDustChem_Carb_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // O
                    else if (k==4 && (spec_indx==All.ISMDustChem_Sil_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // Mg
                    else if (k==6 && (spec_indx==All.ISMDustChem_Sil_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // Si
                    else if (k==7 && (spec_indx==All.ISMDustChem_Sil_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // Fe with special check if its in both silicates and metallic iron or just metallic iron
                    else if (k==10 && ((spec_indx==All.ISMDustChem_FreeIron_Index) || (All.ISMDustChem_SilicateMetallicityFieldIndexTable[3]>0 && (spec_indx==All.ISMDustChem_Sil_Index)))) {
                        species_yields[j] *= yields[k]/dust_yields[k];
                    }
                }
                dust_yields[k] = yields[k];
            }
        }
        // Recalculate total dust yields after renorm
        dust_yields[0]=0;
        for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {dust_yields[0] += dust_yields[k];}
    }
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY]=dust_yields[k];}
    yields[ISMDUSTCHEM_DUST_SOURCE_OFFSET_IN_METALLICITY+source_key] = dust_yields[0]; // total yield goes to the source term of this type
    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+ISMDUSTCHEM_DUST_SPECIES_OFFSET_IN_METALLICITY]=species_yields[k];}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO) && defined(GALSF_FB_FIRE_STELLAREVOLUTION)
    /* Wind dust grain-size yields scale with stellar mass return per step,
     * which is tracked in P[i].MassReturn_ThisTimeStep only under
     * GALSF_FB_FIRE_STELLAREVOLUTION (see declarations/particle_data.h:121).
     * Without stellar-evolution tracking there are no continuous wind yields
     * to apportion across grain-size bins, so skip the call rather than
     * compile-fail. Previously the inner ifdef gated only on GRAINSIZEEVO,
     * which made any GRAINSIZEEVO Config without STELLAREVOLUTION fail to
     * build (latent since the field was introduced). */
    ISMDustChemEvo_get_wind_dust_grain_size_yields(yields,P[i].Mass*P[i].MassReturn_ThisTimeStep); // get dust grain size/mass yields (i is a star-particle index; MassReturn_ThisTimeStep lives on particle_data, not gas_cell_data) //
#endif
}



/* Simple fit to cumulative AGB dust production for a Kroupa IMF stellar population only with specific metallicities and stellar ages (assuming stars become AGBs at the ends of the main sequence lifetime) derived from AGB dust creation data table in Zhukovska+(08) */
double specific_Z_AGB_dust(int spec_indx, double star_age, int z_bound)
{
    /* spec_indx: silicate, carbon, metallic iron */
    /* z_bound: 0 = 2*Z_solar, 1 = Z_solar, 2 = 0.4*Z_solar, 3 = 0.2*Z_solar, 4 = 0.05*Z_solar*/
    double cum_return=0;
    double logt = log10(star_age);
    if (spec_indx==All.ISMDustChem_Sil_Index) {
            if (z_bound == 0) {
                if (star_age < 284) {cum_return = 1.77E-4*logt - 2.87E-4;}
                else if (star_age < 1244) {cum_return = 3.03E-5*(logt - 2.45) + 1.47E-4;}
                else {cum_return = 5.93E-4*(logt - 3.09) + 1.67E-4;}
            }
            else if (z_bound == 1) {
                if (star_age < 295) {cum_return = 4.18E-5*logt - 6.78E-5;}
                else if (star_age < 1808) {cum_return = 1.72E-6*(logt - 2.47) + 3.55E-5;}
                else {cum_return = 1.05E-4*(logt - 3.26) + 3.68E-5;}
            }
            else if (z_bound == 2) {
                if (star_age < 286) {cum_return = 5.00E-6*logt - 8.01E-6;}
                else if (star_age < 3948) {cum_return = 1.85E-9*(logt - 2.46) + 4.25E-6;}
                else {cum_return = 3.35E-7*(logt - 3.60) + 4.26E-6;}
            }
            else if (z_bound == 3) {
                if (star_age < 269) {cum_return = 1.04E-8*logt - 1.64E-08;}
                else if (star_age < 1560) {cum_return = 2.69E-10*(logt - 2.43) + 8.82E-9;}
                else {cum_return = 3.05E-19*(logt - 3.19) + 9.03E-9;}
            }
            else if (z_bound == 4) {
                if (star_age < 147) {cum_return = 5.80E-11*logt - 9.10E-11;}
                else if (star_age < 252) {cum_return = 4.10E-11*(logt - 2.17) + 3.47E-11;}
                else {cum_return = 6.05E-14*(logt - 2.40) + 4.43E-11;}
            }
    }
    else if (spec_indx==All.ISMDustChem_Carb_Index) {
            if (z_bound == 0) {
                if (star_age < 262) {cum_return = 8.10E-7*logt - 1.54E-6;}
                else if (star_age < 840) {cum_return = 4.00E-4*(logt - 2.42) + 4.23E-7;}
                else {cum_return = 7.37E-6*(logt - 2.92) + 2.02E-4;}
            }
            else if (z_bound == 1) {
                if (star_age < 305) {cum_return = 3.66E-6*logt - 6.75E-6;}
                else if (star_age < 1250) {cum_return = 3.71E-4*(logt - 2.48) +  2.34E-6;}
                else {cum_return = 8.67E-6*(logt - 3.10) + 2.29E-4;}
            }
            else if (z_bound == 2) {
                if (star_age < 367) {cum_return = 1.27E-5*logt - 2.34E-5;}
                else if (star_age < 2329) {cum_return = 4.24E-4*(logt - 2.56) + 9.27E-6;}
                else {cum_return = 7.55E-5*(logt - 3.37) + 3.49E-4;}
            }
            else if (z_bound == 3) {
                if (star_age < 344) {cum_return = 1.40E-5*logt - 2.53E-5;}
                else if (star_age < 3105) {cum_return = 4.44E-4*(logt - 2.54) + 1.03e-5;}
                else {cum_return = 9.59e-5*(logt - 3.49) + 4.35E-4;}
            }
            else if (z_bound == 4) {
                if (star_age < 280) {cum_return = 8.48E-6*logt - 1.43E-5;}
                else if (star_age < 4504) {cum_return = 3.10E-4*(logt - 2.45) + 6.47E-6;}
                else {cum_return = 5.73E-5*(logt - 3.65) + 3.80E-4;}
            }
    }
    else if (spec_indx==All.ISMDustChem_FreeIron_Index) {
            if (z_bound == 0) {
                if (star_age < 525) {cum_return = 5.98E-6*logt - 9.97E-6;}
                else if (star_age < 1108) {cum_return = 1.02E-4*(logt - 2.72) + 6.30E-6;}
                else {cum_return = 5.98E-5*(logt - 3.04) + 3.94E-5;}
            }
            else if (z_bound == 1) {
                if (star_age < 339) {cum_return = 2.16E-6*logt - 3.60E-6;}
                else if (star_age < 1074) {cum_return = 4.09E-7*(logt - 2.53) + 1.87E-6;}
                else {cum_return = 1.50E-5*(logt - 3.03) + 2.08E-6;}
            }
            else if (z_bound == 2) {
                if (star_age < 307) {cum_return = 9.86E-7*logt - 1.62E-6;}
                else if (star_age < 4120) {cum_return = 1.13e-9*(logt - 2.49) + 8.34E-7;}
                else {cum_return = 2.23E-7*(logt - 3.61) + 8.36E-07;}
            }
            else if (z_bound == 3) {
                if (star_age < 253) {cum_return = 5.53E-9*logt - 8.71E-9;}
                else if (star_age < 297) {cum_return = 2.50E-9*(logt - 2.40) +  4.57E-9;}
                else {cum_return = 9.50E-12 *(logt - 2.47) + 4.75E-9;}
            }
            else if (z_bound == 4) {
                if (star_age < 128) {cum_return = 3.18E-11*logt - 5.00E-11;}
                else if (star_age < 269) {cum_return = 2.69E-11*(logt - 2.11) + 1.70E-11;}
                else {cum_return = 2.14E-14*(logt - 2.43) + 2.57E-11;}
            }
    }
    else {cum_return = 0.;}

    if (cum_return < 0.) {cum_return = 0.;} // catch case were some fits are negative near the beginning of the AGB start time
    return cum_return;
}


/* Simple fit to cumulative AGB dust mass returns for a stellar population*/
double cumulative_AGB_dust_returns(int dust_type, double star_age, double z)
{
    /* dust_type: silicate, carbon, metallic iron */
    double cumulative_mass;
    if (z <= 0.05)     {cumulative_mass = specific_Z_AGB_dust(dust_type,star_age,4);}
    else if (z <= 0.2) {cumulative_mass = specific_Z_AGB_dust(dust_type,star_age,4) + (z-0.05)/(0.2-0.05)*(specific_Z_AGB_dust(dust_type,star_age,3) - specific_Z_AGB_dust(dust_type,star_age,4));}
    else if (z <= 0.4) {cumulative_mass = specific_Z_AGB_dust(dust_type,star_age,3) + (z-0.2)/(0.4-0.2)*(specific_Z_AGB_dust(dust_type,star_age,2) - specific_Z_AGB_dust(dust_type,star_age,3));}
    else if (z <= 1.)  {cumulative_mass = specific_Z_AGB_dust(dust_type,star_age,2) + (z-0.4)/(1.-0.4)*(specific_Z_AGB_dust(dust_type,star_age,1) - specific_Z_AGB_dust(dust_type,star_age,2));}
    else if (z <= 2.)  {cumulative_mass = specific_Z_AGB_dust(dust_type,star_age,1) + (z-1.)/(2.-1.)*(specific_Z_AGB_dust(dust_type,star_age,0) - specific_Z_AGB_dust(dust_type,star_age,1));}
    else               {cumulative_mass = specific_Z_AGB_dust(dust_type,star_age,0);}
    return cumulative_mass;
}


#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)

/* routine to give grain number and mass yields for dust for different types of SNe (Ia & II) followed in-code */
void ISMDustChemEvo_get_SNe_dust_grain_size_yields(double *yields, int i, int SNeIaFlag, double Msne, struct particle_data *pp, struct gas_cell_data *cell)
{
    int k,l;
    double number_yields[NUM_ISMDUSTCHEM_SPECIES][NUM_ISMDUSTCHEM_SIZE_BINS]={0}, mass_yields[NUM_ISMDUSTCHEM_SPECIES][NUM_ISMDUSTCHEM_SIZE_BINS]={0};
    // Assume initial grain size distribution (dn/da) with (1) a log-normal distribution centered at large radii and (2) a power-law tail to smaller grain sizes following results from Kirchschlager et al. (2019, 2020)
    double a0=0.1E-4, sigma=0.2, a_cut=0.1E-4, gamma=3.5, a_min=All.ISMDustChem_Grain_Size_Min; // center (cm) and standard deviation of log-normal distribution, intersect of the two distributions (cm), power for power law, and minimum size for power law (cm).
    double bulk_dens,dust_atomic_weight,C1_norm,C2_norm,high_edge,low_edge,bin_number,bin_mass; // grain mass density and normalization set to match total dust species mass returned
    double total_Msne = pp[i].SNe_ThisTimeStep * (Msne/UNIT_MASS_IN_SOLAR); // need the total mass returned to get the total dust mass per bin
    for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)
    {
        double total_species_mass = yields[k+ISMDUSTCHEM_DUST_SPECIES_OFFSET_IN_METALLICITY]*total_Msne*UNIT_MASS_IN_CGS;
        ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[k], &dust_atomic_weight, &bulk_dens);
        C1_norm = total_species_mass*3*(gamma-4)*a_cut*pow(a_min,gamma)*exp(-(pow(log(a_cut/a0),2)/(2*sigma*sigma))) /
        (2*M_PI*bulk_dens*(2*pow(a_cut,gamma)*pow(a_min,4) -
        2*pow(a_cut,4)*pow(a_min,gamma) +
        pow(a0,3)*a_cut*pow(a_min,gamma)*sqrt(2*M_PI)*(gamma-4)*sigma*exp((9*pow(sigma,4)+pow(log(a_cut/a0),2))/(2*sigma*sigma))*erfc((log(a_cut/a0)-3*sigma*sigma)/(sqrt(2)*sigma))));
        C2_norm = C1_norm*pow(a_cut,gamma-1)*exp(-(pow(log(a_cut/a0),2)/(2*sigma*sigma)));
        // Now step through grain size bins and fit bins to inital distribution
        for (l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
            low_edge = All.ISMDustChem_GrainBinEdges[l]; high_edge = All.ISMDustChem_GrainBinEdges[l+1];
            if (low_edge>a_cut && high_edge>a_cut) {
                bin_number = C1_norm*sqrt(M_PI/2)*sigma*(erf(log(high_edge/a0)/(sqrt(2)*sigma))-erf(log(low_edge/a0)/(sqrt(2)*sigma)));
                bin_mass = C1_norm*pow(a0,3)*pow(2*M_PI,3./2.)/3*bulk_dens*sigma*exp(9*sigma*sigma/2)*(erf((3*sigma*sigma+log(a0/low_edge))/(sqrt(2)*sigma))-erf((3*sigma*sigma+log(a0/high_edge))/(sqrt(2)*sigma)));
            }
            else if (low_edge<a_cut && high_edge>a_cut) {
                bin_number = C2_norm*pow(a_cut*low_edge,-gamma)*(pow(a_cut,gamma)*low_edge-a_cut*pow(low_edge,gamma))/(gamma-1) +
                C1_norm*sqrt(M_PI/2)*sigma*(erf(log(high_edge/a0)/(sqrt(2)*sigma))-erf(log(a_cut/a0)/(sqrt(2)*sigma)));
                bin_mass = C2_norm*4*M_PI*bulk_dens/(3*(gamma-4))*(pow(low_edge,4-gamma)-pow(a_cut,4-gamma)) +
                C1_norm*pow(a0,3)*pow(2*M_PI,3./2.)/3*bulk_dens*sigma*exp(9*sigma*sigma/2) * (erf((3*sigma*sigma-log(a_cut/a0))/(sqrt(2)*sigma))-erf((3*sigma*sigma-log(high_edge/a0))/(sqrt(2)*sigma)));
            }
            else {
                bin_number = C2_norm/(gamma-1)*pow(high_edge*low_edge,-gamma)*(pow(high_edge,gamma)*low_edge-high_edge*pow(low_edge,gamma));
                bin_mass = (C2_norm*4.*M_PI*bulk_dens)/(3*(gamma-4))*(pow(low_edge,4-gamma)-pow(high_edge,4-gamma));
            }
            // Deal with rounding errors for near empty bins
            if (bin_number<=0 || bin_mass<=0) {bin_number=0;bin_mass=0;}
            number_yields[k][l] = bin_number;
            mass_yields[k][l] = bin_mass;
        }
    }

    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
            // Convert number and mass of grains injected as mass fraction scalars since all the feedback injection routines expect mass fractions. This works for the number of grains even thought its not a true mass fraction.
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+ISMDUSTCHEM_DUST_NUMBERINBIN_OFFSET_IN_METALLICITY]=number_yields[k][l]/total_Msne;
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+ISMDUSTCHEM_DUST_SLOPEINBIN_OFFSET_IN_METALLICITY]=mass_yields[k][l]/(total_Msne*UNIT_MASS_IN_CGS);
        }
    }

}


/* routine to give grain number and mass yields for AGB winds (currently no dust yield assumed for stars younger than AGB age from continuous mass-loss, i.e. O/B winds) */
void ISMDustChemEvo_get_wind_dust_grain_size_yields(double *yields, double Msne)
{
    int k,l;
    double number_yields[NUM_ISMDUSTCHEM_SPECIES][NUM_ISMDUSTCHEM_SIZE_BINS]={0}, mass_yields[NUM_ISMDUSTCHEM_SPECIES][NUM_ISMDUSTCHEM_SIZE_BINS]={0};
    // Assume initial mass distribution per logarithmic grain size (a^4*dn/da) is log-normal
    double a0=0.1E-4, sigma=0.47; // center (cm) and standard deviation of log-normal distribution
    double bulk_dens,dust_atomic_weight,C_norm,high_edge,low_edge; // grain mass density and normalization set to match total dust mass returned
    for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)
    {
        ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[k], &dust_atomic_weight, &bulk_dens);
        C_norm = (yields[k+ISMDUSTCHEM_DUST_SPECIES_OFFSET_IN_METALLICITY]*Msne*UNIT_MASS_IN_CGS)*(3*a0*exp(-(sigma*sigma)/2.))/(pow(2.,5./2.)*pow(M_PI,3./2.)*sigma*bulk_dens);
        // Now step through grain size bins and fit bins to inital distribution
        for (l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++)
        {
            low_edge = All.ISMDustChem_GrainBinEdges[l]; high_edge = All.ISMDustChem_GrainBinEdges[l+1];
            number_yields[k][l] = (C_norm*exp(8*sigma*sigma)*sqrt(M_PI/2)*sigma/(pow(a0,4))) * (erf((4*sigma*sigma+log(high_edge/a0))/(sqrt(2)*sigma))-erf((4*sigma*sigma+log(low_edge/a0))/(sqrt(2)*sigma)));
            mass_yields[k][l] = (C_norm*pow(2*M_PI,3./2.)*bulk_dens*sigma*exp(sigma*sigma/2)/(3*a0)) * (erf((sigma*sigma+log(high_edge/a0))/(sqrt(2)*sigma))-erf((sigma*sigma+log(low_edge/a0))/(sqrt(2)*sigma)));
            // Deal with rounding errors for effectively empty bins
            if (number_yields[k][l]<=0 || mass_yields[k][l]<=0) {number_yields[k][l]=0;mass_yields[k][l]=0;}
        }
    }

    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
            // Convert number and mass of grains injected as mass fraction scalars since all the feedback injection routines expect mass fractions. This works for the number of grains even thought its not a true mass fraction.
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+ISMDUSTCHEM_DUST_NUMBERINBIN_OFFSET_IN_METALLICITY]=number_yields[k][l]/Msne;
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+ISMDUSTCHEM_DUST_SLOPEINBIN_OFFSET_IN_METALLICITY]=mass_yields[k][l]/(Msne*UNIT_MASS_IN_CGS);
        }
    }
}

#endif


/* simple indexing routine to return the value we need when looping over yields and the like */
double return_ismdustchem_species_of_interest_for_diffusion_and_yields(int i, int k, double mass, struct gas_cell_data *cell)
{
    k -= ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY; // convert from absolute Metallicity[] index to dustchem-block-local index
    if(k<NUM_ISMDUSTCHEM_ELEMENTS) {return cell[i].ISMDustChem_Dust_Metal[k];}
    k -= NUM_ISMDUSTCHEM_ELEMENTS;
    if(k<NUM_ISMDUSTCHEM_SOURCES) {return cell[i].ISMDustChem_Dust_Source[k];}
    k -= NUM_ISMDUSTCHEM_SOURCES;
    if(k<NUM_ISMDUSTCHEM_SPECIES) {return cell[i].ISMDustChem_Dust_Species[k];}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    k -= NUM_ISMDUSTCHEM_SPECIES;
    if(k<2*NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS) {
        int j,m;
        // return number "fraction" of grains in bin
        // since diffusion and yields routines expect scalar mass fractions, divide by the particle mass and treat like a mass scalar
        if (mass==0) {mass = cell[i].Mass;} // Providing the particle mass is only needed for FIRE-2 SNe loop since the particle mass is not thread safe to access
        if(k<NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS) {
            j = k / NUM_ISMDUSTCHEM_SIZE_BINS; m = k % NUM_ISMDUSTCHEM_SIZE_BINS;
            return cell[i].ISMDustChem_Dust_NumberInBin[j][m]/mass;
        }
        // return mass fraction of grains in bin
        else {
            k -= NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS;
            j = k / NUM_ISMDUSTCHEM_SIZE_BINS; m = k % NUM_ISMDUSTCHEM_SIZE_BINS;
            return get_ISMDustChemEvo_bin_mass(i,j,m, cell)/(mass*UNIT_MASS_IN_CGS); // note conversion to code units for bin mass
        }
    }
#endif
    return 0;
}

/* ISMDustChem_Return_Mass_Where_Dust_Shocked migrated to
 * solids/ism_dust_chemistry_functions.h as KOKKOS_INLINE_FUNCTION
 * (Phase D 2026-05-21 #20011-D fix — called from mechanical_fb_pair_kernel
 * which is KOKKOS_INLINE_FUNCTION). */


/* Subroutine to update the dust abundances after enrichment in the mechanical feedback subroutine and destroy dust from SNe shocks */
void update_ISMDustChem_after_mechanical_injection(int j, double mass_shocked, double m0, double mf, double *Z_injected)
{
    // If SNe events happened need to first destroy the appropriate amount of dust if there is any dust
    int k,l,spec_indx;
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
    // Mass is injected before this function in the feedback routine so this check will fail if we don't make a temporary mass change
    // For evolving grain sizes the fraction of dust destroyed depends on the initial grain size distribution
    // This uses a novel routine presented in Choban+25 (in prep) to approximate shattering and subsequent sputtering of dust grains
    double mass_frac_shocked = DMIN(1,mass_shocked/m0), bulk_dens, dust_atomic_weight;
    if ((mass_frac_shocked > 0) && (CellP[j].ISMDustChem_Dust_Metal[0] > 0)) { // if no gas shocked or no dust in gas then no dust destroyed
        CellP[j].ISMDustChem_DelayTimeSNeSputtering = All.ISMDustChem_SNeSputteringShutOffTime; // update thermal sputtering delay time due to SNe
        double shocked_init_bin_M[NUM_ISMDUSTCHEM_SIZE_BINS], shocked_init_bin_N[NUM_ISMDUSTCHEM_SIZE_BINS], shocked_init_bin_slope[NUM_ISMDUSTCHEM_SIZE_BINS];
        double shocked_final_bin_M[NUM_ISMDUSTCHEM_SIZE_BINS]={0}, shocked_final_bin_N[NUM_ISMDUSTCHEM_SIZE_BINS]={0}, shocked_final_bin_slope[NUM_ISMDUSTCHEM_SIZE_BINS]={0};
        double unshocked_init_bin_M[NUM_ISMDUSTCHEM_SIZE_BINS], unshocked_init_bin_N[NUM_ISMDUSTCHEM_SIZE_BINS];
        double total_final_bin_M[NUM_ISMDUSTCHEM_SIZE_BINS], total_final_bin_N[NUM_ISMDUSTCHEM_SIZE_BINS];
        double species_yields[NUM_ISMDUSTCHEM_SPECIES]={0}, dust_yields[NUM_ISMDUSTCHEM_ELEMENTS]={0};
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            if (CellP[j].ISMDustChem_Dust_Species[k] <= 0) {continue;} // No dust so nothin to do
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            ISMDustChem_get_species_properties(spec_indx, &dust_atomic_weight, &bulk_dens);
            // First get the mass/number of grain in each bin in the shocked and unshocked gas.
            for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
                double total_bin_mass = get_ISMDustChemEvo_bin_mass(j,k,l, CellP);
                shocked_init_bin_N[l] = mass_frac_shocked * CellP[j].ISMDustChem_Dust_NumberInBin[k][l];
                shocked_init_bin_slope[l] = mass_frac_shocked * CellP[j].ISMDustChem_Dust_SlopeInBin[k][l];
                shocked_init_bin_M[l] = mass_frac_shocked * total_bin_mass;
                unshocked_init_bin_N[l] = (1-mass_frac_shocked) * CellP[j].ISMDustChem_Dust_NumberInBin[k][l];
                unshocked_init_bin_M[l] = (1-mass_frac_shocked) * total_bin_mass;
            }
            // STEP 1: Shatter dust grains
            ISMDustChem_SNe_shattering_step(spec_indx, shocked_init_bin_N, shocked_init_bin_slope, shocked_init_bin_M, shocked_final_bin_N, shocked_final_bin_slope, shocked_final_bin_M, bulk_dens);
            // STEP 2: Sputter dust grains
            ISMDustChem_SNe_sputtering_step(spec_indx, shocked_final_bin_N, shocked_final_bin_slope, shocked_final_bin_M, shocked_init_bin_N, shocked_init_bin_slope, shocked_init_bin_M, bulk_dens);
            // STEP 3: Shatter again
            ISMDustChem_SNe_shattering_step(spec_indx, shocked_init_bin_N, shocked_init_bin_slope, shocked_init_bin_M, shocked_final_bin_N, shocked_final_bin_slope, shocked_final_bin_M, bulk_dens);
            // Update bins and dust species mass fraction
            // Update number and slope in each bin
            for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
                species_yields[k] += unshocked_init_bin_M[l]+shocked_final_bin_M[l];
                update_ISMDustChemEvo_bin_number_and_slope(j, k, l, unshocked_init_bin_N[l]+shocked_final_bin_N[l], unshocked_init_bin_M[l]+shocked_final_bin_M[l], CellP);
            }
            species_yields[k] /= (m0 * UNIT_MASS_IN_CGS); // Convert to mass fraction
        }

        // Determine new dust element fractions and creation sources
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields,species_yields);
        // Assume all dust creation sources are destroyed equally
        for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {CellP[j].ISMDustChem_Dust_Source[k] *= dust_yields[0]/CellP[j].ISMDustChem_Dust_Metal[0];}
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {CellP[j].ISMDustChem_Dust_Species[k] = species_yields[k];}
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {CellP[j].ISMDustChem_Dust_Metal[k] = dust_yields[k];}
    }
#else
    // Assume a set fraction of dust is destroyed in the shocked gas and destroy each dust species equally
    double dest_eff = 0.4; // dust destruction efficiency
    if (mass_shocked > m0) {mass_shocked = m0;} // limit total mass shocked to mass of gas particle
    double massfrac_destroyed = mass_shocked*dest_eff/m0;
    if((massfrac_destroyed > 0) && (CellP[j].ISMDustChem_Dust_Metal[0] > 0))
    {
        CellP[j].ISMDustChem_DelayTimeSNeSputtering = All.ISMDustChem_SNeSputteringShutOffTime; // update thermal sputtering delay time due to SNe
        if (massfrac_destroyed >= 1.) // destroy all dust
        {
            for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {CellP[j].ISMDustChem_Dust_Metal[k]=0.;}
            for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {CellP[j].ISMDustChem_Dust_Source[k]=0.;}
            for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {CellP[j].ISMDustChem_Dust_Species[k]=0.;}
        }
        else
        {
            double protected_frac = 0.; // Fraction of dust protected from destruction (only iron inclusions are currently considered)
            for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
                spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
                // Take out the iron inclusions protected in silicate dust and then add it back in later
                if(GALSF_ISMDUSTCHEM_MODEL & 8 && spec_indx==All.ISMDustChem_InclIron_Index) {
                    protected_frac += CellP[j].ISMDustChem_Dust_Species[k]/CellP[j].ISMDustChem_Dust_Metal[0]; // Dust_Species is packed by tracked-species slot: index by k, not the fixed species ID
                    CellP[j].ISMDustChem_Dust_Metal[10] -= CellP[j].ISMDustChem_Dust_Species[k]; // Assume all dust species are destroyed evenly but leave out iron inclusions
            }
                else {CellP[j].ISMDustChem_Dust_Species[k] *= 1.-massfrac_destroyed;} // Assume all dust species are destroyed evenly
            }

            for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {CellP[j].ISMDustChem_Dust_Source[k] *= (1.-(1.-protected_frac)*massfrac_destroyed);}
            CellP[j].ISMDustChem_Dust_Metal[0] = 0.0;
            for(k=1;k<NUM_ISMDUSTCHEM_ELEMENTS;k++)
            {
                CellP[j].ISMDustChem_Dust_Metal[k] *= 1.-massfrac_destroyed;
                CellP[j].ISMDustChem_Dust_Metal[0] += CellP[j].ISMDustChem_Dust_Metal[k];
            }
            if(GALSF_ISMDUSTCHEM_MODEL & 8) { // Add the protected iron dust back in
                int incl_indx = All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_InclIron_Index];
                CellP[j].ISMDustChem_Dust_Metal[10] += CellP[j].ISMDustChem_Dust_Species[incl_indx];
                CellP[j].ISMDustChem_Dust_Metal[0] += CellP[j].ISMDustChem_Dust_Species[incl_indx];
                // Update amount of free-flying iron and iron inclusions since some of the inclusions are released from silicate. Assume this leads to constant fraction of iron inclusions that scales with amount of silicate dust
                ISMDustChem_update_iron_inclusions(j, P, CellP);
            }
        }
    }
#endif // GALSF_ISMDUSTCHEM_GRAINSIZEEVO
    // Inject newly created dust from star
    int skip_injection = 1;
    // AGB dust routines can give neglible amounts of dust and the feedback routine can cause yields with initially zero dust to have floating point precision errors.
    // To avoid this, only inject dust when the species mass fractional change is greater than a very small number or dust is being injected where none exists.
    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        if ((CellP[j].ISMDustChem_Dust_Species[k] <= 0 && Z_injected[k+ISMDUSTCHEM_DUST_SPECIES_OFFSET_IN_METALLICITY]>0) || fabs(((1./mf) * DMAX(0.,Z_injected[0+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY])) / ((m0/mf) * CellP[j].ISMDustChem_Dust_Metal[0])) > 1E-10) {
            skip_injection = 0;
            break;
        }
    }
    if (!skip_injection) {
        // Z_injection has the total mass injected so need to be careful updating scalars
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {CellP[j].ISMDustChem_Dust_Metal[k]   = (m0/mf)*CellP[j].ISMDustChem_Dust_Metal[k]   + (1./mf)*DMAX(0.,Z_injected[k+ISMDUSTCHEM_DUST_METAL_OFFSET_IN_METALLICITY]);}
        for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {CellP[j].ISMDustChem_Dust_Source[k]  = (m0/mf)*CellP[j].ISMDustChem_Dust_Source[k]  + (1./mf)*DMAX(0.,Z_injected[k+ISMDUSTCHEM_DUST_SOURCE_OFFSET_IN_METALLICITY]);}
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {CellP[j].ISMDustChem_Dust_Species[k] = (m0/mf)*CellP[j].ISMDustChem_Dust_Species[k] + (1./mf)*DMAX(0.,Z_injected[k+ISMDUSTCHEM_DUST_SPECIES_OFFSET_IN_METALLICITY]);}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        // Inject number of grains and corresponding grain mass into each bin then update number and slope in bin
        // This is relatively simple since Z_injection has the total mass and number injected
        double inject_N_in_bin, inject_M_in_bin, new_N_in_bin, new_M_in_bin;
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
                inject_N_in_bin = Z_injected[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+ISMDUSTCHEM_DUST_NUMBERINBIN_OFFSET_IN_METALLICITY];
                inject_M_in_bin = Z_injected[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+ISMDUSTCHEM_DUST_SLOPEINBIN_OFFSET_IN_METALLICITY]*UNIT_MASS_IN_CGS;
                // If either the number of grains or mass of grains injected into the bin are zero then nothing to do here. Also deals with rounding errors that can cause negative values
                if (inject_N_in_bin>0 && inject_M_in_bin>0) {
                    new_N_in_bin = CellP[j].ISMDustChem_Dust_NumberInBin[k][l] + inject_N_in_bin;
                    new_M_in_bin = get_ISMDustChemEvo_bin_mass(j,k,l, CellP) + inject_M_in_bin;
                    update_ISMDustChemEvo_bin_number_and_slope(j,k,l,new_N_in_bin,new_M_in_bin, CellP);
                }
            }
        }
#endif
    } // !skip_injection
    else {
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {CellP[j].ISMDustChem_Dust_Metal[k]   *= (m0/mf);}
        for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {CellP[j].ISMDustChem_Dust_Source[k]  *= (m0/mf);}
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {CellP[j].ISMDustChem_Dust_Species[k] *= (m0/mf);}
        // Dont need to update grain size bins since they track the total number and mass of grains
    }
}

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)

// Approximates the grain shattering process in shocked gas using a simple analytical function, determining the final number and mass of grains in each bin after shattering.
void ISMDustChem_SNe_shattering_step(int spec_indx, double *init_bin_N, double *init_bin_slope, double *init_bin_M, double *final_bin_N, double *final_bin_slope, double *final_bin_M, double bulk_dens)
{
    int l;
    // Total mass of shattered grains, maximum and minimum sizes of fragements from shattered grains
    double total_M_shat=0, a_frag_max= 0.3E-4, a_frag_min=All.ISMDustChem_Grain_Size_Min;
    double shat_eff, alupper, alcenter, allower;
    double bin_dM[NUM_ISMDUSTCHEM_SIZE_BINS]={0};

    double a_shat = a_frag_max, delta_shat; // characteristic grain size above which shattering is efficient, dust species shattering efficiency
    delta_shat=0.2; /* SILCATE & DEFAULT */
    if (spec_indx==All.ISMDustChem_Carb_Index) {delta_shat *= 1.3;} /* CARBONACEOUS */
    else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {delta_shat *= 1.5;} /* METALLIC IRON */
    delta_shat *= All.ISMDustChem_SNeShatteringScaling;
    if (delta_shat > 0) {
        // Get total mass of grains that are shattered and the change in mass for each bin
        for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
            alcenter = All.ISMDustChem_GrainBinCenters[l];
            shat_eff = 1-exp(-delta_shat * alcenter / a_shat);
            total_M_shat += shat_eff * init_bin_M[l];
            bin_dM[l] = -shat_eff * init_bin_M[l];
        }
        // Get total change in mass after injection of fragments
        for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
            allower = All.ISMDustChem_GrainBinEdges[l], alupper = All.ISMDustChem_GrainBinEdges[l+1];
            // Only inject fragments into bins below the maximum fragment size
            if (allower < a_frag_max) {
                if (alupper>a_frag_max) {alupper=a_frag_max;} // Catch bins that have include the maximum fragment size
                bin_dM[l] += total_M_shat * (pow(alupper,0.7) - pow(allower,0.7)) / (pow(a_frag_max,0.7) - pow(a_frag_min,0.7));
            }
            final_bin_M[l] = init_bin_M[l] + bin_dM[l];
        }
        ISMDustChemEvo_get_new_bin_N_and_slope_given_mass_change(bin_dM, init_bin_M, init_bin_N, init_bin_slope, final_bin_N, final_bin_slope, bulk_dens);
    }
}


// Approximates the grain sputtering process in shocked gas using a simple analytical function, determining the final number and mass of grains in each bin after sputtering.
void ISMDustChem_SNe_sputtering_step(int spec_indx, double *init_bin_N, double *init_bin_slope, double *init_bin_M, double *final_bin_N, double *final_bin_slope, double *final_bin_M, double bulk_dens)
{
    int l;
    // bin center grain size, characteristic grain size below which sputtering is efficient, dust species sputtering efficiency
    double alcenter, a_sput = 0.05E-4, delta_sput, sput_eff;
    double bin_dM[NUM_ISMDUSTCHEM_SIZE_BINS]={0};
    // Determine sputtering efficiency for given dust species
    /******** SILCATE & DEFAULT ********/
    delta_sput=0.6;
    /******** CARBONACEOUS ********/
    if (spec_indx==All.ISMDustChem_Carb_Index) {delta_sput *= 0.66;}
    /******** METALLIC IRON ********/
    else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {delta_sput *= 0.8;}
    delta_sput *= All.ISMDustChem_SNeSputteringScaling;

    for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {  // Get total mass of grains that are destroyed by sputtering for each bin
        alcenter = All.ISMDustChem_GrainBinCenters[l];
        sput_eff = 1-exp(-delta_sput / (alcenter / a_sput));
        bin_dM[l] = -sput_eff * init_bin_M[l];
        final_bin_M[l] = (1 - sput_eff) * init_bin_M[l];
    }
    // Even though sputtering is not a mass conserving process, this is just an approximation for SNe so we use the same function as shattering for updating bin number
    ISMDustChemEvo_get_new_bin_N_and_slope_given_mass_change(bin_dM, init_bin_M, init_bin_N, init_bin_slope, final_bin_N, final_bin_slope, bulk_dens);
}

#endif


/* subroutine to update dust masses from growth via gas-dust accretion and destruction via thermal sputtering (and coagulation and shattering for evolving grain sizes) */





// Update dust grains due to thermal sputtering. This primarily depends on the local gas temperature and density.






// Renormalizes the dust element and source mass fractions given the species mass fractions or grain size bin masses
// Useful when rounding errors build up
void ISMDustChemEvo_renormalize_dust_fields(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    int k,j,spec_indx; double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS]={0.};
    double total = 0;
    for (k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {total += cell[i].ISMDustChem_Dust_Source[k];}
    // Zero everything if no dust
    if (total<=0.) {
        for (k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {cell[i].ISMDustChem_Dust_Source[k] = 0.;}
        for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k]=0.;}
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            cell[i].ISMDustChem_Dust_Species[k]=0.;
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
            for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {update_ISMDustChemEvo_bin_number_and_slope(i,k,j,0,0, cell);}
#endif
        }
    }
    else {
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        // If evolving grain sizes, get total species mass from the bins to set the dust species mass fractions
        double total_spec_mass;
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            total_spec_mass=0;
            for (j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {total_spec_mass+=get_ISMDustChemEvo_bin_mass(i,k,j, cell);}
            // Renorm each dust species mass
            cell[i].ISMDustChem_Dust_Species[k] = total_spec_mass/(cell[i].Mass*UNIT_MASS_IN_CGS);
        }
#endif
        ISMDustChem_get_elem_yields_from_species_yields(dust_yields, cell[i].ISMDustChem_Dust_Species);

        // Catch unphysical element dust mass fractions. Seems to happen for elements which are near entirely depleted onto dust.
        for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {
            if (dust_yields[k]>pp[i].Metallicity[k]) {
                // Check each dust species so that we decrease the yields for all dust species which are composed of the given element
                for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
                    spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[j];
                    // C
                    if (k==2 && (spec_indx==All.ISMDustChem_Carb_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // O
                    else if (k==4 && (spec_indx==All.ISMDustChem_Sil_Index || spec_indx==All.ISMDustChem_ORes_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // Mg
                    else if (k==6 && (spec_indx==All.ISMDustChem_Sil_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // Si
                    else if (k==7 && (spec_indx==All.ISMDustChem_Sil_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // Fe with special check if its in both silicates and metallic iron or just metallic
                    else if (k==10 && ((spec_indx==All.ISMDustChem_FreeIron_Index) || (spec_indx==All.ISMDustChem_InclIron_Index) || (All.ISMDustChem_SilicateMetallicityFieldIndexTable[3]>0 && (spec_indx==All.ISMDustChem_Sil_Index)))) {
                        cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];
                    }
                }
                dust_yields[k] = pp[i].Metallicity[k];
            }
        }
        // Recalculate total dust yields after renorm
        dust_yields[0]=0;
        for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {dust_yields[0] += dust_yields[k];}

        for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k]=dust_yields[k];}
        for (k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {cell[i].ISMDustChem_Dust_Source[k] = DMAX(0,cell[i].ISMDustChem_Dust_Metal[0]/total*cell[i].ISMDustChem_Dust_Source[k]);}
    }
}


// Updates the mass fraction of metallic iron inclusions assuming some set fraction of metallic iron is locked in other dust species as inclusions
// The inclusion fraction is set to GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC * fraction of maximum silicate mass formed.


#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)

// Returns the total dust grain mass (in grams) for given particle i, dust species j, and grain size bin k


// For particle i, given the expected number and mass of dust grains of species j in grain size bin k update the number and slope for said bin. Also check for any unphysical values correct accordingly

/* routine to check bin slopes for unphysical values within bin k (i.e dn/da < 0 at bin edge) for a dust species with a given bulk density.
   If found then adjust slope and number such that slope is zero at bin edge and mass in conserved  */


/* routine to update grain size bins for dust species j of particle i given change (only all positive or all negative) in grain size for each bin. For increasing grain sizes, give an expected limit to the mass (in grams) of the dust species so that you don't grow more dust than is availabe from metallicity */


// Update bin numbers and slopes for particle i and dust species j given mass change from mass-conserving processes


/* Returns solution from polynomial function used to update size bins for coagulation and shattering routines*/


// Determines the new bin numbers and slopes given bin mass changes due shattering or coagulation.


// Debugging function to check the total mass of each dust species against the
// total mass calculated from their grain size bins.
// Halts run if this doesn't matchup and tells you what process caused the issue.
// Note the mass argument is only needed when checking update_ISMDustChem_after_mechanical_injection() for FIRE-2
// since the particle mass is not thread-safe.
void ISMDustChemEvo_check_bins_after_update(int i, int update_process, double mass, struct particle_data *pp, struct gas_cell_data *cell)
{
    int k,l;
    double total_bin_mass, species_mass, species_frac;
    double bin_masses[NUM_ISMDUSTCHEM_SIZE_BINS];
    int failed=0;
    double percent_error = 0.001;
    double min_mass_frac = 1E-20; // Minimum mass fraction to consider for debugging. Very small values will always be prone to rounding errors
    double has_nan=0;
    if (mass==0) {mass = cell[i].Mass;} // only needed for certain routines like SNe feedback/injection since particle mass changes are not thread-safe

    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        total_bin_mass=0;
        species_mass = cell[i].ISMDustChem_Dust_Species[k]*mass; // total dust species mass in grams
        species_frac = cell[i].ISMDustChem_Dust_Species[k];
        for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
            bin_masses[l] = get_ISMDustChemEvo_bin_mass(i,k,l, cell);
            total_bin_mass+= bin_masses[l]/UNIT_MASS_IN_CGS;
            if (cell[i].ISMDustChem_Dust_NumberInBin[k][l]<0 || isnan(cell[i].ISMDustChem_Dust_NumberInBin[k][l])) {has_nan=1;}
            if (isnan(cell[i].ISMDustChem_Dust_SlopeInBin[k][l])) {has_nan=1;}
        }

        if (has_nan || species_frac>pp[i].Metallicity[0] || (species_mass<=0 && total_bin_mass>0) || (species_mass>0 && species_frac>min_mass_frac && (fabs((total_bin_mass-species_mass)/species_mass))>percent_error)) {
            printf("Debugging checks not passed for particle %i type %i species %i mass %e \n",i,pp[i].Type,k, mass*UNIT_MASS_IN_SOLAR);
            printf("update_process: %i \n",update_process);
            printf("total_bin_mass: %e total_species_mass: %e species_met: %e total_met: %e \n", total_bin_mass,species_mass,species_frac,pp[i].Metallicity[0]);
            for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
                printf("\t bin: %i number: %e slope: %e mass: %e\n",l,cell[i].ISMDustChem_Dust_NumberInBin[k][l],cell[i].ISMDustChem_Dust_SlopeInBin[k][l], bin_masses[l]);
            }
            for(l=0;l<4;l++) {
                printf("\t source: %i frac: %e \n",l,cell[i].ISMDustChem_Dust_Source[l]);
            }
            fflush(stdout);
            failed = 1;
        }
    }
    if (failed) {endrun(111);}
}

// Debugging function to check dust size bins against expected yields for stardust.
void ISMDustChemEvo_check_yields_before_update(double *bin_nums, double *bin_slopes, double *bin_masses, int yields_process, int species_num, double total_mass)
{
    int k;
    double calc_bin_masses[NUM_ISMDUSTCHEM_SIZE_BINS];
    int check = 0;
    double total_bin_mass = 0;
    double bulk_dens, dust_atomic_weight;
    ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[species_num], &dust_atomic_weight, &bulk_dens);
    for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {
            double alower = All.ISMDustChem_GrainBinEdges[k], aupper = All.ISMDustChem_GrainBinEdges[k+1], acenter=All.ISMDustChem_GrainBinCenters[k];
                        calc_bin_masses[k] = 4*M_PI*bulk_dens/3*((bin_nums[k]/(4*(aupper-alower))-bin_slopes[k]*acenter/4)*(pow(aupper,4)-pow(alower,4))+bin_slopes[k]/5*(pow(aupper,5)-pow(alower,5)));
            total_bin_mass += calc_bin_masses[k];
            if (fabs(calc_bin_masses[k]-bin_masses[k])/bin_masses[k]>0.001) {check+=1;}
    }
    if (total_mass>0 && fabs(total_bin_mass-total_mass)/total_mass>0.01) {check+=1;}
    if (check) {
        printf("Debugging checks not passed\n");
        printf("yields_process: %i species_num: %i \n",yields_process,species_num);
        for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {
            printf("\t bin: %i number: %e slope: %e calc_bin_mass: %e intended_bin_mass: %e \n",k,bin_nums[k],bin_slopes[k],calc_bin_masses[k],bin_masses[k]);
        }
        printf("total bin mass: %e total_mass: %e\n",total_bin_mass, total_mass);
        fflush(stdout);
        endrun(222);
    }
}

#endif // GALSF_ISMDUSTCHEM_GRAINSIZEEVO //


#endif // GALSF_ISMDUSTCHEM_MODEL //
