#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* This module collects the live ism dust chemistry modules developed by Caleb Choban in Choban et al., 2022/25.
    Written by C. Choban, reorganized and collected by PFH.
 */

#if defined(GALSF_ISMDUSTCHEM_MODEL)
#define ACCRETION_T_CUTOFF 300  /* The cutoff temperature for gas-dust accretion. Also used as a cutoff for density enhancements of dust-dust coagulation.  */
#if (GALSF_ISMDUSTCHEM_MODEL & 4)
#define GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC 0.7 /* assumed fraction of iron dust mass locked as inclusions in silicates, this scales with the total fraction of silicate formed vs maximum amount of possible silicate dust */
#else
#define GALSF_ISMDUSTCHEM_VAR_IRON_INCL_FRAC 0 /* no iron inclusions tracked */
#endif
#if ((GALSF_ISMDUSTCHEM_MODEL & 16) || (GALSF_ISMDUSTCHEM_MODEL & 32)) && defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
#define MAXIMUM_SUBCYCLE_STEPS 200 /* maximum number of subcycle steps for grain size evolution. The current choice is somewhat arbitrary but works for default FIRE resolution. Beware using too small of values or else coagulation and shattering wont be properly time-resolved. */
#define ACC_SPUT_SUBCYCLE_PARAMETER 0.3 /* subcycling parameter for accretion and sputtering. This sets the maximum fraction of the smallest grain size bin that can be traversed in one timestep before subcycling must be used. Default is 30% */
#define SHAT_COAG_SUBCYCLE_PARAMETER 0.1 /* subcycling parameter for shattering and coagulation. This sets the maximum fraction of either mass or number of grains that can be removed across all bins in one timestep for shattering and coagulation respectively. Default is 10% */
#define COAGULATION_DENSITY_ENHANCEMENT 2000 /* Enhancement factor for gas density to account for subresolved dust-dust coagulation which is efficient at nH>=10^4 cm^-3. Tuned for 7100 Msol mass simulations so median extinction curve matches mean MW curve.  */
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
    // number of O, Mg, and Si in one formula unit of silicate dust
    All.ISMDustChem_SilicateNumberOfAtomsTable[0] = 3.63;
    All.ISMDustChem_SilicateNumberOfAtomsTable[1] = 1.06;
    All.ISMDustChem_SilicateNumberOfAtomsTable[2] = 1.;
#if (GALSF_ISMDUSTCHEM_MODEL & 16) || (GALSF_ISMDUSTCHEM_MODEL & 32)
    All.ISMDustChem_SilicateNumberOfAtomsTable[0] += 2; // add 2 more O atoms for silicates to account for excess O depletions with no known carrier
#if (GALSF_ISMDUSTCHEM_MODEL & 32)
    All.ISMDustChem_SilicateMetallicityFieldIndexTable[3] = 10; All.ISMDustChem_SilicateNumberOfAtomsTable[3] = 1.; // add extra Fe as well if not accounting for separate iron species
#endif
#endif
#if (!(GALSF_ISMDUSTCHEM_MODEL & 4) && !(GALSF_ISMDUSTCHEM_MODEL & 16)) 
    All.ISMDustChem_SilicateMetallicityFieldIndexTable[3] = 10; All.ISMDustChem_SilicateNumberOfAtomsTable[3] = 0.571; // add Fe as well if not accounting for iron inclusions
#endif
    All.ISMDustChem_EffectiveSilicateDustAtomicWeight = 0.; for(j=0;j<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;j++) {All.ISMDustChem_EffectiveSilicateDustAtomicWeight += All.ISMDustChem_SilicateNumberOfAtomsTable[j] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[j]];}
    if(GALSF_ISMDUSTCHEM_MODEL & 2) {
        All.ISMDustChem_SpeciesBulkDens[0]=3.13;
        All.ISMDustChem_SpeciesBulkDens[1]=2.25;
        All.ISMDustChem_SpeciesBulkDens[2]=3.21;
        All.ISMDustChem_SpeciesBulkDens[3]=7.86;
    }

#if (GALSF_ISMDUSTCHEM_MODEL & 2)
    All.ISMDustChem_Sil_Index = 0;
    All.ISMDustChem_Carb_Index = 1;
    All.ISMDustChem_SiC_Index = 2;
    All.ISMDustChem_FreeIron_Index = 3;
    All.ISMDustChem_ORes_Index = 4;
    All.ISMDustChem_InclIron_Index = 5;
    for (j=0;j<6;j++) {All.ISMDustChem_SpeciesFieldIndexTable[j] = -1;}
    // silicates and carbonaceous dust are always tracked
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Sil_Index] = 0;
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_Carb_Index] = 1;
    All.ISMDustChem_TrackedSpeciesIDTable[0]=All.ISMDustChem_Sil_Index;
    All.ISMDustChem_TrackedSpeciesIDTable[1]=All.ISMDustChem_Carb_Index;
#if (GALSF_ISMDUSTCHEM_MODEL & 4) || (GALSF_ISMDUSTCHEM_MODEL & 8)
    /* 2=SiC, 3=free-flying iron, 4=O reservoir, 5=iron inclusions in silicates */
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_SiC_Index] = 2;
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_FreeIron_Index] = 3;
    All.ISMDustChem_TrackedSpeciesIDTable[2]=All.ISMDustChem_SiC_Index;
    All.ISMDustChem_TrackedSpeciesIDTable[3]=All.ISMDustChem_FreeIron_Index;
#if (GALSF_ISMDUSTCHEM_MODEL & 4) && !(GALSF_ISMDUSTCHEM_MODEL & 8)
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_InclIron_Index] = 4;
    All.ISMDustChem_TrackedSpeciesIDTable[4]=All.ISMDustChem_InclIron_Index;
#elif !(GALSF_ISMDUSTCHEM_MODEL & 4) && (GALSF_ISMDUSTCHEM_MODEL & 8)
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_ORes_Index] = 4;
    All.ISMDustChem_TrackedSpeciesIDTable[4]=All.ISMDustChem_ORes_Index;
#elif (GALSF_ISMDUSTCHEM_MODEL & 4) && (GALSF_ISMDUSTCHEM_MODEL & 8)
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_ORes_Index] = 4;
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_InclIron_Index] = 5;
    All.ISMDustChem_TrackedSpeciesIDTable[4]=All.ISMDustChem_ORes_Index;
    All.ISMDustChem_TrackedSpeciesIDTable[5]=All.ISMDustChem_InclIron_Index;
#endif
#elif ((GALSF_ISMDUSTCHEM_MODEL & 16) || (GALSF_ISMDUSTCHEM_MODEL & 32)) && defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
#if (GALSF_ISMDUSTCHEM_MODEL & 16)
    All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_FreeIron_Index] = 2;
    All.ISMDustChem_TrackedSpeciesIDTable[2]=All.ISMDustChem_FreeIron_Index;
#endif
    All.ISMDustChem_GrainBinSize = pow(10,log10(All.ISMDustChem_Grain_Size_Max/All.ISMDustChem_Grain_Size_Min)/NUM_ISMDUSTCHEM_SIZE_BINS);
    for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS+1;j++) {All.ISMDustChem_GrainBinEdges[j] = pow(All.ISMDustChem_GrainBinSize,j)*All.ISMDustChem_Grain_Size_Min;}
    for(j=0;j<NUM_ISMDUSTCHEM_SIZE_BINS;j++) {All.ISMDustChem_GrainBinCenters[j] = (All.ISMDustChem_GrainBinEdges[j+1]+All.ISMDustChem_GrainBinEdges[j])/2.;}
#endif
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
        double temp_cutoff=1E5, ne=1, nh0=0, nHe0, nHepp, nhp, nHeII, temp, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
        temp = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, P, CellP);
        if(All.Initial_ISMDustChem_Depletion > 0 && temp < temp_cutoff)
        {
            for(j=0;j<NUM_ISMDUSTCHEM_ELEMENTS;j++) {cell[i].ISMDustChem_Dust_Metal[j] = 0.;}
            if(GALSF_ISMDUSTCHEM_MODEL & 1) {
                cell[i].ISMDustChem_Dust_Metal[4] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[4]; // Silicate dust O
                cell[i].ISMDustChem_Dust_Metal[6] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[6]; // Silicate dust Mg
                cell[i].ISMDustChem_Dust_Metal[7] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[7]; // Silicate dust Si
                cell[i].ISMDustChem_Dust_Metal[10] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[10]; // Silicate dust Fe
                cell[i].ISMDustChem_Dust_Metal[2] = DMIN(pp[i].Metallicity[2],cell[i].ISMDustChem_Dust_Metal[4]+cell[i].ISMDustChem_Dust_Metal[6]+cell[i].ISMDustChem_Dust_Metal[7]+cell[i].ISMDustChem_Dust_Metal[10]/All.Initial_ISMDustChem_SiliconToCarbonRatio); // Carbonaceous dust
            }
            if(GALSF_ISMDUSTCHEM_MODEL & 2) {
                double sil_mass_frac=0., spec_indx; 
                for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
                    spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[j];
                    if (spec_indx==All.ISMDustChem_Sil_Index) {
                        // Silicate dust
                        cell[i].ISMDustChem_Dust_Metal[7] = All.Initial_ISMDustChem_Depletion*pp[i].Metallicity[7]; // Set Si depletion
                        sil_mass_frac+=cell[i].ISMDustChem_Dust_Metal[7];
                        for(k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++) { 
                            // Set element depletions for all other elements in silicates given initial Si depletion
                            if(All.ISMDustChem_SilicateMetallicityFieldIndexTable[k] != 7){
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
    Initialize_ISMDustChemEvo_Particle_Variables(i);
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
        for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
            for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {
                cell[i].ISMDustChem_Shat_dMdt[j][k] = 0;
                cell[i].ISMDustChem_Coag_dMdt[j][k] = 0;
            }
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
        for (j=0;j<NUM_ISMDUSTCHEM_SPECIES;j++) {
            for(k=0;k<NUM_ISMDUSTCHEM_SIZE_BINS;k++) {
                cell[i].ISMDustChem_Shat_dMdt[j][k] = 0;
                cell[i].ISMDustChem_Coag_dMdt[j][k] = 0;
            }
        }
    }
#endif
}
#endif


/* Converts species yields/mass fractions to element yields/mass fractions */
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
        /******** SiC ********/
        else if (spec_indx==All.ISMDustChem_SiC_Index) {
            dust_yields[2] += species_yields[k] * All.ISMDustChem_AtomicMassTable[2] / (All.ISMDustChem_AtomicMassTable[2] + All.ISMDustChem_AtomicMassTable[7]);
            dust_yields[7] += species_yields[k] * All.ISMDustChem_AtomicMassTable[7] / (All.ISMDustChem_AtomicMassTable[2] + All.ISMDustChem_AtomicMassTable[7]);
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


// Get general properties of given dust species
void ISMDustChem_get_species_properties(int spec_indx, double *dust_atomic_weight, double *bulk_dens)
{
    *dust_atomic_weight = 1; *bulk_dens = 1;
    /******** SILICATE ********/
    if (spec_indx==All.ISMDustChem_Sil_Index) {
        *dust_atomic_weight = All.ISMDustChem_EffectiveSilicateDustAtomicWeight;
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[0];
    }
    /******** CARBONACEOUS ********/
    else if (spec_indx==All.ISMDustChem_Carb_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[2];
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[1];
    }
    /******** SiC ********/
    else if (spec_indx==All.ISMDustChem_SiC_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[2]+All.ISMDustChem_AtomicMassTable[7];
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[2];
    }
    /******** METALLIC IRON ********/
    else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[10];
        *bulk_dens = All.ISMDustChem_SpeciesBulkDens[3];
    }
    /******** O RESERVOIR ********/
    else if (spec_indx==All.ISMDustChem_ORes_Index) {
        *dust_atomic_weight = All.ISMDustChem_AtomicMassTable[4];
        *bulk_dens = 1; // O res if a bucket for excess oxygen and doesn't correspond to an actual chemical species but setting to one here for safety
    }
}


// Determine the key element for the given dust species (i.e. the least abudant element that comprises the dust species)
//  key_elem will be -1 if you are missing an element
void ISMDustChem_get_species_key_elem(int spec_indx, double *dust_metallicity, int *key_elem, double *key_num_atoms, double *key_mass)
{
    int k;
    *key_elem = -1; *key_num_atoms = 1; *key_mass = 1;
    /******** SILICATE ********/
    if (spec_indx==All.ISMDustChem_Sil_Index) {
        double sil_elem_abunds[GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES] = {0.};
        *key_elem = 0;
        for(k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++)
        {
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
    else if (spec_indx==All.ISMDustChem_Carb_Index) {if (dust_metallicity[2]>0) {*key_elem=2; *key_mass=All.ISMDustChem_AtomicMassTable[*key_elem];}}
    /******** SiC ********/
    else if (spec_indx==All.ISMDustChem_SiC_Index) {
        if (dust_metallicity[2]>0 && dust_metallicity[7]>0)
        {
            if (dust_metallicity[7]/All.ISMDustChem_AtomicMassTable[7] < dust_metallicity[2]/All.ISMDustChem_AtomicMassTable[2]) *key_elem = 7;
            else *key_elem = 2;
            *key_mass = All.ISMDustChem_AtomicMassTable[*key_elem];
        }
    }
    /******** METALLIC IRON ********/
    else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {if (dust_metallicity[10]>0) {*key_elem=10; *key_mass=All.ISMDustChem_AtomicMassTable[*key_elem];}}
    /******** O RESERVOIR ********/
    else if (spec_indx==All.ISMDustChem_ORes_Index) {if (dust_metallicity[4]>0) {*key_elem=4; *key_mass=All.ISMDustChem_AtomicMassTable[*key_elem];}}
}


/* Approximate dust cooling via electron-dust collisions for MRN sized dust in plasmas from Dwek(1987)+Dwek&Werner(1981). Should surpass metal-line cooling for >10^6 K (even without considering dust depletion), but this will also overpredict dust cooling for <10^7 K since cooling is dominated by small grains which should be destroyed via sputtering */
double Lambda_Dust_HighTemperature_Gas_ISM(int target, double T, double n_elec, struct particle_data *pp, struct gas_cell_data *cell)
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


/* routine to give yields for dust for different types of SNe (Ia & II) followed in-code */
void ISMDustChem_get_SNe_dust_yields(double *yields, int i, double t_gyr, int SNeIaFlag, double Msne, struct particle_data *pp, struct gas_cell_data *cell)
{
    double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS]={0}, species_yields[NUM_ISMDUSTCHEM_SPECIES]={0}; double SNeIa_age = 0.03753; int j,k,spec_indx,source_key=1;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    SNeIa_age =  0.044;
#endif
    if(t_gyr < SNeIa_age) {source_key=2;} // 1=1a, 2=II
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+NUM_METAL_SPECIES]=0;} // initialize yields to null
    if(GALSF_ISMDUSTCHEM_MODEL & 1) {
        double C_condens_eff     = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.5), 
               other_condens_eff = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.8);
        dust_yields[2] = C_condens_eff * yields[2];         // C
        dust_yields[6] = other_condens_eff * yields[6];     // Mg
        dust_yields[7] = other_condens_eff * yields[7];     // Si
        dust_yields[10] = other_condens_eff * yields[10];   // Fe
        dust_yields[4] = 16 * (dust_yields[6]/All.ISMDustChem_AtomicMassTable[6] + dust_yields[7]/All.ISMDustChem_AtomicMassTable[7] + dust_yields[10]/All.ISMDustChem_AtomicMassTable[10]); // O
        if(dust_yields[4]>yields[4]) {dust_yields[4]=yields[4];} // Just in case there's not enough O
        for(k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++)  dust_yields[0] += dust_yields[k]; // Fraction of yields that is dust
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+NUM_METAL_SPECIES]=dust_yields[k];}
        yields[NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+source_key] = dust_yields[0]; // total yield goes to the source term of this type
        return; // all done, if only using this model
    } // below follows species model, will be default if above not set
    
    double SNeII_sil_cond = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.2), 
           SNeII_C_cond   = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.2), 
           SNeII_SiC_cond = DMIN(1,All.ISMDustChem_SNeIIDustScaling*0.0003), 
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
                else if (spec_indx==All.ISMDustChem_SiC_Index) {SNeII_cond=SNeII_SiC_cond;}
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
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+NUM_METAL_SPECIES]=dust_yields[k];}
    yields[NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+source_key] = dust_yields[0]; // total yield goes to the source term of this type
    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES]=species_yields[k];}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    ISMDustChemEvo_get_SNe_dust_grain_size_yields(yields,i,SNeIaFlag,Msne, pp, cell); // get dust grain size/mass yields
#endif
}


/* routine to give the dust yields for AGB winds (currently no dust yield assumed for stars younger than AGB age from continuous mass-loss, i.e. O/B winds) */
void ISMDustChem_get_wind_dust_yields(double *yields, int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS]={0}, species_yields[NUM_ISMDUSTCHEM_SPECIES]={0}; int j,k,spec_indx,source_key=3;
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+NUM_METAL_SPECIES]=0;} // initialize yields to null
    double transition_age = 0.03753, star_age = evaluate_stellar_age_Gyr(i); // Assume AGB dust production stars at SNe II to SNe Ia transition. This limits AGB stars with mass < ~8 solar masses
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    transition_age =  0.044;
#endif
    if(star_age <= transition_age) {return;} // no yield here if too young, otherwise continue
    if(GALSF_ISMDUSTCHEM_MODEL & 1) {
        double condens_eff = DMIN(1,All.ISMDustChem_AGBDustScaling*0.8);
        if((yields[2]/All.ISMDustChem_AtomicMassTable[2])/(yields[4]/All.ISMDustChem_AtomicMassTable[4]) > 1.0) // AGB stars with abundace ratio C/O > 1 only produce carbonacous dust
        {
            dust_yields[2] = yields[2] - 0.75*yields[4]; dust_yields[0] = dust_yields[2]; // C
        } else { // AGB stars with abundance C/O < 1 produce general silicate dust
            dust_yields[6] = condens_eff * yields[6]; // Mg
            dust_yields[7] = condens_eff * yields[7]; // Si
            dust_yields[10] = condens_eff * yields[10]; // Fe
            dust_yields[4] = 16 * (dust_yields[6]/All.ISMDustChem_AtomicMassTable[6] + dust_yields[7]/All.ISMDustChem_AtomicMassTable[7] + dust_yields[10]/All.ISMDustChem_AtomicMassTable[10]); // O
            // Check to make sure we dont produce too much O dust given the leftover O dust not in CO
            if (dust_yields[4] > yields[4]-(4./3.*yields[2])) {dust_yields[4] = yields[4]-(4./3.*yields[2]);}
            for(k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {dust_yields[0]+=dust_yields[k];}
        }
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+NUM_METAL_SPECIES]=dust_yields[k];}
        yields[NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+source_key] = dust_yields[0]; // total yield goes to the source term of this type
        return; // end routine
    } // below follows species model, and will be default if above not set
    double dt,Z,elem_yield,wind_rate;
    dt=GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i)*UNIT_TIME_IN_GYR;
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
        wind_rate *= All.StellarMassLoss_Rate_Renormalization;
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
                    if (k==2 && (spec_indx==All.ISMDustChem_Carb_Index || spec_indx==All.ISMDustChem_SiC_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // O
                    else if (k==4 && (spec_indx==All.ISMDustChem_Sil_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // Mg
                    else if (k==6 && (spec_indx==All.ISMDustChem_Sil_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // Si
                    else if (k==7 && (spec_indx==All.ISMDustChem_Sil_Index || spec_indx==All.ISMDustChem_SiC_Index)) {species_yields[j] *= yields[k]/dust_yields[k];}
                    // Fe with special check if its in both silicates and metallic iron or just metallic 
                    else if (k==10 && ((GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES==3 && spec_indx==All.ISMDustChem_FreeIron_Index) || (GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES==4 && (spec_indx==All.ISMDustChem_Sil_Index || spec_indx==All.ISMDustChem_FreeIron_Index)))) { 
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
    for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {yields[k+NUM_METAL_SPECIES]=dust_yields[k];}
    yields[NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+source_key] = dust_yields[0]; // total yield goes to the source term of this type
    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {yields[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES]=species_yields[k];}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
    ISMDustChemEvo_get_wind_dust_grain_size_yields(yields,cell[i].Mass*cell[i].MassReturn_ThisTimeStep); // get dust grain size/mass yields
#endif
}



/* Simple fit to cumulative AGB dust production for a Kroupa IMF stellar population only with specific metallicities and stellar ages (assuming stars become AGBs at the ends of the main sequence lifetime) derived from AGB dust creation data table in Zhukovska+(08) */
double specific_Z_AGB_dust(int spec_indx, double star_age, int z_bound)
{
    /* spec_indx: 0 = silicate, 1 = carbon, 2 = silicon carbide, 3 = metallic iron */
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
    else if (spec_indx==All.ISMDustChem_SiC_Index) {
            if (z_bound == 0) {
                if (star_age < 272) {cum_return = 3.55E-7*logt - 6.64E-7;}
                else if (star_age < 890) {cum_return = 1.03E-4*(logt - 2.44) + 2.00E-7;}
                else {cum_return = 8.64E-7*(logt - 2.95) + 5.31E-5;}
            }
            else if (z_bound == 1) {
                if (star_age < 272) {cum_return = 3.73E-7*logt - 6.33E-7;}
                else if (star_age < 1544) {cum_return = 2.93E-5*(logt - 2.43) +  2.75E-7;}
                else {cum_return = 5.82E-7*(logt - 3.19) + 2.24E-5;}
            }
            else if (z_bound == 2) {
                if (star_age < 235) {cum_return = 1.06E-7*logt - 1.74E-7;}
                else if (star_age < 4812) {cum_return = 1.04E-6*(logt - 2.37) + 7.84E-8;}
                else {cum_return = 2.47E-7*(logt - 3.68) + 1.44E-6;}
            }
            else if (z_bound == 3) {
                if (star_age < 202) {cum_return = 6.38E-9*logt - 1.02E-8;}
                else if (star_age < 3394) {cum_return = 2.48E-8*(logt - 2.31) + 4.52E-9;}
                else {cum_return = 9.51E-9*(logt - 3.53) + 3.48E-8;}
            }
            else if (z_bound == 4) {
                if (star_age < 245) {cum_return = 2.97E-11*logt - 4.81E-11;}
                else if (star_age < 2392) {cum_return = 1.11E-10*(logt - 2.39) + 2.28E-11;}
                else {cum_return = 3.73E-13*(logt - 3.38) + 1.33E-10;}
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
    /* dust_type: 0 = silicate, 1 = carbon, 2 = silicon carbide, 3 = metallic iron */
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
        double total_species_mass = yields[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES]*total_Msne*UNIT_MASS_IN_CGS; 
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
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES]=number_yields[k][l]/total_Msne;
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES+(NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS)]=mass_yields[k][l]/(total_Msne*UNIT_MASS_IN_CGS);
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
        C_norm = (yields[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES]*Msne*UNIT_MASS_IN_CGS)*(3*a0*exp(-(sigma*sigma)/2.))/(pow(2.,5./2.)*pow(M_PI,3./2.)*sigma*bulk_dens);
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
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES]=number_yields[k][l]/Msne;
            yields[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES+(NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS)]=mass_yields[k][l]/(Msne*UNIT_MASS_IN_CGS);
        }
    }
}

#endif


/* simple indexing routine to return the value we need when looping over yields and the like */
double return_ismdustchem_species_of_interest_for_diffusion_and_yields(int i, int k, double mass, struct gas_cell_data *cell)
{
    k -= NUM_METAL_SPECIES;
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

/* return the mass of gas shocked by an SNe in which dust can be destroyed */
double ISMDustChem_Return_Mass_Where_Dust_Shocked(double rho_cell_in_code_units, double Esne51_into_cell, double mass_preshock_in_code_units, double Z_cell)
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


/* Subroutine to update the dust abundances after enrichment in the mechanical feedback subroutine and destroy dust from SNe shocks */
void update_ISMDustChem_after_mechanical_injection(int j, double mass_shocked, double m0, double mf, double *Z_injected)
{
    // If SNe events happened need to first destroy the appropriate amount of dust if there is any dust
    int k,l,spec_indx;
#if ((GALSF_ISMDUSTCHEM_MODEL & 16)  || (GALSF_ISMDUSTCHEM_MODEL & 32)) && defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
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
                double total_bin_mass = get_ISMDustChemEvo_bin_mass(j,k,l, cell);
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
                update_ISMDustChemEvo_bin_number_and_slope(j, k, l, unshocked_init_bin_N[l]+shocked_final_bin_N[l], unshocked_init_bin_M[l]+shocked_final_bin_M[l], cell);
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
                if(GALSF_ISMDUSTCHEM_MODEL & 4 && spec_indx==All.ISMDustChem_InclIron_Index) {
                    protected_frac += CellP[j].ISMDustChem_Dust_Species[spec_indx]/CellP[j].ISMDustChem_Dust_Metal[0];
                    CellP[j].ISMDustChem_Dust_Metal[10] -= CellP[j].ISMDustChem_Dust_Species[spec_indx]; // Assume all dust species are destroyed evenly but leave out iron inclusions
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
            if(GALSF_ISMDUSTCHEM_MODEL & 4) { // Add the protected iron dust back in
                int incl_indx = All.ISMDustChem_SpeciesFieldIndexTable[All.ISMDustChem_InclIron_Index];
                CellP[j].ISMDustChem_Dust_Metal[10] += CellP[j].ISMDustChem_Dust_Species[incl_indx];
                CellP[j].ISMDustChem_Dust_Metal[0] += CellP[j].ISMDustChem_Dust_Species[incl_indx];
                // Update amount of free-flying iron and iron inclusions since some of the inclusions are released from silicate. Assume this leads to constant fraction of iron inclusions that scales with amount of silicate dust
                ISMDustChem_update_iron_inclusions(j, P, CellP);
            }
        }
    }
#endif // GALSF_ISMDUSTCHEM_MODEL & 16 || GALSF_ISMDUSTCHEM_MODEL & 32
    // Inject newly created dust from star
    int skip_injection = 1;
#ifndef GALSF_USE_SNE_ONELOOP_SCHEME 
    // AGB dust routines can give neglible amounts of dust and the feedback routine can cause yields with initially zero dust to have floating point precision errors.
    // To avoid this, only inject dust when the species mass fractional change is greater than a very small number or dust is being injected where none exists.
    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        if ((CellP[j].ISMDustChem_Dust_Species[k] <= 0 && Z_injected[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES]>0) || fabs(((1./mf) * DMAX(0.,Z_injected[0+NUM_METAL_SPECIES])) / ((m0/mf) * CellP[j].ISMDustChem_Dust_Metal[0])) > 1E-10) {
            skip_injection = 0;
            break;
        }
    }
#else
    skip_injection=0; // Cant check particle info for FIRE-2 since its not thread-safe
#endif
    if (~skip_injection) {
        // Z_injection has the total mass injected so need to be careful updating scalars
        for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {CellP[j].ISMDustChem_Dust_Metal[k]   = (m0/mf)*CellP[j].ISMDustChem_Dust_Metal[k]   + (1./mf)*DMAX(0.,Z_injected[k+NUM_METAL_SPECIES]);}
        for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++)  {CellP[j].ISMDustChem_Dust_Source[k]  = (m0/mf)*CellP[j].ISMDustChem_Dust_Source[k]  + (1./mf)*DMAX(0.,Z_injected[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS]);}
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {CellP[j].ISMDustChem_Dust_Species[k] = (m0/mf)*CellP[j].ISMDustChem_Dust_Species[k] + (1./mf)*DMAX(0.,Z_injected[k+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES]);}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        // Inject number of grains and corresponding grain mass into each bin then update number and slope in bin
        // This is relatively simple since Z_injection has the total mass and number injected 
        double inject_N_in_bin, inject_M_in_bin, new_N_in_bin, new_M_in_bin;
        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            for(l=0;l<NUM_ISMDUSTCHEM_SIZE_BINS;l++) {
                inject_N_in_bin = Z_injected[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES];
                inject_M_in_bin = Z_injected[(k*NUM_ISMDUSTCHEM_SIZE_BINS+l)+NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES+(NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS)]*UNIT_MASS_IN_CGS;
                // If either the number of grains or mass of grains injected into the bin are zero then nothing to do here. Also deals with rounding errors that can cause negative values
                if (inject_N_in_bin>0 && inject_M_in_bin>0) {
                    new_N_in_bin = CellP[j].ISMDustChem_Dust_NumberInBin[k][l] + inject_N_in_bin;
                    new_M_in_bin = get_ISMDustChemEvo_bin_mass(j,k,l, cell) + inject_M_in_bin;
                    update_ISMDustChemEvo_bin_number_and_slope(j,k,l,new_N_in_bin,new_M_in_bin, cell);
                }
            }
        }
#endif
    } // ~skip_injection
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
void update_dust_processes(int i, double dtime_gyr, struct particle_data *pp, struct gas_cell_data *cell)
{
    // First renorm dust due to building numerical error that can arise from stellar feedback. This may no longer be necessary.
#if defined(GALSF_USE_SNE_ONELOOP_SCHEME)
    // Renorm dust fields due to building numerical error in stellar feedback routines. 
    // Has to be done here for FIRE-2 since it requires accessing the final particle mass once the 
    // entire stellar feedback loop is complete since this cannot be accessed in a thread-safe manner.
    ISMDustChemEvo_renormalize_dust_fields(i, pp, cell);
#endif
    int k; double ne=1, nh0=0, nHe0, nHepp, nhp, nHeII, temp, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, u0=cell[i].InternalEnergyPred;
    temp = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, P, CellP);
    rho*=UNIT_DENSITY_IN_CGS;
    
#if !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO) 
    // Choban+22 version of the code tracks a simplified dense molecular gas and CO fraction
    update_dense_molecular_fields(i,temp,rho,nh0,ne, pp, cell);
#endif

#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO) && !defined(COOL_MOLECFRAC_NONEQM)
    // Need local mach number for grain size evolution routines which is taken from nonequilibrium H2 calculations
    // If nonequilibrium H2 not used then need to calculate the mach number here

    // define a number of variables needed in the shielding module
    double dx_cell = Get_Particle_Size(i) * All.cf_atime; // cell size
    double surface_density_H2_0 = 5.e14 * PROTONMASS_CGS, x_exp_fac=0.00085, w0=0.2; // characteristic cgs column for -molecular line- self-shielding
    w0 = 0.035; // actual calibration from Drain, Gnedin, Richings, others: 0.2 is more appropriate as a re-calibration for sims doing local eqm without ability to resolve shielding at higher columns
    double v_thermal_rms = 0.111*sqrt(T); // sqrt(3*kB*T/2*mp), since want rms thermal speed of -molecular H2- in kms
    double gradv = velocity_gradient_norm(i);
    double dv_turb=gradv*dx_cell*UNIT_VEL_IN_KMS; // delta-velocity across cell
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
    double surface_density = evaluate_NH_from_GradRho(pp[i].GradRho,pp[i].KernelRadius,cell[i].Density,pp[i].NumNgb,1,i) * UNIT_SURFDEN_IN_CGS; // converts to cgs
    // shielding length giving effective radius of gas particle
    double l_shield = surface_density / rho;
    fH2 = Get_Gas_Molecular_Mass_Fraction(i, temp, nh0, ne, 0., P, CellP);
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


void update_dust_accretion(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell)
{
    int j,k,spec_indx;
    double dF; // change in fraction of element condensed into dust
    double growth_timescale, t_ref, T_ref, avg_grain_radius;
    double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS] = {0.0};
    int source = 0;
    
    /* elemental model */
#if (GALSF_ISMDUSTCHEM_MODEL & 1)
    cell[i].ISMDustChem_Dust_Metal[0] = 0.; // First renorm dust due to building numerical error that can arise from stellar feedback. This may no longer be necessary.
    for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[0] += cell[i].ISMDustChem_Dust_Metal[k];}
    double total = cell[i].ISMDustChem_Dust_Source[0]+cell[i].ISMDustChem_Dust_Source[1]+cell[i].ISMDustChem_Dust_Source[2]+cell[i].ISMDustChem_Dust_Source[3];
    for (k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) cell[i].ISMDustChem_Dust_Source[k] = DMAX(0,cell[i].ISMDustChem_Dust_Metal[0]/total*cell[i].ISMDustChem_Dust_Source[k]);
    
    /* Accretion happens everywhere no matter the gas phase */
    double rho_ref = PROTONMASS_CGS; // 1 H atom cm^-3
    T_ref = 20.; avg_grain_radius = 0.032; /* um */ t_ref = 0.2; /* Gyr */
    growth_timescale = t_ref * (rho_ref / rho) * pow((T_ref / temp), .5) / All.ISMDustChem_DustAccretionScaling;
    // Calculate the fraction of mass of a certain element to be added to dust due to accretion
    for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++)
    {
        double in_mol_frac; // fraction of element in molecular form and unable to accrete onto dust (CO is the only molecule considered)
        if (k==2) {in_mol_frac = cell[i].ISMDustChem_C_in_CO;}
        else if (k==4) {in_mol_frac = cell[i].ISMDustChem_C_in_CO * All.ISMDustChem_AtomicMassTable[4] / All.ISMDustChem_AtomicMassTable[2];}
        else {in_mol_frac = 0.;}
        // If no dust, metals, or all metals in dust then no accretion
        if (pp[i].Metallicity[k] == 0. || cell[i].ISMDustChem_Dust_Metal[k] == 0. || (pp[i].Metallicity[k] - cell[i].ISMDustChem_Dust_Metal[k]) <= 0) {dF = 0.;}
        else
        {
            dF = dtime_gyr * (1. - cell[i].ISMDustChem_Dust_Metal[k] / (pp[i].Metallicity[k] - in_mol_frac)) * (cell[i].ISMDustChem_Dust_Metal[k] / growth_timescale);
            // Check in case we use up the rest of the remaining metal in the gas phase and deal with unphysical values
            dF = DMIN(pp[i].Metallicity[k] - cell[i].ISMDustChem_Dust_Metal[k] - in_mol_frac,DMAX(0.,dF));
            dust_yields[k] = dF;
            dust_yields[0] += dust_yields[k];
        }
    }
    // Update dust yields and creation source
    if (dust_yields[0] != 0.)
    {
        cell[i].ISMDustChem_Dust_Source[source] += dust_yields[0];
        for (k=0;k< NUM_ISMDUSTCHEM_ELEMENTS;k++) {cell[i].ISMDustChem_Dust_Metal[k] += dust_yields[k];}
    }
#endif // model == 1, elemental model
#if (GALSF_ISMDUSTCHEM_MODEL & 2)
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
                if(GALSF_ISMDUSTCHEM_MODEL & 4) {t_ref_CNM = 1.66E-6; t_ref_MC = 0.139E-3;} 
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
            if(GALSF_ISMDUSTCHEM_MODEL & 4) { // Update amount of free-flying iron and iron inclusions since some of the free-flying particles become inclusions in silicates. Scales with local amount of silicates
                ISMDustChem_update_iron_inclusions(i, pp, cell);
            }
        }
#endif
    }  // if (temp <= 300)
#endif // species model
}


// Update dust grains due to thermal sputtering. This primarily depends on the local gas temperature and density.
void update_dust_sputtering(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell)
{       
    // Sputtering timescales are negligable for cool gas
    if (temp>1E4) {
        int k,j,spec_indx;
        double dF; // change in fraction of element condensed into dust
        double sputter_timescale, t_ref, T_ref, avg_grain_radius;
        double dust_yields[NUM_ISMDUSTCHEM_ELEMENTS] = {0.0};

#if (GALSF_ISMDUSTCHEM_MODEL & 1)
        T_ref = 2E6; avg_grain_radius = 0.032; /* um */ t_ref = 0.17; /* Gyr */
        sputter_timescale = t_ref * (avg_grain_radius / 0.1) / (rho*1E27) * (pow((T_ref/ temp), 2.5) + 1.) / All.ISMDustChem_ThermalSputteringScaling;
        // Calculate the fraction of mass of a certain element to be destroyed due to thermal sputtering
        for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++)
        {
            if (cell[i].ISMDustChem_Dust_Metal[k] <= 0.) {dF = 0.;}
            else {dF = - dtime_gyr * (cell[i].ISMDustChem_Dust_Metal[k] / (sputter_timescale / 3.));}
            // can't destroy more dust then there is available and deal with unphysical values
            dF = DMAX(-cell[i].ISMDustChem_Dust_Metal[k],DMIN(0,dF));
            dust_yields[k] = dF;
            dust_yields[0] += dF;
        }
        
        // Update dust yields and sources
        if (dust_yields[0] != 0.)
        {
            // Assume all dust sources are destroyed evenly
            for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {cell[i].ISMDustChem_Dust_Source[k] *= (1.+dust_yields[0]/cell[i].ISMDustChem_Dust_Metal[0]);}
            for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++)
            {
                cell[i].ISMDustChem_Dust_Metal[k] += dust_yields[k];
            }
            // Deal with rounding error causing total dust to not equal zero
            int no_dust = 1;
            for (k=2; k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {if (cell[i].ISMDustChem_Dust_Metal[k] > 0.) {no_dust = 0; break;}}
            if (no_dust)
            {
                cell[i].ISMDustChem_Dust_Metal[0] = 0.;
                // if all dust is destroyed need to zero creation sources
                for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {cell[i].ISMDustChem_Dust_Source[k] = 0.;}
            }
        }
#endif // model == 1, elemental model
#if (GALSF_ISMDUSTCHEM_MODEL & 2) && !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        double species_yields[NUM_ISMDUSTCHEM_SPECIES] = {0.0};
        T_ref = 2E6; /* K */ t_ref = 0.17; /* Gyr */

        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            avg_grain_radius = 0.032; /* um */
            // If assuming nano-particle iron, need to use different grain size
            if ((GALSF_ISMDUSTCHEM_MODEL & 4) && spec_indx==All.ISMDustChem_FreeIron_Index) {avg_grain_radius = 0.0032; }
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
            // If all dust (silicates, carbonaceous, SiC, and free-flying iron) is destroyed zero everything to avoid rounding error
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
            if(GALSF_ISMDUSTCHEM_MODEL & 4) { // Update amount of free-flying iron and iron inclusions since some of the inclusions are released as silicates are sputtered. This scales with local amount of silicates. Note if all free-flying dust is destroyed then we assume all iron inclusions are also destroyed
                    ISMDustChem_update_iron_inclusions(i, pp, cell);
                }
            }
        }
#endif // dust species model w/o size evo

#if (GALSF_ISMDUSTCHEM_MODEL & 2) && defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
        int k_cycle, n_subcycle;
        double species_yields[NUM_ISMDUSTCHEM_SPECIES] = {0.0};
        double carbSput, silSput, ironSput, Y_sput, dadt, dust_formula_mass, clumping_factor;
        double logt = log10(temp);
        double nHcgs = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS;    /* hydrogen number dens in cgs units */
        double bin_da[NUM_ISMDUSTCHEM_SIZE_BINS];
        double dt_sput, dt_subcycle, a1_width;
        clumping_factor = 1+0.5*0.5 * cell[i].ISMDustChem_MachNumber*cell[i].ISMDustChem_MachNumber;

        // Sputtering erosion rates (change in grain size per nH) for silicates, carbonaceous, and metallic iron dust from polynomial fits to Nozawa+(2006). Y=(da/dt)/nH (um/yr cm^3)
        // This is the change in grain radius over time which is independant of grain size.
        carbSput = pow(10,-226.85 + 133.44*logt - 32.572*pow(logt,2) + 4.0057*pow(logt,3) - 0.24747*pow(logt,4) + 0.0061212*pow(logt,5));
        silSput = pow(10,-226.95 + 127.94*logt - 29.920*pow(logt,2) + 3.5354*pow(logt,3) - 0.21055*pow(logt,4) + 0.0050362*pow(logt,5));
        ironSput = pow(10,-156.88 +  82.110*logt - 18.238*pow(logt,2) + 2.0692*pow(logt,3) - 0.11933*pow(logt,4) + 0.0027788*pow(logt,5));

        for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {
            spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
            Y_sput = 0;
            if (spec_indx==All.ISMDustChem_Sil_Index) {Y_sput = silSput;}
            else if (spec_indx==All.ISMDustChem_Carb_Index) {Y_sput = carbSput;}
            else if (spec_indx==All.ISMDustChem_FreeIron_Index) {Y_sput = ironSput;}

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
#endif // dust species model w/ size evo
    } // temperature cutoff
}


void update_dust_shattering_and_coagulation(int i, double dtime_gyr, double temp, double rho, struct particle_data *pp, struct gas_cell_data *cell)
{
#if (GALSF_ISMDUSTCHEM_MODEL & 2) && defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)

    // Gas cell volume (cm^-3), relative velocity between colliding grains (cm/s), mass of shattered grains (g), i, j, k grain velocities (cm/s), mach factor for grain velocities, cos theta for angle of impact between two grains
    double Vcell, vikrel, vkjrel, mshat, vgri, vgrk, vgrj, Mach=cell[i].ISMDustChem_MachNumber, cos_imp_angle, b_time_Mach, clumping_factor;
    double vgr[NUM_ISMDUSTCHEM_SIZE_BINS], vrel[NUM_ISMDUSTCHEM_SIZE_BINS][NUM_ISMDUSTCHEM_SIZE_BINS];
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
        enh_factor = COAGULATION_DENSITY_ENHANCEMENT * All.ISMDustChem_CoagDensityEnhancementScaling;
        nH_cgs *= enh_factor;
        Vcell /= enh_factor;
        // Need to curtail Mach number for grain velocities to be below coagulation threshold
        Mach=1; 
    }
    gsl_rng *random_generator_fordust; /* generate uniform random number for grain impact angle */
    random_generator_fordust = gsl_rng_alloc(gsl_rng_ranlxd1); 
    gsl_rng_set(random_generator_fordust, pp[i].ID + 11 + All.NumCurrentTiStep);

    for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++)  {
        if (cell[i].ISMDustChem_Dust_Species[k] <= 0) {continue;} // No dust nothin to do
        double dM[NUM_ISMDUSTCHEM_SIZE_BINS]={0};
        spec_indx = All.ISMDustChem_TrackedSpeciesIDTable[k];
        double bulk_dens, dust_atomic_weight;
        ISMDustChem_get_species_properties(spec_indx, &dust_atomic_weight, &bulk_dens);
        dMdt_moved=0; dNdt_moved=0; total_N=0;
        
        if (spec_indx==All.ISMDustChem_Sil_Index) {vshat = 2.7E5; P1 = 3E11; gamma = 25; E_young = 5.4E11; nu_poisson = 0.17;}
        else if (spec_indx==All.ISMDustChem_Carb_Index) {vshat = 1.2E5; P1 = 4E10; gamma = 75; E_young = 1E11; nu_poisson = 0.32;}
        else if (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index) {vshat = 2.2E5; P1 = 5.5E10; gamma = 3000; E_young = 2.1E12; nu_poisson = 0.27;}
        else {vshat=2E5; P1 = 3E11; gamma = 25; E_young = 5.4E11; nu_poisson = 0.17;} // default to silicates

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
        cos_imp_angle = 2.0*gsl_rng_uniform(random_generator_fordust)-1.0;
        
        for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
            for (bin_k=0;bin_k<NUM_ISMDUSTCHEM_SIZE_BINS;bin_k++) {
                vrel[bin_i][bin_k] = sqrt(vgr[bin_i]*vgr[bin_i] + vgr[bin_k]*vgr[bin_k] - 2*vgr[bin_i]*vgr[bin_k]*cos_imp_angle); // cm/s
            }
        }        

        // This makes a precursory pass to determine if timestep subcycling is needed given the timestep. If subcycling is needed it sets n_subcycle and loops over them, if no subcycling is needed it updates the grain size bins and ends.
        n_subcycle=0;
        k_cycle=0;
        while (k_cycle <= n_subcycle) {
            dMdt_moved=0; dNdt_moved=0; total_N=0;
            for (bin_i=0;bin_i<NUM_ISMDUSTCHEM_SIZE_BINS;bin_i++) {
                ailower = All.ISMDustChem_GrainBinEdges[bin_i], aiupper = All.ISMDustChem_GrainBinEdges[bin_i+1], aicenter=All.ISMDustChem_GrainBinCenters[bin_i];
                miavg = 4/3*M_PI*bulk_dens*aicenter*aicenter*aicenter;
                mlost_shat = 0; mgained_shat = 0; mlost_coag = 0; mgained_coag = 0;

                for (bin_k=0;bin_k<NUM_ISMDUSTCHEM_SIZE_BINS;bin_k++) {
                    akcenter=All.ISMDustChem_GrainBinCenters[bin_k];
                    mk = 4/3*M_PI*bulk_dens*akcenter*akcenter*akcenter;

                    vikrel = vrel[bin_i][bin_k];
                    // Mass lost from bin i due to shattering collisions with grains in bin k
                    if (vikrel > vshat) {mlost_shat += All.ISMDustChem_ShatteringScaling * vikrel * shattering_coagulation_polynomial(i, k, bin_i, bin_k, cell);}
                    vcoag = All.ISMDustChem_VCoagScaling * 10 * 2.14 * sqrt((aicenter*aicenter*aicenter + akcenter*akcenter*akcenter)/pow(aicenter+akcenter,3))*pow(gamma,5./6.)/(pow((E_young/(2*(1-nu_poisson)*(1-nu_poisson))),1./3.)*pow(aicenter*akcenter/(aicenter + akcenter),5./6.)*sqrt(bulk_dens)); // cm/s
                    if (vcoag > vshat) vcoag = vshat; // Rare cases where coagualation threshold is higher than shattering threshold
                    // Mass lost from bin i due to coagulating collisions with grains in bin k
                    if (vikrel <= vcoag) {mlost_coag += All.ISMDustChem_CoagulationScaling * (vikrel) * shattering_coagulation_polynomial(i, k, bin_i, bin_k, cell);}
                    for (bin_j=0;bin_j<NUM_ISMDUSTCHEM_SIZE_BINS;bin_j++) {
                        ajcenter=All.ISMDustChem_GrainBinCenters[bin_j];
                        mj = 4/3*M_PI*bulk_dens*ajcenter*ajcenter*ajcenter; // Typical mass of grains in bin j
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
                                aremnant = DMAX(0,pow(akcenter*akcenter*akcenter - mej/(4/3*M_PI*bulk_dens),1./3.));
                                mremnant = DMAX(0,mk-mej);
                                if (aremnant > ailower && aremnant <= aiupper) {mkj_shat += mremnant;}
                            }
                            mgained_shat += All.ISMDustChem_ShatteringScaling * vkjrel * mkj_shat * shattering_coagulation_polynomial(i, k, bin_k, bin_j, cell);
                        }
                        vcoag = All.ISMDustChem_VCoagScaling * 10 * 2.14 * sqrt((akcenter*akcenter*akcenter + ajcenter*ajcenter*ajcenter)/pow(akcenter+ajcenter,3))*pow(gamma,5./6.)/(pow((E_young/(2*(1-nu_poisson)*(1-nu_poisson))),1./3.)*pow(akcenter*ajcenter/(akcenter + ajcenter),5./6.)*sqrt(bulk_dens)); // cm/s
                        if (vcoag > vshat) vcoag = vshat; // Rare cases where coagualation threshold is higher than shattering threshold
                        // Mass gained in bin i due to coagulating collisions between grains in bin k and bin j producing aggregate grains
                        if (vkjrel <= vcoag) {
                            aaggregate = pow((mk + mj)/(4*M_PI/3*bulk_dens),1./3.);
                            if (aaggregate < aiupper && aaggregate >= ailower) {mkj_coag = (mk + mj)/2;} // Counted twice so divide by 2
                            else {mkj_coag = 0;}
                            mgained_coag += All.ISMDustChem_CoagulationScaling * (vkjrel) * mkj_coag * shattering_coagulation_polynomial(i, k, bin_k, bin_j, cell);
                        }
                    }
                }
                // Note change in Vcell due to coagulation density enhancement only applied to coagulation mass change
                total_mlost = (mlost_coag+mlost_shat)*M_PI*miavg; // units of g/s cm^3
                total_mgained = (mgained_coag+mgained_shat)*M_PI; // units of g/s cm^3
                if (k_cycle==0) {
                    cell[i].ISMDustChem_Shat_dMdt[k][bin_i] = (mgained_shat-mlost_shat)*clumping_factor/Vcell; // g/s
                    cell[i].ISMDustChem_Coag_dMdt[k][bin_i] = (mgained_coag-mlost_coag)*clumping_factor/Vcell; // g/s
                }
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
    gsl_rng_free(random_generator_fordust);
#endif
}


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
                    if (k==2 && (spec_indx==All.ISMDustChem_Carb_Index || spec_indx==All.ISMDustChem_SiC_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // O
                    else if (k==4 && (spec_indx==All.ISMDustChem_Sil_Index || spec_indx==All.ISMDustChem_ORes_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // Mg
                    else if (k==6 && (spec_indx==All.ISMDustChem_Sil_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // Si
                    else if (k==7 && (spec_indx==All.ISMDustChem_Sil_Index || spec_indx==All.ISMDustChem_SiC_Index)) {cell[i].ISMDustChem_Dust_Species[j] *= pp[i].Metallicity[k]/dust_yields[k];}
                    // Fe with special check if its in both silicates and metallic iron or just metallic 
                    else if (k==10 && ((GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES==3 && (spec_indx==All.ISMDustChem_FreeIron_Index || spec_indx==All.ISMDustChem_InclIron_Index)) || (GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES==4 && (spec_indx==All.ISMDustChem_Sil_Index || spec_indx==All.ISMDustChem_FreeIron_Index)))) { 
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


// Debugging function to check dust element metallicites against total dust species masses.
void check_dust_fields(int i, int update_process, struct particle_data *pp, struct gas_cell_data *cell)
{
    int k,j;
    double elem_yields[NUM_ISMDUSTCHEM_ELEMENTS] = {0.};
    double source_total=0;
    for (k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {source_total += cell[i].ISMDustChem_Dust_Source[k];}
    // silicate
    if (All.ISMDustChem_SpeciesFieldIndexTable[0]!=-1) {
    for (k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++)
    {
        elem_yields[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[0]] * All.ISMDustChem_SilicateNumberOfAtomsTable[k] * All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]] / All.ISMDustChem_EffectiveSilicateDustAtomicWeight;
        }
    }
    // carbonaceous
    if (All.ISMDustChem_SpeciesFieldIndexTable[1]!=-1) {
    elem_yields[2] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[1]];
    }
#if (GALSF_ISMDUSTCHEM_MODEL & 4) || (GALSF_ISMDUSTCHEM_MODEL & 8)
    // SiC
    if (All.ISMDustChem_SpeciesFieldIndexTable[2]!=-1) {
    elem_yields[2] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[2]] * All.ISMDustChem_AtomicMassTable[2] / (All.ISMDustChem_AtomicMassTable[2] + All.ISMDustChem_AtomicMassTable[7]);
    elem_yields[7] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[2]] * All.ISMDustChem_AtomicMassTable[7] / (All.ISMDustChem_AtomicMassTable[2] + All.ISMDustChem_AtomicMassTable[7]);
    }

    // metallic iron
    elem_yields[10] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[3]]

    // metallic iron inclusions
#if (GALSF_ISMDUSTCHEM_MODEL & 4) && !(GALSF_ISMDUSTCHEM_MODEL & 8)
    elem_yields[10] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[4]];
#elif (GALSF_ISMDUSTCHEM_MODEL & 4) && (GALSF_ISMDUSTCHEM_MODEL & 8)
    elem_yields[10] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[5]];
#endif

    // oxygen reservoir
#if (GALSF_ISMDUSTCHEM_MODEL & 8)
    elem_yields[4] += cell[i].ISMDustChem_Dust_Species[All.ISMDustChem_SpeciesFieldIndexTable[4]];
#endif
#endif

    for (k=2;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {elem_yields[0] += elem_yields[k];}
    
    int mismatch = 0;
    for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++){
        if (cell[i].ISMDustChem_Dust_Metal[k]>0 && (fabs(cell[i].ISMDustChem_Dust_Metal[k]-elem_yields[k])/cell[i].ISMDustChem_Dust_Metal[k]>0.0001 || cell[i].ISMDustChem_Dust_Metal[k]>pp[i].Metallicity[k])) {mismatch+=1;}
        if(cell[i].ISMDustChem_Dust_Metal[k]<0) {mismatch+=1;}
    }
    for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
        if(cell[i].ISMDustChem_Dust_Species[k]<0) {mismatch+=1;}
    }

    if (mismatch>0){
        printf("Yield mismatch process %i\n",update_process);
        for (k=0;k<GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES;k++){
            printf("k: %i index: %f atomic: %f weight: %f \n", k, All.ISMDustChem_SilicateNumberOfAtomsTable[k],All.ISMDustChem_AtomicMassTable[All.ISMDustChem_SilicateMetallicityFieldIndexTable[k]], All.ISMDustChem_EffectiveSilicateDustAtomicWeight);
        }
        printf("total spec: %e elem:%e source:%e \n",elem_yields[0],cell[i].ISMDustChem_Dust_Metal[0],source_total);
        for (k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {
            printf("k:%i spec_elem: %e elem:%e metal:%e \n",k,elem_yields[k],cell[i].ISMDustChem_Dust_Metal[k],pp[i].Metallicity[k]);
        }
        for (k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
            printf("k;%i spec: %e \n",k,cell[i].ISMDustChem_Dust_Species[k]);
        }
        for (k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {source_total += cell[i].ISMDustChem_Dust_Source[k];}
        fflush(stdout);
        endrun(11);
    }
}


#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)

// Returns the total dust grain mass (in grams) for given particle i, dust species j, and grain size bin k
double get_ISMDustChemEvo_bin_mass(int i, int j, int k, struct gas_cell_data *cell)
{   
    if(cell[i].ISMDustChem_Dust_NumberInBin[j][k]<=0) {return 0;} // no grains
    double alower = All.ISMDustChem_GrainBinEdges[k], aupper = All.ISMDustChem_GrainBinEdges[k+1], acenter=All.ISMDustChem_GrainBinCenters[k];
    double bulk_dens, dust_atomic_weight;
    ISMDustChem_get_species_properties(All.ISMDustChem_TrackedSpeciesIDTable[j], &dust_atomic_weight, &bulk_dens);
    return DMAX(0, 4*M_PI*bulk_dens/3*((cell[i].ISMDustChem_Dust_NumberInBin[j][k]/(4*(aupper-alower))-cell[i].ISMDustChem_Dust_SlopeInBin[j][k]*acenter/4)*(pow(aupper,4)-pow(alower,4))+cell[i].ISMDustChem_Dust_SlopeInBin[j][k]/5*(pow(aupper,5)-pow(alower,5))));
}


// For particle i, given the expected number and mass of dust grains of species j in grain size bin k update the number and slope for said bin. Also check for any unphysical values correct accordingly
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

/* routine to check bin slopes for unphysical values within bin k (i.e dn/da < 0 at bin edge) for a dust species with a given bulk density. 
   If found then adjust slope and number such that slope is zero at bin edge and mass in conserved  */
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


/* routine to update grain size bins for dust species j of particle i given change (only all positive or all negative) in grain size for each bin. For increasing grain sizes, give an expected limit to the mass (in grams) of the dust species so that you don't grow more dust than is availabe from metallicity */
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
            rebinned_number = new_bin_masses[l+1]/(4/3*M_PI*bulk_dens*pow(m_high_edge,3));
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


// Update bin numbers and slopes for particle i and dust species j given mass change from mass-conserving processes
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


/* Returns solution from polynomial function used to update size bins for coagulation and shattering routines*/
double shattering_coagulation_polynomial(int i, int spec_indx, int bin_i, int bin_j, struct gas_cell_data *cell)
{
    double Ni, Nj, si, sj, ail, aiu, aic, ajl, aju, ajc, Iij;
    ail = All.ISMDustChem_GrainBinEdges[bin_i], aiu = All.ISMDustChem_GrainBinEdges[bin_i+1]; aic = All.ISMDustChem_GrainBinCenters[bin_i];
    ajl = All.ISMDustChem_GrainBinEdges[bin_j], aju = All.ISMDustChem_GrainBinEdges[bin_j+1]; ajc = All.ISMDustChem_GrainBinCenters[bin_j];
    Ni = cell[i].ISMDustChem_Dust_NumberInBin[spec_indx][bin_i]; si = cell[i].ISMDustChem_Dust_SlopeInBin[spec_indx][bin_i];
    Nj = cell[i].ISMDustChem_Dust_NumberInBin[spec_indx][bin_j]; sj = cell[i].ISMDustChem_Dust_SlopeInBin[spec_indx][bin_j];
    /* Interaction rate between grains of bin_i and bin_j.
       An ugly polynomial but it's analytically solvable */
    Iij = (12*(2*aiu*aiu + 3*aiu*(ajl + aju) + 2*(ajl*ajl + ajl*aju + aju*aju))*Ni*Nj + 
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

    return DMAX(0,Iij); // Limit to 0 to avoid negative values due to rounding errors
}


// Determines the new bin numbers and slopes given bin mass changes due shattering or coagulation.
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
