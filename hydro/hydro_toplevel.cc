#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/*! \file hydro_toplevel.c
 *  \brief This contains the "primary" hydro loop, where the hydro fluxes are computed.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


/* some very useful notes on the hydro variables in comoving integrations:

 v_code = a * v_peculiar/physical (canonical momentum)
 r_code = r_physical / a (comoving coordinates)
 m_code = m_physical
 rho_code = rho_physical * a^3 (from length/mass scaling)
 InternalEnergy_code = InternalEnergy_physical
 Pressure_code = InternalEnergy_code * rho_code * (gamma-1) = Pressure_physical * a^3
 B_code = a*a * B_physical (comoving magnetic fields)
 Phi_code = B_code*v_code = a^3 * Phi_physical (damping field for Dedner divergence cleaning)
    (note: spec egy of phi field is: phi*phi/(2*mu0*rho*ch*ch); compare Bfield is B*B/(mu0*rho);
    so [phi]~[B]*[ch], where ch is the signal velocity used in the damping equation);
 -- Time derivatives (rate of change from hydro forces) here are all
        assumed to end up in *physical* units ---
 HydroAccel, dMomentum are assumed to end up in *physical* units
    (note, this is different from GADGET's convention, where
     HydroAccel is in units of (Pcode/rhocode)/rcode)
 DtInternalEnergy and dInternalEnergy are assumed to end up in *physical* units
 DtMass and dMass are assumed to end up in *physical* units

 -----------------------------------------

 // All.cf_atime = a = 1/(1+z), the cosmological scale factor //
 All.cf_atime = All.Time;
 // All.cf_a2inv is just handy //
 All.cf_a2inv = 1 / (All.Time * All.Time);
 // All.cf_a3inv * Density_code = Density_physical //
 All.cf_a3inv = 1 / (All.Time * All.Time * All.Time);
 // time units: proper time dt_phys = 1/hubble_function(a) * dz/(1+z) = dlna / hubble_function(a)
 code time unit in comoving is dlna, so dt_phys = dt_code / All.cf_hubble_a   //
 All.cf_hubble_a = hubble_function(All.Time); // hubble_function(a) = H(a) = H(z) //
 // dt_code * v_code/r_code = All.cf_hubble_a2 * dt_phys * v_phys/r_phys //
 All.cf_hubble_a2 = All.Time * All.Time * hubble_function(All.Time);

 -----------------------------------------
 A REMINDER ABOUT GIZMO/GADGET VELOCITY UNITS:: (from Volker)

 The IC file should contain the *peculiar* velocity divided by sqrt(a),
 not the *physical* velocity. Let "x" denote comoving
 coordinates and "r=a*x" physical coordinates. Then I call

 comoving velocity: dx/dt
 physical velocity: dr/dt = H(a)*r + a*dx/dt
 peculiar velocity: v = a * dx/dt

 The physical velocity is hence the peculiar velocity plus the Hubble flow.

 The internal velocity variable is not given by dx/d(ln a). Rather, it is given by
 the canonical momentum p = a^2 * dx/dt.
 The IC-file and snapshot files of gadget/GIZMO don't
 contain the variable "p" directly because of historical reasons.
 Instead, they contain the velocity variable
 u = v/sqrt(a) = sqrt(a) * dx/dt = p / a^(3/2), which is just what the
 manual says. (The conversion between u and p is done on the fly when
 reading or writing snapshot files.)

 Also note that d(ln a)/dt is equal to the
 Hubble rate, i.e.: d(ln a)/dt = H(a) = H_0 * sqrt(omega_m/a^3 + omega_v
 + (1 - omega_m - omega_v)/a^2).

 Best wishes,
 Volker

 -----------------------------------------
*/


static double fac_mu, fac_vsic_fix;
#ifdef MAGNETIC
static double fac_magnetic_pressure;
#endif


/* --------------------------------------------------------------------------------- */
/* define the kernel structure -- purely for handy purposes to clean up notation */
/* --------------------------------------------------------------------------------- */
/* structure to hold fluxes being passed from the hydro sub-routine */
struct Conserved_var_Riemann
{
    MyDouble rho;
    MyDouble p;
    MyDouble v[3];
    MyDouble u;
    MyDouble cs;
#ifdef MAGNETIC
    MyDouble B[3];
    MyDouble B_normal_corrected;
#ifdef DIVBCLEANING_DEDNER
    MyDouble phi;
#endif
#endif
#ifdef COSMIC_RAY_FLUID
    MyDouble CosmicRayPressure[N_CR_PARTICLE_BINS];
    MyDouble CosmicRayFlux[N_CR_PARTICLE_BINS][3];
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
    MyDouble CosmicRayAlfvenEnergy[N_CR_PARTICLE_BINS][2];
#endif
#endif
};


struct kernel_hydra
{
    double dp[3];
    double r, vsig, sound_i, sound_j;
    double dv[3], vdotr2;
    double wk_i, wk_j, dwk_i, dwk_j;
    double h_i, h_j, dwk_ij, rho_ij_inv;
    double spec_egy_u_i;
#ifdef HYDRO_SPH
    double p_over_rho2_i;
#endif
#ifdef MAGNETIC
    double b2_i, b2_j;
    double alfven2_i, alfven2_j;
#ifdef HYDRO_SPH
    double mf_i, mf_j;
#endif
#endif // MAGNETIC //
};
#ifndef HYDRO_SPH
#include "reimann.h"
#endif


/* ok here we define some important variables for our generic communication
    and flux-exchange structures. these can be changed, and vary across the code, but need to be set! */

#define CORE_FUNCTION_NAME hydro_force_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define INPUTFUNCTION_NAME particle2in_hydra    /* name of the function which loads the element data needed (for e.g. broadcast to other processors, neighbor search) */
#define OUTPUTFUNCTION_NAME out2particle_hydra  /* name of the function which takes the data returned from other processors and combines it back to the original elements */
#define CONDITIONFUNCTION_FOR_EVALUATION if((P[i].Type==0)&&(P[i].Mass>0)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */




/* --------------------------------------------------------------------------------- */
/* inputs to the routine: put here what's needed to do the calculation! */
/* --------------------------------------------------------------------------------- */
struct INPUT_STRUCT_NAME
{
    /* basic hydro variables */
    MyDouble Pos[3];
    MyFloat Vel[3];
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyFloat ParticleVel[3];
#endif
    MyFloat KernelRadius;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat ConditionNumber;
    MyFloat FaceClosureError;
    MyFloat InternalEnergyPred;
    MyFloat SoundSpeed;
    MyFloat dt_hydrostep_i;
    MyFloat DrkernNgbFactor;
#ifdef HYDRO_SPH
    MyFloat DrkernHydroSumFactor;
    MyFloat alpha;
#endif
    
    /* matrix of the conserved variable gradients: rho, u, vx, vy, vz */
    struct
    {
        MyDouble Density[3];
        MyDouble Pressure[3];
        MyDouble Velocity[3][3];
#ifdef MAGNETIC
        MyDouble B[3][3];
#ifdef DIVBCLEANING_DEDNER
        MyDouble Phi[3];
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        MyDouble Metallicity[NUM_METAL_SPECIES][3];
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        MyDouble InternalEnergy[3];
#endif
#ifdef DOGRAD_SOUNDSPEED
        MyDouble SoundSpeed[3];
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_COMPGRAD_EDDINGTON_TENSOR)
        MyDouble Rad_E_gamma_ET[N_RT_FREQ_BINS][3];
#endif
    } Gradients;
    MyDouble NV_T[3][3];
    
#if defined(KERNEL_CRK_FACES)
    MyFloat Tensor_CRK_Face_Corrections[16];
#endif
#ifdef HYDRO_PRESSURE_SPH
    MyFloat EgyWtRho;
#endif
    
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    MyFloat Metallicity[NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION];
#endif
    
#ifdef CHIMES_TURB_DIFF_IONS
    MyDouble ChimesNIons[CHIMES_TOTSIZE];
#endif
    
#ifdef RT_SOLVER_EXPLICIT
    MyDouble Rad_E_gamma[N_RT_FREQ_BINS];
    MyDouble Rad_Kappa[N_RT_FREQ_BINS];
    MyDouble RT_DiffusionCoeff[N_RT_FREQ_BINS];
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
    MyDouble ET[N_RT_FREQ_BINS][6];
#endif
#ifdef RT_EVOLVE_FLUX
    MyDouble Rad_Flux[N_RT_FREQ_BINS][3];
#endif
#ifdef RT_INFRARED
    MyDouble Radiation_Temperature;
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    MyDouble Rad_Intensity_Pred[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
#endif
#endif
    
#ifdef TURB_DIFFUSION
    MyFloat TD_DiffCoeff;
#endif
    
#ifdef CONDUCTION
    MyFloat Kappa_Conduction;
#endif
    
#ifdef MHD_NON_IDEAL
    MyFloat Eta_MHD_OhmicResistivity_Coeff;
    MyFloat Eta_MHD_HallEffect_Coeff;
    MyFloat Eta_MHD_AmbiPolarDiffusion_Coeff;
#endif
    
#ifdef VISCOSITY
    MyFloat Eta_ShearViscosity;
    MyFloat Zeta_BulkViscosity;
#endif
    
#ifdef MAGNETIC
    MyFloat BPred[3];
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
    MyFloat Balpha;
#endif
#ifdef DIVBCLEANING_DEDNER
    MyFloat PhiPred;
#endif
#endif // MAGNETIC //
    
#ifdef COSMIC_RAY_FLUID
    MyDouble CosmicRayPressure[N_CR_PARTICLE_BINS];
    MyDouble CosmicRayDiffusionCoeff[N_CR_PARTICLE_BINS];
    MyDouble CosmicRayFlux[N_CR_PARTICLE_BINS][3];
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
    MyDouble CosmicRayAlfvenEnergy[N_CR_PARTICLE_BINS][2];
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
    MyDouble CR_number_to_energy_ratio[N_CR_PARTICLE_BINS];
#endif
#endif
    
#ifdef GALSF_SUBGRID_WINDS
    MyDouble DelayTime;
#endif
    
#ifdef EOS_ELASTIC
    int CompositionType;
    MyFloat Elastic_Stress_Tensor[3][3];
#endif
    
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;



/* --------------------------------------------------------------------------------- */
/* outputs: this is what the routine needs to return to the particles to set their final values */
/* --------------------------------------------------------------------------------- */
struct OUTPUT_STRUCT_NAME
{
    MyDouble Acc[3];
    //MyDouble dMomentum[3]; //manifest-indiv-timestep-debug//
    MyDouble DtInternalEnergy;
    //MyDouble dInternalEnergy; //manifest-indiv-timestep-debug//
    MyFloat MaxSignalVel;
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    MyFloat MaxKineticEnergyNgb;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble DtMass;
    MyDouble dMass;
    MyDouble GravWorkTerm[3];
#endif

#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    MyFloat Dyield[NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION];
#endif

#ifdef CHIMES_TURB_DIFF_IONS
    MyDouble ChimesIonsYield[CHIMES_TOTSIZE];
#endif

#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    MyFloat Dt_Rad_E_gamma[N_RT_FREQ_BINS];
#endif
#if defined(RT_EVOLVE_FLUX)
    MyFloat Dt_Rad_Flux[N_RT_FREQ_BINS][3];
#endif
#if defined(RT_INFRARED)
    MyFloat Dt_Rad_E_gamma_T_weighted_IR;
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    MyFloat Dt_Rad_Intensity[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
#endif
#endif

#if defined(MAGNETIC)
    MyDouble Face_Area[3];
    MyFloat DtB[3];
    MyFloat divB;
#if defined(DIVBCLEANING_DEDNER)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME // mass-based phi-flux
    MyFloat DtPhi;
#endif
    MyFloat DtB_PhiCorr[3];
#endif
#endif // MAGNETIC //

#ifdef COSMIC_RAY_FLUID
    MyDouble Face_DivVel_ForAdOps;
    MyDouble DtCosmicRayEnergy[N_CR_PARTICLE_BINS];
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    MyDouble DtCREgyNewInjectionFromShocks;
#endif
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    MyDouble DtCosmicRay_Number_in_Bin[N_CR_PARTICLE_BINS];
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
    MyDouble DtCosmicRayAlfvenEnergy[N_CR_PARTICLE_BINS][2];
#endif
#endif

}
*DATARESULT_NAME, *DATAOUT_NAME;




/* --------------------------------------------------------------------------------- */
/* this subroutine actually loads the particle data into the structure to share between nodes */
/* --------------------------------------------------------------------------------- */
static inline void particle2in_hydra(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration);
static inline void particle2in_hydra(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = CellP[i].VelPred[k];
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        in->ParticleVel[k] = CellP[i].ParticleVel[k];
#endif
    }
    in->KernelRadius = P[i].KernelRadius;
    in->Mass = P[i].Mass;
    in->Density = CellP[i].Density;
    in->Pressure = CellP[i].Pressure;
    in->InternalEnergyPred = CellP[i].InternalEnergyPred;
    in->SoundSpeed = Get_Gas_effective_soundspeed_i(i);
    in->dt_hydrostep_i = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i);
    in->ConditionNumber = CellP[i].ConditionNumber;
    in->FaceClosureError = CellP[i].FaceClosureError;
#ifdef MHD_CONSTRAINED_GRADIENT
    /* since it is not used elsewhere, we can use the sign of the condition number as a bit
     to conveniently indicate the status of the parent particle flag, for the constrained gradients */
    if(CellP[i].FlagForConstrainedGradients == 0) {in->ConditionNumber *= -1;}
#endif
    in->DrkernNgbFactor = P[i].DrkernNgbFactor;
#ifdef HYDRO_SPH
    in->DrkernHydroSumFactor = CellP[i].DrkernHydroSumFactor;
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
    in->alpha = CellP[i].alpha_limiter * CellP[i].alpha;
#else
    in->alpha = CellP[i].alpha_limiter;
#endif
#endif
    
#ifdef HYDRO_PRESSURE_SPH
    in->EgyWtRho = CellP[i].EgyWtDensity;
#endif
#if defined(KERNEL_CRK_FACES)
    for(k=0;k<16;k++) {in->Tensor_CRK_Face_Corrections[k] = CellP[i].Tensor_CRK_Face_Corrections[k];}
#endif
    
    int j;
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {in->NV_T[j][k] = CellP[i].NV_T[j][k];}}
    
    
    /* matrix of the conserved variable gradients: rho, u, vx, vy, vz */
    for(k=0;k<3;k++)
    {
        in->Gradients.Density[k] = CellP[i].Gradients.Density[k];
        in->Gradients.Pressure[k] = CellP[i].Gradients.Pressure[k];
        for(j=0;j<3;j++) {in->Gradients.Velocity[j][k] = CellP[i].Gradients.Velocity[j][k];}
#ifdef MAGNETIC
        for(j=0;j<3;j++) {in->Gradients.B[j][k] = CellP[i].Gradients.B[j][k];}
#ifdef DIVBCLEANING_DEDNER
        in->Gradients.Phi[k] = CellP[i].Gradients.Phi[k];
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(j=0;j<NUM_METAL_SPECIES;j++) {in->Gradients.Metallicity[j][k] = CellP[i].Gradients.Metallicity[j][k];}
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        in->Gradients.InternalEnergy[k] = CellP[i].Gradients.InternalEnergy[k];
#endif
#ifdef DOGRAD_SOUNDSPEED
        in->Gradients.SoundSpeed[k] = CellP[i].Gradients.SoundSpeed[k];
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_COMPGRAD_EDDINGTON_TENSOR)
        for(j=0;j<N_RT_FREQ_BINS;j++) {in->Gradients.Rad_E_gamma_ET[j][k] = CellP[i].Gradients.Rad_E_gamma_ET[j][k];}
#endif
    }
    
#ifdef RT_SOLVER_EXPLICIT
    for(k=0;k<N_RT_FREQ_BINS;k++)
    {
        in->Rad_E_gamma[k] = CellP[i].Rad_E_gamma_Pred[k];
        in->Rad_Kappa[k] = CellP[i].Rad_Kappa[k];
        in->RT_DiffusionCoeff[k] = rt_diffusion_coefficient(i,k);
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
        {int k_dir; for(k_dir=0;k_dir<6;k_dir++) in->ET[k][k_dir] = CellP[i].ET[k][k_dir];}
#endif
#ifdef RT_EVOLVE_FLUX
        {int k_dir; for(k_dir=0;k_dir<3;k_dir++) in->Rad_Flux[k][k_dir] = CellP[i].Rad_Flux_Pred[k][k_dir];}
#endif
#if defined(RT_EVOLVE_INTENSITIES)
        {int k_dir; for(k_dir=0;k_dir<N_RT_INTENSITY_BINS;k_dir++) {in->Rad_Intensity_Pred[k][k_dir] = CellP[i].Rad_Intensity_Pred[k][k_dir];}}
#endif
    }
#ifdef RT_INFRARED
    in->Radiation_Temperature = CellP[i].Radiation_Temperature;
#endif
#endif
    
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    for(k=0;k<NUM_METAL_SPECIES;k++) {in->Metallicity[k] = P[i].Metallicity[k];}
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    for(k=NUM_METAL_SPECIES;k<NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;k++) {in->Metallicity[k] = return_ismdustchem_species_of_interest_for_diffusion_and_yields(i,k);}
#endif
#endif

#ifdef CHIMES_TURB_DIFF_IONS
    for (k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++) {in->ChimesNIons[k] = CellP[i].ChimesNIons[k]; }
#endif

#ifdef TURB_DIFFUSION
    in->TD_DiffCoeff = CellP[i].TD_DiffCoeff;
#endif

#ifdef CONDUCTION
    in->Kappa_Conduction = CellP[i].Kappa_Conduction;
#endif

#ifdef MHD_NON_IDEAL
    in->Eta_MHD_OhmicResistivity_Coeff = CellP[i].Eta_MHD_OhmicResistivity_Coeff;
    in->Eta_MHD_HallEffect_Coeff = CellP[i].Eta_MHD_HallEffect_Coeff;
    in->Eta_MHD_AmbiPolarDiffusion_Coeff = CellP[i].Eta_MHD_AmbiPolarDiffusion_Coeff;
#endif


#ifdef VISCOSITY
    in->Eta_ShearViscosity = CellP[i].Eta_ShearViscosity;
    in->Zeta_BulkViscosity = CellP[i].Zeta_BulkViscosity;
#endif

#ifdef MAGNETIC
    for(k = 0; k < 3; k++) {in->BPred[k] = Get_Gas_BField(i,k);}
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
    in->Balpha = CellP[i].Balpha;
#endif
#ifdef DIVBCLEANING_DEDNER
    in->PhiPred = Get_Gas_PhiField(i);
#endif
#endif // MAGNETIC //

#ifdef COSMIC_RAY_FLUID
    for(j=0;j<N_CR_PARTICLE_BINS;j++)
    {
        in->CosmicRayPressure[j] = Get_Gas_CosmicRayPressure(i,j);
        in->CosmicRayDiffusionCoeff[j] = CellP[i].CosmicRayDiffusionCoeff[j];
        for(k=0;k<3;k++) {in->CosmicRayFlux[j][k] = CellP[i].CosmicRayFluxPred[j][k];}
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        for(k=0;k<2;k++) {in->CosmicRayAlfvenEnergy[j][k] = CellP[i].CosmicRayAlfvenEnergyPred[j][k];}
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
        in->CR_number_to_energy_ratio[j] = CellP[i].CosmicRay_Number_in_Bin[j] / (CellP[i].CosmicRayEnergy[j] + MIN_REAL_NUMBER);
        in->CR_number_to_energy_ratio[j] *= CellP[i].Flux_Number_to_Energy_Correction_Factor[j];
#endif
    }
#endif

#ifdef EOS_ELASTIC
    in->CompositionType = CellP[i].CompositionType;
    {int k_v; for(k=0;k<3;k++) {for(k_v=0;k_v<3;k_v++) {in->Elastic_Stress_Tensor[k][k_v] = CellP[i].Elastic_Stress_Tensor_Pred[k][k_v];}}}
#endif

#ifdef GALSF_SUBGRID_WINDS
    in->DelayTime = CellP[i].DelayTime;
#endif

}



/* --------------------------------------------------------------------------------- */
/* this subroutine adds the output variables back to the particle values */
/* --------------------------------------------------------------------------------- */
static inline void out2particle_hydra(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration);
static inline void out2particle_hydra(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int k;
    /* these are zero-d out at beginning of hydro loop so should always be added */
    for(k = 0; k < 3; k++)
    {
        CellP[i].HydroAccel[k] += out->Acc[k];
        //CellP[i].dMomentum[k] += out->dMomentum[k]; //manifest-indiv-timestep-debug//
    }
    CellP[i].DtInternalEnergy += out->DtInternalEnergy;
    //CellP[i].dInternalEnergy += out->dInternalEnergy; //manifest-indiv-timestep-debug//

#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CellP[i].DtMass += out->DtMass;
    CellP[i].dMass += out->dMass;
    for(k=0;k<3;k++) {CellP[i].GravWorkTerm[k] += out->GravWorkTerm[k];}
#endif
    if(CellP[i].MaxSignalVel < out->MaxSignalVel) {CellP[i].MaxSignalVel = out->MaxSignalVel;}
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    if(CellP[i].MaxKineticEnergyNgb < out->MaxKineticEnergyNgb) {CellP[i].MaxKineticEnergyNgb = out->MaxKineticEnergyNgb;}
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    for(k=0;k<NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;k++) {CellP[i].Dyield[k] += out->Dyield[k];}
#endif

#ifdef CHIMES_TURB_DIFF_IONS
    for (k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++)
      CellP[i].ChimesNIons[k] = DMAX(CellP[i].ChimesNIons[k] + out->ChimesIonsYield[k], 0.5 * CellP[i].ChimesNIons[k]);
#endif

#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    for(k=0;k<N_RT_FREQ_BINS;k++) {CellP[i].Dt_Rad_E_gamma[k] += out->Dt_Rad_E_gamma[k];}
#endif
#if defined(RT_EVOLVE_FLUX)
    for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {CellP[i].Dt_Rad_Flux[k][k_dir] += out->Dt_Rad_Flux[k][k_dir];}}
#endif
#if defined(RT_INFRARED)
    CellP[i].Dt_Rad_E_gamma_T_weighted_IR += out->Dt_Rad_E_gamma_T_weighted_IR;
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<N_RT_INTENSITY_BINS;k_dir++) {CellP[i].Dt_Rad_Intensity[k][k_dir] += out->Dt_Rad_Intensity[k][k_dir];}}
#endif
#endif

#if defined(MAGNETIC)
    /* can't just do DtB += out-> DtB, because for some hydro methods, the induction equation is solved in the density loop; need to simply add it here */
    for(k=0;k<3;k++) {CellP[i].DtB[k] += out->DtB[k]; CellP[i].Face_Area[k] += out->Face_Area[k];}
    CellP[i].divB += out->divB;
#if defined(DIVBCLEANING_DEDNER)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME // mass-based phi-flux
    CellP[i].DtPhi += out->DtPhi;
#endif
    for(k=0;k<3;k++) {CellP[i].DtB_PhiCorr[k] += out->DtB_PhiCorr[k];}
#endif // Dedner //
#endif // MAGNETIC //

#ifdef COSMIC_RAY_FLUID
    CellP[i].Face_DivVel_ForAdOps += out->Face_DivVel_ForAdOps;
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    CellP[i].DtCREgyNewInjectionFromShocks += out->DtCREgyNewInjectionFromShocks;
#endif
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        CellP[i].DtCosmicRayEnergy[k] += out->DtCosmicRayEnergy[k];
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        CellP[i].DtCosmicRay_Number_in_Bin[k] += out->DtCosmicRay_Number_in_Bin[k];
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        int kAlf; for(kAlf=0;kAlf<2;kAlf++) {CellP[i].DtCosmicRayAlfvenEnergy[k][kAlf] += out->DtCosmicRayAlfvenEnergy[k][kAlf];}
#endif
    }
#endif
}


/* --------------------------------------------------------------------------------- */
/* need to link to the file "hydro_evaluate" which actually contains the computation part of the loop! */
/* --------------------------------------------------------------------------------- */
#include "hydro_evaluate.h"

/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/* This will perform final operations and corrections on the output from the
    hydro routines, AFTER the neighbors have all been checked and summed */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
void hydro_final_operations_and_cleanup(void)
{
    int i,k;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type == 0 && P[i].Mass > 0)
        {
            double dt; dt = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i);

#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            /* signal velocity needs to include rate of gas flow -over- the resolution element, which can be non-zero here */
            double v2_p = CellP[i].MaxSignalVel*CellP[i].MaxSignalVel;
            for(k=0;k<3;k++) {v2_p += (CellP[i].VelPred[k]-CellP[i].ParticleVel[k])*(CellP[i].VelPred[k]-CellP[i].ParticleVel[k]);}
            CellP[i].MaxSignalVel = sqrt(v2_p);
#endif

#if defined(MAGNETIC)
            /* need to subtract out the source terms proportional to the (non-zero) B-field divergence; to stabilize the scheme */
            for(k = 0; k < 3; k++)
            {
#ifndef HYDRO_SPH
                /* this part of the induction equation has to do with advection of div-B, it is not present in SPH */
                CellP[i].DtB[k] -= CellP[i].divB * CellP[i].VelPred[k]/All.cf_atime;
#endif
                CellP[i].HydroAccel[k] -= CellP[i].divB * Get_Gas_BField(i,k)*All.cf_a2inv;
                CellP[i].DtInternalEnergy -= CellP[i].divB * (CellP[i].VelPred[k]/All.cf_atime) * Get_Gas_BField(i,k)*All.cf_a2inv;
            }

            double magnorm_closure = Get_DtB_FaceArea_Limiter(i);

#if defined(DIVBCLEANING_DEDNER) && !defined(HYDRO_SPH)
            // ok now deal with the divB correction forces and damping fields //
            double tolerance_for_correction,db_vsig_h_norm;
            tolerance_for_correction = 10.0;
            db_vsig_h_norm = 0.1; // can be as low as 0.03 //
            double DtB_PhiCorr=0,DtB_UnCorr=0,db_vsig_h=0,PhiCorr_Norm=1.0;
            for(k=0; k<3; k++)
            {
                DtB_UnCorr += CellP[i].DtB[k] * CellP[i].DtB[k]; // physical units //
                db_vsig_h = db_vsig_h_norm * (CellP[i].BPred[k]*All.cf_atime) * (0.5*CellP[i].MaxSignalVel) / (Get_Particle_Size(i)*All.cf_atime);
                DtB_UnCorr += db_vsig_h * db_vsig_h;
                DtB_PhiCorr += CellP[i].DtB_PhiCorr[k] * CellP[i].DtB_PhiCorr[k];
            }

            /* take a high power of these: here we'll use 4, so it works like a threshold */
            DtB_UnCorr*=DtB_UnCorr; DtB_PhiCorr*=DtB_PhiCorr; tolerance_for_correction *= tolerance_for_correction;
            /* now re-normalize the correction term if its unacceptably large */
            if((DtB_PhiCorr > 0)&&(!isnan(DtB_PhiCorr))&&(DtB_UnCorr>0)&&(!isnan(DtB_UnCorr))&&(tolerance_for_correction>0)&&(!isnan(tolerance_for_correction)))
            {

                if(DtB_PhiCorr > tolerance_for_correction * DtB_UnCorr) {PhiCorr_Norm *= tolerance_for_correction * DtB_UnCorr / DtB_PhiCorr;}
                for(k=0; k<3; k++)
                {
                    CellP[i].DtB[k] += PhiCorr_Norm * CellP[i].DtB_PhiCorr[k];
                    CellP[i].DtInternalEnergy += PhiCorr_Norm * CellP[i].DtB_PhiCorr[k] * Get_Gas_BField(i,k)*All.cf_a2inv;
                }
            }

#ifdef HYDRO_MESHLESS_FINITE_VOLUME // mass-based phi-flux
            CellP[i].DtPhi *= magnorm_closure;
#else
            CellP[i].DtPhi = 0;
#endif
            if((!isnan(CellP[i].divB))&&(P[i].KernelRadius>0)&&(CellP[i].divB!=0)&&(CellP[i].Density>0))
            {
                double tmp_ded = 0.5 * CellP[i].MaxSignalVel / (fac_mu*All.cf_atime); // has units of v_physical now
                /* do a check to make sure divB isn't something wildly divergent (owing to particles being too close) */
                double b2_max = 0.0;
                for(k=0;k<3;k++) {b2_max += Get_Gas_BField(i,k)*Get_Gas_BField(i,k);}
                b2_max = 100.0 * fabs( sqrt(b2_max) * All.cf_a2inv * P[i].Mass / (CellP[i].Density*All.cf_a3inv) * 1.0 / (P[i].KernelRadius*All.cf_atime) );
                if(fabs(CellP[i].divB) > b2_max) {CellP[i].divB *= b2_max / fabs(CellP[i].divB);}
                /* ok now can apply this to get the growth rate of phi */
                // CellP[i].DtPhi -= tmp_ded * tmp_ded * All.DivBcleanHyperbolicSigma * CellP[i].divB;
                CellP[i].DtPhi -= tmp_ded * tmp_ded * All.DivBcleanHyperbolicSigma * CellP[i].divB * CellP[i].Density*All.cf_a3inv; // mass-based phi-flux
            }
#endif
#endif // MAGNETIC


            /* we calculated the flux of conserved variables: these are used in the kick operation. But for
             intermediate drift operations, we need the primive variables, so reduce to those here
             (remembering that v_phys = v_code/All.cf_atime, for the sake of doing the unit conversions to physical) */
            for(k=0;k<3;k++)
            {
                CellP[i].DtInternalEnergy -= (CellP[i].VelPred[k]/All.cf_atime) * CellP[i].HydroAccel[k];
                /* we solved for total energy flux (and remember, HydroAccel is still momentum -- keep units straight here!) */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                CellP[i].DtInternalEnergy += 0.5 * (CellP[i].VelPred[k]/All.cf_atime) * (CellP[i].VelPred[k]/All.cf_atime) * CellP[i].DtMass;
                CellP[i].HydroAccel[k] -= (CellP[i].VelPred[k]/All.cf_atime) * CellP[i].DtMass; /* we solved for momentum flux */
#endif
                CellP[i].HydroAccel[k] /= P[i].Mass; /* we solved for momentum flux */
            }
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            CellP[i].DtInternalEnergy -= CellP[i].InternalEnergyPred * CellP[i].DtMass;
#endif
#ifdef MAGNETIC
#ifndef HYDRO_SPH
            for(k=0;k<3;k++)
            {
                CellP[i].DtInternalEnergy += -Get_Gas_BField(i,k)*All.cf_a2inv * CellP[i].DtB[k];
            }
#endif
            for(k=0;k<3;k++) {CellP[i].DtB[k] *= magnorm_closure;}
#endif
            CellP[i].DtInternalEnergy /= P[i].Mass;
            /* ok, now: HydroAccel = dv/dt, DtInternalEnergy = du/dt (energy per unit mass) */

            /* zero out hydrodynamic PdV work terms if the particle is at the maximum smoothing, these will be incorrect */
            if(P[i].KernelRadius >= 0.99*All.MaxKernelRadius) {CellP[i].DtInternalEnergy = 0;}

            // need to explicitly include adiabatic correction from the hubble-flow (for drifting) here //
            if(All.ComovingIntegrationOn) {CellP[i].DtInternalEnergy -= 3*(GAMMA(i)-1) * CellP[i].InternalEnergyPred * All.cf_hubble_a;}
            // = du/dlna -3*(gamma-1)*u ; then dlna/dt = H(z) =  All.cf_hubble_a //


#if defined(RT_RAD_PRESSURE_FORCES) && defined(RT_EVOLVE_FLUX) && !defined(RT_RADPRESSURE_IN_HYDRO) /* // -- moved for OTVET+FLD to drift-kick operation to deal with limiters more accurately -- // */
            /* calculate the radiation pressure force */
            double radacc[3]; radacc[0]=radacc[1]=radacc[2]=0; int kfreq;
            for(kfreq=0;kfreq<N_RT_FREQ_BINS;kfreq++)
            {
                double vol_inv = CellP[i].Density*All.cf_a3inv/P[i].Mass, f_kappa_abs = rt_absorb_frac_albedo(i,kfreq), vel_i[3]={0}, vdot_h[3]={0}, vdot_D[3]={0}, flux_i[3]={0}, flux_mag=0, erad_i=0, flux_corr=0, work_band=0, radacc_thisband[3]={0}, rmag=0;
                erad_i = CellP[i].Rad_E_gamma_Pred[kfreq]*vol_inv; // radiation energy density, needed below
                for(k=0;k<3;k++) {flux_i[k]=CellP[i].Rad_Flux_Pred[kfreq][k]*vol_inv; vel_i[k]=(C_LIGHT_CODE_REDUCED(i)/C_LIGHT_CODE)*CellP[i].VelPred[k]/All.cf_atime; flux_mag+=flux_i[k]*flux_i[k];}
                eddington_tensor_dot_vector(CellP[i].ET[kfreq],vel_i,vdot_D); // note these 'vdoth' terms shouldn't be included in FLD, since its really assuming the entire right-hand-side of the flux equation reaches equilibrium with the pressure tensor, which gives the expression in rt_utilities
                for(k=0;k<3;k++) {vdot_h[k] = (RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS(i)*C_LIGHT_CODE/C_LIGHT_CODE_REDUCED(i)) * erad_i * (vel_i[k] + vdot_D[k]);} // calculate volume integral of scattering coefficient t_inv * (gas_vel . [e_rad*I + P_rad_tensor]), which gives an additional time-derivative term. this is the P term //
                double flux_thin = erad_i * C_LIGHT_CODE_REDUCED(i); if(flux_mag>0) {flux_mag=sqrt(flux_mag);} else {flux_mag=1.e-20*flux_thin;}
                if(flux_mag > 0) {flux_corr = DMIN(1., flux_thin/flux_mag); // restrict flux here (b/c drifted can exceed physical b/c of integration errors
#if defined(RT_ENABLE_R15_GRADIENTFIX)
                    flux_corr = flux_thin/flux_mag; // set to maximum (optically thin limit)
#endif
                }
                double L_particle=Get_Particle_Size(i)*All.cf_atime, Sigma_particle=P[i].Mass/(M_PI*L_particle*L_particle), abs_per_kappa_dt=C_LIGHT_CODE_REDUCED(i)*(CellP[i].Density*All.cf_a3inv)*dt; // effective surface density through particle & fractional absorption over timestep
                int checker_int = 0; // normal default: only use the corrections below for bands which dont re-emit to the same band
                checker_int = 1; // actually here and above now changed to use the slabfac corrections for all bands. in the resolved limit this should still be correct because the re-emitted photons should be isotropic: otherwise you run into linear momentum conservation problems. this is only an issue if the source is at the center of the distribution.
                double slabfac_rp=1; if(check_if_absorbed_photons_can_be_reemitted_into_same_band(kfreq)<=checker_int) {slabfac_rp=slab_averaging_function(f_kappa_abs*CellP[i].Rad_Kappa[kfreq]*Sigma_particle) * slab_averaging_function(f_kappa_abs*CellP[i].Rad_Kappa[kfreq]*abs_per_kappa_dt);} // reduction factor for absorption over dt
                for(k=0;k<3;k++) {radacc_thisband[k] = slabfac_rp * (CellP[i].Rad_Kappa[kfreq]/C_LIGHT_CODE_REDUCED(i)) * (flux_corr*flux_i[k] - vdot_h[k]); rmag += radacc_thisband[k]*radacc_thisband[k];} // acceleration term before accounting for the 'work' term, which is calculated separately in the absorption/emission loop
                if(check_if_absorbed_photons_can_be_reemitted_into_same_band(kfreq)<=checker_int && f_kappa_abs > MIN_REAL_NUMBER && rmag > MIN_REAL_NUMBER && dt > 0 && P[i].Mass > 0) { // bands that destroy photons upon absorption (e.g. ionization, dust absorption) should limit the imparted momentum to the total photon momentum available - the flux in the solver normally prevents this but this addresses some edge cases with e.g. pathological ICs, rapidly-varying kappa, etc.
                    rmag=sqrt(rmag); double r_from_abs=f_kappa_abs*rmag, abs_dt=rt_absorption_rate(i,kfreq)*dt, dE_abs=erad_i*(1.-exp(-abs_dt)); if(abs_dt<0.01) {dE_abs=erad_i*abs_dt;}
                    double rmag_max_abs=dE_abs/(vol_inv*P[i].Mass*C_LIGHT_CODE_REDUCED(i)*dt); if(rmag_max_abs<r_from_abs) {double cfac=1.+(rmag_max_abs-r_from_abs)/rmag; if(cfac>0 && cfac<1) {for(k=0;k<3;k++) {radacc_thisband[k]*=cfac;}}}
                }
                for(k=0;k<3;k++) { /* now record the total work term and photon momentum imparted to gas */
                    radacc[k]+=radacc_thisband[k]; work_band += radacc_thisband[k] * vel_i[k] * P[i].Mass; // PdV work done by photons [absorbed ones are fully-destroyed, so their loss of energy and momentum is already accounted for by their deletion in this limit -- note that we have to be careful about the RSOL factors here! //
                }
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES
                f_kappa_abs = 0;
#endif
                CellP[i].Dt_Rad_E_gamma[kfreq] += (2.*f_kappa_abs-1.)*work_band; // loss/gain term for the radiation field itself
                CellP[i].DtInternalEnergy -= (C_LIGHT_CODE/C_LIGHT_CODE_REDUCED(i)) * 2.*f_kappa_abs*work_band / P[i].Mass; // correct for rsol factor above which reduced vel_i by rsol; -only- add back this term for gas
            }
            for(k=0;k<3;k++) { /* now actually set the frequency-integrated cell values as needed */
#ifdef RT_RAD_PRESSURE_OUTPUT
                CellP[i].Rad_Accel[k] = radacc[k]; // physical units, as desired
#else
                CellP[i].HydroAccel[k] += radacc[k]; // physical units, as desired
#endif
            }
#endif
#ifdef RT_RADPRESSURE_IN_HYDRO
            int kfreq; for(kfreq=0;kfreq<N_RT_FREQ_BINS;kfreq++) {
                double fac = (1./3.) * return_flux_limiter(i,kfreq) * CellP[i].Rad_E_gamma_Pred[kfreq] * P[i].Particle_DivVel*All.cf_a2inv * (1.-2.*rt_absorb_frac_albedo(i,kfreq));
                CellP[i].Dt_Rad_E_gamma[kfreq] -= (C_LIGHT_CODE_REDUCED(i)/C_LIGHT_CODE) * fac; CellP[i].DtInternalEnergy += fac / P[i].Mass; /* exact energy conservation; for appropriate RSOL definitions - careful of terms here where beta arises */
            }
#endif


#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME)) /* update the metal masses from exchange */
            for(k=0;k<NUM_METAL_SPECIES;k++) {P[i].Metallicity[k] = DMAX(P[i].Metallicity[k] + CellP[i].Dyield[k] / P[i].Mass , 0.01*P[i].Metallicity[k]);}
#if defined(GALSF_ISMDUSTCHEM_MODEL) /* update the dust masses from exchange */
            for(k=0;k<NUM_ISMDUSTCHEM_ELEMENTS;k++) {CellP[i].ISMDustChem_Dust_Metal[k] = DMAX(CellP[i].ISMDustChem_Dust_Metal[k] + CellP[i].Dyield[NUM_METAL_SPECIES+k] / P[i].Mass , 0.01*CellP[i].ISMDustChem_Dust_Metal[k]);}
            for(k=0;k<NUM_ISMDUSTCHEM_SOURCES;k++) {CellP[i].ISMDustChem_Dust_Source[k] = DMAX(CellP[i].ISMDustChem_Dust_Source[k] + CellP[i].Dyield[NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+k] / P[i].Mass , 0.01*CellP[i].ISMDustChem_Dust_Source[k]);}
            for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {CellP[i].ISMDustChem_Dust_Species[k] = DMAX(CellP[i].ISMDustChem_Dust_Species[k] + CellP[i].Dyield[NUM_METAL_SPECIES+NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+k] / P[i].Mass , 0.01*CellP[i].ISMDustChem_Dust_Species[k]);}
#endif
#endif
            
            
#if (defined(COSMIC_RAY_FLUID) && !defined(COOLING_OPERATOR_SPLIT)) || defined(COSMIC_RAY_SUBGRID_LEBRON)
            /* with the spectrum model, we account here the adiabatic heating/cooling of the 'fluid', here, which was solved in the hydro solver but doesn't resolve which portion goes to CRs and which to internal energy, with gamma=GAMMA_COSMICRAY */
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            double P_cr_spec = (1./3.)*CellP[i].SubGrid_CosmicRayEnergyDensity/CellP[i].Density, P_tot_spec = P_cr_spec + (2./3.)*CellP[i].InternalEnergyPred + (1./2.)*pow(Get_Gas_Alfven_speed_i(i),2); // just include CR+thermal+magnetic here
            CellP[i].DtInternalEnergy *= (1.-P_cr_spec/P_tot_spec); /* approximate correction, valid to level here [more sophisticated correction can cause problems since the PdV energy isn't actually being taken -out- of the CR field, as it would be if followed explicitly] */
#else
            double gamma_minus_eCR_tmp=0; for(k=0;k<N_CR_PARTICLE_BINS;k++) {gamma_minus_eCR_tmp+=(GAMMA_COSMICRAY(k)-1.)*CellP[i].CosmicRayEnergyPred[k];} // routine below only depends on the total CR energy, not bin-by-bin energies, when we do it this way here
            double dCR_div = CR_calculate_adiabatic_gasCR_exchange_term(i, dt, gamma_minus_eCR_tmp, 1); // this will handle the update below - separate subroutine b/c we want to allow it to appear in a couple different places
            double u0=DMAX(CellP[i].InternalEnergyPred, All.MinEgySpec) , uf=DMAX(u0 - dCR_div/P[i].Mass , All.MinEgySpec); // final updated value of internal energy per above
            CellP[i].DtInternalEnergy += (uf - u0) / (dt + MIN_REAL_NUMBER); // update gas quantities to be used in cooling function
#endif
#endif
#if defined(COSMIC_RAY_FLUID)
            /* energy transfer from CRs to gas due to the streaming instability (mediated by high-frequency Alfven waves, but they thermalize quickly
                (note this is important; otherwise build up CR 'traps' where the gas piles up and cools but is entirely supported by CRs in outer disks) */
#if !defined(CRFLUID_EVOLVE_SCATTERINGWAVES) // handled in separate solver if explicitly evolving the relevant wave families
            for(k=0;k<N_CR_PARTICLE_BINS;k++) {
                double streamfac = fabs(CR_get_streaming_loss_rate_coefficient(i,k));
                CellP[i].DtInternalEnergy += CellP[i].CosmicRayEnergyPred[k] * streamfac / P[i].Mass; // make sure to divide by mass here to get the correct units since DtInternalEnergy has been converted to specific energy units (while CR energies are absolute)
#if !defined(CRFLUID_EVOLVE_SPECTRUM)
                CellP[i].DtCosmicRayEnergy[k] -= CosmicRayFluid_RSOL_Corrfac(k) * CellP[i].CosmicRayEnergyPred[k] * streamfac; // in the multi-bin formalism, save this operation for the CR cooling ops since can involve bin-to-bin transfer of energy
#endif
            }
#endif
#if defined(MAGNETIC) // only makes sense to include parallel correction below if all these terms enabled //
            /* 'residual' term from parallel scattering of CRs being not-necessarily-in-equilibrium with a two-moment form of the equations */
            double vA_eff=Get_Gas_ion_Alfven_speed_i(i), vol_i=CellP[i].Density*All.cf_a3inv/P[i].Mass, Bmag=0, bhat[3]={0}; // define some useful variables
            for(k=0;k<3;k++) {bhat[k]=CellP[i].BPred[k]; Bmag+=bhat[k]*bhat[k];} // get direction vector for B-field needed below
            if(Bmag>0) {Bmag=sqrt(Bmag); for(k=0;k<3;k++) {bhat[k] /= Bmag;}} // make dimensionless
            if(Bmag>0) {for(k=0;k<N_CR_PARTICLE_BINS;k++) {
                int target_for_cr_betagamma = i; // if this = -1, use the gamma factor at the bin-center for evaluating this, if this = i, use the mean gamma of the bin, weighted by the CR energy -- won't give exactly the same result here
                target_for_cr_betagamma = -1; // the correction terms depend on these being evaluated at their bin-centered locations
                double three_chi = return_cosmic_ray_anisotropic_closure_function_threechi(i,k);
                int m; double grad_P_dot_B=0, gradpcr[3]={0}, F_dot_B=0, e0_cr=CellP[i].CosmicRayEnergyPred[k]*vol_i, p0_cr=(GAMMA_COSMICRAY(k)-1.)*e0_cr, vA_k=vA_eff*return_CRbin_nuplusminus_asymmetry(i,k), fcorr[3]={0}, beta_fac=return_CRbin_beta_factor(target_for_cr_betagamma,k);
                for(m=0;m<3;m++) {gradpcr[m] = CellP[i].Gradients.CosmicRayPressure[k][m] * (All.cf_a3inv/All.cf_atime);}
                for(m=0;m<3;m++) {grad_P_dot_B += bhat[m] * gradpcr[m]; F_dot_B += bhat[m] * CellP[i].CosmicRayFluxPred[k][m] * vol_i;}
                if(F_dot_B < 0) {vA_k *= -1;} // needs to have appropriately-matched signage below //
                double gamma_0=return_CRbin_gamma_factor(target_for_cr_betagamma,k), gamma_fac=gamma_0/(gamma_0-1.); // lorentz factor here, needed in next line, because the loss term here scales with -total- energy, not kinetic energy
                if(beta_fac<0.1) {gamma_fac=2./(beta_fac*beta_fac) -0.5 - 0.125*beta_fac*beta_fac;} // avoid accidental nan
                for(m=0;m<3;m++) {fcorr[m] = bhat[m] * (grad_P_dot_B + (gamma_fac*(F_dot_B/CosmicRayFluid_RSOL_Corrfac(k)) - three_chi*vA_k*(gamma_fac*e0_cr + p0_cr))*(beta_fac*beta_fac)/(3.*CellP[i].CosmicRayDiffusionCoeff[k])) / (CellP[i].Density*All.cf_a3inv);} // physical units
                for(m=0;m<3;m++) {fcorr[m] += (1.-three_chi) * (gradpcr[m] - bhat[m]*grad_P_dot_B) / (CellP[i].Density*All.cf_a3inv);} // physical units
                for(m=0;m<3;m++) {CellP[i].HydroAccel[m] += fcorr[m];} // add correction term back into hydro acceleration terms -- need to check that don't end up with nasty terms for badly-initialized/limited scattering rates above
            }}
#endif
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
            if((CellP[i].DtCREgyNewInjectionFromShocks <= 0) || (CellP[i].DtInternalEnergy <= 0) || (P[i].Mass <= 0)) {CellP[i].DtCREgyNewInjectionFromShocks = 0;} // should never be negative, thats an error from above, or a spurious shock detection if dtinternalenergy summed is negative - don't inject here (additional useful check over kernel)
            if(CellP[i].DtCREgyNewInjectionFromShocks > 0) { // do some checks and adjust the internal energy evolution to ensure total energy conservation now
                double dtThermal = CellP[i].DtInternalEnergy * P[i].Mass; // correct back to total energy units
                CellP[i].DtCREgyNewInjectionFromShocks = DMIN(CellP[i].DtCREgyNewInjectionFromShocks, 0.5*dtThermal); // don't allow more than 1/2 of the total internal energy change to go into CRs (should usually be satisfied but since cr injection pairwise, can be violated if shock not well-resolved and flow is chaotic)
                CellP[i].DtInternalEnergy = (dtThermal - CellP[i].DtCREgyNewInjectionFromShocks) / P[i].Mass; // reset the thermal energy budget appropriately, now total energy will be conserved (just locally shifting from one reservoir to another)
            }
#endif
#endif // COSMIC_RAY_FLUID


#ifdef GALSF_SUBGRID_WINDS
            /* if we have winds, we decouple particles briefly if delaytime>0 */
            if(CellP[i].DelayTime > 0)
            {
                for(k = 0; k < 3; k++) {CellP[i].HydroAccel[k] = 0;}//CellP[i].dMomentum[k] = 0;
                CellP[i].DtInternalEnergy = 0; //CellP[i].dInternalEnergy = 0;
                double windspeed = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) / All.WindEfficiency) * All.Time;
                windspeed *= fac_mu;
                double rkern_c = pow(All.WindFreeTravelDensFac * All.PhysDensThresh / (CellP[i].Density * All.cf_a3inv), (1. / 3.));
                CellP[i].MaxSignalVel = rkern_c * DMAX((2 * windspeed), CellP[i].MaxSignalVel);
            }
#endif


#ifdef BOX_BND_PARTICLES
            /* this flag signals all particles with id=0 are frozen (boundary particles) */
            if(P[i].ID == 0)
            {
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                CellP[i].DtMass = 0;
                CellP[i].dMass = 0;
                for(k = 0; k < 3; k++) CellP[i].GravWorkTerm[k] = 0;
#endif
                CellP[i].DtInternalEnergy = 0;//CellP[i].dInternalEnergy = 0;//manifest-indiv-timestep-debug//
                for(k = 0; k < 3; k++) CellP[i].HydroAccel[k] = 0;//CellP[i].dMomentum[k] = 0;//manifest-indiv-timestep-debug//
#ifdef MAGNETIC
                for(k = 0; k < 3; k++) CellP[i].DtB[k] = 0;
#ifdef DIVBCLEANING_DEDNER
                for(k = 0; k < 3; k++) CellP[i].DtB_PhiCorr[k] = 0;
                CellP[i].DtPhi = 0;
#endif
#endif
#ifdef SPH_BND_BFLD
                for(k = 0; k < 3; k++) CellP[i].B[k] = 0;
#endif
            }
#endif

        } // closes P[i].Type==0 check and so closes loop over particles i
    } // for (loop over active particles) //

    
#ifdef TURB_DRIVING
#ifdef TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE // if this is enabled, we only update as frequently as the driving phases are recomputed, as set by TurbDrive_TimeBetweenTurbUpdates. To avoid large errors, must be set by-hand to be << lambda_min / V where V is the typical turbulent velocity and lambda_min is the smallest driven wavelength.
    if(new_turbforce_needed_this_timestep()){add_turb_accel();}
#else    
    add_turb_accel(); // update turbulent driving fields and TurbAccel fields at same time as update HydroAccel, here
#endif    
#endif

}




/* this function exists to loop over the hydro variables and do any needed 'pre-processing' before they enter the primary hydro force loop */
void hydro_force_initial_operations_preloop(void)
{
    // Set global factors for comoving integration of hydro //
    fac_mu = 1 / (All.cf_atime); // code_vel * fac_mu = sqrt[code_pressure/code_density] = code_soundspeed //
    fac_vsic_fix = All.cf_hubble_a ; // note also that signal_vel in forms below should be in units of code_soundspeed //
#ifdef MAGNETIC
    fac_magnetic_pressure = 1. / All.cf_atime; // code_Bfield*code_Bfield * fac_magnetic_pressure = code_pressure -- use this to get alfven velocities, etc, as well as comoving units for magnetic integration //
#endif

    /* need to zero out all numbers that can be set -EITHER- by an active particle in the domain, or by one of the neighbors we will get sent */
    int i, k;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type==0)
        {
            CellP[i].MaxSignalVel = MIN_REAL_NUMBER;
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
            CellP[i].MaxKineticEnergyNgb = MIN_REAL_NUMBER;
#endif
            CellP[i].DtInternalEnergy = 0; //CellP[i].dInternalEnergy = 0;//manifest-indiv-timestep-debug//
            for(k=0;k<3;k++) {CellP[i].HydroAccel[k] = 0;} //CellP[i].dMomentum[k] = 0;//manifest-indiv-timestep-debug//
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            CellP[i].DtMass = 0; CellP[i].dMass = 0; for(k=0;k<3;k++) CellP[i].GravWorkTerm[k] = 0;
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
            for(k=0;k<NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;k++) {CellP[i].Dyield[k] = 0;}
#endif
#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
            for(k=0;k<N_RT_FREQ_BINS;k++) {CellP[i].Dt_Rad_E_gamma[k] = 0;}
#endif
#if defined(RT_EVOLVE_FLUX)
            for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {CellP[i].Dt_Rad_Flux[k][k_dir] = 0;}}
#endif
#if defined(RT_INFRARED)
            CellP[i].Dt_Rad_E_gamma_T_weighted_IR = 0;
#endif
#if defined(RT_EVOLVE_FLUX)
            for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<3;k_dir++) {CellP[i].Dt_Rad_Flux[k][k_dir] = 0;}}
#endif
#if defined(RT_EVOLVE_INTENSITIES)
            for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<N_RT_INTENSITY_BINS;k_dir++) {CellP[i].Dt_Rad_Intensity[k][k_dir] = 0;}}
#endif
#endif
#ifdef MAGNETIC
            CellP[i].divB = 0; for(k=0;k<3;k++) {CellP[i].Face_Area[k] = 0;}
#ifdef DIVBCLEANING_DEDNER
            for(k=0;k<3;k++) {CellP[i].DtB_PhiCorr[k] = 0;}
#endif
#ifndef HYDRO_SPH
            for(k=0;k<3;k++) {CellP[i].DtB[k] = 0;}
#ifdef DIVBCLEANING_DEDNER
            CellP[i].DtPhi = 0;
#endif
#endif
#endif // magnetic //
#ifdef COSMIC_RAY_FLUID
            CellP[i].Face_DivVel_ForAdOps = 0;
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
            CellP[i].DtCREgyNewInjectionFromShocks = 0;
#endif
            for(k=0;k<N_CR_PARTICLE_BINS;k++)
            {
                CellP[i].DtCosmicRayEnergy[k] = 0;
#if defined(CRFLUID_EVOLVE_SPECTRUM)
                CellP[i].DtCosmicRay_Number_in_Bin[k] = 0;
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
                int kAlf; for(kAlf=0;kAlf<2;kAlf++) {CellP[i].DtCosmicRayAlfvenEnergy[k][kAlf] = 0;}
#endif
            }
#endif
#ifdef WAKEUP
            P[i].wakeup = 0;
#endif
        }
}





/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/*! This function is the driver routine for the calculation of hydrodynamical
 *  force, fluxes, etc. */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
void hydro_force(void)
{
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second();
    hydro_force_initial_operations_preloop(); /* do initial pre-processing operations as needed before main hydro force loop */
    #include "../system/code_block_xchange_perform_ops_malloc.h" /* this calls the large block of code which contains the memory allocations for the MPI/OPENMP/Pthreads parallelization block which must appear below */
    #include "../system/code_block_xchange_perform_ops.h" /* this calls the large block of code which actually contains all the loops, MPI/OPENMP/Pthreads parallelization */
    #include "../system/code_block_xchange_perform_ops_demalloc.h" /* this de-allocates the memory for the MPI/OPENMP/Pthreads parallelization block which must appear above */
    hydro_final_operations_and_cleanup(); /* do final operations on results */
    /* collect timing information */
    double t1; t1 = WallclockTime = my_second(); timeall = timediff(t00_truestart, t1);
    CPU_Step[CPU_HYDCOMPUTE] += timecomp; CPU_Step[CPU_HYDWAIT] += timewait; CPU_Step[CPU_HYDCOMM] += timecomm;
    CPU_Step[CPU_HYDMISC] += timeall - (timecomp + timewait + timecomm);
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */
