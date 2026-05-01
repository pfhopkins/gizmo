/* hydro_structs.h — Shared struct definitions for hydro force computation.
 *
 * Contains: Conserved_var_Riemann, kernel_hydra, hydro_data_in, hydro_data_out.
 * These are used by both hydro_toplevel.cc (CPU tree-walk path) and the GPU TU
 * (density_gpu.cc via hydro_functions.h).
 *
 * Extracted from hydro_toplevel.cc so both translation units share the same
 * layout-identical struct definitions.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef HYDRO_STRUCTS_H
#define HYDRO_STRUCTS_H

/* structure to hold fluxes being passed from the hydro sub-routine */
struct Conserved_var_Riemann
{
    MyDouble rho;
    MyDouble p;
    Vec3<MyDouble> v;
    MyDouble u;
    MyDouble cs;
#ifdef MAGNETIC
    Vec3<MyDouble> B;
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


/* kernel workspace for hydro force computation */
struct kernel_hydra
{
    Vec3<double> dp;
    double r, vsig, sound_i, sound_j;
    Vec3<double> dv; double vdotr2;
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


/* inputs to the hydro force routine */
struct hydro_data_in
{
    /* basic hydro variables */
    Vec3<MyDouble> Pos;
    Vec3<MyFloat> Vel;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    Vec3<MyFloat> ParticleVel;
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
        Vec3<MyDouble> Density;
        Vec3<MyDouble> Pressure;
        Mat3<MyDouble> Velocity;
#ifdef MAGNETIC
        Mat3<MyDouble> B;
#ifdef DIVBCLEANING_DEDNER
        Vec3<MyDouble> Phi;
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        Vec3<MyDouble> Metallicity[NUM_METAL_SPECIES];
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        Vec3<MyDouble> InternalEnergy;
#endif
#ifdef DOGRAD_SOUNDSPEED
        Vec3<MyDouble> SoundSpeed;
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_COMPGRAD_EDDINGTON_TENSOR)
        Vec3<MyDouble> Rad_E_gamma_ET[N_RT_FREQ_BINS];
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        MyFloat Rad_E_gamma_Grad[N_RT_FREQ_BINS][3];
        MyFloat Rad_Flux_Grad[N_RT_FREQ_BINS][3][3];
#endif
    } Gradients;
    SymmetricTensor2<MyDouble> NV_T;

#if defined(KERNEL_CRK_FACES)
    MyFloat Tensor_CRK_Face_Corrections[16];
#endif
#ifdef HYDRO_PRESSURE_SPH
    MyFloat EgyWtRho;
#endif

#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    MyFloat Metallicity[NUM_METAL_SPECIES];
#endif

#ifdef CHIMES_TURB_DIFF_IONS
    MyDouble ChimesNIons[CHIMES_TOTSIZE];
#endif

#ifdef RT_SOLVER_EXPLICIT
    MyDouble Rad_E_gamma[N_RT_FREQ_BINS];
    MyDouble Rad_Kappa[N_RT_FREQ_BINS];
    MyDouble RT_DiffusionCoeff[N_RT_FREQ_BINS];
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
    SymmetricTensor2<MyDouble> ET[N_RT_FREQ_BINS];
#endif
#ifdef RT_EVOLVE_FLUX
    Vec3<MyDouble> Rad_Flux[N_RT_FREQ_BINS];
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

#ifdef MHD_BATTERY_MECHANISMS
    Vec3<MyDouble> E_battery_cell;  /* per-cell comoving battery EMF, copied from CellP at pair-loop dispatch */
#endif

#ifdef VISCOSITY
    MyFloat Eta_ShearViscosity;
    MyFloat Zeta_BulkViscosity;
#endif

#ifdef MAGNETIC
    Vec3<MyFloat> BPred;
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
    MyFloat Balpha;
#endif
#ifdef DIVBCLEANING_DEDNER
    MyFloat PhiPred;
#endif
#ifdef MHD_MODIFIED_GRADIENT
    MyFloat MG_cgcoeff;
#endif
#endif // MAGNETIC //

#ifdef COSMIC_RAY_FLUID
    MyDouble CosmicRayPressure[N_CR_PARTICLE_BINS];
    MyDouble CosmicRayDiffusionCoeff[N_CR_PARTICLE_BINS];
    Vec3<MyDouble> CosmicRayFlux[N_CR_PARTICLE_BINS];
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

#if defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    int CompositionType;
#endif
#ifdef EOS_ELASTIC
    MyFloat Elastic_Stress_Tensor[3][3];
#endif

    int TimeBin;
};


/* outputs from the hydro force routine */
struct hydro_data_out
{
    Vec3<MyDouble> Acc;
    MyDouble DtInternalEnergy;
    MyFloat MaxSignalVel;
#ifdef OUTPUT_SHOCK_MACH_NUMBER
    MyFloat MaxShockMachNumber;
#endif
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    MyFloat MaxKineticEnergyNgb;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble DtMass;
    MyDouble dMass;
    Vec3<MyDouble> GravWorkTerm;
#endif

#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    MyFloat Dyield[NUM_METAL_SPECIES];
#endif

#ifdef CHIMES_TURB_DIFF_IONS
    MyDouble ChimesIonsYield[CHIMES_TOTSIZE];
#endif

#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    MyFloat Dt_Rad_E_gamma[N_RT_FREQ_BINS];
#endif
#if defined(RT_EVOLVE_FLUX)
    Vec3<MyFloat> Dt_Rad_Flux[N_RT_FREQ_BINS];
#endif
#if defined(RT_INFRARED)
    MyFloat Dt_Rad_E_gamma_T_weighted_IR;
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    MyFloat Dt_Rad_Intensity[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS];
#endif
#endif

#if defined(MAGNETIC)
    Vec3<MyDouble> Face_Area;
    Vec3<MyFloat> DtB;
    MyFloat divB;
#if defined(DIVBCLEANING_DEDNER)
#ifdef HYDRO_MESHLESS_FINITE_VOLUME // mass-based phi-flux
    MyFloat DtPhi;
#endif
    Vec3<MyFloat> DtB_PhiCorr;
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
};


#endif /* HYDRO_STRUCTS_H */
