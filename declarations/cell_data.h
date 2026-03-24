/* the following struture holds data that is stored for each fluid cell in addition to the collisionless variables. */
extern struct gas_cell_data
{
    /* the PRIMITIVE and CONSERVED hydro variables used in STATE reconstruction */
    MyDouble Density;               /*!< current baryonic mass density of particle */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble MassTrue;              /*!< true particle mass ('mass' now is -predicted- mass */
    MyDouble dMass;                 /*!< change in particle masses from hydro step (conserved variable) */
    MyDouble DtMass;                /*!< rate-of-change of particle masses (for drifting) */
    Vec3<MyDouble> GravWorkTerm;    /*!< correction term needed for hydro mass flux in gravity */
    Vec3<MyDouble> ParticleVel;     /*!< actual velocity of the mesh-generating points */
#endif
    
    MyDouble Pressure;              /*!< current pressure */
    MyDouble InternalEnergy;        /*!< specific internal energy [internal thermal energy per unit mass] of cell */
    MyDouble InternalEnergyPred;    /*!< predicted value of the specific internal energy at the current time */
    MyDouble DtInternalEnergy;      /*!< rate of change of specific internal energy */
    
    Vec3<MyDouble> VelPred;         /*!< predicted gas cell velocity at the current time */
    Vec3<MyDouble> HydroAccel;      /*!< acceleration due to hydrodynamical force (for drifting) */
    
#ifdef HYDRO_EXPLICITLY_INTEGRATE_VOLUME
    MyDouble Density_ExplicitInt;   /*!< explicitly integrated volume/density variable to be used if integrating the SPH-like form of the continuity directly */
#endif
    
#ifdef HYDRO_VOLUME_CORRECTIONS
    MyDouble Volume_0;              /*!< 0th-order cell volume for mesh-free (MFM/MFV-type) reconstruction at 0th-order volume quadrature */
    MyDouble Volume_1;              /*!< 1st-order cell volume for mesh-free (MFM/MFV-type) reconstruction at 1st-order volume quadrature */
#endif
    
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    MyDouble ISMDustChem_Dust_Source[NUM_ISMDUSTCHEM_SOURCES];  /*!< amount of dust from each source of dust creation. 0=gas-dust accretion, 1=Sne Ia, 2=SNe II, 3=AGB */
    MyDouble ISMDustChem_Dust_Metal[NUM_ISMDUSTCHEM_ELEMENTS];  /*!< metallicity (species-by-species) of dust */
    MyDouble ISMDustChem_Dust_Species[NUM_ISMDUSTCHEM_SPECIES]; /*!< metallicity of dust species types. 0=silicates, 1=carbon, 2=SiC, 3=free-flying iron, (optional) 4=oxygen reservoir, (optional) 5=iron inclusions in silicates */
    MyDouble ISMDustChem_DelayTimeSNeSputtering;       /*!< delay time for thermal sputtering due to recent SNe, used to not double count dust destruction with thermal sputtering */
    MyDouble ISMDustChem_C_in_CO;                      /*!< C metallicity locked in CO */
    MyDouble ISMDustChem_MassFractionInDenseMolecular; /*!< mass fraction of gas in dense MC phase */
#endif
    
#ifdef MAGNETIC
    Vec3<MyDouble> Face_Area;       /*!< vector sum of effective areas of 'faces'; this is used to check closure for meshless methods */
    Vec3<MyDouble> BPred;           /*!< current magnetic field strength */
    Vec3<MyDouble> BField_prerefinement; /*!< safety variable that stores the B-field before a refinement-type operation to allow it to be more conservatively reset correctly after the (de)refinement completes */
    Vec3<MyDouble> B;               /*!< actual B (conserved variable used for integration; can be B*V for flux schemes) */
    Vec3<MyDouble> DtB;             /*!< time derivative of B-field (of -conserved- B-field) */
    MyFloat divB;                   /*!< storage for the 'effective' divB used in div-cleaning procedure */
#ifdef DIVBCLEANING_DEDNER
    Vec3<MyDouble> DtB_PhiCorr;     /*!< correction forces for mid-face update to phi-field */
    MyDouble PhiPred;               /*!< current value of Phi */
    MyDouble Phi;                   /*!< scalar field for Dedner divergence cleaning */
    MyDouble DtPhi;                 /*!< time derivative of Phi-field */
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
    int FlagForConstrainedGradients;/*!< flag indicating whether the B-field gradient is a 'standard' one or the constrained-divB version */
#endif
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
    MyFloat Balpha;                 /*!< effective resistivity coefficient */
#endif
#endif /* MAGNETIC */
    
#if defined(KERNEL_CRK_FACES)
    MyFloat Tensor_CRK_Face_Corrections[16]; /*!< tensor set for face-area correction terms for the CRK formulation of SPH or MFM/V areas */
#endif

#ifdef COSMIC_RAY_FLUID
    MyFloat CosmicRayEnergy[N_CR_PARTICLE_BINS];        /*!< total energy of cosmic ray fluid (the conserved variable) */
    MyFloat CosmicRayEnergyPred[N_CR_PARTICLE_BINS];    /*!< total energy of cosmic ray fluid (the conserved variable) */
    MyFloat DtCosmicRayEnergy[N_CR_PARTICLE_BINS];      /*!< time derivative of cosmic ray energy */
    MyFloat CosmicRayDiffusionCoeff[N_CR_PARTICLE_BINS];/*!< diffusion coefficient kappa for cosmic ray fluid */
    MyFloat Face_DivVel_ForAdOps;                       /*!< face-centered definition of the velocity divergence, needed to carefully handle adiabatic terms when Pcr >> Pgas */
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    MyFloat DtCREgyNewInjectionFromShocks;              /*!< scalar to record energy injection at shock interfaces for acceleration from resolved shocks */
#endif
#if defined(SINK_CR_INJECTION_AT_TERMINATION)
    MyDouble Sink_CR_Energy_Available_For_Injection;     /*!< Energy reservoir from CRs */
#endif
    Vec3<MyFloat> CosmicRayFlux[N_CR_PARTICLE_BINS];       /*!< CR flux vector [explicitly evolved] - conserved-variable */
    Vec3<MyFloat> CosmicRayFluxPred[N_CR_PARTICLE_BINS];   /*!< CR flux vector [explicitly evolved] - conserved-variable */
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
    MyFloat CosmicRayAlfvenEnergy[N_CR_PARTICLE_BINS][2];       /*!< forward and backward-traveling Alfven wave-packet energies */
    MyFloat CosmicRayAlfvenEnergyPred[N_CR_PARTICLE_BINS][2];   /*!< drifted forward and backward-traveling Alfven wave-packet energies */
    MyFloat DtCosmicRayAlfvenEnergy[N_CR_PARTICLE_BINS][2];     /*!< time derivative fof forward and backward-traveling Alfven wave-packet energies */
#endif
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    MyFloat CosmicRay_Number_in_Bin[N_CR_PARTICLE_BINS];         /*!< effective number of CRs in the bin, which we evolve alongside total energy. */
    MyFloat DtCosmicRay_Number_in_Bin[N_CR_PARTICLE_BINS];       /*!< time derivative of effective number of CRs in the bin, which we evolve alongside total energy. */
    MyFloat Flux_Number_to_Energy_Correction_Factor[N_CR_PARTICLE_BINS]; /*!< correction term to compute correct flux of number versus energy, since not identical for finite-bin-width effects. */
#endif
#endif
    
#ifdef SUPER_TIMESTEP_DIFFUSION
    MyDouble Super_Timestep_Dt_Explicit; /*!< records the explicit step being used to scale the sub-steps for the super-stepping */
    int Super_Timestep_j; /*!< records which sub-step if the super-stepping cycle the particle is in [needed for adaptive steps] */
#endif
    
#if (SINGLE_STAR_SINK_FORMATION & 4)
    MyFloat Density_Relative_Maximum_in_Kernel; /*!< hold density_max-density_i, for particle i, so we know if its a local maximum */
#endif
    
    /* matrix of the primitive variable gradients: rho, P, vx, vy, vz, B, phi */
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
#ifdef DOGRAD_SOUNDSPEED
        Vec3<MyDouble> SoundSpeed;
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        Vec3<MyDouble> InternalEnergy;
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        Vec3<MyDouble> Metallicity[NUM_METAL_SPECIES];
#endif
#ifdef COSMIC_RAY_FLUID
        Vec3<MyDouble> CosmicRayPressure[N_CR_PARTICLE_BINS];
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
        Vec3<MyDouble> Rad_E_gamma_ET[N_RT_FREQ_BINS];
#endif
    } Gradients;
    SymmetricTensor2<MyDouble> NV_T; /*!< holds the tensor used for gradient estimation */
    Vec3<MyDouble> NV_T_face_weights; /*!< weighted first moments sum(wk*dp[k]); used for face area estimation */
    MyDouble ConditionNumber;   /*!< condition number of the gradient matrix: needed to ensure stability */
    MyDouble FaceClosureError;      /*!< dimensionless measure of face closure */
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    MyDouble MaxKineticEnergyNgb;   /*!< maximum kinetic energy (with respect to neighbors): use for entropy 'switch' */
#endif
    
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    MyFloat Dyield[NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION];
#endif
    
#ifdef HYDRO_SPH
    MyDouble DrkernHydroSumFactor;   /* for 'traditional' SPH, we need the SPH hydro-element volume estimator */
#endif
    
#ifdef HYDRO_PRESSURE_SPH
    MyDouble EgyWtDensity;          /*!< 'effective' rho to use in hydro equations */
#endif
    
    MyFloat MaxSignalVel;           /*!< maximum signal velocity (needed for time-stepping) */
    int recent_refinement_flag;     /*!< key that tells the code this cell was just refined or de-refined, to know to treat some other operations with care */
    
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    MyFloat Rad_Flux_UV;              /*!< local UV field strength */
    MyFloat Rad_Flux_EUV;             /*!< local (ionizing/hard) UV field strength */
#endif
    
#if defined(SINK_WIND_SPAWN_SET_BFIELD_POLTOR)
    MyDouble IniDen;
    Vec3<MyDouble> IniB;
#endif
    
#ifdef CHIMES_STELLAR_FLUXES
    double Chimes_G0[CHIMES_LOCAL_UV_NBINS];            /*!< 6-13.6 eV flux, in Habing units */
    double Chimes_fluxPhotIon[CHIMES_LOCAL_UV_NBINS];   /*!< ionising flux (>13.6 eV), in cm^-2 s^-1 */
#ifdef CHIMES_HII_REGIONS
    double Chimes_G0_HII[CHIMES_LOCAL_UV_NBINS];
    double Chimes_fluxPhotIon_HII[CHIMES_LOCAL_UV_NBINS];
#endif
#endif
#ifdef CHIMES_TURB_DIFF_IONS
    double ChimesNIons[CHIMES_TOTSIZE];
#endif
#ifdef SINK_COMPTON_HEATING
    MyFloat Rad_Flux_AGN;             /*!< local AGN flux */
#endif
    
    
#if defined(TURB_DRIVING) || defined(OUTPUT_VORTICITY)
    Vec3<MyFloat> Vorticity;
    Vec3<MyFloat> SmoothedVel;
#endif
    
#if defined(SINK_THERMALFEEDBACK)
    MyDouble Injected_Sink_Energy;
#endif
    
#ifdef COOLING
#if !defined(COOLING_OPERATOR_SPLIT)
    int CoolingIsOperatorSplitThisTimestep; /* flag to tell us if cooling is operator split or not on a given timestep */
#endif
#ifndef CHIMES
    MyFloat Ne;  /*!< electron fraction, expressed as local electron number density normalized to the hydrogen number density. Gives indirectly ionization state and mean molecular weight. */
#endif
#endif
#ifdef GALSF
    MyFloat Sfr;                      /*!< particle star formation rate */
#if defined(GALSF_SFR_VIRIAL_CRITERION_TIMEAVERAGED)
    MyFloat AlphaVirial_SF_TimeSmoothed;  /*!< dimensionless number > 0.5 if self-gravitating for smoothed virial criterion */
#endif
#endif
#ifdef GALSF_SUBGRID_WINDS
    MyFloat DelayTime;                /*!< remaining maximum decoupling time of wind particle */
#if (GALSF_SUBGRID_WIND_SCALING==1)
    MyFloat HostHaloMass;             /*!< host halo mass estimator for wind launching velocity */
#endif
#if (GALSF_SUBGRID_WIND_SCALING==2)
    MyFloat  KernelRadiusDM;                   /*!< smoothing length to find neighboring dark matter particles */
    MyDouble NumNgbDM;                /*!< number of neighbor dark matter particles */
    MyDouble DM_Vx;
    MyDouble DM_Vy;
    MyDouble DM_Vz;
    MyDouble DM_VelDisp; /*!< surrounding DM velocity and velocity dispersion */
#endif
#endif
    
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
    MyFloat DelayTimeHII;             /*!< flag indicating particle is ionized by nearby star */
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
    MyFloat DelayTimeCoolingSNe;      /*!< flag indicating cooling is suppressed b/c heated by SNe */
#endif
    
#ifdef TURB_DRIVING
    MyDouble DuDt_diss;               /*!< quantities specific to turbulent driving routines */
    MyDouble DuDt_drive;
    MyDouble EgyDiss;
    MyDouble EgyDrive;
    Vec3<MyDouble> TurbAccel;
#endif
    
#ifdef TURB_DIFFUSION
    MyFloat TD_DiffCoeff;             /*!< effective diffusion coefficient for sub-grid turbulent diffusion */
#ifdef TURB_DIFF_DYNAMIC
    MyDouble h_turb;
    MyDouble MagShear;
    MyFloat TD_DynDiffCoeff;          /*!< improved Smag. coefficient (squared) for sub-grid turb. diff. - D. Rennehan */
#endif
#endif
    
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
    MyFloat NV_DivVel;                /*!< quantities specific to the Cullen & Dehnen viscosity switch */
    MyFloat NV_dt_DivVel;
    Mat3<MyFloat> NV_A;
    Mat3<MyFloat> NV_D;
    MyFloat NV_trSSt;
    MyFloat alpha;
#endif
    
#ifdef HYDRO_SPH
    MyFloat alpha_limiter;                /*!< artificial viscosity limiter (Balsara-like) */
#endif
    
#ifdef CONDUCTION
    MyFloat Kappa_Conduction;                   /*!< conduction coefficient */
#endif
    
#if defined(OUTPUT_MOLECULAR_FRACTION) || defined(COOL_MOLECFRAC_NONEQM)
    MyFloat MolecularMassFraction;              /*!< holder for molecular mass fraction for sims where we evaluate it on-the-fly and wish to save it [different from detailed chemistry modules] */
#if defined(COOL_MOLECFRAC_NONEQM)
    MyFloat MolecularMassFraction_perNeutralH;  /*! molecular mass fraction -of-the-neutral-gas-, which we retain as a separate variable since we have a hybrid model here using implicit updates for the ionization fraction */
#endif
#endif
    
#ifdef MHD_NON_IDEAL
    MyFloat Eta_MHD_OhmicResistivity_Coeff;     /*!< Ohmic resistivity coefficient [physical units of L^2/t] */
    MyFloat Eta_MHD_HallEffect_Coeff;           /*!< Hall effect coefficient [physical units of L^2/t] */
    MyFloat Eta_MHD_AmbiPolarDiffusion_Coeff;   /*!< Hall effect coefficient [physical units of L^2/t] */
#endif
    
    
#if defined(VISCOSITY)
    MyFloat Eta_ShearViscosity;         /*!< shear viscosity coefficient */
    MyFloat Zeta_BulkViscosity;         /*!< bulk viscosity coefficient */
#endif
    
    
#if defined(RADTRANSFER)
    SymmetricTensor2<MyFloat> ET[N_RT_FREQ_BINS]; /*!< eddington tensor - symmetric -> only 6 elements needed: this is dimensionless by our definition */
    MyFloat Rad_Je[N_RT_FREQ_BINS];         /*!< emissivity (includes sources like stars, as well as gas): units=Rad_E_gamma/time  */
    MyFloat Rad_E_gamma[N_RT_FREQ_BINS];    /*!< photon energy (integral of dRad_E_gamma/dvol*dVol) associated with particle [for simple frequency bins, equivalent to photon number] */
    MyFloat Rad_Kappa[N_RT_FREQ_BINS];      /*!< opacity [physical units ~ length^2 / mass]  */
#if defined(COOLING) || defined(RT_INFRARED)
    MyFloat Lambda_RadiativeCooling_toRHDBins[N_RT_FREQ_BINS]; /* cooling rate to the various RHD bins here which is not entirely accounted for elsewhere */
#endif
#ifdef RT_FLUXLIMITER
    MyFloat Rad_Flux_Limiter[N_RT_FREQ_BINS]; /*!< dimensionless flux-limiter (0<lambda<1) */
#endif
#ifdef RT_EVOLVE_INTENSITIES
    MyFloat Rad_Intensity[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS]; /*!< intensity values along different directions, for each frequency */
    MyFloat Rad_Intensity_Pred[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS]; /*!< predicted [drifted] values of intensities */
    MyFloat Dt_Rad_Intensity[N_RT_FREQ_BINS][N_RT_INTENSITY_BINS]; /*!< time derivative of intensities */
#endif
#ifdef RT_EVOLVE_FLUX
    Vec3<MyFloat> Rad_Flux[N_RT_FREQ_BINS];    /*!< photon energy flux density (energy/time/area), for methods which track this explicitly (e.g. M1) */
    Vec3<MyFloat> Rad_Flux_Pred[N_RT_FREQ_BINS];/*!< predicted photon energy flux density for drift operations (needed for adaptive timestepping) */
    Vec3<MyFloat> Dt_Rad_Flux[N_RT_FREQ_BINS]; /*!< time derivative of photon energy flux density */
#else
#define Rad_Flux_Pred Rad_Flux
#endif
#ifdef RT_EVOLVE_ENERGY
    MyFloat Rad_E_gamma_Pred[N_RT_FREQ_BINS]; /*!< predicted Rad_E_gamma for drift operations (needed for adaptive timestepping) */
    MyFloat Dt_Rad_E_gamma[N_RT_FREQ_BINS]; /*!< time derivative of photon number in particle (used only with explicit solvers) */
#else
#define Rad_E_gamma_Pred Rad_E_gamma        /*! define a useful shortcut for use throughout code so we don't have to worry about Pred-vs-true difference */
#endif
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
    MyDouble Interpolated_Opacity[N_RT_FREQ_BINS]; /* opacity values interpolated to gas positions */
    MyDouble InterpolatedGeometricDustCrossSection; /* geometric opacity (frequency independent) */
#endif
#ifdef RT_INFRARED
    MyFloat Radiation_Temperature; /* IR radiation field temperature (evolved variable ^4 power, for convenience) */
    MyFloat Dt_Rad_E_gamma_T_weighted_IR; /* IR radiation temperature-weighted time derivative of photon energy (evolved variable ^4 power, for convenience) */
    MyFloat Dust_Temperature; /* Dust temperature (evolved variable ^4 power, for convenience) */
#ifdef COOLING
    MyFloat Radiation_Temperature_CoolingWeighted; /* Radiation temperature weighted to combine dust+gas emission with existing SED in cooling solver */
#endif
#endif
#ifdef RT_CHEM_PHOTOION
    MyFloat HI;                  /* HI fraction */
    MyFloat HII;                 /* HII fraction */
#ifndef COOLING
    MyFloat Ne;               /* electron fraction */
#endif
#ifdef RT_CHEM_PHOTOION_HE
    MyFloat HeI;                 /* HeI fraction */
    MyFloat HeII;                 /* HeII fraction */
    MyFloat HeIII;                 /* HeIII fraction */
#endif
#endif // end of chem-photoion
#endif // end of radtransfer
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY) && !defined(RADTRANSFER)
    MyFloat Rad_E_gamma[N_RT_FREQ_BINS];
#define Rad_E_gamma_Pred Rad_E_gamma
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX) && !defined(RT_EVOLVE_FLUX)
    Vec3<MyFloat> Rad_Flux[N_RT_FREQ_BINS];
#define Rad_Flux_Pred Rad_Flux
#endif
    
#ifdef RT_RAD_PRESSURE_OUTPUT
    Vec3<MyFloat> Rad_Accel;
#endif
    
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat SubGrid_CosmicRayEnergyDensity;
#endif
    
#ifdef EOS_GENERAL
    MyFloat SoundSpeed;                   /* Sound speed */
#ifdef EOS_CARRIES_TEMPERATURE
    MyFloat Temperature;                  /* Temperature */
#endif
#ifdef EOS_CARRIES_GAMMA
    MyFloat Gamma;                        /* First adiabatic index */
#endif
#ifdef EOS_CARRIES_YE
    MyFloat Ye;                           /* Electron fraction */
#endif
#ifdef EOS_CARRIES_ABAR
    MyFloat Abar;                         /* Average atomic weight (in atomic mass units) */
#endif
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC)
    int CompositionType;                  /* define the composition of the material */
#endif
#ifdef EOS_ELASTIC
    Mat3<MyDouble> Elastic_Stress_Tensor; /* deviatoric stress tensor */
    Mat3<MyDouble> Elastic_Stress_Tensor_Pred;
    Mat3<MyDouble> Dt_Elastic_Stress_Tensor;
#endif
#endif
    
#if defined(OUTPUT_COOLRATE_DETAIL) && defined(COOLING)
    MyFloat CoolingRate;
    MyFloat HeatingRate;
    MyFloat NetHeatingRateQ;
    MyFloat HydroHeatingRate;
    MyFloat MetalCoolingRate;
#endif
    
#if defined(COOLING) && defined(COOL_GRACKLE)
#if (COOL_GRACKLE_CHEMISTRY >= 1)
    gr_float grHI;
    gr_float grHII;
    gr_float grHM;
    gr_float grHeI;
    gr_float grHeII;
    gr_float grHeIII;
#endif
#if (COOL_GRACKLE_CHEMISTRY >= 2)
    gr_float grH2I;
    gr_float grH2II;
#endif
#if (COOL_GRACKLE_CHEMISTRY >= 3)
    gr_float grDI;
    gr_float grDII;
    gr_float grHDI;
#endif
#endif
    
#ifdef TURB_DIFF_DYNAMIC
    Mat3<MyDouble> VelShear_bar;
    MyDouble MagShear_bar;
    Vec3<MyDouble> Velocity_bar;
    Vec3<MyDouble> Velocity_hat;
    MyFloat FilterWidth_bar;
    MyFloat MaxDistance_for_grad;
    MyDouble Norm_hat;
    MyDouble Dynamic_numerator;
    MyDouble Dynamic_denominator;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
    MyDouble TD_DynDiffCoeff_error;
    MyDouble TD_DynDiffCoeff_error_default;
#endif
#endif
    
}
*CellP,                /*!< holds gas cell data on local processor */
*DomainGasBuf;            /*!< buffer for gas cell data in domain decomposition */


