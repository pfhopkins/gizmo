/* hydrogen_molecule_gamma is needed by gamma_eos_value() below.
   Include the inline definition directly so GPU TUs get it inlined
   rather than generating an unresolvable external device call. */
#ifdef EOS_SUBSTELLAR_ISM
#include "../eos/hydrogen_molecule_functions.h"
#endif
#if defined(EOS_DAMAGE_POROSITY)
#include "../solids/jutzi_crush_curve.h"
#endif

/* the following struture holds data that is stored for each fluid cell in addition to the collisionless variables.
   On Kokkos builds, the struct and its inline member functions must be device-compilable
   since they are called from within the GPU-offloaded cooling loop. */
#ifndef __HIP__
#pragma omp begin declare target
#endif
extern struct gas_cell_data
{
    /* the PRIMITIVE and CONSERVED hydro variables used in STATE reconstruction */
    MyDouble Mass;                  /*!< gas cell mass — authoritative for gas (Type==0); synced to P[i].Mass at tree builds and type conversions */
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
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
    Vec3<MyDouble> GradH_numer;     /*!< numerator sum for kernel support gradient: sum_{j!=i} (dwk/r) * dp */
    MyDouble GradH_denom;           /*!< denominator sum for kernel support gradient: sum_{j!=i} u * dwk */
#endif
    
#ifdef OUTPUT_MACH_NUMBER
    MyDouble ISMDustChem_MachNumber;               /*!< mach number used for sub-resolution density enhancements from turbulence */
#endif
#ifdef OUTPUT_SHOCK_MACH_NUMBER
    MyFloat ShockMachNumber;  /*!< estimated local shock Mach number from pairwise Riemann problem */
#endif
#ifdef GALSF_ISMDUSTCHEM_MODEL
    MyDouble ISMDustChem_Dust_Source[NUM_ISMDUSTCHEM_SOURCES];  /*!< amount of dust from each source of dust creation. 0=gas-dust accretion, 1=Sne Ia, 2=SNe II, 3=AGB */
    MyDouble ISMDustChem_Dust_Metal[NUM_ISMDUSTCHEM_ELEMENTS];  /*!< metallicity (species-by-species) of dust */
    MyDouble ISMDustChem_Dust_Species[NUM_ISMDUSTCHEM_SPECIES]; /*!< metallicity of dust species types */
    MyDouble ISMDustChem_DelayTimeSNeSputtering;       /*!< delay time for thermal sputtering due to recent SNe, used to not double count dust destruction with thermal sputtering */
#if (!defined(RADTRANSFER) && !defined(RT_INFRARED)) && (defined(OUTPUT_DUST_TEMPERATURE) && (GALSF_FB_FIRE_STELLAREVOLUTION > 2))
    MyFloat Dust_Temperature;
#endif
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
    MyDouble ISMDustChem_Dust_NumberInBin[NUM_ISMDUSTCHEM_SPECIES][NUM_ISMDUSTCHEM_SIZE_BINS];
    MyDouble ISMDustChem_Dust_SlopeInBin[NUM_ISMDUSTCHEM_SPECIES][NUM_ISMDUSTCHEM_SIZE_BINS];
#else
    MyDouble ISMDustChem_C_in_CO;                      /*!< C metallicity locked in CO */
    MyDouble ISMDustChem_MassFractionInDenseMolecular; /*!< mass fraction of gas in dense MC phase */
#endif
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
#ifdef MHD_MODIFIED_GRADIENT
    MyDouble MG_cgcoeff;            /*!< scalar correction coefficient for modified-gradient (MG) exact div(B)=0 method (Tu et al. 2026) */
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
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        MyFloat Rad_E_gamma_Grad[N_RT_FREQ_BINS][3];
        MyFloat Rad_Flux_Grad[N_RT_FREQ_BINS][3][3];
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        Vec3<MyDouble> ElectronNumberDensity; /*!< grad(n_e) for Biermann battery */
        Vec3<MyDouble> ElectronTemperature;   /*!< grad(T_e) for Biermann battery */
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        Mat3<MyDouble> E_battery_T2;          /*!< gradient tensor of cell-centered Tier-2 battery EMF (radiative-ionization + dust). curl(E_battery_T2) gives -dB/dt|_battery_T2 / c via Mat3::curl(). Filled in gradient pass after E_battery_T2_cell is populated by the per-cell Tier-2 source builders (radiative_E_RI in eos.cc, dust battery in solids/). */
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
    MyFloat Dyield[NUM_METAL_SPECIES];
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
#endif

#ifdef DM_DISPERSION_LOOP_ACTIVE
    /* DM-dispersion support fields. Relocated out of GALSF_SUBGRID_WINDS
     * nesting so DM_HEATING-only builds (no winds) also have these fields
     * available, consistent with the dispersion loop being compiled. */
    MyFloat  KernelRadiusDM;          /*!< smoothing length to find neighboring dark matter particles */
    MyDouble NumNgbDM;                /*!< number of neighbor dark matter particles */
    MyDouble DM_Vx;
    MyDouble DM_Vy;
    MyDouble DM_Vz;
    MyDouble DM_VelDisp;              /*!< surrounding DM 1D velocity dispersion (physical) */
    MyDouble DM_Rho;                  /*!< kernel-weighted DM mass density at gas cell (physical) */
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

#ifdef MHD_BATTERY_MECHANISMS
#if (MHD_BATTERY_MECHANISMS & (2|4|8))
    Vec3<MyDouble> E_battery_T2_cell;           /*!< Tier-2 battery EMF E' [statvolt/cm in physical cgs, multiplied by the same code-unit conversion the gradient pass expects]. Sum of radiative-ionization + dust contributions. Populated by per-cell builders in eos/cooling and solids/. The gradient pass then takes grad(E_battery_T2_cell), and hydro_toplevel.cc applies dB/dt|_T2 = -c * curl(grad). */
#endif
#if (MHD_BATTERY_MECHANISMS & 8)
    Vec3<MyDouble> J_dust_cell;                 /*!< per-cell dust current J_d = -sum_grain (q_d n_d (v_d - v_g)) [physical cgs], summed from grain particles in gas-cell kernel. Repopulated each step before per-cell battery EMF assembly. Soliman, Hopkins & Squire 2025 Eq. 8. */
#endif
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
    MyFloat VolatileSpecies[GRAIN_NUM_VOLATILE_SPECIES]; /*!< gas-phase volatile mass fractions {H2O, CO, CO2, refractory-vapor}; coupled to grain-mantle bits 5 (COND, drains gas->grain) and 6 (SUBL, drains grain->gas) of GRAIN_EVOLUTION. Latent-heat exchange routed into InternalEnergy. */
#endif
#ifdef GIZMO_TRACK_ELECTRON_STATE
    MyDouble n_e_cell;                          /*!< electron number density [physical cgs] from cooling. Cached on SoA so the gradient pass can produce grad(n_e) for Biermann battery and the 2-T integrator can convert between u_e and T_e. Populated at end of cooling step. */
    MyDouble T_e_cell;                          /*!< electron temperature [Kelvin]. Under MHD_BATTERY_MECHANISMS-only this is a per-step cache equal to T_gas, written in eos.cc. Under TWO_TEMPERATURE_PLASMA this is the derived view of u_e_cell, written by the 2-T cooling integrator. Either way readers (Biermann battery, gradient pass, snapshot) consume it identically through T_e(). */
#endif
#ifdef TWO_TEMPERATURE_PLASMA
    MyDouble u_e_cell;                          /*!< specific electron internal energy per gas mass [code units; same units as InternalEnergy]: u_e = (3/2) n_e k_B T_e / rho. Primary state of the 2-T plasma module; evolved by the cooling integrator (Spitzer e-i exchange + electron-side radiative + Compton + PdV/dissipation partition). T_e_cell is the derived cache populated at end of each cooling step. */
#if (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    MyDouble DtInternalEnergy_FromConduction;   /*!< 2-T plasma bit 2: conduction-only contribution to DtInternalEnergy, accumulated from the hydro pair loop. Routed entirely to u_e in the cooling step (Spitzer-Härm thermal conduction is electron heat). The non-conduction hydro work continues to follow the f_e shock partition. Reset to zero each step alongside DtInternalEnergy. */
#endif
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
#ifdef TRANSPORT_SUBCYCLE
    MyFloat Transport_Dt_Subcycle;  /*!< transport-limited timestep, stored separately when subcycling (not folded into hydro dt) */
#if defined(RT_RAD_PRESSURE_FORCES) && defined(RT_EVOLVE_ENERGY)
    MyFloat Dt_Rad_E_gamma_Work[N_RT_FREQ_BINS]; /*!< saved rad pressure work contribution to Dt_Rad_E_gamma, added back each sub-step */
#endif
#if defined(RT_INFRARED) && defined(COOLING)
    MyFloat DtIE_IR_Subcycle; /*!< hydro-pass DtInternalEnergy, saved before subcycle loop to prevent IR heating accumulation */
#endif
#ifdef TRANSPORT_SUBCYCLE_COOLING
    MyFloat Dt_Transport_Subcycle_Saved; /*!< saved DtInternalEnergy (code units) before cooling, restored each sub-step */
#endif
#endif
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
    
    MyFloat Temperature;                  /* Temperature */
    MyFloat Gamma;                        /* First adiabatic index */

#ifdef EOS_GENERAL
    MyFloat SoundSpeed;                   /* Sound speed */
#ifdef EOS_CARRIES_YE
    MyFloat Ye;                           /* Electron fraction */
#endif
#ifdef EOS_CARRIES_ABAR
    MyFloat Abar;                         /* Average atomic weight (in atomic mass units) */
#endif
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
    int CompositionType;                  /* define the composition of the material */
#endif
#ifdef EOS_ANEOS
    int PhaseID;                          /* ANEOS material phase flag */
#endif
#ifdef EOS_ELASTIC
    Mat3<MyDouble> Elastic_Stress_Tensor; /* deviatoric stress tensor */
    Mat3<MyDouble> Elastic_Stress_Tensor_Pred;
    Mat3<MyDouble> Dt_Elastic_Stress_Tensor;
#endif
#ifdef EOS_DAMAGE_POROSITY
    MyFloat Damage;        /* Grady-Kipp scalar damage D in [0,1] */
    MyFloat Distention;    /* Jutzi P-alpha distention alpha in [1, alpha_0] */
    MyFloat ActiveCracks;  /* Grady-Kipp Weibull active-flaw bookkeeping */
#endif
#endif
    
#if defined(OUTPUT_COOLRATE_DETAIL) && defined(COOLING)
    MyFloat CoolingRate;
    MyFloat HeatingRate;
    MyFloat NetHeatingRateQ;
    MyFloat HydroHeatingRate;
    MyFloat MetalCoolingRate;
    MyFloat PElecHeatingRate;
#if defined(GALSF_ISMDUSTCHEM_MODEL) && defined(GALSF_ISMDUSTCHEM_HIGHTEMPDUSTCOOLING)
    MyFloat DustCoolingRate;
#endif
#endif
    
#ifdef NUCLEAR_NETWORK
    MyFloat NuclearEnergyGenerationRate;  /* specific nuclear energy generation rate [code units] */
    MyFloat NuclearBurningTimescale;      /* shortest burning timescale [code units], for timestep control */
#ifdef NUCLEAR_NETWORK_NEUTRINOS
    MyFloat NeutrinoLuminosity[3];        /* neutrino luminosity per flavor (e, ebar, x) [code units] */
    MyFloat NeutrinoMeanEnergy[3];        /* mean neutrino energy per flavor [MeV] */
#endif
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

    /* ---- member functions for derived quantities ---- */
    GIZMO_GPU_FUNCTION inline double nHcgs() const {return HYDROGEN_MASSFRAC * Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;} /*!< hydrogen number density in cgs */

    GIZMO_GPU_FUNCTION inline double density_for_energy() const { /*!< density used in energy equations (accounts for SPH pressure formulation) */
#ifdef HYDRO_PRESSURE_SPH
        return EgyWtDensity;
#endif
        return Density;
    }

    inline void enforce_temperature_floor() { /*!< clamp internal energy to minimum temperature */
        if(All.MinEgySpec) {
            if(InternalEnergy < All.MinEgySpec) {
                InternalEnergy = All.MinEgySpec;
                DtInternalEnergy = 0;
            }
        }
    }

    GIZMO_GPU_FUNCTION inline double flux_limiter(int k_freq) const { /*!< radiation flux limiter for band k_freq */
#ifdef RT_FLUXLIMITER
        return Rad_Flux_Limiter[k_freq];
#endif
        return 1;
    }

    inline double pressure() const {return Pressure;} /*!< gas pressure */
    inline double temperature() const {return Temperature;} /*!< gas temperature (must be precomputed) */
#ifdef GIZMO_TRACK_ELECTRON_STATE
    GIZMO_GPU_FUNCTION inline double T_e() const {return T_e_cell;} /*!< electron temperature [K]. Under battery-only it equals gas T (cached in eos.cc). Under TWO_TEMPERATURE_PLASMA it is the independently evolved value (derived from u_e_cell, cached after each cooling step). Consumers read through this accessor unchanged. */
    GIZMO_GPU_FUNCTION inline double n_e() const {return n_e_cell;} /*!< electron number density [physical cgs]; populated by the cooling pass. */
#endif

    GIZMO_GPU_FUNCTION double gamma_eos_value() const { /*!< effective adiabatic index */
#if defined(COOL_MOLECFRAC_NONEQM)
        double fH = HYDROGEN_MASSFRAC, f = MolecularMassFraction, xe = Ne;
        double f_mono = fH*(xe + 1.-f) + (1.-fH)/4., f_di = fH*f/2., gamma_mono=5./3., gamma_di=7./5.;
#ifdef EOS_SUBSTELLAR_ISM
        gamma_di = hydrogen_molecule_gamma(Temperature); // declared in proto.h, defined in eos/hydrogen_molecule.cc
#endif
        return 1. + (f_mono + f_di) / (f_mono/(gamma_mono-1.) + f_di/(gamma_di-1.));
#endif
        return GAMMA_DEFAULT;
    }

    GIZMO_GPU_FUNCTION inline double soundspeed2_from_u(double u) const { /*!< convert specific internal energy to soundspeed^2 */
        double g = gamma_eos_value(); return g * (g-1.) * u;
    }
    GIZMO_GPU_FUNCTION inline double thermal_soundspeed() const { /*!< thermal sound speed */
        return sqrt(soundspeed2_from_u(InternalEnergyPred));
    }

    GIZMO_GPU_FUNCTION inline double effective_soundspeed() const { /*!< effective soundspeed including non-thermal contributions */
#ifdef EOS_GENERAL
        return SoundSpeed;
#else
        return sqrt(gamma_eos_value() * Pressure / density_for_energy());
#endif
    }

    inline double fast_MHD_wavespeed() const { /*!< fast MHD wave speed: sqrt(cs^2 + vA^2) */
        double cs = thermal_soundspeed(), vA = Alfven_speed();
        return sqrt(cs*cs + vA*vA);
    }

    GIZMO_GPU_FUNCTION inline double rt_photon_number_density(int k) const { /*!< photon number density for RT band k */
#ifdef RT_CHEM_PHOTOION
        return Rad_E_gamma[k] * (Density*All.cf_a3inv/Mass) / (All.rt_nu_eff_eV[k]*ELECTRONVOLT_IN_ERGS/UNIT_ENERGY_IN_CGS);
#else
        return 0;
#endif
    }

    GIZMO_GPU_FUNCTION inline double velocity_gradient_norm() const { /*!< magnitude of velocity gradient tensor |grad v| in physical units */
        double dv2=0; for(int j=0;j<3;j++) {for(int k=0;k<3;k++) {double vt = Gradients.Velocity[j][k]*All.cf_a2inv;
            if(All.ComovingIntegrationOn) {if(j==k) {vt += All.cf_hubble_a;}}
            dv2 += vt*vt;}}
        return sqrt(dv2);
    }

    GIZMO_GPU_FUNCTION double Alfven_speed() const { /*!< Alfven speed */
#if defined(MAGNETIC)
        double bmag = (Bfield() * All.cf_a2inv).norm_sq();
        if(bmag > 0) {return sqrt(bmag / (MIN_REAL_NUMBER + Density*All.cf_a3inv));}
#endif
        return 0;
    }

    GIZMO_GPU_FUNCTION inline double Bfield_microGauss() const { /*!< B-field magnitude in microGauss */
        double Bmag=0;
#ifdef MAGNETIC
        Bmag = (Bfield() * All.cf_a2inv).norm_sq();
#else
        Bmag = 2.*Pressure*All.cf_a3inv;
#endif
        return UNIT_B_IN_GAUSS * sqrt(DMAX(Bmag,0.)) * 1.e6;
    }

    GIZMO_GPU_FUNCTION inline double Bfield_component(int k) const { /*!< B-field component k in code units (B*Vol = BPred * Density / Mass) */
#if defined(MAGNETIC)
        return BPred[k] * Density / Mass;
#endif
        return 0;
    }

    GIZMO_GPU_FUNCTION inline Vec3<double> Bfield() const { /*!< B-field vector in code units */
#if defined(MAGNETIC)
        double fac = Density / Mass;
        return BPred * fac;
#endif
        return {};
    }


    GIZMO_GPU_FUNCTION inline double Urad_eVcm3() const { /*!< radiation energy density in eV/cm^3 */
        double erad = 0.26*All.cf_a3inv/All.cf_atime;
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
        int kfreq; double e_units = (Density*All.cf_a3inv/Mass) * UNIT_PRESSURE_IN_EV;
        for(kfreq=0;kfreq<N_RT_FREQ_BINS;kfreq++) {erad+=Rad_E_gamma_Pred[kfreq]*e_units;}
#else
        double uRad_MW = 0.31 + 0.66, prefac_rad=1, rho_cgs=Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS;
        if(All.ComovingIntegrationOn) {double rhofac = rho_cgs / (1000.*COSMIC_BARYON_DENSITY_CGS);
            if(rhofac < 0.2) {prefac_rad=0;} else {if(rhofac > 200.) {prefac_rad=1;} else {prefac_rad=exp(-1./(rhofac*rhofac));}}}
        prefac_rad *= rho_cgs/(0.01*PROTONMASS_CGS + rho_cgs);
        erad += uRad_MW * prefac_rad;
#endif
        return erad;
    }

#ifdef EOS_TILLOTSON
    inline double calculate_tillotson_eos() { /*!< Tillotson EOS: sets SoundSpeed, returns pressure */
        int type = CompositionType;
        double a=All.Tillotson_EOS_params[type][0], b=All.Tillotson_EOS_params[type][1],
        u0=All.Tillotson_EOS_params[type][2], rho0=All.Tillotson_EOS_params[type][3],
        A0=All.Tillotson_EOS_params[type][4], B0=All.Tillotson_EOS_params[type][5],
        u_s=All.Tillotson_EOS_params[type][6], u_s_prime=All.Tillotson_EOS_params[type][7],
        alpha=All.Tillotson_EOS_params[type][8], beta=All.Tillotson_EOS_params[type][9];
        double rho=Density, u=InternalEnergyPred;
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 4)
        /* Phase 17e bit 2: P-alpha porosity. Tillotson is evaluated at the matrix
         * density rho_s = alpha*rho_bulk; bulk pressure is P_matrix/alpha. */
        double alpha_d = (Distention >= 1.0) ? Distention : 1.0;
        rho *= alpha_d;
#endif
        double eta=rho/rho0, mu=eta-1, u_u0eta2=1+u/(u0*eta*eta), p0=u*rho, z=1/eta-1, press=0, cs=0, press_min=1.e-10*u0*rho0;
        double Pc = (a + b/u_u0eta2)*p0 + A0*mu + B0*mu*mu;
        double c2c_rho = (1+a+b/u_u0eta2)*Pc + A0+B0*(eta*eta-1) + b*(u_u0eta2-1)*(2*p0-Pc)/(u_u0eta2*u_u0eta2);
        double Pe = a*p0 + (b/u_u0eta2*p0 + A0*mu * exp(-beta*z)) * exp(-alpha*z*z);
        double c2e_rho = (1+a+b/u_u0eta2*exp(-alpha*z*z))*Pe + A0*eta*(1+mu*(beta+2*alpha*z-eta)/(eta*eta))*exp(-beta*z-alpha*z*z)
        + b*p0/(u_u0eta2*u_u0eta2*eta*eta)*(2*alpha*z*u_u0eta2*eta + (Pe/(u0*rho)-2*u/u0))*exp(-alpha*z*z);
        double Px = (Pe*(u-u_s) + Pc*(u_s_prime - u)) / (u_s_prime - u_s);
        double c2x_rho = (c2c_rho*(u-u_s) + c2e_rho*(u_s_prime - u)) / (u_s_prime - u_s);
        if(u <= u_s) {press=Pc; cs=c2c_rho;} else {if(u >= u_s_prime) {press=Pe; cs=c2e_rho;} else {press=Px; cs=c2x_rho;}}
        if(cs <= 0) {cs=press_min;}
        SoundSpeed = cs/rho;
#ifdef EOS_ELASTIC
        SoundSpeed += All.Tillotson_EOS_params[CompositionType][10] / rho;
#endif
        SoundSpeed = sqrt(SoundSpeed);
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 4)
        /* Phase 17e bit 2: irreversibly update distention from current matrix
         * pressure, then convert matrix pressure to bulk porous pressure.
         * Sound speed retained as matrix-frame value (CFL-conservative). */
        {
            double alpha_0_p = All.Tillotson_EOS_params[CompositionType][15];
            double P_e_p     = All.Tillotson_EOS_params[CompositionType][16];
            double P_s_p     = All.Tillotson_EOS_params[CompositionType][17];
            double alpha_eq = jutzi_distention_eq8(press, alpha_0_p, P_e_p, P_s_p);
            double alpha_new = (alpha_eq < Distention) ? alpha_eq : Distention;
            if(alpha_new < 1.0) { alpha_new = 1.0; }
            Distention = alpha_new;
            if(alpha_new > 1.0) { press /= alpha_new; }
        }
#endif
        return press;
    }
#endif

}
*CellP,                /*!< holds gas cell data on local processor */
*DomainGasBuf;            /*!< buffer for gas cell data in domain decomposition */
#ifndef __HIP__
#pragma omp end declare target
#endif


