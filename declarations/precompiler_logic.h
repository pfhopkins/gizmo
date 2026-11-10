/*! \file precompiler_logic.h
 *  \brief logic statements that need to be near the top of variable definitions.
 */
/* CHEMCOOL -> SGCHEM rename (2026-10): old option names stop the build */
#ifdef CHEMCOOL
#error "CHEMCOOL was renamed to SGCHEM"
#endif
#ifdef CHEMCOOL_CXX
#error "CHEMCOOL_CXX was removed: SGCHEM uses the C++ backend by default (SGCHEM_FORTRAN selects the legacy Fortran backend)"
#endif
#ifdef SGCHEM_CXX
#error "C++ is now the default; drop SGCHEM_CXX (use SGCHEM_FORTRAN for the legacy backend)"
#endif
#ifdef CHEMCOOL_DGR_INDEPENDENT_OF_METALLICITY
#error "CHEMCOOL_DGR_INDEPENDENT_OF_METALLICITY was renamed to SGCHEM_DGR_INDEPENDENT_OF_METALLICITY"
#endif
#ifdef CHEMCOOL_KEEP_IC_ABUNDANCES
#error "CHEMCOOL_KEEP_IC_ABUNDANCES was renamed to SGCHEM_KEEP_IC_ABUNDANCES"
#endif
#ifdef CHEMCOOL_LEGACY_COOLING
#error "CHEMCOOL_LEGACY_COOLING was renamed to SGCHEM_LEGACY_COOLING"
#endif
#ifdef CHEMCOOL_OMP_CROSSCHECK
#error "CHEMCOOL_OMP_CROSSCHECK was renamed to SGCHEM_OMP_CROSSCHECK"
#endif
#ifdef CHEMCOOL_TNOCHEM
#error "CHEMCOOL_TNOCHEM was renamed to SGCHEM_TNOCHEM"
#endif
#ifdef GALSF_CHEMCOOL_DEBUG_INPUT
#error "GALSF_CHEMCOOL_DEBUG_INPUT was renamed to GALSF_SGCHEM_DEBUG_INPUT"
#endif
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
#error "GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE was renamed to GALSF_SGCHEM_VARIABLE_XH_AND_ABHE"
#endif

/*------- Things that are always recommended (this must follow loading GIZMO_config.h!) -------*/
/* also many logical options to force 'parent' or 'top-level' flags to be enabled for the appropriate methods, if we have enabled something using those methods */

/* macro definition we will use throughout below, make sure this is defined up-top here */
#ifdef NOGRAVITY
#define SELFGRAVITY_OFF
#endif
#define DO_PREPROCESSOR_EXPAND_(VAL)  VAL ## 1
#define EXPAND_PREPROCESSOR_(VAL)     DO_PREPROCESSOR_EXPAND_(VAL) /* checks for a NON-ZERO value of this parameter */
#define CHECK_IF_PREPROCESSOR_HAS_NUMERICAL_VALUE_(VAL) !(EXPAND_PREPROCESSOR_(VAL) == 1) /* returns True if a non-zero int value of VAL is set */

/* Stella x FIRE-3 coupled-mode preset: FIRE-3 SSPs above the cell-mass threshold [Msun], Stella single stars below */
#ifdef GALSF_RESOLVEDISM_HYBRID_DEFAULTS
#define GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL
#define GALSF_RESOLVEDISM_HYBRID_M_SSP (GALSF_RESOLVEDISM_HYBRID_DEFAULTS) /* cells above this mass [Msun] form FIRE SSPs, at or below it Stella stars */
#ifndef FIRE_PHYSICS_DEFAULTS
#define FIRE_PHYSICS_DEFAULTS 3
#endif
#define GALSF_RESOLVEDISM
#define GALSF_RESOLVEDISM_FB
#define GALSF_RESOLVEDISM_SAMPLE_IMF
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#define GALSF_RESOLVEDISM_WINDS
#define GALSF_RESOLVEDISM_WINDS_CONTINUOUS
#define GALSF_RESOLVEDISM_WINDS_THERMAL
#define GALSF_RESOLVEDISM_TYPE_IA
#define GALSF_RESOLVEDISM_PHOTOION
#define GALSF_RESOLVEDISM_G0_VARIABLE
#define GALSF_RESOLVEDISM_RADPRESSURE
#endif

/* set default slope limiters */
#if !defined(SLOPE_LIMITER_TOLERANCE)
#define SLOPE_LIMITER_TOLERANCE 1
#endif

/* lock the 'default' hydro mode */
#if !(defined(HYDRO_MESHLESS_FINITE_VOLUME) || defined(HYDRO_MESHLESS_FINITE_MASS) || defined(HYDRO_DENSITY_SPH) || defined(HYDRO_PRESSURE_SPH)) // default solver is not defined
#if (defined(HYDRO_FIX_MESH_MOTION) && (HYDRO_FIX_MESH_MOTION != 7)) || defined(HYDRO_REGULAR_GRID)
#define HYDRO_MESHLESS_FINITE_VOLUME /* only makes sense to use this modules with this 'backbone' of MFV here */
#else
#define HYDRO_MESHLESS_FINITE_MASS   /* otherwise default to MFM if nothing is specified */
#endif
#endif

/* define the default mesh-motion assumption, if this is not provided by the user */
#if !defined(HYDRO_FIX_MESH_MOTION)
#if defined(HYDRO_REGULAR_GRID)
#define HYDRO_FIX_MESH_MOTION 0     /* default to non-moving for regular grids */
#else
#define HYDRO_FIX_MESH_MOTION 5     /* otherwise default to smoothed motion, only relevant for MFV (MFM/SPH will always move with flow) */
#endif
#endif

/* determine whether the mesh is adaptive via splitting/merging (refinement) or 'frozen' to the initial number of elements */
#if !defined(PREVENT_PARTICLE_MERGE_SPLIT) && (HYDRO_FIX_MESH_MOTION<5)
#define PREVENT_PARTICLE_MERGE_SPLIT  /* particle merging/splitting doesn't make sense with frozen grids */
#endif

#if defined(PARTICLE_MERGE_SPLIT_EVERY_TIMESTEP) && !defined(DISABLE_MAINTAIN_TREE_IN_REARRANGE)
#define MAINTAIN_TREE_IN_REARRANGE
#endif

#if (defined(HYDRO_DENSITY_SPH) || defined(HYDRO_PRESSURE_SPH)) && !defined(HYDRO_SPH)
#define HYDRO_SPH               /* top-level flag for SPH: must be enabled if any SPH method is used */
#endif
#ifdef HYDRO_SPH
#if !defined(SPH_DISABLE_CD10_ARTVISC) && !(defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)) // fancy viscosity switches assume positive pressures //
#define SPHAV_CD10_VISCOSITY_SWITCH        /* Enables Cullen & Dehnen 2010 'inviscid sph' (viscosity suppression outside shocks) */
#endif
#ifndef SPH_DISABLE_PM_CONDUCTIVITY
#define SPHAV_ARTIFICIAL_CONDUCTIVITY      /* Enables mixing entropy (J.Read's improved Price-Monaghan conductivity with Cullen-Dehnen switches) */
#endif
#endif

#if defined(DILATION_FOR_STELLAR_KINEMATICS_ONLY) && !defined(USE_TIMESTEP_DILATION_FOR_ZOOMS)
#define USE_TIMESTEP_DILATION_FOR_ZOOMS
#endif
#if defined(SPECIAL_POINT_WEIGHTED_MOTION) && !defined(SPECIAL_POINT_MOTION)
#define SPECIAL_POINT_MOTION (SPECIAL_POINT_WEIGHTED_MOTION) /* doesn't seem to be working ??? */
#endif
#if defined(SPECIAL_POINT_MOTION) && !defined(SINK_CALC_DISTANCES)
#define SINK_CALC_DISTANCES
#endif

#if defined(EOS_ELASTIC)
#if !defined(DISABLE_SURFACE_VOLCORR) && !defined(HYDRO_KERNEL_SURFACE_VOLCORR)
#define HYDRO_KERNEL_SURFACE_VOLCORR
#endif
#if !defined(DISABLE_EXPLICIT_VOLUME_INTEGRATION) && !defined(HYDRO_EXPLICITLY_INTEGRATE_VOLUME)
#define HYDRO_EXPLICITLY_INTEGRATE_VOLUME
#endif
#endif

#if (defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_HELMHOLTZ) || defined(EOS_ANEOS) || defined(COSMIC_RAY_FLUID) || defined(RT_RADPRESSURE_IN_HYDRO) || defined(EOS_TRUELOVE_PRESSURE) || defined(TRUELOVE_CRITERION_PRESSURE) || defined(EOS_GMC_BAROTROPIC) || defined(COSMIC_RAY_SUBGRID_LEBRON)) && !defined(EOS_GENERAL)
#define EOS_GENERAL
#endif

#if defined(CBE_INTEGRATOR) || defined(DM_FUZZY)
#define AGS_FACE_CALCULATION_IS_ACTIVE
#endif

#if defined(CBE_INTEGRATOR)
#define CBE_INTEGRATOR_NBASIS CBE_INTEGRATOR
#ifdef CBE_INTEGRATOR_SECONDMOMENT
#if (BOX_SPATIAL_DIMENSION==1) || defined(ONEDIM)
#define CBE_INTEGRATOR_NMOMENTS 3  /* [0-norm,1-mom,1-second] */
#elif (BOX_SPATIAL_DIMENSION==2) || defined(TWODIMS)
#define CBE_INTEGRATOR_NMOMENTS 6  /* [0-norm,2-mom,3-symm-tensor] */
#else
#define CBE_INTEGRATOR_NMOMENTS 10  /* 10 non-trivial moments [0th/norm, 3-mom, 6-symm-tensor] */
#endif
#else
#define CBE_INTEGRATOR_NMOMENTS ((BOX_SPATIAL_DIMENSION)+1)
#endif
#endif

#if defined(GRAIN_COLLISIONS)
#define DM_SIDM 8 /* use the SIDM module to handle scattering of otherwise-collisionless particles against each other -- set to Particle Type=3 here */
#endif
#if defined(PIC_MHD) && !defined(PIC_SPEEDOFLIGHT_REDUCTION)
#define PIC_SPEEDOFLIGHT_REDUCTION (1) /* default: no reduction of the speed of light for MHD-PIC */
#endif
#if defined(PIC_MHD)
#define PIC_MHD_NEW_RSOL_METHOD /* prefer new method for dealing with RSOL, should make simulations easier if done correctly */
#ifdef GRAIN_FLUID
#define GRAIN_FLUID_AND_PIC_BOTH_DEFINED /* keyword used later to know what we need to read in */
#else
#define GRAIN_FLUID
#endif
#ifndef GRAIN_LORENTZFORCE
#define GRAIN_LORENTZFORCE
#endif
#ifndef GRAIN_BACKREACTION
#define GRAIN_BACKREACTION
#endif
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(CBE_INTEGRATOR) || defined(DM_FUZZY) || defined(AGS_FACE_CALCULATION_IS_ACTIVE) || defined(DM_SIDM)
#define AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORALL)
#define ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING /* comment out to revert to behavior of taking 'greater' softening in pairwise kernel interactions with adaptive softenings enabled. really only needed currently for this particular AGS model given how it computes zeta terms (could be made optional with one more loop for those as well) */
#endif

#if defined(COOL_GRACKLE) && !defined(COOLING)
#define COOLING
#endif



#if defined(GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL) && defined(GALSF_SFR_CRITERION)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL sets the SF criteria itself (FIRE-3 would silently override GALSF_SFR_CRITERION): remove it from Config.sh"
#endif

/* set 'default' options for FIRE module packages as a whole */
#ifdef FIRE_PHYSICS_DEFAULTS

#define COOLING                             /*! top-level switch for cooling */
#define COOL_LOW_TEMPERATURES               /*! include low-temperature (<1e4 K) cooling */
#define COOL_METAL_LINES_BY_SPECIES         /*! include high-temperature metal-line cooling, species-by-species */
#define GALSF                               /*! top-level switch for star formation */
#define METALS                              /*! follow metals as passive scalars, use in cooling, etc */
#define TURB_DIFF_METALS                    /*! explicit sub-grid diffusivity for metals/passive scalars */
#define TURB_DIFF_METALS_LOWORDER           /*! memory-saving custom mod */
#define STOP_WHEN_BELOW_MINTIMESTEP         /*! this is general good practice */
#if !defined(MULTIPLEDOMAINS)
#define MULTIPLEDOMAINS 32                  /*! slightly closer to our usual default, but users should feel free to adjust */
#endif
#if !defined(DOMAIN_LIGHTWEIGHT_REPARTITION)
#define DOMAIN_LIGHTWEIGHT_REPARTITION      /*! enable lightweight domain repartition that reuses the top tree between full decompositions */
#endif
#define OPENMP_TREE_UPDATE

#define GALSF_FB_MECHANICAL                 /*! top-level switch for mechanical feedback modules */
#define GALSF_FB_FIRE_STELLAREVOLUTION (FIRE_PHYSICS_DEFAULTS) /*! turns on default FIRE processes+lookup tables including gas return, SNe, R-process, etc. this carries a number matching the defaults set you choose */
#if !(defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1) || defined(RT_LOCALRAYGRID))
/* Sub-grid FIRE RT modules: only enabled when no explicit RT solver is active. When a real RT
   method is on, photoionization / local radiation pressure / long-range starlight propagation
   are handled by the explicit solver -- enabling both produces duplicate symbols and
   double-counted physics. */
#define GALSF_FB_FIRE_RT_HIIHEATING         /*! gas within HII regions around young stars is photo-heated to 10^4 K - local stromgren approximation */
#define GALSF_FB_FIRE_RT_LOCALRP            /*! turn on local radiation pressure coupling to gas - account for local multiple-scattering and isotropic local absorption */
#define GALSF_FB_FIRE_RT_LONGRANGE          /*! continuous acceleration from starlight (uses luminosity tree) to propagate FIRE RT */
#endif
#define GALSF_FB_FIRE_AGE_TRACERS 16        /*! tracks a set of passive scalars corresponding to stellar ages for chemical evolution model postprocessing */

#if !(defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL))
#define ADAPTIVE_GRAVSOFT_FORGAS            /*! default choice is adaptive force softening for gas, but not stars [since ambiguously defined] */
#endif

#if (FIRE_PHYSICS_DEFAULTS <= 2)
#define GALSF_USE_SNE_ONELOOP_SCHEME // set to use the 'base' FIRE-2 SNe coupling. if commented out, will user newer version that more accurately manages the injected energy with neighbors moving to inject a specific target
#define GALSF_SFR_CRITERION (0+256) // 0=density threshold, 1=virial criterion (strict), 2=convergent flow, 4=local extremum, 8=no sink in kernel, 16=not falling into sink, 32=hill (tidal) criterion, 64=Jeans criterion, 128=converging flow along all principle axes, 256=self-shielding/molecular, 512=multi-free-fall (smooth dependence on virial), 1024='catch' for un-resolvable densities
#define GALSF_SFR_VIRIAL_SCALING (-1) // old threshold modification with exception when densities 100x above threshold density
#endif

#if (FIRE_PHYSICS_DEFAULTS == 3)
#if !defined(CHIMES) /* CHIMES evolves its own chemistry and does not carry the electron fraction this model reads */
#define COOL_MOLECFRAC_NONEQM
#endif
#define OUTPUT_MOLECULAR_FRACTION
#define OUTPUT_COOLRATE
#if !(defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1) || defined(RT_LOCALRAYGRID))
#define RT_USE_GRAVTREE_SAVE_RAD_FLUX  /* LEBRON-style gravtree flux save; mutex with explicit RT solvers (same reasoning as the GALSF_FB_FIRE_RT_* / SINK_*_HEATING blocks) */
#endif
#define OUTPUT_POTENTIAL
#define GALSF_SFR_CRITERION (0+1+2+64) // 0=density threshold, 1=virial criterion (strict), 2=convergent flow, 4=local extremum, 8=no sink in kernel, 16=not falling into sink, 32=hill (tidal) criterion, 64=Jeans criterion, 128=converging flow along all principle axes, 256=self-shielding/molecular, 512=multi-free-fall (smooth dependence on virial), 1024='catch' for un-resolvable densities
#define ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT (0.1/UNIT_LENGTH_IN_KPC)
#define GALSF_SFR_IMF_SAMPLING /* use the IMF-sampling discrete number of O-star scheme, no penalty at low mass-res */
#define FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT (1) /* ramp the SNe rate and massive stellar feedback fraction of total mass (essentially L/M) at low metallicities, leaves no dwarf stars below [Z/H]<-7 or so ramping down to -5 */
#if !defined(ADAPTIVE_TREEFORCE_UPDATE)
#define ADAPTIVE_TREEFORCE_UPDATE (0.0625) /* re-walk the tree for a particle's gravity only once per this fraction of its dynamical time, extrapolating with the jerk in between */
#endif
#endif // defaults = 3

#if defined(FIRE_MHD)
#define MAGNETIC            /* top-level flag */
#define MHD_B_SET_IN_PARAMS /* B-field must be set in ICs */
#define CONDUCTION          /* enable conduction */
#define CONDUCTION_SPITZER  /* compute proper coefficients and anisotropy for conduction */
#define VISCOSITY           /* enable viscosity */
#define VISCOSITY_BRAGINSKII /* compute proper coefficients and anisotropy for viscosity */
#endif // FIRE_MHD

#if defined(FIRE_CRS)
#if !defined(CRFLUID_SPEEDOFLIGHT_REDUCTION)
#define CRFLUID_SPEEDOFLIGHT_REDUCTION (1.0e8/C_LIGHT_CGS) /*! maximum CR transport speed: 1000 km/s safe for our default diffusivities in constant/variable-kappa model */
#endif
#if (FIRE_CRS == -2)
#define COSMIC_RAY_SUBGRID_LEBRON   /*! this simply uses the sub-grid model */
#else
#define COSMIC_RAY_FLUID            /*! use 'explicit' CR integration in one of the code formulations */
#endif // closes whether to do cr fluid or subgrid
#if defined(COSMIC_RAY_FLUID) /*! spectrum and diffusion defaults apply only to explicit CRs, not the sub-grid model */
#if !defined(CRFLUID_EVOLVE_SPECTRUM) /*! check to enable flags for which CR spectrum or single-bin to evolve */
#if (FIRE_CRS >= 0) && (FIRE_CRS <= 1)
#define CRFLUID_EVOLVE_SPECTRUM 1   /*! evolve proton + electron spectrum */
#endif
#if (FIRE_CRS >= 2)
#define CRFLUID_EVOLVE_SPECTRUM 2   /*! evolve full set of 10 different CR species */
#endif
#endif // closes which spectrum check
#if !defined(CRFLUID_DIFFUSION_MODEL)
#if (FIRE_CRS <= 0)
#define CRFLUID_DIFFUSION_MODEL 0   /*! constant diffusivity (set by params file) */
#else
#define CRFLUID_DIFFUSION_MODEL 8   /*! best-guess for an empirical model which varies as a function of CR gradient scale length from ISM-to-CGM, but not with real plasma properties. lots of options for that. */
//#define CRFLUID_DIFFUSION_MODEL 6   /*! best-guess for variable-kappa model, combining updated SC+ET */
//#define CRFLUID_SET_SC_MODEL (7)    /*! set mode for SC model using best-estimate of fQLT and fCAS, and best model for extrinsic driving of CRs */
//#define CRFLUID_SET_ET_MODEL (-1)   /*! set mode for ET model using best-estimate of fturb from Alfven-wave scattering */
#endif
#endif // closes which diffusion model[s] to enable
#endif // COSMIC_RAY_FLUID
#endif // FIRE_CRS

#if defined(FIRE_BHS)
#define SINK_PARTICLES                 /* top-level flag */
#define SINK_SEED_FROM_LOCALGAS       /* seed BHs locally in SF-ing gas */
#define SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA /* use the total surface-density criterion, not just gas */
#define SINK_CALC_DISTANCES           /* use this for various checks, particularly in seeding */
#if (defined(GALSF_SFR_IMF_SAMPLING) || (FIRE_PHYSICS_DEFAULTS > 2)) && !defined(SINK_REPOSITION_ON_POTMIN) && !defined(SINK_DYNFRICTION_FROMTREE)
#define SINK_DYNFRICTION_FROMTREE /* use the dynamical friction model instead of pinning/forcing BHs to potential minimum */
#endif
#if !defined(SINK_REPOSITION_ON_POTMIN) && !defined(SINK_DYNFRICTION_FROMTREE)
#define SINK_REPOSITION_ON_POTMIN   /* anchor BHs to centers smoothly */
#endif
#if (FIRE_PHYSICS_DEFAULTS > 2)
#define SINK_SCALE_SPAWNINGMASS_WITH_INITIALMASS /* purely a convention-choice when doing spawning, to use fraction of original BH mass -- this one more useful if using multi-resolution (hyper-refinement) techniques */
#endif
#define SINK_SWALLOWGAS               /* allow BHs to accrete in principle */
#if !defined(SINK_GRAVACCRETION)
#define SINK_GRAVACCRETION 1          /* accrete following our standard gravitational torques model */
#define SINK_GRAVACCRETION_STELLARFBCORR          /* account for additional acceleration-dependent retention from stellar FB in Mdot */
#endif
#if !defined(SINK_ALPHADISK_ACCRETION)
#define SINK_ALPHADISK_ACCRETION (1.e10) /* smooth out accretion + allow super-eddington capture with alpha-disk model */
#endif
#if !(defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1) || defined(RT_LOCALRAYGRID))
/* AGN-source counterparts of the FIRE sub-grid stellar RT modules. They share the
   LEBRON/gravity-tree numerical infrastructure (RT_USE_GRAVTREE,
   valid_gas_particle_for_rt accumulators, etc.) and are therefore mutually exclusive
   with explicit RT solvers, exactly as their stellar GALSF_FB_FIRE_RT_* counterparts. */
#define SINK_PHOTONMOMENTUM           /* allow AGN radiation pressure */
#define SINK_COMPTON_HEATING          /* allow Compton heating from AGN spectrum */
#define SINK_HII_HEATING              /* allow photo-ionization heating from AGN spectrum */
#endif
#define SINK_FB_COLLIMATED            /* BHFB directed along collimated axis following BH ang. mom */
#define SINK_WIND_SPAWN (2)           /* spawn module: N=min num spawned/step */
#if !defined(DISABLE_MAINTAIN_TREE_IN_REARRANGE)
#define MAINTAIN_TREE_IN_REARRANGE    /* avoid constant domain decompositions in bottom timebin each time a spawn occurs */
#endif
#if defined(COSMIC_RAY_FLUID) || defined(COSMIC_RAY_SUBGRID_LEBRON)
#define SINK_COSMIC_RAYS              /* allow CR injection from AGN */
#endif
#endif // FIRE_BHS

#if defined(PMGRID)
#if !defined(PM_PLACEHIGHRESREGION)
#if (FIRE_PHYSICS_DEFAULTS == 3)
#define PM_PLACEHIGHRESREGION 2 /* 2 -- if we want to have potentially resolved hyper-velocity stars/bhs/cells, need to avoid them restructuring the PMGRID completely */
#else
#if defined(SINK_PARTICLES)
#define PM_PLACEHIGHRESREGION 51 /* 1+2+16+32 */
#else
#define PM_PLACEHIGHRESREGION 19 /* 1+2+16 */
#endif
#endif
#endif
#endif // PMGRID check

#endif // FIRE_PHYSICS_DEFAULTS clauses


/* SINK_COMPTON_HEATING reads the same angle-weighted sink luminosity tree data as
 * SINK_PHOTONMOMENTUM, so it needs that infrastructure compiled in; the momentum kick itself can
 * still be disabled at runtime (PhotonMomentum_Coupled_Fraction). Kept outside the
 * FIRE_PHYSICS_DEFAULTS block so it applies to any Config that sets SINK_COMPTON_HEATING. */
#if defined(SINK_COMPTON_HEATING) && !defined(SINK_PHOTONMOMENTUM)
#define SINK_PHOTONMOMENTUM
#endif


#ifdef GALSF_SFR_CRITERION // flag for pure cross-compatibility [identical functionality, just ease-of-use for galaxy simulators here]
#define SINGLE_STAR_SINK_FORMATION GALSF_SFR_CRITERION
#endif


/* set default modules for different cosmic ray transport settings/packages */
#ifdef COSMIC_RAY_FLUID
#if defined(GALSF_RESOLVEDISM)
#ifndef CRFLUID_EXCHANGE_PDV_ONLY
#define CRFLUID_EXCHANGE_PDV_ONLY /* Stella default: CRs take only their own P dV work, no share of shock heating */
#endif
#ifndef CRFLUID_SPEEDOFLIGHT_REDUCTION
#define CRFLUID_SPEEDOFLIGHT_REDUCTION (3.e7/C_LIGHT_CGS) /* Stella default: 300 km/s CR transport speed */
#endif
#endif
#if !defined(CRFLUID_SPEEDOFLIGHT_REDUCTION)
#define CRFLUID_SPEEDOFLIGHT_REDUCTION (1.0)
#endif
#if !defined(CRFLUID_EVOLVE_SPECTRUM)
#define GAMMA_COSMICRAY(k) (4.0/3.0)
#endif
#ifndef CRFLUID_DIFFUSION_MODEL
#define CRFLUID_DIFFUSION_MODEL 0
#endif
#if defined(CRFLUID_EVOLVE_SPECTRUM)
#define GAMMA_COSMICRAY(k) ((4.0+gamma_eos_of_crs_in_bin(k))/3.0)
#if (CRFLUID_EVOLVE_SPECTRUM == 2)
#define N_CR_PARTICLE_BINS 70       /*<! set default bin number here -- needs to match hard-coded list in function 'CR_spectrum_define_bins', for now> */
#define N_CR_PARTICLE_SPECIES 8     /*<! total number of CR species to be evolved. must be set here because of references below*/
#else
#define N_CR_PARTICLE_BINS 19       /*<! set default bin number here -- needs to match hard-coded list in function 'CR_spectrum_define_bins', for now> */
#define N_CR_PARTICLE_SPECIES 2     /*<! total number of CR species to be evolved. must be set here because of references below*/
#endif
#endif
#ifndef N_CR_PARTICLE_BINS
#define N_CR_PARTICLE_BINS 1
#endif
#if (N_CR_PARTICLE_BINS > 2) && !defined(CRFLUID_EVOLVE_SPECTRUM)
#define CRFLUID_EVOLVE_SPECTRUM
#endif
#endif


#ifdef COSMIC_RAY_SUBGRID_LEBRON
#define N_CR_PARTICLE_BINS 1
#endif





/* set default options for hybrid single star and ssp model packages */
#if defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL_DEFAULTS) /* options for hybrid/combined FIRE+STARFORGE simulations */
#define SINGLE_STAR_STARFORGE_DEFAULTS   /* parent flag enabling the STARFORGE modules, which themselves enable the FIRE modules */
#define SINGLE_STAR_AND_SSP_HYBRID_MODEL (SINGLE_STAR_AND_SSP_HYBRID_MODEL_DEFAULTS) /* do single-star routines below this mass resolution in solar, FIRE-like above */
#define GALSF_SFR_IMF_SAMPLING           /* use discrete sampling of 'number of O-stars' so we can handle the intermediate-mass regime in at least a simple approximate manner */
#define COOLING              /* only physical if include cooling for both sides, using same cooling functions */
#define MAGNETIC             /* enable MHD, important for systems here */
#define CONDUCTION           /* enable conduction */
#define CONDUCTION_SPITZER   /* compute proper coefficients and anisotropy for conduction */
#define VISCOSITY            /* enable viscosity */
#define VISCOSITY_BRAGINSKII /* compute proper coefficients and anisotropy for viscosity */
#define SINGLE_STAR_FB_JETS  /* enable jets from protostars */
#define SINGLE_STAR_FB_WINDS 0 /* enable continuous mass-loss feedback - will also enable ssp mass-loss */
#define SINGLE_STAR_FB_SNE   /* enable SNe feedback - will also enable ssp mechanical feedback */
#define SINGLE_STAR_FB_RAD   /* enable RHD feedback */
#define RT_COMOVING          /* significantly more stable and accurate formulation given the structure of the problem and method we use */
#define RT_SOURCES (16+32)   /* need to allow -both- ssp-particles and single-star particles to emit */
#if !defined(RT_SPEEDOFLIGHT_REDUCTION)
#define RT_SPEEDOFLIGHT_REDUCTION (0.1)   /* for many problems on these scales, need much larger RSOL than default starforge values (dynamical velocities are big, without this they will severely lag behind) */
#endif
#define ADAPTIVE_TREEFORCE_UPDATE (0.0625) /* rough typical value we use for ensuring stability */
#if defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
#define OUTPUT_ACCELERATION
#define OUTPUT_HYDROACCELERATION
#define OUTPUT_MOLECULAR_FRACTION
#define OUTPUT_TEMPERATURE
#define OUTPUT_ADDITIONAL_RUNINFO
#endif
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
#define PARTICLE_EXCISION
#define OUTPUT_GRADIENT_RHO
#define OUTPUT_GRADIENT_VEL
#define OUTPUT_RT_RAD_FLUX
#define OUTPUT_RT_RAD_OPACITY
#define RT_RAD_PRESSURE_OUTPUT
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES
#define GRAVITY_ANALYTIC
#endif
#endif
#endif // closes hybrid FIRE+STARFORGE model settings




/* set default options for STARFORGE module packages */
#ifdef SINGLE_STAR_STARFORGE_DEFAULTS /* bunch of options -NOT- strictly required here, but this is a temporary convenience block */
#if !defined(MULTIPLEDOMAINS)
#define MULTIPLEDOMAINS 64 /* upper bound on Peano segments per rank; the count follows the particle load (constants.h) */
#if !defined(DOMAIN_TARGET_PARTICLES_PER_SEGMENT)
#define DOMAIN_TARGET_PARTICLES_PER_SEGMENT 8000 /* M2e4 stress case (23.8M cells) at 48 ranks: 8 segments per rank (62k particles
                              each) left ~7x max/mean hydro compute on medium/big steps; 32 (15.5k) measured 0.83x wall, 64 (7.7k) a
                              further 0.96x, 128 (3.9k) flat. A fixed 64 cost +30% wall on shu1977 (16k particles per rank, 8 ranks:
                              top-tree build, domain and walk overheads), which this rule gives the floor of 8. Rule: 48 ranks -> 62,
                              96 ranks -> 31 on M2e4. */
#endif
#endif
#if !defined(DOMAIN_KEEP_OWNERS) && !defined(DOMAIN_KEEP_OWNERS_OFF)
#define DOMAIN_KEEP_OWNERS /* owner-preserving segment assignment (domain.cc); M2e5 at 384 ranks: 99.7% -> 32-57% of particles
                              exchanged per decomposition, domain time 0.41x; M2e4 at 48 ranks on one node: 0.98x wall. Opt out with
                              DOMAIN_KEEP_OWNERS_OFF. */
#endif
#if !defined(STARFORGE_OPTIMIZATIONS) && !defined(STARFORGE_OPTIMIZATIONS_OFF)
#define STARFORGE_OPTIMIZATIONS /* bundle of opt-in performance options, expanded below; opt out with STARFORGE_OPTIMIZATIONS_OFF */
#endif
#define INPUT_POSITIONS_IN_DOUBLE
#define OUTPUT_POTENTIAL
#if !defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL) && !defined(IO_NO_GRADUAL_SNAPSHOT_RESTART) /* opt-out: restart from a snapshot at physical timesteps, no ramp */
#define IO_GRADUAL_SNAPSHOT_RESTART
#endif
#ifndef IO_SINKS_ONLY_SNAPSHOT_FREQUENCY
#define IO_SINKS_ONLY_SNAPSHOT_FREQUENCY 0 /* Determines the number of snapshots with reduced data (stars only) per full snapshots (gas+stars), e.g., setting it to 2 means 2/3 of the snapshots will be reduced, 1/3 will have full data. Setting it to 0 disables it.  */
#endif
#define SINGLE_STAR_SINK_DYNAMICS
#if !defined(PMGRID) && !defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT) && !defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) && !defined(DISABLE_HERMITE_INTEGRATION)
#define HERMITE_INTEGRATION 32 // bitflag for which particles to do 4th-order Hermite integration
#endif
#define ADAPTIVE_GRAVSOFT_FORGAS
#define GRAVITY_ACCURATE_FEWBODY_INTEGRATION
#ifndef SINGLE_STAR_TIMESTEPPING
#define SINGLE_STAR_TIMESTEPPING 0
#endif
#define SINGLE_STAR_ACCRETION 12
#define SINGLE_STAR_SINK_FORMATION (0+1+2+4+8+16+32+64+2048) // 0=density threshold, 1=virial criterion, 2=convergent flow, 4=local extremum, 8=no sink in kernel, 16=not falling into sink, 32=hill (tidal) criterion, 64=Jeans criterion, 128=converging flow along all principle axes, 256=self-shielding/molecular, 512=multi-free-fall (smooth dependence on virial), 1024=numerical escape if too dense, 2048=virial is time-averaged
#ifndef SINGLE_STAR_DIRECT_GRAVITY_RADIUS
#define SINGLE_STAR_DIRECT_GRAVITY_RADIUS (1000.) // distance inside of which star-star gravitational interactions are calculated exactly, in AU
#endif
#ifndef ADAPTIVE_TREEFORCE_UPDATE
#define ADAPTIVE_TREEFORCE_UPDATE (0.0625) // optimization
#endif
#ifndef SINK_MDOT_RESERVOIR_CAP
#define SINK_MDOT_RESERVOIR_CAP 30 /* throttle the reservoir drain of marginally-resolved sinks; see set_sink_mdot in sinks/sink.cc */
#endif
#if !defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM) && !defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT)
#ifndef IO_SUPPRESS_TIMEBIN_STDOUT
#define IO_SUPPRESS_TIMEBIN_STDOUT 16 // only prints outputs to log file if the highest active timebin index is within n of the highest timebin (dt_bin=2^(-N)*dt_bin,max)
#endif
#endif
#define OUTPUT_SINK_ACCRETION_HIST // save accretion histories
#define OUTPUT_SINK_FORMATION_PROPS // save at-formation properties of sink particles
#define INPUT_READ_SINKPROPS
#if ( defined(STARFORGE_GMC_TURBINIT) || defined(STARFORGE_FILAMENT_TURBINIT) ) // these flags should be given numerical values equal to the desired virial parameter
#define TURB_DRIVING
#define GRAVITY_ANALYTIC
#define SELFGRAVITY_OFF
#endif
#define SINK_ALPHADISK_ACCRETION (1.0e6)
#ifdef GRAIN_FLUID
#define SINK_GRAVCAPTURE_NONGAS
#endif
#ifdef MAGNETIC
#define MHD_CONSTRAINED_GRADIENT 1
#endif
#if ( defined(SINGLE_STAR_FB_JETS) || defined(SINGLE_STAR_FB_WINDS) || defined(SINGLE_STAR_FB_RT_HEATING) || defined(SINGLE_STAR_FB_SNE) || defined(SINGLE_STAR_FB_RAD) || defined(SINGLE_STAR_FB_LOCAL_RP))
#define SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION 2 //we are using the protostellar evolution model from ORION
#endif
#if ( defined(SINGLE_STAR_FB_JETS) || defined(SINGLE_STAR_FB_WINDS) || defined(SINGLE_STAR_FB_SNE) ) //enable diffusion for metals and enable tracers for different feedback channels
#define STARFORGE_FEEDBACK_TRACERS 3 // 0 for jets, 1 for winds, 2 for SNe
#define TURB_DIFF_METALS
#define TURB_DIFF_METALS_LOWORDER
#endif
#ifdef SINGLE_STAR_FB_RAD
#define SINGLE_STAR_RT_DEFAULTS
#define RT_OPTICAL_NIR
#define RT_NUV
#define RT_PHOTOELECTRIC
#ifndef RT_CHEM_PHOTOION
#define RT_CHEM_PHOTOION 1
#endif
#define RT_INFRARED
#if !defined(RT_ISRF_BACKGROUND) && !defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL)
#define RT_ISRF_BACKGROUND
#endif
#endif
#if (defined(COOLING) && !defined(COOL_LOWTEMP_THIN_ONLY) && !defined(RT_INFRARED))
#define RT_USE_TREECOL_FOR_NH 6 /* This gives a better approximation for column density than the usual scale-length estimator, but is overkill for typical 1e-3msun-resolving simulations that only marginally resolve the opacity limit. Enable for high (<1e-5msun) resolution sims */
#endif
#ifdef COOLING
#define SIMPLE_STEADYSTATE_CHEMISTRY
#if !defined(CHIMES) /* not compatible with CHIMES, which evolves its own chemistry */
#define COOL_MOLECFRAC_NONEQM
#endif
#define EOS_SUBSTELLAR_ISM
#define OUTPUT_MOLECULAR_FRACTION
#if defined(MAGNETIC) && !defined(CONDUCTION) && !defined(VISCOSITY) // if we have cooling and magnetic fields, enable conduction + viscosity
#define CONDUCTION           /* enable conduction */
#define CONDUCTION_SPITZER   /* compute proper coefficients and anisotropy for conduction */
#define VISCOSITY            /* enable viscosity */
#define VISCOSITY_BRAGINSKII /* compute proper coefficients and anisotropy for viscosity */
#define DIFFUSION_OPTIMIZERS
#endif // MAGNETIC
#if !defined(RT_ISRF_BACKGROUND) && !defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL)
#define RT_ISRF_BACKGROUND  // Draine 1978 ISRF for photoelectric heating (appropriate for solar circle, must be re-scaled for different environments)
#endif
#endif // COOLING
#if defined(SINGLE_STAR_FB_WINDS) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) || defined(COOLING)
#define GALSF_FB_FIRE_STELLAREVOLUTION 3 // enable multi-loop feedback from such sources [this is specific to the DG-MG implementations here, not for public use right now!]. for now set to =2, which should force the code version to match previous iterations, as compared to the newer implementations.
#endif
#endif // closes SINGLE_STAR_STARFORGE_DEFAULTS settings

#ifdef STARFORGE_OPTIMIZATIONS
/* Performance bundle (default under SINGLE_STAR_STARFORGE_DEFAULTS; STARFORGE_OPTIMIZATIONS_OFF removes it). Each component
   has its own opt-out, and setting a component explicitly (e.g. DOMAIN_FREQ_WEIGHT=4) overrides the bundle's value.
   None of them is bitwise: results change at the level of a rank-count change.
   - TREE_REFRESH_INCREMENTAL (opt out: TREE_REFRESH_INCREMENTAL_OFF): sink/feedback moment refreshes re-derive only the
     chains above changed particles. Changes results slightly (untouched nodes keep their drifted moments).
   - DOMAIN_MULTIBIN (opt out: DOMAIN_MULTIBIN_OFF): the decomposition balances every active level up to its own, not one
     composite cost. Changes the domain layout.
   - DOMAIN_FREQ_WEIGHT=9 (opt out: DOMAIN_FREQ_WEIGHT_OFF): particle work x min(2^(top bin - bin), 2^9). Changes the layout.
   - DOMAIN_FREQ_WEIGHT_SPLIT_ONLY (opt out: DOMAIN_FREQ_WEIGHT_SPLIT_ONLY_OFF): that weight enters the split and the
     assignment only; the top tree is refined on the unweighted cost. Without it the weighted refinement doubles the top
     tree on Bate-type runs (deep-bin disks around sinks) and the bundle regresses there (1.02-1.04x). Changes the layout.
   - HYDRO_PAIR_ONCE (opt out: HYDRO_PAIR_ONCE_OFF), pure-MPI MFM builds only: see its definition below (after the hydro
     and threading flags are final).
   Measured: wall over one full top-level timestep cycle, time-matched, vs the same code with STARFORGE_OPTIMIZATIONS_OFF
   on the same nodes; pairs repeated with the order swapped.
     one Genoa node, 48x1 per socket:  STARFORGE M2e4 stress case 0.810/0.805; M2e4 production without RT 0.858/0.862;
                                       Bate et al. 2003 M50 after sink formation 0.774/0.788 (8 sinks), 0.768/0.757 (20 sinks)
     384x1 on 4 Genoa nodes:           M2e4 0.686/0.688 (baseline repeats 1.000/1.000); M2e5 0.776/0.779
   Per component on M2e4 / Bate (10-15 min screens): incremental refresh 0.90 / 0.80-0.87; multibin + split-only weight on
   top 0.90 / 0.88-0.90; multibin with the weight also refining the top tree 0.92 / 1.02-1.04 (rejected). At 384 ranks
   the bundle with DOMAIN_FREQ_WEIGHT_SPLIT_ONLY_OFF gives M2e4 0.739, i.e. split-only gains there too. */
#if !defined(TREE_REFRESH_INCREMENTAL) && !defined(TREE_REFRESH_INCREMENTAL_OFF)
#define TREE_REFRESH_INCREMENTAL
#endif
#if !defined(DOMAIN_MULTIBIN) && !defined(DOMAIN_MULTIBIN_OFF)
#define DOMAIN_MULTIBIN
#endif
#if !defined(DOMAIN_FREQ_WEIGHT) && !defined(DOMAIN_FREQ_WEIGHT_OFF)
#define DOMAIN_FREQ_WEIGHT 9
#endif
#if defined(DOMAIN_FREQ_WEIGHT) && !defined(DOMAIN_FREQ_WEIGHT_SPLIT_ONLY) && !defined(DOMAIN_FREQ_WEIGHT_SPLIT_ONLY_OFF)
#define DOMAIN_FREQ_WEIGHT_SPLIT_ONLY
#endif
#endif // STARFORGE_OPTIMIZATIONS

/* RT_ISRF_BACKGROUND=0 (or any value <= 0) in Config.sh switches the interstellar background off, including where an
   umbrella above would turn it on (those only define it if it is not defined yet). Defined without a value it stays on.
   The value test is written so that an empty definition is not read as 0: CHECK_IF_PREPROCESSOR_HAS_NUMERICAL_VALUE_
   cannot tell 0 from empty. */
#if defined(RT_ISRF_BACKGROUND)
#if !((1 - RT_ISRF_BACKGROUND - 1 == 2) && (RT_ISRF_BACKGROUND + 0 == 0)) /* given a value */
#if (RT_ISRF_BACKGROUND + 0 <= 0)
#undef RT_ISRF_BACKGROUND
#endif
#endif
#endif

#ifdef SINGLE_STAR_RT_DEFAULTS
#define RT_M1
#define RT_COMOVING
#define OUTPUT_RT_RAD_FLUX
#ifndef IO_SUPPRESS_OUTPUT_EDDINGTON_TENSOR
#define IO_SUPPRESS_OUTPUT_EDDINGTON_TENSOR
#endif
#ifndef RT_SOURCES
#define RT_SOURCES 32
#endif
#ifndef RT_SPEEDOFLIGHT_REDUCTION
#define RT_SPEEDOFLIGHT_REDUCTION (3.0e-4)
#endif
#define RT_REPROCESS_INJECTED_PHOTONS
#define RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION
#define RT_REINJECT_ACCRETED_PHOTONS
#define RT_INJECT_PHOTONS_DISCRETELY
#endif // closes SINGLE_STAR_RT_DEFAULTS settings


#ifdef SINGLE_STAR_SINK_DYNAMICS
#define GALSF // top-level switch needed to enable various frameworks
#define METALS  // metals should be active for stellar return
#define SINK_PARTICLES // need to have sink particles active since these are our sink particles
#define SINK_INTERACT_ON_GAS_TIMESTEP // BH-gas interactions (feedback and accretion) occur with frequency set by the gas timestep
#define SINK_CALC_DISTANCES // calculate distance to nearest sink in gravity tree

#ifdef SINGLE_STAR_ACCRETION // figure out flags needed for the chosen sink accretion model
#define SINK_SWALLOWGAS // need to swallow gas [part of sink model]
#ifndef SINK_ALPHADISK_ACCRETION
#define SINK_ALPHADISK_ACCRETION (2.) // all models will use a 'reservoir' of some kind to smooth out accretion rates (and represent unresolved disk)
#endif
#if (SINGLE_STAR_ACCRETION <= 10)
#define SINK_GRAVACCRETION (SINGLE_STAR_ACCRETION) // use one of these pre-built accretion models
#endif
#if (SINGLE_STAR_ACCRETION == 11)
#define SINK_GRAVCAPTURE_GAS // use gravitational capture swallow criterion for resolved gravitational capture
#endif
#if (SINGLE_STAR_ACCRETION == 12)
#define SINK_GRAVCAPTURE_GAS
#define SINK_GRAVCAPTURE_FIXEDSINKRADIUS // modify grav capture to Bate-style, fixed (in time) sink radius based on SF neighbor distance, plus angular momentum criterion
#endif
#endif

#if (defined(SINGLE_STAR_FB_JETS) || defined(SINGLE_STAR_FB_WINDS) || defined(SINGLE_STAR_FB_RT_HEATING) || defined(SINGLE_STAR_FB_SNE) || defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1) || defined(RT_LOCALRAYGRID) || defined(SINGLE_STAR_FB_LOCAL_RP)) && defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_SINK_DYNAMICS)
#define SINGLE_STAR_FB_TIMESTEPLIMIT // general flag indicating feedback is on
#endif

#if defined(SINGLE_STAR_FB_RT_HEATING) && !(defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1) || defined(RT_LOCALRAYGRID))
#define GALSF_FB_FIRE_RT_LONGRANGE  // turn on FIRE RT approximation: no Type-4 particles so don't worry about its approximations
#define SINK_PHOTONMOMENTUM // enable BHs within the FIRE-RT framework.
#define RT_DISABLE_RAD_PRESSURE
#endif

#if (defined(SINGLE_STAR_FB_WINDS) || defined(SINGLE_STAR_FB_SNE)) && !defined(GALSF_FB_MECHANICAL)
#define GALSF_FB_MECHANICAL // we will use the mechanical wind module for low mass loss rate stars (spawning leads to issues). enable regardless if either the winds or sne module is active
#define GALSF_USE_SNE_ONELOOP_SCHEME
#endif

#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && !defined(GALSF_FB_FIRE_STELLAREVOLUTION)
#define GALSF_FB_FIRE_STELLAREVOLUTION 3 /* this particular set of modules depends on the fire modules explicitly now, using them for yields and other infrastructure */
#endif

#if defined(SINGLE_STAR_FB_JETS) || ((defined(SINGLE_STAR_FB_WINDS) || defined(SINGLE_STAR_FB_SNE)) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION))
#define SINK_WIND_SPAWN (2) // leverage the BHFB model already developed within the FIRE-BHs framework. gives accurate launching of arbitrarily-structured jets.
#if !defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM) && !defined(DISABLE_MAINTAIN_TREE_IN_REARRANGE)
#define MAINTAIN_TREE_IN_REARRANGE // don't rebuild the domains/tree every time a particle is spawned - salvage the existing one by redirecting pointers as needed
#endif
#endif

/* TREE_INTEGRITY_AUDITS must be set explicitly in Config.sh; nothing implies it. It enables fatal
   deep tree audits (exactly-once walk, Father/threading cross-check, moment re-derivation) at
   build/refresh/mutation points. It is deliberately not tied to DEVELOPER_MODE, which many test
   configs set only to expose accuracy parameters. Without it only the always-on checks run
   (force_validate_tree_links and the treebuild bookkeeping reduce). */

#if defined(SINK_PARTICLES) && !defined(MAINTAIN_TREE_IN_REARRANGE) && (defined(HERMITE_INTEGRATION) || !defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM))
/* Every sink pass that swallows ends with a cleanup rearrange (run.cc) that eliminates the victims,
   spawning or not. Without maintenance that rearrange only condemns the tree, and the consumers
   that follow in the same step walk it without rebuilding: the Hermite passes (which skip the
   rebuild by design, the HermiteOnlyFlag gate in gravtree.cc), RT source injection from the sink,
   and the feedback neighbour loops. The threading still points at the vacated slots, so each of
   those walks sees a duplicated particle per swallow, and Hermite also refreshes moments on, and
   takes sink forces from, that tree. The parent-only carry is not enough for a tree that keeps
   being walked, so every sink run gets the full per-swap maintenance. Nuclear zoom keeps its
   upstream exclusion above unless Hermite is on; the condemned walks it then makes are reported
   by force_tree_check_walkable. (cf. gizmo-cpp a06e0073) */
#ifdef DISABLE_MAINTAIN_TREE_IN_REARRANGE
#warning "DISABLE_MAINTAIN_TREE_IN_REARRANGE overridden: SINK_PARTICLES requires tree maintenance (this config cannot serve as the no-MAINTAIN A/B arm)"
#endif
#define MAINTAIN_TREE_IN_REARRANGE
#endif

#if defined(SINGLE_STAR_FB_LOCAL_RP) // use standard angle-weighted local coupling to impart photon momentum from stars
#if !defined(SINK_PHOTONMOMENTUM)
#define SINK_PHOTONMOMENTUM
#endif
#if !defined(RT_DISABLE_RAD_PRESSURE)
#define RT_DISABLE_RAD_PRESSURE // we only want the local short-ranged photon momentum, since SF sims can easily get into the badly non-photon-conserving limit where LEBRON fluxes are less accurate
#endif
#endif

#if defined(COOLING) && !defined(COOL_GRACKLE) // if not using grackle modules, need to make sure appropriate cooling is enabled
#ifndef COOL_LOW_TEMPERATURES
#define COOL_LOW_TEMPERATURES // make sure low-temperature cooling is enabled!
#endif
#ifndef COOL_METAL_LINES_BY_SPECIES
#define COOL_METAL_LINES_BY_SPECIES // metal-based cooling enabled
#endif
#define OUTPUT_TEMPERATURE
#endif

#endif // SINGLE_STAR_SINK_DYNAMICS


#if (SINGLE_STAR_SINK_FORMATION & 16)
#ifndef SINGLE_STAR_TIMESTEPPING
#define SINGLE_STAR_TIMESTEPPING 0
#endif
#endif
#if (SINGLE_STAR_SINK_FORMATION & 32)
#define GALSF_SFR_TIDAL_HILL_CRITERION
#endif
#if (SINGLE_STAR_SINK_FORMATION & 512)
#define GALSF_SFR_VIRIAL_SCALING 2
#endif
#if (SINGLE_STAR_SINK_FORMATION & 1)
#ifndef GALSF_SFR_VIRIAL_SCALING
#define GALSF_SFR_VIRIAL_SCALING 0
#endif
#endif
#if (SINGLE_STAR_SINK_FORMATION & 2048)
#define GALSF_SFR_VIRIAL_CRITERION_TIMEAVERAGED
#endif

#ifdef GRAVITY_ACCURATE_FEWBODY_INTEGRATION /* utility flag to enable a few different extra-conservative time-integration flags for gravity */
#if !defined(GRAVITY_HYBRID_OPENING_CRIT)
#define GRAVITY_HYBRID_OPENING_CRIT // use both Barnes-Hut + relative tree opening criterion
#endif
#if !defined(STOP_WHEN_BELOW_MINTIMESTEP)
#define STOP_WHEN_BELOW_MINTIMESTEP // stop when below min timestep to prevent bad timestepping
#endif
#define TIDAL_TIMESTEP_CRITERION // use tidal tensor timestep criterion
#endif
#ifdef STARS_HERMITE_DIRECT_GRAVITY /* Type-4 stars: 4th-order Hermite, exact direct star-star gravity (and jerk), Aarseth steps */
#if defined(SINK_PARTICLES) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_DIRECT_GRAVITY) || defined(HERMITE_INTEGRATION) || defined(SINK_CALC_DISTANCES) || defined(PMGRID)
#error "STARS_HERMITE_DIRECT_GRAVITY takes over the tree's sink node moments for Type 4: not combinable with sinks, SINGLE_STAR_DIRECT_GRAVITY, another HERMITE_INTEGRATION mask, other SINK_CALC_DISTANCES users or PMGRID"
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL) && ((ADAPTIVE_GRAVSOFT_FORALL) & 16)
#error "STARS_HERMITE_DIRECT_GRAVITY needs fixed Type-4 softening: adaptive softening changes between Hermite force evaluations and collapses the steps"
#endif
#define HERMITE_INTEGRATION 16
#define SINGLE_STAR_DIRECT_GRAVITY
#define STAR_DIRECT_GRAVITY_TYPE 4
#define SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES (4)
#ifndef STARS_HERMITE_AARSETH_ETA
#define STARS_HERMITE_AARSETH_ETA 0.02 /* Aarseth (1985) step parameter */
#endif
#endif
#if defined(SINGLE_STAR_DIRECT_GRAVITY) && !defined(STAR_DIRECT_GRAVITY_TYPE)
#define STAR_DIRECT_GRAVITY_TYPE 5
#endif
#ifdef HERMITE_INTEGRATION
#define COMPUTE_JERK_IN_GRAVTREE /* needs to be computed in order to do the Hermite integration */
#ifndef HERMITE_CORRECTOR_ITERATIONS
#define HERMITE_CORRECTOR_ITERATIONS 1 /* P(EC)^n: n>1 re-evaluates gravity at the corrected state and re-applies the corrector (Kokubo, Yoshinaga & Makino 1998) */
#endif
#ifndef TIDAL_TIMESTEP_CRITERION
#define TIDAL_TIMESTEP_CRITERION // use tidal tensor timestep criterion -- otherwise won't effectively leverage the Hermite integrator timesteps
#endif
#if defined(HERMITE_SYMMETRIC_TIMESTEP) && !defined(FORCE_EQUAL_TIMESTEPS)
/* inert without HERMITE_INTEGRATION or on equal steps. The code tests this name rather than relying on
   an #undef, which the re-inclusion of GIZMO_config.h (via eos.h) after this file would undo */
#define HERMITE_SYMMETRIC_TIMESTEP_ACTIVE
#endif
#endif




#if (defined(SINGLE_STAR_FB_SNE) || defined(SINGLE_STAR_FB_WINDS)) && !defined(SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)
#define SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT 2 // determines the maximum number of ejecta particles spawned per timestep, see below
#endif
#if defined(SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)
#define SINGLE_STAR_FB_SNE_N_EJECTA (4*(SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)*((SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)+1)) // maximum number of ejecta cells spawned per timestep - follows from tiling rules for rays from RHD-direct-ray method
#endif


#ifdef ADAPTIVE_TREEFORCE_UPDATE // instead of going into the tree every timestep, only update gravity with a frequency set by this fraction of dynamical timescale (every type; exemptions in needs_new_treeforce)
#ifndef TIDAL_TIMESTEP_CRITERION
#define TIDAL_TIMESTEP_CRITERION // need this to estimate the dynamical time
#endif
#endif
/* a particle whose tree walk is skipped advances its last gravity with the jerk. A force added to GravAccel after the gravity
   call must then not ride along on that value: either its types always take a walk (needs_new_treeforce: grains, fuzzy DM),
   or, for a force every type feels, the tree part is held apart in GravAccel_Retained, and the analytic terms and the forces
   added after gravity are evaluated afresh each step. The shearing-box Coriolis force is velocity-dependent and acts on every
   type, and its tidal term changes on the epicycle timescale, which the walk cadence (set by self-gravity) does not see. */
#if defined(ADAPTIVE_TREEFORCE_UPDATE) && !defined(SELFGRAVITY_OFF) && !defined(RT_SELFGRAVITY_OFF) && defined(BOX_SHEARING)
#define GRAVACCEL_RETAINED_ON_SKIPPED_WALK
#endif


#if defined(SINGLE_STAR_TIMESTEPPING) && (SINGLE_STAR_TIMESTEPPING > 0)
#error "SINGLE_STAR_TIMESTEPPING > 0 (super-timestepping of sink binaries) has been removed. Use SINGLE_STAR_TIMESTEPPING=0; for regularized few-body integration of tight sink systems, use KETJU_REGULARIZATION (branch starforge_dev+ketju)."
#endif
#ifdef SINGLE_STAR_FIND_BINARIES
#error "SINGLE_STAR_FIND_BINARIES has been removed: its only consumer was SINGLE_STAR_TIMESTEPPING > 0 (super-timestepping). Remove it from Config.sh."
#endif
#ifdef TURB_DRIVING_TARGET_SIGMA
#error "TURB_DRIVING_TARGET_SIGMA has been renamed TURB_DRIVING_TARGET_VRMS_REGION (same value: radius of the measured region, 0 for all gas). The target velocity is TurbDrive_ApproxRMSVturb."
#endif


#ifdef SINGLE_STAR_DIRECT_GRAVITY /* exact O(N_star^2) star-star gravity, replacing the tree for those pairs */
#if !defined(SINK_CALC_DISTANCES)
#define SINK_CALC_DISTANCES /* the tree's node-level sink_mass is what tells the walk which nodes to drop for star targets */
#endif
#endif


#ifdef STARS_INTEGRATED_BY_RUMI /* Type-4 stars integrated by RUMI++ (galaxy_sf/rumi_coupling.cc): star-star pairs leave the tree */
#if defined(SINK_PARTICLES) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_DIRECT_GRAVITY) || defined(HERMITE_INTEGRATION) || defined(SINK_CALC_DISTANCES) || defined(SPECIAL_POINT_MOTION) || defined(PMGRID) || defined(KETJU_REGULARIZATION) || defined(STARS_HERMITE_DIRECT_GRAVITY)
#error "STARS_INTEGRATED_BY_RUMI takes over the tree's sink node moments for Type 4: not combinable with sinks, SINGLE_STAR_DIRECT_GRAVITY, Hermite, KETJU, other SINK_CALC_DISTANCES users or PMGRID"
#endif
#define SINK_CALC_DISTANCES /* node star moments (sink_mass/sink_pos) let star targets drop star sources */
#define SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES (4)
#define TREE_STAR_STAR_EXCLUDED_TYPE 4
#endif
#if defined(SINGLE_STAR_DIRECT_GRAVITY) && !defined(TREE_STAR_STAR_EXCLUDED_TYPE)
#define TREE_STAR_STAR_EXCLUDED_TYPE (STAR_DIRECT_GRAVITY_TYPE) /* the tree leaves star-star pairs to star_direct_gravity.cc */
#endif


#ifdef MHD_MODIFIED_GRADIENT
#ifndef MAGNETIC
#define MAGNETIC
#endif
/* MG builds on top of the CG infrastructure (FaceCrossX, FaceDotB, Face_Area),
   but replaces the iterative correction with a global sparse-matrix solve.
   If user set CG explicitly alongside MG, warn them. Otherwise, auto-enable CG=1. */
#ifdef MHD_CONSTRAINED_GRADIENT
#undef MHD_CONSTRAINED_GRADIENT
#endif
#define MHD_CONSTRAINED_GRADIENT 1
#endif


#if defined(SPECIAL_POINT_MOTION)
#if !defined(SINK_CALC_DISTANCES)
#define SINK_CALC_DISTANCES /* make sure the distance tracking is actually enabled */
#endif
#if CHECK_IF_PREPROCESSOR_HAS_NUMERICAL_VALUE_(SPECIAL_POINT_MOTION)
#define SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES (SPECIAL_POINT_MOTION) /* set the special particle type to be used for tracking */
#endif
#endif
#if defined(SINK_CALC_DISTANCES) && !defined(SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES)
#define SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES (5) /* default to type = 5 for this module */
#endif

#if defined(GRAVITY_ANALYTIC)
#if CHECK_IF_PREPROCESSOR_HAS_NUMERICAL_VALUE_(GRAVITY_ANALYTIC)
#if (GRAVITY_ANALYTIC > 0)
#define GRAVITY_ANALYTIC_ANCHOR_TO_PARTICLE /* ok, analytic gravity is defined with a numerical value > 0, indicating we should use this flag */
#ifndef SINK_CALC_DISTANCES
#define SINK_CALC_DISTANCES
#endif
#endif
#endif
#endif


/* drift and kick the sink node moments between tree rebuilds, as the main moments already are. Requires
   sink_vel to exist to drift with, hence the condition below -- which has to be tested here, after every
   site that can turn SINK_CALC_DISTANCES on. SINGLE_STAR_DIRECT_GRAVITY is in the list because it
   subtracts the sink monopole from mixed nodes and so needs sink_pos on the same clock as u.d.s; it can
   be set without SINGLE_STAR_TIMESTEPPING, and where it is, the sink_vel declaration and the moment sums
   that fill it widen to match this macro. */
#if defined(SINK_CALC_DISTANCES) && (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION) || defined(SINGLE_STAR_DIRECT_GRAVITY) || defined(STARS_INTEGRATED_BY_RUMI))
#define SINK_NODE_MOTION_TRACKED
#endif
#if defined(STARS_INTEGRATED_BY_RUMI) && defined(GRAVITY_ANALYTIC_ANCHOR_TO_PARTICLE)
#error "STARS_INTEGRATED_BY_RUMI: GRAVITY_ANALYTIC_ANCHOR_TO_PARTICLE uses the sink node moments"
#endif


#if defined(COOL_MOLECFRAC)
#if (COOL_MOLECFRAC == 6) && !defined(COOL_MOLECFRAC_NONEQM) && !defined(CHIMES)
#define COOL_MOLECFRAC_NONEQM // estimate molecular fractions for thermochemistry+cooling with explicitly-evolved non-equilibirum H2 formation+destruction with clumping and self-shielding (Hopkins et al arXiv:2203.00040)
#endif
#endif


#if defined(EOS_SUBSTELLAR_ISM) || defined(COOL_MOLECFRAC_NONEQM)
#define EOS_GAMMA_VARIABLE
#endif


#if defined(EOS_GAMMA_VARIABLE)
#ifndef EOS_GENERAL
#define EOS_GENERAL /*! needs to be on for this to work */
#endif
#endif
#define GAMMA_DEFAULT (EOS_GAMMA)


#if defined(CONDUCTION) || defined(EOS_GENERAL) || defined(TURB_DIFF_ENERGY)
#define DOGRAD_INTERNAL_ENERGY 1
#endif


#if defined(EOS_GENERAL)
#define DOGRAD_SOUNDSPEED 1
#endif


#if (defined(GALSF) || defined(GALSF_RESOLVEDISM)) && !defined(GALSF_GENERATIONS) /* Stella defines GALSF further down */
#define  GALSF_GENERATIONS     1    /*!< Number of star particles that may be created per gas particle */
#endif


#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
#define OUTPUT_SOFTENING  /*! output softening to snapshots */
#endif


#ifdef GALSF_SFR_IMF_VARIATION
#define N_IMF_FORMPROPS  13  /*!< formation properties of star particles to record for output */
#endif


#define CRFLUID_REDUCED_C_CODE(k) (return_CRbin_M1speed(k)) // allow for bin-to-bin variations in RSOL


#if defined(FOF) || defined(SUBFIND) /* the group finders are the only users of the link types */
#ifndef FOF_PRIMARY_LINK_TYPES
#define FOF_PRIMARY_LINK_TYPES 2
#endif
#ifndef FOF_SECONDARY_LINK_TYPES
#define FOF_SECONDARY_LINK_TYPES 0
#endif
#endif


#if defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)
#define OUTPUT_TIDAL_TENSOR
#define OUTPUT_SOFTENING
#endif


#ifdef RT_INFRARED
#define COOL_LOWTEMP_THIN_ONLY // don't want to double-count trapping of radiation if we're doing it self-consistently
#endif

/* the thermodynamics follows gas into the optically thick regime (cooling not restricted to optically thin gas, the infrared
   radiation that traps it, or a barotropic equation of state), so a collapsing core stalls at the opacity limit: sink formation
   (sfr_eff.cc) and sink capture (sink.cc) both ask this one question, after every block above that can switch on one of its inputs */
#if (defined(COOLING) && !defined(COOL_LOWTEMP_THIN_ONLY)) || defined(RT_INFRARED) || defined(EOS_GMC_BAROTROPIC)
#define SINK_OPACITY_LIMIT_THERMODYNAMICS
#endif



/* ----- giant block of options for RHD modules ------ */

/* Stella tree bands PE, LW, NUV, OPT plus each source's 6-8 eV share of NUV; G0(6-13.6 eV) = PE + share */
#ifdef GALSF_RESOLVEDISM_G0_FROM_PE_PLUS_NUV_6_8EV_SHARE
#if !defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RADTRANSFER) || defined(RT_M1) || defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_LOCALRAYGRID)
#error "GALSF_RESOLVEDISM_G0_FROM_PE_PLUS_NUV_6_8EV_SHARE needs the tree field (GALSF_RESOLVEDISM_G0_VARIABLE) without M1/RADTRANSFER"
#endif
#ifndef GALSF_RESOLVEDISM_NUV_VARIABLE
#define GALSF_RESOLVEDISM_NUV_VARIABLE
#endif
#endif

/* per-cell 6-8 eV flux total: grain heating needs NUV below 6 eV while NUV_flux carries 3.4-8 eV */
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && (defined(GALSF_RESOLVEDISM_NUV_VARIABLE) || defined(GALSF_RESOLVEDISM_RADPRESSURE)) && !defined(EXPERIMENTAL_RADIATION_DISJOINT_BANDS) && !defined(TREE_RAY) && !defined(RADTRANSFER)
#define GALSF_RESOLVEDISM_CELL_FLUX_6_8EV_TOTAL
#endif

/* Stella tree radiation field: band luminosities ride on the gravity-tree source arrays (no LEBRON flux or force) */
#ifdef EXPERIMENTAL_G0_STELLAR_LUM_CODE_UNITS
#define RESOLVEDISM_STELLAR_LUM_TO_CGS (UNIT_LUM_IN_CGS) /* node stellar_lum[] in code units */
#else
#define RESOLVEDISM_STELLAR_LUM_TO_CGS (1.0) /* node stellar_lum[] in erg/s */
#endif
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && !defined(RADTRANSFER)
#define RT_USE_GRAVTREE
#define RT_PHOTOELECTRIC
#define RT_FUV_6_8EV
#define RT_LYMAN_WERNER
#if defined(GALSF_RESOLVEDISM_NUV_VARIABLE) || defined(GALSF_RESOLVEDISM_RADPRESSURE)
#define RT_NUV
#endif
#if defined(GALSF_RESOLVEDISM_OPT_VARIABLE) || defined(GALSF_RESOLVEDISM_RADPRESSURE)
#define RT_OPTICAL_NIR
#endif
#ifndef RT_SOURCES
#define RT_SOURCES (16+32)
#endif
#endif

/* options for FIRE RT method */
#if defined(GALSF_FB_FIRE_RT_LONGRANGE)
#define RT_LEBRON // this flag requires lebron for rhd
#endif
#if defined(RT_LEBRON)
#define RT_USE_GRAVTREE // use gravity tree for flux propagation
#define RT_USE_GRAVTREE_SAVE_RAD_ENERGY
#if !defined(GALSF_FB_FIRE_RT_LONGRANGE)
#define RADTRANSFER // for cross-compatibility reasons, if the FIRE version is not on, need RADTRANSFER flag also enabled
#define RT_USE_GRAVTREE_SAVE_RAD_FLUX
#endif
#endif

/* check whether we want to use the implicit solver [only usable for very special cases, not recommended] */
#if defined(RT_DIFFUSION_IMPLICIT) && (defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION)) // only modules the implicit solver works with
#define RT_DIFFUSION_CG // use our implicit solver [will crash with any other modules, hence checking this before the others below]
#endif

/* options for FLD or OTVET or M1 or Ray/Rad_Intensity modules */
#if defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION) || defined(RT_M1) || defined(RT_LOCALRAYGRID)
#ifndef RADTRANSFER
#define RADTRANSFER // RADTRANSFER is ON, obviously
#endif
#define RT_SOURCE_INJECTION // need source injection enabled to define emissivity
#if !defined(RT_DIFFUSION_CG)
#define RT_SOLVER_EXPLICIT // default to explicit solutions (much more accurate/flexible)
#endif
#endif /* end of options for our general RHD methods */

/* Stella M1: the NUV bin carries 6-8 eV so NUV + PHOTOELECTRIC spans the 6-13.6 eV chemistry FUV band */
#if defined(GALSF_RESOLVEDISM) && defined(RADTRANSFER) && defined(SGCHEM)
#ifndef RT_NUV
#define RT_NUV
#endif
#define RT_NUV_BAND_6_8EV
#define RT_FUV_BANDS_SGCHEM_DUST_LAW /* NUV/PE/LW: pure absorption with the chemistry's dust cross sections and per-cell DGR */
#endif

/* OTVET-specific options [uses the gravity tree to calculate the Eddington tensor] */
#if defined(RT_OTVET)
#define RT_USE_GRAVTREE // use gravity tree for Eddington tensor
#ifndef RT_SEPARATELY_TRACK_LUMPOS
#define RT_SEPARATELY_TRACK_LUMPOS // and be sure to track luminosity locations
#endif
#endif /* end of otvet-specific options */

/* M1-specific options [make sure to add the flux moment */
#if defined(RT_M1)
#define RT_EVOLVE_FLUX // evolve flux moment [not just energy moment assumed by FLD/OTVET]
#endif

/* options for direct/exact Jiang et al. method for direct evolution on an intensity grid */
#if defined(RT_LOCALRAYGRID)
#define RT_EVOLVE_INTENSITIES // evolve the intensities explicitly
#define N_RT_INTENSITY_BINS (4*(RT_LOCALRAYGRID)*((RT_LOCALRAYGRID)+1)) // define number of directional bins, used throughout
#define RT_INTENSITY_BINS_DOMEGA (4.*M_PI/((double)N_RT_INTENSITY_BINS)) // normalization coefficient (for convenience defined here)
#endif

/* check if we are -explicitly- evolving the radiation energy density [0th moment], in which case we need to carry time-derivatives of the field */
#if defined(RT_SOLVER_EXPLICIT) && !defined(RT_EVOLVE_INTENSITIES) // only needed if we are -not- evolving intensities and -are- solving explicitly
#define RT_EVOLVE_ENERGY
#if !defined(RT_EVOLVE_FLUX) // evolving energy explicitly but not flux, flux-limiting is not disabled
#define RT_FLUXLIMITER // default to include flux-limiter under these conditions
#endif
#endif

/* enable radiation pressure forces unless they have been explicitly disabled */
#if defined(RADTRANSFER) && !defined(RT_DISABLE_RAD_PRESSURE) && !defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)
#define RT_RAD_PRESSURE_FORCES
#endif

#ifdef RT_SOURCE_INJECTION
#if defined(GALSF) && !defined(RT_INJECT_PHOTONS_DISCRETELY)
#define RT_INJECT_PHOTONS_DISCRETELY // modules will not work correctly with differential timestepping with point sources without discrete injection
#endif
#if defined(RT_INJECT_PHOTONS_DISCRETELY) && defined(RT_RAD_PRESSURE_FORCES) && (defined(RT_ENABLE_R15_GRADIENTFIX) || defined(GALSF))
#define RT_INJECT_PHOTONS_DISCRETELY_ADD_MOMENTUM_FOR_LOCAL_EXTINCTION // adds correction for un-resolved extinction which cannot generate photon momentum with M1, FLD, OTVET, etc.
#endif
#endif

/* check if we need to explicitly calculate gradients of the radiation pressure tensor for the diffusive step */
#if (defined(RT_FLUXLIMITER) || defined(RT_RAD_PRESSURE_FORCES) || defined(RT_SOLVER_EXPLICIT)) && !defined(RT_COMPGRAD_EDDINGTON_TENSOR) //&& !defined(RT_EVOLVE_FLUX) && !defined(RT_EVOLVE_INTENSITIES))
#define RT_COMPGRAD_EDDINGTON_TENSOR
#endif

/* enable appropriate chemistry flags if we are using the photoionization modules */
#if defined(RT_CHEM_PHOTOION)
#if (RT_CHEM_PHOTOION > 1)
/* enables multi-frequency radiation transport for ionizing photons. Integration variable is the ionising intensity J_nu */
#define RT_CHEM_PHOTOION_HE
#define RT_PHOTOION_MULTIFREQUENCY // if using He-ionization, default to multi-frequency RT [otherwise doesn't make sense] //
#endif
#endif

/* enable appropriate flags for X-ray sub-modules */
#if defined(RT_XRAY)
#if (RT_XRAY == 1)
#define RT_SOFT_XRAY
#endif
#if (RT_XRAY == 2)
#define RT_HARD_XRAY
#endif
#if (RT_XRAY == 3)
#define RT_SOFT_XRAY
#define RT_HARD_XRAY
#endif
#endif

/* default to speed-of-light equal to actual speed-of-light, and stars as photo-ionizing sources */
#ifndef RT_SPEEDOFLIGHT_REDUCTION
#define RT_SPEEDOFLIGHT_REDUCTION (1.0)
#endif
#if (defined(RADTRANSFER) || defined(RT_USE_GRAVTREE) || defined(RT_SOURCE_INJECTION)) && !defined(RT_SOURCES) /* only radiation code reads the source types */
#define RT_SOURCES 1+2+4+8+16+32 // default to allowing all types to act as sources //
#endif

#if !defined(RT_USE_GRAVTREE) && defined(RT_SELFGRAVITY_OFF) && !defined(SELFGRAVITY_OFF)
#define SELFGRAVITY_OFF // safely define SELFGRAVITY_OFF in this case, otherwise we act like there is gravity except in the final setting of accelerations
#endif

/* turn on outputs appropriately */
#ifdef RADTRANSFER
#if !defined(OUTPUT_EDDINGTON_TENSOR) && !defined(IO_SUPPRESS_OUTPUT_EDDINGTON_TENSOR)
#define OUTPUT_EDDINGTON_TENSOR
#endif
#endif


/* set up the 'bins' for different frequencies for the RHD -- there's probably a more elegant way to do this with inline functions
    instead of precompiler logic, but for now this is the solution */
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
#define RT_BIN0 (-1)

#ifndef RT_CHEM_PHOTOION
#define RT_FREQ_BIN_H0 (RT_BIN0+0)
#else
#define RT_FREQ_BIN_H0 (RT_BIN0+1)
#endif

#ifndef RT_PHOTOION_MULTIFREQUENCY
#define RT_FREQ_BIN_He0 (RT_FREQ_BIN_H0+0)
#define RT_FREQ_BIN_He1 (RT_FREQ_BIN_He0+0)
#define RT_FREQ_BIN_He2 (RT_FREQ_BIN_He1+0)
#else
#define RT_FREQ_BIN_He0 (RT_FREQ_BIN_H0+1)
#define RT_FREQ_BIN_He1 (RT_FREQ_BIN_He0+1)
#define RT_FREQ_BIN_He2 (RT_FREQ_BIN_He1+1)
#endif

#ifdef RT_CHEM_PHOTOION
#define RT_BAND_IS_IONIZING(k) ((k==RT_FREQ_BIN_H0) || (k==RT_FREQ_BIN_He0) || (k==RT_FREQ_BIN_He1) || (k==RT_FREQ_BIN_He2))
#else
#define RT_BAND_IS_IONIZING(k) (0) // no ionizing bands without RT_CHEM_PHOTOION
#endif

#ifndef GALSF_FB_FIRE_RT_LONGRANGE
#define RT_FREQ_BIN_FIRE_UV (RT_FREQ_BIN_He2+0)
#define RT_FREQ_BIN_FIRE_OPT (RT_FREQ_BIN_FIRE_UV+0)
#define RT_FREQ_BIN_FIRE_IR (RT_FREQ_BIN_FIRE_OPT+0)
#else
#define RT_FREQ_BIN_FIRE_UV (RT_FREQ_BIN_He2+1)
#define RT_FREQ_BIN_FIRE_OPT (RT_FREQ_BIN_FIRE_UV+1)
#define RT_FREQ_BIN_FIRE_IR (RT_FREQ_BIN_FIRE_OPT+1)
#endif

#ifndef RT_SOFT_XRAY
#define RT_FREQ_BIN_SOFT_XRAY (RT_FREQ_BIN_FIRE_IR+0)
#else
#define RT_FREQ_BIN_SOFT_XRAY (RT_FREQ_BIN_FIRE_IR+1)
#endif

#ifndef RT_HARD_XRAY
#define RT_FREQ_BIN_HARD_XRAY (RT_FREQ_BIN_SOFT_XRAY+0)
#else
#define RT_FREQ_BIN_HARD_XRAY (RT_FREQ_BIN_SOFT_XRAY+1)
#endif

#ifndef RT_PHOTOELECTRIC
#define RT_FREQ_BIN_PHOTOELECTRIC (RT_FREQ_BIN_HARD_XRAY+0)
#else
#define RT_FREQ_BIN_PHOTOELECTRIC (RT_FREQ_BIN_HARD_XRAY+1)
#endif

#ifndef RT_LYMAN_WERNER
#define RT_FREQ_BIN_LYMAN_WERNER (RT_FREQ_BIN_PHOTOELECTRIC+0)
#else
#define RT_FREQ_BIN_LYMAN_WERNER (RT_FREQ_BIN_PHOTOELECTRIC+1)
#endif

#ifndef RT_NUV
#define RT_FREQ_BIN_NUV (RT_FREQ_BIN_LYMAN_WERNER+0)
#else
#define RT_FREQ_BIN_NUV (RT_FREQ_BIN_LYMAN_WERNER+1)
#endif

#ifndef RT_OPTICAL_NIR
#define RT_FREQ_BIN_OPTICAL_NIR (RT_FREQ_BIN_NUV+0)
#else
#define RT_FREQ_BIN_OPTICAL_NIR (RT_FREQ_BIN_NUV+1)
#endif

#ifndef RT_FUV_6_8EV
#define RT_FREQ_BIN_FUV_6_8EV (RT_FREQ_BIN_OPTICAL_NIR+0)
#else
#define RT_FREQ_BIN_FUV_6_8EV (RT_FREQ_BIN_OPTICAL_NIR+1) /* 6-8 eV: with PHOTOELECTRIC (8-13.6 eV) gives the 6-13.6 eV Habing band */
#endif

#ifndef RT_FREEFREE
#define RT_FREQ_BIN_FREEFREE (RT_FREQ_BIN_FUV_6_8EV+0)
#else
#define RT_FREQ_BIN_FREEFREE (RT_FREQ_BIN_FUV_6_8EV+1)
#endif


#ifndef RT_GENERIC_USER_FREQ
#define RT_FREQ_BIN_GENERIC_USER_FREQ (RT_FREQ_BIN_FREEFREE+0)
#else
#define RT_FREQ_BIN_GENERIC_USER_FREQ (RT_FREQ_BIN_FREEFREE+1)
#endif


/* be sure to add all new wavebands to these lists, or else we will run into problems */
/* ALSO, the IR bin here should be the last bin: add additional bins ABOVE this line */
#ifndef RT_INFRARED
#define RT_FREQ_BIN_INFRARED (RT_FREQ_BIN_GENERIC_USER_FREQ+0)
#else
#define RT_FREQ_BIN_INFRARED (RT_FREQ_BIN_GENERIC_USER_FREQ+1)
#endif

#define N_RT_FREQ_BINS (RT_FREQ_BIN_INFRARED+1)

#endif // #if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)


/* ----- end block of options for RHD modules ------ */



#if defined(GALSF) || defined(SINK_PARTICLES) || defined(RADTRANSFER) || defined(OUTPUT_DENS_AROUND_NONGAS) || defined(CHIMES) || defined(RT_REPROCESS_INJECTED_PHOTONS)
#define DO_DENSITY_AROUND_NONGAS_PARTICLES
#if !defined(ALLOW_IMBALANCED_GASPARTICLELOAD)
#define ALLOW_IMBALANCED_GASPARTICLELOAD
#endif
#endif

#if defined(SINK_SWALLOWGAS)
#define SINK_FOLLOW_ACCRETED_COM
#define SINK_FOLLOW_ACCRETED_MOMENTUM
#if defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINK_GRAVCAPTURE_GAS)
#define SINK_FOLLOW_ACCRETED_ANGMOM 0 // follow accreted AM just from explicit 'swallow' operations
#else
#define SINK_FOLLOW_ACCRETED_ANGMOM 1 // follow accreted AM from 'swallowed' BH particles, and from continuous/smooth properties [mdot] of kernel gas near BH
#endif
#endif

#ifdef MAGNETIC /* recommended MHD switches -- only turn these off for de-bugging */
#if defined(MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE) && defined(MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY)
#error "MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE and MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY are alternatives"
#endif
#if defined(DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH) /* constrained psi/c_h cleaning (Tricco & Price 2012; Tricco, Price & Bate 2016; Steinwandel & Price arXiv:2511.19615) replaces Dedner */
#if !defined(MAGNETIC) || !defined(HYDRO_MESHLESS_FINITE_MASS)
#error "DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH requires MAGNETIC and HYDRO_MESHLESS_FINITE_MASS"
#endif
#ifdef DIVBCLEANING_DEDNER
#error "DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH replaces DIVBCLEANING_DEDNER: do not set both"
#endif
#ifndef DIVBCLEANING_CONSERVATIVE_SIGMA
#if !defined(BOX_SPATIAL_DIMENSION) || (BOX_SPATIAL_DIMENSION == 3)
#define DIVBCLEANING_CONSERVATIVE_SIGMA 1.0 /* TPB16 optimum in 3D */
#else
#define DIVBCLEANING_CONSERVATIVE_SIGMA 0.3 /* TPB16 optimum in 2D */
#endif
#endif
#endif
#if !defined(TEST_MHD_NO_DEDNER_CLEANING) && !defined(DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH) /* test only: disables div-cleaning */
#define DIVBCLEANING_DEDNER         /* hyperbolic/parabolic div-cleaing (Dedner 2002), with TP improvements */
#endif
#ifdef HYDRO_SPH /* MHD switches specific to SPH MHD */
#define SPH_TP12_ARTIFICIAL_RESISTIVITY   /* turns on magnetic dissipation ('artificial resistivity'): uses tricco switch =h*|gradB|/|B| */
#endif
#endif

/* constrained gradients fit the cleaning-field gradient at the face mid-points; that needs the Dedner field, so the
   mid-point fit is on exactly when both are, whatever order they were set in (it replaces the ordinary phi gradient) */
#if defined(MHD_CONSTRAINED_GRADIENT) && defined(DIVBCLEANING_DEDNER)
#ifndef MHD_CONSTRAINED_GRADIENT_MIDPOINT
#define MHD_CONSTRAINED_GRADIENT_MIDPOINT
#endif
#else
#undef MHD_CONSTRAINED_GRADIENT_MIDPOINT
#endif

#if defined(TRANSPORT_SUBCYCLE) && !defined(RADTRANSFER)
#error "TRANSPORT_SUBCYCLE subcycles radiation transport and needs a RADTRANSFER build (cosmic rays alone are not subcycled)"
#endif
#if defined(TRANSPORT_SUBCYCLE) && defined(RT_EVOLVE_INTENSITIES)
#error "TRANSPORT_SUBCYCLE does not carry the radiation intensities (Dt_Rad_Intensity is not in the subcycle flux pass); run intensities without subcycling"
#endif

#if defined(TURB_DIFF_ENERGY) || defined(TURB_DIFF_VELOCITY) || defined(TURB_DIFF_MASS) || defined(TURB_DIFF_METALS)
#define TURB_DIFFUSION /* top-level switch to calculate properties needed for scalar turbulent diffusion/mixing: must enable with any specific version */
#if defined(TURB_DIFF_VELOCITY) && !defined(VISCOSITY)
#define VISCOSITY
#endif
#if defined(TURB_DIFF_ENERGY) && !defined(CONDUCTION)
#define CONDUCTION
#endif
#endif

#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS) || defined(GALSF_ISMDUSTCHEM_MODEL) || defined(RT_INFRARED)
#define OUTPUT_DUST_TO_GAS_RATIO // helpful if these special modules are on to see this output and save it for use in analysis
#endif

#if defined(OUTPUT_POTENTIAL) && !defined(EVALPOTENTIAL)
#define EVALPOTENTIAL
#endif

#if defined(SINK_PARTICLES) && (defined(SINK_REPOSITION_ON_POTMIN) || defined(SINK_SEED_FROM_FOF))
#ifndef EVALPOTENTIAL
#define EVALPOTENTIAL
#endif
#endif

#ifdef EVALPOTENTIAL
#ifndef COMPUTE_POTENTIAL_ENERGY
#define COMPUTE_POTENTIAL_ENERGY
#endif
#endif

#ifdef COOL_MOLECFRAC_NONEQM
#ifndef OUTPUT_MOLECULAR_FRACTION
#define OUTPUT_MOLECULAR_FRACTION
#endif
#endif




#ifdef BOX_SHEARING
/* set default compile-time flags for the shearing-box (or shearing-sheet) boundaries */
/* shearing box boundaries: 1=r-z sheet (coordinates [0,1,2] = [r,z,phi]), 2=r-phi sheet [r,phi,z], 3=[r-phi-z] box */
#if (BOX_SHEARING==1)
#define BOX_SHEARING_PHI_COORDINATE 2
#else
#define BOX_SHEARING_PHI_COORDINATE 1
#endif
/* if the r-z or r-phi sheet is set, the code must be compiled in 2D mode */
#if (BOX_SHEARING==1) || (BOX_SHEARING==2)
#ifndef BOX_SPATIAL_DIMENSION
#define BOX_SPATIAL_DIMENSION 2
#endif
#endif
/* box must be periodic in this approximation */
#ifndef BOX_PERIODIC
#define BOX_PERIODIC
#endif
/* if not set, default to q=3/2 (q==-dlnOmega/dlnr, used for boundary and velocity corrections) */
#ifndef BOX_SHEARING_Q
#define BOX_SHEARING_Q (3.0/2.0)
#endif
/* Omega at the box center, in code units; defaults to 1 (time in units of 1/Omega) */
#ifndef BOX_SHEARING_OMEGA_BOX_CENTER
#define BOX_SHEARING_OMEGA_BOX_CENTER 1.0
#endif
/* need analytic gravity on so we can add the appropriate source terms to the EOM */
#ifndef GRAVITY_ANALYTIC
#define GRAVITY_ANALYTIC
#endif
/* if self-gravity is on, we need to make sure the gravitational forces are not periodic. this is going to cause some errors at the x/y 'edges',
 but for now at least, the periodic gravity routines (particularly the FFT's involved) require a regular periodic map, they cannot handle the
 non-standard map that the shearing box represents. */
#if !defined(GRAVITY_NOT_PERIODIC) && !defined(GRAVITY_PERIODIC_XY) /* GRAVITY_PERIODIC_XY handles the shear-periodic map by Ewald summation */
#define GRAVITY_NOT_PERIODIC
#endif
#endif // BOX_SHEARING

#ifdef GRAVITY_PERIODIC_XY_IMAGES /* self-gravity periodic in x/y by explicit image sums, open in z (e.g. tall boxes): no Ewald or PM */
#if !defined(BOX_PERIODIC) || defined(BOX_OUTFLOW_X) || defined(BOX_REFLECT_X) || defined(BOX_OUTFLOW_Y) || defined(BOX_REFLECT_Y)
#error "GRAVITY_PERIODIC_XY_IMAGES requires BOX_PERIODIC with periodic x and y boundaries"
#endif
#if !defined(BOX_OUTFLOW_Z) && !defined(BOX_REFLECT_Z)
#error "GRAVITY_PERIODIC_XY_IMAGES requires a non-periodic z boundary (BOX_OUTFLOW_Z or BOX_REFLECT_Z)"
#endif
#if defined(BOX_SHEARING) || defined(PMGRID) || defined(GRAVITY_SPHERICAL_SYMMETRY) || defined(SUBFIND)
#error "GRAVITY_PERIODIC_XY_IMAGES is not compatible with BOX_SHEARING, PMGRID, GRAVITY_SPHERICAL_SYMMETRY or SUBFIND"
#endif
#ifdef TREE_RADIATION_WRAPS_WITHOUT_GRAVITY
#error "TREE_RADIATION_WRAPS_WITHOUT_GRAVITY is redundant with GRAVITY_PERIODIC_XY_IMAGES: tree radiation and columns already use the x/y nearest image"
#endif
#ifndef GRAVITY_NOT_PERIODIC
#define GRAVITY_NOT_PERIODIC /* switches off the cubic-periodic (Ewald) solver; the tree wraps x/y itself */
#endif
#endif



#ifdef TALLBOX
#error "TALLBOX was removed: use BOX_PERIODIC + BOX_OUTFLOW_Z (+ BOX_LONG_Z) with GRAVITY_PERIODIC_XY_IMAGES"
#endif

#if defined(GRAVITY_ANALYTIC_STRATIFIED_DISK) && !defined(GRAVITY_ANALYTIC)
#define GRAVITY_ANALYTIC
#endif

#ifdef GRAVITY_PERIODIC_XY /* self-gravity (shear-)periodic in x/y, open in z, by a 2D Ewald correction on the tree */
#if !defined(BOX_PERIODIC) || defined(BOX_REFLECT_X) || defined(BOX_OUTFLOW_X) || defined(BOX_REFLECT_Y) || defined(BOX_OUTFLOW_Y)
#error "GRAVITY_PERIODIC_XY requires BOX_PERIODIC with periodic x and y boundaries"
#endif
#if !(defined(BOX_OUTFLOW_Z) || defined(BOX_REFLECT_Z))
#error "GRAVITY_PERIODIC_XY requires a non-periodic z boundary (BOX_OUTFLOW_Z or BOX_REFLECT_Z)"
#endif
#if defined(PMGRID) || defined(GRAVITY_NOT_PERIODIC) || defined(GRAVITY_SPHERICAL_SYMMETRY) || (defined(BOX_SHEARING) && (BOX_SHEARING < 3))
#error "GRAVITY_PERIODIC_XY is incompatible with PMGRID, GRAVITY_NOT_PERIODIC, GRAVITY_SPHERICAL_SYMMETRY and 2D shearing sheets"
#endif
#endif

#ifdef SNE_BROADCAST_TIMESTEP
#if !defined(GALSF_FB_MECHANICAL) || !defined(GALSF_FB_FIRE_STELLAREVOLUTION)
#error "SNE_BROADCAST_TIMESTEP requires GALSF_FB_MECHANICAL with GALSF_FB_FIRE_STELLAREVOLUTION"
#endif
#endif


/* block for metals and other passive scalars, should stay in this order. like the RHD, probably a more elegant way to do this with functions, but designed here to use compiler logic instead */
#ifdef METALS
#ifdef GALSF_FB_FIRE_RPROCESS
#define NUM_RPROCESS_SPECIES (GALSF_FB_FIRE_RPROCESS)
#else
#define NUM_RPROCESS_SPECIES 0
#endif

#ifdef GALSF_FB_FIRE_AGE_TRACERS
#define NUM_AGE_TRACERS (GALSF_FB_FIRE_AGE_TRACERS)
#else
#define NUM_AGE_TRACERS 0
#endif

#ifdef STARFORGE_FEEDBACK_TRACERS
#define NUM_STARFORGE_FEEDBACK_TRACERS ((STARFORGE_FEEDBACK_TRACERS) > 3 ? (STARFORGE_FEEDBACK_TRACERS) : 3) /* the writers fill one field per channel (0=jets, 1=winds, 2=SNe), so at least 3 */
#else
#define NUM_STARFORGE_FEEDBACK_TRACERS 0
#endif

#ifdef COOL_METAL_LINES_BY_SPECIES
#define NUM_LIVE_SPECIES_FOR_COOLTABLES 10
#else
#define NUM_LIVE_SPECIES_FOR_COOLTABLES 0
#endif

#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
/* FIRE-pattern extended, 28 slots:
 *   Metallicity[0]      = total Z (sum of metal mass fractions)
 *   Metallicity[1]      = H        (computed from conservation: 1 − Z − Y, then stored)
 *   Metallicity[2]      = He
 *   Metallicity[3..27]  = 25 individual metals (C, N, O, F, Ne, Na, Mg, Al, Si,
 *                         P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn)
 * Σ_{k=1..27} Metallicity[k] = X + Y + Z = 1 by construction.
 * Index translation: Metallicity slot for StellarElement k = k + 1 (so
 *   Met[ELEM_H+1]=Met[1]=H, Met[ELEM_He+1]=Met[2]=He, ..., Met[ELEM_Zn+1]=Met[27]=Zn).
 * Slot 0 (total Z) is redundant with Σ_{k=3..27} Met[k] but kept for legacy
 * cooling code that reads Metallicity[0] / SolarAbundances[0]. */
#define NUM_METAL_SPECIES 28
#ifdef COOL_METAL_LINES_BY_SPECIES
#error "GALSF_RESOLVEDISM_METALS_INDIVIDUAL is incompatible with COOL_METAL_LINES_BY_SPECIES (its tables and helpers read the FIRE slot order, He at 1)"
#endif
#else
#define NUM_METAL_SPECIES (1+NUM_LIVE_SPECIES_FOR_COOLTABLES+NUM_RPROCESS_SPECIES+NUM_AGE_TRACERS+NUM_STARFORGE_FEEDBACK_TRACERS)
#endif
#define AGE_TRACER_OFFSET (1+NUM_LIVE_SPECIES_FOR_COOLTABLES+NUM_RPROCESS_SPECIES) /* first enrichment-age-tracer slot in Metallicity[] */
#define STARFORGE_FEEDBACK_TRACER_OFFSET (AGE_TRACER_OFFSET+NUM_AGE_TRACERS) /* first feedback-tracer slot in Metallicity[] */
#endif // METALS //


#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION 0 /* placeholder for arbitrary number of additional species to be used for different operations like yields, etc. */


#if defined(GALSF_ISMDUSTCHEM_MODEL) /* define some global and other useful variables for dust chemistry modules which also utilize the metals info above */
#ifndef GALSF_ISMDUSTCHEM_SILICATE_COMPOSITION
#define GALSF_ISMDUSTCHEM_SILICATE_COMPOSITION (1) /* default: olivine-pyroxene mix; the module reads this as a plain integer, so it must always be defined when GALSF_ISMDUSTCHEM_MODEL is active */
#endif
#if defined(COOLING)
#define GALSF_ISMDUSTCHEM_HIGHTEMPDUSTCOOLING // optional, can turn off
#endif
#ifndef OUTPUT_MOLECULAR_FRACTION
#define OUTPUT_MOLECULAR_FRACTION // useful since we track dense molecular fraction
#endif

#define NUM_ISMDUSTCHEM_ELEMENTS (1+NUM_LIVE_SPECIES_FOR_COOLTABLES) // number of metal species evolved for dust
#define NUM_ISMDUSTCHEM_SOURCES (4) // Sources of dust creation/growth 0=gas-dust accretion, 1=SNe Ia, 2=SNe II, 3=AGB outflows
#define NUM_ISMDUSTCHEM_SPECIES (2*(GALSF_ISMDUSTCHEM_MODEL & 1) + (GALSF_ISMDUSTCHEM_MODEL & 2)/2 + (GALSF_ISMDUSTCHEM_MODEL & 4)/4 + (GALSF_ISMDUSTCHEM_MODEL & 8)/8) /* number of dust species tracked, depends on the model */
#define NUM_ISMDUSTCHEM_SPECIES_IDS (5) /* number of possible species IDs (0=silicate,1=carbon,2=free iron,3=O reservoir,4=iron inclusions); the sparse SpeciesFieldIndexTable is indexed by these fixed IDs, not the packed tracked-species count */
#if !(GALSF_ISMDUSTCHEM_MODEL & 1)
#error "GALSF_ISMDUSTCHEM_MODEL requires bit 1 (silicates+carbon are the base species always tracked); e.g. use (1), (1+2), ..."
#endif
#if (GALSF_ISMDUSTCHEM_MODEL & 8) && !(GALSF_ISMDUSTCHEM_MODEL & 2)
#error "GALSF_ISMDUSTCHEM_MODEL bit 8 (iron inclusions in silicates) requires bit 2 (metallic iron)"
#endif
#define GALSF_ISMDUSTCHEM_VAR_ELEM_IN_SILICATES 4 /* O, Mg, Si, and Fe needed to make silicates */
#undef NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION
#ifdef GALSF_ISMDUSTCHEM_GRAINSIZEEVO
#ifndef OUTPUT_MACH_NUMBER // mach number is used for subresolved clumping in grain size evo model
#define OUTPUT_MACH_NUMBER
#endif
#define NUM_ISMDUSTCHEM_SIZE_BINS (GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
#define UNIT_GRAIN_NUMBER      (All.UnitGrainNumber)
#define UNIT_GRAIN_LENGTH      (All.UnitGrainLength_in_cm)
#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION (NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES+(2*NUM_ISMDUSTCHEM_SPECIES*NUM_ISMDUSTCHEM_SIZE_BINS))
#else
#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION (NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES)
#endif
#endif // end of GALSF_ISMDUSTCHEM_MODEL block

/* end of metals block */




/* ---- Stellar-mass BH promotion: auto-enable sink framework ---- */
#if defined(GALSF_RESOLVEDISM_BH_PROMOTION)
#ifndef SINK_PARTICLES
#define SINK_PARTICLES
#endif
#ifndef SINK_GRAVCAPTURE_GAS
#define SINK_GRAVCAPTURE_GAS
#endif
#endif


/* ??? this should be done with an inline function inside the particle data, not a compiler directive */
#define SinkParticle_GravityKernelRadius (All.ForceSoftening[5])


#ifdef SINK_PARTICLES
#define SINK_COUNTPROGS /* carries a counter for each BH that gives the total number of seeds that merged into it */
#define SINK_ENFORCE_EDDINGTON_LIMIT /* put a hard limit on the maximum accretion rate (set SinkEddingtonFactor>>1 to allow super-eddington) */
#if defined(SINK_PHOTONMOMENTUM) || defined(RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION)
#define SINK_CALC_LOCAL_ANGLEWEIGHTS
#endif
#if defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_GRAVACCRETION) || defined(SINK_GRAVCAPTURE_NONGAS) || defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_REPOSITION_ON_POTMIN)
#define SINK_NEIGHBOR_BITFLAG 63 /* allow all particle types in the BH search: 63=2^0+2^1+2^2+2^3+2^4+2^5 */
#else
#define SINK_NEIGHBOR_BITFLAG 33 /* only search for particles of types 0 and 5 (gas and sinks) around a primary BH particle */
#endif
#endif


#if defined(BOX_REFLECT_X) || defined(BOX_REFLECT_Y) || defined(BOX_REFLECT_Z) || defined(BOX_OUTFLOW_X) || defined(BOX_OUTFLOW_Y) || defined(BOX_OUTFLOW_Z)
#define BOX_DEFINED_SPECIAL_XYZ_BOUNDARY_CONDITIONS_ARE_ACTIVE 1 /* flag to let the code know to use everything below */
#define BOX_VALUE_FOR_NOTHING_SPECIAL_BOUNDARY_ 20 /* define a dummy value we won't have the user set for reference below */
#endif


#if defined(CHIMES_HII_REGIONS) && !defined(GALSF_FB_FIRE_RT_HIIHEATING)
#define GALSF_FB_FIRE_RT_HIIHEATING // must be on, this module uses the same code
#endif
#ifdef CHIMES_STELLAR_FLUXES /* The following defines the stellar age bins that we will use to define the UV spectra from stars used in CHIMES. */
#define CHIMES_LOCAL_UV_NBINS 8
#define CHIMES_LOCAL_UV_AGE_LOW 0.0
#define CHIMES_LOCAL_UV_DELTA_AGE_LOW 0.2
#define CHIMES_LOCAL_UV_AGE_MID 1.0
#define CHIMES_LOCAL_UV_DELTA_AGE_HI 1.0
#endif


/* ---- GALSF_RESOLVEDISM: resolved ISM single-star galaxy formation ---- */
#ifdef GALSF_RESOLVEDISM
#ifndef GALSF
#define GALSF
#endif
#ifndef COOLING
#define COOLING
#endif
#ifndef METALS
#define METALS
#endif
/* NOTE (2026-07-04): we deliberately do NOT silently #undef the FIRE feedback
 * flags here. Silently disabling them hid genuine misconfigurations. FIRE +
 * Stella exclusivity is now enforced LOUDLY by the #error block below (Uli:
 * "we don't wanna [silently] set those"). */
#endif /* GALSF_RESOLVEDISM */

/* Stella's resolved-ISM feedback (resolvedism_fb_*) fully REPLACES the FIRE-2/3
   feedback modules. Running both double-counts SN energy/momentum/mass/metals and
   CR injection, and duplicates the radiation feedback. The FIRE routines are gated by
   their own flags and Stella's by GALSF_RESOLVEDISM_FB, with nothing preventing both
   from being enabled — enforce the mutual exclusivity at compile time. */
#ifdef GALSF_RESOLVEDISM_FB
#if defined(GALSF_FB_THERMAL)
#error "GALSF_RESOLVEDISM_FB is mutually exclusive with GALSF_FB_THERMAL: Stella provides all SN/wind/thermal feedback via resolvedism_fb_*; the thermal module would double-count. Disable it."
#endif
#if defined(GALSF_FB_MECHANICAL) && !defined(GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL)
#error "GALSF_RESOLVEDISM_FB is mutually exclusive with GALSF_FB_MECHANICAL unless GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL restricts each to its own stars: otherwise both act on every Type-4 star and double-count."
#endif
#if !defined(GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL) && (defined(GALSF_FB_FIRE_RT_LOCALRP) || defined(GALSF_FB_FIRE_RT_LONGRANGE) || defined(GALSF_FB_FIRE_RT_HIIHEATING) || defined(GALSF_FB_FIRE_STELLAREVOLUTION))
#error "GALSF_RESOLVEDISM_FB is mutually exclusive with the FIRE-2/3 GALSF_FB_FIRE_* modules unless GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL restricts each to its own stars: Stella handles radiation feedback + stellar evolution via GALSF_RESOLVEDISM_G0_VARIABLE/PHOTOION/RADPRESSURE/STELLAR_TABLES (or M1 RT)."
#endif
#endif /* GALSF_RESOLVEDISM_FB mutual exclusivity */

#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) && (defined(GALSF_FB_FIRE_STELLAREVOLUTION) || defined(GALSF_ISMDUSTCHEM_MODEL))
#error "GALSF_RESOLVEDISM_METALS_INDIVIDUAL puts H in Metallicity[1], FIRE yields/dust write He..Fe into fixed slots 1..10: the layouts clash silently"
#endif

/* Stella x FIRE-3 coupled mode: allowed and forbidden combinations */
#ifdef GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL
#if !defined(GALSF_RESOLVEDISM_FB) || !defined(GALSF_RESOLVEDISM_SAMPLE_IMF)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL requires GALSF_RESOLVEDISM_FB and GALSF_RESOLVEDISM_SAMPLE_IMF"
#endif
#if defined(SGCHEM)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL uses the FIRE cooling for all gas: disable SGCHEM"
#endif
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) && !defined(GALSF_RESOLVEDISM_HYBRID_CARRY_ALL_ELEMENTS)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL carries the FIRE metal layout: disable GALSF_RESOLVEDISM_METALS_INDIVIDUAL"
#endif
#if defined(GALSF_RESOLVEDISM_HYBRID_CARRY_ALL_ELEMENTS)
#error "GALSF_RESOLVEDISM_HYBRID_CARRY_ALL_ELEMENTS is not implemented yet"
#endif
#if defined(GALSF_RESOLVEDISM_DUST) || defined(GALSF_ISMDUSTCHEM_MODEL)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL: no dust model yet, disable GALSF_RESOLVEDISM_DUST and GALSF_ISMDUSTCHEM_MODEL"
#endif
#if defined(CHIMES)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL is not compatible with CHIMES"
#endif
#if defined(FIRE_PHYSICS_DEFAULTS) && (FIRE_PHYSICS_DEFAULTS < 3)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL requires FIRE_PHYSICS_DEFAULTS=3"
#endif
#if defined(GALSF_RESOLVEDISM_BH_PROMOTION)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL keeps Stella remnants as Type-4 stars: disable GALSF_RESOLVEDISM_BH_PROMOTION"
#endif
#if defined(GALSF_RESOLVEDISM_HYBRID_SINKS_BELOW)
#error "GALSF_RESOLVEDISM_HYBRID_SINKS_BELOW is not implemented yet"
#endif
#if defined(GALSF_RESOLVEDISM_SN_SPAWN)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL: the SN spawn payload carries the 27-element layout (SpawnEjZ), not the FIRE slots: disable GALSF_RESOLVEDISM_SN_SPAWN"
#endif
#if defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL has no STARFORGE sink tier yet: disable SINGLE_STAR_SINK_DYNAMICS and SINGLE_STAR_AND_SSP_HYBRID_MODEL"
#endif
#if !defined(GALSF_RESOLVEDISM_HYBRID_M_SSP)
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL needs the SSP/Stella cell-mass threshold: set GALSF_RESOLVEDISM_HYBRID_DEFAULTS=(M_SSP)"
#endif
#ifndef GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION
#define GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION (64) /* SF criteria of Stella-channel cells, bits as GALSF_SFR_CRITERION; default = Stella production (Jeans mass only) */
#endif
#if (GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION & ~(2+64)) || ((GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION & ~(SINGLE_STAR_SINK_FORMATION)) & (2+64))
#error "GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION: only bits 2 (convergent flow) and 64 (Jeans mass) have Stella versions"
#endif
#endif /* GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL */

#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
#ifdef RADTRANSFER
#error "GALSF_RESOLVEDISM_G0_VARIABLE and RADTRANSFER are mutually exclusive: M1 RT handles FUV via RT_PHOTOELECTRIC/RT_LYMAN_WERNER"
#endif
#endif

#if defined(CR_BACKGROUND_IONRATE_ATTENUATED) && defined(COSMIC_RAY_FLUID)
#error "CR_BACKGROUND_IONRATE_ATTENUATED is the CR-fluid-off mode: with COSMIC_RAY_FLUID the ionisation rate comes from the CR fluid only"
#endif
#if defined(SGCHEM_CRHEAT_K23) || defined(SGCHEM_PE_K23) || defined(SGCHEM_H2FORM_K23) || defined(SGCHEM_H2SHIELD_B5) || defined(SGCHEM_GASDUST_K23) || defined(SGCHEM_CR_ATTEN_K23)
#error "SGCHEM_*_K23 and SGCHEM_H2SHIELD_B5 were removed: the Kim+2023 (TIGRESS-NCR) physics is CHEMISTRYNETWORK=20"
#endif
#if defined(CR_BACKGROUND_IONRATE_ATTENUATED) && (!defined(SGCHEM) || !defined(RT_USE_TREECOL_FOR_NH) || defined(CR_SCALE_WITH_G0))
#error "CR_BACKGROUND_IONRATE_ATTENUATED needs SGCHEM and the TreeCol column (RT_USE_TREECOL_FOR_NH), and replaces CR_SCALE_WITH_G0"
#endif
#if defined(GALSF_RESOLVEDISM_G0_SCALE_SFR) || defined(GALSF_RESOLVEDISM_GHOSTD_G0_CR) || defined(GHOSTD_G0_CR)
#error "GALSF_RESOLVEDISM_G0_SCALE_SFR, GALSF_RESOLVEDISM_GHOSTD_G0_CR and GHOSTD_G0_CR were renamed: use GHOSDT_G0_CR"
#endif
#if defined(GHOSDT_G0_CR) && !(defined(GALSF) && (defined(RT_ISRF_BACKGROUND) || defined(SGCHEM)))
#error "GHOSDT_G0_CR scales the RT_ISRF_BACKGROUND field (or the SGCHEM G0) and CR rate with the star formation rate: requires GALSF and RT_ISRF_BACKGROUND or SGCHEM"
#endif

#if defined(GALSF_RESOLVEDISM_PHOTOION) && defined(RADTRANSFER)
#error "GALSF_RESOLVEDISM_PHOTOION and RADTRANSFER are mutually exclusive: use RT_CHEM_PHOTOION instead"
#endif

#if defined(GALSF_RESOLVEDISM_PHOTOION_RECOMB_AT_CELL_TEMPERATURE) && !defined(GALSF_RESOLVEDISM_PHOTOION)
#error "GALSF_RESOLVEDISM_PHOTOION_RECOMB_AT_CELL_TEMPERATURE requires GALSF_RESOLVEDISM_PHOTOION"
#endif

#if defined(GALSF_RESOLVEDISM_RADPRESSURE) && defined(RADTRANSFER)
#error "GALSF_RESOLVEDISM_RADPRESSURE and RADTRANSFER are mutually exclusive: M1 RT handles radiation pressure via RT_RAD_PRESSURE_FORCES"
#endif

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
#ifndef N_STELLAR_MASS
#define N_STELLAR_MASS 300
#endif
#if defined(GALSF_RESOLVEDISM_WINDS) && defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && (N_STELLAR_MASS < 5)
#error "GALSF_RESOLVEDISM_WINDS uses the birth mass in MstarSampleIMF[4]: needs N_STELLAR_MASS >= 5"
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
#error "GALSF_RESOLVEDISM_SAMPLE_IMF and GALSF_RESOLVEDISM_STOCHASTIC_IMF are mutually exclusive"
#endif
#endif

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
/* HDF5 stellar evolution tables for single-star feedback */
#endif
#ifdef GALSF_RESOLVEDISM_WINDS
#ifndef GALSF_RESOLVEDISM_STELLAR_TABLES
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#endif
#endif
#ifdef GALSF_RESOLVEDISM_AGB
#ifndef GALSF_RESOLVEDISM_STELLAR_TABLES
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#endif
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
#ifdef GALSF_RESOLVEDISM_METALS_ISOTOPES  /* 40 individual isotopes from stellar tables */
#define NUM_RESOLVEDISM_ELEMENTS 40
#else
#define NUM_RESOLVEDISM_ELEMENTS 27  /* 27 elements: H through Zn */
#endif
#ifndef GALSF_RESOLVEDISM_STELLAR_TABLES
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#endif
#endif
#ifndef NUM_RESOLVEDISM_ELEMENTS
#define NUM_RESOLVEDISM_ELEMENTS 0
#endif
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
#ifndef GALSF_RESOLVEDISM_STELLAR_TABLES
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#endif
/* the band-resolved radp direct term (2026-07-08 correctness pass) couples
 * f_FUV*L_FUV + f_NUV*L_NUV + f_OPT*L_OPT — the NUV/OPT band fields are NOT
 * default-compiled, and without them that contribution silently vanishes
 * (under-coupled direct pressure). RADPRESSURE therefore requires the band
 * luminosities: auto-enable both. */
#ifndef GALSF_RESOLVEDISM_NUV_VARIABLE
#define GALSF_RESOLVEDISM_NUV_VARIABLE
#endif
#ifndef GALSF_RESOLVEDISM_OPT_VARIABLE
#define GALSF_RESOLVEDISM_OPT_VARIABLE
#endif
/* ---- Direct radiation-pressure force from the tree per-cell fluxes
 * (LEBRON-style, Uli 2026-07-08) ----
 * When the variable ISRF is available (G0_VARIABLE turns on the HEALPix tree columns, so
 * every gas cell carries the attenuated per-pixel UV/LW/NUV/OPT_flux) and M1 RT
 * is off, close the subgrid radiation-pressure model by computing the DIRECT
 * single-scattering radp momentum PER GAS CELL from those attenuated fluxes and
 * depositing it LOCALLY (correct direction + shadowing, in the swept-up shell),
 * instead of computing it at the star from the star's local column and kernel-
 * weighting the deposit. This is the production-relevant configuration, so
 * auto-enable it here. MUTUALLY EXCLUSIVE with the star-side DIRECT radp term:
 * that block in resolvedism_fb_momentum.cc is #ifdef'd out when this is set, so
 * radp is not double-counted; the IR-trapping boost (local-Sigma approximation)
 * is DELIBERATELY RETAINED in the star-side path (the per-cell path does the DIRECT
 * UV/NUV/OPT term only). */
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && !defined(RADTRANSFER)
#ifndef GALSF_RESOLVEDISM_RADPRESSURE_TREERAD
#define GALSF_RESOLVEDISM_RADPRESSURE_TREERAD
#endif
#endif
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
#ifndef GALSF_RESOLVEDISM_STELLAR_TABLES
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#endif
#endif
#ifdef GALSF_RESOLVEDISM_DUST
#define NUM_RESOLVEDISM_DUST 5  /* C, O_sil, Mg_sil, Si_sil, Fe_sil */
#ifndef GALSF_RESOLVEDISM_FB
#error "GALSF_RESOLVEDISM_DUST requires GALSF_RESOLVEDISM_FB"
#endif
#ifndef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
#error "GALSF_RESOLVEDISM_DUST requires GALSF_RESOLVEDISM_METALS_INDIVIDUAL"
#endif
#ifdef GALSF_RESOLVEDISM_DUST_SIZE_BINS
#define NUM_RESOLVEDISM_DUST_BINS (GALSF_RESOLVEDISM_DUST_SIZE_BINS)  /* MRN log bins, each with its own v_rel (Hu+19) */
#else
#define NUM_RESOLVEDISM_DUST_BINS 1  /* single MRN a_eff = 0.035 um (Hu+26) */
#endif
#ifdef EXPERIMENTAL_DUST_HU_MODEL
#ifdef GALSF_RESOLVEDISM_DUST_NO_MCKEE
#error "GALSF_RESOLVEDISM_DUST_NO_MCKEE does not apply to EXPERIMENTAL_DUST_HU_MODEL: swept-mass SN destruction is off unless GALSF_RESOLVEDISM_DUST_SN_SWEPT_MASS_DESTRUCTION"
#endif
#ifdef GALSF_RESOLVEDISM_DUST_SHOCK_DESTRUCTION
#error "GALSF_RESOLVEDISM_DUST_SHOCK_DESTRUCTION does not apply to EXPERIMENTAL_DUST_HU_MODEL: nonthermal sputtering is explicit (dust v_rel)"
#endif
#elif defined(GALSF_RESOLVEDISM_DUST_SIZE_BINS) || defined(GALSF_RESOLVEDISM_DUST_NO_BETATRON) || defined(GALSF_RESOLVEDISM_DUST_SN_SWEPT_MASS_DESTRUCTION) || defined(GALSF_RESOLVEDISM_DUST_SN_EJECTA_CONDENSATION)
#error "GALSF_RESOLVEDISM_DUST_SIZE_BINS/NO_BETATRON/SN_SWEPT_MASS_DESTRUCTION/SN_EJECTA_CONDENSATION require EXPERIMENTAL_DUST_HU_MODEL"
#endif
#endif
#ifndef GALSF_RESOLVEDISM_DUST
#define NUM_RESOLVEDISM_DUST 0
#endif

#if defined(GALSF_RESOLVEDISM_BH_PROMOTION) && !defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
#error "GALSF_RESOLVEDISM_BH_PROMOTION requires GALSF_RESOLVEDISM_STELLAR_TABLES"
#endif

#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || defined(GALSF_RESOLVEDISM_DUST)
#undef NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION
/* When METALS_INDIVIDUAL is on, the 27 resolvedism elements live inside
 * Metallicity[] (NUM_METAL_SPECIES=27), so they no longer need passive-scalar
 * slots.  Only Dust still needs extra slots. */
#if defined(GALSF_ISMDUSTCHEM_MODEL)
#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION (NUM_ISMDUSTCHEM_ELEMENTS+NUM_ISMDUSTCHEM_SOURCES+NUM_ISMDUSTCHEM_SPECIES+NUM_RESOLVEDISM_DUST)
#else
#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION (NUM_RESOLVEDISM_DUST)
#endif
#endif

/* ---- SGCHEM: non-equilibrium chemistry solver (Fortran backend) ---- */
#ifdef SGCHEM
#ifndef COOLING
#define COOLING
#endif
#if defined(CHIMES)
#error "SGCHEM and CHIMES are mutually exclusive chemistry solvers"
#endif
#ifndef CHEMISTRYNETWORK
#define CHEMISTRYNETWORK 5
#endif
/* define TRAC_NUM (number of tracked species) based on chemistry network */
#if CHEMISTRYNETWORK == 1
#define TRAC_NUM 6
#elif CHEMISTRYNETWORK == 2
#define TRAC_NUM 10
#elif CHEMISTRYNETWORK == 3
#define TRAC_NUM 19
#elif CHEMISTRYNETWORK == 4
#define TRAC_NUM 2
#elif CHEMISTRYNETWORK == 5
#define TRAC_NUM 3
#elif CHEMISTRYNETWORK == 6
#define TRAC_NUM 4
#elif CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
#define TRAC_NUM 19
#elif CHEMISTRYNETWORK == 8
#define TRAC_NUM 19
#elif CHEMISTRYNETWORK == 9
#define TRAC_NUM 15
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
#define TRAC_NUM 9
#elif CHEMISTRYNETWORK == 17
#define TRAC_NUM 7
#elif CHEMISTRYNETWORK == 18
#define TRAC_NUM 113 /* H2, H+, CO and every ion stage of He C N O Ne Mg Si S Fe (cxx/chem_net18_layout.h) */
#ifdef SGCHEM_FORTRAN
#error "CHEMISTRYNETWORK=18 (RTZ metal ions) exists only in the C++ backend: drop SGCHEM_FORTRAN"
#endif
#elif CHEMISTRYNETWORK == 20
#define TRAC_NUM 3   /* H2, H+ evolved; CO = equilibrium value written back */
#ifdef SGCHEM_FORTRAN
#error "CHEMISTRYNETWORK=20 (TIGRESS-NCR) exists only in the C++ backend: drop SGCHEM_FORTRAN"
#endif
#endif
#if defined(TURB_DIFF_METALS)
/* Include SGCHEM tracked abundances (H2, H+, CO, ...) in turbulent diffusion
   so that TracAbund is diffused consistently with ElementAbundance. Without this,
   diffusion can move carbon out of a cell while leaving CO behind, causing
   CO > abundc and crashing DVODE. Network 17 also tracks D (deuterium) as a
   separate scalar since D+ and HD lock D atoms and can't exceed the D pool. */
#if (CHEMISTRYNETWORK == 17)
#define NUM_D_DIFFUSE 1
#else
#define NUM_D_DIFFUSE 0
#endif
#undef NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || defined(GALSF_RESOLVEDISM_DUST)
/* With NUM_METAL_SPECIES=27 (FIRE-pattern), resolvedism elements live in
 * Metallicity[] — no extra slots needed for them. */
#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION (NUM_RESOLVEDISM_DUST+TRAC_NUM+NUM_D_DIFFUSE)
#else
#define NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION (TRAC_NUM+NUM_D_DIFFUSE)
#endif
#endif
#endif /* SGCHEM */


/* tree columns: bins, deposit scheme and species (shared with the chemistry Fortran) */
#include "treecol_config.h"


#if defined(COOLING) && defined(GALSF_EFFECTIVE_EQS)
#ifndef COOLING_OPERATOR_SPLIT
#define COOLING_OPERATOR_SPLIT /*!< the Springel-Hernquist EOS depends explicitly on the cooling time in a way that requires de-coupled hydro cooling */
#endif
#endif


#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
#if !(CHECK_IF_PREPROCESSOR_HAS_NUMERICAL_VALUE_(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)) // allow to be set to integer value to represent a >1 number of special zoom sites
#undef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
#define SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM (1)
#endif
#endif

#if defined(GALSF_RESOLVEDISM_WINDS_THERMAL) && defined(GALSF_RESOLVEDISM_WINDS_THERMAL_ONLY)
#error "GALSF_RESOLVEDISM_WINDS_THERMAL and _THERMAL_ONLY are mutually exclusive wind energy modes"
#endif

/* ---- TREE_RAY: reverse-ray progressive dust attenuation IN the gravity walk ----
 * (Wunsch+2021 TreeRay onion shells; Stella third radiation branch, 2026-07-07.)
 * Per (HEALPix pixel x log-radial shell): sources deposit L/4pi r^2 at the
 * luminosity-centroid geometry, absorbers deposit gas mass at COM geometry; a
 * post-walk prefix sweep attenuates shell-by-shell -> attenuated UV/LW/NUV/OPT.
 * HIERARCHY, not exclusivity: requires G0_VARIABLE (which turns on the HEALPix tree columns);
 * the H2/CO tree columns stay untouched — H2/CO LINE self-shielding remains target-side
 * in chemistry, applied ONCE (curve-of-growth, not exponential; see the
 * h2-double-shield fix 2026-07-07). TreeRay takes over ONLY the stellar-band dust
 * continuum (calc_photo skips the column on stellar pixels; floor pixels keep the
 * target-side pixel-column attenuation). Future extension (do
 * NOT bolt on casually): per-source H2 line shielding from binned H2 shells —
 * requires splitting stellar-vs-background fshield in rate_eq. */
#ifdef TREE_RAY
#if defined(RADTRANSFER)
#error "TREE_RAY and RADTRANSFER (M1) are mutually exclusive radiation transport schemes"
#endif
#if !defined(GALSF_RESOLVEDISM_G0_VARIABLE)
#error "TREE_RAY requires GALSF_RESOLVEDISM_G0_VARIABLE (it attenuates the per-pixel stellar FUV field)"
#endif
#ifndef TREE_RAY_NSHELL
#define TREE_RAY_NSHELL 12   /*! log-radial shells per pixel for the attenuation sweep */
#endif
#endif /* TREE_RAY */
#if defined(TREE_RAY_PI) && !defined(TREE_RAY)
#error "TREE_RAY_PI requires TREE_RAY"
#endif
#ifdef TREE_RAY_PI
#if (CHEMISTRYNETWORK == 17)
/* net-17 tracks He/He+/He++: carry the full M1-multifreq band structure
 * ({13.6,24.6,54.4,70} eV edges, blackbody-weighted sigmas at Teff=4e4 K)
 * so TreeRay-PI reproduces the validated nested He/H fronts. */
#define TREERAY_PI_HE
#define N_TRPI_BANDS 4
#else
#define N_TRPI_BANDS 1
#endif
#endif
#if defined(TREE_RAY_IR) && !defined(TREE_RAY)
#error "TREE_RAY_IR requires TREE_RAY"
#endif
#if defined(SINK_WIND_SPAWN) && !defined(WAKEUP_REVERSE_KICK_ON_DEMOTION) && !defined(WAKEUP_TRUNCATE_STEP_ON_DEMOTION)
#define WAKEUP_TRUNCATE_STEP_ON_DEMOTION // the WAKEUP kick reversal injects energy and only fires with spawned cells present, so truncate the step instead whenever we are spawning. must come after every SINK_WIND_SPAWN derivation above. WAKEUP_REVERSE_KICK_ON_DEMOTION opts back out
#endif

#if defined(SINK_WIND_SPAWN) && !defined(MERGE_SPLIT_DISCARD_ENERGY) && !defined(MERGE_SPLIT_CONSERVE_ENERGY)
#define MERGE_SPLIT_CONSERVE_ENERGY // merging sets the velocity momentum-conservingly, so the pair's COM-frame kinetic energy leaves the budget; discarding it costs ~23% of the injected wind energy in the adiabatic wind_singlestar test, so return it as heat whenever we are spawning. pairs with the truncation above: the two errors have opposite sign and were cancelling, which is why the discard went unnoticed. MERGE_SPLIT_DISCARD_ENERGY opts back out
#endif

/* The two above fix the ENERGY BOOKKEEPING of a merge; the two below govern WHICH pairs may merge at
   all, which is what preserves the structure a sustained outflow builds. Keyed on the jet/wind modules
   rather than on SINK_WIND_SPAWN, because the concern is a long-lived contact discontinuity between
   outflow and ambient gas: SINGLE_STAR_FB_SNE also spawns cells but is a single impulsive event with
   no such interface to maintain. With the bookkeeping fixed but these off, the adiabatic
   wind_singlestar bubble still misses Weaver's coefficient by 2.4% and the radiative one by 5.7%,
   losing ~11 points of retained energy and expanding closer to momentum- than energy-driven. */
#if (defined(SINGLE_STAR_FB_JETS) || defined(SINGLE_STAR_FB_WINDS)) && !defined(SINK_SPAWN_MERGE_ANY_NEIGHBOR) && !defined(SINK_SPAWN_NO_MERGE) && !defined(SINK_SPAWN_MERGE_WHEN_AMBIENT)
#define SINK_SPAWN_MERGE_WHEN_AMBIENT // retire a spawned cell only once its own state has equilibrated with its kernel, kinematically and thermally, rather than as soon as it is subsonic wrt whichever neighbour is picked as target. A contact discontinuity is in pressure equilibrium with no velocity jump, so outflow material sitting at the contact passes every kinematic test while remaining a distinct phase. SINK_SPAWN_MERGE_ANY_NEIGHBOR opts back out; SINK_SPAWN_NO_MERGE supersedes it
#endif

#if (defined(SINGLE_STAR_FB_JETS) || defined(SINGLE_STAR_FB_WINDS)) && !defined(MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION) && !defined(MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION)
#define MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION // the shock-side counterpart: never merge a pair whose merge would thermalize the majority of the energy involved, which cooling then radiates away, making the merge an artificial unresolved shock. MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION opts back out
#endif


/* ---- KETJU regularized integrator ---- */
#if defined(KETJU_REGULARIZATION) && !defined(OUTPUT_STAR_ENERGY_DIRECT_SUM)
#define OUTPUT_STAR_ENERGY_DIRECT_SUM /* KETJU validation reads ETOT_INCODE */
#endif
#if defined(KETJU_REGULARIZATION)
#if defined(FORCE_EQUAL_TIMESTEPS)
#error "KETJU_REGULARIZATION is not compatible with FORCE_EQUAL_TIMESTEPS"
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL) && ((ADAPTIVE_GRAVSOFT_FORALL) & 16)
#error "KETJU_REGULARIZATION needs fixed star softening: use ADAPTIVE_GRAVSOFT_FORALL without bit 16 and SofteningStars <= KetjuRegionRadius (AGS zeta on stars breaks collisional dynamics)"
#endif
#endif
#if defined(KETJU_MERGE_STARS) && !defined(KETJU_REGULARIZATION)
#error "KETJU_MERGE_STARS requires KETJU_REGULARIZATION"
#endif
#if defined(KETJU_MERGE_BH) && !defined(KETJU_REGULARIZATION)
#error "KETJU_MERGE_BH requires KETJU_REGULARIZATION"
#endif
#if defined(KETJU_LEGACY_C_LIBRARY) && !defined(KETJU_REGULARIZATION)
#error "KETJU_LEGACY_C_LIBRARY requires KETJU_REGULARIZATION"
#endif
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
#error "GALSF_RESOLVEDISM_SAMPLE_IMF: stars merge only on contact; GALSF_MERGER_STARCLUSTER_PARTICLES merges by distance"
#endif
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(SINGLE_STAR_SINK_DYNAMICS)
#error "GALSF_RESOLVEDISM_SAMPLE_IMF: stars merge only on contact; STARFORGE sink dynamics merges bound sinks within the softening"
#endif
#if defined(KETJU_REGULARIZATION) && (defined(GALSF_RESOLVEDISM) || !defined(SINGLE_STAR_SINK_DYNAMICS)) && !defined(KETJU_TYPE4_STARS)
#define KETJU_TYPE4_STARS /* Type-4 stars are chain centers/members (Stella, pure N-body); STARFORGE sink runs keep Type 5 only */
#endif
#if defined(KETJU_REGULARIZATION) && (defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINK_PARTICLES))
#define KETJU_SINKS /* Type-5 sinks (STARFORGE stars or BHs) carry KETJU spin and PN class */
#endif

#if defined(TURB_DIFF_METALS) && (defined(GALSF_RESOLVEDISM) || defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || defined(GALSF_RESOLVEDISM_DUST)) && !defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
#define TURB_DIFF_METALS_EXACT_CONSERVATION /* Stella: element-slot closure and dust diffusion use the pending exchange */
#endif
/* gradient and hydro loops reuse the first gradient pass's local pair searches (mesh/ngb.cc); identical results, memory ~ 4 B per
   neighbour of every active cell. Off where something between those loops could change a search: NGB_PAIR_CACHE_OFF opts out */
#if !defined(NGB_PAIR_CACHE_OFF) && !defined(TURB_DIFF_DYNAMIC) && !defined(MHD_MODIFIED_GRADIENT) && !(defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES) && (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4))
#define NGB_PAIR_CACHE
#endif

/* STARFORGE_OPTIMIZATIONS component, defined here because it needs the final hydro-method and threading flags: each active
   pair's hydro flux is evaluated once, from the smaller-timestep side (opt out: HYDRO_PAIR_ONCE_OFF). Pure-MPI MFM only, so
   OpenMP builds keep the two-sided evaluation. Not bitwise: results differ at about the level of a rank-count change (one-sided
   face fallbacks and RT clamps are taken from the evaluating side). M2e4 96x1 one node, full cycle: 0.93x wall, same step count;
   fb_norad 0.96-0.97x; Bate neutral. */
#if defined(STARFORGE_OPTIMIZATIONS) && !defined(HYDRO_PAIR_ONCE) && !defined(HYDRO_PAIR_ONCE_OFF) && defined(HYDRO_MESHLESS_FINITE_MASS) && !(defined(_OPENMP) || defined(OPENMP) || defined(BOX_SHEARING) || defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) || defined(HYDRO_SPH) || defined(HYDRO_MESHLESS_FINITE_VOLUME))
#define HYDRO_PAIR_ONCE
#endif
#if defined(HYDRO_PAIR_ONCE) && (defined(_OPENMP) || defined(OPENMP) || defined(BOX_SHEARING) || defined(USE_TIMESTEP_DILATION_FOR_ZOOMS) || defined(HYDRO_SPH) || defined(HYDRO_MESHLESS_FINITE_VOLUME))
#error "HYDRO_PAIR_ONCE: pure-MPI MFM only (j-side writes are not thread-safe; shearing, dilation, SPH and MFV fluxes are not handled)"
#endif

#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION) && !defined(TURB_DIFF_METALS)
#error "TURB_DIFF_METALS_EXACT_CONSERVATION requires TURB_DIFF_METALS"
#endif

/* coupled mode: upstream RT bins next to FIRE-3 break FIRE's all-bins consumers (docs/stella_hybrid/P4_DESIGN.md) until P4 masks them */
#if defined(GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL) && !defined(GALSF_RESOLVEDISM_TREE_BANDS_PIXEL_FLUX_ONLY) && (defined(RT_PHOTOELECTRIC) || defined(RT_LYMAN_WERNER) || defined(RT_NUV) || defined(RT_OPTICAL_NIR) || defined(RT_FUV_6_8EV))
#error "GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL with RT_PHOTOELECTRIC/LYMAN_WERNER/NUV/OPTICAL_NIR/FUV_6_8EV needs GALSF_RESOLVEDISM_TREE_BANDS_PIXEL_FLUX_ONLY (P4): FIRE's SSP light, LONGRANGE, EUV fallback, G0/H2/eos and dust-temperature paths would read Stella's bins"
#endif
