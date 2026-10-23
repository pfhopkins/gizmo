#!/bin/bash
####################################################################################################
# Template_Config_stella.sh — every switch of the Stella single-star ISM model (GALSF_RESOLVEDISM).
# Uncommented = production configuration (2026-10-14). Commented = optional, alternative or test-only.
# Rules enforced in declarations/precompiler_logic.h are noted as  [req ...] [excl ...] [auto ...].
# Params-side production values: CosmicRayIonRate 3e-19 (constant), MinGasTemp 10, G0 1.7.
####################################################################################################

########## Core ##########
HYDRO_MESHLESS_FINITE_MASS               % MFM hydro
GALSF_RESOLVEDISM                        % Stella master switch (resolved-ISM star formation)
GALSF_RESOLVEDISM_FB                     % stellar feedback (SN, Ia, winds, AGB, radp)  [req DO_DENSITY_AROUND_NONGAS_PARTICLES] [excl GALSF_FB_MECHANICAL/THERMAL, GALSF_FB_FIRE_*]
GALSF_RESOLVEDISM_STELLAR_TABLES         % HDF5 single-star tables (lifetimes, yields, luminosities)  [auto by WINDS, TYPE_IA, RADPRESSURE, METALS_INDIVIDUAL]
GALSF_SFR_CRITERION=(64)                 % Jeans-mass-only SF criterion (Hu & Sternberg 2023)
DO_DENSITY_AROUND_NONGAS_PARTICLES       % kernel weights around stars (required by FB)

########## Star formation / IMF ##########
GALSF_RESOLVEDISM_SAMPLE_IMF             % sample individual stars from Kroupa IMF (LYRA-style accretion radius)  [excl STOCHASTIC_IMF]
N_STELLAR_MASS=5                         % stellar mass slots per particle  [req >=5 with SAMPLE_IMF; default 300]
#GALSF_RESOLVEDISM_STOCHASTIC_IMF        % alternative: one massive-star mass per star particle  [excl SAMPLE_IMF]
#GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF     % instant SF above nHcutoffSF (param); off = pure Jeans criterion
GALSF_RESOLVEDISM_INSTANT_SN             % SNe before TimeInstantSN (param) explode instantly (start-up)

########## Supernovae / Type Ia / mass return ##########
GALSF_RESOLVEDISM_TYPE_IA                % Type Ia SNe from WD DTD (W7 yields)
#FB_G17_BOUNDARY_INJECTION               % gizmo2017-style SN boundary injection; production choice pending ladder-v2 head-to-head
#GALSF_RESOLVEDISM_SN_SPAWN              % spawn ejecta cells when the FB kernel holds too little gas (prototype)
#RESOLVEDISM_HOTSYNC_CLAMP               % clamp timebins in hot regions (validated with SN_SPAWN)
#GALSF_RESOLVEDISM_FB_HEALPIX            % HEALPix FB weighting (weights are mass-blind)
#GALSF_RESOLVEDISM_IA_DTD_BOOST          % TEST-ONLY: multiply Ia hazard (ia_dtd test case)

########## Stellar winds ##########
GALSF_RESOLVEDISM_WINDS                  % OB/WR winds + AGB mass return
GALSF_RESOLVEDISM_WINDS_CONTINUOUS       % per-step Mdot*dt injection (adopted 2026-07-05)
GALSF_RESOLVEDISM_WINDS_THERMAL          % inelastic-residual heating (adopted 2026-07-05)  [excl WINDS_THERMAL_ONLY]
#GALSF_RESOLVEDISM_WINDS_THERMAL_ONLY    % pure thermal winds (bracketing experiment; disables radp kick)  [excl WINDS_THERMAL]

########## Radiation: FUV / photoionisation / radiation pressure (subgrid) ##########
GALSF_RESOLVEDISM_PHOTOION               % subgrid Stromgren photoionisation (pisub)  [excl RADTRANSFER]
GALSF_RESOLVEDISM_G0_VARIABLE            % variable ISRF from stellar FUV  [req HEALPix RT_USE_TREECOL_FOR_NH, auto RADPRESSURE_TREERAD] [excl RADTRANSFER]
#GALSF_RESOLVEDISM_NUV_VARIABLE          % NUV band field  [auto by RADPRESSURE]
#GALSF_RESOLVEDISM_OPT_VARIABLE          % optical/NIR band field  [auto by RADPRESSURE]
GALSF_RESOLVEDISM_RADPRESSURE            % radiation pressure (direct + IR boost)  [auto NUV/OPT_VARIABLE] [excl RADTRANSFER]
#GALSF_RESOLVEDISM_RADPRESSURE_TREERAD   % per-cell direct radp from tree per-pixel fluxes  [auto by G0_VARIABLE without RADTRANSFER]
#GHOSTD_G0_CR                            % GHOSTD defaults: uniform G0 (and CR rate with CR_SCALE_WITH_G0) scaled with the 30 Myr SFR; G0_VARIABLE/M1/CR fluid override it
RT_USE_TREECOL_FOR_NH=12                 % tree columns, 12*NSIDE^2 HEALPix bins: gas + dust-weighted (G0_VARIABLE), H2 + CO (CHEMCOOL), cut at ShieldingLength
NEED_HEALPIX                             % HEALPix support

########## Alternative: TreeRay photoionisation ##########
#TREE_RAY                                % TreeRay radiation transport  [excl RADTRANSFER] [req G0_VARIABLE]
#TREE_RAY_PI                             % photoionisation via TreeRay instead of subgrid PHOTOION
#TREERAY_PI_HE                           % He ionisation bands in TreeRay PI

########## Alternative: M1 radiation transport (replaces subgrid PHOTOION/G0/RADPRESSURE) ##########
#RT_M1                                   % M1 moment method
#RT_CHEM_PHOTOION=2                      % ionising bins (2 = H0, He0, He1, He2)
#RT_CHEM_PHOTOION_HE                     % He photoionisation in M1 chemistry
#RT_PHOTOION_MULTIFREQUENCY              % multi-frequency ionising spectrum
#RT_PHOTOELECTRIC                        % 8-13.6 eV band (dust PE heating)
#RT_LYMAN_WERNER                         % 11.2-13.6 eV band (H2 dissociation)
#RT_INFRARED                             % dust-reprocessed IR
#RT_SPEEDOFLIGHT_REDUCTION=0.1           % reduced speed of light
#RT_RAD_PRESSURE_FORCES                  % M1 radiation pressure

########## Chemistry / cooling ##########
COOLING                                  % cooling master switch  [auto by CHEMCOOL]
COOLING_OPERATOR_SPLIT                   % operator-split cooling
CHEMCOOL                                 % non-equilibrium chemistry (Fortran DVODE)  [excl CHIMES]
CHEMISTRYNETWORK=17                      % 17 = full network with CO; 5 = reduced (both ABHE 0.1)
CO_SHIELDING                             % CO self-shielding
WSS_CIE_COOL                             % WSS09 metal-line CIE cooling for hot gas
#GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE     % per-particle X_H and abhe from element abundances
#CHEMCOOL_CXX                            % C++/CVODE chemistry backend (in development)
#GALSF_CHEMCOOL_DEBUG_INPUT              % DEBUG: validate species vs pools on DVODE entry
#CR_SCALE_WITH_G0                        % CR ionisation rate scales with G0 (production: constant rate)

########## Metals / dust ##########
METALS                                   % metal tracking
GALSF_RESOLVEDISM_METALS_INDIVIDUAL      % 27 elements H-Zn in Metallicity[]
#GALSF_RESOLVEDISM_METALS_ISOTOPES       % NOT IMPLEMENTED (#error): 40 isotopes instead of 27 elements
TURB_DIFF_METALS                         % turbulent metal diffusion
GALSF_RESOLVEDISM_DUST                   % 5-species dust model  [req FB, METALS_INDIVIDUAL]
#GALSF_RESOLVEDISM_DUST_SHOCK_DESTRUCTION % shock-tracked SN dust destruction
#GALSF_RESOLVEDISM_DUST_NO_MCKEE         % disable McKee SN dust destruction
#GALSF_RESOLVEDISM_DUST_SELFTEST         % TEST-ONLY: dust physics self-test dump

########## Cosmic rays ##########
#COSMIC_RAY_FLUID                        % CR fluid (alternative to constant rate)
#CRFLUID_DIFFUSION_MODEL=0               % constant CR diffusivity
#CRFLUID_SPEEDOFLIGHT_REDUCTION=(1.e8/C_LIGHT_CGS) % 1000 km/s CR transport speed (default c); transport along B needs B != 0
#INPUT_READ_COSMICRAYENERGY              % keep IC CR energy on fresh starts
#CHEMCOOL_KEEP_IC_ABUNDANCES             % keep IC ChemcoolTracAbund on fresh starts

########## Gravity / N-body / KETJU ##########
ADAPTIVE_GRAVSOFT_FORALL=1+16            % adaptive softening for gas + stars
TIDAL_TIMESTEP_CRITERION                 % tidal-tensor timestep criterion
#ADAPTIVE_TREEFORCE_UPDATE=0.0625        % stale tree forces — incompatible with the tree columns and fluxes
KETJU_REGULARIZATION                     % MSTAR chain regularisation for stellar encounters  [excl FORCE_EQUAL_TIMESTEPS]
KETJU_MERGE_STARS                        % stellar collisions (Type 4+4)  [req KETJU_REGULARIZATION]
#KETJU_MERGE_BH                          % BH mergers  [req KETJU_REGULARIZATION]
#KETJU_PN_REMNANT_TAG                    % tag remnant type at SN; PN on Type-4 NS/BH remnants
#GALSF_RESOLVEDISM_BH_PROMOTION          % promote massive remnants to sinks  [req STELLAR_TABLES] [auto SINK_PARTICLES]

########## MHD ##########
MAGNETIC                                 % ideal MHD
MHD_B_SET_IN_PARAMS                      % initial B from params

########## Box ##########
#BOX_PERIODIC                            % only when the box contains all mass
#PMGRID=256                              % PM long-range gravity grid
#SHIFT_BY_HALF_BOX                       % shift IC by half a box on read
#SHIFTBACK_ON_OUTPUT                     % undo the half-box shift in outputs
#GRAVITY_PERIODIC_XY_IMAGES              % tall boxes: x/y-periodic gravity by image sums, open z  [req BOX_PERIODIC + BOX_OUTFLOW_Z]
#GALSF_RESOLVEDISM_TALLBOX               % Hu+21 stellar-disk + vertical NFW external potential

########## Sinks ##########
#SINK_PARTICLES                          % sink particles  [auto by BH_PROMOTION]
#SINK_GRAVCAPTURE_GAS                    % sinks accrete bound gas  [auto by BH_PROMOTION]
#SINK_CALC_DISTANCES                     % sink distance bookkeeping

########## Parallel / output ##########
NO_ISEND_IRECV_IN_DOMAIN                 % blocking domain exchange
MULTIPLEDOMAINS=16                       % domain decomposition granularity
EVALPOTENTIAL                            % output potential
OUTPUT_GRADIENT_VEL                      % output velocity gradients
OUTPUT_TIMESTEP                          % output timesteps
OUTPUT_TEMPERATURE                       % output temperature
OUTPUT_COOLRATE                          % output cooling rate
OUTPUT_INDIVIDUAL_COOLRATES              % output per-process cooling rates

########## Test-only ##########
#GALSF_RESOLVEDISM_ISOLATED_FB_TEST      % TEST-ONLY: isolated-star FB test hooks
#TEST_DUMP_STATE_AROUND_WIND_DEPOSIT     % TEST-ONLY: append raw double state of all particles before/after each wind batch to output/wind_dep_dump.<task>.bin
#STELLA_SELFCHECK                        % TEST-ONLY: runtime invariant checks (harness)
