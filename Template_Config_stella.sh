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
#GALSF_RESOLVEDISM_POPIII_STARS          % Pop III channel: gas below PopIII_Zcrit_solar (1e-4) forms stars at n_H > PopIII_nH_thresh (1e6), div v < 0, x_H2 > PopIII_xH2_min (1e-3), M_J < PopIII_Mreservoir (1000 Msun); Pop III IMF (PopIII_IMF_Type 0: M_char 20, 1: log-flat; 1-300 Msun); full mass or no star; tables: Z = 1e-4 row until a Z = 0 row exists  [req SAMPLE_IMF, SGCHEM, METALS]

########## Supernovae / Type Ia / mass return ##########
# SN energies (resolvedism_sn_energy.h): 1e51 erg for CCSN, ECSN, PPISN; PISN [5.0 + 1.304 (M_He - 64)] 1e51, M_He = (13/24)(M - 20) held to 64-133 Msun (Heger & Woosley 2002, Wise+12a); 0 for failed SNe and direct BHs
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
#GALSF_RESOLVEDISM_PHOTOION_RECOMB_AT_CELL_TEMPERATURE % pisub photon cost per cell at case-B alpha(T_cell), T>=1e4 K  [req PHOTOION]
GALSF_RESOLVEDISM_G0_VARIABLE            % variable ISRF from stellar FUV  [req HEALPix RT_USE_TREECOL_FOR_NH, auto RADPRESSURE_TREERAD] [excl RADTRANSFER]
#GALSF_RESOLVEDISM_NUV_VARIABLE          % NUV band field  [auto by RADPRESSURE]
#GALSF_RESOLVEDISM_G0_FROM_PE_PLUS_NUV_6_8EV_SHARE  % tree bands PE/LW/NUV/OPT + per-source 6-8 eV share of NUV; G0 = PE + share (GPU-port band set)  [tree path only]
#GALSF_RESOLVEDISM_OPT_VARIABLE          % optical/NIR band field  [auto by RADPRESSURE]
GALSF_RESOLVEDISM_RADPRESSURE            % radiation pressure (direct + IR boost)  [auto NUV/OPT_VARIABLE] [excl RADTRANSFER]
#GALSF_RESOLVEDISM_RADPRESSURE_TREERAD   % per-cell direct radp from tree per-pixel fluxes  [auto by G0_VARIABLE without RADTRANSFER]
#GHOSDT_G0_CR                            % GHOSDT defaults: uniform G0 (and CR rate with CR_SCALE_WITH_G0) scaled with the 30 Myr SFR; G0_VARIABLE/M1/CR fluid override it
RT_USE_TREECOL_FOR_NH=12                 % tree columns, 12*NSIDE^2 HEALPix bins: gas + dust-weighted (G0_VARIABLE), H2 + CO (SGCHEM), cut at ShieldingLength
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
COOLING                                  % cooling master switch  [auto by SGCHEM]
COOLING_OPERATOR_SPLIT                   % operator-split cooling
SGCHEM                                 % non-equilibrium chemistry (Fortran DVODE)  [excl CHIMES]
SGCHEM_FORTRAN                % legacy Fortran backend (SGCHEM alone = C++/CVODE)
CHEMISTRYNETWORK=17                      % 17 = full network with CO; 5 = reduced (both ABHE 0.1)
#SGCHEM_PRIMORDIAL_HELIUM                % He/H = 0.079 by number (Y = 0.24) in the chemistry and X_H = 0.76 for the gas (as net 1); cosmological zooms. Default nets 5/17: He/H 0.1, X_H 0.7065  [excl ABHE in Config]
CO_SHIELDING                             % CO self-shielding
WSS_CIE_COOL                             % WSS09 metal-line CIE cooling for hot gas
#GALSF_SGCHEM_VARIABLE_XH_AND_ABHE     % per-particle X_H and abhe from element abundances
#GALSF_SGCHEM_DEBUG_INPUT              % DEBUG: validate species vs pools on DVODE entry
#CR_SCALE_WITH_G0                        % CR ionisation rate scales with G0 (production: constant rate)

########## Metals / dust ##########
METALS                                   % metal tracking
GALSF_RESOLVEDISM_METALS_INDIVIDUAL      % 27 elements H-Zn in Metallicity[]
#GALSF_RESOLVEDISM_METALS_ISOTOPES       % NOT IMPLEMENTED (#error): 40 isotopes instead of 27 elements
TURB_DIFF_METALS                         % turbulent metal diffusion
GALSF_RESOLVEDISM_DUST                   % 5-species dust model  [req FB, METALS_INDIVIDUAL]
SGCHEM_DGR_REMY_RUYER14                  % no live dust: DGR follows Remy-Ruyer+14 (Table 1, X_CO,Z) below 0.26 Zsun, linear in Z above; chemistry (dust cooling, H2 on grains, photoelectric, A_V), TreeCol dust column, radiation pressure  [excl DUST, DGR_INDEPENDENT_OF_METALLICITY]
#SGCHEM_DGR_INDEPENDENT_OF_METALLICITY % no live dust: DGR = DGRnormalized (x MW) at every Z; with DGRnormalized 0.01 = gizmo2017. Default scales with local Z
#GALSF_RESOLVEDISM_DUST_SHOCK_DESTRUCTION % shock-tracked SN dust destruction
#GALSF_RESOLVEDISM_DUST_NO_MCKEE         % disable McKee SN dust destruction
#EXPERIMENTAL_DUST_HU_MODEL              % PHYSICS: Hu+19/23/26 dust (MRN sputtering with v_rel, drag, growth, MW anchor 4.74e-3, seed from C/Si); replaces the default dust physics  [req DUST]
#GALSF_RESOLVEDISM_DUST_SIZE_BINS=8      % MRN log size bins with own v_rel (Hu+19); default 1 (a_eff=0.035 um, Hu+26)  [req EXPERIMENTAL_DUST_HU_MODEL]
#GALSF_RESOLVEDISM_DUST_NO_BETATRON       % drop betatron acceleration of dust v_rel (Hu+26 unmagnetized)  [req EXPERIMENTAL_DUST_HU_MODEL]
#GALSF_RESOLVEDISM_DUST_SN_SWEPT_MASS_DESTRUCTION % McKee89 swept-mass SN dust destruction (coarse cells only; double counts resolved sputtering)  [req EXPERIMENTAL_DUST_HU_MODEL]
#GALSF_RESOLVEDISM_DUST_SN_EJECTA_CONDENSATION % inject Choban+22 SN/Ia condensation dust into the hot deposit (off: Hu+23 has no SN dust)  [req EXPERIMENTAL_DUST_HU_MODEL]
#GALSF_RESOLVEDISM_DUST_SELFTEST         % TEST-ONLY: dust physics self-test dump

########## Cosmic rays ##########
#COSMIC_RAY_FLUID                        % CR fluid (alternative to constant rate)
#CRFLUID_DIFFUSION_MODEL=0               % constant CR diffusivity
#CRFLUID_SPEEDOFLIGHT_REDUCTION=(1.e8/C_LIGHT_CGS) % 1000 km/s CR transport speed (default c); transport along B, no isotropic fallback at B = 0 (Stella setups carry a field)
#INPUT_READ_COSMICRAYENERGY              % keep IC CR energy on fresh starts
#SGCHEM_KEEP_IC_ABUNDANCES             % keep IC SGChemTracAbund on fresh starts

########## Gravity / N-body / KETJU ##########
ADAPTIVE_GRAVSOFT_FORALL=1+16            % adaptive softening for gas + stars
TIDAL_TIMESTEP_CRITERION                 % tidal-tensor timestep criterion
#ADAPTIVE_TREEFORCE_UPDATE=0.0625        % stale tree forces — incompatible with the tree columns and fluxes
KETJU_REGULARIZATION                     % MSTAR chain regularisation for stellar encounters  [excl FORCE_EQUAL_TIMESTEPS]
KETJU_MERGE_STARS                        % stellar collisions (Type 4+4)  [req KETJU_REGULARIZATION]
#KETJU_MERGE_BH                          % BH mergers  [req KETJU_REGULARIZATION]
#KETJU_LEGACY_C_LIBRARY                  % legacy: MSTAR from the C library (reference); default: C++/Kokkos library mstar::mstar (KOKKOS_ROOT)  [req KETJU_REGULARIZATION]
#STARS_INTEGRATED_BY_RUMI                % EXPERIMENTAL: all Type-4 stars integrated by the RUMI++ library (BRIDGE: gas kicks at star steps, star-star pairs out of the tree; make RUMI_LIB_DIR=<prefix>)  [excl KETJU_REGULARIZATION]
#STARS_HERMITE_DIRECT_GRAVITY            % EXPERIMENTAL prototype (A-lite): Type-4 stars on 4th-order Hermite with direct star-star gravity, Aarseth steps (STARS_HERMITE_AARSETH_ETA=0.02)  [excl sinks, KETJU-free Hermite, adaptive Type-4 softening]
#KETJU_PN_REMNANT_TAG                    % tag remnant type at SN; PN on Type-4 NS/BH remnants
#GALSF_RESOLVEDISM_BH_PROMOTION          % promote massive remnants to sinks  [req STELLAR_TABLES] [auto SINK_PARTICLES]

########## MHD ##########
MAGNETIC                                 % ideal MHD
MHD_B_SET_IN_PARAMS                      % initial B from params
#MHD_CONSERVE_ENERGY_ON_KERNEL_VOLUME_CHANGE  % EXPERIMENTAL PHYSICS: charge u with the actual cell E_B change (unused by decision 2026-10-23)
#MHD_RESCALE_B_TO_BOOKED_MAGNETIC_ENERGY  % EXPERIMENTAL PHYSICS: rescale |VB| to the booked E_B (rejected: breaks flux freezing)
#DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH    % EXPERIMENTAL PHYSICS: constrained psi/c_h cleaning (Tricco & Price 2012) replaces Dedner  [req MFM]
#DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH_SYMMETRIC_DIVB  % EXPERIMENTAL: conjugate pair with symmetric div.B  [req CONSERVATIVE_PSI_OVER_CH]
#DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH_NO_DAMPING_HEAT  % EXPERIMENTAL: psi damping loss not returned as heat  [req CONSERVATIVE_PSI_OVER_CH]
#DIVBCLEANING_CONSERVATIVE_SIGMA=1.0      % EXPERIMENTAL: psi damping strength (default 1.0 in 3D, 0.3 in 2D)  [req CONSERVATIVE_PSI_OVER_CH]
#TEST_MHD_ENERGY_LEDGER_OUTPUT            % TEST-ONLY: per-cell MHD energy ledger to mhd_energy_ledger.txt
#TEST_MHD_NO_DEDNER_CLEANING              % TEST-ONLY: no div-B cleaning
#TEST_MHD_NO_POWELL_SOURCE_TERMS          % TEST-ONLY: no Powell source terms
#TEST_DIVBCLEANING_ALTERNATING_CLEANING_SPEED  % TEST-ONLY: c_h doubles every other 0.05 time units  [req CONSERVATIVE_PSI_OVER_CH]

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

########## Experimental (default OFF; unvalidated development, see the merge commits) ##########
#EXPERIMENTAL_SNAPSHOT_RESTART_TEMP_FROM_CHEMISTRY  % RestartFlag 2: set CellP.Temp from chemistry after density()
#EXPERIMENTAL_G0_STELLAR_LUM_CODE_UNITS  % PHYSICS: tree node stellar_lum[] in code units (pixel fluxes convert back to erg/s)
#EXPERIMENTAL_RADIATION_DISJOINT_BANDS  % PHYSICS: radp/caps on disjoint bands (drop LW inside UV, NUV pixel flux 3.4-6 eV)
#GALSF_RESOLVEDISM_HYBRID_DEFAULTS=(1000)  % EXPERIMENTAL prototype: Stella x FIRE-3 coupled mode, FIRE SSPs from cells above M_SSP [Msun], Stella stars below (docs/stella_hybrid/DESIGN.md)
#GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL  % EXPERIMENTAL: per-star owner tag StarModel, each model's physics acts on its own stars  [auto by HYBRID_DEFAULTS]
#GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION=(64)  % EXPERIMENTAL: SF criteria bits of Stella-channel cells (2 and/or 64)  [req PER_PARTICLE_STAR_MODEL]
#GALSF_RESOLVEDISM_TREE_BANDS_PIXEL_FLUX_ONLY  % EXPERIMENTAL, NOT IMPLEMENTED (P4): required by the coupled mode with Stella tree bands
#GALSF_RESOLVEDISM_HYBRID_SINKS_BELOW=(M)  % EXPERIMENTAL, NOT IMPLEMENTED (#error): STARFORGE sink tier below M [Msun]
#GALSF_RESOLVEDISM_HYBRID_CARRY_ALL_ELEMENTS  % EXPERIMENTAL, NOT IMPLEMENTED (#error): carry all 27 Stella elements in the coupled mode

########## Test-only ##########
#GALSF_RESOLVEDISM_ISOLATED_FB_TEST      % TEST-ONLY: isolated-star FB test hooks
#TEST_DUMP_STATE_AROUND_WIND_DEPOSIT     % TEST-ONLY: append raw double state of all particles before/after each wind batch to output/wind_dep_dump.<task>.bin
#STELLA_SELFCHECK                        % TEST-ONLY: runtime invariant checks (harness)
#TEST_GRAVITY_STATIC_PLUMMER             % TEST-ONLY: fixed Plummer potential at the origin (TEST_GRAVITY_STATIC_PLUMMER_MSUN=1000, _PC=0.5)
