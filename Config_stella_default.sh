# Stella default configuration: net 5 on the C++ chemistry backend, no dust model, subgrid photoionization,
# subgrid FUV (tree G0, HEALPix columns RT_USE_TREECOL_FOR_NH=12), no M1/TreeRay transport.
#!/bin/bash
# Config.sh for resolved-ISM single-star galaxy formation (gizmo-cpp)
# Mapped from gizmo2017 configuration

## -- Hydro solver --
HYDRO_MESHLESS_FINITE_MASS

## -- Star formation & resolved ISM --
GALSF_RESOLVEDISM
GALSF_RESOLVEDISM_FB
GALSF_RESOLVEDISM_PHOTOION       % subgrid photoionization (Stromgren)
GALSF_RESOLVEDISM_SAMPLE_IMF
N_STELLAR_MASS=5
#GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF    % disabled — using pure Jeans criterion (Hu 2023)
GALSF_RESOLVEDISM_INSTANT_SN
GALSF_RESOLVEDISM_G0_VARIABLE
#GALSF_RESOLVEDISM_NUV_VARIABLE
#GALSF_RESOLVEDISM_OPT_VARIABLE
GALSF_RESOLVEDISM_STELLAR_TABLES
GALSF_RESOLVEDISM_WINDS
GALSF_RESOLVEDISM_WINDS_CONTINUOUS   % ADOPTED 2026-07-05: per-step Mdot dt injection, exact int(Mdot v)dt, one batched walk/step (six-way matrix verdict)
GALSF_RESOLVEDISM_WINDS_THERMAL      % ADOPTED 2026-07-05: inelastic-residual heating; with CONTINUOUS -> energy-driven bubbles at 1 Msun res (p_r x3.7 vs momentum-only)
#GALSF_RESOLVEDISM_FB_HEALPIX  % HEALPix FB weighting (candidate default; enable deliberately)
#RESOLVEDISM_HOTSYNC_CLAMP     % hot-region timebin clamp (validated with spawn; production adoption undecided)
GALSF_RESOLVEDISM_METALS_INDIVIDUAL
GALSF_RESOLVEDISM_RADPRESSURE
GALSF_RESOLVEDISM_TYPE_IA
#GALSF_RESOLVEDISM_DUST          % off: dust-to-gas ratio fixed from metallicity
GALSF_SFR_CRITERION=(64)       % Jeans mass only (Hu & Sternberg 2023): M_J < N_ngb * m_gas

## -- Cooling & chemistry --
COOLING
COOLING_OPERATOR_SPLIT
CHEMCOOL
CHEMISTRYNETWORK=5
CO_SHIELDING
WSS_CIE_COOL
#GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE     % per-particle X_H and abhe driven from ElementAbundance[H/He]
#GALSF_CHEMCOOL_DEBUG_INPUT      % validate species-vs-pool invariants on DVODE entry, print full context on violation (debug runs)
CHEMCOOL_CXX                     % C++/CVODE chemistry backend (default for Stella)
#CR_SCALE_WITH_G0                % disabled for testing — using fixed CosmicRayIonRate from params
TURB_DIFF_METALS

## -- Tree column densities --
RT_USE_TREECOL_FOR_NH=12         % HEALPix NSIDE=1; gas, dust-weighted, H2, CO columns
NEED_HEALPIX

## -- Metals & MHD --
METALS
MAGNETIC
MHD_B_SET_IN_PARAMS

## -- M1 Radiation Transport (alternative to subgrid PHOTOION/G0_VARIABLE/RADPRESSURE) --
## When enabling, comment out: GALSF_RESOLVEDISM_PHOTOION, GALSF_RESOLVEDISM_G0_VARIABLE, GALSF_RESOLVEDISM_RADPRESSURE
#RT_M1                           % M1 moment method (triggers RT_EVOLVE_FLUX + RADTRANSFER)
#RT_CHEM_PHOTOION=2              % 4 ionizing bins: H0, He0, He1, He2
#RT_PHOTOION_MULTIFREQUENCY
#RT_PHOTOELECTRIC                % 8-13.6 eV (dust PE heating)
#RT_LYMAN_WERNER                 % 11.2-13.6 eV (H2 photodissociation)
#RT_INFRARED                     % dust-reprocessed IR
#RT_SPEEDOFLIGHT_REDUCTION=0.1   % reduced speed of light (10% of c; 0.003 too slow, 0.1 standard)
#RT_RAD_PRESSURE_FORCES          % radiation pressure on gas

## -- Cosmic ray fluid (alternative to CR_SCALE_WITH_G0) --
## When enabling, comment out: CR_SCALE_WITH_G0
#COSMIC_RAY_FLUID
#CRFLUID_DIFFUSION_MODEL=0       % constant diffusivity (set CosmicRayDiffusionCoeff in params)
#INPUT_READ_COSMICRAYENERGY      % keep CosmicRayEnergy from the IC on fresh starts (CR transport tests; default zeroes it)
#CHEMCOOL_KEEP_IC_ABUNDANCES     % keep ChemcoolTracAbund from the IC on fresh starts (molecular cloud ICs; default zeroes it)

## -- Turbulent metal diffusion --
#TURB_DIFF_METALS

## -- SMBH (gravitational capture) --
#SINK_PARTICLES
#SINK_GRAVCAPTURE_GAS
#SINK_CALC_DISTANCES

## -- Stellar-mass BH promotion (dead stars -> accreting Type 5 sinks) --
#GALSF_RESOLVEDISM_BH_PROMOTION    % auto-enables SINK_PARTICLES + SINK_GRAVCAPTURE_GAS

## -- Box geometry & PM grid (for zoom-ins, molecular clouds, or TALLBOX runs) --
#BOX_PERIODIC
#PMGRID=256
#SHIFT_BY_HALF_BOX
#SHIFTBACK_ON_OUTPUT
#TALLBOX

## -- Gravity & density --
ADAPTIVE_GRAVSOFT_FORALL=1+16    % adaptive softening for gas (Type 0) + stars (Type 4)
TIDAL_TIMESTEP_CRITERION         % tidal tensor timestep (more accurate for few-body, Grudic+2021)
#ADAPTIVE_TREEFORCE_UPDATE=0.0625 % disabled — stales the tree columns and radiation fields
DO_DENSITY_AROUND_NONGAS_PARTICLES

## -- KETJU regularized integrator for close encounters --
#KETJU_REGULARIZATION              % MSTAR chain regularization for massive stars/BHs
#KETJU_MERGE_STARS                 % enable stellar collision mergers (Type 4+4) in KETJU
#KETJU_MERGE_BH                    % enable BH mergers (Type 5+5, Type 5+4) in KETJU
#KETJU_PN_REMNANT_TAG              % stamp StellarRemnantType at SN time + dispatch PN on Type 4 NS/BH remnants (needs GALSF_RESOLVEDISM_FB; no sink/BH_PROMOTION required)

## -- Misc --
NO_ISEND_IRECV_IN_DOMAIN
MULTIPLEDOMAINS=16
EVALPOTENTIAL
OUTPUT_GRADIENT_VEL
OUTPUT_TIMESTEP
OUTPUT_TEMPERATURE
OUTPUT_COOLRATE
OUTPUT_INDIVIDUAL_COOLRATES
