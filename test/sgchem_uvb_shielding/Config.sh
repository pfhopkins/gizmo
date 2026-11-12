# Zoom configuration (net 17, TreeCol columns, UV background from TREECOOL) without the periodic box and PM grid: one-zone UVB shielding driver.
# The test appends SGCHEM_UVB_TREECOL_SHIELDING for the shielded build and builds once without it for the optically thin reference.
HYDRO_MESHLESS_FINITE_MASS
GALSF_RESOLVEDISM
GALSF_RESOLVEDISM_FB
GALSF_RESOLVEDISM_PHOTOION       % subgrid photoionization (Stromgren)
GALSF_RESOLVEDISM_SAMPLE_IMF
N_STELLAR_MASS=5
GALSF_RESOLVEDISM_INSTANT_SN
GALSF_RESOLVEDISM_G0_VARIABLE
GALSF_RESOLVEDISM_STELLAR_TABLES
GALSF_RESOLVEDISM_WINDS
GALSF_RESOLVEDISM_WINDS_CONTINUOUS   % ADOPTED 2026-07-05: per-step Mdot dt injection, exact int(Mdot v)dt, one batched walk/step (six-way matrix verdict)
GALSF_RESOLVEDISM_WINDS_THERMAL      % ADOPTED 2026-07-05: inelastic-residual heating; with CONTINUOUS -> energy-driven bubbles at 1 Msun res (p_r x3.7 vs momentum-only)
GALSF_RESOLVEDISM_METALS_INDIVIDUAL
GALSF_RESOLVEDISM_RADPRESSURE
GALSF_RESOLVEDISM_TYPE_IA
GALSF_SFR_CRITERION=(64)       % Jeans mass only (Hu & Sternberg 2023): M_J < N_ngb * m_gas
COOLING
COOLING_OPERATOR_SPLIT
SGCHEM
CHEMISTRYNETWORK=17               % zoom: primordial H/He + D/HD + CO (Uli 2026-11-10)
CO_SHIELDING
WSS_CIE_COOL
TURB_DIFF_METALS
RT_USE_TREECOL_FOR_NH=12         % HEALPix NSIDE=1; gas, dust-weighted, H2, CO columns
NEED_HEALPIX
METALS
MAGNETIC
MHD_B_SET_IN_PARAMS
ADAPTIVE_GRAVSOFT_FORALL=1+16    % adaptive softening for gas (Type 0) + stars (Type 4)
TIDAL_TIMESTEP_CRITERION         % tidal tensor timestep (more accurate for few-body, Grudic+2021)
DO_DENSITY_AROUND_NONGAS_PARTICLES
NO_ISEND_IRECV_IN_DOMAIN
MULTIPLEDOMAINS=16
EVALPOTENTIAL
OUTPUT_GRADIENT_VEL
OUTPUT_TIMESTEP
OUTPUT_TEMPERATURE
OUTPUT_COOLRATE
OUTPUT_INDIVIDUAL_COOLRATES
CR_SCALE_WITH_G0                 % zoom: zeta = clamp(G0/1.7 x CosmicRayIonRate, 1e-21, 2e-16); 1e-21 in pristine gas (Uli 2026-11-10)
STOP_WHEN_BELOW_MINTIMESTEP
