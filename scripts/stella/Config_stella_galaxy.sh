# Stella isolated galaxy (default model with the galaxy run parameters of stella_galaxy.params)
# no box flags: open boundaries, tree gravity, the halo inside the box. Cosmic-ray rate fixed from the parameter file
# (SGChemCosmicRayIonRate); for the CR fluid or the rate scaled with the stars' field see the options at the end.
STELLA_DEFAULTS
SGCHEM
CHEMISTRYNETWORK=5
METALS
GALSF_RESOLVEDISM_TREECOL_PIXELS=12
GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
HYDRO_MESHLESS_FINITE_MASS
MAGNETIC
MHD_B_SET_IN_PARAMS
TURB_DIFF_METALS
GALSF_SINGLE_STARS_27_ELEMENT_YIELDS
GALSF_FB_PHOTOIONIZATION_PER_DIRECTION
GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
ADAPTIVE_GRAVSOFT_FORALL=1+16
EVALPOTENTIAL
OUTPUT_TEMPERATURE
OUTPUT_GRADIENT_VEL
OUTPUT_TIMESTEP
OUTPUT_COOLRATE
OUTPUT_INDIVIDUAL_COOLRATES
OUTPUT_SGCHEM_RADIATION_FIELDS
OUTPUT_MASS_METAL_ENERGY_LEDGER
# options, each off by default
#CR_SCALE_WITH_G0                       # cosmic-ray rate scaled with the stars' field
#COSMIC_RAY_FLUID                       # CR fluid: also CRFLUID_DIFFUSION_MODEL=7 and the CosmicRay* parameters
#CRFLUID_DIFFUSION_MODEL=7
#KETJU_REGULARIZATION                   # needs ADAPTIVE_GRAVSOFT_FORALL=1 (stars at fixed softening)
#KETJU_MERGE_STARS
