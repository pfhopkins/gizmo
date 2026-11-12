# Stella cosmological zoom of a small dwarf galaxy
# ComovingIntegrationOn 1 in the parameter file; the UV background (TREECOOL, copied from data/cooling if the run
# directory has none) reaches the chemistry only where SGChem reads the UV background; the tree is built on the
# periodic box with the PM grid and a nested PM grid around the zoom region.
STELLA_DEFAULTS
SGCHEM
CHEMISTRYNETWORK=17                 # zoom: primordial H/He + D/HD + CO
SGCHEM_PRIMORDIAL_HELIUM            # zoom: He/H = 0.079 (Y = 0.24), X_H = 0.76
SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS  # zoom: H- photodetachment from the stellar band fields
GALSF_SINGLE_STARS_POPIII           # zoom: Pop III channel in gas below PopIII_Zcrit_solar (the Remy-Ruyer+14 dust-to-gas ratio comes with STELLA_DEFAULTS)
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
BOX_PERIODIC
PMGRID=256
PM_PLACEHIGHRESREGION=1+2
INPUT_POSITIONS_IN_DOUBLE              # positions of a large box need more than float precision
OUTPUT_TEMPERATURE
OUTPUT_GRADIENT_VEL
OUTPUT_TIMESTEP
OUTPUT_COOLRATE
OUTPUT_INDIVIDUAL_COOLRATES
OUTPUT_SGCHEM_RADIATION_FIELDS
OUTPUT_MASS_METAL_ENERGY_LEDGER
# split and merge thresholds are 2.01 x the largest and 0.49 x the smallest gas mass in the IC (init.cc); a zoom IC
# carries gas of several masses, so set them from the target cell mass instead (code units, 1e10 Msun/h per unit):
#MERGESPLIT_HARDCODE_MAX_MASS=(2.01*1.0e-10)
#MERGESPLIT_HARDCODE_MIN_MASS=(0.49*1.0e-10)
