########################################
# ketju_tide -- a 200 AU pair in a KETJU region tidally perturbed by a 20 Msun star at 5000 AU; region steps
# span several pair orbits, so the external field reaches the pair as kicks at random orbital phases.
########################################
ADAPTIVE_GRAVSOFT_FORALL=1
TIDAL_TIMESTEP_CRITERION
EVALPOTENTIAL
NO_ISEND_IRECV_IN_DOMAIN
MULTIPLEDOMAINS=16
OUTPUT_TIMESTEP
OUTPUT_IN_DOUBLEPRECISION
KETJU_REGULARIZATION
KETJU_MERGE_STARS
OUTPUT_STAR_ENERGY_DIRECT_SUM
