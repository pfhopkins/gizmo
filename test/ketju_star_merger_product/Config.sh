########################################
# ketju_star_merger_product -- table stars colliding inside a KETJU region: contact radius from the tables, merger product with
# the state of a rejuvenated main-sequence star.
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
METALS
GALSF
GALSF_FB_MECHANICAL
GALSF_SINGLE_STARS_FROM_IMF
GALSF_SINGLE_STARS_EVOLUTION_TABLES
OUTPUT_STAR_ENERGY_DIRECT_SUM
STOP_WHEN_BELOW_MINTIMESTEP
