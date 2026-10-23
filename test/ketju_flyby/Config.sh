########################################
# ketju_flyby -- a 1 Msun star flying at 30 km/s past (1500 AU) and through (600 AU) a bound 200 AU pair inside a
# KETJU region: region step caps and the fly-by pair step against an accurate three-body integration.
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
