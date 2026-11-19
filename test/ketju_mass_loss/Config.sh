########################################
# ketju_snapshot_midstep -- an isolated binary whose region steps span many orbits, written at times inside them.
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
TEST_KETJU_HALVE_STAR1_MASS_AFTER_GRAVITY=1.0e-5   % test: halve star 1's mass after the gravity call at the end of the first region step
