########################################
# ketju_mass_loss -- copy of test/ketju_snapshot_midstep/Config.sh with the mass-loss test hook
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
