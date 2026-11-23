########################################
# ketju_expand_binary -- copy of test/ketju_snapshot_midstep/Config.sh with the energy step trace
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
OUTPUT_ENERGY_STEP_TRACE                 % energy step trace (gizmo.conservation_trace)
