########################################
# ketju_pm -- copy of test/ketju_tide/Config.sh with PMGRID, BOX_PERIODIC and the energy step trace
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
PMGRID=128
BOX_PERIODIC
OUTPUT_ENERGY_STEP_TRACE                 % energy step trace (gizmo.conservation_trace)
